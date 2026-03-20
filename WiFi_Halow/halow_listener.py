#!/usr/bin/env python3
"""
halow_listener.py — Raspberry Pi TCP receiver for the wildlife detector.

Receives JPEG images from ESP32 field cameras over HaLow (802.11ah),
runs YOLOv8n inference on the Hailo 8L accelerator, saves annotated
results, and optionally uploads detections to Firebase Storage.
"""

import socket
import struct
import os
import shutil
import threading
from datetime import datetime
import cv2
import numpy as np
from hailo_utils import HailoYOLO
import firebase_admin
from firebase_admin import credentials, storage

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════
HOST             = "0.0.0.0"
PORT             = 8080
MAX_IMAGE_BYTES  = 10 * 1024 * 1024   # 10 MB ceiling — rejects corrupt headers
RECV_TIMEOUT_S   = 60                 # per-connection socket timeout.
                                       # Original was 15 s — too tight for a
                                       # large XGA JPEG over a variable-latency
                                       # HaLow link.
LISTEN_BACKLOG   = 5                  # max queued connections before accept().
                                       # Original was 2 — a burst from two
                                       # cameras could overflow it, dropping
                                       # the third connection before we ever
                                       # called accept().

OUTPUT_DIR        = "/home/pi/Public/WildLife-Detection/Firebase/Images"
DISK_WARN_BYTES   = 500 * 1024 * 1024  # warn when free space falls below 500 MB
UPLOAD_TO_FIREBASE = False

# FIX: centralise the credential path so it only needs to change in one place.
FIREBASE_CERT     = (
    "/home/pi/Public/WildLife-Detection/Firebase/"
    "real-time-wildlife-detector-firebase-adminsdk-fbsvc-7c1cbed963.json"
)
FIREBASE_BUCKET   = "real-time-wildlife-detector.firebasestorage.app"

MODEL_PATH        = os.path.join(os.path.dirname(__file__), "yolov8n.hef")
LABELS_PATH       = "/home/pi/Public/WildLife-Detection/YOLOv8n/coco.txt"

# Protocol bytes — must match HalowClient.ino
ACK = bytes([0xAC])   # success: full image received and decoded
NAK = bytes([0x00])   # failure: something went wrong, please retry

# Serialises concurrent access to the Hailo inference pipeline.
# The pipeline itself is not thread-safe; the lock prevents two camera
# threads from calling model() simultaneously.
model_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  recv_exact()
#  THE most important fix in this file.
#
#  TCP is a stream protocol. conn.recv(n) is legally allowed to return
#  anywhere from 1 to n bytes — the OS decides how much to deliver based on
#  segment boundaries, buffer state, and MTU. Over HaLow with its variable
#  latency this is especially common.
#
#  The original code called conn.recv(4) for the image-length header and
#  assumed it would always return exactly 4 bytes. When it didn't (e.g. only
#  2 bytes arrived), struct.unpack('<I', ...) unpacked garbage as the length,
#  the subsequent recv() tried to read several gigabytes, failed, and the
#  connection was silently dropped — with no indication of why.
#
#  recv_exact() loops until exactly n bytes have accumulated, or raises
#  ConnectionError if the socket closes early.
# ══════════════════════════════════════════════════════════════════════════════
def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"Connection closed after {len(buf)}/{n} bytes"
            )
        buf += chunk
    return bytes(buf)


# ══════════════════════════════════════════════════════════════════════════════
#  Firebase
# ══════════════════════════════════════════════════════════════════════════════
def initialize_firebase():
    if not UPLOAD_TO_FIREBASE:
        return None
    cred = credentials.Certificate(FIREBASE_CERT)
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
    bucket = storage.bucket()
    print("[Firebase] Initialized")
    return bucket


def upload_to_firebase(bucket, detections: list, timestamp: str):
    if not UPLOAD_TO_FIREBASE or bucket is None or not detections:
        if not detections:
            print("  ⚠ No detections — nothing uploaded to Firebase")
        return
    print(f"  [Firebase] Uploading {len(detections)} frame(s)…")
    for _, path in detections:
        filename = os.path.basename(path)
        blob = bucket.blob(f"detections/{timestamp}/{filename}")
        blob.upload_from_filename(path)
        print(f"    Uploaded: {filename}")
    print("  [Firebase] Upload complete ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  YOLO helpers
# ══════════════════════════════════════════════════════════════════════════════
def initialize_yolo_model(model_path: str, labels_path: str) -> HailoYOLO:
    print(f"[YOLO] Loading model from {model_path}…")
    model = HailoYOLO(model_path, labels_path=labels_path)
    print("[YOLO] Hailo AI HAT+ model ready")
    return model


def run_detection(model: HailoYOLO, frame, frame_num: int):
    """Run inference. Returns (results, has_detection)."""
    results    = model(frame, conf=0.50)
    detections = results[0].boxes
    if len(detections) > 0:
        print(f"  Frame {frame_num}: {len(detections)} object(s) detected")
        for box in detections:
            label = model.names[int(box.cls[0])]
            conf  = float(box.conf[0])
            print(f"    · {label}: {conf:.2f}")
        return results, True
    print(f"  Frame {frame_num}: nothing detected")
    return None, False


def get_labels(results, model: HailoYOLO) -> list:
    return sorted({model.names[int(b.cls[0])] for b in results[0].boxes})


def save_annotated(results, frame_num: int, session_dir: str,
                   model: HailoYOLO) -> str:
    """
    Draw bounding boxes on the frame and save to disk.
    NOTE: this function does NOT need the model_lock — it only calls
    results[0].plot() (OpenCV drawing) and cv2.imwrite(). The original
    code held model_lock across this call, unnecessarily blocking other
    camera threads from running inference while a JPEG was being written.
    """
    labels    = get_labels(results, model)
    label_str = "_".join(labels)[:100]
    filename  = f"frame_{frame_num}_{label_str}.jpg"
    path      = os.path.join(session_dir, filename)
    cv2.imwrite(path, results[0].plot())
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Disk space guard
#  FIX: the original code saved images unconditionally. An unattended Pi in
#  the field will eventually fill its SD card, after which cv2.imwrite()
#  silently returns False and images are lost with no indication of why.
#  This function logs a warning when free space is low and returns False
#  to let the caller skip the save.
# ══════════════════════════════════════════════════════════════════════════════
def check_disk_space(path: str) -> bool:
    try:
        free = shutil.disk_usage(path).free
        if free < DISK_WARN_BYTES:
            print(f"  ⚠ LOW DISK SPACE: only {free // (1024*1024)} MB free "
                  f"on {path}")
            return False
    except Exception as e:
        print(f"  ⚠ Could not check disk space: {e}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  handle_image()
#  Called once per received image. Decodes the JPEG, runs YOLO, saves
#  annotated results, and uploads to Firebase.
# ══════════════════════════════════════════════════════════════════════════════
def handle_image(img_data: bytes, cam_id: str, model: HailoYOLO, bucket):
    timestamp   = datetime.now().strftime("%m_%d_%Y_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, f"cam{cam_id}_motion_{timestamp}")

    print(f"\n{'='*55}")
    print(f"[Cam {cam_id}] {len(img_data):,} bytes  @  {timestamp}")

    # ── Disk space check ───────────────────────────────────────────────────────
    if not check_disk_space(OUTPUT_DIR):
        print(f"[Cam {cam_id}] Skipping save — insufficient disk space")
        print(f"{'='*55}\n")
        return

    os.makedirs(session_dir, exist_ok=True)
    print(f"Saving to: {session_dir}")

    # ── Decode JPEG ────────────────────────────────────────────────────────────
    np_arr = np.frombuffer(img_data, dtype=np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[Cam {cam_id}] Failed to decode JPEG — skipping")
        print(f"{'='*55}\n")
        return

    # Save the unmodified original before any annotation
    original_path = os.path.join(session_dir, "frame_1_original.jpg")
    cv2.imwrite(original_path, frame)

    # ── Inference (lock held only during the Hailo call) ──────────────────────
    # FIX: model_lock now wraps ONLY the inference call.
    # The original code also held the lock across save_annotated(), which
    # calls cv2.imwrite() — a pure disk write with no involvement from the
    # Hailo pipeline. Holding the lock there unnecessarily serialised all
    # camera sessions for the duration of a file write.
    with model_lock:
        results, has_detection = run_detection(model, frame, frame_num=1)

    # ── Annotate and save (lock NOT held here) ─────────────────────────────────
    frames_with_detections = []
    if has_detection:
        path = save_annotated(results, frame_num=1,
                              session_dir=session_dir, model=model)
        frames_with_detections.append((1, path))

    upload_to_firebase(bucket, frames_with_detections, timestamp)

    if frames_with_detections:
        print("✓ Detection saved")
    else:
        print("⚠ No objects detected — original saved only")
    print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  receive_image()
#  Reads one complete framed message from the socket and dispatches it to
#  handle_image().  Sends ACK on success, NAK on any error so the ESP32
#  knows whether to retry.
#
#  Wire format (must match HalowClient.ino):
#    1 byte        — byte-length of camera ID string
#    id_len bytes  — camera ID (ASCII, e.g. "A")
#    4 bytes LE    — image payload length (uint32, little-endian)
#    img_len bytes — raw JPEG payload
#    → 0xAC (ACK) or 0x00 (NAK) sent back
#
#  All reads go through recv_exact() so partial TCP segments can never
#  corrupt the length header or misalign the image stream.
# ══════════════════════════════════════════════════════════════════════════════
def receive_image(conn: socket.socket, addr, model: HailoYOLO, bucket):
    try:
        # Camera ID
        id_len = recv_exact(conn, 1)[0]
        cam_id = recv_exact(conn, id_len).decode("ascii")

        # Image length
        img_len = struct.unpack("<I", recv_exact(conn, 4))[0]
        if not (0 < img_len <= MAX_IMAGE_BYTES):
            print(f"[Cam {cam_id}] Rejected: claimed {img_len} bytes — "
                  f"outside valid range (0, {MAX_IMAGE_BYTES}]")
            conn.sendall(NAK)
            return

        print(f"[Cam {cam_id}] Receiving {img_len:,} bytes from {addr[0]}")

        # Full image payload — may arrive across many TCP segments over HaLow
        img_data = recv_exact(conn, img_len)

        # FIX: send ACK *before* running inference.
        # Inference on the Hailo can take several seconds. If we waited until
        # after handle_image() to send the ACK, the ESP32 would time out
        # waiting for a response and retry an image that was already received
        # correctly. The ACK confirms receipt, not processing completion.
        conn.sendall(ACK)

        handle_image(img_data, cam_id, model, bucket)

    except ConnectionError as e:
        print(f"[{addr[0]}] Connection error: {e}")
        try:
            conn.sendall(NAK)
        except Exception:
            pass
    except Exception as e:
        print(f"[{addr[0]}] Unexpected error: {e}")
        try:
            conn.sendall(NAK)
        except Exception:
            pass


def handle_connection(conn: socket.socket, addr, model: HailoYOLO, bucket):
    with conn:
        # SO_KEEPALIVE lets the OS detect silent drops from field devices
        # rather than leaving a thread hanging on recv_exact() indefinitely.
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # Timeout covers the entire receive phase (header + payload).
        # 60 s is enough for a ~2 MB XGA JPEG over a slow HaLow link.
        conn.settimeout(RECV_TIMEOUT_S)
        try:
            receive_image(conn, addr, model, bucket)
        except socket.timeout:
            print(f"[{addr[0]}] Timed out after {RECV_TIMEOUT_S} s")


# ══════════════════════════════════════════════════════════════════════════════
#  main()
# ══════════════════════════════════════════════════════════════════════════════
def main():
    bucket = initialize_firebase()
    model  = initialize_yolo_model(MODEL_PATH, LABELS_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(LISTEN_BACKLOG)
        print(f"[Server] Listening on {HOST}:{PORT}")
        print(f"[Server] Output: {os.path.abspath(OUTPUT_DIR)}")
        print(f"[Server] Timeout: {RECV_TIMEOUT_S} s  "
              f"Backlog: {LISTEN_BACKLOG}")

        while True:
            conn, addr = srv.accept()
            threading.Thread(
                target=handle_connection,
                args=(conn, addr, model, bucket),
                daemon=True
            ).start()


if __name__ == "__main__":
    main()