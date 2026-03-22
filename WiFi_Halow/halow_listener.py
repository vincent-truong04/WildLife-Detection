#!/usr/bin/env python3
"""
halow_listener.py — Raspberry Pi TCP receiver for the wildlife detector (v3 - Final)

── Architecture ──────────────────────────────────────────────────────────────

Uses persistent TCP sessions. Each camera connects once, performs a handshake,
and streams all images over the same socket until the link drops. The Pi's
session handler loops indefinitely receiving images one at a time until the
camera disconnects. When the camera reconnects the accept() loop catches it
and spawns a fresh session thread.

The ACK is sent to the camera BEFORE inference so the camera never waits for
Hailo processing before knowing its image was received. Inference is dispatched
to a separate thread so the socket closes immediately after the ACK.

── Wire protocol (must match HalowClient.ino) ────────────────────────────────

  Session handshake (once per TCP connection):
    Client → Server : [1 byte]        id_len
    Client → Server : [id_len bytes]  cam_id (ASCII, e.g. "A")
    Server → Client : [1 byte]        0xAC (accepted) | 0x00 (rejected)

  Per-image frame (repeated until connection closes):
    Client → Server : [4 bytes LE]    image_length (uint32)
                                      0x00000000 = heartbeat ping
    Client → Server : [img_len bytes] JPEG payload (omitted for heartbeat)
    Server → Client : [1 byte]        0xAC (received OK) | 0x00 (error)
"""

import socket
import struct
import os
import shutil
import threading
import traceback
from datetime import datetime

import cv2
import numpy as np
from hailo_utils import HailoYOLO
import firebase_admin
from firebase_admin import credentials, storage


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════
HOST              = "0.0.0.0"
PORT              = 8080
MAX_IMAGE_BYTES   = 10 * 1024 * 1024   # 10 MB — rejects corrupt length headers

# Per-recv() timeout. Covers individual recv calls, not the whole session.
# A camera between PIR triggers will be idle on the socket — that is expected
# and does not time out here because recv_exact() is only called when data
# has been announced. Timeout only fires if the radio link dies mid-transfer.
RECV_CHUNK_TIMEOUT_S = 30

LISTEN_BACKLOG    = 5

OUTPUT_DIR        = "/home/pi/Public/WildLife-Detection/Firebase/Images"
DISK_WARN_BYTES   = 500 * 1024 * 1024
UPLOAD_TO_FIREBASE = False

FIREBASE_CERT     = (
    "/home/pi/Public/WildLife-Detection/Firebase/"
    "real-time-wildlife-detector-firebase-adminsdk-fbsvc-7c1cbed963.json"
)
FIREBASE_BUCKET   = "real-time-wildlife-detector.firebasestorage.app"

MODEL_PATH        = os.path.join(os.path.dirname(__file__), "yolov8n.hef")
LABELS_PATH       = "/home/pi/Public/WildLife-Detection/YOLOv8n/coco.txt"

# Protocol bytes — must match HalowClient.ino
ACK = bytes([0xAC])
NAK = bytes([0x00])

# Serialises access to the Hailo inference pipeline which is not thread-safe.
# Only the model() call is protected — file I/O happens outside the lock.
model_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  recv_exact()
#  Reads exactly n bytes from conn, looping over partial TCP segments.
#  TCP is a stream protocol — recv(n) may return anywhere from 1 to n bytes.
#  Over HaLow this is the rule not the exception due to variable burst sizes.
#  Raises ConnectionError if the remote end closes before n bytes arrive.
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
#  Firebase helpers
# ══════════════════════════════════════════════════════════════════════════════
def initialize_firebase():
    if not UPLOAD_TO_FIREBASE:
        return None
    cred = credentials.Certificate(FIREBASE_CERT)
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
    bucket = storage.bucket()
    print("[Firebase] Initialised")
    return bucket


def upload_to_firebase(bucket, detections: list, timestamp: str):
    if not UPLOAD_TO_FIREBASE or bucket is None or not detections:
        if not detections:
            print("  No detections — nothing uploaded to Firebase")
        return
    print(f"  [Firebase] Uploading {len(detections)} frame(s)…")
    for _, path in detections:
        filename = os.path.basename(path)
        blob = bucket.blob(f"detections/{timestamp}/{filename}")
        blob.upload_from_filename(path)
        print(f"    Uploaded: {filename}")
    print("  [Firebase] Upload complete ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  Disk space guard
# ══════════════════════════════════════════════════════════════════════════════
def check_disk_space(path: str) -> bool:
    try:
        free = shutil.disk_usage(path).free
        if free < DISK_WARN_BYTES:
            print(f"  ⚠ LOW DISK SPACE: only {free // (1024 * 1024)} MB free on {path}")
            return False
    except Exception as e:
        print(f"  ⚠ Could not check disk space on {path}: {e}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  YOLO helpers
# ══════════════════════════════════════════════════════════════════════════════
def initialize_yolo_model(model_path: str, labels_path: str) -> HailoYOLO:
    print(f"[YOLO] Loading model from {model_path}…")
    model = HailoYOLO(model_path, labels_path=labels_path)
    print("[YOLO] Hailo 8L model ready")
    return model


def run_detection(model: HailoYOLO, frame, frame_num: int):
    results    = model(frame, conf=0.50)
    detections = results[0].boxes
    if detections:
        print(f"  Frame {frame_num}: {len(detections)} object(s) detected")
        for box in detections:
            label = model.names[int(box.cls[0])]
            conf  = float(box.conf[0])
            print(f"    · {label}: {conf:.2f}")
        return results, True
    print(f"  Frame {frame_num}: nothing detected")
    return None, False


def save_annotated(results, frame_num: int, session_dir: str,
                   model: HailoYOLO) -> str:
    labels    = sorted({model.names[int(b.cls[0])] for b in results[0].boxes})
    label_str = "_".join(labels)[:100]
    filename  = f"frame_{frame_num}_{label_str}.jpg"
    path      = os.path.join(session_dir, filename)
    cv2.imwrite(path, results[0].plot())
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  handle_image()
#  Called from a dedicated thread after ACK has been sent to the camera.
#  Any exception here does NOT affect the TCP session.
# ══════════════════════════════════════════════════════════════════════════════
def handle_image(img_data: bytes, cam_id: str, img_num: int,
                 model: HailoYOLO, bucket):
    timestamp   = datetime.now().strftime("%m_%d_%Y_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, f"cam{cam_id}_motion_{timestamp}")

    print(f"\n{'='*55}")
    print(f"[Cam {cam_id}] Image #{img_num}  {len(img_data):,} bytes  @ {timestamp}")

    if not check_disk_space(OUTPUT_DIR):
        print(f"[Cam {cam_id}] Skipping save — insufficient disk space")
        print(f"{'='*55}\n")
        return

    os.makedirs(session_dir, exist_ok=True)
    print(f"Saving to: {session_dir}")

    np_arr = np.frombuffer(img_data, dtype=np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[Cam {cam_id}] Failed to decode JPEG — skipping")
        print(f"{'='*55}\n")
        return

    original_path = os.path.join(session_dir, "frame_1_original.jpg")
    cv2.imwrite(original_path, frame)

    # model_lock held only for the Hailo call — file I/O happens outside
    with model_lock:
        results, has_detection = run_detection(model, frame, frame_num=1)

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
#  receive_session()
#  Manages one complete camera session: handshake then image-receive loop.
#
#  Session lifecycle:
#    1. Read the camera identity from the handshake.
#    2. Send ACK to confirm Pi is ready.
#    3. Loop:
#         a. Read 4-byte frame header.
#         b. If length is 0 — heartbeat ping, send ACK pong, continue.
#         c. Validate the length.
#         d. Read the full JPEG payload.
#         e. Send ACK immediately before inference — camera must not wait.
#         f. Dispatch handle_image() to its own thread so this loop can
#            immediately receive the next image without waiting for
#            inference and disk I/O to complete.
#    4. When ConnectionError is raised (camera rebooted, link dropped etc.)
#       print the disconnection and return. The accept() loop catches the
#       next connect() and spawns a fresh thread.
# ══════════════════════════════════════════════════════════════════════════════

class SessionError(Exception):
    """Raised when the session must close due to a protocol violation."""


def receive_session(conn: socket.socket, addr: tuple,
                    model: HailoYOLO, bucket):
    remote      = addr[0]
    image_count = 0
    cam_id      = "?"

    try:
        # ── Handshake ──────────────────────────────────────────────────────────
        id_len = recv_exact(conn, 1)[0]
        if id_len == 0:
            print(f"[{remote}] Handshake: zero-length camera ID — rejecting")
            conn.sendall(NAK)
            return

        cam_id = recv_exact(conn, id_len).decode("ascii")
        print(f"[{remote}] Camera '{cam_id}' connected")
        conn.sendall(ACK)

        # ── Image receive loop ─────────────────────────────────────────────────
        while True:
            raw_len = recv_exact(conn, 4)
            img_len = struct.unpack("<I", raw_len)[0]

            # Zero-length frame = heartbeat ping — respond and continue
            if img_len == 0:
                conn.sendall(ACK)
                print(f"[Cam {cam_id}] Heartbeat ✓")
                continue

            # Non-zero but out of range = stream misalignment
            if img_len > MAX_IMAGE_BYTES:
                print(f"[Cam {cam_id}] Invalid image length {img_len} (max {MAX_IMAGE_BYTES}) — closing session")
                conn.sendall(NAK)
                raise SessionError(f"Invalid image length {img_len}")

            print(f"[Cam {cam_id}] Receiving image #{image_count + 1}  ({img_len:,} bytes) from {remote}")

            img_data = recv_exact(conn, img_len)

            # ACK immediately — before inference.
            # Hailo inference can take 1-3 s. If ACK waited until after
            # handle_image() the camera would time out and retry an image
            # that was already received correctly.
            conn.sendall(ACK)

            image_count += 1

            # Dispatch processing to its own thread so this loop can
            # immediately receive the next image without waiting for
            # inference and disk I/O to complete.
            threading.Thread(
                target=_safe_handle_image,
                args=(img_data, cam_id, image_count, model, bucket),
                daemon=True,
            ).start()

    except ConnectionError as e:
        print(f"[Cam {cam_id} @ {remote}] Disconnected after {image_count} image(s): {e}")

    except SessionError as e:
        print(f"[{remote}] Session terminated: {e}")

    except socket.timeout:
        print(f"[{remote}] Session timed out after {RECV_CHUNK_TIMEOUT_S} s of inactivity")

    except Exception as e:
        print(f"[{remote}] Unexpected error in session (after {image_count} images): {e}")
        traceback.print_exc()

    finally:
        print(f"[{remote}] Session closed  (received {image_count} image(s))")


def _safe_handle_image(img_data, cam_id, img_num, model, bucket):
    """Wrapper so exceptions in handle_image() are printed not silently lost."""
    try:
        handle_image(img_data, cam_id, img_num, model, bucket)
    except Exception as e:
        print(f"[Cam {cam_id}] handle_image() raised an exception: {e}")
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  handle_connection()
# ══════════════════════════════════════════════════════════════════════════════
def handle_connection(conn: socket.socket, addr: tuple,
                      model: HailoYOLO, bucket):
    with conn:
        # SO_KEEPALIVE lets OS detect cameras that vanish silently
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        # Tune keepalive to detect dead connections within ~60s
        # rather than the default 2+ hours
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE,  60)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT,    5)
        except (AttributeError, OSError):
            pass  # not available on all platforms

        # Increase socket buffers for HaLow link headroom
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)

        conn.settimeout(RECV_CHUNK_TIMEOUT_S)
        receive_session(conn, addr, model, bucket)


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
        print(f"[Server] Chunk timeout: {RECV_CHUNK_TIMEOUT_S} s  Backlog: {LISTEN_BACKLOG}")

        while True:
            conn, addr = srv.accept()
            print(f"[Server] Incoming connection from {addr[0]}:{addr[1]}")
            threading.Thread(
                target=handle_connection,
                args=(conn, addr, model, bucket),
                daemon=True,
            ).start()


if __name__ == "__main__":
    main()