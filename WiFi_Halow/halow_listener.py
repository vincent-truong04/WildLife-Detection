#!/usr/bin/env python3
import socket
import struct
import os
import threading
from datetime import datetime
import cv2
import numpy as np
from hailo_utils import HailoYOLO
import firebase_admin
from firebase_admin import credentials, storage

# ── Configuration ─────────────────────────────────────────────────────────────
HOST       = "0.0.0.0"
PORT       = 8080
MAX_SIZE   = 10 * 1024 * 1024  # 10MB to handle XGA JPEGs
OUTPUT_DIR = "/home/pi/Public/WildLife-Detection/Firebase"
UPLOAD_TO_FIREBASE = True

# ── Firebase ──────────────────────────────────────────────────────────────────
def initialize_firebase():
    if not UPLOAD_TO_FIREBASE:
        return None
    cred = credentials.Certificate("/home/pi/Public/WildLife-Detection/Firebase/real-time-wildlife-detector-firebase-adminsdk-fbsvc-7c1cbed963.json")
    firebase_admin.initialize_app(cred, {
        "storageBucket": "real-time-wildlife-detector.firebasestorage.app"
    })
    bucket = storage.bucket()
    print("Firebase initialized!")
    return bucket

# ── YOLO ──────────────────────────────────────────────────────────────────────
def initialize_yolo_model(model_path='yolov8n.hef'):
    print(f"Loading Hailo model from {model_path}...")
    model = HailoYOLO(model_path, labels_path="coco.txt")
    print("Hailo AI HAT+ model loaded!")
    return model

def run_detection(model, frame, frame_num):
    """Run YOLO on a single frame. Returns (results, has_detection)."""
    results = model(frame, conf=0.50)
    detections = results[0].boxes

    if len(detections) > 0:
        print(f"  Frame {frame_num}: {len(detections)} object(s) found")
        for box in detections:
            label = model.names[int(box.cls[0])]
            conf  = float(box.conf[0])
            print(f"    - {label}: {conf:.2f}")
        return results, True
    else:
        print(f"  Frame {frame_num}: nothing detected")
        return None, False

def get_labels(results, model):
    """Extract unique sorted label names from results."""
    labels = [model.names[int(b.cls[0])] for b in results[0].boxes]
    return sorted(set(labels))

def save_annotated(results, frame_num, session_dir, model):
    """Save frame with bounding boxes. Filename includes detected labels."""
    labels   = get_labels(results, model)
    label_str = "_".join(labels)[:100]
    filename  = f"frame_{frame_num}_{label_str}.jpg"
    path      = os.path.join(session_dir, filename)
    cv2.imwrite(path, results[0].plot())
    return path

# ── Firebase upload ───────────────────────────────────────────────────────────
def upload_to_firebase(bucket, detections, timestamp):
    if not UPLOAD_TO_FIREBASE or bucket is None or not detections:
        if not detections:
            print("⚠ No detections — nothing uploaded")
        return

    print(f"Uploading {len(detections)} frame(s) to Firebase...")
    for _, path in detections:
        filename = os.path.basename(path)
        blob = bucket.blob(f"detections/{timestamp}/{filename}")
        blob.upload_from_filename(path)
        print(f"  Uploaded {filename}")
    print("✓ Upload complete")

# ── Core handler — called once per received image ────────────────────────────
def handle_image(img_data, cam_id, model, bucket):
    """
    Receives raw JPEG bytes from the network layer, runs YOLO detection,
    saves annotated results, and uploads to Firebase.
    This replaces the camera capture step from the original pipeline —
    everything from detection onward is unchanged.
    """
    timestamp   = datetime.now().strftime("%m_%d_%Y_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, f"cam{cam_id}_motion_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"[Cam {cam_id}] Image received at {timestamp}")
    print(f"Saving to: {session_dir}")

    # Decode JPEG bytes into an OpenCV BGR frame
    np_arr = np.frombuffer(img_data, dtype=np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        print(f"[Cam {cam_id}] Failed to decode JPEG — skipping")
        return

    # Save original before running detection
    original_path = os.path.join(session_dir, "frame_1_original.jpg")
    cv2.imwrite(original_path, frame)

    # Run detection — only one frame per trigger since the ESP32 sends one image
    results, has_detection = run_detection(model, frame, frame_num=1)

    frames_with_detections = []
    if has_detection:
        path = save_annotated(results, frame_num=1, session_dir=session_dir, model=model)
        frames_with_detections.append((1, path))

    upload_to_firebase(bucket, frames_with_detections, timestamp)

    if frames_with_detections:
        print(f"✓ Session complete: detection saved")
    else:
        print(f"⚠ Session complete: no objects detected")
    print(f"{'='*50}\n")

# ── Network receiver ──────────────────────────────────────────────────────────
def receive_image(conn, addr, model, bucket):
    # Read camera ID
    raw_id_len = conn.recv(1)
    if not raw_id_len:
        print(f"[{addr[0]}] No ID header")
        return
    cam_id = conn.recv(raw_id_len[0]).decode('ascii')

    # Read image length
    raw_len = conn.recv(4)
    if len(raw_len) < 4:
        print(f"[Cam {cam_id}] Incomplete length header")
        return
    img_len = struct.unpack('<I', raw_len)[0]

    if not (0 < img_len <= MAX_SIZE):
        print(f"[Cam {cam_id}] Rejected: claimed {img_len} bytes")
        return

    print(f"[Cam {cam_id}] Receiving {img_len:,} bytes")

    # Receive image data in chunks
    img_data = bytearray()
    while len(img_data) < img_len:
        chunk = conn.recv(min(4096, img_len - len(img_data)))
        if not chunk:
            break
        img_data += chunk

    if len(img_data) != img_len:
        print(f"[Cam {cam_id}] Incomplete: {len(img_data)}/{img_len} bytes")
        return

    # Hand off to detection pipeline
    handle_image(bytes(img_data), cam_id, model, bucket)

def handle_connection(conn, addr, model, bucket):
    with conn:
        try:
            receive_image(conn, addr, model, bucket)
        except socket.timeout:
            print(f"[{addr[0]}] Timeout")
        except Exception as e:
            print(f"[{addr[0]}] Error: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    bucket = initialize_firebase()
    model  = initialize_yolo_model('yolov8n.hef')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(2)
        print(f"Listening on {HOST}:{PORT}")
        print(f"Saving to: {os.path.abspath(OUTPUT_DIR)}")

        while True:
            conn, addr = s.accept()
            conn.settimeout(15.0)
            threading.Thread(
                target=handle_connection,
                args=(conn, addr, model, bucket),
                daemon=True
            ).start()

if __name__ == "__main__":
    main()