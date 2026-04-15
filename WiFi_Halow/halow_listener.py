import socket
import struct
import os
import shutil
import threading
import traceback
import time
from datetime import datetime

import anthropic
import base64

import cv2
import numpy as np
from hailo_utils import HailoYOLO
import firebase_admin
from firebase_admin import credentials, storage

#Config
HOST              = "0.0.0.0"
PORT              = 8080
MAX_IMAGE_BYTES   = 10 * 1024 * 1024   
RECV_CHUNK_TIMEOUT_S = 300
LISTEN_BACKLOG    = 5

OUTPUT_DIR        = "/home/pi/Public/WildLife-Detection/Firebase/Images"
DISK_WARN_BYTES   = 500 * 1024 * 1024
UPLOAD_TO_FIREBASE = True

CLAUDE_MODEL = "claude-sonnet-4-6"
ANTHROPIC_API_KEY = "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"

FIREBASE_CERT     = (
    "/home/pi/Public/WildLife-Detection/Firebase/"
    "real-time-wildlife-detector-firebase-adminsdk-fbsvc-7c1cbed963.json"
)
FIREBASE_BUCKET   = "real-time-wildlife-detector.firebasestorage.app"

MODEL_PATH        = os.path.join(os.path.dirname(__file__), "yolov8n.hef")
LABELS_PATH       = "/home/pi/Public/WildLife-Detection/YOLOv8n/coco.txt"

# Protocol bytes
ACK = bytes([0xAC])
NAK = bytes([0x00])

model_lock = threading.Lock()

# Incremented on every new connection. Threads check this before sending
# so a stale session never writes ACK bytes into a new session's handshake
session_lock    = threading.Lock()
current_session = {"gen": 0}



# Read exactly n bytes, looping over partial TCP segments.
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



#  Firebase helpers
def initialize_firebase():
    if not UPLOAD_TO_FIREBASE:
        return None
    cred = credentials.Certificate(FIREBASE_CERT)
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})
    bucket = storage.bucket()
    print("[Firebase] Initialised")
    return bucket

# Upload annotated frames for this motion event.
def upload_to_firebase(bucket, path: str, folder: str, filename: str, timestamp: str, event_folder: str):
    if not UPLOAD_TO_FIREBASE or bucket is None:
        return
    blob = bucket.blob(f"{folder}/{event_folder}/{filename}")
    blob.upload_from_filename(path)
    print(f"    Uploaded to '{folder}/{event_folder}/': {filename}")
    print("  [Firebase] Upload complete ✓")



#  Disk space check
def check_disk_space(path: str) -> bool:
    try:
        free = shutil.disk_usage(path).free
        if free < DISK_WARN_BYTES:
            print(f"LOW DISK SPACE: only {free // (1024 * 1024)} MB free on {path}")
            return False
    except Exception as e:
        print(f"Could not check disk space on {path}: {e}")
    return True

# Claude API Call
def identify_with_claude(frame) -> str:
    print("  [Claude] Sending frame for species identification…")
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_b64 = base64.b64encode(buffer).decode("utf-8")
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model      = CLAUDE_MODEL,
            max_tokens = 256,
            messages   = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type"       : "base64",
                            "media_type" : "image/jpeg",
                            "data"       : image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are a wildlife identification assistant. "
                            "Examine this image and identify all animals present.\n\n"
                            "Rules:\n"
                            "• Reply in this exact format and nothing else:\n"
                            "  COUNT: <number>\n"
                            "  ANIMALS: <animal1>, <animal2>, ...\n"
                            "• Be as specific as possible (e.g. 'white-tailed deer' not just 'deer').\n"
                            "• If the same species appears multiple times, list it once.\n"
                            "• If no animals are present, reply:\n"
                            "  COUNT: 0\n"
                            "  ANIMALS: none"
                        ),
                    },
                ],
            }],
        )
        raw_response = response.content[0].text.strip().lower()
        print(f"  [Claude] Raw response:\n{raw_response}")
        count   = 0
        animals = []
        for line in raw_response.splitlines():
            if line.startswith("count:"):
                try:
                    count = int(line.split(":")[1].strip())
                except ValueError:
                    count = 0
            elif line.startswith("animals:"):
                raw_animals = line.split(":")[1].strip()
                if raw_animals != "none":
                    animals = [
                        a.strip()
                         .replace(" ", "_")
                         .replace("-", "_")
                         .replace("'", "")
                         .replace(".", "")
                        for a in raw_animals.split(",")
                    ]
        safe_label = "_and_".join(animals) if animals else "unknown_animal"
        print(f"  [Claude] Count  : {count}")
        print(f"  [Claude] Animals: {animals}")
        print(f"  [Claude] Label  : '{safe_label}'")
        return safe_label
    except Exception:
        traceback.print_exc()
        print("  [Claude] Identification failed — falling back to 'unknown_animal'")
        return "unknown_animal"


#  YOLO helpers
def initialize_yolo_model(model_path: str, labels_path: str) -> HailoYOLO:
    print(f"[YOLO] Loading model from {model_path}…")
    model = HailoYOLO(model_path, labels_path=labels_path)
    print("[YOLO] Hailo 8L model ready")
    return model


def run_detection(model: HailoYOLO, frame, frame_num: int):
    results    = model(frame, conf=0.15)
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
                   model: HailoYOLO, claude_label: str = None) -> str:
    if claude_label:
        label_str = claude_label
    else:
        yolo_labels = sorted({model.names[int(b.cls[0])] for b in results[0].boxes})
        label_str   = "_".join(yolo_labels)[:100]
    filename = f"frame_{frame_num}_{label_str}.jpg"
    path     = os.path.join(session_dir, filename)
    cv2.imwrite(path, results[0].plot())
    return path


# Decode, run inference, save, and upload. Runs in its own thread after ACK.
def handle_image(img_data: bytes, cam_id: str, img_num: int,
                 model: HailoYOLO, bucket, session_dir: str, timestamp: str):

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

    original_path = os.path.join(session_dir, f"frame_{img_num}_original.jpg")
    cv2.imwrite(original_path, frame)

    with model_lock:
        results, has_detection = run_detection(model, frame, frame_num=img_num)

    if has_detection:
        claude_label = identify_with_claude(frame)
        path = save_annotated(results, frame_num=img_num,
                              session_dir=session_dir, model=model,
                              claude_label=claude_label)
        filename = os.path.basename(path)
        upload_to_firebase(bucket, path, "detected", filename, timestamp,
                           os.path.basename(session_dir))
        print("Detection saved and uploaded")
    else:
        filename = os.path.basename(original_path)
        upload_to_firebase(bucket, original_path, "empty", filename, timestamp,
                           os.path.basename(session_dir))
        print("No objects detected — original saved and uploaded")

    print(f"{'='*55}\n")

# Wrapper so exceptions in handle_image() are logged
def _safe_handle_image(img_data, cam_id, img_num, model, bucket, session_dir, timestamp):
    try:
        handle_image(img_data, cam_id, img_num, model, bucket, session_dir, timestamp)
    except Exception as e:
        print(f"[Cam {cam_id}] handle_image() raised an exception: {e}")
        traceback.print_exc()


#Session
class SessionError(Exception):
    """Raised when the session must close due to a protocol violation."""


def receive_session(conn, addr, model, bucket):
    remote      = addr[0]
    image_count = 0
    cam_id      = "?"

    EVENT_GAP_S     = 10
    event_dir       = None
    event_timestamp = None
    last_image_time = 0.0

    with session_lock:
        current_session["gen"] += 1
        my_gen = current_session["gen"]

    def is_current():
        with session_lock:
            return current_session["gen"] == my_gen

    try:
        print(f"[{remote}] [sess {my_gen}] Awaiting handshake byte...")
        id_len_byte = recv_exact(conn, 1)
        id_len = id_len_byte[0]
        print(f"[{remote}] [sess {my_gen}] Got id_len = 0x{id_len:02X} ({id_len})")

        if id_len == 0 or id_len > 16:
            print(f"[{remote}] [sess {my_gen}] Bogus id_len {id_len} — sending NAK and closing")
            try: conn.sendall(NAK)
            except Exception: pass
            return

        cam_id_raw = recv_exact(conn, id_len)
        try:
            cam_id = cam_id_raw.decode("ascii")
        except UnicodeDecodeError:
            print(f"[{remote}] [sess {my_gen}] Non-ASCII cam_id bytes: {cam_id_raw.hex()} — NAK")
            try: conn.sendall(NAK)
            except Exception: pass
            return

        print(f"[{remote}] [sess {my_gen}] cam_id = '{cam_id}' (raw: {cam_id_raw.hex()})")

        conn.setblocking(False)
        drained = bytearray()
        try:
            while True:
                chunk = conn.recv(1024)
                if not chunk: break
                drained += chunk
        except (BlockingIOError, OSError):
            pass
        conn.setblocking(True)
        conn.settimeout(RECV_CHUNK_TIMEOUT_S)

        if drained:
            print(f"[{remote}] [sess {my_gen}] WARNING: drained {len(drained)} stale bytes: "
                  f"{drained[:32].hex()}{'...' if len(drained)>32 else ''}")

        print(f"[{remote}] [sess {my_gen}] Sending ACK (0x{ACK[0]:02X}) to camera '{cam_id}'")
        conn.sendall(ACK)
        print(f"[{remote}] [sess {my_gen}] Handshake complete")

        while True:
            if not is_current():
                print(f"[Cam {cam_id}] Session {my_gen} superseded — exiting")
                return

            raw_len = recv_exact(conn, 4)
            img_len = struct.unpack("<I", raw_len)[0]

            if img_len == 0:
                if not is_current():
                    return
                conn.sendall(ACK)
                print(f"[Cam {cam_id}] Heartbeat")
                continue

            if img_len > MAX_IMAGE_BYTES:
                print(f"[Cam {cam_id}] Invalid image length {img_len} "
                      f"(max {MAX_IMAGE_BYTES}) — closing session")
                conn.sendall(NAK)
                raise SessionError(f"Invalid image length {img_len}")

            print(f"[Cam {cam_id}] Receiving image #{image_count + 1} "
                  f" ({img_len:,} bytes) from {remote}")

            img_data = recv_exact(conn, img_len)
            if not is_current():
                return
            time.sleep(0.05)

            if not is_current():
                print(f"[Cam {cam_id}] Session {my_gen} superseded before ACK — exiting")
                return
            conn.sendall(ACK)
            image_count += 1

            now = time.time()
            if event_dir is None or (now - last_image_time) > EVENT_GAP_S:
                event_timestamp = datetime.now().strftime("%m_%d_%Y_%H%M%S")
                event_dir       = os.path.join(OUTPUT_DIR, f"cam{cam_id}_motion_{event_timestamp}")
                os.makedirs(event_dir, exist_ok=True)
                print(f"[Cam {cam_id}] New event folder: {event_dir}")
            last_image_time = now

            threading.Thread(
                target=_safe_handle_image,
                args=(img_data, cam_id, image_count, model, bucket, event_dir, event_timestamp),
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
        print(f"[{remote}] Session {my_gen} closed  (received {image_count} image(s))")



# Handle Connection
def handle_connection(conn: socket.socket, addr: tuple,
                      model: HailoYOLO, bucket):
    with conn:
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # Detect dead cameras within ~60 s instead of the OS default 2+ hours
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE,  120)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT,    3)
        except (AttributeError, OSError) as e:
            print(f"[{addr[0]}] keepalive sockopts not set: {e}")
        # Increase socket buffers for HaLow link headroom
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)
        conn.settimeout(2)
        receive_session(conn, addr, model, bucket)


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
        print(f"[Server] Chunk timeout: {RECV_CHUNK_TIMEOUT_S} s  "
              f"Backlog: {LISTEN_BACKLOG}")

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
