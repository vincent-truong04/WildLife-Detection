# test_send.py
import socket
import struct

HOST = "127.0.0.1"
PORT = 8080
IMAGE_PATH = "/home/pi/Public/WildLife-Detection/WiFi_Halow/cat_on_couch.jpg"
CAM_ID = "test"

with open(IMAGE_PATH, "rb") as f:
    img_data = f.read()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    # Handshake — send camera ID length + ID
    cam_id_bytes = CAM_ID.encode("ascii")
    s.sendall(bytes([len(cam_id_bytes)]))
    s.sendall(cam_id_bytes)

    # Wait for ACK
    ack = s.recv(1)
    print(f"Handshake ACK: {ack.hex()}")

    # Send image length + image data
    s.sendall(struct.pack("<I", len(img_data)))
    s.sendall(img_data)

    # Wait for ACK
    ack = s.recv(1)
    print(f"Image ACK: {ack.hex()}")

print("Done — check listener output and Firebase")