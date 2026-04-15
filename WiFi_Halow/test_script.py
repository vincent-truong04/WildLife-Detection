# test_send.py
import socket
import struct

HOST = "127.0.0.1"
PORT = 8080
IMAGE_PATH = "/home/pi/Public/WildLife-Detection/WiFi_Halow/cat_on_couch.jpg"
CAM_ID = "test"
BURST_COUNT = 3
BURST_INTERVAL_S = 1.5

import time

with open(IMAGE_PATH, "rb") as f:
    img_data = f.read()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    # Handshake
    cam_id_bytes = CAM_ID.encode("ascii")
    s.sendall(bytes([len(cam_id_bytes)]))
    s.sendall(cam_id_bytes)

    ack = s.recv(1)
    print(f"Handshake ACK: {ack.hex()}")

    # Burst
    for i in range(1, BURST_COUNT + 1):
        s.sendall(struct.pack("<I", len(img_data)))
        s.sendall(img_data)

        ack = s.recv(1)
        print(f"Image {i}/{BURST_COUNT} ACK: {ack.hex()}")

        if i < BURST_COUNT:
            time.sleep(BURST_INTERVAL_S)

print("Done — check listener output and Firebase")