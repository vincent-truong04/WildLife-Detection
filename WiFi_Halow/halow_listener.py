import socket
import struct
import os
from datetime import datetime

HOST = "0.0.0.0"
PORT = 8080
SAVE_DIR = "captured_wildlife"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def start_listener():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)
        print(f"Monitoring Halow on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            conn.settimeout(10.0)

            with conn:
                print(f"\nIncoming image from {addr} ...")
                try:
                    raw_len = conn.recv(4)
                    if not raw_len:
                        continue
                
                    img_len = struct.unpack('<I', raw_len)[0]
                    print(f"Expecting {img_len} bytes")

                    img_data = b''
                    while len(img_data) < img_len:
                        packet = conn.recv(4096)
                        if not packet:
                            break
                        img_data += packet
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{SAVE_DIR}/image_{timestamp}.jpg"

                    with open(filename, 'wb') as f:
                        f.write(img_data)

                    print(f"Success: Saved {len(img_data)} bytes to {filename}")

                except Exception as e:
                    print(f"ERROR during transfer: {e}")

if __name__ == "__main__":
    start_listener()