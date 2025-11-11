import cv2, time
from ultralytics import YOLO

# --- YOLO settings ---
CONF = 0.35
IMGSZ = 416

# Load pretrained YOLOv8n model
model = YOLO("yolov8n.pt")
names = model.names

# --- Initialize webcam ---
cap = cv2.VideoCapture(0)  # 0 = default webcam; use 1 or 2 for external cameras
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
time.sleep(0.3)

print("Press q to quit")
prev_t = time.time()
fps = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Inference on frame
        results = model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONF,
            verbose=False,
            device="cpu"
        )

        # Draw detections
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{names[cls_id]} {conf:.2f}"

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - th - 8), (int(x1) + tw + 6, int(y1)), (0, 255, 0), -1)
                cv2.putText(frame, label, (int(x1) + 3, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # FPS display
        now = time.time()
        dt = now - prev_t
        prev_t = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Laptop Camera + YOLOv8n", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()