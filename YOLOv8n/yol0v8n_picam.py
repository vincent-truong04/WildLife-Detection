from picamera2 import Picamera2
import cv2, time
from ultralytics import YOLO

CONF = 0.35      # detection confidence
IMGSZ = 416      # inference image size

# Load model (first run will download yolov8n.pt)
model = YOLO("yolov8n.pt")
names = model.names  # class id -> name

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}))
picam2.start()
time.sleep(0.3)

print("Press q to quit")
prev_t = time.time()
fps = 0.0

try:
    while True:
        # Grab RGB frame then convert to BGR for OpenCV + YOLO
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Inference (on CPU)
        results = model.predict(
            source=frame,           # numpy array (BGR)
            imgsz=IMGSZ,
            conf=CONF,
            verbose=False,
            device="cpu"
        )

        # Draw detections on the frame
        # results is a list; we passed 1 image so use results[0]
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                # box.xyxy: (x1,y1,x2,y2), box.conf: confidence, box.cls: class id
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = f"{names[cls_id]} {conf:.2f}"

                # Draw rectangle + label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (int(x1), int(y1)-th-8), (int(x1)+tw+6, int(y1)), (0, 255, 0), -1)
                cv2.putText(frame, label, (int(x1)+3, int(y1)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # FPS overlay
        now = time.time()
        dt = now - prev_t
        prev_t = now
        fps = 0.9*fps + 0.1*(1.0/dt) if dt > 0 else fps
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Show window
        cv2.imshow("Pi Camera + YOLOv8n", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()