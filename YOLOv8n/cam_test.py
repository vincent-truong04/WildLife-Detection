from ultralytics import YOLO
import cv2

CONF  = 0.35   # confidence threshold
IMGSZ = 416    # smaller = faster on Pi 5 
CAM_ID = 0     # 0 = default camera

# Load pretrained model 
model = YOLO("yolov8n.pt")

# Open camera
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Try CAM_ID=1 or check your camera connection.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed, exiting.")
        break

    # Inference 
    result = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]

    # Draw boxes/labels on the frame
    annotated = result.plot()

    # Show window (press 'q' to quit)
    cv2.imshow("YOLOv8n (press q to quit)", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()