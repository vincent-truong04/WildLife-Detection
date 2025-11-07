from picamera2 import Picamera2
from libcamera import ColorSpace
import cv2, time

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}))
picam2.start()

##picam2.set_controls({"AwbEnable": True, "AeEnable": True})
time.sleep(0.3)

print("Press q to quit")
while True:
    frame = picam2.capture_array()
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Pi Camera",bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
picam2.stop()
cv2.destroyAllWindows