#!/usr/bin/env python3
import lgpio
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from datetime import datetime
from hailo_utils import HailoYOLO
import os
import firebase_admin
from firebase_admin import credentials, storage

# ========== Configuration ==========
GPIO_PIN = 17
NUM_FRAMES = 3
FRAME_DELAY = 0.2  # 200ms between frames
OUTPUT_DIR = "/home/pi/Public/WildLife-Detection/Firebase"
UPLOAD_TO_FIREBASE = True  # Set to False to disable Firebase uploads

# ========== Initialization Functions =========

def initialize_firebase():
    """Initialize Firebase connection."""
    if not UPLOAD_TO_FIREBASE:
        return None
    
    cred = credentials.Certificate("/home/pi/Public/WildLife-Detection/Firebase/real-time-wildlife-detector-firebase-adminsdk-fbsvc-7c1cbed963.json")
    firebase_admin.initialize_app(cred, {
        "storageBucket": "real-time-wildlife-detector.firebasestorage.app"
    })
    bucket = storage.bucket()
    print("Firebase initialized!")
    return bucket

def initialize_gpio():
    """Initialize GPIO for reading trigger signal."""
    h = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_input(h, GPIO_PIN)
    print(f"GPIO {GPIO_PIN} initialized for input")
    return h

def initialize_camera():
    """Initialize and start the Pi camera."""
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration()
    picam2.configure(camera_config)
    picam2.start()
    print("Camera initialized!")
    return picam2

def initialize_yolo_model(model_path='yolov8n.hef'): # Changed to .hef
    """Load Hailo YOLO model."""
    print(f"Loading Hailo model from {model_path}...")
    # We ignore the old Ultralytics logic and use our wrapper
    model = HailoYOLO(model_path, labels_path="coco.txt")
    print("Hailo AI HAT+ model loaded!")
    return model

# ========== Frame Capture Functions ==========

def capture_frames(picam2, num_frames, frame_delay, session_dir):
    """Capture multiple frames from the camera."""
    print(f"Capturing {num_frames} frames...")
    frames = []
    
    for i in range(num_frames):
        print(f"Capturing frame {i+1}/{num_frames}...")
        frame = picam2.capture_array()
        
        # Convert from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(frame_bgr)
        
        # Save original frame
        original_path = os.path.join(session_dir, f"frame_{i+1}_original.jpg")
        cv2.imwrite(original_path, frame_bgr)
        
        if i < num_frames - 1:  # Don't delay after last frame
            time.sleep(frame_delay)
    
    return frames

# ========== Detection Functions ==========

def run_detection_on_frame(model, frame, frame_num):
    """Run YOLO detection on a single frame and return results."""
    print(f"Processing frame {frame_num}...")
    
    # Run inference
    results = model(frame, conf=0.50)  # 50% confidence threshold
    detections = results[0].boxes
    
    # Print detected objects
    if len(detections) > 0:
        print(f"  Frame {frame_num}: Found {len(detections)} objects")
        for box in detections:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            print(f"    - {label}: {conf:.2f}")
        return results, True
    else:
        print(f"  Frame {frame_num}: No objects detected - skipping save")
        return None, False

def get_detection_labels(results, model):
    """Extract detected object labels from YOLO results."""
    detections = results[0].boxes
    labels = []
    
    for box in detections:
        cls = int(box.cls[0])
        label = model.names[cls]
        labels.append(label)
    
    # Return unique labels, sorted alphabetically
    return sorted(set(labels))

def create_filename_with_detections(frame_num, labels):
    """Create filename with frame number and detected objects."""
    if not labels:
        return f"frame_{frame_num}_no_detection.jpg"
    
    # Join labels with underscore, limit length for filesystem compatibility
    labels_str = "_".join(labels)
    # Limit filename length (some filesystems have limits)
    if len(labels_str) > 100:
        labels_str = labels_str[:100]
    
    return f"frame_{frame_num}_{labels_str}.jpg"

def save_detected_frame(results, frame_num, session_dir, model):
    """Save frame with bounding boxes drawn, filename includes detected objects."""
    annotated_frame = results[0].plot()
    
    # Get detected labels and create filename
    labels = get_detection_labels(results, model)
    filename = create_filename_with_detections(frame_num, labels)
    
    detected_path = os.path.join(session_dir, filename)
    cv2.imwrite(detected_path, annotated_frame)
    return detected_path

def process_frames_with_detection(model, frames, session_dir):
    """Process all frames with YOLO detection and save only frames with detections."""
    print(f"\nRunning YOLO detection on {len(frames)} frames...")
    frames_with_detections = []
    
    for i, frame in enumerate(frames):
        frame_num = i + 1
        results, has_detection = run_detection_on_frame(model, frame, frame_num)
        
        if has_detection:
            # Pass model to save_detected_frame
            detected_path = save_detected_frame(results, frame_num, session_dir, model)
            frames_with_detections.append((frame_num, detected_path))
    
    return frames_with_detections

# ========== Firebase Functions ==========

def upload_frames_to_firebase(bucket, frames_with_detections, timestamp):
    """Upload detected frames to Firebase Storage."""
    if not UPLOAD_TO_FIREBASE or bucket is None:
        return
    
    if len(frames_with_detections) == 0:
        print(f"\n⚠ No objects detected in any frames - nothing uploaded")
        return
    
    print(f"\nUploading {len(frames_with_detections)} frames with detections to Firebase...")
    
    for frame_num, detected_path in frames_with_detections:
        # Extract just the filename from the full path
        filename = os.path.basename(detected_path)
        
        # Use the same filename in Firebase
        firebase_detected_path = f"detections/{timestamp}/{filename}"
        blob_detected = bucket.blob(firebase_detected_path)
        blob_detected.upload_from_filename(detected_path)
        print(f"  Uploaded {filename}")
    
    print(f"✓ All frames with detections uploaded to Firebase Storage!")

# ========== Main Detection Handler ==========

def handle_motion_detection(picam2, model, bucket):
    """Handle a single motion detection event - capture, detect, save, upload."""
    timestamp = datetime.now().strftime("%m_%d_%Y_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, f"motion_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Motion detected at {timestamp}!")
    print(f"Saving to: {session_dir}")
    
    # Capture frames
    frames = capture_frames(picam2, NUM_FRAMES, FRAME_DELAY, session_dir)
    
    # Run detection and save frames with detections
    frames_with_detections = process_frames_with_detection(model, frames, session_dir)
    
    # Upload to Firebase
    upload_frames_to_firebase(bucket, frames_with_detections, timestamp)
    
    # Summary
    if len(frames_with_detections) > 0:
        print(f"\n✓ Session complete: {len(frames_with_detections)}/{NUM_FRAMES} frames had detections")
    else:
        print(f"\n⚠ Session complete: No objects detected in any frames")
    
    print(f"{'='*50}\n")

# ========== GPIO Monitoring ==========

def wait_for_gpio_low(gpio_handle, pin):
    """Wait for GPIO pin to go LOW before continuing."""
    while lgpio.gpio_read(gpio_handle, pin) == 1:
        time.sleep(0.1)

def monitor_gpio_for_triggers(gpio_handle, picam2, model, bucket):
    """Main loop - monitor GPIO and trigger detection on HIGH signal."""
    print(f"Waiting for trigger on GPIO {GPIO_PIN}...")
    
    try:
        while True:
            # Check if GPIO is HIGH
            if lgpio.gpio_read(gpio_handle, GPIO_PIN) == 1:
                handle_motion_detection(picam2, model, bucket)
                
                # Wait for GPIO to go LOW before watching for next trigger
                wait_for_gpio_low(gpio_handle, GPIO_PIN)
                
                # Small delay to avoid immediate re-trigger
                time.sleep(1)
            
            time.sleep(0.1)  # Check every 100ms
    
    except KeyboardInterrupt:
        print("\n\nExiting...")

# ========== Cleanup ==========

def cleanup(gpio_handle, picam2):
    """Clean up resources before exit."""
    picam2.stop()
    lgpio.gpiochip_close(gpio_handle)
    print("Cleanup complete!")

# ========== Main Program ==========

def main():
    """Main program entry point."""
    # Initialize all components
    bucket = initialize_firebase()
    gpio_handle = initialize_gpio()
    picam2 = initialize_camera()
    model = initialize_yolo_model('yolov8n.hef')  # Change model path here if needed
    
    # Start monitoring for triggers
    try:
        monitor_gpio_for_triggers(gpio_handle, picam2, model, bucket)
    finally:
        cleanup(gpio_handle, picam2)

if __name__ == "__main__":
    main()