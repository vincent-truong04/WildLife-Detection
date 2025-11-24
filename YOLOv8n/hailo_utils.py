import numpy as np
import cv2
import os
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                            ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)

class HailoYOLO:
    def __init__(self, hef_path, labels_path="coco.txt"):
        print(f"Loading HEF: {hef_path}")
        self.hef = HEF(hef_path)
        self.target = VDevice()
        
        # Configure
        self.configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, self.configure_params)[0]
        self.network_group_params = self.network_group.create_params()
        
        # Stream Setup (UINT8 Input / FLOAT32 Output)
        self.input_vstream_params = InputVStreamParams.make(
            self.network_group, quantized=False, format_type=FormatType.UINT8)
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)
            
        # Dimensions
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.height = self.input_info.shape[0]
        self.width = self.input_info.shape[1]
        
        # Labels
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.names = {i: name.strip() for i, name in enumerate(f.readlines())}
        else:
            self.names = {i: str(i) for i in range(80)}

    def __call__(self, frame, conf=0.45):
        # 1. PREPROCESSING
        resized = cv2.resize(frame, (self.width, self.height))
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Input: UINT8 (No division by 255.0)
        input_data = np.expand_dims(resized_rgb, axis=0)

        # 2. INFERENCE
        with InferVStreams(self.network_group, self.input_vstream_params, self.output_vstream_params) as infer_pipeline:
            with self.network_group.activate(self.network_group_params):
                results = infer_pipeline.infer(input_data)

        # 3. OUTPUT PARSING (The Fix for "Inhomogeneous Shape")
        # Get raw output (usually a list of lists)
        raw_output = list(results.values())[0]
        boxes = []
        
        # Scenario A: Ragged Per-Class Output [Batch][80 Classes][Detections]
        # This matches your error: shape (1, 80)
        if isinstance(raw_output, list) and len(raw_output) > 0 and isinstance(raw_output[0], list):
            # Access the first image in the batch
            batch_0 = raw_output[0] 
            
            # Iterate over all 80 classes
            for class_id, class_detections in enumerate(batch_0):
                # Iterate over detections for this specific class
                for det in class_detections:
                    # det is typically [ymin, xmin, ymax, xmax, score]
                    if len(det) >= 5:
                        score = det[4]
                        if score < conf: continue
                        
                        ymin, xmin, ymax, xmax = det[:4]
                        
                        # Scale to original frame
                        h, w, _ = frame.shape
                        boxes.append(FakeBox(
                            cls=class_id, # We know the class from the loop index
                            conf=score,
                            coords=[xmin*w, ymin*h, xmax*w, ymax*h]
                        ))
                        
        # Scenario B: Flat Output (Numpy Array)
        # Fallback for other model types
        else:
            # Safe convert to numpy
            try:
                raw_arr = np.array(raw_output)
                if len(raw_arr.shape) == 3: raw_arr = raw_arr[0] # Unbatch
                
                for det in raw_arr:
                    if len(det) >= 5:
                        score = det[4]
                        if score < conf: continue
                        ymin, xmin, ymax, xmax = det[:4]
                        cls_id = int(det[5]) if len(det) > 5 else 0
                        
                        h, w, _ = frame.shape
                        boxes.append(FakeBox(
                            cls=cls_id,
                            conf=score,
                            coords=[xmin*w, ymin*h, xmax*w, ymax*h]
                        ))
            except Exception:
                pass # Ignore parsing errors if format is totally unknown

        return [FakeResult(boxes, self.names, frame)]

# Helper Classes
class FakeBox:
    def __init__(self, cls, conf, coords):
        self.cls = [float(cls)]
        self.conf = [float(conf)]
        self.xyxy = [coords]

class FakeResult:
    def __init__(self, boxes, names, frame):
        self.boxes = boxes
        self.names = names
        self.orig_img = frame
    
    def plot(self):
        img = self.orig_img.copy()
        for box in self.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label_name = self.names.get(cls_id, f"{cls_id}")
            label_text = f"{label_name} {box.conf[0]:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img