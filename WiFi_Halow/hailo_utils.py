import numpy as np
import cv2
import os
import logging
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams,
    ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType,
)

log = logging.getLogger(__name__)


class HailoYOLO:

    def __init__(self, hef_path: str,
                 labels_path: str = "/home/pi/Public/WildLife-Detection/YOLOv8n/coco.txt"):
        print(f"[Hailo] Loading HEF: {hef_path}")

        self.hef              = None
        self.target           = None
        self.infer_pipeline   = None
        self.ng_activation    = None


        try:
            self.hef    = HEF(hef_path)
            self.target = VDevice()

            configure_params = ConfigureParams.create_from_hef(
                hef=self.hef, interface=HailoStreamInterface.PCIe
            )
            self.network_group        = self.target.configure(self.hef, configure_params)[0]
            self.network_group_params = self.network_group.create_params()


            self.input_vstream_params = InputVStreamParams.make(
                self.network_group, quantized=False, format_type=FormatType.UINT8
            )
            # FLOAT32 output — de-quantised detection scores and coordinates.
            self.output_vstream_params = OutputVStreamParams.make(
                self.network_group, quantized=False, format_type=FormatType.FLOAT32
            )

            # Input tensor dimensions (height, width) from the compiled model.
            input_info   = self.hef.get_input_vstream_infos()[0]
            self.height  = input_info.shape[0]
            self.width   = input_info.shape[1]

            # Class labels 
            if os.path.exists(labels_path):
                with open(labels_path, "r") as f:
                    self.names = {i: name.strip() for i, name in enumerate(f)}
            else:
                log.warning("[Hailo] Labels file not found: %s — "
                            "using numeric class IDs", labels_path)
                self.names = {i: str(i) for i in range(80)}
            
            self.allowed_classes = {0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

            # Open the inference pipeline once at construction time and reuse
            # it for every call — avoids repeated setup/teardown overhead.
            self.infer_pipeline = InferVStreams(
                self.network_group,
                self.input_vstream_params,
                self.output_vstream_params,
            )
            self.infer_pipeline.__enter__()

            self.ng_activation = self.network_group.activate(self.network_group_params)
            self.ng_activation.__enter__()

            print(f"[Hailo] Model ready  "
                  f"({self.width}×{self.height}  "
                  f"{len(self.names)} classes)")

        except Exception:
            log.exception("[Hailo] Initialisation failed — releasing resources")
            self._release()
            raise   # re-raise so the caller knows construction failed

    # ══════════════════════════════════════════════════════════════════════════
    #  __call__()
    #  Run inference on a single BGR frame
    # ══════════════════════════════════════════════════════════════════════════
    def __call__(self, frame, conf: float = 0.45):
        # ── Pre-processing ─────────────────────────────────────────────────────
        fh, fw = frame.shape[:2]
        scale  = min(self.width / fw, self.height / fh)
        nw, nh = int(fw * scale), int(fh * scale)

        resized = cv2.resize(frame, (nw, nh))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Pad to the exact model input size
        canvas       = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        pad_top      = (self.height - nh) // 2
        pad_left     = (self.width  - nw) // 2
        canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = rgb

        input_data = np.expand_dims(canvas, axis=0)  # → (1, H, W, 3)

        # ── Inference ─────────────────────────────────────────────────────────
        raw_results = self.infer_pipeline.infer(input_data)

        # ── Output parsing ────────────────────────────────────────────────────
        #
        #   Format A — ragged per-class list:
        #     raw[batch][class_id][detection] → [ymin, xmin, ymax, xmax, score]
        #     Shape is (1, 80) — one list per class, variable length per class.
        #
        #   Format B — flat numpy array:
        #     raw[batch][detection] → [ymin, xmin, ymax, xmax, score, class_id]
        #
        raw_output = list(raw_results.values())[0]
        try:
            if isinstance(raw_output, list) and len(raw_output) > 0:
                batch = raw_output[0]
                for cid, dets in enumerate(batch):
                    if cid not in self.allowed_classes:  
                        continue
                    for det in dets:
                        if len(det) >= 5 and det[4] > 0.05:
                            print(f"  [DEBUG] class={cid} ({self.names.get(cid,'?')}) "
                            f"score={det[4]:.3f} coords={det[:4]}")
            else:
                arr = np.array(raw_output)
                if arr.ndim == 3: arr = arr[0]
                for det in arr:
                        if len(det) >= 5 and det[4] > 0.05:
                            print(f"  [DEBUG] score={det[4]:.3f} cls={int(det[5]) if len(det)>5 else '?'}")
        except Exception as e:
            print(f"  [DEBUG] failed: {e}")
        boxes      = []

        h, w, _ = frame.shape   # original frame dimensions for coordinate scaling

        if (isinstance(raw_output, list)
                and len(raw_output) > 0
                and isinstance(raw_output[0], list)):
            # ── Format A: ragged per-class output ─────────────────────────────
            batch_0 = raw_output[0]
            for class_id, class_detections in enumerate(batch_0):
                if class_id not in self.allowed_classes:
                    continue
                for det in class_detections:
                    if len(det) < 5:
                        continue
                    score = det[4]
                    if score < conf:
                        continue
                    ymin, xmin, ymax, xmax = det[:4]
                    x1 = (xmin * self.width  - pad_left) / scale
                    y1 = (ymin * self.height - pad_top)  / scale
                    x2 = (xmax * self.width  - pad_left) / scale
                    y2 = (ymax * self.height - pad_top)  / scale
                    boxes.append(FakeBox(
                        cls    = class_id,
                        conf   = score,
                        coords = [x1, y1, x2, y2],
                    ))
        else:
            # ── Format B: flat numpy array ────────────────────────────────────
            try:
                arr = np.array(raw_output)
                if arr.ndim == 3:
                    arr = arr[0]   # unbatch
                for det in arr:
                    if len(det) < 5:
                        continue
                    score = det[4]
                    if score < conf:
                        continue
                    class_id = int(det[5]) if len(det) > 5 else -1
                    if class_id not in self.allowed_classes:
                        continue
                    ymin, xmin, ymax, xmax = det[:4]
                    x1 = (xmin * self.width  - pad_left) / scale
                    y1 = (ymin * self.height - pad_top)  / scale
                    x2 = (xmax * self.width  - pad_left) / scale
                    y2 = (ymax * self.height - pad_top)  / scale
                    boxes.append(FakeBox(
                        cls    = class_id,
                        conf   = score,
                        coords = [x1, y1, x2, y2],
                    ))
            except Exception:
                log.exception("[Hailo] Failed to parse flat output format")

        return [FakeResult(boxes, self.names, frame)]

    # ══════════════════════════════════════════════════════════════════════════
    #  _release()
    #  Safely tears down the Hailo pipeline in the correct order.
    # ══════════════════════════════════════════════════════════════════════════
    def _release(self):
        for resource, name in [
            (self.ng_activation,  "network group activation"),
            (self.infer_pipeline, "inference pipeline"),
        ]:
            if resource is not None:
                try:
                    resource.__exit__(None, None, None)
                except Exception:
                    log.exception("[Hailo] Error releasing %s", name)

    def __del__(self):
        self._release()

class FakeBox:
    """
    Represents one detected object.  Attributes match the ultralytics
    Boxes interface so call-site code (box.cls[0], box.conf[0], box.xyxy[0])
    is identical regardless of the backend.
    """
    def __init__(self, cls: int, conf: float, coords: list):
        self.cls  = [float(cls)]
        self.conf = [float(conf)]
        self.xyxy = [coords]   # [x1, y1, x2, y2] in original pixel space


class FakeResult:
    """
    Container returned by HailoYOLO.__call__().
    .boxes — list of FakeBox
    .names — {class_id: label} dict
    .plot() — returns a copy of the frame with bounding boxes drawn
    """
    def __init__(self, boxes: list, names: dict, frame):
        self.boxes    = boxes
        self.names    = names
        self.orig_img = frame

    def plot(self):
        """Draw all detections onto a copy of the original frame."""
        img = self.orig_img.copy()
        for box in self.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id     = int(box.cls[0])
            label_name = self.names.get(cls_id, str(cls_id))
            label_text = f"{label_name} {box.conf[0]:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, label_text, (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )
        return img