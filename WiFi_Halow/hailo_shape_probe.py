"""
HEF output shape diagnostic.

Run this on your Pi with the same image that Ultralytics detected the
person in at conf=0.936. It prints the exact structure of what your
Hailo model emits, so we can see why the parser is silently dropping
the person detection.

Usage:
    python3 hailo_shape_probe.py /path/to/test_image.jpg
"""

import sys
import os
import numpy as np
import cv2

from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams,
    ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType,
)

# Same HEF and image as your pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8s.hef")


def describe(obj, indent=0, max_depth=4, path="raw_output"):
    """Recursively print the structure of whatever the HEF returned."""
    pad = "  " * indent
    if indent > max_depth:
        print(f"{pad}{path}: (max depth reached)")
        return

    if isinstance(obj, np.ndarray):
        print(f"{pad}{path}: ndarray  shape={obj.shape}  dtype={obj.dtype}")
        if obj.size > 0 and obj.size <= 30:
            print(f"{pad}  values: {obj.flatten().tolist()}")
        elif obj.size > 0:
            flat = obj.flatten()
            print(f"{pad}  first 10: {flat[:10].tolist()}")
            print(f"{pad}  min={flat.min():.4f}  max={flat.max():.4f}  mean={flat.mean():.4f}")
        return

    if isinstance(obj, (list, tuple)):
        type_name = "list" if isinstance(obj, list) else "tuple"
        print(f"{pad}{path}: {type_name} len={len(obj)}")
        if len(obj) > 0:
            # Show first element type
            describe(obj[0], indent + 1, max_depth, f"{path}[0]")
            # If list is long and homogeneous, mention it
            if len(obj) > 1 and type(obj[0]) is type(obj[1]):
                lengths = [len(x) if hasattr(x, "__len__") else "?" for x in obj[:10]]
                print(f"{pad}  element lengths (first 10): {lengths}")
        return

    if isinstance(obj, dict):
        print(f"{pad}{path}: dict keys={list(obj.keys())}")
        for k, v in obj.items():
            describe(v, indent + 1, max_depth, f"{path}[{k!r}]")
        return

    print(f"{pad}{path}: {type(obj).__name__} = {obj!r}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 hailo_shape_probe.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"ERROR: image not found: {img_path}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: HEF not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"[Probe] Loading HEF: {MODEL_PATH}")
    hef    = HEF(MODEL_PATH)
    target = VDevice()

    configure_params = ConfigureParams.create_from_hef(
        hef=hef, interface=HailoStreamInterface.PCIe
    )
    network_group        = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    # Report inputs
    print("\n" + "=" * 60)
    print("INPUT STREAMS:")
    print("=" * 60)
    for info in hef.get_input_vstream_infos():
        print(f"  name:  {info.name}")
        print(f"  shape: {info.shape}")
        print(f"  format: {info.format.type}  order: {info.format.order}")

    # Report outputs — this is the critical part
    print("\n" + "=" * 60)
    print("OUTPUT STREAMS (declared by HEF):")
    print("=" * 60)
    for info in hef.get_output_vstream_infos():
        print(f"  name:  {info.name}")
        print(f"  shape: {info.shape}")
        print(f"  format: {info.format.type}  order: {info.format.order}")

    input_info = hef.get_input_vstream_infos()[0]
    H = input_info.shape[0]
    W = input_info.shape[1]

    input_vstream_params = InputVStreamParams.make(
        network_group, quantized=False, format_type=FormatType.UINT8
    )
    output_vstream_params = OutputVStreamParams.make(
        network_group, quantized=False, format_type=FormatType.FLOAT32
    )

    # Preprocess image (same letterbox as hailo_utils)
    frame = cv2.imread(img_path)
    fh, fw = frame.shape[:2]
    scale  = min(W / fw, H / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    canvas       = np.zeros((H, W, 3), dtype=np.uint8)
    pad_top      = (H - nh) // 2
    pad_left     = (W - nw) // 2
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = rgb
    input_data = np.expand_dims(canvas, axis=0)

    print(f"\n[Probe] Running inference on {img_path}")
    print(f"        Input tensor shape: {input_data.shape}")

    with InferVStreams(network_group, input_vstream_params, output_vstream_params) as pipeline:
        with network_group.activate(network_group_params):
            raw_results = pipeline.infer(input_data)

    print("\n" + "=" * 60)
    print("RAW OUTPUT STRUCTURE:")
    print("=" * 60)
    describe(raw_results, path="raw_results")

    # Pull out the actual tensor/list and probe it for person (class 0) signals
    print("\n" + "=" * 60)
    print("SEARCHING FOR ANY HIGH-CONFIDENCE VALUES:")
    print("=" * 60)

    raw_output = list(raw_results.values())[0]

    # Case 1: ragged per-class (80-long list)
    if isinstance(raw_output, list) and len(raw_output) > 0 and isinstance(raw_output[0], list):
        batch_0 = raw_output[0]
        print(f"  Detected Format A (ragged per-class). {len(batch_0)} class buckets.")
        total = 0
        for cid, dets in enumerate(batch_0):
            if len(dets) > 0:
                total += len(dets)
                label = "person" if cid == 0 else f"class_{cid}"
                print(f"  class {cid:2d} ({label}): {len(dets)} detection(s)")
                for i, det in enumerate(dets[:3]):  # first 3 per class
                    det_arr = np.array(det)
                    print(f"    det[{i}] shape={det_arr.shape}  values={det_arr.tolist()}")
        print(f"  TOTAL raw detections across all classes: {total}")
        if total == 0:
            print("  !!! MODEL RETURNED ZERO DETECTIONS — this is the bug.")
            print("      The HEF itself is failing, not the parser.")

    # Case 2: flat numpy array
    else:
        arr = np.array(raw_output)
        print(f"  Detected Format B (flat array).  shape={arr.shape}")
        if arr.ndim == 3:
            arr = arr[0]
        print(f"  After unbatch: shape={arr.shape}")
        if arr.size > 0:
            # Find rows with any high score
            if arr.ndim == 2 and arr.shape[1] >= 5:
                # Try different score column positions
                for col in [4, 5, -2, -1]:
                    try:
                        col_idx = col if col >= 0 else arr.shape[1] + col
                        top = arr[np.argsort(arr[:, col_idx])[::-1][:5]]
                        print(f"  Top 5 rows by column {col_idx}:")
                        for row in top:
                            print(f"    {row.tolist()}")
                    except Exception as e:
                        pass

    print("\n[Probe] Done.")


if __name__ == "__main__":
    main()