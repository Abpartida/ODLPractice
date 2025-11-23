# !python3 -m pip install depthai

import os
import json
import numpy as np
import cv2
from pathlib import Path
import depthai as dai
import time


def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


# Resolve the blob/json that live under the latest training "result" artifacts
RESULT_DIR = Path(os.environ.get("RESULT_DIR", "resultv5"))


def resolve_model_artifacts(result_dir: Path) -> tuple[str, str]:
    """Pick the YOLO blob + json from the exported result folder."""
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir.resolve()}")

    blob_files = sorted(result_dir.glob("*.blob"))
    if not blob_files:
        raise FileNotFoundError(f"No .blob files found under {result_dir.resolve()}")

    json_files = sorted(result_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found under {result_dir.resolve()}")

    def _pick_preferred(paths):
        for path in paths:
            if "best" in path.stem.lower():
                return path
        return paths[0]

    blob_path = _pick_preferred(blob_files)
    json_path = _pick_preferred(json_files)
    return str(blob_path), str(json_path)


YOLOV8N_MODEL, YOLOV8N_CONFIG = resolve_model_artifacts(RESULT_DIR)
MODEL_CONFIG = load_config(YOLOV8N_CONFIG)

OUTPUT_VIDEO = "vid_result/960-oak-d-live_video.mp4" #Adjust path accordingly

DEFAULT_CAMERA_DIM = (960, 960)


def resolve_input_dimensions(model_config, default_dim):
    size_str = (
        model_config.get("nn_config", {}).get("input_size")
        if isinstance(model_config, dict)
        else None
    )
    if isinstance(size_str, str) and "x" in size_str:
        try:
            width_str, height_str = size_str.lower().split("x")
            return int(width_str), int(height_str)
        except ValueError:
            pass
    return default_dim


CAMERA_PREV_DIM = resolve_input_dimensions(MODEL_CONFIG, DEFAULT_CAMERA_DIM)
LABELS = MODEL_CONFIG.get("mappings", {}).get("labels", ["Detection"])


def create_camera_pipeline(model_config, model_path):
    pipeline = dai.Pipeline()
    nn_config = model_config.get("nn_config", {})
    metadata = nn_config.get("NN_specific_metadata", {})
    classes = int(metadata.get("classes", 0))
    coordinates = int(metadata.get("coordinates", 4))
    anchors = metadata.get("anchors", [])
    anchor_masks = metadata.get("anchor_masks", {})
    iou_threshold = float(metadata.get("iou_threshold", 0.5))
    confidence_threshold = float(metadata.get("confidence_threshold", 0.5))

    input_width, input_height = resolve_input_dimensions(model_config, CAMERA_PREV_DIM)

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(input_width, input_height)
    cam_rgb.setInterleaved(False)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    detection_network = pipeline.create(dai.node.YoloDetectionNetwork)
    detection_network.setConfidenceThreshold(confidence_threshold)
    detection_network.setNumClasses(classes)
    detection_network.setCoordinateSize(coordinates)
    if anchors:
        detection_network.setAnchors(anchors)
    if anchor_masks:
        detection_network.setAnchorMasks(anchor_masks)
    detection_network.setIouThreshold(iou_threshold)
    detection_network.setNumInferenceThreads(2)
    detection_network.setBlobPath(model_path)
    detection_network.input.setBlocking(False)

    rgb_out = pipeline.create(dai.node.XLinkOut)
    rgb_out.setStreamName("rgb")

    nn_out = pipeline.create(dai.node.XLinkOut)
    nn_out.setStreamName("nn")

    cam_rgb.preview.link(detection_network.input)
    detection_network.passthrough.link(rgb_out.input)
    detection_network.out.link(nn_out.input)

    return pipeline

def annotate_frame(frame, detections, fps):
    color = (0, 0, 255)
    for detection in detections:
        bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        label_idx = int(getattr(detection, "label", 0))
        label_text = LABELS[label_idx] if 0 <= label_idx < len(LABELS) else str(label_idx)
        cv2.putText(frame, label_text, (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    
    # Annotate the frame with the FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

# Create pipeline
print(f"[INFO] Using model blob: {YOLOV8N_MODEL}")
print(f"[INFO] Using model config: {YOLOV8N_CONFIG}")
pipeline = create_camera_pipeline(MODEL_CONFIG, YOLOV8N_MODEL)

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    detectionQueue = device.getOutputQueue("nn", maxSize=4, blocking=False)
    rgbQueue = device.getOutputQueue("rgb", maxSize=4, blocking=False)

    fps = 30  # Assuming 30 FPS for the OAK-D camera
    frame_width, frame_height = CAMERA_PREV_DIM
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    start_time = time.time()
    frame_count = 0
    latest_detections = []

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

    while True:
        inRgb = rgbQueue.get()
        frame = inRgb.getCvFrame()
        frame_count += 1

        inDet = detectionQueue.tryGet()
        if inDet is not None:
            latest_detections = inDet.detections
            if latest_detections:
                print("Detections", latest_detections)

        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        annotated = annotate_frame(frame, latest_detections, fps)
        cv2.imshow("Frame", annotated)
        out.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()

print(f"[INFO] Processed live stream and saved to {OUTPUT_VIDEO}")
