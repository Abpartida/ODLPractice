from contextlib import contextmanager
from pathlib import Path
import blobconverter

import cv2
import depthai as dai
import numpy as np

print("[INFO] Starting OAK-D YOLO pipeline...")

# Load label map
with open("labels.txt", "r", encoding="utf-8") as labels_file:
    label_map = [line.strip() for line in labels_file if line.strip()]
print(f"[INFO] Loaded {len(label_map)} labels.")

def resolve_blob_path() -> str:
    candidates = [
        Path("best.rvc2/best.blob"),
        Path("best.rvc2_legacy.rvc2/best.blob"),
        Path("best.rvc3/best.blob"),
        Path("best.superblob"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".superblob":
            try:
                dai.OpenVINO.Blob(str(candidate))
            except RuntimeError:
                continue
        print(f"[INFO] Using blob: {candidate}")
        return str(candidate)
    raise FileNotFoundError("No supported DepthAI blob found.")

# Create pipeline
pipeline = dai.Pipeline()

# Set up color camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setPreviewSize(640, 640)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(30)

rgb_stream = cam_rgb.preview

# Load YOLO blob
nn = pipeline.create(dai.node.NeuralNetwork)
# Use manually built blob:
nn_path = "my_blobs/best_openvino_2022.1_6shave.blob"
nn.setBlobPath(nn_path)
print(f"[INFO] Set neural network blob path: {nn_path}")
rgb_stream.link(nn.input)

use_xlink = hasattr(dai.node, "XLinkOut")
host_outputs: dict[str, dai.Node.Output] = {}

if use_xlink:
    nn_xout = pipeline.create(dai.node.XLinkOut)
    nn_xout.setStreamName("nn")
    nn.out.link(nn_xout.input)

    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("cam")
    rgb_stream.link(cam_xout.input)
    print("[INFO] Using XLinkOut for outputs.")
else:
    host_outputs["nn"] = nn.out
    host_outputs["cam"] = rgb_stream
    print("[INFO] Using host outputs.")

# Device context manager
@contextmanager
def create_device_context(pipeline_obj: dai.Pipeline):
    device = None
    try:
        try:
            device = dai.Device()
        except TypeError:
            device = dai.Device(pipeline_obj)
            yield device
            return

        if hasattr(device, "startPipeline"):
            device.startPipeline(pipeline_obj)
            yield device
        else:
            device.close()
            device = dai.Device(pipeline_obj)
            yield device
    finally:
        if device is not None:
            device.close()

# Start pipeline
with create_device_context(pipeline) as device:
    if use_xlink:
        q_nn = device.getOutputQueue("nn")
        q_cam = device.getOutputQueue("cam")
    else:
        q_nn = host_outputs["nn"].createOutputQueue(maxSize=4, blocking=False)
        q_cam = host_outputs["cam"].createOutputQueue(maxSize=4, blocking=False)

    if q_nn is None or q_cam is None:
        raise RuntimeError("[ERROR] Output queues not initialized.")
    print("[INFO] Output queues initialized.")

    while True:
        in_nn = q_nn.get()
        in_cam = q_cam.get()
        if in_cam is None:
            print("[WARN] No camera frame received.")
            continue

        in_frame = in_cam.getCvFrame()
        print("[DEBUG] Camera frame received.")

        detections = []
        if in_nn is not None:
            print(f"[DEBUG] NN packet type: {type(in_nn)}")
            if hasattr(in_nn, 'detections'):
                detections = in_nn.detections
                print(f"[INFO] {len(detections)} detections received.")
            else:
                print("[WARN] No 'detections' attribute in NN output.")

        for det in detections:
            if det.confidence < 0.3:
                continue  # Ignore low confidence

            x1 = int(det.xmin * in_frame.shape[1])
            y1 = int(det.ymin * in_frame.shape[0])
            x2 = int(det.xmax * in_frame.shape[1])
            y2 = int(det.ymax * in_frame.shape[0])
            label = label_map[det.label] if det.label < len(label_map) else f"ID:{det.label}"
            confidence = det.confidence

            print(f"[DEBUG] Detected {label} ({confidence:.2f}) at [{x1},{y1},{x2},{y2}]")
            cv2.rectangle(in_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(in_frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("OAK-D Inference", in_frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
print("[INFO] Exiting pipeline.")