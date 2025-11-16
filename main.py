from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import depthai as dai
import numpy as np

print("[INFO] Starting OAK-D YOLO pipeline...")

# Load label map
with open("labels.txt", "r", encoding="utf-8") as labels_file:
    label_map = [line.strip() for line in labels_file if line.strip()]
print(f"[INFO] Loaded {len(label_map)} labels.")


def _build_aruco_detector() -> tuple[Any, Any, Any, Any]:
    """Prepare ArUco dictionary, parameters, and detector (with OpenCV version fallback)."""
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    detector = None
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
    return aruco, dictionary, detector, parameters


aruco_module, aruco_dict, aruco_detector, aruco_params = _build_aruco_detector()

# Map each ArUco ID to metadata describing the physical trap in the field.
TRAP_REGISTRY: dict[int, dict[str, str]] = {
    0: {"name": "Trap A", "location": "North block, row 2"},
    1: {"name": "Trap B", "location": "North block, row 5"},
    2: {"name": "Trap C", "location": "East block, row 1"},
}

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

use_xlink = hasattr(dai.node, "XLinkOut")


@dataclass
class CameraSetup:
    name: str
    blob_path: str | None = None

    def resolved_blob_path(self) -> str:
        if self.blob_path:
            candidate = Path(self.blob_path)
            if candidate.exists():
                print(f"[INFO] {self.name}: Using blob {candidate}")
                return str(candidate)
            raise FileNotFoundError(f"{self.name}: blob not found at {self.blob_path}")
        return resolve_blob_path()


@dataclass
class PipelineBundle:
    setup: CameraSetup
    pipeline: dai.Pipeline
    host_outputs: dict[str, dai.Node.Output]
    streams: dict[str, str]


def build_pipeline(setup: CameraSetup) -> PipelineBundle:
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setPreviewSize(640, 640)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    rgb_stream = cam_rgb.preview

    blob_path = setup.resolved_blob_path()
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(blob_path)
    print(f"[INFO] Configured {setup.name} NN with blob: {blob_path}")
    rgb_stream.link(nn.input)

    host_outputs: dict[str, dai.Node.Output] = {}
    stream_names = {"nn": f"{setup.name}_nn", "cam": f"{setup.name}_cam"}

    if use_xlink:
        nn_xout = pipeline.create(dai.node.XLinkOut)
        nn_xout.setStreamName(stream_names["nn"])
        nn.out.link(nn_xout.input)

        cam_xout = pipeline.create(dai.node.XLinkOut)
        cam_xout.setStreamName(stream_names["cam"])
        rgb_stream.link(cam_xout.input)
        print(f"[INFO] {setup.name}: Using XLinkOut for outputs.")
    else:
        host_outputs["nn"] = nn.out
        host_outputs["cam"] = rgb_stream
        print(f"[INFO] {setup.name}: Using host outputs.")

    return PipelineBundle(
        setup=setup,
        pipeline=pipeline,
        host_outputs=host_outputs,
        streams=stream_names,
    )


def detect_pest_traps(frame: np.ndarray) -> list[dict[str, Any]]:
    """Detect ArUco markers and map them to known trap metadata."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if aruco_detector is not None:
        corners, ids, _ = aruco_detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco_module.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    detections: list[dict[str, Any]] = []
    if ids is None:
        return detections

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        trap_info = TRAP_REGISTRY.get(marker_id, {"name": f"Marker {marker_id}", "location": "Unknown"})
        detections.append(
            {
                "marker_id": int(marker_id),
                "trap_name": trap_info["name"],
                "location": trap_info.get("location", "Unknown"),
                "corners": marker_corners.reshape((4, 2)).astype(int),
            }
        )
    return detections


def annotate_traps(frame: np.ndarray, trap_detections: list[dict[str, Any]], camera_name: str) -> None:
    """Draw trap outlines and text labels in-place."""
    for detection in trap_detections:
        corners = detection["corners"]
        cv2.polylines(frame, [corners], isClosed=True, color=(255, 0, 0), thickness=2)
        center = corners.mean(axis=0).astype(int)
        label = f"{detection['trap_name']} (ID {detection['marker_id']})"
        location = detection["location"]
        cv2.putText(frame, label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(
            frame,
            location,
            (center[0], center[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        print(
            f"[INFO] {camera_name}: Trap '{detection['trap_name']}' (ID:{detection['marker_id']}) "
            f"seen around {location}."
        )

# Device context manager
@contextmanager
def create_device_context(pipeline_obj: dai.Pipeline, device_info: dai.DeviceInfo | None = None):
    device = None
    try:
        pipeline_started = False
        try:
            if device_info is None:
                device = dai.Device(pipeline_obj)
            else:
                device = dai.Device(pipeline_obj, device_info)
            pipeline_started = True
        except TypeError:
            if device_info is None:
                device = dai.Device()
            else:
                device = dai.Device(device_info)
        if not pipeline_started and hasattr(device, "startPipeline"):
            device.startPipeline(pipeline_obj)
        yield device
    finally:
        if device is not None:
            device.close()

camera_setups = [
    CameraSetup(name="camera_1", blob_path="my_blobs/best_openvino_2022.1_6shave.blob"),
    CameraSetup(name="camera_2", blob_path="my_blobs/best_openvino_2022.1_6shave.blob"),
    #CameraSetup(name="camera_3", blob_path="my_blobs/alternate_model_a.blob"),
    #CameraSetup(name="camera_4", blob_path="my_blobs/alternate_model_b.blob"),
]

pipeline_bundles: list[PipelineBundle] = []
for setup in camera_setups:
    bundle = build_pipeline(setup)
    pipeline_bundles.append(bundle)

available_devices = dai.Device.getAllAvailableDevices()
if not available_devices:
    raise RuntimeError("[ERROR] No DepthAI devices detected.")

if len(available_devices) < len(pipeline_bundles):
    print(
        f"[WARN] Requested {len(pipeline_bundles)} camera(s) but only "
        f"{len(available_devices)} device(s) detected. Proceeding with available devices."
    )

active_pairs = list(zip(pipeline_bundles, available_devices))

active_devices = []
with ExitStack() as stack:
    for bundle, device_info in active_pairs:
        device = stack.enter_context(create_device_context(bundle.pipeline, device_info))
        print(f"[INFO] Connected to {bundle.setup.name} (MXID: {device.getMxId()})")

        if use_xlink:
            q_nn = device.getOutputQueue(bundle.streams["nn"], maxSize=4, blocking=False)
            q_cam = device.getOutputQueue(bundle.streams["cam"], maxSize=4, blocking=False)
        else:
            q_nn = bundle.host_outputs["nn"].createOutputQueue(maxSize=4, blocking=False)
            q_cam = bundle.host_outputs["cam"].createOutputQueue(maxSize=4, blocking=False)

        if q_nn is None or q_cam is None:
            raise RuntimeError(f"[ERROR] Output queues not initialized for {bundle.setup.name}.")
        active_devices.append(
            {
                "name": bundle.setup.name,
                "nn_queue": q_nn,
                "cam_queue": q_cam,
                "window": f"{bundle.setup.name} Inference",
            }
        )

    if not active_devices:
        raise RuntimeError("[ERROR] No active devices configured.")

    print("[INFO] Output queues initialized for all cameras.")
    running = True
    while running:
        for active in active_devices:
            in_cam = active["cam_queue"].tryGet()
            if in_cam is None:
                continue

            frame = in_cam.getCvFrame()
            print(f"[DEBUG] {active['name']}: Camera frame received.")

            detections = []
            in_nn = active["nn_queue"].tryGet()
            if in_nn is not None:
                print(f"[DEBUG] {active['name']}: NN packet type: {type(in_nn)}")
                if hasattr(in_nn, "detections"):
                    detections = in_nn.detections
                    print(f"[INFO] {active['name']}: {len(detections)} detections received.")
                else:
                    print(f"[WARN] {active['name']}: No 'detections' attribute in NN output.")

            for det in detections:
                if det.confidence < 0.3:
                    continue  # Ignore low confidence

                x1 = int(det.xmin * frame.shape[1])
                y1 = int(det.ymin * frame.shape[0])
                x2 = int(det.xmax * frame.shape[1])
                y2 = int(det.ymax * frame.shape[0])
                label = label_map[det.label] if det.label < len(label_map) else f"ID:{det.label}"
                confidence = det.confidence

                print(
                    f"[DEBUG] {active['name']}: Detected {label} ({confidence:.2f}) "
                    f"at [{x1},{y1},{x2},{y2}]"
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            trap_detections = detect_pest_traps(frame)
            if trap_detections:
                annotate_traps(frame, trap_detections, active["name"])
            cv2.imshow(active["window"], frame)

        if cv2.waitKey(1) == ord("q"):
            running = False

cv2.destroyAllWindows()
print("[INFO] Exiting pipeline.")
