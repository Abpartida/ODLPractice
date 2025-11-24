import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import depthai as dai
import numpy as np

print("[INFO] Starting OAK-D YOLO pipeline...")


# Global variable to hold the latest processed frame for streaming
latest_frame = None


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def resolve_model_artifacts(result_dir: Path) -> tuple[Path, Path]:
    """Pick the YOLO blob + json from the exported result folder."""
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir.resolve()}")

    blob_files = sorted(result_dir.glob("*.blob"))
    if not blob_files:
        raise FileNotFoundError(f"No .blob files found under {result_dir.resolve()}")

    json_files = sorted(result_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found under {result_dir.resolve()}")

    def _pick_preferred(paths: list[Path]) -> Path:
        for path in paths:
            if "best" in path.stem.lower():
                return path
        return paths[0]

    return _pick_preferred(blob_files), _pick_preferred(json_files)


RESULT_DIR = Path(os.environ.get("RESULT_DIR", "my_blobs/pestv5"))
MODEL_CONFIG: dict[str, Any] | None = None
DEFAULT_MODEL_BLOB: Path | None = None
label_map: list[str] = []

try:
    DEFAULT_MODEL_BLOB, MODEL_CONFIG_PATH = resolve_model_artifacts(RESULT_DIR)
    MODEL_CONFIG = load_config(MODEL_CONFIG_PATH)
    label_map = MODEL_CONFIG.get("mappings", {}).get("labels", [])
    print(f"[INFO] Loaded model config from {MODEL_CONFIG_PATH}.")
except FileNotFoundError as err:
    print(f"[WARN] {err}")

if not label_map:
    with open("labels.txt", "r", encoding="utf-8") as labels_file:
        label_map = [line.strip() for line in labels_file if line.strip()]
    print(f"[INFO] Loaded {len(label_map)} labels from labels.txt.")
else:
    print(f"[INFO] Loaded {len(label_map)} labels from model config.")


DEFAULT_CAMERA_DIM = (640, 640)
_BLOB_INPUT_CACHE: dict[str, tuple[int, int]] = {}
_MIN_VALID_DIM = 16


def _parse_size_string(size_str: str) -> tuple[int, int] | None:
    if not isinstance(size_str, str) or "x" not in size_str:
        return None
    try:
        width_str, height_str = size_str.lower().split("x")
        return int(width_str), int(height_str)
    except (ValueError, TypeError):
        return None


def resolve_input_dimensions(
    model_config: dict[str, Any] | None, default_dim: tuple[int, int]
) -> tuple[int, int]:
    if not model_config:
        return default_dim
    size_str = model_config.get("nn_config", {}).get("input_size")
    parsed = _parse_size_string(size_str) if isinstance(size_str, str) else None
    if parsed:
        return parsed
    return default_dim


def _infer_blob_input_size(blob_path: str) -> tuple[int, int] | None:
    if blob_path in _BLOB_INPUT_CACHE:
        return _BLOB_INPUT_CACHE[blob_path]
    try:
        blob = dai.OpenVINO.Blob(blob_path)
    except RuntimeError as err:
        print(f"[WARN] Unable to inspect blob input size for {blob_path}: {err}")
        return None
    inputs = getattr(blob, "networkInputs", {})
    if not inputs:
        return None
    tensor_info = next(iter(inputs.values()))
    dims = getattr(tensor_info, "dims", None)
    if not dims or len(dims) < 4:
        return None
    dims = [int(value) for value in dims if isinstance(value, (int, float))]
    candidate_dims = [val for val in dims if val > _MIN_VALID_DIM]
    width: int | None = None
    height: int | None = None
    if len(candidate_dims) >= 2:
        candidate_dims.sort()
        height = candidate_dims[-2]
        width = candidate_dims[-1]
    elif len(dims) >= 4:
        width = int(dims[-1])
        height = int(dims[-2])

    if width and height and width > _MIN_VALID_DIM and height > _MIN_VALID_DIM:
        _BLOB_INPUT_CACHE[blob_path] = (width, height)
        return _BLOB_INPUT_CACHE[blob_path]

    print(
        f"[WARN] Unable to determine valid input size from blob dims {dims} "
        f"for {blob_path}; falling back to config/default."
    )
    return None


def determine_pipeline_input_dim(
    blob_path: str | None, fallback: tuple[int, int]
) -> tuple[int, int]:
    if "NN_INPUT_SIZE" in os.environ:
        override = _parse_size_string(os.environ["NN_INPUT_SIZE"])
        if override:
            return override
    if "MODEL_INPUT_SIZE" in os.environ:
        override = _parse_size_string(os.environ["MODEL_INPUT_SIZE"])
        if override:
            return override
    if blob_path:
        blob_dim = _infer_blob_input_size(blob_path)
        if blob_dim:
            return blob_dim
    return fallback


CAMERA_PREVIEW_DIM = resolve_input_dimensions(MODEL_CONFIG, DEFAULT_CAMERA_DIM)


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

use_xlink = hasattr(dai.node, "XLinkOut")


@dataclass
class CameraSetup:
    name: str
    blob_path: str

    def resolved_blob_path(self) -> str:
        candidate = Path(self.blob_path)
        if candidate.exists():
            print(f"[INFO] {self.name}: Using blob {candidate}")
            return str(candidate)
        raise FileNotFoundError(f"{self.name}: blob not found at {self.blob_path}")


@dataclass
class PipelineBundle:
    setup: CameraSetup
    pipeline: dai.Pipeline
    host_outputs: dict[str, dai.Node.Output]
    streams: dict[str, str]
    blob_path: str



def create_yolo_pipeline_nodes(
    pipeline: dai.Pipeline, model_config: dict[str, Any], blob_path: str, input_dim: tuple[int, int]
) -> tuple[dai.Node.Output, dai.Node.Output]:
    """Configure the ColorCamera + YoloDetectionNetwork graph from the working script."""
    nn_config = model_config.get("nn_config", {})
    metadata = nn_config.get("NN_specific_metadata", {})

    classes = int(metadata.get("classes", len(label_map)))
    coordinates = int(metadata.get("coordinates", 4))
    anchors = metadata.get("anchors", []) or []
    anchor_masks = metadata.get("anchor_masks", {}) or {}
    iou_threshold = float(metadata.get("iou_threshold", 0.5))
    confidence_threshold = float(metadata.get("confidence_threshold", 0.5))

    input_width, input_height = input_dim

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(input_width, input_height)
    cam_rgb.setInterleaved(False)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
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
    detection_network.setBlobPath(blob_path)
    detection_network.setNumInferenceThreads(2)
    detection_network.input.setBlocking(False)

    cam_rgb.preview.link(detection_network.input)
    return detection_network.passthrough, detection_network.out


def create_yolo_pipeline_nodes(
    pipeline: dai.Pipeline, model_config: dict[str, Any], blob_path: str, input_dim: tuple[int, int]
) -> tuple[dai.Node.Output, dai.Node.Output]:
    """Configure the ColorCamera + YoloDetectionNetwork graph from the working script."""
    nn_config = model_config.get("nn_config", {})
    metadata = nn_config.get("NN_specific_metadata", {})

    classes = int(metadata.get("classes", len(label_map)))
    coordinates = int(metadata.get("coordinates", 4))
    anchors = metadata.get("anchors", []) or []
    anchor_masks = metadata.get("anchor_masks", {}) or {}
    iou_threshold = float(metadata.get("iou_threshold", 0.5))
    confidence_threshold = float(metadata.get("confidence_threshold", 0.5))

    input_width, input_height = input_dim

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(input_width, input_height)
    cam_rgb.setInterleaved(False)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
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
    detection_network.setBlobPath(blob_path)
    detection_network.setNumInferenceThreads(2)
    detection_network.input.setBlocking(False)

    cam_rgb.preview.link(detection_network.input)
    return detection_network.passthrough, detection_network.out


def build_pipeline(setup: CameraSetup) -> PipelineBundle:
    pipeline = dai.Pipeline()

    blob_path = setup.resolved_blob_path()
    host_outputs: dict[str, dai.Node.Output] = {}
    stream_names = {"nn": f"{setup.name}_nn", "cam": f"{setup.name}_cam"}
    input_dim = determine_pipeline_input_dim(blob_path, CAMERA_PREVIEW_DIM)
    print(f"[INFO] {setup.name}: Using input size {input_dim[0]}x{input_dim[1]}.")

    if not MODEL_CONFIG:
        raise RuntimeError("MODEL_CONFIG must be available for YOLOv5 pipelines.")

    cam_output, nn_output = create_yolo_pipeline_nodes(
        pipeline, MODEL_CONFIG, blob_path, input_dim
    )
    print(f"[INFO] {setup.name}: Using YoloDetectionNetwork with blob {blob_path}.")

    if use_xlink:
        nn_xout = pipeline.create(dai.node.XLinkOut)
        nn_xout.setStreamName(stream_names["nn"])
        nn_output.link(nn_xout.input)

        cam_xout = pipeline.create(dai.node.XLinkOut)
        cam_xout.setStreamName(stream_names["cam"])
        cam_output.link(cam_xout.input)
        print(f"[INFO] {setup.name}: Using XLinkOut for outputs.")
    else:
        host_outputs["nn"] = nn_output
        host_outputs["cam"] = cam_output
        print(f"[INFO] {setup.name}: Using host outputs.")

    return PipelineBundle(
        setup=setup,
        pipeline=pipeline,
        host_outputs=host_outputs,
        streams=stream_names,
        blob_path=blob_path,
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
        if device_info is None:
            device = dai.Device(pipeline_obj)
        else:
            try:
                device = dai.Device(device_info, usbSpeed=dai.UsbSpeed.SUPER)
            except TypeError:
                device = dai.Device(device_info)
            if hasattr(device, "startPipeline"):
                device.startPipeline(pipeline_obj)
        yield device
    finally:
        if device is not None:
            device.close()

if __name__ == "__main__":
    camera_setups = [
        CameraSetup(name="camera_1_left", blob_path="my_blobs/pestv5/best_openvino_2022.1_6shave.blob"),
        CameraSetup(name="camera_2_right", blob_path="my_blobs/pestv5/best_openvino_2022.1_6shave.blob"),
        #CameraSetup(name="camera_3_front", blob_path="my_blobs/alternate_model_a.blob"),
        #CameraSetup(name="camera_4_back", blob_path="my_blobs/alternate_model_b.blob"),
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
    if not active_pairs:
        raise RuntimeError("[ERROR] Unable to pair pipelines with available devices.")

    print(f"[INFO] Activating {len(active_pairs)} of {len(pipeline_bundles)} configured camera pipeline(s).")

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
                    "visible_traps": set(),
                }
            )

        if not active_devices:
            raise RuntimeError("[ERROR] No active devices configured.")

        print("[INFO] Output queues initialized for all cameras.")
        unique_traps_seen: set[int] = set()
        running = True
        while running:
            for active in active_devices:
                in_cam = active["cam_queue"].tryGet()
                if in_cam is None:
                    continue

                frame = in_cam.getCvFrame()
                latest_frame = frame
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
                trap_ids_in_view = {detection["marker_id"] for detection in trap_detections}
                active["visible_traps"] = trap_ids_in_view
                unique_traps_seen.update(trap_ids_in_view)
                trap_count_label = f"Traps visible: {len(trap_ids_in_view)}"
                unique_count_label = f"Unique traps seen: {len(unique_traps_seen)}"
                cv2.putText(
                    frame,
                    trap_count_label,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    unique_count_label,
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                if trap_ids_in_view:
                    print(
                        f"[INFO] {active['name']}: Currently viewing {len(trap_ids_in_view)} trap(s): "
                        f"{sorted(trap_ids_in_view)}"
                    )
                    print("[ACTION] Turn Off systems")
                    print(f"[METRIC] Unique traps seen so far: {len(unique_traps_seen)}")
                # cv2.imshow(active["window"], frame)  # Disabled for headless/web streaming

            # if cv2.waitKey(1) == ord("q"):
            #     running = False

    # cv2.destroyAllWindows()
    print("[INFO] Exiting pipeline.")
