"""Main entry point for the ODL practice rig.
This file now exposes three clean sections:
  1. Flask/HTTP server for control + streaming.
  2. DepthAI pipeline and camera processing loop.
  3. Serial/drive control helpers.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from queue import Full, Queue
from typing import Any, Iterable

import cv2
import depthai as dai
import numpy as np
from flask import Flask, Response, jsonify, request
import serial
from serial import SerialException


# ======================================================================================
# Serial / Drive Control
# ======================================================================================

SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/ttyACM0")
SERIAL_BAUD = int(os.environ.get("SERIAL_BAUD", "115200"))
SERIAL_TIMEOUT_SEC = float(os.environ.get("SERIAL_TIMEOUT_SEC", "0.25"))

DRIVE_QUEUE_SIZE = int(os.environ.get("DRIVE_QUEUE_SIZE", "16"))
DRIVE_HANDLER_DEADLINE_SEC = float(os.environ.get("DRIVE_HANDLER_DEADLINE_SEC", "0.2"))
DRIVE_QUEUE_WAIT_FALLBACK_SEC = float(os.environ.get("DRIVE_QUEUE_WAIT_FALLBACK_SEC", "0.02"))

SerialResult = tuple[bool, str]


@dataclass(slots=True)
class SerialJob:
    cmd: str
    future: concurrent.futures.Future[SerialResult]
    created_at: float
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


class SerialController:
    """Manages the ESP32 serial connection and queued drive commands."""

    def __init__(self, port: str, baud: int, timeout_sec: float) -> None:
        self.port = port
        self.baud = baud
        self.timeout_sec = timeout_sec
        self._serial_lock = threading.Lock()
        self._serial: serial.Serial | None = None
        self._drive_queue: "Queue[SerialJob]" = Queue(maxsize=DRIVE_QUEUE_SIZE)
        self._drive_worker_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def send(self, cmd: str) -> SerialResult:
        """Send a command immediately, returning (ok, reply)."""
        with self._serial_lock:
            self._open_serial()
            if self._serial is None or not self._serial.is_open:
                return False, "Serial not connected"

            try:
                payload = (cmd.strip() + "\n").encode("utf-8")
                self._serial.write(payload)
                self._serial.flush()
                return True, "sent"
            except (SerialException, OSError) as exc:
                try:
                    if self._serial is not None:
                        self._serial.close()
                except Exception:
                    pass
                return False, f"Serial error: {exc}"

    def submit_drive_command(self, cmd: str, deadline_sec: float | None = None) -> SerialResult | None:
        """Queue a drive command and wait up to deadline_sec for completion."""
        if deadline_sec is None:
            deadline_sec = DRIVE_HANDLER_DEADLINE_SEC

        self._ensure_drive_worker_started()
        future: concurrent.futures.Future[SerialResult] = concurrent.futures.Future()
        job = SerialJob(cmd=cmd, future=future, created_at=time.monotonic())

        absolute_deadline = time.monotonic() + max(deadline_sec, 0.0)
        remaining = absolute_deadline - time.monotonic()
        if remaining <= 0:
            return None

        queue_wait = min(DRIVE_QUEUE_WAIT_FALLBACK_SEC, remaining)
        try:
            self._drive_queue.put(job, timeout=max(queue_wait, 0.0))
        except Full:
            return None

        remaining = absolute_deadline - time.monotonic()
        if remaining <= 0:
            return None

        try:
            return future.result(timeout=remaining)
        except concurrent.futures.TimeoutError:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _drive_worker_loop(self) -> None:
        while True:
            job = self._drive_queue.get()
            try:
                result = self.send(job.cmd)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as exc:
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                self._drive_queue.task_done()

    def _ensure_drive_worker_started(self) -> None:
        if self._drive_worker_thread and self._drive_worker_thread.is_alive():
            return
        self._drive_worker_thread = threading.Thread(
            target=self._drive_worker_loop,
            name="drive-serial-worker",
            daemon=True,
        )
        self._drive_worker_thread.start()

    def _open_serial(self) -> None:
        if self._serial is not None and self._serial.is_open:
            return
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout_sec,
                write_timeout=self.timeout_sec,
            )
            time.sleep(2.3)  # allow board reset + boot chatter
            try:
                self._serial.reset_input_buffer()
            except Exception:
                pass
        except SerialException as exc:
            self._serial = None
            print(f"[ERROR] Failed to open serial {self.port}: {exc}")


_DIRECTION_ALIAS = {
    "forward": "FORWARD",
    "fwd": "FORWARD",
    "up": "FORWARD",
    "start": "FORWARD",
    "go": "FORWARD",
    "back": "BACKWARD",
    "backward": "BACKWARD",
    "reverse": "BACKWARD",
    "left": "LEFT",
    "right": "RIGHT",
    "stop": "STOP",
    "halt": "STOP",
    "idle": "STOP",
}

JOYSTICK_DEADZONE = float(os.environ.get("JOYSTICK_DEADZONE", "0.3"))


def _command_from_direction(direction: Any) -> str | None:
    if not isinstance(direction, str):
        return None
    normalized = direction.strip().lower()
    if not normalized:
        return None
    return _DIRECTION_ALIAS.get(normalized)


def _command_from_axes(x: float, y: float) -> str:
    """Map analog joystick axes to discrete FORWARD/LEFT/RIGHT/STOP commands."""
    magnitude = max(abs(x), abs(y))
    if magnitude < JOYSTICK_DEADZONE:
        return "STOP"

    if abs(y) >= abs(x):
        if y > 0:
            return "FORWARD"
        if y < 0:
            return "BACKWARD"
        return "STOP"

    return "RIGHT" if x > 0 else "LEFT"


# ======================================================================================
# Frame Compositing + Streaming
# ======================================================================================


class FrameHub:
    """Thread-safe store for latest frames per camera and MJPEG streaming."""

    def __init__(self, grid_slots: int = 4):
        self.grid_slots = grid_slots
        self._frames: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def update(self, camera_name: str, frame: np.ndarray) -> None:
        with self._lock:
            self._frames[camera_name] = frame

    def _snapshot(self) -> list[tuple[str, np.ndarray]]:
        with self._lock:
            return [(name, frame) for name, frame in self._frames.items() if frame is not None]

    def stream_frames(self, idle_sleep: float = 0.1, frame_interval: float = 0.05) -> Iterable[bytes]:
        while True:
            composed = self._compose_tiled_frame()
            if composed is None:
                time.sleep(idle_sleep)
                continue

            success, buffer = cv2.imencode(".jpg", composed)
            if not success:
                time.sleep(frame_interval)
                continue

            payload = (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            yield payload
            time.sleep(frame_interval)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compose_tiled_frame(self) -> np.ndarray | None:
        frames = self._snapshot()
        if not frames:
            return None

        try:
            target_frame = frames[0][1]
            target_size = (target_frame.shape[1], target_frame.shape[0])

            resized: list[np.ndarray] = []
            labels: list[str] = []
            for name, frame in frames[: self.grid_slots]:
                if frame.shape[:2] != target_frame.shape[:2]:
                    frame = cv2.resize(frame, target_size)
                resized.append(frame)
                labels.append(name)

            while len(resized) < self.grid_slots:
                resized.append(np.zeros_like(resized[0]))
                labels.append("")

            row1 = cv2.hconcat(resized[:2])
            row2 = cv2.hconcat(resized[2:4])
            combined = cv2.vconcat([row1, row2])
            self._overlay_labels(combined, target_size, labels)
            return combined
        except cv2.error:
            return None

    def _overlay_labels(self, frame: np.ndarray, target_size: tuple[int, int], labels: list[str]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (0, 255, 0)
        thickness = 2
        positions = [
            (10, 30),
            (target_size[0] + 10, 30),
            (10, target_size[1] + 30),
            (target_size[0] + 10, target_size[1] + 30),
        ]
        for label, (x_pos, y_pos) in zip(labels, positions):
            if label:
                cv2.putText(frame, label, (x_pos, y_pos), font, font_scale, font_color, thickness)


# ======================================================================================
# Flask Application
# ======================================================================================


def create_app(serial_controller: SerialController, frame_hub: FrameHub) -> Flask:
    app = Flask(__name__)

    @app.post("/auth/login")
    def auth_login():
        return jsonify(access_token="dev-token", token_type="bearer", expires_in=3600), 200

    @app.get("/status")
    def status():
        ok, reply = serial_controller.send("STAT?")
        return (
            jsonify(system_status="Nominal" if ok else "Degraded", serial=reply),
            200 if ok else 502,
        )

    @app.post("/api/drive/joystick")
    def api_drive_joystick():
        data = request.get_json(silent=True) or {}
        direction = data.get("direction")
        cmd = _command_from_direction(direction)
        if cmd:
            result = serial_controller.submit_drive_command(cmd)
            if result is None:
                return jsonify(error="drive queue busy", cmd=cmd), 503
            ok, reply = result
            return jsonify(status="ok" if ok else "err", cmd=cmd, serial=reply), 200 if ok else 502

        x = data.get("x", data.get("horizontal", 0.0))
        y = data.get("y", data.get("vertical", 0.0))
        try:
            x_f = float(x)
            y_f = float(y)
        except (TypeError, ValueError):
            return jsonify(error="bad joystick payload"), 400

        cmd = _command_from_axes(x_f, y_f)
        result = serial_controller.submit_drive_command(cmd)
        if result is None:
            return jsonify(error="drive queue busy", cmd=cmd), 503
        ok, reply = result
        return jsonify(status="ok" if ok else "err", cmd=cmd, serial=reply), 200 if ok else 502

    @app.post("/api/drive/stop")
    def api_drive_stop():
        result = serial_controller.submit_drive_command("STOP")
        if result is None:
            return jsonify(error="drive queue busy", cmd="STOP"), 503
        ok, reply = result
        return jsonify(status="stopped" if ok else "err", serial=reply), 200 if ok else 502

    def _simple_serial_endpoint(command: str):
        ok, reply = serial_controller.send(command)
        return jsonify(ok=ok, reply=reply), 200 if ok else 502

    @app.post("/api/lift/up")
    def api_lift_up():
        return _simple_serial_endpoint("LIFT UP")

    @app.post("/api/lift/down")
    def api_lift_down():
        return _simple_serial_endpoint("LIFT DOWN")

    @app.post("/api/lift/stop")
    def api_lift_stop():
        return _simple_serial_endpoint("LIFT STOP")

    @app.post("/api/fan/on")
    def api_fan_on():
        return _simple_serial_endpoint("FAN ON")

    @app.post("/api/fan/off")
    def api_fan_off():
        return _simple_serial_endpoint("FAN OFF")

    @app.route("/video")
    def video():
        return Response(frame_hub.stream_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def index():
        return "<h1>Live Stream</h1><img src=\"/video\"/>"

    return app


# ======================================================================================
# DepthAI Model + Pipeline Utilities
# ======================================================================================


RESULT_DIR = Path(os.environ.get("RESULT_DIR", "my_blobs/pestv5"))
MODEL_CONFIG: dict[str, Any] | None = None
MODEL_CONFIG_PATH: Path | None = None
DEFAULT_MODEL_BLOB: Path | None = None
label_map: list[str] = []


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def resolve_model_artifacts(result_dir: Path) -> tuple[Path, Path]:
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


try:
    DEFAULT_MODEL_BLOB, MODEL_CONFIG_PATH = resolve_model_artifacts(RESULT_DIR)
    MODEL_CONFIG = load_config(MODEL_CONFIG_PATH)
    label_map = MODEL_CONFIG.get("mappings", {}).get("labels", [])
except FileNotFoundError as err:
    print(f"[WARN] {err}")

if not label_map:
    with open("labels.txt", "r", encoding="utf-8") as labels_file:
        label_map = [line.strip() for line in labels_file if line.strip()]


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


def resolve_input_dimensions(model_config: dict[str, Any] | None, default_dim: tuple[int, int]) -> tuple[int, int]:
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


def determine_pipeline_input_dim(blob_path: str | None, fallback: tuple[int, int]) -> tuple[int, int]:
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

    if not MODEL_CONFIG:
        raise RuntimeError("MODEL_CONFIG must be available for YOLOv5 pipelines.")

    cam_output, nn_output = create_yolo_pipeline_nodes(pipeline, MODEL_CONFIG, blob_path, input_dim)

    if use_xlink:
        nn_xout = pipeline.create(dai.node.XLinkOut)
        nn_xout.setStreamName(stream_names["nn"])
        nn_output.link(nn_xout.input)

        cam_xout = pipeline.create(dai.node.XLinkOut)
        cam_xout.setStreamName(stream_names["cam"])
        cam_output.link(cam_xout.input)
    else:
        host_outputs["nn"] = nn_output
        host_outputs["cam"] = cam_output

    return PipelineBundle(
        setup=setup,
        pipeline=pipeline,
        host_outputs=host_outputs,
        streams=stream_names,
        blob_path=blob_path,
    )


def detect_pest_traps(frame: np.ndarray) -> list[dict[str, Any]]:
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
    for detection in trap_detections:
        corners = detection["corners"]
        cv2.polylines(frame, [corners], isClosed=True, color=(255, 0, 0), thickness=2)
        label = f"{detection['trap_name']} (ID {detection['marker_id']})"
        location = detection["location"]
        bl_idx = corners[:, 1].argmax()
        bottom_left = tuple(corners[bl_idx])
        x_bl, y_bl = int(bottom_left[0]), int(bottom_left[1])
        cv2.putText(frame, label, (x_bl, y_bl - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(
            frame,
            location,
            (x_bl, y_bl),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 155),
            1,
        )


@contextmanager
def create_device_context(pipeline_obj: dai.Pipeline, device_info: dai.DeviceInfo | None = None):
    device = None
    try:
        time.sleep(2)
        if device_info is None:
            device = dai.Device(pipeline_obj)
        else:
            try:
                device = dai.Device(pipeline_obj, device_info, dai.UsbSpeed.SUPER)
            except TypeError:
                device = dai.Device(pipeline_obj, device_info)
        yield device
    finally:
        if device is not None:
            device.close()


def start_pipeline(frame_hub: FrameHub) -> None:
    camera_setups = [
        CameraSetup(name="camera_1_left", blob_path="my_blobs/pestv5/best_openvino_2022.1_6shave.blob"),
        CameraSetup(name="camera_2_right", blob_path="my_blobs/pestv5/best_openvino_2022.1_6shave.blob"),
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

    active_devices = []
    with ExitStack() as stack:
        for bundle, device_info in active_pairs:
            try:
                device = stack.enter_context(create_device_context(bundle.pipeline, device_info))
            except RuntimeError as exc:
                print(f"[WARNING] Skipping camera '{bundle.setup.name}' due to error: {exc}")
                continue

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
                    "visible_traps": set(),
                }
            )

        if not active_devices:
            raise RuntimeError("[ERROR] No active devices configured.")

        unique_traps_seen: set[int] = set()
        while True:
            for active in active_devices:
                in_cam = active["cam_queue"].tryGet()
                if in_cam is None:
                    continue

                frame = in_cam.getCvFrame()
                frame_hub.update(active["name"], frame)

                detections = []
                in_nn = active["nn_queue"].tryGet()
                if in_nn is not None and hasattr(in_nn, "detections"):
                    detections = in_nn.detections

                for det in detections:
                    if det.confidence < 0.3:
                        continue

                    x1 = int(det.xmin * frame.shape[1])
                    y1 = int(det.ymin * frame.shape[0])
                    x2 = int(det.xmax * frame.shape[1])
                    y2 = int(det.ymax * frame.shape[0])
                    label = label_map[det.label] if det.label < len(label_map) else f"ID:{det.label}"
                    confidence = det.confidence

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
                height = frame.shape[0]
                cv2.putText(
                    frame,
                    trap_count_label,
                    (10, height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    unique_count_label,
                    (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                if trap_ids_in_view:
                    print("[ACTION] Turn Off systems")
                    print(f"[METRIC] Unique traps seen so far: {len(unique_traps_seen)}")


# ======================================================================================
# Application bootstrap
# ======================================================================================

SERIAL_CONTROLLER = SerialController(
    port=SERIAL_PORT,
    baud=SERIAL_BAUD,
    timeout_sec=SERIAL_TIMEOUT_SEC,
)
FRAME_HUB = FrameHub()
app = create_app(SERIAL_CONTROLLER, FRAME_HUB)


def main() -> None:
    pipeline_thread = threading.Thread(target=start_pipeline, args=(FRAME_HUB,), daemon=True)
    pipeline_thread.start()
    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
