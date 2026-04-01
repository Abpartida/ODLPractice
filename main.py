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
import sqlite3
from datetime import datetime, timezone
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable, Iterable

import cv2
import depthai as dai
import numpy as np
from flask import Flask, Response, abort, jsonify, request
from flask_sock import Sock
from simple_websocket import ConnectionClosed
import serial
from serial import SerialException


# ======================================================================================
# Serial / Drive Control
# ======================================================================================

SERIAL_PORT = os.environ.get("SERIAL_PORT", "/dev/ttyACM0")
SERIAL_BAUD = int(os.environ.get("SERIAL_BAUD", "115200"))
SERIAL_TIMEOUT_SEC = float(os.environ.get("SERIAL_TIMEOUT_SEC", "0.25"))

DRIVE_QUEUE_SIZE = int(os.environ.get("DRIVE_QUEUE_SIZE", "16"))
DRIVE_HANDLER_DEADLINE_SEC = float(os.environ.get("DRIVE_HANDLER_DEADLINE_SEC", "3.0"))
DRIVE_QUEUE_WAIT_FALLBACK_SEC = float(os.environ.get("DRIVE_QUEUE_WAIT_FALLBACK_SEC", "0.02"))

def _read_float_env(var_name: str, default: float) -> float:
    try:
        return float(os.environ.get(var_name, default))
    except (TypeError, ValueError):
        return default


def _read_int_env(var_name: str, default: int, *, minimum: int | None = None) -> int:
    raw_value = os.environ.get(var_name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        print(f"[WARN] Invalid integer for {var_name}='{raw_value}'. Using default ({default}).")
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed

JOYSTICK_DEADZONE = _read_float_env("JOYSTICK_DEADZONE", 0.3)


def _env_flag(name: str, default: str = "0") -> bool:
    value = str(os.environ.get(name, default)).strip().lower()
    return value not in {"", "0", "false", "no", "off"}


DRIVE_DEBUG_LOGS = _env_flag("DRIVE_DEBUG_LOGS", "1")
DRIVE_HEARTBEAT_INTERVAL_SEC = _read_float_env("DRIVE_HEARTBEAT_INTERVAL_SEC", 0.2)

DEFAULT_LOWER_OBSTACLE_HEIGHT_FT = _read_float_env("OBSTACLE_LOWER_HEIGHT_FT", 1.5)
DEFAULT_UPPER_OBSTACLE_HEIGHT_FT = _read_float_env("OBSTACLE_UPPER_HEIGHT_FT", 3.5)


def utc_now_iso(timespec: str = "seconds") -> str:
    """Return an ISO 8601 UTC timestamp with a 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat(timespec=timespec).replace("+00:00", "Z")

# Default actuator commands (override via env vars if firmware differs)
FAN_ON_COMMAND = os.environ.get("FAN_ON_COMMAND", "FAN")
FAN_OFF_COMMAND = os.environ.get("FAN_OFF_COMMAND", "STOP")
LIFT_UP_COMMAND = os.environ.get("LIFT_UP_COMMAND", "UP")
LIFT_DOWN_COMMAND = os.environ.get("LIFT_DOWN_COMMAND", "DOWN")
LIFT_STOP_COMMAND = os.environ.get("LIFT_STOP_COMMAND", "STOP")

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
            if DRIVE_DEBUG_LOGS:
                snapshot = self.drive_queue_snapshot()
                print(
                    f"[DRIVE] Timeout waiting for job '{job.cmd}' "
                    f"(queue_depth={snapshot['depth']}, worker_alive={snapshot['worker_alive']})"
                )
            return None

    def flush_drive_queue(self, reason: str | None = None) -> int:
        """Drop all queued drive jobs, marking their futures as failed."""
        dropped = 0
        while True:
            try:
                job = self._drive_queue.get_nowait()
            except Empty:
                break
            dropped += 1
            if not job.future.done():
                job.future.set_result(
                    (
                        False,
                        reason or "Drive queue flushed",
                    )
                )
            self._drive_queue.task_done()
        if DRIVE_DEBUG_LOGS and dropped:
            print(f"[DRIVE] Flushed {dropped} queued job(s): {reason or 'no reason provided'}")
        return dropped

    def drive_queue_snapshot(self) -> dict[str, Any]:
        """Return lightweight telemetry for queue health."""
        worker_alive = self._drive_worker_thread.is_alive() if self._drive_worker_thread else False
        return {
            "depth": self._drive_queue.qsize(),
            "max_size": self._drive_queue.maxsize,
            "worker_alive": worker_alive,
        }

    def in_waiting(self) -> int:
        """Expose pending byte count for shared serial consumers."""
        with self._serial_lock:
            self._open_serial()
            if self._serial is None or not self._serial.is_open:
                return 0
            try:
                return int(getattr(self._serial, "in_waiting", 0))
            except (SerialException, OSError, ValueError):
                return 0

    def readline(self, timeout_sec: float | None = 0.0) -> bytes:
        """Read a single line from the serial port without blocking the drive worker."""
        with self._serial_lock:
            self._open_serial()
            if self._serial is None or not self._serial.is_open:
                return b""
            previous_timeout = getattr(self._serial, "timeout", None)
            if timeout_sec is not None:
                try:
                    self._serial.timeout = max(timeout_sec, 0.0)
                except (ValueError, AttributeError):
                    pass
            try:
                return self._serial.readline()
            except (SerialException, OSError):
                return b""
            finally:
                if timeout_sec is not None and self._serial is not None:
                    try:
                        self._serial.timeout = previous_timeout
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _drive_worker_loop(self) -> None:
        while True:
            job = self._drive_queue.get()
            try:
                start = time.monotonic()
                if DRIVE_DEBUG_LOGS:
                    print(f"[DRIVE] Executing job {job.job_id}: {job.cmd}")
                result = self.send(job.cmd)
                if not job.future.done():
                    job.future.set_result(result)
                if DRIVE_DEBUG_LOGS:
                    duration_ms = (time.monotonic() - start) * 1000.0
                    ok = result[0] if isinstance(result, tuple) and result else False
                    print(
                        f"[DRIVE] Job {job.job_id} complete (ok={ok}) "
                        f"in {duration_ms:.1f} ms"
                    )
            except Exception as exc:
                if not job.future.done():
                    job.future.set_exception(exc)
                if DRIVE_DEBUG_LOGS:
                    print(f"[DRIVE] Job {job.job_id} failed: {exc}")
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


class DriveHeartbeat:
    """Re-sends the last drive command on a fixed cadence to satisfy firmware watchdogs."""

    def __init__(self, serial_controller: SerialController, interval_sec: float) -> None:
        self._serial = serial_controller
        self._interval = max(interval_sec, 0.05)
        self._lock = threading.Lock()
        self._active_cmd: str | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="drive-heartbeat", daemon=True)
        self._thread.start()

    def set_command(self, command: str | None) -> None:
        with self._lock:
            self._active_cmd = command if command else None

    def clear(self) -> None:
        self.set_command(None)

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            cmd = None
            with self._lock:
                cmd = self._active_cmd
            if cmd and cmd != "STOP":
                ok, reply = self._serial.send(cmd)
                if DRIVE_DEBUG_LOGS:
                    print(
                        f"[DRIVE] Heartbeat cmd={cmd} ok={ok} reply={reply}"
                    )


class SerialWriterAdapter:
    """Lightweight shim so other subsystems can reuse SerialController APIs."""

    def __init__(self, controller: SerialController | None) -> None:
        self._controller = controller

    def write(self, payload: bytes) -> None:
        if not self._controller:
            return
        try:
            command = payload.decode("utf-8", errors="ignore").strip()
        except Exception:
            return
        if command:
            self._controller.send(command)

    @property
    def in_waiting(self) -> int:
        if not self._controller:
            return 0
        return self._controller.in_waiting()

    def readline(self, timeout_sec: float | None = 0.0) -> bytes:
        if not self._controller:
            return b""
        return self._controller.readline(timeout_sec)

    def close(self) -> None:  # pragma: no cover - noop shim
        return


class ModeState:
    """Thread-safe robot mode tracking (manual vs autonomous)."""

    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    _VALID = {MANUAL, AUTONOMOUS}

    def __init__(self, initial_mode: str = MANUAL) -> None:
        self._mode = self._normalize(initial_mode)
        self._condition = threading.Condition()

    @classmethod
    def _normalize(cls, mode: str) -> str:
        if not isinstance(mode, str):
            raise ValueError("Mode must be provided as a string.")
        normalized = mode.strip().lower()
        if normalized not in cls._VALID:
            raise ValueError("Mode must be either 'manual' or 'autonomous'.")
        return normalized

    def set_mode(self, new_mode: str) -> bool:
        normalized = self._normalize(new_mode)
        with self._condition:
            changed = self._mode != normalized
            self._mode = normalized
            if changed:
                self._condition.notify_all()
            return changed

    def get_mode(self) -> str:
        with self._condition:
            return self._mode

    def is_autonomous(self) -> bool:
        return self.get_mode() == self.AUTONOMOUS

    def wait_for_mode(self, target_mode: str, timeout: float | None = None) -> bool:
        normalized_target = self._normalize(target_mode)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._mode != normalized_target:
                if timeout is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True


class ObstacleHeightState:
    """Mutable store for requested lower/upper actuator targets."""

    _MIN_DELTA = 0.05

    def __init__(self, lower_ft: float, upper_ft: float) -> None:
        self._lock = threading.Lock()
        safe_lower = max(lower_ft, 0.1)
        safe_upper = max(upper_ft, safe_lower + self._MIN_DELTA)
        self._lower = safe_lower
        self._upper = safe_upper
        self._version = 0
        self._updated_at = utc_now_iso("seconds")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "lower_height": self._lower,
                "upper_height": self._upper,
                "updated_at": self._updated_at,
                "version": self._version,
            }

    def get_range(self) -> tuple[float, float, int]:
        with self._lock:
            return self._lower, self._upper, self._version

    def set_heights(self, lower: float | None, upper: float | None) -> dict[str, Any]:
        if lower is None and upper is None:
            raise ValueError("At least one of lower or upper height must be provided.")

        with self._lock:
            new_lower = self._lower if lower is None else lower
            new_upper = self._upper if upper is None else upper
            if new_lower <= 0 or new_upper <= 0:
                raise ValueError("Heights must be greater than zero.")
            if new_upper - new_lower < self._MIN_DELTA:
                raise ValueError("Upper height must be greater than lower height.")

            changed = (new_lower != self._lower) or (new_upper != self._upper)
            if changed:
                self._lower = new_lower
                self._upper = new_upper
                self._version += 1
                self._updated_at = utc_now_iso("seconds")

            return {
                "lower_height": self._lower,
                "upper_height": self._upper,
                "updated_at": self._updated_at,
                "version": self._version,
                "updated": changed,
            }

@dataclass(slots=True)
class CommandDispatchResult:
    status_code: int
    payload: dict[str, Any]

    def envelope(self, *, request_id: str | None = None, message_type: str | None = None) -> dict[str, Any]:
        data = dict(self.payload)
        data.setdefault("timestamp", utc_now_iso("seconds"))
        data["status_code"] = self.status_code
        data["status"] = "ok" if self.status_code < 400 else "error"
        if request_id:
            data["request_id"] = request_id
        if message_type:
            data.setdefault("type", message_type)
        return data


class ControlCommandDispatcher:
    """Shared command processor for HTTP + WebSocket control surfaces."""

    _DRIVE_COMMANDS = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}

    def __init__(
        self,
        serial_controller: SerialController,
        mode_state: ModeState,
        joystick_deadzone: float = JOYSTICK_DEADZONE,
        drive_heartbeat: DriveHeartbeat | None = None,
        obstacle_height_state: ObstacleHeightState | None = None,
    ) -> None:
        self._serial = serial_controller
        self._mode = mode_state
        self._deadzone = joystick_deadzone
        self._drive_heartbeat = drive_heartbeat
        self._obstacle_heights = obstacle_height_state
        self._ws_handlers: dict[str, Callable[[dict[str, Any]], CommandDispatchResult]] = {
            "drive": self._ws_drive,
            "lift": self._ws_lift,
            "fan": self._ws_fan,
            "mode": self._ws_mode,
            "status": self._ws_status,
            "ping": self._ws_ping,
        }

    # ------------------------------------------------------------------
    # Public helpers for HTTP endpoints
    # ------------------------------------------------------------------
    def execute_drive_command(self, command: str) -> CommandDispatchResult:
        normalized = self._normalize_keyword(command)
        if normalized not in self._DRIVE_COMMANDS:
            return self._error_result(f"Unsupported drive command '{command}'.")

        if normalized == "STOP":
            return self._handle_stop_command()

        result = self._serial.submit_drive_command(normalized)
        if result is None:
            return CommandDispatchResult(
                503,
                {
                    "command": normalized,
                    "detail": "Drive queue busy",
                    "queue": self._serial.drive_queue_snapshot(),
                },
            )

        ok, reply = result
        self._update_drive_heartbeat(normalized, 200 if ok else 502)
        return CommandDispatchResult(
            200 if ok else 502,
            {
                "command": normalized,
                "serial_reply": reply,
            },
        )

    def execute_drive_from_axes(self, x: float, y: float) -> CommandDispatchResult:
        command = self._reduce_axes_to_command(x, y)
        response = self.execute_drive_command(command)
        response.payload.setdefault("joystick", {"x": x, "y": y})
        return response

    def execute_lift_command(self, verb: str) -> CommandDispatchResult:
        mapping = {
            "UP": LIFT_UP_COMMAND,
            "DOWN": LIFT_DOWN_COMMAND,
            "STOP": LIFT_STOP_COMMAND,
        }
        normalized = self._normalize_keyword(verb)
        if normalized not in mapping:
            return self._error_result("Lift command must be one of: up, down, stop.")
        return self._send_simple(mapping[normalized], topic="lift", verb=normalized)

    def execute_fan_command(self, verb: str) -> CommandDispatchResult:
        normalized = self._normalize_keyword(verb)
        if normalized not in {"ON", "OFF"}:
            return self._error_result("Fan command must be 'on' or 'off'.")
        command = FAN_ON_COMMAND if normalized == "ON" else FAN_OFF_COMMAND
        return self._send_simple(command, topic="fan", verb=normalized)

    def get_obstacle_heights(self) -> CommandDispatchResult:
        if not self._obstacle_heights:
            return self._error_result("Obstacle height subsystem not configured.", status_code=503)
        snapshot = self._obstacle_heights.snapshot()
        return CommandDispatchResult(200, snapshot)

    def set_obstacle_heights(self, lower: Any, upper: Any) -> CommandDispatchResult:
        if not self._obstacle_heights:
            return self._error_result("Obstacle height subsystem not configured.", status_code=503)

        provided_lower = lower is not None
        provided_upper = upper is not None
        if not provided_lower and not provided_upper:
            return self._error_result("Provide at least 'lower' or 'upper' height in feet.")

        lower_value = self._coerce_positive_height(lower) if provided_lower else None
        upper_value = self._coerce_positive_height(upper) if provided_upper else None

        if provided_lower and lower_value is None:
            return self._error_result("Lower height must be a positive number.")
        if provided_upper and upper_value is None:
            return self._error_result("Upper height must be a positive number.")

        try:
            result = self._obstacle_heights.set_heights(lower_value, upper_value)
        except ValueError as exc:
            return self._error_result(str(exc))
        return CommandDispatchResult(200, result)

    def get_mode_response(self) -> CommandDispatchResult:
        return CommandDispatchResult(200, {"mode": self._mode.get_mode()})

    def set_mode_response(self, value: str) -> CommandDispatchResult:
        if not isinstance(value, str):
            return self._error_result("Mode must be provided as a string.")
        try:
            changed = self._mode.set_mode(value)
        except ValueError as exc:
            return self._error_result(str(exc))
        return CommandDispatchResult(200, {"mode": self._mode.get_mode(), "changed": changed})

    def status_probe(self) -> CommandDispatchResult:
        ok, reply = self._serial.send("STAT?")
        return CommandDispatchResult(
            200 if ok else 502,
            {
                "system_status": "Nominal" if ok else "Degraded",
                "serial": reply,
                "mode": self._mode.get_mode(),
            },
        )

    # ------------------------------------------------------------------
    # WebSocket dispatcher
    # ------------------------------------------------------------------
    def dispatch_ws_message(
        self,
        message: dict[str, Any],
        *,
        fallback_request_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = str(message.get("request_id") or message.get("id") or fallback_request_id or uuid.uuid4().hex[:10])
        msg_type = str(message.get("type") or "").strip().lower()
        if not msg_type:
            result = self._error_result("Message 'type' is required.")
            msg_type = "error"
        else:
            handler = self._ws_handlers.get(msg_type)
            if handler is None:
                result = self._error_result(f"Unsupported message type '{msg_type}'.")
                msg_type = "error"
            else:
                try:
                    result = handler(message)
                except Exception as exc:  # pragma: no cover - defensive guard for runtime errors
                    result = self._error_result(f"{msg_type} handler failed: {exc}", status_code=500)
                    msg_type = "error"
        return result.envelope(request_id=request_id, message_type=msg_type)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _send_simple(self, command: str, *, topic: str, verb: str) -> CommandDispatchResult:
        ok, reply = self._serial.send(command)
        payload = {
            "topic": topic,
            "command": verb,
            "serial_reply": reply,
        }
        return CommandDispatchResult(200 if ok else 502, payload)

    def _handle_stop_command(self) -> CommandDispatchResult:
        flushed = self._serial.flush_drive_queue("STOP command preemption")
        ok, reply = self._serial.send("STOP")
        if self._drive_heartbeat:
            self._drive_heartbeat.clear()
        payload = {
            "command": "STOP",
            "serial_reply": reply,
            "flushed_jobs": flushed,
        }
        return CommandDispatchResult(200 if ok else 502, payload)

    def _update_drive_heartbeat(self, command: str, status_code: int) -> None:
        if not self._drive_heartbeat or status_code >= 400:
            return
        if command == "STOP":
            self._drive_heartbeat.clear()
        else:
            self._drive_heartbeat.set_command(command)

    def _normalize_keyword(self, value: str) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip().upper()

    def _reduce_axes_to_command(self, x: float, y: float) -> str:
        x_adj = x if abs(x) >= self._deadzone else 0.0
        y_adj = y if abs(y) >= self._deadzone else 0.0
        if abs(y_adj) >= abs(x_adj) and y_adj > 0:
            return "FORWARD"
        if abs(y_adj) >= abs(x_adj) and y_adj < 0:
            return "BACKWARD"
        if x_adj > 0:
            return "RIGHT"
        if x_adj < 0:
            return "LEFT"
        return "STOP"

    def _coerce_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_positive_height(self, value: Any) -> float | None:
        parsed = self._coerce_float(value)
        if parsed is None or parsed <= 0:
            return None
        return parsed

    def _error_result(self, message: str, status_code: int = 400) -> CommandDispatchResult:
        return CommandDispatchResult(status_code, {"error": message})

    def _ws_drive(self, payload: dict[str, Any]) -> CommandDispatchResult:
        command = payload.get("command") or payload.get("direction") or payload.get("action")
        if isinstance(command, str) and command.strip():
            return self.execute_drive_command(command)

        axes = payload.get("axes")
        ax = ay = None
        if isinstance(axes, dict):
            ax = self._coerce_float(axes.get("x"))
            ay = self._coerce_float(axes.get("y"))
        if ax is None:
            ax = self._coerce_float(payload.get("x"))
        if ay is None:
            ay = self._coerce_float(payload.get("y"))

        if ax is None and ay is None:
            return self._error_result("Drive command requires 'command' or joystick axes (x/y).")

        return self.execute_drive_from_axes(ax or 0.0, ay or 0.0)

    def _ws_lift(self, payload: dict[str, Any]) -> CommandDispatchResult:
        verb = payload.get("command") or payload.get("action")
        if not isinstance(verb, str):
            return self._error_result("Lift messages must include a 'command'.")
        return self.execute_lift_command(verb)

    def _ws_fan(self, payload: dict[str, Any]) -> CommandDispatchResult:
        verb = payload.get("command") or payload.get("state")
        if not isinstance(verb, str):
            return self._error_result("Fan messages must include 'command' or 'state'.")
        return self.execute_fan_command(verb)

    def _ws_mode(self, payload: dict[str, Any]) -> CommandDispatchResult:
        action = str(payload.get("command") or payload.get("action") or "get").lower()
        if action in {"set", "update"}:
            return self.set_mode_response(payload.get("value"))
        return self.get_mode_response()

    def _ws_status(self, payload: dict[str, Any]) -> CommandDispatchResult:
        return self.status_probe()

    def _ws_ping(self, payload: dict[str, Any]) -> CommandDispatchResult:
        return CommandDispatchResult(
            200,
            {
                "message": "pong",
                "mode": self._mode.get_mode(),
            },
        )

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


def create_app(
    serial_controller: SerialController,
    frame_hub: FrameHub,
    mode_state: ModeState,
    db: DetectionDatabase,
    obstacle_height_state: ObstacleHeightState | None = None,
    drive_heartbeat: DriveHeartbeat | None = None,
) -> Flask:
    app = Flask(__name__)
    sock = Sock(app)
    dispatcher = ControlCommandDispatcher(
        serial_controller,
        mode_state,
        drive_heartbeat=drive_heartbeat,
        obstacle_height_state=obstacle_height_state,
    )

    def _json_response(result: CommandDispatchResult):
        return jsonify(result.payload), result.status_code

    @app.post("/auth/login")
    def auth_login():
        return jsonify(access_token="dev-token", token_type="bearer", expires_in=3600), 200

    @app.get("/status")
    def status():
        return _json_response(dispatcher.status_probe())

    def _drive_command_response(command: str):
        return _json_response(dispatcher.execute_drive_command(command))

    @app.post("/api/drive/forward")
    def api_drive_forward():
        return _drive_command_response("FORWARD")

    @app.post("/api/drive/reverse")
    def api_drive_reverse():
        return _drive_command_response("BACKWARD")

    @app.post("/api/drive/left")
    def api_drive_left():
        return _drive_command_response("LEFT")

    @app.post("/api/drive/right")
    def api_drive_right():
        return _drive_command_response("RIGHT")

    @app.post("/api/drive/stop")
    def api_drive_stop():
        return _drive_command_response("STOP")

    @app.post("/api/lift/up")
    def api_lift_up():
        return _json_response(dispatcher.execute_lift_command("UP"))

    @app.post("/api/lift/down")
    def api_lift_down():
        return _json_response(dispatcher.execute_lift_command("DOWN"))

    @app.post("/api/lift/stop")
    def api_lift_stop():
        return _json_response(dispatcher.execute_lift_command("STOP"))

    @app.post("/api/fan/on")
    def api_fan_on():
        return _json_response(dispatcher.execute_fan_command("ON"))

    @app.post("/api/fan/off")
    def api_fan_off():
        return _json_response(dispatcher.execute_fan_command("OFF"))

    @app.get("/api/pests")
    def api_list_pests():
        summaries = db.fetch_pest_summaries()
        return jsonify(
            {
                "pests": summaries,
                "total_tracked": len(summaries),
                "generated_at": utc_now_iso("seconds"),
            }
        )

    @app.get("/api/pests/<path:label>")
    def api_get_pest(label: str):
        normalized = label.strip()
        if not normalized:
            abort(400, description="Pest label cannot be empty.")
        summaries = db.fetch_pest_summaries(normalized)
        if not summaries:
            abort(404, description=f"No records found for '{normalized}'.")
        return jsonify(summaries[0])

    @app.get("/api/mode")
    def api_get_mode():
        return _json_response(dispatcher.get_mode_response())

    @app.post("/api/mode")
    def api_set_mode():
        data = request.get_json(silent=True) or {}
        requested_mode = data.get("mode")
        return _json_response(dispatcher.set_mode_response(requested_mode))

    @app.get("/api/obstacle/heights")
    def api_get_obstacle_heights():
        return _json_response(dispatcher.get_obstacle_heights())

    @app.post("/api/obstacle/heights")
    def api_set_obstacle_heights():
        data = request.get_json(silent=True) or {}
        lower = data.get("lower", data.get("lower_height"))
        upper = data.get("upper", data.get("upper_height"))
        return _json_response(dispatcher.set_obstacle_heights(lower, upper))

    @app.route("/video")
    def video():
        return Response(frame_hub.stream_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def index():
        return "<h1>Live Stream</h1><img src=\"/video\"/>"

    @sock.route("/ws/control")
    def control_websocket(ws):
        session_id = uuid.uuid4().hex[:10]
        print(f"[WS] session {session_id} connected")
        ws.send(
            json.dumps(
                {
                    "type": "welcome",
                    "session_id": session_id,
                    "mode": mode_state.get_mode(),
                    "status": "ok",
                    "timestamp": utc_now_iso("seconds"),
                }
            )
        )
        while True:
            try:
                raw_message = ws.receive()
            except ConnectionClosed:
                break
            if raw_message is None:
                break

            if isinstance(raw_message, bytes):
                raw_text = raw_message.decode("utf-8", errors="ignore")
            else:
                raw_text = raw_message

            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "status": "error",
                            "status_code": 400,
                            "error": f"Invalid JSON: {exc}",
                            "timestamp": utc_now_iso("seconds"),
                        }
                    )
                )
                continue

            if not isinstance(payload, dict):
                ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "status": "error",
                            "status_code": 400,
                            "error": "Payload must be a JSON object.",
                            "timestamp": utc_now_iso("seconds"),
                        }
                    )
                )
                continue

            response = dispatcher.dispatch_ws_message(payload)
            msg_type = str(payload.get("type") or "").strip().lower()
            if msg_type == "drive":
                command = response.get("command") or payload.get("command")
                status = response.get("status")
                status_code = response.get("status_code")
                serial_reply = response.get("serial_reply")
                print(
                    f"[WS] session {session_id} drive cmd={command} status={status} "
                    f"code={status_code} serial={serial_reply}"
                )
            ws.send(json.dumps(response))
        print(f"[WS] session {session_id} disconnected")

    return app


# ======================================================================================
# DepthAI Model + Pipeline Utilities
# ======================================================================================


RESULT_DIR = Path(os.environ.get("RESULT_DIR", "my_blobs/pestv5March"))
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


# === DATABASE SETUP ===
DB_PATH = os.environ.get("DB_PATH", "pest_results.db")


class DetectionDatabase:
    """Encapsulates SQLite initialization, writes, and summary queries."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    xmin INTEGER NOT NULL,
                    ymin INTEGER NOT NULL,
                    xmax INTEGER NOT NULL,
                    ymax INTEGER NOT NULL,
                    frame_w INTEGER NOT NULL,
                    frame_h INTEGER NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trap_sightings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    marker_id INTEGER NOT NULL,
                    trap_name TEXT,
                    location TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pest_summary (
                    label TEXT PRIMARY KEY,
                    detection_count INTEGER NOT NULL DEFAULT 0,
                    last_seen_utc TEXT NOT NULL,
                    last_seen_camera TEXT
                );
                """
            )
            conn.commit()
            print("[INFO] Database initialized successfully.")

    def open_writer(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def record_detection(
        self,
        conn: sqlite3.Connection,
        *,
        camera_name: str,
        label: str,
        confidence: float,
        bounds: tuple[int, int, int, int],
        frame_size: tuple[int, int],
    ) -> str:
        ts_utc = self._utc_now()
        xmin, ymin, xmax, ymax = bounds
        frame_w, frame_h = frame_size
        conn.execute(
            """
            INSERT INTO detections (
                ts_utc, camera_name, label, confidence, xmin, ymin, xmax, ymax, frame_w, frame_h
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts_utc, camera_name, label, confidence, xmin, ymin, xmax, ymax, frame_w, frame_h),
        )
        self._upsert_pest_summary(conn, label, ts_utc, camera_name)
        return ts_utc

    def record_trap_sighting(
        self,
        conn: sqlite3.Connection,
        *,
        camera_name: str,
        marker_id: int,
        trap_name: str,
        location: str | None,
    ) -> str:
        ts_utc = self._utc_now()
        conn.execute(
            """
            INSERT INTO trap_sightings (ts_utc, camera_name, marker_id, trap_name, location)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ts_utc, camera_name, marker_id, trap_name, location),
        )
        return ts_utc

    def fetch_pest_summaries(self, label: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT label, detection_count, last_seen_utc, last_seen_camera
            FROM pest_summary
        """
        params: tuple[Any, ...] = ()
        if label:
            query += " WHERE label = ?"
            params = (label,)
        query += " ORDER BY detection_count DESC, last_seen_utc DESC"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "label": row["label"],
                "count": int(row["detection_count"]),
                "last_seen_utc": row["last_seen_utc"],
                "last_seen_camera": row["last_seen_camera"],
            }
            for row in rows
        ]

    def maybe_commit(self, conn: sqlite3.Connection, last_commit_ts: float, interval_sec: float) -> float:
        now = time.time()
        if now - last_commit_ts >= interval_sec:
            conn.commit()
            return now
        return last_commit_ts

    def _upsert_pest_summary(
        self,
        conn: sqlite3.Connection,
        label: str,
        ts_utc: str,
        camera_name: str | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO pest_summary (label, detection_count, last_seen_utc, last_seen_camera)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(label) DO UPDATE SET
                detection_count = pest_summary.detection_count + 1,
                last_seen_utc = excluded.last_seen_utc,
                last_seen_camera = excluded.last_seen_camera;
            """,
            (label, ts_utc, camera_name),
        )

    @staticmethod
    def _utc_now() -> str:
        return utc_now_iso("milliseconds")


def partition_device_infos(
    all_devices: list[dai.DeviceInfo],
    pest_count: int = 2,
    obstacle_count: int = 2,
) -> tuple[list[dai.DeviceInfo], list[dai.DeviceInfo]]:
    """Split connected DepthAI devices between pest ID and obstacle detection roles."""
    pest_devices = list(all_devices[:pest_count])
    obstacle_devices = list(all_devices[pest_count : pest_count + obstacle_count])

    if len(pest_devices) < pest_count:
        print(
            f"[WARN] Requested {pest_count} pest-ID cameras but only {len(pest_devices)} device(s) available for that role."
        )
    if len(obstacle_devices) < obstacle_count:
        print(
            f"[WARN] Requested {obstacle_count} obstacle cameras but only {len(obstacle_devices)} device(s) allocated."
        )

    print(
        "[INFO] Camera allocation:"
        f" pest={len(pest_devices)}/{pest_count}"
        f" obstacle={len(obstacle_devices)}/{obstacle_count}"
    )

    return pest_devices, obstacle_devices


def start_pipeline(
    frame_hub: FrameHub,
    db: DetectionDatabase,
    device_infos_override: list[dai.DeviceInfo] | None = None,
) -> None:
    db.initialize()
    conn = db.open_writer()
    last_commit = time.time()

    if DEFAULT_MODEL_BLOB is None:
        raise RuntimeError("DEFAULT_MODEL_BLOB is not set; ensure RESULT_DIR points to a valid export.")

    model_blob_path = str(DEFAULT_MODEL_BLOB)

    available_devices = device_infos_override if device_infos_override is not None else dai.Device.getAllAvailableDevices()
    if not available_devices:
        raise RuntimeError("[ERROR] No DepthAI devices detected.")

    requested_camera_count = len(available_devices)
    camera_count_env = os.environ.get("PIPELINE_CAMERA_COUNT")
    if camera_count_env:
        try:
            requested_camera_count = max(1, int(camera_count_env))
        except ValueError:
            print(
                f"[WARN] Invalid PIPELINE_CAMERA_COUNT='{camera_count_env}'. Defaulting to {len(available_devices)} devices."
            )
            requested_camera_count = len(available_devices)

    if requested_camera_count > len(available_devices):
        print(
            f"[WARN] Requested {requested_camera_count} camera(s) but only {len(available_devices)} device(s) detected. "
            "Proceeding with available devices."
        )

    camera_names = {
        0: "camera_1_left",
        1: "camera_2_right",
    }
    active_camera_count = min(requested_camera_count, len(available_devices))
    camera_setups = [
        CameraSetup(name=camera_names.get(idx, f"camera_{idx + 1}"), blob_path=model_blob_path)
        for idx in range(active_camera_count)
    ]

    pipeline_bundles: list[PipelineBundle] = []
    for setup in camera_setups:
        bundle = build_pipeline(setup)
        pipeline_bundles.append(bundle)

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

                    db.record_detection(
                        conn,
                        camera_name=active["name"],
                        label=label,
                        confidence=float(confidence),
                        bounds=(x1, y1, x2, y2),
                        frame_size=(frame.shape[1], frame.shape[0]),
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
                    for trap_det in trap_detections:
                        db.record_trap_sighting(
                            conn,
                            camera_name=active["name"],
                            marker_id=trap_det["marker_id"],
                            trap_name=trap_det["trap_name"],
                            location=trap_det["location"],
                        )

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
                    print()
                    #print("[ACTION] Turn Off systems")
                    #print(f"[METRIC] Unique traps seen so far: {len(unique_traps_seen)}")

            # COMMIT DATABASE EVERY 5 SECONDS
            previous_commit = last_commit
            last_commit = db.maybe_commit(conn, last_commit, interval_sec=5.0)
            if last_commit != previous_commit:
                print("[DEBUG] Database committed.")

    conn.close()



# ======================================================================================
# Obstacle Detection Thread
# ======================================================================================


def run_obstacle_detection(
    serial_controller: SerialController | None,
    device_infos_override: list[dai.DeviceInfo] | None = None,
    mode_state: ModeState | None = None,
    frame_hub: FrameHub | None = None,
    height_state: ObstacleHeightState | None = None,
) -> None:
    """Run the provided obstacle detection loop alongside pest-identification cameras and publish frames."""
    # Hardware Communication Protocols
    SERIAL_PORT_OBST = SERIAL_PORT
    BAUD_RATE = SERIAL_BAUD

    if serial_controller is not None:
        esp32 = SerialWriterAdapter(serial_controller)
    else:
        try:
            esp32 = serial.Serial(SERIAL_PORT_OBST, BAUD_RATE, timeout=0.1)
            time.sleep(2)
        except serial.SerialException as e:
            print(f"Warning: Serial initialization failed: {e}")
            esp32 = None

    FPS_LIMIT = 30
    should_quit = False

    # System Parameters & Constraints
    SAFE_DISTANCE_MM = 600
    MIN_CONTOUR_AREA = 500
    MISSING_LINE_THRESHOLD = 30  # Frame threshold to trigger navigation transition

    OBST_ROI_W, OBST_ROI_H = 200, 100
    OBST_ROI_X = (640 - OBST_ROI_W) // 2
    OBST_ROI_Y = 0

    LINE_ROI_W, LINE_ROI_H = 200, 200
    LINE_ROI_X, LINE_ROI_Y = (640 - LINE_ROI_W) // 2, 280

    # --- FINITE STATE MACHINE VARIABLES ---
    STATE_ROW_OUTWARD = 0
    STATE_ROW_RETURN = 1
    STATE_AISLE_TRANSIT = 2

    current_nav_state = STATE_ROW_OUTWARD

    if height_state is not None:
        lower_height, higher_height, height_version = height_state.get_range()
    else:
        lower_height = DEFAULT_LOWER_OBSTACLE_HEIGHT_FT
        higher_height = DEFAULT_UPPER_OBSTACLE_HEIGHT_FT
        height_version = 0

    current_height = 0.0
    target_height = lower_height
    height_tolerance = 0.1
    is_adjusting_height = False
    last_auto_cmd = b""
    last_auto_cmd_time = 0.0

    height_feedback_available = bool(esp32) and hasattr(esp32, "readline")

    def _target_for_state(state: int) -> float:
        return higher_height if state == STATE_ROW_RETURN else lower_height

    def _set_target_height(desired: float, auto_adjust: bool = True) -> None:
        nonlocal target_height, is_adjusting_height
        target_height = desired
        if not esp32 or not height_feedback_available:
            return
        if auto_adjust and abs(current_height - target_height) > height_tolerance:
            is_adjusting_height = True

    def _sync_height_targets(force: bool = False) -> None:
        nonlocal lower_height, higher_height, height_version
        if height_state is None:
            return
        latest_lower, latest_upper, latest_version = height_state.get_range()
        if not force and latest_version == height_version:
            return
        lower_height = latest_lower
        higher_height = latest_upper
        height_version = latest_version
        _set_target_height(_target_for_state(current_nav_state), auto_adjust=True)

    if height_state is not None:
        _sync_height_targets(force=True)
    else:
        _set_target_height(lower_height, auto_adjust=True)

    # DepthAI Multi-Device Configuration
    device_infos = device_infos_override if device_infos_override is not None else dai.Device.getAllAvailableDevices()
    num_cams = len(device_infos)
    print(f"Initialized hub. Devices detected: {num_cams}")

    if num_cams == 0:
        raise RuntimeError("Hardware Error: Zero OAK-D devices enumerated.")

    camera_labels = [f"obstacle_cam_{idx}" for idx in range(num_cams)]
    if num_cams >= 2:
        camera_labels[0] = "obstacle_rear"
        camera_labels[1] = "obstacle_front"
    elif num_cams == 1:
        camera_labels[0] = "obstacle_front"

    active_cam_idx = 1 if num_cams > 1 else 0
    missing_line_frames = 0
    last_drive_command: bytes | None = None

    # Context manager for safe multi-device USB allocation
    with ExitStack() as stack:
        rgb_qs = []
        depth_qs = []

        # Initialize connected DepthAI devices
        for i, info in enumerate(device_infos):
            pipeline = dai.Pipeline()

            # Configure stereo and RGB nodes
            left = pipeline.create(dai.node.MonoCamera)
            left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            left.setFps(30)

            right = pipeline.create(dai.node.MonoCamera)
            right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            right.setFps(30)

            rgb_cam = pipeline.create(dai.node.ColorCamera)
            rgb_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            rgb_cam.setPreviewSize(640, 480)
            rgb_cam.setInterleaved(False)
            rgb_cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(False)
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

            left.out.link(stereo.left)
            right.out.link(stereo.right)

            rgb_xout = pipeline.create(dai.node.XLinkOut)
            depth_xout = pipeline.create(dai.node.XLinkOut)
            rgb_stream_name = f"obstacle_rgb_{i}"
            depth_stream_name = f"obstacle_depth_{i}"
            rgb_xout.setStreamName(rgb_stream_name)
            depth_xout.setStreamName(depth_stream_name)

            rgb_cam.preview.link(rgb_xout.input)
            stereo.depth.link(depth_xout.input)

            device = stack.enter_context(dai.Device(pipeline, info))

            # Instantiate device-scoped non-blocking queues prior to pipeline execution
            rgb_qs.append(device.getOutputQueue(rgb_stream_name, maxSize=4, blocking=False))
            depth_qs.append(device.getOutputQueue(depth_stream_name, maxSize=4, blocking=False))

            print(f"Pipeline active for Device {i} [ID: {info.getDeviceId()}]")
            time.sleep(0.2)  # Hardware stabilization delay

        print("\nSystem active. Awaiting user interrupt (q).")
        last_frame_time = time.time()

        while not should_quit:
            is_autonomous = mode_state.is_autonomous() if mode_state is not None else True
            if not is_autonomous and esp32 and last_drive_command != b"STOP\n":
                esp32.write(b"STOP\n")
                last_drive_command = b"STOP\n"

            current_time = time.time()
            if current_time - last_frame_time < (1.0 / FPS_LIMIT):
                time.sleep(0.001)
                continue
            last_frame_time = current_time

            _sync_height_targets()

            # 1. READ LIDAR DATA (Non-blocking)
            if esp32 and height_feedback_available:
                reads = 0
                while True:
                    try:
                        waiting = esp32.in_waiting
                    except Exception:
                        break
                    if waiting <= 0 or reads > 10:
                        break
                    try:
                        line = esp32.readline().decode("utf-8", errors="ignore").strip()
                    except Exception:
                        break
                    reads += 1
                    if not line:
                        continue
                    if "LIDAR Distance:" in line:
                        try:
                            raw_lidar_feet = float(line.split(":", 1)[1].strip())
                            current_height = raw_lidar_feet + 1.5  # 1.5ft offset
                        except (ValueError, IndexError):
                            continue

            # Clear buffer queues across all devices to prevent USB overflow
            for i in range(num_cams):
                in_depth = depth_qs[i].tryGet()
                in_rgb = rgb_qs[i].tryGet()
                rgb_frame = in_rgb.getCvFrame() if in_rgb is not None else None

                # Isolate computer vision processing to the active camera stream
                if i == active_cam_idx and in_depth is not None and rgb_frame is not None:
                    depth_frame = in_depth.getFrame()

                    # Depth Processing & Obstacle Detection
                    obst_roi = depth_frame[OBST_ROI_Y : OBST_ROI_Y + OBST_ROI_H, OBST_ROI_X : OBST_ROI_X + OBST_ROI_W]
                    valid_depths = obst_roi[obst_roi > 0]
                    distance = int(np.percentile(valid_depths, 25)) if valid_depths.size else 9999

                    color_status = (0, 0, 255) if (0 < distance < SAFE_DISTANCE_MM) else (0, 255, 0)
                    cv2.rectangle(
                        rgb_frame,
                        (OBST_ROI_X, OBST_ROI_Y),
                        (OBST_ROI_X + OBST_ROI_W, OBST_ROI_Y + OBST_ROI_H),
                        color_status,
                        2,
                    )
                    cv2.putText(
                        rgb_frame,
                        f"Dist: {distance}mm",
                        (OBST_ROI_X, OBST_ROI_Y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color_status,
                        2,
                    )

                    # Vision Processing & Trajectory Calculation
                    line_roi_slice = rgb_frame[LINE_ROI_Y : LINE_ROI_Y + LINE_ROI_H, LINE_ROI_X : LINE_ROI_X + LINE_ROI_W]
                    blurred_roi = cv2.GaussianBlur(line_roi_slice, (9, 9), 0)
                    hsv_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)

                    if current_nav_state in (STATE_ROW_OUTWARD, STATE_ROW_RETURN):
                        lower_color_1 = np.array([0, 60, 60])
                        upper_color_1 = np.array([10, 255, 255])
                        mask1 = cv2.inRange(hsv_roi, lower_color_1, upper_color_1)

                        lower_color_2 = np.array([160, 60, 60])
                        upper_color_2 = np.array([180, 255, 255])
                        mask2 = cv2.inRange(hsv_roi, lower_color_2, upper_color_2)
                        thresh = cv2.bitwise_or(mask1, mask2)
                        target_color_text = "Target: RED"
                    else:
                        lower_green = np.array([40, 60, 60])
                        upper_green = np.array([90, 255, 255])
                        thresh = cv2.inRange(hsv_roi, lower_green, upper_green)
                        target_color_text = "Target: GREEN"

                    kernel = np.ones((5, 5), np.uint8)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                    # Diagnostic output window
                    cv2.imshow("Binary Mask", thresh)

                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    line_detected = False
                    error = 0
                    angle = 0
                    cy = LINE_ROI_H // 2
                    box_center_x = LINE_ROI_W // 2

                    cv2.rectangle(
                        rgb_frame,
                        (LINE_ROI_X, LINE_ROI_Y),
                        (LINE_ROI_X + LINE_ROI_W, LINE_ROI_Y + LINE_ROI_H),
                        (0, 255, 255),
                        2,
                    )

                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(c) > MIN_CONTOUR_AREA:
                            M = cv2.moments(c)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                error = cx - box_center_x

                                topmost = tuple(c[c[:, :, 1].argmin()][0])
                                bottommost = tuple(c[c[:, :, 1].argmax()][0])
                                dx = topmost[0] - bottommost[0]
                                dy = bottommost[1] - topmost[1]
                                if dy == 0:
                                    dy = 1
                                angle = int(np.degrees(np.arctan2(dx, dy)))
                                line_detected = True
                                cv2.circle(
                                    rgb_frame,
                                    (cx + LINE_ROI_X, cy + LINE_ROI_Y),
                                    5,
                                    (0, 0, 255),
                                    -1,
                                )
                                cv2.line(
                                    rgb_frame,
                                    (bottommost[0] + LINE_ROI_X, bottommost[1] + LINE_ROI_Y),
                                    (topmost[0] + LINE_ROI_X, topmost[1] + LINE_ROI_Y),
                                    (0, 255, 0),
                                    3,
                                )

                    # State Machine & Actuation Logic
                    command_to_send = b""
                    status = ""

                    # Hardware index mapping
                    FRONT_CAM_INDEX = 1 if num_cams > 1 else 0
                    REAR_CAM_INDEX = 0

                    if is_adjusting_height and esp32 and height_feedback_available:
                        status = f"Lifting to {target_height:.1f}ft..."
                        candidate_cmd = b""
                        if current_height < target_height - height_tolerance:
                            candidate_cmd = b"UP\n"
                        elif current_height > target_height + height_tolerance:
                            candidate_cmd = b"DOWN\n"
                        else:
                            print(f"\n[SYSTEM] Height {target_height:.1f}ft reached. Resuming driving.")
                            command_to_send = b"STOP\n"
                            is_adjusting_height = False
                            last_auto_cmd = b""
                            missing_line_frames = 0

                        if candidate_cmd and command_to_send == b"":
                            if last_auto_cmd != candidate_cmd or (current_time - last_auto_cmd_time > 0.2):
                                command_to_send = candidate_cmd
                                last_auto_cmd = candidate_cmd
                                last_auto_cmd_time = current_time

                    elif is_adjusting_height and not height_feedback_available:
                        status = "Height adjust unavailable (no feedback)"
                        is_adjusting_height = False

                    elif 0 < distance < SAFE_DISTANCE_MM:
                        status = "STOP! OBSTACLE"
                        command_to_send = b"STOP\n"

                    elif line_detected:
                        missing_line_frames = 0

                        throttle_speed = int(np.interp(cy, [0, LINE_ROI_H], [127, 40]))
                        if esp32 and is_autonomous and not is_adjusting_height:
                            esp32.write(f"SPD:{throttle_speed}\n".encode())

                        if active_cam_idx == FRONT_CAM_INDEX:
                            if error < -100:
                                status, command_to_send = "Edge LEFT", b"LEFT\n"
                            elif error > 100:
                                status, command_to_send = "Edge RIGHT", b"RIGHT\n"
                            elif error < -50 or angle < -50:
                                status, command_to_send = "Tight Arc L", b"TIGHT_ARC_LEFT\n"
                            elif error > 50 or angle > 50:
                                status, command_to_send = "Tight Arc R", b"TIGHT_ARC_RIGHT\n"
                            elif error < -15 or angle < -15:
                                status, command_to_send = "Arc LEFT", b"ARC_LEFT\n"
                            elif error > 15 or angle > 15:
                                status, command_to_send = "Arc RIGHT", b"ARC_RIGHT\n"
                            else:
                                status, command_to_send = "FORWARD", b"FORWARD\n"
                        elif active_cam_idx == REAR_CAM_INDEX:
                            if error < -100:
                                status, command_to_send = "Edge REV L", b"LEFT\n"
                            elif error > 100:
                                status, command_to_send = "Edge REV R", b"RIGHT\n"
                            elif error < -50 or angle < -50:
                                status, command_to_send = "Tight Arc L", b"TIGHT_ARC_REV_RIGHT\n"
                            elif error > 50 or angle > 50:
                                status, command_to_send = "Tight Arc R", b"TIGHT_ARC_REV_LEFT\n"
                            elif error < -15 or angle < -15:
                                status, command_to_send = "Arc REV L", b"ARC_REV_RIGHT\n"
                            elif error > 15 or angle > 15:
                                status, command_to_send = "Arc REV R", b"ARC_REV_LEFT\n"
                            else:
                                status, command_to_send = "BACKWARD", b"BACKWARD\n"

                    else:
                        # Handle trajectory loss and execute failover sequence
                        missing_line_frames += 1
                        status = f"Searching... ({missing_line_frames}/{MISSING_LINE_THRESHOLD})"
                        command_to_send = b"STOP\n"

                        if missing_line_frames >= MISSING_LINE_THRESHOLD:
                            missing_line_frames = 0
                            if current_nav_state == STATE_ROW_OUTWARD:
                                if num_cams > 1:
                                    current_nav_state = STATE_ROW_RETURN
                                    active_cam_idx = REAR_CAM_INDEX
                                    _set_target_height(higher_height, auto_adjust=True)
                                    print("\n[SYSTEM] End of row. Adjusting to HIGHER height. Active feed: REAR CAM (RED TAPE)")
                                else:
                                    status = "END OF TAPE. STOPPED."
                            elif current_nav_state == STATE_ROW_RETURN:
                                current_nav_state = STATE_AISLE_TRANSIT
                                active_cam_idx = FRONT_CAM_INDEX
                                _set_target_height(lower_height, auto_adjust=True)
                                print("\n[SYSTEM] Row complete. Adjusting to LOWER height. Entering aisle. Active feed: FRONT CAM (GREEN TAPE)")
                            elif current_nav_state == STATE_AISLE_TRANSIT:
                                current_nav_state = STATE_ROW_OUTWARD
                                active_cam_idx = FRONT_CAM_INDEX
                                _set_target_height(lower_height, auto_adjust=True)
                                print("\n[SYSTEM] Arrived at new row. Active feed: FRONT CAM (RED TAPE)")
                            time.sleep(0.5)

                    if esp32 and is_autonomous and command_to_send:
                        esp32.write(command_to_send)
                        last_drive_command = command_to_send

                    # Render active camera telemetry
                    cam_label = camera_labels[active_cam_idx].replace("_", " ").upper()
                    cv2.putText(rgb_frame, cam_label, (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                    cv2.putText(rgb_frame, target_color_text, (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(
                        rgb_frame,
                        f"Height: {current_height:.1f}ft / Target: {target_height:.1f}ft",
                        (300, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )
                    cv2.putText(rgb_frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    cv2.imshow("Robot View", rgb_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        should_quit = True

                    if frame_hub is not None:
                        frame_hub.update(camera_labels[i], rgb_frame)
                elif rgb_frame is not None and frame_hub is not None:
                    frame_hub.update(camera_labels[i], rgb_frame)

    # System Teardown
    cv2.destroyAllWindows()
    if esp32:
        print("\nInitiating safe shutdown sequence...")
        esp32.write(b"STOP\n")
        time.sleep(0.1)
        esp32.close()
        print("Actuators disengaged. Offline.")


# ======================================================================================
# Application bootstrap
# ======================================================================================

SERIAL_CONTROLLER = SerialController(
    port=SERIAL_PORT,
    baud=SERIAL_BAUD,
    timeout_sec=SERIAL_TIMEOUT_SEC,
)
FRAME_HUB = FrameHub()
MODE_STATE = ModeState()
OBSTACLE_HEIGHTS = ObstacleHeightState(
    DEFAULT_LOWER_OBSTACLE_HEIGHT_FT,
    DEFAULT_UPPER_OBSTACLE_HEIGHT_FT,
)
DETECTION_DB = DetectionDatabase(DB_PATH)
DRIVE_HEARTBEAT = DriveHeartbeat(SERIAL_CONTROLLER, DRIVE_HEARTBEAT_INTERVAL_SEC)
app = create_app(
    SERIAL_CONTROLLER,
    FRAME_HUB,
    MODE_STATE,
    DETECTION_DB,
    OBSTACLE_HEIGHTS,
    DRIVE_HEARTBEAT,
)


def main() -> None:
    all_device_infos = dai.Device.getAllAvailableDevices()
    default_pest = len(all_device_infos) if all_device_infos else 1
    pest_count = _read_int_env("PEST_CAMERA_COUNT", default_pest, minimum=0)
    obstacle_count = _read_int_env("OBSTACLE_CAMERA_COUNT", 0, minimum=0)
    if pest_count == 0 and obstacle_count == 0:
        pest_count = default_pest
    pest_devices, obstacle_devices = partition_device_infos(all_device_infos, pest_count, obstacle_count)

    if pest_devices:
        pipeline_thread = threading.Thread(
            target=start_pipeline,
            args=(FRAME_HUB, DETECTION_DB, pest_devices),
            daemon=True,
        )
        pipeline_thread.start()
    else:
        print("[WARN] Skipping pest identification pipeline due to missing devices.")

    if obstacle_devices:
        if not MODE_STATE.is_autonomous():
            print("[INFO] Obstacle cameras active in passive mode until autonomous mode is enabled.")
        obstacle_thread = threading.Thread(
            target=run_obstacle_detection,
            args=(SERIAL_CONTROLLER, obstacle_devices, MODE_STATE, FRAME_HUB, OBSTACLE_HEIGHTS),
            daemon=True,
        )
        obstacle_thread.start()
    else:
        print("[WARN] Skipping obstacle detection due to missing dedicated devices.")

    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
