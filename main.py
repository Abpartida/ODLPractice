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
from flask import Flask, Response, abort, jsonify, render_template, request
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
STREAM_CAMERA_FEED_DEFAULT = _env_flag("STREAM_CAMERA_FEED", "1")

# Shared flag so other subsystems (e.g., WebSocket handlers) know when the
# obstacle loop actively manages actuators.
OBSTACLE_THREAD_ACTIVE = threading.Event()


class ObstaclePauseController:
    """Tracks timed pause requests for the obstacle detection loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._resume_at = 0.0

    def pause_for(self, duration_sec: float) -> None:
        if duration_sec <= 0:
            return
        resume_at = time.monotonic() + duration_sec
        with self._lock:
            if resume_at > self._resume_at:
                self._resume_at = resume_at

    def remaining(self) -> float:
        with self._lock:
            remaining = self._resume_at - time.monotonic()
        return remaining if remaining > 0 else 0.0

    def is_paused(self) -> bool:
        return self.remaining() > 0.0


OBSTACLE_PAUSE_CTRL = ObstaclePauseController()


def utc_now_iso(timespec: str = "seconds") -> str:
    """Return an ISO 8601 UTC timestamp with a 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat(timespec=timespec).replace("+00:00", "Z")

# Default actuator commands (override via env vars if firmware differs)
GUI_AVAILABLE = bool(os.environ.get("DISPLAY")) and os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen"
FAN_ON_COMMAND = os.environ.get("FAN_ON_COMMAND", "FAN")
FAN_OFF_COMMAND = os.environ.get("FAN_OFF_COMMAND", "FAN_OFF")
LIFT_UP_COMMAND = os.environ.get("LIFT_UP_COMMAND", "UP")
LIFT_DOWN_COMMAND = os.environ.get("LIFT_DOWN_COMMAND", "DOWN")
LIFT_STOP_COMMAND = os.environ.get("LIFT_STOP_COMMAND", "STOP")

SerialResult = tuple[bool, str]

LIFT_SPEED_FT_PER_SEC = _read_float_env("LIFT_SPEED_FT_PER_SEC", 0.8)
LIFT_MOVE_MIN_SEC = _read_float_env("LIFT_MOVE_MIN_SEC", 0.4)
LIFT_MOVE_MAX_SEC = _read_float_env("LIFT_MOVE_MAX_SEC", 4.0)


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
        self._rx_thread: threading.Thread | None = None
        self._rx_stop_event = threading.Event()
        self._rx_queue: "Queue[bytes]" = Queue(maxsize=256)

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
            self._ensure_rx_thread_started_unlocked()
        except SerialException as exc:
            self._serial = None
            print(f"[ERROR] Failed to open serial {self.port}: {exc}")

    def _reset_serial_input_unlocked(self) -> None:
        serial_obj = self._serial
        if not serial_obj or not serial_obj.is_open:
            return
        try:
            serial_obj.reset_input_buffer()
        except Exception:
            pass

    def reset_input_buffer(self) -> None:
        """Public hook so other components can flush stale serial data."""
        with self._serial_lock:
            self._reset_serial_input_unlocked()
        while not self._rx_queue.empty():
            try:
                self._rx_queue.get_nowait()
            except Empty:
                break

    def readline(self) -> bytes:
        try:
            return self._rx_queue.get_nowait()
        except Empty:
            return b""

    @property
    def in_waiting(self) -> int:
        return self._rx_queue.qsize()

    def _ensure_rx_thread_started_unlocked(self) -> None:
        if self._rx_thread and self._rx_thread.is_alive():
            return
        self._rx_stop_event.clear()
        self._rx_thread = threading.Thread(target=self._rx_worker_loop, name="serial-rx", daemon=True)
        self._rx_thread.start()

    def _rx_worker_loop(self) -> None:
        while not self._rx_stop_event.is_set():
            with self._serial_lock:
                serial_obj = self._serial if self._serial and self._serial.is_open else None
            if not serial_obj:
                time.sleep(0.1)
                continue
            try:
                line = serial_obj.readline()
            except (SerialException, OSError):
                time.sleep(0.1)
                continue
            if not line:
                time.sleep(0.01)
                continue
            decoded = line.decode("utf-8", errors="ignore").strip()
            if decoded:
                print(f"[ESP32] {decoded}")
            try:
                self._rx_queue.put_nowait(line)
            except Full:
                try:
                    self._rx_queue.get_nowait()
                except Empty:
                    pass
                try:
                    self._rx_queue.put_nowait(line)
                except Full:
                    pass


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

    def readline(self) -> bytes:
        if not self._controller:
            return b""
        return self._controller.readline()

    @property
    def in_waiting(self) -> int:
        if not self._controller:
            return 0
        return self._controller.in_waiting

    def reset_input_buffer(self) -> None:  # pragma: no cover - compatibility shim
        if self._controller:
            self._controller.reset_input_buffer()

    def close(self) -> None:  # pragma: no cover - noop shim
        return


class ObstacleHeightState:
    """Tracks upper/lower canopy heights supplied by the app."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lower_ft: float | None = None
        self._upper_ft: float | None = None

    def set_heights(self, lower_ft: float, upper_ft: float) -> None:
        with self._lock:
            self._lower_ft = lower_ft
            self._upper_ft = upper_ft

    def clear(self) -> None:
        with self._lock:
            self._lower_ft = None
            self._upper_ft = None

    def get_heights(self) -> tuple[float, float] | None:
        with self._lock:
            if self._lower_ft is None or self._upper_ft is None:
                return None
            return self._lower_ft, self._upper_ft

    def has_heights(self) -> bool:
        return self.get_heights() is not None


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

    _DRIVE_COMMANDS = {
        "FORWARD",
        "BACKWARD",
        "LEFT",
        "RIGHT",
        "STOP",
        "ARC_LEFT",
        "ARC_RIGHT",
        "ARC_REV_LEFT",
        "ARC_REV_RIGHT",
        "TIGHT_ARC_LEFT",
        "TIGHT_ARC_RIGHT",
        "TIGHT_ARC_REV_LEFT",
        "TIGHT_ARC_REV_RIGHT",
    }

    def __init__(
        self,
        serial_controller: SerialController,
        mode_state: ModeState,
        joystick_deadzone: float = JOYSTICK_DEADZONE,
        height_state: ObstacleHeightState | None = None,
        autonomy_guard: threading.Event | None = None,
    ) -> None:
        self._serial = serial_controller
        self._mode = mode_state
        self._deadzone = joystick_deadzone
        self._height_state = height_state
        self._autonomy_guard = autonomy_guard
        self._ws_handlers: dict[str, Callable[[dict[str, Any]], CommandDispatchResult]] = {
            "drive": self._ws_drive,
            "lift": self._ws_lift,
            "fan": self._ws_fan,
            "mode": self._ws_mode,
            "status": self._ws_status,
            "ping": self._ws_ping,
            "obstacle_heights": self._ws_obstacle_heights,
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

    def get_mode_response(self) -> CommandDispatchResult:
        return CommandDispatchResult(200, {"mode": self._mode.get_mode()})

    def set_mode_response(self, value: str) -> CommandDispatchResult:
        if not isinstance(value, str):
            return self._error_result("Mode must be provided as a string.")
        try:
            normalized = ModeState._normalize(value)
        except ValueError as exc:
            return self._error_result(str(exc))
        if (
            normalized == ModeState.AUTONOMOUS
            and self._height_state
            and not self._height_state.has_heights()
        ):
            return self._error_result(
                "Cannot enter autonomous mode until upper/lower obstacle heights are configured.",
                status_code=409,
            )
        changed = self._mode.set_mode(normalized)
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

    def get_obstacle_heights(self) -> CommandDispatchResult:
        if not self._height_state:
            return self._error_result("Obstacle height tracking unavailable.", status_code=503)
        heights = self._height_state.get_heights()
        payload: dict[str, Any] = {"heights_set": bool(heights)}
        if heights:
            payload["lower_ft"], payload["upper_ft"] = heights
        return CommandDispatchResult(200, payload)

    def set_obstacle_heights(self, lower: Any, upper: Any) -> CommandDispatchResult:
        if not self._height_state:
            return self._error_result("Obstacle height tracking unavailable.", status_code=503)
        lower_ft = self._coerce_float(lower)
        upper_ft = self._coerce_float(upper)
        if lower_ft is None or upper_ft is None:
            return self._error_result("Both lower and upper heights must be numeric.")
        if lower_ft <= 0 or upper_ft <= 0:
            return self._error_result("Heights must be positive values.")
        if lower_ft >= upper_ft:
            return self._error_result("Upper height must be greater than lower height.")
        self._height_state.set_heights(lower_ft, upper_ft)
        return CommandDispatchResult(
            200,
            {
                "lower_ft": lower_ft,
                "upper_ft": upper_ft,
                "heights_set": True,
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
        payload = {
            "command": "STOP",
            "serial_reply": reply,
            "flushed_jobs": flushed,
        }
        return CommandDispatchResult(200 if ok else 502, payload)

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

    def _error_result(self, message: str, status_code: int = 400) -> CommandDispatchResult:
        return CommandDispatchResult(status_code, {"error": message})

    def _ws_manual_override_guard(self, topic: str) -> CommandDispatchResult | None:
        guard = self._autonomy_guard
        if guard and guard.is_set() and self._mode.is_autonomous():
            return CommandDispatchResult(
                423,
                {
                    "error": f"{topic} overrides are disabled while obstacle navigation is active in autonomous mode.",
                    "mode": self._mode.get_mode(),
                    "obstacle_thread_active": True,
                },
            )
        return None

    def _ws_drive(self, payload: dict[str, Any]) -> CommandDispatchResult:
        blocked = self._ws_manual_override_guard("drive")
        if blocked:
            command = payload.get("command") or payload.get("direction") or payload.get("action")
            if command:
                blocked.payload.setdefault("command", command)
            axes = payload.get("axes")
            if axes:
                blocked.payload.setdefault("joystick", axes)
            return blocked
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
        blocked = self._ws_manual_override_guard("lift")
        if blocked:
            verb_candidate = payload.get("command") or payload.get("action")
            if verb_candidate:
                blocked.payload.setdefault("command", verb_candidate)
            return blocked
        verb = payload.get("command") or payload.get("action")
        if not isinstance(verb, str):
            return self._error_result("Lift messages must include a 'command'.")
        return self.execute_lift_command(verb)

    def _ws_fan(self, payload: dict[str, Any]) -> CommandDispatchResult:
        blocked = self._ws_manual_override_guard("fan")
        if blocked:
            verb_candidate = payload.get("command") or payload.get("state")
            if verb_candidate:
                blocked.payload.setdefault("command", verb_candidate)
            return blocked
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

    def _ws_obstacle_heights(self, payload: dict[str, Any]) -> CommandDispatchResult:
        action = str(payload.get("action") or payload.get("mode") or "get").lower()
        if action in {"set", "update"}:
            return self.set_obstacle_heights(payload.get("lower"), payload.get("upper"))
        return self.get_obstacle_heights()

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

    def drop(self, camera_name: str) -> None:
        with self._lock:
            self._frames.pop(camera_name, None)

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
# Stream Control Helpers
# ======================================================================================


class StreamGate:
    """Thread-safe gate that controls whether MJPEG streaming is allowed."""

    def __init__(self, enabled: bool):
        self._enabled = bool(enabled)
        self._lock = threading.Lock()

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set(self, enabled: bool) -> bool:
        with self._lock:
            self._enabled = bool(enabled)
            return self._enabled

    def enable(self) -> bool:
        return self.set(True)

    def disable(self) -> bool:
        return self.set(False)

    def toggle(self) -> bool:
        with self._lock:
            self._enabled = not self._enabled
            return self._enabled


# ======================================================================================
# Flask Application
# ======================================================================================


def create_app(
    serial_controller: SerialController,
    frame_hub: FrameHub,
    stream_gate: StreamGate,
    mode_state: ModeState,
    db: DetectionDatabase,
    obstacle_heights: ObstacleHeightState | None = None,
    obstacle_guard: threading.Event | None = None,
) -> Flask:
    app = Flask(__name__)
    sock = Sock(app)
    dispatcher = ControlCommandDispatcher(
        serial_controller,
        mode_state,
        height_state=obstacle_heights,
        autonomy_guard=obstacle_guard,
    )

    def _json_response(result: CommandDispatchResult):
        return jsonify(result.payload), result.status_code

    def _stream_state_payload() -> dict[str, Any]:
        return {
            "enabled": stream_gate.is_enabled(),
            "timestamp": utc_now_iso("seconds"),
        }

    def _parse_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return None

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
        lower = data.get("lower") or data.get("lower_height")
        upper = data.get("upper") or data.get("upper_height")
        return _json_response(dispatcher.set_obstacle_heights(lower, upper))

    @app.route("/video")
    def video():
        if not stream_gate.is_enabled():
            abort(404, description="Camera streaming disabled by configuration.")

        def _stream_generator():
            for payload in frame_hub.stream_frames():
                if not stream_gate.is_enabled():
                    break
                yield payload

        return Response(_stream_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/")
    def index():
        stream_enabled = stream_gate.is_enabled()
        stream_url = "/video"
        return render_template(
            "live_stream.html",
            title="LYCO TOMI Live Stream",
            stream_url=stream_url,
            stream_enabled=stream_enabled,
            stream_control_url="/api/stream/state",
        )

    @app.get("/api/stream/state")
    def api_stream_get_state():
        return jsonify(_stream_state_payload())

    @app.post("/api/stream/state")
    def api_stream_update_state():
        data = request.get_json(silent=True) or {}
        desired = _parse_bool(data.get("enabled"))
        action = str(data.get("action") or data.get("command") or data.get("state") or "").strip().lower()

        if desired is not None:
            stream_gate.set(desired)
        elif action in {"pause", "disable", "off", "stop"}:
            stream_gate.disable()
        elif action in {"resume", "enable", "on", "start"}:
            stream_gate.enable()
        elif action == "toggle" or not action:
            stream_gate.toggle()

        return jsonify(_stream_state_payload())

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
    0: {"name": "Trap A", "location": "First block, row 1"},
    1: {"name": "Trap B", "location": "Second block, row 1"},
    2: {"name": "Trap C", "location": "Third block, row 1"},
    4: {"name": "Trap D", "location": "Fourth block, row 1"},
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
    # Obstacle cameras claim the earliest enumerated devices so their placement is deterministic.
    obstacle_devices = list(all_devices[:obstacle_count])
    pest_devices = list(all_devices[obstacle_count : obstacle_count + pest_count])

    if len(obstacle_devices) < obstacle_count:
        print(
            f"[WARN] Requested {obstacle_count} obstacle cameras but only {len(obstacle_devices)} device(s) allocated."
        )
    if len(pest_devices) < pest_count:
        print(
            f"[WARN] Requested {pest_count} pest-ID cameras but only {len(pest_devices)} device(s) available for that role."
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
                    OBSTACLE_PAUSE_CTRL.pause_for(5.0)
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


def _run_obstacle_detection_thread(
    serial_controller: SerialController | None,
    device_infos_override: list[dai.DeviceInfo] | None = None,
    mode_state: ModeState | None = None,
    frame_hub: FrameHub | None = None,
    height_state: ObstacleHeightState | None = None,
) -> None:
    """Run the provided obstacle detection loop alongside pest-identification cameras and publish frames."""
    SERIAL_PORT_OBST = SERIAL_PORT
    BAUD_RATE = SERIAL_BAUD

    if serial_controller is not None:
        esp32 = SerialWriterAdapter(serial_controller)
    else:
        try:
            esp32 = serial.Serial(SERIAL_PORT_OBST, BAUD_RATE, timeout=0.1)
            time.sleep(2)
        except serial.SerialException as exc:
            print(f"Warning: Serial initialization failed: {exc}")
            esp32 = None

    FPS_LIMIT = 30
    should_quit = False

    SAFE_DISTANCE_MM = 1800
    MIN_CONTOUR_AREA = 500
    MISSING_LINE_THRESHOLD = 30

    OBST_ROI_W, OBST_ROI_H = 200, 100
    OBST_ROI_X = (640 - OBST_ROI_W) // 2
    OBST_ROI_Y = 0

    LINE_ROI_W, LINE_ROI_H = 200, 200
    LINE_ROI_X, LINE_ROI_Y = (640 - LINE_ROI_W) // 2, 280

    STATE_ROW_OUTWARD = 0
    STATE_ROW_RETURN = 1
    STATE_AISLE_TRANSIT = 2
    current_nav_state = STATE_ROW_OUTWARD
    TURN_GREEN_LIMIT = 3
    FINAL_STOP_HEIGHT_FT = 2.0
    turn_green_counter = 0
    awaiting_green_stop = False
    final_green_stop_latched = False
    green_detection_latched = False

    device_infos = device_infos_override if device_infos_override is not None else dai.Device.getAllAvailableDevices()
    if len(device_infos) > 2:
        device_infos = device_infos[:2]
    num_cams = len(device_infos)
    print(f"Initialized hub. Navigation Devices detected: {num_cams}")

    if num_cams == 0:
        raise RuntimeError("Hardware Error: Zero OAK-D devices enumerated.")

    camera_labels = [f"obstacle_cam_{idx}" for idx in range(num_cams)]
    if num_cams >= 2:
        camera_labels[0] = "obstacle_rear"
        camera_labels[1] = "obstacle_front"
    else:
        camera_labels[0] = "obstacle_front"

    FRONT_CAM_INDEX = 1 if num_cams > 1 else 0
    REAR_CAM_INDEX = 0
    active_cam_idx = FRONT_CAM_INDEX
    missing_line_frames = 0

    def reset_green_turn_tracking() -> None:
        nonlocal turn_green_counter, awaiting_green_stop, final_green_stop_latched, green_detection_latched
        turn_green_counter = 0
        awaiting_green_stop = False
        final_green_stop_latched = False
        green_detection_latched = False

    def format_turn_counter() -> str:
        count = min(turn_green_counter, TURN_GREEN_LIMIT)
        label = f"Turns: {count}/{TURN_GREEN_LIMIT}"
        if final_green_stop_latched:
            label += " (complete)"
        elif awaiting_green_stop:
            label += " (await green)"
        return label

    lift_timer_lock = threading.Lock()
    lift_stop_timer: threading.Timer | None = None
    last_lift_command: str | None = None
    last_lift_cmd_time = 0.0
    lidar_available = hasattr(esp32, "in_waiting") and hasattr(esp32, "readline")
    last_drive_command: bytes | None = None

    DEFAULT_LOWER_FT = 1.5
    DEFAULT_UPPER_FT = 3.0
    lower_height_ft = DEFAULT_LOWER_FT
    upper_height_ft = DEFAULT_UPPER_FT
    target_height_ft: float | None = None
    current_height_ft: float | None = None
    estimated_height_ft: float | None = None
    height_tolerance_ft = 0.1
    last_height_pair_ft: tuple[float, float] | None = None
    is_adjusting_height = False
    height_adjust_pending = False

    def heights_ready() -> bool:
        if height_state is None:
            return True
        return height_state.has_heights()

    def actuation_allowed() -> bool:
        if not esp32:
            return False
        if mode_state is not None and not mode_state.is_autonomous():
            return False
        return heights_ready()

    def send_drive_command(command: bytes, *, force: bool = False) -> None:
        nonlocal last_drive_command
        if not force and not actuation_allowed():
            last_drive_command = None
            return
        if not force and command == last_drive_command:
            return
        try:
            esp32.write(command)
            last_drive_command = command
        except Exception as exc:
            print(f"[WARN] Failed to send drive command: {exc}")

    def _write_ascii_command(command: str) -> None:
        if not esp32:
            return
        try:
            payload = (command.strip() + "\n").encode("utf-8")
            esp32.write(payload)
        except Exception as exc:
            print(f"[WARN] Failed to send lift command '{command}': {exc}")

    def _cancel_lift_timer() -> None:
        nonlocal lift_stop_timer
        with lift_timer_lock:
            if lift_stop_timer is not None:
                lift_stop_timer.cancel()
                lift_stop_timer = None

    def _schedule_lift_stop(delay_sec: float) -> None:
        nonlocal lift_stop_timer
        if delay_sec <= 0:
            _write_ascii_command(LIFT_STOP_COMMAND)
            return

        def _stop_lift() -> None:
            _write_ascii_command(LIFT_STOP_COMMAND)
            with lift_timer_lock:
                lift_stop_timer = None

        timer = threading.Timer(delay_sec, _stop_lift)
        timer.daemon = True
        with lift_timer_lock:
            lift_stop_timer = timer
        timer.start()

    def _send_lift_command(cmd: str, min_interval_sec: float = 0.2) -> None:
        nonlocal last_lift_command, last_lift_cmd_time
        now = time.time()
        if last_lift_command == cmd:
            if cmd == LIFT_STOP_COMMAND:
                return
            if (now - last_lift_cmd_time) < min_interval_sec:
                return
        _write_ascii_command(cmd)
        last_lift_command = cmd
        last_lift_cmd_time = now

    def _begin_height_adjustment(new_target_ft: float, *, immediate: bool = True) -> None:
        nonlocal target_height_ft, is_adjusting_height, last_lift_command, last_lift_cmd_time, height_adjust_pending
        target_height_ft = new_target_ft
        last_lift_command = None
        last_lift_cmd_time = 0.0
        height_adjust_pending = True
        if immediate and actuation_allowed():
            is_adjusting_height = bool(esp32)
            height_adjust_pending = False

    def _update_height_targets_from_state() -> None:
        nonlocal lower_height_ft, upper_height_ft, last_height_pair_ft, estimated_height_ft, target_height_ft
        heights_ft = height_state.get_heights() if height_state else None
        if heights_ft:
            lower_ft, upper_ft = heights_ft
        else:
            lower_ft = DEFAULT_LOWER_FT
            upper_ft = DEFAULT_UPPER_FT
        pair = (round(lower_ft, 4), round(upper_ft, 4))
        if last_height_pair_ft != pair:
            lower_height_ft, upper_height_ft = lower_ft, upper_ft
            last_height_pair_ft = pair
            if estimated_height_ft is None:
                estimated_height_ft = lower_height_ft
            _begin_height_adjustment(lower_height_ft, immediate=False)

    _update_height_targets_from_state()

    with ExitStack() as stack:
        rgb_qs: list[dai.DataOutputQueue] = []
        depth_qs: list[dai.DataOutputQueue] = []

        for i, info in enumerate(device_infos):
            pipeline = dai.Pipeline()

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
            stereo.setOutputSize(640, 480)
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

            rgb_qs.append(device.getOutputQueue(rgb_stream_name, maxSize=4, blocking=False))
            depth_qs.append(device.getOutputQueue(depth_stream_name, maxSize=4, blocking=False))

            try:
                device_id = info.getMxId()
            except AttributeError:
                device_id = getattr(info, "mxid", "unknown")
            print(f"Pipeline active for Device {i} [ID: {device_id}]")
            time.sleep(0.2)

        print("\nSystem active. Awaiting user interrupt (q).")
        last_frame_time = time.time()
        stop_latched = False
        nav_enabled_prev = False

        last_distance_mm = 9999

        while not should_quit:
            heights_ready_now = heights_ready()
            autonomous_mode = mode_state.is_autonomous() if mode_state is not None else True
            nav_enabled = heights_ready_now and autonomous_mode
            nav_wait_reason = None
            if not heights_ready_now:
                nav_wait_reason = "Waiting for height input"
            elif not autonomous_mode:
                nav_wait_reason = "Waiting for autonomous mode"

            if not autonomous_mode:
                reset_green_turn_tracking()

            pause_remaining = OBSTACLE_PAUSE_CTRL.remaining()
            if pause_remaining > 0:
                nav_enabled = False
                nav_wait_reason = f"Paused for traps ({pause_remaining:.1f}s)"

            if nav_enabled and not nav_enabled_prev and target_height_ft is not None:
                is_adjusting_height = bool(esp32)
                height_adjust_pending = False

            allowed_now = actuation_allowed()
            if not nav_enabled:
                is_adjusting_height = False

            if not nav_enabled:
                if esp32 and not stop_latched:
                    send_drive_command(b"STOP\n", force=True)
                    stop_latched = True
                _cancel_lift_timer()
                _send_lift_command(LIFT_STOP_COMMAND)
            else:
                stop_latched = False

            _update_height_targets_from_state()
            if height_adjust_pending and nav_enabled and target_height_ft is not None:
                is_adjusting_height = bool(esp32)
                height_adjust_pending = False

            if lidar_available and esp32:
                try:
                    while getattr(esp32, "in_waiting", 0) > 0:
                        raw_line = esp32.readline().decode("utf-8", errors="ignore").strip()
                        if not raw_line:
                            continue
                        if "LIDAR Distance:" in raw_line:
                            try:
                                raw_feet = float(raw_line.split(":")[1].strip())
                                current_height_ft = raw_feet + 1.5
                            except ValueError:
                                continue
                except Exception:
                    pass

            current_time = time.time()
            if current_time - last_frame_time < (1.0 / FPS_LIMIT):
                time.sleep(0.001)
                continue
            last_frame_time = current_time

            for i in range(num_cams):
                in_depth = depth_qs[i].tryGet()
                in_rgb = rgb_qs[i].tryGet()
                rgb_frame = in_rgb.getCvFrame() if in_rgb is not None else None

                if i == active_cam_idx and rgb_frame is not None:
                    target_color_text = (
                        "Target: RED" if current_nav_state in (STATE_ROW_OUTWARD, STATE_ROW_RETURN) else "Target: GREEN"
                    )
                    if not nav_enabled:
                        status = nav_wait_reason or "Obstacle loop idle"
                        heights_tuple = height_state.get_heights() if height_state else None
                        if heights_tuple:
                            height_text = f"Heights: {heights_tuple[0]:.1f}-{heights_tuple[1]:.1f}ft"
                        else:
                            height_text = "Heights: --"
                        cam_label = "FRONT CAM" if active_cam_idx == FRONT_CAM_INDEX else "REAR CAM"
                        cv2.putText(rgb_frame, cam_label, (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                        cv2.putText(rgb_frame, target_color_text, (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(rgb_frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        cv2.putText(rgb_frame, height_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(rgb_frame, format_turn_counter(), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (135, 206, 250), 2)
                        if GUI_AVAILABLE:
                            cv2.imshow("Robot View", rgb_frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                should_quit = True
                        if frame_hub is not None:
                            frame_hub.update(camera_labels[i], rgb_frame)
                        continue

                    depth_frame = in_depth.getFrame() if in_depth is not None else None

                    distance = last_distance_mm
                    if depth_frame is not None:
                        obst_roi = depth_frame[OBST_ROI_Y : OBST_ROI_Y + OBST_ROI_H, OBST_ROI_X : OBST_ROI_X + OBST_ROI_W]
                        valid_depths = obst_roi[obst_roi > 0]
                        distance = int(np.percentile(valid_depths, 25)) if valid_depths.size else 9999
                        last_distance_mm = distance

                    color_status = (0, 0, 255) if (0 < distance < SAFE_DISTANCE_MM) else (0, 255, 0)
                    cv2.rectangle(
                        rgb_frame,
                        (OBST_ROI_X, OBST_ROI_Y),
                        (OBST_ROI_X + OBST_ROI_W, OBST_ROI_Y + OBST_ROI_H),
                        color_status,
                        2,
                    )
                    dist_label = f"Dist: {distance}mm" if depth_frame is not None else f"Dist: ~{distance}mm (last)"
                    cv2.putText(
                        rgb_frame,
                        dist_label,
                        (OBST_ROI_X, OBST_ROI_Y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color_status,
                        2,
                    )

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

                    if GUI_AVAILABLE:
                        cv2.imshow("Binary Mask", thresh)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    line_detected = False
                    error = 0
                    angle = 0
                    box_center_x = LINE_ROI_W // 2
                    cx = cy = 0

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

                                cv2.circle(rgb_frame, (cx + LINE_ROI_X, cy + LINE_ROI_Y), 5, (0, 0, 255), -1)
                                cv2.line(
                                    rgb_frame,
                                    (bottommost[0] + LINE_ROI_X, bottommost[1] + LINE_ROI_Y),
                                    (topmost[0] + LINE_ROI_X, topmost[1] + LINE_ROI_Y),
                                    (0, 255, 0),
                                    3,
                                )

                    final_stop_triggered = False
                    green_line_seen = (
                        nav_enabled
                        and autonomous_mode
                        and current_nav_state == STATE_AISLE_TRANSIT
                        and line_detected
                        and not final_green_stop_latched
                    )
                    if green_line_seen and not green_detection_latched:
                        green_detection_latched = True
                        turn_green_counter += 1
                        if turn_green_counter >= TURN_GREEN_LIMIT and not final_green_stop_latched:
                            if awaiting_green_stop:
                                awaiting_green_stop = False
                                final_green_stop_latched = True
                                final_stop_triggered = True
                            else:
                                awaiting_green_stop = True
                    elif not green_line_seen:
                        green_detection_latched = False

                    nav_enabled_prev = nav_enabled

                    command_to_send = b"STOP\n"
                    status = ""
                    lift_status: str | None = None

                    adjusting_now = bool(
                        is_adjusting_height and actuation_allowed() and target_height_ft is not None
                    )

                    if adjusting_now:
                        tgt_ft = target_height_ft if target_height_ft is not None else lower_height_ft
                        lift_status = f"Lifting to {tgt_ft:.2f}ft"
                        command_to_send = b"STOP\n"
                        if lidar_available and current_height_ft is not None:
                            if current_height_ft < tgt_ft - height_tolerance_ft:
                                _send_lift_command(LIFT_UP_COMMAND)
                            elif current_height_ft > tgt_ft + height_tolerance_ft:
                                _send_lift_command(LIFT_DOWN_COMMAND)
                            else:
                                _send_lift_command(LIFT_STOP_COMMAND)
                                is_adjusting_height = False
                                estimated_height_ft = tgt_ft
                        else:
                            if estimated_height_ft is None:
                                estimated_height_ft = lower_height_ft
                            distance_ft = abs(tgt_ft - estimated_height_ft)
                            if distance_ft <= height_tolerance_ft:
                                _send_lift_command(LIFT_STOP_COMMAND)
                                is_adjusting_height = False
                                estimated_height_ft = tgt_ft
                            else:
                                speed_ft = max(LIFT_SPEED_FT_PER_SEC, 0.01)
                                duration_sec = distance_ft / speed_ft
                                duration_sec = max(LIFT_MOVE_MIN_SEC, min(LIFT_MOVE_MAX_SEC, duration_sec))
                                direction_cmd = LIFT_UP_COMMAND if tgt_ft > estimated_height_ft else LIFT_DOWN_COMMAND
                                _cancel_lift_timer()
                                _write_ascii_command(direction_cmd)
                                _schedule_lift_stop(duration_sec)
                                estimated_height_ft = tgt_ft
                                is_adjusting_height = False
                                lift_status = f"Lifting to {tgt_ft:.2f}ft (timed)"
                    else:
                        if not is_adjusting_height:
                            _send_lift_command(LIFT_STOP_COMMAND)

                        if 0 < distance < SAFE_DISTANCE_MM:
                            status = "STOP! OBSTACLE"
                            command_to_send = b"STOP\n"

                        elif line_detected:
                            missing_line_frames = 0

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
                            else:
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
                            missing_line_frames += 1
                            status = f"Searching... ({missing_line_frames}/{MISSING_LINE_THRESHOLD})"
                            command_to_send = b"STOP\n"

                            if missing_line_frames >= MISSING_LINE_THRESHOLD:
                                missing_line_frames = 0

                                if current_nav_state == STATE_ROW_OUTWARD:
                                    if num_cams > 1:
                                        current_nav_state = STATE_ROW_RETURN
                                        active_cam_idx = REAR_CAM_INDEX
                                        _begin_height_adjustment(upper_height_ft)
                                        print("\n[SYSTEM] End of row. Reversing. Active feed: REAR CAM (RED TAPE)")
                                    else:
                                        status = "END OF TAPE. STOPPED."
                                elif current_nav_state == STATE_ROW_RETURN:
                                    current_nav_state = STATE_AISLE_TRANSIT
                                    active_cam_idx = FRONT_CAM_INDEX
                                    _begin_height_adjustment(lower_height_ft)
                                    print("\n[SYSTEM] Row complete. Entering aisle. Active feed: FRONT CAM (GREEN TAPE)")
                                else:
                                    current_nav_state = STATE_ROW_OUTWARD
                                    active_cam_idx = FRONT_CAM_INDEX
                                    print("\n[SYSTEM] Arrived at new row. Active feed: FRONT CAM (RED TAPE)")

                                time.sleep(0.5)

                    if final_stop_triggered:
                        status = "Stopping on green target"
                        command_to_send = b"STOP\n"
                        lift_status = f"Lowering to {FINAL_STOP_HEIGHT_FT:.2f}ft"
                        _begin_height_adjustment(FINAL_STOP_HEIGHT_FT)
                    elif final_green_stop_latched:
                        command_to_send = b"STOP\n"
                        if not status:
                            status = "Final green reached"

                    send_drive_command(command_to_send)

                    if lift_status is None and not actuation_allowed():
                        lift_status = "Waiting for autonomy/heights..."

                    heights_tuple = height_state.get_heights() if height_state else None
                    height_text = (
                        f"Heights: {heights_tuple[0]:.1f}-{heights_tuple[1]:.1f}ft"
                        if heights_tuple
                        else "Heights: --"
                    )
                    if current_height_ft is not None:
                        height_text = f"Height: {current_height_ft:.1f}ft"
                    elif estimated_height_ft is not None:
                        height_text = f"Height: ~{estimated_height_ft:.1f}ft"
                    cam_label = "FRONT CAM" if active_cam_idx == FRONT_CAM_INDEX else "REAR CAM"
                    cv2.putText(rgb_frame, cam_label, (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                    cv2.putText(rgb_frame, target_color_text, (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(rgb_frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    if lift_status:
                        cv2.putText(rgb_frame, lift_status, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        cv2.putText(rgb_frame, height_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    else:
                        cv2.putText(rgb_frame, height_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(rgb_frame, format_turn_counter(), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (135, 206, 250), 2)

                    if GUI_AVAILABLE:
                        cv2.imshow("Robot View", rgb_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            should_quit = True

                    if frame_hub is not None:
                        frame_hub.update(camera_labels[i], rgb_frame)
                        for j, label in enumerate(camera_labels):
                            if j != active_cam_idx:
                                frame_hub.drop(label)
                else:
                    pass

    _cancel_lift_timer()
    if GUI_AVAILABLE:
        cv2.destroyAllWindows()
    if esp32:
        print("\nInitiating safe shutdown sequence...")
        try:
            send_drive_command(b"STOP\n", force=True)
        except Exception:
            pass
        time.sleep(0.1)
        try:
            esp32.close()
        except Exception:
            pass
        print("Actuators disengaged. Offline.")


def run_obstacle_detection(
    serial_controller: SerialController | None,
    device_infos_override: list[dai.DeviceInfo] | None = None,
    mode_state: ModeState | None = None,
    frame_hub: FrameHub | None = None,
    height_state: ObstacleHeightState | None = None,
) -> None:
    """Wrapper that marks the obstacle loop as active for manual override guards."""
    OBSTACLE_THREAD_ACTIVE.set()
    try:
        _run_obstacle_detection_thread(
            serial_controller,
            device_infos_override,
            mode_state,
            frame_hub,
            height_state,
        )
    finally:
        OBSTACLE_THREAD_ACTIVE.clear()


# ======================================================================================
# Application bootstrap
# ======================================================================================

SERIAL_CONTROLLER = SerialController(
    port=SERIAL_PORT,
    baud=SERIAL_BAUD,
    timeout_sec=SERIAL_TIMEOUT_SEC,
)
FRAME_HUB = FrameHub()
STREAM_GATE = StreamGate(STREAM_CAMERA_FEED_DEFAULT)
MODE_STATE = ModeState()
OBSTACLE_HEIGHTS = ObstacleHeightState()
DETECTION_DB = DetectionDatabase(DB_PATH)
app = create_app(
    SERIAL_CONTROLLER,
    FRAME_HUB,
    STREAM_GATE,
    MODE_STATE,
    DETECTION_DB,
    obstacle_heights=OBSTACLE_HEIGHTS,
    obstacle_guard=OBSTACLE_THREAD_ACTIVE,
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
