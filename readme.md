# ODL Practice Guide

A multi-camera DepthAI pipeline that tracks greenhouse pests, overlays ArUco trap positions, proxies detections into SQLite, and publishes a lightweight MJPEG stream + HTTP API. The same process also proxies manual/joystick drive commands from a web or mobile client to the Sabertooth-based rover controller.

## Quick Start

### Option A: `run_main.sh` (recommended)
1. Make sure the DepthAI cameras and the Arduino Nano ESP32 (running `ManualControl.ino`) are connected via USB.
2. Run `chmod +x run_main.sh` once, then launch with `./run_main.sh`.
3. The script will
   - locate (or create) the `venv310` virtual environment,
   - install/refresh `requirements.txt` when needed,
   - start `main.py` inside that venv, and
   - open your browser to `http://127.0.0.1:5000/` so you can see the 2×2 camera grid immediately.
4. Pass any `main.py` CLI arguments after `--`, e.g. `./run_main.sh -- --mock-cameras`.

You can override defaults through environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PYTHON_BIN` | `/opt/homebrew/bin/python3.10` | Base interpreter used to build the venv (falls back to `python3.10` → `python3`). |
| `VENV_DIR` | `<repo>/venv310` | Location of the auto-managed virtual environment. |
| `FORCE_PIP_SYNC` | `0` | Set to `1` to reinstall `requirements.txt` even if nothing changed. |
| `OPEN_BROWSER` | `1` | Set to `0` to skip auto-opening the MJPEG viewer. |
| `BROWSER_URL` | `http://127.0.0.1:5000/` | Page opened after the app starts. |
| `BROWSER_CMD` | *(empty)* | Optional custom browser command (e.g. `BROWSER_CMD="/Applications/Firefox.app/Contents/MacOS/firefox"`). |
| `STREAM_CAMERA_FEED` | `1` | Set to `0` to keep the DepthAI pipelines running but disable the `/video` MJPEG endpoint. |
| `VENV_DIR/.deps-installed` | *(file)* | Timestamp marker; delete it to force a dependency refresh. |

### Option B: manual setup
```bash
/opt/homebrew/bin/python3.10 -m venv venv310
source venv310/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```
> DepthAI only ships prebuilt wheels up through Python 3.11, so stay on 3.10 or 3.11 unless you want to compile blob dependencies from source.

## Python dependencies
`requirements.txt` intentionally stays short so the venv install is fast:

- `depthai` – camera pipeline + YOLO runtime.
- `opencv-contrib-python` – image transforms + ArUco detection.
- `numpy` – lightweight frame math.
- `Flask` – MJPEG streamer + REST endpoints for cameras, pests, and drive control.
- `Flask-Sock` – upgrades the control surface to a WebSocket so long-lived clients (OkHttp, browser UIs) can stream commands.
- `pyserial` – bridges joystick/actuator commands from HTTP into the Nano ESP32 firmware.

## DepthAI cameras and model blobs
- By default `RESULT_DIR=my_blobs/pestv5March` and `main.py` auto-picks the `*best*.blob` + its `.json`. Override with the `RESULT_DIR` env var or point individual `CameraSetup` entries at other blobs.
- `CameraSetup` (near the top of `main.py`) lists every OAK-D device you expect; the script automatically matches connected hardware. Missing devices simply log a warning and the rest of the cameras still come up.
- `NN_INPUT_SIZE` or `MODEL_INPUT_SIZE` env vars override the blob-implied preview resolution if you export with something non-square.
- Camera allocation is now explicit. The launcher enumerates the connected OAK-D devices once and splits them into pest-identification vs. navigation pools. Use `PEST_CAMERA_COUNT` (defaults to "all available" unless you also set `OBSTACLE_CAMERA_COUNT`) and `OBSTACLE_CAMERA_COUNT` (default `0`) to pin how many boards each loop may claim. Example: `PEST_CAMERA_COUNT=2 OBSTACLE_CAMERA_COUNT=2 python main.py` reserves the first two MXIDs for the pest YOLO pipeline and the next two for the obstacle thread.

## Autonomous obstacle navigation
When `OBSTACLE_CAMERA_COUNT` is greater than zero, `main.py` launches a second DepthAI pipeline that treats its dedicated OAK-D pair as a front/rear navigation rig. The loop:
- streams RGB + aligned depth into the `/video` mosaic with labels such as `obstacle_front` / `obstacle_rear`,
- maintains a three-state line-following controller (row outward -> row return -> aisle transit) that swaps between the forward and rear cameras automatically, and
- feeds the Sabertooth drive queue with `FORWARD`, `ARC_*`, `TIGHT_ARC_*`, `BACKWARD`, or `STOP` based on the lane error, rotation of the red/green tape, and a 650 mm safety box rendered over the depth ROI.

The navigation loop only actuates when **both** conditions are true:
1. `/api/mode` or the WebSocket `{"type":"mode","command":"set","value":"autonomous"}` succeeds.
2. Canopy heights have been provided (see below). Until then the thread will keep publishing camera frames but enforces `STOP` + `LIFT_STOP`.

### Height gating + automatic lift control
The robot tracks lower/upper canopy heights so it knows which vertical span to target as it leaves a row, reverses, or transits the aisle. Defaults are 1.5 ft (lower) and 3.0 ft (upper); the controller also drops to 2.0 ft when it has counted the configured number of green lane markers at the end of a route.

- REST: `GET /api/obstacle/heights` reports whether heights are latched, and `POST /api/obstacle/heights {"lower":1.6,"upper":3.1}` updates the pair.
- WebSocket: send `{"type":"obstacle_heights","action":"set","lower":1.6,"upper":3.1}` or `{"type":"obstacle_heights","action":"get"}`.
- Mode changes to `autonomous` will fail with HTTP 409 until both values are positive and `upper > lower`.
- While the loop is active in autonomous mode it queues lift commands (using either live lidar feedback from the ESP32 or the timed fallback controlled by `LIFT_SPEED_FT_PER_SEC`, `LIFT_MOVE_MIN_SEC`, and `LIFT_MOVE_MAX_SEC`) whenever it needs to transition between height bands or finish the final stop.

### Trap-aware navigation pauses
The pest cameras gate pauses on ArUco trap markers. When at least `ARUCO_PAUSE_MIN_COUNT` valid markers are visible (defaults to `1`), the rover pauses autonomous drive commands for `ARUCO_PAUSE_DURATION_SEC` (defaults to `3` seconds). Marker validation uses the same contour fit from the pest tagging pipeline, plus an area floor derived from `ARUCO_MARKER_SIZE_MM`/`ARUCO_MIN_AREA_RATIO`, so noisy squares no longer trip the pause.

### Navigation-specific environment toggles
| Variable | Default | Purpose |
| --- | --- | --- |
| `PEST_CAMERA_COUNT` | *(all devices unless both counts are zero)* | Number of OAK-D units reserved for YOLO/trap detection. |
| `OBSTACLE_CAMERA_COUNT` | `0` | Number of OAK-D units reserved for the navigation loop (0 disables it). |
| `ARUCO_MARKER_SIZE_MM` | `25.0` | Physical width of the printed trap markers; used to derive a sane area floor. |
| `ARUCO_MIN_AREA_RATIO` | *(auto from size)* | Override the minimum portion of the frame a marker must cover. |
| `ARUCO_PAUSE_MIN_COUNT` | `1` | Number of simultaneously visible markers required to pause drive commands. |
| `ARUCO_PAUSE_DURATION_SEC` | `3.0` | How long to hold autonomous motion after a marker-triggered pause. |
| `LIFT_SPEED_FT_PER_SEC` | `0.8` | Estimated lift speed used when lidar feedback is unavailable. |
| `LIFT_MOVE_MIN_SEC` / `LIFT_MOVE_MAX_SEC` | `0.4` / `4.0` | Bounds for timed lift moves when running open-loop. |

## Browser stream + REST API
Running `main.py` exposes both the MJPEG composite and JSON endpoints on `http://<host>:5000`.

### Live stream
- `GET /video` – multipart MJPEG with the latest 2×2 grid of cameras.
- `GET /` – minimal HTML wrapper that just embeds `/video` and exposes a Pause/Resume control.
- `GET /api/stream/state` – returns `{enabled: bool, timestamp: ...}` so dashboards can poll the streaming state.
- `POST /api/stream/state` – accepts `{"enabled": true|false}` or `{"action": "pause|resume|toggle"}` to control the live feed. The Pause button on `/` calls this endpoint so operators can stop streaming without killing the camera threads.

### Drive + actuator control
Manual drive commands now travel over a persistent WebSocket: `ws://<host>:5000/ws/control`. Every JSON message requires a `type` field so the dispatcher knows which subsystem to target, and the server replies with `status`, the raw `serial_reply`, and the current rover `mode`.

Common payloads:

```json
{"id":"demo-forward","type":"drive","command":"FORWARD"}
{"id":"axes-1","type":"drive","x":0.15,"y":0.9}
{"type":"lift","command":"up"}
{"type":"fan","command":"off"}
{"type":"mode","command":"set","value":"autonomous"}
{"type":"mode","command":"get"}
{"type":"status"}
{"type":"ping"}
```

- `type: "drive"` accepts either a discrete `command` (`forward`, `backward`, `left`, `right`, `stop`) or joystick axes (`x` + `y`) that get quantized using the `JOYSTICK_DEADZONE` env var (default `0.3`).
- `type: "lift"` maps `up`/`down`/`stop` into the `LIFT_*` serial verbs.
- `type: "fan"` maps `on`/`off` into `FAN_ON_COMMAND`/`FAN_OFF_COMMAND`.
- `type: "mode"` supports `{"command":"get"}` and `{"command":"set","value":"manual|autonomous"}`.
- `type: "status"` probes `STAT?` on the Sabertooth and returns the latest rover mode.
- `type: "obstacle_heights"` mirrors the REST helper; use `{"action":"set","lower":1.6,"upper":3.1}` to seed the canopy span before toggling autonomous mode.
- While the obstacle navigation thread is running **and** the robot is in autonomous mode, drive/lift/fan overrides return HTTP 423 / WebSocket errors so operators do not fight the planner. Drop back to manual mode (or launch without `OBSTACLE_CAMERA_COUNT`) before issuing manual moves.

Legacy REST endpoints like `POST /api/drive/forward`, `/api/lift/up`, `/api/fan/on`, and `POST /api/mode` are still exposed for compatibility, but new tooling should prefer the WebSocket channel to avoid request-per-command latency spikes.

### OkHttp control console
Need a quick manual driver? The `app` module now includes a lightweight console that speaks the WebSocket protocol via OkHttp's `WebSocket` API. Launch it from the repo root with:

```bash
./gradlew :app:run --args="ws://127.0.0.1:5000/ws/control"
```

or export `CONTROL_WS_URL=ws://robot.local:5000/ws/control` and run `./gradlew :app:run`.

Once connected you can type commands such as `drive forward`, `drive axes 0.2 0.9`, `lift down`, `fan on`, `mode set autonomous`, `status`, `ping`, `help`, or `quit`. Every server response is echoed with the JSON envelope so it's easy to see the serial acknowledgements.

### Pest summaries + status
- `GET /api/pests` – ordered summary of `{label, count, last_seen_utc, last_seen_camera}` pulled from SQLite.
- `GET /api/pests/<label>` – detail view for a single pest label (404 if nothing recorded).
- `GET /status` – quick health ping that also probes `STAT?` on the Sabertooth controller.

## Serial / Manual drive stack
- Flash `ManualControl.ino` onto the Arduino Nano ESP32 that sits between the Raspberry Pi (USB) and the pair of Sabertooth drivers. The firmware already understands `DRIVE`, `SPEED`, `W`, `M`, `LIFT`, `FAN`, and `STAT?` commands—see the top-of-file comment for the whole protocol.
- `main.py` queues drive commands through `SerialController` so HTTP requests never block the camera threads. Customize ports/baud at launch using:
  - `SERIAL_PORT` (default `/dev/ttyACM0`),
  - `SERIAL_BAUD` (default `115200`),
  - `SERIAL_TIMEOUT_SEC`, and
  - `DRIVE_QUEUE_SIZE`, `DRIVE_HANDLER_DEADLINE_SEC`, `DRIVE_QUEUE_WAIT_FALLBACK_SEC` if you need to tune responsiveness.
- The `ModeState` helper lets the mobile UI block autonomous commands while a person is actively using manual drive.

## Database + telemetry artifacts
- SQLite lives at `pest_results.db` (override with `DB_PATH`). Deletions are safe; the schema (`detections`, `trap_sightings`, `pest_summary`) is recreated automatically on startup.
- Detections and trap sightings are timestamped in UTC, so correlating them with greenhouse events is straightforward.
- Raw JPEG frames aren’t persisted; if you need historical media, extend `DetectionDatabase.record_detection` to copy frames into `vid_result/` or S3.

## Troubleshooting
- **“DepthAI not detected”** – ensure each OAK-D has its own USB port; the script logs a warning and continues, but no frames will be streamed for the missing cameras.
- **“Serial not connected”** – check `SERIAL_PORT` and confirm the Nano shows up under `/dev`. You can run `python main.py --no-serial` after temporarily editing `SerialController` if you want to test the vision stack alone.
- **Browser not opening** – set `OPEN_BROWSER=0` in CI or provide a `BROWSER_CMD` if you’re on Linux without `open`.
