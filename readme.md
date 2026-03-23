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
- `pyserial` – bridges joystick/actuator commands from HTTP into the Nano ESP32 firmware.

## DepthAI cameras and model blobs
- By default `RESULT_DIR=my_blobs/pestv5March` and `main.py` auto-picks the `*best*.blob` + its `.json`. Override with the `RESULT_DIR` env var or point individual `CameraSetup` entries at other blobs.
- `CameraSetup` (near the top of `main.py`) lists every OAK-D device you expect; the script automatically matches connected hardware. Missing devices simply log a warning and the rest of the cameras still come up.
- `NN_INPUT_SIZE` or `MODEL_INPUT_SIZE` env vars override the blob-implied preview resolution if you export with something non-square.

## Browser stream + REST API
Running `main.py` exposes both the MJPEG composite and JSON endpoints on `http://<host>:5000`.

### Live stream
- `GET /video` – multipart MJPEG with the latest 2×2 grid of cameras.
- `GET /` – minimal HTML wrapper that just embeds `/video`.

### Drive + actuator control
All endpoints respond with JSON and talk to the Arduino via `pyserial`.

- `POST /api/drive/joystick` – accepts `{"direction": "Forward"}` *or* `{"x": 0.3, "y": 0.9}` payloads. Directions map to the verbs accepted by `ManualControl.ino`; analog axes are quantized to FORWARD/BACKWARD/LEFT/RIGHT with a deadzone (tunable via `JOYSTICK_DEADZONE`, default 0.3).
- `POST /api/drive/stop` – emergency STOP.
- `POST /api/lift/up`, `/api/lift/down` – relays to `LIFT_UP_COMMAND`/`LIFT_DOWN_COMMAND` (override via env vars). A `lift/stop` placeholder already exists in `main.py` if you hook that motor up.
- `POST /api/fan/on`, `/api/fan/off` – toggles auxiliary actuators using the customizable `FAN_*` command strings.
- `POST /api/mode` – set the rover state to `manual` or `autonomous`; `GET /api/mode` returns the current mode so the Android app stays in sync.

For every command the server returns whether the queue accepted the job and the raw serial response (so the client can show firmware errors).

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
