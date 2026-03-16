## How to set up the environment
```bash
/opt/homebrew/bin/python3.10 -m venv venv310
source venv310/bin/activate
pip install -r requirements.txt
```

> DepthAI only publishes prebuilt wheels up through Python 3.11, so make sure your virtual environment uses Python 3.10 (recommended) or 3.11. Trying to install with a newer interpreter (e.g., 3.12–3.14) forces a source build that currently fails on macOS.

> `requirements.txt` now only contains the three packages that are actually needed to run `main.py` (`depthai`, `numpy`, and `opencv-contrib-python`). This avoids pinning dozens of transitive dependencies that frequently conflict on fresh machines.

Run the main pipeline with either `python main.py` (while the venv is activated) or `venv310/bin/python main.py`.

## Multi-camera OAK-D monitoring

`main.py` can drive multiple OAK-D cameras simultaneously (see `CameraSetup` near the top of the file). Each entry simply names the camera and points to the blob that should be loaded, and the script automatically pairs those setups with whatever DepthAI devices are enumerated at runtime. Connect as many devices as you have USB bandwidth for, add matching `CameraSetup` items, and the script will launch a YOLO pipeline per camera so detections/overlays are handled independently.

If fewer devices are attached than setups configured, the script logs a warning and only activates the number of cameras that are physically present. This makes it easy to keep a canonical list of expected camera viewpoints while still being able to test with a subset in the lab.

## Browser-based web stream

When you run `python main.py` the script also spins up a lightweight Flask server (default host `0.0.0.0`, port `5000`). All active camera feeds are composited into a 2×2 grid (padding with black tiles if fewer than four cameras are active) and exposed as an MJPEG stream at `http://<host>:5000/video`. Visiting `http://<host>:5000/` in a browser displays the live stream without any other UI.

Because the server binds to `0.0.0.0`, you can monitor the cameras from any machine on the same network by hitting the host’s IP address. Adjust the host/port inside `main.py` if your deployment requires something different.

## Pest summary API for mobile apps

The Flask server now keeps a running count of every pest label observed (e.g., “Fungus gnats”) and stores the latest UTC timestamp + camera that produced it. This data is exposed over HTTP so a mobile app can poll for updates:

- `GET /api/pests` returns an object with `pests` (array of `{label, count, last_seen_utc, last_seen_camera}` items), the number of tracked labels, and the generation timestamp.
- `GET /api/pests/<label>` (URL-encode spaces) returns just the summary for a single pest label and responds with HTTP 404 if nothing has been recorded yet.

Both endpoints read from the `pest_summary` table, which is refreshed in real time as detections are inserted into the database. No authentication is implemented, so keep the service on a trusted network.

## Integrated ArUco marker detection

In addition to YOLO detections, each frame is scanned for 4×4 ArUco markers (OpenCV dictionary `DICT_4X4_50`). Marker IDs are looked up in the `TRAP_REGISTRY` table to attach friendly trap names and locations, and the overlays also include live counters (per camera, plus the total number of unique traps seen so far). Update `TRAP_REGISTRY` in `main.py` with the IDs/metadata that match your printed markers and deployment layout.

These annotations are included both in the native display windows (if you enable them) and in the MJPEG web stream, so remote operators can immediately tell which traps each camera is covering and whether any new traps have entered the scene.

## Convert a YOLO `best.pt` into a DepthAI blob

The repository now contains `convert_to_blob.py` which automates exporting a YOLO checkpoint to ONNX (via Ultralytics) and compiling it into a DepthAI blob through `blobconverter`.

1. Install the optional conversion dependencies (this file also reuses the core runtime requirements):

   ```bash
   pip install -r requirements-convert.txt
   ```

2. Run the converter, pointing it to one or more `.pt` files. The example below emits both the ONNX file and the `.blob` into `my_blobs/`:

   ```bash
   python convert_to_blob.py path/to/best.pt --imgsz 640 --onnx-dir my_blobs --blob-dir my_blobs --shaves 6
   ```

   Additional flags let you select the OpenVINO version (`--openvino-version`), MyriadX compute shaves (`--shaves`), opset, and other knobs. Use `-h` to see the full list of options.
