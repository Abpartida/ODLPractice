## How to set up the environment
```bash
/opt/homebrew/bin/python3.10 -m venv venv310
source venv310/bin/activate
pip install -r requirements.txt
```

> DepthAI only publishes prebuilt wheels up through Python 3.11, so make sure your virtual environment uses Python 3.10 (recommended) or 3.11. Trying to install with a newer interpreter (e.g., 3.12–3.14) forces a source build that currently fails on macOS.

> `requirements.txt` now only contains the three packages that are actually needed to run `main.py` (`depthai`, `numpy`, and `opencv-contrib-python`). This avoids pinning dozens of transitive dependencies that frequently conflict on fresh machines.

Run the main pipeline with either `python main.py` (while the venv is activated) or `venv310/bin/python main.py`.

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
