## How to run the enviroment
python3 -m venv venv310
source newenv/bin/activate
pip install -r requirements.txt


might need depthai-core

to run main.py run this command
venv310/bin/python main.py

## Convert a YOLO `best.pt` into a DepthAI blob

The repository now contains `convert_to_blob.py` which automates exporting a YOLO checkpoint to ONNX (via Ultralytics) and compiling it into a DepthAI blob through `blobconverter`.

1. Install the extra conversion dependencies (ultralytics pulls in PyTorch):

   ```bash
   pip install ultralytics torch
   ```

2. Run the converter, pointing it to one or more `.pt` files. The example below emits both the ONNX file and the `.blob` into `my_blobs/`:

   ```bash
   python convert_to_blob.py path/to/best.pt --imgsz 640 --onnx-dir my_blobs --blob-dir my_blobs --shaves 6
   ```

   Additional flags let you select the OpenVINO version (`--openvino-version`), MyriadX compute shaves (`--shaves`), opset, and other knobs. Use `-h` to see the full list of options.
