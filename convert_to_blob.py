"""
Utility script to export YOLO `.pt` checkpoints to ONNX and DepthAI blobs.

Example:
    python convert_to_blob.py runs/train/weights/best.pt --imgsz 640 \
        --onnx-dir my_blobs --blob-dir my_blobs
"""

from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

import blobconverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export YOLO .pt weights to ONNX and then compile them into DepthAI blobs."
        )
    )
    parser.add_argument(
        "weights",
        nargs="+",
        type=Path,
        help="Path(s) to YOLO checkpoint(s) (best.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square input size used during ONNX export (default: 640).",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path("my_blobs"),
        help="Directory to place exported ONNX files (default: ./my_blobs).",
    )
    parser.add_argument(
        "--blob-dir",
        type=Path,
        default=Path("my_blobs"),
        help="Directory to place compiled .blob files (default: ./my_blobs).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (default: 12).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device used by Ultralytics export (default: cpu).",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable graph simplification when exporting ONNX.",
    )
    parser.add_argument(
        "--openvino-version",
        default=blobconverter.Versions.v2022_1,
        choices=[
            blobconverter.Versions.v2022_3_RVC3,
            blobconverter.Versions.v2022_1,
            blobconverter.Versions.v2021_4,
            blobconverter.Versions.v2021_3,
            blobconverter.Versions.v2021_2,
        ],
        help="OpenVINO version used for blob compilation.",
    )
    parser.add_argument(
        "--shaves",
        type=int,
        default=6,
        help="Number of MyriadX shaves for the blob (default: 6).",
    )
    parser.add_argument(
        "--data-type",
        default="FP16",
        choices=["FP16", "FP32"],
        help="Precision for DepthAI blob (default: FP16).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force blobconverter to bypass cached blobs.",
    )
    return parser.parse_args()


def ensure_weights_exist(weight_paths: Sequence[Path]) -> None:
    missing = [path for path in weight_paths if not path.is_file()]
    if missing:
        missing_str = "\n".join(f" - {path}" for path in missing)
        raise SystemExit(f"The following weight files do not exist:\n{missing_str}")


def export_to_onnx(weights_path: Path, output_dir: Path, imgsz: int, opset: int, device: str, simplify: bool) -> Path:
    try:
        from ultralytics import YOLO  # lazy import to avoid hard dependency for inference
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required for exporting models. Install it via `pip install ultralytics`."
        ) from exc

    model = YOLO(str(weights_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            device=device,
            simplify=simplify,
        )
    )

    desired_path = output_dir / f"{weights_path.stem}.onnx"
    desired_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != desired_path.resolve():
        desired_path.unlink(missing_ok=True)
        shutil.move(str(exported_path), desired_path)
    return desired_path


def compile_to_blob(
    onnx_path: Path,
    blob_dir: Path,
    openvino_version: str,
    shaves: int,
    data_type: str,
    use_cache: bool,
) -> Path:
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blobconverter.from_onnx(
        model=str(onnx_path),
        version=openvino_version,
        shaves=shaves,
        data_type=data_type,
        output_dir=str(blob_dir),
        use_cache=use_cache,
    )
    return Path(blob_path)


def main() -> None:
    args = parse_args()
    ensure_weights_exist(args.weights)

    for weight in args.weights:
        print(f"[INFO] Exporting {weight} to ONNX...")
        onnx_path = export_to_onnx(
            weights_path=weight,
            output_dir=args.onnx_dir,
            imgsz=args.imgsz,
            opset=args.opset,
            device=args.device,
            simplify=not args.no_simplify,
        )
        print(f"[INFO] Saved ONNX model to {onnx_path}")

        print(f"[INFO] Compiling {onnx_path.name} to DepthAI blob...")
        blob_path = compile_to_blob(
            onnx_path=onnx_path,
            blob_dir=args.blob_dir,
            openvino_version=args.openvino_version,
            shaves=args.shaves,
            data_type=args.data_type,
            use_cache=not args.no_cache,
        )
        print(f"[SUCCESS] Blob ready: {blob_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
