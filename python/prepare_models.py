"""Download, clean up and quantize the ONNX models used by the Flutter app.

Usage:
    python -m venv .venv && .venv/bin/pip install -r python/requirements.txt
    .venv/bin/python python/prepare_models.py
"""

import urllib.request
from pathlib import Path

import onnx
from onnx import version_converter
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

ZOO = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis"
MODELS = {
    "ultraface_rfb_320.onnx": f"{ZOO}/ultraface/models/version-RFB-320.onnx",
    "emotion_ferplus.onnx": f"{ZOO}/emotion_ferplus/model/emotion-ferplus-8.onnx",
}

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "models"
CACHE = ROOT / "python" / ".cache"


def download(name: str, url: str) -> Path:
    path = CACHE / name
    if not path.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


TARGET_OPSET = 13


def modernize(src: Path, dst: Path) -> None:
    """Upgrade old Model Zoo exports so they quantize and optimize cleanly.

    1. Convert to opset 13 (the quantizer needs >= 11).
    2. Old exporters listed weights as graph inputs. Removing them lets
       ONNX Runtime constant-fold and keeps the session's input list clean.
    """
    model = onnx.load(src)
    if model.opset_import[0].version < TARGET_OPSET:
        model = version_converter.convert_version(model, TARGET_OPSET)
    weights = {init.name for init in model.graph.initializer}
    real_inputs = [i for i in model.graph.input if i.name not in weights]
    del model.graph.input[:]
    model.graph.input.extend(real_inputs)
    onnx.save(model, dst)


def describe(path: Path) -> None:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    size_mb = path.stat().st_size / 1e6
    print(f"\n{path.name} ({size_mb:.1f} MB)")
    for i in session.get_inputs():
        print(f"  in : {i.name} {i.shape} {i.type}")
    for o in session.get_outputs():
        print(f"  out: {o.name} {o.shape} {o.type}")


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for name, url in MODELS.items():
        modernize(download(name, url), OUT / name)

    # INT8 dynamic quantization: ~4x smaller weights, near-identical accuracy.
    quantize_dynamic(
        OUT / "emotion_ferplus.onnx",
        OUT / "emotion_ferplus_int8.onnx",
        weight_type=QuantType.QUInt8,
    )

    for path in sorted(OUT.glob("*.onnx")):
        describe(path)


if __name__ == "__main__":
    main()
