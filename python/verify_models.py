"""Reference implementation of the app's pipeline: detect -> crop -> classify.

The Dart code in lib/ml/ mirrors these steps exactly, so if the app and this
script disagree on the same image, the bug is in the Dart pre/postprocessing.

Usage:
    .venv/bin/python python/verify_models.py [image.jpg]
"""

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "assets" / "models"
EMOTIONS = ["neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear", "contempt"]


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.3) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        overlap = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter)
        order = order[1:][overlap <= iou]
    return keep


def detect_faces(session, image: Image.Image, threshold: float = 0.7):
    # UltraFace: RGB, 320x240, (pixel - 127) / 128, NCHW.
    x = np.asarray(image.convert("RGB").resize((320, 240)), dtype=np.float32)
    x = ((x - 127.0) / 128.0).transpose(2, 0, 1)[None]
    scores, boxes = session.run(None, {"input": x})
    face_scores = scores[0, :, 1]
    mask = face_scores > threshold
    boxes, face_scores = boxes[0][mask], face_scores[mask]
    keep = nms(boxes, face_scores)
    # Boxes come back as normalized [x1, y1, x2, y2] corners in 0..1.
    return [(boxes[i], float(face_scores[i])) for i in keep]


def classify_emotion(session, face: Image.Image):
    # FER+: grayscale, 64x64, raw 0..255 floats (no normalization), NCHW.
    x = np.asarray(face.convert("L").resize((64, 64)), dtype=np.float32)
    logits = session.run(None, {"Input3": x[None, None]})[0][0]
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "python" / "samples" / "astronaut.png"
    image = Image.open(path)
    w, h = image.size

    detector = ort.InferenceSession(MODELS / "ultraface_rfb_320.onnx")
    faces = detect_faces(detector, image)
    print(f"{path.name}: {len(faces)} face(s)")

    for name in ["emotion_ferplus.onnx", "emotion_ferplus_int8.onnx"]:
        classifier = ort.InferenceSession(MODELS / name)
        print(f"\n[{name}]")
        for box, score in faces:
            x1, y1, x2, y2 = (box * [w, h, w, h]).astype(int)
            start = time.perf_counter()
            probs = classify_emotion(classifier, image.crop((x1, y1, x2, y2)))
            ms = (time.perf_counter() - start) * 1000
            top = probs.argsort()[::-1][:3]
            summary = ", ".join(f"{EMOTIONS[i]} {probs[i]:.0%}" for i in top)
            print(f"  face {score:.2f} at ({x1},{y1})-({x2},{y2}): {summary}  [{ms:.1f} ms]")


if __name__ == "__main__":
    main()
