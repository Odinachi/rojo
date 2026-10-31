# ONNX Emotions: Running ML Models in Flutter with ONNX

A demo app for the talk **"ONNX: Running ML Models in Flutter with ONNX"**.
It chains two ONNX models fully on-device, with no network:

```
Camera frame ─► UltraFace (find faces) ─► crop + grayscale 64×64 ─► FER+ ─► softmax ─► "😄 Happy 92%"
   1.3 MB detector                                                  35 MB / 21 MB INT8 classifier
```

## Quick start

```bash
flutter pub get
flutter run            # pick a real device for the live camera
```

The models are already in `assets/models/`. To regenerate them:

```bash
python3 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
.venv/bin/python python/prepare_models.py   # download, upgrade opset, quantize
.venv/bin/python python/verify_models.py    # run the same pipeline in Python
```

## What's where (talk outline)

| # | Talk section | Where to look |
|---|---|---|
| 1 | **What is ONNX?** One file format, many runtimes | `assets/models/*.onnx` |
| 2 | **Getting a model**: ONNX Model Zoo, export from PyTorch/TF | `python/prepare_models.py` (`MODELS`) |
| 3 | **Real-world cleanup**: old exports need opset upgrades and initializer fixes | `prepare_models.py` → `modernize()` |
| 4 | **Inspect the contract**: inputs/outputs, shapes, dtypes (show Netron) | `prepare_models.py` → `describe()` |
| 5 | **Prototype in Python first**: the reference pipeline | `python/verify_models.py` |
| 6 | **Load in Flutter**: `flutter_onnxruntime`, `createSessionFromAsset` | `lib/ml/face_detector.dart`, `emotion_classifier.dart` |
| 7 | **Preprocessing is where bugs live**: resize, normalize, HWC→NCHW, grayscale | `FaceDetector.preprocess`, `EmotionClassifier.preprocess` |
| 8 | **Run inference**: `OrtValue.fromList` → `session.run` → `asFlattenedList`, and `dispose()` | `FaceDetector.detect`, `EmotionClassifier.classify` |
| 9 | **Postprocessing**: thresholds, NMS, softmax | `FaceDetector.decode`, `EmotionClassifier.softmax` |
| 10 | **Chaining models** | `lib/ml/emotion_pipeline.dart` |
| 11 | **Keep the UI smooth**: isolates, frame dropping | `EmotionPipeline.run`, `lib/ui/camera_screen.dart` |
| 12 | **Camera frames**: BGRA (iOS) vs YUV420 (Android), rotation, mirroring | `lib/camera/frame_converter.dart` |
| 13 | **Quantization**: FP32 vs INT8, size vs accuracy | **INT8 model** toggle in the app |
| 14 | **Execution providers**: CPU vs CoreML / NNAPI | **CoreML / NNAPI** toggle in the app |

## Model contracts

| Model | Input | Output | Preprocessing |
|---|---|---|---|
| UltraFace RFB-320 | `input` [1,3,240,320] | `scores` [1,4420,2], `boxes` [1,4420,4] | RGB, `(px − 127) / 128`, NCHW |
| Emotion FER+ | `Input3` [1,1,64,64] | `Plus692_Output_0` [1,8] | grayscale, **raw 0–255** (no normalization) |

FER+ classes: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt.

## Live demo tips

- **Plan B:** the **Photo** tab has a bundled **Sample** image, so the demo works even if the camera or lighting fails.
- Exaggerated expressions work best (big smile, wide-eyed surprise). FER+ often confuses *neutral* and *contempt*.
- Make sure faces are well lit from the front, since stage backlighting hurts detection.
- Switch to airplane mode on stage to show that everything runs on-device.
- Toggle **INT8** and **CoreML/NNAPI** live and point at the timing line in the stats panel.
- Use a **release** build on stage (`flutter run --release`). Debug-mode timings are several times slower.

## Platform notes

- **iOS:** minimum iOS 16 with static framework linkage (see `ios/Podfile`), as required by ONNX Runtime.
- **Android:** `android/app/proguard-rules.pro` keeps the `ai.onnxruntime` classes for release builds.

## Tests

```bash
flutter test
```

These cover preprocessing, NMS, softmax, camera frame conversion and the config toggles. Inference itself needs native ONNX Runtime, so it's verified by running the app or by `python/verify_models.py`.

## Credits

- [UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) and [Emotion FER+](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) from the [ONNX Model Zoo](https://github.com/onnx/models) (MIT).
- Sample image: "astronaut" (Eileen Collins, NASA, public domain) via scikit-image.
