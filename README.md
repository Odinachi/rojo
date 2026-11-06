# ONNX Emotions: Running ML Models in Flutter with ONNX

A demo Flutter app and companion talk material for **"ONNX: Running ML Models in Flutter with ONNX"**. It chains two ONNX models fully on-device — no network required — to detect faces in real time and classify each face's emotion.

```
Camera frame ─► UltraFace (find faces) ─► crop + grayscale 64×64 ─► FER+ ─► softmax ─► "😄 Happy 92%"
   1.3 MB detector                                                  35 MB / 21 MB INT8 classifier
```

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [How It Works — Step by Step](#how-it-works--step-by-step)
  - [Step 1: Model Acquisition](#step-1-model-acquisition)
  - [Step 2: Model Cleanup and Opset Upgrade](#step-2-model-cleanup-and-opset-upgrade)
  - [Step 3: Quantization (FP32 → INT8)](#step-3-quantization-fp32--int8)
  - [Step 4: Model Contracts — Inputs, Outputs, and Shapes](#step-4-model-contracts--inputs-outputs-and-shapes)
  - [Step 5: Reference Python Pipeline](#step-5-reference-python-pipeline)
  - [Step 6: Loading Models in Flutter](#step-6-loading-models-in-flutter)
  - [Step 7: Preprocessing — Where Bugs Live](#step-7-preprocessing--where-bugs-live)
  - [Step 8: Running Inference](#step-8-running-inference)
  - [Step 9: Postprocessing — Thresholds, NMS, Softmax](#step-9-postprocessing--thresholds-nms-softmax)
  - [Step 10: Chaining Models into a Pipeline](#step-10-chaining-models-into-a-pipeline)
  - [Step 11: Camera Frame Conversion (BGRA / YUV420)](#step-11-camera-frame-conversion-bgra--yuv420)
  - [Step 12: Keeping the UI Smooth — Isolates and Frame Dropping](#step-12-keeping-the-ui-smooth--isolates-and-frame-dropping)
  - [Step 13: Execution Providers — CPU vs CoreML / NNAPI](#step-13-execution-providers--cpu-vs-coreml--nnapi)
- [App UI](#app-ui)
- [Model Contracts Reference](#model-contracts-reference)
- [Configuration Options](#configuration-options)
- [Testing](#testing)
- [Platform Notes](#platform-notes)
- [Live Demo Tips](#live-demo-tips)
- [Regenerating the Models](#regenerating-the-models)
- [Dependencies](#dependencies)
- [Credits](#credits)

---

## Features

- **Two-model pipeline**: UltraFace face detection → FER+ emotion classification, chained on every frame.
- **100% on-device inference**: No internet connection needed at runtime. Toggle airplane mode during demos.
- **Real-time camera feed**: Live front/back camera with face bounding boxes and emotion labels overlaid.
- **Photo mode**: Pick from gallery, take a photo, or use the bundled sample image (failsafe for demos).
- **INT8 quantization toggle**: Switch between FP32 (35 MB) and INT8 (21 MB) emotion model at runtime.
- **Hardware acceleration toggle**: CoreML on iOS, NNAPI on Android — with automatic CPU fallback.
- **Per-stage timing panel**: Shows preprocessing, detection, classification, and total latency in milliseconds.
- **Probability distribution chart**: Full 8-emotion breakdown with animated bars for the most confident face.
- **FPS counter**: Exponentially-smoothed frames-per-second display in live camera mode.
- **Cross-platform**: iOS 16+ and Android, handling the different camera frame formats transparently.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Flutter App                                    │
│                                                                             │
│  ┌─────────────┐     ┌──────────────────────────────────────────────────┐   │
│  │   UI Layer   │     │               ML Layer (lib/ml/)                │   │
│  │              │     │                                                  │   │
│  │ HomeScreen   │     │  ┌──────────────┐    ┌────────────────────┐     │   │
│  │ CameraScreen ├────►│  │ FaceDetector  │    │ EmotionClassifier  │     │   │
│  │ PhotoScreen  │     │  │ (UltraFace)   │───►│ (FER+)             │     │   │
│  │ StatsPanel   │     │  └──────────────┘    └────────────────────┘     │   │
│  │ FaceOverlay  │     │           ▲                                      │   │
│  └──────┬───────┘     │           │     EmotionPipeline                  │   │
│         │             │           │     (orchestrator + isolates)         │   │
│         │             └───────────┼──────────────────────────────────────┘   │
│         │                         │                                          │
│  ┌──────▼───────────┐  ┌─────────┴──────────┐                              │
│  │ PipelineController│  │ FrameConverter     │                              │
│  │ (config + reload) │  │ (BGRA/YUV → RGB)   │                              │
│  └──────────────────┘  └────────────────────┘                              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  flutter_onnxruntime  ←→  ONNX Runtime (C++ / CoreML / NNAPI)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Install Flutter dependencies
flutter pub get

# 2. Run on a real device (camera requires physical hardware)
flutter run

# For demo/stage use, always use a release build:
flutter run --release
```

The ONNX models are already pre-built in `assets/models/`. No Python setup is needed to run the app. See [Regenerating the Models](#regenerating-the-models) if you want to rebuild them from source.

---

## Project Structure

```
├── lib/
│   ├── main.dart                         # App entry point, HomeScreen, ConfigBar
│   ├── camera/
│   │   └── frame_converter.dart          # BGRA/YUV420 → upright RGB conversion
│   ├── ml/
│   │   ├── face_detector.dart            # UltraFace wrapper: preprocess, detect, NMS
│   │   ├── emotion_classifier.dart       # FER+ wrapper: preprocess, classify, softmax
│   │   └── emotion_pipeline.dart         # Chains both models + isolate offloading
│   └── ui/
│       ├── camera_screen.dart            # Live camera feed + frame dropping
│       ├── photo_screen.dart             # Pick/sample photo + run pipeline
│       ├── face_overlay.dart             # CustomPainter for face boxes + emotion labels
│       ├── pipeline_controller.dart      # ChangeNotifier for config + model lifecycle
│       └── stats_panel.dart              # Timing, FPS, probability bars
├── assets/
│   ├── models/
│   │   ├── ultraface_rfb_320.onnx        # 1.3 MB face detector
│   │   ├── emotion_ferplus.onnx          # 35 MB FP32 emotion classifier
│   │   └── emotion_ferplus_int8.onnx     # 21 MB INT8 quantized classifier
│   └── samples/
│       └── astronaut.png                 # Bundled sample image for Plan B demos
├── python/
│   ├── prepare_models.py                 # Download, opset-upgrade, quantize
│   ├── verify_models.py                  # Reference pipeline in Python
│   ├── requirements.txt                  # onnx, onnxruntime, numpy, pillow
│   ├── .cache/                           # Raw downloaded models (gitignored)
│   └── samples/
│       └── astronaut.png                 # Test image for the Python pipeline
├── test/
│   ├── home_screen_test.dart             # Widget test: config chips → pipeline reload
│   ├── camera/
│   │   └── frame_converter_test.dart     # BGRA/YUV, rotation, mirroring
│   └── ml/
│       ├── face_detector_test.dart       # Preprocessing, NMS, IoU
│       └── emotion_classifier_test.dart  # Preprocessing, grayscale luma, softmax
├── pubspec.yaml                          # Flutter project config + dependencies
├── ios/                                  # iOS platform project
└── android/                              # Android platform project
```

---

## How It Works — Step by Step

### Step 1: Model Acquisition

Both models come from the [ONNX Model Zoo](https://github.com/onnx/models), an open collection of pre-trained models in ONNX format:

- **UltraFace RFB-320** — A 1.3 MB ultra-lightweight face detector optimized for mobile. It outputs bounding boxes and confidence scores for up to 4,420 anchor positions.
- **Emotion FER+** — A 35 MB (FP32) facial expression classifier trained on the FER+ dataset with crowd-sourced labels. It classifies faces into 8 emotions.

The Python script `python/prepare_models.py` downloads these from GitHub and caches them locally:

```python
ZOO = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis"
MODELS = {
    "ultraface_rfb_320.onnx": f"{ZOO}/ultraface/models/version-RFB-320.onnx",
    "emotion_ferplus.onnx":   f"{ZOO}/emotion_ferplus/model/emotion-ferplus-8.onnx",
}
```

### Step 2: Model Cleanup and Opset Upgrade

Model Zoo exports can be several years old, built with older tooling that produces `.onnx` files with two issues:

1. **Old opset version**: The ONNX Runtime quantizer requires opset ≥ 11. Old exports may be opset 7 or 8.
2. **Weights listed as graph inputs**: Older exporters (especially CNTK) registered model weights as graph inputs. This means `session.inputNames` returns hundreds of entries instead of just the real input tensor.

The `modernize()` function fixes both:

```python
def modernize(src: Path, dst: Path) -> None:
    model = onnx.load(src)
    if model.opset_import[0].version < TARGET_OPSET:
        model = version_converter.convert_version(model, TARGET_OPSET)

    # Remove initializers from graph inputs
    weights = {init.name for init in model.graph.initializer}
    real_inputs = [i for i in model.graph.input if i.name not in weights]
    del model.graph.input[:]
    model.graph.input.extend(real_inputs)
    onnx.save(model, dst)
```

### Step 3: Quantization (FP32 → INT8)

Dynamic INT8 quantization reduces the emotion model from 35 MB to 21 MB (~40% smaller) by converting floating-point weights to 8-bit integers. Accuracy loss is negligible for this model:

```python
from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic(
    OUT / "emotion_ferplus.onnx",        # 35 MB input
    OUT / "emotion_ferplus_int8.onnx",   # 21 MB output
    weight_type=QuantType.QUInt8,
)
```

The app ships both variants and lets the user toggle between them at runtime via the **INT8 model** chip in the config bar.

### Step 4: Model Contracts — Inputs, Outputs, and Shapes

Before writing any Dart code, you must know the exact input and output specifications of each model. The `describe()` function prints this information:

```python
def describe(path: Path) -> None:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for i in session.get_inputs():
        print(f"  in : {i.name} {i.shape} {i.type}")
    for o in session.get_outputs():
        print(f"  out: {o.name} {o.shape} {o.type}")
```

**Contracts at a glance:**

| Model | Input Name | Input Shape | Output Names | Output Shapes | Preprocessing |
|---|---|---|---|---|---|
| UltraFace RFB-320 | `input` | `[1, 3, 240, 320]` | `scores`, `boxes` | `[1, 4420, 2]`, `[1, 4420, 4]` | RGB, `(px − 127) / 128`, NCHW layout |
| Emotion FER+ | `Input3` | `[1, 1, 64, 64]` | `Plus692_Output_0` | `[1, 8]` | Grayscale, **raw 0–255** (no normalization!), NCHW layout |

> **⚠️ Critical**: FER+ expects raw 0–255 pixel values as `float32`. Do NOT apply mean/std normalization — the model was trained on unnormalized pixel values.

You can also use [Netron](https://netron.app) to open `.onnx` files and visually inspect every node, tensor shape, and dtype.

### Step 5: Reference Python Pipeline

`python/verify_models.py` implements the same pipeline in Python using NumPy and PIL. This serves as the **ground truth**: if the Flutter app and the Python script produce different results on the same image, the bug is in the Dart preprocessing.

```python
def detect_faces(session, image, threshold=0.7):
    x = np.asarray(image.convert("RGB").resize((320, 240)), dtype=np.float32)
    x = ((x - 127.0) / 128.0).transpose(2, 0, 1)[None]   # HWC→NCHW
    scores, boxes = session.run(None, {"input": x})
    # ... threshold + NMS ...

def classify_emotion(session, face):
    x = np.asarray(face.convert("L").resize((64, 64)), dtype=np.float32)
    logits = session.run(None, {"Input3": x[None, None]})[0][0]
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()     # softmax
```

```bash
.venv/bin/python python/verify_models.py python/samples/astronaut.png
# astronaut.png: 1 face(s)
# [emotion_ferplus.onnx]
#   face 0.99 at (165,45)-(345,282): happiness 92%, surprise 5%, neutral 2%
```

### Step 6: Loading Models in Flutter

The app uses [`flutter_onnxruntime`](https://pub.dev/packages/flutter_onnxruntime) to load ONNX models from the Flutter asset bundle. Each model is wrapped in its own class with a static `load()` factory:

```dart
// lib/ml/face_detector.dart
class FaceDetector {
  FaceDetector._(this._session);

  static const asset = 'assets/models/ultraface_rfb_320.onnx';
  final OrtSession _session;

  static Future<FaceDetector> load(OnnxRuntime ort, {OrtSessionOptions? options}) async {
    return FaceDetector._(await ort.createSessionFromAsset(asset, options: options));
  }
}

// lib/ml/emotion_classifier.dart
class EmotionClassifier {
  static const fp32Asset = 'assets/models/emotion_ferplus.onnx';
  static const int8Asset = 'assets/models/emotion_ferplus_int8.onnx';

  static Future<EmotionClassifier> load(OnnxRuntime ort, {bool int8 = false, ...}) async {
    final asset = int8 ? int8Asset : fp32Asset;
    return EmotionClassifier._(await ort.createSessionFromAsset(asset, options: options));
  }
}
```

`createSessionFromAsset` handles the platform-specific asset extraction and native library loading automatically.

### Step 7: Preprocessing — Where Bugs Live

Preprocessing is the #1 source of bugs when porting ML models. You must match the training pipeline **exactly** — wrong normalization, wrong channel order, or wrong layout will silently produce garbage results.

**Face detector preprocessing** (UltraFace):

```dart
// lib/ml/face_detector.dart
static Float32List preprocess(img.Image image) {
  final resized = img.copyResize(image, width: 320, height: 240,
      interpolation: img.Interpolation.linear);
  final rgb = resized.getBytes(order: img.ChannelOrder.rgb);
  const plane = 320 * 240;                          // 76,800 pixels per channel
  final tensor = Float32List(3 * plane);             // total: 230,400 floats
  for (var i = 0; i < plane; i++) {
    tensor[i]             = (rgb[i * 3]     - 127) / 128;  // R channel
    tensor[plane + i]     = (rgb[i * 3 + 1] - 127) / 128;  // G channel
    tensor[2 * plane + i] = (rgb[i * 3 + 2] - 127) / 128;  // B channel
  }
  return tensor;
}
```

Key decisions:
- **Resize** to 320×240 (the model's fixed input size).
- **Normalize** each pixel: `(value − 127) / 128`, mapping 0–255 to roughly −1.0 to +1.0.
- **Reorder** from interleaved HWC (`RGBRGBRGB...`) to planar NCHW (`RRR...GGG...BBB...`).

**Emotion classifier preprocessing** (FER+):

```dart
// lib/ml/emotion_classifier.dart
static Float32List preprocess(img.Image image, Rect box) {
  // Crop the detected face region (box is normalized 0..1)
  final crop = img.copyCrop(image, x: ..., y: ..., width: ..., height: ...);
  final resized = img.copyResize(crop, width: 64, height: 64,
      interpolation: img.Interpolation.linear);
  final rgb = resized.getBytes(order: img.ChannelOrder.rgb);
  final tensor = Float32List(64 * 64);
  for (var i = 0; i < tensor.length; i++) {
    // ITU-R BT.601 luma weights (same as PIL's .convert('L'))
    tensor[i] = (rgb[i * 3] * 299 + rgb[i * 3 + 1] * 587 + rgb[i * 3 + 2] * 114) / 1000;
  }
  return tensor;   // Raw 0..255 floats — NO normalization!
}
```

Key decisions:
- **Crop** the face bounding box from the original image.
- **Resize** to 64×64.
- **Convert to grayscale** using the ITU-R BT.601 luma formula: `Y = 0.299R + 0.587G + 0.114B`.
- **No normalization**: the model expects raw pixel values 0–255.

### Step 8: Running Inference

Inference follows a consistent pattern: create tensor → run session → read outputs → dispose:

```dart
// lib/ml/face_detector.dart
Future<List<Face>> detect(Float32List input, {double threshold = 0.7}) async {
  final tensor = await OrtValue.fromList(input, [1, 3, 240, 320]);
  final outputs = await _session.run({'input': tensor});
  try {
    final scores = await outputs['scores']!.asFlattenedList();
    final boxes  = await outputs['boxes']!.asFlattenedList();
    return decode(scores.cast<num>(), boxes.cast<num>(), threshold: threshold);
  } finally {
    await tensor.dispose();
    for (final value in outputs.values) {
      await value.dispose();
    }
  }
}
```

> **⚠️ Memory management**: Every `OrtValue` (both inputs and outputs) holds native memory allocated by ONNX Runtime. You **must** call `dispose()` when done. The `try/finally` pattern ensures disposal even if decoding throws.

The emotion classifier uses `_session.inputNames.first` and `_session.outputNames.first` instead of hardcoding `'Input3'` and `'Plus692_Output_0'` (which are artifacts of the original CNTK export):

```dart
final outputs = await _session.run({_session.inputNames.first: tensor});
final logits = await outputs[_session.outputNames.first]!.asFlattenedList();
return softmax(logits.cast<num>());
```

### Step 9: Postprocessing — Thresholds, NMS, Softmax

**Face detector postprocessing** involves three stages:

1. **Thresholding**: Discard anchors below the confidence threshold (default 0.7):
   ```dart
   final score = scores[i * 2 + 1].toDouble();
   if (score < threshold) continue;
   ```

2. **Box clamping**: Clamp coordinates to `[0, 1]` to prevent boxes from exceeding the image:
   ```dart
   final box = Rect.fromLTRB(
     boxes[i * 4].toDouble().clamp(0, 1), ...
   );
   ```

3. **Non-max suppression (NMS)**: Sort by confidence, then greedily keep boxes that don't overlap too much with already-kept boxes:
   ```dart
   candidates.sort((a, b) => b.score.compareTo(a.score));
   final kept = <Face>[];
   for (final face in candidates) {
     if (kept.every((k) => iou(k.box, face.box) <= iouThreshold)) {
       kept.add(face);
     }
   }
   ```

**Emotion classifier postprocessing** — Softmax converts raw logits into probabilities:

```dart
static List<double> softmax(List<num> logits) {
  final maxLogit = logits.map((l) => l.toDouble()).reduce(math.max);
  final exps = [for (final l in logits) math.exp(l - maxLogit)];
  final sum = exps.reduce((a, b) => a + b);
  return [for (final e in exps) e / sum];
}
```

Subtracting `maxLogit` before exponentiation prevents numerical overflow — a standard technique.

### Step 10: Chaining Models into a Pipeline

`EmotionPipeline` orchestrates the full detection → classification flow and tracks per-stage timings:

```dart
// lib/ml/emotion_pipeline.dart
Future<PipelineResult> run(img.Image image) async {
  final stopwatch = Stopwatch()..start();

  // 1. Preprocess for face detection (background isolate)
  final detectorInput = await _preprocessDetector(image);
  var preprocess = stopwatch.elapsed;

  // 2. Run face detection
  stopwatch.reset();
  final faces = await _detector.detect(detectorInput);
  final detect = stopwatch.elapsed;

  // 3. Crop and preprocess each face for emotion (background isolate)
  stopwatch.reset();
  final crops = faces.isEmpty ? const [] : await _preprocessFaces(image, faces);
  preprocess += stopwatch.elapsed;

  // 4. Classify each face
  stopwatch.reset();
  final results = <FaceEmotion>[];
  for (var i = 0; i < faces.length; i++) {
    results.add(FaceEmotion(faces[i], await _classifier.classify(crops[i])));
  }
  final classify = stopwatch.elapsed;

  return PipelineResult(faces: results, preprocess: preprocess,
      detect: detect, classify: classify);
}
```

### Step 11: Camera Frame Conversion (BGRA / YUV420)

iOS and Android deliver camera frames in **different pixel formats**. The `camera` plugin configuration selects the right format per platform:

```dart
imageFormatGroup: Platform.isIOS
    ? ImageFormatGroup.bgra8888   // iOS: 4 bytes per pixel, interleaved
    : ImageFormatGroup.yuv420,    // Android: 3 planes, chroma subsampled
```

`RawFrame` is a plain data class that copies the camera image's bytes so it can be safely sent to another `Isolate`:

```dart
factory RawFrame.fromCameraImage(CameraImage image, {int rotation = 0, bool mirror = false}) {
  return RawFrame(
    width: image.width, height: image.height,
    format: ...,
    planes: [for (final p in image.planes) p.bytes],
    rowStrides: [for (final p in image.planes) p.bytesPerRow],
    pixelStrides: [for (final p in image.planes) p.bytesPerPixel ?? 1],
    rotation: rotation, mirror: mirror,
  );
}
```

`convertFrame()` performs BGRA→RGB or YUV420→RGB conversion, then applies rotation and horizontal mirroring:

- **BGRA→RGB**: Swaps channel order (B↔R), respects row stride padding.
- **YUV420→RGB**: Uses BT.601 color conversion coefficients, handles chroma subsampling (U/V planes are half resolution).
- **Rotation**: Android delivers landscape sensor frames that need 90° rotation to appear upright in portrait mode.
- **Mirroring**: The front camera preview on Android is mirrored, so frames are flipped to match.

### Step 12: Keeping the UI Smooth — Isolates and Frame Dropping

Two techniques keep the app responsive:

**Isolate offloading** — CPU-intensive pixel manipulation runs in background isolates:

```dart
// Static methods so the closure doesn't capture `this` (which may not be sendable)
static Future<Float32List> _preprocessDetector(img.Image image) =>
    Isolate.run(() => FaceDetector.preprocess(image));

static Future<List<Float32List>> _preprocessFaces(img.Image image, List<Face> faces) =>
    Isolate.run(() => [for (final face in faces) EmotionClassifier.preprocess(image, face.box)]);
```

**Frame dropping** — The camera delivers frames faster than the pipeline can process them. A simple `_busy` boolean ensures only one frame is processed at a time; all others are silently dropped:

```dart
Future<void> _onFrame(CameraImage image) async {
  if (_busy || pipeline == null) return;  // drop this frame
  _busy = true;
  try {
    // ... convert + run pipeline ...
  } finally {
    _busy = false;  // ready for the next frame
  }
}
```

### Step 13: Execution Providers — CPU vs CoreML / NNAPI

ONNX Runtime supports multiple execution providers. The app exposes this as a runtime toggle:

```dart
// lib/ml/emotion_pipeline.dart
List<OrtProvider> get providers => [
  if (accelerated)
    Platform.isIOS ? OrtProvider.CORE_ML : OrtProvider.NNAPI,
  OrtProvider.CPU,   // always include as fallback
];
```

- **CoreML** (iOS): Delegates to Apple's Neural Engine or GPU.
- **NNAPI** (Android): Delegates to the device's neural processing hardware.
- **CPU fallback**: Any operator the accelerator can't handle is silently routed to CPU. The session always loads successfully.

---

## App UI

The app has two tabs and a config bar:

| Component | Description |
|---|---|
| **Config bar** | Two `FilterChip` toggles: **INT8 model** and **CoreML/NNAPI**. Changing either reloads the pipeline. A spinner shows during reload. |
| **Live camera tab** | Front-camera feed with face boxes + emotion labels drawn via `CustomPainter`. Frames are dropped to keep up. Switch-camera FAB. |
| **Photo tab** | Pick from gallery, take a photo, or tap **Sample** for the bundled astronaut image. |
| **Stats panel** | Shows provider name, model variant, face count, FPS, per-stage timings (pre / detect / classify / total), and a full 8-emotion probability bar chart for the most confident face. |

`PipelineController` (a `ChangeNotifier`) manages the model lifecycle. It uses a `_generation` counter to safely handle rapid config changes — if the config changes while a model is still loading, the stale pipeline is discarded.

---

## Model Contracts Reference

### UltraFace RFB-320

| Property | Value |
|---|---|
| **File** | `ultraface_rfb_320.onnx` (1.3 MB) |
| **Input** | `input` — `float32 [1, 3, 240, 320]` |
| **Output 1** | `scores` — `float32 [1, 4420, 2]` (background, face) per anchor |
| **Output 2** | `boxes` — `float32 [1, 4420, 4]` — normalized `[x1, y1, x2, y2]` |
| **Preprocessing** | RGB channels, `(pixel − 127) / 128` normalization, NCHW layout |
| **Postprocessing** | Confidence thresholding (0.7) + non-max suppression (IoU 0.3) |

### Emotion FER+

| Property | Value |
|---|---|
| **File (FP32)** | `emotion_ferplus.onnx` (35 MB) |
| **File (INT8)** | `emotion_ferplus_int8.onnx` (21 MB) |
| **Input** | `Input3` — `float32 [1, 1, 64, 64]` |
| **Output** | `Plus692_Output_0` — `float32 [1, 8]` (logits) |
| **Preprocessing** | Grayscale (BT.601 luma), **raw 0–255** (no normalization!), NCHW layout |
| **Postprocessing** | Softmax to convert logits to probabilities |
| **Classes** | neutral, happiness, surprise, sadness, anger, disgust, fear, contempt |

---

## Configuration Options

The `PipelineConfig` class drives all runtime configuration:

```dart
class PipelineConfig {
  const PipelineConfig({this.int8 = false, this.accelerated = false});

  final bool int8;          // Use INT8-quantized FER+ model
  final bool accelerated;   // Use CoreML (iOS) or NNAPI (Android)
}
```

| Config | Default | Effect |
|---|---|---|
| `int8` | `false` | Loads `emotion_ferplus_int8.onnx` (21 MB) instead of `emotion_ferplus.onnx` (35 MB) |
| `accelerated` | `false` | Adds `OrtProvider.CORE_ML` or `OrtProvider.NNAPI` before the CPU fallback |

---

## Testing

```bash
flutter test
```

The test suite covers pure-Dart logic that doesn't require the native ONNX Runtime:

| Test file | What it covers |
|---|---|
| `test/home_screen_test.dart` | Config chip taps trigger pipeline reload with correct `PipelineConfig` |
| `test/ml/face_detector_test.dart` | `preprocess` produces NCHW tensors normalized around 0; `decode` filters, clamps, and applies NMS; `iou` computes correct intersection-over-union |
| `test/ml/emotion_classifier_test.dart` | `Emotion` enum order matches FER+ output layout; `preprocess` produces raw 0–255 grayscale with correct BT.601 luma weights; `softmax` sums to 1.0 and preserves argmax |
| `test/camera/frame_converter_test.dart` | BGRA→RGB conversion respects row stride; YUV420→RGB handles gray and saturated colors; rotation and mirroring are applied correctly |

For end-to-end inference verification, run the Python reference pipeline:

```bash
.venv/bin/python python/verify_models.py [image.jpg]
```

---

## Platform Notes

### iOS
- **Minimum version**: iOS 16.
- **Static framework linkage**: Required by ONNX Runtime — see `ios/Podfile` for the `use_frameworks! :linkage => :static` configuration.
- **Camera format**: BGRA8888 (interleaved, 4 bytes per pixel).
- **Accelerator**: CoreML — delegates to the Neural Engine on A12+ / M1+ chips.

### Android
- **ProGuard rules**: `android/app/proguard-rules.pro` keeps the `ai.onnxruntime` classes so they aren't stripped in release builds.
- **Camera format**: YUV420 (planar, chroma subsampled).
- **Accelerator**: NNAPI — delegates to device-specific neural processing hardware.
- **Sensor rotation**: Camera frames are delivered in landscape orientation and must be rotated 90° for portrait display.
- **Front camera mirroring**: The preview is mirrored, so frames are horizontally flipped to match.

---

## Live Demo Tips

- **Plan B**: The **Photo** tab has a bundled **Sample** image (astronaut), so the demo works even if the camera or lighting fails.
- Exaggerated expressions work best (big smile, wide-eyed surprise). FER+ often confuses *neutral* and *contempt*.
- Make sure faces are well lit from the front — stage backlighting hurts detection.
- Switch to **airplane mode** on stage to prove everything runs on-device.
- Toggle **INT8** and **CoreML/NNAPI** live and point at the timing line in the stats panel.
- Use a **release build** on stage: `flutter run --release`. Debug-mode timings are several times slower.

---

## Regenerating the Models

The models are pre-built in `assets/models/`, but you can regenerate them from the ONNX Model Zoo:

```bash
# 1. Set up a Python virtual environment
python3 -m venv .venv && .venv/bin/pip install -r python/requirements.txt

# 2. Download, opset-upgrade, fix initializers, and quantize
.venv/bin/python python/prepare_models.py

# 3. Verify the pipeline produces correct results
.venv/bin/python python/verify_models.py
```

This will:
1. Download `version-RFB-320.onnx` and `emotion-ferplus-8.onnx` from GitHub (cached in `python/.cache/`).
2. Upgrade both to opset 13 and remove weight-as-input artifacts.
3. Produce INT8-quantized `emotion_ferplus_int8.onnx` via dynamic quantization.
4. Print input/output shapes and sizes for each model.

---

## Dependencies

### Flutter (pubspec.yaml)

| Package | Version | Purpose |
|---|---|---|
| `flutter_onnxruntime` | ^1.9.0 | ONNX Runtime bindings for Flutter |
| `camera` | ^0.12.2 | Camera access and live frame streaming |
| `image` | ^4.12.0 | Image decoding, resizing, pixel manipulation |
| `image_picker` | ^1.2.4 | Pick photos from gallery or camera (Photo tab) |
| `cupertino_icons` | ^1.0.8 | iOS-style icons |

### Python (requirements.txt)

| Package | Version | Purpose |
|---|---|---|
| `onnx` | ≥1.17 | Load, inspect, and upgrade ONNX models |
| `onnxruntime` | ≥1.20 | Run inference + dynamic quantization |
| `numpy` | ≥1.26 | Tensor manipulation |
| `pillow` | ≥10 | Image loading and preprocessing |

---

## Credits

- [UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) and [Emotion FER+](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) from the [ONNX Model Zoo](https://github.com/onnx/models) (MIT License).
- Sample image: "astronaut" (Eileen Collins, NASA, public domain) via scikit-image.
