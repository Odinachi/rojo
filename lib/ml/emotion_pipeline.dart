import 'dart:io' show Platform;
import 'dart:isolate';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

import 'emotion_classifier.dart';
import 'face_detector.dart';

/// Which model variant and execution provider to run with.
class PipelineConfig {
  const PipelineConfig({this.int8 = false, this.accelerated = false});

  /// Use the INT8-quantized FER+ model instead of FP32.
  final bool int8;

  /// Use the platform accelerator (CoreML on iOS, NNAPI on Android),
  /// falling back to CPU for any operator it can't handle.
  final bool accelerated;

  static String get acceleratorName => Platform.isIOS ? 'CoreML' : 'NNAPI';

  String get providerName => accelerated ? acceleratorName : 'CPU';

  String get modelName => int8 ? 'FER+ INT8' : 'FER+ FP32';

  List<OrtProvider> get providers => [
    if (accelerated) Platform.isIOS ? OrtProvider.CORE_ML : OrtProvider.NNAPI,
    OrtProvider.CPU,
  ];

  PipelineConfig copyWith({bool? int8, bool? accelerated}) => PipelineConfig(
    int8: int8 ?? this.int8,
    accelerated: accelerated ?? this.accelerated,
  );

  @override
  bool operator ==(Object other) =>
      other is PipelineConfig &&
      other.int8 == int8 &&
      other.accelerated == accelerated;

  @override
  int get hashCode => Object.hash(int8, accelerated);
}

class FaceEmotion {
  const FaceEmotion(this.face, this.probabilities);

  final Face face;

  /// Indexed by [Emotion.index].
  final List<double> probabilities;

  Emotion get top {
    var best = 0;
    for (var i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[best]) best = i;
    }
    return Emotion.values[best];
  }

  double get confidence => probabilities[top.index];
}

class PipelineResult {
  const PipelineResult({
    required this.faces,
    required this.preprocess,
    required this.detect,
    required this.classify,
  });

  final List<FaceEmotion> faces;
  final Duration preprocess;
  final Duration detect;
  final Duration classify;

  Duration get total => preprocess + detect + classify;
}

/// Two ONNX models chained together:
/// image -> UltraFace (find faces) -> crop -> FER+ (classify each face).
class EmotionPipeline {
  EmotionPipeline._(this.config, this._detector, this._classifier);

  final PipelineConfig config;
  final FaceDetector _detector;
  final EmotionClassifier _classifier;

  static Future<EmotionPipeline> load(PipelineConfig config) async {
    final ort = OnnxRuntime();
    final options = OrtSessionOptions(providers: config.providers);
    final detector = await FaceDetector.load(ort, options: options);
    final classifier = await EmotionClassifier.load(
      ort,
      int8: config.int8,
      options: options,
    );
    return EmotionPipeline._(config, detector, classifier);
  }

  /// [image] must be upright RGB, in the same orientation shown on screen.
  Future<PipelineResult> run(img.Image image) async {
    final stopwatch = Stopwatch()..start();

    // Pixel loops are slow on the UI thread, so preprocessing runs in an isolate.
    final detectorInput = await Isolate.run(
      () => FaceDetector.preprocess(image),
    );
    var preprocess = stopwatch.elapsed;

    stopwatch.reset();
    final faces = await _detector.detect(detectorInput);
    final detect = stopwatch.elapsed;

    stopwatch.reset();
    final crops = faces.isEmpty
        ? const []
        : await Isolate.run(
            () => [
              for (final face in faces)
                EmotionClassifier.preprocess(image, face.box),
            ],
          );
    preprocess += stopwatch.elapsed;

    stopwatch.reset();
    final results = <FaceEmotion>[];
    for (var i = 0; i < faces.length; i++) {
      results.add(FaceEmotion(faces[i], await _classifier.classify(crops[i])));
    }
    final classify = stopwatch.elapsed;

    return PipelineResult(
      faces: results,
      preprocess: preprocess,
      detect: detect,
      classify: classify,
    );
  }

  Future<void> close() async {
    await _detector.close();
    await _classifier.close();
  }
}
