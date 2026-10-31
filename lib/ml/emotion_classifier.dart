import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

/// FER+ output classes, in the order the model emits them.
enum Emotion {
  neutral('Neutral', '😐'),
  happiness('Happy', '😄'),
  surprise('Surprised', '😮'),
  sadness('Sad', '😢'),
  anger('Angry', '😠'),
  disgust('Disgusted', '🤢'),
  fear('Afraid', '😨'),
  contempt('Contempt', '😒');

  const Emotion(this.label, this.emoji);

  final String label;
  final String emoji;
}

/// Emotion FER+: an 8-class facial expression model from the ONNX Model Zoo.
///
/// Input:  float32 [1, 1, 64, 64]  grayscale, raw 0..255 (no normalization!)
/// Output: float32 [1, 8]          logits, one per [Emotion]
class EmotionClassifier {
  EmotionClassifier._(this._session);

  static const fp32Asset = 'assets/models/emotion_ferplus.onnx';
  static const int8Asset = 'assets/models/emotion_ferplus_int8.onnx';
  static const inputSize = 64;

  final OrtSession _session;

  static Future<EmotionClassifier> load(
    OnnxRuntime ort, {
    bool int8 = false,
    OrtSessionOptions? options,
  }) async {
    final asset = int8 ? int8Asset : fp32Asset;
    return EmotionClassifier._(
      await ort.createSessionFromAsset(asset, options: options),
    );
  }

  /// Runs the model on a tensor produced by [preprocess] and returns
  /// probabilities indexed by [Emotion.index].
  Future<List<double>> classify(Float32List input) async {
    final tensor = await OrtValue.fromList(input, [1, 1, inputSize, inputSize]);
    // Read names from the session instead of hardcoding 'Input3' and
    // 'Plus692_Output_0', which come from the original CNTK export.
    final outputs = await _session.run({_session.inputNames.first: tensor});
    try {
      final logits = await outputs[_session.outputNames.first]!
          .asFlattenedList();
      return softmax(logits.cast<num>());
    } finally {
      await tensor.dispose();
      for (final value in outputs.values) {
        await value.dispose();
      }
    }
  }

  Future<void> close() => _session.close();

  /// Crop the face, convert to grayscale and resize to 64x64.
  ///
  /// [box] is normalized to 0..1 relative to [image].
  static Float32List preprocess(img.Image image, Rect box) {
    final x = (box.left * image.width).round();
    final y = (box.top * image.height).round();
    final crop = img.copyCrop(
      image,
      x: x,
      y: y,
      width: math.max(1, (box.width * image.width).round()),
      height: math.max(1, (box.height * image.height).round()),
    );
    final resized = img.copyResize(
      crop,
      width: inputSize,
      height: inputSize,
      interpolation: img.Interpolation.linear,
    );
    final rgb = resized.getBytes(order: img.ChannelOrder.rgb);
    final tensor = Float32List(inputSize * inputSize);
    for (var i = 0; i < tensor.length; i++) {
      // ITU-R 601 luma, the same formula PIL uses for convert('L').
      tensor[i] =
          (rgb[i * 3] * 299 + rgb[i * 3 + 1] * 587 + rgb[i * 3 + 2] * 114) /
          1000;
    }
    return tensor;
  }

  static List<double> softmax(List<num> logits) {
    final maxLogit = logits.map((l) => l.toDouble()).reduce(math.max);
    final exps = [for (final l in logits) math.exp(l - maxLogit)];
    final sum = exps.reduce((a, b) => a + b);
    return [for (final e in exps) e / sum];
  }
}
