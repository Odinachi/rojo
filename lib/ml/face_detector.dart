import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' show Rect;

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

/// A detected face. [box] is normalized to 0..1 relative to the input image.
class Face {
  const Face(this.box, this.score);

  final Rect box;
  final double score;
}

/// UltraFace RFB-320: a 1.3 MB face detector from the ONNX Model Zoo.
///
/// Input:  `input`  float32 [1, 3, 240, 320]  RGB, (pixel - 127) / 128
/// Output: `scores` float32 [1, 4420, 2]      [background, face] per anchor
///         `boxes`  float32 [1, 4420, 4]      normalized [x1, y1, x2, y2]
class FaceDetector {
  FaceDetector._(this._session);

  static const asset = 'assets/models/ultraface_rfb_320.onnx';
  static const inputWidth = 320;
  static const inputHeight = 240;

  final OrtSession _session;

  static Future<FaceDetector> load(
    OnnxRuntime ort, {
    OrtSessionOptions? options,
  }) async {
    return FaceDetector._(
      await ort.createSessionFromAsset(asset, options: options),
    );
  }

  /// Runs the model on a tensor produced by [preprocess].
  Future<List<Face>> detect(Float32List input, {double threshold = 0.7}) async {
    final tensor = await OrtValue.fromList(input, [
      1,
      3,
      inputHeight,
      inputWidth,
    ]);
    final outputs = await _session.run({'input': tensor});
    try {
      final scores = await outputs['scores']!.asFlattenedList();
      final boxes = await outputs['boxes']!.asFlattenedList();
      return decode(
        scores.cast<num>(),
        boxes.cast<num>(),
        threshold: threshold,
      );
    } finally {
      await tensor.dispose();
      for (final value in outputs.values) {
        await value.dispose();
      }
    }
  }

  Future<void> close() => _session.close();

  /// Resize to 320x240, normalize to roughly -1..1 and reorder HWC -> NCHW.
  static Float32List preprocess(img.Image image) {
    final resized = img.copyResize(
      image,
      width: inputWidth,
      height: inputHeight,
      interpolation: img.Interpolation.linear,
    );
    final rgb = resized.getBytes(order: img.ChannelOrder.rgb);
    const plane = inputWidth * inputHeight;
    final tensor = Float32List(3 * plane);
    for (var i = 0; i < plane; i++) {
      tensor[i] = (rgb[i * 3] - 127) / 128;
      tensor[plane + i] = (rgb[i * 3 + 1] - 127) / 128;
      tensor[2 * plane + i] = (rgb[i * 3 + 2] - 127) / 128;
    }
    return tensor;
  }

  /// Keep confident anchors, then drop overlapping boxes (non-max suppression).
  static List<Face> decode(
    List<num> scores,
    List<num> boxes, {
    double threshold = 0.7,
    double iouThreshold = 0.3,
  }) {
    final candidates = <Face>[];
    for (var i = 0; i < scores.length ~/ 2; i++) {
      final score = scores[i * 2 + 1].toDouble();
      if (score < threshold) continue;
      final box = Rect.fromLTRB(
        boxes[i * 4].toDouble().clamp(0, 1),
        boxes[i * 4 + 1].toDouble().clamp(0, 1),
        boxes[i * 4 + 2].toDouble().clamp(0, 1),
        boxes[i * 4 + 3].toDouble().clamp(0, 1),
      );
      candidates.add(Face(box, score));
    }
    candidates.sort((a, b) => b.score.compareTo(a.score));

    final kept = <Face>[];
    for (final face in candidates) {
      if (kept.every((k) => iou(k.box, face.box) <= iouThreshold)) {
        kept.add(face);
      }
    }
    return kept;
  }

  static double iou(Rect a, Rect b) {
    final overlap = a.intersect(b);
    if (overlap.width <= 0 || overlap.height <= 0) return 0;
    final intersection = overlap.width * overlap.height;
    final union = a.width * a.height + b.width * b.height - intersection;
    return union <= 0 ? 0 : intersection / math.max(union, 1e-9);
  }
}
