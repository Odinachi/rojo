import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:onnx_emotion_demo/ml/face_detector.dart';

void main() {
  group('FaceDetector.preprocess', () {
    test('produces an NCHW tensor normalized around 0', () {
      final image = img.Image(width: 640, height: 480)
        ..clear(img.ColorRgb8(255, 127, 0));
      final tensor = FaceDetector.preprocess(image);

      const plane = FaceDetector.inputWidth * FaceDetector.inputHeight;
      expect(tensor.length, 3 * plane);
      expect(tensor[0], closeTo(1.0, 1e-6)); // R plane
      expect(tensor[plane], closeTo(0.0, 1e-6)); // G plane
      expect(tensor[2 * plane], closeTo(-127 / 128, 1e-6)); // B plane
    });
  });

  group('FaceDetector.decode', () {
    test('filters by threshold and suppresses overlapping boxes', () {
      final scores = <num>[
        0.05, 0.95, // strong face
        0.10, 0.90, // same face, slightly shifted -> suppressed
        0.60, 0.40, // below threshold
        0.20, 0.80, // separate face
      ];
      final boxes = <num>[
        0.10,
        0.10,
        0.30,
        0.40,
        0.11,
        0.11,
        0.31,
        0.41,
        0.50,
        0.50,
        0.60,
        0.60,
        0.60,
        0.20,
        0.80,
        0.50,
      ];

      final faces = FaceDetector.decode(scores, boxes);

      expect(faces, hasLength(2));
      expect(faces[0].score, closeTo(0.95, 1e-9));
      expect(faces[1].box.left, closeTo(0.60, 1e-9));
    });

    test('clamps boxes into the image', () {
      final faces = FaceDetector.decode(
        <num>[0, 1],
        <num>[-0.2, -0.1, 1.3, 0.5],
      );
      expect(faces.single.box, const Rect.fromLTRB(0, 0, 1, 0.5));
    });
  });

  test('iou', () {
    const a = Rect.fromLTWH(0, 0, 2, 2);
    expect(FaceDetector.iou(a, a), 1);
    expect(
      FaceDetector.iou(a, const Rect.fromLTWH(1, 0, 2, 2)),
      closeTo(1 / 3, 1e-9),
    );
    expect(FaceDetector.iou(a, const Rect.fromLTWH(5, 5, 1, 1)), 0);
  });
}
