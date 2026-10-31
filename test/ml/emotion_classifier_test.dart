import 'dart:ui' show Rect;

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:onnx_emotion_demo/ml/emotion_classifier.dart';

void main() {
  test('Emotion order matches the FER+ output layout', () {
    expect(Emotion.values.map((e) => e.name), [
      'neutral',
      'happiness',
      'surprise',
      'sadness',
      'anger',
      'disgust',
      'fear',
      'contempt',
    ]);
  });

  group('EmotionClassifier.preprocess', () {
    test('crops the box and keeps raw 0..255 grayscale values', () {
      // Left half white, right half black.
      final image = img.Image(width: 200, height: 100);
      img.fillRect(
        image,
        x1: 0,
        y1: 0,
        x2: 99,
        y2: 99,
        color: img.ColorRgb8(255, 255, 255),
      );

      final left = EmotionClassifier.preprocess(
        image,
        const Rect.fromLTWH(0, 0, 0.4, 1),
      );
      final right = EmotionClassifier.preprocess(
        image,
        const Rect.fromLTWH(0.6, 0, 0.4, 1),
      );

      expect(left.length, 64 * 64);
      expect(left.every((v) => v == 255), isTrue);
      expect(right.every((v) => v == 0), isTrue);
    });

    test('uses ITU-R 601 luma weights', () {
      final image = img.Image(width: 10, height: 10)
        ..clear(img.ColorRgb8(255, 0, 0));
      final tensor = EmotionClassifier.preprocess(
        image,
        const Rect.fromLTWH(0, 0, 1, 1),
      );
      expect(tensor.first, closeTo(255 * 0.299, 1e-3));
    });
  });

  test('softmax sums to 1 and preserves the argmax', () {
    final probs = EmotionClassifier.softmax(<num>[1, 5, 2, 1000 - 999]);
    expect(probs.reduce((a, b) => a + b), closeTo(1, 1e-9));
    expect(probs.indexOf(probs.reduce((a, b) => a > b ? a : b)), 1);
  });
}
