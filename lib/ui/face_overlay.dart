import 'package:flutter/material.dart';

import '../ml/emotion_classifier.dart';
import '../ml/emotion_pipeline.dart';

const emotionColors = {
  Emotion.neutral: Color(0xFFB0BEC5),
  Emotion.happiness: Color(0xFFFFD54F),
  Emotion.surprise: Color(0xFF4FC3F7),
  Emotion.sadness: Color(0xFF7986CB),
  Emotion.anger: Color(0xFFE57373),
  Emotion.disgust: Color(0xFF81C784),
  Emotion.fear: Color(0xFFBA68C8),
  Emotion.contempt: Color(0xFFFF8A65),
};

/// Draws a labelled box over each face. Must be laid out at exactly the
/// size of the image the faces were detected in.
class FaceOverlay extends StatelessWidget {
  const FaceOverlay({super.key, required this.faces});

  final List<FaceEmotion> faces;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(painter: _FacePainter(faces), size: Size.infinite);
  }
}

class _FacePainter extends CustomPainter {
  _FacePainter(this.faces);

  final List<FaceEmotion> faces;

  @override
  void paint(Canvas canvas, Size size) {
    for (final face in faces) {
      final box = face.face.box;
      final rect = Rect.fromLTRB(
        box.left * size.width,
        box.top * size.height,
        box.right * size.width,
        box.bottom * size.height,
      );
      final color = emotionColors[face.top]!;
      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(8)),
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3,
      );

      final label = TextPainter(
        text: TextSpan(
          text:
              '${face.top.emoji} ${face.top.label} '
              '${(face.confidence * 100).round()}%',
          style: const TextStyle(
            color: Colors.black,
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final top = rect.top - label.height - 8 < 0
          ? rect.bottom + 4
          : rect.top - label.height - 8;
      final labelRect = Rect.fromLTWH(
        rect.left,
        top,
        label.width + 12,
        label.height + 4,
      );
      canvas.drawRRect(
        RRect.fromRectAndRadius(labelRect, const Radius.circular(6)),
        Paint()..color = color,
      );
      label.paint(canvas, labelRect.topLeft + const Offset(6, 2));
    }
  }

  @override
  bool shouldRepaint(_FacePainter old) => old.faces != faces;
}
