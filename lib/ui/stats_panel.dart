import 'package:flutter/material.dart';

import '../ml/emotion_classifier.dart';
import '../ml/emotion_pipeline.dart';
import 'face_overlay.dart';

/// Shows model/provider, per-stage timings and the full probability
/// distribution for the most confident face.
class StatsPanel extends StatelessWidget {
  const StatsPanel({
    super.key,
    required this.config,
    required this.result,
    this.fps,
  });

  final PipelineConfig config;
  final PipelineResult? result;
  final double? fps;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final result = this.result;
    String ms(Duration d) => '${d.inMilliseconds} ms';
    final face = result == null || result.faces.isEmpty
        ? null
        : result.faces.reduce((a, b) => a.face.score >= b.face.score ? a : b);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
      color: theme.colorScheme.surfaceContainer,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Wrap(
            spacing: 8,
            runSpacing: 4,
            children: [
              _Chip(Icons.memory, config.providerName),
              _Chip(Icons.data_object, config.modelName),
              if (result != null)
                _Chip(Icons.face, '${result.faces.length} face(s)'),
              if (fps != null)
                _Chip(Icons.speed, '${fps!.toStringAsFixed(1)} fps'),
            ],
          ),
          if (result != null) ...[
            const SizedBox(height: 8),
            Text(
              'pre ${ms(result.preprocess)} · detect ${ms(result.detect)} · '
              'classify ${ms(result.classify)} · total ${ms(result.total)}',
              style: theme.textTheme.bodySmall?.copyWith(
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
          ],
          if (face != null) ...[
            const SizedBox(height: 8),
            for (final emotion in Emotion.values)
              _ProbabilityBar(emotion, face.probabilities[emotion.index]),
          ],
        ],
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  const _Chip(this.icon, this.label);

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Chip(
      avatar: Icon(icon, size: 16),
      label: Text(label),
      visualDensity: VisualDensity.compact,
      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
    );
  }
}

class _ProbabilityBar extends StatelessWidget {
  const _ProbabilityBar(this.emotion, this.value);

  final Emotion emotion;
  final double value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 1.5),
      child: Row(
        children: [
          SizedBox(
            width: 112,
            child: Text(
              '${emotion.emoji} ${emotion.label}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: value,
                minHeight: 8,
                color: emotionColors[emotion],
                backgroundColor: Colors.white10,
              ),
            ),
          ),
          SizedBox(
            width: 44,
            child: Text(
              '${(value * 100).round()}%',
              textAlign: TextAlign.right,
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
        ],
      ),
    );
  }
}
