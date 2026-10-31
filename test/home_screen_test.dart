import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:onnx_emotion_demo/main.dart';
import 'package:onnx_emotion_demo/ml/emotion_pipeline.dart';
import 'package:onnx_emotion_demo/ui/pipeline_controller.dart';

void main() {
  testWidgets('config chips reload the pipeline with the new config', (
    tester,
  ) async {
    final requested = <PipelineConfig>[];
    final controller = PipelineController(
      loader: (config) async {
        requested.add(config);
        throw StateError('no native runtime in tests');
      },
    );
    await tester.pumpWidget(EmotionApp(controller: controller));
    await tester.pump();

    FilterChip chip(String label) =>
        tester.widget<FilterChip>(find.widgetWithText(FilterChip, label));
    expect(chip('INT8 model').selected, isFalse);
    expect(find.textContaining('Failed to load models'), findsOneWidget);

    await tester.tap(find.text('INT8 model'));
    await tester.pump();
    expect(chip('INT8 model').selected, isTrue);

    await tester.tap(find.text(PipelineConfig.acceleratorName));
    await tester.pump();

    expect(requested, const [
      PipelineConfig(),
      PipelineConfig(int8: true),
      PipelineConfig(int8: true, accelerated: true),
    ]);
  });
}
