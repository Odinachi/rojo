import 'package:flutter/foundation.dart';

import '../ml/emotion_pipeline.dart';

/// Owns the loaded [EmotionPipeline] and reloads it when the config changes.
class PipelineController extends ChangeNotifier {
  PipelineConfig _config = const PipelineConfig();
  EmotionPipeline? _pipeline;
  Object? _error;
  int _generation = 0;

  PipelineConfig get config => _config;
  EmotionPipeline? get pipeline => _pipeline;
  Object? get error => _error;
  bool get isLoading => _pipeline == null && _error == null;

  Future<void> load([PipelineConfig? config]) async {
    final generation = ++_generation;
    _config = config ?? _config;
    final previous = _pipeline;
    _pipeline = null;
    _error = null;
    notifyListeners();
    await previous?.close();

    try {
      final pipeline = await EmotionPipeline.load(_config);
      if (generation != _generation) {
        await pipeline.close();
        return;
      }
      _pipeline = pipeline;
    } catch (e) {
      if (generation != _generation) return;
      _error = e;
    }
    notifyListeners();
  }

  @override
  void dispose() {
    _generation++;
    _pipeline?.close();
    super.dispose();
  }
}
