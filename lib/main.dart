import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'ml/emotion_pipeline.dart';
import 'ui/photo_screen.dart';
import 'ui/pipeline_controller.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const EmotionApp());
}

class EmotionApp extends StatelessWidget {
  const EmotionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ONNX Emotions',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.dark,
        ),
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _controller = PipelineController();

  @override
  void initState() {
    super.initState();
    _controller.load();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: _controller,
      builder: (context, _) {
        final config = _controller.config;
        return Scaffold(
          appBar: AppBar(
            title: const Text('ONNX Emotions'),
            bottom: PreferredSize(
              preferredSize: const Size.fromHeight(48),
              child: _ConfigBar(
                config: config,
                onChanged: _controller.isLoading ? null : _controller.load,
              ),
            ),
          ),
          body: _body(),
        );
      },
    );
  }

  Widget _body() {
    if (_controller.error case final error?) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text('Failed to load models:\n$error'),
        ),
      );
    }
    return PhotoScreen(controller: _controller);
  }
}

class _ConfigBar extends StatelessWidget {
  const _ConfigBar({required this.config, required this.onChanged});

  final PipelineConfig config;
  final ValueChanged<PipelineConfig>? onChanged;

  @override
  Widget build(BuildContext context) {
    final onChanged = this.onChanged;
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 0, 12, 8),
      child: Row(
        children: [
          FilterChip(
            label: const Text('INT8 model'),
            selected: config.int8,
            onSelected: onChanged == null
                ? null
                : (v) => onChanged(config.copyWith(int8: v)),
          ),
          const SizedBox(width: 8),
          FilterChip(
            label: Text(PipelineConfig.acceleratorName),
            selected: config.accelerated,
            onSelected: onChanged == null
                ? null
                : (v) => onChanged(config.copyWith(accelerated: v)),
          ),
          const Spacer(),
          if (onChanged == null)
            const SizedBox.square(
              dimension: 18,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
        ],
      ),
    );
  }
}
