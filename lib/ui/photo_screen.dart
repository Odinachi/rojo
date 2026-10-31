import 'dart:isolate';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

import '../ml/emotion_pipeline.dart';
import 'face_overlay.dart';
import 'pipeline_controller.dart';
import 'stats_panel.dart';

/// Pick a photo (or use the bundled sample) and run the pipeline on it.
/// A reliable fallback for live demos when lighting or the camera misbehave.
class PhotoScreen extends StatefulWidget {
  const PhotoScreen({super.key, required this.controller});

  final PipelineController controller;

  @override
  State<PhotoScreen> createState() => _PhotoScreenState();
}

class _PhotoScreenState extends State<PhotoScreen> {
  static const _sampleAsset = 'assets/samples/astronaut.png';
  static const _maxSide = 1280;

  img.Image? _image;
  Uint8List? _displayBytes;
  PipelineResult? _result;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_rerun);
  }

  @override
  void dispose() {
    widget.controller.removeListener(_rerun);
    super.dispose();
  }

  Future<void> _loadSample() async {
    final data = await rootBundle.load(_sampleAsset);
    await _setImage(data.buffer.asUint8List());
  }

  Future<void> _pick(ImageSource source) async {
    final file = await ImagePicker().pickImage(source: source);
    if (file == null) return;
    await _setImage(await file.readAsBytes());
  }

  Future<void> _setImage(Uint8List bytes) async {
    setState(() => _busy = true);
    // Decode, apply EXIF rotation and downscale off the UI thread. We display
    // the re-encoded image so the overlay matches exactly what the model saw.
    final (image, display) = await Isolate.run(() {
      var image = img.bakeOrientation(img.decodeImage(bytes)!);
      final longest = math.max(image.width, image.height);
      if (longest > _maxSide) {
        final scale = _maxSide / longest;
        image = img.copyResize(
          image.convert(numChannels: 3),
          width: (image.width * scale).round(),
          height: (image.height * scale).round(),
        );
      }
      image = image.convert(numChannels: 3);
      return (image, img.encodeJpg(image, quality: 90));
    });
    _image = image;
    _displayBytes = display;
    _result = null;
    await _run();
  }

  void _rerun() {
    if (_image != null && widget.controller.pipeline != null) _run();
  }

  Future<void> _run() async {
    final pipeline = widget.controller.pipeline;
    final image = _image;
    if (pipeline == null || image == null) {
      setState(() => _busy = false);
      return;
    }
    setState(() => _busy = true);
    final result = await pipeline.run(image);
    if (!mounted) return;
    setState(() {
      _result = result;
      _busy = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    final image = _image;
    return Column(
      children: [
        Expanded(
          child: Center(
            child: image == null
                ? const Text('Pick a photo to detect emotions')
                : AspectRatio(
                    aspectRatio: image.width / image.height,
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        Image.memory(_displayBytes!, gaplessPlayback: true),
                        if (_result != null) FaceOverlay(faces: _result!.faces),
                        if (_busy)
                          const Center(child: CircularProgressIndicator()),
                      ],
                    ),
                  ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(8),
          child: Wrap(
            spacing: 8,
            alignment: WrapAlignment.center,
            children: [
              FilledButton.tonalIcon(
                onPressed: _busy ? null : () => _pick(ImageSource.gallery),
                icon: const Icon(Icons.photo_library),
                label: const Text('Gallery'),
              ),
              FilledButton.tonalIcon(
                onPressed: _busy ? null : () => _pick(ImageSource.camera),
                icon: const Icon(Icons.photo_camera),
                label: const Text('Take photo'),
              ),
              FilledButton.tonalIcon(
                onPressed: _busy ? null : _loadSample,
                icon: const Icon(Icons.image),
                label: const Text('Sample'),
              ),
            ],
          ),
        ),
        StatsPanel(config: widget.controller.config, result: _result),
      ],
    );
  }
}
