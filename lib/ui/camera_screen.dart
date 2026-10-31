import 'dart:io' show Platform;
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../camera/frame_converter.dart';
import '../ml/emotion_pipeline.dart';
import 'face_overlay.dart';
import 'pipeline_controller.dart';
import 'stats_panel.dart';

/// Live camera feed with emotion boxes drawn on top.
///
/// Frames arrive faster than we can process them, so we only take a new frame
/// once the previous one has finished (frames in between are dropped).
class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key, required this.controller});

  final PipelineController controller;

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  List<CameraDescription> _cameras = const [];
  CameraDescription? _currentDescription;
  CameraController? _camera;
  Object? _error;
  PipelineResult? _result;
  double? _fps;
  bool _busy = false;
  DateTime? _lastFrameAt;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _init();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stop();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) {
      _stop();
    } else if (state == AppLifecycleState.resumed && _camera == null) {
      _start(_currentDescription);
    }
  }

  Future<void> _init() async {
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) throw StateError('No camera found');
      // Default to the selfie camera: that's what an audience demo wants.
      final front = _cameras.where(
        (c) => c.lensDirection == CameraLensDirection.front,
      );
      await _start(front.isNotEmpty ? front.first : _cameras.first);
    } catch (e) {
      if (mounted) setState(() => _error = e);
    }
  }

  Future<void> _start(CameraDescription? description) async {
    if (description == null) return;
    _currentDescription = description;
    final camera = CameraController(
      description,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: Platform.isIOS
          ? ImageFormatGroup.bgra8888
          : ImageFormatGroup.yuv420,
    );
    try {
      await camera.initialize();
      await camera.lockCaptureOrientation(DeviceOrientation.portraitUp);
      await camera.startImageStream(_onFrame);
    } catch (e) {
      await camera.dispose();
      if (mounted) setState(() => _error = e);
      return;
    }
    if (!mounted) {
      await camera.dispose();
      return;
    }
    setState(() {
      _camera = camera;
      _error = null;
      _result = null;
      _fps = null;
    });
  }

  Future<void> _stop() async {
    final camera = _camera;
    _camera = null;
    if (camera == null) return;
    if (camera.value.isStreamingImages) await camera.stopImageStream();
    await camera.dispose();
  }

  Future<void> _switchCamera() async {
    final current = _currentDescription;
    if (_cameras.length < 2 || current == null) return;
    final next = _cameras[(_cameras.indexOf(current) + 1) % _cameras.length];
    await _stop();
    setState(() {});
    await _start(next);
  }

  Future<void> _onFrame(CameraImage image) async {
    final pipeline = widget.controller.pipeline;
    final description = _currentDescription;
    if (_busy || pipeline == null || description == null) return;
    _busy = true;
    try {
      // The UI is locked to portrait. Android delivers landscape sensor frames
      // that need rotating; iOS already delivers them upright. Android's
      // preview is mirrored for the front camera, so mirror frames to match.
      final frame = RawFrame.fromCameraImage(
        image,
        rotation: image.width > image.height
            ? description.sensorOrientation
            : 0,
        mirror:
            Platform.isAndroid &&
            description.lensDirection == CameraLensDirection.front,
      );
      final upright = await Isolate.run(() => convertFrame(frame));
      final result = await pipeline.run(upright);

      final now = DateTime.now();
      final last = _lastFrameAt;
      _lastFrameAt = now;
      if (!mounted) return;
      setState(() {
        _result = result;
        if (last != null) {
          final fps = 1000 / now.difference(last).inMilliseconds.clamp(1, 1e6);
          _fps = _fps == null ? fps : _fps! * 0.8 + fps * 0.2;
        }
      });
    } catch (e) {
      debugPrint('Frame failed: $e');
    } finally {
      _busy = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    final camera = _camera;
    final Widget preview;
    if (_error != null) {
      preview = Padding(
        padding: const EdgeInsets.all(24),
        child: Text('Camera unavailable:\n$_error'),
      );
    } else if (camera == null || !camera.value.isInitialized) {
      preview = const CircularProgressIndicator();
    } else {
      preview = AspectRatio(
        aspectRatio: 1 / camera.value.aspectRatio,
        child: Stack(
          fit: StackFit.expand,
          children: [
            CameraPreview(camera),
            if (_result != null) FaceOverlay(faces: _result!.faces),
          ],
        ),
      );
    }

    return Column(
      children: [
        Expanded(
          child: Stack(
            children: [
              Center(child: preview),
              if (_cameras.length > 1)
                Positioned(
                  right: 12,
                  bottom: 12,
                  child: FloatingActionButton.small(
                    onPressed: _switchCamera,
                    tooltip: 'Switch camera',
                    child: const Icon(Icons.cameraswitch),
                  ),
                ),
            ],
          ),
        ),
        StatsPanel(
          config: widget.controller.config,
          result: _result,
          fps: _fps,
        ),
      ],
    );
  }
}
