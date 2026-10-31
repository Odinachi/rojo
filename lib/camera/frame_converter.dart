import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

enum FrameFormat { bgra8888, yuv420 }

/// A plain copy of a [CameraImage] that can be sent to another isolate.
class RawFrame {
  const RawFrame({
    required this.width,
    required this.height,
    required this.format,
    required this.planes,
    required this.rowStrides,
    required this.pixelStrides,
    this.rotation = 0,
    this.mirror = false,
  });

  /// [rotation] is the clockwise rotation (0/90/180/270) that makes the frame
  /// upright. [mirror] flips it horizontally to match a mirrored preview.
  factory RawFrame.fromCameraImage(
    CameraImage image, {
    int rotation = 0,
    bool mirror = false,
  }) {
    final format = switch (image.format.group) {
      ImageFormatGroup.bgra8888 => FrameFormat.bgra8888,
      ImageFormatGroup.yuv420 => FrameFormat.yuv420,
      final other => throw UnsupportedError('Unsupported camera format $other'),
    };
    return RawFrame(
      width: image.width,
      height: image.height,
      format: format,
      planes: [for (final p in image.planes) p.bytes],
      rowStrides: [for (final p in image.planes) p.bytesPerRow],
      pixelStrides: [for (final p in image.planes) p.bytesPerPixel ?? 1],
      rotation: rotation,
      mirror: mirror,
    );
  }

  final int width;
  final int height;
  final FrameFormat format;
  final List<Uint8List> planes;
  final List<int> rowStrides;
  final List<int> pixelStrides;
  final int rotation;
  final bool mirror;
}

/// Converts a camera frame to an upright RGB image, oriented like the preview.
img.Image convertFrame(RawFrame frame) {
  final rgb = switch (frame.format) {
    FrameFormat.bgra8888 => _bgraToRgb(frame),
    FrameFormat.yuv420 => _yuv420ToRgb(frame),
  };
  var image = img.Image.fromBytes(
    width: frame.width,
    height: frame.height,
    bytes: rgb.buffer,
    numChannels: 3,
  );
  if (frame.rotation % 360 != 0) {
    image = img.copyRotate(image, angle: frame.rotation);
  }
  if (frame.mirror) {
    image = img.flipHorizontal(image);
  }
  return image;
}

Uint8List _bgraToRgb(RawFrame frame) {
  final src = frame.planes[0];
  final stride = frame.rowStrides[0];
  final out = Uint8List(frame.width * frame.height * 3);
  var o = 0;
  for (var y = 0; y < frame.height; y++) {
    var i = y * stride;
    for (var x = 0; x < frame.width; x++, i += 4) {
      out[o++] = src[i + 2];
      out[o++] = src[i + 1];
      out[o++] = src[i];
    }
  }
  return out;
}

Uint8List _yuv420ToRgb(RawFrame frame) {
  final yPlane = frame.planes[0];
  final uPlane = frame.planes[1];
  final vPlane = frame.planes[2];
  final yStride = frame.rowStrides[0];
  final uvStride = frame.rowStrides[1];
  final uvPixel = frame.pixelStrides[1];
  final out = Uint8List(frame.width * frame.height * 3);
  var o = 0;
  for (var y = 0; y < frame.height; y++) {
    for (var x = 0; x < frame.width; x++) {
      final luma = yPlane[y * yStride + x];
      final uv = (y >> 1) * uvStride + (x >> 1) * uvPixel;
      final u = uPlane[uv] - 128;
      final v = vPlane[uv] - 128;
      out[o++] = (luma + 1.402 * v).round().clamp(0, 255);
      out[o++] = (luma - 0.344136 * u - 0.714136 * v).round().clamp(0, 255);
      out[o++] = (luma + 1.772 * u).round().clamp(0, 255);
    }
  }
  return out;
}
