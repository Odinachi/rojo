import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:onnx_emotion_demo/camera/frame_converter.dart';

/// A 2x1 BGRA frame: red on the left, blue on the right, with row padding.
RawFrame redBlueFrame({int rotation = 0, bool mirror = false}) => RawFrame(
  width: 2,
  height: 1,
  format: FrameFormat.bgra8888,
  planes: [
    Uint8List.fromList([0, 0, 255, 255, 255, 0, 0, 255, 9, 9, 9, 9]),
  ],
  rowStrides: [12],
  pixelStrides: [4],
  rotation: rotation,
  mirror: mirror,
);

List<int> rgbAt(img.Image image, int x, int y) {
  final p = image.getPixel(x, y);
  return [p.r.toInt(), p.g.toInt(), p.b.toInt()];
}

void main() {
  test('BGRA is converted to RGB respecting row stride', () {
    final image = convertFrame(redBlueFrame());
    expect([image.width, image.height], [2, 1]);
    expect(rgbAt(image, 0, 0), [255, 0, 0]);
    expect(rgbAt(image, 1, 0), [0, 0, 255]);
  });

  test('rotation is clockwise', () {
    final image = convertFrame(redBlueFrame(rotation: 90));
    expect([image.width, image.height], [1, 2]);
    expect(rgbAt(image, 0, 0), [255, 0, 0]); // left edge became the top
  });

  test('mirror flips horizontally', () {
    final image = convertFrame(redBlueFrame(mirror: true));
    expect(rgbAt(image, 0, 0), [0, 0, 255]);
  });

  test('YUV420 gray and saturated colors convert correctly', () {
    // 2x2 frame, one chroma sample: Y=128, U=V=128 -> mid gray.
    final gray = convertFrame(
      RawFrame(
        width: 2,
        height: 2,
        format: FrameFormat.yuv420,
        planes: [
          Uint8List.fromList([128, 128, 128, 128]),
          Uint8List.fromList([128]),
          Uint8List.fromList([128]),
        ],
        rowStrides: [2, 1, 1],
        pixelStrides: [1, 1, 1],
      ),
    );
    expect(rgbAt(gray, 1, 1), [128, 128, 128]);

    // Pure red in BT.601 full range is roughly Y=76, U=85, V=255.
    final red = convertFrame(
      RawFrame(
        width: 2,
        height: 2,
        format: FrameFormat.yuv420,
        planes: [
          Uint8List.fromList([76, 76, 76, 76]),
          Uint8List.fromList([85]),
          Uint8List.fromList([255]),
        ],
        rowStrides: [2, 1, 1],
        pixelStrides: [1, 1, 1],
      ),
    );
    final [r, g, b] = rgbAt(red, 0, 0);
    expect(r, greaterThan(240));
    expect(g, lessThan(15));
    expect(b, lessThan(15));
  });
}
