"""
Run the detector on the LAPTOP, using the .onnx file and onnxruntime (CPU).

Why this exists: it proves common/yolo.py is correct without involving the
Nano at all. If boxes are wrong here, the bug is in our numpy code. If they
are right here but wrong on the Nano, the bug is in the TensorRT side.

    python laptop/run_onnx.py                       # bus.jpg, 416 model
    python laptop/run_onnx.py some.jpg models/yolov8n_320.onnx
"""
import os
import sys
import time

import cv2
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.yolo import preprocess, postprocess, draw, COCO  # noqa: E402

image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "bus.jpg")
model_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "models", "yolov8n_416.onnx")

if not os.path.isfile(model_path):
    sys.exit("model not found: %s\nexport it first, see the README (models are not in the repository)" % model_path)
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0]
size = inp.shape[2]                       # the 416 baked in at export time
print("model input ", inp.name, inp.shape)

img = cv2.imread(image_path)
if img is None:
    sys.exit("could not read image: %s  (pass a path: python3 %s some.jpg)" % (image_path, sys.argv[0]))              # [H, W, 3] uint8 BGR
x, meta = preprocess(img, size)           # [1, 3, S, S] float32

t0 = time.time()
output = session.run(None, {inp.name: x})[0]   # [1, 84, N]
ms = (time.time() - t0) * 1000
print("model output", output.shape, " inference %.1f ms (laptop CPU)" % ms)

boxes, scores, class_ids = postprocess(output, meta)
for b, s, c in zip(boxes, scores, class_ids):
    print("  %-10s %.3f  [%4d %4d %4d %4d]" % (COCO[c], s, b[0], b[1], b[2], b[3]))

os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
out_path = os.path.join(ROOT, "out", "laptop_" + os.path.basename(image_path))
cv2.imwrite(out_path, draw(img, boxes, scores, class_ids))
print("saved", out_path)
