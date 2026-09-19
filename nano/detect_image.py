"""
Run the detector on the NANO, on one image, using the TensorRT engine.

    python3 nano/detect_image.py                                  # bus.jpg, 416 fp16 engine
    python3 nano/detect_image.py bus.jpg engines/yolov8n_320_fp16.engine

Same preprocess/postprocess as laptop/run_onnx.py; only the middle step (who
runs the network) is different. It also times each stage separately, because
"the model runs at X FPS" and "my program runs at X FPS" are different claims:
the numpy code on the Nano's slow CPU costs real time too.
"""
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.yolo import preprocess, postprocess, draw, COCO  # noqa: E402
from nano.trt_runner import TrtRunner                         # noqa: E402

image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "bus.jpg")
engine_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "engines", "yolov8n_416_fp16.engine")
RUNS = 50

runner = TrtRunner(engine_path)
img = cv2.imread(image_path)
if img is None:
    sys.exit("could not read image: %s  (pass a path: python3 %s some.jpg)" % (image_path, sys.argv[0]))
print("engine %s  input %s" % (os.path.basename(engine_path), (runner.input_shape,)))

# The first few GPU runs are slow (kernels get loaded, clocks ramp up), so
# throw some away before timing anything.
x, meta = preprocess(img, runner.size)
for _ in range(10):
    runner.infer(x)

t_pre = t_inf = t_post = 0.0
for _ in range(RUNS):
    t0 = time.time()
    x, meta = preprocess(img, runner.size)
    t1 = time.time()
    output = runner.infer(x)
    t2 = time.time()
    boxes, scores, class_ids = postprocess(output, meta)
    t3 = time.time()
    t_pre += t1 - t0
    t_inf += t2 - t1
    t_post += t3 - t2

pre, inf, post = [t * 1000.0 / RUNS for t in (t_pre, t_inf, t_post)]
total = pre + inf + post
print("per frame, average of %d runs:" % RUNS)
print("  preprocess  (CPU, numpy/cv2) %6.1f ms" % pre)
print("  inference   (GPU, TensorRT)  %6.1f ms" % inf)
print("  postprocess (CPU, numpy)     %6.1f ms" % post)
print("  total                        %6.1f ms  = %.1f FPS end to end" % (total, 1000.0 / total))

for b, s, c in zip(boxes, scores, class_ids):
    print("  %-10s %.3f  [%4d %4d %4d %4d]" % (COCO[c], s, b[0], b[1], b[2], b[3]))

out_dir = os.path.join(ROOT, "out")
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
out_path = os.path.join(out_dir, "nano_" + os.path.basename(image_path))
cv2.imwrite(out_path, draw(img, boxes, scores, class_ids))
print("saved", out_path)
runner.close()
