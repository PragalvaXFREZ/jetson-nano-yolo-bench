# Results: YOLOv8n on a Jetson Nano 4GB

A speed figure means little until it is clear what was inside the stopwatch. So this file measures the same model three ways, each one including more of the real system than the last, and then looks inside the fastest number to see what it is made of.

Board: Jetson Nano Developer Kit 4GB, JetPack 4.6 (L4T 32.6.1), TensorRT 8.0.1, MAXN power mode, headless, clocks not pinned with `jetson_clocks`. Measured 19 Sep 2026. Raw logs are in [`logs/`](logs/).

The Nano's clock was about 10 hours behind, so log timestamps read 03:xx for runs made at 13:xx.

## 1. The network alone

This is the number that usually gets quoted for a model: the TensorRT engine by itself, with the input already sitting in memory.

`bash nano/bench.sh` loads each finished engine with `trtexec`, warms it up for 1 s, and measures it for 10 s, one engine straight after another so the conditions match. The input is random numbers. That is fine for this model, because a convolution costs the same whatever the pixel values are.

| Engine | FPS | GPU ms per frame | Engine file |
|---|---|---|---|
| 416 px, FP32 | 29.8 | 33.2 | 30 MB |
| 416 px, FP16 | 38.6 | 25.6 | 17 MB |
| 320 px, FP16 | 58.5 | 16.9 | 11 MB |

Each row differs from the middle one in exactly one setting, so the first and second rows isolate precision, and the second and third isolate input size.

For the 416 FP16 engine: host-to-device copy 0.20 ms, device-to-host copy 0.12 ms, CPU enqueue time 10.0 ms mean. The copies are about 1% of the frame, because the Nano's CPU and GPU share one set of memory chips.

## 2. The whole program, one photo

Now the stopwatch includes preprocessing and postprocessing, the numpy code in `common/yolo.py` that turns a photo into the input tensor and the output tensor into boxes.

`python3 nano/detect_image.py` does 10 warm-up runs, then averages 50, with a separate timer on each stage.

| Stage | Runs on | 416 FP16 | 320 FP16 |
|---|---|---|---|
| preprocess (resize, pad, reorder) | CPU | 12.5 ms | 8.1 ms |
| inference (copy in, run engine, copy out) | GPU | 31.7 ms | 18.3 ms |
| postprocess (filter 3549 candidates, NMS) | CPU | 6.8 ms | 3.5 ms |
| **total** | | **51.0 ms = 19.6 FPS** | **29.9 ms = 33.4 FPS** |

38% of the 416 frame is CPU work in plain numpy, on one Cortex-A57 core. The engine alone runs at 38.6 FPS and the program at 19.6, so half the speed is lost outside the neural network.

The same engine takes 31.7 ms when called from the Python loop and 25.6 ms inside `trtexec`. The likely cause is the enqueue time above: the CPU enqueues 185 layers per frame, and a simple loop does not overlap that with GPU work. This was not isolated by an experiment.

## 3. Live video

Finally the full system, which is what someone would actually deploy.

`python3 nano/detect_stream.py <laptop-ip>` takes a laptop webcam to the Nano as H.264 over RTP, runs detection, sends annotated frames back as JPEG over RTP, and writes a recording on the Nano. The run was 90 s and 1080 frames, with the first 30 excluded. Per-frame data: [`logs/live_fps.csv`](logs/live_fps.csv).

| Part of each frame | Time |
|---|---|
| preprocess (CPU) | 10.9 ms |
| inference (GPU) | 28.4 ms |
| postprocess (CPU) | 6.2 ms |
| video decode, drawing, two JPEG encodes (CPU) | about 36 ms |
| **whole loop** | **81 ms = 12.3 FPS** |

The GPU is busy for about a third of each frame. The accelerator is the least loaded part of the system, and the four ARM cores are the bottleneck.

The first inference of the run took 1339 ms and the second 43 ms, which is why every measurement here discards warm-up runs.

## 4. Inside the 25.6 ms

The network-only number looked clean, so it is worth opening up. `trtexec --dumpProfile --separateProfileRun` times every layer of the 416 FP16 engine separately. Profiling adds overhead, so the shares are reliable and the absolute milliseconds are slightly inflated. Full table: [`logs/profile_416_fp16.log`](logs/profile_416_fp16.log).

After TensorRT's layer fusion, the engine runs as 185 layers.

| Kind of layer | Count | Time | Share |
|---|---|---|---|
| Convolutions | 61 | 15.3 ms | 60% |
| Activations and other pointwise layers | 62 | 5.3 ms | 21% |
| Reshape, copy, split, resize, softmax, pooling | 62 | 4.9 ms | 19% |

| Part of the network | Time | Share |
|---|---|---|
| Backbone (model.0 to 9) | 11.2 ms | 44% |
| Neck (model.10 to 21) | 4.9 ms | 19% |
| Head (model.22) | 9.4 ms | 37% |

The slowest single layer is not a convolution. It is the fused SiLU activation after the first layer, `PWN(PWN(/model.0/act/Sigmoid), /model.0/act/Mul)`, at 1.50 ms (5.9%). It operates on the largest tensor in the network, 16 x 208 x 208.

## 5. Fixed cost per frame (estimate)

Going from 416 to 320 pixels leaves 0.59 times as many pixels, yet the time only fell to 0.66. Something in each frame does not shrink with the image. Two input sizes give two equations for a fixed part `a` and a per-pixel cost `b`:

```text
a + b x 173,056 = 25.6 ms     (416 x 416)
a + b x 102,400 = 16.9 ms     (320 x 320)
```

Subtracting one from the other cancels `a`, which gives `b`, and substituting back gives `a` = 4.3 ms. That is about 23 microseconds for each of the 185 layers, a plausible cost for a kernel launch from a slow CPU, and a pixel-dependent part of 21.3 ms at 416 and 12.6 ms at 320. That is why 0.59 times the pixels gave 0.66 times the time.

This is an estimate from two data points, and it assumes the scalable part is exactly proportional to pixel count. A third input size would test that.

## 6. Open question: why FP16 gave only 1.29x

After removing the fixed 4.3 ms, the part that could scale went from 28.9 ms (FP32) to 21.3 ms (FP16), a factor of 1.36.

FP16 is often expected to be close to 2x. One explanation for the shortfall is tempting and does not hold up: "the pointwise layers are limited by memory bandwidth, and FP16 does not help with that".

1. Halving the precision halves the bytes moved, so a bandwidth-limited layer would also speed up by about 2x.
2. The slowest layer reads and writes about 2.8 MB in FP16. At the Nano's 25.6 GB/s that is about 0.1 ms. It measured 1.5 ms.

Candidates consistent with the data: TensorRT kept some layers in FP32 where that timed faster (`--fp16` only permits FP16), format conversion layers between precisions, FP16 kernels that are not well tuned for this 2014 GPU design, and layers too small to speed up (105 of the 185 take under 0.1 ms, where launch overhead dominates).

The data here cannot choose between them. A per-layer profile of the FP32 engine, set beside the FP16 one, would show exactly which layers sped up and which did not. It was not taken before the board was returned.

## 7. Accuracy cost of the smaller input

The 320 engine is 1.5 times faster. This is what that buys and what it costs. Same photo, same thresholds (confidence 0.25, IoU 0.7):

| | 416 px | 320 px |
|---|---|---|
| Objects found | 5 | 4 |
| Highest person score | 0.883 | 0.846 |
| Bus score | 0.854 | 0.835 |
| Half-visible person at the left edge | 0.29 | not found |

## 8. Correctness of the decoder

None of the timings above mean anything if the postprocessing is subtly wrong, because wrong boxes still look plausible. So `common/yolo.py` was checked on the laptop against the Ultralytics pipeline using the same ONNX file: the same five detections, scores equal to three decimals, boxes within one pixel (Ultralytics clips boxes to the image, this code does not). On the Nano through TensorRT FP16 the detections are the same, with scores differing in the third decimal.
