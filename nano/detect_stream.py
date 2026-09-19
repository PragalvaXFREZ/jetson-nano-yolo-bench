"""
Live detection on the NANO, on video that comes from the laptop's webcam.

    laptop webcam --H.264/RTP, UDP port 5000--> NANO: decode -> detect -> draw
    laptop screen <--JPEG/RTP,  UDP port 5001-- NANO

Start order:
    1. nano:    python3 nano/detect_stream.py <laptop-ip>
    2. laptop:  bash laptop/view_result.sh          (window with the boxes)
    3. laptop:  bash laptop/stream_webcam.sh        (starts sending the camera)

It also saves what it sees to out/live.avi and a timing log to bench/logs/live_fps.csv,
so there is evidence to look at after the hardware is gone. Stops by itself
after SECONDS, or with Ctrl+C.

The GStreamer strings below are pipelines: elements joined by "!", each one
handing its output to the next, like a shell pipe for video.
"""
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.yolo import preprocess, postprocess, draw  # noqa: E402
from nano.trt_runner import TrtRunner                  # noqa: E402

laptop_ip = sys.argv[1]
engine_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "engines", "yolov8n_416_fp16.engine")
SECONDS = 90

# RECEIVE. udpsrc: take packets from the network. rtph264depay: unwrap them
# back into an H.264 stream. avdec_h264: decompress to raw frames (on the CPU;
# this Nano image lacks the converter needed for its hardware decoder).
# appsink drop=true max-buffers=1: if we are slower than the camera, throw old
# frames away instead of queueing them, so the picture never lags behind.
recv = ("udpsrc port=5000 caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\" "
        "! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR "
        "! appsink drop=true max-buffers=1 sync=false")
# SEND BACK. JPEG per frame is cheap to encode on the Nano's CPU.
# format=I420 matters: JPEG-over-RTP can only carry the YUV 4:2:0 colour layout.
# Without it the encoder picked another layout and rtpjpegpay silently dropped
# every frame (the first version of this script had exactly that bug).
send = ("appsrc ! videoconvert ! video/x-raw,format=I420 ! jpegenc quality=70 ! rtpjpegpay "
        "! udpsink host=%s port=5001 sync=false" % laptop_ip)

runner = TrtRunner(engine_path)
print("waiting for video on UDP 5000 ... (start laptop/stream_webcam.sh)")
cap = cv2.VideoCapture(recv, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    sys.exit("could not open the receive pipeline")

out_net = out_file = None
log = open(os.path.join(ROOT, "bench", "logs", "live_fps.csv"), "w")
log.write("t_s,pre_ms,infer_ms,post_ms,loop_fps,detections\n")
start = last = time.time()
frames = 0
try:
    while time.time() - start < SECONDS:
        ok, frame = cap.read()                       # [H, W, 3] BGR, blocks until a frame arrives
        if not ok:
            break
        if out_net is None:                          # now we know the frame size
            h, w = frame.shape[:2]
            out_net = cv2.VideoWriter(send, cv2.CAP_GSTREAMER, 0, 30.0, (w, h))
            out_file = cv2.VideoWriter(os.path.join(ROOT, "out", "live.avi"),
                                       cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (w, h))
            start = last = time.time()
            print("receiving %dx%d" % (w, h))

        t0 = time.time()
        x, meta = preprocess(frame, runner.size)
        t1 = time.time()
        output = runner.infer(x)
        t2 = time.time()
        boxes, scores, class_ids = postprocess(output, meta)
        t3 = time.time()

        now = time.time()
        fps = 1.0 / max(now - last, 1e-6)            # whole loop, including waiting for the camera
        last = now
        draw(frame, boxes, scores, class_ids)
        cv2.putText(frame, "%.1f FPS  infer %.0f ms" % (fps, (t2 - t1) * 1000), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out_net.write(frame)
        out_file.write(frame)

        log.write("%.2f,%.1f,%.1f,%.1f,%.1f,%d\n" % (now - start, (t1 - t0) * 1000, (t2 - t1) * 1000,
                                                     (t3 - t2) * 1000, fps, len(scores)))
        frames += 1
        if frames % 30 == 0:
            print("%4d frames  %.1f FPS  %d objects" % (frames, fps, len(scores)))
except KeyboardInterrupt:
    pass
finally:
    log.close()
    cap.release()
    for o in (out_net, out_file):
        if o is not None:
            o.release()
    runner.close()
    print("done: %d frames in %.0f s. saved out/live.avi and bench/logs/live_fps.csv"
          % (frames, time.time() - start))
