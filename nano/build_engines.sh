#!/bin/bash
# Build the extra engines for the benchmark, ONE AT A TIME.
#
# Never run two trtexec builds at once: a build works by timing GPU kernels
# against each other, and a second job on the GPU would corrupt those timings
# (it would pick slow kernels thinking they were fast).
#
# Start it so it survives a dropped SSH connection:
#   nohup bash nano/build_engines.sh > bench/logs/build.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
mkdir -p engines bench/logs
TRTEXEC=/usr/src/tensorrt/bin/trtexec

# If a build is already running (the first one, started by hand), wait for it.
while pgrep -x trtexec > /dev/null; do sleep 15; done

# The hand-built engine was saved in the home folder. Copy it in under a name
# that says what it is: model, input size, precision.
if [ -f "$HOME/yolov8n.engine" ] && [ ! -f engines/yolov8n_416_fp16.engine ]; then
  cp "$HOME/yolov8n.engine" engines/yolov8n_416_fp16.engine
fi

build () {   # build <onnx name> <engine name> [extra trtexec flags]
  local onnx="models/$1" engine="engines/$2"; shift 2
  if [ -f "$engine" ]; then echo "== $engine exists, skipping"; return; fi
  echo "== building $engine  ($(date +%T))"
  $TRTEXEC --onnx="$onnx" --saveEngine="$engine" --workspace=1024 "$@" > "bench/logs/build_$(basename "$engine").log" 2>&1
  echo "== done     $engine  ($(date +%T))  $(grep -c PASSED "bench/logs/build_$(basename "$engine").log") PASSED"
}

build yolov8n_416.onnx yolov8n_416_fp16.engine --fp16   # only if the hand build did not finish
build yolov8n_416.onnx yolov8n_416_fp32.engine          # no --fp16  => 32-bit everywhere
build yolov8n_320.onnx yolov8n_320_fp16.engine --fp16   # fewer pixels
echo "ALL BUILDS DONE"
