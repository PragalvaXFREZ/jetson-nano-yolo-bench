#!/bin/bash
# Measure every engine in engines/ the same way, back to back.
#
#   bash nano/bench.sh            -> prints a table, saves raw logs in bench/
#
# Why not just read the number printed at the end of each build?
#   - each build ran at a different time, with the GPU clocks in a different state
#   - trtexec's default warm-up is 0.2 s, too short for the clocks to ramp up
# So: load each finished engine (seconds, no rebuild), warm up 1 s, measure 10 s.
#
# trtexec feeds the engine random numbers. That is fine: a conv layer costs the
# same number of multiplications whatever the pixel values are.
cd "$(dirname "$0")/.." || exit 1
TRTEXEC=/usr/src/tensorrt/bin/trtexec

printf "%-28s %12s %14s\n" "engine" "FPS" "GPU ms/frame"
for e in engines/*.engine; do
  log="bench/logs/run_$(basename "$e").log"
  $TRTEXEC --loadEngine="$e" --warmUp=1000 --duration=10 > "$log" 2>&1
  # "Throughput: 41.3 qps"            -> frames per second
  # "GPU Compute: min = .., mean = X" -> pure GPU time per frame, no copies
  fps=$(grep "Throughput:" "$log" | sed -E 's/.*Throughput: ([0-9.]+).*/\1/')
  gpu=$(grep "GPU Compute" "$log" | head -1 | sed -E 's/.*mean = ([0-9.]+) ms.*/\1/')
  printf "%-28s %12s %14s\n" "$(basename "$e")" "$fps" "$gpu"
done

# Where does the time go INSIDE the network? One run with per-layer timing.
# (Profiling adds overhead, so use this for proportions, not for the FPS.)
main=engines/yolov8n_416_fp16.engine
if [ -f "$main" ]; then
  $TRTEXEC --loadEngine="$main" --dumpProfile --separateProfileRun --duration=5 > bench/logs/profile_416_fp16.log 2>&1
  echo
  echo "per-layer profile saved to bench/logs/profile_416_fp16.log"
fi
