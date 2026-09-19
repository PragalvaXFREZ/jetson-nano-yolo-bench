#!/bin/bash
# Pretend to be the Nano: send colour bars, as JPEG over RTP, to the viewer on
# this same laptop (port 5001). Tests view_result.sh without the Nano involved.
# Stop with Ctrl+C.
exec gst-launch-1.0 -q videotestsrc is-live=true \
  ! video/x-raw,width=640,height=480,framerate=15/1 ! videoconvert ! video/x-raw,format=I420 \
  ! jpegenc ! rtpjpegpay ! udpsink host=127.0.0.1 port=5001
