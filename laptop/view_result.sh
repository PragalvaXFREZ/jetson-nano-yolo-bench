#!/bin/bash
# Show the annotated video that the Nano sends back (JPEG over RTP, UDP 5001).
exec gst-launch-1.0 -q udpsrc port=5001 \
  caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=JPEG,payload=26" \
  ! rtpjpegdepay ! jpegdec ! videoconvert ! autovideosink sync=false
