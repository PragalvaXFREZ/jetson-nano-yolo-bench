#!/bin/bash
# Send this laptop's webcam to the Nano.   bash laptop/stream_webcam.sh <nano-ip>
#
# v4l2src        read the webcam
# x264enc        compress to H.264. tune=zerolatency: do not buffer frames to
#                compress better, send each one immediately.
#                key-int-max=30: a full picture every second, so the Nano can
#                join (or recover from a lost packet) within a second.
# rtph264pay     cut the stream into network packets.   udpsink: send them.
NANO=${1:?usage: bash laptop/stream_webcam.sh <nano-ip>}
exec gst-launch-1.0 -q v4l2src device=/dev/video0 \
  ! image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert \
  ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 key-int-max=30 \
  ! rtph264pay config-interval=1 pt=96 ! udpsink host="$NANO" port=5000 sync=false
