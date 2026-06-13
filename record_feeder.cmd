@echo off
REM Record RTSP to MP4 for later auto-labeling. Run AFTER connecting to Taipan hotspot.
REM Usage: record_feeder.cmd deer_whitehot 300
set NAME=%~1
if "%NAME%"=="" set NAME=feeder_clip
set SEC=%~2
if "%SEC%"=="" set SEC=300
set OUT=%~dp0feeder_clips\%NAME%.mp4
echo Recording %SEC%s to %OUT%
ffmpeg -y -rtsp_transport tcp -i "rtsp://admin:abcd1234@10.15.12.1:554/Streaming/Channels/101" -t %SEC% -c copy "%OUT%"
echo Done: %OUT%