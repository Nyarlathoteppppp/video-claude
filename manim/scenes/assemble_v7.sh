#!/bin/sh
# Assemble the v7 releases, every cut on the beat:
#   cold open = one bar of the song's climax over the platform cover (coldopen.py: first frame = cover, a punch-in per beat)
#   long  = cold open + the film from its first downbeat (frame 41 = 1.3667 s, on the 1.370 s downbeat: the neuron model lights up)
#   short = cold open + the film from the deep-learning downbeat (frame 3519 = 117.3 s, on the 117.306 s downbeat) to the end (Douyin, about 3:19)
# Loudness: -14 LUFS, true peak -1.5 dB.
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
A=../../ai_history
O=$A/film_out
$PY coldopen.py $A/cover/cover_9x16.jpg 1080 1920 $O/coldopen_9x16.mkv
$PY coldopen.py $A/cover/cover_16x9.jpg 1920 1080 $O/coldopen_16x9.mkv

join() {  # $1 cold open, $2 body video, $3 start in body (a downbeat), $4 output
  ffmpeg -loglevel error -y -i "$1" -ss "$3" -i "$2" \
    -filter_complex "[0:v]fps=30,format=yuv420p,setsar=1[v0];[1:v]fps=30,format=yuv420p,setsar=1[v1];\
[0:a]aresample=48000[a0];[1:a]aresample=48000,afade=t=in:st=0:d=0.012[a1];\
[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a];[a]loudnorm=I=-14:TP=-1.5:LRA=11[an]" \
    -map "[v]" -map "[an]" -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k -ar 48000 "$4"
  echo "$4 $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$4")"
}
join $O/coldopen_9x16.mkv  $O/ai_history_v7_body.mp4      1.366667 $O/ai_history_v7.mp4
join $O/coldopen_16x9.mkv  $O/ai_history_v7_bili_body.mp4 1.366667 $O/ai_history_v7_bili.mp4
join $O/coldopen_9x16.mkv  $O/ai_history_v7_body.mp4      117.3 $O/ai_history_v7_douyin_3min.mp4
