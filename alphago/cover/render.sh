#!/bin/sh
# Render the cover HTML files to exact-size images with headless Chrome (run from anywhere).
set -e
cd "$(dirname "$0")"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
shot() {  # html  width  height  out.png
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 \
    --allow-file-access-from-files --window-size="$2,$3" --virtual-time-budget=3000 \
    --screenshot="$PWD/$4" "file://$PWD/$1" >/dev/null 2>&1
}
shot cover_16x9.html 1920 1080 cover_16x9.png
shot cover_3x4.html 1080 1440 cover_3x4.png
shot cover_9x16.html 1080 1920 cover_9x16.png
# JPEG copies for upload (high quality, small)
for f in cover_16x9 cover_3x4 cover_9x16; do
  ffmpeg -loglevel error -y -i "$f.png" -q:v 2 "../$f.jpg"
done
ls -la cover_*.png ../cover_*.jpg
