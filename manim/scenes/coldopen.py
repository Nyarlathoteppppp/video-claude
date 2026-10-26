"""One bar of cold open on beat: the platform cover (first frame = the cover), a small punch-in on each of the four beats of
the song's climax bar, a quick white-out into the cut, which lands on a downbeat of the film.

    python coldopen.py cover.jpg 1080 1920 out.mp4        # writes video + the bar of music under it
"""
import csv
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))
BGM = os.path.normpath(os.path.join(HERE, "..", "..", "ai_history", "bgm"))
HIT = 283.445                      # the Navier-Stokes downbeat in bgm_v3
FPS = 30


def climax_bar():
    with open(os.path.join(BGM, "bgm_v3.beats.csv")) as fh:
        rows = [(float(r["t"]), int(r["beat_in_bar"])) for r in csv.DictReader(fh)]
    i = min(range(len(rows)), key=lambda k: abs(rows[k][0] - HIT))
    t0 = rows[i][0]
    beats = [t for t, _ in rows[i:i + 5]]
    return t0, [b - t0 for b in beats[:4]], beats[4] - t0          # start, beat offsets, bar length


def main(cover, w, h, out):
    t0, beats, bar = climax_bar()
    base = Image.open(cover).convert("RGB").resize((w, h), Image.LANCZOS)
    n = int(round(bar * FPS))
    tmp = out + ".video.mp4"
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", tmp],
                         stdin=subprocess.PIPE)
    for k in range(n):
        t = k / FPS
        tb = max([b for b in beats if b <= t + 1e-6], default=0.0)
        hit = np.exp(-(t - tb) / 0.16) if k > 0 else 0.0           # frame 0 is the cover, untouched
        sc = 1 + 0.035 * hit
        im = base.resize((int(w * sc), int(h * sc)), Image.LANCZOS)
        x0, y0 = (im.width - w) // 2, (im.height - h) // 2
        im = ImageEnhance.Brightness(im.crop((x0, y0, x0 + w, y0 + h))).enhance(1 + 0.18 * hit)
        a = np.asarray(im, float)
        white = max(0.0, (t - (bar - 0.16)) / 0.16) * 0.55           # white-out into the cut
        a = a * (1 - white) + 255 * white
        p.stdin.write(np.clip(a, 0, 255).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()
    bar = n / FPS                                                    # the sound exactly as long as the picture
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-ss", f"{t0:.3f}", "-t", f"{bar:.4f}",
                    "-i", os.path.join(BGM, "bgm_v3.wav"), "-map", "0:v", "-map", "1:a",
                    "-af", f"afade=t=out:st={bar - 0.012:.3f}:d=0.012", "-c:v", "copy", "-c:a", "pcm_s16le", out], check=True)
    os.remove(tmp)
    print(out, f"{bar:.3f}s, beats at", [round(b, 3) for b in beats])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
