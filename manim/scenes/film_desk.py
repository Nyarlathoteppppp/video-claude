"""二's desk, film 51.0 to 58.3, from the frames film_desk3d.py rendered: the room's own grade (film_room.py: exposure, its film curve, its tone and grain).

    python film_desk.py still T [T ...]        # ink/film_out/film/desk/keys/desk_T.png at 1920, the subtitle of T burned in, to look at
    python film_desk.py seq                    # ink/film_out/film/desk/seq/NNNN.png and desk.mp4, frame 0 at 51.0, no subtitles: film_rough burns them
    python film_desk.py seq hi                 # from plates_1080/ (film_desk3d.py --pct=100 --out=plates_1080): desk_1080/NNNN.png and desk_1080.mp4

A still needs the frame of its time rendered (film_desk3d.py --frames=...); the sequence takes whatever size the frames were rendered at.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

import film_room as room
import ink_subs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "desk")
PLATES = os.path.join(OUT, "plates")
FPS = 30
T0, T1 = ink_subs.at("依我之见"), ink_subs.at("鬼之所以哭")
N = int(round((T1 - T0) * FPS))
EXPOSURE = room.ROOM["exposure"]


def index(t):
    return min(max(int(round((t - T0) * FPS)), 0), N - 1)


def frame(i, w=None, plates=PLATES):
    a = np.load(os.path.join(plates, f"d{i:04d}.npy")).astype(np.float32)
    y = room.room_tone(room.film(a * EXPOSURE))
    f = room.finish(y, i + 51, .0065)
    if w and f.shape[1] != w:
        f = np.asarray(Image.fromarray(f).resize((w, w * 9 // 16), Image.LANCZOS))
    return f


def still(ts):
    os.makedirs(os.path.join(OUT, "keys"), exist_ok=True)
    for t in ts:
        f = frame(index(t), 1920)
        f = ink_subs.apply(f, t)
        path = os.path.join(OUT, "keys", f"desk_{t}.png")
        Image.fromarray(f).save(path)
        print(path, index(t), flush=True)


def seq(hi=False):
    folder = os.path.join(OUT, "desk_1080" if hi else "seq")
    plates = os.path.join(OUT, "plates_1080") if hi else PLATES
    os.makedirs(folder, exist_ok=True)
    for i in range(N):
        Image.fromarray(frame(i, plates=plates)).save(os.path.join(folder, f"{i:04d}.png"))
        if i % 30 == 0:
            print("desk", i, "/", N, flush=True)
    mp4 = os.path.join(OUT, "desk_1080.mp4" if hi else "desk.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(folder, "%04d.png"), "-c:v", "libx264", "-crf", "14",
                    "-preset", "slow", "-pix_fmt", "yuv420p", mp4], check=True)
    print(mp4, N, "frames", f"{N / FPS:.3f} s", flush=True)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "still":
        still([float(x) for x in sys.argv[2:]])
    elif what == "seq":
        seq(sys.argv[2:3] == ["hi"])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
