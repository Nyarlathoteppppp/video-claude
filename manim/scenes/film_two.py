"""二's montage put together: the eight carriers a second each, cut hard, each thing whole with its middle at the same place and the same size (film_docs3d.TWO_MID, TWO_FILL; until
2026-11-08 each one's first sign was held at one place and the things jumped about around it), the camera coming in on it evenly.

    python film_two.py frames [--w 960]       # ink/film_out/film/two/frames/f0000.png ..., 30 to a carrier
    python film_two.py clip [--w 960]         # the frames, then ink/film_out/film/two/two.mp4 (8.0 s, 30 fps); --w 1920: frames_1920/ and two_1080.mp4
    python film_two.py sheet                  # the eight stills, 480 wide, the held point marked: two/sheet.png

The stills are 1920 by 1080: film_docs3d.py --shot two (imprint/mN_NAME.png) for the seven things on the table, film_screens.py still two_term (screens/keys/two_term.png) for the
Terminal. Coming in is done on the still, 4 per cent in a second about the held point: the parallax a real move that small would give is under a pixel. Each frame gets its own grain.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.join(HERE, "..", "..", "ink", "film_out", "film")
OUT = os.path.join(FILM, "two")
MID = (0.5, 0.44)           # film_docs3d.TWO_MID: where every thing's middle sits
ITEMS = (("tablet", "imprint/m1_tablet.png", MID), ("stone", "imprint/m2_stone.png", MID), ("greek", "imprint/m3_greek.png", MID), ("bone", "imprint/m4_bone.png", MID),
         ("slips", "imprint/m5_slips.png", MID), ("song", "imprint/m6_song.png", MID), ("quarto", "imprint/m7_quarto.png", MID),
         ("terminal", "screens/keys/two_term.png", MID))      # the Terminal with hello.c's middle there (film_screens3d --hold)
PUSH, FPS, SECS = .04, 30, 1.0
GRAIN = .010


def push_frame(im, i, n, w, at):
    """The still as the camera sees it i frames into its second: nearer by PUSH * (i + .5) / n of the way, about the point at. Drawn at the still's size, then brought down to w."""
    W, H = im.size
    s = 1 / (1 - PUSH * (i + .5) / n)
    ax, ay = at[0] * W, at[1] * H
    big = im.transform((W, H), Image.AFFINE, (1 / s, 0, ax - ax / s, 0, 1 / s, ay - ay / s), resample=Image.BICUBIC)
    return big.resize((w, int(round(w * H / W))), Image.LANCZOS)


def grain(a, seed):
    p = a.astype(np.float32) / 255
    lum = p.mean(-1, keepdims=True)
    n = gaussian_filter(np.random.default_rng(seed).normal(0, 1, p.shape[:2]).astype(np.float32), .5) * 1.5
    return np.uint8(np.clip(p + n[..., None] * (GRAIN * (1.2 - lum) + .0016), 0, 1) * 255 + .5)


def frames(w):
    folder = os.path.join(OUT, "frames" if w == 960 else f"frames_{w}")
    os.makedirs(folder, exist_ok=True)
    n = int(round(SECS * FPS))
    k = 0
    for name, path, at in ITEMS:
        im = Image.open(os.path.join(FILM, path)).convert("RGB")
        assert im.size == (1920, 1080), (path, im.size)
        for i in range(n):
            a = grain(np.asarray(push_frame(im, i, n, w, at)), 7000 + k)
            Image.fromarray(a).save(os.path.join(folder, f"f{k:04d}.png"), compress_level=1)
            k += 1
        print(name, flush=True)
    return folder, k


def clip(w):
    folder, k = frames(w)
    out = os.path.join(OUT, "two.mp4" if w == 960 else "two_1080.mp4" if w == 1920 else f"two_{w}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(folder, "f%04d.png"), "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p", out], check=True)
    print(out, "%.1f s" % (k / FPS), flush=True)


def sheet():
    tiles = []
    for name, path, at in ITEMS:
        im = Image.open(os.path.join(FILM, path)).convert("RGB").resize((480, 270), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        x, y = at[0] * 480, at[1] * 270
        d.line([(x - 7, y), (x + 7, y)], fill=(230, 40, 30))
        d.line([(x, y - 7), (x, y + 7)], fill=(230, 40, 30))
        tiles.append(im)
    s = Image.new("RGB", (960, 270 * 4))
    for i, t in enumerate(tiles):
        s.paste(t, ((i % 2) * 480, (i // 2) * 270))
    os.makedirs(OUT, exist_ok=True)
    s.save(os.path.join(OUT, "sheet.png"))
    print(os.path.join(OUT, "sheet.png"))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    w = int(sys.argv[sys.argv.index("--w") + 1]) if "--w" in sys.argv else 960
    if what == "frames":
        frames(w)
    elif what == "clip":
        clip(w)
    elif what == "sheet":
        sheet()
    else:
        raise SystemExit(__doc__)
