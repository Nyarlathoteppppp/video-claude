"""一 · the opening's events: what happens inside the shots from the tracks to the rain of grain, so that for the first half minute something new is always happening. Each is the shot's still taken
apart and put back over time, through its own push. In ink_cut.py as move="ink_open:tracks" and so on.

    python ink_open.py strip NAME       # ink/film_out/move/open_NAME.png
    python ink_open.py grade A B        # the grain frames, from their render to their colour
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, label

from ink_color import glaze
from ink_grade import load, sm
from ink_move import _X, _Y, FPS, H, HERE, OUT, STILLS, W, Move, Plate, noise, lookup
from ink_subs import at, chapters

START = {name: a for name, _, a, _ in chapters()}
T_TRACKS, T_GROW = START["一"] + 3.6, START["一"] + 5.8
T_RIFT, T_GRAIN, T_MARSH = at("而天雨粟", -0.6), at("而天雨粟", 1.6), at("鬼夜哭")


def fill(f, m, r=12):
    """The picture with a mask's part taken out, spread in from around it."""
    keep = (1 - m)[..., None]
    wide = gaussian_filter(f * keep, (r, r, 0)) / (gaussian_filter(keep, (r, r, 0)) + 1e-3)
    return f * keep + wide * (1 - keep)


class Tracks:
    """观鸟兽足迹: a bird has walked up this ridge. Its prints come one at a time, from the far end down to the lens."""

    def __init__(self):
        self.p = Plate("myth/tracks", "moon")
        self.move = Move(self.p, mist=0.1)
        ids, n = label((self.p.L > 0.7) & (_Y > 0.34 * H))
        area = np.bincount(ids.ravel())
        keep_mask = np.isin(ids, np.nonzero((area[1:] > 30) & (area[1:] < 9000))[0] + 1)
        ids, n = label(gaussian_filter(keep_mask.astype(np.float32), 2.5) > 0.12)
        self.soft = gaussian_filter((gaussian_filter(keep_mask.astype(np.float32), 7.0) > 0.03).astype(np.float32), 2.0)
        ys = np.array([(_Y[ids == i]).mean() for i in range(1, n + 1)]) if n else np.zeros(0)
        order = np.argsort(np.argsort(ys))
        on = T_TRACKS + 0.1 + 1.9 * (order + 1) / max(n, 1)
        idx = distance_transform_edt(ids == 0, return_distances=False, return_indices=True)
        self.on = np.r_[0.0, on].astype(np.float32)[ids[idx[0], idx[1]]]
        self.ground = fill(self.p.f, self.soft)

    def frame(self, t):
        a = (self.soft * sm((t - self.on) / 0.1))[..., None]
        f = self.ground * (1 - a) + self.p.f * a
        s = 0.08 * (t - T_TRACKS) / (T_GROW - T_TRACKS)
        return self.move.frame(t - T_TRACKS, f, s=max(s, 0.0))


class Grow:
    """作书: the bird's own lines run up out of its foot and are a sign."""

    def __init__(self):
        self.p = Plate("myth/grow", "moon")
        self.move = Move(self.p)
        L = self.p.L
        ids, n = label(L - gaussian_filter(L, 30) > 0.1)
        area = np.bincount(ids.ravel())
        strokes = np.isin(ids, np.nonzero(area[1:] > 60)[0] + 1).astype(np.float32)
        self.m = gaussian_filter((gaussian_filter(strokes, 6.0) > 0.04).astype(np.float32), 2.0)
        self.lines = gaussian_filter(strokes, 1.2)
        self.d = np.hypot(_X - 810, _Y - 960) + 40 * gaussian_filter(lookup(noise(77), _X, _Y), 3)
        self.far = np.percentile(self.d, 99.9)
        self.ground = fill(self.p.f, self.m, 16)

    def frame(self, t):
        x = (t - T_GROW + 0.1) / 1.25
        front = self.far * x ** 0.85 if x > 0 else -1.0
        a = (self.m * sm((front - self.d) / 40))[..., None]
        f = self.ground * (1 - a) + self.p.f * a
        hot = (self.lines * np.exp(-(((front - self.d) / 50) ** 2)))[..., None]
        f = f * (1 - hot) + np.array([1.0, 0.98, 0.92], np.float32) * hot
        return self.move.frame(t - T_GROW, f, s=max(0.04 * (t - T_GROW) / (T_RIFT - T_GROW), 0.0))


SIGNS = [(1050, 105, 150), (195, 285, 280), (1455, 465, 240), (1125, 636, 170), (435, 615, 170)]      # the tear's bird, then sun, tree, man and mountain


class Rift:
    """天: the signs go up into the cloud one after another, each lighting what is round it, and the sky gives way."""

    def __init__(self):
        self.p = Plate("myth/rift", "night")
        self.move = Move(self.p, mist=0.08)
        self.parts = []
        for i, (x, y, r) in enumerate(SIGNS):
            m = sm((r - np.hypot(_X - x, _Y - y)) / (0.5 * r))
            # what is there without it: the bird is dark on the light, so the light without the bird; the others light on the dark, so the picture with that light taken off, and nothing else
            f = (np.maximum if i == 0 else np.minimum)(self.p.f, fill(self.p.f, sm((r * 0.92 - np.hypot(_X - x, _Y - y)) / (0.12 * r)), int(r * 0.45)))
            self.parts.append((m[..., None], f, T_RIFT - 0.1 + 0.3 * i))
        L = self.p.L
        self.tear = gaussian_filter(sm((L - 0.8) / 0.12), 6)[..., None] * (_Y < 0.75 * H)[..., None]

    def frame(self, t):
        f = self.p.f
        for m, ground, on in self.parts:
            a = (m * (1 - sm((t - on) / 0.3)))
            f = f * (1 - a) + ground * a
        burn = 0.18 * sm((t - T_RIFT - 1.5) / 0.7)
        f = f * (1 + burn * self.tear)
        return self.move.frame(t - T_RIFT, np.clip(f, 0, 1), s=max(0.05 * (t - T_RIFT) / (T_GRAIN - T_RIFT), 0.0))


class Grain:
    """天雨粟: millet coming down, in true 3D: ink3d_myth.py --shot=grain --frames=A-B, graded with `python ink_open.py grade A B`. 195 frames from a little before the shot to the end of its join."""

    FRAME0 = T_GRAIN - 0.25

    def frame(self, t):
        n = min(max(int(round((t - self.FRAME0) * FPS)), 0), 194)
        return np.asarray(Image.open(os.path.join(STILLS, "myth", "grain", f"grain_{n:03d}_col.png")).convert("RGB"))


def grade_grain(a, b):
    """The frames in ink and then in colour; what ink_grade.py's command line does, and the glaze over it."""
    work = os.path.join(STILLS, "myth", "grain")
    for n in range(a, b + 1):
        name = f"grain_{n:03d}"
        subprocess.run([sys.executable, os.path.join(HERE, "ink_grade.py"), name, "--night", "--lift=0", "--sky=.2", "--gamma=1.1", "--mist=.04", "--mist-h=8", "--mist-d=.08", f"--out={name}_ink.png"], cwd=work, check=True, stdout=subprocess.DEVNULL)
        ink = np.asarray(Image.open(os.path.join(work, name + "_ink.png")).convert("RGB"))
        Image.fromarray(glaze(ink, load(os.path.join(work, name + "_z.png")), "grain")).save(os.path.join(work, name + "_col.png"))
        print(name, flush=True)


_made = {}


def _make(kind):
    if kind not in _made:
        _made[kind] = kind()
    return _made[kind]


def tracks(t):
    return _make(Tracks).frame(t)


def grow(t):
    return _make(Grow).frame(t)


def rift(t):
    return _make(Rift).frame(t)


def grain(t):
    return _make(Grain).frame(t)


SPANS = dict(tracks=(T_TRACKS, T_GROW), grow=(T_GROW, T_RIFT), rift=(T_RIFT, T_GRAIN), grain=(T_GRAIN, T_MARSH))

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "grade":
        grade_grain(int(sys.argv[2]), int(sys.argv[3]))
        sys.exit()
    names = sys.argv[2:] or list(SPANS)
    for name in names:
        a, b = SPANS[name]
        out = Image.new("RGB", (4 * 640, 2 * 360))
        for i in range(8):
            t = a + (b - a) * i / 7
            out.paste(Image.fromarray(globals()[name](t)).resize((640, 360), Image.LANCZOS), ((i % 4) * 640, (i // 4) * 360))
        out.save(os.path.join(OUT, f"open_{name}.png"))
