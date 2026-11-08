"""二 · 依我之见 on the film's clock: the flight rendered by ink3d_ages.py, each frame in ink and colour by its shot's recipe, with the water and mist moving over it. In ink_cut.py as
move="ink_ages:bone" and so on: each takes film time and gives a 1920x1080 uint8 RGB frame.

    python ink_ages.py grade NAME [A B | A,B,C]     # ages/seq/NAME_NNN.png -> _ink.png -> _col.png
    python ink_ages.py strip NAME [A,B,C]           # ink/film_out/move/ages_NAME.png, by default eight frames across the shot
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from ink_ages_clock import FPS, SPAN, count
from ink_color import fit, glaze
from ink_grade import grade, load
from ink_move import FAR, H, LUM, OUT, STILLS, W, ZMAX, Move, Plate

SEQ = os.path.join(STILLS, "ages", "seq")
RIVER = dict(water=1.0, level=0.22, mist=0.1)
RECIPE = dict(
    bone=dict(palette="day", ink=dict(rho=0.008, mist_h=40.0, gamma=1.3, sky_d=0.5, mist_d=0.4), move=RIVER, far=95.0, wall=(130.0, 20.0)),     # grey dawn: its glaze goes from warm to none and stops, or the sky's edge would show through the mist in blue
    stele=dict(palette="day", ink=dict(rho=0.012, mist_h=40.0, gamma=1.3, sky_d=0.35, mist_d=0.35), move=RIVER),
    screen=dict(palette="night", ink=dict(rho=0.012, mist_h=40.0, gamma=1.45, sky_d=0.6, mist_d=0.62), move=RIVER),
    falls=dict(palette="deep", ink=dict(rho=0.001, mist_h=200.0, gamma=1.15, sky_d=1.1, mist_d=0.6), move=dict(water=1.0, mist=0.15)),
)


def stem(shot, n):
    return os.path.join(SEQ, f"{shot}_{n:03d}")


def paper_of(shot):
    """How bright bare paper is, taken once from the shot's first frame, so that the glaze does not breathe from frame to frame."""
    ink = np.asarray(Image.open(stem(shot, 0) + "_ink.png").convert("RGB"), np.float32) / 255
    return float(np.percentile((ink * LUM).sum(-1), 99.5))


def capped(z, shot):
    """The depth pass as the glaze and the moving water take it."""
    far = RECIPE[shot].get("far")
    if far:
        z = z.copy()
        z[..., 0] = np.minimum(z[..., 0], far / FAR)
    return z


def behind_mist(z, shot, far, zmax):
    """The mist cards are not in the depth pass, so it sees through them to the hills and the gap between the hills. The fog is worked out from what the lens really meets: past this distance, the card."""
    wall = RECIPE[shot].get("wall")
    if not wall:
        return z
    z = z.copy()
    past = (z[..., 0] <= 0) | (z[..., 0] * far > wall[0])
    z[..., 0][past], z[..., 1][past] = wall[0] / far, wall[1] / zmax
    return z


def grade_frames(shot, frames):
    r = RECIPE[shot]
    white = None
    for n in frames:
        name = stem(shot, n)
        if not os.path.exists(name + ".png"):
            continue
        meta = json.load(open(name + ".json"))
        z = load(name + "_z.png")
        i = r["ink"]
        ink = grade(load(name + ".png"), behind_mist(z, shot, meta["far"], meta["zmax"]), True, i["rho"], i["mist_h"], meta["cam_z"], i["gamma"], 1.0, i["sky_d"], 0, meta["far"], meta["zmax"], 0.0, i["mist_d"])
        Image.fromarray(ink).save(name + "_ink.png")
        white = white or paper_of(shot)
        Image.fromarray(glaze(ink, capped(z, shot), r["palette"], far=meta["far"], white=white)).save(name + "_col.png")
        print(os.path.basename(name), flush=True)


class Flown(Plate):
    """One finished frame of a shot, and what its depth pass says about it. The glaze is already on it."""

    def __init__(self, shot, n):
        im = Image.open(stem(shot, n) + "_col.png").convert("RGB")
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)
        self.f = np.asarray(im, np.float32) / 255
        self.L = (self.f * LUM).sum(-1)
        z = capped(fit(load(stem(shot, n) + "_z.png"), H, W), shot)
        self.dist = np.maximum(gaussian_filter(z[..., 0], 1.2) * FAR, 0.5)
        self.sky = z[..., 0] <= 0
        self.dist[self.sky] = FAR
        self.height = z[..., 1] * ZMAX
        self.pool = shot == "falls"

    def water(self, tol=0.05, y0=0.3, level=None):
        if self.pool:      # its surface is dished; all of it low down in the pit is water
            return gaussian_filter(((self.height < 1.5) & ~self.sky).astype(np.float32), 1.5)
        return Plate.water(self, tol, y0, level)


class Shot:
    """The shot at any film time. The water's pattern is kept on the world, not on the lens going over it, by how far the lens has come."""

    def __init__(self, name):
        self.name, self.t0, self.count = name, SPAN[name][0], count(name)
        self.n, self.move = None, None
        with open(os.path.join(SEQ, name + "_path.json")) as f:
            path = json.load(f)
        self.fwd, self.across = [0.0], [0.0]
        for a, b in zip(path, path[1:]):
            step = np.array(b["eye"]) - np.array(a["eye"])
            view = np.array(b["look"]) - np.array(b["eye"])
            view /= np.linalg.norm(view)
            right = np.array((view[1], -view[0], 0.0))
            right /= np.linalg.norm(right) + 1e-6
            self.fwd.append(self.fwd[-1] + float(step @ view))
            self.across.append(self.across[-1] + float(step @ right))

    def frame(self, t):
        n = min(max(int(round((t - self.t0) * FPS)), 0), self.count - 1)
        if n != self.n:
            self.n = n
            self.move = Move(Flown(self.name, n), centre=(0.5, 0.5), **RECIPE[self.name]["move"])
            self.move.v = self.move.v + 22.0 * self.fwd[n]
            self.move.u = self.move.u + 22.0 * self.across[n]
        return self.move.frame(t - self.t0, s=0.0)


_shots = {}


def _shot(name):
    if name not in _shots:
        _shots[name] = Shot(name)
    return _shots[name]


def bone(t):
    return _shot("bone").frame(t)


def stele(t):
    return _shot("stele").frame(t)


def screen(t):
    return _shot("screen").frame(t)


def falls(t):
    return _shot("falls").frame(t)


def strip(name, frames=None):
    os.makedirs(OUT, exist_ok=True)
    frames = frames or [int(round((count(name) - 1) * i / 7)) for i in range(8)]
    rows = (len(frames) + 3) // 4
    out = Image.new("RGB", (4 * 640, rows * 360))
    s = _shot(name)
    for i, n in enumerate(frames):
        f = s.frame(s.t0 + n / FPS)
        a = f.astype(np.float32) / 255
        print(f"{name} {n}: mean {(a * LUM).sum(-1).mean():.2f} dead {((a * LUM).sum(-1) < .08).mean() * 100:.0f}% bright {((a * LUM).sum(-1) >= .5).mean() * 100:.0f}%", flush=True)
        out.paste(Image.fromarray(f).resize((640, 360), Image.LANCZOS), ((i % 4) * 640, (i // 4) * 360))
    path = os.path.join(OUT, f"ages_{name}.png")
    out.save(path)
    print(path)


def parse(args, shot):
    if not args:
        return list(range(count(shot)))
    if "," in args[0] or len(args) == 1:
        return [int(x) for x in args[0].split(",")]
    return list(range(int(args[0]), int(args[1]) + 1))


if __name__ == "__main__":
    what, shot = sys.argv[1], sys.argv[2]
    if what == "grade":
        grade_frames(shot, parse(sys.argv[3:], shot))
    elif what == "strip":
        strip(shot, parse(sys.argv[3:], shot) if sys.argv[3:] else None)
    else:
        raise SystemExit(__doc__)
