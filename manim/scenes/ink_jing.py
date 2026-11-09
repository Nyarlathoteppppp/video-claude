"""三's climax, one move under water. 字，成了精: the ink of the whole pool is only drifting, and then all of it turns at once, the way a shoal turns, as if something had woken. 那个精，便是我: it is drawn in
from everywhere until it stands as the character, and holds. At the end of the chapter the current takes it. The gathering is a simulation of the character coming apart, played backwards, so the ink
does the one thing ink never does.

    python ink_jing.py sim           # once: ink/film_out/move/jing_{gather,leave,far}.npy
    python ink_jing.py strip [A B]   # ink/film_out/move/jing_strip.png
    python ink_jing.py [A B]         # ink/film_out/move/jing.mp4
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

import ink_fluid
from ink_film import glyph_mask
from ink_grade import INK_K, fft_noise, sm
from ink_move import _X, _Y, FPS, H, OUT, W, sample
from ink_subs import at, chapters

START = {name: a for name, _, a, _ in chapters()}
T0 = at("字，成了精") - 0.8       # going under
TURN = at("字，成了精") + 2.0     # all of it turns
LAND = at("那个精") + 2.8         # it stands
GO = START["四"] - 1.5            # the current takes it
T1 = START["四"] + 0.85           # the join to the surface is over

ND = int(os.environ.get("JING_ND", 2304))      # the sheet, and how many frame pixels to one of its pixels
K = 2304 / ND
ND_FAR = ND // 2
K_FAR = 2.2 * 2304 / ND
CX, CY = 860.0, 525.0                          # where the character stands in the frame
GLYPH = 930                                    # and how tall it is drawn, in frame pixels
N_GATHER = int((LAND - T0) * FPS) + 1
N_LEAVE = int((T1 - LAND) * FPS) + 2
N_FAR = int((T1 - T0) * FPS) + 2
GATHER, LEAVE, FAR = (os.path.join(OUT, f"jing_{x}.npy") for x in ("gather", "leave", "far"))

TOP = np.array([0.93, 0.93, 0.885], np.float32)
LOW = np.array([0.16, 0.26, 0.36], np.float32)


def glyph():
    """我, in density, on a sheet."""
    size = int(GLYPH / K)
    g = glyph_mask("我", "kaib", size, 40)
    g = gaussian_filter(g, 0.8)
    out = np.zeros((ND, ND), np.float32)
    h = g.shape[0]
    y, x = (ND - h) // 2, (ND - h) // 2
    out[y:y + h, x:x + h] = g
    return 7.0 * out * np.clip(1 + 0.12 * fft_noise(ND, ND, 2.2, 11), 0.6, 1.2)


def gather_rate(t):
    """Water seconds a film second, for the part played backwards: drifting, then drawn in, and slowing as it lands."""
    x = sm((t - TURN - 0.2) / 1.6)
    land = sm((LAND - t) / 0.9)
    return (0.45 + 0.95 * x) * (0.12 + 0.88 * land)


def leave_rate(t):
    """Holding, its edges all but still; then running."""
    x = sm((t - GO) / 1.3)
    return 0.05 + 1.0 * x


def simulate():
    # the character coming apart: its weight takes it down in plumes, and a slow current opens the whole thing out; played backwards this is everything coming in
    steps = [gather_rate(T0 + (N_GATHER - 1 - n) / FPS) / FPS for n in range(N_GATHER)]
    print(f"gather: {N_GATHER} sheets, {sum(steps):.2f} s of water", flush=True)

    def opens(T, X, Y):
        k = 0.34 * math.exp(-T / 3.0) + 0.05
        return (X - 0.5) * k, (Y - 0.5) * k

    ink_fluid.simulate(GATHER, steps, [], nv=ND // 4, nd=ND, seed=5, weight=0.45, curl=1.5, spread=1.6, pull=opens, start=glyph(), skin=False, stir=2.2)
    water = np.cumsum([leave_rate(LAND + n / FPS) / FPS for n in range(N_LEAVE)])
    runs = water[int((GO - LAND) * FPS)]
    steps = [leave_rate(LAND + n / FPS) / FPS for n in range(N_LEAVE)]

    def away(T, X, Y):
        s = sm((T - runs) / 0.8)
        return -0.05 * s + 0.0 * X, -0.34 * s + 0.0 * Y

    ink_fluid.simulate(LEAVE, steps, [], nv=ND // 4, nd=ND, seed=9, weight=0.25, curl=1.5, spread=1.6, pull=away, start=glyph(), skin=False, stir=2.2)
    # the rest of the pool's ink, far off, big and slow
    n = fft_noise(ND_FAR, ND_FAR, 3.1, 31)
    start = 2.4 * sm((n - 0.25) / 0.9) * np.clip(1 + 0.5 * fft_noise(ND_FAR, ND_FAR, 2.2, 32), 0.2, 2)
    ink_fluid.simulate(FAR, [0.8 / FPS] * N_FAR, [], nv=ND_FAR // 4, nd=ND_FAR, seed=6, weight=0.3, curl=1.5, spread=1.2, start=start, skin=False, stir=2.0, warm=2.0)


class Deep:
    """The water, looked up into from far down: the surface a shaking ceiling of light, rays fanning from it."""

    def __init__(self, cx=0.45):
        self.cx = cx
        u, v = _X / W, _Y / H
        self.g = np.exp(-(((u - cx) / 0.42) ** 2 + ((v - 0.2) / 0.78) ** 2)) ** 0.8
        self.angle = np.arctan2(_X - cx * W, _Y + 0.55 * H)
        r = np.random.default_rng(83)
        rays = np.cumsum(r.normal(0, 1, 4096))
        rays = gaussian_filter(rays - np.linspace(rays[0], rays[-1], 4096), 6)
        self.rays = rays / np.abs(rays).max()
        self.band = 360
        ub, vb = u[: self.band], v[: self.band]
        t = 1 / (vb + 0.16)
        self.bx, self.by = (ub - cx) * t * 48, t * 30
        self.n1, self.n2 = fft_noise(1024, 1024, 3.0, 87), fft_noise(1024, 1024, 2.6, 89)
        self.ceil = np.exp(-(vb / 0.11) ** 2) * np.exp(-((ub - cx) / 0.6) ** 2)
        self.fade = np.exp(-v / 0.8)
        self.wash = (1 + 0.035 * np.clip(fft_noise(H, W, 3.0, 91), -2, 2)) * (1 + 0.014 * fft_noise(H, W, 0.7, 92))
        self.r2 = ((_X - CX) / 620.0) ** 2 + ((_Y - CY) / 560.0) ** 2

    def frame(self, t):
        ray = np.interp(self.angle * 2600 + 2048 + 9 * t, np.arange(4096), self.rays).astype(np.float32)
        L = 0.40 + self.g * (0.60 + 0.34 * ray * self.fade)
        net = map_coordinates(self.n1, [(self.by + 3 * t) % 1024, (self.bx + 5 * t) % 1024], order=1, mode="wrap") + 0.45 * map_coordinates(self.n2, [(self.by * 2.1 - 4 * t) % 1024, (self.bx * 2.3 + 3 * t) % 1024], order=1, mode="wrap")
        net = np.clip(1 - np.abs(net) * 1.25, 0, 1) ** 2.2
        L[: self.band] += self.ceil * (0.08 + 0.5 * net)
        # when it stands, the water behind it clears and the light comes up
        L = L + np.exp(-self.r2) * 0.3 * sm((t - LAND + 1.4) / 1.8) * (1 - 0.8 * sm((t - GO - 0.4) / 1.2))
        L = np.clip(L, 0, 1.05)[..., None] * self.wash[..., None]
        return LOW * (1 - L) + TOP * L


def density(sheet):
    return ink_fluid.DENSITY * (np.asarray(sheet, np.float32) / 255) ** 2


def turned(t, r):
    """How far round the ink still has to come, in radians: the middle goes first and the edge follows, in about the time a shoal takes."""
    return -0.62 * (1 - sm((t - TURN - r * 0.32) / 0.75))


_deep = _gather = _leave = _far = _r = None


def _load():
    global _deep, _gather, _leave, _far, _r
    if _deep is None:
        _deep = Deep()
        _gather, _leave, _far = np.load(GATHER, mmap_mode="r"), np.load(LEAVE, mmap_mode="r"), np.load(FAR, mmap_mode="r")
        _r = np.hypot(_X - CX, _Y - CY) / 1100.0


def near_at(t):
    if t < LAND:
        d = density(_gather[int(np.clip(N_GATHER - 1 - round((t - T0) * FPS), 0, N_GATHER - 1))])
    else:
        d = density(_leave[int(np.clip(round((t - LAND) * FPS), 0, N_LEAVE - 1))])
    return d


def laid(d, t, k, size, dt=0.0):
    """A sheet in the frame, with the turn in it. The whole of it also comes a little nearer as it turns."""
    a = turned(t + dt, _r)
    c, s = np.cos(a), np.sin(a)
    grow = 1 + 0.1 * (1 - sm((t + dt - TURN) / 1.1))
    x, y = (_X - CX) * grow, (_Y - CY) * grow
    return sample(d, (s * x + c * y) / k + size / 2, (c * x - s * y) / k + size / 2)


def frame(t):
    _load()
    f = _deep.frame(t)
    # the far ink goes where everything goes: in, and it thins as it goes, so the water clears
    fn = int(np.clip(round((t - T0) * FPS), 0, N_FAR - 1))
    gone = sm((t - TURN - 0.4) / 3.6)
    d = laid(density(_far[fn]), t, K_FAR * (1 - 0.45 * gone), ND_FAR)
    d = gaussian_filter(d, 2.5) * (1 - 0.94 * gone)
    f = f * np.exp(-(0.42 * d)[..., None] * INK_K)
    near = near_at(t)
    fast = abs(t - TURN - 0.45) < 0.65
    if fast:           # the turn is quick: it has to smear a little, or it looks like a picture being rotated
        d = sum(laid(near, t, K, ND, x / FPS) for x in (-0.33, 0.0, 0.33)) / 3
    else:
        d = laid(near, t, K, ND)
    d = 0.86 * d + 0.28 * gaussian_filter(d, 4) + 0.16 * gaussian_filter(d, 18)
    return np.clip(f * np.exp(-d[..., None] * INK_K) * 255 + 0.5, 0, 255).astype(np.uint8)


def strip(a=T0, b=T1, n=18):
    rows = []
    ts = np.linspace(a, b, n)
    for i in range(0, n, 6):
        rows.append(np.concatenate([np.asarray(Image.fromarray(frame(float(t))).resize((640, 360), Image.LANCZOS)) for t in ts[i:i + 6]], 1))
    path = os.path.join(OUT, "jing_strip.png")
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    print(path)


def film(a=T0, b=T1):
    path = os.path.join(OUT, "jing.mp4")
    pipe = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for n in range(int(a * FPS), int(b * FPS)):
        pipe.stdin.write(frame(n / FPS).tobytes())
    pipe.stdin.close()
    pipe.wait()
    print(path)


if __name__ == "__main__":
    if sys.argv[1:] == ["sim"]:
        simulate()
    elif sys.argv[1:2] == ["strip"]:
        strip(*(float(x) for x in sys.argv[2:4]))
    else:
        film(*(float(x) for x in sys.argv[1:3]))
