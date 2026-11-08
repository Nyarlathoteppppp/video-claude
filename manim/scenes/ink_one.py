"""一 · the end of the chapter and the start of the next, one continuous move. 或曰：鬼神之秘，自此藏不住了: a sweep of light finds the empty valley crowded with lights, and they begin to fall. Down at the
water's skin they come in one at a time, each with its ring, and the last to fall is a drop of the blackest ink. We go under with it, and that is why the crying stops: the hard silence on the
sound track is the silence under water. 依我之见 is set over the ink hanging there, all but still. With the first sound of water it runs again, and a current draws every thread of it out into one
thin stream. We come up where the stream is lying on a bank, beside the first thing anyone ever wrote on.

    python ink_one.py sim        # the ink, once: ink/film_out/move/fluid_{near,far}.npy
    python ink_one.py strip [A B]    # ink/film_out/move/one_strip.png
    python ink_one.py [A B]      # ink/film_out/move/one.mp4
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter

import ink_fluid
from ink_fluid import SKIN
from ink_grade import INK_K, fft_noise, sm
from ink_mix import PLIPS
from ink_move import _X, _Y, FPS, H, OUT, PAPER, W, Move, Plate, sample
from ink_subs import at, chapters

START = {name: a for name, _, a, _ in chapters()}
LINE = at("或曰：/鬼神")
IGNITE0, IGNITE1 = LINE + 0.2, LINE + 1.8    # the sweep of light
OUT0, OUT1 = LINE + 1.5, LINE + 3.6          # the valley's lights going out, as they fall
TO_SKIN = LINE + 2.75                        # to the water's skin
HIT = PLIPS[-1]                              # the last to fall is the ink
DIVE0, DIVE1 = HIT + 0.03, HIT + 1.1
STOP1 = at("依我之见", 3.2)                  # the first sound of water
RISE0, RISE1 = at("鬼之所以哭") - 0.85, at("鬼之所以哭")
RUNS = 0.7                                   # how fast the water goes once it runs again, against the film
FULL = int((RISE1 - HIT) * FPS) + 2

# the fall_plate camera, for where things on the water land
FOCAL, TILT, EYE = 2667.0, math.atan(0.0788), 0.20
LAND = (828.0, 771.0)
FLAT_RINGS = 0.17

# the stream on the bank in ages/bone, in the frame; the ink is drawn out along this
BANK = np.array([(0, 760), (350, 808), (480, 835), (600, 865), (720, 850), (840, 880), (960, 915), (1100, 905), (1280, 945), (1920, 1040)], np.float32)
NEAR, FAR = os.path.join(OUT, "fluid_near.npy"), os.path.join(OUT, "fluid_far.npy")
NEAR_SIZE, FAR_SIZE = 1536, 1024
NEAR_K, FAR_K = 1.0, 2.05     # frame pixels to a sheet's pixel
NEAR_X, FAR_X = 60.0, -90.0   # where a sheet's left edge is in the frame
UNDER = -150.0                # where the skin ends up, above the frame


def rate(t):
    """How fast the water goes against the film: a rush as the drop goes in, dying away to all but nothing through the silence, then running again."""
    if t < STOP1:
        return 2.1 * math.exp(-(t - HIT) / 0.4) + 0.015
    x = min((t - STOP1) / 0.6, 1.0)
    return 0.015 + (RUNS - 0.015) * x * x * (3 - 2 * x)


def steps():
    return [rate(HIT + n / FPS) / FPS for n in range(FULL)]


def simulate():
    water = np.cumsum(steps())
    runs = water[int((STOP1 - HIT) * FPS)] + 0.08     # the water time at which the current takes over

    def current(size, k, x0):
        X_ = (BANK[:, 0] - x0) / k / size
        Y_ = (BANK[:, 1] - UNDER + SKIN * size * k) / k / size

        def pull(T, X, Y):
            if T < runs:
                return 0.0, 0.0
            s = sm((T - runs) / 0.5)
            if not hasattr(pull, "yc"):
                pull.yc = np.interp(X, X_, Y_).astype(np.float32)
            xc = X_[5]
            return s * 0.4 * (X - xc) * sm((0.5 - np.abs(X - 0.5)) / 0.1), s * 2.8 * (pull.yc - Y)
        return pull

    slack = lambda T: sm((T - runs) / 0.4)
    ink_fluid.simulate(NEAR, steps(), [(0.0, (LAND[0] - NEAR_X) / NEAR_K / NEAR_SIZE, 0.022)], nv=384, nd=NEAR_SIZE, seed=3, pull=current(NEAR_SIZE, NEAR_K, NEAR_X), slack=slack)
    r = np.random.default_rng(12)
    far = [(-r.uniform(0.0, 1.0), x, r.uniform(0.012, 0.02)) for x in (0.12, 0.24, 0.33, 0.45, 0.62, 0.71, 0.82, 0.91)]
    ink_fluid.simulate(FAR, steps(), far, nv=256, nd=FAR_SIZE, seed=23, weight=3.0, pull=current(FAR_SIZE, FAR_K, FAR_X), slack=slack, warm=1.1)


def land_at(y_water):
    """Where on the frame a point on the water y_water away lands, and how far from the lens that is."""
    d = y_water * math.cos(TILT) + EYE * math.sin(TILT)
    v = y_water * math.sin(TILT) - EYE * math.cos(TILT)
    return 540 - FOCAL * v / d, d


def tear(f, x, y, sx, sy, colour, a=1.0, tail=2.0):
    """A soft drop on picture f, in place: round below and drawn up into a point."""
    x0, x1, y0, y1 = int(x - 4 * sx - 1), int(x + 4 * sx + 2), int(y - (2.5 + 2.5 * tail) * sy - 1), int(y + 3 * sy + 2)
    x0, x1, y0, y1 = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
    if x0 >= x1 or y0 >= y1:
        return
    dy = (_Y[y0:y1, x0:x1] - y) / sy
    width = sx * np.where(dy > 0, 1.0, np.clip(1 + dy / (2.5 * tail), 0.04, 1))
    long_ = np.where(dy > 0, dy, dy / tail)
    g = a * np.exp(-(long_ ** 2 + ((_X[y0:y1, x0:x1] - x) / width) ** 2))
    f[y0:y1, x0:x1] = f[y0:y1, x0:x1] * (1 - g[..., None]) + np.asarray(colour, np.float32) * g[..., None]


LIGHT = np.array([0.95, 0.97, 0.90], np.float32)
HALO = np.array([0.66, 0.84, 0.76], np.float32)
BLACK = np.array([0.035, 0.035, 0.04], np.float32)


class Valley:
    def __init__(self):
        self.dark = Plate("myth/bare_dark", "ghost", white=0.9)
        lit = Plate("myth/bare_lit", "ghost")
        self.move = Move(self.dark, water=1.0, level=0.0, mist=0.12)
        spots = gaussian_filter(np.maximum(lit.L - gaussian_filter(lit.L, 14) - 0.08, 0), 1.2)      # where the render's lights are; what is drawn there is drawn here
        peaks = (spots == maximum_filter(spots, 7)) & (spots > 0.05)
        self.y, self.x = np.nonzero(peaks)
        n = len(self.x)
        r = np.random.default_rng(9)
        self.b = np.clip(spots[peaks] / np.percentile(spots[peaks], 90), 0.35, 1) * r.uniform(0.7, 1, n)
        self.on = IGNITE0 + (IGNITE1 - IGNITE0) * np.clip(self.x / W * 0.82 + self.y / H * 0.18 + r.normal(0, 0.035, n), 0, 1)
        self.off = OUT0 + (OUT1 - OUT0) * r.random(n) ** 0.8
        self.fall = r.uniform(120, 260, n)
        self.phase, self.speed = r.uniform(0, 6.3, n), r.uniform(5, 12, n)
        print(n, "lights")

    def frame(self, t):
        age = t - self.off
        b = self.b * sm((t - self.on) / 0.08) * (1 + 1.3 * np.exp(-np.maximum(t - self.on, 0) / 0.22)) * (0.82 + 0.18 * np.sin(self.phase + self.speed * t)) * (1 - sm(age / 0.45))
        y = np.clip((self.y + self.fall * np.maximum(age, 0) ** 2).astype(np.int32), 0, H - 1)
        size = self.y / H
        f = self.dark.f.copy()
        core, halo = 0, 0
        for lo, hi, sx in ((0, 0.55, 1.1), (0.55, 0.8, 1.7), (0.8, 2, 2.6)):
            pick = (size >= lo) & (size < hi)
            spikes = np.zeros((H, W), np.float32)
            np.add.at(spikes, (y[pick], self.x[pick]), b[pick])
            core = core + gaussian_filter(spikes, (sx * 1.7, sx)) * 2 * np.pi * sx * sx * 1.7
            halo = halo + gaussian_filter(spikes, sx * 5) * 2 * np.pi * (sx * 5) ** 2 * 0.09
        core, halo = np.clip(core, 0, 1)[..., None], np.clip(halo, 0, 0.6)[..., None]
        f = f * (1 - halo) + HALO * halo
        f = f * (1 - core) + LIGHT * core
        return self.move.frame(t, f, s=0.03 * sm((t - LINE) / 5.0))


class Skin:
    """The water's skin, level with the lens."""

    def __init__(self):
        self.flat = Plate("myth/fall_flat", "ghost", white=0.9)
        self.rung = Plate("myth/fall_plate", "ghost", white=0.9)
        self.flat.f, self.rung.f = gaussian_filter(self.flat.f, (2.2, 2.2, 0)), gaussian_filter(self.rung.f, (1.2, 1.2, 0))     # all of it out of focus: the render's noise is no part of it
        r = np.random.default_rng(5)
        self.falls, rings = [], []
        for when in PLIPS[3:-1]:
            ys, d = land_at(r.uniform(1.0, 4.5))
            xs = r.uniform(70, W - 70)
            while abs(xs - LAND[0]) < 200:
                xs = r.uniform(70, W - 70)
            self.falls.append((when, xs, ys, d))
            rings.append((when, xs / W, ys / H, FLAT_RINGS, 0.7 + 1.6 / d))
        self.move = Move(self.flat, water=0.6, level=0.0, rings=rings)
        L = self.flat.L
        self.lights = sm((L - gaussian_filter(L, 40) - 0.05) / 0.2)[..., None]
        self.bg = gaussian_filter(self.flat.f, (30, 30, 0))
        self.ring_r = np.hypot(_X - LAND[0], (_Y - LAND[1]) / FLAT_RINGS)

    def frame(self, t):
        a = self.lights * sm((t - TO_SKIN) / (HIT - TO_SKIN))     # the lights behind go out as they come down
        f = self.flat.f * (1 - a) + self.bg * a
        age = t - HIT
        if age > 0:                                                # the rings the ink makes go out from where it went in
            o = sm((1500 * age ** 0.6 - self.ring_r) / 160)[..., None]
            f = f * (1 - o) + self.rung.f * o
        else:
            f = f.copy()
        for when, x, y_land, d in self.falls:
            u = 1 + (t - when) / 0.55
            s = 300.0 / d
            if 0 < u < 1:
                y = -120 + (y_land + 120) * u * u
                blur_ = 1.5 + 5 * abs(d - 1.2) / d
                tear(f, x, y, s * 0.16 + blur_, s * 0.32 + blur_, HALO, 0.5, 2.0)
                tear(f, x, y, s * 0.09 + blur_ * 0.5, s * 0.18 + blur_ * 0.5, LIGHT, 0.95, 2.4)
                ry = 2 * y_land - y
                if ry < H and y_land - y < 340:
                    tear(f, x, ry, s * 0.12 + blur_, s * 0.3 + blur_, LIGHT, 0.32 * (1 - (y_land - y) / 340), 0.6)
            elif -0.3 < t - when - 0.0 and t >= when and t < when + 0.3:
                tear(f, x, y_land, s * 0.5 + 2, s * 0.14 + 1, LIGHT, 0.55 * (1 - (t - when) / 0.3), 1.0)
        if -0.34 < age <= 0:                                       # the ink
            u = 1 + age / 0.34
            y = -160 + (LAND[1] + 160) * u * u
            tear(f, LAND[0], y, 7.5, 13.0, BLACK, 0.98, 1.0 + 5 * u)
            if LAND[1] - y < 300:
                tear(f, LAND[0], 2 * LAND[1] - y, 7, 14, BLACK, 0.4 * (1 - (LAND[1] - y) / 300), 0.5)
        return self.move.frame(t, f, s=0.0)


class Water:
    """Under it: bright, because the light is all coming down through the skin, and the ink in it."""

    def __init__(self):
        y = _Y / H
        top, low = np.array([0.91, 0.925, 0.885], np.float32), np.array([0.60, 0.70, 0.745], np.float32)
        u = (sm(y * 0.75 + 0.1) ** 0.9)[..., None]
        wash = 1 + 0.045 * np.clip(fft_noise(H, W, 3.0, 71), -2, 2)
        self.base = (top * (1 - u) + low * u) * wash[..., None] * (1 + 0.016 * fft_noise(H, W, 0.7, 72))[..., None]
        self.angle = np.arctan2(_X - W * 0.42, _Y + 900.0)
        r = np.random.default_rng(73)
        self.rays = np.cumsum(r.normal(0, 1, 4096))
        self.rays = gaussian_filter(self.rays - np.linspace(self.rays[0], self.rays[-1], 4096), 9)
        self.rays /= np.abs(self.rays).max()
        self.fade = np.exp(-y / 0.75)
        self.near = np.load(NEAR, mmap_mode="r")
        self.far = np.load(FAR, mmap_mode="r")
        self.falling = [(when, x + r.uniform(-30, 30), r.uniform(0.4, 1.0), r.uniform(150, 420)) for when, x in zip(PLIPS[1:-1], r.uniform(100, W - 100, len(PLIPS) - 2))]

    def frame(self, t, skin=UNDER):
        n = min(max(int(round((t - HIT) * FPS)), 0), FULL - 1)
        rays = np.interp(self.angle * 2600 + 2048 + 14 * t, np.arange(4096), self.rays).astype(np.float32)
        f = self.base * (1 + (0.16 * rays * self.fade)[..., None])
        # the lights that came in before the ink, each going down a little way and going out, and the ink each has let go, far off
        for when, x, a, depth in self.falling:
            age = t - HIT + 2.2 - (when - PLIPS[1]) * 0.25
            if 0 < age < 3.2 + (when - PLIPS[1]) * 0.5:
                y = skin + 120 + depth * (1 - math.exp(-age / 1.6))
                o = a * (1 - sm((age - 1.0 - (when - PLIPS[1]) * 0.5) / 2.2))
                tear(f, x, y, 16, 24, HALO, 0.42 * o, 1.2)
                tear(f, x, y, 4.5, 8, LIGHT, 0.95 * o, 1.8)
        d = ink_fluid.DENSITY * (np.asarray(self.far[n], np.float32) / 255) ** 2
        ys, xs = (_Y - skin + SKIN * FAR_SIZE * FAR_K) / FAR_K, (_X - FAR_X) / FAR_K
        d = gaussian_filter(sample(d, ys, xs), 2.2) * (1 + 5 * sm((t - STOP1 - 0.8) / 2.6))
        f = f * np.exp(-(0.42 * d)[..., None] * INK_K)
        d = ink_fluid.DENSITY * (np.asarray(self.near[n], np.float32) / 255) ** 2
        ys, xs = (_Y - skin + SKIN * NEAR_SIZE * NEAR_K) / NEAR_K, (_X - NEAR_X) / NEAR_K
        d = sample(d, ys, xs)
        d = 0.85 * d + 0.3 * gaussian_filter(d, 5) + 0.18 * gaussian_filter(d, 22)      # the threads, and what has bled from them into the water
        d = d * (1 + 7 * sm((t - STOP1 - 0.8) / 2.6))                                   # all that ink in a thread is darker than it was spread out
        return f * np.exp(-d[..., None] * INK_K)


def waterline(t, y):
    """The skin seen edge on as the lens goes through it: where it is across the frame."""
    return y + 6 * np.sin(_X[0] * 0.006 + t * 3.1) + 3 * np.sin(_X[0] * 0.017 - t * 5.3) + 1.5 * np.sin(_X[0] * 0.041 + t * 7.0)


def through(above, below, t, y):
    """Above the skin one picture, below it the other. Both float."""
    line = waterline(t, y)[None]
    a = sm((line - _Y) / 2.5)[..., None]
    f = below * (1 - a) + above * a
    under = np.exp(-np.maximum(_Y - line, 0) / 30)[..., None] * (1 - a)     # under the skin it mirrors the deep
    f = f * (1 - 0.3 * under)
    e = np.exp(-(((_Y - line) / 2.4) ** 2))[..., None]
    return f * (1 - 0.8 * e) + PAPER * 0.8 * e


_parts = {}


def part(name):
    if name not in _parts:
        if name == "bank":
            from ink_cut import RIVER
            _parts[name] = Move(Plate("ages/bone", "day"), centre=(0.5, 0.5), **RIVER)
        else:
            _parts[name] = dict(valley=Valley, skin=Skin, water=Water)[name]()
    return _parts[name]


def frame(t):
    if t < TO_SKIN - 0.25:
        return part("valley").frame(t)
    if t < TO_SKIN + 0.25:
        u = sm((t - TO_SKIN + 0.25) / 0.5)
        return (part("valley").frame(t) * (1 - u) + part("skin").frame(t) * u + 0.5).astype(np.uint8)
    if t < DIVE0:
        return part("skin").frame(t)
    if t < DIVE1:
        x = (t - DIVE0) / (DIVE1 - DIVE0)
        y = H + 40 - (H + 40 - UNDER) * (1 - (1 - x) ** 2.4)
        f = through(part("skin").frame(t).astype(np.float32) / 255, part("water").frame(t, y), t, y)
        return np.clip(f * 255 + 0.5, 0, 255).astype(np.uint8)
    if t < RISE0:
        return np.clip(part("water").frame(t) * 255 + 0.5, 0, 255).astype(np.uint8)
    x = min((t - RISE0) / (RISE1 - RISE0), 1.0)                 # up again: the skin comes down the frame, and the stream stays where it is
    y = -40 + (H + 80) * x ** 1.6
    f = through(part("bank").frame(t - RISE1, s=0.0).astype(np.float32) / 255, part("water").frame(t), t, y)
    return np.clip(f * 255 + 0.5, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    args = sys.argv[1:]
    if args[:1] == ["sim"]:
        simulate()
    elif args[:1] == ["strip"]:
        a, b = (float(args[1]), float(args[2])) if len(args) > 2 else (LINE, RISE1)
        cols, rows, n = 5, 4, 20
        out = Image.new("RGB", (cols * 512, rows * 288))
        for i in range(n):
            t = a + (b - a) * i / (n - 1)
            out.paste(Image.fromarray(frame(t)).resize((512, 288), Image.LANCZOS), ((i % cols) * 512, (i // cols) * 288))
        out.save(os.path.join(OUT, "one_strip.png"))
    else:
        a, b = (float(args[0]), float(args[1])) if len(args) > 1 else (LINE, RISE1)
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", os.path.join(OUT, "one.mp4")]
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for n in range(int(a * FPS), int(b * FPS)):
            pipe.stdin.write(frame(n / FPS).tobytes())
        pipe.stdin.close()
        pipe.wait()
