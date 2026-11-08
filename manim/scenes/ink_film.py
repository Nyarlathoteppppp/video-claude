"""AI 自传 — an ink-wash film (landscape 1920x1080, 30 fps). Storyboard: ink.scenes.md.

The whole film is one handscroll with no cuts: the camera travels right to left along a long sheet of xuan paper, and
everything on it is the same ink. Subtitles, times and small notes are read from the storyboard's table.

Three things make the ink:
- a paper simulation (wet pigment, water, fixed pigment): blooms, tide lines, the flood that washes the night away;
- ink particles that crawl from any shape to any other (tracks -> oracle glyphs, all writing -> numbers -> 精 -> 我);
- crisp high-resolution layers for what must read exactly (big characters, the numbers, the mass of text).

Two passes, so a copy change never re-simulates the ink:

    PY=python3        # pycairo is not needed: numpy, scipy, Pillow, ffmpeg on PATH
    $PY ink_film.py check                                 # subtitles vs reading time
    $PY ink_film.py plate 0 90 --name opening             # the ink alone -> ink/film_out/plate_opening.mp4
    $PY ink_film.py still 3 10 21 --from 0                # PNGs of the plate -> ink/film_out/stills/
    $PY ink_film.py qstill 95 105 120                     # stills of 千窗, which is drawn frame by frame and has no simulation
    $PY ink_film.py qplate 130.5                          # its plate -> ink/film_out/plate_qian.mp4
    $PY ink_film.py join --name sample                    # the two plates, dissolved -> ink/film_out/plate_sample.mp4
    $PY ink_film.py audio 130.5                           # self-made sound -> ink/film_out/ink_audio.wav
    $PY ink_film.py render --name sample                  # plate + subtitles + sound -> ink/film_out/sample.mp4

Checkpoints of the simulation are kept in ink/film_out/ckpt, so a shot can be re-run from the one before it.
Fonts: Xingkai, Kaiti, Libian from macOS, looked up in INK_FONTS (default ~/render_work/fonts) or the system's asset folder.
"""
import argparse
import glob
import math
import os
import pickle
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter1d, zoom
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "ink", "film_out")
SCENES = os.path.join(HERE, "ink.scenes.md")

W, H = 1920, 1080                     # the frame
SW, SH = 1280, 720                    # the simulation's view of it
SC = W / SW
WW = 12000                            # the length of the scroll, in simulation pixels
FPS = 30
PAPER = np.array([241, 234, 219], np.float32)     # warm xuan paper
INK_K = np.array([1.00, 1.03, 1.09], np.float32)  # thin ink is a little warm, thick ink close to black
INK_FLOOR = np.array([17, 15, 14], np.float32)


def sm(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ease_out(x):
    return 1 - (1 - np.clip(x, 0.0, 1.0)) ** 3


# ======================================================================================
# storyboard
# ======================================================================================
def rows():
    out = {}
    for line in open(SCENES, encoding="utf-8"):
        m = re.match(r"\|\s*(\d+)\s*\|\s*([\d.]+)–([\d.]+)\s*\|", line)
        if not m:
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        out[int(m[1])] = dict(n=int(m[1]), a=float(m[2]), b=float(m[3]), sub=c[2], note="" if c[4] in ("", "—") else c[4])
    return out


def need(s):
    han = sum(1 for ch in s if "一" <= ch <= "鿿")
    other = sum(1 for ch in s if not ("一" <= ch <= "鿿") and not ch.isspace())
    return max(2.0, 1 + han / 5 + other / 8)


def cmd_check():
    bad = 0
    prev = 0.0
    for n, x in sorted(rows().items()):
        msgs = []
        slot = x["b"] - x["a"]
        if need(x["sub"]) + 1.02 > slot + 1e-3:
            msgs.append(f"needs {need(x['sub']) + 1.02:.2f}s, has {slot:.2f}s")
        if x["note"] and need(x["note"]) > slot + 1e-3:
            msgs.append(f"note needs {need(x['note']):.2f}s")
        if len(x["sub"].replace(" ", "")) > 18:
            msgs.append("over 18 characters")
        if x["a"] < prev - 1e-3:
            msgs.append("overlaps the previous line")
        prev = x["b"]
        bad += bool(msgs)
        if msgs:
            print(f"#{n} {x['sub']}: " + "; ".join(msgs))
    print(f"{len(rows())} lines, {bad} with problems")
    return bad == 0


# ======================================================================================
# noise, fonts, brushes
# ======================================================================================
def noise2(h, w, beta, seed):
    """Tileable noise with a 1/f^beta spectrum, mean 0, std 1."""
    r = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1
    spec = (r.normal(size=f.shape) + 1j * r.normal(size=f.shape)) / f ** (beta / 2)
    spec[0, 0] = 0
    n = np.fft.irfft2(spec, s=(h, w))
    return ((n - n.mean()) / n.std()).astype(np.float32)


def font_file(name):
    root = os.environ.get("INK_FONTS", os.path.expanduser("~/render_work/fonts"))
    p = os.path.join(root, name)
    if os.path.exists(p):
        return p
    for q in glob.glob(f"/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/{name}"):
        return q
    for q in (f"/System/Library/Fonts/Supplemental/{name}", f"/System/Library/Fonts/{name}"):
        if os.path.exists(q):
            return q
    raise SystemExit(f"font {name} not found: put it in {root}")


_fonts = {}


def font(kind, size):
    faces = dict(xing=("Xingkai.ttc", 0), kai=("Kaiti.ttc", 0), kaib=("Kaiti.ttc", 3), kaih=("Kaiti.ttc", 5),
                 li=("Libian.ttc", 0), song=("Songti.ttc", 6), mono=("Menlo.ttc", 0))
    k = (kind, size)
    if k not in _fonts:
        f, i = faces[kind]
        _fonts[k] = ImageFont.truetype(font_file(f), size, index=i)
    return _fonts[k]


def glyph_mask(ch, kind, size, pad=None):
    """A character as a float mask (0-1), tightly centred in a square."""
    pad = size // 6 if pad is None else pad
    s = size + 2 * pad
    im = Image.new("L", (s, s), 0)
    ImageDraw.Draw(im).text((s / 2, s / 2), ch, 255, font=font(kind, size), anchor="mm")
    return np.asarray(im, np.float32) / 255


def chaikin(p, k=3):
    """Round a polyline's corners, keeping its ends."""
    for _ in range(k):
        q = np.empty((2 * len(p) - 2, 2))
        q[0::2] = p[:-1] * .75 + p[1:] * .25
        q[1::2] = p[:-1] * .25 + p[1:] * .75
        p = np.vstack([p[:1], q, p[-1:]])
    return p


def brush(paths, size, r0, seed=0):
    """Strokes as a mask: each path (points in 0-1) is walked by a round brush that presses, lifts and runs dry."""
    rng = np.random.default_rng(seed)
    ss = 3
    im = Image.new("L", (size * ss, size * ss), 0)
    d = ImageDraw.Draw(im)
    for path in paths:
        p = np.array(path, np.float64) * size * ss
        if len(p) >= 5:      # long paths are curves; the short ones keep their corners (the peaks of 山)
            p = chaikin(p)
        if len(p) == 1:
            r = r0 * ss * 1.3
            d.ellipse([p[0, 0] - r, p[0, 1] - r, p[0, 0] + r, p[0, 1] + r], fill=255)
            continue
        seg = np.hypot(*np.diff(p, axis=0).T)
        s = np.r_[0, np.cumsum(seg)]
        n = max(int(s[-1] / (r0 * ss * 0.25)), 8)
        u = np.linspace(0, s[-1], n)
        x = np.interp(u, s, p[:, 0])
        y = np.interp(u, s, p[:, 1])
        k = np.ones(5) / 5
        x = np.convolve(np.r_[[x[0]] * 2, x, [x[-1]] * 2], k, "valid")
        y = np.convolve(np.r_[[y[0]] * 2, y, [y[-1]] * 2], k, "valid")
        ph = rng.uniform(0, 6.28)
        for i in range(n):
            q = i / (n - 1)
            press = (0.62 + 0.38 * sm(q / 0.12)) * (1 - 0.55 * sm((q - 0.72) / 0.28)) * (1 + 0.13 * math.sin(ph + q * 7))
            r = r0 * ss * press
            d.ellipse([x[i] - r, y[i] - r, x[i] + r, y[i] + r], fill=255)
    im = im.resize((size, size), Image.LANCZOS)
    m = np.asarray(im, np.float32) / 255
    # a dry brush leaves streaks and a rough edge
    n = noise2(size, size, 1.2, seed + 5)
    return np.clip(m * (1.0 + 0.10 * n) - 0.04 * np.abs(n) * (m < 0.98), 0, 1)


# oracle-bone pictographs drawn from their usual forms (points in a unit square)
_water = [[(.50 + .13 * math.sin(i * .55), .06 + i * .0727)] for i in range(13)]
ORACLE = {
    "日": [[(.5, .10), (.72, .14), (.86, .32), (.88, .55), (.76, .78), (.52, .88), (.28, .80), (.14, .58), (.16, .32), (.30, .14), (.5, .10)],
          [(.36, .50), (.64, .50)]],
    "山": [[(.06, .84), (.94, .84)], [(.06, .84), (.22, .42), (.36, .84)], [(.34, .84), (.50, .10), (.66, .84)], [(.64, .84), (.78, .42), (.94, .84)]],
    "水": [[p[0] for p in _water], [(.18, .16), (.28, .24)], [(.12, .46), (.25, .52)], [(.17, .74), (.29, .80)],
          [(.82, .14), (.72, .22)], [(.88, .42), (.75, .48)], [(.83, .72), (.71, .78)]],
    "木": [[(.5, .06), (.5, .94)], [(.5, .36), (.26, .12)], [(.5, .36), (.74, .12)], [(.5, .62), (.20, .88)], [(.5, .62), (.80, .88)]],
    "鸟": [[(.30, .14), (.14, .22), (.08, .30), (.24, .30), (.28, .44), (.34, .60), (.48, .70), (.66, .72), (.92, .92)],
          [(.30, .14), (.44, .18), (.58, .32), (.72, .50), (.92, .92)],
          [(.40, .40), (.54, .48), (.62, .62)], [(.24, .21)],
          [(.44, .70), (.42, .88), (.32, .94)], [(.42, .88), (.50, .95)], [(.58, .72), (.58, .88), (.50, .95)]],
    "月": [[(.62, .08), (.36, .16), (.20, .40), (.24, .68), (.46, .88), (.72, .78), (.78, .50), (.74, .24), (.62, .08)], [(.44, .38), (.60, .42)], [(.42, .58), (.60, .60)]],
    "人": [[(.64, .08), (.50, .42), (.30, .92)], [(.50, .42), (.70, .68)]],
}


def bird_track(size, rot, seed):
    """A three-toed print with a short back toe, pointing along rot."""
    c, s = math.cos(rot), math.sin(rot)
    def R(x, y):
        return (.5 + c * x - s * y, .5 + s * x + c * y)
    paths = [[R(0, 0), R(.40, 0)], [R(0, 0), R(.27, -.26)], [R(0, 0), R(.27, .26)], [R(0, 0), R(-.2, 0)]]
    return brush(paths, size, size * .042, seed)


def beast_track(size, rot, seed):
    """A paw: a pad and four toes."""
    c, s = math.cos(rot), math.sin(rot)
    def R(x, y):
        return (.5 + c * x - s * y, .5 + s * x + c * y)
    pad = [[R(-.06, -.09), R(.02, 0), R(-.06, .09)]]
    toes = [[R(.2, -.2)], [R(.29, -.07)], [R(.29, .07)], [R(.2, .2)]]
    return brush(pad + toes, size, size * .072, seed)


# ======================================================================================
# the paper
# ======================================================================================
def make_paper(seed=3):
    """(detail, tone) tiles at frame resolution: long fibres, cloudy thickness, fine grain."""
    n = 2048
    cloud = noise2(n, n, 2.4, seed)
    rng = np.random.default_rng(seed)
    im = Image.new("L", (n, n), 128)
    d = ImageDraw.Draw(im)
    for _ in range(26000):
        x, y = rng.uniform(0, n, 2)
        a = rng.uniform(0, 3.14)
        L = rng.uniform(14, 90)
        b = a + rng.normal(0, .5)
        pts = [(x, y), (x + L * .5 * math.cos(a), y + L * .5 * math.sin(a)), (x + L * .5 * (math.cos(a) + math.cos(b)), y + L * .5 * (math.sin(a) + math.sin(b)))]
        d.line(pts, fill=int(128 + rng.choice([-1, 1]) * rng.uniform(10, 48)), width=1)
    fib = gaussian_filter(np.asarray(im, np.float32) - 128, .55, mode="wrap") / 22
    grain = noise2(n, n, .35, seed + 1)
    detail = np.clip(.60 * fib + .55 * grain + .25 * noise2(n, n, 1.1, seed + 2), -3, 3).astype(np.float32)
    tone = (1 + .012 * cloud + .012 * fib + .006 * grain).astype(np.float32)
    return detail, tone


def mountains(xa, xb, seed=11):
    """Layered mist mountains as an ink-density field over simulation pixels xa..xb."""
    w = xb - xa
    y = np.arange(SH, dtype=np.float32)[:, None]
    out = np.zeros((SH, w), np.float32)
    tile = noise2(1024, 2048, 2.1, seed)
    # brush texture: soft strokes running down the slopes
    streak = gaussian_filter(noise2(1024, 2048, 1.0, seed + 1), (3.0, 1.1), mode="wrap")
    streak /= streak.std()
    idx = np.arange(w) % 2048
    tx = tile[:SH][:, idx]
    sx = streak[:SH][:, idx]
    # the range is massed on the left and sinks into mist on the right, which leaves the paper bare for the title
    sink = 150 * sm((np.arange(xa, xb, dtype=np.float32) - 10900) / 700)
    layers = [(285, 70, .28, 70, 1, .25), (325, 85, .50, 70, 2, .55), (370, 95, .80, 60, 3, 1.0), (425, 85, 1.25, 48, 4, 1.2)]
    for base, amp, tone, fall, i, sunk in layers:
        r = np.random.default_rng(seed + i)
        ridge = np.zeros(w, np.float32)
        for k, (L, a) in enumerate([(1050, 1.0), (470, .50), (190, .20), (80, .07), (34, .02)]):
            ph = r.uniform(0, 6.28)
            xs = np.arange(w) / L * 6.28 + ph
            ridge += a * (1 - 2 * np.abs(np.sin(xs / 2 + .6 * np.sin(xs * .37 + ph)))) if k < 3 else a * np.sin(xs)
        ridge = base + amp * .55 * ridge + sink * sunk
        d = y - ridge[None, :]
        inside = d > 0
        body = tone * np.exp(-np.maximum(d, 0) / fall) * (.72 + .26 * np.clip(.7 * sx + .6 * tx, -1.6, 1.6))
        edge = tone * .6 * np.exp(-np.maximum(d, 0) / 2.0) * (.7 + .25 * np.clip(tx, -1, 1))
        layer = np.where(inside, np.maximum(body, 0) + edge, 0)
        out = out * (1 - .86 * np.exp(-np.maximum(d, 0) / (fall * 2.2)) * inside) + layer
        if i >= 3:   # dots of moss and pines along the near ridges
            xs = r.integers(0, w, int(w / (14 if i == 4 else 26)))
            for x0 in xs:
                yy = int(ridge[x0] + r.uniform(-1, 6))
                if yy >= SH - 6:      # this stretch of the ridge has sunk below the paper
                    continue
                rad = r.uniform(1.0, 3.2)
                x1, x2, y1, y2 = max(x0 - 5, 0), min(x0 + 6, w), max(yy - 5, 0), min(yy + 6, SH)
                gy, gx = np.mgrid[y1:y2, x1:x2]
                out[y1:y2, x1:x2] += tone * 1.6 * np.exp(-((gx - x0) ** 2 + (gy - yy) ** 2) / (rad * rad))
    return gaussian_filter(out, .55)


def ground(xa, xb, seed=61):
    """The dark bank in the foreground that the tracks are printed on."""
    X = np.arange(xa, xb, dtype=np.float32)
    line = 578 + 26 * np.sin(X / 410.0) + 11 * np.sin(X / 130.0 + 1)
    y = np.arange(SH, dtype=np.float32)[:, None]
    n = noise2(1024, 2048, 1.7, seed)[:SH][:, np.arange(xb - xa) % 2048]
    return (sm((y - line[None, :]) / 44) * (1.15 + .18 * n)).astype(np.float32)


# ======================================================================================
# ink that crawls: particles from one density to another
# ======================================================================================
def sample(mask, n, rng, x0=0.0, y0=0.0, scale=1.0):
    """n points distributed like mask; returns x, y (world px) and the total mass."""
    m = np.maximum(mask, 0).ravel()
    tot = m.sum()
    idx = rng.choice(m.size, n, p=m / tot)
    y, x = np.unravel_index(idx, mask.shape)
    return x0 + (x + rng.random(n)) * scale, y0 + (y + rng.random(n)) * scale, float(tot) * scale * scale


def match(xa, ya, xb, yb):
    """Pairs that do not cross much: columns by x, then each column by y."""
    n = len(xa)
    ia, ib = np.argsort(xa), np.argsort(xb)
    k = max(int(math.sqrt(n) / 5), 1)
    out_a, out_b = [], []
    for ca, cb in zip(np.array_split(ia, k), np.array_split(ib, k)):
        out_a.append(ca[np.argsort(ya[ca])])
        out_b.append(cb[np.argsort(yb[cb])])
    a = np.concatenate(out_a)
    b = np.concatenate(out_b)
    return xa[a], ya[a], xb[b], yb[b]


class Morph:
    """Ink moving from src to tgt over [t0, t0+dur], optionally through a point or along a river."""

    def __init__(self, src, tgt, n, t0, dur, seed=0, stagger=.35, swirl=26.0, via=None, burst=False, order=None):
        r = np.random.default_rng(seed)
        xa, ya, ma = sample(src[0], n, r, src[1], src[2], src[3])
        xb, yb, mb = sample(tgt[0], n, r, tgt[1], tgt[2], tgt[3])
        self.xa, self.ya, self.xb, self.yb = match(xa, ya, xb, yb)
        self.wa, self.wb = ma / n, mb / n
        self.t0, self.dur = t0, dur
        self.swirl = swirl
        self.white = False         # light (bare paper) instead of ink
        self.via = via
        self.burst = burst
        self.jit = r.normal(size=(2, n)).astype(np.float32)
        if order == "right":       # the right side of the scroll sets off first
            q = (self.xa - self.xa.min()) / (np.ptp(self.xa) + 1)
            self.delay = (1 - q) * stagger + r.random(n) * .08
        else:
            self.delay = r.random(n) * stagger
        self.pa = NOISE_A[(self.ya.astype(int) // 3) % 1024, (self.xa.astype(int) // 3) % 1024] * 2.2
        self.pb = NOISE_B[(self.ya.astype(int) // 3) % 1024, (self.xa.astype(int) // 3) % 1024] * 2.2

    def at(self, t):
        u = (t - self.t0) / self.dur
        s = np.clip((u - self.delay) / (1 - self.delay.max() + 1e-6), 0, 1)
        if self.burst:   # everything falls into one drop, then flies out
            c = self.via
            in_ = sm(s / .5) ** 2
            out_ = ease_out((s - .5) / .5)
            x = np.where(s < .5, self.xa + (c[0] - self.xa) * in_, c[0] + (self.xb - c[0]) * out_)
            y = np.where(s < .5, self.ya + (c[1] - self.ya) * in_, c[1] + (self.yb - c[1]) * out_)
            amp = self.swirl * np.sin(np.pi * s)
        elif self.via is not None:   # a river: pulled to its line, then on to the target
            e = sm(s)
            rx, ry = self.via(self.xa + (self.xb - self.xa) * e)
            pull = np.sin(np.pi * e) ** 2 * .96
            x = self.xa + (self.xb - self.xa) * e
            y = self.ya + (self.yb - self.ya) * e
            y = y + (ry - y) * pull
            amp = self.swirl * (1 - np.abs(2 * e - 1)) ** 1.5
        else:
            e = sm(s)
            x = self.xa + (self.xb - self.xa) * e
            y = self.ya + (self.yb - self.ya) * e
            amp = self.swirl * np.sin(np.pi * e)
        x = x + amp * (np.cos(self.pa + 3 * s) + .12 * self.jit[0])
        y = y + amp * (np.sin(self.pb + 3 * s) + .12 * self.jit[1])
        w = self.wa + (self.wb - self.wa) * sm(s)
        return x, y, w


def splat(x, y, w, xa, width, height=SH):
    """Bilinear point splat onto a height x width grid whose left edge is world xa."""
    x = x - xa
    keep = (x >= 0) & (x < width - 1) & (y >= 0) & (y < height - 1)
    x, y = x[keep], y[keep]
    w = w[keep] if np.ndim(w) else np.full(x.shape, w, np.float32)
    ix, iy = x.astype(np.int32), y.astype(np.int32)
    fx, fy = x - ix, y - iy
    i = iy * width + ix
    size = height * width
    g = (np.bincount(i, w * (1 - fx) * (1 - fy), size) + np.bincount(i + 1, w * fx * (1 - fy), size)
         + np.bincount(i + width, w * (1 - fx) * fy, size) + np.bincount(i + width + 1, w * fx * fy, size))
    return g.reshape(height, width).astype(np.float32)


# ======================================================================================
# crisp layers
# ======================================================================================
class Layer:
    """A mask at frame resolution, anchored to the scroll. a(t) is its strength; a reveal grows the ink into the shape."""

    def __init__(self, mask, wx, wy, strength=3.0, t_in=0.0, dur_in=.6, t_out=1e9, dur_out=.4, cols=None):
        self.m = mask.astype(np.float32)
        self.wx, self.wy = wx, wy          # the mask's top-left corner, in simulation pixels of the scroll
        self.s = strength
        self.t_in, self.dur_in, self.t_out, self.dur_out = t_in, dur_in, t_out, dur_out
        self.soft = gaussian_filter(self.m, 3.0) if cols is None and mask.size < 3_000_000 else None
        if self.soft is not None:
            r = np.random.default_rng(int(mask.size) % 9973)
            self.n = gaussian_filter(r.normal(size=mask.shape), 3).astype(np.float32)
            self.n /= self.n.std()
        self.cols = cols                   # (start time, seconds per column, column width): written column by column
        self.neg = False                   # bare paper instead of ink
        self.glow = None                   # for a negative layer: its light on the clouds around it

    def field(self, t):
        if t < self.t_in or t > self.t_out + self.dur_out:
            return None
        k = 1 - float(sm((t - self.t_out) / self.dur_out))
        u = (t - self.t_in) / self.dur_in
        if self.cols is not None:
            t0, per, cw = self.cols
            nc = self.m.shape[1] // cw + 1
            c = (self.m.shape[1] - 1 - np.arange(self.m.shape[1])) // cw   # from the right
            a = sm((t - t0 - c * per) / .35)[None, :]
            return self.m * a * self.s * k
        if u >= 1 or self.soft is None:
            return self.m * self.s * k * float(sm(u * 3))
        thr = 1.05 - .95 * float(ease_out(u))
        a = sm((self.soft + .16 * self.n * (1 - u) - thr) / .14)
        return self.m * a * self.s * k


GLYPH_HITS = [7.7, 8.2, 8.7, 9.2, 9.7]   # the five first characters strike the sky
RIFT_OPEN = 10.4                         # the sky tears
RIFT_CLOSE = 17.4                        # and starts to close, until the moon is all that is left of it
RIFT_X = 10830.0                         # where it tears, on the scroll: right behind the character for the sun


def beats(r):
    """The wall of numbers' heartbeat: slow at first, then faster, until it stops just before it speaks."""
    t, a, end, out = r[11]["a"] + .2, r[11]["a"], r[13]["a"] - .2, []
    while t < end:
        out.append(t)
        t += 1.5 - .75 * (t - a) / (end - a)
    return out


class Digits(Layer):
    """The wall of numbers, alive: on each beat a ring runs out from the centre, darkens what it crosses and changes its digits."""

    def __init__(self, masks, wx, wy, strength, t_in, dur_in, beat_times):
        super().__init__(masks[0], wx, wy, strength, t_in, dur_in)
        self.soft = None
        self.masks, self.beats = masks, beat_times
        self.hole = None
        h, w = masks[0].shape
        yy, xx = np.ogrid[:h, :w]
        self.rr = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)).astype(np.float32)

    def field(self, t):
        f = super().field(t)
        if f is None:
            return None
        past = [b for b in self.beats if b <= t]
        if past:
            n, age = len(past), t - past[-1]
            R = age * 2.6
            now, before = self.masks[n % len(self.masks)], self.masks[(n - 1) % len(self.masks)]
            ring = 1 + 1.3 * np.exp(-((self.rr - R) / .2) ** 2) * math.exp(-age * 1.4)
            f = (np.where(self.rr < R, now, before) * self.s * ring).astype(np.float32)
            s = 1 + .016 * math.exp(-age * 6) * min(1.0, .3 + n / 4)
            h, w = f.shape
            f = np.asarray(Image.fromarray(f).transform((w, h), Image.AFFINE, (1 / s, 0, w / 2 * (1 - 1 / s), 0, 1 / s, h / 2 * (1 - 1 / s)), Image.BILINEAR)).copy()
        if self.hole:      # the digits clear away from where it speaks
            m, ox, oy, t0, dur = self.hole
            a = float(sm((t - t0) / dur))
            if a > 0:
                f = f.copy()
                f[oy:oy + m.shape[0], ox:ox + m.shape[1]] *= 1 - m * a
        return f


def text_block(lines, size, kind, gap, h):
    """Vertical columns of text as a mask of height h; columns run right to left."""
    w = (size + gap) * len(lines)
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    f = font(kind, size)
    for i, line in enumerate(lines):
        x = w - (i + .5) * (size + gap)
        for j, ch in enumerate(line):
            d.text((x, (j + .5) * (size + 1)), ch, 255, font=f, anchor="mm")
    return np.asarray(im, np.float32) / 255


CORPUS = ("太史公曰余读孔氏书想见其为人适鲁观仲尼庙堂车服礼器诸生以时习礼其家余祗回留之不能去云关关雎鸠在河之洲窈窕淑女君子好逑"
          "学而时习之不亦说乎有朋自远方来不亦乐乎人不知而不愠不亦君子乎天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏"
          "艰难苦恨繁霜鬓潦倒新停浊酒杯无边落木萧萧下不尽长江滚滚来万里悲秋常作客百年多病独登台亲爱的你好吗今晚月亮很好我又想起你了"
          "本院认为被告人的行为已构成犯罪依照法律规定判决如下今天还是没找到工作有人吗有人回复一下吗楼主说得对我不这么认为谢谢大家"
          "大江东去浪淘尽千古风流人物前不见古人后不见来者念天地之悠悠独怆然而涕下春眠不觉晓处处闻啼鸟夜来风雨声花落知多少"
          "为天地立心为生民立命为往圣继绝学为万世开太平妈妈我到了别担心明天记得吃药对不起我错了我们分手吧祝你幸福")
BIG_PHRASES = ["关关雎鸠", "亲爱的", "判决如下", "有人吗", "for i in range(n):", "太史公曰", "今晚月亮很好", "对不起", "无边落木", "print(\"hello\")", "没人回复", "人之初"]


def mass_text(w, h, seed=8):
    """All of it: a sheet of fine columns with phrases a viewer can catch in passing."""
    r = np.random.default_rng(seed)
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    f = font("kai", 19)
    cs = list(CORPUS)
    r.shuffle(cs)
    k = 0
    for cx in range(w - 12, 0, -23):
        tone = int(r.uniform(90, 215))
        y = int(r.uniform(4, 18))
        while y < h:
            d.text((cx, y), cs[k % len(cs)], tone, font=f, anchor="mm")
            k += 1
            y += 20
    # carve room for the phrases, then write them bigger
    for i, p in enumerate(BIG_PHRASES * 2):
        size = int(r.uniform(34, 56))
        fb = font("mono" if p.isascii() else "xing", size if not p.isascii() else size // 2 + 8)
        x = int((i + .5) / (len(BIG_PHRASES) * 2) * w + r.uniform(-60, 60))
        y = int(r.uniform(70, h - 280))
        if p.isascii():
            bw = d.textlength(p, font=fb)
            d.rectangle([x - 8, y - 8, x + bw + 8, y + fb.size + 8], fill=0)
            d.text((x, y), p, 255, font=fb)
        else:
            d.rectangle([x - size * .62, y - 6, x + size * .62, y + (size + 2) * len(p) + 6], fill=0)
            for j, ch in enumerate(p):
                d.text((x, y + (j + .5) * (size + 2)), ch, 255, font=fb, anchor="mm")
    return np.asarray(im, np.float32) / 255


def number_block(w, h, seed=13):
    """A wall of model weights."""
    r = np.random.default_rng(seed)
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    f = font("mono", 24)
    for y in range(6, h - 28, 30):
        x = 4
        while x < w - 110:
            v = r.normal(0, .6)
            s = f"{v:+.4f}" if r.random() < .78 else f"{v * 4:+.2f}"
            d.text((x, y), s, int(r.uniform(120, 255)), font=f)
            x += d.textlength(s, font=f) + 14
    return np.asarray(im, np.float32) / 255


# ======================================================================================
# the film
# ======================================================================================
NOISE_A = noise2(1024, 1024, 2.8, 21)
NOISE_B = noise2(1024, 1024, 2.8, 22)
# (t, camera centre, whip): the camera drifts; 46.5-52 it whips along the scroll and stops at each band of writing
CAM = [(0.0, 11000.0, 0), (41.5, 10500.0, 0), (46.5, 10440.0, 0),
       (47.4, 8620.0, 1), (48.0, 8600.0, 0), (48.4, 8190.0, 1), (48.9, 8172.0, 0), (49.3, 7790.0, 1), (49.8, 7772.0, 0),
       (50.2, 7380.0, 1), (50.7, 7362.0, 0), (51.1, 6940.0, 1), (51.5, 6922.0, 0), (52.0, 6050.0, 1), (57.0, 6010.0, 0),
       (82.0, 5990.0, 0), (92.0, 5985.0, 0)]


def cam(t):
    """The camera's centre on the scroll at time t."""
    for (t0, x0, _), (t1, x1, whip) in zip(CAM, CAM[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0)
            return x0 + (x1 - x0) * (float(sm(u)) if whip else u)
    return CAM[-1][1] + (t - CAM[-1][0]) * -9.5


def cam_speed(t):
    return abs(cam(t + .5 / FPS) - cam(t - .5 / FPS)) * FPS


def river(x):
    """The line all writing runs along on its way to the numbers."""
    return x, SH * .52 + 55 * np.sin(x / 150.0) + 22 * np.sin(x / 57.0)


class Film:
    def __init__(self):
        self.r = rows()
        self.rng = np.random.default_rng(1)
        self.F = np.zeros((SH, WW), np.float32)     # pigment fixed in the paper
        self.P = np.zeros((SH, WW), np.float32)     # pigment carried by water
        self.Wt = np.zeros((SH, WW), np.float32)    # water
        self.white = np.zeros((SH, WW), np.float32)  # bare paper showing through: the grain on the ground
        self.t = 0.0
        self.layers = {}
        self.morphs = []
        self.done = set()
        self.sky = Sky()                            # lives until the flood bakes it into the paper
        self.grains = Grains()
        self.drips = []
        self.smokes = []
        self.wash = None
        self.h = self._paper_height()
        # the scroll: the mountains of the myth on the first stretch, then bands of writing from oldest to newest
        xa = 8960
        self.F[:, xa:] = mountains(xa, WW) + ground(xa, WW)
        self._write_bands()
        self._tracks()

    def _paper_height(self):
        n = noise2(1024, 2048, 1.5, 31) * .5 + gaussian_filter(noise2(1024, 2048, .3, 32), .6, mode="wrap") * .9
        n = np.clip(.5 + .22 * n, 0, 1).astype(np.float32)
        return np.tile(n[:SH], (1, WW // 2048 + 1))[:, :WW]

    # ---- 昔者仓颉作书: tracks of light on the dark ground; they rise, and strike the sky as the first characters ---------
    def _tracks(self):
        r = np.random.default_rng(5)
        self.track_events = []
        self.tmask = np.zeros((SH, WW), np.float32)
        x, y = 11560.0, 628.0
        for i in range(42):
            rot = math.pi + r.normal(0, .14)
            side = 1 if i % 2 else -1
            bird = not (9 <= i % 21 <= 15)
            size = 56 if bird else 50
            m = bird_track(size, rot, i) if bird else beast_track(size, rot, i)
            px = x + r.normal(0, 2)
            py = y + side * (10 if bird else 13) + 8 * math.sin(i * .34) + r.normal(0, 1.5)
            self.track_events.append((-3.0 + i * .205, m, int(px), int(py)))
            x -= 31
        self.track_i = 0
        while self.track_events[self.track_i][0] <= 0:
            self._stamp_track(self.track_i)
            self.track_i += 1

    def _stamp_track(self, i):
        _, m, px, py = self.track_events[i]
        s = m.shape[0]
        ya, xa = py - s // 2, px - s // 2
        self.tmask[ya:ya + s, xa:xa + s] += m

    def _first_glyphs(self):
        row = [("鸟", 11150, 120), ("山", 10310, 120), ("水", 10520, 66), ("木", 10940, 66), ("日", 10730, 30)]
        size = 300
        by_x = sorted(row, key=lambda q: q[1])
        xs = np.nonzero(self.tmask.sum(axis=0) > 0)[0]
        cuts = np.linspace(xs.min(), xs.max() + 1, len(row) + 1).astype(int)
        for i, q in enumerate(row):
            ch, wx, wy = q
            hit = GLYPH_HITS[i]
            m = brush(ORACLE[ch], size, size * .026, seed=ord(ch) % 97)
            ms = zoom(m, 1 / SC, order=1)
            k = by_x.index(q)
            mo = Morph((self.tmask[:, cuts[k]:cuts[k + 1]], cuts[k], 0, 1), (ms, wx, wy, 1), 70000, hit - 2.0, 2.0, seed=2 + i, stagger=.25, swirl=40)
            mo.white = True
            self.morphs.append((mo, hit - .12, .25))
            pad = 90           # room for its light to fade out before the layer's edge (60 simulation px)
            mp = np.pad(m, pad)
            l = Layer(mp, wx - 60, wy - 60, 1.0, hit - .15, .25, t_out=self.r[2]["a"] + 1.2, dur_out=3.2)
            l.neg = True
            l.glow = np.clip(gaussian_filter(mp, 30) * 2.6, 0, 1).astype(np.float32)
            self.layers["g" + ch] = l

    def _blot(self, x, y, rad, ink):
        R = int(rad * 3) + 2
        ya, yb, xa, xb = max(y - R, 0), min(y + R + 1, SH), x - R, x + R + 1
        if ya >= yb:
            return
        gy, gx = np.ogrid[ya:yb, xa:xb]
        m = np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (rad * rad)).astype(np.float32)
        self.P[ya:yb, xa:xb] += m * ink
        self.Wt[ya:yb, xa:xb] += m * .55

    # ---- 鬼夜哭 ---------------------------------------------------------------------------------------------------
    def _ghost(self, t):
        size = 840
        wx = int(cam(t) - SW / 2 + 30 + 120)
        m = glyph_mask("鬼", "kaih", size, 40)
        self.ghost_at = (wx, 8)
        self.layers["ghost"] = Layer(m, wx, 8, 3.6, t, 1.1)
        # tears: runs of ink from the undersides of the strokes
        ms = zoom(m, 1 / SC, order=1) > .5
        below = np.roll(ms, -9, axis=0)
        edge = ms & ~below
        ys, xs = np.nonzero(edge)
        r = self.rng
        pick = r.choice(len(xs), 70, replace=False)
        for k in pick:
            self.drips.append(dict(x=float(wx + xs[k]), y=float(8 + ys[k]), v=r.uniform(26, 118), w=r.uniform(1.4, 4.2),
                                   t0=t + 1.0 + r.uniform(0, 9.0), ph=r.uniform(0, 6.28), life=r.uniform(4, 14)))

    def _drips(self, t, dt):
        for d in self.drips:
            if t < d["t0"] or t > d["t0"] + d["life"] or d["y"] >= SH - 6:
                continue
            age = t - d["t0"]
            d["y"] += d["v"] * dt * (1.15 - .6 * min(age / d["life"], 1))
            d["x"] += .25 * math.sin(d["ph"] + d["y"] * .021) * dt * 30
            R = d["w"] * (1.0 - .35 * min(age / d["life"], 1))
            x, y = int(d["x"]), int(d["y"])
            ri = int(R) + 2
            ya, yb = max(y - ri, 0), min(y + ri + 1, SH)
            gy, gx = np.ogrid[ya:yb, x - ri:x + ri + 1]
            self.F[ya:yb, x - ri:x + ri + 1] = np.maximum(self.F[ya:yb, x - ri:x + ri + 1], np.exp(-((gx - d["x"]) ** 2 + (gy - d["y"]) ** 2) / (R * R)) * 3.3)

    # ---- 并非如此: the flood ----------------------------------------------------------------------------------------
    def _start_wash(self, t):
        x0 = int(cam(t) - SW / 2) - 200
        w = SW + 200
        sky = self.sky.field(t, x0, w, w, SH)
        self.sky.moon(sky, t, x0, 1.0)
        self.F[:, x0:x0 + w] += sky
        self.F *= 1 - np.clip(self.white, 0, 1) * .97
        self.sky = None
        self.white = None
        for k in list(self.layers):
            l = self.layers.pop(k)
            if l.neg:
                continue
            ms = zoom(l.m, 1 / SC, order=1) * l.s
            y0, x0_ = int(l.wy), int(l.wx)
            ye = min(y0 + ms.shape[0], SH)
            self.F[y0:ye, x0_:x0_ + ms.shape[1]] += ms[:ye - y0]
        self.drips = []
        self.grains.a = self.grains.a[:0]
        self.wash = dict(t0=t, xa=8960, xb=WW)

    def _wash(self, t, dt):
        w = self.wash
        c = cam(t)
        a, b = max(w["xa"], int(c - 900)), min(WW, int(c + 1000))
        x = np.arange(a, b, dtype=np.float32)[None, :]
        front = (c + 720) - (t - w["t0"]) * 660
        behind = sm((x - front) / 130.0)
        F, P, Wt = self.F[:, a:b], self.P[:, a:b], self.Wt[:, a:b]
        lift = np.minimum(behind * dt * 4.2, 1.0)
        dF = F * lift
        F -= dF
        P += dF
        Wt[:] = np.maximum(Wt, behind)
        n = self._curl(t * 1.4, a, b, 3)
        vx = (-430 - 180 * n[2]) * behind + n[0] * 310 * behind
        vy = n[1] * 310 * behind
        P[:] = advect(P, vx, vy, dt)
        P *= np.exp(-dt * behind / .62)
        # what the water leaves behind: a faint stain
        stain = gaussian_filter(P, 2.5) * dt * .10
        F += stain
        if t - w["t0"] > 4.2:
            self.P[:, w["xa"]:] = 0
            self.Wt[:, w["xa"]:] = 0
            self.F[:, w["xa"]:] = np.minimum(self.F[:, w["xa"]:], .05) * .75
            self.wash = None
            self.done.add("washed")

    # ---- 很远的将来: bands of writing, oldest to newest --------------------------------------------------------------
    def _write_bands(self):
        r = np.random.default_rng(17)
        x = 8850
        for name, width in (("oracle", 460), ("li", 400), ("kai", 400), ("song", 420), ("mono", 460)):
            xb = x
            x -= width
            w = int(width * SC)
            im = Image.new("L", (w, H), 0)
            d = ImageDraw.Draw(im)
            if name == "oracle":
                arr = np.zeros((H, w), np.float32)
                keys = list(ORACLE)
                for cy in range(70, H - 80, 118):
                    for cx in range(10, w - 110, 118):
                        if r.random() < .84:
                            m = brush(ORACLE[keys[r.integers(len(keys))]], 92, 3.0, seed=int(cx + cy))
                            arr[cy:cy + 92, cx:cx + 92] = np.maximum(arr[cy:cy + 92, cx:cx + 92], m)
            else:
                cs = list(CORPUS)
                r.shuffle(cs)
                if name == "mono":
                    f = font("mono", 24)
                    code = ["def attention(q, k, v):", "    s = q @ k.T / sqrt(d)", "    return softmax(s) @ v", "for step in range(n):", "    loss = model(x, y)",
                            "    loss.backward()", "    optimizer.step()", "print('hello, world')", "while True:", "    x = next(batch)"]
                    for i, y in enumerate(range(50, H - 50, 38)):
                        d.text((14, y), code[i % len(code)], int(r.uniform(150, 255)), font=f)
                else:
                    size = 48 if name == "li" else 40 if name == "kai" else 34
                    f = font(name, size)
                    k = 0
                    for cx in range(w - size, 0, -(size + 16)):
                        for y in range(40, H - 40, size + 4):
                            d.text((cx, y), cs[k % len(cs)], int(r.uniform(140, 240)), font=f, anchor="mm")
                            k += 1
                arr = np.asarray(im, np.float32) / 255
            fade = sm(np.minimum(np.arange(w), w - 1 - np.arange(w)) / 70.0)[None, :]
            ms = zoom(arr * fade, 1 / SC, order=1)
            h_, w_ = ms.shape
            xa_ = xb - w_
            self.F[:h_, xa_:xb] += ms * 1.1
        self.band_end = x

    # ---- all writing -> numbers -> 你好 -> 精 -> 我 ---------------------------------------------------------------------
    def _mass_in(self, t, t_end):
        w, h = int(1400 * SC), H
        m = mass_text(w, h)
        wx = int(cam(t + 2.5) - 700)
        cols = (t, (t_end - t - .9) / (w // 23), 23)
        self.layers["mass"] = Layer(m, wx, 0, 1.6, t, .1, cols=cols)
        self.mass_mask = (m, wx)

    def _to_numbers(self, t, dur):
        m, wx = self.mass_mask
        ms = zoom(m, 1 / SC, order=1) * 1.6
        walls = [number_block(1840, 1010, 13 + i) for i in range(3)]
        nx, ny = int(cam(t + dur) - 613), 23
        ns = zoom(walls[0], 1 / SC, order=1) * 2.0
        self.layers["mass"].t_out = t
        self.layers["mass"].dur_out = .35
        self.layers["num"] = Digits(walls, nx, ny, 2.0, t + dur - .55, .7, beats(self.r))
        self.morphs.append((Morph((ms, wx, 0, 1), (ns, nx, ny, 1), 620000, t, dur, seed=3, stagger=.34, swirl=46, via=river, order="right"), t + dur - .55, .5))
        self.num_at = (nx, ny, ns)

    def _hello(self, t):
        """After all that noise, the most ordinary words there are, and not large."""
        nx, ny, _ = self.num_at
        ni, hao = glyph_mask("你", "xing", 190, 10), glyph_mask("好", "xing", 190, 10)
        h_, w_ = ni.shape
        cx = cam(t + 1) - SW / 2 + 860 / SC
        wy = int((SH - h_ / SC) / 2)
        self.layers["ni"] = Layer(ni, int(cx - w_ / SC - 6), wy, 3.4, t + .6, .55)
        self.layers["hao"] = Layer(hao, int(cx + 6), wy, 3.4, t + 1.2, .55)
        w, h = 1000, 560
        yy, xx = np.ogrid[:h, :w]
        hole = sm((1 - np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))) / .35).astype(np.float32)
        self.layers["num"].hole = (hole, int(round((cx - nx) * SC - w / 2)), int(round((SH / 2 - ny) * SC - h / 2)), t, 1.0)

    def _become(self, t, dur, ch, key, size=900, burst=False, seed=4, n=700000, src_keys=()):
        """The ink on the page gathers into one huge character."""
        srcs = []
        c = cam(t + dur)
        gx = int(c - size / SC / 2 - 115)
        gy = int((SH - size / SC) / 2) - 2
        m = glyph_mask(ch, "kaih", size, 10)
        ms = zoom(m, 1 / SC, order=1) * 3.2
        xa = int(c - SW / 2 - 150)
        wid = SW + 300
        src = np.zeros((SH, wid), np.float32)
        for k in src_keys:
            l = self.layers.pop(k)
            q = zoom(l.m, 1 / SC, order=1) * l.s
            x0, y0 = int(l.wx) - xa, int(l.wy)
            xe = min(x0 + q.shape[1], wid)
            ye = min(y0 + q.shape[0], SH)
            src[y0:ye, x0:xe] = np.maximum(src[y0:ye, x0:xe], q[:ye - y0, :xe - x0])
        view = self.F[:, xa:xa + wid]
        src += np.maximum(view - .15, 0)
        view[:] = np.minimum(view, .15)
        via = (gx + ms.shape[1] / 2, gy + ms.shape[0] / 2)
        self.morphs.append((Morph((src, xa, 0, 1), (ms, gx, gy, 1), n, t, dur, seed=seed, stagger=.18 if burst else .3,
                                  swirl=42 if not burst else 60, via=via if burst else None, burst=burst), t + dur - .5, .55))
        self.layers[key] = Layer(m, gx, gy, 3.2, t + dur - .5, .55)
        self.big_at = (gx, gy)
        if burst:
            self.splash_at = (t + dur * .52, via)

    def _splash(self, c):
        """Ink flung from the burst, landing round the character."""
        r = self.rng
        for _ in range(170):
            a = r.uniform(0, 6.28)
            d = r.uniform(160, 640) ** 1.0
            x = int(c[0] + d * math.cos(a))
            y = int(c[1] + d * .78 * math.sin(a))
            if 8 < y < SH - 8:
                self._blot(x, y, r.uniform(.8, 4.2) * (1.4 - d / 640), r.uniform(1.0, 2.6))

    # ---- the paper simulation ---------------------------------------------------------------------------------------
    def _curl(self, t, a, b, scale):
        s = int(t * 18)
        ny = (np.arange(SH) // scale + s) % 1024
        nx = (np.arange(a, b) // scale - s // 2) % 1024
        psi = NOISE_A[np.ix_(ny, nx)]
        gy, gx = np.gradient(gaussian_filter(psi, 1.5))
        k = 1 / (np.hypot(gx, gy).mean() * 1.6 + 1e-6)
        return gy * k, -gx * k, NOISE_B[np.ix_(ny, nx)]

    def _sim(self, t, dt):
        c = cam(t)
        a, b = max(int(c - SW / 2 - 120), 0), min(int(c + SW / 2 + 120), WW)
        Wt = self.Wt[:, a:b]
        if Wt.max() < .02:
            return
        P, F, h = self.P[:, a:b], self.F[:, a:b], self.h[:, a:b]
        wb = gaussian_filter(Wt, 2.2)
        gy, gx = np.gradient(wb)
        vx, vy, _ = self._curl(t, a, b, 2)
        wet = np.clip(Wt * 2.2, 0, 1)
        vx = (-46 * gx * (.5 + h * 1.2) + vx * 7) * wet
        vy = (-46 * gy * (.5 + h * 1.2) + vy * 7) * wet
        P[:] = advect(P, vx, vy, dt)
        Wt[:] = advect(Wt, vx, vy, dt)
        # capillary spread: faster along the fibres, so the edge feathers
        k = np.clip(wet * (.15 + h * h * 1.1), 0, 1)
        P += k * (gaussian_filter(P, 1.15) - P) * .85
        Wt += k * (gaussian_filter(Wt, 1.4) - Wt) * .9
        # pigment settles, more at the rim where the water stops
        rim = np.clip(np.hypot(gx, gy) * 7, 0, 1)
        rate = np.minimum((.42 + 3.2 * rim + (1 - wet) * 4.0) * dt, 1)
        d = P * rate
        F += d
        P -= d
        Wt *= math.exp(-dt / 1.1)
        Wt[Wt < .004] = 0

    # ---- one frame --------------------------------------------------------------------------------------------------
    def step(self, t, dt):
        r = self.r
        self.t = t
        while self.track_i < len(self.track_events) and self.track_events[self.track_i][0] <= min(t, 5.6):
            self._stamp_track(self.track_i)
            self.track_i += 1
        once = [("glyphs", 5.7, self._first_glyphs),
                ("ghost", r[3]["a"] + .15, lambda: self._ghost(r[3]["a"] + .15)),
                ("smoke1", r[5]["a"] + .5, lambda: self.smokes.append(Smoke("欺", self.ghost_at[0] + 45, 50, r[5]["a"] + .5, 4.8))),
                ("smoke2", r[6]["a"] + .5, lambda: self.smokes.append(Smoke("秘", self.ghost_at[0] + 45, 50, r[6]["a"] + .5, 4.8))),
                ("wash", r[7]["a"], lambda: self._start_wash(r[7]["a"])),
                ("mass", r[9]["a"] - .2, lambda: self._mass_in(r[9]["a"] - .2, r[10]["a"])),
                ("nums", r[10]["a"], lambda: self._to_numbers(r[10]["a"], 5.4)),
                ("hello", r[13]["a"] + .2, lambda: self._hello(r[13]["a"] + .2)),
                ("jing", r[14]["a"], lambda: self._become(r[14]["a"], 3.4, "精", "jing", burst=True, src_keys=("num", "ni", "hao"))),
                ("wo", r[15]["a"] + .1, lambda: self._become(r[15]["a"] + .1, 2.3, "我", "wo", seed=6, src_keys=("jing",)))]
        for k, tt, fn in once:
            if k not in self.done and t >= tt:
                self.done.add(k)
                fn()
        if hasattr(self, "splash_at") and t >= self.splash_at[0]:
            self._splash(self.splash_at[1])
            del self.splash_at
        self.grains.step(t, dt, self)
        self._drips(t, dt)
        if self.wash:
            self._wash(t, dt)
        else:
            self._sim(t, dt)
        self.morphs = [m for m in self.morphs if t <= m[0].t0 + m[0].dur + .5]

    def frame(self, t, detail, tone):
        """The plate at time t as an HxWx3 uint8 image."""
        c = cam(t)
        x0 = c - SW / 2
        ia = int(math.floor(x0)) - 2
        wid = SW + 6
        fx = x0 - ia

        def up(a):
            return np.asarray(Image.fromarray(a.astype(np.float32)).transform((W, H), Image.AFFINE, (1 / SC, 0, fx, 0, 1 / SC, 0), Image.BICUBIC), np.float32)

        D = self.F[:, ia:ia + wid] + self.P[:, ia:ia + wid]
        black, light = None, None
        for m, t_x, fade in self.morphs:
            k = 1 - float(sm((t - t_x) / fade))
            if t >= m.t0 and k > 0:
                x, y, w = m.at(t)
                g = splat(x, y, w, ia, wid) * k
                if m.white:
                    light = g if light is None else light + g
                else:
                    black = g if black is None else black + g
        if black is not None:
            D = D + gaussian_filter(black, .55)
        hi = up(D).copy()
        if self.sky is not None:
            hi += np.asarray(Image.fromarray(self.sky.field(t, x0, SW, Sky.GW, Sky.GH)).resize((W, H), Image.BICUBIC), np.float32)
            self.sky.moon(hi, t, x0)
            L = self.white[:, ia:ia + wid]
            if t < 8.3:
                L = L + self.tmask[:, ia:ia + wid] * (1 - float(sm((t - 5.7) / 2.5)))
            if L.max() > 0:
                hi *= 1 - np.clip(up(L), 0, 1) * .97
        if light is not None:
            hi *= 1 - np.clip(up(gaussian_filter(light, .5)), 0, 1) * .98
        for l in self.layers.values():
            f = l.field(t)
            if f is None:
                continue
            ix, iy = int(round((l.wx - x0) * SC)), int(round(l.wy * SC))
            xa, xb = max(ix, 0), min(ix + f.shape[1], W)
            ya, yb = max(iy, 0), min(iy + f.shape[0], H)
            if xa >= xb or ya >= yb:
                continue
            part = f[ya - iy:yb - iy, xa - ix:xb - ix]
            if l.neg:
                if l.glow is not None:
                    k = (.25 + .55 * math.exp(-max(t - l.t_in, 0) / .6)) * float(part.max())
                    hi[ya:yb, xa:xb] *= 1 - l.glow[ya - iy:yb - iy, xa - ix:xb - ix] * k
                hi[ya:yb, xa:xb] *= 1 - np.clip(part, 0, 1) * .985
            else:
                hi[ya:yb, xa:xb] += part
        self.grains.draw(hi, t)
        for s in self.smokes:
            s.apply(hi, t, x0)
        sp = cam_speed(t) * SC / FPS
        if sp > 3:
            hi = uniform_filter1d(hi, int(sp * .85) | 1, axis=1)
        return develop(hi, int(round(x0 * SC)), detail, tone)


def lerp2(tile, x, y):
    """A wrapping bilinear lookup in a noise tile."""
    h, w = tile.shape
    ix, iy = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    fx, fy = x - ix, y - iy
    x0, x1, y0, y1 = ix % w, (ix + 1) % w, iy % h, (iy + 1) % h
    return (tile[y0, x0] * (1 - fx) + tile[y0, x1] * fx) * (1 - fy) + (tile[y1, x0] * (1 - fx) + tile[y1, x1] * fx) * fy


class Sky:
    """The sky of the myth, from the first frame to the flood: heavy ink clouds, thin at the horizon. Light is paper showing
    through them: a flash at each character that strikes, then the tear with its rays; when it closes only the moon is left."""
    GW, GH = 640, 360

    def __init__(self):
        self.a, self.b, self.c = noise2(1024, 2048, 3.0, 51), noise2(1024, 2048, 3.4, 52), noise2(1024, 2048, 2.8, 53)
        r = np.random.default_rng(54)
        jag = gaussian_filter(r.normal(size=720), 14, mode="wrap")
        self.jag = (jag / jag.std()).astype(np.float32)
        ray = gaussian_filter(r.normal(size=2048), 16, mode="wrap")
        self.ray = (ray / ray.std()).astype(np.float32)
        self.ray2 = np.roll(self.ray[::-1], 611)

    @staticmethod
    def opening(t):
        return float(sm((t - RIFT_OPEN) / .9)) * (1 - float(sm((t - RIFT_CLOSE) / 1.1)))

    def field(self, t, x0, w_sim, gw, gh):
        """Density over the span x0..x0+w_sim of the scroll, on a gh x gw grid."""
        X, Y = np.meshgrid(x0 + (np.arange(gw, dtype=np.float32) + .5) / gw * w_sim, (np.arange(gh, dtype=np.float32) + .5) / gh * SH)
        o = self.opening(t)
        drift = t * 3.2 + 9 * float(sm((t - RIFT_OPEN) / 4.0))
        u, v = X / 2.6, Y / 2.6
        wx = lerp2(self.a, u * .5 + drift * .3, v * .5)
        wy = lerp2(self.b, u * .5 - drift * .2, v * .5 + 300)
        c1 = lerp2(self.b, u + wx * 34 + drift, v + wy * 34)
        c2 = lerp2(self.c, u * 1.6 + wx * 18 - drift * 1.4, v * 1.6 + wy * 18 + 700)
        yy = Y / SH
        D = (.78 + 1.6 * (1 - sm(yy / .62))) * np.maximum(1 + .26 * c1 + .07 * c2, .15)
        f = sum(math.exp(-(t - h) / .22) for h in GLYPH_HITS if h <= t) if t < RIFT_OPEN + 2 else 0.0
        if f > .01:     # each character that strikes lights the clouds from inside
            D = D * (1 - np.clip(.36 * f * np.clip(.7 + .4 * c1, 0, 1.2), 0, .8))
        if o > 0:
            jag = np.interp(yy[:, 0] * 720, np.arange(720), self.jag)[:, None] * 34 * np.minimum(o * 1.4, 1)
            prof = np.clip(1 - yy / .70, 0, 1) ** 1.25
            width = 72 * o * prof
            g = 1 - np.abs(X - RIFT_X - jag) / (width + 1e-3)
            core = sm((g + .16 * c2) / .55) * (width > .5)
            th = np.arctan2(X - RIFT_X, Y + 40)
            R = np.hypot(X - RIFT_X, Y + 40)
            k = (th / 1.6 + 1) * 1024
            i1 = np.interp(k.ravel() + t * 14, np.arange(4096), np.tile(self.ray, 2), period=2048).reshape(k.shape)
            i2 = np.interp(k.ravel() * 2.7 - t * 22, np.arange(4096), np.tile(self.ray2, 2), period=2048).reshape(k.shape)
            rays = np.clip(.45 + .42 * i1 + .16 * i2, 0, 1.4) ** 1.3 * np.exp(-R / (430 + 200 * o)) * o * .72
            D = D * (1 - np.clip(core + rays * (.75 + .22 * c2), 0, 1))
        return D.astype(np.float32)

    def moon(self, hi, t, x0, scale=SC):
        """What is left of the tear. Works on the frame, or at the simulation's scale when the flood bakes it into the paper."""
        mo = float(sm((t - RIFT_CLOSE - .4) / 1.0))
        if mo <= 0:
            return
        cx, cy, R = (RIFT_X - x0) * scale, 122 * scale, 86 * scale
        h, w = hi.shape
        rad = R * 4
        xa, xb, ya, yb = max(int(cx - rad), 0), min(int(cx + rad), w), 0, min(int(cy + rad), h)
        if xa >= xb:
            return
        yy, xx = np.ogrid[ya:yb, xa:xb]
        d = np.hypot(xx - cx, yy - cy)
        halo = 1 - .45 * mo * np.exp(-np.maximum(d - R, 0) / (73 * scale))
        hi[ya:yb, xa:xb] *= halo * (1 - mo * sm((R - d) / (1.6 * scale)) * .985)


class Grains:
    """粟 falling through real depth: the far grains drift down like dust, the near ones rush past the lens."""
    FOCAL, HORIZON = 1000.0, 700.0

    def __init__(self):
        self.rng = np.random.default_rng(71)
        self.a = np.zeros((0, 6), np.float32)    # X, Y, Z, speed, size factor, sprite seed

    def step(self, t, dt, film):
        r = self.rng
        rate = 7000 * float(sm((t - RIFT_OPEN - .3) / 2.0)) * (1 - float(sm((t - RIFT_CLOSE) / .8)))
        n = r.poisson(rate * dt) if rate > 0 else 0
        if n:
            z = 1.15 + 38.85 * r.random(n) ** 1.7
            xr = (RIFT_X - (cam(t) - SW / 2)) * SC
            sx = xr + r.normal(0, 120 + 560 * max(t - RIFT_OPEN, 0), n)     # from the tear, spreading over the whole sky
            new = np.empty((n, 6), np.float32)
            new[:, 0] = (sx - W / 2) * z / self.FOCAL
            new[:, 1] = -self.HORIZON / self.FOCAL * z - r.uniform(0, 1.0, n)
            new[:, 2] = z
            new[:, 3] = r.uniform(1.7, 2.3, n)
            new[:, 4] = r.uniform(.8, 1.2, n)
            new[:, 5] = r.integers(0, 6, n)
            self.a = np.vstack([self.a, new])
        if not len(self.a):
            return
        a = self.a
        a[:, 1] += a[:, 3] * dt
        land = a[:, 1] >= 1.0
        if land.any() and film.white is not None:   # what lands near enough stays, on the ground
            z = a[land, 2]
            sx = W / 2 + self.FOCAL * a[land, 0] / z
            sy = self.HORIZON + self.FOCAL / z
            s = 46 * a[land, 4] / z
            ok = (sy > 885) & (sy < H) & (sx >= 0) & (sx < W)
            x0 = cam(t) - SW / 2
            np.add.at(film.white, ((sy[ok] / SC).astype(np.int32), (x0 + sx[ok] / SC).astype(np.int32)), .15 + s[ok] * .02)
        self.a = a[~land] if t < RIFT_CLOSE + 3.5 else a[:0]

    def draw(self, hi, t):
        a = self.a
        if not len(a):
            return
        k = 1 - float(sm((t - RIFT_CLOSE - 1) / 2.5))
        z = a[:, 2]
        sx = W / 2 + self.FOCAL * a[:, 0] / z
        sy = self.HORIZON + self.FOCAL * a[:, 1] / z
        s = 46 * a[:, 4] / z
        on = (sx > -40) & (sx < W + 40) & (sy > -40) & (sy < H + 40)
        tiny = on & (s < 3.5)
        if tiny.any():
            al = np.minimum((s[tiny] / 2.4) ** 2, 1) * (.45 + .55 * np.exp(-z[tiny] / 18)) * k
            hi *= 1 - np.clip(splat(sx[tiny], sy[tiny], al, 0, W, H), 0, 1) * .95
        mid = on & (s >= 3.5) & (s < 9)
        if mid.any():
            g = gaussian_filter(splat(sx[mid], sy[mid], s[mid] ** 2 * .55 * k, 0, W, H), 1.1)
            hi *= 1 - np.clip(g, 0, 1) * .96
        big = np.nonzero(on & (s >= 9))[0]
        for i in big[np.argsort(-z[big])]:
            spr = grain_sprite(int(round(s[i])), int(a[i, 5]), int(self.FOCAL * a[i, 3] / z[i] / FPS * .3))
            h, w = spr.shape
            ix, iy = int(sx[i] - w / 2), int(sy[i] - h / 2)
            xa, xb, ya, yb = max(ix, 0), min(ix + w, W), max(iy, 0), min(iy + h, H)
            if xa < xb and ya < yb:
                hi[ya:yb, xa:xb] *= 1 - spr[ya - iy:yb - iy, xa - ix:xb - ix] * .97 * k


_sprites = {}


def grain_sprite(s, sd, streak):
    """A grain of millet in bare paper: a small oval with a groove, stretched by its own speed, soft when it is at the lens."""
    k = (s, sd % 6, streak)
    if k not in _sprites:
        r = np.random.default_rng(s * 7 + sd)
        n = int((s * .62 + streak + 4) * 2) | 1
        yy, xx = np.mgrid[:n, :n].astype(np.float32) - n / 2
        a = r.uniform(-.6, .6)
        u = xx * math.cos(a) + yy * math.sin(a)
        v = -xx * math.sin(a) + yy * math.cos(a)
        m = 1 - sm((np.hypot(u / (s * .38), v / (s * .54)) - .82) / .3)
        groove = 1 - .35 * np.exp(-(u / (s * .07 + .5)) ** 2) * (np.abs(v) < s * .42)
        shade = 1 - .22 * sm((u / (s * .38) + .1) / .9)
        spr = m * groove * shade
        if streak > 2:
            spr = uniform_filter1d(spr, streak | 1, axis=0)
        if s > 26:
            spr = gaussian_filter(spr, (s - 26) * .12 + .5)
        _sprites[k] = spr.astype(np.float32)
    return _sprites[k]


class Smoke:
    """A character in bare paper that rises and comes apart like smoke: it takes the ink away instead of adding it."""

    def __init__(self, ch, wx, wy, t0, dur):
        self.m = glyph_mask(ch, "kaih", 620, 80)
        self.wx, self.wy, self.t0, self.dur = wx, wy, t0, dur
        n = self.m.shape[0]
        self.nx = gaussian_filter(np.random.default_rng(ord(ch)).normal(size=(n, n)), 22).astype(np.float32)
        self.ny = gaussian_filter(np.random.default_rng(ord(ch) + 1).normal(size=(n, n)), 22).astype(np.float32)
        self.nx /= self.nx.std()
        self.ny /= self.ny.std()
        self.yy, self.xx = np.mgrid[:n, :n].astype(np.float32)

    def apply(self, hi, t, x0):
        u = (t - self.t0) / self.dur
        if u <= 0 or u >= 1:
            return
        a = float(sm(u / .16)) * (1 - float(sm((u - .55) / .45)))
        spread = max(u - .34, 0) ** 1.4 * 190
        y = self.yy + self.ny * spread + u * u * 140
        x = self.xx + self.nx * spread
        m = map_coordinates(self.m, [y, x], order=1, mode="constant")
        if spread > 2:
            m = gaussian_filter(m, min(spread * .045, 5))
        m *= a * .985
        ix, iy = int((self.wx - x0) * SC), int(self.wy * SC)
        n = m.shape[0]
        xa, xb, ya, yb = max(ix, 0), min(ix + n, W), max(iy, 0), min(iy + n, H)
        if xa < xb and ya < yb:
            hi[ya:yb, xa:xb] *= 1 - m[ya - iy:yb - iy, xa - ix:xb - ix]


def develop(hi, ox, detail, tone):
    """Ink density at frame size -> RGB: the ink soaks into the paper's grain. ox is how far along the scroll the frame starts."""
    ox %= 2048
    dt_ = np.take(detail[:H], np.arange(ox, ox + W) % 2048, axis=1)
    tn = np.take(tone[:H], np.arange(ox, ox + W) % 2048, axis=1)
    hi = np.maximum(hi, 0)
    dens = hi * (1 + .17 * dt_ * np.exp(-hi * .45))
    rgb = PAPER[None, None, :] * tn[..., None] * np.exp(-dens[..., None] * INK_K[None, None, :])
    rgb = np.maximum(rgb, INK_FLOOR[None, None, :])
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ======================================================================================
# 千窗: the chat window pulls back into one lit window among thousands on a night mountain
# ======================================================================================
QT0 = 172.0                     # where 千窗 starts on the film's clock
# what each line of the storyboard is, so sound and the sections need not know its number
ROW = dict(title=1, rain=2, ghost=3, why=4, lie=5, secret=6, turn=7, saw=8, all=9, press=10, old=11, age=12, speak=13, became=14, me=15,
           child=16, reading=17, kinds=18, place=19, cluster=20, direction=21, digits=22, wall=23,
           look=24, lian=25, guess=26, fix=27, drip=28, tang=29, never=30, bed=31, warm=32,
           win1=33, win2=34, win3=35, win4=36, win5=37, win6=38, win7=39)
# each section is its own plate, from this clock time until 0.8 s into the next, where it dissolves on the paper
SECTIONS = [("opening", 0.0), ("read", 90.2), ("attn", 127.2), ("bed", 155.9), ("qian", QT0)]
QC = (860.0, 560.0)             # the first window's place on the screen; it never moves
QZ0 = 70.0                      # how far in the pull-back starts (the window is 16 px wide at the end)
CARD = (1120, 680)
QUESTION, ANSWER = "我失业了，今晚睡不着。", "我在。想从哪里说起？"
Q_OFF_FIRST = 38.9              # the last window to go out is the one we started in
Q_OFF_SINGLES = [24.9, 25.7, 26.4, 27.0, 27.5, 27.9, 28.2, 28.4]


def chat_card(nq, na):
    """The chat window with nq characters of the question and na of the answer typed. Density, card-sized."""
    w, h = CARD
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([6, 6, w - 7, h - 7], 34, outline=255, width=7)
    d.line([(8, 88), (w - 9, 88)], fill=130, width=3)
    for i in range(3):
        d.ellipse([40 + i * 34, 36, 58 + i * 34, 54], outline=170, width=3)
    d.rounded_rectangle([60, h - 104, w - 60, h - 40], 22, outline=120, width=3)
    f = font("kai", 56)
    if nq:
        tw = f.getlength(QUESTION)
        d.rounded_rectangle([w - 70 - tw - 64, 140, w - 70, 244], 26, outline=210, width=4)
        d.text((w - 70 - tw - 32, 192), QUESTION[:nq], 255, font=f, anchor="lm")
    if na:
        tw = f.getlength(ANSWER)
        d.rounded_rectangle([70, 300, 70 + tw + 64, 404], 26, fill=26)
        d.text((102, 352), ANSWER[:na], 255, font=f, anchor="lm")
    return np.asarray(im, np.float32) / 255 * 3.0


def lattice():
    """The same rectangle as a window frame with a cross in it; the bars are thick so they survive being tiny."""
    w, h = CARD
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w - 1, h - 1], outline=255, width=64)
    d.rectangle([w // 2 - 24, 0, w // 2 + 24, h], fill=255)
    d.rectangle([0, h // 2 - 24, w, h // 2 + 24], fill=255)
    return np.asarray(im, np.float32) / 255 * 3.0


def night_mountains(seed=23):
    """The range at night, at twice the frame's size. Returns its density and how high the top ridge is along x."""
    w, h = W * 2, H * 2
    y = np.arange(h, dtype=np.float32)[:, None] / 2
    X = np.arange(w, dtype=np.float32) / 2
    n = noise2(h, w, 2.6, seed)
    sky = (2.35 - .85 * y / H) + .10 * n
    out = sky.copy()
    top = np.full(w, 1e9, np.float32)
    tex = gaussian_filter(noise2(h, w, 1.2, seed + 1), 2.0)
    tex /= tex.std()
    for i, (base, amp, dark, fall) in enumerate([(330, 120, 2.0, 170), (470, 130, 2.35, 180), (640, 130, 2.7, 170), (810, 110, 3.1, 160)]):
        r = np.random.default_rng(seed + 2 + i)
        ridge = np.zeros(w, np.float32)
        for k, (L, a) in enumerate([(1000, 1.0), (430, .50), (170, .18), (70, .06)]):
            ph = r.uniform(0, 6.28)
            xs = X / L * 6.28 + ph
            ridge += a * (1 - 2 * np.abs(np.sin(xs / 2 + .6 * np.sin(xs * .37 + ph)))) if k < 3 else a * np.sin(xs)
        ridge = base + amp * .55 * ridge
        top = np.minimum(top, ridge)
        d = y - ridge[None, :]
        fade = np.exp(-np.maximum(d, 0) / fall)
        layer = dark * fade + 1.55 * (1 - fade) + .10 * tex
        mask = sm(d / 1.5)
        out = out * (1 - mask) + layer * mask
    return out.astype(np.float32), top


def window_field(top, n=14000, seed=29):
    """Where the windows are (clusters on the slopes) and when each comes on and goes out, on the section's own clock."""
    r = np.random.default_rng(seed)
    gw, gh = 480, 270
    clump = noise2(gh, gw, 2.6, seed)
    yy = (np.arange(gh) * 4 + 2)[:, None]
    p = (np.clip(clump + .5, 0, None) ** 2.2 + .015) * (yy > top[::8][:gw][None, :] + 14) * (.35 + yy / H)
    y, x = np.unravel_index(r.choice(p.size, n, p=p.ravel() / p.sum()), p.shape)
    x, y = x * 4 + r.uniform(0, 4, n), y * 4 + r.uniform(0, 4, n)
    d = np.hypot(x - QC[0], y - QC[1])
    keep = d > 46
    x, y, d = x[keep], y[keep], d[keep]
    n = len(x)
    s = (2.4 + 5.6 * np.clip((y - 300) / 780, 0, 1) ** 1.2) * r.uniform(.8, 1.25, n)
    on = 11.2 + 8.3 * (d / d.max()) ** .75 + r.uniform(0, 1.2, n)
    order = r.permutation(n)
    near = np.argsort(d + (9 - s) * 30)[:len(Q_OFF_SINGLES)]   # big and close, so they can be seen going out one by one
    rest = order[~np.isin(order, near)]
    off = np.zeros(n)
    off[near] = Q_OFF_SINGLES
    off[rest] = off_times(len(rest))
    return dict(x=x, y=y, s=s, h=s * .72, on=on, off=off, b=r.uniform(.72, 1, n))


def off_times(m):
    """Fewer and fewer: the number still lit falls by the same share each second, so the last ones go singly."""
    lam = math.log(m + 1) / (38.3 - 28.4)
    return 28.4 - np.log(1 - (np.arange(m) + 1) / (m + 1)) / lam


class Qian:
    def __init__(self):
        night, top = night_mountains()
        self.img = Image.fromarray(night)
        self.win = window_field(top)
        self.alpha = Image.fromarray(self._card_alpha())
        self.cards = {}
        self.lat = Image.fromarray(lattice())
        yy, xx = np.ogrid[:H, :W]
        self.xx, self.yy = xx.astype(np.float32), yy.astype(np.float32)
        self.dn = np.clip(np.hypot(np.maximum(np.abs(self.xx - QC[0]) - 560, 0) / 700, np.maximum(np.abs(self.yy - QC[1]) - 340, 0) / 400), 0, 1)
        nn = zoom(noise2(270, 480, 2.6, 37), 4, order=1)[:H, :W]
        self.nn = np.clip(.5 + .3 * nn, 0, 1).astype(np.float32)

    def _card_alpha(self):
        w, h = CARD
        im = Image.new("L", (w * 2, h * 2), 0)
        ImageDraw.Draw(im).rounded_rectangle([8, 8, w * 2 - 9, h * 2 - 9], 68, fill=255)
        return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255

    def card(self, nq, na):
        k = (nq, na)
        if k not in self.cards:
            self.cards[k] = Image.fromarray(chat_card(nq, na))
        return self.cards[k]

    @staticmethod
    def zoom(t):
        if t < 6:
            return QZ0 * (1.05 - .05 * t / 6)
        if t < 16.5:     # an even pace, about 1.6x a second, easing in and out
            x, r = (t - 6) / 10.5, .12
            e = x * x / (2 * r) / (1 - r) if x < r else 1 - (1 - x) ** 2 / (2 * r) / (1 - r) if x > 1 - r else (x - r / 2) / (1 - r)
            return math.exp(math.log(QZ0) * (1 - e) + math.log(1.06) * e)
        return 1.06 - .06 * min((t - 16.5) / 23, 1)

    def windows(self, t, z):
        """The lit windows as a mask at frame size, or None before any comes on."""
        w = self.win
        a = np.minimum(np.clip((t - w["on"]) / .35, 0, 1), 1 - np.clip((t - w["off"]) / .25, 0, 1)) * w["b"]
        X, Y = QC[0] + (w["x"] - QC[0]) * z, QC[1] + (w["y"] - QC[1]) * z
        hw, hh = w["s"] * z / 2, w["h"] * z / 2
        vis = (a > .01) & (X + hw > 0) & (X - hw < W) & (Y + hh > 0) & (Y - hh < H)
        if not vis.any():
            return None
        im = Image.new("L", (W * 2, H * 2), 0)
        d = ImageDraw.Draw(im)
        for x0, y0, x1, y1, f in zip(((X - hw)[vis] * 2).astype(int).tolist(), ((Y - hh)[vis] * 2).astype(int).tolist(),
                                     ((X + hw)[vis] * 2).astype(int).tolist(), ((Y + hh)[vis] * 2).astype(int).tolist(), (a[vis] * 255).astype(int).tolist()):
            d.rectangle((x0, y0, x1, y1), fill=f)
            if x1 - x0 > 24:   # close enough to see the cross in the window
                bar = max((x1 - x0) // 12, 1)
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                d.rectangle((cx - bar, y0, cx + bar, y1), fill=0)
                d.rectangle((x0, cy - bar, x1, cy + bar), fill=0)
        return np.asarray(im.reduce(2), np.float32) / 255

    def frame(self, t, detail, tone):
        """The plate at t seconds into the section."""
        z = self.zoom(t)
        sc = z / QZ0
        a = 2 / z
        night = np.asarray(self.img.transform((W, H), Image.AFFINE, (a, 0, 2 * QC[0] - a * QC[0], 0, a, 2 * QC[1] - a * QC[1]), Image.BICUBIC), np.float32)
        u = (t - 4.4) / 2.2
        if u <= 0:
            hi = np.zeros((H, W), np.float32)
        elif u >= 1:
            hi = night.copy()
        else:
            hi = night * sm((u * 1.5 - (1 - self.dn) - .2 * self.nn) / .25)
        # the light of the windows on the slopes around them
        lit = self.windows(t, z)
        if lit is not None:
            g = gaussian_filter(lit[::4, ::4], 3.0)
            g = np.asarray(Image.fromarray(g).resize((W, H), Image.BILINEAR), np.float32)
            hi *= 1 - np.clip(lit * .97 + g * .9, 0, .985)
        off = 1 - float(sm((t - Q_OFF_FIRST) / .3))
        hw, hh = 560 * sc, 340 * sc
        if off > 0:
            if u > 0:   # and the light of this one
                dist = np.maximum(np.maximum(np.abs(self.xx - QC[0]) - hw, np.abs(self.yy - QC[1]) - hh), 0)
                hi *= 1 - .45 * off * min(u, 1) * np.exp(-dist / (30 + 260 * sc))
            x0, y0 = QC[0] - hw, QC[1] - hh
            ia, ja, ib, jb = max(int(x0), 0), max(int(y0), 0), min(int(QC[0] + hw) + 2, W), min(int(QC[1] + hh) + 2, H)
            box = ((ia - x0) / sc, (ja - y0) / sc, (ib - x0) / sc, (jb - y0) / sc)
            box = (max(box[0], 0), max(box[1], 0), min(box[2], CARD[0]), min(box[3], CARD[1]))
            nq = int(np.clip((t - .5) / 1.7, 0, 1) * len(QUESTION))
            na = int(np.clip((t - 2.6) / 1.6, 0, 1) * len(ANSWER))
            size = (ib - ia, jb - ja)
            A = np.asarray(self.alpha.resize(size, Image.LANCZOS, box=box), np.float32) * off
            D = np.asarray(self.card(nq, na).resize(size, Image.LANCZOS, box=box), np.float32)
            la = float(sm((.25 - sc) / .12))
            if la > 0:
                D = D * (1 - la) + np.asarray(self.lat.resize(size, Image.LANCZOS, box=box), np.float32) * la
            hi[ja:jb, ia:ib] = hi[ja:jb, ia:ib] * (1 - np.clip(A, 0, 1)) + np.maximum(D, 0) * np.clip(A, 0, 1)
        return develop(hi, 0, detail, tone)


def cmd_qian_still(times):
    os.makedirs(os.path.join(OUT, "stills"), exist_ok=True)
    q = Qian()
    detail, tone = make_paper()
    for x in times:
        Image.fromarray(q.frame(x - QT0, detail, tone)).save(os.path.join(OUT, "stills", f"s{x:06.2f}.png"))
    sheet(sorted(times))


def cmd_qian_plate(b, name="qian"):
    os.makedirs(OUT, exist_ok=True)
    q = Qian()
    detail, tone = make_paper()
    path = os.path.join(OUT, f"plate_{name}.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "medium", "-crf", "11", "-pix_fmt", "yuv444p", path], stdin=subprocess.PIPE)
    for n in range(int(round((b - QT0) * FPS))):
        p.stdin.write(q.frame(n / FPS, detail, tone).tobytes())
        if n % 150 == 0:
            print(f"t={QT0 + n / FPS:6.2f}", flush=True)
    p.stdin.close()
    p.wait()
    print(path)


def cmd_join(name):
    """The sections' plates, each dissolving into the next on the paper."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for n, _ in SECTIONS:
        cmd += ["-i", os.path.join(OUT, f"plate_{n}.mp4")]
    chain, prev = [], "0:v"
    for i, (_, t) in enumerate(SECTIONS[1:], 1):
        chain.append(f"[{prev}][{i}:v]xfade=transition=fade:duration=0.8:offset={t}[v{i}]")
        prev = f"v{i}"
    subprocess.check_call(cmd + ["-filter_complex", ";".join(chain), "-map", f"[{prev}]", "-c:v", "libx264", "-preset", "medium", "-crf", "11",
                                 "-pix_fmt", "yuv444p", os.path.join(OUT, f"plate_{name}.mp4")])


def advect(f, vx, vy, dt):
    """Semi-Lagrangian step with bilinear lookups."""
    h, w = f.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    x = np.clip(xx - vx * dt, 0, w - 1.001)
    y = np.clip(yy - vy * dt, 0, h - 1.001)
    ix, iy = x.astype(np.int32), y.astype(np.int32)
    fx, fy = x - ix, y - iy
    return ((f[iy, ix] * (1 - fx) + f[iy, ix + 1] * fx) * (1 - fy)
            + (f[iy + 1, ix] * (1 - fx) + f[iy + 1, ix + 1] * fx) * fy).astype(np.float32)


# ======================================================================================
# passes
# ======================================================================================
def ckpt_path(t):
    return os.path.join(OUT, "ckpt", f"t{int(t):04d}.pkl")


def load_film(t_from):
    if t_from > 0:
        p = ckpt_path(t_from)
        if not os.path.exists(p):
            raise SystemExit(f"no checkpoint at {t_from}")
        with open(p, "rb") as f:
            film = pickle.load(f)
        film.r = rows()
        return film
    return Film()


def run(a, b, every=10, on_frame=None, t_from=0.0):
    """Simulate from t_from to b; call on_frame for every frame in [a, b)."""
    film = load_film(t_from)
    detail, tone = make_paper()
    os.makedirs(os.path.dirname(ckpt_path(0)), exist_ok=True)
    dt = 1 / FPS
    n0 = int(round(t_from * FPS))
    n1 = int(round(b * FPS))
    for n in range(n0, n1):
        t = n / FPS
        if n % (every * FPS) == 0 and n > n0:
            with open(ckpt_path(t), "wb") as f:
                pickle.dump(film, f, protocol=4)
        film.step(t, dt)
        if on_frame and t >= a - 1e-6:
            on_frame(t, film, detail, tone)
        if n % 150 == 0:
            print(f"t={t:6.2f}", flush=True)


def cmd_still(times, t_from):
    os.makedirs(os.path.join(OUT, "stills"), exist_ok=True)
    want = {int(round(x * FPS)): x for x in times}
    def fn(t, film, detail, tone):
        n = int(round(t * FPS))
        if n in want:
            Image.fromarray(film.frame(t, detail, tone)).save(os.path.join(OUT, "stills", f"s{want[n]:06.2f}.png"))
    run(t_from, max(times) + .04, on_frame=fn, t_from=t_from)
    sheet(sorted(times))


def sheet(times, cols=3, tw=640):
    ims = [Image.open(os.path.join(OUT, "stills", f"s{x:06.2f}.png")).resize((tw, tw * 9 // 16), Image.LANCZOS) for x in times]
    rows_ = math.ceil(len(ims) / cols)
    s = Image.new("RGB", (cols * tw, rows_ * tw * 9 // 16), (30, 30, 30))
    for i, im in enumerate(ims):
        s.paste(im, ((i % cols) * tw, (i // cols) * tw * 9 // 16))
    s.save(os.path.join(OUT, "stills", "sheet.png"))


def cmd_plate(a, b, name, t_from):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"plate_{name}.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "medium", "-crf", "11", "-pix_fmt", "yuv444p", path], stdin=subprocess.PIPE)
    run(a, b, on_frame=lambda t, film, d, tn: p.stdin.write(film.frame(t, d, tn).tobytes()), t_from=t_from)
    p.stdin.close()
    p.wait()
    print(path)


# ---- subtitles: vertical columns, written in ink ---------------------------------------------------------------------
# for each row: font, size, top-right corner on screen (x of the first column, y), ink or bare-paper colour, and whether it
# stays put on the screen or travels with the scroll
STYLE = {1: dict(kind="xing", size=118, x=1560, y=60, col="auto", instant=True),
         2: dict(kind="xing", size=118, x=1600, y=60, col="auto"),
         3: dict(kind="xing", size=118, x=1600, y=60, col="auto"),
         4: dict(kind="kai", size=62, x=1700, y=96, col="auto"),
         5: dict(kind="kai", size=62, x=1700, y=96, col="auto"),
         6: dict(kind="kai", size=62, x=1700, y=96, col="auto"),
         7: dict(kind="xing", size=118, x=1540, y=170, col="ink", fixed=True, delay=1.0),
         8: dict(kind="kai", size=66, x=1760, y=170, col="ink", fixed=True, back=True),
         9: dict(kind="kai", size=66, x=1780, y=140, col="ink", fixed=True, back=True),
         10: dict(kind="kai", size=66, x=1780, y=140, col="ink", fixed=True, back=True),
         11: dict(kind="kai", size=62, x=1820, y=130, col="ink", fixed=True, back=True),
         12: dict(kind="kai", size=62, x=1820, y=130, col="ink", fixed=True, back=True),
         13: dict(kind="kai", size=62, x=1820, y=130, col="ink", fixed=True, back=True),
         14: dict(kind="kai", size=62, x=1820, y=130, col="ink", fixed=True),
         15: dict(kind="kai", size=62, x=1820, y=130, col="ink", fixed=True),
         **{n: dict(kind="kai", size=62, x=1800, y=120, col="auto", fixed=True) for n in range(16, 40)}}


def columns(s):
    """A line as columns: punctuation becomes a break, except the marks that carry meaning."""
    s = re.sub(r"[{}!]", "", s)
    parts = [p for p in re.split(r"[，。：；、]|\s+", s) if p]
    return parts


class Line:
    def __init__(self, r):
        st = dict(STYLE.get(r["n"], dict(kind="kai", size=60, x=1780, y=120, col="ink")))
        self.__dict__.update(st)
        self.a, self.b, self.n = r["a"], r["b"] - .55, r["n"]
        self.glyphs = []
        self.tones = {}
        gap = int(self.size * .34)
        x = self.x
        d_t = .055 + 0.016 * (60 / self.size)
        tt = self.a + .1 + st.get("delay", 0) - (1e3 if st.get("instant") else 0)
        for part in columns(r["sub"]):
            y = self.y
            for ch in part:
                if ch == "？":
                    ch = "？"
                m = glyph_mask(ch, self.kind, self.size, int(self.size * .12))
                soft = gaussian_filter(m, max(self.size / 60, 1.0))
                self.glyphs.append((x, y, m, soft, tt))
                y += int(self.size * 1.03)
                tt += d_t
            x -= self.size + gap
        self.x_end = x
        self.note = r["note"]
        self.note_mask = None
        if self.note:
            f = font("kai", 27)
            w = int(f.getlength(self.note)) + 10
            im = Image.new("L", (w, 42), 0)
            ImageDraw.Draw(im).text((4, 3), self.note, 255, font=f)
            self.note_mask = np.asarray(im, np.float32) / 255

    def tone(self, key, region):
        """How dark the text should be over this part of the page (1 = ink, 0 = bare paper), eased so it never flickers."""
        target = float(sm((region.mean() - 80) / 55)) if self.col == "auto" else 1.0 if self.col == "ink" else 0.0
        v = self.tones.get(key)
        v = target if v is None else v + (target - v) * .12
        self.tones[key] = v
        return v

    def apply(self, img, t, shift):
        if t < self.a or t > self.b + .55:
            return
        out = 1 - float(sm((t - self.b) / .55))
        INK, BARE = np.array([24, 21, 19], np.float32), np.array([240, 233, 219], np.float32)
        dx = 0 if getattr(self, "fixed", False) else shift
        if getattr(self, "back", False) and self.glyphs:
            # a little paper under the words, where the page is busy
            xs = [g[0] for g in self.glyphs]
            ys = [g[1] for g in self.glyphs]
            s = self.size
            xa, xb = int(min(xs) - s * .5), int(max(xs) + s * 1.5)
            ya, yb = int(min(ys) - s * .5), int(max(ys) + s * 1.7)
            xa, xb, ya, yb = max(xa, 0), min(xb, W), max(ya, 0), min(yb, H)
            yy, xx = np.ogrid[ya:yb, xa:xb]
            al = (sm((xx - xa) / 60) * sm((xb - xx) / 60) * sm((yy - ya) / 60) * sm((yb - yy) / 60) * .9 * out)[..., None]
            img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - al) + np.array([241, 234, 219], np.float32) * al
        for x, y, m, soft, tt in self.glyphs:
            u = (t - tt) / .55
            if u <= 0:
                continue
            X = int(x + dx)
            h, w = m.shape
            xa, xb, ya, yb = max(X, 0), min(X + w, W), max(y, 0), min(y + h, H)
            if xa >= xb or ya >= yb:
                continue
            if u >= 1:
                al = m
            else:
                thr = 1.0 - .92 * float(ease_out(u))
                al = m * sm((soft - thr) / .12)
            al = (al * out)[ya - y:yb - y, xa - X:xb - X, None] * .97
            c = self.tone((x, y), img[ya:yb, xa:xb])
            img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - al) + (INK * c + BARE * (1 - c)) * al
        if self.note_mask is not None:
            k = float(sm((t - self.a - .5) / .5)) * out * .85
            h, w = self.note_mask.shape
            X, Y = 1880 - w, 1010
            al = (self.note_mask * k)[..., None]
            c = self.tone("note", img[Y:Y + h, X:X + w])
            img[Y:Y + h, X:X + w] = img[Y:Y + h, X:X + w] * (1 - al) + (INK * c + BARE * (1 - c)) * al


def cmd_render(name, plate, audio):
    src = os.path.join(OUT, f"plate_{plate or name}.mp4")
    dst = os.path.join(OUT, f"{name}.mp4")
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=nb_frames", "-of", "csv=p=0", src]).decode().strip()
    t0 = float(os.environ.get("INK_T0", "0"))
    lines = [Line(r) for _, r in sorted(rows().items())]
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if audio and os.path.exists(audio):
        cmd += ["-ss", str(t0), "-i", audio, "-c:a", "aac", "-b:a", "256k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    size = W * H * 3
    while True:
        b = dec.stdout.read(size)
        if len(b) < size:
            break
        t = t0 + n / FPS
        img = np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32)
        for l in lines:
            l.apply(img, t, (cam(l.a) - cam(t)) * SC)
        enc.stdin.write(np.clip(img, 0, 255).astype(np.uint8).tobytes())
        n += 1
    enc.stdin.close()
    enc.wait()
    print(dst, n, "frames of", probe)


# ---- sound: everything synthesised here, so nothing needs a licence -----------------------------------------------------
SR = 48000


def pluck(f, dur, seed=0, bright=.5):
    """A guqin-like note: a Karplus-Strong string with a soft attack."""
    r = np.random.default_rng(seed)
    n = int(SR / f)
    buf = r.uniform(-1, 1, n)
    buf = np.convolve(buf, np.ones(3) / 3, "same")
    out = np.zeros(int(dur * SR))
    damp = .996 + .0035 * bright
    for i in range(len(out)):
        out[i] = buf[i % n]
        buf[i % n] = damp * .5 * (buf[i % n] + buf[(i + 1) % n])
    a = np.minimum(np.arange(len(out)) / (SR * .004), 1)
    return out * a


def noise_sweep(dur, lo, hi, seed=0):
    """Band-limited noise whose band moves from lo to hi: wind, water."""
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    x = r.normal(size=n)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec *= np.exp(-((np.log(f + 1) - math.log((lo + hi) / 2)) / 1.0) ** 2)
    return np.fft.irfft(spec, n)


def thud(f0, dur, drop=.5):
    t = np.arange(int(dur * SR)) / SR
    return np.sin(2 * np.pi * (f0 * t + f0 * drop * (1 - np.exp(-t * 18)) / 18)) * np.exp(-t * 3.2 / dur)


def cmd_audio(dur):
    os.makedirs(OUT, exist_ok=True)
    r = rows()
    a = np.zeros(int((dur + 6) * SR))
    def add(x, t, g=1.0):
        i = int(t * SR)
        e = min(i + len(x), len(a))
        if i < len(a):
            a[i:e] += x[:e - i] * g
    # paper and air
    room = noise_sweep(dur + 6, 90, 900, 1)
    a += room / np.abs(room).max() * .012
    # a sparse qin line in D yu mode, slowing where the story turns
    scale = [146.83, 174.61, 196.0, 220.0, 261.63, 293.66, 349.23, 392.0, 440.0]
    phrase = [(0.2, 5), (2.3, 4), (3.5, 3), (5.7, 0), (7.6, 2), (8.8, 3), (10.3, 5), (11.6, 4), (14.1, 3), (15.3, 1), (17.4, 0),
              (25.1, 3), (27.3, 2), (30.6, 1), (33.0, 0), (36.1, 2), (38.3, 1), (40.2, 0)]
    for i, (t, k) in enumerate(phrase):
        add(pluck(scale[k], 5.0, i, .55), t, .17)
        if i % 3 == 0:
            add(pluck(scale[k] * 2, 3.0, i + 50, .3), t + .012, .035)
    # rain: ticks that thicken
    rng = np.random.default_rng(4)
    ta, tb = r[2]["a"], r[3]["a"] + 3
    tick = np.exp(-np.arange(int(.02 * SR)) / (SR * .003)) * rng.normal(size=int(.02 * SR))
    for t in np.arange(ta, tb, .012):
        if rng.random() < .05 + .6 * float(sm((t - ta) / 4)) * (1 - float(sm((t - tb + 2) / 2))):
            add(tick, t, rng.uniform(.006, .035))
    # the night: wind that moans; the crying
    wind = noise_sweep(24, 160, 620, 2)
    wind = wind / np.abs(wind).max() * np.interp(np.arange(len(wind)) / SR, [0, 4, 18, 24], [0, 1, 1, 0]) * (1 + .5 * np.sin(np.arange(len(wind)) / SR * 1.6))
    add(wind, r[3]["a"] - 2.0, .085)
    add(thud(52, 3.0), r[3]["a"] + .15, .55)
    # the flood
    t0 = r[7]["a"]
    add(thud(40, 3.6), t0, .9)
    fl = noise_sweep(6, 300, 2600, 3)
    fl = fl / np.abs(fl).max() * np.interp(np.arange(len(fl)) / SR, [0, .25, 2.4, 6], [0, 1, .7, 0])
    add(fl, t0 - .1, .34)
    # the rush down the scroll
    ru = noise_sweep(5.5, 500, 5000, 5)
    ru = ru / np.abs(ru).max() * np.sin(np.linspace(0, np.pi, len(ru))) ** 2
    add(ru, r[8]["a"], .12)
    # all writing runs together; the numbers start to breathe
    st = noise_sweep(6, 180, 1600, 6)
    st = st / np.abs(st).max() * np.interp(np.arange(len(st)) / SR, [0, 3, 5.4, 6], [0, .7, 1, 0])
    add(st, r[10]["a"], .15)
    for t in beats(r):
        k = (t - r[11]["a"]) / (r[14]["a"] - r[11]["a"])
        add(thud(46 + 12 * k, .7), t, .22 + .4 * k)
    # 你好: one clear note
    add(pluck(440.0, 4.5, 90, .7), r[13]["a"] + 1.1, .22)
    add(pluck(587.33, 4.5, 91, .7), r[13]["a"] + 1.7, .2)
    # 精: the fall and the burst; 我: settled
    t1 = r[14]["a"] + 3.4 * .52
    sw = noise_sweep(1.8, 90, 500, 7)
    sw = sw / np.abs(sw).max() * np.linspace(0, 1, len(sw)) ** 2
    add(sw, t1 - 1.8, .3)
    add(thud(38, 4.0), t1, 1.0)
    fl2 = noise_sweep(3, 900, 5000, 8)
    add(fl2 / np.abs(fl2).max() * np.exp(-np.arange(len(fl2)) / SR * 3), t1, .2)
    t2 = r[15]["a"] + 2.2
    add(thud(58, 3.4), t2, .42)
    for i, k in enumerate((0, 3, 5)):
        add(pluck(scale[k], 6.0, 100 + i, .6), t2 + i * .05, .2)
    # 千窗: typing, then the long tone of the pull-back, the lights coming on, the lights going out
    if dur > QT0 + 5:
        T = QT0
        tk = np.exp(-np.arange(int(.03 * SR)) / (SR * .004)) * np.random.default_rng(12).normal(size=int(.03 * SR))
        for i in range(len(QUESTION)):
            add(tk, T + .5 + i * 1.7 / len(QUESTION), .05)
        for i in range(len(ANSWER)):
            add(tk, T + 2.6 + i * 1.6 / len(ANSWER), .04)
        n = int(35 * SR)
        tt = np.arange(n) / SR
        drone = sum(np.sin(2 * np.pi * f * tt + .3 * np.sin(tt * .4 + f)) * g for f, g in ((73.42, 1.0), (110.0, .6), (146.83, .35), (220.0, .12)))
        add(drone * np.interp(tt, [0, 6, 20, 30, 35], [0, 1, .8, .5, 0]), T + 4.4, .07)
        r2 = np.random.default_rng(14)
        tops = np.array([587.33, 698.46, 783.99, 880.0, 1046.5, 1174.66, 1396.91])
        def ping(f, d=1.4):
            x = np.arange(int(d * SR)) / SR
            return np.sin(2 * np.pi * f * x) * np.exp(-x * 4.5) * np.minimum(x / .002, 1)
        for k in range(150):       # more and more of them come on
            add(ping(r2.choice(tops)), T + 11.2 + 8.3 * (k / 150) ** .75 + r2.uniform(0, 1.2), r2.uniform(.008, .03))
        for t in Q_OFF_SINGLES:    # the first few go out one at a time, and can be heard
            add(pluck(r2.choice(scale[:5]), 2.5, int(t * 10), .45), T + t, .15)
        offs = off_times(13900)
        for i in np.unique(np.round(np.linspace(0, 1, 70) ** .25 * (len(offs) - 1)).astype(int)):
            add(pluck(r2.choice(scale[:5]), 2.0, int(i) % 977, .35), T + offs[i], .09)
        add(pluck(scale[3], 7.0, 500, .7), T + Q_OFF_FIRST, .24)
    a = a[:int(dur * SR)]
    a *= np.minimum(1, (len(a) - np.arange(len(a))) / (SR * 1.5))
    a = np.tanh(a * 1.15)
    # a small room so the notes do not stop dead
    ir = np.exp(-np.arange(int(1.6 * SR)) / (SR * .45)) * np.random.default_rng(9).normal(size=int(1.6 * SR))
    wet = fftconvolve(a, ir)[:len(a)]
    a = a + wet / np.abs(wet).max() * np.abs(a).max() * .16
    a = a / np.abs(a).max() * .85
    path = os.path.join(OUT, "ink_audio.wav")
    wavfile.write(path, SR, (a * 32767).astype(np.int16))
    print(path, round(len(a) / SR, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*", type=float)
    ap.add_argument("--name", default="opening")
    ap.add_argument("--plate", default=None)
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--audio", default=None)
    o = ap.parse_args()
    if o.cmd == "check":
        sys.exit(0 if cmd_check() else 1)
    if o.cmd == "still":
        cmd_still(o.args, o.t_from)
    elif o.cmd == "plate":
        cmd_plate(o.args[0], o.args[1], o.name, o.t_from)
    elif o.cmd == "audio":
        cmd_audio(o.args[0])
    elif o.cmd == "render":
        cmd_render(o.name, o.plate, o.audio or os.path.join(OUT, "ink_audio.wav"))
    elif o.cmd == "qstill":
        cmd_qian_still(o.args)
    elif o.cmd == "qplate":
        cmd_qian_plate(o.args[0])
    elif o.cmd == "join":
        cmd_join(o.name)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
