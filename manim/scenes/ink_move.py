"""What makes a still of the ink film move, when rendering every frame in Cycles would take eight days. A finished picture and its depth pass are enough for
most of it: the camera pushing through real depth, water that runs and shimmers in true perspective, mist drifting between the layers, lights trembling, rings
spreading on still water. Everything here is painted over the coloured picture, so it works on any shot.

    python ink_move.py test myth/marsh ghost [SECONDS]      # ink/film_out/move/NAME.mp4 and a strip of six frames
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label

from ink_color import depth_of, fit, glaze
from ink_grade import fft_noise, load, sm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
STILLS = os.path.join(ROOT, "ink", "film_out", "3d")
OUT = os.path.join(ROOT, "ink", "film_out", "move")
W, H, FPS = 1920, 1080, 30
FAR, ZMAX = 400.0, 100.0
LENS = 1400.0    # pixels; only sets the size of the waves on the water, so it need not match the shot's lens exactly
LUM = np.array([0.30, 0.59, 0.11], np.float32)
PAPER = np.array([241, 234, 219], np.float32) / 255
_Y, _X = np.mgrid[0:H, 0:W].astype(np.float32)
_noise = {}


def noise(seed, beta=2.4):
    """A big tile of noise, kept: every layer looks up into these."""
    if (seed, beta) not in _noise:
        _noise[(seed, beta)] = fft_noise(1024, 1024, beta, seed)
    return _noise[(seed, beta)]


def sample(f, ys, xs):
    """Bilinear lookup at per-pixel positions, about four times quicker than map_coordinates on three channels."""
    xs, ys = np.clip(xs, 0, f.shape[1] - 1.001), np.clip(ys, 0, f.shape[0] - 1.001)
    x0, y0 = xs.astype(np.int32), ys.astype(np.int32)
    ax, ay = xs - x0, ys - y0
    if f.ndim == 3:
        ax, ay = ax[..., None], ay[..., None]
    top = f[y0, x0] * (1 - ax) + f[y0, x0 + 1] * ax
    bottom = f[y0 + 1, x0] * (1 - ax) + f[y0 + 1, x0 + 1] * ax
    return top * (1 - ay) + bottom * ay


def lookup(tile, u, v):
    """The noise tile laid on the world: u, v in tile pixels, wrapping."""
    x0, y0 = np.floor(u).astype(np.int32), np.floor(v).astype(np.int32)
    ax, ay = u - x0, v - y0
    x0, y0, x1, y1 = x0 & 1023, y0 & 1023, (x0 + 1) & 1023, (y0 + 1) & 1023
    return (tile[y0, x0] * (1 - ax) + tile[y0, x1] * ax) * (1 - ay) + (tile[y1, x0] * (1 - ax) + tile[y1, x1] * ax) * ay


class Plate:
    """One coloured still at film size, with what its depth pass says about it."""

    def __init__(self, still, palette, depth=None, white=None):
        path = os.path.join(STILLS, still)
        found = depth or depth_of(path)
        ink = np.asarray(Image.open(path + "_ink.png").convert("RGB"))
        z = load(found) if found else None
        im = Image.fromarray(glaze(ink, z, palette, white=white))
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)
        self.f = np.asarray(im, np.float32) / 255
        self.L = (self.f * LUM).sum(-1)
        if z is None:
            self.dist = np.full((H, W), 60.0, np.float32) * (1.6 - _Y / H)
            self.height = np.ones((H, W), np.float32)
        else:
            z = fit(z, H, W)
            self.dist = np.maximum(gaussian_filter(z[..., 0], 1.2) * FAR, 0.5)
            self.dist[z[..., 0] <= 0] = FAR
            self.height = z[..., 1] * ZMAX

    def water(self, tol=0.05, y0=0.3, level=None):
        """The water: the level given, or the flattest level in the lower part of the frame. Its reflections are drawn on it, so they move with it."""
        if level is None:
            low = self.height[int(y0 * H):]
            counts, edges = np.histogram(low[low > 0], bins=400)
            level = edges[counts.argmax()]
        return gaussian_filter(((np.abs(self.height - level) < tol) & (_Y > y0 * H)).astype(np.float32), 1.5)


class Move:
    """A shot in motion. Set up once from a Plate, then frame(t) for any t."""

    def __init__(self, plate, push=(0.0, 0.0), centre=(0.5, 0.5), water=0.0, flow=(0.0, 1.0), mist=0.0, flames=0.0, rings=(), seconds=8.0, level=None, billow=0.0):
        self.p, self.push, self.seconds = plate, push, seconds
        self.cx, self.cy = centre[0] * W, centre[1] * H
        near = np.percentile(plate.dist, 5)
        self.w = np.clip(near / plate.dist, 0.05, 1.0) ** 0.5     # how much of the push each pixel feels: the near parts move, the far parts hardly
        self.water_amp, self.flow, self.mist, self.rings = water, flow, mist, list(rings)
        self.wm = plate.water(level=level) if water or self.rings else None
        if water:
            self.u = (_X - W / 2) * plate.dist / LENS * 22.0
            self.v = plate.dist * 22.0
            self.scale = np.clip(near / plate.dist, 0.02, 1.0)
        if mist:
            self.mm = sm((plate.dist - 0.12 * FAR) / (0.3 * FAR)) * (1 - sm((plate.L - 0.75) / 0.2))
            self.mu, self.mv = _X * 0.22, _Y * 0.5
        self.flame_amp, self.billow = flames, billow
        if flames:
            from ink_color import blur
            g = np.clip((plate.L - blur(plate.L, 40) - 0.12) / 0.3, 0, 1) * sm((_Y / H - 0.3) / 0.05 + 1)
            self.g = gaussian_filter(g, 2.0)
            ids, n = label(self.g > 0.25)
            phase = np.random.default_rng(3).uniform(0, 100, n + 1)
            phase[0] = 0
            self.phase = gaussian_filter(phase[ids].astype(np.float32), 3.0)
            self.bg = gaussian_filter(plate.f, (9, 9, 0))

    def frame(self, t, f=None, s=None):
        """f: another picture to treat as this plate, for a shot whose lights change under the same camera. s: how far the push has gone, if not by the clock."""
        p = self.p
        u = np.clip(t / self.seconds, 0, 1)
        s = self.push[0] + (self.push[1] - self.push[0]) * u if s is None else s
        zoom = 1 + s * self.w
        xs, ys = self.cx + (_X - self.cx) / zoom, self.cy + (_Y - self.cy) / zoom
        lit = 0
        if self.wm is not None:
            dx, dy, lit = self._water(t)
            xs, ys = xs + dx, ys + dy
        if self.billow:       # ink in water: the whole picture turns over slowly
            xs = xs + self.billow * lookup(noise(31, 3.4), _X * 0.28 + 7 * t, _Y * 0.28 - 3 * t)
            ys = ys + self.billow * lookup(noise(32, 3.4), _X * 0.28 - 4 * t, _Y * 0.28 + 6 * t)
        f = p.f if f is None else f
        if self.flame_amp:
            f = self._flames(t)
        out = sample(f, ys, xs)
        if self.wm is not None:
            out = out * (1 + lit[..., None])
        if self.mist:
            drift = lookup(noise(11, 3.2), self.mu + 9 * t, self.mv + 1.5 * t) * 0.6 + lookup(noise(12, 3.2), self.mu * 0.5 - 5 * t, self.mv * 0.5) * 0.4
            a = (self.mist * self.mm * np.clip(0.5 + 0.8 * drift, 0, 1))[..., None]
            out = out * (1 - a) + PAPER * a
        return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)

    def _water(self, t):
        dx, dy = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
        lit = np.zeros((H, W), np.float32)
        if self.water_amp:
            a, b = self.flow
            n = lookup(noise(21), self.u + a * 30 * t, self.v * 2.2 + b * 30 * t)
            m = lookup(noise(22), self.u * 0.6 - a * 21 * t, self.v * 1.5 + b * 17 * t)
            k = self.water_amp * self.scale * self.wm
            dx = k * (n + 0.6 * m) * 5.0
            dy = k * (m - 0.4 * n) * 2.0
            lit = k * 0.012 * (n - m)
        for t0, x, y, squash, size in self.rings:
            age = t - t0
            if age <= 0 or age > 7:
                continue
            ex, ey = _X - x * W, (_Y - y * H) / squash
            r = np.hypot(ex, ey)
            front = 90 * size * age ** 0.7
            env = np.exp(-((r - front) / (34 * size + 26 * age)) ** 2) * np.exp(-age / 3.2)
            wave = np.sin((r - front) / (9 * size)) * env * self.wm
            dy += 7 * size * wave
            lit += 0.16 * wave
        return dx, dy, lit

    def _flames(self, t):
        """Each light trembles on its own: dims a little toward what is around it."""
        p = self.p
        q = np.sin(self.phase + t * 7.3) * 0.5 + np.sin(self.phase * 1.7 + t * 11.9) * 0.3 + np.sin(self.phase * 0.6 + t * 3.1) * 0.2
        a = (self.flame_amp * self.g * (0.5 - 0.5 * q))[..., None]
        return p.f * (1 - a) + self.bg * a


def write(move, seconds, path):
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", path]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(int(seconds * FPS)):
        pipe.stdin.write(move.frame(n / FPS).tobytes())
    pipe.stdin.close()
    pipe.wait()


def test(still, palette, seconds=4.0):
    os.makedirs(OUT, exist_ok=True)
    plate = Plate(still, palette)
    wm = plate.water()
    print(still, "water share", round(float(wm.mean()), 3), "dist", [round(float(x)) for x in np.percentile(plate.dist, [5, 50, 95])])
    move = Move(plate, push=(0.0, 0.05), water=1.0, mist=0.22, flames=0.8, rings=[(0.4, 0.625, 0.68, 0.22, 1.0), (1.6, 0.34, 0.74, 0.22, 1.2)], seconds=seconds)
    name = still.replace("/", "_")
    path = os.path.join(OUT, name + ".mp4")
    write(move, seconds, path)
    strip = Image.new("RGB", (640 * 3, 360 * 2))
    for i in range(6):
        strip.paste(Image.fromarray(move.frame(seconds * i / 5.5)).resize((640, 360), Image.LANCZOS), (640 * (i % 3), 360 * (i // 3)))
    strip.save(os.path.join(OUT, name + "_strip.png"))
    print(path)


if __name__ == "__main__":
    if sys.argv[1:2] == ["test"]:
        test(sys.argv[2], sys.argv[3], *(float(x) for x in sys.argv[4:5]))
    else:
        raise SystemExit(__doc__)
