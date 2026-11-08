"""Wet ink on wet paper, for the opening myth 鬼夜哭. A real two-layer simulation, after Curtis et al. 1997 but with a porous sheet instead of a shallow-water solver:

    water    moves down its own gradient through the fibres, only above what the paper holds on to, so a wet edge stops and is sharp; the fibres are conduits, so it creeps along them
    pigment  rides the water, fine and coarse apart, the coarse lagging and settling first, so a pale halo runs ahead of the dark; it piles up where the water stops and cannot go on: the tide line
    the sheet  takes pigment into its valleys (granulation) and gives a little back when it is wetted again, so clean water dropped in damp ink pushes a dark curled rim outward: the bloom
    a reserve  stops both. A ghost is a reserve: paper the ink never reached

A plate runs coarse first for the big movement, then at full size for the edges. Wash.plate() is what the frames take: the ink as a density, in film pixels.

    python film_myth_wash.py test [scale]     # one sheet of clouds with no figure, to look at the ink alone

Everything goes to ink/film_out/film/myth/keys/. This file is imported by film_myth.py and changes nothing else in the film.
"""
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates

from ink_grade import fft_noise, sm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "myth", "keys")
W, H = 1920, 1080
PAPER_RGB = np.array((244, 237, 221), np.float32)      # warm and bright: the night is said by the clouds and the moon, never by darkening the sheet
INK_ABSORB = np.array((1.0, 1.0, 0.965), np.float32)   # thin soot ink leans a little cool on a warm sheet, thick is a warm black
QING_ABSORB = np.array((1.0, 0.70, 0.62), np.float32)  # the thin blue-green over a ghost's body


def band(h, w, lo, hi, seed):
    v = np.random.default_rng(seed).normal(0, 1, (h, w)).astype(np.float32)
    v = gaussian_filter(v, lo) - gaussian_filter(v, hi)
    return v / v.std()


class Sheet:
    """One handmade sheet: what is seen of it, and what the water and the pigment meet in it. All at film size."""

    def __init__(self, seed=91, w=W, h=H):
        self.w, self.h, self.seed = w, h, seed
        r = np.random.default_rng(seed)
        # the fibres that show, long and lying every way (the recipe of film_hand.sheet)
        im = Image.new("L", (w * 2, h * 2), 128)
        d = ImageDraw.Draw(im)
        for _ in range(int(w * h / 900)):
            x, y, a, n = r.uniform(0, w * 2), r.uniform(0, h * 2), r.uniform(0, np.pi), r.uniform(50, 300)
            bend = r.uniform(-0.6, 0.6)
            d.line([(x + np.cos(a + bend * u) * n * u, y + np.sin(a + bend * u) * n * u) for u in np.linspace(0, 1, 8)], fill=int(r.choice([104, 116, 142, 156])), width=1)
        fibres = np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 128 - 1
        # the fibres water runs along: more of them, finer, most of them near the direction the sheet was lifted
        im = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(im)
        for _ in range(int(w * h / 260)):
            x, y, n = r.uniform(-60, w), r.uniform(-60, h), r.uniform(18, 150)
            a = r.normal(0, 0.42) if r.uniform() < 0.62 else r.uniform(0, np.pi)
            bend = r.uniform(-0.5, 0.5)
            d.line([(x + np.cos(a + bend * u) * n * u, y + np.sin(a + bend * u) * n * u) for u in np.linspace(0, 1, 5)], fill=int(r.uniform(120, 255)), width=1)
        self.conduit = gaussian_filter(np.asarray(im, np.float32) / 255, 0.55)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        wobble = 2.2 * fft_noise(h, w, 3.6, seed + 2)
        laid = 0.5 * np.sin(2 * np.pi * (yy + wobble) / 6.3) + 0.5 * np.exp(-(((xx + wobble * 2) % 172 - 86) / 1.4) ** 2)
        height = 0.28 * fft_noise(h, w, 1.2, seed + 3) + 0.25 * fft_noise(h, w, 2.4, seed + 4) + 2.2 * fibres + 0.26 * laid
        gy, gx = np.gradient(height)
        relief = gx * -0.62 + gy * -0.78
        fall = 1 - 0.035 * np.hypot((xx - w * 0.3) / w, (yy - h * 0.2) / h)
        flocs = band(h, w, 9, 40, seed + 6)
        tone = 1 + 0.008 * fft_noise(h, w, 3.4, seed + 5) + 0.03 * fibres + 0.02 * relief + 0.004 * flocs
        self.rgb = np.clip(PAPER_RGB * (tone * fall)[..., None], 0, 255)
        v = gaussian_filter(height, 0.7) - gaussian_filter(height, 5)
        self.tooth = np.float32(v / v.std())
        self.flocs = flocs
        # what the water meets
        self.n_k = 0.62 * fft_noise(h, w, 3.3, seed + 11) + 0.42 * fft_noise(h, w, 2.4, seed + 12) + 0.16 * band(h, w, 2.0, 7, seed + 13)
        self.n_hold = 0.62 * fft_noise(h, w, 3.0, seed + 14) + 0.28 * band(h, w, 3.0, 18, seed + 15) + 0.10 * band(h, w, 0.7, 3, seed + 16)
        fine_tooth = gaussian_filter(self.tooth, 0.6)
        self.valley = np.float32(0.5 * band(h, w, 0.8, 2.4, seed + 17) - 0.5 * fine_tooth / fine_tooth.std())
        along = gaussian_filter(self.conduit, 1.2)
        along = along / along.std()
        self.warp = (np.float32(0.9 * band(h, w, 1.2, 5, seed + 18) + 1.5 * band(h, w, 4, 16, seed + 19) + 0.6 * along), np.float32(0.9 * band(h, w, 1.2, 5, seed + 20) + 1.5 * band(h, w, 4, 16, seed + 21)))


def shrink(a, s):
    if s == 1:
        return a
    h, w = int(round(a.shape[0] * s)), int(round(a.shape[1] * s))
    return np.asarray(Image.fromarray(np.float32(a), "F").resize((w, h), Image.BILINEAR if s > 0.75 else Image.BOX), np.float32)


def grow(a, h, w):
    return np.asarray(Image.fromarray(np.float32(a), "F").resize((w, h), Image.BICUBIC), np.float32)


_dabn = {}


def _dab_noise(h, w, seed):
    key = (h, w, seed % 6)
    if key not in _dabn:
        _dabn[key] = 0.75 * fft_noise(h, w, 2.6, 300 + seed % 6) + 0.25 * band(h, w, 1.2, 6, 310 + seed % 6)
    return _dabn[key]


class Wash:
    """The water and pigment on one sheet, on a grid at scale s of the film. Fine and coarse pigment: index 0 and 1."""

    MOB = (0.92, 0.42)      # how readily each rides the water
    SETTLE = (0.0005, 0.0035)  # how fast each settles out of deep water

    def __init__(self, sheet, s=1, reserve=None, relief=None, tilt=(0.0, 0.0)):
        self.sheet, self.s = sheet, s
        self.reserve_full = np.zeros((sheet.h, sheet.w), np.float32) if reserve is None else np.float32(reserve)
        self.relief_full = relief
        self.tilt = tilt
        self._fields(s)
        h, w = self.shape
        self.w_ = np.zeros((h, w), np.float32)
        self.c = [np.zeros((h, w), np.float32) for _ in range(2)]
        self.d = [np.zeros((h, w), np.float32) for _ in range(2)]
        self.fix = [np.zeros((h, w), np.float32) for _ in range(2)]
        self.count = 0
        self.edge = np.zeros((h, w), np.float32)

    def _fields(self, s):
        sh = self.sheet
        self.s = s
        k = np.exp(1.0 * shrink(sh.n_k, s)) * (0.5 + 1.8 * shrink(sh.conduit, s) * (1.0 if s == 1 else 1.6))
        k = k / np.median(k)
        block = shrink(self.reserve_full, s)
        self._block = block
        k = k * (1 - np.clip(block, 0, 1)) ** 3
        hold = 0.05 * np.exp(1.0 * shrink(sh.n_hold, s)) + 3.0 * block
        self.shape = h, w = k.shape
        self.valley = shrink(sh.valley, s) if s == 1 else np.zeros((h, w), np.float32)
        tx, ty = self.tilt
        faces = [((slice(None), slice(None, -1)), (slice(None), slice(1, None)), 1.0, tx),
                 ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), 1.0, ty),
                 ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(1, None)), 0.5, tx + ty),
                 ((slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1)), 0.5, ty - tx)]
        self.faces = [(a, b, np.float32(0.5 * (k[a] + k[b]) * q), np.float32(0.5 * (hold[a] + hold[b])), g) for a, b, q, g in faces]
        self.hold = hold

    # ---- putting things on the sheet: arrays at film size, 0 to 1

    def load(self, mask, water, fine=0.0, coarse=0.0):
        m = shrink(mask, self.s)
        self.w_ += m * water * (1 - np.clip(shrink(self.reserve_full, self.s), 0, 1))
        keep = 1 - np.clip(shrink(self.reserve_full, self.s), 0, 1)
        if np.any(fine):
            self.c[0] += m * fine * keep
        if np.any(coarse):
            self.c[1] += m * coarse * keep

    def dab(self, cx, cy, rx, ry, angle=0.0, water=0.3, fine=0.0, coarse=0.0, seed=0, rough=0.55):
        """One touch of a loaded brush, film coordinates: a lobed patch, loaded into what is already there."""
        s, (h, w) = self.s, self.shape
        R = 2.0 * max(rx, ry) * s
        x0, x1, y0, y1 = int(max(cx * s - R, 0)), int(min(cx * s + R + 1, w)), int(max(cy * s - R, 0)), int(min(cy * s + R + 1, h))
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        ca, sa = np.cos(angle), np.sin(angle)
        u, v = ((xx / s - cx) * ca + (yy / s - cy) * sa) / rx, (-(xx / s - cx) * sa + (yy / s - cy) * ca) / ry
        n = _dab_noise(h, w, seed)[y0:y1, x0:x1]
        e = np.exp(-(u * u + v * v))
        m = sm((e * (1 + rough * n) - 0.34) / 0.30) * sm(1 - self._block[y0:y1, x0:x1] / 0.12)
        self.w_[y0:y1, x0:x1] += m * water
        if fine:
            self.c[0][y0:y1, x0:x1] += m * fine
        if coarse:
            self.c[1][y0:y1, x0:x1] += m * coarse

    def run(self, steps, evap=3e-4, edge_loss=4.0, kw=1.0, note=""):
        """Steps and evaporation are given as at scale 0.5: a finer grid takes more steps for the same movement."""
        t0 = time.time()
        tf = (self.s / 0.5) ** 2
        steps, evap = int(steps * tf), evap / tf
        w, c, d, fix = self.w_, self.c, self.d, self.fix
        for i in range(steps):
            n = self.count
            self.count += 1
            for a, b, K, Hf, g in (self.faces[:2] if n % 2 == 0 else self.faces[2:]):
                wa, wb = w[a], w[b]
                dw = wa - wb
                if g:
                    dw = dw + g * (wa + wb)
                up = np.where(dw > 0, wa, wb)
                m = np.maximum(up - Hf, 0)
                D = np.minimum(K * m * (m + 0.3) * kw, 0.22)
                F = D * dw
                ww = w + 1e-3
                pos, neg = np.maximum(F, 0), np.minimum(F, 0)
                near = np.minimum(wa, wb)
                for j in range(2):
                    conc = np.minimum(c[j] / ww, 40.0)
                    ca, cb = conc[a], conc[b]
                    P = (pos * ca + neg * cb) * self.MOB[j] + (0.010 if j == 0 else 0.003) * (ca - cb) * near
                    c[j][a] -= P
                    c[j][b] += P
                wa -= F
                wb += F
            if n % 2 == 0:
                # what settles, what is lifted again, what is gone into the air
                if n % 16 == 0:
                    wet = np.float32(w > self.hold * 0.8)
                    self.edge = wet * (1 - gaussian_filter(wet, 5 * self.s + 1.0))
                thin = np.clip(1 - w / 0.11, 0, 1)
                thin = thin * thin
                for j in range(2):
                    rate = 2 * (self.SETTLE[j] * (1 + 0.9 * np.maximum(self.valley, -0.8)) + 0.05 * thin)
                    dep = np.maximum(c[j], 0) * np.minimum(rate, 1.0)
                    lift = 2 * 0.006 * d[j] * np.clip((w - 0.10) / 0.3, 0, 1)
                    c[j] += lift - dep
                    d[j] += dep - lift
                w[:] = np.maximum(w - 2 * evap * (1 + edge_loss * self.edge), 0)
                if n % 32 == 0:
                    dry = w < 0.008
                    for j in range(2):
                        fix[j][dry] += d[j][dry] + np.maximum(c[j][dry], 0)
                        d[j][dry] = 0
                        c[j][dry] = 0
        if note:
            print(f"  {note}: {steps} steps at {self.s}, {time.time() - t0:.0f}s, water {w.sum() / w.size:.3f}, wet {float((w > 0.02).mean()):.2f}", flush=True)

    def to_scale(self, s):
        """Carry on on a finer grid. What the water is doing goes across."""
        h, w = int(round(self.sheet.h * s)), int(round(self.sheet.w * s))
        carry = lambda a: np.maximum(grow(a, h, w), 0)
        self.w_ = carry(self.w_)
        self.c = [carry(a) for a in self.c]
        self.d = [carry(a) for a in self.d]
        self.fix = [carry(a) for a in self.fix]
        self._fields(s)
        self.w_ *= 1 - np.clip(shrink(self.reserve_full, s), 0, 1)
        self.edge = np.zeros((h, w), np.float32)
        self.count += self.count % 2

    def density(self):
        """All the pigment on and in the sheet, film size, in units of optical density once develop() has had it. The grid is coarser than the film, so the last millimetre of every edge is put in here: the field is pulled about along the sheet's own fibres, which leaves soft insides soft and makes every hard edge go ragged the way the paper would."""
        fine = self.fix[0] + self.d[0] + np.maximum(self.c[0], 0)
        coarse = self.fix[1] + self.d[1] + np.maximum(self.c[1], 0)
        h, w = self.sheet.h, self.sheet.w
        if self.shape != (h, w):
            fine, coarse = np.maximum(grow(fine, h, w), 0), np.maximum(grow(coarse, h, w), 0)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = self.sheet.warp
        fine = map_coordinates(fine, [yy + dy, xx + dx], order=1, mode="nearest")
        coarse = map_coordinates(coarse, [yy + dy, xx + dx], order=1, mode="nearest")
        return fine, coarse


def develop(sheet, fine, coarse, gain=1.0, grain=0.22, line=None, qing=None, top=2.55):
    """The pigment as seen on the sheet: coarse pigment sits in the valleys of the paper and stays grainy, fine stains evenly. Thin ink shows the tooth most; a solid black fills it."""
    v = np.clip(sheet.valley, -1.6, 1.8)
    D = fine * (1 + 0.025 * v) + coarse * (1 + grain * v)
    D = top * (1 - np.exp(-gain * D / top))
    if line is not None:
        D = D + line
    f = sheet.rgb * np.exp(-D[..., None] * INK_ABSORB)
    if qing is not None:
        f = f * np.exp(-qing[..., None] * QING_ABSORB)
    return np.clip(f, 0, 255)


def noise(sheet, beta, seed):
    return fft_noise(sheet.h, sheet.w, beta, seed)


def lumps(sheet, env, seed, beta=3.0, level=0.0, soft=0.32, amount=0.55):
    """Where a loaded brush went: the shape wanted, roughened at every scale, so it ends in lobes and not in a blur."""
    return sm((env + amount * noise(sheet, beta, seed) - level) / soft)


def blob(x, y, cx, cy, rx, ry, angle=0.0):
    ca, sa = np.cos(angle), np.sin(angle)
    u, v = (x - cx) * ca + (y - cy) * sa, -(x - cx) * sa + (y - cy) * ca
    return np.exp(-(u / rx) ** 2 - (v / ry) ** 2)


def save(f, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".png")
    Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255))).save(path)
    print(path, flush=True)
    return path


def drops(sheet, n, seed, env, rmin=8, rmax=40):
    """Clean water dropped in here and there: small round wet patches, where env says."""
    r = np.random.default_rng(seed)
    h, w = sheet.h, sheet.w
    im = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    placed = 0
    for _ in range(n * 30):
        if placed >= n:
            break
        x, y = r.uniform(0, w), r.uniform(0, h)
        if env[int(y), int(x)] < r.uniform(0.3, 0.9):
            continue
        rad = r.uniform(rmin, rmax)
        x0, x1, y0, y1 = int(max(x - 3 * rad, 0)), int(min(x + 3 * rad, w)), int(max(y - 3 * rad, 0)), int(min(y + 3 * rad, h))
        im[y0:y1, x0:x1] = np.maximum(im[y0:y1, x0:x1], np.exp(-(((xx[y0:y1, x0:x1] - x) / rad) ** 2 + ((yy[y0:y1, x0:x1] - y) / (rad * r.uniform(0.7, 1.1))) ** 2)))
        placed += 1
    return im


def touches(wash, sheet, env, n, seed, rx=(30, 110), ry=(22, 60), water=0.26, fine=1.1, coarse=0.5, every=40, spread=0.25, evap=1.2e-4):
    """A brush going back into the same cloud again and again while it is wet: each touch pushes the last aside, so the mass is made of lobes with pale seams."""
    r = np.random.default_rng(seed)
    done = 0
    for _ in range(n * 60):
        if done >= n:
            break
        x, y = r.uniform(0, sheet.w), r.uniform(0, sheet.h)
        if env[int(y), int(x)] < r.uniform(0.35, 1.0):
            continue
        k = r.uniform(0.55, 1.2)
        wash.dab(x, y, r.uniform(*rx), r.uniform(*ry), r.normal(0, spread), water=water * r.uniform(0.7, 1.3), fine=fine * k, coarse=coarse * k, seed=int(r.integers(1000)))
        wash.run(every, evap=evap)
        done += 1


def test(scale=0.5):
    """The ink alone, on the whole sheet."""
    sheet = Sheet()
    h, w = sheet.h, sheet.w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wash = Wash(sheet, s=scale)
    env = 1.25 * blob(xx, yy, 660, 300, 660, 170, 0.08) + 1.15 * blob(xx, yy, 1250, 720, 620, 170, -0.1)
    wash.load(lumps(sheet, env, 21, level=0.26, soft=0.20), 0.30)
    wash.run(50, evap=1e-4, note="wet")
    touches(wash, sheet, np.clip(env - 0.15, 0, 1), 24, 5, rx=(90, 240), ry=(36, 90), water=0.24, fine=0.26, coarse=0.06, every=16)
    wash.run(320, evap=4e-4, note="pale dries a while")
    touches(wash, sheet, np.clip(env - 0.40, 0, 1), 20, 6, rx=(36, 130), ry=(20, 60), water=0.22, fine=0.8, coarse=0.2, every=20)
    wash.run(220, evap=4e-4, note="mid dries a while")
    touches(wash, sheet, np.clip(env - 0.7, 0, 1), 14, 8, rx=(20, 70), ry=(14, 34), water=0.16, fine=1.0, coarse=0.3, every=20)
    wash.run(100, evap=2e-4, note="settle")
    touches(wash, sheet, np.clip(env - 0.35, 0, 1), 16, 7, rx=(10, 46), ry=(8, 34), water=0.32, every=36, fine=0, coarse=0)
    wash.run(900, evap=5e-4, note="dry")
    fine, coarse = wash.density()
    print("max", float(fine.max()), float(coarse.max()), "bright", float((fine + coarse < 0.1).mean()), "wet", float((wash.w_ > 0.01).mean()))
    save(develop(sheet, fine, coarse), "wash_test")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "test"
    if what == "test":
        test(float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
    else:
        raise SystemExit(__doc__)
