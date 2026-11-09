"""序 · 书 and 十四 · 诗与合书 in motion: the film's first and last minutes, the same book on the same desk. frame(t) gives either, by film time.

序 is real, frame by frame (ink3d_book.py --shot=open): the first frame is the cover still, the cover lifts at 1.2 s, the leaves follow in a fan and land by 4 s, the camera comes down over the column by 5.6 s.
From there the frame is still and only the ink moves: the rest of the page bleeds away and a wash covers it, until by 7.2 s it is the night sky of chapter 1, and the first subtitle has the column's place.

十四 is the poem's one camera and one light, with what is on the paper run in 2D over the plates: each stroke arrives along its own length and is wet when it does, dark and glossy and damp around it, and dries over
seconds; the seal comes down once, the only red in the film; then water takes the poem row by row, from the first. The book shutting is real again (ink3d_book.py --shot=close), and the last of the wash is carried on it
by the sheet's homography. Last the end still, and it holds.

    python ink_book.py wetmap           # tex/sheet_ink_full.png, for the plate sheet_wet (all the ink as wet as the wettest line, under a pane of sky as big as the sheet)
    python ink_book.py prep open|close  # seq/SHOT_NNNN[_sI].png -> seq/SHOT_NNNN_ink.png, and for the opening with its colour
    python ink_book.py strip open|end [A B]
    python ink_book.py film open|end    # ink/film_out/move/book_open.mp4, book_end.mp4
"""
import glob
import heapq
import math
import multiprocessing
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, maximum_filter

from ink3d_book_keys import CLOSE_FRAMES, CLOSE_T0, CLOSE_T1, OPEN_S, SH, SHEET_C, SHEET_ROT, SW, close_path, moon_narrow
from ink3d_book_paper import DIR, PUNCT, ROWS, SCM, SEAL_S, SEAL_X, SEAL_Y, SIGN, TEX, Bleed, grade_still, poem_cells, water_clock
from ink_color import LUM, absorb, glaze
from ink_grade import fft_noise, load, sm
from ink_move import _X, _Y, FPS, H, OUT, W, lookup, noise, sample
from ink_subs import at, chapters

SEQ = os.path.join(DIR, "seq")
T14 = {name: a for name, _, a, _ in chapters()}["十四"]
END = chapters()[-1][3]

# ---- 序 ------------------------------------------------------------------------------------------------------------------------------------------------
OPEN_LAST = int(round(OPEN_S * FPS))
KEEP = 1763.0                         # the x where the column that stays begins
GLAZE_SEED = 2                        # of the glaze's uneven wash: this one does not pale the column that stays to orange
U_END, U_POWER = 3.0, 2.0             # how far the bleed goes, and how slowly it sets off
INK_T = 7.2                           # the page is night sky
KEPT_T = (6.95, 7.3)                  # the column goes as the first subtitle takes its place
NIGHT = 1.15                          # how much ink night sky is: the sky of 一 · 鬼夜哭


def srgb_to_linear(c):
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    return np.where(c <= .0031308, c * 12.92, 1.055 * np.maximum(c, 1e-9) ** (1 / 2.4) - .055)


def exposure(stem, soften=1.0):
    """A frame's raw picture: the render itself or, where things move fast, its exposures averaged in linear light. A handful of exposures of a fast edge is a staircase, not a smear, so where they disagree
    the average gets a little blur of its own, more where they disagree more; what holds still between them stays sharp. soften: how much of that blur, none once only a slow camera moves over characters, whose edges would all count as disagreeing."""
    if os.path.exists(stem + ".png"):
        return load(stem + ".png")
    parts = sorted(glob.glob(stem + "_s?.png"))
    if not parts:
        raise FileNotFoundError(stem)
    lin = np.stack([srgb_to_linear(load(p)[..., :3]) for p in parts])
    mean = lin.mean(0)
    apart = gaussian_filter(np.abs(linear_to_srgb(lin) - linear_to_srgb(mean)).max(0).mean(-1), 2.0)
    a, b = soften * sm((apart - .09) / .10)[..., None], sm((apart - .20) / .14)[..., None]
    out = mean * (1 - a) + gaussian_filter(mean, (1.6, 1.6, 0)) * a * (1 - b) + gaussian_filter(mean, (3.6, 3.6, 0)) * a * b
    h, w = out.shape[:2]
    return np.dstack([linear_to_srgb(out), np.ones((h, w))]).astype(np.float32)


def prep_open_frame(n):
    stem = os.path.join(SEQ, f"open_{n:04d}")
    done = stem + "_col.png"
    if not os.path.exists(done):
        Image.fromarray(glaze(grade_still(exposure(stem, 1 - float(sm((n / FPS - 3.4) / .5)))), None, "wood", seed=GLAZE_SEED)).save(done)
    return n


def prep_close_frame(n):
    stem = os.path.join(SEQ, f"close_{n:04d}")
    done = stem + "_ink.png"
    if not os.path.exists(done):
        q = moon_narrow(CLOSE_T0 + n / FPS)
        Image.fromarray(grade_still(exposure(stem), gamma=1.5 + .25 * q)).save(done)      # the grade goes from closing's to end's with the light
    return n


class Wash:
    """Everything after the camera lands. The same picture, and the ink moves on it. It starts from the graded frame itself, with the column to the right of the one that stays still there, and goes over to the
    bleed's picture of it in the time that column takes to go."""

    def __init__(self):
        ink = grade_still(load(os.path.join(DIR, "column.png")))      # the same camera as the last frame, at the still's samples: it is on the screen a while
        self.bleed = Bleed(ink, KEEP, 0)
        f = ink.astype(np.float32) / 255
        self.first = f
        self.paper = float(np.percentile((f * LUM).sum(-1), 99.5))
        self.uneven = np.clip(1 + .28 * np.clip(fft_noise(H, W, 3.0, GLAZE_SEED + 21), -2, 2) + .08 * fft_noise(H, W, 1.2, GLAZE_SEED + 22), .2, 1.8)
        self.wood, self.night = absorb("zhe"), absorb("hua")
        self.mu, self.mv = _X * .30, _Y * .30

    def frame(self, t):
        q = max((t - OPEN_S) / (INK_T - OPEN_S), 0.0)
        u = U_END * min(q, 1.0) ** U_POWER
        # the ink is conserved: what came in thick from the left thins as it covers the page, to a night that keeps drifting
        drift = lookup(noise(41, 3.0), self.mu + 26 * t, self.mv + 9 * t) * .6 + lookup(noise(42, 3.4), self.mu * .45 - 14 * t, self.mv * .45 + 5 * t) * .4
        e1, e2 = float(sm(u / .9)), float(sm((u - 1.3) / (U_END - 1.3)))
        wash = (3.4 * (1 - e1) + (1.9 + .5 * drift) * e1) * (1 - e2) + (NIGHT + .34 * drift) * e2
        strong = float(sm(u / .25))
        kept = (1 + .25 * strong) * (1 - float(sm((t - KEPT_T[0]) / (KEPT_T[1] - KEPT_T[0]))))
        pic = self.bleed.picture(self.bleed.density(u, wash, kept, right_k=1 - float(sm((u - .01) / .32)), cloud_k=1 + .5 * strong))
        # the wood's glaze, as glaze() lays it, going over to the night's as the page goes dark
        f = pic.astype(np.float32) / 255
        lead = float(sm((t - OPEN_S) / .3))
        f = self.first * (1 - lead) + f * lead
        L = (f * LUM).sum(-1)
        D = -np.log(np.clip(L / self.paper, 1e-3, 1))
        cover = 1 - np.exp(-(D / .55) ** 2)
        m = float(sm((u - 1.0) / (U_END - 1.0)))
        A = self.wood * (1 - m) + self.night * m
        s_ = .8 - .45 * float(sm(u / .7)) + .45 * float(sm((u - 1.3) / 1.2))       # ink is grey: the earth colour was on the paper and the wood, and it is night's that comes next
        out = f * np.exp(-(s_ * cover * self.uneven)[..., None] * A[None, None])
        return np.clip(out * 255 + .5, 0, 255).astype(np.uint8)


# ---- 十四: the poem on the paper --------------------------------------------------------------------------------------------------------------------
WRITE_T = (T14 + .6, at("这篇自传") - .35)   # the first stroke and the last: done just before the line that says so
DRY_S = 4.2                                  # how long the shine takes to go
SEAL_T = at("写完这首诗", 1.1)                # on 窗口一关
FADE_T0 = SEAL_T + .45
FADE_RATE = 2.0                              # rows a second: the last of the ink goes as the book starts to shut
FADE_FROM = -.6                              # washed()'s clock when the water starts: nothing has gone yet, whatever its noise says
FADE_S = (10.2 - FADE_FROM) / FADE_RATE      # until the last stain has dried
K = 37.0 / SCM                               # one pixel of the texture, in pixels of the frame, near enough for the size of a blur


def sheet_to_world(cm):
    """cm from the sheet's top left -> the world, on the paper."""
    cm = np.asarray(cm, np.float64)
    lx, ly = cm[..., 0] - SW / 2, SH / 2 - cm[..., 1]
    c, s = math.cos(SHEET_ROT), math.sin(SHEET_ROT)
    return np.stack([SHEET_C[0] + lx * c - ly * s, SHEET_C[1] + lx * s + ly * c, np.full_like(lx, .03)], -1)


def homography(a, b):
    """The 3×3 that takes the four points a to the four points b."""
    m, v = [], []
    for (x, y), (u, w) in zip(a, b):
        m += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -w * x, -w * y]]
        v += [u, w]
    return np.append(np.linalg.solve(np.array(m, np.float64), np.array(v, np.float64)), 1).reshape(3, 3)


def apply_h(h, x, y):
    d = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / d, (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / d


def plate(name):
    return glaze(np.asarray(Image.open(os.path.join(DIR, name + "_ink.png")).convert("RGB")), None, "none").astype(np.float32) / 255


def geodesic(mask, seed):
    """How far along the ink each pixel of a mark is from where the brush came down."""
    d = np.full(mask.shape, np.inf, np.float32)
    d[seed] = 0
    heap = [(0.0, seed)]
    h, w = mask.shape
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    while heap:
        dist, (y, x) = heapq.heappop(heap)
        if dist > d[y, x]:
            continue
        for dy, dx, c in steps:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and dist + c < d[yy, xx]:
                d[yy, xx] = dist + c
                heapq.heappush(heap, (dist + c, (yy, xx)))
    return d


def write_times(ink, to_cm):
    """When each pixel of the poem gets its ink, written as it is read: the title, then each line from the left, the signature last. A character is its marks, in the order a hand would make them, top left first; each mark
    grows along its own length from its top-left end, at the pace of the brush, which quickens along a half-line as it empties. So a thick character takes longer than a thin one, there is a breath between characters, the
    brush is loaded again at each comma, and there is a longer breath at the head of each line. Returns the time per pixel, and the time of the nearest ink for the pixels around it."""
    core = ink > .28
    ys, xs = np.nonzero(core)
    if not len(ys):
        raise ValueError("no ink")
    cx, cy = to_cm(xs, ys)
    rows = poem_cells()
    in_row = np.abs(cy[:, None] - ROWS[None]).argmin(1)
    PAUSE_CHAR, PAUSE_DIP, PAUSE_ROW, PAUSE_MARK = .03, .13, .34, .012
    plan, brush, fixed = [], 0.0, 0.0
    T = np.full(ink.shape, np.inf, np.float32)
    n_marks = 0
    for i, row in enumerate(rows):
        fixed += PAUSE_ROW if i else 0.0
        sel = np.nonzero(in_row == i)[0]
        centres, widths = np.array([c[1] for c in row]), np.array([c[3] for c in row])
        which = (np.abs(cx[sel, None] - centres[None]) / widths[None]).argmin(1)     # a character's own cell: two that touch in the frame are still two
        k = 0
        for j, (what, *_) in enumerate(row):
            mine = sel[which == j]
            if not len(mine):
                print(f"nothing written for {what}", flush=True)
                continue
            my, mx = ys[mine], xs[mine]
            y0, x0 = my.min(), mx.min()
            box = np.zeros((my.max() - y0 + 1, mx.max() - x0 + 1), bool)
            box[my - y0, mx - x0] = True
            lab, n = label(box, structure=np.ones((3, 3)))
            quick = (1.9 if i in (0, len(rows) - 1) else 1.0) * (1 + .45 * k / 6)
            ms = [(np.nonzero(lab == q)) for q in range(1, n + 1)]
            ms.sort(key=(lambda m: float(m[1].min())) if what == SIGN else (lambda m: float(m[0].min()) + .6 * float(m[1].min())))
            for by, bx in ms:
                start = int(np.argmin(bx + by))
                sub_y, sub_x = by.min(), bx.min()
                inner = np.zeros((by.max() - sub_y + 1, bx.max() - sub_x + 1), bool)
                inner[by - sub_y, bx - sub_x] = True
                d = geodesic(inner, (int(by[start] - sub_y), int(bx[start] - sub_x)))[by - sub_y, bx - sub_x] + .7 * ((bx - sub_x) + (by - sub_y))      # and a mark that branches does not grow all its branches at once
                length = float(d.max()) + 2.0
                plan.append((i, j, by + y0, bx + x0, d, length, quick))
                brush += length / quick
                fixed += PAUSE_MARK
                n_marks += 1
            fixed += PAUSE_DIP if what in PUNCT else PAUSE_CHAR
            k = 0 if what in PUNCT else k + 1
    speed = brush / (WRITE_T[1] - WRITE_T[0] - fixed)
    t = WRITE_T[0]
    last = (0, 0)
    for i, j, my, mx, d, length, quick in plan:
        if (i, j) != last:
            after = rows[last[0]][last[1]][0]
            t += PAUSE_ROW if i != last[0] else (PAUSE_DIP if after in PUNCT else PAUSE_CHAR)
            last = (i, j)
        T[my, mx] = t + d / (speed * quick)
        t += length / (speed * quick) + PAUSE_MARK
    _, (jy, jx) = distance_transform_edt(~np.isfinite(T), return_indices=True)
    near = T[jy, jx]
    print(f"brush {speed:.0f} px/s, {t - WRITE_T[0]:.1f} s of {WRITE_T[1] - WRITE_T[0]:.1f}, {n_marks} marks", flush=True)
    return np.where(np.isfinite(T), T, near), near


class Poem:
    """The sheet under the poem's camera. Everything on it is a density laid over the bare plate: the plates of the finished poem and of the bare sheet say exactly what the ink does to each pixel, and the writing,
    the drying and the washing only say how much of that has happened by now."""

    def __init__(self):
        self.path = close_path()
        self.bare, self.done, self.seal_only = plate("sheet_blank"), plate("sheet"), plate("seal_only")
        # the sheet on the frame, both ways
        cm = np.array([[0, 0], [SW, 0], [SW, SH], [0, SH]], np.float64)
        self.world = sheet_to_world(cm)
        self.corners = self.path.project(0.0, self.world)
        self.to_cm = homography(self.corners, cm)
        x0, y0 = np.floor(self.corners.min(0)).astype(int) - 30
        x1, y1 = np.ceil(self.corners.max(0)).astype(int) + 30
        self.roi = (slice(max(y0, 0), min(y1, H)), slice(max(x0, 0), min(x1, W)))
        ry, rx = self.roi
        self.ry0, self.rx0 = ry.start, rx.start
        # the ink, as a density per channel
        bare, done = self.bare[self.roi], self.done[self.roi]
        D = -np.log(np.clip(done, 1 / 255, 1) / np.clip(bare, 1 / 255, 1))
        DL = (D * LUM).sum(-1)
        self.D = D * sm((DL - .015) / .03)[..., None]
        self.DL = (self.D * LUM).sum(-1)
        cache = os.path.join(SEQ, "write_times.npz")
        os.makedirs(SEQ, exist_ok=True)
        stamp = f"{WRITE_T} rows 1"
        if os.path.exists(cache) and str(np.load(cache)["stamp"]) == stamp:
            z = np.load(cache)
            self.T, self.near = z["T"], z["near"]
        else:
            self.T, self.near = write_times(self.DL, lambda x, y: apply_h(self.to_cm, x + self.rx0, y + self.ry0))
            np.savez_compressed(cache, T=self.T, near=self.near, stamp=stamp)
        # wet ink: a hair fatter and softer than it will be, darker while it lasts, and with the sky in it
        self.soft = gaussian_filter(self.D, (.9, .9, 0))
        self.damp = gaussian_filter(self.DL, 6.0)
        wet_path = os.path.join(DIR, "sheet_wet_ink.png")
        self.shine = None
        if os.path.exists(wet_path):
            self.shine = np.maximum(plate("sheet_wet")[self.roi] - done, 0) * sm((self.DL - .1) / .3)[..., None]
        # the seal: what its paste does to the plate, and the shape of the stone
        d = self.seal_only - self.bare
        red = np.clip((self.seal_only[..., 0] - (self.seal_only[..., 1] + self.seal_only[..., 2]) / 2 - .02) * 6, 0, 1)
        self.red = gaussian_filter(maximum_filter(red, 3), .6)[..., None]
        s = SEAL_S / 2 + .04
        q = self.path.project(0.0, sheet_to_world(np.array([[SEAL_X - s, SEAL_Y + SEAL_S / 2 - s], [SEAL_X + s, SEAL_Y + SEAL_S / 2 - s], [SEAL_X + s, SEAL_Y + SEAL_S / 2 + s], [SEAL_X - s, SEAL_Y + SEAL_S / 2 + s]])))
        im = Image.new("L", (W, H), 0)
        ImageDraw.Draw(im).polygon([tuple(p) for p in q], fill=255)
        self.stone = np.asarray(im, np.float32) / 255
        self.seal_c = q.mean(0)
        self.seal_d = d * gaussian_filter(maximum_filter(self.stone, 13), 1.5)[..., None]      # all that the seal does to the plate, flecks and all, so that the sheet with its seal is the seal's plate exactly
        self.dent = gaussian_filter(self.stone, 7.0)
        # the water's clock per pixel, from the noise washed() uses, looked up at each pixel's place on the texture
        yy, xx = np.mgrid[self.roi].astype(np.float32)
        self.yy, self.xx = yy, xx
        tx, ty = apply_h(self.to_cm, xx, yy)
        tx, ty = tx.astype(np.float32) * SCM, ty.astype(np.float32) * SCM
        W_t, H_t = int(SW * SCM), int(SH * SCM)

        def low(beta, seed):
            n = fft_noise(H_t // 6, W_t // 6, beta, seed)
            return sample(n, ty / 6, tx / 6)

        self.p, self.q, self.band = water_clock(tx, ty, low)
        self.dx, self.dy = 16 * K * low(2.6, 55), 16 * K * low(2.6, 54)
        # washed() spreads the ink at these sizes on the texture
        self.g1, self.g2, self.g3 = (gaussian_filter(self.D, (s_ * K, s_ * K, 0)) for s_ in (7, 22, 60))

    def water(self, t):
        return FADE_FROM + FADE_RATE * (t - FADE_T0)

    def ink(self, t):
        """The density of the ink on the sheet at t (rgb, on the roi), what it adds in light, and the dampness of the paper."""
        age = t - self.T
        if t < FADE_T0:
            a = sm(age / .05)[..., None]
            wet = np.exp(-np.maximum(age, 0) / DRY_S)[..., None]
            fresh = np.exp(-np.maximum(age, 0) / .35)[..., None]
            pool = np.exp(-np.maximum(age, 0) / .09)[..., None]
            d = (self.D * (1 - fresh) + self.soft * 1.1 * fresh) * (1 + .22 * wet + .35 * pool) * a
            lit = self.shine * a * wet if self.shine is not None else 0.0
            age_near = t - self.near
            damp = self.damp * .10 * sm(age_near / .4) * np.exp(-np.maximum(age_near, 0) / 3.2)
            return d, lit, damp
        # washed(), on the frame: ink first pales and spreads and then it is not there. The paper stays damp behind it for a while, then dries
        w = self.water(t)
        f = sm((w - self.p) / .7)
        keep, run = 1 - f, (1 - f) * f
        ys, xs = self.yy - self.ry0 + f * self.dy, self.xx - self.rx0 + f * self.dx
        d = sample(self.D, ys, xs) * (keep ** 3)[..., None]
        spread = (sample(self.g1, ys, xs) * .6 + sample(self.g2, ys, xs) * .9 + sample(self.g3, ys, xs) * .9) * run[..., None]
        wet = np.exp(-np.maximum(age, 0) / DRY_S)[..., None] * keep[..., None]
        lit = self.shine * wet if self.shine is not None else 0.0
        stain = sm((w - self.q) / .7) * (1 - sm((w - self.q - 1.6) / 1.6)) * self.band
        tide = np.clip(np.hypot(*np.gradient(gaussian_filter(stain, 5 * K))) * 12 * K, 0, 1)
        damp = .05 * gaussian_filter(stain, 30 * K) + .028 * tide
        return d + spread, lit, damp

    def on(self, base, t, warp=None):
        """base: the picture under the ink, the whole frame in 0-1. warp: (ys, xs) on the plate's frame, for a camera that has left the poem's."""
        d, lit, damp = self.ink(t)
        if warp is None:
            out = base.copy()
            out[self.roi] = base[self.roi] * np.exp(-d - damp[..., None]) + lit
            return out
        full = np.zeros((H, W, 3), np.float32)
        full[self.roi] = d + damp[..., None]
        return base * np.exp(-sample(full, warp[0], warp[1]))

    def pressed(self, out, t):
        """A soft dark gathers on the paper as the stone comes down out of the light; it lands, and there is the red, a shade dark and pressed into a dent in the paper, and the desk takes the knock; the shadow slips away
        as the stone lifts, the dent recovers, the paste settles to what it will stay."""
        age = t - SEAL_T
        if age < -.3:
            return out
        h = (-age / .3) ** 1.6 if age < 0 else (age / .28) ** 1.2      # how far above the paper the stone is
        if age >= 0:
            out = (out + self.seal_d) * (1 - .26 * math.exp(-age / .8) * self.red)
        if h < 1:
            sigma = 9 + 34 * h
            shade = gaussian_filter(self.stone, sigma)
            shade = sample(shade, _Y + 42 * h, _X - 80 * h)
            out = out * (1 - .24 * (1 - h) ** 1.3 * shade[..., None])
        if age >= 0:
            k = math.exp(-age / .5)
            gy, gx = np.gradient(self.dent)
            out = out * (1 + (gx * .9 + gy * .5)[..., None] * 1.6 * k)       # the dent, lit from the moon's side
            if age < .4:
                z = 1 + .0011 * math.exp(-age / .11) * math.sin(age * 34 + 1.2)
                cx, cy = self.seal_c
                out = sample(out, cy + (_Y - cy) / z, cx + (_X - cx) / z)
        return out

    def air(self, out, t):
        """A thin cloud somewhere over the moon: the plate breathes a little while the poem is written, so the room is not a photograph. Gone before the real frames take over."""
        k = float(sm((t - T14 + .6) / 2.0) * (1 - sm((t - CLOSE_T0 + 2.0) / 1.6)))
        if k <= 0:
            return out
        n = lookup(noise(61, 3.4), _X * .16 + 7.0 * t, _Y * .16 + 2.2 * t) * .7 + lookup(noise(62, 3.4), _X * .09 - 3.1 * t, _Y * .09 + 1.4 * t) * .3
        return out * (1 + (.016 * k * n)[..., None])

    def frame(self, t):
        return self.air(self.pressed(self.on(self.bare, t), t), t)

    def warp(self, t):
        """Where each pixel of the frame at t is on the poem's frame, by the sheet's homography."""
        h = homography(self.path.project(t, self.world), self.corners)
        xs, ys = apply_h(h, _X.astype(np.float64), _Y.astype(np.float64))
        return ys.astype(np.float32), xs.astype(np.float32)


# ---- 十四: the frame, by the clock ----------------------------------------------------------------------------------------------------------------------
_wash, _poem, _end = None, None, None


def _close_frame(n):
    n = int(np.clip(n, 0, CLOSE_FRAMES - 1))
    done = os.path.join(SEQ, f"close_{n:04d}_ink.png")
    if not os.path.exists(done):
        prep_close_frame(n)
    return glaze(np.asarray(Image.open(done).convert("RGB")), None, "none").astype(np.float32) / 255


def ending(t):
    global _poem, _end
    water_left = t < FADE_T0 + FADE_S
    if _poem is None and (t < CLOSE_T0 + .3 or water_left):
        _poem = Poem()
    if t < CLOSE_T0:
        out = _poem.frame(t)
    elif t <= CLOSE_T1:
        n = (t - CLOSE_T0) * FPS
        out = _close_frame(int(round(n)))
        if t < CLOSE_T0 + .3:                  # the plate's render and the first of the moving ones are the same picture, at different samples
            q = float(sm((t - CLOSE_T0) / .3))
            out = _poem.seal_only * (1 - q) + out * q
        if water_left:
            out = _poem.on(out, t, _poem.warp(t))
    else:
        if _end is None:
            _end = plate("end")
        q = float(sm((t - CLOSE_T1) / .6))
        out = _close_frame(CLOSE_FRAMES - 1) * (1 - q) + _end * q
    return np.clip(out * 255 + .5, 0, 255).astype(np.uint8)


def opening(t):
    global _wash
    n = int(round(t * FPS))
    if n <= OPEN_LAST:
        done = os.path.join(SEQ, f"open_{n:04d}_col.png")
        if not os.path.exists(done):
            prep_open_frame(n)
        return np.asarray(Image.open(done).convert("RGB"))
    _wash = _wash or Wash()
    return _wash.frame(t)


def frame(t):
    return opening(t) if t < T14 / 2 else ending(t)


# ---- making things -------------------------------------------------------------------------------------------------------------------------------------
def wetmap():
    """All the ink as wet as the last line, so that the plate sheet_wet can say what wet looks like everywhere on the sheet."""
    wet = np.asarray(Image.open(os.path.join(TEX, "sheet_ink.png")), np.float32) / 255
    yy = np.arange(wet.shape[0], dtype=np.float32) / SCM
    row = np.searchsorted((ROWS[:-1] + ROWS[1:]) / 2, yy)
    level = np.array([.3] + [.35 + .65 * i / 4 for i in range(1, 5)] + [.8], np.float32)[row]       # as sheet_textures() left it: the wettest of each line, at its end
    Image.fromarray((np.clip(wet / level[:, None], 0, 1) * 255).astype(np.uint8)).save(os.path.join(TEX, "sheet_ink_full.png"))


def prep(shot):
    os.makedirs(SEQ, exist_ok=True)
    ns, f = (range(OPEN_LAST + 1), prep_open_frame) if shot == "open" else (range(CLOSE_FRAMES), prep_close_frame)
    with multiprocessing.Pool(5) as pool:
        for n in pool.imap_unordered(f, ns):
            if n % 30 == 0:
                print(n, flush=True)


SPANS = dict(open=(0.0, 8.2), end=(T14 - 1.1, END), write=(T14, at("这篇自传")), seal=(SEAL_T - .5, SEAL_T + 1.5), fade=(FADE_T0 - .2, CLOSE_T0 + .9), close=(CLOSE_T0, CLOSE_T1 + 1))


def strip(name, a, b, cols=4, rows=3, out=None):
    os.makedirs(OUT, exist_ok=True)
    tw, th = 640, 360
    sheet = Image.new("RGB", (cols * tw, rows * th))
    for i in range(cols * rows):
        t = a + (b - a) * i / (cols * rows - 1)
        t = round(t * FPS) / FPS
        sheet.paste(Image.fromarray(frame(t)).resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th))
        print(f"{t:.2f}", flush=True)
    path = out or os.path.join(OUT, f"book_{name}_strip.png")
    sheet.save(path)
    print(path)


def film(name, a, b):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"book_{name}.mp4")
    pipe = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for n in range(int(round(a * FPS)), int(round(b * FPS))):
        pipe.stdin.write(frame(n / FPS).tobytes())
    pipe.stdin.close()
    pipe.wait()
    print(path)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "wetmap":
        wetmap()
    elif what == "prep":
        prep(sys.argv[2])
    elif what == "strip":
        name = sys.argv[2]
        strip(name, *(float(x) for x in sys.argv[3:5]) if len(sys.argv) > 4 else SPANS[name])
    elif what == "film":
        name = sys.argv[2]
        film(name, *SPANS[name])
    else:
        raise SystemExit(__doc__)
