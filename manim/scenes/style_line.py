"""线条动画：一根线，被混沌撕成上千根。One thread of light on the Lorenz attractor.

A thousand threads start a hair apart, so for the first seconds they are one silk ribbon. Chaos
stretches the gap exponentially until the ribbon frays and floods the whole butterfly. Nothing is
scripted but the camera and the clock: the climax is the equation's own. Rendered like a long
exposure, with tens of millions of points through a real lens: stochastic depth of field and
per-channel dispersion.
"""
import math
import os
from functools import lru_cache

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from styles_common import H, OUT, W, surface

N, STEPS = 400, 15000
TAU_END = 30.0
DT = TAU_END / STEPS
SPREAD = .0003
F = 1500.0
CENTRE = np.array((0., 0., 25.5))

TG = np.linspace(0, 5, 1201)


def keyed(*keys):
    k = np.array(keys, float)
    return PchipInterpolator(k[:, 0], k[:, 1])(TG)


# sim time: slow-motion macro first, then fast, then easing to a hold
RATE = keyed((0, .4), (1, .5), (1.6, 1.3), (2.3, 2.8), (3, 3.6), (3.5, 10), (4.2, 10), (4.6, 3), (5, .6))
TAU = 8.0 + np.cumsum(RATE) * (TG[1] - TG[0])


def tau_of(t):
    return float(np.interp(t, TG, TAU))


@lru_cache(1)
def table():
    """(STEPS + 1, N, 3) float32, shared read-only between processes."""
    path = os.path.join(OUT, "cache_line", f"lorenz_{N}_{STEPS}_{SPREAD}.npy")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s = np.array((1., 1., 20.))
        for _ in range(4000):
            s = rk4(s, .004)
        along = np.array((.6, .3, .74))
        k = np.linspace(-.5, .5, N)[:, None]
        p = s + k * along * SPREAD
        out = np.empty((STEPS + 1, N, 3), np.float32)
        out[0] = p
        for i in range(STEPS):
            p = rk4(p, DT)
            out[i + 1] = p
        tmp = f"{path}.{os.getpid()}.npy"
        np.save(tmp, out)
        os.replace(tmp, path)
    return np.load(path, mmap_mode="r")


def lorenz(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.stack((10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z), -1)


def rk4(p, h):
    a = lorenz(p)
    b = lorenz(p + a * h / 2)
    c = lorenz(p + b * h / 2)
    d = lorenz(p + c * h)
    return p + (a + 2 * b + 2 * c + d) * h / 6


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


AZ = keyed((0, 128), (1, 118), (2, 105), (3, 92), (3.6, 78), (4.4, 64), (5, 60))
EL = keyed((0, -6), (1, -2), (2, 2), (3, 4), (5, 5))
ROLL = keyed((0, 94), (1, 100), (2, 110), (3, 135), (3.6, 160), (4.4, 176), (5, 178))
DIST = keyed((0, 17), (1, 19.5), (2, 25), (3, 33), (3.6, 42), (4.4, 53), (5, 57))
SHIFT_X = keyed((0, 70), (2, 30), (3.5, 0), (5, 0))
SHIFT_Y = keyed((0, 135), (1, 100), (2, 40), (3.2, 0), (4.4, -25), (5, -25))
WING = np.array((8.49, 8.49, 27.))


def camera(t):
    az, el, roll = (math.radians(float(np.interp(t, TG, k))) for k in (AZ, EL, ROLL))
    dist = float(np.interp(t, TG, DIST))
    e = float(smooth((t - .6) / 3.4))
    target = WING * (1 - e) + CENTRE * e
    away = np.array((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    pos = target + away * dist
    fwd = -away
    right = np.cross(fwd, (0, 0, 1))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    r = math.cos(roll) * right + math.sin(roll) * up
    u = -math.sin(roll) * right + math.cos(roll) * up
    aperture = float(np.interp(t, (0, 1.5, 3.2, 4.4, 5), (.8, .7, .28, .1, .08)))
    focus = dist + float(np.interp(t, (0, 2.3, 3.3, 5), (24, 24, 0, 0)))
    return pos, np.stack((r, u, fwd)), focus, aperture


@lru_cache(1)
def thread_gain():
    return np.exp(np.random.default_rng(3).normal(0, .75, N)).astype(np.float32)


@lru_cache(1)
def thread_colours():
    k = np.linspace(0, 1, N, dtype=np.float32)[:, None]
    cold = np.array((.10, .42, 1.0), np.float32) * (1 - k) + np.array((.16, .90, 1.0), np.float32) * k
    warm = (np.array((1.0, .70, .22), np.float32) * (1 - k) + np.array((1.0, .46, .20), np.float32) * k) * 1.3
    return cold, warm


PEARL_MEAN = .1494                                           # mean of sin^28 over a period


def accumulate(t, frame):
    P = table()
    K = min(STEPS - 1, int(tau_of(t) / DT))
    pos, R, focus, aperture = camera(t)
    rng = np.random.default_rng(90000 + frame)
    fray = float(smooth((t - 2.5) / 1.1))

    # how many points each step needs for the thread to stay continuous on screen
    probe = np.asarray(P[max(0, K - 400):K + 1, N // 2], np.float32) - pos
    d = probe @ R[2]
    good = d > .8
    if good.sum() > 4:
        sx = probe[good] @ R[0] / d[good] * F
        sy = probe[good] @ R[1] / d[good] * F
        px = np.hypot(np.diff(sx), np.diff(sy))
        need = float(np.percentile(px, 85))
    else:
        need = 8.0
    M = int(np.clip(need / .9 * (1 + 3 * (1 - fray)), 4 + 12 * (1 - fray), 60))   # a defocused thread needs far more points to stay smooth
    if K * N * M > 120_000_000:
        M = max(1, int(120_000_000 / (K * N)))

    index = np.arange(K, dtype=np.float32)
    age = K - index
    floor = .34 + .1 * fray
    trail = 500 + 1600 * fray
    w_step = floor + (1 - floor) * np.exp(-age / trail)
    pearls = ((np.sin(index * .31) ** 2) ** 14).astype(np.float32)
    w_step = (w_step / (N * M)).astype(np.float32)
    w_head = ((46 - 14 * fray) * np.exp(-age / 4.5) / (N * M)).astype(np.float32)
    amp = 7.0 * (1 - float(smooth((t - 2.15) / .35)))           # gone before the threads part, or the beads line up across them as rungs

    acc = np.zeros((3, H * W), np.float64)
    soft = np.zeros((3, H * W), np.float64)
    haze = np.zeros((3, H * W), np.float64)
    sx, sy = float(np.interp(t, TG, SHIFT_X)), float(np.interp(t, TG, SHIFT_Y))
    cold, warm = thread_colours()
    gain = thread_gain()
    disp = (1.0022, 1.0, .9978)
    A = P[:K]
    B = P[1:K + 1]
    Rf = R.astype(np.float32)

    # A fixed count per step leaves the quick stretches past the lens as sand: each step gets as many points as it crosses pixels.
    probe = np.asarray(P[:K + 1, ::N // 8], np.float32) - pos.astype(np.float32)
    depth = np.maximum(probe @ Rf[2], .7)
    ux, uy = probe @ Rf[0] / depth * F, probe @ Rf[1] / depth * F
    cross = np.hypot(np.diff(ux, axis=0), np.diff(uy, axis=0)).max(axis=1)
    extra = np.clip(np.ceil(cross / .9) - M, 0, 160 - M).astype(np.int64)
    room = max(0, 120_000_000 // N - K * M)
    if extra.sum() > room:
        extra = (extra * room / extra.sum()).astype(np.int64)
    per = (M + extra).astype(np.float32)
    # The newest stretch of a bundle that has just come apart is far too wide for 400 threads to fill, and its edges read as a
    # hard capsule. Where it is wide it becomes a faint haze that condenses into threads as it ages; a thin thread keeps its edge.
    spread = np.hypot(ux.std(axis=1), uy.std(axis=1))[:-1]
    wide = gaussian_filter1d(smooth((spread - 4) / 22), 12)
    gate = float(smooth((t - 2.15) / .3)) * (1 - float(smooth((t - 3.4) / .5)))
    fresh = (gate * wide * np.exp(-(age / 280) ** 2)).astype(np.float32)
    front = gate * smooth(spread / 8) * np.exp(-(age / 120) ** 2)                          # threads all stop at one step: its cut would show as a wedge
    w_step = (w_step * (1 - fresh * (.72 + .28 * np.exp(-age / 60))) * (1 - front)).astype(np.float32)
    w_head = (w_head * (1 - fresh) / (1 + (spread / 8) ** 2)).astype(np.float32)        # a head shared across a wide bundle is no longer a point of light

    def splat(seg, u):
        q = A[seg] + (B[seg] - A[seg]) * u[..., None] - pos.astype(np.float32)
        z = q @ Rf[2]
        keep = z > .7
        x = q @ Rf[0] / np.where(keep, z, 1) * F
        y = -(q @ Rf[1]) / np.where(keep, z, 1) * F
        coc = np.minimum(aperture * F * np.abs(1 / np.where(keep, z, 1) - 1 / focus), 240.0) + .55
        # the beads of light belong to the thread in focus; out of focus they melt into its mean brightness
        sharp = 1 / (1 + (coc[keep] / 5.0) ** 2)
        bead = np.broadcast_to(pearls[seg][:, None], keep.shape)[keep]
        lit = 1 + amp * (sharp * bead + (1 - sharp) * PEARL_MEAN)
        r = coc[keep] * np.sqrt(rng.random(keep.sum(), dtype=np.float32))
        th = rng.random(keep.sum(), dtype=np.float32) * 6.2832
        xs = x[keep] + r * np.cos(th)
        ys = y[keep] + r * np.sin(th)
        wing = .5 + .5 * np.tanh((q[..., 0][keep] + float(pos[0])) / 3.2)
        thread = np.broadcast_to(np.arange(N)[None], keep.shape)[keep]
        # likewise the bright head: passing close to the lens it must not swell into a grainy blob
        head = np.broadcast_to(w_head[seg][:, None], keep.shape)[keep] / (1 + (coc[keep] / 18.0) ** 2)
        share = np.broadcast_to((M / per[seg])[:, None], keep.shape)[keep]
        w = (np.broadcast_to(w_step[seg][:, None], keep.shape)[keep] * lit + head) * gain[thread] * share
        hz = np.broadcast_to(fresh[seg][:, None], keep.shape)[keep]
        mix = np.clip((coc[keep] - 14) / 22, 0, 1) * (1 - hz)
        for ch in range(3):
            cx = W / 2 + sx + xs * disp[ch]
            cy = H / 2 + sy + ys * disp[ch]
            ix = cx.astype(np.int32)
            iy = cy.astype(np.int32)
            ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (cx >= 0) & (cy >= 0)
            col = cold[thread[ok], ch] * (1 - wing[ok]) + warm[thread[ok], ch] * wing[ok]
            at = iy[ok] * W + ix[ok]
            acc[ch] += np.bincount(at, weights=w[ok] * col * (1 - mix[ok] - hz[ok]), minlength=H * W)
            soft[ch] += np.bincount(at, weights=w[ok] * col * mix[ok], minlength=H * W)
            if gate > 0:
                haze[ch] += np.bincount(at, weights=w[ok] * col * hz[ok], minlength=H * W)

    every = np.arange(K)
    for m in range(M):
        splat(every, (m + rng.random((K, N), dtype=np.float32)) / M)  # each thread its own phase: shared phases line up as bands across a bundle
    seg = np.repeat(every, extra)
    if len(seg):
        nth = np.arange(len(seg)) - np.repeat(np.cumsum(extra) - extra, extra)
        for chunk in np.array_split(np.arange(len(seg)), len(seg) // 20000 + 1):
            s = seg[chunk]
            splat(s, ((nth[chunk][:, None] + rng.random((len(s), N), dtype=np.float32)) / extra[s][:, None]).astype(np.float32))
    # wide discs are too thinly sampled to look like anything but sand: gather them apart and blur the grain out
    acc = acc.reshape(3, H, W)
    acc = acc + np.stack([gaussian_filter(c.reshape(H, W), 8) for c in soft])
    if gate > 0:
        acc = acc + np.stack([gaussian_filter(c.reshape(H, W), 26) for c in haze])
    return acc.astype(np.float32)


EXPOSURE = PchipInterpolator((0, .5, 1, 1.6, 2.2, 2.6, 3, 3.4, 3.8, 4.2, 4.6, 5), (42, 42, 34, 28, 19, 16, 25, 30, 38, 38, 34, 32))


def line(t):
    frame = int(round(float(t) * 30))
    t = frame / 30
    acc = accumulate(t, frame)
    e = float(EXPOSURE(t))
    img = 1 - np.exp(-e * acc ** .9)
    small = img[:, ::2, ::2]
    bloom = np.stack([gaussian_filter(c, 4) * .13 + gaussian_filter(c, 20) * .10 for c in small])
    bloom = bloom.repeat(2, 1).repeat(2, 2)[:, :H, :W]
    img = 1 - (1 - np.clip(img, 0, 1)) * (1 - np.clip(bloom, 0, 1))
    y, x = np.mgrid[0:H, 0:W]
    base = np.array((.012, .018, .040), np.float32)[None, None] * (1.25 - .7 * (((x - W / 2) / W) ** 2 + ((y - H / 2) / H) ** 2) * 2)[..., None]
    out = 1 - (1 - base) * (1 - np.transpose(img, (1, 2, 0)))
    out = np.clip(out, 0, 1) ** .9
    out = out + np.random.default_rng(frame).normal(0, .006, out.shape[:2])[..., None]
    return surface(np.clip(out, 0, 1).astype(np.float32))
