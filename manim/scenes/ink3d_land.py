"""Mountains that have been rained on: the land rises and rivers cut it, the way the real thing got its shape, so every ridge and gully drains into the valley below.

Stream-power erosion, vectorised with numpy so it can run on millions of cells (Braun & Willett's model with a pointer-doubling drainage area).
It starts coarse, and each finer pass cuts smaller gullies into the larger ones. Used by ink3d_qian.py and ink3d_channel.py, in the render environment; Blender loads the result.
"""
import numpy as np
from scipy.ndimage import zoom

NB = ((-1, -1, 1.4142), (-1, 0, 1.0), (-1, 1, 1.4142), (0, -1, 1.0), (0, 1, 1.0), (1, -1, 1.4142), (1, 0, 1.0), (1, 1, 1.4142))


def spectral(n, beta, seed):
    r = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(n)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1
    s = (r.normal(size=f.shape) + 1j * r.normal(size=f.shape)) / f ** (beta / 2)
    s[0, 0] = 0
    v = np.fft.irfft2(s, s=(n, n))
    return ((v - v.mean()) / v.std()).astype(np.float32)


def receivers(h, cell):
    """Each cell's steepest way down: the cell's index, the slope's run, and whether it has no way down at all."""
    n, m = h.shape
    pad = np.pad(h, 1, mode="edge")
    idx = np.arange(n * m, dtype=np.int64).reshape(n, m)
    ipad = np.pad(idx, 1, mode="edge")
    best = np.zeros_like(h)
    rcv = idx.copy()
    run = np.full(h.shape, cell, np.float32)
    low = np.full(h.shape, np.inf, np.float32)
    for dy, dx, d in NB:
        hn = pad[1 + dy:n + 1 + dy, 1 + dx:m + 1 + dx]
        low = np.minimum(low, hn)
        s = (h - hn) / d
        k = s > best
        best[k] = s[k]
        rcv[k] = ipad[1 + dy:n + 1 + dy, 1 + dx:m + 1 + dx][k]
        run[k] = cell * d
    return rcv, run, best <= 0, low


def drainage(rcv):
    """How many cells drain through each cell: pointer doubling, so it takes a handful of passes however long the rivers are."""
    n = rcv.size
    r = rcv.ravel().copy()
    me = np.arange(n, dtype=np.int64)
    a = np.ones(n, np.float64)
    for _ in range(20):
        k = r != me
        if not k.any():
            break
        a += np.bincount(r[k], weights=a[k], minlength=n)
        r = r[r]
    return a


def evolve(h, uplift, base, cell, steps, dt, K=.02, m=.5, diffuse=.12):
    """uplift: how fast each cell rises. base: cells held at their height (the rivers the land drains to, and the edge of the map)."""
    h = h.astype(np.float32)
    held = h[base].copy()
    for _ in range(steps):
        h = h + uplift * dt
        h[base] = held
        rcv, run, pit, low = receivers(h, cell)
        h[pit & ~base] = low[pit & ~base] + 1e-2         # a one-cell pit fills, so water finds its way out
        a = drainage(rcv).reshape(h.shape)
        F = (dt * K * (a * cell * cell) ** m / run).astype(np.float32)
        h = (h + F * h.ravel()[rcv]) / (1 + F)
        pad = np.pad(h, 1, mode="edge")
        lap = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] - 4 * h
        h = h + diffuse * lap
        h[base] = held
    return h


def carve(uplift_fn, base_fn, levels, seed, K=.02):
    """levels: (n, steps, dt, noise) from coarse to fine. uplift_fn(n) and base_fn(n) give the fields at that size; base_fn returns (mask, heights)."""
    h = None
    for i, (n, steps, dt, noise, cell) in enumerate(levels):
        u = uplift_fn(n)
        mask, held = base_fn(n)
        if h is None:
            h = np.zeros((n, n), np.float32)
        else:
            h = zoom(h, n / h.shape[0], order=3)[:n, :n].astype(np.float32)
        h = h + noise * spectral(n, 2.0, seed + i)
        h[mask] = held[mask]
        h = evolve(h, u, mask, cell, steps, dt, K=K)
    return h


def hillshade(h, cell, az=(.5, .6, .62)):
    gy, gx = np.gradient(h, cell)
    n = np.sqrt(gx * gx + gy * gy + 1)
    return np.clip((-gx * az[0] - gy * az[1] + az[2]) / n, 0, 1)
