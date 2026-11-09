"""How one stretch of the ink film becomes the next without a cut. Each takes two frames (uint8, 1080x1920x3) and progress u from 0 to 1:

    water(a, b, u)              a front of water crosses from the right, the way the film reads; what is ahead of it bends, what is behind it is the next place
    ink(a, b, u, source=mask)   ink spreads out of a mask (the printed words on a page, say) until the frame is dark, and the next place clears out of the dark
    surface(a, b, u, down=True) the waterline crosses the frame: going under when down, coming up out of the water when not

    python ink_join.py test A.png B.png OUT.png   # a strip of every join at five moments
"""
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from ink_grade import fft_noise, sm

H, W = 1080, 1920
INK = np.array([18, 16, 15], np.float32)
_grid = np.mgrid[0:H, 0:W].astype(np.float32)
_dist = {}


def _warp(f, dy, dx):
    """The frame sampled at an offset per pixel, only where the offset is not zero."""
    out = f.copy()
    moved = (np.abs(dy) + np.abs(dx)) > 0.05
    if moved.any():
        ys, xs = _grid[0][moved] + dy[moved], _grid[1][moved] + dx[moved]
        for i in range(3):
            out[..., i][moved] = map_coordinates(f[..., i], [ys, xs], order=1, mode="nearest")
    return out


def water(a, b, u, seed=1, width=0.14):
    """The front is a wandering edge. Water ahead of it bends what it passes over, and it leaves the dark line wet paper dries with."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    big, fine = fft_noise(H, W, 3.2, seed), fft_noise(H, W, 2.2, seed + 1)
    place = 1 - _grid[1] / W + 0.075 * big + 0.012 * fine
    front = -0.25 + 1.5 * sm(u)
    d = (front - place) / width                               # above 0 is behind the front
    band = np.exp(-(d / 1.1) ** 2)
    gy, gx = np.gradient(big)
    push = 650 * band
    a2, b2 = _warp(a, gy * push, gx * push), _warp(b, gy * push * 0.6, gx * push * 0.6)
    m = sm((d + 0.06) / 0.12)[..., None]
    out = a2 * (1 - m) + b2 * m
    edge = np.exp(-((d - 0.02) / 0.09) ** 2) * (0.5 + 0.2 * np.clip(fine, -1, 1))
    shine = np.exp(-((d + 0.35) / 0.25) ** 2) * 0.07
    out = (out * (1 + shine[..., None])) * (1 - edge[..., None]) + INK * edge[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _reach(source, seed):
    """How far each pixel is from the source, so that ink arrives there later, with the wander ink has on wet paper."""
    k = (id(source) if source is not None else None, seed)
    if k not in _dist:
        if source is None:
            r = np.hypot(_grid[0] - H / 2, _grid[1] - W / 2)
        else:
            r = distance_transform_edt(source < 0.5).astype(np.float32)
        r = r / max(r.max(), 1)
        _dist[k] = np.sqrt(r) + 0.16 * fft_noise(H, W, 3.4, seed) + 0.012 * fft_noise(H, W, 2.4, seed + 1)
    return _dist[k]


def ink(a, b, u, source=None, seed=5):
    """A band of ink rolls out over the reach: its leading edge turns a to wash, its trailing edge clears to b. The frame is never all dark, and the
    band is a real wash, thick at its edges and thin in the middle."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    r = _reach(source, seed)
    lo, hi = float(r.min()), float(r.max())
    first = lo + (hi - lo + 0.3) * sm(u / 0.74) - 0.15
    second = lo + (hi - lo + 0.3) * sm((u - 0.26) / 0.74) - 0.15
    dark = sm((first - r) / 0.15)[..., None]
    clear = sm((second - r) / 0.18)[..., None]
    mottled = 0.5 + 0.28 * np.clip(fft_noise(H, W, 3.0, seed + 2), -1.6, 1.6)
    thick = np.exp(-((first - r) / 0.08) ** 2) + np.exp(-((r - second) / 0.08) ** 2)
    wash = (0.62 * mottled * (1 - np.clip(thick, 0, 1)))[..., None]
    tone = INK + (np.array([241, 234, 219], np.float32) - INK) * wash
    out = a * (1 - dark) + tone * dark
    out = out * (1 - clear) + b * clear
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def surface(a, b, u, down=True, seed=9):
    """The camera crosses a water surface: the line sweeps the frame with a swell on it, a bright lip along it, and the far side bends near the line."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    x = _grid[1][0] / W
    swell = fft_noise(1, W, 2.4, seed)[0]
    v = sm(u)
    line = (1.12 - 1.24 * v) * H if down else (-0.12 + 1.24 * v) * H        # going under, the water rises up the frame; coming out, it drops away
    line = line + 26 * swell + 14 * np.sin(x * 21 + u * 9)
    d = (_grid[0] - line[None, :]) / 60.0                                   # above 0 is below the line
    wet, dry = (b, a) if down else (a, b)                                   # under the line is the side that is under water
    near = np.exp(-(d / 2.0) ** 2)
    gy, gx = np.gradient(fft_noise(H, W, 2.8, seed + 1))
    wet2 = _warp(wet, gy * 420 * near, gx * 420 * near)
    m = sm(d / 0.08 + 0.5)[..., None]
    out = dry * (1 - m) + wet2 * m
    lip = (np.exp(-(d / 0.05) ** 2) * (0.55 + 0.45 * np.clip(fft_noise(H, W, 1.6, seed + 2), -1, 1)))[..., None]
    dark = np.exp(-((d - 0.5) / 0.45) ** 2)[..., None] * 0.28 * (d[..., None] > 0)
    out = out * (1 - dark) + INK * dark
    out = out * (1 - lip * 0.5) + np.array([241, 234, 219], np.float32) * lip * 0.5
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def main():
    if len(sys.argv) != 5 or sys.argv[1] != "test":
        raise SystemExit(__doc__)
    from PIL import Image
    a, b = (np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)) for p in sys.argv[2:4])
    src = np.zeros((H, W), np.float32)
    src[250:830, 1500:1560] = 1
    rows = []
    for fn in (water, lambda p, q, u: ink(p, q, u, source=src), surface, lambda p, q, u: surface(p, q, u, down=False)):
        row = [np.asarray(Image.fromarray(fn(a, b, u)).resize((384, 216), Image.LANCZOS)) for u in (0.12, 0.32, 0.5, 0.68, 0.88)]
        rows.append(np.hstack(row))
    Image.fromarray(np.vstack(rows)).save(sys.argv[4])


if __name__ == "__main__":
    main()
