"""What the events layer shares: the stills in their colour, where a shot's frame sits on its still, filling a hole in a picture with its own texture, and rings on water.

Nothing here runs at import: ink_cut imports ink_events, which imports this, so ink_cut itself is only fetched when a still is first needed.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, label

W, H = 1920, 1080
_Y, _X = np.mgrid[0:H, 0:W].astype(np.float32)
_shots = {}
_plates = {}


def shot(still, k=0):
    """The k-th use of a still in the cut, with its end."""
    from ink_cut import END, SHOTS
    hits = [i for i, s in enumerate(SHOTS) if s["still"] == still]
    i = hits[k]
    return SHOTS[i], (SHOTS[i + 1]["start"] if i + 1 < len(SHOTS) else END)


def plate(still, palette=None):
    """A still in its colour, float32 RGB in 0-255, as the cut sees it. Kept: every layer starts from these."""
    key = (still, palette)
    if key not in _plates:
        from ink_cut import SHOTS, picture
        s = next(x for x in SHOTS if x["still"] == still)
        s = dict(s, palette=palette or s["palette"])
        _plates[key] = np.asarray(picture(s), np.float32).copy()
    return _plates[key]


def box(z, c, t, dur):
    """The part of the still a shot with no move shows at t, exactly as ink_cut.framed works it out."""
    p = t / max(dur, 1e-3)
    base = max(z[0] + (z[1] - z[0]) * p, 1.0)
    w, h = W / base, H / base
    cx, cy = c[0] * W, c[1] * H
    x0, y0 = min(max(cx - w / 2, 0), W - w), min(max(cy - h / 2, 0), H - h)
    return x0, y0, x0 + w, y0 + h


_depth = {}


def region_from(still, t, dur, region, k=0):
    """For a shot that moves through its depth, where the frame's pixels in region (y0, y1, x0, x1) come from on the still, water aside. As ink_cut.framed and ink_move.Move.frame work it out."""
    from ink_cut import mover
    s = shot(still, k)[0]
    if still not in _depth:
        mv = mover(s)
        _depth[still] = (mv.w, mv.cx, mv.cy)
    w, cx, cy = _depth[still]
    z0, z1 = s["z"]
    base = min(z0, z1)
    p = t / max(dur, 1e-3)
    push = max(z0 + (z1 - z0) * p - base, 0.0)
    y0, y1, x0, x1 = region
    bw, bh = W / base, H / base
    ax, ay = min(max(s["c"][0] * W - bw / 2, 0), W - bw), min(max(s["c"][1] * H - bh / 2, 0), H - bh)
    fx, fy = ax + _X[y0:y1, x0:x1] / base, ay + _Y[y0:y1, x0:x1] / base
    zoom = 1 + push * w[np.clip(fy, 0, H - 1).astype(np.int32), np.clip(fx, 0, W - 1).astype(np.int32)]
    return cy + (fy - cy) / zoom, cx + (fx - cx) / zoom


def to8(a):
    return np.clip(a + 0.5, 0, 255).astype(np.uint8)


def crop8(a, b):
    """A float or uint8 RGB picture through a box, as a uint8 frame."""
    im = Image.fromarray(a if a.dtype == np.uint8 else to8(a))
    return np.asarray(im.resize((W, H), Image.BILINEAR, box=b))


def crop1(a, b):
    """A float mask through a box."""
    return np.asarray(Image.fromarray(a).resize((W, H), Image.BILINEAR, box=b))


def up1(a, size=(W, H)):
    """A small float field brought up to frame size."""
    return np.asarray(Image.fromarray(a).resize(size, Image.BILINEAR))


def fill(f, hole, sigma=26.0, offset=(0, 0), feather=2.5, fine=None):
    """f with the hole (bool) painted out from what surrounds it: the broad tone from around the hole, the fine texture lifted from another part of the same picture.
    offset: (dy, dx) where the texture comes from, relative to each pixel; it must land on clean picture. fine: how wide a blur that texture is measured against, sigma if not given:
    small takes only the grain, so that out-of-focus writing elsewhere does not come along with it. Returns a float32 copy."""
    f = f.astype(np.float32)
    hole = hole.astype(bool)
    keep = (~hole).astype(np.float32)
    broad = np.empty_like(f)
    w = gaussian_filter(keep, sigma) + 1e-3
    # far from the edge of a big hole there is nothing near: a wider blur reaches in
    w2 = gaussian_filter(keep, sigma * 4) + 1e-3
    near = np.clip(w / 0.25, 0, 1)
    for c in range(3):
        a, b = gaussian_filter(f[..., c] * keep, sigma) / w, gaussian_filter(f[..., c] * keep, sigma * 4) / w2
        broad[..., c] = a * near + b * (1 - near)
    detail = f - np.stack([gaussian_filter(f[..., c], fine or sigma) for c in range(3)], -1)
    dy, dx = offset
    ys, xs = np.clip(_Y.astype(np.int32) + dy, 0, H - 1), np.clip(_X.astype(np.int32) + dx, 0, W - 1)
    tex = detail[ys, xs]
    m = gaussian_filter(hole.astype(np.float32), feather)
    m = np.clip(m * 2.0, 0, 1)[..., None]
    return f * (1 - m) + (broad + tex) * m


def membrane(f, hole, feel=0.0):
    """What the picture's colours do across a hole if they just carry on from its rim: a Laplace fill, coarse to fine, so a big hole takes up each side's colour rather than one grey. feel: a blur on f first, so the rim's fine marks are not drawn inward. Returns the fill, float32, at full size, f outside the hole."""
    f = f.astype(np.float32)
    g = np.stack([gaussian_filter(f[..., c], feel) for c in range(3)], -1) if feel else f
    keep = (~hole.astype(bool)).astype(np.float32)
    u = None
    for scale in (32, 16, 8, 4):
        w, h = W // scale, H // scale
        k = np.asarray(Image.fromarray(keep).resize((w, h), Image.BOX))
        known = k > 0.9
        ch = [np.asarray(Image.fromarray(g[..., c] * keep).resize((w, h), Image.BOX)) / np.maximum(k, 1e-3) for c in range(3)]
        small = np.stack(ch, -1)
        if u is None:
            u = np.where(known[..., None], small, small[known].mean(0))
        else:
            u = np.stack([np.asarray(Image.fromarray(u[..., c]).resize((w, h), Image.BILINEAR)) for c in range(3)], -1)
        for _ in range(240):
            v = (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1)) * 0.25
            u = np.where(known[..., None], small, v)
    return np.stack([np.asarray(Image.fromarray(u[..., c].astype(np.float32)).resize((W, H), Image.BILINEAR)) for c in range(3)], -1)


def reach(mask, seeds):
    """Steps along the mask (bool) from the seeds (bool), a slow but plain flood; every other pixel is given its nearest mask pixel's value, plus how far off it is."""
    out = np.full(mask.shape, np.inf, np.float32)
    front = seeds & mask
    out[front] = 0
    d = 0
    from scipy.ndimage import binary_dilation
    done = front.copy()
    st = np.ones((3, 3), bool)
    while front.any():
        d += 1
        nxt = binary_dilation(front, st) & mask & ~done
        out[nxt] = d
        done |= nxt
        front = nxt
    return out


def stroke_order(ink, seed_noise=None, gap=14.0):
    """The order a character's strokes come in: pieces of the character from the top down, each flooding from its own top, the next starting a little after the last. ink: bool. Returns a float map,
    0 at the first stroke's start, with the rest of the cell filled by the nearest ink's value plus its distance."""
    lab, n = label(ink, np.ones((3, 3), bool))
    order = np.full(ink.shape, np.inf, np.float32)
    tops = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if ys.size == 0:
            continue
        tops.append((ys.min() + 0.25 * xs.min(), i))
    t = 0.0
    for _, i in sorted(tops):
        m = lab == i
        ys, xs = np.nonzero(m)
        k = ys.argmin()
        s = np.zeros(ink.shape, bool)
        s[ys[k], xs[k]] = True
        r = reach(m, s)
        good = np.isfinite(r)
        order[good] = r[good] + t
        t += float(r[good].max()) * 0.55 + gap
    (dist, (iy, ix)) = distance_transform_edt(~ink, return_indices=True)
    if not ink.any():
        return np.zeros(ink.shape, np.float32)
    full = order[iy, ix] + dist.astype(np.float32) * 1.5
    return full


class Rings:
    """Rings on a surface, drawn on a small grid: each is an ellipse in the frame, flattened by its squash. Add up what the live ones do, and bring the result up. Cheap enough for dozens at once."""

    def __init__(self, scale=4, size=(W, H)):
        self.s = scale
        self.size = size
        h, w = size[1] // scale, size[0] // scale
        self.y, self.x = np.mgrid[0:h, 0:w].astype(np.float32) * scale

    def field(self, rings, t, speed=95.0, wave=13.0, life=2.2, amp=1.0):
        """rings: (t0, x, y, squash, size). Returns height and the small grid's light, both in small grids."""
        h = np.zeros(self.x.shape, np.float32)
        for t0, x, y, squash, size in rings:
            a = t - t0
            if a <= 0 or a > life:
                continue
            r = np.hypot(self.x - x, (self.y - y) / squash)
            front = speed * size * a ** 0.75
            width = (20 + 34 * a) * size
            env = np.exp(-((r - front) / width) ** 2) * np.exp(-a / (life * 0.42)) * np.clip(r / (6 * size), 0, 1)
            h += amp * size ** 0.5 * np.sin((r - front) / (wave * size * 0.32)) * env
        return h

    def warp(self, h, gain=7.0, lit=0.2):
        """dx, dy and the light from a height field, at full size."""
        gy, gx = np.gradient(h)
        dx, dy = up1(gx * gain, self.size), up1(gy * gain, self.size)
        return dx, dy, up1(h * lit, self.size)


def sample3(f, ys, xs):
    from ink_move import sample
    return sample(f, ys, xs)
