"""Twelve pair-specific RGB transitions, each made of its own clip's material, without blank/flash frames.

Each transition occupies 12 frames at 30 fps; sample p = frame / 11.
Inputs are float32 RGB in [0, 1]. Cached fields never retain input frames.
"""
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

W, H = 1920, 1080
TRANSITION_FRAMES = 12


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _edge(d, width=1.5):
    return _smooth(0.5 + d / width)


def _mix(a, b, mask):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[..., None]
    return a + (b - a) * mask


def _tint(out, color, mask):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[..., None]
    out *= 1.0 - mask
    out += np.asarray(color, dtype=np.float32) * mask
    return out


def _finish(out):
    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32, copy=False)


@lru_cache(maxsize=4)
def _grid(h=H, w=W):
    return np.arange(w, dtype=np.float32)[None, :], np.arange(h, dtype=np.float32)[:, None]


@lru_cache(maxsize=4)
def _resize_coordinates(h, w):
    coords = np.empty((2, H, W), dtype=np.float32)
    x, y = _grid()
    coords[0] = (y + 0.5) * (h / H) - 0.5
    coords[1] = (x + 0.5) * (w / W) - 0.5
    return coords


def _enlarge(small):
    coords = _resize_coordinates(*small.shape[:2])
    if small.ndim == 2:
        return map_coordinates(small, coords, order=1, mode="nearest", prefilter=False)
    out = np.empty((H, W, 3), dtype=np.float32)
    for c in range(3):
        map_coordinates(small[..., c], coords, output=out[..., c],
                        order=1, mode="nearest", prefilter=False)
    return out


@lru_cache(maxsize=12)
def _noise(h, w, seed, sigma):
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((h, w), dtype=np.float32)
    field = gaussian_filter(field, sigma, mode="reflect")
    field -= field.mean()
    field /= max(float(field.std()), 1e-6)
    return field


@lru_cache(maxsize=1)
def _paper_seam():
    _, y = _grid()
    return W * 0.5 + 0.075 * (y - H * 0.5) + 1.8 * np.sin(y * 0.047)


def crossfade(a, b, p):
    """Neutral fallback for non-adjacent pairs when clips are missing."""
    if p <= 0.0:
        return a.copy()
    if p >= 1.0:
        return b.copy()
    return _finish(_mix(a, b, float(_smooth(p))))


def _reveal(a, b, p, field, feather=0.025):
    """An ordered matte in [0,1], always revealing actual footage."""
    if p <= 0.0:
        return a.copy()
    if p >= 1.0:
        return b.copy()
    threshold = (1 + 2 * feather) * float(_smooth(p)) - feather
    return _finish(_mix(a, b, _edge(threshold - field, feather)))


def pencil_split(a, b, p):
    """A quiet paper seam opens outward, without moving the source image."""
    x, _ = _grid()
    field = np.abs(x - _paper_seam()) / (W * 0.5 + 85)
    return _reveal(a, b, p, field, 0.012)


def isometric_tile_flip(a, b, p):
    """One isometric diagonal, not a field of tumbling tiles."""
    x, y = _grid()
    return _reveal(a, b, p, (x + (H - 1 - y) * 1.73) / (W + H * 1.73), 0.035)


@lru_cache(maxsize=1)
def _earth_radius():
    x, y = _grid()
    radius = np.hypot(x - 1190.0, y - 565.0)
    return radius, float(radius.max())


def earth_iris(a, b, p):
    """A soft aperture from the rocket's field into the light drawing."""
    radius, maximum = _earth_radius()
    return _reveal(a, b, p, radius / maximum, 0.055)


def luminous_line_sweep(a, b, p):
    """A narrow left-to-right sweep; no artificial luminous outline."""
    x, _ = _grid()
    return _reveal(a, b, p, x / (W - 1), 0.018)


@lru_cache(maxsize=1)
def _ink_arrival():
    x, y = _grid(H // 4, W // 4)
    x, y = x * 4.0, y * 4.0
    field = np.full((H // 4, W // 4), np.inf, dtype=np.float32)
    for cx, cy, speed in ((375, 330, 1.0), (940, 720, 1.12), (1430, 410, 0.9),
                          (1680, 850, 0.85), (790, 190, 0.78)):
        np.minimum(field, np.hypot(x - cx, y - cy) / speed, out=field)
    field += 38.0 * _noise(H // 4, W // 4, 4101, 5.0)
    field += 9.0 * _noise(H // 4, W // 4, 4102, 0.9)
    field = _enlarge(field)
    field -= field.min()
    field /= field.max()
    return field


def ink_bleed(a, b, p):
    """Let several ragged ink blooms spread and uncover the morph's paper."""
    if p <= 0.0:
        return a.copy()
    if p >= 1.0:
        return b.copy()
    d = float(_smooth(p)) * 1.12 - 0.06 - _ink_arrival()
    out = _mix(a, b, _edge(d, 0.023))
    return _finish(out)


def _travel(p):
    """Most of a transition's path at an even pace, so it is on screen for nearly all twelve frames."""
    return 0.45 * p + 0.55 * float(_smooth(p))


def _travel(p):
    """Most of a transition's path at an even pace, so it is on screen for nearly all twelve frames."""
    return 0.45 * p + 0.55 * float(_smooth(p))


def _rounded_box(u, v, half_u, half_v, radius):
    """Signed distance to a rounded rectangle, negative inside."""
    du, dv = np.abs(u) - half_u + radius, np.abs(v) - half_v + radius
    return np.hypot(np.maximum(du, 0.0), np.maximum(dv, 0.0)) + np.minimum(np.maximum(du, dv), 0.0) - radius


def sticker_pop(a, b, p):
    """The next picture lands as a die-cut sticker: a thin white cut line, a contact shadow, and a little overshoot as it settles."""
    if p <= 0.0:
        return a.copy()
    if p >= 1.0:
        return b.copy()
    t = min(p / 0.95, 1.0)
    scale = 1.0 + 1.9 * (t - 1.0) ** 3 + 0.9 * (t - 1.0) ** 2            # ease out, past 1 and back
    grown = min(scale, 1.0)
    turn = np.deg2rad(-5.0) * (1.0 - grown) ** 1.2
    cos_t, sin_t = float(np.cos(turn)), float(np.sin(turn))
    x, y = _grid()
    dx, dy = x - W * 0.5, y - H * 0.5
    u, v = cos_t * dx + sin_t * dy, cos_t * dy - sin_t * dx
    radius = 46.0 * grown * (1.0 - t * t)
    half_u, half_v = W * 0.5 * scale, H * 0.5 * scale
    inside = _edge(-_rounded_box(u, v, half_u, half_v, radius), 1.6)
    cut = 13.0 * grown
    border = _edge(-_rounded_box(u, v, half_u + cut, half_v + cut, radius + cut), 1.6)
    # Contact shadow, thrown down and to the right of the sticker's edge.
    shadow = _smooth(1.0 - (_rounded_box(u - 8.0, v - 15.0, half_u + cut, half_v + cut, radius + cut) + 8.0) / 42.0) * 0.4
    out = a * (1.0 - shadow[..., None])
    _tint(out, (0.985, 0.98, 0.965), border)
    coords = np.stack((v / scale + H * 0.5, u / scale + W * 0.5))
    picture = np.empty_like(b)
    for c in range(3):
        map_coordinates(b[..., c], coords, output=picture[..., c], order=1, mode="nearest", prefilter=False)
    sheen = np.exp(-(((u + 0.55 * v) / W - 0.9 + 1.5 * t) / 0.07) ** 2) * 0.15 * (1.0 - t)     # vinyl catching the light
    picture += sheen[..., None] * (1.0 - picture)
    return _finish(_mix(out, picture, inside))


@lru_cache(maxsize=1)
def _peel_axes():
    x, y = _grid()
    cos_t, sin_t = float(np.cos(np.deg2rad(32.0))), float(np.sin(np.deg2rad(32.0)))
    across = cos_t * y - sin_t * x
    across = (across - float(across.mean())) / (0.5 * float(np.ptp(across)))
    along = cos_t * x + sin_t * y + 46.0 * across * across                # the fold bows, leading at its middle
    return along, float(along.max()), cos_t, sin_t


def diagonal_sticker_peel(a, b, p):
    """The sheet rolls off from the lower right: its pale liner curls over the fold, with the picture showing faintly through."""
    if p <= 0.0:
        return a.copy()
    if p >= 1.0:
        return b.copy()
    roll = 210.0
    along, top, cos_t, sin_t = _peel_axes()
    front = top + 30.0 - (top + 2.0 * roll + 60.0) * _travel(p)
    width = roll * float(_smooth((top - front) / (roll * 1.3)))
    past = along - front                                                 # > 0 where the sheet has already gone
    out = _mix(a, b, _edge(past, 1.6))
    out *= (1.0 - (np.exp(-np.maximum(past, 0.0) / 75.0) * _edge(past, 2.0) * 0.55))[..., None]
    if width < 1.0:
        return _finish(out)
    ahead = -past - width                                                # > 0 on the sheet still lying flat
    out *= (1.0 - (np.exp(-np.maximum(ahead, 0.0) / 26.0) * _edge(ahead, 2.0) * 0.3))[..., None]
    u = np.clip(-past / width, 0.0, 1.0)
    light = 0.46 + 0.5 * np.sin(np.pi * u ** 0.72)
    light += 0.09 * np.exp(-((u - 0.6) / 0.05) ** 2)                     # the roll's highlight
    x, y = _grid()
    fold_x = np.clip(x - 2.0 * past * cos_t, 0, W - 1).astype(np.int32)
    fold_y = np.clip(y - 2.0 * past * sin_t, 0, H - 1).astype(np.int32)
    liner = np.asarray((0.93, 0.905, 0.85), np.float32) * 0.87 + a[fold_y, fold_x] * 0.13
    return _finish(_mix(out, liner * light[..., None], _edge(-past, 1.6) * _edge(-ahead, 1.6)))


@lru_cache(maxsize=1)
def _signal_bands():
    """Uneven horizontal slices: when each one drops, and how far it slides."""
    rng = np.random.default_rng(7311)
    edges = [0]
    while edges[-1] < H:
        edges.append(edges[-1] + int(rng.choice((6, 14, 22, 40, 64, 96), p=(0.12, 0.2, 0.24, 0.22, 0.14, 0.08))))
    edges[-1] = H
    spans = list(zip(edges, edges[1:]))
    bands, dropped = [], 0
    for i in rng.permutation(len(spans)):                                # in random order, but an even share of the frame per frame
        top, bottom = spans[i]
        bands.append((top, bottom, 0.04 + 0.58 * dropped / H, float(rng.choice((-1.0, 1.0)) * rng.uniform(70.0, 380.0))))
        dropped += bottom - top
    return bands


def archival_glitch(a, b, p):
    """The signal loses its hold slice by slice; each slice slips, then locks into the printed page."""
    if p <= 0.0:
        return a.copy()
    if p >= 1.0:
        return b.copy()
    out = np.empty_like(a)
    for top, bottom, delay, slip in _signal_bands():
        q = float(np.clip((p - delay) / 0.34, 0.0, 1.0))
        if q > 0.0:
            source, off = b, slip * (1.0 - q) ** 3
        else:
            source, off = a, -slip * 0.2 * float(_smooth(p / max(delay, 0.01)))
        shift, fringe = int(round(off)), int(min(6.0, abs(off) * 0.04))
        band = source[top:bottom]
        for c, d in enumerate((fringe, 0, -fringe)):
            out[top:bottom, :, c] = np.roll(band[..., c], shift + d, axis=1)
        if shift:                                                        # a moving slice lifts a touch, and its top edge catches
            out[top:bottom] *= 1.0 + 0.04 * min(abs(off) / 80.0, 1.0)
            out[top] += (np.asarray((0.55, 0.9, 1.0), np.float32) - out[top]) * (0.28 * min(abs(off) / 80.0, 1.0))
    return _finish(out)


@lru_cache(maxsize=1)
def _tear_field():
    """Where the sweep reaches each point, with notches at three scales, and how wide the torn white core runs."""
    x, y = _grid()
    line = x + 0.16 * (y - H * 0.5) + 50.0 * _enlarge(_noise(H // 8, W // 8, 9104, 4.5)) + 14.0 * _enlarge(_noise(H // 4, W // 4, 9105, 2.2))
    field = line + 2.2 * _noise(H, W, 9106, 1.4)
    core = np.clip(8.0 + 7.0 * _enlarge(_noise(H // 8, W // 8, 9108, 2.5)), 2.5, 22.0)
    return line, field, float(field.min()), float(field.max()), core, _noise(H, W, 9107, 1.2)


def newsprint_tear(a, b, p):
    """The page rips across: a fibrous white core along the tear, its lifted edge shading what lies beneath."""
    if p <= 0.0:
        return a.copy()
    if p >= 1.0:
        return b.copy()
    line, field, low, high, core, fibre = _tear_field()
    front = low - 34.0 + (high - low + 68.0) * _travel(p)
    d = front - field                                                    # > 0 where it has torn away
    pull = int(26.0 * _travel(p))                                 # what's left is being pulled away from the tear
    out = _mix(a if not pull else np.concatenate((a[:, :1].repeat(pull, axis=1), a[:, :-pull]), axis=1), b, _edge(d, 1.4))
    out *= (1.0 - np.exp(-np.maximum(front - line, 0.0) / 17.0) * _edge(d, 1.6) * 0.36)[..., None]
    reach = core + 2.0 * fibre
    rim = _edge(d + reach, 1.2) * _edge(-d, 1.2)
    white = 0.955 + 0.03 * fibre - 0.09 * np.clip(-d / np.maximum(reach, 1.0), 0.0, 1.0)
    paper = np.stack((white, white * 0.975, white * 0.915), axis=-1)
    return _finish(_mix(out, paper, rim))


def _quarter_blur(frame, sigma):
    small = frame.reshape(H // 4, 4, W // 4, 4, 3).mean(axis=(1, 3), dtype=np.float32)
    small = gaussian_filter(small, (sigma / 4.0, sigma / 4.0, 0.0), mode="reflect", truncate=3.0)
    blurred = _enlarge(small)
    if sigma < 5.0:
        return _mix(frame, blurred, float(_smooth(sigma / 5.0)))
    return blurred


def frosted_focus_cross(a, b, p):
    """A sheet of the clip's own frosted glass crosses the frame, and the picture changes behind it."""
    if p <= 0.0:
        return a.copy()
    if p >= 1.0:
        return b.copy()
    half, margin = W * 0.23, 70.0
    x, y = _grid()
    lean = x + 0.07 * (y - H * 0.5)
    d = lean - (W + 0.035 * H + half + margin - (W + 0.07 * H + 2.0 * half + 2.0 * margin) * _travel(p))
    away = np.abs(d)
    behind = _edge(d, 1.0)
    out = _mix(a, b, behind)
    out *= (1.0 - np.exp(-np.maximum(away - half, 0.0) / 48.0) * 0.26)[..., None]
    frost = _mix(_quarter_blur(a, 28.0), _quarter_blur(b, 28.0), _smooth(0.5 + d / (half * 0.9)))
    grey = frost.mean(axis=2, keepdims=True)
    frost = np.clip(grey + (frost - grey) * 1.18, 0.0, 1.0)                # frosted glass keeps colours rich
    glow = 0.06 + 0.1 * np.exp(-np.maximum(half - away, 0.0) / 16.0)
    frost += (glow + 0.012 * _noise(H, W, 5151, 0.6))[..., None] * (1.0 - frost)
    out = _mix(out, frost, _edge(half - away, 1.6))
    return _finish(_tint(out, (1.0, 1.0, 1.0), np.exp(-((away - half) / 1.5) ** 2) * 0.5))


_MOSAIC = (1, 4, 6, 10, 15, 24, 30, 24, 15, 10, 6, 1)                     # each divides 1920 and 1080


@lru_cache(maxsize=1)
def _flip_field():
    """When each cell turns over: out from about where the two suns sit, with a little disorder cell by cell."""
    x, y = _grid()
    radius = np.hypot(x - W * 0.52, (y - H * 0.28) * 1.2)
    rng = np.random.default_rng(6120)
    speckle = rng.random((H // 30 + 1, W // 30 + 1), dtype=np.float32)
    speckle = speckle.repeat(30, axis=0)[:H].repeat(30, axis=1)[:, :W]
    field = 0.8 * radius / float(radius.max()) + 0.2 * speckle
    return (field - float(field.min())) / float(np.ptp(field))


def _mosaic(frame, size):
    if size == 1:
        return frame
    cells = frame.reshape(H // size, size, W // size, size, 3).mean(axis=(1, 3), dtype=np.float32)
    return cells.repeat(size, axis=0).repeat(size, axis=1)


def crt_to_pixels(a, b, p):
    """The picture coarsens into blocks, the blocks turn over from the sun outward, and it resolves as pixel art."""
    if p <= 0.0:
        return a.copy()
    if p >= 1.0:
        return b.copy()
    size = _MOSAIC[int(round(p * (TRANSITION_FRAMES - 1)))]
    cells = _flip_field()[size // 2::size, size // 2::size] < float(np.clip((p - 0.12) / 0.76, 0.0, 1.0))
    flip = cells.repeat(size, axis=0).repeat(size, axis=1)[..., None]
    return _finish(np.where(flip, _mosaic(b, size), _mosaic(a, size)))


def metallic_liquid_rise(a, b, p):
    """Liquid gold floods up from below: a bright rolled lip, and the old picture mirrored in its surface."""
    if p <= 0.0:
        return a.copy()
    if p >= 1.0:
        return b.copy()
    x, y = _grid()
    swell = float(np.sin(np.pi * p))
    level = H + 90.0 - (H + 180.0) * _travel(p)
    surface = level + swell * (26.0 * np.sin(x / 215.0 + 5.5 * p) + 13.0 * np.sin(x / 84.0 - 8.0 * p)
                               - 44.0 * np.exp(-((x - 300.0 - 1300.0 * p) / 340.0) ** 2))
    d = y - surface                                                      # > 0 inside the liquid
    out = _mix(a, b, _edge(d, 1.6))
    out *= (1.0 - np.exp(-np.maximum(-d, 0.0) / 24.0) * _edge(-d, 2.0) * 0.28)[..., None]     # the picture darkens where it meets the metal
    depth = np.maximum(d, 0.0)
    within = _edge(d, 1.6)
    out *= (1.0 - 0.6 * np.exp(-((depth - 17.0) / 11.0) ** 2) * within)[..., None]            # under the lip
    mirror = np.clip(surface - 1.3 * depth, 0, H - 1).astype(np.int32)
    columns = np.broadcast_to(np.arange(W), mirror.shape)
    gold = np.asarray((1.0, 0.8, 0.45), np.float32)
    out += a[mirror, columns] * gold * (np.exp(-depth / 42.0) * 0.8 * within)[..., None]
    glint = 0.62 + 0.38 * np.sin(x / 61.0 + 9.0 * p + np.sin(x / 23.0))
    lip = np.exp(-((depth - 3.5) / 3.2) ** 2) * within * glint
    return _finish(_tint(out, (1.0, 0.89, 0.62), lip * 0.92))


TRANSITIONS = [
    ("handdrawn", "port", pencil_split),
    ("port", "flat", isometric_tile_flip),
    ("flat", "line", earth_iris),
    ("line", "letterpress", luminous_line_sweep),
    ("letterpress", "morph", ink_bleed),
    ("morph", "sticker", sticker_pop),
    ("sticker", "cyberhud", diagonal_sticker_peel),
    ("cyberhud", "collage", archival_glitch),
    ("collage", "glass", newsprint_tear),
    ("glass", "synthwave", frosted_focus_cross),
    ("synthwave", "pixel", crt_to_pixels),
    ("pixel", "splash", metallic_liquid_rise),
]
