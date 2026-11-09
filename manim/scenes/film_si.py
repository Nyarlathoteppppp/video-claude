"""六's 「书家说」: a real 思, written by 張照 (清), the first character of his couplet 「思樂清涼界」 (National Palace Museum, 故書000536, open data, CC BY 4.0). The scan stays where Opus put it on the Studio and is never copied into the repo; only frames made from it are written.

The cut comes in from 六's window while the orange arc of attention is still rising from 低 towards 思, up and to the right. It lands on the couplet, wide, with the arc's tip on the start of the dry hairs (飞白) that leave the 心's middle dot going up and right into the right dot, so the streak carries the arc's line on. The character is shown finished (REVEAL 0): writing that last stroke along its own path is kept in source() but off. Then the camera goes in at one pace until the scan is shown one pixel to one (it is never enlarged), 思 nearly filling the frame.

    python film_si.py still            # the cut frame beside the window's last frame, and the 1:1 frame: ink/film_out/film/learn/stills/si_*.png
    python film_si.py geometry         # where the stroke starts in the frame at the cut, and which way it goes

The real 思 sits near the right edge of the scan, so the push goes in about a point on the frame's right edge: that keeps every frame inside the scan.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.expanduser("~/render_work/scans/npm_zhangzhao/K2B000536N000000000PAE.jpg")
W, H = 1920, 1080
CREDIT = "張照《書聯》（清）· 國立故宮博物院 故書000536 · CC BY 4.0"

# the push, in scan pixels per frame pixel's inverse: frame = (scan - origin) * s
S0, S1, SECS = 0.63, 1.0, 2.5           # from the couplet to one to one, in the 2.5 s of the shot
FIX = (1920.0, 507.0)                    # the frame point that stays where it is
ORIGIN1 = (1135.0, 470.0)                # the last frame's top left in the scan: 思 in the middle of what the scan allows, its foot clear of the subtitle

# the last stroke: from the middle dot, the dry hairs up and right, into the right dot. In scan pixels.
LAST = ((2179, 1169), (2196, 1122), (2215, 1107), (2270, 1097), (2300, 1082), (2350, 1070), (2450, 1070), (2453, 1195), (2360, 1192), (2280, 1170))
PATH = ((2172, 1166), (2228, 1131), (2276, 1110), (2330, 1110), (2445, 1130))    # the way the brush went through it
START = PATH[0]
REACH = ((0, 20), (60, 26), (110, 30), (150, 48), (240, 60), (330, 52))    # how wide the brush is along its path, half, scan pixels: the dry hairs, then pressed into the dot
SILK_FROM = (0, 360)                     # silk with nothing on it, straight below the stroke between 思 and 樂: what is under the stroke before it is written
REVEAL = 0                               # how long the last stroke takes to write, seconds; 0: shown finished, the push only. At 0.42 the swept disc left a round stamped front on the dry hairs (Opus's test, 2026-11-07), so it is off
FRONT = 5.0                              # how soft the brush's leading edge is, scan pixels: ink edges are sharp


def rate():
    return math.log(S1 / S0) / SECS


def place(t):
    """Scale and top left in the scan, t seconds into the shot."""
    s = S0 * math.exp(rate() * min(max(t, 0.0), SECS))
    fx, fy = FIX[0] / S1 + ORIGIN1[0], FIX[1] / S1 + ORIGIN1[1]       # the scan point under FIX
    return s, (fx - FIX[0] / s, fy - FIX[1] / s)


def to_frame(p, t):
    s, (ox, oy) = place(t)
    return (p[0] - ox) * s, (p[1] - oy) * s


def direction():
    """Which way the stroke leaves the middle dot: degrees above the horizontal."""
    (x0, y0), (x1, y1) = PATH[0], PATH[1]
    return math.degrees(math.atan2(y0 - y1, x1 - x0))


_cache = {}


def _scan():
    if "scan" not in _cache:
        _cache["scan"] = np.asarray(Image.open(SCAN).convert("RGB"), np.float32)
    return _cache["scan"]


def _smooth(a, sigma):
    from scipy.ndimage import gaussian_filter
    a = np.asarray(a, np.float32)
    return gaussian_filter(a, (sigma, sigma) + (0,) * (a.ndim - 2))


def _stroke():
    """The last stroke's ink, the silk to put in its place, and how far along the brush's path each of its points is: all over its box."""
    if "stroke" in _cache:
        return _cache["stroke"]
    a = _scan()
    xs, ys = [p[0] for p in LAST], [p[1] for p in LAST]
    m = 40
    x0, y0, x1, y1 = min(xs) - m, min(ys) - m, max(xs) + m, max(ys) + m
    box = a[y0:y1, x0:x1]
    region = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(region).polygon([(x - x0, y - y0) for x, y in LAST], fill=255)
    region = np.asarray(region.filter(ImageFilter.GaussianBlur(1.0)), np.float32) / 255
    lum = box @ np.array((0.299, 0.587, 0.114), np.float32)
    clear = (region < 0.02) & (lum > np.percentile(lum, 55))
    silk_l, ink_l = float(np.median(lum[clear])), float(np.percentile(lum[region > 0.5], 3))
    # every trace of the stroke, the faintest dry hair too
    ink = np.clip((silk_l - 6 - lum) / (silk_l - ink_l) * 3.0, 0, 1)
    ink = np.asarray(Image.fromarray(np.uint8(ink * 255)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.5)), np.float32) / 255
    hide = np.clip(ink * 1.3, 0, 1) * region
    # the silk under it: the weave of a bare piece of the same silk, on the light of the silk round the stroke carried in under it
    dx, dy = SILK_FROM
    patch = a[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    bare = 1 - _smooth(np.clip(ink + region, 0, 1), 3.0) * (ink > 0.02) - region
    bare = np.clip(bare, 0, 1) * (lum > silk_l - 30)
    light = _smooth(box * bare[..., None], 22) / np.maximum(_smooth(bare, 22), 1e-3)[..., None]
    silk = patch - _smooth(patch, 22) + light
    # when the brush gets to each point: it goes along its path as a disc as wide as the stroke is there, and a point is written when the disc first covers it
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    pts, run = [], 0.0
    for (ax, ay), (bx, by) in zip(PATH, PATH[1:]):
        L = math.hypot(bx - ax, by - ay)
        for u in np.arange(0, L, 2.0):
            pts.append((ax + (bx - ax) * u / L, ay + (by - ay) * u / L, run + u))
        run += L
    along = np.full(xx.shape, run + FRONT, np.float32)
    for px_, py_, p in pts:
        r = float(np.interp(p, *zip(*REACH)))
        inside = (xx - px_) ** 2 + (yy - py_) ** 2 <= r * r
        along = np.where(inside & (along > p), p, along)
    # give or take a hair: the bristles do not all reach a place at once
    across = np.random.default_rng(3).normal(0, 1, (y1 - y0) // 3 + 2).astype(np.float32)
    along += np.interp(yy[:, 0] - y0, np.arange(len(across)) * 3, _smooth(across[:, None], 1.0)[:, 0] * 5.0)[:, None] * (along < 120)
    _cache["stroke"] = (x0, y0, x1, y1, box, silk, hide, along, run)
    return _cache["stroke"]


def source(t):
    """The scan as it is t seconds into the shot: the last stroke written as far as the brush has got."""
    a = _scan()
    if REVEAL <= 0 or t >= REVEAL:
        return a
    x0, y0, x1, y1, box, silk, hide, along, run = _stroke()
    u = min(max(t / REVEAL, 0.0), 1.0)
    u = u * u * (3 - 2 * u) * 0.35 + u * 0.65            # the brush takes off quickly and slows a little into the dot
    front = -FRONT + (run + 2 * FRONT) * u
    written = np.clip((front - along) / FRONT, 0, 1)
    if u >= 1:
        return a
    out = a.copy()
    k = (hide * (1 - written))[..., None]
    out[y0:y1, x0:x1] = box * (1 - k) + silk * k
    return out


def frame(t, r=1.0):
    """The frame t seconds into the shot, r times 1920 wide, RGB uint8."""
    s, (ox, oy) = place(t)
    src = source(t)
    w, h = W / s, H / s
    im = Image.fromarray(np.uint8(np.clip(src + 0.5, 0, 255)))
    out = (int(round(W * r)), int(round(H * r)))
    return np.asarray(im.resize(out, Image.LANCZOS, box=(ox, oy, ox + w, oy + h)))


def geometry():
    s, o = place(0)
    print("cut: scale %.3f, scan from (%.1f, %.1f) to (%.1f, %.1f)" % (s, o[0], o[1], o[0] + W / s, o[1] + H / s))
    print("the stroke starts at frame (%.1f, %.1f), going %.1f degrees above the horizontal" % (*to_frame(START, 0), direction()))
    s, o = place(SECS)
    print("end: scale %.3f, scan from (%.1f, %.1f) to (%.1f, %.1f); 思 frame y %.0f-%.0f" % (s, o[0], o[1], o[0] + W / s, o[1] + H / s, (620 - o[1]) * s, (1420 - o[1]) * s))
    print("pace %.3f a second in log scale (%.2f times a second)" % (rate(), math.exp(rate())))


if __name__ == "__main__":
    if sys.argv[1] == "geometry":
        geometry()
