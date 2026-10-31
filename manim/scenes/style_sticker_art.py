"""The die-cuts for the sticker clip. Each sticker is lifted from another clip's own frame, or set in type; this
writes their colour and foil maps, cuts a clean mesh along each outline, and lays them all out over a laptop lid.

    python style_sticker_art.py            then    Blender -b --factory-startup -P style_sticker_blender.py

Falls back to the newest test frame for a clip that has no final render yet: run again once they all do."""
import glob
import json
import math
import os
import random
import re
import subprocess

import cairo
import numpy as np
from PIL import Image
from scipy.ndimage import (binary_closing, binary_dilation, binary_opening, binary_fill_holes, distance_transform_edt, gaussian_filter,
                           label, map_coordinates, maximum_filter)

from styles_common import FUTURA, OUT, SERIF, SEQ, alpha_of, canvas, font, hx, poly, rgb_of, rrect

CACHE = os.path.join(OUT, "cache_sticker")
SRC = os.path.join(CACHE, "src")
LID_W, LID_H = 13.0, 8.6                                   # the lid, in sticker units (one unit is about 2.3 cm)
BORDER = 0.075                                             # the white margin the die leaves round the print
THICK = 0.007
PPU = 230                                                  # print resolution, px per unit


# ======================================================================================
# source frames
# ======================================================================================
def frame(clip, t):
    """One frame of a finished clip, or the nearest test frame while the clip is not rendered yet."""
    path = os.path.join(SRC, f"{clip}_{t:05.2f}.png")
    mp4 = os.path.join(OUT, f"{clip}.mp4")
    if os.path.exists(mp4):
        if not os.path.exists(path) or os.path.getmtime(path) < os.path.getmtime(mp4):
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", mp4, "-frames:v", "1", path], check=True)
    else:
        tests = glob.glob(os.path.join(SEQ, "test", f"{clip}_f[0-9][0-9][0-9][0-9].png"))
        if not tests:
            raise FileNotFoundError(clip)
        path = min(tests, key=lambda p: abs(int(re.search(r"_f(\d+)", p).group(1)) / 30 - t))
    im = Image.open(path).convert("RGB")
    return im.resize((1920, 1080), Image.LANCZOS) if im.width != 1920 else im


# ======================================================================================
# die lines: each returns a boolean mask for a print of (h, w) px
# ======================================================================================
def rounded(h, w, r):
    s, c = canvas(None, w, h)
    rrect(c, 0, 0, w, h, r)
    c.set_source_rgb(1, 1, 1)
    c.fill()
    return np.asarray(rgb_of(s)[..., 0]) > 0.5


def shape_mask(kind, h, w):
    if callable(kind):                                     # a die cut round the drawing itself
        return kind(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx + 0.5) / w * 2 - 1, (yy + 0.5) / h * 2 - 1
    if kind == "oval":
        return u * u + v * v < 1
    if kind == "round":
        return rounded(h, w, min(h, w) * 0.09)
    if kind == "squircle":                                 # an app icon's corner
        return rounded(h, w, min(h, w) * 0.23)
    if kind == "arch":
        top = w / 2
        return (yy > top) | ((xx - w / 2) ** 2 + (yy - top) ** 2 < top * top)
    if kind == "ticket":
        m = rounded(h, w, min(h, w) * 0.06)
        r = h * 0.11
        for x0 in (0, w):
            m &= (xx - x0) ** 2 + (yy - h / 2) ** 2 > r * r
        return m
    if kind == "hex":
        return (np.abs(u) * 0.866 + np.abs(v) * 0.5 < 0.866) & (np.abs(v) < 1)
    raise ValueError(kind)


def lift(a, bg, tau, pad, trim=0.22):
    """What Photos' 'lift subject' does: everything far from the backdrop colour, closed up, with one margin of backdrop."""
    d = np.linalg.norm(a - np.asarray(bg, np.float32), axis=2)
    m = binary_closing(d > tau, iterations=6)
    m = binary_fill_holes(m)
    m = binary_opening(m, iterations=int(pad * trim))      # hair-thin strokes do not get a die line of their own
    lab, n = label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = sizes[lab] > sizes.max() * 0.12
    m = binary_dilation(m, iterations=1)
    far = distance_transform_edt(~m)
    return gaussian_filter((far < pad).astype(np.float32), pad * 1.0) > 0.5


# ======================================================================================
# foil: red channel is gold leaf, green is holographic film
# ======================================================================================
def gold_mask(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    warm = (r > b + 0.1) & (g > b) & (r - g < 0.42 * r) & (g > 0.34 * r)
    return gaussian_filter(warm.astype(np.float32), 0.7)


# ======================================================================================
# the sheet: (name, clip, time, crop, width in units, die, options)
# ======================================================================================
ART = [
    ("koi", "handdrawn", 4.8, (60, 150, 1540, 930), 6.3, "lift",
     dict(bg=(0.07, 0.10, 0.19), tau=0.42, pad=46, gold=True, ppu=420, press=True)),
    ("shan", "morph", 4.8, (330, 110, 1560, 1050), 2.2, "lift", dict(bg=(0.91, 0.90, 0.86), tau=0.55, pad=54, gloss=0.0)),
    ("rocket", "flat", 0.0, (500, 0, 1500, 1080), 2.0, "round", dict(gloss=0.6)),
    ("dusk", "pixel", 2.5, (560, 200, 1640, 830), 3.0, "ticket", dict(gloss=0.4, nearest=True)),
    ("crown", "splash", 3.3, (200, 230, 1760, 1000), 3.0, "oval", dict(gloss=0.6)),
    ("type", "letterpress", 0.8, (800, 0, 1600, 800), 2.2, "hex", dict(gloss=0.3)),
    ("weather", "glass", 0.0, (30, 50, 1910, 1040), 3.0, "round", dict(gloss=0.55, holo_rim=True)),
    ("dash", "glass", 4.9, (100, 10, 1790, 1010), 2.7, "round", dict(gloss=0.5)),
    ("hud", "cyberhud", 0.0, (300, 0, 1920, 911), 2.6, "round", dict(gloss=0.6)),
    ("lock", "cyberhud", 4.9, (560, 0, 1360, 1080), 1.52, "arch", dict(gloss=0.6)),
    ("wave", "collage", 1.5, (40, 0, 1400, 950), 3.2, "round", dict(gloss=0.25)),
    ("car", "synthwave", 0.0, (880, 480, 1740, 965), 2.1, "round", dict(gloss=0.45)),
    ("port", "port", 2.5, (640, 0, 1920, 1010), 2.6, "round", dict(gloss=0.4)),
    ("ship", "port", 0.0, (0, 140, 1240, 1080), 2.4, "round", dict(gloss=0.4)),
    ("lorenz", "line", 4.8, (120, 0, 1800, 1080), 4.3, "lift",
     dict(bg=(0.0, 0.0, 0.02), tau=0.13, pad=34, trim=2.4, ppu=320, holo_art=True, holo_rim=True, last=True)),
]


# ======================================================================================
# stickers set in type: each made the way the real thing is made, since these are the ones people lean in to read
# ======================================================================================
LABEL_PPU = 460
HEI, WEIBEI, XING, CHALK, OCR, DINC, DINA = "STHeiti", "Weibei SC", "Xingkai SC", "Chalkduster", "Andale Mono", "DIN Condensed", "DIN Alternate"


def grid(h, w):
    return np.mgrid[0:int(h), 0:int(w)].astype(np.float32)


def field(h, w, sigma, seed, sx=None):
    """Smooth noise, zero mean and unit deviation; a separate sx draws it out sideways, into fibre or brushing."""
    h, w, sx = int(h), int(w), sigma if sx is None else sx
    k = max(1, int(min(sigma, sx) / 3))
    n = np.random.default_rng(seed).standard_normal((h // k + 2, w // k + 2)).astype(np.float32)
    n = gaussian_filter(n, (sigma / k, sx / k))
    n = np.asarray(Image.fromarray(n).resize((w, h), Image.BICUBIC)) if k > 1 else n[:h, :w]
    return n / n.std()


def cover(w, h, draw):
    """What draw(c) puts down, as coverage 0-1."""
    s, c = canvas(None, int(w), int(h))
    draw(c)
    return alpha_of(s).copy()


def wander(a, amp, sigma, seed):
    """Edges never quite follow the plate: every pixel pushed a little way along a slow noise."""
    h, w = a.shape[:2]
    yy, xx = grid(h, w)
    at = [yy + field(h, w, sigma, seed) * amp, xx + field(h, w, sigma, seed + 1) * amp]
    return map_coordinates(a, at, order=1, mode="nearest")


def plate(m, seed, amp=0.5, pin=2.7):
    """A clean mask as a press prints it: edges wander and round off, solids go thin in places, pinholes open."""
    h, w = m.shape
    m = np.clip((gaussian_filter(wander(m, amp, 3.5, seed), 0.7) - 0.36) / 0.28, 0, 1)
    hole = gaussian_filter((field(h, w, 0.6, seed + 2) > pin).astype(np.float32), 0.5) * 1.6
    return m * (1 - np.clip(hole, 0, 1)) * (0.965 + 0.022 * field(h, w, 24, seed + 3) + 0.008 * field(h, w, 0.8, seed + 4))


def stock(h, w, col, seed, fibre=0.02):
    """Paper: a warm sheet with a faint cloud, and short fibres lying every way."""
    rnd = random.Random(seed)

    def fibres(k):
        k.set_line_width(0.9)
        for _ in range(int(h * w / 800)):
            a, n = rnd.uniform(0, math.pi), rnd.uniform(6, 26)
            k.move_to(rnd.uniform(0, w), rnd.uniform(0, h))
            k.rel_line_to(n * math.cos(a), n * math.sin(a))
            k.set_source_rgba(0, 0, 0, rnd.uniform(0.1, 0.5))
            k.stroke()

    t = fibre * (0.35 * field(h, w, 0.8, seed) + 0.35 * field(h, w, 3, seed + 1) + 0.6 * field(h, w, 28, seed + 2) - 2.4 * gaussian_filter(cover(w, h, fibres), 0.5))
    return np.array(hx(col), np.float32) * (1 + t[..., None])


def lay(art, m, col, over=False):
    """Ink down on a print: opaque, or overprinting so that what is under it shows through."""
    col = np.array(hx(col), np.float32)
    m = m[..., None]
    return art * (1 - m) + (art * col if over else col) * m


def scuffs(h, w, seed, reach, level=1.6):
    """Rubbed away at the corners and edges first, where a thumb or a carton has caught it."""
    yy, xx = grid(h, w)
    edge = np.minimum.reduce((xx, w - xx, yy, h - yy)) / reach
    n = field(h, w, 1.2, seed) * 0.45 + field(h, w, 7, seed + 1) * 0.6 + field(h, w, 40, seed + 2) * 0.5
    return np.clip((n - level - np.clip(edge, 0, 2.6)) * 3, 0, 1)


def ramp(v, stops):
    """Colour stops along v."""
    at = [s[0] for s in stops]
    cols = np.array([hx(s[1]) for s in stops], np.float32)
    return np.stack([np.interp(v, at, cols[:, i]) for i in range(3)], axis=-1).astype(np.float32)


def spaced(c, body, x, y, size, face, bold=True, width=None, tracking=0.0, align="left"):
    """A line set a letter at a time, tracked out to a measure."""
    font(c, face, size, bold)
    adv = [c.text_extents(ch).x_advance for ch in body]
    if width is not None:
        tracking = (width - sum(adv)) / max(1, len(body) - 1)
    x -= {"left": 0, "center": 0.5, "right": 1}[align] * (sum(adv) + tracking * (len(body) - 1))
    for ch, d in zip(body, adv):
        c.move_to(x, y)
        c.show_text(ch)
        x += d + tracking
    c.new_path()


def ring_text(c, body, cx, cy, r, size, col, face, bold=True, bottom=False, tracking=0.0):
    """Set round a circle: over the top reading clockwise, or under the bottom reading the other way."""
    font(c, face, size, bold)
    adv = [c.text_extents(ch).x_advance + tracking for ch in body]
    total = sum(adv) / r
    c.set_source_rgb(*hx(col))
    a = (math.pi / 2 + total / 2) if bottom else (-math.pi / 2 - total / 2)
    for ch, d in zip(body, adv):
        a += (-d / 2 / r) if bottom else (d / 2 / r)
        c.save()
        c.translate(cx + r * math.cos(a), cy + r * math.sin(a))
        c.rotate(a - math.pi / 2 if bottom else a + math.pi / 2)
        c.move_to(-(d - tracking) / 2, 0)
        c.show_text(ch)
        c.restore()
        c.new_path()
        a += (-d / 2 / r) if bottom else (d / 2 / r)


def label_hazard(c, w, h):
    """A carton label off a flexo press: a yellow flood, the black plate a hair out of register, the pictogram boxes knocked out of both."""
    H_, W_ = int(h), int(w)
    side, top, inset = h * 0.42, h * 0.29, w * 0.035
    boxes = (inset, w - inset - side)

    def knock(k):
        for x in boxes:
            rrect(k, x, top, side, side, side * 0.05)
            k.fill()

    def black(k):
        band = h * 0.125
        for edge in (0, h - band):
            for i in range(-2, int(w / (band * 1.1)) + 3):
                x = i * band * 1.1
                poly(k, [(x, edge + band), (x + band * 0.55, edge + band), (x + band * 1.1, edge), (x + band * 0.55, edge)])
                k.fill()
        k.set_line_width(h * 0.012)
        for x in boxes:
            rrect(k, x, top, side, side, side * 0.05)
            k.stroke()
        measure, mid = w - 2 * (inset + side) - w * 0.05, w / 2
        spaced(k, "轻拿轻放", mid, h * 0.5, h * 0.235, HEI, width=measure, align="center")
        spaced(k, "HANDLE WITH CARE", mid, h * 0.64, h * 0.118, DINC, width=measure, align="center")
        spaced(k, "易碎物品  向上  怕摔  GB/T 191", mid, h * 0.725, h * 0.042, HEI, width=measure, align="center")
        x, y, s = boxes[0] + side / 2, top + side * 0.13, side             # ISO 780's stemmed glass, with its crack
        k.move_to(x - s * 0.24, y)
        k.line_to(x + s * 0.24, y)
        k.curve_to(x + s * 0.27, y + s * 0.3, x + s * 0.1, y + s * 0.43, x + s * 0.02, y + s * 0.43)
        k.line_to(x - s * 0.02, y + s * 0.43)
        k.curve_to(x - s * 0.1, y + s * 0.43, x - s * 0.27, y + s * 0.3, x - s * 0.24, y)
        k.fill()
        k.rectangle(x - s * 0.018, y + s * 0.42, s * 0.036, s * 0.26)
        k.rectangle(x - s * 0.17, y + s * 0.67, s * 0.34, s * 0.035)
        k.fill()
        k.set_operator(cairo.OPERATOR_CLEAR)
        k.set_line_width(s * 0.024)
        poly(k, [(x + s * 0.07, y - s * 0.01), (x - s * 0.03, y + s * 0.11), (x + s * 0.05, y + s * 0.17), (x - s * 0.04, y + s * 0.29)], close=False)
        k.stroke()
        k.set_operator(cairo.OPERATOR_OVER)
        x, y = boxes[1] + side / 2, top + side * 0.14                      # this way up
        for dx in (-0.17, 0.17):
            cx = x + dx * s
            poly(k, [(cx, y), (cx + s * 0.12, y + s * 0.2), (cx + s * 0.038, y + s * 0.2), (cx + s * 0.038, y + s * 0.62),
                     (cx - s * 0.038, y + s * 0.62), (cx - s * 0.038, y + s * 0.2), (cx - s * 0.12, y + s * 0.2)])
            k.fill()
        k.rectangle(x - s * 0.3, y + s * 0.67, s * 0.6, s * 0.035)
        k.fill()

    rub = scuffs(H_, W_, 23, h * 0.09)
    art = stock(H_, W_, "#F5F1E4", 5)
    art = lay(art, plate(np.roll(1 - cover(w, h, knock), (2, 3), axis=(0, 1)), 17, pin=3.1) * (1 - rub * 0.9), "#F8CC17", True)
    return lay(art, plate(cover(w, h, black), 11) * (1 - rub), "#1A191C", True)


def label_cert(c, w, h):
    """合格证: green letterpress on uncoated slip, the date from a roller stamp, and the inspector's red chop struck across it."""
    H_, W_ = int(h), int(w)

    def printed(k):
        k.set_line_width(h * 0.018)
        rrect(k, w * 0.03, h * 0.05, w * 0.94, h * 0.9, h * 0.03)
        k.stroke()
        k.set_line_width(h * 0.005)
        rrect(k, w * 0.046, h * 0.078, w * 0.908, h * 0.844, h * 0.02)
        k.stroke()
        spaced(k, "合格证", w * 0.5, h * 0.335, h * 0.21, SERIF, width=w * 0.42, align="center")
        spaced(k, "CERTIFICATE OF CONFORMITY", w * 0.5, h * 0.42, h * 0.044, "Copperplate", width=w * 0.42, align="center")
        k.set_line_width(h * 0.005)
        for y, a in ((h * 0.615, "检验员"), (h * 0.815, "检验日期")):
            spaced(k, a, w * 0.1, y, h * 0.085, SERIF)
            k.move_to(w * 0.37, y + h * 0.022)
            k.line_to(w * 0.905, y + h * 0.022)
            k.stroke()

    def numbered(k):
        spaced(k, "No 0013137", w * 0.915, h * 0.175, h * 0.06, DINA, tracking=h * 0.003, align="right")

    def dated(k):
        rnd = random.Random(41)
        for body, x, y in (("07", w * 0.39, h * 0.605), ("1984.10.30", w * 0.39, h * 0.805)):
            for ch in body:
                k.save()
                k.translate(x, y + rnd.gauss(0, h * 0.004))
                k.rotate(-0.025 + rnd.gauss(0, 0.012))
                font(k, DINA, h * 0.108, True)
                k.move_to(0, 0)
                k.show_text(ch)
                x += k.text_extents(ch).x_advance + h * 0.006
                k.restore()
                k.new_path()

    cx, cy, R = w * 0.785, h * 0.66, h * 0.215

    def chop(k):
        k.translate(cx, cy)
        k.rotate(-0.26)
        k.set_line_width(h * 0.015)
        k.arc(0, 0, R, 0, 2 * math.pi)
        k.stroke()
        k.set_line_width(h * 0.004)
        k.arc(0, 0, R * 0.9, 0, 2 * math.pi)
        k.stroke()
        ring_text(k, "质量检验专用章", 0, 0, R * 0.62, h * 0.046, "#000000", HEI, tracking=h * 0.012)
        spaced(k, "合格", 0, h * 0.055, h * 0.105, HEI, align="center")
        spaced(k, "QC 07", 0, h * 0.135, h * 0.062, DINC, tracking=h * 0.006, align="center")

    yy, xx = grid(H_, W_)
    art = stock(H_, W_, "#F6F2E3", 31, 0.026)
    art = lay(art, plate(cover(w, h, printed), 33, 0.3, 3.2), "#0F6B44", True)
    art = lay(art, plate(cover(w, h, numbered), 35, 0.3, 3.0), "#C4201A", True)
    violet = plate(cover(w, h, dated), 37, 0.4, 2.5) * np.clip(0.8 + 0.28 * field(H_, W_, 12, 39) + 0.12 * field(H_, W_, 0.9, 40), 0.25, 1)
    art = lay(art, violet, "#2C2A7E", True)
    # it was struck leaning: heavy at the lower left, nearly dry at the upper right
    lean = 0.74 + 0.3 * ((cx - xx) * 0.55 + (yy - cy) * 0.83) / R
    red = plate(cover(w, h, chop), 47, 0.45, 2.3) * np.clip(lean + 0.2 * field(H_, W_, 11, 49) + 0.14 * field(H_, W_, 0.9, 51), 0.12, 1)
    return lay(art, red, "#CC1F1A", True)


EAN_L = "0001101 0011001 0010011 0111101 0100011 0110001 0101111 0111011 0110111 0001011".split()
EAN_FIRST = "LLLLLL LLGLGG LLGGLG LLGGGL LGLLGG LGGLLG LGGGLL LGLGLG LGLGGL LGGLGL".split()


def ean13(code):
    """The 95 modules of a real EAN-13, 1 for a bar: a scanner would read this one."""
    d = [int(x) for x in code]
    inv = lambda s: "".join("10"[int(b)] for b in s)
    left = "".join(EAN_L[n] if p == "L" else inv(EAN_L[n])[::-1] for n, p in zip(d[1:7], EAN_FIRST[d[0]]))
    return "101" + left + "01010" + "".join(inv(EAN_L[n]) for n in d[7:]) + "101"


def label_barcode(c, w, h):
    """A thermal label on a carton: a true EAN-13 with its long guard bars, a dead dot on the head, and the digits under it."""
    H_, W_ = int(h), int(w)
    code = "6913000013137"
    X = w * 0.86 / 113
    x0 = w * 0.07 + 11 * X
    top, bottom = h * 0.17, h * 0.72
    rnd = random.Random(13)

    def bars(k):
        bits = ean13(code)
        i = 0
        while i < 95:
            if bits[i] == "0":
                i += 1
                continue
            j = i
            while j < 95 and bits[j] == "1":
                j += 1
            guard = i < 3 or j > 92 or 45 <= i < 50
            a, b = x0 + i * X + rnd.gauss(0, 0.45), x0 + j * X + rnd.gauss(0, 0.45)
            k.rectangle(a, top, b - a, (h * 0.8 if guard else bottom) - top)
            k.fill()
            i = j
        spaced(k, "风格库 STYLE REEL  13 式", x0, h * 0.12, h * 0.06, HEI)
        size = h * 0.125
        spaced(k, code[0], x0 - 8.4 * X, h * 0.885, size, OCR)
        spaced(k, code[1:7], x0 + 24 * X, h * 0.885, size, OCR, width=38 * X, align="center")
        spaced(k, code[7:], x0 + 71 * X, h * 0.885, size, OCR, width=38 * X, align="center")

    K = np.clip(plate(cover(w, h, bars), 61, 0.16, 3.2) * 1.05, 0, 1) * (0.97 + 0.025 * field(H_, W_, 50, 63))
    K[int(h * 0.42):int(h * 0.42) + 2] *= 0.12             # the dead dot, and a second one going
    K[int(h * 0.59):int(h * 0.59) + 1] *= 0.5
    return lay(stock(H_, W_, "#F9F8F2", 65, 0.008), K, "#1E1D22", True)


def label_chip(c, w, h):
    """A quad-flat package on a green board: tinned legs on gold pads, traces and silkscreen round it, a brushed lid with the laser's own marks."""
    H_, W_ = int(h), int(w)
    n, B, T, P = 14, w * 0.32, w * 0.392, w * 0.41
    pitch = 2 * B * 0.94 / n
    c.set_source_rgb(*hx("#0D3A2B"))
    c.paint()
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    rnd = random.Random(29)
    for side in range(4):
        c.save()
        c.translate(w / 2, h / 2)
        c.rotate(side * math.pi / 2)
        for i in range(n):
            x = (i - (n - 1) / 2) * pitch
            far = x * (1.0 + rnd.choice((0.0, 0.06, 0.12)))
            c.set_source_rgb(*hx("#176A4C"))               # the trace, under the solder mask
            c.set_line_width(w * 0.0058)
            poly(c, [(x, -P), (x, -P - w * 0.018), (far, -P - w * 0.05), (far, -w * 0.52)], close=False)
            c.stroke()
            if i % 3 == 1:                                 # a via
                c.arc(far, -P - w * 0.062, w * 0.0085, 0, 2 * math.pi)
                c.set_source_rgb(*hx("#C9A24E"))
                c.fill()
                c.arc(far, -P - w * 0.062, w * 0.0036, 0, 2 * math.pi)
                c.set_source_rgb(*hx("#07170F"))
                c.fill()
            c.rectangle(x - pitch * 0.34, -P, pitch * 0.68, P - B * 1.02)
            c.set_source_rgb(*hx("#C69C42"))
            c.fill()
            g = cairo.LinearGradient(0, -T, 0, -B)
            for at, col in ((0, "#C4C6CA"), (0.3, "#F5F6F7"), (0.5, "#8A8D93"), (0.62, "#5F6268"), (0.8, "#ECEDEF"), (1, "#3A3C42")):
                g.add_color_stop_rgb(at, *hx(col))
            c.rectangle(x - pitch * 0.25, -T, pitch * 0.5, T - B)
            c.set_source(g)
            c.fill()
        c.restore()
    x0 = w / 2 - B
    poly(c, [(x0 + w * 0.05, x0), (w - x0, x0), (w - x0, h - x0), (x0, h - x0), (x0, x0 + w * 0.05)])  # the moulded body, its pin-one corner cut off
    c.set_source_rgb(*hx("#18191C"))
    c.fill_preserve()
    c.set_source_rgb(*hx("#34353A"))
    c.set_line_width(w * 0.004)
    c.stroke()
    for x, y, a in ((w * 0.058, h * 0.058, 0.0), (w * 0.94, h * 0.058, 0.0), (w * 0.058, h * 0.94, 0.0)):  # chip capacitors in the corners
        c.rectangle(x - w * 0.03, y - w * 0.015, w * 0.06, w * 0.03)
        c.set_source_rgb(*hx("#B58F5E"))
        c.fill()
        for dx in (-1, 1):
            c.rectangle(x + dx * w * 0.03 - w * 0.009, y - w * 0.015, w * 0.018, w * 0.03)
            c.set_source_rgb(*hx("#D9DADD"))
            c.fill()
    art = rgb_of(c.get_target()).copy()
    yy, xx = grid(H_, W_)
    weave = np.sin(xx * 2 * math.pi / 9) * np.sin(yy * 2 * math.pi / 9)
    board = (art[..., 1] > art[..., 0] * 1.5)[..., None]   # the green only: glass cloth showing through the mask
    art = np.where(board, art * (1 + (0.05 * weave + 0.05 * field(H_, W_, 30, 66))[..., None]), art)

    def silk(k):
        spaced(k, "U1", w * 0.905, h * 0.965, h * 0.036, OCR)
        spaced(k, "C13", w * 0.03, h * 0.12, h * 0.03, OCR)
        spaced(k, "C05", w * 0.905, h * 0.12, h * 0.03, OCR)
        spaced(k, "R30", w * 0.03, h * 0.9, h * 0.03, OCR)

    art = lay(art, plate(cover(w, h, silk), 67, 0.2, 3.2), "#E7E3D2")
    L = w * 0.262
    lid = cover(w, h, lambda k: (rrect(k, w / 2 - L, h / 2 - L, 2 * L, 2 * L, w * 0.016), k.fill()))
    tone = 0.66 + 0.035 * field(H_, W_, 0.6, 71, 34) + 0.02 * field(H_, W_, 1.4, 72, 90) + 0.09 * np.sin((xx + yy) / w * 3.6 + 0.3)
    soft = gaussian_filter(lid, 3.0)
    gy, gx = np.gradient(soft)
    tone = tone + 2.2 * (0.6 * gx + 0.8 * gy) * lid
    metal = tone[..., None] * np.array((0.99, 1.0, 1.03), np.float32)
    art = art * (1 - lid[..., None]) + metal * lid[..., None]

    def marks(k):
        cx = w / 2
        spaced(k, "M5", cx, h * 0.5, h * 0.205, DINA, align="center")
        spaced(k, "MAX", cx, h * 0.585, h * 0.08, DINA, tracking=h * 0.012, align="center")
        spaced(k, "18 CORES", cx, h * 0.64, h * 0.034, OCR, align="center")
        spaced(k, "339S01305 2644", cx, h * 0.685, h * 0.026, OCR, align="center")
        k.set_line_width(h * 0.004)
        k.arc(cx - L + w * 0.05, h / 2 - L + w * 0.05, w * 0.016, 0, 2 * math.pi)
        k.stroke()

    etch = cover(w, h, marks) * (0.9 + 0.1 * np.sin(yy * 2 * math.pi / 3))
    art = art + (np.roll(etch, (2, 1), axis=(0, 1)) * (1 - etch))[..., None] * 0.16 * lid[..., None]  # the lit far wall of each cut
    return art * (1 - etch[..., None] * 0.84)


def clapper_parts(w, h):
    board = [(w * 0.02, h * 0.345), (w * 0.98, h * 0.345), (w * 0.98, h * 0.985), (w * 0.02, h * 0.985)]
    fixed = [(w * 0.02, h * 0.25), (w * 0.98, h * 0.25), (w * 0.98, h * 0.345), (w * 0.02, h * 0.345)]
    a, L, t = -0.12, w * 0.96, h * 0.095
    ox, oy, dx, dy = w * 0.02, h * 0.24, math.cos(a), math.sin(a)
    arm = [(ox, oy), (ox + dx * L, oy + dy * L), (ox + dx * L + dy * t, oy + dy * L - dx * t), (ox + dy * t, oy - dx * t)]
    return board, fixed, arm, (ox, oy, a, L, t)


def die_clapper(h, w):
    def cut(k):
        for part in clapper_parts(w, h)[:3]:
            poly(k, part)
            k.fill()
        rrect(k, w * 0.018, h * 0.175, w * 0.082, h * 0.168, w * 0.012)
        k.fill()
    return cover(w, h, cut) > 0.5


def label_clapper(c, w, h):
    """A film slate: matt black acrylic with its printed grid, screws in the hinge, and this reel's numbers in chalk over old erasures."""
    H_, W_ = int(h), int(w)
    board, fixed, arm, (ox, oy, a, L, t) = clapper_parts(w, h)
    c.set_source_rgb(1, 1, 1)
    c.paint()

    def stripes(x0, y0, ww, hh, lean):
        c.save()
        c.rectangle(x0, y0, ww, hh)
        c.clip()
        c.set_source_rgb(*hx("#141417"))
        c.paint()
        k = ww / 7.0
        c.set_source_rgb(*hx("#EFEDE4"))
        for i in range(-1, 9):
            x = x0 + i * k
            poly(c, [(x + lean * hh, y0), (x + lean * hh + k * 0.5, y0), (x + k * 0.5, y0 + hh), (x, y0 + hh)])
            c.fill()
        g = cairo.LinearGradient(0, y0, 0, y0 + hh)        # a little round: lit along the top, dark along the bottom
        for at, col, al in ((0, "#FFFFFF", 0.3), (0.12, "#FFFFFF", 0.0), (0.8, "#000000", 0.0), (1, "#000000", 0.42)):
            g.add_color_stop_rgba(at, *hx(col), al)
        c.set_source(g)
        c.paint()
        c.restore()

    stripes(w * 0.02, h * 0.25, w * 0.96, h * 0.095, -0.7)
    c.save()
    c.translate(ox, oy)
    c.rotate(a)
    stripes(0, -t, L, t, 0.7)
    c.restore()
    poly(c, board)
    c.set_source_rgb(*hx("#17181B"))
    c.fill()
    g = cairo.LinearGradient(0, h * 0.19, 0, h * 0.34)     # the hinge plate, across both sticks
    for at, col in ((0, "#D6D8DB"), (0.5, "#8D9096"), (1, "#C4C6CA")):
        g.add_color_stop_rgb(at, *hx(col))
    rrect(c, w * 0.018, h * 0.175, w * 0.082, h * 0.168, w * 0.012)
    c.set_source(g)
    c.fill_preserve()
    c.set_source_rgb(*hx("#55575C"))
    c.set_line_width(w * 0.003)
    c.stroke()
    for x, y in ((w * 0.059, h * 0.207), (w * 0.059, h * 0.305)):
        g = cairo.RadialGradient(x - w * 0.004, y - w * 0.004, 0, x, y, w * 0.014)
        g.add_color_stop_rgb(0, *hx("#F1F2F3"))
        g.add_color_stop_rgb(1, *hx("#62646A"))
        c.arc(x, y, w * 0.014, 0, 2 * math.pi)
        c.set_source(g)
        c.fill()
        c.save()
        c.translate(x, y)
        c.rotate(0.5 + x)
        c.rectangle(-w * 0.012, -w * 0.002, w * 0.024, w * 0.004)
        c.set_source_rgb(*hx("#26272B"))
        c.fill()
        c.restore()
    art = rgb_of(c.get_target()).copy()
    inside = cover(w, h, lambda k: (poly(k, board), k.fill()))

    def screen(k):
        k.set_line_width(h * 0.0055)
        for y in (0.475, 0.665, 0.855):
            k.move_to(w * 0.02, h * y)
            k.line_to(w * 0.98, h * y)
            k.stroke()
        k.move_to(w * 0.5, h * 0.475)
        k.line_to(w * 0.5, h * 0.855)
        k.stroke()
        spaced(k, "PROD.", w * 0.04, h * 0.39, h * 0.04, DINC, tracking=h * 0.004)
        for x, y, a_ in ((0.04, 0.515, "SCENE"), (0.52, 0.515, "TAKE"), (0.04, 0.705, "FPS"), (0.52, 0.705, "SEC")):
            spaced(k, a_, w * x, h * y, h * 0.04, DINC, tracking=h * 0.004)
        for i, a_ in enumerate(("INT", "EXT", "DAY", "NITE", "SYNC", "MOS")):
            spaced(k, a_, w * (0.04 + (i + 0.5) * 0.92 / 6), h * 0.937, h * 0.042, DINC, tracking=h * 0.004, align="center")

    def chalk(k):
        font(k, XING, h * 0.1, True)
        k.move_to(w * 0.15, h * 0.452)
        k.show_text("风格库")
        font(k, CHALK, h * 0.06, False)
        k.move_to(w * 0.15 + h * 0.33, h * 0.45)
        k.show_text("STYLE REEL")
        for x, y, v in ((0.47, 0.637, "13"), (0.95, 0.637, "05"), (0.47, 0.827, "30"), (0.95, 0.827, "5.0")):
            font(k, CHALK, h * 0.125, False)
            k.move_to(w * x - k.text_extents(v).x_advance, h * y)
            k.show_text(v)
        k.set_line_width(h * 0.0065)
        for x in (0.04 + 1.5 * 0.92 / 6, 0.04 + 3.5 * 0.92 / 6):   # ringed: an exterior, at night
            pts = [(w * x + w * 0.05 * math.cos(u / 40 * 6.9) * (1 + 0.04 * math.sin(u)), h * 0.922 + h * 0.034 * math.sin(u / 40 * 6.9)) for u in range(41)]
            poly(k, pts, close=False)
            k.stroke()

    white = plate(cover(w, h, screen), 83, 0.16, 3.0) * inside
    art = lay(art, white, "#D9D7CC")
    mark = wander(cover(w, h, chalk), 0.6, 3, 85)
    drag = gaussian_filter(mark, (1.6, 16)) * 0.3 + np.clip(field(H_, W_, 11, 87, 34), 0, None) ** 2 * 0.012  # what the eraser left
    art = art + ((drag + np.clip(field(H_, W_, 0.7, 88, 12) - 2.2, 0, 1) * 0.08) * inside)[..., None] * np.array(hx("#DAD8CE"), np.float32)
    dust = np.clip(0.9 + 0.26 * field(H_, W_, 0.8, 89), 0, 1)
    return lay(art, mark * dust, "#EFECE2")


def label_seal(c, w, h):
    """A seal struck on xuan paper: stele-cut strokes left white in cinnabar, read right column first, the ink thin wherever the stone did not press."""
    H_, W_ = int(h), int(w)
    s = w * 0.86

    def place(k):
        k.translate(w / 2, h / 2)
        k.rotate(-0.028)

    def block(k):
        place(k)
        k.rectangle(-s / 2, -s / 2, s, s)
        k.fill()

    def carved(k):
        place(k)
        cell = s * 0.45
        for ch, cx, cy in (("琢", 1, -1), ("帧", 1, 1), ("手", -1, -1), ("绘", -1, 1)):
            font(k, WEIBEI, 100, False)
            e = k.text_extents(ch)
            k.save()
            k.translate(cx * s * 0.243, cy * s * 0.243)
            k.scale(cell / e.width, cell / e.height)
            k.move_to(-e.x_bearing - e.width / 2, -e.y_bearing - e.height / 2)
            k.show_text(ch)
            k.restore()
            k.new_path()

    art = stock(H_, W_, "#F1E9D6", 91, 0.034)
    area = wander(cover(w, h, block), 2.0, 16, 93)
    area = np.clip((gaussian_filter(area, 1.2) + 0.1 * field(H_, W_, 1.3, 94) - 0.42) / 0.16, 0, 1)
    chips = np.clip(gaussian_filter(((field(H_, W_, 5, 95) > 2.2) * (1 - gaussian_filter(area, 12) > 0.08)).astype(np.float32), 0.8) * 2, 0, 1)
    area = np.clip(area - chips, 0, 1)
    cut = wander(cover(w, h, carved), 1.0, 4, 97)
    cut = np.clip((gaussian_filter(cut, 0.8) + 0.07 * field(H_, W_, 1.6, 99) - 0.4) / 0.2, 0, 1)
    stone = area * (1 - cut)
    rim = np.clip(stone - gaussian_filter(stone, 3.5), 0, 1)
    tooth = field(H_, W_, 1.0, 101) * 0.6 + field(H_, W_, 3.2, 102) * 0.6
    press = 0.5 * field(H_, W_, 34, 103) + 0.35 * field(H_, W_, 9, 104)
    ink = stone * np.clip((2.05 + 0.7 * press - tooth) * 1.8, 0, 1)
    art = art * (1 - (gaussian_filter(area, 22) * 0.07)[..., None] * np.array((0.1, 0.45, 0.8), np.float32))  # the oil creeping out into the paper
    red = ramp(np.clip(0.5 + 0.16 * press + rim * 0.7, 0, 1), [(0, "#CE3B24"), (0.5, "#BB2519"), (1, "#8C150F")])
    return art * (1 - ink[..., None]) + red * ink[..., None]


def satin(t, s, pitch, seed, tatami=0.0):
    """A run of thread: ridged across its pitch, every thread its own shade and, in a long fill, stitch ends staggered along it."""
    rnd = np.random.default_rng(seed).random((2, 4096)).astype(np.float32)
    f = t / pitch
    i = np.floor(f).astype(np.int64) % 4096
    tone = 0.76 + 0.32 * np.sin(np.pi * (f - np.floor(f))) ** 0.7 + 0.1 * (rnd[0][i] - 0.5)
    if tatami:
        u = (s / tatami + rnd[1][i]) % 1
        tone = tone * (1 - 0.15 * np.exp(-(np.minimum(u, 1 - u) / 0.035) ** 2))
    return tone


def label_patch(c, w, h):
    """A sew-on mission patch: flat colour turned into stitches, satin and fill on a twill ground, inside a merrowed edge."""
    H_, W_ = int(h), int(w)
    cx, cy, R = w / 2, h / 2, h / 2
    inner = R * 0.74
    RED, CREAM, NAVY, SUN, SEA, FLAME = "#D8331C", "#F5EEDC", "#14296B", "#F7861C", "#1BAEC2", "#FBBE28"
    opts = cairo.FontOptions()
    opts.set_antialias(cairo.ANTIALIAS_NONE)
    c.set_font_options(opts)
    c.set_antialias(cairo.ANTIALIAS_NONE)
    c.set_source_rgb(*hx(RED))
    c.paint()
    ring_text(c, "MISSION 05", cx, cy, R * 0.785, R * 0.125, CREAM, FUTURA, True, tracking=R * 0.022)
    ring_text(c, "火箭升空", cx, cy, R * 0.895, R * 0.128, CREAM, HEI, True, bottom=True, tracking=R * 0.05)
    c.save()
    c.arc(cx, cy, inner, 0, 2 * math.pi)
    c.clip()
    c.set_source_rgb(*hx(NAVY))
    c.paint()
    c.set_source_rgb(*hx(SUN))                             # the sun on the horizon
    c.arc(cx + inner * 0.5, cy + inner * 0.5, inner * 0.46, 0, 2 * math.pi)
    c.fill()
    c.set_source_rgb(*hx(SEA))                             # the sea
    c.arc(cx - inner * 0.1, cy + inner * 2.15, inner * 1.62, 0, 2 * math.pi)
    c.fill()
    c.set_source_rgb(*hx(CREAM))
    for sx, sy, r in ((-0.6, -0.42, 0.05), (-0.28, -0.7, 0.036), (0.58, -0.55, 0.045), (0.72, -0.08, 0.032), (-0.72, 0.1, 0.034), (0.2, -0.82, 0.028)):
        c.arc(cx + sx * inner, cy + sy * inner, r * inner, 0, 2 * math.pi)
        c.fill()
    c.set_line_width(inner * 0.04)                         # the trail it climbed
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    c.move_to(cx - inner * 0.82, cy + inner * 0.55)
    c.curve_to(cx - inner * 0.42, cy + inner * 0.5, cx - inner * 0.3, cy + inner * 0.34, cx - inner * 0.16, cy + inner * 0.16)
    c.stroke()
    c.translate(cx + inner * 0.12, cy - inner * 0.14)
    c.rotate(0.7)
    k = inner * 0.66
    c.set_source_rgb(*hx(FLAME))                           # the flame
    poly(c, [(-k * 0.14, k * 0.62), (0, k * 1.08), (k * 0.14, k * 0.62)])
    c.fill()
    c.set_source_rgb(*hx(RED))                             # the fins
    for sx in (-1, 1):
        poly(c, [(sx * k * 0.17, k * 0.2), (sx * k * 0.42, k * 0.68), (sx * k * 0.17, k * 0.54)])
        c.fill()

    def hull():
        c.move_to(0, -k)
        c.curve_to(k * 0.3, -k * 0.62, k * 0.26, k * 0.22, k * 0.17, k * 0.6)
        c.line_to(-k * 0.17, k * 0.6)
        c.curve_to(-k * 0.26, k * 0.22, -k * 0.3, -k * 0.62, 0, -k)

    c.set_source_rgb(*hx(CREAM))                           # the body
    hull()
    c.fill()
    c.save()                                               # its red nose
    c.rectangle(-k, -k * 1.1, 2 * k, k * 0.5)
    c.clip()
    hull()
    c.set_source_rgb(*hx(RED))
    c.fill()
    c.restore()
    c.arc(0, -k * 0.28, k * 0.13, 0, 2 * math.pi)
    c.set_source_rgb(*hx(FLAME))
    c.fill()
    c.arc(0, -k * 0.28, k * 0.09, 0, 2 * math.pi)
    c.set_source_rgb(*hx(NAVY))
    c.fill()
    c.restore()

    pal = np.array([hx(x) for x in (RED, CREAM, NAVY, SUN, SEA, FLAME)], np.float32)
    flat = rgb_of(c.get_target())
    idx = np.argmin(((flat[..., None, :] - pal) ** 2).sum(-1), axis=-1)
    colour = pal[idx]
    yy, xx = grid(H_, W_)
    dx, dy = xx - cx, yy - cy
    r, arc = np.hypot(dx, dy), np.arctan2(dy, dx)
    rim = r > inner
    warp = field(H_, W_, 16, 111) * 1.5
    spin = arc * R * 0.85 + warp

    def along(deg):
        a_ = math.radians(deg)
        return xx * math.cos(a_) + yy * math.sin(a_) + warp, -xx * math.sin(a_) + yy * math.cos(a_)

    def sheen(deg):
        return 0.93 + 0.13 * math.cos(2 * math.radians(deg - 53))

    spun = 0.93 + 0.17 * np.cos(2 * (arc - math.radians(53)))
    tone = np.ones((H_, W_), np.float32)
    height = np.zeros((H_, W_), np.float32)
    occ = np.ones((H_, W_), np.float32)
    groups = [                                             # which pixels, thread path, stitch path, pitch, fill length, proud of the cloth, sheen
        ((idx == 0) & rim, spin, r, 7.0, 62.0, 0.7, spun),
        ((idx == 1) & rim, spin, r, 4.5, 0.0, 1.3, spun),
        (idx == 2, *along(45), 6.0, 0.0, 0.15, sheen(45)),
        (idx == 4, *along(8), 8.0, 74.0, 0.8, sheen(8)),
        (idx == 3, *along(100), 8.0, 74.0, 0.8, sheen(100)),
        ((idx == 1) & ~rim, *along(40), 6.0, 0.0, 1.2, sheen(40)),
        ((idx == 0) & ~rim, *along(40), 5.5, 0.0, 1.2, sheen(40)),
        (idx == 5, *along(130), 5.0, 0.0, 1.2, sheen(130)),
    ]
    for i, (m, t, s_, pitch, fill, lift, gleam) in enumerate(groups):
        d = distance_transform_edt(m).astype(np.float32)
        tex = satin(t, s_, pitch, 120 + i, fill) * gleam
        if lift < 0.5:                                     # the twill ground is cloth, not stitches
            tex = 0.95 + (tex - 0.95) * 0.55
        tone = np.where(m, tex, tone)
        height = np.where(m, lift * (1 - np.exp(-d / (pitch * 1.3))), height)
        occ = np.where(m, np.clip(d / 2.4, 0, 1), occ)
    gy, gx = np.gradient(gaussian_filter(height, 1.3))
    lit = 1 + 3.2 * (0.6 * gx + 0.8 * gy)
    art = colour * (tone * lit * (0.6 + 0.4 * occ))[..., None]
    edge = R * 0.925
    u = (r - (edge + R) / 2) / ((R - edge) / 2)            # the merrow: overlock thread wrapped slantwise round the rim
    wrap = satin(arc * R + (r - edge) * 1.3 + warp * 0.4, r, 9.0, 131) * np.sqrt(np.clip(1 - u * u * 0.85, 0, 1))
    wrap = wrap * (1 + 0.4 * -u * (np.cos(arc) * 0.6 + np.sin(arc) * 0.8)) * (0.93 + 0.13 * np.cos(2 * (arc - 0.9)))
    art = np.where((r > edge)[..., None], np.array(hx("#E4AE34"), np.float32) * wrap[..., None], art)
    art = art * (1 - (np.exp(-((r - edge) / 5) ** 2) * 0.55)[..., None])     # the groove it sinks into
    c.set_font_options(cairo.FontOptions())
    return gaussian_filter(art, (0.5, 0.5, 0))


BITMAP = {
    "I": "01110 00100 00100 00100 00100 00100 01110", "N": "10001 11001 10101 10011 10001 10001 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110", "E": "11111 10000 10000 11110 10000 10000 11111",
    "R": "11110 10001 10001 11110 10100 10010 10001", "T": "11111 00100 00100 00100 00100 00100 00100",
    "C": "01110 10001 10000 10000 10000 10001 01110", "O": "01110 10001 10001 10001 10001 10001 01110",
    "D": "11110 10001 10001 10001 10001 10001 11110", "0": "01110 10001 10011 10101 11001 10001 01110",
    " ": "00000 00000 00000 00000 00000 00000 00000",
}


def label_coin(c, w, h):
    """The arcade's two words on its own tube: a real 5x7 font, each pixel a beam on a shadow mask, blooming through bowed glass in a moulded bezel."""
    H_, W_ = int(h), int(w)
    lip = h * 0.075
    pw, ph = w - 2 * lip, h - 2 * lip
    cols = 96
    u = pw / cols
    rows = int(round(ph / u))
    px = np.tile(np.array(hx("#05061A"), np.float32), (rows, cols, 1))

    def cell(x, y, col):
        px[y, x] = hx(col)

    def write(body, x, y, col):
        for ch in body:
            for j, row in enumerate(BITMAP[ch].split()):
                for i, bit in enumerate(row):
                    if bit == "1":
                        cell(x + i, y + j, col)
            x += 6

    x0, y0, r = 12.5, rows // 2, 8.0
    for j in range(rows):
        for i in range(cols):
            d = math.hypot(i + 0.5 - x0, j + 0.5 - y0)
            if d < r:
                col = "#C8780F" if d > r - 1.6 else "#FFD21F"
                if abs(i + 0.5 - x0) < 1.5 and abs(j + 0.5 - y0) < 4.6:
                    col = "#C8780F"
                elif (i + 0.5 - x0) + (j + 0.5 - y0) < -7.2 and d < r - 1.6:
                    col = "#FFF3A8"
                cell(i, j, col)
    write("INSERT COIN", 24, y0 - 9, "#FFD21F")
    write("CREDIT 00", 24, y0 + 2, "#3CE8E0")

    yy, xx = grid(H_, W_)
    nx, ny = (xx - w / 2) / (pw / 2), (yy - h / 2) / (ph / 2)
    gx = (nx * (1 + 0.02 * ny * ny) + 1) / 2 * cols
    gy = (ny * (1 + 0.06 * nx * nx) + 1) / 2 * rows
    fx, fy = gx - np.floor(gx) - 0.5, gy - np.floor(gy) - 0.5
    P = px[np.clip(np.floor(gy).astype(np.int64), 0, rows - 1), np.clip(np.floor(gx).astype(np.int64), 0, cols - 1)]
    beam = np.exp(-(fy / 0.4) ** 2) * (0.72 + 0.28 * np.exp(-(fx / 0.48) ** 2))
    tri = xx.astype(np.int64) // 2 % 3                     # the aperture grille
    mask = np.stack([0.66 + 0.68 * (tri == i) for i in range(3)], axis=-1).astype(np.float32)
    lit = P * beam[..., None]
    glow = gaussian_filter(lit, (u * 0.7, u * 0.7, 0)) * 1.0 + gaussian_filter(lit, (u * 2.8, u * 2.8, 0)) * 0.42
    tube = 1 - np.exp(-(lit * mask * 1.5 + glow) * 1.65)
    tube = tube + np.array((0.018, 0.022, 0.026), np.float32)  # the phosphor is never black
    tube = tube * (1 - 0.42 * (nx ** 4 * 0.6 + ny ** 4 * 0.5))[..., None]
    tube = tube + (np.clip(1 - ((nx + 0.5) / 0.55) ** 2 - ((ny + 0.62) / 0.5) ** 2, 0, 1) ** 1.4 * 0.06)[..., None]  # the room, on the glass
    screen = gaussian_filter(rounded(int(ph), int(pw), ph * 0.13).astype(np.float32), 1.2)
    M = np.zeros((H_, W_), np.float32)
    y_, x_ = int(round((h - screen.shape[0]) / 2)), int(round((w - screen.shape[1]) / 2))
    M[y_:y_ + screen.shape[0], x_:x_ + screen.shape[1]] = screen
    dy_, dx_ = np.gradient(gaussian_filter(M, 3.5))
    bezel = 0.058 + 0.008 * field(H_, W_, 1.0, 141) + 1.3 * np.clip(-(0.6 * dx_ + 0.8 * dy_), -0.2, 0.3)
    return tube * M[..., None] + bezel[..., None] * np.array((1.0, 1.0, 1.06), np.float32) * (1 - M[..., None])


NEON = {
    "O": [[(0.5, 0), (1, 0), (1, 1), (0, 1), (0, 0), (0.5, 0)]], "P": [[(0, 0), (0, 1), (1, 1), (1, 0.44), (0, 0.44)]],
    "E": [[(1, 1), (0, 1), (0, 0), (1, 0)], [(0, 0.52), (0.84, 0.52)]], "N": [[(0, 0), (0, 1), (1, 0), (1, 1)]],
    "2": [[(0, 0.7), (0, 1), (1, 1), (1, 0.56), (0, 0), (1, 0)]], "4": [[(0.78, 0), (0.78, 1), (0, 0.3), (1, 0.3)]],
    "H": [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(0, 0.52), (1, 0.52)]], "U": [[(0, 1), (0, 0), (1, 0), (1, 1)]],
    "R": [[(0, 0), (0, 1), (1, 1), (1, 0.48), (0, 0.48)], [(0.45, 0.48), (1, 0)]],
    "S": [[(1, 0.78), (1, 1), (0, 1), (0, 0.52), (1, 0.52), (1, 0), (0, 0), (0, 0.22)]],
}


def bent(k, pts, r):
    """A line of glass: every corner eased round the radius a tube can take."""
    P = [np.array(p, np.float64) for p in pts]
    k.move_to(*P[0])
    for a, b, d in zip(P, P[1:], P[2:]):
        ra, rd = min(r, np.linalg.norm(a - b) / 2), min(r, np.linalg.norm(d - b) / 2)
        p0, p1 = b + (a - b) / np.linalg.norm(a - b) * ra, b + (d - b) / np.linalg.norm(d - b) * rd
        k.line_to(*p0)
        k.curve_to(*(p0 + (b - p0) * 0.6), *(p1 + (b - p1) * 0.6), *p1)
    k.line_to(*P[-1])


def label_open(c, w, h):
    """A real neon sign: one bent tube per colour, the jumps between letters painted out, clipped off a brick wall that takes its light from the gas."""
    H_, W_ = int(h), int(w)
    big, small = h * 0.46, h * 0.1
    sizes = {"O": 0.8, "P": 0.6, "E": 0.56, "N": 0.74}

    def layout(body, x, base, tall, widths, gap):
        places = []
        for ch in body:
            wd = widths.get(ch, 0.62) * tall
            if ch != " ":
                places.append((ch, x, wd))
            x += wd + gap * tall
        return places

    total = sum(sizes.values()) * big + 3 * 0.15 * big
    run = layout("OPEN", (w - total) / 2, h * 0.63, big, sizes, 0.15)
    tiny = "24 HOURS"
    total = sum(0.62 * small if ch != " " else 0.62 * small for ch in tiny) + 7 * 0.34 * small
    fine = layout(tiny, (w - total) / 2, h * 0.84, small, {}, 0.34)

    def letters(k, places, base, tall, bend):
        for ch, x, wd in places:
            for stroke in NEON[ch]:
                r_ = wd * 0.5 if ch == "O" else bend
                bent(k, [(x + a * wd, base - b * tall) for a, b in stroke], r_)
                k.stroke()

    def tubes_big(k, lw):
        k.set_line_width(lw)
        k.set_line_cap(cairo.LINE_CAP_ROUND)
        k.set_line_join(cairo.LINE_JOIN_ROUND)
        letters(k, run, h * 0.63, big, big * 0.13)

    def tubes_small(k, lw):
        k.set_line_width(lw)
        k.set_line_cap(cairo.LINE_CAP_ROUND)
        letters(k, fine, h * 0.84, small, small * 0.2)
        x0, y0, x1, y1 = w * 0.05, h * 0.09, w * 0.95, h * 0.91
        bent(k, [(w * 0.8, y1), (x0, y1), (x0, y0), (x1, y0), (x1, y1), (w * 0.84, y1)], h * 0.12)   # the border, one tube with its two ends at the foot
        k.stroke()

    def painted_out(k):
        k.set_line_width(h * 0.017)
        k.set_line_cap(cairo.LINE_CAP_ROUND)
        y = h * 0.635
        for (ch, x, wd), (_, x2, _) in zip(run, run[1:]):
            k.move_to(x + wd * (0.5 if ch == "O" else 0.0 if ch == "P" else 1.0), y)
            k.line_to(x2, y)
            k.stroke()
        k.set_line_width(h * 0.016)
        for x in (0.8, 0.84):
            k.move_to(w * x, h * 0.91)
            k.line_to(w * x, h * 0.965)
            k.stroke()

    # the wall: fired brick in running bond, mortar sunk in, each brick its own
    rnd = np.random.default_rng(151)
    yy, xx = grid(H_, W_)
    course = h / 8.0
    long_ = course * 2.2
    row = np.floor(yy / course).astype(np.int64)
    col = np.floor(xx / long_ + 0.5 * (row % 2)).astype(np.int64)
    tints = rnd.uniform(0.72, 1.18, (32, 64)).astype(np.float32)
    hues = rnd.uniform(-0.05, 0.05, (32, 64)).astype(np.float32)
    mx = np.abs((xx / long_ + 0.5 * (row % 2)) % 1 - 0.5) * long_
    my = np.abs((yy / course) % 1 - 0.5) * course
    joint = np.clip(np.minimum(long_ / 2 - mx, course / 2 - my) / (h * 0.009), 0, 1)
    tone = tints[row % 32, col % 64] * (1 + 0.035 * field(H_, W_, 1.5, 153) + 0.05 * field(H_, W_, 7, 154))
    wall = np.stack([tone * (0.42 + hues[row % 32, col % 64]), tone * 0.215, tone * 0.17], axis=-1).astype(np.float32)
    wall = wall * (0.34 + 0.66 * joint)[..., None]

    pink_m = cover(w, h, lambda k: tubes_big(k, h * 0.026))
    cyan_m = cover(w, h, lambda k: tubes_small(k, h * 0.016))
    pink, cyan = np.array((1.0, 0.09, 0.38), np.float32), np.array((0.06, 0.82, 1.0), np.float32)

    def throw(m, near, far):
        a, b = gaussian_filter(m, h * near), gaussian_filter(m, h * far)
        return a / a.max(), b / b.max()

    pa, pb = throw(pink_m, 0.022, 0.13)
    ca, cb = throw(cyan_m, 0.02, 0.07)
    light = 0.05 + (pb * 1.5 + pa * 0.85)[..., None] * pink * 2.0 + (cb * 1.2 + ca * 0.6)[..., None] * cyan * 1.6
    art = wall * light
    art = art + (pa ** 1.6 * 0.5)[..., None] * pink + (ca ** 1.6 * 0.4)[..., None] * cyan      # the haze round the glass
    sh = gaussian_filter(np.roll(pink_m + cyan_m, (int(h * 0.012), int(h * 0.006)), axis=(0, 1)), h * 0.005)
    art = art * (1 - np.clip(sh, 0, 1)[..., None] * 0.35)

    drawn = cover(w, h, painted_out)
    art = lay(art, drawn, "#120F11")
    art = art + (np.clip(drawn - np.roll(drawn, int(h * 0.009), axis=0), 0, 1) * 0.1)[..., None]
    clips = [(x + wd * a, h * 0.63 - big * b) for ch, x, wd in run for a, b in ((0.0, 0.5), (1.0, 0.55)) if ch != "E" or a == 0.0]
    clips += [(w * 0.05, h * 0.5), (w * 0.95, h * 0.5), (w * 0.5, h * 0.09), (w * 0.3, h * 0.91)]

    def stand(k):
        for x, y in clips:
            k.arc(x, y, h * 0.0105, 0, 2 * math.pi)
            k.fill()

    for m, core_w, col in ((pink_m, h * 0.011, pink), (cyan_m, h * 0.007, cyan)):
        draw = (lambda k: tubes_big(k, core_w)) if m is pink_m else (lambda k: tubes_small(k, core_w))
        core = gaussian_filter(cover(w, h, draw), 1.2)
        glass = gaussian_filter(m, 1.0)
        tube = col * 1.25 + (1 - col) * 0.2
        art = art * (1 - glass[..., None]) + tube * glass[..., None]
        art = art + core[..., None] * (1 - col * 0.25) * 0.95
    feet = cover(w, h, stand)
    art = art * (1 - feet[..., None] * 0.62)                                                 # the glass standoffs, dark against their tube
    return art


def label_chrome(c, w, h):
    """1984 in airbrushed chrome: a real bevel reflecting sky and ground, an extruded edge, glints, over the striped sun and the glowing grid."""
    H_, W_ = int(h), int(w)
    yy, xx = grid(H_, W_)
    horizon = h * 0.71
    art = ramp(yy / horizon, [(0, "#0C0426"), (0.55, "#2A0E5C"), (0.86, "#7C1D8E"), (1, "#E42F8C")])
    sx, sy, sr = w / 2, h * 0.46, h * 0.37
    disc = np.clip((sr - np.hypot(xx - sx, yy - sy)) / 1.5, 0, 1) * (yy < horizon)
    for i in range(6):
        y0, y1 = h * (0.46 + i * 0.044), h * (0.46 + i * 0.044 + 0.005 + i * 0.0052)
        disc = disc * (1 - np.clip((yy - y0) / 1.2, 0, 1) * np.clip((y1 - yy) / 1.2, 0, 1))
    art = art + gaussian_filter(np.clip((sr - np.hypot(xx - sx, yy - sy)) / 1.5, 0, 1), h * 0.055)[..., None] * np.array((0.55, 0.14, 0.26), np.float32) * (yy < horizon)[..., None]
    sun = ramp((yy - (sy - sr)) / (horizon - sy + sr), [(0, "#FFF2A6"), (0.45, "#FFA63E"), (1, "#FF1E82")])
    art = art * (1 - disc[..., None]) + sun * disc[..., None]
    ground = ramp((yy - horizon) / (h - horizon), [(0, "#1E0740"), (1, "#070212")])
    art = np.where((yy >= horizon)[..., None], ground, art)

    def lines(k):
        k.set_line_width(h * 0.0065)
        for i in range(-11, 12):
            k.move_to(w / 2 + i * w * 0.035, horizon)
            k.line_to(w / 2 + i * w * 0.24, h)
            k.stroke()
        for j in range(1, 9):
            y = horizon + (h - horizon) * (j / 8) ** 2
            k.move_to(0, y)
            k.line_to(w, y)
            k.stroke()

    grid_m = cover(w, h, lines)
    art = art + (grid_m * 0.95 + gaussian_filter(grid_m, 3.5) * 1.1)[..., None] * np.array((0.95, 0.24, 0.92), np.float32) * np.clip((yy - horizon) / (h * 0.1) + 0.25, 0, 1)[..., None]

    def word(k):
        font(k, FUTURA, h * 0.55, True)
        tw = k.text_extents("1984").x_advance
        k.transform(cairo.Matrix(1, 0, -0.14, 1, h * 0.76 * 0.14, 0))
        k.move_to((w - tw) / 2, h * 0.76)
        k.show_text("1984")

    T = cover(w, h, word)
    inside = T > 0.5
    r = h * 0.042
    top = np.nonzero(inside.any(axis=1))[0]
    y0, y1 = float(top[0]), float(top[-1])
    H = np.clip((gaussian_filter(T, r * 0.42) - 0.5) * 2, 0, 1)
    H = gaussian_filter(1 - (1 - H) ** 2, 0.8)
    gy, gx = np.gradient(H)
    v = (yy - y0) / (y1 - y0) - gy * r * 0.34 - gx * r * 0.12 + 0.008 * np.sin(xx * 0.05) + 0.005 * field(H_, W_, 18, 161)
    chrome = ramp(v, [(-0.45, "#21389A"), (0.0, "#3F7FE6"), (0.28, "#A9E4FF"), (0.485, "#F2FBFF"), (0.5, "#1A0F4A"), (0.57, "#40206E"), (0.74, "#FF72C2"), (0.92, "#FFD3EC"), (1.4, "#FFF6FB")])
    spec = np.clip((0.6 * gx + 0.8 * gy) * r * 0.9, 0, 1) ** 2.6
    chrome = chrome + spec[..., None] * 0.85

    def pulled(k):
        for i in range(2, 2 * int(h * 0.03)):
            k.save()
            k.translate(i * 0.36, i * 0.5)
            word(k)
            k.restore()
            k.new_path()

    ext = cover(w, h, pulled)
    art = lay(art, gaussian_filter(np.roll(ext, (int(h * 0.012), int(h * 0.008)), axis=(0, 1)), h * 0.012) * 0.65, "#050112")
    side = ramp((yy - y0) / (y1 - y0), [(0, "#FF3AA0"), (0.55, "#6A1AA8"), (1, "#1A0848")])
    art = art * (1 - ext[..., None]) + side * ext[..., None]
    ring = np.clip((h * 0.008 - distance_transform_edt(~inside).astype(np.float32)) / 1.5, 0, 1)
    art = lay(art, ring, "#0A0326")
    art = art * (1 - T[..., None]) + chrome * T[..., None]
    for x, y, s_ in ((w * 0.335, y0 + h * 0.012, 1.0), (w * 0.655, y0 + h * 0.01, 0.7)):    # the two stars every airbrush artist left on the chrome
        dx, dy = np.abs(xx - x), np.abs(yy - y)
        star = np.exp(-(dx / (h * 0.085 * s_)) ** 2 - (dy / 1.2) ** 2) + np.exp(-(dy / (h * 0.085 * s_)) ** 2 - (dx / 1.2) ** 2) + np.exp(-(dx ** 2 + dy ** 2) / (h * 0.012 * s_) ** 2)
        art = art + star[..., None] * 0.95
    return art * (1 + 0.016 * field(H_, W_, 0.8, 163))[..., None]


def container_axes(w, h):
    k = min(w * 0.96 / 320, h * 0.96 / 275)
    return (250 * k, -110 * k), (-70 * k, -40 * k), 62 * k, (w - 320 * k) / 2 + 70 * k, h - (h - 275 * k) / 2, k


def die_container(h, w):
    L, S, U, x, y, _ = container_axes(w, h)
    up = 2.02 * U

    def cut(k):
        poly(k, [(x, y), (x + L[0], y + L[1]), (x + L[0], y + L[1] - up), (x + L[0] + S[0], y + L[1] + S[1] - up), (x + S[0], y + S[1] - up), (x + S[0], y + S[1])])
        k.fill()
    return cover(w, h, cut) > 0.5


def label_container(c, w, h):
    """Two forty-foot boxes, stacked: corrugated steel with the paint on the ribs, corner castings, locking bars, an ISO owner code, and a working life of rust."""
    H_, W_ = int(h), int(w)
    L, S, U, x, y, k = container_axes(w, h)
    c.set_source_rgb(0.9, 0.9, 0.9)
    c.paint()

    def box(ox, oy, side, end, top, paint):
        def pt(a, b, u):
            return (ox + L[0] * a + S[0] * b, oy + L[1] * a + S[1] * b - U * u)

        def quad(a0, a1, u0, u1, col, al=1.0):
            poly(c, [pt(a0, 0, u0), pt(a1, 0, u0), pt(a1, 0, u1), pt(a0, 0, u1)])
            c.set_source_rgba(*hx(col), al)
            c.fill()

        def door(b0, b1, u0, u1, col, al=1.0):
            poly(c, [pt(0, b0, u0), pt(0, b1, u0), pt(0, b1, u1), pt(0, b0, u1)])
            c.set_source_rgba(*hx(col), al)
            c.fill()

        quad(0, 1, 0, 1, side)
        n = math.hypot(*L)
        paint(c, pt, n)
        ribs = 26                                          # paint first, then the corrugation pressed through it
        for i in range(ribs):
            a0 = 0.02 + i / ribs * 0.96
            da = 0.96 / ribs
            quad(a0 + da * 0.22, a0 + da * 0.34, 0.06, 0.94, "#000000", 0.34)
            quad(a0 + da * 0.34, a0 + da * 0.74, 0.06, 0.94, "#000000", 0.13)
            quad(a0 + da * 0.74, a0 + da * 0.86, 0.06, 0.94, "#FFFFFF", 0.16)
        quad(0, 1, 0.94, 1, "#FFFFFF", 0.2)                # top rail
        quad(0, 1, 0, 0.06, "#000000", 0.32)               # bottom rail
        for a0 in (0, 0.975):                              # corner posts
            quad(a0, a0 + 0.025, 0, 1, "#000000", 0.26)
        door(0, 1, 0, 1, end)
        for b0 in (0.0, 0.5):                              # two door leaves, ribbed
            for i in range(5):
                b = b0 + 0.03 + i * 0.094
                door(b, b + 0.04, 0.1, 0.9, "#000000", 0.17)
        door(0.49, 0.51, 0.02, 0.98, "#000000", 0.75)
        door(0, 1, 0.94, 1, "#FFFFFF", 0.14)
        door(0, 1, 0, 0.06, "#000000", 0.3)
        for b in (0.17, 0.35, 0.65, 0.83):                 # locking bars, each with its cam handle
            door(b - 0.012, b + 0.012, 0.03, 0.97, "#E2E4E7")
            door(b - 0.012, b + 0.09, 0.42, 0.46, "#C9CBD0")
        poly(c, [pt(0, 0, 1), pt(1, 0, 1), pt(1, 1, 1), pt(0, 1, 1)])
        c.set_source_rgb(*hx(top))
        c.fill()
        for i in range(1, 30):                             # the roof's shallow pressings
            a = i / 30
            poly(c, [pt(a, 0.04, 1), pt(a, 0.96, 1)], close=False)
            c.set_line_width(k * 1.5)
            c.set_source_rgba(0, 0, 0, 0.13 if i % 2 else 0.05)
            c.stroke()
        for a, b, u in ((0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 1, 0)):  # the castings at each corner
            ax, ay = pt(a, b, u)
            s_ = k * 4.8
            rrect(c, ax - s_, ay - s_, 2 * s_, 2 * s_, k * 1.2)
            c.set_source_rgb(*hx("#3B3C40"))
            c.fill()
            c.arc(ax, ay, k * 1.5, 0, 2 * math.pi)
            c.set_source_rgb(*hx("#0B0B0D"))
            c.fill()

    def on_side(c_, pt, n, a, u, body, size, face, col="#F6F4EC"):
        c_.save()
        ox, oy = pt(a, 0, u)
        c_.transform(cairo.Matrix(L[0] / n, L[1] / n, 0, 1, ox, oy))
        font(c_, face, size, True)
        c_.set_source_rgb(*hx(col))
        c_.move_to(0, 0)
        c_.show_text(body)
        c_.restore()
        c_.new_path()

    def lower(c_, pt, n):
        c_.save()                                          # the owner code, its check digit in the box the standard asks for
        ox, oy = pt(0.07, 0, 0.27)
        c_.transform(cairo.Matrix(L[0] / n, L[1] / n, 0, 1, ox, oy))
        font(c_, "Avenir Next Condensed", U * 0.54, True)
        c_.set_source_rgb(*hx("#F6F4EC"))
        c_.move_to(0, 0)
        c_.show_text("XGHU 198430")
        adv = c_.text_extents("XGHU 198430").x_advance + U * 0.14
        c_.move_to(adv, 0)
        c_.show_text("8")
        wd = c_.text_extents("8").x_advance
        c_.set_line_width(k * 1.3)
        c_.rectangle(adv - U * 0.06, U * 0.07, wd + U * 0.12, -U * 0.54)
        c_.stroke()
        c_.restore()

    def upper(c_, pt, n):
        on_side(c_, pt, n, 0.07, 0.2, "XINGHE", U * 0.84, "Avenir Next Condensed")
        on_side(c_, pt, n, 0.74, 0.66, "XGHU 240413 6", U * 0.13, "DIN Alternate")
        on_side(c_, pt, n, 0.74, 0.46, "45G1", U * 0.13, "DIN Alternate")

    box(x, y, "#1C4B9A", "#143974", "#3268BA", lower)
    box(x, y - U * 1.02, "#E4561A", "#AF400E", "#F27C42", upper)
    art = rgb_of(c.get_target()).copy()
    yy, xx = grid(H_, W_)
    # rust runs down from the rails and castings, dirt collects low, the paint fades in patches
    up = U * 2.02
    roof = cover(w, h, lambda k_: (poly(k_, [(x, y - up), (x + L[0], y + L[1] - up), (x + L[0] + S[0], y + L[1] + S[1] - up), (x + S[0], y + S[1] - up)]), k_.fill()))
    runs = np.clip(field(H_, W_, 26, 171, 1.5) - 0.9, 0, 1) * np.clip(field(H_, W_, 50, 172) * 0.8 + 0.9, 0, 1) * (1 - roof)
    runs = np.maximum(runs, np.clip(field(H_, W_, 5, 177) * 0.7 + field(H_, W_, 22, 178) * 0.7 - 1.5, 0, 1) * roof)
    art = art * (1 - (runs * 0.58)[..., None] * np.array((0.28, 0.5, 0.78), np.float32))
    art = art * (1 + (0.075 * field(H_, W_, 22, 173) + 0.018 * field(H_, W_, 0.9, 174))[..., None])
    chips = np.clip((field(H_, W_, 1.3, 175) - 2.5) * 3, 0, 1) * np.clip(field(H_, W_, 30, 176) + 0.3, 0, 1)
    return lay(art, chips, "#6E4A34")


def label_appicon(c, w, h):
    """A weather app's icon: a glowing sun behind a soft cloud on the system's blue. No rays: icons have not drawn rays for years."""
    H_, W_ = int(h), int(w)
    yy, xx = grid(H_, W_)
    art = ramp(yy / h, [(0, "#1562EE"), (1, "#74BEFF")])
    x, y, r = w * 0.635, h * 0.355, h * 0.215
    d = np.hypot(xx - x, yy - y)
    art = art + (np.exp(-np.clip(d - r, 0, None) / (h * 0.085)) * 0.6)[..., None] * np.array((1.0, 0.74, 0.3), np.float32)
    sun = np.clip((r - d) / 1.5, 0, 1)
    art = art * (1 - sun[..., None]) + ramp(np.clip(((xx - x) * -0.3 + (yy - y) * 0.95) / r * 0.5 + 0.5, 0, 1), [(0, "#FFEC8E"), (1, "#FFAF1E")]) * sun[..., None]

    def cloud(k):
        for cx, cy, rr in ((0.27, 0.68, 0.14), (0.44, 0.55, 0.19), (0.63, 0.62, 0.15), (0.76, 0.7, 0.115)):
            k.new_sub_path()
            k.arc(w * cx, h * cy, h * rr, 0, 2 * math.pi)
        rrect(k, w * 0.15, h * 0.62, w * 0.72, h * 0.2, h * 0.1)
        k.fill()

    m = cover(w, h, cloud)
    art = art * (1 - (gaussian_filter(np.roll(m, int(h * 0.03), axis=0), h * 0.035) * 0.34)[..., None] * np.array((0.8, 0.55, 0.3), np.float32))
    body = ramp((yy - h * 0.36) / (h * 0.46), [(0, "#FFFFFF"), (0.55, "#F2F7FF"), (1, "#C8DDFB")])
    shade = gaussian_filter(np.roll(m, int(-h * 0.03), axis=0) * m, h * 0.02)  # a little weight along the underside
    body = body * (0.93 + 0.07 * np.clip(shade, 0, 1))[..., None]
    return art * (1 - m[..., None]) + body * m[..., None]


LABELS = [
    ("hazard", label_hazard, 2.7, 1.4, "round", dict(gloss=0.4)),
    ("cert", label_cert, 2.0, 1.2, "round", dict(gloss=0.0)),
    ("barcode", label_barcode, 1.9, 1.05, "round", dict(gloss=0.0)),
    ("chip", label_chip, 1.55, 1.55, "round", dict(gloss=0.8)),
    ("clapper", label_clapper, 1.8, 1.6, die_clapper, dict(gloss=0.5)),
    ("seal", label_seal, 1.25, 1.25, "round", dict(gloss=0.0)),
    ("patch", label_patch, 1.95, 1.95, "oval", dict(gloss=0.2)),
    ("coin", label_coin, 2.4, 0.86, "round", dict(gloss=0.6)),
    ("open", label_open, 2.2, 1.15, "round", dict(gloss=0.8)),
    ("chrome", label_chrome, 2.1, 1.25, "ticket", dict(gloss=1.0, holo_rim=True)),
    ("container", label_container, 1.75, 1.45, die_container, dict(gloss=0.4)),
    ("appicon", label_appicon, 1.3, 1.3, "squircle", dict(gloss=1.0)),
]


# ======================================================================================
# from a print and its die line to files
# ======================================================================================
def build(name, art, mask, width, opts):
    """art: float (h, w, 3) print; mask: bool, where the print stops. The white margin grows out from the mask."""
    h, w = mask.shape
    ppu = w / width
    b = int(round(BORDER * ppu))
    pad = b + 6
    art = np.pad(art, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    mask = np.pad(mask, pad)
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    phi = np.where(mask, inside, -outside) + b             # 0 on the die line, positive inside the sticker
    phi = gaussian_filter(phi.astype(np.float32), 2.0)
    blend = np.clip(gaussian_filter(mask.astype(np.float32), 0.8), 0, 1)[..., None]
    white = np.array((0.965, 0.96, 0.945), np.float32)
    colour = art * blend + white * (1 - blend)
    foil = np.zeros(mask.shape + (3,), np.float32)
    if opts.get("gold"):
        foil[..., 0] = gold_mask(art) * blend[..., 0]
    if opts.get("holo_rim"):
        foil[..., 1] = 1 - blend[..., 0]
    if opts.get("holo_art"):
        foil[..., 1] = np.maximum(foil[..., 1], np.clip(art.max(axis=2) * 1.1, 0, 1) ** 1.8 * 0.2 * blend[..., 0])
    Image.fromarray((np.clip(colour, 0, 1) * 255).astype(np.uint8)).save(os.path.join(CACHE, f"{name}_colour.png"))
    Image.fromarray((foil * 255).astype(np.uint8)).save(os.path.join(CACHE, f"{name}_foil.png"))

    # marching squares over phi: square interior cells stay quads, and the outline runs along the zero level exactly
    step = max(3, int(round((0.02 if opts.get("press") else 0.045) * ppu)))
    H_, W_ = phi.shape
    xs = np.arange(0, W_ - 1, step)
    ys = np.arange(0, H_ - 1, step)
    v = phi[np.ix_(ys, xs)]
    index, verts, faces, count = {}, [], [], []

    def corner(i, j):
        key = (i, j)
        if key not in index:
            index[key] = len(verts)
            verts.append((xs[j], ys[i]))
        return index[key]

    def edge(i0, j0, i1, j1):
        key = (min(i0, i1), min(j0, j1), i0 != i1)
        if key not in index:
            a, b_ = v[i0, j0], v[i1, j1]
            k = min(0.98, max(0.02, a / (a - b_)))
            index[key] = len(verts)
            verts.append((xs[j0] + (xs[j1] - xs[j0]) * k, ys[i0] + (ys[i1] - ys[i0]) * k))
        return index[key]

    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            ring = ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))
            flags = [v[p] > 0 for p in ring]
            if not any(flags):
                continue
            poly_ = []
            for k in range(4):
                p, q = ring[k], ring[(k + 1) % 4]
                if flags[k]:
                    poly_.append(corner(*p))
                if flags[k] != flags[(k + 1) % 4]:
                    poly_.append(edge(*p, *q))
            if len(set(poly_)) >= 3:
                faces.extend(poly_)
                count.append(len(poly_))
    verts = np.array(verts, np.float32)
    uv = np.column_stack((verts[:, 0] / W_, 1 - verts[:, 1] / H_))
    local = np.column_stack((verts[:, 0] - W_ / 2, H_ / 2 - verts[:, 1])) / ppu
    keep = np.ones(len(local), bool)
    return dict(name=name, local=local, uv=uv, faces=np.array(faces, np.int32), count=np.array(count, np.int32),
                keep=keep, size=(W_ / ppu, H_ / ppu), opts=opts, phi=phi, ppu=ppu)


def print_of(name, clip, t, crop, width, die, o):
    x0, y0, x1, y1 = crop
    ppu = o.get("ppu", PPU)
    w = int(round(width * ppu))
    h = int(round(w * (y1 - y0) / (x1 - x0)))
    im = frame(clip, t).crop(crop).resize((w, h), Image.NEAREST if o.get("nearest") else Image.LANCZOS)
    art = np.asarray(im, np.float32) / 255
    mask = lift(art, o["bg"], o["tau"], o["pad"] * ppu / 230, o.get("trim", 0.22)) if die == "lift" else shape_mask(die, h, w)
    return build(name, art, mask, width, o)


def label_art(fn, wd, ht):
    w, h = int(round(wd * LABEL_PPU)), int(round(ht * LABEL_PPU))
    s, c = canvas(None, w, h)
    art = fn(c, float(w), float(h))
    return (rgb_of(s).copy() if art is None else np.clip(art, 0, 1)).astype(np.float32)


def type_print(name, fn, wd, ht, die, o):
    art = label_art(fn, wd, ht)
    return build(name, art, shape_mask(die, *art.shape[:2]), wd, o)


# ======================================================================================
# laying the lot over the lid
# ======================================================================================
# where each one lands on the lid (x, y in units from its centre, degrees), placed by hand so that every subject stays readable
PLACE = {
    "koi": (-2.75, -2.18, -6), "lorenz": (0.0, 0.2, 5),
    "hud": (-5.05, 3.08, 6), "dusk": (-1.55, 3.3, -4), "weather": (1.95, 3.3, 5), "rocket": (5.35, 2.75, -9),
    "crown": (-4.75, 0.95, -8), "lock": (-2.55, 1.38, -3), "shan": (-3.0, -0.7, 6),
    "wave": (4.25, 0.35, 5), "port": (5.0, -1.95, -5), "ship": (5.0, -3.36, 4),
    "dash": (2.0, -2.05, -4), "type": (2.95, -3.18, 8), "car": (2.6, 1.95, -6),
    "hazard": (-0.2, 2.2, -3), "open": (3.85, 3.55, -6), "appicon": (-3.55, 3.0, 10),
    "clapper": (3.05, -1.05, 8), "chip": (5.65, -0.45, 7), "seal": (-0.35, -2.02, -8),
    "patch": (-5.5, -0.85, -10), "coin": (-5.05, -3.8, -2), "cert": (-2.9, -3.62, 4),
    "barcode": (-0.85, -3.7, -3), "chrome": (1.05, -3.55, -3), "container": (4.05, -2.6, -6),
}
# bottom to top: the large prints first and slowly, then the rhythm picks up with the labels on top of what they overlap
ORDER = ["koi", "dash", "lock", "crown", "shan", "hud", "dusk", "weather", "rocket", "port", "ship", "type", "wave", "car", "hazard", "open",
         "appicon", "clapper", "chip", "patch", "coin", "cert", "barcode", "chrome", "container", "seal", "lorenz"]


def drape(stickers, placed, order):
    """Every sticker lies over whatever is already there: its edges raise a real ridge in the ones stuck on after it."""
    ppu = 100
    H_, W_ = int(LID_H * ppu), int(LID_W * ppu)
    height = np.zeros((H_, W_), np.float32)
    for name in order:
        st = stickers[name]
        cx, cy, deg = placed[name]
        a = math.radians(deg)
        p = st["local"] @ np.array(((math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))), np.float32)
        world = p + np.array((cx, cy), np.float32)
        under = gaussian_filter(maximum_filter(height, 3), 1.2)
        col = np.clip(((world[:, 0] + LID_W / 2) * ppu).astype(int), 0, W_ - 1)
        row = np.clip(((LID_H / 2 - world[:, 1]) * ppu).astype(int), 0, H_ - 1)
        st["z"] = under[row, col] + THICK + 0.0015
        st["world"] = world
        k = ppu / st["ppu"]
        big = Image.fromarray(((st["phi"] > 0) * 255).astype(np.uint8))
        big = big.resize((max(2, int(big.width * k)), max(2, int(big.height * k))), Image.BILINEAR)
        m = np.asarray(big.rotate(deg, expand=True, resample=Image.BILINEAR), np.float32) / 255 > 0.5
        y0 = int(round((LID_H / 2 - cy) * ppu - m.shape[0] / 2))
        x0 = int(round((cx + LID_W / 2) * ppu - m.shape[1] / 2))
        ya, xa = max(0, y0), max(0, x0)
        yb, xb = min(H_, y0 + m.shape[0]), min(W_, x0 + m.shape[1])
        crop = m[ya - y0:yb - y0, xa - x0:xb - x0]
        height[ya:yb, xa:xb] = np.where(crop, under[ya:yb, xa:xb] + THICK + 0.0015, height[ya:yb, xa:xb])


def main():
    os.makedirs(SRC, exist_ok=True)
    stickers = {}
    for spec in ART:
        st = print_of(*spec)
        stickers[st["name"]] = st
        print(spec[0], len(st["local"]), "verts", flush=True)
    for spec in LABELS:
        st = type_print(*spec)
        stickers[st["name"]] = st
    drape(stickers, PLACE, ORDER)
    sheet = []
    for name in ORDER:
        st = stickers[name]
        np.savez_compressed(os.path.join(CACHE, f"{name}.npz"), world=st["world"].astype(np.float32),
                            z=st["z"].astype(np.float32), uv=st["uv"].astype(np.float32),
                            faces=st["faces"], count=st["count"])
        o = st["opts"]
        sheet.append(dict(name=name, size=st["size"], at=PLACE[name], gloss=o.get("gloss", 0.8),
                          gold=bool(o.get("gold")), holo=bool(o.get("holo_rim") or o.get("holo_art"))))
    with open(os.path.join(CACHE, "sheet.json"), "w") as fh:
        json.dump(dict(stickers=sheet, lid=(LID_W, LID_H), thick=THICK), fh)
    preview(sheet)
    print(f"{len(ORDER)} stickers", flush=True)


def preview(sheet):
    """A flat top view, to judge the layout before any render."""
    ppu = 100
    im = Image.new("RGB", (int(LID_W * ppu), int(LID_H * ppu)), (38, 40, 44))
    for s in sheet:
        c = Image.open(os.path.join(CACHE, s["name"] + "_colour.png")).convert("RGBA")
        d = np.load(os.path.join(CACHE, s["name"] + ".npz"))
        w, h = int(s["size"][0] * ppu), int(s["size"][1] * ppu)
        c = c.resize((w, h), Image.BILINEAR)
        mesh = Image.new("L", (w, h), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mesh)
        uv = d["uv"]
        at = 0
        for n in d["count"]:
            pts = [(uv[i, 0] * w, (1 - uv[i, 1]) * h) for i in d["faces"][at:at + n]]
            draw.polygon(pts, fill=255)
            at += n
        c.putalpha(mesh)
        c = c.rotate(s["at"][2], expand=True, resample=Image.BILINEAR)
        im.paste(c, (int((s["at"][0] + LID_W / 2) * ppu - c.width / 2), int((LID_H / 2 - s["at"][1]) * ppu - c.height / 2)), c)
    im.save(os.path.join(CACHE, "layout.png"))


if __name__ == "__main__":
    main()
