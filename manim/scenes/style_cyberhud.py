"""The HUD over the rendered flight: rain, bloom, then vector instruments that track the 3D world frame by frame.

    python style_cyberhud.py signs      makes the neon signs the Blender scene hangs in its streets
"""
import json
import math
import os
import random
import sys
from functools import lru_cache

import cairo
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from styles_common import (FUTURA, H, OUT, SANS, W, add, canvas, font, glow, hx, poly, rgb_of, rrect, seg, seq, smooth,
                           surface, text)

CACHE = os.path.join(OUT, "cache_cyberhud")
LINE, TXT, AMB, LOCK, INK = "#A6F1FF", "#EAFBFF", "#FFC857", "#FF2D3D", "#04070D"
NUM = "DIN Alternate"


# ======================================================================================
# the signs
# ======================================================================================
TUBES = ["#FF2D78", "#19F0FF", "#FFB000", "#7CFF4F", "#B56BFF", "#FF4B2B"]
UPRIGHT = ["拉面", "当铺", "药房", "夜市", "酒吧", "旅馆", "电玩", "回收", "维修", "义体", "黑市", "数据", "算力", "浴场", "典当", "寿司"]
WIDE = ["脑机接口", "算力租赁", "24H 营业", "义体维修", "霓虹夜市", "数据回收"]


ENGLISH = ["RAMEN", "PAWN", "PHARMACY", "NIGHT MARKET", "BAR", "HOTEL", "ARCADE", "RECYCLE", "REPAIR", "CYBERWARE", "BLACK MARKET",
           "DATA", "COMPUTE", "BATHS", "LOANS", "SUSHI"]


def tube_mask(w, h, draw):
    s = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
    draw(cairo.Context(s))
    s.flush()
    return np.frombuffer(s.get_data(), np.uint8).reshape(h, s.get_stride())[:, :w].astype(np.float32) / 255


def glass(rgb, mask, tube, core=0.62):
    """Lit glass tube: a wide soft halo, a tight saturated one, a nearly white core. Blurred, so there are no stroke corners."""
    col = np.array(hx(tube), np.float32)
    wide, tight = gaussian_filter(mask, mask.shape[1] * 0.045), gaussian_filter(mask, mask.shape[1] * 0.012)
    out = rgb + col * (wide * 0.8 + tight * 0.75)[..., None]
    white = col * (1 - core) + core
    out = out * (1 - mask[..., None]) + white * mask[..., None]
    return np.clip(out, 0, 1)


def glyph(word, x, y, size, face=SANS):
    def draw(c):
        c.select_font_face(face, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        c.set_font_size(size)
        ext = c.text_extents(word)
        c.move_to(x - ext.x_advance / 2, y)
        c.show_text(word)
    return draw


def save_rgb(rgb, path):
    Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def make_signs():
    os.makedirs(CACHE, exist_ok=True)
    rnd = random.Random(9)
    for i, word in enumerate(UPRIGHT):
        w, h = 512, 1304
        tube = TUBES[i % len(TUBES)]
        col = np.array(hx(tube), np.float32)
        ground = np.array(hx("#07060C"), np.float32) * 0.8 + col * 0.05
        rgb = np.broadcast_to(ground, (h, w, 3)).copy()
        def frame(c):
            c.set_source_rgba(1, 1, 1, 1)
            c.set_line_width(9)
            rrect(c, 24, 24, w - 48, h - 48, 30)
            c.stroke()
        mask = tube_mask(w, h, frame)
        for k, ch in enumerate(word):
            mask = np.maximum(mask, tube_mask(w, h, glyph(ch, w / 2, 262 + 74 + 540 * k + 150, 400)))
        rgb = glass(rgb, mask, tube)
        panel = tube_mask(w, h, lambda c: (rrect(c, 54, 54, w - 108, 88, 14), c.set_source_rgba(1, 1, 1, 1), c.fill()))
        rgb = rgb * (1 - panel[..., None]) + np.minimum(col * 1.05, 1) * panel[..., None]
        label = tube_mask(w, h, glyph(ENGLISH[i], w / 2, 114, 50 if len(ENGLISH[i]) < 10 else 40, FUTURA))
        rgb = rgb * (1 - label[..., None]) + np.array(hx("#07060C"), np.float32) * label[..., None]
        save_rgb(rgb, os.path.join(CACHE, f"sign_{i:02d}.png"))
    for i, word in enumerate(WIDE):
        w, h = 1200, 720
        tube = TUBES[(i * 2 + 1) % len(TUBES)]
        a, b = np.array(hx(tube), np.float32), np.array(hx(TUBES[(i * 2 + 3) % len(TUBES)]), np.float32)
        u = (np.arange(w)[None, :] / w + np.arange(h)[:, None] / h)[..., None] / 2
        rgb = (a * (1 - u) + b * u) * 0.3 + np.array(hx("#050409"), np.float32) * 0.7
        rows = (np.arange(h) % 26 < 11)
        rgb[rows] *= 0.62
        columns = (np.arange(w) % 8 < 2)
        rgb[:, columns] *= 0.8
        mask = tube_mask(w, h, glyph(word, w / 2, h * 0.68, 600 if len(word) == 2 else 225))
        rgb = glass(rgb, mask, "#FFFFFF" if rnd.random() < 0.3 else tube)
        save_rgb(rgb, os.path.join(CACHE, f"sign_{len(UPRIGHT) + i:02d}.png"))
    print("signs written")


# ======================================================================================
# the world, as the renderer saw it
# ======================================================================================
@lru_cache(maxsize=1)
def track():
    with open(os.path.join(CACHE, "track.json")) as fh:
        return json.load(fh)


class View:
    def __init__(self, t):
        f = max(0, min(149, int(round(t * 30))))
        cam = track()["cam"][f]
        self.c = np.array(cam["c"])
        self.r = np.array(cam["r"]).reshape(3, 3)
        self.k = cam["lens"] / 36 * W
        self.v = np.array(cam["v"])
        self.drone = np.array(cam["drone"])

    def project(self, p):
        q = self.r.T @ (np.asarray(p, float) - self.c)
        if q[2] > -1:
            return None
        return W / 2 + q[0] / -q[2] * self.k, H / 2 - q[1] / -q[2] * self.k, -q[2]

    def rect(self, x0, x1, y0, y1, z0, z1):
        """The screen rectangle of a world box, or None unless it is well in view."""
        pts = [self.project((x, y, z)) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
        if any(p is None for p in pts):
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def heading(self):
        f = -self.r[:, 2]
        return math.degrees(math.atan2(f[0], f[1])) % 360

    @property
    def pitch(self):
        return math.degrees(math.asin(-self.r[2, 2]))

    @property
    def roll(self):
        return math.degrees(math.atan2(self.r[2, 0], self.r[2, 1]))


# ======================================================================================
# the plate: rain and bloom
# ======================================================================================
def bloom(rgb):
    hot = np.clip(rgb[::2, ::2] - 0.6, 0, None) / 0.4
    b = gaussian_filter(hot, (5, 5, 0)) * 0.5 + gaussian_filter(hot, (18, 18, 0)) * 0.42 + gaussian_filter(hot, (55, 55, 0)) * 0.42
    b = np.repeat(np.repeat(b, 2, 0), 2, 1)[:rgb.shape[0], :rgb.shape[1]]
    return 1 - (1 - rgb) * (1 - np.clip(b, 0, 1))


DROPS = None
DROP_LAYERS = (                                            # width, alpha, length: far to near
    ((0.7, 0.11, 16), (1.1, 0.16, 34), (1.7, 0.22, 70)),   # the avenue
    ((0.55, 0.085, 12), (0.8, 0.11, 26), (1.4, 0.14, 64)),  # the open sky
)


def rain(c, view, t):
    """In the avenue the streaks fly out of the point we head for; once clear of it, they simply fall past us.
    Most are hair-fine and far, and only a few thick ones pass close, so the windows of the far city stay whole."""
    global DROPS
    if DROPS is None:
        rnd = np.random.default_rng(4)
        DROPS = (rnd.uniform(0, 2 * math.pi, 420), rnd.uniform(0, 1, 420), rnd.uniform(0.35, 1.6, 420), rnd.choice(3, 420, p=(0.62, 0.28, 0.10)),
                 rnd.uniform(0, W + 300, 420))
    foe = view.project(view.c + view.v)
    fx, fy = (W / 2, H / 2) if foe is None else foe[:2]
    fx, fy = min(W, max(0, fx)), min(H, max(0, fy))
    speed = np.linalg.norm(view.v) / 120
    open_air = smooth(seg(t, 2.1, 3.2))
    ang, phase, rate, layer, across = DROPS
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    for lay in range(3):
        pick = layer == lay
        if open_air < 1:
            lw, a, length = DROP_LAYERS[0][lay]
            u = (phase[pick] + t * rate[pick] * (0.7 + 0.5 * lay) * speed) % 1
            c.set_source_rgba(0.82, 0.93, 1.0, a * (1 - open_air))
            c.set_line_width(lw)
            r = 60 + (u ** 1.7) * 1500
            a0 = ang[pick] + 0.28
            ca, sa = np.cos(a0), np.sin(a0)
            x0, y0 = fx + ca * r, fy + sa * r + r * 0.12
            ln = length * (0.25 + u) * speed
            for xa, ya, xb, yb in zip(x0, y0, x0 + ca * ln, y0 + sa * ln + ln * 0.3):
                c.move_to(xa, ya)
                c.line_to(xb, yb)
            c.stroke()
        if open_air > 0:
            lw, a, length = DROP_LAYERS[1][lay]
            few = pick & (np.arange(420) < 250)
            c.set_source_rgba(0.82, 0.93, 1.0, a * open_air)
            c.set_line_width(lw)
            fall = (phase[few] + t * (1.5 + lay * 0.7)) % 1
            x0, y0 = across[few] - fall * 240, fall * (H + 160) - 80
            for xa, ya in zip(x0, y0):
                c.move_to(xa, ya)
                c.line_to(xa - length * 0.2, ya + length)
            if lay == 2:                                   # close to the lens: out of focus, so a soft edge round it
                c.stroke_preserve()
                c.set_source_rgba(0.82, 0.93, 1.0, 0.03 * open_air)
                c.set_line_width(5)
            c.stroke()


# ======================================================================================
# instruments: one pale phosphor colour, thin lines, amber for what is being tracked, red only for the lock
# ======================================================================================
def spaced(c, s, x, y, size, col=TXT, face=SANS, bold=False, align="left", track=0.0, a=1.0):
    """Baseline text with letter-spacing in ems; returns its width."""
    font(c, face, size, bold)
    adv = [c.text_extents(ch).x_advance for ch in s]
    w = sum(adv) + track * size * (len(s) - 1)
    x -= {"left": 0, "center": w / 2, "right": w}[align]
    c.set_source_rgba(*hx(col), a)
    for ch, d in zip(s, adv):
        c.move_to(x, y)
        c.show_text(ch)
        x += d + track * size
    c.new_path()
    return w


def runs(s):
    """Split into stretches of Chinese and of everything else: the instrument face has no Chinese."""
    out = []
    for ch in s:
        cjk = ord(ch) > 0x2E7F
        if out and out[-1][0] == cjk:
            out[-1][1] += ch
        else:
            out.append([cjk, ch])
    return out


def rich_width(c, s, size):
    w = 0.0
    for cjk, part in runs(s):
        font(c, SANS if cjk else NUM, size * 0.92 if cjk else size, not cjk)
        w += c.text_extents(part).x_advance
    return w


def rich(c, s, x, y, size, col, align="left", a=1.0):
    """Baseline text in the right face for each stretch."""
    x -= {"left": 0, "center": rich_width(c, s, size) / 2, "right": rich_width(c, s, size)}[align]
    c.set_source_rgba(*hx(col), a)
    for cjk, part in runs(s):
        font(c, SANS if cjk else NUM, size * 0.92 if cjk else size, not cjk)
        c.move_to(x, y)
        c.show_text(part)
        x += c.text_extents(part).x_advance
    c.new_path()


def etched(c, s, x, y, size, col, track=0.0):
    """Text with no ground, right-aligned at x: a soft dark edge keeps it legible over sky or windows."""
    parts = []
    for cjk, part in runs(s):
        face, sz = (SANS, size * 0.92) if cjk else (NUM, size)
        font(c, face, sz, not cjk)
        parts += [(face, sz, not cjk, ch, c.text_extents(ch).x_advance) for ch in part]
    x -= sum(p[4] for p in parts) + track * size * (len(parts) - 1)
    c.new_path()
    for face, sz, bold, ch, adv in parts:
        font(c, face, sz, bold)
        c.move_to(x, y)
        c.text_path(ch)
        x += adv + track * size
    c.set_line_join(cairo.LINE_JOIN_ROUND)
    for lw, a in ((7, 0.22), (3, 0.8)):
        c.set_source_rgba(*hx(INK), a)
        c.set_line_width(lw)
        c.stroke_preserve()
    c.set_source_rgb(*hx(col))
    c.fill()


def brackets(c, x0, y0, x1, y1, col, k=22, lw=1.75, a=1.0):
    c.set_source_rgba(*hx(col), a)
    c.set_line_width(lw)
    for x, y, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x1, y1, -1, -1), (x0, y1, 1, -1)):
        poly(c, [(x + sx * k, y), (x, y), (x, y + sy * k)], close=False)
        c.stroke()


def panel(c, x, y, w, h, a=0.6, col=LINE, k=9):
    """Dark ground with only its corners marked: no outline, no bevels."""
    c.rectangle(x, y, w, h)
    c.set_source_rgba(*hx(INK), a)
    c.fill()
    brackets(c, x, y, x + w, y + h, col, k, 1.25, 0.85)


def ground(c, x, y, title, value, col):
    """A small dark ground holding two lines: what it is, and its numbers. The accent runs down its left edge."""
    font(c, SANS, 18)
    w, h = max(c.text_extents(title).x_advance, rich_width(c, value, 19)) + 26, 56
    c.rectangle(x, y, w, h)
    c.set_source_rgba(*(0, 0, 0) if col == LOCK else hx(INK), 0.86 if col == LOCK else 0.66)
    c.fill()
    c.set_source_rgba(*hx(col), 1)
    c.set_line_width(1.5)
    c.move_to(x, y)
    c.line_to(x, y + h)
    c.stroke()
    text(c, title, x + 13, y + 23, 18, TXT, SANS)
    rich(c, value, x + 13, y + 48, 19, col)


def callout(c, ax, ay, lx, ly, title, value, col):
    """A leader from what it names to its ground, always arriving from the left."""
    ground(c, lx, ly - 28, title, value, col)
    c.set_source_rgba(*hx(col), 1)
    c.set_line_width(1.25)
    poly(c, [(ax, ay), (lx - 22, ly), (lx, ly)], close=False)
    c.stroke()
    c.arc(ax, ay, 2.5, 0, 2 * math.pi)
    c.fill()


def heading_tape(c, view):
    cx, y, half = W / 2, 66, 300
    h = view.heading
    c.save()
    c.rectangle(cx - half, y - 40, half * 2, 70)
    c.clip()
    for deg in range(int(h) - 60, int(h) + 61):
        if deg % 5:
            continue
        x = cx + (deg - h) * 6.2
        fade = max(0.0, 1 - (abs(x - cx) / half) ** 3) * 0.85
        major = deg % 15 == 0
        c.set_source_rgba(*hx(LINE), fade)
        c.set_line_width(1.25)
        c.move_to(x, y + 18)
        c.line_to(x, y + (6 if major else 12))
        c.stroke()
        if major and abs(x - cx) > 50:
            name = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}.get(deg % 360, f"{deg % 360:03d}")
            text(c, name, x, y - 2, 18, LINE, NUM, True, "center", fade)
    c.restore()
    c.rectangle(cx - 40, y - 28, 80, 34)
    c.set_source_rgba(*hx(INK), 0.72)
    c.fill()
    brackets(c, cx - 40, y - 28, cx + 40, y + 6, LINE, 7, 1.25)
    text(c, f"{h:03.0f}", cx, y - 3, 25, TXT, NUM, True, "center")
    poly(c, [(cx - 6, y + 28), (cx + 6, y + 28), (cx, y + 20)])
    c.set_source_rgba(*hx(TXT), 1)
    c.fill()


def ladder(c, x, value, label, unit, side, step=10, scale=4.2):
    """A scale at the edge of the picture with its reading beside it, never on top of it."""
    cy, half = H / 2, 200
    c.save()
    c.rectangle(x - 80, cy - half, 160, half * 2)
    c.clip()
    base = int(value // step) * step
    for k in range(-8, 9):
        v = base + k * step
        yy = cy + (value - v) * scale
        fade = max(0.0, 1 - (abs(yy - cy) / half) ** 2.5) * 0.9
        major = v % (step * 5) == 0
        c.set_source_rgba(*hx(LINE), fade)
        c.set_line_width(1.25)
        c.move_to(x, yy)
        c.line_to(x + side * (15 if major else 7), yy)
        c.stroke()
        if major and v >= 0:
            text(c, f"{v:.0f}", x + side * 22, yy + 6, 18, LINE, NUM, True, "left" if side > 0 else "right", fade)
    c.restore()
    w, h = 118, 58
    bx = x + 10 if side < 0 else x - 10 - w                # on the inner side of the scale
    c.rectangle(bx, cy - h / 2, w, h)
    c.set_source_rgba(*hx(INK), 0.72)
    c.fill()
    brackets(c, bx, cy - h / 2, bx + w, cy + h / 2, LINE, 8, 1.25)
    tip = x + 2 if side < 0 else x - 2
    poly(c, [(tip, cy), (tip - side * 7, cy - 5), (tip - side * 7, cy + 5)])
    c.set_source_rgba(*hx(TXT), 1)
    c.fill()
    text(c, label, bx + 11, cy - 10, 15, LINE, SANS)
    text(c, unit, bx + w - 10, cy - 10, 13, LINE, NUM, True, "right", 0.8)
    text(c, f"{value:.0f}", bx + 11, cy + 20, 28, TXT, NUM, True)


def attitude(c, view, t):
    """The horizon and pitch rungs turn with the aircraft; they clear off once the target has the picture."""
    k = view.k * math.pi / 180
    rad = math.radians(view.roll)
    cx, cy = W / 2, H / 2
    a = 0.6 * (1 - smooth(seg(t, 2.8, 3.6)))
    c.save()
    c.arc(cx, cy, 250, 0, 2 * math.pi)
    c.clip()
    c.new_path()
    c.translate(cx, cy)
    c.rotate(-rad)
    c.translate(0, view.pitch * k)
    for deg in range(-40, 41, 10):
        yy = -deg * k
        c.set_source_rgba(*hx(LINE), a)
        c.set_line_width(1.75 if deg == 0 else 1.25)
        half = 150 if deg == 0 else 66
        if deg < 0:
            c.set_dash([7, 5])
        c.move_to(-half, yy)
        c.line_to(-26, yy)
        c.move_to(26, yy)
        c.line_to(half, yy)
        c.stroke()
        c.set_dash([])
        if deg:
            text(c, f"{abs(deg)}", half + 8, yy + 5, 15, LINE, NUM, True, "left", a)
    c.restore()
    c.set_source_rgba(*hx(TXT), 0.55 + a * 0.5)
    c.set_line_width(1.5)
    c.arc(cx, cy, 7, 0, 2 * math.pi)
    c.stroke()
    for x0, y0, x1, y1 in ((-7, 0, -24, 0), (7, 0, 24, 0), (0, -7, 0, -15)):
        c.move_to(cx + x0, cy + y0)
        c.line_to(cx + x1, cy + y1)
    c.stroke()


def mix(a, b, u):
    return tuple(p + (q - p) * u for p, q in zip(hx(a), hx(b)))


def minimap(c, view, t):
    """Heading-up plan of the real scene: filled footprints with the streets as what is left between them, our track along
    the avenue, what the camera sees, and the bearing to the target."""
    size = 280
    x, y = W - size - 48, H - size - 48
    c.rectangle(x, y, size, size)
    c.set_source_rgba(*hx("#050B12"), 0.86)
    c.fill()
    cx, cy = x + size / 2, y + size * 0.72
    h = math.radians(view.heading)
    ca, sa = math.cos(h), math.sin(h)
    scale = 0.34

    def at(px, py):
        dx, dy = px - view.c[0], py - view.c[1]
        return cx + (dx * ca - dy * sa) * scale, cy - (dx * sa + dy * ca) * scale

    data = track()
    c.save()
    c.rectangle(x, y, size, size)
    c.clip()
    for tx, ty, w, d, tall in sorted(data["foot"], key=lambda f: f[4]):
        if abs(tx - view.c[0]) > 640 or abs(ty - view.c[1]) > 640:
            continue
        poly(c, [at(tx + sx * w / 2, ty + sy * d / 2) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        c.set_source_rgb(*mix("#142C37", "#3F7A8A", min(1, tall / 300)))
        c.fill()
    tar = data["target"]
    poly(c, [at(-23, -1100), at(23, -1100), at(23, tar[2]), at(-23, tar[2])])    # the avenue we are flying
    c.set_source_rgba(*hx("#050B12"), 0.82)
    c.fill()
    c.set_source_rgba(*hx(LINE), 0.4)
    c.set_line_width(1)
    for ex in (-23, 23):
        c.move_to(*at(ex, -1100))
        c.line_to(*at(ex, tar[2]))
    c.stroke()
    c.set_source_rgba(*hx(LINE), 0.22)                     # range rings
    c.set_line_width(1)
    for rr in (200, 400):
        c.arc(cx, cy, rr * scale, 0, 2 * math.pi)
        c.stroke()
    half = math.atan(18 / track()["cam"][max(0, min(149, int(round(t * 30))))]["lens"])
    c.move_to(cx, cy)
    c.arc(cx, cy, 400, -math.pi / 2 - half, -math.pi / 2 + half)
    c.close_path()
    c.set_source_rgba(*hx(LINE), 0.1)
    c.fill()
    f = max(0, min(149, int(round(t * 30))))
    cams = data["cam"]
    c.set_line_width(1.75)
    c.set_source_rgba(*hx(TXT), 0.9)                       # flown
    for i in range(0, f + 1, 3):
        px, py = at(*cams[i]["c"][:2])
        c.move_to(px, py) if i == 0 else c.line_to(px, py)
    c.line_to(cx, cy)
    c.stroke()
    tx, ty = (tar[0] + tar[1]) / 2, (tar[2] + tar[3]) / 2
    mx, my = at(tx, ty)
    col = LOCK if t >= 3.6 else AMB
    c.set_source_rgba(*hx(col), 0.9)                       # still to fly
    c.set_line_width(1.25)
    c.set_dash([5, 5])
    c.move_to(cx, cy)
    c.line_to(mx, my)
    c.stroke()
    c.set_dash([])
    mx, my = max(x + 14, min(x + size - 14, mx)), max(y + 14, min(y + size - 14, my))
    poly(c, [(mx, my - 7), (mx + 7, my), (mx, my + 7), (mx - 7, my)])
    c.set_source_rgb(*hx(col))
    c.fill()
    c.restore()
    nx, ny = x + size - 27, y + 27                         # a compass, turning against us
    c.arc(nx, ny, 15, 0, 2 * math.pi)
    c.set_source_rgba(*hx(INK), 0.72)
    c.fill_preserve()
    c.set_source_rgba(*hx(LINE), 0.8)
    c.set_line_width(1)
    c.stroke()
    c.move_to(nx + 15 * sa * -1, ny - 15 * ca)
    c.line_to(nx + 21 * sa * -1, ny - 21 * ca)
    c.set_line_width(1.5)
    c.stroke()
    text(c, "N", nx, ny + 5, 14, TXT, NUM, True, "center")
    poly(c, [(cx, cy - 11), (cx + 7, cy + 7), (cx, cy + 3), (cx - 7, cy + 7)])
    c.set_source_rgb(*hx(TXT))
    c.fill()
    brackets(c, x, y, x + size, y + size, LINE, 11, 1.25, 0.85)
    c.rectangle(x + 9, y + 9, 96, 25)
    c.set_source_rgba(*hx(INK), 0.66)
    c.fill()
    spaced(c, "航向朝上", x + 16, y + 27, 15, TXT, SANS, False, "left", 0.12)
    bar = 100 * scale
    c.set_source_rgba(*hx(TXT), 0.9)
    c.set_line_width(1.25)
    poly(c, [(x + 14, y + size - 20), (x + 14, y + size - 15), (x + 14 + bar, y + size - 15), (x + 14 + bar, y + size - 20)], close=False)
    c.stroke()
    text(c, "100 M", x + 14 + bar + 8, y + size - 13, 14, TXT, NUM, True)


def status(c, view, t):
    x, y, w, h = 48, H - 48 - 150, 330, 150
    panel(c, x, y, w, h, 0.66)
    text(c, "SKY-07", x + 16, y + 32, 23, TXT, NUM, True)
    mode = "巡航" if t < 2.1 else ("爬升" if t < 3.55 else "跟踪")
    spaced(c, mode, x + w - 16, y + 31, 17, AMB, SANS, False, "right", 0.3)
    c.set_source_rgba(*hx(LINE), 0.35)
    c.set_line_width(1)
    c.move_to(x + 16, y + 44)
    c.line_to(x + w - 16, y + 44)
    c.stroke()
    bx, bw = x + 66, 168
    charge = 0.87 - t * 0.012
    yy = y + 72
    text(c, "电量", x + 16, yy, 16, LINE, SANS)
    c.rectangle(bx, yy - 8, bw, 3)
    c.set_source_rgba(*hx(LINE), 0.22)
    c.fill()
    c.rectangle(bx, yy - 8, bw * charge, 3)
    c.set_source_rgba(*hx(LINE), 1)
    c.fill()
    text(c, f"{charge * 100:.0f}%", x + w - 16, yy, 18, TXT, NUM, True, "right")
    yy = y + 104
    text(c, "信号", x + 16, yy, 16, LINE, SANS)
    c.set_source_rgba(*hx(LINE), 0.95)
    c.set_line_width(1.25)
    for i in range(70):
        u = i * 0.42 + t * 9
        v = math.sin(u * 0.31) * 0.5 + math.sin(u * 0.083 + 1) * 0.3 + math.sin(u * 0.77) * 0.2
        px, py = bx + i * bw / 69, yy - 6 - v * 7
        c.move_to(px, py) if i == 0 else c.line_to(px, py)
    c.stroke()
    text(c, f"{-58 - 4 * abs(math.sin(t * 2.1)):.0f} DBM", x + w - 16, yy, 18, TXT, NUM, True, "right")
    yy = y + 134
    text(c, "风", x + 16, yy, 16, LINE, SANS)
    rich(c, "西北 7 M/S", bx, yy, 18, TXT)
    text(c, "1013 HPA", x + w - 16, yy, 18, TXT, NUM, True, "right")


def corners(c, t):
    """Clock and weather top left, the recording top right, each on its own ground."""
    c.rectangle(40, 36, 234, 76)
    c.set_source_rgba(*hx(INK), 0.6)
    c.fill()
    text(c, "23:47:12", 54, 68, 28, TXT, NUM, True)
    text(c, "小雨 · 能见度 1.2 KM", 54, 98, 17, LINE, SANS)
    c.rectangle(W - 214, 36, 174, 76)
    c.set_source_rgba(*hx(INK), 0.6)
    c.fill()
    if int(t * 3) % 2 == 0:
        c.arc(W - 124, 58, 6, 0, 2 * math.pi)
        c.set_source_rgb(*hx(LOCK))
        c.fill()
    text(c, "REC", W - 54, 66, 24, TXT, NUM, True, "right")
    text(c, f"00:{t:05.2f}", W - 54, 98, 21, LINE, NUM, True, "right")


def trackers(c, view, t):
    """The cargo drone the HUD picks up along the way: its box and distance come from the 3D scene, and the track drops as it leaves."""
    if t < 1.75:
        p = view.drone
        r = view.rect(p[0] - 10, p[0] + 10, p[1] - 10, p[1] + 10, p[2] - 5, p[2] + 1)
        if r:
            d = float(np.linalg.norm(p - view.c))
            pad = 10
            x, y = r[0] - pad, r[3] + pad
            a = seg(x, 170, 240)
            if a <= 0:
                return
            brackets(c, x, r[1] - pad, r[2] + pad, y, LINE, 12, 1.5, a)
            c.push_group()
            c.set_source_rgba(*hx(LINE), 1)
            c.set_line_width(1.5)
            c.move_to(x, y)
            c.line_to(x, y + 28)
            c.stroke()
            ground(c, x, y + 28, "货运无人机 CN-4471", f"{d:.0f} M   载重 18 KG", LINE)
            c.pop_group_to_source()
            c.paint_with_alpha(a * seg(t, 0.45, 0.55))


def lock_on(c, view, t):
    """From 2.5 s the reticle hunts the tower's crown, closes on it, and at 3.6 s it holds."""
    data = track()
    x0, x1, y0, y1, z0, z1 = data["target"]
    r = view.rect(x0, x1, y0, y1, z0, z1)
    if r is None:
        return
    cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
    a = seg(t, 2.5, 3.6)
    if a <= 0:
        return
    hunt = smooth(a)
    wobble = (1 - hunt) * 60
    gx = W / 2 * (1 - hunt) + cx * hunt + wobble * math.sin(t * 19)
    gy = H / 2 * (1 - hunt) + cy * hunt + wobble * math.cos(t * 23)
    half = max(70, min(158, max(r[2] - r[0], r[3] - r[1]) / 2 + 18))
    radius = half + (1 - hunt) * 240
    locked = t >= 3.6
    col = LOCK if locked else AMB
    c.set_source_rgb(*hx(col))
    c.set_line_width(1.75)
    c.save()
    c.translate(gx, gy)
    c.rotate(t * (0.5 if locked else 1.8))
    for i in range(4):
        c.arc(0, 0, radius, i * math.pi / 2 + 0.34, i * math.pi / 2 + 1.23)
        c.stroke()
    c.set_line_width(1.25)
    for i in range(36):                                    # a fine scale outside it
        u = i * math.pi / 18
        rr = radius + (14 if i % 3 == 0 else 9)
        c.move_to(math.cos(u) * (radius + 5), math.sin(u) * (radius + 5))
        c.line_to(math.cos(u) * rr, math.sin(u) * rr)
    c.stroke()
    c.restore()
    c.set_line_width(1.75)
    for ang in (0, 1, 2, 3):
        dx, dy = math.cos(ang * math.pi / 2), math.sin(ang * math.pi / 2)
        c.move_to(gx + dx * (radius - 13), gy + dy * (radius - 13))
        c.line_to(gx + dx * (radius + 3), gy + dy * (radius + 3))
    c.stroke()
    p = np.array(((x0 + x1) / 2, (y0 + y1) / 2, 280.0))
    d = view.project(p)[2]
    closing = float(np.dot(view.v, (p - view.c) / np.linalg.norm(p - view.c)))
    lx, ly = min(gx + radius + 92, W - 440), max(176, gy - radius * 0.62)
    u = math.radians(-38)
    callout(c, gx + radius * math.cos(u), gy + radius * math.sin(u), lx, ly, f"荒坂塔 · 高 {z1:.0f} M", f"{d:.0f} M   接近 {closing:.0f} M/S", col)
    lock_mark(c, gx, gy, radius, t)


def lock_mark(c, gx, gy, radius, t):
    """The lock is the reticle's own: a flag on the ring, diametrically opposite the target's tag, with nothing filled behind it."""
    p = smooth(seg(t, 3.6, 3.78))
    if p <= 0:
        return
    u = math.radians(142)
    ax, ay = gx + radius * math.cos(u), gy + radius * math.sin(u)
    lx, ly = gx - radius - 92, gy + radius * 0.62
    flick = t > 3.8 or int(t * 30) % 2 == 0
    c.set_source_rgba(*hx(LOCK), 1.0 if flick else 0.35)
    c.set_line_width(1.25)
    poly(c, [(ax, ay), (lx + 22, ly), (lx, ly)], close=False)
    c.stroke()
    c.arc(ax, ay, 2.5, 0, 2 * math.pi)
    c.fill()
    c.set_line_width(1.5)
    c.move_to(lx, ly - 28 * p)
    c.line_to(lx, ly + 28 * p)
    c.stroke()
    if p > 0.85 and flick:
        etched(c, "目标锁定", lx - 14, ly - 1, 28, LOCK, 0.32)
        etched(c, "TRK 01   置信度 98%", lx - 14, ly + 23, 16, TXT)


def glitch(rgb, t):
    """Two frames as the lock engages: a thin band or two slip sideways with a hair of colour split, and that is all."""
    rnd = random.Random(int(t * 30))
    out = rgb.copy()
    for _ in range(2):
        y, h = rnd.randrange(0, H - 12), rnd.randint(3, 10)
        out[y:y + h] = np.roll(rgb[y:y + h], rnd.choice((-1, 1)) * rnd.randint(10, 26), axis=1)
    out[..., 0] = np.roll(out[..., 0], 2, axis=1)
    out[..., 2] = np.roll(out[..., 2], -2, axis=1)
    return out


def plate_frame(t):
    if not TEST:
        return seq("cyberhud", t)
    path = os.path.join(OUT, "frames", "test", f"cyberhud_f{int(round(t * 30)) + 1:04d}.png")
    return np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR), np.float32) / 255


TEST = False


def cyberhud(t):
    view = View(t)
    rgb = bloom(plate_frame(t))
    s = surface(rgb)
    c = cairo.Context(s)
    rain(c, view, t)
    edge = cairo.RadialGradient(W / 2, H / 2, 760, W / 2, H / 2, 1240)     # the instruments sit on a quieter ground
    edge.add_color_stop_rgba(0, 0, 0, 0, 0)
    edge.add_color_stop_rgba(1, 0, 0, 0, 0.36)
    c.set_source(edge)
    c.paint()
    if 3.6 <= t < 4.0:
        c.set_source_rgba(*hx(LOCK), 0.09 * (1 - seg(t, 3.6, 4.0)))
        c.paint()

    hud, g = canvas(None)
    trackers(g, view, t)
    lock_on(g, view, t)
    attitude(g, view, t)
    heading_tape(g, view)
    ladder(g, 132, np.linalg.norm(view.v) * 3.6, "速度", "KM/H", -1, 20, 2.1)
    ladder(g, W - 132, view.c[2], "高度", "M", 1)
    corners(g, t)
    status(g, view, t)
    minimap(g, view, t)
    add(c, glow(hud, ((2, 0.18), (6, 0.08))))
    c.set_source_surface(hud)
    c.paint()
    if 3.6 <= t < 3.65:
        return surface(glitch(rgb_of(s), t))
    return s


if __name__ == "__main__":
    if sys.argv[1:] == ["signs"]:
        make_signs()
    elif sys.argv[1] == "test":                            # HUD over the small test plates, without a final render
        TEST = True
        os.makedirs(os.path.join(OUT, "snaps"), exist_ok=True)
        for f in sys.argv[2].split(","):
            cyberhud((int(f) - 1) / 30).write_to_png(os.path.join(OUT, "snaps", f"cyberhud_hud_{int(f):04d}.png"))
