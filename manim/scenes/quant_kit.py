"""Shared kit for episode 4 (一口气教你学会量化交易): storyboard, music grid, the dark-lab look, formulas, drawing helpers.

quant_film.py (lineup, subtitles, frames, audio, commands) and the shot modules quant_shots_*.py import from here.
Every helper draws into a cairo context; a frame is a pure function of film time t (lru caches only).

The look: a near-black navy lab with a faint grid; thin luminous lines (cyan = programs and signals, amber = money and
the highlighted path, violet = risk), red = up and green = down as in A-shares; bilingual labels in letter-spaced
Avenir Next; numbers in DIN; LaTeX formulas painted as light, each marked term lighting up on the beat that explains it.

Timing: 0-HOOK is the Liang Wenfeng hook; the title, the body and the #53 meme follow with no music, only effects,
on an even 80.2 BPM grid (a bar = 2.995 s); from #56 King Gnu's AIZO (95.0 BPM, a bar = 2.525 s) plays to the end, the
only song in the film. Rows keep their v2 numbers, so the table is in time order, not number order: the ending runs
#53 #56 #54 #51 #55 #57. Every row from #3 on starts on a bar or half bar. Shots count beats from their row's start with bt(t, n) and build up one element per
beat.
"""
import csv
import math
import os
import re
from functools import lru_cache

import cairo
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PROJ = os.path.join(ROOT, "quant")
OUT = os.path.join(PROJ, "film_out")
SCENES = os.path.join(HERE, "quant.scenes.md")
FORMULAS = os.path.join(PROJ, "formulas")
MEMES = os.path.join(PROJ, "memes")
SFX = os.path.join(ROOT, "manim", "sfx")

HOOK = 10.0                                 # the title lands here, on the grid's first downbeat
BEAT = 2.995 / 4                            # 80.2 BPM
FINALE_SONG = os.path.join(PROJ, "bgm", "aizo.wav")          # King Gnu「AIZO」(B站 BV1sySSBgEsr), #56 to the end
FINALE_BEATS = os.path.join(PROJ, "bgm", "aizo.beats.csv")
FINALE_FROM = 77.601                        # the second chorus's downbeat lands on A(56), with its first line
FINALE_GAIN = -4.5                          # dB: the master runs ~5 LU hotter than the panel BGM
FINALE_BEAT = (97.802 - 77.601) / 32        # 95.0 BPM
FX_LIFT = 12.0                              # dB on the effects where no music plays (hook, body): v3 measured -35 LUFS
FX_CEIL = 0.7                               # the effects stem is limited here before the music joins

W, H, FPS = 1920, 1080, 30
SERIF, SANS, MONO, LATIN, DIGITS = "Songti SC", "PingFang SC", "Menlo", "Avenir Next", "DIN Alternate"
COL = dict(bg="#05080f", bg2="#0b1424", grid="#132038", ink="#e8eef8", ink2="#8e9cb6", dim="#465572",
           cyan="#3ee6d8", ice="#8ddcff", amber="#ffb23f", gold="#f6c453", red="#ff4d5e", green="#22d47e",
           violet="#a084ff", blue="#4f8dff", white="#ffffff", glass="#0a1220")
COL.update(paper=COL["bg"], night=COL["bg"], gold_hi=COL["gold"])        # names the older shots used
SUB_Y, SUB_SIZE = 984, 44
NOTE_SIZE = 22
NOTE_X = 1860.0


# ======================================================================================
# colour and easing
# ======================================================================================
def hx(c):
    if not isinstance(c, str):
        return np.asarray(c, dtype=float)
    c = COL.get(c, c)                                         # a palette name or a hex string
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255


def mix(a, b, k):
    return a * (1 - k) + b * k


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def sm(x):
    return float(smooth(x))


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = min(max(x, 0.0), 1.0)
    return x ** 3


def pop(x):
    """0 -> 1 with a small overshoot: stickers, stamps and slams land with it."""
    x = min(max(x, 0.0), 1.0)
    return 1 + 2.2 * (x - 1) ** 3 + 1.2 * (x - 1) ** 2


def win(t, a, b, fin=0.25, fout=0.25):
    """1 inside [a, b), easing in and out at the edges."""
    return sm((t - a) / fin) * (1 - sm((t - b + fout) / fout))


def rng(seed):
    return np.random.default_rng(seed)


def lerp_pts(p, q, k):
    return [(x0 + (x1 - x0) * k, y0 + (y1 - y0) * k) for (x0, y0), (x1, y1) in zip(p, q)]


# ======================================================================================
# storyboard and beats
# ======================================================================================
@lru_cache(None)
def rows():
    out = {}
    for line in open(SCENES, encoding="utf-8"):
        m = re.match(r"\|\s*(\d+)\s*\|\s*([\d.]+)–([\d.]+)\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        out[int(m[1])] = dict(n=int(m[1]), a=float(m[2]), b=float(m[3]), sub="" if cells[2] == "—" else cells[2],
                              note="" if cells[4] in ("", "—") else cells[4])
    return out


def A(n):
    return rows()[n]["a"]


def B(n):
    return rows()[n]["b"]


END = max(r["b"] for r in rows().values())


def plain(s):
    return re.sub(r"[{}!]", "", s)


def need(s):
    """Reading time (handbook §4): max(2 s, 1 s + Han/5 + other/8), spaces not counted."""
    s = plain(s)
    han = sum(1 for ch in s if "一" <= ch <= "鿿")
    other = sum(1 for ch in s if not ("一" <= ch <= "鿿") and not ch.isspace())
    return max(2.0, 1 + han / 5 + other / 8)


LOOK = 0.3          # at least this much picture without a subtitle after every line
QUICK = {4, 5, 6}   # the owner found #4-#6 too slow: these lines run back to back, a subtitle may stay until the cut


def shown(r):
    """A subtitle stays for its reading time plus up to 0.6 s, never past its slot."""
    return r["a"], r["a"] + min(need(r["sub"]) + 0.6, r["b"] - r["a"] - 0.12)


@lru_cache(None)
def beats():
    """(film time, beat in bar), part by part: the even grid under the title and again from #4 (so the body's bars
    start on #4, as in v3) to the end of the #53 meme, AIZO from #56. A beat within 50 ms of the next part's start
    is that part's first beat, not one of its own."""
    out = []
    for a, b, path, s0 in ((A(3), A(4), None, 0), (A(4), A(56), None, 0), (A(56), END, FINALE_BEATS, FINALE_FROM)):
        if path is None:
            out += [(a + i * BEAT, i % 4 + 1) for i in range(int((b - a) / BEAT) + 1) if a + i * BEAT < b - 0.05]
            continue
        src = [(float(r["t"]), int(r["beat_in_bar"])) for r in csv.DictReader(open(path))]
        out += [(a + t - s0, k) for t, k in src if s0 - 1e-3 <= t and a + t - s0 < b - 0.05]
    return sorted(out)


def bt(t, n):
    """Beats since row n started (rows start on a beat, so integer values are beats)."""
    return (t - A(n)) / BEAT


def at(n, k):
    """Film time of beat k (may be fractional) of row n."""
    return A(n) + k * BEAT


def on(t, t0, d=0.22):
    """0 -> 1 from t0 over d seconds, easing out: the default entrance."""
    return ease_out((t - t0) / d)


def kick(t, t0, d=0.35):
    """A hit that decays 1 -> 0 after t0: flashes, glow pulses."""
    x = t - t0
    return math.exp(-x / (d / 3)) if x >= 0 else 0.0


# ======================================================================================
# cairo <-> numpy helpers
# ======================================================================================
_keep = []


def surf_from(rgb, alpha, dither=True):
    """rgb: (3,) or (h,w,3) floats 0-1, alpha (h,w) 0-1 -> premultiplied ARGB32 surface."""
    h, w = alpha.shape
    if np.ndim(rgb) == 1:
        rgb = np.broadcast_to(rgb, (h, w, 3))
    a = alpha[..., None]
    pm = np.concatenate([rgb[..., ::-1] * a, a], axis=2) * 255
    if dither:
        pm = pm + np.random.default_rng(7).uniform(-0.5, 0.5, pm.shape)
    arr = np.ascontiguousarray(np.clip(np.round(pm), 0, 255).astype(np.uint8))
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
    if stride != w * 4:
        arr = np.ascontiguousarray(np.pad(arr, ((0, 0), (0, stride // 4 - w), (0, 0))))
    _keep.append(arr)
    return cairo.ImageSurface.create_for_data(memoryview(arr), cairo.FORMAT_ARGB32, w, h, stride)


def surf_alpha(s):
    w, h, st = s.get_width(), s.get_height(), s.get_stride()
    return np.ndarray((h, st // 4, 4), np.uint8, s.get_data())[:, :w, 3].astype(float) / 255


def surf_rgb(img, alpha=None):
    """(h,w,3) floats 0-1 (and optional (h,w) alpha) -> surface, without keeping a reference per frame."""
    h, w, _ = img.shape
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    v = np.ndarray((h, s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :w]
    a = np.ones((h, w)) if alpha is None else np.clip(alpha, 0, 1)
    v[..., :3] = np.clip(img[..., ::-1] * a[..., None] * 255, 0, 255)
    v[..., 3] = np.clip(a * 255, 0, 255)
    s.mark_dirty()
    return s


def paint_img(ctx, surf, x, y, a=1.0, scale=1.0, add=False):
    """Paint a surface with its top-left at (x, y); add=True blends it as light."""
    ctx.save()
    if add:
        ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.translate(x, y); ctx.scale(scale, scale)
    ctx.set_source_surface(surf, 0, 0); ctx.paint_with_alpha(a)
    ctx.restore()


# ======================================================================================
# the room: dark lab, faint grid, light
# ======================================================================================
@lru_cache(None)
def background(w=W, h=H, oy=0):
    """Near-black navy, lifted in the centre, a 64 px grid with brighter dots on every fourth crossing. A taller
    canvas (the vertical short) holds the film's frame with its top at row oy and continues the room around it."""
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    ys -= oy
    r2 = ((xs - w / 2) / (0.72 * W)) ** 2 + ((ys - 0.46 * H) / (0.72 * H)) ** 2
    img = mix(hx("bg2")[None, None], hx("bg")[None, None], smooth(r2 / 1.2)[..., None])
    g = ((xs % 64) < 1.2) | ((ys % 64) < 1.2)
    img[g] = mix(img[g], hx("grid"), 0.55)
    d = ((xs % 256) < 3) & ((ys % 256) < 3)
    img[d] = mix(img[d], hx("dim"), 0.6)
    img += gaussian_filter(rng(2026).standard_normal((h, w)), 1.0)[..., None] * 0.006
    return surf_from(np.clip(img, 0, 1), np.ones((h, w)))


@lru_cache(None)
def sprite_glow(size=256):
    c = size / 2
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    rr = np.hypot(xx - c, yy - c) / (size / 512)
    return np.clip(0.9 * np.exp(-(rr / 12) ** 2) + 0.4 * np.exp(-(rr / 45) ** 2) + 0.12 * np.exp(-(rr / 120) ** 2), 0, 1)


@lru_cache(None)
def glow_surface(col):
    return surf_from(hx(col), sprite_glow())


def glow_at(ctx, x, y, col, scale, alpha):
    """A soft additive light blob at (x, y); scale 1 is about 250 px across."""
    if alpha <= 0.002 or scale <= 0.001:
        return
    ctx.save()
    ctx.set_operator(cairo.OPERATOR_ADD)
    ctx.translate(x, y); ctx.scale(scale, scale)
    ctx.set_source_surface(glow_surface(col if isinstance(col, str) else "#%02x%02x%02x" % tuple(
        int(v * 255) for v in col)), -128, -128)
    ctx.paint_with_alpha(min(1.0, alpha))
    ctx.restore()


def path(ctx, pts, close=False):
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    if close:
        ctx.close_path()


def gline(ctx, pts, col, lw=3.0, a=1.0, glow=1.0, close=False, core=True, dash=None):
    """A luminous polyline: two additive halos, the coloured line and a thin hot core."""
    if a <= 0.002 or len(pts) < 2:
        return
    rgb = hx(col)
    ctx.save()
    ctx.set_line_join(cairo.LINE_JOIN_ROUND); ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    if dash:
        ctx.set_dash(dash)
    if glow > 0:
        ctx.set_operator(cairo.OPERATOR_ADD)
        for w_, al in ((lw * 6.0, 0.05), (lw * 2.6, 0.13)):
            path(ctx, pts, close); ctx.set_source_rgba(*rgb, al * a * glow); ctx.set_line_width(w_); ctx.stroke()
        ctx.set_operator(cairo.OPERATOR_OVER)
    path(ctx, pts, close); ctx.set_source_rgba(*rgb, a); ctx.set_line_width(lw); ctx.stroke()
    if core and glow > 0 and lw >= 2:
        path(ctx, pts, close); ctx.set_source_rgba(*mix(rgb, np.ones(3), 0.65), 0.55 * a); ctx.set_line_width(lw * 0.35)
        ctx.stroke()
    ctx.restore()


def dot(ctx, x, y, r, col, a=1.0, glow=1.0):
    """A glowing node."""
    if a <= 0.002:
        return
    if glow > 0:
        glow_at(ctx, x, y, col, r / 22 * (1 + glow), 0.55 * a * glow)
    ctx.arc(x, y, r, 0, 2 * math.pi); ctx.set_source_rgba(*hx(col), a); ctx.fill()
    ctx.arc(x, y, r * 0.45, 0, 2 * math.pi); ctx.set_source_rgba(*mix(hx(col), np.ones(3), 0.7), 0.8 * a); ctx.fill()


def rrect(ctx, x, y, w, h, r):
    r = max(0.0, min(r, w / 2, h / 2))
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0); ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi); ctx.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    ctx.close_path()


def fill(ctx, col, a=1.0, keep=False):
    ctx.set_source_rgba(*hx(col), a)
    ctx.fill_preserve() if keep else ctx.fill()


def stroke(ctx, col, lw, a=1.0):
    ctx.set_source_rgba(*hx(col), a); ctx.set_line_width(lw); ctx.stroke()


def panel(ctx, x, y, w, h, a=1.0, edge="cyan", r=10, fill_a=0.62, corners=True, glow=0.0):
    """A dark glass panel: translucent fill, hairline border, bright corner ticks (and an optional edge glow)."""
    if a <= 0.002:
        return
    rrect(ctx, x, y, w, h, r); fill(ctx, "glass", fill_a * a, keep=True); stroke(ctx, edge, 1.4, 0.35 * a)
    if glow > 0:
        gline(ctx, [(x, y), (x + w, y), (x + w, y + h), (x, y + h)], edge, 1.6, glow * a, glow=1.0, close=True, core=False)
    if corners:
        L = min(26, w / 4, h / 4)
        ctx.set_line_cap(cairo.LINE_CAP_SQUARE)
        for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            ctx.move_to(cx + dx * L, cy); ctx.line_to(cx, cy); ctx.line_to(cx, cy + dy * L)
            stroke(ctx, edge, 2.6, 0.9 * a)


def arrow(ctx, x0, y0, x1, y1, col, lw, a=1.0, head=None, glow=0.8):
    head = head or lw * 3.4
    ang = math.atan2(y1 - y0, x1 - x0)
    bx, by = x1 - head * math.cos(ang), y1 - head * math.sin(ang)
    gline(ctx, [(x0, y0), (bx, by)], col, lw, a, glow)
    ctx.move_to(x1, y1)
    ctx.line_to(bx + head * 0.6 * math.sin(ang), by - head * 0.6 * math.cos(ang))
    ctx.line_to(bx - head * 0.6 * math.sin(ang), by + head * 0.6 * math.cos(ang))
    ctx.close_path(); fill(ctx, col, a)


def flow(ctx, pts, t, col, n=8, speed=0.6, r=4.0, a=1.0, seed=0):
    """Light pulses travelling along a polyline (pts), n of them, speed in path lengths per second."""
    if a <= 0.002 or len(pts) < 2:
        return
    seg = [math.dist(p, q) for p, q in zip(pts, pts[1:])]
    total = sum(seg) or 1.0
    ph = rng(seed).random(n)
    for i in range(n):
        u = ((t * speed + ph[i] + i / n) % 1.0) * total
        for (p, q), L in zip(zip(pts, pts[1:]), seg):
            if u <= L:
                k = u / max(L, 1e-6)
                dot(ctx, p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k, r, col, a, 1.0)
                break
            u -= L


# ======================================================================================
# type
# ======================================================================================
SYMBOLS = set("×÷→≈√−±≠") | set("αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")  # drawn in Songti: other faces lack them


def runs(s, font):
    out = []
    for ch in s:
        f = SERIF if font != SERIF and ch in SYMBOLS else font  # neither PingFang nor DIN/Avenir draw these
        if font in (MONO, LATIN, DIGITS) and ("一" <= ch <= "鿿" or ch in "，。：；（）「」『』《》〈〉“”‘’？！、·…—"):
            f = SANS                                          # Latin faces have no Chinese: PingFang fills in
        if out and out[-1][1] == f:
            out[-1] = (out[-1][0] + ch, f)
        else:
            out.append((ch, f))
    return out


def set_font(ctx, font, size, bold=False):
    ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)


def text_width(ctx, s, size, font=SANS, bold=False, spacing=0.0):
    w = 0.0
    for piece, f in runs(s, font):
        set_font(ctx, f, size, bold)
        w += ctx.text_extents(piece).x_advance + spacing * size * len(piece)
    return w


def text(ctx, s, x, y, size, col, alpha=1.0, font=SANS, bold=False, align="left", halo=0.0, halo_col=None,
         spacing=0.0):
    """Baseline text. col: palette name, hex or rgb; halo > 0 strokes the glyphs in the room colour first;
    spacing adds that fraction of the size between letters (for the letter-spaced Latin labels)."""
    if alpha <= 0.002:
        return 0.0
    rgb = hx(col)
    w = text_width(ctx, s, size, font, bold, spacing)
    if align == "right":
        x -= w
    elif align == "center":
        x -= w / 2
    for piece, f in runs(s, font):
        set_font(ctx, f, size, bold)
        chunks = list(piece) if spacing else [piece]
        for ch in chunks:
            if halo:
                ctx.move_to(x, y); ctx.text_path(ch)
                ctx.set_source_rgba(*hx(halo_col or "bg"), halo * alpha); ctx.set_line_width(size * 0.22)
                ctx.set_line_join(cairo.LINE_JOIN_ROUND); ctx.stroke()
            ctx.set_source_rgba(*rgb, alpha)
            ctx.move_to(x, y); ctx.show_text(ch)
            x += ctx.text_extents(ch).x_advance + spacing * size * (1 if spacing else 0)
    ctx.new_path()
    return w


def label(ctx, s, x, y, size, col="ink2", a=1.0, align="left", bold=False, spacing=0.22):
    """Letter-spaced Latin caps, e.g. 'IMPLIED VOLATILITY SURFACE' (Chinese in it falls back to PingFang)."""
    return text(ctx, s, x, y, size, col, a, font=LATIN, bold=bold, align=align, spacing=spacing)


@lru_cache(maxsize=512)
def _glow_text_surface(s, size, font, bold, col, spacing, blur):
    tmp = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 4, 4))
    w = text_width(tmp, s, size, font, bold, spacing)
    pad = int(blur * 3 + size * 0.3)
    sw, sh = int(w + 2 * pad), int(size * 1.45 + 2 * pad)
    m = cairo.ImageSurface(cairo.FORMAT_ARGB32, sw, sh)
    c = cairo.Context(m)
    text(c, s, pad, pad + size, size, "white", 1.0, font, bold, spacing=spacing)
    a = surf_alpha(m)
    halo = np.clip(gaussian_filter(a, blur) * 1.6 + gaussian_filter(a, blur * 3) * 0.9, 0, 1)
    crisp = cairo.ImageSurface(cairo.FORMAT_ARGB32, sw, sh)
    c = cairo.Context(crisp)
    text(c, s, pad, pad + size, size, col, 1.0, font, bold, spacing=spacing)
    return surf_from(mix(hx(col), np.ones(3), 0.15), halo), crisp, sw, pad + size, pad


def glow_text(ctx, s, x, y, size, col="ink", a=1.0, font=SANS, bold=True, align="left", glow=0.8, spacing=0.0,
              scale=1.0, rot=0.0):
    """Text that gives off light (titles, big numbers). (x, y) is the baseline point given by align."""
    if a <= 0.002 or not s:
        return
    halo, crisp, sw, base, pad = _glow_text_surface(s, int(size), font, bold, col if isinstance(col, str) else
                                                    "#%02x%02x%02x" % tuple(int(v * 255) for v in col),
                                                    spacing, max(2.0, size * 0.09))
    ox = {"left": -pad, "center": -sw / 2, "right": -(sw - pad)}[align]
    ctx.save()
    ctx.translate(x, y); ctx.rotate(rot); ctx.scale(scale, scale)
    if glow > 0:
        ctx.save(); ctx.set_operator(cairo.OPERATOR_ADD)
        ctx.set_source_surface(halo, ox, -base); ctx.paint_with_alpha(min(1.0, 0.75 * glow * a)); ctx.restore()
    ctx.set_source_surface(crisp, ox, -base); ctx.paint_with_alpha(a)
    ctx.restore()


def num(ctx, s, x, y, size, col="ink", a=1.0, align="left", glow=0.7, scale=1.0):
    """A glowing number in DIN (falls back to PingFang for any Chinese)."""
    glow_text(ctx, s, x, y, size, col, a, font=DIGITS, bold=True, align=align, glow=glow, scale=scale)


def typed(s, k):
    """The first round(k * len) characters of s: a typing effect."""
    n = int(round(max(0.0, min(1.0, k)) * len(s)))
    return s[:n]


# ======================================================================================
# formulas: LaTeX PNGs whose marked terms print in pure colours (quant/make_formulas.py)
# ======================================================================================
FORMULA_INFO = {   # name -> (who, year, English name); the finale (#54) shows them in this order
    "bachelier": ("巴舍利耶", "1900", "RANDOM WALK"),
    "markowitz": ("马科维茨", "1952", "PORTFOLIO THEORY"),
    "kelly": ("凯利", "1956", "KELLY CRITERION"),
    "sharpe": ("夏普", "1966", "SHARPE RATIO"),
    "black_scholes": ("布莱克–斯科尔斯", "1973", "OPTION PRICING"),
    "grinold": ("格林诺德", "1989", "FUNDAMENTAL LAW"),
    "fama_french": ("法玛–弗伦奇", "1993", "THREE-FACTOR MODEL"),
    "macd": ("阿佩尔", "1970s", "MACD · DIF"),
    "factor": ("多因子模型", "", "MULTI-FACTOR MODEL"),
    "edge": ("大数定律", "", "LAW OF LARGE NUMBERS"),
    "chern_simons": ("陈省身–西蒙斯", "1974", "CHERN–SIMONS FORM"),
}
TERM_RGB = {(1, 0, 0): 1, (0, 1, 0): 2, (0, 0, 1): 3, (1, 0, 1): 4, (0, 1, 1): 5, (1, 1, 0): 6}


@lru_cache(None)
def formula_layers(name):
    """(ink coverage, term label per pixel: 0 = plain, 1..6 = \\T1..\\T6) from the colour-on-white PNG."""
    rgb = np.asarray(Image.open(os.path.join(FORMULAS, name + ".png")).convert("RGB"), float) / 255
    ink = 1 - rgb.min(axis=2)
    c = 1 - (1 - rgb) / np.maximum(ink, 1e-3)[..., None]       # the pure colour each pixel was blended from
    bits = (c > 0.5).astype(int)
    lab = np.zeros(ink.shape, int)
    for key, k in TERM_RGB.items():
        lab[(bits[..., 0] == key[0]) & (bits[..., 1] == key[1]) & (bits[..., 2] == key[2]) & (ink > 0.04)] = k
    return np.clip(ink * 1.12, 0, 1), lab


@lru_cache(None)
def formula_surface(name, term, col, blur=0):
    """Surface of the whole formula (term None), or of one marked term; blur > 0 gives its glow."""
    ink, lab = formula_layers(name)
    m = ink if term is None else ink * (lab == term)
    if blur:
        m = np.clip(gaussian_filter(m, blur) * 2.2 + gaussian_filter(m, blur * 3) * 1.4, 0, 1)
    return surf_from(hx(col), m)


def formula_size(name, width):
    ink, _ = formula_layers(name)
    return width, ink.shape[0] * width / ink.shape[1]


def formula(ctx, name, cx, cy, width, a=1.0, lit=None, col="ink", lit_col="amber", glow=0.55, rot=0.0, scale=1.0):
    """A formula centred at (cx, cy), width px wide, painted as light.
    lit: {term: level 0-1} lights \\T<term> in lit_col (a dict value may also be (level, colour))."""
    if a <= 0.002:
        return 0.0
    ink, _ = formula_layers(name)
    k = width / ink.shape[1]
    hh = ink.shape[0]
    ctx.save()
    ctx.translate(cx, cy); ctx.rotate(rot); ctx.scale(k * scale, k * scale)
    ox, oy = -ink.shape[1] / 2, -hh / 2
    if glow > 0:
        ctx.save(); ctx.set_operator(cairo.OPERATOR_ADD)
        ctx.set_source_surface(formula_surface(name, None, col, 9), ox, oy); ctx.paint_with_alpha(0.22 * glow * a)
        ctx.restore()
    ctx.set_source_surface(formula_surface(name, None, col), ox, oy); ctx.paint_with_alpha(0.92 * a)
    for term, v in (lit or {}).items():
        lvl, c2 = (v if isinstance(v, tuple) else (v, lit_col))
        if lvl <= 0.002:
            continue
        ctx.set_source_surface(formula_surface(name, term, c2), ox, oy); ctx.paint_with_alpha(min(1.0, lvl) * a)
        ctx.save(); ctx.set_operator(cairo.OPERATOR_ADD)
        ctx.set_source_surface(formula_surface(name, term, c2, 7), ox, oy)
        ctx.paint_with_alpha(min(1.0, 0.9 * lvl * a))
        ctx.restore()
    ctx.restore()
    return hh * k * scale


def formula_card(ctx, name, cx, cy, width, a=1.0, lit=None, scale=1.0, rot=0.0, info=True, edge="cyan", lit_col="amber",
                 en=True):
    """The formula's authority panel: the formula on dark glass, who and when underneath (Chinese, + English if en)."""
    if a <= 0.002:
        return
    _, fh = formula_size(name, width)
    w, h = width + 110, fh + (128 if info else 70)
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(rot); ctx.scale(scale, scale)
    panel(ctx, -w / 2, -h / 2, w, h, a, edge=edge)
    formula(ctx, name, 0, -h / 2 + 35 + fh / 2, width, a, lit, lit_col=lit_col)
    if info and name in FORMULA_INFO:
        who, year, eng = FORMULA_INFO[name]
        eng = eng if en else ""
        y = h / 2 - 34
        s = f"{who} · {year}" if year else who
        tw = text_width(ctx, s, 30, SANS, True)
        lw = text_width(ctx, eng, 18, LATIN, False, 0.22) + 26 if eng else 0.0
        x0 = -(tw + lw) / 2
        text(ctx, s, x0, y, 30, "amber", a, bold=True)
        if eng:
            label(ctx, eng, x0 + tw + 26, y - 2, 18, "ink2", a)
    ctx.restore()


# ======================================================================================
# memes
# ======================================================================================
@lru_cache(None)
def meme_img(name):
    return np.asarray(Image.open(os.path.join(MEMES, name)).convert("RGB"), float) / 255


@lru_cache(None)
def meme(name):
    im = meme_img(name)
    return surf_rgb(im), im.shape[1], im.shape[0]


def sticker(ctx, name, cx, cy, width, a=1.0, rot=0.0, scale=1.0, border=10, edge="white"):
    """A meme photo as a sticker: light border, soft glow behind."""
    if a <= 0.002 or scale <= 0.01:
        return 0.0
    s, w, h = meme(name)
    k = width / w
    hh = h * k
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(rot); ctx.scale(scale, scale)
    glow_at(ctx, 0, 0, "ice", (width + hh) / 380, 0.25 * a)
    rrect(ctx, -width / 2 - border, -hh / 2 - border, width + 2 * border, hh + 2 * border, 12); fill(ctx, edge, a)
    ctx.save(); ctx.translate(-width / 2, -hh / 2); ctx.scale(k, k)
    ctx.set_source_surface(s, 0, 0); ctx.paint_with_alpha(a)
    ctx.restore()
    ctx.restore()
    return hh


def portrait(ctx, name, cx, cy, r, a=1.0, ring="amber", focus=(0.5, 0.42), zoom=1.0):
    """A meme photo cropped into a circle with a glowing ring (the investor-panel look)."""
    if a <= 0.002:
        return
    s, w, h = meme(name)
    k = 2 * r / min(w, h) * zoom
    ctx.save()
    ctx.arc(cx, cy, r, 0, 2 * math.pi); ctx.clip()
    ctx.translate(cx - focus[0] * w * k, cy - focus[1] * h * k); ctx.scale(k, k)
    ctx.set_source_surface(s, 0, 0); ctx.paint_with_alpha(a)
    ctx.restore()
    gline(ctx, [(cx + r * math.cos(u), cy + r * math.sin(u)) for u in np.linspace(0, 2 * math.pi, 120)], ring, 4, a,
          glow=1.0)


def stamp(ctx, s, cx, cy, size, col, a, k, rot=-0.14):
    """A seal stamp that slams down: k runs 0 -> 1."""
    if k <= 0 or a <= 0.002:
        return
    sc = 1 + 1.6 * (1 - ease_out(k / 0.35)) if k < 0.35 else 1.0
    al = a * min(1, k / 0.2)
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(rot); ctx.scale(sc, sc)
    tw = text_width(ctx, s, size, bold=True)
    rrect(ctx, -tw / 2 - 26, -size * 0.85, tw + 52, size * 1.45, 10)
    fill(ctx, "bg", 0.55 * al, keep=True); stroke(ctx, col, 7, al)
    glow_text(ctx, s, 0, size * 0.32, size, col, al, align="center", glow=0.6)
    ctx.restore()


# ======================================================================================
# market pictures
# ======================================================================================
@lru_cache(None)
def walk(seed, n, drift=0.0, vol=1.0):
    """Deterministic OHLC candles from a random walk: (n, 4) array of open, high, low, close."""
    r = rng(seed)
    c = np.cumsum(r.normal(drift, vol, n)) + 100
    o = np.concatenate([[100], c[:-1]])
    hi = np.maximum(o, c) + np.abs(r.normal(0, vol * 0.6, n))
    lo = np.minimum(o, c) - np.abs(r.normal(0, vol * 0.6, n))
    return np.stack([o, hi, lo, c], 1)


def candles(ctx, data, x0, y0, w, h, upto=None, a=1.0, lo=None, hi=None, mono=None, glow=0.0):
    """Candles in the box (x0, y0, w, h); upto: how many to draw (float for a growing last one). Returns y(value)."""
    n = len(data)
    upto = n if upto is None else upto
    lo = data[:, 2].min() if lo is None else lo
    hi = data[:, 1].max() if hi is None else hi
    Y = lambda v: y0 + h - (v - lo) / (hi - lo) * h  # noqa: E731
    cw = w / n
    for i in range(min(n, int(math.ceil(upto)))):
        o, hh, ll, c = data[i]
        k = min(1.0, upto - i)
        col = mono or ("red" if c >= o else "green")
        xc = x0 + (i + 0.5) * cw
        ctx.move_to(xc, Y(hh)); ctx.line_to(xc, Y(ll)); stroke(ctx, col, max(1.2, cw * 0.1), a * k)
        top, bot = Y(max(o, c)), Y(min(o, c))
        ctx.rectangle(xc - cw * 0.32, top, cw * 0.64, max(2.0, bot - top)); fill(ctx, col, a * k)
    if glow > 0 and upto > 0:
        i = min(n, int(math.ceil(upto))) - 1
        glow_at(ctx, x0 + (i + 0.5) * cw, Y(data[i, 3]), "red" if data[i, 3] >= data[i, 0] else "green", 0.4, glow * a)
    return Y


def curve_pts(ys, x0, y0, w, h, upto=None, lo=None, hi=None):
    """Points of a series in the box (x0, y0, w, h); upto in [0, 1] cuts it (with an interpolated end)."""
    ys = np.asarray(ys, float)
    lo = ys.min() if lo is None else lo
    hi = ys.max() if hi is None else hi
    n = len(ys)
    m = (n - 1) * (1.0 if upto is None else max(0.0, min(1.0, upto)))
    i = int(m)
    pts = [(x0 + j / (n - 1) * w, y0 + h - (ys[j] - lo) / (hi - lo + 1e-9) * h) for j in range(i + 1)]
    if i < n - 1 and m > i:
        f = m - i
        v = ys[i] * (1 - f) + ys[i + 1] * f
        pts.append((x0 + m / (n - 1) * w, y0 + h - (v - lo) / (hi - lo + 1e-9) * h))
    return pts


def head(ctx, pts, col, a=1.0, r=7):
    """The glowing head dot at the end of a drawn line."""
    if pts:
        dot(ctx, *pts[-1], r, col, a, 1.4)


@lru_cache(None)
def equity(seed, n, drift, vol, crash_at=None, crash=0.0):
    r = rng(seed)
    x = r.normal(drift, vol, n)
    if crash_at is not None:
        x[crash_at:] += crash
    return np.cumsum(x)


def ema(x, n):
    out, a, v = [], 2 / (n + 1), x[0]
    for p in x:
        v = a * p + (1 - a) * v
        out.append(v)
    return np.array(out)


# ======================================================================================
# 3D wireframes (the chapter pages, the title, the Simons shot)
# ======================================================================================
def project(p, cx, cy, s, yaw, pitch, dist=4.0):
    """(x, y, z) in model units -> screen (x, y), z up, perspective from dist units away."""
    x, y, z = p
    x, y = x * math.cos(yaw) - y * math.sin(yaw), x * math.sin(yaw) + y * math.cos(yaw)
    y, z = y * math.cos(pitch) - z * math.sin(pitch), y * math.sin(pitch) + z * math.cos(pitch)
    f = dist / (dist + y)
    return cx + x * s * f, cy - z * s * f


def surface(ctx, f, cx, cy, s, yaw, pitch, a=1.0, prog=1.0, n=16, cols=("cyan", "amber"), lw=1.6, glow=0.8,
            span=1.0):
    """A wireframe of z = f(x, y) over [-span, span]^2, coloured from cols[0] to cols[1] along x, drawn up to prog."""
    if a <= 0.002:
        return
    g = np.linspace(-span, span, n)
    c0, c1 = hx(cols[0]), hx(cols[1])
    m = 40
    for axis in (0, 1):
        for i, u in enumerate(g):
            k = (i / (n - 1))
            if k > prog * 1.0001:
                break
            vs = np.linspace(-span, span, m)
            pts = [project((u, v, f(u, v)) if axis == 0 else (v, u, f(v, u)), cx, cy, s, yaw, pitch) for v in vs]
            col = mix(c0, c1, k if axis == 0 else 0.5)
            gline(ctx, pts, col, lw, a * (0.55 + 0.45 * k), glow, core=False)


def torus(ctx, cx, cy, R, r, yaw, pitch, a=1.0, prog=1.0, n=28, m=14, cols=("violet", "amber"), lw=1.4, glow=0.8):
    """A wireframe torus (the Chern–Simons nod in the Simons shot)."""
    if a <= 0.002:
        return
    c0, c1 = hx(cols[0]), hx(cols[1])
    for i in range(int(n * prog)):
        u = 2 * math.pi * i / n
        pts = [project(((R + r * math.cos(v)) * math.cos(u), (R + r * math.cos(v)) * math.sin(u), r * math.sin(v)),
                       cx, cy, 1.0, yaw, pitch, 1e9) for v in np.linspace(0, 2 * math.pi, 40)]
        gline(ctx, pts, c0, lw, a * 0.8, glow, core=False)
    for j in range(int(m * prog)):
        v = 2 * math.pi * j / m
        rad = R + r * math.cos(v)
        pts = [project((rad * math.cos(u), rad * math.sin(u), r * math.sin(v)), cx, cy, 1.0, yaw, pitch, 1e9)
               for u in np.linspace(0, 2 * math.pi, 80)]
        gline(ctx, pts, mix(c0, c1, 0.5 + 0.5 * math.cos(v)), lw, a * 0.9, glow, core=False)


def lattice_nodes(steps, x0, y0, dx, dy):
    """Binomial tree: {(i, j): (x, y)} for step i and up-moves j (0..i)."""
    return {(i, j): (x0 + i * dx, y0 + (i - 2 * j) * dy) for i in range(steps + 1) for j in range(i + 1)}


def lattice(ctx, steps, x0, y0, dx, dy, a=1.0, prog=1.0, col="cyan", lw=1.3, r=4.5):
    """The tree's thin edges and nodes, grown column by column up to prog (0-1)."""
    nodes = lattice_nodes(steps, x0, y0, dx, dy)
    upto = prog * steps
    for (i, j), (x, y) in nodes.items():
        if i > upto + 1e-6:
            continue
        if i < steps:
            for jj in (j, j + 1):
                k = min(1.0, upto - i)
                if k > 0:
                    x1, y1 = nodes[(i + 1, jj)]
                    gline(ctx, [(x, y), (x + (x1 - x) * k, y + (y1 - y) * k)], col, lw, 0.45 * a, 0.5, core=False)
        dot(ctx, x, y, r, col, a * 0.9, 0.6)
    return nodes


# ======================================================================================
# big textures: the 5000-stock wall, the GPU wall
# ======================================================================================
@lru_cache(None)
def cell_values(cols, rows_, seed):
    r = rng(seed)
    return r.normal(0, 1, (rows_, cols)), r.random((rows_, cols))


def wall_image(cols, rows_, cell, gap, seed, scan=None, lit=None, hue=None):
    """A grid of cells as an RGBA numpy image. Cells are red/green by their value; scan in [0, 1] is a sweep column
    that brightens; lit (rows_, cols) in [0, 1] overrides with the hue colour (e.g. the GPU wall lighting up)."""
    v, ph = cell_values(cols, rows_, seed)
    step = cell + gap
    hh, ww = rows_ * step, cols * step
    red, green = hx("red"), hx("green")
    base = np.where(v[..., None] > 0, red, green) * (0.25 + 0.45 * np.clip(np.abs(v[..., None]) / 2, 0, 1))
    if scan is not None:
        d = np.abs(np.arange(cols)[None, :] / cols - scan)
        base = base * (1 + 1.6 * np.exp(-(d / 0.03) ** 2))[..., None]
    if lit is not None:
        base = mix(base, hx(hue or "ice")[None, None] * (0.6 + 0.4 * ph[..., None]), lit[..., None])
    big = np.repeat(np.repeat(base, step, 0), step, 1)[:hh, :ww]
    mask = np.zeros((step, step)); mask[:cell, :cell] = 1
    return np.clip(big, 0, 1), np.tile(mask, (rows_, cols))
