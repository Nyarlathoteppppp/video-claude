"""五 字的位置, 六 回头看, 七 十二万遍: you asked how I learned to speak. What happened to me I cannot see, so I raised a small one the same way and open its page beside the chat. It is really trained, on the Studio, by ink_train.py; every number and every line of it on screen is read from what it left in ink/film_out/learn.

The window is film_page's, where film_weibo puts its window (PLACE): the Dock hidden, the foot of the frame left to the subtitles, the real wallpaper and the menu bar clock. This is another conversation on another evening, not the outer window, so its clock is its own and stands still (CLOCK). The chat is on the left; the small one's page on the right is white, a hairline round it, its name on one line and nothing else of the app's. Its lowest 40 points stay empty. What is on that page is drawn here, at the output's pixels times S, so a push in stays sharp; the window round it is Chrome's (ink_mac.snap).

    python film_learn.py still T [T ...]       # 1920 stills with their subtitles, ink/film_out/film/learn/stills/
    python film_learn.py sheet A B N [COLS]    # N frames between film times A and B on one sheet, 480 wide each
    python film_learn.py clip [A B] [--full]   # 30 fps, 960 by 540, no subtitles: ink/film_out/film/learn/learn.mp4; --full at 1920 by 1080, learn_1920.mp4
    python film_learn.py geometry              # where the caret is at 五's first frame, and when 六's looks are drawn
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

import film_page as ui
import ink_mac as mac
import ink_subs
from ink_learn import LEARN, DATA, face, glyph, put as _put, stamp, text as _text

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "learn")
W, H, FPS = 1920, 1080, 30
S = 2                                    # drawn at S times the output, then brought down
Z = mac.Z                                # frame pixels to a point of the Mac's screen
PAD = mac.PAD

# ---------------------------------------------------------------- the window, in its own points

WX, WY, WW, WH = 110, 56, 1000, 514      # film_weibo.PLACE: frame pixels for where, points for how big; its foot is at 56 + 514 * 1.7 = 930 px
TOP = 56
ASK = "你是怎么学会说话的？"
TITLE = ASK[:-1]
SAY = ("我自己的那一段，我看不见。", "我照那法子养了个小的，只给它读全唐诗。你看。")
NAME = "小模型"                          # the page's one title line, and its card in the chat
GAP_P = 8                                # the page's inset from the window's edges
SPLIT = 420                              # where the page starts once it is open
CHAT_W, BOX_W_S, TEXT_W_S, PILL_W_S = SPLIT - GAP_P, 372, 316, 340     # the chat beside it
HEAD, EMPTY = 40, 40                     # the page's title line, and its foot, which stays empty
CARD_W, CARD_H = 300, 58


def panel_rect(left=SPLIT):
    return (left, ui.BAR_H + GAP_P, WW - GAP_P, WH - GAP_P)


def inner(left=SPLIT):
    """What the page may draw in: under its title line, above its empty foot, inside its hairline."""
    x0, y0, x1, y1 = panel_rect(left)
    return (x0 + 1, y0 + HEAD, x1 - 1, y1 - EMPTY)


CONTENT = (SPLIT + 18, ui.BAR_H + GAP_P + HEAD, WW - GAP_P - 18, WH - GAP_P - EMPTY)    # the page's own margins: 438, 87, 974, 466

# colours, RGB 0-255
WHITE = np.array((255, 255, 255), np.float32)
INK = np.array((16, 16, 16), np.float32)
GREY = np.array((95, 95, 95), np.float32)
FAINT = np.array((170, 170, 170), np.float32)
HAIR = np.array((232, 232, 232), np.float32)      # rgba(0,0,0,.09) on white, film_page's RULE
CLAY = np.array((215, 119, 87), np.float32)       # the app's clay, #d77757: only 我, the dot of a call and what I point at


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def ramp(x, a, b, soft):
    """0 to 1 between a and b at one steady pace, softened over `soft` seconds at each end that has one (soft is (start, end))."""
    s0, s1 = soft
    d = b - a
    # a speed that rises over s0, holds, falls over s1, with area 1
    v = 1.0 / (d - s0 / 2 - s1 / 2)
    t = float(np.clip(x - a, 0, d))
    if t < s0:
        return v * t * t / (2 * s0)
    out = v * s0 / 2
    if t < d - s1:
        return out + v * (t - s0)
    out += v * (d - s0 - s1)
    u = t - (d - s1)
    return min(out + v * (u - u * u / (2 * s1)), 1.0)


# ---------------------------------------------------------------- the clock of the chapters

CLOCK = "10月30日 周五 20:13"            # the menu bar all 42.5 s: the evening of this conversation, as 四's cut lands on it (film_four.FIVE_CLOCK)


def _lines(chapter):
    return [c["t0"] for c in ink_subs.cues() if c["chapter"] == chapter]


_ch = {name: (a, b) for name, _, a, b in ink_subs.chapters()}
C5, C6, C7 = _ch["五"][0], _ch["六"][0], _ch["七"][0]
END = _ch["七"][1]
L5, L6, L7 = _lines("五"), _lines("六"), _lines("七")

# 五, seconds into the chapter
P1_AT, P2_AT, PACE = 0.5, 2.2, 26.0      # the reply's two paragraphs, and how fast a reply runs
CARD_AT = P2_AT + len(SAY[1]) / PACE + 0.2
MOVE0, MOVE1, PRESS = 4.1, 4.9, 4.95     # the pointer comes to the card and clicks it
OPEN0, OPEN1 = 5.05, 5.45                # the page opens beside the chat
PUSH0, PUSH1 = 7.0, 10.4                 # the field held a second, then in at one pace
ZF = 7.0                                 # how far in
POINT0, POINT1, CLICK = 11.0, 11.45, 11.5
SLIDE0, SLIDE1 = 11.55, 11.9             # 我's card comes in from the right
SCROLL0, SCROLL1 = 12.1, 14.5            # its numbers go up, and stop 1.5 s before the chapter ends

# 六, film seconds
SI = 20                                  # 思
LOOK = (4, 10, 16, SI)                   # 光 霜 月 思 look back in turn while the first line is up; 思's hardest look comes last and holds, then the cut to the guesses at L6[1]
LOOK0, LOOK_EVERY = 0.3, 1.15            # the first look, seconds into the line, and the time between one character's looks and the next's
ARC_GROW, ARC_STAGGER = 0.4, 0.08        # how long an arc takes to draw, and how far apart a character's arcs start
# 七
HOLD7 = C7 + 0.5                         # 差了 0.04 stays this long into 七, then the call
RUN1 = L7[1] + 3.3                       # the log has reached the last step
READ0 = L7[1] + 3.5                      # the picture-only line: what it writes now, to be read
TANG = L7[2]                             # the printed page


# ---------------------------------------------------------------- what the small one left

class Model:
    def __init__(self):
        v = json.load(open(os.path.join(LEARN, "vocab.json"), encoding="utf-8"))
        self.chars = v["chars"]
        self.ids = {c: i for i, c in enumerate(self.chars)}
        self.e = np.load(os.path.join(LEARN, "embed", "0120000.npy")).astype(np.float32)
        u = self.e / np.maximum(np.linalg.norm(self.e, axis=1, keepdims=True), 1e-6)
        self.unit = u
        self.map = np.load(os.path.join(DATA, "map_f12.npz"), allow_pickle=True)
        self.map_chars = json.loads(str(self.map["chars"]))
        self.map_ids = {c: i for i, c in enumerate(self.map_chars)}
        self.xy = self.map["xy"][-1].astype(np.float64)               # the last snapshot, step 120,000
        a = np.load(os.path.join(DATA, "attn.npz"), allow_pickle=True)
        self.w = a["w"]
        self.line = [str(c) for c in a["chars"]]
        self.micro = json.load(open(os.path.join(DATA, "micro.json"), encoding="utf-8"))
        self.log = [json.loads(x) for x in open(os.path.join(LEARN, "log.jsonl"), encoding="utf-8")]

    def nearest(self, c, k=6):
        s = self.unit @ self.unit[self.ids[c]]
        s[self.ids[c]] = s[0] = -9
        return [self.chars[i] for i in np.argsort(-s)[:k]]

    def looks(self, i, k=5, least=0.2):
        """Where a character of the line looks hardest, over the layers past the first and all their heads: the three hardest, and the ones furthest back."""
        best = {}
        for layer in range(1, self.w.shape[0]):
            for head in range(self.w.shape[1]):
                for j in range(i):
                    v = float(self.w[layer, head, i, j])
                    if v >= least and v > best.get(j, 0.0):
                        best[j] = v
        strong = sorted(best.items(), key=lambda kv: -kv[1])
        return strong[:3] + sorted(strong[3:], key=lambda kv: kv[0])[:k - 3]


_m = None


def model():
    global _m
    if _m is None:
        _m = Model()
    return _m


# Characters read per training step, counted, not estimated. ink_train.py: Data.batch draws BATCH 64 windows and the model is fed BLOCK 64 characters of
# each (x = xs[:, :-1]); the model is character-level and there is no gradient accumulation, so the state logged at step s has read s * 4096 characters.
# The steps shown: 120000 (the last checkpoint, 五六's map) -> 491,520,000 = 4.9 亿; log.jsonl "step" 1568 (七 at 180.3) -> 6,422,528 = 642 万;
# micro.json "ckpt" 0000709.pt (六's guesses) -> 2,904,064 = 290 万.
CHARS_PER_STEP = 64 * 64


def read(step):
    """读了 N 字: the characters the small one had read by this step, step times CHARS_PER_STEP."""
    n = step * CHARS_PER_STEP
    if round(n / 1e4) >= 1e4:
        return f"读了 {n / 1e8:.1f} 亿字"
    if n >= 1e6:
        return f"读了 {n / 1e4:,.0f} 万字"
    if n >= 1e4:
        return f"读了 {n / 1e4:.1f} 万字"
    return f"读了 {n:,} 字"


# ---------------------------------------------------------------- the window, as Chrome draws it

def card_html(width, opened=False):
    """My page, as a card in the reply."""
    bg = "#f6f6f6" if opened else "#fff"
    return f'''<div style="width:{width}px;height:{CARD_H}px;margin:2px 0 {ui.GAP}px;border:1px solid rgba(0,0,0,.12);border-radius:12px;box-sizing:border-box;background:{bg};display:flex;align-items:center;gap:12px;padding:0 12px">
      <span style="width:34px;height:34px;flex:none;border-radius:8px;background:#f0f0f0;color:{ui.GREY};display:flex;align-items:center;justify-content:center">{mac.icon("artifacts", 17)}</span>
      <span style="font-size:14px;font-weight:500;color:{ui.INK}">{NAME}</span></div>'''


def panel_html(left):
    x0, y0, x1, y1 = panel_rect(left)
    return f'''<div style="position:absolute;left:{x0}px;top:{y0}px;width:{x1 - x0}px;height:{y1 - y0}px;z-index:4;background:#fff;border:1px solid {ui.RULE};border-radius:12px;box-sizing:border-box">
      <div style="position:absolute;left:17px;top:0;height:{HEAD}px;display:flex;align-items:center;font-size:13.5px;font-weight:500;color:{ui.INK}">{NAME}</div></div>'''


def chat_html(said, u, streaming=False, card=False, opened=False, waiting=None, caret=False, blocks=None):
    """The window: what is said so far, and the page open by u (0 shut, 1 open)."""
    full = 1 - u
    cw = SPLIT - GAP_P + (WW - (SPLIT - GAP_P)) * full
    box_w = BOX_W_S + (ui.BOX_W - BOX_W_S) * full
    text_w = TEXT_W_S + (ui.TEXT_W - TEXT_W_S) * full
    pill_w = PILL_W_S + (ui.PILL_W - PILL_W_S) * full
    if blocks is None:
        blocks = ui.you(ASK, width=round(pill_w, 2))
        for i, p in enumerate(said):
            tail = ui.caret() if streaming and i == len(said) - 1 else ""
            blocks += ui.para(p, tail, width=round(text_w, 2))
        if card:
            blocks += card_html(CARD_W, opened)
    left = cw + GAP_P
    panel = panel_html(round(left, 2)) if u > 0 else ""
    return ui.window(blocks, typing=caret, title=TITLE, w=WW, h=WH, top=TOP, waiting=waiting,
                     chat_w=None if u == 0 else round(cw, 2), box_w=ui.BOX_W if u == 0 else round(box_w, 2), panel=panel)


class Cam:
    """The frame: zoom z about frame point c, drawn r times 1920 wide, at S times that."""

    def __init__(self, z=1.0, c=(W / 2, H / 2), r=1.0):
        self.z, self.c, self.r = z, c, r
        self.k = r * S
        self.size = (int(round(W * self.k)), int(round(H * self.k)))

    def px(self, fx, fy):
        return ((fx - self.c[0]) * self.z + W / 2) * self.k, ((fy - self.c[1]) * self.z + H / 2) * self.k

    def pt(self, x, y):
        return self.px(WX + x * Z, WY + y * Z)

    @property
    def scale(self):
        """Output pixels to a window point."""
        return Z * self.z * self.k


_desk, _base, _arrow = {}, {}, {}


def base(html, clock, r):
    """The desktop with the window on it, at r times 1920, times S: an RGB float array."""
    key = (html, clock, r)
    if key not in _base:
        if len(_base) > 6:
            _base.clear()
        zoom = Z * S * r
        if (clock, r) not in _desk:
            _desk[clock, r] = mac.desktop(app="Claude", clock=clock, show_dock=False, zoom=zoom).astype(np.float32)
        f = _desk[clock, r].copy()
        want = (int(round(H * S * r)), int(round(W * S * r)))
        if f.shape[:2] != want:             # the desktop comes a pixel short at some scales (1919 wide at r 0.5): carry its edge on
            f = f[:want[0], :want[1]]
            f = np.pad(f, ((0, want[0] - f.shape[0]), (0, want[1] - f.shape[1]), (0, 0)), mode="edge")
        layer = mac.snap(html, WW + 2 * PAD, WH + 2 * PAD, zoom).astype(np.float32)
        x, y = int(round((WX - PAD * Z) * S * r)), int(round((WY - PAD * Z) * S * r))
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + layer.shape[1], f.shape[1]), min(y + layer.shape[0], f.shape[0])
        part = layer[y0 - y:y1 - y, x0 - x:x1 - x]
        f[y0:y1, x0:x1] = f[y0:y1, x0:x1] * (1 - part[..., 3:] / 255) + part[..., :3] * part[..., 3:] / 255
        _base[key] = f
    return _base[key]


def background(html, clock, cam):
    b = base(html, clock, cam.r)
    if cam.z == 1.0 and tuple(cam.c) == (W / 2, H / 2):
        return b.copy()
    k = cam.k
    w, h = W / cam.z * k, H / cam.z * k
    im = Image.fromarray(np.uint8(np.clip(b + 0.5, 0, 255)))
    # inside the picture by construction (五's push stays on the frame); the clamp only takes off float noise
    box = (max(cam.c[0] * k - w / 2, 0.0), max(cam.c[1] * k - h / 2, 0.0), min(cam.c[0] * k + w / 2, im.width), min(cam.c[1] * k + h / 2, im.height))
    return np.asarray(im.resize(cam.size, Image.LANCZOS, box=box), np.float32).copy()


def pointer(f, cam, x, y, press=0.0):
    """The Mac's arrow, its tip on window point x, y."""
    zoom = Z * cam.k
    if zoom not in _arrow:
        _arrow[zoom] = mac.snap(mac.pointer_html(), 24, 30, zoom).astype(np.float32)
    a = _arrow[zoom]
    px, py = cam.pt(x, y)
    px, py = int(round(px - 4.2 * zoom)), int(round(py - 2.6 * zoom))
    y0, y1, x0, x1 = max(py, 0), min(py + a.shape[0], f.shape[0]), max(px, 0), min(px + a.shape[1], f.shape[1])
    if y0 < y1 and x0 < x1:
        p = a[y0 - py:y1 - py, x0 - px:x1 - px]
        f[y0:y1, x0:x1] = f[y0:y1, x0:x1] * (1 - p[..., 3:] / 255) + p[..., :3] * p[..., 3:] / 255


class Layer:
    """Part of the window drawn here, not by Chrome: a rectangle of window points, clipped to the frame."""

    def __init__(self, f, cam, rect, fill=WHITE):
        self.f, self.cam = f, cam
        x0, y0 = cam.pt(rect[0], rect[1])
        x1, y1 = cam.pt(rect[2], rect[3])
        self.X0, self.Y0 = max(int(math.floor(x0)), 0), max(int(math.floor(y0)), 0)
        self.X1, self.Y1 = min(int(math.ceil(x1)), f.shape[1]), min(int(math.ceil(y1)), f.shape[0])
        self.ok = self.X0 < self.X1 and self.Y0 < self.Y1
        self.cv = f[self.Y0:self.Y1, self.X0:self.X1] if self.ok else np.zeros((1, 1, 3), np.float32)
        if fill is not None and self.ok:
            self.cv[:] = fill

    def xy(self, x, y):
        px, py = self.cam.pt(x, y)
        return px - self.X0, py - self.Y0

    def px(self, size):
        return max(int(round(size * self.cam.scale)), 1)

    def put(self, ch, x, y, size, colour, op=1.0, kind="song"):
        """A character centred on x, y, size points."""
        if self.ok:
            px, py = self.xy(x, y)
            _put(self.cv, ch, px, py, self.px(size), colour, op, kind)

    def text(self, s, x, y, size, colour, op=1.0, kind="ping", right=False, centre=False):
        """A run of text, its top left at x, y. Returns where it ended, in points."""
        if not self.ok:
            return x
        px, py = self.xy(x, y)
        end = _text(self.cv, s, px, py, self.px(size), colour, op, kind, right, centre)
        return x + (end - px) / self.cam.scale

    def mono(self, s, x, y, size, colour, op=1.0, cjk=None):
        """A terminal's line: every character one cell of Menlo, the wide ones two."""
        if not self.ok:
            return x
        f = face("mono", self.px(size))
        adv = f.getlength("0")
        px, py = self.xy(x, y)
        start = px
        for ch in s:
            if ord(ch) > 0x2E7F:
                _put(self.cv, ch, px + adv, py + self.px(size) * 0.58, self.px(size), colour if cjk is None else cjk, op, "ping")
                px += 2 * adv
            else:
                a, ox, oy = glyph(ch, "mono", self.px(size))
                stamp(self.cv, a, px + ox, py + oy, colour, op)
                px += adv
        return x + (px - start) / self.cam.scale

    def rect(self, x0, y0, x1, y1, colour, op=1.0, r=0.0, line=0.0):
        """A rectangle, its corners rounded by r points; line: only its edge, this many points wide."""
        if not self.ok:
            return
        a0, b0 = self.xy(x0, y0)
        a1, b1 = self.xy(x1, y1)
        rr = r * self.cam.scale
        lw = line * self.cam.scale
        h, w = self.cv.shape[:2]
        X0, Y0, X1, Y1 = max(int(a0) - 1, 0), max(int(b0) - 1, 0), min(int(a1) + 2, w), min(int(b1) + 2, h)
        if X0 >= X1 or Y0 >= Y1:
            return
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32) + 0.5
        rr = min(rr, (a1 - a0) / 2, (b1 - b0) / 2)
        qx = np.abs(xx - (a0 + a1) / 2) - ((a1 - a0) / 2 - rr)
        qy = np.abs(yy - (b0 + b1) / 2) - ((b1 - b0) / 2 - rr)
        sd = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rr      # how far outside the shape, negative inside
        al = np.clip(0.5 - sd, 0, 1)
        if line:
            al = al * np.clip(sd + lw + 0.5, 0, 1)
        al = (al * op)[..., None]
        part = self.cv[Y0:Y1, X0:X1]
        part += (colour - part) * al

    def shadow(self, x0, y0, x1, y1, r, spread, op):
        """The soft shadow a card casts on the page."""
        if not self.ok:
            return
        a0, b0 = self.xy(x0, y0)
        a1, b1 = self.xy(x1, y1)
        s = spread * self.cam.scale
        h, w = self.cv.shape[:2]
        X0, Y0, X1, Y1 = max(int(a0 - 3 * s), 0), max(int(b0 - 2 * s), 0), min(int(a1 + 3 * s), w), min(int(b1 + 4 * s), h)
        if X0 >= X1 or Y0 >= Y1:
            return
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32) + 0.5
        yy = yy - s * 0.5
        dx = np.maximum(np.maximum(a0 - xx, xx - a1), 0)
        dy = np.maximum(np.maximum(b0 - yy, yy - b1), 0)
        al = (np.exp(-(np.hypot(dx, dy) / max(s, 1e-3)) ** 2) * op)[..., None]
        part = self.cv[Y0:Y1, X0:X1]
        part += (INK - part) * al * (np.hypot(dx, dy) > 0)[..., None]

    def fade(self, rect, top=0.0, bottom=0.0, left=0.0, right=0.0, colour=WHITE):
        """White comes in over the last few points at the edges of a rectangle of the layer, so that what the edge cuts goes out softly."""
        if not self.ok:
            return
        a0, b0 = self.xy(rect[0], rect[1])
        a1, b1 = self.xy(rect[2], rect[3])
        h, w = self.cv.shape[:2]
        X0, Y0, X1, Y1 = max(int(a0), 0), max(int(b0), 0), min(int(math.ceil(a1)), w), min(int(math.ceil(b1)), h)
        if X0 >= X1 or Y0 >= Y1:
            return
        sc = self.cam.scale
        yy = np.arange(Y0, Y1, dtype=np.float32)[:, None] + 0.5
        xx = np.arange(X0, X1, dtype=np.float32)[None, :] + 0.5
        k = np.ones((Y1 - Y0, X1 - X0), np.float32)
        for edge, d in ((top, yy - b0), (bottom, b1 - yy), (left, xx - a0), (right, a1 - xx)):
            if edge:
                k = k * np.clip(d / (edge * sc), 0, 1)
        part = self.cv[Y0:Y1, X0:X1]
        part += (colour - part) * (1 - k * k * (3 - 2 * k))[..., None]

    def lines(self, pts, width, colour, op=1.0):
        """A path through window points, width in points."""
        if not self.ok or len(pts) < 2:
            return
        h, w = self.cv.shape[:2]
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).line([self.xy(*p) for p in pts], fill=255, width=max(int(round(width * self.cam.scale)), 1), joint="curve")
        al = (np.asarray(m, np.float32) / 255 * op)[..., None]
        self.cv += (colour - self.cv) * al


def finish(f, cam):
    """Down from S times to the frame."""
    im = Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255)))
    out = (int(round(W * cam.r)), int(round(H * cam.r)))
    return np.asarray(im.resize(out, Image.LANCZOS))


# ---------------------------------------------------------------- 五

def _reply(x):
    """What the chat holds x seconds into 五, before the page opens."""
    said, streaming, waiting = [], False, None
    if x < P1_AT:
        said, streaming = [""], True             # the reply not begun: its caret where the words will come, as 四's cut lands on it
    else:
        n1 = int((x - P1_AT) * PACE)
        said = [SAY[0][:n1]] if n1 < len(SAY[0]) else [SAY[0]]
        streaming = n1 < len(SAY[0])
        if x >= P2_AT:
            n2 = int((x - P2_AT) * PACE)
            said.append(SAY[1][:max(n2, 1)])
            streaming = n2 < len(SAY[1])
    return said, streaming, waiting


def _card_point():
    """The middle of the card in the chat, in window points, before the page opens: the pill, the two paragraphs, the card."""
    text_left = WW / 2 - (ui.BOX_W / 2 - ui.BOX_LEFT)
    y = TOP + 42 + ui.GAP + 4 + 2 * (ui.LINE + ui.GAP) + 2
    return text_left + CARD_W / 2, y + CARD_H / 2


def caret_box(u=0.0):
    """Where the caret in the empty box is, window points: the box's padding, the plus and its gap."""
    text_left = WW / 2 - (ui.BOX_W / 2 - ui.BOX_LEFT)
    return text_left - ui.BOX_LEFT + 14 + 16 + 11 + 1, WH - 30 - 25


class Field:
    """The 1,500 commonest characters where the last snapshot put them, turned so that 我 and its kind sit left of the middle; the push goes in on them."""

    def __init__(self, target="我", kind="我予吾君余尔汝他彼"):
        m = model()
        xy = m.xy
        c = np.median(xy[[m.map_ids[ch] for ch in kind if ch in m.map_ids]], 0)
        # turn the map so that the cluster faces left: the plane has no up or down of its own
        a = math.atan2(c[1] - xy[:, 1].mean(), c[0] - xy[:, 0].mean())
        turn = math.pi - a
        R = np.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]])
        q = (xy - xy.mean(0)) @ R.T
        lo, hi = q.min(0), q.max(0)
        x0, y0, x1, y1 = CONTENT
        k = min((x1 - x0) / (hi[0] - lo[0]), (y1 - y0) / (hi[1] - lo[1]))
        p = (q - lo) * k
        span = (hi - lo) * k
        p[:, 0] += x1 - span[0]                          # right against the page's margin: the slack goes to the left
        p[:, 1] += y0 + ((y1 - y0) - span[1]) / 2
        self.p = p
        self.chars = m.map_chars
        self.ids = m.map_ids
        self.target = target
        self.kind = kind
        self.centre = np.median(p[[self.ids[ch] for ch in kind if ch in self.ids]], 0)
        n = len(self.chars)
        rank = np.arange(n)
        self.size = 6.0 + 5.5 * np.exp(-rank / 160)
        self.op = 0.42 + 0.5 * np.exp(-rank / 380)
        self.grow = 0.5
        self.apart = self._apart(ZF)

    def _apart(self, z):
        """Where characters would sit on one another once in close, how far to move them apart so each can be read. Nearness is kept."""
        q = self.centre + (self.p - self.centre) * z
        s = self.size * z ** self.grow
        x0, y0, x1, y1 = CONTENT
        near = np.nonzero((q[:, 0] > x0 - 120) & (q[:, 0] < x1 + 120) & (q[:, 1] > y0 - 120) & (q[:, 1] < y1 + 120))[0]
        r = q[near].copy()
        need = (s[near][:, None] + s[near][None]) * 0.62
        far = np.eye(len(near)) * 1e6
        for _ in range(160):
            d = r[:, None] - r[None]
            dist = np.sqrt((d * d).sum(-1)) + far + 1e-3
            push = np.clip(need - dist, 0, None)[..., None] * d / dist[..., None]
            r = r + 0.3 * push.sum(1)
        out = np.zeros_like(self.p)
        out[near] = (r - q[near]) / z
        return out

    def at(self, z):
        """Window points of every character, and their sizes, at zoom z about the cluster."""
        k = float(np.clip((z - 1) / (ZF - 1), 0, 1))
        q = self.centre + (self.p + self.apart * ease(k) - self.centre) * z
        return q, self.size * z ** self.grow

    def draw(self, L, z, dim=1.0):
        q, s = self.at(z)
        mine = self.ids.get(self.target)
        x0, y0, x1, y1 = inner()
        for i in range(len(self.chars) - 1, -1, -1):
            x, y = q[i]
            if i == mine or x < x0 - s[i] or x > x1 + s[i] or y < y0 - s[i] or y > y1 + s[i]:
                continue
            L.put(self.chars[i], x, y, s[i], INK, float(self.op[i]) * dim)
        if mine is not None:
            L.put(self.target, q[mine][0], q[mine][1], s[mine] * 1.08, CLAY, 1.0, "songb")
        return q[mine] if mine is not None else None


_fields = {}


def field(target="我", kind="我予吾君余尔汝他彼"):
    if kind not in _fields:
        _fields[kind] = Field(target, kind)
    return _fields[kind]


def hud(f, cam, step, op=1.0):
    """读了 N 字, at the right of the page's title line."""
    x0, y0, x1, _ = panel_rect()
    T = Layer(f, cam, (x0 + 1, y0 + 1, x1 - 1, y0 + HEAD), fill=None)
    T.text(read(step), x1 - 18, y0 + (HEAD - 12.5) / 2 - 1.5, 12.5, GREY, op, "ping", right=True)


def five_z(x):
    return math.exp(math.log(ZF) * ramp(x, PUSH0, PUSH1, (0.5, 0.5)))


def me_card(L, x):
    """我, clicked: a white card from the right, its 384 numbers going slowly up and then still."""
    m = model()
    u = ease((x - SLIDE0) / (SLIDE1 - SLIDE0))
    if u <= 0:
        return
    x0p, y0p, x1p, y1p = inner()
    w = 318
    cx0 = x1p - w + (w + 24) * (1 - u)
    y0, y1 = y0p, y1p
    S_ = Layer(L.f, L.cam, (x0p, y0p, x1p, panel_rect()[3] - 1), fill=None)
    S_.shadow(cx0, y0 - 20, cx0 + w + 40, panel_rect()[3] + 20, 0, 7, 0.07)
    S_.rect(cx0, y0, x1p + 1, y1, WHITE, 1.0)          # below this the page is white already and its corner is round
    S_.rect(cx0, y0, cx0 + 1, panel_rect()[3] - 1, np.array((226, 226, 226), np.float32), 1.0)
    cx0 += 8
    y0 += 6
    L.put("我", cx0 + 46, y0 + 46, 50, CLAY, 1.0, "songb")
    L.text("384 个数", cx0 + 88, y0 + 22, 14, INK, 1.0, "ping")
    L.text("挨着的字　" + " ".join(m.nearest("我")), cx0 + 88, y0 + 48, 12.5, GREY, 1.0, "ping")
    top = y0 + 94
    L.rect(cx0 + 16, top - 10, cx0 + w - 16, top - 9.4, HAIR, 1.0)
    row = m.e[m.ids["我"]]
    # the numbers, five to a row, in a window of the card that they go up through
    sub = Layer(L.f, L.cam, (cx0 + 1, top - 9, cx0 + w - 10, y1), fill=None)
    k = ramp(x, SCROLL0, SCROLL1, (0.6, 0.8))
    scroll = 6.2 * 19 * k
    pitch = 56.5
    for i in range(0, len(row), 5):
        y = top + 4 + (i // 5) * 19 - scroll
        if y < top - 30 or y > y1:
            continue
        for j, v in enumerate(row[i:i + 5]):
            sub.mono(f"{v:+.3f}", cx0 + 16 + j * pitch, y, 11.5, INK if v >= 0 else GREY)
    # the list goes on under the hairline and past the foot: it goes out softly at both
    sub.fade((cx0 + 1, top - 9, cx0 + w - 10, y1), top=12, bottom=46)


def five(t, cam=None, alt=False):
    """五, film time t."""
    x = t - C5
    cam = cam or Cam()
    clock = CLOCK
    if x < OPEN0:
        said, streaming, waiting = _reply(x)
        html = chat_html(said, 0.0, streaming, card=x >= CARD_AT, opened=x >= PRESS, waiting=waiting, caret=False)
        f = background(html, clock, cam)
        if x >= MOVE0:
            u = ease((x - MOVE0) / (MOVE1 - MOVE0))
            sx, sy = _card_point()[0] + 250, WH - 70
            tx, ty = _card_point()
            px, py = sx + (tx - sx) * u, sy + (ty - sy) * u - 40 * math.sin(math.pi * u)
            pointer(f, cam, px, py)
        return finish(f, cam)
    u = ease((x - OPEN0) / (OPEN1 - OPEN0))
    html = chat_html(SAY, u, card=True, opened=True, caret=False)
    f = background(html, clock, cam)
    left = (SPLIT - GAP_P + (WW - (SPLIT - GAP_P)) * (1 - u)) + GAP_P
    L = Layer(f, cam, inner(left))
    kind = "父母兄弟夫妇儿女妻男姑子" if alt else "我予吾君余尔汝他彼"
    F = field("我", kind)
    z = five_z(x)
    me = F.draw(L, z)
    L.fade(inner(left), top=16, bottom=16, left=12, right=12)
    if u >= 1:
        hud(f, cam, 120000)
    me_card(L, x)
    px, py = _card_point()
    if x >= POINT0 and me is not None:
        k = ease((x - POINT0) / (POINT1 - POINT0))
        tx, ty = me[0] + 2, me[1] + 4
        px, py = px + (tx - px) * k, py + (ty - py) * k - 30 * math.sin(math.pi * k)
    pointer(f, cam, px, py)
    return finish(f, cam)


# ---------------------------------------------------------------- 六

CELL6 = 20.0
SIZE6 = 17.5
LINE_Y6 = 384.0
LEFT6 = CONTENT[0] + 4                   # the line starts at the page's margin


def line_x(i, cell=CELL6, n=22, left=None):
    x0 = (CONTENT[0] + CONTENT[2]) / 2 - n * cell / 2 if left is None else left
    return x0 + (i + 0.5) * cell


def arc_h(dx):
    return 14 + 0.36 * abs(dx)


def arc_pts(i, j, s1, y, cell=CELL6, s0=0.0):
    """The arc from character i to j, from s0 to s1 of the way, above the line at y. What is looked at flows into what looks: the arc goes from the earlier character to the later one, up and to the right first."""
    xa, xb = line_x(i, cell, left=LEFT6), line_x(j, cell, left=LEFT6)
    h = arc_h(xa - xb)
    n = max(int(64 * (s1 - s0)), 2)
    return [(xa + (xb - xa) * s, y - h * math.sin(math.pi * s)) for s in np.linspace(s0, s1, n)]


def look_times():
    """When each character of LOOK starts to look back, film seconds, and for each of its arcs when it starts. The last character's hardest look starts
    after its others, so that it is the last thing drawn before the cut."""
    out = []
    for n, i in enumerate(LOOK):
        begin = L6[0] + LOOK0 + n * LOOK_EVERY
        arcs = model().looks(i)
        if n == len(LOOK) - 1:
            arcs = arcs[1:] + arcs[:1]
        out.append((i, begin, [(j, v, begin + k * ARC_STAGGER + (ARC_STAGGER if n == len(LOOK) - 1 and k == len(arcs) - 1 else 0.0)) for k, (j, v) in enumerate(arcs)]))
    return out


def _look(L, t):
    top = LINE_Y6 - SIZE6 * 0.62
    looking = None
    times = look_times()
    for n, (i, begin, arcs) in enumerate(times):
        if t < begin:
            break
        now = n == len(times) - 1 or t < times[n + 1][1]
        for j, v, start in arcs:
            k = ease((t - start) / ARC_GROW)
            if k <= 0:
                continue
            L.lines(arc_pts(j, i, k, top), 0.6 + 2.6 * v if now else 0.7, CLAY if now else FAINT, min(0.35 + 0.7 * v, 1.0) if now else 0.7)
        if now:
            looking = i
    for i, ch in enumerate(model().line):
        L.put(ch, line_x(i, left=LEFT6), LINE_Y6, SIZE6, CLAY if i == looking else INK, 1.0, "songb" if i == looking else "song")


def six(t, cam=None, r=1.0):
    """读一句话: 光 霜 月 思 look back in turn, 思's hardest look last, held a beat once drawn; at L6[1] a hard cut to the guesses."""
    html = chat_html(SAY, 1.0, card=True, opened=True)
    cam = cam or Cam(r=r)
    f = background(html, CLOCK, cam)
    L = Layer(f, cam, inner())
    if t < L6[1]:
        _look(L, t)
        hud(f, cam, 120000)
    else:
        guess(L, t)
    return finish(f, cam)


GUESS_Y = 128.0
BAR0, BAR_W = 498.0, 390.0


def guess(L, t):
    """下一个字，我来猜 and 猜错了，就改几个数: an early snapshot's guesses, the right one, one change of all its numbers, what it guesses after."""
    m = model()
    M = m.micro
    a, b = M["records"][0], M["records"][1]
    step = int(M["ckpt"].split(".")[0])
    hud(L.f, L.cam, step)
    wrong = t >= L6[2]
    after = ease((t - L6[2] - 2.0) / 0.8)
    # the line, and the place after it
    for i, ch in enumerate(m.line):
        L.put(ch, line_x(i, 23.0), GUESS_Y, 19, INK)
    bx = line_x(len(m.line), 23.0)
    L.rect(bx - 11.5, GUESS_Y - 12.5, bx + 11.5, GUESS_Y + 12.5, CLAY if wrong else HAIR, 1.0, 4, line=0.9)
    pa = {x["char"]: x["probability"] for x in a["top8"][:5]}
    pb = {x["char"]: x["probability"] for x in b["top8"][:5]}
    best_a = max(pa, key=pa.get)
    pick = ease((t - L6[1] - 1.5) / 0.3)
    if not wrong:
        if pick > 0:
            L.put(best_a, bx, GUESS_Y, 19, INK, pick)
    else:
        swap = after
        L.put(best_a, bx, GUESS_Y, 19, GREY, 1 - swap)
        if swap > 0:
            L.put(M["target"], bx, GUESS_Y, 19, CLAY, swap, "songb")
    # the guesses: five, as bars, ranked; after the change they go where they then are
    ra = {c: i for i, c in enumerate(pa)}
    rb = {c: i for i, c in enumerate(pb)}
    grow = ease((t - L6[1] - 0.2) / 0.6)
    y0, pitch = 172.0, 27.0
    for c in dict.fromkeys(list(pa) + list(pb)):
        r = ra.get(c, 5.4) + (rb.get(c, 5.4) - ra.get(c, 5.4)) * after
        p = (pa.get(c, 0.0) + (pb.get(c, 0.0) - pa.get(c, 0.0)) * after) * grow
        if r > 4.6:
            continue
        y = y0 + r * pitch
        hit = wrong and c == M["target"]
        col = CLAY if hit else INK
        L.put(c, CONTENT[0] + 26, y, 17, col, 1.0, "songb" if hit else "song")
        L.rect(BAR0, y - 5, BAR0 + max(p * BAR_W, 1.5), y + 5, CLAY if hit else np.array((196, 196, 196), np.float32), 1.0, 1.2)
        L.mono(f"{p * 100:.1f}%", BAR0 + max(p * BAR_W, 2.0) + 10, y - 7, 11.5, col if hit else GREY)
    if not wrong:
        return
    # how far off it was, and a few of the numbers it changed: the ones that moved most, of the two characters that moved most
    k = ease((t - L6[2] - 0.25) / 0.3)
    loss = a["loss"] + (b["loss"] - a["loss"]) * after
    yb = 340.0
    L.text("差了", CONTENT[0] + 16, yb + 6, 13, GREY, k)
    L.mono(f"{loss:.2f}", CONTENT[0] + 56, yb - 6, 30, INK, k)
    v = M["first_step"]["vectors"]
    rows = []
    for c in ("故", "思"):
        before, now = np.array(v[c]["before"]), np.array(v[c]["after"])
        for i in np.argsort(-np.abs(now - before))[:2 if c == "故" else 1]:
            rows.append((c, int(i), float(before[i]), float(now[i])))
    rows = [rows[0], rows[2], rows[1]]
    done = ease((t - L6[2] - 1.0) / 0.9)
    for n, (c, i, p, q) in enumerate(rows):
        kk = ease((t - L6[2] - 0.5 - n * 0.12) / 0.3)
        y = yb - 8 + n * 25
        L.put(c, 700, y + 9, 15, INK, kk)
        L.text(f"第 {i + 1} 个数", 716, y + 1, 12, GREY, kk)
        L.mono(f"{p:+.3f}", 790, y, 12, GREY, kk)
        L.text("→", 852, y + 1, 12, GREY, kk)
        L.mono(f"{p + (q - p) * done:+.3f}", 872, y, 12, INK, kk)


# ---------------------------------------------------------------- 七

def _log_text(r):
    """A line of the run's log, as ink_train printed it: the step, how far off, what it wrote after 低头思, what it wrote after 春, to the first line break."""
    return f"{r['step']:>6} {r['loss']:.3f} ", r["lines"]["低头思"], r["lines"]["春"].split("\n")[0]


def _rows_shown(t):
    n = len(model().log)
    x = (t - HOLD7) / (RUN1 - HOLD7)
    return int(np.clip(np.interp(x, [0.0, 0.22, 0.82, 1.0], [1, 13, n - 3, n]), 1, n) + 1e-6)


def _call(L, x, y):
    """A call of mine, as Claude Code sets it: the dot, what I ran; under it the elbow and what came back first."""
    L.rect(x, y + 5.5, x + 7, y + 12.5, CLAY, 1.0, 3.5)
    L.mono("python ink_train.py train 120000", x + 16, y + 1, 12, INK)
    L.rect(x + 18, y + 22, x + 19, y + 31, FAINT, 1.0)
    L.rect(x + 18, y + 30, x + 27, y + 31, FAINT, 1.0)
    L.text("全唐诗　6,369 个字　13,118,208 个数", x + 34, y + 21, 12.5, GREY)


def _cells(s):
    return sum(2 if ord(c) > 0x2E7F else 1 for c in s)


def _fit(s, cells):
    out = ""
    for c in s:
        if _cells(out + c) > cells:
            break
        out += c
    return out


LOG_SIZE = 10.5


def _log(L, t, x, y0, y1, right):
    m = model()
    k = _rows_shown(t)
    pitch = 15.5
    room = int((y1 - y0) // pitch)
    first = max(k - room, 0)
    cells = int((right - x) / (0.6021 * LOG_SIZE))
    for n, r in enumerate(m.log[first:k]):
        y = y0 + n * pitch
        head, low, spring = _log_text(r)
        end = L.mono(head, x, y, LOG_SIZE, GREY)
        end = L.mono(low, end, y, LOG_SIZE, INK)
        L.mono(_fit(" " + spring, cells - _cells(head + low)), end, y, LOG_SIZE, INK)


def _spring(r):
    """What it wrote after 春, to the last full stop if it has one."""
    s = r["lines"]["春"].split("\n")[0]
    if "。" in s:
        s = s[:s.rindex("。") + 1]
    return s


def _now(L, t):
    """The page during the run: what the small one writes now, after 春, big enough to read."""
    m = model()
    r = m.log[_rows_shown(t) - 1]
    hud(L.f, L.cam, r["step"])
    # the page takes the newest line it can read, not every one going by: it changes at most every 0.35 s
    r = m.log[_rows_shown(min(HOLD7 + math.floor((t - HOLD7) / 0.35) * 0.35, RUN1) if t < RUN1 else t) - 1]
    L.text("它现在接着「春」写", CONTENT[0] + 6, CONTENT[1] + 30, 13, GREY)
    s = _spring(r)
    # couplets on their own lines when it makes them, else rows of twelve
    parts = [p + "。" for p in s.split("。") if p] if "。" in s else [s[i:i + 12] for i in range(0, len(s), 12)]
    size, pitch = 25.0, 46.0
    y = (CONTENT[1] + CONTENT[3]) / 2 - (len(parts) - 1) * pitch / 2 + 10
    for n, p in enumerate(parts[:4]):
        x = CONTENT[0] + 30
        for ch in p:
            L.put(ch, x + size / 2, y + n * pitch, size, INK)
            x += size * 1.04


def seven(t, cam=None, r=1.0):
    if t >= TANG:
        return tang_placeholder(t, r)
    clock = CLOCK
    cam = cam or Cam(r=r)
    if t < HOLD7:
        html = chat_html(SAY, 1.0, card=True, opened=True)
        f = background(html, clock, cam)
        L = Layer(f, cam, inner())
        guess(L, t)
        return finish(f, cam)
    html = chat_html((), 1.0, blocks="")
    f = background(html, clock, cam)
    chat = Layer(f, cam, (24, ui.BAR_H + 10, SPLIT - GAP_P - 4, WH - ui.BOX_ROOM - 14), fill=None)
    _call(chat, 40, 70)
    _log(chat, t, 40, 118, WH - ui.BOX_ROOM - 14, SPLIT - GAP_P - 14)
    L = Layer(f, cam, inner())
    _now(L, t)
    return finish(f, cam)


def tang_placeholder(t, r=1.0):
    """The printed page: the real opening of 《唐詩三百首註疏》 卷六上 (Harvard drs:437097988), its ink taken off the scan by film_tang.py and lit by
    film_tang3d.py. The sun's sweep is tang/seq/NNNN.png at 30 fps; the frames after the last one rendered hold it, since only the sun moves and it
    has stopped on 低頭思故鄉."""
    seq = os.path.join(OUT, "tang", "seq")
    have = sorted(int(n[:4]) for n in os.listdir(seq) if n.endswith(".png") and n[:4].isdigit()) if os.path.isdir(seq) else []
    want = max(int(round((t - TANG) * FPS)), 0)
    have = [i for i in have if i <= want]
    path = os.path.join(seq, "%04d.png" % have[-1]) if have else os.path.join(OUT, "tang", "tang_still.png")
    if os.path.exists(path):
        im = Image.open(path).convert("RGB").resize((int(W * r), int(H * r)), Image.LANCZOS)
        return np.asarray(im)
    im = Image.new("RGB", (int(W * r), int(H * r)), (238, 233, 222))
    d = ImageDraw.Draw(im)
    d.text((W * r * 0.08, H * r * 0.12), "占位：印好的唐诗页（film_tang.py）", (70, 62, 52), font=face("ping", int(34 * r)))
    return np.asarray(im)


# ---------------------------------------------------------------- all of it

def frame(t, r=1.0):
    if t < C6:
        return five(t, Cam(r=r))
    if t < C7:
        return six(t, r=r)
    return seven(t, r=r)


def stills(ts, alt=False):
    out = os.path.join(OUT, "stills")
    os.makedirs(out, exist_ok=True)
    for t in ts:
        f = five(t, Cam(), alt=True) if alt else frame(t)
        f = ink_subs.apply(np.ascontiguousarray(f), t)
        name = f"learn_{t:07.2f}{'_alt' if alt else ''}.png"
        Image.fromarray(f).save(os.path.join(out, name))
        print(os.path.join(out, name), flush=True)


def sheet(a, b, n, cols=4):
    out = os.path.join(OUT, "stills")
    os.makedirs(out, exist_ok=True)
    ts = np.linspace(a, b, n)
    rows = (n + cols - 1) // cols
    im = Image.new("RGB", (cols * 480, rows * 270))
    for i, t in enumerate(ts):
        f = frame(float(t), r=0.5)
        im.paste(Image.fromarray(f).resize((480, 270), Image.LANCZOS), (i % cols * 480, i // cols * 270))
    path = os.path.join(out, f"sheet_{a:.1f}_{b:.1f}.png")
    im.save(path)
    print(path, [round(float(t), 2) for t in ts])


def clip(a=C5, b=END, full=False):
    os.makedirs(OUT, exist_ok=True)
    whole = (a, b) == (C5, END)
    r = 1.0 if full else 0.5
    path = os.path.join(OUT, ("learn" if whole else f"learn_{a:.1f}_{b:.1f}") + ("_1920" if full else "") + ".mp4")
    w, h = int(W * r), int(H * r)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for n in range(int(round((b - a) * FPS))):
        t = a + n / FPS
        enc.stdin.write(np.ascontiguousarray(frame(t, r=r)).tobytes())
        if n % 60 == 0:
            print(f"{t:.2f}", flush=True)
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError(f"encoding failed: {path}")
    print(path)


def geometry():
    x, y = reply_caret()
    print(f"五 first frame: the reply's caret, window point ({x:.1f}, {y:.1f}) = frame px ({WX + x * Z:.1f}, {WY + y * Z:.1f}); clock {CLOCK}")
    for i, begin, arcs in look_times():
        print(f"六 {model().line[i]} looks back from {begin:.2f}: " + ", ".join(f"{model().line[j]} {v:.2f} at {start:.2f}" for j, v, start in arcs))
    i, begin, arcs = look_times()[-1]
    done = arcs[-1][2] + ARC_GROW
    print(f"  {model().line[i]}'s hardest look is drawn by {done:.2f} and held {L6[1] - done:.2f} s; the cut to the guesses at {L6[1]:.2f}")


def reply_caret():
    """Where the reply's caret stands before its first word, window points: the left of the reply's text, the line under the question."""
    text_left = WW / 2 - (ui.BOX_W / 2 - ui.BOX_LEFT)
    return text_left, TOP + 42 + ui.GAP + ui.LINE / 2

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "still":
        stills([float(v) for v in sys.argv[2:]])
    elif what == "alt":
        stills([float(v) for v in sys.argv[2:]], alt=True)
    elif what == "sheet":
        sheet(float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 4)
    elif what == "clip":
        span = [float(v) for v in sys.argv[2:] if v != "--full"]
        clip(*(span[:2] if len(span) >= 2 else (C5, END)), full="--full" in sys.argv)
    elif what == "geometry":
        geometry()
    else:
        raise SystemExit(__doc__)
