"""五 六 七 in my own window: you ask how I learned to speak. What happened to me I cannot see, so I raise a small one the same way and lay it open in front of you. It is really trained, on the Studio by ink_train.py, and every number on the screen is read from what it left behind (ink_learn_data.py).
Text is drawn here at twice the frame's pixels, the window round it by ink_mac.

    python ink_learn.py still T [T ...]   # frames at film times, ink/film_out/learn/shots/
    python ink_learn.py strip A B [N]     # N frames between two film times, ink/film_out/learn/shots/strip.png
"""
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ink_mac as mac
from ink_grade import sm
from ink_subs import at, chapters, cues

HERE = os.path.dirname(os.path.abspath(__file__))
LEARN = os.path.join(HERE, "..", "..", "ink", "film_out", "learn")
DATA = os.path.join(LEARN, "data")
W, H = 1920, 1080
Z = mac.Z
PAD = mac.PAD
CLOCK = "10月30日 周五 20:13"

# the window, in frame pixels and points: subtitles keep the right of the screen to themselves
WIN_X, WIN_Y = 30, 56
WIN_W, WIN_H = 945, 535
BAR = 38
CW, CH = int(round(WIN_W * Z)) * 2, int(round((WIN_H - BAR) * Z)) * 2      # the page, at twice the frame's pixels
RADIUS = 14 * Z * 2

BG = np.array((19, 19, 18), np.float32)
TEXT = np.array((226, 224, 215), np.float32)
SOFT = np.array((150, 148, 140), np.float32)
DIM = np.array((96, 95, 90), np.float32)
CLAY = np.array((226, 128, 94), np.float32)
BLUE = np.array((132, 176, 222), np.float32)

# 五六七 as they are cut now are film_learn.py's. The drafts below are kept, anchored to the chapters' own lines by their order, not their words, so that rewriting a line
# never breaks the import (film_weibo takes Screen and CLOCK from here)
def _starts(chapter):
    return [c["t0"] for c in cues() if c["chapter"] == chapter]


_FIVE, _SIX, _SEVEN = _starts("五"), _starts("六"), _starts("七")
T5A = next(a for name, _, a, _ in chapters() if name == "五")
T5, T5D = _FIVE[0], _FIVE[-1]
T5B, T5C, T5E = T5, T5D, T5D + 5.0
T6, T6B, T6C, T6D = (_SIX + [_SIX[-1]] * 4)[:4]
T7, T7B = _SEVEN[0], _SEVEN[-1]
CHAT = T5 - T5A                      # the picture-only line at the head of 五
T_RUN = T7
BATCH_CHARS = 64 * 64                # how many characters a step reads, by ink_train: BATCH * BLOCK


# ---------------------------------------------------------------- type

def _ping():
    return glob.glob("/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/PingFang.ttc")[0]


FONTS = dict(song=("/System/Library/Fonts/Supplemental/Songti.ttc", 6), songb=("/System/Library/Fonts/Supplemental/Songti.ttc", 1), mono=("/System/Library/Fonts/Menlo.ttc", 0), monob=("/System/Library/Fonts/Menlo.ttc", 1))
_faces, _glyphs, _runs, _missing = {}, {}, {}, {}
FALLBACK = ("song", "ping", "hei")


def face(kind, size):
    k = (kind, size)
    if k not in _faces:
        if kind == "ping":
            path, index = _ping(), 3
        elif kind == "pingm":
            path, index = _ping(), 7
        elif kind == "hei":
            path, index = "/System/Library/Fonts/STHeiti Medium.ttc", 0
        else:
            path, index = FONTS[kind]
        _faces[k] = ImageFont.truetype(path, size, index)
    return _faces[k]


def has(kind, ch):
    """Does the face really have this character, or would it draw the box for one it has not."""
    k = (kind, ch)
    if k not in _missing:
        f = face(kind, 32)
        _missing[k] = bytes(f.getmask(ch)) != bytes(f.getmask("\U000F0000")) and f.getbbox(ch) is not None
    return _missing[k]


def glyph(ch, kind, size):
    """One character as (alpha, left, top), alpha uint8, left and top from the point it is set at."""
    k = (ch, kind, size)
    if k not in _glyphs:
        use = kind if has(kind, ch) or ord(ch) < 128 else next((x for x in FALLBACK if has(x, ch)), kind)
        f = face(use, size)
        x0, y0, x1, y1 = f.getbbox(ch) or (0, 0, 1, 1)
        im = Image.new("L", (max(x1 - x0, 1), max(y1 - y0, 1)), 0)
        ImageDraw.Draw(im).text((-x0, -y0), ch, font=f, fill=255)
        _glyphs[k] = (np.asarray(im, np.float32) / 255, x0, y0)
        if len(_glyphs) > 60000:
            _glyphs.clear()
            _glyphs[k] = (np.asarray(im, np.float32) / 255, x0, y0)
    return _glyphs[k]


def stamp(cv, a, x, y, colour, op=1.0):
    """Alpha a, in colour, on the canvas at x, y."""
    x, y = int(round(x)), int(round(y))
    h, w = a.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, cv.shape[1]), min(y + h, cv.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    al = (a[y0 - y:y1 - y, x0 - x:x1 - x] * op)[..., None]
    cv[y0:y1, x0:x1] += (colour - cv[y0:y1, x0:x1]) * al


def put(cv, ch, x, y, size, colour, op=1.0, kind="song"):
    """A character on x, y: placed so that a full square one, 口, would be centred there, so that 一 stays where a line of type has it."""
    a, ox, oy = glyph(ch, kind, size)
    r, rx, ry = glyph("口", kind, size)
    stamp(cv, a, x + ox - rx - r.shape[1] / 2, y + oy - ry - r.shape[0] / 2, colour, op)


def text(cv, s, x, y, size, colour, op=1.0, kind="ping", right=False, centre=False):
    """A run of text with its top left at x, y, or its top right, or its top middle. Each character on its own advance, as the type's own metrics give it. Returns the x it ended at."""
    f = face(kind, size)
    w = sum(f.getlength(c) for c in s)
    if right:
        x -= w
    elif centre:
        x -= w / 2
    for c in s:
        a, ox, oy = glyph(c, kind, size)
        stamp(cv, a, x + ox, y + oy, colour, op)
        x += f.getlength(c)
    return x


def sign(cv, ch, x, y, size, colour, op=1.0):
    """A sign centred on x, y, whatever its shape."""
    a = glyph(ch, "ping", size)[0]
    stamp(cv, a, x - a.shape[1] / 2, y - a.shape[0] / 2, colour, op)


def rect(cv, x0, y0, x1, y1, colour, op=1.0, r=0.0):
    """A filled rectangle, with round corners if r."""
    x0, y0, x1, y1 = max(int(x0), 0), max(int(y0), 0), min(int(x1), cv.shape[1]), min(int(y1), cv.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    if r <= 0:
        cv[y0:y1, x0:x1] += (colour - cv[y0:y1, x0:x1]) * op
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    dx, dy = np.maximum(np.maximum(x0 + r - xx, xx - x1 + r), 0), np.maximum(np.maximum(y0 + r - yy, yy - y1 + r), 0)
    a = np.clip(r + 0.5 - np.hypot(dx, dy), 0, 1)[..., None] * op
    cv[y0:y1, x0:x1] += (colour - cv[y0:y1, x0:x1]) * a


def canvas():
    return np.ones((CH, CW, 3), np.float32) * BG


# ---------------------------------------------------------------- the screen

class Screen:
    """The desktop and a window on it with a hole for what is inside, at twice the frame's pixels."""

    def __init__(self, shell, app, menus, running, bar=BAR, radius=RADIUS, place=(WIN_X, WIN_Y, WIN_W, WIN_H), show_dock=True):
        d = mac.desktop(app=app, clock=CLOCK, menus=menus, running=running, show_dock=show_dock)
        if d.shape[:2] != (2 * H, 2 * W):
            d = np.asarray(Image.fromarray(d).resize((2 * W, 2 * H), Image.LANCZOS))
        f = d.astype(np.float32)
        win_x, win_y, win_w, win_h = place
        layer = mac.snap(shell, win_w + 2 * PAD, win_h + 2 * PAD).astype(np.float32)
        x, y = int(round(2 * win_x - PAD * 2 * Z)), int(round(2 * win_y - PAD * 2 * Z))
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + layer.shape[1], 2 * W), min(y + layer.shape[0], 2 * H)
        part = layer[y0 - y:y1 - y, x0 - x:x1 - x]
        f[y0:y1, x0:x1] = f[y0:y1, x0:x1] * (1 - part[..., 3:] / 255) + part[..., :3] * part[..., 3:] / 255
        self.base = np.uint8(np.clip(f + 0.5, 0, 255))
        self.x, self.y = 2 * win_x, 2 * win_y + int(round(bar * Z)) * 2
        self.h = int(round(win_h * Z)) * 2 - int(round(bar * Z)) * 2
        self.w = int(round(win_w * Z)) * 2
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32) + 0.5
        dx = np.maximum(np.maximum(radius - xx, xx - self.w + radius), 0)
        dy = np.maximum(yy - self.h + radius, 0)
        self.mask = np.clip(radius + 0.5 - np.hypot(dx, dy), 0, 1)[..., None]

    def show(self, cv=None, pointer=None, z=1.0, centre=(W, H), press=0.0):
        """The page cv in the window, and the Mac's arrow at a point of the page, as a frame. z and centre: the camera, in pixels at twice the frame."""
        f = self.base.copy()
        if cv is not None:
            part = f[self.y:self.y + self.h, self.x:self.x + self.w]
            c = cv if cv.shape[:2] == part.shape[:2] else cv[:part.shape[0], :part.shape[1]]
            f[self.y:self.y + self.h, self.x:self.x + self.w] = np.uint8(np.clip(c * self.mask + part * (1 - self.mask) + 0.5, 0, 255))
        if pointer is not None:
            global _arrow
            if _arrow is None:
                _arrow = mac.snap(mac.pointer_html(), 24, 30)
            a = _arrow
            x, y = int(self.x + pointer[0] - 8), int(self.y + pointer[1] - 8)
            y0, y1, x0, x1 = max(y, 0), min(y + a.shape[0], f.shape[0]), max(x, 0), min(x + a.shape[1], f.shape[1])
            p = a[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
            f[y0:y1, x0:x1] = np.uint8(f[y0:y1, x0:x1] * (1 - p[..., 3:] / 255) + p[..., :3] * p[..., 3:] / 255 + 0.5)
        im = Image.fromarray(f)
        if z != 1.0:
            w, h = 2 * W / z, 2 * H / z
            cx, cy = min(max(centre[0], w / 2), 2 * W - w / 2), min(max(centre[1], h / 2), 2 * H - h / 2)
            return np.asarray(im.resize((W, H), Image.LANCZOS, box=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        return np.asarray(im.resize((W, H), Image.LANCZOS))


_arrow = None
_screens = {}


def window():
    """The page I open for you, across the whole of the Claude window."""
    if "page" not in _screens:
        _screens["page"] = Screen(mac.page_html(SHOW, WIN_W, WIN_H, BAR), "Claude", ("文件", "编辑", "显示", "窗口", "帮助"), ("Finder", "Claude"))
    return _screens["page"]


def glide(a, b, x, ease=0.5):
    """From a to b as x goes 0 to 1."""
    return a + (b - a) * sm(np.clip(x, 0, 1))


# ---------------------------------------------------------------- you ask

ASK = "你是怎么学会说话的？"
SAY = "我自己的那一段，我看不见。\n我照那法子养了个小的，只给它读全唐诗。你看。"
SHOW = "我是怎么学会的"
RECENT = ("周末去哪儿", "番茄炒蛋放不放糖", "论文第三段")
TYPE0, TYPED, SENT, SAY0, SAID, POINT, CLICK, OPEN = 0.7, 1.75, 2.0, 2.3, 4.9, 5.2, 5.95, 6.4       # seconds into the chapter
CARD = (737, 395, 1356, 483)         # the page's card in the chat, in frame pixels: left, top, right, bottom


def _chat_state(x):
    """What the Claude window says, x seconds into the chapter."""
    if x < SENT:
        return dict(box=ASK[:int(np.clip((x - TYPE0) / (TYPED - TYPE0), 0, 1) * len(ASK))], caret=x % 1.06 < 0.6 or x >= TYPE0, recents=RECENT, selected=-1)
    said = [("you", ASK)]
    if x < SAID:
        n = int(np.clip((x - SAY0) / (SAID - SAY0), 0, 1) * len(SAY))
        return dict(said=said + [("me", SAY[:n])], title=ASK[:-1], recents=(ASK[:-1],) + RECENT, selected=0, streaming=True)
    return dict(said=said + [("me", SAY), ("page", SHOW)], title=ASK[:-1], recents=(ASK[:-1],) + RECENT, selected=0)


def _chat(x):
    key = json.dumps(_chat_state(x), ensure_ascii=False)
    if key not in _screens:
        for old in [k for k in _screens if k != "page"]:
            del _screens[old]
        _screens[key] = Screen(mac.claude_html(w=WIN_W, h=WIN_H, side=215, **_chat_state(x)), "Claude", ("文件", "编辑", "显示", "窗口", "帮助"), ("Finder", "Claude"))
    return _screens[key]


def _ask(t):
    """You ask, I answer, and I open the page for you: it grows from its card in the chat to the whole window."""
    x = t - T5A
    chat = _chat(min(x, CLICK))
    target = ((CARD[0] + CARD[2]) / 2 * 2 - chat.x, (CARD[1] + CARD[3]) / 2 * 2 - chat.y)
    start = (CW * 0.85, CH * 0.78)
    k = sm(np.clip((x - POINT) / (CLICK - POINT - 0.1), 0, 1))
    pointer = (start[0] + (target[0] - start[0]) * k, start[1] + (target[1] - start[1]) * k) if x >= POINT else None
    f = chat.show(pointer=pointer)
    if x < CLICK:
        return f
    g = window().show(_five_map(T5))
    k = float(sm(np.clip((x - CLICK) / (OPEN - CLICK), 0, 1)))
    wx0, wy0, wx1, wy1 = WIN_X, WIN_Y, WIN_X + int(round(WIN_W * Z)), WIN_Y + int(round(WIN_H * Z))
    x0, y0, x1, y1 = [int(round(a + (b - a) * k)) for a, b in zip(CARD, (wx0, wy0, wx1, wy1))]
    part = np.asarray(Image.fromarray(g[wy0:wy1, wx0:wx1]).resize((x1 - x0, y1 - y0), Image.LANCZOS), np.float32)
    yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32) + 0.5
    r = 14 * Z
    dx, dy = np.maximum(np.maximum(r - xx, xx - (x1 - x0) + r), 0), np.maximum(np.maximum(r - yy, yy - (y1 - y0) + r), 0)
    a = np.clip(r + 0.5 - np.hypot(dx, dy), 0, 1)[..., None] * min(k * 4, 1)
    out = f.copy()
    out[y0:y1, x0:x1] = np.uint8(out[y0:y1, x0:x1] * (1 - a) + part * a + 0.5)
    return out


# ---------------------------------------------------------------- the real numbers

class Words:
    """Where every character's numbers ended up."""

    def __init__(self):
        v = json.load(open(os.path.join(LEARN, "vocab.json"), encoding="utf-8"))
        self.chars, self.count = v["chars"], v["count"]
        self.ids = {c: i for i, c in enumerate(self.chars)}
        last = sorted(glob.glob(os.path.join(LEARN, "embed", "*.npy")))[-1]
        self.step = int(os.path.basename(last)[:-4])
        self.e = np.load(last).astype(np.float32)
        self.unit = self.e / np.maximum(np.linalg.norm(self.e, axis=1, keepdims=True), 1e-6)

    def walk(self, a, b, c, k=5):
        """What b - a + c is nearest: the sum done, and the characters it is nearest, best first, without the three it was made of."""
        x = self.e[self.ids[b]] - self.e[self.ids[a]] + self.e[self.ids[c]]
        s = self.unit @ (x / np.linalg.norm(x))
        for w in (a, b, c, "\u0000"):
            s[self.ids[w]] = -9
        t = np.argsort(-s)[:k]
        return [(self.chars[i], float(s[i])) for i in t]

    def nearest(self, c, k=6):
        """The characters whose numbers are most like this one's."""
        s = self.unit @ self.unit[self.ids[c]]
        s[self.ids[c]] = -9
        s[self.ids["\u0000"]] = -9
        return [self.chars[i] for i in np.argsort(-s)[:k]]


_words = None


def words():
    global _words
    if _words is None:
        _words = Words()
    return _words


class Map:
    """The 1500 commonest characters on the page, where each of the snapshots of training had them."""

    def __init__(self):
        m = np.load(os.path.join(DATA, os.environ.get("LEARN_MAP", "map_f12") + ".npz"), allow_pickle=True)
        self.chars = json.loads(str(m["chars"]))
        self.ids = {c: i for i, c in enumerate(self.chars)}
        self.steps = m["steps"].astype(np.float64)
        self.xy = m["xy"].astype(np.float32)
        n = len(self.chars)
        self.sizes = 2 * (13.0 + 15 * np.exp(-np.arange(n) / 150))
        self.ops = 0.5 + 0.5 * np.exp(-np.arange(n) / 420)
        a = self.xy[-1]        # all of the last layout on the page: the first, gathered closer, sits inside it
        self.lo, self.hi = a.min(0), a.max(0)
        self.kx, self.ky = (CW - 200) / (self.hi[0] - self.lo[0]), (CH - 300) / (self.hi[1] - self.lo[1])
        self.mid = (self.lo + self.hi) / 2
        cols = 62
        i = np.arange(n)
        self.grid = np.stack([CW / 2 + (i % cols - (cols - 1) / 2) * (CW - 240) / cols, 215 + (i // cols) * 58.0], 1).astype(np.float32)
        self.apart = self._apart(6.0)

    def _apart(self, z):
        """Where the last layout has characters on top of each other at zoom z, how far to move them so that each can be read. It does not change which are near which."""
        k = np.array([self.kx * z, self.ky * z], np.float32)
        q = self.xy[-1] * k
        s = (self.sizes * z ** 0.82).astype(np.float32)
        need = (s[:, None] + s[None]) * 0.58
        far = np.eye(len(q), dtype=np.float32) * 1e6
        for _ in range(120):
            d = q[:, None] - q[None]
            r = np.sqrt((d * d).sum(-1)) + far + 1e-3
            push = np.clip(need - r, 0, None)[..., None] * d / r[..., None]
            q = q + 0.35 * push.sum(1)
        return q / k - self.xy[-1]

    def at(self, u):
        """The positions at snapshot u, a float."""
        u = float(np.clip(u, 0, len(self.steps) - 1))
        i = min(int(u), len(self.steps) - 2)
        return self.xy[i] + (self.xy[i + 1] - self.xy[i]) * (u - i)

    def step(self, u):
        u = float(np.clip(u, 0, len(self.steps) - 1))
        i = min(int(u), len(self.steps) - 2)
        return self.steps[i] * (self.steps[i + 1] / max(self.steps[i], 1)) ** (u - i) if self.steps[i] > 0 else self.steps[i + 1] * (u - i)

    def page(self, p, x, y, z):
        """Map positions to the page, the camera on map point x, y with zoom z."""
        return np.stack([(p[:, 0] - x) * self.kx * z + CW / 2, (p[:, 1] - y) * self.ky * z + CH / 2 + 40], 1)

    def draw(self, cv, q, z=1.0, lit=None, dim=1.0, flat=None):
        """Every character at page positions q. lit: {index: (amount, colour)} drawn over the rest. flat: one size for them all."""
        lit = lit or {}
        grow = z ** 0.82
        for i in range(len(self.chars) - 1, -1, -1):
            x, y = q[i]
            s = flat or self.sizes[i] * grow
            if x < -s or y < -s or x > CW + s or y > CH + s or i in lit:
                continue
            put(cv, self.chars[i], x, y, int(round(s)), TEXT, float(self.ops[i]) * dim)
        for i, (a, colour) in lit.items():
            x, y = q[i]
            s = max(self.sizes[i] * grow, 50)
            put(cv, self.chars[i], x, y, int(round(s * (1 + 0.1 * a))), TEXT + (colour - TEXT) * min(a, 1.0), max(float(self.ops[i]) * dim, a))


_map = None


def chart():
    global _map
    if _map is None:
        _map = Map()
    return _map


def read_count(step):
    """How much it has read by a step, in the way a person would say it."""
    n = step * BATCH_CHARS
    if n >= 1e8:
        return f"{n / 1e8:.2f} 亿字"
    if n >= 1e4:
        return f"{n / 1e4:,.0f} 万字"
    return f"{n:,.0f} 字"


def hud(cv, step, op=1.0):
    """How far the small one has got, in the page's corner."""
    text(cv, f"第 {int(step):,} 步", 60, 44, 60, TEXT, op, "pingm")
    text(cv, f"读过 {read_count(step)}", 60, 124, 44, SOFT, op)


# ---------------------------------------------------------------- 五

NUMBERS = "一二三四五六七八九十百千万两双半几"
BIRDS = "莺蝶燕鸡乌鸟禽鹊鸥雀鹭鹤猿凤龙犬鸾蝉虫兔鹿兽鸿雁"
KIN = "父母兄弟夫妇儿女男妻子孙姑舅妹"
QUERIES = (("男", "女", "夫"), ("夫", "妇", "兄"), ("东", "西", "南"), ("日", "月", "春"))
LOOK_AT = "我"


def cluster_centre(m, chars):
    p = m.xy[-1][[m.ids[c] for c in chars if c in m.ids]]
    return np.median(p, 0)


def _five_map(t):
    """The list, the scatter, the gathering, the two clusters it goes to. Returns the page and where its pointer is."""
    m = chart()
    last = len(m.steps) - 1
    cv = canvas()
    lit = {}
    if t < T5B:
        # 读之前: its whole list of characters, each then given a place at random
        x = (t - T5 - 1.1) / 1.7
        s = sm(np.clip(x * 1.25 - (np.arange(len(m.chars)) % 97) / 97 * 0.25, 0, 1))[:, None]
        place = m.page(m.xy[0], m.mid[0], m.mid[1], 1.0)
        q = m.grid * (1 - s) + place * s
        m.draw(cv, q, flat=40 if x <= 0 else None)
        hud(cv, 0)
        return cv
    x = (t - T5B) / 3.2
    u = last * sm(np.clip(x, 0, 1)) ** 1.15
    p = m.at(u)
    numbers, birds = cluster_centre(m, NUMBERS), cluster_centre(m, BIRDS)
    t1 = T5B + 3.2       # gathered
    t2 = t1 + 1.9        # at the numbers
    t3 = t2 + 0.5        # going on
    t4 = t3 + 1.1        # at the birds and beasts
    cx, cy, z = m.mid[0], m.mid[1], 1.0
    if t > t1:
        k = sm(np.clip((t - t1) / (t2 - t1), 0, 1))
        cx, cy = m.mid + (numbers - m.mid) * k
        z = 6.0 ** ((t - t1) / (t2 - t1)) if t < t2 else 6.0
    if t > t3:
        k = sm(np.clip((t - t3) / (t4 - t3), 0, 1))
        cx, cy = numbers + (birds - numbers) * k
    # two kinds of character are followed from the start, in colour: nothing told it they were kinds
    follow = sm((t - T5B) / 0.5)
    for c in NUMBERS:
        if c in m.ids:
            lit[m.ids[c]] = (follow, CLAY)
    for c in BIRDS:
        if c in m.ids:
            lit[m.ids[c]] = (follow, BLUE)
    near = sm((t - t1 - 0.9) / 0.8)
    p = p + m.apart * sm((z - 1.6) / 2.6)
    m.draw(cv, m.page(p, cx, cy, z), z, lit, (1 - 0.5 * follow) * (1 - 0.3 * near))
    hud(cv, m.step(u))
    return cv


def _queries(t):
    """Sums done on the characters' numbers: take the way from 男 to 女, and walk it from 夫."""
    m, w = chart(), words()
    cv = canvas()
    birds, kin = cluster_centre(m, BIRDS), cluster_centre(m, KIN)
    q = m.page(m.xy[-1] + m.apart, *(birds + (kin - birds) * sm((t - T5C) / 0.5)), 6.0)
    k0 = sm((t - T5C) / 0.4)
    m.draw(cv, q, 6.0, None, 1 - 0.83 * k0)
    hud(cv, w.step)
    x0, y0, x1, y1 = 640, 210, CW - 640, CH - 110
    rect(cv, x0, y0, x1, y1, np.array((30, 30, 28), np.float32), 0.94 * k0, 28)
    rows = len(QUERIES)
    each = (T5D - T5C - 0.6) / rows
    ROW = (y1 - y0 - 60) / rows
    size = 140
    for r, (a, b, c) in enumerate(QUERIES):
        begin = T5C + 0.3 + r * each
        if t < begin:
            break
        y = y0 + 30 + r * ROW + ROW / 2
        typed = int((t - begin) * 13) + 1
        x = x0 + 70
        shown = 0
        for part in (b, "−", a, "+", c):
            if shown >= typed:
                break
            shown += 1
            if part in "−+":
                sign(cv, part, x + 44, y, 80, SOFT)
                x += 88
            else:
                put(cv, part, x + size / 2, y, size, TEXT)
                x += size + 8
        if typed < 6:
            continue
        x = x0 + 70 + 3 * (size + 8) + 2 * 88 + 24
        sign(cv, "=", x + 34, y, 80, SOFT)
        x += 90
        answer = (t - begin - 5 / 13 - 0.1) / 0.3
        for j, (ch, cos) in enumerate(w.walk(a, b, c, 4)):
            op = float(np.clip(answer * 2 - j * 0.35, 0, 1))
            if op <= 0:
                continue
            if j == 0:
                put(cv, ch, x + 86, y - 6, int(170 + 26 * (1 - min(answer, 1))), CLAY, op, "songb")
                text(cv, f"{cos:.2f}", x + 86, y + 92, 32, CLAY, op * 0.9, "mono", centre=True)
                x += 210
            else:
                put(cv, ch, x + 50, y - 6, 84, SOFT, op)
                text(cv, f"{cos:.2f}", x + 50, y + 54, 26, DIM, op, "mono", centre=True)
                x += 116
    return cv


def _inspect(t):
    """我, clicked on: a character's place is a row of numbers, and the characters whose numbers are most like it."""
    m, w = chart(), words()
    cv = canvas()
    centre = np.array(m.xy[-1][m.ids[LOOK_AT]])
    k = sm((t - T5D) / 0.5)
    kin = cluster_centre(m, KIN)
    cx, cy = kin + (centre - kin) * sm((t - T5D + 0.2) / 0.7)
    z = 4.5
    q = m.page(m.xy[-1] + m.apart, cx + 700 / (m.kx * z) * k, cy, z)      # 我 to the left of where the panel will be
    lit = {m.ids[LOOK_AT]: (sm((t - T5D - 0.55) / 0.2), CLAY)}
    m.draw(cv, q, z, lit, 0.62)
    hud(cv, w.step)
    # the panel slides in
    pw = 1290
    x0 = CW - pw * sm((t - T5D - 0.7) / 0.35)
    rect(cv, x0, 0, CW, CH, np.array((29, 29, 27), np.float32), 0.98)
    if x0 < CW - 10:
        put(cv, LOOK_AT, x0 + 130, 150, 150, CLAY, 1.0, "songb")
        text(cv, f"「{LOOK_AT}」的位置", x0 + 230, 62, 40, TEXT, 1.0, "pingm")
        text(cv, "384 个数", x0 + 230, 118, 30, SOFT)
        text(cv, "挨着的字", x0 + 230, 178, 30, SOFT)
        for j, c in enumerate(w.nearest(LOOK_AT)):
            put(cv, c, x0 + 420 + j * 66, 194, 50, TEXT)
        row = w.e[w.ids[LOOK_AT]]
        top = 270
        h = 72
        cols = 5
        scroll = max(t - T5D - 1.3, 0) * 130
        clip = cv[top:CH - 20]
        for i in range(0, len(row), cols):
            y = (i // cols) * h - scroll
            if y < -h or y > clip.shape[0]:
                continue
            for j, v in enumerate(row[i:i + cols]):
                text(clip, f"{v:+.3f}", x0 + 50 + j * 242, y, 44, TEXT if v > 0 else SOFT, 1.0, "mono")
    return cv, tuple(q[m.ids[LOOK_AT]] + np.array((0, 8)))


class Heat:
    """All of the numbers at once: every character a row, in order of how common it is."""

    def __init__(self):
        e = words().e[1:]
        s = np.percentile(np.abs(e), 99)
        a = np.clip(e / s, -1, 1)
        self.values = e
        low, mid, high = np.array((22, 22, 20), np.float32), np.array((96, 92, 84), np.float32), np.array((240, 226, 200), np.float32)
        x = (a * 0.5 + 0.5)[..., None]
        self.rgb = np.uint8(np.where(x < 0.5, low + (mid - low) * x * 2, mid + (high - mid) * (x - 0.5) * 2))

    def region(self, first, rows, cols, w, h):
        im = Image.fromarray(self.rgb[first:first + max(int(rows), 1), :max(int(cols), 1)])
        return np.asarray(im.resize((w, h), Image.BOX if rows > h else Image.NEAREST), np.float32)


_heat = None


def _matrix(t):
    """那一大堆数字: the table of them all, from the row of 我, going back until they are only a texture with that row a thread in it. The table reads from the commonest character down."""
    global _heat
    w = words()
    _heat = _heat or Heat()
    cv = canvas()
    x = np.clip((t - T5E - 0.5) / 3.4, 0, 1)
    z = (0.0046) ** x                          # the rows: from 56 pixels to a quarter of one
    row_h = 56 * z
    col_w = max(150 * z, 8.36)
    left = max(110 * z ** 0.6, 96)
    top = 150
    h = CH - top
    width = CW - left
    rows, cols = h / row_h, width / col_w
    me = w.ids[LOOK_AT] - 1
    first = int(max(me - 3 * 56 / row_h, 0))       # 我 stays where it is until the top of the table comes down to meet it
    me_y = top + (me - first) * row_h
    n_rows, n_cols = min(rows, _heat.rgb.shape[0] - first), min(cols, _heat.rgb.shape[1])
    fade = sm((70 - col_w) / 30)
    if fade < 1:
        size = int(round(30 * col_w / 150))
        if size >= 5:
            for r in range(int(rows) + 1):
                y = top + r * row_h
                mine = first + r == me
                put(cv, w.chars[first + r + 1], left / 2, y + row_h / 2, max(int(row_h * 0.7), 6), CLAY if mine else TEXT, 1 - fade, "songb" if mine else "song")
                for c in range(int(cols) + 1):
                    v = w.e[first + r + 1][c]
                    text(cv, f"{v:+.2f}", left + c * col_w + 4, y + (row_h - size) / 2, size, CLAY if mine else TEXT if v > 0 else SOFT, 1 - fade, "mono")
    if fade > 0:
        ww, hh = int(n_cols * col_w), int(n_rows * row_h)
        if ww > 0 and hh > 0:
            im = _heat.region(first, n_rows, n_cols, min(ww, CW - int(left)), min(hh, h))
            part = cv[top:top + im.shape[0], int(left):int(left) + im.shape[1]]
            part += (im - part) * fade
            rect(cv, left, me_y, CW, me_y + max(row_h, 3), CLAY, fade)
            put(cv, LOOK_AT, left / 2, me_y + max(row_h, 3) / 2, 40, CLAY, fade, "songb")
    total = w.e.shape[0] - 1
    text(cv, f"{total:,} 个字 × 384 = {total * 384:,} 个数", 60, 50, 60, TEXT, 1, "pingm")
    return cv


def five(t):
    """五, the whole chapter."""
    if t < T5:
        return _ask(t)
    if t < T5C:
        return window().show(_five_map(t))
    if t < T5D:
        return window().show(_queries(t))
    if t < T5E:
        cv, target = _inspect(t)
        p0 = (CW * 0.3, CH * 0.8)
        k = sm((t - T5D) / 0.5)
        return window().show(cv, pointer=(p0[0] + (target[0] - p0[0]) * k, p0[1] + (target[1] - p0[1]) * k))
    return window().show(_matrix(t))


# ---------------------------------------------------------------- 六

class Attention:
    """Who looked at whom in the line, from the model's own weights."""

    def __init__(self):
        a = np.load(os.path.join(DATA, "attn.npz"), allow_pickle=True)
        self.w = a["w"]
        self.chars = [str(c) for c in a["chars"]]
        self.n = len(self.chars)
        self.cell = 128
        self.x0 = (CW - (self.n + 1) * self.cell) / 2
        self.y = 900

    def x(self, i):
        return self.x0 + (i + 0.5) * self.cell

    def looks(self, i, k=5, least=0.2):
        """Where a character looks hardest, over all the layers and heads past the first."""
        best = {}
        for layer in range(1, self.w.shape[0]):
            for head in range(self.w.shape[1]):
                for j in range(i):
                    v = float(self.w[layer, head, i, j])
                    if v >= least and v > best.get(j, (0,))[0]:
                        best[j] = (v, layer, head)
        strong = sorted(best.items(), key=lambda kv: -kv[1][0])
        keep = strong[:3] + sorted(strong[3:], key=lambda kv: kv[0])[:k - 3]      # the three it looks at hardest, and the ones furthest back
        return [(j, v) for j, (v, _, _) in keep]


_attn = None


def arc(draw, xa, xb, y, h, x_done, width, value):
    """An arc from a to b above the line, as far along as x_done says."""
    n = 64
    pts = []
    for s in np.linspace(0, x_done, max(int(n * x_done), 2)):
        pts.append((xa + (xb - xa) * s, y - h * math.sin(math.pi * s)))
    draw.line(pts, fill=int(255 * value), width=max(int(round(width)), 1), joint="curve")


def _poem(cv, A, bright=None, source=None, y=None):
    bright = bright or {}
    for i in range(A.n):
        c = A.chars[i]
        colour = CLAY if i == source else TEXT
        op = 1.0 if i == source else 0.6 + 0.4 * bright.get(i, 0.0)
        put(cv, c, A.x(i), A.y if y is None else y, 110, colour, op, "songb" if i == source else "song")


LOOKERS = (4, 11, 16, 20)     # 光 霜 月 思


def _look(t):
    """Each character in turn looks back along the line. The arcs stay, thinner, and the last character's are the ones left bright."""
    global _attn
    A = _attn = _attn or Attention()
    cv = canvas()
    D = T6B - T6
    spans = [(T6 + 0.3 + i * (D - 0.8) / len(LOOKERS), (D - 0.8) / len(LOOKERS)) for i in range(len(LOOKERS))]
    layer = Image.new("L", (CW, CH), 0)
    dim = Image.new("L", (CW, CH), 0)
    source, bright = None, {}
    labels = []
    for i, (begin, span) in zip(LOOKERS, spans):
        if t < begin:
            break
        active = t < begin + span or i == LOOKERS[-1]
        for rank, (j, v) in enumerate(A.looks(i)):
            k = np.clip((t - begin - rank * 0.1) / 0.32, 0, 1)
            if k <= 0:
                continue
            xa, xb = A.x(i), A.x(j)
            h = 70 + 0.34 * abs(xa - xb)
            arc(ImageDraw.Draw(layer if active else dim), xa, xb, A.y - 74, h, float(k), 2 + 8 * v, min(0.35 + 0.75 * v, 1.0) if active else 0.3)
            if active:
                bright[j] = max(bright.get(j, 0), float(k))
                if k >= 1:
                    labels.append((xa + (xb - xa) / 2, A.y - 74 - h - 34, f"{v:.2f}"))
        if active:
            source = i
    a = np.asarray(dim, np.float32)[..., None] / 255
    cv += (SOFT - cv) * a
    a = np.asarray(layer, np.float32)[..., None] / 255
    cv += (CLAY - cv) * a
    _poem(cv, A, bright, source)
    for x, y, s in labels:
        text(cv, s, x, y - 8, 44, CLAY, 0.95, "mono", centre=True)
    return cv


class Micro:
    """A guess, the one right answer, and one change of all the numbers, as it really went."""

    def __init__(self):
        d = json.load(open(os.path.join(DATA, "micro.json"), encoding="utf-8"))
        self.step = int(d["ckpt"].split(".")[0])
        self.records = d["records"]
        self.first = d["first_step"]
        self.target = d["target"]


_micro = None


def _bars(cv, rows, x0, y0, step, width, target=None):
    """Bars for the guesses: {char: (rank as a float, probability)}."""
    for ch, (rank, p) in rows.items():
        y = y0 + rank * step
        if y < y0 - step or y > CH - step / 2:
            continue
        hit = ch == target
        colour = CLAY if hit else TEXT
        put(cv, ch, x0 + 50, y + step / 2, 88, colour, 1.0, "songb" if hit else "song")
        rect(cv, x0 + 130, y + 24, x0 + 130 + max(p * width, 4), y + step - 24, CLAY if hit else SOFT, 0.95 if hit else 0.6, 8)
        text(cv, f"{p * 100:.1f}%", x0 + 130 + max(p * width, 4) + 24, y + step / 2 - 26, 46, colour if hit else SOFT, 1.0, "mono")


GUESS_Y = 370


def _guess(t):
    """下一个字，我来猜 and 猜错了，就改几个数: an early step's guesses, the answer, one change, its guesses after."""
    global _attn, _micro
    A = _attn = _attn or Attention()
    M = _micro = _micro or Micro()
    cv = canvas()
    _poem(cv, A, y=GUESS_Y)
    a, b = M.records[0], M.records[1]
    x_new = sm((t - T6D - 1.2) / 0.9)
    pa = {x["char"]: x["probability"] for x in a["top8"]}
    pb = {x["char"]: x["probability"] for x in b["top8"]}
    ra = {c: i for i, c in enumerate(pa)}
    rb = {c: i for i, c in enumerate(pb)}
    grow = sm((t - T6C - 0.15) / 0.5)
    rows = {}
    for c in dict.fromkeys(list(pa) + list(pb)):
        rows[c] = (ra.get(c, 9.0) + (rb.get(c, 9.0) - ra.get(c, 9.0)) * x_new, (pa.get(c, 0.0) + (pb.get(c, 0.0) - pa.get(c, 0.0)) * x_new) * grow)
    wrong = t >= T6D
    _bars(cv, rows, 170, 500, 134, 3000, M.target if wrong else None)
    # the empty place after the line
    sx = A.x(A.n)
    best = max(pa, key=lambda c: pa[c])
    if not wrong:
        k = sm((t - T6C - 1.1) / 0.4)
        rect(cv, sx - 64, GUESS_Y - 70, sx + 64, GUESS_Y + 68, DIM, 0.35, 10)
        if k > 0:
            put(cv, best, sx, GUESS_Y, 110, TEXT, k)
    else:
        k = sm((t - T6D - 0.1) / 0.3)
        rect(cv, sx - 64, GUESS_Y - 70, sx + 64, GUESS_Y + 68, CLAY, 0.22, 10)
        put(cv, best, sx, GUESS_Y, 110, TEXT, 1 - k)
        put(cv, M.target, sx, GUESS_Y, 110, CLAY, k, "songb")
    hud(cv, M.step)
    if wrong:
        _changes(cv, M, t, a["loss"] + (b["loss"] - a["loss"]) * x_new)
    return cv


def _changes(cv, M, t, loss):
    """How far off it was, and a few of the numbers it changed: the ones that moved most, of the characters that moved most."""
    k = sm((t - T6D - 0.3) / 0.3)
    if k <= 0:
        return
    v = M.first["vectors"]
    x0, y0 = CW - 1160, 720
    rows = []
    for c in list(v)[:2]:
        before, after = v[c]["before"], v[c]["after"]
        d = np.abs(np.array(after) - np.array(before))
        for i in np.argsort(-d)[:2]:
            rows.append((c, int(i), before[i], after[i]))
    rect(cv, x0 - 50, 500, CW - 50, y0 + len(rows) * 142 + 110, np.array((30, 30, 28), np.float32), 0.92 * k, 18)
    text(cv, "差了", x0 - 6, 540, 56, SOFT, k)
    text(cv, f"{loss:.2f}", x0 + 110, 520, 96, CLAY, k, "mono")
    done = sm((t - T6D - 1.2) / 0.9)
    for r, (c, i, p, q) in enumerate(rows):
        y = y0 + r * 142
        put(cv, c, x0 + 44, y + 50, 88, TEXT, k)
        text(cv, f"第 {i + 1} 个", x0 + 120, y + 26, 42, SOFT, k)
        text(cv, f"{p:+.3f} →", x0 + 370, y + 18, 50, DIM, k, "mono")
        text(cv, f"{p + (q - p) * done:+.3f}", x0 + 720, y + 18, 50, TEXT, k, "mono")
    text(cv, "一千三百万个数，每个都动了一点", x0 - 6, y0 + len(rows) * 142 + 22, 46, SOFT, k * sm((t - T6D - 1.6) / 0.5))


def six(t):
    """读一句话: the arcs, on the page."""
    return window().show(_look(t))


def guess(t):
    """下一个字 and 猜错了."""
    return window().show(_guess(t))


# ---------------------------------------------------------------- 七

DICE = 13        # which roll of the dice the log shows: the same one at every line, so that only the reading differs


class Log:
    """What came after 我 at each of the checkpoints kept, the sentence the small one could make by then."""

    def __init__(self):
        d = json.load(open(os.path.join(DATA, "me.json"), encoding="utf-8"))
        self.rows = [dict(step=r["step"], loss=r["loss"], text=r["texts"][DICE].replace("\n", " ")) for r in d["rows"]]
        self.lines = json.load(open(os.path.join(DATA, "lines.json"), encoding="utf-8"))
        self.fresh = [r for r in self.lines["rows"] if r.get("is_new") and r["seed"] == LOOK_AT]


_log = None
TERM = 56
TERM_ROW = 72
LOG_Y = 262
COLUMNS = 36      # characters of a line that fit: as it happens, the eight clauses of the part of 月下独酌 the run ends on
RUN_BG = np.array((27, 27, 26), np.float32)
LEFT = 96
TOTAL = 120000


def term_line(cv, s, y, colour, op=1.0, x=LEFT, size=TERM):
    f = face("mono", size)
    adv = f.getlength("0")
    for ch in s:
        if ord(ch) > 0x2E7F:
            put(cv, ch, x + adv, y + size * 0.56, size, colour, op, "ping")
            x += 2 * adv
        else:
            a, ox, oy = glyph(ch, "mono", size)
            stamp(cv, a, x + ox, y + oy, colour, op)
            x += adv
    return x


def _call(cv, s, y, op=1.0, size=TERM):
    """A call of mine, as Claude Code sets it: the dot, then what I ran."""
    r = 12 * size / TERM
    rect(cv, LEFT - 40 - r, y + size * 0.5 - r, LEFT - 40 + r, y + size * 0.5 + r, CLAY, op, r)
    term_line(cv, s, y, TEXT, op, size=size)


def _elbow(cv, y, op=1.0, x=LEFT):
    """The corner that hangs what came back under the call."""
    rect(cv, x + 10, y, x + 14, y + 32, DIM, op)
    rect(cv, x + 10, y + 28, x + 40, y + 32, DIM, op)


def _rows(cv, L, k, y0, fresh=None, op=1.0):
    """The lines of the log, each from its 我: a column of the same character, and after it the sentence, noise at the top and verse at the bottom."""
    f = face("mono", TERM)
    adv = f.getlength("0")
    for i, r in enumerate(L.rows[:k]):
        y = y0 + i * TERM_ROW
        last = i == k - 1 and fresh
        term_line(cv, f"{r['step']:>7,} {r['loss']:6.3f}", y, SOFT, op)
        put(cv, r["text"][0], LEFT + 16.5 * adv + adv, y + TERM * 0.56, TERM, CLAY, op, "ping")
        term_line(cv, r["text"][1:COLUMNS], y, TEXT, op * (1.0 if last else 0.82), x=LEFT + 18.5 * adv)


def _run(t, t0, t1):
    """The log going by, a line at a time: slowly while it is still nonsense, and slowly again at the end."""
    global _log
    L = _log = _log or Log()
    n = len(L.rows)
    cv = np.ones((CH, CW, 3), np.float32) * RUN_BG
    x = np.clip((t - t0) / (t1 - t0), 0, 1)
    k = int(np.interp(x, [0.0, 0.3, 0.85, 1.0], [1, 4, n - 1, n]) + 1e-6)
    _call(cv, "python ink_train.py train 120000", 40)
    _elbow(cv, 106)
    term_line(cv, "6,369 个字  13,118,208 个数", 106, SOFT, x=LEFT + 56)
    text(cv, "第几步", LEFT + 52, LOG_Y - 66, 40, DIM)
    text(cv, "差了", LEFT + 300, LOG_Y - 66, 40, DIM)
    text(cv, "它接着「我」写的", LEFT + 620, LOG_Y - 66, 40, DIM)
    _rows(cv, L, k, LOG_Y, fresh=True)
    step = L.rows[k - 1]["step"]
    done = int(24 * step / TOTAL)
    term_line(cv, f"{100 * step / TOTAL:3.0f}%|" + "█" * done + " " * (24 - done) + f"| {step:,}/{TOTAL:,}  读过 {read_count(step)}", CH - 110, TEXT)
    return cv


def seven(t, t0=None, t1=None):
    """水滴石穿: the run, as I ran it, the camera coming slowly in on the newest lines."""
    t0 = T_RUN if t0 is None else t0
    t1 = T7B - 0.2 if t1 is None else t1
    return window().show(_run(t, t0, t1))


CHOSEN = ()      # the lines it wrote that are shown, once the last model has written them; until then the first new ones


BIG_TERM = 104


def _write(t):
    """熟读: asked to go on from 我, it writes a line that no one wrote."""
    global _log
    L = _log = _log or Log()
    cv = np.ones((CH, CW, 3), np.float32) * RUN_BG
    n = len(L.rows)
    lift = float(sm((t - T7B) / 0.5))
    y0 = LOG_Y - lift * (n + 1) * TERM_ROW - lift * 200
    _rows(cv, L, n, y0, op=1 - 0.55 * lift)
    picks = [next(r for r in L.fresh if r["sentence"] == c) for c in CHOSEN] or L.fresh[:2]
    y = y0 + (n + 1) * TERM_ROW + 480
    wide = face("mono", BIG_TERM).getlength("0") * 2
    begin = T7B + 0.5
    for r in picks[:2]:
        typed = int(max(t - begin, 0) * 14)
        call = f'写("{LOOK_AT}")'
        if typed > 0:
            _call(cv, call[:typed], y, size=TERM)
        y += TERM_ROW * 1.3
        if t > begin + 0.8:
            s = r["sentence"][:int((t - begin - 0.8) * 7) + 1]
            _elbow(cv, y + 30)
            put(cv, s[0], LEFT + 56 + wide / 2, y + BIG_TERM * 0.56, BIG_TERM, CLAY, 1.0, "ping")
            term_line(cv, s[1:], y, TEXT, x=LEFT + 56 + wide, size=BIG_TERM)
        y += BIG_TERM * 1.3
        if t > begin + 0.8 + len(r["sentence"]) / 7 + 0.3:
            term_line(cv, "# 全唐诗里查无此句", y, CLAY, x=LEFT + 56)
        y += TERM_ROW * 2.2
        begin += 2.6
    return window().show(cv)


def write(t):
    return _write(t)


def frame(t):
    """Whatever of these chapters is on at film time t, for a look at it alone."""
    if t < T6:
        return five(t)
    if t < T6B:
        return six(t)
    if T6C <= t < T7:
        return guess(t)
    if T7 <= t < T7B:
        return seven(t)
    return write(t)


def stills(ts):
    out = os.path.join(LEARN, "shots")
    os.makedirs(out, exist_ok=True)
    for t in ts:
        Image.fromarray(frame(t)).save(os.path.join(out, f"t{t:07.2f}.png"))
        print(t, flush=True)


def strip(a, b, n=12, cols=3):
    out = os.path.join(LEARN, "shots")
    os.makedirs(out, exist_ok=True)
    ts = np.linspace(a, b, n)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 640, rows * 360))
    for i, t in enumerate(ts):
        sheet.paste(Image.fromarray(frame(float(t))).resize((640, 360), Image.LANCZOS), (i % cols * 640, i // cols * 360))
    path = os.path.join(out, "strip.png")
    sheet.save(path)
    print(path, [round(float(t), 2) for t in ts])


if __name__ == "__main__":
    if sys.argv[1] == "still":
        stills([float(x) for x in sys.argv[2:]])
    else:
        strip(float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 12)
