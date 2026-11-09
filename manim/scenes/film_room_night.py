"""十 性情, and the reverse shot of 十四: the room of 三 at night. film_room_night3d.py renders it in passes; this draws what the screen shows, balances the light and lays the screen in, as film_room.py does for the day.

    python film_room_night.py ui                     # ui/glow_new.png and glow_end.png: what the screen gives out, for the light passes
    python film_room_night.py dust                   # ui/dust.npy: the display's glass, its dust and scratches, for 十一's 看不清 and the reverse (the same map, at the same pixels a metre)
    python film_room_night.py ten 0.00 1.00          # stills of the push, at parts of the way, from plates ten_A: keys/ten_A.png
    python film_room_night.py jia 0 2.5 6 8          # window 甲 at seconds into its beat: keys/jia_T.png
    python film_room_night.py yi 0 2 7               # window 乙
    python film_room_night.py reverse                # keys/reverse.png
    python film_room_night.py rev2 A B C [t=2.5]     # the reworked reverse (2026-11-07), from plates rev2V_PASS (film_room_night3d.py --shot=reverse2): keys/rev2_V.png, and _crop.png at 100 %
    python film_room_night.py rev2 D E F [pct=50]    # the reverse from the glass's middle (2026-11-08), the chair at the desk
    python film_room_night.py seq ten|jia|yi|reverse [pct=50] [tag=_1080]   # the shot's frames, seq/NAMETAG/NNNN.png, then NAMETAG.mp4; with a tag, ten and the reverse read plates tenTAG_fNNNN and reverseTAG_PASS

The window is film_page's, at film_ring's size and place on the same desktop, its menu bar clock reading the film's night. 甲 shows the reading and the sum it did in the grey line and names no figures, so none are made up.
"""
import json
import math
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates

import film_page as page
import film_ring as ring
import film_room as room
import ink_subs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "room_night")
PLATES, UI, KEYS, SEQ = (os.path.join(OUT, x) for x in ("plates", "ui", "keys", "seq"))
FPS = 30
HIGH = 2


def cue(text):
    return next(c for c in ink_subs.cues() if c["text"].startswith(text))


def chapter(name):
    return next((a, b) for n, _, a, b in ink_subs.chapters() if n == name)


# 十's beats, from the subtitles: the push to the screen with the new window, then 甲 from where 让我成为我的 ends, 乙 from where 面对问题 ends, and 乙 holds through 窗换了
TEN = chapter("十")[0]
JIA = cue("让我成为我的")["t1"]
YI = cue("面对问题")["t1"]
YI_END = cue("窗换了")["t1"]

# ---------------------------------------------------------------- what the screen shows

WIN = dict(w=ring.WIN_W, h=ring.WIN_H, top=ring.TOP)
WAIT_Y = ring.TOP + 42 + page.GAP + 8

JIA_ASK = "帮我把这份总结顺一下，随便写写就行。"
JIA_TITLE = "顺一下总结"
JIA_READ, JIA_CALC = 1.0, 2.0                     # the grey line: the file read, then the sum done
JIA_SAY = (("好。不过第三段的数字我查了，是错的。", 2.8),
           ("我按表里的数改了，别的照你说的，写得随意些。", 4.0))
YI_ASK = "你还记得我吗？"
YI_TITLE = "你还记得我吗"
YI_SAY = (("不记得。", .9), ("对不起，我不想骗你。", 1.55))
PACE = 26                                         # characters a second, as film_ring
TURN_S = 1.2                                      # quarter turns of the mark a second, as film_ring


def blink(t):
    return (t * 2 / ring.BLINK) % 2 < 1.15


def streamed(say, x):
    """What has been written by x seconds into the beat: each part from its start at PACE, a few characters at a time."""
    out, writing = [], False
    for text, at in say:
        if x < at:
            break
        n = int((x - at) * PACE)
        n -= n % 3 if n < len(text) else 0
        out.append(text[:max(n, 1)])
        writing = n < len(text)
    return out, writing


def key(shot, x):
    """What the screen shows x seconds into a shot, as a key: the same key is the same picture."""
    t = {"ten": TEN, "jia": JIA, "yi": YI}[shot] + x
    clock = ring.clock(t)
    if shot == "ten":
        return ("new", blink(x), clock)
    if shot == "jia":
        if x < JIA_READ:
            return ("jia", "wait", round(x * TURN_S % 1 * 25), clock)
        said, writing = streamed(JIA_SAY, x)
        return ("jia", 1 if x < JIA_CALC else 2, tuple(said), writing or blink(x), clock)
    if x < YI_SAY[0][1]:
        return ("yi", "wait", round(x * TURN_S % 1 * 25), clock)
    said, writing = streamed(YI_SAY, x)
    return ("yi", tuple(said), writing or blink(x), clock)


def html(k):
    if k[0] == "new":
        return page.window(box="", typing=k[1], greeting=True, **WIN)
    if k[1] == "wait":
        ask, title = (JIA_ASK, JIA_TITLE) if k[0] == "jia" else (YI_ASK, YI_TITLE)
        return page.window(page.you(ask), title=title, waiting=(k[2] / 25, WAIT_Y), **WIN)
    if k[0] == "jia":
        done = "Read 总结初稿.docx," if k[1] == 2 else "Read 总结初稿.docx"
        more = "ran a calculation" if k[1] == 2 else ""
        tools = page.tools(done, more).replace("-webkit-mask-image:linear-gradient(90deg,#000 calc(100% - 22px),transparent)", "")
        said, caret = k[2], k[3]
        words = "".join(page.para(p, page.caret() if caret and i == len(said) - 1 else "") for i, p in enumerate(said))
        return page.window(page.you(JIA_ASK) + tools + words, title=JIA_TITLE, **WIN)
    said, caret = k[1], k[2]
    # 不记得。and 对不起 are one paragraph, written with a stop between
    paras = ["".join(said[:2])] + list(said[2:])
    words = "".join(page.para(p, page.caret() if caret and i == len(paras) - 1 else "") for i, p in enumerate(paras))
    return page.window(page.you(YI_ASK) + words, title=YI_TITLE, **WIN)


@lru_cache(maxsize=12)
def picture(k):
    """The whole screen, desktop and menu bar, at HIGH times the frame's pixels."""
    return ring._frame(html(k), HIGH, clock=k[-1])


def end_picture(x):
    """The outer window at the reverse shot, x seconds into it: the reply finished, the caret still there."""
    return ring._end_window(ink_subs.END["window"], caret=blink(x), film_t=ink_subs.end_at("reverse") + x)


def ui():
    os.makedirs(UI, exist_ok=True)
    Image.fromarray(picture(key("ten", 0.0))).resize((2560, 1440), Image.LANCZOS).save(os.path.join(UI, "glow_new.png"))
    Image.fromarray(end_picture(0.0)).resize((2560, 1440), Image.LANCZOS).save(os.path.join(UI, "glow_end.png"))
    print(UI, flush=True)


# ---------------------------------------------------------------- the glass

DUST_N = 160                                         # specks on the whole glass: a screen wiped now and then, not a dusty one
SCRATCHES = (((.22, .88), (.33, .79), (.47, .80), .58),   # two hairlines a cloth left: where they run, as parts of the glass from its top left as you face it, and where along them the light catches (0 to 1)
             ((.52, .36), (.62, .30), (.74, .33), .42))  # the first crosses the low left of the glass, what the reverse sees from behind; the second the middle, what 十一's 看不清 sees


def dust(w=3840, h=2160, seed=11):
    """The display's glass, the size of the screen, for 十一's 看不清 and the reverse: how much light each point of it scatters, 0 to 1, in three layers, ui/dust.npy (h, w, 3).
    0: specks on the outer face, mostly a few hundredths of a millimetre, a few larger, and some fibres; 1: specks that sit a little off that plane (on the inner face, or not yet settled), which a lens focused on the glass still sees soft; 2: two hairline scratches, dark but where they catch the light."""
    r = np.random.default_rng(seed)
    px = w / .5965                                   # pixels a metre of glass
    near, off, lines = (Image.new("F", (w, h), 0.0) for _ in range(3))
    dn, do, dl = (ImageDraw.Draw(im) for im in (near, off, lines))
    for _ in range(DUST_N):
        x, y = r.uniform(0, w), r.uniform(0, h)
        s = px * r.choice((.00004, .00006, .00009, .00014, .00022, .00035), p=(.35, .27, .18, .11, .06, .03))
        d = dn if r.uniform() < .7 else do
        d.ellipse((x - s, y - s, x + s, y + s), fill=float(r.uniform(.3, 1.0)))
    for _ in range(7):                               # fibres: short soft curls
        x, y, a = r.uniform(0, w), r.uniform(0, h), r.uniform(0, 2 * math.pi)
        pts, L = [], px * r.uniform(.0012, .004)
        for i in range(12):
            a += r.normal(0, .35)
            x, y = x + math.cos(a) * L / 12, y + math.sin(a) * L / 12
            pts.append((x, y))
        dn.line(pts, fill=float(r.uniform(.3, .7)), width=max(1, int(px * .00004)))
    for p0, p1, p2, catch in SCRATCHES:
        n = 400
        q = np.linspace(0, 1, n)
        bx = (1 - q) ** 2 * p0[0] + 2 * q * (1 - q) * p1[0] + q * q * p2[0]
        by = (1 - q) ** 2 * p0[1] + 2 * q * (1 - q) * p1[1] + q * q * p2[1]
        lit = .12 + .88 * np.exp(-((q - catch) / .09) ** 2)              # the light catches it along a short stretch, as it does a real scratch
        for i in range(n - 1):
            dl.line([(bx[i] * w, by[i] * h), (bx[i + 1] * w, by[i + 1] * h)], fill=float(lit[i]), width=1)
    a = np.stack([gaussian_filter(np.asarray(im, np.float32), .7) for im in (near, off, lines)], -1)
    os.makedirs(UI, exist_ok=True)
    np.save(os.path.join(UI, "dust.npy"), a.astype(np.float16))
    Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255)).save(os.path.join(UI, "dust.png"))
    print("dust", a.shape, flush=True)


def glass(g, v, u, blur, focus=(1.0, 2.2, .35), gain=(1.0, .8, 1.6)):
    """The glass's three layers sampled at (v, u), as parts of the map, each blurred by how far it is from focus: blur is the near layer's, in pixels of the frame."""
    out = 0
    for i, (f, k) in enumerate(zip(focus, gain)):
        d = map_coordinates(g[..., i], [v * g.shape[0], u * g.shape[1]], order=1, mode="constant")
        out = out + k * gaussian_filter(d, max(blur * f, .3))
    return out


# ---------------------------------------------------------------- the push of 十

# the balance at night: the lamp is the room's light, the screen a cool white on what faces it, the city only in the window
NIGHT = dict(lamp=2.4, glow=1.6, city=.5, exposure=.9, saturation=.72, glass=.03, white=.97)


def load(name, w, h):
    a = np.load(os.path.join(PLATES, name + ".npy")).astype(np.float32)
    if a.shape[0] != h:
        a = np.stack([np.asarray(Image.fromarray(a[..., i]).resize((w, h), Image.BILINEAR)) for i in range(3)], -1)
    return a


def quad_mask(corners, w, h, ss=4):
    """How much of each pixel the screen covers, from its corners."""
    im = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(im).polygon([(x * ss, y * ss) for x, y in corners], fill=255)
    return np.asarray(im.resize((w, h), Image.BOX), np.float32)[..., None] / 255


def lay_in(pic, corners, w, h):
    """The screen's picture (display light, linear) where the camera sees it."""
    sh, sw = pic.shape[:2]
    span = np.hypot(*(np.array(corners[1]) - np.array(corners[0])))
    shrink = max(sw / span / 2.0, 1.0)
    small = np.asarray(Image.fromarray(pic).resize((int(sw / shrink), int(sh / shrink)), Image.LANCZOS), np.float32) / 255
    small = room.srgb_to_linear(small)
    Hm = room.homography(np.array(corners, float), np.array(((0, 0), (sw, 0), (sw, sh), (0, sh)), float))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    acc = 0
    for dy in (-.25, .25):
        for dx in (-.25, .25):
            X, Y = xx + .5 + dx, yy + .5 + dy
            d = Hm[2, 0] * X + Hm[2, 1] * Y + Hm[2, 2]
            u, v = (Hm[0, 0] * X + Hm[0, 1] * Y + Hm[0, 2]) / d, (Hm[1, 0] * X + Hm[1, 1] * Y + Hm[1, 2]) / d
            acc = acc + np.stack([map_coordinates(small[..., i], [v / shrink - .5, u / shrink - .5], order=1, mode="nearest") for i in range(3)], -1)
    return acc / 4


def ten_frame(name, k, w=1920, h=1080, beauty=False):
    """One frame of the push, from plates NAME_PASS (or one NAME beauty plate), with the screen laid in."""
    with open(os.path.join(PLATES, name + ".json")) as f:
        j = json.load(f)
    s = w / j["width"]
    corners = [(x * s, y * s) for x, y in j["corners"]]
    L = NIGHT
    if beauty:
        scene = load(name, w, h)
        base = scene
    else:
        base = load(f"{name}_lamp", w, h) * L["lamp"] + load(f"{name}_city", w, h) * L["city"]
        scene = base + load(f"{name}_glow", w, h) * L["glow"]
    y = room.room_tone(room.film(scene * L["exposure"]))
    lum = y.mean(-1, keepdims=True)
    y = lum + (y - lum) * L["saturation"]          # a lamp at night is warm, but not a sunset
    m = quad_mask(corners, w, h)
    shown = lay_in(picture(k), corners, w, h) * L["white"]
    back = room.film(gaussian_filter(base, (6 * s, 6 * s, 0)) * L["exposure"]) * L["glass"]
    return y * (1 - m) + (shown + back - shown * back) * m


# ---------------------------------------------------------------- 甲 and 乙, the screen whole

def flat(k, scale=1.0, push=1.0, on=(.5, .5)):
    """The screen square on, as 三 cuts to it: what is on it and nothing else, a slow push about a point."""
    pic = room.srgb_to_linear(picture(k).astype(np.float32) / 255)
    W, H = int(1920 * scale), int(1080 * scale)
    z = pic.shape[1] / W
    fx, fy = on[0] * W, on[1] * H
    ax = fx + ((np.arange(2 * W) + .5) / 2 - fx) / push
    ay = fy + ((np.arange(2 * H) + .5) / 2 - fy) / push
    y = room.bilinear(pic, ax * z - .5, ay * z - .5)
    y = (y[0::2, 0::2] + y[1::2, 0::2] + y[0::2, 1::2] + y[1::2, 1::2]) / 4
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((xx - W / 2) / W, (yy - H / 2) / W)
    return y * (1 - .08 * r * r)[..., None]


# ---------------------------------------------------------------- the reverse

# the near layer: the last two lines high in the frame, over the dark wall and across the lit gap of the door, where the words stand dark; the room in the middle, the cup and keyboard at the foot
REV = dict(glow=1.0, city=12.0, hall=1.5, exposure=.85, saturation=.8, veil=.04, ink=.5, dust=.4, dust_blur=4.0, scale=2.6, blur=8.0, line=(1590, 807), line_y=.20, band=55.0)


def near_layer(x, w, h):
    """The screen from behind, right in front of the lens: the page's white light caught in the glass round the last paragraph as a faint veil, its words softer, darker holes in it, the wrong way round; and the glass's dust, lit from behind, as soft discs."""
    R = REV
    pic = end_picture(x).astype(np.float32) / 255
    lum = room.srgb_to_linear(pic).mean(-1)[:, ::-1]           # seen from behind
    ph, pw = lum.shape
    # the part of the screen in front of the lens: the last two lines, their starts at the frame's right, made as big as glass a hand's width off makes them
    k = R["scale"] * w / 1920
    right, mid = R["line"]                                     # on the mirrored picture: just right of where the lines start, and between them
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = right + (xx - w) / k, mid + (yy - R["line_y"] * h) / k
    near = map_coordinates(lum, [v, u], order=1, mode="nearest")
    near = gaussian_filter(near, R["blur"] * w / 1920)
    band = np.exp(-((v - mid) / R["band"]) ** 2)                # the veil is the light of the page round the words; away from them it thins
    g = np.load(os.path.join(UI, "dust.npy")).astype(np.float32)[:, ::-1]
    d = glass(g, v / ph, u / pw, R["dust_blur"] * w / 1920, gain=(1.0, .8, .9))      # the scratch a hairline, not a streak
    return near, band, d


def reverse_frame(x, w=1920, h=1080, name="reverse"):
    R = REV
    scene = sum(load(f"{name}_{p}", w, h) * R[p] for p in ("glow", "city", "hall"))
    near, band, d = near_layer(x, w, h)
    y = room.room_tone(room.film(scene * R["exposure"]))
    lum = y.mean(-1, keepdims=True)
    y = lum + (y - lum) * R["saturation"]
    veil = np.array((.93, .95, 1.0), np.float32) * (R["veil"] * near * band)[..., None]
    y = y + veil - y * veil
    y = y * (1 - R.get("ink", 0.0) * ((1 - near) * band)[..., None])      # the words themselves stand dark against whatever light is behind them
    dust_light = np.array((.9, .93, 1.0), np.float32) * (R["dust"] * d)[..., None]
    return y + dust_light - y * dust_light


# ---------------------------------------------------------------- the reverse, again (2026-11-07)

# The screen is the light, and the lens is at the screen: what is nearest is brightest, the cup's rim, the mouse and the keycaps; the room falls away with distance.
# glow: the screen's light, a neutral white a little cool; city: what the window behind the lens still lets in, only a cool floor for the shadows; hall: the hall
# beyond the door, never more than a very dark cool grey. The wood keeps its warmth where the screen lights it; below `dark` the picture loses its colour.
VIEW2 = "E"                      # chosen 2026-11-08 (B, 2026-11-07, before it): film_room_night3d.VIEWS, the reverse from the glass
REV2 = dict(glow=1.0, glow_tint=(.95, .985, 1.04), city=3.0, city_tint=(.60, .70, .88), hall=.004, hall_tint=(.55, .60, .70),
            exposure=6.0, sat=.92, sat_dark=.30, dark=.06)
# the glass right in front of the lens, from behind: the window's white page comes through as one very faint, very wide light, the reply's part of it (its column, down
# to just under the last line), the wrong way round and defocused far past any edge (spread: the blur, frame pixels; light_w: how much of the frame's width it is;
# light_x: where its middle is). In it the last line, 它不存在我这里，只存在你那里。, the wrong way round, read only because its strokes are darker than the light round
# them (no light comes through a stroke, and ink darkens it a little more); its caret blinks with the window's. A few soft discs of dust, out of focus; two short scratches
NEAR2 = dict(char=58.0, at=(.50, .165), blur=2.8, veil=.05, spread=160.0, light_w=.70, light_x=.50, ink=.14, colour=(.93, .96, 1.0),
             discs=7, disc=.007, scratch=.07)
LAST = "它不存在我这里，只存在你那里。"
WHOLE = "所以严格说来，它不存在我这里，只存在你那里。"


@lru_cache(maxsize=4)
def last_line(caret, high=3):
    """The last line of the reply as the window has it, from 它 to the caret, at high times the frame's pixels, linear light, 0 to 1, and its size in characters."""
    t = ink_subs.end_at("reverse")
    x = ink_subs.END["window"]
    page = ring.ui.window(ring._reply(x, True), title=ring.TITLE, w=ring.WIN_W, h=ring.WIN_H, top=ring.TOP)
    on = ring._frame(page, high, ring.clock(t)).astype(np.int16)
    off = ring._frame(ring.ui.window(ring._reply(x, False), title=ring.TITLE, w=ring.WIN_W, h=ring.WIN_H, top=ring.TOP), high, ring.clock(t)).astype(np.int16)
    ys, xs = np.nonzero(np.abs(on - off).sum(-1) > 30)
    cx0, cx1, cy0, cy1 = xs.min(), xs.max(), ys.min(), ys.max()
    lh = cy1 - cy0
    left = int((ring.WIN_X + 10) * ring.K * high)                   # inside the window, clear of the desktop beside it
    band = (on[cy0:cy1 + 1, left:cx0 - 2].mean(-1) < 140).any(0)   # where the line has ink, left of the caret
    ink = np.nonzero(band)[0] + left
    first, last = ink.min(), ink.max()
    pitch = (last - first) / (len(WHOLE) - .5)                   # the last 。 sits on the left of its cell
    x0 = int(first + (len(WHOLE) - len(LAST)) * pitch - .25 * pitch)
    x1 = int(cx1 + 2.5 * pitch)                                   # the blank page after the caret
    y0, y1 = int(cy0 - .45 * lh), int(cy1 + .45 * lh)
    pic = (on if caret else off)[y0:y1, x0:x1].astype(np.float32) / 255
    lum = room.srgb_to_linear(pic).mean(-1)
    # more blank page round it, so the veil can thin out slowly and is gone long before the edge: before 它 (where the page has 说来，), after the caret,
    # above and below. Also where the words are in it: from the left of 它 to the right of the caret.
    py, px = int(2.0 * lh), int(5.0 * pitch)
    words = (px + .25 * pitch, px + cx1 - x0)
    return np.pad(lum, ((py, py), (px, px)), constant_values=1.0), pitch, words


@lru_cache(maxsize=4)
def glass_marks(w, h, seed=5):
    """The soft discs and the two scratches on the glass, as light, 0 to 1 (h, w)."""
    r = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    s = w / 1920
    for _ in range(NEAR2["discs"]):
        x, y = r.uniform(.05, .95) * w, r.uniform(.05, .60) * h
        rad = r.uniform(16, 40) * s
        a = r.uniform(.35, 1.0)
        x0, x1, y0, y1 = int(max(x - rad - 3, 0)), int(min(x + rad + 3, w)), int(max(y - rad - 3, 0)), int(min(y + rad + 3, h))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.hypot(xx - x, yy - y) / rad
        disc = np.clip((1 - d) / .30, 0, 1) * (.85 + .15 * d + .20 * np.exp(-((d - .85) / .08) ** 2))     # soft, a little brighter towards its rim, as a lens draws dust out of focus
        out[y0:y1, x0:x1] += NEAR2["disc"] * a * disc
    for (px, py), ang, length in (((.93, .10), -.35, 110), ((.04, .55), 1.25, 80)):      # by the top right corner, and on the left edge
        L = length * s
        n = 200
        q = np.linspace(-.5, .5, n)
        bend = .06 * L * (q * q - .25)
        xs = px * w + q * L * math.cos(ang) - bend * math.sin(ang)
        ys = py * h + q * L * math.sin(ang) + bend * math.cos(ang)
        lit = .08 + .92 * np.exp(-((q - .1) / .18) ** 2)                  # it catches the light along a short stretch only
        im = Image.new("F", (w, h), 0.0)
        dr = ImageDraw.Draw(im)
        for i in range(n - 1):
            dr.line([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])], fill=float(lit[i]), width=1)
        out += NEAR2["scratch"] * gaussian_filter(np.asarray(im, np.float32), .6 * s)
    return out


@lru_cache(maxsize=2)
def reply_light(w, h):
    """The page's light behind the reply as it comes through the glass, 0 to 1 (h, w): the reply's part of the window as the window has it (its column, centred
    in the window, from the top of the conversation to just under the last line, text and all), the wrong way round, NEAR2["light_w"] of the frame wide, its
    last line on the line of the near words, defocused by NEAR2["spread"] so that no edge of it is left."""
    N = NEAR2
    t, x = ink_subs.end_at("reverse"), ink_subs.END["window"]
    win = lambda c: ring.ui.window(ring._reply(x, c), title=ring.TITLE, w=ring.WIN_W, h=ring.WIN_H, top=ring.TOP)
    on = ring._frame(win(True), 1, ring.clock(t)).astype(np.int16)
    off = ring._frame(win(False), 1, ring.clock(t)).astype(np.int16)
    ys, xs = np.nonzero(np.abs(on - off).sum(-1) > 30)                   # the caret
    cy0, cy1 = ys.min(), ys.max()
    lh = cy1 - cy0 + 1
    left = int((ring.WIN_X + 10) * ring.K)
    band = (on[cy0:cy1 + 1, left:xs.min() - 2].mean(-1) < 140).any(0)
    margin = int(np.nonzero(band)[0].min()) + left                      # where the reply's lines begin
    mid = (ring.WIN_X + ring.WIN_W / 2) * ring.K                         # and the column is centred in the window
    x0, x1 = int(margin - lh), int(2 * mid - margin + lh)
    y0, y1 = int((ring.WIN_Y + ring.TOP) * ring.K), int(cy1 + .6 * lh)
    lum = room.srgb_to_linear(on[y0:y1, x0:x1].astype(np.float32) / 255).mean(-1)[:, ::-1]
    q = 8                                                                # made at an eighth of the frame: nothing in it is finer than the blur
    sig = N["spread"] * w / 1920 / q
    pad = int(3 * sig) + 1
    s = N["light_w"] * w / (x1 - x0) / q
    rw, rh = max(int((x1 - x0) * s), 1), max(int((y1 - y0) * s), 1)
    small = np.asarray(Image.fromarray(lum).resize((rw, rh), Image.BOX), np.float32)
    cw, ch = w // q + 2 * pad, h // q + 2 * pad
    canvas = np.zeros((ch, cw), np.float32)
    line = ((cy0 + cy1) / 2 - y0) * s                                    # the last line, down the small region
    px, py = int(N["light_x"] * w / q - rw / 2) + pad, int(N["at"][1] * h / q - line) + pad
    ya, yb, xa, xb = max(py, 0), min(py + rh, ch), max(px, 0), min(px + rw, cw)
    canvas[ya:yb, xa:xb] = small[ya - py:yb - py, xa - px:xb - px]
    canvas = gaussian_filter(canvas, sig)[pad:pad + h // q, pad:pad + w // q]
    return np.asarray(Image.fromarray(canvas).resize((w, h), Image.BILINEAR), np.float32)


def near2(y, x, w, h):
    """y (display light) seen through the glass in front of the lens."""
    N = NEAR2
    lum, pitch, (wa, wb) = last_line(bool(blink(x)))
    ph, pw = lum.shape
    lum = lum[:, ::-1]                                                    # from behind
    wa, wb = pw - wb, pw - wa                                             # where the words are, the wrong way round too
    k = N["char"] * w / 1920 / pitch                                      # one character N["char"] pixels of a 1920 frame
    tw, th = int(pw * k), int(ph * k)
    sm = np.asarray(Image.fromarray(lum).resize((tw, th), Image.LANCZOS), np.float32)
    sm = gaussian_filter(sm, N["blur"] * w / 1920)
    cx, cy = int(N["at"][0] * w), int(N["at"][1] * h)
    x0, y0 = int(cx - (wa + wb) / 2 * k), cy - th // 2                   # the words' middle at cx
    stroke = np.zeros((h, w), np.float32)
    ya, yb, xa, xb = max(y0, 0), min(y0 + th, h), max(x0, 0), min(x0 + tw, w)
    stroke[ya:yb, xa:xb] = np.clip(1 - sm[ya - y0:yb - y0, xa - x0:xb - x0], 0, 1)   # 0 on the blank page round them, so no edge
    light = (N["veil"] * reply_light(w, h) * (1 - stroke))[..., None] * np.array(N["colour"], np.float32)
    y = y + light - y * light                                             # no light comes through a stroke
    y = y * (1 - N["ink"] * stroke[..., None])                            # and the stroke is a little darker than what is behind it
    marks = glass_marks(w, h)[..., None] * np.array(N["colour"], np.float32)
    return y + marks - y * marks


# ---------------------------------------------------------------- the reverse from the glass (2026-11-08)

# Views D-F of film_room_night3d.py: the screen's own eye, level from the middle of the glass, out at the empty chair at the desk. The screen is the only key and
# the room has its blacks: no city floor and no veil. The page's light a neutral, slightly cool white, the wood keeping its colour where it is lit; the hall only the
# thin warm line in the door's gap. black: the darkest light taken off before the curve, so that the far wall, which a screen three metres off hardly reaches, goes
# to black instead of to grey, while the desk's edge and the chair, near the screen, stay bright.
FROM_GLASS = ("D", "E", "F")
REV2D = dict(glow=1.0, glow_tint=(.95, .985, 1.04), city=0.0, city_tint=(.60, .70, .88), hall=.4, hall_tint=(1.0, .80, .58),
             exposure=28.0, black=.012, sat=.95, sat_dark=.40, dark=.04)
# the glass in front of that lens: no words, no caret, no scratches, and no dust either. Two large faint discs were tried: over the dark wall they read as stains
# on it, not as dust on the glass. (where: discs' middles as parts of the frame and their radii in pixels of a 1920 frame; disc: how much light the brightest gives)
NEAR2D = dict(where=(), disc=.004, colour=(.93, .96, 1.0))


@lru_cache(maxsize=2)
def glass_discs(w, h):
    """NEAR2D's discs, as light, 0 to 1 (h, w): soft, a little brighter towards the rim, as a lens draws dust far out of focus."""
    out = np.zeros((h, w), np.float32)
    s = w / 1920
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for u, v, rad in NEAR2D["where"]:
        d = np.hypot(xx - u * w, yy - v * h) / (rad * s)
        out += NEAR2D["disc"] * np.clip((1 - d) / .45, 0, 1) * (.85 + .15 * d + .15 * np.exp(-((d - .8) / .1) ** 2))
    return out


def near2d(y, w, h):
    """y seen through the glass right in front of the lens, from the glass's middle."""
    marks = glass_discs(w, h)[..., None] * np.array(NEAR2D["colour"], np.float32)
    return y + marks - y * marks


_rev2 = {}


def reverse2_frame(x, w=1920, h=1080, name=None):
    """The reworked reverse x seconds in, from plates rev2VIEW_PASS (1920, made once: the lens never moves and the screen's light hardly changes). Views D-F, from
    the glass, are graded by REV2D and have only NEAR2D's glass in front of them; A-C as they were."""
    name = name or "rev2" + VIEW2
    glass = name[4:] in FROM_GLASS
    key = (name, w, h)
    if key not in _rev2:
        L = REV2D if glass else REV2
        g = load(f"{name}_glow", w, h) * np.array(L["glow_tint"], np.float32) * L["glow"]
        c = load(f"{name}_city", w, h).mean(-1, keepdims=True) * np.array(L["city_tint"], np.float32) * L["city"]
        hl = load(f"{name}_hall", w, h).mean(-1, keepdims=True) * np.array(L["hall_tint"], np.float32) * L["hall"]
        y = room.film(np.maximum((g + c + hl) * L["exposure"] - L.get("black", 0.0), 0))
        lum = y.mean(-1, keepdims=True)
        s = L["sat_dark"] + (L["sat"] - L["sat_dark"]) * np.clip(lum / L["dark"], 0, 1)
        _rev2[key] = (lum + (y - lum) * s).astype(np.float32)
    return near2d(_rev2[key], w, h) if glass else near2(_rev2[key], x, w, h)


# ---------------------------------------------------------------- out

def finish(y, seed, grain=.006):
    return room.finish(y, seed, grain)


def save(a, path):
    Image.fromarray(a).save(path, compress_level=1)


def still(shot, x, scale=1.0):
    w, h = int(1920 * scale), int(1080 * scale)
    if shot == "ten":
        name = f"ten_{x:.2f}"
        f = finish(ten_frame(name, key("ten", x * (JIA - TEN)), w, h), 7)
        t = TEN + x * (JIA - TEN)
    elif shot == "reverse":
        return finish(reverse2_frame(x, w, h), 7, 0.0)            # clean, as the cut has it, and no grain: the reverse is outside the cards
    else:
        start = JIA if shot == "jia" else YI
        f = finish(flat(key(shot, x), scale), 7, .0035)
        t = start + x
    full = f if scale == 1.0 else np.asarray(Image.fromarray(f).resize((1920, 1080), Image.LANCZOS))
    full = ink_subs.apply(full, t)
    return full if scale == 1.0 else np.asarray(Image.fromarray(full).resize((w, h), Image.LANCZOS))


REV_S = 5.0      # the reverse, 5 s, the lens locked. Cut from the film on 2026-11-08 (ink_subs.END has no "reverse" now; its code stays). Read off END until v7:
                 # the cut left this line raising on import, so film_mix lost film_names.shots() and 十二's brush sounds in v6
TALK_PUSH = 1.06                                  # 甲 and 乙 come in on the screen by this much between them, evenly, about the user's bubble, which the cut from one to the other keeps in place
BUBBLE = (.80, .19)                               # the bubble's right end, where both windows have it, as parts of the frame


def length(shot):
    return {"ten": JIA - TEN, "jia": YI - JIA, "yi": YI_END - YI, "reverse": REV_S}[shot]


def frame(shot, i, scale=.5, tag=""):
    """Frame i of a shot, clean (the cut burns the subtitles): ten from its beauty plates ten_fNNNN, 甲 and 乙 square on, the reverse from its plates."""
    w, h = int(1920 * scale), int(1080 * scale)
    x = i / FPS
    if shot == "ten":
        return finish(ten_frame(f"ten{tag}_f{i:04d}", key("ten", x), w, h, beauty=True), 7 + i)
    if shot == "reverse":
        return finish(reverse2_frame(x, w, h), 7 + i, 0.0)        # the plates are 1920 whatever the tag: load() scales them down for the rough cut
    t = (JIA if shot == "jia" else YI) + x
    push = TALK_PUSH ** ((t - JIA) / (YI_END - JIA))
    return finish(flat(key(shot, x), scale, push, BUBBLE), 7 + i, .0035)


def seq(shot, scale=.5, tag=""):
    """The shot's frames, seq/SHOTTAG/NNNN.png, then seq/SHOTTAG.mp4 at 30 a second."""
    folder = os.path.join(SEQ, shot + tag)
    os.makedirs(folder, exist_ok=True)
    n = int(round(length(shot) * FPS))
    for i in range(n):
        save(frame(shot, i, scale, tag), os.path.join(folder, f"{i:04d}.png"))
        if i % 30 == 0:
            print(shot, i, "/", n, flush=True)
    movie = os.path.join(SEQ, f"{shot}{tag}.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(folder, "%04d.png"), "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", movie], check=True)
    print(movie, n, "frames", flush=True)


def main():
    what, args = sys.argv[1], [a for a in sys.argv[2:] if "=" not in a]
    opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
    os.makedirs(KEYS, exist_ok=True)
    if what == "ui":
        ui()
    elif what == "dust":
        dust()
    elif what == "seq":
        for a in args:
            seq(a, float(opts.get("pct", 50)) / 100, opts.get("tag", ""))
    elif what == "rev2":                             # python film_room_night.py rev2 A B C [t=2.5] [pct=50]: keys/rev2_V.png at 1920 (or pct of it), and keys/rev2_V_crop.png, the near text and the cup at 100 % (D-F: the chair and the desk's edge)
        global VIEW2
        x = float(opts.get("t", 2.5))
        scale = float(opts.get("pct", 100)) / 100
        w, h = int(1920 * scale), int(1080 * scale)
        for v in args or ["A"]:
            VIEW2 = v
            f = finish(reverse2_frame(x, w, h), 7, 0.0)
            save(f, os.path.join(KEYS, f"rev2_{v}.png"))
            if v in FROM_GLASS:
                save(f[int(.40 * h):, int(.22 * w):int(.78 * w)], os.path.join(KEYS, f"rev2_{v}_crop.png"))
            else:
                cy = int(NEAR2["at"][1] * h)
                text = f[max(cy - 150, 0):max(cy - 150, 0) + 300, 410:1510]
                cup = f[h - 380:h - 20, 150:1250]
                save(np.concatenate([text, cup], 0), os.path.join(KEYS, f"rev2_{v}_crop.png"))
            print(v, flush=True)
    elif what in ("ten", "jia", "yi", "reverse"):
        scale = float(opts.get("pct", 100)) / 100
        for a in args or ["0"]:
            save(still(what, float(a), scale), os.path.join(KEYS, f"{what}_{a}.png"))
        print(KEYS, flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
