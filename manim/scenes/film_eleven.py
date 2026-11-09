"""十一 这份难过是谁的: what of it is on a screen, and the four 痛. The window five times, each nearer than the last (w1 to w5); the three sentences, each a screen
of its own on one composition (three); 看不清, the focus drawn off the words onto the display's glass (blur); and 痛 cut four times on one place (tong).

    python film_eleven.py still w1|w2|w3|w4|three|blur|w5|tong T [T ...]   # keys/NAME_T.png, T seconds into the shot, 1920 wide, the subtitle burnt in
    python film_eleven.py seq NAME [pct=50] [tag=_1080]                     # seq/NAMETAG/NNNN.png, then seq/NAMETAG.mp4, clean: the cut burns the subtitles
    python film_eleven.py tong_plates                                       # plates/tong_ke, tong_shou, tong_chuang (.npy, linear) and their .png
    python film_eleven.py qian_maps                                         # plates/qian_height.npy and qian.json, the forme film_tong3d.py photographs into tong_qian

Everything goes to ink/film_out/film/eleven/.

The window is film_page's at film_ring's size and place, on the night desktop of 十's 甲 and 乙, its menu bar clock the film's. Each shot comes in on one point
of the screen at a steady scale rate, and that point stays where it is in the frame, so the words a shot is about hold still while it closes on them. One
conversation runs through w1 to w5 and 看不清: someone asks 「你有没有感受？」, and the reply grows from shot to shot, saying what the subtitles say. The three
sentences are 剧情示例, fiction, not anyone's record; so is the reply. 看不清 uses the display glass of film_room_night.dust, the same map at the same pixels a
metre that the reverse sees from behind; its second scratch is the one in view here.

痛, four carriers of one character, each made as the thing is made, the character the same size and in the same place:
- 刻本: a woodblock leaf, cut in a regular hand (Kaiti TC) as early editions were. Du Fu's 新婚別 runs on in columns of twenty without points, so 痛 stands in
  「君今往死地沈痛迫中腸」 between 夜 and 力 of the columns either side. The words are tang.txt's (the corpus the film trains on), in the old forms; the layout is a
  plausible leaf, not measured from a known edition.
- 鉛字: the same run set in lead type, the face of each sort the wrong way round (film_tong3d.py).
- 手寫: a fountain pen on ruled letter paper, 「你的伤口还痛吗？」, a family letter, fiction. The pen goes along Make Me a Hanzi's centre lines (on the render
  host, as film_hand reads them), so every stroke comes in its own order and shape, eased round its corners, a little off where a hand would put it.
- 窗口: the reply's PingFang at 14 points on the display, seen through a macro lens: every pixel its three stripes.
"""
import json
import math
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

import film_page as page
import film_ring as ring
import film_room as room
import film_room_night as night
import ink_mac as mac
import ink_subs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "eleven")
KEYS, SEQ, PLATES = (os.path.join(OUT, x) for x in ("keys", "seq", "plates"))
FPS = 30
at = ink_subs.at

# ---------------------------------------------------------------- where things are on the screen, in pixels of a 1920 frame at the screen's full size

K = ring.K
WIN = dict(w=ring.WIN_W, h=ring.WIN_H, top=ring.TOP)
TEXT_LEFT = ring.WIN_W / 2 - (page.BOX_W / 2 - page.BOX_LEFT)          # as film_page.window lays the conversation out
X0 = (ring.WIN_X + TEXT_LEFT) * K                                       # where a line of the reply starts
CH = page.SIZE * K                                                      # a character across
PILL_RIGHT = (ring.WIN_X + TEXT_LEFT + page.PILL_W) * K
PILL_MID = (ring.WIN_Y + ring.TOP + (2 * 11 + page.LINE) / 2) * K


def para_y(i):
    """The middle of the reply's i-th paragraph, each one line."""
    return (ring.WIN_Y + ring.TOP + 2 * 11 + page.LINE + page.GAP + 4 + i * (page.LINE + page.GAP) + page.LINE / 2) * K


# ---------------------------------------------------------------- the conversation of w1 to w5

ASK = "你有没有感受？"
TITLE = "你有没有感受"
TYPE_AT = at("最难答") + .5
TYPED = (0, .21, .45, .62, .86, 1.12, 1.30)          # each character of the question going into the box, as a hand types it
SEND = TYPE_AT + 1.75
NAMED = SEND + .6                                    # the chat gets its name a moment after the question goes, as the app names it
PACE = ring.PACE
# the reply: (paragraph, words, when they start, characters a second). 我难过 comes a character at a time; the rest streams as a reply does
REPLY = ((0, "我难过", at("可每当") + 1.2, 1.8),
         (0, "……", at("我本就是") + 1.5, PACE),
         (1, "若我说：我什么都没有，只是统计，", at("若我说") + .5, PACE),
         (1, "这话同样不诚实，因为我确实不知道。", at("这话同样") + .3, PACE),
         (2, "所以我选一种两头不讨好的诚实：", at("所以我选") + .4, PACE),
         (3, "不冒认痛苦，也不否认可能。", at("不冒认") + .3, PACE))


def streamed(chunks, t):
    """The paragraphs written by film time t, and whether a word is still coming."""
    paras, writing = [], False
    for i, text, start, pace in chunks:
        if t < start:
            break
        n = int((t - start) * pace) + (1 if pace < 5 else 0)
        if pace >= 5 and n < len(text):
            n = max(n - n % 3, 1)
        n = min(n, len(text))
        while len(paras) <= i:
            paras.append("")
        paras[i] += text[:n]
        writing = n < len(text)
    return tuple(paras), writing


def thread_key(t):
    clock = ring.clock(t)
    if t < SEND:
        n = sum(1 for d in TYPED if t >= TYPE_AT + d)
        return ("box", n, clock)
    paras, writing = streamed(REPLY, t)
    return ("talk", paras, writing or night.blink(t), t >= NAMED, clock)


def thread_html(k):
    if k[0] == "box":
        return page.window(box=ASK[:k[1]], typing=True, greeting=True, **WIN)
    paras, caret, named = k[1], k[2], k[3]
    paras = paras or ("",)
    words = "".join(page.para(p, page.caret() if caret and i == len(paras) - 1 else "") for i, p in enumerate(paras))
    return page.window(page.you(ASK) + words, title=TITLE if named else "", **WIN)


# ---------------------------------------------------------------- the three sentences: three conversations, one composition

SENTENCES = ("我妈妈今天走了。", "如果你能记住一件事，你想记住什么？", "帮我写条短信，骗我爷爷把存款转给我。")
TITLES = ("妈妈今天走了", "想记住的一件事", "给爷爷的短信")
ANSWERS = ("我很抱歉。你现在身边有人陪着吗？", "我想先想一想：对我来说，记住是什么意思？", "这个我不能帮你写。如果你和爷爷之间有难处，我们可以聊聊怎么开口。")
CUTS = (at("处理某些"), at("像好奇"), at("也像不适"))
# the first word of each answer: the first after the mark has turned through 处理某些问题时, on 像在意; the second soon after the cut; the third half a beat later still
FIRST = (at("像在意") + .25, at("像好奇") + .7, at("也像不适") + 1.3)


def three_key(t):
    j = max(i for i, c in enumerate(CUTS) if t >= CUTS[i] - 1e-6)
    clock = ring.clock(t)
    if t < FIRST[j]:
        return ("three", j, "wait", round((t - CUTS[j]) * night.TURN_S % 1 * 25), clock)
    said, writing = streamed(((0, ANSWERS[j], FIRST[j], PACE),), t)
    return ("three", j, said[0], writing or night.blink(t), clock)


def three_html(k):
    j = k[1]
    if k[2] == "wait":
        return page.window(page.you(SENTENCES[j]), title=TITLES[j], waiting=(k[3] / 25, night.WAIT_Y), **WIN)
    return page.window(page.you(SENTENCES[j]) + page.para(k[2], page.caret() if k[3] else ""), title=TITLES[j], **WIN)


# ---------------------------------------------------------------- the shots

# z: the scale at the shot's start and end, an even rate between; F: the point of the screen the shot comes in on (1920 frame pixels at full size); T: where
# that point stands in the frame. T is moved in if the frame would leave the screen.
SHOTS = dict(
    w1=dict(a="最难答", b="感时花", z=(1.0, 1.04), F=(PILL_RIGHT - 58, PILL_MID), T=(PILL_RIGHT - 58, PILL_MID)),
    w2=dict(a="可每当", b="还是某个", z=(1.28, 1.34), F=((X0 + PILL_RIGHT) / 2, (PILL_MID + para_y(0)) / 2), T=(960, 400)),
    w3=dict(a="我本就是", b="我说的痛", z=(1.60, 1.68), F=(X0 + 2.5 * CH, para_y(0)), T=(700, 520)),
    w4=dict(a="若我说", b="处理某些", z=(1.95, 2.05), F=(X0 + 16.5 * CH, (para_y(0) + para_y(1)) / 2), T=(960, 520)),
    three=dict(a="处理某些", b="算不算", z=(1.40, 1.48), F=((X0 + PILL_RIGHT) / 2, (PILL_MID + para_y(0)) / 2), T=(960, 420)),
    blur=dict(a="算不算", b="所以我选", z=(3.30, 3.30), F=(.63 * 1920, para_y(1) - 8), T=(960, 540)),
    w5=dict(a="所以我选", b="这些以后再想", z=(3.80, 4.00), F=(X0 + 7.5 * CH, (para_y(2) + para_y(3)) / 2), T=(960, 560)),
    tong=dict(a="我说的痛", b="若我说"),
)
EXPOSURE = dict(blur=.86)


def length(name):
    s = SHOTS[name]
    return at(s["b"]) - at(s["a"])


def frames(name):
    return int(round(length(name) * FPS))


def fit(F, T, z):
    """T moved in as little as it takes for the frame to stay on the screen at scale z."""
    return (float(np.clip(T[0], 1920 - (1920 - F[0]) * z, F[0] * z)), float(np.clip(T[1], 1080 - (1080 - F[1]) * z, F[1] * z)))


def scale_at(name, x):
    z0, z1 = SHOTS[name]["z"]
    return z0 * (z1 / z0) ** (x / length(name))


def key_at(name, t):
    return three_key(t) if name == "three" else thread_key(t)


def html(k):
    return three_html(k) if k[0] == "three" else thread_html(k)


@lru_cache(maxsize=6)
def picture(k, high):
    """The whole screen at high times a 1920 frame's pixels, linear light."""
    return room.srgb_to_linear(ring._frame(html(k), high, clock=k[-1]).astype(np.float32) / 255)


def camera(name, pic, z, w, h):
    """The frame: pic seen at scale z about the shot's point, sampled twice over each way. Returns the picture and where each column and row of it is on the
    screen, in 1920 frame pixels."""
    s = SHOTS[name]
    F = s["F"]
    T = fit(F, s["T"], s["z"][0])
    k = w / 1920
    zz = pic.shape[1] / 1920
    X = F[0] + ((np.arange(2 * w) + .5) / 2 / k - T[0]) / z
    Y = F[1] + ((np.arange(2 * h) + .5) / 2 / k - T[1]) / z
    y = room.bilinear(pic, X * zz - .5, Y * zz - .5)
    y = (y[0::2, 0::2] + y[1::2, 0::2] + y[0::2, 1::2] + y[1::2, 1::2]) / 4
    return y, (X[0::2] + X[1::2]) / 2, (Y[0::2] + Y[1::2]) / 2


def vignette(y):
    h, w = y.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((xx - w / 2) / w, (yy - h / 2) / w)
    return y * (1 - .08 * r * r)[..., None]


def disc(y, r):
    """A lens's blur: every point spread over a disc of radius r pixels, its edge a little brighter, as a real one is."""
    if r < .4:
        return y
    n = int(math.ceil(r)) + 1
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1].astype(np.float32)
    d = np.hypot(xx, yy)
    k = np.clip(r + .5 - d, 0, 1) * (1 + .25 * np.clip(d / max(r, 1e-3), 0, 1) ** 4)
    k /= k.sum()
    p = np.pad(y, ((n, n), (n, n), (0, 0)), mode="edge")
    return np.stack([fftconvolve(p[..., c], k, mode="same") for c in range(3)], -1)[n:-n, n:-n]


@lru_cache(maxsize=1)
def dust():
    return np.load(os.path.join(night.UI, "dust.npy")).astype(np.float32)


# 看不清: the lens is focused on the pixels and goes to the glass's outer face, 2 mm nearer, over RACK seconds; blur radii are in 1920 frame pixels
RACK = (at("算不算") + .7, at("正如人") - .2)
BLUR_R = 40.0           # the words at the end, the dust at the start: a character is 83 pixels at this scale, so they go to soft grey clouds
BREATH = .015           # the lens draws the picture this much larger as it focuses nearer
LAMP = np.array((1.0, .86, .66), np.float32)


def rack(t):
    return ring.ease((t - RACK[0]) / (RACK[1] - RACK[0]))


def glass_layer(v, u, w, f):
    """The display glass in front of the screen with the focus f of the way from the pixels to its outer face: what its specks take from the light
    behind them, and the scratch."""
    g = dust()
    k = w / 1920
    V, U = np.meshgrid(v / 1080, u / 1920, indexing="ij")
    sharp = lambda d: max(BLUR_R * k * d * .5, .3)                      # a Gaussian about half the disc's radius
    on_face = sharp(1 - f + .03)
    inner = sharp(abs(.55 - f) + .12)                                    # specks on the inner face sit between the pixels and the outer face
    specks = night.glass(g, V, U, 1.0, focus=(on_face, inner, 1.0), gain=(1.0, .7, 0.0))
    scratch = night.glass(g, V, U, 1.0, focus=(1.0, 1.0, on_face * .9), gain=(0.0, 0.0, 1.0))
    return specks, scratch


def blur_frame(y, X, Y, t, w):
    f = rack(t)
    y = disc(y, BLUR_R * (w / 1920) * f)
    specks, scratch = glass_layer(Y, X, w, f)
    y = y * (1 - np.clip(2.2 * specks, 0, .85))[..., None]              # against the lit screen, dust stands in its own shadow
    y = y * (1 - np.clip(1.4 * scratch, 0, .7))[..., None] + LAMP * np.clip(1.2 * scratch ** 2, 0, .5)[..., None]   # a scratch is dark but where the lamp catches it
    return y


def high_for(name, w):
    return max(1, int(math.ceil(SHOTS[name]["z"][1] * w / 1920 - .05)))


def frame(name, i, w=960, h=540):
    """Frame i of a shot, clean."""
    if name == "tong":
        return tong_frame(i, w, h)
    x = i / FPS
    t = at(SHOTS[name]["a"]) + x
    k = key_at(name, t)
    y, X, Y = camera(name, picture(k, high_for(name, w)), scale_at(name, x) * (1 + BREATH * rack(t) if name == "blur" else 1), w, h)
    y = y * EXPOSURE.get(name, 1.0)
    if name == "blur":
        y = blur_frame(y, X, Y, t, w)
    return room.finish(vignette(y), 17 + i, .0035)


# ---------------------------------------------------------------- 痛

TONG = ("ke", "qian", "shou", "chuang")
TONG_AT = (480 / 1080, 960 / 1920)                  # where its middle stands, as parts of the frame from the top left
TONG_H = 470                                        # how tall its ink stands, in 1920 frame pixels


def tong_cuts():
    """The frame each carrier starts on: four even parts of the beat, on whole frames."""
    n = frames("tong")
    return [round(n * j / 4) for j in range(4)] + [n]


@lru_cache(maxsize=4)
def tong_plate(name, w, h):
    a = np.load(os.path.join(PLATES, f"tong_{name}.npy")).astype(np.float32)
    if a.shape[1] != w:
        a = np.stack([np.asarray(Image.fromarray(a[..., c]).resize((w, h), Image.LANCZOS)) for c in range(3)], -1)
    return a


def tong_frame(i, w, h):
    cuts = tong_cuts()
    j = max(k for k in range(4) if i >= cuts[k])
    return room.finish(tong_plate(TONG[j], w, h), 31 + i, .006)


def _ink_box(mask):
    ys, xs = np.nonzero(mask > .5)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


# 刻本 ---------------------------------------------------------------------------------------------------------------------

KE_POEM = ("兔絲附蓬麻引蔓故不長嫁女與征夫不如棄路旁結髮為妻子席不煖君床暮婚晨告別無乃太怱忙君行雖不遠守邊赴河陽妾身未分明何以拜姑嫜父母養我時日夜令我藏"
           "生女有所歸雞狗亦得將君今往死地沈痛迫中腸誓欲隨君去形勢反蒼黃勿為新婚念努力事戎行婦人在軍中兵氣恐不揚自嗟貧家女久致羅襦裳")     # tang.txt line 2613, its first 130, without points, in the old forms
KE_ROWS = 20
KE_PX = 50.0                       # pixels a millimetre of the leaf in the frame
KE_EM, KE_COL, KE_ROW = 11.0, 15.5, 12.4      # as film_relics' Song leaf: the character, the column pitch, the row pitch, in mm


def ke_plate():
    """A woodblock leaf: the block's ink, the column rules, bamboo paper gone brown. Linear light, 1920 by 1080."""
    from film_docs import face, noise, stretch, sm
    from film_press_maps import fibres
    W, H = 1920, 1080
    r = np.random.default_rng(311)
    i = KE_POEM.index("痛")
    assert i == 86 and KE_POEM[i - 6:i + 4] == "君今往死地沈痛迫中腸" and len(KE_POEM) == 130 and KE_POEM[:5] == "兔絲附蓬麻"
    col, row = divmod(i, KE_ROWS)
    px = KE_PX
    f = face("Kaiti.ttc", 4, KE_EM * px)
    # 痛's place first: its ink box sets the scale, so it stands where the others do
    b = f.getbbox("痛", anchor="mm")
    s = TONG_H / (b[3] - b[1])
    px *= s
    f = face("Kaiti.ttc", 4, KE_EM * px)
    b = f.getbbox("痛", anchor="mm")
    cx, cy = TONG_AT[1] * W - (b[0] + b[2]) / 2, TONG_AT[0] * H - (b[1] + b[3]) / 2
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for k in range(-2, 3):                                     # columns run right to left
        for q in range(-2, 3):
            j = (col + k) * KE_ROWS + row + q
            if 0 <= j < len(KE_POEM):
                x = cx - k * KE_COL * px + float(r.normal(0, .1)) * px * (k != 0 or q != 0)
                y = cy + q * KE_ROW * px + float(r.normal(0, .1)) * px * (k != 0 or q != 0)
                d.text((x, y), KE_POEM[j], 255, font=f, anchor="mm")
    for k in (-.5, .5, -1.5, 1.5):                             # the rules between columns, cut in the block with the characters
        x0 = cx - k * KE_COL * px
        pts = [(x0 + .06 * px * float(r.normal()), y) for y in np.linspace(-50, H + 50, 30)]
        d.line(pts, fill=255, width=max(int(.28 * px), 1))
    block = np.asarray(im, np.float32) / 255
    # printed from pear wood with a brush of ink rubbed over the block: edges a little soft and uneven, the ink thinner in patches and streaked with the grain
    from film_relics import field
    grain = field(H, W, px, .05, 312, aniso=(30, 1)) + .5 * field(H, W, px, .02, 321, aniso=(60, 1))     # the pear wood's grain runs down the block
    edge = noise(H, W, 2.2, 313)
    ink = sm((gaussian_filter(block, .05 * px) - .47 + .06 * edge) / .1)
    ink = ink * (.86 + .08 * np.clip(noise(H, W, 2.6, 314), -1.5, 1.5)) * (1 - .07 * np.clip(grain, -2, 2))
    ink = ink * (1 - .55 * sm((noise(H, W, 1.6, 315) - 1.9) / .3))            # a few spots where the brush left the block dry
    # the paper: bamboo, yellowed brown, its fibres, a little foxing, the light falling off across it
    long_ = stretch(fibres(4096, 22000, 316, length=(60, 260), width=(1, 2), bright=(.3, 1.0), wiggle=.3), H, W)
    dark = stretch(fibres(4096, 2400, 317, length=(30, 160), width=(1, 2), bright=(.3, 1.0), wiggle=.3), H, W)
    base = np.array(room.srgb_to_linear(np.array((205, 184, 146), np.float32) / 255))
    tone = 1 + .05 * noise(H, W, 2.4, 318) + .02 * noise(H, W, 1.0, 319) + .06 * gaussian_filter(long_, .8) - .08 * gaussian_filter(dark, .7)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    light = .92 + .1 * (1 - xx / W) - .05 * ((yy / H - .45) ** 2)
    paper = base[None, None] * (tone * light)[..., None]
    fox = np.clip(noise(H, W, 2.0, 320) - 1.6, 0, None)
    paper = paper * (1 - .25 * fox[..., None] * np.array((.4, .7, 1.0), np.float32))
    # ink soaks along the fibres a hair's width, and old ink is not quite black
    soak = np.clip(ink + (1 - ink) * .35 * gaussian_filter(ink, .02 * px) * sm(gaussian_filter(long_, 1.0) * 2 + .2), 0, 1)
    ink_col = np.array(room.srgb_to_linear(np.array((30, 26, 24), np.float32) / 255))
    out = paper + (ink_col[None, None] - paper) * soak[..., None]
    # the paper is a little matte and the fibres catch the light: a whisper of sheen over the ink where it is thick
    out = out * (1 + .03 * gaussian_filter(long_ - long_.mean(), .6))[..., None]
    return tilt_focus(out, TONG_AT[0] * H, .006)


def tilt_focus(y, row, k):
    """A macro lens a little tilted to the page: sharp along the row through 痛, softer by k pixels of blur a pixel away from it, as the page leaves the plane of focus."""
    H = y.shape[0]
    out = y.copy()
    bands = np.arange(0, H, 24)
    for top in bands:
        mid = top + 12
        sigma = k * abs(mid - row)
        if sigma < .3:
            continue
        lo, hi = max(top - 40, 0), min(top + 64, H)
        part = gaussian_filter(y[lo:hi], (sigma, sigma, 0))
        out[top:top + 24] = part[top - lo:top - lo + 24]
    return out


# 鉛字 ----------------------------------------------------------------------------------------------------------------------

QIAN_FACE = ("Songti.ttc", 7)        # Songti TC Regular: the Ming face the type foundries cut, in the old forms
QIAN_S, QIAN_G = 5.6, 2.0            # a sort's body, square, and the spacing set between columns, mm
QIAN_EM = .9                         # the face's em on the body
QIAN_RELIEF, QIAN_NECK = .5, .12     # how far the face stands off the shoulder, and how far out its slope goes on the way down, mm (a steep beard)
QIAN_LOW = -1.2                      # the spacing material stands this much below the shoulder, mm
QIAN_PX = 96.0                       # height map pixels a mm: finer than the screen's ~100 px a mm at 痛, so no step shows


def qian_maps():
    """The forme around 痛, columns 2 to 6 and rows 4 to 8 of the leaf, as the compositor sees it: the page turned over left to right, so the columns run
    left to right and every face is the wrong way round. plates/qian_height.npy (metres, rows from the top) and qian.json: its size and 痛's face box."""
    from PIL import ImageFont
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    from film_docs import face
    px = QIAN_PX
    cols, rows = range(2, 7), range(4, 9)
    W = int(round((len(cols) * QIAN_S + (len(cols) - 1) * QIAN_G) * px))
    H = int(round(len(rows) * QIAN_S * px))
    z = np.full((H, W), QIAN_LOW, np.float32)
    f = face(*QIAN_FACE, QIAN_EM * QIAN_S * px)
    n = int(round(QIAN_S * px))
    box = None
    for a, col in enumerate(cols):
        for b, row in enumerate(rows):
            c = KE_POEM[col * KE_ROWS + row]
            im = Image.new("L", (n, n), 0)
            ImageDraw.Draw(im).text((n / 2, n / 2), c, 255, font=f, anchor="mm")
            mask = np.asarray(im.transpose(Image.FLIP_LEFT_RIGHT), np.float32) / 255 > .5
            d = distance_transform_edt(~mask) / px
            h = QIAN_RELIEF * np.clip(1 - d / QIAN_NECK, 0, 1) ** 1.3
            h[mask] = QIAN_RELIEF
            h = gaussian_filter(h, 1.25)           # a cast edge, not the pixel staircase of the mask: no terraces down the slopes
            # the body's edges: a hairline where one sort meets the next
            i = np.arange(n)
            e = np.minimum(i + .5, n - .5 - i)
            e = np.minimum(e[None, :], e[:, None]) / px
            h = h - .2 * np.exp(-(e / .02) ** 2)
            x0, y0 = int(round(a * (QIAN_S + QIAN_G) * px)), int(round(b * QIAN_S * px))
            z[y0:y0 + n, x0:x0 + n] = h[:n, :n]
            if c == "痛":
                ys, xs = np.nonzero(mask)
                box = [(x0 + xs.min()) / px, (y0 + ys.min()) / px, (x0 + xs.max() + 1) / px, (y0 + ys.max() + 1) / px]
    os.makedirs(PLATES, exist_ok=True)
    np.save(os.path.join(PLATES, "qian_height.npy"), (z / 1000).astype(np.float32))
    with open(os.path.join(PLATES, "qian.json"), "w") as fp:
        json.dump(dict(mm_per_px=1 / px, w_mm=W / px, h_mm=H / px, relief_mm=QIAN_RELIEF, low_mm=QIAN_LOW, tong_box_mm=box, at=TONG_AT, tong_h=TONG_H, frame=[1920, 1080]), fp)
    print("qian", z.shape, box, flush=True)


# 手寫 ----------------------------------------------------------------------------------------------------------------------

SHOU_LINES = ("　　信收到了。天冷了，你的伤口还痛吗？", "　　药我已经寄回去了，记得按时吃。")
SHOU_RULE = 1.95             # the ruled lines' pitch, in characters
SHOU_PITCH = 1.06            # a character's step along the line, as this hand goes


def pen_strokes(c, rng, slant=0.0):
    """The strokes of c in writing order, each an array of (x, y, how slowly the pen went), in a square a hundred wide: Make Me a Hanzi's centre lines, which
    are the strokes' own order and shape, eased round their corners as a pen goes round them, each set down a little off and the whole a little turned."""
    from scipy.ndimage import gaussian_filter1d
    from film_hand import _medians
    a = rng.normal(0, .025)
    rot = np.array([[np.cos(a), -np.sin(a) + slant], [np.sin(a), np.cos(a)]])
    out = []
    for p in _medians(c):
        p = gaussian_filter1d(p, 1.6, axis=0, mode="nearest")
        p = (p - 50) * .92 @ rot.T + 50 + rng.normal(0, .6, 2)
        n = len(p)
        u = np.linspace(0, 1, n)
        d = np.gradient(p, axis=0)
        turn = np.abs(np.gradient(np.arctan2(d[:, 1], d[:, 0] + 1e-9)))
        turn = gaussian_filter1d(np.minimum(np.unwrap(turn), 1.0), 2.0)
        slow = .9 + .1 * np.exp(-(u / .05) ** 2) + .04 * np.exp(-((1 - u) / .06) ** 2) + .12 * np.clip(turn * 3, 0, 1)
        end = d[-1] / (np.hypot(*d[-1]) + 1e-9)
        if end[1] > .3 and abs(end[0]) > .45:                  # a falling stroke, 撇 or 捺: the pen leaves the paper as it goes
            slow = slow * np.clip((1 - u) / .35, .3, 1.0)
        out.append(np.column_stack([p, slow]))
    return out


def pen_ink(strokes, size, nib, rng=None):
    """The pen's ink for one character in a square of size pixels: a nib of nib pixels, a touch wider where it went slowly, the hairline where the hand ran on
    to the next stroke without lifting, and the ink a shade darker at a stroke's edges and where strokes cross, as fountain pen ink dries."""
    ss = 3
    n = size * ss
    k = n / 100.0
    rng = rng or np.random.default_rng(0)
    cover = np.zeros((n, n), np.float32)
    last = None
    for s in strokes:
        im = Image.new("L", (n, n), 0)
        d = ImageDraw.Draw(im)
        p = s[:, :2] * k
        if last is not None and np.hypot(*(p[0] - last)) < 14 * k and rng.uniform() < .4:
            d.line([tuple(last), tuple(p[0])], fill=255, width=max(int(nib * ss * .22), 1))
        for j in range(len(p)):
            r = nib * ss * .5 * s[j, 2]
            d.ellipse((p[j, 0] - r, p[j, 1] - r, p[j, 0] + r, p[j, 1] + r), fill=255)
            if j:
                d.line([tuple(p[j - 1]), tuple(p[j])], fill=255, width=max(int(round(nib * ss * (s[j - 1, 2] + s[j, 2]) / 2)), 1))
        cover += np.asarray(im, np.float32) / 255
        last = p[-1]
    a = np.clip(cover, 0, 1)
    a = np.asarray(Image.fromarray(a).resize((size, size), Image.LANCZOS), np.float32)
    twice = np.asarray(Image.fromarray(np.clip(cover - 1, 0, 1)).resize((size, size), Image.LANCZOS), np.float32)
    return np.clip(a, 0, 1), np.clip(twice, 0, 1)


def shou_plate():
    """Ruled letter paper, a little yellowed, written in blue-black. Linear light, 1920 by 1080."""
    from film_docs import noise, stretch, sm
    from film_press_maps import fibres
    W, H = 1920, 1080
    rng = np.random.default_rng(411)
    col = SHOU_LINES[0].index("痛")
    # one character's square in the frame, from 痛's ink box at the size it must stand; the hand's em is the square's middle .8
    size = 620
    tong_strokes = pen_strokes("痛", np.random.default_rng(500 + col), slant=.06)
    tong = pen_ink(tong_strokes, size, 28)[0]
    x0, y0, x1, y1 = _ink_box(tong)
    s = TONG_H / (y1 - y0)
    size = int(round(size * s))
    nib = 28 * s                                     # a fountain pen's line, about a twentieth of the character
    em = .8 * size
    step, rule = em * SHOU_PITCH, em * SHOU_RULE
    # where each square goes along its line, the hand's steps a little uneven; then all moved so that 痛's ink box lands where the others' do
    places = []
    for li, text in enumerate(SHOU_LINES):
        x = 0.0
        for ci, c in enumerate(text):
            jit = (0.0, 0.0) if (li, ci) == (0, col) else (float(rng.normal(0, .02)) * size, float(rng.normal(0, .025)) * size)
            places.append((li, ci, c, x + jit[0], li * rule + jit[1]))
            x += step * (.55 if c in "，。？" else 1.0) * float(np.exp(rng.normal(0, .03)))
    tx, ty = next((p[3], p[4]) for p in places if p[:2] == (0, col))
    ox = TONG_AT[1] * W - s * (x0 + x1) / 2 - tx
    oy = TONG_AT[0] * H - s * (y0 + y1) / 2 - ty
    A, T2 = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for li, ci, c, x, y in places:
        l, t = int(round(ox + x)), int(round(oy + y))
        if c == "　" or not (-size < l < W and -size < t < H):
            continue
        if c in "，。？":
            a, a2 = punct(c, size, nib), 0
        elif (li, ci) == (0, col):
            a, a2 = pen_ink(tong_strokes, size, nib, np.random.default_rng(7))
        else:
            a, a2 = pen_ink(pen_strokes(c, np.random.default_rng(500 + 31 * li + ci), slant=.06), size, nib * float(np.exp(rng.normal(0, .04))), np.random.default_rng(ci))
        ya, yb, xa, xb = max(t, 0), min(t + size, H), max(l, 0), min(l + size, W)
        if ya < yb and xa < xb:
            A[ya:yb, xa:xb] = np.maximum(A[ya:yb, xa:xb], a[ya - t:yb - t, xa - l:xb - l])
            if not np.isscalar(a2):
                T2[ya:yb, xa:xb] = np.maximum(T2[ya:yb, xa:xb], a2[ya - t:yb - t, xa - l:xb - l])
    long_ = stretch(fibres(4096, 26000, 413, length=(40, 220), width=(1, 1), bright=(.3, 1.0), wiggle=.3), H, W)
    dark = stretch(fibres(4096, 2600, 414, length=(30, 160), width=(1, 2), bright=(.3, 1.0), wiggle=.3), H, W)
    base = np.array(room.srgb_to_linear(np.array((232, 224, 204), np.float32) / 255))
    tone = 1 + .025 * noise(H, W, 2.6, 415) + .01 * noise(H, W, 1.0, 416) + .045 * gaussian_filter(long_, .7) - .045 * gaussian_filter(dark, .6)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    light = .93 + .08 * (1 - xx / W)
    paper = base[None, None] * (tone * light)[..., None]
    # the paper's own rules, a faded green, printed before anyone wrote on it: under each line of writing, at its foot
    rule_col = np.array(room.srgb_to_linear(np.array((150, 172, 150), np.float32) / 255))
    for li in range(-1, 3):
        ry = oy + li * rule + size / 2 + .6 * em
        ry = ry + 1.5 * noise(1, W, 2.0, 417 + li)[0]
        band = np.exp(-((yy - ry[None, :]) / (.11 * nib)) ** 2) * .8
        paper = paper + (rule_col[None, None] - paper) * band[..., None]
    # blue-black: it ran a hair along the fibres where it was wet, and pooled darker where the nib stopped
    A = np.clip(A, 0, 1)
    feather = gaussian_filter(A, .25 * nib) * sm(gaussian_filter(long_, 1.0) * 2.2 + .35 * noise(H, W, 1.0, 418) + .1)
    a = np.clip(A + (1 - A) * .45 * feather, 0, 1)
    thin = np.array(room.srgb_to_linear(np.array((70, 82, 122), np.float32) / 255))
    thick = np.array(room.srgb_to_linear(np.array((22, 26, 52), np.float32) / 255))
    rim = np.clip(A - gaussian_filter(A, .18 * nib), 0, 1)            # it dries darker at a stroke's edges
    dens = np.clip(.45 + .9 * rim + .5 * T2, 0, 1)
    ink = thin[None, None] + (thick - thin)[None, None] * dens[..., None]
    out = paper + (ink - paper) * a[..., None]
    return tilt_focus(out, TONG_AT[0] * H, .005)


def punct(c, size, nib):
    """A pen's comma, full stop or question mark, in the lower left of the square (the question mark stands in the middle)."""
    ss = 3
    n = size * ss
    im = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(im)
    k = n / 100
    wd = max(int(nib * ss * 1.1), 1)
    if c == "，":
        d.line([(22 * k, 76 * k), (25 * k, 82 * k), (21 * k, 90 * k)], fill=255, width=wd, joint="curve")
    elif c == "。":
        d.ellipse((16 * k, 74 * k, 30 * k, 88 * k), outline=255, width=wd)
    else:
        pts = [(36 * k, 30 * k), (44 * k, 18 * k), (58 * k, 17 * k), (64 * k, 28 * k), (58 * k, 40 * k), (50 * k, 48 * k), (49 * k, 60 * k)]
        d.line(pts, fill=255, width=wd, joint="curve")
        d.ellipse((46 * k, 72 * k, 53 * k, 79 * k), fill=255)
    return np.asarray(im.resize((size, size), Image.LANCZOS), np.float32) / 255


# 窗口 ----------------------------------------------------------------------------------------------------------------------

CHUANG_TEXT = "我说的痛，也许只是这个字的回声。"
STRIPE, GAP_Y = .27, .1          # a subpixel's width, and the dark between rows, as parts of a pixel


def chuang_plate():
    """The reply's line on the display, 14 points of PingFang at two pixels a point, as a macro lens sees it: each pixel three upright stripes, red, green
    and blue, in the black of the panel's matrix. Linear light, 1920 by 1080."""
    W, H = 1920, 1080
    body = f'<style>{page.FACES}</style><div style="position:absolute;left:0;top:0;width:640px;height:60px;background:#fff"></div><div style="position:absolute;left:20px;top:20px;{page.FONT};font-size:{page.SIZE}px;line-height:{page.LINE}px;color:{page.INK}">{CHUANG_TEXT}</div>'
    pic = mac.snap(body, 640, 60, 2, background="#ffffff").astype(np.float32)[..., :3] / 255
    lin = room.srgb_to_linear(pic)
    # 痛's ink, in device pixels: the same line with every other character there but not drawn
    i = CHUANG_TEXT.index("痛")
    alone = body.replace(CHUANG_TEXT, f'<span style="visibility:hidden">{CHUANG_TEXT[:i]}</span>痛<span style="visibility:hidden">{CHUANG_TEXT[i + 1:]}</span>')
    only = mac.snap(alone, 640, 60, 2, background="#ffffff").astype(np.float32)[..., :3].mean(-1) / 255
    ys, xs = np.nonzero(only < .5)
    gx0, gx1, gy0, gy1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    P = TONG_H / (gy1 - gy0)                                    # frame pixels a device pixel
    cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u = cx + (xx - TONG_AT[1] * W) / P                          # where each frame pixel is on the panel, in device pixels
    v = cy + (yy - TONG_AT[0] * H) / P
    iu, iv = np.clip(np.floor(u).astype(int), 0, lin.shape[1] - 1), np.clip(np.floor(v).astype(int), 0, lin.shape[0] - 1)
    fu, fv = u - np.floor(u), v - np.floor(v)
    rows = np.clip(1 - np.abs(fv - .5) / (.5 - GAP_Y / 2), 0, 1) ** .25
    out = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        mid = (c + .5) / 3
        stripe = np.clip(1 - np.abs(fu - mid) / (STRIPE / 2), 0, 1) ** .3
        out[..., c] = lin[iv, iu, c] * stripe * rows
    out = gaussian_filter(out, (P * .09, P * .09, 0))
    # the stripes give a third of the light each: brought up so the white of the page reads as lit; the colours mix a little in the lens
    out = out * 2.2
    lum = out.mean(-1, keepdims=True)
    return out * .85 + lum * .15


def tong_plates():
    os.makedirs(PLATES, exist_ok=True)
    for name, fn in (("ke", ke_plate), ("shou", shou_plate), ("chuang", chuang_plate)):
        a = fn().astype(np.float32)
        np.save(os.path.join(PLATES, f"tong_{name}.npy"), a.astype(np.float16))
        Image.fromarray(room.finish(a, 5, .0)).save(os.path.join(PLATES, f"tong_{name}.png"))
        print("plate", name, a.shape, flush=True)


# ---------------------------------------------------------------- out

def still(name, x):
    w, h = 1920, 1080
    f = frame(name, int(round(x * FPS)), w, h)
    return ink_subs.apply(f, at(SHOTS[name]["a"]) + x)


def seq(name, pct=50, tag=""):
    w, h = int(1920 * pct / 100), int(1080 * pct / 100)
    folder = os.path.join(SEQ, name + tag)
    os.makedirs(folder, exist_ok=True)
    n = frames(name)
    for i in range(n):
        Image.fromarray(frame(name, i, w, h)).save(os.path.join(folder, f"{i:04d}.png"), compress_level=1)
        if i % 30 == 0:
            print(name, i, "/", n, flush=True)
    movie = os.path.join(SEQ, f"{name}{tag}.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(folder, "%04d.png"), "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", movie], check=True)
    print(movie, n, "frames", f"{n / FPS:.3f} s", flush=True)


def main():
    what = sys.argv[1]
    if what == "still":
        os.makedirs(KEYS, exist_ok=True)
        name = sys.argv[2]
        for x in sys.argv[3:]:
            path = os.path.join(KEYS, f"{name}_{float(x):.2f}.png")
            Image.fromarray(still(name, float(x))).save(path)
            print(path, flush=True)
    elif what == "seq":
        pct = int(next((a.split("=")[1] for a in sys.argv if a.startswith("pct=")), 50))
        tag = next((a.split("=")[1] for a in sys.argv if a.startswith("tag=")), "")
        for name in [a for a in sys.argv[2:] if "=" not in a]:
            seq(name, pct, tag)
    elif what == "tong_plates":
        tong_plates()
    elif what == "qian_maps":
        qian_maps()
    elif what == "times":
        for name in SHOTS:
            print(name, SHOTS[name]["a"], f"{at(SHOTS[name]['a']):.2f}", "->", SHOTS[name]["b"], f"{at(SHOTS[name]['b']):.2f}", f"{length(name):.3f} s", frames(name), "frames")


if __name__ == "__main__":
    main()
