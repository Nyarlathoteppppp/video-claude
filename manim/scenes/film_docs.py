"""The carriers of 七击, each a thing with a time of its own: a clay tablet, a family letter, a leaf with a poem, a leaf of the 1609 quarto, a ruling. Worked out here flat and in millimetres: its ink, its paper, the shape it has taken, and the face of the die that would press the same characters. film_docs3d.py lays them on the old table and photographs them.

    python film_docs.py table                         # the old table
    python film_docs.py write letter|poem|ruling      # maps/NAME_col.png, NAME_rough.png, NAME_relief.png, NAME.npz, NAME.json, and dies/NAME.npz
    python film_docs.py write tablet|quarto           # the same for 七击's first and fourth carriers: a clay tablet from Nineveh (its mesh solid, every wedge real) and a leaf of the 1609 quarto
    python film_docs.py shape letter|poem|ruling      # the sheet's shape alone
    python film_docs.py check                         # every text against what was checked, every character in its face

Everything goes to ink/film_out/film/imprint/. Three hands. The letter is small quick writing: a calligrapher's glyph gives only the structure, the stroke data's centre lines are laid along its ink and each takes from it how wide the brush pressed there, then film_big's hairs go along those lines and no outline is used. Each time a character is written its strokes are put down a little differently, so no two 之 are alike. The poem is a page of practised large writing in film_hand's way: the ink is worked out inside the glyph's outline, stroke by stroke, with the corners rounded, each stroke moved a little from the rest and the edge roughened by the paper. The ruling is set in metal type, and a font is what it should be.

A die in dies/ is a height in millimetres, read as the sheet is read: top row is the top of the page, and x runs to the right. So it records what the die leaves behind, the characters the right way round; the metal itself is this flipped left to right. See DIE_NOTE.
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d, label, map_coordinates

from film_big import Hand, _band
from film_hand import Glyph, THIN, _distance, _medians
from film_press_maps import fibres, spectral
from ink_grade import sm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "ink", "film_out", "film", "imprint")
MAPS, DIES = os.path.join(OUT, "maps"), os.path.join(OUT, "dies")
DIE_NOTE = "height in mm, as read on the sheet: row 0 is the top of the page, x runs right, so the characters read the right way round. This is what the die leaves in the material, seen from above. The metal face itself is this array flipped left to right."
FACE_RELIEF, CHAMFER = 1.2, 0.5    # how far a die's face stands, and how far out its shoulder goes on the way down, in mm

# 《與子儼等疏》, from 少學琴書 to 吾復何言, as checked against 维基文库
LETTER = "少學琴書偶愛閒靜開卷有得便欣然忘食見樹木交蔭時鳥變聲亦復歡然有喜常言五六月中北窗下臥遇涼風暫至自謂是羲皇上人意淺識罕謂斯言可保日月遂往機巧好疏緬求在昔眇然如何" \
         "疾患以來漸就衰損親舊不遺每以藥石見救自恐大分將有限也汝輩稚小家貧每役柴水之勞何時可免念之在心若何可言然汝等雖不同生當思四海皆兄弟之義鮑叔管仲分財無猜歸生伍舉班荊道舊遂能以敗為成因喪立功他人尚爾況同父之人哉潁川韓元長漢末名士身處卿佐七十而終兄弟同居至於沒齒濟北氾稚春晉時操行人也七世同財家人無怨色詩曰高山仰止景行行止雖不能爾至心尚之汝其慎哉吾復何言"
KEY = "念之在心若何可言"
POEM_TITLE = "春望"
POEM = ("國破山河在", "城春草木深", "感時花濺淚", "恨別鳥驚心", "烽火連三月", "家書抵萬金", "白頭搔更短", "渾欲不勝簪")
POEM_KEY = (2, 3)
# 平政院裁決書, as 中国民商法律网 quotes it. Nothing else may be set: no day, no names of the court's officers
RULING_TITLE = "平政院裁決書"
RULING_PARTIES = (("原告", "周樹人"), ("被告", "教育總長", "章士釗"))
RULING_ORDER = "教育部之處分取消之"
RULING_REASON = "被告停辦國立女師大學原告兼任該校教員是否確有反抗部令情事被告未能證明縱使屬實涉及文官懲戒條例規定範圍自應交付懲戒由該委員會依法議決處分方為合法被告遽行呈請免職確與現行規定程序不符至被告答辯內稱原擬循例交付懲戒其時形勢嚴重若不採用行政處分深恐群相效尤等語不知原告果有反抗部令嫌疑先行將原告停職或依法交付懲戒已足示儆何患群相效尤又何至迫不及待必須採取非常處分答辯各節並無理由據此論斷所有被告呈請免職之處分係屬違法應予取消茲據行政訴訟法第二十三條之規定裁決如主文"
RULING_DATE = "中華民國十五年三月"

XING = ("xing", (1, 3))      # Xingkai TC: a heavy master, and a light one for the lines to find
STAND_IN = {"潁": "穎"}      # the stroke data has no 潁; it is written like 穎 and only the water differs, and its lines are laid on 潁's own glyph


def srgb(*rgb):
    a = np.array(rgb, np.float32) / 255
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(a):
    a = np.clip(a, 0, 1)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055)


def save8(a, name):
    """A picture of values in 0-1, dithered so that a slow fade over a sheet does not step."""
    r = np.random.default_rng(1)
    im = Image.fromarray(np.uint8(np.clip(a * 255 + r.uniform(0, 1, a.shape), 0, 255)))
    im.save(os.path.join(MAPS, name), compress_level=3)
    print(name, im.size, flush=True)


def noise(h, w, beta, seed):
    return spectral(h, beta, seed, w)


def face(name, index, px):
    paths = [f"/System/Library/Fonts/Supplemental/{name}", os.path.expanduser(f"~/render_work/fonts/{name}")] + glob.glob(f"/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/{name}")
    return ImageFont.truetype(next(p for p in paths if os.path.exists(p)), int(round(px)), index=index)


# ---------------------------------------------------------------- the brush, along lines that come from a calligrapher's glyph

class Traced(Glyph):
    """A glyph with the stroke data's lines laid on it. Only its lines, and how far inside the heavy master each point of them lies, are ever used."""

    def __init__(self, c, kind, faces, n=320, em=240, tilt=0.0):
        self.c, self.frame, self.g, self.tilt = STAND_IN.get(c, c), n, 1, tilt
        self.n, self.em = n, em
        self.light, self.bold = _distance(c, faces[1], n, em, tilt, kind), _distance(c, faces[0], n, em, tilt, kind)
        self.mapped = False


_skeletons = {}
SQUARE = 320 / 240           # a square's side, in ems
WIDTH_K = 0.85               # what a mark's half width comes to, for a pressure of 1 and half a brush


def along_ink(g, rest, near, used, spacing=4.0, reach=44, grid=4):
    """One stroke's way over this glyph. The stroke data says what shape it has and how long it is; film_hand's fit says about where it lies, but has drawn it in to the thin master's middle, so an inner stroke comes out a dot. So: the data's own line, put at the fit's place, and each of its points free to move about on the ink, for what it costs to bend the line. Viterbi over the points' offsets. used: ink an earlier stroke has already been along, which costs a little, so that three strokes of a 口 take three sides of it."""
    p = rest - rest.mean(0) + near.mean(0)
    far = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    s = np.linspace(0, far[-1], max(int(far[-1] / spacing) + 1, 3))
    p = np.stack([np.interp(s, far, p[:, 0]), np.interp(s, far, p[:, 1])], 1)
    off = np.arange(-reach, reach + 1, grid, dtype=np.float32)
    ox, oy = [a.ravel() for a in np.meshgrid(off, off)]
    bend = 0.018 * ((ox[:, None] - ox[None]) ** 2 + (oy[:, None] - oy[None]) ** 2)
    thick = 0.035 * g.em
    n = g.n - 1
    cost = None
    back = []
    for q in p:
        i, j = np.clip(np.rint(q[1] + oy).astype(int), 0, n), np.clip(np.rint(q[0] + ox).astype(int), 0, n)
        d = g.bold[i, j]
        here = np.where(d > 0, 1 - np.minimum(d / thick, 1), 1.6 + 0.12 * -d) + used[i, j] + 0.00012 * (ox * ox + oy * oy)
        if cost is None:
            cost = here
        else:
            total = cost[None] + bend
            back.append(total.argmin(1))
            cost = total.min(1) + here
    k = int(cost.argmin())
    way = [k]
    for b in reversed(back):
        k = int(b[k])
        way.append(k)
    way = np.array(way[::-1])
    return gaussian_filter1d(p + np.column_stack([ox[way], oy[way]]), 0.8, axis=0, mode="nearest")


def skeleton(c, style):
    """c's strokes in writing order, each an array of x, y, and how far inside the heavy master, in a square a hundred wide."""
    if (c, style) not in _skeletons:
        g = Traced(c, *style)
        g._map()
        used = np.zeros((g.n, g.n), np.float32)
        out = []
        for rest, near in zip(_medians(g.c), g.lines):
            line = along_ink(g, rest * g.em / 100, near, used)
            far = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
            t = np.arange(0, far[-1] + 1, 1.0).clip(0, far[-1])
            fine = np.stack([np.interp(t, far, line[:, 0]), np.interp(t, far, line[:, 1])], 1)
            at = np.clip(np.rint(fine).astype(int), 0, g.n - 1)
            out.append(np.column_stack([fine * 100 / g.n, g.bold[at[:, 1], at[:, 0]] * 100 / g.n]))
            yy, xx = np.ogrid[0:g.n, 0:g.n]
            for x, y in fine[::4]:
                r = max(0.03 * g.em, 4.0)
                y0, y1, x0, x1 = int(max(y - r, 0)), int(min(y + r + 1, g.n)), int(max(x - r, 0)), int(min(x + r + 1, g.n))
                used[y0:y1, x0:x1] = 0.4
        _skeletons[c, style] = out
    return _skeletons[c, style]


def marks(skel, r, body, weight, floor=0.3, ink=1.0, dry=1.0, quick=0.3, loose=1.0, runs=0.012, link=0.0):
    """The strokes as this hand puts them down this time: the whole a little turned, sheared, wider or narrower, each stroke a little moved and turned about its own middle and pressed harder or lighter. Returned as film_big's table: path of (x, y, pressure, how fast), ink, how soon it runs out. floor: the lightest a stroke goes on its way, so that it stays one stroke. loose: how far from the glyph. link: how often the tip stays down on its way to the next stroke, when that is near, and leaves a hair of ink."""
    a, shear = r.normal(0, 0.03 * loose), r.normal(0, 0.05 * loose)
    sx, sy = np.exp(r.normal(0, 0.04 * loose, 2))
    whole = np.array([[np.cos(a) * sx, -np.sin(a) * sy + shear], [np.sin(a) * sx, np.cos(a) * sy]])
    out = []
    for k, s in enumerate(skel):
        p = s[:, :2] - 50
        mid = p.mean(0)
        t = r.normal(0, 0.035 * loose)
        p = (p - mid) @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]) + mid + r.normal(0, 0.8 * loose, 2)
        p = p @ whole.T + 50
        half = gaussian_filter1d(np.maximum(s[:, 2] - THIN * (1 - weight) * 75, 0), 3.0, mode="nearest") * np.exp(r.normal(0, 0.1 * loose))
        far = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
        if far[-1] < 1.5:
            # a dot the lines have closed up on
            u = (p[-1] - p[0]) / far[-1] if far[-1] > 0.01 else np.array((0.6, 0.8))
            c = p.mean(0)
            pts = [(*(c - u), 0.2, 0.0), (*c, 0.5, 0.0), (*(c + u), 0.1, 0.0)]
        else:
            at = np.unique(np.rint(np.linspace(0, len(p) - 1, max(int(far[-1] / 2.5) + 1, 4))).astype(int))
            P = np.clip(half[at] / (body * 0.5) / WIDTH_K, 0, 1.2)
            P[1:-1] = np.maximum(P[1:-1], floor)
            P[0], P[-1] = min(P[0], 0.14), min(P[-1], 0.05)
            v = quick * np.clip((far[at] / far[-1] - 0.55) / 0.45, 0, 1)
            pts = [(p[i, 0], p[i, 1], P[j], v[j]) for j, i in enumerate(at)]
        if out and link and r.uniform() < link:
            a, b = np.array(last[-1][:2]), np.array(pts[0][:2])
            gap = np.hypot(*(b - a))
            if 3 < gap < 20:
                side = np.array((-(b - a)[1], (b - a)[0])) * r.uniform(-0.18, 0.18)
                m = (a + b) / 2 + side
                out.append(([(*a, 0.13, 0.5), (*((a + m) / 2), 0.1, 0.7), (*m, 0.09, 0.7), (*((m + b) / 2), 0.1, 0.6), (*b, 0.14, 0.3)], ink - runs * k, 1.3))
        last = pts
        out.append((pts, ink - runs * k, dry))
    return out


def extent(ms):
    """Where a character's lines go in its square: left, top, right, bottom."""
    p = np.array([q[:2] for path, *_ in ms for q in path])
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def over(A, ink, x, y, side, px):
    """A square of ink, side mm wide with its corner at x, y mm, into A."""
    n = int(round(side * px))
    a = np.asarray(Image.fromarray(np.float32(np.clip(ink, 0, 1)), "F").resize((n, n), Image.LANCZOS), np.float32).clip(0, 1)
    l, t = int(round(x * px)), int(round(y * px))
    y0, y1, x0, x1 = max(t, 0), min(t + n, A.shape[0]), max(l, 0), min(l + n, A.shape[1])
    b = A[y0:y1, x0:x1]
    A[y0:y1, x0:x1] = b + (1 - b) * a[y0 - t:y1 - t, x0 - l:x1 - l]


class Pen:
    """One brush for all of a sheet. Each character turns the paper's grain under it a different way, so its hairs catch differently."""

    def __init__(self, size, body, flat=0.42, rough=0.6, seed=5):
        self.hand = Hand(size, seed=seed, body=body, flat=flat, frame=size // 2, rough=rough)
        self.tooth, self.warp, self.fibre = self.hand.tooth, self.hand.warp, self.hand.fibre
        self.r = np.random.default_rng(seed)

    def write(self, ms, seed, tone=0.97, soak=1.0, each=None):
        h, s = self.hand, self.tooth.shape[0]
        i, j = self.r.integers(0, s, 2)
        h.brush.seed = seed
        h.tooth, h.fibre = np.roll(self.tooth, (i, j), (0, 1)), np.roll(self.fibre, (j, i), (0, 1))
        return h.design(ms, tone=tone, soak=soak, each=each)


def solid(ms):
    """The same strokes as if the brush never ran out: the shape of the character, for a die."""
    return [(path, 1.5, 0.0, *rest) for path, _, _, *rest in ms]


def die_face(mask, px):
    """A die's height over a mask of where it prints, made as type is made: the whole printing face at one height, however thin a stroke, and a shoulder sloping down and away from it."""
    return np.float32(np.clip(1 - distance_transform_edt(mask <= 0.5) / px / CHAMFER, 0, 1) * FACE_RELIEF)


def stopped(mask, px, most=1.2):
    """A brush's shape with the small rings a loaded tip leaves in a dot closed up, smaller than most square mm: nobody would cut those in metal."""
    holes, n = label(mask <= 0.5)
    sizes = np.bincount(holes.ravel(), minlength=n + 1)
    small = sizes < most * px * px
    small[0] = False
    return np.where(small[holes], 1.0, mask)


def save_die(name, height, px, **extra):
    os.makedirs(DIES, exist_ok=True)
    np.savez_compressed(os.path.join(DIES, name + ".npz"), height=np.float32(height), mm_per_px=1.0 / px, note=DIE_NOTE, **extra)
    print("die", name, height.shape, "%.3f mm a pixel" % (1.0 / px), flush=True)


# ---------------------------------------------------------------- what paper has in common

def page_grid(w, h, step, wander, seed):
    """A grid over a page, rows from its foot up. Its boundary is the sheet's real edge: wander moves the outermost points about, and the movement dies away inwards. Returns U, V, the page's own coordinates in mm with V up, and X, Y where those points really are."""
    us, vs = np.linspace(0, w, int(w / step) + 1, dtype=np.float32), np.linspace(0, h, int(h / step) + 1, dtype=np.float32)
    U, V = np.meshgrid(us, vs)
    if not wander:
        return U, V, U, V
    r = np.random.default_rng(seed)

    def rim(n):
        a = gaussian_filter1d(r.normal(0, 1, n), 40 / step, mode="wrap")
        b = gaussian_filter1d(r.normal(0, 1, n), 5 / step, mode="wrap")
        c = gaussian_filter1d(r.normal(0, 1, n), 0.6 / step, mode="wrap")
        return np.float32(a / a.std() * 0.7 + b / b.std() * 0.3 + c / c.std() * 0.1) * wander
    die = lambda e: np.exp(-e / 3.0)
    X = U + rim(len(vs))[:, None] * die(U) + rim(len(vs))[:, None] * die(w - U)
    Y = V + rim(len(us))[None, :] * die(V) + rim(len(us))[None, :] * die(h - V)
    return U, V, X, Y


def stretch(tile, h, w):
    return np.asarray(Image.fromarray(tile).resize((w, h), Image.BICUBIC), np.float32)


def bumps(U, V, w, h, across, amp, seed):
    """Smooth lumps over the sheet, about across mm wide and amp mm high. Under a low light even a lump of a slope of 1 in 30 shows: paper is kept smooth here."""
    step = max(across / 6, 0.5)
    n, m = int(h / step) + 8, int(w / step) + 8
    t = gaussian_filter(np.random.default_rng(seed).normal(0, 1, (n, m)).astype(np.float32), across / 2.4 / step, mode="wrap")
    return map_coordinates(t / t.std(), [V / step + 4, U / step + 4], order=3, mode="wrap") * amp


def along(n, sigma, seed):
    v = gaussian_filter1d(np.random.default_rng(seed).normal(0, 1, n), sigma, mode="wrap")
    return np.float32(v / v.std())


def hinge(d, w):
    """0 on one side of a fold, d on the other, its bend w wide."""
    return w * np.logaddexp(0, np.clip(d / w, -40, 40))


def edges(U, V, w, h):
    return U, w - U, V, h - V


def curl(U, V, w, h, amps, lengths, seed):
    """Each of a page's edges lifting, more in some places than others, and most at its corners. amps: how far up, lengths: over how far in, for left, right, foot, head."""
    r = np.random.default_rng(seed)
    z = np.zeros_like(U)
    for k, (e, a, L) in enumerate(zip(edges(U, V, w, h), amps, lengths)):
        s = (V if k < 2 else U)
        side = h if k < 2 else w
        wave = along(512, 40, seed + k)
        vary = 0.8 * np.exp(0.5 * np.interp(s / side * 511, np.arange(512), wave))
        z += a * vary * np.exp(-e / L)
    return z


def edge_shade(u_px, v_px, w, h, reach, seed):
    """How far each pixel of a map is from the page's edge, worn a little unevenly: 1 at the edge, going to 0 inwards."""
    e = np.minimum.reduce(edges(u_px, v_px, w, h))
    n = noise(*u_px.shape, 2.2, seed)
    return np.exp(-np.maximum(e + 0.25 * reach * n, 0) / reach)


# ---------------------------------------------------------------- the letter: an old hand copy, folded in three

LETTER_W, LETTER_H, LETTER_PX = 262.0, 340.0, 16
LETTER_EM = 14.5
LETTER_PITCH = 19.0
LETTER_BODY = 7.6
LETTER_TOP, LETTER_FOOT = 26.0, 22.0
LETTER_STYLE = XING
LETTER_FOLDS = (106.0, 233.0)                       # from the head of the page: folded by hand, not quite in thirds
LETTER_MID = sum(LETTER_FOLDS) / 2


def letter_plan():
    """Where every character of the letter goes, columns from the right. A hand does not rule its columns: each leans and wanders a little, and each character is as tall as it happens to be, with what room is left shared between them. The key run is on one column, in its middle, between the folds."""
    r = np.random.default_rng(23)
    chars = []
    # every character as this hand writes it this time, with the brush dipped now and then
    load, until = 1.14, 0
    body, weight = LETTER_BODY, 0.48
    for i, c in enumerate(LETTER):
        if i >= until:
            load, until = float(r.uniform(1.08, 1.17)), i + int(r.uniform(11, 26))
        ms = marks(skeleton(c, LETTER_STYLE), r, body, weight, floor=0.32, ink=load, dry=1.0 + (1.14 - load) * 2.4, quick=0.34, loose=1.0, link=0.32)
        load -= float(r.uniform(0.008, 0.02))
        n = len(ms)
        em = LETTER_EM * float(np.exp(r.normal(0, 0.055))) * (1 + 0.006 * np.clip(n - 9, -6, 8))
        x0, y0, x1, y1 = extent(ms)
        side = em * SQUARE
        chars.append(dict(c=c, i=i, marks=ms, side=side, em=em, ink=load, top=y0 / 100 * side, high=(y1 - y0) / 100 * side, mid=(x0 + x1) / 200 * side, seed=int(r.integers(1, 1 << 30))))
    start = LETTER.index(KEY)
    room = LETTER_H - LETTER_TOP - LETTER_FOOT
    gap0 = 1.9
    # as many characters above the run on its own column as bring that column's head to where the others start
    run_length = sum(c["high"] + gap0 for c in chars[start:start + len(KEY)]) - gap0
    above = np.cumsum([c["high"] + gap0 for c in chars[start - 1::-1]])
    before = 1 + int(np.argmin(np.abs(above - (LETTER_MID - run_length / 2 - LETTER_TOP))))
    # the columns before the run's, as many as fit, breaking where the height says
    head = chars[:start - before]
    total = sum(c["high"] + gap0 for c in head)
    m = max(int(round(total / room)), 1)
    cum = np.cumsum([c["high"] + gap0 for c in head])
    cuts = [int(np.searchsorted(cum, total * (j + 1) / m)) + int(r.integers(-1, 2)) for j in range(m - 1)]
    cols = [head[a:b] for a, b in zip([0] + cuts, cuts + [len(head)])]
    # and from there on, as they come
    col, used = [], 0.0
    rest = chars[start - before:]
    key_col = len(cols)
    for j, c in enumerate(rest):
        if col and used + c["high"] > room:
            cols.append(col)
            col, used = [], 0.0
        col.append(c)
        used += c["high"] + gap0
    cols.append(col)
    # now on the page
    x = LETTER_W - 22.0
    run_y = None
    for k, col in enumerate(cols):
        last = k == len(cols) - 1
        height = sum(c["high"] for c in col)
        target = room * float(r.uniform(0.96, 1.0))
        gap = gap0 * 1.05 if last else float(np.clip((target - height) / max(len(col) - 1, 1), 0.7, 4.2))
        gaps = gap * r.uniform(0.6, 1.4, len(col))
        lean, bow = float(r.normal(0, 0.012)), float(r.normal(0, 1.0))
        y = LETTER_TOP + float(r.normal(0, 1.8))
        if k == key_col:
            # the run goes in the middle of the page, between the folds, and its column is written with some care
            lean, bow = lean * 0.4, bow * 0.4
            ahead = sum(c["high"] + g for c, g in zip(col[:before], gaps[:before]))
            length = sum(c["high"] + g for c, g in zip(col[before:before + len(KEY)], gaps[before:before + len(KEY)])) - gaps[before + len(KEY) - 1]
            y = LETTER_MID - length / 2 - ahead
        for j, (c, g) in enumerate(zip(col, gaps)):
            u = (y - LETTER_H / 2) / LETTER_H
            cx = x + lean * (y - LETTER_H / 2) + bow * (1 - 4 * u * u) + float(r.normal(0, 0.45))
            c.update(col=k, x=cx - c["mid"], y=y - c["top"], box=(cx - c["mid"] * 1.0, y, c["high"]))
            y += c["high"] + g
        x -= LETTER_PITCH * float(np.exp(r.normal(0, 0.03)))
    run = [c for c in chars if start <= c["i"] < start + len(KEY)]
    assert len({c["col"] for c in run}) == 1 and "".join(c["c"] for c in run) == KEY
    assert LETTER_FOLDS[0] + 1 < run[0]["y"] + run[0]["top"] and run[-1]["y"] + run[-1]["top"] + run[-1]["high"] < LETTER_FOLDS[1] - 1, (run[0]["y"], run[-1]["y"])
    assert x > 4 and all(c["y"] + c["top"] + c["high"] < LETTER_H - 6 for c in chars), (x, max(c["y"] + c["top"] + c["high"] for c in chars))
    print(len(cols), "columns of", [len(col) for col in cols], "the run on column", key_col, flush=True)
    return chars, run


def letter_shape():
    """Folded in three with the writing inside, so both ends of the page still rise from their folds. The folds were made by hand and do not quite run true. The middle panel lies on the table."""
    w, h = LETTER_W, LETTER_H
    U, V, X, Y = page_grid(w, h, 0.25, 0.55, 31)
    n = U.shape[1]
    wob = [np.interp(U, np.linspace(0, w, 512), gaussian_filter1d(np.random.default_rng(40 + k).normal(0, 1, 512), 30, mode="wrap") * 12) + (U - w / 2) * s for k, s in enumerate((0.006, -0.004))]
    lo, hi = h - LETTER_FOLDS[1] + wob[0], h - LETTER_FOLDS[0] + wob[1]     # rows go from the foot up
    tilt = lambda seed, base: np.tan(np.radians(base)) * (1 + 0.28 * np.interp(U, np.linspace(0, w, 512), along(512, 60, seed)))
    z = tilt(50, 9.0) * 34 * (1 - np.exp(-hinge(lo - V, 0.3) / 34)) + tilt(51, 7.0) * 42 * (1 - np.exp(-hinge(V - hi, 0.3) / 42))
    # the folds themselves, pressed in, the paper a little broken along them
    for k, f in enumerate((lo, hi)):
        d = V - f
        z += -0.09 * np.exp(-(d / 0.3) ** 2) + 0.03 * np.interp(U, np.linspace(0, w, 2048), along(2048, 30, 60 + k)) * np.exp(-(d / 1.6) ** 2)
    z += bumps(U, V, w, h, 90, 0.55, 33) + bumps(U, V, w, h, 16, 0.035, 34) + curl(U, V, w, h, (1.0, 1.7, 1.5, 2.2), (9, 9, 8, 10), 35)
    # a corner that was bent once
    c = (w - U) + V - 26.0
    z += 0.22 * hinge(-c, 0.8) * (hinge(-c, 0.8) < 26)
    return X, Y, z, (lo, hi)


def letter_paper(A, px):
    """Old thin paper: not much yellowed, browner at its edges, a dirty line in each fold, an old tide mark, a little foxing."""
    h, w = A.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (xx + 0.5) / px, LETTER_H - (yy + 0.5) / px
    col = np.broadcast_to(srgb(228, 219, 198), (h, w, 3)).copy()
    long_ = stretch(fibres(4096, 26000, 71, length=(40, 220), width=(1, 1), bright=(.3, 1.0), wiggle=.3), h, w)
    dark = stretch(fibres(4096, 2600, 72, length=(30, 160), width=(1, 2), bright=(.3, 1.0), wiggle=.3), h, w)
    tone = 1 + 0.03 * noise(h, w, 2.6, 73) + 0.012 * noise(h, w, 1.0, 74) + 0.05 * gaussian_filter(long_, 0.7) - 0.05 * gaussian_filter(dark, 0.6)
    col = col * tone[..., None]
    rim = edge_shade(u, v, LETTER_W, LETTER_H, 7.0, 75)
    col = col * (1 - rim[..., None] * np.array((0.17, 0.21, 0.30), np.float32))
    # the folds: dust in them, and the top of the paper rubbed pale
    _, _, _, (lo, hi) = letter_shape()
    cols = np.linspace(0, LETTER_W, lo.shape[1])
    for f, seed in ((lo[0], 76), (hi[0], 77)):
        d = v - np.interp(u, cols, f)
        n = noise(h, w, 1.8, seed)
        col = col * (1 - (0.12 * np.exp(-(d / 2.2) ** 2) * (0.7 + 0.5 * n))[..., None] * np.array((0.9, 1.0, 1.25), np.float32))
        col = col + (srgb(238, 230, 212) - col) * (0.5 * np.exp(-(d / 0.16) ** 2) * sm((n + 0.4) / 0.6))[..., None]
        A = A * (1 - 0.55 * np.exp(-(d / 0.3) ** 2) * sm((noise(h, w, 1.2, seed + 9) + 0.6) / 0.9))
    # an old tide mark, and where it has been handled
    warp = noise(h, w, 2.6, 78) * 16 + noise(h, w, 1.8, 79) * 2.0
    d = np.hypot(u - 46, v - 58) + warp - 44
    tide = np.exp(-np.maximum(d, -0.5) ** 2 / 2.2) * (d > -1.4) * 0.5 + 0.1 * sm(-d / 18)
    col = col * (1 - (tide * 0.32)[..., None] * np.array((0.55, 0.8, 1.25), np.float32))
    thumb = np.exp(-(((u - 228) / 34) ** 2 + ((v - 30) / 26) ** 2)) * sm((noise(h, w, 2.0, 80) + 1) / 2)
    col = col * (1 - (thumb * 0.13)[..., None] * np.array((0.8, 0.95, 1.2), np.float32))
    r = np.random.default_rng(81)
    spots = np.zeros((h, w), np.float32)
    for x, y, s, a in zip(r.uniform(0, LETTER_W, 90), r.uniform(0, LETTER_H, 90), r.uniform(0.2, 1.1, 90), r.uniform(0.15, 0.6, 90)):
        i, j, m = int((LETTER_H - y) * px), int(x * px), int(s * px * 3) + 2
        a0, a1, b0, b1 = max(i - m, 0), min(i + m, h), max(j - m, 0), min(j + m, w)
        yy2, xx2 = np.ogrid[a0:a1, b0:b1]
        spots[a0:a1, b0:b1] = np.maximum(spots[a0:a1, b0:b1], a * np.exp(-(((yy2 - i) ** 2 + (xx2 - j) ** 2) / (s * px) ** 2)))
    col = col * (1 - (spots * 0.5)[..., None] * np.array((0.45, 0.75, 1.15), np.float32))
    # the ink: it ran a little way along the fibres when it was wet, and has gone from black to something browner where it was thin
    A = 1 - (1 - A) ** 1.5
    feather = gaussian_filter(A, 0.24 * px) * sm((gaussian_filter(long_, 1.2) * 2.4 + 0.45 * noise(h, w, 1.0, 82) + 0.1))
    a = np.clip(A + (1 - A) * 0.55 * feather, 0, 1)
    ink = srgb(24, 20, 19)[None, None] + (srgb(62, 50, 42)[None, None] - srgb(24, 20, 19)[None, None]) * (1 - A)[..., None]
    col = col + (ink - col) * a[..., None]
    rough = 0.84 - 0.1 * a
    return col, rough, np.zeros_like(a), a


def write_letter():
    chars, run = letter_plan()
    px = LETTER_PX
    h, w = int(LETTER_H * px), int(LETTER_W * px)
    A, S = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    pen = Pen(640, LETTER_BODY, seed=5)
    for c in chars:
        over(A, pen.write(c["marks"], c["seed"], tone=0.97 - 0.1 * max(1.1 - c["ink"], 0)), c["x"], c["y"], c["side"], px)
        over(S, pen.write(solid(c["marks"]), c["seed"]), c["x"], c["y"], c["side"], px)
    print("written", flush=True)
    col, rough, relief, a = letter_paper(A, px)
    finish("letter", col, rough, relief, letter_shape()[:3], LETTER_W, LETTER_H, px, chars, dict(run=[box_of(c) for c in run]), stopped(S, px), folds=LETTER_FOLDS)


def box_of(c):
    """A character's ink, near enough, in mm from the page's left and top: x0, y0, x1, y1."""
    if "box" in c and "marks" not in c:
        return c["box"]
    x0, y0, x1, y1 = extent(c["marks"])
    k = c["side"] / 100
    return (c["x"] + x0 * k, c["y"] + y0 * k, c["x"] + x1 * k, c["y"] + y1 * k)


def save_shape(name, X, Y, Z):
    Z = np.maximum(Z - np.percentile(Z, 0.5) + 0.12, 0.02)
    np.savez(os.path.join(MAPS, name + ".npz"), X=np.float32(X), Y=np.float32(Y), Z=np.float32(Z))
    return Z


def finish(name, col, rough, relief, shape, w, h, px, chars, keys, face_mask, **extra):
    """Everything film_docs3d.py and the press need to know about a sheet."""
    os.makedirs(MAPS, exist_ok=True)
    save8(to_srgb(col), name + "_col.png")
    save8(np.clip(rough, 0, 1), name + "_rough.png")
    save8(np.clip(relief, 0, 1), name + "_relief.png")
    Z = save_shape(name, *shape)
    listing = [dict(c=c["c"], col=c.get("col", 0), box=[round(v, 2) for v in box_of(c)]) for c in chars]
    with open(os.path.join(MAPS, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(w=w, h=h, px=px, keys=keys, chars=listing, **extra), f, ensure_ascii=False)
    save_die(name, die_face(face_mask, px), px, w_mm=w, h_mm=h, keys=json.dumps(keys), chars=json.dumps(listing, ensure_ascii=False))
    print("mesh", Z.shape, "z up to %.1f mm" % float(Z.max()), flush=True)


# ---------------------------------------------------------------- the poem: a leaf of good paper

POEM_W, POEM_H, POEM_PX = 470.0, 300.0, 12
POEM_EM = 37.0
POEM_PITCH = 46.0
POEM_TOP = 34.0
POEM_WEIGHT = 0.56
POEM_FEET = (1.24, 0.8, 1.14, 0.86, 1.0)        # 二三: how each of a line's five is pressed
POEM_HEAVIER = dict(驚=1.3)                      # this one comes apart if it is written as light as its strokes would have it
# and written as one, as heavy as those on either side of it. The upper weight above fixed its place on the page, and that stays
POEM_JOINED = dict(驚=dict(shape=dict(split=0.529, lift=0.04, wide=1.16, link=((0.404, 0.478, 0.398, 0.535, 0.022),), room=0.022, dense=(0.52, 0.55, 0.11, 0.075, 0.012), gaps=((0.527, 0.528, 0.565, 0.512, 0.007), (0.527, 0.558, 0.565, 0.54, 0.007), (0.48, 0.545, 0.497, 0.535, 0.005), (0.478, 0.612, 0.493, 0.603, 0.005))), weight=0.8, u=0.57, steady=0.5))
POEM_DIPS = (0.0, 0.0, 0.38, 0.0, 0.12, 0.0, 0.4, 0.0, 0.4)     # how far from its last dip the brush is at each column's head


def joined(g, split, lift, wide, link, room, dense, gaps):
    """A character whose upper and lower parts the typeface leaves as two, brought together before anything is written, on both masters so the stroke lines follow. The lower part comes up by lift at its head and not at all at its foot, and widens to sit under the upper; link: where the brush runs on from the one into the other, in parts of the square, and how wide. All of it stays inside the box the character was given, so nothing round it moves."""
    n = g.n
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    cut, up, mid = split * n, lift * n, 0.505 * n
    foot = float(np.nonzero((g.bold > 0).any(1))[0][-1])
    source_y = cut + (yy - (cut - up)) * (foot - cut) / (foot - cut + up)
    source_x = mid + (xx - mid) / wide
    for name, thinner in (("light", THIN * g.em), ("bold", 0.0)):
        d = getattr(g, name)
        lower = map_coordinates(np.minimum(d, yy - cut), [source_y, source_x], order=1, mode="nearest")
        out = np.maximum(np.minimum(d, cut - yy), lower)
        for x0, y0, x1, y1, half in link:
            a, b = np.array((x0, y0), np.float32) * n, np.array((x1, y1), np.float32) * n
            t = np.clip(((xx - a[0]) * (b - a)[0] + (yy - a[1]) * (b - a)[1]) / np.sum((b - a) ** 2), 0, 1)
            out = np.maximum(out, half * g.em - thinner - np.hypot(xx - a[0] - t * (b - a)[0], yy - a[1] - t * (b - a)[1]))
        setattr(g, name, np.float32(gaussian_filter(out, 1.0)))
    # the heavy master has this many strokes run into one mass: it goes only so far out from the light one's lines, and what is between them stays paper
    x, y, rx, ry, there = dense
    close = np.exp(-(((xx / n - x) / rx) ** 2 + ((yy / n - y) / ry) ** 2))
    g.bold = np.minimum(g.bold, g.light + (room + (there - room) * close) * g.em)
    # and where three bars lie closest, the paper between them left as a brush lifting would leave it, as 鳥's are
    for x0, y0, x1, y1, half in gaps:
        a, b = np.array((x0, y0), np.float32) * n, np.array((x1, y1), np.float32) * n
        t = np.clip(((xx - a[0]) * (b - a)[0] + (yy - a[1]) * (b - a)[1]) / np.sum((b - a) ** 2), 0, 1)
        g.bold = np.minimum(g.bold, np.hypot(xx - a[0] - t * (b - a)[0], yy - a[1] - t * (b - a)[1]) - half * g.em)


def full(g, tooth, weight, u, seed, steady=1.0, keep=None):
    """A character of the poem: film_hand's ink over the glyph, then taken further from the typeface. Its corners rounded, which no brush can cut so sharp; each stroke moved a little from where the others have it, the whole slowly pulled about, and the edge roughened by the paper's grain. Returns how dark, and the plain shape of it, moved together."""
    n, em = g.n, g.em
    g.bold = gaussian_filter(g.bold, 0.008 * em)
    if keep:        # only its first strokes so far: the later ones were never there, so where they would have left, the edge is rounded as a brush leaves it. keep: how many strokes, or what says where the ink is
        g._map()
        shape = gaussian_filter(np.float32(keep(g) if callable(keep) else g.K < keep), 0.014 * em) > 0.5
        edge = np.float32(distance_transform_edt(shape) - distance_transform_edt(~shape))
        g.bold = edge if callable(keep) else np.minimum(g.bold, edge)
    F = g.written(tooth, weight=weight, ink=1.1 - 0.16 * u, dry=0.01 + 0.5 * u ** 1.6, tone=0.985 - 0.05 * u, seed=seed % 100000, most=0.8)
    M = sm(g.bold - THIN * (1 - weight) * em + 0.5)
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    move = []
    for axis in range(2):
        strokes = r.normal(0, 0.013 * em * min(np.sqrt(8 / g.count), 1) * steady, g.count + 1)
        v = gaussian_filter(np.float32(strokes[g.K]), 0.012 * em)
        slow = gaussian_filter(r.normal(0, 1, (n // 4, n // 4)).astype(np.float32), 0.075 * em, mode="reflect")
        v = v + np.asarray(Image.fromarray(slow / slow.std()).resize((n, n), Image.BICUBIC), np.float32) * 0.012 * em * steady
        fine = gaussian_filter(np.roll(tooth, (axis * 700, axis * 400), (0, 1)), 0.6)
        move.append(v + fine / fine.std() * 0.45)
    shift = lambda v: map_coordinates(v, [yy + move[0], xx + move[1]], order=1, mode="nearest")
    g.shift = shift
    return shift(F), shift(M)


def poem_plan():
    """Eight lines of five, a line to a column, and its title to the right of them, smaller. A hand that has practised, in film_hand's way: every stroke its own weight, pressed at its head, lifting and drying as it goes. Dipped at the head of every other column; it has run nearly dry by the foot of the next."""
    r = np.random.default_rng(29)
    cols = [(POEM_TITLE, 0.62, 1.0, 0.0)] + [(line, 1.0, 1.0, 0.0) for line in POEM]
    chars = []
    x = POEM_W - 36.0
    for k, (line, scale, _, _) in enumerate(cols):
        written = []
        for j, c in enumerate(line):
            em = POEM_EM * scale * float(np.exp(r.normal(0, 0.03))) * (1 + 0.004 * (len(_medians(c)) - 8))
            feet = 1.0 if k == 0 else POEM_FEET[j]
            weight = min(POEM_WEIGHT * feet * float(np.exp(r.normal(0, 0.07))), 0.82) * (1 - 0.012 * max(len(_medians(c)) - 10, 0))
            weight = min(weight * POEM_HEAVIER.get(c, 1.0), 0.82)
            n = int(em * SQUARE * POEM_PX)
            g = Traced(c, *XING, n=n, em=em * POEM_PX, tilt=float(np.clip(r.normal(0, 1.2), -2.4, 2.4)))
            box = g.box(weight)
            written.append(dict(c=c, col=k, g=g, weight=float(weight), em=em, side=n / POEM_PX, l=box[0] / POEM_PX, t=box[1] / POEM_PX, r=box[2] / POEM_PX, b=box[3] / POEM_PX, seed=int(r.integers(1, 1 << 30))))
        room = POEM_H - 2 * POEM_TOP
        high = sum(c["b"] - c["t"] for c in written)
        gap = 7.0 if k == 0 else (room - high) / (len(line) - 1)
        y = POEM_TOP + (6.0 if k == 0 else float(r.normal(0, 1.0)))
        for j, c in enumerate(written):
            u = POEM_DIPS[k] + 0.6 * j / (len(line) - 1) if k else 0.1      # how far from the last dip
            cx = x + float(r.normal(0, 0.9))
            c.update(u=u, x=cx - (c["l"] + c["r"]) / 2, y=y - c["t"], box=(cx - (c["r"] - c["l"]) / 2, y, cx + (c["r"] - c["l"]) / 2, y + c["b"] - c["t"]))
            y += c["b"] - c["t"] + gap * float(r.uniform(0.86, 1.14))
            chars.append(c)
        x -= POEM_PITCH * (0.9 if k == 0 else float(np.exp(r.normal(0, 0.012))))
    assert x > 0, x
    return chars


def poem_shape():
    """Good paper, kept flat for a long time and then rolled once: two ends lift, and it has a slow swell in it."""
    w, h = POEM_W, POEM_H
    U, V, X, Y = page_grid(w, h, 0.4, 0.7, 131)
    z = bumps(U, V, w, h, 130, 0.7, 133) + bumps(U, V, w, h, 22, 0.03, 134) + curl(U, V, w, h, (3.6, 7.0, 0.8, 1.1), (42, 40, 10, 10), 135)
    return X, Y, z


def poem_paper(A, px):
    h, w = A.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (xx + 0.5) / px, POEM_H - (yy + 0.5) / px
    col = np.broadcast_to(srgb(239, 233, 218), (h, w, 3)).copy()
    # 宣: its long fibres show, a few darker bits in it, and the faint lines of the screen it was lifted on
    long_ = stretch(fibres(4096, 9000, 171, length=(120, 700), width=(2, 4), bright=(.3, 1.0), wiggle=.2), h, w)
    bits = stretch(fibres(4096, 800, 172, length=(20, 110), width=(1, 2), bright=(.4, 1.0), wiggle=.35), h, w)
    laid = 0.5 + 0.5 * np.sin(2 * np.pi * (yy / px + noise(h, w, 3, 173) * 0.5) / 1.4)
    tone = 1 + 0.018 * noise(h, w, 2.8, 174) + 0.008 * noise(h, w, 1.0, 175) + 0.05 * gaussian_filter(long_, 0.8) - 0.1 * gaussian_filter(bits, 0.5) + 0.006 * laid
    col = col * tone[..., None]
    rim = edge_shade(u, v, POEM_W, POEM_H, 5.0, 176)
    col = col * (1 - rim[..., None] * np.array((0.08, 0.1, 0.15), np.float32))
    # good ink, wet: it stays where it is put, a fine rim of it into the fibres
    feather = gaussian_filter(A, 0.22 * px) * sm(gaussian_filter(long_, 0.8) * 2.2 + 0.3 * noise(h, w, 1.0, 177))
    a = np.clip(A + (A < 0.04) * 0.3 * feather, 0, 1)
    ink = srgb(16, 15, 17)[None, None] + (srgb(52, 50, 54)[None, None] - srgb(16, 15, 17)[None, None]) * (1 - A)[..., None]
    a = 1 - (1 - a) ** 1.7
    col = col + (ink - col) * a[..., None]
    return col, 0.86 - 0.26 * a, np.zeros_like(a), a


def write_poem():
    chars = poem_plan()
    px = POEM_PX
    h, w = int(POEM_H * px), int(POEM_W * px)
    A, S = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    grain = _band(2048, 1.0, 4.5, 301) * 0.7 + _band(2048, 3.0, 12.0, 302) * 0.5
    for c in chars:
        g, u = c["g"], c["u"]
        tooth = np.roll(grain, (c["seed"] % 2048, (c["seed"] >> 11) % 2048), (0, 1))[:g.n, :g.n]
        once = POEM_JOINED.get(c["c"])
        if once:
            joined(g, **once["shape"])
        F, M = full(g, tooth, once["weight"] if once else c["weight"], once["u"] if once else u, c["seed"], once["steady"] if once else 1.0)
        over(A, F, c["x"], c["y"], c["side"], px)
        over(S, M, c["x"], c["y"], c["side"], px)
        c["g"] = None
        print(c["c"], end=" ", flush=True)
    print(flush=True)
    col, rough, relief, a = poem_paper(A, px)
    keys = dict(lines=[[c["box"] for c in chars if c["col"] == k + 1] for k in POEM_KEY])
    finish("poem", col, rough, relief, poem_shape(), POEM_W, POEM_H, px, chars, keys, S)


# ---------------------------------------------------------------- the ruling: machine paper, a red-ruled form, metal type

RULING_W, RULING_H, RULING_PX = 300.0, 235.0, 16
FRAME = (18.0, 282.0, 22.0, 213.0)        # left, right, top, bottom of the printed form
RULING_COLS = 24
TYPE_EM, TYPE_STEP = 8.9, 9.4
SONG, SONG_BOLD = ("Songti.ttc", 7), ("Songti.ttc", 2)      # Songti TC


_cast = {}


def cast(c, f, em, px):
    """A piece of type's face, large enough to wear."""
    if (c, f, em) not in _cast:
        k = 3
        n = int(em * px * 1.3 * k)
        im = Image.new("L", (n, n), 0)
        ImageDraw.Draw(im).text((n / 2, n / 2), c, 255, font=face(f[0], f[1], em * px * k), anchor="mm")
        _cast[c, f, em] = np.asarray(im.resize((n // k, n // k), Image.BOX), np.float32) / 255
    return _cast[c, f, em]


def set_type(A, R, S, c, cx, cy, f, em, r, px):
    """One sort pressed into the sheet: not quite straight, not quite where it should be, a little worn, its ink pushed to its edges, and it sinks into the paper."""
    a = cast(c, f, em, px)
    a = np.asarray(Image.fromarray(a).rotate(float(r.normal(0, 0.14)), Image.BICUBIC), np.float32)
    n = a.shape[0]
    wear = gaussian_filter(r.normal(0, 1, (n, n)).astype(np.float32), 2.0)
    wear /= wear.std()
    worn = sm((gaussian_filter(a, 0.03 * px) - 0.4 - 0.035 * wear) * 3.2 + 0.5)
    rim = np.exp(-distance_transform_edt(worn > 0.5) / (0.14 * px))
    level = float(np.clip(r.normal(0.965, 0.018), 0.9, 0.995))
    grain = 1 - 0.035 * sm(gaussian_filter(r.normal(0, 1, (n, n)).astype(np.float32), 0.8) * 3)
    ink = worn * level * (0.965 + 0.035 * rim) * grain
    l, t = int(round((cx + r.normal(0, 0.05)) * px - n / 2)), int(round((cy + r.normal(0, 0.05)) * px - n / 2))
    for out, v in ((A, ink), (R, gaussian_filter(worn, 0.06 * px)), (S, worn)):
        b = out[t:t + n, l:l + n]
        b[:] = np.maximum(b, v)


def ruling_plan():
    """The columns of the form from the right, what goes on each: title, the parties, the order, the reasons, the date. Returns a list of (column, position down it in steps, character, kind)."""
    out = []
    cap = int((FRAME[3] - FRAME[2] - 6) / TYPE_STEP)
    col = 1
    for j, c in enumerate(RULING_TITLE):
        out.append((col, 2.2 + j * 1.5, c, "title"))
    col += 2
    for party in RULING_PARTIES:
        at = 1
        for part in party:
            for c in part:
                out.append((col, at, c, "type"))
                at += 1
            at += 1
        col += 1
    at = 1
    for c in "主文":
        out.append((col, at, c, "type"))
        at += 1
    at += 1
    for c in RULING_ORDER:
        out.append((col, at, c, "type"))
        at += 1
    col += 1
    at = 1
    for c in "理由":
        out.append((col, at, c, "type"))
        at += 1
    at += 1
    for c in RULING_REASON:
        if at >= cap:
            col += 1
            at = 2
        out.append((col, at, c, "type"))
        at += 1
    col += 1
    for j, c in enumerate(RULING_DATE):
        out.append((col, 2 + j, c, "type"))
    assert col < RULING_COLS, col
    return out, col


def column_x(k):
    """The middle of a column, counted from the right of the form."""
    return FRAME[1] - (k + 0.5) * (FRAME[1] - FRAME[0]) / RULING_COLS


def column_y(at):
    """How far down the page the place at steps from a column's head is."""
    return FRAME[2] + 3.5 + TYPE_STEP * (at + 0.5)


def form_lines(h, w, px):
    """The form, printed in red from brass rule: every column's line, and a double frame round them, the outer heavy. What the rule did not quite touch is thin, but it is one straight line from end to end."""
    r = np.random.default_rng(211)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x_mm, y_mm = (xx + 0.5) / px, (yy + 0.5) / px
    x0, x1, y0, y1 = FRAME
    inside = (y_mm > y0) & (y_mm < y1) & (x_mm > x0) & (x_mm < x1)
    line = np.zeros((h, w), np.float32)

    def bar(d, wide):
        return sm((wide / 2 - np.abs(d)) * px + 0.5)
    for k in range(1, RULING_COLS):
        x = x1 - k * (x1 - x0) / RULING_COLS + float(r.normal(0, 0.015))
        line = np.maximum(line, bar(x_mm - x, 0.27) * ((y_mm > y0) & (y_mm < y1)))
    def outer(wide, gap):
        down = (y_mm > y0 - gap - wide / 2) & (y_mm < y1 + gap + wide / 2)
        over = (x_mm > x0 - gap - wide / 2) & (x_mm < x1 + gap + wide / 2)
        return np.maximum.reduce([bar(x_mm - (x0 - gap), wide) * down, bar(x_mm - (x1 + gap), wide) * down, bar(y_mm - (y0 - gap), wide) * over, bar(y_mm - (y1 + gap), wide) * over])
    line = np.maximum.reduce([line, outer(0.3, 0.0), outer(1.3, 1.5)])
    level = 0.9 + 0.06 * noise(h, w, 2.4, 212) - 0.05 * sm(noise(h, w, 0.8, 213))
    return line, np.clip(line * level, 0, 1)


def seal_map(cx, cy, side, h, w, px):
    """A seal we have never seen: a block of red with the crowded square lines an office's seal has, printed long ago from an old pad, so that it says nothing. Nothing in it is a character."""
    r = np.random.default_rng(221)
    n = int(side * px)
    k = 4
    im = Image.new("L", (n * k, n * k), 0)
    d = ImageDraw.Draw(im)
    m = int(0.045 * n * k)
    wide = int(0.05 * n * k)
    d.rectangle((m, m, n * k - m, n * k - m), outline=255, width=wide)
    # lines turning square corners, close together, wandering at random over the middle
    cells = 15
    cell = (n * k - 2 * m - 2 * wide) / cells
    o = m + wide
    for _ in range(100):
        i, j = r.integers(0, cells + 1, 2)
        path = [(o + i * cell, o + j * cell)]
        for _ in range(int(r.integers(2, 6))):
            if r.uniform() < 0.5:
                i = int(np.clip(i + r.integers(-3, 4), 0, cells))
            else:
                j = int(np.clip(j + r.integers(-3, 4), 0, cells))
            path.append((o + i * cell, o + j * cell))
        d.line(path, fill=255, width=int(cell * 0.4), joint="curve")
    a = np.asarray(im.rotate(1.3, Image.BICUBIC).resize((n, n), Image.BOX), np.float32) / 255
    a = gaussian_filter(a, 0.12 * px)
    # the pad was dry in places, the paste has spread and faded; what was a line is now mostly a stain
    blotch = spectral(n, 2.4, 222)
    fine = spectral(n, 1.1, 223)
    a = sm((a - 0.4 - 0.28 * blotch - 0.12 * fine) / 0.34)
    a = gaussian_filter(a, 0.17 * px) * (0.84 + 0.12 * blotch)
    a = a + 0.13 * sm(blotch * 0.8 + 0.4) * sm(1 - np.abs(np.linspace(-1, 1, n))[None, :] ** 6) * sm(1 - np.abs(np.linspace(-1, 1, n))[:, None] ** 6)
    out = np.zeros((h, w), np.float32)
    l, t = int((cx - side / 2) * px), int((cy - side / 2) * px)
    out[t:t + n, l:l + n] = np.clip(a, 0, 1)
    return out


def scrawl(r):
    """A signature no one could read: the ways of a few characters, run together into one quick movement of the brush down the column and squeezed narrow, so that it moves as writing moves and says nothing."""
    way = []
    y = 8.0
    for c in r.choice(list("永水也於之以"), 4, replace=False):
        k, narrow, x = r.uniform(0.2, 0.27), r.uniform(0.75, 1.05), r.normal(50, 2.5)
        for line in skeleton(c, XING):
            way.extend(line[:: max(len(line) // 9, 1), :2] * [k * narrow, k] + [x - 50 * k * narrow, y - 50 * k + 12])
        y += 100 * k * 0.8 + r.uniform(-1, 3)
    way = gaussian_filter1d(np.array(way), 0.8, axis=0, mode="nearest")
    far = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(way, axis=0).T))])
    t = np.arange(0, far[-1], 1.0)
    way = np.stack([np.interp(t, far, way[:, 0]), np.interp(t, far, way[:, 1])], 1)
    press = np.clip(0.5 + 0.3 * gaussian_filter1d(r.normal(0, 1, len(way)), 6) * 5, 0.28, 0.95)
    pts = [(*p, float(q), 0.25 + 0.35 * i / len(way)) for i, (p, q) in enumerate(zip(way, press))]
    pts[0] = pts[0][:2] + (0.12, 0.0)
    pts[-1] = pts[-1][:2] + (0.03, 0.6)
    return [(pts, 1.15, 0.7)]


def ruling_shape():
    """Folded once across and filed flat for a long time: hardly any swell, cut edges, the fold still there, corners that have begun to lift."""
    w, h = RULING_W, RULING_H
    U, V, X, Y = page_grid(w, h, 0.25, 0.03, 231)
    f = h / 2 + (U - w / 2) * 0.002
    d = V - f
    z = np.tan(np.radians(2.2)) * 60 * (1 - np.exp(-np.abs(hinge(-d, 0.25)) / 60)) + np.tan(np.radians(3.0)) * 60 * (1 - np.exp(-hinge(d, 0.25) / 60))
    z += -0.06 * np.exp(-(d / 0.3) ** 2) + bumps(U, V, w, h, 120, 0.22, 233) + bumps(U, V, w, h, 20, 0.012, 234) + curl(U, V, w, h, (0.5, 1.2, 0.7, 1.1), (6, 8, 7, 10), 235)
    return X, Y, z, f


def write_ruling():
    px = RULING_PX
    h, w = int(RULING_H * px), int(RULING_W * px)
    A, relief, S = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    r = np.random.default_rng(201)
    plan, last = ruling_plan()
    chars = []
    for col, at, c, kind in plan:
        cx = column_x(col)
        cy = column_y(at)
        f, em = (SONG_BOLD, 10.3) if kind == "title" else (SONG, TYPE_EM)
        set_type(A, relief, S, c, cx, cy, f, em, r, px)
        half = em / 2
        chars.append(dict(c=c, col=col, side=em, x=cx - half, y=cy - half, marks=[([(0, 0, 0), (100, 100, 0)], 1, 1)], kind=kind))
    rule_shape, red = form_lines(h, w, px)
    S = np.maximum(S, rule_shape)
    print("set", flush=True)
    # the signature, then the seal on top of it and the date's end, as it was done
    cx, cy = column_x(last), column_y(2 + len(RULING_DATE)) + 13.0
    pen = Pen(1400, 8.5, seed=9)
    sign = np.zeros((h, w), np.float32)
    rs = np.random.default_rng(217)
    over(sign, pen.write(scrawl(rs), 31, tone=0.98), column_x(last) - 33, column_y(2 + len(RULING_DATE)) - 1.0, 66.0, px)
    seal = seal_map(cx, cy, 54.0, h, w, px)
    # the paper
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (xx + 0.5) / px, RULING_H - (yy + 0.5) / px
    col = np.broadcast_to(srgb(235, 232, 222), (h, w, 3)).copy()
    pulp = stretch(fibres(4096, 30000, 241, length=(24, 90), width=(1, 1), bright=(.3, 1.0), wiggle=.3), h, w)
    tone = 1 + 0.012 * noise(h, w, 2.8, 242) + 0.008 * noise(h, w, 1.0, 243) + 0.012 * gaussian_filter(pulp, 0.6)
    col = col * tone[..., None]
    col = col * (1 - edge_shade(u, v, RULING_W, RULING_H, 4.0, 244)[..., None] * np.array((0.08, 0.1, 0.17), np.float32))
    _, _, _, f = ruling_shape()
    d = v - np.interp(u, np.linspace(0, RULING_W, f.shape[1]), f[0])
    n = noise(h, w, 1.6, 245)
    col = col * (1 - (0.06 * np.exp(-(d / 1.8) ** 2) * (0.7 + 0.5 * n))[..., None])
    wear = 1 - 0.5 * np.exp(-(d / 0.28) ** 2) * sm((n + 0.5) / 0.8)
    red_col, black = srgb(166, 38, 32), srgb(22, 20, 20)
    col = col + (red_col - col) * (red * wear)[..., None]
    a = A * wear
    col = col + (black - col) * a[..., None]
    col = col + (srgb(26, 22, 24) - col) * sign[..., None]
    seal = seal * (1 - 0.35 * np.exp(-(d / 0.5) ** 2))
    # the paste sits on top of everything and goes darker over the black
    col = col * (1 - seal[..., None] * (1 - srgb(150, 26, 24)))
    rough = 0.72 - 0.2 * a - 0.14 * red - 0.2 * seal
    keys = dict(title=[box_of(c) for c in chars if c["kind"] == "title"], order=[box_of(c) for c in chars if c["col"] == 5], date=[box_of(c) for c in chars if c["col"] == last], seal=[cx - 27, cy - 27, cx + 27, cy + 27])
    finish("ruling", col, rough, gaussian_filter(np.maximum(relief, rule_shape * 0.6), 0.04 * px), ruling_shape()[:3], RULING_W, RULING_H, px, chars, keys, S, frame=FRAME, last=last)


# ---------------------------------------------------------------- the tablet: clay from Nineveh, its first line the first words of 《吉尔伽美什》

# ša | na-aq-ba | i-mu-ru: the standard version's opening, the seven signs film_wo.py and film_press_maps.py check by their Unicode names. Nothing after them is known to us, so the rest of the
# tablet is what most of these tablets are: worn. Its other lines are wedges that no longer make a sign, abraded and flaked away; none of them is a word.
GILGAMESH_WORDS = ("𒊭", "𒈾𒀝𒁀", "𒄿𒈬𒊒")
TABLET_W, TABLET_H, TABLET_PX = 112.0, 146.0, 20
TABLET_STEP = 0.1                   # the mesh, in mm
TABLET_EM = 6.2                     # a sign's em
TABLET_PITCH = 7.6                  # line to line
TABLET_LEFT, TABLET_LINE1 = 9.0, 13.0     # the first sign's left edge, the first line's middle, from the tablet's top left
TABLET_HIGH, TABLET_EDGE = 21.0, 14.0     # how high the middle of its face stands off the table, and its rim before it rounds over
CUNE = ("NotoSansCuneiform-Regular.ttf", 0)


def cuneiform_check():
    import unicodedata
    names = [unicodedata.name(c).split()[-1] for c in "".join(GILGAMESH_WORDS)]
    assert names == ["SHA", "NA", "AK", "BA", "I", "MU", "RU"], names


def impression(mask, px, slope=1.15, most=1.7):
    """What a stylus leaves: a V cut into the clay, deepest where the wedge is widest. mask 0-1 at px a mm. Depth in mm."""
    d = distance_transform_edt(mask > 0.5) / px
    return np.float32(np.minimum(d * slope, most) * sm(np.clip(mask * 1.6 - 0.3, 0, 1)))


def wedge(d, x, y, ang, size, tail, px):
    """One wedge drawn into an ImageDraw at px: a triangular head at (x, y) mm, pointing along ang, and its tail behind it."""
    c, s = np.cos(ang), np.sin(ang)
    hx, hy = x * px, y * px
    L, W = size * 0.95 * px, size * 0.5 * px
    tip = (hx + c * L, hy + s * L)
    left, right = (hx - s * W, hy + c * W), (hx + s * W, hy - c * W)
    d.polygon([left, right, tip], fill=255)
    if tail > 0:
        t = size * 0.11 * px
        e = (hx + c * (L + tail * px), hy + s * (L + tail * px))
        d.polygon([(hx + c * L * 0.6 - s * t, hy + s * L * 0.6 + c * t), (hx + c * L * 0.6 + s * t, hy + s * L * 0.6 - c * t), (e[0] + s * t * 0.4, e[1] - c * t * 0.4), (e[0] - s * t * 0.4, e[1] + c * t * 0.4)], fill=255)


def tablet_outline(u, v):
    """Signed distance in mm to the tablet's edge, negative inside: a rounded oblong, worked by hand, so no edge is straight."""
    hw, hh, R = TABLET_W / 2 - 1.5, TABLET_H / 2 - 1.5, 16.0
    qx, qy = np.abs(u - TABLET_W / 2) - (hw - R), np.abs(v - TABLET_H / 2) - (hh - R)
    sd = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - R
    a = np.arctan2(v - TABLET_H / 2, u - TABLET_W / 2)
    wob = 0.0
    r = np.random.default_rng(401)
    for k, amp in ((2, 0.9), (3, 0.6), (5, 0.35), (9, 0.2), (17, 0.12)):
        wob = wob + amp * np.sin(k * a + r.uniform(0, 6.28))
    return sd + wob


def tablet_plan(r):
    """The signs of the first line, where they go, and the worn wedges of every other line."""
    f = face(CUNE[0], CUNE[1], TABLET_EM * TABLET_PX)
    signs, x = [], TABLET_LEFT
    for w_i, word in enumerate(GILGAMESH_WORDS):
        for c in word:
            box = f.getbbox(c, anchor="lm")
            wide = (box[2] - box[0]) / TABLET_PX
            signs.append(dict(c=c, x=x - box[0] / TABLET_PX, y=TABLET_LINE1 + float(r.normal(0, 0.15)), box=(x, TABLET_LINE1 + box[1] / TABLET_PX, x + wide, TABLET_LINE1 + box[3] / TABLET_PX)))
            x += wide + 1.2
        x += 3.2
    return f, signs, x


def write_tablet():
    cuneiform_check()
    px, w, h = TABLET_PX, TABLET_W, TABLET_H
    H, W_ = int(h * px), int(w * px)
    r = np.random.default_rng(402)
    yy, xx = np.mgrid[0:H, 0:W_].astype(np.float32)
    u, v = (xx + 0.5) / px, (yy + 0.5) / px          # mm from the top left
    sd = tablet_outline(u, h - v)
    # the body: thick in the middle, a pillow, its rim rounded over down to the table
    cx, cy = (u - w / 2) / (w / 2), (v - h / 2) / (h / 2)
    body = TABLET_EDGE + (TABLET_HIGH - TABLET_EDGE) * (1 - 0.55 * cx ** 2 - 0.45 * cy ** 2) + 0.6 * noise(H, W_, 3.4, 403)
    e = np.clip(-sd / 7.0, 0, 1)
    Z = body * np.sqrt(1 - (1 - e) ** 2)
    # the first line
    f, signs, line_end = tablet_plan(r)
    im = Image.new("L", (W_, H), 0)
    d = ImageDraw.Draw(im)
    for s in signs:
        d.text((s["x"] * px, s["y"] * px), s["c"], 255, font=f, anchor="lm")
    first = np.asarray(im, np.float32) / 255
    cut_first = impression(first, px)
    # every other line, and the rest of the first: signs of the script set close as a scribe set them, then worn. Wear takes a sign a wedge at a time, so what is left of any of them is
    # never a whole sign, and no line can be read: none of it is text
    pool = [chr(c) for c in range(0x12000, 0x1236F) if f.getbbox(chr(c)) and (f.getbbox(chr(c))[2] - f.getbbox(chr(c))[0]) / px < TABLET_EM * 0.95]
    im = Image.new("L", (W_, H), 0)
    d = ImageDraw.Draw(im)
    y = TABLET_LINE1
    rows = []
    while y < h - 8:
        x = line_end if y == TABLET_LINE1 else TABLET_LEFT + float(r.uniform(-0.4, 0.6))
        lean = float(r.normal(0, 0.01))
        while x < w - 10:
            c = pool[int(r.integers(len(pool)))]
            box = f.getbbox(c, anchor="lm")
            d.text(((x * px) - box[0], (y + (x - TABLET_LEFT) * lean) * px), c, 255, font=f, anchor="lm")
            x += (box[2] - box[0]) / px + float(r.uniform(0.6, 1.6)) + (float(r.uniform(1.5, 3.0)) if r.random() < 0.18 else 0.0)
        rows.append(y)
        y += TABLET_PITCH * float(np.exp(r.normal(0, 0.012)))
    worn = np.asarray(im, np.float32) / 255
    cut_worn = impression(worn, px, slope=1.05, most=1.5)
    # how much of each place has kept its face: patchy at the scale of a wedge, worse towards the foot and the edges; the seven signs untouched
    edge_wear = sm(np.clip(1 - (-sd) / 9.0, 0, 1))
    keep = sm((noise(H, W_, 2.0, 404) * 0.45 + noise(H, W_, 3.0, 405) * 0.35 + 0.55 - 0.45 * np.clip((v - 20) / 120, 0, 1) - 0.6 * edge_wear) / 0.45)
    first_zone = np.zeros_like(u)
    for s_ in signs:
        x0, y0, x1, y1 = s_["box"]
        first_zone = np.maximum(first_zone, ((u > x0 - 2) & (u < x1 + 2) & (v > y0 - 2) & (v < y1 + 2)).astype(np.float32))
    first_zone = gaussian_filter(first_zone, 1.5 * px)
    keep = np.maximum(keep, sm(first_zone * 2))
    worn_cut = gaussian_filter(cut_worn, 0.3 * px) * (1 - keep) * 0.3 + cut_worn * keep
    # the lines were ruled with a string pressed into the clay before writing
    rule = np.zeros_like(u)
    for y in rows[1:]:
        yy0 = y - TABLET_PITCH / 2 + 1.2 * noise(1, W_, 2.0, int(y * 10))[0] * 0.15
        rule = np.maximum(rule, sm((0.22 - np.abs(v - yy0)) / 0.12) * sm((noise(H, W_, 1.4, 406) + 1.2) / 0.6))
    rule = rule * (0.18 * keep + 0.03)
    # flakes: the face has come away in places, leaving a rough floor a millimetre down with a broken edge
    chip = np.exp(-(((u - w * 0.93) / 26) ** 2 + ((v - h * 0.95) / 22) ** 2)) * 1.4
    spall = sm((noise(H, W_, 2.6, 407) * 0.6 + noise(H, W_, 1.2, 408) * 0.1 - 1.05 + chip + 0.25 * np.clip((v - 70) / 70, 0, 1)) / 0.04) * (1 - sm(first_zone * 3))
    floor = 0.9 + 0.35 * noise(H, W_, 1.6, 409)
    surface = -cut_first - worn_cut - rule
    Z = Z + (surface * (1 - spall) - floor * spall) * sm(np.clip(-sd / 6.0, 0, 1))
    # the clay's own grain, pits and a few hairline cracks from firing
    grain = 0.03 * noise(H, W_, 1.0, 410) + 0.05 * noise(H, W_, 1.8, 411)
    Z = Z + grain
    Z = np.where(sd < 0, Z, -2.0).astype(np.float32)
    depth = np.clip(cut_first + worn_cut + rule, 0, None)
    # its colour: buff clay fired unevenly in the fire that destroyed the palace, dirt in every cut, salts that came up out of it
    base = srgb(178, 150, 116)
    burnt = srgb(98, 86, 76)
    pale = srgb(206, 186, 156)
    blot = sm((noise(H, W_, 3.2, 412) * 0.7 + noise(H, W_, 2.2, 420) * 0.3 + 0.4) / 1.4)
    col = base + (burnt - base) * (0.32 * blot ** 2)[..., None]
    col = col * (1 + 0.06 * noise(H, W_, 1.4, 413)[..., None] + 0.03 * noise(H, W_, 0.8, 414)[..., None])
    col = col + (pale - col) * (0.35 * spall * sm((noise(H, W_, 1.0, 415) + 0.5) / 1.0))[..., None]
    dirt = sm(depth / 0.5) * 0.45 + sm(gaussian_filter(depth, 0.4 * px) / 0.6) * 0.15
    col = col * (1 - dirt[..., None] * np.array((0.55, 0.6, 0.65), np.float32))
    salt = sm((noise(H, W_, 0.6, 416) - 1.6) / 0.4) * sm((noise(H, W_, 2.5, 417) + 0.3) / 0.8)
    col = col + (srgb(232, 226, 214) - col) * (salt * 0.6)[..., None]
    rim = sm(np.clip(1 - (-sd) / 5.0, 0, 1))
    col = col * (1 - 0.18 * rim[..., None])
    rough = np.clip(0.86 + 0.06 * noise(H, W_, 1.2, 418) - 0.08 * (1 - spall) * (1 - dirt), 0, 1)
    relief = np.clip(0.5 + 2.2 * grain + 0.15 * spall * noise(H, W_, 0.7, 419), 0, 1)
    # the mesh: every TABLET_STEP, rows from the foot up
    k = int(round(TABLET_STEP * px))
    Zg = gaussian_filter(Z, 0.4 * k)[::k, ::k][::-1]
    gy, gx = Zg.shape
    X, Y = np.meshgrid((np.arange(gx) * k + 0.5) / px, h - (np.arange(gy)[::-1] * k + 0.5) / px)
    os.makedirs(MAPS, exist_ok=True)
    save8(to_srgb(col), "tablet_col.png")
    save8(rough, "tablet_rough.png")
    save8(relief, "tablet_relief.png")
    np.savez(os.path.join(MAPS, "tablet.npz"), X=np.float32(X), Y=np.float32(Y), Z=np.float32(Zg), solid=True)
    chars = [dict(c=s["c"], col=0, box=[round(v_, 2) for v_ in s["box"]]) for s in signs]
    keys = dict(run=[s["box"] for s in signs], first=list(signs[0]["box"]))
    with open(os.path.join(MAPS, "tablet.json"), "w", encoding="utf-8") as fp:
        json.dump(dict(w=w, h=h, px=px, keys=keys, chars=chars, solid=True), fp, ensure_ascii=False)
    # the die carries the seven signs alone, raised as a stamp's are: every wedge's whole shape stands, so its thin tail prints as well as its head, and the head stands
    # higher, so it goes in deeper, as the stylus went
    stand = 0.6 * die_face(first, px) / FACE_RELIEF + 0.4 * np.clip(cut_first / 1.2, 0, 1)
    save_die("tablet", np.float32(stand * FACE_RELIEF), px, w_mm=w, h_mm=h, keys=json.dumps(keys), chars=json.dumps(chars, ensure_ascii=False))
    print("tablet", Zg.shape, "z %.1f..%.1f mm" % (float(Zg.min()), float(Zg.max())), "first line ends at %.1f mm" % line_end, flush=True)


# ---------------------------------------------------------------- the quarto: a leaf of SHAKE-SPEARES SONNETS, 1609, set in lead type

QUARTO_W, QUARTO_H, QUARTO_PX = 136.0, 186.0, 24
QUARTO_EM, QUARTO_LEAD = 4.05, 4.75         # a pica roman, solid with a little lead
QUARTO_LEFT, QUARTO_FIRST = 24.0, 39.0      # the text's left, the first line's baseline
QUARTO_INDENT = 5.2                        # the couplet stands in
QUARTO_ROMAN = ("Hoefler Text.ttc", 0)
QUARTO_HEAD = "SONNETS."                   # the running title of a recto


def sonnet_lines():
    from film_wo_texts import lines
    out = lines("sonnet")
    assert len(out) == 14 and out[0].startswith("Shall I compare") and out[-1] == "So long liues this, and this giues life to thee,", out
    return [t.replace("'", "\u2019") for t in out]      # the apostrophe as the case held it


def set_line(text, x, base, f, px, r, A, R, S, wear=1.0):
    """A line of cast type pressed into the sheet, sort by sort: each a little off its place, inked a little differently, its ink pushed to its edges, its face sunk into the paper. Returns its box in mm."""
    x0 = x
    for i, c in enumerate(text):
        adv = f.getlength(text[:i + 1]) / px - f.getlength(text[:i]) / px
        if c != " ":
            box = f.getbbox(c, anchor="ls")
            l, t = int(np.floor((x * px) + box[0])) - 4, int(np.floor(base * px + box[1])) - 4
            n_w, n_h = box[2] - box[0] + 9, box[3] - box[1] + 9
            im = Image.new("L", (n_w * 3, n_h * 3), 0)
            ImageDraw.Draw(im).text(((x * px - l) * 3 + float(r.normal(0, 0.02 * px)) * 3, (base * px - t) * 3 + float(r.normal(0, 0.03 * px)) * 3), c, 255, font=f.font_variant(size=f.size * 3), anchor="ls")
            a = np.asarray(im.resize((n_w, n_h), Image.BOX), np.float32) / 255
            n = gaussian_filter(r.normal(0, 1, a.shape).astype(np.float32), 1.6)
            n /= n.std() + 1e-6
            spread = float(r.uniform(0.3, 0.42))      # how hard this sort was inked and pulled: the ink is squeezed out past its face
            worn = sm((gaussian_filter(a, 0.025 * px) - spread - 0.07 * n * wear) * 3.0 + 0.5)
            rim = np.exp(-distance_transform_edt(worn > 0.5) / (0.05 * px))
            level = float(np.clip(r.normal(0.95, 0.06), 0.7, 1.0))
            mottle = 1 - 0.22 * sm(gaussian_filter(r.normal(0, 1, a.shape).astype(np.float32), 0.05 * px) * 2.5)
            ink = worn * level * (0.78 + 0.22 * rim) * mottle
            for out, val in ((A, ink), (R, gaussian_filter(worn, 0.05 * px)), (S, worn)):
                b = out[t:t + n_h, l:l + n_w]
                b[:] = np.maximum(b, val[:b.shape[0], :b.shape[1]])
        x += adv
    return (x0, base - QUARTO_EM * 0.72, x, base + QUARTO_EM * 0.25)


def quarto_shape():
    """A leaf out of the book, flat a long time: a slow swell, its edges a little cockled, one corner turned once."""
    w, h = QUARTO_W, QUARTO_H
    U, V, X, Y = page_grid(w, h, 0.25, 0.35, 501)
    z = bumps(U, V, w, h, 90, 0.5, 503) + bumps(U, V, w, h, 14, 0.03, 504) + curl(U, V, w, h, (0.9, 1.4, 1.0, 1.2), (8, 10, 9, 9), 505)
    c = (w - U) + V - 18.0
    z += 0.16 * hinge(-c, 0.6) * (hinge(-c, 0.6) < 18)
    return X, Y, z


def write_quarto():
    px, w, h = QUARTO_PX, QUARTO_W, QUARTO_H
    H, W_ = int(h * px), int(w * px)
    A, R, S = np.zeros((H, W_), np.float32), np.zeros((H, W_), np.float32), np.zeros((H, W_), np.float32)
    head = np.zeros_like(A)
    r = np.random.default_rng(511)
    roman = face(QUARTO_ROMAN[0], QUARTO_ROMAN[1], QUARTO_EM * px)
    text = sonnet_lines()
    # the running title and the number, centred over the text's measure
    measure = max(roman.getlength(t) for t in text) / px
    big = face(QUARTO_ROMAN[0], QUARTO_ROMAN[1], QUARTO_EM * 1.12 * px)
    hw = big.getlength(QUARTO_HEAD) / px
    set_line(QUARTO_HEAD, QUARTO_LEFT + measure / 2 - hw / 2, QUARTO_FIRST - 22.0, big, px, r, head, R, np.zeros_like(S))
    num = "18."
    nb = set_line(num, QUARTO_LEFT + measure / 2 - roman.getlength(num) / px / 2, QUARTO_FIRST - 7.0, roman, px, r, A, R, S)
    boxes = []
    for i, t in enumerate(text):
        x = QUARTO_LEFT + (QUARTO_INDENT if i >= 12 else 0.0)
        boxes.append(set_line(t, x, QUARTO_FIRST + i * QUARTO_LEAD, roman, px, r, A, R, S))
    print("set", len(text), "lines, measure %.1f mm" % measure, flush=True)
    yy, xx = np.mgrid[0:H, 0:W_].astype(np.float32)
    u, v = (xx + 0.5) / px, h - (yy + 0.5) / px
    # laid rag paper: its chain lines every 25 mm or so and the fine laid lines across them, browned at the edges, foxed, the text of the other side faintly through it
    col = np.broadcast_to(srgb(222, 205, 168), (H, W_, 3)).copy()
    pulp = stretch(fibres(4096, 14000, 512, length=(30, 150), width=(1, 2), bright=(.3, 1.0), wiggle=.3), H, W_)
    chain = np.exp(-((np.mod(u - 7.0 + 0.6 * noise(H, W_, 3.0, 513), 25.4) - 12.7) / 0.5) ** 2)
    laid = 0.5 + 0.5 * np.sin(2 * np.pi * (v + 0.25 * noise(H, W_, 2.5, 514)) / 1.05)
    tone = 1 + 0.02 * noise(H, W_, 2.8, 515) + 0.01 * noise(H, W_, 1.0, 516) + 0.02 * gaussian_filter(pulp, 0.6) - 0.012 * chain + 0.006 * laid
    col = col * tone[..., None]
    col = col * (1 - edge_shade(u, v, w, h, 14.0, 517)[..., None] * np.array((0.16, 0.25, 0.4), np.float32))
    stain = sm((noise(H, W_, 3.4, 518) - 0.6) / 0.5) * np.clip((u - 70) / 60, 0, 1)
    col = col * (1 - (stain * 0.1)[..., None] * np.array((0.5, 0.8, 1.2), np.float32))
    fox = np.zeros((H, W_), np.float32)
    for x, y, s, a in zip(r.uniform(0, w, 70), r.uniform(0, h, 70), r.uniform(0.3, 2.2, 70), r.uniform(0.15, 0.7, 70)):
        i, j, m = int((h - y) * px), int(x * px), int(s * px * 3) + 2
        a0, a1, b0, b1 = max(i - m, 0), min(i + m, H), max(j - m, 0), min(j + m, W_)
        yy2, xx2 = np.ogrid[a0:a1, b0:b1]
        fox[a0:a1, b0:b1] = np.maximum(fox[a0:a1, b0:b1], a * np.exp(-(((yy2 - i) ** 2 + (xx2 - j) ** 2) / (s * px) ** 2)))
    col = col * (1 - (fox * 0.28)[..., None] * np.array((0.35, 0.62, 1.0), np.float32))
    through = gaussian_filter(np.maximum(A, head)[:, ::-1], 0.35 * px)
    through = np.roll(through, (int(-1.5 * px), int(3.0 * px)), (0, 1))
    col = col * (1 - (through * 0.06)[..., None])
    # the ink: lamp black in varnish, pressed out to the edges of each letter, thin on the worn sorts
    a = np.maximum(A, head)
    ink = srgb(17, 15, 14)
    col = col + (ink - col) * np.clip(a * 1.08, 0, 1)[..., None]
    rough = 0.8 - 0.14 * a
    relief = gaussian_filter(R, 0.04 * px)
    sb = roman.getbbox(text[0][0], anchor="ls")
    first = [QUARTO_LEFT + sb[0] / px, QUARTO_FIRST + sb[1] / px, QUARTO_LEFT + sb[2] / px, QUARTO_FIRST + sb[3] / px]
    keys = dict(lines=[[list(b)] for b in boxes], last=[list(boxes[-1])], first=first, number=[list(nb)])
    chars = [dict(c=t, col=i, box=[round(v_, 2) for v_ in b]) for i, (t, b) in enumerate(zip(text, boxes))]
    finish("quarto", col, rough, relief, quarto_shape(), w, h, px, chars, keys, S)


def first_glyph(name):
    """The box of the first character a reader meets on a carrier, in mm from its top left: where 二's montage puts every carrier's first letter."""
    with open(os.path.join(MAPS, name + ".json"), encoding="utf-8") as fp:
        info = json.load(fp)
    return info["keys"]["first"]


# ---------------------------------------------------------------- the table

def table():
    """The film's first desk, gone dark: walnut-coloured with age and old oil, finer pores than the picture it starts from can hold, marks of use."""
    desk = os.path.join(ROOT, "ink", "film_out", "3d", "book", "tex")
    k = 2
    a = Image.open(os.path.join(desk, "desk.png")).convert("L")
    w0, h0 = a.size
    w, h = w0 * k, h0 * k
    a = np.asarray(a.resize((w, h), Image.BICUBIC), np.float32) / 255
    bump = np.asarray(Image.open(os.path.join(desk, "desk_bump.png")).convert("L").resize((w, h), Image.BICUBIC), np.float32) / 255
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    n = np.clip((a - lo) / (hi - lo), 0, 1)
    pores = gaussian_filter(np.random.default_rng(301).normal(0, 1, (h, w)).astype(np.float32), (0.7, 26))
    pores /= pores.std()
    streak = gaussian_filter(np.random.default_rng(302).normal(0, 1, (h, w)).astype(np.float32), (3, 160))
    streak /= streak.std()
    patina = noise(h, w, 2.8, 303)
    shade = np.clip(n ** 1.1 + 0.07 * streak, 0, 1)
    dark, light = srgb(28, 18, 13), srgb(82, 54, 36)
    col = dark + (light - dark) * shade[..., None]
    col = col * (1 + 0.14 * patina[..., None] - 0.2 * sm(pores - 0.6)[..., None])
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    r = np.random.default_rng(304)
    for _ in range(420):
        x, y, an, ln = r.uniform(0, w), r.uniform(0, h), r.normal(0, 0.9), r.uniform(10, 280)
        d.line((x, y, x + np.cos(an) * ln, y + np.sin(an) * ln), fill=int(r.uniform(30, 170)), width=1)
    scratches = gaussian_filter(np.asarray(im, np.float32) / 255, 0.5)
    col = col + scratches[..., None] * srgb(70, 54, 42)
    rough = np.clip(0.58 + 0.12 * patina + 0.14 * sm(pores) + 0.08 * scratches, 0.4, 0.9)
    relief = np.clip(0.5 + 0.4 * (bump - 0.5) - 0.11 * sm(pores - 0.3) - 0.05 * scratches, 0, 1)
    os.makedirs(MAPS, exist_ok=True)
    save8(to_srgb(col), "table_col.png")
    save8(rough, "table_rough.png")
    save8(relief, "table_relief.png")


# ---------------------------------------------------------------- check

def check():
    """Every character in the face that sets it, none of them the missing-glyph box; the texts as they were checked; the run on one column."""
    def missing(name, index, text):
        f = face(name, index, 64)
        box = bytes(f.getmask("\U000FFFFD"))
        return "".join(sorted({c for c in text if bytes(f.getmask(c)) == box or f.getmask(c).getbbox() is None}))
    letter, poem = LETTER, POEM_TITLE + "".join(POEM)
    ruling = RULING_TITLE + "".join("".join(p) for p in RULING_PARTIES) + "主文" + RULING_ORDER + "理由" + RULING_REASON + RULING_DATE
    for name, index, text in (("Xingkai.ttc", 1, letter), ("Xingkai.ttc", 3, letter), ("Xingkai.ttc", 1, poem), ("Xingkai.ttc", 3, poem), ("Songti.ttc", 7, ruling), ("Songti.ttc", 2, RULING_TITLE)):
        m = missing(name, index, text)
        assert not m, (name, index, m)
    from film_hand import _graphics
    print("stroke data lacks:", "".join(sorted({c for c in letter + poem if c not in _graphics()})), "(stood in for by", STAND_IN, ")", flush=True)
    assert KEY in LETTER and len(KEY) == 8
    assert all(len(x) == 5 for x in POEM)
    chars, run = letter_plan()
    assert "".join(c["c"] for c in chars) == LETTER
    print("letter", len(LETTER), "poem", len(poem), "ruling", len(ruling), "every character in its face", flush=True)
    for name, text in (("letter", letter), ("poem", poem), ("ruling", ruling)):
        with open(os.path.join(MAPS, name + ".json"), encoding="utf-8") as f:
            info = json.load(f)
        assert "".join(c["c"] for c in info["chars"]) == text, name
        a = np.load(os.path.join(DIES, name + ".npz"))
        h, w = a["height"].shape
        assert abs(h * float(a["mm_per_px"]) - info["h"]) < 0.2 and abs(w * float(a["mm_per_px"]) - info["w"]) < 0.2 and json.loads(str(a["chars"])) == info["chars"], name
        print("die", name, h, w, "%.1f mm high at most," % float(a["height"].max()), "%.1f%% of it stands" % (100 * float((a["height"] > 0).mean())), "as written", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "table":
        table()
    elif what == "check":
        check()
    elif what == "shape":
        for name in sys.argv[2:]:
            save_shape(name, *dict(letter=letter_shape, poem=poem_shape, ruling=ruling_shape)[name]()[:3])
    elif what == "write":
        for name in sys.argv[2:]:
            dict(letter=write_letter, poem=write_poem, ruling=write_ruling, tablet=write_tablet, quarto=write_quarto)[name]()
