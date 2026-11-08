"""The paper and the type of the press shots, in millimetres: world x to the right, y away from the press's front, z up, the sheet's top at z = 0. The forme's raised face and the
impression it leaves share the same (x, y), so a character drawn upright here reads right way round on the paper and mirrored on the metal seen from below.

    python film_press_maps.py bake [--px 30]        # text maps, four states of the sheet, the forme's face, the fibre tile, into ink/film_out/film/press/maps/
    python film_press_maps.py face [--px 30]        # the forme's face alone
    python film_press_maps.py fibre                 # the fibre tile alone
    python film_press_maps.py proof [--px 30]       # top-down hillshade of the last state, to read what was pressed
    python film_press_maps.py frame                 # where the sheet and the rows fall in shot C's frame

The three cameras live here too, since the sheet's grid is laid out for the last one. press_paper(), sheet_grid() and compose() take any camera, so a later shot can move over the
same sheet. film_press3d.py renders what this bakes. Importing this needs only numpy.
"""
import glob
import json
import math
import os
import sys
import unicodedata

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "ink", "film_out", "film", "press")
MAPS = os.path.join(OUT, "maps")

W, H = 1920, 1080
AREA = (-90.0, 90.0, -107.0, 70.0)      # x0, x1, y0, y1 of the rasters
SHEET = dict(left=-86.0, right=86.0, near=-103.0, far=66.0, thick=1.15)
RIM = dict(left=-97.0, right=97.0, near=-114.0, far=77.0, width=9.0, top=.2)        # the iron frame the sheet lies in
OLD_BOX = (-205.0, 190.0, 175.0)        # the old printing was laid out on a bigger page than this sheet: left, right, top

TYPE_HIGH = 23.3            # real type height
BLANK = 2.8                 # leads and quads stand this far behind the face
FORME_RES = 32              # samples per mm of the metal face
FACE_PX = 96                # the distance field behind it, 3 to each sample
SLOPE = 2.3                 # the chamfer of a sort, rise over run, the same on every stroke
FORME_AREA = (-74.0, 74.0, -27.0, 60.0)

# ---- the six rows, oldest and furthest first: baselines in mm, all flush left ----------------------------------------------------------------------
LEFT = -70.0
Y1, Y2, Y3, Y4, Y5, Y6 = 43.9, 32.6, 16.3, 0.0, -11.5, -23.0

# ---- the three cameras ------------------------------------------------------------------------------------------------------------------------------
SENSOR = 36.0
LENS_A, LENS_B, LENS_C = 50.0, 35.0, 35.0
# A: the whole machine, floor to crown, from where someone standing in front of it would look at its chest. The camera is level, so that the uprights stay upright and the head and crown stand over the eye; a lens shift puts the horizon at height cy. It creeps in by 2 % over the shot.
A = dict(target=(-26.0, -14.0, 85.0), d=2025.0, az=28.0, el=0.0, creep=.02, cy=390.0)
# B: from under the head, the face square on but for a small tilt that lets the far rows recede. Image up is +y, so the metal reads as a printer reads it: lines in order, each one mirrored.
B = dict(y=16.0, d=187.0, tilt=14.0, cy=540.0, fall=26.0)
# C: from the front, above. Never moves.
C = dict(target=(0.0, -19.0, 0.0), d=330.0, az=0.0, el=46.0, cy=470.0)
C_LIFT = 1.60               # the head's rise is stretched by this in shot C, where nothing shows its stroke, so that it leaves the frame, about 7.7 s in


def sm(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def orbit(target, d, az, el, lens, cy):
    az, el = math.radians(az), math.radians(el)
    loc = np.array(target) + d * np.array((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    return dict(loc=tuple(loc), target=tuple(target), lens=lens, sensor=SENSOR, cy=cy)


def camera_a(t):
    return orbit(A["target"], A["d"] * (1 - A["creep"] * t / 3), A["az"], A["el"], LENS_A, A["cy"])


def camera_b():
    """The forme's middle is at (0, B["y"], head height). Up in the image is returned as well, since the camera looks almost straight up."""
    tilt = math.radians(B["tilt"])
    loc = (0.0, B["y"] - B["d"] * math.sin(tilt), -B["d"] * math.cos(tilt))
    return dict(loc=loc, target=(0.0, B["y"], 0.0), up=(0.0, math.cos(tilt), -math.sin(tilt)), lens=LENS_B, sensor=SENSOR, cy=B["cy"])


def camera_c():
    return orbit(C["target"], C["d"], C["az"], C["el"], LENS_C, C["cy"])


CAM = camera_c()

FONTS = dict(
    song=("Songti.ttc", 6), songb=("Songti.ttc", 1), kai=("Kaiti.ttc", 0), hei=("PingFang.ttc", 3), heim=("PingFang.ttc", 7),
    mono=("Menlo.ttc", 1), monor=("Menlo.ttc", 0), bask=("Baskerville.ttc", 4), baskr=("Baskerville.ttc", 0),
    hiero=("NotoSansEgyptianHieroglyphs-Regular.ttf", 0), cune=("NotoSansCuneiform-Regular.ttf", 0),
)


def font(key, px):
    from PIL import ImageFont
    name, index = FONTS[key]
    paths = [f"/System/Library/Fonts/Supplemental/{name}", f"/System/Library/Fonts/{name}"] + glob.glob(f"/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/{name}")
    return ImageFont.truetype(next(p for p in paths if os.path.exists(p)), int(round(px)), index=index)


def hieroglyphs(*names):
    return "".join(unicodedata.lookup("EGYPTIAN HIEROGLYPH " + n) for n in names)


# ---- what was pressed --------------------------------------------------------------------------------------------------------------------------------
PTOLEMY = ("Q003", "X001", "V004", "E023", "AA015", "M017", "M017", "S029")     # p t w l m y s, between the two ends of the name ring
RING = hieroglyphs("V011A"), hieroglyphs("V011B")
GILGAMESH = "𒊭 𒈾𒀝𒁀 𒄿𒈬𒊒"     # ša na-aq-ba i-mu-ru
assert [unicodedata.name(c).split()[-1] for c in GILGAMESH.replace(" ", "")] == ["SHA", "NA", "AK", "BA", "I", "MU", "RU"]
HOMER = "ΜΗΝΙΝ ΑΕΙΔΕ ΘΕΑ ΠΗΛΗΙΑΔΕΩ ΑΧΙΛΗΟΣ"
SHAKESPEARE = "So long lives this, and this gives life to thee."
HELLO = 'printf("hello, world\\n");'
LI_BAI = "君不见黄河之水天上来"
MORTAL = "人固有一死"

# The six rows of the forme, in the order they appear on the sheet: (text, font, em, x of the first sort, baseline). Row 1's left half is the name ring, drawn by name_ring().
RING_EM = 9.0
RING_K = 3.75       # the ring's own em against the signs': its height, .41 em, has to hold signs 1.25 em tall
RING_FOOT = .19     # the font draws the ring this far above its baseline, in its em
CUNE_X = 9.0
ROWS = [
    (GILGAMESH, "cune", 6.4, CUNE_X, Y1 + 1.1),
    (HOMER, "bask", 6.2, LEFT, Y2),
    (MORTAL, "song", 14.5, LEFT, Y3),
    (LI_BAI, "song", 11.5, LEFT, Y4),
    (SHAKESPEARE, "bask", 6.0, LEFT, Y5),
    (HELLO, "mono", 7.6, LEFT, Y6),
]
FORME_SEED = 11
DEPTH = (.27, .27, .28, .30, .28, .34)      # how deep each row is pressed
INK_DENSITY = (.80, .84, .90, .94, .975, 1.0)    # the newest line is black, each older one has faded further (the eye reads these on a log scale); the old pages under them keep no ink at all
ALREADY_THERE = 3                           # rows on the sheet when the film reaches it; each blow adds the next

# Older pages under all of it, faintest first: every script the sheet has seen. Classics are only lines already checked against their source; the rest is ordinary writing made up for the film.
# Nothing here says 我: that word belongs to the next shot.
OLD = [
    dict(font="song", em=5.6, rot=-.8, lead=8.0, lines=["本院认为，被告人的行为已构成盗窃罪，依法应予惩处。", "判决如下：被告人犯盗窃罪，判处有期徒刑一年。", "如不服本判决，可在接到判决书的第二日起十日内提出上诉。", "审判长、审判员、书记员。"]),
    dict(font="hiero", em=11.0, rot=1.3, lead=17.0, thick=.14, lines=[hieroglyphs(*PTOLEMY)]),
    dict(font="hei", em=4.8, rot=1.1, lead=7.0, lines=["今收到货款叁仟元整。", "米二十斤，盐一包，灯油半瓶。", "本月结余四十二元。", "经手人：", "下月初还清，不另立字据。"]),
    dict(font="song", em=13.0, rot=-1.4, lead=16.0, vertical=True, lines=["烽火连三月，家书抵万金", "念天地之悠悠，独怆然而涕下", "不以物喜，不以己悲", "十年生死两茫茫", "问君能有几多愁", "此恨绵绵无绝期"]),
    dict(font="cune", em=8.0, rot=-.5, lead=13.0, lines=[GILGAMESH]),
    dict(font="baskr", em=8.5, rot=.6, lead=11.0, lines=["Dear Anna, the rain has not stopped since you left.", "Please keep the receipt. The minutes of the last meeting are enclosed.", "Yours, as ever,"]),
    dict(font="hei", em=7.0, rot=-.5, lead=10.5, lines=["楼主说得不对，先看原文再来吵。", "沙发。", "有人吗？", "回复(0)", "这也能吵起来？", "三楼说得对，顶。"]),
    dict(font="baskr", em=6.4, rot=-1.1, lead=9.5, lines=[HOMER]),
    dict(font="kai", em=10.0, rot=1.6, lead=14.0, lines=["见字如面。", "你走以后，家里一切都好，勿念。", "今天下雨了，想起你说过的那家面馆。", "天冷了，记得加衣。", "窗台的花还开着。", "等你回来，再一起去看海。"]),
    dict(font="monor", em=5.2, rot=-.3, lead=7.4, lines=["if (count == 0) {", "    return NULL;", "}", "for (int i = 0; i < n; i++) {", "    total += page[i];", "}", "#include <stdio.h>", "main()"]),
    dict(font="song", em=6.2, rot=.9, lead=9.0, lines=["少盐，多放葱，小火炖半个时辰。", "每日三次，每次两片，饭后服用。", "请勿靠近火源。", "开水冲泡，趁热喝。"]),
    dict(font="baskr", em=7.0, rot=.4, lead=12.0, lines=[SHAKESPEARE]),
    dict(font="songb", em=14.0, rot=-.9, lead=30.0, lines=["长太息以掩涕兮", "岂不痛哉", "昔者苍颉作书，而天雨粟，鬼夜哭", "本院认为", "判决如下"]),
]
OLD_DEPTHS = (.022, .026, .026, .03, .03, .034, .036, .04, .042, .045, .05, .05, .06)    # the oldest are almost gone

for _layer in OLD:
    assert not any("我" in s for s in _layer["lines"])
assert not any("我" in r[0] for r in ROWS)


def check_glyphs():
    """Every character of every row against its font's .notdef: no missing-glyph boxes may reach the paper."""
    def bad(key, text):
        f = font(key, 64)
        box = bytes(f.getmask("\U000FFFFD"))
        return [c for c in set(text) if not c.isspace() and (bytes(f.getmask(c)) == box or f.getmask(c).getbbox() is None)]
    pairs = [(l["font"], "".join(l["lines"])) for l in OLD] + [(r[1], r[0]) for r in ROWS] + [("hiero", hieroglyphs(*PTOLEMY) + "".join(RING))]
    missing = {k: bad(k, t) for k, t in pairs if bad(k, t)}
    assert not missing, missing
    print("glyphs: all present in", sorted({k for k, _ in pairs}), flush=True)


# ---- small tools -------------------------------------------------------------------------------------------------------------------------------------
def spectral(n, beta, seed, m=None):
    """Tileable noise with a 1/f^beta spectrum, mean 0, std 1."""
    m = m or n
    r = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(m)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1
    s = (r.normal(size=f.shape) + 1j * r.normal(size=f.shape)) / f ** (beta / 2)
    s[0, 0] = 0
    v = np.fft.irfft2(s, s=(n, m))
    return ((v - v.mean()) / v.std()).astype(np.float32)


def to_px(x, y, px):
    return (x - AREA[0]) * px, (AREA[3] - y) * px


def canvas(px):
    from PIL import Image
    return Image.new("L", (int((AREA[1] - AREA[0]) * px), int((AREA[3] - AREA[2]) * px)), 0)


def gauss(a, mm, px):
    from scipy.ndimage import gaussian_filter
    s = mm * px
    return gaussian_filter(a, s, truncate=3.0) if s >= .25 else a


# ---- rasters ---------------------------------------------------------------------------------------------------------------------------------------
def old_layer(layer, seed, px):
    """A page of old printing, line after line until the sheet is covered, slightly askew as every pass through the press is."""
    from PIL import Image, ImageDraw
    im = canvas(px)
    d = ImageDraw.Draw(im)
    f = font(layer["font"], layer["em"] * px)
    r = np.random.default_rng(seed)
    lines, lead, em = layer["lines"], layer["lead"], layer["em"]
    if layer.get("vertical"):
        x, i = OLD_BOX[1] - 4, 0
        while x > OLD_BOX[0]:
            y = OLD_BOX[2] - r.uniform(2, 30)
            line = lines[i % len(lines)]
            while y > AREA[2]:
                for ch in line:
                    if ch in "，。":
                        y -= em * .6
                        continue
                    if AREA[0] - em < x < AREA[1] + em and y < AREA[3] + em:
                        d.text(to_px(x, y - em / 2, px), ch, 255, font=f, anchor="mm")
                    y -= em * 1.08
                y -= em * 1.6
            x -= lead
            i += 1
    else:
        y, i = OLD_BOX[2] - 3, int(r.integers(0, len(lines)))
        while y > AREA[2]:
            x = OLD_BOX[0] + r.uniform(0, 6)
            on = y < AREA[3] + em
            line = "    ".join(lines[(i + k) % len(lines)] for k in range(40)) if layer["font"] != "monor" else None
            if line is None:
                # code keeps its lines: one line of text per row, no running on
                line = lines[i % len(lines)]
                x = -150 + (i // len(lines)) % 3 * 95
                for off in (0, 95, -95, 190):
                    if on:
                        d.text(to_px(x + off, y, px), line, 255, font=f, anchor="ls")
            elif on:
                d.text(to_px(x, y, px), line, 255, font=f, anchor="ls")
            y -= lead
            i += 1
    if layer.get("rot"):
        im = im.rotate(layer["rot"], resample=Image.BICUBIC, center=to_px(0, 0, px))
    a = np.asarray(im, np.float32) / 255
    if layer.get("thick"):
        from scipy.ndimage import maximum_filter
        a = maximum_filter(a, int(layer["thick"] * px) * 2 + 1)
    return a


class Forme:
    """All six rows, locked up. .mask is the printing face in sheet coordinates, .level each sort's small difference in height (1 is the nominal face), .row_masks one mask per row, .sorts their boxes."""

    def __init__(self, px, seed=FORME_SEED):
        self.px, self.seed = px, seed
        r = np.random.default_rng(seed)
        from PIL import Image, ImageDraw
        from scipy.ndimage import gaussian_filter
        self.mask = np.zeros((int((AREA[3] - AREA[2]) * px), int((AREA[1] - AREA[0]) * px)), np.float32)
        self.level = np.ones_like(self.mask)
        self.sorts = []
        self.row_masks = []
        wear = spectral(512, 1.1, seed + 1)
        for k, (text, key, em, x0, y0) in enumerate(ROWS):
            f = font(key, em * px)
            start = self.mask.copy()
            for i, ch in enumerate(text):
                x = x0 + f.getlength(text[:i]) / px
                adv = (f.getlength(text[:i + 1]) - f.getlength(text[:i])) / px
                sort = dict(x0=x, x1=x + adv, y0=y0 - em * .28, y1=y0 + em * .95, em=em, ch=ch)
                if not ch.isspace():
                    size = int(max(em * 1.6, adv * 1.25) * px) | 1
                    p = Image.new("L", (size, size), 0)
                    ImageDraw.Draw(p).text((size / 2, size / 2 + em * .28 * px), ch, 255, font=f, anchor="ms")
                    rot = float(r.normal(0, .12))
                    p = p.rotate(rot, resample=Image.BICUBIC)
                    a = np.asarray(p, np.float32) / 255
                    # a worn face: corners a little round, the edge very slightly uneven, never broken
                    a = gaussian_filter(a, max(.03 * px, .4))
                    ox, oy = r.integers(0, 512, 2)
                    yy, xx = np.mgrid[:size, :size]
                    n = wear[(yy // 2 + oy) % 512, (xx // 2 + ox) % 512]
                    a = np.clip((a - .5 - n * .02) * max(.35 * px, 2) + .5, 0, 1)
                    lv = float(np.clip(r.normal(1, .04), .9, 1.08))
                    dx, dy = float(r.normal(0, .02)), float(r.normal(0, .02))
                    sort.update(key=key, baseline=y0, rot=rot, lv=lv, cx=x + adv / 2 + dx, cy=y0 + em * .28 + dy)
                    cx = (sort["cx"] - AREA[0]) * px
                    cy = (AREA[3] - sort["cy"]) * px
                    t, l = int(cy - size / 2), int(cx - size / 2)
                    sl = np.s_[max(t, 0):t + size, max(l, 0):l + size]
                    sa = np.s_[max(-t, 0):size + min(self.mask.shape[0] - t - size, 0), max(-l, 0):size + min(self.mask.shape[1] - l - size, 0)]
                    self.mask[sl] = np.maximum(self.mask[sl], a[sa])
                    rt, rb = int((AREA[3] - sort["y1"]) * px), int((AREA[3] - sort["y0"]) * px)
                    rl, rr = int((x - AREA[0]) * px), int((x + adv - AREA[0]) * px)
                    self.level[max(rt, 0):max(rb, 0), max(rl, 0):max(rr, 0)] = lv
                self.sorts.append(sort)
            self.row_masks.append(np.maximum(self.mask - start, 0))
        ring, box = name_ring(px)
        self.mask = np.maximum(self.mask, ring)
        self.row_masks[0] = np.maximum(self.row_masks[0], ring)
        self.sorts.append(dict(x0=box[0], x1=box[1], y0=box[2], y1=box[3], em=RING_EM, ch="ring"))


def solid_glyph(key, ch, em, px, grow=.14, carve=.13, fill=True):
    """A sign cut as a solid block: the font's outline filled in, its inner lines kept as thin cuts, so that the lion and the reeds print as shapes and not as hairlines. Returns (array, advance in mm)."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt, gaussian_filter
    f = font(key, em * px)
    size = int(em * 2.6 * px) | 1
    p = Image.new("L", (size, size), 0)
    ImageDraw.Draw(p).text((size * .1, size * .7), ch, 255, font=f, anchor="ls")
    line = np.asarray(p, np.float32) > 80
    solid = binary_fill_holes(line) if fill else line
    inner = line & binary_erosion(solid, iterations=max(int(.012 * em * px) + 2, 2)) if fill else np.zeros_like(line)
    cut = distance_transform_edt(~inner) < carve * px / 2 if inner.any() else np.zeros_like(line)
    fat = distance_transform_edt(~solid) < grow * px
    return gaussian_filter((fat & ~cut).astype(np.float32), max(.03 * px, .4)), f.getlength(ch) / px


def paste(mask, a, x, baseline, px):
    """Puts the output of solid_glyph() with its pen origin at (x, baseline) in mm."""
    size = a.shape[0]
    l, t = int((x - AREA[0]) * px - size * .1), int((AREA[3] - baseline) * px - size * .7)
    sl = np.s_[max(t, 0):t + size, max(l, 0):l + size]
    sa = np.s_[max(-t, 0):size + min(mask.shape[0] - t - size, 0), max(-l, 0):size + min(mask.shape[1] - l - size, 0)]
    mask[sl] = np.maximum(mask[sl], a[sa])


def name_ring(px):
    """Row 1's left half: Ptolemy's name inside its ring, cut as one block. The ring's two ends are the font's own two signs, enlarged to hold the name; the font has no middle for them, so straight rails join them. Returns (mask, block's box in mm)."""
    mask = np.zeros((int((AREA[3] - AREA[2]) * px), int((AREA[1] - AREA[0]) * px)), np.float32)
    em, left = RING_EM, LEFT
    ring_em = em * RING_K
    bottom = Y1 - .42 * em - RING_FOOT * ring_em            # the ends' pen line: the font draws them this far above it
    start, adv_end = solid_glyph("hiero", RING[0], ring_em, px, grow=.05, carve=0.0, fill=False)
    end, _ = solid_glyph("hiero", RING[1], ring_em, px, grow=.05, carve=0.0, fill=False)
    x = left + .95 * em
    signs = []
    for name in PTOLEMY:
        a, adv = solid_glyph("hiero", hieroglyphs(name), em, px)
        signs.append((a, x))
        x += adv + .05 * em
    right = x + 1.0 * em - adv_end                          # the right end's inner curve just clears the last sign
    paste(mask, start, left, bottom, px)
    end[:, :int(end.shape[0] * .1 + adv_end * px * .12)] = 0        # the font closes the end's rails with a stroke: the ring runs on through it
    paste(mask, end, right, bottom, px)
    # the rails: where the left end's own rails leave it
    cols = np.where(start.max(axis=0) > .5)[0]
    rows = np.where(start[:, cols[-1] - 2] > .5)[0]
    runs = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    top_px = int((AREA[3] - bottom) * px - start.shape[0] * .7)
    l, r_ = int((left + adv_end * .7 - AREA[0]) * px), int((right + adv_end * .3 - AREA[0]) * px)
    for run in runs:
        mask[top_px + run[0]:top_px + run[-1] + 1, l:r_] = 1
    for a, sx in signs:
        paste(mask, a, sx, Y1, px)
    print(f"name ring spans x {left:.1f} .. {right + adv_end:.1f}; cuneiform starts at {CUNE_X}", flush=True)
    return mask, (left - .4, right + adv_end + .4, Y1 - .42 * em - .8, Y1 - .42 * em + .41 * ring_em + .8)


def press(h, mask, depth, px, level=None, edge=.03, pillow=.35, pillow_k=.14):
    """One pass: the flat face pushes everything it touches down to its own depth, older marks underneath flattened unless they already go deeper; around each stroke the sheet is dragged down a little."""
    m = gauss(mask, edge, px)
    target = -depth * (1 if level is None else level)
    out = h * (1 - m) + np.minimum(h, target) * m
    out -= depth * pillow_k * gauss(mask, pillow, px) * (1 - m)
    return out


def age(h, px, keep=.965, blur=.05):
    """Paper springs back: earlier marks grow shallower and rounder with every later pass."""
    return (h * .5 + gauss(h, blur, px) * .5) * keep


def press_paper(px, old=OLD, depths=OLD_DEPTHS, forme=None, log=print):
    """Yields (name, height map in mm, ink density) after the old pages and the rows already there, then after each blow, oldest first. Old pages are soft and shallow, pressed long ago, and have kept no ink; the arrays are reused between yields."""
    forme = forme or Forme(px)
    h = np.zeros((int((AREA[3] - AREA[2]) * px), int((AREA[1] - AREA[0]) * px)), np.float32)
    ink = np.zeros_like(h)
    for k, (layer, d) in enumerate(zip(old, depths)):
        h = age(h, px, blur=.14)
        h = press(h, old_layer(layer, 200 + k, px), d, px, edge=.12 + .008 * (len(old) - k), pillow=.5, pillow_k=.06)
        log(f"old layer {k}: {layer['font']} {layer['em']} mm")
    for k, m in enumerate(forme.row_masks):
        h = age(h, px, keep=.985, blur=.03) if k < ALREADY_THERE else age(h, px)
        h = press(h, m, DEPTH[k], px, level=forme.level, edge=.04 if k < ALREADY_THERE else .03)
        ink = np.maximum(ink, INK_DENSITY[k] * gauss(m, .025, px))
        log(f"row {k + 1}")
        if k >= ALREADY_THERE - 1:
            yield f"s{k + 1 - ALREADY_THERE}", h, ink


def fibres(n, count, seed, length=(40, 300), width=(2, 3), bright=(.35, 1.0), wiggle=.28):
    """Cotton fibres: curling lines drawn at random onto a tile, heights between bright[0] and bright[1]."""
    from PIL import Image, ImageDraw
    r = np.random.default_rng(seed)
    pad = length[1] // 2
    im = Image.new("L", (n + 2 * pad, n + 2 * pad), 0)
    d = ImageDraw.Draw(im)
    for _ in range(count):
        L = r.uniform(*length)
        k = max(int(L / 12), 3)
        a = r.uniform(0, 2 * math.pi)
        turn = np.cumsum(r.normal(0, wiggle, k)) + a
        xs = np.cumsum(np.cos(turn)) * L / k + r.uniform(0, n) + pad
        ys = np.cumsum(np.sin(turn)) * L / k + r.uniform(0, n) + pad
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=int(r.uniform(*bright) * 255), width=int(r.integers(width[0], width[1] + 1)))
    a = np.asarray(im, np.float32) / 255
    out = a[pad:pad + n, pad:pad + n].copy()
    out[:pad, :] = np.maximum(out[:pad, :], a[n + pad:n + 2 * pad, pad:pad + n])
    out[-pad:, :] = np.maximum(out[-pad:, :], a[:pad, pad:pad + n])
    out[:, :pad] = np.maximum(out[:, :pad], a[pad:pad + n, n + pad:n + 2 * pad])
    out[:, -pad:] = np.maximum(out[:, -pad:], a[pad:pad + n, :pad])
    return out


FELT_PX = 60        # samples per mm of the fibre tiles
FELT_N = 4096


def felt_tiles():
    """The surface of thick cotton rag: a felt of fibres over soft cloud. Returns (height tile in units of about one fibre, dark fibres tile, cloud tile at 3 px/mm)."""
    from scipy.ndimage import gaussian_filter
    os.makedirs(MAPS, exist_ok=True)
    p = os.path.join(MAPS, "felt2.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["felt"], z["dark"], z["cloud"], z["thread"]
    fine = fibres(FELT_N, 140000, 5, length=(30, 220), width=(1, 2))
    long_ = fibres(FELT_N, 5000, 6, length=(200, 620), width=(2, 3), wiggle=.2)
    felt = gaussian_filter(fine, .6, mode="wrap") * .8 + gaussian_filter(long_, .7, mode="wrap") * .35 + spectral(FELT_N, 2.2, 7) * .12 + spectral(FELT_N, .6, 8) * .05
    felt -= felt.mean()
    felt /= felt.std()
    dark = gaussian_filter(fibres(FELT_N, 2600, 9, length=(120, 520), width=(2, 4), wiggle=.22), .7, mode="wrap")
    thread = gaussian_filter(fibres(FELT_N, 2400, 12, length=(180, 620), width=(3, 4), bright=(.45, 1.0), wiggle=.17), 1.0, mode="wrap")
    cloud = spectral(1024, 2.6, 10)
    np.savez(p, felt=felt.astype(np.float32), dark=dark.astype(np.float32), cloud=cloud, thread=thread.astype(np.float32))
    return felt, dark, cloud, thread


def png16(path, a):
    """An H x W x 3 array of values in 0-1 as a 16-bit PNG, which PIL can't write."""
    import struct
    import zlib
    h, w = a.shape[:2]
    rows = np.frombuffer((np.clip(a, 0, 1) * 65535 + .5).astype(">u2").tobytes(), np.uint8).reshape(h, w * 6)
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rows], 1).tobytes()
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


def fibre_tile():
    """What the sheet's shader adds on top of its grid, which is too coarse to hold a single fibre: red the fine felt, green the longer strands that stand out of it, blue the darker ones. Seen at 2x these are still fibres, not a blur."""
    from scipy.ndimage import gaussian_filter
    felt = felt_tiles()[0]
    strands = gaussian_filter(fibres(FELT_N, 11000, 41, length=(70, 340), width=(4, 7), bright=(.3, 1.0), wiggle=.26), 1.0, mode="wrap")
    dark = gaussian_filter(fibres(FELT_N, 1700, 42, length=(60, 280), width=(3, 5), bright=(.3, 1.0), wiggle=.3), .8, mode="wrap")
    png16(os.path.join(MAPS, "fibre_tile.png"), np.stack([felt / 8 + .5, strands, dark], -1))
    print("fibre tile", flush=True)


def camera_axes(cam):
    loc, tgt = np.array(cam["loc"], float), np.array(cam["target"], float)
    look = tgt - loc
    look /= np.linalg.norm(look)
    right = np.cross(look, cam.get("up", (0, 0, 1)))
    right /= np.linalg.norm(right)
    return loc, look, right, np.cross(right, look)


def project(cam, x, y, z):
    """World point to frame pixels (1920 x 1080). The camera looks at its target, and a lens shift puts the target at height cy."""
    loc, look, right, up = camera_axes(cam)
    d = np.array([x, y, z], float) - loc
    w = d @ look
    k = cam["lens"] / w * W / cam["sensor"]
    return W / 2 + d @ right * k, cam["cy"] - d @ up * k


def sheet_grid(cam, dx=.09, dpx=.62, sheet=SHEET):
    """A grid in world axes over the sheet: columns even, rows as dense as this camera sees them (about dpx of a 1080p pixel apart), coarse where it doesn't see the sheet at all. Returns X, Y in mm."""
    xs = np.arange(sheet["left"] - 2, sheet["right"] + 2 + dx, dx, dtype=np.float32)
    ys, y = [], sheet["near"] - 2
    while y < sheet["far"] + 2:
        ys.append(y)
        a, b = project(cam, 0, y, 0)[1], project(cam, 0, y + .01, 0)[1]
        step = float(np.clip(dpx / max(abs(a - b) / .01, 1e-3), .03, .2))
        if abs(y - sheet["near"]) < 3 or abs(y - sheet["far"]) < 3:     # its own edges, which the camera sees when it is low
            step = min(step, .08)
        y += step
    X, Y = np.meshgrid(xs, np.array(ys, np.float32))
    return X, Y


def pyramid(a, levels):
    out = [a]
    for _ in range(levels):
        p = out[-1]
        h, w = p.shape[0] & ~1, p.shape[1] & ~1
        out.append((p[:h:2, :w:2] + p[1:h:2, :w:2] + p[:h:2, 1:w:2] + p[1:h:2, 1:w:2]) * .25)
    return out


def sample(pyr, c, r, lod, wrap=False):
    """Bilinear lookup at (column, row) in level-0 pixels, blended between levels by lod, so that far and fast-receding parts don't sparkle."""
    from scipy.ndimage import map_coordinates
    lod = np.clip(lod, 0, len(pyr) - 1 - 1e-3)
    lo = np.floor(lod).astype(np.int32)
    out = np.zeros(c.shape, np.float32)
    for l in range(len(pyr) - 1):
        for level, weight in ((l, 1 - (lod - lo)), (l + 1, lod - lo)):
            where = lo == l
            if not where.any():
                continue
            s = .5 ** level
            coords = np.stack([r[where] * s, c[where] * s])
            out[where] += weight[where] * map_coordinates(pyr[level], coords, order=1, mode="grid-wrap" if wrap else "nearest").astype(np.float32)
    return out


def edge_distance(X, Y, seed=31, sheet=SHEET):
    """Signed distance in mm inside the sheet's soft deckled edges, all four."""
    from scipy.ndimage import gaussian_filter1d
    r = gaussian_filter1d(spectral(4096, 2.4, seed, 8)[:, 0], 6.0, mode="wrap")
    r2 = gaussian_filter1d(spectral(4096, 1.2, seed + 1, 8)[:, 0], 3.0, mode="wrap")
    i = ((Y * 8).astype(np.int32)) % 4096
    j = ((X * 8).astype(np.int32)) % 4096
    fray = lambda k, shift: r[(k + shift) % 4096] * .55 + r2[(k + shift) % 4096] * .08
    return np.minimum.reduce([sheet["right"] + fray(i, 0) - X, X - sheet["left"] + fray(i, 900), sheet["far"] + fray(j, 0) - Y, Y - sheet["near"] + fray(j, 1700)])


PAPER = np.array((.90, .855, .72), np.float32)
INK = np.array((.010, .010, .012), np.float32)


def compose(X, Y, h, ink, px, felt, dark, cloud, thread, felt_amp=.0075, thread_amp=.007, sheet=SHEET):
    """The grid's final surface: pressed height, felt (flattened where the type has pressed), the sheet's edges. Returns z, colour with the ink in its alpha, pressed amount."""
    pyr = pyramid(h, 4)
    ipyr = pyramid(ink, 4)
    fpyr, dpyr, tpyr = pyramid(felt, 5), pyramid(dark, 5), pyramid(thread, 5)
    # footprint of one grid cell
    dy = np.abs(np.gradient(Y, axis=0))
    dx = np.abs(np.gradient(X, axis=1))
    cell = np.maximum(dx, dy)
    lod_h = np.log2(np.maximum(cell * px, 1))
    lod_f = np.log2(np.maximum(cell * FELT_PX, 1))
    c, r = (X - AREA[0]) * px, (AREA[3] - Y) * px
    tz = sample(pyr, c, r, lod_h)
    ik = sample(ipyr, c, r, lod_h)
    fc, fr = (X * FELT_PX) % FELT_N, (Y * FELT_PX) % FELT_N
    f = sample(fpyr, fc, fr, lod_f, wrap=True)
    d = sample(dpyr, fc + 777, fr + 333, lod_f, wrap=True)
    th = sample(tpyr, fc + 1999, fr + 1234, lod_f, wrap=True)
    cl = sample([cloud, cloud], (X * 3) % 1024, (Y * 3) % 1024, np.zeros_like(X), wrap=True)
    pressed = sm(-tz / .10)
    # far texture is averaged down by lod, and flattens: put a little of its energy back, or the far paper looks like plastic
    flat = np.clip(lod_f / 5, 0, 1)
    z = tz + (f * felt_amp * (1 + flat) + th * thread_amp) * (1 - .72 * pressed) + cl * .016
    e = edge_distance(X, Y, sheet=sheet)
    inside = sm(e / .75)
    z = z * inside - (sheet["thick"] + .2) * (1 - inside)
    v = 1 + cl * .012 + f * .032 + th * .03
    col = PAPER[None, None] * v[..., None]
    col = col * (1 - .06 * pressed[..., None])
    col = col + (INK - col) * (d[..., None] * .12)
    ik = ik * (1 - .2 * sm((f - .5) / 1.6))           # the paper's tooth shows through the ink
    col = col + (INK - col) * ik[..., None]
    col = np.concatenate([col, ik[..., None]], -1).astype(np.float32)        # alpha carries the ink: paper_material() stops the sheet glowing through it
    return z.astype(np.float32), col, pressed.astype(np.float32)


# ---- the metal ------------------------------------------------------------------------------------------------------------------------------------------
def face_distance(forme):
    """The distance in mm from every point of the forme to the nearest printing face, at FACE_PX: each sort drawn again from its font, large and with no wear on it, in exactly the place Forme() set it. Cast type has its chamfer cut straight, whatever the paper shows of a worn edge."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    px = FACE_PX
    x0, x1, y0, y1 = FORME_AREA
    dist = np.full((int(round((y1 - y0) * px)), int(round((x1 - x0) * px))), 9.0, np.float32)

    def stamp(d, l, t):
        ny, nx = dist.shape
        sl = np.s_[max(t, 0):min(t + d.shape[0], ny), max(l, 0):min(l + d.shape[1], nx)]
        sa = np.s_[max(-t, 0):max(-t, 0) + sl[0].stop - sl[0].start, max(-l, 0):max(-l, 0) + sl[1].stop - sl[1].start]
        dist[sl] = np.minimum(dist[sl], d[sa])

    fonts = {}
    pad = int(px)
    for s in forme.sorts:
        if s["ch"].isspace() or s["ch"] == "ring":
            continue
        f = fonts.setdefault((s["key"], s["em"]), font(s["key"], s["em"] * px))
        size = (int(max(s["em"] * 1.6, (s["x1"] - s["x0"]) * 1.25) * px) + 2 * pad) | 1
        p = Image.new("L", (size, size), 0)
        ImageDraw.Draw(p).text((size / 2, size / 2 + s["em"] * .28 * px), s["ch"], 255, font=f, anchor="ms")
        a = gaussian_filter(np.asarray(p.rotate(s["rot"], resample=Image.BICUBIC), np.float32) / 255, .015 * px)
        d = (distance_transform_edt(a < .5) / px).astype(np.float32)
        stamp(d, int(round((s["cx"] - x0) * px - size / 2)), int(round((y1 - s["cy"]) * px - size / 2)))
    ring, box = name_ring(px)
    c0, c1 = int((box[0] - 1 - AREA[0]) * px), int((box[1] + 1 - AREA[0]) * px)
    r0, r1 = int((AREA[3] - box[3] - 1) * px), int((AREA[3] - box[2] + 1) * px)
    d = (distance_transform_edt(ring[r0:r1, c0:c1] < .5) / px).astype(np.float32)
    del ring
    stamp(d, int(round(c0 + (AREA[0] - x0) * px)), int(round(r0 - (AREA[3] - y1) * px)))
    return dist


def block_mean(a, k):
    h, w = a.shape[0] // k, a.shape[1] // k
    return a[:h * k, :w * k].reshape(h, k, w, k).mean((1, 3))


def forme_face(forme):
    """The underside of the forme on a regular grid in sheet coordinates: z above the printing plane, ink on the faces in 0-1, the bare metal's lightness (see type_material()). Flat faces, a chamfer of one constant slope, straight wherever the stroke is straight, then the shoulder."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import gaussian_filter
    res = FORME_RES
    x0, x1, y0, y1 = FORME_AREA
    dist = face_distance(forme)
    k = FACE_PX // res
    sh = np.zeros_like(dist)
    for s in forme.sorts:
        l, t, r_, b = [int(round(v * FACE_PX)) for v in (s["x0"] - x0, y1 - s["y1"], s["x1"] - x0, y1 - s["y0"])]
        sh[max(t, 0):max(b, 0), max(l, 0):max(r_, 0)] = .28 + .075 * s["em"]
    z_sort = block_mean(gaussian_filter(np.minimum(dist * SLOPE, sh), 1.0), k).astype(np.float32)      # a third of a sample of blur, only to take the pixel steps out of the field
    dist = block_mean(dist, k).astype(np.float32)
    del sh
    ny, nx = dist.shape
    sort_im, seam_im, level_im, tone_im = Image.new("L", (nx, ny), 0), Image.new("L", (nx, ny), 0), Image.new("F", (nx, ny), 0.0), Image.new("F", (nx, ny), .35)
    ds_, dl, dv, dt = ImageDraw.Draw(sort_im), ImageDraw.Draw(seam_im), ImageDraw.Draw(level_im), ImageDraw.Draw(tone_im)
    r = np.random.default_rng(forme.seed + 9)
    # leads across the whole forme, quads broken up along them
    y = y0
    while y < y1:
        v = (y1 - y) * res
        dl.line([0, v, nx, v], fill=170, width=2)
        x = x0 + float(r.uniform(0, 30))
        while x < x1:
            dl.line([(x - x0) * res, v, (x - x0) * res, v - float(r.uniform(3, 7)) * res], fill=170, width=2)
            x += float(r.uniform(12, 46))
        y += float(r.uniform(3.2, 7.0))
    for s in forme.sorts:
        box = [(s["x0"] - x0) * res, (y1 - s["y1"]) * res, (s["x1"] - x0) * res, (y1 - s["y0"]) * res]
        ds_.rectangle(box, fill=255)
        dv.rectangle(box, fill=(1 - s.get("lv", 1.0)) * .35)
        dt.rectangle(box, fill=float(r.uniform(.72, 1.0)))
        dl.rectangle(box, fill=0, outline=255, width=2)
    is_sort = np.asarray(sort_im, np.float32) / 255
    seam = np.asarray(seam_im, np.float32) / 255
    tone = np.asarray(tone_im, np.float32)
    z_sort = z_sort + np.asarray(level_im, np.float32)
    z = z_sort * is_sort + BLANK * (1 - is_sort)
    z = z + seam * .4
    z = z + spectral(max(nx, ny), 2.6, forme.seed + 3)[:ny, :nx] * .0012         # any more of this bends the metal's reflections into clouds
    grime = sm((spectral(max(nx, ny), 3.0, forme.seed + 5)[:ny, :nx] - .1) / 1.6)      # old ink never quite wiped off, most of all in what doesn't print
    tone = tone * (1 - .3 * grime * (.2 - .15 * is_sort))
    ink = sm(1 - dist / .05) * is_sort
    mott = spectral(max(nx, ny), 1.4, forme.seed + 4)[:ny, :nx]
    ink = np.clip(ink * (1 - .25 * sm((mott - 1.1) / .8)) + sm(1 - dist / .14) * .3 * is_sort, 0, 1)
    ink = np.maximum(ink, sm(1 - dist / .46) * .62 * is_sort)     # ink collects on the slopes round every stroke, and takes the shine off them: type, not chrome letters
    # a little old ink left in the corners of the forme and along its edges, where wiping doesn't reach
    gx, gy = np.meshgrid((np.arange(nx) + .5) / res + x0, y1 - (np.arange(ny) + .5) / res)
    to_edge = np.minimum.reduce([gx - x0, x1 - gx, gy - y0, y1 - gy])
    to_corner = np.hypot(np.minimum(gx - x0, x1 - gx), np.minimum(gy - y0, y1 - gy))
    marks = sm(1 - to_corner / 9) * .8 + sm(1 - to_edge / 1.6) * .35
    marks = marks * sm((spectral(max(nx, ny), 2.0, forme.seed + 6)[:ny, :nx] + .2) / 1.0) * (1 - is_sort)
    ink = np.maximum(ink, marks * .7)
    return z.astype(np.float32), ink.astype(np.float32), (tone * (1 - seam * .85)).astype(np.float32)


# ---- baking ---------------------------------------------------------------------------------------------------------------------------------------------
def bake_face(forme):
    z, ink, tone = forme_face(forme)
    np.savez(os.path.join(MAPS, "forme_all.npz"), z=z, ink=ink, tone=tone)
    print("forme face", z.shape, flush=True)


def bake(px):
    os.makedirs(MAPS, exist_ok=True)
    check_glyphs()
    X, Y = sheet_grid(CAM)
    np.save(os.path.join(MAPS, "grid_x.npy"), X)
    np.save(os.path.join(MAPS, "grid_y.npy"), Y)
    print(f"grid {X.shape[1]} x {X.shape[0]}  = {X.size / 1e6:.1f} M vertices", flush=True)
    felt, dark, cloud, thread = felt_tiles()
    fibre_tile()
    forme = Forme(px)
    print("forme set", flush=True)
    bake_face(forme)
    for name, h, ink in press_paper(px, forme=forme, log=lambda s: print(s, flush=True)):
        z, col, pressed = compose(X, Y, h, ink, px, felt, dark, cloud, thread)
        np.save(os.path.join(MAPS, f"paper_{name}_z.npy"), z)
        np.save(os.path.join(MAPS, f"paper_{name}_col.npy"), col)
        np.save(os.path.join(MAPS, f"paper_{name}_pressed.npy"), pressed)
        if name == "s3":
            np.save(os.path.join(MAPS, "proof_h.npy"), h[::2, ::2])
        print("baked", name, flush=True)
    with open(os.path.join(MAPS, "meta.json"), "w") as f:
        json.dump(dict(px=px, cam=CAM), f)


def proof():
    """Top-down hillshade of the last state, lit from the upper left: it must read right way round."""
    from PIL import Image
    h = np.load(os.path.join(MAPS, "proof_h.npy"))
    gy, gx = np.gradient(h)
    shade = np.clip(.72 + (gx - gy) * 2.2, 0, 1)
    im = Image.fromarray((shade * 255).astype(np.uint8))
    im.thumbnail((2400, 2400))
    im.save(os.path.join(OUT, "proof_top.png"))
    print(os.path.join(OUT, "proof_top.png"))


def frame_report():
    """Numbers to compose by, no rendering."""
    c = camera_c()
    print("C cam", tuple(round(v, 1) for v in c["loc"]))
    for k, (x, y) in dict(sheet_near_l=(SHEET["left"], SHEET["near"]), sheet_near_r=(SHEET["right"], SHEET["near"]), sheet_far_l=(SHEET["left"], SHEET["far"]), sheet_far_r=(SHEET["right"], SHEET["far"]),
                          rim_near_l=(RIM["left"], RIM["near"]), rim_near_r=(RIM["right"], RIM["near"]), rim_far_l=(RIM["left"], RIM["far"]), rim_far_r=(RIM["right"], RIM["far"])).items():
        print(f"  {k:13s} {project(c, x, y, 0)[0]:6.0f} {project(c, x, y, 0)[1]:6.0f}")
    for k, y in dict(r1=Y1, r2=Y2, r3=Y3, r4=Y4, r5=Y5, r6=Y6).items():
        print(f"  {k} baseline y {project(c, LEFT, y, 0)[1]:5.0f} .. {project(c, -LEFT, y, 0)[1]:5.0f}   x {project(c, LEFT, y, 0)[0]:5.0f}   {(project(c, -LEFT, y, 0)[0] - project(c, LEFT, y, 0)[0]) / (2 * -LEFT):4.1f} px/mm")
    for z in (0, 40, 80, 100, 110, 120, 130, 140):
        print(f"  head at z {z:3d}: near lip screen y {project(c, 0, -38, z + .8)[1]:5.0f}, cornice lip {project(c, 0, -44, z + 22)[1]:5.0f}")
    print("B", {k: tuple(round(v, 1) for v in camera_b()[k]) for k in ("loc", "up")})
    a = camera_a(0)
    print("A cam", tuple(round(v, 1) for v in a["loc"]))
    for k, (x, y, z) in dict(front_foot=(157, -204, -300), back_foot=(157, 218, -300), crown=(0, 110, 212), wheel_top=(-250, 0, 176), wheel_ped=(-308, 92, -300), sheet_near=(0, -103, 0), sheet_far=(0, 66, 0), wall_floor=(0, 420, -300)).items():
        print(f"  A {k:12s} {project(a, x, y, z)[0]:6.0f} {project(a, x, y, z)[1]:6.0f}")


if __name__ == "__main__":
    def arg(name, default):
        return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
    if sys.argv[1] == "bake":
        bake(arg("--px", 30))
    elif sys.argv[1] == "face":
        bake_face(Forme(arg("--px", 30)))
    elif sys.argv[1] == "fibre":
        fibre_tile()
    elif sys.argv[1] == "proof":
        proof()
    elif sys.argv[1] == "frame":
        frame_report()
    elif sys.argv[1] == "row1":
        from PIL import Image
        check_glyphs()
        m = Forme(30).row_masks[0]
        t = int((AREA[3] - Y1 - 16) * 30)
        Image.fromarray((255 - np.clip(m[t:t + 26 * 30], 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, "row1.png"))
