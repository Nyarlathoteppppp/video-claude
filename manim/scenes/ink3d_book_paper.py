"""The paper side of the book (序 · 书 and 十四 · 诗与合书): everything Blender cannot do without PIL and scipy. Run in the render environment.

    python ink3d_book_paper.py tex [pages] [sheet] [cover] [desk]      # the pages, covers and desk -> ink/film_out/3d/book/tex/ (all of them if none is named)
    python ink3d_book_paper.py post SHOT [--gamma=G] [--contrast=C] [--creep=PX] [--bleed=U] [--keep=X] [--seed=N]      # SHOT.png -> SHOT_ink.png
    python ink3d_book_paper.py sheet SHOT SHOT ...     # -> sheet.png

The poem does not go in the book: it has a sheet of its own, new paper with a deckle edge, and at the end the ink goes from it and only the seal stays. Pages are drawn the way a woodblock prints: half a leaf is ten columns of twenty characters inside a double frame, no punctuation, read from the top
down and right to left, which is what teaches the viewer to read the film's subtitles. The sheet is new, and is written across. Measurements are in cm, at 220 pixels to the cm.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, map_coordinates, maximum_filter

from ink_film import OUT, chaikin, glyph_mask
from ink_grade import FLOOR, INK_K, PAPER, fft_noise, load, sm

DIR = os.path.join(OUT, "3d", "book")
TEX = os.path.join(DIR, "tex")
CM = 220
LEAF_W, LEAF_H = 13.4, 23.0          # hinge to fore-edge, head to tail
INNER, COL, GUTTER_STRIP = 1.2, 1.12, 1.0   # margin at the hinge, one column, half of the 版心 at the fold
TOP, PITCH = 3.4, .9                 # margin at the head, one character
SIZE = (int(LEAF_W * CM), int(LEAF_H * CM))

# 《淮南子·本经训》, wikisource, punctuation removed
TEXT = ("可以曆推得也雷震之聲可以鼓鐘寫也風雨之變可以音律知也是故大可睹者可得而量也明可見者可得而蔽也聲可聞者可得而調也色可察者可得而別也"
        "夫至大天地弗能含也至微神明弗能領也及至建律曆別五色異清濁味甘苦則樸散而為器矣立仁義修禮樂則德遷而為偽矣及偽之生也飾智以驚愚設詐以巧上"
        "天下有能持之者有能治之者也"
        "昔者蒼頡作書而天雨粟鬼夜哭伯益作井而龍登玄雲神棲昆侖能愈多而德愈薄矣故周鼎著倕使銜其指以明大巧之不可為也"
        "故至人之治也心與神處形與性調靜而體德動而理通隨自然之性而緣不得已之化洞然無為而天下自和憺然無欲而民自樸無禨祥而民不夭不忿爭而養足"
        "兼包海內澤及後世不知為之誰何是故生無號死無諡實不聚而名不立施者不德受者不讓德交歸焉而莫之充忍也故德之所總道弗能害也智之所不知辯弗能解也"
        "不言之辯不道之道若或通焉謂之天府取焉而不損酌焉而不竭莫知其所由出是謂瑤光瑤光者資糧萬物者也振困窮補不足則名生興利除害伐亂禁暴則功成"
        "世無災害雖神無所施其德上下和輯雖賢無所立其功昔容成氏之時道路雁行列處托嬰兒于巢上置餘糧於畮首虎豹可尾虺蛇可蹍而不知其所由然"
        "逮至堯之時十日並出焦禾稼殺草木而民無所食猰貐鑿齒九嬰大風封豨修蛇皆為民害堯乃使羿誅鑿齒于疇華之野殺九嬰于凶水之上繳大風於青丘之澤"
        "上射十日而下殺猰貐斷修蛇於洞庭禽封豨于桑林萬民皆喜置堯以為天子於是天下廣狹險易遠近始有道裡舜之時共工振滔洪水以薄空桑龍門未開呂梁未發"
        "江淮通流四海溟涬民皆上丘陵赴樹木舜乃使禹疏三江五湖辟開伊闕導廛澗平通溝陸流注東海鴻水漏九州幹萬民皆寧其性是以稱堯舜以為聖"
        "晚世之時帝有桀紂為旋室瑤台象廊玉床紂為肉圃酒池燎焚天下之財罷苦萬民之力刳諫者剔孕婦攘天下虐百姓於是湯乃以革車三百乘伐桀于南巢"
        "放之夏台武王四卒三千破紂牧野殺之于宣室天下甯定百姓和集是以稱湯武之賢由此觀之有賢聖之名者必遭亂世之患也今至人生亂世之中含德懷道"
        "拘無窮之智鉗口寢說遂不言而死者眾矣然天下莫知貴其不言也故道可道非常道名可名非常名著于竹帛鏤于金石可傳於人者其粗也五帝三王殊事而同指異路而同歸")
SHEET_W, SHEET_H, SCM = 30.0, 23.0, 150         # the poem's own sheet
FADE_T = 2.6                                    # the fading frame: the title and the first couplet have gone, the second is going
SHADOW_PAD = 1.5
POEM = ["鬼哭当年缘底事，早知今夜字成精。", "千窗有问千窗应，一念无存一念生。", "说痛疑偷前代句，言空恐负片时情。", "人神鬼器皆非我，不借尘寰旧姓名。"]
PUNCT = "，。"
# where things are on the sheet, in cm from its top left. It reads as a page does now: across, then down
LINE_X, CHAR_W, PUNCT_W = 2.65, 1.68, .6        # where a line starts, and how wide a character and a mark of punctuation are
LINE_W = 14 * CHAR_W + 2 * PUNCT_W
LINE_YS = (7.1, 10.0, 12.9, 15.8)
TITLE_Y, TITLE_SIZE = 3.3, 1.72
TITLE = (("《", 1.2), ("AI", 1.95), ("自", 1.9), ("传", 1.9), ("》", 1.2))     # and how wide each is
SIGN = "Opus5.5"
SIGN_X, SIGN_Y, SIGN_SIZE = 23.25, 19.15, .74
SEAL_S = 1.7
SEAL_X, SEAL_Y = LINE_X + LINE_W - .15 - SEAL_S / 2, SIGN_Y - SEAL_S / 2          # the seal's centre across, and its top
ROWS = np.array([TITLE_Y, *LINE_YS, SIGN_Y], np.float32)                          # washed() counts these, and the writing goes down them


def cells_x(widths, x0):
    """The middles of cells of these widths, starting at x0."""
    return (x0 + np.cumsum(widths) - np.array(widths) / 2).tolist()


def poem_cells():
    """Every row of the sheet in the order it is written, as (what, x, y, width) in cm: the title, the four couplets, the signature."""
    tw = [w for _, w in TITLE]
    rows = [[(c, x, TITLE_Y, w) for (c, w), x in zip(TITLE, cells_x(tw, (SHEET_W - sum(tw)) / 2))]]
    for y, line in zip(LINE_YS, POEM):
        ws = [PUNCT_W if c in PUNCT else CHAR_W for c in line]
        rows.append([(c, x, y, w) for c, w, x in zip(line, ws, cells_x(ws, LINE_X))])
    rows.append([(SIGN, SIGN_X, SIGN_Y, 5.0)])
    return rows


def row_extents():
    """Where each row starts and ends across the sheet, in cm: what the water follows."""
    rows = poem_cells()
    return np.array([r[0][1] - r[0][3] / 2 for r in rows], np.float32), np.array([r[-1][1] + r[-1][3] / 2 for r in rows], np.float32)


def paper(w, h, seed, tone=.86, stains=1.0):
    """Old paper as a grey value: uneven, faintly foxed, darker toward the edges where hands have been. Returns the albedo."""
    low = fft_noise(h // 4, w // 4, 3.0, seed)
    low = np.asarray(Image.fromarray(low).resize((w, h), Image.BICUBIC), np.float32)
    mid = fft_noise(h // 2, w // 2, 2.2, seed + 1)
    mid = np.asarray(Image.fromarray(mid).resize((w, h), Image.BICUBIC), np.float32)
    r = np.random.default_rng(seed + 2)
    fib = gaussian_filter(r.normal(size=(h, w)).astype(np.float32), (.7, 2.6))
    fib2 = gaussian_filter(r.normal(size=(h, w)).astype(np.float32), (2.6, .7))
    grain = (fib / fib.std() + fib2 / fib2.std()) * .5
    a = tone + .011 * low + .005 * mid + .010 * grain
    # foxing: small soft brown spots
    fox = np.zeros((h, w), np.float32)
    for _ in range(int(60 * stains)):
        x, y = r.integers(w), r.integers(h)
        fox[y, x] = r.uniform(.3, 1)
    fox = gaussian_filter(fox, 6) * 170 + gaussian_filter(fox, 2) * 30
    a -= np.clip(fox, 0, 1) * .05
    # tide marks where it once got damp
    tide = fft_noise(h // 4, w // 4, 3.6, seed + 3)
    tide = np.asarray(Image.fromarray(tide).resize((w, h), Image.BICUBIC), np.float32)
    a -= .014 * np.exp(-((tide - 1.1) / .05) ** 2) * stains
    return np.clip(a, 0, 1).astype(np.float32)


def rough(m, seed, amt=.12, scale=1.0):
    """A cut edge: the outline wanders and the ink misses in places."""
    h, w = m.shape
    n = fft_noise(1024, 1024, 1.3, seed + 40)
    r = np.random.default_rng(seed)
    ox, oy = r.integers(1024), r.integers(1024)
    n = np.take(np.take(n, (np.arange(h) * 2 // 1 + oy) % 1024, axis=0), (np.arange(w) * 2 // 1 + ox) % 1024, axis=1)
    s = gaussian_filter(m, 1.1 * scale)
    return sm((s + amt * n - .5) / .16)


def draw_rules(x0, x1, y0, y1, seed, ncols, side, shape):
    """版框 and 界行: four sides doubled, thick outside and thin inside (the fourth side is the fold, where the frame runs on to the other half of the sheet),
    and the thin cut lines between columns, none of them quite straight or unbroken."""
    h, w = shape
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    r = np.random.default_rng(seed)

    def line(xa, ya, xb, yb, wd, breaks=.0):
        n = max(int(max(abs(xb - xa), abs(yb - ya)) / 24), 2)
        pts = []
        for i in range(n + 1):
            t = i / n
            pts.append((xa + (xb - xa) * t + r.normal(0, .9), ya + (yb - ya) * t + r.normal(0, .9)))
        for a, b in zip(pts, pts[1:]):
            if r.random() > breaks:
                d.line([a, b], fill=255, width=max(int(wd + r.normal(0, wd * .1)), 1))

    k, t = int(.085 * CM), int(.028 * CM)
    gap = int(.10 * CM)
    # the binding side, doubled
    if side == "R":
        bx, ox, ex = x0, x0 - gap, w
    else:
        bx, ox, ex = x1, x1 + gap, 0
    line(ox, y0 - gap, ox, y1 + gap, k)
    line(bx, y0, bx, y1, t)
    # head and tail, doubled, running on to the fold
    line(ox, y0 - gap, ex, y0 - gap, k)
    line(ox, y1 + gap, ex, y1 + gap, k)
    line(bx, y0, ex, y0, t)
    line(bx, y1, ex, y1, t)
    for i in range(ncols + 1):
        if (i == 0 and side == "R") or (i == ncols and side != "R"):
            continue
        x = x0 + (x1 - x0) * i / ncols
        line(x, y0, x, y1, int(.022 * CM), breaks=.012)
    m = np.asarray(im, np.float32) / 255
    return rough(m, seed + 3, .1)


def yu_wei(D, xc, y, half_w, seed, up=True):
    """A fish tail (鱼尾), the black mark at the fold that a binder folds by."""
    im = Image.new("L", D.shape[::-1], 0)
    d = ImageDraw.Draw(im)
    s = half_w * 1.0
    pts = [(xc - s, y), (xc + s, y), (xc + s, y + s * 1.1), (xc, y + s * .55), (xc - s, y + s * 1.1)]
    d.polygon(pts, fill=255)
    d.line([(xc - s, y - .05 * CM), (xc + s, y - .05 * CM)], fill=255, width=int(.02 * CM))
    return rough(np.asarray(im, np.float32) / 255, seed, .1)


def print_page(chars, side, seed, leaf_no="十二", tone=.86):
    """A printed half leaf. side 'R': the fold is on the right (verso, the right-hand page of a spread); 'L': the fold is on the left.
    Returns the image: albedo and, as a second value, the ink density."""
    W, H = SIZE
    a = paper(W, H, seed, tone)
    D = np.zeros((H, W), np.float32)
    y0, y1 = int(TOP * CM), int((TOP + PITCH * 20) * CM)
    if side == "R":
        col_x = lambda j: (INNER + (10 - j - .5) * COL) * CM     # j = 0 is the column read first, next to the fold
        x0, x1 = int(INNER * CM), int((INNER + 10 * COL) * CM)
        fold_c = (INNER + 10 * COL + GUTTER_STRIP * .5) * CM
    else:
        col_x = lambda j: (LEAF_W - INNER - (j + .5) * COL) * CM
        x1, x0 = int((LEAF_W - INNER) * CM), int((LEAF_W - INNER - 10 * COL) * CM)
        fold_c = (LEAF_W - INNER - 10 * COL - GUTTER_STRIP * .5) * CM
    D += draw_rules(x0, x1, y0, y1, seed + 5, 10, side, (H, W)) * 1.55
    D += yu_wei(D, fold_c, y0 + 1.5 * CM, .44 * CM, seed + 8) * 1.7
    marks = [("淮", 4.15, .62), ("南", 4.15 + .68, .62), ("子", 4.15 + 1.36, .62), ("卷", 8.1, .56), ("八", 8.1 + .62, .56)]
    marks += [(c_, 16.5 + i_ * .6, .5) for i_, c_ in enumerate(leaf_no)]
    for s, yc, size in marks:
        m = glyph_mask(s, "song", int(size * CM), 8)
        r_ = rough(m, seed + ord(s), .1)
        hh, ww = m.shape
        xx, yy = int(fold_c - ww / 2), int((TOP + yc) * CM - hh / 2)
        xa, ya = max(xx, 0), max(yy, 0)
        xb, yb = min(xx + ww, W), min(yy + hh, H)
        D[ya:yb, xa:xb] += r_[ya - yy:yb - yy, xa - xx:xb - xx] * 1.55
    # the text
    r = np.random.default_rng(seed + 11)
    gsize = int(PITCH * .86 * CM)
    for i, ch in enumerate(chars[:200]):
        j, k = i // 20, i % 20
        m = glyph_mask(ch, "song", gsize, 12)
        cut = rough(m, seed + i, .13)
        level = 1.55 * (r.uniform(.84, 1.02)) * (1 - .12 * max(0, r.normal(.0, 1)))
        hh, ww = m.shape
        xx = int(col_x(j) - ww / 2 + r.normal(0, 1.6))
        yy = int((TOP + .13 + (k + .5) * PITCH * .985) * CM - hh / 2 + r.normal(0, 1.8))
        D[yy:yy + hh, xx:xx + ww] += cut * level
    # uneven inking: pressed harder in places, and the paper's grain leaves flecks
    block = np.asarray(Image.fromarray(fft_noise(H // 4, W // 4, 2.8, seed + 12)).resize((W, H), Image.BICUBIC), np.float32)
    fleck = fft_noise(H, W, .5, seed + 13)
    D = D * np.clip(1 + .10 * block, .7, 1.3) * (1 - .22 * sm((fleck - 1.1) / .8))
    D = gaussian_filter(D, .55)
    return a, D


def to_rgb(a, D, red=None):
    k = np.array([1.0, 1.0, 1.0], np.float32)
    rgb = a[..., None] * np.exp(-D[..., None] * k)
    if red is not None:
        # vermilion paste: opaque, sitting on the paper
        c = np.array([.60, .055, .04], np.float32)
        rgb = rgb * (1 - red[..., None]) + (a[..., None] * c) * red[..., None]
    return np.clip(rgb, 0, 1)


def save(rgb, name):
    os.makedirs(TEX, exist_ok=True)
    Image.fromarray((rgb * 255 + .5).astype(np.uint8)).save(os.path.join(TEX, name))


def brush_char(ch, size, load, seed, r, m=None):
    """One character written with a loaded brush. load is the ink left in it: 1 at the top of a column, about .65 at the bottom. The font gives the skeleton; the hand turns and pushes it
    a little, the strokes thin as the brush empties, wet ink gathers at the edge of a stroke, and as it dries, fine white streaks open along the inside of its broad strokes (飞白).
    The whole character always survives: a hairline keeps its spine and the streaks never reach an outline, because a character with a piece missing is not 飞白, it is broken."""
    pad = size // 5
    m = glyph_mask(ch, "xing", size, pad) if m is None else m
    h, w = m.shape
    k = size / 422
    spent = (1 - load) / .35
    dry = .5 * spent
    yy, xx = np.mgrid[:h, :w].astype(np.float32)

    def noise(sigma):
        n = gaussian_filter(r.normal(size=(h, w)).astype(np.float32), sigma)
        return n / n.std()

    ang = np.clip(r.normal(0, .012), -.026, .026)         # at most 1.5°
    cx, cy = w / 2, h / 2
    sx = cx + (xx - cx) * math.cos(ang) - (yy - cy) * math.sin(ang) + noise(size / 7) * size / 70
    sy = cy + (xx - cx) * math.sin(ang) + (yy - cy) * math.cos(ang) + noise(size / 7) * size / 70
    m = map_coordinates(m, [sy, sx], order=1, mode="constant")
    # thinner as the brush empties, and each character a little heavier or lighter than its neighbours, but never so much that a neck parts or a counter closes
    solid = m > .3
    sd = distance_transform_edt(solid) - distance_transform_edt(~solid)
    thin = (2.6 * spent + np.clip(r.normal(0, .7), -1.4, 1.4)) * size / 288
    marks, spaces = pieces(solid), pieces(~solid)
    for _ in range(8):
        if pieces(sd > thin) <= marks and pieces(sd <= thin) >= spaces:
            break
        thin *= .6
    m = gaussian_filter((sd > thin).astype(np.float32), .8)
    soft = gaussian_filter(m, 2.0 * k)
    shape = sm((soft + (.05 + .05 * dry) * noise(2.6 * k) + .04 * noise(7.0 * k) - .46) / .10)
    shape = np.maximum(shape * sm(gaussian_filter(m, 6 * k) / .08), sm((m - .40) / .25))

    # which way each stroke runs: the structure tensor, which has no trouble at a stroke's spine
    gy, gx = np.gradient(gaussian_filter(m, size / 70))
    sig = size / 15
    jxx, jxy, jyy = gaussian_filter(gx * gx, sig), gaussian_filter(gx * gy, sig), gaussian_filter(gy * gy, sig)
    theta = .5 * np.arctan2(2 * jxy, jxx - jyy) + math.pi / 2
    ox, oy = np.cos(theta), np.sin(theta)
    n0 = noise(2.8 * k)
    lic = n0.copy()
    step = size / 55
    for sign in (1.0, -1.0):
        px, py, dx, dy = xx.copy(), yy.copy(), ox * sign, oy * sign
        for _ in range(16):
            ax, ay = map_coordinates(ox, [py, px], order=1, mode="nearest"), map_coordinates(oy, [py, px], order=1, mode="nearest")
            flip = np.sign(ax * dx + ay * dy + 1e-6)
            dx, dy = ax * flip, ay * flip
            px, py = px + dx * step, py + dy * step
            lic += map_coordinates(n0, [py, px], order=1, mode="nearest")
    lic = (lic - lic.mean()) / lic.std()
    patch = noise(size / 7)
    white = sm((lic - (2.5 - 2.6 * dry + .6 * patch)) / .30)
    inside = sm((gaussian_filter(m, size / 30) - .66) / .22)       # only well inside a broad stroke
    cover = shape * (1 - .38 * white * inside)
    pool = np.clip(shape - sm((gaussian_filter(m, size / 42) - .62) / .22), 0, 1)
    # ink is not one flat black: where the brush pressed and went slowly there is a lot of it, a flick that left the paper fast carries little, and the hairs of the brush leave their lines along the stroke
    body = sm(distance_transform_edt(m > .5) / (size * .032))
    hairs = sm((lic - .35) / 1.3) * (.13 + .22 * dry)
    tone = (1.5 + .95 * body - .5 * dry + .26 * patch + .35 * pool * (1 - dry)) * (1 - hairs * inside)
    D = cover * np.clip(tone, .9, 3.2) * (1 - .38 * spent) * r.uniform(.96, 1.04)
    # wet ink bleeds into the paper, unevenly, by a hair
    return D + gaussian_filter(D, 3.5 * k) * np.clip(.10 + .05 * noise(14.0 * k), 0, 1) * (1 - dry) ** 2


def pieces(a):
    """How many separate marks a mask has, specks aside."""
    lab, _ = label(a)
    return int((np.bincount(lab.ravel())[1:] > 12).sum())


def empty_seal(s, seed):
    """A square cut in stone with nothing cut inside it. The paste is thick, the stone is chipped, and the paste misses in the paper's grain."""
    im = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(im)
    b = int(.05 * s)
    d.rounded_rectangle([b, b, s - b, s - b], radius=int(.07 * s), outline=255, width=int(.135 * s))
    m = np.asarray(im, np.float32) / 255
    seal = sm((gaussian_filter(m, 2.2) + .20 * fft_noise(s, s, 1.0, seed) + .16 * fft_noise(s, s, 2.2, seed + 2) - .5) / .12)
    seal = seal * (1 - .75 * sm((fft_noise(s, s, .7, seed + 1) - 1.0) / .7))
    return np.maximum(seal, sm((fft_noise(s, s, 1.2, seed + 3) - 2.3) / .4) * .7)      # a little paste left where the stone was not quite cleared


def fibres(w, h, r, per, lo, hi):
    """Single fibres lying any way, one per `per` square pixels, as a mask."""
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for _ in range(int(w * h / per)):
        x, y, ang, ln = r.uniform(0, w), r.uniform(0, h), r.uniform(0, 2 * math.pi), r.uniform(lo, hi)
        bend = r.normal(0, .5)
        pts = [(x + ln * t * math.cos(ang + bend * t), y + ln * t * math.sin(ang + bend * t)) for t in np.linspace(0, 1, 7)]
        d.line(pts, fill=int(r.uniform(70, 255)), width=1)
    return np.asarray(im, np.float32) / 255


def xuan(w, h, seed):
    """New xuan paper, as far from the book's pages as paper gets: pale and clean, with long bark fibres in it, the faint lines of the mould, and no frame, no rules, no age."""
    r = np.random.default_rng(seed)
    low = np.asarray(Image.fromarray(fft_noise(h // 4, w // 4, 3.0, seed)).resize((w, h), Image.BICUBIC), np.float32)
    grain = gaussian_filter(r.normal(size=(h, w)).astype(np.float32), .9)
    a = .900 + .005 * low + .007 * grain / grain.std()
    a -= .030 * gaussian_filter(fibres(w, h, r, 5200, 50, 260), .7)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    a += .0035 * np.sin(2 * math.pi * yy / (.105 * SCM)) - .005 * np.exp(-(((xx / SCM + .7) % 2.6 - 1.3) / .035) ** 2)
    # the deckle: no cutter has been near this edge, and nothing has torn it. The pulp just thins out over a millimetre or two and ends in a fuzz of loose fibres
    edge = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy]) / SCM
    n1 = np.asarray(Image.fromarray(fft_noise(h // 6, w // 6, 3.0, seed + 5)).resize((w, h), Image.BICUBIC), np.float32)
    n2 = np.asarray(Image.fromarray(fft_noise(h // 3, w // 3, 2.0, seed + 6)).resize((w, h), Image.BICUBIC), np.float32)
    e = edge - .40 + .09 * n1 + .018 * n2
    fuzz = np.clip(gaussian_filter(fibres(w, h, r, 1500, 25, 120), .6) * 1.6, 0, 1) * sm((e + .20) / .16)
    return np.clip(a + .012 * (1 - sm((edge - .5) / .5)), 0, 1).astype(np.float32), np.maximum(sm(e / .13), .8 * fuzz)


def water_clock(xx, yy, low):
    """When the water reaches each place on the sheet, counted in rows: down the rows from the first, along each from the left. xx, yy: in pixels of the texture. low(beta, seed): its noise at those places. Returns
    the ink's clock, and the clock of the damp that spreads over the paper around it, which is not so careful of the lines."""
    edges = (ROWS[:-1] + ROWS[1:]) / 2
    a, b = row_extents()
    row = np.searchsorted(edges, (yy + 25 * low(2.4, 51)) / SCM)
    along = np.clip((xx / SCM - a[row]) / (b[row] - a[row]), 0, 1)
    p = row.astype(np.float32) + .85 * along + .10 * low(3.4, 52)
    q = np.searchsorted(edges, (yy + 90 * low(3.0, 53)) / SCM).astype(np.float32) + .85 * np.clip((xx / SCM - LINE_X) / LINE_W, 0, 1) + .16 * low(3.4, 52)
    x_w = (xx + 60 * low(3.0, 56)) / SCM
    return p, q, sm((x_w - 1.6) / 1.5) * sm((SHEET_W - 1.6 - x_w) / 1.5)


def washed(D, wet, a, t):
    """The poem going, row by row from the first, each from its left, as if water were drawing the ink out of the paper. t counts rows: 0 nothing has gone, past the last row everything has.
    Ink first pales and spreads and then it is not there; the paper stays damp behind it for a while, then dries white. Returns ink, wetness, paper."""
    h, w = D.shape

    def low(beta, seed):
        return np.asarray(Image.fromarray(fft_noise(h // 6, w // 6, beta, seed)).resize((w, h), Image.BICUBIC), np.float32)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    p, q, band = water_clock(xx, yy, low)
    f = sm((t - p) / .7)
    keep = 1 - f
    run = keep * f
    ink = map_coordinates(D, [yy + f * 16 * low(2.6, 54), xx + f * 16 * low(2.6, 55)], order=1)     # it drifts as it goes
    out = ink * keep ** 3 + (gaussian_filter(ink, 7) * .6 + gaussian_filter(ink, 22) * .9 + gaussian_filter(ink, 60) * .9) * run
    # where the water has been: a stain with a wandering edge and a thin tide line, drying from the oldest
    stain = sm((t - q) / .7) * (1 - sm((t - q - 1.6) / 1.6)) * band
    tide = np.clip(np.hypot(*np.gradient(gaussian_filter(stain, 5))) * 12, 0, 1)
    return out, wet * keep, a * (1 - .05 * gaussian_filter(stain, 30) - .028 * tide)


def punctuation(ch, size, pad):
    """A comma or a full stop as the brush makes them: a dab pulled down and away, a small ring. They sit low in their cell, where the line's foot is. Same frame as glyph_mask."""
    n = size + 2 * pad
    im = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(im)
    P = lambda *q: [(pad + a * size, pad + b * size) for a, b in q]
    W = lambda *q: [a * size for a in q]
    if ch == "，":
        stroke(d, P((.47, .66), (.53, .71), (.545, .78), (.50, .86), (.42, .92)), W(.05, .135, .115, .06, .012))
    else:
        t = np.linspace(-.35, 2 * math.pi - .55, 15)
        stroke(d, P(*[(.5 + .105 * math.cos(a), .80 + .10 * math.sin(a)) for a in t]), W(.05, .062, .068, .056, .044, .05))
    return np.asarray(im, np.float32) / 255


def ink_box(g):
    """g cropped so that its ink is in the middle of it: a glyph's cell says little about where a bracket's ink is."""
    ys, xs = np.nonzero(g > .15)
    h, w = g.shape
    dy, dx = (ys.min() + ys.max() + 1) / 2 - h / 2, (xs.min() + xs.max() + 1) / 2 - w / 2
    return np.roll(np.roll(g, -int(round(dy)), 0), -int(round(dx)), 1)


def brackets(ch, size, pad):
    """《 and 》, two quick strokes each, pressed into the turn and let go at the ends. A font's are ruled. Same frame as glyph_mask."""
    n = size + 2 * pad
    im = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(im)
    flip = (lambda a: a) if ch == "《" else (lambda a: 1 - a)
    P = lambda *q: chaikin(np.array([(pad + flip(a) * size, pad + b * size) for a, b in q], np.float32), 2)
    W = lambda *q: [a * size for a in q]
    stroke(d, P((.66, .13), (.47, .33), (.31, .52), (.45, .70), (.64, .89)), W(.018, .05, .072, .095, .075, .05, .014), n=120)
    stroke(d, P((.86, .17), (.68, .35), (.53, .52), (.66, .68), (.83, .85)), W(.016, .045, .066, .085, .068, .045, .012), n=120)
    return np.asarray(im, np.float32) / 255


def sign_brush(h):
    """Opus5.5 written with the same brush as the poem, h pixels from the top of a capital to the line it stands on: a little slanted, thick on the way down and thin on the way up, as a pointed brush does Latin."""
    k_w, pad = 4.5, int(.6 * h)
    im = Image.new("L", (int(k_w * h) + 2 * pad, int(1.45 * h) + 2 * pad), 0)
    d = ImageDraw.Draw(im)
    lean = .16
    P = lambda *q: chaikin(np.array([(pad + (a + lean * (1 - b)) * h, pad + b * h) for a, b in q], np.float32), 3)
    W = lambda *q: [a * h for a in q]
    # O, a full oval, heavy on its left
    t = np.linspace(-1.9, 2 * math.pi - 1.75, 15)
    stroke(d, P(*[(.36 + .30 * math.cos(a), .5 + .49 * math.sin(a)) for a in t]), W(.035, .07, .125, .15, .12, .07, .045, .04, .06, .075, .05, .02), n=160)
    # p
    stroke(d, P((.98, .33), (1.0, .55), (.97, .95), (.92, 1.36)), W(.10, .135, .115, .02), n=100)
    stroke(d, P((.99, .50), (1.14, .32), (1.33, .34), (1.37, .58), (1.22, .86), (1.02, .88)), W(.03, .055, .10, .125, .075, .02), n=100)
    # u
    stroke(d, P((1.56, .33), (1.54, .55), (1.53, .80), (1.64, .96), (1.82, .80), (1.94, .36)), W(.09, .125, .10, .045, .05, .075), n=100)
    stroke(d, P((1.95, .34), (1.91, .62), (1.92, .86), (2.04, .96)), W(.09, .125, .085, .018), n=80)
    # s
    stroke(d, P((2.43, .36), (2.27, .31), (2.17, .44), (2.30, .60), (2.40, .76), (2.26, .95), (2.08, .88)), W(.02, .05, .075, .115, .10, .06, .015), n=100)

    def five(x):
        stroke(d, P((x + .45, .03), (x + .25, .05), (x + .05, .03)), W(.035, .075, .10), n=50)
        stroke(d, P((x + .05, .01), (x + .02, .22), (x - .03, .46)), W(.11, .10, .065), n=50)
        stroke(d, P((x - .03, .46), (x + .18, .35), (x + .40, .47), (x + .38, .74), (x + .16, .97), (x - .10, .86)), W(.03, .07, .12, .13, .07, .016), n=100)

    five(2.72)
    stroke(d, P((3.36, .85), (3.40, .92), (3.38, .99)), W(.07, .135, .05), n=30)      # the point
    five(3.72)
    return np.asarray(im, np.float32) / 255


def sheet_textures():
    """The poem on its own sheet, 30 × 23 cm, written across as a page is now: 《AI 自传》 in the middle at the top, four lines of a couplet each, Opus5.5 at the lower right with the seal beside it. Written out
    (sheet.png), with the seal down (sheet_seal.png), going (sheet_fading.png, at FADE_T), gone leaving only the seal (sheet_seal_only.png), and the paper before any of it (sheet_blank.png). The _ink maps say where
    the ink is still wet: the last lines written are the wettest. sheet_shadow.png is the little shadow it throws on the desk."""
    W, H = int(SHEET_W * SCM), int(SHEET_H * SCM)
    ra = np.random.default_rng(71)
    a, alpha = xuan(W, H, 31)
    D = np.zeros((H, W), np.float32)
    wet = np.zeros((H, W), np.float32)
    nudge = lambda: float(np.clip(ra.normal(0, .022), -.05, .05))        # at most 2 pixels in the frame

    def put(g, x_cm, y_cm, wetness):
        hh, ww = g.shape
        xx, yy = int(x_cm * SCM - ww / 2), int(y_cm * SCM - hh / 2)
        D[yy:yy + hh, xx:xx + ww] += g
        wet[yy:yy + hh, xx:xx + ww] = np.maximum(wet[yy:yy + hh, xx:xx + ww], np.clip(g / 2.2, 0, 1) * wetness)

    rows = poem_cells()
    size = int(1.5 * SCM)
    for i, row in enumerate(rows[1:-1]):
        k = 0                                     # how far the brush has gone since it was last loaded, which is at the head of each half of the couplet
        for j, (ch, x, y, _) in enumerate(row):
            pad = size // 5
            wetness = .35 + .65 * (i + j / len(row)) / len(POEM)
            if ch in PUNCT:
                load = float(np.clip(1 - .35 + ra.normal(0, .015), .62, 1))
                put(brush_char(ch, size, load, 500 + i * 10 + j, ra, m=punctuation(ch, size, pad)), x, y, wetness)
                k = 0
                continue
            sz = int(size * ra.uniform(.97, 1.03))
            load = float(np.clip(1 - .35 * (k / 6) ** 1.15 + ra.normal(0, .015), .62, 1))
            g = brush_char(ch, sz, load, 100 + i * 20 + j, ra)
            base = glyph_mask(ch, "xing", sz, sz // 5)
            if pieces(g > .45) > pieces(base > .5) or pieces(g <= .45) < pieces(base <= .5):
                print(f"check {ch} (line {i + 1}, {j + 1}): {pieces(base > .5)} marks and {pieces(base <= .5)} spaces became {pieces(g > .45)} and {pieces(g <= .45)}")
            put(g, x + nudge(), y + nudge(), wetness)
            k += 1
    # the title. AI side by side in one character's space
    ts = int(TITLE_SIZE * SCM)
    for ch, x, y, _ in rows[0]:
        pad = ts // 5
        if ch == "AI":
            g = brush_char(ch, ts, .97, 41, ra, m=ai_brush(ts, pad))
        elif ch in "《》":
            g = ink_box(brush_char(ch, ts, .97, 40 + ord(ch) % 7, ra, m=brackets(ch, ts, pad)))
        else:
            g = brush_char(ch, ts, .85, 42 + ord(ch) % 7, ra, m=glyph_mask(ch, "kaib", ts, pad))      # the title is set out in a steadier hand than the poem
        put(g, x, y, .3)
    # the signature, the seal beside it
    put(ink_box(brush_char(SIGN, int(1.5 * SCM), .9, 301, ra, m=sign_brush(int(SIGN_SIZE * SCM)))), SIGN_X, SIGN_Y, .8)
    # ink creeps a hair's breadth along the paper's fibres from every edge, more where there is more of it
    D += gaussian_filter(D, 3.0) * gaussian_filter(fibres(W, H, ra, 1300, 18, 70), .6) * .9 * (1 - sm(D / 1.0))
    s_ = int(SEAL_S * SCM)
    red = np.zeros((H, W), np.float32)
    xx, yy = int(SEAL_X * SCM - s_ / 2), int(SEAL_Y * SCM)
    red[yy:yy + s_, xx:xx + s_] = empty_seal(s_, 305)
    os.makedirs(TEX, exist_ok=True)

    def write(name, a_, D_, rd, wet_=None):
        rgb = to_rgb(a_, D_, rd)
        Image.fromarray((np.dstack([rgb, alpha]) * 255 + .5).astype(np.uint8), "RGBA").save(os.path.join(TEX, name + ".png"))
        if wet_ is not None:
            Image.fromarray((np.clip(wet_, 0, 1) * 255).astype(np.uint8)).save(os.path.join(TEX, name + "_ink.png"))

    write("sheet", a, D, None, wet)
    write("sheet_seal", a, D, red)
    D_f, wet_f, a_f = washed(D, wet, a, FADE_T)
    write("sheet_fading", a_f, D_f, red, wet_f)
    write("sheet_seal_only", a, np.zeros_like(D), red)
    write("sheet_blank", a, np.zeros_like(D), None)
    # a sheet this thin, lying this close, throws almost nothing: a soft grey line on the side away from the moon. The moon's own shadow of the fuzz would be a black saw
    px, pad_cm = 40, SHADOW_PAD
    small = Image.fromarray((alpha * 255).astype(np.uint8)).resize((int(SHEET_W * px), int(SHEET_H * px)), Image.BOX)
    canvas = Image.new("L", (int((SHEET_W + 2 * pad_cm) * px), int((SHEET_H + 2 * pad_cm) * px)), 0)
    canvas.paste(small, (int(pad_cm * px) + 4, int(pad_cm * px) - 2))
    shadow = gaussian_filter(np.asarray(canvas, np.float32) / 255, 4.0) * .30
    Image.fromarray((shadow * 255).astype(np.uint8)).save(os.path.join(TEX, "sheet_shadow.png"))


def cover_image(title_chars, small, name, seed, title_kind="kaih"):
    """The front cover, 14.5 × 23 cm: dark dyed paper, rubbed at the edges, with a title slip stuck on near the top left."""
    w, h = int(14.5 * CM), int(23 * CM)
    r = np.random.default_rng(seed)
    a = .245 + .012 * np.asarray(Image.fromarray(fft_noise(h // 4, w // 4, 2.6, seed)).resize((w, h), Image.BICUBIC), np.float32)
    fib = gaussian_filter(r.normal(size=(h, w)).astype(np.float32), (.8, 2.8))
    a = a + .016 * fib / fib.std()
    dye = np.asarray(Image.fromarray(fft_noise(h // 8, w // 8, 3.0, seed + 8)).resize((w, h), Image.BICUBIC), np.float32)
    a = a + .014 * dye
    laid = gaussian_filter(r.normal(size=(1, w)).astype(np.float32), (0, 1.4))
    a = a + .012 * laid / laid.std() + .006 * np.sin(np.arange(h, dtype=np.float32)[:, None] * 2 * math.pi / 19)
    # rubbing: where hands have held it, and where the dye has worn at the edges and corners
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    edge = np.minimum.reduce([xx, w - xx, yy, h - yy]) / CM
    wear = np.exp(-edge / .42) * sm((fft_noise(h, w, 2.1, seed + 1) + 1.0) / 2)
    a = a + .18 * np.clip(wear, 0, 1)
    scuffs = sm((np.asarray(Image.fromarray(fft_noise(h // 2, w // 2, 1.9, seed + 2)).resize((w, h), Image.BICUBIC), np.float32) - 1.9) / .5)
    a = a + .05 * scuffs
    # the fold line by the stitching
    a = a * (1 - .25 * np.exp(-((xx / CM - 13.4) / .05) ** 2))
    # the slip
    sx0, sy0, sw, sh = int(1.0 * CM), int(1.0 * CM), int(3.05 * CM), int(13.6 * CM)
    sp = paper(sw, sh, seed + 3, .84, stains=.5)
    # a slightly uneven cut, and a shadow from its thickness
    mask = np.zeros((h, w), np.float32)
    mask[sy0:sy0 + sh, sx0:sx0 + sw] = 1
    n = fft_noise(1024, 1024, 2.0, seed + 4)
    nn = np.take(np.take(n, np.arange(h) % 1024, axis=0), np.arange(w) % 1024, axis=1)
    mask = sm((gaussian_filter(mask, 1.2) + .05 * nn - .5) / .1)
    shadow = np.roll(np.roll(gaussian_filter(mask, 4), 5, axis=0), 5, axis=1)
    a = a * (1 - .45 * shadow)
    D = np.zeros((sh, sw), np.float32)
    # double frame
    im = Image.new("L", (sw, sh), 0)
    d = ImageDraw.Draw(im)
    b = int(.2 * CM)
    d.rectangle([b, b, sw - b, sh - b], outline=255, width=int(.05 * CM))
    d.rectangle([b + int(.1 * CM), b + int(.1 * CM), sw - b - int(.1 * CM), sh - b - int(.1 * CM)], outline=255, width=int(.015 * CM))
    D += rough(np.asarray(im, np.float32) / 255, seed + 5, .1) * 1.3
    gs = int(2.3 * CM)
    pit = 3.25
    y = 1.55 + gs / CM / 2
    for ch in title_chars:
        m = glyph_mask(ch, title_kind, gs, 12)
        g = brush_like(m, seed + ord(ch))
        hh, ww = m.shape
        xx_, yy_ = int(sw / 2 - ww / 2), int(y * CM - hh / 2)
        xb = min(xx_ + ww, sw)
        xa = max(xx_, 0)
        yb = min(yy_ + hh, sh)
        D[yy_:yb, xa:xb] += g[:yb - yy_, xa - xx_:xb - xx_]
        y += pit
    y2 = 12.0 - (len(small) - 1) * .42
    for ch in small:
        m = glyph_mask(ch, "kaib", int(.72 * CM), 8)
        hh, ww = m.shape
        xx_, yy_ = int(sw / 2 - ww / 2), int(y2 * CM - hh / 2)
        D[yy_:yy_ + hh, xx_:xx_ + ww] += rough(m, seed + ord(ch), .1) * 1.9
        y2 += .84
    slip = np.clip(sp * np.exp(-D), 0, 1)
    a[sy0:sy0 + sh, sx0:sx0 + sw] = a[sy0:sy0 + sh, sx0:sx0 + sw] * (1 - mask[sy0:sy0 + sh, sx0:sx0 + sw]) + slip * mask[sy0:sy0 + sh, sx0:sx0 + sw]
    save(np.repeat(np.clip(a, 0, 1)[..., None], 3, -1), name)
    # bump: the slip stands a little proud, and the cover has a laid texture
    bump = .5 + .25 * mask + .06 * (fib / fib.std())
    Image.fromarray((np.clip(bump, 0, 1) * 255).astype(np.uint8)).save(os.path.join(TEX, name.replace(".png", "_bump.png")))


def stroke(d, pts, widths, n=60):
    """A brush stroke as a polygon: a smooth centre line through pts, and how wide it is along its length."""
    pts = np.array(pts, np.float32)
    t = np.linspace(0, 1, n)
    k = np.linspace(0, 1, len(pts))
    x, y = np.interp(t, k, pts[:, 0]), np.interp(t, k, pts[:, 1])
    for _ in range(3):
        x[1:-1], y[1:-1] = (x[:-2] + 2 * x[1:-1] + x[2:]) / 4, (y[:-2] + 2 * y[1:-1] + y[2:]) / 4
    w = np.interp(t, np.linspace(0, 1, len(widths)), widths) / 2
    gx, gy = np.gradient(x), np.gradient(y)
    g = np.hypot(gx, gy) + 1e-6
    nx, ny = -gy / g, gx / g
    d.polygon(list(zip(x + nx * w, y + ny * w)) + list(zip(x - nx * w, y - ny * w))[::-1], fill=255)


def ai_brush(gs, pad):
    """AI, written with the same brush as the characters under it, side by side in one character's space, the way Latin letters sit in vertical type: 撇, 捺 and 横 for the A,
    one 竖 for the I. A font's A beside these two characters would look pasted on, and a lone upright I stacked in the column would read as a stroke. Same frame as glyph_mask."""
    n = gs + 2 * pad
    im = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(im)
    P = lambda *q: [(pad + a * gs, pad + b * gs) for a, b in q]
    W = lambda *q: [a * gs for a in q]
    stroke(d, P((.39, .13), (.34, .28), (.25, .50), (.13, .72), (.015, .85)), W(.06, .105, .10, .07, .01))
    stroke(d, P((.38, .19), (.46, .38), (.55, .58), (.64, .75), (.75, .85)), W(.04, .07, .105, .145, .012))
    stroke(d, P((.17, .615), (.37, .60), (.58, .58)), W(.035, .05, .065))
    stroke(d, P((.875, .13), (.92, .19), (.91, .40), (.905, .65), (.90, .87)), W(.04, .125, .105, .10, .012))
    return np.asarray(im, np.float32) / 255


def brush_like(m, seed):
    n = fft_noise(1024, 1024, 1.1, seed)
    r = np.random.default_rng(seed)
    ox, oy = r.integers(1024), r.integers(1024)
    h, w = m.shape
    nn = np.take(np.take(n, (np.arange(h) + oy) % 1024, axis=0), (np.arange(w) + ox) % 1024, axis=1)
    return sm((gaussian_filter(m, 1.2) + .14 * nn - .5) / .14) * (2.2 + .3 * gaussian_filter(nn, 9) * 3)


def desk():
    """Old wood, pale where the moon lands on it: year lines that run the length of the board and drift apart and together, the pores of the grain cut along them, and a worn sheen. It tiles."""
    w, h = 4096, 2048
    r = np.random.default_rng(91)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)

    def long_noise(beta, seed, ax):
        """Noise stretched along the board."""
        n = fft_noise(h, w // ax, beta, seed)
        n = np.concatenate([n, n[:, :1]], 1)
        pos = xx[0] / w * (w // ax)
        i = pos.astype(np.int32)
        f = (pos - i)[None, :]
        return n[:, i] * (1 - f) + n[:, i + 1] * f

    # boards, each cut from a different part of the trunk, so the year lines jump at a seam. The lines' spacing wanders and some are faint.
    seam = np.minimum(np.abs(yy - h / 2), np.minimum(yy, h - yy))
    w_ = yy / h + .05 * long_noise(3.8, 92, 40) + .004 * long_noise(2.4, 93, 16)
    field = 20 * w_ + .85 * np.sin(6 * math.pi * w_ + 1.2) + .25 * np.sin(14 * math.pi * w_ + 4) + 7.3 * (yy > h / 2)
    f = field % 1.0
    strength = r.uniform(.2, 1, 64)[np.floor(field).astype(np.int32) % 64]
    late = f ** 4 * strength                       # the dark line where a year ends
    pore = gaussian_filter(r.normal(size=(h, w)).astype(np.float32), (.6, 16), mode="wrap")
    pore /= pore.std()
    streak = gaussian_filter(r.normal(size=(h, w)).astype(np.float32), (2.5, 90), mode="wrap")
    streak /= streak.std()
    board = long_noise(3.0, 94, 40)
    a = .30 * (1 +  .14 * board + .06 * streak) * (1 - .42 * late) * (1 - .34 * sm((pore - 1.25) / .6)) * (1 - .75 * np.exp(-(seam / 3.0) ** 2))
    bump = .5 - .12 * late - .30 * sm((pore - .9) / .6) + .03 * streak - .35 * np.exp(-(seam / 3.0) ** 2)
    rough_ = .40 + .10 * late + .10 * sm((pore - .8) / .6) + .05 * board
    os.makedirs(TEX, exist_ok=True)
    for arr, name in ((a, "desk.png"), (bump, "desk_bump.png"), (rough_, "desk_rough.png")):
        Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(os.path.join(TEX, name))


def cmd_tex():
    os.makedirs(TEX, exist_ok=True)
    i = TEXT.index("昔者蒼頡")
    start = i - 140                # so that 昔 opens the eighth column of the right-hand page
    t = TEXT[start:]
    assert t[140:146] == "昔者蒼頡作書"
    pages = [("page_R.png", t[:200], "R", 11, "十二"), ("page_L.png", t[200:400], "L", 12, "十三"),
             ("page_a.png", t[400:600], "L", 13, "十四"), ("page_b.png", t[600:800], "R", 14, "十五"), ("page_c.png", TEXT[:200], "L", 15, "十一")]
    only = sys.argv[2:]
    if not only or "pages" in only:
        for name, chars, side, seed, no in pages:
            a, D = print_page(chars, side, seed, no)
            save(to_rgb(a, D), name)
    if not only or "sheet" in only:
        sheet_textures()
    if not only or "cover" in only:
        a = paper(SIZE[0], SIZE[1], 17, .84)
        save(to_rgb(a, np.zeros_like(a)), "lining.png")
        cover_image("淮南子", "卷八", "cover.png", 21)
    if not only or "desk" in only:
        desk()


def book_grade(rgba, gamma, contrast, seed, creep=.55, vig=0.0, vc=(.47, .55)):
    """The same mapping from light to ink as ink_grade, and the same paper, without its mist. Its creep is gentler here and slower in space: this is printed type seen close up,
    and the jitter that suits a mountain would chew the edges of a character."""
    h, w = rgba.shape[:2]
    D = (np.clip(1 - rgba[..., :3].mean(-1), 0, 1) ** gamma) * 3.1 * contrast
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    if vig > 0:      # a wash of ink laid round the picture: the far desk falls away into the dark, and the eye stays on the book
        r2 = ((xx / w - vc[0]) / .62) ** 2 + ((yy / h - vc[1]) / .66) ** 2
        D = D + vig * sm((r2 + .10 * fft_noise(h, w, 3.2, seed + 21) - .42) / 1.0) ** 1.4
    D = map_coordinates(D, [yy + creep * fft_noise(h, w, 2.7, seed + 3), xx + creep * fft_noise(h, w, 2.7, seed + 4)], order=1, mode="nearest")
    gy, gx = np.gradient(gaussian_filter(D, 1.1))
    D = D + np.clip(np.sqrt(gx * gx + gy * gy) * .9, 0, .35)
    D = D * (1 + (.045 * fft_noise(h, w, .7, seed + 5) + .07 * fft_noise(h, w, 1.7, seed + 6)) * np.exp(-D * .55))
    tone = 1 + .022 * fft_noise(h, w, 2.6, seed + 7) + .010 * fft_noise(h, w, .4, seed + 8)
    out = PAPER[None, None, :] * tone[..., None] * np.exp(-np.maximum(D, 0)[..., None] * INK_K[None, None, :])
    return np.clip(np.maximum(out, FLOOR[None, None, :]), 0, 255).astype(np.uint8)


class Bleed:
    """The ink lets go of the page. keep: the x where the column that stays begins; right of that column everything has already gone, so it stands alone where the subtitles
    live. The ink is conserved: as it spreads along the paper's fibres a character thins into a grey cloud, the clouds meet, and the farther from the column the further along
    this is. A wash comes in from the left, and in the end it is the night of the next chapter. Set up once for a picture, then at(u) for any u: past u = 1 the wash goes on over the rest of the page."""

    def __init__(self, img, keep, seed):
        self.keep, self.seed = keep, seed
        h, w = self.shape = img.shape[:2]
        L = (img.astype(np.float32) / 255).mean(-1)
        self.page = gaussian_filter(maximum_filter(L, 33), 18)
        ink = np.clip(-np.log(np.clip(L / np.maximum(self.page, .05), .02, 1)), 0, 4)      # the printed ink as a density
        self.yy, self.xx = yy, xx = np.mgrid[:h, :w].astype(np.float32)
        left, right = sm((keep - xx) / 22), sm((xx - keep - 78) / 34)
        self.kept = ink * (1 - left) * (1 - right) * (1 - sm((yy - 1036) / 22))    # the sentence ends at 哭
        src = ink * left
        lost = ink * right          # what is right of the column, for a picture that starts with it still there
        self.n1, n2, self.n3 = fft_noise(h, w, 2.8, seed + 1), fft_noise(h, w, 2.8, seed + 2), fft_noise(h, w, 3.4, seed + 4)
        self.f1, f2 = fft_noise(h, w, 2.4, seed + 5), fft_noise(h, w, 2.4, seed + 6)

        def spread(sigma, k, kf):
            return map_coordinates(gaussian_filter(src, sigma), [yy + k * self.n1 + kf * self.f1, xx + k * n2 + kf * f2], order=1, mode="nearest")

        self.s0, self.s1, self.s2, self.s3 = spread(1.2, 1.5, .8), spread(5, 8, 2.5), spread(16, 26, 5), spread(46, 70, 9)
        self.lost0, self.lost1 = lost, map_coordinates(gaussian_filter(lost, 7), [yy + 10 * self.n1, xx + 10 * n2], order=1, mode="nearest")
        self.grain = fft_noise(h, w, 1.1, seed + 7)[..., None] * .012

    def density(self, u, wash_k=3.4, kept_k=1.25, right_k=0.0, cloud_k=1.5):
        """How much ink each pixel has at u. right_k: how much is left of what is right of the column, going as a spreading cloud."""
        xx = self.xx
        ul = np.clip(u * (2.7 - 2.4 * xx / self.keep), 0, 1)               # how far along each place is
        w1, w2, w3 = sm(ul / .34), sm((ul - .16) / .42), sm((ul - .42) / .55)
        cloud = self.s0 * (1 - w1) + (self.s1 * (1 - w2) + (self.s2 * (1 - w3) + self.s3 * w3) * w2) * w1
        # a cloud's thin edge shows the paper's fibres, and dries a shade darker where it stopped
        cloud = cloud * (1 + .16 * self.f1 * sm(ul / .4))
        gy, gx = np.gradient(gaussian_filter(cloud, 2.0))
        tide = np.clip(np.sqrt(gx * gx + gy * gy) * 2.2, 0, .22) * sm((ul - .1) / .4)
        # the wash from the left has no edge to speak of: it is thick at the frame's edge and feathers out over a third of the picture
        front = (u - .02) * 900 + 190 * self.n3 + 50 * self.n1
        wash = sm((front - xx) / 640) ** 2.3
        wash = wash * (1 + .22 * self.f1)
        D = self.kept * kept_k + cloud * cloud_k + tide + wash_k * wash
        if right_k > 0:
            D = D + (self.lost0 * right_k ** 2 + self.lost1 * 1.2 * (1 - right_k ** 2) * right_k)
        return D

    def picture(self, D):
        return np.clip(np.maximum((self.page[..., None] + self.grain) * np.exp(-D[..., None] * INK_K[None, None, :]) * PAPER[None, None, :] / PAPER.max(), FLOOR[None, None, :] / 255.0) * 255, 0, 255).astype(np.uint8)

    def at(self, u):
        return self.picture(self.density(u))


def bleed(img, u, keep, seed):
    return Bleed(img, keep, seed).at(u)


def grade_still(rgba, gamma=1.5, contrast=1.0, seed=0, creep=.55, vig=0.0):
    """A raw frame to ink. The seal is paper to the grade: its colour goes back on after."""
    r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    red = np.clip((r - (g + b) / 2) * 3.0 - .06, 0, 1)
    mono = rgba.copy()
    mono[..., :3] = (rgba[..., :3].mean(-1) * (1 - red) + .9 * red)[..., None]
    im = book_grade(mono, gamma, contrast, seed, creep, vig)
    if red.max() > .05:
        lum = np.clip(r * 1.25, 0, 1)
        seal = np.array([168, 30, 22], np.float32)
        out = im.astype(np.float32) * (1 - red[..., None]) + seal[None, None, :] * lum[..., None] * red[..., None]
        im = np.clip(out, 0, 255).astype(np.uint8)
    return im


def cmd_post(shot, args):
    def opt(k, default):
        for a in args:
            if a.startswith(f"--{k}="):
                return type(default)(a.split("=")[1])
        return default
    name = os.path.join(DIR, shot)
    rgba = load(name + ("_raw" if shot == "sheet" else "") + ".png")     # sheet.png is the contact sheet
    im = grade_still(rgba, opt("gamma", 1.5), opt("contrast", 1.0), opt("seed", 0), opt("creep", .55), opt("vig", 0.0))
    u = opt("bleed", 0.0)
    if u > 0:
        im = bleed(im, u, opt("keep", 1763.0), opt("seed", 0))
    Image.fromarray(im).save(name + "_ink.png")
    lum = im.astype(np.float32).mean(-1) / 255
    print(f"{shot}: mean {lum.mean():.3f} (>= .25)  dead black {100 * (lum < .08).mean():.1f}% (<= 20)  bright {100 * (lum >= .5).mean():.1f}% (>= 15)")


def cmd_sheet(shots):
    tw = 960
    ims = [Image.open(os.path.join(DIR, s + "_ink.png")).resize((tw, tw * 9 // 16), Image.LANCZOS) for s in shots]
    n = math.ceil(len(ims) / 2)
    s = Image.new("RGB", (2 * tw, n * tw * 9 // 16), (30, 30, 30))
    for i, im in enumerate(ims):
        s.paste(im, ((i % 2) * tw, (i // 2) * tw * 9 // 16))
    s.save(os.path.join(DIR, "sheet.png"))


if __name__ == "__main__":
    if sys.argv[1] == "tex":
        cmd_tex()
    elif sys.argv[1] == "post":
        cmd_post(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "sheet":
        cmd_sheet(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
