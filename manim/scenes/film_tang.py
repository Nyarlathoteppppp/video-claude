"""七's last shot: a thread-bound 唐詩三百首 lying open on the old table at 卷六's 五言絕句, the spread with 李白's 靜夜思 on its right page. Worked out flat and in millimetres: the paper, the print, the shape the open book takes. film_tang3d.py lays it on the table and lets the sun come across it.

    python -I -c "import sys; sys.path.insert(0, '.'); import film_tang; film_tang.write()"   # from manim/scenes: maps/spread_col.png, spread_rough.png, spread.npz, spread.json
    python film_tang.py write typeset  # the old typeset placeholder instead of the scan
    python -I -c "import sys; sys.path.insert(0, '.'); import film_tang; film_tang.write(eaten=True)"   # 七's last beat: spread_col_eaten.png, spread_rough_eaten.png, key_0..4.png, pieces.json
    python film_tang.py check          # every typeset text against tang300.txt, every character in the face

The ink is the real page's: 《唐詩三百首註疏》卷六上, 掃葉山房石印本, Harvard Library drs:437097988, by way of Wikimedia Commons, public domain (SCAN_PDF; read where it lies on the Studio, run with -I, never copied into the repo). Its page 4 is one opening: on the right half-leaf 宿建德江's lines, 春曉 and 李白's poem, which this edition titles 夜思, with the commentary in small double columns; on the left 怨情 and 八陣圖; the 版心 (唐詩三百首註疏 卷六上 三 掃葉山房石印, one black 魚尾) at each fore-edge. Its ink takes the place of the typeset one and the rest stays: the paper, the book's shape. Scale: the scan's paper, head to foot, is the spread's 235 mm (8.01 scan pixels a mm; the frame at 1920 asks at most 5.7 a mm of the page), each half-leaf's fold on its fore-edge.

The typeset placeholder is still here (write typeset): a woodblock half-leaf as such books were cut (左右雙邊, 白口, 單黑魚尾, ten columns of twenty), the poems that stand around 靜夜思 in 蘅塘退士's order, their words checked against tang300.txt.

Everything is in mm from the spread's top left, x to the right, y down; the gutter is at x = 150. A 線裝 book is read from the right, its read leaves piled on the right: this is early in the volume, so the right stack is thin and the left one thick.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter

from film_docs import bumps, curl, face, noise, srgb, stretch, to_srgb
from film_press_maps import fibres
from ink_grade import sm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "ink", "film_out", "film", "learn", "tang")
MAPS = os.path.join(OUT, "maps")
TANG300 = os.path.join(ROOT, "ink", "film_out", "learn", "tang300.txt")

W, H, PX = 300.0, 235.0, 14           # the spread, and samples per mm
GUTTER = W / 2
TOP, ROWS, CELL = 38.0, 20, 8.8       # the frame's head, and its twenty cells
FOOT = TOP + ROWS * CELL              # 214: the frame's foot; 21 mm of 地腳 below it
COLS, PITCH = 10, 11.0                # columns to a half-leaf
HALF = 7.0                            # half the 版心, between its rule and the fold
EM = 7.9                              # the face cut, in mm
FACE = ("LiSongPro.ttf", 0)           # 儷宋: the old forms of the characters, near what a block cutter's 匠體 gives
STACK = dict(left=9.0, right=4.0)     # how high each side of the open book stands, mm; at the gutter both come down to GUTTER_Z
GUTTER_Z = 2.2
COVER = 0.8                           # the soft paper cover under the block

# the right page (後半葉 of one leaf), read first, from its 版心 at the fore-edge towards the gutter; then the left page (前半葉 of the next), from the gutter to its 版心
# a title column: (title, author or ""); a poem column: its twenty characters
RIGHT = (("宿建德江", "孟浩然"), "移舟泊煙渚日暮客愁新野曠天低樹江清月近人",
         ("春曉", ""), "春眠不覺曉處處聞啼鳥夜來風雨聲花落知多少",
         ("靜夜思", "李白"), "床前明月光疑是地上霜舉頭望明月低頭思故鄉",
         ("怨情", ""), "美人捲珠簾深坐顰蛾眉但見淚痕濕不知心恨誰",
         ("八陣圖", "杜甫"), "功蓋三分國名高八陣圖江流石不轉遺恨失吞吳")
LEFT = (("登鸛雀樓", "王之渙"), "白日依山盡黃河入海流欲窮千里目更上一層樓",
        ("送靈澈", "劉長卿"), "蒼蒼竹林寺杳杳鐘聲晚荷笠帶夕陽青山獨歸遠",
        ("彈琴", ""), "泠泠七絲上靜聽松風寒古調雖自愛今人多不彈",
        ("送上人", ""), "孤雲將野鶴豈向人間住莫買沃洲山時人已知處",
        ("秋夜寄邱員外", "韋應物"), "懷君屬秋夜散步詠涼天山空松子落幽人應未眠")
KEY = "低頭思故鄉"
TITLE_AT, AUTHOR_GAP = 2, 1           # a title stands two cells down, its poet one cell after it
INK = srgb(20, 17, 16)
PAPER = srgb(233, 221, 189)           # bamboo paper, yellowed a little

# the real page, measured on the scan (pixels of the PDF's page 4 as it holds it, 2400 by 1947), 2026-11-07
SCAN_PDF = os.path.expanduser("~/render_work/scans/harvard_tang300_v6/tang300_zhushu_v6.pdf")
SCAN_PAGE = 4
SCAN_SOURCE = "《唐詩三百首註疏》卷六上，掃葉山房石印本，哈佛大学图书馆藏（drs:437097988），经 Wikimedia Commons，公有领域"
SCAN_PAPER = (31, 33, 2372, 1916)     # the opening's paper: the left half-leaf's fold, the head, the right one's fold, the foot
SCAN_SEAM = (1186, 1192)              # where the two half-leaves meet in the binding: each page takes ink only from its own side
SCAN_PX = (SCAN_PAPER[3] - SCAN_PAPER[1]) / H      # 8.01 scan pixels to a mm
SCAN_INK, SCAN_FLOOR = 55.0, 0.12     # the lithograph's solid ink, and how much darkening is the paper's own (show-through, grain): 99 in 100 of the blank margins stay under 0.08
SCAN_KEY = ((1460, 545, 1532, 601), (1460, 604, 1532, 676), (1460, 680, 1532, 740), (1460, 745, 1532, 815), (1460, 817, 1532, 890))    # 低頭思故鄉
SCAN_TITLE = ((1650, 408, 1713, 468), (1650, 473, 1713, 535))        # 夜思


def column_x(page, k):
    """The middle of the k-th column (0 = read first) of a page."""
    if page == "right":
        return W - HALF - PITCH * (k + 0.5)
    return GUTTER - (GUTTER - HALF - COLS * PITCH) - PITCH * (k + 0.5)


def cell_y(j):
    return TOP + CELL * (j + 0.5)


def plan():
    """Every character's place: c, page, column, cell, x, y (its middle)."""
    out = []
    for page, cols in (("right", RIGHT), ("left", LEFT)):
        for k, col in enumerate(cols):
            x = column_x(page, k)
            if isinstance(col, tuple):
                title, poet = col
                cells = [(TITLE_AT + i, c) for i, c in enumerate(title)]
                if poet:
                    cells += [(TITLE_AT + len(title) + AUTHOR_GAP + i, c) for i, c in enumerate(poet)]
                kind = "title"
            else:
                assert len(col) == ROWS, col
                cells, kind = list(enumerate(col)), "poem"
            for j, c in cells:
                out.append(dict(c=c, page=page, col=k, cell=j, kind=kind, x=round(x, 3), y=round(cell_y(j), 3)))
    return out


def box(ch):
    return [round(ch["x"] - EM / 2, 2), round(ch["y"] - EM / 2, 2), round(ch["x"] + EM / 2, 2), round(ch["y"] + EM / 2, 2)]


def cut(f, ch, r, A):
    """One character as the block gives it: the cut face a little worn, inked unevenly, the ink drawn a little way into the paper."""
    px = PX
    size = f.size
    pad = int(size * 0.35)
    n = size + 2 * pad
    big = Image.new("L", (n * 3, n * 3), 0)
    jx, jy = r.normal(0, 0.04 * px), r.normal(0, 0.05 * px)
    ImageDraw.Draw(big).text((n * 1.5 + jx * 3, n * 1.5 + jy * 3), ch["c"], 255, font=f.font_variant(size=size * 3), anchor="mm")
    a = np.asarray(big.resize((n, n), Image.BOX), np.float32) / 255
    nz = gaussian_filter(r.normal(0, 1, a.shape).astype(np.float32), 0.06 * px)
    nz /= nz.std() + 1e-6
    # the face is a light 宋: a cut block prints fuller than that, the ink squeezed out past the wood's edges
    spread = float(r.uniform(0.15, 0.24))           # how hard this part of the block was inked and how worn its face is
    worn = sm((gaussian_filter(a, 0.1 * px) - spread - 0.05 * nz) * 3.2 + 0.5)
    level = float(np.clip(r.normal(0.97, 0.03), 0.85, 1.0))
    mottle = 1 - 0.12 * sm(gaussian_filter(r.normal(0, 1, a.shape).astype(np.float32), 0.08 * px) * 2.2)
    rim = np.exp(-distance_transform_edt(worn > 0.5) / (0.06 * px))
    halo = gaussian_filter(worn, 0.045 * px) * 0.32
    ink = np.maximum(worn * level * mottle * (0.82 + 0.18 * rim), halo * level)
    cx, cy = int(round(ch["x"] * px)), int(round(ch["y"] * px))
    t, l = cy - n // 2, cx - n // 2
    b = A[t:t + n, l:l + n]
    b[:] = np.maximum(b, ink[:b.shape[0], :b.shape[1]])
    return worn


def rules(Hp, Wp, r):
    """The frame, the column rules and the half 魚尾 at each fold, as the block left them: their edges ragged, the thin rules broken here and there."""
    px = PX
    im = Image.new("L", (Wp * 2, Hp * 2), 0)
    d = ImageDraw.Draw(im)
    s = 2 * px

    def bar(x0, y0, x1, y1, v=255):
        d.rectangle([x0 * s, y0 * s, x1 * s, y1 * s], fill=v)

    thick, thin, gap = 0.85, 0.24, 0.55
    for page in ("right", "left"):
        fold_rule = W - HALF if page == "right" else HALF                # the 版心's rule
        outer = W - HALF - COLS * PITCH if page == "right" else HALF + COLS * PITCH
        fold = W if page == "right" else 0.0
        o = -1 if page == "right" else 1                                  # outwards, away from the text
        x_out = outer + o * (gap + thin)
        lo, hi = sorted((x_out + o * thick, fold))
        bar(lo, TOP - thick, hi, TOP)                                     # head and foot run out to the fold
        bar(lo, FOOT, hi, FOOT + thick)
        bar(min(x_out, x_out + o * thick), TOP - thick, max(x_out, x_out + o * thick), FOOT + thick)                 # 左右雙邊: thick outside, thin inside
        bar(min(outer, outer + o * thin), TOP, max(outer, outer + o * thin), FOOT)
        bar(fold_rule - thin / 2, TOP, fold_rule + thin / 2, FOOT)
        for k in range(1, COLS):                                          # 界行
            x = (W - HALF - PITCH * k) if page == "right" else (HALF + PITCH * k)
            bar(x - 0.09, TOP, x + 0.09, FOOT, 210)
        # the upper 魚尾, black, at a quarter of the frame: its half on this page, notched below
        y0, y1 = TOP + 0.25 * ROWS * CELL - 4.5, TOP + 0.25 * ROWS * CELL + 3.5
        a, b_ = sorted((fold_rule, fold))
        notch = 3.8                                                       # its fork, cut up from below at the fold
        if page == "right":
            poly = [(a, y0), (b_, y0), (b_, y1 - notch), (a, y1)]
        else:
            poly = [(a, y0), (b_, y0), (b_, y1), (a, y1 - notch)]
        d.polygon([(x * s, y * s) for x, y in poly], fill=255)
        bar(a, y0 - 0.9, b_, y0 - 0.65)                                   # the hair above it
    m = np.asarray(im.resize((Wp, Hp), Image.BOX), np.float32) / 255
    nz = noise(Hp, Wp, 2.0, 701)
    rag = sm((gaussian_filter(m, 0.035 * px) - 0.45 - 0.08 * nz) * 3.0 + 0.5)
    breaks = 1 - 0.85 * sm((noise(Hp, Wp, 2.6, 702) - 2.1) * 2.0)      # where the wood has gone
    return rag * np.minimum(breaks + (m > 0.97) * 0.4, 1.0)


def shape():
    """The open book: each side's leaves come up from the gutter and lie nearly flat, their edges lifting a very little. Returns X, Y (mm, rows from the foot up) and Z (mm above the table)."""
    step = 0.6
    us, vs = np.linspace(0, W, int(W / step) + 1, dtype=np.float32), np.linspace(0, H, int(H / step) + 1, dtype=np.float32)
    U, V = np.meshgrid(us, vs)
    d = np.sqrt((U - GUTTER) ** 2 + 1.4 ** 2) - 1.4
    side = np.where(U >= GUTTER, STACK["right"], STACK["left"])
    reach = np.where(U >= GUTTER, 11.0, 15.0)
    z = COVER + GUTTER_Z + (side - GUTTER_Z) * (1 - np.exp(-d / reach))
    z -= 0.7 * np.exp(-(GUTTER - d) / 5.0)                     # the fore-edges roll down over the block's edge
    z += 0.5 * np.sin(np.pi * V / H) * (1 - np.exp(-d / 30))  # the leaves rise a little through their middle
    z += bumps(U, V, W, H, 70, 0.45, 711) + bumps(U, V, W, H, 22, 0.06, 712)      # bamboo paper cockles: under a low sun it shows
    z += curl(U, V, W, H, (0.4, 0.5, 0.35, 0.3), (6, 7, 6, 6), 713) * (d > 8)
    return U, V, z.astype(np.float32)


def _pdf_jpeg(path, page):
    """The JPEG a scanned PDF holds for one page (1 = first), its bytes as they are: the page tree followed, nothing decoded or re-encoded."""
    import re
    data = open(path, "rb").read()
    objs = {}
    for m in re.finditer(rb"(?<![0-9])(\d+)\s+0\s+obj\b", data):
        objs[int(m.group(1))] = (m.end(), data.find(b"endobj", m.end()))
    head = lambda n: data[objs[n][0]:objs[n][1]].split(b"stream")[0]
    pages = []

    def walk(n):
        d = head(n)
        if re.search(rb"/Type\s*/Pages\b", d):
            for k in re.findall(rb"(\d+)\s+0\s+R", re.search(rb"/Kids\s*\[([^\]]*)\]", d).group(1)):
                walk(int(k))
        elif re.search(rb"/Type\s*/Page\b", d):
            pages.append(n)
    root = next(n for n in objs if re.search(rb"/Type\s*/Catalog\b", head(n)))
    walk(int(re.search(rb"/Pages\s+(\d+)\s+0\s+R", head(root)).group(1)))
    d = head(pages[page - 1])
    res = re.search(rb"/Resources\s*(\d+)\s+0\s+R", d)
    rs = head(int(res.group(1))) if res else d
    xo = re.search(rb"/XObject\s*(\d+)\s+0\s+R", rs)
    xs = head(int(xo.group(1))) if xo else re.search(rb"/XObject\s*<<(.*?)>>", rs, re.S).group(1)
    im = int(re.findall(rb"/\w+\s+(\d+)\s+0\s+R", xs)[0])
    idict = head(im)
    assert b"/DCTDecode" in idict and b"/FlateDecode" not in idict, idict[:200]
    m = re.search(rb"/Length\s+(\d+)(\s+0\s+R)?", idict)
    length = int(re.search(rb"\d+", head(int(m.group(1)))).group(0)) if m.group(2) else int(m.group(1))
    i = data.find(b"stream", objs[im][0]) + 6
    i += 2 if data[i:i + 2] == b"\r\n" else 1
    jpeg = data[i:i + length]
    assert jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9", "not a whole JPEG"
    return jpeg


def scan_ink(Hp, Wp):
    """The real page's ink on the spread's grid, 0 to 1: how much darker than its own paper each place of the scan is, less what blank paper does too, each half-leaf from its own side of the seam."""
    import io
    from scipy.ndimage import grey_closing, map_coordinates
    L = np.asarray(Image.open(io.BytesIO(_pdf_jpeg(SCAN_PDF, SCAN_PAGE))).convert("L"), np.float32)
    paper = gaussian_filter(grey_closing(L, size=(31, 31)), 12)          # the paper's own light, carried in under the strokes
    a = np.clip((paper - L) / np.maximum(paper - SCAN_INK, 60), 0, 1)
    a = np.clip((a - SCAN_FLOOR) / (0.95 - SCAN_FLOOR), 0, 1)
    x0, y0, x1, y1 = SCAN_PAPER
    a[:y0 + 2], a[y1 - 2:], a[:, :x0 + 2], a[:, x1 - 2:] = 0, 0, 0, 0       # the scanner's black round the book is not ink
    yy, xx = np.mgrid[0:Hp, 0:Wp].astype(np.float32)
    u, v = (xx + 0.5) / PX, (yy + 0.5) / PX                               # mm from the spread's top left
    right = u >= GUTTER
    X = np.where(right, x1 - (W - u) * SCAN_PX, x0 + u * SCAN_PX)
    Y = y0 + v * SCAN_PX
    out = np.clip(map_coordinates(a, [Y, X], order=3, mode="constant", cval=0.0), 0, 1).astype(np.float32)
    out[right & (X < SCAN_SEAM[1])] = 0
    out[~right & (X > SCAN_SEAM[0])] = 0
    return out


def scan_box(b):
    """A box on the scan, in the spread's mm (the right half-leaf)."""
    x0, y0, x1, y1 = b
    return [round(W - (SCAN_PAPER[2] - x0) / SCAN_PX, 2), round((y0 - SCAN_PAPER[1]) / SCAN_PX, 2), round(W - (SCAN_PAPER[2] - x1) / SCAN_PX, 2), round((y1 - SCAN_PAPER[1]) / SCAN_PX, 2)]


EAT_GHOST = 0.12      # what of a lifted character's ink the paper keeps: the ink had sunk in a little, so where it was stays faintly grey
EAT_PAD = 0.6         # mm round each of the five boxes that goes with the character
EAT_RULE = 3          # pixels either side of a rule that stay on the page with it


def lift(a):
    """七's last beat (2026-11-08): 低頭思故鄉 is taken off the page, and 八 cuts to the crab chewing. a, the page's ink, comes back with only a ghost
    of the five characters, and each of them is written as a piece, key_N.png (N = 0 for 低 down to 4 for 鄉): the ink's colour, and as alpha how
    much of the page's ink it is. That alpha is a (1 - g) / (1 - g a), so that the piece laid on the ghost gives back the page as it was, to the
    pixel, before anything moves. pieces.json: each piece's box in mm, as key_N.png covers it."""
    out, boxes = a.copy(), []
    ink = np.uint8(np.clip(np.round(to_srgb(INK) * 255), 0, 255))
    key = [scan_box(s) for s in SCAN_KEY]
    for k, b in enumerate(key):
        # padded, but where two characters of the column are nearer than that, each piece goes only to the middle of the gap between them
        y0 = b[1] - EAT_PAD if k == 0 else max(b[1] - EAT_PAD, (key[k - 1][3] + b[1]) / 2)
        y1 = b[3] + EAT_PAD if k == len(key) - 1 else min(b[3] + EAT_PAD, (b[3] + key[k + 1][1]) / 2)
        x0, x1 = b[0] - EAT_PAD, b[2] + EAT_PAD
        j0, j1 = int(np.floor(x0 * PX)), int(np.ceil(x1 * PX))
        i0, i1 = int(round(y0 * PX)), int(round(y1 * PX))          # rounded, not widened: two pieces that meet share no row
        c = a[i0:i1, j0:j1]
        # the column's rules belong to the page, not to the character: a pixel column inked down most of the frame's height is a rule, and stays
        rule = (a[int(TOP * PX):int(FOOT * PX), j0:j1] > 0.3).mean(0) > 0.6
        rule = np.convolve(rule.astype(np.float32), np.ones(2 * EAT_RULE + 1), "same") > 0
        alpha = np.where(rule[None, :], 0.0, c * (1 - EAT_GHOST) / (1 - EAT_GHOST * c))
        out[i0:i1, j0:j1] = np.where(rule[None, :], c, c * EAT_GHOST)
        rgba = np.zeros(c.shape + (4,), np.uint8)
        rgba[..., :3] = ink
        rgba[..., 3] = np.uint8(np.clip(np.round(alpha * 255), 0, 255))
        Image.fromarray(rgba, "RGBA").save(os.path.join(MAPS, "key_%d.png" % k))
        boxes.append(dict(c=KEY[k], box=[round(j0 / PX, 4), round(i0 / PX, 4), round(j1 / PX, 4), round(i1 / PX, 4)]))
    with open(os.path.join(MAPS, "pieces.json"), "w", encoding="utf-8") as fp:
        json.dump(dict(ghost=EAT_GHOST, pieces=boxes), fp, ensure_ascii=False)
    print("lifted", [p["c"] for p in boxes], flush=True)
    return out


def write(source="scan", eaten=False):
    """eaten: the spread after 七's last beat, as spread_col_eaten.png and spread_rough_eaten.png, with the five pieces (lift); the paper, its shape and spread.json stay the plain write's."""
    os.makedirs(MAPS, exist_ok=True)
    Hp, Wp = int(H * PX), int(W * PX)
    r = np.random.default_rng(705)
    yy, xx = np.mgrid[0:Hp, 0:Wp].astype(np.float32)
    u, v = (xx + 0.5) / PX, H - (yy + 0.5) / PX
    f = face(FACE[0], FACE[1], EM * PX)
    A = np.zeros((Hp, Wp), np.float32)
    for ch in plan():
        cut(f, ch, r, A)            # with the scan too, and thrown away: it draws on r as before, so the paper's foxing below falls where it did
    if source == "scan":
        a = scan_ink(Hp, Wp)
        if eaten:
            a = lift(a)
        chars = []
    else:
        chars = plan()
        L = rules(Hp, Wp, r)
        # the block was inked by hand: a little more on some parts of it than others, and each page was its own pull
        pull = 0.9 + 0.06 * noise(Hp, Wp, 3.2, 706) + 0.04 * np.where(u > GUTTER, 1.0, -0.5)
        a = np.clip(np.maximum(A, L * 0.95) * pull, 0, 1)
    # bamboo paper: long fibres, the faint laid lines of its screen, yellowed towards the edges and darker where hands held it
    col = np.broadcast_to(PAPER, (Hp, Wp, 3)).copy()
    pulp = stretch(fibres(4096, 9000, 707, length=(60, 280), width=(1, 2), bright=(.25, 1.0), wiggle=.1), Hp, Wp)
    laid = 0.5 + 0.5 * np.sin(2 * np.pi * (v + 0.2 * noise(Hp, Wp, 2.4, 708)) / 0.9)
    chain = np.exp(-((np.mod(u - 9.0 + 0.5 * noise(Hp, Wp, 3.0, 709), 21.0) - 10.5) / 0.6) ** 2)
    tone = 1 + 0.018 * noise(Hp, Wp, 2.8, 710) + 0.008 * noise(Hp, Wp, 1.2, 714) + 0.022 * gaussian_filter(pulp, 0.5) - 0.006 * chain + 0.004 * laid
    col *= tone[..., None]
    e = np.minimum.reduce((np.minimum(u, W - u), v, H - v))            # the fore-edges, head and foot; not the gutter, which nothing reaches
    sh = np.exp(-np.maximum(e + 2.2 * noise(Hp, Wp, 2.2, 715), 0) / 9.0)
    col *= 1 - sh[..., None] * np.array((0.08, 0.14, 0.26), np.float32)
    thumb = np.exp(-(((u - (W - 22)) / 26) ** 2 + ((v - 30) / 34) ** 2)) * (0.6 + 0.4 * sm(noise(Hp, Wp, 2.5, 716) + 0.5))
    col *= 1 - (thumb * 0.05)[..., None] * np.array((0.6, 0.85, 1.2), np.float32)
    fox = np.zeros((Hp, Wp), np.float32)
    for x, y, s_, k in zip(r.uniform(0, W, 40), r.uniform(0, H, 40), r.uniform(0.2, 1.4, 40), r.uniform(0.1, 0.5, 40)):
        i, j, m = int(y * PX), int(x * PX), int(s_ * PX * 3) + 2
        a0, a1, b0, b1 = max(i - m, 0), min(i + m, Hp), max(j - m, 0), min(j + m, Wp)
        y2, x2 = np.ogrid[a0:a1, b0:b1]
        fox[a0:a1, b0:b1] = np.maximum(fox[a0:a1, b0:b1], k * np.exp(-(((y2 - i) ** 2 + (x2 - j) ** 2) / (s_ * PX) ** 2)))
    col *= 1 - (fox * 0.22)[..., None] * np.array((0.35, 0.6, 1.0), np.float32)
    col = col + (INK - col) * a[..., None]
    rough = 0.86 - 0.06 * a
    if eaten:
        save8(to_srgb(col), "spread_col_eaten.png")
        save8(np.clip(rough, 0, 1), "spread_rough_eaten.png")
        return
    save8(to_srgb(col), "spread_col.png")
    save8(np.clip(rough, 0, 1), "spread_rough.png")
    U, V, Z = shape()
    np.savez(os.path.join(MAPS, "spread.npz"), X=U, Y=V, Z=Z)
    if source == "scan":
        # the sun comes to rest on the real 低頭思故鄉: its column's middle, the top of 低
        key = [scan_box(b) for b in SCAN_KEY]
        keys = dict(key=key, column=key, title=[scan_box(b) for b in SCAN_TITLE], cut=key[0][1], x=round((key[0][0] + key[0][2]) / 2, 2))
        note = dict(source=SCAN_SOURCE, page=SCAN_PAGE, scan_px_per_mm=round(SCAN_PX, 3))
        chars = [dict(c=c, page="right", box=b) for c, b in zip(KEY, key)]
    else:
        key = [c for c in chars if c["page"] == "right" and c["col"] == 5]
        assert "".join(c["c"] for c in key) == RIGHT[5]
        keys = dict(key=[box(c) for c in key[-5:]], column=[box(c) for c in key], title=[box(c) for c in chars if c["page"] == "right" and c["col"] == 4],
                    cut=round(TOP + CELL * (ROWS - 5), 2), x=key[0]["x"])
        note = dict(placeholder="typeset in LiSong as a woodblock half-leaf would be; not measured from a scan")
        chars = [dict(c=c["c"], page=c["page"], col=c["col"], cell=c["cell"], box=box(c)) for c in chars]
    with open(os.path.join(MAPS, "spread.json"), "w", encoding="utf-8") as fp:
        json.dump(dict(w=W, h=H, px=PX, gutter=GUTTER, stack=STACK, cover=COVER, keys=keys, chars=chars, **note), fp, ensure_ascii=False)
    print("spread", Wp, Hp, "chars", len(chars), "z %.1f-%.1f mm" % (float(Z.min()), float(Z.max())), flush=True)


def save8(a, name):
    rr = np.random.default_rng(1)
    im = Image.fromarray(np.uint8(np.clip(a * 255 + rr.uniform(0, 1, a.shape), 0, 255)))
    im.save(os.path.join(MAPS, name), compress_level=3)
    print(name, im.size, flush=True)


SIMPLE = str.maketrans(dict(p for p in "煙烟 曠旷 樹树 覺觉 曉晓 處处 聞闻 鳥鸟 來来 風风 聲声 舉举 頭头 鄉乡 捲卷 簾帘 顰颦 見见 淚泪 濕湿 誰谁 蓋盖 國国 陣阵 圖图 轉转 遺遗 吳吴 盡尽 黃黄 窮穷 層层 樓楼 蒼苍 鐘钟 帶带 陽阳 獨独 歸归 遠远 絲丝 靜静 聽听 調调 雖虽 愛爱 彈弹 雲云 將将 鶴鹤 豈岂 間间 買买 時时 懷怀 屬属 詠咏 涼凉 應应".split()))       # only what these ten poems need, to read them against tang300.txt


def check():
    with open(TANG300, encoding="utf-8") as fp:
        lines = {l.strip().replace("，", "").replace("。", "") for l in fp}
    f = face(FACE[0], FACE[1], 100)
    for col in RIGHT + LEFT:
        if isinstance(col, str):
            s = col.translate(SIMPLE)
            print("ok " if s in lines else "NOT IN tang300 ", col, s, flush=True)
    miss = [c["c"] for c in plan() if f.getmask(c["c"]).getbbox() is None]
    print("missing from the face:", miss or "none", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "write":
        write(sys.argv[2] if len(sys.argv) > 2 else "scan")
    else:
        dict(check=check)[sys.argv[1]]()
