"""What the room's objects are printed with: the table's wood and its old cup rings, the white keyboard's legends, the books' backs and the weave of their cloth, a page of a notebook, and what is outside the window. Plain Python, once, on the Studio:

    python film_room_tex.py       # ink/film_out/film/room/tex/

The keyboard's layout is here as well, and film_room3d.py builds the keys from it, so importing this file needs neither Pillow nor Blender.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TEX = os.path.join(ROOT, "ink", "film_out", "film", "room", "tex")
DESK = os.path.join(ROOT, "ink", "film_out", "3d", "book", "tex")

U = 0.0188                       # the pitch of a key, in metres
FN_H = 0.0105                    # the height of the top row
ROW_H = 0.0188
ROWS = (
    (("esc", 1.42), ("F1", 1), ("F2", 1), ("F3", 1), ("F4", 1), ("F5", 1), ("F6", 1), ("F7", 1), ("F8", 1), ("F9", 1), ("F10", 1), ("F11", 1), ("F12", 1), ("", 1)),
    (("`", 1), ("1", 1), ("2", 1), ("3", 1), ("4", 1), ("5", 1), ("6", 1), ("7", 1), ("8", 1), ("9", 1), ("0", 1), ("-", 1), ("=", 1), ("delete", 1.42)),
    (("tab", 1.42), ("Q", 1), ("W", 1), ("E", 1), ("R", 1), ("T", 1), ("Y", 1), ("U", 1), ("I", 1), ("O", 1), ("P", 1), ("[", 1), ("]", 1), ("\\", 1)),
    (("中/英", 1.71), ("A", 1), ("S", 1), ("D", 1), ("F", 1), ("G", 1), ("H", 1), ("J", 1), ("K", 1), ("L", 1), (";", 1), ("'", 1), ("return", 1.71)),
    (("shift", 2.24), ("Z", 1), ("X", 1), ("C", 1), ("V", 1), ("B", 1), ("N", 1), ("M", 1), (",", 1), (".", 1), ("/", 1), ("shift", 2.18)),
    (("fn", 1), ("control", 1), ("option", 1), ("command", 1.24), ("", 4.98), ("command", 1.24), ("option", 1), ("", 1), ("", 1), ("", 1)),
)
KEYS_W = 14.42 * U
KEYS_H = FN_H + 5 * ROW_H
PX = 8000                        # pixels of the legend picture to a metre


def keys():
    """Every key as (legend, x0, y0, x1, y1) in metres, x from the left of the keyboard and y down from its top."""
    out, y = [], 0.0
    for i, row in enumerate(ROWS):
        h = FN_H if i == 0 else ROW_H
        x = 0.0
        for legend, width in row:
            out.append((legend, x, y, x + width * U, y + h))
            x += width * U
        y += h
    # the arrows, an inverted T of half-height keys
    _, x, y0, _, y1 = out[-3]
    mid = (y0 + y1) / 2
    return out[:-3] + [("", x, mid, x + U, y1), ("", x + U, y0, x + 2 * U, mid), ("", x + U, mid, x + 2 * U, y1), ("", x + 2 * U, mid, x + 3 * U, y1)]


def linear_to_srgb(a):
    a = np.clip(a, 0, 1)
    return np.where(a <= .0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - .055)


def spectral(n, m, beta, seed):
    r = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(m)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1
    v = np.fft.irfft2((r.normal(size=f.shape) + 1j * r.normal(size=f.shape)) / f ** (beta / 2), s=(n, m))
    return ((v - v.mean()) / v.std()).astype(np.float32)


def save(a, name):
    from PIL import Image
    Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255 + .5)).save(os.path.join(TEX, name))


def ring(yy, xx, u, v, radius, width, aspect):
    d = np.hypot((xx - u) * aspect, yy - v)
    return np.exp(-((d - radius) / width) ** 2)


def table():
    """The wood of the film's first desk, in the colour of a table somebody uses: oak, gone darker and smoother where the arms rest, with rings from old cups."""
    from PIL import Image
    a = np.asarray(Image.open(os.path.join(DESK, "desk.png")).convert("L"), np.float32) / 255
    bump = np.asarray(Image.open(os.path.join(DESK, "desk_bump.png")).convert("L"), np.float32) / 255
    rough = np.asarray(Image.open(os.path.join(DESK, "desk_rough.png")).convert("L"), np.float32) / 255
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    n = np.clip((a - lo) / (hi - lo), 0, 1)[..., None]
    h, w = a.shape
    dark, light = np.array((.17, .094, .046), np.float32), np.array((.50, .325, .185), np.float32)
    col = dark + (light - dark) * n ** .85
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy, xx = yy / h, xx / w                     # v is 0 at the wall, 1 at the sitter's edge: the picture's top is the far side
    worn = np.exp(-((yy - .94) / .10) ** 2) * np.exp(-((xx - .53) / .20) ** 2)
    col = col * (1 + .10 * worn[..., None]) + np.array((.012, .010, .008), np.float32) * worn[..., None]
    stains = 0
    aspect = w / h
    for u, v, rad, strength in ((.70, .50, .037, .34), (.725, .61, .037, .22), (.235, .77, .036, .18), (.655, .86, .037, .13)):
        stains = stains + strength * ring(yy, xx, u, v, rad, .0028, aspect) * (.55 + .45 * np.clip(spectral(h, w, 1.5, int(u * 1000)) * .6 + .5, 0, 1))
    col = col * (1 - stains[..., None] * np.array((.70, .78, .86), np.float32))
    fine = spectral(h, w, .8, 5)
    col *= (1 + .018 * fine[..., None])
    r = np.random.default_rng(11)
    sc = np.zeros((h, w), np.float32)
    from PIL import ImageDraw
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for _ in range(46):
        x, y = r.uniform(0, w), r.uniform(0, h)
        an, ln = r.normal(0, .5), r.uniform(30, 240)
        d.line((x, y, x + np.cos(an) * ln, y + np.sin(an) * ln), fill=int(r.uniform(40, 120)), width=1)
    sc = np.asarray(im, np.float32) / 255
    col = col + sc[..., None] * .09
    rough = np.clip(.40 + .34 * rough - .10 * worn + .10 * stains + .06 * fine, .22, .85)
    save(linear_to_srgb(col), "table.png")
    save(rough, "table_rough.png")
    save(bump, "table_bump.png")


def legends():
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (int(KEYS_W * PX), int(KEYS_H * PX)), (236, 236, 233))
    d = ImageDraw.Draw(im)
    helvetica = lambda s: ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", s)
    song = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 30)
    pen = (58, 58, 60)
    for legend, x0, y0, x1, y1 in keys():
        if not legend:
            continue
        cx, cy = (x0 + x1) / 2 * PX, (y0 + y1) / 2 * PX
        if legend == "中/英":
            d.text((cx, cy), legend, pen, font=song, anchor="mm")
        elif len(legend) == 1:
            d.text((cx, cy), legend, pen, font=helvetica(40), anchor="mm")
        elif legend.startswith("F") and len(legend) <= 3:
            d.text((cx, cy + 12), legend, pen, font=helvetica(18), anchor="mm")
        else:
            d.text((x0 * PX + 14, y1 * PX - 14), legend, pen, font=helvetica(19), anchor="ls")
    im.save(os.path.join(TEX, "keys.png"))


SUPP = "/System/Library/Fonts/Supplemental/"
FONT = os.path.join(os.environ.get("INK_FONTS", os.path.expanduser("~/render_work/fonts")), "Kaiti.ttc")
# the texts the film really uses, oldest at the bottom of the pile, each title in its own script. All in metres and lying flat: w along the side that faces the room, d away from it, h thick; x and y where that side's middle stands on the table, yaw in degrees.
# case: a hardback with a rounded back. board: how thick its boards, bulge: how far the back rounds out, groove: how wide the joint between back and board. skin: leather (cell: the size of its grain) or cloth (pitch: the distance between its threads). stamp: gold or press
# paper: a paperback
# title: em, track in ems, at: the title's middle as a part of w, lines: how far in from each end of the back a rule runs
BOOKS = (
    dict(kind="case", title="𓂋𓏤𓈓𓏌𓏤𓉐𓂋𓏏𓂻𓅓𓉔𓂋𓅱𓇳𓏤", face=(SUPP + "NotoSansEgyptianHieroglyphs-Regular.ttf", 0), em=.0105, track=.0, at=.5, heavy=.00012, lines=(.017, .0205), stamp="gold",
         skin="leather", cell=.0011, colour=(.060, .030, .015), worn=(.15, .085, .045), rough=.50, spec=.24, band=((.28, .05, .03), (.50, .44, .30)),
         w=.240, d=.170, h=.041, board=.0034, bulge=.0058, groove=.0075, x=-.444, y=.448, yaw=-1.6),
    dict(kind="case", title="𒀭𒄑𒂆𒈦", face=(SUPP + "NotoSansCuneiform-Regular.ttf", 0), em=.0135, track=.04, at=.47, heavy=0, lines=(.016,), stamp="press",
         skin="cloth", pitch=.0009, colour=(.46, .35, .22), worn=(.62, .52, .38), rough=.82, spec=.2, ink=(.055, .035, .02), band=((.60, .52, .36), (.20, .12, .06)),
         w=.214, d=.148, h=.029, board=.0029, bulge=.0040, groove=.0065, x=-.460, y=.454, yaw=2.4),
    dict(kind="case", title="ΙΛΙΑΣ", face=("/System/Library/Fonts/Palatino.ttc", 0), em=.0135, track=.34, at=.5, heavy=0, lines=(.018,), stamp="gold",
         skin="cloth", pitch=.0007, colour=(.028, .052, .105), worn=(.16, .19, .24), rough=.78, spec=.25, band=((.05, .09, .22), (.50, .49, .43)),
         w=.226, d=.156, h=.034, board=.0031, bulge=.0046, groove=.0070, x=-.444, y=.450, yaw=-3.2),
    dict(kind="case", title="SHAKE-SPEARES SONNETS", face=(SUPP + "BigCaslon.ttf", 0), em=.0070, track=.03, at=.5, heavy=.00012, lines=(.0115, .0135), stamp="gold",
         skin="leather", cell=.00075, colour=(.125, .018, .016), worn=(.24, .07, .05), rough=.48, spec=.24, band=((.60, .50, .30), (.30, .03, .03)),
         w=.160, d=.104, h=.0215, board=.0024, bulge=.0030, groove=.0048, x=-.489, y=.470, yaw=3.6),
    dict(kind="paper", title="The C Programming Language", face=("/System/Library/Fonts/Helvetica.ttc", 0), em=.0066, track=.0, at=.40, author="Kernighan · Ritchie", author_em=.0048, author_at=.80, colour=(.80, .79, .76),
         w=.205, d=.158, h=.0185, x=-.464, y=.466, yaw=-1.3),
)
# the two that stand beside the pile with their fronts to the room, the volume leaning on the case. In metres: w across, h high, t thick; x and y where the left edge of the front meets the table; slip: a title slip's width and height, how far from the left and from the top
CASE = dict(title="李太白全集", w=.152, h=.245, t=.030, x=-.806, y=.500, yaw=-10.0, board=.0026, pitch=.00065, colour=(.055, .085, .14), worn=(.20, .23, .28), rough=.8, spec=.22,
            slip=(.0255, .147, .011, .012), em=.0172, step=.0225, clasps=(.30, .70), clasp_from_right=.0165)
VOLUME = dict(title="淮南子", w=.142, h=.235, t=.010, x=-.6869, y=.4394, yaw=-7.0, lean=7.0, bow=.005, colour=(.018, .030, .066), silk=(.56, .52, .42), thread=(.62, .60, .52),
              slip=(.0237, .141, .009, .010), em=.0165, step=.0275, stitch=.0125)
BOOK_PX = 12000                  # pixels of a title's picture to a metre: the shot that looks closest shows 3800 of them


def inked(b, em, text, at, track, px, heavy=0.0):
    """A line of type, white on black, on a picture of the book's back: its marks in the middle of the height, whatever the font's own metrics."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = int(b["w"] * px), int(b["h"] * px)
    f = ImageFont.truetype(b["face"][0], int(em * px), index=b["face"][1])
    steps = [f.getlength(c) + track * em * px for c in text]
    x = W * at - (sum(steps) - track * em * px) / 2
    tall = 2 * max(H, int(em * px * 3))
    im = Image.new("L", (W, tall), 0)
    d = ImageDraw.Draw(im)
    for c, step in zip(text, steps):
        d.text((x, tall * .6), c, 255, font=f, anchor="ls", stroke_width=int(heavy * px))
        x += step
    a = np.asarray(im, np.float32) / 255
    rows = np.where(a.max(1) > .5)[0]
    y0 = (rows[0] + rows[-1] + 1 - H) // 2
    return a[y0:y0 + H]


def uneven(a, seed, px, soft=.00008, wander=.09):
    """What a stamp or a printing plate leaves: not quite the drawn edge."""
    from scipy.ndimage import gaussian_filter
    b = gaussian_filter(a, soft * px)
    n = spectral(*a.shape, 1.1, seed)
    return np.clip((b - .5 + wander * n) * 5 + .5, 0, 1)


def stamped(i, b):
    """Where the die came down on a hardback's back: the title, and a rule or two towards each end."""
    a = inked(b, b["em"], b["title"], b["at"], b["track"], BOOK_PX, b["heavy"])
    H, W = a.shape
    a = a.copy()
    for off in b["lines"]:
        for x in (off, b["w"] - off):
            c = int(x * BOOK_PX)
            top, low = int((b["board"] * .6) * BOOK_PX), H - int((b["board"] * .6) * BOOK_PX)
            a[top:low, c - 2:c + 2] = 1
    from scipy.ndimage import gaussian_filter
    a = uneven(a, 300 + i, BOOK_PX)
    save(np.stack([a, gaussian_filter(a, .00016 * BOOK_PX), a], -1), f"title_{i}.png")
    rows = np.where(a[:, W // 3:W * 2 // 3].max(1) > .5)[0]
    print(b["title"], "字高", round((rows[-1] - rows[0] + 1) / BOOK_PX * 1000, 1), "mm，书脊", b["h"] * 1000, "mm", flush=True)


def creased(i, b):
    """A paperback's back: black type, and across it the white lines where it has been opened flat."""
    a = inked(b, b["em"], b["title"], b["at"], b["track"], BOOK_PX)
    a = np.maximum(a, inked(dict(b), b["author_em"], b["author"], b["author_at"], 0.0, BOOK_PX))
    H, W = a.shape
    r = np.random.default_rng(70 + i)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    crease = np.zeros_like(a)
    for x0, depth in zip(r.uniform(.06, .95, 5) * W, r.uniform(.3, 1, 5)):
        along = x0 + np.cumsum(r.normal(r.uniform(-.07, .07), .5, H))[:, None]
        fade = np.clip(r.uniform(.7, 1.3) - np.abs(yy / H - r.uniform(.3, .7)) * 2.4, 0, 1)
        crease = np.maximum(crease, depth * fade * np.exp(-((xx - along) / 2.6) ** 2))
    save(uneven(a, 300 + i, BOOK_PX, .00004, .05) * (1 - .85 * crease), f"title_{i}.png")
    save(crease, f"crease_{i}.png")
    print(b["title"], "/", b["author"], flush=True)


def sheets(n, r, thin, thick):
    """Sheets of paper seen edge on, row by row: how bright each row is, how high it stands, which sheet it belongs to."""
    tone, high, which = [], [], []
    k = 0
    while len(tone) < n:
        m = max(int(r.uniform(thin, thick)), 3)
        t = r.uniform(.84, 1.0)
        e = np.sin(np.linspace(0, np.pi, m)) ** .5
        tone += list(t * (.52 + .48 * e))
        high += list(e)
        which += [k] * m
        k += 1
    return np.array(tone[:n], np.float32), np.array(high[:n], np.float32), np.array(which[:n])


def pages():
    """A book's pages seen from the end, a tile 30 mm high: the sheets, gathered every so often, and never quite straight."""
    H, W = 4096, 1024
    tone, _, which = sheets(H, np.random.default_rng(3), 11, 22)
    gathered = np.random.default_rng(5).uniform(.86, 1.04, which.max() // 16 + 2)
    tone = tone * gathered[which // 16] * np.where((which % 16) == 15, .66, 1)
    yy = np.mgrid[0:H, 0:W][0]
    save(tone[(yy + (9 * spectral(H, W, 2.6, 4)).astype(int)) % H], "pages.png")


def weave():
    """A tile of plain weave, 64 threads across: each thread its own thickness and shade, thicker and thinner along its length, going over one and under the next. Its height, and how it looks in the light."""
    T, P = 64, 32
    N = T * P
    r = np.random.default_rng(7)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32) / P
    xx, yy = (xx + .28 * spectral(N, N, 2.6, 10)) % T, (yy + .28 * spectral(N, N, 2.6, 11)) % T
    ix, iy = np.floor(xx).astype(int) % T, np.floor(yy).astype(int) % T
    fx, fy = xx - ix - .5, yy - iy - .5
    over = np.where((ix + iy) % 2 == 0, 1.0, -1.0)
    warp_w = r.uniform(.66, 1.08, T)[ix] * (1 + .26 * spectral(N, T, 1.6, 8))[(yy * P).astype(int) % N, ix]
    weft_w = r.uniform(.66, 1.08, T)[iy] * (1 + .26 * spectral(T, N, 1.6, 9))[iy, (xx * P).astype(int) % N]
    warp = np.sqrt(np.clip(1 - (2 * fx / warp_w) ** 2, 0, 1)) * (.55 + .45 * np.cos(np.pi * fy) * over)
    weft = np.sqrt(np.clip(1 - (2 * fy / weft_w) ** 2, 0, 1)) * (.55 - .45 * np.cos(np.pi * fx) * over)
    h = np.maximum(warp, weft)
    shade = np.where(warp >= weft, r.uniform(.94, 1.04, T)[ix], r.uniform(.92, 1.03, T)[iy])
    save(h, "weave.png")
    save(shade * (.70 + .30 * h), "weave_tone.png")


def slip(c, name, seed):
    """A paper title slip: its title cut in a block and printed in a double frame, one character under the other, each upright, from the top."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = int(c["slip"][0] * BOOK_PX), int(c["slip"][1] * BOOK_PX)
    f = ImageFont.truetype(FONT, int(c["em"] * BOOK_PX), index=3)
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    out, inner = int(.0019 * BOOK_PX), int(.0031 * BOOK_PX)
    d.rectangle((out, out, W - 1 - out, H - 1 - out), outline=255, width=int(.00062 * BOOK_PX))
    d.rectangle((inner, inner, W - 1 - inner, H - 1 - inner), outline=255, width=int(.00024 * BOOK_PX))
    for k, ch in enumerate(c["title"]):
        d.text((W / 2, inner + .0045 * BOOK_PX + (k + .5) * c["step"] * BOOK_PX), ch, 255, font=f, anchor="mm", stroke_width=1)
    a = uneven(np.asarray(im, np.float32) / 255, seed, BOOK_PX, .00007, .12)
    # the block never quite inks everywhere
    a = a * np.clip(.92 + .09 * spectral(H, W, 1.3, seed + 1), 0, 1) * (spectral(H, W, .8, seed + 2) < 2.3)
    paper = np.array((.84, .81, .73), np.float32) * (1 + .03 * spectral(H, W, 1.6, seed + 3))[..., None]
    picture = paper * (1 - a[..., None]) + np.array((.016, .015, .016), np.float32) * a[..., None]
    save(linear_to_srgb(picture), name)
    print(c["title"], "书签", W, "×", H, flush=True)


def case_folds(c):
    """Where a case's front is not flat: the creases where the boards hinge and the cloth turns over the edges, and at each corner the cloth folded double, with the line of the fold. Red is its height about the middle, green how far it has rubbed."""
    w, h = c["w"], c["h"]
    H, W = int(h * BOOK_PX), int(w * BOOK_PX)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / BOOK_PX
    sides = (xx, w - xx, yy, h - yy)
    wander = .00028 * spectral(H, W, 2.6, 60)
    crease = np.maximum.reduce([np.exp(-((d + wander - at) / wide) ** 2) * deep for d, at, wide, deep in zip(sides, (.0027, .0027, .0017, .0017), (.00045, .00045, .00035, .00035), (1, 1, .55, .55))])
    raised = np.zeros_like(xx)
    for a, b, s in ((sides[0], sides[2], .0118), (sides[1], sides[2], .0105), (sides[0], sides[3], .0098), (sides[1], sides[3], .0112)):
        along = a + b + 1.6 * wander
        raised = np.maximum(raised, np.clip((s - along) / .0006, 0, 1))
        crease = np.maximum(crease, .8 * np.exp(-((along - s) / .00032) ** 2))
    edge = np.minimum.reduce(sides)
    rub = (np.exp(-edge / .0012) * .85 + .4 * crease + .3 * raised * np.exp(-edge / .004)) * np.clip(.75 + .3 * spectral(H, W, 1.4, 61), 0, 1.2)
    save(np.stack([np.clip(.5 + .26 * raised - .44 * crease, 0, 1), np.clip(rub, 0, 1), np.zeros_like(xx)], -1), "case_folds.png")


def books():
    pages()
    weave()
    for i, b in enumerate(BOOKS):
        {"case": stamped, "paper": creased}[b["kind"]](i, b)
    slip(CASE, "slip_case.png", 80)
    slip(VOLUME, "slip_volume.png", 84)
    case_folds(CASE)


def notebook():
    """The open page: ruled, with the week's small things on it. Written in the machine's handwriting faces, and small enough in the frame that nobody reads more than that it is written."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (1000, 1400), (238, 233, 220))
    d = ImageDraw.Draw(im)
    for y in range(120, 1380, 62):
        d.line((46, y, 954, y), (192, 200, 208), 2)
    from ink_subs import _font
    f = _font("xing", 58)
    pen = (38, 44, 74)
    for i, line in enumerate(("周六", "买菜", "给妈打电话", "论文第三段，数字再查", "还书", "问问它")):
        d.text((76, 120 + 62 * i * 2 - 14), line, pen, font=f, anchor="ls")
    d.line((70, 120 + 62 * 2 - 30, 200, 120 + 62 * 2 - 34), pen, 4)
    a = np.asarray(im, np.float32) / 255
    a *= (1 + .012 * spectral(1400, 1000, 1.0, 3)[..., None])
    save(a, "notebook.png")


def outside():
    """What the window looks out on, out of focus: a bright sky over the top of a tree. Only its light and its large shapes ever get to the camera. The picture's top is the plane's top, 5 m up, and its foot is 1 m down."""
    h, w = 1024, 2048
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32) / h
    z = -1 + 6 * (1 - yy)
    n = spectral(h, w, 2.4, 21) + .45 * spectral(h, w, 1.4, 22)
    leaf = np.clip((.5 * n + (1.5 - z) * 2.0) * 3.0, 0, 1)[..., None]
    sky = np.array((.97, .95, .90), np.float32) * (1 - .12 * yy[..., None])
    tone = (spectral(h, w, 2.0, 23) * .5 + .5)[..., None]
    green = np.array((.10, .14, .06), np.float32) + tone * np.array((.10, .13, .05), np.float32)
    save(linear_to_srgb(sky * (1 - leaf) + np.clip(green, 0, 1) * leaf), "outside.png")


def outside_night():
    """The same view at night (九, 十): the tree is the same tree, now a shape darker than the sky, and over and through it the block across the road, its windows lit or not. The sky is a warm near-black, browner low down where the city's light comes back off it; no blue. Stored as light, not as a picture: values over 1 are kept by a gain in the alpha's place, see night_gain."""
    h, w = 1024, 2048
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy, xx = yy / h, xx / w
    z = -1 + 6 * (1 - yy)
    x = -9.6 + 18 * xx
    n = spectral(h, w, 2.4, 21) + .45 * spectral(h, w, 1.4, 22)
    leaf = np.clip((.5 * n + (1.5 - z) * 2.0) * 3.0, 0, 1)
    sky = np.array((.010, .0075, .0055), np.float32) + np.clip(1 - (z + 1) / 5, 0, 1)[..., None] ** 2 * np.array((.030, .019, .010), np.float32)
    # the block across the road: 30 m off, so its 2.24 m windows come out about half a metre wide on a plane 7 m out
    r = np.random.default_rng(31)
    wall_top = 3.6
    face = (z < wall_top).astype(np.float32)
    lit = np.zeros((h, w, 3), np.float32)
    bay, floor_h = .78, .55
    for row in range(-2, int(wall_top / floor_h)):
        for col in range(int(-9.6 / bay) - 1, int(8.4 / bay) + 1):
            if r.random() > .46:
                continue
            x0, z0 = col * bay + .12, row * floor_h + .14
            x1, z1 = x0 + .40, z0 + .25
            warm = r.choice(((1.0, .70, .42), (1.0, .76, .52), (.95, .82, .64), (1.0, .62, .34)))
            level = r.uniform(.10, .38) * (.4 if r.random() < .25 else 1.0)
            m = (x >= x0) & (x < x1) & (z >= z0) & (z < z1)
            lit[m] = np.array(warm, np.float32) * level
    wall = np.array((.016, .012, .009), np.float32)
    picture = sky * (1 - face[..., None]) + (wall + lit * .9) * face[..., None]
    tree = np.array((.004, .0045, .003), np.float32)
    picture = picture * (1 - leaf[..., None]) + tree * leaf[..., None]
    # light: kept linear, scaled into 8 bits by night_gain
    np.save(os.path.join(TEX, "outside_night.npy"), picture.astype(np.float16))
    save(linear_to_srgb(picture / NIGHT_GAIN), "outside_night.png")


NIGHT_GAIN = 1.0       # outside_night.png holds light divided by this; film_room_night3d.py multiplies it back


def main():
    os.makedirs(TEX, exist_ok=True)
    if sys.argv[1:] == ["night"]:
        outside_night()
        print(TEX)
        return
    table()
    legends()
    books()
    notebook()
    outside()
    print(TEX)


if __name__ == "__main__":
    main()
