"""What the keyboard of the code shot is printed with, up close, and how its key moves. The layout and the legends follow apple.com's picture of the Magic Keyboard (USB-C), read on 2026-11-01: symbols on delete, return and shift, two legends on the punctuation, the caps 16.2 mm on a 18.8 mm pitch with the aluminium between them. Plain Python, once, on the Studio:

    python film_screens_tex.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
OUT = os.path.join(ROOT, "ink", "film_out", "film", "screens")
TEX = os.path.join(OUT, "tex")

U = .0188                        # the pitch of a key, in metres
GAP = .0026                      # the aluminium between two caps
WELL = .00026                    # the dark line between a cap and its hole
CAP_R = .0014                    # the corner of a cap
CAP_H = .00085                   # how far a cap stands above the aluminium: at the bottom of its travel it is just under it
TRAVEL = .0010                   # how far it goes down
DOWN, BOTTOM = 71.9, 72.16       # the return key starts down, and is down, at these film times: the last frame before the cut at 72.2 has it at the bottom

# from the right edge of the keys, row by row, 0 the function row: (name, its width in pitches, upper legend, lower legend). A name of one letter is centred, delete, return and shift are drawn
ROWS = (
    (("lock", 1, "", ""), ("F12", 1, "", "F12"), ("F11", 1, "", "F11"), ("F10", 1, "", "F10"), ("F9", 1, "", "F9"), ("F8", 1, "", "F8"), ("F7", 1, "", "F7"), ("F6", 1, "", "F6"), ("F5", 1, "", "F5"), ("F4", 1, "", "F4")),
    (("delete", 1.5, "", ""), ("=", 1, "+", "="), ("-", 1, "_", "-"), ("0", 1, ")", "0"), ("9", 1, "(", "9"), ("8", 1, "*", "8"), ("7", 1, "&", "7"), ("6", 1, "^", "6"), ("5", 1, "%", "5"), ("4", 1, "$", "4")),
    (("\\", 1, "|", "\\"), ("]", 1, "}", "]"), ("[", 1, "{", "["), ("P", 1, "", "P"), ("O", 1, "", "O"), ("I", 1, "", "I"), ("U", 1, "", "U"), ("Y", 1, "", "Y"), ("T", 1, "", "T"), ("R", 1, "", "R")),
    (("return", 1.75, "", ""), ("'", 1, "quotes", "quote"), (";", 1, ":", ";"), ("L", 1, "", "L"), ("K", 1, "", "K"), ("J", 1, "", "J"), ("H", 1, "", "H"), ("G", 1, "", "G"), ("F", 1, "", "F"), ("D", 1, "", "D")),
    (("shift", 2.25, "", ""), ("/", 1, "?", "/"), (".", 1, ">", "."), (",", 1, "<", ","), ("M", 1, "", "M"), ("N", 1, "", "N"), ("B", 1, "", "B"), ("V", 1, "", "V"), ("C", 1, "", "C")),
    (("right", 1, "", ""), ("updown", 1, "", ""), ("left", 1, "", ""), ("option", 1, "", "option"), ("command", 1.25, "", "command"), ("space", 5.0, "", "")),
)
RETURN_ROW = 3
# the part of the keyboard that is printed, in metres, the return row's middle at y 0 and the keys' right edge at x 0
AREA = (-.190, .006, -.052, .070)
PX = 50000                       # pixels of the legends to a metre
# the post's panel: a 15.6-inch 1366 by 768 of the time. The part of it that is rendered, in its own pixels; the texture's pixels to one of the panel's; the width of one, in metres
S5_CROP = (120, 0, 1040, 540)
LCD = 8
PIXEL = .000252


def keycaps():
    """Every cap as (name, upper, lower, x0, y0, x1, y1): y up the keyboard, away from whoever types."""
    out = []
    for r, row in enumerate(ROWS):
        x = 0.0
        yc = (RETURN_ROW - r) * U
        for name, width, upper, lower in row:
            x0, x1 = x - width * U, x
            out.append((name, upper, lower, x0 + GAP / 2, yc - U / 2 + GAP / 2, x1 - GAP / 2, yc + U / 2 - GAP / 2))
            x = x0
    return out


def sm(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def travel(t, release=None):
    """How far the return key has gone down at film time t. A third of the way against the dome, slowly, then the dome gives and the plate stops it dead. Let go, it comes up faster than it went and passes its rest by a hair."""
    u = (t - DOWN) / (BOTTOM - DOWN)
    down = TRAVEL * (.32 * sm(u / .6) + .68 * sm((u - .5) / .5) ** 1.4)
    if release is None or t < release:
        return float(down)
    x = (t - release) / .045
    return float(TRAVEL * np.exp(-x * 1.2) * np.cos(x * 1.6) * (1 - .12 * np.exp(-x)) if x < 7 else 0.0)


def spectral(n, m, beta, seed):
    r = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(m)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1
    v = np.fft.irfft2((r.normal(size=f.shape) + 1j * r.normal(size=f.shape)) / f ** (beta / 2), s=(n, m))
    return ((v - v.mean()) / v.std()).astype(np.float32)


def _px(x, y):
    return (x - AREA[0]) * PX, (AREA[3] - y) * PX


def _font(size, weight=300):
    from PIL import ImageFont
    f = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", size)
    try:
        axes = f.get_variation_axes()
        f.set_variation_by_axes([max(a["minimum"], min(a["maximum"], weight)) if a["name"] in (b"Weight", "Weight") else a["default"] for a in axes])
    except OSError:
        pass
    return f


def _line(d, pts, width, pen):
    p = [_px(*q) for q in pts]
    d.line(p, fill=pen, width=int(round(width * PX)), joint="curve")
    r = width * PX / 2
    for x, y in (p[0], p[-1]):
        d.ellipse((x - r, y - r, x + r, y + r), fill=pen)


def _arc(a, b, c, n=8):
    """From a to c round the corner b."""
    return [tuple((1 - t) ** 2 * np.array(a) + 2 * t * (1 - t) * np.array(b) + t * t * np.array(c)) for t in np.linspace(0, 1, n)]


def _symbol(d, name, x1, y0, pen):
    """delete, return, shift and the arrows, drawn, as they are on the caps, in thin line at the right of the foot of the key. x1, y0: the cap's right and lower edges."""
    w = .00028
    P = lambda dx, dy: (x1 + dx, y0 + dy)
    if name == "return":
        # as on the cap: from the left along the top, round the right, and back along the foot to an arrow
        l, r, low, top, h = -.0086, -.0047, .0054, .0076, .0011
        _line(d, [P(l + .0014, top), P(r - h, top)] + _arc(P(r - h, top), P(r, top), P(r, top - h)) + _arc(P(r, top - h), P(r, low), P(r - h, low)) + [P(l, low)], w, pen)
        _line(d, [P(l + .0012, low - .0012), P(l, low), P(l + .0012, low + .0012)], w, pen)
    elif name == "delete":
        _line(d, [P(-.00660, .00440), P(-.00555, .00580), P(-.00300, .00580), P(-.00280, .00560), P(-.00280, .00320), P(-.00300, .00300), P(-.00555, .00300), P(-.00660, .00440)], w, pen)
        _line(d, [P(-.00500, .00510), P(-.00360, .00370)], w, pen)
        _line(d, [P(-.00360, .00510), P(-.00500, .00370)], w, pen)
    elif name == "shift":
        _line(d, [P(-.00440, .00590), P(-.00270, .00430), P(-.00360, .00430), P(-.00360, .00300), P(-.00520, .00300), P(-.00520, .00430), P(-.00610, .00430), P(-.00440, .00590)], w, pen)
    elif name in ("left", "right"):
        s = 1 if name == "right" else -1
        cx, cy = x1 - .0081, y0 + .0081
        d.polygon([_px(cx + s * .0010, cy), _px(cx - s * .0006, cy + .0009), _px(cx - s * .0006, cy - .0009)], fill=pen)
    elif name == "updown":
        cx, cy = x1 - .0081, y0 + .0081
        for s in (1, -1):
            d.polygon([_px(cx, cy + s * .0040), _px(cx - .0008, cy + s * .0026), _px(cx + .0008, cy + s * .0026)], fill=pen)
    elif name == "lock":
        cx, cy = x1 - .0081, y0 + .0081
        _line(d, [(cx - .0022, cy - .0024), (cx + .0022, cy - .0024), (cx + .0022, cy + .0006), (cx - .0022, cy + .0006), (cx - .0022, cy - .0024)], w, pen)
        _line(d, _arc((cx - .0013, cy + .0006), (cx - .0013, cy + .0028), (cx, cy + .0028)) + _arc((cx, cy + .0028), (cx + .0013, cy + .0028), (cx + .0013, cy + .0006)), w, pen)


def legends():
    """The caps' colour and what is lasered on them, and how much the fingers have polished each."""
    from PIL import Image, ImageDraw
    W, H = int((AREA[1] - AREA[0]) * PX), int((AREA[3] - AREA[2]) * PX)
    cap = (240, 240, 235)
    im = Image.new("RGB", (W, H), cap)
    d = ImageDraw.Draw(im)
    pen = (74, 74, 78)
    big, small, tiny = _font(int(.0060 * PX), 340), _font(int(.0050 * PX), 340), _font(int(.0021 * PX), 340)
    for name, upper, lower, x0, y0, x1, y1 in keycaps():
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if name in ("return", "delete", "shift", "left", "right", "updown", "lock"):
            _symbol(d, name, x1, y0, pen)
        elif name == "'":
            # both ticks lean, on the cap
            for dy, dxs in ((.0037, (-.0006, .0006)), (-.0037, (0,))):
                for dx in dxs:
                    _line(d, [(cx + dx + .00026, cy + dy + .0010), (cx + dx - .00026, cy + dy - .0010)], .00034, pen)
        elif len(name) == 1 and not upper and name.isalpha():
            d.text(_px(cx, cy), lower, pen, font=big, anchor="mm")
        elif name in ("option", "command"):
            d.text(_px(cx, y0 + .0024), lower, pen, font=tiny, anchor="mm")
            if name == "command":
                d.text(_px(cx + .0040, y1 - .0045), "⌘", pen, font=_font(int(.0034 * PX)), anchor="mm")
        elif name.startswith("F") and len(name) > 1:
            d.text(_px(cx, y0 + .0030), lower, pen, font=tiny, anchor="mm")
        else:
            digit = lower.isdigit()
            d.text(_px(cx, cy + .0037), upper, pen, font=small, anchor="mm")
            d.text(_px(cx, cy - .0035), lower, pen, font=big if digit else small, anchor="mm")
    im.save(os.path.join(TEX, "keys_macro.png"))
    # polish: the return key most, under where a finger lands, and the keys beside it a little
    rough = np.full((H // 4, W // 4), .46, np.float32)
    yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32)
    x, y = AREA[0] + (xx + .5) * 4 / PX, AREA[3] - (yy + .5) * 4 / PX
    uneven = np.clip(.85 + .2 * spectral(H // 4, W // 4, 3.6, 7), .4, 1.15)
    for name, _, _, x0, y0, x1, y1 in keycaps():
        strength = dict(zip(("return", "'", ";", "L", "P", "[", "shift", "delete"), (.21, .05, .07, .09, .07, .04, .05, .08))).get(name, .03)
        cx, cy = x0 + (x1 - x0) * (.60 if name == "return" else .5), (y0 + y1) / 2 + .0006
        rough -= strength * uneven * np.exp(-((x - cx) / (.0062 if name != "return" else .0088)) ** 2 - ((y - cy) / .0052) ** 2)
    Image.fromarray(np.uint8(np.clip(rough, 0, 1) * 255 + .5)).save(os.path.join(TEX, "keys_macro_rough.png"))
    print("legends", W, H, flush=True)


def glass():
    """What lies on the glass of a screen somebody uses: R a little dust, G where a cloth went and left a haze. One pixel to 0.146 mm of the Studio Display's glass."""
    from PIL import Image
    W, H = 4096, 2304
    r = np.random.default_rng(23)
    dust = np.zeros((H, W), np.float32)
    for _ in range(1100):
        x, y, s = int(r.uniform(0, W - 3)), int(r.uniform(0, H - 3)), r.uniform(.25, .9)
        dust[y, x] = s
        if r.random() < .3:
            dust[y, x + 1] = s * r.uniform(.3, .9)
        if r.random() < .2:
            dust[y + 1, x] = s * r.uniform(.3, .9)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    haze = np.zeros((H, W), np.float32)
    for cx, cy, rad, width, strength in ((900, -600, 1500, 150, .8), (300, 2600, 2300, 200, .55), (1800, 300, 1100, 120, .5)):
        haze += strength * np.exp(-((np.hypot(xx - cx, yy - cy) - rad) / width) ** 2)
    haze *= np.clip(.75 + .4 * spectral(H, W, 2.6, 41), 0, 1.3)
    Image.fromarray(np.uint8(np.clip(np.stack((dust, np.clip(haze, 0, 1), np.zeros_like(dust)), -1), 0, 1) * 255 + .5)).save(os.path.join(TEX, "glass_dust.png"))
    print("glass", flush=True)


if __name__ == "__main__":
    os.makedirs(TEX, exist_ok=True)
    legends()
    glass()
