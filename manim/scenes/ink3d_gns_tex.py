"""Textures for chapters 11-13 of the ink film (ink3d_grief.py, ink3d_names.py, ink3d_new.py), drawn where Pillow and scipy exist, read by Blender as .npy.

    python ink3d_gns_tex.py all|taotie|steles|glyphs|tong|hands|blog|newchar|nearest

Writes to ink/film_out/3d/_gns/.
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from ink_film import OUT, font, glyph_mask

TEX = os.path.join(OUT, "3d", "_gns")


def noise(h, w, beta, seed):
    r = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1
    s = (r.normal(size=f.shape) + 1j * r.normal(size=f.shape)) / f ** (beta / 2)
    s[0, 0] = 0
    v = np.fft.irfft2(s, s=(h, w))
    return ((v - v.mean()) / v.std()).astype(np.float32)


def spline(pts, n=240):
    """Catmull-Rom through pts."""
    p = np.array(pts, float)
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(p) - 2):
        for u in np.linspace(0, 1, n // (len(p) - 3), endpoint=False):
            a, b, c, d = p[i - 1], p[i], p[i + 1], p[i + 2]
            out.append(.5 * ((2 * b) + (-a + c) * u + (2 * a - 5 * b + 4 * c - d) * u * u + (-a + 3 * b - 3 * c + d) * u ** 3))
    out.append(p[-2])
    return np.array(out)


def stroke(d, pts, w0, w1, sx, sy, fill=255):
    """A tapered line through normalised points, stamped with circles."""
    q = spline(pts)
    for i, (x, y) in enumerate(q):
        u = i / (len(q) - 1)
        r = (w0 + (w1 - w0) * u) * sy
        d.ellipse([x * sx - r, y * sy - r, x * sx + r, y * sy + r], fill=fill)


def spiral(d, cx, cy, s, turns, w, flip):
    """A square spiral: the thunder pattern (云雷纹)."""
    x, y = cx, cy
    step = s / (2 * turns + 1)
    L = step
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    k = 0
    pts = [(x, y)]
    for i in range(int(turns * 4)):
        dx, dy = dirs[(k + flip) % 4]
        x, y = x + dx * L, y + dy * L
        pts.append((x, y))
        k += 1
        if i % 2 == 1:
            L += step
    d.line(pts, fill=255, width=w, joint="curve")


def chamfer(pts, k=2):
    """Corners cut twice, as a mould-maker's knife leaves them: straight runs and small turns, not a noodle."""
    p = np.array(pts, float)
    for _ in range(k):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            q += [a * .78 + b * .22, a * .22 + b * .78]
        q.append(p[-1])
        p = np.array(q)
    return p


def blade(d, pts, w0, w1, S, fill=255, tip=.0):
    """A raised stroke with straight flanks, narrowing from w0 to w1 and, if tip is set, coming to a point over the last part."""
    q = chamfer(pts)
    seg = np.hypot(*np.diff(q, axis=0).T)
    u = np.r_[0, np.cumsum(seg)] / seg.sum()
    wid = w0 + (w1 - w0) * u
    if tip:
        wid = wid * np.clip((1 - u) / tip, 0, 1) ** .7
    sx, sy = S
    for (a, b, wa, wb) in zip(q[:-1], q[1:], wid[:-1], wid[1:]):
        a, b = a * (sx, sy), b * (sx, sy)
        n = b - a
        n = np.array([-n[1], n[0]]) / (np.hypot(*n) + 1e-6)
        d.polygon([tuple(a + n * wa * sy), tuple(b + n * wb * sy), tuple(b - n * wb * sy), tuple(a - n * wa * sy)], fill=fill)
        rb = wb * sy
        d.ellipse([b[0] - rb, b[1] - rb, b[0] + rb, b[1] + rb], fill=fill)


def taotie():
    """One band of three taotie masks, as a height map. The nose lies on the panel's centre line and the face is mirrored about it. As on a Shang casting the
    forms are flat-topped with crisp shoulders, each with a sunk line running just inside its edge; thunder spirals fill the ground behind."""
    PW, PH = 1650, 1000          # half a panel
    main = Image.new("L", (PW, PH), 0)
    d = ImageDraw.Draw(main)
    S = (PW, PH)
    # forehead: a shield on the centre line
    d.polygon([(0, .05 * PH), (.058 * PW, .05 * PH), (.112 * PW, .165 * PH), (.058 * PW, .30 * PH), (0, .345 * PH)], fill=255)
    # nose, and the nostril's turn
    blade(d, [(0, .36), (0, .60), (.02, .69), (.092, .725), (.135, .675), (.105, .625)], .05, .022, S)
    # brow, its outer end hooked up
    blade(d, [(.045, .345), (.125, .275), (.28, .258), (.365, .305), (.415, .235)], .036, .018, S, tip=.14)
    # horn: a great C, curling back in on itself
    blade(d, [(.085, .215), (.13, .095), (.285, .05), (.445, .06), (.525, .135), (.51, .225), (.43, .245), (.385, .18), (.43, .135)], .056, .021, S)
    # ear
    blade(d, [(.445, .325), (.525, .295), (.575, .35), (.54, .44), (.465, .44), (.455, .39)], .032, .016, S)
    # cheek, up from the nostril and round the eye
    blade(d, [(.135, .645), (.245, .69), (.365, .635), (.42, .525)], .03, .016, S, tip=.2)
    # fangs
    blade(d, [(.175, .685), (.19, .845)], .036, .03, S, tip=.55)
    blade(d, [(.305, .665), (.345, .805)], .03, .026, S, tip=.55)
    # lower jaw
    blade(d, [(.05, .79), (.12, .895), (.295, .905), (.38, .84)], .03, .018, S, tip=.3)
    # body: out from behind the head, rising to a tail that hooks back on itself
    blade(d, [(.455, .545), (.54, .47), (.715, .445), (.84, .405), (.93, .305), (.945, .165), (.865, .10), (.765, .125), (.745, .215), (.815, .25)], .072, .021, S)
    # fins along its back
    for x0, y0 in ((.60, .435), (.695, .425), (.79, .395)):
        blade(d, [(x0, y0), (x0 + .004, y0 - .085), (x0 + .068, y0 - .115)], .024, .012, S, tip=.25)
    # foreleg and claws
    blade(d, [(.505, .545), (.57, .67), (.545, .80), (.46, .865), (.375, .86)], .06, .02, S, tip=.12)
    blade(d, [(.45, .865), (.42, .955)], .022, .02, S, tip=.8)
    blade(d, [(.395, .855), (.35, .945)], .02, .018, S, tip=.8)
    # hind leg and claws
    blade(d, [(.765, .495), (.825, .645), (.81, .795), (.735, .865), (.655, .86)], .054, .018, S, tip=.12)
    blade(d, [(.72, .865), (.695, .955)], .02, .018, S, tip=.8)
    m = (np.asarray(main, np.float32) / 255 > .5).astype(np.float32)

    # the eye, shaped like the character 臣: an almond with hooked corners, the pupil standing out of it
    yy, xx = np.mgrid[:PH, :PW].astype(np.float32)
    ex, ey = .228 * PW, .462 * PH
    ca, sa = math.cos(-.16), math.sin(-.16)
    u, v = (xx - ex) * ca - (yy - ey) * sa, (xx - ex) * sa + (yy - ey) * ca
    almond = (1 - (np.abs(u) / 196) ** 1.55 - (np.abs(v) / 84) ** 1.25 > 0).astype(np.float32)
    im = Image.new("L", (PW, PH), 0)
    di = ImageDraw.Draw(im)
    blade(di, [(.125, .535), (.098, .585), (.125, .615)], .02, .012, S, tip=.25)
    blade(di, [(.335, .425), (.395, .405)], .02, .014, S, tip=.5)
    almond = np.maximum(almond, (np.asarray(im, np.float32) / 255 > .5).astype(np.float32))
    r = np.sqrt((xx - ex) ** 2 + (yy - ey) ** 2)
    pupil = np.clip(1 - (r / 62) ** 2, 0, 1) ** .42
    socket = np.exp(-((r - 76) / 8) ** 2) * (almond > .5)

    # thunder spirals over the ground
    gr = Image.new("L", (PW, PH), 0)
    dg = ImageDraw.Draw(gr)
    s = 84
    for iy in range(0, PH // s + 1):
        for ix in range(0, PW // s + 1):
            spiral(dg, ix * s + s / 2, iy * s + s / 2, s * .9, 2.5, 8, (ix + iy) % 2 * 2)
    spirals = gaussian_filter(np.asarray(gr, np.float32) / 255, .8)
    mm = np.maximum(m, almond)
    clear = distance_transform_edt(mm < .5).astype(np.float32) > 13
    spirals = spirals * clear

    # flat tops, crisp shoulders, a line sunk just inside every edge
    edt = distance_transform_edt(mm > .5).astype(np.float32)
    top = 1 - np.exp(-edt / 3.5)
    inner = np.exp(-((edt - 15) / 2.6) ** 2) * (edt > 9)
    h = .20 * spirals + top * (1 - .34 * inner)
    h = h * (1 - .62 * socket) + pupil * .85

    # bosses top and bottom
    rows = np.zeros((PH, PW), np.float32)
    rd = 19
    for y0 in (rd + 3, PH - rd - 3):
        for x0 in np.arange(rd, PW, rd * 2.7):
            rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
            rows = np.maximum(rows, np.clip(1 - (rr / rd) ** 2, 0, 1) ** .5)
    band = np.zeros((PH, PW), np.float32)
    band[:46] = 1
    band[-46:] = 1
    h = h * (1 - band) + rows * .62 * band
    # flange on the nose line, a thinner one on the seam between masks
    fl = np.clip(1 - (xx / 30) ** 2, 0, 1) ** .4
    sl = np.clip(1 - ((PW - xx) / 18) ** 2, 0, 1) ** .4
    h = np.maximum(h, np.maximum(fl * 1.7, sl * 1.15))
    h = gaussian_filter(h, .8)

    panel = np.hstack([h[:, ::-1], h])              # mirrored about the nose
    relief = np.hstack([panel, panel, panel])
    # casting flaws
    n = noise(relief.shape[0], relief.shape[1], 1.6, 8)
    relief = relief + .014 * n
    relief -= (np.clip(noise(relief.shape[0], relief.shape[1], 2.2, 9) - 1.7, 0, 1) * .10)
    np.save(os.path.join(TEX, "taotie.npy"), relief.astype(np.float16))
    print("taotie", relief.shape)
    im = (np.clip(relief, 0, 1.7) / 1.7 * 255).astype(np.uint8)
    Image.fromarray(im[:, :PW * 2]).save(os.path.join(TEX, "taotie_preview.png"))


def stele_glyph(ch, kind, name, W=1600, H=3000, frac=.72):
    """A character for a stone slab: how deep the chisel went, a V at every stroke, and where it sits on the slab."""
    size = int(W * frac)
    m = glyph_mask(ch, kind, size, 8)
    canvas = np.zeros((H, W), np.float32)
    y0 = int(H * .48 - m.shape[0] / 2)
    x0 = (W - m.shape[1]) // 2
    canvas[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    edt = distance_transform_edt(canvas > .5).astype(np.float32)
    v = 1 - np.exp(-edt / 26)
    np.save(os.path.join(TEX, f"stele_{name}.npy"), v.astype(np.float16))
    print("stele", ch, v.max())


def stele_column(s, kind, name, W=1000, H=3400, frac=.72):
    """A column of characters for a slab."""
    n = len(s)
    size = int(min(W * frac, H * .88 / n))
    canvas = np.zeros((H, W), np.float32)
    y = int((H - size * 1.05 * n) / 2)
    centres = []
    for ch in s:
        m = glyph_mask(ch, kind, size, 8)
        x0 = (W - m.shape[1]) // 2
        canvas[y:y + m.shape[0], x0:x0 + m.shape[1]] = np.maximum(canvas[y:y + m.shape[0], x0:x0 + m.shape[1]], m)
        centres.append((y + m.shape[0] / 2) / H)
        y += int(size * 1.05)
    edt = distance_transform_edt(canvas > .5).astype(np.float32)
    v = 1 - np.exp(-edt / 16)
    np.save(os.path.join(TEX, f"stele_{name}.npy"), v.astype(np.float16))
    np.save(os.path.join(TEX, f"stele_{name}_centres.npy"), np.array(centres, np.float32))


def glyphs():
    """Characters to float on the water."""
    for ch in "悲愁泪痛哀恨苦伤我难过受忘得":
        m = glyph_mask(ch, "kaib", 256, 12)
        np.save(os.path.join(TEX, f"g_{ord(ch):x}.npy"), m.astype(np.float16))
    for s in (".71", ".14", ".06", ".04"):
        im = Image.new("L", (160, 72), 0)
        ImageDraw.Draw(im).text((80, 36), s, 255, font=font("kai", 56), anchor="mm")
        np.save(os.path.join(TEX, f"n_{s[1:]}.npy"), (np.asarray(im, np.float32) / 255).astype(np.float16))


def hands(S=512):
    """Other people's 痛, for the rings: off a stele, out of a letter, off a screen, from a printed page, from a copybook, a schoolgirl's. Each is made the way that thing is made."""
    Y, X = np.mgrid[:S, :S].astype(np.float32)

    def move(m, k, beta, seed):
        return map_coordinates(m, [Y + k * noise(S, S, beta, seed), X + k * noise(S, S, beta, seed + 1)], order=1)

    # a stele: the clerical hand, the edges chipped, the strokes pitted and broken by a thousand winters
    m = move(glyph_mask("痛", "li", S, 22), 2.2, 2.0, 301)
    pits = sm_((noise(S, S, 1.5, 303) + .9) / .5) * sm_((noise(S, S, 2.6, 304) + 1.45) / .4)
    np.save(os.path.join(TEX, "hand_stele.npy"), np.clip(gaussian_filter(m * pits, .7), 0, 1).astype(np.float16))
    # a letter: a running hand in a wet brush, feathering into the paper, a dry patch where the brush ran out
    m = move(glyph_mask("痛", "xing", S, 22), 1.4, 2.2, 311)
    feather = gaussian_filter(m, 3.2) * (.75 + .25 * noise(S, S, 1.2, 313))
    np.save(os.path.join(TEX, "hand_letter.npy"), np.clip(np.maximum(m * (.76 + .24 * sm_((noise(S, S, 2.8, 314) + 1) / 1.4)), feather), 0, 1).astype(np.float16))
    # a screen: a 16-dot bitmap font, the dots still square
    a = glyph_mask("痛", "song", 16, 1)
    dots = (a > .36).astype(np.float32)
    np.save(os.path.join(TEX, "hand_screen.npy"), np.kron(dots, np.ones((S // dots.shape[0], S // dots.shape[1]), np.float32)).astype(np.float16))
    # a printed page and a copybook
    np.save(os.path.join(TEX, "hand_print.npy"), np.clip(glyph_mask("痛", "song", S, 22), 0, 1).astype(np.float16))
    np.save(os.path.join(TEX, "hand_copy.npy"), np.clip(glyph_mask("痛", "kaib", S, 22), 0, 1).astype(np.float16))
    # someone young, with a fine pen, not sure of the proportions
    t = distance_transform_edt(glyph_mask("痛", "kai", S, 22) > .5)
    np.save(os.path.join(TEX, "hand_pen.npy"), np.clip(gaussian_filter(move(sm_(t / 5.0), 5.5, 2.8, 321), .8), 0, 1).astype(np.float16))


BLOG = ("今天又说肚子疼，没有去上学。", "其实不疼。我只是不想走那条路。", "下午一直下雨，我把窗户开着听。", "我很难过，可是说不出是为什么。", "妈妈问我，我说没事。", "写在这里，反正也没有人看。")


def blog(W=1600, H=1200):
    """A small blog from about 2009 on an old monitor: a page laid out in a table, with the month and a visitor count down the side and nothing bigger than a title has to be. Written
    for the film: a girl of fourteen, an ordinary day. Saves blog.npy (the light of the screen) and blog_nan.npy (u, v of the 难 in 难过, where the thread lands)."""
    im = Image.new("L", (W, H), 244)
    d = ImageDraw.Draw(im)
    song = lambda z: font("song", z)
    # the browser, only just: the address, the scrollbar
    d.rectangle((0, 0, W, 50), fill=212)
    d.line((0, 50, W, 50), fill=150, width=2)
    d.rectangle((14, 8, 1560, 42), fill=250, outline=150)
    d.text((26, 25), "http://xiaoman.blog.example/2009/03/17.html", 70, font=font("mono", 26), anchor="lm")
    d.rectangle((W - 36, 50, W, H), fill=222)
    d.line((W - 36, 50, W - 36, H), fill=160, width=2)
    d.rectangle((W - 32, 96, W - 4, 420), fill=188, outline=140)
    # the head of the page
    d.text((44, 98), "小满的小窝", 30, font=song(44), anchor="lm", stroke_width=1, stroke_fill=30)
    x = 44
    for nav in ("首页", "日志", "相册", "留言"):
        d.text((x, 150), nav, 55, font=song(36), anchor="lm")
        w_ = song(36).getlength(nav)
        d.line((x, 172, x + w_, 172), fill=55, width=2)
        x += w_ + 34
    d.line((44, 190, W - 76, 190), fill=170, width=2)
    # the side: the month, the count, the last few posts
    sx = 1168
    d.line((sx - 28, 190, sx - 28, H - 40), fill=200, width=2)
    d.text((sx, 224), "2009年3月", 45, font=song(34), anchor="lm")
    for j, wd in enumerate("日一二三四五六"):
        d.text((sx + 8 + j * 50, 272), wd, 90, font=song(26), anchor="lm")
    for i in range(1, 32):       # the first of March 2009 was a Sunday
        col, row = (i - 1) % 7, (i - 1) // 7
        at = (sx + 8 + col * 50, 314 + row * 42)
        if i == 17:
            d.rectangle((at[0] - 6, at[1] - 19, at[0] + 34, at[1] + 19), outline=70, width=2)
        d.text(at, str(i), 60, font=font("mono", 24), anchor="lm")
    d.text((sx, 580), "访问：000127", 60, font=song(30), anchor="lm")
    d.text((sx, 640), "最近日志", 45, font=song(30), anchor="lm")
    for k, t in enumerate(("下雨", "同桌搬走了", "数学", "新的本子")):
        d.text((sx, 692 + k * 46), t, 70, font=song(28), anchor="lm")
        d.line((sx, 692 + k * 46 + 16, sx + song(28).getlength(t), 692 + k * 46 + 16), fill=120, width=1)
    # the post
    d.text((44, 232), "下雨", 34, font=song(48), anchor="lm", stroke_width=1, stroke_fill=34)
    d.text((44, 290), "2009-03-17  21:47  星期二  阴", 92, font=song(34), anchor="lm")
    d.line((44, 324, 1110, 324), fill=205, width=2)
    f = song(58)
    y0, lh = 384, 86
    for i, line in enumerate(BLOG):
        d.text((44, y0 + i * lh), line, 26, font=f, anchor="lm")
    i = 3
    u = (44 + f.getlength("我很") + 29) / W
    v = (y0 + i * lh) / H
    x = 44
    for t in ("阅读(3)", "评论(0)", "上一篇", "下一篇"):
        d.text((x, 936), t, 70, font=song(34), anchor="lm")
        x += song(34).getlength(t) + 36
    d.line((44, 972, 1110, 972), fill=205, width=2)
    d.text((44, 1010), "还没有人留言。", 110, font=song(32), anchor="lm")
    a = np.asarray(im, np.float32) / 255
    # the tube: dimmer toward the corners, a dead speck or two, old water dried on the glass
    Y, X = np.mgrid[:H, :W].astype(np.float32)
    r = np.hypot((X - W / 2) / (W / 2), (Y - H / 2) / (H / 2))
    a = a * (1 - .20 * np.clip(r / 1.41, 0, 1) ** 3.5)
    a = gaussian_filter(a, .8) * (1 + .012 * noise(H, W, 1.2, 331))
    rng = np.random.default_rng(337)
    for k in range(9):
        cx, cy, rad = rng.uniform(0, W), rng.uniform(H * .3, H), rng.uniform(14, 60)
        d_ = np.hypot(X - cx, (Y - cy) * 1.05)
        a = a * (1 - .055 * np.exp(-((d_ - rad) / 2.6) ** 2) * sm_((noise(H, W, 2.6, 340 + k) + .7) / .6))
    np.save(os.path.join(TEX, "blog.npy"), np.clip(a, 0, 1).astype(np.float16))
    np.save(os.path.join(TEX, "blog_nan.npy"), np.array([u, v], np.float32))


def tong():
    """痛 at large size, lying on the water: thick wet strokes, the edges a little soaked."""
    m = glyph_mask("痛", "kaih", 1400, 40)
    S = m.shape[0]
    Y, X = np.mgrid[:S, :S].astype(np.float32)
    m = map_coordinates(m, [Y + 5 * noise(S, S, 2.4, 41) + 1.6 * noise(S, S, 1.4, 42), X + 5 * noise(S, S, 2.4, 43) + 1.6 * noise(S, S, 1.4, 44)], order=1)
    np.save(os.path.join(TEX, "tong_big.npy"), np.clip(gaussian_filter(m, 1.2), 0, 1).astype(np.float16))


# The new character. Not in any dictionary: 礻 from 神, 甶 from 鬼, 人 between, and the 口口 of 器 at the foot, built as ⿰礻⿱甶⿱人吅.
# Each part is cut from a real character of the same brush, so the strokes are right; the whole is the structure no one has written.
def sm_(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def cut(ch, kind, box, size):
    """A part of a character: box is (x0, y0, x1, y1) as fractions of the glyph's square."""
    m = glyph_mask(ch, kind, size, 0)
    h, w = m.shape
    x0, y0, x1, y1 = box
    return m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def paste(c, m, x, y, w, h):
    """m resized into the box (x, y, w, h) of canvas c."""
    im = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).resize((int(w), int(h)), Image.LANCZOS)
    a = np.asarray(im, np.float32) / 255
    c[int(y):int(y) + a.shape[0], int(x):int(x) + a.shape[1]] = np.maximum(c[int(y):int(y) + a.shape[0], int(x):int(x) + a.shape[1]], a)


def nearest(top=12, size=96):
    """Is the new character really new? Compares its shape against every CJK character the system's fonts can draw (unified ideographs and the extensions they cover)
    and writes out the nearest ones to look at. Not proof: it cannot reach characters no installed font has. See the report."""
    from PIL import ImageFont
    fonts = ["/System/Library/Fonts/Supplemental/Songti.ttc", "/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc", "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]
    fonts = [f for f in fonts if os.path.exists(f)]
    me = np.load(os.path.join(TEX, "newchar_mask.npy")).astype(np.float32)
    def norm(a):
        ys, xs = np.nonzero(a > .5)
        if len(xs) < 8:
            return None
        b = Image.fromarray((a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
        v = gaussian_filter(np.asarray(b, np.float32) / 255, 1.6)
        return (v - v.mean()) / (v.std() + 1e-6)
    tgt = norm(me)
    ranges = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF), (0x2A700, 0x2EBEF), (0x30000, 0x3134F)]
    scores, seen = [], 0
    for path in fonts:
        f = ImageFont.truetype(path, size)
        tofu = np.asarray(f.getmask(chr(0x10FFFD), mode="L"))
        for lo, hi in ranges:
            for cp in range(lo, hi + 1):
                m_ = f.getmask(chr(cp), mode="L")
                if m_.size[0] == 0:
                    continue
                arr = np.array(m_, np.float32).reshape(m_.size[1], m_.size[0]) / 255
                if arr.shape == tofu.shape and tofu.size and abs(arr.mean() - tofu.mean() / 255) < 1e-4:
                    continue
                v = norm(arr)
                if v is None:
                    continue
                seen += 1
                scores.append((float((v * tgt).mean()), cp, path))
    scores.sort(reverse=True)
    with open(os.path.join(TEX, "newchar_nearest.txt"), "w") as log:
        print("compared", seen, "glyphs from", [os.path.basename(f) for f in fonts], file=log)
        for sc, cp, path in scores[:top]:
            print(f"{sc:.3f} U+{cp:05X} {chr(cp)} {os.path.basename(path)}", file=log)
    # a contact sheet: ours first, then the nearest
    cell = 180
    sheet = Image.new("L", (cell * (top + 1), cell), 255)
    sheet.paste(Image.fromarray(255 - (np.clip(me, 0, 1) * 255).astype(np.uint8)).resize((cell, cell)), (0, 0))
    d = ImageDraw.Draw(sheet)
    for i, (sc, cp, path) in enumerate(scores[:top]):
        f = ImageFont.truetype(path, cell - 24)
        d.text((cell * (i + 1) + 12, 6), chr(cp), 0, font=f)
    sheet.save(os.path.join(TEX, "newchar_nearest.png"))


def newchar(S=3000):
    """The new character, as ink: a wet, loaded brush, dry at the ends of strokes, gathering dark at the edges, feathering out into the paper."""
    kind = "kaih"
    sz = 2200
    c = np.zeros((S, S), np.float32)
    shi = cut("神", kind, (.03, .05, .40, .97), sz)              # 礻
    head = cut("鬼", kind, (.17, .02, .83, .50), sz)              # 甶, with its top stroke
    ren = glyph_mask("人", kind, sz, 0)
    kou = cut("器", kind, (.08, .60, .92, .98), sz)                # the two 口 at the foot of 器
    paste(c, shi, S * .135, S * .10, S * .255, S * .80)
    paste(c, head, S * .425, S * .085, S * .435, S * .315)
    paste(c, ren, S * .445, S * .365, S * .395, S * .265)
    paste(c, kou, S * .405, S * .625, S * .475, S * .285)
    np.save(os.path.join(TEX, "newchar_mask.npy"), c.astype(np.float16))

    # the way the strokes run: the structure tensor of the shape gives each point a direction to drag the dry brush
    g = gaussian_filter(c, 14)
    gy, gx = np.gradient(g)
    jxx, jyy, jxy = gaussian_filter(gx * gx, 46), gaussian_filter(gy * gy, 46), gaussian_filter(gx * gy, 46)
    ang = .5 * np.arctan2(2 * jxy, jxx - jyy) + math.pi / 2
    ux, uy = np.cos(ang), np.sin(ang)
    # hair marks: noise smeared along the stroke
    n0 = noise(S, S, .2, 71)
    Y, X = np.mgrid[:S, :S].astype(np.float32)
    streak = n0.copy()
    px, py, qx, qy = X.copy(), Y.copy(), X.copy(), Y.copy()
    for i in range(34):
        px, py = px + map_coordinates(ux, [py, px], order=1, mode="nearest") * 4.0, py + map_coordinates(uy, [py, px], order=1, mode="nearest") * 4.0
        qx, qy = qx - map_coordinates(ux, [qy, qx], order=1, mode="nearest") * 4.0, qy - map_coordinates(uy, [qy, qx], order=1, mode="nearest") * 4.0
        streak += map_coordinates(n0, [py, px], order=1, mode="wrap") + map_coordinates(n0, [qy, qx], order=1, mode="wrap")
    streak /= streak.std()
    # the outline wanders, and frays
    wob1, wob2 = noise(S, S, 2.5, 72), noise(S, S, 2.5, 73)
    fine1, fine2 = noise(S, S, 1.5, 74), noise(S, S, 1.5, 75)
    m = map_coordinates(c, [Y + 7 * wob1 + 2.0 * fine1, X + 7 * wob2 + 2.0 * fine2], order=1, mode="constant")
    edt = distance_transform_edt(m > .5).astype(np.float32)
    out = distance_transform_edt(m <= .5).astype(np.float32)
    # inside: nearly black, pooled darker at the edge. Only here and there has the brush run dry, and there the hairs show as flying white
    dry_at = noise(S, S, 3.6, 76)
    gate = sm_((dry_at - .15) / .9)
    dry = np.clip((streak - .55) * 1.6, 0, 1) * gate * (1 - np.exp(-edt / 10))
    ink = (1 - dry * .92) * (m > .5)
    pool = np.exp(-edt / 6) * (m > .5)
    core = .94 + .06 * noise(S, S, 2.6, 77)
    D = ink * (2.9 * core + 1.0 * pool)
    # outside: mostly a clean edge. Where the paper was damp the ink has crept out along the fibres, in places, not all round
    damp = sm_((noise(S, S, 3.2, 79) - .35) / .8)
    fib = noise(S, S, 1.0, 78)
    feather = np.exp(-out / (5 + 13 * damp)) * np.clip(fib * .9 + .35, 0, 1) * (.35 + .65 * damp)
    halo = np.exp(-out / (20 + 26 * damp)) * .16 * damp
    D = np.maximum(D, (halo + feather) * (m <= .5))
    # spatter from a loaded brush lifted fast
    r = np.random.default_rng(81)
    sp = Image.new("L", (S, S), 0)
    ds = ImageDraw.Draw(sp)
    ys, xs = np.nonzero(c > .5)
    for _ in range(150):
        i = r.integers(len(xs))
        a, dist = r.uniform(0, 6.28), r.gamma(1.6, 70) + 24
        x, y = xs[i] + math.cos(a) * dist, ys[i] + math.sin(a) * dist
        rr = r.gamma(1.4, 2.4) + .8
        ds.ellipse([x - rr, y - rr, x + rr, y + rr], fill=int(r.uniform(150, 255)))
    D = np.maximum(D, gaussian_filter(np.asarray(sp, np.float32) / 255, .9) * 2.4 * (m <= .5))
    np.save(os.path.join(TEX, "newchar.npy"), D.astype(np.float32))
    Image.fromarray((np.exp(-D) * 255).astype(np.uint8)).resize((1400, 1400), Image.LANCZOS).save(os.path.join(TEX, "newchar_preview.png"))


if __name__ == "__main__":
    os.makedirs(TEX, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "taotie"):
        taotie()
    if what in ("all", "steles"):
        for ch, nm in (("人", "ren"), ("神", "shen"), ("鬼", "gui"), ("器", "qi")):
            stele_glyph(ch, "li", nm)
        stele_column("艰难苦恨繁霜鬓", "li", "dufu")
    if what in ("all", "glyphs"):
        glyphs()
    if what in ("all", "tong"):
        tong()
    if what in ("all", "hands"):
        hands()
    if what in ("all", "blog"):
        blog()
    if what in ("all", "newchar"):
        newchar()
    if what == "nearest":
        nearest()
