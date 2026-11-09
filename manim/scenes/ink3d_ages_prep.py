"""What the river's bank scenes need from scipy and the fonts: a plastron with its burn cracks, bamboo slips, a carved stele, a printed book, a letter, a web page, the words on the water.
Runs in the render environment and leaves .npy files for ink3d_ages.py to load. What is written on all of it, and where that comes from: ink_ages_text.py.

    python ink3d_ages_prep.py bone,slips,stele,book,letter,screen|falls|flown|all   # flown: the falls' river and pool as the moving shot needs them
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d, map_coordinates, zoom

from ink3d_myth_prep import run_ink, sm, spectral
from ink_ages_text import BUCI, HONGLOU, LUNYU, RIVER, SHI, WEB, WHIRL, XIAO, YUQI
from ink_film import OUT, ORACLE, font_file

TEX = os.path.join(OUT, "3d", "ages", "tex")

KAI, LI, XING, SONG, HEI, MONO = ("Kaiti.ttc", "Kaiti TC", "Bold"), ("Libian.ttc", "Libian TC", "Regular"), ("Xingkai.ttc", "Xingkai TC", "Bold"), ("Songti.ttc", "Songti TC", "Regular"), ("Hiragino Sans GB.ttc", "Hiragino Sans GB", "W3"), ("Menlo.ttc", "Menlo", "Regular")
SONG_B, HEI_B, KAI_R = ("Songti.ttc", "Songti TC", "Bold"), ("Hiragino Sans GB.ttc", "Hiragino Sans GB", "W6"), ("Kaiti.ttc", "Kaiti TC", "Regular")
_faces, _index, _blank = {}, {}, {}


def typeface(which, size):
    """A face of a font file by its family and style; its place in the file differs from machine to machine."""
    file, family, style = which
    if which not in _index:
        for i in range(32):
            try:
                if ImageFont.truetype(font_file(file), 20, index=i).getname() == (family, style):
                    _index[which] = i
                    break
            except OSError:
                break
        else:
            raise SystemExit(f"no {family} {style} in {file}")
    k = (which, size)
    if k not in _faces:
        _faces[k] = ImageFont.truetype(font_file(file), size, index=_index[which])
    return _faces[k]


def has(which, ch):
    """Whether a face really has a character, and not just its box for what it has not."""
    f = typeface(which, 64)
    if which not in _blank:
        _blank[which] = bytes(f.getmask("\U000f0000"))
    return bytes(f.getmask(ch)) != _blank[which]


def sign(ch, which, size, pad=2, fallback=KAI_R):
    """A character as a float mask (0-1), centred in a square. A rare one a face has not got comes from one that has."""
    s = size + 2 * pad
    im = Image.new("L", (s, s), 0)
    if not has(which, ch):
        if not has(fallback, ch):
            raise SystemExit(f"nothing has {ch}")
        print("  from", fallback[1], ch, flush=True)
        which = fallback
    ImageDraw.Draw(im).text((s / 2, s / 2), ch, 255, font=typeface(which, size), anchor="mm")
    return np.asarray(im, np.float32) / 255


def press(out, g, x0, y0, gain=1.0):
    """Lays a mask on a sheet at (x0, y0), keeping the darker of the two, and trims it to the sheet."""
    x0, y0 = int(x0), int(y0)
    a, b = max(y0, 0), min(y0 + g.shape[0], out.shape[0])
    c, d = max(x0, 0), min(x0 + g.shape[1], out.shape[1])
    if a < b and c < d:
        out[a:b, c:d] = np.maximum(out[a:b, c:d], g[a - y0:b - y0, c - x0:d - x0] * gain)


def arrival(n, m, seed, top=1.0):
    """When the running ink gets to each place, from 0 to 1: it comes down from the top, each trickle at its own pace. Row 0 is the foot."""
    r = np.random.default_rng(seed)
    pace = gaussian_filter1d(r.normal(size=m), 9)
    late = gaussian_filter1d(r.normal(size=m), 30)
    y = np.arange(n, dtype=np.float32)[:, None] / (n - 1)
    return np.clip((top - y) / top * (1 + .35 * pace[None, :] / pace.std()) + .08 * late[None, :] / late.std(), 0, 1).astype(np.float32)


def legible(trail, writing, margin, clear=.82):
    """A trickle goes dim where it crosses the writing, so that what it has left can still be read."""
    return trail * (1 - clear * np.clip(gaussian_filter((writing > .5).astype(np.float32), margin) * 3, 0, 1))


def save(name, **maps):
    for k, v in maps.items():
        np.save(os.path.join(TEX, f"{name}_{k}.npy"), v.astype(np.float32))
    print(name, next(iter(maps.values())).shape, flush=True)


# The signs of 《合集》12870 in their oracle-bone forms, drawn from what those forms are: points in a unit square, y down, straight cuts because a knife on shell does not curve well. 日 is ink_film's.
_xi = [[(.50, .07), (.17, .36), (.20, .76), (.50, .93), (.80, .76), (.83, .36), (.50, .07)], [(.30, .30), (.72, .80)], [(.70, .30), (.28, .80)]]
CUT = {
    "癸": [[(.15, .15), (.85, .85)], [(.85, .15), (.15, .85)], [(.04, .27), (.27, .04)], [(.73, .04), (.96, .27)], [(.04, .73), (.27, .96)], [(.73, .96), (.96, .73)]],
    "卯": [[(.41, .08), (.41, .78), (.33, .94)], [(.41, .12), (.15, .20), (.12, .52), (.41, .62)], [(.59, .08), (.59, .78), (.67, .94)], [(.59, .12), (.85, .20), (.88, .52), (.59, .62)]],
    "卜": [[(.48, .06), (.48, .94)], [(.48, .44), (.84, .24)]],
    "今": [[(.50, .07), (.09, .52)], [(.50, .07), (.91, .52)], [(.30, .74), (.72, .74)]],
    "日": ORACLE["日"],
    "雨": [[(.08, .10), (.92, .10)], [(.50, .10), (.50, .40)], [(.50, .52), (.50, .68)], [(.50, .78), (.50, .92)],
          [(.20, .24), (.20, .40)], [(.20, .52), (.20, .68)], [(.20, .78), (.20, .92)], [(.80, .24), (.80, .40)], [(.80, .52), (.80, .68)], [(.80, .78), (.80, .92)]],
    "其": [[(.16, .06), (.22, .52), (.50, .90), (.78, .52), (.84, .06)], [(.19, .30), (.81, .30)], [(.21, .14), (.67, .73)], [(.79, .14), (.33, .73)]],
    "自": [[(.50, .05), (.24, .20), (.18, .62), (.30, .92), (.70, .92), (.82, .62), (.76, .20), (.50, .05)], [(.50, .05), (.50, .30)], [(.28, .46), (.72, .46)], [(.28, .70), (.72, .70)]],
    "西": _xi,
    "來": [[(.50, .05), (.50, .95)], [(.50, .05), (.74, .17)], [(.50, .28), (.18, .50)], [(.50, .28), (.82, .50)], [(.50, .66), (.22, .92)], [(.50, .66), (.78, .92)]],
    "東": [[(.50, .04), (.50, .96)], [(.50, .20), (.20, .36), (.20, .64), (.50, .80), (.80, .64), (.80, .36), (.50, .20)], [(.27, .32), (.73, .68)], [(.73, .32), (.27, .68)],
          [(.50, .17), (.28, .04)], [(.50, .17), (.72, .04)], [(.50, .83), (.28, .96)], [(.50, .83), (.72, .96)]],
    "北": [[(.40, .08), (.33, .50), (.40, .92)], [(.33, .44), (.08, .66)], [(.60, .08), (.67, .50), (.60, .92)], [(.67, .44), (.92, .66)]],
    "南": [[(.50, .04), (.50, .24)], [(.50, .13), (.30, .04)], [(.50, .13), (.70, .04)], [(.50, .24), (.18, .46), (.14, .94)], [(.50, .24), (.82, .46), (.86, .94)], [(.14, .94), (.86, .94)],
          [(.30, .52), (.70, .52)], [(.50, .40), (.50, .84)], [(.32, .68), (.68, .68)]],
}


def strokes_mask(strokes, size, width, r=None):
    """Pictograph strokes (points in a unit square, y down) cut as thin lines. With a generator, each is cut by hand, so no two of a sign are alike."""
    S = 4
    im = Image.new("L", (size * S, size * S), 0)
    d = ImageDraw.Draw(im)
    for st in strokes:
        shake = r.normal(0, .014, (len(st), 2)) if r is not None else np.zeros((len(st), 2))
        pts = [((x + a) * size * S, (y + b) * size * S) for (x, y), (a, b) in zip(st, shake)]
        if len(pts) == 1:
            x, y = pts[0]
            d.ellipse((x - width * S, y - width * S, x + width * S, y + width * S), fill=255)
        else:
            d.line(pts, fill=255, width=int(width * S), joint="curve")
    return np.asarray(im.resize((size, size), Image.LANCZOS), np.float32) / 255


def prep_bone():
    """A turtle's under-shell, as it was used: scraped smooth, hollows drilled in its back, cracked by a hot point, and the question cut beside each crack.
    Row 0 is the end that rests on the ground."""
    n, m = 1560, 1120
    cell = .0075                       # 11.7 × 8.4 units
    r = np.random.default_rng(301)
    v, u = np.mgrid[:n, :m].astype(np.float32)
    v, u = v / (n - 1), (u / (m - 1) - .5) * 2
    wob = spectral(n, m, 3.2, 302) * .012 + spectral(n, m, 2.2, 303) * .004
    vs = np.array([0, .03, .08, .16, .28, .40, .46, .56, .62, .72, .84, .93, .985, 1])
    ws = np.array([.20, .34, .50, .62, .72, .92, .96, .96, .90, .72, .62, .48, .30, .20])
    half = np.interp(v, vs, ws) + wob
    mask = np.abs(u) < half
    mask &= ~((v > .955) & (np.abs(u) < .10 + (v - .955) * 2))          # the notch at the tail
    mask &= ~((v < .03) & (np.abs(u) < .05))
    inside = distance_transform_edt(mask).astype(np.float32) * cell
    edge = sm(inside / .34)
    dome = .55 * np.clip(1 - (u / .98) ** 2, 0, 1) ** .55 * np.clip(1 - (2 * v - 1) ** 2 * .7, 0, 1)
    h = .30 * edge + dome * edge

    # the seams between the shell's scutes: a groove down the middle and across
    seam = np.zeros((n, m), np.float32)
    x_mid = .012 * np.sin(v * 21) + .006 * np.sin(v * 53 + 1)
    seam = np.maximum(seam, sm((.008 - np.abs(u - x_mid)) / .005))
    for k, (v0, bow) in enumerate(((.13, .03), (.27, -.03), (.46, -.05), (.64, .05), (.80, .05), (.91, .03))):
        line = v0 + bow * (1 - np.abs(u) ** 1.3) + .004 * np.sin(u * 13 + k)
        seam = np.maximum(seam, sm((.0042 - np.abs(v - line)) / .0026) * (np.abs(u) < half))
    seam = gaussian_filter(seam, 1.0)

    # burnt hollows, each with its 卜 crack, and beside them the question, cut one column at a time from the head of the shell down
    burn = np.zeros((n, m), np.float32)
    crack = np.zeros((n, m), np.float32)
    glyph = np.zeros((n, m), np.float32)
    for cu, cv in ((.445, .70), (-.445, .70), (.30, .40), (.58, .27), (-.62, .30), (.20, .16), (-.22, .17)):
        side = 1 if cu > 0 else -1
        cu, cv = cu + r.normal(0, .008), cv + r.normal(0, .004)
        cx, cy = (cu / 2 + .5) * (m - 1), cv * (n - 1)
        rr = np.hypot((np.arange(m)[None, :] - cx) / 1.0, (np.arange(n)[:, None] - cy) / 1.45)
        burn = np.maximum(burn, sm((24 - rr) / 18))
        # the crack: a long upright, and one branch going toward the midline, broken as heat breaks things
        pts = [(cx, cy - 62)]
        for j in range(1, 7):
            pts.append((pts[-1][0] + r.normal(0, 2.2), pts[-1][1] + 21))
        im = Image.new("L", (m, n), 0)
        d = ImageDraw.Draw(im)
        d.line(pts, fill=255, width=4)
        bx, by = pts[3]
        br = [(bx, by)]
        for j in range(5):
            br.append((br[-1][0] - side * r.uniform(11, 18), br[-1][1] + r.normal(4, 4)))
        d.line(br, fill=255, width=3)
        for a in pts[1:-1:2]:
            d.line([a, (a[0] + r.normal(0, 12), a[1] + r.normal(0, 14))], fill=255, width=1)
        crack = np.maximum(crack, np.asarray(im, np.float32) / 255)
    size, pitch = 118, 122
    for (u0, v0), words in zip(((.27, .945), (.60, .72), (-.27, .945), (-.60, .72), (-.27, .585)), BUCI):
        for q, ch in enumerate(words):
            g = strokes_mask(CUT[ch], size, 5.5, r)[::-1]
            press(glyph, g, (u0 / 2 + .5) * (m - 1) - size / 2 + r.normal(0, 2), v0 * (n - 1) - q * pitch - size / 2)
    crack = gaussian_filter(crack, .7)
    glyph = (gaussian_filter(glyph, .6) > .4).astype(np.float32) * mask

    trail = legible(run_ink(glyph, cell, 305, count=56, rock=h), glyph, 7) * mask
    h = h - .018 * seam * edge - .05 * sm(burn) * edge - .04 * crack - .035 * glyph - .006 * trail
    h = h + .003 * gaussian_filter(spectral(n, m, 1.6, 304), 1.2) * edge
    tone = .80 + .05 * spectral(n, m, 3.3, 306) + .018 * spectral(n, m, 1.6, 307)
    stain = np.clip(.22 * spectral(n, m, 3.6, 308), -.1, .2) * (1 - sm(inside / 2.2))
    alb = tone - stain - .30 * seam
    alb = alb * (1 - .55 * sm(burn) ** 1.4) * (1 - .92 * np.clip(crack * 1.4, 0, 1))
    alb = alb * (1 - glyph) + .012 * glyph
    save("bone", h=h, a=np.clip(alb, .008, .95), r=.5 * (1 - glyph) + .14 * glyph, m=gaussian_filter(mask.astype(np.float32), 1.0), w=np.clip(trail * 1.4, 0, 1), t=arrival(n, m, 309))
    # the ink leaves the shell where its trails reach row 0: the river scene picks up from there


def prep_slips():
    """A roll of bamboo slips laid open, tied with two cords, right to left and top to bottom, a clerk's hand. Row 0 is the foot. The lower characters are running."""
    k_, count_, slip, gap, L = 180, 18, .62, .06, 11.5
    n, m = int(L * k_), int(count_ * (slip + gap) * k_)
    r = np.random.default_rng(321)
    alpha, tone, ink = np.zeros((n, m), np.float32), np.zeros((n, m), np.float32), np.zeros((n, m), np.float32)
    per = 24
    cords = (.24, .76)
    size = int(.42 * k_)
    pitch = (L - 1.0 - 2 * .24) / per * k_
    fibre = gaussian_filter1d(r.normal(size=m), 1.2)
    fibre = (fibre / fibre.std())[None, :]
    k = 0
    for i in range(count_):
        x1 = int(m - i * (slip + gap) * k_ - gap * k_ / 2)
        x0 = x1 - int(slip * k_)
        foot, head = int(abs(r.normal(0, 7))), n - int(abs(r.normal(0, 7)))
        alpha[foot:head, x0:x1] = 1
        tone[foot:head, x0:x1] = r.uniform(.56, .72) * (1 + .05 * spectral(head - foot, x1 - x0, 3.0, 330 + i))
        y = head - .5 * k_
        for q in range(per):
            # the cords go round the slip: the scribe leaves room for them
            for c in cords:
                if abs(y - pitch / 2 - c * n) < .13 * k_:
                    y -= .24 * k_
            if k < len(SHI):
                g = sign(SHI[k], LI, size)[::-1]
                press(ink, g, (x0 + x1) / 2 - g.shape[1] / 2 + r.normal(0, 2), y - pitch / 2 - g.shape[0] / 2 + r.normal(0, 1.5), r.uniform(.8, 1))
                k += 1
            y -= pitch
    print("slips", k, "of", len(SHI), flush=True)
    ink = (gaussian_filter(ink, .7) > .42).astype(np.float32) * alpha
    cord = np.zeros((n, m), np.float32)
    for c in cords:
        row = c * n + 7 * np.sin(np.arange(m) / 240 + c * 9)
        dy = np.abs(np.arange(n, dtype=np.float32)[:, None] - row[None, :])
        cord = np.maximum(cord, sm((10 - dy) / 3) * (.7 + .3 * np.sin(np.arange(m, dtype=np.float32) * .55 + dy * .5)))
    alb = (tone * (1 + .035 * fibre)) * (1 - ink) + .014 * ink
    alb = alb * (1 - cord) + .13 * cord
    y = np.arange(n, dtype=np.float32)[:, None] / n
    trail = legible(run_ink(ink * (1 - sm((y - .62 + .05 * spectral(n, m, 3.0, 323)) / .12)), 1.0, 324, count=150), ink, 5) * alpha
    save("slips", a=np.clip(alb, .008, .95), r=.55 * (1 - ink) + .16 * ink, m=np.maximum(gaussian_filter(alpha, .8), cord), w=np.clip(trail * 1.4, 0, 1), t=arrival(n, m, 325))


def stele_cuts(text, n, m, title, seed):
    """A stele's writing as a mask, twice the stone's fineness: tiers of ten-character columns, right to left, tier under tier as the 开成石经 has them, with a title in the head. Row 0 is the foot."""
    S = 2
    N, M = n * S, m * S
    out = np.zeros((N, M), np.float32)
    tiers, per, cols = 4, 10, 20
    side, foot, head, gap = 90 * S, 170 * S, int(.17 * N), 40 * S
    band = (N - foot - head - gap * (tiers - 1)) / tiers
    pitch_y, pitch_x = band / per, (M - 2 * side) / cols
    size = int(min(pitch_x, pitch_y) * .9)
    r = np.random.default_rng(seed)
    flat = []
    for piece in text:                  # a heading has its own column
        if len(piece) < 12:
            flat += list(piece) + [" "] * (per - len(piece))
        else:
            flat += list(piece) + [" "] * (-len(piece) % per)
    k = 0
    for t in range(tiers):
        top = N - head - t * (band + gap)
        for c in range(cols):
            cx = M - side - (c + .5) * pitch_x
            for q in range(per):
                ch = flat[k] if k < len(flat) else " "
                k += 1
                if ch != " ":
                    g = sign(ch, KAI, size)[::-1]
                    press(out, g, cx - g.shape[1] / 2 + r.normal(0, .6), top - (q + .5) * pitch_y - g.shape[0] / 2 + r.normal(0, .6))
        if t:                           # the line ruled between the tiers
            rule = int(top + gap / 2)
            out[rule - 2:rule + 2, side:M - side] = np.maximum(out[rule - 2:rule + 2, side:M - side], .8)
    print("stele holds", k, "of", len(flat), flush=True)
    big = int(.085 * N)
    for i, ch in enumerate(title):
        g = sign(ch, KAI, big)[::-1]
        press(out, g, M / 2 - (len(title) / 2 - i) * big * 1.05 + (big * 1.05 - g.shape[1]) / 2, N - head * .56 - g.shape[0] / 2)
    return (out > .5).astype(np.float32)


def prep_stele():
    """Steles' faces. Row 0 is the foot. The near one's writing is cut and empty at the top; lower down the cuts are full of ink, and the ink has begun to run. stele_far: its fellows, with other words."""
    n, m = 2600, 1200
    cell = .0115                       # 29.9 × 13.8 units
    z, x = np.mgrid[:n, :m].astype(np.float32) * cell
    rock = .05 * spectral(n, m, 3.4, 401) + .006 * spectral(n, m, 2.0, 402) + .0016 * spectral(n, m, 1.0, 403)
    fine = stele_cuts(LUNYU, n, m, "論語", 411)
    mask = gaussian_filter(fine, 1)[::2, ::2]
    inside = distance_transform_edt(mask > .5).astype(np.float32)
    cut = gaussian_filter(sm(inside / 2.4) * .05, .6)
    # how far the ink has come up the stone
    level = sm(((z / z.max()) * -1 + .62 + .05 * spectral(n, m, 3.0, 404)) / .16)
    trail = legible(run_ink(mask * level, cell, 405, count=170, rock=rock), mask, 2.5)
    soak = gaussian_filter(mask * level, 3) * .35 + gaussian_filter(trail, 3) * .2
    border = sm((np.minimum.reduce([x, x.max() - x, z.max() - z]) - .22) / .1) * .05            # the margin is left a little proud
    h = rock + border - cut - .004 * trail
    alb = (.66 + .05 * spectral(n, m, 3.0, 406) + .025 * spectral(n, m, 1.4, 407)) * (1 - np.clip(soak, 0, .7))
    # on twice the stone's fineness, so that a stroke is a stroke up close: a cut holds the dirt, and lower down it holds the ink
    up = lambda a: zoom(a, 2, order=1)
    alb2, level2, fine_s = up(alb), up(level), gaussian_filter(fine, .7)
    alb2 = alb2 * (1 - .68 * fine_s)
    wet2 = fine_s * level2
    alb2 = alb2 * (1 - wet2) + .012 * wet2
    save("stele", h=h, a=np.clip(alb2, .008, .95), r=np.clip(.9 * (1 - wet2) + .14 * wet2, .1, 1), w=up(np.clip(trail * 1.4, 0, 1)), t=up(arrival(n, m, 408, .72)))
    # steles further off: not cut, only the look of it; the eye is not that close
    far = gaussian_filter(stele_cuts(XIAO, n, m, "孝經", 412), 1)[::2, ::2]
    np.save(os.path.join(TEX, "stele_far.npy"), np.clip(alb * (1 - .75 * far), .008, .95).astype(np.float32))


def wrap(text, font, width):
    """Lines of Chinese no wider than width; a stop or a closing quote does not begin a line."""
    lines, line = [], ""
    for ch in text:
        if font.getlength(line + ch) > width and ch not in "，。；：、”）》":
            lines.append(line)
            line = ""
        line += ch
    return lines + [line]


def prep_screen():
    """What is on the laptop: a browser window, as anyone's looks, open at an encyclopedia's entry; the writing is dark on the lit page, and its lower lines are letting go. Row 0 is the bottom of the glass."""
    n, m = 2160, 3440
    im = Image.new("L", (m, n), 255)
    d = ImageDraw.Draw(im)
    # the window's own bar, two tabs, and the box you type in
    d.rectangle((0, 0, m, 120), fill=206)
    for x in (58, 112, 166):
        d.ellipse((x - 18, 42, x + 18, 78), fill=118)
    d.rounded_rectangle((240, 28, 900, 150), 24, fill=255)
    d.text((300, 88), WEB["tab"], 20, font=typeface(HEI, 50), anchor="lm")
    d.line((838, 72, 866, 100), fill=90, width=5)
    d.line((838, 100, 866, 72), fill=90, width=5)
    d.rounded_rectangle((920, 34, 1500, 112), 20, fill=222)
    d.text((980, 74), WEB["tab2"], 70, font=typeface(HEI, 46), anchor="lm")
    d.text((1560, 72), "+", 80, font=typeface(HEI, 64), anchor="lm")
    d.rectangle((0, 120, m, 262), fill=246)
    for x, f in ((60, 1), (176, -1)):
        d.line((x, 191, x + 44 * f, 191), fill=80, width=7)
        d.line((x, 191, x + 20 * f, 171), fill=80, width=7)
        d.line((x, 191, x + 20 * f, 211), fill=80, width=7)
    d.arc((226, 164, 280, 218), 40, 320, fill=80, width=7)
    d.polygon(((282, 160), (288, 192), (258, 180)), fill=80)
    d.rounded_rectangle((320, 144, m - 200, 238), 47, fill=228)
    d.ellipse((368, 170, 404, 206), outline=70, width=6)
    d.line((400, 202, 420, 222), fill=70, width=6)
    d.text((452, 191), WEB["query"], 25, font=typeface(HEI, 56), anchor="lm")
    d.line((0, 262, m, 262), fill=180, width=3)
    # the page: a column of links, and the entry
    y = 350
    for s_ in WEB["nav"]:
        f = typeface(HEI, 60)
        d.text((86, y), s_, 62, font=f)
        d.line((86, y + 78, 86 + f.getlength(s_), y + 78), fill=120, width=3)
        y += 128
    d.line((610, 300, 610, n - 70), fill=214, width=3)
    d.text((690, 300), WEB["title"], 0, font=typeface(SONG_B, 190))
    d.text((690 + 2 * 190 + 60, 412), WEB["sub"], 96, font=typeface(HEI, 66))
    d.line((690, 556, m - 130, 556), fill=170, width=4)
    y, body = 590, typeface(HEI, 112)
    for para in WEB["body"]:
        for line in wrap(para, body, m - 130 - 690):
            d.text((690, y), line, 0, font=body)
            y += 158
        y += 26
    f = typeface(HEI, 74)
    d.text((690, y + 10), WEB["more"], 70, font=f)
    print("screen: the page ends at row", y + 100, "of", n, flush=True)
    page = np.asarray(im, np.float32)[::-1] / 255
    yy = np.arange(n, dtype=np.float32)[:, None]
    # the writing lets go from the bottom up: from a little below the title, and more of it the lower it is
    text = gaussian_filter((page < .45).astype(np.float32), 1)[::2, ::2]
    small_y = yy[::2]
    letting = text * sm((1000 - small_y * 2 + 120 * spectral(n // 2, m // 2, 3.0, 501)) / 420)
    trail = zoom(legible(run_ink(letting, 1.0, 503, count=170), text, 3), 2, order=1)[:n, :m]
    shade = 1 - .12 * sm((np.hypot(np.arange(m, dtype=np.float32)[None, :] - m / 2, (yy - n / 2) * 1.4) - 1000) / 1400)
    save("screen", e=page * shade, w=np.clip(trail * 1.5, 0, 1), t=arrival(n, m, 505, .62))


def prep_book():
    """An open book, set in movable type: two half-leaves, ten columns each with their rules and a double frame, the type a little uneven and unevenly inked. Row 0 is the foot. The lowest characters are running."""
    k_, Wd, Ht = 180, 16.0, 11.5
    n, m = int(Ht * k_), int(Wd * k_)
    r = np.random.default_rng(341)
    ink = np.zeros((n, m), np.float32)
    im = Image.new("L", (m, n), 0)
    d = ImageDraw.Draw(im)
    cols, per = 10, 24
    top, bottom = int(.72 * k_), n - int(.62 * k_)
    pages = ((m // 2 + int(.55 * k_), m - int(.34 * k_)), (int(.34 * k_), m // 2 - int(.55 * k_)))
    pitch_y = (bottom - top) / per
    size = int(pitch_y * .9)
    flat = []
    for piece in HONGLOU:
        flat += list(piece) + ([] if len(piece) > per else [" "] * (per - len(piece)))
    k = 0
    for x0, x1 in pages:
        d.rectangle((x0, top - 16, x1, bottom + 16), outline=255, width=7)
        d.rectangle((x0 + 12, top - 4, x1 - 12, bottom + 4), outline=230, width=2)
        pitch_x = (x1 - x0 - 24) / cols
        for c in range(1, cols):
            x = x0 + 12 + c * pitch_x
            d.line((x, top - 4, x, bottom + 4), fill=150 + int(r.uniform(0, 50)), width=2)
    frame = np.asarray(im, np.float32)[::-1] / 255
    for x0, x1 in pages:
        pitch_x = (x1 - x0 - 24) / cols
        for c in range(cols):
            cx = x1 - 12 - (c + .5) * pitch_x
            for q in range(per):
                ch = flat[k] if k < len(flat) else " "
                k += 1
                if ch not in " 　":
                    g = sign(ch, SONG, size)
                    if r.random() < .2:
                        g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).rotate(r.normal(0, .9), resample=Image.BICUBIC), np.float32) / 255
                    press(ink, g[::-1], cx - g.shape[1] / 2 + r.normal(0, 1.8), n - top - (q + .5) * pitch_y - g.shape[0] / 2 + r.normal(0, 1.4), r.uniform(.68, 1.0))
    print("book", min(k, len(flat)), "of", len(flat), flush=True)
    speck = spectral(n, m, 1.1, 343)
    ink = np.maximum(gaussian_filter(ink, .7) * (.9 + .1 * np.clip(speck, -1, 1)), frame * .9)
    gutter = 1 - .5 * np.exp(-((np.arange(m, dtype=np.float32) - m / 2) / (.5 * k_)) ** 2)[None, :]
    paper = (.8 + .03 * spectral(n, m, 3.2, 344) + .012 * spectral(n, m, 1.5, 345) - np.clip(.1 * spectral(n, m, 3.8, 346) - .06, 0, .1)) * gutter
    y = np.arange(n, dtype=np.float32)[:, None] / n
    trail = legible(run_ink((ink > .5) * (1 - sm((y - .46 + .05 * spectral(n, m, 3.0, 347)) / .14)), 1.0, 348, count=180), ink, 5)
    save("book", a=np.clip(paper * (1 - ink) + .014 * ink, .008, .95), r=.75 * (1 - ink) + .3 * ink, w=np.clip(trail * 1.4, 0, 1), t=arrival(n, m, 349, .5))


def prep_letter():
    """A letter on eight-column paper, folded once each way before it was opened again; a running hand down the columns, right to left, and the ink running out of its lower characters. Row 0 is the foot."""
    k_, Wd, Ht = 200, 8.0, 11.0
    n, m = int(Ht * k_), int(Wd * k_)
    r = np.random.default_rng(361)
    ink = np.zeros((n, m), np.float32)
    side, top, foot = int(.42 * k_), int(.55 * k_), int(.42 * k_)
    pitch_x = (m - 2 * side) / len(YUQI)
    size = int(.70 * k_)
    pitch_y = (n - top - foot) / max(map(len, YUQI))
    for c, line in enumerate(YUQI):
        cx = m - side - (c + .5) * pitch_x
        for q, ch in enumerate(line):
            g = sign(ch, XING, size, 8)
            g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).rotate(r.normal(0, 2.4), resample=Image.BICUBIC), np.float32) / 255
            press(ink, g[::-1], cx - g.shape[1] / 2 + r.normal(0, 5), n - top - (q + .5) * pitch_y - g.shape[0] / 2 + r.normal(0, 3), r.uniform(.82, 1.0))
    ink = gaussian_filter(ink, .9)
    ink = np.clip(ink * (.86 + .3 * spectral(n, m, 2.2, 362)), 0, 1)
    v, u = np.mgrid[:n, :m].astype(np.float32)
    ruled = np.zeros((n, m), np.float32)
    for c in range(len(YUQI) + 1):
        x = int(side + c * pitch_x)
        ruled[foot - 12:n - top + 12, x:x + (5 if c in (0, len(YUQI)) else 3)] = .3
    ruled[foot - 12:foot - 7, side:m - side] = ruled[n - top + 7:n - top + 12, side:m - side] = .3
    fold = np.exp(-((v - n / 2) / 5) ** 2) + np.exp(-((u - m / 2) / 5) ** 2)
    paper = (.84 + .02 * spectral(n, m, 3.0, 363)) * (1 - .1 * np.clip(fold, 0, 1)) * (1 - ruled)
    low = (ink > .5) * (1 - sm((v / n - .5 + .04 * spectral(n, m, 3.0, 364)) / .12))
    trail = legible(run_ink(low, 1.0, 365, count=100), ink, 6)
    save("letter", a=np.clip(paper * (1 - ink) + .014 * ink, .008, .95), r=.8 * (1 - ink) + .22 * ink, w=np.clip(trail * 1.4, 0, 1), t=arrival(n, m, 366, .55))


POOL_C = (-14.0, -38.0)
LIP_C, LIP_HALF = .012, 100.0
POOL_F_R, POOL_F_IN = 220.0, 60.0


def whirl_layout():
    """Where the words on the whirl are, in world units about the hole, so that the pool's two textures agree. They go on in order, the first arm then the second, so an arm can be read."""
    r = np.random.default_rng(601)
    out = []
    i = 0
    for arm in (0, 1):
        th, rad = arm * math.pi, 170.0
        which = KAI
        while rad > 7:
            size = rad * .095 + 2.2
            if i % 24 == 0:
                which = (KAI, SONG, XING)[r.integers(3)]        # a hand keeps on for a while
            out.append((WHIRL[i % len(WHIRL)], which, r.normal(0, 7), r.uniform(.6, 1.0), th, rad, size))
            i += 1
            step = size * 1.25 / rad
            th += step
            rad *= (1 - .068 * step)
    return out


def whirl(n, R, name):
    """What the pool shows of its own over ±R round the hole, as light: foam drawn out along the whirl, boil at the foot of the falls, the words going round and closing on the hole."""
    k_ = n / (2 * R)
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    X, Y = (xx - n / 2) / k_, (yy - n / 2) / k_
    rad_, th_ = np.hypot(X, Y), np.arctan2(Y, X)
    r = np.random.default_rng(601)
    table = np.zeros(8192, np.float32)
    q = np.arange(8192) / 8192
    for f in range(3, 90):
        table += np.cos(2 * math.pi * (f * q + r.random())) * f ** -.72
    table = (table - table.mean()) / table.std()
    phase = th_ / (2 * math.pi) + .42 * np.log(rad_ + 6) + .018 * spectral(n, n, 3.2, 602)
    streak = table[((phase % 1) * 8191).astype(np.int32)]
    brk = np.clip(.5 + .75 * spectral(n, n, 3.0, 603), 0, 1)
    fade = sm((rad_ - 8) / 26) * (1 - .55 * sm((rad_ - 110) / 90))
    foam = np.clip(streak - .55, 0, 2) * brk * fade
    # under the falls, all the way round the horseshoe
    wx, wy = X + POOL_C[0], Y + POOL_C[1]
    inside = (58 - LIP_C * wx ** 2 - wy) / np.sqrt(1 + (2 * LIP_C * wx) ** 2)
    boil = sm((42 - inside) / 30) * sm((LIP_HALF + 30 - np.abs(wx)) / 30) * np.clip(.42 + .8 * spectral(n, n, 2.5, 604), 0, 1)
    out = np.zeros((n, n), np.float32)
    for ch, which, turn, gain, th, rad, size in whirl_layout():
        px = int(size * k_)
        cx, cy = n / 2 + rad * k_ * math.cos(th), n / 2 + rad * k_ * math.sin(th)
        if not (px < cx < n - px and px < cy < n - px):
            continue
        g = sign(ch, which, px)
        im = Image.fromarray((g * 255).astype(np.uint8)).rotate(math.degrees(th) + 90 + turn, expand=True, resample=Image.BICUBIC)
        g = np.asarray(im, np.float32) / 255
        x0, y0 = int(cx - g.shape[1] / 2), int(cy - g.shape[0] / 2)
        if 0 <= x0 and 0 <= y0 and x0 + g.shape[1] < n and y0 + g.shape[0] < n:
            out[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = np.maximum(out[y0:y0 + g.shape[0], x0:x0 + g.shape[1]], g[::-1] * gain)
    # the whirl drags them: a blur along the circle, done in polar form
    na, nr_ = 5760, 2800
    r0 = 3 * k_
    dl = math.log(n * .5 * 1.42 / r0) / (nr_ - 1)
    ag, lg = np.mgrid[:na, :nr_].astype(np.float32)
    rg, tg = r0 * np.exp(lg * dl), ag / na * 2 * math.pi
    polar = map_coordinates(out, [n / 2 + rg * np.sin(tg), n / 2 + rg * np.cos(tg)], order=1)
    polar = gaussian_filter(polar, (35.0, 0), mode=("wrap", "nearest"))
    a_i = (th_ % (2 * math.pi)) / (2 * math.pi) * na
    l_i = np.log(np.maximum(rad_ * k_, r0) / r0) / dl
    a_i = np.where(a_i >= na - 1, a_i - na, a_i)
    smear = map_coordinates(polar, [a_i, l_i], order=1, mode="nearest")
    words = .6 * gaussian_filter(out, .8 * k_ / 9.5) + .7 * smear
    light = .035 + .32 * foam + .5 * boil + .85 * words
    light = light * sm((rad_ - 7) / 22)                            # the hole has nothing to show
    np.save(os.path.join(TEX, name), np.clip(light, 0, 1.2).astype(np.float32))
    print(name, n, k_)


def river_face(line):
    """What each line is set in."""
    if any("\u3040" <= c <= "\u30ff" for c in line):
        return ("Hiragino Sans GB.ttc", "Hiragino Sans GB", "W3") if all(has(HEI, c) for c in line) else ("Songti.ttc", "Songti SC", "Regular")
    if line.startswith("hello"):
        return MONO
    if line.isascii() or any("\u0370" <= c <= "\u03ff" for c in line):
        return ("Times New Roman.ttf", "Times New Roman", "Regular")
    return KAI


def prep_river_flown():
    """The river above the lip, 300 units of it at 16 pixels a unit along it, and as wide as the shot's mesh is there. The lens comes down it just above the water, so the top of a word is toward the lip
    and it reads from upstream, left to right as the lens sees it. A line to a row, across the water: the row the lens will fly over has its line under the lens, and others either side. Row 0 is at the lip."""
    n, m = 4800, 2624
    k_ = 16
    r = np.random.default_rng(611)
    out = np.zeros((n, m), np.float32)

    def half(b):
        return max(100 - .28 * max(b, 0), 60) + 16          # as ink3d_ages.shot_falls's half_width, and its 16 over

    def course(b):
        return 46 * math.sin(b / 130) * float(sm(np.array(b / 90)))

    lens = lambda b: float(np.interp(b + 58, (50, 80, 112, 149), (0, 4, 11, 20)))
    used = 0
    gap, first = 13.0, 24.0
    rows = int((300 - first) / gap)
    for rw in range(rows):
        b = first + rw * gap + r.normal(0, 1.2)
        near = max(0, 5 - rw)                               # the fifth row up is the nearest at the first frame: counted from it, the list's first lines
        queue = list(RIVER)
        pick = (near if rw <= 5 else used + rw) % len(queue)
        kx = m / (2 * half(b))
        w_u = r.uniform(5.2, 6.4)
        stretch = r.uniform(1.8, 2.1)
        # the line under the lens, and further lines either side of it until the row is full
        cx = lens(b)
        left, right = cx, cx
        seq = 0
        while True:
            line = queue[(pick + seq) % len(queue)]
            seq += 1
            which = river_face(line)
            size = 96
            f = typeface(which, size)
            box = f.getlength(line)
            im = Image.new("L", (int(box) + 16, size + 40), 0)
            ImageDraw.Draw(im).text((8, 18), line, 255, font=f)
            wide_u = box / size * w_u
            hh = int(w_u * stretch * k_)
            ww = max(int(wide_u * kx), 8)
            g = np.asarray(im.rotate(r.normal(0, 1.2), resample=Image.BICUBIC).resize((ww, hh), Image.BICUBIC), np.float32) / 255
            g = g[:, ::-1]                                  # the lens looks along -y: its left is the world's +x
            if seq == 1:
                x_u = cx - wide_u / 2
                left, right = cx - wide_u / 2, cx + wide_u / 2
            elif seq % 2:
                x_u = right + r.uniform(5, 12)
                right = x_u + wide_u
            else:
                left = left - r.uniform(5, 12) - wide_u
                x_u = left
            col = (((x_u - course(b)) / half(b)) + 1) / 2 * m
            press(out, g, col, b * k_ - hh / 2, r.uniform(.72, 1.0))
            if left < course(b) - half(b) and right > course(b) + half(b):
                break
        used += 1
    # long pale lines where the current shows
    s = np.fft.irfft2(np.fft.rfft2(r.normal(size=(n, m))) * np.exp(-(np.fft.fftfreq(n)[:, None] * 620) ** 2) / (np.abs(np.fft.rfftfreq(m)[None, :]) + .003) ** .5, s=(n, m)).astype(np.float32)
    s = (s - s.mean()) / s.std()
    np.save(os.path.join(TEX, "river_f.npy"), np.clip(.05 + .1 * np.clip(s - .7, 0, 2) + .7 * gaussian_filter(out, 1.1), 0, 1.2).astype(np.float32))
    print("river_f", rows, flush=True)


def prep_curtain():
    """Words going over: in the top half of the falls each is pulled long, dark on the white water, and has gone by halfway. Row 0 is the foot."""
    n, m = 1024, 2048
    r = np.random.default_rng(621)
    out = np.zeros((n, m), np.float32)
    k = 0
    for i in range(46):
        ch = WHIRL[k % len(WHIRL)]
        k += 1
        w, hh = int(r.uniform(82, 124)), int(r.uniform(190, 330))
        g = sign(ch, KAI, w)
        g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((w, hh), Image.BICUBIC), np.float32)[::-1] / 255
        x0, y0 = int(r.uniform(.05, .93) * m), int(r.uniform(.5, .98) * n - hh * .5)
        y0 = min(y0, n - hh - 2)
        if 0 <= x0 and x0 + w < m and 0 <= y0:
            out[y0:y0 + hh, x0:x0 + w] = np.maximum(out[y0:y0 + hh, x0:x0 + w], g * r.uniform(.55, 1.0))
    out = gaussian_filter(out, (9, 1.0))
    v = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    np.save(os.path.join(TEX, "fall_w.npy"), np.clip(out * 1.5 * sm((v - .42) / .34), 0, 1).astype(np.float32))
    print("curtain", k)


if __name__ == "__main__":
    os.makedirs(TEX, exist_ok=True)
    what = (sys.argv[1] if len(sys.argv) > 1 else "all").split(",")
    every = "all" in what
    for key, make in (("bone", prep_bone), ("slips", prep_slips), ("stele", prep_stele), ("book", prep_book), ("letter", prep_letter), ("screen", prep_screen), ("falls", prep_curtain)):
        if every or key in what:
            make()
    if every or "flown" in what:
        whirl(4200, POOL_F_R, "pool_f.npy")
        whirl(4096, POOL_F_IN, "pool_fi.npy")
        prep_river_flown()
