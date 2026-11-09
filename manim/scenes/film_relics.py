"""二's carriers that film_docs.py does not make: a stone with Ptolemy's name ring, a papyrus leaf of the Iliad, a turtle plastron with a 我 cut in it, a bundle of bamboo
slips of 報任安書, and a torn leaf of a Song print of 將進酒. With film_docs.py's tablet and quarto and the terminal of film_screens.py they are 二's eight.

    python film_relics.py write stone|greek|bone|slips|song|all

Each is a thing lying on the old table, made as film_docs.py makes its tablet: a height field in mm with Z below 0 wherever the thing is not (that part goes under the
table), and its colour, roughness and fine relief as maps: maps/NAME.npz (X, Y, Z, rows from the foot up), maps/NAME_col.png, _rough.png, _relief.png, and maps/NAME.json,
whose keys.first is the box of the first sign a reader meets, in mm from the thing's top left. 二 cuts on that sign. film_docs3d.py photographs them.

Only what was checked is written. Every other mark is worn past reading: texture, not text.
- stone: Ptolemy's name, p t w l m y s, in its ring (film_press_maps.PTOLEMY: Gardiner Q3 X1 V4 E23 Aa15 M17 M17 S29). The rest of its lines are signs of the script worn
  away; no group of them can be read.
- greek: Iliad 1.1-10 as film_wo_texts gives it, in capitals without accents, run together as the papyri write it, with lunate sigma and round epsilon.
- bone: 我 as it is cut on 合6057 (Sinica 43.EFAB; Wikimedia Commons File:我-oracle.svg, public domain), as knife lines. Nothing else is cut in it: a plastron, its seams, the cracks the fire drew and their scorches.
- slips: 報任安書 from 「人固有一死」, as zh.wikisource gives it (read 2026-11-07), 13 characters to a slip, without punctuation, as slips are written.
- song: 將進酒 as zh.wikisource gives it: the title and the first three lines, cut on the block; the leaf is torn across under the eighth row, so only the head of each column is left.
"""
import json
import os
import sys
import unicodedata

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt, gaussian_filter, grey_dilation
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from film_docs import MAPS, face, impression, save8, srgb, to_srgb
from ink_grade import sm


# ---------------------------------------------------------------- what they share

def field(H, W, px, sigma, seed, aniso=(1.0, 1.0)):
    """Smooth noise of std 1 at H x W pixels of px a mm: a Gaussian of sigma mm (times aniso along y and x), made coarse and blown up, so a wide one costs no more than a fine one."""
    sy, sx = max(sigma * aniso[0] * px, 0.5), max(sigma * aniso[1] * px, 0.5)
    ky, kx = max(1, int(sy / 3)), max(1, int(sx / 3))
    h, w = H // ky + 4, W // kx + 4
    a = gaussian_filter(np.random.default_rng(seed).normal(0, 1, (h, w)).astype(np.float32), (sy / ky, sx / kx), mode="wrap")
    a /= a.std() + 1e-9
    if ky == 1 and kx == 1:
        return a[:H, :W]
    big = np.asarray(Image.fromarray(a).resize((w * kx, h * ky), Image.BICUBIC), np.float32)
    return big[ky:ky + H, kx:kx + W]


def plane(w, h, px):
    H, W = int(round(h * px)), int(round(w * px))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return H, W, (xx + 0.5) / px, (yy + 0.5) / px


def outline(poly, H, W, px, seed, rough):
    """Signed distance in mm to a broken edge, negative inside: a polygon in mm from the top left, its edge broken at the scales in rough, (amount mm, size mm) each."""
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).polygon([(x * px, y * px) for x, y in poly], fill=255)
    m = np.asarray(im) > 127
    sd = (distance_transform_edt(~m) - distance_transform_edt(m)).astype(np.float32) / px
    for k, (amp, size) in enumerate(rough):
        sd += amp * field(H, W, px, size, seed + k)
    return sd


def paste(dst, a, l, t):
    """a's pixels into dst with its top left at pixel (l, t), keeping the larger."""
    h, w = a.shape
    y0, x0, y1, x1 = max(t, 0), max(l, 0), min(t + h, dst.shape[0]), min(l + w, dst.shape[1])
    if y1 > y0 and x1 > x0:
        dst[y0:y1, x0:x1] = np.maximum(dst[y0:y1, x0:x1], a[y0 - t:y1 - t, x0 - l:x1 - l])


def keep_out(H, W, px, boxes, pad):
    """1 over some boxes in mm, softly, 0 elsewhere: what wear must not touch."""
    a = np.zeros((H, W), np.float32)
    for x0, y0, x1, y1 in boxes:
        a[max(int((y0 - pad) * px), 0):int((y1 + pad) * px), max(int((x0 - pad) * px), 0):int((x1 + pad) * px)] = 1
    return np.clip(gaussian_filter(a, pad * px / 3) * 1.5, 0, 1)


def save(name, col, rough, relief, Z, w, h, px, step, keys, chars=(), **extra):
    """The maps and the mesh, as film_docs3d.Sheet reads them."""
    os.makedirs(MAPS, exist_ok=True)
    save8(to_srgb(col), name + "_col.png")
    save8(np.clip(rough, 0, 1), name + "_rough.png")
    save8(np.clip(relief, 0, 1), name + "_relief.png")
    k = max(1, int(round(step * px)))
    Zg = gaussian_filter(Z, 0.4 * k)[::k, ::k][::-1]
    gy, gx = Zg.shape
    X, Y = np.meshgrid((np.arange(gx) * k + 0.5) / px, h - (np.arange(gy)[::-1] * k + 0.5) / px)
    np.savez(os.path.join(MAPS, name + ".npz"), X=np.float32(X), Y=np.float32(Y), Z=np.float32(Zg), solid=True)
    with open(os.path.join(MAPS, name + ".json"), "w", encoding="utf-8") as fp:
        json.dump(dict(w=w, h=h, px=px, keys=keys, chars=list(chars), solid=True, **extra), fp, ensure_ascii=False)
    print(name, "mesh", Zg.shape, "z %.1f..%.1f mm" % (float(Zg.min()), float(Zg.max())), "first", [round(v, 2) for v in keys["first"]], flush=True)


def ink_box(mask, px, x0=0.0, y0=0.0):
    ys, xs = np.nonzero(mask > 0.5)
    return [x0 + xs.min() / px, y0 + ys.min() / px, x0 + (xs.max() + 1) / px, y0 + (ys.max() + 1) / px]


# ---------------------------------------------------------------- the stone: granodiorite, Ptolemy's name in its ring, cut in sunk relief

STONE_W, STONE_H, STONE_PX = 240.0, 172.0, 10
STONE_FACE = 32.0                   # how high its dressed face stands off the table
STONE_EM, STONE_PITCH = 12.0, 19.0
STONE_ROWS = [22.0 + STONE_PITCH * i for i in range(8)]
RING_ROW, RING_X, RING_SIGN, RING_INNER = 60.0, 74.0, 8.6, 11.0      # the ring's line, its left end, its signs' em, and how tall a sign may be inside it
STONE_POLY = [(16, 36), (46, 18), (96, 22), (150, 7), (206, 13), (234, 40), (229, 96), (237, 140), (206, 166), (140, 161), (80, 168), (30, 157), (8, 110), (12, 68)]
# the name, grouped as the scribes group it: p over t, the lasso, the lion, m, the two reeds, the folded cloth
PTOLEMY_GROUPS = ((("Q003",), ("X001",)), (("V004",),), (("E023",),), (("AA015",),), (("M017", "M017"),), (("S029",),))


def hiero(name):
    return unicodedata.lookup("EGYPTIAN HIEROGLYPH " + name)


def sign(ch, f):
    """A sign as a carver cuts it in sunk relief: its outline filled, and its inner lines, which he cuts deeper. Returns (solid, inner), cropped to the sign."""
    size = int(f.size * 2.6)
    im = Image.new("L", (size, size), 0)
    ImageDraw.Draw(im).text((size * .15, size * .75), ch, 255, font=f, anchor="ls")
    line = np.asarray(im) > 90
    if not line.any():
        return None
    ys, xs = np.nonzero(line)
    line = np.pad(line[ys.min():ys.max() + 1, xs.min():xs.max() + 1], 3)
    solid = binary_fill_holes(line)
    inner = line & binary_erosion(solid, iterations=2)
    return solid.astype(np.float32), inner.astype(np.float32)


def group(rows, f, px, gap):
    """Signs stacked in rows, each row side by side, centred: one quadrat. Returns (solid, inner)."""
    built = []
    for row in rows:
        parts = [sign(hiero(n) if len(n) > 1 else n, f) for n in row]
        hh = max(p[0].shape[0] for p in parts)
        ww = sum(p[0].shape[1] for p in parts) + gap * (len(parts) - 1)
        s, n = np.zeros((hh, ww), np.float32), np.zeros((hh, ww), np.float32)
        x = 0
        for a, b in parts:
            t = (hh - a.shape[0]) // 2
            s[t:t + a.shape[0], x:x + a.shape[1]] = a
            n[t:t + a.shape[0], x:x + a.shape[1]] = b
            x += a.shape[1] + gap
        built.append((s, n))
    hh = sum(s.shape[0] for s, _ in built) + gap * (len(built) - 1)
    ww = max(s.shape[1] for s, _ in built)
    S, N = np.zeros((hh, ww), np.float32), np.zeros((hh, ww), np.float32)
    y = 0
    for s, n in built:
        l = (ww - s.shape[1]) // 2
        S[y:y + s.shape[0], l:l + s.shape[1]] = s
        N[y:y + s.shape[0], l:l + s.shape[1]] = n
        y += s.shape[0] + gap
    return S, N


def fit(a, most_h, most_w):
    """Shrinks a pair of masks to fit, never enlarges."""
    s, n = a
    k = min(1.0, most_h / s.shape[0], most_w / s.shape[1])
    if k >= 1.0:
        return s, n
    size = (max(int(s.shape[1] * k), 1), max(int(s.shape[0] * k), 1))
    return [np.asarray(Image.fromarray(np.uint8(m * 255)).resize(size, Image.LANCZOS), np.float32) / 255 for m in (s, n)]


def stone_pool(f, r, n=320):
    """Signs of the script to wear away: (solid, inner) and whether they are flat, narrow or whole."""
    pool = []
    codes = r.choice(np.arange(0x13000, 0x1342F), n, replace=False)
    em = f.size
    for c in codes:
        s = sign(chr(int(c)), f)
        if s is None:
            continue
        h, w = s[0].shape
        if h > 1.15 * em or w > 1.35 * em or h < 0.15 * em:
            continue
        pool.append((s, "flat" if h < 0.45 * em else "narrow" if w < 0.38 * em else "whole"))
    return pool


def write_stone():
    px, w, h = STONE_PX, STONE_W, STONE_H
    H, W, u, v = plane(w, h, px)
    r = np.random.default_rng(601)
    sd = outline(STONE_POLY, H, W, px, 602, ((2.2, 9.0), (0.8, 2.5), (0.25, 0.7)))
    solid, inner = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    # the ring: a groove round the name, a bar across its right end, the name inside it
    f_in = face("NotoSansEgyptianHieroglyphs-Regular.ttf", 0, RING_SIGN * px)
    gap = int(1.0 * px)
    x = RING_X + 4.0
    first = None
    for rows in PTOLEMY_GROUPS:
        g = fit(group(rows, f_in, px, int(0.6 * px)), RING_INNER * px, 14.0 * px)
        l, t = int(x * px), int(RING_ROW * px - g[0].shape[0] / 2)
        paste(solid, g[0], l, t)
        paste(inner, g[1], l, t)
        if first is None:
            first = ink_box(g[0], px, l / px, t / px)
        x += (g[0].shape[1] + gap) / px
    ring_r = x + 3.0
    ring = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(ring)
    half = RING_INNER / 2 + 2.2
    d.rounded_rectangle([RING_X * px, (RING_ROW - half) * px, ring_r * px, (RING_ROW + half) * px], radius=half * px, outline=255, width=int(1.3 * px))
    d.line([((ring_r + 1.4) * px, (RING_ROW - half - 0.5) * px), ((ring_r + 1.4) * px, (RING_ROW + half + 0.5) * px)], fill=255, width=int(1.3 * px))
    solid = np.maximum(solid, np.asarray(ring, np.float32) / 255)
    ring_box = (RING_X - 1, RING_ROW - half - 1, ring_r + 3, RING_ROW + half + 1)
    print("ring %.1f .. %.1f mm" % (RING_X, ring_r + 1.4), flush=True)
    # every other line: signs of the script in quadrats, cut as well, then worn
    f = face("NotoSansEgyptianHieroglyphs-Regular.ttf", 0, STONE_EM * px)
    pool = stone_pool(f, r)
    flat = [p for p in pool if p[1] == "flat"]
    narrow = [p for p in pool if p[1] == "narrow"]
    em = STONE_EM * px
    fill_s, fill_n = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for y in STONE_ROWS:
        spans = [(6.0, w - 4)] if y != RING_ROW else [(6.0, RING_X - 2.5), (ring_r + 4.5, w - 4)]
        for a0, a1 in spans:
            x = a0 + float(r.uniform(0, 3))
            while x < a1:
                s, kind = pool[int(r.integers(len(pool)))]
                if kind == "flat":
                    b = flat[int(r.integers(len(flat)))][0]
                    s = stack(s, b, int(0.8 * px))
                elif kind == "narrow" and r.random() < 0.7:
                    b = narrow[int(r.integers(len(narrow)))][0]
                    s = side(s, b, int(0.8 * px))
                s = fit(s, 1.05 * em, 1.4 * em)
                if x + s[0].shape[1] / px > a1:
                    break
                l, t = int(x * px), int(y * px - s[0].shape[0] / 2 + r.normal(0, 0.3) * px)
                paste(fill_s, s[0], l, t)
                paste(fill_n, s[1], l, t)
                x += s[0].shape[1] / px + float(r.uniform(1.2, 2.6))
    # how much of each place has kept its face: patchy, worse towards the broken edges and the foot; the name and its ring untouched
    guard = keep_out(H, W, px, [ring_box], 3.0)
    edge = sm(np.clip(1 - (-sd) / 22.0, 0, 1))
    keep = sm((0.5 * field(H, W, px, 3.0, 603) + 0.3 * field(H, W, px, 1.1, 604) + 0.25 - 0.9 * edge - 0.3 * np.clip((v - 70) / 90, 0, 1)) / 0.35)
    keep = np.maximum(keep, guard)
    cut_name = impression(solid, px, slope=3.0, most=1.1) + 0.35 * gaussian_filter(inner, 0.04 * px) * (solid > 0.5)
    cut_fill = impression(fill_s, px, slope=3.0, most=1.0) + 0.3 * gaussian_filter(fill_n, 0.04 * px) * (fill_s > 0.5)
    cut_fill = cut_fill * keep * (1 - guard) + gaussian_filter(cut_fill, 0.7 * px) * 0.25 * (1 - keep) * (1 - guard)
    cut = cut_name + cut_fill
    # flakes off the face: a rough floor a millimetre or two down, with a broken edge; never on the ring
    spall = sm((0.7 * field(H, W, px, 5.0, 605) + 0.2 * field(H, W, px, 1.4, 606) - 1.7 + 1.3 * edge) / 0.05) * (1 - guard)
    floor = 1.4 + 0.5 * field(H, W, px, 1.2, 607)
    # the face, dressed flat and polished; round it the fracture, steep, uneven, chipped along its arris
    slope_w = 5.0 + 2.5 * field(H, W, px, 12.0, 608)
    e = np.clip(-sd / np.maximum(slope_w, 2.0), 0, 1)
    Z = STONE_FACE * e ** 0.55 + 0.12 * field(H, W, px, 20.0, 609) + 0.025 * field(H, W, px, 1.5, 610)
    fracture = 1 - sm(np.clip(e * 1.1, 0, 1))
    Z += fracture * (1.4 * field(H, W, px, 2.5, 611) + 0.5 * field(H, W, px, 0.8, 612))
    chips = sm((field(H, W, px, 1.8, 613) - 1.0) / 0.6) * np.exp(-np.maximum(-sd - slope_w, 0) / 2.0) * (1 - fracture) * 1.4
    Z -= chips
    face_part = (1 - fracture)
    Z = Z - (cut * (1 - spall) + floor * spall) * face_part
    Z = np.where(sd < 0, Z, -2.0).astype(np.float32)
    # its colour: crystals, dark mica and hornblende, grey quartz, white and pink feldspar, a pink vein across a corner; what is cut or broken shows paler
    n = int(w * h * 6.0)
    P = r.uniform((0, 0), (w, h), (n, 2))
    uu = u + 0.15 * field(H, W, px, 0.35, 614)
    vv = v + 0.15 * field(H, W, px, 0.35, 615)
    _, idx = cKDTree(P).query(np.stack([uu.ravel(), vv.ravel()], 1), workers=-1)
    kind = r.choice(4, n, p=(.55, .3, .12, .03))
    palette = np.stack([srgb(56, 56, 60), srgb(68, 68, 72), srgb(92, 90, 88), srgb(102, 91, 87)])         # seen from a reading distance granodiorite is a dark grey with a quiet speckle
    tone = palette[kind] * np.exp(0.08 * r.normal(0, 1, (n, 1))).astype(np.float32)
    col = tone[idx].reshape(H, W, 3).astype(np.float32)
    quartz = (kind[idx] == 1).reshape(H, W)
    Q = r.uniform((0, 0), (w, h), (int(w * h * 0.0015), 2))
    dq, iq = cKDTree(Q).query(np.stack([uu.ravel(), vv.ravel()], 1), workers=-1)
    big = (dq < r.uniform(0.7, 1.5, len(Q))[iq]).reshape(H, W)
    col = np.where(big[..., None], srgb(98, 96, 93) * (1 + 0.04 * field(H, W, px, 0.3, 616))[..., None], col)
    vein_d = np.abs((u - 30) * 0.42 - (v - 10) * 0.91 + 3.0 * field(H, W, px, 6.0, 617))
    vein = np.exp(-(vein_d / 7.0) ** 2) * sm((field(H, W, px, 2.0, 618) + 1.2) / 1.0)
    col = col + (srgb(128, 108, 102) - col) * (0.35 * vein)[..., None]
    col = gaussian_filter(col, (0.6, 0.6, 0))
    depth = np.clip(cut * (1 - spall) * face_part, 0, None)
    pale = np.clip(0.55 * sm((depth - 0.15) / 0.3) + 0.4 * spall * face_part + 0.35 * fracture + 0.3 * sm(chips / 0.6), 0, 0.65)       # only a real cut shows pale: a worn sign stays the face's own grey
    col = col + (srgb(138, 135, 130) * (1 + 0.08 * field(H, W, px, 0.4, 619))[..., None] - col) * pale[..., None]
    dust = sm(depth / 0.6) * 0.25
    col = col * (1 + dust[..., None] * 0.3)
    rough = np.clip(0.42 + 0.04 * field(H, W, px, 2.0, 620) + 0.4 * pale, 0, 1)
    relief = np.clip(0.5 + 0.06 * quartz - 0.05 * big + 0.12 * pale * field(H, W, px, 0.25, 621), 0, 1)
    keys = dict(first=first, ring=list(ring_box))
    save("stone", col, rough, relief, Z, w, h, px, 0.15, keys, chars=[dict(c="".join(hiero(c) for g in PTOLEMY_GROUPS for row in g for c in row), col=0, box=list(ring_box))])


def stack(a, b, gap):
    """Two flat signs, one over the other."""
    ww = max(a[0].shape[1], b[0].shape[1])
    hh = a[0].shape[0] + b[0].shape[0] + gap
    S, N = np.zeros((hh, ww), np.float32), np.zeros((hh, ww), np.float32)
    for (s, n), y in ((a, 0), (b, a[0].shape[0] + gap)):
        l = (ww - s.shape[1]) // 2
        S[y:y + s.shape[0], l:l + s.shape[1]] = s
        N[y:y + s.shape[0], l:l + s.shape[1]] = n
    return S, N


def side(a, b, gap):
    """Two narrow signs, side by side."""
    hh = max(a[0].shape[0], b[0].shape[0])
    ww = a[0].shape[1] + b[0].shape[1] + gap
    S, N = np.zeros((hh, ww), np.float32), np.zeros((hh, ww), np.float32)
    for (s, n), x in ((a, 0), (b, a[0].shape[1] + gap)):
        t = (hh - s.shape[0]) // 2
        S[t:t + s.shape[0], x:x + s.shape[1]] = s
        N[t:t + s.shape[0], x:x + s.shape[1]] = n
    return S, N


# ---------------------------------------------------------------- the papyrus: the Iliad's first lines in a book hand, on a leaf broken at three sides

PAP_W, PAP_H, PAP_PX = 150.0, 118.0, 20
GREEK_CAP, GREEK_LEAD = 3.1, 6.1          # a letter's height, line to line
GREEK_LEFT, GREEK_FIRST = 30.0, 24.0      # the column's left, the first line's foot
GREEK_LINES = 10
GREEK_FACE = ("Seravek.ttc", 0)          # a humanist sans with the Greek, the lunate sigma and the round epsilon in it (its lighter weights have no Greek): thinned to the line the reed follows
NIB, NIB_ANGLE = 0.4, 8.0                 # the reed's broad edge, mm, and its angle off the horizontal: thick downstrokes, thin crossbars
PAP_POLY = [(9, 9), (52, 6), (104, 11), (143, 8), (147, 46), (133, 71), (144, 108), (98, 113), (61, 106), (22, 112), (6, 74), (11, 38)]


def uncial(line):
    """A line as a book hand of the Roman period writes it: capitals, no accents or breathings, no spaces or stops, round epsilon and sigma."""
    s = "".join(c for c in unicodedata.normalize("NFD", line) if not unicodedata.combining(c)).upper()
    s = "".join(c for c in s if c.isalpha())
    return s.replace("Σ", "Ϲ").replace("Ε", "Є")


def greek_lines():
    from film_wo_texts import lines
    out = [uncial(t) for t in lines("greek")[:GREEK_LINES]]
    assert out[0] == "ΜΗΝΙΝΑЄΙΔЄΘЄΑΠΗΛΗΙΑΔЄΩΑΧΙΛΗΟϹ", out[0]
    return out


def nib(px):
    """The reed's edge as a footprint: a short line at NIB_ANGLE."""
    n = int(NIB * px) | 1
    a = np.radians(NIB_ANGLE)
    im = Image.new("L", (n, n), 0)
    c = n / 2
    ImageDraw.Draw(im).line([(c - np.cos(a) * n / 2, c + np.sin(a) * n / 2), (c + np.cos(a) * n / 2, c - np.sin(a) * n / 2)], fill=255, width=1)
    return np.asarray(im) > 0


def write_greek():
    px, w, h = PAP_PX, PAP_W, PAP_H
    H, W, u, v = plane(w, h, px)
    r = np.random.default_rng(701)
    sd = outline(PAP_POLY, H, W, px, 702, ((1.6, 6.0), (0.7, 1.6), (0.25, 0.4)))
    # the edges fray along the fibres: threads stand out sideways
    fray = np.maximum(field(H, W, px, 0.08, 703, aniso=(1, 30)), 0) * 0.9
    sd = sd - fray * np.exp(-np.abs(sd) / 1.2)
    # the writing: each letter a little off its place, the reed going dry from one dip to the next
    probe = -face(GREEK_FACE[0], GREEK_FACE[1], 10 * px).getbbox("Η", anchor="ls")[1] / px      # a capital's height at 10 mm, in mm
    f = face(GREEK_FACE[0], GREEK_FACE[1], 10 * GREEK_CAP / probe * px)
    capk = -f.getbbox("Η", anchor="ls")[1]
    xk = -f.getbbox("ω", anchor="ls")[1]
    L = np.zeros((H, W), np.float32)
    level = np.zeros((H, W), np.float32)
    text = greek_lines()
    first = None
    left_after = 0
    for i, t in enumerate(text):
        x, base0 = GREEK_LEFT + float(r.normal(0, 0.25)), GREEK_FIRST + i * GREEK_LEAD
        drift = float(r.uniform(0, 6.3))
        for j, c in enumerate(t):
            if left_after <= 0:
                left_after = int(r.integers(9, 15))         # letters to the next dip
                wet = 1.0
            wet = max(wet - 0.035, 0.6)
            left_after -= 1
            ch = "ω" if c == "Ω" else c
            box = f.getbbox(ch, anchor="ls")
            size = (box[2] - box[0] + 8, box[3] - box[1] + 8)
            im = Image.new("L", size, 0)
            ImageDraw.Draw(im).text((4 - box[0], 4 - box[1]), ch, 255, font=f, anchor="ls")
            a = np.asarray(im, np.float32) / 255
            top = -box[1] + 4
            if c == "Ω":         # the book hand's omega is as tall as the rest
                k = capk / xk
                a = np.asarray(Image.fromarray(np.uint8(a * 255)).resize((int(a.shape[1] * 1.08), int(a.shape[0] * k)), Image.LANCZOS), np.float32) / 255
                top = int(top * k)
            l = int(x * px + r.normal(0, 0.04) * px)
            base = base0 + 0.22 * np.sin((x - GREEK_LEFT) / 26.0 + drift)        # the line wanders as a hand ruling nothing wanders
            tt = int(base * px - top + r.normal(0, 0.05) * px)
            paste(L, gaussian_filter(binary_erosion(a > 0.5, iterations=2).astype(np.float32), 0.5), l, tt)        # the letter's spine, 0.1 mm in from each side: the reed's edge makes the weight
            lv = np.where(a > 0.3, wet * float(r.uniform(0.9, 1.0)), 0).astype(np.float32)
            paste(level, lv, l, tt)
            if first is None:
                first = ink_box(a, px, l / px, tt / px)
            x += (box[2] - box[0]) / px + 0.62 + float(r.normal(0, 0.06))
    pen = grey_dilation(L, footprint=nib(px))
    level = grey_dilation(level, footprint=nib(px))
    print("greek", len(text), "lines, the first ends at %.1f mm" % x, flush=True)
    # papyrus: a layer of strips with the fibres across, laid on a layer with them down; where the top layer has gone the lower one shows, and with it the writing goes
    top_f = 0.45 * field(H, W, px, 0.07, 704, aniso=(1, 30)) + 0.5 * field(H, W, px, 0.3, 705, aniso=(1, 12)) + 0.4 * field(H, W, px, 2.0, 706, aniso=(1, 6))      # short uneven fibres, not a brushed sheet
    low_f = 0.6 * field(H, W, px, 0.06, 707, aniso=(90, 1)) + 0.45 * field(H, W, px, 0.25, 708, aniso=(30, 1)) + 0.3 * field(H, W, px, 2.0, 709, aniso=(10, 1))
    edge = np.exp(-np.maximum(-sd, 0) / 5.0)
    guard = keep_out(H, W, px, [first], 4.0)
    lost = sm((0.7 * field(H, W, px, 2.5, 710) + 0.3 * field(H, W, px, 0.7, 711) - 1.15 + 1.3 * edge) / 0.12) * (1 - guard)
    holes = sm((0.7 * field(H, W, px, 2.0, 712) + 0.3 * field(H, W, px, 0.6, 713) - 1.9 + 1.4 * edge) / 0.05) * (1 - guard)
    inside = (sd < 0) & (holes < 0.5)
    base = srgb(200, 165, 110)
    col = base * (1 + 0.05 * top_f)[..., None]
    patch = sm((field(H, W, px, 9.0, 714) + 0.4) / 1.4)
    col = col + (srgb(150, 110, 66) - col) * (0.28 * patch)[..., None]
    lower = srgb(170, 130, 82) * (1 + 0.09 * low_f)[..., None]
    col = col + (lower - col) * lost[..., None]
    col = col * (1 - 0.4 * np.exp(-np.maximum(-sd, 0) / 1.8))[..., None]
    rim = sm((holes - 0.2) / 0.3) * (1 - sm((holes - 0.45) / 0.05))
    col = col * (1 - 0.35 * rim)[..., None]
    # carbon ink: flat, soaked into the fibres, not standing on them; at its edges a little has run along the fibres, which go across
    core = sm((gaussian_filter(pen, 0.03 * px) - 0.35) / 0.3)
    bleed = gaussian_filter(core, (0.025 * px, 0.07 * px))
    ink = np.maximum(core, 0.42 * bleed) * level * (0.96 + 0.04 * np.clip(top_f, -1, 1)) * (1 - lost)
    ink = ink * (1 - 0.5 * sm((field(H, W, px, 0.05, 715) - 1.4) / 0.4))
    ink = ink * (1 - 0.55 * sm((field(H, W, px, 3.5, 719) - 0.6) / 0.8) * (1 - guard))       # where the ink has faded off the fibres
    col = col + (srgb(28, 22, 18) - col) * np.clip(ink * 1.05, 0, 1)[..., None]
    Z = 0.25 + 0.28 * np.clip(field(H, W, px, 22.0, 716) + 1.0, 0, None) + 0.05 * field(H, W, px, 4.0, 717)
    Z += 0.5 * edge * sm((field(H, W, px, 12.0, 718) + 0.5) / 1.0)
    Z += 0.008 * gaussian_filter(top_f * (1 - lost) + low_f * lost, 0.15 * px) - 0.07 * lost      # the fibres stand up only a little: under a low light more reads as a brushed sheet
    Z = np.where(inside, Z, -2.0).astype(np.float32)
    rough = np.clip(0.74 + 0.1 * lost + 0.03 * ink, 0, 1)          # carbon ink is as matt as the leaf
    relief = np.clip(0.5 + 0.1 * top_f * (1 - lost) + 0.1 * low_f * lost, 0, 1)
    save("greek", col, rough, relief, Z, w, h, px, 0.1, dict(first=first), chars=[dict(c=t, col=i, box=[]) for i, t in enumerate(text)])


# ---------------------------------------------------------------- the bone: a piece of turtle plastron, the cracks the fire drew in it, a 我 cut beside one

BONE_W, BONE_H, BONE_PX = 220.0, 172.0, 12
BONE_T, BONE_DOME = 6.0, 3.0
BONE_POLY = [(26, 24), (70, 8), (132, 6), (186, 12), (212, 40), (216, 92), (206, 140), (172, 164), (112, 168), (58, 154), (20, 116), (10, 66)]
MIDLINE = 182.0                                      # the seam down the middle of the plastron: the piece is of its left half and a strip of its right
SEAMS = ((52.0, 7.0, 34.0), (124.0, -6.0, 40.0))     # the seams across, between its shields: where they meet the midline, how far they bow, over how long
WO_AT, WO_SIZE, CUT = (146.0, 40.0), 21.0, 0.72      # 我's middle and height, mm, and the knife's line
# 我 as it is cut on 合6057 (賓組), Sinica's form 43.EFAB, from Wikimedia Commons File:我-oracle.svg (public domain): the middle lines of its drawn strokes, which is where the knife ran
# (the outline filled, thinned, traced; crew A, 2026-11-07). A long-shafted weapon: its toothed blade to the left, the hook to the right, a foot. Unit square, y down, height 1.
WO_FORM = (
    ((0.624, 0.282), (0.631, 0.271), (0.645, 0.030), (0.672, 0.019), (0.847, 0.041)),
    ((0.163, 0.060), (0.406, 0.072), (0.407, 0.180), (0.413, 0.190)),
    ((0.401, 0.335), (0.406, 0.202), (0.413, 0.190)),
    ((0.601, 0.282), (0.571, 0.256), (0.451, 0.198), (0.413, 0.190)),
    ((0.864, 0.258), (0.629, 0.295)),
    ((0.403, 0.370), (0.445, 0.366), (0.587, 0.295), (0.601, 0.282)),
    ((0.624, 0.282), (0.601, 0.282)),
    ((0.628, 0.295), (0.624, 0.282)),
    ((0.597, 0.974), (0.605, 0.697), (0.628, 0.295)),
    ((0.161, 0.324), (0.400, 0.335)),
    ((0.401, 0.336), (0.403, 0.370)),
    ((0.403, 0.370), (0.395, 0.507), (0.387, 0.560), (0.366, 0.566), (0.141, 0.579)),
    ((0.597, 0.974), (0.463, 0.980)),
    ((0.597, 0.974), (0.731, 0.976)),
)
# the cracks, in rows down each side of the midline as the diviners made them: where the crack starts, how long it runs down, where its branch leaves it, to which side, how long
CRACKS = (((166.0, 22.0), 30.0, (166.0, 31.0), -1, 12.0), ((166.0, 74.0), 30.0, (166.0, 83.0), -1, 12.0), ((166.0, 124.0), 26.0, (166.0, 132.0), -1, 11.0),
          ((198.0, 30.0), 26.0, (198.0, 38.0), 1, 10.0), ((198.0, 84.0), 26.0, (198.0, 92.0), 1, 10.0),
          ((112.0, 50.0), 30.0, (112.0, 59.0), -1, 12.0), ((112.0, 104.0), 28.0, (112.0, 112.0), -1, 11.0), ((60.0, 66.0), 28.0, (60.0, 75.0), -1, 11.0), ((62.0, 118.0), 24.0, (62.0, 126.0), -1, 10.0))


def strokes_mask(H, W, px, paths, at, size, width, r):
    """Straight cuts with a knife, each a little off true."""
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for path in paths:
        pts = [((at[0] + (x - 0.5) * size + r.normal(0, 0.12)) * px, (at[1] + (y - 0.5) * size + r.normal(0, 0.12)) * px) for x, y in path]
        d.line(pts, fill=255, width=max(int(width * px), 1), joint="curve")
    return gaussian_filter(np.asarray(im, np.float32) / 255, 0.08 * px)


def crack(d, start, length, branch_at, side_, blen, r, px):
    """A crack the fire drew: a wandering line down, and a branch off it to one side."""
    def walk(p, ang, n, step):
        out = [p]
        for _ in range(n):
            ang += float(r.normal(0, 0.16))
            p = (p[0] + np.cos(ang) * step, p[1] + np.sin(ang) * step)
            out.append(p)
        return out
    main = walk(start, np.pi / 2, int(length / 1.5), 1.5)
    br = walk(branch_at, np.pi / 2 + side_ * np.radians(64), int(blen / 1.2), 1.2)
    for line, wd in ((main, 0.4), (br, 0.32)):
        d.line([(x * px, y * px) for x, y in line], fill=255, width=max(int(wd * px), 1), joint="curve")


def write_bone():
    px, w, h = BONE_PX, BONE_W, BONE_H
    H, W, u, v = plane(w, h, px)
    r = np.random.default_rng(801)
    sd = outline(BONE_POLY, H, W, px, 802, ((1.4, 8.0), (0.7, 2.2), (0.25, 0.6)))
    wo = strokes_mask(H, W, px, WO_FORM, WO_AT, WO_SIZE, CUT, np.random.default_rng(803))
    for _ in range(44):
        r.normal(0, 0.12)           # what the provisional 我 drew from r, so that the cracks after it are the ones already shot
    first = ink_box(wo, px)
    carve = impression(wo, px, slope=1.4, most=0.5)
    # the seams between its shields: a zigzag suture down the middle, and two across it, bowed
    mid = np.abs(u - (MIDLINE + 0.5 * np.sin(v / 3.1) + 0.8 * field(H, W, px, 4.0, 805)))
    seam = np.exp(-(mid / 0.35) ** 2)
    for k, (y0, bow, L) in enumerate(SEAMS):
        t = np.clip((MIDLINE - u) / 160.0, 0, 1)
        yy = y0 + bow * np.sin(np.pi * t) * (1 if k == 0 else 1) + 0.6 * field(H, W, px, 3.0, 806 + k) + 0.4 * np.sin(u / 2.7)
        seam = np.maximum(seam, np.exp(-((v - yy) / 0.32) ** 2) * (u < MIDLINE))
        yy2 = y0 + 4.0 + 0.6 * field(H, W, px, 3.0, 808 + k) + 0.4 * np.sin(u / 2.9)       # and on the right half, where it meets the midline a little lower
        seam = np.maximum(seam, np.exp(-((v - yy2) / 0.32) ** 2) * (u > MIDLINE))
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for c in CRACKS:
        crack(d, *c, r, px)
    cracks = gaussian_filter(np.asarray(im, np.float32) / 255, 0.05 * px)
    # behind each crack the diviners bored a hollow and put fire to it: through the thin bone it shows as a dark oval where the crack branches
    scorch = np.zeros((H, W), np.float32)
    for (_, _, at, _, _) in CRACKS:
        scorch = np.maximum(scorch, np.exp(-(((u - at[0] - 1.5) / 4.0) ** 2 + ((v - at[1] + 1.0) / 6.5) ** 2) ** 1.5))
    scorch *= sm((field(H, W, px, 1.0, 807) + 1.6) / 1.2)
    # its body: a shallow dome, broken all round on old breaks that show the spongy bone inside
    dome = BONE_T + BONE_DOME * (1 - ((u - 120) / 130) ** 2 - ((v - 90) / 110) ** 2)
    e = np.clip(-sd / (2.2 + 0.8 * field(H, W, px, 8.0, 808)), 0, 1)
    Z = dome * e ** 0.45 + 0.1 * field(H, W, px, 15.0, 809) + 0.01 * field(H, W, px, 0.6, 810)
    brk = 1 - sm(np.clip(e * 1.05, 0, 1))
    pores = sm((field(H, W, px, 0.25, 811) - 0.3) / 0.4)
    Z += brk * (0.5 * field(H, W, px, 1.2, 812) - 0.35 * pores)
    Z -= (carve + 0.15 * seam + 0.12 * cracks) * (1 - brk)
    Z = np.where(sd < 0, Z, -2.0).astype(np.float32)
    # colour: old bone, honey going to brown where the earth stained it, darker towards the breaks, dirt in every cut, the scorch
    col = srgb(206, 182, 136) * (1 + 0.035 * field(H, W, px, 0.5, 815) + 0.04 * (u > MIDLINE))[..., None]
    mott = sm((0.6 * field(H, W, px, 12.0, 813) + 0.4 * field(H, W, px, 3.0, 814) + 0.1) / 1.2)
    col = col + (srgb(146, 110, 70) - col) * (0.5 * mott)[..., None]
    blot = sm((field(H, W, px, 2.0, 818) - 1.2) / 0.5)
    col = col + (srgb(98, 72, 50) - col) * (0.35 * blot)[..., None]
    col = col + (srgb(112, 72, 44) - col) * (0.75 * scorch)[..., None]
    col = col + (srgb(52, 38, 28) - col) * np.clip(cracks * 1.3, 0, 1)[..., None]
    cut_dirt = sm(carve / 0.2)
    col = col + (srgb(78, 60, 44) - col) * (0.9 * cut_dirt)[..., None]
    col = col + (srgb(120, 92, 64) - col) * (0.75 * seam)[..., None]
    col = col + (srgb(176, 146, 104) * (1 - 0.35 * pores)[..., None] - col) * brk[..., None]
    col = col * (1 - 0.3 * np.exp(-np.maximum(-sd, 0) / 5.0))[..., None]
    rough = np.clip(0.4 + 0.06 * field(H, W, px, 3.0, 816) + 0.35 * brk + 0.25 * cut_dirt, 0, 1)
    relief = np.clip(0.5 + 0.15 * field(H, W, px, 0.3, 817) * (0.3 + brk), 0, 1)
    save("bone", col, rough, relief, Z, w, h, px, 0.15, dict(first=first), chars=[dict(c="我", col=0, box=first)])


# ---------------------------------------------------------------- the slips: seven of bamboo, bound with two cords, 報任安書 in a clerical hand

SLIP_N, SLIP_W, SLIP_GAP, SLIP_LEN, SLIP_THICK = 7, 9.0, 1.1, 140.0, 2.2
SLIPS_W, SLIPS_H, SLIPS_PX = 124.0, 150.0, 18
SLIP_RIGHT = 84.0                         # the first slip's middle, mm from the left: its cords run on past it to the right
SLIP_TOP = 6.0
SLIP_EM, SLIP_SQUASH, SLIP_PITCH, SLIP_HEAD = 7.6, 0.8, 8.4, 16.0      # a character's em and how flat the clerical hand sets it, its pitch, and the first one's middle below the slip's head
SLIP_CHARS, SLIP_CORD_SLOT = 13, 6
REN_AN = "人固有一死死或重於泰山或輕於鴻毛用之所趣異也太上不辱先其次不辱身其次不辱理色其次不辱辭令其次詘體受辱其次易服受辱其次關木索被箠楚受辱其次剔毛髮嬰金鐵受辱其次毀肌膚斷肢體受辱最下腐刑極矣"
CORD_R = 0.95


def slip_rows():
    return [SLIP_HEAD + SLIP_PITCH * (i if i < SLIP_CORD_SLOT else i + 1) for i in range(SLIP_CHARS)]


def write_slips():
    px, w, h = SLIPS_PX, SLIPS_W, SLIPS_H
    H, W, u, v = plane(w, h, px)
    r = np.random.default_rng(901)
    assert REN_AN.startswith("人固有一死") and "我" not in REN_AN and len(REN_AN) >= SLIP_N * SLIP_CHARS, len(REN_AN)
    f = face("Libian.ttc", 1, SLIP_EM * px)
    Z = np.full((H, W), -2.0, np.float32)
    col = np.zeros((H, W, 3), np.float32)
    on = np.zeros((H, W), np.float32)
    which = np.full((H, W), -1, np.int16)
    tone = np.zeros((H, W), np.float32)
    A = np.zeros((H, W), np.float32)
    first = None
    lefts = []
    cx = SLIP_RIGHT
    rows = slip_rows()
    for k in range(SLIP_N):
        hw = SLIP_W / 2 + float(r.normal(0, 0.12))
        top = SLIP_TOP + float(r.normal(0, 0.7))
        tilt = float(r.normal(0, 0.004))
        xc = cx + tilt * (v - 70) + float(r.normal(0, 0.3)) * ((v - 70) / 70) ** 2 + 0.06 * field(H, W, px, 6.0, 902 + k)
        d = np.abs(u - xc) - (hw + 0.05 * field(H, W, px, 0.5, 910 + k))
        along = np.minimum(v - top, top + SLIP_LEN - v)
        inside = (d < 0) & (along > 0)
        prof = np.sqrt(np.clip(1 - (np.maximum(d + 0.8, 0) / 0.8) ** 2, 0, 1)) * np.sqrt(np.clip(1 - (np.maximum(0.6 - along, 0) / 0.6) ** 2, 0, 1))
        z = SLIP_THICK * prof + 0.04 * field(H, W, px, 20.0, 920 + k)
        Z = np.where(inside, z, Z)
        which = np.where(inside, k, which)
        tone = np.where(inside, float(r.normal(0, 0.06)), tone)
        on = np.maximum(on, inside.astype(np.float32) * np.clip(-d / 0.6, 0, 1))
        lefts.append((cx - hw, cx + hw))
        # its thirteen characters, down its middle
        for i, y in enumerate(rows):
            c = REN_AN[k * SLIP_CHARS + i]
            box = f.getbbox(c, anchor="mm")
            gw, gh = box[2] - box[0] + 10, box[3] - box[1] + 10
            im = Image.new("L", (gw * 2, gh * 2), 0)
            ImageDraw.Draw(im).text((gw, gh), c, 255, font=f.font_variant(size=f.size * 2), anchor="mm")
            a = np.asarray(im.resize((gw, max(int(gh * SLIP_SQUASH), 1)), Image.BOX), np.float32) / 255
            xm = cx + tilt * (top + y - 70) + float(r.normal(0, 0.15))
            l, t = int(xm * px - a.shape[1] / 2), int((top + y) * px - a.shape[0] / 2 + r.normal(0, 0.15) * px)
            lv = float(np.clip(r.normal(0.93, 0.05), 0.75, 1.0))
            paste(A, a * lv, l, t)
            if first is None:
                first = ink_box(a, px, l / px, t / px)
        cx -= SLIP_W + SLIP_GAP + float(r.normal(0, 0.15))
    print("slips", SLIP_N, "the last character", REN_AN[SLIP_N * SLIP_CHARS - 1], flush=True)
    # bamboo: fibres along the slip, its vascular strands dark flecks along them, each slip its own shade, stains and the ends darker
    fib = 0.6 * field(H, W, px, 0.05, 930, aniso=(80, 1)) + 0.4 * field(H, W, px, 0.2, 931, aniso=(25, 1))
    strand = sm((field(H, W, px, 0.04, 932, aniso=(30, 1)) - 1.7) / 0.4)
    base = srgb(146, 104, 62)
    col = base * (1 + tone + 0.07 * fib)[..., None]
    col = col * (1 - 0.35 * strand)[..., None]
    stain = sm((field(H, W, px, 7.0, 933) - 0.2) / 0.9)
    col = col + (srgb(110, 76, 44) - col) * (0.3 * stain)[..., None]
    col = col * (0.72 + 0.28 * sm(on))[..., None]
    # the brush's ink, a little ragged where it met the fibres
    ink = sm((gaussian_filter(A, 0.03 * px) - 0.42 + 0.06 * fib) / 0.16)
    ink = ink * (1 - 0.35 * sm((field(H, W, px, 0.08, 934, aniso=(4, 1)) - 1.2) / 0.5))
    ink = ink * (1 - 0.3 * sm((field(H, W, px, 4.0, 935) - 0.5) / 0.8))
    col = col + (srgb(30, 23, 17) - col) * np.clip(ink * (which >= 0), 0, 1)[..., None]
    # the cords: two-ply hemp across every slip, the twist showing on top, past the first slip into a knot and two loose ends lying on the table
    cords = np.zeros((H, W), np.float32)
    cord_on = np.zeros((H, W), np.float32)
    x_end = SLIP_RIGHT + SLIP_W / 2 + 5.0
    for j, yc in enumerate((SLIP_TOP + 7.0, SLIP_TOP + rows[SLIP_CORD_SLOT - 1] + SLIP_PITCH)):
        y_line = yc + 0.4 * np.sin(u / 7.0 + j) + 0.15 * field(H, W, px, 3.0, 940 + j)
        dist = np.abs(v - y_line)
        over = (u > lefts[-1][0] - 2.0) & (u < x_end)
        bed = np.where(which >= 0, np.maximum(Z, 0), 0.6)
        tube = np.sqrt(np.clip(CORD_R ** 2 - dist ** 2, 0, None))
        z = np.where(over & (dist < CORD_R), bed + CORD_R * 0.7 + tube, -9)
        # the knot and the two ends
        kx, ky = x_end + 2.5, yc + 0.6
        knot = np.sqrt(np.clip(2.0 ** 2 - (u - kx) ** 2 - (v - ky) ** 2, 0, None)) * 1.1
        z = np.maximum(z, np.where(knot > 0, knot + 0.4, -9))
        for s, (bend, length) in enumerate(((0.35, 22.0), (0.8, 17.0))):
            t = np.clip((u - kx) / length, 0, 1)
            yy = ky + bend * length * t ** 1.6 + (s - 0.5) * 1.6
            dd = np.abs(v - yy - 0.3 * field(H, W, px, 2.0, 950 + 2 * j + s))
            rr = CORD_R * (1 - 0.55 * sm((t - 0.75) / 0.25))
            tail = np.where((u > kx) & (u < kx + length) & (dd < rr), rr + np.sqrt(np.clip(rr ** 2 - dd ** 2, 0, None)), -9)
            z = np.maximum(z, tail)
        cord_on = np.maximum(cord_on, (z > -5).astype(np.float32))
        cords = np.maximum(cords, z)
    twist = 0.5 + 0.5 * np.sin(2 * np.pi * (u * 0.71 + v * 0.71) / 1.5 + 0.8 * field(H, W, px, 0.3, 960))
    cord_z = cords - 0.12 * (1 - twist) * cord_on
    top_cord = (cord_z > Z) & (cord_on > 0)
    Z = np.where(top_cord, cord_z, Z).astype(np.float32)
    hemp = srgb(86, 70, 52) * (0.8 + 0.3 * twist + 0.06 * field(H, W, px, 0.2, 961))[..., None]
    col = np.where(top_cord[..., None], hemp, col)
    col = np.where((Z < 0)[..., None], srgb(90, 70, 50), col)
    rough = np.clip(np.where(top_cord, 0.9, 0.62 - 0.12 * ink + 0.05 * fib), 0, 1)
    relief = np.clip(0.5 + 0.3 * fib * (~top_cord) + 0.3 * (twist - 0.5) * top_cord, 0, 1)
    save("slips", col, rough, relief, Z, w, h, px, 0.1, dict(first=first), chars=[dict(c=REN_AN[k * SLIP_CHARS:(k + 1) * SLIP_CHARS], col=k, box=[]) for k in range(SLIP_N)])


# ---------------------------------------------------------------- the Song leaf: a half leaf of a woodblock print, torn, worm-eaten

SONG_W, SONG_H, SONG_PX = 122.0, 196.0, 14
SONG_EM, SONG_COL, SONG_ROW = 11.0, 15.5, 12.4
SONG_RIGHT, SONG_TOP = 99.0, 40.0           # the frame's inner right line, and the head of the text
SONG_TITLE = "將進酒"
SONG_TEXT = "君不見黃河之水天上來奔流到海不復迴君不見高堂明鏡悲白髮朝如青絲暮成雪人生得意須盡歡莫使金樽空對月"
SONG_ROWS = 18
SONG_POLY = [(37.6, 3.0), (121.0, 2.0), (121.5, 127.0), (104.0, 133.5), (86.0, 131.0), (63.0, 137.5), (37.6, 139.0)]     # torn across under the eighth row, so that in 二 the foot of the frame is the table's, for the subtitle (film.director.md 3.6). It was torn at 160 to 177.5
SONG_TEAR = 125.0           # below this the edge is the tear
SONG_SPLIT = 98.0           # where the block had split across (it was 128)


def write_song():
    px, w, h = SONG_PX, SONG_W, SONG_H
    H, W, u, v = plane(w, h, px)
    r = np.random.default_rng(1001)
    assert "我" not in SONG_TEXT and SONG_TEXT.startswith("君不見黃河之水天上來") and SONG_TEXT.endswith("空對月")
    sd_cut = outline(SONG_POLY, H, W, px, 1002, ((0.3, 8.0), (0.1, 1.0)))
    torn = outline(SONG_POLY, H, W, px, 1003, ((1.1, 3.0), (0.5, 0.9), (0.2, 0.25)))
    tear_side = ((u < 39.5) | (v > SONG_TEAR)).astype(np.float32)
    sd = sd_cut * (1 - tear_side) + torn * tear_side
    # the block: frame lines, column rules, the characters, cut in pear wood and printed with a brush of ink rubbed over it
    f = face("Kaiti.ttc", 4, SONG_EM * px)
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    def line(x0, y0, x1, y1, wd):
        n = 24
        pts = [(x0 + (x1 - x0) * t + 0.08 * r.normal(), y0 + (y1 - y0) * t + 0.08 * r.normal()) for t in np.linspace(0, 1, n)]
        d.line([(x * px, y * px) for x, y in pts], fill=255, width=max(int(wd * px), 1))
    top_y = SONG_TOP - 1.6
    line(SONG_RIGHT + 0.4, top_y, SONG_RIGHT + 0.4, 260, 0.32)
    line(SONG_RIGHT + 1.9, top_y - 0.4, SONG_RIGHT + 1.9, 260, 0.95)
    line(20.0, top_y, SONG_RIGHT + 2.3, top_y, 0.95)
    for k in range(1, 6):
        x = SONG_RIGHT - SONG_COL * k
        line(x, top_y, x, 260, 0.28)
    first = None
    cells = [(0, 2 + i, c) for i, c in enumerate(SONG_TITLE)] + [(1 + i // SONG_ROWS, i % SONG_ROWS, c) for i, c in enumerate(SONG_TEXT)]
    for k, row, c in cells:
        x = SONG_RIGHT - SONG_COL * (k + 0.5) + float(r.normal(0, 0.12))
        y = SONG_TOP + SONG_ROW * (row + 0.5) + float(r.normal(0, 0.12))
        d.text((x * px, y * px), c, 255, font=f, anchor="mm")
        if first is None and k == 1:
            b = f.getbbox(c, anchor="mm")
            first = [x + b[0] / px, y + b[1] / px, x + b[2] / px, y + b[3] / px]
    block = np.asarray(im, np.float32) / 255
    grain = field(H, W, px, 0.12, 1004, aniso=(1, 40))
    ink = sm((gaussian_filter(block, 0.04 * px) - 0.45 + 0.05 * field(H, W, px, 0.05, 1005)) / 0.12)
    ink = ink * (0.86 + 0.1 * np.clip(grain, -1.5, 1.5)) * (1 - 0.7 * sm((field(H, W, px, 0.05, 1006) - 2.0) / 0.3))
    # the block had split across: a fine white line through everything printed on it
    split = np.abs(v - (SONG_SPLIT + 0.035 * (u - 60) + 0.25 * field(H, W, px, 4.0, 1007)))
    ink = ink * (1 - np.exp(-(split / 0.12) ** 2))
    # bamboo paper, yellowed, darker at its edges, foxed, a tide line; worm holes all through it, none on the first character
    guard = keep_out(H, W, px, [first], 3.0)
    holes = np.zeros((H, W), np.float32)
    hm = Image.new("L", (W, H), 0)
    dh = ImageDraw.Draw(hm)
    for _ in range(34):
        x, y = float(r.uniform(38, 121)), float(r.uniform(4, 190))
        a = float(r.uniform(0, 2 * np.pi))
        wd = float(r.uniform(0.8, 1.7))
        pts = [(x, y)]
        for _ in range(int(r.integers(1, 9))):
            a += float(r.normal(0, 0.7))
            x, y = x + np.cos(a) * 0.9, y + np.sin(a) * 0.9
            pts.append((x, y))
        dh.line([(p * px, q * px) for p, q in pts], fill=255, width=int(wd * px), joint="curve")
        dh.ellipse([(pts[0][0] - wd / 2) * px, (pts[0][1] - wd / 2) * px, (pts[0][0] + wd / 2) * px, (pts[0][1] + wd / 2) * px], fill=255)
    holes = gaussian_filter(np.asarray(hm, np.float32) / 255, 0.1 * px) * (1 - guard)
    holes = sm((holes + 0.15 * field(H, W, px, 0.2, 1008) - 0.45) / 0.1)
    for yy in (24.0, 118.0):          # the old binding's thread holes
        holes = np.maximum(holes, sm((1.0 - np.hypot(u - 113.0, v - yy)) / 0.15))
    inside = (sd < 0) & (holes < 0.5)
    paper = srgb(214, 190, 146) * (1 + 0.035 * field(H, W, px, 0.06, 1009) + 0.03 * field(H, W, px, 3.0, 1010))[..., None]
    paper = paper * (1 + 0.012 * np.sin(2 * np.pi * v / 1.1))[..., None]
    edge = np.exp(-np.maximum(-sd, 0) / 9.0)
    paper = paper * (1 - edge[..., None] * np.array((0.12, 0.2, 0.32), np.float32))
    tide = np.hypot((u - 100) / 1.0, (v - 150) / 0.8) + 3.0 * field(H, W, px, 5.0, 1011)
    paper = paper * (1 - (0.1 * sm((tide - 22) / 3.0) * (1 - sm((tide - 26) / 1.0)) + 0.04 * (tide < 25))[..., None] * np.array((0.6, 0.9, 1.3), np.float32))
    fox = sm((field(H, W, px, 0.6, 1012) - 2.1) / 0.3)
    paper = paper * (1 - (0.25 * fox)[..., None] * np.array((0.4, 0.7, 1.0), np.float32))
    ring = sm((holes - 0.05) / 0.2) * (1 - sm((holes - 0.4) / 0.1))
    paper = paper * (1 - 0.3 * ring)[..., None]
    col = paper + (srgb(20, 18, 17) - paper) * np.clip(ink, 0, 1)[..., None]
    Z = 0.12 + 0.3 * np.clip(field(H, W, px, 25.0, 1013) + 1.2, 0, None) + 0.35 * edge * sm((field(H, W, px, 10.0, 1014) + 0.4) / 1.0)
    Z += 0.012 * field(H, W, px, 0.1, 1015)
    Z = np.where(inside, Z, -2.0).astype(np.float32)
    rough = np.clip(0.8 - 0.15 * ink, 0, 1)
    relief = np.clip(0.5 + 0.2 * field(H, W, px, 0.08, 1016) - 0.15 * ink, 0, 1)
    save("song", col, rough, relief, Z, w, h, px, 0.12, dict(first=first), chars=[dict(c=SONG_TITLE, col=0, box=[]), dict(c=SONG_TEXT, col=1, box=[])])


WRITERS = dict(stone=write_stone, greek=write_greek, bone=write_bone, slips=write_slips, song=write_song)

if __name__ == "__main__":
    what = sys.argv[2] if len(sys.argv) > 2 else "all"
    assert sys.argv[1] == "write", __doc__
    for name in (WRITERS if what == "all" else [what]):
        WRITERS[name]()
