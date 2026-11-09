"""The ends of the three papers (81.6–85.4). Each is where its writer stopped, and then it goes on, in that writer's own hand, with no one there: the first strokes of one character, which is never named. Worked out over the very maps film_docs.py made for the sheet, so the paper, its grain and its fibres are the ones already photographed, and only the place of the new ink changes.

    python film_docs_end.py letter|poem|ruling        # maps/end/DOC_before_*.png, DOC_stop_*.png, DOC_when.png
    Blender -b --factory-startup -P film_docs3d.py -- --doc letter --shot end --state stop --samples 400
    Blender -b --factory-startup -P film_docs3d.py -- --doc letter --shot end --seq --samples 100
    python film_docs_end.py low                       # wo/end_low.mp4 and its contact sheet, from the frames of --seq
    python film_docs_end.py hi                        # wo/end_low_1080.mp4, the same frames at their own 1920

Goes to ink/film_out/film/imprint/maps/end/. before: as it was left, the letter's last stroke not yet dry. stop: the new strokes as far as they go, not yet dry either. Each state has _col and _rough, and _wet where something is wet, and the ruling _relief. when: how far along the writing each point of the new ink is reached.
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d

import film_docs as D
from film_big import _band
from film_press_maps import fibres
from ink_grade import sm

END = os.path.join(D.MAPS, "end")
FIRST = "我"              # never said


def read(name, mode):
    return np.asarray(Image.open(os.path.join(D.MAPS, name)).convert(mode))


def linear(a):
    a = np.float32(a) / 255
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)


def box_of(mask, margin):
    """Where something is, with room round it: top, bottom, left, right."""
    ys, xs = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
    h, w = mask.shape
    return max(ys[0] - margin, 0), min(ys[-1] + margin + 1, h), max(xs[0] - margin, 0), min(xs[-1] + margin + 1, w)


def changed(base8, new, box, seed):
    """base8 with the part in box replaced, dithered as save8 dithers; everything else stays the very same bytes."""
    y0, y1, x0, x1 = box
    out = base8.copy()
    r = np.random.default_rng(seed)
    out[y0:y1, x0:x1] = np.uint8(np.clip(new * 255 + r.uniform(0, 1, new.shape), 0, 255))
    return out


def save(doc, state, **maps):
    os.makedirs(END, exist_ok=True)
    for kind, a in maps.items():
        Image.fromarray(a).save(os.path.join(END, f"{doc}_{state}_{kind}.png"), compress_level=3)
    print(doc, state, ", ".join(maps), flush=True)


def wet_film(wet):
    """How high a wet stroke stands, smoothed as a film of ink lies: 8 bits are enough for the little of it there is."""
    return np.uint8(np.clip(wet * 255, 0, 255) + 0.5)


def save_when(doc, when, A, box):
    """When each point of the new ink is reached, 0 to 0.97 of the way, the sheet's size in 16 bits. What feathers out of the ink goes with the ink nearest it; everything else never comes."""
    y0, y1, x0, x1 = box
    near = distance_transform_edt(A[y0:y1, x0:x1] < 0.02, return_distances=False, return_indices=True)
    out = np.ones(A.shape, np.float32)
    out[y0:y1, x0:x1] = when[y0:y1, x0:x1][near[0], near[1]] * 0.97
    Image.fromarray(np.uint16(out * 65535 + 0.5)).save(os.path.join(END, f"{doc}_when.png"), compress_level=3)


# ---------------------------------------------------------------- the letter: 言 not yet dry, then below it two strokes, the second stopping

STOP_AT = 0.56            # how far along its way the second stroke has got


def stopping(path):
    """A stroke's way as a hand goes that hesitates: pressing a little where it stopped once on the way, and at the place it stops now, where the ink gathers."""
    far = np.concatenate([[0], np.cumsum([np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])])])
    out = []
    for p, s in zip(path, far / far[-1]):
        press = 0.1 * np.exp(-((s - 0.27) / 0.05) ** 2) + 0.2 * np.exp(-((s - STOP_AT) / 0.05) ** 2)
        out.append((p[0], p[1], p[2] + press, *p[3:]))
    return out


def letter_end():
    chars, _ = D.letter_plan()
    px = D.LETTER_PX
    h, w = int(D.LETTER_H * px), int(D.LETTER_W * px)
    pen = D.Pen(640, D.LETTER_BODY, seed=5)
    hand, size = pen.hand, pen.tooth.shape[0]
    for _ in chars[:-1]:      # each was written twice, as itself and as its solid shape: the grain under the last must be the same
        pen.r.integers(0, size, 2)
        pen.r.integers(0, size, 2)
    last = chars[-1]
    assert last["c"] == "言"
    i, j = pen.r.integers(0, size, 2)
    tooth0, fibre0 = pen.tooth, pen.fibre
    hand.brush.seed = last["seed"]
    hand.tooth, hand.fibre = np.roll(tooth0, (i, j), (0, 1)), np.roll(fibre0, (j, i), (0, 1))
    states = []
    hand.design(last["marks"], tone=0.97 - 0.1 * max(1.1 - last["ink"], 0), each=lambda out, way: states.append(out.copy()))
    hand.brush.track = False
    final = np.float32(np.clip((states[-1] - states[-2]) * 3, 0, 1))
    pen.r.integers(0, size, 2)
    W_last = np.zeros((h, w), np.float32)
    D.over(W_last, final, last["x"], last["y"], last["side"], px)
    # below it, in the same hand, on the same load
    r = np.random.default_rng(1926)
    ms = D.marks(D.skeleton(FIRST, D.LETTER_STYLE), r, D.LETTER_BODY, 0.48, floor=0.32, ink=last["ink"] + 0.03, dry=1.0 + (1.14 - last["ink"]) * 2.4, quick=0.34, loose=1.0)
    x0, y0, x1, y1 = D.extent(ms)
    em = D.LETTER_EM * 1.02
    side = em * D.SQUARE
    cx, foot = last["x"] + last["mid"], last["y"] + last["top"] + last["high"]
    x, y = cx - (x0 + x1) / 200 * side, foot + 2.6 - y0 / 100 * side
    A = np.zeros((h, w), np.float32)
    path, ink, dry = ms[1]
    ways = []
    D.over(A, pen.write([ms[0], (stopping(path), ink, dry, STOP_AT)], 58213, tone=0.98, each=lambda out, way: ways.append(way)), x, y, side, px)
    reached = np.full((size, size), np.inf, np.float32)
    for k, ((t0, t1, l0, l1), first, along, _, _) in enumerate(ways):
        u = np.where(np.isfinite(first), (k + np.clip((first - along[0]) / (along[-1] - along[0]), 0, 1)) / 2, np.inf)
        reached[t0:t1, l0:l1] = np.minimum(reached[t0:t1, l0:l1], u)
    near = distance_transform_edt(~np.isfinite(reached), return_distances=False, return_indices=True)
    n_ = int(round(side * px))
    when = np.ones((h, w), np.float32)
    l, t = int(round(x * px)), int(round(y * px))
    when[t:t + n_, l:l + n_] = np.asarray(Image.fromarray(reached[near[0], near[1]]).resize((n_, n_), Image.BILINEAR), np.float32)
    ys, xs = np.nonzero(A > 0.1)
    os.makedirs(END, exist_ok=True)
    with open(os.path.join(END, "letter_place.json"), "w") as f:
        json.dump(dict(x=x, y=y, side=side, box=[float(xs.min()) / px, float(ys.min()) / px, float(xs.max()) / px, float(ys.max()) / px]), f)
    print("言", last["x"], last["y"], "the new strokes at", x, y, "side", side, flush=True)
    # the paper's own fibres and how letter_paper lets ink run along them
    save_when("letter", when, A, box_of(gaussian_filter(A, 2.0) > 0.003, 60))
    long_ = D.stretch(fibres(4096, 26000, 71, length=(40, 220), width=(1, 1), bright=(.3, 1.0), wiggle=.3), h, w)
    near = D.noise(h, w, 1.0, 82)
    col8, rough8 = read("letter_col.png", "RGB"), read("letter_rough.png", "L")
    for state, new in (("before", 0.0), ("stop", 1.0)):
        ink_ = A * new
        wet0 = np.maximum(W_last, ink_)
        box = box_of(gaussian_filter(wet0, 2.0) > 0.003, 60)
        y0_, y1_, x0_, x1_ = box
        col = linear(col8[y0_:y1_, x0_:x1_])
        rough = np.float32(rough8[y0_:y1_, x0_:x1_]) / 255
        a0 = ink_[y0_:y1_, x0_:x1_]
        a1 = 1 - (1 - a0) ** 1.5
        feather = gaussian_filter(a1, 0.24 * px) * sm(gaussian_filter(long_[y0_:y1_, x0_:x1_], 1.2) * 2.4 + 0.45 * near[y0_:y1_, x0_:x1_] + 0.1)
        a = np.clip(a1 + (1 - a1) * 0.55 * feather, 0, 1)
        tone = D.srgb(24, 20, 19)[None, None] + (D.srgb(62, 50, 42)[None, None] - D.srgb(24, 20, 19)[None, None]) * (1 - a1)[..., None]
        col = col + (tone - col) * a[..., None]
        rough = rough - 0.1 * a
        # wet: darker, not yet gone into the paper, shining a little where the light finds it
        was = W_last[y0_:y1_, x0_:x1_]
        wet = gaussian_filter(np.clip(np.maximum(was, a), 0, 1), 0.05 * px)
        col = col * (1 - 0.5 * wet[..., None])
        rough = rough + (0.24 - 0.1 * gaussian_filter(was, 0.05 * px) - rough) * wet       # 言 is what shows it is not yet dry: what comes after it, thin and scarcely there, has less of it to show
        film = np.zeros((h, w), np.float32)
        film[y0_:y1_, x0_:x1_] = gaussian_filter(np.clip(np.maximum(was, 0.5 * a), 0, 1), 0.12 * px)
        save("letter", state, col=changed(col8, D.to_srgb(col), box, 11), rough=changed(rough8, np.clip(rough, 0, 1), box, 12), wet=wet_film(film))


# ---------------------------------------------------------------- the poem: after 簪, its next column, two strokes and most of the way down a third

POEM_STOP = 0.72          # how far along its way the third stroke has got; it is the hook, past that, which would finish it


def poem_so_far(g):
    """Where the ink is: the first two strokes, and the third as far as it has got, ended as a brush ends that has only paused. In this face the third stands proud of the second's top edge, and that edge has a step in it where the hand was never going: the top edge is one movement here, and the third starts under it."""
    n, em = g.n, g.em
    ink = g.bold > 0
    yy, xx = np.mgrid[0:n, 0:n]
    line = g.lines[2]
    far = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    stop = POEM_STOP * g.length[2]
    i = int(np.clip(np.searchsorted(far, stop), 0, len(line) - 1))
    third = ((g.K == 2) & (g.S < stop)) | (np.hypot(xx - line[i, 0], yy - line[i, 1]) < 0.92 * g.bold[int(line[i, 1]), int(line[i, 0])])
    second = ink & (g.K == 1)
    tops = np.float32(np.argmax(second, 0))
    has = second.any(0)
    tops = np.interp(np.arange(n), np.nonzero(has)[0], tops[has])
    left, right = g.lines[1][:, 0].min() + 0.03 * em, g.lines[1][:, 0].max() + 0.012 * em       # its head, where it joins the first stroke's tail, and its end, are as they are
    sweep = sm((np.arange(n) - left) / (0.07 * em)) * sm((right - np.arange(n)) / (0.06 * em))
    top = tops + (gaussian_filter1d(tops, 0.06 * em, mode="nearest") - tops) * sweep
    above = (yy < top[xx]) & (sweep[xx] > 0) & (g.K >= 1)
    fill = (yy >= top[xx]) & (yy <= tops[xx]) & (sweep[xx] > 0) & has[xx]
    return ((ink & ((g.K == 0) | second | third)) & ~above) | fill


def poem_reached(g):
    """How far along the writing each point of the character is: stroke by stroke down its way, the middle of a stroke a little before its sides. What of the third lies in the second came with the second, and the third is only seen to start where it leaves it."""
    n = g.n
    yy, xx = np.mgrid[0:n, 0:n]
    second = (g.K == 1) & (g.bold > 0)
    has = second.any(0)
    under = np.interp(np.arange(n), np.nonzero(has)[0], (n - 1 - np.argmax(second[::-1], 0))[has])
    across = (g.K == 2) & (yy <= under[xx])
    leaves = float(g.S[across].max()) if across.any() else 0.0
    line = g.lines[1]
    far = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    bulge = 0.5 * g.V * g.V / g.R
    u = np.clip((g.S + bulge) / np.array([g.length[0], g.length[1], 1.0], np.float32)[np.minimum(g.K, 2)], 0, 1)
    u = np.where(g.K >= 2, np.clip((g.S + bulge - leaves) / (POEM_STOP * g.length[2] - leaves), 0, 1), u)
    k = np.minimum(g.K, 2)
    u = np.where(across, np.interp(xx, line[:, 0], far) / g.length[1], u)
    k = np.where(across, 1, k)
    return g.shift(np.float32((k + u) / 3))


def poem_end():
    chars = D.poem_plan()
    px = D.POEM_PX
    h, w = int(D.POEM_H * px), int(D.POEM_W * px)
    col8 = [c for c in chars if c["col"] == 8]
    cx = float(np.mean([(c["box"][0] + c["box"][2]) / 2 for c in col8])) - D.POEM_PITCH
    em = D.POEM_EM * 1.0
    weight, u, seed = D.POEM_WEIGHT * 1.14, 0.3, 417263
    n = int(em * D.SQUARE * px)
    g = D.Traced(FIRST, *D.XING, n=n, em=em * px, tilt=-0.8)
    l, t, r_, b = [v / px for v in g.box(weight)]
    x, y = cx - (l + r_) / 2, D.POEM_TOP - t
    grain = _band(2048, 1.0, 4.5, 301) * 0.7 + _band(2048, 3.0, 12.0, 302) * 0.5
    tooth = np.roll(grain, (seed % 2048, (seed >> 11) % 2048), (0, 1))[:n, :n]
    F, _ = D.full(g, tooth, weight, u, seed, keep=poem_so_far)
    A = np.zeros((h, w), np.float32)
    D.over(A, F, x, y, n / px, px)
    when = np.ones((h, w), np.float32)
    l_, t_ = int(round(x * px)), int(round(y * px))
    when[t_:t_ + n, l_:l_ + n] = poem_reached(g)
    os.makedirs(END, exist_ok=True)
    save_when("poem", when, A, box_of(gaussian_filter(A, 2.0) > 0.003, 60))
    print("the new column at", cx, "its ink from", x + l, "to", x + r_, "and", y + t, "to", y + b, flush=True)
    long_ = D.stretch(fibres(4096, 9000, 171, length=(120, 700), width=(2, 4), bright=(.3, 1.0), wiggle=.2), h, w)
    near = D.noise(h, w, 1.0, 177)
    col8_, rough8 = read("poem_col.png", "RGB"), read("poem_rough.png", "L")
    box = box_of(gaussian_filter(A, 2.0) > 0.003, 60)
    y0, y1, x0, x1 = box
    col = linear(col8_[y0:y1, x0:x1])
    a0 = A[y0:y1, x0:x1]
    feather = gaussian_filter(a0, 0.22 * px) * sm(gaussian_filter(long_[y0:y1, x0:x1], 0.8) * 2.2 + 0.3 * near[y0:y1, x0:x1])
    a = np.clip(a0 + (a0 < 0.04) * 0.3 * feather, 0, 1)
    tone = D.srgb(16, 15, 17)[None, None] + (D.srgb(52, 50, 54)[None, None] - D.srgb(16, 15, 17)[None, None]) * (1 - a0)[..., None]
    a = 1 - (1 - a) ** 1.7
    col = col + (tone - col) * a[..., None]
    rough = np.float32(rough8[y0:y1, x0:x1]) / 255 - 0.26 * a
    save("poem", "before", col=col8_, rough=rough8)
    save("poem", "stop", col=changed(col8_, D.to_srgb(col), box, 21), rough=changed(rough8, np.clip(rough, 0, 1), box, 22))
    with open(os.path.join(END, "poem_place.json"), "w") as f:
        json.dump(dict(x=x, y=y, side=n / px, cx=cx, box=[x + l, y + t, x + r_, y + b]), f)


# ---------------------------------------------------------------- the ruling: after the date, the next column's head, two strokes of the same type

# in Songti's 我, in hundredths of a square at 0.9 of an em: its first stroke, cut along its own lower edge where the upright comes off it; its long stroke's two ends. The long stroke's top and bottom across the uprights are taken from the face itself
SLANT = [(5, 36), (31.7, 28.3), (37.7, 26.95), (52, 26), (52, 10), (5, 10)]
LONG_LEFT, LONG_RIGHT, LONG_END = (5, 25), (62, 95), [(62, 41.9), (77.2, 41.9), (83.5, 34.8), (95, 34.8), (95, 47), (62, 47)]
FIRST_WAY, LONG_WAY = ((48.5, 20.0), (12.5, 30.0)), (9.0, 91.0)


def ruling_strokes(n):
    """The two strokes' shapes in a square of type n wide."""
    scale = (1 / 1.3) / 0.9
    to = lambda u, v: ((50 + (u - 50) * scale) / 100 * n, (50 + (v - 50) * scale) / 100 * n)
    a = Image.new("L", (n, n), 0)
    ImageDraw.Draw(a).text((n / 2, n / 2), FIRST, 255, font=D.face(D.SONG[0], D.SONG[1], D.TYPE_EM * D.RULING_PX * 3), anchor="mm")
    a = np.asarray(a, np.float32) / 255
    rows = np.nonzero(a[:, int(to(25, 0)[0])] > 0.5)[0]
    rows = rows[(rows > to(0, 38)[1]) & (rows < to(0, 50)[1])]
    shape = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(shape)
    d.polygon([to(u, v) for u, v in SLANT], fill=255)
    d.polygon([to(u, v) for u, v in LONG_END], fill=255)
    d.rectangle([to(LONG_LEFT[0], 39.5), to(LONG_LEFT[1], 48)], fill=255)
    d.rectangle([to(LONG_LEFT[1], 0)[0], rows[0], to(LONG_RIGHT[0], 0)[0], rows[-1]], fill=255)
    return a * (np.asarray(shape, np.float32) / 255)


def ruling_end():
    px = D.RULING_PX
    h, w = int(D.RULING_H * px), int(D.RULING_W * px)
    k = 3
    n = int(D.TYPE_EM * px * 1.3 * k)
    D._cast[FIRST + "二", D.SONG, D.TYPE_EM] = np.asarray(Image.fromarray(ruling_strokes(n)).resize((n // k, n // k), Image.BOX), np.float32)
    A, R, S = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    plan, last = D.ruling_plan()
    cx, cy = D.column_x(last + 1), D.column_y(2)
    D.set_type(A, R, S, FIRST + "二", cx, cy, D.SONG, D.TYPE_EM, np.random.default_rng(217), px)
    col8, rough8, relief8 = read("ruling_col.png", "RGB"), read("ruling_rough.png", "L"), read("ruling_relief.png", "L")
    box = box_of(A > 0.003, 40)
    y0, y1, x0, x1 = box
    a0 = A[y0:y1, x0:x1]
    col = linear(col8[y0:y1, x0:x1])
    col = col + (D.srgb(22, 20, 20) - col) * a0[..., None]
    rough = np.float32(rough8[y0:y1, x0:x1]) / 255 - 0.2 * a0
    relief = np.maximum(np.float32(relief8[y0:y1, x0:x1]) / 255, gaussian_filter(R, 0.04 * px)[y0:y1, x0:x1])
    print("the strokes at", cx, cy, "column", last + 1, flush=True)
    # as type that sets itself: down the first stroke from its head, then along the second from the left, its middle a little before its edges
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = 50 + (xx / px - cx) * 90 / D.TYPE_EM, 50 + (yy / px - cy) * 90 / D.TYPE_EM
    (hx, hy), (tx, ty) = FIRST_WAY
    long_ = np.hypot(tx - hx, ty - hy)
    along, off = ((u - hx) * (tx - hx) + (v - hy) * (ty - hy)) / long_, ((u - hx) * (ty - hy) - (v - hy) * (tx - hx)) / long_
    when = np.where(v < 34 + 0.0 * u, np.clip((along + 0.08 * off * off) / long_, 0, 1) / 2, 0.5 + np.clip((u - LONG_WAY[0] + 0.5 * (v - 43.7) ** 2) / (LONG_WAY[1] - LONG_WAY[0]), 0, 1) / 2)
    when = when + 0.004 * gaussian_filter(np.random.default_rng(218).normal(0, 1, (h, w)).astype(np.float32), 0.3 * px) / 0.1
    save_when("ruling", np.float32(np.clip(when, 0, 1)), A, box_of(A > 0.003, 40))
    save("ruling", "before", col=col8, rough=rough8, relief=relief8)
    save("ruling", "stop", col=changed(col8, D.to_srgb(col), box, 31), rough=changed(rough8, np.clip(rough, 0, 1), box, 32), relief=changed(relief8, np.clip(relief, 0, 1), box, 33))
    with open(os.path.join(END, "ruling_place.json"), "w") as f:
        json.dump(dict(x=cx, y=cy, em=D.TYPE_EM, col=last + 1), f)


# ---------------------------------------------------------------- the three together, small, for looking at

FRAMES = os.path.join(D.OUT, "seq")
WO = os.path.join(os.path.dirname(D.OUT.rstrip("/")), "wo")
FPS = 30


def contact(frames, stamps, path):
    """Eight frames of a shot and when each is."""
    k = [round(i * (len(frames) - 1) / 7) for i in range(8)]
    sheet = Image.new("RGB", (4 * 480 + 3 * 6, 2 * 270 + 6), (40, 40, 40))
    for j, i in enumerate(k):
        im = Image.fromarray(frames[i]).resize((480, 270), Image.BOX)
        ImageDraw.Draw(im).text((8, 6), "%.2f" % stamps[i], (230, 40, 40), font=D.face("Songti.ttc", 7, 18))
        sheet.paste(im, (j % 4 * 486, j // 4 * 276))
    sheet.save(path)


def low(w=960):
    """end_low.mp4 at 960, or at 1920 end_low_1080.mp4: the same frames, which are 1920 already."""
    os.makedirs(WO, exist_ok=True)
    tag = "" if w == 960 else "_1080"
    out = os.path.join(WO, f"end_low{tag}.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{w * 9 // 16}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    frames, stamps = [], []
    t = 81.6
    for name, count in (("d1_end", 48), ("d2_end", 36), ("d3_end", 30)):
        for i in range(count):
            f = np.asarray(Image.open(os.path.join(FRAMES, name, "%04d.png" % i)).convert("RGB"))      # no subtitle: those go on in the rough cut
            small = f if f.shape[1] == w else np.asarray(Image.fromarray(f).resize((w, w * 9 // 16), Image.BOX))
            ff.stdin.write(small.tobytes())
            frames.append(small if w == 960 else np.asarray(Image.fromarray(small).resize((960, 540), Image.BOX)))
            stamps.append(t)
            t += 1 / FPS
    ff.stdin.close()
    ff.wait()
    contact(frames, stamps, os.path.join(WO, f"end_low{tag}_contact.png"))
    print("wrote", out, len(frames), "frames", flush=True)


if __name__ == "__main__":
    for doc in sys.argv[1:]:
        dict(letter=letter_end, poem=poem_end, ruling=ruling_end, low=low, hi=lambda: low(1920))[doc]()
