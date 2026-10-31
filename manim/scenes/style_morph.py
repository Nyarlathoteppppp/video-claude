"""形变动画：山骨成字。An ink painting of karst peaks becomes the character 山, by moving its ink.

Each column of the painting is carried to a column of the glyph by 1-D optimal transport, so the three peaks
land on the three strokes and the ink runs wet between them. The character takes its bones from Weibei and is then
written, not typeset: every hair of a brush is traced along the three strokes in stroke order, so the ink inside the
form has pressed heads, a turn, and flying white that runs along the stroke as the hairs go dry.
"""
import math
from functools import lru_cache

import cairo
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d, map_coordinates

from styles_common import H, W, hx, surface

PAPER = np.array(hx("#F3F1E8"), np.float32)
INK = np.array(hx("#15191B"), np.float32)
ABSORB = -np.log(INK)
CINNABAR = np.array(hx("#B8391F"), np.float32)

# the three strokes of 山 in writing order: x, y, brush width, ink left in the hairs; and when each is written
STROKES = (
    (2.42, 2.76, ((918, 64, 150, 1.0), (946, 155, 190, 1.0), (936, 280, 140, .98), (921, 586, 108, .84), (921, 758, 118, .60), (921, 850, 118, .40))),
    (2.70, 3.06, ((584, 456, 130, 1.0), (612, 550, 190, 1.0), (583, 694, 120, .88), (589, 801, 116, .72), (637, 864, 128, .62), (788, 834, 114, .54),
                  (917, 808, 92, .46), (1046, 795, 90, .38), (1175, 799, 104, .28), (1283, 815, 110, .20), (1330, 826, 60, .10))),
    (3.00, 3.24, ((1290, 456, 130, 1.0), (1322, 550, 190, .96), (1288, 680, 136, .72), (1267, 780, 118, .44), (1252, 844, 100, .26), (1250, 905, 80, .12))),
)
HEADS = ((860, 1020, 106, 272), (530, 690, 500, 668), (1215, 1395, 504, 672))     # each stroke's head: x range, top, and the row where it joins the body
GLYPH_SIZE = 1206           # 840 px wide
GLYPH_TOP = 110


def norm(a):
    return (a - a.mean()) / (a.std() + 1e-6)


def noise2(seed, sigma, shape=(H, W), mode="reflect"):
    return norm(gaussian_filter(np.random.default_rng(seed).standard_normal(shape), sigma, mode=mode)).astype(np.float32)


def noise1(seed, sigma, n=W):
    return norm(gaussian_filter1d(np.random.default_rng(seed).standard_normal(n + 400), sigma))[200:200 + n]


def ridged1(seed, base, octaves=6):
    out = np.zeros(W)
    amp = 1.0
    for i in range(octaves):
        out += amp * (1 - 2 * np.abs(noise1(seed + i, base / 2 ** i) * .5).clip(0, 1))
        amp *= .55
    return out


@lru_cache(1)
def stock():
    # xuan paper: long fibres lying every way, not a weave
    r = np.random.default_rng(1)
    strands, c = a8()
    for _ in range(16000):
        x, y, ang = r.uniform(0, W), r.uniform(0, H), r.uniform(0, math.pi)
        L, bend = r.uniform(14, 110), r.uniform(-.7, .7)
        c.move_to(x, y)
        c.curve_to(x + math.cos(ang) * L * .33, y + math.sin(ang) * L * .33, x + math.cos(ang + bend) * L * .66, y + math.sin(ang + bend) * L * .66,
                   x + math.cos(ang + bend) * L, y + math.sin(ang + bend) * L)
        c.set_line_width(r.uniform(.4, 1.2))
        c.set_source_rgba(0, 0, 0, r.uniform(.15, .75))
        c.stroke()
    fibre = norm(gaussian_filter(a8_array(strands), .5) + .22 * noise2(3, 1.2)).astype(np.float32)
    cloud = noise2(4, 70)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1 - .05 * (((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
    tone = (1 + .010 * fibre + .012 * cloud) * vig
    return PAPER[None, None] * tone[..., None], fibre, cloud


def gauss(x, c, s):
    return np.exp(-((x - c) / s) ** 2)


# centre, height, left reach, right reach, sharpness: no two towers share a profile
TOWERS = ((.515, .80, .046, .084, 2.6), (.275, .62, .072, .036, 2.2), (.755, .57, .036, .088, 2.8),
          (.395, .36, .028, .05, 1.8), (.635, .33, .062, .028, 2.0), (.13, .42, .094, .04, 1.7),
          (.895, .40, .044, .104, 1.9), (.0, .25, .1, .1, 2.), (1.0, .28, .1, .1, 2.))


def warped():
    x = np.linspace(0, 1, W)
    return x + .016 * noise1(12, 130) + .006 * noise1(13, 40)


@lru_cache(1)
def ridge():
    """The main skyline: three karst towers on the positions of the three strokes, with lesser shoulders."""
    x = warped()
    body = np.maximum.reduce([h * np.exp(-np.abs((x - c) / np.where(x < c, a, b)) ** p) for c, h, a, b, p in TOWERS])
    body = body + .22 * np.exp(-((x - .5) / .55) ** 2)
    ledges = .012 * np.sin(body * 74 + 2.2 * noise1(14, 60))
    rough = ridged1(11, 26) - 1.6
    top = H * (1.02 - (body + ledges) * .96) + rough * 21 * (.45 + body)
    return gaussian_filter1d(top, 1.2).astype(np.float32)


@lru_cache(1)
def faces():
    """-1 on the lit left face of whichever tower owns this column, +1 on its shaded right face."""
    x = warped()
    owner = np.argmax([h * np.exp(-np.abs((x - c) / np.where(x < c, a, b)) ** p) for c, h, a, b, p in TOWERS], 0)
    c = np.array([k[0] for k in TOWERS])[owner]
    reach = np.array([k[3] for k in TOWERS])[owner]
    return np.tanh((x - c) / reach * 2.4 + .55 * noise1(15, 44)).astype(np.float32)


def wedge(c, x, y, ang, length, width, a):
    """An axe-cut stroke: wide and wet where the brush lands, dragged out to a dry point."""
    s = np.linspace(0, 1, 9)
    bend = .18 * math.sin(x * .01 + y * .013)
    px = x + np.cumsum(np.cos(ang + bend * s)) * length / 9
    py = y + np.cumsum(np.sin(ang + bend * s)) * length / 9
    w = width * (1 - s) ** .7 * (.3 + .7 * np.minimum(1, s * 7))
    nx, ny = -np.sin(ang), np.cos(ang)
    c.set_source_rgba(0, 0, 0, a)
    c.move_to(px[0] + nx * w[0], py[0] + ny * w[0])
    for i in range(1, 9):
        c.line_to(px[i] + nx * w[i], py[i] + ny * w[i])
    for i in range(8, -1, -1):
        c.line_to(px[i] - nx * w[i], py[i] - ny * w[i])
    c.close_path()
    c.fill()


def a8():
    s = cairo.ImageSurface(cairo.FORMAT_A8, W, H)
    return s, cairo.Context(s)


def a8_array(s):
    s.flush()
    return np.ndarray((H, W), np.uint8, s.get_data(), strides=(s.get_stride(), 1)).astype(np.float32) / 255


@lru_cache(1)
def painting():
    """The mountain as ink density, built the way it would be painted: washes, then masses, then cuts, then moss."""
    top = ridge()
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.random.default_rng(5)
    _, fibre, _ = stock()
    depth = y - top[None]
    inside = (depth > 0).astype(np.float32)
    mist = np.clip((H * 1.0 - y) / (H * .46), 0, 1) ** 1.25

    # overlapping rock masses, each dark at its own top edge and dissolving downward
    d = np.zeros((H, W), np.float32)
    for i in range(16):
        seed = 100 + i
        off = 26 + i * 44 + r.uniform(-14, 14)
        inner = top + off + (ridged1(seed, 22 + i, 5) - 1.4) * (26 + i * 2) + noise1(seed + 50, 90) * (60 + i * 7)
        gap = y - inner[None]
        reach = r.uniform(70, 210)
        d = np.maximum(d, (.30 + .58 * r.random()) * np.exp(-np.maximum(gap, 0) / reach) * np.clip(gap / 2.5, 0, 1))
    wash = .10 + .09 * noise2(20, 60) + .05 * noise2(21, 14)
    d = (d + np.maximum(wash, 0)) * inside

    # cliff shading: the light comes from upper left
    gx = np.gradient(2.2 * noise2(30, 16) + noise2(31, 6), axis=1)
    light = np.clip(.5 + 2.8 * gx, 0, 1.3)
    face = faces()
    d *= (.55 + .75 * light) * (1 + .48 * face[None])

    # the cuts, thousands of them, following each face down and outward, denser on the shaded side
    cuts, c = a8()
    for _ in range(5600):
        px = r.uniform(0, W)
        side = float(face[int(px)])
        if r.random() > .62 + .38 * side:
            continue
        t = top[int(px)]
        py = t + (r.random() ** 1.5) * max(0, H - t) * .92
        ang = math.pi / 2 - side * r.uniform(.05, .62) + r.normal(0, .2)
        wedge(c, px, py, ang, r.uniform(30, 170), r.uniform(2, 11), r.uniform(.18, .62))
    cuts = gaussian_filter(a8_array(cuts), .6)
    grain = np.clip(.72 + .34 * fibre + .22 * noise2(33, .8), 0, 1)
    d = d + 1.05 * cuts * grain * inside

    # dark crisp line along the skyline, moss dots and tiny pines for scale
    mark, c = a8()
    for px in range(0, W, 3):
        t = float(top[px])
        if r.random() < .5:
            wedge(c, px, t - 1, math.pi / 2 + r.uniform(-.4, .4), r.uniform(10, 58), r.uniform(1.5, 5.5), r.uniform(.45, .95))
    # moss: dabbed in clusters high on the rock, each dab its own shape
    for _ in range(300):
        px = r.uniform(0, W)
        t = top[int(px)] + (r.random() ** 2.4) * 280
        for _ in range(r.integers(2, 8)):
            blot(c, px + r.normal(0, 10), t + r.normal(0, 5), r.uniform(.8, 2.9), r.uniform(-.5, .5), r.uniform(1, 1.8),
                 (r.uniform(.05, .22, 4), r.uniform(0, 6.283, 4)), r.uniform(.45, .95))
    d = d + 1.1 * a8_array(mark) * (.86 + .14 * grain)
    d *= mist ** .9
    return gaussian_filter(d, .55).astype(np.float32)


@lru_cache(1)
def far():
    """Peaks behind, in pale wash; they go when the mist comes up."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.zeros((H, W), np.float32)
    for i, (lift, tone, seed) in enumerate(((.78, .14, 61), (.62, .24, 62))):
        t = H * (1 - lift) + (ridged1(seed, 60, 5) - 1.4) * 42 + noise1(seed + 9, 200) * 120
        gap = y - t[None]
        layer = np.where(gap > 0, tone * np.exp(-gap / 230), 0) * (1 + .22 * noise2(seed, 22))
        out = np.maximum(out, layer)
    fade = np.clip((H * .92 - y) / (H * .36), 0, 1)
    return (out * fade).astype(np.float32)


@lru_cache(1)
def letterform():
    """山 in Weibei, the carved northern hand: the heads, the waists and the swoop of the foot come from here."""
    s, c = a8()
    c.select_font_face("Weibei SC", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    c.set_font_size(GLYPH_SIZE)
    e = c.text_extents("山")
    c.move_to(W / 2 - e.width / 2 - e.x_bearing, GLYPH_TOP - e.y_bearing)
    c.text_path("山")
    c.set_source_rgba(0, 0, 0, 1)
    c.fill()
    # the type's ends read as sawn off: redraw each head as the brush lands it, from the high point on the left, rolling down
    # over a dome to a shoulder on the right, and pinching in under the shoulder where the body begins
    row = a8_array(s)
    for x0, x1, T, split in HEADS:
        xs = np.flatnonzero(row[split, x0:x1] > .5) + x0
        xl, xr = float(xs[0]), float(xs[-1])
        w, h = xr - xl, split - T
        c.set_operator(cairo.OPERATOR_CLEAR)
        c.rectangle(x0 - 40, T - 60, x1 - x0 + 80, split - T + 60)
        c.fill()
        c.set_operator(cairo.OPERATOR_OVER)
        c.move_to(xl, split)
        c.curve_to(xl + .02 * w, split - .25 * h, xl + .07 * w, T + .62 * h, xl + .03 * w, T + .4 * h)
        c.curve_to(xl - .01 * w, T + .24 * h, xl + .02 * w, T + .06 * h, xl + .12 * w, T)
        c.curve_to(xl + .34 * w, T - .03 * h, xl + .74 * w, T + .06 * h, xr + .12 * w, T + .24 * h)
        c.curve_to(xr + .24 * w, T + .33 * h, xr + .2 * w, T + .46 * h, xr + .02 * w, T + .6 * h)
        c.curve_to(xr - .07 * w, T + .7 * h, xr - .05 * w, split - .12 * h, xr, split)
        c.close_path()
        c.fill()
    # the outline is where the brush was aimed; hand and paper move it
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = 1.6 * noise2(71, 30) + .7 * noise2(72, 8), 1.6 * noise2(73, 30) + .7 * noise2(74, 8)
    return map_coordinates(gaussian_filter(a8_array(s), 2.6), (y + dy, x + dx), order=1).astype(np.float32)


@lru_cache(1)
def brush():
    """The written character: ink density, the moment each pixel is written, and how wet the brush was there."""
    SS = 2
    acc = np.zeros((H * SS, W * SS), np.float32)
    when = np.zeros_like(acc)
    wet = np.zeros_like(acc)
    sheen = np.zeros_like(acc)
    r = np.random.default_rng(21)
    streak = np.random.default_rng(22)
    for t0, t1, ctrl in STROKES:
        ctrl = np.array(ctrl, np.float64)
        chord = np.concatenate(((0,), np.cumsum(np.hypot(*np.diff(ctrl[:, :2], axis=0).T))))
        L = chord[-1]
        s = np.arange(0, L, .55)
        px, py = CubicSpline(chord, ctrl[:, 0])(s), CubicSpline(chord, ctrl[:, 1])(s)
        width = PchipInterpolator(chord, ctrl[:, 2])(s)
        load = PchipInterpolator(chord, ctrl[:, 3])(s)
        tx, ty = gaussian_filter1d(np.gradient(px), 22), gaussian_filter1d(np.gradient(py), 22)
        k = np.hypot(tx, ty)
        nx, ny = -ty / k, tx / k
        hairs = int(width.max() * 5.2)
        u = np.sort(r.uniform(-1, 1, hairs))
        clump = gaussian_filter1d(r.standard_normal(hairs + 120), 10)[60:60 + hairs]
        clump = clump / (clump.std() + 1e-6)
        lean = r.uniform(.5, 1.0)
        # flying white comes in patches: bundles of hairs give out together, some for a long way and some for a moment
        rows, knots = int(width.max() / 1.6), np.arange(0, L + 8, 8)
        long_run = norm(gaussian_filter(streak.standard_normal((rows, knots.size)), (1.0, 12)))
        short_run = norm(gaussian_filter(streak.standard_normal((rows, knots.size)), (.7, 3)))
        dry = np.clip((s / L - .66) / .2, 0, 1) * np.clip((.7 - load) / .3, 0, 1)
        for i in range(hairs):
            ui = u[i]
            head = 30 * ((1 - ui * lean) / 2) ** 1.3 + r.uniform(0, 5)
            tail = L - (46 * (.3 + .7 * abs(ui)) ** 1.3) * r.uniform(.2, 1.5)
            keep = (s > head) & (s < tail)
            phase = r.uniform(0, 6.283, 3)
            wander = (1.1 * np.sin(s / 170 + phase[0] + clump[i]) + .7 * np.sin(s / 47 + phase[1]) + .35 * np.sin(s / 13 + phase[2]))
            edge = 1 + .05 * np.sin(s / 90 + 7 * ui + phase[0])
            jit = r.uniform(-.6, .6, s.size)                  # no two hairs sample the grid alike, or the ink shows a weave
            x = px + nx * (width / 2 * ui * edge + wander) + tx / k * jit
            y = py + ny * (width / 2 * ui * edge + wander) + ty / k * jit
            along = np.interp(s, np.arange(0, L + 40, 40), r.standard_normal(int(L / 40) + 2))
            fine = np.interp(s, np.arange(0, L + 9, 9), r.standard_normal(int(L / 9) + 2))
            row = int((ui + 1) / 2 * (rows - 1))
            run = .26 * np.interp(s, knots, long_run[row]) + .16 * np.interp(s, knots, short_run[row])
            ink = load + .20 * np.tanh(clump[i]) + .10 * along + .05 * fine + .05 * r.normal() + dry * run
            # the hairs only run dry in the last third of the stroke
            a = np.maximum(np.clip((ink - .16) / .1, 0, 1), np.clip((.72 - s / L) / .06, 0, 1)) * keep
            ix = np.clip((x * SS).astype(int), 0, W * SS - 1)
            iy = np.clip((y * SS).astype(int), 0, H * SS - 1)
            np.add.at(acc, (iy, ix), a)
            np.add.at(when, (iy, ix), a * (t0 + (t1 - t0) * (s / L) ** .85))
            np.add.at(wet, (iy, ix), a * load ** 2)
            np.add.at(sheen, (iy, ix), a * (.5 + .34 * np.tanh(clump[i]) + r.uniform(-.16, .16)))
    soft = gaussian_filter(acc, 1.05)
    around = gaussian_filter(acc, 2.2)
    arrive = np.where(around > .02, gaussian_filter(when, 2.2) / np.maximum(around, 1e-3), 99.0)
    wetness = gaussian_filter(wet, 24) / np.maximum(gaussian_filter(acc, 24), 1e-4)

    def small(a):
        return a[::2, ::2] * .25 + a[1::2, ::2] * .25 + a[::2, 1::2] * .25 + a[1::2, 1::2] * .25
    d = small(1.62 * (1 - np.exp(-soft * 2.8))).astype(np.float32)
    _, fibre, _ = stock()
    # the edge of the form, and only the edge: smooth where the brush was wet, chewed by the tooth of the paper where it was dry
    m = letterform()
    dry = np.clip(1.25 - small(wetness), .35, 1.1)
    d = d * np.clip((m - .5 + 4 * m * (1 - m) * dry * (.2 * noise2(75, 1.2) + .16 * noise2(76, 3.2) + .1 * fibre)) * 5, 0, 1)
    # where the hairs are thin the ink only catches on the tooth of the paper
    body = gaussian_filter(d, 4)
    tooth = noise2(81, .8) + .35 * fibre
    d = d * np.clip(.45 + 1.7 * body + .55 * tooth, 0, 1)
    # a little pooling inside the wet heads, and the lie of the hairs in the wet ink: dark on dark, never a white scratch
    lie = small(gaussian_filter(sheen, .9) / np.maximum(gaussian_filter(acc, .9), 1e-3)).astype(np.float32)
    d = d * (1 + .10 * noise2(82, 26) + .05 * noise2(83, 7)) * (1 + .5 * (np.clip(lie, 0, 1) - .5))
    # what the hairs missed is wiped by the nearest hair, so no gathered ink outlives the brush
    arrive = small(arrive)
    arrive = arrive[tuple(distance_transform_edt(arrive > 4.5, return_distances=False, return_indices=True))]
    return d.astype(np.float32), arrive.astype(np.float32), small(wetness).astype(np.float32)


@lru_cache(1)
def bleed():
    """How far ink will creep out of the character: well out of a wet head, hardly at all from a dry tail, never evenly."""
    written, _, wetness = brush()
    _, fibre, _ = stock()
    dist = distance_transform_edt(written < .4)
    patches = np.clip(.25 + .85 * noise2(41, 30), 0, 1.5)
    reach = (1 + 22 * np.clip(wetness, 0, 1) ** 1.3 * patches) * np.clip(1 + .4 * noise2(43, 7) + .26 * noise2(42, 2.2) + .22 * fibre, .1, 2.6)
    tone = .15 + .03 * noise2(44, 9) + .02 * noise2(45, 2.5)
    return dist.astype(np.float32), reach.astype(np.float32), tone.astype(np.float32)


@lru_cache(1)
def glyph():
    """Top and bottom of the character in each screen column."""
    d = brush()[0]
    a = gaussian_filter(d, 5) > .3
    has = a.any(0)
    top = np.where(has, a.argmax(0), 0).astype(np.float32)
    bot = np.where(has, H - 1 - a[::-1].argmax(0), 0).astype(np.float32)
    xs = np.flatnonzero(has)
    return top, bot, int(xs[0]), int(xs[-1])


@lru_cache(1)
def transport():
    """Column u of the painting -> column Xf(u) of the glyph, matching cumulative mass."""
    m_top = ridge()
    g_top, g_bot, xa, xb = glyph()
    source = np.maximum(H * .80 - m_top, 12) ** 1.7
    target = np.maximum(g_bot - g_top, 0) + .5
    target[:xa] = 0
    target[xb + 1:] = 0
    cs = np.cumsum(source) / source.sum()
    ct = np.cumsum(target) / target.sum()
    keep = np.concatenate(((True,), np.diff(ct) > 0))
    xf = np.interp(cs, ct[keep], np.arange(W, dtype=np.float64)[keep])
    return np.clip(gaussian_filter1d(xf, 2.0), xa, xb)


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)


def morph_amount(t):
    """A column's progress: slow pull, snap, overshoot, settle. Tower tops move first."""
    u = np.linspace(0, 1, W)
    delay = .34 * np.abs(u - .515) / .5
    x = (t - .62 - delay) / 2.1
    e = ease(x)
    over = .052 * np.sin(np.clip(x - 1, 0, 2) * 7.5) * np.exp(-np.clip(x - 1, 0, 9) * 3.2)
    return e + over


@lru_cache(1)
def soft_painting():
    """The same ink running wet: what the columns carry while they are on the move."""
    return gaussian_filter(painting(), (5, 8)).astype(np.float32)


@lru_cache(1)
def currents():
    """Slow eddies that push the ink about as it runs: two scales for each axis. They tile, because each frame rolls them."""
    return tuple(noise2(seed, sigma, mode="wrap") for seed, sigma in ((51, 80), (52, 80), (53, 24), (54, 24), (55, 42), (56, 12)))


@lru_cache(1)
def seal():
    """A Han seal: straight strokes filling the square, cut in reverse into the stone, its border chipped by use."""
    N = 400
    s = cairo.ImageSurface(cairo.FORMAT_A8, N, N)
    c = cairo.Context(s)
    c.set_source_rgba(0, 0, 0, 1)
    c.rectangle(16, 16, N - 32, N - 32)
    c.fill()
    c.set_operator(cairo.OPERATOR_DEST_OUT)
    # each stroke four corners cut by eye: square ends, no two edges quite parallel
    for poly in (((174, 52), (226, 50), (229, 300), (172, 302)),                 # the middle, tallest
                 ((64, 112), (113, 108), (117, 300), (66, 303)),                 # left
                 ((289, 110), (337, 113), (335, 302), (284, 299)),               # right
                 ((64, 286), (337, 282), (339, 342), (62, 346))):                # the foot they all stand on
        c.move_to(*poly[0])
        for q in poly[1:]:
            c.line_to(*q)
        c.close_path()
        c.fill()
    s.flush()
    a = np.ndarray((N, N), np.uint8, s.get_data(), strides=(s.get_stride(), 1)).astype(np.float32) / 255
    r = np.random.default_rng(9)

    def n(sigma):
        return norm(gaussian_filter(r.standard_normal((N, N)), sigma)).astype(np.float32)
    y, x = np.mgrid[0:N, 0:N].astype(np.float32)
    a = map_coordinates(gaussian_filter(a, 1.6), (y + 2.4 * n(26) + 1.2 * n(7), x + 2.4 * n(26) + 1.2 * n(7)), order=1)
    # stone chips, paste that did not take
    a = np.clip((a - .5 + .12 * n(2.2) + .08 * n(6)) * 5, 0, 1)
    a *= np.clip(1.9 + .34 * n(1.4) + .2 * n(9) + .22 * n(40), 0, 1)
    return gaussian_filter(a, 1.0).astype(np.float32)


@lru_cache(1)
def spatter():
    """Flung off the brush in thin sprays: mostly fine flecks, some long drops, a rare real blot close to the stroke."""
    r = np.random.default_rng(13)
    out = []
    for t0, x0, y0, ang, spread, far_, n in ((2.76, 921, 868, .42, .13, 260, 13), (3.05, 1350, 790, -.42, .10, 400, 22), (3.22, 1262, 912, 1.0, .12, 160, 8)):
        for _ in range(n):
            a = ang + r.normal(0, spread)
            f = r.random() ** 1.2
            kind = r.random()
            if kind < .74:
                size, draw = .55 + r.random(), r.uniform(1, 1.5)
            elif kind < .95:
                size, draw = 1.5 + 1.6 * r.random(), r.uniform(1.4, 2.6)
            else:
                size, draw = 3.2 + 3.2 * r.random() * (1 - f), r.uniform(1.05, 1.4)
            shape = (r.uniform(.04, .1 + size * .025, 4), r.uniform(0, 6.283, 4))
            out.append((t0 + r.uniform(0, .05), x0 + r.normal(0, 7), y0 + r.normal(0, 7), math.cos(a), math.sin(a), 30 + far_ * f, size, draw, shape))
    return out


def blot(c, x, y, size, ang, stretch, shape, a=1.0):
    """A drop as it lands: blunt at the front where it hit, drawn to a tail behind, never a circle."""
    amps, phases = shape
    c.save()
    c.translate(x, y)
    c.rotate(ang)
    for i in range(36):
        th = i / 36 * 6.283
        rad = size * (1 + sum(amps[k] * math.sin((k + 2) * th + phases[k]) for k in range(4)))
        back = max(0, -math.cos(th))
        c.line_to(rad * math.cos(th) * (1 + 2 * (stretch - 1) * back ** 3), rad * math.sin(th) * (1 - .3 * min(1, stretch - 1) * back))
    c.close_path()
    c.restore()
    c.set_source_rgba(0, 0, 0, a)
    c.fill()


def morph(t):
    t = float(t)
    base, fibre, cloud = stock()
    P, soft_p = painting(), soft_painting()
    m_top = ridge()
    g_top, g_bot, xa, xb = glyph()
    written, arrive, _ = brush()
    xf = transport()
    e = morph_amount(t)

    # the camera eases back over the first beat
    z = 1 + .13 * (1 - ease(t / 1.0))
    uu = np.arange(W, dtype=np.float64)
    X = (1 - e) * uu + e * xf
    X = np.maximum.accumulate(X + uu * 1e-6)
    gt = np.interp(xf, uu, g_top)
    gb = np.interp(xf, uu, g_bot)
    bot_m = np.full(W, H + 4.0)
    skyline = gaussian_filter1d(m_top, 2.0)
    TOP = gaussian_filter1d((1 - e) * m_top + e * gt, 2.0)
    BOT = gaussian_filter1d((1 - e) * bot_m + e * gb, 4.0)
    yg, xg = np.mgrid[0:H, 0:W].astype(np.float32)
    xz = W / 2 + (xg - W / 2) / z
    yz = H / 2 + (yg - H / 2) / z

    # on the move the ink is a liquid: eddies push it about, so nothing keeps a straight edge
    moving = np.sin(np.pi * np.clip(np.interp(xz, X, e), 0, 1)).astype(np.float32)
    flow = int(t * 46)
    fx, fy, gx, gy, drip, fine = currents()
    xw = xz + moving * (58 * np.roll(fx, flow, 0) + 15 * np.roll(gx, -flow, 1))
    yw = yz + moving * (44 * np.roll(fy, flow // 2, 1) + 13 * np.roll(gy, flow, 0))

    u = np.interp(xw, X, uu).astype(np.float32)
    top_px = np.interp(xw, X, TOP).astype(np.float32)
    bot_px = np.interp(xw, X, BOT).astype(np.float32)
    v = (yw - top_px) / np.maximum(bot_px - top_px, .5)
    span = np.interp(u, uu, bot_m - skyline).astype(np.float32)
    y0 = np.interp(u, uu, skyline).astype(np.float32) + v * span
    coords = (np.clip(y0, 0, H - 1), np.clip(u, 0, W - 1))
    sharp = map_coordinates(P, coords, order=1)
    if moving.max() > .02:
        ink = sharp + (map_coordinates(soft_p, coords, order=1) - sharp) * np.clip(moving * 1.3, 0, .9)
    else:
        ink = sharp

    # ink is conserved: squeeze a column and it gets darker, smoothly, with no stripes down it
    du = gaussian_filter1d(np.gradient(np.interp(uu, X, uu)), 16)
    squeeze = np.clip(du, .2, 60).astype(np.float32)
    height = np.maximum(BOT - TOP, 1).astype(np.float32)
    vertical = np.clip(gaussian_filter1d(np.interp(uu, X, bot_m - m_top) / height, 16), .3, 40).astype(np.float32)
    E = np.interp(xw, X, np.clip(e, 0, 1)).astype(np.float32)
    # as each column lands it stops being a column: unevenly, patch by patch, its ink comes to rest in the true form
    settled = np.clip((E - .66 + .07 * cloud + .04 * fine) / .24, 0, 1)
    settled = settled * settled * (3 - 2 * settled)
    density = ink * (1 + ((squeeze * vertical)[None] ** .5 - 1) * E)
    # gathered ink is loaded ink: the pale mist at the foot of the painting does not survive the move
    full = E ** 2.2
    density = density * (1 - full) + (density * .25 * (1 - settled) + 1.2 + .3 * settled + .22 * cloud + .1 * fine) * full

    # edges run: crisp at rest, feathering into the wet paper mid-move, the foot breaking up in tendrils
    grain = .35 * fibre * (1 - moving)
    edge_top = np.clip((yw - top_px) / (1.1 + 9 * moving) + grain, 0, 1)
    tendril = 70 * drip + 34 * fine
    edge_bot = np.clip((bot_px - yw + moving * tendril) / (1.2 + 120 * moving) + grain, 0, 1)
    form = np.clip((map_coordinates(letterform(), (np.clip(yw, 0, H - 1), np.clip(xw, 0, W - 1)), order=1) - .35) * 3, 0, 1)
    density = density * (edge_top * edge_bot * (1 - settled) + form * settled)

    # columns squeezed out of the ends dissolve, wide mid-move and crisp once they have landed
    ends = float(np.clip(e[0], 0, 1))
    reach = 1 + 59 * (1 - ends) + 110 * math.sin(math.pi * ends) ** 2
    wisp = (26 * (1 - ends) + 50 * math.sin(math.pi * ends)) * cloud + 16 * (1 - ends) * drip
    density = density * np.clip((xw - X[0] + wisp) / reach, 0, 1) * np.clip((X[-1] - xw + wisp) / reach, 0, 1)

    # far peaks: the mist comes up and takes them
    rise = float(ease((t - .25) / 1.4))
    if rise < 1:
        d_far = map_coordinates(far(), (np.clip(yz + rise * 90, 0, H - 1), np.clip(xz, 0, W - 1)), order=1)
        density = np.maximum(density, d_far * (1 - rise))

    # the brush comes through in stroke order and writes the gathered ink down
    if t > 2.3:
        done = np.clip((t - arrive) / .11, 0, 1)
        done = done * done * (3 - 2 * done)
        density = density * (1 - done) + written * done

    # the seep: ink creeps out along the fibres, only where the brush was wet, thinning as it goes
    seep = float(np.clip((t - 3.0) / 1.9, 0, 1))
    if seep > 0:
        dist, reach, tone = bleed()
        front = np.maximum(reach * seep ** .55, .01)
        density = np.maximum(density, np.clip(1 - dist / front, 0, 1) ** 2 * tone)

    # drops thrown off the strokes
    if t > 2.74:
        s, c = a8()
        for t0, x0_, y0_, dx, dy, reach_, size, draw, shape in spatter():
            b = t - t0
            if b <= 0:
                continue
            travel = reach_ * (1 - math.exp(-b * 11))
            px, py = x0_ + dx * travel, y0_ + dy * travel
            blot(c, px, py, size, math.atan2(dy, dx), draw + 3 * math.exp(-b * 8), shape)
        drops = gaussian_filter(a8_array(s), .5)
        density = np.maximum(density, drops * 1.25 * np.clip(.86 + .24 * fibre, 0, 1))

    img = base * np.exp(-np.maximum(density, 0)[..., None] * ABSORB[None, None])

    # the seal lands last
    ts = 3.98
    if t > ts:
        b = t - ts
        k = 1 + .55 * math.exp(-b * 22)
        size = int(92 * k)
        q = np.linspace(0, 399, size)
        patch = map_coordinates(seal(), np.meshgrid(q, q, indexing="ij"), order=1).astype(np.float32)
        cx, cy = 1486, 884
        y1, x1 = cy - size // 2, cx - size // 2
        opacity = min(1.0, b / .05)
        reg = img[y1:y1 + size, x1:x1 + size]
        m = (patch * opacity * np.clip(.86 + .12 * fibre[y1:y1 + size, x1:x1 + size], 0, 1))[..., None]
        img[y1:y1 + size, x1:x1 + size] = reg * (1 - m) + CINNABAR * m
    return surface(np.clip(img, 0, 1).astype(np.float32))
