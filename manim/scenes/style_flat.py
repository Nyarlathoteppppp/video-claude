"""A single flat-vector launch shot. Six global inks, filled paths only.

Smoke uses closed-form advected lobe families in fixed depth layers, each
rendered tone-major: shadow silhouette, middle mass, then a smaller lit mass.
No simulation state, images, gradients, outlines, or transparent layers.
"""
import math
from functools import lru_cache

import cairo
import numpy as np
from styles_common import W, H, canvas, hx

TAU = 2.0 * math.pi
DUSK = tuple(map(hx, ("#10172C", "#61425E", "#B94B50", "#EF814B", "#FFD260", "#FFF9F0")))
AIR = tuple(map(hx, ("#071530", "#203878", "#2858CD", "#468CEE", "#A7CBEA", "#F8FCFF")))
SPACE = tuple(map(hx, ("#060B19", "#17284C", "#2459BE", "#32A9BB", "#FF8C4A", "#F8FCFF")))


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def mix(a, b, f):
    return tuple(x + (y - x) * f for x, y in zip(a, b))


def inks(t):
    a, b, f = (DUSK, AIR, ease((t - 0.7) / .28)) if t < 2.6 else (AIR, SPACE, ease((t - 2.6) / 0.6))
    return tuple(mix(ca, cb, f) for ca, cb in zip(a, b))


def steel(t, p, dark_from):
    """Hardware ink. p[0] only while blue sky is behind the part; on a dark sky p[0] would make it vanish."""
    day = ease((t - .62) / .2) * (1 - ease((t - dark_from) / .18))
    return mix(p[2], p[0], day)


def poly(c, pts, color):
    c.new_path()
    c.move_to(*pts[0])
    for p in pts[1:]:
        c.line_to(*p)
    c.close_path()
    c.set_source_rgb(*color)
    c.fill()


def rect(c, x, y, w, h, color):
    c.new_path()
    c.rectangle(x, y, w, h)
    c.set_source_rgb(*color)
    c.fill()


def disc(c, x, y, r, color):
    c.new_path()
    c.arc(x, y, r, 0, TAU)
    c.set_source_rgb(*color)
    c.fill()


def discs(c, marks, color):
    c.new_path()
    for x, y, r in marks:
        if r > 0.15 and -r < x < W + r and -r < y < H + r:
            c.new_sub_path()
            c.arc(x, y, r, 0, TAU)
    c.set_source_rgb(*color)
    c.fill()


def contour(c, pts):
    """Closed Catmull-Rom contour as cubic vector segments, added to the current path."""
    c.move_to(*pts[0])
    count = len(pts)
    for j in range(count):
        a, b, d, e = pts[(j - 1) % count], pts[j], pts[(j + 1) % count], pts[(j + 2) % count]
        c.curve_to(b[0] + (d[0] - a[0]) / 6, b[1] + (d[1] - a[1]) / 6,
                   d[0] - (e[0] - b[0]) / 6, d[1] - (e[1] - b[1]) / 6, d[0], d[1])
    c.close_path()


def curved(c, pts, color):
    c.new_path()
    contour(c, pts)
    c.set_source_rgb(*color)
    c.fill()


@lru_cache(maxsize=64)
def cloud_profile(seed):
    rng = np.random.default_rng(4500 + seed)
    u = np.linspace(0, 1, 92)
    h = np.full_like(u, .24)
    for _ in range(17):
        centre = rng.uniform(-.08, 1.08)
        width = rng.uniform(.019, .12)
        h += rng.uniform(.12, .38) * np.exp(-((u - centre) / width) ** 2)
    return tuple(zip(u.tolist(), h.tolist()))


def bank(c, x, y, width, height, seed, color):
    profile = cloud_profile(seed)
    # Both ends taper to a point: a bank is a lens of cloud, never a slab cut off with a knife.
    taper = [math.sin(math.pi * min(1.0, max(0.0, u))) ** .5 for u, _ in profile]
    top = [(x + u * width, y - height * h * e) for (u, h), e in zip(profile, taper)]
    bottom = [(x + u * width, y + height * (.36 + .18 * h) * e) for (u, h), e in zip(profile, taper)]
    curved(c, top + bottom[::-1], color)


@lru_cache(maxsize=1)
def bank_layout():
    """Every bank owns its x, width, height and profile; no two visible tips may line up in a column."""
    rng = np.random.default_rng(4471)
    ends, banks, seed = [], [], 20

    def place(low, high, wmin, wmax):
        for _ in range(400):
            x0, width = float(rng.uniform(low, high)), float(rng.uniform(wmin, wmax))
            tips = [e for e in (x0, x0 + width) if -40 < e < W + 40]
            if all(abs(e - o) > 95 for e in tips for o in ends):
                break
        ends.extend(tips)
        return x0, width

    for speed, offsets, spans, hmin, hmax, tone, cap in (
            (.28, (-430, -1030, -1570), ((-420, -40, 760, 1280), (1130, 1560, 760, 1200)), 62, 96, 0, 1),
            (.62, (-1000, -2650, -3520), ((-520, -140, 900, 1500), (1150, 1620, 700, 1200)), 84, 124, 2, 3)):
        for offset in offsets:
            for span in spans:
                x0, width = place(*span)
                y = offset + float(rng.uniform(-170, 170))
                height = float(rng.uniform(hmin, hmax))
                banks.append((speed, y, x0, width, height, seed, tone))
                seed += 1
                # The lit cap is its own shape, inset by a different amount on every bank.
                inset, scale = float(rng.uniform(30, 190)), float(rng.uniform(.56, .84))
                banks.append((speed, y + float(rng.uniform(-30, 26)) * speed * 2, x0 + inset, width * scale,
                              height * float(rng.uniform(.48, .64)), seed, cap))
                seed += 1
    return tuple(banks)


def air_banks(c, t, p):
    if not .58 < t < 3.16:
        return
    g = ground(t)
    # While the sky is p[2], a p[2] cap would bite a sky-coloured hole out of its own cloud.
    night = ease((t - 1.7) / .4)
    tones = (p[1], mix(p[3], p[2], night), mix(p[1], p[2], night), mix(p[4], p[3], night))
    # Fixed altitude families: the near banks fall 2.2x faster than far banks.
    # Positions are not wrapped or recycled; each bank passes only once.
    for speed, offset, x0, width, height, seed, tone in bank_layout():
        y = speed * g + offset
        if -250 < y < H + 200:
            bank(c, x0, y, width, height, seed, tones[tone])
    # Thin, hard atmospheric strata share the horizon curvature.
    for offset, thickness in ((-1250, 13), (-3650, 8), (-4250, 5)):
        y = .83 * g + offset
        if -40 < y < H + 30:
            pts = [(x, y + (x - 960) ** 2 / 32000) for x in range(-20, 1941, 40)]
            pts += [(x, y + thickness + (x - 960) ** 2 / 32000) for x in range(1940, -21, -40)]
            poly(c, pts, p[3])


def front_banks(c, t, p):
    """Two banks nearer than the rocket: it flies behind them, so the cloud layer has real depth."""
    if not 1.1 < t < 2.75:
        return
    g = ground(t)
    for speed, offset, x0, width, height, seed in ((1.3, -3010, 560, 1120, 118, 44), (1.05, -5900, 800, 1320, 96, 45)):
        y = speed * g + offset
        if -260 < y < H + 260:
            bank(c, x0, y + height * .34, width, height, seed, p[1])
            bank(c, x0 + 26, y, width * .97, height, seed, p[3])
            bank(c, x0 + 120, y - height * .26, width * .71, height * .62, seed + 1, p[4])


def ground(t):
    return 935.0 + 950.0 * t * t


def nose_y(t):
    # Thrust outruns the tracking camera, then tracking catches up after max-Q.
    # Both screen motion and exhaust birth positions use this same trajectory.
    if t <= 3.0:
        lag = 96.0 * (ease((t - .9) / .9) - ease((t - 1.8) / .8))
        return 125.0 - lag
    return 125.0 + 25.0 * ease(t - 3.0)


def angle(t):
    return math.radians(5.0 + 43.0 * ease((t - 3.45) / 1.4))


def pull(t):
    # Portrait of the rocket first; the camera backs off for the plume, then closes on the upper stage as it flies clear.
    return (1 - .38 * ease((t - .75) / 1.75)) * (1 + .62 * ease((t - 3.4) / 1.6))


def pivot(t):
    # After staging the rocket flies out across the sunburst.
    f = ease((t - 3.4) / 1.6)
    return 1125 + 190 * f, 340 - 60 * f


def vehicle(c, t):
    c.save()
    z = pull(t)
    c.translate(*pivot(t))
    c.scale(z, z)
    c.translate(0, nose_y(t) - 340)
    c.rotate(angle(t))
    c.scale(0.9, 0.9)


def emitter(t, end=900.0):
    """Screen position of a point on the rocket's axis, `end` below the nose."""
    z = pull(t)
    px, py = pivot(t)
    return (px - z * 0.9 * end * math.sin(angle(t)),
            py + z * (nose_y(t) - 340 + 0.9 * end * math.cos(angle(t))))


@lru_cache(maxsize=1)
def smoke_seeds():
    rng = np.random.default_rng(81379)
    parents = []
    children = []
    # Pre-ignited trench: large masses are already spreading at the first frame.
    for i in range(64):
        parents.append((float(rng.uniform(-2.7, -0.12)), -1.0 if i % 2 else 1.0,
                        float(rng.uniform(0.7, 1.35)), float(rng.uniform(0.72, 1.28)),
                        float(rng.uniform(0, TAU))))
        for _ in range(35):
            children.append((i, float(rng.uniform(0, TAU)), float(rng.uniform(0.68, 1.08)),
                             float(rng.uniform(0.055, 0.22))))
    flight = []
    for _ in range(1600):
        flight.append((float(rng.uniform(-0.1, 3.12)), float(rng.uniform(-1, 1)),
                       float(rng.uniform(0.7, 1.4)), float(rng.uniform(0, TAU))))
    # Fresh lobes hugging the jet, the hottest smoke in the frame.
    for i in range(len(parents), len(parents) + 14):
        parents.append((float(rng.uniform(-.34, -.02)), -1.0 if i % 2 else 1.0, float(rng.uniform(.12, .42)),
                        float(rng.uniform(.5, .82)), float(rng.uniform(0, TAU))))
        for _ in range(16):
            children.append((i, float(rng.uniform(0, TAU)), float(rng.uniform(.8, 1.04)), float(rng.uniform(.12, .26))))
    return tuple(parents), tuple(children), tuple(flight)


@lru_cache(maxsize=1)
def scenery():
    rng = np.random.default_rng(1184)
    stars = tuple((float(rng.uniform(30, 1890)), float(rng.uniform(30, 830)),
                   float(rng.uniform(0.8, 2.2)), float(rng.uniform(0, 1))) for _ in range(92))
    clouds = tuple((float(x), float(rng.uniform(-55, 55)), float(rng.uniform(65, 145)))
                   for x in np.linspace(-200, 2140, 65))
    chips = tuple((float(rng.uniform(0.05, 1.0)), float(rng.uniform(150, 780)),
                  float(rng.choice((-1, 1))), float(rng.uniform(40, 170)),
                  float(rng.uniform(3, 8))) for _ in range(78))
    return stars, clouds, chips


def sky(c, t, p):
    # Staging: for a few frames the whole sky takes the flash, then falls back to black.
    c.set_source_rgb(*mix(p[0], p[2], .9 * (1 - ease((t - 3.18) / .16)) if t > 3.18 else 0))
    c.paint()
    # Huge circles are shallow, horizontal atmospheric shells, never diagonals.
    reveal = ease((t - .45) / .75)
    blue_edge = 1250 * (1 - reveal) + (-1050 + 380 * t * t) * reveal
    radius = 16000.0
    disc(c, 960, radius + blue_edge, radius, p[1])
    disc(c, 960, radius + 60 + blue_edge, radius, p[2])
    # No p[1] shell on the horizon: the tower is p[1] and must stay a silhouette against it.
    horizon = 610 + 700 * t * t
    disc(c, 960, radius + 62 + horizon, radius, p[2])
    disc(c, 960, radius + 112 + horizon, radius, p[3])
    disc(c, 960, radius + 148 + horizon, radius, p[4])
    # Stars emerge as individual solid marks; no transparent sky cross-fade.
    if t > 2.8:
        for x, y, r, gate in scenery()[0]:
            rr = r * ease((t - 2.8 - gate * 0.3) / 0.35)
            if rr > 0.18:
                rect(c, x, y, rr, rr, p[5] if gate < 0.45 else p[3])


@lru_cache(maxsize=1)
def streak_seeds():
    rng = np.random.default_rng(66)
    return tuple((float(rng.uniform(0, W)), float(rng.uniform(-5200, -200)), float(rng.uniform(60, 330)),
                  float(rng.uniform(1.5, 5)), int(rng.integers(0, 3))) for _ in range(80))


def streaks(c, t, p):
    """Air tearing past: hard vertical slivers that pass once, at the speed of the falling ground."""
    if not .8 < t < 3.05:
        return
    g = ground(t)
    for x, offset, length, width, kind in streak_seeds():
        y = .9 * g + offset
        if -length < y < H + 1.5 * length:
            # A long sliver closing to a point at both ends; bottom-heavy ones read as raindrops.
            s = np.linspace(0, 1, 13)
            w = .7 * width * np.sin(math.pi * s) ** .8
            yy = y - 1.5 * length + 1.5 * length * s
            poly(c, np.vstack((np.stack((x - w, yy), 1), np.stack((x + w, yy), 1)[::-1])).tolist(), p[(3, 3, 4)[kind]])


def pad(c, t, p):
    g = ground(t)
    if g - 900 > H:
        return
    # Ground and a separate tower: both stay at fixed world elevations.
    rect(c, 0, g - 12, W, 1300, p[0])
    for j in range(19):
        x = 35 + j * 106
        hh = 12 + (j * 23) % 43
        if not 940 < x < 1130:                                # nothing dark directly behind the engines
            rect(c, x, g - hh, 71, hh, p[0])
    left, right, top = 780, 914, g - 865
    steel = p[1]
    rect(c, left, top, 13, 865, steel)
    rect(c, right - 13, top, 13, 865, steel)
    rect(c, left - 15, top - 10, 164, 16, steel)
    for j in range(17):
        y = top + j * 50
        rect(c, left, y, right - left, 6, steel)
        poly(c, [(left + 8, y), (left + 16, y), (right - 8, y + 50), (right - 16, y + 50)], steel)
        poly(c, [(right - 8, y), (right - 16, y), (left + 8, y + 50), (left + 16, y + 50)], steel)
    # Elevator shaft and stair landings are silhouettes, not ornamental outlines.
    rect(c, left + 45, top, 22, 850, steel)
    for j in range(9):
        rect(c, left - 13, top + 35 + 90 * j, 43, 9, steel)
    retract = ease(t / 0.65)
    for height, reach in ((740, 195), (515, 174), (280, 152)):
        reach *= 1 - 0.78 * retract
        y = g - height
        poly(c, [(right, y), (right + reach, y + 6), (right + reach, y + 18), (right, y + 25)], steel)
        poly(c, [(right, y + 35), (right + reach * .86, y + 14),
                 (right + reach * .86, y + 21), (right, y + 43)], steel)
    rect(c, 995, g - 22, 164, 58, p[0])


def smoke(c, t, p):
    if t >= 3.18:
        return
    parents, children, flight = smoke_seeds()
    g = ground(t)
    # Depth comes from birth order, which never changes, so no lobe can pop between layers.
    rank = {i: n for n, i in enumerate(sorted(range(64), key=lambda i: parents[i][0]))}
    layers = [([], [], [], []) for _ in range(6)]
    centres = []
    # Closed-form drag + buoyancy + fixed smooth deflection field.
    for i, (birth, side, speed, rad, phase) in enumerate(parents):
        age = t - birth
        x = 1055 + side * (90 + 1550 * speed * (1 - math.exp(-0.7 * age)))
        y = g + 130 - 55 * age - 450 * t + 35 * math.sin(phase + age * 0.55)
        r = rad * (112 + 70 * min(age, 3.0) + 420 * t)
        sign = 1 if x < 1055 else -1
        base, mid, lit, under = layer = layers[rank[i] * 5 // 64 if i < 64 else 5]
        base.append((x, y, r))
        mid_at = (x + sign * r * .13, y - r * .11, r * .88)
        mid.append(mid_at)
        # Only some lobes catch the light; the rest stay middle tone and cut the lit mass into lumps.
        lit_at = (x + sign * r * .34, y - r * .3, r * .44) if i % 3 else None
        if lit_at:
            lit.append(lit_at)
            # Middle tone laid back over the lit disc's underside: what is left is a cap of light, not a round egg.
            under.append((lit_at[0] - sign * r * .03, lit_at[1] + r * .32, r * .5))
        centres.append((x, y, r, mid_at, lit_at, layer))
    for parent, a, distance, size in children:
        x, y, r, mid_at, lit_at, (base, mid, lit, under) = centres[parent]
        theta = a + 0.08 * t * math.sin(a * 3)
        cos, sin = math.cos(theta), math.sin(theta)
        # Never further out than its parent's edge, so a small lobe cannot float free as a loose dot.
        distance = min(distance, 1 + size * .35)
        # Tiny rim lobes only leave flecks of shadow when you zoom in.
        if r * size >= 15:
            base.append((x + cos * r * distance, y + sin * r * distance, r * size))
        # Bumps sit on the edge of their own tone's mass, so a bright bubble can never float alone.
        mid.append((mid_at[0] + cos * mid_at[2] * distance, mid_at[1] + sin * mid_at[2] * distance, r * size * .96))
        if lit_at and sin < -.35:
            lit.append((lit_at[0] + cos * lit_at[2] * distance, lit_at[1] + sin * lit_at[2] * distance, r * size * .62))
            # The cap's lower edge billows too.
            under.append((lit_at[0] + cos * r * .5 * distance, lit_at[1] + r * .32 + sin * r * .5 * distance, r * size * .55))
    # Upper plume trails earlier engine positions; all births are analytical.
    base, mid, lit, _ = layers[4]
    for birth, lateral, speed, phase in flight:
        age = t - birth
        if not 0 < age < 1.55:
            continue
        b = max(0.0, birth)
        ex, ey = emitter(b)
        fall = ground(t) - ground(b)
        spread = 18 + 100 * age ** 1.4
        x = ex + lateral * spread + 15 * math.sin(phase + age * 1.2)
        y = ey + fall + 520 * speed * age
        r = 12 + 45 * age * speed
        sign = 1 if x < 1055 else -1
        base.append((x, y, r))
        mid.append((x + sign * r * .17, y - r * .12, r * .86))
        if r >= 25:
            lit.append((x + sign * r * .32, y - r * .17, r * .62))
    # Once the sky turns blue p[2] IS the sky: as a smoke tone it would punch holes and strand the shadow as crescents.
    air = ease((t - .55) / .3)
    rim, shade = mix(p[2], p[1], air), mix(p[2], p[3], air)
    far = (p[1], shade, mix(p[3], p[4], air))
    close = (rim, p[3], p[4])
    jet = (rim, p[4], p[5])
    for j, (base, mid, lit, under) in enumerate(layers):
        dark, middle, light = far if j < 2 else jet if j == 5 else close
        discs(c, base, dark)
        discs(c, mid, middle)
        discs(c, lit, light)
        discs(c, under, middle)


def cloud_deck(c, t, p):
    y = ground(t) - 3600
    if y < -350 or y > H + 350:
        return
    gap = 80 + 140 * ease((t - 1.42) / .6)
    cx = 1090
    c.save()
    c.new_path()
    c.rectangle(0, 0, W, H)
    contour(c, [(cx - gap * 1.2, y - 300), (cx + gap * 1.2, y - 300), (cx + gap * .62, y - 60),
                (cx + gap * .8, y + 90), (cx + gap * 1.6, y + 300), (cx - gap * 1.5, y + 300),
                (cx - gap * .75, y + 90), (cx - gap * .7, y - 60)])
    c.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    c.clip()
    c.set_fill_rule(cairo.FILL_RULE_WINDING)
    bank(c, -250, y + 20, 2450, 140, 12, p[2])
    bank(c, -250, y - 15, 2450, 106, 12, p[4])
    bank(c, -250, y - 44, 2450, 65, 12, p[5])
    c.restore()
    # The torn lips of the hole, thrown up and outward by the plume.
    for side in (-1, 1):
        marks = [(cx + side * (gap * (.8 + .12 * k) + 26 * k), y - 58 + 15 * k - 12 * (k % 2), 58 - 7 * k) for k in range(7)]
        discs(c, [(x, yy + 24, r * 1.04) for x, yy, r in marks], p[4])
        discs(c, marks, p[5])


def hull(c, top, bottom, width, p, pointed=False):
    half = width * .5
    c.save()
    c.new_path()
    c.move_to(-half, bottom)
    c.line_to(-half, top + (105 if pointed else 0))
    if pointed:
        c.curve_to(-half, top + 55, -10, top + 8, 0, top)
        c.curve_to(10, top + 8, half, top + 55, half, top + 105)
    else:
        c.line_to(half, top)
    c.line_to(half, bottom)
    c.close_path()
    c.clip()
    # Four body tones drawn from the same six poster inks as the environment.
    rect(c, -half, top, width, bottom - top, p[2])             # p[1] would melt into the navy rays and chisel the nose
    rect(c, -half + 9, top, width - 25, bottom - top, p[4])
    rect(c, -half + 22, top, width * .31, bottom - top, p[5])
    rect(c, half - 20, top, 11, bottom - top, p[3])
    for y in range(int(top + 125), int(bottom - 18), 72):
        rect(c, -half, y, width, 3, p[1])
        rect(c, -half + 15, y + 4, width - 27, 2, p[5])
    for y in (top + 148, bottom - 42):
        rect(c, -half, y, width, 9, p[0])
    # Roll checker is contained in the cylindrical silhouette.
    yy = bottom - 115
    rect(c, -half, yy, width, 52, p[5])
    for j in range(6):
        xx = -half + j * width / 6
        rect(c, xx, yy + (0 if j % 2 else 26), width / 6, 26, p[0])
    for j in range(9):
        rect(c, -half + 6 + j * (width - 12) / 9, bottom - 29, 2, 25, p[1])
    c.restore()


def engines(c, y, metal, p, upper=False, lit=True):
    xs = (0,) if upper else (-31, -10, 11, 32)
    for x in xs:
        r = 21 if upper else 12
        poly(c, [(x - r * .55, y), (x + r * .55, y),
                 (x + r, y + 34), (x - r, y + 34)], metal)
        # The bell's lit side and its lip, so it reads as a turned part and not a dark wedge.
        poly(c, [(x - r * .36, y), (x - r * .14, y), (x - r * .3, y + 29), (x - r * .68, y + 29)], p[3])
        rect(c, x - r, y + 29, r * 2, 6, p[4])
        if lit:
            # The open mouth of the bell, seen slightly from below; a round disc would read as a pearl.
            c.save()
            c.translate(x, y + 36)
            c.scale(1, .34)
            disc(c, 0, 0, r * .86, p[5])
            c.restore()


def ascent_flame(c, t, p):
    sep = t - 3.18
    if sep >= .2:
        return
    vehicle(c, t)
    if sep > 0:
        # Engine cut-off: the flame is no longer fed, drifts with the spent stage and collapses from the nozzle down.
        c.translate(-1550 * sep, 580 * sep + 1200 * sep * sep)
    # The shear layer entrains surrounding air: its cross section GROWS below
    # the nozzle. Each band is one connected contour, never a chain of dots.
    d = np.linspace(0.0, 1550.0, 118)
    spread = 42 + .66 * d + 21 * (1 - np.exp(-d / 45))
    def edge(side):
        out = np.zeros_like(d)
        for k in range(18):
            centre = 40 + k * 91 + 20 * math.sin(k * 2.17 + side)
            width = 22 + k * 3.3
            amplitude = 9 + k * 3.0
            out += amplitude * np.exp(-((d - centre) / width) ** 2)
        return out * (1 - np.exp(-d / 45))
    left, right = edge(-1), edge(1)
    bend = 10 * np.sin(d / 340 + t * .6) * (1 - np.exp(-d / 80))
    cut = 6200 * max(0.0, sep)
    for fraction, col in ((1.20, 1), (1.0, 3), (.72, 4), (.36, 5)):
        # The core starves first, so the layers open from the inside.
        start = cut * (1 + .7 * (1.2 - fraction))
        # On the first frame a thread still joins it to the nozzle; after that the flame is a free spire falling away.
        keep = np.nonzero(d >= (0 if sep < .035 else start))[0]
        if len(keep) < 4:
            continue
        # Gas pinching off, never a cross-section sliced flat or a dome.
        arch = np.clip((d - start) / 760, .012 * fraction if sep < .035 else 0, 1) ** .72 if cut > 0 else np.ones_like(d)
        pts = [(float(bend[j] - arch[j] * fraction * (spread[j] + left[j])), float(891 + d[j])) for j in keep]
        pts += [(float(bend[j] + arch[j] * fraction * (spread[j] + right[j])), float(891 + d[j])) for j in keep[::-1]]
        curved(c, pts, p[col])
    # Shock diamonds in the core, standing still against the rocket with a slight flutter.
    for k in range(6):
        y = 925 + k * (88 + 6 * k) + 5 * math.sin(t * 31 + k)
        hw = 27 - 2.5 * k
        hh = 34 - 2 * k
        if sep <= 0:
            poly(c, [(-hw, y), (0, y - hh), (hw, y), (0, y + hh)], p[4 if k % 2 else 3])
    c.restore()


def trail_bend(t):
    """Curvature shared by the plume and the trail: the arc of the path the vehicle has just pitched away from."""
    return .00038 * ease((t - 3.45) / 1.4)


@lru_cache(maxsize=1)
def plume_seeds():
    rng = np.random.default_rng(3141)
    # Each gas layer ends in its own irregular tongues: (lateral, sharpness, reach, flicker rate, phase).
    tongues = tuple(tuple((float(rng.uniform(-.92, .92)), float(rng.uniform(.035, .09)), float(rng.uniform(.14, .4)),
                           float(rng.uniform(17, 34)), float(rng.uniform(0, TAU))) for _ in range(count))
                    for count in (7, 6, 5, 3))
    # Rags of gas that have left the tongues: (lateral, rate, phase, length, width, ink).
    wisps = tuple((float(rng.uniform(-.85, .85)), float(rng.uniform(1.3, 2.4)), float(rng.uniform(0, 1)),
                   float(rng.uniform(40, 120)), float(rng.uniform(3, 8)), int(rng.choice((2, 2, 3))))
                  for _ in range(26))
    return tongues, wisps


def vacuum(c, t, p):
    if t < 3.23:
        return
    vehicle(c, t)
    start = ease((t - 3.23) / .3)
    fade = 1 - .55 * ease((t - 4.0) / .9)
    k = trail_bend(t)
    tongues, wisps = plume_seeds()
    # Vacuum plume: the gas fans straight out of the bell and thins away. No outer boundary ever closes it;
    # it ends where each ray of gas gives out, in tongues, then in rags.
    v = np.linspace(-1, 1, 241)

    def fan(ell, width):
        return 17 + width * (1 - np.exp(-ell / 170)) + .05 * ell * width / 132

    for (width, length, col), family in zip(((116, 560, 2), (84, 450, 3), (48, 280, 4), (20, 170, 5)), tongues):
        width, length = width * start * (1 - .2 * ease((t - 4.0) / .9)), length * start * fade
        lick = np.zeros_like(v)
        for v0, sharp, amount, rate, phase in family:
            tent = np.clip(1 - np.abs(v - v0) / (3.4 * sharp), 0, 1) ** 2
            lick = np.maximum(lick, amount * (1 + .22 * math.sin(rate * t + phase)) * tent)
        ell = length * (.3 + .58 * (1 - v * v) ** 2.4 + lick)
        tail = np.stack((v * fan(ell, width) + k * ell * ell, 358 + ell), 1)
        side_l = np.linspace(0, ell[0], 22)[:-1]
        side_r = np.linspace(0, ell[-1], 22)[:-1]
        left = np.stack((-fan(side_l, width) + k * side_l * side_l, 358 + side_l), 1)
        right = np.stack((fan(side_r, width) + k * side_r * side_r, 358 + side_r), 1)
        poly(c, np.vstack((left, tail, right[::-1])).tolist(), p[col])
    width, length = 116 * start, 560 * start * fade
    for lateral, rate, phase, size, thick, ink in wisps:
        u = (t * rate + phase) % 1.0
        ell = length * (.6 + .62 * u)
        wide = float(fan(ell, width))
        x, y = lateral * wide + k * ell * ell, 358 + ell
        # Direction of this ray of gas, so the rags keep flying where the tongue was pointing.
        dx, dy = lateral * (width * math.exp(-ell / 170) / 170 + .05 * width / 132) + 2 * k * ell, 1.0
        norm = math.hypot(dx, dy)
        dx, dy = dx / norm, dy / norm
        w, h = thick * (1 - u) * start, size * (1 - .5 * u) * start
        poly(c, [(x - dy * w, y + dx * w), (x + dx * h, y + dy * h), (x + dy * w, y - dx * w), (x - dx * h * .4, y - dy * h * .4)], p[ink])
    c.restore()


@lru_cache(maxsize=1)
def staging_seeds():
    rng = np.random.default_rng(318)
    needles = tuple(sorted((float(rng.uniform(0, TAU)), float(rng.uniform(.42, 1.0))) for _ in range(15)))
    shards = tuple((float(rng.uniform(0, TAU)), float(rng.uniform(240, 1200)), float(rng.uniform(-9, 9)),
                    float(rng.uniform(0, TAU)), float(rng.uniform(6, 22)), int(rng.choice((5, 4, 3))),
                    float(rng.uniform(.55, 1.2))) for _ in range(28))
    # Harmonics that tear the shock rings: (order, phase) pairs, one list per ring.
    tears = tuple(tuple((int(rng.integers(3, 17)), float(rng.uniform(0, TAU))) for _ in range(5)) for _ in range(2))
    return needles, shards, tears


def ring(c, radius, thick, squash, a0, a1, tear, age, color):
    """Part of a blast ring in the plane of the joint. Heavy on the near side; as it ages it tears into tapering arcs."""
    th = np.linspace(a0, a1, 150)
    rough = sum(np.sin(n * th + phase) for n, phase in tear) / 2.4
    w = thick * (1 - age ** 3) * (.34 + .66 * (.5 + .5 * np.sin(th))) * np.clip(rough * .8 + 1.25 - 2.1 * age, 0, 1)
    swell = radius * (1 + .018 * rough)
    outer = np.stack((swell * np.cos(th), 345 + swell * squash * np.sin(th)), 1)
    inner = np.stack(((swell - w) * np.cos(th), 345 + (swell - w) * squash * np.sin(th)), 1)
    poly(c, np.vstack((outer, inner[::-1])).tolist(), color)


def shock(c, sep, p, near):
    a0, a1 = (0, math.pi) if near else (math.pi, TAU)
    grow = 1 - math.exp(-6 * sep)
    tears = staging_seeds()[2]
    for radius, thick, life, tear, ink in ((70 + 1650 * grow, 52, .46, tears[0], p[5]), (46 + 900 * grow, 40, .38, tears[1], p[4])):
        if sep < life:
            ring(c, radius, thick, .3, a0, a1, tear, sep / life, ink)


def burst(c, sep, p):
    """Hard attack, full by 3.27 s, holding to 3.335 s: the first frame after staging already has to hit."""
    x = sep / .31
    if x >= 1:
        return
    flash = (1 - (1 - min(1.0, x / .3)) ** 3) * (1 - ease((x - .5) / .5))
    needles = staging_seeds()[0]
    # The blast leaves in the plane of the joint: a flat crown of spikes, much wider than it is tall.
    for reach, waist, ink in ((640, 104, p[4]), (390, 66, p[5])):
        pts = []
        for j, (a, length) in enumerate(needles):
            r = reach * length * flash
            pts.append((r * math.cos(a), 345 + .34 * r * math.sin(a)))
            b = a + (needles[(j + 1) % len(needles)][0] - a) % TAU * .5
            pts.append((waist * flash * math.cos(b), 345 + waist * flash * .56 * math.sin(b)))
        poly(c, pts, ink)


def debris(c, sep, p):
    """Bits of the interstage. They lag the accelerating upper stage and share the spent stage's sideways drift."""
    if sep > 1.25:
        return
    fall = 300 * sep + 430 * sep * sep
    drift = -560 * sep
    spread = (1 - math.exp(-2.8 * sep)) / 2.8
    plate = ((-1, -.5), (.3, -.72), (1, -.14), (.62, .55), (-.72, .46))
    for a, speed, spin, turn, size, ink, life in staging_seeds()[1]:
        s = size * (1 - ease(sep / life))
        if s < .4:
            continue
        x = math.cos(a) * speed * spread + drift
        y = 345 + .4 * math.sin(a) * speed * spread + fall
        rot = turn + spin * sep
        co, si = math.cos(rot), math.sin(rot)
        # A convex torn plate with a shaded side, so it reads as metal and not as confetti.
        poly(c, [(x + s * (u * co - v * si), y + s * (u * si + v * co)) for u, v in plate], p[ink])
        poly(c, [(x + s * (u * co - v * si), y + s * (u * si + v * co)) for u, v in plate[2:]], p[2 if ink != 2 else 1])
    # The two big halves of the interstage skirt, ribbed like the hull they came off. They clear the plume on each side.
    shrink = 1 - ease(sep / 1.25)
    for side in (-1, 1):
        x = side * (40 + 1750 * spread) + drift * .5
        y = 345 - 150 * spread + fall
        c.save()
        c.translate(x, y)
        c.rotate(side * (.25 + 4.1 * sep))
        c.scale(shrink, shrink)
        s = np.linspace(-1, 1, 15)
        bow = 10 * s * s
        half = 42
        poly(c, np.vstack((np.stack((half * s, -21 - bow), 1), np.stack((half * s, 21 - bow), 1)[::-1])).tolist(), p[4])
        poly(c, np.vstack((np.stack((half * s, -21 - bow), 1), np.stack((half * s, -9 - bow), 1)[::-1])).tolist(), p[5])
        poly(c, np.vstack((np.stack((half * s, 13 - bow), 1), np.stack((half * s, 21 - bow), 1)[::-1])).tolist(), p[2])
        for u in np.linspace(-.82, .82, 9):
            rect(c, half * u - 1, -9 - 10 * u * u, 2, 22, p[1])
        c.restore()


def rocket(c, t, p):
    vehicle(c, t)
    sep = max(0.0, t - 3.18)
    fins = steel(t, p, 1.88)
    bells = steel(t, p, 2.12)
    if 0 < sep:
        shock(c, sep, p, near=False)
    # Single upper stage with a fairing, not two pointed miniature rockets.
    hull(c, 0, 326, 90, p, pointed=True)
    if sep > 0:
        engines(c, 326, bells, p, upper=True)
    c.save()
    # A separation impulse clears the expansion cone immediately. Rotation is
    # about the detached stage's upper end, not the vehicle's distant nose.
    c.translate(-1550 * sep, 328 + 580 * sep + 1200 * sep * sep)
    c.rotate(1.65 * sep)
    c.translate(0, -328)
    hull(c, 328, 855, 90, p)
    for side in (-1, 1):
        x = side * 45
        poly(c, [(x, 360), (x + side * 28, 373), (x + side * 28, 407), (x, 397)], fins)
        for j in range(3):
            rect(c, min(x, x + side * 28) + 3, 376 + j * 8, 21, 3, p[3])
        poly(c, [(x, 795), (x + side * 22, 858), (x + side * 22, 873), (x, 857)], fins)
    engines(c, 855, bells, p, lit=t < 3.18)
    c.restore()
    if sep > 0:
        debris(c, sep, p)
        shock(c, sep, p, near=True)
        burst(c, sep, p)
    c.restore()


def ice(c, t, p):
    if t > 1.5:
        return
    for birth, height, side, speed, size in scenery()[2]:
        age = t - birth
        if not 0 < age < .65:
            continue
        ex, ey = emitter(t, height)
        x = ex + side * (46 + speed * age)
        y = ey + 170 * age * age
        poly(c, [(x, y - size), (x + size * .5, y), (x, y + size),
                 (x - size * .7, y + size * .3)], p[5])


def collar(c, t, p):
    # Analytic constant-acceleration ascent through an exponential atmosphere.
    # q ~ velocity^2 * density; condensation exists only close to peak pressure.
    hscale = 3300.0
    peak = math.sqrt(hscale / 950.0)
    q = (t / peak) ** 2 * math.exp(1.0 - (t / peak) ** 2)
    f = ease((q - .982) / .018)
    if f <= 0 or t > 3.0:
        return
    mach = 1.12 + .52 * max(0.0, t - 1.65)
    cone = math.asin(1.0 / mach)
    length = 74.0 * f
    flare = min(165.0, length * math.tan(cone))
    vehicle(c, t)
    for side in (-1, 1):
        # The condensation follows the shoulder's expansion cone. Two smooth
        # boundaries define a thin shell, not a rectangle placed over the hull.
        c.new_path()
        c.move_to(side * 44, 211)
        c.curve_to(side * (44 + flare * .16), 220,
                   side * (44 + flare * .78), 211 + length * .72,
                   side * (44 + flare), 211 + length)
        c.curve_to(side * (44 + flare * .91), 222 + length,
                   side * (44 + flare * .44), 237 + length * .48,
                   side * 44, 231)
        c.close_path()
        c.set_source_rgb(*p[5])
        c.fill()
    c.restore()


def orbit_trail(c, t, p):
    if t < 3.5:
        return
    grow = ease((t - 3.5) / .5)
    k = trail_bend(t)
    # Leaves the plume on the plume's own axis and reaches the limb as a point, wherever the limb is this frame.
    ell = np.linspace(0, 2200, 300)
    x, y = k * ell * ell, 358 + ell
    a, z = angle(t), pull(t)
    px, py = pivot(t)
    sx = px + z * .9 * (x * math.cos(a) - y * math.sin(a))
    sy = py + z * (nose_y(t) - 340 + .9 * (x * math.sin(a) + y * math.cos(a)))
    _, _, cx, cy, radius = planet(t)
    below = np.nonzero(sy >= cy - np.sqrt((radius + 15) ** 2 - (sx - cx) ** 2))[0]
    end = float(ell[below[0]]) if below.size else float(ell[-1])
    tx, ty = 2 * k * ell, np.ones_like(ell)
    norm = np.hypot(tx, ty)
    tx, ty = tx / norm, ty / norm
    vehicle(c, t)
    # Each ink dies earlier than the one around it, so the trail thins to a single blue thread.
    for reach, width, col in ((1.0, 17, 2), (.74, 10, 3), (.46, 3.6, 5)):
        s = np.minimum(1.0, ell / (end * reach))
        w = width * grow * (1 - s) ** .8
        pts = np.vstack((np.stack((x - ty * w, y + tx * w), 1), np.stack((x + ty * w, y - tx * w), 1)[::-1]))
        poly(c, pts.tolist(), p[col])
    c.restore()


def planet(t):
    u = ease((t - 3.05) / 1.45)
    top = 1180 - 465 * u
    return u, top, 730.0, top + 6400.0, 6400.0


def sunburst(c, t, p):
    """Poster rays from the sun on the limb: two flat inks, fanning open as the planet rises."""
    if t < 3.05:
        return
    u, _, cx, cy, radius = planet(t)
    x = 1190.0
    y = cy - math.sqrt(radius * radius - (x - cx) ** 2)
    # Each ray opens to half its pitch, so the fan stays striped to the end. It blows open on the staging beat.
    half = .5 * math.pi / 34 * ease((t - 3.18) / .25)
    ray = mix(p[2], p[1], ease((t - 3.2) / .35))              # hot on the beat, cooling back to navy
    for j in range(17):
        if half <= 0:
            break
        a = -math.pi + (j + .5) * math.pi / 17 + .05 * (t - 3.15)
        poly(c, [(x, y), (x + 2600 * math.cos(a - half), y + 2600 * math.sin(a - half)),
                 (x + 2600 * math.cos(a + half), y + 2600 * math.sin(a + half))], ray)
    r = 138 * u
    c.new_path()                                              # one thin halo, then the disc
    c.arc(x, y, r * 1.42, 0, TAU)
    c.new_sub_path()
    c.arc(x, y, r * 1.34, 0, TAU)
    c.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    c.set_source_rgb(*p[2])
    c.fill()
    c.set_fill_rule(cairo.FILL_RULE_WINDING)
    disc(c, x, y, r, p[4])
    disc(c, x, y, r * .68, p[5])


def streak(c, x0, length, y0, height, seed, phase, cx, radius, color):
    """A band of cloud lying on the planet: a lens that closes to a point at both ends, never a cut-off pipe."""
    upper, lower = [], []
    for s, h in cloud_profile(seed):
        e = math.sin(math.pi * s) ** .85
        x = x0 + length * s
        y = y0 + (x - cx) ** 2 / (2 * radius) + 14 * math.sin(s * 3 + phase)
        upper.append((x, y - height * h * e))
        lower.append((x, y + height * .26 * e))
    curved(c, upper + lower[::-1], color)


def cyclone(c, x, y, size, squash, turn, p):
    """A storm seen from orbit, the one thing that says Earth at a glance: spiral arms, each one a tapering lens."""
    s = np.linspace(0, 1, 44)
    for k, col in enumerate((3, 5, 3, 3, 5, 3)):
        start = turn + k * TAU / 6
        th = start + 3.1 * s
        r = size * (.1 + .9 * s ** 1.25) * (1 if k % 3 == 1 else .86)
        w = size * .075 * np.sin(np.pi * s ** .72) ** 1.15 * (1 if col == 5 else 1.5)
        outer = np.stack((x + (r + w) * np.cos(th), y + (r + w) * np.sin(th) * squash), 1)
        inner = np.stack((x + (r - w) * np.cos(th), y + (r - w) * np.sin(th) * squash), 1)
        poly(c, np.vstack((outer, inner[::-1])).tolist(), p[col])


def earth(c, t, p):
    if t < 3.05:
        return
    u, top, cx, cy, radius = planet(t)
    disc(c, cx, cy, radius + 15, p[2])
    disc(c, cx, cy, radius + 6, p[3])
    # Ocean stays visibly indigo, rather than being erased to the sky colour.
    disc(c, cx, cy, radius - 3, p[1])
    c.save()
    c.new_path()
    c.arc(cx, cy, radius - 22, 0, TAU)
    c.clip()
    # Invented coastlines: continuous nested bays and peninsulas, not map assets.
    continents = (
        ((-200, 100), (-70, 70), (50, 115), (160, 120), (210, 80),
         (310, 85), (340, 130), (420, 155), (475, 130), (560, 195),
         (535, 220), (600, 242), (570, 275), (650, 310), (630, 350),
         (460, 400), (300, 370), (150, 410), (-120, 370)),
        ((1550, 90), (1670, 80), (1780, 105), (1900, 70), (2020, 125),
         (2050, 310), (1760, 350), (1610, 290), (1650, 235),
         (1520, 205), (1450, 150)),
        ((875, 270), (925, 258), (951, 281), (1005, 290), (1040, 325),
         (1000, 339), (949, 314), (907, 324), (864, 298)),
    )
    for points in continents:
        curved(c, [(x, top + y + (x - cx) ** 2 / (2 * radius)) for x, y in points], p[2])
    # Wind-shaped cloud ribbons share the planet's projection and curvature.
    ribbons = ((1210, 720, 84, 11, 0), (-40, 640, 104, 13, 1), (640, 420, 132, 13, 0), (1060, 980, 186, 25, 1),
               (-170, 760, 196, 21, 0), (800, 640, 292, 31, 1), (1460, 620, 320, 26, 0), (-90, 420, 352, 24, 0))
    for j, (x0, length, depth, height, crest) in enumerate(ribbons):
        streak(c, x0, length, top + depth, height, 13 + j % 3, j, cx, radius, p[3])
        if crest:
            # Very small white crests, subordinate to the engine's white point.
            streak(c, x0 + length * .3, length * .44, top + depth - height * .2, height * .5, 16 + j % 2, j, cx, radius, p[5])
    cyclone(c, 500, top + 268 + (500 - cx) ** 2 / (2 * radius), 165, .34, .3 * t, p)
    c.restore()
    if u > .28:
        amount = ease((u - .28) / .72)
        # Sunlight running along the atmosphere: crescents lying exactly on the limb, closing to nothing at both ends.
        for reach, thick, col in ((760, 11, 4), (330, 7, 5)):
            x0, x1 = 1190 - reach * amount, 1190 + reach * amount
            pts = []
            for j in range(97):
                x = x0 + (x1 - x0) * j / 96
                yy = cy - math.sqrt((radius + 15) ** 2 - (x - cx) ** 2)
                pts.append((x, yy - thick * amount * math.sin(math.pi * j / 96) ** 2.4))
            for j in range(96, -1, -1):
                x = x0 + (x1 - x0) * j / 96
                yy = cy - math.sqrt((radius + 15) ** 2 - (x - cx) ** 2)
                pts.append((x, yy + .4 * thick * amount * math.sin(math.pi * j / 96) ** 2.4))
            poly(c, pts, p[col])


def flat(t):
    """Pure frame function; all finite t are clamped, never wrapped."""
    t = float(t)
    if not math.isfinite(t):
        t = 0.0
    t = max(0.0, min(4.999999, t))
    p = inks(t)
    surface, c = canvas(p[0], W, H)
    c.set_antialias(cairo.ANTIALIAS_BEST)
    sky(c, t, p)
    sunburst(c, t, p)
    air_banks(c, t, p)
    streaks(c, t, p)
    pad(c, t, p)
    smoke(c, t, p)
    cloud_deck(c, t, p)
    orbit_trail(c, t, p)
    earth(c, t, p)
    ascent_flame(c, t, p)
    vacuum(c, t, p)
    rocket(c, t, p)
    ice(c, t, p)
    collar(c, t, p)
    front_banks(c, t, p)
    surface.flush()
    return surface
