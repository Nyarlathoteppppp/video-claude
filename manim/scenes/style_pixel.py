"""红日落入像素海 — a 320×180 indexed scene on a 64-register palette, clocked at 12 Hz.

No RGB paint operations and no spatial interpolation, as on the hardware it pretends to be. The sun
sets for real: layers of clouds and each row of the sea slide by whole pixels at their own speed,
the rows just above the horizon shimmer as a raster effect, and palette programming (cycling, dusk
to night, the pulse of light that runs down the reflection, stars that were always there) does what
only an indexed machine can do. A junk crosses the sun from first frame to last, bow in at the start,
dead centre as the pulse lands, stern out at the end, its upside-down twin cutting a notch in the
path of light. A basalt headland frames the left of the bay, and a stone lantern on it is lit at dusk.

Registers (inclusive): 0–2 basalt; 3–10 atmosphere; 11–16 cloud underlighting; 17–24 solar rim;
25–28 ocean; 29–40 reflection; 41–44 solar disc; 45–53 latent stars, three brightnesses over three
sky registers (identical to those sky registers until nightfall); 54 sail cloth; 55, 56 rim light on
the junk's bow-facing and stern-facing edges; 57 basalt middle tone; 58–60 lamplit basalt; 61 flame;
62, 63 lamp glow. 255 is transparent, in layers only.
"""
from functools import lru_cache
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

from styles_common import surface

WIDTH, HEIGHT, SCALE, RATE = 320, 180, 6, 12
CX, CY0, RADIUS, HORIZON = 183, 58, 37, 96
CLEAR = 255
PAD = 60
REGISTERS = 64
STARS, CLOTH, RIM_BOW, RIM_STERN, BASALT_MID = 45, 54, 55, 56, 57
LIT_DARK, LIT_MID, LIT_RIM, FLAME, GLOW_NEAR, GLOW_FAR = 58, 59, 60, 61, 62, 63
WATERLINE = 100      # where the junk meets the sea
LIGHTING = 44        # the step at which the lantern is touched off

# The headland's skyline, walked by hand; the level stretch at x 34–48 is the lantern's ledge.
RIDGE_X = (0, 3, 6, 9, 12, 15, 18, 22, 25, 28, 31, 34, 48, 51, 54, 57, 60, 63, 65, 69, 72, 76, 80, 85, 88, 93, 97, 103, 108, 113, 118)
RIDGE_Y = (57, 53, 56, 51, 57, 55, 61, 60, 66, 65, 72, 79, 79, 86, 91, 98, 103, 104, 112, 114, 124, 126, 137, 139, 149, 151, 161, 163, 172, 174, 186)

# Top edge of each sky register from 4 up: deliberately uneven, the way dusk actually stacks.
SKY_EDGES = (11, 27, 38, 57, 66, 78, 88)
SKY_SWAY = (1.3, 2.2, 1.6, 2.8, 1.8, 1.4, .9)

# (x, y) by hand. 1 px by the handful with thin patches between, a few doubles, and three that earn a cross.
STAR_DUST = ((5, 6), (13, 14), (19, 3), (33, 1), (44, 3), (52, 15), (61, 2), (69, 14), (84, 3), (97, 1), (104, 14),
             (112, 3), (121, 7), (137, 4), (141, 19), (148, 9), (156, 2), (160, 24), (167, 13), (176, 5), (179, 8),
             (183, 17), (186, 6), (188, 31), (195, 10), (201, 3), (207, 19), (222, 4), (228, 26), (236, 14),
             (241, 30), (244, 6), (258, 25), (263, 2), (270, 33), (279, 5), (284, 24), (298, 4), (305, 26),
             (312, 9), (316, 33))
STAR_DOUBLES = ((22, 9), (76, 2), (172, 10), (251, 3), (150, 30), (308, 16))
STAR_CROSSES = ((128, 13), (214, 24), (292, 31))

JUNK = """
....................................#.........................................
....................................#.............##..........................
....................................#...........##ccr.........#...............
....................................#........###ccccr.........#...............
....................................#......##ccccccccr........#......#........
....................................#...###cccccccccc##.......#.....#cr.......
....................................#.##cccccccccc###.........#...##ccr.......
...........#.......................###cccccccc####..lcr.......#..#cccccr......
...........#........##...........##c##ccccc###...lcccccr......#.#ccccccr......
...........#.......#ccr........##ccc##cc###...lccccccccr......##cccccccc##....
...........#....###cccr........lcccc####cccccccccccccc###....##ccccccc##......
............#..#cccccccr.......lc#####cccccccccccc####......#c#ccccc##.lr.....
............###ccccccccc##.....##ccc##cccccccc####...lccr..#cc#cc###.lccr.....
...........##cccccccc###.......lcccc##ccc#####...lccccccr..lcc###..lccccr.....
........###c#ccccc###..lr.....lccccc#####....lccccccccc###.lc##ccccccccc##....
.......#cccc#cc###..lcccr.....lcc#####cr.lccccccc######....##c#ccccc####....f.
.......lcccc###..lccccccr.....###ccc##cccccc#####.....lcr.lccc#c####...r....#.
.......lc####r.lccccccccc##...lccccc##c#####cr...lccccccr.lc####...lcccr.....#
.......##cccc#cccccc#####.....lcc######cccccccccccccccccr.##cr#.lcccccr.....##
.......lccccc#ccc###....lr....###ccc##ccccccccccccccc####.cccc#cc#######....#r
......lccccc#####...lccccr...lcccccc##ccccc##########.....#######..........##r
......lcc###c#cccccccccccr...lcccc#########.........lcccr..lcc#cccccr......###
......###cccc#ccccccccc####..#####cc##ccr..lccccccccccccr..lcc#ccccr....####r.
......lcccccc#ccc######......lcccccc##cccccccccccccccccr...########.###...##r.
......lccc#######.....lcr...lccccccc##cccc###############.....#...##..#..###r.
......####ccr#..lcccccccr...##############....................###..#.########.
......lccccccc#cccccccccr...lccccccc##....lcccccccccccr....####.#..####ww##r..
......lccccccc#cccccc#####..lccccccc##ccccccccccccccccr..##.#.#.####w##ww##r..
.....lcccc###########.......lccccccc##ccc##############.....########w#######..
l....#####cccc#.....lccr...##############r..............########w######==###..
#l...lcccccccc#ccccccccr...lcccccccc##ccccccccccccccr...####w###w###===###r...
.#l..lcccccccc#cccccccr....lcccccccc##cccccccccccccr....####w#####==######r...
.##l.##################....########################.....#######===########....
.###l..........#....................##..................####===###########....
..###l.###################################################==#############.....
..####l#.......#....................##.................##################..#..
...##ww#################################################################..##..
...################################################============#########.###..
....####===========================####============####################..###..
.....#################################################################..####..
.....#################################################################.#####..
......#####===================###=========================###########..#####..
........############################################################..######..
.........##########################################################..######...
............######################################################..#######...
..............##################################################...########...
"""

# A stone lantern: jewel, curved roof, light chamber, platform, shaft, foot.
LANTERN = """
.....#.....
....###....
...#####...
.#########.
###########
#.uuuuuuu.#
..#efffe#..
..#efffe#..
..#eefee#..
.#########.
...uuuuu...
....###....
....###....
....###....
...#####...
.#########.
"""
LANTERN_AT = (36, 64)
LAMP = (41, 71)

# A wind-bent pine on the crag: flat pads of needle over a leaning trunk.
PINE = """
............++####..........
.........####+++####........
.......###############+.....
.....######..##########+....
...........##....#####+.....
............##..............
.....++###...##...##+.......
...###########.##.####+.....
..##############.######+....
......####....##...###......
...............##...........
...............##...........
..............##............
.............###............
.............##.............
............###.............
............###.............
"""
PINE_AT = (-4, 35)

# The headland's slabs, each (register, polygon): long wedges falling the way the strata fall, overlapping en echelon,
# lighter where a face turns toward the sun. Placed by eye, not by rule, and deliberately never closing into cells.
FACETS = (
    (BASALT_MID, ((3, 58), (9, 53), (12, 60), (14, 76), (8, 84), (4, 72))),
    (BASALT_MID, ((16, 62), (24, 68), (30, 75), (33, 84), (28, 98), (20, 84))),
    (BASALT_MID, ((36, 82), (49, 82), (52, 94), (57, 106), (50, 112), (42, 100))),
    (BASALT_MID, ((6, 96), (16, 92), (22, 112), (24, 134), (14, 124))),
    (BASALT_MID, ((24, 104), (34, 100), (42, 118), (46, 140), (36, 132))),
    (BASALT_MID, ((44, 112), (56, 112), (66, 132), (64, 146), (52, 130))),
    (BASALT_MID, ((68, 134), (78, 142), (88, 160), (84, 176), (74, 156))),
    (BASALT_MID, ((50, 146), (62, 152), (72, 178), (60, 180), (54, 164))),
    (BASALT_MID, ((30, 146), (38, 150), (46, 176), (36, 180), (32, 164))),
    (BASALT_MID, ((0, 138), (6, 136), (12, 160), (10, 180), (0, 180))),
    (1, ((6, 56), (9, 54), (11, 60), (9, 68))),
    (1, ((25, 68), (29, 72), (31, 78), (28, 82))),
    (1, ((50, 87), (54, 92), (58, 100), (55, 104), (50, 96))),
    (1, ((61, 105), (65, 113), (62, 118), (58, 110))),
    (1, ((70, 116), (73, 125), (70, 131), (66, 122))),
    (1, ((81, 139), (85, 141), (88, 150), (84, 152), (79, 144))),
    (1, ((98, 162), (103, 165), (107, 173), (103, 176), (97, 168))),
)
# Joints across the slabs, opposing the fall of the strata; long ones first, then the short ones that die out.
FISSURES = (
    ((2, 90), (14, 84), (21, 87)),
    ((26, 100), (37, 92), (46, 96)),
    ((10, 122), (19, 118), (27, 123)),
    ((40, 124), (52, 118), (60, 122)),
    ((56, 148), (67, 140), (77, 146)),
    ((30, 158), (41, 152), (47, 155)),
    ((62, 168), (74, 162), (80, 166)),
    ((6, 66), (13, 62)),
    ((22, 76), (29, 72)),
    ((6, 108), (13, 104)),
    ((36, 112), (44, 108)),
    ((47, 138), (54, 134)),
    ((4, 150), (10, 147)),
    ((72, 150), (79, 147)),
)
# A low reef off the foot of the headland, in front of the path of light.
REEF = ((118, 180), (122, 177), (126, 177), (129, 174), (132, 175), (135, 172), (138, 173), (141, 176), (145, 176), (148, 178), (154, 180))


def _sprite(art, codes):
    rows = art.split()
    out = np.full((len(rows), len(rows[0])), CLEAR, np.uint8)
    for j, row in enumerate(rows):
        for i, ch in enumerate(row):
            if ch != ".":
                out[j, i] = codes[ch]
    return out


def _rgb(values):
    return np.array([tuple(bytes.fromhex(s)) for s in values.split()], dtype=np.float64)


def _ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _clean(a):
    """Remove small same-register islands, not just isolated foreground dots.

    Four-connected groups of fewer than three pixels are reassigned to their
    strongest bordering register. Two-pixel-wide deliberate dither clusters
    survive; accidental one-pixel stair-step debris does not.
    """
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    height, width = a.shape
    while True:
        changed = False
        for k in range(STARS):
            components, count = label(a == k, cross)
            if not count:
                continue
            sizes = np.bincount(components.ravel())
            for component in np.flatnonzero((sizes < 3) & (np.arange(len(sizes)) != 0)):
                ys, xs = np.nonzero(components == component)
                neighbours = []
                for yy, xx in zip(ys, xs):
                    for oy, ox in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = yy + oy, xx + ox
                        if 0 <= ny < height and 0 <= nx < width and a[ny, nx] != k:
                            neighbours.append(int(a[ny, nx]))
                if neighbours:
                    a[ys, xs] = np.bincount(neighbours, minlength=REGISTERS).argmax()
                    changed = True
        # Each replacement merges an entire component into a touching one,
        # strictly reducing the component count. Thus this terminates, and
        # unlike a median pass it actually enforces the minimum cluster size.
        if not changed:
            return a


@lru_cache(maxsize=1)
def _sky():
    """Bands of unequal height whose edges wander, each edge feathered with a few long slivers of its neighbour,
    and the latent stars: every star pixel is an alias of the sky register it replaces, so at dusk it cannot be seen."""
    rng = np.random.default_rng(23)
    rows = np.arange(HEIGHT)[:, None]
    xx = np.arange(WIDTH)
    a = np.full((HEIGHT, WIDTH), 3, np.uint8)
    edges = []
    for k, (edge, sway) in enumerate(zip(SKY_EDGES, SKY_SWAY)):
        ph = rng.uniform(0, 2 * np.pi, 3)
        wander = np.rint(edge + sway * (np.sin(xx / 83 + ph[0]) + .6 * np.sin(xx / 37 + ph[1]) + .25 * np.sin(xx / 17 + ph[2])))
        edges.append(wander)
        a[rows >= wander[None]] = 4 + k
    for k, wander in enumerate(edges):
        for _ in range(int(rng.integers(4, 8))):
            x0 = int(rng.integers(0, WIDTH))
            half = int(rng.integers(5, 26))
            above = rng.random() < .5
            off = int(rng.integers(1, 4))
            row = int(wander[x0]) + (-off - 1 if above else off)
            lo, hi = max(0, x0 - half), min(WIDTH, x0 + half)
            if 0 <= row < HEIGHT:
                a[row, lo:hi] = 4 + k if above else 3 + k
                inner = half // 2
                if half > 14 and 0 <= row + 1 < HEIGHT:
                    a[row + 1, max(0, x0 - inner):min(WIDTH, x0 + inner)] = 4 + k if above else 3 + k
    a = _clean(a)

    def star(x, y, tier):
        if 3 <= a[y, x] <= 5:
            a[y, x] = STARS + (a[y, x] - 3) * 3 + tier

    for x, y in STAR_DUST:
        star(x, y, 0)
    for x, y in STAR_DOUBLES:
        star(x, y, 1)
        star(x + 1, y, 1)
    for x, y in STAR_CROSSES:
        star(x, y, 2)
        for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            star(x + ox, y + oy, 1)
    return a


def _deck_layer(decks):
    """Long cloud decks, shaped by several scales of wind shear. Their undersides
    are broken ribbons, not outlines around a pile of ellipses. Wider than the frame, to drift."""
    y, x = np.mgrid[:HEIGHT, -PAD:WIDTH + PAD]
    xx = x[0]
    a = np.full(x.shape, CLEAR, np.uint8)
    for left, right, mid, thick, ph in decks:
        u = (xx - left) / (right - left)
        envelope = np.maximum(0, np.sin(np.clip(u, 0, 1) * np.pi)) ** .55
        shear = 1.7 * np.sin(xx / 34 + ph) + .7 * np.sin(xx / 11 + ph)
        bottom = mid + shear + thick * .35 * envelope
        top = bottom - thick * envelope * (1 + .13 * np.sin(xx / 8 + ph))
        body = (x >= left) & (x < right) & (y >= top[None]) & (y <= bottom[None])
        depth = bottom[None] - y
        a[body] = 4 if mid < 50 else 5
        middle = body & (depth < thick * .55 * envelope[None])
        a[middle] = 5 if mid < 50 else 6
        ribbon = body & (depth < 2.1) & (envelope[None] > .13)
        phase = (np.floor(x / 14 + .65 * np.sin(x / 21 + ph)) +
                 np.floor(depth * .8)).astype(int)
        a[ribbon] = (11 + phase[ribbon] % 6).astype(np.uint8)
        # A second, thinner exposed strata, only on the thickest decks.
        ledge = body & (depth > 4) & (depth < 5.7) & (np.sin(x / 19 + ph) > .1)
        a[ledge] = (11 + (phase[ledge] + 2) % 6).astype(np.uint8)
    return _clean(a)


@lru_cache(maxsize=1)
def _clouds():
    """High decks drift fast and pass behind the sun; the low ones are slow and pass in front of it."""
    high = _deck_layer([(-40, 150, 27, 12, .4), (229, 357, 18, 10, 2.1), (18, 101, 9, 4, 3.2)])
    mid = _deck_layer([(-25, 122, 59, 8, 4.3), (251, 350, 58, 8, 1.7)])
    low = _deck_layer([(-38, 142, 82, 5, 2.4), (219, 354, 87, 5, 5.0), (120, 214, 90, 3, 1.1)])
    return (high, 3.4), (mid, 2.0), (low, 1.0)


@lru_cache(maxsize=1)
def _sun():
    """The sun's rings and filaments around (0, 0), in a local grid that gets stamped wherever it has sunk to."""
    r0 = 72
    rng = np.random.default_rng(5)
    dy, dx = np.mgrid[-r0:r0 + 1, -r0:r0 + 1]
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)

    def glow(reach):
        # Terraces of light with a wandering edge, each row ending at its own length, like strokes laid down by hand.
        ph = rng.uniform(0, 2 * np.pi, 3)
        wander = 1.5 * np.sin(3 * theta + ph[0]) + np.sin(5 * theta + ph[1]) + .6 * np.sin(11 * theta + ph[2])
        rows = rng.normal(0, 1.1, 2 * r0 + 1)
        return (r > RADIUS) & (r < reach + wander + rows[dy + r0])

    rings = tuple((level, glow(reach)) for level, reach in ((5, 70), (6, 62), (7, 54), (8, 47), (9, 42)))
    # A clean limb two pixels thick, in long arcs that the palette cycles round: a shimmer, not a broken edge.
    core = np.full(r.shape, CLEAR, np.uint8)
    limb = (r > RADIUS) & (r < RADIUS + 2.3)
    core[limb] = (17 + np.floor((theta[limb] + np.pi) * 16 / (2 * np.pi)).astype(int) % 8).astype(np.uint8)
    # The disc in four flat terraces, gold above and reddening toward the horizon. Their edges wander
    # like the sky's, with a few long slivers of the next colour either side.
    level = (dy / RADIUS + .32) / .52 + .07 * np.sin(dx / 13.0 + 1) + .03 * np.sin(dx / 5.0)
    terrace = np.clip(np.floor(level).astype(int) + 1, 0, 3)
    for boundary in (-.32, .2, .72):
        row = int(round(boundary * RADIUS))
        for _ in range(5):
            off = int(rng.integers(-5, 6))
            x0 = int(rng.integers(-30, 31))
            half = int(rng.integers(4, 15))
            value = np.clip(np.floor((row + off) / RADIUS / .52 + .32 / .52 + (1 if off < 0 else 0)).astype(int) + 1, 0, 3)
            terrace[r0 + row + off, r0 + x0 - half:r0 + x0 + half] = value
    core[r <= RADIUS] = (41 + terrace[r <= RADIUS]).astype(np.uint8)
    return rings, core, r0


@lru_cache(maxsize=1)
def _sea():
    """The wave field on a strip twice the frame wide, so that each row can slide without showing a seam.
    Tone goes down shelf by shelf, each shelf with its own unrelated run of swell, so nothing can ever close
    into a contour; shelves and their strokes grow toward the viewer. Curving crests ride on top."""
    width = 2 * WIDTH
    rng = np.random.default_rng(11)
    xx = np.arange(width)
    a = np.full((HEIGHT, width), CLEAR, np.uint8)
    row = HORIZON
    while row < HEIGHT:
        depth = row - HORIZON
        d = depth / 84
        thick = 1 + int(d * 2.4 + rng.random() * (.7 + d))
        scale = 8 + 62 * d
        ph = rng.uniform(0, 2 * np.pi, 3)
        wave = (.9 * np.sin(2.2 * np.sqrt(depth + 1)) + np.sin(xx / (scale * 2.7) + ph[0])
                + .7 * np.sin(xx / scale + ph[1]) + .4 * np.sin(xx / (scale * .29) + ph[2]))
        shade = np.where(wave > 1.15, 2, np.where(wave > .1, 1, 0))
        for k in range(thick):
            if row + k < HEIGHT:
                a[row + k] = 25 + np.roll(shade, int(rng.integers(-2, 3)) * k)
        row += thick
    # Horizon has distance-compressed ripples, not a vector straight rule.
    for row in (96, 98, 100, 103):
        keep = np.sin(xx / 11 + row * 2.1) + .5 * np.sin(xx / 5) > -.3
        a[row, keep] = 28
    # Curving crests; nested leeward shelves supply dark structure beneath the light.
    rows = [105, 108, 112, 116, 121, 127, 134, 142, 151, 161, 173]
    for j, row in enumerate(rows):
        d = (row - HORIZON) / 84
        ph = rng.uniform(0, 2 * np.pi, 3)
        curve = np.rint(row + (1 + 2 * d) * np.sin(xx / (18 + 30 * d) + j * 1.7)
                        + d * np.sin(xx / 11 + j)).astype(int)
        gate = np.sin(xx / (7 + 12 * d) + ph[0]) + .55 * np.sin(xx / 29 + ph[1]) + .4 * np.sin(xx / (61 + 40 * d) + ph[2])
        for k in range(1 + int(d * 2)):
            yy = np.clip(curve + k, HORIZON, HEIGHT - 1)
            keep = gate > (-.15 if k == 0 else .4)
            a[yy[keep], xx[keep]] = 28 if k == 0 else 27
        yy = np.minimum(curve + 2 + int(d * 3), HEIGHT - 1)
        keep = gate > -.1
        a[yy[keep], xx[keep]] = 25
    return _clean(a)


def _ridge():
    return np.rint(np.interp(np.arange(WIDTH), RIDGE_X, RIDGE_Y, right=190)).astype(int)


@lru_cache(maxsize=1)
def _rock():
    """A single foreground basalt headland: cut into planes by joints at three scales, rim-lit only where a face
    actually turns to the sun. Returns it twice, as dusk leaves it and as the lantern lights it."""
    rng = np.random.default_rng(3)
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    ridge = _ridge()
    rock = y >= ridge[None]
    a = np.full(y.shape, CLEAR, np.uint8)
    a[rock] = 0
    for register, points in FACETS:
        mask = Image.new("L", (WIDTH, HEIGHT), 0)
        ImageDraw.Draw(mask).polygon(points, fill=255)
        a[(np.asarray(mask) > 0) & rock] = register
    canvas = Image.fromarray(a)
    draw = ImageDraw.Draw(canvas)
    for points in FISSURES:
        draw.line(points, fill=0)
    # The smallest cracks: short ticks along the fall of the strata, close under the skyline and gone in deep shadow.
    for _ in range(70):
        x0, y0 = int(rng.integers(0, 118)), int(rng.integers(52, 180))
        if 0 <= y0 - ridge[x0] < 34:
            draw.line((x0, y0, x0 + int(rng.integers(1, 4)), y0 + int(rng.integers(2, 6))),
                      fill=BASALT_MID if a[y0, x0] == 0 else 0)
    a = np.array(canvas)
    a[~rock] = CLEAR
    # Each slab's right-hand edge catches a little of the sun, in broken runs.
    edge = (a == BASALT_MID) & (np.roll(a, -1, axis=1) == 0)
    runs = rng.random(HEIGHT // 3 + 1) < .55
    a[edge & runs[y // 3]] = 1
    # Rim light: where the skyline falls away to the right it faces the sun. Broken into runs, never a neat outline.
    x_ = 0
    while x_ < 118:
        run = int(rng.integers(2, 9))
        for k in range(x_, min(118, x_ + run)):
            if ridge[min(k + 2, WIDTH - 1)] >= ridge[max(k - 2, 0)] + 1 and ridge[k] < HEIGHT:
                a[ridge[k], k] = 2
                if ridge[min(k + 1, WIDTH - 1)] > ridge[k] + 1:   # a steep step: light the whole riser
                    a[ridge[k]:min(HEIGHT, ridge[k + 1]), k] = 2
        x_ += run + int(rng.integers(1, 5))
    # A reef standing off the foot of the headland, in front of the path of light.
    reef = Image.new("L", (WIDTH, HEIGHT), 0)
    ImageDraw.Draw(reef).polygon(REEF, fill=255)
    reef = np.asarray(reef) > 0
    a[reef] = 0
    a[reef & (x > 133) & (x < 139) & (y > 175)] = BASALT_MID
    tops = np.argmax(reef, axis=0)
    for k in np.flatnonzero(reef.any(axis=0)):
        if tops[k + 2] > tops[k] and rng.random() < .7:
            a[tops[k], k] = 2
    # Pine and lantern are part of the headland, and the same light.
    _stamp(a, _sprite(PINE, {"#": 0, "+": 2}), PINE_AT[0], PINE_AT[1])
    # What the lantern reaches once it burns: the crag behind it and the ledge it stands on, and nothing under
    # the ledge, which is in its shadow. Warmth falls away in two steps, with a ragged edge.
    rows = rng.normal(0, 1.4, HEIGHT)
    reach = np.hypot(x - LAMP[0], y - LAMP[1]) + rows[y]
    above = y < 81 + rows[x // 5]
    lit = a.copy()
    lit[above & (reach < 30) & ((a == BASALT_MID) | (a == 1))] = LIT_DARK
    lit[above & (reach < 18) & (a == 0)] = LIT_DARK
    lit[above & (reach < 18) & ((a == BASALT_MID) | (a == 1))] = LIT_MID
    lit[above & (reach < 34) & (a == 2)] = LIT_RIM
    return a, lit


@lru_cache(maxsize=3)
def _pennant(frame):
    art = ("...#####", "########", ".....###"), ("......##", "#.######", "..######"), (".....###", "..######", "#####...")
    return _sprite("\n".join(art[frame]), {"#": 0})


@lru_cache(maxsize=2)
def _lantern(burning):
    return _sprite(LANTERN, {"#": 0, "u": LIT_MID, "f": FLAME, "e": LIT_RIM if burning else 0})


@lru_cache(maxsize=1)
def _junk():
    """A three-masted junk, bow to the left: fan-ribbed sails glowing with the sun through them, sky between
    the battens, a high stern with lit windows, a great rudder hung clear of the counter."""
    return _sprite(JUNK, {"#": 0, "=": 1, "c": CLOTH, "l": RIM_BOW, "r": RIM_STERN, "w": 42, "f": FLAME})


def _stamp(a, patch, ox, oy, floor=HEIGHT):
    """Lay a layer down with its corner at (ox, oy), clipped to the frame and to rows above `floor`."""
    h, w = patch.shape
    x0, x1 = max(0, ox), min(WIDTH, ox + w)
    y0, y1 = max(0, oy), min(floor, oy + h)
    if x0 >= x1 or y0 >= y1:
        return
    part = patch[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    region = a[y0:y1, x0:x1]
    region[part != CLEAR] = part[part != CLEAR]


@lru_cache(maxsize=64)
def _fragments(shelf):
    """Where the light breaks on one shelf of water: scattered fragments, crowding the middle of the path."""
    rng = np.random.default_rng(shelf + 512)
    n = int(rng.integers(4, 8))
    u = rng.uniform(-1, 1, n)
    u = np.sign(u) * np.abs(u) ** 1.6
    half = rng.uniform(.25, 1.0, n) ** 1.5 * (1.15 - .75 * np.abs(u))
    return u, half, rng.uniform(4.5, 10.5, n), rng.uniform(0, 2 * np.pi, n)


def _reflection(a, t):
    """Mirror column: not bars but fragments, each a little lens that tapers to a point at both ends and grows as
    its shelf of water flows toward the viewer. Each register principally owns one depth shelf, so the one-off
    flash can walk the depth addresses."""
    narrowing = 1 - .46 * float(_ease((t - 1.4) / 3.2))
    for row in range(HORIZON, HEIGHT):
        depth = row - HORIZON
        perspective = depth / (HEIGHT - HORIZON - 1)
        centre = CX + 2.4 * math.sin(depth * .17) + 3 * perspective
        halfwidth = (15 + 34 * perspective ** 1.35) * narrowing
        register = 29 + min(11, int(perspective * 11.99))
        if depth < 7:
            # Near the horizon the light is a crowded, mirror-bright band.
            if row % 3 != 1:
                reach = .8 * halfwidth * (.72 + .28 * math.sin(row * 2.3 + t * 3.1))
                a[row, int(round(centre - reach)):int(round(centre + reach)) + 1] = register
            continue
        w = 3.3 * math.sqrt(depth + 1) - 1.5 * t
        shelf = math.floor(w)
        f = w - shelf
        if f >= .44:
            continue
        taper = .3 + .7 * (1 - (2 * f / .44 - 1) ** 2)
        span = (2.5 + 9 * perspective) * (.55 + .45 * narrowing)     # how far a fragment may reach from its middle, in pixels
        fragments = _fragments(shelf)
        keep = math.ceil(len(fragments[0]) * (.2 + .8 * narrowing))
        lit = np.zeros(WIDTH, bool)
        for u, half, rate, phase in zip(*(f_[:keep] for f_ in fragments)):
            # Capped before the taper, and each row skewed a little, so no end ever stacks up flush like a sawn plank.
            reach = min(7, span * half * (1 + .3 * math.sin(rate * t + phase))) * taper
            mid = centre + u * halfwidth + (f - .22) * (7 if u > 0 else -7)
            x0, x1 = int(round(mid - reach)), int(round(mid + reach)) + 1
            lit[max(0, x0):max(0, min(WIDTH, x1))] = True
        # Fragments that happen to fuse into a plank get cracked apart again. The cracks are fixed per shelf and run
        # on a slant through its rows, so they neither leave a pinhole in a thick fragment nor a square-cut end.
        period = 14 + (shelf * 7) % 5
        lean = depth if shelf % 2 else -depth
        lit &= (np.arange(WIDTH) + shelf * 13 + lean) % period >= (1 if perspective < .5 else 2)
        a[row, lit] = register


def _mirror(a, junk, jx, t):
    """The junk upside down in the water, each row slipped sideways by the swell and torn more the further it falls
    from the hull, pressed into the sea as its darkest register: it cuts a dark notch out of the path of light."""
    solid = junk != CLEAR
    height, width = solid.shape
    xx = np.arange(width)
    for k in range(min(height, HEIGHT - WATERLINE)):
        # Below the hull, whole rows of sea come through it, running away down the reflection.
        if k >= 9 and (k + int(t * 5)) % 3 == 0:
            continue
        source = solid[height - 1 - k]
        slip = int(round((.6 + .12 * k) * math.sin(k * .83 + t * 7.0) + .3 * k ** .5 * math.sin(k * .31 - t * 3.1)))
        keep = source
        if k > 3:
            keep = source & ~(np.sin(xx / (2.6 + .05 * k) + k * 1.9 + t * 2.3) + .5 * np.sin(xx / 11 + k * .7) > 1.35 - .03 * k)
        xs = np.flatnonzero(keep) + jx + slip
        a[WATERLINE + k, xs[(xs >= 0) & (xs < WIDTH)]] = 25


def _wake(a, jx, step):
    """White water at the junk's stem, along the hull, and trailing astern to the right."""
    foam = lambda x, rate, pace: math.sin(x * rate + step * pace) + .6 * math.sin(x * rate * .37 - step * pace * .6)
    for x_ in range(jx - 3, jx + 66):
        if 0 <= x_ < WIDTH and foam(x_, .71, .9) > -.2:
            a[WATERLINE, x_] = 28
    for ox, oy, register in ((-4, 0, 28), (-3, -1, 28), (-2, -1, 28), (-1, -2, 28), (-5, 1, 27), (-3, 1, 28), (-7, 2, 26), (-4, 2, 27)):
        x_, y_ = jx + ox, WATERLINE + oy
        if 0 <= x_ < WIDTH and (ox + step) % 5:
            a[y_, x_] = register
    for start, stop, row, register in ((64, 80, 0, 28), (70, 94, 1, 28), (86, 104, 0, 27), (90, 120, 2, 27), (108, 138, 1, 26), (118, 152, 3, 26)):
        for x_ in range(jx + start, jx + stop):
            if 0 <= x_ < WIDTH and foam(x_ - jx, .43, .5) > -.45:
                a[WATERLINE + row, x_] = register


@lru_cache(maxsize=1)
def _glow_edge():
    return np.random.default_rng(9).normal(0, 1.1, HEIGHT)


def _flame(step):
    """A lamp lit by hand: it catches, gutters, catches again, then holds with a small tremble."""
    if step < LIGHTING:
        return 0.0
    keys = (.35, .08, .6, .22, .85, .55)
    if step - LIGHTING < len(keys):
        return keys[step - LIGHTING]
    return 1 - .16 * ((step * 7) % 5 == 1) - .07 * ((step * 3) % 4 == 2)


def index_map(step):
    t = step / RATE
    a = _sky().copy()
    (high, v_high), (mid, v_mid), (low, v_low) = _clouds()
    _stamp(a, high, -PAD + int(v_high * t), 0)
    _stamp(a, mid, -PAD + int(v_mid * t), 0)

    cy = CY0 + int(round(66 * float(_ease(t / 5.0)) ** 2))
    rings, core, r0 = _sun()
    y0, y1 = max(0, cy - r0), min(HORIZON, cy + r0 + 1)
    x0, x1 = max(0, CX - r0), min(WIDTH, CX + r0 + 1)
    if y0 < y1:
        region = a[y0:y1, x0:x1]
        crop = (slice(y0 - cy + r0, y1 - cy + r0), slice(x0 - CX + r0, x1 - CX + r0))
        for level, ring in rings:
            mask = ring[crop]
            v = region[mask]
            # Light only ever lifts the sky: a darker register rises, and so does a latent star (≥ 45), which would
            # otherwise sit in the glow as a black hole. The inner rings wash out any cloud behind them too.
            keep = (v >= level) & (v <= 10) | ((level < 8) & (v >= 11) & (v <= 16))
            region[mask] = np.where(keep, v, level)
        _stamp(a, core, CX - r0, cy - r0, HORIZON)
    _stamp(a, low, -PAD + int(v_low * t), 0)
    # Heat shimmer: the rows nearest the horizon are rolled sideways, a scanline raster effect.
    for row in range(HORIZON - 9, HORIZON):
        reach = (row - (HORIZON - 9)) / 9
        a[row] = np.roll(a[row], int(round(1.6 * reach * math.sin(row * .95 + t * 11))))

    # Each row of the sea slides by its own whole number of pixels: slow at the horizon, quick up close.
    sea = _sea()
    for row in range(HORIZON, HEIGHT):
        d = (row - HORIZON) / (HEIGHT - HORIZON - 1)
        shift = int(t * (1.2 + 15 * d * d))
        a[row] = sea[row, shift:shift + WIDTH]
    _reflection(a, t)

    # The junk crosses the sun from end to end of the film: bow into the disc on the first frame,
    # centred as the pulse lands, its high stern leaving the last sliver on the last.
    junk = _junk()
    jx = int(round(144 + 14 * (3 - t)))
    jy = WATERLINE - junk.shape[0] + 1 + (step // 7) % 2
    _mirror(a, junk, jx, t)
    _wake(a, jx, step)
    _stamp(a, junk, jx, jy, WATERLINE + 1)
    # The pennant at the main masthead, streaming ahead of a following wind.
    _stamp(a, _pennant((step // 2) % 3), jx + 28, jy - 1)

    flame = _flame(step)
    rock, lit = _rock()
    if flame > 0:
        # The lantern's glow in the air, stamped only on sky and cloud; the rock's own light is in `lit`.
        lx, ly = LAMP
        grow = flame ** .5
        dy, dx = np.mgrid[:HORIZON, :WIDTH]
        reach = np.hypot(dx - lx, dy - ly) + _glow_edge()[dy]
        a[:HORIZON][reach < 17 * grow] = GLOW_FAR
        a[:HORIZON][reach < 9 * grow] = GLOW_NEAR
    _stamp(a, lit if flame > 0 else rock, 0, 0)
    _stamp(a, _lantern(flame >= .8), LANTERN_AT[0], LANTERN_AT[1])
    # Surf working along the foot of the headland, where it meets the water.
    for x_ in range(36, 118):
        y_ = int(np.interp(x_, RIDGE_X, RIDGE_Y))
        sway = math.sin(x_ * .43 + step * .8) + .6 * math.sin(x_ * .13 - step * .35)
        if HORIZON < y_ - 1 < HEIGHT and sway > -.25:
            a[y_ - 1, x_] = 28
        if HORIZON < y_ - 2 < HEIGHT and sway > .75:
            a[y_ - 2, x_] = 28
    for x_ in range(120, 156):
        sway = math.sin(x_ * .61 + step * .7) + .6 * math.sin(x_ * .19 - step * .4)
        y_ = 179
        if sway > -.1 and a[y_, x_] != 0:
            a[y_, x_] = 28
    return a


@lru_cache(maxsize=1)
def _keys():
    """Hand-tuned, hue-shifted register ramps: gold / crimson / nightfall with an ember on the horizon."""
    skies = [
        "402338 59293e 793242 a14749 c4604b df8053 efae68 fbdc91",
        "1a1035 2c1542 481b50 6e2658 a03560 cc5265 e8805f f7b070",
        "080e24 101831 19223f 272f55 3f3a66 6f4a70 ae5f66 e8915c",
    ]
    clouds = ["8f3445 ba4644 e86a43 ff9e52 ffd681 e77b49",
              "6b2650 96315f c8485f ee7a58 f9b077 c0505c",
              "3a2a4a 5a3450 8a4450 c05f4f e08a5a 8a4450"]
    rim = ["ffb04a ffc65a ffe18a fff3bc fff0b0 ffd475 ffad49 ff9a3f",
           "f2683c ff8a4a ffb870 ffd79a ffc887 ff9a55 f5733f e6552f",
           "e85a30 f0703a ff9a55 ffb870 ff9a55 f0703a e85a30 dc4a2c"]
    seas = ["211c31 3d2840 663b4b 956050",
            "10132b 251d41 472957 744073",
            "070e21 101d34 1a304a 305574"]
    reflection = ["a8443e cf6140 ee8a48 ffb858 ffdc86 fff2ae fff8d2 ffeaa0 ffd07a f59a4d d9703f bb5240",
                  "b8443c dc5c3c f07a42 ff9a55 ffba78 ffd8a4 ffe6bc ffcf94 ffb070 f5864c e2643f c84c3c",
                  "84303a a83a32 cc4f32 e8683a ff9650 ffb878 ffc98a f8a860 f08a48 dc6a3c bc4c36 9c3a34"]
    disc = ["ffdc78 ffb23e ff7a2e e8482a",
            "ffcf70 ff9a3e f5622f d83a30",
            "ffb868 ff8a40 f0582e d63c2c"]
    cloth = ["6e2a2c", "4e1c36", "22152e"]
    bright = ["ffc27a", "ff9a5e", "e8783e"]
    dim = ["8a4038", "6a2a40", "3a2238"]
    palettes = []
    for i in range(3):
        p = np.zeros((REGISTERS, 3), dtype=np.float64)
        p[:3] = _rgb(["0a0a14 2c1f30 b0694c", "060612 20162f 8e4c5c",
                      "040813 101a2c 4c6c84"][i])
        p[3:11] = _rgb(skies[i])
        p[11:17] = _rgb(clouds[i])
        p[17:25] = _rgb(rim[i])
        p[25:29] = _rgb(seas[i])
        p[29:41] = _rgb(reflection[i])
        p[41:45] = _rgb(disc[i])
        p[CLOTH] = _rgb(cloth[i])
        p[RIM_BOW] = _rgb(bright[i])
        p[RIM_STERN] = _rgb(dim[i])
        p[BASALT_MID] = _rgb(["1a1422", "120e24", "0a1220"][i])
        palettes.append(p)
    return tuple(palettes)


def palette(step):
    """One global colour table, including the depth-addressed flash."""
    t = step / RATE
    gold, crimson, night = _keys()
    if t < 1.8:
        u = float(_ease(t / 1.8))
        p = gold * (1 - u) + crimson * u
    else:
        u = float(_ease((t - 1.8) / 2.6))
        p = crimson * (1 - u) + night * u
    # The rim light changes sides as the junk passes in front of the sun.
    swap = float(_ease((t - 2.4) / 1.2))
    bow, stern = p[RIM_BOW].copy(), p[RIM_STERN].copy()
    p[RIM_BOW] = bow * (1 - swap) + stern * swap
    p[RIM_STERN] = stern * (1 - swap) + bow * swap
    # Integer rotations only. Opposing directions and unrelated divisors keep
    # the clouds, angular filaments and water from breathing in lockstep.
    p[11:17] = np.roll(p[11:17].copy(), -(step // 3), axis=0)
    p[17:25] = np.roll(p[17:25].copy(), step, axis=0)
    p[29:41] = np.roll(p[29:41].copy(), step // 2, axis=0)
    # Sea shadow registers remain ordered: only the reflection cycles brightly.
    pulse = math.exp(-((t - 3.0) / .145) ** 2)
    p[17:25] += (np.array([255, 249, 219]) - p[17:25]) * pulse * .92
    p[41:45] += (np.array([255, 226, 150]) - p[41:45]) * pulse * .4
    # The pulse comes through the sail cloth, like light through a paper screen.
    p[CLOTH] += (np.array([214, 96, 52]) - p[CLOTH]) * pulse
    for j in range(12):
        arrival = 3.06 + 1.04 * (j / 11) ** .72
        flash = math.exp(-((t - arrival) / (.11 + .045 * j / 11)) ** 2)
        p[29 + j] += (np.array([255, 244, 205]) - p[29 + j]) * flash
    # At landfall, the existing ocean shelves catch the light once, then return to blue.
    landfall = math.exp(-((t - 4.13) / .19) ** 2)
    p[26:29] += (np.array([[70, 48, 58], [130, 80, 70], [230, 160, 110]])
                 - p[26:29]) * landfall
    # Stars: aliases of their sky register until their moment; the brightest come first. Each register
    # breathes at its own rate, so nothing twinkles in step.
    light = _rgb("6f86a8 a9c1d6 eef2e0")
    for band in range(3):
        for tier, onset in enumerate((4.05, 3.85, 3.65)):
            reg = STARS + band * 3 + tier
            s = float(_ease((t - onset) / .9))
            breath = 1 - .2 * (.5 + .5 * math.sin(step * (.9 + .23 * reg) + reg * 2.1))
            p[reg] = p[3 + band] * (1 - s) + light[tier] * breath * s
    # The lantern, and the rock it warms.
    flame = _flame(step)
    p[LIT_DARK] = p[0] * (1 - flame) + _rgb("24151e")[0] * flame
    p[LIT_MID] = p[BASALT_MID] * (1 - flame) + _rgb("4e2e29")[0] * flame
    p[LIT_RIM] = p[2] * (1 - flame) + _rgb("e89a58")[0] * flame
    # A low flame is red; only a full one goes pale.
    p[FLAME] = np.array([np.interp(flame, (0, .5, 1), (p[0, c], (200, 92, 40)[c], (255, 230, 160)[c])) for c in range(3)])
    p[GLOW_NEAR] = p[7] * (1 - flame) + _rgb("a05a58")[0] * flame
    p[GLOW_FAR] = p[7] * (1 - flame) + _rgb("5c4268")[0] * flame
    # Round registers BEFORE lookup. Every output pixel is one of these RGBs.
    return np.clip(np.rint(p), 0, 255).astype(np.uint8)


def pixel(t):
    step = min(59, max(0, int(float(t) * RATE + 1e-7)))
    native = palette(step)[index_map(step)]
    enlarged = np.repeat(np.repeat(native, SCALE, axis=0), SCALE, axis=1)
    return surface(enlarged.astype(np.float32) / 255.0)
