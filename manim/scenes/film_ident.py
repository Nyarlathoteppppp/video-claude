"""The studio's mark, five seconds with no place on the film's clock: a block of Shoushan stone comes down on bright xuan paper and leaves Clawd in cinnabar, with 小克影业 printed under it.

    python film_ident.py tile                                                # the paper's fibres, into ink/film_out/film/head/maps/
    python film_ident.py maps                                                # the stone's face and what it leaves there, with proofs on flat paper
    Blender -b --factory-startup -P film_ident.py -- h1 --pct 50 --samples 96     # a still at its second: h1 hanging, h2 down, h3 the finished print
    Blender -b --factory-startup -P film_ident.py -- seq --pct 50 --samples 48    # all five seconds into head/seq/; --from N --to M only those frames
    python film_ident.py mp4                                                 # those frames, with grain, as head/ident_low.mp4
    python film_ident.py grade h1                                            # the grain, from raw/h1.png to h1.png

The world is in centimetres, x to the right of the frame, y away from the camera, z up, the sheet at z = 0. The print and the stone's face share the same x and y, so what reads right on the paper is mirrored on the stone by itself. The shape on the face is read out of ink_mac.CLAWD, its cell one wide and two tall as ink_mac draws it.

The camera starts low and rises as the stone leaves, because no one camera both sees the face of a hanging stone and reads a print square on. The stone hangs turned half way round on its own axis, so that the crab stands upright on its face for the first frame, and it comes round as it comes down: tipped toward a camera with no turn it would show the crab on its back. pose and cam_at hold the five seconds, so that the stills are frames of the shot.
"""
import ast
import math
import os
import sys

import numpy as np

try:
    import bpy
except ImportError:
    bpy = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from film_press_maps import fibres, png16, sm, spectral  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "ink", "film_out", "film", "head")
MAPS = os.path.join(OUT, "maps")

SIDE, TALL = 3.0, 5.4            # the stone
RECESS = .09                     # how deep the face is cut round what prints
FRAME = (.022, .072)             # the thin frame on the face: how far in it starts and ends, of the side
CRAB_W = .77                     # how wide CLAWD's grid stands, of the side: its claws, one column in from each end of the grid, come to 80% of the width inside the frame
AREA = (-5.0, 5.0, -6.5, 3.5)    # what the print's maps cover: x0, x1, y0, y1
PX = 400                         # samples per centimetre of those maps
TILE_N, TILE_CM = 4096, 15.0     # the fibres repeat this far apart, wider than the last frame
TEXT = "小克影业"
EM, PITCH, TEXT_Y = .42, .72, -2.36    # how big, how far apart their middles, where their middle stands
BUMP = .00133                    # how high the fibres stand
DEEP, LEFT = .10, .016           # how far the stone sinks the paper, and how much of that stays

SUN_AZ, SUN_EL, SUN_E, SUN_COLOUR = 100.0, 46.0, 2.0, (1.0, .84, .64)      # from the frame's left and a little behind it: what the stone throws lies beside it and comes in under it as it falls
SUN_SIZE = 7.0                    # degrees across: a low sun through haze, so that what hangs high throws a soft edge
FIBRE = (1.8, 1.2)                # how much of the fibres to show, low and high: at a low angle they are foreshortened nearly out of sight
SKY_E, SKY_COLOUR = .68, (1.0, .955, .90)
LOW = dict(el=24.0, d=16.2, target=(.9, 0.0, 3.2), lens=40.0, stop=.62)
HIGH = dict(el=80.0, d=15.2, target=(0.0, -.45, 0.0), lens=40.0, stop=.5)
HANG = dict(at=(0.0, -.4, 3.0), tilt=72.0, yaw=172.0, roll=-2.0)
PANES = ((-6.0, 56.0, 60.0, 5.0, 8.0), (3.5, 56.0, 60.0, 3.5, 8.0))     # the window the polish sees, up behind the stone, where the far end of its top face looks: x, y, z, half width, half height
WINDOW_E, ROOM_E = 3.6, .16
LAND, RISE, GONE = .7, 1.2, 2.0
CRANE = (1.2, 2.4)
PUSH = .03
FPS = 24
SHOTS = dict(h1=0.0, h2=.75, h3=3.5)
NAME_IN = (2.4, 2.9)             # the name comes up on the paper
PULL = .1                        # how long the lens takes to come from the stone down to the print, from the moment it lifts
SHUTTER = (.5, .3)               # of a frame, as the stone falls and from the crane on: at the crane's fastest the wider one drags the print 7.6 px at 960 wide, this one 4.6


def clawd():
    """The crab's rows, read from ink_mac without importing it."""
    with open(os.path.join(HERE, "ink_mac.py"), encoding="utf-8") as f:
        for node in ast.parse(f.read()).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CLAWD":
                return ast.literal_eval(node.value)


def lerp(a, b, u):
    return a + (b - a) * u


def pose(t):
    """The stone at t: where the middle of its face is, and how far it has come round from how it hangs. None once it has left the frame."""
    if t >= GONE:
        return None
    if t < LAND:
        u = (t / LAND) ** 2.6        # slowly, then fast
        return (lerp(HANG["at"][0], 0.0, u), lerp(HANG["at"][1], 0.0, u), lerp(HANG["at"][2], 0.0, u)), u
    if t < RISE:
        return (0.0, 0.0, -sunk(t)), 1.0
    return (0.0, 0.0, lerp(-sunk(RISE - .001), 16.0, float(sm((t - RISE) / (GONE - RISE))))), 1.0


def sunk(t):
    """How far down the paper is under the stone at t: a little past where it rests, the moment it lands."""
    if t < LAND:
        return 0.0
    if t < RISE:
        return DEEP * (1 + .22 * math.exp(-(t - LAND) / .05))
    return lerp(DEEP, LEFT, float(sm((t - RISE) / .35)))


def cam_at(t):
    u = float(sm((t - CRANE[0]) / (CRANE[1] - CRANE[0])))
    el = math.radians(lerp(LOW["el"], HIGH["el"], u))
    d = lerp(LOW["d"], HIGH["d"], u) * (1 - PUSH * max(t - CRANE[1], 0) / (5.0 - CRANE[1]))
    target = np.array([lerp(a, b, u) for a, b in zip(LOW["target"], HIGH["target"])])
    loc = target + d * np.array((0.0, -math.cos(el), math.sin(el)))
    return dict(loc=tuple(loc), target=tuple(target), lens=LOW["lens"], stop=lerp(LOW["stop"], HIGH["stop"], u))


def focus_on(t):
    """What the lens holds sharp at t: the stone as it falls and rests; from the moment it lifts, the print, so that the print is sharp as soon as it shows and the stone goes up out of focus."""
    rest = np.array(pose(min(t, RISE))[0]) + (0.0, 0.0, .5)
    return rest * (1 - float(sm((t - RISE) / PULL)))


# ---------------------------------------------------------------- the maps

def tile():
    """Xuan paper up close, over TILE_CM and wrapping: red the height, green the long fibres, blue the few dark bits of bark. The screen it was lifted on shows in its fine lines."""
    from scipy.ndimage import gaussian_filter as gf
    os.makedirs(MAPS, exist_ok=True)
    n = TILE_N
    felt = fibres(n, 210000, 501, length=(24, 150), width=(1, 1), wiggle=.3)
    long_ = fibres(n, 15000, 502, length=(150, 760), width=(1, 2), wiggle=.14)
    bundles = fibres(n, 1200, 503, length=(240, 1000), width=(2, 3), wiggle=.1)
    bark = fibres(n, 520, 504, length=(30, 220), width=(1, 2), wiggle=.22)
    wobble = spectral(n, 3.2, 506)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    laid = np.sin(2 * np.pi * (yy * 136 + .35 * wobble))
    chain = np.exp(-((((xx * 5 + .02 * wobble) % 1) - .5) / .012) ** 2)
    h = gf(felt, .6, mode="wrap") * .7 + gf(long_, .7, mode="wrap") * .9 + gf(bundles, 1.0, mode="wrap") + spectral(n, 2.2, 505) * .11 + laid * .05 + chain * .06
    h = (h - h.mean()) / h.std()
    strands = np.clip(gf(long_ + bundles * 1.2, .8, mode="wrap"), 0, 1)
    dark = gf(bark, .6, mode="wrap")
    np.save(os.path.join(MAPS, "tile.npy"), np.stack([h, strands, dark], -1).astype(np.float32))
    png16(os.path.join(MAPS, "tile.png"), np.stack([h / 8 + .5, strands, dark], -1))
    print("tile", flush=True)


def face_shape():
    """The face as the knife left it: how far inside what prints each point is, in millimetres, over the side of the stone, top row the top of the print. The crab is the real pixels, its corners square and its edges cut by hand."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import distance_transform_edt as edt
    n = int(SIDE * PX)
    im = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(im)
    a, b = round(FRAME[0] * n), round(FRAME[1] * n)
    d.rectangle((a, a, n - 1 - a, n - 1 - a), fill=255)
    d.rectangle((b, b, n - 1 - b, n - 1 - b), fill=0)
    rows = clawd()
    cols = len(rows[0])
    u = CRAB_W * n / cols
    x0, y0 = (n - u * cols) / 2, (n - 2 * u * len(rows)) / 2
    for j, row in enumerate(rows):
        for i, c in enumerate(row):
            if c == "#":
                d.rectangle((round(x0 + i * u), round(y0 + j * 2 * u), round(x0 + (i + 1) * u) - 1, round(y0 + (j + 1) * 2 * u) - 1), fill=255)
    m = np.asarray(im) > 0
    mm = PX / 10
    sd = (edt(m) - edt(~m)) / mm
    sd = sd + .045 * spectral(n, 2.8, 601) + .022 * spectral(n, 1.3, 602)
    # where the stone has chipped: the thin frame broken in two places, and small bites all round
    r = np.random.default_rng(603)
    edge = np.argwhere(np.abs(sd) < .03)
    chips = [(.045, .72, 1.05), (.955, .27, .95), (.31, .045, .55), (.74, .955, .5)]
    for i in r.integers(0, len(edge), 34):
        chips.append((edge[i, 1] / n, edge[i, 0] / n, r.uniform(.07, .22)))
    cut = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(cut)
    for cx, cy, rad in chips:
        # stone breaks in flakes: a few straight sides, longer one way than the other
        k, turn, long_ = int(r.integers(5, 9)), r.uniform(0, np.pi), r.uniform(1.0, 1.7)
        ang = np.sort(r.uniform(0, 2 * np.pi, k))
        reach = rad * mm * r.uniform(.5, 1.15, k)
        px, py = np.cos(ang) * reach * long_, np.sin(ang) * reach / long_
        d.polygon([(cx * n + a * np.cos(turn) - b * np.sin(turn), cy * n + a * np.sin(turn) + b * np.cos(turn)) for a, b in zip(px, py)], fill=255)
    cm = np.asarray(cut) > 0
    return np.minimum(sd, (edt(~cm) - edt(cm)) / mm)


def maps():
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import distance_transform_edt as edt
    from scipy.ndimage import gaussian_filter as gf
    from scipy.ndimage import map_coordinates
    os.makedirs(MAPS, exist_ok=True)
    x0, x1, y0, y1 = AREA
    n = int((x1 - x0) * PX)
    mm = PX / 10
    sd = face_shape()
    f = sd.shape[0]
    top = int((y1 - SIDE / 2) * PX)
    left = int((-SIDE / 2 - x0) * PX)
    d = np.full((n, n), -9.0, np.float32)
    d[top:top + f, left:left + f] = sd
    outer = edt(d <= -8.9).astype(np.float32) / mm          # past the stone's edge, how far from it
    d[d <= -8.9] = -outer[d <= -8.9] - 2.0

    # the paper's tooth under the print, the same fibres the shader lays there
    fib = np.load(os.path.join(MAPS, "tile.npy"), mmap_mode="r")
    rr, cc = np.mgrid[0:n, 0:n].astype(np.float32)
    wx, wy = x0 + (cc + .5) / PX, y1 - (rr + .5) / PX
    tc, tr = (wx / TILE_CM % 1) * TILE_N - .5, (1 - wy / TILE_CM % 1) * TILE_N - .5
    at = lambda k: map_coordinates(np.asarray(fib[..., k]), [tr, tc], order=1, mode="grid-wrap").astype(np.float32)
    height, strands = at(0), at(1)
    tooth = height - gf(height, 12)
    tooth /= tooth.std()

    wide, mid, fine = spectral(n, 3.0, 611), spectral(n, 2.1, 612), spectral(n, 1.2, 613)
    # the stone did not come down quite true: its upper right bears least; and the pad was not even
    plane = (wx / (SIDE / 2) + wy / (SIDE / 2)) * .12
    dry = sm((mid * .55 + wide * .35 + plane + .02 * fine - .62) / .55)
    rim = np.exp(-np.maximum(d, 0) / .24)                  # pressed out to the edge of what prints, the paste stands thicker
    load = .86 - .40 * dry + .1 * fine + .24 * rim * (1 - .6 * dry)
    touch = sm((tooth * .17 + load - .44) / .11)
    inside = sm((d + .05 * np.clip(tooth, -2.2, 2.2) + .02 * fine) / .05)
    red = inside * touch * np.clip(.93 + .09 * fine, 0, 1)
    # it creeps a little way along the fibres it touches, and the oil a little further
    out = edt(red < .5).astype(np.float32) / mm
    creep = sm((strands * .95 - out / .34 - .07) / .13) * (out < .7) * .62
    red = np.maximum(red, creep * (1 - inside))
    halo = np.exp(-out / .75) * (out > 0) * sm((3.0 - out) / 2)
    thick = inside * touch * np.clip(.42 + .45 * rim + .22 * mid - .3 * dry, 0, 1)

    # cinnabar: never one colour. It goes deep where it lies thick and to orange where it is thin, and it is ground, so it has a grain
    g = gf(np.random.default_rng(614).normal(0, 1, (n, n)).astype(np.float32), .9)
    m = np.clip(.5 + .24 * mid + .12 * fine + .32 * g / g.std() * .25 + .34 * (dry - .3) - .3 * rim, 0, 1)[..., None]
    low, middle, high = [np.array(c, np.float32) for c in ((144, 26, 20), (182, 42, 27), (218, 66, 34))]
    colour = np.where(m < .5, low + (middle - low) * m / .5, middle + (high - middle) * (m - .5) / .5)
    Image.fromarray(np.uint8(np.clip(colour, 0, 255))).save(os.path.join(MAPS, "ink_colour.png"))

    # the name, set in Song and printed, the ink catching the paper's grain
    faces = "/System/Library/Fonts/Supplemental/Songti.ttc"
    font = ImageFont.truetype(faces, int(EM * PX), index=6)
    tm = Image.new("L", (n, n), 0)
    dr = ImageDraw.Draw(tm)
    for i, c in enumerate(TEXT):
        cx = (i - (len(TEXT) - 1) / 2) * PITCH
        dr.text(((cx - x0) * PX, (y1 - TEXT_Y) * PX), c, font=font, fill=255, anchor="mm")
    tm = np.asarray(tm, np.float32) / 255
    text = sm((gf(tm, 1.1) - .4 + tooth * .035) / .22) * sm((tooth * .1 + .62) / .1) * .97

    png16(os.path.join(MAPS, "ink_cover.png"), np.stack([red, text, thick], -1))
    png16(os.path.join(MAPS, "relief.png"), np.stack([gf(sm((d + .02) / .2), 4), gf(tm, 2.2), halo], -1))

    # the stone's own face: its height, the paste on what stands, the stain the pad left in what is cut away
    ff = f
    knife = gf(np.random.default_rng(615).normal(0, 1, (ff, ff)).astype(np.float32), (1.5, 15)) + gf(np.random.default_rng(616).normal(0, 1, (ff, ff)).astype(np.float32), (15, 1.5))
    knife /= knife.std()
    stand = sm(sd / .22)
    fy = top
    part = lambda a: a[fy:fy + ff, left:left + ff]
    paste = sm((sd + .05) / .1) * np.clip(.93 + .1 * part(mid) - .22 * part(dry), 0, 1)
    stain = np.exp(-np.maximum(-sd, 0) / .55) * (1 - stand) * sm(part(fine) * .4 + part(mid) * .5 + .28)
    png16(os.path.join(MAPS, "face.png"), np.stack([np.clip(stand * (1 - .03 * part(fine) ** 2) + (1 - stand) * (.06 + .035 * knife), 0, 1), paste, np.clip(stain, 0, 1)], -1))

    # proofs: what the print comes to on flat paper, at the size the shot shows it and at three times that
    base = np.array((250, 245, 233), np.float32)
    light = np.clip(1 + .012 * tooth + .02 * strands, 0, 1.05)[..., None]
    p = base * light
    p = p * (1 - .05 * halo[..., None]) + np.array((250, 228, 180), np.float32) * .05 * halo[..., None]
    p += (colour - p) * red[..., None]
    p += (np.array((26, 22, 20), np.float32) - p) * text[..., None]
    a, b = int((y1 + .0 - 1.9) * PX), int((y1 - TEXT_Y + .6) * PX)
    l, r_ = int((-2.0 - x0) * PX), int((2.0 - x0) * PX)
    crop = Image.fromarray(np.uint8(np.clip(p[a:b, l:r_], 0, 255)))
    crop.save(os.path.join(MAPS, "proof_full.png"))
    crop.resize((int(crop.width * 12.67 / 40), int(crop.height * 12.67 / 40)), Image.LANCZOS).save(os.path.join(MAPS, "proof_shot.png"))
    print("maps", red[top:top + f, left:left + f].mean(), flush=True)


# ---------------------------------------------------------------- the world

def image_at(nt, path, vector, colourspace="Non-Color", extension="REPEAT"):
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(path, check_existing=True)
    t.image.colorspace_settings.name = colourspace
    t.extension = extension
    nt.links.new(vector, t.inputs["Vector"])
    return t.outputs["Color"]


def split(nt, colour):
    s = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(colour, s.inputs[0])
    return s.outputs


def box_coords(nt, x0, x1, y0, y1):
    """Object coordinates to the 0-1 of a map laid over that box."""
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / (x1 - x0), 1 / (y1 - y0), 1)
    mp.inputs["Location"].default_value = (-x0 / (x1 - x0), -y0 / (y1 - y0), 0)
    nt.links.new(tc.outputs["Object"], mp.inputs[0])
    return mp.outputs[0]


def paper_material(K):
    """Xuan paper, nearly white and warm, with no gloss to veil it however low the camera. Where the stone came down, the paste lies on it with the paper's own fibres showing through and a little oily sheen. Two values, "printed" and "named", say whether the print is there yet and how far the name has come up."""
    m, nt, b = K.new_mat("paper")
    fib = split(nt, image_at(nt, os.path.join(MAPS, "tile.png"), K.obj_coords(nt, (1 / TILE_CM, 1 / TILE_CM, 1))))
    h, strands, dark = fib[0], fib[1], fib[2]
    grain = K.nmath(nt, "MULTIPLY", K.nmath(nt, "SUBTRACT", h, .5), 8.0)
    fibre = nt.nodes.new("ShaderNodeValue")
    fibre.name = "fibre"
    fibre = fibre.outputs[0]
    shade = K.nmath(nt, "MULTIPLY", fibre, K.nmath(nt, "ADD", K.nmath(nt, "MULTIPLY", grain, .011), K.nmath(nt, "SUBTRACT", K.nmath(nt, "MULTIPLY", strands, .022), K.nmath(nt, "MULTIPLY", dark, .10))))
    tint = nt.nodes.new("ShaderNodeVectorMath")
    tint.operation = "SCALE"
    tint.inputs[0].default_value = (.925, .893, .825)
    nt.links.new(K.nmath(nt, "ADD", 1.0, shade), tint.inputs["Scale"])
    printed, named = nt.nodes.new("ShaderNodeValue"), nt.nodes.new("ShaderNodeValue")
    printed.name, named.name = "printed", "named"
    box = box_coords(nt, *AREA)
    cover = split(nt, image_at(nt, os.path.join(MAPS, "ink_cover.png"), box, extension="EXTEND"))
    shape = split(nt, image_at(nt, os.path.join(MAPS, "relief.png"), box, extension="EXTEND"))
    ink = image_at(nt, os.path.join(MAPS, "ink_colour.png"), box, "sRGB", "EXTEND")
    on = printed.outputs[0]
    red, words = K.nmath(nt, "MULTIPLY", cover[0], on), K.nmath(nt, "MULTIPLY", cover[1], named.outputs[0])
    thick, pressed = K.nmath(nt, "MULTIPLY", cover[2], on), K.nmath(nt, "MULTIPLY", shape[0], on)
    colour = K.nmix(nt, K.nmath(nt, "MULTIPLY", K.nmath(nt, "MULTIPLY", shape[2], on), .05), tint.outputs[0], (.95, .77, .46, 1))
    colour = K.nmix(nt, red, colour, ink)
    colour = K.nmix(nt, words, colour, (.0103, .0080, .0070, 1))
    relief = K.nmath(nt, "MULTIPLY", K.nmath(nt, "MULTIPLY", grain, BUMP), fibre)
    relief = K.nmath(nt, "ADD", K.nmath(nt, "MULTIPLY", relief, K.nmath(nt, "SUBTRACT", 1.0, K.nmath(nt, "MULTIPLY", pressed, .45))), K.nmath(nt, "SUBTRACT", K.nmath(nt, "MULTIPLY", thick, .0014), K.nmath(nt, "MULTIPLY", pressed, .0125)))
    relief = K.nmath(nt, "SUBTRACT", relief, K.nmath(nt, "MULTIPLY", K.nmath(nt, "MULTIPLY", shape[1], named.outputs[0]), .0036))
    nt.links.new(colour, b.inputs["Base Color"])
    nt.links.new(K.nbump(nt, relief, 1.0, 1.0), b.inputs["Normal"])
    nt.links.new(K.nmixf(nt, red, .84, .5), b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = .1
    b.inputs["Sheen Weight"].default_value = 0.0
    return m


def stone_shader(nt, b, K, cut=False):
    """Shoushan stone, cut and polished: a dark honey wax with light a little way into it, the cloud in it faint, soft specks of cotton in the body of it and fine grains of mineral in drifts. Where the knife has been it is darker and has no polish."""
    tc = nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value, noise.inputs["Detail"].default_value = .7, 3.0
    nt.links.new(tc, noise.inputs["Vector"])
    centred = nt.nodes.new("ShaderNodeVectorMath")
    centred.operation = "SUBTRACT"
    nt.links.new(noise.outputs["Color"], centred.inputs[0])
    centred.inputs[1].default_value = (.5, .5, .5)
    warp = nt.nodes.new("ShaderNodeVectorMath")
    warp.operation = "ADD"
    nt.links.new(tc, warp.inputs[0])
    nt.links.new(centred.outputs[0], warp.inputs[1])
    p = warp.outputs[0]
    cloud = K.nrange(nt, K.nnoise(nt, p, .55, 5.0, .55), .38, .64)
    colour = K.nmix(nt, cloud, (.150, .082, .033, 1), (.245, .146, .060, 1))
    # the cotton: soft wisps, and finer flecks drawn out one way
    wisps = K.nrange(nt, K.nnoise(nt, p, 2.2, 6.0, .62), .54, .72)
    colour = K.nmix(nt, K.nmath(nt, "MULTIPLY", wisps, .13), colour, (.42, .30, .17, 1))
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (6.0, 6.0, 2.4)
    nt.links.new(p, mp.inputs[0])
    flecks = K.nrange(nt, K.nnoise(nt, mp.outputs[0], 2.6, 5.0, .66), .60, .70)
    colour = K.nmix(nt, K.nmath(nt, "MULTIPLY", flecks, .14), colour, (.50, .38, .24, 1))
    # the mineral grains, dark ones and fewer pale ones, each the size of a grain, gathered in drifts
    drift = K.nrange(nt, K.nnoise(nt, p, 1.3, 4.0, .55), .47, .62)
    def grains(scale, size, rarity, tint, how):
        v = nt.nodes.new("ShaderNodeTexVoronoi")
        v.inputs["Scale"].default_value = scale
        nt.links.new(tc, v.inputs["Vector"])
        which = split(nt, v.outputs["Color"])
        on = K.nmath(nt, "MULTIPLY", K.nmath(nt, "MULTIPLY", K.nrange(nt, v.outputs["Distance"], size * .6, size, 1.0, 0.0), K.nrange(nt, which[0], rarity, rarity + .01)), drift)
        return K.nmix(nt, K.nmath(nt, "MULTIPLY", on, how), colour, tint), on
    colour, dark_grain = grains(14.0, .11, .8, (.035, .022, .014, 1), .7)
    colour, _ = grains(22.0, .12, .84, (.52, .44, .33, 1), .55)
    colour, _ = grains(52.0, .2, .62, (.05, .03, .018, 1), .42)
    # a hair of a vein here and there, healed: darker honey, nothing to catch a nail on
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = .8
    nt.links.new(p, vor.inputs["Vector"])
    crack = K.nmath(nt, "MULTIPLY", K.nrange(nt, vor.outputs["Distance"], .001, .0045, 1.0, 0.0), K.nrange(nt, K.nnoise(nt, p, 1.1, 3.0, .5), .56, .66))
    colour = K.nmix(nt, K.nmath(nt, "MULTIPLY", crack, .22), colour, (.07, .038, .016, 1))
    if cut:
        colour = K.nmix(nt, .55, colour, (.075, .020, .008, 1))      # darker, and years of paste in it
    nt.links.new(colour, b.inputs["Base Color"])
    if cut:
        b.inputs["Roughness"].default_value = .62
    else:
        nt.links.new(K.nmixf(nt, dark_grain, .15, .45), b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = .07 if cut else .5
    b.inputs["Subsurface Weight"].default_value = .55
    b.inputs["Subsurface Radius"].default_value = (1.0, .5, .2)
    b.inputs["Subsurface Scale"].default_value = .5
    nt.links.new(K.nbump(nt, K.nmath(nt, "ADD", K.nmath(nt, "MULTIPLY", K.nnoise(nt, tc, 1.7, 3.0, .5), .0035), K.nmath(nt, "MULTIPLY", K.nnoise(nt, tc, 60.0, 3.0, .5), .00022)), 1.0, 1.0), b.inputs["Normal"])


def stone_material(K):
    m, nt, b = K.new_mat("stone")
    stone_shader(nt, b, K)
    return m


def face_material(K):
    """The stone again, with the paste on what stands, thick and a little wet, and its stain in what is cut away."""
    m, nt, b = K.new_mat("face")
    stone_shader(nt, b, K, cut=True)
    out = nt.nodes["Material Output"]
    box = box_coords(nt, -SIDE / 2, SIDE / 2, -SIDE / 2, SIDE / 2)
    maps_ = split(nt, image_at(nt, os.path.join(MAPS, "face.png"), box, extension="EXTEND"))
    tc = nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tone = K.nrange(nt, K.nnoise(nt, tc, 2.5, 5.0, .55), .36, .66)
    nt.links.new(K.nmix(nt, tone, (.30, .018, .009, 1), (.47, .036, .014, 1)), p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = .52
    p.inputs["Specular IOR Level"].default_value = .3
    nt.links.new(K.nbump(nt, K.nmath(nt, "MULTIPLY", K.nnoise(nt, tc, 34.0, 5.0, .62), .0012), 1.0, 1.0), p.inputs["Normal"])
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(K.nmath(nt, "ADD", maps_[1], K.nmath(nt, "MULTIPLY", maps_[2], .5)), mix.inputs[0])
    nt.links.new(b.outputs[0], mix.inputs[1])
    nt.links.new(p.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


def stone_body(mat):
    """The block with its origin at the middle of its face: faces cut flat, the four long edges and the shoulders cut away in facets, every edge taken off by a hair. Open at the foot, where the face goes."""
    import bmesh
    r = np.random.default_rng(631)
    a = SIDE / 2
    rings = [(.05, 0.0), (.0, .055), (.0, TALL - .62), (.075, TALL - .5), (.62, TALL - .07), (.68, TALL)]
    cut = [.11, .11, .11, .13, .24, .2]       # how much each ring's corners are cut
    bm = bmesh.new()
    loops = []
    for (inset, z), c in zip(rings, cut):
        s = a - inset
        pts = [(s - c, -s), (s, -s + c), (s, s - c), (s - c, s), (-s + c, s), (-s, s - c), (-s, -s + c), (-s + c, -s)]
        loops.append([bm.verts.new((x + r.normal(0, .004), y + r.normal(0, .004), z + (r.normal(0, .004) if z > 0 else 0.0))) for x, y in pts])
    for lo, hi in zip(loops, loops[1:]):
        for i in range(8):
            bm.faces.new((lo[i], lo[(i + 1) % 8], hi[(i + 1) % 8], hi[i]))
    bm.faces.new(loops[-1])
    bm.normal_update()
    edges = [e for e in bm.edges if not any(v.co.z == 0.0 for v in e.verts)]
    done = bmesh.ops.bevel(bm, geom=edges, offset=.022, segments=4, profile=.5, affect="EDGES")
    for f in done["faces"]:
        f.smooth = True          # across what the polish rounded, the sun finds one narrow line
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("stone")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    o = bpy.data.objects.new("stone", me)
    bpy.context.collection.objects.link(o)
    return o


def stone_face(mat):
    """The cut face, real relief, its top row the top of the print."""
    from ink3d import mesh_grid
    im = bpy.data.images.load(os.path.join(MAPS, "face.png"), check_existing=True)
    w = im.size[0]
    a = np.empty(w * w * 4, np.float32)
    im.pixels.foreach_get(a)
    h = a[0::4].reshape(w, w)      # Blender holds an image's rows bottom up: row 0 is the bottom of the print
    s = SIDE / 2 - .05
    n = 1000
    u = np.linspace(-s, s, n)
    # read between the map's pixels: the nearest of them, at this ratio, leaves a fine regular hatch across the flats
    h = (h + np.roll(h, 1, 0) + np.roll(h, -1, 0) + np.roll(h, 1, 1) + np.roll(h, -1, 1)) / 5
    f = np.clip((u + SIDE / 2) / SIDE * (w - 1), 0, w - 1.001)
    i, c = f.astype(int), (f - f.astype(int))[None, :]
    r, e = i[::-1], c[0, ::-1][:, None]
    z = (h[np.ix_(r, i)] * (1 - c) + h[np.ix_(r, i + 1)] * c) * (1 - e) + (h[np.ix_(r + 1, i)] * (1 - c) + h[np.ix_(r + 1, i + 1)] * c) * e
    y, x = np.meshgrid(u[::-1], u, indexing="ij")
    z = np.where(np.maximum(np.abs(x), np.abs(y)) > SIDE / 2 - FRAME[0] * SIDE + .004, 1.0, z)      # outside the frame nothing was cut: the rim meets the block
    # the grid's corners come in to the cut corners of the block
    c = s + s - .11
    over = np.maximum(np.abs(x) + np.abs(y) - c, 0) / 2
    x, y = x - np.sign(x) * over, y - np.sign(y) * over
    o = mesh_grid("face", x, y, (1.0 - z) * RECESS, uv=False)      # what prints stands lowest, what is cut away goes up into the stone
    o.data.materials.append(mat)
    return o


def axis(lo, hi, step, a, b, step_in):
    return np.unique(np.concatenate([np.arange(lo, a, step), np.arange(a, b, step_in), np.arange(b, hi + step, step)]))


def sheet_height(x, y):
    """The sheet a little cockled all over, and what a centimetre of the stone's sinking does to that: the deep square it sits in going out to nothing as an elastic bed does under a flat block, and the creases it pulled from its corners."""
    z = .03 * wrap_noise(x, y, 46.0, 641) + .0045 * wrap_noise(x, y, 7.0, 642)
    a = SIDE / 2
    dx, dy = np.maximum(np.abs(x) - a, 0), np.maximum(np.abs(y) - a, 0)
    ds = np.hypot(dx, dy)
    soft = np.hypot(ds, .012)
    s = -(2 / np.pi) * np.arcsin(np.clip(a / (a + soft), 0, 1))
    th = np.arctan2(y, x)
    n = 720
    lobes = np.random.default_rng(643).normal(0, 1, n)
    lobes = np.real(np.fft.ifft(np.fft.fft(lobes) * np.exp(-(np.fft.fftfreq(n) * n / 26) ** 2)))
    lobes /= lobes.std()
    wr = np.interp((th + np.pi) / (2 * np.pi) * n, np.arange(n + 1), np.append(lobes, lobes[0]))
    s = s - .14 * wr * sm(ds / .2) * np.exp(-ds / 1.1)
    for sx in (-1, 1):
        for sy in (-1, 1):
            along = (sx * (x - sx * a) + sy * (y - sy * a)) / math.sqrt(2)
            across = (sx * (x - sx * a) - sy * (y - sy * a)) / math.sqrt(2)
            s = s - .22 * np.exp(-(across / .05) ** 2) * sm(along / .1) * np.exp(-np.maximum(along, 0) / 1.5)
    return z, s


_noises = {}


def wrap_noise(x, y, span, seed):
    if seed not in _noises:
        _noises[seed] = spectral(512, 3.0, seed)
    t = _noises[seed]
    u, v = (x / span % 1) * 512, (y / span % 1) * 512
    i, j = u.astype(int) % 512, v.astype(int) % 512
    fu, fv = u - np.floor(u), v - np.floor(v)
    return (t[j, i] * (1 - fu) + t[j, (i + 1) % 512] * fu) * (1 - fv) + (t[(j + 1) % 512, i] * (1 - fu) + t[(j + 1) % 512, (i + 1) % 512] * fu) * fv


def only_seen_in_polish(o):
    o.visible_camera = o.visible_diffuse = o.visible_transmission = o.visible_volume_scatter = o.visible_shadow = False
    o.visible_glossy = True


def room(K):
    """What the polish reflects: a dim room around the set, and in it the two panes of a window. Nothing else sees them."""
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=900.0)
    me = bpy.data.meshes.new("room")
    bm.to_mesh(me)
    bm.free()
    def lit(name, e, colour):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        em, out = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
        em.inputs[0].default_value, em.inputs[1].default_value = (*colour, 1), e
        nt.links.new(em.outputs[0], out.inputs[0])
        return m
    me.materials.append(lit("room", ROOM_E, (1.0, .9, .78)))
    o = bpy.data.objects.new("room", me)
    bpy.context.collection.objects.link(o)
    only_seen_in_polish(o)
    glass = lit("window", WINDOW_E, (1.0, .93, .82))
    for i, (x, y, z, hw, hh) in enumerate(PANES):
        verts = [(x - hw, y - hh, z), (x + hw, y - hh, z), (x + hw, y + hh, z), (x - hw, y + hh, z)]
        pane = bpy.data.meshes.new(f"pane{i}")
        pane.from_pydata(verts, [], [(3, 2, 1, 0)])
        pane.materials.append(glass)
        p = bpy.data.objects.new(f"pane{i}", pane)
        bpy.context.collection.objects.link(p)
        only_seen_in_polish(p)


class Shot:
    """The whole set, with everything that moves keyed frame by frame, so that each frame has its motion blur."""

    def __init__(self, samples, pct, frames):
        from ink3d import mesh_grid, reset
        from mathutils import Matrix, Quaternion, Vector
        import film_press_machine as K
        self.sc = sc = reset(samples, pct)
        sc.render.fps, sc.render.fps_base = FPS, 1.0
        sc.render.film_transparent = False
        sc.render.image_settings.color_mode = "RGB"
        sc.render.image_settings.color_depth = "8"
        sc.render.dither_intensity = .6
        sc.render.use_persistent_data = True
        sc.render.use_motion_blur = True
        sc.render.motion_blur_shutter = .5
        sc.cycles.max_bounces = 8
        sc.cycles.diffuse_bounces = 4
        sc.cycles.glossy_bounces = 4
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
        sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        sc.cycles.denoising_prefilter = "NONE"
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"

        w = sc.world.node_tree.nodes["Background"]
        w.inputs[0].default_value = (*SKY_COLOUR, 1)
        w.inputs[1].default_value = SKY_E
        s = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        bpy.context.collection.objects.link(s)
        s.data.energy, s.data.color, s.data.angle = SUN_E, SUN_COLOUR, math.radians(SUN_SIZE)
        az, el = math.radians(SUN_AZ), math.radians(SUN_EL)
        s.rotation_euler = Vector((-math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))).to_track_quat("Z", "Y").to_euler()

        xs, ys = axis(-160.0, 160.0, .5, -5.0, 5.0, .02), axis(-50.0, 280.0, .5, -6.0, 5.0, .02)
        x, y = np.meshgrid(xs, ys)
        self.flat, self.sink = sheet_height(x, y)
        self.sheet = mesh_grid("sheet", x, y, self.flat, uv=False)
        self.paper = paper_material(K)
        self.sheet.data.materials.append(self.paper)
        self.xy = np.stack([x, y], -1).astype(np.float32)
        self.down = 0.0

        self.body, self.face = stone_body(stone_material(K)), stone_face(face_material(K))
        room(K)
        hang = (Matrix.Rotation(math.radians(-HANG["tilt"]), 4, "X") @ Matrix.Rotation(math.radians(HANG["yaw"]), 4, "Z") @ Matrix.Rotation(math.radians(HANG["roll"]), 4, "Y")).to_quaternion()
        data = bpy.data.cameras.new("cam")
        data.lens, data.sensor_width = LOW["lens"], 36.0
        data.clip_start, data.clip_end = .5, 2000.0
        self.cam = bpy.data.objects.new("cam", data)
        bpy.context.collection.objects.link(self.cam)
        sc.camera = self.cam
        data.dof.use_dof = True
        data.dof.aperture_blades = 7
        for o in (self.body, self.face, self.cam):
            o.rotation_mode = "QUATERNION"
        for f in frames:
            t = f / FPS
            p = pose(t)
            if p:
                q = hang.slerp(Quaternion(), p[1])
                for o in (self.body, self.face):
                    o.location, o.rotation_quaternion = p[0], q
                    o.keyframe_insert("location", frame=f)
                    o.keyframe_insert("rotation_quaternion", frame=f)
            c = cam_at(t)
            loc, target = Vector(c["loc"]), Vector(c["target"])
            self.cam.location, self.cam.rotation_quaternion = loc, (target - loc).to_track_quat("-Z", "Y")
            self.cam.keyframe_insert("location", frame=f)
            self.cam.keyframe_insert("rotation_quaternion", frame=f)

    def render(self, f, path):
        t = f / FPS
        sc = self.sc
        sc.frame_set(f)
        for o in (self.body, self.face):
            o.hide_render = pose(t) is None
        down = sunk(t)
        if abs(down - self.down) > 1e-5:
            self.down = down
            z = self.flat + self.sink * down
            co = np.concatenate([self.xy, z[..., None].astype(np.float32)], -1)
            self.sheet.data.vertices.foreach_set("co", co.ravel())
            self.sheet.data.update()
        nodes = self.paper.node_tree.nodes
        nodes["printed"].outputs[0].default_value = float(t >= RISE)
        nodes["named"].outputs[0].default_value = float(sm((t - NAME_IN[0]) / (NAME_IN[1] - NAME_IN[0])))
        nodes["fibre"].outputs[0].default_value = lerp(FIBRE[0], FIBRE[1], float(sm((t - CRANE[0]) / (CRANE[1] - CRANE[0]))))
        sc.render.motion_blur_shutter = lerp(*SHUTTER, float(sm((t - RISE) / .2)))
        c = cam_at(t)
        data = self.cam.data
        data.dof.aperture_fstop = c["stop"]
        loc, target = np.array(c["loc"]), np.array(c["target"])
        look = (target - loc) / np.linalg.norm(target - loc)
        data.dof.focus_distance = float((focus_on(t) - loc).dot(look))
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print("rendered", path, flush=True)


def stop_down(t):
    """Seen from above the paper gives back a little more of the sun than from low down: the operator closes down a touch through the crane, so that the sheet stays one sheet."""
    return lerp(1.0, .975, float(sm((t - CRANE[0]) / (CRANE[1] - CRANE[0]))))


def grade(name):
    """What any camera adds, and no more."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter as gf
    a = np.asarray(Image.open(os.path.join(OUT, "raw", name + ".png")).convert("RGB"), np.float32) / 255 * stop_down(SHOTS[name])
    n = np.random.default_rng(650 + sum(map(ord, name))).normal(0, 1, a.shape[:2]).astype(np.float32)
    if a.shape[0] >= 1000:
        n = gf(n, .55) * 1.5
    lum = a.mean(-1, keepdims=True)
    a = a + n[..., None] * (.006 * (1.2 - lum) + .0016)
    os.makedirs(OUT, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255 + .5)).save(os.path.join(OUT, name + ".png"))
    print(os.path.join(OUT, name + ".png"), flush=True)


def mp4():
    """The frames with their grain, as a film."""
    import subprocess
    from PIL import Image
    d = os.path.join(OUT, "seq")
    names = sorted(n for n in os.listdir(d) if n.endswith(".png"))
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x540", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", os.path.join(OUT, "ident_low.mp4")], stdin=subprocess.PIPE)
    for i, name in enumerate(names):
        a = np.asarray(Image.open(os.path.join(d, name)).convert("RGB"), np.float32) / 255 * stop_down(i / FPS)
        n = np.random.default_rng(900 + i).normal(0, 1, a.shape[:2]).astype(np.float32)
        a = a + n[..., None] * (.006 * (1.2 - a.mean(-1, keepdims=True)) + .0016)
        p.stdin.write(np.uint8(np.clip(a, 0, 1) * 255 + .5).tobytes())
    p.stdin.close()
    p.wait()
    print(len(names), "frames", flush=True)


if __name__ == "__main__":
    if bpy:
        argv = sys.argv[sys.argv.index("--") + 1:]
        opt = lambda k, d: type(d)(argv[argv.index(k) + 1]) if k in argv else d
        names = [a for a in argv if a in SHOTS or a == "seq"]
        keys = list(range(int(5 * FPS)))
        shot = Shot(opt("--samples", 256), opt("--pct", 100), keys)
        if "seq" in names:
            os.makedirs(os.path.join(OUT, "seq"), exist_ok=True)
            for f in keys[opt("--from", 0):opt("--to", len(keys))]:
                shot.render(f, os.path.join(OUT, "seq", f"f_{f:04d}.png"))
        for name in names:
            if name != "seq":
                os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
                shot.render(round(SHOTS[name] * FPS), os.path.join(OUT, "raw", name + ".png"))
    elif sys.argv[1] == "tile":
        tile()
    elif sys.argv[1] == "maps":
        maps()
    elif sys.argv[1] == "grade":
        for name in sys.argv[2:]:
            grade(name)
    elif sys.argv[1] == "mp4":
        mp4()
