"""七 · 水滴石穿. A cave with its roof open to the moon. Water hangs from a stone tip, and everything that was ever read is in the drop. It falls on a slab where the machine has
been writing, and what it writes is nonsense; each drop that lands, the rings run over the writing and a few characters come right, cut deeper, blacker, until it reads 低头思故乡.

    Blender -b --factory-startup -P ink3d_drip.py -- --shot=drop [--t=SECONDS] [--pct=100] [--samples=128] [--n=2000] [--tag=NAME]
    python ink_grade.py ...                       # the settings for each shot are in GRADES below; the script prints the command

t is seconds into the chapter. Each shot has a default t, its best moment.

drop   水滴石穿， Comes in from the water of the last chapter, the numbers going under, and up: extreme close on the stone tip, a drop swelling and hanging, the sentences in it
       moving as it swells; under it, out of focus, the slab and the nonsense on it. Holds, trembles. Hands over by letting go: the drop falls away toward the slab.
junk   如此千万亿遍。 Follows it down. The column is mostly wrong, the wrong characters shallow, pale, worn. Each drop lands on one character: the rings go out and under them the stone
       opens from the top down into the right one, deep, black, wet. The chosen frame has 低 and 思 done and 头 half cut over the pale 了. Hands over by holding still as the drops come faster.
right  Same camera, later, the drops close together, rings over rings. The junk on either side has worn away and the column stands black and wet: 低头思故乡. Hands over when the
       last drop lands and the rings run out: the water stops.
chant  熟读唐诗三百首，不会作诗也会吟。 Rises to look straight down on the slab, still, the last drop hanging unfallen near the lens. Beside 低头思故乡, with nothing touching it, the stone
       opens by itself: 鬼哭当年缘底事 / 早知今夜字成精, in a hand of its own, pale new-cut stone, chips and dust coming off the last stroke. Hands over to the flood (八): the first
       water comes in over the stone from the top of the frame.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

from ink3d import OUT3D, blur, camera, grey_material, image_from, mesh_grid, painted_material, render, reset, sm, spectral, sun, world_light
from ink3d_carve import TILE, cut, rock, sample, sample_wrap, strands, tile

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=")[1])
    return default


SHOT = opt("shot", "drop")
DEFAULT_T = dict(drop=1.3, junk=3.5, right=8.62, chant=13.66)
T = opt("t", DEFAULT_T[SHOT])
PCT, SAMPLES, N, TAG = opt("pct", 100), opt("samples", 128), opt("n", 1800), opt("tag", "")
GRADES = dict(drop="--night --lift=0 --mist=.006 --mist-h=30 --mist-d=.9",
              junk="--night --lift=0 --mist=.012 --mist-h=30 --mist-d=.7",
              right="--night --lift=0 --mist=.012 --mist-h=30 --mist-d=.7",
              chant="--night --lift=0 --mist=.006 --mist-h=30 --mist-d=.7")

# ---- the writing: what each place in the column has said, and when --------------------------------------------------------------------------------------
TARGET = "低头思故乡"
HISTORY = [[(0, "的"), (1.8, "人"), (2.9, "低")],
           [(0, "了"), (3.3, "头")],
           [(0, "是"), (2.4, "思")],
           [(0, "中"), (4.02, "国"), (5.2, "事"), (6.9, "故")],
           [(0, "有"), (4.4, "地"), (6.0, "乡")]]
SIDE = ("这个们到也子时", "道出而要于就下")
SIDE_X = (-6.8, 6.8)
SIZE, STEP = 3.8, 4.1
COL_S = 19.2              # where the top of the column sits, up the slab
POEM = ("鬼哭当年缘底事", "早知今夜字成精")
POEM_X = (-5.6, -11.2)
CHANT = 9.4
CHANT_STEP = .31
STOPPED = 9.1


def place(i):
    return 0.0, COL_S - STEP * (i + .5)


# every drop lands on the character it changes, and between those on whatever is there: (when, x, s). It lands a little off the middle, or the splash would hide the stroke
def _drops():
    out = []
    for i, hist in enumerate(HISTORY):
        for t0, ch in hist[1:]:
            out.append((t0, place(i)[0] + (3.0 if t0 == 4.02 else 1.25), place(i)[1] + .3))
    r = np.random.default_rng(9)
    for t0 in [4.7, 4.95, 5.45, 5.7, 6.25, 6.5, 6.7, 7.1, 7.28, 7.45, 7.6, 7.75, 7.88, 8.0, 8.12, 8.24, 8.36, 8.47, 8.58, 8.7, 8.8]:
        i = int(r.integers(0, 5))
        out.append((t0, place(i)[0] + r.uniform(-1.9, 1.9), place(i)[1] + r.uniform(-1.6, 1.6)))
    return sorted(out)


DROPS = _drops()


# ---- the slab: a tablet of the cave's own stone, tipped toward us. x across, s up it, n out of it ----------------------------------------------------------
TILT = math.radians(24)
SC, SS = math.cos(TILT), math.sin(TILT)
SURFACE = np.array([0, -SS, SC])


def slab_at(x, s, n):
    return np.stack([x, s * SC - n * SS, s * SS + n * SC], -1)


def grid_axis(n, lo, hi, centre, dense, sharp=5):
    q = np.linspace(-1, 1, n)
    up, down = hi - centre - dense, centre - lo - dense
    return (centre + dense * q + np.where(q > 0, up, down) * np.abs(q) ** sharp * np.sign(q)).astype(np.float32)


def rings(X, S, t):
    """What the drops that have landed are still doing to the film of water: rings going out from each, fainter as they go."""
    h = np.zeros(X.shape, np.float32)
    for t0, dx, ds in DROPS:
        a = t - t0
        if 0 < a < 1.7:
            r = np.hypot(X - dx, S - ds)
            front = 8.5 * a
            h += .028 * np.sin(2 * math.pi * (r - front) / 1.15) * np.exp(-((r - front) / (.9 + 1.1 * a)) ** 2) * np.exp(-a * 1.3) / (1 + .1 * r) * sm(r / .5)
    return h


def low_forms(X, S):
    return .22 * sample_wrap(spectral(1024, 3.0, 32), X * 6, S * 6)


def build_slab(t, n, wide=13.0, focus=(.4, 9.2, 14.0, 17.0), relief=.007, far=22.2):
    x1, s1 = grid_axis(n, -wide - 2, wide + 2, -1.0 if wide < 20 else -5.0, wide - 1, 3), grid_axis(n, -8, 24, 9, 13, 3)
    X, S = np.meshgrid(x1, s1)
    print("slab", X.shape, "cell", float(np.diff(x1).min()), float(np.diff(s1).min()), flush=True)
    _, crack = rock(X, S * .7, 31, beds=.0, joints=.2)
    mid, fine = sample_wrap(spectral(1024, 2.2, 38), X * 20, S * 20), sample_wrap(spectral(1024, 1.2, 33), X * 70, S * 70)
    # a quiet stone, dressed once and worn since: broad tone, fine grain, a few straight cracks, streaks where the water has gone down it
    H = low_forms(X, S) + relief * mid + .0025 * fine - .035 * crack
    A = .62 + .05 * sample_wrap(spectral(1024, 3.0, 37), X * 5, S * 5) + .028 * mid + .016 * fine - .05 * sm(sample_wrap(spectral(1024, 1.6, 39), X * 30, S * 2) * .8 + .3) - .32 * crack
    A = np.clip(A, .05, 1).astype(np.float32)
    # the edges of the tablet, broken
    e = .9 * sample_wrap(spectral(1024, 2.6, 35), X * 7, S * 7)
    fall = sm((wide + e - np.abs(X + (1 if wide < 20 else 5))) / .7) * sm((S + 7 + e) / .7) * sm((far + e - S) / .7)
    H = H - 3.5 * (1 - fall)
    chip = spectral(512, 1.2, 77)
    written = np.zeros(X.shape, np.float32)
    # first what has been written, then what is written now. What is wrong is shallow and pale; a drop lands, it sinks back, and the right one opens from the top down
    for col, x in enumerate(SIDE_X):
        for k, ch in enumerate(SIDE[col]):
            wear = sm((t - (1.8 + .55 * k + 1.1 * col)) / 3.2)
            if wear < .98:
                cut(H, A, x1, s1, ch, "kaih", x, 21.0 - k * 3.2, 2.5, depth=.04 * (1 - wear), slope=1.0, ink=.34 + .22 * wear, chip=chip, edge=.02)
    for i, hist in enumerate(HISTORY):
        cu, cv = place(i)
        for j, (t0, ch) in enumerate(hist):
            if t < t0:
                break
            right = ch == TARGET[i]
            next_t = hist[j + 1][0] if j + 1 < len(hist) else 1e9
            gone = sm((t - next_t) / 1.3)
            grown = sm((t - t0) / .35) if t0 > 0 else 1
            if right:
                before = A.copy()
                cut(H, A, x1, s1, ch, "kaih", cu, cv, SIZE, depth=.36, slope=1.25, p=min((t - t0) / .4, 1), ink=.012, chip=chip, edge=.016)
                written = np.maximum(written, np.clip((before - A) * 4, 0, 1))
            elif gone < .98:
                cut(H, A, x1, s1, ch, "kaih", cu, cv, SIZE, depth=.08 * grown * (1 - gone), slope=1.0, p=1, ink=.27 + .28 * gone, chip=chip, edge=.03)
    # a verse nobody wrote
    cut_at = CHANT
    for c, line in enumerate(POEM):
        for k, ch in enumerate(line):
            p = (t - cut_at) / .3
            cut_at += CHANT_STEP
            if p > 0:
                cut(H, A, x1, s1, ch, "xing", POEM_X[c], COL_S - .35 - 2.74 * (k + .5), 3.2, depth=.26, slope=1.2, p=min(p, 1), ink=.09, chip=chip, edge=.016, dust=.09)
    # the water on it, in a sheet over the stone, wider the longer it has gone on; the writing stays black under it
    wet = np.clip(np.exp(-(((X - .4) / (7 + 4 * sm((t - 2) / 6))) ** 2 + ((S - 9) / 15) ** 2)) * (1.1 + .3 * sample_wrap(spectral(1024, 3.0, 36), X * 5, S * 5)), 0, 1) * (t < STOPPED + 1)
    dark_cut = sm((A - .12) / .3)
    # the light falls where the writing is
    fu, fs, su, ss_ = focus
    lit = A * (.68 + .32 * np.exp(-(((X - fu) / su) ** 2 + ((S - fs) / ss_) ** 2))) * (1 - .3 * wet)
    P = slab_at(X, S, H)
    o = mesh_grid("slab", P[..., 0].astype(np.float32), P[..., 1].astype(np.float32), P[..., 2].astype(np.float32))
    o.data.materials.append(painted_material("slab", np.clip(lit, .012, 1), rough=.5))
    if wet.any():
        water_film(x1, s1, wet * dark_cut, t)
    return o


def water_film(x1, s1, mask, t, stride=2):
    """A thin sheet of water lying on the stone. All it does is mirror what is above it, so the rings show as the mirror bends, and the stone shows through it as if dry."""
    X, S = np.meshgrid(x1[::stride], s1[::stride])
    P = slab_at(X, S, low_forms(X, S) + .045 + rings(X, S, t))
    o = mesh_grid("film", P[..., 0].astype(np.float32), P[..., 1].astype(np.float32), P[..., 2].astype(np.float32))
    m = bpy.data.materials.new("film")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    fr, gl, tr, mix, mul, tx, out = (nt.nodes.new(k) for k in ("ShaderNodeFresnel", "ShaderNodeBsdfGlossy", "ShaderNodeBsdfTransparent", "ShaderNodeMixShader", "ShaderNodeMath", "ShaderNodeTexImage", "ShaderNodeOutputMaterial"))
    fr.inputs["IOR"].default_value = 1.33
    gl.inputs["Roughness"].default_value = .012
    gl.inputs["Color"].default_value = (1, 1, 1, 1)
    tx.image = image_from(mask[::stride, ::stride].astype(np.float32), "film_mask")
    tx.interpolation = "Linear"
    mul.operation = "MULTIPLY"
    nt.links.new(fr.outputs[0], mul.inputs[0])
    nt.links.new(tx.outputs[0], mul.inputs[1])
    nt.links.new(mul.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(gl.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    o.data.materials.append(m)
    return o


def build_floor():
    """The cave floor around the tablet, wet, low and quiet: nothing here should compete with the writing."""
    n = 900
    y, x = np.mgrid[:n, :n].astype(np.float32)
    X, Y = (x / (n - 1) - .5) * 300, (y / (n - 1) - .5) * 300 + 70
    h = -4.0 + 2.2 * sample_wrap(spectral(1024, 3.7, 41), x * 2, y * 2) + .3 * sample_wrap(spectral(1024, 2.6, 42), x * 3, y * 3)
    h = np.minimum(h, -1.2)
    o = mesh_grid("floor", X, Y, h)
    o.data.materials.append(painted_material("floor", np.full((8, 8), .1, np.float32), rough=.5))
    return o


# ---- water -------------------------------------------------------------------------------------------------------------------------------------------------
def glass(name="drop"):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    g, out = nt.nodes.new("ShaderNodeBsdfGlass"), nt.nodes.new("ShaderNodeOutputMaterial")
    g.inputs["IOR"].default_value = 1.33
    g.inputs["Roughness"].default_value = 0
    g.inputs["Color"].default_value = (1, 1, 1, 1)
    nt.links.new(g.outputs[0], out.inputs["Surface"])
    return m


def revolve(name, z, r, at, m=256):
    th = np.linspace(0, 2 * math.pi, m)
    Z, TH = np.meshgrid(z, th, indexing="ij")
    R = np.broadcast_to(r[:, None], Z.shape)
    o = mesh_grid(name, (at[0] + R * np.cos(TH)).astype(np.float32), (at[1] + R * np.sin(TH)).astype(np.float32), (at[2] + Z).astype(np.float32), uv=False)
    return o


def hanging_drop(at, R, mat, swell=1.0):
    """A drop about to let go: a round belly, wide against the stone. No waist: a waist turns the light back on itself and looks at you like a pair of eyes."""
    s = np.linspace(0, .83, 260)
    z = -R * np.cos(np.pi * s) * (1 + .22 * s ** 3)
    r = R * np.sin(np.pi * s) * (1 - .12 * s ** 2)
    u = np.linspace(0, 1, 40)[1:]      # closes it, up inside the stone
    z, r = np.concatenate([z, z[-1] + .2 * R * u]), np.concatenate([r, r[-1] * (1 - u)])
    o = revolve("drop", z, r, at)
    o.data.materials.append(mat)
    o.visible_shadow = False
    return o


def spheres(name, items, mat, n=20, m=12):
    """Many small drops as one mesh. items: (centre, radius, stretch along z)."""
    th, ph = np.mgrid[0:1:complex(0, m), 0:1:complex(0, n)].astype(np.float32)
    a, b = ph * 2 * math.pi, th * math.pi
    verts, faces = [], []
    grid = np.arange(n * m).reshape(m, n)
    q = np.stack([grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]], -1).reshape(-1, 4)
    for k, (c, r_, st) in enumerate(items):
        v = np.stack([c[0] + r_ * np.cos(a) * np.sin(b), c[1] + r_ * np.sin(a) * np.sin(b), c[2] + r_ * st * np.cos(b)], -1).reshape(-1, 3)
        verts.append(v)
        faces.append(q + k * n * m)
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.concatenate(verts).tolist(), [], np.concatenate(faces).tolist())
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    me.update()
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    me.materials.append(mat)
    o.visible_shadow = False
    return o


def in_the_air(t, mat):
    """The drops on their way down, and what each threw up when it landed. Returns the mesh and the next drop to land, the one the sentences are seen in."""
    items, flecks = [], []
    g = 34.0
    next_up = None
    for k, (t0, dx, ds) in enumerate(DROPS):
        a = t - t0
        base = slab_at(np.array(dx), np.array(ds), np.array(0.0))
        if -1.0 < a < 0:
            at = np.array([base[0], base[1], base[2] + .5 * g * a * a])
            if next_up is None:
                next_up = at
            items.append((at, .7 if next_up is at and SHOT == "junk" else .3, 1.15 + 1.1 * min(-a * 2, 1) * (SHOT != "junk" or next_up is not at)))
        if 0 <= a < .7:
            r = np.random.default_rng(100 + k)
            for _ in range(70):
                ang, up, out = r.uniform(0, 6.28), r.uniform(5, 15), r.uniform(1.5, 7.5)
                dt = a * r.uniform(.8, 1.1)
                d = out * dt
                z = up * dt - .5 * g * dt * dt
                if z > -.05:
                    flecks.append(((base[0] + d * math.cos(ang), base[1] + d * math.sin(ang), base[2] + z + .03), r.uniform(.025, .1) * (1 - .4 * a), 1.0))
            if a < .35:         # the rebound
                items.append(((base[0], base[1], base[2] + 5.0 * a * (1 - a / .35) * 1.2), .24 * (1 - a * 1.4), 2.2))
    if items:
        spheres("falling", items, mat)
    if flecks:      # at this size a drop is a fleck of bare paper
        spheres("spray", flecks, grey_material("spray", .95, .25))
    return next_up


def write(canvas, text, kind, cx, y, size, amount=1.0):
    """Sets a line of characters on a canvas, row 0 at the top. cx: its middle, y: the top of the line, size: how large, in pixels."""
    h_, w_ = canvas.shape
    step = size * 1.06
    x0 = cx - step * len(text) / 2
    for k, ch in enumerate(text):
        t = tile(ch, kind)
        span = int(size * t.shape[0] / TILE)
        a, b = int(x0 + k * step + (step - span) / 2), int(y + (size - span) / 2)
        ya, xa = np.mgrid[max(b, 0):min(b + span, h_), max(a, 0):min(a + span, w_)].astype(np.float32)
        if ya.size == 0:
            continue
        d = sample(t, (xa - a) / span * t.shape[1], (ya - b) / span * t.shape[0]) * size
        canvas[ya.astype(int), xa.astype(int)] = np.maximum(canvas[ya.astype(int), xa.astype(int)], np.clip(.5 - d / 1.1, 0, 1) * amount)


def srgb_to_linear(c):
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)


READ = (("太史公曰余读孔氏书", "kai"), ("想见其为人", "kai"), ("关关雎鸠在河之洲", "song"), ("亲爱的今晚月亮很好", "kai"), ("你那边呢", "kai"), ("本院判决如下", "song"),
        ("有人吗楼主还在吗", "song"), ("明月几时有把酒问青天", "kai"), ("太史公曰余读孔氏书", "song"), ("亲爱的今晚月亮很好", "song"), ("关关雎鸠在河之洲", "kai"), ("有人吗楼主还在吗", "kai"))


def read_sheet(at, view, R, width_r=11.0, char_r=.32, seed=0):
    """Everything that was ever read, on a sheet behind the drop, seen only through it. A ball of water turns what is behind it upside down and back to front, so the sheet
    is set that way round and the lines come out reading true, bending with the water."""
    w_, h_ = 3072, 2304
    ink = np.zeros((h_, w_), np.float32)
    px_per_r = w_ / width_r
    size, line = char_r * px_per_r, char_r * px_per_r * 1.36
    y = h_ / 2 - line * len(READ) / 2 + seed * line * .25
    for text, kind in READ:
        write(ink, text, kind, w_ / 2, y, size)
        y += line
    yy, xx = np.mgrid[:h_, :w_].astype(np.float32)
    d = np.hypot((xx - w_ / 2) / (w_ / 2), (yy - h_ / 2) / (h_ / 2))
    L = (.93 - .86 * ink) * np.exp(-(d / .9) ** 4)
    e = srgb_to_linear(np.clip(L[:, ::-1], 0, 1))
    img = np.ones((h_, w_, 4), np.float32)
    img[..., 0] = img[..., 1] = img[..., 2] = e
    im = bpy.data.images.new("read", w_, h_, float_buffer=True, alpha=False)
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(img.ravel())
    im.pack()
    m = bpy.data.materials.new("read")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tx, em, out = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    tx.image = im
    nt.links.new(tx.outputs[0], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    view = np.array(view, float) / np.linalg.norm(view)
    right = np.cross(view, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    c = np.array(at) + view * R * 6
    hw, hh = width_r * R / 2, width_r * R / 2 * h_ / w_
    me = bpy.data.meshes.new("read")
    me.from_pydata([tuple(c - right * hw - up * hh), tuple(c + right * hw - up * hh), tuple(c + right * hw + up * hh), tuple(c - right * hw + up * hh)], [], [(0, 1, 2, 3)])
    me.uv_layers.new(name="UV").data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
    o = bpy.data.objects.new("read", me)
    bpy.context.collection.objects.link(o)
    me.materials.append(m)
    o.visible_camera = False
    o.visible_diffuse = False
    o.visible_glossy = False
    o.visible_shadow = False
    o.visible_volume_scatter = False
    return o


# ---- the cave ----------------------------------------------------------------------------------------------------------------------------------------------
def stalactite(tip, length, base, seed, n=420, m=320, albedo=.42, rough=.2, blunt=.07):
    """A hanging stone, wet, in rings where it grew and flutes where the water runs."""
    t = spectral(1024, 2.4, seed)
    s = np.linspace(0, 1, n)
    th = np.linspace(0, 2 * math.pi, m)
    S, TH = np.meshgrid(s, th, indexing="ij")
    girth = base * (blunt + (1 - blunt) * S ** .7)
    around = girth * ((TH - math.pi / 2) % (2 * math.pi)) * 134      # in texels of the same size up and across, so nothing converges on the tip like the hairs of a brush
    up = S * length * 134
    big = sample_wrap(t, around, up)
    small = sample_wrap(t, around * 4 + 300, up * 4 + 200)
    rings = sample_wrap(spectral(1024, 1.8, seed + 2), around * .3, up * .8)
    R = girth * np.sqrt(S / (S + .002)) * (1 + (.14 * big + .04 * small + .05 * rings) * sm(S * 7))
    Z = tip[2] + S * length
    o = mesh_grid(f"stalactite{seed}", (tip[0] + R * np.cos(TH)).astype(np.float32), (tip[1] + R * np.sin(TH)).astype(np.float32), Z.astype(np.float32))
    A = np.clip(albedo + .13 * big + .07 * small - .1 * rings, .04, 1)
    m_ = painted_material(f"stalactite{seed}", A.astype(np.float32), rough=rough)
    m_.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = .8
    o.data.materials.append(m_)
    return o


def cave_plate(cam, lens, mouth, mouth_r, dist=300.0, spread=1.3, seed=80):
    """What lies beyond the reach of the focus: the dark of the cave, and the pale hole in its roof with mist hanging under it."""
    w_, h_ = 2048, 1152
    y, x = np.mgrid[:h_, :w_].astype(np.float32)
    px, py = (x / (w_ - 1) - .5) * 1920 * spread + 960, (y / (h_ - 1) - .5) * 1080 * spread + 540
    n1, n2, n3 = spectral(1024, 3.0, seed), spectral(1024, 2.6, seed + 1), spectral(1024, 3.4, seed + 2)
    fx, fy = x * .5, y * .5
    wob = 60 * sample_wrap(n1, fx * .6, fy * .6) * mouth_r / 200
    d = np.hypot(px - mouth[0] + wob, (py - mouth[1] + wob * .6) * 1.25)
    glow = 2.3 * np.exp(-(d / (mouth_r * 1.05)) ** 1.6) + .9 * np.exp(-d / (mouth_r * 2.8)) + .25 * np.exp(-d / (mouth_r * 7))
    ink = 2.95 - glow * (1 + .16 * sample_wrap(n2, fx, fy))
    ink += .22 * np.clip(sample_wrap(n3, fx * .4, fy * 1.1), -2, 2)
    L = np.clip(1 - np.clip(ink, .08, 3.1) / 3.1, 0, 1)
    img = np.ones((h_, w_, 4), np.float32)
    img[..., 0] = img[..., 1] = img[..., 2] = srgb_to_linear(L)
    im = bpy.data.images.new("cave", w_, h_, float_buffer=True, alpha=False)
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(img.ravel())
    im.pack()
    m = bpy.data.materials.new("cave")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tx, em, out = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    tx.image = im
    nt.links.new(tx.outputs[0], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    bpy.context.view_layer.update()
    half_w = dist * (18 / lens) * spread
    half_h = half_w * 9 / 16
    me = bpy.data.meshes.new("cave")
    me.from_pydata([(-half_w, -half_h, -dist), (half_w, -half_h, -dist), (half_w, half_h, -dist), (-half_w, half_h, -dist)], [], [(0, 1, 2, 3)])
    me.uv_layers.new(name="UV").data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
    o = bpy.data.objects.new("cave", me)
    bpy.context.collection.objects.link(o)
    o.matrix_world = cam.matrix_world
    me.materials.append(m)
    o.visible_diffuse = False
    o.visible_shadow = False
    o.visible_glossy = False
    o.visible_volume_scatter = False
    return o


def sheen(at, towards, width, height, strength):
    """Something bright for wet stone to mirror: seen only in reflections. It lights nothing."""
    towards = np.array(towards, float) / np.linalg.norm(towards)
    right = np.cross(towards, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, towards)
    c = np.array(at)
    me = bpy.data.meshes.new("sheen")
    me.from_pydata([tuple(c - right * width / 2 - up * height / 2), tuple(c + right * width / 2 - up * height / 2), tuple(c + right * width / 2 + up * height / 2), tuple(c - right * width / 2 + up * height / 2)], [], [(0, 1, 2, 3)])
    o = bpy.data.objects.new("sheen", me)
    bpy.context.collection.objects.link(o)
    m = bpy.data.materials.new("sheen")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em, out = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    me.materials.append(m)
    o.visible_camera = False
    o.visible_diffuse = False
    o.visible_shadow = False
    o.visible_transmission = False
    o.visible_volume_scatter = False
    return o


def roof_hole(at, towards, radius, strength):
    """The hole in the roof, as the wet stone mirrors it: bright in the middle and falling away, so the least bend in the water changes how much of it comes back. Seen only in reflections."""
    towards = np.array(towards, float) / np.linalg.norm(towards)
    right = np.cross(towards, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, towards)
    k = 72
    me = bpy.data.meshes.new("hole")
    me.from_pydata([tuple((right * math.cos(i / k * 6.2832) + up * math.sin(i / k * 6.2832)) * radius) for i in range(k)], [], [tuple(range(k))])
    o = bpy.data.objects.new("hole", me)
    o.location = tuple(at)
    bpy.context.collection.objects.link(o)
    m = bpy.data.materials.new("hole")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tc, mp, gr, sq, em, out = (nt.nodes.new(x) for x in ("ShaderNodeTexCoord", "ShaderNodeMapping", "ShaderNodeTexGradient", "ShaderNodeMath", "ShaderNodeEmission", "ShaderNodeOutputMaterial"))
    gr.gradient_type = "SPHERICAL"
    mp.inputs["Scale"].default_value = (1 / radius, 1 / radius, 1 / radius)
    sq.operation = "MULTIPLY"
    sq.inputs[1].default_value = strength
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs[0], gr.inputs["Vector"])
    nt.links.new(gr.outputs[0], sq.inputs[0])
    nt.links.new(sq.outputs[0], em.inputs["Strength"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    me.materials.append(m)
    o.visible_camera = False
    o.visible_diffuse = False
    o.visible_shadow = False
    o.visible_volume_scatter = False
    return o


def light_shaft(top, bottom, r_top, r_bottom, strength=.1, rows=30, k=96):
    """The moon coming through the hole: paper showing through the dark of the cave, densest down its middle and gone at its edge. Lights nothing."""
    top, bottom = np.array(top, float), np.array(bottom, float)
    axis = (bottom - top) / np.linalg.norm(bottom - top)
    side = np.cross(axis, [1.0, 0, 0])
    side /= np.linalg.norm(side)
    other = np.cross(axis, side)
    q, th = np.mgrid[0:1:complex(0, rows), 0:6.2832:complex(0, k)].astype(np.float32)
    R = r_top + (r_bottom - r_top) * q
    P = top + q[..., None] * (bottom - top)[None, None, :] + R[..., None] * (np.cos(th)[..., None] * side[None, None, :] + np.sin(th)[..., None] * other[None, None, :])
    o = mesh_grid("shaft", P[..., 0].astype(np.float32), P[..., 1].astype(np.float32), P[..., 2].astype(np.float32), uv=False)
    m = bpy.data.materials.new("shaft")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    lw, inv, pow_, em, tr, add, out = (nt.nodes.new(x) for x in ("ShaderNodeLayerWeight", "ShaderNodeMath", "ShaderNodeMath", "ShaderNodeEmission", "ShaderNodeBsdfTransparent", "ShaderNodeAddShader", "ShaderNodeOutputMaterial"))
    inv.operation, pow_.operation = "SUBTRACT", "POWER"
    inv.inputs[0].default_value = 1.0
    pow_.inputs[1].default_value = 3.0
    lw.inputs["Blend"].default_value = .5
    nt.links.new(lw.outputs["Facing"], inv.inputs[1])
    nt.links.new(inv.outputs[0], pow_.inputs[0])
    nt.links.new(pow_.outputs[0], em.inputs["Strength"])
    em.inputs["Color"].default_value = (strength, strength, strength, 1)
    nt.links.new(em.outputs[0], add.inputs[0])
    nt.links.new(tr.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    o.data.materials.append(m)
    o.visible_diffuse = False
    o.visible_glossy = False
    o.visible_shadow = False
    o.visible_transmission = False
    o.visible_volume_scatter = False
    return o


def moonlight(towards, energy, angle=4.0):
    """A sun from the direction towards (ink3d.sun turns the horizontal part round). Seen by nothing wet: a sun in wet stone comes out as white blotches."""
    o = sun((towards[0], towards[1], -towards[2]), energy=energy, angle=angle)
    o.visible_glossy = False
    return o


def light_rays(sc, bounces=12):
    sc.cycles.max_bounces = bounces
    sc.cycles.transmission_bounces = bounces
    sc.cycles.glossy_bounces = 6
    sc.cycles.diffuse_bounces = 2
    sc.cycles.caustics_refractive = False
    sc.cycles.caustics_reflective = False


# ---- the shots -------------------------------------------------------------------------------------------------------------------------------------------
def shot_drop(sc):
    lens, R = 35.0, .26
    view = np.array([-.18, .72, -.67])
    view /= np.linalg.norm(view)
    # the line of sight goes through the drop to the slab beside the column, so the writing lies out of focus under it
    at = slab_at(np.array(3.4), np.array(9.0), np.array(0.0)) - view * 13
    cam_at = at - view * 1.78
    cam = camera(tuple(cam_at), tuple(at + np.array([.02, 0, .1])), lens=lens)
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 1.78 - R * 1.1
    cam.data.dof.aperture_fstop = 2.8
    light_rays(sc)
    water = glass()
    hanging_drop(at, R, water)
    stalactite(at + [0, 0, R * .82], 7.0, .96, 51, albedo=.62, blunt=.13)
    read_sheet(at, view, R)
    build_slab(T, N)
    build_floor()
    world_light(.05)
    moonlight((.35, .55, .75), 6.0)
    moonlight((-.55, -.62, .56), 3.0)        # the pale of the cave coming back off the far wall, on the side of the stone we see
    roof_hole(np.array([-1.0, 8.0, 29.0]), (.18, .017, -.98), 32, 8.0)
    sheen(at + np.array([-1.2, -2.0, 1.4]), (1.2, 2.0, -1.4), .8, 1.1, 7.0)
    return []


def slab_camera(sc):
    lens = 28.0
    cam_at = (1.0, -17.0, 11.0)
    target = (.1, 3.0, 4.37)
    cam = camera(cam_at, target, lens=lens)
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 25.0
    cam.data.dof.aperture_fstop = .5
    return cam, lens


def shot_slab(sc):
    cam, lens = slab_camera(sc)
    light_rays(sc)
    water = glass()
    build_slab(T, N)
    build_floor()
    up_next = in_the_air(T, water)
    if up_next is not None and SHOT == "junk":
        view = up_next - np.array(cam.matrix_world.translation)
        read_sheet(up_next, view, .7, char_r=.52, width_r=11.0)
    for at, length, base, seed, albedo in (((-9.4, -2.0, 6.0), 16.0, 2.3, 54, .5), ((-3.5, 27.0, 13.4), 14.0, 2.0, 55, .3), ((6.5, 30.0, 12.2), 14.0, 2.6, 56, .3), ((12.0, 26.0, 14.8), 12.0, 2.2, 57, .3)):
        stalactite(np.array(at), length, base, seed, albedo=albedo).visible_shadow = False      # they stand in the dark; the light on the slab comes down past them
    plate = cave_plate(cam, lens, (1000, 1160), 400, dist=150.0)
    world_light(.05)
    moonlight((-.32, .5, .8), 9.0, angle=5.0)
    moonlight((-.12, -.6, .6), 1.2, angle=20.0)        # what comes back off the cave
    roof_hole(np.array([-1.9, 15.0, 36.0]), (.05, -.3, -.95), 34, 11.0)
    shaft = light_shaft((-5.5, 20, 52), (.4, 8.0, 4.0), 7.0, 11.0, strength=.14)
    return [plate, shaft]


def chips(t):
    """The stone opening by itself: splinters and dust coming off the stroke being cut."""
    paths = []
    cut_at = CHANT
    for c, line in enumerate(POEM):
        for k, ch in enumerate(line):
            age = t - cut_at
            start = cut_at
            cut_at += CHANT_STEP
            if not 0 < age < 1.1:
                continue
            r = np.random.default_rng(int(start * 100))
            cx, cs = POEM_X[c], COL_S - .35 - 2.74 * (k + .5)
            for _ in range(90):
                t0 = r.uniform(0, .3)
                a = age - t0
                if a <= 0:
                    continue
                x0, s0 = cx + r.uniform(-.42, .42) * 3.2, cs + 1.6 - 3.2 * t0 / .3 + r.normal(0, .1)
                vx, vs, vn = r.normal(0, 1.8), r.normal(0, 1.8), r.uniform(1.0, 5.2)
                x, s_, n_ = x0 + vx * a, s0 + vs * a, .1 + vn * a - 8 * a * a
                if n_ < .02:
                    continue
                rad = r.uniform(.01, .045)
                q = np.linspace(0, 1, 3)
                paths.append((slab_at(x + q * .06 * r.normal(), s_ + q * .06 * r.normal(), n_ + q * .06), np.full(3, rad, np.float32)))
    return paths


def shot_chant(sc):
    lens = 30.0
    at = slab_at(np.array(-4.3), np.array(3.2), np.array(33.5))
    target = slab_at(np.array(-4.3), np.array(8.7), np.array(0.0))
    cam = camera(tuple(at), tuple(target), lens=lens)
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 33.5
    cam.data.dof.aperture_fstop = .2
    light_rays(sc, 6)
    build_slab(T, N, wide=27.0, focus=(4.0, 7.0, 17.0, 17.0), relief=.005, far=24.6)
    build_floor()
    strands("chips", chips(T), grey_material("stone", .85), shadow=False)
    # the drop that did not fall, close to the lens
    near = np.array(at) + (np.array(target) - np.array(at)) / np.linalg.norm(np.array(target) - np.array(at)) * 9.0 + np.array([2.7, .9, 1.25])
    water = glass()
    hanging_drop(near, .34, water)
    stalactite(near + [0, 0, .34 * 1.42], 8.0, .9, 58, albedo=.55).visible_shadow = False
    world_light(.05)
    moonlight((.55, .25, .8), 8.0, angle=4.0)      # further round now, across the stone: what is cut shows
    return []


SHOTS = dict(drop=shot_drop, junk=shot_slab, right=shot_slab, chant=shot_chant)

if __name__ == "__main__":
    sc = reset(samples=SAMPLES, pct=PCT)
    gone_from_depth = SHOTS[SHOT](sc)
    name = os.path.join(OUT3D, "drip", "work", f"{SHOT}{TAG}")
    render(name, depth=False)
    for ob in gone_from_depth:
        ob.hide_render = True
    render(name)
    print("GRADE python ink_grade.py", name, GRADES[SHOT], flush=True)
