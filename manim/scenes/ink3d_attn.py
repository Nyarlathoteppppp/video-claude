"""六 · 回头看. A cliff by the river at night. A line of verse is being cut into the rock by nobody, and each character, as it is cut, draws threads of ink
back to the ones before it: thick where it looks hard, a single hair where it barely looks. Same moon as ever.

    Blender -b --factory-startup -P ink3d_attn.py -- --shot=look [--t=SECONDS] [--pct=100] [--samples=128] [--n=2000] [--tag=NAME]
    python ink_grade.py ...                       # the settings for each shot are in GRADES below; ink3d_attn.py prints the command

t is seconds into the chapter. Each shot has a default t, its best moment.

look   读一句话，每个字都回头看别的字。 We come up the river out of the delta's mist with the moon ahead, and the cliff slides in on the left: the camera drifts
       upstream the whole time. 举头望明月 is cut one character at a time; as 月 is finished its threads fly back to 明 and 望 and the older threads go thin.
       Hands over by pushing in on the left-hand column as 低头思 is cut.
lian   书家说：笔断意连。 The push ends a few metres off the rock, level with 思, looking up the column toward the moon. The threads from 思 fly out one after another,
       each with a kick: 低, 头, 明, 月, 望; thin ones to 举 and 头. Fine as the hair at the end of a brush stroke, they land on the far character. Hands over by easing back as the words fade.
guess  下一个字，我来猜。 At an angle to the rock, below 思. A rough blank has been dressed for the next character; in the air in front of it hang four guesses of
       ink vapour, darkest the likeliest, each throwing a shadow on the stone, breathing with the threads that hold them. Hands over as 家 starts to sink into the blank.
fix    猜错了，就改几个数。 家 is cut; then 故 comes down over it, and water runs down the rock carrying 家's ink away to the foot of the cliff, where a few of the numbers
       cut along the waterline change. Above, the vapour characters change darkness. Hands over to the cave (七) by following the water down and in.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

from ink3d import (OUT3D, blur, camera, erode, grey_material, image_from, ink_albedo, massif, mesh_grid, painted_material, render, reset, sm, spectral, sun, water_material, world_light)
from ink3d_carve import TILE, bundle, cut, rock, run_down, sample, sample_wrap, stains, strands, tile

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=")[1])
    return default


SHOT = opt("shot", "look")
DEFAULT_T = dict(look=3.35, lian=8.4, guess=11.6, fix=15.4)
T = opt("t", DEFAULT_T[SHOT])
PCT, SAMPLES, N, TAG = opt("pct", 100), opt("samples", 128), opt("n", 1800), opt("tag", "")
# how to turn each shot into ink
GRADES = dict(look="--night --mist=.012 --mist-h=30 --lift=0 --mist-d=.62 --sky=2.5",
              lian="--night --mist=.012 --mist-h=30 --lift=0 --mist-d=.62 --sky=2.5",
              guess="--night --mist=.012 --mist-h=30 --lift=0 --mist-d=.62 --sky=2.5",
              fix="--night --mist=.012 --mist-h=30 --lift=0 --mist-d=.62 --sky=2.5")

# ---- the cliff: a face through the origin running away to the right, u along it, v up it, w out of it toward the river --------------------------------
PHI = math.radians(50)
NORMAL = np.array([math.sin(PHI), -math.cos(PHI), 0.0])
ALONG = np.array([math.cos(PHI), math.sin(PHI), 0.0])
UP = np.array([0, 0, 1.0])


def world(u, v, w):
    return np.asarray(u)[..., None] * ALONG + np.asarray(v)[..., None] * UP + np.asarray(w)[..., None] * NORMAL


# the verse: right column first, top down
SIZE, STEP, TOP = 2.4, 2.75, 15.4
COL1, COL2 = 7.5, 2.5
PANEL_U, PANEL_HALF = 5.5, 7.6


def place(i):
    """Where the i-th character of 举头望明月低头思故乡 sits on the rock."""
    return (COL1, TOP - STEP * (i + .5)) if i < 5 else (COL2, TOP - STEP * (i - 5 + .5))


VERSE = "举头望明月低头思"
LANDS = [.55, 1.17, 1.79, 2.41, 3.03, 3.6, 4.12, 4.62]
CUT_FOR = .5
LOOKS = {1: {0: .9}, 2: {0: .35, 1: .6}, 3: {0: .1, 1: .12, 2: .5}, 4: {0: .1, 1: .1, 2: .55, 3: 1.0}, 5: {0: .55, 1: .15, 2: .1, 3: .06, 4: .2},
         6: {0: .25, 1: .45, 2: .08, 3: .05, 4: .08, 5: .95}, 7: {0: .45, 1: .15, 2: .7, 3: .7, 4: .85, 5: .75, 6: .8}}
LIAN = 5.2
# the great threads from 思, in the order they tear out: to, half width at the belly, seconds after LIAN
SWEEPS = [(5, .032, .15), (6, .036, .45), (3, .04, .80), (4, .046, 1.25), (2, .042, 1.80), (0, .008, 2.30), (1, .004, 2.60)]


def grid_axis(n, lo, hi, centre, dense, sharp=5):
    """n samples from lo to hi, fine near centre (within about ±dense), coarse far off."""
    s = np.linspace(-1, 1, n)
    a = dense
    up, down = hi - centre - a, centre - lo - a
    return (centre + a * s + np.where(s > 0, up, down) * np.abs(s) ** sharp * np.sign(s)).astype(np.float32)


def grid(shot):
    if shot == "look":
        return grid_axis(N, -70, 90, 6, 18, 5), grid_axis(N, -5, 80, 8, 12, 5)
    if shot == "lian":
        return grid_axis(N, -60, 90, 6, 14, 5), grid_axis(N, -5, 80, 8, 10, 5)
    return grid_axis(N, -60, 90, 2.5, 10, 5), grid_axis(N, -5, 80, 5, 8, 5)


def rock_material(name, albedo):
    m = painted_material(name, albedo, rough=.82)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc, n1, n2, add, bump = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeBump")
    n1.inputs["Scale"].default_value = 9
    n1.inputs["Detail"].default_value = 10
    n1.inputs["Roughness"].default_value = .62
    n2.inputs["Scale"].default_value = 90
    n2.inputs["Detail"].default_value = 6
    add.inputs[1].default_value = 0
    bump.inputs["Strength"].default_value = .16
    bump.inputs["Distance"].default_value = .03
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
    nt.links.new(n1.outputs[0], add.inputs[0])
    nt.links.new(n2.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    return m


# where the moon seems to fall hardest in each shot: the rest of the rock sinks away
FOCUS = dict(look=(PANEL_U, 8, 12, 16), lian=(PANEL_U, 8, 12, 16), guess=(2.6, 5.4, 7.5, 6), fix=(2.8, 4, 8, 7))


def cliff_top(U):
    teeth = 3.5 * sample_wrap(spectral(1024, 1.8, 23), U * 5 + 40, np.zeros_like(U) + 7) + 6 * sample_wrap(spectral(1024, 2.6, 24), U * 2 + 90, np.zeros_like(U) + 3)
    return 2 + 24 * sm((31 - U) / 16) + 40 * sm((12 - U) / 40) ** 1.2 + teeth * (.22 + .78 * sm((6 - U) / 30)) * sm((31 - U) / 6)


def build_cliff(shot, t):
    u1, v1 = grid(shot)
    U, V = np.meshgrid(u1, v1)
    print("cliff", U.shape, "cell", float(np.diff(u1).min()), float(np.diff(v1).min()), flush=True)
    H, crack = rock(U, V, 7)
    # the face is dressed where the verse goes: the large forms stay, the flutes are gone, and the mason's chisel has left its slant in the stone
    edge = 1.5 * sample_wrap(spectral(512, 3.6, 3), U * 2.6, V * 2.6)
    panel = sm((PANEL_HALF + edge - np.abs(U - PANEL_U)) / 3.2) * sm((TOP + 2.4 + edge - V) / 2.4) * sm((V - .9 + edge * .3) / 1.2)
    near = sm((PANEL_HALF + 12 - np.abs(U - PANEL_U)) / 12)
    H = H * (1 - .55 * near)
    worn = sm((PANEL_HALF + 3 + edge - np.abs(U - PANEL_U)) / 3.2) * sm((TOP + 5 + edge - V) / 2.4) * sm((V + 1 + edge * .3) / 1.2)
    H = H * (1 - worn) + blur(H, 40) * worn
    level = float((H * panel).sum() / (panel.sum() + 1e-6))
    H = H * (1 - panel) + (level + .1 * sample_wrap(spectral(1024, 3.4, 4), U * 4, V * 4)) * panel
    crack = crack * (1 - .55 * panel)
    chisel = sample_wrap(spectral(1024, 1.4, 5), (U + V * .55) * 42, (V - U * .55) * 3.2) + .7 * sample_wrap(spectral(1024, 1.4, 6), (U - V * .4) * 30, (V + U * .4) * 3)
    H = H + .0035 * chisel * panel
    A = np.full(U.shape, .64, np.float32)
    A = stains(A, U, V, H, crack, 9, amount=.3 - .2 * panel)
    chip = spectral(512, 1.2, 77)
    stain = np.zeros(U.shape, np.float32)
    for i, ch in enumerate(VERSE):
        p = (t - LANDS[i]) / CUT_FOR
        if p > 0:
            cu, cv = place(i)
            cut(H, A, u1, v1, ch, "kaih", cu, cv, SIZE, depth=.15, slope=1.0, p=min(p, 1), chip=chip, stain=stain)
    # the ink has run down from the strokes, in thin lines
    runs = run_down(stain, .994) * smooth_lines(U, V)
    A = np.clip(A * (1 - .42 * np.clip(runs, 0, 1)), .025, 1)
    cu, cv = place(8)
    if shot in ("guess", "fix"):      # the place for the next character, roughed out with the point of the chisel
        hw = SIZE * .62
        b = sm((hw + .12 * edge - np.abs(U - cu)) / .12) * sm((hw + .12 * edge - np.abs(V - cv)) / .12)
        H -= .035 * b
        H += .006 * b * sample_wrap(spectral(1024, 1.2, 8), (U - V * .3) * 90, (V + U * .3) * 16)
        A = A * (1 - .3 * b) + .68 * .3 * b
    if t > JIA:                       # the wrong guess, and then what is left of it once the water has been over it
        wash = sm((t - FIX) / .9)
        cut(H, A, u1, v1, "家", "kaih", cu, cv, SIZE, depth=.15 - .12 * wash, slope=1.0, p=min((t - JIA) / CUT_FOR, 1), ink=.03 + .39 * wash, chip=chip)
    if t > FIX:
        cut(H, A, u1, v1, "故", "kaih", cu, cv, SIZE, depth=.17, slope=1.1, p=min((t - FIX) / CUT_FOR, 1), ink=.018, chip=chip)
        A = washed(A, U, V, sm((t - FIX - .2) / 1.0))
    for ch, du, dv, fresh in digits(t):
        old = shot in ("look", "lian")      # there from the start, but nothing draws the eye to them until the water does
        cut(H, A, u1, v1, ch, "mono", du, dv, .46, depth=.02 if old else .05, slope=1.2, ink=.42 if old else .05, fresh=fresh, edge=.006)
    # the moon falls hardest where the verse is
    fu, fv, su, sv = FOCUS[shot]
    A = A * (.28 + .72 * np.exp(-(((U - fu) / su) ** 2 + ((V - fv) / sv) ** 2)))
    # wet and dark at the waterline
    A = A * (.2 + .8 * sm((V + .3) / 1.0))
    # the line of the cliff against the sky: high at the near end, a step, then down to a headland and into the river
    top = cliff_top(U)
    # the rows above the working part of the face are shared out between it and the top of the cliff, and the crest rolls back from the river
    k = np.minimum(18.0, top - 1.0)
    Vt = np.where(V > k, k + (V - k) / (v1[-1] - k) * (top - k), V)
    idx = np.clip(np.searchsorted(v1, k[0]), 0, len(v1) - 1)
    Hk = np.take_along_axis(H, idx[None, :], 0)
    q = sm((V - k) / 6) * (1 - sm((top - 18) / 14))
    W = H * (1 - q) + Hk * q
    W = W + 2.2 * np.exp(-np.maximum(V, -3) / .55) + .9 * blur(sample_wrap(spectral(1024, 3.1, 21), U * 2, V * .5), 24) * (.25 + .75 * (1 - panel))
    W = W - .05 * np.maximum(Vt - 18, 0) ** 1.2 - 6 * sm((Vt - (top - 5)) / 5) ** 2
    P = world(U, Vt, W)
    gone = sm((U - 31) / 7)
    P[..., 2] = P[..., 2] * (1 - gone) - 3 * gone
    o = mesh_grid("cliff", P[..., 0].astype(np.float32), P[..., 1].astype(np.float32), P[..., 2].astype(np.float32))
    o.data.materials.append(rock_material("rock", A))
    FACE.update(u=u1, v=v1, w=W)
    return o


FACE = {}


def on_face(u, v):
    """How far out of the rock the face stands at (u, v)."""
    a, b = np.clip(np.searchsorted(FACE["u"], u), 1, len(FACE["u"]) - 1), np.clip(np.searchsorted(FACE["v"], v), 1, len(FACE["v"]) - 1)
    fa, fb = (u - FACE["u"][a - 1]) / (FACE["u"][a] - FACE["u"][a - 1]), (v - FACE["v"][b - 1]) / (FACE["v"][b] - FACE["v"][b - 1])
    w = FACE["w"]
    return (w[b - 1, a - 1] * (1 - fa) + w[b - 1, a] * fa) * (1 - fb) + (w[b, a - 1] * (1 - fa) + w[b, a] * fa) * fb


# the water that takes 家 away: a few channels down the rock from under the character to the river
CHANNELS = [(-.62, 11), (.28, 12), (.92, 13), (-1.15, 15)]


def channel(k, v):
    off, seed = CHANNELS[k]
    cu = place(8)[0] + off
    return cu + .1 * np.sin(v * 1.3 + seed) + .05 * np.sin(v * 3.1 + seed * 2) + .03 * np.sin(v * 7.3 + seed)


def washed(A, U, V, amount):
    """家's ink going down the rock with the water, wet and dark along each channel, spreading a little as it goes, gone pale by the bottom."""
    top = place(8)[1] - SIZE * .38
    for k in range(len(CHANNELS)):
        wide = .08 + .03 * np.maximum(top - V, 0)
        d = (U - channel(k, V)) / wide
        A = A * (1 - amount * .55 * np.exp(-d * d) * sm((top + .6 - V) / 1.2) * sm((V + .2) / 1.5))
    return np.clip(A, .02, 1)


def rills(t):
    """The water itself: white threads following each channel, catching the moon."""
    amount = sm((t - FIX - .2) / 1.0)
    paths = []
    if amount <= 0:
        return paths
    r = np.random.default_rng(71)
    top = place(8)[1] - SIZE * .38
    for k in range(len(CHANNELS)):
        for _ in range(int(6 * amount)):
            start = top + .2 - r.uniform(0, .9)
            v = np.linspace(start, -.05, 180)
            off = r.normal(0, .06) + .05 * np.sin(v * r.uniform(2, 5) + r.uniform(0, 6))
            u = channel(k, v) + off * (.5 + .1 * np.maximum(top - v, 0))
            R = r.uniform(.002, .0055) * (1 + .5 * np.sin(v * r.uniform(3, 9) + r.uniform(0, 6)) ** 2) * sm((start - v) / .4) * sm((start - v) / .8 + .2)
            paths.append((world(u, v, on_face(u, v) + .015), R.astype(np.float32)))
    return paths


NUMBERS = ("0.41 -1.27 0.08 2.33 -0.56 0.91 -0.14 1.60 -0.73 0.22", "1.72 0.35 -0.88 0.62 -2.04 0.17 1.09 -0.31 0.57 2.18", "-0.65 1.14 0.29 -1.46 0.83 -0.07 1.31 0.44 -1.92 0.76")
# which of them the water changes, and to what: row, column in the row, the new digit, seconds after FIX
CHANGED = ((0, 8, "6", .9), (0, 21, "4", 1.05), (1, 3, "9", .95), (1, 17, "3", 1.15), (1, 26, "8", 1.0), (2, 13, "0", 1.1), (2, 6, "7", 1.2), (0, 33, "1", 1.25))


def digits(t):
    """The numbers cut along the waterline. Dark with old ink, except the few that have just changed, which are pale new-cut stone."""
    out = []
    for row, line in enumerate(NUMBERS):
        for col, ch in enumerate(line):
            if ch == " ":
                continue
            fresh = 0.0
            for r_, c_, new, dt in CHANGED:
                if r_ == row and c_ == col and t > FIX + dt:
                    ch, fresh = new, 1.0
            out.append((ch, -4.2 + col * .36 + row * .17, 1.42 - row * .52, fresh))
    return out


def boulder(centre, radius, seed, squash=.72):
    """A block broken off the cliff: flat faces meeting at edges, worn a little, wet and dark toward the water."""
    n = 160
    th, ph = np.mgrid[0:1:complex(0, n), 0:1:complex(0, n)].astype(np.float32)
    a, b = ph * 2 * math.pi, th * math.pi
    r = np.random.default_rng(seed)
    d = np.stack([np.cos(a) * np.sin(b), np.sin(a) * np.sin(b), np.cos(b)], -1)
    planes = r.normal(size=(16, 3))
    planes /= np.linalg.norm(planes, axis=1)[:, None]
    R = np.min(r.uniform(.66, 1.0, 16) / np.maximum(d @ planes.T.astype(np.float32), .05), axis=-1)
    R = blur(np.minimum(R, 1.3), 1.0)
    t = spectral(1024, 2.2, seed)
    R = radius * R * (1 + .025 * sample_wrap(t, ph * 900, th * 450) * np.sin(b))
    x, y, z = centre[0] + R * np.cos(a) * np.sin(b), centre[1] + R * np.sin(a) * np.sin(b), centre[2] + R * squash * np.cos(b)
    o = mesh_grid(f"boulder{seed}", x.astype(np.float32), y.astype(np.float32), z.astype(np.float32))
    wet = np.clip(.56 + .08 * sample_wrap(t, ph * 512, th * 256) - .4 * sm(1 - z / (radius * .7)), .03, 1)
    o.data.materials.append(rock_material(f"boulder{seed}", wet))
    return o


def moon_road(cam_at, look, bearing, count=460, seed=4, strength=1.2):
    """The moon's road on the water: broken white dashes along the line toward it, narrowing away into the mist, as a painter leaves them."""
    r = np.random.default_rng(seed)
    c, s_ = math.cos(math.radians(bearing)), math.sin(math.radians(bearing))
    d = np.array([look[0] * c + look[1] * s_, -look[0] * s_ + look[1] * c, 0])
    side = np.array([d[1], -d[0], 0])
    verts, faces = [], []
    for k in range(count):
        rr = 7 * math.exp(r.uniform(0, math.log(420 / 7)))
        off = r.normal(0, 1) * rr * .055
        half_l = rr * r.uniform(.006, .05) * (1 - min(abs(off) / (rr * .14), .85))
        half_t = rr * rr * r.uniform(.0008, .0028) / max(cam_at[2], .5) * .5 + .004
        base = np.array(cam_at) * [1, 1, 0] + d * rr + side * off + np.array([0, 0, .03])
        n = len(verts)
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(tuple(base + side * sx * half_l + d * sy * half_t))
        faces.append((n, n + 1, n + 2, n + 3))
    me = bpy.data.meshes.new("road")
    me.from_pydata(verts, [], faces)
    o = bpy.data.objects.new("road", me)
    bpy.context.collection.objects.link(o)
    m = bpy.data.materials.new("road")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em, out = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    em.inputs["Color"].default_value = (1, 1, 1, 1)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    me.materials.append(m)
    o.visible_diffuse = False
    o.visible_glossy = False
    o.visible_shadow = False
    return o


def smooth_lines(U, V):
    return sm(sample_wrap(spectral(1024, .8, 31), U * 22, V * .15) * 1.1 - .55)


def anchor(i, side, r):
    """A point just off the rock at the edge of character i, on the side facing the way the thread is going."""
    cu, cv = place(i)
    return np.array([cu + side[0] * SIZE * .42 + r.uniform(-.25, .25), cv + side[1] * SIZE * .42 + r.uniform(-.25, .25), .08])


def thread_path(a, b, r, out=1.0):
    """A thread from character a to character b, the way a cursive hand joins two strokes: a long shallow S that leaves the rock a little on the way. Never a loop."""
    ca, cb = np.array(place(a)), np.array(place(b))
    d = cb - ca
    dist = float(np.hypot(*d))
    swing = (.1 + .03 * dist) * r.choice((-1, 1)) * r.uniform(.8, 1.2)
    if ca[0] == cb[0]:
        side = -1 if ca[0] == COL1 else 1
        p0 = anchor(a, (side, np.sign(d[1]) * .6), r)
        p3 = anchor(b, (side, -np.sign(d[1]) * .6), r)
        ctrl1 = p0 + [side * (.3 + .06 * dist), d[1] * .38, (.25 + .07 * dist) * out]
        ctrl2 = p3 + [side * (.05 + .02 * dist), -d[1] * .38, (.2 + .05 * dist) * out]
    else:
        sx = np.sign(d[0])
        p0, p3 = anchor(a, (sx, np.sign(d[1]) * .4), r), anchor(b, (-sx, -np.sign(d[1]) * .4), r)
        ctrl1 = p0 + [d[0] * .42, d[1] * .05 + swing, (.25 + .07 * dist) * out]
        ctrl2 = p3 + [-d[0] * .42, -d[1] * .05 - swing, (.2 + .05 * dist) * out]
    return p0, ctrl1, ctrl2, p3


def weight_look(q):
    """A thread's look from how hard the character looks: half width, hairs, dryness. The heaviest is still only a fine stroke; the lightest a single hair."""
    half = .003 + .032 * q ** 1.5
    return half, 1 + int(26 * q ** 1.4), .15 + .55 * q


def threads(t, r0=.0036):
    r = np.random.default_rng(5)
    paths, beads = [], []
    for i in range(1, len(VERSE)):
        t0 = LANDS[i] + .12
        for j, q in LOOKS[i].items():
            ctrl = thread_path(i, j, r)
            if t < t0:
                continue
            next_t = LANDS[i + 1] if i + 1 < len(VERSE) else LIAN
            prog = 1 - (1 - min((t - t0) / .42, 1)) ** 3
            fade = 1 - .72 * sm((t - next_t - .05) / .5)
            fade *= 1 - .92 * sm((t - GUESS) / .8)
            half, count, dry = weight_look(q)
            count = max(int(count * fade), 1)
            for P, R in bundle(ctrl, half, count, r0, dry, 100 * i + j, prog=prog, thin=.6 + .4 * fade):
                paths.append((world(P[:, 0], P[:, 1], P[:, 2]), R))
    return paths, beads


GUESS = 9.7
JIA = 12.9
FIX = 13.7


def sweeps(t, r0=.0036):
    r = np.random.default_rng(15)
    paths = []
    a = place(7)
    for k, (j, half, dt) in enumerate(SWEEPS):
        t0 = LIAN + dt
        ctrl = thread_path(7, j, r)
        if t < t0:
            continue
        prog = 1 - (1 - min((t - t0) / .45, 1)) ** 3
        fade = (1 - .93 * sm((t - GUESS) / .9)) * (1 - .6 * sm((t - FIX - 1) / 1))
        count = max(int((1 + 760 * half) * fade), 1)
        for P, R in bundle(ctrl, half, count, r0, .6, 300 + k, prog=prog, twist=1.3, thin=.6 + .4 * fade):
            paths.append((world(P[:, 0], P[:, 1], P[:, 2]), R))
    return paths


def threads_in_air(paths, mat):
    """All the hairs show, but few throw a shadow: a dense shadow would read as a second, thicker thread on the stone."""
    strands("threads", paths[::5], mat)
    strands("threads_", [p for k, p in enumerate(paths) if k % 5], mat, shadow=False)


def wet_ink(value=.02, rough=.3):
    m = bpy.data.materials.new("ink")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (value, value, value, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = 1.0
    return m


def moonlight(towards, energy, angle=1.2):
    """A sun from the direction towards. ink3d.sun() turns x and y round, so they are handed over reversed."""
    return sun((towards[0], towards[1], -towards[2]), energy=energy, angle=angle)


def srgb_to_linear(c):
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)


def sky_plane(cam, lens, moon_px, moon_r, dist=700.0, spread=1.25):
    """The sky, painted for this camera: the moon as bare paper with a little grey in it, its glow in the thin cloud around it, darkness further off."""
    w_, h_ = 2048, 1152
    y, x = np.mgrid[:h_, :w_].astype(np.float32)
    px = (x / (w_ - 1) - .5) * 1920 * spread + 960
    py = (y / (h_ - 1) - .5) * 1080 * spread + 540
    d = np.hypot(px - moon_px[0], py - moon_px[1])
    fx, fy = x * 1024 / w_, y * 1024 / w_
    c1 = sample_wrap(spectral(1024, 3.2, 61), fx * .9, fy * 2.6)
    c2 = sample_wrap(spectral(1024, 2.4, 62), fx * 1.6, fy * 5.5)
    c3 = sample_wrap(spectral(1024, 2.8, 63), fx * 3.0, fy * 3.0)
    clouds = np.clip(.62 * c1 + .38 * c2, -2.5, 2.5)
    # how much light is in the air at each point
    glow = np.exp(-d / (moon_r * 2.9)) * 1.6 + np.exp(-d / (moon_r * 8.5)) * .85 + .1
    ink = 2.65 - glow * (1.05 + .55 * clouds)
    ink = np.clip(ink + .22 * np.clip(clouds, -1, 1), .42, 2.9)
    ring = np.exp(-((d - moon_r * 3.3) / (moon_r * .32)) ** 2) * .16
    ink = ink - ring
    # the moon, seen through a thin scarf of cloud
    disc = sm((moon_r + 1.2 - d) / 2.4)
    dx, dy = px - moon_px[0], py - moon_px[1]      # the same face every time
    mare = np.clip(.5 * sample_wrap(spectral(1024, 2.9, 64), dx * .9 + 1289, dy * .9 + 716) + .5 * sample_wrap(spectral(1024, 2.1, 65), dx * 2.6 + 3692, dy * 2.6 + 2054), -2, 2)
    scarf = sm(sample_wrap(spectral(1024, 3.0, 66), dx * .2346 + 389, dy * 1.4495 + 1340) * .9 - .25) * .5
    face = .04 + .12 * sm(mare * .7 + .5) + scarf
    ink = ink * (1 - disc) + face * disc
    L = np.clip(1 - ink / 3.1, 0, 1)
    e = srgb_to_linear(L)
    img = np.ones((h_, w_, 4), np.float32)
    img[..., 0] = img[..., 1] = img[..., 2] = e
    im = bpy.data.images.new("sky", w_, h_, float_buffer=True, alpha=False)
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(img.ravel())
    im.pack()
    m = bpy.data.materials.new("sky")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    t, em, out = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    t.image = im
    nt.links.new(t.outputs[0], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    bpy.context.view_layer.update()
    half_w = dist * (18 / lens) * spread
    half_h = half_w * 9 / 16
    me = bpy.data.meshes.new("sky")
    me.from_pydata([(-half_w, -half_h, -dist), (half_w, -half_h, -dist), (half_w, half_h, -dist), (-half_w, half_h, -dist)], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name="UV")
    uv.data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
    o = bpy.data.objects.new("sky", me)
    bpy.context.collection.objects.link(o)
    o.matrix_world = cam.matrix_world
    me.materials.append(m)
    o.visible_diffuse = False
    o.visible_shadow = False
    o.visible_volume_scatter = False
    o.visible_transmission = False
    return o


def far_land(cam_at, look, peaks):
    """Mountains placed by where they stand in the picture. peaks: (degrees right of the camera's axis, range, height, radius)."""
    size = 1000.0
    n = 1500
    y, x = np.mgrid[:n, :n].astype(np.float32)
    X, Y = (x / (n - 1) - .5) * size + cam_at[0] + look[0] * 340, (y / (n - 1) - .5) * size + cam_at[1] + look[1] * 340
    placed = []
    for b, rng_, top, rad in peaks:
        c, s_ = math.cos(math.radians(b)), math.sin(math.radians(b))
        d = (look[0] * c + look[1] * s_, -look[0] * s_ + look[1] * c)
        placed.append((cam_at[0] + d[0] * rng_, cam_at[1] + d[1] * rng_, top, rad))
    h = massif(X, Y, placed, 57)
    h = erode(h, size / n, 58, wavelength=30, octaves=6, gain=.5, strength=.25)
    h = h - 4 * (1 - sm(h / 7))
    o = mesh_grid("land", X, Y, h)
    o.data.materials.append(painted_material("hills", ink_albedo(np.maximum(h, 0), 59, size / n)))
    return o


def river(reach=900.0):
    xy = np.array([[-reach, reach], [-reach, reach]], np.float32)
    wx, wy = np.meshgrid([-reach, reach], [-reach, reach * 1.2])
    o = mesh_grid("river", wx.astype(np.float32), wy.astype(np.float32), np.zeros((2, 2), np.float32))
    m = water_material(tint=.72, rough=.035, normal_scale=2.2, normal_strength=.07)
    nt = m.node_tree
    # long low swells across the line of sight, so the moon's reflection pulls out into a column
    tc = next(n for n in nt.nodes if n.type == "TEX_COORD")
    noise = next(n for n in nt.nodes if n.type == "TEX_NOISE")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (.5, 3.0, 1)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs[0], noise.inputs["Vector"])
    o.data.materials.append(m)
    return o


def reeds(root, count, spread, height, seed, lean=(.25, .05)):
    """Reeds on the near bank, a dark screen to look through: thin bent blades, some with a plume."""
    r = np.random.default_rng(seed)
    paths = []
    n = 26
    for k in range(count):
        x, y = root[0] + r.normal(0, spread), root[1] + r.normal(0, spread * .45)
        tall = height * r.uniform(.55, 1.15)
        bend = np.array([lean[0] + r.normal(0, .12), lean[1] + r.normal(0, .08)])
        s = np.linspace(0, 1, n)
        P = np.stack([x + bend[0] * tall * s ** 2.1, y + bend[1] * tall * s ** 2.1, -.2 + tall * s * (1 - .12 * s)], 1)
        R = .022 * (1 - s) ** .7 + .0035
        paths.append((P, R.astype(np.float32)))
    return paths


def chips(t):
    """The cut going on with nothing cutting it: splinters and dust coming off the newest character."""
    paths = []
    for i, land in enumerate(LANDS):
        age = t - land
        if not 0 < age < 1.2:
            continue
        cu, cv = place(i)
        r = np.random.default_rng(40 + i)
        for _ in range(170):
            t0 = r.uniform(0, CUT_FOR)
            a = age - t0
            if a <= 0:
                continue
            frac = t0 / CUT_FOR
            x0, y0 = cu + r.uniform(-.42, .42) * SIZE, cv + SIZE * (.5 - frac) + r.normal(0, .08)
            vu, vv, vw = r.normal(0, 1.5), r.uniform(-.8, 2.2), r.uniform(.5, 2.8)
            u, v, w = x0 + vu * a, y0 + vv * a - 4.9 * a * a, .1 + vw * a
            rad = r.uniform(.006, .03)
            s = np.linspace(0, 1, 3)
            P = world(u + s * .03 * r.normal(), v + s * .03 * r.normal(), w + s * .03)
            paths.append((P, np.full(3, rad, np.float32)))
    return paths


# ---- the shots -------------------------------------------------------------------------------------------------------------------------------------
def shot_look(sc):
    lens = 24.0
    mid = world(np.array(5.0), np.array(8.5), np.array(0.0))
    alpha = math.radians(36)
    D = 20.0
    cam_at = mid + D * (math.cos(alpha) * NORMAL - math.sin(alpha) * ALONG)
    cam_at[2] = 2.0
    # the verse sits left of centre, the moon's side stays open
    yaw = math.radians(19)
    to = mid - cam_at
    to[2] = 0
    to /= np.linalg.norm(to)
    look = np.array([to[0] * math.cos(yaw) + to[1] * math.sin(yaw), -to[0] * math.sin(yaw) + to[1] * math.cos(yaw), 0])
    target = cam_at + look * 100 + UP * 100 * math.tan(math.radians(12))
    cam = camera(tuple(cam_at), tuple(target), lens=lens)
    sc.cycles.max_bounces = 4
    sky = sky_plane(cam, lens, (1420, 1080 - 290), 84)
    build_cliff("look", T)
    far_land(cam_at, look, [(8, 210, 26, 38), (17, 190, 18, 34), (27, 230, 30, 44), (38, 200, 24, 40), (4, 360, 62, 60), (14, 340, 44, 54), (24, 380, 58, 66), (35, 340, 66, 60),
                            (46, 360, 84, 70), (0, 540, 110, 90), (12, 560, 96, 90), (25, 540, 128, 96), (38, 560, 110, 90)])
    river()
    moon_road(cam_at, look, math.degrees(math.atan((1420 - 960) / 1278)))
    world_light(.02)
    moonlight(.55 * NORMAL + .75 * ALONG + .38 * UP, 5.4)
    ink = wet_ink()
    p, _ = threads(T, r0=.0065)      # further off: the same fine stroke needs a thicker hair to hold
    threads_in_air(p, ink)
    strands("chips", chips(T), grey_material("stone", .85))
    side = np.array([look[1], -look[0], 0])
    strands("reeds", reeds(np.array(cam_at) + look * 7.0 + side * 5.2, 70, .55, 3.1, 8, lean=(-.2, .05)), grey_material("reed", .02, .7))
    f = lambda u, w: world(np.array(u), np.array(0.0), np.array(w))
    for k, (u, w, rad) in enumerate(((-1.4, 7.8, 1.5), (1.9, 5.4, .8), (12.5, 4.3, 1.2), (15.5, 4.8, .6))):
        boulder(f(u, w), rad, 200 + k)
    return sky


def shot_lian(sc):
    lens = 22.0
    cam_at = world(np.array(-4.5), np.array(8.4), np.array(13.4))
    target = world(np.array(8.8), np.array(9.2), np.array(0.0))
    cam = camera(tuple(cam_at), tuple(target), lens=lens)
    sc.cycles.max_bounces = 4
    moon = (1560, 1080 - 210)
    sky = sky_plane(cam, lens, moon, 58)
    build_cliff("lian", T)
    look = (np.array(target) - cam_at) * [1, 1, 0]
    look /= np.linalg.norm(look)
    far_land(cam_at, look, [(26, 240, 30, 44), (34, 360, 56, 60), (30, 520, 90, 90), (42, 300, 40, 50), (20, 420, 60, 60), (38, 560, 110, 90)])
    river()
    moon_road(cam_at, look, math.degrees(math.atan((moon[0] - 960) / (lens / 36 * 1920))), strength=.7)
    world_light(.02)
    moonlight(.62 * NORMAL + .70 * ALONG + .36 * UP, 5.8, angle=5.0)
    ink = wet_ink()
    p, _ = threads(T)
    threads_in_air(p + sweeps(T), ink)
    return sky


def vapour(ch, cu, cv, w, size, density, seed, layers=14, thick=.42, res=320):
    """A character of ink hanging in still air: a stack of thin sheets, solid in the middle of the stroke, smoking away at the edges and more so at the front and back of
    the stack, so it has a body when seen from the side and throws a soft shadow of its own."""
    t = tile(ch, "kaih")
    y, x = np.mgrid[:res, :res].astype(np.float32)
    span = size * 1.5
    lu, lv = (x / (res - 1) - .5) * span, (y / (res - 1) - .5) * span
    n1, n2, n3 = spectral(512, 2.7, seed), spectral(512, 2.7, seed + 1), spectral(512, 1.9, seed + 2)
    tau = 3.2 * density
    alpha_ = 1 - math.exp(-tau / layers)
    out = []
    for k in range(layers):
        f = k / (layers - 1) * 2 - 1
        fx, fy = x * .7 + 37 * k, y * .7 + 11 * k
        wx, wy = (.006 + .022 * abs(f)) * sample_wrap(n1, fx, fy), (.006 + .022 * abs(f)) * sample_wrap(n2, fx, fy)
        d = sample(t, (lu + wx * size) / size * TILE + t.shape[1] / 2, (-(lv + wy * size)) / size * TILE + t.shape[0] / 2)
        soft = .012 + .05 * abs(f)
        body = sm((-d + .012 * sample_wrap(n3, x * 2, y * 2)) / soft)
        smoke = .22 * np.exp(-np.maximum(d, 0) / (.05 + .04 * abs(f))) * np.clip(.65 + .55 * sample_wrap(n1, fy * .8, fx * .8), 0, 1)
        a = np.clip(alpha_ * (body + smoke), 0, 1)
        m = bpy.data.materials.new(f"vap{seed}_{k}")
        m.use_nodes = True
        nt = m.node_tree
        b = nt.nodes["Principled BSDF"]
        tx = nt.nodes.new("ShaderNodeTexImage")
        tx.image = image_from(a.astype(np.float32), f"vap{seed}_{k}")
        tx.interpolation = "Linear"
        nt.links.new(tx.outputs[0], b.inputs["Alpha"])
        b.inputs["Base Color"].default_value = (.012, .012, .012, 1)
        b.inputs["Roughness"].default_value = .9
        b.inputs["Specular IOR Level"].default_value = 0
        ww = w + f * thick / 2
        gu, gv = np.meshgrid([cu - span / 2, cu + span / 2], [cv - span / 2, cv + span / 2])
        P = world(gu, gv, np.full((2, 2), ww))
        o = mesh_grid(f"vapour{seed}_{k}", P[..., 0].astype(np.float32), P[..., 1].astype(np.float32), P[..., 2].astype(np.float32))
        o.data.materials.append(m)
        out.append(o)
    return out


# the guesses: character, where it hangs (u, v, w), size, how dark before the correction and after
GUESSES = (("家", 2.3, 5.8, 3.2, 2.5, 1.0, .14), ("故", -.9, 4.5, 2.2, 2.1, .46, .5), ("亲", 5.0, 3.6, 2.4, 1.8, .26, .16), ("念", 2.9, 2.95, 1.6, 1.45, .13, .07))


def hanging(t):
    k = sm((t - FIX - .3) / 1.0)
    breath = 1 + .04 * math.sin(t * 2.1)
    objs = []
    for i, (ch, cu, cv, w, size, d0, d1) in enumerate(GUESSES):
        objs += vapour(ch, cu, cv, w, size, (d0 + (d1 - d0) * k) * breath, 500 + 10 * i)
    return objs


def shot_guess(sc):
    lens = 28.0
    cu, cv = place(8)
    cam_at = world(np.array(cu - 3.6), np.array(cv + 1.0), np.array(12.4))
    target = world(np.array(cu + 1.2), np.array(cv + .9), np.array(0.0))
    camera(tuple(cam_at), tuple(target), lens=lens)
    sc.cycles.max_bounces = 4
    sc.cycles.transparent_max_bounces = 64
    build_cliff("guess", T)
    river()
    world_light(.02)
    moonlight(.62 * NORMAL + .70 * ALONG + .36 * UP, 5.3, angle=5.0)
    p, _ = threads(T)
    threads_in_air(p + sweeps(T), wet_ink())
    return hanging(T)


def shot_fix(sc):
    lens = 25.0
    cu, cv = place(8)
    cam_at = world(np.array(cu - 3.0), np.array(3.6), np.array(15.0))
    target = world(np.array(cu + 2.6), np.array(3.4), np.array(0.0))
    camera(tuple(cam_at), tuple(target), lens=lens)
    sc.cycles.max_bounces = 4
    sc.cycles.transparent_max_bounces = 64
    build_cliff("fix", T)
    river()
    world_light(.02)
    moonlight(.62 * NORMAL + .70 * ALONG + .36 * UP, 5.3, angle=5.0)
    p, _ = threads(T)
    threads_in_air(p + sweeps(T), wet_ink())
    strands("rills", rills(T), grey_material("water", .98, .12), shadow=False)
    return hanging(T)


SHOTS = dict(look=shot_look, lian=shot_lian, guess=shot_guess, fix=shot_fix)

if __name__ == "__main__":
    sc = reset(samples=SAMPLES, pct=PCT)
    gone_from_depth = SHOTS[SHOT](sc)        # the sky and the ink in the air stand for nothing in the depth pass: no mist on them
    name = os.path.join(OUT3D, "attn", "work", f"{SHOT}{TAG}")
    render(name, depth=False)
    for ob in gone_from_depth if isinstance(gone_from_depth, list) else [gone_from_depth]:
        ob.hide_render = True
    render(name)
    print("GRADE python ink_grade.py", name, GRADES[SHOT], flush=True)
