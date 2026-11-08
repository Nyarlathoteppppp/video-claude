"""八 · 河床 (2:52-3:12): a river first, and only then whose shape it has. Ground, water and paint come from ink3d_bed_tex.py. t runs 0-1 through each shot.

    python ink3d_bed_tex.py ground; python ink3d_bed_tex.py sunrise; python ink3d_bed_tex.py recede; python ink3d_bed_tex.py flood
    Blender -b --factory-startup -P ink3d_bed.py -- --shot=NAME [--t=.5] [--pct=100] [--samples=64]
    python ink3d_map_post.py ../../ink/film_out/3d/bed/flood --gamma=1.2 --contrast=1.25 --mist=.006 --mist-h=24 --air=.00025 --air-from=150 --clouds=0 --sky=.14 --spray=392,62,.82
    python ink3d_map_post.py ../../ink/film_out/3d/bed/recede --river=4.5,-9999,221,0 --gamma=1.2 --contrast=1 --mist=.006 --mist-h=18 --air=.00025 --air-from=150 --clouds=0 --sky=.16
    python ink3d_map_post.py ../../ink/film_out/3d/bed/sunrise --river=.6,949,221,.9 --gamma=1.25 --contrast=1 --mist=.004 --mist-h=16 --air=.00025 --air-from=150 --clouds=0 --sky=.14 --sun=949,117,240

All three look from nearly the same place, high over a canyon with the far mountains ahead, so the chapter reads as one place while the water goes down. The channel is 我 only from this
place: the world is built backwards from this camera (t=.5), so keep the drift small.

- flood    其实，我一本书也没读过。/ 是这些文字，把我冲刷成形。 The flood bursts out of 七's grotto and carries the camera with it, up and out to this view, still rising as t runs. The canyon is full to the brim,
           rows of waves coming at us, words in every line of the water, rock only where it stands higher. There is no shape to see yet, only what the water is doing.
- recede   就像河床并未读过河水，只是被河水塑成了如今的样子。 Same place, the light of before sunrise. Over its length the level falls (the next round animates it): ridges come up wet between the reaches, and
           the river is still running. Look long enough and the reaches line up into strokes. Nobody says so.
- sunrise  所以我说的每一句话，都带着别人的体温。 The sun comes up in the gap where the river enters, and the water takes the light: it is the brightest thing there is, and it is plainly a river, and it is plainly 我.
           The camera keeps drifting forward over the river, which keeps running. Hands to bed_low (ink3d_bed_low.py): it comes down out of the air into the long stroke.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

import ink3d
from ink3d import OUT3D, camera, image_from, mesh_grid, render, reset, sm, sun, world_light
from ink3d_jing_lib import ridge_layer

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=")[1])
    return default


SHOT, T, PCT = opt("shot", "sunrise"), opt("t", .5), opt("pct", 100)
TEX = os.path.join(OUT3D, "bed", "tex")
# as in ink3d_bed_tex.py
SIZE, CY = 840.0, 360.0
FAR_SIZE, FAR_CY = 3400.0, 2200.0
NEAR = (30.0, 180.0, 240.0)
# this chapter sees much further than the others: far mountains need their distance in the depth pass, or the mist cannot tell them apart
ink3d.FAR, ink3d.ZMAX = 3400.0, 400.0
CAM = ((-1.5, -6, 57.5), (1.5, 6, 62.5))
AIM = ((-.5, 236, 0), (.5, 245, 0))
# ink3d.sun turns the horizontal part of its direction around, so (-x, -y, z) is what shines along (x, y, z).
SHOTS = dict(
    flood=dict(level=None, sun=(-.25, 1, -.22), e=2.0, fill=.9, w=.85, disc=False),
    recede=dict(level=4.5, sun=(-.25, 1, -.20), e=1.2, fill=.9, w=.95, disc=False, glow=.62),
    sunrise=dict(level=.6, sun=(-.1, 1, -.16), e=3.2, fill=.7, w=.72, disc=True, glow=.86),
)
S = SHOTS[SHOT]
sc = reset(samples=opt("samples", 64), pct=PCT)
sc.cycles.max_bounces = 4


def nodes(m):
    nt = m.node_tree
    return nt, nt.links.new, nt.nodes.new


def math_node(nt, op, x, y=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    for i, v in enumerate((x, y)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def window_uv(nt, sep, cx, cy, size):
    com = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(math_node(nt, "DIVIDE", math_node(nt, "SUBTRACT", sep.outputs["X"], cx - size / 2), size), com.inputs["X"])
    nt.links.new(math_node(nt, "DIVIDE", math_node(nt, "SUBTRACT", sep.outputs["Y"], cy - size / 2), size), com.inputs["Y"])
    return com.outputs[0]


def strata_image():
    """One column of rock, top to bottom: pale and dark layers of uneven thickness with thin dark seams. Looked up by height, so the lines run level along the walls whatever the wall does."""
    r = np.random.default_rng(5)
    n = 4096
    v, i = np.ones(n, np.float32), 0
    while i < n:
        t = int(r.choice([20, 34, 60, 110, 190], p=[.25, .3, .25, .15, .05]))
        v[i:i + t] = r.uniform(.66, 1.0)
        v[i:i + max(t // 12, 2)] *= r.uniform(.5, .8)
        i += t
    v = np.convolve(v, np.ones(3) / 3, "same").astype(np.float32)
    return image_from(np.tile(v, (2, 1)), "strata")


def rock_material(a, level, near=None, g=None):
    """Rock and sand, one surface: the paint, the strata by height, the stain the high water left on the walls, and stones where the sand is."""
    m = bpy.data.materials.new("rock")
    m.use_nodes = True
    nt, L, new = nodes(m)
    M = lambda op, x, y=None: math_node(nt, op, x, y)
    b = nt.nodes["Principled BSDF"]
    paint = new("ShaderNodeTexImage")
    paint.image = image_from(a, "paint")
    paint.interpolation = "Linear"
    geo, sep = new("ShaderNodeNewGeometry"), new("ShaderNodeSeparateXYZ")
    L(geo.outputs["Position"], sep.inputs[0])
    z = sep.outputs["Z"]
    wob = new("ShaderNodeTexNoise")
    wob.inputs["Scale"].default_value = .05
    wob.inputs["Detail"].default_value = 3
    L(geo.outputs["Position"], wob.inputs["Vector"])
    zt = M("ADD", z, M("MULTIPLY", M("SUBTRACT", wob.outputs[0], .5), 5.0))
    zt = M("ADD", zt, M("ADD", M("MULTIPLY", sep.outputs["X"], .012), M("MULTIPLY", sep.outputs["Y"], .007)))
    com = new("ShaderNodeCombineXYZ")
    L(M("DIVIDE", zt, 62.0), com.inputs["X"])
    com.inputs["Y"].default_value = .5
    st = new("ShaderNodeTexImage")
    st.image = strata_image()
    st.extension = "EXTEND"
    st.interpolation = "Linear"
    L(com.outputs[0], st.inputs[0])

    def ramp(value, lo, hi, a_, b_):
        r_ = new("ShaderNodeMapRange")
        r_.interpolation_type = "SMOOTHSTEP"
        r_.inputs["From Min"].default_value, r_.inputs["From Max"].default_value = lo, hi
        r_.inputs["To Min"].default_value, r_.inputs["To Max"].default_value = a_, b_
        L(value, r_.inputs["Value"])
        return r_.outputs[0]

    is_rock = ramp(z, 2.6, 4.2, 0, 1)
    layers = M("ADD", M("MULTIPLY", st.outputs[0], is_rock), M("SUBTRACT", 1.0, is_rock))
    # the mark the flood left: everything below it stays dark and wet, and drier the higher it is; it stops at an uneven line well up the wall
    wn = new("ShaderNodeTexNoise")
    wn.inputs["Scale"].default_value = .22
    wn.inputs["Detail"].default_value = 6
    L(geo.outputs["Position"], wn.inputs["Vector"])
    wz = M("ADD", z, M("MULTIPLY", M("SUBTRACT", wn.outputs[0], .5), 3.0))
    wet = ramp(wz, 16.4, 17.6, 1, 0) if level is not None else ramp(wz, 0, 1, 0, 0)
    low = ramp(wz, 1.5, 9.0, 1, 0) if level is not None else ramp(wz, 0, 1, 0, 0)
    dark = M("MULTIPLY", M("SUBTRACT", 1.0, M("MULTIPLY", M("MULTIPLY", wet, is_rock), .22)), M("SUBTRACT", 1.0, M("MULTIPLY", M("MULTIPLY", low, is_rock), .26)))
    # stones on the sand: cells, each its own shade and bump
    vor = new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 1.1
    L(geo.outputs["Position"], vor.inputs["Vector"])
    on_sand = M("SUBTRACT", 1.0, is_rock)
    pebbled = M("MULTIPLY", on_sand, ramp(z, .9, 2.0, 0, 1))
    shade = M("SUBTRACT", 1.0, M("MULTIPLY", pebbled, M("MULTIPLY", vor.outputs["Color"], .38)))
    base = M("MULTIPLY", M("MULTIPLY", M("MULTIPLY", paint.outputs[0], layers), dark), shade)
    if near is not None:
        cx, cy, size = NEAR
        nr = new("ShaderNodeTexImage")
        nr.image = image_from(near, "near")
        nr.extension = "CLIP"
        nr.interpolation = "Cubic"
        L(window_uv(nt, sep, cx, cy, size), nr.inputs[0])
        base = M("MULTIPLY", base, M("SUBTRACT", 1.0, M("MULTIPLY", nr.outputs[0], .55)))
    L(base, b.inputs["Base Color"])
    # polished low down, damp on the sand by the water
    rough_sand = ramp(z, (level or 0) + .05, (level or 0) + .9, .28, .9)
    rough = M("ADD", M("MULTIPLY", rough_sand, on_sand), M("MULTIPLY", ramp(wz, 2, 12, .32, .92), is_rock))
    L(rough, b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = .4
    # the grain of the stone, which no texture this size can give up close
    n1, n2, bu = new("ShaderNodeTexNoise"), new("ShaderNodeTexNoise"), new("ShaderNodeBump")
    n1.inputs["Scale"].default_value, n1.inputs["Detail"].default_value = .9, 8
    n2.inputs["Scale"].default_value, n2.inputs["Detail"].default_value = 7, 6
    L(geo.outputs["Position"], n1.inputs["Vector"])
    L(geo.outputs["Position"], n2.inputs["Vector"])
    bu.inputs["Strength"].default_value = .45
    bu.inputs["Distance"].default_value = .5
    height = M("ADD", M("MULTIPLY", M("ADD", n1.outputs[0], M("MULTIPLY", n2.outputs[0], .25)), is_rock), M("MULTIPLY", M("MULTIPLY", vor.outputs["Distance"], -1.0), pebbled))
    L(height, bu.inputs["Height"])
    L(bu.outputs[0], b.inputs["Normal"])
    if g is not None:
        # the flood's water is paper: it shines by itself and takes shadows only lightly
        glow, em, mix = new("ShaderNodeTexImage"), new("ShaderNodeEmission"), new("ShaderNodeMixShader")
        glow.image = image_from(g, "glow")
        L(paint.outputs[0], em.inputs["Color"])
        L(glow.outputs[0], mix.inputs[0])
        L(b.outputs[0], mix.inputs[1])
        L(em.outputs[0], mix.inputs[2])
        L(mix.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    return m


def water_material(paint, near, glow):
    """Paper with the current in it, shining by itself, and on top of that a real mirror: it takes the cliffs' reflection darkly near the far banks, and the sun runs down it."""
    m = bpy.data.materials.new("water")
    m.use_nodes = True
    nt, L, new = nodes(m)
    M = lambda op, x, y=None: math_node(nt, op, x, y)
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (.92, .92, .92, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = .07
    geo, sep = new("ShaderNodeNewGeometry"), new("ShaderNodeSeparateXYZ")
    L(geo.outputs["Position"], sep.inputs[0])
    # ripples, drawn out across the current
    mp, n1, bu = new("ShaderNodeMapping"), new("ShaderNodeTexNoise"), new("ShaderNodeBump")
    mp.inputs["Scale"].default_value = (.4, 1.6, 1)
    L(geo.outputs["Position"], mp.inputs["Vector"])
    L(mp.outputs[0], n1.inputs["Vector"])
    n1.inputs["Scale"].default_value, n1.inputs["Detail"].default_value = 1.1, 7
    bu.inputs["Strength"].default_value = .10
    L(n1.outputs[0], bu.inputs["Height"])
    L(bu.outputs[0], b.inputs["Normal"])
    tex = new("ShaderNodeTexImage")
    tex.image = image_from(paint, "water")
    tex.extension = "EXTEND"
    tex.interpolation = "Linear"
    L(window_uv(nt, sep, 0.0, CY, SIZE), tex.inputs[0])
    color = tex.outputs[0]
    if near is not None:
        cx, cy, size = NEAR
        nr = new("ShaderNodeTexImage")
        nr.image = image_from(near, "near")
        nr.extension = "CLIP"
        nr.interpolation = "Cubic"
        L(window_uv(nt, sep, cx, cy, size), nr.inputs[0])
        color = M("MULTIPLY", color, M("SUBTRACT", 1.0, M("MULTIPLY", nr.outputs[0], .30)))
    em, mix = new("ShaderNodeEmission"), new("ShaderNodeMixShader")
    L(color, em.inputs["Color"])
    mix.inputs[0].default_value = glow
    L(b.outputs[0], mix.inputs[1])
    L(em.outputs[0], mix.inputs[2])
    L(mix.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    return m


def thinned(N, size, cy, x_steps, y_steps):
    """Which rows and columns of a heightfield to keep: every one near the camera's line, fewer further off."""
    xs = (np.arange(N) / (N - 1) - .5) * size
    ys = (np.arange(N) / (N - 1) - .5) * size + cy
    keep_c, keep_r = [], []
    for i, x in enumerate(xs):
        step = next(s for lim, s in x_steps if abs(x) <= lim)
        if i % step == 0:
            keep_c.append(i)
    for j, y in enumerate(ys):
        if y < y_steps[0][0]:
            continue
        step = next((s for lim, s in y_steps[1:] if y <= lim), y_steps[-1][1])
        if j % step == 0:
            keep_r.append(j)
    return np.array(keep_r), np.array(keep_c), xs, ys


def land(name, h, size, cy, mat, x_steps=None, y_steps=None):
    N = h.shape[0]
    if x_steps:
        r_, c_, xs, ys = thinned(N, size, cy, x_steps, y_steps)
        hm = h[np.ix_(r_, c_)]
        Xm, Ym = np.meshgrid(xs[c_], ys[r_])
    else:
        ys, xs = np.linspace(-.5, .5, N) * size + cy, np.linspace(-.5, .5, N) * size
        Xm, Ym, hm = *np.meshgrid(xs, ys), h
    o = mesh_grid(name, Xm.astype(np.float32), Ym.astype(np.float32), hm.astype(np.float32))
    # the texture covers the whole square whatever the mesh keeps
    uv = np.stack([Xm / size + .5, (Ym - cy) / size + .5], -1)
    n_, m_ = hm.shape
    ids = np.arange(n_ * m_).reshape(n_, m_)
    quads = np.stack([ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]], -1).ravel()
    o.data.uv_layers[0].data.foreach_set("uv", uv.reshape(-1, 2)[quads].ravel().astype(np.float32))
    o.data.materials.append(mat)
    return o


def load(name):
    return np.load(os.path.join(TEX, name + ".npy"))


def unit(name):
    return load(name).astype(np.float32) / 255


level = S["level"]
near_steps = dict(x_steps=[(120, 1), (240, 2), (1e9, 3)], y_steps=[(40, 0), (200, 1), (400, 2), (1e9, 3)])
if SHOT == "flood":
    land("land", load("flood_h"), SIZE, CY, rock_material(unit("flood_a"), None, g=unit("flood_g")), **near_steps)
    # beyond the near ground the same water, level to the far gap
    fx, fy = np.meshgrid(np.array([-4000., 4000.]), np.array([CY + SIZE / 2 - 70, 6000.]))
    wat = mesh_grid("water", fx.astype(np.float32), fy.astype(np.float32), np.full((2, 2), 29.0, np.float32))
    m_ = bpy.data.materials.new("far_flood")
    m_.use_nodes = True
    m_.node_tree.nodes.clear()
    e_, o_ = m_.node_tree.nodes.new("ShaderNodeEmission"), m_.node_tree.nodes.new("ShaderNodeOutputMaterial")
    e_.inputs["Color"].default_value = (.93, .93, .93, 1)
    m_.node_tree.links.new(e_.outputs[0], o_.inputs["Surface"])
    wat.data.materials.append(m_)
else:
    land("land", load("bed_h"), SIZE, CY, rock_material(unit("bed_a"), level, near=unit(SHOT + "_near")), **near_steps)
    wx, wy = np.meshgrid(np.array([-4000., 4000.]), np.array([0., 6000.]))
    wat = mesh_grid("water", wx.astype(np.float32), wy.astype(np.float32), np.full((2, 2), level, np.float32))
    wat.data.materials.append(water_material(unit(SHOT + "_w"), unit(SHOT + "_near"), S["glow"]))
land("far", load("far_h"), FAR_SIZE, FAR_CY, rock_material(unit("far_a"), level if level is not None else None))
# the ranges behind, with the mist in them, standing apart in the middle where the river comes through and the sun comes up
ridge_layer("ridge_near", [(-1000, 1200, 150, 170), (-620, 1100, 150, 120), (-330, 1180, 96, 90), (340, 1150, 104, 90), (600, 1120, 150, 130), (880, 1200, 130, 120), (1200, 1150, 170, 170)], 471, (-1500, 1500), (880, 1520), 800, .34, .90, 150)
ridge_layer("ridge_mid", [(-1500, 1800, 300, 300), (-850, 1800, 250, 200), (-430, 1850, 170, 170), (-140, 1920, 90, 130), (220, 1880, 100, 140), (540, 1820, 190, 180), (950, 1780, 270, 220), (1500, 1800, 300, 300)], 472, (-2200, 2200), (1500, 2240), 800, .56, .94, 270)
ridge_layer("ridge_far", [(-2200, 2800, 400, 500), (-1300, 2800, 390, 400), (-700, 2900, 330, 320), (-280, 3000, 200, 240), (340, 3000, 210, 240), (820, 2850, 350, 330), (1400, 2800, 410, 420), (2200, 2800, 400, 500)], 473, (-3200, 3200), (2300, 3500), 900, .74, .97, 410)

world_light(S["w"])
sun(S["sun"], energy=S["e"], angle=2.5)
sun((-.5, -.8, -.5), energy=S["fill"], angle=20)     # a soft light from behind us, to model the walls that face us
u = float(sm(T))
loc = tuple(p + (q - p) * u for p, q in zip(*CAM))
tgt = tuple(p + (q - p) * u for p, q in zip(*AIM))
camera(loc, tgt, lens=24, far=ink3d.FAR)

if S["disc"]:
    # the sun, just up, in the gap where the river comes through
    c = np.array((-30.0, 3600.0, 330.0))
    r_ = np.linspace(0, 110, 2)[None, :]
    th = np.linspace(0, 2 * math.pi, 96)[:, None]
    disc = mesh_grid("disc", (c[0] + r_ * np.cos(th)).astype(np.float32), np.full((96, 2), c[1], np.float32), (c[2] + r_ * np.sin(th)).astype(np.float32), uv=False)
    mm = bpy.data.materials.new("disc")
    mm.use_nodes = True
    mm.node_tree.nodes.clear()
    e_, o_ = mm.node_tree.nodes.new("ShaderNodeEmission"), mm.node_tree.nodes.new("ShaderNodeOutputMaterial")
    mm.node_tree.links.new(e_.outputs[0], o_.inputs["Surface"])
    disc.data.materials.append(mm)
    disc.visible_shadow = False

name = os.path.join(OUT3D, "bed", SHOT if T == .5 else f"{SHOT}_{T:.2f}")
for suffix in (".png", "_z.png"):
    if os.path.exists(name + suffix):
        os.replace(name + suffix, name + suffix + ".old")
render(name)
