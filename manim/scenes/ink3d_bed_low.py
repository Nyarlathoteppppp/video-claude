"""八 · 河床, the low shot: 所以我说的每一句话，都带着别人的体温。 Down on the river among the walls, the sun just up at the far end, a hand's depth of water running over words in the sand, and the sun lying in it.
Its ground is tex_low (kept from the first canyon); its water's paint comes from `python ink3d_bed_tex.py low`.

    Blender -b --factory-startup -P ink3d_bed_low.py -- --shot=bed_low [--t=.5] [--pct=100] [--samples=64]
    python ink3d_map_post.py ../../ink/film_out/3d/bed/bed_low --gamma=1.2 --contrast=1.3 --mist=.012 --mist-h=8 --air=.0025 --air-from=40 --clouds=0 --sky=.2 --sun=960,340,260 --net=1.02,960

Takes over from sunrise as the camera comes down out of the air into the long stroke. It glides upstream along the floor toward the sun, over the words, and the water keeps coming at us. Hands on to 九: night falls, and a small light comes on in the water.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

from ink3d import OUT3D, camera, image_from, mesh_grid, render, reset, sm, sun, world_light

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=")[1])
    return default


T, PCT = opt("t", .5), opt("pct", 100)
TEX = os.path.join(OUT3D, "bed", "tex_low")
SIZE = 440.0
NEAR = (50.0, -50.0, 110.0)      # the sharp words, as near.npy was painted
FLOW = (40.0, -20.0, 180.0)      # the water's paint, as low_w.npy in ink3d_bed_tex.py
LEVEL = 1.02                     # as LOW_LEVEL in ink3d_bed_tex.py
CROP = (-110, 120, -115, 150)
SUN = (-.26, .95, -.20)          # ink3d.sun turns the horizontal part of its direction around, so (-x, -y, z) is what shines along (x, y, z)
CAM, AIM = ((50.5, -74, 4.8), (43, -52, 4.6)), ((22, 22, 9), (19, 34, 9))
sc = reset(samples=opt("samples", 64), pct=PCT)
sc.cycles.max_bounces = 5


def strata_image():
    """One column of rock, top to bottom: pale and dark layers of uneven thickness with thin dark seams. Looked up by height, so the lines run level along the walls whatever the wall does."""
    r = np.random.default_rng(5)
    n = 4096
    v, i = np.ones(n, np.float32), 0
    while i < n:
        t = int(r.choice([20, 34, 60, 110, 190], p=[.25, .3, .25, .15, .05]))
        v[i:i + t] = r.uniform(.60, 1.0)
        v[i:i + max(t // 12, 2)] *= r.uniform(.45, .8)
        i += t
    v = np.convolve(v, np.ones(3) / 3, "same").astype(np.float32)
    return image_from(np.tile(v, (2, 1)), "strata")


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


def window_uv(nt, x, y, cx, cy, size):
    com = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(math_node(nt, "DIVIDE", math_node(nt, "SUBTRACT", x, cx - size / 2), size), com.inputs["X"])
    nt.links.new(math_node(nt, "DIVIDE", math_node(nt, "SUBTRACT", y, cy - size / 2), size), com.inputs["Y"])
    return com.outputs[0]


def ramp(nt, value, lo, hi, a_, b_):
    r_ = nt.nodes.new("ShaderNodeMapRange")
    r_.interpolation_type = "SMOOTHSTEP"
    r_.inputs["From Min"].default_value, r_.inputs["From Max"].default_value = lo, hi
    r_.inputs["To Min"].default_value, r_.inputs["To Max"].default_value = a_, b_
    nt.links.new(value, r_.inputs["Value"])
    return r_.outputs[0]


def rock_material(a, g, near):
    """The first canyon's rock, unchanged, and its floor, now under water: wet and darker, the words seen through ripples that shift them a little, and the sun's net of light moving over them."""
    m = bpy.data.materials.new("rock")
    m.use_nodes = True
    nt = m.node_tree
    L, new = nt.links.new, nt.nodes.new
    M = lambda op, x, y=None: math_node(nt, op, x, y)
    b = nt.nodes["Principled BSDF"]
    paint, glow = new("ShaderNodeTexImage"), new("ShaderNodeTexImage")
    paint.image, glow.image = image_from(a, "paint"), image_from(g, "glow")
    paint.interpolation = "Cubic"
    glow.interpolation = "Linear"
    geo, sep = new("ShaderNodeNewGeometry"), new("ShaderNodeSeparateXYZ")
    L(geo.outputs["Position"], sep.inputs[0])
    # which layer of rock: height, tilted, with a wobble
    wob = new("ShaderNodeTexNoise")
    wob.inputs["Scale"].default_value = .05
    wob.inputs["Detail"].default_value = 3
    L(geo.outputs["Position"], wob.inputs["Vector"])
    z = M("ADD", sep.outputs["Z"], M("MULTIPLY", M("SUBTRACT", wob.outputs[0], .5), 5.0))
    z = M("ADD", z, M("ADD", M("MULTIPLY", sep.outputs["X"], .012), M("MULTIPLY", sep.outputs["Y"], .007)))
    com = new("ShaderNodeCombineXYZ")
    L(M("DIVIDE", z, 62.0), com.inputs["X"])
    com.inputs["Y"].default_value = .5
    st = new("ShaderNodeTexImage")
    st.image = strata_image()
    st.extension = "EXTEND"
    st.interpolation = "Linear"
    L(com.outputs[0], st.inputs[0])
    # how wet: the lower walls still run with water, and it stops at an uneven line
    wn = new("ShaderNodeTexNoise")
    wn.inputs["Scale"].default_value = .22
    wn.inputs["Detail"].default_value = 6
    L(geo.outputs["Position"], wn.inputs["Vector"])
    wz = M("ADD", sep.outputs["Z"], M("MULTIPLY", M("SUBTRACT", wn.outputs[0], .5), 8.0))
    wet = ramp(nt, wz, 3.5, 13.0, 1, 0)
    dark = M("SUBTRACT", 1.0, M("MULTIPLY", wet, .42))
    L(M("MULTIPLY", M("MULTIPLY", paint.outputs[0], st.outputs[0]), dark), b.inputs["Base Color"])
    rough = new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value, rough.inputs["To Max"].default_value = .9, .12
    L(wet, rough.inputs["Value"])
    L(rough.outputs[0], b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = .5
    # the grain of the stone, which no texture this size can give up close
    n1, n2, bu = new("ShaderNodeTexNoise"), new("ShaderNodeTexNoise"), new("ShaderNodeBump")
    n1.inputs["Scale"].default_value, n1.inputs["Detail"].default_value = .9, 8
    n2.inputs["Scale"].default_value, n2.inputs["Detail"].default_value = 7, 6
    L(geo.outputs["Position"], n1.inputs["Vector"])
    L(geo.outputs["Position"], n2.inputs["Vector"])
    bu.inputs["Strength"].default_value = .5
    bu.inputs["Distance"].default_value = .5
    L(M("ADD", n1.outputs[0], M("MULTIPLY", n2.outputs[0], .25)), bu.inputs["Height"])
    L(bu.outputs[0], b.inputs["Normal"])
    # the floor is paper and shines by itself. Under the water it is darker and the words shift with the ripples
    under = ramp(nt, sep.outputs["Z"], LEVEL - .02, LEVEL + .22, 1, 0)
    sw = new("ShaderNodeTexNoise")
    sw.inputs["Scale"].default_value, sw.inputs["Detail"].default_value = 1.3, 3
    L(geo.outputs["Position"], sw.inputs["Vector"])
    sc_ = new("ShaderNodeSeparateColor")
    L(sw.outputs["Color"], sc_.inputs[0])
    wx = M("ADD", sep.outputs["X"], M("MULTIPLY", M("SUBTRACT", sc_.outputs[0], .5), .55))
    wy = M("ADD", sep.outputs["Y"], M("MULTIPLY", M("SUBTRACT", sc_.outputs[1], .5), .85))
    nr = new("ShaderNodeTexImage")
    nr.image = image_from(near, "near")
    nr.extension = "CLIP"
    nr.interpolation = "Cubic"
    L(window_uv(nt, wx, wy, *NEAR), nr.inputs[0])
    color = M("MULTIPLY", paint.outputs[0], M("SUBTRACT", 1.0, M("MULTIPLY", nr.outputs[0], .44)))
    color = M("MULTIPLY", color, M("SUBTRACT", 1.0, M("MULTIPLY", under, .13)))
    # the sun's net: the zero lines of a noise, thin and always curved
    mp, ca = new("ShaderNodeMapping"), new("ShaderNodeTexNoise")
    mp.inputs["Scale"].default_value = (1.0, .55, 1)
    L(geo.outputs["Position"], mp.inputs["Vector"])
    L(mp.outputs[0], ca.inputs["Vector"])
    ca.inputs["Scale"].default_value, ca.inputs["Detail"].default_value, ca.inputs["Distortion"].default_value = 1.15, 2, .9
    net = ramp(nt, M("ABSOLUTE", M("SUBTRACT", ca.outputs[0], .5)), .0, .016, 1, 0)
    color = M("ADD", color, M("MULTIPLY", M("MULTIPLY", net, under), .13))
    em, mix = new("ShaderNodeEmission"), new("ShaderNodeMixShader")
    L(color, em.inputs["Color"])
    L(glow.outputs[0], mix.inputs[0])
    L(b.outputs[0], mix.inputs[1])
    L(em.outputs[0], mix.inputs[2])
    L(mix.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    return m


def water_material(paint):
    """A real surface: clear looking down, a mirror looking along, as water is. It takes the walls darkly and the sun in pieces. What the current draws on it comes from the paint: long pale lines and the froth at its edges."""
    m = bpy.data.materials.new("water")
    m.use_nodes = True
    nt = m.node_tree
    L, new = nt.links.new, nt.nodes.new
    nt.nodes.clear()
    geo, sep, mp, n1, n2, bu = (new(k) for k in ("ShaderNodeNewGeometry", "ShaderNodeSeparateXYZ", "ShaderNodeMapping", "ShaderNodeTexNoise", "ShaderNodeTexNoise", "ShaderNodeBump"))
    L(geo.outputs["Position"], sep.inputs[0])
    # ripples lie across the current, and it runs along the canyon
    mp.inputs["Rotation"].default_value = (0, 0, math.atan2(.28, .96))
    mp.inputs["Scale"].default_value = (.45, 1.5, 1)
    L(geo.outputs["Position"], mp.inputs["Vector"])
    L(mp.outputs[0], n1.inputs["Vector"])
    L(mp.outputs[0], n2.inputs["Vector"])
    n1.inputs["Scale"].default_value, n1.inputs["Detail"].default_value = 1.0, 6
    n2.inputs["Scale"].default_value, n2.inputs["Detail"].default_value = 5.0, 4
    bu.inputs["Strength"].default_value = 1.0
    bu.inputs["Distance"].default_value = opt("ripple", .018)
    L(math_node(nt, "ADD", n1.outputs[0], math_node(nt, "MULTIPLY", n2.outputs[0], .22)), bu.inputs["Height"])
    fr, gl, tr, mix, em, mix2, out = (new(k) for k in ("ShaderNodeFresnel", "ShaderNodeBsdfGlossy", "ShaderNodeBsdfTransparent", "ShaderNodeMixShader", "ShaderNodeEmission", "ShaderNodeMixShader", "ShaderNodeOutputMaterial"))
    fr.inputs["IOR"].default_value = 1.5            # a little more mirror than water really is: the walls in it are what says water
    L(bu.outputs[0], fr.inputs["Normal"])
    L(bu.outputs[0], gl.inputs["Normal"])
    gl.inputs["Roughness"].default_value = .008
    gl.inputs["Color"].default_value = (1, 1, 1, 1)
    L(fr.outputs[0], mix.inputs[0])
    L(tr.outputs[0], mix.inputs[1])
    L(gl.outputs[0], mix.inputs[2])
    tex = new("ShaderNodeTexImage")
    tex.image = image_from(paint, "flow")
    tex.extension = "CLIP"
    tex.interpolation = "Linear"
    L(window_uv(nt, sep.outputs["X"], sep.outputs["Y"], *FLOW), tex.inputs[0])
    em.inputs["Color"].default_value = (.99, .99, .99, 1)
    L(tex.outputs[0], mix2.inputs[0])
    L(mix.outputs[0], mix2.inputs[1])
    L(em.outputs[0], mix2.inputs[2])
    L(mix2.outputs[0], out.inputs["Surface"])
    return m


def mirror_sky(w, glossy=1.0):
    """What lights the rock is one sky; what lies in the water is a brighter one with slow cloud in it, or the water would come out greyer than the sand under it."""
    nt = bpy.context.scene.world.node_tree
    bg = nt.nodes["Background"]
    lp, tc, cl, rng, mx = (nt.nodes.new(k) for k in ("ShaderNodeLightPath", "ShaderNodeTexCoord", "ShaderNodeTexNoise", "ShaderNodeMapRange", "ShaderNodeMix"))
    cl.inputs["Scale"].default_value, cl.inputs["Detail"].default_value = 2.2, 4
    nt.links.new(tc.outputs["Generated"], cl.inputs["Vector"])
    rng.inputs["From Min"].default_value, rng.inputs["From Max"].default_value = .36, .64
    rng.inputs["To Min"].default_value, rng.inputs["To Max"].default_value = .66, 1.0
    nt.links.new(cl.outputs[0], rng.inputs["Value"])
    nt.links.new(lp.outputs["Is Glossy Ray"], mx.inputs[0])
    mx.inputs[2].default_value = w
    nt.links.new(math_node(nt, "MULTIPLY", rng.outputs[0], glossy), mx.inputs[3])
    nt.links.new(mx.outputs[0], bg.inputs[1])


def load(name):
    return np.load(os.path.join(TEX, name + ".npy"))


h = load("bed_h")
a = load("bed_a").astype(np.float32) / 255
g = load("bed_g").astype(np.float32) / 255
N = h.shape[0]
y, x = np.mgrid[:N, :N].astype(np.float32)
X, Y = (x / (N - 1) - .5) * SIZE, (y / (N - 1) - .5) * SIZE
x0, x1, y0, y1 = CROP
ix0, ix1 = int((x0 + SIZE / 2) / SIZE * N), int((x1 + SIZE / 2) / SIZE * N)
iy0, iy1 = int((y0 + SIZE / 2) / SIZE * N), int((y1 + SIZE / 2) / SIZE * N)
Xm, Ym, hm = X[iy0:iy1, ix0:ix1], Y[iy0:iy1, ix0:ix1], h[iy0:iy1, ix0:ix1]
land = mesh_grid("land", Xm, Ym, hm)
# the texture covers the whole ground: the uvs follow the crop
uv = np.stack([(Xm + SIZE / 2) / SIZE, (Ym + SIZE / 2) / SIZE], -1)
n_, m_ = hm.shape
ids = np.arange(n_ * m_).reshape(n_, m_)
quads = np.stack([ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]], -1).ravel()
land.data.uv_layers[0].data.foreach_set("uv", uv.reshape(-1, 2)[quads].ravel().astype(np.float32))
land.data.materials.append(rock_material(a, g, load("near").astype(np.float32) / 255))

wx, wy = np.meshgrid(np.array(CROP[:2], np.float32), np.array(CROP[2:], np.float32))
water = mesh_grid("water", wx, wy, np.full((2, 2), LEVEL, np.float32))
water.data.materials.append(water_material(load("low_w").astype(np.float32) / 255))

world_light(.40)
mirror_sky(.40)
sun(SUN, energy=4.6, angle=2.5)
u = float(sm(T))
loc = tuple(p + (q - p) * u for p, q in zip(*CAM))
tgt = tuple(p + (q - p) * u for p, q in zip(*AIM))
camera(loc, tgt, lens=22)

# the sun itself, a little above the far end of the canyon
d = np.array(SUN, float)
d = np.array([d[0], d[1], -d[2]]) / np.linalg.norm(d)     # toward the sun, undoing ink3d.sun's turn
c = np.array(loc) + d * 385
th = np.linspace(0, 2 * math.pi, 96)
r_ = np.linspace(0, 26.0, 2)[None, :]
right = np.cross(d, (0, 0, 1))
right /= np.linalg.norm(right)
up_ = np.cross(right, d)
Dx = c[0] + r_ * np.cos(th)[:, None] * right[0] + r_ * np.sin(th)[:, None] * up_[0]
Dy = c[1] + r_ * np.cos(th)[:, None] * right[1] + r_ * np.sin(th)[:, None] * up_[1]
Dz = c[2] + r_ * np.cos(th)[:, None] * right[2] + r_ * np.sin(th)[:, None] * up_[2]
disc = mesh_grid("disc", Dx.astype(np.float32), Dy.astype(np.float32), Dz.astype(np.float32), uv=False)
mm = bpy.data.materials.new("disc")
mm.use_nodes = True
mm.node_tree.nodes.clear()
e_, o_ = mm.node_tree.nodes.new("ShaderNodeEmission"), mm.node_tree.nodes.new("ShaderNodeOutputMaterial")
e_.inputs["Strength"].default_value = 2.5
mm.node_tree.links.new(e_.outputs[0], o_.inputs["Surface"])
disc.data.materials.append(mm)
disc.visible_shadow = False

name = os.path.join(OUT3D, "bed", "bed_low" if T == .5 else f"bed_low_{T:.2f}")
for suffix in (".png", "_z.png"):
    if os.path.exists(name + suffix):
        os.replace(name + suffix, name + suffix + ".old")
render(name)
