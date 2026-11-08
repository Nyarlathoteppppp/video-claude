"""一 · 鬼夜哭: the world before writing, the sky answering it, the marsh at night, and the source of the river. Storyboard ink3d.scenes.md.
The marsh shots live in ink3d_marsh.py; this is still the script to run.

    Blender -b --factory-startup -P ink3d_myth.py -- --shot=NAME [--t=SECONDS] [--pct=100] [--samples=64]
    python ink3d_myth_prep.py all                       # first: rock, carved glyphs, mud (scipy and fonts live outside Blender)

Shots, in film order. Time t runs from 0 within the shot.

clouds  Comes straight out of the page: the ink closes over the book and we are falling through cloud at night. The cloud breaks, peaks stand
        out of it. Moves: slow descent, a slight roll settling level. Hands on by diving toward the valley, where the river mud is.
tracks  Skims low along the river mud. Prints pressed in it light up one by one away from the lens, and the light lifts off them. Hands on by
        tilting up to follow the sparks into the cloud.
grow    The camera lifts over the mud as the toes of the last prints run on, cross, and become the strokes of the sign for bird. The story's turn: tracks becoming writing.
        Hands on by the finished sign rising off the mud into the sky, with the other four after it.
rift    Looks up from the valley. The signs go into the cloud one by one, each lighting only the patch it strikes; the last tears the sky along a jagged line, and columns of light
        falls through it. Hands on by the first grains leaving the tear.
grain   At ground level in the light: heads of millet in front, grain coming down in ropes against a white sky, a floor of separate grains building. Hands on by the
        light narrowing as the tear closes overhead into the moon.
marsh   The tear has closed into a full moon, with moonlit cloud round it. Low over a marsh at night: reeds all laid over one way, old burial mounds, stones with
        nothing written on them, hundreds of lights hanging over the water. Rings spread on the still water and nothing has fallen. Nothing here is a ghost and nothing
        is crying. Moves: creeps in on the nearest light as it trembles, each tremble a ring. Hands on to lie by getting close enough to see its reflection.
lie     One light over the water, close. Its reflection has slid off to the side, while the lights behind it reflect where they should. Moves: holds; the reflection drifts.
        Hands on by lifting to the cloud, where the signs are still glowing.
bare_dark, bare_lit  One valley from one lens, seen from high up. Dark: grey, mounds and stones only just there, not a light anywhere: it looks empty. Lit: the same valley crowded
        with thousands of small lights, on the mounds, round the stones, over the water and in it. The sweep of light from the signs in the cloud is composited between the two.
        Hands on as the lights it has passed begin to fall.
fall, fall_plate  Down at the water's skin: one drop of the blackest ink going in, the moment the crying on the sound track stops. fall_plate is the same lens and water without the
        drop or the ink, for compositing them over it, and fall_flat the same without the rings, for before it lands. The marsh behind it is soft, the lights still coming
        down, and from their rings threads of ink come through the water to this bloom and leave together toward the lens. Moves: follows the thread as it runs low across the
        water, over a lip of stone, and becomes a thin stream: the river's source. Hands on to 二, which begins on this thread.
lone    The same marsh, the same lens, later: all of it gone out but one light at the water's edge. Chapter 十二 comes back here.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

from ink3d import (OUT3D, blur, camera, erode, image_from, ink_albedo, lerp, massif, mesh_grid, painted_material, render, reset, sm, spectral, sun, water_material, world_light)

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
TEX = os.path.join(OUT3D, "myth", "tex")


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=")[1])
    return default


# ---- things more than one shot uses ---------------------------------------------------------------------------------------------------
def shine(direction, energy=3.0, angle=4.0):
    """A sun travelling along direction. ink3d.sun turns its light half a circle about the vertical, so it gets the opposite heading."""
    return sun((-direction[0], -direction[1], direction[2]), energy, angle)


def fresh(name):
    """Clears the last attempt at this frame out of render()'s way, keeping it as .old."""
    for suffix in (".png", "_z.png"):
        if os.path.exists(name + suffix):
            os.replace(name + suffix, name + suffix + ".old")


def shoot(name, hidden_from_depth=()):
    """A beauty pass, then a depth pass without the things that have no solid surface (moon, halo, volumes)."""
    fresh(name)
    render(name, depth=False)
    for o in hidden_from_depth:
        o.hide_render = True
    render(name, depth=True)


def paint(name, albedo, rough):
    """A surface with both its colour and its gloss painted: wet ink shines, dry stone does not."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    ta, tr = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeTexImage")
    ta.image, tr.image = image_from(albedo, name + "_a"), image_from(rough, name + "_r")
    ta.interpolation = tr.interpolation = "Cubic"
    nt.links.new(ta.outputs[0], b.inputs["Base Color"])
    nt.links.new(tr.outputs[0], b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = .55
    return m


def image_emission(name, value):
    """A surface that shines by a picture, whatever falls on it."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    ti, e, o = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    ti.image = image_from(value, name + "_e")
    nt.links.new(ti.outputs[0], e.inputs[0])
    nt.links.new(e.outputs[0], o.inputs["Surface"])
    return m


def emission(name, value=1.0, strength=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e, o = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    e.inputs[0].default_value = (value, value, value, 1)
    e.inputs[1].default_value = strength
    nt.links.new(e.outputs[0], o.inputs["Surface"])
    return m


def moon(loc, radius, cam_loc):
    """A full moon as an ink painter leaves it: a clean round of bare paper with a slightly soft edge, no markings, a breath of light in the air around it. One card turned to the lens, in a list, to keep it out of the depth pass."""
    n = 1024
    y, x = np.mgrid[:n, :n].astype(np.float32)
    r_ = np.hypot(x / (n - 1) * 2 - 1, y / (n - 1) * 2 - 1)
    spread = 5.5
    disc = sm((1 / spread - r_) * spread / .05)
    glow = .5 * np.clip(1 - r_, 0, 1) ** 2.6
    alpha = np.maximum(disc, glow).astype(np.float32)
    loc_, eye = np.array(loc, float), np.array(cam_loc, float)
    d = (loc_ - eye) / np.linalg.norm(loc_ - eye)
    right = np.cross(d, (0, 0, 1.0))
    right /= np.linalg.norm(right)
    up = np.cross(right, d)
    X, Y, Z = [np.zeros((2, 2), np.float32) for _ in range(3)]
    for j, dv in enumerate((-1, 1)):
        for i, du in enumerate((-1, 1)):
            X[j, i], Y[j, i], Z[j, i] = loc_ + (right * du + up * dv) * radius * spread
    card = mesh_grid("moon", X, Y, Z)
    m = bpy.data.materials.new("moon")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    ti, tr, e, mx, o = (nt.nodes.new(k) for k in ("ShaderNodeTexImage", "ShaderNodeBsdfTransparent", "ShaderNodeEmission", "ShaderNodeMixShader", "ShaderNodeOutputMaterial"))
    ti.image = image_from(alpha, "moon_a")
    nt.links.new(ti.outputs[0], mx.inputs[0])
    nt.links.new(tr.outputs[0], mx.inputs[1])
    nt.links.new(e.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], o.inputs["Surface"])
    card.data.materials.append(m)
    card.visible_shadow = False
    return [card]


def tube_mesh(segments, name, sides=7):
    """Tapered tubes from (p0, p1, r0, r1) segments."""
    verts, faces = [], []
    for p0, p1, r0, r1 in segments:
        p0, p1 = np.array(p0, float), np.array(p1, float)
        d = p1 - p0
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        d /= L
        up = np.array([0, 0, 1.0]) if abs(d[2]) < .9 else np.array([1.0, 0, 0])
        a = np.cross(d, up)
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        base = len(verts)
        for c, r in ((p0, r0), (p1, r1)):
            for k in range(sides):
                th = 2 * math.pi * k / sides
                verts.append(tuple(c + (a * math.cos(th) + b * math.sin(th)) * r))
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append((base + k, base + k2, base + sides + k2, base + sides + k))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.polygons.foreach_set("use_smooth", np.ones(len(faces), bool))
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    return o


def points_mesh(name, pos, rad, sub=1):
    """Many small spheres as one mesh, built with numpy: icosahedra, for flecks of light and such."""
    t = (1 + 5 ** .5) / 2
    V = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    V /= np.linalg.norm(V[0])
    F = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)])
    return instances_mesh(name, V, F, pos, np.c_[rad, rad, rad])


def instances_mesh(name, V, F, pos, scale, quat=None):
    """Copies of one small mesh, each moved, scaled and turned, merged into one object through numpy so that a hundred thousand of them stay cheap."""
    k, nv, nf = len(pos), len(V), len(F)
    P = V[None, :, :] * scale[:, None, :]
    if quat is not None:
        w, u = quat[:, None, 0:1], quat[:, None, 1:]
        c = np.cross(u, P)
        P = P + 2 * (w * c + np.cross(u, c))
    P = P + pos[:, None, :]
    me = bpy.data.meshes.new(name)
    me.vertices.add(k * nv)
    me.vertices.foreach_set("co", P.astype(np.float32).ravel())
    corners = (F[None, :, :] + (np.arange(k) * nv)[:, None, None]).ravel()
    me.loops.add(len(corners))
    me.loops.foreach_set("vertex_index", corners.astype(np.int32))
    me.polygons.add(k * nf)
    me.polygons.foreach_set("loop_start", (np.arange(k * nf) * 3).astype(np.int32))
    if hasattr(me.polygons[0], "loop_total") and False:
        pass
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", np.ones(k * nf, bool))
    me.update()
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    return o


def far_mountains(peaks, size, n, seed, y_center=0.0, strength=.24):
    """Painted mountains on a square of land, the same recipe as the look test."""
    y, x = np.mgrid[:n, :n].astype(np.float32)
    X, Y = (x / (n - 1) - .5) * size, (y / (n - 1) - .5) * size + y_center
    h = massif(X, Y, peaks, seed)
    h = erode(h, size / n, seed + 1, wavelength=size * .1, octaves=7, gain=.5, strength=strength)
    return X, Y, np.maximum(h, 0)


def mist_sheet(name, center, width, height, cam, value=.9, density=1.0, seed=0, grain=3.2, top=.5):
    """A bank of mist standing in the air: a card facing the lens with drifting noise for opacity, thinning to nothing at its edges.
    It has no surface, so it stays out of the depth pass: hand what this returns to shoot()."""
    n, m = 512, 1024
    y, x = np.mgrid[:n, :m].astype(np.float32)
    u, v = x / (m - 1), y / (n - 1)
    a = lerp(spectral(1024, grain, seed), x, y) + .45 * lerp(spectral(1024, grain - .9, seed + 1), x, y)
    edge = sm(u / .22) * sm((1 - u) / .22) * sm(v / .12) * sm((1 - v) / (top + .02))
    alpha = np.clip(sm((a + 1.0) / 2.2) * edge * density, 0, 1).astype(np.float32)
    f = np.array(center, float) - np.array(cam, float)
    f[2] = 0
    f /= np.linalg.norm(f)
    r = np.array((f[1], -f[0], 0.0))
    X, Y, Z = [np.zeros((2, 2), np.float32) for _ in range(3)]
    for j, dz in enumerate((-height / 2, height / 2)):
        for i, dx in enumerate((-width / 2, width / 2)):
            X[j, i], Y[j, i], Z[j, i] = np.array(center) + r * dx + np.array((0, 0, dz))
    o = mesh_grid(name, X, Y, Z)
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    mt.blend_method = "BLEND"
    nt = mt.node_tree
    nt.nodes.clear()
    ti, tr, e, mx, out = (nt.nodes.new(k) for k in ("ShaderNodeTexImage", "ShaderNodeBsdfTransparent", "ShaderNodeEmission", "ShaderNodeMixShader", "ShaderNodeOutputMaterial"))
    ti.image = image_from(alpha, name + "_a")
    ti.interpolation = "Cubic"
    e.inputs[0].default_value = (value, value, value, 1)
    nt.links.new(ti.outputs[0], mx.inputs[0])
    nt.links.new(tr.outputs[0], mx.inputs[1])
    nt.links.new(e.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs["Surface"])
    o.data.materials.append(mt)
    o.visible_shadow = False
    return o


def haze_box(name, x, y, z, density):
    """Even air, thick enough over a long distance to take the far things back. Light from a sun shows as shafts in it."""
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    nt = mt.node_tree
    nt.nodes.clear()
    vol, out = nt.nodes.new("ShaderNodeVolumePrincipled"), nt.nodes.new("ShaderNodeOutputMaterial")
    vol.inputs["Color"].default_value = (1, 1, 1, 1)
    vol.inputs["Density"].default_value = density
    nt.links.new(vol.outputs[0], out.inputs["Volume"])
    bpy.ops.mesh.primitive_cube_add(size=1, location=((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2), scale=(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mt)
    return o


def cloud_volume(name, x, y, z, top, thick, density=.12, scale=.012, seed=1.0, cut=None, bottom=None, fray=1.8, albedo=1.0, glow=0.0, glow_map=None):
    """Real cloud: a box of volume whose density comes from layered noise and thins out toward the height top. x, y, z are (min, max).
    cut: optional (image, x0, y0, width, height) of a mask that takes the cloud away where it is 0.
    glow: the cloud's own light where it is thick, as a linear value, broken up by the big noise so the billows show; it sets the grey of a cloud whose real light is elsewhere.
    glow_map: optional (image, x0, y0, width, height, gain) of more of that light, laid on from above: a patch stays where it is put, as no lamp in a cloud will."""
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    nt = mt.node_tree
    nt.nodes.clear()
    geo, sep = nt.nodes.new("ShaderNodeNewGeometry"), nt.nodes.new("ShaderNodeSeparateXYZ")
    nz, nz2 = nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeTexNoise")
    nz.noise_dimensions = nz2.noise_dimensions = "4D"
    nz.inputs["Scale"].default_value, nz.inputs["Detail"].default_value, nz.inputs["Roughness"].default_value = scale, 9.0, .58
    nz2.inputs["Scale"].default_value, nz2.inputs["Detail"].default_value, nz2.inputs["Roughness"].default_value = scale * 4.3, 7.0, .6
    nz.inputs["W"].default_value, nz2.inputs["W"].default_value = seed, seed + 7
    nt.links.new(geo.outputs["Position"], nz.inputs["Vector"])
    nt.links.new(geo.outputs["Position"], nz2.inputs["Vector"])
    nt.links.new(geo.outputs["Position"], sep.inputs[0])

    def math_(op, a, b):
        n_ = nt.nodes.new("ShaderNodeMath")
        n_.operation = op
        for i, v in enumerate((a, b)):
            if isinstance(v, (int, float)):
                n_.inputs[i].default_value = v
            else:
                nt.links.new(v, n_.inputs[i])
        return n_.outputs[0]

    # billows: big noise raises and lowers the top, small noise frays it
    height = math_("DIVIDE", math_("SUBTRACT", top, sep.outputs["Z"]), thick)
    if bottom is not None:
        height = math_("MINIMUM", height, math_("DIVIDE", math_("SUBTRACT", sep.outputs["Z"], bottom), thick))
    big = math_("MULTIPLY", math_("SUBTRACT", nz.outputs[0], .5), 5.0)
    small = math_("MULTIPLY", math_("SUBTRACT", nz2.outputs[0], .5), fray)
    d = math_("MULTIPLY", math_("ADD", math_("ADD", height, big), small), 2.6)
    d = math_("MINIMUM", math_("MAXIMUM", d, 0.0), 1.0)
    if cut is not None:
        img, x0, y0, wid, hei = cut
        map_ = nt.nodes.new("ShaderNodeMapping")
        map_.vector_type = "POINT"
        map_.inputs["Location"].default_value = (x0, y0, 0)
        map_.inputs["Scale"].default_value = (wid, hei, 1)
        ti = nt.nodes.new("ShaderNodeTexImage")
        ti.image = img
        ti.extension = "EXTEND"
        nt.links.new(geo.outputs["Position"], map_.inputs["Vector"])
        nt.links.new(map_.outputs[0], ti.inputs[0])
        d = math_("MULTIPLY", d, ti.outputs[0])
    vol, out = nt.nodes.new("ShaderNodeVolumePrincipled"), nt.nodes.new("ShaderNodeOutputMaterial")
    vol.inputs["Color"].default_value = (albedo, albedo, albedo, 1)
    vol.inputs["Anisotropy"].default_value = .25
    nt.links.new(math_("MULTIPLY", d, density), vol.inputs["Density"])
    if glow > 0:
        mottle = math_("MINIMUM", math_("MAXIMUM", math_("DIVIDE", math_("SUBTRACT", nz.outputs[0], .33), .34), .12), 1.0)
        light = math_("MULTIPLY", mottle, glow * 1.6)
        if glow_map is not None:
            img, x0, y0, wid, hei, gain = glow_map
            map2 = nt.nodes.new("ShaderNodeMapping")
            map2.vector_type = "POINT"
            map2.inputs["Location"].default_value = (x0, y0, 0)
            map2.inputs["Scale"].default_value = (wid, hei, 1)
            ti2 = nt.nodes.new("ShaderNodeTexImage")
            ti2.image = img
            ti2.extension = "EXTEND"
            nt.links.new(geo.outputs["Position"], map2.inputs["Vector"])
            nt.links.new(map2.outputs[0], ti2.inputs[0])
            light = math_("ADD", light, math_("MULTIPLY", ti2.outputs[0], gain))
        nt.links.new(math_("MULTIPLY", math_("MULTIPLY", d, density), light), vol.inputs["Emission Strength"])
    nt.links.new(vol.outputs[0], out.inputs["Volume"])
    bpy.ops.mesh.primitive_cube_add(size=1, location=((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2), scale=(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mt)
    return o


# ---- out of the cloud, over a world with nothing written on it ----------------------------------------------------------------------
def shot_clouds(t=0.0):
    SIZE, N = 1000.0, opt("n", 2000)
    y, x = np.mgrid[:N, :N].astype(np.float32)
    X, Y = (x / (N - 1) - .5) * SIZE, (y / (N - 1) - .5) * SIZE + 150
    peaks = [(-140, 10, 158, 62), (-66, -40, 60, 24), (-226, 96, 112, 54), (138, 74, 118, 58), (214, -10, 62, 30), (70, 230, 96, 50), (-60, 300, 136, 80), (-300, 330, 168, 100),
             (286, 320, 176, 110), (40, 590, 196, 140), (-260, 620, 186, 120), (340, 640, 196, 130), (-350, 30, 80, 40), (350, 110, 92, 46)]
    h = massif(X, Y, peaks, 61)
    h = erode(h, SIZE / N, 62, wavelength=80, octaves=8, gain=.5, strength=.17)
    h = np.maximum(h, 0)
    land = mesh_grid("land", X, Y, h)
    land.data.materials.append(painted_material("rock", ink_albedo(h, 63, SIZE / N) * .62))
    cam = (-6.0, -330.0, 136.0)
    camera(cam, (14.0, 170.0, 34.0), lens=30, far=1200)
    bpy.context.scene.cycles.volume_bounces = 8
    sea = cloud_volume("sea", (-560, 560), (-340, 720), (0, 70), top=opt("top", 42.0), thick=20.0, density=opt("rho", .34), scale=opt("cs", .0075), seed=3.0, fray=2.8)
    near = [mist_sheet("wisp0", (-92, -262, 116), 120, 62, cam, value=.86, density=1.0, seed=70),
            mist_sheet("wisp1", (86, -272, 104), 110, 46, cam, value=.8, density=.9, seed=74),
            # the air itself: each range stands a little paler behind the last
            haze_box("air", (-900, 900), (-340, 1200), (0, 420), opt("air", .0005))]
    world_light(opt("world", .24))
    shine((-.60, -.46, -.62), energy=opt("sun", 6.0), angle=9.0)
    shoot(os.path.join(OUT3D, "myth", "raw", "clouds"), [sea] + near)


# ---- prints in the river mud, lighting one by one -----------------------------------------------------------------------------------
def lit_mud(name, xx, yy, h, a, lit, glow=14.0):
    """Mud whose prints shine: each lights to bare paper, the wet mud round it takes a little, and the light pools. Returns the light as painted."""
    lit = lit + .10 * blur(lit, 16) + .05 * blur(lit, 44)
    mud = mesh_grid(name, xx, yy, h)
    mt = paint(name, a, np.clip(.55 - .4 * sm((.15 - h) / .4), .1, 1))
    nt = mt.node_tree
    b = nt.nodes["Principled BSDF"]
    te = nt.nodes.new("ShaderNodeTexImage")
    te.image = image_from(lit.astype(np.float32), name + "_lit")
    te.interpolation = "Cubic"
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = opt("glow", glow)
    nt.links.new(te.outputs[0], mul.inputs[0])
    nt.links.new(mul.outputs[0], b.inputs["Emission Strength"])
    b.inputs["Emission Color"].default_value = (1, 1, 1, 1)
    mud.data.materials.append(mt)
    return lit


def shot_grow(t=0.0):
    """The prints have not stopped: their toes run on, cross one another, and what they are making is the sign for bird."""
    cell = .012
    h, a, p = (np.load(os.path.join(TEX, f"grow_{k}.npy")) for k in "hap")
    n, m = h.shape
    yy, xx = np.mgrid[:n, :m].astype(np.float32) * cell
    lit = lit_mud("mud", xx, yy, h, a, p, glow=5.0)
    # the light going up off the sign, a little
    r = np.random.default_rng(29)
    ys, xs = np.nonzero(lit > .6)
    picks = r.integers(0, len(ys), opt("flecks", 700))
    lift = r.random(len(picks)) ** 1.8 * 9
    pos = np.c_[xs[picks] * cell + r.normal(0, .02 + .03 * lift), ys[picks] * cell + r.normal(0, .02 + .03 * lift), h[ys[picks], xs[picks]] + .03 + lift]
    fl = points_mesh("flecks", pos, (.010 + .016 * r.random(len(picks)) ** 3))
    fl.data.materials.append(emission("fleck", 1.0, 6.0))
    world_light(opt("world", .5))
    shine((.7, -.5, -.6), energy=opt("sun", .7), angle=10.0)
    cam = (16.4, -1.5, 17.0)
    camera(cam, (16.4, 10.4, 0.0), lens=30, far=1200)
    shoot(os.path.join(OUT3D, "myth", "raw", "grow"), [fl])


def shot_tracks(t=0.0):
    cell = .012
    h, a, p = (np.load(os.path.join(TEX, f"tracks_{k}.npy")) for k in "hap")
    n, m = h.shape
    yy, xx = np.mgrid[:n, :m].astype(np.float32) * cell
    front = opt("front", 21.0) + t * 6.0                 # how far along the line the light has reached
    lit = p * sm((front - yy) / 1.2)
    lit = lit_mud("mud", xx, yy, h, a, lit)

    # beyond the detailed mud: the same bank going on, the river, the valley
    N = 1400
    gy, gx = np.mgrid[:N, :N].astype(np.float32)
    GX, GY = (gx / (N - 1) - .5) * 900 + 40, (gy / (N - 1)) * 900 - 40
    bend = 7.4 + 3.0 * np.sin(GY / 5.1 + .7) * sm((40 - GY) / 40) + .24 * GY - 16 * np.sin(GY / 60 + .3) * sm((GY - 25) / 60)
    nz = lerp(spectral(1024, 3.0, 13), gx / N * 1024, gy / N * 1024)
    G = .70 * np.clip((GX - bend) / 6, -1.2, 1) + .1 * nz
    M = massif(GX, GY, [(-90, 240, 30, 30), (-150, 340, 52, 44), (-30, 470, 60, 56), (110, 230, 26, 26), (170, 340, 56, 44), (70, 500, 66, 66), (-240, 260, 44, 50), (280, 260, 50, 56)], 14)
    M = erode(M, 900 / N, 15, wavelength=30, octaves=7, gain=.5, strength=.28)
    G = G + np.maximum(M, 0) - .14
    land = mesh_grid("valley", GX, GY, G)
    land.data.materials.append(painted_material("valley_rock", ink_albedo(G, 16, 900 / N) * 1.05))
    wx, wy = np.meshgrid(np.linspace(-500, 500, 2), np.linspace(-60, 900, 2))
    water = mesh_grid("river", wx.astype(np.float32), wy.astype(np.float32), np.full((2, 2), -.28, np.float32))
    from ink3d_marsh import mirror_water
    water.data.materials.append(mirror_water(.92, .03))                 # water left as bare paper: the sky over again

    # the light going up: flecks leave each lit print and rise, thinning, into the sky
    r = np.random.default_rng(23)
    ys, xs = np.nonzero(lit > .6)
    # most of it goes up from the prints the light has only just reached, in a thin rising column; the older ones are nearly spent
    weight = np.exp((ys * cell - front) / 5.0)
    picks = r.choice(len(ys), opt("flecks", 1400), p=weight / weight.sum())
    px_, py_ = xs[picks] * cell, ys[picks] * cell
    lift = r.random(len(picks)) ** 1.7 * 34
    pos = np.c_[px_ + r.normal(0, .03 + .014 * lift), py_ + r.normal(0, .03 + .014 * lift), h[ys[picks], xs[picks]] + .03 + lift]
    rad = (.012 + .02 * r.random(len(picks)) ** 3) * (1 - .3 * lift / 34)
    flecks = points_mesh("flecks", pos, rad)
    flecks.data.materials.append(emission("fleck", 1.0, 6.0))

    cam = (12.6, -3.0, 4.7 + opt("camz", .0))
    co = camera(cam, (13.0, 8.0, 1.4), lens=20, far=1200)
    co.data.dof.use_dof = True
    co.data.dof.focus_distance = 7.0
    co.data.dof.aperture_fstop = opt("fstop", .9)
    world_light(opt("world", .85))
    shine((.7, -.5, -.6), energy=opt("sun", .9), angle=12.0)            # low, across the mud, so that a dent shows even unlit
    mists = [mist_sheet("mud0", (10, 46, 3.4), 110, 14, cam, value=.8, density=.9, seed=30, top=.3),
             mist_sheet("mud1", (14, 110, 12), 340, 40, cam, value=.8, density=1.0, seed=34, top=.3)]
    shoot(os.path.join(OUT3D, "myth", "raw", "tracks"), mists + [flecks])


# ---- the signs strike the cloud, the last tears it ------------------------------------------------------------------------------------
def oracle_tubes(strokes, center, size, name, rad):
    """A pictograph as thick tapered strokes standing in the air, facing the valley (points in a unit square, y down)."""
    segs = []
    cx, cy, cz = center
    for st in strokes:
        if len(st) == 1:
            st = [st[0], (st[0][0] + .012, st[0][1] + .012)]
        L = len(st) - 1
        for i in range(L):
            (x0, y0), (x1, y1) = st[i], st[i + 1]
            f0, f1 = i / max(L, 1), (i + 1) / max(L, 1)
            w0, w1 = (1 - .55 * abs(2 * f0 - 1) ** 2), (1 - .55 * abs(2 * f1 - 1) ** 2)
            segs.append(((cx + (x0 - .5) * size, cy, cz + (.5 - y0) * size), (cx + (x1 - .5) * size, cy, cz + (.5 - y1) * size), rad * w0, rad * w1))
    return tube_mesh(segs, name, sides=9)


# the rip's course over the valley. Near the lens it lurches about; further off it runs straight out from the lens, or the cloud's own thickness would shut it from view
TEAR_Y = np.array([-120, -60, -10, 30, 75, 115, 160, 210, 260, 330, 400, 470, 560, 650, 760.])
TEAR_X = np.array([118, 62, 104, 34, 70, -20, -10, -93, -106, -124, -142, -160, -183, -206, -235.])


def tear_mask(n, m, seed, x0, y0, width, depth):
    """Where the sky has gone: a jagged rip running from over the lens away to the far ridge, like a crack across glass, wide overhead and thin in the distance. In the image's rows and columns."""
    r = np.random.default_rng(seed)
    gy, gx = np.mgrid[:n, :m].astype(np.float32)
    X, Y = x0 + gx / (m - 1) * width, y0 + gy / (n - 1) * depth
    ys, xs = TEAR_Y, TEAR_X
    mid = np.interp(Y, ys, xs)
    wid = np.interp(Y, [-120, -40, 40, 120, 220, 360, 500, 650, 760], [75, 66, 54, 40, 30, 24, 18, 12, 3])
    fine = 7 * lerp(spectral(1024, 2.0, seed + 1), gx * 1024 / m, gy * 1024 / n)
    tear = sm((wid - np.abs(X - mid - fine)) / 2.5)
    tear = tear * sm((Y + 200) / 40) * sm((770 - Y) / 20)
    # side cracks
    for k in range(9):
        y1 = r.uniform(-20, 520)
        x1 = np.interp(y1, ys, xs)
        side = (-1) ** k
        L = r.uniform(50, 150) * (1 - .001 * y1)
        qx, qy = x1 + side * np.linspace(0, L, 6), y1 + np.r_[0, np.cumsum(r.normal(0, L * .13, 5))]
        ww = np.linspace(6, 1.2, 6) * (1 - .0012 * y1)
        for i in range(5):
            ax, ay, bx, by = qx[i], qy[i], qx[i + 1], qy[i + 1]
            t = np.clip(((X - ax) * (bx - ax) + (Y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2 + 1e-9), 0, 1)
            dd = np.hypot(X - ax - t * (bx - ax), Y - ay - t * (by - ay))
            tear = np.maximum(tear, sm((ww[i] - dd) / 1.5))
    return tear.astype(np.float32)


def light_column(name, top, direction, cam, width, strength, seed):
    """A shaft of light from top down along direction to the ground: a card turned to the lens, wider at its foot, fading as it falls. Through it the world goes toward bare paper."""
    n, m = 256, 96
    gy, gx = np.mgrid[:n, :m].astype(np.float32)
    v, u = gy / (n - 1), gx / (m - 1) * 2 - 1
    streak = lerp(spectral(1024, 2.6, seed), gx * 1.6, gy * .2)
    a = sm((1 - np.abs(u)) / .85) * (1 - v) ** 1.25 * np.clip(.85 + .35 * streak, .35, 1.2) * strength
    d = np.array(direction, float)
    d = d / np.linalg.norm(d)
    L = top[2] / -d[2]
    top_ = np.array(top, float)
    rv, ru = np.mgrid[:17, :3].astype(np.float32)
    rv, ru = rv / 16, ru - 1
    wd = width * (1 + .7 * rv)
    # a ribbon, turned to the lens row by row: one flat card would go edge-on to it somewhere along its length and show as a scratch
    C = top_[None, None, :] + d[None, None, :] * (L * rv)[..., None]
    s = np.cross(d, np.array(cam, float) - C)
    s = s / np.linalg.norm(s, axis=-1, keepdims=True)
    P = [C[..., i] + s[..., i] * ru * wd for i in range(3)]
    o = mesh_grid(name, P[0].astype(np.float32), P[1].astype(np.float32), P[2].astype(np.float32))
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    nt = mt.node_tree
    nt.nodes.clear()
    ti, tr, em, mx, out = (nt.nodes.new(k) for k in ("ShaderNodeTexImage", "ShaderNodeBsdfTransparent", "ShaderNodeEmission", "ShaderNodeMixShader", "ShaderNodeOutputMaterial"))
    ti.image = image_from(np.clip(a, 0, 1).astype(np.float32), name + "_a")
    nt.links.new(ti.outputs[0], mx.inputs[0])
    nt.links.new(tr.outputs[0], mx.inputs[1])
    nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs["Surface"])
    o.data.materials.append(mt)
    o.visible_shadow = False
    return o


def shot_rift(t=0.0):
    from ink_oracle import ORACLE
    SIZE, N = 1100.0, 1400
    y, x = np.mgrid[:N, :N].astype(np.float32)
    X, Y = (x / (N - 1) - .5) * SIZE, (y / (N - 1) - .5) * SIZE + 260
    h = massif(X, Y, [(-240, 170, 74, 66), (-110, 330, 62, 60), (250, 150, 80, 76), (120, 360, 66, 60), (-330, 430, 110, 100), (350, 440, 120, 110), (0, 620, 96, 120), (-50, 60, 16, 22), (70, 40, 20, 24)], 51)
    h = erode(h, SIZE / N, 52, wavelength=50, octaves=7, gain=.5, strength=.2)
    cx = 26 * np.sin(Y / 90 + 1.0)
    h = np.maximum(h * sm((np.abs(X - cx) - 12) / 40) ** .7, 0)
    land = mesh_grid("land", X, Y, h + .4)
    land.data.materials.append(painted_material("rock", ink_albedo(h, 53, SIZE / N)))
    # far off the valley is shut by mountains that go up into the cloud
    X2, Y2, H2 = far_mountains([(-1000, 1560, 210, 250), (-640, 1600, 240, 270), (-300, 1520, 190, 230), (40, 1590, 230, 260), (380, 1530, 200, 240), (720, 1600, 240, 270), (1060, 1560, 210, 250)], 2800, 900, 61, y_center=1400, strength=.3)
    far_range = mesh_grid("range", X2, Y2, H2 - 3)
    far_range.data.materials.append(emission("rangew", opt("rangev", .26)))
    wx, wy = np.meshgrid(np.linspace(-700, 700, 2), np.linspace(-300, 1000, 2))
    water = mesh_grid("river", wx.astype(np.float32), wy.astype(np.float32), np.full((2, 2), .25, np.float32))
    water.data.materials.append(water_material(rough=.02, normal_scale=20, normal_strength=.015))

    cam = (0.0, -150.0, 5.0)
    co = camera(cam, (6.0, 200.0, 138.0), lens=opt("lens", 20), far=1500)

    # the cloud: a real volume, its middle gone along the tear
    wx0, wy0, ww, wh = -1000.0, -300.0, 2000.0, 1500.0
    tear = tear_mask(1500, 2000, 81, wx0, wy0, ww, wh)
    cut_img = image_from(1 - tear, "tear")
    # the signs have gone into the cloud one after another, each lighting only the patch it struck, in no order. The first are fading; the last has torn the sky and stands black against it
    hits = [("山", (-300, 470, 196), 66, .3), ("木", (120, 520, 196), 76, .45), ("水", (215, 250, 186), 72, .52), ("日", (-225, 130, 186), 58, .6)]
    gy, gx = np.mgrid[:750, :1000].astype(np.float32)
    PX, PY = wx0 + gx / 999 * ww, wy0 + gy / 749 * wh
    patch = np.zeros_like(PX)
    for ch, at, size, lamp in hits:
        patch = np.maximum(patch, lamp * np.exp(-((PX - at[0]) ** 2 + (PY - at[1]) ** 2) / (2 * (size * opt("spread", .85)) ** 2)))
    patch = patch * np.clip(1 + 1.15 * lerp(spectral(1024, 2.1, 83), gx * 1024 / 1000, gy * 1024 / 750), .05, 1.5)
    # and the torn edge catches what is behind it
    small = tear[::2, ::2]
    rim = np.clip(blur(small, 4) * 1.7, 0, 1) * (1 - small)
    lit = image_from(np.clip(np.maximum(patch, opt("rim", .45) * rim), 0, 1.5).astype(np.float32), "cloud_lit")
    bpy.context.scene.cycles.volume_bounces = 6
    ceil = cloud_volume("ceiling", (wx0, wx0 + ww), (wy0, 2300.0), (186, 300), top=opt("top", 244.0), thick=22.0, density=opt("rho", .05), scale=.0052, seed=9.0,
                        cut=(cut_img, -wx0 / ww, -wy0 / wh, 1 / ww, 1 / wh), bottom=opt("bottom", 190.0), fray=2.4, albedo=opt("albedo", .5), glow=opt("cglow", .08),
                        glow_map=(lit, -wx0 / ww, -wy0 / wh, 1 / ww, 1 / wh, opt("hit", .5)))
    air = haze_box("air", (-1000, 1000), (-300, 1200), (0, 180), opt("air", .0009))     # clear of the cloud's box: rays lose track of a volume when two faces coincide
    # what is seen through the tear is nothing but light. It lights nothing itself: the sun does that
    px = mesh_grid("heaven", np.array([[-1400., 1400.], [-1400., 1400.]], np.float32), np.array([[-400., -400.], [1300., 1300.]], np.float32), np.full((2, 2), 330., np.float32))
    px.data.materials.append(emission("heaven", 1.0, opt("heaven", 4.0)))
    px.visible_shadow = px.visible_diffuse = px.visible_volume_scatter = False
    # and the sun comes only through the tear: a plate over the cloud, shifted along the light so that its hole falls on the rip. All else is left to a fill that nothing stops
    sun_d = (.10, -.34, -.93)
    plate_z, mid_z = 305.0, 217.0
    dt = (plate_z - mid_z) / -sun_d[2]
    sx_, sy_ = -sun_d[0] * dt, -sun_d[1] * dt
    plate = mesh_grid("plate", np.array([[wx0 + sx_, wx0 + ww + sx_]] * 2, np.float32), np.array([[wy0 + sy_] * 2, [wy0 + wh + sy_] * 2], np.float32), np.full((2, 2), plate_z, np.float32))
    pm = bpy.data.materials.new("plate")
    pm.use_nodes = True
    nt = pm.node_tree
    nt.nodes.clear()
    ti_, tr_, df_, mx_, ou_ = (nt.nodes.new(k_) for k_ in ("ShaderNodeTexImage", "ShaderNodeBsdfTransparent", "ShaderNodeBsdfDiffuse", "ShaderNodeMixShader", "ShaderNodeOutputMaterial"))
    # what comes through is broken by the rags of cloud in the rip: columns of light, not one wall of it
    py_, px_ = np.mgrid[:1500, :2000].astype(np.float32)
    bars = sm((lerp(spectral(1024, 1.7, 85), px_ * .13, py_ * .95) + .15) / .3)
    ti_.image = image_from(tear * bars, "tear_plate")
    df_.inputs["Color"].default_value = (0, 0, 0, 1)
    nt.links.new(ti_.outputs[0], mx_.inputs[0])
    nt.links.new(df_.outputs[0], mx_.inputs[1])
    nt.links.new(tr_.outputs[0], mx_.inputs[2])
    nt.links.new(mx_.outputs[0], ou_.inputs["Surface"])
    plate.data.materials.append(pm)
    plate.visible_camera = plate.visible_diffuse = plate.visible_glossy = plate.visible_transmission = plate.visible_volume_scatter = False
    world_light(opt("world", .05))
    shine(sun_d, energy=opt("sun", 9.0), angle=1.0)
    fill = bpy.data.lights.new("fill", "SUN")
    fill.energy = opt("fill", .7)
    fill.use_shadow = False
    if hasattr(fill, "cycles") and hasattr(fill.cycles, "cast_shadow"):
        fill.cycles.cast_shadow = False
    bpy.context.collection.objects.link(bpy.data.objects.new("fill", fill))

    for i, (ch, at, size, lamp) in enumerate(hits):
        tube = oracle_tubes(ORACLE[ch], at, size, f"sign{i}", size * .036)
        tube.data.materials.append(emission(f"sign{i}", 1.0, 30.0 * lamp))
        tube.visible_shadow = False
    last = oracle_tubes(ORACLE["鸟"], (25, 33, 175), 40.0, "last", 1.9)
    last.data.materials.append(emission("last", .004, 1.0))
    # the columns of light: the haze alone shows them too faintly to read, so they are painted, as paper left bare
    rb = np.random.default_rng(87)
    beams = []
    for k, (by, w0, a0, fan) in enumerate(((35, 20, .5, -.42), (70, 13, .34, .38), (105, 24, .62, -.2), (140, 11, .3, .55), (175, 17, .46, .2), (215, 12, .36, -.46), (255, 16, .42, .42), (300, 9, .26, -.26), (350, 13, .34, .3), (410, 8, .22, -.14), (480, 8, .2, .16))):
        beams.append(light_column(f"beam{k}", (np.interp(by, TEAR_Y, TEAR_X) + rb.normal(0, 6), by, 192.0), (sun_d[0] + fan, sun_d[1], sun_d[2]), cam, w0 * 1.5, a0 * opt("beam", .7), 90 + k))
    shoot(os.path.join(OUT3D, "myth", "raw", "rift"), [ceil, air, px, plate] + beams)


# ---- millet ---------------------------------------------------------------------------------------------------------------------------
def grain_shape():
    """A husked millet grain, a little more than a sphere: egg-shaped, flattened on one side where the germ is. Unit size."""
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1)
    o = bpy.context.active_object
    me = o.data
    me.calc_loop_triangles()
    V = np.array([v.co[:] for v in me.vertices], float)
    F = np.array([t.vertices[:] for t in me.loop_triangles])
    bpy.data.objects.remove(o)
    V[:, 2] = V[:, 2] * 1.12 + .10 * V[:, 2] ** 2 - .05
    V[:, 0] *= .94
    germ = np.exp(-((V[:, 0]) ** 2 + (V[:, 2] + .55) ** 2) / .07) * (V[:, 1] > 0)
    V[:, 1] = V[:, 1] * .92 - .22 * germ
    return V, F


def random_quats(r, k):
    q = r.normal(size=(k, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def heap_height(x, y, r0, top):
    d = np.hypot(x, y * 1.15) / r0
    return top * np.exp(-d ** 1.7)


def millet_head(r, base, direction, length, radius, droop, lobes=170, per_lobe=26):
    """A foxtail-millet head: a drooping stem carrying knots of grain, each knot with its bristles. Returns grain centres, bristle segments, and the stem's line."""
    d = np.array(direction, float)
    d /= np.linalg.norm(d)
    side = np.cross(d, (0, 1, 0))
    side /= np.linalg.norm(side) + 1e-9
    up = np.cross(side, d)
    fall = np.array((0, 0, -1.0))
    s = (r.random(lobes) ** .92)
    a = r.uniform(0, 2 * math.pi, lobes)
    prof = radius * np.sin(np.clip(s, .02, 1) ** .62 * math.pi) ** .6
    axis = base + d[None] * (s * length)[:, None] + fall[None] * (droop * s ** 2)[:, None]
    out = side[None] * np.cos(a)[:, None] + up[None] * np.sin(a)[:, None]
    knots = axis + out * prof[:, None]
    lobe_r = (radius * .3) * (.7 + .5 * np.sin(s * math.pi))
    grains = knots[:, None, :] + r.normal(0, 1, (lobes, per_lobe, 3)) * lobe_r[:, None, None]
    bristles = []
    for i in range(0, lobes, 2):
        tip = knots[i] + (out[i] + r.normal(0, .5, 3)) * r.uniform(radius * .5, radius * 1.0)
        bristles.append((knots[i], tip, .0022, .0004))
    line = [base + d * q * length + fall * droop * max(q, 0) ** 2 for q in np.linspace(-.9, 1, 26)]
    return grains.reshape(-1, 3), bristles, line


def millet_material(name, value, grain, rough=.42):
    """Every grain a little different, and mottled the way a husk is."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc, nz, rp = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeValToRGB")
    nz.inputs["Scale"].default_value = 1 / (grain * 1.1)
    nz.inputs["Detail"].default_value = 3
    rp.color_ramp.elements[0].position, rp.color_ramp.elements[1].position = .3, .7
    rp.color_ramp.elements[0].color = (value * .55, value * .55, value * .55, 1)
    rp.color_ramp.elements[1].color = (value, value, value, 1)
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    nt.links.new(nz.outputs[0], rp.inputs[0])
    nt.links.new(rp.outputs[0], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    return m


def shot_grain(t=0.0):
    sc = bpy.context.scene
    r = np.random.default_rng(101)
    V, F = grain_shape()
    grain = .009
    cam = np.array((-.55, -2.1, .62))
    # the ground and a low heap where the rope lands, flecked to the size of a grain so that what is too far to model still reads as grain
    N = 600
    gy, gx = np.mgrid[:N, :N].astype(np.float32)
    GX, GY = (gx / (N - 1) - .5) * 60 + cam[0], (gy / (N - 1)) * 60 - 16
    hx0, hy0, top = .05, 3.2, opt("heap", .62)
    G = heap_height(GX - hx0, GY - hy0, 2.3, top) - .05
    ground = mesh_grid("ground", GX, GY, G)
    fleck = np.tile(spectral(1024, .5, 4), (4, 4))
    ground.data.materials.append(painted_material("earth", np.clip(.64 + .17 * fleck / fleck.std(), .12, .92).astype(np.float32)))
    grain_mat = millet_material("millet", .86, grain)

    # a floor of grain, one by one, thick under the lens and thinning away over the flecks
    k = opt("floor", 70000)
    fy = .2 + r.random(k) ** 1.6 * 7
    fx = cam[0] + r.uniform(-1, 1, k) * (.9 + (fy - cam[1]) * .46)
    fz = heap_height(fx - hx0, fy - hy0, 2.3, top) + r.uniform(-.03, .012, k)
    # and the heap
    kh = opt("heapn", 50000)
    th = r.uniform(0, 2 * math.pi, kh)
    ra = r.random(kh) ** .55 * 3.0
    hx, hy = hx0 + ra * np.cos(th), hy0 + ra * np.sin(th) / 1.15
    hz = heap_height(hx - hx0, hy - hy0, 2.3, top) + r.uniform(-.03, .012, kh)
    pos = np.r_[np.c_[fx, fy, fz], np.c_[hx, hy, hz]]
    s = grain * r.uniform(.84, 1.12, len(pos))
    floor = instances_mesh("floor", V, F, pos, np.c_[s, s, s], random_quats(r, len(pos)))
    floor.data.materials.append(grain_mat)

    # ropes of grain coming down out of the light, narrow above and loosening as they fall, moving together so that a shutter draws them out
    ropes = [(hx0 + .02, hy0, 6000, .05, .02), (1.4, 5.4, 1500, .025, .01), (-1.25, 7.0, 2400, .035, .012), (2.9, 9.5, 1800, .035, .012), (-3.3, 12.0, 1400, .04, .012)]
    ax, ay, az = [], [], []
    rope_of = []
    for rx, ry, kk, s0, s1 in ropes:
        zmax = .7 + .3 * (ry - cam[1])
        zz = r.random(kk) ** 1.1 * zmax + .03
        wide_k = 1 + 2 * (r.random(kk) < .1)
        nx_, ny_ = r.normal(0, 1, kk), r.normal(0, 1, kk)
        spread = (s0 + s1 * (zmax - zz)) * wide_k
        ax.append(rx + nx_ * spread)
        ay.append(ry + ny_ * spread)
        az.append(zz)
        rope_of.append(np.c_[np.full(kk, rx), np.full(kk, ry), nx_ * wide_k, ny_ * wide_k, np.full(kk, s0), np.full(kk, s1), np.full(kk, zmax)])
    # what the rope throws out where it lands
    ks = opt("wide", 1500)
    ax.append(hx0 + r.normal(0, .6, ks))
    ay.append(hy0 - .2 + r.normal(0, .5, ks))
    hop_h = r.random(ks) ** 2.2 * .5
    az.append(top + hop_h)
    rope_of = np.concatenate(rope_of)
    ax, ay, az = np.concatenate(ax), np.concatenate(ay), np.concatenate(az)
    s = grain * r.uniform(.84, 1.12, len(ax))
    air = instances_mesh("air", V, F, np.c_[ax, ay, az], np.c_[s, s, s], random_quats(r, len(ax)))
    air.data.materials.append(grain_mat)
    # a few close enough to brush the lens: big, soft, gone
    near_p = np.array([(-.76, -1.3, .8), (-.36, -1.05, .5), (-.88, -.95, .36), (-.2, -1.2, .84)])
    frames_ = opt("frames", "")
    if frames_:     # falling, there are more of them, for they pass in a moment
        near_p = np.c_[r.uniform(-1.25, .15, 14), r.uniform(-1.45, -.7, 14), r.uniform(-.2, 1.6, 14)]
    s = np.full(len(near_p), grain * 1.6) * r.uniform(.9, 1.1, len(near_p))
    nr = instances_mesh("brush", V, F, near_p, np.c_[s, s, s], random_quats(r, len(near_p)))
    nr.data.materials.append(grain_mat)

    # heads of millet in front, on their stems, heavy enough to bow: this is what says food
    heads = [((-1.85, .62, 1.18), (1.0, 0, -.05), 1.0, .125, .36, 200, 26), ((.95, 1.4, 1.32), (-.85, .05, -.2), .9, .115, .36, 170, 26)]
    ear_mat, stem_mat = millet_material("ear", .55, grain * .62, .5), principled_grey("stem", .16, .6)
    nodding = []
    for i, (base, d, L, rad, dr, lb, pl) in enumerate(heads):
        base = np.array(base, float)
        g_, br_, line = millet_head(r, base, d, L, rad, dr, lb, pl)
        seg = [(a_ - base, b_ - base, w0_, w1_) for a_, b_, w0_, w1_ in br_]
        for a_, b_ in zip(line[:-1], line[1:]):
            seg.append((a_ - base, b_ - base, .012, .011))
        s = grain * .62 * r.uniform(.85, 1.1, len(g_))
        ear = instances_mesh(f"ears{i}", V, F, g_ - base, np.c_[s, s, s], random_quats(r, len(g_)))
        ear.data.materials.append(ear_mat)
        stems = tube_mesh(seg, f"stems{i}", sides=4)
        stems.data.materials.append(stem_mat)
        for o_ in (ear, stems):
            o_.location = tuple(base)
            nodding.append((o_, i))

    # motion: what is in the air falls during the shutter
    sc.frame_start, sc.frame_end = 1, 2
    sc.frame_set(1)
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = opt("shutter", .9)
    for o_, fall in ((air, opt("fallv", .05)), (nr, .12)):
        o_.location = (0, 0, 0)
        o_.keyframe_insert("location", frame=1)
        o_.location = (0, 0, -fall)
        o_.keyframe_insert("location", frame=2)
        o_.location = (0, 0, 0)

    co = camera(tuple(cam), (-.55, 6.0, .76), lens=opt("lens", 40), far=1200)
    co.data.dof.use_dof = True
    co.data.dof.focus_distance = opt("focus", 2.9)
    co.data.dof.aperture_fstop = opt("fstop", 6.0)

    # behind everything the light itself: white, falling off only a little at the sides, so the grain stands against it
    bn, bm = 256, 512
    by_, bx_ = np.mgrid[:bn, :bm].astype(np.float32)
    bu, bv = bx_ / (bm - 1) * 2 - 1, by_ / (bn - 1)
    glow = np.clip(.84 + .16 * np.exp(-((bu + .02) / .55) ** 2) * (.45 + .55 * bv) + .012 * lerp(spectral(1024, 3.2, 21), bx_ * 2, by_ * 4), 0, 1)
    xs_ = np.array([[-90., 90.], [-90., 90.]], np.float32)
    back = mesh_grid("shaft", xs_, np.full((2, 2), 70., np.float32), np.array([[-6., -6.], [40., 40.]], np.float32))
    back.data.materials.append(image_emission("shaft", glow.astype(np.float32)))
    back.visible_shadow = False
    world_light(opt("world", .62))
    shine((.5, .6, -.62), energy=opt("sun", 1.1), angle=8.0)
    if not frames_:
        shoot(os.path.join(OUT3D, "myth", "raw", "grain"), [back])
        return

    # the shot itself, frame by frame: --frames=A-B. One build; what is in the air falls and each frame only moves it. It comes in a little as it goes, and the ears nod under what is landing on them
    os.makedirs(os.path.join(OUT3D, "myth", "grain"), exist_ok=True)
    fallv = opt("fallv", .05)
    air_me, near_me = air.data, nr.data
    nv_ = len(V)
    air_p = np.empty(len(air_me.vertices) * 3, np.float32)
    air_me.vertices.foreach_get("co", air_p)
    air_p = air_p.reshape(-1, nv_, 3)
    air_c = air_p - np.c_[ax, ay, az][:, None, :]
    near_v = np.empty(len(near_me.vertices) * 3, np.float32)
    near_me.vertices.foreach_get("co", near_v)
    near_v = near_v.reshape(-1, nv_, 3)
    near_c = near_v - near_p[:, None, :]
    n_rope = len(rope_of)
    rx_, ry_, nx_, ny_, s0_, s1_, zm_ = rope_of.T
    zr = az[:n_rope].copy()
    hop_x, hop_y, phase = ax[n_rope:], ay[n_rope:], r.random(ks)
    hop_t = np.maximum(6.0, 26 * np.sqrt(hop_h / .5))     # frames a hop takes
    hop_z = heap_height(hop_x - hx0, hop_y - hy0, 2.3, top)
    a_, b_ = [int(v) for v in frames_.split("-")]
    view = np.array((-.55, 6.0, .76)) - cam
    view /= np.linalg.norm(view)
    FRAMES = 195
    for n in range(a_, b_ + 1):
        u = n / (FRAMES - 1)
        z_now = (zr - fallv * n - .03) % zm_ + .03
        spread = s0_ + s1_ * (zm_ - z_now)
        pos = np.empty((len(ax), 3), np.float32)
        pos[:n_rope, 0], pos[:n_rope, 1], pos[:n_rope, 2] = rx_ + nx_ * spread, ry_ + ny_ * spread, z_now
        q = (n / hop_t + phase) % 1
        pos[n_rope:, 0], pos[n_rope:, 1], pos[n_rope:, 2] = hop_x, hop_y, hop_z + hop_h * 4 * q * (1 - q)
        air_me.vertices.foreach_set("co", (air_c + pos[:, None, :]).astype(np.float32).ravel())
        air_me.update()
        pn = near_p.copy()
        pn[:, 2] = (near_p[:, 2] - .12 * n + .2) % 1.8 - .2
        near_me.vertices.foreach_set("co", (near_c + pn[:, None, :]).astype(np.float32).ravel())
        near_me.update()
        for o_, i in nodding:
            ph = n / 30 * 2.1 + i * 1.7
            o_.rotation_euler = (math.radians(1.3) * math.sin(ph), math.radians(.5) * math.sin(ph * 1.7), math.radians(.8) * math.sin(ph * .6))
        c_ = cam + view * .42 * u
        co = camera(tuple(c_), tuple(c_ + view * 8.1), lens=opt("lens", 40), far=1200)
        co.data.dof.use_dof = True
        co.data.dof.focus_distance = opt("focus", 2.9) - .42 * u
        co.data.dof.aperture_fstop = opt("fstop", 6.0)
        sc.frame_set(1)
        name = os.path.join(OUT3D, "myth", "grain", f"grain_{n:03d}")
        render(name, depth=False)
        back.hide_render = True
        render(name, depth=True)
        back.hide_render = False
        print("frame", n, flush=True)


def principled_grey(name, value, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (value, value, value, 1)
    b.inputs["Roughness"].default_value = rough
    return m


SHOTS = {"clouds": shot_clouds, "tracks": shot_tracks, "grow": shot_grow, "rift": shot_rift, "grain": shot_grain}

if __name__ == "__main__":
    from ink3d_marsh import MARSH
    SHOTS.update(MARSH)
    shot = opt("shot", "marsh")
    reset(samples=opt("samples", 64), pct=opt("pct", 100))
    SHOTS[shot](opt("t", 0.0))
