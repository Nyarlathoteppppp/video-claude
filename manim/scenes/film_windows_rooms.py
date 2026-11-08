"""九 千窗: the rooms behind the windows. Three of them seen from across the street, at the same minute, through a long lens and their own window frame, nobody in them: the room and what is on its screen say who it is. And small pictures of other rooms, for the windows of the city.

    Blender -b --factory-startup -P film_windows_rooms.py -- C2A [--pct=100] [--samples=256]   # plates/C2A_room.npy and C2A_words.npy, under ink/film_out/film/windows/
    Blender -b --factory-startup -P film_windows_rooms.py -- pics [--samples=48] [--figures=1]  # pics/*.png, 480 by 300, as wide as the building's windows; nobody in them unless --figures=1

The room and what is on its screen are two renders, added in film_windows_ui.py: the denoiser eats words, and nothing on a screen needs it. x runs to the right as you look in, y into the room from the window's wall, z up, in metres.
"""
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from mathutils import Matrix, Vector

from ink3d import ROOT, aim, mesh_grid, reset

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


OUT = os.path.join(ROOT, "ink", "film_out", "film", "windows")
TEX = os.path.join(OUT, "tex")
PLATES = os.path.join(OUT, "plates")
PICS = os.path.join(OUT, "pics")

WIN = (-1.12, 1.12, .95, 2.35)      # the window: left, right, sill, head. 2.24 by 1.4, the same as every window of the building
ROOM_X, ROOM_Y, ROOM_Z = 1.75, 3.5, 2.7
ZT = .74                            # a table's top
DISP_W, DISP_H, DISP_T, DISP_UP = .623, .362, .0185, .116       # a display, as in film_room3d.py
SCREEN_W, SCREEN_H = .5965, .3356
LEMON = (1.0, .80, .58)             # a warm bulb, as a colour of light


# ---------------------------------------------------------------- materials

def surface(name, colour, rough=.5, metal=0.0, spec=.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*colour, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Specular IOR Level"].default_value = spec
    return m


def noisy(m, scale, strength, distance=.002, detail=6):
    """A fine bump on a surface, in object space."""
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc, n, bp = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeBump")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    bp.inputs["Strength"].default_value = strength
    bp.inputs["Distance"].default_value = distance
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
    nt.links.new(n.outputs[0], bp.inputs["Height"])
    nt.links.new(bp.outputs[0], b.inputs["Normal"])
    return m


def blotched(name, colour, amount=.1, scale=2.2, rough=.85):
    """A painted wall that has been lived against: its colour wanders a little, and marks are darker."""
    m = surface(name, colour, rough, 0, .25)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc, n, ramp = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeValToRGB")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = 7
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = .3, .7
    ramp.color_ramp.elements[0].color = tuple(c * (1 - amount) for c in colour) + (1,)
    ramp.color_ramp.elements[1].color = tuple(min(c * (1 + amount * .6), 1) for c in colour) + (1,)
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
    nt.links.new(n.outputs[0], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], b.inputs["Base Color"])
    return noisy(m, 130, .08, .003)


def printed(name, picture, rough=.6, spec=.3):
    m = surface(name, (1, 1, 1), rough, 0, spec)
    nt = m.node_tree
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(os.path.join(TEX, picture), check_existing=True)
    t.interpolation = "Cubic"
    nt.links.new(nt.nodes.new("ShaderNodeUVMap").outputs[0], t.inputs[0])
    nt.links.new(t.outputs[0], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return m


def screen_material(picture, strength, spill):
    """What the screen shows goes in the second render; in the first it only lights the room, a little more than you would see on it."""
    m = bpy.data.materials.new("screen")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    t, e, o = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    t.image = bpy.data.images.load(os.path.join(TEX, picture), check_existing=True)
    t.interpolation = "Cubic"
    nt.links.new(nt.nodes.new("ShaderNodeUVMap").outputs[0], t.inputs[0])
    nt.links.new(t.outputs[0], e.inputs[0])
    path, off = nt.nodes.new("ShaderNodeLightPath"), nt.nodes.new("ShaderNodeMath")
    off.operation = "SUBTRACT"
    off.inputs[0].default_value = 1
    nt.links.new(path.outputs["Is Camera Ray"], off.inputs[1])
    gain = nt.nodes.new("ShaderNodeMath")
    gain.operation = "MULTIPLY"
    gain.inputs[1].default_value = strength * spill
    nt.links.new(off.outputs[0], gain.inputs[0])
    nt.links.new(gain.outputs[0], e.inputs[1])
    nt.links.new(e.outputs[0], o.inputs[0])
    m["words"] = strength
    return m


def glass_material():
    """A pane with nothing on it but what it reflects."""
    m = bpy.data.materials.new("pane")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tr, gl, fr, mix, o = nt.nodes.new("ShaderNodeBsdfTransparent"), nt.nodes.new("ShaderNodeBsdfGlossy"), nt.nodes.new("ShaderNodeFresnel"), nt.nodes.new("ShaderNodeMixShader"), nt.nodes.new("ShaderNodeOutputMaterial")
    tr.inputs[0].default_value = (.93, .94, .92, 1)
    gl.inputs["Roughness"].default_value = .015
    fr.inputs[0].default_value = 1.52
    nt.links.new(fr.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(gl.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], o.inputs[0])
    return m


# ---------------------------------------------------------------- shapes

def link(o):
    bpy.context.collection.objects.link(o)
    return o


def mesh_object(name, verts, faces, mat, uv=None, matrix=None, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], faces)
    me.update(calc_edges=True)
    if smooth:
        me.shade_smooth()
    if uv is not None:
        layer = me.uv_layers.new(name="UV")
        for poly in me.polygons:
            for li in poly.loop_indices:
                layer.data[li].uv = uv[me.loops[li].vertex_index]
    me.materials.append(mat)
    o = link(bpy.data.objects.new(name, me))
    if matrix is not None:
        o.matrix_world = matrix
    return o


def bevelled(o, width, segments=2, angle=35.0):
    m = o.modifiers.new("bevel", "BEVEL")
    m.width, m.segments, m.limit_method = width, segments, "ANGLE"
    m.angle_limit = math.radians(angle)
    return o


def box(name, x0, x1, y0, y1, z0, z1, mat, bevel=0.0, matrix=None):
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    o = mesh_object(name, v, f, mat, matrix=matrix)
    return bevelled(o, bevel) if bevel else o


def outline(x0, x1, y0, y1, r, n=8):
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def slab(name, x0, x1, y0, y1, z0, z1, r, mat, matrix=None, n=8):
    p = outline(x0, x1, y0, y1, r, n)
    k = len(p)
    v = [(x, y, z0) for x, y in p] + [(x, y, z1) for x, y in p]
    f = [tuple(range(k - 1, -1, -1)), tuple(range(k, 2 * k))] + [(i, (i + 1) % k, k + (i + 1) % k, k + i) for i in range(k)]
    return mesh_object(name, v, f, mat, matrix=matrix)


def lathe(name, profile, x, y, z, mat, n=48):
    """Turned about the vertical: profile is (radius, height)."""
    p = np.array(profile, np.float32)
    a = np.linspace(0, 2 * math.pi, n + 1)
    r = p[:, 0][:, None]
    o = mesh_grid(name, x + r * np.cos(a), y + r * np.sin(a), z + np.broadcast_to(p[:, 1][:, None], (len(p), n + 1)))
    o.data.materials.append(mat)
    return o


def tube(name, points, radius, mat, res=3):
    c = bpy.data.curves.new(name, "CURVE")
    c.dimensions = "3D"
    c.bevel_depth, c.bevel_resolution = radius, res
    c.use_fill_caps = True
    s = c.splines.new("NURBS")
    s.points.add(len(points) - 1)
    for i, p in enumerate(points):
        s.points[i].co = (*p, 1)
    s.use_endpoint_u = True
    s.order_u = min(4, len(points))
    s.resolution_u = 10
    c.materials.append(mat)
    return link(bpy.data.objects.new(name, c))


def ellipsoid(name, radii, loc, mat, n=32):
    u, v = np.meshgrid(np.linspace(0, 2 * math.pi, 2 * n + 1), np.linspace(-math.pi / 2, math.pi / 2, n + 1))
    o = mesh_grid(name, radii[0] * np.cos(v) * np.cos(u), radii[1] * np.cos(v) * np.sin(u), radii[2] * np.sin(v))
    o.data.materials.append(mat)
    o.location = loc
    return o


def loft(name, rows, mat, loc, power=.72):
    """A shape swept up through ellipses: (height, half width, half depth). Under 1, power makes them squarer, as shoulders are."""
    a = np.linspace(0, 2 * math.pi, 73)
    rows = np.array(rows, np.float32)
    zs = np.linspace(rows[0, 0], rows[-1, 0], 48)
    k = np.array([.1, .2, .4, .2, .1])
    hw = np.convolve(np.pad(np.interp(zs, rows[:, 0], rows[:, 1]), 2, "edge"), k, "valid")
    hd = np.convolve(np.pad(np.interp(zs, rows[:, 0], rows[:, 2]), 2, "edge"), k, "valid")
    c, s = np.cos(a), np.sin(a)
    x = hw[:, None] * (np.sign(c) * np.abs(c) ** power)
    y = hd[:, None] * (np.sign(s) * np.abs(s) ** power)
    o = mesh_grid(name, x, y, np.broadcast_to(zs[:, None], x.shape).copy())
    o.data.materials.append(mat)
    o.location = loc
    return o


# ---------------------------------------------------------------- a room and what every room has

def shell(wall, floor, ceiling=(.62, .58, .52)):
    """Four walls, the one you look through with its window in it."""
    x0, x1, s, h = WIN
    t = .24
    w = blotched("wall", wall)
    box("wall_l", -ROOM_X - t, x0, -t, 0, 0, ROOM_Z, w)
    box("wall_r", x1, ROOM_X + t, -t, 0, 0, ROOM_Z, w)
    box("wall_low", x0, x1, -t, 0, 0, s, w)
    box("wall_high", x0, x1, -t, 0, h, ROOM_Z, w)
    box("wall_left", -ROOM_X - t, -ROOM_X, 0, ROOM_Y + t, 0, ROOM_Z, w)
    box("wall_right", ROOM_X, ROOM_X + t, 0, ROOM_Y + t, 0, ROOM_Z, w)
    box("wall_back", -ROOM_X, ROOM_X, ROOM_Y, ROOM_Y + t, 0, ROOM_Z, w)
    box("ceiling", -ROOM_X, ROOM_X, 0, ROOM_Y, ROOM_Z, ROOM_Z + .1, surface("ceiling", ceiling, .9, 0, .2))
    box("floor", -ROOM_X, ROOM_X, 0, ROOM_Y, -.1, 0, surface("floor", floor, .5))


def window_frame():
    """An ordinary sliding window in aluminium, with a pane in each half, and a little of the wet weather of its years along the bottom."""
    x0, x1, s, h = WIN
    metal = noisy(surface("frame", (.30, .285, .26), .42, .85), 400, .05, .0006)
    b, y0, y1 = .055, -.20, -.09
    box("frame_l", x0, x0 + b, y0, y1, s, h, metal)
    box("frame_r", x1 - b, x1, y0, y1, s, h, metal)
    box("frame_t", x0 + b, x1 - b, y0, y1, h - b, h, metal)
    box("frame_b", x0 + b, x1 - b, y0, y1, s, s + b, metal)
    m = (x0 + x1) / 2
    box("frame_m0", m - .05, m, y0, y0 + .05, s + b, h - b, metal)
    box("frame_m1", m, m + .05, y0 + .05, y1, s + b, h - b, metal)
    box("sill", x0 - .04, x1 + .04, -.30, -.2, s - .03, s, surface("sill", (.34, .32, .29), .8))
    g = glass_material()
    mesh_object("pane_l", [(x0 + b, y0 + .02, s + b), (m, y0 + .02, s + b), (m, y0 + .02, h - b), (x0 + b, y0 + .02, h - b)], [(0, 1, 2, 3)], g)["keep"] = 1
    mesh_object("pane_r", [(m, y0 + .07, s + b), (x1 - b, y0 + .07, s + b), (x1 - b, y0 + .07, h - b), (m, y0 + .07, h - b)], [(0, 1, 2, 3)], g)["keep"] = 1


def street(strength=.5, seed=3):
    """What the pane reflects: the lit windows of the far side of the street, out of focus and faint. Nothing else sees them."""
    rng = random.Random(seed)
    v, f = [], []
    for row in range(-16, 2):
        for col in range(-24, -4):
            if rng.random() > .42:
                continue
            x, z = col * 3.4 + rng.uniform(-.2, .2), row * 2.9
            n = len(v)
            v += [(x, -150, z), (x + 2.24, -150, z), (x + 2.24, -150, z + 1.4), (x, -150, z + 1.4)]
            f.append((n, n + 1, n + 2, n + 3))
    m = bpy.data.materials.new("street")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e, o = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    e.inputs[0].default_value = (1.0, .78, .52, 1)
    e.inputs[1].default_value = strength
    nt.links.new(e.outputs[0], o.inputs[0])
    ob = mesh_object("street", v, f, m)
    ob.visible_camera = ob.visible_diffuse = ob.visible_shadow = ob.visible_transmission = ob.visible_volume_scatter = False
    return ob


def monitor(x, y, yaw, picture, strength=1.0, spill=4.0, z=ZT):
    """A display on a table: an aluminium foot and arm, a thin case, and the glass facing the room's window."""
    M = Matrix.Translation((x, y, z)) @ Matrix.Rotation(yaw, 4, "Z")
    alu = noisy(surface("alu", (.72, .72, .72), .38, 1.0), 2600, .02, .0002, 3)
    box("foot", -.076, .076, -.057, .11, .0, .007, alu, .002, M)
    box("arm", -.076, .076, .06, .072, .007, .22, alu, .001, M)
    L = M @ Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    W, y0, y1 = DISP_W / 2, DISP_UP, DISP_UP + DISP_H
    slab("case", -W, W, y0, y1, -DISP_T, -.0005, .007, alu, L)
    slab("bezel", -W + .0007, W - .0007, y0 + .0007, y1 - .0007, -.0006, 0, .0063, surface("bezel", (.003, .003, .0035), .06), L)
    vc = (y0 + y1) / 2
    x0, x1, s0, s1 = -SCREEN_W / 2, SCREEN_W / 2, vc - SCREEN_H / 2, vc + SCREEN_H / 2
    o = mesh_object("screen", [(x0, s0, .0002), (x1, s0, .0002), (x1, s1, .0002), (x0, s1, .0002)], [(0, 1, 2, 3)], screen_material(picture, strength, spill), uv=[(0, 0), (1, 0), (1, 1), (0, 1)], matrix=L)
    o["keep"] = 1
    return o


def spine_book(name, x, w, h, depth, y, z, sheet, cell, cells, rng, cover):
    """A book standing up, its spine towards you."""
    box(name, x, x + w, y, y + depth, z, z + h, cover, .001)
    u0, u1 = cell / cells, (cell + 1) / cells
    v0 = 1 - h / .30
    mesh_object(name + "_spine", [(x + .0005, y - .0005, z + .001), (x + w - .0005, y - .0005, z + .001), (x + w - .0005, y - .0005, z + h - .001), (x + .0005, y - .0005, z + h - .001)], [(0, 1, 2, 3)], sheet, uv=[(u0, v0), (u1, v0), (u1, 1), (u0, 1)])


def shelf_row(x0, x1, y, z, sheet, rng, cover, depth=.17, cells=16, leaning=True):
    """Books side by side on a shelf, the odd one taller, with gaps where somebody took one away."""
    x, i = x0, 0
    deck = list(range(cells))
    rng.shuffle(deck)
    while x < x1:
        if rng.random() < .07:
            x += rng.uniform(.02, .07)
            continue
        w, h = rng.uniform(.017, .046), rng.uniform(.17, .25)
        if x + w > x1:
            break
        spine_book(f"book{i}_{z:.2f}", x, w, h, depth * rng.uniform(.8, 1), y + rng.uniform(0, .012), z, sheet, deck[i % cells], cells, rng, cover)
        x += w + .0006
        i += 1


def stack(x, y, z, count, sheet, rng, cover, cells=16, yaw_range=6):
    """Books lying flat, one on another, none quite square to the last. Their spines read sideways."""
    deck = list(range(cells))
    rng.shuffle(deck)
    for i in range(count):
        w, d, h = rng.uniform(.19, .25), rng.uniform(.14, .18), rng.uniform(.018, .04)
        M = Matrix.Translation((x + rng.uniform(-.012, .012), y + rng.uniform(-.008, .008), z)) @ Matrix.Rotation(math.radians(rng.uniform(-yaw_range, yaw_range)), 4, "Z")
        box(f"flat{i}_{x:.2f}", -w / 2, w / 2, 0, d, 0, h, cover, .001, M)
        u0, u1 = deck[i % cells] / cells, (deck[i % cells] + 1) / cells
        v0 = 1 - w / .30
        mesh_object(f"flat{i}_{x:.2f}_spine", [(-w / 2, -.0005, .001), (w / 2, -.0005, .001), (w / 2, -.0005, h - .001), (-w / 2, -.0005, h - .001)], [(0, 1, 2, 3)], sheet, uv=[(u0, 1), (u0, v0), (u1, v0), (u1, 1)], matrix=M)
        z += h
    return z


# ---------------------------------------------------------------- what is left of them when they have got up

def woven(m, pitch=.0018):
    """Cotton: threads over and under, too fine to see except close to, and a little unevenness in how it took the dye."""
    nt = m.node_tree
    tc, wx, wy, add, slub, bp = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeTexWave"), nt.nodes.new("ShaderNodeTexWave"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeBump")
    for wave, axis in ((wx, "X"), (wy, "Y")):
        wave.wave_type, wave.bands_direction = "BANDS", axis
        wave.inputs["Scale"].default_value = .314 / pitch
        nt.links.new(tc.outputs["Object"], wave.inputs["Vector"])
    add.operation = "MULTIPLY"
    nt.links.new(wx.outputs["Fac"], add.inputs[0])
    nt.links.new(wy.outputs["Fac"], add.inputs[1])
    bp.inputs["Strength"].default_value = .5
    bp.inputs["Distance"].default_value = .0009
    nt.links.new(add.outputs[0], bp.inputs["Height"])
    b = nt.nodes["Principled BSDF"]
    nt.links.new(bp.outputs[0], b.inputs["Normal"])
    slub.inputs["Scale"].default_value = 30
    mix = nt.nodes.new("ShaderNodeMath")
    mix.operation = "MULTIPLY_ADD"
    nt.links.new(slub.outputs[0], mix.inputs[0])
    mix.inputs[1].default_value = .12
    mix.inputs[2].default_value = .94
    nt.links.new(tc.outputs["Object"], slub.inputs["Vector"])
    # a stripe woven into it, one colour with itself, as a duvet cover has: it follows the folds where a plain cloth would show you nothing
    stripe = nt.nodes.new("ShaderNodeTexWave")
    stripe.wave_type, stripe.bands_direction, stripe.wave_profile = "BANDS", "X", "SAW"
    stripe.inputs["Scale"].default_value = .314 / .017
    nt.links.new(tc.outputs["Object"], stripe.inputs["Vector"])
    edge, both = nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeMath")
    edge.operation, edge.inputs[1].default_value = "GREATER_THAN", .5
    nt.links.new(stripe.outputs["Fac"], edge.inputs[0])
    both.operation, both.inputs[1].default_value, both.inputs[2].default_value = "MULTIPLY_ADD", .10, .95
    nt.links.new(edge.outputs[0], both.inputs[0])
    times = nt.nodes.new("ShaderNodeMath")
    times.operation = "MULTIPLY"
    nt.links.new(mix.outputs[0], times.inputs[0])
    nt.links.new(both.outputs[0], times.inputs[1])
    tint = nt.nodes.new("ShaderNodeVectorMath")
    tint.operation = "SCALE"
    tint.inputs[0].default_value = tuple(b.inputs["Base Color"].default_value)[:3]
    nt.links.new(times.outputs[0], tint.inputs["Scale"])
    nt.links.new(tint.outputs[0], b.inputs["Base Color"])
    return m


def bulb(loc, strength, size=.08, colour=LEMON):
    d = bpy.data.lights.new("bulb", "POINT")
    d.energy, d.color, d.shadow_soft_size = strength, colour, size
    o = link(bpy.data.objects.new("bulb", d))
    o.location = loc
    return o


def curtain(x0, x1, colour, seed=1, z0=.6, z1=2.42, y=.10):
    """Cloth drawn back to one side of the window. It hangs in folds, and it is out of focus."""
    r = np.random.default_rng(seed)
    xx, zz = np.meshgrid(np.linspace(x0, x1, 160), np.linspace(z0, z1, 30))
    yy = y + .035 * np.sin(xx * 58 + r.uniform(0, 6)) + .018 * np.sin(xx * 131 + zz * 1.3 + r.uniform(0, 6))
    o = mesh_grid("curtain", xx, yy, zz)
    o.data.materials.append(surface("curtain", colour, .9, 0, .15))
    return o


def thrown(name, x, y, width, top, drop, mat, turn=0.0, seed=1):
    """A piece of clothing over the back of a chair: down the near side in folds, its hem uneven."""
    r = np.random.default_rng(seed)
    u, v = np.meshgrid(np.linspace(-1, 1, 120), np.linspace(0, 1, 90))
    bag = sum(a * np.sin(fu * u + fv * v + r.uniform(0, 6.28)) for fu, fv, a in ((5, 2, .014), (11, 5, .010), (19, 3, .006), (31, 9, .003)))
    xx = width * u * (1 + .08 * v) + .012 * np.sin(v * 7 + u * 3)
    yy = -.018 - .035 * np.sin(v * math.pi) ** .5 - bag * (.3 + v)
    hem = .06 * np.sin(u * 4 + r.uniform(0, 6.28)) * v
    zz = top - drop * v + hem + .018 * (1 - u ** 2) * (1 - v) ** 8
    o = mesh_grid(name, xx, yy, zz)
    o.data.materials.append(mat)
    o.matrix_world = Matrix.Translation((x, y, 0)) @ Matrix.Rotation(turn, 4, "Z")
    return o


def office_chair(x, y, turn, dress):
    """Pushed back from the desk, turned a little, as it is left."""
    M = Matrix.Translation((x, y, 0)) @ Matrix.Rotation(turn, 4, "Z")
    black = surface("chair", (.025, .025, .027), .55, 0, .3)
    R = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    slab("chair_back", -.215, .215, .58, 1.0, -.03, .02, .07, black, M @ R)
    slab("chair_seat", -.23, .23, -.05, .40, .44, .50, .07, black, M)
    box("chair_stem", -.02, .02, .10, .14, .1, .44, surface("chrome", (.5, .5, .5), .25, 1.0), 0, M)
    thrown("hoodie", x, y, .225, 1.015, .40, dress, turn, 3)
    sleeve = [(.21, -.03, .98), (.255, -.05, .86), (.27, -.04, .72), (.255, -.05, .58)]
    tube("hoodie_sleeve", [tuple(M @ Vector(p)) for p in sleeve], .046, dress, 3)


def wooden_chair(x, y, turn, wood, cushion):
    """His chair, pushed back, a flat cushion gone the shape of him."""
    M = Matrix.Translation((x, y, 0)) @ Matrix.Rotation(turn, 4, "Z")
    box("chair_rail", -.25, .25, 0, .03, .80, .855, wood, .006, M)
    box("chair_rail2", -.22, .22, 0, .022, .64, .675, wood, .004, M)
    for s in (-1, 1):
        box("chair_post", s * .235 - .017, s * .235 + .017, 0, .03, .45, .82, wood, .004, M)
    box("chair_seat", -.25, .25, 0, .44, .43, .46, wood, .004, M)
    slab("cushion", -.21, .21, .03, .41, .46, .50, .06, cushion, M)


DX, DY = -.556, 3.12        # the display: the lens is off to the right of it


# ---------------------------------------------------------------- A: a student's room, under a bunk, everything on the one desk

def room_a():
    rng = random.Random(11)
    shell((.60, .56, .48), (.22, .20, .17))
    curtain(.93, 1.25, (.30, .33, .30), 2)
    desk = noisy(surface("desk", (.48, .38, .26), .45), 60, .03, .001)
    box("desk", -1.8, .75, 2.82, ROOM_Y, ZT - .025, ZT, desk, .002)
    board = surface("board", (.44, .35, .24), .5)
    sheet = printed("spines_a", "spines_a.png", .55)
    paper = surface("book_paper", (.70, .67, .58), .8)
    for z in (1.255, 1.64):
        box("shelf", -1.8, .75, 3.24, ROOM_Y, z, z + .025, board, .002)
        shelf_row(-1.75, .7, 3.27, z + .025, sheet, rng, paper)
    monitor(DX, DY, math.radians(4), "screen_a.png", 1.0, 3.0)
    # the books of this term on the desk, the timetable over them, this month beside it with today ringed
    stack(-.115, 3.12, ZT, 8, sheet, rng, paper)
    tape = surface("tape", (.75, .70, .50), .5)
    def pinned(name, x0, x1, z0, z1):
        mesh_object(name, [(x0, ROOM_Y - .002, z0), (x1, ROOM_Y - .002, z0), (x1, ROOM_Y - .002, z1), (x0, ROOM_Y - .002, z1)], [(0, 1, 2, 3)], printed(name, name + ".png", .7), uv=[(0, 0), (1, 0), (1, 1), (0, 1)])
        for x in (x0, x1):
            box("tape", x - .018, x + .018, ROOM_Y - .003, ROOM_Y - .0015, z1 - .007, z1 + .007, tape)
    pinned("timetable", -.354, -.054, 1.0, 1.186)
    pinned("calendar", -1.30, -1.04, .94, 1.20)
    # a cup of noodles finished hours ago, with its fork; a can; a mug of pens; a bottle
    lathe("noodles", [(.0, .0), (.052, .0), (.069, .105), (.071, .107), (.0, .107)], -.235, 2.97, ZT, surface("cup_paper", (.74, .56, .32), .55))
    box("fork", -.31, -.16, 2.965, 2.972, ZT + .107, ZT + .110, surface("fork", (.7, .68, .6), .3))
    lathe("can", [(.0, .0), (.031, .0), (.033, .01), (.033, .105), (.027, .116), (.0, .116)], -.03, 2.93, ZT, surface("can", (.42, .05, .04), .25, .7))
    lathe("mug", [(.0, .0), (.04, .0), (.042, .1), (.038, .1), (.036, .01), (.0, .01)], -.40, 2.92, ZT, surface("mug", (.62, .60, .54), .25))
    for i, c in enumerate(((.05, .05, .25), (.5, .08, .05), (.03, .03, .03))):
        tube(f"pen{i}", [(-.40 + i * .012 - .012, 2.92, ZT + .03), (-.40 + i * .035 - .035, 2.92 + i * .01, ZT + .205)], .0048, surface(f"pen{i}", c, .35), 2)
    lathe("bottle", [(.0, .0), (.036, .0), (.038, .02), (.038, .19), (.022, .235), (.015, .25), (.015, .27), (.0, .27)], .10, 3.3, ZT, surface("bottle", (.25, .30, .30), .15, 0, .6))
    # on the left of the desk the rest of it: more books lying open-faced, a box of instant noodles, a power strip with too much in it
    stack(-1.02, 3.08, ZT, 5, sheet, rng, paper, yaw_range=11)
    box("noodle_box", -.98, -.76, 3.24, 3.44, ZT, ZT + .19, surface("carton", (.46, .30, .16), .75), .002)
    box("noodle_flap", -.98, -.76, 3.235, 3.24, ZT + .10, ZT + .19, surface("carton2", (.60, .16, .10), .6))
    box("strip", -1.18, -.86, 2.92, 2.97, ZT, ZT + .03, surface("strip", (.66, .64, .60), .4), .004)
    black = surface("plug", (.02, .02, .02), .5)
    for i in range(4):
        box("plug", -1.15 + i * .075, -1.10 + i * .075, 2.925, 2.965, ZT + .03, ZT + .065, black, .004)
    tube("lead", [(-1.1, 2.97, ZT + .05), (-1.0, 3.06, ZT + .01), (-.8, 3.0, ZT + .005), (DX + .05, 3.2, ZT + .005)], .003, black, 2)
    # a keyboard, a cable going nowhere tidy, papers
    box("keyboard", DX - .22, DX + .22, 2.84, 2.98, ZT, ZT + .016, surface("kb", (.05, .05, .055), .45), .003)
    tube("cable", [(DX + .1, 3.25, ZT + .005), (-.42, 3.02, ZT + .005), (-.5, 2.9, ZT + .005), (-.2, 2.86, ZT + .005), (.05, 3.0, ZT + .005)], .0028, surface("cable", (.02, .02, .02), .5), 2)
    for i in range(5):
        M = Matrix.Translation((-.16 + rng.uniform(-.05, .05), 2.9 + rng.uniform(-.02, .02), ZT + .0005 + i * .0006)) @ Matrix.Rotation(math.radians(rng.uniform(-14, 14)), 4, "Z")
        box("sheet", -.105, .105, -.075, .075, 0, .0004, surface("sheet", (.75, .74, .70), .85), matrix=M)
    # the bunk overhead: its post, its ladder, a towel on a rung
    steel = surface("steel", (.20, .21, .22), .4, .8)
    box("post", -1.02, -.97, 2.68, 2.73, 0, ROOM_Z, steel)
    for z in (.55, 1.05, 1.55):
        box("rung", -1.24, -1.02, 2.69, 2.72, z, z + .022, steel)
    box("bunk_rail", -1.8, .75, 2.68, 2.73, 1.62, 1.66, steel)
    towel = noisy(surface("towel", (.26, .29, .30), .95), 300, .3, .002)
    xx, zz = np.meshgrid(np.linspace(-1.22, -1.04, 24), np.linspace(.62, 1.07, 24))
    yy = 2.68 + .012 * np.sin((xx + 1.2) * 60 + zz * 4) * (1.1 - zz)
    mesh_grid("towel", xx, yy, zz).data.materials.append(towel)
    # the chair, pushed back, a hoodie over it
    office_chair(DX - .57, 2.40, math.radians(16), noisy(surface("hoodie", (.30, .31, .33), .95, 0, .15), 180, .5, .004))
    # the lamp clipped under the shelf, and the ceiling light somebody left on
    lamp = surface("lamp", (.05, .05, .05), .4)
    slab("lamp_head", -.04, .04, -.03, .03, 0, .022, .02, lamp, Matrix.Translation((-.24, 3.33, 1.232)))
    bulb((-.24, 3.3, 1.2), 2.8, .03)
    bulb((-.7, 2.95, 2.5), 12, .3)
    bulb((-1.1, 1.5, 2.4), 7, .4)
    return .28


# ---------------------------------------------------------------- B: an old man's desk: his lamp, his books, his stick, and a screen he has only lately learned

def pleated_shade(x, y, z, r0, r1, h, power):
    a = np.linspace(0, 2 * math.pi, 721)
    s = np.linspace(0, 1, 17)[:, None]
    r = (r0 + (r1 - r0) * s) * (1 + .010 * np.abs(np.sin(a * 42)))
    xx, yy = x + r * np.cos(a), y + r * np.sin(a)
    zz = np.broadcast_to(z + h * s, xx.shape).copy()
    o = mesh_grid("shade", xx, yy, zz)
    glow = (.55 + .45 * np.sin(s * math.pi) ** .8) * (.92 + .08 * np.abs(np.sin(a * 42)))
    layer = o.data.color_attributes.new("glow", "FLOAT_COLOR", "POINT")
    c = np.ones((glow.size, 4), np.float32)
    c[:, :3] = glow.ravel()[:, None]
    layer.data.foreach_set("color", c.ravel())
    m = bpy.data.materials.new("shade")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    at, e, gain, out = nt.nodes.new("ShaderNodeVertexColor"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeOutputMaterial")
    at.layer_name = "glow"
    e.inputs[0].default_value = (1.0, .63, .33, 1)
    gain.operation = "MULTIPLY"
    gain.inputs[1].default_value = power
    nt.links.new(at.outputs[0], gain.inputs[0])
    nt.links.new(gain.outputs[0], e.inputs[1])
    nt.links.new(e.outputs[0], out.inputs[0])
    o.data.materials.append(m)
    return o


def room_b():
    rng = random.Random(23)
    shell((.40, .35, .27), (.16, .10, .06))
    wood = noisy(surface("oak", (.20, .115, .06), .35), 40, .03, .001)
    box("desk", -1.8, .8, 2.72, ROOM_Y, ZT - .035, ZT, wood, .003)
    case = noisy(surface("bookcase", (.15, .085, .048), .4), 40, .02, .001)
    sheet = printed("spines_b", "spines_b.png", .62)
    paper = surface("old_paper", (.60, .54, .41), .85)
    box("case_back", -1.8, .8, ROOM_Y - .012, ROOM_Y, ZT, ROOM_Z - .1, case)
    for z in (ZT + .012, 1.04, 1.36, 1.70, 2.04):
        box("case_board", -1.8, .8, 3.22, ROOM_Y, z - .022, z, case, .002)
        shelf_row(-1.77, .77, 3.27, z, sheet, rng, paper, .2)
    for x in (-1.3, -.72, -.14, .44):
        box("case_side", x, x + .025, 3.22, ROOM_Y, ZT, ROOM_Z - .1, case, .002)
    monitor(DX, DY, math.radians(4), "screen_b.png", 1.0, 2.0)
    # the lamp: a brass stem and a pleated shade, the only other light
    brass = noisy(surface("brass", (.60, .42, .18), .3, 1.0), 900, .03, .0003)
    lx, ly = .0, 3.0
    lathe("lamp_foot", [(.0, .0), (.075, .0), (.075, .012), (.03, .03), (.012, .05), (.0, .05)], lx, ly, ZT, brass)
    tube("lamp_stem", [(lx, ly, ZT + .05), (lx, ly, ZT + .22), (lx, ly, ZT + .36)], .008, brass, 2)
    pleated_shade(lx, ly, ZT + .30, .165, .095, .19, .95)
    trim = surface("trim", (.42, .28, .12), .4, .6)
    for rad, h in ((.165, .30), (.095, .49)):
        tube("lamp_trim", [(lx + rad * math.cos(a), ly + rad * math.sin(a), ZT + h) for a in np.linspace(0, 2 * math.pi, 25)], .0035, trim, 2)
    bulb((lx, ly, ZT + .38), 6.5, .04)
    # his books, his glasses on them, his cup with its lid
    top = stack(-.19, 2.88, ZT, 4, sheet, rng, paper, yaw_range=3)
    ring = surface("rim", (.16, .13, .08), .25, .8)
    for dx in (-.034, .034):
        loop = [(-.19 + dx + .027 * math.cos(a), 2.875, top + .026 + .021 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 15)]
        tube("lens", loop, .0015, ring, 2)
    cup = surface("cup", (.70, .67, .58), .18)
    lathe("cup", [(.0, .0), (.034, .0), (.043, .095), (.040, .095), (.0, .096)], -.33, 2.84, ZT, cup)
    lathe("cup_lid", [(.044, .0), (.044, .006), (.022, .018), (.008, .02), (.008, .03), (.0, .03)], -.33, 2.84, ZT + .095, cup)
    # on the left a radio older than the books, with two knobs and a cloth front; a tin of tea
    M = Matrix.Translation((-1.03, 3.04, ZT)) @ Matrix.Rotation(math.radians(-5), 4, "Z")
    box("radio", -.15, .15, 0, .09, 0, .17, noisy(surface("radio", (.15, .075, .04), .3), 60, .02, .001), .006, M)
    box("radio_cloth", -.135, .04, -.001, 0, .02, .15, noisy(surface("cloth", (.34, .29, .2), .95), 900, .6, .001), 0, M)
    wood = surface("radio_slat", (.13, .065, .035), .35)
    for k in range(8):      # slats across the cloth, a tuning scale above the knobs, a handle on top
        box("radio_slat", -.135, .04, -.005, -.001, .028 + k * .0165, .035 + k * .0165, wood, 0, M)
    box("radio_dial", .055, .135, -.003, -.001, .135, .158, surface("radio_scale", (.62, .50, .30), .4), 0, M)
    for k in range(9):
        box("radio_tick", .06 + k * .0088, .0612 + k * .0088, -.0036, -.003, .14, .153, surface("radio_tick", (.05, .04, .03), .5), 0, M)
    tube("radio_handle", [tuple(M @ Vector(p)) for p in ((-.09, .045, .17), (-.09, .045, .195), (-.05, .045, .212), (.05, .045, .212), (.09, .045, .195), (.09, .045, .17))], .006, surface("radio_strap", (.09, .05, .03), .5), 2)
    knob = surface("knob", (.60, .52, .36), .3, .7)
    for z in (.05, .12):
        o = lathe("radio_knob", [(.0, .0), (.015, .0), (.015, .012), (.0, .013)], 0, 0, 0, knob, 24)
        o.matrix_world = M @ Matrix.Translation((.095, -.001, z)) @ Matrix.Rotation(math.pi / 2, 4, "X")
    lathe("tea_tin", [(.0, .0), (.045, .0), (.045, .125), (.046, .127), (.046, .155), (.0, .155)], -.79, 2.94, ZT, surface("tin", (.34, .14, .07), .35, .6))
    # his chair, and what he has left on it
    # the lamp's light as far as the radio and the chair: only along their edges
    rim = bpy.data.lights.new("rim", "SPOT")
    rim.energy, rim.color, rim.spot_size, rim.spot_blend, rim.shadow_soft_size = 3.6, (1.0, .70, .40), math.radians(60), .6, .03
    o = link(bpy.data.objects.new("rim", rim))
    o.location = (-.42, 2.70, ZT + .34)
    aim(o, o.location, (-1.0, 3.02, ZT + .06))
    wooden_chair(DX - 1.05, 2.28, math.radians(14), surface("chair_b", (.095, .052, .028), .42), noisy(surface("cushion", (.28, .19, .11), .9), 300, .4, .002))
    bulb((-.6, 1.4, ROOM_Z - .05), 2.2, .4)
    bulb((-1.15, 2.85, 1.95), 1.5, .2)
    return .28


# ---------------------------------------------------------------- C: a bedroom in the small hours, nothing on but the screen

def pillow(name, loc, tilt, turn, mat, seed):
    """A pillow stood on end: flat at its corners and fat in the middle, the slip dented where a head has been."""
    r = np.random.default_rng(seed)
    u, v = np.meshgrid(np.linspace(-1, 1, 90), np.linspace(-1, 1, 60))
    fat = ((1 - u ** 4) * (1 - v ** 4)).clip(0) ** .55
    crease = .006 * np.sin(u * 17 + v * 5 + r.uniform(0, 6)) + .004 * np.sin(u * 7 - v * 23 + r.uniform(0, 6))
    dent = 1 - .35 * np.exp(-((u - .1) / .5) ** 2 - ((v + .05) / .5) ** 2)
    x, z = .25 * u * (1 - .05 * v ** 2), .15 * v * (1 - .06 * u ** 2)
    front = mesh_grid(name, x, -.07 * fat * dent - crease * fat, z)
    back = mesh_grid(name + "_back", x[:, ::-1], (.07 * fat)[:, ::-1], z[:, ::-1])
    for o in (front, back):
        o.data.materials.append(mat)
        o.location = loc
        o.rotation_euler = (math.radians(tilt), math.radians(turn), 0)


def room_c():
    rng = np.random.default_rng(37)
    shell((.44, .40, .35), (.14, .10, .08))
    curtain(.90, 1.25, (.14, .12, .10), 5, 1.0, 2.42)
    top = .62
    table = noisy(surface("table_c", (.26, .19, .13), .4), 60, .02, .001)
    box("table", -1.25, -.17, 2.92, ROOM_Y, top - .025, top, table, .002)
    monitor(DX, DY, math.radians(4), "screen_c.png", .92, 6.0, top)
    lathe("glass", [(.0, .0), (.032, .0), (.036, .115), (.033, .115), (.03, .01), (.0, .01)], -.235, 3.0, top, surface("tumbler", (.55, .55, .52), .08))
    # a picture frame lying on its face; a phone, also on its face; a box of tissues and what has come out of it
    box("frame_down", -.34, -.20, 3.14, 3.33, top, top + .013, surface("frame_back", (.10, .07, .05), .55), .002)
    box("phone", -1.03, -.96, 2.97, 3.12, top, top + .008, surface("phone_back", (.04, .04, .045), .25, .3), .002, Matrix.Rotation(math.radians(-16), 4, "Z"))
    box("tissue_box", -1.19, -.95, 3.16, 3.30, top, top + .085, surface("tissue_box", (.44, .38, .30), .6), .003)
    # the bed nobody has got into: the duvet thrown back as far as the table in big soft folds, a pillow against the wall. Cotton: nothing on it is sharp but a few small wrinkles, where it has been pulled
    xx, yy = np.meshgrid(np.linspace(-1.75, .85, 1040), np.linspace(1.0, 2.90, 760))
    zz = .60 + .10 * np.exp(-((xx + .85) / .62) ** 2 - ((yy - 2.35) / .42) ** 2) + .055 * np.exp(-((xx + .02) / .45) ** 2 - ((yy - 2.5) / .34) ** 2)
    for angle, at, a, wide, bend in ((-32, -.78, .070, .15, .10), (-20, -.20, .058, .13, .08), (24, -1.1, .050, .17, .12), (-42, .22, .046, .12, .06), (12, -.46, .040, .16, .10)):     # the big folds
        c, s_ = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        d = c * (xx - at) + s_ * (yy - 2.45) + bend * np.sin(2.1 * (-s_ * xx + c * yy) + at * 3)
        zz = zz + a * np.exp(-(d / wide) ** 2)
    # where a fold is pulled it comes to a soft edge and goes in sharply: that is what makes it cloth and not sand
    for angle, x0, y0, a, wide, long_ in ((-48, -.95, 2.42, .022, .05, .46), (-30, -.52, 2.46, .026, .055, .50), (-62, -.30, 2.62, .018, .045, .36), (-18, -.02, 2.40, .020, .05, .42), (38, -1.28, 2.55, .018, .05, .38)):
        c, s_ = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        d, along = c * (xx - x0) + s_ * (yy - y0), -s_ * (xx - x0) + c * (yy - y0)
        zz = zz - a * np.exp(-np.abs(d) / wide) * np.exp(-(along / long_) ** 2)
    for angle, x0, y0, a, wide, long_ in ((-55, -.66, 2.60, .009, .016, .17), (-38, -.40, 2.72, .008, .014, .14), (-70, -.92, 2.52, .007, .015, .15), (-25, .05, 2.66, .007, .013, .12), (-60, -.25, 2.50, .006, .014, .13)):     # small wrinkles, where the folds pull
        c, s_ = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        d, along = c * (xx - x0) + s_ * (yy - y0), -s_ * (xx - x0) + c * (yy - y0)
        zz = zz + a * np.exp(-(d / wide) ** 2 - (along / long_) ** 2)
    # the edge of the top layer, turned back: a hem, thick as cloth folded over once, running across all of it
    d = yy - (2.20 + .09 * np.sin(1.5 * xx + .5) + .05 * xx)
    zz = zz + .013 * (1 + np.tanh(d / .004)) / 2 + .012 * np.exp(-((d - .028) / .03) ** 2)
    sheet = woven(surface("bedding", (.46, .44, .41), .95, 0, .12))
    mesh_grid("bed", xx, yy, zz).data.materials.append(sheet)
    box("bed_under", -1.75, .85, 1.0, 2.90, .0, .56, surface("base", (.05, .04, .03), .8))
    case = noisy(surface("pillow", (.52, .50, .46), .95), 80, .25, .004)
    pillow("pillow", (.30, 2.74, .72), -24, -8, case, 0)
    bulb((.6, .5, 2.2), 1.6, .8, (1.0, .78, .55))
    return .05


ROOMS = dict(
    # cam: where the lens is, across the street, off to one side and at about the same floor. at: what it looks at, in the plane of the screen. width: how many metres across the frame takes in there
    C2A=dict(make=room_a, cam=(3.3, -9.0, 1.77), at=(DX, DY, .994), width=1.33, fstop=5.6, world=(.030, .023, .017)),
    C2B=dict(make=room_b, cam=(3.3, -9.0, 1.77), at=(DX, DY, .994), width=1.33, fstop=5.6, world=(.020, .015, .011)),
    C2C=dict(make=room_c, cam=(3.3, -9.0, 3.6), at=(DX, DY, .848), width=1.33, fstop=5.0, world=(.012, .009, .007)),
)


# ---------------------------------------------------------------- rendering

def plate(sc, path):
    """The render as linear light, float16, top row first."""
    exr = path + ".exr"
    sc.render.filepath = exr
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(exr)
    w, h = im.size
    a = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(a)
    np.save(path + ".npy", a.reshape(h, w, 4)[::-1][..., :3].astype(np.float16))
    bpy.data.images.remove(im)
    os.remove(exr)


def words_only():
    """Everything black but the screen and the panes it is seen through; the screen shows itself, to the lens only."""
    black = surface("black", (0, 0, 0), 1.0, 0, 0)
    for o in bpy.data.objects:
        if o.type not in ("MESH", "CURVE", "CURVES"):
            continue
        if o.get("keep"):
            continue
        o.data.materials.clear()
        o.data.materials.append(black)
    for o in list(bpy.data.objects):
        if o.type == "LIGHT" or o.name.startswith("street"):
            bpy.data.objects.remove(o)
    screen = bpy.data.objects["screen"]
    m = screen.data.materials[0]
    nt = m.node_tree
    e = next(n for n in nt.nodes if n.type == "EMISSION")
    for link_ in list(e.inputs[1].links):
        nt.links.remove(link_)
    e.inputs[1].default_value = m["words"]
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.0


def shot(key, pct, samples):
    r = ROOMS[key]
    os.makedirs(PLATES, exist_ok=True)
    t0 = time.time()
    sc = reset(samples, pct)
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.color_mode = "RGB"
    sc.cycles.max_bounces = 6
    sc.cycles.sample_clamp_indirect = 6.0
    sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    bg = sc.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (*r["world"], 1)
    bg.inputs[1].default_value = 1.0
    window_frame()
    street(r["make"]())
    c = bpy.data.cameras.new("cam")
    far = (Vector(r["at"]) - Vector(r["cam"])).length
    c.lens = 36.0 * far / r["width"]
    c.clip_start, c.clip_end = .5, 400.0
    c.dof.use_dof = True
    c.dof.focus_distance = far
    c.dof.aperture_fstop = r["fstop"]
    o = link(bpy.data.objects.new("cam", c))
    sc.camera = o
    aim(o, r["cam"], r["at"])
    plate(sc, os.path.join(PLATES, key + "_room"))
    print(key, "room", round(time.time() - t0, 1), flush=True)
    t1 = time.time()
    words_only()
    sc.cycles.use_denoising = False
    sc.cycles.samples = max(samples // 2, 32)
    plate(sc, os.path.join(PLATES, key + "_words"))
    print(key, "words", round(time.time() - t1, 1), flush=True)


# ---------------------------------------------------------------- the rest of the city: what a window looks like from across the street, for the windows too far off to know

def figure(x, y, z=0.0, pose="sits", turn=0.0, small=1.0, hairdo="short"):
    """A shape and nothing else: black, unlit, a few pixels of outline in a window across the street. The way it sits is all it has. y points the way it faces, and it faces away."""
    black = bpy.data.materials.get("nobody") or surface("nobody", (0, 0, 0), 1.0, 0, 0)
    lean = dict(sits=-3, back=13, hunch=-24)[pose]
    M = Matrix.Translation((x, y, z + .46)) @ Matrix.Rotation(turn, 4, "Z") @ Matrix.Rotation(math.radians(lean), 4, "X") @ Matrix.Scale(small, 4)
    made = [loft("torso", [(.0, .21, .13), (.38, .25, .13), (.47, .245, .12), (.52, .19, .10), (.56, .10, .075), (.59, .07, .062)], black, (0, 0, 0)),
            loft("neck", [(.56, .062, .06), (.64, .058, .056)], black, (0, 0, 0), 1.0),
            ellipsoid("head", (.082, .092, .108), (0, 0, .705), black, 16)]
    if hairdo == "long":
        made.append(loft("hair", [(.40, .085, .03), (.56, .105, .05), (.72, .098, .075)], black, (0, -.04, 0), .9))
    elif hairdo == "bun":
        made.append(ellipsoid("bun", (.042, .042, .042), (0, -.035, .86), black, 12))
    for o in made:
        o.matrix_world = M @ Matrix.Translation(o.location)
    arms = dict(sits=(((-.22, 0, .5), (-.27, .04, .24), (-.13, .3, .3)), ((.22, 0, .5), (.27, .04, .24), (.13, .3, .3))),
                hunch=(((-.21, 0, .5), (-.3, .12, .22), (-.12, .3, .34)), ((.21, 0, .5), (.3, .12, .22), (.12, .3, .34))),
                back=(((-.21, 0, .5), (-.27, -.04, .24), (-.31, -.02, .06)), ((.21, 0, .5), (.27, .1, .25), (.14, .35, .3))))[pose]
    for arm in arms:
        tube("arm", [tuple(M @ Vector(p)) for p in arm], .042 * small, black, 2)


def mirrored(side):
    """The same room the other way round: a flat lies as it likes."""
    if side < 0:
        bpy.context.view_layer.update()
        flip = Matrix.Scale(-1, 4, (1, 0, 0))
        for o in bpy.data.objects:
            o.matrix_world = flip @ o.matrix_world


def glow_material(colour, power):
    m = bpy.data.materials.new("glow")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e, o = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    e.inputs[0].default_value = (*colour, 1)
    e.inputs[1].default_value = power
    nt.links.new(e.outputs[0], o.inputs[0])
    return m


def pendant(x, y, z, power=1.1, colour=(1.0, .7, .42)):
    tube("flex", [(x, y, z + .15), (x, y, ROOM_Z)], .004, surface("flex", (.03, .03, .03), .5), 2)
    lathe("pendant", [(.21, .0), (.10, .16)], x, y, z, glow_material(colour, power))
    bulb((x, y, z + .06), 15, .1)


def kitchen(r):
    """A kitchen after dinner: cupboards, a hood, a fridge, the table under its lamp."""
    shell(r.choice(((.66, .60, .50), (.60, .56, .46))), (.30, .25, .20))
    doors = surface("doors", r.choice(((.56, .50, .38), (.42, .30, .20), (.62, .60, .52))), .35)
    gap = surface("gap", (.02, .02, .02), .6)
    box("base", -1.75, .55, 2.95, ROOM_Y, 0, .86, doors)
    box("counter", -1.75, .55, 2.93, ROOM_Y, .86, .90, surface("stone", (.18, .17, .16), .25))
    box("splash", -1.75, .55, ROOM_Y - .01, ROOM_Y, .90, 1.5, surface("tile", (.66, .64, .58), .2, 0, .6))
    box("wall_cups", -1.75, -.2, 3.15, ROOM_Y, 1.5, 2.25, doors)
    for i in range(1, 4):
        box("cup_gap", -1.75 + i * .39, -1.745 + i * .39, 3.145, 3.15, 1.5, 2.25, gap)
    box("hood", -.15, .5, 3.0, ROOM_Y, 1.62, 1.82, surface("hood", (.42, .42, .40), .3, .8))
    box("hood_up", .05, .3, 3.25, ROOM_Y, 1.82, ROOM_Z, surface("hood2", (.42, .42, .40), .3, .8))
    fridge = surface("fridge", (.60, .58, .54), .3)
    box("fridge", .7, 1.45, 2.75, ROOM_Y, 0, 1.85, fridge, .01)
    box("fridge_gap", .7, 1.45, 2.745, 2.75, 1.25, 1.26, gap)
    box("fridge_grip", .76, .79, 2.72, 2.745, 1.3, 1.65, surface("grip", (.2, .2, .2), .3, .8))
    for x, k in ((-1.5, .3), (-1.3, .24), (-1.15, .28), (-.8, .2)):
        lathe("jar", [(.0, .0), (.04, .0), (.04, k * .7), (.018, k), (.0, k)], x, 3.25, .9, surface("jar", (r.uniform(.15, .5), r.uniform(.1, .3), r.uniform(.05, .2)), .3))
    lathe("kettle", [(.0, .0), (.085, .0), (.09, .12), (.05, .2), (.0, .21)], .2, 3.2, .9, surface("kettle", (.5, .48, .45), .25, .9))
    box("table", -.95, .55, 1.75, 2.65, .72, .76, noisy(surface("table", (.40, .27, .15), .4), 50, .02, .001), .004)
    pendant(r.uniform(-.5, .1), 2.2, 1.52)
    bulb((0, 1.0, 2.5), 14, .4)
    return r.uniform(-.5, .1), 2.42, .76, 1.68, 0.0


def children(r):
    """A child's room: a bed up a ladder, soft things on it, boxes of colours, paper stars and drawings on the wall."""
    shell(r.choice(((.74, .62, .42), (.62, .66, .50), (.74, .56, .44))), (.34, .26, .18))
    wood = surface("pine", (.56, .40, .22), .5)
    box("bunk", -1.75, -.2, 1.3, ROOM_Y, 1.42, 1.52, wood)
    box("bunk_top", -1.75, -.2, 1.3, ROOM_Y, 1.52, 1.66, surface("quilt", r.choice(((.54, .26, .18), (.24, .36, .40), (.62, .50, .22))), .9))
    box("bunk_rail", -.22, -.18, 1.3, ROOM_Y, 1.66, 1.92, wood)
    for y in (1.3, 2.4, 3.46):
        box("bunk_post", -.24, -.18, y, y + .05, 0, 1.92, wood)
    for i in range(5):
        box("ladder", -.21, -.16, 1.34, 1.9, .3 + i * .3, .33 + i * .3, wood)
    box("duvet", -1.45, -.3, 1.45, 2.75, 1.66, 1.78, surface("duvet", r.choice(((.62, .54, .42), (.50, .50, .46), (.58, .44, .34))), .95), .05)
    ellipsoid("pillow", (.28, .17, .07), (-.85, 3.05, 1.73), surface("pillow", (.70, .66, .58), .9), 20)
    box("cubes", .15, 1.65, 3.15, ROOM_Y, 0, 1.15, wood, .004)
    for i in range(4):
        for j in range(3):
            box("cube", .2 + i * .36, .2 + i * .36 + .33, 3.13, 3.16, .06 + j * .36, .06 + j * .36 + .32, surface("bin", r.choice(((.62, .24, .14), (.2, .34, .36), (.68, .54, .16), (.5, .4, .3))), .7))
    n = r.randint(3, 5)
    for i in range(n):
        x, z, w, h = r.uniform(-.1, 1.3), r.uniform(1.25, 1.85), r.uniform(.25, .42), r.uniform(.2, .34)
        y = ROOM_Y - .004 - i * .002
        mesh_object("drawing", [(x, y, z), (x + w, y, z), (x + w, y, z + h), (x, y, z + h)], [(0, 1, 2, 3)], surface("drawing", r.choice(((.72, .68, .56), (.66, .44, .30), (.40, .52, .50))), .9))
    for i in range(14):
        x, z, k = r.uniform(.1, 1.6), r.uniform(1.75, 2.45), r.uniform(.025, .05)
        y = ROOM_Y - .004 - n * .002 - .002
        mesh_object("star", [(x - k, y, z), (x, y, z - k), (x + k, y, z), (x, y, z + k)], [(0, 1, 2, 3)], surface("star", (.8, .68, .32), .6))
    lathe("globe", [(.0, -.11), (.07, -.09), (.11, .0), (.07, .09), (.0, .11)], 1.3, 3.3, 1.26, surface("globe", (.25, .36, .34), .25), 24)
    bulb((.3, 1.7, 2.55), 42, .3)
    box("small_desk", .1, 1.2, 2.55, 3.1, .68, .71, wood, .003)
    return r.uniform(.4, .9), 2.82, .71, 2.2, .14


def bedside(r):
    """A bedroom with the bed made and two small lamps on: the bed's head against the far wall, a picture over it."""
    shell(r.choice(((.50, .43, .35), (.46, .42, .36))), (.20, .14, .10))
    box("headboard", -1.3, .4, ROOM_Y - .08, ROOM_Y, .4, 1.28, noisy(surface("headboard", r.choice(((.30, .22, .15), (.36, .32, .26))), .85), 200, .3, .002), .02)
    box("bed", -1.3, .4, 1.6, ROOM_Y - .08, 0, .58, surface("bedding", (.56, .52, .46), .9))
    for x in (-.85, -.1):
        ellipsoid("pillow", (.30, .10, .14), (x, 3.25, .76), surface("pillow", (.62, .58, .52), .9), 20)
    box("art", -.85, -.05, ROOM_Y - .03, ROOM_Y, 1.5, 2.0, surface("art_frame", (.08, .06, .04), .4))
    box("art_in", -.81, -.09, ROOM_Y - .033, ROOM_Y - .03, 1.54, 1.96, surface("art", r.choice(((.46, .36, .24), (.30, .34, .30))), .8))
    for x in (-1.55, .65):
        box("nightstand", x - .22, x + .22, 3.05, ROOM_Y, 0, .55, surface("stand", (.22, .14, .08), .4), .004)
        lathe("lamp_pot", [(.0, .0), (.05, .0), (.06, .12), (.015, .22), (.0, .22)], x, 3.25, .55, surface("pot", (.45, .36, .24), .3))
        lathe("lamp_shade", [(.14, .0), (.09, .17)], x, 3.25, .79, glow_material((1.0, .66, .38), .9))
        bulb((x, 3.2, .86), 15, .05)
    box("wardrobe", 1.05, 1.75, 2.6, ROOM_Y, 0, 2.3, surface("wardrobe", (.26, .18, .11), .4), .005)
    box("side_desk", -.1, 1.0, 2.55, 3.05, .70, .73, surface("side_desk", (.30, .2, .12), .4), .003)
    return r.uniform(.3, .6), 2.8, .73, 2.15, 0.08


def dormitory(r, sheets):
    """Four to a room, or was: steel bunks, things on every flat place, washing across the middle of it."""
    shell((.58, .56, .50), (.26, .25, .22))
    steel = surface("steel", (.30, .31, .32), .4, .7)
    for x in (-1.72, -.25, .25, 1.72):
        for y in (1.2, ROOM_Y - .05):
            box("post", x - .025, x + .025, y, y + .05, 0, 2.2, steel)
    for x0, x1 in ((-1.72, -.25), (.25, 1.72)):
        box("upper", x0, x1, 1.2, ROOM_Y, 1.62, 1.68, steel)
        box("mattress", x0, x1, 1.2, ROOM_Y, 1.68, 1.8, surface("mattress", r.choice(((.42, .40, .34), (.30, .34, .36))), .9))
        for k in range(r.randint(2, 4)):      # boxes and bags on the top bunk
            w, h = r.uniform(.25, .5), r.uniform(.15, .36)
            x = r.uniform(x0 + .05, x1 - w - .05)
            box("thing", x, x + w, r.uniform(2.6, 3.0), 3.4, 1.8, 1.8 + h, surface("thing", r.choice(((.46, .30, .16), (.2, .24, .3), (.52, .48, .4), (.5, .2, .14))), .8), .004)
        box("under_shelf", x0, x1, 3.2, ROOM_Y, 1.28, 1.31, steel)
        shelf_row(x0 + .03, x1 - .03, 3.24, 1.31, r.choice(sheets), r, surface("pages", (.62, .58, .48), .8))
        box("desk", x0, x1, 2.7, ROOM_Y, .72, .75, surface("desk", (.42, .32, .20), .45), .003)
        for k in range(r.randint(3, 6)):
            w, h = r.uniform(.08, .26), r.uniform(.08, .3)
            x = r.uniform(x0 + .05, x1 - w - .05)
            box("clutter", x, x + w, r.uniform(2.95, 3.15), 3.2, .75, .75 + h, surface("clutter", r.choice(((.6, .56, .48), (.3, .3, .32), (.46, .22, .14), (.6, .48, .24))), .7), .003)
    y = r.uniform(1.7, 2.4)
    box("line", -1.72, 1.72, y, y + .012, 2.14, 2.152, surface("line", (.4, .4, .38), .5))
    x = -1.6
    while x < 1.4:
        w, h = r.uniform(.28, .46), r.uniform(.4, .75)
        if r.random() < .55:
            box("wash", x, x + w, y, y + .01, 2.14 - h, 2.14, surface("wash", r.choice(((.40, .26, .18), (.32, .36, .38), (.54, .50, .42), (.26, .26, .3))), .9))
        x += w + r.uniform(.05, .3)
    bulb((0, 1.8, 2.6), 44, .5, (1.0, .86, .68))
    side = r.choice((-1, 1))
    return side * r.uniform(.7, 1.2), 3.05, .75, 2.4, 0.0


def study(r, sheets):
    """Books to the ceiling and a lamp on the desk."""
    shell((.40, .34, .26), (.16, .10, .06))
    wood = noisy(surface("oak", (.20, .115, .06), .35), 40, .03, .001)
    paper = surface("pages", (.60, .54, .41), .8)
    box("case_back", -1.75, 1.75, ROOM_Y - .012, ROOM_Y, 0, ROOM_Z, wood)
    for z in (.42, .78, 1.14, 1.5, 1.86, 2.22):
        box("board", -1.75, 1.75, 3.22, ROOM_Y, z - .022, z, wood, .002)
        shelf_row(-1.72, 1.72, 3.27, z, sheets[1], r, paper, .2)
    for x in (-1.75, -.9, 0, .9, 1.72):
        box("side", x, x + .03, 3.22, ROOM_Y, 0, ROOM_Z, wood, .002)
    box("desk", -1.1, 1.1, 2.45, 3.1, .74, .78, wood, .004)
    lx = r.choice((-.75, .75))
    lathe("lamp_shade", [(.15, .0), (.09, .18)], lx, 2.85, 1.12, glow_material((1.0, .63, .33), .95))
    tube("lamp_stem", [(lx, 2.85, .78), (lx, 2.85, 1.14)], .008, surface("brass", (.6, .42, .18), .3, 1.0), 2)
    bulb((lx, 2.82, 1.16), 26, .05)
    bulb((0, 1.4, 2.5), 9, .4)
    return r.uniform(-.3, .3), 2.72, .78, 2.1, 0.0


def darkroom(r):
    """Nobody has turned a lamp on. There is a screen, and what it reaches."""
    shell((.36, .33, .30), (.10, .08, .07))
    box("bed", -1.75, -.1, 1.3, ROOM_Y, 0, .52, surface("bedding", (.38, .36, .33), .9))
    box("wardrobe", 1.0, 1.75, 2.7, ROOM_Y, 0, 2.3, surface("wardrobe", (.18, .13, .09), .5), .005)
    box("low_table", -.05, 1.0, 2.6, 3.1, .72, .75, surface("table", (.24, .16, .1), .4), .003)
    bulb((0, .6, 2.3), .5, .6)
    return r.uniform(.3, .65), 2.86, .75, 2.22, .03


PIC_KINDS = (kitchen, children, bedside, dormitory, study, darkroom)
PIC_EACH = 6


def pic_shot(sc, path, height=2.0, side=0.0):
    """From across the street, from a little to one side and a little above, as the street is: the frame is the window's opening and nothing else, a shifted lens making sure of it."""
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 480, 300, 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "8"
    wide, mid = WIN[1] - WIN[0], (WIN[2] + WIN[3]) / 2
    c = bpy.data.cameras.new("cam")
    c.lens, c.clip_start, c.clip_end = 36.0 * 30 / wide, 1.0, 100.0
    c.shift_x, c.shift_y = -side / wide, -height / wide
    o = link(bpy.data.objects.new("cam", c))
    sc.camera = o
    aim(o, (side, -30, mid + height), (side, 0, mid + height))
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def show_screen():
    """On the far side of the street the screen is simply there, words and all."""
    for o in bpy.data.objects:
        if o.name.startswith("screen"):
            m = o.data.materials[0]
            e = next(n for n in m.node_tree.nodes if n.type == "EMISSION")
            for l in list(e.inputs[1].links):
                m.node_tree.links.remove(l)
            e.inputs[1].default_value = m["words"]


def pics(samples):
    import json
    os.makedirs(PICS, exist_ok=True)
    with open(os.path.join(TEX, "manifest.json")) as f:
        info = json.load(f)
    chats = [n for n in info["pool"] if n.startswith("chat")]
    t0 = time.time()
    for name, make in (("room_a", room_a), ("room_b", room_b), ("room_c", room_c)):
        sc = reset(samples, 100)
        sc.world.node_tree.nodes["Background"].inputs[0].default_value = (.02, .016, .012, 1)
        make()
        show_screen()
        pic_shot(sc, os.path.join(PICS, name + ".png"), 2.4, 2.0)
    if opt("rooms", 0):     # the three rooms alone, when only they have changed
        return
    n = 0
    for kind in PIC_KINDS:
        for i in range(PIC_EACH):
            r = random.Random(900 + n)
            sc = reset(samples, 100)
            sc.world.node_tree.nodes["Background"].inputs[0].default_value = (.02, .016, .012, 1)
            sheets = [printed("spines_a", "spines_a.png", .55), printed("spines_b", "spines_b.png", .6)]
            dx, dy, top, seat_y, rise = kind(r, sheets) if kind in (dormitory, study) else kind(r)
            side = (-1) ** i
            mirrored(side)
            dx *= side
            spill = 9.0 if kind is darkroom else 3.0
            monitor(dx, dy, math.radians(r.uniform(-7, 7)), r.choice(chats) + ".png", .78, spill, top)
            show_screen()
            pose = r.choice(("sits", "sits", "back", "hunch", "hunch"))
            hairdo = r.choice(("short", "short", "short", "long"))
            off = r.uniform(.30, .52) * r.choice((-1, 1))
            child = kind is children
            who = (dx + off, seat_y, top - .74 + rise + r.uniform(-.015, .03), pose, math.radians(-off * 40 + r.uniform(-12, 12)), .62 if child else .84, hairdo)
            if opt("figures", 0):       # nobody, unless asked: the small dark figures read as cut-outs. The draws above are made either way, so every other room keeps its look
                figure(*who)
                if child and i % 2:         # and somebody bent over to help
                    figure(dx - off * 1.5, seat_y - .12, top - .74 + .28, "hunch", math.radians(off * 30), .84, "bun")
            pic_shot(sc, os.path.join(PICS, f"p{n:02d}.png"), r.uniform(.3, 2.2), r.uniform(-3, 3))
            print("pic", n, kind.__name__, pose, flush=True)
            n += 1
    print("pics", n, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    if ARGS[0] in ROOMS:
        shot(ARGS[0], opt("pct", 100), opt("samples", 256))
    elif ARGS[0] == "pics":
        pics(opt("samples", 48))
