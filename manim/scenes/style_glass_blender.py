"""A weather app in frosted glass, photographed in Cycles: one afternoon in Melbourne goes by in five seconds.

    Blender -b --factory-startup -P manim/scenes/style_glass_blender.py -- --frames=1,60,110,150 [--pct=50] [--samples=64]

The cards are real slabs of rough glass with bevelled edges. The sun is a real sphere behind them: sharp in the gaps,
a bloom through the frost. Every word and number is font geometry lying on the glass, so it stays crisp in 3D.
"""
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAMES = os.path.join(ROOT, "styles", "film_out", "frames")
FPS, N = 30, 150
NUM = "/System/Library/Fonts/Avenir.ttc"                    # Book, face 0. Not SFNS: a variable font's overlapping contours leave nicks in Blender's fill
HEI = "/System/Library/Fonts/Hiragino Sans GB.ttc"           # Blender opens face 0 of a collection: W3
THICK = 0.38
LIFT = THICK + 0.012
FIELD_Z = -9.0
LAND = 0.3                                                  # seconds from a card setting off to its reaching home
SUNSET = 3.47                                               # the moment the clock reads 18:42


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


def lin(c):
    c = c.lstrip("#")
    out = []
    for i in (0, 2, 4):
        v = int(c[i:i + 2], 16) / 255
        out.append(v / 12.92 if v < 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
    return tuple(out)


def link(o):
    bpy.context.collection.objects.link(o)
    return o


def seg(t, a, b):
    return min(1.0, max(0.0, (t - a) / (b - a)))


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def spring(t, damping=8.0, frequency=13.0):
    if t <= 0:
        return 0.0
    return 1 - math.exp(-damping * t) * (math.cos(frequency * t) + damping / frequency * math.sin(frequency * t))


def clock(t):
    return 15 * 60 + 222 * t / SUNSET                       # minutes past midnight: 15:00 to 18:42 at SUNSET


# ======================================================================================
# the air behind the cards
# ======================================================================================
DAY = [((-8, 6), "#2C8CFF", 5.5), ((8, 6), "#1238C8", 6.0), ((-7, -6), "#5B2BE8", 6.0), ((8, -6), "#0FA3D0", 5.0), ((0, 0), "#1E58E6", 7.0)]
DUSK = [((-8, 6), "#191349", 7.0), ((8, 6), "#3A1C86", 6.0), ((-7, -6), "#A92A92", 5.5), ((8, -6), "#FF6A2B", 5.5), ((0, 0), "#5A1FA0", 6.0)]
DAY = [(p, np.array(lin(c)), s) for p, c, s in DAY]
DUSK = [(p, np.array(lin(c)), s) for p, c, s in DUSK]
FW, FH = 640, 360
SPAN_X, SPAN_Y = 72.0, 40.5
XX, YY = np.meshgrid((np.arange(FW) + 0.5) / FW * SPAN_X - SPAN_X / 2, (np.arange(FH) + 0.5) / FH * SPAN_Y - SPAN_Y / 2)
SHADE = np.array(lin("#05061A"))
SUN_FROM, SUN_TO, SUN_Z = (0.21, 1.77), (8.4, -5.2), -3.6     # from dead behind the icon, as the first frame sees it, to below the frame


def sun_at(t):
    p = t / SUNSET
    q = max(0.0, p - 0.06) / 0.94                            # a beat behind the icon, then away: clear of the low-temperature text by the time the camera pulls back
    return (SUN_FROM[0] + (SUN_TO[0] - SUN_FROM[0]) * q ** 1.15, SUN_FROM[1] + (SUN_TO[1] - SUN_FROM[1]) * p ** 1.8, SUN_Z)


def field_pixels(t, eye, cards):
    """A mesh gradient: blobs of colour with normalised weights, sliding from afternoon to dusk, with light round the sun.
    The cards' shadows are painted in here, the farthest thing there is, so that they can never fall across the sun."""
    m = ease(seg(t, 0.7, 4.3))
    total, colour = np.zeros_like(XX), np.zeros((FH, FW, 3))
    for k, ((p, a, sa), (_, b, sb)) in enumerate(zip(DAY, DUSK)):
        cx = p[0] + 1.6 * math.sin(0.5 * t + k * 1.9)
        cy = p[1] + 1.3 * math.cos(0.42 * t + k * 2.7)
        s = sa * (1 - m) + sb * m
        w = np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * s * s))
        total += w
        colour += w[..., None] * (a * (1 - m) + b * m)
    colour /= total[..., None]
    sx, sy, sz = sun_at(t)
    reach = (eye[2] - FIELD_Z) / (eye[2] - sz)                # where the camera sees the sun against the field
    gx, gy = eye[0] + reach * (sx - eye[0]), eye[1] + reach * (sy - eye[1])
    tone = np.array(lin("#FF8A2A")) * (1 - m) + np.array(lin("#FF6A2B")) * m
    glow = np.exp(-((XX - gx) ** 2 + (YY - gy) ** 2) / (2 * (4.6 + 1.2 * m) ** 2))[..., None] * (0.12 + 0.3 * m)
    colour = colour * (1 - glow * m) + tone * glow           # by day this light adds to the blue; at dusk it takes over, or the magenta turns it salmon
    for c in cards:
        if not c.holder.hide_render:
            a = c.shade(eye)[..., None] * 0.42
            colour = colour * (1 - a) + SHADE * a
    return np.concatenate((colour, np.ones((FH, FW, 1))), axis=2).astype(np.float32)


# ======================================================================================
# materials
# ======================================================================================
def glass_material():
    """Frost: rough glass, a little white in it, stronger towards the top left like a designer's gradient; lit edges."""
    m = bpy.data.materials.new("frost")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    g = nt.nodes.new("ShaderNodeBsdfGlass")
    g.distribution = "MULTI_GGX"
    rn = nt.nodes.new("ShaderNodeTexCoord")                   # the faces are frosted; the bevel round them is polished, and bends light cleanly
    rs = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(rn.outputs["Normal"], rs.inputs[0])
    ra = nt.nodes.new("ShaderNodeMath")
    ra.operation = "ABSOLUTE"
    nt.links.new(rs.outputs["Z"], ra.inputs[0])
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["From Min"].default_value, rr.inputs["From Max"].default_value = 0.985, 0.9995
    rr.inputs["To Min"].default_value, rr.inputs["To Max"].default_value = 0.06, opt("rough", 0.34)
    nt.links.new(ra.outputs[0], rr.inputs["Value"])
    nt.links.new(rr.outputs[0], g.inputs["Roughness"])
    g.inputs["IOR"].default_value = 1.3
    g.inputs["Color"].default_value = (1, 1, 1, 1)
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs[0])
    sub = nt.nodes.new("ShaderNodeMath")
    sub.operation = "SUBTRACT"
    nt.links.new(sep.outputs["Y"], sub.inputs[0])
    nt.links.new(sep.outputs["X"], sub.inputs[1])
    scale = nt.nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY_ADD"
    nt.links.new(sub.outputs[0], scale.inputs[0])
    scale.inputs[1].default_value = 0.035
    scale.inputs[2].default_value = 0.035
    weight = nt.nodes.new("ShaderNodeLayerWeight")
    weight.inputs["Blend"].default_value = 0.12
    rim = nt.nodes.new("ShaderNodeMath")
    rim.operation = "MULTIPLY"
    nt.links.new(weight.outputs["Fresnel"], rim.inputs[0])
    rim.inputs[1].default_value = 0.5
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    nt.links.new(scale.outputs[0], add.inputs[0])
    nt.links.new(rim.outputs[0], add.inputs[1])
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1, 1, 1, 1)
    nt.links.new(add.outputs[0], em.inputs["Strength"])
    sum_ = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(g.outputs[0], sum_.inputs[0])
    nt.links.new(em.outputs[0], sum_.inputs[1])
    nt.links.new(sum_.outputs[0], out.inputs["Surface"])
    return m


UI = {}
FADES = {}                                                     # card name -> the fade nodes of everything drawn on it
CARD = [None]                                                  # the card being drawn on; None never fades


def finish(nt, shader, alpha):
    """Connect a light to the output through its opacity and, on a card that arrives late, that card's fade."""
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if alpha >= 1 and CARD[0] is None:
        nt.links.new(shader, out.inputs["Surface"])
        return
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = alpha
    if CARD[0] is not None:
        fade = nt.nodes.new("ShaderNodeValue")
        fade.outputs[0].default_value = 1.0
        FADES.setdefault(CARD[0], []).append(fade)
        amount = nt.nodes.new("ShaderNodeMath")
        amount.operation = "MULTIPLY"
        amount.inputs[0].default_value = alpha
        nt.links.new(fade.outputs[0], amount.inputs[1])
        nt.links.new(amount.outputs[0], mix.inputs[0])
    nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
    nt.links.new(shader, mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])


def ui(col="#FFFFFF", alpha=1.0, strength=0.9):
    """Flat light with an opacity: what a screen draws. Nothing else in the scene sees it."""
    key = (CARD[0], col, alpha, strength)
    if key in UI:
        return UI[key]
    m = bpy.data.materials.new(f"ui {col} {alpha}")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*lin(col), 1)
    em.inputs["Strength"].default_value = strength
    finish(nt, em.outputs[0], alpha)
    UI[key] = m
    return m


def icon(top, bottom, r):
    """An icon's fill: a soft vertical gradient, as a designer would draw it."""
    key = (CARD[0], top, bottom, r)
    if key in UI:
        return UI[key]
    m = bpy.data.materials.new("icon")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    span = nt.nodes.new("ShaderNodeMapRange")
    span.inputs["From Min"].default_value, span.inputs["From Max"].default_value = -r, r
    nt.links.new(sep.outputs["Y"], span.inputs["Value"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.name = "fill"
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (*lin(bottom), 1)
    mix.inputs["B"].default_value = (*lin(top), 1)
    nt.links.new(span.outputs[0], mix.inputs["Factor"])
    em = nt.nodes.new("ShaderNodeEmission")
    em.name = "glow"
    em.inputs["Strength"].default_value = 0.96
    nt.links.new(mix.outputs["Result"], em.inputs["Color"])
    finish(nt, em.outputs[0], 1.0)
    UI[key] = m
    return m


def flat(o):
    o.visible_diffuse = o.visible_glossy = o.visible_transmission = o.visible_shadow = False
    return o


# ======================================================================================
# geometry
# ======================================================================================
def outline(w, h, r, d, n=14):
    """A rounded rectangle, inset by d."""
    w, h = w / 2 - d, h / 2 - d
    r = max(r - d, 0.012)
    pts = []
    for cx, cy, a0 in ((w - r, h - r, 0), (-w + r, h - r, 90), (-w + r, -h + r, 180), (w - r, -h + r, 270)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def slab(name, w, h, r, parent, mat):
    """A slab of glass with rounded corners and a soft bevel all round."""
    b, steps = 0.11, 7
    rings = [(b * (1 - math.sin(math.radians(90 * i / steps))), b * (1 - math.cos(math.radians(90 * i / steps)))) for i in range(steps + 1)]
    rings += [(b * (1 - math.sin(math.radians(90 * i / steps))), THICK - b * (1 - math.cos(math.radians(90 * i / steps)))) for i in range(steps, -1, -1)]
    verts, faces = [], []
    n = len(outline(w, h, r, 0))
    for d, z in rings:
        verts += [(x, y, z) for x, y in outline(w, h, r, d)]
    for k in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            faces.append((k * n + i, (k + 1) * n + i, (k + 1) * n + j, k * n + j))
    side = len(faces)
    faces.append(tuple(range(n - 1, -1, -1)))
    last = (len(rings) - 1) * n
    faces.append(tuple(last + i for i in range(n)))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.polygons.foreach_set("use_smooth", [True] * side + [False] * 2)
    me.update()
    uv = me.uv_layers.new(name="UVMap")
    for loop in me.loops:
        v = verts[loop.vertex_index]
        uv.data[loop.index].uv = ((v[0] + w / 2) / w, (v[1] + h / 2) / h)
    me.materials.append(mat)
    o = link(bpy.data.objects.new(name, me))
    o.parent = parent
    return o


def text(parent, s, x, y, size, font=HEI, align="LEFT", mat=None, thin=0.0, z=LIFT):
    cu = bpy.data.curves.new("t", "FONT")
    cu.body = s
    cu.font = bpy.data.fonts.load(font, check_existing=True)
    cu.size = size
    cu.align_x = align
    cu.resolution_u = 14
    cu.offset = -thin
    cu.materials.append(mat or ui())
    o = flat(link(bpy.data.objects.new("t", cu)))
    o.parent = parent
    o.location = (x, y, z)
    return o


def shape(parent, polys, x, y, mat, z=LIFT):
    verts, faces = [], []
    for loop in polys:
        k = len(verts)
        verts += [(a, b, 0) for a, b in loop]
        faces.append(tuple(range(k, k + len(loop))))
    me = bpy.data.meshes.new("shape")
    me.from_pydata(verts, [], faces)
    me.update()
    me.materials.append(mat)
    o = flat(link(bpy.data.objects.new("shape", me)))
    o.parent = parent
    o.location = (x, y, z)
    return o


def circle(cx, cy, r, n=48):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def bar(x0, y0, x1, y1, w):
    """A stroke with round ends."""
    a = math.atan2(y1 - y0, x1 - x0)
    pts = []
    for cx, cy, a0 in ((x1, y1, a - math.pi / 2), (x0, y0, a + math.pi / 2)):
        pts += [(cx + w / 2 * math.cos(a0 + math.pi * i / 8), cy + w / 2 * math.sin(a0 + math.pi * i / 8)) for i in range(9)]
    return pts


def sun_shapes(r):
    polys = [circle(0, 0, r)]
    for k in range(8):
        a = math.pi / 4 * k
        polys.append(bar(1.38 * r * math.cos(a), 1.38 * r * math.sin(a), 1.78 * r * math.cos(a), 1.78 * r * math.sin(a), 0.2 * r))
    return polys


def cloud_shapes(r):
    return [circle(-0.72 * r, -0.12 * r, 0.56 * r), circle(0.02 * r, 0.22 * r, 0.72 * r), circle(0.74 * r, -0.12 * r, 0.56 * r),
            [(-0.72 * r, -0.68 * r), (0.74 * r, -0.68 * r), (0.74 * r, 0.1 * r), (-0.72 * r, 0.1 * r)]]


def moon_shape(r):
    """A crescent, horns to the right: the big circle's rim, then back along a shallower arc through both horns."""
    deg = 52
    outer = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for a in np.linspace(deg, 360 - deg, 40)]
    hx, hy = outer[0]
    cx = hx - 0.42 * r                                  # the bite: a circle through both horns, centred on the axis
    rr = math.hypot(hx - cx, hy)
    start = math.degrees(math.atan2(-hy, hx - cx)) % 360
    end = math.degrees(math.atan2(hy, hx - cx)) % 360
    inner = [(cx + rr * math.cos(math.radians(a)), rr * math.sin(math.radians(a))) for a in np.linspace(start, end, 30)[1:-1]]
    return outer + inner


def annulus(cx, cy, r0, r1, a0, a1, n=72):
    """A thick arc from angle a0 to a1 in degrees, with round caps."""
    steps = max(2, int(abs(a1 - a0) / 360 * n))
    outer = [(cx + r1 * math.cos(math.radians(a0 + (a1 - a0) * i / steps)), cy + r1 * math.sin(math.radians(a0 + (a1 - a0) * i / steps))) for i in range(steps + 1)]
    inner = [(cx + r0 * math.cos(math.radians(a0 + (a1 - a0) * i / steps)), cy + r0 * math.sin(math.radians(a0 + (a1 - a0) * i / steps))) for i in range(steps, -1, -1)]
    return outer + inner


def reshape(o, polys):
    me = o.data
    me.clear_geometry()
    verts, faces = [], []
    for loop in polys:
        k = len(verts)
        verts += [(a, b, 0) for a, b in loop]
        faces.append(tuple(range(k, k + len(loop))))
    me.from_pydata(verts, [], faces)
    me.update()


# ======================================================================================
# the sky
# ======================================================================================
def sphere(name, r, loc, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=r, location=loc)
    o = bpy.context.object
    o.name = name
    bpy.ops.object.shade_smooth()
    o.data.materials.append(mat)
    return o


def sky_body(name, strength, centre=1.3, limb=1.0, maria=False):
    """A disc of light, a little hotter at the centre. Its colour is set per frame; a sun also has a white-hot core, a moon a fade."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    weight = nt.nodes.new("ShaderNodeLayerWeight")
    weight.inputs["Blend"].default_value = 0.5
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    r = ramp.color_ramp
    r.elements[0].position, r.elements[0].color = 0.0, (centre, centre, centre, 1)
    r.elements[1].position, r.elements[1].color = 1.0, (limb, limb, limb, 1)
    nt.links.new(weight.outputs["Facing"], ramp.inputs["Fac"])
    tint = nt.nodes.new("ShaderNodeRGB")
    tint.name = "tint"
    mul = nt.nodes.new("ShaderNodeMix")
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    nt.links.new(ramp.outputs["Color"], mul.inputs["A"])
    nt.links.new(tint.outputs[0], mul.inputs["B"])
    if maria:                                                # a moon has seas on it: broad and faint, never a noisy surface
        tc = nt.nodes.new("ShaderNodeTexCoord")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 1.15
        noise.inputs["Detail"].default_value = 1.5
        nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
        seas = nt.nodes.new("ShaderNodeMapRange")
        seas.inputs["From Min"].default_value, seas.inputs["From Max"].default_value = 0.4, 0.6
        seas.inputs["To Min"].default_value, seas.inputs["To Max"].default_value = 0.86, 1.0
        nt.links.new(noise.outputs["Fac"], seas.inputs["Value"])
        mul2 = nt.nodes.new("ShaderNodeMix")
        mul2.data_type = "RGBA"
        mul2.blend_type = "MULTIPLY"
        mul2.inputs["Factor"].default_value = 1.0
        nt.links.new(mul.outputs["Result"], mul2.inputs["A"])
        nt.links.new(seas.outputs[0], mul2.inputs["B"])
        mul = mul2
    else:                                                    # hotter at the heart: white while the sun is high, gold as it sets
        core = nt.nodes.new("ShaderNodeValue")
        core.name = "core"
        falloff = nt.nodes.new("ShaderNodeMath")
        falloff.operation = "MULTIPLY_ADD"
        falloff.use_clamp = True
        nt.links.new(weight.outputs["Facing"], falloff.inputs[0])
        falloff.inputs[1].default_value = -4.0
        falloff.inputs[2].default_value = 1.0
        amount = nt.nodes.new("ShaderNodeMath")
        amount.operation = "MULTIPLY"
        nt.links.new(falloff.outputs[0], amount.inputs[0])
        nt.links.new(core.outputs[0], amount.inputs[1])
        add = nt.nodes.new("ShaderNodeMix")
        add.data_type = "RGBA"
        add.blend_type = "ADD"
        nt.links.new(amount.outputs[0], add.inputs["Factor"])
        nt.links.new(mul.outputs["Result"], add.inputs["A"])
        add.name = "heart"
        mul = add
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = strength
    nt.links.new(mul.outputs["Result"], em.inputs["Color"])
    if maria:                                                # the moon fades in through clear air, never as a dark disc
        fade = nt.nodes.new("ShaderNodeValue")
        fade.name = "fade"
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(fade.outputs[0], mix.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    else:                                                    # its limb melts into the halo behind it: a glowing body, not a plastic ball
        soft = nt.nodes.new("ShaderNodeMapRange")
        soft.interpolation_type = "SMOOTHSTEP"
        nt.links.new(weight.outputs["Facing"], soft.inputs["Value"])
        soft.inputs["From Min"].default_value, soft.inputs["From Max"].default_value = 0.5, 0.97
        soft.inputs["To Min"].default_value, soft.inputs["To Max"].default_value = 1.0, 0.0
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(soft.outputs[0], mix.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m, tint


def halo(name, z, power=3.0):
    """A bloom round a body of light: a disc of added light falling to exactly nothing at its rim. Scale it to size per frame."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
    me.update()
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    length = nt.nodes.new("ShaderNodeVectorMath")
    length.operation = "DOT_PRODUCT"
    nt.links.new(tc.outputs["Object"], length.inputs[0])
    nt.links.new(tc.outputs["Object"], length.inputs[1])
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.use_clamp = True
    inv.inputs[0].default_value = 1.0
    nt.links.new(length.outputs["Value"], inv.inputs[1])
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    nt.links.new(inv.outputs[0], pw.inputs[0])
    pw.inputs[1].default_value = power
    amp = nt.nodes.new("ShaderNodeValue")
    amp.name = "amp"
    times = nt.nodes.new("ShaderNodeMath")
    times.operation = "MULTIPLY"
    nt.links.new(pw.outputs[0], times.inputs[0])
    nt.links.new(amp.outputs[0], times.inputs[1])
    tint = nt.nodes.new("ShaderNodeRGB")
    tint.name = "tint"
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(tint.outputs[0], em.inputs["Color"])
    nt.links.new(times.outputs[0], em.inputs["Strength"])
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    me.materials.append(m)
    o = link(bpy.data.objects.new(name, me))
    o.location.z = z
    o.visible_diffuse = o.visible_glossy = o.visible_shadow = False
    return o, tint.outputs[0], amp.outputs[0]


def softbox(name, w, h, loc, strength):
    me = bpy.data.meshes.new(name)
    me.from_pydata([(-w / 2, -h / 2, 0), (w / 2, -h / 2, 0), (w / 2, h / 2, 0), (-w / 2, h / 2, 0)], [], [(0, 1, 2, 3)])
    me.update()
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    me.materials.append(m)
    o = link(bpy.data.objects.new(name, me))
    o.location = loc
    o.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat("Z", "Y").to_euler()
    o.visible_camera = o.visible_transmission = o.visible_diffuse = o.visible_shadow = False
    return o


# ======================================================================================
# the dashboard
# ======================================================================================
ERF_X = np.linspace(-4, 4, 1601)
ERF_Y = np.array([math.erf(v) for v in ERF_X])


def blurred_edge(u, half, sigma):
    """One axis of a rectangle blurred by a gaussian."""
    k = sigma * 1.41421
    return (np.interp((u + half) / k, ERF_X, ERF_Y) - np.interp((u - half) / k, ERF_X, ERF_Y)) / 2


class Card:
    def __init__(self, dash, name, cx, cy, w, h, z, mat, enter=None, delay=0.0):
        self.name = name
        self.holder = bpy.data.objects.new(name, None)
        link(self.holder).parent = dash
        self.home = Vector((cx, cy, z))
        self.enter, self.delay = Vector(enter or (0, 0, 0)), delay
        self.w, self.h = w, h
        self.glass = slab(name + " glass", w, h, 0.62 if h > 3.5 else 0.5, self.holder, mat)

    def at(self, t):
        if self.delay:                                         # hardly any overshoot: a card must not swing over its neighbour
            k = spring(t - self.delay, 9.0, 8.0)
        else:
            k = spring(t, 7.5, 11.5)
        self.holder.location = self.home + self.enter * (1 - k)
        swing = 0.3 if self.delay == 0 else 1.0                # the first card is already nearly settled: it has to be read at once
        self.holder.rotation_euler = (0.5 * swing * (1 - k), -0.32 * swing * (1 - k), 0)
        # hidden until it has somewhere to be, so its glass can't catch the sun off screen
        gone = t < self.delay - 0.02
        self.holder.hide_render = gone
        # a late card's words arrive once its glass has landed: nothing gets sliced by an edge on the way in
        fade = ease(seg(t, self.delay + LAND, self.delay + LAND + 0.15)) if self.delay else 1.0
        for node in FADES.get(self.name, ()):
            node.outputs[0].default_value = fade
        for c in self.holder.children:
            c.hide_render = gone or (fade <= 0 and c != self.glass)

    def shade(self, eye):
        """How much of the field this card shades: a blurred rectangle a little behind and below it, along the camera's rays."""
        inv = np.array(self.holder.matrix_world.inverted())
        e = inv[:3, :3] @ np.array(eye) + inv[:3, 3]
        p = XX[..., None] * inv[:3, 0] + YY[..., None] * inv[:3, 1] + FIELD_Z * inv[:3, 2] + inv[:3, 3]
        s = (-0.8 - e[2]) / (p[..., 2] - e[2])
        u = e[0] + s * (p[..., 0] - e[0])
        v = e[1] + s * (p[..., 1] - e[1]) + 0.28
        return blurred_edge(u, self.w / 2, 0.55) * blurred_edge(v, self.h / 2, 0.55)


def build(dash):
    mat = glass_material()
    main = Card(dash, "main", -3.2, 1.7, 10.2, 5.2, 0.35, mat)
    hourly = Card(dash, "hourly", -3.2, -2.75, 10.2, 3.1, 0.0, mat, (0, -8, -1.6), 0.95)
    sunset = Card(dash, "sunset", 5.25, 2.25, 6.1, 4.1, 0.15, mat, (11, 1.5, -1.6), 1.2)
    air = Card(dash, "air", 3.65, -2.2, 2.9, 4.2, 0.0, mat, (5, -10, -1.6), 1.45)
    wind = Card(dash, "wind", 6.85, -2.2, 2.9, 4.2, 0.1, mat, (9, -9, -1.6), 1.65)
    w = {}

    # main
    CARD[0] = None
    h = main.holder
    text(h, "墨尔本", -4.5, 1.56, 0.74)
    w["clock"] = text(h, "15:00", 4.5, 1.6, 0.62, NUM, "RIGHT", ui(alpha=0.95))
    w["temp"] = text(h, "24°", -4.72, -1.28, 3.7, NUM)
    w["mood"] = text(h, "晴   体感 25°", -4.5, -2.14, 0.5)
    text(h, "最高 27°   最低 14°", 4.5, -2.14, 0.5, align="RIGHT")
    w["bigsun"] = shape(h, sun_shapes(0.66), 2.95, 0.12, icon("#FFE88A", "#FF9F2E", 1.2))
    w["bigmoon"] = shape(h, [moon_shape(0.86)], 2.95, 0.12, icon("#FFFFFF", "#B9C8FF", 0.9))

    # hourly
    CARD[0] = "hourly"
    h = hourly.holder
    xs = [-4.2 + i * 1.68 for i in range(6)]
    w["xs"] = xs
    w["pill"] = shape(h, [outline(1.44, 2.72, 0.42, 0)], xs[0], 0, ui(alpha=0.17), z=LIFT - 0.004)
    icons = ("sun", "sun", "partly", "partly", "moon", "moon")
    temps = (24, 23, 22, 21, 20, 19)
    for i, x in enumerate(xs):
        text(h, f"{15 + i}时", x, 0.82, 0.42, align="CENTER")
        y = 0.02
        if icons[i] == "sun":
            shape(h, sun_shapes(0.21), x, y, icon("#FFE88A", "#FF9F2E", 0.38))
        elif icons[i] == "moon":
            shape(h, [moon_shape(0.3)], x, y, icon("#FFFFFF", "#B9C8FF", 0.32))
        else:
            shape(h, sun_shapes(0.15), x + 0.16, y + 0.17, icon("#FFE88A", "#FF9F2E", 0.28))
            shape(h, cloud_shapes(0.3), x - 0.05, y - 0.06, ui(), z=LIFT + 0.004)
        text(h, f"{temps[i]}°", x + 0.05, -1.14, 0.5, NUM, "CENTER")

    # sunset
    CARD[0] = "sunset"
    h = sunset.holder
    text(h, "日落", -2.5, 1.26, 0.5)
    text(h, "日出 06:48", 2.5, 1.26, 0.42, align="RIGHT", mat=ui(alpha=0.85))
    text(h, "18:42", -2.53, 0.3, 1.02, NUM)
    w["arc_cy"], w["arc_rx"], w["arc_ry"] = -1.3, 2.42, 1.42
    shape(h, [bar(-2.65, -1.3, 2.65, -1.3, 0.022)], 0, 0, ui(alpha=0.55))
    w["arc_dim"] = shape(h, [[(0, 0), (1, 0), (0, 1)]], 0, 0, ui(alpha=0.3))
    w["arc_lit"] = shape(h, [[(0, 0), (1, 0), (0, 1)]], 0, 0, ui("#FFD45E"), z=LIFT + 0.002)
    w["dot_halo"] = shape(h, [circle(0, 0, 0.27)], 0, 0, ui("#FFE7A0", alpha=0.3), z=LIFT + 0.004)
    w["dot"] = shape(h, [circle(0, 0, 0.14)], 0, 0, ui("#FFF3C4"), z=LIFT + 0.006)

    # air quality
    CARD[0] = "air"
    h = air.holder
    text(h, "空气质量", -1.14, 1.36, 0.46)
    shape(h, [annulus(0, -0.14, 0.8, 0.94, 225, -45)], 0, 0, ui(alpha=0.2))
    w["aqi_ring"] = shape(h, [[(0, 0), (1, 0), (0, 1)]], 0, 0, ui("#7CFFB2"), z=LIFT + 0.002)
    w["aqi"] = text(h, "28", 0, -0.36, 0.68, NUM, "CENTER")
    text(h, "优", 0, -1.72, 0.46, align="CENTER", mat=ui("#7CFFB2"))

    # wind
    CARD[0] = "wind"
    h = wind.holder
    text(h, "风速", -1.14, 1.36, 0.46)
    ticks = []
    for k in range(36):
        a = math.radians(10 * k)
        r0 = 0.82 if k % 9 == 0 else 0.87
        ticks.append(bar(r0 * math.cos(a), -0.14 + r0 * math.sin(a), 0.94 * math.cos(a), -0.14 + 0.94 * math.sin(a), 0.024))
    shape(h, ticks, 0, 0, ui(alpha=0.6))
    arrow = [(0.0, 0.72), (0.17, 0.42), (0.035, 0.45), (0.035, -0.62), (-0.035, -0.62), (-0.035, 0.45), (-0.17, 0.42)]
    w["arrow"] = shape(h, [arrow], 0, -0.14, ui("#7FE3FF"))
    w["wind"] = text(h, "12 km/h", 0, -1.72, 0.42, NUM, "CENTER")
    w["cards"] = (main, hourly, sunset, air, wind)
    return w


def arc_point(w, deg):
    a = math.radians(deg)
    return w["arc_rx"] * math.cos(a), w["arc_cy"] + w["arc_ry"] * math.sin(a)


def arc_polys(w, a0, a1):
    n = max(2, int(abs(a1 - a0) / 3))
    pts = [arc_point(w, a0 + (a1 - a0) * i / n) for i in range(n + 1)]
    return [bar(*pts[i], *pts[i + 1], 0.04) for i in range(n)]


def update(w, t):
    for c in w["cards"]:
        c.at(t)
    minutes = clock(t)
    w["clock"].data.body = f"{int(minutes // 60):02d}:{int(minutes % 60):02d}"
    w["temp"].data.body = f"{24 - min(5, int(max(0, t - 0.7) / 0.62))}°"
    w["mood"].data.body = "晴夜   体感 18°" if t > SUNSET + 0.12 else "晴   体感 25°"
    gone = spring(t - SUNSET, 9, 13)
    w["bigsun"].scale = [max(0.001, 1 - gone)] * 3
    w["bigsun"].hide_render = gone > 0.985
    w["bigsun"].rotation_euler = (0, 0, -0.5 * t)
    heat = 1 - ease(seg(t, 0.5, 1.7))                        # lit from behind by the real sun it is white-hot, and cools as that slides away
    nodes = w["bigsun"].active_material.node_tree.nodes
    nodes["glow"].inputs["Strength"].default_value = 0.96 + 0.9 * heat
    fill = nodes["fill"]
    fill.inputs["A"].default_value = (*(np.array(lin("#FF9F2E")) * (1 - heat) + np.array(lin("#FFE9A0")) * heat), 1)
    fill.inputs["B"].default_value = (*(np.array(lin("#FFE88A")) * (1 - heat) + np.array(lin("#FFFDF0")) * heat), 1)
    rise = spring(t - SUNSET - 0.1, 8, 13)
    w["bigmoon"].scale = [max(0.001, rise)] * 3
    w["bigmoon"].hide_render = rise < 0.015
    hour = min(5.0, (minutes - 900) / 60)
    k = int(hour)
    x0, x1 = w["xs"][k], w["xs"][min(5, k + 1)]
    w["pill"].location.x = x0 + (x1 - x0) * ease(seg(hour - k, 0.72, 1.0))
    day = (minutes - 408) / (1122 - 408)                     # 06:48 to 18:42
    deg = 180 * (1 - day)
    reshape(w["arc_dim"], arc_polys(w, 0, 180))
    reshape(w["arc_lit"], arc_polys(w, 180, max(deg, 0)))
    px, py = arc_point(w, deg)
    py = py if deg >= 0 else w["arc_cy"] + deg * 0.016
    px = px if deg >= 0 else w["arc_rx"] + (-deg) * 0.002
    w["dot"].location.x, w["dot"].location.y = px, py
    w["dot_halo"].location.x, w["dot_halo"].location.y = px, py
    aqi = 28 + 13 * ease(seg(t, 1.5, 4.8))
    w["aqi"].data.body = f"{int(aqi)}"
    fill = spring(t - 1.6, 5, 9) * aqi / 100 * 270
    reshape(w["aqi_ring"], [annulus(0, -0.14, 0.8, 0.94, 225, 225 - max(1.0, fill))])
    w["arrow"].rotation_euler = (0, 0, math.radians(215) * spring(t - 1.8, 3.2, 6) + 0.12 * math.sin(2.4 * t))
    w["wind"].data.body = f"{12 + int(4 * ease(seg(t, 2.2, 4.8)))} km/h"


def main():
    sc = bpy.context.scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    sc.cycles.device = "GPU"
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.015
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.max_bounces = 6
    sc.cycles.transmission_bounces = 4
    sc.cycles.glossy_bounces = 3
    sc.cycles.diffuse_bounces = 1
    sc.cycles.transparent_max_bounces = 6
    sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
    sc.cycles.sample_clamp_indirect = 5
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.dither_intensity = 1.0
    sc.render.fps = FPS
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    world = bpy.data.worlds.new("void")
    world.use_nodes = True
    sky = world.node_tree.nodes["Background"].inputs["Color"]     # never seen directly: what the polished bevels bend towards, so they read as light, not a black keyline
    sc.world = world

    img = bpy.data.images.new("field", FW, FH, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    mat = bpy.data.materials.new("field")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image, tex.interpolation, tex.extension = img, "Cubic", "EXTEND"
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    me = bpy.data.meshes.new("field")
    me.from_pydata([(-SPAN_X / 2, -SPAN_Y / 2, 0), (SPAN_X / 2, -SPAN_Y / 2, 0), (SPAN_X / 2, SPAN_Y / 2, 0), (-SPAN_X / 2, SPAN_Y / 2, 0)], [], [(0, 1, 2, 3)])
    me.update()
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", (0, 0, 1, 0, 1, 1, 0, 1))
    me.materials.append(mat)
    link(bpy.data.objects.new("field", me)).location = (0, 0, FIELD_Z)

    sun_mat, sun_tint = sky_body("sun", 1.12)
    sun = sphere("sun", 2.3, sun_at(0), sun_mat)
    sun_core = sun_mat.node_tree.nodes["core"].outputs[0]
    sun_heart = sun_mat.node_tree.nodes["heart"].inputs["B"]
    sun_halo, sun_halo_tint, sun_halo_amp = halo("sun halo", SUN_Z)
    moon_mat, moon_tint = sky_body("moon", 0.95, centre=1.04, limb=0.92, maria=True)
    moon = sphere("moon", 0.62, (-2.0, 2.6, -3.0), moon_mat)
    moon_tint.outputs[0].default_value = (*lin("#EEF2FF"), 1)
    moon_fade = moon_mat.node_tree.nodes["fade"].outputs[0]
    moon_halo, moon_halo_tint, moon_halo_amp = halo("moon halo", -3.0)
    moon_halo_tint.default_value = (*lin("#BFCDFF"), 1)
    moon_halo.scale = (2.0, 2.0, 1)
    # only the open sky shows the moon: never a grey disc in the frost, nor a streak along a card's top edge
    moon.visible_transmission = moon.visible_glossy = moon.visible_diffuse = moon.visible_shadow = False
    moon_halo.visible_transmission = False

    softbox("top bar", 34, 1.6, (-2, 15, 16), 6.0)
    softbox("left bar", 1.6, 22, (-20, 2, 15), 2.5)
    softbox("low bar", 26, 1.0, (4, -15, 12), 1.5)

    dash = link(bpy.data.objects.new("dash", None))
    dash.rotation_mode = "XYZ"
    w = build(dash)

    cd = bpy.data.cameras.new("cam")
    cd.lens, cd.sensor_width = 85, 36
    cd.clip_start, cd.clip_end = 1, 300
    cam = link(bpy.data.objects.new("cam", cd))
    sc.camera = cam

    test = [int(f) for f in opt("frames", "").split(",") if f]
    if test:
        sc.render.resolution_percentage = opt("pct", 40)
        sc.cycles.samples = opt("samples", 64)
        out = os.path.join(FRAMES, "test")
    else:
        sc.render.resolution_percentage = 100
        sc.cycles.samples = opt("samples", 72)
        out = os.path.join(FRAMES, "glass")
    os.makedirs(out, exist_ok=True)
    for f in test or range(1, N + 1):
        path = os.path.join(out, f"glass_f{f:04d}.png" if test else f"f{f:04d}.png")
        if not test and os.path.exists(path):
            continue
        t0 = time.time()
        t = (f - 1) / FPS
        p = t / SUNSET
        sun.location = sun_at(t)
        size = 0.72 + 0.28 * ease(seg(p, 0.5, 1.0))                         # a low sun looks bigger
        sun.scale = (size, size, size)
        sun_halo.location.x, sun_halo.location.y = sun.location.x, sun.location.y
        sun_halo.scale = (2.6 * 2.3 * size, 2.6 * 2.3 * size, 1)
        warm = ease(seg(p, 0.15, 0.75))
        a, b = np.array(lin("#FFC23A")), np.array(lin("#FF6A1E"))
        colour = a * (1 - warm) + b * warm
        sun_tint.outputs[0].default_value = (*colour, 1)
        sun_core.default_value = 0.16
        sun_heart.default_value = (*(np.array(lin("#FFE6B0")) * (1 - warm) + np.array(lin("#FFB030")) * warm), 1)
        ember = np.array(lin("#FF8A2A")) * (1 - warm) + b * warm            # the bloom runs deeper than the disc: gold, orange, orchid, then the sky
        sun_halo_tint.default_value = (*ember, 1)
        sun_halo_amp.default_value = 0.6
        moon.location = (-2.0, 2.6 + 2.23 * ease(seg(t, 2.8, 5.0)), -3.0)    # it rises as the sun goes down, and stops clear of the frame
        up = ease(seg(t, 3.3, 3.8))                                         # only once the app itself says it is night
        moon_fade.default_value = up
        moon.hide_render = moon_halo.hide_render = up < 0.01
        dusk = ease(seg(t, 0.7, 4.3))
        sky.default_value = (*(np.array(lin("#7C9BFF")) * (1 - dusk) + np.array(lin("#D0608F")) * dusk) * 0.8, 1)
        moon_halo.location.x, moon_halo.location.y = moon.location.x, moon.location.y
        moon_halo_amp.default_value = 0.1 * up
        pull = ease(seg(t, 0.75, 2.9))
        dist = 25.5 * (47.5 / 25.5) ** pull * (1 - 0.02 * seg(t, 2.9, 5.0))
        cam.location = (-3.3 * (1 - pull), 1.72 * (1 - pull), dist)
        dash.rotation_euler = (math.radians(4 * (1 - pull) - 13 * ease(seg(t, 1.2, 5.0))), math.radians(-5 * (1 - pull) + 15 * ease(seg(t, 1.2, 5.0))), 0)
        update(w, t)
        sc.frame_set(f)
        bpy.context.view_layer.update()
        img.pixels.foreach_set(field_pixels(t, tuple(cam.location), w["cards"]).ravel())
        img.update()
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f"glass f{f:04d}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
