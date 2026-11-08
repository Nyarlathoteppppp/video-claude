"""An ordinary room in the afternoon: a wooden table under a window, a Studio Display and its keyboard on it, a cup, a pile of books. The film comes back here whenever it is simply about somebody and a computer, so the set is built here once and shots only choose a camera.

Inside Blender, from manim/scenes:

    Blender -b --factory-startup -P film_room3d.py -- --shot=wide --passes=sky,sun,glow,mask [--pct=50] [--samples=128]

Everything is rendered in linear light and kept apart, so that film_room.py can balance it without coming back here:
    sky     daylight from the window, no sun; the screen dark
    sun     the sun alone
    glow    the screen's light alone, on the room
    mask    how much of each pixel is the screen, after focus
Each goes to ink/film_out/film/room/plates/SHOT_PASS.npy, float16, top row first. SHOT.json has the camera and where the screen's corners fall in the frame.
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from mathutils import Matrix, Vector

from film_room_tex import BOOKS, CASE, KEYS_H, KEYS_W, VOLUME, keys
from ink3d import ROOT, mesh_grid, reset

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


SHOT = opt("shot", "wide")
PASSES = opt("passes", "sky,sun,glow,mask").split(",")
PCT, SAMPLES = opt("pct", 100), opt("samples", 128)
POSE = opt("pose", 0)                # how far the air has moved the curtain, in steps of POSE_S seconds
OUT = os.path.join(ROOT, "ink", "film_out", "film", "room")
TEX = os.path.join(OUT, "tex")
PLATES = os.path.join(OUT, "plates")

# the room, in metres: x to the right as you face the window wall, y towards it, z up
WALL_Y = 0.80
LEFT_X, RIGHT_X, BACK_Y, CEIL = -2.0, 1.9, -2.6, 2.62
WIN = (-1.28, -0.20, 0.98, 2.28)    # the window in that wall: left, right, sill, head
ZT = 0.74                           # the table's top
TABLE = (-0.82, 0.92, -0.045, 0.695)
DISP_YAW = 0.0

# Studio Display with the tilt stand, from apple.com/studio-display/specs and its side and back elevations: 62.3 wide, 47.8 high, 16.8 deep on the stand, the case 36.2 high and about 1.9 thick (the 3.1 on the spec sheet is the VESA version, adapter included)
DISP_X, DISP_Y = 0.03, 0.40         # the middle of its width, and the plane of the glass
DISP_W, DISP_H, DISP_T = 0.623, 0.362, 0.0185
DISP_UP = 0.116                     # the case's lower edge above the table
DISP_R = 0.007
STAND_W = 0.152
SCREEN_W, SCREEN_H = 0.5965, 0.3356   # 5120 by 2880 at 218 pixels an inch, the black round it about 1.3 cm on every side

POSE_S = 1.0

SHOTS = dict(
    # nearly square on and level, the lens shifted rather than turned: what stands parallel to the wall stays square in the frame. centre: the point in the plane of the glass that falls in the middle of the frame, forward: how far in front of the glass the focus lies, for the keyboard
    wide=dict(loc=(-0.30, -2.20, 1.28), centre=(0.12, 0.86), lens=46.0, fstop=2.8, forward=0.08),
    # the screen, square on: only to see what its glass reflects. at: the point of the screen in the middle of the frame, as parts of its width and height, width: how much of its width the frame takes in
    close=dict(at=(0.5, 0.68), width=0.82, lens=50.0, fstop=8.0),
    # corners of the wide shot from the same place through a longer lens, with the same blur (the books' focused on their backs instead of the keyboard): to look at them closely, not in the film
    zoom_studio=dict(loc=(-0.30, -2.20, 1.28), centre=(0.692, 0.796), lens=138.0, fstop=8.4, forward=0.08),
    zoom_desk=dict(loc=(-0.30, -2.20, 1.28), centre=(0.12, 0.6745), lens=138.0, fstop=8.4, forward=0.08),
    zoom_books=dict(loc=(-0.30, -2.20, 1.28), centre=(-0.5526, 0.8717), lens=184.0, fstop=11.2, forward=-0.06),
)


# ---------------------------------------------------------------- materials

def tex(name, colorspace="sRGB"):
    im = bpy.data.images.load(os.path.join(TEX, name), check_existing=True)
    im.colorspace_settings.name = colorspace
    return im


def surface(name, colour, rough=.5, metal=0.0, spec=.5, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*colour, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Specular IOR Level"].default_value = spec
    b.inputs["Coat Weight"].default_value = coat
    return m


def noisy(m, scale, strength, distance=.002, detail=8):
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


def printed(name, picture, rough=.6, spec=.4, metal=0.0):
    m = surface(name, (1, 1, 1), rough, metal, spec)
    nt = m.node_tree
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = tex(picture)
    t.interpolation = "Cubic"
    nt.links.new(nt.nodes.new("ShaderNodeUVMap").outputs[0], t.inputs[0])
    nt.links.new(t.outputs[0], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return m


def table_material():
    m = printed("table", "table.png")
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    uv = nt.nodes.new("ShaderNodeUVMap")
    r, h, bp = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeBump")
    r.image, h.image = tex("table_rough.png", "Non-Color"), tex("table_bump.png", "Non-Color")
    nt.links.new(uv.outputs[0], r.inputs[0])
    nt.links.new(uv.outputs[0], h.inputs[0])
    nt.links.new(r.outputs[0], b.inputs["Roughness"])
    bp.inputs["Strength"].default_value = .35
    bp.inputs["Distance"].default_value = .0012
    nt.links.new(h.outputs[0], bp.inputs["Height"])
    nt.links.new(bp.outputs[0], b.inputs["Normal"])
    return m


def table_wood(grain, across, at, tint=1.0):
    """The same wood where the top's picture would only streak: grain and across are the axes it runs along and over ("XY" turns corners), at where in the picture to begin, in its upper half that has no rings."""
    m = table_material()
    nt = m.node_tree
    pos = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"], pos.inputs[0])

    def run(axes, size, start):
        s = pos.outputs[axes[0]]
        for a in axes[1:]:
            s = calc(nt, "ADD", s, pos.outputs[a])
        return calc(nt, "ADD", calc(nt, "DIVIDE", s, size), start)

    uv = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(run(grain, TABLE[1] - TABLE[0], at[0]), uv.inputs[0])
    nt.links.new(run(across, TABLE[3] - TABLE[2], at[1]), uv.inputs[1])
    for t in [n for n in nt.nodes if n.bl_idname == "ShaderNodeTexImage"]:
        nt.links.new(uv.outputs[0], t.inputs[0])
    b = nt.nodes["Principled BSDF"]
    if tint != 1.0:
        nt.links.new(blended(nt, 1.0, b.inputs["Base Color"].links[0].from_socket, (tint, tint, tint), "MULTIPLY"), b.inputs["Base Color"])
    return m


def plaster():
    m = surface("plaster", (.82, .80, .75), .9, 0, .25)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc, n, ramp, bp, n2 = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeValToRGB"), nt.nodes.new("ShaderNodeBump"), nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = 1.6
    n.inputs["Detail"].default_value = 6
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = .3, .7
    ramp.color_ramp.elements[0].color = (.79, .765, .715, 1)
    ramp.color_ramp.elements[1].color = (.84, .815, .77, 1)
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
    nt.links.new(n.outputs[0], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], b.inputs["Base Color"])
    n2.inputs["Scale"].default_value = 90
    n2.inputs["Detail"].default_value = 7
    nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
    bp.inputs["Strength"].default_value = .10
    bp.inputs["Distance"].default_value = .004
    nt.links.new(n2.outputs[0], bp.inputs["Height"])
    nt.links.new(bp.outputs[0], b.inputs["Normal"])
    return m


def linen():
    """A thin curtain: most of the light goes through it, some of it straight."""
    m = surface("linen", (.86, .83, .76), 1.0, 0, .1)
    nt = m.node_tree
    out = nt.nodes["Material Output"]
    b = nt.nodes["Principled BSDF"]
    tr, tp, mix1, mix2 = nt.nodes.new("ShaderNodeBsdfTranslucent"), nt.nodes.new("ShaderNodeBsdfTransparent"), nt.nodes.new("ShaderNodeMixShader"), nt.nodes.new("ShaderNodeMixShader")
    tr.inputs[0].default_value = (.92, .88, .80, 1)
    mix1.inputs[0].default_value = .62
    mix2.inputs[0].default_value = .30
    nt.links.new(b.outputs[0], mix1.inputs[1])
    nt.links.new(tr.outputs[0], mix1.inputs[2])
    nt.links.new(mix1.outputs[0], mix2.inputs[1])
    nt.links.new(tp.outputs[0], mix2.inputs[2])
    nt.links.new(mix2.outputs[0], out.inputs["Surface"])
    return noisy(m, 900, .3, .0004, 4)


def lit_picture(name, picture, strength):
    """A picture that only gives out light, for what is outside."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    t, e, o = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    t.image = tex(picture)
    nt.links.new(nt.nodes.new("ShaderNodeUVMap").outputs[0], t.inputs[0])
    nt.links.new(t.outputs[0], e.inputs[0])
    e.inputs[1].default_value = strength
    nt.links.new(e.outputs[0], o.inputs[0])
    return m


def screen_material():
    """Glass: what it reflects always, what it shows only in the glow pass."""
    m = surface("screen", (.004, .004, .004), .07, 0, .5)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(os.path.join(OUT, "ui", "glow.png"), check_existing=True)
    nt.links.new(nt.nodes.new("ShaderNodeUVMap").outputs[0], t.inputs[0])
    nt.links.new(t.outputs[0], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = 0.0
    return m


# ---------------------------------------------------------------- shapes

def link(o):
    bpy.context.collection.objects.link(o)
    return o


def mesh_object(name, verts, faces, mat, smooth=False, uv=None, matrix=None):
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


def bevelled(o, width, segments=3, angle=35.0):
    m = o.modifiers.new("bevel", "BEVEL")
    m.width, m.segments, m.limit_method = width, segments, "ANGLE"
    m.angle_limit = math.radians(angle)
    return o


def box(name, x0, x1, y0, y1, z0, z1, mat, bevel=0.0, matrix=None):
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    o = mesh_object(name, v, f, mat, matrix=matrix)
    return bevelled(o, bevel) if bevel else o


def outline(x0, x1, y0, y1, r, n=10):
    """A rounded rectangle, counter-clockwise."""
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def slab(name, x0, x1, y0, y1, z0, z1, r, mat, bevel=0.0, matrix=None, n=10, uv=None):
    """A flat thing with round corners. uv: (x0, x1, y0, y1) in the slab's own plane, to print a picture on it."""
    p = outline(x0, x1, y0, y1, r, n)
    k = len(p)
    v = [(x, y, z0) for x, y in p] + [(x, y, z1) for x, y in p]
    f = [tuple(range(k - 1, -1, -1)), tuple(range(k, 2 * k))] + [(i, (i + 1) % k, k + (i + 1) % k, k + i) for i in range(k)]
    tuv = None
    if uv:
        a, b, c, d = uv
        tuv = [((x - a) / (b - a), (y - c) / (d - c)) for x, y, _ in v]
    o = mesh_object(name, v, f, mat, uv=tuv, matrix=matrix)
    me = o.data
    me.shade_smooth()
    me.set_sharp_from_angle(angle=math.radians(40))
    return bevelled(o, bevel) if bevel else o


def lathe(name, profile, x, y, z, mat, n=72):
    """Turned about the vertical: profile is (radius, height) from the outside bottom over the lip to the inside."""
    p = np.array(profile, np.float32)
    a = np.linspace(0, 2 * math.pi, n + 1)
    r = p[:, 0][:, None]
    xx, yy, zz = x + r * np.cos(a), y + r * np.sin(a), z + np.broadcast_to(p[:, 1][:, None], (len(p), n + 1))
    return_ = mesh_grid(name, xx, yy, zz)
    return_.data.materials.append(mat)
    return return_


def ellipsoid(name, radii, loc, mat, rot=(0, 0, 0), n=48):
    u, v = np.meshgrid(np.linspace(0, 2 * math.pi, 2 * n + 1), np.linspace(-math.pi / 2, math.pi / 2, n + 1))
    x, y, z = radii[0] * np.cos(v) * np.cos(u), radii[1] * np.cos(v) * np.sin(u), radii[2] * np.sin(v)
    o = mesh_grid(name, x, y, z)
    o.data.materials.append(mat)
    o.location, o.rotation_euler = loc, rot
    return o


def tube(name, points, radius, mat, res=4):
    """A soft line through the points, for cables and pens."""
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
    s.resolution_u = 12
    c.materials.append(mat)
    return link(bpy.data.objects.new(name, c))


# ---------------------------------------------------------------- the room

def room():
    wall = plaster()
    x0, x1, s, h = WIN
    t = .22
    box("wall_l", LEFT_X - .2, x0, WALL_Y, WALL_Y + t, 0, CEIL, wall)
    box("wall_r", x1, RIGHT_X + .2, WALL_Y, WALL_Y + t, 0, CEIL, wall)
    box("wall_low", x0, x1, WALL_Y, WALL_Y + t, 0, s, wall)
    box("wall_high", x0, x1, WALL_Y, WALL_Y + t, h, CEIL, wall)
    box("wall_left", LEFT_X - .2, LEFT_X, BACK_Y - .2, WALL_Y + t, 0, CEIL, wall)
    box("wall_right", RIGHT_X, RIGHT_X + .2, BACK_Y - .2, WALL_Y + t, 0, CEIL, wall)
    box("wall_back", LEFT_X - .2, RIGHT_X + .2, BACK_Y - .2, BACK_Y, 0, CEIL, wall)
    box("ceiling", LEFT_X - .2, RIGHT_X + .2, BACK_Y - .2, WALL_Y + t, CEIL, CEIL + .1, surface("ceiling", (.72, .70, .66), .9, 0, .2))
    box("floor", LEFT_X - .2, RIGHT_X + .2, BACK_Y - .2, WALL_Y + t, -.1, 0, surface("floor", (.26, .17, .10), .45))
    # skirting and the window's wood, painted a long time ago
    paint = noisy(surface("paint", (.74, .71, .64), .42, 0, .4), 40, .06, .001)
    box("skirt", LEFT_X, RIGHT_X, WALL_Y - .012, WALL_Y, 0, .09, paint)
    b = .045
    box("frame_l", x0, x0 + b, WALL_Y + .1, WALL_Y + .16, s, h, paint)
    box("frame_r", x1 - b, x1, WALL_Y + .1, WALL_Y + .16, s, h, paint)
    # each set a little back from the last: pieces overlapping in one plane leave black patches where the light can't decide between them
    box("frame_t", x0, x1, WALL_Y + .101, WALL_Y + .16, h - b, h, paint)
    box("frame_b", x0, x1, WALL_Y + .101, WALL_Y + .16, s, s + b, paint)
    mid = (x0 + x1) / 2
    box("frame_m", mid - b / 2, mid + b / 2, WALL_Y + .102, WALL_Y + .159, s, h, paint)
    box("frame_h", x0, x1, WALL_Y + .103, WALL_Y + .15, s + .92, s + .92 + b * .8, paint)
    box("sill", x0 - .03, x1 + .03, WALL_Y - .035, WALL_Y + .14, s - .03, s, paint, .003)
    # what is out there
    ww, zz = 9.0, 5.0
    xs = np.array([[-ww, ww], [-ww, ww]], np.float32) - .6
    ys = np.full((2, 2), WALL_Y + 7.0, np.float32)
    zs = np.array([[-1.0, -1.0], [zz, zz]], np.float32)
    o = mesh_grid("outside", xs, ys, zs)
    o.data.materials.append(lit_picture("outside", "outside.png", 1.5))
    o.visible_shadow = False
    # the curtain, drawn over the half of the window next to the table, hanging in loose folds from a rail
    rail = surface("rail", (.05, .05, .05), .4, .6)
    tube("rail", [(x0 - .12, WALL_Y - .055, h + .09), (x1 + .22, WALL_Y - .055, h + .09)], .009, rail)
    cx0, cx1, nx, nz = mid + .22, x1 + .17, 300, 90
    xx, zv = np.meshgrid(np.linspace(cx0, cx1, nx), np.linspace(.03, h + .085, nz))
    q = (zv - zv.min()) / (zv.max() - zv.min())
    r = np.random.default_rng(4)
    yv = np.zeros_like(xx)
    for wave, amp in ((.062, .024), (.041, .011), (.113, .018)):
        yv += amp * np.sin(2 * math.pi * (xx - cx0) / wave + r.uniform(0, 6.28) + (1 - q) * r.uniform(-1.6, 1.6))
    yv *= .75 + .35 * (1 - q)
    # a little air from the open half of the window: it takes the lower part of the cloth into the room and lets it back, the wave running along the folds
    t = POSE * POSE_S
    swing = (1 - q) ** 1.4
    yv += -.028 * swing * (.5 - .5 * math.cos(2 * math.pi * t / 7.5))
    yv += .006 * swing * np.sin(2 * math.pi * ((xx - cx0) / .34 - t / 4.2))
    c = mesh_grid("curtain", xx, WALL_Y - .055 + yv, zv)
    c.data.materials.append(linen())


def table():
    wood = table_material()
    x0, x1, y0, y1 = TABLE
    o = slab("table", x0, x1, y0, y1, ZT - .032, ZT, .004, wood, .0015, n=3, uv=(x0, x1, y1, y0))
    o.data.materials.append(table_wood("XY", "Z", (.2, -.337)))
    for side in o.data.polygons[2:]:
        side.material_index = 1
    for n, (x, y) in enumerate(((x0 + .06, y0 + .06), (x1 - .06, y0 + .06), (x0 + .06, y1 - .06), (x1 - .06, y1 - .06))):
        box("leg", x - .025, x + .025, y - .025, y + .025, 0, ZT - .03, table_wood("Z", "XY", (.25 + .17 * n, .64 + .08 * n - (x + y) / (y1 - y0)), .68), .003)
    box("apron", x0 + .06, x1 - .06, y0 + .05, y0 + .07, ZT - .11, ZT - .03, table_wood("X", "Z", (.5, -.151), .68), .002)


# ---------------------------------------------------------------- the computer

def aluminium(name="aluminium"):
    return noisy(surface(name, (.84, .89, .98), .38, 1.0, .5), 2600, .02, .0002, 3)     # bead-blasted, all metal: neutral_in_metal() keeps the room's colours out of it


def extruded(name, profile, x0, x1, mat, matrix=None):
    """A side view, counter-clockwise in (y, z), carried across from x0 to x1."""
    n = len(profile)
    v = [(x0, y, z) for y, z in profile] + [(x1, y, z) for y, z in profile]
    f = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))] + [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    return mesh_object(name, v, f, mat, matrix=matrix)


def bezier(a, c, b, n=10):
    return [tuple((1 - t) ** 2 * np.array(a) + 2 * t * (1 - t) * np.array(c) + t * t * np.array(b)) for t in np.linspace(0, 1, n)]


def stand(M, alu):
    """One sheet of aluminium, bent: the foot flat on the table, reaching 5.7 cm in front of the glass and 11 behind it, then up and forward to the hinge behind the case. y behind the glass, z above the table."""
    t = .0105
    zb, zt = .0012, .0072
    front, rear = -.0573, .1107
    hinge, foot = np.array((.0304, .2055)), np.array((.0800, zt))
    u = (hinge - foot) / np.linalg.norm(hinge - foot)
    n = np.array((u[1], -u[0]))                     # towards the back
    face = lambda side, z: foot + side * n * t / 2 + u * (z - foot[1] - (side * n * t / 2)[1]) / u[1]
    rc, fc = face(1, zt), face(-1, zt)
    pts = [(front + .002, zb), (rear - .002, zb), (rear, zb + .0015), (rear, zt - .0015), (rear - .002, zt)]
    pts += bezier((rc[0] + .012, zt), rc, rc + u * .012)
    a0 = math.atan2(n[1], n[0])
    pts += [tuple(hinge + t / 2 * np.array((math.cos(a0 + q), math.sin(a0 + q)))) for q in np.linspace(0, math.pi, 12)]
    pts += bezier(fc + u * .032, fc, (fc[0] - .032, zt))
    pts += [(front + .002, zt), (front, zt - .0015), (front, zb + .0015)]
    arm = extruded("stand", pts, -STAND_W / 2, STAND_W / 2, alu, M)
    # the oval hole for the cable: its lower end shows just under the case
    c = foot + u * (.136 - foot[1]) / u[1]
    ring = []
    for q in np.linspace(0, 2 * math.pi, 48, endpoint=False):
        along = .0105 * np.sign(math.sin(q)) + .019 * math.sin(q)
        ring.append((.019 * math.cos(q), along))
    rows = [[(x, *(c + u * w + n * d)) for x, w in ring] for d in (-.03, .03)]
    k = len(ring)
    v = rows[0] + rows[1]
    f = [tuple(range(k - 1, -1, -1)), tuple(range(k, 2 * k))] + [(i, (i + 1) % k, k + (i + 1) % k, k + i) for i in range(k)]
    cut = mesh_object("stand_hole", v, f, alu, matrix=M)
    b = arm.modifiers.new("hole", "BOOLEAN")
    b.operation, b.object, b.solver = "DIFFERENCE", cut, "EXACT"
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.modifier_apply(modifier="hole")
    bpy.data.objects.remove(cut)
    arm.data.shade_smooth()
    arm.data.set_sharp_from_angle(angle=math.radians(28))
    bevelled(arm, .0005, 2, 40)
    barrel = surface("hinge", (.62, .62, .62), .28, 1.0)
    tube_x = [tuple(M @ Vector((x, hinge[0], hinge[1]))) for x in (-STAND_W / 2 + .002, 0, STAND_W / 2 - .002)]
    tube("hinge", tube_x, .0098, barrel, 4)
    rubber = surface("rubber", (.03, .03, .03), .7)
    for x in (-STAND_W / 2 + .010, STAND_W / 2 - .010):
        for y in (front + .008, rear - .008):
            box("foot", x - .009, x + .009, y - .003, y + .003, 0, zb, rubber, 0, M)


def display():
    """The Studio Display. Returns the screen, its corners, and the matrix of the glass: x across, y up from the table, z out of the glass towards the room."""
    M = Matrix.Translation((DISP_X, DISP_Y, ZT)) @ Matrix.Rotation(DISP_YAW, 4, "Z")
    alu = aluminium()
    stand(M, alu)
    L = M @ Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    W, y0, y1 = DISP_W / 2, DISP_UP, DISP_UP + DISP_H
    slab("case", -W, W, y0, y1, -DISP_T, -.0005, DISP_R, alu, .0007, L, n=8)
    glass = surface("bezel", (.003, .003, .0035), .05, 0, .5)
    e = .0007
    slab("glass", -W + e, W - e, y0 + e, y1 - e, -.0006, 0, DISP_R - e, glass, 0, L, n=8)
    lens = surface("lens", (.012, .016, .022), .12, 0, .5)
    slab("camera", -.0016, .0016, y1 - .0066 - .0016, y1 - .0066 + .0016, 0, .00006, .0016, lens, 0, L, n=6)
    vc = (y0 + y1) / 2
    x0, x1, s0, s1 = -SCREEN_W / 2, SCREEN_W / 2, vc - SCREEN_H / 2, vc + SCREEN_H / 2
    z = .00012
    screen = mesh_object("screen", [(x0, s0, z), (x1, s0, z), (x1, s1, z), (x0, s1, z)], [(0, 1, 2, 3)], screen_material(), uv=[(0, 0), (1, 0), (1, 1), (0, 1)], matrix=L)
    corners = [tuple(L @ Vector(p)) for p in ((x0, s1, z), (x1, s1, z), (x1, s0, z), (x0, s0, z))]     # top left, clockwise
    return screen, corners, L


def on_screen(L, u, v):
    """A point of the screen, as parts of its width and of its height from the top left, in the world."""
    vc = DISP_UP + DISP_H / 2
    return L @ Vector((-SCREEN_W / 2 + SCREEN_W * u, vc + SCREEN_H / 2 - SCREEN_H * v, .00012))


def magic_keyboard(x, y, yaw):
    """Silver aluminium and white keys: 27.9 by 11.49, 0.41 thick at the front and 1.09 at the back."""
    W, D, low, high = .279, .1149, .0041, .0109
    K = Matrix.Translation((x, y, ZT)) @ Matrix.Rotation(yaw, 4, "Z")
    p = outline(-W / 2, W / 2, -D / 2, D / 2, .0095, 8)
    n = len(p)
    v = [(a, b, .0005) for a, b in p] + [(a, b, low + (b + D / 2) / D * (high - low)) for a, b in p]
    f = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))] + [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    alu = aluminium("kb_alu")
    o = mesh_object("keyboard", v, f, alu, matrix=K)
    o.data.shade_smooth()
    o.data.set_sharp_from_angle(angle=math.radians(35))
    bevelled(o, .0006, 2, 30)
    tilt = math.atan2(high - low, D)
    top = K @ Matrix.Translation((0, -D / 2, low)) @ Matrix.Rotation(tilt, 4, "X")
    run = D / math.cos(tilt)
    left, far = -KEYS_W / 2, run - .0052
    verts, faces, uvs = [], [], []
    gap = .0006
    for legend, x0, y0, x1, y1 in keys():
        q = outline(left + x0 + gap, left + x1 - gap, far - y1 + gap, far - y0 - gap, .0013, 4)
        k, at = len(q), len(verts)
        for z in (-.0002, .00045):
            for a, b in q:
                verts.append((a, b, z))
                uvs.append(((a - left) / KEYS_W, 1 - (far - b) / KEYS_H))
        faces.append(tuple(at + k + i for i in range(k)))
        faces += [(at + i, at + (i + 1) % k, at + k + (i + 1) % k, at + k + i) for i in range(k)]
    mesh_object("keys", verts, faces, printed("keys", "keys.png", .40, .4), uv=uvs, matrix=top)


def magic_mouse(x, y, yaw):
    """A low white shell on an aluminium foot: 11.35 long, 5.71 wide, 2.16 high."""
    a, b, h = .02855, .05675, .0216
    phi = np.linspace(0, 2 * math.pi, 97)
    rho = np.linspace(0, 1, 40)[:, None]
    cx, sy = np.cos(phi), np.sin(phi)
    ex, ey = np.sign(cx) * np.abs(cx) ** .78, np.sign(sy) * np.abs(sy) ** .78
    arch = 1 - .22 * (rho * ey + .18) ** 2                # highest a little behind its middle
    zz = .0034 + (h - .0034) * np.clip(1 - rho ** 2.6, 0, 1) ** .55 * arch
    M = Matrix.Translation((x, y, ZT)) @ Matrix.Rotation(yaw, 4, "Z")
    pts = np.stack([a * rho * ex, b * rho * ey, zz, np.ones_like(zz)], -1) @ np.array(M).T
    top = mesh_grid("mouse", pts[..., 0], pts[..., 1], pts[..., 2])
    top.data.materials.append(surface("mouse", (.86, .86, .85), .28, 0, .5))
    pts = np.stack([np.stack([a * .97 * ex, a * .998 * ex]), np.stack([b * .97 * ey, b * .998 * ey]), np.stack([np.full_like(ex, .0006), np.full_like(ex, .0036)]), np.ones((2, len(ex)))], -1) @ np.array(M).T
    foot = mesh_grid("mouse_foot", pts[..., 0], pts[..., 1], pts[..., 2])
    foot.data.materials.append(aluminium("mouse_alu"))


def fine_mesh(name):
    """Dark grey, drilled with holes a millimetre apart."""
    m = surface(name, (.03, .03, .032), .5)
    nt = m.node_tree
    tc, ch = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeTexChecker")
    ch.inputs["Color1"].default_value = (.006, .006, .007, 1)
    ch.inputs["Color2"].default_value = (.055, .055, .058, 1)
    ch.inputs["Scale"].default_value = 1000
    nt.links.new(tc.outputs["Object"], ch.inputs["Vector"])
    nt.links.new(ch.outputs[0], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return m


def mac_studio(x, y, yaw):
    """Measured off the real thing's front. A 19.7 cm square with large round corners, 8.8 high, on a dark mesh ring set well in from it, so that it seems to float: 9.5 in all. On its front two upright USB-C slots, the card slot lying down, and the white light, x as parts of the width from its left, all at 82% of its height from the top."""
    M = Matrix.Translation((x, y, ZT)) @ Matrix.Rotation(yaw, 4, "Z")
    s, z0, z1 = .0985, .0075, .095
    slab("studio", -s, s, -s, s, z0, z1, .030, aluminium("studio_alu"), .0012, M, n=24)
    inset, narrow = .12 * 2 * s, .0045
    hi, lo = outline(-s + inset, s - inset, -s + inset, s - inset, .024, 12), outline(-s + inset + narrow, s - inset - narrow, -s + inset + narrow, s - inset - narrow, .022, 12)
    k = len(hi)
    v = [(a, b, 0.0) for a, b in lo] + [(a, b, z0 + .001) for a, b in hi]
    f = [tuple(range(k - 1, -1, -1)), tuple(range(k, 2 * k))] + [(i, (i + 1) % k, k + (i + 1) % k, k + i) for i in range(k)]
    ring = mesh_object("studio_mesh", v, f, fine_mesh("studio_mesh"), matrix=M)
    ring.data.shade_smooth()
    front = M @ Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    along = lambda part: -s + part * 2 * s
    zc = z1 - .82 * (z1 - z0)
    dark = surface("port", (.005, .005, .006), .4)
    for part in (.16, .236):
        slab("usb", along(part) - .00135, along(part) + .00135, zc - .0045, zc + .0045, s - .0003, s + .00012, .00133, dark, 0, front, n=6)
    slab("sd", along(.307), along(.445), zc - .00135, zc + .00135, s - .0003, s + .00012, .00133, dark, 0, front, n=6)
    led = surface("light", (.9, .9, .9), .3)
    led.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (1, 1, 1, 1)
    led.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 1.2
    slab("light", along(.84) - .0015, along(.84) + .0015, zc - .0015, zc + .0015, s - .0003, s + .00012, .00149, led, 0, front, n=8)


# ---------------------------------------------------------------- what is on the table

def calc(nt, op, *inputs, clamp=False):
    """A maths node: each input a socket or a number."""
    n = nt.nodes.new("ShaderNodeMath")
    n.operation, n.use_clamp = op, clamp
    for i, v in enumerate(inputs):
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def blended(nt, factor, a, b, how="MIX"):
    """Two colours, each a socket or (r, g, b)."""
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type, n.blend_type = "RGBA", how
    for i, v in ((0, factor), (6, a), (7, b)):
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        elif isinstance(v, tuple):
            n.inputs[i].default_value = (*v, 1)
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[2]


def picture_in(nt, name, into, box=False):
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = tex(name, "Non-Color")
    t.interpolation = "Linear" if box else "Cubic"
    if box:
        t.projection, t.projection_blend = "BOX", .15
    nt.links.new(into, t.inputs[0])
    return t.outputs[0]


def bumped(nt, height, *bsdfs):
    """height in metres, as a shadow would see it."""
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Distance"].default_value = 1.0
    nt.links.new(height, bp.inputs["Height"])
    for b in bsdfs:
        nt.links.new(bp.outputs[0], b.inputs["Normal"])


def grain_of(nt, skin, where):
    """What a book's skin is made of: its height from 0 in the creases to 1 on top, how bright each point of it is, and how high in metres that 1 stands. Leather is cells pressed together; cloth is the woven tile from film_room_tex.py, threads going over and under each other."""
    if skin["skin"] == "leather":
        c = skin["cell"]
        wander = nt.nodes.new("ShaderNodeTexNoise")
        wander.inputs["Scale"].default_value = 1 / (3 * c)
        wander.inputs["Detail"].default_value = 2
        nt.links.new(where, wander.inputs["Vector"])
        moved = nt.nodes.new("ShaderNodeVectorMath")
        moved.operation = "SCALE"
        nt.links.new(wander.outputs["Color"], moved.inputs[0])
        moved.inputs["Scale"].default_value = c * 1.1
        at = nt.nodes.new("ShaderNodeVectorMath")
        nt.links.new(where, at.inputs[0])
        nt.links.new(moved.outputs[0], at.inputs[1])
        cells = nt.nodes.new("ShaderNodeTexVoronoi")
        cells.feature = "DISTANCE_TO_EDGE"
        cells.inputs["Scale"].default_value = 1 / c
        nt.links.new(at.outputs[0], cells.inputs["Vector"])
        up = nt.nodes.new("ShaderNodeMapRange")
        up.interpolation_type = "SMOOTHSTEP"
        up.inputs["From Max"].default_value = .24
        nt.links.new(cells.outputs["Distance"], up.inputs["Value"])
        relief = up.outputs[0]
        return relief, calc(nt, "ADD", .74, calc(nt, "MULTIPLY", relief, .26)), .00009
    p = skin["pitch"]
    m = nt.nodes.new("ShaderNodeMapping")
    m.inputs["Scale"].default_value = (1 / (64 * p),) * 3
    nt.links.new(where, m.inputs["Vector"])
    relief = picture_in(nt, "weave.png", m.outputs[0], True)
    tone = picture_in(nt, "weave_tone.png", m.outputs[0], True)
    return relief, calc(nt, "MULTIPLY", tone, 1.15), p * .12


def bound(name, b, title=None):
    """The skin of a hardback, in the book's own coordinates: rubbed along the edges of its boards and hardest on their corners, along the ends and shoulders of its back. title: the picture of where the die came down on the back, gold or only pressed."""
    m = surface(name, (1, 1, 1), b["rough"], 0, b["spec"])
    nt = m.node_tree
    skin = nt.nodes["Principled BSDF"]
    where = nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]
    relief, tone, high = grain_of(nt, b, where)
    part = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(where, part.inputs[0])
    in_x = calc(nt, "SUBTRACT", b["w"] / 2, calc(nt, "ABSOLUTE", part.outputs["X"]))
    if title:
        hc, hs = b["h"] / 2, b["h"] / 2 - .25 * b["board"]
        off_shoulder = calc(nt, "ABSOLUTE", calc(nt, "SUBTRACT", calc(nt, "ABSOLUTE", calc(nt, "SUBTRACT", part.outputs["Z"], hc)), hs))
        rub = calc(nt, "MAXIMUM", calc(nt, "SUBTRACT", .6, calc(nt, "DIVIDE", in_x, .0016)), calc(nt, "SUBTRACT", .1, calc(nt, "DIVIDE", off_shoulder, .0012)))
        in_joint = calc(nt, "DIVIDE", calc(nt, "SUBTRACT", part.outputs["Y"], -b["d"] / 2 + b["bulge"] + .0004), .0016, clamp=True)
    else:
        in_y = calc(nt, "SUBTRACT", b["d"] / 2, part.outputs["Y"])
        edge = calc(nt, "SUBTRACT", .72, calc(nt, "DIVIDE", calc(nt, "MINIMUM", in_x, in_y), .0014))
        corner = calc(nt, "SUBTRACT", 1.0, calc(nt, "DIVIDE", calc(nt, "MAXIMUM", in_x, in_y), .0045))
        rub = calc(nt, "MAXIMUM", edge, corner)
    worn_by = nt.nodes.new("ShaderNodeTexNoise")
    worn_by.inputs["Scale"].default_value = 1 / .0016
    worn_by.inputs["Detail"].default_value = 4
    nt.links.new(where, worn_by.inputs["Vector"])
    wear = calc(nt, "ADD", rub, calc(nt, "MULTIPLY", calc(nt, "SUBTRACT", worn_by.outputs[0], .5), 1.5), clamp=True)
    # cloth wears off the tops of its threads
    bare = calc(nt, "MULTIPLY", wear, relief) if b["skin"] == "cloth" else wear
    colour = blended(nt, 1.0, b["colour"], tone, "MULTIPLY")
    colour = blended(nt, bare, colour, b["worn"])
    if title:
        colour = blended(nt, in_joint, colour, (.46, .46, .46), "MULTIPLY")
    nt.links.new(calc(nt, "ADD", b["rough"], calc(nt, "MULTIPLY", bare, .3)), skin.inputs["Roughness"])
    height = calc(nt, "MULTIPLY", relief, high)
    bsdfs = [skin]
    if title:
        die = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(picture_in(nt, title, nt.nodes.new("ShaderNodeUVMap").outputs[0]), die.inputs[0])
        pressed, dent = die.outputs[0], die.outputs[1]
        height = calc(nt, "MULTIPLY", height, calc(nt, "SUBTRACT", 1, calc(nt, "MULTIPLY", dent, .7)))
        if b["stamp"] == "gold":
            # leaf: never quite evenly laid, and not quite flat
            leaf = nt.nodes.new("ShaderNodeTexNoise")
            leaf.inputs["Scale"].default_value = 1 / .0035
            leaf.inputs["Detail"].default_value = 6
            nt.links.new(where, leaf.inputs["Vector"])
            crumple = nt.nodes.new("ShaderNodeTexNoise")
            crumple.inputs["Scale"].default_value = 1 / .0005
            crumple.inputs["Detail"].default_value = 3
            nt.links.new(where, crumple.inputs["Vector"])
            height = calc(nt, "SUBTRACT", height, calc(nt, "MULTIPLY", dent, .00018))
            height = calc(nt, "ADD", height, calc(nt, "MULTIPLY", calc(nt, "MULTIPLY", pressed, crumple.outputs[0]), .000035))
            gold = nt.nodes.new("ShaderNodeBsdfPrincipled")
            gold.name = "gold"
            nt.links.new(blended(nt, 1.0, (.98, .70, .30), calc(nt, "ADD", .5, calc(nt, "MULTIPLY", leaf.outputs[0], 1.0)), "MULTIPLY"), gold.inputs["Base Color"])
            gold.inputs["Metallic"].default_value = 1.0
            gold.inputs["Roughness"].default_value = .25
            lay = nt.nodes.new("ShaderNodeMixShader")
            # gold takes on the tops of the grain first, and never quite everywhere
            nt.links.new(calc(nt, "MULTIPLY", pressed, calc(nt, "ADD", .62, calc(nt, "MULTIPLY", relief, .7)), clamp=True), lay.inputs[0])
            nt.links.new(skin.outputs[0], lay.inputs[1])
            nt.links.new(gold.outputs[0], lay.inputs[2])
            nt.links.new(lay.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
            bsdfs.append(gold)
        else:
            height = calc(nt, "SUBTRACT", height, calc(nt, "MULTIPLY", dent, .00034))
            colour = blended(nt, calc(nt, "MULTIPLY", pressed, .94), colour, b["ink"])
            nt.links.new(calc(nt, "SUBTRACT", b["rough"], calc(nt, "MULTIPLY", pressed, .3)), skin.inputs["Roughness"])
    nt.links.new(colour, skin.inputs["Base Color"])
    bumped(nt, height, *bsdfs)
    return m


def pages_material(seed, colour=(.78, .72, .58), across="Z"):
    """The edge of a stack of sheets: the lines of pages.png stay at one place along across, whichever side they are seen from."""
    m = surface(f"pages{seed}", (1, 1, 1), .85, 0, .2)
    nt = m.node_tree
    part = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], part.inputs[0])
    at = nt.nodes.new("ShaderNodeCombineXYZ")
    u, v = [part.outputs[k] for k in "XYZ" if k != across]
    nt.links.new(calc(nt, "DIVIDE", calc(nt, "ADD", u, v), .25), at.inputs[0])
    nt.links.new(calc(nt, "ADD", calc(nt, "DIVIDE", part.outputs[across], .03), seed * .37), at.inputs[1])
    sheets = picture_in(nt, "pages.png", at.outputs[0])
    nt.links.new(blended(nt, 1.0, colour, sheets, "MULTIPLY"), nt.nodes["Principled BSDF"].inputs["Base Color"])
    bumped(nt, calc(nt, "MULTIPLY", sheets, .00008), nt.nodes["Principled BSDF"])
    return m


def placed(b, z):
    return Matrix.Translation((b["x"], b["y"] + b["d"] / 2, z)) @ Matrix.Rotation(math.radians(b["yaw"]), 4, "Z")


def back_profile(b):
    """A hardback's back from its end, (y, z), from under the top board's skin down into the joint, up over the shoulder, round and back the same way: the skin's middle."""
    d, h, t, bulge, g = b["d"], b["h"], b["board"], b["bulge"], b["groove"]
    yb = -d / 2 + bulge + g
    hc, hs = h / 2, h / 2 - .25 * t
    arc = [(-d / 2 + bulge * (z / hs) ** 2, hc + z) for z in np.linspace(hs, 0, 30)]
    half = [(yb + .002, h - .0008), (yb - .0002, h - .0007), (yb - .45 * g, h - .85 * t)] + arc
    line = np.array(half + [(y, h - z) for y, z in reversed(half[:-1])])
    # the skin takes no sharp bends: walk along it, soften, and keep a point every .7 mm
    run = np.r_[0, np.cumsum(np.hypot(*np.diff(line, axis=0).T))]
    fine = np.linspace(0, run[-1], 1200)
    smooth = np.stack([np.interp(fine, run, line[:, 0]), np.interp(fine, run, line[:, 1])], -1)
    sigma = .00055 / (fine[1] - fine[0])
    kernel = np.exp(-.5 * (np.arange(-int(4 * sigma), int(4 * sigma) + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    wide = np.pad(smooth, ((pad, pad), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(wide[:, k], kernel, "valid") for k in (0, 1)], -1)
    return smooth[::max(int(.0007 / (fine[1] - fine[0])), 1)].tolist() + [smooth[-1].tolist()]


def case_book(i, b, z):
    """A hardback lying flat, its back towards the room: boards that stand 3 mm proud of the pages, and between them the back, rounded, with a joint on each side."""
    w, d, h, t, bulge, g = b["w"], b["d"], b["h"], b["board"], b["bulge"], b["groove"]
    M = placed(b, z)
    skin = bound(f"book{i}", b)
    yb = -d / 2 + bulge + g
    for k, (z0, z1) in enumerate(((0, t), (h - t, h))):
        slab(f"book{i}_board{k}", -w / 2, w / 2, yb, d / 2, z0, z1, .0013, skin, .00035, M, n=6)
    yz = back_profile(b)
    # its ends stop a little short of the boards, most in the middle, where the silk roll behind shows: never cut quite straight
    r = np.random.default_rng(20 + i)
    phase = r.uniform(0, 6, 2)
    verts, uv = [], []
    for y, zz in yz:
        across = min(abs(zz - h / 2) / (h / 2 - .25 * t), 1)
        short = .0005 + .0017 * (1 - across * across)
        x0, x1 = -w / 2 + short + .00016 * math.sin(zz * 260 + phase[0]), w / 2 - short - .00016 * math.sin(zz * 240 + phase[1])
        verts += [(x0, y, zz), (x1, y, zz)]
        uv += [((x0 + w / 2) / w, zz / h), ((x1 + w / 2) / w, zz / h)]
    faces = [(2 * j, 2 * j + 2, 2 * j + 3, 2 * j + 1) for j in range(len(yz) - 1)]
    back = mesh_object(f"book{i}_back", verts, faces, bound(f"book{i}_backskin", b, f"title_{i}.png"), smooth=True, uv=uv, matrix=M)
    thick = back.modifiers.new("skin", "SOLIDIFY")
    thick.thickness, thick.offset = .0011, 0.0
    bevelled(back, .00035, 2, 50.0)
    # the pages, their back rounded with the book's and their fore-edge hollowed by it
    hc, hs = h / 2, h / 2 - t
    x0, x1 = -w / 2 + .003, w / 2 - .003
    zs = np.linspace(h - t, t, 25)
    bow = bulge * ((zs - hc) / hs) ** 2
    ring = [(-d / 2 + .0017 + s, zz) for s, zz in zip(bow, zs)] + [(d / 2 - .003 - bulge + s, zz) for s, zz in zip(bow[::-1], zs[::-1])]
    n = len(ring)
    verts = [(x0, y, zz) for y, zz in ring] + [(x1, y, zz) for y, zz in ring]
    faces = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))] + [(j, n + j, n + (j + 1) % n, (j + 1) % n) for j in range(n)]
    mesh_object(f"book{i}_pages", verts, faces, pages_material(i + 1), matrix=M)
    # the little silk rolls at each end of the pages' back
    band = surface(f"book{i}_band", b["band"][0], .4, 0, .5)
    nt = band.node_tree
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = 170
    wave.inputs["Distortion"].default_value = 0
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], wave.inputs["Vector"])
    nt.links.new(blended(nt, calc(nt, "GREATER_THAN", wave.outputs[1], .5), b["band"][0], b["band"][1]), nt.nodes["Principled BSDF"].inputs["Base Color"])
    for k, x in enumerate((x0 - .0005, x1 + .0005)):
        tube(f"book{i}_band{k}", [tuple(M @ Vector((x, -d / 2 + .0024 + s, zz))) for s, zz in zip(bow, zs)], .0012, band, 3)


def paper_cover(name, colour):
    """Handmade paper: the lines of the mould it was lifted on, a mottling, and fibres. All very faint."""
    m = surface(name, (1, 1, 1), .74, 0, .28)
    nt = m.node_tree
    where = nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]
    laid = nt.nodes.new("ShaderNodeTexWave")
    laid.bands_direction = "Z"
    laid.inputs["Scale"].default_value = 340
    laid.inputs["Distortion"].default_value = .6
    laid.inputs["Detail Scale"].default_value = 4
    nt.links.new(where, laid.inputs["Vector"])
    cloud = nt.nodes.new("ShaderNodeTexNoise")
    cloud.inputs["Scale"].default_value = 1 / .018
    cloud.inputs["Detail"].default_value = 6
    nt.links.new(where, cloud.inputs["Vector"])
    tone = calc(nt, "ADD", .86, calc(nt, "MULTIPLY", cloud.outputs[0], .28))
    nt.links.new(blended(nt, 1.0, colour, tone, "MULTIPLY"), nt.nodes["Principled BSDF"].inputs["Base Color"])
    bumped(nt, calc(nt, "MULTIPLY", laid.outputs[1], .000035), nt.nodes["Principled BSDF"])
    return m


def silk_cloth(name, colour):
    m = surface(name, colour, .58, 0, .35)
    nt = m.node_tree
    relief, tone, high = grain_of(nt, dict(skin="cloth", pitch=.00042), nt.nodes.new("ShaderNodeTexCoord").outputs["Object"])
    nt.links.new(blended(nt, 1.0, colour, tone, "MULTIPLY"), nt.nodes["Principled BSDF"].inputs["Base Color"])
    bumped(nt, calc(nt, "MULTIPLY", relief, high), nt.nodes["Principled BSDF"])
    return m


def case_cloth(name, c):
    """The blue cloth of a case: real threads, and over them the creases and rubbed places of its front, from film_room_tex.py."""
    m = surface(name, (1, 1, 1), c["rough"], 0, c["spec"])
    nt = m.node_tree
    relief, tone, high = grain_of(nt, dict(skin="cloth", pitch=c["pitch"]), nt.nodes.new("ShaderNodeTexCoord").outputs["Object"])
    folds = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(picture_in(nt, "case_folds.png", nt.nodes.new("ShaderNodeUVMap").outputs[0]), folds.inputs[0])
    colour = blended(nt, 1.0, c["colour"], tone, "MULTIPLY")
    nt.links.new(blended(nt, calc(nt, "MULTIPLY", folds.outputs[1], relief), colour, c["worn"]), nt.nodes["Principled BSDF"].inputs["Base Color"])
    bumped(nt, calc(nt, "ADD", calc(nt, "MULTIPLY", relief, high), calc(nt, "MULTIPLY", calc(nt, "SUBTRACT", folds.outputs[0], .5), .0007)), nt.nodes["Principled BSDF"])
    return m


def slipcase(c):
    """A cloth case of the old Chinese kind, four-sided, standing with its front to the room: hard boards on four sides and the volumes' ends showing above, a title slip to the left of its front, and two bone clasps at its right edge."""
    w, h, t, b = c["w"], c["h"], c["t"], c["board"]
    M = Matrix.Translation((c["x"], c["y"], ZT)) @ Matrix.Rotation(math.radians(c["yaw"]), 4, "Z")
    # the same again turned up, for what lies on the front: x across, y up the front, z out of it
    F = M @ Matrix.Rotation(math.radians(90), 4, "X")
    outer, inner = outline(0, w, 0, t, .0016, 6), outline(b, w - b, b, t - b, .0006, 6)
    k = len(outer)
    verts = [(x, y, 0) for x, y in outer] + [(x, y, h) for x, y in outer] + [(x, y, 0) for x, y in inner] + [(x, y, h) for x, y in inner]
    faces = []
    for i in range(k):
        j = (i + 1) % k
        faces += [(i, j, k + j, k + i), (3 * k + i, 3 * k + j, 2 * k + j, 2 * k + i), (k + i, k + j, 3 * k + j, 3 * k + i), (2 * k + i, 2 * k + j, j, i)]
    shell = mesh_object("case", verts, faces, case_cloth("case", c), uv=[(x / w, z / h) for x, y, z in verts], matrix=M)
    shell.data.shade_smooth()
    shell.data.set_sharp_from_angle(angle=math.radians(40))
    bevelled(shell, .0005, 2)
    # what is in it: the ends of three volumes
    across = (t - 2 * b - .0008) / 3
    for n in range(3):
        y0 = b + .0002 + n * (across + .0002)
        box(f"case_volume{n}", b + .001, w - b - .001, y0, y0 + across, .003, h - .0024, pages_material(20 + n, (.66, .56, .40), "Y"), 0, M)
    sw, sh, sl, st = c["slip"]
    label = noisy(printed("slip_case", "slip_case.png", .72, .3), 1600, .25, .0001, 4)
    slab("slip_case", sl, sl + sw, h - st - sh, h - st, .00005, .0003, .00015, label, 0, F @ Matrix.Translation((sl, h - st - sh / 2, 0)) @ Matrix.Rotation(math.radians(.3), 4, "Z") @ Matrix.Translation((-sl, -(h - st - sh / 2), 0)), n=2, uv=(sl, sl + sw, h - st - sh, h - st))
    strap = silk_cloth("case_strap", tuple(v * .92 for v in c["colour"]))
    bone = noisy(surface("bone", (.70, .63, .46), .34, 0, .5), 900, .15, .0003, 3)
    x = w - c["clasp_from_right"]
    for n, part in enumerate(c["clasps"]):
        z = h * part
        slab(f"strap{n}", x + .006, w + .0002, z - .0055, z + .0055, .0001, .0009, .0007, strap, .00018, F, n=3)
        slab(f"clasp{n}", x - .012, x + .012, z - .0042, z + .0042, .0005, .0026, .0039, bone, .0005, F, n=8)
        slab(f"loop{n}", x - .007, x - .002, z - .0068, z + .0068, .0001, .0034, .0006, strap, .0003, F, n=3)


def stitched(c):
    """A stitched Chinese volume leaning on the case, its cover to the room: soft enough to sag a little, the title slip on the left, on the right four-hole stitching, silk round the two corners, and between the stitching and the back only the plain cover."""
    w, h, t, yaw, lean = c["w"], c["h"], c["t"], math.radians(c["yaw"]), math.radians(c["lean"])
    M = Matrix.Translation((c["x"], c["y"], ZT + t * math.sin(lean))) @ Matrix.Rotation(yaw, 4, "Z") @ Matrix.Rotation(-lean, 4, "X")
    F = M @ Matrix.Rotation(math.radians(90), 4, "X")
    x_s = w - c["stitch"]

    def face(x, z):
        """How far back the cover lies at a point of it: bent back towards the case in the middle of its height, the free edge curling forward, drawn in along the stitching."""
        free = np.clip(1 - x / w, 0, 1)
        return c["bow"] * np.sin(np.pi * np.clip(z / h, 0, 1)) - .0024 * free ** 2.2 * (.35 + .65 * np.abs(2 * z / h - 1)) + .00028 * np.exp(-((x - x_s) / .0016) ** 2) + free * (.00042 * np.sin(z / h * 7.3 + x / w * 2.1 + .7) + .00022 * np.sin(z / h * 15.1 - x / w * 4.3))

    def grid(name, x, y, z, mat, thick=0.0, offset=-1.0):
        o = mesh_grid(name, x, y, z)
        o.data.materials.append(mat)
        o.matrix_world = M
        if thick:
            s = o.modifiers.new("thick", "SOLIDIFY")
            s.thickness, s.offset = thick, offset
        return o

    xs, zs = np.meshgrid(np.linspace(0, w, 90), np.linspace(0, h, 110))
    cover = paper_cover("volume_cover", c["colour"])
    grid("volume_front", xs, face(xs, zs), zs, cover, .0005)
    grid("volume_back", xs, face(xs, zs) + t, zs, cover, .0005, 1.0)
    # the sheets between them, where they show: at the back, the head and the free edge
    sheets = pages_material(30, (.70, .60, .43), "Y")
    across = np.linspace(.0005, t - .0005, 5)
    az, ay = np.meshgrid(np.linspace(0, h, 60), across, indexing="ij")
    for name, x in (("volume_right", w - .00012), ("volume_left", .00025)):
        grid(name, np.full(az.shape, x), face(x, az) + ay, az, sheets)
    ax, ay = np.meshgrid(np.linspace(0, w, 40), across, indexing="ij")
    grid("volume_head", ax, face(ax, h - .00025) + ay, np.full(ax.shape, h - .00025), sheets)
    # the slip, on a shade of skew
    sw, sh, sl, st = c["slip"]
    sx, sz = np.meshgrid(np.linspace(sl, sl + sw, 8), np.linspace(h - st - sh, h - st, 50))
    sx = sx + (sz - h / 2) * .004
    grid("volume_slip", sx, face(sx, sz) - .00026, sz, noisy(printed("slip_volume", "slip_volume.png", .72, .3), 1600, .25, .0001, 4), .0002, 1.0)
    # silk round the corners at the back: over the cover, the back and the end
    silk = silk_cloth("volume_silk", c["silk"])
    lift = .00016
    corner_w, corner_h = .0105, .0155
    for n, (z0, z1, end) in enumerate(((h - corner_h, h + lift, h + lift), (-.00002, corner_h, None))):
        x, z = np.meshgrid(np.linspace(w - corner_w, w + lift, 6), np.linspace(z0, z1, 8))
        grid(f"corner_front{n}", x, face(x, z) - lift, z, silk)
        z, y = np.meshgrid(np.linspace(z0, z1, 8), np.linspace(-lift, t + lift, 4), indexing="ij")
        grid(f"corner_side{n}", np.full(z.shape, w + lift), face(w, z) + y, z, silk)
        if end:
            x, y = np.meshgrid(np.linspace(w - corner_w, w + lift, 6), np.linspace(-lift, t + lift, 4), indexing="ij")
            grid(f"corner_end{n}", x, face(x, h) + y, np.full(x.shape, end), silk)
    # the stitching
    thread = surface("volume_thread", c["thread"], .5, 0, .4)
    hole = surface("volume_hole", (.008, .008, .01), .8)
    r = .00034
    on = lambda x, z, off=r * .8: tuple(M @ Vector((x, float(face(x, z)) - lift * (x > w - corner_w and (z > h - corner_h or z < corner_h)) - off, z)))
    round_back = lambda x, z: [tuple(M @ Vector((w + lift + r * .8, float(face(w, z)) + y, z))) for y in (.0008, t * .5, t - .0008)] + [tuple(M @ Vector((w - .002, float(face(w, z)) + t + lift + r * .8, z)))]
    holes = (.024, h * .35, h * .65, h - .024)
    for n, z in enumerate(holes):
        fy = float(face(x_s, z))
        slab(f"hole{n}", x_s - .00055, x_s + .00055, z - .00055, z + .00055, -fy, -fy + .00005, .0005, hole, 0, F, n=3)
        tube(f"across{n}", [on(x_s, z, 0.0), on(x_s + .002, z), on(w - .004, z), on(w - .0005, z)] + round_back(z=z, x=0), r, thread, 2)
        if n < 3:
            tube(f"along{n}", [on(x_s, z, 0.0)] + [on(x_s, q) for q in np.linspace(z + .004, holes[n + 1] - .004, 7)] + [on(x_s, holes[n + 1], 0.0)], r, thread, 2)
    for n, (z, end) in enumerate(((holes[0], -lift), (holes[3], h + lift))):
        out = 1 if end > 0 else -1
        y0 = float(face(x_s, max(min(end, h), 0)))
        tube(f"over{n}", [on(x_s, z, 0.0), on(x_s, z + out * .004), on(x_s, end - out * .002)] + [tuple(M @ Vector((x_s, y0 + y, end + out * r * .8))) for y in (.0005, t * .5, t - .0005)] + [tuple(M @ Vector((x_s, y0 + t + lift + r * .8, end - out * .002)))], r, thread, 2)


def paper_book(i, b, z):
    """A paperback lying flat: a sheet of thin card glued round the pages, scored along the back, and its back cracked across where it has been held open."""
    w, d, h = b["w"], b["d"], b["h"]
    M = placed(b, z)
    c = .00035
    card = surface(f"book{i}_card", b["colour"], .4, 0, .45)
    xs, ys = np.meshgrid(np.linspace(-w / 2, w / 2, 40), np.linspace(-d / 2 + .0012, d / 2, 110))
    lift = .0012 * sm_((xs / w + .5 - .55) / .45) * sm_((ys / d + .5 - .35) / .65)
    zs = h - c / 2 - .0004 * np.exp(-((ys + d / 2 - .0072) / .0007) ** 2) + lift
    lid = mesh_grid(f"book{i}_lid", xs, ys, zs)
    lid.data.materials.append(card)
    lid.matrix_world = M
    thick = lid.modifiers.new("card", "SOLIDIFY")
    thick.thickness, thick.offset = c, 0.0
    box(f"book{i}_under", -w / 2, w / 2, -d / 2 + .0012, d / 2, 0, c, card, 0, M)
    box(f"book{i}_block", -w / 2 + .0002, w / 2 - .00015, -d / 2 + .0007, d / 2 - .0002, c, h - c, pages_material(i + 1, (.80, .78, .71)), 0, M)
    # the back, square but for its two soft edges and a very slight belly
    hc, hs = h / 2, h / 2 - .0012
    corner = [(.0024, h - c / 2), (.0012, h - c / 2 - .00002), (.0005, h - c / 2 - .0003), (.00008, h - c / 2 - .0011)]
    run = [(-.00032 * (1 - (t / hs) ** 2), hc + t) for t in np.linspace(hs, -hs, 32)]
    yz = corner + run + [(y, h - zz) for y, zz in reversed(corner)]
    verts, uv = [], []
    for y, zz in yz:
        verts += [(-w / 2, -d / 2 + y, zz), (w / 2, -d / 2 + y, zz)]
        uv += [(0, zz / h), (1, zz / h)]
    look = surface(f"book{i}_backcard", b["colour"], .4, 0, .45)
    nt = look.node_tree
    at = nt.nodes.new("ShaderNodeUVMap").outputs[0]
    ink, crack = picture_in(nt, f"title_{i}.png", at), picture_in(nt, f"crease_{i}.png", at)
    nt.links.new(blended(nt, ink, blended(nt, crack, b["colour"], (.86, .86, .86), "MULTIPLY"), (.012, .012, .014)), nt.nodes["Principled BSDF"].inputs["Base Color"])
    bumped(nt, calc(nt, "MULTIPLY", crack, -.00024), nt.nodes["Principled BSDF"])
    faces = [(2 * j, 2 * j + 2, 2 * j + 3, 2 * j + 1) for j in range(len(yz) - 1)]
    back = mesh_object(f"book{i}_back", verts, faces, look, smooth=True, uv=uv, matrix=M)
    thick = back.modifiers.new("card", "SOLIDIFY")
    thick.thickness, thick.offset = c, 0.0


def cup():
    """A plain white cup with what is left of the tea in it. A little way off to the right, where a right hand leaves it."""
    x, y = .395, .205
    glaze = surface("glaze", (.82, .80, .74), .16, 0, .5, .4)
    prof = [(.0, .0), (.033, .0), (.0365, .004), (.0405, .022), (.0415, .06), (.0415, .085), (.0405, .0885), (.0385, .0885), (.038, .085), (.038, .02), (.034, .008), (.0, .006)]
    lathe("cup", prof, x, y, ZT, glaze)
    tea = surface("tea", (.14, .055, .012), .05, 0, .5)
    lathe("tea", [(.0, .054), (.0379, .054)], x, y, ZT, tea, 72)
    stain = surface("stain", (.34, .22, .12), .35)
    lathe("stain", [(.03785, .054), (.03785, .0595)], x, y, ZT, stain, 72)
    # the handle, turned away from the screen and towards the sitter's right
    a = math.radians(-58)
    ux, uy = math.cos(a), math.sin(a)
    pts = []
    for i in range(9):
        q = i / 8 * math.pi
        rr = .040 + .021 * math.sin(q)
        pts.append((x + ux * rr, y + uy * rr, ZT + .045 + .027 * math.cos(q)))
    tube("handle", pts, .0052, glaze, 3)


def things():
    # the texts the film uses, lying flat, the oldest at the bottom
    z = ZT
    for i, b in enumerate(BOOKS):
        dict(case=case_book, paper=paper_book)[b["kind"]](i, b, z)
        z += b["h"]
    slipcase(CASE)
    stitched(VOLUME)
    # a notebook left open, its pen across it
    M = Matrix.Translation((-.385, .165, ZT)) @ Matrix.Rotation(math.radians(-13), 4, "Z")
    pg = printed("notebook", "notebook.png", .75, .25)
    slab("notebook", -.074, .074, -.102, .102, 0, .0075, .004, pg, .0004, M, n=3, uv=(-.074, .074, -.102, .102))
    pen = surface("pen", (.02, .02, .025), .3, .3)
    P = lambda px, py, pz: tuple(M @ Vector((px, py, pz)))
    tube("pen", [P(-.052, -.052, .0105), P(.0, .0, .0105), P(.052, .058, .0105)], .0042, pen, 3)
    tip = surface("tip", (.6, .58, .52), .25, 1.0)
    tube("tip", [P(.052, .058, .0105), P(.0585, .0655, .0105)], .0022, tip, 2)
    cup()
    # a letter opened and put down again, half under the notebook
    paper = surface("letter", (.80, .79, .74), .85)
    M = Matrix.Translation((-.495, .135, ZT + .0007)) @ Matrix.Rotation(math.radians(11), 4, "Z")
    slab("letter", -.105, .105, -.074, .074, 0, .0004, .0006, paper, 0, M, n=1)
    magic_keyboard(DISP_X, .135, math.radians(-1.0))
    magic_mouse(.262, .125, math.radians(-6.0))
    mac_studio(.615, .50, math.radians(1.5))


def neutral_in_metal():
    """Metal has no colour of its own but the room's, and here that is wood and warm plaster: aluminium would turn gold. So whatever a glossy ray lands on shows its brightness without its colour, and a little more of it: the room behind the camera is dim for a silver as bright as this."""
    for m in bpy.data.materials:
        b = m.node_tree.nodes.get("Principled BSDF") if m.node_tree else None
        if b is None or b.inputs["Metallic"].default_value > .5:
            continue
        nt = m.node_tree
        into = b.inputs["Base Color"]
        if into.is_linked:
            source = into.links[0].from_socket
        else:
            rgb = nt.nodes.new("ShaderNodeRGB")
            rgb.outputs[0].default_value = into.default_value
            source = rgb.outputs[0]
        ray, grey, more, mix = nt.nodes.new("ShaderNodeLightPath"), nt.nodes.new("ShaderNodeRGBToBW"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        more.operation = "MULTIPLY"
        more.inputs[1].default_value = 1.55
        nt.links.new(source, grey.inputs[0])
        nt.links.new(grey.outputs[0], more.inputs[0])
        nt.links.new(ray.outputs["Is Glossy Ray"], mix.inputs[0])
        nt.links.new(source, mix.inputs[6])
        nt.links.new(more.outputs[0], mix.inputs[7])
        nt.links.new(mix.outputs[2], into)


def sm_(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- light

def set_light_world(strength):
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (.92, .95, 1.0, 1)
    bg.inputs[1].default_value = strength


def daylight():
    """The sun, low in the afternoon, comes in over the left part of the sill and lies across the far left of the table."""
    d = bpy.data.lights.new("sun", "SUN")
    d.energy, d.angle = 5.5, math.radians(1.2)
    d.color = (1.0, .90, .76)
    o = link(bpy.data.objects.new("sun", d))
    o.rotation_euler = Vector((.50, -.62, -.60)).to_track_quat("-Z", "Y").to_euler()
    # a portal in the window, so the sky's light is found
    x0, x1, s, h = WIN
    p = bpy.data.lights.new("portal", "AREA")
    p.shape, p.size, p.size_y = "RECTANGLE", x1 - x0, h - s
    p.cycles.is_portal = True
    po = link(bpy.data.objects.new("portal", p))
    po.location = ((x0 + x1) / 2, WALL_Y + .17, (s + h) / 2)
    po.rotation_euler = (math.radians(90), 0, 0)
    # the rest of a room: light off a floor and a far wall that a single window does not quite reach
    f = bpy.data.lights.new("fill", "AREA")
    f.shape, f.size, f.size_y = "RECTANGLE", 2.4, 1.6
    f.energy, f.color = 36.0, (1.0, .97, .93)
    fo = link(bpy.data.objects.new("fill", f))
    fo.visible_glossy = False
    fo.location = (.4, -1.5, 2.2)
    fo.rotation_euler = Vector((-.2, .75, -.6)).to_track_quat("-Z", "Y").to_euler()
    return o, po, fo


# ---------------------------------------------------------------- rendering

def plate_name(name):
    return f"{name}_p{POSE}" if POSE else name


def save_plate(name, path_exr):
    im = bpy.data.images.load(path_exr)
    w, h = im.size
    a = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(a)
    a = a.reshape(h, w, 4)[::-1]
    np.save(os.path.join(PLATES, f"{SHOT}_{name}.npy"), a[..., :3].astype(np.float16))
    bpy.data.images.remove(im)
    os.remove(path_exr)


def render(name):
    sc = bpy.context.scene
    t0 = time.time()
    path = os.path.join(PLATES, f"{SHOT}_{name}.exr")
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    save_plate(plate_name(name) if name in ("sky", "sun") else name, path)
    print(f"{SHOT} {name}: {time.time() - t0:.0f} s", flush=True)


def main():
    os.makedirs(PLATES, exist_ok=True)
    sc = reset(SAMPLES, PCT)
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.exr_codec = "ZIP"
    sc.view_settings.view_transform = "Raw"
    sc.cycles.max_bounces = 8
    sc.cycles.diffuse_bounces = 5
    sc.cycles.glossy_bounces = 4
    sc.cycles.transparent_max_bounces = 12
    sc.cycles.transmission_bounces = 4
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.blur_glossy = 1.0
    sc.cycles.adaptive_threshold = .012

    room()
    table()
    screen, corners, lid = display()
    things()
    sunlight, portal, fill = daylight()
    set_light_world(1.0)
    neutral_in_metal()
    everything = [o for o in bpy.data.objects if o.type in ("MESH", "CURVE")]

    shot = SHOTS[SHOT]
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens, cam_data.clip_start, cam_data.clip_end = shot["lens"], .02, 60
    cam_data.sensor_width = 36.0
    cam = link(bpy.data.objects.new("cam", cam_data))
    sc.camera = cam
    if "at" in shot:
        target = on_screen(lid, *shot["at"])
        normal = (lid.to_3x3() @ Vector((0, 0, 1))).normalized()
        focus = shot["width"] * SCREEN_W * shot["lens"] / 36
        cam.location = target + normal * focus
        cam.rotation_euler = (-normal).to_track_quat("-Z", "Y").to_euler()
    else:
        loc = Vector(shot["loc"])
        cam.location = loc
        cam.rotation_euler = (math.radians(90), 0, 0)
        focus = DISP_Y - loc.y
        wide = focus * 36 / shot["lens"]
        cam_data.shift_x = (shot["centre"][0] - loc.x) / wide
        cam_data.shift_y = (shot["centre"][1] - loc.z) / wide
    bpy.context.view_layer.update()
    print("camera", tuple(cam.location), "focus", focus, flush=True)
    cam_data.dof.use_dof = True
    cam_data.dof.focus_distance = focus - shot.get("forward", 0.0)
    cam_data.dof.aperture_fstop = shot["fstop"]
    cam_data.dof.aperture_blades = 7

    # where the screen falls in the frame
    from bpy_extras.object_utils import world_to_camera_view
    W, H = sc.render.resolution_x, sc.render.resolution_y
    frame = []
    depth = []
    for p in corners:
        v = world_to_camera_view(sc, cam, Vector(p))
        frame.append((v.x * W, (1 - v.y) * H))
        depth.append(v.z)
    json.dump(dict(corners=frame, depth=depth, focus=focus, lens=shot["lens"], fstop=shot["fstop"], sensor=36.0, width=W, height=H), open(os.path.join(PLATES, f"{SHOT}.json"), "w"), indent=1)
    print("screen in frame", frame, flush=True)

    bg = sc.world.node_tree.nodes["Background"]
    outside = bpy.data.objects["outside"]
    emission = screen.data.materials[0].node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
    out_strength = outside.data.materials[0].node_tree.nodes["Emission"].inputs[1]
    keep = out_strength.default_value

    def lights(sky, sun_on, glow):
        bg.inputs[1].default_value = 1.0 if sky else 0.0
        out_strength.default_value = keep if sky else 0.0
        fill.hide_render = not sky
        portal.hide_render = not sky
        sunlight.hide_render = not sun_on
        emission.default_value = 1.0 if glow else 0.0

    for name in PASSES:
        for o in everything:
            o.is_holdout = False
            o.visible_camera = True
            o.hide_render = False
        sc.cycles.samples = SAMPLES
        sc.cycles.use_denoising = True
        if name == "sky":
            lights(True, False, False)
            render("sky")
        elif name == "sun":
            lights(False, True, False)
            render("sun")
        elif name == "glow":
            lights(False, False, True)
            render("glow")
        elif name == "mask":
            lights(False, False, False)
            black = bpy.data.materials.new("nothing")
            black.use_nodes = True
            black.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0, 0, 0, 1)
            black.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0
            white = bpy.data.materials.new("all")
            white.use_nodes = True
            nt = white.node_tree
            nt.nodes.clear()
            e, oo = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
            e.inputs[1].default_value = 1.0
            nt.links.new(e.outputs[0], oo.inputs[0])
            saved = {}
            for o in everything:
                saved[o.name] = list(o.data.materials)
                o.data.materials.clear()
                o.data.materials.append(white if o is screen else black)
            sc.cycles.use_denoising = False
            sc.cycles.samples = max(SAMPLES, 128)
            sc.cycles.use_adaptive_sampling = False
            render("mask")
            sc.cycles.use_adaptive_sampling = True
            for o in everything:
                o.data.materials.clear()
                for m in saved[o.name]:
                    o.data.materials.append(m)
    print("done", flush=True)


if __name__ == "__main__":
    main()
