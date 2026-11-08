"""The machine of the press shot, built in Blender from nothing but code: a 19th-century cast-iron power press, millimetres, sheet's top at z = 0, front towards -y.

Two open cast side frames on A-shaped legs, stayed by girders and tie rods, the floor showing through below; a machined table between them, two polished guide posts, a head that slides straight up and down on them carrying an iron chase locked up with wooden wedges and bolts,
a connecting rod and a crank above it, a pair of gears, and outside the left frame a spoked flywheel on its own pedestal, whose belt runs up out of frame. It stands 1.8 m tall in an old printing house: worn boards, rough plaster, a type case and a drying line in the dark behind it.
Parts that move come back as objects whose origin is their pivot. film_press3d.py animates them. Every shape is a bevelled prism or a lofted section; surfaces (cast skin, chipped paint, oil, wear on the edges) are in the materials. The room's textures come from film_press_room.py.
"""
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import film_press_room as ROOM  # noqa: E402

# ---- dimensions the animation also needs ----------------------------------------------------------------------------------------------------------------
CRANK_R = 38.0          # stroke 76
ROD_L = 54.0
WRIST_H = 48.0          # wrist pin, above the printing plane, when the head rests on the sheet
SLACK = 7.7             # the pin rides in a slot: the head stays down on the sheet while the crank goes through the bottom of its turn
CRANK_Z = WRIST_H + ROD_L + CRANK_R
ROD_Y = 0.0             # the crankshaft runs left to right above the head, at this y
GEAR_R = 38.0
WHEEL_Z = CRANK_Z - 2 * GEAR_R
WHEEL_R = 112.0
WHEEL_X = -250.0        # outside the left frame, where a camera on the right sees all of it against the wall
TABLE_TOP = -1.15
POST_X, POST_Y = 114.0, 92.0
FLOOR = ROOM.FLOOR_Z


# the room's furniture
CASE = dict(x0=170.0, w=262.0, depth=135.0, drawers=18, drawer=20.0, plinth=26.0, ajar=11)
LINE = dict(x0=-1700.0, x1=900.0, y=340.0, z=318.0, sag=30.0, sheets=((-1190.0, .01), (-1040.0, -.012), (-890.0, .008), (-745.0, -.006)))


# ---- node helpers --------------------------------------------------------------------------------------------------------------------------------------
def _link(nt, value, socket):
    if hasattr(value, "links"):
        nt.links.new(value, socket)
    else:
        socket.default_value = value


def nmath(nt, op, a, b=0.0, c=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    _link(nt, a, n.inputs[0])
    _link(nt, b, n.inputs[1])
    if c is not None:
        _link(nt, c, n.inputs[2])
    return n.outputs[0]


def nrange(nt, v, a, b, c=0.0, d=1.0):
    n = nt.nodes.new("ShaderNodeMapRange")
    n.interpolation_type = "SMOOTHSTEP"
    _link(nt, v, n.inputs["Value"])
    n.inputs["From Min"].default_value, n.inputs["From Max"].default_value = a, b
    n.inputs["To Min"].default_value, n.inputs["To Max"].default_value = c, d
    return n.outputs[0]


def nmix(nt, f, a, b, blend="MIX"):
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = "RGBA"
    n.blend_type = blend
    _link(nt, f, n.inputs["Factor"])
    _link(nt, a, n.inputs["A"])
    _link(nt, b, n.inputs["B"])
    return n.outputs["Result"]


def nmixf(nt, f, a, b):
    n = nt.nodes.new("ShaderNodeMix")
    _link(nt, f, n.inputs["Factor"])
    _link(nt, a, n.inputs["A"])
    _link(nt, b, n.inputs["B"])
    return n.outputs["Result"]


def nnoise(nt, vec, scale, detail=4.0, rough=.5):
    n = nt.nodes.new("ShaderNodeTexNoise")
    nt.links.new(vec, n.inputs["Vector"])
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    return n.outputs[0]


def nao(nt, distance, inside=False, samples=6):
    n = nt.nodes.new("ShaderNodeAmbientOcclusion")
    n.inputs["Distance"].default_value = distance
    n.inside = inside
    n.only_local = inside
    n.samples = samples
    return n.outputs["AO"]


def nbump(nt, height, strength, distance):
    n = nt.nodes.new("ShaderNodeBump")
    n.inputs["Strength"].default_value = strength
    n.inputs["Distance"].default_value = distance
    nt.links.new(height, n.inputs["Height"])
    return n.outputs[0]


def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    return m, nt, nt.nodes["Principled BSDF"]


def obj_coords(nt, scale=None):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    if scale is None:
        return tc.outputs["Object"]
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = scale
    nt.links.new(tc.outputs["Object"], mp.inputs[0])
    return mp.outputs[0]


# ---- materials -----------------------------------------------------------------------------------------------------------------------------------------
def cast_material(name, paint=(.022, .022, .023), bare=(.22, .22, .215), chip=.5, metallic=.35, skin=.6, scuffed=.55):
    """Cast iron under old paint: a sand-cast skin with pits and dents, paint worn through at the edges and in scuffs down to dull metal, grime in the corners, patches of oil."""
    m, nt, b = new_mat(name)
    p = obj_coords(nt)
    fine, mid, big = nnoise(nt, p, 1.3, 6, .55), nnoise(nt, p, .17, 5, .55), nnoise(nt, p, .022, 3, .5)
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    nt.links.new(p, vor.inputs["Vector"])
    vor.inputs["Scale"].default_value = .55
    pits = nmath(nt, "MULTIPLY", nrange(nt, vor.outputs["Distance"], .02, .11, 1.0, 0.0), nrange(nt, nnoise(nt, p, .11, 3, .5), .56, .68))    # only here and there, never in a grid
    edge = nmath(nt, "SUBTRACT", 1.0, nao(nt, 1.3, inside=True))
    wear = nrange(nt, nmath(nt, "ADD", edge, nmath(nt, "MULTIPLY", nmath(nt, "SUBTRACT", mid, .5), .85)), chip * .62, chip * .62 + .16)
    scuff = nrange(nt, nnoise(nt, obj_coords(nt, (.05, .05, .3)), 1.0, 6, .6), .71, .77, 0.0, scuffed)
    wear = nmath(nt, "MAXIMUM", wear, scuff)
    grime = nmath(nt, "SUBTRACT", 1.0, nao(nt, 7.0))
    oil = nrange(nt, big, .50, .74)
    paint_col = nmix(nt, nrange(nt, big, .35, .65), tuple(c * .72 for c in paint) + (1,), tuple(c * 1.3 for c in paint) + (1,))
    col = nmix(nt, wear, paint_col, tuple(bare) + (1,))
    col = nmix(nt, nmath(nt, "MULTIPLY", grime, .75), col, (.006, .006, .006, 1))
    col = nmix(nt, nmath(nt, "MULTIPLY", oil, .35), col, (.012, .011, .010, 1))
    nt.links.new(col, b.inputs["Base Color"])
    nt.links.new(nmixf(nt, wear, 0.0, metallic + .35), b.inputs["Metallic"])
    rough = nmixf(nt, wear, nrange(nt, mid, .35, .65, .40, .58), .42)
    rough = nmixf(nt, nmath(nt, "MULTIPLY", oil, .8), rough, .26)
    nt.links.new(rough, b.inputs["Roughness"])
    h = nmath(nt, "ADD", nmath(nt, "MULTIPLY", fine, .5), nmath(nt, "ADD", nmath(nt, "MULTIPLY", mid, 1.2), nmath(nt, "MULTIPLY", pits, -.35)))
    nt.links.new(nbump(nt, h, skin, .4), b.inputs["Normal"])
    b.inputs["Specular IOR Level"].default_value = .45
    return m


def table_material():
    """The table's machined top: iron planed long ago, dull, a neutral grey, the faintest long tool marks, ink ground into it in broad stains. Neutral, so that the paper is the only warm white."""
    m, nt, b = new_mat("table")
    p = obj_coords(nt)
    marks = nnoise(nt, obj_coords(nt, (.006, 2.4, 1)), 1.0, 3, .45)
    big = nnoise(nt, p, .012, 4, .55)
    stain = nrange(nt, nnoise(nt, p, .02, 7, .66), .52, .57)
    col = nmix(nt, nrange(nt, marks, .40, .60), (.070, .071, .074, 1), (.082, .083, .086, 1))
    col = nmix(nt, nmath(nt, "MULTIPLY", stain, .32), col, (.024, .024, .026, 1))
    edge = nmath(nt, "SUBTRACT", 1.0, nao(nt, 1.2, inside=True))
    col = nmix(nt, nrange(nt, edge, .18, .4), col, (.20, .20, .20, 1))
    nt.links.new(col, b.inputs["Base Color"])
    b.inputs["Metallic"].default_value = .32
    nt.links.new(nmixf(nt, stain, nrange(nt, marks, .35, .65, .46, .54), .66), b.inputs["Roughness"])
    nt.links.new(nbump(nt, marks, .008, .05), b.inputs["Normal"])
    return m


def steel_material(name="steel", col=(.70, .70, .69), rough=.13):
    """Polished, oiled steel: guide posts and shafts. Fine streaks along the length, a little dirt collecting."""
    m, nt, b = new_mat(name)
    streak = nnoise(nt, obj_coords(nt, (2.6, 2.6, .012)), 1.0, 6, .6)
    dirt = nnoise(nt, obj_coords(nt), .05, 5, .6)
    b.inputs["Metallic"].default_value = 1.0
    nt.links.new(nmix(nt, nrange(nt, dirt, .58, .68), tuple(col) + (1,), (.28, .28, .27, 1)), b.inputs["Base Color"])
    nt.links.new(nmixf(nt, nrange(nt, dirt, .58, .68), nrange(nt, streak, .3, .7, rough, rough + .1), .4), b.inputs["Roughness"])
    nt.links.new(nbump(nt, streak, .03, .05), b.inputs["Normal"])
    return m


def machined_material():
    """Gears, the crank arm's faces: bare dark iron, oiled, shiny where it works."""
    m, nt, b = new_mat("machined")
    p = obj_coords(nt)
    n = nnoise(nt, p, .25, 5, .55)
    edge = nmath(nt, "SUBTRACT", 1.0, nao(nt, 1.0, inside=True))
    col = nmix(nt, nrange(nt, edge, .15, .35), (.075, .075, .078, 1), (.44, .44, .43, 1))
    nt.links.new(nmix(nt, nmath(nt, "MULTIPLY", nrange(nt, n, .5, .6), .6), col, (.012, .012, .012, 1)), b.inputs["Base Color"])
    b.inputs["Metallic"].default_value = .9
    nt.links.new(nrange(nt, n, .35, .65, .24, .42), b.inputs["Roughness"])
    nt.links.new(nbump(nt, nnoise(nt, p, 2.0, 4, .5), .08, .1), b.inputs["Normal"])
    return m


def fastener_material():
    """Bolts and washers: dark oxidised steel, dull, a little brighter where spanners have been."""
    m, nt, b = new_mat("fastener")
    p = obj_coords(nt)
    n = nnoise(nt, p, .6, 5, .55)
    edge = nmath(nt, "SUBTRACT", 1.0, nao(nt, .8, inside=True))
    col = nmix(nt, nrange(nt, n, .4, .62), (.05, .05, .052, 1), (.095, .095, .098, 1))
    col = nmix(nt, nrange(nt, edge, .2, .42), col, (.26, .26, .25, 1))
    nt.links.new(col, b.inputs["Base Color"])
    b.inputs["Metallic"].default_value = .7
    nt.links.new(nrange(nt, n, .35, .65, .46, .62), b.inputs["Roughness"])
    nt.links.new(nbump(nt, nnoise(nt, p, 3.0, 4, .5), .06, .05), b.inputs["Normal"])
    return m


def wood_material(along="x", inked=.8):
    """Oak furniture and wedges, old, darkened by ink and hands; the grain runs along x or y, end grain crushed by the mallet."""
    m, nt, b = new_mat("wood_" + along)
    swap = (lambda a, c: (a, 1, c)) if along == "x" else (lambda a, c: (1, a, c))
    g = nnoise(nt, obj_coords(nt, swap(.07, 1.7)), 1.0, 7, .62)
    g2 = nnoise(nt, obj_coords(nt, (.02, .5, .55) if along == "x" else (.5, .02, .55)), 1.0, 3, .5)
    ink = nnoise(nt, obj_coords(nt), .16, 5, .6)
    col = nmix(nt, nrange(nt, g, .34, .62), (.040, .025, .015, 1), (.135, .085, .048, 1))
    col = nmix(nt, nmath(nt, "MULTIPLY", nrange(nt, g2, .50, .64), .7), col, (.016, .013, .011, 1))
    col = nmix(nt, nmath(nt, "MULTIPLY", nrange(nt, ink, .46, .6), inked), col, (.012, .011, .011, 1))     # decades of ink and hands
    col = nmix(nt, nmath(nt, "MULTIPLY", nmath(nt, "SUBTRACT", 1.0, nao(nt, 1.0)), .8), col, (.006, .005, .005, 1))
    nt.links.new(col, b.inputs["Base Color"])
    nt.links.new(nrange(nt, g, .3, .7, .46, .68), b.inputs["Roughness"])
    nt.links.new(nbump(nt, g, .35, .12), b.inputs["Normal"])
    return m


def type_material():
    """The underside of the forme: silver type metal, the printing faces black with ink. Reads the "ink" and "tone" attributes: tone is the bare metal's lightness, near 0 in the seams, about .35 on leads and quads, .7 to 1 on a sort's shoulder."""
    m, nt, b = new_mat("type")
    ink, tone = nt.nodes.new("ShaderNodeAttribute"), nt.nodes.new("ShaderNodeAttribute")
    ink.attribute_name, tone.attribute_name = "ink", "tone"
    p = obj_coords(nt)
    n = nnoise(nt, p, .12, 5, .5)
    rub = nnoise(nt, obj_coords(nt, (.08, 2.2, 1)), 1.0, 5, .6)     # rubbed along the lines by decades of wiping
    metal = nmix(nt, nrange(nt, n, .35, .65), (.725, .725, .71, 1), (.745, .745, .73, 1))      # hardly any drift at this scale: more of it reads as clouds reflected in the metal
    metal = nmix(nt, nmath(nt, "MULTIPLY", nrange(nt, rub, .54, .68), .22), metal, (.46, .46, .45, 1))
    col = nmix(nt, tone.outputs["Fac"], (.02, .02, .021, 1), metal)
    nt.links.new(nmix(nt, ink.outputs["Fac"], col, (.004, .004, .005, 1)), b.inputs["Base Color"])
    nt.links.new(nmixf(nt, ink.outputs["Fac"], .5, .18), b.inputs["Specular IOR Level"])
    nt.links.new(nmixf(nt, ink.outputs["Fac"], .95, 0.0), b.inputs["Metallic"])
    nt.links.new(nmixf(nt, ink.outputs["Fac"], nmath(nt, "ADD", nrange(nt, n, .35, .65, .20, .225), nmath(nt, "MULTIPLY", nrange(nt, rub, .5, .7), .07)), .5), b.inputs["Roughness"])
    nt.links.new(nbump(nt, nmath(nt, "ADD", n, nmath(nt, "MULTIPLY", rub, .4)), .012, .02), b.inputs["Normal"])
    return m


def leather_material(cut_z=None):
    """The belt. With cut_z, everything below that height in the world is not there: the strand runs on past where it leaves the wheel."""
    m, nt, b = new_mat("belt")
    p = obj_coords(nt)
    n = nnoise(nt, p, .35, 6, .6)
    col = nmix(nt, nrange(nt, n, .3, .7), (.028, .020, .015, 1), (.042, .030, .022, 1))
    nt.links.new(col, b.inputs["Base Color"])
    nt.links.new(nrange(nt, n, .35, .65, .42, .6), b.inputs["Roughness"])
    nt.links.new(nbump(nt, n, .05, .2), b.inputs["Normal"])
    if cut_z is not None:
        g = nt.nodes.new("ShaderNodeNewGeometry")
        s = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(g.outputs["Position"], s.inputs[0])
        tr, mx = nt.nodes.new("ShaderNodeBsdfTransparent"), nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(nmath(nt, "GREATER_THAN", s.outputs[2], cut_z), mx.inputs[0])
        nt.links.new(tr.outputs[0], mx.inputs[1])
        nt.links.new(b.outputs[0], mx.inputs[2])
        nt.links.new(mx.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    return m


def image_node(nt, path, uv, colour=False, extension="EXTEND"):
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(path, check_existing=True)
    n.image.colorspace_settings.name = "sRGB" if colour else "Non-Color"
    n.extension = extension
    nt.links.new(uv, n.inputs["Vector"])
    return n


def baked_material(name, maps, stem, height, strength=1.0):
    """A surface from film_press_room.py: colour, and a map with height in red and roughness in green. UV-mapped."""
    m, nt, b = new_mat(name)
    uv = nt.nodes.new("ShaderNodeTexCoord").outputs["UV"]
    nt.links.new(image_node(nt, os.path.join(maps, stem + "_col.png"), uv, colour=True).outputs["Color"], b.inputs["Base Color"])
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(image_node(nt, os.path.join(maps, stem + "_hr.png"), uv).outputs["Color"], sep.inputs[0])
    nt.links.new(sep.outputs[1], b.inputs["Roughness"])
    nt.links.new(nbump(nt, sep.outputs[0], strength, height), b.inputs["Normal"])
    b.inputs["Specular IOR Level"].default_value = .35
    return m


def brass_material():
    """Drawer pulls and label holders: old brass, dull, handled."""
    m, nt, b = new_mat("brass")
    n = nnoise(nt, obj_coords(nt), .4, 5, .55)
    nt.links.new(nmix(nt, nrange(nt, n, .4, .62), (.17, .115, .05, 1), (.34, .25, .12, 1)), b.inputs["Base Color"])
    b.inputs["Metallic"].default_value = .95
    nt.links.new(nrange(nt, n, .35, .65, .36, .56), b.inputs["Roughness"])
    return m


def hemp_material():
    """The drying line."""
    m, nt, b = new_mat("hemp")
    n = nnoise(nt, obj_coords(nt), .9, 4, .55)
    nt.links.new(nmix(nt, nrange(nt, n, .35, .65), (.16, .13, .09, 1), (.30, .25, .18, 1)), b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = .9
    return m


def drying_paper(maps, k):
    """A printed sheet on the line, a little translucent."""
    m, nt, b = new_mat(f"dry_{k}")
    uv = nt.nodes.new("ShaderNodeTexCoord").outputs["UV"]
    tex = image_node(nt, os.path.join(maps, f"dry_{k}.png"), uv, colour=True)
    nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = .86
    tr, mx = nt.nodes.new("ShaderNodeBsdfTranslucent"), nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tex.outputs["Color"], tr.inputs["Color"])
    mx.inputs[0].default_value = .22
    nt.links.new(b.outputs[0], mx.inputs[1])
    nt.links.new(tr.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    return m


# ---- shapes --------------------------------------------------------------------------------------------------------------------------------------------
AXES = dict(y=lambda a, b, c: (a, c, b), x=lambda a, b, c: (c, a, b), z=lambda a, b, c: (a, b, c))


def circle(r, n=64, cx=0.0, cy=0.0, start=0.0):
    return [(cx + r * math.cos(start + 2 * math.pi * i / n), cy + r * math.sin(start + 2 * math.pi * i / n)) for i in range(n)]


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def rounded(corners, n=8):
    """A polygon from corners (a, b, r), each rounded to radius r. Castings have no sharp inside corners."""
    out = []
    m = len(corners)
    for i, (a, b, r) in enumerate(corners):
        if r <= 0:
            out.append((a, b))
            continue
        c = np.array((a, b), float)
        u, v = np.array(corners[i - 1][:2], float) - c, np.array(corners[(i + 1) % m][:2], float) - c
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        u, v = u / lu, v / lv
        half = math.acos(float(np.clip(u @ v, -1, 1))) / 2
        t = min(r / math.tan(half), lu * .49, lv * .49)
        r_ = t * math.tan(half)
        o = c + (u + v) / np.linalg.norm(u + v) * r_ / math.sin(half)
        a0, a1 = c + u * t - o, c + v * t - o
        a0, a1 = math.atan2(a0[1], a0[0]), math.atan2(a1[1], a1[0])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        out += [(o[0] + r_ * math.cos(a0 + da * k / n), o[1] + r_ * math.sin(a0 + da * k / n)) for k in range(n + 1)]
    return out


def inset(corners, d):
    """The same corners (a, b, r) with every side moved d inwards. What goes round a convex corner gets tighter, round a concave one wider."""
    p = np.array([c[:2] for c in corners], float)
    sign = 1.0 if np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]) > 0 else -1.0
    e0, e1 = p - np.roll(p, 1, 0), np.roll(p, -1, 0) - p
    e0 /= np.linalg.norm(e0, axis=1, keepdims=True)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    n0, n1 = np.stack([-e0[:, 1], e0[:, 0]], 1) * sign, np.stack([-e1[:, 1], e1[:, 0]], 1) * sign
    m = n0 + n1
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    q = p + m * (d / np.maximum((m * n0).sum(1), .3))[:, None]
    turn = (e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]) * sign
    return [(q[i, 0], q[i, 1], (max(c[2] - d, 1.2) if turn[i] > 0 else c[2] + d) if c[2] > 0 else 0.0) for i, c in enumerate(corners)]


def _finish(bm, name, bevel, segs, mat, parent, location, hard_angle=33.0):
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    curved = bm.faces.layers.int.new("curved")
    hard = []
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        a = e.calc_face_angle(0.0)
        if a > math.radians(hard_angle):
            hard.append(e)
        elif a > math.radians(.5):
            for f in e.link_faces:
                f[curved] = 1
    if bevel > 0 and hard:
        r = bmesh.ops.bevel(bm, geom=hard, offset=bevel, offset_type="OFFSET", segments=segs, profile=.5, affect="EDGES", clamp_overlap=True)
        for f in r["faces"]:
            f[curved] = 1
    for f in bm.faces:
        f.smooth = bool(f[curved])
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if mat is not None:
        me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    if parent is not None:
        o.parent = parent
    if location is not None:
        o.location = location
    return o


def prism(name, outline, c0, c1, holes=(), axis="y", bevel=1.0, segs=2, mat=None, parent=None, location=None):
    """A 2D outline (with holes) extruded along an axis, its sharp edges chamfered, curved sides smooth."""
    to3 = AXES[axis]
    bm = bmesh.new()
    loops = [outline] + list(holes)
    bot, top, fill_edges = {}, {}, []
    side = []
    for l, pts in enumerate(loops):
        vb = [bm.verts.new(to3(a, b, c0)) for a, b in pts]
        vt = [bm.verts.new(to3(a, b, c1)) for a, b in pts]
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            fill_edges.append(bm.edges.new((vb[i], vb[j])))
            side.append((vb[i], vb[j], vt[j], vt[i]))
        for u, v in zip(vb, vt):
            top[u] = v
    r = bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=False, edges=fill_edges)
    caps = [g for g in r["geom"] if isinstance(g, bmesh.types.BMFace)]
    for f in caps:
        bm.faces.new([top[v] for v in f.verts])
    for q in side:
        bm.faces.new(q)
    return _finish(bm, name, bevel, segs, mat, parent, location)


def cyl(name, r, c0, c1, axis="y", at=(0.0, 0.0), n=64, bevel=.6, segs=2, mat=None, parent=None, location=None, hole=None):
    holes = [list(reversed(circle(hole, n, at[0], at[1])))] if hole else ()
    return prism(name, circle(r, n, at[0], at[1]), c0, c1, holes=holes, axis=axis, bevel=bevel, segs=segs, mat=mat, parent=parent, location=location)


def box(name, x0, x1, y0, y1, z0, z1, bevel=.6, segs=2, mat=None, parent=None):
    return prism(name, rect(x0, y0, x1, y1), z0, z1, axis="z", bevel=bevel, segs=segs, mat=mat, parent=parent)


def bolt(name, x, z, y0, out=3.2, r=4.4, mat=None, washer=None, parent=None, spin=0.0):
    """A hexagon head and washer on a face looking towards -y."""
    if washer:
        cyl(name + "_w", r * 1.42, y0 - .9, y0 + .1, at=(x, z), n=32, bevel=.2, segs=1, mat=washer, parent=parent)
    o = prism(name, circle(r, 6, x, z, spin), y0 - out - .9, y0 - .8, axis="y", bevel=.42, segs=2, mat=mat, parent=parent)
    return o


def hull(a, b, ra, rb, n=20, pinch=0.0):
    """The outline around two circles on the z axis (b above a), optionally pinched at the waist."""
    pts = []
    for i in range(n + 1):
        t = math.pi + math.pi * i / n
        pts.append((a[0] + ra * math.cos(t), a[1] + ra * math.sin(t)))
    if pinch:
        pts.append(((a[0] + b[0]) / 2 + (ra + rb) / 2 * (1 - pinch), (a[1] + b[1]) / 2))
    for i in range(n + 1):
        t = math.pi * i / n
        pts.append((b[0] + rb * math.cos(t), b[1] + rb * math.sin(t)))
    if pinch:
        pts.append(((a[0] + b[0]) / 2 - (ra + rb) / 2 * (1 - pinch), (a[1] + b[1]) / 2))
    return pts


def gear_outline(r, teeth, module, phase=0.0):
    root, tip = r - 1.25 * module, r + module
    pts = []
    pitch = 2 * math.pi / teeth
    for i in range(teeth):
        a = phase + i * pitch
        for da, rr in ((-.30, root), (-.17, tip), (.17, tip), (.30, root)):
            pts.append((rr * math.cos(a + da * pitch), rr * math.sin(a + da * pitch)))
    return pts


def loft(name, rings, mat=None, parent=None):
    """A smooth body through closed rings of equal size, capped at both ends."""
    bm = bmesh.new()
    vs = [[bm.verts.new(p) for p in ring] for ring in rings]
    n = len(rings[0])
    for a, b in zip(vs, vs[1:]):
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((a[i], a[j], b[j], b[i]))
    bm.faces.new(vs[0])
    bm.faces.new(list(reversed(vs[-1])))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:
        f.smooth = True
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if mat is not None:
        me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    if parent is not None:
        o.parent = parent
    return o


def uv_quad(name, corners, uvs, mat):
    """A flat quad with its own texture coordinates."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(corners, [], [(0, 1, 2, 3)])
    layer = me.uv_layers.new(name="UV")
    for i, uv in enumerate(uvs):
        layer.data[i].uv = uv
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    return o


def empty(name, location=(0.0, 0.0, 0.0)):
    o = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(o)
    o.location = location
    return o


# ---- the machine ---------------------------------------------------------------------------------------------------------------------------------------
class Machine:
    """Builds everything. .head, .rod, .crank, .wheel, .belts are the objects to animate; .head holds the forme's face (film_press3d.py adds its mesh)."""

    def __init__(self, maps):
        self.maps = maps
        self.cast = cast_material("cast")
        self.cast2 = cast_material("cast_table", paint=(.024, .024, .025), chip=.46)
        self.cast_leg = cast_material("cast_leg", skin=.2, chip=.78, scuffed=.1)      # the same casting, seen from further off and at a grazing light, where every chip would read as hammering
        self.rim_mat = cast_material("rim", paint=(.030, .030, .031), bare=(.30, .30, .29), chip=.30, skin=.16)
        self.screw = cast_material("screw", paint=(.03, .03, .03), bare=(.075, .075, .075), chip=.9, metallic=.2, skin=.1)
        self.slot = cast_material("slot", paint=(.003, .003, .003), bare=(.003, .003, .003), skin=0.0)
        self.steel = steel_material()
        self.shaft = steel_material("shaft", col=(.55, .55, .54), rough=.2)
        self.machined = machined_material()
        self.fast = fastener_material()
        self.wood = wood_material()
        self.wood_y = wood_material("y")
        self.table_mat = table_material()
        self.room()
        self.frame()
        self.legs()
        self.head_parts()
        self.rod_parts()
        self.crank_parts()
        self.wheel_parts()

    def room(self):
        s = 6000.0
        x0, x1, y0, y1 = ROOM.FLOOR_RECT
        fu, fv = lambda x: (x - x0) / (x1 - x0), lambda y: (y - y0) / (y1 - y0)
        uv_quad("floor", [(-s, -s, FLOOR), (s, -s, FLOOR), (s, ROOM.WALL_Y, FLOOR), (-s, ROOM.WALL_Y, FLOOR)], [(fu(-s), fv(-s)), (fu(s), fv(-s)), (fu(s), fv(ROOM.WALL_Y)), (fu(-s), fv(ROOM.WALL_Y))], baked_material("floor", self.maps, "floor", ROOM.FLOOR_H))
        x0, x1, z0, z1 = ROOM.WALL_RECT
        wu, wv = lambda x: (x - x0) / (x1 - x0), lambda z: (z - z0) / (z1 - z0)
        uv_quad("wall", [(-s, ROOM.WALL_Y, FLOOR), (s, ROOM.WALL_Y, FLOOR), (s, ROOM.WALL_Y, s), (-s, ROOM.WALL_Y, s)], [(wu(-s), wv(FLOOR)), (wu(s), wv(FLOOR)), (wu(s), wv(s)), (wu(-s), wv(s))], baked_material("wall", self.maps, "wall", ROOM.WALL_H))
        self.type_case()
        self.drying_line()

    def type_case(self):
        """The cabinet the type lives in: one column of shallow drawers, each with two cup pulls and a card in a brass holder, on a plinth against the wall. Everything here is named case_, and only the wide shot has it."""
        wood, brass = wood_material("x", inked=.45), brass_material()
        labels = new_mat("case_label")
        uv = labels[1].nodes.new("ShaderNodeTexCoord").outputs["UV"]
        labels[1].links.new(image_node(labels[1], os.path.join(self.maps, "case_labels.png"), uv, colour=True).outputs["Color"], labels[2].inputs["Base Color"])
        labels[2].inputs["Roughness"].default_value = .85
        x0, x1, y0, y1 = CASE["x0"], CASE["x0"] + CASE["w"], ROOM.WALL_Y - CASE["depth"], ROOM.WALL_Y - 3
        n, dh, plinth = CASE["drawers"], CASE["drawer"], CASE["plinth"]
        top = FLOOR + plinth + n * dh
        box("case_plinth", x0 + 4, x1 - 4, y0 + 5, y1, FLOOR, FLOOR + plinth, bevel=1.0, mat=wood)
        for tag, a, b in (("l", x0, x0 + 11), ("r", x1 - 11, x1)):
            box(f"case_side_{tag}", a, b, y0, y1, FLOOR + 3, top + 2, bevel=1.1, mat=wood)
        box("case_top", x0 - 5, x1 + 5, y0 - 5, y1, top + 2, top + 13, bevel=1.6, mat=wood)
        box("case_back", x0 + 8, x1 - 8, y1 - 5, y1, FLOOR + 3, top + 2, bevel=0, mat=wood)
        r = np.random.default_rng(8)
        for k in range(n):
            z = FLOOR + plinth + k * dh
            out = float(r.uniform(-.7, 1.2)) + (9.0 if k == CASE["ajar"] else 0.0)
            yf = y0 + .5 - out
            box(f"case_drawer{k}", x0 + 11.4, x1 - 11.4, yf, y1 - 6, z + .55, z + dh - .55, bevel=.55, segs=1, mat=wood)
            for s_ in (-1, 1):
                cx = (x0 + x1) / 2 + s_ * CASE["w"] * .30
                cup = [(cx + 9.5 * math.cos(a), z + dh * .62 + 9.5 * math.sin(a) * .95) for a in np.linspace(math.pi, 2 * math.pi, 18)]
                prism(f"case_pull{k}{s_}", cup, yf - 2.8, yf + .2, axis="y", bevel=.5, segs=1, mat=brass)
            cx, cz, hw, hh = (x0 + x1) / 2, z + dh * .5, 21.0, 6.6
            prism(f"case_holder{k}", rect(cx - hw, cz - hh, cx + hw, cz + hh), yf - 1.1, yf + .1, holes=[list(reversed(rect(cx - hw + 1.6, cz - hh + 1.6, cx + hw - 1.6, cz + hh - 1.6)))], axis="y", bevel=.25, segs=1, mat=brass)
            u0, v0, u1, v1 = ROOM.label_uv(k % len(ROOM.LABELS))
            yc = yf - .25
            uv_quad(f"case_card{k}", [(cx - hw + 1.3, yc, cz - hh + 1.3), (cx + hw - 1.3, yc, cz - hh + 1.3), (cx + hw - 1.3, yc, cz + hh - 1.3), (cx - hw + 1.3, yc, cz + hh - 1.3)], [(u0, v0), (u1, v0), (u1, v1), (u0, v1)], labels[0])

    def drying_line(self):
        """A hemp line right across the back of the shop, sagging, and printed sheets thrown over it to dry. Everything named dry_, and only the wide shot has it."""
        x0, x1, zt, sag, y = LINE["x0"], LINE["x1"], LINE["z"], LINE["sag"], LINE["y"]
        line_z = lambda x: zt - sag * (1 - (2 * (x - x0) / (x1 - x0) - 1) ** 2)
        hemp, rope_r = hemp_material(), 1.5
        rings = []
        for x in np.arange(x0, x1, 1.0):
            ring = []
            for q in np.linspace(0, 2 * math.pi, 18, endpoint=False):
                rad = rope_r * (1 + .2 * math.cos(3 * q - x * 1.35))           # three strands laid round each other
                ring.append((float(x) + .25 * math.sin(q), y + rad * math.cos(q), line_z(x) + rad * math.sin(q)))
            rings.append(ring)
        loft("dry_line", rings, mat=hemp)
        for k, (cx, tilt) in enumerate(LINE["sheets"]):
            self.drying_sheet(k, cx, line_z(cx), y, tilt, rope_r)

    def drying_sheet(self, k, cx, zl, yl, tilt, rope_r):
        """A sheet folded over the line: a long strip of arc over the rope, the printed half towards the room, a little wave down its length, its corners curling."""
        W_, L = ROOM.SHEET_W, ROOM.SHEET_L
        r = np.random.default_rng(700 + k)
        nu, nv = 48, 96
        a1, a2, a3 = r.uniform(.8, 2.0, 3)
        p1, p2, p3 = r.uniform(0, 6.28, 3)
        verts, uvs = [], []
        R = rope_r + .4
        ts = np.concatenate([np.linspace(-L * .55, -.001, nv // 4), np.linspace(0, math.pi * R, 8), math.pi * R + np.linspace(.001, L, nv)])
        for t in ts:
            for u in np.linspace(-.5, .5, nu):
                x = cx + u * W_
                if t < 0:                       # the back half, hanging behind the line
                    py, pz = R + (-t) * .03, -(-t) * .999
                elif t <= math.pi * R:
                    a = t / R
                    py, pz = R * math.cos(a), R * math.sin(a)
                else:                           # the printed half, towards the room
                    s = t - math.pi * R
                    f = s / L
                    py = -R - s * .035 - f * f * a1 * 3.5 * math.sin(u * 2.2 + p1) - a2 * .9 * math.sin(f * 4.2 + u * 3.0 + p2) * f - a3 * 2.6 * (abs(u) * 2) ** 4 * f ** 2
                    pz = -s + (abs(u) * 2) ** 3 * f * 1.5
                ca, sa = math.cos(tilt), math.sin(tilt)
                verts.append((x + (pz * sa), yl + py, zl + pz * ca))
                v = 1 - (t - math.pi * R) / L if t > math.pi * R else 1.02
                uvs.append((u + .5, v))
        me = bpy.data.meshes.new(f"dry_sheet{k}")
        n = len(ts)
        faces = [(i * nu + j, i * nu + j + 1, (i + 1) * nu + j + 1, (i + 1) * nu + j) for i in range(n - 1) for j in range(nu - 1)]
        me.from_pydata(verts, [], faces)
        layer = me.uv_layers.new(name="UV")
        for poly in me.polygons:
            poly.use_smooth = True
            for li in poly.loop_indices:
                layer.data[li].uv = uvs[me.loops[li].vertex_index]
        me.materials.append(drying_paper(self.maps, k % len(ROOM.PAGES)))
        bpy.context.collection.objects.link(bpy.data.objects.new(me.name, me))

    def frame(self):
        c, c2, m = self.cast, self.cast2, self.machined
        # the table: T-slots either side of the sheet, a deep skirt below its machined top
        slot = lambda x: rect(x - 3.5, -122, x + 3.5, 52)
        prism("table", rect(-132, -132, 132, 118), -9, TABLE_TOP, holes=[list(reversed(slot(-114))), list(reversed(slot(114)))], axis="z", bevel=.5, segs=2, mat=self.table_mat)
        box("table_slots", -131, 131, -131, 117, -13, -8.6, bevel=.2, mat=c2)
        box("table_skirt", -129, 129, -129, 115, -36, -9, bevel=1.4, mat=c2)
        # the iron frame the sheet lies in, screwed down at its corners
        rx0, rx1, ry0, ry1, rw = -97.0, 97.0, -114.0, 77.0, 9.0
        prism("rim", rect(rx0, ry0, rx1, ry1), TABLE_TOP - .1, .2, holes=[list(reversed(rect(rx0 + rw, ry0 + rw, rx1 - rw, ry1 - rw)))], axis="z", bevel=.3, segs=2, mat=self.rim_mat)
        for k, (sx, sy) in enumerate(((rx0 + rw / 2, ry0 + rw / 2), (rx1 - rw / 2, ry0 + rw / 2), (rx0 + rw / 2, ry1 - rw / 2), (rx1 - rw / 2, ry1 - rw / 2), (rx0 + rw / 2, 0.0), (rx1 - rw / 2, 0.0))):
            cyl(f"rim_screw{k}", 2.7, 0.0, .45, axis="z", at=(sx, sy), n=20, bevel=.12, segs=1, mat=self.screw)
            box(f"rim_slot{k}", sx - 2.1, sx + 2.1, sy - .28, sy + .28, .38, .456, bevel=0, mat=self.slot)
        for x in (-92, -46, 0, 46, 92):
            box(f"table_rib{x}", x - 3.5, x + 3.5, -129.5, -100, -36, -11, bevel=1.0, mat=c2)
        # the upper part of the side frames, as shot B sees past the head; legs() casts them whole for the other shots. The gap keeps the window's light on the sheet, the arm above it carries the crankshaft over the head
        cz = CRANK_Z
        self.arm = arm = [(150, 196)] + [(134 + 16 * math.cos(a), 196 + 16 * math.sin(a)) for a in np.linspace(0, math.pi / 2, 10)][1:] + [(24, 212)] + [(36 * math.cos(a), cz + 36 * math.sin(a)) for a in np.linspace(math.radians(80), math.radians(262), 30)] + [(14, cz - 34)] + [(52 + 26 * math.cos(a), 84 + 26 * math.sin(a)) for a in np.linspace(math.pi / 2, 0, 10)]
        for s, tag in ((1, "r"), (-1, "l")):
            x0, x1 = sorted((s * 134, s * 152))
            prism(f"frame_b_{tag}", arm + [(78, -36), (150, -36)], x0, x1, axis="x", bevel=2.0, segs=3, mat=c)
            # the shaft's bosses, standing proud of the casting
            xb0, xb1 = sorted((s * 122, s * 164))
            cyl(f"frame_boss_{tag}", 30, xb0, xb1, axis="x", at=(0, cz), n=64, bevel=1.4, segs=2, mat=c)
            for a in (40, 130, 220, 310):
                ya, za = 23 * math.cos(math.radians(a)), cz + 23 * math.sin(math.radians(a))
                xn0, xn1 = sorted((s * 164, s * 167))
                prism(f"boss_stud_{tag}{a}", circle(3.0, 6, ya, za), xn0, xn1, axis="x", bevel=.3, segs=1, mat=m)
            for z in (-20, 34, 80, 150):
                xn0, xn1 = sorted((s * 157, s * 159.4))
                prism(f"frame_bolt_{tag}{z}", circle(3.3, 6, 116, z), xn0, xn1, axis="x", bevel=.3, segs=1, mat=m)
        # the crown tying them together behind the head
        prism("crown", [(76, 168), (150, 168), (150, 212), (92, 212), (76, 198)], -134, 134, axis="x", bevel=2.2, segs=3, mat=c)
        for x in (-90, -30, 30, 90):
            box(f"crown_rib{x}", x - 4, x + 4, 71, 90, 170, 196, bevel=1.0, mat=c)
        # the bracket hanging from the left arm, for the wheel's shaft
        prism("hanger", hull((0, WHEEL_Z), (0, cz - 16), 22, 26, n=20, pinch=.26), -152, -134, holes=[], axis="x", bevel=1.6, segs=3, mat=c)
        cyl("hanger_boss", 22, -160, -126, axis="x", at=(0, WHEEL_Z), n=48, bevel=1.2, mat=c)
        cyl("hanger_cap", 16, -126.5, -121, axis="x", at=(0, WHEEL_Z), n=40, bevel=.8, mat=m)
        for a in (30, 150, 270):
            prism(f"hanger_stud{a}", circle(2.4, 6, 12 * math.cos(math.radians(a)), WHEEL_Z + 12 * math.sin(math.radians(a))), -121.2, -119.2, axis="x", bevel=.25, segs=1, mat=m)
        # the guide posts and their collars
        for s in (-1, 1):
            x = s * POST_X
            cyl(f"post{s}", 12, -2, 178, axis="z", at=(x, POST_Y), n=96, bevel=.4, segs=2, mat=self.steel)
            cyl(f"collar{s}", 19, TABLE_TOP - .2, 7, axis="z", at=(x, POST_Y), n=64, bevel=1.1, segs=2, mat=c)
            cyl(f"collar_top{s}", 19, 158, 172, axis="z", at=(x, POST_Y), n=64, bevel=1.1, segs=2, mat=c)
            for k in range(4):
                a = math.radians(45 + 90 * k)
                prism(f"collar_stud{s}{k}", circle(2.8, 6, x + 15 * math.cos(a), POST_Y + 15 * math.sin(a)), 7, 9.2, axis="z", bevel=.3, segs=1, mat=m)

    def legs(self):
        """What carries it: each side frame cast in one piece, arm above, A-shaped legs below with a girder's depth under the table and nothing at all in the arch between them, so the floor shows through. A flange round every edge and a rib up each leg, feet bolted to the boards, girders and tie rods from one frame to the other."""
        c, c2, m = self.cast_leg, self.cast2, self.machined
        arm = [(a, b, 0.0) for a, b in self.arm]
        # from the back foot, up the back of the frame, over the arm, down into the gap, forward under the table, down the front leg, along the floor, up the arch and down to the back foot again
        foot_b, foot_f = (154, 218), (-204, -134)
        below = [(78, -14, 12), (-142, -14, 7), (-188, -280, 5), (foot_f[0], -288, 2.5), (foot_f[0], FLOOR, 0), (foot_f[1], FLOOR, 0), (foot_f[1], -288, 2.5), (-126, -280, 5), (-98, -80, 34), (100, -80, 34), (148, -280, 5), (foot_b[0], -288, 2.5), (foot_b[0], FLOOR, 0), (foot_b[1], FLOOR, 0), (foot_b[1], -288, 2.5), (206, -280, 5), (150, -18, 60)]
        corners = arm + below
        outline = rounded(corners)
        ring = rounded(inset(corners, 7.0))
        # the same flange on the inner face, below the table
        lower = [(-145, -44, 3), (-188, -280, 5), (foot_f[0], -288, 2.5), (foot_f[0], -278, 0), (foot_f[1], -278, 0), (-126, -280, 5), (-98, -80, 34), (100, -80, 34), (148, -280, 5), (foot_b[0], -278, 0), (foot_b[1], -278, 0), (foot_b[1], -288, 2.5), (206, -280, 5), (153, -44, 3)]
        low_out, low_in = rounded(lower), rounded(inset(lower, 6.5))
        for s, tag in ((1, "r"), (-1, "l")):
            x0, x1 = sorted((s * 134, s * 152))
            prism(f"cheek_{tag}", outline, x0, x1, axis="x", bevel=2.0, segs=3, mat=c)
            xr0, xr1 = sorted((s * 152, s * 157))
            prism(f"cheek_flange_{tag}", outline, xr0, xr1, holes=[list(reversed(ring))], axis="x", bevel=1.0, segs=2, mat=c)
            xi0, xi1 = sorted((s * 134, s * 128.5))
            prism(f"cheek_inner_{tag}", low_out, xi0, xi1, holes=[list(reversed(low_in))], axis="x", bevel=1.0, segs=2, mat=c)
            # a rib up the middle of each leg, on both faces
            for name, a, b in (("f", (-120, -46), (-158, -278)), ("b", (127, -46), (176, -278))):
                nv = np.array((a[1] - b[1], b[0] - a[0]), float)
                nv = nv / np.linalg.norm(nv) * 3.2
                rib = [(a[0] - nv[0], a[1] - nv[1]), (b[0] - nv[0], b[1] - nv[1]), (b[0] + nv[0], b[1] + nv[1]), (a[0] + nv[0], a[1] + nv[1])]
                for face, u0, u1 in (("o", 152, 155.6), ("i", 134, 130.4)):
                    x2, x3 = sorted((s * u0, s * u1))
                    prism(f"cheek_rib_{tag}{name}{face}", rib, x2, x3, axis="x", bevel=.9, segs=2, mat=c)
            # the feet, widened into pads, each held down by bolts through the boards
            for name, (y0, y1) in (("f", foot_f), ("b", foot_b)):
                xp0, xp1 = sorted((s * 122, s * 168))
                box(f"cheek_pad_{tag}{name}", xp0, xp1, y0 - 2, y1 + 2, FLOOR, FLOOR + 10, bevel=1.6, segs=2, mat=c2)
                for bx, by in ((xp0 + 7, y0 + 5), (xp1 - 7, y0 + 5), (xp0 + 7, y1 - 5), (xp1 - 7, y1 - 5)):
                    prism(f"cheek_bolt_{tag}{name}{bx:.0f}{by:.0f}", circle(3.2, 6, bx, by, bx * .1), FLOOR + 10, FLOOR + 12.6, axis="z", bevel=.3, segs=1, mat=self.fast)
        # two girders under the table, frame to frame
        for name, y in (("f", -109.0), ("b", 105.0)):
            box(f"girder_{name}", -135, 135, y - 7, y + 7, -74, -35, bevel=.8, mat=c2)
            box(f"girder_{name}_low", -135, 135, y - 14, y + 14, -80, -73, bevel=1.1, mat=c2)
        # tie rods low down, a collar on each side of the leg, a nut on the outside
        for name, y, z in (("f", -152.0, -238.0), ("b", 166.0, -238.0)):
            cyl(f"stay_{name}", 6.2, -163, 163, axis="x", at=(y, z), n=32, bevel=.3, segs=1, mat=self.shaft)
            for s in (-1, 1):
                x0, x1 = sorted((s * 123, s * 129))
                cyl(f"stay_{name}_collar{s}", 11.5, x0, x1, axis="x", at=(y, z), n=32, bevel=.7, segs=1, mat=c)
                x0, x1 = sorted((s * 155.6, s * 164))
                prism(f"stay_{name}_nut{s}", circle(9.0, 6, y, z, .3), x0, x1, axis="x", bevel=.6, segs=1, mat=m)
        # the flywheel's own pedestal, floor to shaft, with a window cast through it, tied back to the frame under the wheel
        wz = WHEEL_Z
        top = [(23 * math.cos(a), wz + 23 * math.sin(a)) for a in np.linspace(math.radians(-25), math.radians(205), 26)]
        ped = [(-98, FLOOR), (98, FLOOR), (98, FLOOR + 11), (36, -262), (21, wz - 12)] + top + [(-21, wz - 12), (-36, -262), (-98, FLOOR + 11)]
        window = hull((0, -222), (0, wz - 58), 15, 8, n=12)
        xp0, xp1 = WHEEL_X - 58, WHEEL_X - 35
        prism("wheel_ped", ped, xp0, xp1, holes=[list(reversed(window))], axis="x", bevel=1.8, segs=3, mat=c)
        box("wheel_ped_pad", xp0 - 16, xp1 + 16, -98, 98, FLOOR, FLOOR + 9, bevel=1.6, segs=2, mat=c2)
        cyl("wheel_ped_boss", 19, xp0 - 4, xp1 + 4, axis="x", at=(0, wz), n=48, bevel=1.1, segs=2, mat=c)
        box("wheel_ped_cap", xp0, xp1, -24, 24, wz + 18, wz + 25, bevel=.9, mat=m)
        for y in (-18, 18):
            prism(f"wheel_ped_stud{y}", circle(2.8, 6, (xp0 + xp1) / 2, y), wz + 25, wz + 27.6, axis="z", bevel=.3, segs=1, mat=m)
        for y in (-84, 84):
            for x in (xp0 - 8, xp1 + 8):
                prism(f"wheel_ped_bolt{y}{x:.0f}", circle(3.2, 6, x, y, .3), FLOOR + 9, FLOOR + 11.6, axis="z", bevel=.3, segs=1, mat=self.fast)
        cyl("wheel_tie", 6.2, xp1 - 2, -150, axis="x", at=(0, -68), n=32, bevel=.3, segs=1, mat=self.shaft)
        for x0, x1, name in ((-161, -155, "a"), (xp1 - 1, xp1 + 5, "b")):
            cyl(f"wheel_tie_collar{name}", 11.5, x0, x1, axis="x", at=(0, -68), n=32, bevel=.7, segs=1, mat=c)

    def head_parts(self):
        c, m = self.cast, self.machined
        self.head = h = empty("head")
        # the chase: an iron frame round the type. Near rail in two parts, with a channel between them for the wooden wedges
        box("chase_low", -99, 99, -38, -29, .8, 10.2, bevel=.7, mat=c, parent=h)
        box("chase_chan", -99, 99, -34.8, -29, 10, 18.2, bevel=.2, mat=c, parent=h)
        box("chase_up", -99, 99, -38, -29, 18, 22.4, bevel=.6, mat=c, parent=h)
        box("chase_far", -99, 99, 62, 71, .8, 22.4, bevel=.7, mat=c, parent=h)
        for s in (-1, 1):
            xa, xb = sorted((s * 90, s * 99))
            box(f"chase_side{s}", xa, xb, -38, 71, .8, 22.4, bevel=.7, mat=c, parent=h)
        # the wedges and furniture along the front: oak, in pairs driven against each other
        y0, y1 = -37.4, -34.6
        z0, z1 = 10.5, 17.7
        x = -99.0
        r = np.random.default_rng(4)
        i = 0
        while x < 99:
            kind = i % 3
            if kind == 0:       # a long block of furniture
                L = float(r.uniform(19, 26))
                box(f"furn{i}", x + .15, x + L - .15, y0 + float(r.uniform(0, .35)), y1, z0, z1, bevel=.28, segs=1, mat=self.wood, parent=h)
            else:               # a pair of wedges: two long triangles' worth of oak, sharing a slope
                L = float(r.uniform(23, 28))
                a, b = x + .15, x + L - .15
                t = .9
                prism(f"wedge{i}a", [(a, z0), (b, z0), (b, z0 + t + .4), (a, z1 - t)], y0 + .1, y1, axis="y", bevel=.25, segs=1, mat=self.wood, parent=h)
                prism(f"wedge{i}b", [(a, z1 - t + .35), (b, z0 + t + .75), (b, z1), (a, z1)], y0 - .35, y1, axis="y", bevel=.25, segs=1, mat=self.wood, parent=h)
            x += L
            i += 1
        # the lock-up bolts along the bottom rail
        for x_, spin in ((-74.0, .2), (-26.0, .7), (26.0, .05), (74.0, .5)):
            bolt(f"lock{x_}", x_, 5.5, -38, out=3.0, r=4.1, mat=self.fast, washer=self.fast, parent=h, spin=spin)
        # the head's casting above it: a cornice, a deck, ribs running to the guide bosses
        box("head_corn", -106, 106, -44, 84, 22, 28, bevel=1.1, segs=3, mat=c, parent=h)
        box("head_deck", -99, 99, -40, 80, 27.6, 37, bevel=1.2, segs=2, mat=c, parent=h)
        for x_ in (-66, 66):
            box(f"head_rib{x_}", x_ - 3.2, x_ + 3.2, -39, 79, 36.6, 42, bevel=1.0, mat=c, parent=h)
        box("head_rib_f", -66, 66, -39, -33, 36.6, 42, bevel=1.0, mat=c, parent=h)
        for k, x_ in enumerate((-88, -44, 44, 88)):
            prism(f"deck_bolt{k}", circle(3.6, 6, x_, -22, x_ * .03), 36.8, 39.4, axis="z", bevel=.35, segs=1, mat=self.fast, parent=h)
            prism(f"deck_bolt_f{k}", circle(3.6, 6, x_, 60, x_ * .05), 36.8, 39.4, axis="z", bevel=.35, segs=1, mat=self.fast, parent=h)
        # bosses that ride the guide posts
        for s in (-1, 1):
            x_ = s * POST_X
            cyl(f"head_boss{s}", 19, 8, 42, axis="z", at=(x_, POST_Y), n=64, bevel=1.1, segs=2, mat=c, parent=h, hole=12.2)
            a, b = sorted((s * 60, s * 106))
            box(f"head_arm{s}", a, b, 72, 88, 27.6, 40, bevel=1.0, mat=c, parent=h)
            for z_ in (14, 36):
                prism(f"gib{s}{z_}", circle(3.0, 6, x_ - s * 15, POST_Y - 12), z_ - 1.2, z_ + 1.2, axis="z", bevel=.2, segs=1, mat=m, parent=h)
        # the wrist: two cheeks on the deck with a slot for the pin
        slot = hull((ROD_Y, WRIST_H - .5), (ROD_Y, WRIST_H + SLACK + .5), 6.3, 6.3, n=12)
        cheek = [(ROD_Y - 21, 36), (ROD_Y + 21, 36), (ROD_Y + 15, WRIST_H + SLACK + 11), (ROD_Y - 15, WRIST_H + SLACK + 11)]
        for tag, a, b in (("l", -21, -14), ("r", 14, 21)):
            prism(f"cheek{tag}", cheek, a, b, holes=[list(reversed(slot))], axis="x", bevel=.7, mat=c, parent=h)
        # the back of the forme, and what locks it in the chase, seen from below: thin strips of wood at its near and far ends, and on each side a long block and two pairs of wedges running along y
        fx, fy0, fy1 = 74.0, -27.0, 60.0
        box("forme_back", -fx, fx, fy0, fy1, 3, 23.3, bevel=.1, segs=1, mat=m, parent=h)
        box("furn_n", -90, 90, -29, fy0, 3.2, 22, bevel=.2, segs=1, mat=self.wood, parent=h)
        box("furn_f", -90, 90, fy1, 62, 3.2, 22, bevel=.2, segs=1, mat=self.wood, parent=h)
        for s in (-1, 1):
            xa, xb = sorted((s * fx, s * 79))
            box(f"furn_side{s}", xa + .1, xb - .1, fy0 + .1, fy1 - .1, 3.2, 22, bevel=.25, segs=1, mat=self.wood_y, parent=h)
            ym = (fy0 + fy1) / 2
            for k, (ya, yb) in enumerate(((fy0, ym), (ym, fy1))):
                # seen from below the two wedges of a pair share a long slope
                inner, outer = s * 79.2, s * 89.8
                a1, a2 = s * 87.4, s * 81.6
                pts_a = [(inner, ya + .1), (a1, ya + .1), (a2, yb - .1), (inner, yb - .1)]
                pts_b = [(a1 + s * .5, ya + .1), (outer, ya + .1), (outer, yb - .1), (a2 + s * .5, yb - .1)]
                for tag, pts in (("a", pts_a), ("b", pts_b)):
                    prism(f"wedge_side{s}{k}{tag}", pts if s > 0 else pts[::-1], 3.2 + .3 * (tag == "b"), 22, axis="z", bevel=.28, segs=1, mat=self.wood_y, parent=h)

    def rod_parts(self):
        """The connecting rod swings forwards and back: its outline is in (y, z), its origin the wrist pin."""
        c, m = self.cast, self.machined
        self.rod = o = empty("rod")
        L = ROD_L
        out = hull((0, 0), (0, L), 11, 17, n=18, pinch=.50)
        prism("rod_web", out, -6.5, 6.5, holes=[list(reversed(circle(6.0, 24))), list(reversed(circle(9.4, 28, 0, L)))], axis="x", bevel=.8, segs=2, mat=c, parent=o)
        # raised edges, so that it reads as an I-section casting
        for s in (-1, 1):
            rail = [(s * 5.2, 11), (s * 7.4, 11), (s * 12.0, L - 13), (s * 9.6, L - 13)]
            if s < 0:
                rail = rail[::-1]
            prism(f"rod_rail{s}", rail, -8.6, 8.6, axis="x", bevel=.6, mat=c, parent=o)
        # the big end's cap, bolted on the far side of the pin, and the wrist pin
        prism("rod_cap", rect(-18, L + 9.5, 18, L + 14), -8.8, 8.8, axis="x", bevel=.5, mat=m, parent=o)
        for s in (-14, 14):
            cyl(f"rod_stud{s}", 2.9, L + 2, L + 16.5, axis="z", at=(0, s), n=6, bevel=.3, segs=1, mat=m, parent=o)
        cyl("wrist_pin", 5.9, -22.2, 22.2, axis="x", at=(0, 0), n=32, bevel=.5, mat=self.shaft, parent=o)
        cyl("oil_cup", 3.2, L + 13.5, L + 19, axis="z", at=(0, 0), n=24, bevel=.4, segs=1, mat=m, parent=o)

    def crank_parts(self):
        """The crankshaft runs along x above the head, from the gear outside the left frame through both frames. Pin at the bottom when the angle is 0."""
        c, m = self.cast, self.machined
        self.crank = o = empty("crank", (0, ROD_Y, CRANK_Z))
        R = CRANK_R
        for tag, a, b in (("l", -178, -24), ("r", 24, 170)):
            cyl(f"shaft_{tag}", 13, a, b, axis="x", at=(0, 0), n=64, bevel=.4, mat=self.shaft, parent=o)
        # two webs and the pin between them
        web = hull((0, -R), (0, 0), 16, 25, n=18, pinch=.16)
        for tag, a, b in (("l", -26, -10.5), ("r", 10.5, 26)):
            prism(f"web_{tag}", web, a, b, axis="x", bevel=1.3, segs=3, mat=c, parent=o)
            cyl(f"web_hub_{tag}", 17, a - 1.2 if a < 0 else a, b if a < 0 else b + 1.2, axis="x", at=(0, 0), n=48, bevel=.6, mat=c, parent=o)
        cyl("crank_pin", 9, -11, 11, axis="x", at=(0, -R), n=40, bevel=.3, mat=self.shaft, parent=o)
        # the gear at the end of it, a nut and a key
        holes = [list(reversed(circle(7.0, 28, GEAR_R * .56 * math.cos(a), GEAR_R * .56 * math.sin(a)))) for a in np.linspace(0, 2 * math.pi, 6, endpoint=False)]
        prism("crank_gear", gear_outline(GEAR_R, 24, 3.0), -188, -170, holes=holes, axis="x", bevel=.45, segs=2, mat=m, parent=o)
        cyl("crank_gear_hub", 15, -192, -166, axis="x", at=(0, 0), n=48, bevel=.8, mat=m, parent=o)
        prism("crank_nut", circle(8.5, 6), -196, -192, axis="x", bevel=.5, segs=1, mat=m, parent=o)

    def wheel_parts(self):
        c, m = self.cast, self.machined
        self.wheel = o = empty("wheel", (0, ROD_Y, WHEEL_Z))
        pitch = math.pi / 24
        X, R = WHEEL_X, WHEEL_R
        cyl("wheel_shaft", 12, X - 70, -124, axis="x", at=(0, 0), n=48, bevel=.4, mat=self.shaft, parent=o)
        holes = [list(reversed(circle(7.0, 28, GEAR_R * .56 * math.cos(a), GEAR_R * .56 * math.sin(a)))) for a in np.linspace(0, 2 * math.pi, 6, endpoint=False)]
        prism("wheel_gear", gear_outline(GEAR_R, 24, 3.0, phase=pitch), -188, -170, holes=holes, axis="x", bevel=.45, segs=2, mat=m, parent=o)
        cyl("wheel_gear_hub", 15, -192, -166, axis="x", at=(0, 0), n=48, bevel=.8, mat=m, parent=o)
        # rim: a heavy flat face with a web inside it. Hub. Six straight oval spokes that spread where they meet hub and rim.
        cyl("wheel_rim", R, X - 19, X + 19, axis="x", at=(0, 0), n=256, bevel=.9, segs=2, mat=c, parent=o, hole=R - 11)
        cyl("wheel_rim2", R - 10, X - 7, X + 7, axis="x", at=(0, 0), n=256, bevel=1.0, segs=2, mat=c, parent=o, hole=R - 20)
        cyl("wheel_hub", 24, X - 17, X + 17, axis="x", at=(0, 0), n=64, bevel=1.4, segs=3, mat=c, parent=o)
        cyl("wheel_hub2", 31, X - 7.5, X + 7.5, axis="x", at=(0, 0), n=64, bevel=1.2, segs=2, mat=c, parent=o)
        prism("wheel_key", rect(-3.2, 11, 3.2, 17), X - 19.5, X - 16.8, axis="x", bevel=.25, segs=1, mat=m, parent=o)
        prism("wheel_nut", circle(9.0, 6), X - 74, X - 68, axis="x", bevel=.5, segs=1, mat=m, parent=o)
        for k in range(6):
            a = 2 * math.pi * k / 6 + .2
            rings = []
            r0, r1 = 26.0, R - 17.0
            for t in np.linspace(0, 1, 16):
                r = r0 + t * (r1 - r0)
                flare = 5.0 * (math.exp(-t * 12) + math.exp(-(1 - t) * 12))
                w, d = 7.6 - 2.8 * t + flare, 4.6 - 1.4 * t + flare * .5
                ring = []
                for q in np.linspace(0, 2 * math.pi, 20, endpoint=False):
                    u, v = w * math.cos(q), d * math.sin(q)
                    ring.append((X + v, r * math.cos(a) - u * math.sin(a), r * math.sin(a) + u * math.cos(a)))
                rings.append(ring)
            loft(f"spoke{k}", rings, mat=c, parent=o)
        # the belt: wraps under the wheel, its two strands run straight up and out of frame, moving
        arc_o = [((R + 3.2) * math.cos(a), (R + 3.2) * math.sin(a)) for a in np.linspace(math.pi, 2 * math.pi, 96)]
        arc_i = [((R + .1) * math.cos(a), (R + .1) * math.sin(a)) for a in np.linspace(2 * math.pi, math.pi, 96)]
        prism("belt_arc", arc_o + arc_i, X - 12, X + 12, axis="x", bevel=.2, segs=1, mat=leather_material(), location=(0, ROD_Y, WHEEL_Z))
        cut = leather_material(cut_z=WHEEL_Z)
        self.belts = []
        for s, tag in ((-1, "up"), (1, "down")):
            a, b = sorted((ROD_Y + s * (R + .1), ROD_Y + s * (R + 3.2)))
            st = box(f"belt_{tag}", X - 12, X + 12, a, b, -5000, 7000, bevel=.15, segs=1, mat=cut)
            self.belts.append(st)
