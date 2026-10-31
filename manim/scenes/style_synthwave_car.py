"""The hero of the outrun: a mid-engined eighties wedge. The body is one lofted cage, subdivided with creased character lines,
so the neon slides along real curves: wide hips, flying buttresses round a louvred engine deck, side strakes, a tail of
horizontal slats over the lamps, quad exhausts, a wing on fins. Imported by styles_blender.synthwave()."""
import math

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

# half a cross-section, bottom centre to top centre:
#   0 centre of the floor, 1 floor edge, 2 sill, 3 widest line, 4 shoulder crease, 5 waist, 6 roof rail / top of the buttress,
#   7 inner edge of the deck or roof, 8 centre line
STATIONS = [
    (-2.06, [(0, .33), (.62, .33), (.80, .35), (.85, .56), (.83, .80), (.68, .835), (.45, .84), (.22, .845), (0, .845)]),    # floor of the tail recess
    (-2.215, [(0, .30), (.66, .30), (.85, .33), (.905, .56), (.885, .825), (.71, .862), (.47, .868), (.23, .872), (0, .872)]),
    (-2.255, [(0, .27), (.70, .27), (.90, .30), (.955, .56), (.93, .86), (.74, .895), (.49, .90), (.24, .905), (0, .905)]),   # the lip
    (-2.19, [(0, .24), (.72, .24), (.93, .27), (.985, .55), (.955, .885), (.77, .925), (.52, .93), (.26, .935), (0, .935)]),
    (-1.95, [(0, .19), (.74, .19), (.95, .23), (1.0, .545), (.97, .875), (.79, .94), (.66, .95), (.50, .925), (0, .925)]),
    (-1.72, [(0, .18), (.74, .18), (.955, .22), (1.005, .54), (.972, .865), (.80, .95), (.69, .965), (.52, .90), (0, .90)]),
    (-1.30, [(0, .17), (.74, .17), (.96, .21), (1.005, .53), (.965, .85), (.81, .96), (.70, .985), (.52, .905), (0, .905)]),
    (-0.85, [(0, .17), (.74, .17), (.94, .21), (.985, .50), (.925, .80), (.77, .905), (.665, 1.03), (.46, .925), (0, .92)]),
    (-0.55, [(0, .17), (.74, .17), (.92, .21), (.955, .49), (.89, .775), (.745, .875), (.625, 1.085), (.40, 1.02), (0, 1.015)]),
    (-0.30, [(0, .17), (.74, .17), (.905, .21), (.935, .48), (.875, .75), (.735, .85), (.59, 1.115), (.33, 1.13), (0, 1.135)]),
    (0.10, [(0, .17), (.74, .17), (.90, .21), (.925, .46), (.87, .71), (.76, .79), (.565, 1.125), (.30, 1.145), (0, 1.15)]),
    (0.50, [(0, .17), (.74, .17), (.90, .21), (.925, .45), (.875, .68), (.77, .745), (.585, 1.04), (.31, 1.07), (0, 1.075)]),
    (0.95, [(0, .17), (.72, .17), (.90, .21), (.93, .44), (.885, .645), (.75, .70), (.665, .75), (.34, .785), (0, .79)]),
    (1.35, [(0, .17), (.70, .17), (.885, .21), (.935, .41), (.895, .615), (.72, .655), (.52, .67), (.26, .68), (0, .685)]),
    (1.85, [(0, .18), (.64, .18), (.83, .22), (.885, .38), (.84, .54), (.64, .58), (.44, .59), (.22, .60), (0, .605)]),
    (2.20, [(0, .22), (.52, .22), (.69, .25), (.74, .34), (.69, .43), (.52, .46), (.36, .47), (.18, .475), (0, .48)]),
    (2.28, [(0, .27), (.38, .27), (.52, .29), (.56, .34), (.52, .39), (.40, .41), (.28, .415), (.14, .42), (0, .42)]),
]
PAINT, GLASS, BLACK = 0, 1, 2
REAR_WHEEL = (-1.32, 0.335, 0.385, 0.36)                     # y, tyre radius, arch radius, tyre width
FRONT_WHEEL = (1.33, 0.315, 0.355, 0.27)


def link(o):
    bpy.context.collection.objects.link(o)
    return o


def hexc(h):
    h = h.lstrip("#")
    return tuple(((int(h[i:i + 2], 16) / 255 + 0.055) / 1.055) ** 2.4 for i in (0, 2, 4))


def material(name, color, metallic=0.0, rough=0.5, **extra):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Roughness"].default_value = rough
    for k, v in extra.items():
        p.inputs[k].default_value = v
    return m


def emission(name, color, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1)
    e.inputs["Strength"].default_value = strength
    nt.links.new(e.outputs["Emission"], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    return m


def loft(name, rings, mats, parent, band=None, level=0, long_crease=None, ring_crease=None, cap=True, closed=True, sharp=None):
    """A skin through rings of points. band(s, j) picks a material for each quad; long_crease is {ring index: weight} for
    the edges running from ring to ring, ring_crease {ring number: weight} for the edges round a ring."""
    n, m = len(rings), len(rings[0])
    verts = [tuple(float(c) for c in p) for ring in rings for p in ring]
    faces, kinds = [], []
    for s in range(n - 1):
        for j in range(m if closed else m - 1):
            faces.append((s * m + j, s * m + (j + 1) % m, (s + 1) * m + (j + 1) % m, (s + 1) * m + j))
            kinds.append(band(s, j) if band else 0)
    if cap and closed:
        faces += [tuple(range(m)), tuple((n - 1) * m + j for j in range(m))]
        kinds += [band(0, 0) if band else 0, band(n - 2, 0) if band else 0]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    for mat in mats:
        me.materials.append(mat)
    me.polygons.foreach_set("material_index", kinds)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    if long_crease or ring_crease:
        attr = me.attributes.new("crease_edge", "FLOAT", "EDGE")
        for e in me.edges:
            a, b = sorted(e.vertices)
            sa, ja = divmod(a, m)
            sb, jb = divmod(b, m)
            w = 0.0
            if ja == jb and long_crease:
                w = long_crease.get(ja, 0.0)
            elif sa == sb and ring_crease:
                w = ring_crease.get(sa, 0.0)
            attr.data[e.index].value = w
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()
    o = link(bpy.data.objects.new(name, me))
    o.parent = parent
    if level:
        sub = o.modifiers.new("sub", "SUBSURF")
        sub.levels = sub.render_levels = level
        sub.use_creases = True
        bake(o)
    if sharp:
        o.data.set_sharp_from_angle(angle=math.radians(sharp))
    return o


def bake(o):
    """Freeze the modifier stack so it is not evaluated again on every frame."""
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
    o.modifiers.clear()
    o.data = me
    return o


def circle(r, y=None, x=None, cy=0.0, cz=0.0, cx=0.0, n=64):
    """A ring about the y axis (give y) or the x axis (give x)."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    if y is not None:
        return [(cx + r * math.cos(u), y, cz + r * math.sin(u)) for u in a]
    return [(x, cy + r * math.cos(u), cz + r * math.sin(u)) for u in a]


def body(mats):
    K = len(STATIONS[0][1])
    rings = [[(a, y, b) for a, b in half + [(-c, d) for c, d in reversed(half[1:-1])]] for y, half in STATIONS]
    index = {round(y, 3): s for s, (y, _) in enumerate(STATIONS)}

    def band(s, j):
        b = min(j, 2 * K - 3 - j)
        y0, y1 = STATIONS[s][0], STATIONS[s + 1][0]
        if b <= 1 or y0 < -2.2 or y1 <= -2.215:
            return BLACK
        if b == 5 and -0.30 <= y0 and y1 <= 0.95:             # side windows
            return GLASS
        if b >= 6 and 0.50 <= y0 and y1 <= 0.95:              # windscreen
            return GLASS
        if b == 7 and -0.85 <= y0 and y1 <= -0.30:            # rear window, sunk between the buttresses
            return GLASS
        if b == 7 and -1.95 <= y0 and y1 <= -0.85:            # the engine deck under its louvres
            return BLACK
        return PAINT

    crease = {4: 0.85, 5: 0.55, 6: 0.75, 2: 0.45}
    long_crease = {}
    for h, w in crease.items():
        long_crease[h] = w
        long_crease[2 * K - 2 - h] = w
    o = loft("car body", rings, mats, None, band, 3, long_crease, {index[-2.255]: 1.0, index[-2.215]: 0.9, index[-2.06]: 1.0})
    cutters = []
    for dy, _, arch, _ in (REAR_WHEEL, FRONT_WHEEL):
        z = (REAR_WHEEL[1] if dy < 0 else FRONT_WHEEL[1])
        for x0, x1 in ((0.58, 1.3), (-1.3, -0.58)):
            cutters.append(loft("arch", [circle(arch, x=x0, cy=dy, cz=z, n=96), circle(arch, x=x1, cy=dy, cz=z, n=96)], [mats[BLACK]], None))
    for c in cutters:
        mod = o.modifiers.new("arch", "BOOLEAN")
        mod.operation, mod.object, mod.solver = "DIFFERENCE", c, "EXACT"
        mod.material_mode = "TRANSFER"
    bake(o)
    for c in cutters:
        bpy.data.objects.remove(c, do_unlink=True)
    o.data.set_sharp_from_angle(angle=math.radians(42))
    return o


def toucher(o):
    """Rays onto the finished skin: skin(y, z) is the side at +x, deck(x, y) the top."""
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bvh = BVHTree.FromBMesh(bm)

    def skin(y, z):
        hit = bvh.ray_cast((2.0, y, z), (-1.0, 0.0, 0.0))[0]
        return hit.x if hit else 0.9

    def deck(x, y):
        hit = bvh.ray_cast((x, y, 2.0), (0.0, 0.0, -1.0))[0]
        return hit.z if hit else 0.9

    return skin, deck


def strakes(skin, paint, black, parent):
    """The side intake ahead of each rear wheel: five long fins over a dark recess."""
    ys = np.linspace(-0.93, 0.30, 26)
    for side in (1, -1):
        zs = np.linspace(0.34, 0.655, 8)
        sheet = [[(side * (skin(y, z) + 0.004), y, z) for z in zs] for y in ys]
        loft("strake recess", sheet, [black], parent, closed=False)
        for k, z in enumerate((0.385, 0.442, 0.499, 0.556, 0.613)):
            rings = []
            for y in ys:
                u = (y - ys[0]) / (ys[-1] - ys[0])
                stand = 0.02 * min(1.0, u * 9, (1 - u) * 5 + 0.25)    # sinks back into the door at its front
                x = skin(y, z)
                rings.append([(side * (x - 0.02), y, z - 0.012), (side * (x + stand), y, z - 0.008),
                              (side * (x + stand), y, z + 0.008), (side * (x - 0.02), y, z + 0.012)])
            loft("strake", rings, [paint], parent, sharp=30)


def louvres(deck, paint, parent):
    """Black gloss slats across the engine deck, which catch the sun as a ladder of lines, their ends hidden under two rails."""
    slat = material("louvre", hexc("#0B0B10"), rough=0.12, **{"Coat Weight": 1.0, "Coat Roughness": 0.02})

    def edge(y):
        return 0.49 - 0.06 * (y + 1.93)

    for y in np.linspace(-1.86, -0.98, 9):
        w = edge(y) + 0.01
        rings = []
        for x in np.linspace(-w, w, 15):
            z = deck(x, y) + 0.012
            rings.append([(x, y - 0.042, z - 0.004), (x, y - 0.042, z + 0.004), (x, y + 0.042, z + 0.024), (x, y + 0.042, z + 0.016)])
        loft("louvre", rings, [slat], parent, sharp=30)
    for side in (1, -1):
        rings = []
        for y in np.linspace(-1.93, -0.93, 12):
            x = side * edge(y)
            z = deck(x, y)
            rings.append([(x - 0.022, y, z - 0.01), (x - 0.016, y, z + 0.036), (x + 0.016, y, z + 0.036), (x + 0.022, y, z - 0.01)])
        loft("deck rail", rings, [paint], parent, sharp=35)


def lamp_material():
    """Red all the way across, burning hottest round the two lamps at the outer ends. Returns the strength input, for the launch."""
    m = bpy.data.materials.new("tail lamps")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ab = nt.nodes.new("ShaderNodeMath")
    ab.operation = "ABSOLUTE"
    scale = nt.nodes.new("ShaderNodeMath")
    scale.operation = "DIVIDE"
    scale.inputs[1].default_value = 0.85
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (0.32, 0.0, 0.012, 1)
    els[1].position, els[1].color = 1.0, (0.45, 0.0, 0.015, 1)
    for pos, c in ((0.34, (0.42, 0.0, 0.015)), (0.44, (1.0, 0.012, 0.035)), (0.62, (1.0, 0.09, 0.09)), (0.82, (1.0, 0.012, 0.035))):
        els.new(pos).color = (*c, 1)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 3.0
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    nt.links.new(sep.outputs["X"], ab.inputs[0])
    nt.links.new(ab.outputs[0], scale.inputs[0])
    nt.links.new(scale.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    return m, em.inputs["Strength"]


def flame_material():
    """Pale blue at the pipe, blue, then magenta, nothing beyond the tip. Object y runs from 0 at the pipe to -1 at the tip. From behind the
    flame is seen nearly end-on, so its sides are almost edge-on to the camera: they stay bright, and only the outline itself fades."""
    m = bpy.data.materials.new("flame")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    neg = nt.nodes.new("ShaderNodeMath")
    neg.operation = "MULTIPLY"
    neg.inputs[1].default_value = -1.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (0.3, 0.5, 1.0, 1)
    els[1].position, els[1].color = 1.0, (0.0, 0.0, 0.0, 1)
    els.new(0.16).color = (0.05, 0.16, 1.0, 1)
    els.new(0.42).color = (0.28, 0.06, 1.0, 1)
    els.new(0.7).color = (0.72, 0.01, 0.58, 1)
    em = nt.nodes.new("ShaderNodeEmission")
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    soft = nt.nodes.new("ShaderNodeMapRange")
    soft.inputs["From Min"].default_value, soft.inputs["From Max"].default_value = 0.62, 1.0
    soft.inputs["To Min"].default_value, soft.inputs["To Max"].default_value = 0.8, 0.0
    nt.links.new(lw.outputs["Facing"], soft.inputs["Value"])
    # licks running the length of the flame, so it is not one solid fill
    stretch = nt.nodes.new("ShaderNodeMapping")
    stretch.inputs["Scale"].default_value = (22.0, 2.2, 22.0)
    nt.links.new(tc.outputs["Object"], stretch.inputs["Vector"])
    licks = nt.nodes.new("ShaderNodeTexNoise")
    licks.inputs["Detail"].default_value = 2.0
    nt.links.new(stretch.outputs[0], licks.inputs["Vector"])
    gain = nt.nodes.new("ShaderNodeMapRange")
    gain.inputs["From Min"].default_value, gain.inputs["From Max"].default_value = 0.34, 0.66
    gain.inputs["To Min"].default_value, gain.inputs["To Max"].default_value = 0.3, 1.35
    nt.links.new(licks.outputs["Fac"], gain.inputs["Value"])
    both = nt.nodes.new("ShaderNodeMath")
    both.operation = "MULTIPLY"
    nt.links.new(soft.outputs[0], both.inputs[0])
    nt.links.new(gain.outputs[0], both.inputs[1])
    nt.links.new(both.outputs[0], em.inputs["Strength"])
    add = nt.nodes.new("ShaderNodeAddShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    nt.links.new(sep.outputs["Y"], neg.inputs[0])
    nt.links.new(neg.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
    nt.links.new(tr.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    return m


def flame_skin(seed):
    """A tongue of fire from the pipe (y 0) to the tip (y -1): fat just past the pipe, lumpier and more crooked the further it gets."""
    rnd = np.random.default_rng(seed)
    p1, p2, p3 = rnd.uniform(0, 2 * math.pi, 3)
    rings = []
    for u in np.r_[0, 0.025, np.linspace(0.05, 1, 13)]:
        r = np.interp(u, (0, 0.05, 0.3, 0.55, 0.8, 1.0), (0.002, 0.042, 0.072, 0.06, 0.03, 0.004))
        bx, bz = 0.03 * u * math.cos(p3 + 2 * u), 0.03 * u * math.sin(p3 + 2 * u)
        ring = []
        for a in np.linspace(0, 2 * math.pi, 32, endpoint=False):
            lump = 1 + (0.05 + 0.2 * u) * (0.6 * math.sin(3 * a + p1 + 4 * u) + 0.4 * math.sin(5 * a + p2 - 6 * u))
            ring.append((bx + r * lump * math.cos(a), -u, bz + r * lump * math.sin(a)))
        rings.append(ring)
    return rings


def tail(paint, black, chrome, parent):
    """Everything inside the recess, top to bottom: lamps behind slats, the bumper, the diffuser and its four pipes."""
    def plan(x):                                              # the tail curves away towards its corners
        return 0.05 * (x / 0.85) ** 2

    lamps, strength = lamp_material()
    xs = np.linspace(-0.86, 0.86, 25)
    loft("tail lamps", [[(x, -2.135 + plan(x), z) for z in np.linspace(0.585, 0.85, 3)] for x in xs], [lamps], parent, closed=False)
    for z in np.linspace(0.612, 0.822, 6):
        rings = [[(x, -2.205 + plan(x), z - 0.008), (x, -2.205 + plan(x), z + 0.008), (x, -2.145 + plan(x), z + 0.008), (x, -2.145 + plan(x), z - 0.008)] for x in xs]
        loft("slat", rings, [black], parent, sharp=30)
    # the bumper, wrapping forward at its ends
    profile = [(-2.10, 0.60), (-2.245, 0.60), (-2.30, 0.565), (-2.318, 0.48), (-2.30, 0.40), (-2.245, 0.365), (-2.10, 0.365)]
    rings = []
    for x in np.linspace(-0.95, 0.95, 13):
        wrap = 0.14 * (abs(x) / 0.95) ** 4
        rings.append([(x, y + wrap, z) for y, z in profile])
    bump = loft("bumper", rings, [paint], parent, level=2, long_crease={0: 0.8, 6: 0.8})
    plate(bump, black, chrome, parent)
    # the diffuser
    rings = [[(x, -2.10, 0.375), (x, -2.27, 0.375), (x, -2.255, 0.21), (x, -2.10, 0.19)] for x in np.linspace(-0.9, 0.9, 7)]
    loft("valance", rings, [black], parent, sharp=30)
    for x in np.linspace(-0.33, 0.33, 7):
        loft("fin", [[(x - 0.006, -2.30, 0.21), (x - 0.006, -2.30, 0.37), (x - 0.006, -2.2, 0.37), (x - 0.006, -2.2, 0.2)],
                     [(x + 0.006, -2.30, 0.21), (x + 0.006, -2.30, 0.37), (x + 0.006, -2.2, 0.37), (x + 0.006, -2.2, 0.2)]], [black], parent, sharp=30)
    hot = emission("pipe glow", hexc("#FF5A1E"), 0.5)
    fm = flame_material()
    flames = []
    for x in (-0.68, -0.5, 0.5, 0.68):
        z = 0.292
        loft("pipe", [circle(0.056, y=-2.12, cx=x, cz=z), circle(0.056, y=-2.318, cx=x, cz=z), circle(0.05, y=-2.332, cx=x, cz=z),
                      circle(0.046, y=-2.32, cx=x, cz=z), circle(0.046, y=-2.23, cx=x, cz=z)], [chrome], parent, sharp=55)
        loft("pipe glow", [circle(0.001, y=-2.232, cx=x, cz=z, n=24), circle(0.045, y=-2.232, cx=x, cz=z, n=24)], [hot], parent, closed=True, cap=False)
        fl = loft("flame", flame_skin(int(x * 100) + 200), [fm], parent, cap=False)
        fl.location = (x, -2.3, z)
        fl.rotation_euler = (0.15, 0, math.copysign(0.09, x))  # fanning out and down a touch from the pipes: seen from behind, it shows its length
        fl.scale = (1, 0.01, 1)
        fl.visible_shadow = False
        flames.append(fl)
    return strength, flames


def rounded(w, h, r, cz=0.0, n=5):
    """A rounded rectangle in (x, z), anticlockwise from the right edge."""
    pts = []
    for cx0, cz0, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for k in range(n + 1):
            u = math.radians(a0 + 90 * k / n)
            pts.append((cx0 + r * math.cos(u), cz + cz0 + r * math.sin(u)))
    return pts


def stamped(body, font, size, depth, bevel, mat, parent, x, y, z, spacing=1.0):
    """Letters pressed up out of the plate."""
    cu = bpy.data.curves.new("t", "FONT")
    cu.body, cu.font, cu.size = body, font, size
    cu.extrude, cu.bevel_depth, cu.bevel_resolution = depth / 2, bevel, 2 if bevel else 0
    cu.align_x, cu.align_y = "CENTER", "CENTER"
    cu.space_character = spacing
    t = link(bpy.data.objects.new("t", cu))
    me = bpy.data.meshes.new_from_object(t.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(t, do_unlink=True)
    bpy.data.curves.remove(cu)
    me.materials.append(mat)
    o = link(bpy.data.objects.new(body, me))
    o.rotation_euler = (math.pi / 2, 0, 0)
    o.location = (x, y, z)
    o.parent = parent
    return o


def plate(bump, black, chrome, parent):
    """A real plate, not a caption: pressed steel at the true 2:1, deep blue enamel with a seven-character registration painted
    flat gold on figures standing proud, a raised rim, bolts and registration stickers, set in a pocket in the bumper and lit only by
    the small lamp in the pocket's top edge. Everything it adds to the car is tagged, so the car's own lights can leave it out."""
    made = set(parent.children)
    w, h, cz = 0.31, 0.155, 0.4825
    front = -2.312
    pocket = rounded(w + 0.026, h + 0.026, 0.016, cz)
    cutter = loft("pocket", [[(x, -2.45, z) for x, z in pocket], [(x, -2.272, z) for x, z in pocket]], [black], None)
    mod = bump.modifiers.new("pocket", "BOOLEAN")
    mod.operation, mod.object, mod.solver = "DIFFERENCE", cutter, "EXACT"
    mod.material_mode = "TRANSFER"
    bake(bump)
    bpy.data.objects.remove(cutter, do_unlink=True)
    bump.data.set_sharp_from_angle(angle=math.radians(42))

    enamel = material("plate enamel", hexc("#101D4A"), metallic=0.25, rough=0.38, **{"Coat Weight": 0.6, "Coat Roughness": 0.12})
    gold = material("plate gold", hexc("#D2A636"), rough=0.68, **{"Specular IOR Level": 0.25})
    rings = []
    for inset, y in ((0.0, front + 0.02), (0.0, front + 0.002), (0.0035, front - 0.0025), (0.0085, front - 0.0025), (0.012, front + 0.0015), (0.018, front + 0.002)):
        rings.append([(x, y, z) for x, z in rounded(w - 2 * inset, h - 2 * inset, 0.012 - min(inset, 0.009), cz)])
    loft("plate", rings, [enamel], parent, sharp=28)

    din = bpy.data.fonts.load("/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf", check_existing=True)
    stamped("2BNR984", din, 0.088, 0.004, 0.0012, gold, parent, 0, front - 0.001, cz - 0.004, 1.02)
    stamped("SUNSET", din, 0.0165, 0.0012, 0.0, gold, parent, 0, front - 0.0002, cz - 0.0585, 1.3)
    for x, colour in ((-0.123, "#D8362B"), (0.123, "#E8C21F")):
        sticker = material("sticker", hexc(colour), rough=0.6, **{"Specular IOR Level": 0.3})
        loft("sticker", [[(x + d, front - 0.0012, z) for z in (cz + 0.041, cz + 0.06)] for d in (-0.018, 0.018)], [sticker], parent, closed=False)
    for x in (-0.09, 0.09):
        for z in (cz + 0.052, cz - 0.052):
            loft("bolt", [circle(0.0062, y=front + 0.001, cx=x, cz=z, n=16), circle(0.0058, y=front - 0.0032, cx=x, cz=z, n=16),
                          circle(0.0034, y=front - 0.0044, cx=x, cz=z, n=16)], [chrome], parent, sharp=50)
    lamp = bpy.data.lights.new("plate lamp", "AREA")
    lamp.shape, lamp.size, lamp.size_y = "RECTANGLE", 0.26, 0.012
    lamp.energy, lamp.color = 0.15, (1.0, 0.9, 0.74)
    lo = link(bpy.data.objects.new("plate lamp", lamp))
    lo.location = (0, -2.385, cz + h / 2 + 0.012)
    lo.rotation_euler = (math.radians(28), 0, 0)
    lo.parent = parent
    lo.visible_camera = lo.visible_glossy = False
    for o in set(parent.children) - made:
        o["plate"] = True


def underside(parent):
    """Far off and from low down, the sightline runs right under the floor to the mirror road ahead of the car, and the sun shows
    between the rear wheels. A dead-black block between the wheels closes it; it casts nothing, so the glow beneath the car still gets out."""
    me = bpy.data.meshes.new("underside")
    x, y0, y1, z0, z1 = 0.6, -2.12, 1.9, 0.004, 0.3
    me.from_pydata([(a, b, c) for a in (-x, x) for b in (y0, y1) for c in (z0, z1)], [],
                   [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    me.materials.append(material("underside", hexc("#020203"), rough=1.0, **{"Specular IOR Level": 0.0}))
    me.update()
    o = link(bpy.data.objects.new("underside", me))
    o.parent = parent
    o.visible_shadow = False


def wing(paint, black, parent):
    """A blade across the tail between two fins that grow out of the rear shoulders."""
    rings = []
    for x in np.linspace(-0.93, 0.93, 13):
        lift = 0.03 * (x / 0.93) ** 2
        y0, z0 = -1.93 - 0.03 * (x / 0.93) ** 2, 1.165 + lift
        rings.append([(x, y0, z0), (x, y0 - 0.10, z0 + 0.03), (x, y0 - 0.24, z0 + 0.034), (x, y0 - 0.37, z0 + 0.04), (x, y0 - 0.37, z0 + 0.018),
                      (x, y0 - 0.24, z0 - 0.012), (x, y0 - 0.10, z0 - 0.018)])
    loft("wing", rings, [paint], parent, level=2, long_crease={3: 1.0, 4: 1.0})
    third = emission("high lamp", hexc("#FF1436"), 2.6)
    loft("high lamp", [[(x, -2.314, z) for z in (1.176, 1.19)] for x in (-0.34, 0.34)], [third], parent, closed=False)
    outline = [(-1.50, 0.84), (-1.72, 0.97), (-1.88, 1.10), (-1.93, 1.21), (-2.32, 1.235), (-2.345, 1.12), (-2.20, 0.97), (-2.14, 0.84)]
    for x in (-0.915, 0.915):
        rings = [[(x + d, y, z) for y, z in outline] for d in (-0.012, 0.012)]
        loft("fin", rings, [paint], parent, sharp=35)


def wheels(chrome, parent):
    spokes = []
    tyre = material("tyre", hexc("#2A2A30"), rough=0.5, **{"Specular IOR Level": 0.6})
    rim = material("rim", (0.62, 0.62, 0.66), metallic=1.0, rough=0.32)
    dark = material("rim dark", (0.03, 0.03, 0.04), metallic=0.8, rough=0.4)
    for dy, r, _, w in (REAR_WHEEL, FRONT_WHEEL):
        for side in (1, -1):
            x1 = side * (0.965 if dy < 0 else 0.905)          # the outer face, tucked just under the arch
            def at(u):
                return x1 - side * u
            profile = [(w, r * 0.62), (w, r * 0.88), (w - 0.02, r * 0.97)]
            for k in (0.8, 0.62, 0.38, 0.2):
                u = w * k
                profile += [(u + 0.012, r), (u + 0.006, r - 0.012), (u - 0.006, r - 0.012), (u - 0.012, r)]
            profile += [(0.02, r * 0.97), (0.0, r * 0.88), (0.0, r * 0.66), (0.008, r * 0.63)]
            rings = [circle(a, x=at(u), cy=dy, cz=r, n=96) for u, a in profile]
            loft("tyre", rings, [tyre], parent, cap=False, sharp=38)
            lip = [(0.008, r * 0.64), (-0.004, r * 0.625), (0.0, r * 0.59), (0.05, r * 0.55), (0.1, r * 0.5)]
            loft("rim lip", [circle(a, x=at(u), cy=dy, cz=r, n=96) for u, a in lip], [chrome], parent, cap=False, sharp=50)
            disc = [(0.1, r * 0.5), (0.095, r * 0.36), (0.07, r * 0.2), (0.05, r * 0.16), (0.045, r * 0.1), (0.05, 0.002)]
            loft("rim", [circle(a, x=at(u), cy=dy, cz=r, n=96) for u, a in disc], [rim], parent, cap=False, sharp=50)
            inner = [(w, r * 0.62), (w, 0.002)]
            loft("wheel back", [circle(a, x=at(u), cy=dy, cz=r, n=48) for u, a in inner], [dark], parent, cap=False)
            hub = bpy.data.objects.new("spokes", None)
            link(hub)
            hub.parent = parent
            hub.location = (x1, dy, r)
            for k in range(5):                                # a five-armed star, tapering to the lip
                a = 2 * math.pi * k / 5
                c, s = math.cos(a), math.sin(a)
                def arm(rad, half, u):
                    py, pz = rad * c, rad * s
                    qy, qz = -s * half, c * half
                    return [(-side * u, py - qy, pz - qz), (-side * u, py + qy, pz + qz), (-side * (u + 0.03), py + qy, pz + qz), (-side * (u + 0.03), py - qy, pz - qz)]
                loft("spoke", [arm(r * 0.06, r * 0.13, 0.03), arm(r * 0.3, r * 0.075, 0.02), arm(r * 0.56, r * 0.06, 0.012)], [rim], hub, sharp=30)
            spokes.append(hub)
    return spokes


def mirrors(paint, glass, parent):
    for side in (1, -1):
        rings = [[(side * x, y0, z0), (side * x, y1, z0), (side * x, y1, z1), (side * x, y0, z1)] for x, y0, y1, z0, z1 in
                 ((0.86, 0.56, 0.60, 0.80, 0.82), (0.93, 0.50, 0.62, 0.785, 0.85), (1.04, 0.50, 0.60, 0.785, 0.855), (1.07, 0.52, 0.58, 0.80, 0.84))]
        loft("mirror", rings, [paint], parent, level=2)
        loft("mirror glass", [[(side * x, 0.497, z) for z in (0.793, 0.848)] for x in (0.945, 1.035)], [glass], parent, closed=False)


def seams(skin, black, parent):
    """The shut lines a real door has."""
    for side in (1, -1):
        for y in (0.42, -0.90):
            zs = np.linspace(0.24, 0.70, 16)
            loft("seam", [[(side * (skin(y + d, z) + 0.002), y + d, z) for z in zs] for d in (-0.004, 0.004)], [black], parent, closed=False)


def outrun_car(chrome, paint_hex="#C8102E", metallic=0.45, rough=0.34):
    car = bpy.data.objects.new("car", None)
    link(car)
    paint = material("paint", hexc(paint_hex), metallic=metallic, rough=rough, **{"Coat Weight": 1.0, "Coat Roughness": 0.015})
    glass = material("car glass", hexc("#04050A"), metallic=0.5, rough=0.03, **{"Coat Weight": 1.0, "Coat Roughness": 0.0})
    black = material("car black", hexc("#08080C"), rough=0.3)
    shell = body([paint, glass, black])
    shell.parent = car
    skin, deck = toucher(shell)
    strakes(skin, paint, black, car)
    louvres(deck, paint, car)
    seams(skin, black, car)
    strength, flames = tail(paint, black, chrome, car)
    underside(car)
    wing(paint, black, car)
    spokes = wheels(chrome, car)
    mirrors(paint, glass, car)
    return car, strength, flames, spokes
