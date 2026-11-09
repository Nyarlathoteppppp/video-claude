"""The two screens among the imprints, as things in the world. Inside Blender, from manim/scenes:

    Blender -b --factory-startup -P film_screens3d.py -- --shot=s4_screen --passes=room,panel --page=PATH [--samples=256]
    Blender -b --factory-startup -P film_screens3d.py -- --shot=s4_key --t=72.18 --name=s4_key_down [--samples=256]
    Blender -b --factory-startup -P film_screens3d.py -- --shot=s5_post --page=PATH --gain=4.1 [--samples=256] [--sent=1]
    Blender -b --factory-startup -P film_screens3d.py -- --shot=two_term --passes=room,panel --page=PATH --at=U,V [--hold=X,Y]     # 二's terminal: U,V on the screen held at X,Y of the frame

With --list=PATH, one session renders many pictures, each line of PATH what it shows (a page, or for s4_key a film time) and its plate, tab between.
Each pass goes to ink/film_out/film/screens/plates/NAME.npy in linear light, and film_screens.py finishes it.

s4_screen is the room of film_room3d.py at night, with one lamp and the Terminal on the Studio Display: the glass is what makes it a screen, so it gives back the lamp and the room, out of focus because they are far behind it, with its dust in focus. s4_key is the keyboard's right end, built for a macro lens: each cap a real shape in a real hole in the aluminium. s5_post is only a panel in the dark.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from mathutils import Matrix, Vector

import film_room3d as rm
import film_screens_tex as kt
from film_room3d import aluminium, bevelled, link, mesh_object, noisy, opt, printed, slab, surface, table_material, tex, tube
from ink3d import reset

OUT = kt.OUT
PLATES = os.path.join(OUT, "plates")
SHOT = opt("shot", "s4_key")
PASSES = opt("passes", "beauty").split(",")
PAGE = opt("page", "")
SAMPLES, PCT = opt("samples", 256), opt("pct", 100)
T = opt("t", 72.10)
GAIN = opt("gain", 1.0)
NAME = opt("name", "")
SENT = opt("sent", 0)
LIST = opt("list", "")
JOBS = [x.rstrip("\n").split("\t") for x in open(LIST)] if LIST else []
PAGE = PAGE or (JOBS[0][0] if JOBS and SHOT != "s4_key" else "")


def setup():
    sc = reset(SAMPLES, PCT)
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.exr_codec = "ZIP"
    sc.view_settings.view_transform = "Raw"
    sc.cycles.max_bounces = 8
    sc.cycles.diffuse_bounces = 4
    sc.cycles.glossy_bounces = 4
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.blur_glossy = .5
    sc.cycles.adaptive_threshold = .008
    return sc


def world(strength, colour=(1, 1, 1)):
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (*colour, 1)
    bg.inputs[1].default_value = strength


def camera(loc, aim, lens, fstop, focus_on):
    sc = bpy.context.scene
    d = bpy.data.cameras.new("cam")
    d.lens, d.clip_start, d.clip_end, d.sensor_width = lens, .005, 40, 36.0
    cam = link(bpy.data.objects.new("cam", d))
    cam.location = loc
    forward = (Vector(aim) - Vector(loc)).normalized()
    cam.rotation_euler = forward.to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    d.dof.use_dof = True
    d.dof.focus_distance = (Vector(focus_on) - Vector(loc)).dot(forward)
    d.dof.aperture_fstop = fstop
    d.dof.aperture_blades = 9
    return cam


def render(name, denoise=True):
    sc = bpy.context.scene
    os.makedirs(PLATES, exist_ok=True)
    t0 = time.time()
    sc.cycles.use_denoising = denoise
    path = os.path.join(PLATES, f"{name}.exr")
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(path)
    w, h = im.size
    a = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(a)
    np.save(os.path.join(PLATES, f"{name}.npy"), a.reshape(h, w, 4)[::-1][..., :3].astype(np.float16))
    bpy.data.images.remove(im)
    os.remove(path)
    print(f"{name}: {time.time() - t0:.0f} s", flush=True)


def shows(material, picture):
    """Put another page on the panel, and let go of the last: they are big."""
    t = material.node_tree.nodes["page"]
    old = t.image
    t.image = bpy.data.images.load(picture, check_existing=True)
    if old is not t.image:
        bpy.data.images.remove(old)


def emitting(name, picture, strength):
    """Only light, from a picture."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    t, e, o = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    t.name = "page"
    t.image = bpy.data.images.load(picture, check_existing=True)
    t.interpolation = "Linear"
    t.extension = "EXTEND"
    nt.links.new(nt.nodes.new("ShaderNodeUVMap").outputs[0], t.inputs[0])
    nt.links.new(t.outputs[0], e.inputs[0])
    e.inputs[1].default_value = strength
    nt.links.new(e.outputs[0], o.inputs[0])
    return m


# ---------------------------------------------------------------- s4_screen

def glass_material(picture):
    """The panel behind its glass. What it shows, in the panel pass; the rest is the surface, in every pass."""
    m = surface("glass_panel", (.004, .004, .004), .05, 0, .5)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.name = "page"
    t.image = bpy.data.images.load(picture, check_existing=True)
    uv = nt.nodes.new("ShaderNodeUVMap").outputs[0]
    nt.links.new(uv, t.inputs[0])
    nt.links.new(t.outputs[0], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = 0.0
    d = nt.nodes.new("ShaderNodeTexImage")
    d.image = tex(os.path.join(kt.TEX, "glass_dust.png"), "Non-Color")
    nt.links.new(uv, d.inputs[0])
    split = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(d.outputs[0], split.inputs[0])
    nt.links.new(rm.calc(nt, "ADD", .05, rm.calc(nt, "MULTIPLY", split.outputs[1], .05)), b.inputs["Roughness"])
    nt.links.new(rm.blended(nt, split.outputs[0], (.004, .004, .004), (.55, .53, .5)), b.inputs["Base Color"])
    return m


def floor_lamp(x, y):
    """The one lamp, behind whoever would sit here: a stem and a cloth shade with a bulb in it. Only the glass ever sees it."""
    brass = surface("lamp_brass", (.45, .32, .15), .3, 1.0)
    rm.lathe("lamp_foot", [(.13, .0), (.13, .015), (.02, .03), (.0, .03)], x, y, 0, brass)
    tube("lamp_stem", [(x, y, .02), (x, y, .5), (x, y, .98)], .009, brass, 2)
    shade = bpy.data.materials.new("lamp_shade")
    shade.use_nodes = True
    nt = shade.node_tree
    nt.nodes.clear()
    e, o = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    e.inputs[0].default_value = (1.0, .60, .30, 1)
    # the bulb shows through the cloth: brightest where you look straight at it, dull at the edges
    facing = nt.nodes.new("ShaderNodeLayerWeight")
    facing.inputs[0].default_value = .5
    nt.links.new(rm.calc(nt, "MULTIPLY", rm.calc(nt, "SUBTRACT", 1.0, rm.calc(nt, "MULTIPLY", facing.outputs["Facing"], .95)), 4.6), e.inputs[1])
    nt.links.new(e.outputs[0], o.inputs[0])
    rm.lathe("lamp_shade", [(.17, .95), (.11, 1.16)], x, y, 0, shade)
    bulb = bpy.data.lights.new("lamp", "POINT")
    bulb.energy, bulb.shadow_soft_size, bulb.color = 22.0, .05, (1.0, .66, .36)
    b = link(bpy.data.objects.new("lamp", bulb))
    b.location = (x, y, 1.05)


def s4_screen():
    sc = setup()
    rm.room()
    rm.table()
    screen, corners, L = rm.display()
    rm.things()
    bpy.data.objects["outside"].hide_render = True
    world(.006, (.6, .7, 1.0))
    floor_lamp(-.37, -1.6)
    m = glass_material(PAGE)
    screen.data.materials[0] = m
    target = rm.on_screen(L, .135, .14)
    x, up, out = (L.to_3x3() @ Vector((1, 0, 0))).normalized(), (L.to_3x3() @ Vector((0, 1, 0))).normalized(), (L.to_3x3() @ Vector((0, 0, 1))).normalized()
    yaw, pitch, dist = math.radians(19), math.radians(5), .26
    loc = target + (x * math.sin(yaw) + out * math.cos(yaw)) * dist * math.cos(pitch) - up * math.sin(pitch) * dist
    camera(loc, target, 50.0, 7.1, rm.on_screen(L, .075, .18))
    strength = m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
    bulb, shade = bpy.data.objects["lamp"], bpy.data.objects["lamp_shade"]
    for name in PASSES:
        if name == "room":
            strength.default_value = 0.0
            render("s4_screen_room")
        elif name == "panel":
            strength.default_value = 1.0
            bulb.hide_render = shade.hide_render = True
            world(0.0)
            for page, name in JOBS or [(PAGE, NAME or "s4_screen_panel")]:
                shows(m, page)
                render(name, denoise=False)


def two_term():
    """二's eighth: the same display in the same room, near, from a little right and below. at: the point of the screen held at hold in the frame. Until 2026-11-08 that was printf's p
    at film_docs3d.TWO_AT (.44, .30); since, the middle of hello.c at film_docs3d.TWO_MID, where 二 now has the middle of every thing (film_screens.py two_mid)."""
    sc = setup()
    rm.room()
    rm.table()
    screen, corners, L = rm.display()
    rm.things()
    bpy.data.objects["outside"].hide_render = True
    world(.006, (.6, .7, 1.0))
    floor_lamp(-.37, -1.6)
    m = glass_material(PAGE)
    screen.data.materials[0] = m
    u, v = [float(c) for c in str(opt("at", "0.0387,0.1347")).split(",")]
    target = rm.on_screen(L, u, v)
    x, up, out = (L.to_3x3() @ Vector((1, 0, 0))).normalized(), (L.to_3x3() @ Vector((0, 1, 0))).normalized(), (L.to_3x3() @ Vector((0, 0, 1))).normalized()
    yaw, pitch, field, lens = math.radians(14), math.radians(9), .06, 85.0
    dist = field * lens / 20.25
    loc = target + (x * math.sin(yaw) + out * math.cos(yaw)) * dist * math.cos(pitch) - up * math.sin(pitch) * dist
    cam = camera(loc, target, lens, 11.0, target)
    hx, hy = [float(c) for c in str(opt("hold", ".44,.30")).split(",")]
    cam.data.shift_x, cam.data.shift_y = .5 - hx, (hy - .5) * 9 / 16
    strength = m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
    bulb, shade = bpy.data.objects["lamp"], bpy.data.objects["lamp_shade"]
    for name in PASSES:
        if name == "room":
            strength.default_value = 0.0
            render("two_term_room")
        elif name == "panel":
            strength.default_value = 1.0
            bulb.hide_render = shade.hide_render = True
            world(0.0)
            render("two_term_panel", denoise=False)


# ---------------------------------------------------------------- s4_key

def ring(x0, x1, y0, y1, r, corner=8, side=10):
    """A rounded rectangle, counter-clockwise, always of the same number of points: so that one inside another can be joined."""
    pts = []
    corners = ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180))
    for n, (cx, cy, a0) in enumerate(corners):
        for i in range(corner + 1):
            a = math.radians(a0 + 90 * i / corner)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        nx, ny, na = corners[(n + 1) % 4]
        a, b = math.radians(a0 + 90), math.radians(na)
        ax, ay = cx + r * math.cos(a), cy + r * math.sin(a)
        bx, by = nx + r * math.cos(b), ny + r * math.sin(b)
        for i in range(1, side):
            k = i / side
            pts.append((ax + (bx - ax) * k, ay + (by - ay) * k))
    return pts


def keycap(name, x0, y0, x1, y1, mat, matrix=None):
    """One cap: its wall down into the hole, the top edge rolled, and across the top a very shallow dish that runs off both sides, as on the real ones. Coordinates of the keyboard, the aluminium at height 0."""
    h, roll, dish = kt.CAP_H, .00026, .00014
    H = y1 - y0
    yc = (y0 + y1) / 2
    d_max = min(x1 - x0, H) / 2 - .0002
    rings = [(0.0, -.0022, False), (0.0, h - roll, True)]
    for a in (22, 45, 68, 90):
        rings.append((roll * (1 - math.cos(math.radians(a))), h - roll + roll * math.sin(math.radians(a)), True))
    d = roll + .00045
    while d < d_max:
        rings.append((d, h, True))
        d *= 1.7
    verts, uvs = [], []
    for inset, z, dished in rings:
        p = ring(x0 + inset, x1 - inset, y0 + inset, y1 - inset, max(kt.CAP_R - inset, .00006))
        for a, b in p:
            sink = dish * (1 - (2 * (b - yc) / H) ** 2) if dished else 0.0
            verts.append((a, b, z - sink))
            uvs.append(((a - kt.AREA[0]) / (kt.AREA[1] - kt.AREA[0]), (b - kt.AREA[2]) / (kt.AREA[3] - kt.AREA[2])))
    k = len(verts) // len(rings)
    cx, cy = (x0 + x1) / 2, yc
    verts.append((cx, cy, h - dish))
    uvs.append(((cx - kt.AREA[0]) / (kt.AREA[1] - kt.AREA[0]), (cy - kt.AREA[2]) / (kt.AREA[3] - kt.AREA[2])))
    faces = []
    for r in range(len(rings) - 1):
        for i in range(k):
            j = (i + 1) % k
            faces.append((r * k + i, r * k + j, (r + 1) * k + j, (r + 1) * k + i))
    last = (len(rings) - 1) * k
    faces += [(last + i, last + (i + 1) % k, len(verts) - 1) for i in range(k)]
    o = mesh_object(name, verts, faces, mat, uv=uvs, matrix=matrix)
    o.data.shade_smooth()
    o.data.set_sharp_from_angle(angle=math.radians(50))
    return o


def cap_material():
    m = printed("cap", os.path.join(kt.TEX, "keys_macro.png"), .42, .5)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    r = nt.nodes.new("ShaderNodeTexImage")
    r.image = tex(os.path.join(kt.TEX, "keys_macro_rough.png"), "Non-Color")
    nt.links.new(nt.nodes.new("ShaderNodeUVMap").outputs[0], r.inputs[0])
    nt.links.new(r.outputs[0], b.inputs["Roughness"])
    # white ABS is not quite smooth, but only a little: its grain, finer than a legend's stroke
    return noisy(m, 26000, .018, .00004, 3)


def deck(caps):
    """The aluminium: one sheet, a hole for every cap, on a dark tray. Holes cut by a boolean and their edges broken."""
    x0, x1, y0, y1 = -.2755, .0017, -.0568, .0581
    alu = noisy(surface("deck", (.70, .71, .72), .50, 1.0, .5), 30000, .02, .00006, 3)
    plate = slab("deck", x0, x1, y0, y1, -.0009, 0, .0095, alu, 0, n=8)
    verts, faces = [], []
    for name, _, _, a, b, c, d in caps:
        w = kt.WELL
        p = ring(a - w, c + w, b - w, d + w, kt.CAP_R + w)
        k, at = len(p), len(verts)
        verts += [(u, v, -.004) for u, v in p] + [(u, v, .004) for u, v in p]
        faces.append(tuple(at + i for i in range(k - 1, -1, -1)))
        faces.append(tuple(at + k + i for i in range(k)))
        faces += [(at + i, at + (i + 1) % k, at + k + (i + 1) % k, at + k + i) for i in range(k)]
    cutter = mesh_object("holes", verts, faces, alu)
    cutter.hide_render = True
    b = plate.modifiers.new("holes", "BOOLEAN")
    b.operation, b.object = "DIFFERENCE", cutter
    plate.data.shade_smooth()
    plate.data.set_sharp_from_angle(angle=math.radians(30))
    bevelled(plate, .00006, 2, 30)
    tray = surface("tray", (.012, .012, .013), .55)
    rm.box("tray", x0 + .001, x1 - .001, y0 + .001, y1 - .001, -.0030, -.0022, tray)
    # the side of the keyboard under the sheet, falling to the table
    rm.box("keyboard_side", x0, x1, y0, y1, -.0085, -.0009, surface("side", (.78, .80, .82), .3, 1.0), .0006)


def s4_key():
    sc = setup()
    caps = kt.keycaps()
    deck(caps)
    cap = cap_material()
    ret = None
    for name, _, _, a, b, c, d in caps:
        o = keycap(f"cap_{name}", a, b, c, d, cap)
        if name == "return":
            ret, key = (a, b, c, d), o
    wood = table_material()
    x0, x1, y0, y1 = -1.0, 1.0, -.6, .8
    slab("table", x0, x1, y0, y1, -.04, -.0085, .004, wood, 0, n=3, uv=(x0, x1, y1, y0))
    a, b, c, d = ret
    # the lamp, warm, a shaded one on the right: a pool on the return key that dies away over the keyboard
    lamp = bpy.data.lights.new("lamp", "SPOT")
    lamp.spot_size, lamp.spot_blend, lamp.shadow_soft_size = math.radians(22), 1.0, .11
    lamp.energy, lamp.color = 11.5, (1.0, .81, .62)
    lo = link(bpy.data.objects.new("lamp", lamp))
    lo.location = (.30, -.12, .44)
    lo.rotation_euler = (Vector((-.03, 0, .0)) - lo.location).normalized().to_track_quat("-Z", "Y").to_euler()
    # and the same lamp on the polish: only where a cap gives it back to the lens
    gleam = bpy.data.lights.new("gleam", "AREA")
    gleam.shape, gleam.size, gleam.size_y = "ELLIPSE", .05, .05
    gleam.energy, gleam.color = .10, (1.0, .80, .60)
    ge = link(bpy.data.objects.new("gleam", gleam))
    ge.visible_diffuse = False
    # the screen behind the keyboard, with a Terminal on it: not much, and cool
    glow = bpy.data.lights.new("glow", "AREA")
    glow.shape, glow.size, glow.size_y = "RECTANGLE", .6, .34
    glow.energy, glow.color = .22, (.72, .82, 1.0)
    go = link(bpy.data.objects.new("glow", glow))
    go.location = (-.138, .265, .30)
    go.rotation_euler = (math.radians(90), 0, 0)
    world(.02, (1.0, .84, .68))
    # where the key is when it starts and when it has stopped both in focus, with its edges: the shot is that they move. Square on to the rows, so that the whole row can be read
    focus = Vector((c - .0067, b + .0070, kt.CAP_H - .0005))
    aim = Vector((-.0395, -.0015, 0))
    az, el, dist = math.radians(1), math.radians(33), .215
    loc = aim + Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))) * dist
    ge.location = focus + Vector((-math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))) * .55
    ge.rotation_euler = (focus - ge.location).normalized().to_track_quat("-Z", "Y").to_euler()
    camera(loc, aim, 85.0, 6.3, focus)
    for t, name in JOBS or [(T, NAME or "s4_key")]:
        key.matrix_world = Matrix.Translation((0, 0, -kt.travel(float(t))))     # nobody presses it: it goes down by itself, level
        render(name)


# ---------------------------------------------------------------- s5_post

def s5_post():
    setup()
    world(0.0)
    PIXEL = kt.PIXEL
    x0, y0, x1, y1 = kt.S5_CROP
    w, h = (x1 - x0) * PIXEL, (y1 - y0) * PIXEL
    # one metre to the panel, its pixels from the top left: centred here on the crop's middle
    verts = [(-w / 2, 0, -h / 2), (w / 2, 0, -h / 2), (w / 2, 0, h / 2), (-w / 2, 0, h / 2)]
    m = emitting("panel", PAGE, GAIN)
    mesh_object("panel", verts, [(0, 1, 2, 3)], m, uv=[(0, 0), (1, 0), (1, 1), (0, 1)])
    # the frame's middle, the text and the button both near the focus
    mid = lambda px, py: Vector(((px - (x0 + x1) / 2) * PIXEL, 0, -(py - (y0 + y1) / 2) * PIXEL))
    target = mid(560, 242)
    focus = mid(560, 250)
    yaw, pitch, dist = math.radians(9), math.radians(5), .278
    if SENT:
        # the second look, nearer, the foot of the empty box above, and its focus on the line that grows
        target, focus, dist = mid(620, 288), mid(500, 385), .25
    loc = target + Vector((-math.sin(yaw) * math.cos(pitch), -math.cos(yaw) * math.cos(pitch), -math.sin(pitch))) * dist
    camera(loc, target, 50.0, 8.0, focus)
    for page, name in JOBS or [(PAGE, (NAME or "s5_post") + "_panel")]:
        shows(m, page)
        render(name, denoise=False)


if __name__ == "__main__":
    dict(s4_screen=s4_screen, s4_key=s4_key, s5_post=s5_post, two_term=two_term)[SHOT]()
