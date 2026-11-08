"""The room of 三 at night, for 十 and for the reverse shot of 十四. The set is film_room3d.py's, built by its own functions and not touched: the same table, display, books, cup and window. What the night adds is here:

    a floor lamp standing just right of the table, outside the day's wide frame, so the day never saw it
    the block across the road outside the window, its windows lit (film_room_tex.py night)
    the half of the room the day's camera had behind it: the back wall with its door, half open on a dark hall, a switch by the door

Inside Blender, from manim/scenes:

    Blender -b --factory-startup -P film_room_night3d.py -- --shot=ten --at=0 --passes=lamp,glow,city [--pct=100] [--samples=256]
    Blender -b --factory-startup -P film_room_night3d.py -- --shot=reverse --passes=glow,city,hall [--pct=100] [--samples=512]
    Blender -b --factory-startup -P film_room_night3d.py -- --shot=ten --frames=0-365 --beauty [--pct=50] [--samples=96]
    Blender -b --factory-startup -P film_room_night3d.py -- --shot=reverse2 --view=A --passes=glow,city,hall [--chair=X,Y,YAW] [--name=rev2A]    # the reverse, again, with the chair; views D-F (2026-11-08) from the glass's middle, the chair at the desk
    Blender -b --factory-startup -P film_room_night3d.py -- --shot=chaircheck --view=A [--chair=X,Y,YAW]      # does any other shot see the chair: prints, renders nothing

Passes are kept apart in linear light as film_room3d.py keeps them, so that film_room_night.py can balance them without coming back here:
    lamp    the floor lamp alone
    glow    the screen's light alone, what it shows given by ui/NAME.png
    city    what comes in at the window: the lit block across the road and the sky's last warmth
    hall    the light in the hall beyond the door (only the reverse uses it)
A pass goes to ink/film_out/film/room_night/plates/SHOT_PASS.npy (SHOT_PASS_fNNNN.npy for a frame of a sequence), float16, top row first; SHOT.json (SHOT_fNNNN.json) says where the screen's corners fall in the frame.
--beauty renders the passes together at the strengths film_room_night.py settled (BEAUTY), denoised, one picture a frame, for the push of 十.
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

import film_room3d as R
from film_room3d import (BACK_Y, CEIL, DISP_H, DISP_UP, DISP_X, DISP_Y, LEFT_X, RIGHT_X, ZT, box, link, lathe, slab, surface, tube)
from ink3d import ROOT, reset

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
        if a == f"--{name}" and isinstance(default, bool):
            return True
    return default


SHOT = opt("shot", "ten")
PASSES = opt("passes", "lamp,glow,city").split(",")
PCT, SAMPLES = opt("pct", 100), opt("samples", 256)
AT = opt("at", 0.0)                  # how far along the push of 十, 0 to 1
FRAMES = opt("frames", "")
BEAUTY = dict(lamp=2.4, glow=1.6, city=.5, hall=0.0)       # film_room_night.NIGHT's balance of the passes, so a beauty frame is the passes summed as the stills sum them
OUT = os.path.join(ROOT, "ink", "film_out", "film", "room_night")
PLATES = os.path.join(OUT, "plates")
UI = os.path.join(OUT, "ui")

# the floor lamp: a weighted foot, a pole, a drum of linen; it stands right of the table and a little behind its front edge
LAMP_X, LAMP_Y = 1.40, .30
SHADE_Z, SHADE_R, SHADE_H = 1.36, .20, .30
WARM = (1.0, .78, .58)               # a 3000 K bulb behind linen

# the back wall's door: the opening in the wall, the leaf hinged on its left as you face it from the desk (the right as the screen sees it), opened into the room, so that from the screen you look into the opening
DOOR_X0, DOOR_X1, DOOR_H = -.55, .33, 2.04
DOOR_OPEN = 52.0                     # degrees
HALL = 1.25                          # how wide the hall is, and it runs off left and right beyond the door

# 十: the push. It starts on the day's wide frame exactly, from the same place through the same lens, and goes in along the line to a point of the screen, which stays where it is in the frame: the frame only gets bigger round it
PUSH_FROM = dict(loc=(-0.30, -2.20, 1.28), centre=(0.12, 0.86), lens=46.0)
PUSH_ON = (.5, .30)                  # the point of the screen held still, as parts of its width and height from its top left: high, so that the screen grows down into the frame and its top stays in
PUSH_TO = .455                       # how far from that point the camera ends, as a part of where it starts: 2.2 times as big, 1.07 times a second
FSTOP = 2.8

# the reverse: from the glass, looking out into the room. A little right of the screen's middle and below the last lines of the reply, looking down a little, so that the cup and the keyboard's back are near at the foot of the frame and the door far in its middle
REVERSE = dict(loc=(.26, DISP_Y - .006, ZT + DISP_UP + .145), yaw=-3.0, pitch=-16.0, lens=20.0, focus=.30, fstop=5.6)

# the reverse again (2026-11-07), --shot=reverse2 --view=A|B|C: the screen is the only light, and the lens is where it is, so what is nearest is brightest. On the glass itself cup,
# mouse and keyboard (20 to 30 cm off, 25 below) cannot all be whole in any lens under about 12 mm, so the lens stands back behind the glass (back, metres), the display
# unseen by it but still giving its light: everything it sees is still seen from the screen. x, z: where on the glass it is; yaw towards +x turns it to the frame's left.
# The chair: pushed back from the table and turned towards the door. Where it may stand is narrow: 十二's 人 (from the front left, "no chair") sees the floor in front
# of the table as far as the line y = -1.30 + .342 (x + 1.32), and 三's wide sees anything higher than .80 nearer than y = -.94; --shot=chaircheck tests every camera
# B is the one chosen; its pitch puts the top of the door's casing (z 2.11) about 30 px under the top of the frame and the cup's foot still 60 px above its bottom
VIEWS = dict(
    A=dict(back=.30, x=.16, z=.98, yaw=1.0, pitch=-6.0, lens=20.0, focus=.58, fstop=8.0, chair=(.75, -1.25, 30.0)),
    B=dict(back=.38, x=.17, z=1.02, yaw=1.0, pitch=-5.5, lens=22.0, focus=.66, fstop=8.0, chair=(.62, -1.35, 34.0)),
    C=dict(back=.30, x=.20, z=.99, yaw=0.0, pitch=-4.0, lens=20.0, focus=.58, fstop=8.0, chair=(.35, -1.60, 45.0)),
    # the reverse a third time (2026-11-08), D, E, F: the screen's own eye, in the middle of the glass, looking level out at the empty seat. B stood back and high and read as
    # a product shot of the desk with a chair by the far wall. Here the chair is at the desk, pushed back and turned a little as one leaves it, in the middle of the
    # frame and in focus, and the keyboard's far edge is a soft band at the foot. The door behind is only ajar (door: degrees open), and only the hall's light from just
    # outside it is on (hall: which of back_wall's lights): the one round the corner lit the whole leaf, a lit door instead of a gap of light.
    # Two things fix how near the chair can be. From the eye, 30 cm over the table, the table's front edge hides whatever is lower than the line over it: the seat
    # (.455 high) shows only once its front is about .7 beyond that edge. And 十二's 人 sees the floor up to about y = -.84 at x = .05. Nearer than y = -.85 the chair is
    # seen there, or its seat is hidden here; --shot=chaircheck passes D-F (十三's crab room, without door or chair, sees any chair, as it saw B's).
    # Turned towards +x (yaw below zero): the way the door's gap is, as if one stood up and went out that way.
    # Wherever the keyboard's far edge comes into the frame, the cup's side comes in at its left edge, and the notebook's corner at its right: so for this shot the
    # cup stands a few centimetres further right and the notebook (its pen and the letter under it) further left, as they would after an evening's use (moved: metres)
    # dark: the room's walls, ceiling and door painted this much darker (dark_room); chair_tint: the chair's wood a little lighter than B's, so the screen's light on it reads
    # E is the one chosen (2026-11-08), as it was drafted but for four things. The chair 15 cm further back: at -1.00 its seat's front lay on the table's edge, as if
    # the chair stood on the table; at -1.15 the seat's lower edge is 20 px (of a 960 frame) clear of it, the legs going on down behind. F's door, open 30, its gap
    # and the warm wedge on the floor readable beside the chair. The mouse out of the frame (12 cm towards the screen, 6 right), whose soft white tip in the corner
    # read as a smudge; and the switch by the door gone (away: parts of the room not in this shot), a lit grey square where the background is to be only wall and door.
    D=dict(back=.22, x=.03, z=1.037, yaw=0.0, pitch=-8.0, lens=24.0, focus=1.52, fstop=2.8, chair=(-.15, -.95, -12.0), door=25.0, hall=("hall_spill",),
           moved=dict(cup=(.09, 0.0), notebook=(-.12, 0.0)), dark=.30, chair_tint=1.8),
    E=dict(back=.27, x=.03, z=1.037, yaw=0.0, pitch=-8.0, lens=28.0, focus=1.95, fstop=2.8, chair=(-.10, -1.15, -15.0), door=30.0, hall=("hall_spill",),
           moved=dict(cup=(.09, 0.0), notebook=(-.12, 0.0), mouse=(.06, .12)), dark=.30, chair_tint=1.8, away=("switch",)),
    F=dict(back=.28, x=.03, z=1.037, yaw=0.0, pitch=-7.0, lens=26.0, focus=1.62, fstop=2.8, chair=(-.22, -1.00, -18.0), door=30.0, hall=("hall_spill",),
           moved=dict(cup=(.09, 0.0), notebook=(-.12, 0.0)), dark=.30, chair_tint=1.8),
)
# the lens may not stand back further than about .38: the window's wall is at .80. Nor look so far down that the frame's foot reaches the table behind the glass, which
# the screen does not light (the lowest ray must meet the table before y = .34)
PARTS = dict(cup=("cup", "tea", "stain", "handle"), notebook=("notebook", "pen", "tip", "letter"), mouse=("mouse", "mouse_foot"))      # what of film_room3d.things a view may move, by its parts' names
SEAT_H, SEAT_W, SEAT_D, SEAT_T, BACK_TOP = .455, .430, .410, .024, .800       # an ordinary wooden chair, metres
CHAIR_TINT = 1.3        # the table's wood a shade lighter: 1.6 m from the screen the table's own shade went to a dark silhouette, and the brief wants the light on its seat and back
HALL_DARK = .30         # the reverse, again: the hall painted this much of the room's plaster, so beyond the door is only a very dark cool grey, never one more lit wall


def plain(name, colour, rough=.6, spec=.3):
    return R.noisy(surface(name, colour, rough, 0, spec), 300, .04, .0006)


# ---------------------------------------------------------------- what the night adds

def floor_lamp():
    """Black steel and white linen, the bulb inside the drum. The drum glows; its open top and foot throw the light up the wall and down to the floor."""
    steel = surface("lamp_steel", (.025, .024, .023), .35, .9)
    lathe("lamp_foot", [(.0, .0), (.135, .0), (.14, .006), (.135, .018), (.02, .024), (.0, .024)], LAMP_X, LAMP_Y, 0, steel)
    tube("lamp_pole", [(LAMP_X, LAMP_Y, .02), (LAMP_X, LAMP_Y, SHADE_Z + .05)], .009, steel, 3)
    # the shade: cloth over a wire ring at each end, its seam towards the wall
    m = bpy.data.materials.new("shade")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    d, tr, mix = nt.nodes.new("ShaderNodeBsdfDiffuse"), nt.nodes.new("ShaderNodeBsdfTranslucent"), nt.nodes.new("ShaderNodeMixShader")
    d.inputs[0].default_value = (.80, .76, .68, 1)
    tr.inputs[0].default_value = (.95, .86, .70, 1)
    mix.inputs[0].default_value = .55
    nt.links.new(d.outputs[0], mix.inputs[1])
    nt.links.new(tr.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], o.inputs[0])
    z0, z1 = SHADE_Z - SHADE_H / 2, SHADE_Z + SHADE_H / 2
    shade = lathe("lamp_shade", [(SHADE_R, z0), (SHADE_R * .985, z1)], LAMP_X, LAMP_Y, 0, m, 96)
    sol = shade.modifiers.new("cloth", "SOLIDIFY")
    sol.thickness = .0015
    for z, r in ((z0, SHADE_R), (z1, SHADE_R * .985)):
        bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=.0016, location=(LAMP_X, LAMP_Y, z), major_segments=96, minor_segments=6)
        bpy.context.object.data.materials.append(steel)
    for a in (0, 120, 240):
        q = math.radians(a)
        tube("lamp_spoke", [(LAMP_X, LAMP_Y, SHADE_Z + .05), (LAMP_X + SHADE_R * .98 * math.cos(q), LAMP_Y + SHADE_R * .98 * math.sin(q), z1)], .0012, steel, 2)
    b = bpy.data.lights.new("bulb", "POINT")
    b.shadow_soft_size, b.energy, b.color = .035, 55.0, WARM
    bo = link(bpy.data.objects.new("bulb", b))
    bo.location = (LAMP_X, LAMP_Y, SHADE_Z - .02)
    glass = surface("bulb_glass", (1, 1, 1), .2)
    glass.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (*WARM, 1)
    glass.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 6.0
    R.ellipsoid("bulb_glass", (.03, .03, .036), (LAMP_X, LAMP_Y, SHADE_Z - .02), glass, n=24)
    return [bo], [o for o in bpy.data.objects if o.name.startswith("bulb_glass")]


def back_wall(opened=DOOR_OPEN):
    """The wall behind the day's camera, with the door in it. The plain wall film_room3d.py built there goes, and this stands in its place: the same plaster, a painted casing round the opening, the leaf half open out into the hall, a lever handle, and a switch. Out in the hall, the leaf's face takes the hall's light and stands against the dark of the room as a door; opened into the room it only showed the lens its unlit side, a dark slab that read as a box.
    opened: how far the leaf is open, degrees (the reverse from the glass has it only ajar)."""
    bpy.data.objects.remove(bpy.data.objects["wall_back"])
    wall = R.plaster()
    t = .2
    y0, y1 = BACK_Y - t, BACK_Y
    box("wall_back_l", LEFT_X - .2, DOOR_X0, y0, y1, 0, CEIL, wall)
    box("wall_back_r", DOOR_X1, RIGHT_X + .2, y0, y1, 0, CEIL, wall)
    box("wall_back_top", DOOR_X0, DOOR_X1, y0, y1, DOOR_H, CEIL, wall)
    paint = R.noisy(surface("door_paint", (.74, .72, .67), .45, 0, .4), 40, .05, .001)
    c = .07
    box("casing_l", DOOR_X0 - c, DOOR_X0, y1, y1 + .014, 0, DOOR_H, paint, .002)      # up to the head's casing, not into it: where they overlapped the corners rendered black
    box("casing_r", DOOR_X1, DOOR_X1 + c, y1, y1 + .014, 0, DOOR_H, paint, .002)
    box("casing_t", DOOR_X0 - c, DOOR_X1 + c, y1, y1 + .014, DOOR_H, DOOR_H + c, paint, .002)
    box("jamb_l", DOOR_X0, DOOR_X0 + .012, y0, y1, 0, DOOR_H, paint)
    box("jamb_r", DOOR_X1 - .012, DOOR_X1, y0, y1, 0, DOOR_H, paint)
    box("jamb_t", DOOR_X0, DOOR_X1, y0, y1, DOOR_H - .012, DOOR_H, paint)
    box("door_skirt_l", LEFT_X, DOOR_X0 - c, y1, y1 + .012, 0, .09, paint)
    box("door_skirt_r", DOOR_X1 + c, RIGHT_X, y1, y1 + .012, 0, .09, paint)
    # the leaf: a flat slab with two shallow panels, turning on the jamb at DOOR_X0 out into the hall; x along it from the hinge, its face to the room at y = 0
    w, th = DOOR_X1 - DOOR_X0 - .024, .038
    a = math.radians(opened)
    M = Matrix.Translation((DOOR_X0 + .012, y0, 0)) @ Matrix.Rotation(-a, 4, "Z")
    leaf = R.noisy(surface("door_leaf", (.70, .68, .63), .5, 0, .35), 25, .03, .0012)
    box("door", 0, w, -th, 0, .008, DOOR_H - .014, leaf, .002, M)
    line = surface("door_line", (.60, .58, .54), .55, 0, .3)
    for z0, z1 in ((.18, 1.0), (1.12, 1.88)):
        for side in (-th - .0015, 0):
            box("door_panel", .09, w - .09, side, side + .0015, z0, z1, line, .0005, M)
    steel = surface("handle", (.55, .55, .56), .25, 1.0)
    for side, out in ((-th - .004, -.05), (.004, .05)):
        tube("handle", [tuple(M @ Vector(p)) for p in ((w - .07, side, 1.0), (w - .07, side + out, 1.0), (w - .19, side + out, 1.0))], .0085, steel, 3)
    # the switch beside the opening, at the height a hand finds it
    box("switch", DOOR_X1 + .11, DOOR_X1 + .19, y1, y1 + .009, 1.26, 1.34, surface("switch", (.80, .79, .76), .4, 0, .4), .002)
    # the hall: a floor, a far wall and a ceiling going off both ways, nobody, and a little warm light from round the corner at its right
    hall_wall = R.plaster()
    hy0 = y0 - HALL
    box("hall_floor", DOOR_X0 - 2.5, DOOR_X1 + 2.5, hy0 - .2, y0, -.1, 0, surface("hall_floor", (.22, .15, .09), .5))
    box("hall_far", DOOR_X0 - 2.5, DOOR_X1 + 2.5, hy0 - .2, hy0, 0, CEIL, hall_wall)
    box("hall_ceiling", DOOR_X0 - 2.5, DOOR_X1 + 2.5, hy0 - .2, y0, CEIL, CEIL + .1, surface("hall_ceiling", (.70, .68, .64), .9, 0, .2))
    box("hall_end_l", DOOR_X0 - 2.7, DOOR_X0 - 2.5, hy0 - .2, y0, 0, CEIL, hall_wall)
    box("hall_end_r", DOOR_X1 + 2.5, DOOR_X1 + 2.7, hy0 - .2, y0, 0, CEIL, hall_wall)
    h = bpy.data.lights.new("hall", "AREA")
    h.shape, h.size, h.size_y = "RECTANGLE", .5, 1.6
    h.energy, h.color = 9.0, (1.0, .72, .46)
    ho = link(bpy.data.objects.new("hall", h))
    ho.location = (DOOR_X1 + 2.3, hy0 + HALL / 2, 1.0)
    ho.rotation_euler = (0, math.radians(90), 0)
    ho.visible_glossy = False
    ho["base"] = 9.0
    # and the hall's own ceiling light, left on just outside the door: through the gap it lays a wedge of warm light on the room's floor
    s = bpy.data.lights.new("hall_spill", "AREA")
    s.shape, s.size, s.size_y = "RECTANGLE", .28, .28
    s.color = (1.0, .74, .50)
    so = link(bpy.data.objects.new("hall_spill", s))
    so.location = ((DOOR_X0 + DOOR_X1) / 2 + .15, y0 - .55, CEIL - .03)
    so.rotation_euler = (math.radians(28), 0, 0)               # down, and leaning towards +y, through the door
    so.visible_glossy = False
    so["base"] = 14.0
    return [ho, so]


# ---------------------------------------------------------------- the chair, for the reverse only

def obj_wood(grain, across, at, tint=1.0):
    """film_room3d's table wood, run along the part's own axes instead of the world's, so that a part turned with the chair keeps its grain along it."""
    m = R.table_material()
    nt = m.node_tree
    pos = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], pos.inputs[0])

    def run(axes, size, start):
        s = pos.outputs[axes[0]]
        for a in axes[1:]:
            s = R.calc(nt, "ADD", s, pos.outputs[a])
        return R.calc(nt, "ADD", R.calc(nt, "DIVIDE", s, size), start)

    uv = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(run(grain, R.TABLE[1] - R.TABLE[0], at[0]), uv.inputs[0])
    nt.links.new(run(across, R.TABLE[3] - R.TABLE[2], at[1]), uv.inputs[1])
    for t in [n for n in nt.nodes if n.bl_idname == "ShaderNodeTexImage"]:
        nt.links.new(uv.outputs[0], t.inputs[0])
    b = nt.nodes["Principled BSDF"]
    if tint != 1.0:
        nt.links.new(R.blended(nt, 1.0, b.inputs["Base Color"].links[0].from_socket, (tint, tint, tint), "MULTIPLY"), b.inputs["Base Color"])
    return m


def rod(name, p0, p1, r0, r1, mat, n=24, rings=6):
    """A turned round part from p0 to p1 (world), r0 thick at p0 and r1 at p1, its own z along it, so that the grain runs along it."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    verts, faces = [], []
    for j in range(rings + 1):
        f = j / rings
        r = r0 + (r1 - r0) * f
        verts += [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), d.length * f) for i in range(n)]
    for j in range(rings):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            faces.append((a, b, b + n, a + n))
    faces += [tuple(range(n - 1, -1, -1)), tuple(range(rings * n, rings * n + n))]
    o = R.mesh_object(name, verts, faces, mat, smooth=True, matrix=Matrix.Translation(p0) @ d.to_track_quat("Z", "Y").to_matrix().to_4x4())
    o.data.set_sharp_from_angle(angle=math.radians(50))
    return R.bevelled(o, min(r0, r1) * .15, 2, 50)


def dished(name, w, d, r, t, dish, mat, M, rings=7, n=8):
    """The seat: a board t thick with round corners, its top hollowed by dish where one sits, a little behind its middle."""
    out = R.outline(-w / 2, w / 2, -d / 2, d / 2, r, n)
    k = len(out)
    cx, cy = 0.0, -.03
    verts, faces = [], []
    for j in range(rings):
        s = 1 - j / rings
        verts += [(cx + (x - cx) * s, cy + (y - cy) * s, t - dish * (1 - s * s)) for x, y in out]
    mid = len(verts)
    verts.append((cx, cy, t - dish))
    for j in range(rings - 1):
        for i in range(k):
            a, b = j * k + i, j * k + (i + 1) % k
            faces.append((a, b, b + k, a + k))
    last = (rings - 1) * k
    faces += [(last + i, last + (i + 1) % k, mid) for i in range(k)]
    base = len(verts)
    verts += [(x, y, 0.0) for x, y in out]
    faces.append(tuple(range(base + k - 1, base - 1, -1)))
    faces += [(base + i, base + (i + 1) % k, (i + 1) % k, i) for i in range(k)]
    o = R.mesh_object(name, verts, faces, mat, smooth=True, matrix=M)
    o.data.set_sharp_from_angle(angle=math.radians(40))
    return R.bevelled(o, .005, 3, 40)


def bent_board(name, half, y_mid, z0, z1, rake, radius, t, mat, M, nu=28):
    """The back: a board bent round the sitter, its middle furthest back, raked back as the posts are. y_mid is its face to the sitter, at its middle."""
    phi = math.asin(half / radius)
    zm = (z0 + z1) / 2
    verts = []
    for side in (0, 1):
        rr = radius + side * t
        for z in (z0, z1):
            for i in range(nu + 1):
                p = -phi + 2 * phi * i / nu
                verts.append((rr * math.sin(p), y_mid + radius - rr * math.cos(p) - (z - zm) * math.tan(rake), z))
    ix = lambda side, top, i: side * 2 * (nu + 1) + top * (nu + 1) + i
    faces = []
    for i in range(nu):
        faces.append((ix(0, 0, i), ix(0, 1, i), ix(0, 1, i + 1), ix(0, 0, i + 1)))
        faces.append((ix(1, 0, i), ix(1, 0, i + 1), ix(1, 1, i + 1), ix(1, 1, i)))
        faces.append((ix(0, 1, i), ix(1, 1, i), ix(1, 1, i + 1), ix(0, 1, i + 1)))
        faces.append((ix(0, 0, i), ix(0, 0, i + 1), ix(1, 0, i + 1), ix(1, 0, i)))
    faces.append((ix(0, 0, 0), ix(1, 0, 0), ix(1, 1, 0), ix(0, 1, 0)))
    faces.append((ix(0, 0, nu), ix(0, 1, nu), ix(1, 1, nu), ix(1, 0, nu)))
    o = R.mesh_object(name, verts, faces, mat, smooth=True, matrix=M)
    o.data.set_sharp_from_angle(angle=math.radians(40))
    return R.bevelled(o, .003, 3, 40)


def moved(objs, names, dx, dy):
    """The parts of objs named names (Blender's .001 endings aside) moved dx, dy metres along the table, for one shot."""
    for o in objs:
        if o.name.split(".")[0] in names:
            o.location.x += dx
            o.location.y += dy


def paint_darker(o, k):
    """o given its own copy of its material, its colour k times as light."""
    m = o.material_slots[0].material.copy()
    nt = m.node_tree
    src = nt.nodes["Principled BSDF"].inputs["Base Color"]
    if src.links:
        nt.links.new(R.blended(nt, 1.0, src.links[0].from_socket, (k, k, k), "MULTIPLY"), src)
    else:
        c = src.default_value
        src.default_value = (c[0] * k, c[1] * k, c[2] * k, c[3])
    o.material_slots[0].material = m


def dark_hall(k):
    """The hall's walls and ceiling k times as light as they were (the reverse, again, only). Straight across from the screen, a plaster hall took as much
    of its light as the room's own wall did, and the door opened on one more lit wall."""
    for name in ("hall_far", "hall_end_l", "hall_end_r", "hall_ceiling"):
        o = bpy.data.objects.get(name)
        if o is None or not o.material_slots:
            continue
        paint_darker(o, k)


ROOM_PAINT = ("wall_", "ceiling", "casing_", "jamb_", "door", "switch", "skirt")      # the room's plaster and paint, by the start of its parts' names


def dark_room(k):
    """The room's walls, ceiling and door k times as light (the reverse from the glass only). The screen, a big soft light, lit the white plaster three metres off
    almost as brightly as the chair half as far: the walls handed its light round the room, and the far wall stood as light as the seat. Darker paint, seen only
    in this shot, takes less and hands less on, so the light falls away from the screen as the brief has it."""
    for o in list(bpy.data.objects):
        if o.type == "MESH" and o.material_slots and o.name.split(".")[0].startswith(ROOM_PAINT):
            paint_darker(o, k)


def chair(x, y, yaw, tint=CHAIR_TINT):
    """An ordinary wooden chair, as one is made: a dished seat on four turned legs, the back legs going on up through the seat as posts, a bent board across
    them, rails under the seat, stretchers low down. x, y: the middle of its seat over the floor; yaw: degrees it is turned from facing the table, towards the door;
    tint: its wood against the table's. Returns its parts."""
    before = set(bpy.data.objects.keys())
    M = Matrix.Translation((x, y, 0)) @ Matrix.Rotation(math.radians(yaw), 4, "Z")
    P = lambda px, py, pz: M @ Vector((px, py, pz))
    under = SEAT_H - SEAT_T
    dished("chair_seat", SEAT_W, SEAT_D, .035, SEAT_T, .008, obj_wood("X", "Y", (.31, -.25), tint), M @ Matrix.Translation((0, 0, under)))
    turned = obj_wood("Z", "XY", (.62, -.21), tint)         # the table's own wood: the chair came with it
    for sx in (-1, 1):
        rod("chair_leg", P(sx * .205, .200, 0), P(sx * .188, .178, under), .0128, .0165, turned)          # the front legs, out a little at the foot
        rod("chair_leg", P(sx * .205, -.215, 0), P(sx * .190, -.188, under + .01), .0132, .0168, turned)  # the back legs, up into the seat
        rod("chair_post", P(sx * .190, -.188, under + .005), P(sx * .184, -.262, BACK_TOP), .0160, .0138, turned)   # and on through it as posts, raked back
        rod("chair_rung", P(sx * .199, .192, .155), P(sx * .200, -.206, .155), .0095, .0095, turned)
    rod("chair_rung", P(-.199, -.008, .155), P(.199, -.008, .155), .0088, .0088, turned)
    rail = obj_wood("X", "YZ", (.45, -.30), tint)
    side = obj_wood("Y", "XZ", (.52, -.28), tint)
    box("chair_rail", -.172, .172, .169, .187, under - .060, under, rail, .002, M)
    box("chair_rail", -.174, .174, -.197, -.179, under - .060, under, rail, .002, M)
    for sx in (-1, 1):
        box("chair_rail", min(sx * .179, sx * .197), max(sx * .179, sx * .197), -.172, .162, under - .060, under, side, .002, M)
    # the board meets the posts at their middle height there, its ends on them, its middle 3.75 cm further back
    z0, z1 = .668, .776
    rake = math.atan2(.074, BACK_TOP - under - .005)
    post_y = -.188 - .074 * ((z0 + z1) / 2 - under - .005) / (BACK_TOP - under - .005)
    radius, half = .55, .200
    bent_board("chair_back", half, post_y - radius * (1 - math.cos(math.asin(half / radius))), z0, z1, rake, radius, .019, obj_wood("X", "Z", (.36, -.22), tint), M)
    return [o for o in bpy.data.objects if o.name not in before]


def night_outside():
    """What is outside: the picture of the block across the road at night, as light."""
    o = bpy.data.objects["outside"]
    nt = o.data.materials[0].node_tree
    t = next(n for n in nt.nodes if n.type == "TEX_IMAGE")
    t.image = bpy.data.images.load(os.path.join(R.TEX, "outside_night.png"), check_existing=True)
    t.image.colorspace_settings.name = "sRGB"
    em = next(n for n in nt.nodes if n.type == "EMISSION")
    em.inputs[1].default_value = 1.0
    return em.inputs[1]


def night_screen(screen, picture):
    """The screen gives out what film_room_night.py drew for it."""
    nt = screen.data.materials[0].node_tree
    t = next(n for n in nt.nodes if n.type == "TEX_IMAGE")
    t.image = bpy.data.images.load(os.path.join(UI, picture), check_existing=True)
    return nt.nodes["Principled BSDF"].inputs["Emission Strength"]


# ---------------------------------------------------------------- the cameras

def push_camera(cam, lid, at):
    """The push of 十 at a part of the way: the same lens and shift as the day's wide, the camera on the line to the held point, nearer it by an even ratio a second."""
    loc0 = Vector(PUSH_FROM["loc"])
    on = R.on_screen(lid, *PUSH_ON)
    s = 1 - PUSH_TO ** at                       # the distance to the point falls by the same ratio every second
    loc = loc0 + (on - loc0) * s
    cam.location = loc
    cam.rotation_euler = (math.radians(90), 0, 0)
    d = cam.data
    d.lens = PUSH_FROM["lens"]
    # the shift that puts the held point where the day's wide had it: it does not change as the camera comes along the line
    focus0 = DISP_Y - loc0.y
    wide0 = focus0 * 36 / d.lens
    cx, cz = PUSH_FROM["centre"]
    sx0, sy0 = (cx - loc0.x) / wide0, (cz - loc0.z) / wide0
    # where the held point is from the frame's middle, as a part of the frame's width, at the start; the same all the way
    deep0 = on.y - loc0.y
    px, pz = (on.x - loc0.x) / (deep0 * 36 / d.lens) - sx0, (on.z - loc0.z) / (deep0 * 36 / d.lens) - sy0
    deep = on.y - loc.y
    d.shift_x = (on.x - loc.x) / (deep * 36 / d.lens) - px
    d.shift_y = (on.z - loc.z) / (deep * 36 / d.lens) - pz
    d.dof.use_dof = True
    d.dof.focus_distance = DISP_Y - loc.y
    d.dof.aperture_fstop = FSTOP
    d.dof.aperture_blades = 7


def reverse_camera(cam):
    r = dict(REVERSE)
    if opt("rev", ""):                                       # trying a framing: --rev=dx,dz,yaw,pitch,lens, the move from REVERSE's place and the rest outright
        dx, dz, r["yaw"], r["pitch"], r["lens"] = (float(v) for v in opt("rev", "").split(","))
        r["loc"] = (r["loc"][0] + dx, r["loc"][1], r["loc"][2] + dz)
    cam.location = Vector(r["loc"])
    y, p = math.radians(r["yaw"]), math.radians(r["pitch"])
    look = Vector((math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p)))      # yaw turns it towards +x, the left of the frame; pitch below zero looks down
    cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
    d = cam.data
    d.lens, d.shift_x, d.shift_y = r["lens"], 0.0, 0.0
    d.dof.use_dof = True
    d.dof.focus_distance = r["focus"]
    d.dof.aperture_fstop = r["fstop"]
    d.dof.aperture_blades = 7


def reverse2_camera(cam, v):
    """The second reverse: VIEWS[view], the lens back behind the glass by v["back"]."""
    cam.location = Vector((v["x"], DISP_Y - .006 + v["back"], v["z"]))
    y, p = math.radians(v["yaw"]), math.radians(v["pitch"])
    cam.rotation_euler = Vector((math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p))).to_track_quat("-Z", "Y").to_euler()
    d = cam.data
    d.lens, d.shift_x, d.shift_y = v["lens"], 0.0, 0.0
    d.dof.use_dof = True
    d.dof.focus_distance = v["focus"]
    d.dof.aperture_fstop = v["fstop"]
    d.dof.aperture_blades = 7


def borrowed(fname):
    """Everything another of the room's scripts defines, without the call to main at its end."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    with open(path, encoding="utf-8") as f:
        src = f.read().rstrip()
    assert src.endswith("\nmain()"), fname
    ns = {"__file__": path, "__name__": fname[:-3] + "_parts"}
    exec(compile(src[:-len("main()")], path, "exec"), ns)
    return ns


def check_chair(sc, cam, lid, parts):
    """Whether any shot already made could see the chair: every camera of the room's other shots, straight on and in the display's glass, the chair's every
    point tested for being in the frame and for nothing standing between. 十三's crab has a room of its own (crew C's, in mm, with no door); the chair is put
    where it would stand in it, its table's front edge the same, and tested against its lens at the start, middle and end of its push."""
    from bpy_extras.object_utils import world_to_camera_view
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in parts:
        e = o.evaluated_get(dg)
        me = e.to_mesh()
        pts += [o.matrix_world @ v.co for v in me.vertices]
        e.to_mesh_clear()
    mine = {o.name for o in parts}
    normal = (lid.to_3x3() @ Vector((0, 0, 1))).normalized()
    mid = R.on_screen(lid, .5, .5)
    inv = lid.inverted()
    W, H = sc.render.resolution_x, sc.render.resolution_y
    d = cam.data

    def clear(a, b):
        v = b - a
        hit, _, _, _, ob, _ = sc.ray_cast(dg, a, v.normalized(), distance=max(v.length - 2e-3, 0.0))
        return not hit or ob.name in mine

    def box_of(p):
        if not p:
            return "-"
        xs, ys = [a for a, _ in p], [b for _, b in p]
        return f"x {min(xs):.0f}..{max(xs):.0f} y {min(ys):.0f}..{max(ys):.0f}"

    def seen(label):
        bpy.context.view_layer.update()
        c = cam.matrix_world.translation.copy()
        direct, glass = [], []
        for q in pts:
            v = world_to_camera_view(sc, cam, q)
            if 0 <= v.x <= 1 and 0 <= v.y <= 1 and v.z > 0 and clear(c, q):
                direct.append((v.x * W, (1 - v.y) * H))
            q2 = q - 2 * (q - mid).dot(normal) * normal
            dd = q2 - c
            den = dd.dot(normal)
            if abs(den) < 1e-9:
                continue
            t = (mid - c).dot(normal) / den
            if not 0 < t < 1:
                continue
            g = c + dd * t
            lg = inv @ g
            if not (abs(lg.x) <= R.DISP_W / 2 and R.DISP_UP <= lg.y <= R.DISP_UP + R.DISP_H):
                continue
            v = world_to_camera_view(sc, cam, g)
            if 0 <= v.x <= 1 and 0 <= v.y <= 1 and v.z > 0 and clear(c, g) and clear(g + (q - g).normalized() * 2e-3, q):
                glass.append((v.x * W, (1 - v.y) * H))
        print(f"chaircheck {label:24s} direct {len(direct):5d}/{len(pts)} {box_of(direct):30s} glass {len(glass):5d} {box_of(glass)}", flush=True)

    push_camera(cam, lid, 0.0)
    seen("三 wide = 十 at 0")
    for a in range(1, 11):
        push_camera(cam, lid, a / 10)
        seen(f"十 push {a / 10:.1f}")
    D = borrowed("film_desk3d.py")
    for i in (0, 36, 73, 109, 146, 182, 218):
        cam.location = D["camera_at"](i)
        cam.rotation_euler = (math.radians(90), 0, 0)
        d.lens = D["LENS"]
        d.shift_x, d.shift_y = D["shift"]()
        seen(f"二 desk {i}")
    Nm = borrowed("film_names3d.py")
    for s in ("ren", "qi"):
        Nm["aim"](cam, Nm["SHOTS"][s])
        seen(f"十二 {s}")
    ax = lambda v: (lid.to_3x3() @ Vector(v)).normalized()
    xx, up, out = ax((1, 0, 0)), ax((0, 1, 0)), ax((0, 0, 1))

    def at_screen(u, v, yaw, pitch, dist, lens):
        target = R.on_screen(lid, u, v)
        loc = target + (xx * math.sin(yaw) + out * math.cos(yaw)) * dist * math.cos(pitch) - up * math.sin(pitch) * dist
        cam.location = loc
        cam.rotation_euler = (target - loc).normalized().to_track_quat("-Z", "Y").to_euler()
        d.lens, d.shift_x, d.shift_y = lens, 0.0, 0.0

    at_screen(.135, .14, math.radians(19), math.radians(5), .26, 50.0)
    seen("s4 screen")
    for u in (.1, .3, .5, .7, .9):
        for v in (.2, .5, .8):
            at_screen(u, v, math.radians(14), math.radians(9), .06 * 85 / 20.25, 85.0)
            seen(f"二 terminal {u},{v}")
    close = R.SHOTS["close"]
    at_screen(*close["at"], 0.0, 0.0, close["width"] * R.SCREEN_W * close["lens"] / 36, close["lens"])
    seen("三 close")
    e = math.radians(10.0)
    for dist in (2150.0, 1150.0, 620.0):
        cp = Vector((0.0, -dist * math.cos(e), 50.0 + dist * math.sin(e)))
        fwd, right, upv = Vector((0, math.cos(e), -math.sin(e))), Vector((1, 0, 0)), Vector((0, math.sin(e), math.cos(e)))
        hits = []
        for q in pts:
            p = Vector((-1000 * q.x + 30, 400 + (-.045 - q.y) * 1000, (q.z - ZT) * 1000))
            v = p - cp
            z = v.dot(fwd)
            if z <= 0:
                continue
            u, w = v.dot(right) / z * 40 / 36, v.dot(upv) / z * 40 / 36
            if abs(u) > .5 or abs(w) > .5 * 9 / 16:
                continue
            k = (400 - cp.y) / v.y
            if 0 < k < 1 and cp.z + v.z * k < 0:          # under the line from the lens over its table's front edge
                continue
            hits.append(((u + .5) * 1920, 540 - w * 1920))
        print(f"chaircheck {'十三 crab ' + str(int(dist)):24s} direct {len(hits):5d}/{len(pts)} {box_of(hits)}", flush=True)


# ---------------------------------------------------------------- rendering

def save_exr(path_exr, out):
    im = bpy.data.images.load(path_exr)
    w, h = im.size
    a = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(a)
    np.save(out, a.reshape(h, w, 4)[::-1][..., :3].astype(np.float16))
    bpy.data.images.remove(im)
    os.remove(path_exr)


def render(name):
    sc = bpy.context.scene
    t0 = time.time()
    exr = os.path.join(PLATES, name + ".exr")
    sc.render.filepath = exr
    bpy.ops.render.render(write_still=True)
    save_exr(exr, os.path.join(PLATES, name + ".npy"))
    print(name, f"{time.time() - t0:.0f} s", flush=True)


def corners_json(sc, cam, corners, name):
    from bpy_extras.object_utils import world_to_camera_view
    W, H = sc.render.resolution_x * sc.render.resolution_percentage // 100, sc.render.resolution_y * sc.render.resolution_percentage // 100
    frame = []
    for p in corners:
        v = world_to_camera_view(sc, cam, Vector(p))
        frame.append((v.x * W, (1 - v.y) * H))
    with open(os.path.join(PLATES, name + ".json"), "w") as f:
        json.dump(dict(corners=frame, width=W, height=H, loc=list(cam.location), lens=cam.data.lens, shift=(cam.data.shift_x, cam.data.shift_y), focus=cam.data.dof.focus_distance, fstop=cam.data.dof.aperture_fstop), f, indent=1)


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
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.blur_glossy = 1.0
    sc.cycles.sample_clamp_indirect = 8.0
    sc.cycles.adaptive_threshold = .01

    R.room()
    R.table()
    before = set(bpy.data.objects.keys())
    screen, corners, lid = R.display()
    display_parts = [o for o in bpy.data.objects if o.name not in before]
    before = set(bpy.data.objects.keys())
    R.things()
    things = [o for o in bpy.data.objects if o.name not in before]
    view = VIEWS[opt("view", "A")]
    if SHOT == "reverse2":
        for what, d in view.get("moved", {}).items():
            moved(things, PARTS[what], *d)
    back_hall = back_wall(view.get("door", DOOR_OPEN) if SHOT == "reverse2" else DOOR_OPEN)
    if SHOT == "reverse2" and "hall" in view:
        for h in back_hall:
            if h.name not in view["hall"]:
                h.data.energy, h.hide_render = 0.0, True
        back_hall = [h for h in back_hall if h.name in view["hall"]]
    if SHOT == "reverse2":
        for name in view.get("away", ()):
            bpy.data.objects[name].hide_render = True
    bulbs, bulb_glass = floor_lamp()
    out_strength = night_outside()
    glow_strength = night_screen(screen, "glow_new.png" if SHOT == "ten" else "glow_end.png")
    bg = sc.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (.020, .015, .011, 1)       # the night sky's warmth, all that comes in at the window besides the block across the road
    chair_parts = []
    if SHOT in ("reverse2", "chaircheck"):
        x, y, yaw = (float(c) for c in opt("chair", ",".join(str(c) for c in view["chair"])).split(","))
        chair_parts = chair(x, y, yaw, view.get("chair_tint", CHAIR_TINT) if SHOT == "reverse2" else CHAIR_TINT)
    if SHOT == "reverse2":
        dark_hall(HALL_DARK)
        if "dark" in view:
            dark_room(view["dark"])
    R.neutral_in_metal()

    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_width, cam_data.clip_start, cam_data.clip_end = 36.0, .01, 60
    cam = link(bpy.data.objects.new("cam", cam_data))
    sc.camera = cam

    def lights(lamp=0.0, glow=0.0, city=0.0, hall=0.0):
        for b in bulbs:
            b.data.energy = 55.0 * lamp
            b.hide_render = lamp == 0
        for g in bulb_glass:
            g.hide_render = False
            g.data.materials[0].node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 6.0 * lamp
        glow_strength.default_value = glow
        out_strength.default_value = city
        bg.inputs[1].default_value = city
        for h in back_hall:
            h.data.energy = h["base"] * hall
            h.hide_render = hall == 0

    def one(prefix):
        if opt("beauty", False):
            lights(**BEAUTY)
            render(prefix)
            return
        for p in PASSES:
            lights(**{p: 1.0})
            render(f"{prefix}_{p}")

    if SHOT == "chaircheck":
        check_chair(sc, cam, lid, chair_parts)
    elif SHOT == "reverse2":
        for o in display_parts:                              # the lens is behind the glass: the display gives its light but is not seen
            o.visible_camera = False
        reverse2_camera(cam, view)
        bpy.context.view_layer.update()
        name = opt("name", "rev2" + opt("view", "A"))
        corners_json(sc, cam, corners, name)
        one(name)
    elif SHOT == "reverse":
        reverse_camera(cam)
        bpy.context.view_layer.update()
        corners_json(sc, cam, corners, opt("name", "reverse"))
        one(opt("name", "reverse"))
    elif FRAMES:
        a, b = (int(v) for v in FRAMES.split("-"))
        n = opt("count", 366)
        sc.render.use_persistent_data = True
        for i in range(a, b + 1):
            push_camera(cam, lid, i / (n - 1))
            bpy.context.view_layer.update()
            pre = opt("prefix", "ten")                   # --prefix=ten_1080: the same frames under another name, at another size
            corners_json(sc, cam, corners, f"{pre}_f{i:04d}")
            one(f"{pre}_f{i:04d}")
    else:
        push_camera(cam, lid, AT)
        bpy.context.view_layer.update()
        name = f"ten_{AT:.2f}"
        corners_json(sc, cam, corners, name)
        one(name)
    print("done", flush=True)


main()
