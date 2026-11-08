"""Photographs the three papers of film_docs.py where they lie on the old table, each under the same low soft light from the left of the frame and a little behind it, so that a fold shows.

    Blender -b --factory-startup -P film_docs3d.py -- --doc letter --shot wide --pct 50 --samples 64

doc and shot: letter wide, letter stop, poem stop, ruling stop, and DOC end with --state before|stop, from film_docs_end.py's maps, or with --seq, the shot itself: the new ink reached as DOC_when.png says, by the timing in SEQ, the camera going very slowly. --shot two: 二's montage, any of tablet stone greek bone slips song quarto (film_docs.py and film_relics.py make them), the first sign at TWO_AT; with --seq its second of frames into seq/mN_NAME/. A stop also writes NAME.json with each key character's box in the picture's pixels at full size, for the match cut. Goes to ink/film_out/film/imprint/, NAME_pPCT.png when not at full size, seq/NAME/ for frames.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

from ink3d import ROOT, mesh_grid, reset

OUT = os.path.join(ROOT, "ink", "film_out", "film", "imprint")
MAPS = os.path.join(OUT, "maps")
FIBRE = os.path.join(ROOT, "ink", "film_out", "film", "press", "maps", "fibre_tile.png")
FIBRE_MM = 4096 / 60.0
THICK = dict(letter=.00011, poem=.00014, ruling=.00012, quarto=.00013)     # things whose maps say solid have their own sides in their mesh and are not thickened
FIBRE_BUMP = dict(letter=.00002, poem=.00004, ruling=.000012, quarto=.000015)
RELIEF_BUMP = dict(letter=0.0, poem=0.0, ruling=.00007, quarto=.00005, tablet=.0001, stone=.00001, greek=.000006, bone=.00004, slips=.00003, song=.00003)
YAW = dict(letter=-2.5, poem=1.5, ruling=2.5, quarto=1.8, tablet=-3.0, stone=4.0, greek=-2.0, bone=6.0, slips=2.0, song=-3.5)     # how the sheet lies, in degrees
POWER = dict(letter=62.0, poem=62.0, ruling=54.0, quarto=60.0, tablet=46.0, stone=72.0, greek=58.0, bone=40.0, slips=62.0, song=56.0)   # the ruling's near panel leans into the light
SHEEN = dict(letter=.34, poem=.34, ruling=.14, quarto=.3, tablet=.25, stone=.4, greek=.3, bone=.45, slips=.4, song=.3)      # machine paper seen low against the light would go to a veil
NAMES = dict(letter=dict(wide="d1_letter_wide", stop="d1_letter", end="d1_end"), poem=dict(stop="d2_poem", end="d2_end"), ruling=dict(stop="d3_ruling", end="d3_end"),
             tablet=dict(stop="d0_tablet", two="m1_tablet"), stone=dict(two="m2_stone"), greek=dict(two="m3_greek"), bone=dict(two="m4_bone"), slips=dict(two="m5_slips"),
             song=dict(two="m6_song"), quarto=dict(stop="d4_quarto", two="m7_quarto"))
# 二's montage: every thing under the same light, the same camera, its first sign where every other's was. TWO: how many mm of it the frame is tall, at that sign
TWO = dict(tablet=70.0, stone=110.0, greek=108.0, bone=120.0, slips=78.0, song=175.0, quarto=125.0)     # the papyrus, the leaf and the quarto wide enough that the foot of the frame is past their writing (were 60, 125, 80)
TWO_TILT, TWO_LENS, TWO_AT = 28.0, 85.0, (0.44, 0.30)     # down from the vertical, from the foot of the thing; where the first sign sits in the frame, from the top left
TWO_PUSH, TWO_SECS = .04, 1.0                              # how much nearer the camera is by the end of its second
# since 2026-11-08 each thing is seen whole, its middle at TWO_MID, all of them alike in size: its box TWO_AREA of the frame, but no more than TWO_FILL of its width and height,
# the subtitle's band free under it. The eye finds each where the last one was, and a stranger names it in a second. Until then the first sign was held at TWO_AT and the things
# came at any size and place (--first draws that again)
TWO_AREA, TWO_FILL, TWO_MID = .33, (.80, .80), (.5, .45)
# 三's tablet under 「汇于一处」: seen from its foot at the angle ①'s low shot sees the die's face, coming in evenly on its first line until the line is where that shot has it
D0_TILT, D0_FROM, D0_TO, D0_SECS, D0_AT = 33.0, 110.0, 46.9, 3.2, (.5015, .352)
D0_NET = (13.0, 4.0, .6)        # the push's net (net()): its edge in mm under the line's middle, its height over the clay, how much of the light it takes
TWO_NET = dict(tablet=(.70, 5.0, .6), greek=(.70, 5.0, .6), bone=(.70, 5.0, .6), quarto=(.70, 5.0, .6))     # 二's: where its edge comes in the height at the carrier's first frame, its height, how much it takes
# ④'s quarto: the couplet from the foot of the page, its last line where the die's will come down
D4_TILT, D4_FIELD, D4_AT = 30.0, 70.0, (.5, .56)
# the papers' stops leave the foot of the frame to the subtitle (film.director.md 3.6): the page's foot is above 0.85 of the height, the table below it (--probe says where)
# the letter's stop: the whole page down to its foot, its column 念之在心若何可言 high in the frame where ②'s low shot has the die's (film_imprint_clay BLOWS[1]["low"]).
# It was 135.9 mm and -.0208, the column filling the height, before the subtitle had its band
LETTER_FIELD, LETTER_SHIFT = 285.0, -.16
POEM_FIELD, POEM_SHIFT = 360.0, -.06        # was 272 mm and -.012
RULING_K, RULING_SHIFT = 1.2, -.08          # the camera's distance as a factor of the old, which was 1.0 and 0.0
WET_FILM, WET_SHEEN = .00018, 1.6     # how high ink not yet dry stands, and how much it gives back
FPS = 30
SOFT = .012                           # how long a point takes to come, of the whole writing
# each shot: how long it is, and how far along the writing it is, by when: it goes and it waits
SEQ = dict(
    letter=dict(secs=1.6, go=((.30, 0), (.62, .485), (.82, .485), (1.02, .718), (1.16, .718), (1.42, 1.0))),
    poem=dict(secs=1.2, go=((.12, 0), (.36, .323), (.46, .323), (.62, .49), (.68, .49), (.8, .647), (.88, .647), (1.06, 1.0))),
    ruling=dict(secs=1.0, go=((.10, 0), (.34, .485), (.44, .485), (.62, .72), (.70, .72), (.86, 1.0))),
)
# how far the camera goes in a shot, as parts of how far off it is: across, up, in
DRIFT = dict(letter=(.002, .001, -.008), poem=(-.002, .001, -.007), ruling=(.002, -.001, -.008))


def image(path, colorspace="sRGB"):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = colorspace
    return im


def node(nt, kind, **values):
    n = nt.nodes.new(kind)
    for k, v in values.items():
        if hasattr(n, k):
            setattr(n, k, v)
        else:
            n.inputs[k].default_value = v
    return n


def picture(nt, path, vector, colorspace="sRGB", extension="REPEAT"):
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = image(path, colorspace)
    t.extension = extension
    nt.links.new(vector, t.inputs["Vector"])
    return t


def sheet_material(doc, w, h, state=""):
    m = bpy.data.materials.new(doc)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Specular IOR Level"].default_value = SHEEN[doc]
    uv = nt.nodes.new("ShaderNodeTexCoord").outputs["UV"]
    named = lambda kind, st=state: os.path.join(MAPS, "end", f"{doc}_{st}_{kind}.png") if st else os.path.join(MAPS, f"{doc}_{kind}.png")
    if state == "seq":
        # how far the writing has got, and so whether each point shows as it was or as it will be
        far = nt.nodes.new("ShaderNodeValue")
        far.name = "far"
        when = picture(nt, os.path.join(MAPS, "end", f"{doc}_when.png"), uv, "Non-Color", "EXTEND")
        gap = node(nt, "ShaderNodeMath", operation="SUBTRACT")
        nt.links.new(far.outputs[0], gap.inputs[0])
        nt.links.new(when.outputs[0], gap.inputs[1])
        come = node(nt, "ShaderNodeMath", operation="DIVIDE", use_clamp=True)
        nt.links.new(gap.outputs[0], come.inputs[0])
        come.inputs[1].default_value = SOFT

    def shown(kind, colorspace="Non-Color"):
        if state != "seq":
            return picture(nt, named(kind), uv, colorspace, "EXTEND").outputs[0]
        mix = nt.nodes.new("ShaderNodeMixRGB")
        nt.links.new(come.outputs[0], mix.inputs[0])
        nt.links.new(picture(nt, named(kind, "before"), uv, colorspace, "EXTEND").outputs[0], mix.inputs[1])
        nt.links.new(picture(nt, named(kind, "stop"), uv, colorspace, "EXTEND").outputs[0], mix.inputs[2])
        return mix.outputs[0]

    nt.links.new(shown("col", "sRGB"), b.inputs["Base Color"])
    nt.links.new(shown("rough"), b.inputs["Roughness"])
    # its fibres, standing up a very little, and for the ruling what the type pushed down
    map_ = node(nt, "ShaderNodeMapping")
    map_.inputs["Scale"].default_value = (w / FIBRE_MM, h / FIBRE_MM, 1)
    nt.links.new(uv, map_.inputs["Vector"])
    tile = picture(nt, FIBRE, map_.outputs[0], "Non-Color")
    parts = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(tile.outputs[0], parts.inputs[0])
    up = node(nt, "ShaderNodeMath", operation="ADD")
    weight = node(nt, "ShaderNodeMath", operation="MULTIPLY")
    weight.inputs[1].default_value = .35
    nt.links.new(parts.outputs[1], weight.inputs[0])
    nt.links.new(parts.outputs[0], up.inputs[0])
    nt.links.new(weight.outputs[0], up.inputs[1])
    fibre = node(nt, "ShaderNodeBump")
    fibre.inputs["Strength"].default_value = 1.0
    fibre.inputs["Distance"].default_value = FIBRE_BUMP.get(doc, 0.0)      # clay, stone, bone and bamboo have no paper fibres: their grain is in the relief map
    nt.links.new(up.outputs[0], fibre.inputs["Height"])
    normal = fibre.outputs[0]
    if RELIEF_BUMP.get(doc):
        sunk = node(nt, "ShaderNodeBump", invert=True)
        sunk.inputs["Strength"].default_value = 1.0
        sunk.inputs["Distance"].default_value = RELIEF_BUMP[doc]
        nt.links.new(shown("relief"), sunk.inputs["Height"])
        nt.links.new(normal, sunk.inputs["Normal"])
        normal = sunk.outputs[0]
    if state and os.path.exists(named("wet", "before" if state == "seq" else state)):
        # ink not yet dry: a film standing on the paper, a little of the light on it
        film = shown("wet")
        stand = node(nt, "ShaderNodeBump")
        stand.inputs["Strength"].default_value = 1.0
        stand.inputs["Distance"].default_value = WET_FILM
        nt.links.new(film, stand.inputs["Height"])
        nt.links.new(normal, stand.inputs["Normal"])
        normal = stand.outputs[0]
        shine = node(nt, "ShaderNodeMapRange")
        shine.inputs["To Min"].default_value = SHEEN[doc]
        shine.inputs["To Max"].default_value = WET_SHEEN
        nt.links.new(film, shine.inputs["Value"])
        nt.links.new(shine.outputs[0], b.inputs["Specular IOR Level"])
    nt.links.new(normal, b.inputs["Normal"])
    return m


def table():
    """A wide plane, the desk's picture mirrored beyond its own edges and the sheet in the middle of one copy, so that no seam is near."""
    W, H = 1.74, .74
    x, y = np.meshgrid(np.array((-W * 2, W * 2)), np.array((-H * 2, H * 2)))
    o = mesh_grid("table", x, y, np.zeros_like(x, np.float32))
    m = bpy.data.materials.new("table")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Specular IOR Level"].default_value = .22
    map_ = node(nt, "ShaderNodeMapping")
    map_.inputs["Scale"].default_value = (4, 4, 1)
    map_.inputs["Location"].default_value = (-1.5, -1.5, 0)
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["UV"], map_.inputs["Vector"])
    v = map_.outputs[0]
    nt.links.new(picture(nt, os.path.join(MAPS, "table_col.png"), v, extension="MIRROR").outputs[0], b.inputs["Base Color"])
    nt.links.new(picture(nt, os.path.join(MAPS, "table_rough.png"), v, "Non-Color", "MIRROR").outputs[0], b.inputs["Roughness"])
    bump = node(nt, "ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = .0005
    nt.links.new(picture(nt, os.path.join(MAPS, "table_relief.png"), v, "Non-Color", "MIRROR").outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    o.data.materials.append(m)
    return o


def lights(power, ambient):
    """One source: a big soft window low on the left of the frame, a little behind it. The room gives back a little of it."""
    az, el = math.radians(24), math.radians(17)
    d = Vector((-math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    l = bpy.data.lights.new("side", "AREA")
    l.shape, l.size, l.size_y = "RECTANGLE", 1.3, .7
    l.energy, l.color = power, (1.0, .975, .945)
    o = bpy.data.objects.new("side", l)
    bpy.context.collection.objects.link(o)
    o.location = d * 1.3
    o.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (.9, .84, .78, 1)
    bg.inputs[1].default_value = ambient


def shoot(loc, target, up, lens, stop=None, focus=None, shift_y=0.0):
    c = bpy.data.cameras.new("cam")
    c.lens, c.sensor_width, c.clip_start, c.clip_end, c.shift_y = lens, 36.0, .01, 20.0, shift_y
    o = bpy.data.objects.new("cam", c)
    bpy.context.collection.objects.link(o)
    bpy.context.scene.camera = o
    z = (loc - target).normalized()
    x = up.cross(z).normalized()
    y = z.cross(x)
    o.matrix_world = Matrix((x, y, z)).transposed().to_4x4()
    o.location = loc
    if stop:
        c.dof.use_dof = True
        c.dof.aperture_fstop = stop
        c.dof.aperture_blades = 7
        c.dof.focus_distance = (loc - (focus or target)).dot(z)
    return o


class Sheet:
    """A page on the table. Points are given in mm from its top left corner as the maps have them."""

    def __init__(self, doc, state=""):
        with open(os.path.join(MAPS, doc + ".json"), encoding="utf-8") as f:
            self.info = json.load(f)
        self.w, self.h = self.info["w"], self.info["h"]
        a = np.load(os.path.join(MAPS, doc + ".npz"))
        self.Z = a["Z"]
        self.o = mesh_grid(doc, (a["X"] - self.w / 2) / 1000, (a["Y"] - self.h / 2) / 1000, a["Z"] / 1000)
        self.o.rotation_euler = (0, 0, math.radians(YAW[doc]))
        if not self.info.get("solid"):
            s = self.o.modifiers.new("thick", "SOLIDIFY")
            s.thickness, s.offset, s.use_rim = THICK[doc], -1, True
        self.o.data.materials.append(sheet_material(doc, self.w, self.h, state))
        bpy.context.view_layer.update()
        a = math.radians(YAW[doc])
        self.right, self.up = Vector((math.cos(a), math.sin(a), 0)), Vector((-math.sin(a), math.cos(a), 0))

    def at(self, x, y):
        """Where a point of the page is in the world."""
        n, m = self.Z.shape
        i, j = int(np.clip((self.h - y) / self.h * (n - 1), 0, n - 1)), int(np.clip(x / self.w * (m - 1), 0, m - 1))
        return self.right * ((x - self.w / 2) / 1000) + self.up * ((self.h / 2 - y) / 1000) + Vector((0, 0, float(self.Z[i, j]) / 1000))

    def centre(self, boxes):
        """The middle of some characters' boxes on the page."""
        b = np.array(boxes)
        return (b[:, 0].min() + b[:, 2].max()) / 2, (b[:, 1].min() + b[:, 3].max()) / 2

    def pixels(self, cam, box, res):
        x0, y0, x1, y1 = box
        out = []
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            p = world_to_camera_view(bpy.context.scene, cam, self.at(x, y))
            out.append((p.x * res[0], (1 - p.y) * res[1]))
        out = np.array(out)
        return [round(float(v), 1) for v in (out[:, 0].min(), out[:, 1].min(), out[:, 0].max(), out[:, 1].max())]


def tilted(sheet, target, field, lens, side, up_tilt=0.0):
    """A camera looking at the target from the page's right by side degrees, the page's up still up and square to the lens along it, so that a column stays the same size from head to foot. field: how many mm of the column the picture is tall. up_tilt: from the foot of the page as well."""
    d = field * lens / 20.25 / 1000
    s, t = math.radians(side), math.radians(up_tilt)
    v = sheet.right * math.sin(s) * math.cos(t) - sheet.up * math.sin(t) + Vector((0, 0, 1)) * math.cos(s) * math.cos(t)
    return target + v * d


def band_marks(sheet, cam):
    """Where the sheet's foot and its lowest writing come in a stop, as fractions of the height: what the subtitle's band (from BAND down) has in it."""
    W, H = 1920, 1080
    foot = []
    for i in range(41):
        p = world_to_camera_view(bpy.context.scene, cam, sheet.at(sheet.w * i / 40, sheet.h - .01))
        if 0 <= p.x <= 1:
            foot.append(round(1 - p.y, 4))
    low = 0.0
    for c in sheet.info.get("chars", []):
        if c.get("box"):
            x0, y0, x1, y1 = c["box"]
            q = world_to_camera_view(bpy.context.scene, cam, sheet.at((x0 + x1) / 2, y1))
            if .3 <= q.x <= .7:
                low = max(low, 1 - q.y) if 1 - q.y <= 1.2 else low
    down = [[v, round(1 - world_to_camera_view(bpy.context.scene, cam, sheet.at(sheet.w / 2, v)).y, 4)] for v in range(0, int(sheet.h) + 1, 10)]     # down the middle of the sheet: mm from its head, where in the height
    return dict(foot=[min(foot), max(foot)] if foot else None, writing=round(low, 4), down=down)


def two_camera(sheet, field, push=0.0):
    """二's camera: down on the thing from its foot, the first sign held at TWO_AT whatever the thing, field mm of it tall there. push: how far it has come in, along the line to that sign, so the sign stays put."""
    x0, y0, x1, y1 = sheet.info["keys"]["first"]
    P = sheet.at((x0 + x1) / 2, (y0 + y1) / 2)
    t = math.radians(TWO_TILT)
    v = (-sheet.up * math.sin(t) + Vector((0, 0, 1)) * math.cos(t)).normalized()
    d = field * TWO_LENS / 20.25 / 1000 * (1 - push)
    cam = shoot(P + v * d, P, sheet.up, TWO_LENS, stop=float(np.clip(2400.0 / field, 5.6, 22.0)), focus=P)
    cam.data.shift_x, cam.data.shift_y = .5 - TWO_AT[0], (TWO_AT[1] - .5) * 9 / 16
    return cam


def two_whole(sheet):
    """二's camera since 2026-11-08: down on the thing from its foot at TWO_TILT, the whole of it, its box TWO_AREA of the frame (within TWO_FILL), its middle at TWO_MID.
    The box is the thing's own, where its relief stands above the table, not its map's; measured off the camera, not worked out."""
    n, m = sheet.Z.shape
    rows, cols = np.nonzero(sheet.Z > .01)
    bx0, bx1 = cols.min() / (m - 1) * sheet.w, cols.max() / (m - 1) * sheet.w
    by0, by1 = sheet.h - rows.max() / (n - 1) * sheet.h, sheet.h - rows.min() / (n - 1) * sheet.h          # the map's rows run up the page (Sheet.at)
    P = sheet.at((bx0 + bx1) / 2, (by0 + by1) / 2)
    t = math.radians(TWO_TILT)
    v = (-sheet.up * math.sin(t) + Vector((0, 0, 1)) * math.cos(t)).normalized()
    rim = [(x, y) for x in np.linspace(bx0, bx1, 17) for y in (by0, by1)] + [(x, y) for y in np.linspace(by0, by1, 17) for x in (bx0, bx1)]
    lift = [Vector((0, 0, 0)), Vector((0, 0, float(sheet.Z.max()) / 1000))]        # the rim on the table and as high as the thing stands: a slab's far top edge shows above its foot
    field = 2.0 * max(sheet.w, sheet.h)
    for i in range(5):
        cam = shoot(P + v * (field * TWO_LENS / 20.25 / 1000), P, sheet.up, TWO_LENS, stop=float(np.clip(2400.0 / field, 5.6, 22.0)), focus=P)
        bpy.context.view_layer.update()
        q = np.array([tuple(world_to_camera_view(bpy.context.scene, cam, sheet.at(x, y) * Vector((1, 1, 0)) + z))[:2] for x, y in rim for z in lift])
        (x0, y0), (x1, y1) = q.min(0), q.max(0)
        k = max(math.sqrt((x1 - x0) * (y1 - y0) / TWO_AREA), (x1 - x0) / TWO_FILL[0], (y1 - y0) / TWO_FILL[1])
        if i == 4 or abs(k - 1) < 1e-4:
            break
        bpy.data.objects.remove(cam)
        field *= k
    # a lens shift moves the picture on the film and nothing else: the middle goes to TWO_MID (shift is in widths; the view's y goes up)
    cam.data.shift_x, cam.data.shift_y = (x0 + x1) / 2 - TWO_MID[0], ((y0 + y1) / 2 - (1 - TWO_MID[1])) * 9 / 16
    print("two_whole", round(field, 1), "mm tall, fill", round(float(x1 - x0), 3), round(float(y1 - y0), 3), flush=True)
    return cam


def from_foot(sheet, P, field, tilt, at, lens=85.0, stop=None):
    """Down on P from the foot of the page by tilt degrees, field mm tall there, P held at the frame's point at, the page's lines level."""
    t = math.radians(tilt)
    v = (-sheet.up * math.sin(t) + Vector((0, 0, 1)) * math.cos(t)).normalized()
    cam = shoot(P + v * (field * lens / 20.25 / 1000), P, sheet.up, lens, stop=stop or float(np.clip(2400.0 / field, 5.6, 22.0)), focus=P)
    cam.data.shift_x, cam.data.shift_y = .5 - at[0], (at[1] - .5) * 9 / 16
    return cam


def d0_camera(sheet, field):
    return from_foot(sheet, sheet.at(*sheet.centre(sheet.info["keys"]["run"])), field, D0_TILT, D0_AT)


def net(sheet, edge, high, density, to=None):
    """A net between the window and the near part of the thing, as a gaffer sets one: from edge mm under the sheet's head to well past its foot, high mm over it, taking density of
    the light. The camera doesn't see it; its soft shade does the rest: the foot of the frame goes quietly darker, and the subtitle over it reads, light on the shade (film.director.md 3.6).
    to: how far down the sheet it has to clear (the foot of the frame), if not to the sheet's foot: a page curling up out of the picture may go through it."""
    n = sheet.Z.shape[0]
    row = lambda y: int(np.clip((sheet.h - y) / sheet.h * (n - 1), 0, n - 1))
    z = (float(sheet.Z[row(to) if to is not None else 0:row(edge) + 1].max()) + high) / 1000         # over the highest of what it covers
    pts = [sheet.right * ((u - sheet.w / 2) / 1000) + sheet.up * ((sheet.h / 2 - v) / 1000) for u, v in ((-400, edge), (sheet.w + 400, edge), (sheet.w + 400, sheet.h + 600), (-400, sheet.h + 600))]
    me = bpy.data.meshes.new("net")
    me.from_pydata([(p.x, p.y, z) for p in pts], [], [(0, 1, 2, 3)])
    mat = bpy.data.materials.new("net")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out, mix, hole, cloth = nt.nodes.new("ShaderNodeOutputMaterial"), nt.nodes.new("ShaderNodeMixShader"), nt.nodes.new("ShaderNodeBsdfTransparent"), nt.nodes.new("ShaderNodeBsdfDiffuse")
    cloth.inputs["Color"].default_value = (.01, .01, .01, 1)
    mix.inputs["Fac"].default_value = density
    nt.links.new(hole.outputs[0], mix.inputs[1])
    nt.links.new(cloth.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    me.materials.append(mat)
    o = bpy.data.objects.new("net", me)
    bpy.context.collection.objects.link(o)
    o.visible_camera = o.visible_glossy = o.visible_transmission = False


def d0(sc, sheet, a):
    """The tablet's push, a frame at a time: the size of the line grows by the same factor every frame, about the line's middle, which stays put. The last frame is d0_tablet.png's framing."""
    sc.render.use_persistent_data = True
    sc.cycles.seed, sc.cycles.use_animated_seed = 0, False
    sc.render.image_settings.compression = 40
    below, high, density = json.loads(a.net) if a.net else D0_NET
    if density:
        net(sheet, sheet.centre(sheet.info["keys"]["run"])[1] + below, high, density)
    folder = os.path.join(OUT, "seq", NAMES["tablet"]["stop"] + a.tag)
    os.makedirs(folder, exist_ok=True)
    count = round(D0_SECS * FPS)
    only = [int(v) for v in a.frames.split(",")] if a.frames else range(count)
    cam = None
    for i in only:
        field = D0_FROM * (D0_TO / D0_FROM) ** (i / (count - 1))
        new = d0_camera(sheet, field)
        if cam is None:
            cam = new
        else:
            cam.location, cam.data.dof.focus_distance, cam.data.dof.aperture_fstop = new.location, new.data.dof.focus_distance, new.data.dof.aperture_fstop
            bpy.data.objects.remove(new)
        bpy.context.scene.camera = cam
        sc.render.filepath = os.path.join(folder, "%04d.png" % i)
        bpy.ops.render.render(write_still=True)
        print("wrote", sc.render.filepath, flush=True)


def two_net(sheet, cam, spec):
    """二's net, its edge where the carrier's first frame has frac of its height."""
    if not spec or not spec[2]:
        return
    frac, high, density = spec

    def down(f):
        lo, hi = -100.0, sheet.h + 300.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if 1 - world_to_camera_view(bpy.context.scene, cam, sheet.at(sheet.w / 2, mid)).y < f:
                lo = mid
            else:
                hi = mid
        return lo
    net(sheet, down(frac), high, density, to=down(1.03))


def two(sc, sheet, a):
    """The thing's second of 二: still, the camera coming in evenly. Frames to seq/NAME/."""
    sc.render.use_persistent_data = True
    sc.cycles.seed, sc.cycles.use_animated_seed = 0, False
    sc.render.image_settings.compression = 40
    name = NAMES[a.doc]["two"]
    folder = os.path.join(OUT, "seq", name)
    os.makedirs(folder, exist_ok=True)
    count = round(TWO_SECS * FPS)
    only = [int(v) for v in a.frames.split(",")] if a.frames else range(count)
    field = a.field or TWO[a.doc]
    first = two_camera(sheet, field)
    bpy.context.view_layer.update()
    two_net(sheet, first, json.loads(a.net) if a.net else TWO_NET.get(a.doc))
    bpy.data.objects.remove(first)
    cam = None
    for i in only:
        push = TWO_PUSH * (i + .5) / count
        if cam is None:
            cam = two_camera(sheet, field, push)
        else:
            new = two_camera(sheet, field, push)
            cam.location, cam.data.dof.focus_distance = new.location, new.data.dof.focus_distance
            bpy.data.objects.remove(new)
        bpy.context.scene.camera = cam
        sc.render.filepath = os.path.join(folder, "%04d.png" % i)
        bpy.ops.render.render(write_still=True)
        print("wrote", sc.render.filepath, flush=True)


def film(sc, sheet, cam, a):
    """The shot, a frame at a time, in the one scene: only how far the writing has got and where the camera is change."""
    sc.render.use_persistent_data = True
    sc.cycles.seed, sc.cycles.use_animated_seed = 0, False
    sc.render.image_settings.compression = 40
    tree = sheet.o.data.materials[0].node_tree
    far = tree.nodes["far"].outputs[0]
    secs, go = SEQ[a.doc]["secs"], SEQ[a.doc]["go"]
    count = round(secs * FPS)
    x, y, z = [cam.matrix_world.to_3x3().col[i].copy() for i in range(3)]
    at, depth = cam.location.copy(), cam.data.dof.focus_distance
    away = (at - sheet.at(sheet.w / 2, sheet.h / 2)).length
    dx, dy, dz = [v * away for v in DRIFT[a.doc]]
    folder = os.path.join(OUT, "seq", NAMES[a.doc]["end"])
    os.makedirs(folder, exist_ok=True)
    only = [int(v) for v in a.frames.split(",")] if a.frames else range(count)
    for i in only:
        t = i / FPS
        now = 0.0
        for (t0, v0), (t1, v1) in zip(go, go[1:]):
            if t >= t0:
                u = min((t - t0) / (t1 - t0), 1.0)
                now = v0 + (v1 - v0) * u * u * (3 - 2 * u)
        far.default_value = now
        u = (i + .5) / count - .5
        cam.location = at + x * dx * u + y * dy * u + z * dz * u
        cam.data.dof.focus_distance = depth + dz * u
        sc.render.filepath = os.path.join(folder, "%04d.png" % i)
        bpy.ops.render.render(write_still=True)
        print("wrote", sc.render.filepath, "%.3f" % now, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--doc", required=True)
    p.add_argument("--shot", default="stop")
    p.add_argument("--pct", type=int, default=100)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--power", type=float, default=0.0)
    p.add_argument("--ambient", type=float, default=.07)
    p.add_argument("--tag", default="")
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--state", default="")
    p.add_argument("--seq", action="store_true")
    p.add_argument("--frames", default="")     # only these of a --seq or --push
    p.add_argument("--push", action="store_true")     # the tablet's stop as its push: seq/d0_tablet/
    p.add_argument("--field", type=float, default=0.0)     # a stop's or a two shot's field (the ruling's: its distance, as a factor), to try one
    p.add_argument("--shift", type=float, default=None)    # and its shift_y
    p.add_argument("--net", default="")                   # the push's net to try, as [below, high, density]
    p.add_argument("--first", action="store_true")         # 二's shot as it was until 2026-11-08: the first sign held at TWO_AT
    p.add_argument("--probe", action="store_true")         # a stop's or two shot's marks and where the paper lies in the subtitle's band, no picture
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
    sc = reset(a.samples, a.pct)
    sc.render.film_transparent = False
    sc.render.image_settings.color_mode = "RGB"
    sc.cycles.max_bounces = 4
    sc.cycles.adaptive_threshold = .006
    sc.cycles.use_denoising = a.denoise
    sc.cycles.filter_width = 1.2      # the ink's edges are what is looked at
    table()
    sheet = Sheet(a.doc, "seq" if a.seq else a.state)
    lights(a.power or POWER[a.doc], a.ambient)
    keys = sheet.info["keys"]
    marks = {}
    place = lambda cam: bpy.context.view_layer.update()      # or a camera just made is still at its old place
    if a.shot == "two" and a.seq:
        return two(sc, sheet, a)
    if a.doc == "tablet" and a.shot == "stop" and a.push:
        return d0(sc, sheet, a)
    if a.shot == "two" and not a.first:
        cam = two_whole(sheet)          # no net: the thing ends above the subtitle's band
        place(cam)
        marks = dict(first=sheet.pixels(cam, keys["first"], (1920, 1080)), whole=sheet.pixels(cam, (0, 0, sheet.w, sheet.h), (1920, 1080)))
    elif a.shot == "two":
        cam = two_camera(sheet, a.field or TWO[a.doc])
        place(cam)
        two_net(sheet, cam, json.loads(a.net) if a.net else TWO_NET.get(a.doc))
        marks = dict(first=sheet.pixels(cam, keys["first"], (1920, 1080)))
    elif a.shot == "end":
        # each where it ends: the same side, the same glass as that sheet's stop, nearer what is left
        last = sheet.info["chars"]
        if a.doc == "letter":
            run = [c["box"] for c in last[-8:]]
            x, y = sheet.centre(run)
            target = sheet.at(x, y + 11)
            focus = sheet.at(*sheet.centre(run[-1:]))
            cam = shoot(tilted(sheet, target, 133, 85, 31), target, sheet.up, 85, stop=5.0, focus=focus, shift_y=-.018)      # what comes after 言 clear of the subtitle
        elif a.doc == "poem":
            run = [c["box"] for c in last[-5:]]
            x, y = sheet.centre(run)
            target = sheet.at(x - 8, y)
            cam = shoot(tilted(sheet, target, 288, 85, 30), target, sheet.up, 85, stop=2.4, shift_y=-.03)     # the column's foot clear of the subtitle
        else:
            run = [c["box"] for c in last[-9:]]
            x, y = sheet.centre(run)
            target = sheet.at(x - 8, y)
            k = .48
            loc = target + (sheet.right * .17 - sheet.up * .285 + Vector((0, 0, .36))) * k
            with open(os.path.join(MAPS, "end", "ruling_place.json")) as f:
                new = json.load(f)
            head = sheet.centre(run[:1])
            cam = shoot(loc, target, sheet.up + Vector((0, 0, .75)), 50, stop=9.0, focus=sheet.at((new["x"] + head[0]) / 2, (new["y"] + head[1]) / 2))      # between the column's head and what has come beside it
        place(cam)
        marks = dict(run=[sheet.pixels(cam, b, (1920, 1080)) for b in run])
        place_file = os.path.join(MAPS, "end", a.doc + "_place.json")
        if os.path.exists(place_file):
            with open(place_file) as f:
                where = json.load(f)
            if "box" in where:
                marks["new"] = sheet.pixels(cam, where["box"], (1920, 1080))
            else:
                marks["new"] = sheet.pixels(cam, [where["x"] - where["em"] / 2, where["y"] - where["em"] / 2, where["x"] + where["em"] / 2, where["y"] + where["em"] / 2], (1920, 1080))
    elif a.doc == "tablet" and a.shot == "stop":
        cam = d0_camera(sheet, D0_TO)
        place(cam)
        marks = dict(run=[sheet.pixels(cam, b, (1920, 1080)) for b in keys["run"]])
    elif a.doc == "quarto" and a.shot == "stop":
        P = sheet.at(*sheet.centre(keys["last"]))
        cam = from_foot(sheet, P, D4_FIELD, D4_TILT, D4_AT, stop=4.0)
        place(cam)
        marks = dict(last=[sheet.pixels(cam, b, (1920, 1080)) for b in keys["last"]], lines=[sheet.pixels(cam, l[0], (1920, 1080)) for l in keys["lines"]])
    elif a.doc == "letter" and a.shot == "wide":
        c = sheet.at(sheet.w / 2, sheet.h / 2)
        el = math.radians(61)
        field = 392
        loc = c + Vector((.05, -math.cos(el), math.sin(el))).normalized() * (field * 72 / 20.25 / 1000)
        cam = shoot(loc, c, sheet.up + Vector((0, 0, .5)), 72, stop=8.0)
    elif a.doc == "letter":
        x, y = sheet.centre(keys["run"])
        target = sheet.at(x, y)
        cam = shoot(tilted(sheet, target, a.field or LETTER_FIELD, 85, 31), target, sheet.up, 85, stop=3.5, shift_y=LETTER_SHIFT if a.shift is None else a.shift)
        place(cam)
        marks["run"] = [dict(c=c, box=sheet.pixels(cam, b, (1920, 1080))) for c, b in zip("念之在心若何可言", keys["run"])]
    elif a.doc == "poem":
        x, y = sheet.centre(keys["lines"][0] + keys["lines"][1])
        target = sheet.at(x, y)
        cam = shoot(tilted(sheet, target, a.field or POEM_FIELD, 85, 30), target, sheet.up, 85, stop=2.4, shift_y=POEM_SHIFT if a.shift is None else a.shift)
        place(cam)
        marks["lines"] = [[sheet.pixels(cam, b, (1920, 1080)) for b in col] for col in keys["lines"]]
    else:
        # from the near right of the form, high enough that its order is what shows: the title and the parties and the order close and sharp, the reasons running away to the date and the seal
        x, y = sheet.centre(keys["title"] + keys["order"])
        target = sheet.at(x - 62, y + 22)
        loc = target + (sheet.right * .17 - sheet.up * .285 + Vector((0, 0, .36))) * (a.field or RULING_K)
        cam = shoot(loc, target, sheet.up + Vector((0, 0, .75)), 50, stop=5.6, focus=sheet.at(*sheet.centre(keys["order"])), shift_y=RULING_SHIFT if a.shift is None else a.shift)
        place(cam)
        marks = dict(seal=sheet.pixels(cam, keys["seal"], (1920, 1080)), title=[sheet.pixels(cam, b, (1920, 1080)) for b in keys["title"]])
    if a.seq:
        return film(sc, sheet, cam, a)
    if a.shot in ("stop", "two"):
        marks["band"] = band_marks(sheet, cam)
    if a.probe:
        print("probe", a.doc, json.dumps(marks, ensure_ascii=False), flush=True)
        return
    name = NAMES[a.doc][a.shot] + ("_" + a.state if a.state else "") + a.tag + ("" if a.pct == 100 else "_p%d" % a.pct)
    sc.render.filepath = os.path.join(OUT, "test" if a.pct != 100 else "", name + ".png")
    bpy.ops.render.render(write_still=True)
    if marks:
        with open(os.path.join(OUT, NAMES[a.doc][a.shot] + ".json"), "w", encoding="utf-8") as f:
            json.dump(marks, f, ensure_ascii=False)
    print("wrote", sc.render.filepath, flush=True)


if __name__ == "__main__":
    main()
