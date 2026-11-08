"""七's last shot, photographed: the 唐詩三百首 of film_tang.py lying open on the old table, the room in shade, and a bar of sun from a gap above a sill coming across the page as the afternoon goes, until it lies on 低頭思故鄉 and stays.

    Blender -b --factory-startup -P film_tang3d.py -- --t 4.4 [--pct 50] [--samples 192] [--name tang_still]
    Blender -b --factory-startup -P film_tang3d.py -- --seq [--frames 0,40,134]
    Blender -b --factory-startup -P film_tang3d.py -- --eat [--frames 0,12,24] [--pct 50] [--samples 64]

--eat (2026-11-08) is the shot's last beat, EAT seconds of it: the line has gone, and 低頭思故鄉 comes off the page, 鄉 first, and is drawn away along
its column and out at the foot of the frame, as the crab drew the letters across the board in the bridge; 八 then cuts to him chewing. It writes
tang/eat/NNNN.png from the beat's start (tang/eat_pNN at another size). Until a piece moves, piece and page give back the page as it was (film_tang.lift).

t: seconds into the shot (183.1 in the film). --seq writes every frame of it to tang/seq/. The sun is the only thing that moves: its bearing turns, nothing grows and nothing blows. The wall with the gap is never seen, only its shadow. Writes NAME.json beside a still with the five characters' boxes in the picture.
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

TANG = os.path.join(ROOT, "ink", "film_out", "film", "learn", "tang")
MAPS = os.path.join(TANG, "maps")
TABLE = os.path.join(ROOT, "ink", "film_out", "film", "imprint", "maps")
FIBRE = os.path.join(ROOT, "ink", "film_out", "film", "press", "maps", "fibre_tile.png")
FIBRE_MM = 4096 / 60.0
FPS, SECS = 30, 4.5
YAW = -1.5                 # how the book lies, degrees
EL = 20.0                  # the sun's height
SWEEP = (-24.0, 0.0)       # its bearing from straight behind the book's head, from and to, degrees: from the left, so that its bar comes in over the right fore-edge
GO = 2.5                   # seconds for the sweep; then it holds
SLOT = 24.0                # the gap's width, mm
SILL_GAP = 1.6             # the sill's shadow stands this far above 低, mm
SUN, SKY = 7.0, .42        # strengths: the room in shade still light enough to read in, the bar of sun plainly sun
SUN_COLOUR, SKY_COLOUR = (1.0, .94, .84), (1.0, .965, .92)
EAT = (4.5, 6.0)           # the last beat, in the shot's seconds: from the line's end to the cut to 八 (ink_subs.EAT_S long)
# the line comes off as one: 鄉 first, the others after it up the column, until 低頭思故鄉 hangs over the page, plainly itself, its shadow apart from
# it in the sun; then it is drawn away like a noodle, 鄉 leading, and the empty column holds a moment before the cut
EAT0, EAT_EVERY = 4.55, .05    # when 鄉 starts to come loose, and how far apart the five do
PEEL, PEEL_UP, PEEL_TILT = .2, 2.0, 10.0        # seconds to come loose; mm it rises (higher, its shadow fell on the next character and read as a misprint); degrees its upper edge comes up by.
# Not the lower edge: tipped towards the book's head, a piece took the low sun's glint straight into the lens, and the ink went pale grey
PULL0, PULL_EVERY = 4.95, .06  # when 鄉 is drawn away, and how far apart the five go
PULL = 1600.0              # mm a second a second along the column: out of the frame in under half a second
CLIMB = .25                # mm up for each mm along: drawn up towards a mouth above the table, clear of the print it passes over
TURN = (7.0, -9.0, 5.0, -6.0, 10.0)    # degrees each piece has turned by the time it is 40 mm on, 低 to 鄉


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


def table():
    """The old table of film_docs3d.py, its picture mirrored beyond its own edges."""
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
    map_.inputs["Location"].default_value = (-1.3, -1.6, 0)
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["UV"], map_.inputs["Vector"])
    v = map_.outputs[0]
    nt.links.new(picture(nt, os.path.join(TABLE, "table_col.png"), v, extension="MIRROR").outputs[0], b.inputs["Base Color"])
    nt.links.new(picture(nt, os.path.join(TABLE, "table_rough.png"), v, "Non-Color", "MIRROR").outputs[0], b.inputs["Roughness"])
    bump = node(nt, "ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = .0005
    nt.links.new(picture(nt, os.path.join(TABLE, "table_relief.png"), v, "Non-Color", "MIRROR").outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    o.data.materials.append(m)


def page_material(w, h, eaten=False):
    m = bpy.data.materials.new("page")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Specular IOR Level"].default_value = .3
    uv = nt.nodes.new("ShaderNodeTexCoord").outputs["UV"]
    tail = "_eaten" if eaten else ""
    nt.links.new(picture(nt, os.path.join(MAPS, "spread_col%s.png" % tail), uv, extension="EXTEND").outputs[0], b.inputs["Base Color"])
    nt.links.new(picture(nt, os.path.join(MAPS, "spread_rough%s.png" % tail), uv, "Non-Color", "EXTEND").outputs[0], b.inputs["Roughness"])
    map_ = node(nt, "ShaderNodeMapping")
    map_.inputs["Scale"].default_value = (w / FIBRE_MM, h / FIBRE_MM, 1)
    nt.links.new(uv, map_.inputs["Vector"])
    tile = picture(nt, FIBRE, map_.outputs[0], "Non-Color")
    parts = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(tile.outputs[0], parts.inputs[0])
    bump = node(nt, "ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = .00006
    nt.links.new(parts.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    return m


def plain(name, rgb, rough, streaks=0.0):
    """A matt paper colour; streaks: the leaves' edges, as faint bands along the block."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = .2
    lin = [((c / 255 + .055) / 1.055) ** 2.4 for c in rgb]
    if not streaks:
        b.inputs["Base Color"].default_value = (*lin, 1)
        return m
    co = nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]
    map_ = node(nt, "ShaderNodeMapping")
    map_.inputs["Scale"].default_value = (6, 6, 2600)
    nt.links.new(co, map_.inputs["Vector"])
    tex = node(nt, "ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 1.0
    tex.inputs["Detail"].default_value = 3.0
    nt.links.new(map_.outputs[0], tex.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.inputs[1].default_value = (*[c * (1 - streaks) for c in lin], 1)
    mix.inputs[2].default_value = (*lin, 1)
    nt.links.new(tex.outputs["Fac"], mix.inputs[0])
    nt.links.new(mix.outputs[0], b.inputs["Base Color"])
    return m


class Book:
    """The open book: the leaves as film_tang.py shaped them, the block's edges under them, the soft cover under that. Points are in mm from the spread's top left."""

    def __init__(self, eaten=False):
        with open(os.path.join(MAPS, "spread.json"), encoding="utf-8") as f:
            self.info = json.load(f)
        self.w, self.h = self.info["w"], self.info["h"]
        a = np.load(os.path.join(MAPS, "spread.npz"))
        U, V, Z = a["X"], a["Y"], a["Z"]
        self.U, self.V, self.Z = U, V, Z
        self.root = bpy.data.objects.new("book", None)
        bpy.context.collection.objects.link(self.root)
        self.root.rotation_euler = (0, 0, math.radians(YAW))
        X, Y = (U - self.w / 2) / 1000, (V - self.h / 2) / 1000
        page = mesh_grid("leaves", X, Y, Z / 1000)
        page.data.materials.append(page_material(self.w, self.h, eaten))
        page.parent = self.root
        cover = self.info["cover"] / 1000
        edge = plain("edges", (214, 200, 166), .9, streaks=.16)
        for name, xs, ys, zs in (("foot", X[0], Y[0], Z[0]), ("head", X[-1], Y[-1], Z[-1]), ("fore_l", X[:, 0], Y[:, 0], Z[:, 0]), ("fore_r", X[:, -1], Y[:, -1], Z[:, -1])):
            o = mesh_grid(name, np.stack([xs, xs]), np.stack([ys, ys]), np.stack([np.full_like(zs, cover), zs / 1000]), uv=False)
            o.data.materials.append(edge)
            o.parent = self.root
        bpy.ops.mesh.primitive_cube_add(size=1)
        c = bpy.context.active_object
        c.scale = ((self.w + 6) / 1000, (self.h + 6) / 1000, cover)
        c.location = (0, 0, cover / 2)
        c.data.materials.append(plain("cover", (122, 96, 66), .82))
        c.parent = self.root
        bpy.context.view_layer.update()

    def at(self, x, y):
        """Where a point of the spread is in the world, on the leaves."""
        n, m = self.Z.shape
        i, j = int(np.clip((self.h - y) / self.h * (n - 1), 0, n - 1)), int(np.clip(x / self.w * (m - 1), 0, m - 1))
        local = Vector(((x - self.w / 2) / 1000, (self.h / 2 - y) / 1000, float(self.Z[i, j]) / 1000))
        return self.root.matrix_world @ local

    def pixels(self, cam, box, res):
        x0, y0, x1, y1 = box
        out = []
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            p = world_to_camera_view(bpy.context.scene, cam, self.at(x, y))
            out.append((p.x * res[0], (1 - p.y) * res[1]))
        out = np.array(out)
        return [round(float(v), 1) for v in (out[:, 0].min(), out[:, 1].min(), out[:, 0].max(), out[:, 1].max())]


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class Pieces:
    """The five characters as film_tang.lift cut them, each a thin film of ink lying on the leaf where it was printed: a small grid on the leaf's own
    heights, 0.01 mm above it, its picture key_N.png with the ink's share as alpha. Each turns about its own middle; parented to the book, so that
    "along the column" is the book's -y."""

    def __init__(self, book):
        with open(os.path.join(MAPS, "pieces.json"), encoding="utf-8") as f:
            self.info = json.load(f)
        U, V, Z = book.U, book.V, book.Z
        us, vs = U[0], V[:, 0]
        iu, iv = (np.argsort(us), np.argsort(vs))

        def height(x, v):
            """The leaf's height in mm at x mm across and v mm up from the foot, between the grid's points."""
            fj = np.interp(x, us[iu], iu.astype(np.float64))
            fi = np.interp(v, vs[iv], iv.astype(np.float64))
            j0, i0 = np.clip(np.floor(fj).astype(int), 0, Z.shape[1] - 2), np.clip(np.floor(fi).astype(int), 0, Z.shape[0] - 2)
            a, b = fj - j0, fi - i0
            return (Z[i0, j0] * (1 - a) * (1 - b) + Z[i0, j0 + 1] * a * (1 - b) + Z[i0 + 1, j0] * (1 - a) * b + Z[i0 + 1, j0 + 1] * a * b)

        self.objects, self.rest = [], []
        for k, p in enumerate(self.info["pieces"]):
            x0, y0, x1, y1 = p["box"]
            x, y = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(y1, y0, 9))      # rows from its foot up, as the picture's v runs
            z = height(x, book.h - y) + .01
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            cz = float(height(np.array(cx), np.array(book.h - cy))) + .01
            o = mesh_grid("piece%d" % k, (x - cx) / 1000, (cy - y) / 1000, (z - cz) / 1000)
            o.data.materials.append(self.material(k))
            o.parent = book.root
            o.location = ((cx - book.w / 2) / 1000, (book.h / 2 - cy) / 1000, cz / 1000)
            o.rotation_mode = "XYZ"
            self.objects.append(o)
            self.rest.append(Vector(o.location))

    @staticmethod
    def material(k):
        m = bpy.data.materials.new("piece%d" % k)
        m.use_nodes = True
        nt = m.node_tree
        b = nt.nodes["Principled BSDF"]
        b.inputs["Roughness"].default_value = .80          # the page's own where it is all ink (film_tang: 0.86 - 0.06)
        b.inputs["Specular IOR Level"].default_value = .3
        t = picture(nt, os.path.join(MAPS, "key_%d.png" % k), nt.nodes.new("ShaderNodeTexCoord").outputs["UV"], extension="CLIP")
        nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
        # its shadow comes as it comes loose: lying on the leaf, a piece that also shaded the paper seen through its strokes' soft edges darkened them
        # twice, and the five went bolder than the page they had to match. "shadow" (0 to 1) is keyed with the peel
        gain = nt.nodes.new("ShaderNodeValue")
        gain.name = "shadow"
        gain.outputs[0].default_value = 0.0
        shadow_ray = nt.nodes.new("ShaderNodeLightPath").outputs["Is Shadow Ray"]

        def op(kind, a, b_):
            n = nt.nodes.new("ShaderNodeMath")
            n.operation = kind
            for i, v in enumerate((a, b_)):
                if isinstance(v, float):
                    n.inputs[i].default_value = v
                else:
                    nt.links.new(v, n.inputs[i])
            return n.outputs[0]
        lost = op("MULTIPLY", shadow_ray, op("SUBTRACT", 1.0, gain.outputs[0]))
        nt.links.new(op("MULTIPLY", t.outputs["Alpha"], op("SUBTRACT", 1.0, lost)), b.inputs["Alpha"])
        return m

    @staticmethod
    def pose(k, t):
        """Piece k (0 低 .. 4 鄉) t seconds into the shot: mm along the column towards the foot, mm up, degrees its upper edge has come up, degrees turned."""
        tau = t - (EAT0 + (4 - k) * EAT_EVERY)
        if tau <= 0:
            return 0.0, 0.0, 0.0, 0.0
        peel = _smooth(tau / PEEL)
        go = max(t - (PULL0 + (4 - k) * PULL_EVERY), 0.0)
        along = .5 * PULL * go * go
        return along, PEEL_UP * peel + CLIMB * along, PEEL_TILT * peel, TURN[k] * min(along / 40.0, 1.0)

    def place(self, t, frame=None):
        for k, o in enumerate(self.objects):
            along, up, tilt, turn = self.pose(k, t)
            o.location = self.rest[k] + Vector((0.0, -along / 1000, up / 1000))
            o.rotation_euler = (math.radians(tilt), 0.0, math.radians(turn))
            shadow = o.data.materials[0].node_tree.nodes["shadow"].outputs[0]
            shadow.default_value = min(up / PEEL_UP, 1.0)
            if frame is not None:
                o.keyframe_insert("location", frame=frame)
                o.keyframe_insert("rotation_euler", frame=frame)
                shadow.keyframe_insert("default_value", frame=frame)


def towards_sun(bearing):
    """From the table to the sun: behind the book's head, turned by bearing degrees to the right of the frame."""
    a, e = math.radians(bearing + YAW), math.radians(EL)
    return Vector((math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)))


def window(book):
    """The wall the sun comes through, seen by nothing but the sun: a gap SLOT wide above a sill, placed so that when the sun has come round, the sill's shadow stands just above 低 and the gap's middle is on the column."""
    keys = book.info["keys"]
    cut = book.at(keys["x"], keys["cut"] - SILL_GAP)
    s = towards_sun(SWEEP[1])
    flat = Vector((s.x, s.y, 0)).normalized()
    # the wall stands square to the sun's last bearing, far enough behind the book's head that it never falls on the book
    far = (book.at(book.w / 2, -40.0) - cut).dot(flat)
    sill = cut + s * (far / Vector((s.x, s.y, 0)).length)
    across = Vector((flat.y, -flat.x, 0))
    pieces = (
        (-1.5, -SLOT / 2000, 0.0, 1.6),          # left of the gap, right of it, and the sill under it
        (SLOT / 2000, 1.5, 0.0, 1.6),
        (-SLOT / 2000, SLOT / 2000, 0.0, sill.z),
    )
    for k, (a0, a1, z0, z1) in enumerate(pieces):
        corners = [sill + across * a0 + Vector((0, 0, z0 - sill.z)), sill + across * a1 + Vector((0, 0, z0 - sill.z)),
                   sill + across * a1 + Vector((0, 0, z1 - sill.z)), sill + across * a0 + Vector((0, 0, z1 - sill.z))]
        me = bpy.data.meshes.new("wall%d" % k)
        me.from_pydata([tuple(c) for c in corners], [], [(0, 1, 2, 3)])
        o = bpy.data.objects.new("wall%d" % k, me)
        bpy.context.collection.objects.link(o)
        o.visible_camera = o.visible_diffuse = o.visible_glossy = o.visible_transmission = o.visible_volume_scatter = False
    print("sill %.3f m up, %.3f m behind the cut" % (sill.z, far), flush=True)


def lights():
    l = bpy.data.lights.new("sun", "SUN")
    l.energy, l.angle, l.color = SUN, math.radians(.53), SUN_COLOUR
    o = bpy.data.objects.new("sun", l)
    bpy.context.collection.objects.link(o)
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (*SKY_COLOUR, 1)
    bg.inputs[1].default_value = SKY
    return o


def aim(sun, t):
    u = min(max(t / GO, 0.0), 1.0)
    u = 1 - (1 - u) ** 3                       # it slows as it comes to its place
    bearing = SWEEP[0] + (SWEEP[1] - SWEEP[0]) * u
    d = -towards_sun(bearing)
    sun.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return bearing


def shoot(book, lens=60.0, field=215.0, el=64.0, stop=8.0):
    """From the foot of the book, looking down on it: the whole of 靜夜思's column in, its foot clear of the subtitle."""
    keys = book.info["keys"]
    target = book.at(203.0, 133.0)
    focus = book.at(keys["x"], keys["cut"] + 22.0)
    d = field * lens / 20.25 / 1000
    a = math.radians(YAW)
    e = math.radians(el)
    loc = target + Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))) * d
    c = bpy.data.cameras.new("cam")
    c.lens, c.sensor_width, c.clip_start, c.clip_end = lens, 36.0, .01, 20.0
    o = bpy.data.objects.new("cam", c)
    bpy.context.collection.objects.link(o)
    bpy.context.scene.camera = o
    up = Vector((-math.sin(a), math.cos(a), 0))
    z = (loc - target).normalized()
    x = up.cross(z).normalized()
    y = z.cross(x)
    o.matrix_world = Matrix((x, y, z)).transposed().to_4x4()
    o.location = loc
    c.dof.use_dof = True
    c.dof.aperture_fstop = stop
    c.dof.aperture_blades = 7
    c.dof.focus_distance = (loc - focus).dot(z)
    bpy.context.view_layer.update()
    return o


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t", type=float, default=SECS - .1)
    p.add_argument("--pct", type=int, default=100)
    p.add_argument("--samples", type=int, default=192)
    p.add_argument("--name", default="tang_still")
    p.add_argument("--seq", action="store_true")
    p.add_argument("--eat", action="store_true")
    p.add_argument("--frames", default="")
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
    sc = reset(a.samples, a.pct)
    sc.render.film_transparent = False
    sc.render.image_settings.color_mode = "RGB"
    sc.cycles.max_bounces = 4
    sc.cycles.adaptive_threshold = .006
    sc.cycles.use_denoising = False
    sc.cycles.filter_width = 1.2
    table()
    book = Book(eaten=a.eat)
    window(book)
    sun = lights()
    cam = shoot(book)
    res = (1920, 1080)
    marks = dict(key=[book.pixels(cam, b, res) for b in book.info["keys"]["key"]], column=[book.pixels(cam, b, res) for b in book.info["keys"]["column"]])
    if a.eat:
        # every frame keyed, linear, and a half-open shutter: the pieces go fast enough at the end to smear, as they would on film
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
        pieces = Pieces(book)
        n = round((EAT[1] - EAT[0]) * FPS)
        for i in range(-1, n + 1):
            pieces.place(EAT[0] + i / FPS, frame=i)
        sc.render.use_motion_blur = True
        sc.render.motion_blur_shutter = .5
        sc.render.use_persistent_data = True
        sc.cycles.seed, sc.cycles.use_animated_seed = 0, False
        aim(sun, EAT[0])
        folder = os.path.join(TANG, "eat" if a.pct == 100 else "eat_p%d" % a.pct)
        os.makedirs(folder, exist_ok=True)
        only = [int(v) for v in a.frames.split(",")] if a.frames else range(n)
        for i in only:
            sc.frame_set(i)
            sc.render.filepath = os.path.join(folder, "%04d.png" % i)
            bpy.ops.render.render(write_still=True)
            print("wrote", sc.render.filepath, [tuple(round(v, 1) for v in Pieces.pose(k, EAT[0] + i / FPS)) for k in range(5)], flush=True)
        return
    if a.seq:
        sc.render.use_persistent_data = True
        sc.cycles.seed, sc.cycles.use_animated_seed = 0, False
        folder = os.path.join(TANG, "seq")
        os.makedirs(folder, exist_ok=True)
        only = [int(v) for v in a.frames.split(",")] if a.frames else range(round(SECS * FPS))
        for i in only:
            b = aim(sun, i / FPS)
            sc.render.filepath = os.path.join(folder, "%04d.png" % i)
            bpy.ops.render.render(write_still=True)
            print("wrote", sc.render.filepath, "bearing %.2f" % b, flush=True)
        return
    b = aim(sun, a.t)
    name = a.name + ("" if a.pct == 100 else "_p%d" % a.pct)
    sc.render.filepath = os.path.join(TANG, name + ".png")
    bpy.ops.render.render(write_still=True)
    with open(os.path.join(TANG, a.name + ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(t=a.t, bearing=b, **marks), f, ensure_ascii=False)
    print("wrote", sc.render.filepath, "bearing %.2f" % b, marks["key"], flush=True)


main()
