"""A laptop lid gets sticker-bombed. Blender 5.x / Cycles / Metal.

    python style_sticker_art.py
    Blender -b --factory-startup -P manim/scenes/style_sticker_blender.py -- --frames=1,30,60,90,120,150

The first sticker, a gold-leaf koi, is rolling down onto the bare metal in macro; the camera climbs away as the others
slap on faster and faster, each one rolled along a moving front without stretching, draped over the edges beneath it.
The last one fills the gap left in the middle, in holographic film, as a bar of light crosses the lid.
"""
import json
import math
import random
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "styles/film_out/cache_sticker"
FRAMES = ROOT / "styles/film_out/frames"
FPS, N = 30, 150
Z_LID = 0.36


def opt(name, default):
    for a in ARGS:
        if a.startswith("--" + name + "="):
            return type(default)(a.split("=", 1)[1])
    return default


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def link(o):
    bpy.context.collection.objects.link(o)
    return o


def aim(o, loc, target):
    o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()


def reset():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    sc.cycles.device = "GPU"
    sc.cycles.samples = 160
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = .012
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.max_bounces = 8
    sc.cycles.glossy_bounces = 4
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, N
    sc.view_settings.view_transform = "Standard"           # the prints are finished artwork: nothing may tone-map them
    sc.view_settings.look = "None"
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (.5, .54, .62, 1)
    bg.inputs["Strength"].default_value = opt("world", .07)
    return sc


def material(name, rgb, metal=0, rough=.4):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (*rgb, 1)
    p.inputs["Metallic"].default_value = metal
    p.inputs["Roughness"].default_value = rough
    return m


def image_node(nt, filename, data=False):
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(str(CACHE / filename), check_existing=True)
    if data:
        n.image.colorspace_settings.name = "Non-Color"
    n.extension = "EXTEND"
    return n


def math_node(nt, operation, a=None, b=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = operation
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def mix_colour(nt, fac, a, b):
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = "RGBA"
    nt.links.new(fac, n.inputs["Factor"])
    for name, v in (("A", a), ("B", b)):
        if isinstance(v, tuple):
            n.inputs[name].default_value = (*v, 1)
        else:
            nt.links.new(v, n.inputs[name])
    return n.outputs["Result"]


def spectrum(nt):
    """Diffraction film: colour runs with the viewing angle, with a slow drift across the sheet."""
    uv = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs[0])
    facing = nt.nodes.new("ShaderNodeLayerWeight")
    facing.inputs["Blend"].default_value = .5
    phase = math_node(nt, "MULTIPLY", facing.outputs["Facing"], 3.4)
    phase = math_node(nt, "ADD", phase, math_node(nt, "MULTIPLY", sep.outputs["X"], 2.1))
    phase = math_node(nt, "ADD", phase, math_node(nt, "MULTIPLY", sep.outputs["Y"], 1.2))
    phase = math_node(nt, "FRACT", phase)
    node = nt.nodes.new("ShaderNodeValToRGB")
    ramp = node.color_ramp
    stops = [(0, (1.0, .10, .36, 1)), (.17, (.72, .20, 1.0, 1)), (.34, (.10, .40, 1.0, 1)), (.49, (.08, 1.0, .90, 1)),
             (.64, (.70, 1.0, .18, 1)), (.80, (1.0, .60, .10, 1)), (1.0, (1.0, .10, .36, 1))]
    ramp.elements[0].position, ramp.elements[0].color = stops[0]
    ramp.elements[1].position, ramp.elements[1].color = stops[-1]
    for pos, col in stops[1:-1]:
        ramp.elements.new(pos).color = col
    nt.links.new(phase, node.inputs["Fac"])
    return node.outputs["Color"]


def print_material(s):
    m = bpy.data.materials.new(s["name"])
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    gloss = s["gloss"]
    colour = image_node(nt, s["name"] + "_colour.png").outputs["Color"]
    base, metal = colour, None
    rough = .54 - .42 * gloss
    p.inputs["Coat Weight"].default_value = .55 * gloss
    p.inputs["Coat Roughness"].default_value = .1
    p.inputs["Specular IOR Level"].default_value = .22 + .2 * gloss
    if s["gold"] or s["holo"]:
        foil = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(image_node(nt, s["name"] + "_foil.png", True).outputs["Color"], foil.inputs[0])
        if s["gold"]:
            base = mix_colour(nt, foil.outputs[0], base, (1.0, .70, .24))
            metal = foil.outputs[0]
        if s["holo"]:
            base = mix_colour(nt, foil.outputs[1], base, spectrum(nt))
            metal = foil.outputs[1] if metal is None else math_node(nt, "MAXIMUM", metal, foil.outputs[1])
        nt.links.new(metal, p.inputs["Metallic"])
        rr = nt.nodes.new("ShaderNodeMapRange")
        rr.inputs["To Min"].default_value, rr.inputs["To Max"].default_value = rough, .2
        nt.links.new(metal, rr.inputs["Value"])
        nt.links.new(rr.outputs["Result"], p.inputs["Roughness"])
    else:
        p.inputs["Roughness"].default_value = rough
    nt.links.new(base, p.inputs["Base Color"])
    noise = nt.nodes.new("ShaderNodeTexNoise")             # the faint orange peel of a laminate
    noise.inputs["Scale"].default_value = 60
    noise.inputs["Detail"].default_value = 3
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .05
    bump.inputs["Distance"].default_value = .006
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    return m


class Sticker:
    """Top and bottom skins of one die-cut, with a flat-shaded rim. Moving means re-writing its vertices."""

    def __init__(self, s, thick, backing, plan):
        d = np.load(CACHE / (s["name"] + ".npz"))
        self.s, self.plan, self.thick = s, plan, thick
        self.p = d["world"].astype(np.float64)
        self.z0 = d["z"].astype(np.float64) + Z_LID
        uv, faces, count = d["uv"], d["faces"], d["count"]
        nv = len(self.p)
        self.nv = nv
        starts = np.concatenate(((0,), np.cumsum(count)[:-1]))
        top, bottom, seen = [], [], {}
        for st, n in zip(starts, count):
            ring = faces[st:st + n][::-1].tolist()         # the art script walks them clockwise seen from above
            top.append(ring)
            bottom.append([i + nv for i in reversed(ring)])
            for a, b in zip(ring, ring[1:] + ring[:1]):
                seen[(a, b)] = True
        rim = [[b, a, a + nv, b + nv] for a, b in seen if (b, a) not in seen]
        polys = top + bottom + rim
        total = sum(len(r) for r in polys)
        me = bpy.data.meshes.new(s["name"])
        me.vertices.add(2 * nv)
        me.loops.add(total)
        me.polygons.add(len(polys))
        me.loops.foreach_set("vertex_index", [i for r in polys for i in r])
        me.polygons.foreach_set("loop_start", np.concatenate(((0,), np.cumsum([len(r) for r in polys])[:-1])).astype(np.int32))
        me.update(calc_edges=True)
        ntop, nbottom = len(top), len(bottom)
        me.polygons.foreach_set("use_smooth", [True] * (ntop + nbottom) + [False] * len(rim))
        me.polygons.foreach_set("material_index", [0] * ntop + [1] * (nbottom + len(rim)))
        layer = me.uv_layers.new(name="UVMap")
        doubled = np.vstack((uv, uv))
        layer.data.foreach_set("uv", doubled[[i for r in polys for i in r]].ravel())
        me.materials.append(print_material(s))
        me.materials.append(backing)
        self.me = me
        self.ob = link(bpy.data.objects.new(s["name"], me))
        a = plan["angle"]
        self.d = np.array((math.cos(a), math.sin(a)))
        self.n = np.array((-math.sin(a), math.cos(a)))
        self.along = self.p @ self.d
        self.across = self.p @ self.n
        self.resting = False
        self.write(self.p, self.z0, np.zeros(nv), np.ones(nv))

    def write(self, xy, z, nx, nz):
        """Skins: top at z, bottom one thickness behind it, away from the normal (-nx along d, nz up)."""
        co = np.empty((2 * self.nv, 3), np.float32)
        co[:self.nv, 0], co[:self.nv, 1], co[:self.nv, 2] = xy[:, 0], xy[:, 1], z
        co[self.nv:, 0] = xy[:, 0] + self.thick * nx * self.d[0]
        co[self.nv:, 1] = xy[:, 1] + self.thick * nx * self.d[1]
        co[self.nv:, 2] = z - self.thick * nz
        self.me.vertices.foreach_set("co", co.ravel())
        self.me.update()

    def pose(self, f):
        pl = self.plan
        begin, fly, roll = pl["start"], pl["fly"], pl["roll"]
        self.ob.hide_render = f < begin
        if f < begin:
            return
        a = (f - begin - fly) / roll
        if a >= 1:
            if not self.resting:
                self.write(self.p, self.z0, np.zeros(self.nv), np.ones(self.nv))
                self.resting = True
            return
        self.resting = False
        a = max(a, 0.0)
        tw = pl["twist"]
        wmax = np.abs(self.across).max()
        lo, hi = self.along.min() - abs(tw) * wmax, self.along.max() + abs(tw) * wmax
        front = lo + (hi - lo) * (a if pl.get("linear") else a ** 1.25) + tw * self.across
        u = self.along - front
        radius = pl["r0"] + (pl["r1"] - pl["r0"]) * a
        tmax = pl["t0"] + (pl["t1"] - pl["t0"]) * a
        theta = np.clip(u / radius, 0, tmax)
        straight = np.maximum(u - radius * tmax, 0)
        along = np.where(u > 0, front + radius * np.sin(theta) + straight * math.cos(tmax), self.along)
        z = self.z0 + radius * (1 - np.cos(theta)) + straight * math.sin(tmax)
        xy = self.p + self.d[None] * (along - self.along)[:, None]
        if fly and f < begin + fly:                        # still on its way in from above the frame
            k = (1 - (f - begin) / fly) ** 2
            xy = xy + np.array(pl["from"][:2]) * k
            z = z + pl["from"][2] * k
        self.write(xy, z, np.sin(theta), np.cos(theta))


def plans(sheet):
    """When each sticker comes down. The koi is already falling at frame 1; the gaps between the rest keep shrinking."""
    rnd = random.Random(5)
    rest = sheet[1:-1]
    first, end = 42.0, 116.0
    gaps = np.exp(-np.arange(len(rest)) * 3.0 / len(rest))
    at = first + np.concatenate(((0,), np.cumsum(gaps[:-1]))) / gaps[:-1].sum() * (end - first)
    out = {sheet[0]["name"]: dict(start=-22, fly=0, roll=68, angle=math.radians(-6), twist=.05, linear=True,
                                  r0=2.0, r1=1.0, t0=1.12, t1=.5, **{"from": (0, 0, 0)}),
           sheet[-1]["name"]: dict(start=118, fly=4, roll=15, angle=math.radians(35), twist=.1,
                                   r0=1.3, r1=.7, t0=.95, t1=.15, **{"from": (-2.2, 2.6, 6.5)})}
    for s, start in zip(rest, at):
        spot = s["size"][0] * s["size"][1]
        out[s["name"]] = dict(start=float(start), fly=3, roll=5 + min(5, spot * .9), angle=rnd.uniform(0, 2 * math.pi),
                              twist=rnd.uniform(-.14, .14), r0=.5 + spot * .05, r1=.3, t0=rnd.uniform(.7, 1.0), t1=.1,
                              **{"from": (rnd.uniform(-3, 3), rnd.uniform(1, 3.5), 7.0)})
    return out


def slab(name, w, h, thick, z0, radius, mat):
    pts = []
    for cx, cy, a0 in ((w / 2 - radius, h / 2 - radius, 0), (-w / 2 + radius, h / 2 - radius, 90),
                       (-w / 2 + radius, -h / 2 + radius, 180), (w / 2 - radius, -h / 2 + radius, 270)):
        for i in range(13):
            a = math.radians(a0 + i * 7.5)
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    n = len(pts)
    verts = [(x, y, z0) for x, y in pts] + [(x, y, z0 + thick) for x, y in pts]
    faces = [list(range(n - 1, -1, -1)), list(range(n, 2 * n))]
    faces += [[i, (i + 1) % n, (i + 1) % n + n, i + n] for i in range(n)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.materials.append(mat)
    for poly in me.polygons[2:]:
        poly.use_smooth = True
    ob = link(bpy.data.objects.new(name, me))
    bev = ob.modifiers.new("edge", "BEVEL")
    bev.width, bev.segments, bev.limit_method, bev.angle_limit = .018, 3, "ANGLE", math.radians(40)
    return ob


def bead_blast(name, rgb):
    m = material(name, rgb, metal=1.0, rough=.42)
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 900
    noise.inputs["Detail"].default_value = 2
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .14
    bump.inputs["Distance"].default_value = .004
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    return m


def area(name, loc, target, sx, sy, energy, colour=(1, 1, 1)):
    d = bpy.data.lights.new(name, "AREA")
    d.shape = "RECTANGLE"
    d.size, d.size_y = sx, sy
    d.energy = energy
    d.color = colour
    o = link(bpy.data.objects.new(name, d))
    aim(o, loc, target)
    return o


def bezier(a, b, c, u):
    return (1 - u) ** 2 * a + 2 * u * (1 - u) * b + u * u * c


def camera_pose(cam, f):
    x = (f - 1) / (N - 1)
    u = 1 - (1 - min(1.0, x / .92)) ** 1.9                 # a steady climb that comes to rest just before the end
    u += .012 * max(0.0, x - .9) / .1                      # and never quite stops
    pos = bezier(np.array((-6.05, -3.0, Z_LID + .34)), np.array((-6.6, -9.0, 3.2)), np.array((.8, -10.9, 14.6)), u)
    tgt = bezier(np.array((-1.9, -2.42, Z_LID + 1.05)), np.array((-3.0, -2.2, Z_LID)), np.array((0.0, -.5, Z_LID)), u)
    aim(cam, tuple(pos), tuple(tgt))
    cam.data.lens = 26 + 14.4 * u
    cam.data.dof.focus_distance = float(np.linalg.norm(pos - tgt))
    cam.data.dof.aperture_fstop = .7 * (9.0 ** u)


def main():
    with open(CACHE / "sheet.json") as fh:
        book = json.load(fh)
    sc = reset()
    w, h = book["lid"]
    plan = plans(book["stickers"])
    backing = material("adhesive side", (.9, .89, .86), rough=.55)
    stickers = [Sticker(s, book["thick"], backing, plan[s["name"]]) for s in book["stickers"]]
    print(f"{len(stickers)} stickers, {sum(s.nv for s in stickers) * 2} vertices", flush=True)

    desk = material("desk", (.006, .0065, .008), rough=.5)
    me = bpy.data.meshes.new("desk")
    me.from_pydata([(-200, -200, 0), (200, -200, 0), (200, 200, 0), (-200, 200, 0)], [], [(0, 1, 2, 3)])
    me.materials.append(desk)
    link(bpy.data.objects.new("desk", me))
    metal = bead_blast("anodised lid", (.12, .125, .135))
    base = slab("base", w, h, .185, 0.0, .42, metal)
    lid = slab("lid", w, h, Z_LID - .2, .2, .42, metal)

    only_stickers = bpy.data.collections.new("stickers")
    laptop = bpy.data.collections.new("laptop")            # the desk stays dark: only the laptop gets the grazing rim
    for s in stickers:
        only_stickers.objects.link(s.ob)
        laptop.objects.link(s.ob)
    laptop.objects.link(base)
    laptop.objects.link(lid)
    area("key", (-7, -7, 15), (0, 0, 0), 15, 10, opt("key", 2000), (1.0, .97, .93))
    area("fill", (9, -4, 9), (0, 0, 0), 8, 8, opt("fill", 500), (.86, .92, 1.0))
    rim = area("rim", (0, 15, 3.6), (0, 0, 0), 26, 3, opt("rim", 1100), (1.0, .95, .88))
    rim.light_linking.receiver_collection = laptop
    bar = area("bar", (0, 8, 16), (0, 0, 0), 30, 1.1, opt("bar", 700))
    bar.light_linking.receiver_collection = only_stickers

    cd = bpy.data.cameras.new("macro to the whole lid")
    cam = link(bpy.data.objects.new("camera", cd))
    sc.camera = cam
    cd.sensor_width = 36
    cd.clip_start, cd.clip_end = .02, 400
    cd.dof.use_dof = True

    test = [int(f) for f in opt("frames", "").split(",") if f]
    if test:
        sc.render.resolution_percentage = int(opt("pct", "40"))
        sc.cycles.samples = 64
        out = FRAMES / "test"
    else:
        out = FRAMES / "sticker"
    out.mkdir(parents=True, exist_ok=True)
    for f in test or range(1, N + 1):
        path = out / (f"sticker_f{f:04d}.png" if test else f"f{f:04d}.png")
        if not test and path.exists():
            continue
        t0 = time.time()
        sc.frame_set(f)
        for s in stickers:
            s.pose(f)
        camera_pose(cam, f)
        aim(bar, (-2 + 6 * ease((f - 118) / 32), 2 + 24 * ease((f - 118) / 32), 16), (0, 0, 0))
        bpy.context.view_layer.update()
        sc.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        print(f"sticker f{f:04d}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
