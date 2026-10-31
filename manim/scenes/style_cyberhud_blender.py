"""A drone flies a rain-soaked neon avenue, under skybridges and over a hanging monorail, climbs out over the rooftops
and finds the great tower of the city. Cycles / Metal.

    python style_cyberhud.py signs
    Blender -b --factory-startup -P manim/scenes/style_cyberhud_blender.py -- --frames=1,40,90,150

This renders the plate only. The HUD is vector work laid over it in style_cyberhud.py, so that it stays razor sharp;
it follows the world through track.json, which this script writes: the camera of every frame, the things worth a
bracket, and the tower footprints for the minimap. Haze is a distance-and-height mix in each shader, so it costs no
volume samples and leaves no noise.
"""
import json
import math
import os
import random
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAMES = os.path.join(ROOT, "styles", "film_out", "frames")
CACHE = os.path.join(ROOT, "styles", "film_out", "cache_cyberhud")
FPS, N = 30, 150
HAZE = 640.0
AVENUE = 23.0
TARGET = (24.0, 560.0)
SIGNS = 16
RNG = random.Random(2077)


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


# ======================================================================================
# the flight: everything the plate and the HUD agree on
# ======================================================================================
def along(t):
    """Distance down the avenue: quickening at first, then easing off to hang in front of the tower."""
    if t <= 2.2:
        return -300 + 78 * t + 8 * t * t
    u = min(t, 4.6) - 2.2
    x = u / 2.4
    return -89.68 + 113.2 * u - 71 * 2.4 * (x ** 3 - x ** 4 / 2) + 42.2 * max(0.0, t - 4.6)


def flight(t):
    """Camera position, what it looks at, roll in radians, lens in mm, at t seconds."""
    y = along(t)
    climb = ease((t - 2.1) / 2.7)
    x = 8.5 * math.sin(1.25 * t + 0.3) * (1 - climb) + 6 * climb
    z = 47 + 5 * t + 128 * climb
    pos = np.array((x, y, z))
    ahead = np.array((5.5 * math.sin(1.25 * (t + 0.5) + 0.3), y + 150, z + 7))
    tower = np.array((TARGET[0], TARGET[1], 222.0))
    swing = ease((t - 2.0) / 1.5)
    aim = ahead * (1 - swing) + tower * swing
    roll = math.radians(12) * math.sin(1.25 * t + 0.95) * (1 - ease((t - 2.4) / 1.8))
    lens = 19 + 13 * ease((t - 2.2) / 2.6)
    return pos, aim, roll, lens


def camera_matrix(pos, aim, roll):
    """Camera to world rotation: columns right, up, back."""
    f = aim - pos
    f = f / np.linalg.norm(f)
    r = np.cross(f, (0, 0, 1.0))
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    r, u = r * math.cos(roll) + u * math.sin(roll), u * math.cos(roll) - r * math.sin(roll)
    return np.column_stack((r, u, -f))


def drone_at(t):
    return np.array((-13.0, -215 + 60 * t, 73.0 + 1.5 * math.sin(2.1 * t)))


# ======================================================================================
# scene plumbing
# ======================================================================================
def setup():
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
    sc.cycles.samples = opt("samples", 96)
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.max_bounces = 6
    sc.cycles.transparent_max_bounces = 16
    sc.cycles.glossy_bounces = 3
    sc.cycles.diffuse_bounces = 2
    sc.cycles.sample_clamp_indirect = 8
    sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, N
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.08
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = opt("exposure", 0.0)

    w = bpy.data.worlds.new("night")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    bg.inputs["Strength"].default_value = 1.0
    gen = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(gen.outputs["Generated"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    rr = ramp.color_ramp
    rr.elements[0].position, rr.elements[0].color = 0.0, (*lin("#4A2C88"), 1)
    rr.elements[1].position, rr.elements[1].color = 0.6, (*lin("#060B18"), 1)
    rr.elements.new(0.05).color = (*lin("#33297A"), 1)
    rr.elements.new(0.15).color = (*lin("#1A2A62"), 1)
    rr.elements.new(0.32).color = (*lin("#0C1838"), 1)
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    # a cloud deck: the direction is projected on to a ceiling, so the bands crowd together towards the horizon
    lift = maths(nt, "ADD", sep.outputs["Z"], 0.14)
    ceil = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(maths(nt, "DIVIDE", sep.outputs["X"], lift), ceil.inputs[0])
    nt.links.new(maths(nt, "DIVIDE", sep.outputs["Y"], lift), ceil.inputs[1])
    cloud = nt.nodes.new("ShaderNodeTexNoise")
    cloud.inputs["Scale"].default_value = 0.85
    cloud.inputs["Detail"].default_value = 9
    cloud.inputs["Roughness"].default_value = 0.56
    cloud.inputs["Distortion"].default_value = 0.4
    nt.links.new(ceil.outputs[0], cloud.inputs["Vector"])
    bright = nt.nodes.new("ShaderNodeMapRange")
    bright.inputs["From Min"].default_value, bright.inputs["From Max"].default_value = 0.34, 0.68
    bright.inputs["To Min"].default_value, bright.inputs["To Max"].default_value = 0.3, 1.9
    nt.links.new(cloud.outputs["Fac"], bright.inputs["Value"])
    nt.links.new(bright.outputs["Result"], bg.inputs["Strength"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    return sc


def maths(nt, op, a=None, b=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


HAZE_NODES = []


def hazed(nt, shader):
    """Far away everything sinks into the colour of the air at its own height, in slow drifting patches."""
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    view = nt.nodes.new("ShaderNodeCameraData")
    thin = maths(nt, "MULTIPLY", view.outputs["View Distance"], -1.0 / HAZE)
    HAZE_NODES.append(thin.node)
    fac = maths(nt, "SUBTRACT", 1.0, maths(nt, "EXPONENT", thin))
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    rr = ramp.color_ramp
    rr.elements[0].position, rr.elements[0].color = 0.0, (*lin("#6A2470"), 1)
    rr.elements[1].position, rr.elements[1].color = 1.0, (*lin("#0E1A3A"), 1)
    rr.elements.new(0.14).color = (*lin("#4A2A86"), 1)
    rr.elements.new(0.36).color = (*lin("#2B3A8C"), 1)
    rr.elements.new(0.66).color = (*lin("#1B2C66"), 1)
    nt.links.new(maths(nt, "DIVIDE", sep.outputs["Z"], 420.0), ramp.inputs["Fac"])
    drift = nt.nodes.new("ShaderNodeTexNoise")
    drift.inputs["Scale"].default_value = 0.006
    drift.inputs["Detail"].default_value = 4
    nt.links.new(geo.outputs["Position"], drift.inputs["Vector"])
    patch = nt.nodes.new("ShaderNodeMapRange")
    patch.inputs["From Min"].default_value, patch.inputs["From Max"].default_value = 0.32, 0.68
    patch.inputs["To Min"].default_value, patch.inputs["To Max"].default_value = 0.28, 1.0
    nt.links.new(drift.outputs["Fac"], patch.inputs["Value"])
    air = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(ramp.outputs["Color"], air.inputs["Color"])
    nt.links.new(patch.outputs["Result"], air.inputs["Strength"])
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fac, mix.inputs[0])
    nt.links.new(shader, mix.inputs[1])
    nt.links.new(air.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])


def facade_material():
    """Dark wall, glass panes that reflect the street, and behind some panes someone at home.

    Each wall has a UV in metres. The attribute carries a seed, how many windows are lit, a palette, and a gain. Lit
    windows come in patches, the way one tenant fills a stretch of a floor, and each has blinds that catch the light
    at the top."""
    m = bpy.data.materials.new("facade")
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "facade"
    seed = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(attr.outputs["Color"], seed.inputs[0])
    uv = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs[0])
    u = maths(nt, "DIVIDE", sep.outputs["X"], maths(nt, "ADD", 1.0, maths(nt, "MULTIPLY", seed.outputs[0], 0.6)))
    v = maths(nt, "DIVIDE", sep.outputs["Y"], 2.0)
    cell = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(maths(nt, "FLOOR", u), cell.inputs[0])
    nt.links.new(maths(nt, "FLOOR", v), cell.inputs[1])
    nt.links.new(maths(nt, "MULTIPLY", seed.outputs[0], 977.0), cell.inputs[2])
    draw = nt.nodes.new("ShaderNodeTexWhiteNoise")
    draw.noise_dimensions = "3D"
    nt.links.new(cell.outputs[0], draw.inputs["Vector"])
    fx, fy = maths(nt, "FRACT", u), maths(nt, "FRACT", v)
    pane = maths(nt, "MULTIPLY", maths(nt, "MULTIPLY", maths(nt, "GREATER_THAN", fx, 0.13), maths(nt, "LESS_THAN", fx, 0.87)),
                 maths(nt, "MULTIPLY", maths(nt, "GREATER_THAN", fy, 0.24), maths(nt, "LESS_THAN", fy, 0.84)))
    tenants = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(maths(nt, "ADD", maths(nt, "MULTIPLY", sep.outputs["X"], 0.045), maths(nt, "MULTIPLY", seed.outputs[0], 91.0)), tenants.inputs[0])
    nt.links.new(maths(nt, "ADD", maths(nt, "MULTIPLY", sep.outputs["Y"], 0.16), maths(nt, "MULTIPLY", seed.outputs[0], 37.0)), tenants.inputs[1])
    patch = nt.nodes.new("ShaderNodeTexNoise")
    patch.inputs["Scale"].default_value = 1.0
    patch.inputs["Detail"].default_value = 2
    nt.links.new(tenants.outputs[0], patch.inputs["Vector"])
    busy = nt.nodes.new("ShaderNodeMapRange")
    busy.inputs["From Min"].default_value, busy.inputs["From Max"].default_value = 0.42, 0.62
    busy.inputs["To Min"].default_value, busy.inputs["To Max"].default_value = 0.1, 2.6
    nt.links.new(patch.outputs["Fac"], busy.inputs["Value"])
    chance = maths(nt, "MULTIPLY", seed.outputs[1], busy.outputs[0])
    lit = maths(nt, "MULTIPLY", pane, maths(nt, "LESS_THAN", draw.outputs["Value"], chance))
    hue = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(draw.outputs["Color"], hue.inputs[0])
    palette = nt.nodes.new("ShaderNodeValToRGB")
    pr = palette.color_ramp
    pr.interpolation = "CONSTANT"
    pr.elements[0].position, pr.elements[0].color = 0.0, (*lin("#FFA24A"), 1)
    pr.elements[1].position, pr.elements[1].color = 0.92, (*lin("#8CFFC0"), 1)
    pr.elements.new(0.42).color = (*lin("#FFD9A8"), 1)
    pr.elements.new(0.66).color = (*lin("#A6E9FF"), 1)
    pr.elements.new(0.82).color = (*lin("#FF6FB0"), 1)
    nt.links.new(maths(nt, "ADD", seed.outputs[2], maths(nt, "MULTIPLY", hue.outputs[0], 0.3)), palette.inputs["Fac"])
    nt.links.new(palette.outputs["Color"], p.inputs["Emission Color"])
    blinds = maths(nt, "ADD", 0.45, maths(nt, "MULTIPLY", fy, 0.75))
    power = maths(nt, "ADD", 0.14, maths(nt, "MULTIPLY", maths(nt, "POWER", hue.outputs[1], 4.0), 1.5))
    nt.links.new(maths(nt, "MULTIPLY", maths(nt, "MULTIPLY", lit, maths(nt, "MULTIPLY", power, blinds)), attr.outputs["Alpha"]), p.inputs["Emission Strength"])
    # rain has streaked the walls: stretched noise darkens them and makes them shine
    streak = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(maths(nt, "MULTIPLY", sep.outputs["X"], 0.9), streak.inputs[0])
    nt.links.new(maths(nt, "MULTIPLY", sep.outputs["Y"], 0.035), streak.inputs[1])
    grime = nt.nodes.new("ShaderNodeTexNoise")
    grime.inputs["Scale"].default_value = 1.0
    grime.inputs["Detail"].default_value = 5
    nt.links.new(streak.outputs[0], grime.inputs["Vector"])
    wall = nt.nodes.new("ShaderNodeMix")
    wall.data_type = "RGBA"
    nt.links.new(grime.outputs["Fac"], wall.inputs["Factor"])
    wall.inputs["A"].default_value = (*lin("#090B10"), 1)
    wall.inputs["B"].default_value = (*lin("#1C2029"), 1)
    colour = nt.nodes.new("ShaderNodeMix")
    colour.data_type = "RGBA"
    nt.links.new(pane, colour.inputs["Factor"])
    nt.links.new(wall.outputs["Result"], colour.inputs["A"])
    colour.inputs["B"].default_value = (*lin("#10141A"), 1)
    nt.links.new(colour.outputs["Result"], p.inputs["Base Color"])
    nt.links.new(maths(nt, "MULTIPLY", pane, 0.4), p.inputs["Metallic"])
    rough = maths(nt, "ADD", maths(nt, "MULTIPLY", maths(nt, "SUBTRACT", 1.0, pane), 0.32), maths(nt, "MULTIPLY", grime.outputs["Fac"], 0.3))
    nt.links.new(maths(nt, "ADD", rough, 0.03), p.inputs["Roughness"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.9
    bump.inputs["Distance"].default_value = 0.08
    nt.links.new(maths(nt, "SUBTRACT", 1.0, pane), bump.inputs["Height"])
    nt.links.new(bump.outputs[0], p.inputs["Normal"])
    hazed(nt, p.outputs[0])
    return m


def neon_material():
    m = bpy.data.materials.new("neon")
    m.use_nodes = True
    nt = m.node_tree
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "glow"
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(attr.outputs["Color"], em.inputs["Color"])
    nt.links.new(maths(nt, "MULTIPLY", attr.outputs["Alpha"], 3.0), em.inputs["Strength"])
    hazed(nt, em.outputs[0])
    return m


def pulse_material():
    """The tower's own lights: low, until the moment of the lock sends a wave up the building that leaves them burning."""
    m = bpy.data.materials.new("pulse")
    m.use_nodes = True
    nt = m.node_tree
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "glow"
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    front = nt.nodes.new("ShaderNodeValue")
    front.name = "front"
    front.outputs[0].default_value = -60.0
    gap = maths(nt, "DIVIDE", maths(nt, "SUBTRACT", sep.outputs["Z"], front.outputs[0]), 16.0)
    wave = maths(nt, "EXPONENT", maths(nt, "MULTIPLY", maths(nt, "MULTIPLY", gap, gap), -1.0))
    em = nt.nodes.new("ShaderNodeEmission")
    tint = nt.nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    nt.links.new(wave, tint.inputs["Factor"])
    nt.links.new(attr.outputs["Color"], tint.inputs["A"])
    tint.inputs["B"].default_value = (1, 1, 1, 1)
    nt.links.new(tint.outputs["Result"], em.inputs["Color"])
    behind = maths(nt, "MINIMUM", 1.0, maths(nt, "MAXIMUM", maths(nt, "ADD", 0.5, maths(nt, "DIVIDE", maths(nt, "SUBTRACT", front.outputs[0], sep.outputs["Z"]), 12.0)), 0.0))
    lit = maths(nt, "ADD", 0.22, maths(nt, "MULTIPLY", behind, 0.78))
    nt.links.new(maths(nt, "ADD", maths(nt, "MULTIPLY", maths(nt, "MULTIPLY", attr.outputs["Alpha"], 3.0), lit), maths(nt, "MULTIPLY", wave, 9.0)), em.inputs["Strength"])
    hazed(nt, em.outputs[0])
    return m, front


def sign_material(i, strength=2.1):
    m = bpy.data.materials.new(f"sign{i}")
    m.use_nodes = True
    nt = m.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.join(CACHE, f"sign_{i:02d}.png"), check_existing=True)
    tex.extension = "EXTEND"
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = strength
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    hazed(nt, em.outputs[0])
    return m


def light_material(name, colour, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*colour, 1)
    em.inputs["Strength"].default_value = strength
    hazed(nt, em.outputs[0])
    return m


class Bag:
    """Solids gathered into one mesh, each face with a UV in metres and a colour attribute."""

    def __init__(self, dark_roofs=False):
        self.verts, self.faces, self.uv, self.col = [], [], [], []
        self.dark_roofs = dark_roofs

    def _roof(self, col):
        return (col[0], 0.0, col[2], col[3]) if self.dark_roofs else col

    def box(self, x0, x1, y0, y1, z0, z1, col):
        x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        k = len(self.verts)
        self.verts += [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        walls = (((0, 1, 5, 4), x0, x1), ((1, 2, 6, 5), y0, y1), ((2, 3, 7, 6), x1, x0), ((3, 0, 4, 7), y1, y0))
        for q, a, b in walls:
            self.faces.append(tuple(k + i for i in q))
            self.uv += [(a, z0), (b, z0), (b, z1), (a, z1)]
            self.col += [col] * 4
        self.faces.append((k + 4, k + 5, k + 6, k + 7))
        self.uv += [(0, 0)] * 4
        self.col += [self._roof(col)] * 4

    def prism(self, loop, z0, z1, col, scale=1.0, cap=True, under=False):
        """A footprint (anticlockwise from above) extruded up, narrowing or flaring about its centre."""
        cx, cy = sum(p[0] for p in loop) / len(loop), sum(p[1] for p in loop) / len(loop)
        n, k, s = len(loop), len(self.verts), 0.0
        self.verts += [(x, y, z0) for x, y in loop]
        self.verts += [(cx + (x - cx) * scale, cy + (y - cy) * scale, z1) for x, y in loop]
        for i in range(n):
            j = (i + 1) % n
            step = math.hypot(loop[j][0] - loop[i][0], loop[j][1] - loop[i][1])
            self.faces.append((k + i, k + j, k + n + j, k + n + i))
            self.uv += [(s, z0), (s + step, z0), (s + step, z1), (s, z1)]
            self.col += [col] * 4
            s += step
        if cap:
            self.faces.append(tuple(k + n + i for i in range(n)))
            self.uv += [(0, 0)] * n
            self.col += [self._roof(col)] * n
        if under:
            self.faces.append(tuple(k + i for i in reversed(range(n))))
            self.uv += [(0, 0)] * n
            self.col += [self._roof(col)] * n

    def tube(self, points, r, col):
        """A cable: a square section swept along a polyline."""
        for a, b in zip(points, points[1:]):
            a, b = np.array(a, float), np.array(b, float)
            d = (b - a) / max(1e-6, np.linalg.norm(b - a))
            s = np.cross(d, (0, 0, 1.0))
            s = s / max(1e-6, np.linalg.norm(s)) * r
            w = np.cross(s, d)
            k = len(self.verts)
            for c in (a, b):
                self.verts += [tuple(c - s - w), tuple(c + s - w), tuple(c + s + w), tuple(c - s + w)]
            for i in range(4):
                j = (i + 1) % 4
                self.faces.append((k + i, k + j, k + 4 + j, k + 4 + i))
                self.uv += [(0, 0)] * 4
                self.col += [self._roof(col)] * 4

    def extend(self, other):
        """Append generated parts without making another Blender object."""
        k = len(self.verts)
        self.verts.extend(other.verts)
        self.faces.extend(tuple(k + i for i in face) for face in other.faces)
        self.uv.extend(other.uv)
        self.col.extend(other.col)

    def mesh(self, name, attr, mat):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.verts, [], self.faces)
        me.update()
        me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.array(self.uv, np.float32).ravel())
        me.color_attributes.new(attr, "FLOAT_COLOR", "CORNER").data.foreach_set("color", np.array(self.col, np.float32).ravel())
        me.materials.append(mat)
        return link(bpy.data.objects.new(name, me))


TONES = [lin(c) for c in ("#FF2A6D", "#19F0FF", "#FFB000", "#B56BFF", "#FF4B2B", "#37FF8B")]
SHOPS = [lin(c) for c in ("#FFC27A", "#FFE3B8", "#FF4F7E", "#FF8A3C", "#7CF7E8", "#FFD24A", "#C58BFF", "#8FFFB0")]
WARM = (0.0, 0.1, 0.2, 0.42, 0.56)


def chamfered(x, y, w, d, k):
    w, d = w / 2, d / 2
    return [(x - w + k, y - d), (x + w - k, y - d), (x + w, y - d + k), (x + w, y + d - k),
            (x + w - k, y + d), (x - w + k, y + d), (x - w, y + d - k), (x - w, y - d + k)]


def round_loop(x, y, r, n=8, turn=math.pi / 8):
    return [(x + r * math.cos(turn + 2 * math.pi * i / n), y + r * math.sin(turn + 2 * math.pi * i / n)) for i in range(n)]


def tower(bag, neon, x, y, w, d, h, lights=True, first=None):
    """A shaft with setbacks. Most stand dark against the sky; a few wear a crown of light or lit bands."""
    seed = RNG.random()
    col = (seed, RNG.uniform(0.07, 0.3), RNG.choice(WARM), 1.0)
    dark = (seed, 0.0, 0.0, 1.0)
    tiers = RNG.choice((1, 2, 2, 3, 4)) if h > 70 else 1
    z, ww, dd = 0.0, w, d
    tone = RNG.choice(TONES)
    style = RNG.random() if lights else 1.0
    corner = RNG.random() < 0.34
    for i in range(tiers):
        top = h if i == tiers - 1 else first if first and i == 0 else z + (h - z) * RNG.uniform(0.45, 0.72)
        if corner:
            bag.prism(chamfered(x, y, ww, dd, min(ww, dd) * RNG.uniform(0.12, 0.3)), z, top, col)
        else:
            bag.box(x - ww / 2, x + ww / 2, y - dd / 2, y + dd / 2, z, top, col)
        for k in range(RNG.randint(1, 4)):
            lz = z + (top - z) * RNG.random()
            bag.box(x - ww / 2 - 0.35, x + ww / 2 + 0.35, y - dd / 2 - 0.35, y + dd / 2 + 0.35, lz, lz + 0.7, dark)
        if style < 0.1 and (i == tiers - 1 or RNG.random() < 0.4):       # a crown of light on the setback
            e = 0.14
            neon.box(x - ww / 2 - e, x + ww / 2 + e, y - dd / 2 - e, y - dd / 2 + e, top - 0.6, top, (*tone, 0.8))
            neon.box(x - ww / 2 - e, x + ww / 2 + e, y + dd / 2 - e, y + dd / 2 + e, top - 0.6, top, (*tone, 0.8))
            neon.box(x - ww / 2 - e, x - ww / 2 + e, y - dd / 2, y + dd / 2, top - 0.6, top, (*tone, 0.8))
            neon.box(x + ww / 2 - e, x + ww / 2 + e, y - dd / 2, y + dd / 2, top - 0.6, top, (*tone, 0.8))
        z, ww, dd = top, ww * RNG.uniform(0.62, 0.85), dd * RNG.uniform(0.62, 0.85)
    if 0.1 <= style < 0.17:                                          # thin bands of light every so many floors
        step = RNG.choice((12, 18, 26))
        for lz in np.arange(step, h - 4, step):
            neon.box(x - w / 2 - 0.12, x + w / 2 + 0.12, y - d / 2 - 0.12, y - d / 2 + 0.1, lz, lz + 0.3, (*tone, 0.45))
    for k in range(RNG.randint(1, 4)):
        bx, by = x + RNG.uniform(-0.3, 0.3) * ww, y + RNG.uniform(-0.3, 0.3) * dd
        bw = RNG.uniform(1.5, 5)
        bag.box(bx - bw, bx + bw, by - bw, by + bw, h, h + RNG.uniform(1.5, 6), dark)
    if RNG.random() < 0.3:
        ax, ay = x + RNG.uniform(-0.3, 0.3) * ww, y + RNG.uniform(-0.3, 0.3) * dd
        ah = RNG.uniform(10, 34)
        bag.box(ax - 0.3, ax + 0.3, ay - 0.3, ay + 0.3, h, h + ah, dark)
        neon.box(ax - 0.5, ax + 0.5, ay - 0.5, ay + 0.5, h + ah, h + ah + 0.9, (*lin("#FF2020"), 1.2))


def dress(bag, neon, side, y0, y1, h):
    """What a wall you pass close to carries: shops at its foot, then fins, pipes, ledges and air conditioners."""
    def out(a, b):
        return side * (AVENUE - a), side * (AVENUE - b)
    seed = RNG.random()
    dark = (seed, 0.0, 0.0, 1.0)
    y = y0
    while y < y1 - 2:                                                # the shops, and the lit fascia over each
        w = min(RNG.uniform(3.5, 11), y1 - y)
        tone = RNG.choice(SHOPS)
        neon.box(*out(-0.1, 0.25), y, y + w, 0.2, RNG.uniform(3.4, 4.6), (*tone, RNG.uniform(0.55, 1.15)))
        if RNG.random() < 0.7:
            neon.box(*out(0.25, 0.8), y + 0.3, y + w - 0.3, 4.9, RNG.uniform(5.8, 7.6), (*RNG.choice(SHOPS), RNG.uniform(0.8, 1.5)))
        bag.box(*out(0.0, 2.4), y, y + w, 4.6, 4.85, dark)           # awning
        y += w + RNG.uniform(0.3, 0.9)
    for k in range(RNG.randint(1, 3)):                               # clusters of vertical fins
        a = RNG.uniform(y0, y1 - 8)
        z0 = RNG.uniform(8, h * 0.4)
        z1 = min(h, z0 + RNG.uniform(30, 110))
        for i in range(RNG.randint(5, 18)):
            bag.box(*out(0.0, RNG.uniform(0.4, 0.9)), a + i * 0.95, a + i * 0.95 + 0.26, z0, z1, dark)
    for k in range(RNG.randint(3, 9)):                               # ledges
        lz = RNG.uniform(9, h)
        bag.box(*out(0.0, RNG.uniform(0.3, 0.7)), y0, y1, lz, lz + RNG.uniform(0.25, 0.6), dark)
    for k in range(RNG.randint(1, 4)):                               # pipes
        a = RNG.uniform(y0, y1 - 1)
        bag.box(*out(0.1, 0.42), a, a + 0.28, 5, h * RNG.uniform(0.4, 0.95), dark)
    for k in range(RNG.randint(40, 110)):                            # air conditioners
        a, lz = RNG.uniform(y0, y1 - 2), RNG.uniform(8, min(h, 130))
        bag.box(*out(0.0, RNG.uniform(0.6, 1.1)), a, a + RNG.uniform(0.9, 1.5), lz, lz + RNG.uniform(0.7, 1.1), dark)
    for k in range(RNG.randint(2, 7)):                               # strip lights, here and there
        a, lz, ln = RNG.uniform(y0, y1 - 4), RNG.uniform(10, min(h, 120)), RNG.uniform(1.5, 8)
        neon.box(*out(0.0, 0.12), a, a + ln, lz, lz + 0.16, (*RNG.choice(TONES), 0.7))


def skybridge(bag, neon, y, z, w, h):
    """A lit glass corridor from wall to wall, with a line of light down its belly."""
    seed = RNG.random()
    bag.box(-AVENUE, AVENUE, y, y + w, z, z + h, (seed, 0.55, RNG.choice(WARM), 1.5))
    dark = (seed, 0.0, 0.0, 1.0)
    bag.box(-AVENUE, AVENUE, y - 0.4, y + w + 0.4, z - 0.5, z, dark)
    bag.box(-AVENUE, AVENUE, y - 0.4, y + w + 0.4, z + h, z + h + 0.5, dark)
    tone = RNG.choice(TONES)
    for a in (y + 0.5, y + w - 0.7):
        neon.box(-AVENUE, AVENUE, a, a + 0.22, z - 0.66, z - 0.5, (*tone, 1.5))
    for x in np.arange(-AVENUE + 3.8, AVENUE, 7.6):                 # the trusses
        bag.box(x - 0.18, x + 0.18, y - 0.45, y - 0.05, z, z + h, dark)


def wires(bag, y0, y1, count):
    dark = (0.5, 0.0, 0.0, 1.0)
    for _ in range(count):
        ya = RNG.uniform(y0, y1)
        yb = ya + RNG.uniform(-14, 14)
        za = RNG.uniform(86, 150) if ya < -125 and RNG.random() < 0.7 else RNG.uniform(10, 27)
        zb = za + RNG.uniform(-4, 4)
        sag = RNG.uniform(1, 4.5)
        pts = []
        for i in range(13):
            u = i / 12
            pts.append((-AVENUE + 2 * AVENUE * u, ya + (yb - ya) * u, za + (zb - za) * u - sag * 4 * u * (1 - u)))
        bag.tube(pts, RNG.uniform(0.035, 0.08), dark)


def monorail(bag, neon):
    """Two hanging rails down the avenue, each with a bracket out of the wall every so often."""
    dark = (0.2, 0.0, 0.0, 1.0)
    for x in (-13.5, 13.5):
        bag.box(x - 0.55, x + 0.55, -470, 450, 33.0, 34.1, dark)
        neon.box(x - 0.6, x - 0.55, -470, 450, 33.3, 33.38, (*lin("#19F0FF"), 0.22))
        neon.box(x + 0.55, x + 0.6, -470, 450, 33.3, 33.38, (*lin("#19F0FF"), 0.22))
        for y in np.arange(-460, 450, 34):
            wall = math.copysign(AVENUE, x)
            bag.box(min(x, wall), max(x, wall), y - 0.3, y + 0.3, 34.1, 34.9, dark)


class Train:
    """A monorail set hanging under its rail, its windows lit, sliding through the frame."""

    def __init__(self, name, x, start, speed, tone):
        self.start, self.speed = start, speed
        body, glow = Bag(True), Bag()
        for i in range(7):
            y0, y1 = i * 17.0, i * 17.0 + 16.2
            body.box(x - 1.45, x + 1.45, y0, y1, 28.6, 32.2, (0.3, 0.0, 0.0, 1.0))
            glow.box(x - 1.5, x + 1.5, y0 + 0.5, y1 - 0.5, 29.9, 31.3, (*lin("#EAF6FF"), 1.0))
            glow.box(x - 1.5, x + 1.5, y0, y1, 29.2, 29.42, (*tone, 1.5))
            body.box(x - 0.3, x + 0.3, y0 + 2, y0 + 3, 32.2, 33.0, (0.3, 0.0, 0.0, 1.0))
            body.box(x - 0.3, x + 0.3, y1 - 3, y1 - 2, 32.2, 33.0, (0.3, 0.0, 0.0, 1.0))
        self.parts = (body.mesh(name, "facade", bpy.data.materials["facade"]), glow.mesh(name + " light", "glow", bpy.data.materials["neon"]))

    def at(self, t):
        for o in self.parts:
            o.location = (0, self.start + self.speed * t, 0)


def landmark(bag, pulse):
    """荒坂塔: stacked flaring sections, an eastern silhouette you can't mistake, black and red, with no lettering on it."""
    tx, ty = TARGET
    col = (0.37, 0.34, 0.2, 1.9)
    dark = (0.37, 0.0, 0.0, 1.0)
    rim, spine = (*lin("#FF1E2D"), 1.2), (*lin("#B00C1C"), 0.45)
    bag.prism(chamfered(tx, ty, 124, 124, 26), 0, 30, col)
    bag.prism(chamfered(tx, ty, 92, 92, 20), 30, 76, col, 0.9)
    z, r = 76.0, 40.0
    for i in range(7):
        h = 28.0
        flare = 1.13
        bag.prism(round_loop(tx, ty, r), z, z + h, col, flare)
        pulse.prism(round_loop(tx, ty, r * flare + 0.25), z + h - 0.3, z + h + 0.5, rim, 1.0, False)      # the lip of each section
        for k in range(8):                                           # a line of light up every corner
            a = math.pi / 8 + 2 * math.pi * k / 8
            u = (math.cos(a), math.sin(a))
            pulse.prism(round_loop(tx + u[0] * (r + 0.1), ty + u[1] * (r + 0.1), 0.42, 4, 0.0), z, z + h, spine, 1.0, False)
        z += h
        r *= 0.885
    crown = r * 1.05
    bag.prism(round_loop(tx, ty, crown), z, z + 26, col, 0.46)
    pulse.prism(round_loop(tx, ty, crown * 0.46 + 0.2), z + 25.6, z + 26.4, rim, 1.0, False)
    z += 26
    bag.prism(round_loop(tx, ty, crown * 0.42), z, z + 9, dark, 0.45)
    z += 9
    bag.prism(round_loop(tx, ty, 2.4), z, z + 34, dark, 0.14)
    for dz, rr in ((5, 3.0), (13, 2.2), (21, 1.5)):
        pulse.prism(round_loop(tx, ty, rr), z + dz, z + dz + 0.5, rim, 1.0, True, True)
    top = z + 34
    beacon = light_material("beacon", lin("#FF1030"), 5.0)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.5, location=(tx, ty, top + 1.0))
    bpy.context.object.data.materials.append(beacon)
    return top


def beam_material(colour, strength, length):
    """A searchlight seen through rainy air: strongest where you look through most of it, fading along its length."""
    m = bpy.data.materials.new("beam")
    m.use_nodes = True
    m.use_transparent_shadow = True
    nt = m.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    nt.links.new(geo.outputs["Incoming"], dot.inputs[0])
    nt.links.new(geo.outputs["Normal"], dot.inputs[1])
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    along = maths(nt, "DIVIDE", sep.outputs["Z"], length)
    fade = maths(nt, "MULTIPLY", maths(nt, "POWER", maths(nt, "MAXIMUM", maths(nt, "SUBTRACT", 1.0, along), 0.0), 1.5),
                 maths(nt, "MINIMUM", maths(nt, "MULTIPLY", along, 7.0), 1.0))
    soft = maths(nt, "POWER", maths(nt, "ABSOLUTE", dot.outputs["Value"]), 3.6)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*colour, 1)
    nt.links.new(maths(nt, "MULTIPLY", maths(nt, "MULTIPLY", soft, fade), strength), em.inputs["Strength"])
    add = nt.nodes.new("ShaderNodeAddShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(tr.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    return m


class Beam:
    def __init__(self, x, y, base_z, tone, strength, tilt, phase, rate, L=1000.0, r=46.0):
        n = 28
        verts = [(0.5 * math.cos(2 * math.pi * i / n), 0.5 * math.sin(2 * math.pi * i / n), 0) for i in range(n)]
        verts += [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), L) for i in range(n)]
        faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
        me = bpy.data.meshes.new("beam")
        me.from_pydata(verts, [], faces)
        me.polygons.foreach_set("use_smooth", [True] * n)
        me.update()
        me.materials.append(beam_material(tone, strength, L))
        self.o = link(bpy.data.objects.new("beam", me))
        self.o.location = (x, y, base_z)
        self.o.rotation_mode = "XYZ"
        self.tilt, self.phase, self.rate = tilt, phase, rate
        self.o.visible_shadow = False
        self.o.visible_diffuse = self.o.visible_glossy = False

    def at(self, t):
        a = self.phase + self.rate * t
        self.o.rotation_euler = (self.tilt * math.cos(a), self.tilt * math.sin(a), 0)


def plane(name, corners, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(corners, [], [(0, 1, 2, 3)])
    me.update()
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", (0, 0, 1, 0, 1, 1, 0, 1))
    me.materials.append(mat)
    return link(bpy.data.objects.new(name, me))


def blade(bag, i, side, y, z0, depth, height, mats):
    """A sign standing out from a facade into the avenue, on a dark frame and brackets: it faces the traffic."""
    a, b = (AVENUE - depth, AVENUE) if side > 0 else (-AVENUE, -AVENUE + depth)
    dark = (0.6, 0.0, 0.0, 1.0)
    bag.box(a - 0.14, b, y + 0.02, y + 0.7, z0 - 0.16, z0 + height + 0.16, dark)
    return plane(f"blade{i}", [(a, y, z0), (b, y, z0), (b, y, z0 + height), (a, y, z0 + height)], mats[i % SIGNS])


def skyline_hazed(nt, shader):
    """Four depth anchors: dark foreground, two receding ranks, faint far silhouettes.

    Continuous interpolation avoids bands across a facade or popping during the flight.
    This does not alter the avenue/landmark hazed() shader or the cloud deck.
    """
    view = nt.nodes.new("ShaderNodeCameraData")
    depth = nt.nodes.new("ShaderNodeValToRGB")
    rr = depth.color_ramp
    rr.interpolation = "LINEAR"
    rr.elements[0].position, rr.elements[0].color = 500 / 1500, (0.035, 0.035, 0.035, 1)
    rr.elements[1].position, rr.elements[1].color = 1.0, (0.86, 0.86, 0.86, 1)
    rr.elements.new(800 / 1500).color = (0.24, 0.24, 0.24, 1)
    rr.elements.new(1100 / 1500).color = (0.55, 0.55, 0.55, 1)
    nt.links.new(maths(nt, "DIVIDE", view.outputs["View Distance"], 1500.0), depth.inputs["Fac"])
    fog = depth.outputs["Color"]
    air = nt.nodes.new("ShaderNodeEmission")
    air.inputs["Color"].default_value = (*lin("#45456F"), 1)
    air.inputs["Strength"].default_value = 0.85
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fog, mix.inputs[0])
    nt.links.new(shader, mix.inputs[1])
    nt.links.new(air.outputs[0], mix.inputs[2])
    return mix.outputs[0]


def skyline_roof_shader(nt):
    """Dark weathered roofs with sparse 2--4 m maintenance pads; no emissive roof grid."""
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    p.inputs["Roughness"].default_value = 0.68
    p.inputs["Metallic"].default_value = 0.12
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    pos = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], pos.inputs[0])
    weather = nt.nodes.new("ShaderNodeTexNoise")
    weather.inputs["Scale"].default_value = 0.24
    weather.inputs["Detail"].default_value = 2.0
    nt.links.new(geo.outputs["Position"], weather.inputs["Vector"])
    base = nt.nodes.new("ShaderNodeMix")
    base.data_type = "RGBA"
    nt.links.new(weather.outputs["Fac"], base.inputs["Factor"])
    base.inputs["A"].default_value = (*lin("#090E14"), 1)
    base.inputs["B"].default_value = (*lin("#1A2530"), 1)
    u, v = maths(nt, "DIVIDE", pos.outputs["X"], 8.0), maths(nt, "DIVIDE", pos.outputs["Y"], 10.0)
    cell = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(maths(nt, "FLOOR", u), cell.inputs[0])
    nt.links.new(maths(nt, "FLOOR", v), cell.inputs[1])
    nt.links.new(pos.outputs["Z"], cell.inputs[2])
    draw = nt.nodes.new("ShaderNodeTexWhiteNoise")
    draw.noise_dimensions = "3D"
    nt.links.new(cell.outputs[0], draw.inputs["Vector"])
    fx, fy = maths(nt, "FRACT", u), maths(nt, "FRACT", v)
    pad = maths(nt, "MULTIPLY", maths(nt, "MULTIPLY", maths(nt, "GREATER_THAN", fx, 0.22), maths(nt, "LESS_THAN", fx, 0.56)),
                maths(nt, "MULTIPLY", maths(nt, "GREATER_THAN", fy, 0.20), maths(nt, "LESS_THAN", fy, 0.54)))
    pad = maths(nt, "MULTIPLY", pad, maths(nt, "LESS_THAN", draw.outputs["Value"], 0.16))
    surface = nt.nodes.new("ShaderNodeMix")
    surface.data_type = "RGBA"
    nt.links.new(pad, surface.inputs["Factor"])
    nt.links.new(base.outputs["Result"], surface.inputs["A"])
    surface.inputs["B"].default_value = (*lin("#303C46"), 1)
    nt.links.new(surface.outputs["Result"], p.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.28
    bump.inputs["Distance"].default_value = 0.12
    nt.links.new(pad, bump.inputs["Height"])
    nt.links.new(bump.outputs[0], p.inputs["Normal"])
    return p


def skyline_street_field(nt, position):
    """One world-space source field shared by the road pools and the rising wall bounce."""
    pos = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(position, pos.inputs[0])
    dx = maths(nt, "MULTIPLY", maths(nt, "ABSOLUTE", maths(nt, "SUBTRACT", maths(nt, "FRACT", maths(nt, "DIVIDE", pos.outputs["X"], 124.0)), 0.5)), 124.0)
    dy = maths(nt, "MULTIPLY", maths(nt, "ABSOLUTE", maths(nt, "SUBTRACT", maths(nt, "FRACT", maths(nt, "DIVIDE", pos.outputs["Y"], 168.0)), 0.5)), 168.0)
    dist = maths(nt, "MINIMUM", dx, dy)
    pool = maths(nt, "EXPONENT", maths(nt, "MULTIPLY", maths(nt, "MULTIPLY", dist, dist), -1.0 / 450.0))
    # Project to the street plane so hue stays consistent as its light rises up a wall.
    ground = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(pos.outputs["X"], ground.inputs[0])
    nt.links.new(pos.outputs["Y"], ground.inputs[1])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.012
    noise.inputs["Detail"].default_value = 1.0
    nt.links.new(ground.outputs[0], noise.inputs["Vector"])
    tint = nt.nodes.new("ShaderNodeValToRGB")
    tint.color_ramp.elements[0].position = 0.40
    tint.color_ramp.elements[0].color = (*lin("#F3AA70"), 1)
    tint.color_ramp.elements[1].position = 0.60
    tint.color_ramp.elements[1].color = (*lin("#CF528F"), 1)
    nt.links.new(noise.outputs["Fac"], tint.inputs["Fac"])
    return maths(nt, "MULTIPLY", pool, maths(nt, "ADD", 0.5, maths(nt, "MULTIPLY", noise.outputs["Fac"], 0.5))), tint.outputs["Color"]


def skyline_material(source, glow=False, roof_only=False):
    """One shared material per merged layer; attributes, not materials, distinguish tenants.

    Keep the existing transition after 2.5 s. Bays stay at architectural scale, NOT distance LOD:
    2.8--3.3 m clear panes project to about 9--10 / 6--7 / 3--4 px at 550 / 850 / 1400 m
    with the final 32 mm camera. Clear height is 1.7--1.9 m (about 5--6 px in the near rank).
    """
    m = source.copy()
    m.name = "skyline roof" if roof_only else "skyline glow" if glow else "skyline facade"
    nt = m.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    legacy = out.inputs["Surface"].links[0].from_socket
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "glow" if glow else "facade"
    if glow:
        p = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(attr.outputs["Color"], p.inputs["Color"])
        nt.links.new(maths(nt, "MULTIPLY", attr.outputs["Alpha"], 1.6), p.inputs["Strength"])
    elif roof_only:
        p = skyline_roof_shader(nt)
    else:
        p = nt.nodes.new("ShaderNodeBsdfPrincipled")
        p.inputs["Base Color"].default_value = (*lin("#18212C"), 1)
        p.inputs["Roughness"].default_value = 0.52
        p.inputs["Metallic"].default_value = 0.18
        seed = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(attr.outputs["Color"], seed.inputs[0])
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        pos, normal = nt.nodes.new("ShaderNodeSeparateXYZ"), nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(geo.outputs["Position"], pos.inputs[0])
        nt.links.new(geo.outputs["Normal"], normal.inputs[0])
        tc = nt.nodes.new("ShaderNodeTexCoord")
        uv = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["UV"], uv.inputs[0])
        # Constant across each building: world-Y scaling would bend floors along its side walls.
        lod = nt.nodes.new("ShaderNodeAttribute")
        lod.attribute_name = "skyline rank"
        lod_channels = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(lod.outputs["Color"], lod_channels.inputs[0])
        rank = lod_channels.outputs[0]
        bay = maths(nt, "ADD", 3.9, maths(nt, "MULTIPLY", seed.outputs[0], 0.7))
        floor = maths(nt, "ADD", 3.15, maths(nt, "MULTIPLY", seed.outputs[0], 0.3))
        u, v = maths(nt, "DIVIDE", uv.outputs["X"], bay), maths(nt, "DIVIDE", uv.outputs["Y"], floor)
        fx, fy = maths(nt, "FRACT", u), maths(nt, "FRACT", v)
        pane = maths(nt, "MULTIPLY", maths(nt, "MULTIPLY", maths(nt, "GREATER_THAN", fx, 0.14), maths(nt, "LESS_THAN", fx, 0.86)),
                     maths(nt, "MULTIPLY", maths(nt, "GREATER_THAN", fy, 0.20), maths(nt, "LESS_THAN", fy, 0.74)))
        # Three independent scales: dark wings, multi-floor tenants, individual switched-off rooms.
        def draw_cells(width, floors, salt):
            cell = nt.nodes.new("ShaderNodeCombineXYZ")
            nt.links.new(maths(nt, "FLOOR", maths(nt, "DIVIDE", u, width)), cell.inputs[0])
            nt.links.new(maths(nt, "FLOOR", maths(nt, "DIVIDE", v, floors)), cell.inputs[1])
            nt.links.new(maths(nt, "ADD", maths(nt, "MULTIPLY", seed.outputs[0], 977.0), salt), cell.inputs[2])
            draw = nt.nodes.new("ShaderNodeTexWhiteNoise")
            draw.noise_dimensions = "3D"
            nt.links.new(cell.outputs[0], draw.inputs["Vector"])
            return draw
        zone, tenant, window = draw_cells(12.0, 8.0, 17.0), draw_cells(4.0, 2.0, 43.0), draw_cells(1.0, 1.0, 89.0)
        occupied = maths(nt, "MULTIPLY", maths(nt, "GREATER_THAN", zone.outputs["Value"], 0.12),
                         maths(nt, "LESS_THAN", tenant.outputs["Value"], 0.70))
        chance = maths(nt, "ADD", 0.48, maths(nt, "MULTIPLY", seed.outputs[1], 0.55))
        # About 30--41% lit normally; roughly a tenth of towers have just a few occupants.
        quiet = maths(nt, "MAXIMUM", maths(nt, "LESS_THAN", seed.outputs[0], 0.10), maths(nt, "LESS_THAN", seed.outputs[1], 0.04))
        chance = maths(nt, "MULTIPLY", chance, maths(nt, "SUBTRACT", 1.0, maths(nt, "MULTIPLY", quiet, 0.86)))
        lit = maths(nt, "MULTIPLY", maths(nt, "MULTIPLY", pane, occupied), maths(nt, "LESS_THAN", window.outputs["Value"], chance))
        lit = maths(nt, "MULTIPLY", lit, maths(nt, "GREATER_THAN", seed.outputs[1], 0.001))
        room_rgb, tenant_rgb = nt.nodes.new("ShaderNodeSeparateColor"), nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(window.outputs["Color"], room_rgb.inputs[0])
        nt.links.new(tenant.outputs["Color"], tenant_rgb.inputs[0])
        palette = nt.nodes.new("ShaderNodeValToRGB")
        pr = palette.color_ramp
        pr.interpolation = "CONSTANT"
        pr.elements[0].position, pr.elements[0].color = 0.0, (*lin("#D6A174"), 1)
        pr.elements[1].position, pr.elements[1].color = 0.88, (*lin("#C38BAE"), 1)
        pr.elements.new(0.18).color = (*lin("#E5C5A1"), 1)
        pr.elements.new(0.38).color = (*lin("#C1CDD4"), 1)
        pr.elements.new(0.53).color = (*lin("#95B4D0"), 1)
        pr.elements.new(0.72).color = (*lin("#80B9B0"), 1)
        nt.links.new(seed.outputs[2], palette.inputs["Fac"])
        accent = nt.nodes.new("ShaderNodeMix")
        accent.data_type = "RGBA"
        nt.links.new(maths(nt, "GREATER_THAN", tenant_rgb.outputs[0], 0.5), accent.inputs["Factor"])
        accent.inputs["A"].default_value = (*lin("#C38BAE"), 1)
        accent.inputs["B"].default_value = (*lin("#80B9B0"), 1)
        tenant_colour = nt.nodes.new("ShaderNodeMix")
        tenant_colour.data_type = "RGBA"
        rare = maths(nt, "MAXIMUM", maths(nt, "LESS_THAN", tenant_rgb.outputs[0], 0.05), maths(nt, "GREATER_THAN", tenant_rgb.outputs[0], 0.95))
        nt.links.new(rare, tenant_colour.inputs["Factor"])
        nt.links.new(palette.outputs["Color"], tenant_colour.inputs["A"])
        nt.links.new(accent.outputs["Result"], tenant_colour.inputs["B"])
        colour = nt.nodes.new("ShaderNodeMix")
        colour.data_type = "RGBA"
        colour.inputs["Factor"].default_value = 0.055
        nt.links.new(tenant_colour.outputs["Result"], colour.inputs["A"])
        nt.links.new(window.outputs["Color"], colour.inputs["B"])
        nt.links.new(colour.outputs["Result"], p.inputs["Emission Color"])
        # 70% dim / 25% medium / 5% bright, independent of the occupancy random number.
        levels = maths(nt, "ADD", 0.12, maths(nt, "ADD", maths(nt, "MULTIPLY", maths(nt, "GREATER_THAN", room_rgb.outputs[1], 0.70), 0.22),
                       maths(nt, "MULTIPLY", maths(nt, "GREATER_THAN", room_rgb.outputs[1], 0.95), 0.44)))
        depth = maths(nt, "MINIMUM", 1.0, maths(nt, "MAXIMUM", maths(nt, "DIVIDE", maths(nt, "SUBTRACT", rank, 1.0), 0.559), 0.0))
        transmission = maths(nt, "SUBTRACT", 0.95, maths(nt, "MULTIPLY", depth, 0.60))
        gain = maths(nt, "MULTIPLY", attr.outputs["Alpha"], maths(nt, "ADD", 0.8, maths(nt, "MULTIPLY", tenant_rgb.outputs[1], 0.4)))
        power = maths(nt, "MULTIPLY", maths(nt, "MULTIPLY", levels, gain), transmission)
        nt.links.new(maths(nt, "MULTIPLY", lit, power), p.inputs["Emission Strength"])
        # Shared ground light, not uniform tenant-coloured emission across every facade.
        # The G attribute stores this building's actual height, including the existing setbacks.
        rise = maths(nt, "DIVIDE", pos.outputs["Z"], maths(nt, "MULTIPLY", maths(nt, "MAXIMUM", lod_channels.outputs[1], 40.0), 0.27))
        low = maths(nt, "MULTIPLY", maths(nt, "EXPONENT", maths(nt, "MULTIPLY", maths(nt, "MULTIPLY", rise, rise), -1.0)),
                    maths(nt, "LESS_THAN", maths(nt, "ABSOLUTE", normal.outputs["Z"]), 0.5))
        pool, street_colour = skyline_street_field(nt, geo.outputs["Position"])
        bounce = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(street_colour, bounce.inputs["Color"])
        nt.links.new(maths(nt, "MULTIPLY", maths(nt, "MULTIPLY", low, 0.62), maths(nt, "ADD", 0.15, maths(nt, "MULTIPLY", pool, 0.85))), bounce.inputs["Strength"])
        wall = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(p.outputs[0], wall.inputs[0])
        nt.links.new(bounce.outputs[0], wall.inputs[1])
        roof = skyline_roof_shader(nt)
        p = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(maths(nt, "GREATER_THAN", normal.outputs["Z"], 0.98), p.inputs[0])
        nt.links.new(wall.outputs[0], p.inputs[1])
        nt.links.new(roof.outputs[0], p.inputs[2])
    clear = skyline_hazed(nt, p.outputs[0])
    reveal = nt.nodes.new("ShaderNodeValue")
    reveal.name = "skyline reveal"
    reveal.outputs[0].default_value = 0.0
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.name = "skyline transition"
    nt.links.new(reveal.outputs[0], mix.inputs[0])
    nt.links.new(legacy, mix.inputs[1])
    nt.links.new(clear, mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


def skyline_street(nt):
    """Broad, broken pools along distant streets, added to the existing wet ground only after the climb."""
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    legacy = out.inputs["Surface"].links[0].from_socket
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    pos = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], pos.inputs[0])
    pool, street_colour = skyline_street_field(nt, geo.outputs["Position"])
    # Soft district boundaries; no luminous rectangular edge to the ground patch.
    border = maths(nt, "MULTIPLY", maths(nt, "MINIMUM", 1.0, maths(nt, "MAXIMUM", maths(nt, "DIVIDE", maths(nt, "SUBTRACT", pos.outputs["Y"], 650.0), 120.0), 0.0)),
                   maths(nt, "MINIMUM", 1.0, maths(nt, "MAXIMUM", maths(nt, "DIVIDE", maths(nt, "SUBTRACT", 1620.0, pos.outputs["Y"]), 120.0), 0.0)))
    border = maths(nt, "MULTIPLY", border, maths(nt, "MINIMUM", 1.0, maths(nt, "MAXIMUM", maths(nt, "DIVIDE", maths(nt, "SUBTRACT", 820.0, maths(nt, "ABSOLUTE", pos.outputs["X"])), 120.0), 0.0)))
    reveal = nt.nodes.new("ShaderNodeValue")
    reveal.name = "skyline street reveal"
    reveal.outputs[0].default_value = 0.0
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(street_colour, em.inputs["Color"])
    strength = maths(nt, "MULTIPLY", maths(nt, "MULTIPLY", pool, border), 0.75)
    nt.links.new(maths(nt, "MULTIPLY", strength, reveal.outputs[0]), em.inputs["Strength"])
    add = nt.nodes.new("ShaderNodeAddShader")
    add.name = "skyline street addition"
    nt.links.new(legacy, add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    # Leave the original output connected until skyline_at() explicitly enables the pools.


def skyline_hidden(x, y, w, d, h, blockers):
    """Only remodel silhouettes hidden by an opaque avenue core at the handoff.

    All eight corners must be behind ONE convex core (not merely their centre behind a wall).
    Test both sides of the shutter and adjacent frames. Other silhouettes remain exactly as built.
    """
    corners = np.array([(xx, yy, zz) for xx in (x - w / 2 - 1, x + w / 2 + 1)
                        for yy in (y - d / 2 - 1, y + d / 2 + 1) for zz in (0.0, h + 40)])
    lo, hi = np.array([b[0] for b in blockers]), np.array([b[1] for b in blockers])
    for t in (74 / FPS - 0.003, 75 / FPS, 76 / FPS + 0.003):
        pos = flight(t)[0]
        ray = corners[:, None, :] - pos
        ray = np.where(np.abs(ray) < 1e-8, 1e-8, ray)
        a, b = (lo[None, :, :] - pos) / ray, (hi[None, :, :] - pos) / ray
        enter = np.maximum(np.minimum(a, b).max(axis=2), 0.0)
        leave = np.minimum(np.maximum(a, b).min(axis=2), 0.999)
        if not np.any(np.all(leave > enter, axis=0)):
            return False
    return True


def skyline_tower(bag, neon, x, y, w, d, h, index):
    """Five quiet architectural families inside the original footprint and height budget."""
    rnd = random.Random(9107 + index * 31)  # never consume the avenue/track RNG
    family = rnd.randrange(5)
    # Broad districts with valleys between them, staggered with depth rather than a uniform wall.
    rhythm = 0.5 + 0.5 * math.cos((x + (int(y / 260) % 2) * 115) / 155.0)
    h *= 0.48 + 0.52 * rhythm
    seed = rnd.random()
    occupancy = rnd.choice((0.015, 0.09, 0.18, 0.25, 0.34))
    col = (seed, occupancy, rnd.choice((0.04, 0.22, 0.42, 0.59, 0.77, 0.92)), rnd.uniform(0.55, 1.15))
    dark = (seed, 0.0, col[2], 0.7)
    tone = lin(rnd.choice(("#FFD39A", "#9CD9E9", "#B7C9E6")))
    tip = min(22.0, h * 0.15) if family in (2, 4) else 0.0
    roof = h - tip
    if family == 0:  # broad slab, one deep shoulder and an offset penthouse
        bag.box(x - w / 2, x + w / 2, y - d / 2, y + d / 2, 0, roof * 0.78, col)
        bag.box(x - w * 0.40, x + w * 0.18, y - d * 0.34, y + d * 0.34, roof * 0.78, roof, col)
        cw, cd, cx = w * 0.58, d * 0.68, x - w * 0.11
    elif family in (1, 2):  # three pronounced setbacks, or a chamfered shaft and tapered crown
        levels = ((0, 0.65, 1.0), (0.65, 0.85, 0.72), (0.85, 1.0, 0.46)) if family == 1 else ((0, 0.82, 1.0), (0.82, 1.0, 0.70))
        for a, b, s in levels:
            bag.prism(chamfered(x, y, w * s, d * s, min(w, d) * s * 0.18), roof * a, roof * b, col,
                      0.46 if family == 2 and b == 1.0 else 1.0)
        s = 0.46 if family == 1 else 0.322
        cw, cd, cx = w * s, d * s, x
    elif family == 3:  # two unequal wings: a dark notch instead of a luminous outline
        bag.box(x - w / 2, x + w / 2, y - d / 2, y + d / 2, 0, roof * 0.30, col)
        bag.box(x - w / 2, x - w * 0.06, y - d * 0.42, y + d * 0.42, roof * 0.30, roof, col)
        bag.box(x + w * 0.10, x + w / 2, y - d * 0.30, y + d * 0.46, roof * 0.30, roof * 0.79, col)
        cw, cd, cx = w * 0.44, d * 0.84, x - w * 0.28
    else:  # octagonal business tower, with a separate mechanical crown
        bag.prism(chamfered(x, y, w, d, min(w, d) * 0.28), 0, roof * 0.90, col)
        bag.prism(chamfered(x, y, w * 0.56, d * 0.56, min(w, d) * 0.12), roof * 0.90, roof, dark)
        cw, cd, cx = w * 0.56, d * 0.56, x
    if tip:
        bag.prism(round_loop(cx, y, 2.0, 4), roof, h - 2, dark, 0.30)
        neon.box(cx - 0.85, cx + 0.85, y - 0.85, y + 0.85, h - 2, h, (*lin("#FF3232"), 1.8))
    elif rnd.random() < 0.18:  # a substantial recessed front crown, never all the edges
        neon.box(cx - cw * 0.36, cx + cw * 0.36, y - cd / 2 - 0.12, y - cd / 2, roof - 4.2, roof - 1.4, (*tone, 0.48))
    if family == 0 and rnd.random() < 0.45:
        # One service spine on a few slabs; wide enough to survive the final camera and denoising.
        sx = x + w * 0.22
        neon.box(sx, sx + 1.8, y - d / 2 - 0.12, y - d / 2, roof * 0.22, roof * 0.72, (*tone, 0.28))


def skyline_at(frame):
    """Frames 1--75 use ONLY the original city, including reflections and shadows."""
    active = frame >= 76
    for names, hidden in ((("skyline original", "skyline original lights"), active),
                          (("skyline resolved", "skyline resolved lights"), not active)):
        for name in names:
            o = bpy.data.objects[name]
            if o.hide_render != hidden:
                o.hide_render = hidden
    clarity = ease((frame - 76) / 14)
    for name in ("skyline facade", "skyline glow", "skyline roof"):
        nt = bpy.data.materials[name].node_tree
        mix = nt.nodes["skyline transition"]
        value = nt.nodes["skyline reveal"].outputs[0]
        if value.default_value != clarity:
            value.default_value = clarity
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        # Avoid evaluating the legacy microfacade at all once the transition is complete.
        surface = mix.inputs[2].links[0].from_socket if frame >= 90 else mix.outputs[0]
        if out.inputs["Surface"].links[0].from_socket != surface:
            nt.links.new(surface, out.inputs["Surface"])
    nt = bpy.data.materials["wet street"].node_tree
    add = nt.nodes["skyline street addition"]
    value = nt.nodes["skyline street reveal"].outputs[0]
    if value.default_value != clarity:
        value.default_value = clarity
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    surface = add.outputs[0] if frame > 76 else add.inputs[0].links[0].from_socket
    if out.inputs["Surface"].links[0].from_socket != surface:
        nt.links.new(surface, out.inputs["Surface"])


def build():
    fac, neo = facade_material(), neon_material()
    puls, front = pulse_material()
    signs = [sign_material(i) for i in range(SIGNS)]
    bag, neon, pulse = Bag(True), Bag(), Bag()
    foot = []
    walls = {-1: [], 1: []}
    blockers = []
    roof_ranges = []  # Only building roofs, not bridges, signs, the drone or the landmark.
    sign_no = 0
    # the avenue: unbroken walls on both sides, climbing with distance
    for side in (-1, 1):
        y = -440.0
        while y < 430:
            d = RNG.uniform(30, 54)
            w = RNG.uniform(26, 46)
            h = RNG.uniform(95, 175) + (RNG.random() < 0.18) * RNG.uniform(40, 90) if y < 10 else RNG.uniform(62, 112)
            base = h * RNG.uniform(0.7, 0.88)                # the first setback: nothing hangs on the wall above it
            x = side * (AVENUE + w / 2)
            first_face = len(bag.faces)
            tower(bag, neon, x, y + d / 2, w, d, h, lights=y > -80, first=base)
            roof_ranges.append((first_face, len(bag.faces)))
            # An inset core lies inside even the deepest chamfer; safe for the skyline handoff test.
            inset = min(w, d) * 0.31
            blockers.append(((x - w / 2 + inset, y + inset, 0.0),
                             (x + w / 2 - inset, y + d - inset, base)))
            dress(bag, neon, side, y, y + d, base)
            walls[side].append((y, y + d, base))
            foot.append((x, y + d / 2, w, d, h))
            for k in range(RNG.randint(1, 3)):              # signs that stand out into the avenue
                depth = RNG.uniform(5.0, 8.0)
                height = depth * 2.55
                ceil = min(base - height - 1, 170)
                if ceil > 14:
                    blade(bag, sign_no, side, y + RNG.uniform(3, d - 3), RNG.uniform(min(38, ceil), ceil), depth, height, signs)
                    sign_no += 1
            bw = RNG.uniform(18, 30)
            bh = bw * 0.6
            if RNG.random() < 0.6 and base - bh > 38:       # and billboards flat on the wall, seen at a slant
                yy, zz = y + RNG.uniform(2, max(3, d - bw)), RNG.uniform(36, min(110, base - bh - 1))
                xx = side * (AVENUE - 0.3)
                n = SIGNS + RNG.randrange(6)
                bag.box(xx - 0.3, xx + 0.3, yy - 0.25, yy + bw + 0.25, zz - 0.25, zz + bh + 0.25, (0.6, 0.0, 0.0, 1.0))
                xx = side * (AVENUE - 0.65)
                plane(f"wall{sign_no}", [(xx, yy, zz), (xx, yy + bw, zz), (xx, yy + bw, zz + bh), (xx, yy, zz + bh)] if side < 0 else
                      [(xx, yy + bw, zz), (xx, yy, zz), (xx, yy, zz + bh), (xx, yy + bw, zz + bh)], BILLS[n - SIGNS])
                sign_no += 1
            gap = RNG.uniform(7, 14)
            neon.box(side * (AVENUE + 26), side * (AVENUE + 26.5), y + d, y + d + gap, 0, RNG.uniform(14, 34), (*RNG.choice(SHOPS), 0.35))  # the glow at the end of an alley
            y += d + gap
    # overhead: what you fly under and over. A corridor needs a wall at both ends, and above the climb it stays clear
    # of the camera and the drone; below us only a few, so that the street shows.
    def anchored(y, w):
        tops = [next((b for a0, a1, b in walls[side] if a0 + 1 < y and y + w < a1 - 1), None) for side in (-1, 1)]
        return None if None in tops else min(tops)
    y = -238.0
    while y < 430:
        w, h = RNG.uniform(5.5, 8), RNG.uniform(3.6, 4.8)
        top = anchored(y, w)
        if top is None:
            y += 5
            continue
        if y < -125 and RNG.random() < 0.8 and top - h - 1 > 88:
            skybridge(bag, neon, y, RNG.uniform(88, min(150, top - h - 1)), w, h)
        elif RNG.random() < 0.42:
            skybridge(bag, neon, y, RNG.uniform(15, 25), w, h)
        y += RNG.uniform(34, 76)
    monorail(bag, neon)
    wires(bag, -300, 430, 46)
    # the rest of the city: keep the exact legacy RNG calls and footprints for write_track.
    # Only the layer wholly behind the landmark gets new materials/architecture.
    old_city, old_lights, city, city_lights = Bag(True), Bag(), Bag(True), Bag()
    skyline_count, remodelled = 0, 0
    city_ranks = []
    tx, ty = TARGET
    for _ in range(760):
        x, y = RNG.uniform(-760, 760), RNG.uniform(-520, 1500)
        w, d = RNG.uniform(22, 58), RNG.uniform(22, 58)
        if abs(x) < AVENUE + 54 + w / 2 and y < 440:
            continue
        away = math.hypot(x - tx, y - ty)
        if away < 104:
            continue
        cone = abs(x - (tx * (y + 100) / 660)) < 70 + max(0, y + 100) * 0.42 and -100 < y < ty + 80
        cap = (64 + away * 0.13) if cone else 86 + away * 0.3
        h = min(cap, RNG.uniform(38, 250))
        if y - d / 2 > ty + 80:
            skyline_count += 1
            original, lights = Bag(True), Bag()
            tower(original, lights, x, y, w, d, h)
            old_city.extend(original)
            old_lights.extend(lights)
            first_corner, first_vertex = len(city.col), len(city.verts)
            if skyline_hidden(x, y, w, d, h, blockers):
                remodelled += 1
                skyline_tower(city, city_lights, x, y, w, d, h, len(foot))
            else:
                city.extend(original)
                city_lights.extend(lights)
            scale = 1.0 + max(0.0, y - 640.0) * 0.00065
            height = max(v[2] for v in city.verts[first_vertex:])
            city_ranks.extend([(scale, height, 0.0, 1.0)] * (len(city.col) - first_corner))
        else:
            first_face = len(bag.faces)
            tower(bag, neon, x, y, w, d, h)
            roof_ranges.append((first_face, len(bag.faces)))
        foot.append((x, y, w, d, h))
    print(f"skyline: {skyline_count} coarse facades, {remodelled} safely hidden silhouette replacements", flush=True)
    old_city.mesh("skyline original", "facade", fac)
    old_lights.mesh("skyline original lights", "glow", neo)
    resolved = city.mesh("skyline resolved", "facade", skyline_material(fac))
    resolved.data.color_attributes.new("skyline rank", "FLOAT_COLOR", "CORNER").data.foreach_set("color", np.array(city_ranks, np.float32).ravel())
    resolved.hide_render = True
    city_lights.mesh("skyline resolved lights", "glow", skyline_material(neo, True)).hide_render = True
    top = landmark(bag, pulse)
    foot.append((tx, ty, 124, 124, top))
    structures = bag.mesh("towers", "facade", fac)
    structures.data.materials.append(skyline_material(fac, roof_only=True))
    roof_faces = {i for a, b in roof_ranges for i in range(a, b)}
    for face in structures.data.polygons:
        if face.index in roof_faces and face.normal.z > 0.98 and face.center.z > 45.0:
            face.material_index = 1
    neon.mesh("neon", "glow", neo)
    pulse.mesh("pulse", "glow", puls)
    ground = bpy.data.materials.new("wet street")
    ground.use_nodes = True
    nt = ground.node_tree
    p = nt.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*lin("#07090D"), 1)
    p.inputs["Metallic"].default_value = 0.2
    pools = nt.nodes.new("ShaderNodeTexNoise")
    pools.inputs["Scale"].default_value = 0.09
    pools.inputs["Detail"].default_value = 6
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["From Min"].default_value, rough.inputs["From Max"].default_value = 0.42, 0.58
    rough.inputs["To Min"].default_value, rough.inputs["To Max"].default_value = 0.03, 0.34
    nt.links.new(pools.outputs["Fac"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], p.inputs["Roughness"])
    hazed(nt, p.outputs[0])
    skyline_street(nt)
    plane("street", [(-3000, -3000, 0), (3000, -3000, 0), (3000, 3000, 0), (-3000, 3000, 0)], ground)
    return foot, top, front


BILLS = []


class Traffic:
    """Streams of lights: tail-lights flying with us, headlights coming the other way. One mesh each, moved every frame."""

    def __init__(self, name, colour, strength, lanes, count, speed):
        self.speed = speed
        rnd = random.Random(len(name) * 31 + count)
        self.lanes = np.array([rnd.choice(lanes) for _ in range(count)], float)
        self.ys = np.array([rnd.uniform(-460, 800) for _ in range(count)])
        verts, faces = [], []
        for _ in range(count):
            for sx in (-1, 1):
                k = len(verts)
                cx = sx * 0.7
                verts += [(cx - 0.16, -1.5, -0.1), (cx + 0.16, -1.5, -0.1), (cx + 0.16, 1.5, -0.1), (cx - 0.16, 1.5, -0.1),
                          (cx - 0.16, -1.5, 0.1), (cx + 0.16, -1.5, 0.1), (cx + 0.16, 1.5, 0.1), (cx - 0.16, 1.5, 0.1)]
                faces += [(k, k + 1, k + 5, k + 4), (k + 1, k + 2, k + 6, k + 5), (k + 2, k + 3, k + 7, k + 6),
                          (k + 3, k, k + 4, k + 7), (k + 4, k + 5, k + 6, k + 7)]
        self.local = np.array(verts, np.float32)
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        me.materials.append(light_material(name, colour, strength))
        self.me = me
        link(bpy.data.objects.new(name, me))

    def at(self, t):
        ys = (self.ys + self.speed * t + 460) % 1260 - 460
        offs = np.empty_like(self.local)
        per = len(self.local) // len(self.ys)
        offs[:, 0] = np.repeat(self.lanes[:, 0], per)
        offs[:, 1] = np.repeat(ys, per)
        offs[:, 2] = np.repeat(self.lanes[:, 1], per)
        self.me.vertices.foreach_set("co", (self.local + offs).ravel())
        self.me.update()


def write_track(foot, top, sign_box):
    os.makedirs(CACHE, exist_ok=True)
    cams = []
    for f in range(N):
        t = f / FPS
        pos, aim, roll, lens = flight(t)
        R = camera_matrix(pos, aim, roll)
        ahead = flight(t + 0.05)[0] - flight(t - 0.05)[0]
        drone = drone_at(t)
        cams.append(dict(c=pos.tolist(), r=R.ravel().tolist(), lens=lens, v=(ahead / 0.1).tolist(), drone=drone.tolist()))
    tx, ty = TARGET
    with open(os.path.join(CACHE, "track.json"), "w") as fh:
        json.dump(dict(cam=cams, foot=[list(map(float, x)) for x in foot if abs(x[0]) < 520],
                       target=[tx - 13, tx + 13, ty - 13, ty + 13, top - 88, top], sign=sign_box), fh)


def cargo_drone():
    """A big cargo quadcopter: pod, arms, four ducted rotors, a crate slung underneath, running lights."""
    hull, lights = Bag(True), Bag()
    dark = (0.1, 0.0, 0.0, 1.0)
    hull.prism(chamfered(0, 0, 4.4, 9.5, 1.2), -0.9, 0.9, dark, 0.72, True, True)
    hull.prism(chamfered(0, 0, 4.4, 9.5, 1.2), -1.8, -0.9, dark, 1.0, False)
    for x, y in ((-6.6, -6.2), (6.6, -6.2), (-6.6, 6.2), (6.6, 6.2)):
        a, b = np.array((x, y, 0.3)), np.array((x * 0.22, y * 0.3, 0.1))
        hull.tube([tuple(a), tuple(b)], 0.32, dark)
        hull.tube([tuple(a + (0, 0, -0.5)), tuple(b + (0, 0, -0.6))], 0.14, dark)
        for i in range(24):                                          # the duct
            a0, a1 = 2 * math.pi * i / 24, 2 * math.pi * (i + 1) / 24
            hull.prism([(x + 3.0 * math.cos(a0), y + 3.0 * math.sin(a0)), (x + 3.0 * math.cos(a1), y + 3.0 * math.sin(a1)),
                        (x + 3.4 * math.cos(a1), y + 3.4 * math.sin(a1)), (x + 3.4 * math.cos(a0), y + 3.4 * math.sin(a0))], -0.45, 0.7, dark, 1.0, True, True)
            lights.prism([(x + 3.18 * math.cos(a0), y + 3.18 * math.sin(a0)), (x + 3.18 * math.cos(a1), y + 3.18 * math.sin(a1)),
                          (x + 3.25 * math.cos(a1), y + 3.25 * math.sin(a1)), (x + 3.25 * math.cos(a0), y + 3.25 * math.sin(a0))], -0.52, -0.46, (*lin("#19F0FF"), 0.18), 1.0, False, True)
        hull.prism(round_loop(x, y, 0.45), -0.5, 0.75, dark)
        lights.tube([tuple(a + (0, 0, -0.34)), tuple(b + (0, 0, -0.34))], 0.05, (*lin("#EFFCFF"), 0.5))
        lights.prism(round_loop(x, y, 0.16), -0.56, -0.5, (*lin("#EFFCFF"), 1.6), 1.0, False, True)
    hull.box(-2.0, 2.0, -3.6, 3.6, -5.2, -2.6, (0.72, 0.0, 0.0, 1.0))  # the crate
    for k in range(5):                                               # hazard stripes on its belly
        lights.box(-1.85, 1.85, -3.3 + k * 1.45, -2.65 + k * 1.45, -5.24, -5.2, (*lin("#FFB000"), 1.0))
    lights.box(-1.5, 1.5, -2.2, 2.2, -1.86, -1.82, (*lin("#19F0FF"), 0.5))
    hull.box(-0.2, 0.2, -2.8, -2.4, -2.6, -1.8, dark)
    hull.box(-0.2, 0.2, 2.4, 2.8, -2.6, -1.8, dark)
    lights.box(-2.03, 2.03, -3.63, 3.63, -4.0, -3.8, (*lin("#FFB000"), 1.3))
    lights.box(-1.6, 1.6, -4.76, -4.72, -0.5, -0.25, (*lin("#EFFCFF"), 2.6))     # headlamp bar
    lights.box(-10.0, -9.8, -6.3, -6.1, 0.0, 0.3, (*lin("#FF2020"), 2.6))
    lights.box(9.8, 10.0, -6.3, -6.1, 0.0, 0.3, (*lin("#20FF60"), 2.6))
    lights.box(-1.3, 1.3, 4.5, 4.72, -0.4, -0.2, (*lin("#FF2020"), 2.0))
    parts = [hull.mesh("drone", "facade", bpy.data.materials["facade"]), lights.mesh("drone lights", "glow", bpy.data.materials["neon"])]
    disc = bpy.data.materials.new("rotor")
    disc.use_nodes = True
    nt = disc.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    wheel = nt.nodes.new("ShaderNodeTexGradient")          # the blur of three blades: denser towards their tips
    wheel.gradient_type = "RADIAL"
    nt.links.new(tc.outputs["Object"], wheel.inputs["Vector"])
    nt.links.new(maths(nt, "ADD", 0.16, maths(nt, "MULTIPLY", 0.14, maths(nt, "SINE", maths(nt, "MULTIPLY", wheel.outputs["Fac"], 6 * math.pi * 3)))), mix.inputs[0])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    gl = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl.inputs["Roughness"].default_value = 0.35
    gl.inputs["Color"].default_value = (0.32, 0.36, 0.42, 1)
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(gl.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    verts, faces = [], []
    for x, y in ((-6.6, -6.2), (6.6, -6.2), (-6.6, 6.2), (6.6, 6.2)):
        k = len(verts)
        verts += [(x + 3.0 * math.cos(2 * math.pi * i / 24), y + 3.0 * math.sin(2 * math.pi * i / 24), 0.25) for i in range(24)]
        faces.append(tuple(k + i for i in range(24)))
    me = bpy.data.meshes.new("rotors")
    me.from_pydata(verts, [], faces)
    me.materials.append(disc)
    parts.append(link(bpy.data.objects.new("rotors", me)))
    return parts


def main():
    sc = setup()
    for i in range(6):
        BILLS.append(sign_material(SIGNS + i))
    foot, top, front = build()
    # the first thing in frame: a tall blade close by on the right
    hero = (15.5, 22.6, -271.0, -271.0, 30.0, 66.0)
    plane("hero", [(hero[0], hero[2], hero[4]), (hero[1], hero[2], hero[4]), (hero[1], hero[2], hero[5]), (hero[0], hero[2], hero[5])],
          bpy.data.materials[f"sign{opt('hero', 3)}"])
    write_track(foot, top, list(hero))
    if "--track" in ARGS:
        return

    drone = cargo_drone()
    traffic = [Traffic("tail", lin("#FF1830"), 12.0, [(-16, 22), (-10, 22), (10, 26), (16, 26), (-19, 66), (19, 72)], 170, 62.0),
               Traffic("head", lin("#FFF0D8"), 14.0, [(-5, 14), (5, 14), (-19, 40), (19, 38), (-8, 18), (8, 18)], 170, -74.0)]
    trains = [Train("train a", 13.5, -380.0, 34.0, lin("#FF2A6D")), Train("train b", -13.5, -30.0, -46.0, lin("#19F0FF"))]
    beams = [Beam(-250, 760, 60, lin("#7FD8FF"), 0.2, 0.42, 0.3, 0.42), Beam(300, 900, 60, lin("#FF7AC8"), 0.14, 0.38, 2.4, -0.5),
             Beam(90, 1150, 60, lin("#7FD8FF"), 0.16, 0.3, 4.2, 0.36)]
    spot = Beam(0, 0, 0, lin("#DDF3FF"), 0.2, 0.0, 0.0, 0.0, L=56.0, r=7.5)

    cd = bpy.data.cameras.new("drone")
    cd.sensor_width = 36
    cd.clip_start, cd.clip_end = 0.5, 6000
    cam = link(bpy.data.objects.new("drone", cd))
    cam.rotation_mode = "QUATERNION"
    sc.camera = cam
    for f in range(0, N + 2):                              # keyed so that Cycles has a shutter's worth of camera motion
        pos, aim, roll, lens = flight((f - 1) / FPS)
        m = Matrix(camera_matrix(pos, aim, roll).tolist())
        cam.location = pos.tolist()
        cam.rotation_quaternion = m.to_quaternion()
        cd.lens = lens
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        cd.keyframe_insert("lens", frame=f)

    test = [int(f) for f in opt("frames", "").split(",") if f]
    if test:
        sc.render.resolution_percentage = opt("pct", 40)
        sc.cycles.samples = opt("samples", 64)
        out = os.path.join(FRAMES, "test")
    else:
        out = os.path.join(FRAMES, opt("out", "cyberhud"))
    os.makedirs(out, exist_ok=True)
    for f in test or range(1, N + 1):
        path = os.path.join(out, f"cyberhud_f{f:04d}.png" if test else f"f{f:04d}.png")
        if not test and os.path.exists(path):
            continue
        t0 = time.time()
        t = (f - 1) / FPS
        sc.frame_set(f)
        skyline_at(f)
        for s in (*traffic, *trains, *beams):
            s.at(t)
        at = drone_at(t).tolist()
        for o in drone:
            o.location = at
        spot.o.location = (at[0], at[1] - 1.0, at[2] - 5.2)
        spot.o.rotation_euler = (math.pi + 0.28 * math.sin(1.9 * t), 0.2 * math.cos(1.4 * t), 0)
        front.outputs[0].default_value = -60 + 420 * seg(t, 3.6, 4.5)
        for n in HAZE_NODES:
            n.inputs[1].default_value = -1.0 / (HAZE + 2400.0 * seg(t, 2.3, 3.5))
        bpy.context.view_layer.update()
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f"cyberhud f{f:04d}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
