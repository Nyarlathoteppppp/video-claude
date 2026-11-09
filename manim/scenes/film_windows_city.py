"""九 千窗: the city. Ordinary blocks of flats at night, their windows lit, up to the cut.

    Blender -b --factory-startup -P film_windows_city.py -- C1  [--pct=100] [--samples=128]
    Blender -b --factory-startup -P film_windows_city.py -- C3a [--pct=100] [--samples=128]

The other side of the cut is film_windows_grid.py.

Plates go under ink/film_out/film/windows/plates: NAME_room (the street, denoised, the windows only lighting it) and NAME_words (what is in the windows, as sharp as it is). film_windows_ui.py adds them. x along the street, y away from the lens, z up, in metres. A block's face looks along -y.
"""
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

import film_windows3d as w3
from film_windows3d import Build
from ink3d import ROOT, aim, reset

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


OUT = os.path.join(ROOT, "ink", "film_out", "film", "windows")
TEX = os.path.join(OUT, "tex")
PLATES = os.path.join(OUT, "plates")
PICS = os.path.join(OUT, "pics")

BAY, STOREY, WIN_W, WIN_H, SILL = 3.4, 2.9, 2.24, 1.4, .8       # a window is 1.6 by 1, as a Claude window is
MARGIN = (BAY - WIN_W) / 2
HAZE = (.066, .038, .019)           # what the air over a city does at night: a warm brown, from its own lights, and nothing like neon
AIR = (.30, .20, .10)
HALO = 11.0                          # how much the light out of a window is worth on the wall around it
PICTURES = tuple(sorted(f[:-4] for f in os.listdir(PICS) if len(f) == 7 and f.startswith("p") and f.endswith(".png"))) if os.path.isdir(PICS) else ()
# more of the rooms you can see into than of the ones with only a screen on
LIVELY = PICTURES[:12] * 3 + PICTURES[12:18] * 2 + PICTURES[18:24] * 3 + PICTURES[24:30] * 2 + PICTURES[30:]
TILE_COLS, TILE_FLOORS = 30, 20     # how much building one wall picture covers


# ---------------------------------------------------------------- materials

def surface(name, colour, rough=.6, metal=0.0, spec=.4):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*colour, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Specular IOR Level"].default_value = spec
    return m


def pictured(name, path, rough=.8):
    m = surface(name, (1, 1, 1), rough)
    nt = m.node_tree
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(path, check_existing=True)
    nt.links.new(nt.nodes.new("ShaderNodeUVMap").outputs[0], t.inputs[0])
    nt.links.new(t.outputs[0], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return m


def walled_material(name):
    """Its picture, and the picture's lines pressed into it, so that a low light finds the tiles."""
    m = pictured(name, os.path.join(TEX, name + ".png"), .82)
    nt = m.node_tree
    t = next(n for n in nt.nodes if n.type == "TEX_IMAGE")
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .55
    bump.inputs["Distance"].default_value = .012
    nt.links.new(t.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], nt.nodes["Principled BSDF"].inputs["Normal"])
    return m


def dark_glass():
    """An unlit pane: a very little of the sky in it, going off towards one corner, and more in some panes than in others. No picture of anything."""
    m = surface("glass", (.006, .006, .006), .2, 0, .3)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    uv, sep, tilt, geo, n, times, power = nt.nodes.new("ShaderNodeUVMap"), nt.nodes.new("ShaderNodeSeparateXYZ"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeNewGeometry"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeMath")
    nt.links.new(uv.outputs[0], sep.inputs[0])
    tilt.operation = "MULTIPLY_ADD"
    nt.links.new(sep.outputs["Y"], tilt.inputs[0])
    tilt.inputs[1].default_value = .8
    tilt.inputs[2].default_value = .05
    n.inputs["Scale"].default_value = .22
    n.inputs["Detail"].default_value = 1
    nt.links.new(geo.outputs["Position"], n.inputs["Vector"])
    times.operation = "MULTIPLY"
    nt.links.new(tilt.outputs[0], times.inputs[0])
    nt.links.new(n.outputs[0], times.inputs[1])
    power.operation = "MULTIPLY"
    nt.links.new(times.outputs[0], power.inputs[0])
    power.inputs[1].default_value = .03
    b.inputs["Emission Color"].default_value = (1.0, .66, .40, 1)
    nt.links.new(power.outputs[0], b.inputs["Emission Strength"])
    return m


def drawn_curtain():
    """Curtains drawn in a dark room: pleats, and only as much light on them as gets in."""
    m = surface("drawn", (.03, .024, .019), .8, 0, .2)
    nt = m.node_tree
    uv, sep, wave, up, bump = nt.nodes.new("ShaderNodeUVMap"), nt.nodes.new("ShaderNodeSeparateXYZ"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeBump")
    nt.links.new(uv.outputs[0], sep.inputs[0])
    up.operation = "MULTIPLY"
    nt.links.new(sep.outputs["X"], up.inputs[0])
    up.inputs[1].default_value = 90.0
    wave.operation = "SINE"
    nt.links.new(up.outputs[0], wave.inputs[0])
    bump.inputs["Strength"].default_value = .8
    bump.inputs["Distance"].default_value = .02
    nt.links.new(wave.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], nt.nodes["Principled BSDF"].inputs["Normal"])
    return m


def halo_material(colour, power):
    m = bpy.data.materials.new("halo")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e, o = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    e.inputs[0].default_value = (*colour, 1)
    e.inputs[1].default_value = power
    nt.links.new(e.outputs[0], o.inputs[0])
    return m


def shy(o):
    """It lights what is around it and nothing else ever sees it."""
    o.visible_camera = o.visible_glossy = o.visible_transmission = o.visible_volume_scatter = o.visible_shadow = False
    o["halo"] = 1
    return o


def lit_window(name, path, gain, spill=1.0, cubic=False):
    """A window seen from outside. In the first plate it only lights the street; the second shows it."""
    m = bpy.data.materials.new("lit_" + name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    t, e, o = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    t.image = bpy.data.images.load(path, check_existing=True)
    t.extension = "EXTEND"
    if cubic:
        t.interpolation = "Cubic"
    nt.links.new(nt.nodes.new("ShaderNodeUVMap").outputs[0], t.inputs[0])
    nt.links.new(t.outputs[0], e.inputs[0])
    path_, off, mul = nt.nodes.new("ShaderNodeLightPath"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeMath")
    off.operation, mul.operation = "SUBTRACT", "MULTIPLY"
    off.inputs[0].default_value = 1
    nt.links.new(path_.outputs["Is Camera Ray"], off.inputs[1])
    nt.links.new(off.outputs[0], mul.inputs[0])
    mul.inputs[1].default_value = gain * spill
    nt.links.new(mul.outputs[0], e.inputs[1])
    nt.links.new(e.outputs[0], o.inputs[0])
    m["words"] = gain
    return m


# ---------------------------------------------------------------- pieces

def cuboid(b, T, x0, x1, y0, y1, z0, z1):
    b.add([T(x0, y0, z0), T(x1, y0, z0), T(x1, y1, z0), T(x0, y1, z0), T(x0, y0, z1), T(x1, y0, z1), T(x1, y1, z1), T(x0, y1, z1)],
          [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])


def walled(b, T, x0, x1, z0, z1, cols, floors):
    """A piece of the wall's face, its picture laid on it by where it is."""
    W, H = TILE_COLS * BAY, TILE_FLOORS * STOREY
    uv = [(x0 / W, z0 / H), (x1 / W, z0 / H), (x1 / W, z1 / H), (x0 / W, z1 / H)]
    b.add([T(x0, 0, z0), T(x1, 0, z0), T(x1, 0, z1), T(x0, 0, z1)], [(0, 1, 2, 3)], uv)


def placed(ox, oy, oz=0.0, turn=0.0):
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    return lambda x, y, z: (ox + c * x - s * y, oy + s * x + c * y, oz + z)


class Street:
    """Everything built so far, gathered by what it is made of."""

    def __init__(self, reveal=.12, walls=("brick_a", "brick_b")):
        self.reveal = reveal        # how far the glass sits back in the wall
        self.walls = {w: Build() for w in walls}
        self.parts = {k: Build() for k in ("trim", "reveal", "frame", "glass", "drawn", "rail", "steel", "ac", "pipe", "roof", "gable")}
        self.front = Build()
        self.lit = {}
        self.halo = {}
        with open(os.path.join(PICS, "means.json")) as f:
            self.means = json.load(f)

    def done(self, gain, spill=1.0):
        for w, b in self.walls.items():
            if b.v:
                b.object(w, walled_material(w))
        mats = dict(
            trim=surface("trim", (.40, .37, .32), .85), reveal=surface("reveal", (.24, .22, .19), .85), frame=surface("frame", (.14, .135, .125), .42, .6),
            glass=dark_glass(), drawn=drawn_curtain(), rail=surface("rail", (.045, .045, .042), .45, .7),
            steel=surface("steel", (.12, .115, .105), .6, .5), ac=surface("case", (.50, .48, .43), .55), pipe=surface("pipe", (.30, .28, .25), .6), roof=surface("roofing", (.08, .075, .07), .9),
            gable=surface("gable", (.17, .16, .14), .9))
        for k, b in self.parts.items():
            if b.v:
                b.object(k, mats[k])
        if self.front.v:
            self.front.object("ac_front", pictured("ac_front", os.path.join(TEX, "ac.png"), .55))
        for name, b in self.lit.items():
            b.object("lit_" + name, lit_window(name, os.path.join(PICS, name + ".png"), gain, spill))
        for name, b in self.halo.items():
            r, g, b_ = self.means[name]
            level = max(r, g, b_, 1e-4)
            shy(b.object("halo_" + name, halo_material((1.0, .5 + .5 * g / level, .3 + .7 * b_ / level), HALO * (.2126 * r + .7152 * g + .0722 * b_))))

    def window(self, T, pic, x0, x1, z0, z1):
        b = self.lit.setdefault(pic, Build())
        r = self.reveal
        b.add([T(x0, r, z0), T(x1, r, z0), T(x1, r, z1), T(x0, r, z1)], [(0, 1, 2, 3)], [(0, 0), (1, 0), (1, 1), (0, 1)])
        if not pic.endswith("_m80"):    # the light from it, on the wall, the sill and whatever hangs under it
            y = -.62
            self.halo.setdefault(pic.split("_m")[0], Build()).add([T(x0 + .3, y, z0 + .15), T(x1 - .3, y, z0 + .15), T(x1 - .3, y, z1 - .15), T(x0 + .3, y, z1 - .15)], [(0, 1, 2, 3)])


def plan(cols, floors, seed, share, near=()):
    """Which windows are on, and whose room each one is. near: (col, floor, picture) of the windows chosen by hand."""
    rng = random.Random(seed)
    chosen = {(c, f): p for c, f, p in near}
    out = {}
    for f in range(floors):
        for c in range(cols):
            if (c, f) in chosen:
                out[c, f] = chosen[c, f]
            elif rng.random() < share:
                seen = {out.get((c - 1, f)), out.get((c, f - 1))}
                pic = rng.choice(LIVELY)
                while pic in seen:
                    pic = rng.choice(LIVELY)
                out[c, f] = pic
    return out


def block(st, T, cols, floors, on, seed, wall, detail=2, deep=True, shrink=None):
    """One block of flats. detail 0: a wall and lit windows, for what is a long way off. 1: frames, sills, slabs, a roofline. 2: air conditioners, balconies, bars, drainpipes, for what you can see. deep: real balconies; seen along the wall they would hide the windows beyond them, so a wall seen that way has railings flat across its windows. shrink: which ready-shrunk picture a window at (col, floor) wants."""
    rng = random.Random(seed)
    REVEAL = st.reveal
    W, H = cols * BAY, floors * STOREY
    b = st.walls[wall]
    for f in range(floors):
        z0 = f * STOREY
        walled(b, T, 0, W, z0, z0 + SILL, cols, floors)
        walled(b, T, 0, W, z0 + SILL + WIN_H, z0 + STOREY, cols, floors)
        walled(b, T, 0, MARGIN, z0 + SILL, z0 + SILL + WIN_H, cols, floors)
        walled(b, T, W - MARGIN, W, z0 + SILL, z0 + SILL + WIN_H, cols, floors)
        for c in range(cols - 1):
            x = c * BAY + MARGIN + WIN_W
            walled(b, T, x, x + 2 * MARGIN, z0 + SILL, z0 + SILL + WIN_H, cols, floors)
    # the building's end and its top
    p = st.parts
    p["gable"].add([T(W, 0, 0), T(W, 14, 0), T(W, 14, H), T(W, 0, H)], [(0, 1, 2, 3)])
    p["gable"].add([T(0, 14, 0), T(0, 0, 0), T(0, 0, H), T(0, 14, H)], [(0, 1, 2, 3)])
    p["roof"].add([T(0, 0, H), T(W, 0, H), T(W, 14, H), T(0, 14, H)], [(0, 1, 2, 3)])
    # which stacks of windows have balconies: it goes in a pattern, as it does in a real block
    pattern = rng.choice(((0, 1, 0, 0, 1, 0), (0, 0, 1, 0, 0, 1), (1, 0, 0, 1, 0, 0, 0)))
    for f in range(floors):
        z0 = f * STOREY
        for c in range(cols):
            x0, x1, s, h = c * BAY + MARGIN, c * BAY + MARGIN + WIN_W, z0 + SILL, z0 + SILL + WIN_H
            if (c, f) in on:
                st.window(T, on[c, f] + (shrink(c, f) if shrink else "_m80" if detail == 0 else ""), x0, x1, s, h)
            else:
                p["glass" if rng.random() < .72 else "drawn"].add([T(x0, REVEAL, s), T(x1, REVEAL, s), T(x1, REVEAL, h), T(x0, REVEAL, h)], [(0, 1, 2, 3)], [(0, 0), (1, 0), (1, 1), (0, 1)])
            if detail == 0:
                continue
            for part in ((x0, x0, 0, REVEAL, s, h), (x1, x1, REVEAL, 0, s, h)):
                p["reveal"].add([T(part[0], part[2], s), T(part[0], part[3], s), T(part[0], part[3], h), T(part[0], part[2], h)], [(0, 1, 2, 3)])
            p["reveal"].add([T(x0, 0, h), T(x1, 0, h), T(x1, REVEAL, h), T(x0, REVEAL, h)], [(0, 1, 2, 3)])
            cuboid(p["trim"], T, x0 - .05, x1 + .05, -.05, REVEAL, s - .05, s)
            # the aluminium frame and the meeting rail, in front of the picture
            y0, y1 = REVEAL - .07, REVEAL - .005
            cuboid(p["frame"], T, x0, x0 + .05, y0, y1, s, h)
            cuboid(p["frame"], T, x1 - .05, x1, y0, y1, s, h)
            cuboid(p["frame"], T, x0, x1, y0, y1, s, s + .05)
            cuboid(p["frame"], T, x0, x1, y0, y1, h - .05, h)
            m = (x0 + x1) / 2
            cuboid(p["frame"], T, m - .035, m + .035, y0, y1, s, h)
            if detail < 2:
                continue
            stack = pattern[c % len(pattern)]
            if stack and f > 0 and not deep:
                n = int(WIN_W / .12)
                for i in range(n + 1):
                    x = x0 - .03 + i * (WIN_W + .06) / n
                    cuboid(p["rail"], T, x - .007, x + .007, -.10, -.086, s - .05, s + .34)
                cuboid(p["rail"], T, x0 - .05, x1 + .05, -.11, -.075, s + .33, s + .36)
                cuboid(p["rail"], T, x0 - .05, x1 + .05, -.10, -.08, s + .02, s + .04)
            elif stack and f > 0:
                self_x0, self_x1 = c * BAY + .12, (c + 1) * BAY - .12
                cuboid(p["trim"], T, self_x0, self_x1, -1.0, 0, z0 - .12, z0)
                n = int((self_x1 - self_x0) / .12)
                for i in range(n + 1):
                    x = self_x0 + i * (self_x1 - self_x0) / n
                    cuboid(p["rail"], T, x - .008, x + .008, -.98, -.964, z0, z0 + 1.1)
                for i in range(1, 8):
                    y = -i * .12
                    for x in (self_x0, self_x1):
                        cuboid(p["rail"], T, x - .008, x + .008, y - .008, y + .008, z0, z0 + 1.1)
                cuboid(p["rail"], T, self_x0 - .02, self_x1 + .02, -1.0, -.94, z0 + 1.08, z0 + 1.12)
                cuboid(p["rail"], T, self_x0 - .02, self_x1 + .02, -.985, -.96, z0 + .08, z0 + .1)
                for x in (self_x0, self_x1):
                    cuboid(p["rail"], T, x - .02, x + .02, -1.0, 0, z0 + 1.08, z0 + 1.12)
            else:
                # an air conditioner on its brackets, under the sill or beside it, not every window
                if rng.random() < .62:
                    w_, h_, d_ = .82, .55, .30
                    ax = x0 + rng.uniform(.0, WIN_W - w_)
                    az = z0 + rng.uniform(.06, .14)
                    cuboid(p["ac"], T, ax, ax + w_, -d_ - .06, -.06, az, az + h_)
                    st.front.add([T(ax, -d_ - .0605, az), T(ax + w_, -d_ - .0605, az), T(ax + w_, -d_ - .0605, az + h_), T(ax, -d_ - .0605, az + h_)], [(0, 1, 2, 3)], [(0, 0), (1, 0), (1, 1), (0, 1)])
                    for bx in (ax + .1, ax + w_ - .13):
                        cuboid(p["steel"], T, bx, bx + .03, -d_ - .1, 0, az - .03, az)
                        cuboid(p["steel"], T, bx, bx + .03, -.03, 0, az - .3, az)
                    cuboid(p["pipe"], T, ax + w_ - .03, ax + w_ + .02, -.06, -.02, az + .1, s - .03)
                if f < 3 and rng.random() < .5:      # bars on the low windows
                    for i in range(9):
                        x = x0 + (i + .5) * WIN_W / 9
                        cuboid(p["rail"], T, x - .008, x + .008, -.16, -.144, s + .02, h + .02)
                    for z in (s + .3, h - .1):
                        cuboid(p["rail"], T, x0 - .01, x1 + .01, -.17, -.14, z, z + .02)
        if detail >= 1:
            cuboid(p["trim"], T, 0, W, -.07, 0, z0 - .10, z0 + .10)
    if detail >= 1:
        for c in range(0, cols + 1, 6):     # the frame of the thing: a column at each stair, standing a little proud of the tiles
            cuboid(p["trim"], T, max(c * BAY - .22, 0), min(c * BAY + .22, W), -.11, 0, 0, H)
    if detail >= 2:
        for c in range(3, cols, 6):
            x = c * BAY + .02
            cuboid(p["pipe"], T, x, x + .09, -.1, -.01, 0, H)
    if detail >= 1:
        walled(st.walls[wall], T, 0, W, H, H + .95, cols, floors)
        cuboid(p["trim"], T, 0, W, -.08, .25, H + .95, H + 1.08)
        cuboid(p["trim"], T, W - .25, W, 0, 14, H, H + 1.05)
        cuboid(p["trim"], T, 0, .25, 0, 14, H, H + 1.05)
        for _ in range(cols // 4 if detail < 2 else 0):     # a tank on the near roof is a black slab against the sky and nothing else
            x = rng.uniform(1, W - 5)
            cuboid(p["roof"], T, x, x + 2.0, 2, 3.2, H + .9, H + 2.3)
            for ly in (2.2, 3.0):
                cuboid(p["steel"], T, x + .1, x + .16, ly, ly + .06, H, H + .9)


def far_city(st, seed, count, near_y=170.0, far_y=820.0, width=620.0, low=9, high=32):
    """The rest of it: slabs behind slabs as far as the air lets you see, a third of the way lit."""
    rng = random.Random(seed)
    for i in range(count):
        y = near_y * (far_y / near_y) ** rng.random()
        x = rng.uniform(-width, width) * (y / far_y + .25)
        cols, floors = rng.randint(8, 22), rng.randint(low, high)
        on = plan(cols, floors, seed + 100 + i, rng.uniform(.25, .42))
        block(st, placed(x, y, 0, rng.uniform(-14, 14)), cols, floors, on, seed + i, rng.choice(("brick_a", "brick_b")), 0)


# ---------------------------------------------------------------- the air, the sky

def sky(sc, sun=.07, top=(.0022, .0018, .0015), horizon=HAZE):
    """Almost black and warm overhead, and at the horizon the warm brown of a city's own lights thrown back. What little light comes down from it is enough to see that a wall is there."""
    w = sc.world
    nt = w.node_tree
    bg = nt.nodes["Background"]
    tc, sep, ramp, n, mix = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeSeparateXYZ"), nt.nodes.new("ShaderNodeValToRGB"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeMath")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    n.inputs["Scale"].default_value = 2.4
    n.inputs["Detail"].default_value = 3
    mix.operation = "MULTIPLY_ADD"
    nt.links.new(n.outputs[0], mix.inputs[0])
    mix.inputs[1].default_value = .012
    nt.links.new(sep.outputs["Z"], mix.inputs[2])
    nt.links.new(mix.outputs[0], ramp.inputs[0])
    cr = ramp.color_ramp
    cr.interpolation = "EASE"
    cr.elements[0].position, cr.elements[0].color = .0, (*horizon, 1)
    cr.elements[1].position, cr.elements[1].color = .62, (*top, 1)
    for at, k in ((.012, .62), (.03, .3), (.06, .12), (.12, .04)):
        e = cr.elements.new(at)
        e.color = (*(t + (h - t) * k for t, h in zip(top, horizon)), 1)
    nt.links.new(ramp.outputs[0], bg.inputs[0])
    bg.inputs[1].default_value = 1.0
    if sun:
        d = bpy.data.lights.new("cloud", "SUN")
        d.energy, d.color, d.angle = sun, (1.0, .80, .58), math.radians(70)
        o = bpy.data.objects.new("cloud", d)
        bpy.context.collection.objects.link(o)
        o.rotation_euler = (math.radians(72), 0, math.radians(-16))
        o.visible_glossy = False            # or every pane of dark glass is a flat grey mirror of it


def low_city(st, seed, count, y0=420.0, y1=4200.0):
    """A long way off, and lower than you are: roofs and small lights, thinning out into the brown, and the roads between them marked by their lamps."""
    rng = random.Random(seed)
    body = Build()
    lights = [Build() for _ in range(3)]
    for i in range(count):
        y = y0 * (y1 / y0) ** rng.random()
        half = .42 * y + 40
        x = rng.uniform(-half, half)
        w, d, h = rng.uniform(22, 80), rng.uniform(12, 30), rng.uniform(9, 30)
        cuboid(body, lambda a, b, c: (x + a, y + b, c), 0, w, 0, d, 0, h)
        share = rng.uniform(.14, .4)
        for f in range(int(h / STOREY)):
            for c in range(int(w / BAY)):
                if rng.random() < share:
                    x0, z0 = x + c * BAY + MARGIN, f * STOREY + SILL
                    lights[rng.randrange(2)].add([(x0, y - .1, z0), (x0 + WIN_W, y - .1, z0), (x0 + WIN_W, y - .1, z0 + WIN_H), (x0, y - .1, z0 + WIN_H)], [(0, 1, 2, 3)])
    # the lamps: roads running away from you, and a few across
    for k in range(-14, 15):
        x = k * 170.0 + rng.uniform(-30, 30)
        y = y0
        while y < y1:
            if abs(x) < .42 * y + 40:
                lights[2].add([(x - .5, y, 8.0), (x + .5, y, 8.0), (x + .5, y, 9.0), (x - .5, y, 9.0)], [(0, 1, 2, 3)])
            y += rng.uniform(32, 46) * (1 + y / 2500)
    body.object("far_body", surface("far_body", (.03, .025, .02), .9))
    for k, (b, colour, level) in enumerate(zip(lights, ((1.0, .80, .55), (1.0, .90, .74), (1.0, .55, .22)), (1.3, 1.5, 2.6))):
        o = b.object(f"lit_far{k}", glow(f"far{k}", colour, level))
        o.visible_diffuse = o.visible_glossy = o.visible_shadow = o.visible_transmission = o.visible_volume_scatter = False


def glow(name, colour, level, spill=0.0):
    """Plain light, seen in the second plate only, and thrown about only if it has any spill."""
    m = bpy.data.materials.new("lit_" + name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e, o = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    e.inputs[0].default_value = (*colour, 1)
    path_, off, mul = nt.nodes.new("ShaderNodeLightPath"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeMath")
    off.operation, mul.operation = "SUBTRACT", "MULTIPLY"
    off.inputs[0].default_value = 1
    nt.links.new(path_.outputs["Is Camera Ray"], off.inputs[1])
    nt.links.new(off.outputs[0], mul.inputs[0])
    mul.inputs[1].default_value = spill
    nt.links.new(mul.outputs[0], e.inputs[1])
    nt.links.new(e.outputs[0], o.inputs[0])
    m["words"] = level
    return m


def air(length, y0=25.0, y1=1400.0, half=1500.0, top=400.0, colour=HAZE, tint=AIR, scatter=0.0, floor=-30.0):
    m = bpy.data.materials.new("air")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    ab, own, add, out = nt.nodes.new("ShaderNodeVolumeAbsorption"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeAddShader"), nt.nodes.new("ShaderNodeOutputMaterial")
    sigma = 1.0 / length
    ab.inputs["Color"].default_value = (*tint, 1)
    ab.inputs["Density"].default_value = sigma
    own.inputs[0].default_value = (*(h * (1 - a) for h, a in zip(colour, tint)), 1)
    own.inputs[1].default_value = sigma
    nt.links.new(ab.outputs[0], add.inputs[0])
    nt.links.new(own.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Volume"])
    if scatter:     # the windows' light, turned towards you by the air close to them
        sc_, more = nt.nodes.new("ShaderNodeVolumeScatter"), nt.nodes.new("ShaderNodeAddShader")
        sc_.inputs["Color"].default_value = (1.0, .84, .66, 1)
        sc_.inputs["Density"].default_value = scatter
        nt.links.new(add.outputs[0], more.inputs[0])
        nt.links.new(sc_.outputs[0], more.inputs[1])
        nt.links.new(more.outputs[0], out.inputs["Volume"])
    me = bpy.data.meshes.new("air")
    v = [(x, y, z) for x in (-half, half) for y in (y0, y1) for z in (floor, top)]
    me.from_pydata(v, [], [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    me.materials.append(m)
    o = bpy.data.objects.new("air", me)
    bpy.context.collection.objects.link(o)
    o["own"] = 1
    return o


# ---------------------------------------------------------------- the frames

C1_CAM = ((0.0, 0.0, 40.0), (1.5, 46.0, 35.6), 58.0)


def street_c1():
    """From about the same height across the way: a block of flats close enough to look into, one farther off, and behind them the city, going on to where the sky starts."""
    st = Street()
    near = plan(13, 13, 71, .52)
    near.pop((12, 10), None)        # it would be lit right behind the end of the subtitle
    block(st, placed(-37.0, 46.0, 0, -5), 13, 13, near, 71, "brick_a", 2)
    block(st, placed(9.0, 96.0, 0, 7), 15, 10, plan(15, 10, 72, .46), 72, "brick_b", 1)
    block(st, placed(52.0, 160.0, 0, -3), 16, 12, plan(16, 12, 73, .4), 73, "brick_b", 1)
    far_city(st, 703, 34, 230.0, 700.0, 380.0, 5, 11)
    st.done(1.9, .55)
    low_city(st, 74, 1500)
    air(1500.0, 25.0, 4600.0, 2400.0, 64.0)        # no higher than the roofs: what the sky gets of it thins out as you look up


# the wall that fills the frame at the cut, seen along itself: the nearest of its windows large enough to read as what they will become, and the rest going away from there to specks. The three we know among the near ones
CUT_COLS, CUT_FLOORS = 78, 64
A_AT, B_AT, C_AT = (77, 32), (76, 32), (77, 31)
CUT_LENS, CUT_YAW, CUT_PITCH = 30.0, 55.0, 3.0
CUT_AT = (80 * BAY, -13.0, 92.4)
CUT_REVEAL = .06
CUT_AIR = 300.0
SUBTITLE = (530, 1400, 925, 1012)   # where the words go: on a lit window they could not be read


def on_screen(c, f):
    """Where the window at (col, floor) comes in the frame: left, right, top, foot, in pixels."""
    yaw, pitch = math.radians(CUT_YAW), math.radians(CUT_PITCH)
    fw = (-math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch))
    rt = (math.cos(yaw), math.sin(yaw), 0.0)
    up = (rt[1] * fw[2] - rt[2] * fw[1], rt[2] * fw[0] - rt[0] * fw[2], rt[0] * fw[1] - rt[1] * fw[0])
    px = 1920 * CUT_LENS / 36.0
    xs, ys = [], []
    for x in (c * BAY + MARGIN, c * BAY + MARGIN + WIN_W):
        for z in (f * STOREY + SILL, f * STOREY + SILL + WIN_H):
            d = (x - CUT_AT[0], CUT_REVEAL - CUT_AT[1], z - CUT_AT[2])
            depth = max(sum(a * b for a, b in zip(d, fw)), .5)
            xs.append(960 + px * sum(a * b for a, b in zip(d, rt)) / depth)
            ys.append(540 - px * sum(a * b for a, b in zip(d, up)) / depth)
    return min(xs), max(xs), min(ys), max(ys)


def seen_width(c, f):
    """How many pixels across the frame the window at (col, floor) comes to."""
    x0, x1, _, _ = on_screen(c, f)
    return x1 - x0


def cut_plan():
    known = ((*A_AT, "room_a"), (*B_AT, "room_b"), (*C_AT, "room_c"))
    out = plan(CUT_COLS, CUT_FLOORS, 91, .75, known)
    rng = random.Random(94)
    for c in range(CUT_COLS - 8, CUT_COLS):     # the near ones are what you read the wall by
        for f in range(CUT_FLOORS):
            if (c, f) not in out and rng.random() < .75:
                out[c, f] = rng.choice(LIVELY)
    for at in list(out):
        x0, x1, y0, y1 = on_screen(*at)
        if x1 > SUBTITLE[0] and x0 < SUBTITLE[1] and y1 > SUBTITLE[2] and y0 < SUBTITLE[3]:
            del out[at]
    return out


def cut_camera():
    """The same camera on both sides of the cut."""
    yaw, pitch = math.radians(CUT_YAW), math.radians(CUT_PITCH)
    x, y, z = CUT_AT
    return CUT_AT, (x - 40 * math.sin(yaw) * math.cos(pitch), y + 40 * math.cos(yaw) * math.cos(pitch), z + 40 * math.sin(pitch)), CUT_LENS


def shrunk(px):
    """The picture ready made for a window this size on the screen."""
    return "" if px >= 160 else "_m320" if px >= 80 else "_m160" if px >= 40 else "_m80"


def street_c3a():
    st = Street(CUT_REVEAL)
    block(st, placed(0.0, 0.0), CUT_COLS, CUT_FLOORS, cut_plan(), 91, "brick_a", 2, False, lambda c, f: shrunk(seen_width(c, f)))
    # the stair end of the block: no windows in the last of it, so that the nearest one has the whole of itself in frame
    T, W, H = placed(0.0, 0.0), CUT_COLS * BAY, CUT_FLOORS * STOREY
    walled(st.walls["brick_a"], T, W, W + 24, 0, H, 0, 0)
    for f in range(CUT_FLOORS):
        cuboid(st.parts["trim"], T, W, W + 24, -.07, 0, f * STOREY - .10, f * STOREY + .10)
    cuboid(st.parts["trim"], T, W + 1.2, W + 1.64, -.11, 0, 0, H)
    cuboid(st.parts["pipe"], T, W + 2.3, W + 2.39, -.1, -.01, 0, H)
    st.done(2.2, .5)
    air(CUT_AIR, -300.0, 300.0, 700.0, 330.0)
    widths = [seen_width(*at) for at in (A_AT, B_AT, C_AT)]
    print("A B C wide", [round(w) for w in widths], flush=True)


# ---------------------------------------------------------------- rendering

def plate(sc, path):
    exr = path + ".exr"
    sc.render.filepath = exr
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(exr)
    w, h = im.size
    a = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(a)
    np.save(path + ".npy", a.reshape(h, w, 4)[::-1][..., :3].astype(np.float16))
    bpy.data.images.remove(im)
    os.remove(exr)


def only_the_windows(sc):
    """Everything black, the windows shown as they are, the air still between you and them."""
    black = surface("black", (0, 0, 0), 1.0, 0, 0)
    for o in list(bpy.data.objects):
        if o.type != "MESH":
            continue
        if o.get("halo"):
            bpy.data.objects.remove(o)
        elif o.get("glass"):    # you still see through it, as far as it lets you; it just has nothing to shine with
            o.data.materials[0].node_tree.nodes["shine"].inputs["Color"].default_value = (0, 0, 0, 1)
        elif o.name.startswith("lit_"):
            m = o.data.materials[0]
            e = next(n for n in m.node_tree.nodes if n.type == "EMISSION")
            for l in list(e.inputs[1].links):
                m.node_tree.links.remove(l)
            e.inputs[1].default_value = m["words"]
        elif o.get("own"):
            nt = o.data.materials[0].node_tree
            next(n for n in nt.nodes if n.type == "EMISSION").inputs[1].default_value = 0
            for n in nt.nodes:
                if n.type == "VOLUME_SCATTER":      # what it would have turned aside is still gone from the way
                    next(a for a in nt.nodes if a.type == "VOLUME_ABSORPTION").inputs["Density"].default_value += n.inputs["Density"].default_value * .5
                    n.inputs["Density"].default_value = 0
        else:
            o.data.materials.clear()
            o.data.materials.append(black)
    sc.world.node_tree.links.clear()
    sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.0


def set_up(samples, pct):
    sc = reset(samples, pct)
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.color_mode = "RGB"
    sc.cycles.max_bounces = 4
    sc.cycles.diffuse_bounces = 2
    sc.cycles.glossy_bounces = 2
    sc.cycles.transparent_max_bounces = 8
    sc.cycles.volume_bounces = 0
    sc.cycles.sample_clamp_indirect = 5.0
    sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    return sc


def lens_on(sc, at, target, lens):
    c = bpy.data.cameras.new("cam")
    c.lens, c.clip_start, c.clip_end = lens, 1.0, 4000.0
    o = bpy.data.objects.new("cam", c)
    bpy.context.collection.objects.link(o)
    sc.camera = o
    aim(o, at, target)


def mask_plate(sc, path, materials, keep_rest=False):
    """Which pixels are window, as white on black, with nothing in the way unless keep_rest, which leaves the frames and the bars standing: to measure the cut by."""
    white = bpy.data.materials.new("white")
    white.use_nodes = True
    nt = white.node_tree
    nt.nodes.clear()
    e, out = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(e.outputs[0], out.inputs[0])
    for o in list(bpy.data.objects):
        if o.type != "MESH":
            continue
        if any(o.name.startswith(x) for x in materials):
            o.data.materials.clear()
            o.data.materials.append(white)
        elif not keep_rest:
            bpy.data.objects.remove(o)
    sc.cycles.use_denoising = False
    sc.cycles.samples = 16
    sc.world.node_tree.links.clear()
    sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.0
    plate(sc, path)
    a = np.load(path + ".npy").astype(np.float32)[..., 0]
    np.save(path + ".npy", a)


def main():
    key, pct, samples = ARGS[0], opt("pct", 100), opt("samples", 128)
    os.makedirs(PLATES, exist_ok=True)
    t0 = time.time()
    sc = set_up(samples, pct)
    if key == "C1":
        sky(sc)
        street_c1()
        lens_on(sc, C1_CAM[0], C1_CAM[1], C1_CAM[2])
        sc.camera.data.clip_end = 6000.0        # past the far end of the air, or a ray that stops short of it gets none of it, and the sky has a step at that height
    else:
        sky(sc)
        street_c3a()
        lens_on(sc, *cut_camera())
    print("built", round(time.time() - t0, 1), flush=True)
    plate(sc, os.path.join(PLATES, key + "_room"))
    print(key, "street", round(time.time() - t0, 1), flush=True)
    t1 = time.time()
    only_the_windows(sc)
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.samples = max(samples, 64)
    plate(sc, os.path.join(PLATES, key + "_words"))
    print(key, "windows", round(time.time() - t1, 1), flush=True)
    if key == "C3a":
        mask_plate(sc, os.path.join(PLATES, key + "_seen"), ("lit_",), True)
        mask_plate(sc, os.path.join(PLATES, key + "_mask"), ("lit_",))


if __name__ == "__main__":
    main()
