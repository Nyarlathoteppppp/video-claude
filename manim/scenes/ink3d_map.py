"""五 · 字的位置 (2:00-2:22): the place words go before I can read them. Word vectors as an ink map of a river delta.

Ground and paint come from ink3d_map_tex.py; this file only places the camera and the light. t runs 0-1 through each shot.

    python ink3d_map_tex.py SET
    Blender -b --factory-startup -P ink3d_map.py -- --shot=NAME [--t=.5] [--pct=100] [--samples=64]
    python ink_grade.py ../../ink/film_out/3d/map/NAME --mist=... --mist-h=...

Shots, and how each passes to the next:
- delta   读之前，每个字先要有个位置。 Comes up out of the river's surface (四's night water, the phone drifting away): the camera rises until the river is a line and its mouth
          is spreading words like silt. It keeps rising and turns slowly.
- map     方以类聚，物以群分。 The silt has become land. Same rise, slowing to a hover; each island is a kind of meaning, and the words on it are its place names.
- route   男到女，王到后，走的是同一个方向。 Descends onto the middle island. Two marches get drawn, one stroke at a time, and they run the same way.
- coords  一个字的位置，便是一串数字。 Pushes in. Each place name unrolls its coordinates, as small notes.
- numbers 那一大堆数字，记的便是这些。 Lies on the pool of 三, so it has its own script: ink3d_map_numbers.py.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

from ink3d import OUT3D, W, H, aim, camera, image_from, mesh_grid, render, reset, sm, sun, world_light

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=")[1])
    return default


SHOT, T, PCT = opt("shot", "map"), opt("t", .5), opt("pct", 100)
TEX = os.path.join(OUT3D, "map", "tex")
SIZE = 380.0
# shot: set of paint, camera (from, to), its target (from, to), lens, sun direction, sun energy, world light.
# Note the sun: ink3d.sun turns the horizontal part of its direction around, so (-x, -y, z) is what shines along (x, y, z).
SHOTS = dict(
    delta=dict(set="delta", cam=((30, -190, 330), (24, -210, 300)), tgt=((10, 24, 0), (8, 18, 0)), lens=30, sun=(-.8, .5, -.52), e=3.8, w=.30),
    map=dict(set="map", cam=((-6, -312, 236), (-4, -300, 226)), tgt=((-8, -14, 0), (-8, -12, 0)), lens=33, sun=(-.8, .5, -.52), e=3.8, w=.30),
    route=dict(set="route", cam=((0, -122, 70), (-1, -112, 62)), tgt=((-1, -8, 2), (-1, -8, 2)), lens=35, sun=(-.8, .5, -.52), e=3.8, w=.30),
    coords=dict(set="coords", cam=((-1, -112, 62), (-2, -98, 54)), tgt=((-1, -8, 2), (-1, -14, 2)), lens=35, sun=(-.8, .5, -.52), e=3.8, w=.30),
)
S = SHOTS[SHOT]
sc = reset(samples=opt("samples", 64), pct=PCT)
sc.cycles.max_bounces = 3

h = np.load(os.path.join(TEX, S["set"] + "_h.npy"))
a = np.load(os.path.join(TEX, S["set"] + "_a.npy")).astype(np.float32) / 255
N = h.shape[0]
y, x = np.mgrid[:N, :N].astype(np.float32)
X, Y = (x / (N - 1) - .5) * SIZE, (y / (N - 1) - .5) * SIZE
land = mesh_grid("land", X, Y, h)
m = bpy.data.materials.new("paint")
m.use_nodes = True
b = m.node_tree.nodes["Principled BSDF"]
t = m.node_tree.nodes.new("ShaderNodeTexImage")
t.image = image_from(a, "paint")
t.interpolation = "Linear"
m.node_tree.links.new(t.outputs[0], b.inputs["Base Color"])
b.inputs["Roughness"].default_value = .95
b.inputs["Specular IOR Level"].default_value = .1
# the water is paper with paint on it: it takes no shadows, it only shows its wash
nt = m.node_tree
geo, sep, rng, em, mix = (nt.nodes.new(k) for k in ("ShaderNodeNewGeometry", "ShaderNodeSeparateXYZ", "ShaderNodeMapRange", "ShaderNodeEmission", "ShaderNodeMixShader"))
rng.inputs["From Min"].default_value, rng.inputs["From Max"].default_value = .08, .22
rng.inputs["To Min"].default_value, rng.inputs["To Max"].default_value = 1, 0
nt.links.new(geo.outputs["Position"], sep.inputs[0])
nt.links.new(sep.outputs["Z"], rng.inputs["Value"])
nt.links.new(t.outputs[0], em.inputs["Color"])
nt.links.new(rng.outputs[0], mix.inputs[0])
nt.links.new(b.outputs[0], mix.inputs[1])
nt.links.new(em.outputs[0], mix.inputs[2])
nt.links.new(mix.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
land.data.materials.append(m)

# the sea goes on past the edge of the map, as bare paper
sx, sy = np.meshgrid(np.linspace(-1500, 1500, 2), np.linspace(-1500, 1500, 2))
sea = mesh_grid("sea", sx.astype(np.float32), sy.astype(np.float32), np.full((2, 2), -.05, np.float32))
bs = bpy.data.materials.new("sea")
bs.use_nodes = True
bs.node_tree.nodes.clear()
em2, out2 = bs.node_tree.nodes.new("ShaderNodeEmission"), bs.node_tree.nodes.new("ShaderNodeOutputMaterial")
em2.inputs["Color"].default_value = (.985, .985, .985, 1)
bs.node_tree.links.new(em2.outputs[0], out2.inputs["Surface"])
sea.data.materials.append(bs)

world_light(S["w"])
sun(S["sun"], energy=S["e"], angle=3)
u = float(sm(T))
loc = tuple(p + (q - p) * u for p, q in zip(*S["cam"]))
tgt = tuple(p + (q - p) * u for p, q in zip(*S["tgt"]))
camera(loc, tgt, lens=S["lens"])
name = os.path.join(OUT3D, "map", SHOT if T == .5 else f"{SHOT}_{T:.2f}")
for suffix in (".png", "_z.png"):
    if os.path.exists(name + suffix):
        os.replace(name + suffix, name + suffix + ".old")
render(name)
