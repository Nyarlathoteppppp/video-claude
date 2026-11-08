"""What chapters 三 and 四 share inside Blender: options, the painted sky that water mirrors, far ridges parted by mist, and ink volumes read back from a Mantaflow cache.

A night in this film is grey ink over paper, never black: the sky carries a large wash from paper to mid grey, water mirrors it, and everything else is measured against that.
"""
import math
import os
import sys

import bpy
import numpy as np

from ink3d import OUT3D, blur, erode, image_from, ink_albedo, lerp, massif, mesh_grid, painted_material, sm, spectral

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
TEX = os.path.join(OUT3D, "tex_jr")
CACHE = os.path.expanduser("~/render_work/cache")


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=")[1])
    return default


def tex(name):
    return np.load(os.path.join(TEX, name + ".npy"))


def link(o):
    bpy.context.collection.objects.link(o)
    return o


def fresh(path):
    """render() skips what is on disk, so what is on disk moves aside."""
    old = os.path.join(os.path.dirname(path), "old")
    os.makedirs(old, exist_ok=True)
    for tail in (".png", "_z.png"):
        if os.path.exists(path + tail):
            os.replace(path + tail, os.path.join(old, os.path.basename(path) + tail))
    return path


def to_linear(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= .04045, v / 12.92, ((v + .055) / 1.055) ** 2.4).astype(np.float32)


def sky_world(moon_dir=None, overhead=.52, horizon=.86, haze=.42, haze_deg=26.0, horizon_deg=7.0, cloud=.12, seed=7):
    """The sky as the render and the water both see it, painted in display values: a wash from overhead to a pale horizon band, the moon's haze, thin cloud lying along the horizon."""
    h, w = 1024, 2048
    jj, ii = np.mgrid[:h, :w].astype(np.float32)
    lat = (jj + .5) / h * math.pi - math.pi / 2
    lon = (.5 - (ii + .5) / w) * 2 * math.pi
    la = np.maximum(lat, 0)                                     # below the horizon: what mist there is
    L = overhead + (horizon - overhead) * np.exp(-(np.degrees(la) / horizon_deg) ** 2)
    if moon_dir is not None:
        d = np.stack([np.cos(la) * np.cos(lon), np.cos(la) * np.sin(lon), np.sin(la)], -1)
        cosang = np.clip((d * np.array(moon_dir, np.float32)).sum(-1), -1, 1)
        L = L + haze * np.exp(-(np.degrees(np.arccos(cosang)) / haze_deg) ** 2)
    tile = spectral(1024, 3.1, seed)
    fine = spectral(1024, 2.5, seed + 1)
    n = lerp(tile, ii * .5, jj * 1.6) + .35 * lerp(fine, ii * 1.1, jj * 3.2)
    L = L * (1 + cloud * n * sm((np.degrees(la) + 1) / 3))
    nt = bpy.context.scene.world.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = image_from(to_linear(L), "sky")
    bg = nt.nodes["Background"]
    bg.inputs[1].default_value = 1.0
    nt.links.new(env.outputs[0], bg.inputs[0])
    return L


def ridge_layer(name, peaks, seed, xr, yr, n, f0, f1, zm, mist=.84, soft=5.0):
    """One line of far mountains. The mist is in the material, not the grade, so the water's reflection of it agrees: f0 of the way to mist at the top, f1 at the foot."""
    y, x = np.mgrid[:n, :n].astype(np.float32)
    X, Y = x / (n - 1) * (xr[1] - xr[0]) + xr[0], y / (n - 1) * (yr[1] - yr[0]) + yr[0]
    cell = (xr[1] - xr[0]) / n
    h = massif(X, Y, peaks, seed)
    h = h * sm((Y - yr[0]) / 16)
    h = blur(h, soft)                                           # from this far off a ridge is a soft line, not teeth
    h = erode(h, cell, seed + 1, wavelength=70, octaves=4, gain=.38, strength=.08)
    h = np.maximum(h - .2, -.5)
    o = mesh_grid(name, X, Y, h)
    m = painted_material(name, ink_albedo(h, seed + 2, cell) * .7)
    nt = m.node_tree
    bs = nt.nodes["Principled BSDF"]
    out = nt.nodes["Material Output"]
    e, mix, geo, sp, rm = (nt.nodes.new(k) for k in ("ShaderNodeEmission", "ShaderNodeMixShader", "ShaderNodeNewGeometry", "ShaderNodeSeparateXYZ", "ShaderNodeMapRange"))
    c = float(to_linear(np.array(mist)))
    e.inputs["Color"].default_value = (c, c, c, 1)
    nt.links.new(geo.outputs["Position"], sp.inputs[0])
    nt.links.new(sp.outputs["Z"], rm.inputs["Value"])
    rm.inputs["From Min"].default_value, rm.inputs["From Max"].default_value = 0, zm
    rm.inputs["To Min"].default_value, rm.inputs["To Max"].default_value = f1, f0
    nt.links.new(rm.outputs[0], mix.inputs[0])
    nt.links.new(bs.outputs[0], mix.inputs[1])
    nt.links.new(e.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    o.data.materials.append(m)
    return o


def mirror(name, tint=.86, rough=.025):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (tint, tint, tint, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = rough
    return m


def far_water(tint=.86, z=-.004):
    """Beyond the fine mesh: the same mirror, flat, out to the far shore."""
    fx, fy = np.meshgrid(np.array([-900, 900], np.float32), np.array([-60, 900], np.float32))
    o = mesh_grid("water_far", fx, fy, np.full((2, 2), z, np.float32))
    o.data.materials.append(mirror("water_far", tint, .03))
    return o
