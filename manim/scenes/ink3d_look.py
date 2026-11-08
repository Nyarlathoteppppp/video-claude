"""The look test: a river going away between steep mountains, nothing else. If this is not beautiful, nothing built on it will be.

    Blender -b --factory-startup -P manim/scenes/ink3d_look.py -- [--pct=100] [--n=2000] [--samples=64]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from ink3d import (OUT3D, camera, erode, ink_albedo, massif, mesh_grid, painted_material, render, reset, sm, sun, water_material, world_light)

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=")[1])
    return default


PCT, N = opt("pct", 100), opt("n", 2000)
SIZE = 340.0
sc = reset(samples=opt("samples", 64), pct=PCT)

y, x = np.mgrid[:N, :N].astype(np.float32)
X, Y = (x / (N - 1) - .5) * SIZE, (y / (N - 1) - .5) * SIZE
PEAKS = [(-66, 18, 92, 34), (-106, -30, 52, 26), (-22, 58, 60, 26), (52, 44, 66, 28), (92, -8, 42, 24), (8, 134, 88, 44), (-56, 118, 66, 34), (70, 124, 60, 34),
         (-44, -84, 13, 12), (36, -84, 9, 10), (-140, 44, 64, 36), (140, 54, 64, 38)]
h = massif(X, Y, PEAKS, 31)
# the river: comes toward the camera, winding, and the mountains stand back from it
cx = 11 * np.sin(Y / 34 + .4) + 5 * np.sin(Y / 15 + 2)
d = np.abs(X - cx)
width = 5.5 + .035 * (Y + SIZE / 2)
h = h * sm((d - width) / 26) ** .7
h = erode(h, SIZE / N, 41, wavelength=36, octaves=7, gain=.5, strength=.25)
h = np.maximum(h, 0) + .55 * sm((d - width) / 4)
h -= .5 * (1 - sm((d - width * .8) / 2.2))
ground = mesh_grid("land", X, Y, h)
ground.data.materials.append(painted_material("rock", ink_albedo(h, 21, SIZE / N)))

wx, wy = np.meshgrid(np.linspace(-SIZE / 2, SIZE / 2, 2), np.linspace(-SIZE / 2, SIZE / 2, 2))
water = mesh_grid("water", wx.astype(np.float32), wy.astype(np.float32), np.full((2, 2), .22, np.float32))
water.data.materials.append(water_material(rough=.05, normal_scale=36, normal_strength=.05))

world_light(.30)
sun((.82, .28, -.34), energy=2.4, angle=6)

cam_y = -116.0
c0 = 11 * math.sin(cam_y / 34 + .4) + 5 * math.sin(cam_y / 15 + 2)
camera((c0 + 1, cam_y, 6.0), (c0 + 5, cam_y + 60, 15.0), lens=30)
render(os.path.join(OUT3D, "look", "river"))
