"""五 · 字的位置, last frame: 那一大堆数字，记的便是这些。 The camera has gone on rising and the delta has shrunk to digits, lying on the water, and it is the same water as in 三: the pool, its far ranges, its fall, its moon in the mirror,
a ring going out across it. Where the islands were the numbers run darker and the mirror is dull; the names can still just be read. So what filled the pool all along was this map.

    python ink3d_map_tex.py numbers --n=1200 --tex=6144
    Blender -b --factory-startup -P ink3d_map_numbers.py -- --shot=numbers [--t=19] [--pct=100] [--samples=96]     # t: the ring's radius; the next round animates it, one ring per heartbeat as in 三
    python ink3d_map_post.py ../../ink/film_out/3d/map/numbers --gamma=1.15 --mist=0 --air=0 --clouds=0

Takes over from coords as the camera climbs and the land's colour drains out into the digits. Hands on to 六 with the ring still travelling: the same pulse that closed 三.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

from ink3d import OUT3D, camera, lerp, render, reset, sm, spectral, sun
from ink3d_jing_lib import fresh, opt, sky_world
from ink3d_jing import gorge, moon, ring_wave, water_with_digits, waterfall          # with --shot=numbers, ink3d_jing itself renders nothing

TEX = os.path.join(OUT3D, "map", "tex")
EYE = (-.6, -10.0, 7.0)          # as POOL in ink3d_map_tex.py
SIZE, CY = 110.0, 38.0
HEART = (4.3, 35.0)              # at the far edge of the delta, in the strait between two islands, so the moon does not sit on the names


def load(name):
    return np.load(os.path.join(TEX, f"numbers_{name}.npy"))


t = opt("t", 19.0)
t2 = t * .62
sc = reset(samples=opt("samples", 96), pct=opt("pct", 100))
sc.render.film_transparent = False
sc.cycles.use_denoising = False
sc.cycles.use_adaptive_sampling = False
fine = spectral(1024, 2.7, 5)
r_of = lambda X, Y: np.sqrt((X - HEART[0]) ** 2 + (Y - HEART[1]) ** 2)


def height(X, Y):
    r = r_of(X, Y)
    return (ring_wave(r - t, .05, 1.1) + ring_wave(r - t2, .014, 1.0)) * sm((r - 3.4) / 1.4) + .0016 * sm((r - 3.4) / 5) * lerp(fine, X * 7, Y * 7)


land, relief, names = load("land"), load("relief"), load("names")
Q = land.shape[0]
by, bx = np.mgrid[:Q, :Q].astype(np.float32)
BX, BY = (bx + .5) / Q * SIZE - SIZE / 2, (by + .5) / Q * SIZE - SIZE / 2 + CY
r = r_of(BX, BY)
d = r - t
crest = np.exp(-(d / 1.2) ** 2)
wake = .55 * sm(-d / .6) * np.exp(np.minimum(d, 0) / 2.4)
older = .5 * np.exp(-((r - t2) / .55) ** 2)
turned = np.clip(crest + wake + older, 0, 1)
fade = np.clip(1.62 - np.sqrt((BX - EYE[0]) ** 2 + (BY - EYE[1]) ** 2) / 44, 0, 1)
patch = np.clip(.86 + .2 * lerp(spectral(1024, 3.0, 9), BX * 8, BY * 8), .4, 1)
# the numbers: darker where the land was. The mirror: dull where the land was, duller on its high ground, and the names are the dullest of all
ink = fade * patch * (.5 + .5 * land) * (1 - turned)
bright = 1.5 * (crest + wake * .55 + older * .7) * fade * (1 - names)
shade = (1 - .22 * land - .14 * relief) * (1 - .62 * crest * fade - .28 * older * fade) * (1 - .9 * names)
water_with_digits(SIZE, 0, CY, opt("n", 2600), height, ink, bright, shade=shade, changed=(d < 0).astype(np.float32), digit=.62)
gorge()
waterfall(-11.0, 122.0, 15.0, 1.1)
md = moon(EYE, HEART, 340.0, 8.5, 6.0)
sky_world(md, overhead=opt("over", .72), horizon=opt("horizon", .95), haze=opt("haze", .35), haze_deg=opt("hazedeg", 28.0), cloud=.10)
sun((.4, .8, -.42), energy=1.0, angle=4)
bpy.data.objects["sun"].visible_glossy = False
camera(EYE, (-.6, 14.0, 1.0), lens=28)
render(fresh(os.path.join(OUT3D, "map", "numbers")))
