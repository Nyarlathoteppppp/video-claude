"""二's desk (film 51.0 to 58.3): the ink is over, and here are the real books, on the desk of 三's room in the bridge's afternoon light. One slow even push
towards the pile, never stopping, and the cut to the clay tablet of two.mp4.

Inside Blender, from manim/scenes:

    Blender -b --factory-startup -P film_desk3d.py -- --frames=0,30,165,218 [--pct=100] [--samples=128]
    Blender -b --factory-startup -P film_desk3d.py -- --frames=0:219:2 [--pct=50]      # every other frame, from 0 up to but not 218

The set is film_room3d.py's, built the same way, so it is the room the film comes back to. The light is the bridge's balance (film_room.ROOM: the sky
1.5 and the sun .7) put into the lights themselves, so one render a frame gives what film_room adds up from two; the screen is dark and out of the frame.

Where the camera goes: the vertical 「依我之见/并非如此」 stands at the right edge of the frame, from y 80 to 640. What lies behind the pile there, the only
plain and quiet thing the room has, is the curtain over the window's right half, lit through from behind; the Studio Display stands 4 cm right of the
pile at the same depth, so the frame's right edge runs between them. The push holds still the point of the curtain where the sill's line crosses it, at
FIX in the frame: the bright cloth above it, where the line is read, never grows a step, and the books, nearer, grow faster than the cloth behind them.
Frames go to ink/film_out/film/desk/plates/dNNNN.npy, linear, float16, top row first, and stay as exr/dNNNN.exr too.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from mathutils import Vector

import film_room3d as R
import ink_subs
from ink3d import ROOT, reset

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = os.path.join(ROOT, "ink", "film_out", "film", "desk", "plates")    # --out=plates_1080 puts them beside, under another name
FPS = 30
T0, T1 = ink_subs.at("依我之见"), ink_subs.at("鬼之所以哭")     # 51.0 and 58.3
N = int(round((T1 - T0) * FPS))
SKY, SUN = 1.5, .7               # film_room.ROOM's balance

# the camera: level and facing the window wall, the lens shifted, not turned, as the room's other shots
CAM0 = (-.70, -1.70, .90)        # where it starts, metres
LENS, FSTOP = 50.0, 2.8
SPINES_Y = .45                   # the plane it focuses on: the backs of the pile and the fronts of the two standing
P = (-.30, .745, .98)            # the point it moves towards: the curtain, where the sill's line runs behind it, just left of the screen's edge
FIX = (1885.0, 560.0)            # where P stays in a 1920 by 1080 frame
PUSH = .235                      # of the way to P the camera goes in the shot, at an even pace in scale: the books 1.35 times bigger at the end, their feet
                                 # still above the bottom 110 px the ordinary line takes


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


def frames():
    s = opt("frames", "0")
    if ":" in s:
        a, b, *c = (int(x) for x in s.split(":"))
        return list(range(a, b, c[0] if c else 1))
    return [int(x) for x in s.split(",")]


def camera_at(i):
    """The camera's place for frame i: the way to P shrinks by the same factor every frame, from the first frame to the last."""
    u = i / (N - 1)
    k = (1 - PUSH) ** u
    return Vector(P) + (Vector(CAM0) - Vector(P)) * k


def shift():
    """The lens shift that puts P at FIX from where the camera starts; along the line to P it stays there."""
    d = Vector(P) - Vector(CAM0)
    sx = LENS * d.x / d.y / 36 + .5 - FIX[0] / 1920
    sy = LENS * d.z / d.y / 36 - (540 - FIX[1]) / 1920
    return sx, sy


def lights():
    """The bridge's afternoon in one render: the sky's lights at SKY, the sun at SUN, the screen dark."""
    sun, portal, fill = R.daylight()
    R.set_light_world(SKY)
    fill.data.energy *= SKY
    sun.data.energy *= SUN
    out = bpy.data.objects["outside"].data.materials[0].node_tree.nodes["Emission"].inputs[1]
    out.default_value *= SKY


def main():
    global OUT
    OUT = os.path.join(os.path.dirname(OUT), opt("out", "plates"))
    os.makedirs(os.path.join(OUT, "exr"), exist_ok=True)
    sc = reset(opt("samples", 128), opt("pct", 100))
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "16"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.exr_codec = "ZIP"
    sc.view_settings.view_transform = "Raw"
    sc.cycles.max_bounces = 8
    sc.cycles.diffuse_bounces = 5
    sc.cycles.glossy_bounces = 4
    sc.cycles.transparent_max_bounces = 12
    sc.cycles.transmission_bounces = 4
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.blur_glossy = 1.0
    sc.cycles.adaptive_threshold = .012
    sc.cycles.seed = 7                      # the same noise every frame, so what the denoiser leaves does not crawl

    R.room()
    R.table()
    screen, corners, lid = R.display()
    R.things()
    lights()
    R.neutral_in_metal()
    screen.data.materials[0].node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 0.0

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens, cam_data.clip_start, cam_data.clip_end = LENS, .02, 60
    cam_data.sensor_width = 36.0
    cam_data.shift_x, cam_data.shift_y = shift()
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = FSTOP
    cam_data.dof.aperture_blades = 7
    cam = R.link(bpy.data.objects.new("cam", cam_data))
    cam.rotation_euler = (math.radians(90), 0, 0)
    sc.camera = cam
    print("frames", N, "shift", cam_data.shift_x, cam_data.shift_y, flush=True)

    for i in frames():
        t0 = time.time()
        c = camera_at(i)
        cam.location = c
        cam_data.dof.focus_distance = SPINES_Y - c.y
        bpy.context.view_layer.update()
        exr = os.path.join(OUT, "exr", f"d{i:04d}.exr")
        sc.render.filepath = exr
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(exr)
        w, h = im.size
        a = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(a)
        np.save(os.path.join(OUT, f"d{i:04d}.npy"), a.reshape(h, w, 4)[::-1, :, :3].astype(np.float16))
        bpy.data.images.remove(im)
        print(f"desk {i} {w}x{h} {time.time() - t0:.0f} s", flush=True)
    print("done", flush=True)


main()
