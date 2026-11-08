"""The title beat's stone, in light: the ident's own stone (film_ident.py's block, face, polish and the room its polish sees, its sun and sky) comes down onto a sheet beside the signature, presses, lifts and is gone. Only light is rendered here; the sheet's picture, the poem and the name, is film_hand.py's, and film_title.py lays this light over it: what the stone does to the light on the paper is the render divided by the same paper with nothing on it.

Inside Blender, from manim/scenes (python film_title.py plan first):

    Blender -b --factory-startup -P film_title3d.py -- flat                    # title3d/flat.npy: the sheet, nothing on it, not pressed
    Blender -b --factory-startup -P film_title3d.py -- left                    # title3d/left.npy: the sheet after, the stone gone, what of the press stays
    Blender -b --factory-startup -P film_title3d.py -- frames 117 177          # title3d/fNNNN.npy and aNNNN.npy: light and the stone's cover, frames of the beat (30 a second) from 117 to 177
    ... [--pct 100] [--samples 128]

The world is in centimetres, as the ident's: x to the right of the frame, y up it, z up from the sheet. The camera looks straight down from where 48 cm of the sheet fills the frame through a 40 mm lens, and never moves. Every pass is linear light, float16, top row first.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

import film_ident as I
import film_press_machine as K
from ink3d import mesh_grid, reset

OUT = os.path.join(I.ROOT, "ink", "film_out", "film", "hand", "title3d")
with open(os.path.join(OUT, "plan.json")) as _f:
    P = json.load(_f)
FPS = P["fps"]
PXCM = P["px_cm"]
SX, SY, SS = P["seal"]
X0, Y0 = (SX + SS / 2 - P["w"] / 2) / PXCM, (P["h"] / 2 - (SY + SS / 2)) / PXCM      # the middle of the seal's place on the sheet
LENS = 40.0
HIGH_CM = P["w"] / PXCM * LENS / 36.0           # how high the camera is for the frame to hold 48 cm
STOP = 11.0 / 100                               # f/11, in a world of centimetres that Blender takes for metres
# where the stone is when the beat begins to show it: above and outside the frame at the lower right, turned and tipped as a held thing is, coming round as it comes down
START = dict(at=(X0 + 7.0, Y0 - 9.0, 34.0), tilt=26.0, yaw=-16.0, roll=7.0)
UP = 40.0                                       # where it goes when it lifts: out past the frame's edge long before


def lerp(a, b, u):
    return a + (b - a) * u


def sunk(t):
    """How far down the paper is under the stone at t, as film_ident.sunk with this beat's times."""
    if t < P["land"]:
        return 0.0
    if t < P["rise"]:
        return I.DEEP * (1 + .22 * math.exp(-(t - P["land"]) / .05))
    return lerp(I.DEEP, I.LEFT, float(I.sm((t - P["rise"]) / .35)))


def pose(t):
    """Where the middle of the stone's face is at t and how far it has come round from START; None while it is out of the world of the shot."""
    if t < P["fall"] or t >= P["gone"]:
        return None
    if t < P["land"]:
        u = ((t - P["fall"]) / (P["land"] - P["fall"])) ** 2.6        # slowly, then fast, as the ident's
        return tuple(lerp(a, b, u) for a, b in zip(START["at"], (X0, Y0, 0.0))), u
    if t < P["rise"]:
        return (X0, Y0, -sunk(t)), 1.0
    return (X0, Y0, lerp(-sunk(P["rise"] - .001), UP, float(I.sm((t - P["rise"]) / (P["gone"] - P["rise"]))))), 1.0


class Set:
    def __init__(self, samples, pct, frames=()):
        self.sc = sc = reset(samples, pct)
        sc.render.resolution_x, sc.render.resolution_y = P["w"], P["h"]
        sc.render.fps, sc.render.fps_base = FPS, 1.0
        sc.render.film_transparent = False
        sc.render.image_settings.file_format = "OPEN_EXR"
        sc.render.image_settings.color_depth = "32"
        sc.render.image_settings.color_mode = "RGBA"
        sc.render.image_settings.exr_codec = "ZIP"
        sc.view_settings.view_transform = "Raw"
        sc.render.use_persistent_data = True
        sc.render.use_motion_blur = True
        sc.render.motion_blur_shutter = .5
        sc.cycles.max_bounces = 8
        sc.cycles.diffuse_bounces = 4
        sc.cycles.glossy_bounces = 4
        sc.cycles.sample_clamp_indirect = 8.0          # as film_room3d: the polish sees the window panes by way of the paper, and a few such paths make stars
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
        sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"

        # the ident's light: its sky, and its sun from the frame's left and a little beyond its top
        w = sc.world.node_tree.nodes["Background"]
        w.inputs[0].default_value = (*I.SKY_COLOUR, 1)
        w.inputs[1].default_value = I.SKY_E
        s = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        bpy.context.collection.objects.link(s)
        s.data.energy, s.data.color, s.data.angle = I.SUN_E, I.SUN_COLOUR, math.radians(I.SUN_SIZE)
        az, el = math.radians(I.SUN_AZ), math.radians(I.SUN_EL)
        s.rotation_euler = Vector((-math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))).to_track_quat("Z", "Y").to_euler()

        # the sheet, finely made only where the stone presses; the ident's cockle and give, about the seal's place
        half_w, half_h = P["w"] / PXCM / 2 + 2, P["h"] / PXCM / 2 + 2
        xs = I.axis(-half_w, half_w, .25, X0 - 2.8, X0 + 2.8, .02)
        ys = I.axis(-half_h, half_h, .25, Y0 - 2.8, Y0 + 2.8, .02)
        x, y = np.meshgrid(xs, ys)
        self.flat, self.sink = I.sheet_height(x - X0, y - Y0)
        self.sheet = mesh_grid("sheet", x, y, self.flat, uv=False)
        m, nt, b = K.new_mat("paper")
        b.inputs["Base Color"].default_value = (.925, .893, .825, 1)          # the ident's paper, without its fibres: they are in the picture this light goes over
        b.inputs["Roughness"].default_value = .84
        b.inputs["Specular IOR Level"].default_value = .1
        self.sheet.data.materials.append(m)
        self.xy = np.stack([x, y], -1).astype(np.float32)
        self.down = 0.0

        self.body, self.face = I.stone_body(I.stone_material(K)), I.stone_face(I.face_material(K))
        I.room(K)
        self.hang = (Matrix.Rotation(math.radians(-START["tilt"]), 4, "X") @ Matrix.Rotation(math.radians(START["yaw"]), 4, "Z") @ Matrix.Rotation(math.radians(START["roll"]), 4, "Y")).to_quaternion()

        data = bpy.data.cameras.new("cam")
        data.lens, data.sensor_width = LENS, 36.0
        data.clip_start, data.clip_end = .5, 2000.0
        data.dof.use_dof = True
        data.dof.focus_distance = HIGH_CM
        data.dof.aperture_fstop = STOP
        data.dof.aperture_blades = 7
        self.cam = bpy.data.objects.new("cam", data)
        bpy.context.collection.objects.link(self.cam)
        self.cam.location, self.cam.rotation_euler = (0.0, 0.0, HIGH_CM), (0.0, 0.0, 0.0)
        sc.camera = self.cam

        for o in (self.body, self.face):
            o.rotation_mode = "QUATERNION"
        # keyed a frame either side of every frame asked for, so that each has its blur
        for f in sorted({g for f in frames for g in (f - 1, f, f + 1)}):
            p = pose(f / FPS)
            if p:
                for o in (self.body, self.face):
                    o.location, o.rotation_quaternion = p[0], self.hang.slerp(Quaternion(), p[1])
                    o.keyframe_insert("location", frame=f)
                    o.keyframe_insert("rotation_quaternion", frame=f)

    def press(self, down):
        if abs(down - self.down) > 1e-6:
            self.down = down
            z = self.flat + self.sink * down
            co = np.concatenate([self.xy, z[..., None].astype(np.float32)], -1)
            self.sheet.data.vertices.foreach_set("co", co.ravel())
            self.sheet.data.update()

    def shot(self, name, alpha=False):
        sc = self.sc
        t0 = time.time()
        exr = os.path.join(OUT, name + ".exr")
        sc.render.filepath = exr
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(exr)
        w, h = im.size
        a = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(a)
        a = a.reshape(h, w, 4)[::-1]
        np.save(os.path.join(OUT, name + ".npy"), (a[..., 3] if alpha else a[..., :3]).astype(np.float16))
        bpy.data.images.remove(im)
        os.remove(exr)
        print(name, f"{time.time() - t0:.0f} s", flush=True)

    def frame(self, f, samples):
        """The light at frame f, and how much of each pixel is the stone (the sheet held out, the sky away)."""
        sc = self.sc
        t = f / FPS
        sc.frame_set(f)
        seen = pose(t) is not None
        for o in (self.body, self.face):
            o.hide_render = not seen
        self.press(sunk(t))
        sc.cycles.samples = samples
        self.sheet.hide_render = False
        sc.render.film_transparent = False
        self.shot("f%04d" % f)
        if seen:
            # the stone's cover: the sheet gone and nothing behind, few samples are enough for an edge
            sc.cycles.samples = max(16, samples // 8)
            self.sheet.hide_render = True
            sc.render.film_transparent = True
            self.shot("a%04d" % f, alpha=True)
            self.sheet.hide_render = False
            sc.render.film_transparent = False


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    opt = lambda k, d: type(d)(argv[argv.index(k) + 1]) if k in argv else d
    samples, pct = opt("--samples", 128), opt("--pct", 100)
    what = argv[0]
    if what == "frames":
        a, b = int(argv[1]), int(argv[2])
        s = Set(samples, pct, range(a, b + 1))
        for f in range(a, b + 1):
            s.frame(f, samples)
    else:
        s = Set(samples, pct)
        for o in (s.body, s.face):
            o.hide_render = True
        s.press(0.0 if what == "flat" else I.LEFT)
        s.shot(what)
    print("done", flush=True)


main()
