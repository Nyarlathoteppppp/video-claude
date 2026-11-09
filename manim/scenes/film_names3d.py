"""十二's two inserts in the room, at night as 十 and 十三 have it: 人, the room with nobody in it and the screen answering; 器, the keyboard and the computer, low across the table in the screen's light. Nothing of the room is built here: the set is film_room3d.py's and the night (lamp, the block across the road, the back wall) film_room_night3d.py's, loaded without its call to main. Only the cameras are new, and what the screen shows.

Inside Blender, from manim/scenes:

    Blender -b --factory-startup -P film_names3d.py -- --shot=ren [--passes=lamp,glow,city] [--pct=100] [--samples=256]
    Blender -b --factory-startup -P film_names3d.py -- --shot=qi

Passes go to ink/film_out/film/names/plates/SHOT_PASS.npy, float16, linear, as film_room_night3d.py writes its own, and SHOT.json where the screen's corners fall; film_names.py balances them with film_room_night's strengths and lays the screen in. What the screen gives out is names/ui/glow_SHOT.png (film_names.py ui).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy
from mathutils import Vector

# film_room_night3d.py renders as soon as it is loaded: take everything it defines, without the call at its end
_path = os.path.join(HERE, "film_room_night3d.py")
with open(_path, encoding="utf-8") as _f:
    _src = _f.read().rstrip()
assert _src.endswith("\nmain()"), "film_room_night3d.py no longer ends in its call to main"
N = {"__file__": _path, "__name__": "film_room_night3d_parts"}
exec(compile(_src[:-len("main()")], _path, "exec"), N)
R = N["R"]
opt = N["opt"]

OUT = os.path.join(N["ROOT"], "ink", "film_out", "film", "names")
N["PLATES"] = os.path.join(OUT, "plates")         # its render() and corners_json() write here now
UI = os.path.join(OUT, "ui")
SHOT = opt("shot", "ren")
PASSES = opt("passes", "lamp,glow,city").split(",")

ZT = R.ZT
# loc: where the camera stands; at: what it looks at; focus: the point that is sharp. Metres, in film_room3d's room: x right as you face the window wall, y towards it, z up
SHOTS = dict(
    # from behind and to the left, at a standing eye's height: the table, the lit screen turned a little away, the window with the block across the road, the lamp. Nobody, and no chair
    ren=dict(loc=(-1.32, -1.30, 1.46), at=(0.06, 0.34, 0.90), lens=30.0, fstop=4.0, focus=(0.03, 0.30, ZT + 0.10)),
    # low over the table from the front left, across the keyboard to the Mac Studio, the display's foot between them, the screen's light on the keys
    qi=dict(loc=(-0.34, -0.22, ZT + 0.105), at=(0.40, 0.40, ZT + 0.075), lens=40.0, fstop=2.8, focus=(0.03, 0.135, ZT + 0.006)),
)


def aim(cam, s):
    loc, at = Vector(s["loc"]), Vector(s["at"])
    cam.location = loc
    cam.rotation_euler = (at - loc).to_track_quat("-Z", "Y").to_euler()
    d = cam.data
    d.lens, d.shift_x, d.shift_y = s["lens"], 0.0, 0.0
    d.dof.use_dof = True
    d.dof.focus_distance = (Vector(s["focus"]) - loc).length
    d.dof.aperture_fstop = s["fstop"]
    d.dof.aperture_blades = 7


def main():
    os.makedirs(N["PLATES"], exist_ok=True)
    sc = N["reset"](N["SAMPLES"], N["PCT"])
    # the scene's settings exactly as film_room_night3d's main has them
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.exr_codec = "ZIP"
    sc.view_settings.view_transform = "Raw"
    sc.cycles.max_bounces = 8
    sc.cycles.diffuse_bounces = 5
    sc.cycles.glossy_bounces = 4
    sc.cycles.transparent_max_bounces = 12
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.cycles.blur_glossy = 1.0
    sc.cycles.sample_clamp_indirect = 8.0
    sc.cycles.adaptive_threshold = .01

    R.room()
    R.table()
    screen, corners, lid = R.display()
    R.things()
    N["back_wall"]()
    bulbs, bulb_glass = N["floor_lamp"]()
    out_strength = N["night_outside"]()
    glow_strength = N["night_screen"](screen, os.path.join(UI, f"glow_{SHOT}.png"))
    bg = sc.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (.020, .015, .011, 1)
    R.neutral_in_metal()

    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_width, cam_data.clip_start, cam_data.clip_end = 36.0, .01, 60
    cam = R.link(bpy.data.objects.new("cam", cam_data))
    sc.camera = cam
    aim(cam, SHOTS[SHOT])
    bpy.context.view_layer.update()
    N["corners_json"](sc, cam, corners, SHOT)

    def lights(lamp=0.0, glow=0.0, city=0.0):
        for b in bulbs:
            b.data.energy = 55.0 * lamp
            b.hide_render = lamp == 0
        for g in bulb_glass:
            g.hide_render = False
            g.data.materials[0].node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 6.0 * lamp
        glow_strength.default_value = glow
        out_strength.default_value = city
        bg.inputs[1].default_value = city

    for p in PASSES:
        lights(**{p: 1.0})
        N["render"](f"{SHOT}_{p}")
    print("done", flush=True)


main()
