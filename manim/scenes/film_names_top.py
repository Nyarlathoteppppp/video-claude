"""十二's last shot: straight down on the 器 sheet, and up and away until the four sheets lie on the table with the blank between 神 and 鬼. The table, the sheets and the light are 十三's, built by film_crab_new.py's own scene (crab left out): this only puts a camera over it and renders two plates, so that the pull back can be made from them as a picture, which for a flat table seen straight down is exact.

Inside Blender, from manim/scenes:

    Blender -b --factory-startup -P film_names_top.py -- [--samples=256]

    names/top/wide.png    the last frame's view, all four sheets, at twice the frame's pixels
    names/top/close.png   round the 器 sheet, where the pull back starts, at twice the frame's pixels of the frames it serves

16 bits, as film_crab_new's scene leaves its render before it adds grain; film_names.py top makes the frames from them. film_crab_new.scene is called as it is; its own render is made tiny and thrown away, and the plates rendered from where it hands over to film_imprint_clawd.finish.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import film_crab_new as crab          # nothing of Blender's at its top: film_names.py reads the plates' places from here outside Blender

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
SAMPLES = int(next((a.split("=")[1] for a in ARGS if a.startswith("--samples=")), 256))
OUT = os.path.join(crab.FILM, "names", "top")

# the pull back, in mm on the table (film_crab_new's: x across, y away from the screen's side, the sheets' tops that way): where the frame's middle is and how wide it is, first and last
QI = crab.LAY[3]
# 300 mm of paper high round 器, which film_crab_new sets a twenty-fifth of the sheet above its middle; the frame's middle a little right of the sheet's, so that nothing of 鬼 is in it but its sheet's edge
START = dict(centre=(QI[0] + 30.0, QI[1] + 0.04 * crab.PAPER_H), width=300.0 * 16 / 9)
# the four and the blank between them, the table round them, and only a little of the dark beyond the table's far edge (crab.EDGE_Y)
END = dict(centre=(0.0, -100.0), width=2200.0)
PX = (3840, 2160)


def fixed():
    """The one point of the table that stays where it is in the frame all the way, the camera going straight up over it."""
    k = END["width"] / START["width"]
    return tuple((k * a - b) / (k - 1) for a, b in zip(START["centre"], END["centre"]))


def plates():
    """Two views cover the whole pull back at twice the frame's pixels: the last frame's, and one round the start good for every frame up to twice the start's width."""
    f = fixed()
    w = 2 * START["width"]
    k = w / START["width"]
    close = tuple(p + (a - p) * k for a, p in zip(START["centre"], f))
    return dict(wide=dict(centre=END["centre"], width=END["width"]), close=dict(centre=close, width=w))


def hook(src, path, seed, grain):
    import bpy
    import film_imprint_clawd as clawd
    sc = bpy.context.scene
    data = bpy.data.cameras.new("top")
    data.type = "ORTHO"
    data.clip_start, data.clip_end = 5.0, 60000.0
    cam = bpy.data.objects.new("top", data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = PX
    sc.render.resolution_percentage = 100
    sc.cycles.samples = SAMPLES
    os.makedirs(OUT, exist_ok=True)
    for name, p in plates().items():
        data.ortho_scale = p["width"]
        cam.location = (p["centre"][0], p["centre"][1], clawd.FLOOR + 1500.0)       # under the room's ceiling (1880 above the table), over everything on it
        cam.rotation_euler = (0.0, 0.0, 0.0)          # straight down, the frame's top towards +y
        sc.render.filepath = src
        bpy.ops.render.render(write_still=True)
        shutil.copyfile(src, os.path.join(OUT, name + ".png"))
        print("plate", name, p, flush=True)


if __name__ == "__main__":
    import film_imprint_clawd
    film_imprint_clawd.finish = hook
    crab.scene(dict(name="_names_top_unused", t=0.0, draft=False, empty=True, view="top", sit=1.0, glow=1.0, samples=1, w=64, h=36, fstop=8.0, exp=.5, mesh=.5))
