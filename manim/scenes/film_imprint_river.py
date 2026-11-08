"""八「河床」: 「这块泥，也没读过那些字。」「河床，也没读过河水。」 Six seconds. The block after the last blow (film_imprint_clay.state_path(LAST)), in the framing the film left it in, in the dawn light that came after. A beat, for the eye to find the black notch at the lower right of what was pressed; then the camera goes in, at one even pace about one fixed point and rising as it goes, until the notch is at the corner of the frame and the clay is a dry river seen from very high up. It is the same height field as the dawn's: nothing is put together again and no blow is added. Frames go to imprint/seven/river/.

    blender -b --factory-startup -P film_imprint_river.py -- frames 0 3             # this worker's share (0 of 3), 960 by 540, into river/
    ... -- still r0 0.0 [--w 1920 --h 1080 --samples 256]                           # seconds into the shot, into river/
    ... -- frames 0 1 --w 1920 --samples 128 --dir river_1080 --step-w 960          # the final's frames, into seven/river_1080/ (--dir: any folder of seven/), the mesh the 960 drafts'
"""
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from film_imprint_clay import A, B, LAST, SEVEN, state_path
from film_imprint_clawd import finish
from film_imprint3d import CAM_AZ, D1, DAWN_AZ, ELEV, LENS, TARGET, block_of, board_object, camera, clay_material, crumbs, dawn, set_up, sm
from ink3d import reset

LONG, HOLD = 6.0, .5                    # seconds; how long it stays still first
FROM_S, FROM_E, FROM_F = 1.7, ELEV, 22.0
TO_S, TO_E, TO_T, TO_F = 7.0, 60.0, (52.0, -44.0, -4.0), 300.0
OPEN_FILL = 4.0                         # how much more of the sky the whole block gets at the start than the raking sun's own: the wall and the hollows would be black, and the channels only need the raking light once the push has got to them
FPS = 30                                # the film's rate: frames(..., fps) would otherwise repeat every fourth


def going(t):
    """How far it has gone, 0 to 1: nothing, a quarter of a second to get going, then one even pace to the end."""
    ramp = .3
    x = np.maximum(t - HOLD, 0.0)
    done = np.where(x < ramp, x * x / (2 * ramp), x - ramp / 2)
    return float(done / (LONG - HOLD - ramp / 2))


def view(t):
    """Size, how high, what it looks at, the f number. The view's centre goes evenly with the inverse of the size, so that one point of the clay stays where it is in the frame."""
    u = going(t)
    s = FROM_S * (TO_S / FROM_S) ** u
    r = FROM_S / TO_S
    fixed = [(T1 - T0 * r) / (1 - r) for T0, T1 in zip(TARGET, TO_T)]
    target = tuple(c + (T0 - c) * FROM_S / s for c, T0 in zip(fixed, TARGET))
    return s, FROM_E + (TO_E - FROM_E) * u, target, FROM_F * (TO_F / FROM_F) ** u


def seen(s, elev, target, w):
    """The part of the block's top in the frame, and how many mm a frame pixel is at its nearest."""
    e, a = math.radians(elev), math.radians(CAM_AZ)
    d = D1 / s
    t = np.array(target)
    back = np.array((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
    p = t + d * back
    right = np.array((math.cos(a), math.sin(a), 0.0))
    up = np.cross(right, -back)
    half = 18.0 / LENS
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            ray = -back + right * sx * half + up * sy * half * 9 / 16
            pts.append(p + ray * ((-3.0 - p[2]) / ray[2]))
    pts = np.array(pts)
    win = (max(pts[:, 0].min() - 5, -A + 1), min(pts[:, 0].max() + 5, A - 1), max(pts[:, 1].min() - 5, -B + 1), min(pts[:, 1].max() + 5, B - 1))
    return tuple(float(v) for v in win), float(min(np.linalg.norm(q - p) for q in pts) * 36.0 / LENS / w)


def still(name, t, a):
    sc = reset(a["samples"], 100)
    set_up(a, sc)
    sc.cycles.max_bounces, sc.cycles.diffuse_bounces = 5, 2
    mat = clay_material()
    s, elev, target, fstop = view(t)
    # the block's mesh goes by the pixel, so at 1920 it came out twice as fine as the 960 drafts and showed the height field's steps as rows of
    # terraces on the slopes (2026-11-07). --step-w 960 makes the mesh the drafts' at any size: what is drawn stays theirs, only sharper
    win, px = seen(s, elev, target, a.get("step_w") or a["w"])
    step = max(px, 1 / 24)
    z = np.load(state_path(LAST))
    block_of(z["H"], int(z["res"]), float(z["lost"]), z["burnish"], z["bulge"], None, win, mat, near=(.5, step, step), NT=30)
    board_object()
    crumbs(mat)
    dawn(sc, DAWN_AZ)
    sc.world.node_tree.nodes["Background"].inputs[1].default_value = 1.1 * (1 + a.get("fill", OPEN_FILL) * (1 - float(sm(going(t) / .6))))
    camera(sc, s, fstop, elev=elev, target=target)
    path = os.path.join(SEVEN, a.get("dir", "river"), name + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sc.render.image_settings.file_format, sc.render.image_settings.color_depth = "PNG", "16"
    deep = f"/tmp/river16_{os.getpid()}.png"
    sc.render.filepath = deep
    bpy.ops.render.render(write_still=True)
    finish(deep, path, int(t * 1000) + 7, 1.3)
    print("wrote", path, f"size {s:.2f} high {elev:.1f}", flush=True)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]

    def opt(name, default, kind=float):
        return kind(argv[argv.index(name) + 1]) if name in argv else default

    def settings(w, samples):
        w = opt("--w", w, int)
        return dict(w=w, h=opt("--h", w * 9 // 16, int), samples=opt("--samples", samples, int), denoise=True, exp=.5, fill=opt("--fill", OPEN_FILL), dir=opt("--dir", "river", str), step_w=opt("--step-w", 0, int))

    if argv[0] == "still":
        still(argv[1], float(argv[2]), settings(1920, 256))
    elif argv[0] == "frames":
        me, of = int(argv[1]), int(argv[2])
        held = int(HOLD * FPS)      # what stands still is drawn once
        for i in range(int(LONG * FPS)):
            n = max(i, held)
            if n % of == me and not os.path.exists(os.path.join(SEVEN, opt("--dir", "river", str), f"f{n:05d}.png")):
                still(f"f{n:05d}", n / FPS, settings(960, 64))
