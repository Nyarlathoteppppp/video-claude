"""The little crab's poses for the credits, the list in head/crab_poses_needed.md: each a transparent picture with the same camera, the same light and the same canvas, made from his real mesh and read as they are by film_credits.py. Fat and ball are the same skin on its way to a sphere, so his marks, his eyes and his scar stay on it; the claws and legs sink in, the feet stay where they are, and he grows up and out from there.

    blender -b --factory-startup -P film_imprint_poses.py -- all                 # into head/crab/; a picture already there is drawn again
    ... -- some fat3 ball spin5
    ... -- test crab/poses_test all --draft                                       # into imprint/crab/poses_test/

The floor is still there for the light it throws up at him; only the camera can't see it.
"""
import math
import os
import sys
from types import SimpleNamespace

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import film_imprint_clawd as c
from film_imprint_clawd import BODY_D, BODY_H, BODY_W, EYE_HW, FLOOR, LEGS, LEG_H, LEG_Y, U, ZB, ZM, Clawd, curve, put_mesh, waves
from film_imprint3d import set_up, sm
from film_imprint_clay import OUT
from ink3d import aim, reset

T = 300.0                         # long after anything he does on the bridge clock: all that is left is what is held here
CANVAS = 1000
LENS, DIST, ELEV, AT_Z = 122.0, 1060.0, 3.0, 60.0      # the bridge's distance; 3.2 px to the mm, the canvas 312 mm each way
BALL_R, BALL_GAP = 96.0, 3.0      # how big the ball is, and how far off the floor its feet hold it
LOOK_DOWN = 14.0                  # as a ball he has no neck: his face just goes round a little
UNDER = ((0.0, 25.3, 42.2, 52.3, 69.2, 81.0), (0.0, 13.6, 30.4, 36.6, 53.4, 66.0))     # across, on his underside: as he stands, and on the ball
TILT_LINE = (AT_Z + DIST * math.sin(math.radians(ELEV)) - FLOOR) / (DIST * math.cos(math.radians(ELEV)))      # how far above the floor at the foot of the ball the camera's line through it runs, for each mm towards the camera
UNDER_LIGHT = 1.2
HELD = dict(BOW=0.0, STRETCH=0.0, CHEW=0.0, MUNCH=0.0, CHEEKS=0.0, BREATH=0.0, EAT=0.0, YAWN=0.0, SQUINT=0.0, RAISE=0.0, BLINK=0.0, YAW=0.0, SLUMP=0.0, WAVE=0.0)


def spin(i, of=16):
    return dict(BREATH=1.0, m=1.0, yaw=360.0 * i / of)


def fat(m):
    return dict(BREATH=1.0, BOW=.35 * (1 - m), m=m)


POSES = dict(
    front=dict(),
    down=dict(BOW=.72),
    up=dict(BOW=-.45),
    bite=dict(EAT=.55, BOW=.85),
    inhale=dict(BREATH=1.0, BOW=.75),
    chewa=dict(MUNCH=1.0, CHEW=.5, BOW=.25, CHEEKS=.2, GO=(T - .25, 1.0)),
    chewb=dict(MUNCH=1.0, CHEW=.5, BOW=.25, CHEEKS=.2, GO=(T + .25, 1.0)),
    **{f"fat{i + 1}": fat(m) for i, m in enumerate((.14, .28, .42, .58, .74, .88))},
    ball=fat(1.0),
    **{f"spin{i + 1}": spin(i) for i in range(16)},
    ball_stop=dict(MUNCH=1.0, GO=(T, 1.0), m=1.0, yaw=3.0, lean=1.0),
    ball_chewa=dict(MUNCH=1.0, CHEW=.35, CHEEKS=.4, GO=(T - .25, 1.0), m=1.0, sz=.955, sx=1.025, yaw=3.5, lean=1.2),
    ball_chewb=dict(MUNCH=1.0, CHEW=.35, CHEEKS=.4, GO=(T + .25, 1.0), m=1.0, sz=.955, sx=1.025, yaw=-3.5, lean=-1.2),
    ball_swallow=dict(MUNCH=1.0, GO=(T, 1.0), m=1.0, r=.96, throat=10.0),
    pop1=dict(MUNCH=1.0, GO=(T, 1.0), m=.72, sz=1.06, sx=.98),
    pop2=dict(RAISE=.4, m=.4, sz=.9, sx=1.06),
    pop3=dict(RAISE=.12, m=.12, sz=1.1, sx=.95),
    pop4=dict(m=-.05, sz=.97, sx=1.03),
    leap_a=dict(CHEW=1.6, BOW=.2, RAISE=-.3),
    leap_b=dict(STRETCH=.5, RAISE=1.0, legs=1.25),
    leap_c=dict(CHEW=2.4, BOW=.15, SQUINT=.5, RAISE=-.3),
    bow=dict(CHEW=1.75, BOW=-.35),
    **{f"walk{j + 1}": dict(walk=j, RAISE=.08, yaw=18.0, lean=2.0 + 1.2 * math.sin(math.pi * j / 2)) for j in range(4)},
)


def held(p):
    """Everything he does on the bridge clock, stopped at these values."""
    for name, v in HELD.items():
        v = p.get(name, v)
        setattr(c, name, curve((0.0, v), (1000.0, v)))
    c.GO = p.get("GO", (T, 1.0))


def on_ball(x, y, z):
    """The direction of a point of his body (from the middle of it) on the ball: as if he were as deep as the ball is round, so that his face keeps its size and proportions. Low down the bottom of him is drawn together, his legs closer, each as wide as it was."""
    low = sm((-z - 25.0) / 29.0)
    x = x + (np.sign(x) * np.interp(np.abs(x), *UNDER) - x) * low
    v = np.stack([x, y * (BALL_R / (BODY_D / 2)) * (1 - .75 * low), z * (1 + low)], -1)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    a = math.radians(LOOK_DOWN) * (1 - low)
    cs, sn = np.cos(a), np.sin(a)
    return np.stack([v[:, 0], v[:, 1] * cs - v[:, 2] * sn, v[:, 1] * sn + v[:, 2] * cs], -1)


def to_ball(k, fine):
    """Where each point of his skin goes on the ball. A limb goes to the place its root was. The clay has stretched to get there, so all but his face is worked over again by the same hands, at the same size as on him."""
    Q = k.R
    v = on_ball(np.clip(Q[:, 0], -BODY_W / 2, BODY_W / 2), Q[:, 1], np.clip(Q[:, 2] - ZM, -BODY_H / 2, BODY_H / 2))
    at = (v * BALL_R).astype(np.float32)
    skin = SimpleNamespace(R=at + np.array((0.0, 0.0, ZM), np.float32), N=v.astype(np.float32))
    marks = waves(at, 90.0, 15) * 1.2 + waves(at, 52.0, 11) * .9 + waves(at, 20.0, 12) * .22 + waves(at, 7.0, 13) * .04 + Clawd.pinches(skin, 240) + Clawd.prints(skin, fine)
    a = math.radians(LOOK_DOWN)
    his_face = sm((v @ np.array((0.0, -math.cos(a), -math.sin(a))) - .62) / .2)
    k.ball, k.ball_at, k.ball_static = v.astype(np.float32), at, (k.static * his_face + marks * (1 - his_face)).astype(np.float32)
    k.leg = np.clip((ZB - Q[:, 2]) / LEG_H, 0, 1).astype(np.float32)
    roots = on_ball(np.array(LEGS, np.float64), np.full(len(LEGS), LEG_Y), np.full(len(LEGS), -BODY_H / 2)) * BALL_R
    k.leg_root = roots[np.argmin(np.abs(Q[:, 0, None] - np.array(LEGS)), axis=1), :2].astype(np.float32)


def sphere(k, disp, r=1.0, throat=0.0):
    """His skin on the ball, the legs stubs down to the floor."""
    size = BALL_R * r + k.ball_static + disp
    if throat:      # a swallow going down: a bump in the throat, below the mouth
        to = np.array((0.0, -.8, -.6))
        size = size + throat * np.exp(-(np.arccos(np.clip(k.ball @ (to / np.linalg.norm(to)), -1, 1)) / .42) ** 2)
    P = k.ball * size[:, None]
    P[:, 2] += FLOOR + BALL_GAP + BALL_R
    P[:, :2] += (k.leg_root - P[:, :2]) * .2 * k.leg[:, None]
    P[:, 2] += (FLOOR - P[:, 2]) * k.leg
    return P


def stride(k, j):
    """One of the four moments of his walk: sideways, as a crab goes, to our right. Every other leg is off the floor, going forward, while the others push back. His body rides a little higher as they pass."""
    Q = k.R
    w = np.clip((ZB + 2.0 - Q[:, 2]) / LEG_H, 0, 1) ** .7
    d = np.zeros((len(Q), 3), np.float32)
    for i, lx in enumerate(LEGS):
        on = np.abs(Q[:, 0] - lx) < U
        phase = j / 4 + (i % 2) / 2
        d[on, 0] = (-5.0 * math.cos(2 * math.pi * phase) * w)[on]     # any further and a pair of legs runs together
        d[on, 2] = (9.0 * max(math.sin(2 * math.pi * phase), 0) * w)[on]
    d[:, 2] += (2.5 * (j % 2)) * np.clip((Q[:, 2] - FLOOR) / LEG_H, 0, 1)
    return d


def glaze_turned_away(mat, m):
    """On the ball his eyes face the sun as they never do on him, and it shows in the glaze as a white smear above the one glint: it only reflects a quarter as much here."""
    b = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    add = b.inputs["Specular IOR Level"].links[0].from_node
    add.inputs[1].links[0].from_node.inputs[1].default_value = .5 * (1 - .75 * min(max(m, 0) * 2, 1))


def from_below():
    """His underside faces a floor nobody sees and would go black against the dark of the credits, where film_credits.py would take the biggest dark place low on him for his mouth. A little warm light from the bench, through the floor: it cannot get into the mouth, whose colour is nearly black anyway."""
    d = bpy.data.lights.new("from_below", "SUN")
    d.energy, d.angle, d.color = UNDER_LIGHT, math.radians(40.0), (1.0, .78, .62)
    o = bpy.data.objects.new("from_below", d)
    bpy.context.collection.objects.link(o)
    aim(o, (0.0, -700.0, -2000.0), (0.0, 0.0, 0.0))


def camera(sc):
    from mathutils import Matrix
    e = math.radians(ELEV)
    p = np.array((0.0, -DIST * math.cos(e), AT_Z + DIST * math.sin(e)))
    data = bpy.data.cameras.new("cam")
    data.lens, data.sensor_width = LENS, 36.0
    data.clip_start, data.clip_end = 5.0, 60000.0
    data.dof.use_dof = False
    o = bpy.data.objects.new("cam", data)
    bpy.context.collection.objects.link(o)
    o.matrix_world = Matrix(((1, 0, 0, p[0]), (0, math.sin(e), -math.cos(e), p[1]), (0, math.cos(e), math.sin(e), p[2]), (0, 0, 0, 1)))
    sc.camera = o


def draw(k, name, folder, draft):
    p = POSES[name]
    sc = reset(48 if draft else 256, 100)
    set_up(dict(w=CANVAS, h=CANVAS, denoise=True, exp=.5), sc)
    sc.cycles.max_bounces, sc.cycles.diffuse_bounces = 8, 4
    sc.render.film_transparent = True
    sc.render.image_settings.file_format, sc.render.image_settings.color_mode, sc.render.image_settings.color_depth = "PNG", "RGBA", "8"
    held(p)
    m = p.get("m", 0.0)
    c.EYE_HW = EYE_HW * (1 + .28 * max(m, 0))      # the ball turns his face away from us at the eyes: they stay as wide as they looked
    disp, eye_d, mouth_d, glint_d = c.face(k, T)
    P = k.R + k.N * (k.static + disp)[:, None]
    if "walk" in p:
        P = P + stride(k, p["walk"])
    if "legs" in p:     # in the air they hang, and spread
        below = np.clip((ZB - k.R[:, 2]) / LEG_H, 0, 1)
        P[:, 2] -= (p["legs"] - 1) * LEG_H * below
        P[:, 0] += 3.0 * np.sign(k.R[:, 0]) * below
    P = c.pose(P, T)
    rest = k.R
    if m:
        P = P * (1 - m) + sphere(k, disp, p.get("r", 1.0), p.get("throat", 0.0)) * m
        rest = k.R * (1 - m) + k.ball_at * m
    psi = math.radians(p.get("yaw", 0.0))
    x, y = P[:, 0] * math.cos(psi) - P[:, 1] * math.sin(psi), P[:, 0] * math.sin(psi) + P[:, 1] * math.cos(psi)
    z = P[:, 2] - FLOOR
    wide = p.get("sx", 1.0)
    z = z * p.get("sz", 1.0)
    if m == 1.0:    # turning, nothing of him goes below the line his feet stand on as the camera sees it, so that the ball stays where it is on film_credits.py's floor
        z = np.maximum(z, TILT_LINE * np.maximum(-y, 0))
    P = np.stack([x * wide + z * math.tan(math.radians(p.get("lean", 0.0))), y * wide, FLOOR + z], -1).astype(np.float32)
    mat = c.clay()
    glaze_turned_away(mat, m)
    put_mesh("clawd", P, k.quads, dict(rest=rest, eye=eye_d, mouth=mouth_d, glint=glint_d, deep=k.deep, warm=c.warmth(k, T)), mat)
    c.floor(T, 3)
    for n in ("board", "beyond", "near"):
        bpy.data.objects[n].visible_camera = False
        bpy.data.objects[n].visible_shadow = False
    c.light(sc, T)
    from_below()
    camera(sc)
    path = os.path.join(folder, name + ".png")
    os.makedirs(folder, exist_ok=True)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("wrote", path, "height %.1f width %.1f mm" % (float(P[:, 2].max() - P[:, 2].min()), float(P[:, 0].max() - P[:, 0].min())), flush=True)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]
    draft = "--draft" in argv
    if draft:
        argv.remove("--draft")
    k = Clawd(.5 if draft else .25, not draft)
    to_ball(k, not draft)
    if argv[0] == "test":
        folder, names = os.path.join(OUT, argv[1]), argv[2:]
    else:
        folder, names = os.path.join(os.path.dirname(OUT), "head", "crab"), argv[1:]
    names = list(POSES) if names == ["all"] or argv[0] == "all" else names
    for name in names:
        draw(k, name, folder, draft)
