"""The press as moving pictures: every frame the real height field the press went through at that moment (film_imprint_clay.py wrote them to seven/seq_h/), never a dissolve. Seven dies, each a different stuff, come in from above the frame and only their lower ends are ever seen. Then the dawn light sweeps over what they left, and the camera comes in evenly; then the groove goes by itself.

    blender -b --factory-startup -P film_imprint_seq.py -- frames b1 0 2                  # segment (b1 to b7, dawn, groove), this worker's share of it: into seven/seq_press/b1/f00027.png, frames on 三's clock (T3)
    ... -- frame b3 2104 --full --name k3_hit                                             # one frame, at any size
    ... -- frames b1 0 1 --full --samples 48 --tag _1080                                  # the same frames at 1920, into seq_press/b1_1080/
"""
import json
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from film_imprint_clay import A, B, LAST, OUT, SEVEN, THICK, state_path
from film_imprint_clawd import finish
from film_imprint3d import DAWN_AZ, DAWN_SKY, LENS, TARGET, Nodes, block_of, board_object, camera, clay_material, corner_warp, crumbs, dawn, lights, set_up, sm
from ink3d import aim, reset

SEQ_H = os.path.join(SEVEN, "seq_h")
SEQ_OUT = os.path.join(SEVEN, "seq_press")
FPS = 30
DAWN_FRAMES = (552, 635)                # T3 18.4 to 21.2
FROM_S, TO_S, TO_T, TO_F = 1.72, 3.2, (32.0, -10.0, -8.0), 300.0     # the last blow's framing to k6_final's
GROOVE_S, GROOVE_T, GROOVE_ELEV, GROOVE_SUN = 4.4, (-16.0, 22.8, -3.0), 62.0, (28.0, 13.0)   # from well above: from the old height the relief round it hides it. The sun is up a little by now                       # g3's
SWEEP = (18.5, 19.9)
LAMPS_OUT = 18.4                        # T3: the blows' lamp goes as the morning comes
SWEEP_HIGH = 600.0
SWEEP_Q = (-300.0, 340.0)
SWEEP_TO = 350.0

# how each die's walls go up from its face: how far out from the face's edge at how far above it. All of them go on up out of the frame
WALLS = (((0, 0), (1.4, .7), (2.2, 2.2), (2.4, 6), (2.0, 24), (1.0, 30), (-6, 40), (-10, 60), (-12, 120), (-14, 1500)),
         ((0, 0), (.6, .6), (.9, 3), (1.0, 40), (1.0, 1500)),
         ((0, 0), (.5, .3), (.7, 1), (.7, 9), (.2, 10.2), (.2, 12), (.7, 13.2), (.7, 15), (-1, 18), (-8, 25), (-20, 31), (-24, 40), (-24, 1500)),
         ((0, 0), (.25, .2), (.3, .6), (.3, 23.5), (0, 24), (-.6, 24.6), (-.6, 27), (-7, 33), (-10, 40), (-10, 1500)),
         ((0, 0), (.3, .3), (.4, 3.2), (-2.5, 3.6), (-2.5, 1500)),
         ((0, 0), (1.2, 1.2), (1.2, 22), (.2, 23), (-5, 26), (-5, 1500)),
         ((0, 0), (2.5, 2.5), (3, 5), (3, 16), (1.5, 17.5), (1.5, 19), (3, 20.5), (3, 34), (-6, 44), (-14, 60), (-14, 1500)))
CORNERS = (7.0, 3.0, 5.0, 1.2, 1.0, 1.5, 8.0)
QUARTO, TYPE_HIGH, LINE_REACH, SORT_PAD = 3, 24.0, 2.6, .5     # the forme: how high its sticks of type stand, how far a line's body reaches up and down from its letters, and past its last letter


def type_lines(Z, C, lx, ly):
    """A forme of set type is lines of sticks, each line as long as its words: past a line's last letter there is no metal for the height of the type, so the block's
    sides step line by line and its bottom is not a slab. Rows between the letters belong to the nearer line."""
    has = C > .08
    ny, nx = has.shape
    found = has.any(axis=1)
    right = np.where(found, lx[nx - 1 - np.argmax(has[:, ::-1], axis=1)], -np.inf)
    left = np.where(found, lx[np.argmax(has, axis=1)], np.inf)
    n = max(int(round(LINE_REACH / abs(float(ly[1] - ly[0])))), 1)
    R, L = right.copy(), left.copy()
    for d in range(1, n + 1):
        R[d:], R[:-d] = np.maximum(R[d:], right[:-d]), np.maximum(R[:-d], right[d:])
        L[d:], L[:-d] = np.minimum(L[d:], left[:-d]), np.minimum(L[:-d], left[d:])
    gone = (lx[None, :] > R[:, None] + SORT_PAD) | (lx[None, :] < L[:, None] - SORT_PAD)
    return np.where(gone, Z + TYPE_HIGH, Z).astype(np.float32)


def die_material(k):
    """0 coarse pottery, uneven from the hand. 1 a block of pear wood, the knife's cuts in it, dark with ink. 2 old bronze: green in its hollows and over its shoulders, worn to metal where it has pressed. 3 lead: a forme of cast type, columns of it.
    4 an engraved copper plate. 5 cold bright steel, a machine's. 6 dark heavy steel."""
    N = Nodes(f"die{k}")
    t, b = N.t, N.b
    rest = t.nodes.new("ShaderNodeAttribute")
    rest.attribute_name = "rest"
    N.obj = rest.outputs["Vector"]
    cav, wall = N.attr("cav"), N.attr("wall")
    if k == 0:
        tone = N.range(N.noise(.06, 3.0, .55), .35, .65)
        col = N.mix(tone, (.5, .37, .27), (.33, .235, .165))
        col = N.mix(N.range(N.noise(3.0, 2.0, .5), .66, .72, 0, .8, True), col, (.11, .08, .06))
        col = N.mix(N.range(N.noise(5.0, 2.0, .5), .69, .73, 0, .6, True), col, (.72, .62, .5))
        col = N.mix(N.math("MULTIPLY", cav, .78), col, (.075, .052, .04))
        t.links.new(col, b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = .86
        b.inputs["Specular IOR Level"].default_value = .3
        mp = t.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (.035, .035, .5)      # where the hand went round it
        t.links.new(N.obj, mp.inputs[0])
        n = N.bump(N.noise(1.2, 4.0, .6), .4, 1.0)
        n = N.bump(N.noise(.14, 3.0, .5), 1.1, 1.0, n)
        t.links.new(N.bump(N.noise(1.0, 3.0, .55, vec=mp.outputs[0]), 1.0, 1.0, n), b.inputs["Normal"])
    elif k == 1:
        mp = t.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (.6, .6, .02)       # the grain runs up the block
        t.links.new(N.obj, mp.inputs[0])
        grain = N.noise(1.0, 8.0, .6, vec=mp.outputs[0])
        streak = N.range(grain, .36, .66, 0, 1, True)
        col = N.mix(streak, (.2, .13, .075), (.1, .062, .036))
        col = N.mix(N.math("MULTIPLY", cav, .9), col, (.035, .03, .028))       # old ink down in the cuts
        col = N.mix(N.math("MULTIPLY", N.math("SUBTRACT", 1.0, wall), N.math("SUBTRACT", 1.0, cav)), col, (.05, .042, .038))     # and on the printing face
        t.links.new(col, b.inputs["Base Color"])
        t.links.new(N.range(grain, .3, .7, .45, .7), b.inputs["Roughness"])
        b.inputs["Specular IOR Level"].default_value = .4
        t.links.new(N.bump(N.noise(3.0, 4.0, .55, vec=mp.outputs[0]), .08, 1.0), b.inputs["Normal"])
    elif k == 2:
        crust = N.range(N.noise(.05, 5.0, .62), .38, .66, 0, 1, True)
        green = N.math("MINIMUM", N.math("ADD", N.math("ADD", N.math("MULTIPLY", cav, .95), N.math("MULTIPLY", wall, .42)), N.math("MULTIPLY", crust, .5)), 1.0)
        green = N.range(green, .3, .78, 0, 1, True)
        metal = N.mix(N.range(N.noise(.4, 3.0, .5), .35, .65), (.56, .34, .14), (.3, .165, .065))
        verd = N.mix(N.range(N.noise(.5, 4.0, .6), .36, .64), (.135, .29, .24), (.05, .125, .105))
        t.links.new(N.mix(green, metal, verd), b.inputs["Base Color"])
        t.links.new(N.math("SUBTRACT", 1.0, green), b.inputs["Metallic"])
        t.links.new(N.range(green, 0, 1, .36, .82), b.inputs["Roughness"])
        t.links.new(N.bump(N.math("MULTIPLY", N.noise(1.5, 4.0, .6), green), .3, 1.0), b.inputs["Normal"])
    elif k == 3:
        mott = N.range(N.noise(.05, 3.0, .55), .36, .64)
        col = N.mix(mott, (.31, .32, .335), (.2, .205, .22))
        col = N.mix(N.math("MULTIPLY", N.range(N.noise(.7, 3.0, .5), .5, .66, 0, 1, True), N.math("ADD", N.math("MULTIPLY", cav, .6), .12)), col, (.5, .5, .49))
        t.links.new(col, b.inputs["Base Color"])
        b.inputs["Metallic"].default_value = .8
        t.links.new(N.range(mott, 0, 1, .46, .64), b.inputs["Roughness"])
        col_run = N.math("ABSOLUTE", N.math("SINE", N.math("MULTIPLY", N.attr("run"), math.pi / 2.4)))
        n = N.bump(N.math("MULTIPLY", N.math("POWER", col_run, .22), N.attr("type")), .22, 1.0)     # the sticks of type end where the casting they are locked in begins
        t.links.new(N.bump(N.noise(2.5, 3.0, .5), .05, 1.0, n), b.inputs["Normal"])
    elif k == 4:
        b.inputs["Metallic"].default_value = 1.0
        tar = N.range(N.noise(.03, 6.0, .6), .42, .62, 0, 1, True)
        col = N.mix(tar, (.62, .36, .22), (.2, .11, .07))
        col = N.mix(N.math("MULTIPLY", cav, .85), col, (.05, .035, .03))     # ink left in the engraving
        t.links.new(col, b.inputs["Base Color"])
        t.links.new(N.range(N.noise(.5, 5.0, .55), .3, .7, .22, .4), b.inputs["Roughness"])
        vec = t.nodes.new("ShaderNodeMapping")
        vec.inputs["Scale"].default_value = (.05, 1.0, 12.0)
        t.links.new(N.obj, vec.inputs[0])
        t.links.new(N.bump(N.noise(3.0, 3.0, .5, vec=vec.outputs[0]), .02, 1.0), b.inputs["Normal"])
    elif k == 5:
        mp = t.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (.04, 7.0, 7.0)
        t.links.new(N.obj, mp.inputs[0])
        streak = N.range(N.noise(1.0, 3.0, .5, vec=mp.outputs[0]), .3, .7)
        t.links.new(N.mix(streak, (.7, .735, .77), (.58, .61, .65)), b.inputs["Base Color"])
        b.inputs["Metallic"].default_value = 1.0
        t.links.new(N.range(streak, 0, 1, .22, .32), b.inputs["Roughness"])
        t.links.new(N.bump(streak, .012, 1.0), b.inputs["Normal"])
    else:
        blot = N.range(N.noise(.025, 3.0, .55), .36, .64)
        col = N.mix(blot, (.13, .135, .145), (.07, .073, .08))
        col = N.mix(N.math("MULTIPLY", N.range(N.noise(.3, 3.0, .5), .58, .68, 0, 1, True), .6), col, (.05, .037, .03))
        t.links.new(col, b.inputs["Base Color"])
        b.inputs["Metallic"].default_value = 1.0
        t.links.new(N.range(blot, 0, 1, .34, .5), b.inputs["Roughness"])
        t.links.new(N.bump(N.noise(.11, 3.0, .5), .35, 1.0), b.inputs["Normal"])
    return N.m


def die_object(k, pose, f, mat, shallow=1.0):
    """The die: its real face (the one that made the marks), f cells of its grid to one here, and its walls going up out of the frame. pose as the press wrote it. shallow: for the low shot, so tall a relief hides its own signs seen from the side."""
    d = np.load(os.path.join(SEQ_H, f"die_{k}.npz"))
    res, w, h = float(d["res"]), float(d["w"]), float(d["h"])
    face, cav = d["face"], d["cav"]
    ny, nx = face.shape[0] // f, face.shape[1] // f
    cut = lambda a: a[:ny * f, :nx * f].reshape(ny, f, nx, f).mean(axis=(1, 3)).astype(np.float32)
    Z, C = cut(face) * shallow, cut(cav)
    lx, ly = (np.arange(nx) + .5) * f / res - w / 2, h / 2 - (np.arange(ny) + .5) * f / res
    if k == QUARTO:
        Z = type_lines(Z, C, lx, ly)
    X, Y = np.meshgrid(lx, ly)
    r = CORNERS[k]
    ax, ay = float(lx[-1]), float(ly[0])
    p, q = np.clip((np.abs(X) - (ax - r)) / r, 0, 1), np.clip((np.abs(Y) - (ay - r)) / r, 0, 1)
    pp, qq = p * np.sqrt(1 - q * q / 2), q * np.sqrt(1 - p * p / 2)
    X, Y = np.where(p > 0, np.sign(X) * (ax - r + pp * r), X), np.where(q > 0, np.sign(Y) * (ay - r + qq * r), Y)
    idx = np.arange(nx * ny, dtype=np.int32).reshape(ny, nx)
    loop = np.concatenate([idx[0, :-1], idx[:-1, -1], idx[-1, :0:-1], idx[:0:-1, 0]])
    px, py, pz = X.ravel()[loop], Y.ravel()[loop], Z.ravel()[loop]
    dx, dy = np.roll(px, -1) - np.roll(px, 1), np.roll(py, -1) - np.roll(py, 1)
    ln = np.hypot(dx, dy) + 1e-6
    ox, oy = -dy / ln, dx / ln
    run = np.cumsum(ln / 2)
    n = 13
    level = np.convolve(np.concatenate([pz[-n:], pz, pz[:n]]), np.ones(2 * n + 1) / (2 * n + 1), "valid")
    rows, kept = [], []
    for out, up in WALLS[k][1:]:
        o = out + np.zeros_like(px)
        if k == 0:      # the potter's hands
            o = o + (.9 * np.sin(run * .071 + up * .05) + .45 * np.sin(run * .19 + 1.0 + up * .11)) * min(up / 2.0, 1)
        rows.append(np.stack([px + ox * o, py + oy * o, level + up], -1))
        kept.append((level + up <= TYPE_HIGH + pz.min()) if k == QUARTO else np.full(len(px), up <= TYPE_HIGH))     # where the sticks of type are, and above them the casting
    side = np.stack(rows)
    co = np.concatenate([np.stack([X, Y, Z], -1).reshape(-1, 3), side.reshape(-1, 3)]).astype(np.float64)
    base = nx * ny
    nr, npm = side.shape[:2]
    rings = np.vstack([loop[None, :], base + np.arange(nr * npm, dtype=np.int32).reshape(nr, npm)])
    a = idx
    top_q = np.stack([a[:-1, :-1], a[:-1, 1:], a[1:, 1:], a[1:, :-1]], -1).reshape(-1, 4)
    side_q = np.stack([rings[:-1], rings[1:], np.roll(rings[1:], -1, axis=1), np.roll(rings[:-1], -1, axis=1)], -1).reshape(-1, 4)
    quads = np.concatenate([top_q, side_q])
    rest = co.astype(np.float32).copy()
    # into the world: leaned towards the low camera, the rocking of the last one, turned, set down
    th = math.radians(-pose["tilt"])
    lx_, ly_, lz_ = co[:, 0], co[:, 1], co[:, 2]
    ly2, lz2 = ly_ * math.cos(th) - lz_ * math.sin(th), ly_ * math.sin(th) + lz_ * math.cos(th)
    lz2 = lz2 + (lx_ - pose["tx"]) * pose["sx"] + (ly_ - pose["ty"]) * pose["sy"]
    t = math.radians(pose["a"])
    wx = pose["x"] + lx_ * math.cos(t) - ly2 * math.sin(t)
    wy = pose["y"] + lx_ * math.sin(t) + ly2 * math.cos(t)
    wz = pose["z"] + pose["up"] + lz2
    P = np.stack([wx, wy, wz], -1).astype(np.float32)
    me = bpy.data.meshes.new("die")
    me.vertices.add(len(P))
    me.vertices.foreach_set("co", P.ravel())
    me.loops.add(quads.size)
    me.loops.foreach_set("vertex_index", quads.ravel())
    me.polygons.add(len(quads))
    me.polygons.foreach_set("loop_start", np.arange(len(quads), dtype=np.int32) * 4)
    me.update(calc_edges=True)
    me.shade_smooth()
    at = me.attributes.new("rest", "FLOAT_VECTOR", "POINT")
    at.data.foreach_set("vector", rest.ravel())
    n_side = nr * npm
    for name, v in (("cav", np.concatenate([C.ravel(), np.zeros(n_side, np.float32)])), ("wall", np.concatenate([np.zeros(base, np.float32), np.ones(n_side, np.float32)])), ("run", np.concatenate([np.zeros(base, np.float32), np.tile(run, nr).astype(np.float32)])), ("type", np.concatenate([np.zeros(base, np.float32), np.concatenate(kept).astype(np.float32)]))):
        at = me.attributes.new(name, "FLOAT", "POINT")
        at.data.foreach_set("value", v.astype(np.float32))
    o = bpy.data.objects.new("die", me)
    bpy.context.collection.objects.link(o)
    me.materials.append(mat)
    return o


# ---- frames ------------------------------------------------------------------------------------------------------------------------------------------------
def small2(h):
    ny, nx = h.shape
    return h[:ny // 2 * 2, :nx // 2 * 2].reshape(ny // 2, 2, nx // 2, 2).mean(axis=(1, 3)).astype(np.float32)


def manifest(name):
    with open(os.path.join(SEQ_H, name, "manifest.json")) as fp:
        return json.load(fp)


def segment(name):
    """The film frames of a segment, in order."""
    if name.startswith("b"):
        return [fr["f"] for fr in manifest(name)["frames"]]
    if name == "dawn":
        return list(range(DAWN_FRAMES[0], DAWN_FRAMES[1] + 1))
    return [fr["f"] for fr in manifest("g")["frames"]]


SIDE = {}


def side_data(name):
    if name not in SIDE:
        SIDE.clear()
        z = np.load(os.path.join(SEQ_H, name, "side.npz"))
        SIDE[name] = {k: z[k] for k in z.files}
    return SIDE[name]


def pressed(name, fr, mat, plain=None, near=None, dense=None, NT=30):
    """The block as it was in this frame of a blow."""
    S = side_data(name)
    w = fr["w"]
    H = np.load(os.path.join(SEQ_H, name, fr["h"]))
    burnish = S["burnish0"] + (S["burnish1"] - S["burnish0"]) * w
    bulge = S["bulge0"] + (S["bulge1"] - S["bulge0"]) * w
    lost = float(S["lost0"]) + (float(S["lost1"]) - float(S["lost0"])) * w
    if dense:
        return block_of(H, 16, lost, burnish, bulge, None, dense, mat, near=near, NT=NT)
    return block_of(H, 16, lost, burnish, bulge, None, None, mat, plain=plain, NT=NT)


BASE5 = {}


def grooved(fr, mat, dense, near):
    """The block with as much of the groove as had come by this frame."""
    if not BASE5:
        z = np.load(state_path(LAST))
        BASE5.update(H=small2(z["H"]), burnish=z["burnish"], bulge=z["bulge"], lost=float(z["lost"]))
    wx0, wx1, wy0, wy1 = manifest("g")["window"]
    H = BASE5["H"].copy()
    fresh = np.zeros_like(BASE5["burnish"], dtype=np.float32)
    if fr["h"]:
        res = 16
        r0, c0 = int((B - wy1) * res), int((wx0 + A) * res)
        patch = small2(np.load(os.path.join(SEQ_H, "g", fr["h"])))
        H[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]] = patch
        wet = np.load(os.path.join(SEQ_H, "g", "fresh" + fr["h"][1:]))
        fresh[r0:r0 + wet.shape[0], c0:c0 + wet.shape[1]] = wet
    return block_of(H, 16, BASE5["lost"], BASE5["burnish"], BASE5["bulge"], fresh, dense, mat, near=near, NT=30)


def view_window(s, target, px):
    """What the camera sees of the block's top, with a margin: the mesh is fine there and coarse elsewhere."""
    half_w = 810.0 / s * (36.0 / LENS) / 2 * 1.2
    half_h = half_w * 9 / 16 / math.sin(math.radians(30.0)) * 1.1
    win = (max(target[0] - half_w, -A + 1), min(target[0] + half_w, A - 1), max(target[1] - half_h, -B + 1), min(target[1] + half_h, B - 1))
    return win, (.5, px, px * 1.4)


LOW_TILT, LOW_DOWN, LOW_BACK, LOW_SHALLOW = 1.6, -25.0, 1.2, .55       # the low shot, against what the press wrote: leaned further, the camera below the top of the block, further off


def low_camera(sc, die, low, fstop):
    D, zt = low[0] * LOW_BACK, low[2]          # low: (how far off, -, the height looked at, the die's face's height): the die's whole face in, with room above it
    p, t = (die["x"], die["y"] - D, LOW_DOWN), (die["x"], die["y"], zt)
    data = bpy.data.cameras.new("cam")
    data.lens = LENS
    data.sensor_width = 36.0
    data.clip_start, data.clip_end = 5.0, 20000.0
    data.dof.use_dof = True
    data.dof.focus_distance = D
    data.dof.aperture_fstop = fstop * 2 / 1000.0
    o = bpy.data.objects.new("cam", data)
    bpy.context.collection.objects.link(o)
    aim(o, p, t)
    sc.camera = o


def face_lamp(die, zair):
    """A broad lamp low down, near the camera and a little to the left, for the half second its face is shown: enough to read the signs by."""
    lamp = bpy.data.lights.new("face", "AREA")
    lamp.size = 260.0
    lamp.energy = FACE_E
    lamp.color = (1.0, .95, .88)
    o = bpy.data.objects.new("face", lamp)
    bpy.context.collection.objects.link(o)
    aim(o, (die["x"] - 110.0, die["y"] - 420.0, -28.0), (die["x"], die["y"], zair))
    o.visible_camera = False


FACE_E = 1.3e6


def metal_card():
    """Something large and pale up behind the camera, seen only in what shines: a metal wall with nothing to reflect is just a black wall."""
    card = bpy.data.lights.new("card", "AREA")
    card.shape = "RECTANGLE"
    card.size, card.size_y = 1800.0, 900.0
    card.energy = 1.5e6
    card.color = (.93, .96, 1.0)
    o = bpy.data.objects.new("card", card)
    bpy.context.collection.objects.link(o)
    aim(o, (-500.0, -1300.0, 1100.0), (0.0, 0.0, 60.0))
    o.visible_diffuse = False
    o.visible_camera = False


def sweep(sc, t):
    """The first light of the morning, coming over the block from the side the sun is on: the shadow of something high and far off draws back, so its edge goes across. The lamp of the blows goes out as it comes."""
    u = float(sm((t - SWEEP[0]) / (SWEEP[1] - SWEEP[0])))
    if u < 1:
        q = SWEEP_Q[0] + (SWEEP_Q[1] - SWEEP_Q[0]) * u
        a, e = math.radians(DAWN_AZ), math.radians(15.0)
        back = SWEEP_HIGH / math.tan(e)
        dx, dy = math.cos(a), -math.sin(a)
        c0 = (q - back) * np.array((dx, dy))
        px, py = -dy, dx
        wide = SWEEP_TO - q         # only as far as the sun's rays to the block need it: the lamps' rays go by it
        co = [(c0[0] + s * px * 5000.0 + l * dx * wide, c0[1] + s * py * 5000.0 + l * dy * wide, SWEEP_HIGH) for s, l in ((-1, 0), (1, 0), (1, 1), (-1, 1))]
        me = bpy.data.meshes.new("night")
        me.from_pydata(co, [], [(0, 1, 2, 3)])
        me.update()
        o = bpy.data.objects.new("night", me)
        bpy.context.collection.objects.link(o)
        o.visible_camera = False
        o.visible_diffuse = False
        o.visible_glossy = False
    bpy.data.lights["gold"].energy *= float(sm((t - SWEEP[0] - .5) / (SWEEP[1] - SWEEP[0])))
    bpy.data.lights["dawn"].angle = math.radians(.7 + .9 * u)      # a harder edge while it goes across, and it softens to the sun's own


def lamps_down(sc, t):
    k = 1 - float(sm((t - LAMPS_OUT) / .9))
    for name in ("side", "sheen", "bounce"):
        bpy.data.lights[name].energy *= k
    return k


def frame(seg, f, a):
    t = f / FPS
    sc = reset(a["samples"], 100)
    set_up(a, sc)
    sc.cycles.max_bounces, sc.cycles.diffuse_bounces = 5, 2
    mat = clay_material()
    px = .357 * 960.0 / a["w"]
    name_b = None
    if seg.startswith("b") or (seg == "dawn" and f <= round(18.65 * FPS)):        # the last die is still going out of the top of the frame
        name_b = seg if seg.startswith("b") else f"b{LAST}"
        m = manifest(name_b)
        fr = next((x for x in m["frames"] if x["f"] == f), None)
    k_die = int(name_b[1:]) - 1 if name_b else None
    if seg.startswith("b"):
        low = fr["cam"] == "low"
        s = m["scale"]
        step = max(px / s, 1 / 16)
        pressed(seg, fr, mat, plain=(step * (1.0 if not low else .9), step * (1.4 if not low else 2.2)), NT=30)
        lights(38.0)
        if low:
            low_camera(sc, fr["die"], m["low"], a["fstop"])
            face_lamp(fr["die"], m["low"][3])
            bg = sc.world.node_tree.nodes["Background"]
            bg.inputs[1].default_value = .28
            die_object(k_die, dict(fr["die"], tilt=fr["die"]["tilt"] * LOW_TILT), 1, die_material(k_die), LOW_SHALLOW)
        else:
            camera(sc, s, a["fstop"], shake=tuple(fr["shake"]))
            if k_die >= 2:
                metal_card()
            die_object(k_die, fr["die"], max(int(round(a["die_f"])), 1), die_material(k_die))
    elif seg == "dawn":
        u = (f - DAWN_FRAMES[0]) / (DAWN_FRAMES[1] + 1 - DAWN_FRAMES[0])
        s = FROM_S * (TO_S / FROM_S) ** u
        # about one fixed point, so that nothing swerves: the view's centre goes evenly with the inverse of the size
        r0, c = FROM_S / TO_S, []
        for T0, T1 in zip(TARGET, TO_T):
            c.append((T1 - T0 * r0) / (1 - r0))
        target = tuple(cc + (T0 - cc) * FROM_S / s for cc, T0 in zip(c, TARGET))
        win, near = view_window(s, target, max(px / s, 1 / 16))
        if name_b:
            pressed(f"b{LAST}", fr, mat, dense=win, near=near)
            metal_card()
            die_object(LAST - 1, fr["die"], 8, die_material(LAST - 1))
        else:
            pressed(f"b{LAST}", dict(h="after.npy", w=1.0), mat, dense=win, near=near)
        dawn(sc, DAWN_AZ)
        sweep(sc, t)
        lights(38.0)
        k = lamps_down(sc, t)
        bg = sc.world.node_tree.nodes["Background"]
        for i in range(3):
            bg.inputs[0].default_value[i] = DAWN_SKY[i] * (1 - k) + (.070, .062, .052)[i] * k
        bg.inputs[1].default_value = 1.1 - .3 * k
        camera(sc, s, 22.0 * (TO_F / 22.0) ** u, target=target)
    else:
        fr = next(x for x in manifest("g")["frames"] if x["f"] == f)
        win, near = view_window(GROOVE_S, GROOVE_T, max(px / GROOVE_S, 1 / 16))
        grooved(fr, mat, win, near)
        dawn(sc, DAWN_AZ, *GROOVE_SUN)
        camera(sc, GROOVE_S, TO_F, elev=GROOVE_ELEV, target=GROOVE_T)
    board_object()
    crumbs(mat)
    path = os.path.join(SEVEN, a["name"] + ".png") if a.get("name") else os.path.join(SEQ_OUT, seg + a.get("tag", ""), f"f{f:05d}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sc.render.image_settings.file_format, sc.render.image_settings.color_depth = "PNG", "16"
    deep = f"/tmp/seq16_{os.getpid()}.png"
    sc.render.filepath = deep
    bpy.ops.render.render(write_still=True)
    finish(deep, path, f, a["grain"])
    print("wrote", path, flush=True)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]

    def opt(name, default, kind=float):
        return kind(argv[argv.index(name) + 1]) if name in argv else default

    full = "--full" in argv
    settings = dict(samples=opt("--samples", 256 if full else 48, int), w=opt("--w", 1920 if full else 960, int), h=opt("--h", 1080 if full else 540, int), fstop=opt("--fstop", 22.0), exp=.5, denoise=True, grain=opt("--grain", 1.0), die_f=opt("--die-f", 2 if full else 6), name=opt("--name", None, str), tag=opt("--tag", "", str))     # --tag _1080: the frames into seq_press/SEG_1080/
    if argv[0] == "frames":
        seg, me, of = argv[1], int(argv[2]), int(argv[3])
        for i, f in enumerate(segment(seg)):
            if i % of == me and not os.path.exists(os.path.join(SEQ_OUT, seg + settings["tag"], f"f{f:05d}.png")):
                frame(seg, f, settings)
    elif argv[0] == "frame":
        frame(argv[1], int(argv[2]), settings)
    print("finished", flush=True)
