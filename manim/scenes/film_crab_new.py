"""十三 新东西: the four sheets of 十二 lying on the table at night, a blank between 神 and 鬼. The little crab comes into the blank and sits down there, his face to us and a little turned towards the screen behind us, whose light on him comes and goes as the reply goes on being written. The screen itself is never seen: it is ours, for the reverse shot at the end. He is a small lump of clay among four big characters, and he writes nothing.

    python film_crab_new.py papers                                               # render env: the four sheets as pictures with torn edges, into thirteen/
    blender -b --factory-startup -P film_crab_new.py -- still n13_a 9.0 [--draft] [--empty]      # name, seconds into the shot: into thirteen/
    blender -b --factory-startup -P film_crab_new.py -- frames shot 0 2                          # into thirteen/shot/: this worker's share (0 of 2) of the frames

The shot, 20.4 s on a clock of its own (the chapter's): 0 to 2.4 the four sheets, empty, from low down; 2.4 to 7.6 he comes in from our right, sideways as a crab goes, along the far side of the sheets, and turns in through the blank to its middle, slowing and turning a little towards the screen from 6.5; his feet, which never slide, step back under him by 7.9; 7.9 to 8.65 he sits down and settles, the four sheets still all in the frame; 11.0 to 15.6 one steady push in on him, fixed on him, even in scale; 15.6 to 20.4 held. (Until 2026-11-08 it was 12 s, the push 7.9 to 10.2.) His poses are on twos as in the bridge; the push and the screen's light are on ones.

Everything is in mm, the table's top at the crab's floor. The sheets face the screen's side, so they read upright from here: what was written was written towards it. The crab's mesh, face and poses are film_imprint_clawd's; only sitting is new.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))
FILM = os.path.join(ROOT, "ink", "film_out", "film")
OUT = os.path.join(FILM, "thirteen")
PAPERS = (("人", "ren"), ("神", "shen"), ("鬼", "gui"), ("器", "qi"))      # left to right as seen from here; the sheets are 十二's own (crew D's, hand/sheets/)
PAPER_W, PAPER_H = 340.0, 470.0          # mm: big sheets, as for writing one large character
PX_MM = 2074 / PAPER_W                   # the sheets' pictures: 2074 by 2867, about 6.1 px a mm
# where each sheet lies: middle x, y, turned (degrees). A blank a sheet and a quarter wide between 神 and 鬼, as in 十二
LAY = ((-688.0, 8.0, 1.6), (-322.0, -5.0, -1.1), (321.0, 4.0, .8), (684.0, -9.0, -1.9))
EDGE_Y = 400.0                           # the table's front edge, on the far side of the sheets from here
TABLE_T = 32.0


# ---------------------------------------------------------------- the sheets, outside Blender

def papers():
    """十二's sheets as they are, with a torn edge: a few mm of wandering and a fringe of fibres, into thirteen/paper_NAME.png (RGBA)."""
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    os.makedirs(OUT, exist_ok=True)
    for i, (ch, name) in enumerate(PAPERS):
        out = np.asarray(Image.open(os.path.join(FILM, "hand", "sheets", name + ".png")).convert("RGB")).astype(np.float32)
        h, w = out.shape[:2]
        r = np.random.default_rng(130 + i)
        inside = np.ones((h, w), bool)
        inside[0, :] = inside[-1, :] = inside[:, 0] = inside[:, -1] = False
        d = distance_transform_edt(inside) / PX_MM                                  # mm from the edge
        wander = gaussian_filter(r.normal(0, 1, (h, w)).astype(np.float32), 9 * PX_MM)
        wander = wander / wander.std() * .9 + gaussian_filter(r.normal(0, 1, (h, w)).astype(np.float32), 1.2 * PX_MM) / .08 * .25 * (2.7 / PX_MM)
        fibre = gaussian_filter(r.normal(0, 1, (h, w)).astype(np.float32), .35 * PX_MM)
        fibre = fibre / fibre.std()
        edge = d - (1.6 + wander)
        alpha = np.clip(edge * PX_MM / 1.2 + .5, 0, 1)
        alpha = np.maximum(alpha, np.clip((fibre - 1.6) * 2, 0, 1) * (edge > -1.2) * .8)
        # the paper a touch warmer and duller towards its edges, where it has been handled
        out = out * (1 - .035 * np.exp(-np.maximum(d, 0) / 6.0))[..., None]
        rgba = np.dstack([np.clip(out, 0, 255), alpha * 255]).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(os.path.join(OUT, f"paper_{name}.png"))
        print("paper", ch, name, w, h, flush=True)


# ---------------------------------------------------------------- the shot's clock
SHOT_S = 20.4                  # 12.0 until 2026-11-08, when the chapter's last line became three (ink_subs 十三): the four sheets stay in the frame through 「可是这四个字…」, the push comes with 「作为新生之物…」
T0 = 300.0                     # his own curves (film_imprint_clawd's) are read at T0 + the shot's time, long after anything of the bridge
WALK = (2.4, 7.6)              # in from our right, slowing over the last DECEL_S
DECEL_S = 1.8
PATH_Y, ARC_R, X_IN = 280.0, 220.0, 1190.0     # along the far side of the sheets, clear of them; then a quarter turn in through the blank
SIT = ((7.9, 0.0), (8.25, .58), (8.45, .47), (8.65, .5))      # how far down he sits: half way, his legs short stubs under him
TURN = ((6.5, 0.0), (7.6, -17.0))                            # a little round towards the screen behind us
PUSH = (11.0, 15.6)            # (7.9, 10.2) until 2026-11-08
WIDE, NEAR, ELEV, LENS, AIM_Z = 2150.0, 620.0, 10.0, 40.0, 50.0
BLINKS = (8.85, 13.2, 17.7)
# the screen: the reply going on being written, small changes several times a second, and now and then the page scrolls to a new level
LEVELS = ((0.0, 1.0), (3.7, .9), (6.3, 1.06), (9.1, .95), (11.2, 1.04), (13.9, .92), (16.6, 1.05), (19.0, .97))
SCREEN_E = 1.3e6        # a cool fill from in front: what lights his face, and what comes and goes
# the key: a warm lamp off the top of the frame, a 3000 K bulb in a shade (film_room_night3d's colour), its pool on the four sheets
LAMP_AT, LAMP_E, LAMP_R = (-200.0, 120.0, 980.0), 3.4e7, 110.0
LAMP_COLOUR = (1.0, .78, .58)
SCREEN_AT = (-170.0, -800.0, 116.0 + 168.0)     # the middle of the screen: behind us and a little to our left, its lower edge 116 mm above the table, upright


def path_length():
    return (X_IN - ARC_R) + ARC_R * math.pi / 2 + (PATH_Y - ARC_R)


def where(s):
    """Point and heading (unit vector) at s mm along his way in."""
    a = X_IN - ARC_R
    if s <= a:
        return (X_IN - s, PATH_Y), (-1.0, 0.0)
    s -= a
    q = ARC_R * math.pi / 2
    if s <= q:
        th = math.pi / 2 + s / ARC_R
        return (ARC_R + ARC_R * math.cos(th), PATH_Y - ARC_R + ARC_R * math.sin(th)), (-math.sin(th), math.cos(th))
    s = min(s - q, PATH_Y - ARC_R)
    return (0.0, PATH_Y - ARC_R - s), (0.0, -1.0)


def walked(t):
    """How far along he is at t, and how fast he goes: steady as he comes in, then slowing to a stop."""
    L = path_length()
    T = WALK[1] - WALK[0]
    v = L / (T - DECEL_S / 2)
    u = t - WALK[0]
    if u <= 0:
        return v * u, v          # before he is in frame he is still coming: off to our right
    if u >= T:
        return L, 0.0
    if u <= T - DECEL_S:
        return v * u, v
    r = u - (T - DECEL_S)
    return v * (T - DECEL_S) + v * r - v * r * r / (2 * DECEL_S), v * (1 - r / DECEL_S)


def lerp_curve(pts, t):
    tt = [p[0] for p in pts]
    vv = [p[1] for p in pts]
    if t <= tt[0]:
        return vv[0]
    for i in range(1, len(tt)):
        if t <= tt[i]:
            x = (t - tt[i - 1]) / (tt[i] - tt[i - 1])
            x = x * x * (3 - 2 * x)
            return vv[i - 1] + (vv[i] - vv[i - 1]) * x
    return vv[-1]


def glow(t):
    lv = LEVELS[0][1]
    for at, v in LEVELS[1:]:
        x = min(max((t - at) / .14, 0.0), 1.0)
        lv += (v - lv) * x * x * (3 - 2 * x)
    k = int(math.floor(t * 9.0))
    import random
    small = random.Random(1300 + k).uniform(-1, 1) * .03 + random.Random(7700 + int(t * 23.0)).uniform(-1, 1) * .012
    return lv * (1 + small)


def push(t):
    """The camera's distance: the wide, then one steady push, even in scale (its rate ramps up over the first fifth and down over the last), then held."""
    u = min(max((t - PUSH[0]) / (PUSH[1] - PUSH[0]), 0.0), 1.0)
    a = .2
    if u < a:
        f = u * u / (2 * a)
    elif u <= 1 - a:
        f = u - a / 2
    else:
        f = 1 - a - (1 - u) ** 2 / (2 * a)
    return WIDE * (NEAR / WIDE) ** (f / (1 - a))


def frame_times():
    """Each film frame at 24 fps: (index, the time the camera and the light are taken at, the time his pose is taken at). Everything on twos as
    stop motion is, but for the push, which is on ones so that it glides."""
    out = []
    for i in range(int(round(SHOT_S * 24))):
        t = i / 24
        tp = (i // 2) * 2 / 24
        out.append((i, t if PUSH[0] <= t <= PUSH[1] else tp, tp))
    return out


def frame_files():
    """For each film frame, the frame whose picture it shows: itself, or the one before it when nothing has moved between them."""
    src, last, key = [], None, None
    for i, t, tp in frame_times():
        if (t, tp) != key:
            last, key = i, (t, tp)
        src.append(last)
    return src


LIM = (5.0, 8.0)        # how far a planted foot may get from under its leg, across him and along him, before it has to step (further, and a pair runs together)
LIFT_H, BOB = 9.0, 2.0


def v_full():
    return walked(WALK[0] + .1)[1]


def heading_yaw(n):
    t = n / 12
    s_mm, v = walked(t)
    (px, py), (hx, hy) = where(max(s_mm, 0.0))
    if s_mm < 0:
        px -= s_mm
    return (px, py), (hx, hy), math.radians(lerp_curve(TURN, t)), v


def gait(legs, leg_y):
    """His feet, pose by pose (on twos): a foot that is down stays where it was put on the table, however he moves over it; when it has got
    as far from under him as it may, it comes up for one pose and is put down again a little ahead. Fast, the two sets of feet take turns
    every pose, so no foot is down for two poses running; slowing, they stay down longer and he moves over them; stopped, any foot left
    out of place steps back under him. For each pose: each leg's offset in his own frame (across, along), how high it is lifted, his bob."""
    n_all = int(SHOT_S * 12) + 2
    group = [i % 2 for i in range(len(legs))]
    anchor = [None] * len(legs)          # where each down foot is on the table
    up_since = {0: None, 1: -1}
    out = []

    def rel(i, body, yaw, wx, wy):
        dx, dy = wx - body[0], wy - body[1]
        c, s_ = math.cos(-yaw), math.sin(-yaw)
        return dx * c - dy * s_ - legs[i], dx * s_ + dy * c - leg_y

    def world(i, body, yaw, ox, oy):
        x, y = legs[i] + ox, leg_y + oy
        c, s_ = math.cos(yaw), math.sin(yaw)
        return body[0] + x * c - y * s_, body[1] + x * s_ + y * c

    def far(o):
        return (o[0] / LIM[0]) ** 2 + (o[1] / LIM[1]) ** 2 > 1.0

    for n in range(n_all):
        body, (hx, hy), yaw, v = heading_yaw(n)
        nxt, *_ = heading_yaw(n + 1)
        step = math.hypot(nxt[0] - body[0], nxt[1] - body[1])
        c, s_ = math.cos(-yaw), math.sin(-yaw)
        bx, by = hx * c - hy * s_, hx * s_ + hy * c          # his way, in his own frame
        down = {g: up_since[g] is None for g in (0, 1)}
        offs = [None] * len(legs)
        sitting = n / 12 >= SIT[0][0]        # once he starts to sit his feet stay where they are
        for g in (0, 1):
            if down[g]:
                if any(anchor[i] is None for i in range(len(legs)) if group[i] == g):
                    for i in range(len(legs)):
                        if group[i] == g:
                            anchor[i] = world(i, body, yaw, 0.0, 0.0)
                o = [rel(i, body, yaw, *anchor[i]) for i in range(len(legs)) if group[i] == g]
                if any(far(x) for x in o) and not sitting:
                    down[g] = False
        moving = step > .05
        for g in (0, 1):
            if up_since[g] is not None and up_since[g] < n:
                # it has been up a pose: down again, ahead by as much as it may, or half of what is still to come
                k = min(1.0, step / 2 / max(LIM[0] * abs(bx) + LIM[1] * abs(by), 1e-6)) if moving else 0.0
                for i in range(len(legs)):
                    if group[i] == g:
                        anchor[i] = world(i, body, yaw, LIM[0] * bx * k, LIM[1] * by * k)
                up_since[g] = None
                down[g] = True
        for g in (0, 1):
            if not down[g] and up_since[g] is None:
                up_since[g] = n
        # never both up: the one that came up last stays down a pose longer
        if up_since[0] is not None and up_since[1] is not None and up_since[0] == up_since[1] == n:
            up_since[1] = None
        if not moving and not sitting:
            for g in (0, 1):
                if up_since[g] is None and up_since[1 - g] is None:
                    o = [rel(i, body, yaw, *anchor[i]) for i in range(len(legs)) if group[i] == g]
                    if any(math.hypot(*x) > 1.0 for x in o):
                        up_since[g] = n
                        break
        lifts = []
        for i in range(len(legs)):
            g = group[i]
            if up_since[g] is None:
                offs[i] = rel(i, body, yaw, *anchor[i])
                lifts.append(0.0)
            else:
                offs[i] = (LIM[0] * .5 * bx if moving else 0.0, LIM[1] * .5 * by if moving else 0.0)
                lifts.append(LIFT_H * (1.0 if moving else .6))
        out.append(dict(offs=offs, lifts=lifts, bob=BOB if any(lifts) else 0.0, body=body, yaw=yaw))
    return out


# ---------------------------------------------------------------- the scene, in Blender
K = None


def scene(a):
    import bpy
    import numpy as np
    import film_imprint_clawd as c
    from film_imprint3d import set_up
    from ink3d import aim, reset
    from film_imprint_clawd import FLOOR, LEG_H, LEGS, U, ZB, Clawd, curve, put_mesh
    global K

    t, tp = a["t"], a.get("tp", a["t"])
    sc = reset(a["samples"], 100)
    set_up(dict(w=a["w"], h=a["h"], denoise=True, exp=a["exp"]), sc)
    sc.cycles.max_bounces, sc.cycles.diffuse_bounces = 8, 4
    sc.view_settings.look = "AgX - Base Contrast"

    def material(name, colour, rough, spec=.3, noise=None):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        b = nt.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*colour, 1)
        b.inputs["Roughness"].default_value = rough
        b.inputs["Specular IOR Level"].default_value = spec
        if noise:
            tc, n, ramp = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeValToRGB")
            n.inputs["Scale"].default_value, n.inputs["Detail"].default_value = noise[0], 6.0
            ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = .3, .7
            ramp.color_ramp.elements[0].color = (*[v * (1 - noise[1]) for v in colour], 1)
            ramp.color_ramp.elements[1].color = (*[v * (1 + noise[1]) for v in colour], 1)
            nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
            nt.links.new(n.outputs[0], ramp.inputs[0])
            nt.links.new(ramp.outputs[0], b.inputs["Base Color"])
        return m

    def box(name, x0, x1, y0, y1, z0, z1, m):
        v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        me = bpy.data.meshes.new(name)
        me.from_pydata(v, [], f)
        me.update()
        me.materials.append(m)
        sc.collection.objects.link(bpy.data.objects.new(name, me))

    # -- the table: the room's own wood, its grain along x
    def table():
        tex = os.path.join(FILM, "room", "tex")
        m = bpy.data.materials.new("table")
        m.use_nodes = True
        nt = m.node_tree
        b = nt.nodes["Principled BSDF"]
        uv = nt.nodes.new("ShaderNodeUVMap")
        ims = []
        for name, space in (("table.png", "sRGB"), ("table_rough.png", "Non-Color"), ("table_bump.png", "Non-Color")):
            tx = nt.nodes.new("ShaderNodeTexImage")
            tx.image = bpy.data.images.load(os.path.join(tex, name), check_existing=True)
            tx.image.colorspace_settings.name = space
            tx.interpolation, tx.extension = "Cubic", "MIRROR"
            nt.links.new(uv.outputs[0], tx.inputs[0])
            ims.append(tx)
        nt.links.new(ims[0].outputs[0], b.inputs["Base Color"])
        nt.links.new(ims[1].outputs[0], b.inputs["Roughness"])
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value, bp.inputs["Distance"].default_value = .35, 1.2
        nt.links.new(ims[2].outputs[0], bp.inputs["Height"])
        nt.links.new(bp.outputs[0], b.inputs["Normal"])
        x0, x1, y0, y1 = -1100.0, 1100.0, -2400.0, EDGE_Y      # on past the screen towards us, out of frame: no edge, nothing under it seen
        z0 = FLOOR
        v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y1, z0 - TABLE_T), (x1, y1, z0 - TABLE_T)]
        f = [(0, 1, 2, 3), (3, 2, 5, 4)]
        me = bpy.data.meshes.new("table")
        me.from_pydata(v, [], f)
        me.update()
        lay = me.uv_layers.new(name="UV")
        # the room's table is 1740 by 740 mm: the same size of grain
        for poly in me.polygons:
            for li in poly.loop_indices:
                x, y, z = v[me.loops[li].vertex_index]
                yy = y if z >= z0 - .01 else y + (z0 - z) * .3
                lay.data[li].uv = ((x + 870.0) / 1740.0, (EDGE_Y - yy) / 740.0)
        me.materials.append(m)
        o = bpy.data.objects.new("table", me)
        sc.collection.objects.link(o)
        bev = o.modifiers.new("bevel", "BEVEL")
        bev.width, bev.segments = 2.0, 3

    # -- the room beyond the table, as film_room3d has it, seen from the screen's side: bare plaster walls, the wood floor 740 below the table's top
    def room():
        plaster = material("plaster", (.82, .80, .75), .9, .25, noise=(.0016, .03))
        floor_m = material("floor", (.26, .17, .10), .45, .4, noise=(.004, .12))
        paint = material("skirt", (.86, .85, .82), .5, .4)
        ground, ceil = FLOOR - 740.0, FLOOR - 740.0 + 2620.0
        back, left, right = EDGE_Y + 2555.0, 2030.0, -1870.0
        box("wall_back", right - 200, left + 200, back, back + 200, ground, ceil, plaster)
        box("wall_r", left, left + 200, -2600.0, back, ground, ceil, plaster)
        box("wall_l", right - 200, right, -2600.0, back, ground, ceil, plaster)
        box("floor", right - 200, left + 200, -2600.0, back + 200, ground - 100, ground, floor_m)
        box("skirt_back", right, left, back - 12, back, ground, ground + 90, paint)
        # the table's legs, under its far edge
        leg = material("table_leg", (.20, .13, .08), .5, .4)
        for x in (-1060.0, 1020.0):
            box("leg", x, x + 40, EDGE_Y - 80, EDGE_Y - 40, ground, FLOOR - TABLE_T, leg)

    # -- the sheets: not quite flat, as paper never is
    def sheet(name, cx, cy, turn, seed):
        m = bpy.data.materials.new("paper_" + name)
        m.use_nodes = True
        nt = m.node_tree
        b = nt.nodes["Principled BSDF"]
        tx = nt.nodes.new("ShaderNodeTexImage")
        tx.image = bpy.data.images.load(os.path.join(OUT, f"paper_{name}.png"), check_existing=True)
        tx.interpolation = "Cubic"
        nt.links.new(tx.outputs[0], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = .88
        b.inputs["Specular IOR Level"].default_value = .25
        b.inputs["Sheen Weight"].default_value = .25
        b.inputs["Sheen Roughness"].default_value = .6
        b.inputs["Subsurface Weight"].default_value = .06
        b.inputs["Subsurface Radius"].default_value = (2.0, 1.6, 1.2)
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(tx.outputs["Alpha"], mix.inputs[0])
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(b.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], nt.nodes["Material Output"].inputs[0])
        tooth = nt.nodes.new("ShaderNodeTexNoise")
        tooth.inputs["Scale"].default_value, tooth.inputs["Detail"].default_value = 1.4, 6.0
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value, bp.inputs["Distance"].default_value = .25, .08
        nt.links.new(tooth.outputs[0], bp.inputs["Height"])
        nt.links.new(bp.outputs[0], b.inputs["Normal"])
        nu, nv = 49, 67
        u, v = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv))
        x, y = (u - .5) * PAPER_W, (v - .5) * PAPER_H
        r = np.random.default_rng(seed)
        lift = .0
        for _ in range(4):
            k = r.normal(size=2)
            lift = lift + np.sin((x * k[0] + y * k[1]) / 70.0 + r.uniform(0, 6))
        corner = (np.abs(x) / (PAPER_W / 2)) ** 4 + (np.abs(y) / (PAPER_H / 2)) ** 4
        z = FLOOR + .3 + .35 * (lift / 4 + 1) + 1.6 * np.clip(corner - .55, 0, None) * r.uniform(.5, 1.2)
        th = math.radians(turn)
        X, Y = cx + x * math.cos(th) - y * math.sin(th), cy + x * math.sin(th) + y * math.cos(th)
        verts = np.stack([X, Y, z], -1).reshape(-1, 3)
        idx = np.arange(nu * nv).reshape(nv, nu)
        faces = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
        me = bpy.data.meshes.new("sheet_" + name)
        me.from_pydata(verts.tolist(), [], faces.tolist())
        me.update()
        lay = me.uv_layers.new(name="UV")
        uvs = np.stack([u, v], -1).reshape(-1, 2)      # the top of the picture at the far end of the sheet
        for poly in me.polygons:
            for li in poly.loop_indices:
                lay.data[li].uv = uvs[me.loops[li].vertex_index]
        me.shade_smooth()
        me.materials.append(m)
        sc.collection.objects.link(bpy.data.objects.new("sheet_" + name, me))

    table()
    room()
    for i, ((ch, name), (cx, cy, turn)) in enumerate(zip(PAPERS, LAY)):
        sheet(name, cx, cy, turn, 140 + i)

    # -- the crab: in from our right, sideways as a crab goes, then in through the blank, and down
    s_mm, speed = walked(tp)
    (px, py), (hx, hy) = where(max(s_mm, 0.0))
    if s_mm < 0:
        px -= s_mm
    in_frame = px < X_IN + 40
    if in_frame and not a["empty"]:
        sit = lerp_curve(SIT, tp)
        yaw = lerp_curve(TURN, tp)
        breath = [(T0 + 8.65 + .4 * i, .025 * math.sin(2 * math.pi * .4 * i / 3.3)) for i in range(12)]
        held = dict(BOW=-.06, CHEW=0.0, MUNCH=0.0, CHEEKS=0.0, BREATH=0.0, EAT=0.0, YAWN=0.0, SQUINT=0.0, RAISE=0.0, SLUMP=.35, WAVE=0.0)
        for key, val in held.items():
            setattr(c, key, curve((0.0, val), (1000.0, val)))
        # the settle: as his weight comes down on his legs he squashes a little and comes back, then breathes
        c.STRETCH = curve((0.0, 0.0), (T0 + 8.19, 0.0), (T0 + 8.33, -.05), (T0 + 8.48, .012), (T0 + 8.61, 0.0), *breath[1:])
        c.YAW = curve((0.0, yaw), (1000.0, yaw))
        bl = []
        for b0 in BLINKS:
            bl += [(T0 + b0, 0), (T0 + b0 + .05, 1), (T0 + b0 + .1, 1), (T0 + b0 + .17, 0)]
        c.BLINK = curve(*bl)
        c.EYE_GOT, c.EYE_OPEN = curve((0.0, 1.0), (1000.0, 1.0)), curve((0.0, 1.0), (1000.0, 1.0))
        if K is None:
            K = Clawd(a["mesh"], a.get("prints", not a["draft"]))
        k = K
        disp, eye_d, mouth_d, glint_d = c.face(k, T0 + tp)
        P = k.R + k.N * (k.static + disp)[:, None]
        Q = k.R
        # the walk: each foot where gait() has it, planted on the table or up for a pose
        n = int(round(tp * 12))
        G = gait(LEGS, c.LEG_Y)[min(n, int(SHOT_S * 12))]
        if any(G["lifts"]) or any(math.hypot(*o) > 0 for o in G["offs"]):
            w = np.clip((ZB + 2.0 - Q[:, 2]) / LEG_H, 0, 1) ** .7
            for i, lx in enumerate(LEGS):
                on = np.abs(Q[:, 0] - lx) < U
                P[on, 0] += G["offs"][i][0] * w[on]
                P[on, 1] += G["offs"][i][1] * w[on]
                P[on, 2] += G["lifts"][i] * w[on]
            P[:, 2] += G["bob"] * np.clip((Q[:, 2] - FLOOR) / LEG_H, 0, 1)
        # leaning a little into his way while he goes, more the faster
        _, (hx_, hy_), yaw_r, v_ = heading_yaw(n)
        lean = math.radians(3.0) * min(speed / max(v_full(), 1e-6), 1.0)
        if lean > 0:
            cy_, sy_ = math.cos(-yaw_r), math.sin(-yaw_r)
            bx_, by_ = hx_ * cy_ - hy_ * sy_, hx_ * sy_ + hy_ * cy_
            up = np.clip(Q[:, 2] - ZB, 0, None) * math.sin(lean)
            P[:, 0] += bx_ * up
            P[:, 1] += by_ * up
        # sitting: the legs fold under him and lean in a little, his body comes down onto them; the feet stay where they were put
        if sit > 0:
            legw = np.clip((ZB - Q[:, 2]) / LEG_H, 0, 1) * np.clip((Q[:, 2] - FLOOR) / (.35 * LEG_H), 0, 1)
            P[:, 0] += np.sign(Q[:, 0]) * 2.0 * sit * legw
            P[:, 1] += 5.0 * sit * legw
            P[:, 2] = np.where(Q[:, 2] < ZB, FLOOR + (P[:, 2] - FLOOR) * (1 - .9 * sit), P[:, 2] - .9 * sit * LEG_H)
        P = c.pose(P, T0 + tp)
        P[:, 0] += px
        P[:, 1] += py
        put_mesh("clawd", P, k.quads, dict(rest=k.R, eye=eye_d, mouth=mouth_d, glint=glint_d, deep=k.deep, warm=c.warmth(k, T0 + tp)), c.clay())

    # -- the light: the lamp's warm pool on the desk, and the screen, where it stands on the table, upright and facing the room, its cool light coming and going
    #    with what is on it. The table on our side of the screen gets only the lamp's edge: the foreground falls away, the far wall gets a little
    screen = bpy.data.lights.new("screen", "AREA")
    screen.shape, screen.size, screen.size_y = "RECTANGLE", 600.0, 340.0
    screen.color = (.92, .96, 1.0)
    screen.energy = SCREEN_E * glow(t) * a["glow"]
    o = bpy.data.objects.new("screen", screen)
    sc.collection.objects.link(o)
    aim(o, (SCREEN_AT[0], SCREEN_AT[1], FLOOR + SCREEN_AT[2]), (SCREEN_AT[0], 0.0, FLOOR + SCREEN_AT[2]))     # square to the table's edge, so where its light ends runs straight across
    lamp = bpy.data.lights.new("lamp", "SPOT")
    lamp.energy, lamp.color, lamp.shadow_soft_size = LAMP_E, LAMP_COLOUR, LAMP_R
    lamp.spot_size, lamp.spot_blend = math.radians(150.0), .85
    lo = bpy.data.objects.new("lamp", lamp)
    sc.collection.objects.link(lo)
    aim(lo, (LAMP_AT[0], LAMP_AT[1], FLOOR + LAMP_AT[2]), (LAMP_AT[0] * .3, 0.0, FLOOR))
    wd = sc.world.node_tree.nodes["Background"]
    wd.inputs[0].default_value = (.010, .0085, .0075, 1)
    wd.inputs[1].default_value = .35

    # -- the camera: low, fixed on where he sits; the push only brings it nearer along the same line
    from mathutils import Matrix
    dist = a.get("dist") or push(t)
    e = math.radians(ELEV)
    target = np.array((0.0, 0.0, FLOOR + AIM_Z))
    p = target + dist * np.array((0.0, -math.cos(e), math.sin(e)))
    data = bpy.data.cameras.new("cam")
    data.lens, data.sensor_width = LENS, 36.0
    data.clip_start, data.clip_end = 5.0, 60000.0
    data.dof.use_dof = True
    data.dof.focus_distance = dist - 27.0
    data.dof.aperture_fstop = a["fstop"] / 1000.0
    data.dof.aperture_blades = 7
    cam = bpy.data.objects.new("cam", data)
    sc.collection.objects.link(cam)
    cam.matrix_world = Matrix(((1, 0, 0, p[0]), (0, math.sin(e), -math.cos(e), p[1]), (0, math.cos(e), math.sin(e), p[2]), (0, 0, 0, 1)))
    sc.camera = cam
    path = os.path.join(OUT, a["name"] + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    deep16 = f"/tmp/crab13_{os.getpid()}.png"
    sc.render.image_settings.file_format, sc.render.image_settings.color_depth = "PNG", "16"
    sc.render.filepath = deep16
    bpy.ops.render.render(write_still=True)
    c.finish(deep16, path, int(t * 1000) + 13, 1.6 if a["draft"] else 1.3)
    print("wrote", path, "t %.4f pose %.4f dist %.0f glow %.3f at %.0f %.0f" % (t, tp, dist, glow(t), px, py), flush=True)


if __name__ == "__main__":
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]

        def opt(name, default, kind=float):
            return kind(argv[argv.index(name) + 1]) if name in argv else default

        draft = "--draft" in argv
        common = dict(empty="--empty" in argv, glow=opt("--glow", 1.0), fstop=opt("--fstop", 8.0), exp=opt("--exp", .5), dist=opt("--dist", 0.0))
        if argv[0] == "still":
            t = float(argv[2])
            scene(dict(name=argv[1], t=t, tp=t, draft=draft, samples=opt("--samples", 64 if draft else 256, int), w=opt("--w", 960 if draft else 1920, int), h=opt("--h", 540 if draft else 1080, int), mesh=opt("--mesh", .5 if draft else .25), **common))
        elif argv[0] == "frames":     # one worker's share; a frame already there is left alone. On twos but for the push (frame_times)
            name, me, of = argv[1], int(argv[2]), int(argv[3])
            hi = "--hi" in argv       # the same frames at 1920 by 1080 for the final: no finger prints and his skin .5 as the drafts have them (--mesh to choose), the grain the final's
            src = frame_files()
            own = [x for x in frame_times() if src[x[0]] == x[0]]          # only the frames that are pictures of their own
            for j, (i, t, tp) in enumerate(own):
                f = os.path.join(OUT, name, f"f{i:03d}.png")
                if j % of == me and not os.path.exists(f):
                    scene(dict(name=f"{name}/f{i:03d}", t=t, tp=tp, draft=not hi, prints=False, samples=opt("--samples", 128 if hi else 64, int), w=1920 if hi else 960, h=1080 if hi else 540, mesh=opt("--mesh", .5), **common))
    elif len(sys.argv) > 1 and sys.argv[1] == "papers":
        papers()
    elif len(sys.argv) > 1 and sys.argv[1] == "list":         # python film_crab_new.py list NAME > list.txt: the frames as an ffmpeg concat list
        for j in frame_files():
            print(f"file '{os.path.join(OUT, sys.argv[2], f'f{j:03d}.png')}'\nduration {1 / 24:.6f}")
    else:
        raise SystemExit(__doc__)
