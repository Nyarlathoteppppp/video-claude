"""九 · 千窗. The river is the valley floor; the windows live on the mountains that stand over it, and every one of them has its reflection in that same river.

Shots and how they move (the first window never moves on screen: the camera slides along the line through it, as in a steady pull-back):

    chat    The riverbed left by chapter 8 goes dark, and a small patch of its white lights up. The camera pushes in until it is a chat window set in the wet stone: 你好 / 你好！有什么可以帮你的吗？
            It leaves by pulling back at an even pace; while the window still fills the frame its writing fades into the paper and its frame becomes a lattice.
    lattice The same pull-back: now it is a real paper window with wooden bars in a lime wall by the water, the village unlit behind it. It keeps shrinking towards qian. On its paper the 你好 is still there.
    qian    It is a small light by the river. From it outwards they come on, window by window, until the valley is lit, mountain behind mountain, and the water under them is full of their reflections.
            The nearest few can be read, and no two say the same. --t runs the lights: --t=0 only the first, --t=1 all of them, 1 to 2 the going-out (see lit()).
    going   The same camera at --t=1.5: the ones we could read went first, and each reflection went with its window. The river stays.
    dark    The same camera, past --t=2: the first one was the last to go. Black mountains under cloud, and the river, empty. It hands over to chapter 10, which opens on black.

The windows and their reflections are a layer of their own: windows_<shot>.npz (written by the render) holds each window's place in the world and on the frame, its mirror image's place, its brightness and when it
comes on and goes out, so the lights can be run in 2D without rendering again (ink3d_nightgrade.water_layer).

    python ink3d_qian.py land [nlevels]; python ink3d_qian.py chattex
    Blender -b --factory-startup -P ink3d_qian.py -- --shot=qian [--t=1] [--pct=100] [--n=1536] [--samples=64] [--tag=]
    python ink3d_qian.py grade --shot=qian [--tag=]; python ink3d_qian.py sheet
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import bpy
except ImportError:
    bpy = None

SEED = 77
SIZE, CENTRE_Y = 1100.0, 350.0
LAND_V = "m"              # bump to rebuild the cached land
CAM_HERO = (-10.0, -165.0, 5.0)         # low over the water, as the camera is over the marsh in chapter 1: the ghost fires there, the windows here, near to far, standing on their reflections
TARGET_HERO = (-10.0, 235.0, -2.5)
HORIZON = 500.0                         # where the water would meet the sky, in pixels
# The windows we can read: (x on the frame, distance, size of house, what is on its paper). No two the same. A row along the bank, each clear of the one before it. The first is the first window: it stays where it is on the frame from lattice to dark.
# The two nearest go out first: the subtitles name them a moment later.
HERO_WINDOWS = ((975.0, 147.0, 5.0, "你好"), (1080.0, 92.0, 5.5, "会好吗"), (1235.0, 58.0, 5.0, "我失业了睡不着"), (1510.0, 36.0, 5.0, "帮我改改这封情书"))
FAR_ = 1300.0               # these mountains go further than the shared depth pass reaches
LENS = 40.0
PIXELS = LENS / 36 * 1920   # focal length in pixels
PATCH = (30.0, -38.0, 320.0)        # the near hill: centre x, y and side of the finer piece of land
MOON_PX = (790.0, 125.0)            # the moon behind thin cloud, in the gap of the valley
WATER_Z = .22
NEAR_D = 175.0                      # closer than this a window is on a whole house and has bars
PULL_FRAMES, PULL_A, PULL_B = 165, 6.833, 147.0   # the pull-back: this many frames, from where the first window's paper is as big as the Claude window on the screen (ink_nine.py) to the place of the hero frame


def pull_near(n, ease=.15):
    """How far from the first window the camera is on frame n of the pull-back: an even pace in scale, easing only at its two ends."""
    x, v = min(max(n / (PULL_FRAMES - 1), 0.0), 1.0), 1 / (1 - ease)
    u = v * x * x / (2 * ease) if x < ease else 1 - v * (1 - x) ** 2 / (2 * ease) if x > 1 - ease else v * (x - ease / 2)
    return PULL_A * (PULL_B / PULL_A) ** u


CHAT_C, CHAT_U = (35.6, -35.6), (-.34, .94)     # where the chat window lies in chapter 8's long stroke, and which way is up it
CHAT_W, CHAT_H = 10.0, 6.5


def bank_sdf(x, y):
    """The river, down the middle of the valley and past the camera. Less than 0 is water, more is how far from it."""
    import numpy as np
    right = np.interp(y, [-200, -130, -50, 70, 215, 400, 540], [-3, -3.5, -7, -15, -30, -34, -44])
    left = np.interp(y, [-200, -47, 72, 215, 400, 540], [-120, -70, -62, -62, -58, -46])
    return np.maximum(x - right, left - x) + 1.4 * np.sin(y / 13 + x / 40) + .8 * np.sin(y / 7.3 + 1) + 1.2 * np.maximum(y - 540, 0)


def window_px(d):
    """How many pixels wide a window is at distance d: true to its size close by, and further off a little larger than life, so that a far one is still square and not a speck."""
    import numpy as np
    return np.maximum(19 * (NEAR_D / np.maximum(d, 1)) ** 1.3, 3.0)


# ============================================================================================================================================================
# Blender
# ============================================================================================================================================================
if bpy is not None:
    import numpy as np

    import ink3d
    ink3d.FAR = FAR_
    from ink3d import OUT3D, blur, camera, grey_material, ink_albedo, mesh_grid, painted_material, render, reset, sm, spectral, sun, water_material

    ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    def opt(name, default):
        for a in ARGS:
            if a.startswith(f"--{name}="):
                return type(default)(a.split("=")[1])
        return default

    SHOT, PCT, N, TAG, T = opt("shot", "qian"), opt("pct", 100), opt("n", 3072), opt("tag", ""), opt("t", 1.0)
    CACHE = os.path.join(OUT3D, "qian", "cache")

    def cached(key, fn):
        path = os.path.join(CACHE, f"{key}_{LAND_V}_{N}.npy")
        if os.path.exists(path):
            return np.load(path)
        v = fn()
        os.makedirs(CACHE, exist_ok=True)
        np.save(path, v)
        return v

    def land():
        y, x = np.mgrid[:N, :N].astype(np.float32)
        X, Y = (x / (N - 1) - .5) * SIZE, (y / (N - 1) - .5) * SIZE + CENTRE_Y
        return X, Y, np.load(os.path.join(CACHE, f"land_{LAND_V}_{N}.npy"))

    def near_hill():
        """The finer piece of land near the camera, and its height at any point."""
        cx, cy, side = PATCH
        fine = np.load(os.path.join(CACHE, f"patch_{LAND_V}.npy"))
        m = fine.shape[0]
        y, x = np.mgrid[:m, :m].astype(np.float32)
        return (x / (m - 1) - .5) * side + cx, (y / (m - 1) - .5) * side + cy, fine

    def height_at(h, x, y):
        cell = SIZE / (N - 1)
        fx, fy = (x + SIZE / 2) / cell, (y - CENTRE_Y + SIZE / 2) / cell
        ix, iy = np.clip(fx.astype(int), 0, N - 2), np.clip(fy.astype(int), 0, N - 2)
        fx, fy = fx - ix, fy - iy
        return (h[iy, ix] * (1 - fx) + h[iy, ix + 1] * fx) * (1 - fy) + (h[iy + 1, ix] * (1 - fx) + h[iy + 1, ix + 1] * fx) * fy

    def height_fine(fine, x, y):
        """On the near hill where there is one, for the points inside it."""
        cx, cy, side = PATCH
        m = fine.shape[0]
        cell = side / (m - 1)
        fx, fy = (x - cx + side / 2) / cell, (y - cy + side / 2) / cell
        ix, iy = np.clip(fx.astype(int), 0, m - 2), np.clip(fy.astype(int), 0, m - 2)
        fx, fy = fx - ix, fy - iy
        return (fine[iy, ix] * (1 - fx) + fine[iy, ix + 1] * fx) * (1 - fy) + (fine[iy + 1, ix] * (1 - fx) + fine[iy + 1, ix + 1] * fx) * fy

    def camera_axes(cam, tgt):
        c, t = np.array(cam, float), np.array(tgt, float)
        f = (t - c) / np.linalg.norm(t - c)
        r_ = np.cross(f, (0, 0, 1))
        r_ /= np.linalg.norm(r_)
        return c, f, r_, np.cross(r_, f)

    def project(x, y, z):
        """Where points fall on the hero frame, in pixels at full size."""
        c, f, r_, u = camera_axes(CAM_HERO, TARGET_HERO)
        px, py, pz = x - c[0], y - c[1], z - c[2]
        depth = px * f[0] + py * f[1] + pz * f[2]
        return 960 + (px * r_[0] + py * r_[1] + pz * r_[2]) / depth * PIXELS, 540 - (px * u[0] + py * u[1] + pz * u[2]) / depth * PIXELS

    def hero_places(fine):
        """The world position of each window we can read: its centre, from where it is on the frame and how far off it is."""
        c, f, r_, u = camera_axes(CAM_HERO, TARGET_HERO)
        out = []
        for px, dist, s_, line in HERO_WINDOWS:
            x, y = c[0] + (px - 960) / PIXELS * dist, c[1] + dist
            g = float(height_fine(fine, np.array([x], np.float32), np.array([y], np.float32))[0])
            out.append((x, y, max(g, WATER_Z + .5) + .65 * s_, s_))
        return np.array(out, np.float64)

    # ---- building up merged meshes -------------------------------------------------------------------------------------------------------------------------
    def poly_mesh(name, V, polys):
        """A mesh from an N×3 array of vertices and k×s arrays of polygons, one array for each polygon size."""
        me = bpy.data.meshes.new(name)
        me.vertices.add(len(V))
        me.vertices.foreach_set("co", V.ravel().astype(np.float32))
        loops = np.concatenate([p.ravel() for p in polys]).astype(np.int32)
        sizes = np.concatenate([np.full(len(p), p.shape[1]) for p in polys])
        me.loops.add(len(loops))
        me.loops.foreach_set("vertex_index", loops)
        me.polygons.add(len(sizes))
        me.polygons.foreach_set("loop_start", np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int32))
        me.update(calc_edges=True)
        o = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(o)
        return o

    def instances(tmpl, x, y, z, yaw, scale, sx=None, sz=None):
        """Copies of a template (m×3, local x along the front, local -y out of the front), n of them: n·m×3. sx widens each; sz raises its walls and what stands on them."""
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        tz = tmpl[:, 2][None, :]
        if sz is not None:
            tz = tz + (sz[:, None] - 1) * H_ * np.clip(tz / H_, 0, 1)
        lx, ly, lz = tmpl[:, 0][None, :] * scale[:, None], tmpl[:, 1][None, :] * scale[:, None], tz * scale[:, None]
        if sx is not None:
            lx = lx * sx[:, None]
        return np.stack([x[:, None] + c * lx - s * ly, y[:, None] + s * lx + c * ly, z[:, None] + lz], -1).reshape(-1, 3)

    def faces_of(tmpl_faces, n, m):
        f = np.asarray(tmpl_faces, np.int64)
        return (f[None, :, :] + (np.arange(n) * m)[:, None, None]).reshape(-1, f.shape[1])

    # the house, in units of its own scale: front wall along x at y = -.5
    L, D_, H_, R_, O_ = .8, .5, .72, .42, .12
    WALL = np.array([(-L, -D_, -1.2), (L, -D_, -1.2), (L, D_, -1.2), (-L, D_, -1.2), (-L, -D_, H_), (L, -D_, H_), (L, D_, H_), (-L, D_, H_), (-L, 0, H_ + R_), (L, 0, H_ + R_)], np.float32)
    WALL_Q, WALL_P = [(0, 1, 5, 4), (2, 3, 7, 6)], [(3, 0, 4, 8, 7), (1, 2, 6, 9, 5)]
    ROOF = np.array([(-L - O_, -D_ - O_, H_ - .08), (L + O_, -D_ - O_, H_ - .08), (L + O_, 0, H_ + R_), (-L - O_, 0, H_ + R_), (L + O_, D_ + O_, H_ - .08), (-L - O_, D_ + O_, H_ - .08)], np.float32)
    ROOF_Q = [(0, 1, 2, 3), (4, 5, 3, 2)]
    PX0, PX1, PZ0, PZ1 = -.3, .3, .1, .58       # the window is the main thing on the house
    PANE = np.array([(PX0, -D_ - .012, PZ0), (PX1, -D_ - .012, PZ0), (PX1, -D_ - .012, PZ1), (PX0, -D_ - .012, PZ1)], np.float32)

    def roof_template(nx=6, ny=5):
        """A tiled roof that sags between ridge and eave and turns up at the corners, both slopes."""
        V, F = [], []
        for side in (-1, 1):
            base = len(V)
            for j in range(ny + 1):
                t = j / ny
                for i in range(nx + 1):
                    x = (L + O_) * (2 * i / nx - 1)
                    V.append((x, side * (D_ + O_) * (1 - t), H_ - .06 + (R_ + .06) * t ** 1.4 + .09 * (abs(x) / (L + O_)) ** 4 * (1 - t) ** 2))
            for j in range(ny):
                for i in range(nx):
                    a = base + j * (nx + 1) + i
                    F.append((a, a + 1, a + nx + 2, a + nx + 1))
        return np.array(V, np.float32), np.array(F)

    ROOF_V, ROOF_F = roof_template()
    DOOR = np.array([(.42, -D_ - .008, -.62), (.68, -D_ - .008, -.62), (.68, -D_ - .008, .34), (.42, -D_ - .008, .34)], np.float32)

    def box(x0, x1, y0, y1, z0, z1):
        return [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]

    BOX_Q = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (4, 5, 6, 7)]

    def lattice_template(nx=3, nz=3, bar=.016):
        """Window bars in front of the pane, in the house's units."""
        x0, x1, z0, z1 = PX0, PX1, PZ0, PZ1
        y0, y1 = -D_ - .05, -D_ - .012
        fr = .035
        vs = [box(x0 - fr, x1 + fr, y0, y1, z0 - fr, z0), box(x0 - fr, x1 + fr, y0, y1, z1, z1 + fr), box(x0 - fr, x0, y0, y1, z0, z1), box(x1, x1 + fr, y0, y1, z0, z1)]
        for i in range(1, nx):
            xc = x0 + (x1 - x0) * i / nx
            vs.append(box(xc - bar, xc + bar, y0, y1, z0, z1))
        for j in range(1, nz):
            zc = z0 + (z1 - z0) * j / nz
            vs.append(box(x0, x1, y0, y1, zc - bar, zc + bar))
        V = np.array(vs, np.float32)
        F = np.concatenate([np.array(BOX_Q) + 8 * i for i in range(len(vs))])
        return V.reshape(-1, 3), F

    def window_material(name, strength, sampled=True):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        a, e, o, mul = nt.nodes.new("ShaderNodeAttribute"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial"), nt.nodes.new("ShaderNodeMath")
        a.attribute_name = "lit"
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = strength
        nt.links.new(a.outputs["Fac"], mul.inputs[0])
        nt.links.new(mul.outputs[0], e.inputs["Strength"])
        e.inputs["Color"].default_value = (1, 1, 1, 1)
        nt.links.new(e.outputs[0], o.inputs["Surface"])
        m.cycles.emission_sampling = "AUTO" if sampled else "NONE"
        return m

    def set_lit(o, per_quad):
        """Brightness of each window (one value per four vertices)."""
        me = o.data
        name = "lit"
        if name not in me.attributes:
            me.attributes.new(name, "FLOAT", "POINT")
        me.attributes[name].data.foreach_set("value", np.repeat(per_quad, 4).astype(np.float32))

    # ---- where the houses stand -------------------------------------------------------------------------------------------------------------------------------
    def frame_mask(x, y, z, cam, tgt, margin=1.15):
        """Which points are in the view of a camera at cam looking at tgt, with some margin; and how far away they are."""
        c, t = np.array(cam, float), np.array(tgt, float)
        f = (t - c) / np.linalg.norm(t - c)
        r_ = np.cross(f, (0, 0, 1))
        r_ /= np.linalg.norm(r_)
        u = np.cross(r_, f)
        px, py, pz = x - c[0], y - c[1], z - c[2]
        depth = px * f[0] + py * f[1] + pz * f[2]
        sx = (px * r_[0] + py * r_[1] + pz * r_[2]) / np.maximum(depth, 1e-3) * PIXELS
        sy = (px * u[0] + py * u[1] + pz * u[2]) / np.maximum(depth, 1e-3) * PIXELS
        return (depth > 4) & (np.abs(sx) < 960 * margin) & (np.abs(sy) < 540 * margin), np.sqrt(px * px + py * py + pz * pz)

    def scatter(x, y, gap):
        grid, keep = {}, []
        size = float(gap.max())
        for k, (a, b, g) in enumerate(zip(x.tolist(), y.tolist(), gap.tolist())):
            i, j = int(a // size), int(b // size)
            ok = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for u, v in grid.get((i + di, j + dj), ()):
                        if (a - u) ** 2 + (b - v) ** 2 < g * g:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                grid.setdefault((i, j), []).append((a, b))
                keep.append(k)
        return np.array(keep, int)

    def houses(h, fine):
        """Villages on the slopes that face the valley. They follow the contours, as terraces do, so the lights draw the shape of the mountain."""
        r = np.random.default_rng(SEED + 5)
        cell = SIZE / (N - 1)
        hs = blur(h, 2.2 / cell)
        gy, gx = np.gradient(hs, cell)
        clump = spectral(1024, 2.9, SEED + 6)
        hamlet = spectral(1024, 2.2, SEED + 7)
        m = 6000000
        x, y = r.uniform(-SIZE / 2, SIZE / 2, m).astype(np.float32), r.uniform(CENTRE_Y - SIZE / 2, CENTRE_Y + SIZE / 2, m).astype(np.float32)
        z = height_at(h, x, y)
        cx_, cy_, side_ = PATCH
        inside = (np.abs(x - cx_) < side_ / 2 - 2) & (np.abs(y - cy_) < side_ / 2 - 2)
        z[inside] = height_fine(fine, x[inside], y[inside])
        vis, d = frame_mask(x, y, z, CAM_HERO, TARGET_HERO)
        ix, iy = np.clip(((x + SIZE / 2) / cell).astype(int), 0, N - 1), np.clip(((y - CENTRE_Y + SIZE / 2) / cell).astype(int), 0, N - 1)
        sx, sy = gx[iy, ix], gy[iy, ix]
        zs = hs[iy, ix]
        slope = np.hypot(sx, sy)
        dx, dy, dz = CAM_HERO[0] - x, CAM_HERO[1] - y, CAM_HERO[2] - z
        facing = (-sx * dx - sy * dy + dz) / np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1)      # the surface normal toward the camera
        c = clump[(y * 1024 / 420).astype(np.int64) % 1024, (x * 1024 / 420).astype(np.int64) % 1024]
        c2 = hamlet[(y * 1024 / 150).astype(np.int64) % 1024, (x * 1024 / 150).astype(np.int64) % 1024]
        f_ = (zs / 3.2 + .35 * c2) % 1
        band = sm((np.abs(f_ - .5) - .34) / .1)                 # rows along the contours, as houses stand on terraces, so the lights draw the shape of the mountain
        sdf = bank_sdf(x, y)
        village = np.maximum(.01 + .99 * sm((c + .1) / .7), .95 * np.exp(-(np.maximum(sdf, 0) / 62) ** 2) * (y < 560))      # the most of them along the river, all the way up the valley
        p = village * (.25 + .75 * sm((c2 + .5) / .8)) * (.15 + .85 * band) * sm(facing / .2) * sm((2.2 - slope) / 1.1) * sm((zs - 1.2) / 2.5)
        p *= .08 + .92 * np.exp(-zs / 120)      # thinner higher up, but the lights go up the mountain
        f2 = (zs / 5.2 + .2 * c2) % 1
        p = np.where(d < NEAR_D, .6 * (.04 + .96 * sm((np.abs(f2 - .5) - .3) / .1)) * sm(facing / .15) * sm((2.4 - slope) / 1.1) * sm((zs - 1.6) / 1.5), p)     # the near hill is one village, row above row
        p *= sm(sdf / 3 - .2)
        p *= d > 80                             # closest of all, only the windows we can read
        p *= np.where((x < -30) & (d < 260), .3, 1)     # the far shore close to is mostly trees, or its roofs stack into a staircase
        hero = hero_places(fine)
        for hx_, hy_, _, hs_ in hero:
            p *= (x - hx_) ** 2 + (y - hy_) ** 2 > (1.5 * hs_) ** 2
        keep = vis & (r.random(m) < p)
        order = np.nonzero(keep)[0]
        wide = window_px(d[order]) * d[order] / PIXELS
        gap = np.where(d[order] < NEAR_D, 6.4, np.maximum(2.1 * wide, 2.6)).astype(np.float32)       # close to, a house has room round it
        order = order[scatter(x[order], y[order], gap)]
        print("houses", len(order))
        x, y, z, d = x[order], y[order], z[order], d[order]
        yaw = np.arctan2(-(CAM_HERO[0] - x), CAM_HERO[1] - y) * 1.0 + r.normal(0, .22, x.size)
        yaw = yaw + np.pi       # local -y points at the camera
        return x, y, z, d, yaw, r, hero

    def switching(x, y, d, r):
        """When each window comes on and when it goes out, each as a fraction of its phase. x, y: the first window first, then the rest.
        On: from the first outwards. Off: the ones we can see clearly go first, one by one; then the many; then a last few alone; the first one last of all."""
        n = x.size
        far = np.hypot(x - x[0], y - y[0])
        on = (far / far.max()) ** .7 * .94 + r.uniform(0, .05, n)
        off = .1 + .74 * r.random(n)
        close = np.nonzero(d < 130)[0]
        off[close] = r.uniform(0, .12, close.size)
        last = r.choice(n - 1, min(14, n - 1), replace=False) + 1
        off[last] = np.linspace(.86, .98, last.size)
        on[0], off[0] = 0, 1
        return on, off

    def lit(t, on, off):
        """t runs 0-1 as they come on and 1-2 as they go out; past 2 even the first is out. So the animation is only a loop over t."""
        if t <= 1:
            return sm((t - on) / .03 + 1)
        return 1 - sm((t - 1 - off) / .015)

    def village_lamps(x, y, z, shown, cell=38.0, gain=None):
        """What thousands of windows do together: the slope around a village is lit from below. One unseen lamp a little above each village, as strong as it has windows on."""
        gain = opt("lamps", 3.0) if gain is None else gain
        i = ((x + SIZE) // cell).astype(np.int64) + 4096 * ((y + SIZE) // cell).astype(np.int64)
        ids, inv = np.unique(i, return_inverse=True)
        n = np.bincount(inv, weights=shown)
        cx, cy, cz = np.bincount(inv, weights=x * shown) / np.maximum(n, 1e-3), np.bincount(inv, weights=y * shown) / np.maximum(n, 1e-3), np.bincount(inv, weights=z * shown) / np.maximum(n, 1e-3)
        k = 0
        for a, b, c, w in zip(cx, cy, cz, n):
            if w < 6:
                continue
            d_ = bpy.data.lights.new("glow", "POINT")
            d_.energy, d_.shadow_soft_size = gain * w, 6.0
            o = bpy.data.objects.new("glow", d_)
            bpy.context.collection.objects.link(o)
            o.location = (a, b, c + 7.0)
            o.visible_glossy = False
            k += 1
        print("village lamps", k)

    def set_attr(o, name, values):
        me = o.data
        if name not in me.attributes:
            me.attributes.new(name, "FLOAT", "POINT")
        me.attributes[name].data.foreach_set("value", np.asarray(values, np.float32))

    def attr_node(nt, name):
        a = nt.nodes.new("ShaderNodeAttribute")
        a.attribute_name = name
        return a.outputs["Fac"]

    def wall_material():
        """Limewash over a footing of stone, weathered, each wall its own age."""
        m = bpy.data.materials.new("plaster")
        m.use_nodes = True
        nt = m.node_tree
        new, L_ = nt.nodes.new, nt.links.new
        b = nt.nodes["Principled BSDF"]
        geo, n1 = new("ShaderNodeNewGeometry"), new("ShaderNodeTexNoise")
        n1.inputs["Scale"].default_value, n1.inputs["Detail"].default_value = .9, 7
        L_(geo.outputs["Position"], n1.inputs["Vector"])
        foot, mul, mul2 = new("ShaderNodeMapRange"), new("ShaderNodeMath"), new("ShaderNodeMath")
        mul.operation = mul2.operation = "MULTIPLY"
        foot.inputs["From Min"].default_value, foot.inputs["From Max"].default_value = -.22, -.1
        foot.inputs["To Min"].default_value, foot.inputs["To Max"].default_value = .38, .8
        L_(attr_node(nt, "zrel"), foot.inputs["Value"])
        L_(foot.outputs[0], mul.inputs[0])
        L_(attr_node(nt, "tint"), mul.inputs[1])
        L_(mul.outputs[0], mul2.inputs[0])
        L_(n1.outputs[0], mul2.inputs[1])
        ad = new("ShaderNodeMath")
        ad.operation = "MULTIPLY"
        ad.inputs[1].default_value = 1.9
        L_(mul2.outputs[0], ad.inputs[0])
        L_(ad.outputs[0], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = .95
        b.inputs["Specular IOR Level"].default_value = 0
        return m

    def roof_material():
        """Rows of round tile, from the distance along the roof."""
        m = bpy.data.materials.new("tile")
        m.use_nodes = True
        nt = m.node_tree
        new, L_ = nt.nodes.new, nt.links.new
        b = nt.nodes["Principled BSDF"]
        k, si = new("ShaderNodeMath"), new("ShaderNodeMath")
        k.operation, si.operation = "MULTIPLY", "SINE"
        k.inputs[1].default_value = 2 * math.pi / .32
        L_(attr_node(nt, "u"), k.inputs[0])
        L_(k.outputs[0], si.inputs[0])
        col = new("ShaderNodeMapRange")
        col.inputs["From Min"].default_value, col.inputs["From Max"].default_value = -1, 1
        col.inputs["To Min"].default_value, col.inputs["To Max"].default_value = .05, .17
        L_(si.outputs[0], col.inputs["Value"])
        L_(col.outputs[0], b.inputs["Base Color"])
        bu = new("ShaderNodeBump")
        bu.inputs["Strength"].default_value, bu.inputs["Distance"].default_value = .8, .06
        L_(si.outputs[0], bu.inputs["Height"])
        L_(bu.outputs[0], b.inputs["Normal"])
        b.inputs["Roughness"].default_value = .55
        b.inputs["Specular IOR Level"].default_value = .25
        return m

    def blob_template(lat=4, lon=7):
        V, F = [], []
        for i in range(lat + 1):
            th = math.pi * i / lat
            for j in range(lon):
                ph = 2 * math.pi * j / lon
                V.append((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th) * .8))
        for i in range(lat):
            for j in range(lon):
                a, b = i * lon + j, i * lon + (j + 1) % lon
                F.append((a, b, b + lon, a + lon))
        return np.array(V, np.float32), np.array(F)

    def trees(x, y, z, d, h, fine, r):
        """Between the houses and along the tops, dark trees: the dabs of ink a painter puts round a village. They also break up the rows."""
        m = 34000
        tx, ty = r.uniform(-128, 180, m).astype(np.float32), r.uniform(-150, 110, m).astype(np.float32)
        tz = height_fine(fine, tx, ty)
        keep = (bank_sdf(tx, ty) > 4) & (tz > 2.5)
        near_house = np.zeros(m, bool)
        cell = {}
        hp = hero_places(fine)
        for a, b in list(zip(x[d < 320].tolist(), y[d < 320].tolist())) + [(u_, v_) for u_, v_ in hp[:, :2].tolist()]:
            cell.setdefault((int(a // 4), int(b // 4)), []).append((a, b))
        for k in np.nonzero(keep)[0]:
            a, b = float(tx[k]), float(ty[k])
            i, j = int(a // 4), int(b // 4)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for u, v in cell.get((i + di, j + dj), ()):
                        if (a - u) ** 2 + (b - v) ** 2 < 16.0:
                            near_house[k] = True
        clump = spectral(1024, 2.6, SEED + 71)
        c_ = clump[(ty * 1024 / 90).astype(np.int64) % 1024, (tx * 1024 / 90).astype(np.int64) % 1024]
        keep &= ~near_house & (c_ > .15) & (r.random(m) < .5)
        idx = np.nonzero(keep)[0]
        print("tree dabs", len(idx))
        if not len(idx):
            return
        s_ = r.uniform(.9, 2.3, len(idx)).astype(np.float32)
        Vb, Fb = blob_template()
        V = instances(Vb, tx[idx], ty[idx], tz[idx] + s_ * .9, r.uniform(0, 6.28, len(idx)).astype(np.float32), s_)
        V = V + r.normal(0, .1, V.shape).astype(np.float32)
        # a dab of ink is a dab only from far off: none close to either camera
        w0 = hp[0][:2]
        far_off = (np.hypot(tx[idx] - CAM_HERO[0], ty[idx] - CAM_HERO[1]) > 90) & (np.hypot(tx[idx] - w0[0], ty[idx] - w0[1]) > 40)
        V = V.reshape(len(idx), -1, 3)[far_off].reshape(-1, 3)
        o = poly_mesh("trees", V, [faces_of(Fb, int(far_off.sum()), len(Vb))])
        o.data.materials.append(grey_material("leaves", .07, .95))

    def build_town(h, fine, t=1.0, name="qian", windows=True):
        x, y, z, d, yaw, r, hero = houses(h, fine)
        k_h = len(hero)
        hd = np.linalg.norm(hero[:, :3] - np.array(CAM_HERO), axis=1)
        on, off = switching(np.r_[hero[:, 0], x], np.r_[hero[:, 1], y], np.r_[hd, d], r)
        shown_all = lit(t, on, off)
        shown = shown_all[k_h:]
        scale = 2.5 * r.uniform(.88, 1.12, x.size)
        close = d < NEAR_D
        pane_scale = np.where(close, scale, window_px(d) * d / PIXELS / (PX1 - PX0))
        near = d < NEAR_D
        order_near = np.nonzero(near)[0]
        order_far = np.nonzero(~near)[0]
        Fp = np.arange(4)[None, :]
        out = {}
        bright = .35 + .65 * r.random(x.size) ** 2.2       # most of them dim, a few bright, as lamps are
        # the table every later layer works from
        zc = z + .34 * pane_scale
        sx, sy = project(x, y, zc)
        rx, ry = project(x, y, 2 * WATER_Z - zc)
        hsx, hsy = project(hero[:, 0], hero[:, 1], hero[:, 2])
        hrx, hry = project(hero[:, 0], hero[:, 1], 2 * WATER_Z - hero[:, 2])
        os.makedirs(os.path.join(OUT3D, "qian"), exist_ok=True)
        np.savez(os.path.join(OUT3D, "qian", f"windows_{name}.npz"), x=np.r_[hero[:, 0], x], y=np.r_[hero[:, 1], y], z=np.r_[hero[:, 2], zc], d=np.r_[hd, d], sx=np.r_[hsx, sx], sy=np.r_[hsy, sy], rx=np.r_[hrx, rx], ry=np.r_[hry, ry],
                 w_px=np.r_[.68 * hero[:, 3] / hd * PIXELS, (PX1 - PX0) * pane_scale / d * PIXELS], h_px=np.r_[.56 * hero[:, 3] / hd * PIXELS, (PZ1 - PZ0) * pane_scale / d * PIXELS],
                 bright=np.r_[np.ones(k_h), bright], on=on, off=off, n_hero=k_h)
        print("windows", k_h + x.size, "first at", hero[0])
        if not windows:
            near, order_near, order_far = near * 0, order_near[:0], order_far[:0]
        for key, idx, sampled, E in (("near", order_near, True, 18.0), ("far", order_far, False, 40.0)):
            if not len(idx):
                continue
            V = instances(PANE, x[idx], y[idx], z[idx], yaw[idx], pane_scale[idx])
            o = poly_mesh("win_" + key, V, [faces_of(Fp, len(idx), 4)])
            o.data.materials.append(window_material("win_" + key, E, sampled))
            o.visible_camera = False        # the camera sees them in their own pass; the denoiser would smear them here
            o.visible_diffuse = sampled     # only the near ones light the ground for real
            o.visible_glossy = False        # the water's reflections are their own layer, so each can go out with its window
            set_lit(o, bright[idx] * shown[idx])
            out[key] = (o, idx, bright[idx])
        # every window that is more than a speck has bars, or it would not be a window: three by three up close, a cross further off
        wood = grey_material("wood", .05, .7)
        for name, idx, tmpl in (("bars", np.nonzero(close)[0], lattice_template()), ("bars_far", np.nonzero(~close & (d < 340))[0], lattice_template(2, 2, .03))):
            if len(idx):
                bars = poly_mesh(name, instances(tmpl[0], x[idx], y[idx], z[idx], yaw[idx], pane_scale[idx]), [faces_of(tmpl[1], len(idx), len(tmpl[0]))])
                bars.data.materials.append(wood)
        # the houses themselves, as far as they can be seen: no two the same width or the same height, walls of different ages
        idx = np.nonzero(d < 300)[0]
        if len(idx):
            k_ = len(idx)
            sx_, sz_ = r.uniform(.95, 1.45, k_), np.maximum(r.normal(1.12, .16, k_), 1.0)
            flip = r.choice([-1.0, 1.0], k_)
            Vw = instances(WALL, x[idx], y[idx], z[idx], yaw[idx], scale[idx], sx_, sz_)
            walls = poly_mesh("walls", Vw, [faces_of(WALL_Q, k_, 10), faces_of(WALL_P, k_, 10)])
            walls.data.materials.append(wall_material())
            set_attr(walls, "tint", np.repeat(r.uniform(.72, 1.0, k_), 10))
            set_attr(walls, "zrel", np.tile(WALL[:, 2], k_))
            Vr = instances(ROOF_V, x[idx], y[idx], z[idx], yaw[idx], scale[idx], sx_, sz_)
            roofs = poly_mesh("roofs", Vr, [faces_of(ROOF_F, k_, len(ROOF_V))])
            roofs.data.materials.append(roof_material())
            set_attr(roofs, "u", (ROOF_V[:, 0][None, :] * (scale[idx] * sx_)[:, None]).ravel())
            ridge = np.array(box(-L - O_ - .03, L + O_ + .03, -.035, .035, H_ + R_ - .03, H_ + R_ + .05), np.float32)
            rid_ = poly_mesh("ridges", instances(ridge, x[idx], y[idx], z[idx], yaw[idx], scale[idx], sx_, sz_), [faces_of(BOX_Q, k_, 8)])
            rid_.data.materials.append(grey_material("ridge", .05, .8))
            Vd = instances(DOOR, x[idx], y[idx], z[idx], yaw[idx], scale[idx], sx_ * flip)
            doors = poly_mesh("doors", Vd, [faces_of(np.arange(4)[None, :], k_, 4)])
            doors.data.materials.append(grey_material("door", .07, .8))
        trees(x, y, z, d, h, fine, r)
        village_lamps(x, y, z, shown)
        shared = dict(plaster=textured("plaster_h", plaster_paint(), .95, .35), wood=grey_material("wood_h", .08, .75), sill=grey_material("sill", .3, .9),
                      foot=grey_material("footing", .22, .95), tile=grey_material("tile_h", .1, .55), dark=grey_material("pane_back", .03, .9))
        tex = list(np.load(os.path.join(CACHE, "pane_tex.npy")))
        big = os.path.join(CACHE, "pane_first.npy")     # the first window fills the frame at the start of the pull-back
        if os.path.exists(big):
            tex[0] = np.load(big)
        for i, (hx_, hy_, hz_, hs_) in enumerate(hero):
            fine_house(i, np.array([hx_, hy_, hz_]), hs_, tex[i], float(shown_all[i]), shared)
        return out, (x, y, z, d)

    def night_world(diffuse=.05, glossy=.42, horizon=None):
        """Very little light for the rock, but water sees a pale sky, as water does at night, palest low down: so the far reaches of the river are silver and the near ones dark."""
        w = bpy.context.scene.world
        nt = w.node_tree
        bg = nt.nodes["Background"]
        lp, tc, sp = nt.nodes.new("ShaderNodeLightPath"), nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["Generated"], sp.inputs[0])
        el = nt.nodes.new("ShaderNodeMapRange")
        el.interpolation_type = "SMOOTHSTEP"
        el.inputs["From Min"].default_value, el.inputs["From Max"].default_value = .01, .42
        el.inputs["To Min"].default_value, el.inputs["To Max"].default_value = glossy if horizon is None else horizon, glossy
        nt.links.new(sp.outputs["Z"], el.inputs["Value"])
        ad = nt.nodes.new("ShaderNodeMapRange")
        ad.inputs["From Min"].default_value, ad.inputs["From Max"].default_value = 0, 1
        ad.inputs["To Min"].default_value = diffuse
        nt.links.new(el.outputs[0], ad.inputs["To Max"])
        nt.links.new(lp.outputs["Is Glossy Ray"], ad.inputs["Value"])
        nt.links.new(ad.outputs[0], bg.inputs[1])
        bg.inputs[0].default_value = (1, 1, 1, 1)

    def rock_material(name, paint):
        m = painted_material(name, paint)
        nt = m.node_tree
        b = nt.nodes["Principled BSDF"]
        b.inputs["Specular IOR Level"].default_value = 0
        new, L_ = nt.nodes.new, nt.links.new
        tc, n1, n2, bu, add = new("ShaderNodeTexCoord"), new("ShaderNodeTexNoise"), new("ShaderNodeTexNoise"), new("ShaderNodeBump"), new("ShaderNodeMath")
        n1.inputs["Scale"].default_value, n1.inputs["Detail"].default_value = .22, 9
        n2.inputs["Scale"].default_value, n2.inputs["Detail"].default_value = 2.1, 7
        L_(tc.outputs["Object"], n1.inputs["Vector"])
        L_(tc.outputs["Object"], n2.inputs["Vector"])
        add.operation = "ADD"
        L_(n1.outputs[0], add.inputs[0])
        L_(n2.outputs[0], add.inputs[1])
        bu.inputs["Strength"].default_value, bu.inputs["Distance"].default_value = .9, 1.4
        L_(add.outputs[0], bu.inputs["Height"])
        L_(bu.outputs[0], b.inputs["Normal"])
        return m

    def make_water():
        """A mirror that the air has just touched: ripples drawn out sideways, so what stands on the bank stands on a broken reflection."""
        wx, wy = np.meshgrid(np.linspace(-SIZE / 2, SIZE / 2, 2), np.linspace(CENTRE_Y - SIZE / 2, CENTRE_Y + SIZE / 2, 2))
        water = mesh_grid("water", wx.astype(np.float32), wy.astype(np.float32), np.full((2, 2), .22, np.float32))
        m = water_material(rough=.025, normal_scale=1.0, normal_strength=opt("ripple", .14))
        nt = m.node_tree
        new, L_ = nt.nodes.new, nt.links.new
        noise = next(n for n in nt.nodes if n.type == "TEX_NOISE")
        tc = next(n for n in nt.nodes if n.type == "TEX_COORD")
        mp = new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (.35, 2.6, 1)
        L_(tc.outputs["Object"], mp.inputs["Vector"])
        L_(mp.outputs[0], noise.inputs["Vector"])
        noise.inputs["Detail"].default_value = 5
        next(n for n in nt.nodes if n.type == "BUMP").inputs["Distance"].default_value = .12
        water.data.materials.append(m)

    def moon_light():
        """A moon behind thin cloud: mostly it comes from the whole sky, a little from where it stands, a little off the mist behind us. Ink3d.sun's horizontal part comes out mirrored."""
        night_world(opt("diffuse", .05), opt("glossy", .3), opt("horizon", .58))
        sun((-.5, .7, -.55), energy=opt("moon", .5), angle=14)
        sun((.35, -.7, -.6), energy=opt("fill", .12), angle=35)

    def render_windows(name):
        """The windows alone, with everything else cutting them off."""
        path = name + "_w.png"
        if os.path.exists(path):
            return
        sc = bpy.context.scene
        keep = sc.cycles.samples, sc.cycles.use_denoising, sc.cycles.max_bounces, sc.cycles.filter_width
        sc.cycles.samples, sc.cycles.use_denoising, sc.cycles.max_bounces, sc.cycles.filter_width = 48, False, 0, 1.1
        hidden = []
        for o in bpy.context.collection.objects:
            if o.type == "LIGHT":
                o.hide_render = True
                hidden.append(o)
            elif o.type == "MESH" and not o.name.startswith("win_"):
                o.is_holdout = True
                hidden.append(o)
        bg = sc.world.node_tree.nodes["Background"]
        for link in list(bg.inputs[1].links):
            sc.world.node_tree.links.remove(link)
        bg.inputs[1].default_value = 0
        for o in bpy.context.collection.objects:
            if o.type == "LIGHT":
                o.hide_render = True
        sc.view_settings.view_transform = "Raw"
        for o in bpy.context.collection.objects:
            if o.name.startswith("win_"):
                o.visible_camera = True
                for n_ in o.data.materials[0].node_tree.nodes:
                    if n_.type == "MATH":
                        n_.inputs[1].default_value = 1.0        # here its brightness is the lamp's: most dim, a few bright
            elif o.name.startswith("back"):
                o.hide_render = True
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)

    def border(sc):
        """--border=x0,y0,x1,y1, in pixels at full size: render only that part of the frame. `python ink3d_qian.py patch` lays it over an earlier render."""
        b = opt("border", "")
        if b:
            x0, y0, x1, y1 = [float(v) for v in b.split(",")]
            sc.render.use_border, sc.render.use_crop_to_border = True, False
            sc.render.border_min_x, sc.render.border_max_x = x0 / 1920, x1 / 1920
            sc.render.border_min_y, sc.render.border_max_y = 1 - y1 / 1080, 1 - y0 / 1080

    def scene_qian(shot="qian", t=None):
        sc = reset(samples=opt("samples", 64), pct=PCT)
        border(sc)
        X, Y, h = land()
        ground = mesh_grid("land", X, Y, h)
        paint = cached("paint", lambda: ink_albedo(h, SEED + 3, SIZE / N))
        rock = rock_material("rock", paint)
        ground.data.materials.append(rock)
        make_water()
        hx, hy, fine = near_hill()
        cx_, cy_, side_ = PATCH
        dip = (np.abs(X - cx_) < side_ / 2 - 1.5) & (np.abs(Y - cy_) < side_ / 2 - 1.5)
        ground.data.vertices.foreach_set("co", np.stack([X, Y, np.where(dip, h - 1.5, h)], -1).ravel().astype(np.float32))
        ground.data.update()
        hill = mesh_grid("hill", hx, hy, fine)
        hill_paint = ink_albedo(fine, SEED + 4, side_ / fine.shape[0])
        hill.data.materials.append(rock_material("rock2", hill_paint))
        town, _ = build_town(h, fine, T if t is None else t, shot)
        moon_light()
        camera(CAM_HERO, TARGET_HERO, lens=LENS, far=FAR_)
        name = os.path.join(OUT3D, "qian", "dev", shot + TAG)
        render(name)
        render_windows(name)

    # ---- the chat window in the riverbed ------------------------------------------------------------------------------------------------------------------------
    def scene_chat():
        """The same ground chapter 8 left, the same place in its long stroke: its height and its paint, with the window let into the stone under a skin of water."""
        sc = reset(samples=opt("samples", 96), pct=PCT)
        sc.cycles.max_bounces = 8
        tex = os.path.join(OUT3D, "bed", "tex")
        h0, a0 = np.load(os.path.join(tex, "bed_h.npy"), mmap_mode="r"), np.load(os.path.join(tex, "bed_a.npy"), mmap_mode="r")
        cx, cy = CHAT_C
        R, B, k = 46.0, 440.0, 2
        n0, t0 = h0.shape[0], a0.shape[0]
        i0, i1 = int((cx - R + B / 2) / B * n0), int((cx + R + B / 2) / B * n0)
        j0, j1 = int((cy - R + B / 2) / B * n0), int((cy + R + B / 2) / B * n0)
        hm = np.array(h0[j0:j1, i0:i1], np.float32).repeat(k, 0).repeat(k, 1)
        hm = blur(hm, k * .7) + .012 * spectral(hm.shape[0], 1.6, SEED + 31, hm.shape[1])
        n, m = hm.shape
        x0, x1 = (i0 / n0 - .5) * B, (i1 / n0 - .5) * B
        y0, y1 = (j0 / n0 - .5) * B, (j1 / n0 - .5) * B
        X, Y = np.meshgrid(np.linspace(x0, x1, m).astype(np.float32), np.linspace(y0, y1, n).astype(np.float32))
        ux, uy = CHAT_U
        vx, vy = uy, -ux        # local right
        U, V = (X - cx) * ux + (Y - cy) * uy, (X - cx) * vx + (Y - cy) * vy
        W_, H_, r_ = CHAT_W, CHAT_H, .5
        sdf = np.hypot(np.maximum(np.abs(V) - W_ / 2 + r_, 0), np.maximum(np.abs(U) - H_ / 2 + r_, 0)) - r_
        z0 = float(np.median(hm[sdf < 1]))
        flat = sm(1 - (sdf - .5) / 2.6)
        hm = hm * (1 - flat) + z0 * flat
        hm = np.where(sdf < 0, z0 - .015, hm) + .012 * np.exp(-(sdf / .12) ** 2)      # let into the stone, with a worn lip
        land = mesh_grid("land", X, Y, hm)
        ti0, ti1 = int(i0 / n0 * t0), int(i1 / n0 * t0)
        tj0, tj1 = int(j0 / n0 * t0), int(j1 / n0 * t0)
        paint = np.array(a0[tj0:tj1, ti0:ti1], np.float32) / 255
        m_ = bpy.data.materials.new("stone")
        m_.use_nodes = True
        nt = m_.node_tree
        new, L_ = nt.nodes.new, nt.links.new
        b = nt.nodes["Principled BSDF"]
        t_ = new("ShaderNodeTexImage")
        t_.image = ink3d.image_from(paint, "paint")
        t_.interpolation = "Cubic"
        L_(t_.outputs[0], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = opt("stone_rough", .34)
        b.inputs["Specular IOR Level"].default_value = .5
        tc, n1, n2, bu = new("ShaderNodeTexCoord"), new("ShaderNodeTexNoise"), new("ShaderNodeTexNoise"), new("ShaderNodeBump")
        n1.inputs["Scale"].default_value, n1.inputs["Detail"].default_value = 1.1, 8
        n2.inputs["Scale"].default_value, n2.inputs["Detail"].default_value = 9.0, 6
        L_(tc.outputs["Object"], n1.inputs["Vector"])
        L_(tc.outputs["Object"], n2.inputs["Vector"])
        add = new("ShaderNodeMath")
        add.operation = "ADD"
        L_(n1.outputs[0], add.inputs[0])
        L_(n2.outputs[0], add.inputs[1])
        bu.inputs["Strength"].default_value, bu.inputs["Distance"].default_value = .5, .12
        L_(add.outputs[0], bu.inputs["Height"])
        L_(bu.outputs[0], b.inputs["Normal"])
        land.data.materials.append(m_)
        # the window: a rounded pane of lit paper. What the camera sees is the pane; what lights the stone is an unseen soft lamp just above it, since a flat light cannot light the flat stone beside it
        th = np.linspace(0, 2 * math.pi, 73)[:-1]
        cor = [(np.sign(np.cos(a)) * (W_ / 2 - r_) + r_ * np.cos(a), np.sign(np.sin(a)) * (H_ / 2 - r_) + r_ * np.sin(a)) for a in th + .01]
        Pv = np.array([c[0] for c in cor], np.float32)
        Pu = np.array([c[1] for c in cor], np.float32)
        V3 = np.stack([cx + Pu * ux + Pv * vx, cy + Pu * uy + Pv * vy, np.full(len(Pu), z0 - .005, np.float32)], -1)
        pane = poly_mesh("pane", V3, [np.arange(len(Pu))[None, :]])
        uvl = pane.data.uv_layers.new(name="UV")
        uvl.data.foreach_set("uv", np.stack([Pv / W_ + .5, Pu / H_ + .5], -1).ravel().astype(np.float32))
        chat = np.load(os.path.join(CACHE, "chat_tex.npy"))
        pm = bpy.data.materials.new("pane")
        pm.use_nodes = True
        nt = pm.node_tree
        nt.nodes.clear()
        ti, em, ou = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
        ti.image = ink3d.image_from(chat, "chat")
        ti.interpolation = "Cubic"
        nt.links.new(ti.outputs[0], em.inputs["Color"])
        em.inputs["Strength"].default_value = opt("pane", 1.5)
        nt.links.new(em.outputs[0], ou.inputs["Surface"])
        pane.data.materials.append(pm)
        pane.visible_diffuse = False
        lamp_v = np.stack([cx + 1.25 * (Pu * ux + Pv * vx), cy + 1.25 * (Pu * uy + Pv * vy), np.full(len(Pu), z0 + opt("lamp_h", 2.4), np.float32)], -1)
        lamp = poly_mesh("lamp", lamp_v, [np.arange(len(Pu))[None, :]])
        lm = bpy.data.materials.new("lamp")
        lm.use_nodes = True
        nt = lm.node_tree
        nt.nodes.clear()
        em, ou = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
        em.inputs["Strength"].default_value = opt("spill", 3.2)
        nt.links.new(em.outputs[0], ou.inputs["Surface"])
        lamp.data.materials.append(lm)
        lamp.visible_camera = lamp.visible_glossy = lamp.visible_transmission = False
        # the skin of water
        wx, wy = np.meshgrid(np.array([x0, x1], np.float32), np.array([y0, y1], np.float32))
        water = mesh_grid("water", wx, wy, np.full((2, 2), z0 + opt("film", .07), np.float32))
        wm = bpy.data.materials.new("film")
        wm.use_nodes = True
        nt = wm.node_tree
        b = nt.nodes["Principled BSDF"]
        b.inputs["Transmission Weight"].default_value = 1.0
        b.inputs["IOR"].default_value = 1.33
        b.inputs["Roughness"].default_value = .02
        tc, n1, bu = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeBump")
        n1.inputs["Scale"].default_value, n1.inputs["Detail"].default_value = 1.3, 4
        nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
        bu.inputs["Strength"].default_value, bu.inputs["Distance"].default_value = opt("film_bump", .22), .05
        nt.links.new(n1.outputs[0], bu.inputs["Height"])
        nt.links.new(bu.outputs[0], b.inputs["Normal"])
        water.data.materials.append(wm)
        water.visible_shadow = False
        night_world(opt("diffuse", .05), opt("glossy", .3))
        sun((-.5, .7, -.55), energy=opt("moon", .22), angle=14)
        el, D = math.radians(opt("elev", 50.0)), opt("dist", 20.0)
        cam = (cx - ux * D * math.cos(el), cy - uy * D * math.cos(el), z0 + D * math.sin(el))
        camera(cam, (cx + vx * .9, cy + vy * .9, z0), lens=35.0, far=FAR_)
        render(os.path.join(OUT3D, "qian", "dev", "chat" + TAG))

    # ---- the first window, up close ------------------------------------------------------------------------------------------------------------------------------
    def local_object(name, V, polys, frame):
        o = poly_mesh(name, np.asarray(V, np.float32), polys)
        o.matrix_world = frame
        return o

    def boxes(name, spec, frame, mat):
        """spec: (x0, x1, y0, y1, z0, z1) in the house's own axes, all one mesh."""
        V = np.array([box(*b) for b in spec], np.float32).reshape(-1, 3)
        F = np.concatenate([np.array(BOX_Q + [(3, 2, 1, 0)]) + 8 * i for i in range(len(spec))])
        o = local_object(name, V, [F], frame)
        o.data.materials.append(mat)
        return o

    def textured(name, img, rough=.9, bump=.0, rect=(-2.0, 2.0, -1.2, 1.0)):
        """A material painted from an array laid over the house's front: x along its width, z up its height."""
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        new, L_ = nt.nodes.new, nt.links.new
        b = nt.nodes["Principled BSDF"]
        tc, mp, t_ = new("ShaderNodeTexCoord"), new("ShaderNodeMapping"), new("ShaderNodeTexImage")
        mp.inputs["Location"].default_value = (-rect[0] / (rect[1] - rect[0]), -rect[2] / (rect[3] - rect[2]), 0)
        mp.inputs["Scale"].default_value = (1 / (rect[1] - rect[0]), 1 / (rect[3] - rect[2]), 1)
        sw, cb = new("ShaderNodeSeparateXYZ"), new("ShaderNodeCombineXYZ")
        L_(tc.outputs["Object"], sw.inputs[0])
        L_(sw.outputs["X"], cb.inputs["X"])
        L_(sw.outputs["Z"], cb.inputs["Y"])
        L_(cb.outputs[0], mp.inputs["Vector"])
        L_(mp.outputs[0], t_.inputs[0])
        t_.image = ink3d.image_from(img, name + "_paint")
        t_.interpolation = "Cubic"
        t_.extension = "EXTEND"
        L_(t_.outputs[0], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = rough
        b.inputs["Specular IOR Level"].default_value = .15
        if bump:
            bu = new("ShaderNodeBump")
            bu.inputs["Strength"].default_value, bu.inputs["Distance"].default_value = bump, .02
            L_(t_.outputs[0], bu.inputs["Height"])
            L_(bu.outputs[0], b.inputs["Normal"])
        return m

    def plaster_paint(n=1024):
        """An old lime wall: water has run down it from the eaves, damp has come up it from the ground, and in places the lime has come away from the mud."""
        r = np.random.default_rng(SEED + 50)
        a = .66 + .05 * spectral(n, 2.8, SEED + 51) + .012 * spectral(n, .8, SEED + 52)
        col = blur(r.normal(0, 1, (8, n)).astype(np.float32), 2.0)[0]
        col = (col - col.mean()) / col.std()
        z = np.linspace(0, 1, n)[:, None]
        run = np.clip(col[None, :] + .55 * spectral(n, 2.4, SEED + 53), 0, 3)
        a = a - .13 * run * np.exp(-(1 - z) / .33)
        a = a - .22 * sm((.16 + .05 * spectral(n, 2.6, SEED + 54) - z) / .1)
        flake = sm((spectral(n, 2.5, SEED + 55) - 1.55) / .08)
        a = a * (1 - .5 * flake)
        return np.clip(a, .04, 1).astype(np.float32)

    def paper_pane(n=512):
        """Window paper with a lamp behind it, brighter in the middle, the grain showing."""
        y, x = np.mgrid[:n, :n].astype(np.float32) / (n - 1) - .5
        a = 1.0 - .22 * (x * x + y * y) * 4 / 2 + .012 * spectral(n, 1.2, SEED + 56)
        return np.clip(a, 0, 1).astype(np.float32)

    def tile_rolls(x0, x1, step, y_eave, z_eave, y_ridge, z_ridge, rad=.042):
        """Round tiles running up the roof, with a lip at each course, and a disc at the eave end of each."""
        V, Q, E = [], [], []
        arc, rings = 8, 15
        for i, xc in enumerate(np.arange(x0, x1, step)):
            base = len(V)
            for j in range(rings):
                f = j / (rings - 1)
                bulge = 1 + .14 * (1 - (j % 2))
                y_, z_ = y_eave + (y_ridge - y_eave) * f, z_eave + (z_ridge - z_eave) * f
                for a in range(arc):
                    th = math.pi * a / (arc - 1)
                    V.append((xc + math.cos(th) * rad * bulge, y_ - .25 * math.sin(th) * rad, z_ + math.sin(th) * rad * bulge))
            for j in range(rings - 1):
                for a in range(arc - 1):
                    p = base + j * arc + a
                    Q.append((p, p + 1, p + arc + 1, p + arc))
            E.append(tuple(base + a for a in range(arc)))
        return np.array(V, np.float32), np.array(Q), np.array(E)

    def fine_house(i, w0, s_, tex, shown, shared):
        """A house we can look at: lime wall, a wooden window with paper behind its bars, a tiled roof with a real eave. w0 is the middle of the window, s_ the house's size.
        The bars are a wide middle pane with a border of small ones, so there is room on the paper for what is written there."""
        from mathutils import Matrix
        line = np.array(CAM_HERO) - w0
        b_ = np.array([line[0], line[1], 0.0])
        b_ /= np.linalg.norm(b_)
        ex, ey = np.array([b_[1], -b_[0], 0.0]) * -1, -b_
        ex = -ex if ex[0] < 0 else ex
        turn = 0.0 if i == 0 else .16 * (1 if i % 2 else -1)     # the first window squarely faces the line of the pull-back: it is the Claude window on the screen, edge for edge, when that starts
        ex, ey = ex * math.cos(turn) + ey * math.sin(turn), -ex * math.sin(turn) + ey * math.cos(turn)
        frame = Matrix(((ex[0] * s_, ey[0] * s_, 0, w0[0]), (ex[1] * s_, ey[1] * s_, 0, w0[1]), (0, 0, s_, w0[2]), (0, 0, 0, 1)))
        W2, H2, wl, wr, zt, zb_, wall_d = .34, .28, -.62, .62, .55, -1.1, .9
        plaster, wood = shared["plaster"], shared["wood"]
        tag = str(i)
        boxes("wall" + tag, [(wl, wr, 0, wall_d, zb_, -H2), (wl, wr, 0, wall_d, H2, zt), (wl, -W2, 0, wall_d, -H2, H2), (W2, wr, 0, wall_d, -H2, H2)], frame, plaster)
        fr, dp, bar = .03, .05, .007
        spec = [(-W2 - fr, W2 + fr, -dp, .012, H2, H2 + fr), (-W2 - fr, W2 + fr, -dp, .012, -H2 - fr, -H2), (-W2 - fr, -W2, -dp, .012, -H2, H2), (W2, W2 + fr, -dp, .012, -H2, H2)]
        for xc in (-W2 * .68, W2 * .68):
            spec.append((xc - bar, xc + bar, -dp * .7, .012, -H2, H2))
        for zc in (-H2 * .7, H2 * .7):
            spec.append((-W2, W2, -dp * .7, .012, zc - bar, zc + bar))
        boxes("bars" + tag, spec, frame, wood)
        boxes("sill" + tag, [(-W2 - .1, W2 + .1, -.09, .02, -H2 - .07, -H2 - fr)], frame, shared["sill"])
        boxes("foot" + tag, [(wl - .04, wr + .04, -.05, wall_d + .05, zb_, -.52)], frame, shared["foot"])
        back = local_object("back" + tag, [(-W2, .04, -H2), (W2, .04, -H2), (W2, .04, H2), (-W2, .04, H2)], [np.array([(0, 1, 2, 3)])], frame)
        back.data.materials.append(shared["dark"])
        pane = local_object("win_hero" + tag, [(-W2, .03, -H2), (W2, .03, -H2), (W2, .03, H2), (-W2, .03, H2)], [np.array([(0, 1, 2, 3)])], frame)
        uvl = pane.data.uv_layers.new(name="UV")
        uvl.data.foreach_set("uv", np.array([0, 0, 1, 0, 1, 1, 0, 1], np.float32))
        pm = bpy.data.materials.new("paper" + tag)
        pm.use_nodes = True
        nt = pm.node_tree
        nt.nodes.clear()
        ti, em, ou = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
        ti.image = ink3d.image_from(tex, "paper" + tag)
        nt.links.new(ti.outputs[0], em.inputs["Color"])
        em.inputs["Strength"].default_value = shown
        nt.links.new(em.outputs[0], ou.inputs["Surface"])
        pane.data.materials.append(pm)
        pane.visible_camera = pane.visible_diffuse = pane.visible_glossy = False
        if shown > .01:
            # what the window throws: an unseen lamp just outside it, or a flat pane in a flat wall would light nothing but the sill
            lamp = local_object("lamp" + tag, [(-W2, -.26, -H2), (W2, -.26, -H2), (W2, -.26, H2), (-W2, -.26, H2)], [np.array([(0, 1, 2, 3)])], frame)
            lm = bpy.data.materials.new("lamp" + tag)
            lm.use_nodes = True
            nt = lm.node_tree
            nt.nodes.clear()
            em, ou = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
            em.inputs["Strength"].default_value = opt("spill", 2.2) * shown
            nt.links.new(em.outputs[0], ou.inputs["Surface"])
            lamp.data.materials.append(lm)
            lamp.visible_camera = lamp.visible_glossy = lamp.visible_shadow = False
        y_e, z_e, y_r, z_r = -.5, zt - .02, wall_d / 2, zt + .5
        boxes("rafters" + tag, [(x_, x_ + .03, y_e + .03, 0, zt - .03, zt - .005) for x_ in np.arange(wl, wr, .22)], frame, wood)
        roof_v = [(wl - .12, y_e, z_e - .02), (wr + .12, y_e, z_e - .02), (wr + .12, y_r, z_r - .02), (wl - .12, y_r, z_r - .02),
                  (wl - .12, wall_d - y_e, z_e - .02), (wr + .12, wall_d - y_e, z_e - .02)]
        rb = local_object("roof" + tag, roof_v, [np.array([(0, 1, 2, 3), (3, 2, 5, 4)])], frame)
        rb.data.materials.append(shared["tile"])
        tv, tq, te = tile_rolls(wl - .1, wr + .12, .085, y_e, z_e, y_r, z_r, rad=.036)
        tiles = local_object("tiles" + tag, tv, [tq, te], frame)
        tiles.data.materials.append(shared["tile"])
        gv = [(wr, 0, zt), (wr, wall_d, zt), (wr, wall_d / 2, zt + .5), (wl, 0, zt), (wl, wall_d, zt), (wl, wall_d / 2, zt + .5)]
        gb = local_object("gable" + tag, gv, [np.array([(0, 1, 2), (3, 4, 5)])], frame)
        gb.data.materials.append(plaster)

    def lattice_world():
        """The town unlit but for the first window, in the same place and under the same moon as the hero frame. Returns where that window is, the line the camera slides along, and which way it looks."""
        sc = reset(samples=opt("samples", 64), pct=PCT)
        border(sc)
        sc.cycles.max_bounces = 6
        X, Y, h = land()
        ground = mesh_grid("land", X, Y, h)
        ground.data.materials.append(rock_material("rock", cached("paint", lambda: ink_albedo(h, SEED + 3, SIZE / N))))
        make_water()
        hx, hy, fine = near_hill()
        cx_, cy_, side_ = PATCH
        dip = (np.abs(X - cx_) < side_ / 2 - 1.5) & (np.abs(Y - cy_) < side_ / 2 - 1.5)
        ground.data.vertices.foreach_set("co", np.stack([X, Y, np.where(dip, h - 1.5, h)], -1).ravel().astype(np.float32))
        ground.data.update()
        hill = mesh_grid("hill", hx, hy, fine)
        hill.data.materials.append(rock_material("rock2", ink_albedo(fine, SEED + 4, side_ / fine.shape[0])))
        build_town(h, fine, 0.0, "lattice", windows=False)
        w0 = hero_places(fine)[0][:3]
        c, f, r_, u = camera_axes(CAM_HERO, TARGET_HERO)
        moon_light()
        return w0, (c - w0) / np.linalg.norm(c - w0), f

    def scene_lattice():
        """The first window from close to."""
        w0, line, f = lattice_world()
        cam = w0 + line * opt("near", 13.2)
        camera(tuple(cam), tuple(cam + f * 100), lens=LENS, far=FAR_)
        name = os.path.join(OUT3D, "qian", "dev", "lattice" + TAG)
        render(name)
        render_windows(name)

    def scene_path():
        """The pull-back itself, frame by frame: --frames=A-B of the schedule in pull_near. One build, and dev/path_NNN.png and its passes for each; the windows last, since their pass takes the scene apart."""
        w0, line, f = lattice_world()
        a_, b_ = [int(v) for v in opt("frames", f"0-{PULL_FRAMES - 1}").split("-")]
        frames = [(n, os.path.join(OUT3D, "qian", "dev", f"path_{n:03d}")) for n in range(a_, b_ + 1)]
        for n, name in frames:
            cam = w0 + line * pull_near(n)
            camera(tuple(cam), tuple(cam + f * 100), lens=LENS, far=FAR_)
            render(name)
            print("frame", n, flush=True)
        for n, name in frames:
            cam = w0 + line * pull_near(n)
            camera(tuple(cam), tuple(cam + f * 100), lens=LENS, far=FAR_)
            render_windows(name)

    {"qian": scene_qian, "chat": scene_chat, "lattice": scene_lattice, "path": scene_path, "dark": lambda: scene_qian("dark", 2.05), "going": lambda: scene_qian("going", opt("t", 1.5))}[SHOT]()


# ============================================================================================================================================================
# the ink (runs in the render environment)
# ============================================================================================================================================================
else:
    NIGHT = dict(rho=.005, mist_h=28.0, mist_d=1.05, lift=.05, contrast=.8, aerial=.0005, sky_top=1.55, sky_hor=.8, mist_floor=.48, moon=MOON_PX + (58.0,), horizon=HORIZON)
    NIGHT.update(sky_top=1.15, sky_hor=.5)
    GRADES = {"qian": dict(NIGHT, lit_mist=1.0, lights=(.6, .35, .3, .18), reflect=.75, drop=((0, 0, 1920, 215),)),
              "chat": dict(rho=0.0, lift=.02, use_win=0.0, sky_top=0.0, sky_hor=0.0),
              "lattice": dict(NIGHT, horizon=None, lit_mist=0.0, lights=(.6, .3, 0.0, 0.0), contrast=.66, lift=.1, rho=.012, mist_d=.55, mist_floor=.3, sky_top=.55, sky_hor=.2)}
    GRADES["going"] = GRADES["qian"]
    GRADES["dark"] = dict(GRADES["qian"], moon_veil=.5, moon_glow=.25)      # cloud is coming over: chapter 10 opens on black
    GRADES["dark"].update(sky_top=.85, sky_hor=.34, mist_d=.9, mist_floor=.3)

    def patch_passes(base, new, box, out, feather=40):
        """Lay a --border render over an earlier one, with a soft seam on its left, and leave the result as a render of its own."""
        import shutil
        import subprocess

        import numpy as np

        from ink_grade import load
        x0, y0, x1, y1 = box
        for suffix in (".png", "_z.png", "_w.png"):
            if not os.path.exists(base + suffix) or not os.path.exists(new + suffix):
                continue
            a, b = load(base + suffix), load(new + suffix)
            h, w = a.shape[:2]
            m = np.zeros((h, w, 1), np.float32)
            m[y0:y1, x0:x1] = 1
            m[y0:y1, x0:x0 + feather] = np.linspace(0, 1, feather, dtype=np.float32)[None, :, None]
            c = a * (1 - m) + b * m
            raw = (np.clip(c, 0, 1) * 65535 + .5).astype(np.uint16).tobytes()
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba64le", "-s", f"{w}x{h}", "-i", "-", out + suffix], input=raw, check=True)
        shutil.copy(base + ".json", out + ".json")

    # (y of the crest, height, width, phases). They rise with distance, so from low down each stands clear of the one before it
    RANGES = ((70, .252, 30, 1.0, 2.0), (170, .353, 34, 2.6, .4), (290, .48, 38, 4.1, 3.0), (430, .635, 44, .7, 1.2), (590, .81, 50, 3.3, 5.0), (760, 1.0, 58, 5.1, 2.2))
    SOURCE_Y = 500.0        # where the river begins, up among the far ranges, so the mountains close the view
    PEAKS = ((45, -25, .45, 62), (-125, -45, .2, 50), (38, 175, .72, 85))     # (x, y, height, radius): the hill the near village stands on, the low spur across the water, and the mountain that closes the valley on the right, so the river goes in at a V under the moon

    def make_land(levels, K=.05):
        """The mountains: even uplift cut by rivers that drain to the main one, so the ridges come down to it from both sides, overlapping."""
        import numpy as np
        from scipy.ndimage import zoom

        from ink3d_land import evolve, spectral

        def sm_(x):
            x = np.clip(x, 0, 1)
            return x * x * (3 - 2 * x)

        def rid(n):
            import types
            sys.modules.setdefault("bpy", types.ModuleType("bpy"))
            from ink3d import ridged
            return ridged(n, SEED + 11, base=7.0, octaves=6, warp=.06)

        def fields(n):
            y, x = np.mgrid[:n, :n].astype(np.float32)
            X, Y = (x / (n - 1) - .5) * SIZE, (y / (n - 1) - .5) * SIZE + CENTRE_Y
            d = bank_sdf(X, Y)
            low = zoom(spectral(512, 3.6, SEED + 1), n / 512, order=1)[:n, :n]
            ranges = np.full((n, n), .07, np.float32)
            valley = .3 + .7 * np.clip(d / 150, 0, 1) ** 1.2 + .7 * np.clip((Y - 300) / 160, 0, 1)       # the near ranges come down towards the river on both sides, so it runs in a long V
            valley = np.minimum(valley, 1.0)
            for y0, top, wid, a, b in RANGES:
                crest = y0 + 24 * np.sin(X / 83 + a) + 11 * np.sin(X / 31 + b)
                ranges = np.maximum(ranges, top * valley * np.exp(-((Y - crest) / wid) ** 2))
            for px, py, top, rad in PEAKS:
                ranges = np.maximum(ranges, top * np.exp(-(np.hypot(X - px, (Y - py) * 1.2) / rad) ** 2))
            u = .34 * ranges * np.clip(.8 + .25 * low, .3, 1.3) * np.clip(d / 14, 0, 1)
            m = d < 0
            m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
            return d, u, m

        h = None
        out = os.path.join(HERE, "..", "..", "ink", "film_out", "3d", "qian", "cache")
        os.makedirs(out, exist_ok=True)
        for i, (n, steps, dt, noise, diffuse) in enumerate(levels):
            d, u, m = fields(n)
            if h is None:
                h = 4 * spectral(n, 2.6, SEED + 2) + .05 * d
            else:
                h = zoom(h, n / h.shape[0], order=3)[:n, :n].astype(np.float32) + noise * spectral(n, 2.0, SEED + 3 + i)
            h[m] = 0
            h = evolve(h, u, m, SIZE / n, steps, dt, K=K, diffuse=diffuse)
            print(n, "top", h.max(), flush=True)
            final = 200 * (np.maximum(h, 0) / h.max()) ** 1.3       # steeper near the top than erosion alone leaves it, as these mountains are in paintings
            final = final * (1 + .3 * (rid(n) - .3) * sm_(final / 50))        # and broken along the crests
            y_, x_ = np.mgrid[:n, :n].astype(np.float32)
            X_, Y_ = (x_ / (n - 1) - .5) * SIZE, (y_ / (n - 1) - .5) * SIZE + CENTRE_Y
            final = np.maximum(final, np.minimum(.5 * d, 22) * sm_((70 - Y_) / 50))         # the right bank climbs straight from the water, terrace on terrace
            final = np.maximum(final, 0) + .75 * np.clip(d / 3, 0, 1)
            final = np.where(m, -.5 * np.clip(-d / 3, 0, 1), final)
            np.save(os.path.join(out, f"land_{LAND_V}_{n}.npy"), final.astype(np.float32))
        # the near hill, finer: the same land, with more small gullies cut into it
        cx, cy, side = PATCH
        n = final.shape[0]
        cell = SIZE / (n - 1)
        i0, j0 = int((cx - side / 2 + SIZE / 2) / cell), int((cy - side / 2 - CENTRE_Y + SIZE / 2) / cell)
        k = int(side / cell)
        crop = final[j0:j0 + k, i0:i0 + k]
        m = 2048
        fine = zoom(crop, m / k, order=3)[:m, :m].astype(np.float32)
        ring = np.zeros((m, m), bool)
        ring[:8, :] = ring[-8:, :] = ring[:, :8] = ring[:, -8:] = True
        fine = fine + .05 * spectral(m, 2.0, SEED + 9)
        fine[ring] = zoom(crop, m / k, order=3)[:m, :m].astype(np.float32)[ring]
        fine = evolve(fine, np.zeros_like(fine), ring, side * 1.0 / m, 14, .3, K=K, diffuse=.02)
        np.save(os.path.join(out, f"patch_{LAND_V}.npy"), fine)
        print("patch", fine.min(), fine.max(), "from", (i0, j0, k), flush=True)

    def pane_texture(k=1):
        """The paper of the windows we can read, seen with the lamp behind it. Nothing is written on it here: what is on a window is a chat, and ink_nine.py lays that over the render, in its real colours.
        k=4 is the first of them alone at four times the size: it fills the frame at the start of the pull-back."""
        import numpy as np
        from PIL import Image
        from scipy.ndimage import gaussian_filter, map_coordinates

        from ink_grade import fft_noise
        W_, H_ = 640 * k, 512 * k
        out = []
        for i in range(len(HERO_WINDOWS) if k == 1 else 1):
            im = Image.new("L", (W_, H_), 255)
            a = np.asarray(im, np.float32) / 255
            yy, xx = np.mgrid[:H_, :W_].astype(np.float32)
            a = map_coordinates(a, [yy + 1.3 * k * fft_noise(H_, W_, 2.3, SEED + 80 + i), xx + 1.3 * k * fft_noise(H_, W_, 2.3, SEED + 90 + i)], order=1, mode="nearest")
            a = gaussian_filter(a, .9 * k ** .6)
            lamp = 1 - .24 * (((xx / W_ - .5) * 2) ** 2 + ((yy / H_ - .5) * 2) ** 2) / 2
            a = a * lamp * (1 + .018 * fft_noise(H_, W_, 1.4, SEED + 100 + i))
            out.append(np.flipud(np.clip(a, 0, 1)))
        cache = os.path.join(HERE, "..", "..", "ink", "film_out", "3d", "qian", "cache")
        np.save(os.path.join(cache, "pane_tex.npy"), np.array(out, np.float32)) if k == 1 else np.save(os.path.join(cache, "pane_first.npy"), out[0].astype(np.float32))

    def chat_texture():
        """What is on the window, as lit paper with ink on it: the same two lines that were on the pool's surface in chapter 2... and a place to type. No colour; the bubbles are a wash and a line."""
        import numpy as np
        from PIL import Image, ImageDraw
        from scipy.ndimage import gaussian_filter, map_coordinates

        from ink_film import font
        from ink_grade import fft_noise
        W_, H_ = 2400, 1560
        im = Image.new("L", (W_, H_), 255)
        d = ImageDraw.Draw(im)
        ink, line = 22, 120
        d.line([(0, 168), (W_, 168)], fill=line, width=6)
        for i in range(3):
            d.ellipse([(110 + i * 76 - 18, 84 - 18), (110 + i * 76 + 18, 84 + 18)], outline=line, width=6)
        d.rounded_rectangle((1560, 300, 2250, 520), radius=70, fill=196)
        d.text((1905, 410), "你好", font=font("kaib", 124), fill=ink, anchor="mm")
        s = "你好！有什么可以帮你的吗？"
        d.rounded_rectangle((150, 650, 1960, 870), radius=70, outline=line, width=6)
        d.text((1055, 760), s, font=font("kaib", 124), fill=ink, anchor="mm")
        d.rounded_rectangle((150, 1290, 2250, 1460), radius=80, outline=line, width=6)
        d.line([(240, 1325), (240, 1425)], fill=ink, width=8)
        a = np.asarray(im, np.float32) / 255
        # it has lain in the water a long time: the edges of the ink are soft, the paper a little uneven, brighter in the middle
        yy, xx = np.mgrid[:H_, :W_].astype(np.float32)
        a = map_coordinates(a, [yy + 1.2 * fft_noise(H_, W_, 2.4, SEED + 41), xx + 1.2 * fft_noise(H_, W_, 2.4, SEED + 42)], order=1, mode="nearest")
        a = gaussian_filter(a, 1.2)
        v = 1 - .07 * (((xx / W_ - .5) * 2) ** 2 + ((yy / H_ - .5) * 2) ** 2)
        a = a * v * (1 + .012 * fft_noise(H_, W_, 1.8, SEED + 43))
        out = os.path.join(HERE, "..", "..", "ink", "film_out", "3d", "qian", "cache", "chat_tex.npy")
        np.save(out, np.flipud(np.clip(a, 0, 1)).astype(np.float32))

    if __name__ == "__main__":
        if sys.argv[1] == "chattex":
            chat_texture()
            pane_texture()
            pane_texture(4)
            sys.exit()
        if sys.argv[1] == "land":
            make_land([(768, 320, 2.0, 0, .1), (1536, 40, .6, .2, .06), (3072, 10, .3, .08, .03)][:int(sys.argv[2]) if len(sys.argv) > 2 else 3])
            sys.exit()
        if sys.argv[1] == "patch":     # python ink3d_qian.py patch SHOT BASE_TAG NEW_TAG OUT_TAG x0,y0,x1,y1
            shot, a_, b_, c_, box = sys.argv[2:7]
            d_ = os.path.join(HERE, "..", "..", "ink", "film_out", "3d", "qian", "dev", shot)
            patch_passes(d_ + a_, d_ + b_, [int(v) for v in box.split(",")], d_ + c_)
            sys.exit()
        if sys.argv[1] == "path":       # python ink3d_qian.py path A B: frames A to B of the pull-back. The grade goes from lattice's to qian's with the distance, so the valley's mist comes in as it comes into view; the sky and moon are qian's throughout
            import numpy as np

            from ink3d_nightgrade import night
            base = os.path.join(HERE, "..", "..", "ink", "film_out", "3d", "qian")
            os.makedirs(os.path.join(base, "path"), exist_ok=True)
            T_ = np.load(os.path.join(base, "windows_qian.npz"))
            a_, b_ = GRADES["lattice"], GRADES["qian"]
            for n in range(int(sys.argv[2]), int(sys.argv[3]) + 1):
                near = pull_near(n)
                u = min(max(np.log(near / 13.2) / np.log(PULL_B / 13.2), 0), 1)
                kw = {k: a_[k] + (b_[k] - a_[k]) * u for k in ("rho", "mist_d", "lift", "contrast", "mist_floor", "lit_mist")}
                kw.update(lights=tuple(x + (y - x) * u for x, y in zip(a_["lights"], b_["lights"])), moon=b_["moon"], horizon=HORIZON, reflect=b_["reflect"],
                          mist_h=b_["mist_h"], aerial=b_["aerial"], sky_top=b_["sky_top"], sky_hor=b_["sky_hor"])
                s_ = PULL_B / near
                table = os.path.join(base, "dev", f"path_{n:03d}_first.npz")
                np.savez(table, sx=T_["sx"][:1], sy=T_["sy"][:1], rx=T_["sx"][:1] + (T_["rx"][:1] - T_["sx"][:1]) * s_, ry=T_["sy"][:1] + (T_["ry"][:1] - T_["sy"][:1]) * s_,
                         w_px=T_["w_px"][:1] * s_, h_px=T_["h_px"][:1] * s_, bright=T_["bright"][:1], d=np.array([near]), n_hero=1)
                night(os.path.join(base, "dev", f"path_{n:03d}"), out=os.path.join(base, "path", f"path_{n:03d}_ink.png"), windows=table, **kw)
            sys.exit()
        if sys.argv[1] == "sheet":
            from PIL import Image
            OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "3d", "qian")
            ims = [Image.open(os.path.join(OUT, f"{s_}_ink.png")).convert("RGB").resize((960, 540), Image.LANCZOS) for s_ in ("chat", "lattice", "qian", "dark")]
            sheet = Image.new("RGB", (1920, 1080))
            for i, im in enumerate(ims):
                sheet.paste(im, ((i % 2) * 960, (i // 2) * 540))
            sheet.save(os.path.join(OUT, "sheet.png"))
            sys.exit()
        from ink3d_nightgrade import night
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--shot", default="qian")
        ap.add_argument("--tag", default="")
        ap.add_argument("--set", default="")     # e.g. --set=rho=.03,mist_d=1.2
        ap.add_argument("--out", default=None)
        o = ap.parse_args()
        kw = dict(GRADES[o.shot])
        for kv in filter(None, o.set.split(",")):
            k, v = kv.split("=")
            kw[k] = float(v)
        base = os.path.join(HERE, "..", "..", "ink", "film_out", "3d", "qian")
        kw["windows"] = os.path.join(base, f"windows_{o.shot}.npz") if o.shot in ("qian", "dark", "going") else None
        print(night(os.path.join(base, "dev", o.shot + o.tag), out=o.out or os.path.join(base, o.shot + "_ink.png"), **kw))
