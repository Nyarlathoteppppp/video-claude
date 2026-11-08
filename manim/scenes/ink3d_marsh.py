"""The marsh at night: what the ghosts left. No ghost and no crying anywhere in it: reeds laid over, lights over the water, rings where nothing fell.
ink3d_myth.py runs these (--shot=marsh|lie|bare_dark|bare_lit|fall|fall_plate|fall_flat|lone); its docstring says how each one moves and hands on.
"""
import math
import os

import bpy
import numpy as np

from ink3d import OUT3D, camera, grey_material, image_from, lerp, mesh_grid, painted_material, sm, spectral, world_light
from ink3d_myth import cloud_volume, emission, far_mountains, haze_box, image_emission, mist_sheet, moon, opt, shine, shoot

RAW = os.path.join(OUT3D, "myth", "raw")
WATER_Z = 0.0


def raw_mesh(name, verts, tris, smooth=True):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", np.asarray(verts, np.float32).ravel())
    me.loops.add(tris.size)
    me.loops.foreach_set("vertex_index", tris.astype(np.int32).ravel())
    me.polygons.add(len(tris))
    me.polygons.foreach_set("loop_start", (np.arange(len(tris)) * 3).astype(np.int32))
    me.update(calc_edges=True)
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(len(tris), bool))
        me.update()
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    return o


class Ribbons:
    """Blades, stalks and plume hairs, each a strip turned to face the lens, all merged into one mesh."""

    def __init__(self, cam):
        self.cam, self.V, self.T, self.n = np.array(cam, float), [], [], 0

    def add(self, pts, wid):
        """pts: (k, s, 3), wid: (k, s) half-widths."""
        k, s = pts.shape[:2]
        tan = np.gradient(pts, axis=1)
        side = np.cross(tan, self.cam - pts)
        side /= np.linalg.norm(side, axis=2, keepdims=True) + 1e-9
        a, b = pts + side * wid[..., None], pts - side * wid[..., None]
        v = np.empty((k, s * 2, 3), np.float32)
        v[:, ::2], v[:, 1::2] = a, b
        idx = np.arange(s - 1)
        quad = np.stack([idx * 2, idx * 2 + 1, idx * 2 + 3, idx * 2, idx * 2 + 3, idx * 2 + 2], 1).reshape(-1, 3)
        self.T.append((quad[None] + (self.n + np.arange(k) * s * 2)[:, None, None]).reshape(-1, 3))
        self.V.append(v.reshape(-1, 3))
        self.n += k * s * 2

    def mesh(self, name, material):
        o = raw_mesh(name, np.concatenate(self.V), np.concatenate(self.T))
        o.data.materials.append(material)
        return o


def bend(base, height, theta0, power, heading, s):
    """A line growing from base, leaning more the higher it gets. theta0: its lean at the tip, radians from upright."""
    th = theta0[:, None] * s[None] ** power
    ds = np.gradient(s)[None]
    horiz = np.cumsum(np.sin(th) * ds, 1) * height[:, None]
    up = np.cumsum(np.cos(th) * ds, 1) * height[:, None]
    return np.stack([base[:, 0, None] + horiz * np.cos(heading)[:, None], base[:, 1, None] + horiz * np.sin(heading)[:, None], base[:, 2, None] + up], 2)


def reeds(cam, base, height, lean, seed, leaves=4, hairs=8, width=1.0, heading=0.0):
    """Common reed: a thin stalk, a few long blades let go downwind, a feathered head. Returns (dark strips, pale strips)."""
    r = np.random.default_rng(seed)
    k = len(base)
    dark, pale = Ribbons(cam), Ribbons(cam)
    s = np.linspace(0, 1, 9)
    head = heading + r.normal(0, .14, k)
    theta = lean * r.uniform(.8, 1.2, k)
    stalk = bend(base, height, theta, 1.6, head, s)
    dark.add(stalk, width * (.016 + .004 * height)[:, None] * (1 - .72 * s[None]))
    t = np.linspace(0, 1, 7)
    for _ in range(leaves):
        at = r.integers(1, 7, k)
        root = stalk[np.arange(k), at]
        L = height * r.uniform(.28, .46, k)
        th0, th1 = r.uniform(.2, .6, k), r.uniform(1.5, 2.4, k)
        hd = head + r.normal(0, .5, k)
        th = th0[:, None] + (th1 - th0)[:, None] * t[None] ** 1.3
        ds = np.gradient(t)[None]
        h_ = np.cumsum(np.sin(th) * ds, 1) * L[:, None]
        u_ = np.cumsum(np.cos(th) * ds, 1) * L[:, None]
        leaf = np.stack([root[:, 0, None] + h_ * np.cos(hd)[:, None], root[:, 1, None] + h_ * np.sin(hd)[:, None], root[:, 2, None] + u_], 2)
        dark.add(leaf, width * (.026 + .007 * height)[:, None] * np.sin(np.pi * t[None] ** .55) ** .8)
    top = stalk[:, -1]
    for _ in range(hairs):
        L = height * r.uniform(.11, .2, k)
        th0, th1 = theta + r.normal(-.1, .28, k), theta + r.uniform(.5, 1.1, k)
        hd = head + r.normal(0, .3, k)
        th = th0[:, None] + (th1 - th0)[:, None] * t[None] ** 1.2
        ds = np.gradient(t)[None]
        h_ = np.cumsum(np.sin(th) * ds, 1) * L[:, None]
        u_ = np.cumsum(np.cos(th) * ds, 1) * L[:, None]
        off = r.normal(0, .015, (k, 1)) * height[:, None]
        hair = np.stack([top[:, 0, None] + h_ * np.cos(hd)[:, None], top[:, 1, None] + h_ * np.sin(hd)[:, None] + off, top[:, 2, None] + u_ - .1 * L[:, None] * r.random((k, 1))], 2)
        pale.add(hair, width * (.012 + .003 * height)[:, None] * np.sin(np.pi * (.08 + .92 * t[None]) ** .7) ** .6)
    return dark, pale


def merge(name, parts, material):
    all_ = Ribbons((0, 0, 0))
    for p in parts:
        n0 = all_.n
        for v, t in zip(p.V, p.T):
            all_.V.append(v)
            all_.T.append(t + n0)
        all_.n += p.n
    return all_.mesh(name, material)


def mirror_water(value=.86, rough=.015):
    """Still water as an ink painter paints it: the sky over again, a shade down."""
    m = bpy.data.materials.new("marsh_water")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    g, o = nt.nodes.new("ShaderNodeBsdfGlossy"), nt.nodes.new("ShaderNodeOutputMaterial")
    g.inputs["Color"].default_value = (value, value, value, 1)
    g.inputs["Roughness"].default_value = rough
    nt.links.new(g.outputs[0], o.inputs["Surface"])
    return m


def rings(X, Y, list_):
    """Rings spreading on still water: each is a few crests leaving (x, y)."""
    out = np.zeros_like(X)
    for x0, y0, R, lam, amp in list_:
        r_ = np.hypot(X - x0, Y - y0)
        env = np.exp(-((r_ - R * .55) / (R * .5)) ** 2) * sm((R - r_) / (R * .12))
        out += amp * np.sin(2 * math.pi * (r_ - R) / lam) * env
    return out


def water_surface(cam, ring_list, value=.8, y_max=1700.0, rows=1500, cols=1800, d_min=.8):
    """A sheet whose cells are about even on screen: fine where rings must show, coarse at the horizon."""
    cx, cy = cam[0], cam[1]
    d = np.exp(np.linspace(math.log(d_min), math.log(y_max), rows)).astype(np.float32)
    u = np.linspace(-1.15, 1.15, cols, dtype=np.float32)
    Y = cy + d[:, None] + 0 * u[None]
    X = cx + d[:, None] * u[None]
    Z = WATER_Z + rings(X, Y, ring_list) if ring_list else np.full_like(X, WATER_Z)
    g = mesh_grid("water", X, Y, Z)
    g.data.materials.append(mirror_water(value))
    return g


def flame_atlas(n=6, cell=420, seed=7):
    """Flames as a brush leaves them: bare paper in the ink around it, no two alike. Some slim and tall, some squat, each leaning and wavering its own way, the edge dissolving
    into a breath of light. Teardrops still, for they will fall like tears."""
    r = np.random.default_rng(seed)
    size = n * cell
    alpha, value = np.zeros((size, size), np.float32), np.zeros((size, size), np.float32)
    y, x = np.mgrid[:cell, :cell].astype(np.float32)
    u, v = x / (cell - 1) * 2 - 1, y / (cell - 1) * 2 - 1
    for i in range(n):
        for j in range(n):
            fat, foot, tip, neck = r.uniform(.62, 1.3), r.uniform(-.72, -.5), r.uniform(.45, .88), r.uniform(1.0, 1.8)
            lean, waver, phase = r.normal(0, .16), r.uniform(.02, .12), r.uniform(0, 6.3)
            t = np.clip((v - foot) / (tip - foot), 0, 1)
            centre = lean * t ** 1.8 + waver * np.sin(t * 6.5 + phase) * t * t
            at = .46 / (.46 + neck)
            w = .40 * fat * t ** .46 * (1 - t) ** neck / (at ** .46 * (1 - at) ** neck)
            # the distance outside the flame's outline, in card units
            d = np.abs(u - centre) - w
            d = np.where(v < foot, np.hypot(u - centre, v - foot), d)
            d = np.where(v > tip, np.hypot(u - centre, v - tip), d)
            d = d + .02 * lerp(spectral(1024, 2.4, seed + i * 9 + j), x * 2.4, y * 2.4)
            core = sm((.03 - d) / .16)
            halo = .42 * np.exp(-np.maximum(d, 0) / .2) * sm((1 - np.hypot(u, v)) / .3)
            alpha[j * cell:(j + 1) * cell, i * cell:(i + 1) * cell] = np.maximum(core, halo)
            value[j * cell:(j + 1) * cell, i * cell:(i + 1) * cell] = .92 + .08 * core
    return alpha, value


_atlas = None


def flame_material():
    global _atlas
    if _atlas is None:
        al, va = flame_atlas()
        m = bpy.data.materials.new("flame")
        m.use_nodes = True
        m.blend_method = "BLEND"
        nt = m.node_tree
        nt.nodes.clear()
        ta, tv, tr, e, mx, out = (nt.nodes.new(k) for k in ("ShaderNodeTexImage", "ShaderNodeTexImage", "ShaderNodeBsdfTransparent", "ShaderNodeEmission", "ShaderNodeMixShader", "ShaderNodeOutputMaterial"))
        ta.image, tv.image = image_from(al, "flame_a"), image_from(va, "flame_v")
        nt.links.new(tv.outputs[0], e.inputs[0])
        nt.links.new(ta.outputs[0], mx.inputs[0])
        nt.links.new(tr.outputs[0], mx.inputs[1])
        nt.links.new(e.outputs[0], mx.inputs[2])
        nt.links.new(mx.outputs[0], out.inputs["Surface"])
        _atlas = m
    return _atlas


def fire_cards(name, cam, pos, size, seed=3, stretch=None, n=6):
    """A card for each light, turned to the lens, each a little taller or fatter than the next and leaning its own way. size: about half its height. stretch: how much taller than wide (for a light that is falling)."""
    r = np.random.default_rng(seed)
    pos = np.asarray(pos, float)
    size = np.asarray(size, float)
    k = len(pos)
    stretch = np.ones(k) if stretch is None else np.asarray(stretch, float)
    size = size * np.exp(r.normal(0, .22, k))
    slim = r.uniform(.7, 1.3, k)
    tilt = np.radians(r.normal(-4, 9, k))
    f = pos - np.array(cam, float)
    f[:, 2] = 0
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    right = np.c_[f[:, 1], -f[:, 0], np.zeros(k)]
    up = np.tile(np.array((0, 0, 1.0)), (k, 1))
    ca, sa = np.cos(tilt)[:, None], np.sin(tilt)[:, None]
    R_, U_ = right * ca + up * sa, up * ca - right * sa
    wid, hei = (size / stretch * slim)[:, None], size[:, None]
    verts = np.stack([pos - R_ * wid - U_ * hei, pos + R_ * wid - U_ * hei, pos + R_ * wid + U_ * hei, pos - R_ * wid + U_ * hei], 1).reshape(-1, 3)
    idx = np.arange(k)[:, None, None] * 4
    tris = (np.array([[0, 1, 2], [0, 2, 3]])[None] + idx).reshape(-1, 3)
    o = raw_mesh(name, verts, tris, smooth=False)
    cell = r.integers(0, n, (k, 2)).astype(np.float32)
    uv = np.stack([cell, cell + (1, 0), cell + (1, 1), cell + (0, 1)], 1) / n
    # loops run through tris: 0 1 2 0 2 3
    uv_loops = uv[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 2)
    lay = o.data.uv_layers.new(name="UVMap")
    lay.data.foreach_set("uv", uv_loops.astype(np.float32).ravel())
    o.data.materials.append(flame_material())
    o.visible_shadow = False
    return o


def ink_drop(name, base, height, radius):
    """A drop falling: round at the bottom and drawn up to a thin tail, black and wet, shining only where it catches the sky. radius: at its widest."""
    n, m = 160, 96
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    th = np.linspace(0, 2 * math.pi, m, dtype=np.float32)[None, :]
    prof = radius * t ** .5 * (1 - t) ** 1.6 / .316
    X = base[0] + prof * np.cos(th)
    Y = base[1] + prof * np.sin(th)
    Z = base[2] + height * t + 0 * th
    o = mesh_grid(name, X, Y, Z)
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    b = mt.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (.002, .002, .002, 1)
    b.inputs["Roughness"].default_value = .06
    b.inputs["Specular IOR Level"].default_value = .7
    o.data.materials.append(mt)
    return o


def broken_stones(spec, seed, z0=0.0):
    """Stones that stood up once, with nothing written on them: slabs with a broken top. spec: (x, y, width, height, thick, lean, yaw)."""
    r = np.random.default_rng(seed)
    objs = []
    mat = grey_material("old_stone", .5, .9)
    for i, (x, y, w, H, th, lean, yaw) in enumerate(spec):
        m_, n_ = 40, 48
        gy, gx = np.mgrid[:n_, :m_].astype(np.float32)
        fx, fz = gx / (m_ - 1) - .5, gy / (n_ - 1)
        brk = np.interp(np.linspace(0, 1, m_), np.linspace(0, 1, 7), r.uniform(.45, 1.0, 7))
        brk = brk + np.interp(np.linspace(0, 1, m_), np.linspace(0, 1, 25), r.normal(0, .03, 25))
        Z = fz * H * brk[None, :]
        lx = fx * w * (1 - .12 * fz) + .04 * w * lerp(spectral(1024, 2.9, seed + i + 40), gx * 10, gy * 6)
        rough = .05 * lerp(spectral(1024, 2.6, seed + i), gx * 16, gy * 16) * w
        ca, sa = math.cos(yaw), math.sin(yaw)
        for side, sgn in (("f", 1), ("b", -1)):
            off = sgn * th / 2 + sgn * rough
            X = x + lx * ca + off * sa + Z * math.sin(lean)
            Y = y + lx * sa - off * ca
            o = mesh_grid(f"stone{i}{side}", X.astype(np.float32), Y.astype(np.float32), (z0 + Z * math.cos(lean) - .3).astype(np.float32))
            o.data.materials.append(mat)
            objs.append(o)
        # the top and the two sides as one strip
        ring_l, ring_z = np.r_[lx[:, 0], lx[-1, :], lx[::-1, -1]], np.r_[Z[:, 0], Z[-1, :], Z[::-1, -1]]
        sx, sy, sz = [], [], []
        for off in (th / 2, -th / 2):
            sx.append(x + ring_l * ca + off * sa + ring_z * math.sin(lean))
            sy.append(y + ring_l * sa - off * ca)
            sz.append(z0 + ring_z * math.cos(lean) - .3)
        e = mesh_grid(f"stone{i}e", np.array(sx, np.float32), np.array(sy, np.float32), np.array(sz, np.float32))
        e.data.materials.append(mat)
        objs.append(e)
    return objs


def channel(Y):
    """The open water the lens looks down."""
    return 9 * np.sin(Y / 60) * sm(Y / 60), 12 + .22 * np.maximum(Y, 0)


MOUNDS = [(-58, 118, 10, 19), (-26, 150, 6, 14), (36, 136, 13, 23), (78, 176, 9, 18), (-104, 196, 15, 29), (8, 246, 17, 36), (120, 262, 16, 33), (-170, 290, 19, 40), (-48, 340, 16, 38), (210, 340, 19, 45), (96, 410, 16, 42), (-260, 170, 11, 25), (250, 160, 9, 22)]


def marsh_land(seed=401, mounds=MOUNDS, size=1000.0, N=1500):
    """Mud banks hardly out of the water, and old mounds. Returns a function telling where reeds can stand."""
    y, x = np.mgrid[:N, :N].astype(np.float32)
    X, Y = (x / (N - 1) - .5) * size, y / (N - 1) * size - 20
    sx, sy = x * 1024 / N, y * 1024 / N
    nz, nz2 = lerp(spectral(1024, 3.2, seed), sx, sy), lerp(spectral(1024, 2.3, seed + 1), sx, sy)
    cc, cw = channel(Y)
    bank = .34 * nz + .10 * nz2 + .38 * sm((np.abs(X - cc) - cw) / 14) - .24 - .7 * sm((cw - np.abs(X - cc)) / 5)
    m = np.zeros_like(X)
    for mx, my, top, rad in mounds:
        d = np.hypot(X - mx + 2.4 * nz2, (Y - my) * 1.08 + 2.4 * nz) / rad
        m = np.maximum(m, top * np.clip(1 - d ** 2.3, 0, 1) ** .75)
    h = np.maximum(bank, -.5) + m + .05 * nz2 * sm(m / 3)
    g = mesh_grid("banks", X, Y, h)
    alb = np.clip(.52 + .10 * nz + .06 * nz2 - .10 * sm(m / 10) * nz, .1, .9)
    g.data.materials.append(painted_material("banks", alb.astype(np.float32)))

    def stands(px, py):
        c, w = channel(py)
        ix = np.clip(((px + size / 2) / size * (N - 1)).astype(int), 0, N - 1)
        iy = np.clip(((py + 20) / size * (N - 1)).astype(int), 0, N - 1)
        return (h[iy, ix] > -.02) & (h[iy, ix] < 1.5) & (np.abs(px - c) > w * .9), h[iy, ix]

    def ground(px, py):
        ix = np.clip(((px + size / 2) / size * (N - 1)).astype(int), 0, N - 1)
        iy = np.clip(((py + 20) / size * (N - 1)).astype(int), 0, N - 1)
        return np.maximum(h[iy, ix], 0)
    stands.ground = ground
    return stands


def reed_beds(cam, stands, seed, lean, far_y=420.0, n_mid=26000, near=(), heading=0.0):
    """Reeds in beds from the lens to the far banks, and the hand-placed ones that frame the shot."""
    r = np.random.default_rng(seed)
    k = n_mid * 3
    # more of them close to the lens
    py = 6 + (r.random(k) ** 1.9) * far_y
    px = cam[0] + r.uniform(-1, 1, k) * (10 + py * .9)
    ok, hh = stands(px, py)
    px, py, hh = px[ok][:n_mid], py[ok][:n_mid], hh[ok][:n_mid]
    base = np.c_[px, py, np.maximum(hh, 0) - .05]
    height = r.uniform(2.1, 3.6, len(px))
    near_part = py < 90
    parts = []
    for mask, lv, hr, sd in ((near_part, 3, 5, seed + 1), (~near_part, 1, 2, seed + 2)):
        if mask.sum():
            parts.append(reeds(cam, base[mask], height[mask], lean, sd, leaves=lv, hairs=hr, width=1.0 if sd == seed + 1 else 1.7, heading=heading))
    for nb, nh, sd, ct in near:
        pts = np.asarray(nb, float)
        kk = int(ct)
        # a clump: stems close together round each point
        c = pts[r.integers(0, len(pts), kk)] + np.c_[r.normal(0, .36, kk), r.normal(0, .36, kk), np.zeros(kk)]
        parts.append(reeds(cam, c, r.uniform(nh * .72, nh * 1.15, kk), lean, sd, leaves=2, hairs=30, width=.62, heading=heading))
    return merge("reeds", [p[0] for p in parts], grey_material("reed", .05, .7)), merge("plumes", [p[1] for p in parts], grey_material("plume", .6, .8))


def sky(cam, moon_at, radius=150.0, deck=True, seed=411, base=.46, rho=.012):
    """Moonlit cloud: a backdrop that grows pale toward the moon, the moon itself in a gap, and a real cloud deck over it all, lit from behind. Returns what must stay out of the depth pass."""
    cam, moon_at = np.array(cam, float), np.array(moon_at, float)
    n, m = 512, 1024
    gy, gx = np.mgrid[:n, :m].astype(np.float32)
    W, H, Y0 = 12000.0, 4200.0, 3600.0
    # where on the card the moon stands
    f = moon_at - cam
    t = (Y0 - cam[1]) / f[1]
    mu, mv = (cam[0] + f[0] * t) / W + .5, (cam[2] + f[2] * t + H * .36) / H
    wx, wy = (gx / (m - 1) - mu) * W, (gy / (n - 1) - mv) * H
    r_ = np.hypot(wx, wy * 1.2)
    wash = lerp(spectral(1024, 3.4, seed), gx, gy * 2) * .05
    v = base + .50 * np.exp(-r_ / 1100.0) + .10 * np.exp(-r_ / 3400.0) + wash
    v = v + .10 * sm((H * .4 - (gy / (n - 1)) * H) / 800.0)           # toward the horizon the mist lets more through
    card = mesh_grid("skycard", np.array([[-W / 2 + cam[0], W / 2 + cam[0]]] * 2, np.float32), np.full((2, 2), Y0, np.float32), np.array([[-H * .36 + cam[2]] * 2, [H * .64 + cam[2]] * 2], np.float32))
    card.data.materials.append(image_emission("skycard", np.clip(v, 0, 1).astype(np.float32)))
    card.visible_shadow = False
    extras = [card] + moon(tuple(moon_at), radius, tuple(cam))
    # at most one thin streak of cloud goes across it
    extras.append(mist_sheet("wisp", tuple(moon_at + (cam - moon_at) * .02 + np.array((radius * .4, 0, -radius * .3))), radius * 7, radius * .62, cam, value=.62, density=.62, seed=seed + 9, grain=3.0, top=.4))
    if deck:
        bpy.context.scene.cycles.volume_bounces = 3
        # the gap round the moon, along the line of sight through the deck
        bx0, bx1, by0, by1 = -3400.0, 3400.0, 0.0, 3400.0
        n2, m2 = 1024, 2048
        gy2, gx2 = np.mgrid[:n2, :m2].astype(np.float32)
        wx2 = bx0 + gx2 / (m2 - 1) * (bx1 - bx0)
        wy2 = by0 + gy2 / (n2 - 1) * (by1 - by0)
        along = (wx2 - cam[0]) * f[0] / np.hypot(f[0], f[1]) + (wy2 - cam[1]) * f[1] / np.hypot(f[0], f[1])
        across = -(wx2 - cam[0]) * f[1] / np.hypot(f[0], f[1]) + (wy2 - cam[1]) * f[0] / np.hypot(f[0], f[1])
        warp = lerp(spectral(1024, 3.0, seed + 2), gx2 / 2, gy2) * 90
        hole = sm((np.abs(across + warp) - radius * 1.5) / (radius * 1.4)) + sm((1700 - along) / 300)
        mask = np.clip(hole, 0, 1).astype(np.float32)
        deck_o = cloud_volume("deck", (bx0, bx1), (by0, by1), (470, 800), top=720.0, thick=46.0, density=opt("deckrho", rho), scale=.0014, seed=6.0, bottom=560.0, fray=3.0,
                              cut=(image_from(mask, "gap"), -bx0 / (bx1 - bx0), -by0 / (by1 - by0), 1 / (bx1 - bx0), 1 / (by1 - by0)))
        extras.append(deck_o)
    return extras


def far_ranges():
    """Mountains behind mountains, each a shade closer to the mist. The grade's fog can only see so far, so distance is painted too."""
    for k, (val, peaks, seed, yc) in enumerate(((.72, [(-500, 1240, 200, 230), (-120, 1320, 250, 270), (300, 1260, 215, 240), (680, 1300, 235, 270), (-860, 1320, 250, 280)], 421, 1200),
                                                (.58, [(-340, 920, 118, 160), (40, 980, 150, 180), (420, 930, 125, 170), (-720, 960, 135, 190), (760, 960, 120, 180)], 431, 900),
                                                (.45, [(-470, 660, 64, 110), (-180, 700, 80, 120), (170, 680, 70, 110), (480, 700, 78, 120), (-720, 640, 56, 100)], 441, 650))):
        X, Y2, H = far_mountains(peaks, 2400, 900, seed, y_center=yc, strength=.3)
        m = mesh_grid(f"range{k}", X, Y2, H - 3)            # each stands on a square of land reaching under the lens: its flat must stay below the water
        m.data.materials.append(emission(f"rangew{k}", val))


MOON_DIR = (-.27, .92, .205)     # from the lens toward the moon


def moon_at(cam, d=2600.0, direction=MOON_DIR):
    return tuple(np.array(cam, float) + np.array(direction) / np.linalg.norm(direction) * d)


def moonlight(world=.62, energy=1.8, direction=MOON_DIR):
    world_light(opt("world", world))
    v = np.array(direction) / np.linalg.norm(direction)
    shine((-v[0], -v[1], -abs(v[2]) - .22), energy=opt("sun", energy), angle=2.0)
    # the sky card and the deck shut out the world's light, and the moon is behind everything: a little from the lens's side, so that stone and mound have a face
    f = bpy.data.lights.new("fill", "SUN")
    f.energy = opt("fill", 1.5)
    f.use_shadow = False
    if hasattr(f.cycles, "cast_shadow"):
        f.cycles.cast_shadow = False
    fo = bpy.data.objects.new("fill", f)
    fo.rotation_euler = (math.radians(70), 0, 0)
    bpy.context.collection.objects.link(fo)


def scatter_fires(seed, count, near_y=26.0, far_y=430.0):
    """Lights hanging over the water and among the reeds, thicker near the channel, thinning away to pinpricks."""
    r = np.random.default_rng(seed)
    py = near_y + r.random(count) ** 1.5 * (far_y - near_y)
    c, w = channel(py)
    px = c + r.normal(0, 1, count) * (w * .35 + py * .22)
    pz = r.uniform(1.0, 3.2, count) + .012 * py * r.random(count)
    size = r.uniform(.7, 1.05, count)
    return np.c_[px, py, pz], size


# ---- the shots ---------------------------------------------------------------------------------------------------------------------
CAM = (0.0, 0.0, 4.5)
TARGET = (-3.0, 100.0, .9)
LENS = 28.0
NEAR_REEDS = [([(-7.2, 8.6, 0), (-6.4, 9.4, 0), (-8.0, 9.9, 0), (-5.6, 8.9, 0)], 4.6, 451, 18),
              ([(-8.5, 15, 0), (-7.4, 16.5, 0), (-9.8, 17, 0)], 5.0, 452, 16),
              ([(3.2, 10.2, 0), (3.9, 11.0, 0), (2.7, 10.8, 0)], 3.0, 453, 14)]
STONES = [(-13, 40, 3.4, 3.8, .8, .12, .3), (-17, 46, 2.6, 2.4, .7, -.16, -.2), (16, 54, 3.6, 4.2, .8, -.10, .15), (6.6, 72, 3.0, 2.8, .7, .16, -.1), (-30, 64, 3.2, 3.2, .7, .06, .2), (32, 86, 3.4, 3.8, .8, .08, -.2)]
RINGS = [(-5.0, 17.2, 5.2, .82, .009), (3.0, 25, 4.2, .9, .012), (-9.5, 33, 5.0, 1.1, .014), (8.5, 44, 5.5, 1.2, .016)]         # the widest is on the moon's own reflection


def marsh_common(cam=CAM, target=TARGET, lens=LENS, ring_list=RINGS, lean=1.05, near=NEAR_REEDS, stones=STONES, mid=26000, deck=True, mists=True, moon_dir=MOON_DIR, water=None):
    co = camera(cam, target, lens=lens, far=2800)
    stands = marsh_land()
    water_surface(cam, ring_list, **(water or {}))
    reed_beds(cam, stands, 461, lean, near=near, n_mid=mid)
    broken_stones(stones, 471)
    far_ranges()
    moonlight(direction=moon_dir)
    hide = sky(cam, moon_at(cam, direction=moon_dir), deck=deck)
    if mists:
        hide += [mist_sheet("bank0", (-30, 112, 7), 300, 16, cam, value=.82, density=.75, seed=481, top=.45),
                 mist_sheet("bank1", (40, 230, 12), 560, 30, cam, value=.8, density=.9, seed=485, top=.4),
                 mist_sheet("bank2", (-40, 420, 26), 1000, 60, cam, value=.76, density=1.0, seed=489, top=.35)]
    return co, hide


def shot_marsh(t=0.0):
    _, hide = marsh_common()
    pos, size = scatter_fires(491, opt("fires", 720))
    close = np.array([(4.2, 18, 1.1), (-1.4, 30, 1.2), (9.0, 36, 1.4), (-9, 42, 1.3), (3, 55, 1.5), (-15, 60, 1.6), (14, 72, 1.7)])
    close_s = np.array([1.25, 1.05, 1.0, .95, .95, .95, .95])
    hide.append(fire_cards("fires", CAM, np.r_[close, pos], np.r_[close_s, size]))
    shoot(os.path.join(RAW, "marsh"), hide)


def shot_lone(t=0.0):
    # the same lens on the same place, with everything gone out but one, at the water's edge by a stone
    _, hide = marsh_common(ring_list=[(1.5, 21.6, 2.6, .9, .010)], lean=.55)
    hide.append(fire_cards("fires", CAM, [(1.5, 22, 1.1)], [1.2]))
    shoot(os.path.join(RAW, "lone"), hide)


def shot_lie(t=0.0):
    # close on one light. Its reflection is a second card the lens cannot see: only the water can, and the water shows it in the wrong place
    cam = (0.0, 0.0, 1.5)
    target = (.2, 40.0, -1.3)
    fire = np.array((-1.5, 9.5, 1.25))
    ghost = np.array((2.05, 10.4, 1.25))
    near = [([(-6.5, 8, 0), (-5.6, 9.5, 0)], 3.6, 501, 24), ([(6.2, 7.2, 0), (6.9, 9.2, 0)], 3.2, 502, 14)]
    stones = STONES[2:]
    _, hide = marsh_common(cam=cam, target=target, lens=opt("lens", 30.0), ring_list=[(ghost[0], ghost[1] - .9, 2.4, .46, .007)], near=near, stones=stones, lean=.8, mid=16000, moon_dir=(-.62, .74, .42))
    f = fire_cards("fire", cam, [fire], [1.05])
    f.visible_glossy = False
    g = fire_cards("ghost", cam, [ghost], [1.05])
    g.visible_camera = False
    far_pos, far_size = scatter_fires(503, 110, near_y=48.0)
    hide += [f, g, fire_cards("others", cam, far_pos, far_size, seed=5)]
    shoot(os.path.join(RAW, "lie"), hide)


def bare_valley(name, lit):
    cam = (0.0, -60.0, 52.0)
    camera(cam, (6.0, 250.0, 22.0), lens=26, far=2800)
    stands = marsh_land()
    water_surface(cam, [], value=.72)
    reed_beds(cam, stands, 461, 1.0, n_mid=30000, near=())
    spec = STONES + [(-60, 130, 4, 4.5, .9, .1, .2), (70, 150, 4, 5, .9, -.1, .1), (-20, 190, 4, 4, .9, .1, .3), (40, 220, 4, 4.5, .9, .06, -.2), (-110, 240, 4.5, 5, 1, .05, 0), (20, 90, 3.5, 4, .8, -.1, .2), (-30, 105, 3.2, 3.4, .8, .12, .1), (90, 110, 3.6, 3.8, .8, .08, -.1)]
    broken_stones(spec, 471)
    far_ranges()
    away = (-.5, .8, .8)            # the moon is up behind the cloud, out of the frame
    moonlight(world=.5, energy=.8, direction=away)
    hide = sky(cam, moon_at(cam, direction=away), deck=True, rho=.004) + [haze_box("air", (-1600, 1600), (-200, 1800), (0, 222), opt("air", .0016))]
    if lit:
        r = np.random.default_rng(511)
        k = opt("fires", 26000)
        # even over the ground: far off they crowd into one carpet of light, near the lens each stands apart with the water and its reflection round it
        depth = np.sqrt(r.uniform(90 ** 2, 850 ** 2, k))
        px, py = cam[0] + r.uniform(-1, 1, k) * depth * .74, cam[1] + depth
        size = 1.25 * r.lognormal(0, .2, k)
        # close under the lens a few are big enough to see their shape
        kn = opt("close", 240)
        dn = 100 + r.random(kn) * 110
        px, py = np.r_[px, cam[0] + r.uniform(-1, 1, kn) * dn * .74], np.r_[py, cam[1] + dn]
        size = np.r_[size, r.uniform(1.7, 2.8, kn)]
        # they crowd the mounds and stand round every stone
        for mx, my, top_, rad in MOUNDS:
            km = int(rad * 16)
            th, ra = r.uniform(0, 6.3, km), rad * r.random(km) ** .6
            px, py = np.r_[px, mx + ra * np.cos(th)], np.r_[py, my + ra * np.sin(th)]
            size = np.r_[size, 1.25 * r.lognormal(0, .2, km)]
        for sx_, sy_, *_ in spec:
            th, ra = r.uniform(0, 6.3, 34), r.uniform(2.5, 9, 34)
            px, py = np.r_[px, sx_ + ra * np.cos(th)], np.r_[py, sy_ + ra * np.sin(th)]
            size = np.r_[size, r.uniform(.9, 1.5, 34)]
        g_ = stands.ground(px, py)
        # over the land they stand clear of the reeds, so that no bed is left bald
        pz = g_ + np.where(g_ > .1, r.uniform(4.0, 7.0, len(px)), r.uniform(.6, 2.6, len(px)))
        hide.append(fire_cards("fires", cam, np.c_[px, py, pz], size))
    shoot(os.path.join(RAW, name), hide)


def shot_bare_dark(t=0.0):
    bare_valley("bare_dark", False)


def shot_bare_lit(t=0.0):
    bare_valley("bare_lit", True)


def ink_sheet(name, rect, ink, ring_list, grid=500):
    """Ink lying on the water: a sheet that rides the ripples just above them, its alpha painted."""
    xs_ = np.linspace(rect[0], rect[1], grid, dtype=np.float32)
    ys_ = np.linspace(rect[2], rect[3], grid, dtype=np.float32)
    X, Y = np.meshgrid(xs_, ys_)
    sheet = mesh_grid(name, X, Y, rings(X, Y, ring_list) + .0007)
    im = bpy.data.materials.new(name)
    im.use_nodes = True
    nt = im.node_tree
    b_ = nt.nodes["Principled BSDF"]
    b_.inputs["Base Color"].default_value = (.003, .003, .003, 1)
    b_.inputs["Roughness"].default_value = .12
    ti = nt.nodes.new("ShaderNodeTexImage")
    ti.image = image_from(ink.astype(np.float32), name)
    nt.links.new(ti.outputs[0], b_.inputs["Alpha"])
    sheet.data.materials.append(im)
    sheet.visible_shadow = False
    return sheet


def thread_field(PX, PY, paths, seed):
    """Threads of ink going through the water: each path is a polyline (x, y, half-width), the water shoulders them about a little."""
    n, m = PX.shape
    gy, gx = np.mgrid[:n, :m].astype(np.float32)
    w1 = lerp(spectral(1024, 2.8, seed), gx * 1024 / m, gy * 1024 / n)
    span = (PX.max() - PX.min()) * .012
    QX = PX + span * w1
    out = np.zeros_like(PX)
    for path in paths:
        path = np.asarray(path, float)
        for (ax, ay, wa), (bx, by, wb) in zip(path[:-1], path[1:]):
            s_ = np.clip(((QX - ax) * (bx - ax) + (PY - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2 + 1e-9), 0, 1)
            dd = np.hypot(QX - ax - s_ * (bx - ax), PY - ay - s_ * (by - ay))
            w_ = wa + (wb - wa) * s_
            out = np.maximum(out, sm((w_ - dd) / (w_ * .6 + 1e-6)))
    return out


def curve(p0, p1, p2, w0, w1, k=14):
    """A quadratic curve as a polyline with a width that changes along it."""
    t_ = np.linspace(0, 1, k)[:, None]
    p0, p1, p2 = np.array(p0, float), np.array(p1, float), np.array(p2, float)
    q = (1 - t_) ** 2 * p0 + 2 * (1 - t_) * t_ * p1 + t_ ** 2 * p2
    return np.c_[q, w0 + (w1 - w0) * t_]


def shot_fall(t=0.0, plate=False, still=False):
    # down at the water's skin. One drop of the blackest ink, going in; the marsh behind it gone soft, the lights still coming down
    cam = (0.0, 0.0, .20)
    target = (.0, 40.0, .20 - 40 * .0788)
    AT = (-.06, 1.2)
    ring_list = [(AT[0], AT[1], .36, .05, .0008), (AT[0], AT[1], .78, .085, .0012), (AT[0], AT[1], 1.4, .13, .0014), (.72, 3.0, .5, .1, .0016), (-1.1, 2.6, .55, .11, .0018), (.2, 4.6, .8, .14, .002)]
    co, hide = marsh_common(cam=cam, target=target, lens=50, ring_list=[] if still else ring_list, near=(), mid=16000, moon_dir=(-.30, .92, .2), water={"value": .84, "d_min": .55, "rows": 2200, "y_max": 900.0})
    co.data.dof.use_dof = True
    co.data.dof.focus_distance = 1.2
    co.data.dof.aperture_fstop = opt("fstop", 2.0)
    if not plate:
        fall_ink(AT, ring_list)

    # the lights coming down behind it, drawn out by falling, none in front of the drop
    r2 = np.random.default_rng(543)
    k = opt("fires", 110)
    py = 12 + r2.random(k) ** 1.5 * 170
    px = r2.uniform(-1, 1, k) * py * .42
    clear = (np.abs(px / py + .05) < .11) & (py < 90)
    px = np.where(clear, px + np.sign(px / py + .05) * py * .14, px)
    pz = .4 + r2.random(k) * py * .15
    hide.append(fire_cards("drops", cam, np.c_[px, py, pz], r2.uniform(.5, .85, k), stretch=r2.uniform(1.6, 3.0, k)))
    shoot(os.path.join(RAW, "fall_flat" if still else "fall_plate" if plate else "fall"), hide)


def shot_fall_plate(t=0.0):
    shot_fall(t, plate=True)


def shot_fall_flat(t=0.0):
    shot_fall(t, plate=True, still=True)


def fall_ink(AT, ring_list):
    drop = ink_drop("the_drop", (AT[0], AT[1], -.006), opt("dropH", .22), opt("dropR", .05))
    drop.visible_glossy = False     # the water would only show it as black again, and take away its shape

    # the ink it has already let go: a bloom round where it goes in, with a thread leaving toward the lens and the left: the river's first yard
    rect = (-.62, .62, .55, 1.95)
    n_, m_ = 2048, 1800
    wy, wx = np.mgrid[:n_, :m_].astype(np.float32)
    PX, PY = rect[0] + wx / (m_ - 1) * (rect[1] - rect[0]), rect[2] + wy / (n_ - 1) * (rect[3] - rect[2])
    ang = np.arctan2(PY - AT[1], PX - AT[0])
    wob = lerp(spectral(1024, 2.2, 533), wx * 1024 / m_, wy * 1024 / n_)
    dist = np.hypot(PX - AT[0], (PY - AT[1])) * (1 + .5 * wob + .22 * np.sin(ang * 7 + 1.0))
    bloom = sm((.062 - dist) / .05)
    r = np.random.default_rng(535)
    paths = [curve((AT[0], AT[1] - .02), (-.16, .92), (-.44, .56), .012, .006), curve((AT[0] + .02, AT[1]), (.1, 1.02), (.3, .86), .004, .0008), curve((AT[0] - .03, AT[1] + .01), (-.28, 1.1), (-.5, 1.22), .004, .0008)]
    paths += [curve((AT[0] + r.normal(0, .02), AT[1]), (AT[0] + c * .7, AT[1] + d_ * .7), (AT[0] + c * 1.5, AT[1] + d_ * 1.5), .003, .0006) for c, d_ in ((.2, .1), (.1, .34), (-.14, .3), (.28, -.06), (-.26, .06))]
    paths += [curve((.3, 1.95), (.1, 1.6), (AT[0] + .01, AT[1] + .03), .002, .004), curve((-.5, 1.95), (-.2, 1.55), (AT[0] - .01, AT[1] + .03), .002, .004)]
    near_ink = np.maximum(bloom, thread_field(PX, PY, paths, 537) * .96)
    ink_sheet("ink_near", rect, near_ink, ring_list)
    # further off, the threads from the other rings, all coming this way
    rect2 = (-2.6, 2.6, 1.95, 7.0)
    n2, m2 = 1200, 1400
    wy2, wx2 = np.mgrid[:n2, :m2].astype(np.float32)
    PX2, PY2 = rect2[0] + wx2 / (m2 - 1) * (rect2[1] - rect2[0]), rect2[2] + wy2 / (n2 - 1) * (rect2[3] - rect2[2])
    paths2 = [curve((.72, 3.0), (.5, 2.3), (.3, 1.95), .012, .007), curve((-1.1, 2.6), (-.8, 2.2), (-.5, 1.95), .012, .007), curve((.2, 4.6), (.75, 3.8), (.72, 3.0), .012, .012), curve((-1.9, 5.6), (-1.6, 3.6), (-1.1, 2.6), .012, .012), curve((1.7, 6.2), (1.5, 4.0), (.72, 3.0), .012, .012)]
    far_ink = thread_field(PX2, PY2, paths2, 539)
    for cx_, cy_, ra in ((.72, 3.0, .14), (-1.1, 2.6, .15), (.2, 4.6, .2)):
        far_ink = np.maximum(far_ink, sm((ra - np.hypot(PX2 - cx_, PY2 - cy_)) / (ra * .7)))
    ink_sheet("ink_far", rect2, far_ink, ring_list)


MARSH = {"marsh": shot_marsh, "lie": shot_lie, "bare_dark": shot_bare_dark, "bare_lit": shot_bare_lit, "fall": shot_fall, "fall_plate": shot_fall_plate, "fall_flat": shot_fall_flat, "lone": shot_lone}
