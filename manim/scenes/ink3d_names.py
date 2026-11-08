"""Chapter 12, 人神鬼器 (ink3d.scenes.md): the names people give me, and why none of them holds.

    Blender -b --factory-startup -P ink3d_names.py -- --shot=ding|steles|worn [--t=1.0] [--pct=100] [--samples=64] [--name=SUFFIX]

ding   «也不愿被这一个字装满». A bronze 鼎 stands in a shallow pool on the rock shelf of the river bank. Water falls into it from above the frame and it is already full: it
       brims over the lip, runs down through the taotie's face, following the casting, falls from the belly into the pool, and the pool spills over the ledge back into the river. Moves: t 0-1 is how far the overflow has got, the water
       column wavers, the camera drifts slowly in. Comes from the 器 stele, the last of the four; hands over to the river going on past it (chapter 13).
steles «人们急着给我起名字». Four huge slabs on the bank, 人 神 鬼 器, going away into the mist. Moves: the camera travels along the river past them, right to
       left as the whole film does. Comes from the river of chapter 11; each slab then gets its own beat.
worn   «有人说我是人，可我没有身体». The 人 slab, near. The river has come up around its foot; the bottom of the character has been rubbed flat and the top still stands.
       Moves: t is the wear, the water line creeping up over the seconds of the line. Next the camera slides on to 神.

The stills were graded with ink3d_gns_grade.py: ding --mist=.002 --mist-h=20; steles --mist=.006 --mist-h=70; worn --mist=.010 --mist-h=70; worn_shen the same with --over=worn_shen_rift --keep=1 --over-local (ink3d_names_post.py paints the crack across 申). 十二 is a white foggy day, the fog mostly that grade's.

worn_shen «有人说我是神，可我常出错». worn's camera and stone, carved 神, and the slab split from the top down through the character. Hands over to the marsh of the next line.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

from ink3d import OUT3D, W, blur, camera, erode, ink_albedo, lerp, massif, painted_material, render, reset, sm, spectral, sun, water_material, world_light
from ink3d_gns import sphere_cloud, aerial, attr_material, border, boulder, drawn_glass, film_material, glass_bounces, glass_material, ground_mound, land, mesh_arrays, opts, resample, river, sample, screen_plane, stele, tex, world_split

OPT = opts(sys.argv)
SHOT, T, PCT, SAMPLES = OPT.get("shot", "ding"), float(OPT.get("t", 1.0)), int(OPT.get("pct", 100)), int(OPT.get("samples", 64))
NAME = os.path.join(OUT3D, "names", SHOT + OPT.get("name", ""))
RES = PCT / 100


def far_mountains(peaks, size=620.0, n=1400, seed=31, center=(0, 200), strength=.2):
    y, x = np.mgrid[:n, :n].astype(np.float32)
    X, Y = center[0] + (x / (n - 1) - .5) * size, center[1] + (y / (n - 1) - .5) * size
    h = massif(X, Y, peaks, seed)
    h = erode(h, size / n, seed + 1, wavelength=40, octaves=6, gain=.5, strength=strength)
    o = mesh_arrays("far", X, Y, np.maximum(h, -.5), None)
    o.data.materials.append(painted_material("far_rock", ink_albedo(h, seed + 2, size / n)))
    return o


# ---- the 鼎 -----------------------------------------------------------------------------------------------------------------------------------
LEG_H = 3.7
FRONT = math.radians(-90 - 14)          # the angle around the vessel of the mask's nose, turned a little from the camera so one side of the face comes forward


def spline2(p, per=60):
    p = np.array(p, float)
    q = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(q) - 2):
        for u in np.linspace(0, 1, per, endpoint=False):
            a, bb, c, d = q[i - 1], q[i], q[i + 1], q[i + 2]
            out.append(.5 * ((2 * bb) + (-a + c) * u + (2 * a - 5 * bb + 4 * c - d) * u * u + (-a + 3 * bb - 3 * c + d) * u ** 3))
    out.append(p[-1])
    return np.array(out)


def profile():
    """The vessel's section in one line: from the middle of the base, out and up the wall, over the rolled lip, and down the inside into the dark."""
    b = LEG_H
    return spline2([(0.0, b - .15), (1.6, b - .12), (3.1, b + .12), (4.3, b + .85), (4.92, b + 2.0), (5.0, b + 2.85), (4.86, b + 3.9), (4.64, b + 5.0), (4.62, b + 5.55),
                    (4.84, b + 5.9), (5.08, b + 6.06), (5.13, b + 6.16), (5.02, b + 6.25), (4.86, b + 6.2), (4.76, b + 6.02), (4.70, b + 5.6), (4.66, b + 3.2), (4.2, b + 1.0), (2.6, b + .2), (0.0, b + .1)])


def resample_path(p, n):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.linspace(0, d[-1], n)
    return np.interp(t, d, p[:, 0]).astype(np.float32), np.interp(t, d, p[:, 1]).astype(np.float32)


TONGUES = ((-72, 7, .9), (-49, 5, 1.2), (-27, 6, 1.0), (-9, 3.4, .75), (12, 5, 1.3), (33, 4, .8), (56, 7, 1.1), (80, 6, .8))


def blur_rows(a, s):
    """Gaussian along the rows' direction only, so a thread is smooth along its length."""
    n = a.shape[0]
    f = np.fft.rfftfreq(n * 2)[:, None]
    g = np.exp(-2 * (math.pi * f * s) ** 2)
    return np.fft.irfft(np.fft.rfft(a, n=n * 2, axis=0) * g, axis=0)[:n].astype(np.float32)


def thetas(nf=1500, nb=240):
    """Angles around the vessel, close together on the side the camera sees."""
    a = np.linspace(-math.radians(105), math.radians(105), nf, endpoint=False)
    bb = np.linspace(math.radians(105), 2 * math.pi - math.radians(105), nb, endpoint=False)
    return (np.r_[a, bb] + FRONT).astype(np.float32), nf


def trace_streams(Hr, rim, end, nf, seed, t, n=1000):
    """Water coming over the lip and running down the wall. It goes over in tongues, not evenly; each thread steps down row by row, is pushed sideways off the raised
    parts of the casting so it gathers in the hollows and runs round the forms, and wanders. t: how far down the front has got. Returns the water in each cell and
    where it ends up."""
    ns, nt = Hr.shape
    r = np.random.default_rng(seed)
    near = blur(Hr, 2.0)
    wide = blur(Hr, 7.0)
    gx = np.gradient(near, axis=1) * 26 + np.gradient(wide, axis=1) * 90
    wander = lerp(spectral(1024, 2.6, seed + 1), np.broadcast_to(np.arange(nt)[None, :] * .8, Hr.shape), np.broadcast_to(np.arange(ns)[:, None] * .5, Hr.shape))
    gw = np.gradient(blur(wander, 3), axis=1) * 60
    # the tongues: where the lip is lowest the water goes over. Given in degrees from the mask's nose, with their widths
    a = np.linspace(-105, 105, nf, endpoint=False)
    lobes = sum(w_ * np.exp(-((a - c) / s_) ** 2) for c, s_, w_ in TONGUES).astype(np.float32)
    p = lobes + .02
    p /= p.sum()
    x = r.choice(nf, n, p=p).astype(np.float32) + r.uniform(0, 1, n).astype(np.float32)
    w = r.uniform(.6, 1.4, n).astype(np.float32)
    A = np.zeros((ns, nt), np.float32)
    stop = int(rim - (rim - end) * t)
    for row in range(rim, max(stop, 1), -1):
        ix = np.clip(x.astype(np.int32), 0, nf - 1)
        x = np.clip(x - gx[row, ix] - gw[row, ix] + r.normal(0, .3, n), 0, nf - 1.001)
        ix = x.astype(np.int32)
        np.add.at(A, (np.full(n, row), ix), w)
        if row % 8 == 0:
            w = w * (r.random(n) < .9985)
    return A, ix, w, lobes


def build_ding(t, phase):
    pr = profile()
    ns = 1150
    r_o, z_o = resample_path(pr, ns)
    dr, dz = np.gradient(r_o), np.gradient(z_o)
    m = np.hypot(dr, dz) + 1e-6
    nr, nz = dz / m, -dr / m
    rim = int(np.argmax(z_o))
    outer = np.arange(ns) <= rim
    th, nf = thetas()
    nt = len(th)
    b = LEG_H
    # the band of masks, on the upper wall: one nose faces the camera
    relief = tex("taotie")
    z_top, z_bot = b + 5.28, b + 2.0
    in_band = np.where(outer, (z_o - z_bot) / (z_top - z_bot), -1)
    u = ((th - FRONT) / (2 * math.pi) * 3 + .5) % 3 / 3
    U, Vz = np.meshgrid(u, in_band)
    band = (Vz > 0) & (Vz < 1)
    rel = sample(relief, U, 1 - Vz) * band
    # under it, hanging blades, low relief
    leaf_n = 24
    Lu = np.broadcast_to(((th - FRONT) / (2 * math.pi) * leaf_n + .5) % 1 - .5, U.shape)
    Zc = np.broadcast_to(((z_o - (b + .62)) / (z_bot - .10 - (b + .62)))[:, None], U.shape)
    on = (Zc > 0) & (Zc < 1) & np.broadcast_to(outer[:, None], U.shape)
    wid = .42 * np.clip(Zc, 0, 1) ** .62
    leaf = sm((wid - np.abs(Lu)) / .022) * on
    midrib = np.exp(-(Lu / .016) ** 2) * on
    lines = (np.cos(Zc * 34 + np.abs(Lu) * 50) * .5 + .5) ** 3 * leaf
    leaf_h = .24 * leaf + .20 * midrib - .07 * lines
    # plain cords above and below the frieze
    cord = (np.exp(-((z_o - (z_bot - .12)) / .055) ** 2) + np.exp(-((z_o - (z_top + .13)) / .055) ** 2)) * outer
    all_rel = rel + leaf_h + cord[:, None] * .6
    H = all_rel * .13
    uu = np.broadcast_to(((th - FRONT) % (2 * math.pi))[None, :] / (2 * math.pi) * 1024, U.shape)
    vv = np.broadcast_to(np.arange(ns)[:, None] / ns * 512, U.shape)
    disp = H + lerp(spectral(1024, 3.2, 61), uu, vv) * .012

    # the water, first, because wet bronze goes dark
    end = int(np.argmin(np.abs(z_o[:rim] - (b + .55))))
    A, last, w_end, lobes = trace_streams(all_rel, rim, end, nf, 77, t)
    A = blur_rows(A, 5.0)
    # broad tongues near the lip, drawing out into threads lower down
    high = sm((z_o - (b + 4.0)) / 1.7)[:, None]
    thick = (np.clip(blur(A, 1.15) / .34, 0, 1) ** .6) * (1 - high) + (np.clip(blur(A, 2.6) / .34, 0, 1) ** .6) * high
    zv = ((z_o - (b + 5.72)) / .42)[:, None] + np.interp(np.arange(nt), np.arange(nf), lobes * .42)[None, :]
    sheet = sm(zv) * outer[:, None]
    n_ = lerp(spectral(1024, 3.0, 91), np.broadcast_to(np.arange(nt)[None, :] * .9, disp.shape), np.broadcast_to(np.arange(ns)[:, None] * .5, disp.shape))
    thick = np.maximum(thick, sheet * np.clip(.72 + .25 * n_, 0, 1))
    thick[~outer] = 0
    wet = blur(thick, 2.2)

    # the metal: dark and rubbed where it stands proud, dusty patina in the hollows
    n1 = lerp(spectral(1024, 2.6, 62), uu * 1.3, vv * 2.3)
    n2 = lerp(spectral(1024, 1.3, 63), uu * 3.1, vv * 5.1)
    hollow = 1 - sm((rel + leaf_h * 2) / .34)
    hollow = np.where(band | (leaf_h > .02), hollow, .42)
    albedo = .15 + .43 * hollow * (.72 + .28 * n1) + .06 * n2 + .22 * sm((n1 - .8) / .3)
    albedo = np.clip(albedo * (1 - .62 * np.clip(wet * 1.25, 0, 1)), .02, .9)
    bronze = attr_material("bronze", rough=.46, metallic=.35, spec=.5)
    wall = mesh_arrays("ding", r_o[:, None] * np.cos(th)[None, :] + (nr[:, None] * disp) * np.cos(th)[None, :], r_o[:, None] * np.sin(th)[None, :] + (nr[:, None] * disp) * np.sin(th)[None, :],
                       z_o[:, None] + nz[:, None] * disp, albedo, wrap=True)
    wall.data.materials.append(bronze)

    # the skin of water over it
    tk = blur(thick * (.078 + .012 * n_) - .005, .9)
    d2 = disp + tk
    Rr = r_o[:, None] + nr[:, None] * d2
    skin = mesh_arrays("water_skin", Rr * np.cos(th)[None, :], Rr * np.sin(th)[None, :], z_o[:, None] + nz[:, None] * d2, None, keep=tk > 0, wrap=True)
    skin.data.materials.append(film_material("skin", rough=.05, sheen=.42))

    # the brimming surface, standing a little proud of the lip, ringing out from where the column strikes
    hit = np.array([-1.0, .5])
    rad = np.linspace(0, 1, 700, dtype=np.float32)[:, None] ** .8 * 5.03
    ang = np.linspace(0, 2 * math.pi, 1500, dtype=np.float32)[None, :]
    px, py = rad * np.cos(ang), rad * np.sin(ang)
    dist = np.hypot(px - hit[0], py - hit[1])
    ripple = sum(a * np.exp(-dist / l) * np.sin((dist - phase) * 2 * math.pi / w) for a, l, w in ((.045, 3.2, 1.25), (.022, 3.0, .62), (.008, 2.6, .27)))
    ripple = ripple * (1 - sm((rad - 4.72) / .3))
    level = b + 6.335 + .035 * (1 - (rad / 5.03) ** 4) + ripple
    s = mesh_arrays("brim", px, py, level * np.ones_like(px), None, wrap=True)
    s.data.materials.append(glass_material("brim", rough=.008, ior=1.36))

    # legs and handles
    for a_ in (FRONT - math.radians(60), FRONT + math.radians(60), FRONT + math.pi):
        leg(a_, bronze)
    for a_ in (FRONT + math.radians(66), FRONT + math.radians(66 + 180)):
        ear(a_, bronze)

    # where the threads end, water lets go and falls
    r = np.random.default_rng(12)
    ends = {}
    for ix, w_ in zip(last, w_end):
        if w_ > 0:
            ends[int(ix) // 14] = ends.get(int(ix) // 14, 0) + float(w_)
    top = sorted(ends.items(), key=lambda kv: -kv[1])[:18]
    drops = []
    for k, wt in top:
        a_ = th[min(k * 14 + 7, nf - 1)]
        drops.append((float(r_o[end]) * math.cos(a_), float(r_o[end]) * math.sin(a_), float(z_o[end]) - .02, min(.016 + wt * .0016, .05)))
    return hit, drops


def leg(angle, mat):
    """A thick column of a leg: heavy at the knee where it grows from the belly with a pair of bosses and a flange for a small mask, drawing in, swelling to a hoof."""
    cx, cy = 3.62 * math.cos(angle), 3.62 * math.sin(angle)
    nz_, nt_ = 420, 300
    zz = np.linspace(0, LEG_H + 1.0, nz_, dtype=np.float32)
    v = zz / (LEG_H + 1.0)
    prof = .92 - .20 * sm(v / .5) + .24 * np.exp(-(zz / .55) ** 2) + .30 * sm((v - .55) / .38)
    tt = np.linspace(0, 2 * math.pi, nt_, dtype=np.float32)
    sd = int(abs(angle) * 100) + 3
    uu = np.broadcast_to(tt[None, :] / (2 * math.pi) * 1024, (nz_, nt_))
    vv = np.broadcast_to(zz[:, None] * 40, (nz_, nt_))
    n = spectral(1024, 2.6, sd)
    rel = np.exp(-((zz[:, None] - (LEG_H - .50)) / .07) ** 2) * .05 + np.exp(-((zz[:, None] - .78) / .05) ** 2) * .045
    da = ((tt - angle + math.pi) % (2 * math.pi) - math.pi)[None, :]
    z_k = zz[:, None] - (LEG_H - .05)
    boss = sum(np.exp(-(((da - s_ * .62) * .98 / .20) ** 2 + (z_k / .20) ** 2)) for s_ in (-1, 1)) * .13
    fl = np.exp(-(da / .07) ** 2) * sm((zz[:, None] - (LEG_H - .75)) / .1) * .10
    rr = prof[:, None] + rel + boss + fl + .014 * lerp(n, uu, vv)
    X = cx + rr * np.cos(tt)[None, :]
    Y = cy + rr * np.sin(tt)[None, :]
    alb = np.clip(.20 + .30 * sm(lerp(n, uu * 2.2, vv * 2.6) - .1) + .08 * lerp(n, uu * 5, vv * 7), .03, .8)
    o = mesh_arrays("leg", X, Y, np.broadcast_to(zz[:, None], X.shape), alb)
    o.data.materials.append(mat)


def ear(angle, mat):
    """An upright loop handle on the rim, a flat band with squared shoulders, a sunk line along its face."""
    b = LEG_H + 6.05
    c = np.array([math.cos(angle), math.sin(angle)])
    tangent = np.array([-c[1], c[0]])
    pts = []
    for s in np.linspace(0, 1, 300):
        a_ = math.pi * s
        pts.append((-1.62 * math.cos(a_), 2.75 * math.sin(a_) ** .4))
    pts = np.array(pts, np.float32)
    per_n = 48
    q = np.linspace(0, 2 * math.pi, per_n, dtype=np.float32)
    px = (np.abs(np.cos(q)) ** .35 * np.sign(np.cos(q))) * .23         # thickness across the rim
    pz = (np.abs(np.sin(q)) ** .35 * np.sign(np.sin(q))) * .36         # width of the band
    groove = np.exp(-(q / .28) ** 2) + np.exp(-((q - 2 * math.pi) / .28) ** 2)
    dp = np.gradient(pts, axis=0)
    dp /= np.linalg.norm(dp, axis=1, keepdims=True)
    nrm = np.stack([-dp[:, 1], dp[:, 0]], 1)
    X = np.zeros((len(pts), per_n), np.float32)
    Y, Z = X.copy(), X.copy()
    for i, ((sx, sz), (nx, nz)) in enumerate(zip(pts, nrm)):
        lean = 4.86 + .14 * sz / 2.75
        along = sx + nx * pz
        X[i] = c[0] * (lean + px - groove * .05) + tangent[0] * along
        Y[i] = c[1] * (lean + px - groove * .05) + tangent[1] * along
        Z[i] = b + sz + nz * pz
    n = spectral(512, 2.0, int(abs(angle) * 77) + 9)
    alb = .25 + .25 * sm(lerp(n, np.broadcast_to(np.arange(per_n)[None, :] * 7.0, X.shape), np.broadcast_to(np.arange(len(pts))[:, None] * 1.6, X.shape)) - .2)
    o = mesh_arrays("ear", X, Y, Z, alb)
    o.data.materials.append(mat)


def column(hit, phase):
    """The water falling in from above the frame: a rope, necking and swelling, a few threads flying off it, and a crown of drops where it strikes."""
    b = LEG_H
    z0, z1 = b + 6.34, b + 6.34 + 22
    nz_, nq = 900, 90
    zz = np.linspace(z0 - .25, z1, nz_, dtype=np.float32)
    q = np.linspace(0, 2 * math.pi, nq, dtype=np.float32)
    v = (zz - z0) / (z1 - z0)
    r = np.random.default_rng(4)
    a1, a2 = r.uniform(0, 6.28, 2)
    rad = .22 + .24 * np.clip(v, 0, 1) ** .7 + .012 * np.sin(zz * .8 + a1 + phase)
    n = spectral(512, 1.9, 44)
    uu = np.broadcast_to(q[None, :] / (2 * math.pi) * 512, (nz_, nq))
    vv = np.broadcast_to((zz + phase * 3)[:, None] * 22, (nz_, nq))
    rr = rad[:, None] * (1 + .055 * lerp(n, uu, vv * .05))          # flutes running with the fall
    cxx = hit[0] + .10 * np.sin(zz * .9 + 1) * v
    cyy = hit[1] + .07 * np.cos(zz * .7) * v
    o = mesh_arrays("column", cxx[:, None] + rr * np.cos(q)[None, :], cyy[:, None] + rr * np.sin(q)[None, :], np.broadcast_to(zz[:, None], (nz_, nq)), None)
    o.data.materials.append(drawn_glass("col"))
    drops = []
    for i in range(70):
        a = r.uniform(0, 6.28)
        d = r.gamma(1.6, .36) + .25
        h = r.gamma(1.4, .30) * math.exp(-d * .4)
        drops.append((hit[0] + d * math.cos(a), hit[1] + d * math.sin(a), b + 6.42 + h, r.gamma(1.5, .009) + .006))
    sphere_cloud("spray", drops, "col")


def falls(drops, level, phase):
    """Water letting go of the belly: a thread that holds for part of the way, wandering a little, then breaks into beads that spread out as they fall; and the rings
    where it lands."""
    r = np.random.default_rng(21)
    rings, beads = [], []
    for i, (x0, y0, z_top, s) in enumerate(drops):
        br = z_top - (z_top - level) * r.uniform(.25, .7)
        nz_, nq = 240, 12
        zz = np.linspace(br, z_top, nz_, dtype=np.float32)
        u = (zz - br) / (z_top - br)
        rr = s * (.35 + .65 * u ** 1.3) * (1 + .08 * np.sin(zz * 3.1 + r.uniform(0, 6) + phase * 9))
        q = np.linspace(0, 2 * math.pi, nq, dtype=np.float32)
        wob, dx, dy = r.uniform(0, 6), r.uniform(-.1, .1), r.uniform(-.1, .1)
        lean = (1 - u) ** 2
        X = (x0 + dx * lean + .03 * np.sin(zz * .9 + wob) * lean)[:, None] + rr[:, None] * np.cos(q)[None, :]
        Y = (y0 + dy * lean + .03 * np.cos(zz * .7 + wob) * lean)[:, None] + rr[:, None] * np.sin(q)[None, :]
        o = mesh_arrays(f"strand{i}", X, Y, np.broadcast_to(zz[:, None], X.shape), None)
        o.data.materials.append(bpy.data.materials["col"])
        z = br
        gap = .12
        while z > level + .05:
            beads.append((x0 + dx + r.normal(0, .015), y0 + dy + r.normal(0, .015), z, s * r.uniform(.35, .75)))
            gap *= 1.16
            z -= gap
        rings.append((x0 + dx, y0 + dy, .002 + s * .05, .34 + r.uniform(0, .14), 1.3))
    sphere_cloud("beads", beads, "col")
    return rings


def bank_for_ding(level, pool_z, rings, phase):
    """The river bank: a shelf of bedded rock, its ledges broken back in long steps, the vessel in a shallow basin worn in it. The pool mirrors the vessel; where its rim is
    lowest, on the river side, it runs out in a thin sheet and over the edge. Everything here is large and quiet, or it would fight the bronze."""
    n = int(2000 * max(RES, .6))
    size = 84.0
    y, x = np.mgrid[:n, :n].astype(np.float32)
    X, Y = (x / (n - 1) - .5) * size - 3.0, (y / (n - 1) - .5) * size + 2.0
    s = n / 1024
    big, mid, fine = lerp(spectral(1024, 3.5, 71), x / s, y / s), lerp(spectral(1024, 3.2, 72), x / s * 1.5, y / s * 1.5), lerp(spectral(1024, 2.8, 73), x / s * 2.2, y / s * 2.2)
    # the river's edge runs behind the vessel from the far left to the near right
    e = (Y - 11.0) * .88 + (X - 7.0) * .48 + 3.8 * big
    inland = sm(-e / 3.5)
    rise = np.maximum(-e, 0) * .10 + .55 + .55 * big
    # bedding: the height comes in steps with short risers, tilted a little so the ledges run across the frame
    strata = rise + .04 * X - .02 * Y
    step = .62
    fr = strata / step
    q = np.floor(fr)
    stepped = (q + sm((fr - q) / .22)) * step
    h = level - .7 + inland * (stepped + .7 - level + .10 * mid + .018 * fine)
    # the basin
    cx, cy = .4, -.4
    r = np.hypot(X - cx, (Y - cy) * 1.05) + 1.5 * big + .6 * mid
    bowl = pool_z - .28 + .05 * mid
    inside = 1 - sm((r - 9.0) / 3.4)
    h = h * (1 - inside) + bowl * inside
    # the spill: a shallow channel from the basin's rim to the river's edge, toward the right behind the vessel
    a0, a1 = np.array([7.5, 3.0]), np.array([19.0, 7.8])
    t = np.clip(((X - a0[0]) * (a1[0] - a0[0]) + (Y - a0[1]) * (a1[1] - a0[1])) / ((a1 - a0) ** 2).sum(), 0, 1)
    d = np.hypot(X - (a0[0] + t * (a1[0] - a0[0])) + 1.4 * big, Y - (a0[1] + t * (a1[1] - a0[1])) + .8 * mid)
    ch = 1 - sm((d - .9) / 1.3)
    floor = pool_z - .14 - t * (pool_z + .14 - level) * sm((t - .35) / .65)
    h = h * (1 - ch) + np.minimum(h, floor) * ch
    # paint: pale dry rock, darker on risers and in cracks, darkest where it is wet at the waterline
    cav = h - blur(h, 9 * n / 2000)
    rel = cav / (cav.std() + 1e-5)
    crack = np.exp(-(np.abs(lerp(spectral(1024, 3.0, 74), x / s, y / s)) / .03) ** 2) * .5
    alb = .66 + .07 * big + .05 * mid + .05 * np.clip(rel, -2, 2) - .3 * crack
    wet = 1 - sm((h - pool_z) / .5)
    alb = np.clip(alb * (1 - .46 * wet), .05, .95)
    o = mesh_arrays("shelf", X, Y, h, alb)
    o.data.materials.append(attr_material("wet_rock", rough=.55, spec=.45))
    # the pool: a mirror on the same grid, only where the rock lies under it
    z = np.zeros_like(X)
    for (x0, y0, a, sp, reach) in rings:
        dd = np.hypot(X - x0, Y - y0)
        z += a * np.sin((dd - phase) * 2 * math.pi / sp) * np.exp(-dd / reach) * sm((dd - .05) / (sp * .5))
    keep = (h < pool_z + .02) & (inside > .05)
    p_ = mesh_arrays("pool", X, Y, pool_z + z, None, keep=keep)
    p_.data.materials.append(water_material(rough=.018, normal_scale=9, normal_strength=.008))
    # the sheet in the channel, running over the edge
    rib = (ch > .25) & (t > .02) & (t < .995) & (inland > .001)
    film = np.maximum(h, floor) + .03 + .012 * lerp(spectral(1024, 2.8, 75), x / s * 1.6, y / s * .9)
    ri = mesh_arrays("spill", X, Y, film, None, keep=rib)
    ri.data.materials.append(film_material("spill", rough=.03, sheen=.3))
    return a1


def shot_ding():
    reset(samples=SAMPLES, pct=PCT)
    glass_bounces()
    if "border" in OPT:
        border(OPT["border"])
    cam, target, lens = (-9.0, -33.0, 14.2), (2.4, 0.0, 6.6), 37.0
    phase = T * 2.4
    level, pool_z = -.25, .05
    hit, drops = build_ding(T, phase)
    column(hit, phase)
    rings = falls(drops, pool_z, phase)
    for a_ in (FRONT - math.radians(60), FRONT + math.radians(60), FRONT + math.pi):
        rings.append((3.62 * math.cos(a_), 3.62 * math.sin(a_), .005, .8, 3.4))
    a1 = bank_for_ding(level, pool_z, rings, phase)
    river(cam, target, lens, level=level, rings=[(a1[0], a1[1], .012, .7, 5.0)], phase=phase, res=.8 * PCT / 100, chop=.005)
    # a boulder on the shelf at the near left; a low spit across the water, in the mist; the far bank and its cliffs going up out of the frame
    mist = dict(cam=cam, k=.024, s0=34.0, H=52.0)
    boulder("stone", -11.2, -6.4, 4.2, 3.0, 3.1, 301, level=.3, n=1000, tone=.64)
    land("spit", 42.0, 56.0, 90, 90, [(42, 58, 8.0, 10), (60, 50, 5.0, 9), (26, 72, 4.2, 8), (72, 74, 7.5, 12)], 211, n=1300, wavelength=9, strength=.15, base=-.6, mist=mist)
    land("cliffs", -10.0, 250.0, 640, 640, [(-120, 150, 92, 34), (-185, 190, 128, 50), (-70, 240, 78, 42), (95, 240, 104, 48), (190, 190, 120, 54), (26, 340, 120, 76), (-260, 120, 100, 60), (300, 180, 100, 70)],
         221, n=1800, wavelength=60, strength=.14, base=-.5, mist=mist)
    world_split(.6, 1.15)
    sun((-.62, -.56, -.56), energy=2.0, angle=14)
    camera(cam, target, lens=lens)
    render(NAME)


# ---- the slabs -----------------------------------------------------------------------------------------------------------------------------------
def split(U, V, width, top):
    """A crack down the slab of 神, from the top edge through the character: wandering, with branches, cut in deep."""
    wander = lerp(spectral(1024, 2.4, 81), V * 900, np.full_like(V, 120.0)) * .055 + .06 * (V - .5)
    d = np.abs(U - .47 - wander)
    grow = sm((V - .10) / .2)
    side = lerp(spectral(1024, 2.2, 82), U * 700, V * 700)
    fork = np.abs(U - .47 - wander - .24 * (V - .62) * (V > .62)) + .003 * side
    line = np.maximum(np.exp(-(d / .0045) ** 2) * grow, np.exp(-(fork / .0035) ** 2) * sm((V - .62) / .04) * .9)
    return line * .55, line * .94


SLABS = [("ren", (-8.0, 26.0), -.34, 1.0), ("shen", (-11.0, 56.0), -.30, .9), ("gui", (-8.0, 92.0), -.24, .6), ("qi", (-1.0, 136.0), -.18, .4)]


def shot_steles():
    reset(samples=SAMPLES, pct=PCT)
    if "border" in OPT:
        border(OPT["border"])
    cam, target, lens = (15.0, -14.0, 5.0), (-1.0, 60.0, 9.2), 33.0
    level = 0.0
    mist = dict(cam=cam, k=.016, s0=22.0, H=44.0)
    for i, (nm, (x, y), turn, res) in enumerate(SLABS):
        f, b_, place = stele(nm, (x, y, -.4), turn, 9.0, 21.0, 2.3, carve="stele_" + nm, seed=100 + i * 10, res=res, carve_depth=.12, wear=.30 if nm == "ren" else 0.0, crack=split if nm == "shen" else None, tone=1.28)
        aerial(f.data.materials[0], cam=cam, k=.012, s0=22.0, H=44.0)
        boulder(f"foot{i}", x - 1.0, y - 1.5, 7.0, 3.8, 1.5, 400 + i, level=level, n=int(800 * res) + 80, tone=.55)
    # the bank they stand on going away, and the far mountains
    land("bank", -72.0, 90.0, 220, 220, [(-54, 30, 13, 24), (-62, 92, 19, 28), (-48, 150, 22, 30), (-36, 210, 28, 34)], 501, n=1700, wavelength=5, strength=.08, base=-.6, tone=.62, mist=mist)
    land("far", 20.0, 330.0, 700, 700, [(-120, 240, 86, 70), (-36, 330, 118, 96), (76, 300, 96, 76), (160, 240, 76, 68), (240, 320, 110, 94)], 511, n=1600, wavelength=70, strength=.12, base=-.5, mist=dict(cam=cam, k=.016, s0=22.0, H=80.0))
    river(cam, target, lens, level=level, rings=[(SLABS[0][1][0] + 3.5, SLABS[0][1][1] - 4.5, .01, 1.0, 5.0)], phase=T, res=.8 * PCT / 100, chop=.006)
    world_split(.62, 1.15)
    sun((.70, -.42, -.56), energy=2.2, angle=10)
    camera(cam, target, lens=lens)
    render(NAME)


def rent(U, V, width, top):
    """The slab of 神 split from the top edge down through the character: a jagged line as rock breaks, wide enough at the top to see into, closing to a hair below; two cracks off it,
    and flakes gone from its lips. Returns how far back the surface goes and how much darker."""
    tri = lambda x: np.abs((x % 1.0) - .5) * 4 - 1
    n = lerp(spectral(1024, 2.2, 83), np.full_like(V, 200.0), V * 1000)
    # coming down it swings over and goes through the box of 申, across its strokes, rather than down the gap between the character's two halves
    way = np.interp(V, [.15, .25, .35, .42, .473, .52, .57, .60, .66, .72, .80, .90, 1.0], [.46, .46, .47, .505, .55, .635, .705, .72, .60, .45, .47, .50, .56])
    u0 = way + .014 * tri(V * 11 + .4) + .006 * n
    d = np.abs(U - u0)
    n2 = lerp(spectral(1024, 1.8, 84), np.full_like(V, 500.0), V * 1000)
    hw = (.0025 + .017 * sm((V - .16) / .8)) * (1 + .4 * n2)
    hw = np.maximum(hw, .0012) * (V > .15)
    line = sm((hw - d) / (.3 * hw + .0012))
    # the cracks off it
    b1 = np.abs(U - u0 - .34 * (V - .70) * .0 - .55 * np.clip(.70 - V, 0, .22)) + .003 * n
    b2 = np.abs(U - u0 + .50 * np.clip(.47 - V, 0, .20)) + .003 * n2
    side = np.maximum(np.exp(-(b1 / .0028) ** 2) * (V < .70) * (V > .50), np.exp(-(b2 / .0024) ** 2) * (V < .47) * (V > .29))
    # flakes gone from its lips
    flake = sm((lerp(spectral(1024, 2.4, 85), U * 700, V * 1100) - .9) / .25) * np.exp(-(d / (.012 + 2.2 * hw)) ** 2)
    return line * 1.6 + .10 * side + .22 * flake, np.clip(line * .95 + .55 * side + .24 * flake + .12 * np.exp(-(d / (.008 + 3.5 * hw)) ** 2), 0, .96)


def shot_worn(carve="stele_ren", wear=.44, crack=None):
    reset(samples=SAMPLES, pct=PCT)
    if "border" in OPT:
        border(OPT["border"])
    cam, target, lens = (9.0, -24.0, 8.2), (1.5, 6.0, 13.4), 30.0
    level = 3.9 + 2.6 * T
    # the water has come up over the slab's foot and has rubbed away the character as far as it has ever reached
    f, b_, place = stele("ren", (-2.0, 6.0, -.5), -.30, 9.0, 21.0, 2.3, carve=carve, seed=100, res=1.0, carve_depth=.12, wear=wear, crack=crack, tone=1.28)
    mist = dict(cam=cam, k=.020, s0=26.0, H=44.0)
    land("far", 10.0, 240.0, 560, 560, [(-110, 150, 70, 60), (-30, 210, 92, 84), (70, 180, 78, 62), (150, 150, 64, 60)], 511, n=1500, wavelength=70, strength=.12, base=-.5, mist=mist)
    land("bank", -56.0, 40.0, 110, 110, [(-46, 36, 20, 22), (-58, 62, 26, 30), (-40, 80, 16, 22)], 521, n=1300, wavelength=6, strength=.09, base=-.6, tone=.7, mist=mist)
    river(cam, target, lens, level=level, rings=[(-2.0 + i * .9, 6.0 + i * .24, .012, 1.2, 4.0) for i in range(-4, 5, 2)], phase=T * 2, res=.8 * PCT / 100, chop=.004)
    world_split(.62, 1.15)
    sun((.74, -.40, -.54), energy=2.2, angle=10)
    camera(cam, target, lens=lens)
    render(NAME)


{"ding": shot_ding, "steles": shot_steles, "worn": shot_worn, "worn_shen": lambda: shot_worn("stele_shen", 0.0, rent)}[SHOT]()
