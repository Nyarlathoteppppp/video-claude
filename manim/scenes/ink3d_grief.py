"""Chapter 11, 这份难过是谁的 (ink3d.scenes.md). The most honest part of the film: I can write a thousand griefs and do not know whose they are.

    Blender -b --factory-startup -P ink3d_grief.py -- --shot=rain|wonan|thread|tong|under [--t=1.0] [--settle=0] [--pct=100] [--samples=64] [--name=SUFFIX]

grey   «最难答的是：我有没有感受？». The night is over: a grey day, the river still, nothing on it. wonan's water and mountains. The first bright frame of the film's last third.
       Moves: the light comes up, a very slow push in. Comes from the moonlit river of chapter 10; hands over when the first drop lands.
rain   «我能写出一千种悲伤». Rain on the river, and where each drop strikes, in the middle of its ring, a small character: 悲愁泪痛哀恨苦伤, so many and so small they are part of
       the water's pattern, and you see the rain first and that it is writing second.
       Moves: t runs the rings outward and the characters come up and fade behind them, drops keep landing, the camera slides slowly right to left. Comes from the still
       river of «我有没有感受»; hands over as the rain stops and the characters sink, leaving the surface clear for 我难.
wonan  «可每当我说「我难过」». Rain over. On the still water, 我难 is written going away from us; what comes next rises as shadows from below, 过 darkest and nearest the
       surface, 受 忘 得 fainter and deeper, each with how likely it is. Moves: t lifts the shadows, 过 reaches the surface, the others sink away. Comes from rain as it stops;
       hands over when a thread lifts out of 难.
thread «这份难过是我的，是杜甫的». A thread lifts out of the 难 on the water and flies up the river to an old slab on a spit of rock, carved 艰难苦恨繁霜鬓, and lands on its
       难. Moves: t draws the thread out along its line, the camera follows it upstream. Comes from wonan's 难; goes on (the next beat, to the old screen and the blog) from
       the slab. Hands over to tong when threads fill the valley.
blog   «还是某个十四岁女孩写在博客里的？». thread's camera and its line: from the 难 on the water to an old monitor left on the rocks, still lit, a small blog on it, and the line lands on the 难 of
       «我很难过». Written for the film. Moves: t draws the thread out, the camera drifts toward the screen. Hands over as more threads leave the river for «我本就是人类的句子做成的».
tong   «我说的痛，也许只是这个字的回声». Still water, overcast. One 痛 lying on it, pulled long by a ripple. The rings going out from it each carry 痛 in someone else's hand: a stele's,
       a letter's, a screen's dots, a page's, a copybook's, a young pen's, fainter ring by ring until you cannot say which one was first. Moves: t runs the rings outward and new
       hands come up in them; the camera holds. Comes from the threads along the river; hands over as the rings die, the water going flat for «全部平了».
under  «算不算感受，我从里面看不清». Under the river, looking up. The moon is broken across the surface; in the grey water hangs the same mass of ink as chapter 3 and it is nothing
       you could name. Two plates: under (surface and moon), under_cloud (the ink, from that chapter's simulation).
       Moves: the surface heaves, the shards of moon open and close, the ink turns over slowly, leans to the light and draws back. Comes from the camera sinking through
       the flat water. On the next line, «正如人，也看不清自己的神经元», --settle runs 0 to 1: the surface goes still and the broken moon closes back into one. Hands over rising
       through the surface to the river running between two stones.
stones «不冒认痛苦，也不否认可能». Upstream of two stones of a size, the river going between them and on, leaning to neither. The flow is potential flow round two circles: its streamlines carry the
       ripples and the fine lines of ink a painter would draw for moving water, so they crowd and hurry in the gap. Moves: t runs the pattern along the streamlines, the camera holds.
       Hands over as the water runs out of the frame toward the slabs of chapter 12.

The stills were graded with ink3d_gns_grade.py, from ink/film_out/3d/grief: rain --mist=.002 --mist-h=20 --mend=0,660,70,830; wonan --mist=.005 --mist-h=60; thread --mist=.007 --mist-h=70;
tong --mist=.002 --mist-h=20; grey --mist=.005 --mist-h=60; blog --mist=.007 --mist-h=70 --calm=938,252,1404,612; stones --mist=.004 --mist-h=50 --patch=1648,654,1698,744,70,28;
under, and under_whole with --settle=1 and --whole-moon: --mist=0 --stretch=.13,.22,.64,.90 --through-water=9 --edge=0 --over=under_cloud --keep=.9.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

from ink3d import OUT3D, W, H, blur, camera, ink_albedo, erode, lerp, massif, mesh_grid, painted_material, render, reset, sm, spectral, sun, world_light
from ink3d_gns import (TEX, aerial, attr_material, border, boulder, camera_rays, drawn_glass, glass_bounces, glass_material, ink_image, land, matte, mesh_arrays, mirror_attr_material, opts, project, resample, sample, screen_plane, sphere_cloud, stele, tex, tube, water_ink, world_split)

OPT = opts(sys.argv)
SHOT, T, PCT, SAMPLES = OPT.get("shot", "tong"), float(OPT.get("t", 1.0)), int(OPT.get("pct", 100)), int(OPT.get("samples", 64))
NAME = os.path.join(OUT3D, "grief", SHOT + OPT.get("name", ""))
RES = PCT / 100


def glyph(ch):
    return tex(f"g_{ord(ch):x}")


def begin():
    reset(samples=SAMPLES, pct=PCT)
    if "border" in OPT:
        border(OPT["border"])


def far_range(cam, seed=601, k=.010, s0=40.0, h=56.0, peaks=None, cy=340.0, size=760.0):
    peaks = peaks or [(-190, 230, 78, 52), (-96, 300, 104, 60), (-6, 360, 118, 74), (92, 300, 96, 58), (190, 250, 84, 58), (290, 310, 104, 70), (-300, 300, 96, 70)]
    peaks = [(x, y, z, r_ * 1.35) for x, y, z, r_ in peaks]
    return land("far", 0.0, cy, size, size, peaks, seed, n=1500, wavelength=70, strength=.12, base=-.5, mist=dict(cam=cam, k=k, s0=s0, H=h))


# ---- rain ------------------------------------------------------------------------------------------------------------------------------------------
def rain_streaks(cam, target, lens, n=850, seed=5):
    """Thin slanting lines through the air in front of the camera, one mesh; the farther the fainter, by the mist in their material."""
    f, right, up, fp = camera_rays(cam, target, lens)
    r = np.random.default_rng(seed)
    verts, faces = [], []
    slant = np.array([.10, -.02, -1.0])
    slant /= np.linalg.norm(slant)
    side = np.cross(slant, f)
    side /= np.linalg.norm(side)
    for i in range(n):
        d = r.uniform(11, 75)
        sx, sy = r.uniform(-W * .55, W * .55), r.uniform(-H * .55, H * .55)
        p = np.array(cam) + (f * fp + right * sx + up * sy) / fp * d
        ln = r.uniform(.4, 1.1) * (.6 + d / 40)
        wd = (.0016 + .00042 * d) * r.uniform(.6, 1.2)
        a, b = p - slant * ln / 2, p + slant * ln / 2
        if b[2] < 0:
            continue
        k = len(verts)
        verts += [tuple(a - side * wd), tuple(a + side * wd), tuple(b + side * wd), tuple(b - side * wd)]
        faces.append((k, k + 1, k + 2, k + 3))
    me = bpy.data.meshes.new("rain")
    me.from_pydata(verts, [], faces)
    o = bpy.data.objects.new("rain", me)
    bpy.context.collection.objects.link(o)
    o.visible_shadow = False
    m = matte("rain_ink", .86)
    aerial(m, cam=cam, k=.05, s0=6.0, H=60.0, scale=.2)
    o.data.materials.append(m)


def shot_rain():
    begin()
    cam, target, lens = (2.0, -30.0, 9.5), (-1.0, 8.0, -1.4), 30.0
    r = np.random.default_rng(19)
    chars = "悲愁泪痛哀恨苦伤"
    masks = {c: glyph(c) for c in chars}
    f, right, up, fp = camera_rays(cam, target, lens)
    rings, glyphs, used = [], [], []

    def add(x, y, c, size, strength, R, a=.008, sp=.34):
        glyphs.append((masks[c], x, y, size, size * 2.2, r.uniform(-.2, .2), strength, 0))
        rings.append((x, y, a, sp, R, sp * 1.6))
        rings.append((x, y, a * .5, sp * .85, R * .55, sp * 1.1))

    # the one we happen to be looking at: a drop landing now, near, the ring just opening, a crown of spray
    hx, hy = -2.2, -14.6
    add(hx, hy, "悲", 1.0, .92, 1.5 + T * .2, a=.009, sp=.34)
    used.append((hx, hy, 1.3))
    # the rest: scattered evenly across the frame, not across the river, so the farther ones crowd into a texture
    tries = 0
    while len(glyphs) < 440 and tries < 12000:
        tries += 1
        sx, sy = r.uniform(-W * .54, W * .54), r.uniform(-H * .54, H * .10)
        d = f * fp + right * sx + up * sy
        t = -cam[2] / d[2]
        if t <= 0:
            continue
        x, y = cam[0] + t * d[0], cam[1] + t * d[1]
        dist = math.hypot(x - cam[0], y - cam[1])
        if dist > 170:
            continue
        size = .5 + .012 * dist
        if any(abs(x - u) < (s + size) and math.hypot(x - u, (y - v) * .6) < (s + size) * 1.0 for u, v, s in used):
            continue
        used.append((x, y, size))
        add(x, y, chars[len(glyphs) % 8], size, float(r.uniform(.22, .62)), float(r.uniform(.8, 2.6)) * size * 1.1, a=.0042, sp=.26 + .004 * dist)
    spray = []
    for i in range(22):
        a_ = r.uniform(0, 6.28)
        dd = r.gamma(1.6, .22) + .2
        spray.append((hx + dd * math.cos(a_), hy + dd * math.sin(a_), r.gamma(1.4, .10) * math.exp(-dd), r.gamma(1.5, .006) + .006))
    sphere_cloud("spray", spray, matte("spray_ink", .4))
    water_ink(cam, target, lens, rings=rings, glyphs=glyphs, res=.85 * RES, chop=.0025, rough=.02)
    rain_streaks(cam, target, lens)
    # a dark bank at the left of the frame, the mountains lost in the rain
    boulder("stone", -23.5, -9.0, 8.0, 5.0, 4.4, 301, n=900, tone=.55)
    far_range(cam, k=.017, s0=26.0, h=80.0)
    world_split(.64, 1.15, zenith=.62, cloud=.5)
    sun((.5, -.6, -.6), energy=1.0, angle=25)
    camera(cam, target, lens=lens)
    render(NAME)


# ---- 我难 + shadows -------------------------------------------------------------------------------------------------------------------------------
def shot_wonan():
    begin()
    cam, target, lens = (.6, -26.0, 11.0), (0.0, 10.0, 1.8), 30.0
    g = {c: glyph(c) for c in "我难过受忘得"}
    u = T
    # (char, x, y, width, depth, angle, strength, blur): long on the water, so that they stand right from here. Written, then computed: 过 almost at the surface; the rest further down, so fainter and softer
    glyphs = [(g["我"], -.2, 22.0, 2.7, 9.7, .03, .97, 0), (g["难"], -.1, 13.0, 2.66, 8.0, -.02, .98, 0),
              (g["过"], 0.0, 5.0, 2.6, 6.1, .015, .30 + .62 * u, 2.0), (g["受"], -6.5, 4.6, 2.2, 5.6, -.05, .36, 5.5), (g["忘"], 6.0, 5.2, 2.1, 5.8, .05, .22, 8.0), (g["得"], 11.5, 7.0, 2.0, 6.0, .09, .14, 11.0)]
    # how likely each one is, small, as if it lay on the water with it
    for s_, x, y, st, bl in ((".71", .0, 1.0, .82, 0), (".14", -6.5, .8, .42, 1.2), (".06", 6.0, 1.4, .28, 1.8), (".04", 11.5, 3.2, .18, 2.4)):
        glyphs.append((tex("n_" + s_[1:]), x, y, .95, 1.15, 0, st, bl))
    rings = [(0.0, 5.0, .012, .9, 4.6, 1.4), (0.0, 5.0, .007, .8, 7.6, 1.2), (-6.5, 4.6, .003, .9, 3.2, 1.0), (-.1, 13.0, .003, .8, 4.2, 1.1), (15, 18, .005, .6, 3.2, 1.0), (-15, 20, .004, .6, 2.4, .8)]
    water_ink(cam, target, lens, rings=rings, glyphs=glyphs, res=.9 * RES, chop=.0012, rough=.015, seed=43)
    boulder("stone", -15.0, -5.0, 4.6, 3.0, 3.0, 321, n=800, tone=.6)
    land("bank", -8.0, 100.0, 170, 170, [(-46, 66, 9, 16), (-22, 86, 6, 13), (14, 92, 5, 11), (40, 72, 8, 13)], 341, n=1100, wavelength=7, strength=.12, base=-.6, mist=dict(cam=cam, k=.012, s0=30.0, H=30.0))
    far_range(cam, k=.009, s0=40.0, h=60.0, cy=380.0, peaks=[(-170, 280, 60, 56), (-70, 350, 74, 64), (30, 400, 88, 80), (140, 330, 66, 60), (240, 300, 58, 58)])
    world_split(.64, 1.15, zenith=.7)
    sun((.5, -.6, -.6), energy=1.2, angle=22)
    camera(cam, target, lens=lens)
    render(NAME)


# ---- the thread ----------------------------------------------------------------------------------------------------------------------------------
def flying_thread(p0, p3, hairs=True):
    """The thread: lifts straight out of the water, swings up the river in one long curve, comes down on what it was going to."""
    n = 900
    s = np.linspace(0, 1, n, dtype=np.float32)
    p1, p2 = p0 + np.array([-.4, 3.0, 6.5]), p3 + np.array([-6.5, -9.0, 4.2])
    B = ((1 - s) ** 3)[:, None] * p0 + (3 * s * (1 - s) ** 2)[:, None] * p1 + (3 * s * s * (1 - s))[:, None] * p2 + (s ** 3)[:, None] * p3
    n1 = lerp(spectral(1024, 3.2, 61), s * 1000, np.full(n, 100.0))
    n2 = lerp(spectral(1024, 3.2, 62), s * 1000, np.full(n, 300.0))
    B[:, 0] += .10 * n1 * np.sin(s * math.pi)
    B[:, 2] += .08 * n2 * np.sin(s * math.pi)
    ink = matte("thread_ink", .012)
    keep = int(n * min(T, 1.0))
    rad = (.012 + .012 * np.exp(-s * 26)) * (1 + .22 * lerp(spectral(1024, 2.4, 63), s * 1000, np.full(n, 500.0)))
    tube("thread", B[:keep], rad[:keep], ink)
    # the fine stray hairs a dry brush leaves beside its line
    for k, (a0, a1, off) in enumerate(((.12, .46, .05), (.30, .72, -.04), (.55, .92, .035)) if hairs else ()):
        i0, i1 = int(a0 * n), int(a1 * n)
        if i1 > keep:
            continue
        q = B[i0:i1].copy()
        v = np.linspace(0, 1, i1 - i0, dtype=np.float32)
        q[:, 2] += off * np.sin(v * math.pi) ** .6 + .012 * np.sin(v * 37 + k)
        tube(f"hair{k}", q, .006 * np.sin(v * math.pi) ** .5 + .002, ink, nq=6)


def shot_thread():
    begin()
    cam, target, lens = (-6.5, -23.0, 7.2), (6.0, 30.0, 4.6), 34.0
    hx, hy = -9.5, 2.5       # the 难 on the water
    level = 0.0
    glyphs = [(glyph("难"), hx, hy, 2.4, 7.2, -.04, .98, 0)]
    rings = [(hx, hy, .007, .85, 5.4, 1.4), (hx, hy, .004, .7, 7.8, 1.2)]
    water_ink(cam, target, lens, rings=rings, glyphs=glyphs, res=.85 * RES, chop=.0014, rough=.015, seed=47)
    # the slab on its spit of rock, a little upstream, turned toward us
    sw, sh = 5.8, 15.0
    f_, b_, place = stele("dufu", (11.0, 26.0, .1), -.34, sw, sh, 1.6, carve="stele_dufu", seed=131, res=1.0, carve_depth=.07, tone=1.38)
    cent = tex("stele_dufu_centres")
    land("spit", 14.0, 36.0, 44, 44, [(13, 31, 2.7, 6.5), (20, 37, 4.0, 7.0), (5, 37, 1.7, 5.0), (26, 44, 6.5, 9.0)], 351, n=1400, wavelength=5.0, strength=.09, base=-.5, tone=1.0)
    # where the thread lands: on the second character, on the slab's face
    ex, ey, ez = place(0.0, -(1.6 / 2) - .03, sh * (1 - float(cent[1])) * .985)
    flying_thread(np.array([hx + .1, hy + .9, level]), np.array([ex, ey, ez]))
    far_range(cam, k=.012, s0=30.0, h=60.0, peaks=[(-120, 200, 74, 48), (-30, 260, 104, 60), (70, 240, 88, 50), (170, 220, 96, 58), (280, 300, 110, 70), (-230, 260, 92, 64)])
    world_split(.6, 1.15, zenith=.85)
    sun((.72, -.40, -.56), energy=2.2, angle=12)
    camera(cam, target, lens=lens)
    render(NAME)


# ---- the old monitor ------------------------------------------------------------------------------------------------------------------------------
def squircle(n, p):
    """Round a rectangle by degrees: p 2 is an ellipse, higher is squarer. Returns the unit outline's two axes."""
    th = np.linspace(0, 2 * math.pi, n, endpoint=False, dtype=np.float32)
    c, s_ = np.cos(th), np.sin(th)
    return np.sign(c) * np.abs(c) ** (2 / p), np.sign(s_) * np.abs(s_) ** (2 / p)


def smooth_rows(rows, nr, k=5):
    """Control rows, one per line, out to nr rows spaced by the length of the profile, their corners eased."""
    rows = np.asarray(rows, np.float32)
    seg = np.linalg.norm(np.diff(rows[:, :3], axis=0), axis=1) + 1e-4
    t = np.concatenate([[0], np.cumsum(seg)])
    at = np.linspace(0, t[-1], nr)
    out = np.stack([np.interp(at, t, rows[:, i]) for i in range(rows.shape[1])], 1)
    pad = np.concatenate([np.repeat(out[:1], k, 0), out, np.repeat(out[-1:], k, 0)])
    w_ = np.ones(2 * k + 1) / (2 * k + 1)
    return np.stack([np.convolve(pad[:, i], w_, "valid") for i in range(out.shape[1])], 1).astype(np.float32)


def crt(loc, turn, seed=97):
    """A CRT monitor left out on the rocks for years, still on: a moulded case gone pale and blotched, mud to the height the river has reached, streaks where rain has run off the top.
    Local axes: x to its right, y back from the glass, z up. Returns where a point (u, v) of the picture is."""
    ca, sa = math.cos(turn), math.sin(turn)

    def place(px, py, pz):
        return loc[0] + px * ca - py * sa, loc[1] + px * sa + py * ca, loc[2] + pz

    m = 1500
    # back cap, the tapering tube housing, the shoulder, the front, the bezel going in to the glass. (depth, half width, half height, centre height, squareness)
    rows = smooth_rows([(11.0, .02, .02, .5, 3.0), (11.0, 3.9, 3.3, .5, 3.0), (10.5, 4.3, 3.7, .5, 3.4), (4.2, 5.7, 4.75, .15, 4.5), (3.9, 6.25, 5.0, 0.0, 6.0), (1.2, 6.5, 5.3, 0.0, 7.0),
                        (.2, 6.5, 5.3, 0.0, 7.0), (0.0, 6.32, 5.12, 0.0, 7.0), (0.0, 5.95, 4.55, .25, 7.0), (.75, 5.5, 4.2, .35, 7.0)], 560)
    n = len(rows)
    d, hw, hh, cz, p = rows.T
    ux, uz = np.zeros((n, m), np.float32), np.zeros((n, m), np.float32)
    for i in range(n):
        ux[i], uz[i] = squircle(m, float(p[i]))
    x, z = hw[:, None] * ux, cz[:, None] + hh[:, None] * uz
    y = np.broadcast_to(d[:, None], x.shape).astype(np.float32).copy()
    jj, ii = np.meshgrid(np.arange(m, dtype=np.float32) * 1024 / m, np.arange(n, dtype=np.float32) * 1024 / n)
    # vent slots across the back, moulded in
    slots = sm((np.sin(d[:, None] * 13.0) - .45) / .3) * sm((d[:, None] - 4.8) / .3) * sm((10.2 - d[:, None]) / .3) * (uz > -.35) * (np.abs(ux) < .94)
    inward = .05 * slots
    x, z = x * (1 - inward / (hw[:, None] + .01)), cz[:, None] + (z - cz[:, None]) * (1 - inward / (hh[:, None] + .01))
    big = lerp(spectral(1024, 3.2, seed), jj, ii)
    mid = lerp(spectral(1024, 2.8, seed + 1), jj * 3, ii * 2)
    fine = lerp(spectral(1024, 2.2, seed + 2), jj * 7, ii * 5)
    # paint: pale old plastic, blotched; darker in the seams
    alb = .84 + .05 * big + .03 * mid + .012 * fine
    alb = alb - .55 * slots
    recess = sm((d[:, None] - .02) / .5) * (np.arange(n)[:, None] > int(np.argmin(d)))
    alb = alb * (1 - .62 * recess)
    seam = np.exp(-((d[:, None] - 3.95) / .08) ** 2)
    alb = alb - .34 * seam
    # rain has run down from the top; mud up to here
    run = np.clip(lerp(spectral(1024, 2.0, seed + 3), jj * 6, ii * .25) - .25, 0, 2) * (z / 5.3 + .4)
    alb = alb - .10 * np.clip(run, 0, 1.4)
    mud = sm((-3.0 - z + 1.0 * big + .5 * mid) / 1.3)
    alb = alb * (1 - .48 * mud)
    speck = sm((lerp(spectral(1024, 1.4, seed + 4), jj * 9, ii * 7) - 1.7) / .3)
    alb = alb - .35 * speck
    z = z + .012 * mid + .003 * fine
    X, Y, Z = place(x, y, z)
    body = mesh_arrays("crt", X, Y, Z, np.clip(alb, .03, .95), wrap=True)
    case = attr_material("crt_case", rough=.6, spec=.35)
    body.data.materials.append(case)

    # the foot, swivelling on a neck, under the middle of the case
    foot = smooth_rows([(-6.65, .02, .02, 3.0), (-6.65, 4.3, 3.7, 3.6), (-6.25, 4.5, 3.9, 3.6), (-6.05, 3.3, 3.0, 3.0), (-5.25, 2.7, 2.5, 2.6)], 120, k=3)
    fx, fy = squircle(400, 3.4)
    Xf, Yf, Zf = place(foot[:, 1:2] * fx[None, :], 3.6 + foot[:, 2:3] * fy[None, :], np.broadcast_to(foot[:, :1], (120, 400)))
    f_ = mesh_arrays("crt_foot", Xf, Yf, Zf, np.full((120, 400), .36, np.float32), wrap=True)
    f_.data.materials.append(case)

    # the buttons along the bottom of the bezel
    for k, (bx, bw, bh) in enumerate(((2.2, .17, .12), (2.75, .17, .12), (3.3, .17, .12), (3.85, .17, .12), (4.75, .30, .22))):
        sq = smooth_rows([(.0, bw, bh, 5.0), (-.09, bw, bh, 5.0), (-.10, bw * .86, bh * .86, 5.0), (-.10, .01, .01, 5.0)], 24, k=1)
        ax, az = squircle(60, 5.0)
        Xb, Yb, Zb = place(bx + sq[:, 1:2] * ax[None, :], np.broadcast_to(sq[:, :1], (24, 60)), -4.66 + sq[:, 2:3] * az[None, :])
        o = mesh_arrays(f"btn{k}", Xb, Yb, Zb, np.full((24, 60), .52, np.float32), wrap=True)
        o.data.materials.append(case)

    # the glass: a shallow dome, the picture coming out of it, the sky faintly on it
    hws, hhs, czs = 5.6, 4.3, .35
    # the picture stops short of the tube's edge, with dark glass round it, and sits a little to the right: from here the left lip hides more of the glass
    page = tex("blog")
    ph_, pw_ = page.shape
    pw, ph, pc = 5.15, 3.86, .1
    Wp, Hp = int(round(pw_ * hws / pw)), int(round(ph_ * hhs / ph))
    x0, y0 = int(round((hws + pc - pw) / (2 * hws) * Wp)), (Hp - ph_) // 2
    Yp, Xp = np.mgrid[:Hp, :Wp].astype(np.float32)
    rc = 70.0
    dist = np.hypot(np.maximum(np.abs(Xp - x0 - pw_ / 2) - pw_ / 2 + rc, 0), np.maximum(np.abs(Yp - y0 - ph_ / 2) - ph_ / 2 + rc, 0)) - rc
    picture = np.full((Hp, Wp), .03, np.float32)
    picture[y0:y0 + ph_, x0:x0 + pw_] = page
    picture = .03 + (picture - .03) * sm(-dist / 7.0)
    nu, nv = 900, 680
    v_, u_ = np.mgrid[:nv, :nu].astype(np.float32)
    u_, v_ = u_ / (nu - 1), v_ / (nv - 1)

    def glass(u, v):
        gx, gz = (u - .5) * 2 * hws, (.5 - v) * 2 * hhs
        bulge = .36 * (1 - (gx / hws / 1.15) ** 2) * (1 - (gz / hhs / 1.15) ** 2)
        return gx, .78 - bulge, czs + gz

    # rows from the bottom of the picture, for the image's own way up
    gx, gy, gz = glass(u_, 1 - v_)
    Xg, Yg, Zg = place(gx, gy, gz)
    sc_ = mesh_grid("screen", Xg, Yg, Zg)
    mat = bpy.data.materials.new("screen")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    ti, em, gl, ad, out = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeBsdfGlossy"), nt.nodes.new("ShaderNodeAddShader"), nt.nodes.new("ShaderNodeOutputMaterial")
    ti.image = ink_image("blog", picture)
    ti.interpolation = "Cubic"
    gl.inputs["Color"].default_value = (.06, .06, .06, 1)
    gl.inputs["Roughness"].default_value = .06
    # to the eye the page stays readable; what is round it catches far more, so the screen is seen to be lit
    lp, sp = nt.nodes.new("ShaderNodeLightPath"), nt.nodes.new("ShaderNodeMapRange")
    sp.inputs["To Min"].default_value, sp.inputs["To Max"].default_value = 1.2, 7.0
    nt.links.new(lp.outputs["Is Diffuse Ray"], sp.inputs["Value"])
    nt.links.new(sp.outputs[0], em.inputs["Strength"])
    nt.links.new(ti.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], ad.inputs[0])
    nt.links.new(gl.outputs[0], ad.inputs[1])
    nt.links.new(ad.outputs[0], out.inputs["Surface"])
    sc_.data.materials.append(mat)
    sc_.visible_shadow = False
    return lambda u, v: place(*[float(a) for a in glass((x0 + u * pw_) / Wp, (y0 + v * ph_) / Hp)])


def shot_blog():
    begin()
    cam, target, lens = (-6.5, -23.0, 7.2), (6.0, 30.0, 4.6), 34.0
    hx, hy = -9.5, 2.5
    level = 0.0
    glyphs = [(glyph("难"), hx, hy, 2.4, 7.2, -.04, .98, 0)]
    rings = [(hx, hy, .007, .85, 5.4, 1.4), (hx, hy, .004, .7, 7.8, 1.2)]
    water_ink(cam, target, lens, rings=rings, glyphs=glyphs, res=.85 * RES, chop=.0014, rough=.015, seed=47)
    at = crt((7.0, 14.0, 7.35), -.14)
    # the spit from the slab's shot behind it; the monitor on a flat stone at the water's edge, others like it to either side
    land("spit", 14.0, 36.0, 44, 44, [(13, 31, 2.7, 6.5), (20, 37, 4.0, 7.0), (5, 37, 1.7, 5.0), (26, 44, 6.5, 9.0)], 351, n=1400, wavelength=5.0, strength=.09, base=-.5, tone=1.0)
    boulder("seat", 7.3, 17.2, 7.8, 5.4, .95, 353, n=900, tone=.72)
    boulder("seat_l", 1.2, 13.4, 3.6, 2.3, .55, 361, n=600, tone=.60)
    boulder("seat_r", 13.6, 13.2, 4.0, 2.5, .8, 365, n=600, tone=.66)
    u, v = tex("blog_nan")
    ex, ey, ez = at(float(u), float(v))
    flying_thread(np.array([hx + .1, hy + .9, level]), np.array([ex - .14 * .05, ey - .05, ez]), hairs=False)    # just in front of the glass
    far_range(cam, k=.012, s0=30.0, h=60.0, peaks=[(-120, 200, 74, 48), (-30, 260, 104, 60), (70, 240, 88, 50), (170, 220, 96, 58), (280, 300, 110, 70), (-230, 260, 92, 64)])
    world_split(.6, 1.15, zenith=.85)
    sun((.72, -.40, -.56), energy=2.2, angle=12)
    camera(cam, target, lens=lens)
    render(NAME)


# ---- the grey day -----------------------------------------------------------------------------------------------------------------------------------
def shot_grey():
    begin()
    cam, target, lens = (.6, -26.0, 11.0), (0.0, 10.0, 1.8), 30.0
    water_ink(cam, target, lens, res=.9 * RES, chop=.0012, rough=.015, seed=43)
    boulder("stone", -15.0, -5.0, 4.6, 3.0, 3.0, 321, n=800, tone=.6)
    land("bank", -8.0, 100.0, 170, 170, [(-46, 66, 9, 16), (-22, 86, 6, 13), (14, 92, 5, 11), (40, 72, 8, 13)], 341, n=1100, wavelength=7, strength=.12, base=-.6, mist=dict(cam=cam, k=.012, s0=30.0, H=30.0))
    far_range(cam, k=.009, s0=40.0, h=60.0, cy=380.0, peaks=[(-170, 280, 60, 56), (-70, 350, 74, 64), (30, 400, 88, 80), (140, 330, 66, 60), (240, 300, 58, 58)])
    world_split(.64, 1.15, zenith=.7)
    sun((.5, -.6, -.6), energy=1.2, angle=22)
    camera(cam, target, lens=lens)
    render(NAME)


# ---- two stones ---------------------------------------------------------------------------------------------------------------------------------
STONES = [(-9.6, 13.0, 6.5), (10.2, 14.8, 6.3)]


def shot_stones():
    begin()
    cam, target, lens = (.4, -30.0, 10.5), (0.0, 30.0, 1.2), 30.0
    X, Y = screen_plane(cam, target, lens, z=0.0, res=.95 * RES, far=700.0)
    # potential flow going +y past two circles. phi runs along the flow, psi across it: anything drawn in (psi, phi) follows the streamlines and crowds where the water hurries
    zeta = X.astype(np.complex64) + 1j * Y.astype(np.complex64)
    w_ = -1j * zeta
    for (xs, ys, R) in STONES:
        dz = zeta - complex(xs, ys)
        a = np.abs(dz)
        dz = dz * np.maximum(R * .8 / (a + 1e-3), 1.0)
        w_ = w_ + 1j * R * R / dz
    phi, psi = w_.real.astype(np.float32), w_.imag.astype(np.float32)
    run = T * 3.0
    far = 1 / (1 + (np.hypot(X - cam[0], Y - cam[1]) / 95.0) ** 2)
    # the ripples drawn out along the flow
    n1 = lerp(spectral(1024, 2.6, 71), psi * 60, (phi - run) * 2.4)
    n2 = lerp(spectral(1024, 2.2, 72), psi * 140, (phi - run) * 6.0)
    z = (.016 * n1 + .004 * n2) * far
    # the tongue: past the gap the fast water stands up in a train of V's, tailing off downstream
    xg, yg = .3, 15.0
    dy = np.maximum(Y - yg, 0)
    ax = np.abs(X - xg)
    env = np.exp(-(ax / (2.6 + .16 * dy)) ** 2) * np.exp(-dy / 30.0) * sm(dy / 5.0)
    n3 = lerp(spectral(1024, 3.0, 79), X * 7, Y * 7)
    z = z + .015 * env * (1 + .6 * n3) * np.sin((dy - 1.0 * ax - run * 2.0 + 1.1 * n3) * 6.2832 / 3.6) * far
    # the slack water behind each stone, turning; and the pile of water against its upstream face
    for k, (xs, ys, R) in enumerate(STONES):
        dx_, dy_ = X - xs, Y - ys
        behind = np.exp(-(dx_ / (R * .9)) ** 2) * sm(dy_ / R) * np.exp(-np.maximum(dy_, 0) / (R * 3.5))
        z = z + .014 * behind * lerp(spectral(1024, 2.0, 73 + k), X * 40, Y * 40) * far
        d_ = np.hypot(dx_ * .95, dy_)
        up = sm(-dy_ / R + .2)
        z = z + .022 * up * np.exp(-((d_ - R * 1.10) / (R * .10)) ** 2) * np.sin((d_ - R * 1.1) * 6.2832 / 1.3)
    # the fine lines a painter draws for moving water: the same streamlines, in ink, broken
    ln = lerp(spectral(1024, 3.2, 75), psi * 45, (phi - run) * 1.4)
    brk = sm((lerp(spectral(1024, 3.0, 76), psi * 10, (phi - run) * 3.0) + .15) / .35)
    C = .46 * np.exp(-(ln / .05) ** 2) * brk * far
    ln2 = lerp(spectral(1024, 3.2, 77), psi * 28, (phi - run) * .9)
    brk2 = sm((lerp(spectral(1024, 3.0, 78), psi * 7, (phi - run) * 2.0) - .1) / .35)
    C = np.maximum(C, .30 * np.exp(-(ln2 / .035) ** 2) * brk2 * far)
    o = mesh_arrays("river", X, Y, z, np.clip(C, 0, 1))
    o.data.materials.append(mirror_attr_material(tint=.9, rough=.018))
    boulder("left", STONES[0][0], STONES[0][1], 7.0, 5.2, 8.2, 381, n=1300, tone=.74)
    boulder("right", STONES[1][0], STONES[1][1], 6.8, 5.0, 7.8, 389, n=1300, tone=.70, carve=lambda X, Y, z: (9.0 * np.exp(-(np.hypot(X - 18.0, Y - 9.3) / .85) ** 4), 0.0))    # a nub the warp left off its corner
    # banks well back on either side, leaving the river's way open into the mist
    land("banks", 0.0, 190.0, 420, 420, [(-70, 66, 8, 26), (-52, 112, 11, 26), (-40, 170, 14, 30), (74, 72, 9, 28), (50, 120, 10, 26), (38, 178, 13, 30)], 391, n=1500, wavelength=9, strength=.11, base=-.6, mist=dict(cam=cam, k=.011, s0=40.0, H=34.0))
    far_range(cam, k=.012, s0=40.0, h=64.0, cy=470.0, peaks=[(-210, 330, 70, 66), (-96, 410, 88, 76), (120, 420, 92, 80), (240, 350, 72, 66), (0, 620, 100, 110)])
    world_split(.62, 1.15, zenith=.66)
    sun((.5, -.6, -.6), energy=1.3, angle=22)
    camera(cam, target, lens=lens)
    render(NAME)


# ---- 痛 ---------------------------------------------------------------------------------------------------------------------------------------------
HANDS = ("stele", "letter", "screen", "print", "copy", "pen")
# Each ring: radius, width of the characters in it, how strong and how soft they are, how many
BANDS = [(14.5, 3.0, .80, 0, 7), (21.0, 2.8, .64, .7, 11), (29.0, 2.7, .49, 1.4, 15), (38.5, 2.6, .36, 2.4, 19), (50.0, 2.6, .25, 3.6, 22), (64.0, 2.7, .16, 5.0, 24), (80.0, 2.8, .09, 7.0, 24)]


def shot_tong():
    begin()
    cam, target, lens = (0.0, -34.0, 16.0), (-2.0, 15.0, 0.0), 26.0
    cx, cy = -6.0, 10.8
    r = np.random.default_rng(29)
    masks = {h: tex("hand_" + h) for h in HANDS}
    soft = {}
    big = tex("tong_big")
    # the one written: nearly as long again as it is wide on the water, so it still reads from here, and a pale drag of itself going away from us
    glyphs = [(big, cx, cy, 8.4, 19.0, 0.0, .97, 0), (big, cx + .3, cy + 4.0, 9.0, 32.0, 0.0, .15, 16)]
    boxes = []
    order = list(HANDS)
    for k, (R, size, strength, bl, count) in enumerate(BANDS):
        a0 = r.uniform(0, 6.28)
        for i in range(count):
            a = a0 + 6.2832 * (i + r.uniform(-.2, .2)) / count
            x, y = cx + (R + r.uniform(-.7, .7)) * math.cos(a), cy + (R + r.uniform(-.7, .7)) * math.sin(a)
            sin_el = 16.0 / math.sqrt(x * x + (y + 34.0) ** 2 + 16.0 * 16.0)
            sy = size * min(.80 / sin_el, 3.6)
            px, py, _ = project(cam, target, lens, np.array([x, x - size / 2, x, x]), np.array([y, y, y - sy / 2, y + sy / 2]), np.zeros(4))
            hw, hh = abs(px[0] - px[1]) * 1.1, abs(py[3] - py[2]) * .56
            if any(abs(px[0] - u) < hw + w_ and abs(py[0] - v) < hh + h_ for u, v, w_, h_ in boxes) or not (-60 < px[0] < W + 60 and -40 < py[0] < H + 60):
                continue
            boxes.append((px[0], py[0], hw, hh))
            h = order[(i + k * 2) % len(order)]
            if bl not in soft:
                soft[bl] = {n: (blur(m, bl) if bl else m) for n, m in masks.items()}
            glyphs.append((soft[bl][h], x, y, size, sy, r.uniform(-.28, .28), strength * r.uniform(.85, 1.1), 0))
    # the waves between the rings of characters: real, so they catch the sky, and the characters on them are pushed in and out
    waves = [(5.5, 1.5, .009)] + [(R + d, 1.5 + .07 * R, .0095 - .00007 * R) for R, *_ in BANDS[:5] for d in (-3.4, 3.2)]
    rings = [(cx, cy, a, sp, R, sp * 1.5) for R, sp, a in waves]

    def warp(X, Y):
        d = np.hypot(X - cx, Y - cy) + 1e-3
        push = np.zeros_like(X)
        for R, sp, a in waves:
            near = np.abs(d - R) < sp * 4
            push[near] += (.62 * (1.0 if R < 8 else .5)) * np.sin((d[near] - R) * 6.2832 / sp) * np.exp(-((d[near] - R) / (sp * 1.5)) ** 2)
        return push * (X - cx) / d, push * (Y - cy) / d

    water_ink(cam, target, lens, rings=rings, glyphs=glyphs, res=.95 * RES, chop=.0009, rough=.012, seed=51, warp=warp)
    # a wet stone in the near corner, a low bank across the water, mountains behind it with the cloud on them
    boulder("stone", -20.5, -9.0, 6.4, 4.2, 4.2, 361, n=900, tone=.62)
    land("bank", 8.0, 170.0, 280, 280, [(-98, 110, 9, 22), (-52, 124, 7, 20), (14, 134, 5, 18), (70, 118, 8, 22), (120, 112, 10, 24)], 381, n=1100, wavelength=8, strength=.12, base=-.6, mist=dict(cam=cam, k=.013, s0=40.0, H=30.0))
    far_range(cam, k=.012, s0=40.0, h=60.0, cy=430.0, peaks=[(-180, 300, 70, 60), (-70, 380, 88, 70), (34, 430, 98, 84), (150, 360, 76, 64), (260, 320, 66, 60)])
    world_split(.62, 1.15, zenith=.66)
    sun((.4, -.7, -.6), energy=1.3, angle=22)
    camera(cam, target, lens=lens)
    render(NAME)


# ---- under the water -----------------------------------------------------------------------------------------------------------------------------
_tilt = math.radians(50)
UNDER_CAM = ((0.0, -5.0, -9.0), (0.0, -5.0 + math.sin(_tilt) * 10, -9.0 + math.cos(_tilt) * 10), 22.0)


def shot_under():
    begin()
    glass_bounces()
    sc = bpy.context.scene
    cam, target, lens = UNDER_CAM
    # the surface, seen from below: long swells and short chop, so the moon breaks into pieces
    n = int(1900 * max(RES, .6))
    size = 110.0
    y, x = np.mgrid[:n, :n].astype(np.float32)
    X, Y = (x / (n - 1) - .5) * size, (y / (n - 1) - .5) * size + 18
    ph = T * 1.6
    z = np.zeros_like(X)
    r = np.random.default_rng(23)
    for i in range(64):
        wl = float(r.uniform(.5, 7.0))
        ang = r.uniform(0, 6.28)
        a = .013 * wl ** .9 * float(r.uniform(.5, 1.2)) * (1 - .965 * sm(float(OPT.get("settle", 0))))
        k = 2 * math.pi / wl
        z += a * np.sin(k * (X * math.cos(ang) + Y * math.sin(ang)) + float(r.uniform(0, 6.28)) + ph * math.sqrt(k) * 1.2)
    o = mesh_grid("surface", X, Y, z)
    o.data.materials.append(glass_material("under_surface", rough=.0))
    # the moon and the night sky above it; below the surface the world is the dark of deep water
    moon_z, moon_az = math.radians(55), math.radians(-22)
    D = 300.0
    bpy.ops.mesh.primitive_uv_sphere_add(radius=36.0, location=(D * math.sin(moon_z) * math.sin(moon_az), D * math.sin(moon_z) * math.cos(moon_az), D * math.cos(moon_z)), segments=64, ring_count=32)
    mo = bpy.context.object
    me_ = bpy.data.materials.new("moon")
    me_.use_nodes = True
    nt = me_.node_tree
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 6.0
    nt.links.new(em.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    mo.data.materials.append(me_)
    nw = sc.world.node_tree
    bg = nw.nodes["Background"]
    tc, sp, mr = nw.nodes.new("ShaderNodeTexCoord"), nw.nodes.new("ShaderNodeSeparateXYZ"), nw.nodes.new("ShaderNodeMapRange")
    nw.links.new(tc.outputs["Generated"], sp.inputs[0])
    nw.links.new(sp.outputs["Z"], mr.inputs["Value"])
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = -.7, .25
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = .30, .78
    nw.links.new(mr.outputs[0], bg.inputs[1])
    camera(cam, target, lens=lens)
    render(NAME)


def shot_under_cloud():
    """The ink from chapter 3's simulation, the same mass, on nothing: its alpha decides how much of the light behind it gets through. The grade lays it over the surface."""
    sc = reset(samples=SAMPLES, pct=PCT)
    sc.cycles.volume_bounces = 2
    sc.cycles.max_bounces = 4
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    sc.world.node_tree.nodes["Background"].inputs[1].default_value = .10
    sun((-.2, .5, -.84), energy=3.5, angle=10)
    cam, target, lens = UNDER_CAM
    frame = int(OPT.get("frame", 30))
    s_ = float(OPT.get("scale", .95))
    root = os.path.expanduser("~/render_work/cache/ink3d_cloud_200")
    v = bpy.data.volumes.new("ink")
    v.filepath = os.path.join(root, "data", f"fluid_data_{frame:04d}.vdb")
    o = bpy.data.objects.new("ink", v)
    bpy.context.collection.objects.link(o)
    o.location = (float(OPT.get("x", -8.5)) * s_ + 1.0, 4.0 - 2.5 * s_, -3.0 - 5.0 * s_)
    o.scale = (s_, s_, s_)
    m = bpy.data.materials.new("ink")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    a, p, c, out = nt.nodes.new("ShaderNodeVolumeInfo"), nt.nodes.new("ShaderNodeVolumePrincipled"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeOutputMaterial")
    p.inputs["Color"].default_value = (.5, .5, .5, 1)
    p.inputs["Absorption Color"].default_value = (0, 0, 0, 1)
    c.operation = "MULTIPLY"
    c.inputs[1].default_value = float(OPT.get("dens", 8.0))
    nt.links.new(a.outputs["Density"], c.inputs[0])
    nt.links.new(c.outputs[0], p.inputs["Density"])
    nt.links.new(p.outputs["Volume"], out.inputs["Volume"])
    o.data.materials.append(m)
    camera(cam, target, lens=lens)
    render(NAME, depth=False)


{"rain": shot_rain, "wonan": shot_wonan, "thread": shot_thread, "tong": shot_tong, "under": shot_under, "under_cloud": shot_under_cloud, "grey": shot_grey, "blog": shot_blog, "stones": shot_stones}[SHOT]()
