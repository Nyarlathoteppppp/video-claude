"""二 · 依我之见: the river flown down through three thousand years, the writing on its banks going into the water, and the falls. Storyboard ink3d.scenes.md.

    Blender -b --factory-startup -P ink3d_ages.py -- --shot=NAME [--t=SECONDS] [--pct=100] [--samples=64]
    Blender -b --factory-startup -P ink3d_ages.py -- --shot=NAME --frames=A-B [--resume=1]     # the shot itself: ages/seq/NAME_NNN.png and _z.png; or --frames=0,20,40 to test a path
    python ink3d_ages_prep.py all                       # first: everything with writing on it (scipy and fonts live outside Blender); what it says is in ink_ages_text.py
    python ink_ages.py grade NAME                       # then: each frame in ink and in colour

The chapter is one fast flight downstream, low over the water: the camera never cuts, the things on the bank go by and each is held for about a second.
Time t runs from 0 within each shot.

bone    Takes over from the source: the fog opens downstream and the first thing on the bank, at grey dawn, is a tortoise shell, cracked by fire and cut with a real divination about rain, and then a roll of bamboo slips.
        The ink in the cuts runs off each and across the sand into the river. Moves: the camera passes them close on their right. Hands on by racing downstream on the plume.
stele   Full day and mist. A forest of steles along the bank, the nearest cut with the Analects in tiers, its lower lines full of ink and running. Moves: threads past the row as it recedes.
screen  Night again. On the bank a printed book, then a letter, then the last thing, a laptop, open, the only light for miles, showing a web page. The writing on each is running, and off its foot into the river. Moves: comes in low over the water
        with the reflection breaking under the lens, and on into the page. Hands on in its words, to a river wide and black and full of them.
falls   Low over the river, black and full of words, going the way it goes, to a horseshoe lip; the white water goes down on both hands, the words pulled long on it; below, a black pool
        turns round one hole and carries them in. Moves: comes out over the lip and sinks toward the hole, which turns slowly under the lens. Hands on to 三: the hole closes over the lens and the dark is the inside of the numbers.

Every shot is one build, and then a lens going forward at an even pace from its first frame to its last (fly below). The stills are what --frames leaves out: the same build from the first camera.
"""
import json
import math
import os
import sys
import time

from mathutils import Euler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

from ink3d import (OUT3D, aim, blur, camera, depth_material, erode, grey_material, image_from, ink_albedo, lerp, massif, mesh_grid, painted_material, render, reset, sm, spectral, water_material, world_light)
from ink3d_myth import (instances_mesh, mist_sheet, opt, paint, shine, shoot, tube_mesh)
from ink_ages_clock import count

TEX = os.path.join(OUT3D, "ages", "tex")
RAW = os.path.join(OUT3D, "ages", "raw")
SEQ = os.path.join(OUT3D, "ages", "seq")
WATER_Z = .22


# ---- the flight ----------------------------------------------------------------------------------------------------------------------
def curve(keys, n=6000, even=True):
    """A smooth line through the keys, read by how far along it you are: an even clock is an even pace, from the first frame. even=False: the keys come at equal times, for where the lens looks."""
    P = np.array(keys, float)
    Q = np.r_[[2 * P[0] - P[1]], P, [2 * P[-1] - P[-2]]]
    k = len(P) - 1
    s = np.linspace(0, k, n)
    i = np.minimum(s.astype(int), k - 1)
    f = (s - i)[:, None]
    p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
    pts = .5 * (2 * p1 + (p2 - p0) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f ** 2 + (3 * p1 - p0 - 3 * p2 + p3) * f ** 3)
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if not even:
        L_ = L[-1]
        L = np.linspace(0, L_, n)

    def at(u):
        return np.array([np.interp(u * L[-1], L, pts[:, j]) for j in range(3)])
    at.length = L[-1]
    return at


def wanted(N):
    """--frames=A-B, a list of frames, or one."""
    v = opt("frames", "")
    if "," in v:
        return [int(x) for x in v.split(",")]
    if "-" in v:
        a, b = v.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v)] if v else list(range(N))


def fly(shot, co, hidden, eye, look, each=None, roll=None, focus=None):
    """Without --frames the still, from where the lens is. With it the shot: eye is keys for curve(), look keys at equal times, each(n, t, eye) moves what else moves, and the shutter stays open across a quarter of a frame, so that writing going by can still be read.
    --pass=depth makes the same frames' depth instead: a run of its own, since swapping every material in and out for each frame costs more than the picture does."""
    if not opt("frames", ""):
        for o in hidden:
            face(o, co.location)
        shoot(os.path.join(RAW, shot), hidden)
        return
    sc = bpy.context.scene
    N = count(shot)
    eye_, look_ = curve(eye), curve(look, even=False)
    print(f"{shot}: {N} frames, {eye_.length:.0f} units, {eye_.length / (N - 1) * 30:.1f} a second", flush=True)
    os.makedirs(SEQ, exist_ok=True)
    with open(os.path.join(SEQ, shot + "_path.json"), "w") as f:
        json.dump([dict(eye=list(eye_(n / (N - 1))), look=list(look_(n / (N - 1)))) for n in range(N)], f)
    sc.render.use_persistent_data = True
    for k in opt("hide", "").split(","):       # to see what is making what
        if k:
            bpy.data.objects[k].hide_render = True
    deep = opt("pass", "beauty") == "depth"
    if deep:
        for o in hidden:
            o.hide_render = True
        bpy.context.view_layer.material_override = depth_material()
        sc.cycles.samples, sc.cycles.use_denoising, sc.view_settings.view_transform = 2, False, "Raw"
        co.data.dof.use_dof = False
    else:
        sc.render.use_motion_blur = opt("blur", 1) > 0
        sc.render.motion_blur_shutter = opt("shutter", .25)
    sc.frame_start, sc.frame_end = 1, 3
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    last = Euler(co.rotation_euler)
    for n in wanted(N):
        name = os.path.join(SEQ, f"{shot}_{n:03d}")
        target = name + ("_z.png" if deep else ".png")
        if opt("resume", 0) and os.path.exists(target):
            continue
        t0 = time.time()
        if os.path.exists(target):
            os.replace(target, target + ".old")
        u = n / (N - 1)
        here = eye_(u)
        for f, m in ((1, n - 1), (2, n), (3, n + 1)):
            um = min(max(m / (N - 1), 0), 1)
            aim(co, tuple(eye_(um)), tuple(look_(um)), roll(um) if roll else 0.0)
            co.rotation_euler.make_compatible(last)
            last = Euler(co.rotation_euler)
            co.keyframe_insert("location", frame=f)
            co.keyframe_insert("rotation_euler", frame=f)
        if not deep:
            if focus:
                focus(co.data.dof, u, here)
            for o in hidden:
                face(o, here)
            if each:
                each(n, n / 30.0, here)
        sc.frame_set(2)
        if deep:
            sc.render.filepath = target
            bpy.ops.render.render(write_still=True)
        else:
            render(name, depth=False)
        print(f"frame {n} {time.time() - t0:.1f}", flush=True)


def standing(name, center, width, height, value=.9, density=1.0, seed=0, top=.5, grain=3.2, thin=None):
    """A bank of mist that keeps turning to the lens as it goes by. thin: how near it starts to go, and how near it has gone, for one the lens goes through."""
    o = mist_sheet(name, (0, 0, 0), width, height, (0, -1, 0), value=value, density=density, seed=seed, top=top, grain=grain)
    o.location = center
    o["thin"] = thin or 0
    if thin:
        nt = o.data.materials[0].node_tree
        mx = next(k for k in nt.nodes if k.bl_idname == "ShaderNodeMixShader")
        ti = next(k for k in nt.nodes if k.bl_idname == "ShaderNodeTexImage")
        ma = nt.nodes.new("ShaderNodeMath")
        ma.operation = "MULTIPLY"
        ma.name = "thin"
        nt.links.new(ti.outputs[0], ma.inputs[0])
        nt.links.new(ma.outputs[0], mx.inputs[0])
    return o


def lying(name, center, width, depth, value=.9, density=1.0, seed=0, grain=3.2):
    """A sheet of mist lying flat over the water, for a lens that looks down."""
    o = mist_sheet(name, (0, 0, 0), width, depth, (0, -1, 0), value=value, density=density, seed=seed, top=.2, grain=grain)
    o.location = center
    o.rotation_euler = (math.pi / 2, 0, 0)
    o["flat"] = 1
    return o


def face(o, eye):
    """Turns a bank of mist to the lens, about the upright."""
    if o.get("flat"):
        return
    c = np.array(o.location, float)
    f = c - np.array(eye, float)
    o.rotation_euler = (0, 0, math.atan2(-f[0], f[1]))
    if o.get("thin"):
        go = float(sm((np.linalg.norm(f) - o["thin"] * .3) / (o["thin"] * .7)))
        o.data.materials[0].node_tree.nodes["thin"].inputs[1].default_value = go


def maths(nt):
    """Arithmetic on a shader's sockets."""
    def m(op, a, b=None):
        k = nt.nodes.new("ShaderNodeMath")
        k.operation = op
        for i, v in enumerate((a, b)):
            if isinstance(v, (int, float)):
                k.inputs[i].default_value = v
            elif v is not None:
                nt.links.new(v, k.inputs[i])
        return k.outputs[0]
    return m


def paint_run(name, albedo, rough, trail, arrive, glow=0.0, dark=.8):
    """paint(), with ink that runs: trail is where it goes and arrive when it gets there, from 0 to 1; dark how much of what is under it still shows. glow: how much the thing shows of itself in the dark. Returns the material and the value to set each frame, how far the ink has come."""
    mt = paint(name, albedo, rough)
    nt = mt.node_tree
    b = nt.nodes["Principled BSDF"]
    m = maths(nt)
    ta, tr = b.inputs["Base Color"].links[0].from_node, b.inputs["Roughness"].links[0].from_node
    tw, tt, front = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeValue")
    tw.image, tt.image = image_from(trail, name + "_w"), image_from(arrive, name + "_t")
    wet = m("MULTIPLY", m("MULTIPLY", tw.outputs[0], dark), m("MINIMUM", m("MAXIMUM", m("ADD", m("DIVIDE", m("SUBTRACT", front.outputs[0], tt.outputs[0]), .03), .5), 0.0), 1.0))
    keep = m("SUBTRACT", 1.0, wet)
    colour = m("ADD", m("MULTIPLY", ta.outputs[0], keep), m("MULTIPLY", wet, .012))
    nt.links.new(colour, b.inputs["Base Color"])
    nt.links.new(m("ADD", m("MULTIPLY", tr.outputs[0], keep), m("MULTIPLY", wet, .14)), b.inputs["Roughness"])
    if glow:
        nt.links.new(colour, b.inputs["Emission Color"])
        b.inputs["Emission Strength"].default_value = glow
    return mt, front.outputs[0]


def leaning(name, foot, head, width, shape=None, nu=300, nv=300):
    """A sheet laid on a slope from foot to head, its face to the lens that comes up from the foot: columns go left to right as the lens sees them, rows from foot to head. shape(a, b): how far it stands off the slope, across and along it, in units.
    Returns the object and the slope's frame."""
    F, H = np.array(foot, float), np.array(head, float)
    L = float(np.linalg.norm(H - F))
    B = (H - F) / L
    A = np.cross(B, (0, 0, 1))
    A = A / np.linalg.norm(A)
    N = np.cross(A, B)
    gv, gu = np.mgrid[:nv, :nu].astype(np.float32)
    a, b = (gu / (nu - 1) - .5) * width, gv / (nv - 1) * L
    off = shape(a, b) if shape else np.zeros_like(a)
    P = [(F[i] + a * A[i] + b * B[i] + off * N[i]).astype(np.float32) for i in range(3)]
    return mesh_grid(name, P[0], P[1], P[2]), (F, A, B, N, L)


def under(name, frame, width, material, thick=.9, gain=(1.4, 1.4), lift=.06):
    """A slab of stone under a sheet that leans, along the same slope."""
    from mathutils import Matrix
    F, A, B, N, L = frame
    o = rbox(name, (width + gain[0], L + gain[1], thick), material, .18)
    M_ = Matrix((A, B, N)).transposed().to_4x4()
    M_.translation = tuple(F + B * L / 2 - N * (thick / 2 + lift))
    o.matrix_world = M_
    return o


def falling(name, frame, width, seed, count=6, reach=2.6):
    """Ink going off the foot of what leans, down the stone and into the water."""
    F, A, B, N, L = frame
    r = np.random.default_rng(seed)
    segs = []
    for q in r.uniform(-.42, .42, count) * width:
        p = F + A * q + N * .1
        rad = r.uniform(.03, .08)
        for i in range(5):
            nx_ = p - B * reach / 5 + np.array((r.normal(0, .04), r.normal(0, .04), -.02))
            nx_[2] = max(nx_[2], WATER_Z)
            segs.append((p, nx_, rad, rad * .96))
            p = nx_
    o = tube_mesh(segs, name, sides=6)
    o.data.materials.append(painted_ink())
    return o


def river_skin(lines=.05, near=0.0):
    """What a lens going fast over still water needs to see go by: long lines of the current, moving, and if near is given a swell close under the lens that breaks what the water reflects.
    Returns the thing to set each frame: how far the water has gone."""
    mt = bpy.data.materials["water"]
    nt = mt.node_tree
    b, first = nt.nodes["Principled BSDF"], nt.nodes["Bump"]
    m = maths(nt)
    tc, geo = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeCameraData")
    mp, no, bu = nt.nodes.new("ShaderNodeMapping"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeBump")
    mp.inputs["Scale"].default_value = (1.0, .07, 1.0)
    no.inputs["Scale"].default_value = 1.3
    no.inputs["Detail"].default_value = 3
    bu.inputs["Strength"].default_value = lines
    bu.inputs["Distance"].default_value = .8
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs[0], no.inputs["Vector"])
    nt.links.new(no.outputs[0], bu.inputs["Height"])
    nt.links.new(first.outputs[0], bu.inputs["Normal"])
    last = bu
    swell = None
    if near:
        swell, n2, b2 = nt.nodes.new("ShaderNodeMapping"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeBump")
        swell.inputs["Scale"].default_value = (.8, .35, 1.0)
        n2.inputs["Scale"].default_value = 1.1
        n2.inputs["Detail"].default_value = 4
        b2.inputs["Distance"].default_value = 1.2
        nt.links.new(tc.outputs["Object"], swell.inputs["Vector"])
        nt.links.new(swell.outputs[0], n2.inputs["Vector"])
        nt.links.new(n2.outputs[0], b2.inputs["Height"])
        nt.links.new(m("MULTIPLY", near, m("POWER", 2.718, m("DIVIDE", geo.outputs["View Distance"], -13.0))), b2.inputs["Strength"])
        nt.links.new(bu.outputs[0], b2.inputs["Normal"])
        last = b2
    nt.links.new(last.outputs[0], b.inputs["Normal"])

    def went(y):
        mp.inputs["Location"].default_value = (0, -.07 * y, 0)
        if swell:
            swell.inputs["Location"].default_value = (0, -.35 * y, 0)
        for socket, length in _plumes:
            socket.default_value = 13 * y / length
    return went


def river_x(Y):
    return 12 * np.sin(Y / 46 + .4) + 5 * np.sin(Y / 19 + 2)


def river_w(Y):
    return 9 + .03 * (Y + 60)


def bank(d, w):
    return .55 * sm((d - w) / 4)


def valley(peaks, seed, size=640.0, N=2000, yc=190.0, dark=1.0, sink=None, rough=.16):
    """The land either side of the river: the same recipe as the look test, so every bank is in the same world.
    sink: (x0, x1, y0, y1) of the stretch a near_bank will cover, pressed down out of its way."""
    y, x = np.mgrid[:N, :N].astype(np.float32)
    X, Y = (x / (N - 1) - .5) * size, (y / (N - 1) - .5) * size + yc
    h = massif(X, Y, peaks, seed)
    d, w = np.abs(X - river_x(Y)), river_w(Y)
    h = h * sm((d - w) / 26) ** .7
    h = erode(h, size / N, seed + 1, wavelength=36, octaves=7, gain=.5, strength=.25)
    h = np.maximum(h, 0) + bank(d, w)
    h -= .5 * (1 - sm((d - w * .8) / 2.2))
    if sink is not None:
        x0, x1, y0, y1 = sink
        h -= 1.6 * sm((X - x0) / 2) * sm((x1 - X) / 2) * sm((Y - y0) / 2) * sm((y1 - Y) / 2)
    g = mesh_grid("land", X, Y, h)
    g.data.materials.append(painted_material("rock", ink_albedo(h, seed + 2, size / N) * dark))
    wx, wy = np.meshgrid(np.linspace(-size, size, 2), np.linspace(-size / 2, size * 1.5, 2))
    water = mesh_grid("water", wx.astype(np.float32), wy.astype(np.float32), np.full((2, 2), WATER_Z, np.float32))
    water.data.materials.append(water_material(rough=rough, normal_scale=36, normal_strength=opt("ripple", .05)))
    return g


def poly_distance(X, Y, pts):
    """Distance from each point to a polyline, and how far along it (0-1) the nearest point lies."""
    best, along = np.full(X.shape, 1e9, np.float32), np.zeros(X.shape, np.float32)
    k = len(pts) - 1
    for i in range(k):
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        t = np.clip(((X - ax) * (bx - ax) + (Y - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2 + 1e-9), 0, 1)
        d = np.hypot(X - ax - t * (bx - ax), Y - ay - t * (by - ay))
        c = d < best
        best = np.where(c, d, best)
        along = np.where(c, (i + t) / k, along)
    return best, along


def near_bank(x, y, cell, seed, runs=(), bumps=(), alb=.56, rock=.42):
    """The stretch of bank under the prop, fine enough to hold a thread of ink. runs: polylines with (width at start, at end). bumps: (x, y, height, radius)."""
    nx, ny = int((x[1] - x[0]) / cell), int((y[1] - y[0]) / cell)
    size = max(nx, ny)
    gy, gx = np.mgrid[:ny, :nx].astype(np.float32)
    X, Y = x[0] + gx * cell, y[0] + gy * cell
    sc_x, sc_y = gx * 1024 / size, gy * 1024 / size
    d, w = np.abs(X - river_x(Y)), river_w(Y)
    h = bank(d, w) + .03
    h -= .5 * (1 - sm((d - w * .8) / 2.2))
    h = h + .05 * lerp(spectral(1024, 3.0, seed), sc_x, sc_y) + .010 * lerp(spectral(1024, 2.1, seed + 1), sc_x, sc_y)
    h += .006 * np.sin((Y + X * .4 + 1.4 * lerp(spectral(1024, 3.3, seed + 2), sc_x, sc_y)) * 4.5)
    stone = np.zeros_like(h)
    craggy = lerp(spectral(1024, 3.6, seed + 6), sc_x, sc_y)
    for bx, by, top, rad in bumps:
        s_ = np.exp(-(np.hypot(X - bx, Y - by) / rad) ** 2)
        h = h + top * s_ * (1 + .22 * craggy)
        stone = np.maximum(stone, s_)
    a = alb + .05 * lerp(spectral(1024, 2.8, seed + 3), sc_x, sc_y) + .012 * lerp(spectral(1024, 1.3, seed + 4), sc_x, sc_y)
    wet = np.zeros_like(h)
    for pts, w0, w1 in runs:
        dist, along = poly_distance(X, Y, pts)
        wiggle = 1 + .4 * lerp(spectral(1024, 2.6, seed + 5), sc_x, sc_y)
        wid = (w0 + (w1 - w0) * along) * np.clip(wiggle, .3, 1.8)
        core = sm((wid - dist) / (.45 * wid + 1e-3))
        h = h - .05 * core
        wet = np.maximum(wet, core)
        a = a * (1 - .3 * sm((wid * 3.2 - dist) / (2.2 * wid)))        # the sand on either side has drunk some
    a = a * (1 - rock * sm((stone - .12) / .3) * np.clip(1 + .6 * craggy, .3, 1.4))        # what stands up out of the sand is stone
    # the edge of the water darkens the sand
    a = a * (1 - .45 * sm((.28 - h) / .25))
    a = a * (1 - wet) + .012 * wet
    r = .95 * (1 - wet) + .12 * wet
    g = mesh_grid("near", X, Y, h)
    g.data.materials.append(paint("sand", np.clip(a, .008, .95).astype(np.float32), r.astype(np.float32)))
    return g


_plumes = []


def plume(name, start, length, seed, wide=5.0, reach=.8):
    """Ink entering the river and drawn out downstream: a black veil on the water, widening and thinning, with thicker clots of it going down. It lies just above the water plane."""
    n, m = 1024, 768
    gy, gx = np.mgrid[:n, :m].astype(np.float32)
    v, u = gy / (n - 1), gx / (m - 1) * 2 - 1
    warp = lerp(spectral(1024, 2.7, seed), gx, gy) * .16 + lerp(spectral(1024, 2.0, seed + 1), gx, gy) * .03
    centre = .28 * np.sin(v * 3.4 + seed) * v
    width = .06 + .55 * v ** .8
    a = sm((width - np.abs(u - centre + warp)) / (width * .65)) * (1 - sm((v - reach) / (1 - reach + .01)))
    # in long threads that open out as it goes
    across = (u - centre + warp) / width
    threads = lerp(spectral(1024, 2.3, seed + 2), across * 240 + 512, gy * .07)
    a = a * (.95 - .5 * v) * np.clip(.3 + .95 * sm((threads + .25) / .8), 0, 1)
    sx, sy = start
    xs = np.linspace(sx - wide, sx + wide, m)[None, :].repeat(n, 0)
    ys = np.linspace(sy, sy + length, n)[:, None].repeat(m, 1)
    # the river bends: the veil follows it
    xs = xs + (river_x(ys) - river_x(np.full_like(ys, sy)))
    g = mesh_grid(name, xs.astype(np.float32), ys.astype(np.float32), np.full((n, m), WATER_Z + .012, np.float32))
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    mt.blend_method = "BLEND"
    nt = mt.node_tree
    bs = nt.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (.008, .008, .008, 1)
    bs.inputs["Roughness"].default_value = .12
    ti = nt.nodes.new("ShaderNodeTexImage")
    ti.image = image_from(np.clip(a, 0, 1).astype(np.float32), name + "_a")
    # the clots: a second picture that slides down the first
    clot = np.clip(.62 + .55 * lerp(spectral(1024, 2.5, seed + 3), gx * 1.1, gy * .22), .2, 1)
    tc, sep, comb, flow, tk = (nt.nodes.new(k) for k in ("ShaderNodeTexCoord", "ShaderNodeSeparateXYZ", "ShaderNodeCombineXYZ", "ShaderNodeValue", "ShaderNodeTexImage"))
    m = maths(nt)
    nt.links.new(tc.outputs["UV"], sep.inputs[0])
    nt.links.new(sep.outputs["X"], comb.inputs["X"])
    nt.links.new(m("SUBTRACT", sep.outputs["Y"], flow.outputs[0]), comb.inputs["Y"])
    nt.links.new(comb.outputs[0], tk.inputs[0])
    tk.image = image_from(clot.astype(np.float32), name + "_k")
    tk.extension = "EXTEND"
    nt.links.new(m("MULTIPLY", ti.outputs[0], tk.outputs[0]), bs.inputs["Alpha"])
    g.data.materials.append(mt)
    g.visible_shadow = False
    _plumes.append((flow.outputs[0], length))
    return g


def rbox(name, size, material, bevel=.05):
    """A box centred on the origin with its edges rounded."""
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel > 0:
        b = o.modifiers.new("b", "BEVEL")
        b.width, b.segments, b.limit_method = bevel, 3, "ANGLE"
    bpy.ops.object.shade_smooth()
    o.data.materials.append(material)
    return o


# ---- the shell --------------------------------------------------------------------------------------------------------------------------
def shot_bone(t=0.0):
    PEAKS = [(-60, 90, 58, 26), (-92, 40, 36, 22), (-34, 150, 78, 34), (66, 80, 60, 28), (108, 30, 40, 24), (34, 190, 84, 42), (-110, 170, 70, 40), (120, 170, 70, 42), (0, 330, 92, 70), (-60, -20, 16, 13)]
    valley(PEAKS, 201, dark=opt("dark", .55), sink=(-40, 6, -24, 24))
    O = np.array((-15.0, -8.0, .5))
    tilt, yaw = math.radians(opt("tilt", 56.0)), math.radians(14.0)
    # the shell, leaning on a stone with its head up
    h, a, ro, ma, wet_, when = (np.load(os.path.join(TEX, f"bone_{k}.npy")) for k in "harmwt")
    n, m = h.shape
    cell = .0075
    gy, gx = np.mgrid[:n, :m].astype(np.float32)
    lx, ly = (gx - m / 2) * cell, gy * cell
    W, L, Nrm = np.array((math.cos(yaw), math.sin(yaw), 0)), np.array((-math.sin(yaw) * math.cos(tilt), math.cos(yaw) * math.cos(tilt), math.sin(tilt))), None
    Nrm = np.cross(W, L)
    P = [O[i] + lx * W[i] + ly * L[i] + h * Nrm[i] for i in range(3)]
    gone = blur(ma, 5) < .002
    P[2] = np.where(gone, P[2] - 40, P[2])          # what is not shell goes out of the way of the depth pass
    shell = mesh_grid("shell", P[0], P[1], P[2])
    mt, shell_ink = paint_run("shell", a, ro, wet_, when, dark=.7)
    nt = mt.node_tree
    tm = nt.nodes.new("ShaderNodeTexImage")
    tm.image = image_from(ma, "shell_m")
    nt.links.new(tm.outputs[0], nt.nodes["Principled BSDF"].inputs["Alpha"])
    shell.data.materials.append(mt)

    foot = O + np.array((0, 0, 0))
    run = [(foot[0] - 1.4, foot[1] - .3), (foot[0] + .5, foot[1] - 2.2), (foot[0] + 3.4, foot[1] - 1.8), (foot[0] + 6.0, foot[1] - 3.6), (foot[0] + 9.2, foot[1] - 3.0), (foot[0] + 13.0, foot[1] - 4.5)]
    run3 = [(foot[0] - 3.2, foot[1] - .2), (foot[0] - 1.6, foot[1] - 1.5), (foot[0] + .3, foot[1] - 2.1)]
    # further on, a roll of bamboo slips laid open against a boulder, its foot at the water
    sa, sr, sa_, sw, st = (np.load(os.path.join(TEX, f"slips_{k}.npy")) for k in "armwt")
    Ws = sa.shape[1] / 180
    per_slip = .68

    def sag(a_, b_):
        s_ = (a_ / per_slip) % 1
        return .05 * np.sqrt(np.clip(1 - (2 * s_ - 1) ** 2, 0, 1)) + .05 * np.sin(b_ * .6 + a_ * .5) + .03 * np.sin(a_ * 1.7 + 1)
    slips, slope = leaning("slips", (-8.0, 3.6, .62), (-9.0, 12.1, 8.2), Ws, shape=sag, nu=520, nv=240)
    smt, slip_ink = paint_run("slips", sa, sr, sw, st)
    tm = smt.node_tree.nodes.new("ShaderNodeTexImage")
    tm.image = image_from(sa_, "slips_m")
    smt.node_tree.links.new(tm.outputs[0], smt.node_tree.nodes["Principled BSDF"].inputs["Alpha"])
    slips.data.materials.append(smt)
    under("slips_rest", slope, Ws, hewn_stone("slips_stone", .34))
    slip_run = [(-7.4, 3.3), (-5.8, 2.2), (-3.8, 1.2), (-1.0, 1.4)]
    near_bank((-40, 6), (-24, 24), .022, 211, runs=[(run, .34, .95), (run3, .18, .3), (slip_run, .2, .5)],
              bumps=[(O[0] - 1.5, O[1] + 8.5, 6.2, 3.3), (O[0] - 7, O[1] + 9, 3.4, 4), (-9.5, 15.5, 7.5, 6.5)], alb=.5, rock=opt("rock", .36))
    plume("plume", (foot[0] + 13.0, foot[1] - 4.5), 150, 217)

    world_light(opt("world", .34))
    shine((.72, .46, -.52), energy=opt("sun", 2.3), angle=6.0)
    cam = (-7.0, -28.0, 6.4)
    co = camera(cam, (-10.5, 0.0, 3.9), lens=28, far=1200)
    co.data.dof.use_dof = True
    co.data.dof.focus_distance = 20.0
    co.data.dof.aperture_fstop = opt("fstop", .9)
    extras = [standing("m0", (10, 40, 8), 240, 34, value=.92, density=.8, seed=221, top=.3),
              standing("m1", (30, -10, 2.2), 62, 10, value=.86, density=.5, seed=223, top=.4),          # low over the water, so that the shell keeps the eye
              standing("m2", (-8, 96, 26), 420, 78, value=.9, density=.92, seed=225, top=.3),           # and the country behind goes back into the morning
              standing("m3", (-58, 62, 30), 130, 64, value=.9, density=.55, seed=227, top=.35)]         # the great peak behind the shell, a wash and no more
    went = river_skin()
    # past the shell on its right, turning a little after it, then away down the river over the ink
    def focus(dof, u, e):
        on = O + (0, 3, 4) + (np.array((-8.5, 8.0, 4.4)) - (O + (0, 3, 4))) * sm((u - .2) / .2)      # the shell, then the slips
        dof.focus_distance = 20.0 * (1 - sm(u / .06)) + sm(u / .06) * (float(np.linalg.norm(e - on)) + 34 * sm((u - .6) / .3))
        dof.aperture_fstop = .9 + 4.1 * sm(u / .12)
    def each(n, t_, e):
        went(1.4 * t_)
        shell_ink.default_value = .02 + 1.08 * n / 61
        slip_ink.default_value = .12 + n / 44
    fly("bone", co, extras,
        eye=[cam, (-5, -21, 5.9), (-2.6, -13, 5.2), (.4, -3, 4.4), (3.5, 12, 3.3)],
        look=[(-10.5, 0, 3.9), (-12.2, .3, 4.2), (-10.5, 8, 3.7), (-9, 30, 1.8), (5, 72, -2.5)],
        each=each,
        focus=focus)


# ---- the steles ------------------------------------------------------------------------------------------------------------------------
def hewn_stone(name, value):
    """Stone that has been dressed and has weathered since: not one flat grey for a lens that goes close to it."""
    mt = grey_material(name, value, .9)
    nt = mt.node_tree
    b = nt.nodes["Principled BSDF"]
    m = maths(nt)
    tc = nt.nodes.new("ShaderNodeTexCoord")

    def noise(scale, detail):
        no = nt.nodes.new("ShaderNodeTexNoise")
        no.inputs["Scale"].default_value = scale
        no.inputs["Detail"].default_value = detail
        no.inputs["Roughness"].default_value = .6
        nt.links.new(tc.outputs["Object"], no.inputs["Vector"])
        return no.outputs[0]
    big, fine = noise(.35, 5), noise(5.0, 6)
    nt.links.new(m("MULTIPLY", value, m("ADD", m("MULTIPLY", big, 1.3), m("MULTIPLY", fine, .7))), b.inputs["Base Color"])
    bu = nt.nodes.new("ShaderNodeBump")
    bu.inputs["Strength"].default_value = .5
    bu.inputs["Distance"].default_value = .25
    nt.links.new(m("ADD", big, m("MULTIPLY", fine, .4)), bu.inputs["Height"])
    nt.links.new(bu.outputs[0], b.inputs["Normal"])
    return mt


def stele_mesh(name, xc, y0, base, w, H, relief, material, side_material, yaw=0.0, thick=2.6):
    """A slab with a rounded head. relief is its face's height map, or None for a flat one with a painted face."""
    n, m = (relief.shape if relief is not None else (64, 32))
    gy, gx = np.mgrid[:n, :m].astype(np.float32)
    fz, fx = gy / (n - 1), gx / (m - 1) - .5
    head = .84
    shrink = np.where(fz > head, np.sqrt(np.clip(1 - ((fz - head) / (1 - head)) ** 2, 0, 1)) * .55 + .45, 1.0)
    lx = fx * w * shrink
    Z = base + fz * H
    r = relief if relief is not None else np.zeros((n, m), np.float32)
    ca, sa = math.cos(yaw), math.sin(yaw)
    X = xc + lx * ca + r * sa
    Y = y0 + lx * sa - r * ca
    o = mesh_grid(name, X.astype(np.float32), Y.astype(np.float32), Z.astype(np.float32))
    o.data.materials.append(material)
    # the edge and the back
    ring_x = np.r_[lx[:, 0], lx[-1, :], lx[::-1, -1]]
    ring_z = np.r_[Z[:, 0], Z[-1, :], Z[::-1, -1]]
    sx, sy, sz = [], [], []
    for off in (0.0, thick):
        sx.append(xc + ring_x * ca - off * sa)
        sy.append(y0 + ring_x * sa + off * ca)
        sz.append(ring_z)
    e = mesh_grid(name + "_edge", np.array(sx, np.float32), np.array(sy, np.float32), np.array(sz, np.float32))
    e.data.materials.append(side_material)
    return o


def shot_stele(t=0.0):
    PEAKS = [(-130, 130, 62, 30), (-170, 70, 44, 28), (-110, 250, 84, 40), (72, 110, 66, 30), (120, 40, 44, 26), (40, 260, 86, 44), (-190, 230, 76, 44), (130, 220, 76, 46), (-20, 400, 96, 72)]
    valley(PEAKS, 231, dark=1.0, sink=(-44, 8, -30, 12))
    h, a, ro, wet_, when = (np.load(os.path.join(TEX, f"stele_{k}.npy")) for k in "harwt")
    far_a = np.load(os.path.join(TEX, "stele_far.npy"))
    stone, stele_ink = paint_run("stele", a, ro, wet_, when)
    far_m = painted_material("stele_far", far_a)
    side = hewn_stone("stele_side", .42)
    base_m = hewn_stone("plinth", .5)
    # the near one
    xc, y0 = -24.0, 2.0
    H, w = h.shape[0] * .0115, h.shape[1] * .0115
    stele_mesh("hero", xc, y0, 2.7, w, H, h, stone, side, yaw=math.radians(10))
    # its fellows in two rows going away along the bank
    r = np.random.default_rng(233)
    bases = []
    for k, (rx, ry) in enumerate([(-27, 30), (-29, 54), (-30, 80), (-30, 108), (-28, 138), (-26, 170), (-22, 205), (-48, 18), (-50, 44), (-52, 70), (-55, 100), (-56, 132), (-58, 168), (-60, 205), (-80, 34), (-84, 66), (-86, 104)]):
        Hk, wk = H * r.uniform(.72, 1.05), w * r.uniform(.72, 1.0)
        rx = rx + river_x(np.array(float(ry))).item() - 9
        stele_mesh(f"s{k}", rx, ry, 2.3, wk, Hk, None, far_m, side, yaw=math.radians(r.normal(8, 3)))
        bases.append((rx, ry, wk))
    for k, (rx, ry, wk) in enumerate(bases):
        b = rbox(f"p{k}", (wk * 1.22, 6.0, 2.4), base_m, .12)
        b.location = (rx, ry + 1.2, 1.2)
    # the plinth of the near one and the ink coming off it
    run = [(-20.5, -.9), (-19.5, -3.8), (-16.6, -5.0), (-14.0, -8.2), (-10.0, -8.0), (-6.0, -10.5), (-1.0, -10.5)]
    run2 = [(-27.0, -.9), (-26.6, -2.8), (-23.0, -4.2), (-19.5, -3.8)]
    plinth = rbox("plinth", (w * 1.26, 6.4, 2.7), base_m, .14)
    plinth.location = (xc, y0 + 1.0, 1.35)
    plinth.rotation_euler = (0, 0, math.radians(10))
    near_bank((-44, 8), (-30, 12), .024, 241, runs=[(run, .3, .9), (run2, .22, .3)], alb=.56, bumps=[(xc - 1.5, y0 - 4.2, 1.5, 5.0), (xc + 7, y0 - 3.6, 1.2, 4.0), (xc - 9, y0 - 2.0, 1.5, 4.0)], rock=0)
    plume("plume", (-1.0, -10.5), 170, 247)
    world_light(opt("world", .34))
    shine((.74, .44, -.5), energy=opt("sun", 2.2), angle=6.0)
    cam = (-3.5, -44.0, 3.4)
    co = camera(cam, (-10.0, 4.0, 12.0), lens=27, far=1200)
    co.data.dof.use_dof = True
    co.data.dof.focus_distance = 49.0
    co.data.dof.aperture_fstop = 1.4
    went = river_skin()
    def each(n, t_, e):
        went(1.4 * t_)
        stele_ink.default_value = .2 + .85 * n / 66
    # up the water with the row on the left hand, in under the shoulder of the near one, with the others going away behind it
    fly("stele", co, [],
        eye=[(-1.6, -50, 3.2), cam, (-5.6, -34, 3.6), (-7.6, -22, 3.8), (-9, -8, 4.0)],
        look=[(-8, -6, 11.5), (-10, 4, 12), (-14, 20, 12.5), (-20, 42, 13), (-26, 70, 13)],
        each=each,
        focus=lambda dof, u, e: setattr(dof, "focus_distance", max(float(np.hypot(e[0] - xc, e[1] - y0)), 14.0)))


# ---- the screen ------------------------------------------------------------------------------------------------------------------------
def shot_screen(t=0.0):
    from mathutils import Euler, Matrix, Vector
    PEAKS = [(-74, 86, 56, 30), (-110, 30, 40, 26), (-30, 170, 80, 36), (70, 90, 60, 30), (116, 26, 42, 26), (36, 210, 88, 44), (-124, 180, 70, 42), (126, 170, 74, 44), (0, 350, 96, 72)]
    pz = opt("peakz", .42)            # low hills: a wide empty sky, nothing but this one light under it
    valley([(a, b, c * pz, d) for a, b, c, d in PEAKS], 251, dark=.7, sink=(-34, 12, -42, 12), rough=.05)
    body = grey_material("alu", .2, .32)
    black = grey_material("key", .012, .45)
    W, D = 12.0, 8.3
    origin = Vector((-7.3, -18.8, .66))
    M = Matrix.Translation(origin) @ Euler((0, 0, math.radians(opt("yaw", 14.0)))).to_matrix().to_4x4() @ Matrix.Scale(1.25, 4)
    parts = []
    base = rbox("base", (W, D, .3), body, .1)
    base.location = (0, 0, .15)
    parts.append(base)
    deck = rbox("deck", (W - 1.0, 4.6, .03), grey_material("deck", .05, .5), 0)
    deck.location = (0, 1.15, .31)
    parts.append(deck)
    # keys
    kx, ky = np.meshgrid(np.linspace(-5.0, 5.0, 14), np.linspace(-.8, 3.1, 6))
    pos = np.c_[kx.ravel(), ky.ravel(), np.full(kx.size, .335)]
    cube = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], float)
    F = np.array([(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)])
    keys = instances_mesh("keys", cube, F, pos, np.tile(np.array((.32, .30, .022)), (len(pos), 1)))
    keys.data.materials.append(black)
    parts.append(keys)
    pad = rbox("pad", (3.6, 2.2, .015), grey_material("pad", .3, .35), 0)
    pad.location = (0, -2.7, .31)
    parts.append(pad)
    # the lid and its glass
    lid_angle = math.radians(opt("lid", 112.0))
    hinge = Matrix.Translation((0, D / 2 - .05, .3)) @ Matrix.Rotation(math.pi / 2 - lid_angle, 4, "X")
    Hs = 7.9
    lid = rbox("lid", (W, .2, Hs), body, .08)
    lid.matrix_world = hinge @ Matrix.Translation((0, .10, Hs / 2))
    parts.append(lid)
    glass = rbox("bezel", (W - .1, .02, Hs - .1), grey_material("bezel", .006, .08), 0)
    glass.matrix_world = hinge @ Matrix.Translation((0, -.005, Hs / 2))
    parts.append(glass)
    e, wet_, when = (np.load(os.path.join(TEX, f"screen_{k}.npy")) for k in "ewt")
    sw, sh = W - .9, (W - .9) * e.shape[0] / e.shape[1]
    gy, gx = np.mgrid[:2, :2].astype(np.float32)
    sc = mesh_grid("screen", (gx - .5) * sw, np.full((2, 2), -.02, np.float32), gy * sh)
    mt = bpy.data.materials.new("screen")
    mt.use_nodes = True
    nt = mt.node_tree
    nt.nodes.clear()
    ti, em, out, mul = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial"), nt.nodes.new("ShaderNodeMath")
    ti.image = image_from(e, "screen_e")
    ti.interpolation = "Cubic"
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = opt("glow", 1.0)          # any more and a pixel half covered by a stroke clips to white, and the page goes blank
    # dark where the writing is, and where it has run to by now
    m_ = maths(nt)
    tw, tt, screen_ink = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeValue")
    tw.image, tt.image = image_from(wet_, "screen_w"), image_from(when, "screen_t")
    come = m_("MINIMUM", m_("MAXIMUM", m_("ADD", m_("DIVIDE", m_("SUBTRACT", screen_ink.outputs[0], tt.outputs[0]), .03), .5), 0.0), 1.0)
    nt.links.new(m_("MULTIPLY", ti.outputs[0], m_("SUBTRACT", 1.0, m_("MULTIPLY", tw.outputs[0], come))), mul.inputs[0])
    nt.links.new(mul.outputs[0], em.inputs["Strength"])
    em.inputs[0].default_value = (1, 1, 1, 1)
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    sc.data.materials.append(mt)
    sc.matrix_world = hinge @ Matrix.Translation((0, 0, (Hs - sh) / 2))
    parts.append(sc)
    # the glass is exposed for its text; what it throws on the bank and the water is stronger than that, so it comes from a lamp the lens does not see
    sl = bpy.data.lights.new("spill", "AREA")
    sl.shape, sl.size, sl.size_y = "RECTANGLE", sw, sh
    sl.energy = opt("spill", 900.0)
    so = bpy.data.objects.new("spill", sl)
    bpy.context.collection.objects.link(so)
    so.matrix_world = hinge @ Matrix.Translation((0, -.25, Hs / 2)) @ Matrix.Rotation(-math.pi / 2, 4, "X")
    so.visible_camera = so.visible_glossy = False
    parts.append(so)
    # the ink that has come down the glass and across the keys, off the front edge
    r = np.random.default_rng(257)
    segs = []
    for q in r.uniform(-4.8, 4.8, 9):
        x0 = float(q)
        p = np.array((x0, D / 2 - .1, .4))
        rad = r.uniform(.022, .05)
        while p[1] > -D / 2 + .05:
            nx_ = np.array((p[0] + r.normal(0, .08), p[1] - .55, .4))
            segs.append((p, nx_, rad, rad * .97))
            p = nx_
        segs.append((p, p + np.array((r.normal(0, .02), -.1, -.42)), rad, rad * .8))
    trails = tube_mesh(segs, "trails", sides=6)
    trails.data.materials.append(painted_ink())
    parts.append(trails)
    bpy.context.view_layer.update()
    for o in parts:
        o.matrix_world = M @ o.matrix_world

    def world(lx, ly):
        return tuple((M @ Vector((lx, ly, 0)))[:2])
    run = [world(-3.0, -4.3), world(-2.0, -6.2), world(1.0, -7.2), world(2.4, -9.6)]
    run2 = [world(2.2, -4.3), world(2.6, -6.2), world(1.0, -7.2)]
    near_bank((-34, 12), (-42, 12), .024, 261, runs=[(run, .18, .55), (run2, .12, .16)], alb=.42)
    plume("plume", run[-1], 170, 267, wide=6.0)
    # before it on the bank, as the lens comes: a printed book, and a letter, each leaning on its stone with its foot at the water
    rest = hewn_stone("rest", .3)
    laptop_rest = rbox("laptop_rest", (W + 2.0, D + 2.0, 1.0), rest, .18)                # it stands on its stone in the shallows, as the others do
    laptop_rest.matrix_world = M @ Matrix.Translation((0, 0, -.5))
    ba, br, bw, bt = (np.load(os.path.join(TEX, f"book_{k}.npy")) for k in "arwt")
    Wb = ba.shape[1] / 180

    def leaves(a_, b_):
        q = np.abs(a_) / (Wb / 2)
        return .25 + 1.05 * sm(q / .16) * (1 - .62 * sm((q - .22) / .78)) + .05 * np.sin(b_ * .5 + a_ * .2)
    book, slope_b = leaning("book", (-23.0, -50.0, 1.0), (-26.0, -42.0, 8.7), Wb, shape=leaves, nu=420, nv=160)
    bmt, book_ink = paint_run("book", ba, br, bw, bt, glow=opt("page", .3))
    book.data.materials.append(bmt)
    block = rbox("book_block", (Wb * .985, slope_b[4] * .985, .5), grey_material("edges", .62, .9), .04)
    from mathutils import Matrix as M3
    block_m = M3((slope_b[1], slope_b[2], slope_b[3])).transposed().to_4x4()
    block_m.translation = tuple(slope_b[0] + slope_b[2] * slope_b[4] / 2)
    block.matrix_world = block_m
    under("book_rest", slope_b, Wb, rest, thick=1.1, lift=.3)
    falling("book_fall", slope_b, Wb, 277, count=9, reach=3.2)
    plume("plume_b", (-15.6, -48.5), 110, 281, wide=3.0)
    la, lr, lw, lt = (np.load(os.path.join(TEX, f"letter_{k}.npy")) for k in "arwt")
    Wl = la.shape[1] / 200

    def sheet(a_, b_):
        return .08 * np.sin(a_ * .6 + b_ * .35) + .55 * sm((b_ - 8.5) / 2.5) * sm((-a_ - .5) / 3.5) - .03 * np.exp(-((b_ - 5.5) / .07) ** 2) - .03 * np.exp(-(a_ / .07) ** 2) + .12
    letter, slope_l = leaning("letter", (-15.0, -38.6, .95), (-16.6, -31.7, 9.3), Wl, shape=sheet, nu=240, nv=320)
    lmt, letter_ink = paint_run("letter", la, lr, lw, lt, glow=opt("page", .3))
    letter.data.materials.append(lmt)
    under("letter_rest", slope_l, Wl, rest, thick=1.0)
    falling("letter_fall", slope_l, Wl, 279, count=6, reach=3.0)
    plume("plume_l", (-12.0, -37.5), 90, 283, wide=2.4)
    world_light(opt("world", .26))               # a night of grey ink on paper, not a black one
    shine((-.5, -.5, -.7), energy=opt("sun", .9), angle=3.0)
    cam = (3.0, -30.0, 4.6)
    co = camera(cam, (-8.0, 0.0, 4.6), lens=opt("lens", 30), far=1200)
    extras = [standing("m0", (0, 90, 9), 380, 30, value=.62, density=.75, seed=271, top=.3)]
    went = river_skin(near=opt("swell", .35))
    def each(n, t_, e):
        went(1.0 * t_)
        book_ink.default_value = n / 36
        letter_ink.default_value = (n - 24) / 34
        screen_ink.outputs[0].default_value = .12 + .9 * n / 105
    # up the bend of the river, past a book and a letter, over the screen's reflection and on into the page, until its words are all there is
    fly("screen", co, extras,
        eye=[(-17, -86, 3.0), (-9.5, -66, 3.0), (-3, -51, 3.0), (-.8, -42, 3.1), (-2.6, -35.8, 3.4), (-5.2, -26, 4.2), (-7.8, -16.4, 5.2)],
        look=[(-11, -10, 2.6), (-11, -12, 2.8), (-10, -12, 3.4), (-9, -12, 4.6), (-9, -12, 5.4), (-9, -12, 5.6)],
        each=each)


def painted_ink():
    m = bpy.data.materials.new("wet_ink")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (.008, .008, .008, 1)
    b.inputs["Roughness"].default_value = .45
    b.inputs["Specular IOR Level"].default_value = .05         # it must not catch the screen's light: it is a thin black line, not a cord
    return m


# ---- the falls -------------------------------------------------------------------------------------------------------------------------
C_LIP = .012          # how deep the horseshoe is: from the middle of the lip the white water goes down on both hands, with the pool in sight between them
LIP_OUT = 9.0         # how far the water leaves the rock
LIP_HALF = 100.0
HOLE = (-14.0, -38.0)
POOL_R, POOL_IN = 220.0, 60.0     # as in the prep: the pool's two textures, the whole of it and close round the hole
RIVER_L = 300.0


def lip_of(a):
    return 58 - C_LIP * a ** 2


def lip_normal(a):
    """Off the lip, into the pit."""
    g = 2 * C_LIP * a
    k = np.sqrt(1 + g * g)
    return -g / k, -1 / k


def noise_1d(r, n, s):
    """A smooth random line, unit variance."""
    a = np.fft.irfft(np.fft.rfft(r.normal(size=n)) * np.exp(-(np.fft.rfftfreq(n) * s) ** 2), n=n).astype(np.float32)
    return (a - a.mean()) / a.std()


def cliff_material(name, albedo):
    """A painting laid on from above is right for what faces up; down a wall it pulls out into stripes. So the walls are painted on the wall itself, in the rock's own beds."""
    mt = painted_material(name, albedo)
    nt = mt.node_tree
    b = nt.nodes["Principled BSDF"]
    top = b.inputs["Base Color"].links[0].from_socket
    m = maths(nt)
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Normal"], sep.inputs[0])

    def beds(sx, sz, seed, detail):
        mp, no = nt.nodes.new("ShaderNodeMapping"), nt.nodes.new("ShaderNodeTexNoise")
        mp.inputs["Scale"].default_value = (sx, sx, sz)
        mp.inputs["Location"].default_value = (seed, seed * .7, 0)
        no.inputs["Scale"].default_value = 1.0
        no.inputs["Detail"].default_value = detail
        no.inputs["Roughness"].default_value = .62
        nt.links.new(geo.outputs["Position"], mp.inputs["Vector"])
        nt.links.new(mp.outputs[0], no.inputs["Vector"])
        return no.outputs[0]
    big, seam, fine = beds(.012, .1, 3.1, 4), beds(.03, .42, 7.7, 6), beds(.14, .3, 1.3, 5)
    tone = m("ADD", .05, m("MULTIPLY", .95, m("MULTIPLY", m("MULTIPLY", m("SUBTRACT", big, .18), 2.0), m("MULTIPLY", m("SUBTRACT", seam, .22), 2.6))))
    tone = m("MULTIPLY", tone, m("ADD", .55, m("MULTIPLY", fine, .9)))
    wall = nt.nodes.new("ShaderNodeMapRange")
    wall.interpolation_type = "SMOOTHSTEP"
    wall.inputs["From Min"].default_value = .22
    wall.inputs["From Max"].default_value = .62
    wall.inputs["To Min"].default_value = 1.0
    wall.inputs["To Max"].default_value = 0.0
    nt.links.new(sep.outputs["Z"], wall.inputs["Value"])
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    nt.links.new(wall.outputs[0], mx.inputs["Factor"])
    nt.links.new(top, mx.inputs["A"])
    nt.links.new(tone, mx.inputs["B"])
    nt.links.new(mx.outputs["Result"], b.inputs["Base Color"])
    bu = nt.nodes.new("ShaderNodeBump")
    bu.inputs["Strength"].default_value = .8
    bu.inputs["Distance"].default_value = 2.0
    nt.links.new(m("ADD", m("MULTIPLY", seam, 1.0), m("MULTIPLY", fine, .35)), bu.inputs["Height"])
    nt.links.new(bu.outputs[0], b.inputs["Normal"])
    return mt


def dark_lamp(mt):
    """What it shines with is for the lens: left a lamp, a million triangles of white water show in every glossy thing whatever the ray settings say, and slow the frame."""
    mt.cycles.emission_sampling = "NONE"


def water_shader(nt, strength, gloss, rough, dull=None):
    """Black water: what it reflects, and on top the light it has of its own. dull: a socket from 0 to 1, how much of its shine it keeps."""
    em, gl, add, out = (nt.nodes.new(k) for k in ("ShaderNodeEmission", "ShaderNodeBsdfGlossy", "ShaderNodeAddShader", "ShaderNodeOutputMaterial"))
    em.inputs[0].default_value = (1, 1, 1, 1)
    nt.links.new(strength, em.inputs[1])
    gl.inputs["Color"].default_value = (gloss, gloss, gloss, 1)
    gl.inputs["Roughness"].default_value = rough
    if dull is not None:
        tint = maths(nt)("MULTIPLY", dull, gloss)
        nt.links.new(tint, gl.inputs["Color"])
    nt.links.new(em.outputs[0], add.inputs[0])
    nt.links.new(gl.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])


def river_material(name, light, gain=1.0, gloss=.55, rough=.03):
    """The river with its words on it, going toward row 0, and white where it goes over. Returns the material and the value that says how far it has gone."""
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    nt = mt.node_tree
    nt.nodes.clear()
    m = maths(nt)
    tc, sep, comb, flow, ti = (nt.nodes.new(k) for k in ("ShaderNodeTexCoord", "ShaderNodeSeparateXYZ", "ShaderNodeCombineXYZ", "ShaderNodeValue", "ShaderNodeTexImage"))
    nt.links.new(tc.outputs["UV"], sep.inputs[0])
    nt.links.new(sep.outputs["X"], comb.inputs["X"])
    nt.links.new(m("ADD", sep.outputs["Y"], flow.outputs[0]), comb.inputs["Y"])
    ti.image = image_from(light, name + "_e")
    ti.interpolation = "Cubic"
    nt.links.new(comb.outputs[0], ti.inputs[0])
    mp, no = nt.nodes.new("ShaderNodeMapping"), nt.nodes.new("ShaderNodeTexNoise")
    mp.inputs["Scale"].default_value = (2.6, .02, 1.0)
    no.inputs["Scale"].default_value = 1.0
    no.inputs["Detail"].default_value = 8
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs[0], no.inputs["Vector"])
    foam = m("POWER", 2.718, m("MULTIPLY", sep.outputs["Y"], -RIVER_L / 13))
    white = m("MULTIPLY", foam, m("MINIMUM", m("MAXIMUM", m("MULTIPLY", m("SUBTRACT", no.outputs[0], .3), 2.2), .22), 1.05))
    gone = nt.nodes.new("ShaderNodeMapRange")          # the words have gone under before it goes over
    gone.interpolation_type = "SMOOTHSTEP"
    gone.inputs["From Min"].default_value, gone.inputs["From Max"].default_value = 7.0 / RIVER_L, 22.0 / RIVER_L
    nt.links.new(sep.outputs["Y"], gone.inputs["Value"])
    water_shader(nt, m("MULTIPLY", m("ADD", m("MULTIPLY", ti.outputs[0], gone.outputs[0]), white), gain), gloss, rough)
    dark_lamp(mt)
    return mt, flow.outputs[0]


def pool_material(name, whole, close, gain=1.0, gloss=.55, rough=.04, base=.2):
    """The pool's own light, read through a turn: the nearer the hole the faster it goes, and all of it draws in. Returns the material and the value that says how long it has been turning."""
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    nt = mt.node_tree
    nt.nodes.clear()
    m = maths(nt)
    tc, sep, clock = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeSeparateXYZ"), nt.nodes.new("ShaderNodeValue")
    nt.links.new(tc.outputs["UV"], sep.inputs[0])
    dx, dy = m("SUBTRACT", sep.outputs["X"], .5), m("SUBTRACT", sep.outputs["Y"], .5)
    r = m("SQRT", m("ADD", m("MULTIPLY", dx, dx), m("MULTIPLY", dy, dy)))
    turn = m("DIVIDE", m("MULTIPLY", clock.outputs[0], .05), m("ADD", r, .05))
    c, s = m("COSINE", turn), m("SINE", turn)
    grow = m("ADD", 1.0, m("MULTIPLY", clock.outputs[0], .05))
    x2 = m("MULTIPLY", m("ADD", m("MULTIPLY", dx, c), m("MULTIPLY", dy, s)), grow)
    y2 = m("MULTIPLY", m("SUBTRACT", m("MULTIPLY", dy, c), m("MULTIPLY", dx, s)), grow)

    def lay(pic, label, k):
        co, ti = nt.nodes.new("ShaderNodeCombineXYZ"), nt.nodes.new("ShaderNodeTexImage")
        nt.links.new(m("ADD", m("MULTIPLY", x2, k), .5), co.inputs["X"])
        nt.links.new(m("ADD", m("MULTIPLY", y2, k), .5), co.inputs["Y"])
        ti.image = image_from(pic, label)
        ti.interpolation = "Cubic"
        ti.extension = "EXTEND"
        nt.links.new(co.outputs[0], ti.inputs[0])
        return ti.outputs[0]
    far, near = lay(whole, name + "_w", 1.0), lay(close, name + "_c", POOL_R / POOL_IN)
    mix = nt.nodes.new("ShaderNodeMapRange")
    mix.interpolation_type = "SMOOTHSTEP"
    mix.inputs["From Min"].default_value = .72 * POOL_IN / POOL_R / 2
    mix.inputs["From Max"].default_value = .92 * POOL_IN / POOL_R / 2
    nt.links.new(r, mix.inputs["Value"])
    out = m("ADD", m("MULTIPLY", near, m("SUBTRACT", 1.0, mix.outputs[0])), m("MULTIPLY", far, mix.outputs[0]))
    bright = nt.nodes.new("ShaderNodeMapRange")
    bright.interpolation_type = "SMOOTHSTEP"
    bright.inputs["From Min"].default_value = 8.0 / POOL_R / 2
    bright.inputs["From Max"].default_value = 26.0 / POOL_R / 2
    nt.links.new(r, bright.inputs["Value"])
    water_shader(nt, m("ADD", m("MULTIPLY", out, gain), base), gloss, rough, dull=bright.outputs[0])
    dark_lamp(mt)
    return mt, clock.outputs[0]


def curtain_material(name, edge, cover, tone, words, strength):
    """The white water: what does not move (its torn sides, where the rock shows through) and what does (the streaks, fast, and the words pulled long, slower). Returns the material and the two values that say how far each has gone."""
    mt = bpy.data.materials.new(name)
    mt.use_nodes = True
    mt.blend_method = "BLEND"
    nt = mt.node_tree
    nt.nodes.clear()
    m = maths(nt)
    tc, sep = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["UV"], sep.inputs[0])
    fast, slow = nt.nodes.new("ShaderNodeValue"), nt.nodes.new("ShaderNodeValue")

    def tex(pic, label, went=None):
        ti = nt.nodes.new("ShaderNodeTexImage")
        ti.image = image_from(pic.astype(np.float32), name + label)
        if went is not None:
            co = nt.nodes.new("ShaderNodeCombineXYZ")
            nt.links.new(sep.outputs["X"], co.inputs["X"])
            nt.links.new(m("ADD", sep.outputs["Y"], went), co.inputs["Y"])
            nt.links.new(co.outputs[0], ti.inputs[0])
        return ti.outputs[0]
    e, c, t, w = tex(edge, "_e"), tex(cover, "_c", fast.outputs[0]), tex(tone, "_t", fast.outputs[0]), tex(words, "_w", slow.outputs[0])
    fade = nt.nodes.new("ShaderNodeMapRange")          # they have gone by halfway down
    fade.interpolation_type = "SMOOTHSTEP"
    fade.inputs["From Min"].default_value, fade.inputs["From Max"].default_value = .3, .62
    nt.links.new(sep.outputs["Y"], fade.inputs["Value"])
    word = m("MULTIPLY", m("MULTIPLY", w, fade.outputs[0]), e)
    tr, em, mx, out = (nt.nodes.new(k) for k in ("ShaderNodeBsdfTransparent", "ShaderNodeEmission", "ShaderNodeMixShader", "ShaderNodeOutputMaterial"))
    em.inputs[1].default_value = strength
    nt.links.new(m("MULTIPLY", t, m("SUBTRACT", 1.0, m("MULTIPLY", word, .88))), em.inputs[0])
    nt.links.new(m("MAXIMUM", m("MULTIPLY", e, c), word), mx.inputs[0])
    nt.links.new(tr.outputs[0], mx.inputs[1])
    nt.links.new(em.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], out.inputs["Surface"])
    dark_lamp(mt)
    return mt, fast.outputs[0], slow.outputs[0]


def shot_falls(t=0.0):
    SIZE, N = 760.0, 2400
    y, x = np.mgrid[:N, :N].astype(np.float32)
    X, Y = (x / (N - 1) - .5) * SIZE, (y / (N - 1) - .5) * SIZE - 40
    TOP = opt("top", 112.0)
    tx, ty = x * 1024 / N, y * 1024 / N

    def course(b):
        return 46 * np.sin((b - 58) / 130) * sm((b - 58) / 90)

    def half_width(b):
        return np.maximum(LIP_HALF - .28 * np.maximum(b - 58, 0), 60)

    lip = lip_of(X) + 7 * lerp(spectral(1024, 3.0, 281), tx, ty)
    # how far back from the edge, across it, wherever the edge is heading; the wall goes back in beds, it is not one face, and only beside the water does it stand sheer, to hold the river in
    across = (Y - lip) / np.sqrt(1 + (2 * C_LIP * X) ** 2)
    side = np.abs(X - course(Y)) - half_width(Y) + 10 * lerp(spectral(1024, 2.8, 303), tx, ty)
    by_water = 1 - sm((side - 8) / 26)
    cliff = 0.0
    ledges = ((0, .2, 3.0, 5, 295), (9, .17, 3.0, 5, 296), (20, .2, 3.5, 6, 297), (33, .16, 4.0, 7, 298), (48, .15, 5.0, 8, 299), (66, .12, 6.0, 9, 300))
    for at_, part, width, wander, seed in ledges:
        cliff = cliff + part * sm((across - at_ - wander * lerp(spectral(1024, 2.7, seed), tx, ty)) / width)
    cliff = cliff * (1 - by_water) + sm(across / 3.5) * by_water
    rolling = sm((across - 34) / 20) * sm(side / 20) * (10 * lerp(spectral(1024, 3.6, 311), tx, ty) + 2.5 * lerp(spectral(1024, 2.8, 312), tx, ty))
    hills = massif(X, Y, [(-210, 130, 64, 60), (-270, 10, 84, 70), (-230, -150, 96, 70), (230, 150, 60, 60), (280, 0, 88, 76), (240, -170, 100, 76)], 283)
    h = TOP * cliff + rolling + hills * sm((across - 50) / 70) * sm(side / 40)
    h = erode(h, SIZE / N, 285, wavelength=44, octaves=7, gain=.5, strength=.13)
    h = np.maximum(h, 0)
    # the river's bed above the lip, and under the pool a pit deep enough for the hole
    h = h - 3.5 * (1 - sm(side / 9)) * sm((across + 6) / 10)
    h = h - 14 * (1 - sm((np.abs(X - np.interp(Y, (50, 80, 112, 149), (0, 4, 11, 20))) - 20) / 12)) * sm((Y - 62) / 6)      # no rocks close by the lens's way down the river
    h = h - 34 * (1 - sm((np.hypot(X - HOLE[0], Y - HOLE[1]) - 60) / 40)) - 8 * (1 - sm(across / 5))
    land = mesh_grid("cliff", X, Y, h)
    land.data.materials.append(cliff_material("cliff_rock", ink_albedo(np.maximum(h, 0), 287, SIZE / N) * opt("rockv", .72)))

    # the river: black, wide, full of words. Row 0 is where it goes over
    nr, mr = 700, 240
    gv, gu = np.mgrid[:nr, :mr].astype(np.float32)
    gv, gu = gv / (nr - 1), gu / (mr - 1) * 2 - 1
    ground = 58 + gv * RIVER_L
    ra = course(ground) + gu * (half_width(ground) + 16)
    nx_, ny_ = lip_normal(ra)
    close_ = (1 - gv) ** 40 * (2 * sm((LIP_HALF + 2 - np.abs(ra)) / 6) - 1)        # past the ends of the falls it goes in under the rock
    RX = ra + nx_ * LIP_OUT * close_
    RY = ground + (lip_of(ra) - 58) * (1 - gv) ** 6 + ny_ * LIP_OUT * close_
    river = mesh_grid("river", RX.astype(np.float32), RY.astype(np.float32), np.full((nr, mr), TOP - 1.6, np.float32))
    river_m, river_flow = river_material("river", np.load(os.path.join(TEX, "river_f.npy")), gain=opt("wordr", 1.0))
    river.data.materials.append(river_m)
    river.visible_shadow = False

    # the curtain of water: falls off the lip all the way round it, bowing out as it goes
    nz, nf = 512, 1600
    gy, gx = np.mgrid[:nz, :nf].astype(np.float32)
    fz, fx = gy / (nz - 1), gx / (nf - 1) * 2 - 1
    tab = np.linspace(-LIP_HALF, LIP_HALF, 8001)
    arc = np.cumsum(np.sqrt(1 + (2 * C_LIP * tab) ** 2)) * (tab[1] - tab[0])
    arc = arc - arc[4000]
    ca = np.interp(fx * arc[-1], arc, tab)
    off = LIP_OUT + 15 * (1 - fz) ** 1.6
    nx_, ny_ = lip_normal(ca)
    CX, CY = ca + nx_ * off, lip_of(ca) + ny_ * off
    CZ = fz * (TOP - 1.8) + .2
    rnd = np.random.default_rng(289)
    kx = np.fft.rfftfreq(2048)[None, :]
    ky = np.fft.fftfreq(1024)[:, None]
    spec = np.fft.rfft2(rnd.normal(size=(1024, 2048))) * np.exp(-(ky * 120) ** 2) / (np.abs(kx) + .002) ** .55
    streak = np.fft.irfft2(spec, s=(1024, 2048)).astype(np.float32)
    streak = (streak - streak.mean()) / streak.std()
    uu = np.linspace(-1, 1, 2048, dtype=np.float32)[None, :]
    # it is not one sheet: here and there the rock shows through, and at its two ends it thins out strand by strand
    body = sm((noise_1d(rnd, 2048, 70)[None, :] + 1.25) / .5)
    ends = sm((1 - np.abs(uu) + .05 * noise_1d(rnd, 2048, 9)[None, :]) / .16)
    edge = (.3 + .7 * body) * ends * np.ones((1024, 2048), np.float32)
    cover = np.clip(.68 + .5 * streak, 0, 1)
    tone = np.clip(.74 + .3 * streak, .25, 1)
    curtain = mesh_grid("falls", CX.astype(np.float32), CY.astype(np.float32), CZ.astype(np.float32))
    falls_m, fast, slow = curtain_material("falls", edge, cover, tone, np.load(os.path.join(TEX, "fall_w.npy")), opt("water", 1.05))
    curtain.data.materials.append(falls_m)
    curtain.visible_shadow = False
    curtain.visible_glossy = False
    curtain["flat"] = 1             # it hides from the depth pass with the mist, but it is the one thing there that must not turn to the lens

    # the pool: one surface, dished round the hole and ridged along the whirl, so that what it reflects bends the way the water goes
    nP = 1500
    gy, gx = np.mgrid[:nP, :nP].astype(np.float32)
    dx, dy = (gx / (nP - 1) * 2 - 1) * POOL_R, (gy / (nP - 1) * 2 - 1) * POOL_R
    rr, th = np.hypot(dx, dy), np.arctan2(dy, dx)
    ridge = np.sin(5 * th + 2 * math.pi * 5 * .42 * np.log(rr + 6))
    PZ = .2 - opt("dish", 18.0) * np.exp(-(rr / 17) ** 1.5) - 1.6 * np.exp(-(rr / 80) ** 2) + .32 * ridge * sm((rr - 8) / 20) * np.exp(-rr / 130)
    pool = mesh_grid("pool", dx.astype(np.float32), dy.astype(np.float32), PZ.astype(np.float32))
    pool.location = (HOLE[0], HOLE[1], 0)
    pool_m, clock = pool_material("pool", np.load(os.path.join(TEX, "pool_f.npy")), np.load(os.path.join(TEX, "pool_fi.npy")), gain=opt("wordp", 1.2), gloss=opt("gloss", .55), base=opt("base", .2))
    pool.data.materials.append(pool_m)
    pool.visible_shadow = False

    world_light(opt("world", .2))
    shine((.5, .55, -.66), energy=opt("sun", 3.6), angle=5.0)
    bpy.data.objects["sun"].visible_glossy = False          # its glint along every ridge of the whirl is a crack in the picture
    co = camera((-34.0, -330.0, 230.0), (-4.0, -8.0, 54.0), lens=opt("lens", 31), far=1800)
    # the far end of the pit goes off into mist, and the spray climbs its walls and lies on the water
    extras = [curtain,
              standing("m_far", (-10, -225, 90), 1000, 300, value=.9, density=1.0, seed=291, top=.25),
              standing("m_lip", (0, 26, 112), 320, 44, value=.94, density=.6, seed=293, top=.4, thin=70),
              standing("m_wl", (-122, -84, 44), 260, 120, value=.86, density=.82, seed=295, top=.4),
              standing("m_wr", (122, -84, 44), 260, 120, value=.86, density=.82, seed=297, top=.4),
              lying("l_l", (-82, -22, 9), 76, 190, value=.92, density=.85, seed=301),
              lying("l_r", (80, -22, 9), 76, 190, value=.92, density=.85, seed=303),
              lying("l_b", (-2, 36, 9), 220, 46, value=.92, density=.9, seed=305),
              lying("l_u", (-46, 14, 26), 110, 60, value=.94, density=.45, seed=307)]

    def each(n, t_, e):
        river_flow.default_value = t_ * opt("runs", 22.0) / RIVER_L
        fast.default_value = t_ * 1.1
        slow.default_value = t_ * .32
        clock.default_value = t_
        pool.rotation_euler = (0, 0, .1 * t_)
    # along the river the way it goes, over the lip, and down the middle of the pit to the hole
    fly("falls", co, extras,
        eye=[(20, 149, 124), (11, 112, 123.5), (4, 80, 123), (0, 50, 120), (-3, 22, 111), (-7, -4, 94), (-11, -20, 72), (-13, -26, 56)],
        look=[(10, 90, 112), (5, 50, 108), (0, 15, 88), (-5, -15, 58), (-10, -32, 25), (-14, -38, -6), (-14, -38.5, -18)],
        each=each,
        roll=lambda u: -math.radians(32) * float(sm((u - .45) / .55)))


SHOTS = {"bone": shot_bone, "stele": shot_stele, "screen": shot_screen, "falls": shot_falls}

if __name__ == "__main__":
    shot = opt("shot", "bone")
    reset(samples=opt("samples", 64), pct=opt("pct", 100))
    SHOTS[shot](opt("t", 0.0))
