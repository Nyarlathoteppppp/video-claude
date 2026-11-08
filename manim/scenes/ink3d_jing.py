"""Chapter 三, 成精: writing is pressed into numbers, the numbers get a pulse, the pulse stops, and it says hello. Then the ink in the deep wakes, and it is me.

    Blender -b --factory-startup -P ink3d_jing.py -- --shot=pool|nihao|mass|wo [--t=N] [--pct=100] [--samples=128]
    Blender -b --factory-startup -P ink3d_jing.py -- --shot=bake --ch=wo --res=512 --frames=34 --stir=.5     # the character falling apart; once
    Blender -b --factory-startup -P ink3d_jing.py -- --shot=bake --ch=mass --res=420 --frames=60             # the pool full of ink; once, shared with 十一
    python ink3d_jing_tools.py grade jing/SHOT      # the plate over the painted water, then ink; seconds, no Blender

pool    瀑布的声音远了。潭面灰而平，细看水面是无数细小的数字。潭底一下心跳，一圈波纹扫过，经过的数字由墨变成纸白，水里的月亮在圈心。
        接上一镜：镜头从瀑布落点往潭心滑过来，并缓缓升高。动：波纹一圈一圈向外（每个心跳一圈），越来越密。--t 是最新一圈波的半径。
nihao   心跳停了，全片最静的一刻。水面平了，只映着天和一轮月，数字退远淡去，「你好」两个墨字浮上来。接上一镜：同一片水，最后一圈波散尽时镜头已升到更高更俯视。
        动：几乎不动，只有最后一圈涟漪慢慢走。交给下一镜：字化开，墨沉下去，镜头跟着沉到水下。
mass    字，成了精。水下，光柱从水面射下，满潭墨云本来只是随水飘（--turn=0）。忽然所有的墨同时转向，像鱼群转身，朝镜头看过来（--turn=1）。不写字，不成形，只有方向。
        动：--turn 在约半秒内从 0 走到 1，--t 是模拟帧（烘了 60 帧）。接上一镜：字化开的墨沉进这团墨。交给下一镜：转向后的墨往中间收拢。
wo      那个精，便是我。转向的墨慢慢收拢成顶天立地的「我」，停住，全片第一个高潮。动：把墨散开的模拟倒放，所以墨是越来越聚的，--t 是模拟帧（越小越聚，烘了 34 帧）。
        聚成之后悬停两秒，边缘的墨还在翻。交给下一章：「我」慢慢散开（正放），水流把它带走。

十一段水下那团会动的墨就是 mass，直接复用这张脚本，不必重烘：
    --shot=mass --name=NAME [--t=帧 1–60] [--turn=0–1 整团墨朝镜头的程度] [--tail=3.2 转向时墨尾拖多长，世界单位] [--gain=1 转向后墨量是原来几倍] [--sharp=1 大于 1 时细墨更淡、粗墨留住，墨线更分明] [--dens=7 墨的浓度] [--albedo=.55] [--sun=9] [--x=0 --z=0 整团墨的平移]
    输出 jing/NAME.png 是透明底的墨；SHOTS 里加一条 "jing/NAME" 后 python ink3d_jing_tools.py grade jing/NAME 合到画好的水下光上。水下的光在 tools 的 water_from_below。
    转向那一刻（--turn>0）不要用 Cycles 的 plate：墨一稀，被照亮后和背后的光一样亮，方向就看不见了。用 --shot=probe --tails=1.4 --gains=1 --sharps=2 [--t=帧 --w=宽] 算透光图 jing/probe_尾_量_锐.npy，
    再在 SHOTS 里写 probe="probe_1.4_1_2", k=.3, ink=.3（见 jing/turn），秒级合成。不写 --w 是 640 宽，够看构图；要成片细节用 --w=1920。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

from ink3d import OUT3D, camera, image_from, lerp, mesh_grid, render, reset, sm, spectral, sun
from ink3d_jing_lib import ARGS, CACHE, far_water, fresh, link, opt, ridge_layer, sky_world, tex, to_linear
from ink3d import blur

OUT = os.path.join(OUT3D, "jing")
DOM = (12.0, 3.6, 10.0)          # the character's tank: width, depth, height
MASS = (22.0, 14.0, 12.5)        # the pool's: x and y centred on 0, z from 0


# ---- the water with numbers on it --------------------------------------------------------------------------------------------------------------
def water_with_digits(size, cx, cy, n, height, ink, bright, shade=None, changed=None, digit=.46, tint=.86, seed=3):
    """The pool: a real surface (height: the wave field), a mirror of the sky, printed with small numbers. All maps cover the same square.
    ink: how dark the numbers are in the mirror. bright: how far they turn to bare paper instead. shade: how far the mirror itself is dulled.
    The rows drift a little, like anything lying on water, and lean with the wave's slope, so they swim where a ring passes. changed: where a ring has passed the numbers are different ones."""
    y, x = np.mgrid[:n, :n].astype(np.float32)
    X, Y = (x / (n - 1) - .5) * size + cx, (y / (n - 1) - .5) * size + cy
    Z = height(X, Y)
    o = mesh_grid("pool", X, Y, Z)
    gy, gx = np.gradient(Z, size / n)
    me = o.data
    a = np.arange(n * n).reshape(n, n)
    quads = np.stack([a[:-1, :-1], a[:-1, 1:], a[1:, 1:], a[1:, :-1]], -1).ravel()
    tile = 64 * digit
    sw, sv = spectral(512, 3.4, seed), spectral(512, 3.4, seed + 1)
    uu = (X + .8 * lerp(sw, X * 3, Y * 3) + gx * .7) / tile
    vv = (Y + .55 * lerp(sv, X * 3, Y * 3) + gy * .7) / tile
    me.uv_layers["UV"].data.foreach_set("uv", np.stack([uu.ravel(), vv.ravel()], -1)[quads].ravel().astype(np.float32))
    uvw_ = me.uv_layers.new(name="UVW")
    uvw_.data.foreach_set("uv", np.stack([((X - cx) / size + .5).ravel(), ((Y - cy) / size + .5).ravel()], -1)[quads].ravel().astype(np.float32))

    m = bpy.data.materials.new("pool")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = .025
    uvm, uvw = nt.nodes.new("ShaderNodeUVMap"), nt.nodes.new("ShaderNodeUVMap")
    uvm.uv_map, uvw.uv_map = "UV", "UVW"

    def area(arr, nm):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = image_from(arr.astype(np.float32), nm)
        t.interpolation = "Linear"
        t.extension = "EXTEND"
        nt.links.new(uvw.outputs[0], t.inputs[0])
        return t.outputs[0]

    def math_(op, x, y):
        k = nt.nodes.new("ShaderNodeMath")
        k.operation = op
        for i, v in enumerate((x, y)):
            if isinstance(v, (int, float)):
                k.inputs[i].default_value = v
            else:
                nt.links.new(v, k.inputs[i])
        return k.outputs[0]

    dig = nt.nodes.new("ShaderNodeTexImage")
    im = image_from(tex("digits")[::-1], "digits")
    dig.image = im
    dig.interpolation = "Cubic"
    nt.links.new(uvm.outputs[0], dig.inputs[0])
    d_out = dig.outputs[0]
    if changed is not None:
        dig2, off, mx = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeVectorMath"), nt.nodes.new("ShaderNodeMix")
        dig2.image = im
        dig2.interpolation = "Cubic"
        off.operation = "ADD"
        off.inputs[1].default_value = (.37, .211, 0)
        nt.links.new(uvm.outputs[0], off.inputs[0])
        nt.links.new(off.outputs[0], dig2.inputs[0])
        nt.links.new(area(changed, "changed"), mx.inputs[0])
        nt.links.new(dig.outputs[0], mx.inputs[2])
        nt.links.new(dig2.outputs[0], mx.inputs[3])
        d_out = mx.outputs[0]
    # numbers are strong: lift their mask so even the faint ones print
    d_out = math_("MINIMUM", math_("MULTIPLY", d_out, 1.5), 1.0)
    col = math_("SUBTRACT", 1.0, math_("MULTIPLY", d_out, area(ink, "ink")))
    if shade is not None:
        col = math_("MULTIPLY", col, area(shade, "shade"))
    col = math_("MULTIPLY", col, tint)
    nt.links.new(col, b.inputs["Base Color"])
    b.inputs["Emission Color"].default_value = (1, 1, 1, 1)
    nt.links.new(math_("MULTIPLY", d_out, area(bright, "bright")), b.inputs["Emission Strength"])
    me.materials.append(m)
    far_water(tint)
    return o


def moon(eye, point, dist, radius, strength=4.0):
    """The moon, where a mirror at height 0 shows it at point when looked at from eye. It is only ever seen in the water. Returns its direction."""
    d = np.array([point[0] - eye[0], point[1] - eye[1], eye[2]], float)
    d = d / np.linalg.norm(d)
    c = np.array([eye[0], eye[1], -eye[2]], float) + d * dist
    n = 512
    yy, xx = np.mgrid[:n, :n].astype(np.float32) / (n - 1) * 2 - 1
    r = np.sqrt(xx * xx + yy * yy) * 1.1
    disc = sm((1.0 - r) / .03)
    mare = .90 + .10 * np.clip(lerp(spectral(512, 3.0, 8), xx * 90 + 256, yy * 90 + 256), -1.5, 1.5)
    x = np.array([[-1, 1], [-1, 1]], np.float32) * radius * 1.1
    z = np.array([[-1, -1], [1, 1]], np.float32) * radius * 1.1
    o = mesh_grid("moon", x, np.zeros((2, 2), np.float32), z)
    m = bpy.data.materials.new("moon")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    t, c_, e, tr, mix, out = (nt.nodes.new(k) for k in ("ShaderNodeTexImage", "ShaderNodeTexImage", "ShaderNodeEmission", "ShaderNodeBsdfTransparent", "ShaderNodeMixShader", "ShaderNodeOutputMaterial"))
    t.image = image_from(disc.astype(np.float32), "moon_a")
    c_.image = image_from((mare * disc).astype(np.float32), "moon_c")
    nt.links.new(c_.outputs[0], e.inputs["Color"])
    e.inputs["Strength"].default_value = strength
    nt.links.new(t.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(e.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    o.data.materials.append(m)
    o.location = tuple(c)
    o.rotation_euler = (math.atan2(c[2] - eye[2], c[1] - eye[1]), 0, 0)
    return tuple(d)


def waterfall(x, y, top, width):
    """The fall that fed the pool, from here a pale thread."""
    n = 256
    a = np.clip(.8 + .2 * lerp(spectral(512, 2.2, 77), np.broadcast_to(np.arange(1024, dtype=np.float32)[None, :] * .4, (n, 1024)), np.broadcast_to(np.arange(n, dtype=np.float32)[:, None] * .03, (n, 1024))), 0, 1)
    across = sm(np.minimum(np.arange(1024), np.arange(1024)[::-1]) / 260.0)[None, :]
    along = (sm(np.arange(n, dtype=np.float32) / 70.0) * sm((n - 1 - np.arange(n, dtype=np.float32)) / 30.0))[:, None]
    edge = np.broadcast_to(across * along, (n, 1024))
    xs, zs = np.array([[-1, 1], [-1, 1]], np.float32) * width / 2, np.array([[0, 0], [1, 1]], np.float32) * top
    o = mesh_grid("fall", xs + x, np.full((2, 2), y, np.float32), zs)
    m = bpy.data.materials.new("fall")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    t, ta, e, tr, mix, out = (nt.nodes.new(k) for k in ("ShaderNodeTexImage", "ShaderNodeTexImage", "ShaderNodeEmission", "ShaderNodeBsdfTransparent", "ShaderNodeMixShader", "ShaderNodeOutputMaterial"))
    t.image = image_from(a.astype(np.float32), "fall")
    ta.image = image_from(np.array(edge, np.float32), "fall_a")
    nt.links.new(t.outputs[0], e.inputs["Color"])
    e.inputs["Strength"].default_value = 1.2
    nt.links.new(ta.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(e.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    o.data.materials.append(m)
    return o


def ring_wave(d, a, wave=.85):
    """One ring on still water: a short steep front and a longer, weakening train behind it. d: distance from the crest, outward positive."""
    env = np.where(d > 0, np.exp(-(d / .5) ** 2), np.exp(-(d / 1.3) ** 2))
    return a * env * np.cos(2 * math.pi * d / wave)


def gorge(eye_x=0.0):
    """Three lines of mountains, near, middle and far, with the mist between them."""
    ridge_layer("ridge_far", [(-190, 400, 40, 70), (-70, 420, 48, 70), (60, 400, 42, 60), (180, 410, 50, 80)], 63, (-420, 420), (300, 500), 700, .74, .97, 44)
    ridge_layer("ridge_mid", [(-140, 245, 21, 44), (-52, 262, 26, 48), (40, 240, 20, 40), (130, 256, 25, 48)], 61, (-340, 340), (180, 320), 800, .50, .92, 24)
    ridge_layer("ridge_near", [(-122, 140, 10, 28), (-66, 150, 8, 24), (10, 142, 7, 22), (74, 136, 11, 28), (150, 150, 11, 30)], 62, (-270, 270), (105, 190), 900, .20, .86, 11)


EYE = (-.6, -7.0, 4.6)
HEART = (-2.1, 11.7)             # under the moon's reflection: the pulse comes up from below it


def shot_pool():
    t = opt("t", 7.5)            # the newest ring's radius; the one before it is at .62 of that
    sc = reset(samples=opt("samples", 128), pct=opt("pct", 100))
    sc.render.film_transparent = False
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    size, cy = 72.0, 24.0
    r_of = lambda X, Y: np.sqrt((X - HEART[0]) ** 2 + (Y - HEART[1]) ** 2)
    t2 = t * .62
    fine = spectral(1024, 2.7, 5)

    def height(X, Y):
        r = r_of(X, Y)
        return (ring_wave(r - t, opt("amp", .03)) + ring_wave(r - t2, .009, .8)) * sm((r - 2.6) / 1.0) + .0012 * sm((r - 2.6) / 4) * lerp(fine, X * 9, Y * 9)     # under the moon the water has gone still again

    nb = 2048
    by, bx = np.mgrid[:nb, :nb].astype(np.float32)
    BX, BY = (bx / (nb - 1) - .5) * size, (by / (nb - 1) - .5) * size + cy
    r = r_of(BX, BY)
    d = r - t
    crest = np.exp(-(d / .95) ** 2)
    wake = .55 * sm(-d / .5) * np.exp(np.minimum(d, 0) / 1.9)
    older = .5 * np.exp(-((r - t2) / .42) ** 2)
    turned = np.clip(crest + wake + older, 0, 1)                # what the ring has touched: the numbers go from ink to bare paper
    fade = np.clip(1.32 - np.sqrt((BX - EYE[0]) ** 2 + (BY - EYE[1]) ** 2) / 34, 0, 1)
    patch = np.clip(.86 + .2 * lerp(spectral(1024, 3.0, 9), BX * 10, BY * 10), .4, 1)
    ink = .96 * patch * fade * (1 - turned)
    bright = 1.5 * (crest + wake * .55 + older * .7) * fade
    shade = 1 - .62 * crest * fade - .28 * older * fade         # the ring's own band is dark, so the white numbers on it have the frame's hardest edge
    water_with_digits(size, 0, cy, opt("n", 2600), height, ink, bright, shade=shade, changed=(d < 0).astype(np.float32))
    gorge()
    waterfall(-11.0, 122.0, 15.0, 1.1)
    md = moon(EYE, HEART, 340.0, opt("moonr", 8.5), 6.0)
    sky_world(md, overhead=opt("over", .50), horizon=opt("horizon", .90), haze=opt("haze", .40), haze_deg=opt("hazedeg", 17.0), cloud=.16)
    sun((.4, .8, -.42), energy=1.0, angle=4)
    bpy.data.objects["sun"].visible_glossy = False             # it lights the rock; the only light in the water is the moon
    camera(EYE, (-.6, 13.0, .53), lens=28)
    render(fresh(os.path.join(OUT, opt("name", "pool"))), depth=False)


def shot_nihao():
    sc = reset(samples=opt("samples", 128), pct=opt("pct", 100))
    sc.render.film_transparent = False
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    size, cy = 90.0, 18.0
    tx, ty = opt("tx", -1.3), 8.3     # where the two characters float
    eye = (-.2, -6.0, 12.0)

    def height(X, Y):
        r = np.sqrt((X - tx) ** 2 + (Y - ty) ** 2)
        return ring_wave(r - 7.6, .0055, 1.0) * sm((r - 4) / 2)  # the very last ring, leaving

    nb = 2048
    by, bx = np.mgrid[:nb, :nb].astype(np.float32)
    BX, BY = (bx / (nb - 1) - .5) * size, (by / (nb - 1) - .5) * size + cy
    r = np.sqrt((BX - tx) ** 2 + (BY - ty) ** 2)
    # the numbers have drawn back from the two characters, and go out as they go
    ink = opt("dig", .42) * sm((r - 7.0) / 5.0) * np.clip(1.5 - r / 24, 0, 1) * np.clip(.8 + .3 * lerp(spectral(1024, 3.0, 9), BX * 10, BY * 10), .2, 1)
    water_with_digits(size, 0, cy, opt("n", 2200), height, ink, np.zeros_like(ink), digit=.40)

    a = tex("nihao")
    h_, w_ = a.shape
    wd, ht = opt("wd", 4.5), opt("wd", 4.5) * h_ / w_ * 1.28     # the water tilts away; stretched a little so the characters still stand upright to the eye
    xs = np.array([[-1, 1], [-1, 1]], np.float32) * wd / 2 + tx
    ys = np.array([[-1, -1], [1, 1]], np.float32) * ht / 2 + ty
    o = mesh_grid("nihao", xs, ys, np.full((2, 2), .012, np.float32))
    al = np.maximum(a, blur(a, 9.0) * .34)[::-1].astype(np.float32)   # wet ink on still water: a little of it has let go at the edges
    m = bpy.data.materials.new("nihao")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tt, e, tr, mix, out = (nt.nodes.new(k) for k in ("ShaderNodeTexImage", "ShaderNodeBsdfDiffuse", "ShaderNodeBsdfTransparent", "ShaderNodeMixShader", "ShaderNodeOutputMaterial"))
    tt.image = image_from(al, "nihao")
    tt.interpolation = "Cubic"
    e.inputs["Color"].default_value = (.004, .004, .004, 1)
    nt.links.new(tt.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(e.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    o.data.materials.append(m)

    md = moon(eye, (5.9, 19.2), 340.0, 10.0, opt("moon", 1.5))  # the water has gone so flat that the moon lies in it whole, far from the words
    sky_world(md, overhead=opt("over", .62), horizon=.90, haze=opt("haze", .30), haze_deg=opt("hazedeg", 30.0), cloud=opt("cloud", .05))
    camera(eye, (-.2, 8.3, 0), lens=24)
    render(fresh(os.path.join(OUT, opt("name", "nihao"))), depth=False)


# ---- the ink ------------------------------------------------------------------------------------------------------------------------------------
def glyph_solid(mask, height, depth, z0, x0=0.0, cells=420):
    """A character as a closed solid: the mask's cells extruded front to back, with a rough edge, which is where the ink starts to curl. Stair-stepped, which the simulation does not mind.
    The faces must wind outward: Blender decides what is inside by the normal of the first face a ray along +x meets, so inverted walls fill the counters and hollow the strokes."""
    step = max(1, mask.shape[0] // cells)
    h, w = mask.shape[0] // step, mask.shape[1] // step
    m = mask[:h * step, :w * step].reshape(h, step, w, step).mean((1, 3))
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    rough = lerp(spectral(512, 2.8, 13), xx * .9, yy * .9)
    b = blur(m, 1.2)
    m = b + .2 * rough * (b > .03) > opt("thin", .6)
    sz = height / h
    px = lambda i: (i - w / 2) * sz + x0
    pz = lambda j: (h / 2 - j) * sz + z0
    ids = {}
    verts, faces = [], []

    def v(i, j, k):
        key = (i, j, k)
        if key not in ids:
            ids[key] = len(verts)
            verts.append((px(i), -depth / 2 + depth * k, pz(j)))
        return ids[key]

    for j, i in zip(*np.nonzero(m)):
        faces.append((v(i, j + 1, 0), v(i + 1, j + 1, 0), v(i + 1, j, 0), v(i, j, 0)))
        faces.append((v(i, j, 1), v(i + 1, j, 1), v(i + 1, j + 1, 1), v(i, j + 1, 1)))
        if i == 0 or not m[j, i - 1]:
            faces.append((v(i, j + 1, 1), v(i, j + 1, 0), v(i, j, 0), v(i, j, 1)))
        if i == w - 1 or not m[j, i + 1]:
            faces.append((v(i + 1, j, 1), v(i + 1, j, 0), v(i + 1, j + 1, 0), v(i + 1, j + 1, 1)))
        if j == 0 or not m[j - 1, i]:
            faces.append((v(i, j, 1), v(i, j, 0), v(i + 1, j, 0), v(i + 1, j, 1)))
        if j == h - 1 or not m[j + 1, i]:
            faces.append((v(i + 1, j + 1, 1), v(i + 1, j + 1, 0), v(i, j + 1, 0), v(i, j + 1, 1)))
    me = bpy.data.meshes.new("glyph")
    me.from_pydata(verts, [], faces)
    me.update()
    return link(bpy.data.objects.new("glyph", me))


def blobs(dom, count, seed, rmin=.35, rmax=1.1):
    """Ink let go all through the pool: squashed closed spheres, heavier than the water. They fall, and each makes its own plumes."""
    r = np.random.default_rng(seed)
    verts, faces = [], []
    la, lo = 10, 16
    for _ in range(count):
        c = (r.uniform(-.46, .46) * dom[0], r.uniform(-.46, .46) * dom[1], r.uniform(.2, .96) * dom[2])
        rad = math.exp(r.uniform(math.log(rmin), math.log(rmax)))
        sq = (r.uniform(.8, 1.5), r.uniform(.8, 1.5), r.uniform(.45, 1.0))
        base = len(verts)
        for i in range(la + 1):
            th = -math.pi / 2 + math.pi * i / la
            for j in range(lo):
                ph = 2 * math.pi * j / lo
                verts.append((c[0] + rad * sq[0] * math.cos(th) * math.cos(ph), c[1] + rad * sq[1] * math.cos(th) * math.sin(ph), c[2] + rad * sq[2] * math.sin(th)))
        for i in range(la):
            for j in range(lo):
                j2 = (j + 1) % lo
                faces.append((base + i * lo + j, base + i * lo + j2, base + (i + 1) * lo + j2, base + (i + 1) * lo + j))
    me = bpy.data.meshes.new("blobs")
    me.from_pydata(verts, [], faces)
    me.update()
    return link(bpy.data.objects.new("blobs", me))


def cache_dir(ch, res):
    return os.path.join(CACHE, f"ink3d_{ch}_{res}")


def bake():
    """--ch=wo: the ink hangs in the water as a character for one instant, heavier than the water, and falls apart: strands sink, curl into mushrooms, and spread. Played backwards it gathers.
    --ch=mass: ink let go all through the pool, which is what fills 成精 and comes back in 十一."""
    ch, res, frames = opt("ch", "wo"), opt("res", 256), opt("frames", 120)
    sc = reset()
    dims = MASS if ch == "mass" else DOM
    me = bpy.data.meshes.new("dom")
    dx, dy, dz = dims
    me.from_pydata([(sx * dx / 2, sy * dy / 2, sz * dz) for sz in (0, 1) for sy in (-1, 1) for sx in (-1, 1)], [],
                   [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)])
    dom = link(bpy.data.objects.new("domain", me))
    m = dom.modifiers.new("fluid", "FLUID")
    m.fluid_type = "DOMAIN"
    ds = m.domain_settings
    ds.domain_type = "GAS"
    ds.resolution_max = res
    ds.alpha = opt("alpha", -1.1)        # negative: ink is heavier than water
    ds.beta = 0.0
    ds.vorticity = opt("vort", .35)
    ds.use_dissolve_smoke = False
    ds.time_scale = opt("ts", .6)
    ds.cfl_condition = 2.0
    ds.timesteps_max, ds.timesteps_min = 6, 1
    ds.use_adaptive_domain = False
    ds.use_noise = False
    ds.cache_type = "ALL"
    ds.cache_data_format = "OPENVDB"
    ds.openvdb_data_depth = "16"
    ds.cache_frame_start, ds.cache_frame_end = 1, frames
    ds.cache_directory = cache_dir(ch, res)
    if ch == "mass":
        g = blobs(MASS, opt("seeds", 64), 31)
    else:
        g = glyph_solid(tex("glyph_" + ch), opt("gh", 9.6), opt("gd", .55), DOM[2] * .53)
    f = g.modifiers.new("fluid", "FLUID")
    f.fluid_type = "FLOW"
    fs = f.flow_settings
    fs.flow_type, fs.flow_behavior, fs.flow_source = "SMOKE", "GEOMETRY", "MESH"
    fs.density = 1.0
    fs.temperature = 0.0
    fs.volume_density = 1.0
    fs.surface_distance = 1.0
    g.hide_render = True
    # stirred water, so the ink does not fall evenly
    for s, sz_, seed, loc in ((2.2, 1.1, 1, (0, 0, 5)), (1.2, .45, 2, (2, 0, 3))):
        fo = link(bpy.data.objects.new(f"stir{seed}", None))
        fo.location = loc
        with bpy.context.temp_override(object=fo, active_object=fo, selected_objects=[fo], selected_editable_objects=[fo]):
            bpy.ops.object.forcefield_toggle()
        fo.field.type = "TURBULENCE"
        fo.field.strength = s * opt("stir", 1.0)
        fo.field.size = sz_
        fo.field.seed = seed * 17
        fo.field.flow = 0.0
    bpy.context.view_layer.update()
    with bpy.context.temp_override(scene=sc, object=dom, active_object=dom, selected_objects=[dom], selected_editable_objects=[dom]):
        bpy.ops.fluid.bake_all()
    print("baked", ds.cache_directory, flush=True)


def vdb_path(ch, res, frame):
    return os.path.join(cache_dir(ch, res), "data", f"fluid_data_{frame:04d}.vdb")


def load_ink(path, location=(0, 0, 0), scale=1.0, name="ink"):
    """A density file in the tank: grid index 0 is the tank's lower corner, and the grid's own matrix already holds the cell size."""
    v = bpy.data.volumes.new(name)
    v.filepath = path
    o = link(bpy.data.objects.new(name, v))
    o.location = location
    o.scale = (scale, scale, scale)
    return o


def combed_density(ch, res, frame, turn, tail, dims, gain=1.0, sharp=1.0):
    """All the ink turns at once. The direction is the camera's axis, so seen from the camera every wisp is a line through one vanishing point: each voxel's ink is smeared
    toward the viewer with an exponential tail, then the whole field is bent a little so it reads as water and not as a lens effect. Fast enough to run per frame (a few seconds).
    gain: how much more ink there is than before the turn, 1 is exactly the same. sharp above 1 thins the thin ink and keeps the thick, so the wisps stay separate lines."""
    import openvdb
    grids, _ = openvdb.readAll(vdb_path(ch, res, frame))
    g = next(x for x in grids if x.name == "density")
    cell = max(dims) / res
    shape = tuple(int(round(d / cell)) for d in dims)
    a = np.zeros(shape, np.float32)
    g.copyToArray(a, ijk=(0, 0, 0))
    total = a.sum()
    a = np.clip(a, 0, None) ** sharp
    a *= total / max(a.sum(), 1e-6)
    decay = math.exp(-cell / max(tail * turn, 1e-3))
    keep = (1 - decay) * gain                                    # the tail's sum is keep / (1 - decay): this is what keeps the frame from filling with ink
    r = a.copy() * keep
    for j in range(shape[1] - 2, -1, -1):                        # from the far side to the near side: -y is toward the camera
        r[:, j, :] = a[:, j, :] * keep + decay * r[:, j + 1, :]
    r = a * (1 - turn) + r * turn
    # the bend
    rng = np.random.default_rng(23)
    xi, yi, zi = [np.arange(s, dtype=np.float32) * cell for s in shape]
    Xc = xi[:, None, None]
    Yc = yi[None, :, None]
    Zc = zi[None, None, :]
    near = (dims[1] - Yc) / dims[1]                              # the bend grows toward the camera
    dx = np.zeros(shape, np.float32)
    dz = np.zeros(shape, np.float32)
    for _ in range(4):
        k1, k2, ph1, ph2 = rng.uniform(.18, .6), rng.uniform(.18, .6), rng.uniform(0, 6.28), rng.uniform(0, 6.28)
        dx += np.sin(Xc * k1 + Zc * k2 + ph1) * (.5 * turn / 4) * 2
        dz += np.sin(Xc * k2 - Zc * k1 + ph2) * (.5 * turn / 4) * 2
    dx *= near ** 2
    dz *= near ** 2
    ix = np.clip((Xc + dx) / cell, 0, shape[0] - 1.001)
    iz = np.clip((Zc + dz) / cell, 0, shape[2] - 1.001)
    iy = np.broadcast_to(np.arange(shape[1], dtype=np.int32)[None, :, None], shape)
    i0, k0 = ix.astype(np.int32), iz.astype(np.int32)
    fx, fz = ix - i0, iz - k0
    out = (r[i0, iy, k0] * (1 - fx) * (1 - fz) + r[i0 + 1, iy, k0] * fx * (1 - fz) + r[i0, iy, k0 + 1] * (1 - fx) * fz + r[i0 + 1, iy, k0 + 1] * fx * fz)
    return out, g.transform


def combed(ch, res, frame, turn, tail, dims, gain=1.0, sharp=1.0):
    import openvdb
    out, transform = combed_density(ch, res, frame, turn, tail, dims, gain, sharp)
    path = os.path.join(cache_dir(ch, res), f"combed_{frame:04d}_{int(turn * 100):03d}_{int(tail * 10):03d}.vdb")
    o = openvdb.FloatGrid()
    o.copyFromArray(out.astype(np.float32), ijk=(0, 0, 0), tolerance=.002)
    o.name = "density"
    o.transform = transform
    openvdb.write(path, grids=[o])
    return path


def ink_material(density, albedo):
    m = bpy.data.materials.new("ink")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    a, p, c, out = nt.nodes.new("ShaderNodeVolumeInfo"), nt.nodes.new("ShaderNodeVolumePrincipled"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeOutputMaterial")
    p.inputs["Color"].default_value = (albedo, albedo, albedo, 1)
    p.inputs["Absorption Color"].default_value = (0, 0, 0, 1)
    p.inputs["Anisotropy"].default_value = .25
    c.operation = "MULTIPLY"
    c.inputs[1].default_value = density
    nt.links.new(a.outputs["Density"], c.inputs[0])
    nt.links.new(c.outputs[0], p.inputs["Density"])
    nt.links.new(p.outputs["Volume"], out.inputs["Volume"])
    return m


def ink_scene(sun_e, sun_dir=(-.2, .72, -.66)):
    """Ink alone, on nothing: the light behind it is painted in the grade, where it takes seconds to change. The film's alpha is how much of that light gets through."""
    sc = reset(samples=opt("samples", 96), pct=opt("pct", 100))
    sc.cycles.volume_bounces = 2
    sc.cycles.max_bounces = 4
    sc.cycles.use_denoising = opt("denoise", 1) > 0
    world = sc.world.node_tree.nodes["Background"]
    world.inputs[0].default_value = (1, 1, 1, 1)
    world.inputs[1].default_value = opt("amb", .08)
    sun(sun_dir, energy=sun_e, angle=10)
    return sc


CAM_INK = dict(loc=(.87, -14.0, 3.8), target=(.87, 0, 5.05), lens=28)
# the pool, looked into: +y lands at the frame's (864, 470), where the painted light comes down, so that is where the turned ink points
CAM_MASS = dict(loc=(0, -12.5, 5.6), target=(1.0, 0, 4.95), lens=24)


def shot_mass():
    """The plate for 成精, and for 十一: see the head of this file for its options."""
    res, frame, turn = opt("res", 420), opt("t", 52), opt("turn", 0.0)
    sc = ink_scene(opt("sun", 9.0), (-.15, .25, -1.0))
    path = combed("mass", res, frame, turn, opt("tail", 3.2), MASS, opt("gain", 1.0), opt("sharp", 1.0)) if turn > 0 else vdb_path("mass", res, frame)
    o = load_ink(path, (-MASS[0] / 2 + opt("x", 0.0), -MASS[1] / 2, opt("z", 0.0)))
    o.data.materials.append(ink_material(opt("dens", 7.0), opt("albedo", .55)))
    camera(**CAM_MASS)
    render(fresh(os.path.join(OUT, opt("name", "mass"))), depth=False)


def shot_probe():
    """What the turn lets through, by ray-marching the density in numpy, no Cycles: --tails=1.6,3.2 --gains=1,1.6 --sharps=1,1.6 [--w=640], one .npy each in jing/; tools' comp_ink puts it over the painted light."""
    res, frame, dens = opt("res", 420), opt("t", 52), opt("dens", 7.0)
    cell = max(MASS) / res
    w = opt("w", 640)
    h = w * 9 // 16
    loc, target = np.array(CAM_MASS["loc"], float), np.array(CAM_MASS["target"], float)
    fw = (target - loc) / np.linalg.norm(target - loc)
    right = np.cross(fw, (0, 0, 1))
    right /= np.linalg.norm(right)
    up = np.cross(right, fw)
    f = CAM_MASS["lens"] / 36.0
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    u, v = (xs + .5) / w - .5, (.5 - (ys + .5) / h) * h / w
    d = fw[None, None] * f + right[None, None] * u[..., None] + up[None, None] * v[..., None]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    ts = np.linspace(4.0, 24.0, 220)
    for tail in str(opt("tails", "3.2")).split(","):
        for gain in str(opt("gains", "1")).split(","):
            for sharp in str(opt("sharps", "1")).split(","):
                rho, _ = combed_density("mass", res, frame, opt("turn", 1.0), float(tail), MASS, float(gain), float(sharp))
                tau = np.zeros((h, w), np.float32)
                for t in ts:
                    p = loc[None, None] + d * t
                    i = np.floor((p[..., 0] + MASS[0] / 2) / cell).astype(int)
                    j = np.floor((p[..., 1] + MASS[1] / 2) / cell).astype(int)
                    k = np.floor(p[..., 2] / cell).astype(int)
                    inside = (i >= 0) & (i < rho.shape[0]) & (j >= 0) & (j < rho.shape[1]) & (k >= 0) & (k < rho.shape[2])
                    tau[inside] += rho[i[inside], j[inside], k[inside]]
                np.save(os.path.join(OUT, f"probe_{tail}_{gain}_{sharp}.npy"), np.exp(-tau * dens * (ts[1] - ts[0])))
                print("probe", tail, gain, sharp, flush=True)


def shot_wo():
    """The character's plate: the sim frame where it holds, a later frame thin around it for the ink still on its way in, and the pool's ink thin behind."""
    res, frame = opt("res", 512), opt("t", 26)
    sc = ink_scene(opt("sun", 9.0))
    o = load_ink(vdb_path("wo", res, frame), (-DOM[0] / 2, -DOM[1] / 2, 0))
    o.data.materials.append(ink_material(opt("dens", 22.0), opt("albedo", .3)))
    veil = opt("veilt", 0)
    if veil:
        o2 = load_ink(vdb_path("wo", res, veil), (-DOM[0] / 2, -DOM[1] / 2, 0), name="veil")
        o2.data.materials.append(ink_material(opt("veild", 2.5), .45))
    camera(**CAM_INK)
    render(fresh(os.path.join(OUT, opt("name", "wo"))), depth=False)


def shot_back():
    """The pool's ink behind 我: thin, large, far, so the water is not empty and the character is seen to have come out of it."""
    sc = ink_scene(opt("sun", 11.0))
    o = load_ink(vdb_path("mass", opt("res", 420), opt("t", 44)), (-MASS[0] / 2 * 1.5 + 1.0, 6.0, -5.0), scale=1.5)
    o.data.materials.append(ink_material(opt("dens", 2.4), .7))
    camera(**CAM_INK)
    render(fresh(os.path.join(OUT, opt("name", "back"))), depth=False)


SHOT = opt("shot", "pool")
if SHOT == "bake":
    bake()
elif SHOT == "pool":
    shot_pool()
elif SHOT == "nihao":
    shot_nihao()
elif SHOT == "mass":
    shot_mass()
elif SHOT == "probe":
    shot_probe()
elif SHOT == "wo":
    shot_wo()
elif SHOT == "back":
    shot_back()
