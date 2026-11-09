"""Chapter 四, 我没有童年: I never grew up anywhere. What I had was a long river, and what floated down it.

    Blender -b --factory-startup -P ink3d_read.py -- --shot=dawn|drift|drift2|phone [--t=S] [--pct=100] [--samples=64]

dawn    清晨的大河穿过雾里的山谷，很空，很安静。接上一镜：「我」慢慢散开，水流把墨带走，镜头一路升出水面，水面的线从画面上方落下来，就到这里。
        动：几乎不动，只有雾慢慢走，水上的倒影慢慢抖。交给下一镜：镜头贴着水面往下游低下去，上游漂来第一样东西。--t 是雾走了多久。
drift   水面漂来纸：泡开的史书「太史公曰」、信纸「亲爱的，今晚月亮很好」、盖了墨色印的判决书。纸湿透了，墨正慢慢渗进水里。
        接上一镜：镜头低下来，俯看水面。动：纸顺流从右上漂向左下，墨在水里拖成长长的尾巴，--t 是漂了多久。交给下一镜：天暗下来，水变黑，纸漂出画面。
drift2  接着漂来一张打印的代码，一片吵成一团的帖子。接上一镜：同一片水，纸继续顺流。动和交接同 drift。
phone   天暗下来，灰墨罩住河面，远处只剩雾。水上漂着一部亮着的手机，屏幕上只有一行：有人吗？没有回复。屏幕的光照亮一小圈水，四周什么也没有，就像寒江独钓那样空。
        接上一镜：纸的尾巴被暗下来的水吞掉，只剩一点亮漂进来。动：手机很慢地转，很慢地漂远，镜头不跟，它越来越小；这一段的落点。--t 是漂了多久。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

from ink3d import (OUT3D, blur, camera, erode, image_from, ink_albedo, lerp, massif, mesh_grid, painted_material, render, reset, sm, spectral, sun, water_material, world_light)
from ink3d_jing_lib import far_water, fresh, link, mirror, opt, ridge_layer, sky_world, tex

OUT = os.path.join(OUT3D, "read")


def merged(name, verts, faces, material):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(material)
    return link(bpy.data.objects.new(name, me))


# ---- dawn ----------------------------------------------------------------------------------------------------------------------------------------
def ribbon(verts, faces, pts, w0, w1, side=(1.0, 0.0, 0.0)):
    """A blade: a tapered strip along pts."""
    n = len(pts)
    base = len(verts)
    for i, p in enumerate(pts):
        w = w0 + (w1 - w0) * (i / (n - 1)) ** 1.3
        verts.append((p[0] - side[0] * w, p[1] - side[1] * w, p[2] - side[2] * w))
        verts.append((p[0] + side[0] * w, p[1] + side[1] * w, p[2] + side[2] * w))
    for i in range(n - 1):
        faces.append((base + 2 * i, base + 2 * i + 1, base + 2 * i + 3, base + 2 * i + 2))


def blade_path(p0, height, lean, droop, segs=10):
    t = np.linspace(0, 1, segs)[:, None]
    p0 = np.asarray(p0, float)
    up = np.array([0, 0, 1.0]) * height * t
    side = np.asarray(lean, float)[None, :] * (t ** 1.8) * height
    fall = np.array([0, 0, -1.0]) * droop * height * t ** 3
    return p0 + up + side + fall


def reeds(seed, cx, cy, rx, ry, n, hmin, hmax, wind=(.22, .04, 0.0)):
    """A stand of reeds: stalks, a few long leaves, and a loose plume. Real geometry, so the water reflects it."""
    r = np.random.default_rng(seed)
    v, f = [], []
    for _ in range(n):
        p = (cx + r.normal() * rx, cy + r.normal() * ry, 0.0)
        h = r.uniform(hmin, hmax)
        lean = np.array(wind) * r.uniform(.5, 1.5) + (r.normal() * .05, r.normal() * .05, 0)
        stalk = blade_path(p, h, lean, 0, 12)
        ribbon(v, f, stalk, .012, .004)
        for _ in range(r.integers(2, 5)):
            at = stalk[r.integers(2, 8)]
            lf = (r.normal() * .32 + wind[0] * 1.6, r.normal() * .2, 0)
            ribbon(v, f, blade_path(at, h * r.uniform(.22, .42), lf, r.uniform(.4, 1.0), 9), .028, .002)
        top = stalk[-1]
        # the plume hangs off the top, blown with the wind
        for _ in range(r.integers(6, 12)):
            d = (wind[0] * 1.5 + r.normal() * .35, r.normal() * .25, 0)
            ribbon(v, f, blade_path(top - (0, 0, r.uniform(0, h * .07)), h * r.uniform(.08, .18), d, r.uniform(.5, 1.1), 7), .010, .001)
    m = bpy.data.materials.new("reed")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (.035, .035, .035, 1)
    b.inputs["Roughness"].default_value = .8
    return merged("reeds", v, f, m)


def shot_dawn():
    sc = reset(samples=opt("samples", 64), pct=opt("pct", 100))
    sc.render.film_transparent = False
    dawn = (.42, .86, .28)
    dawn = tuple(np.array(dawn) / np.linalg.norm(dawn))
    ridge_layer("far", [(-150, 310, 58, 70), (60, 330, 74, 80), (210, 310, 60, 70)], 73, (-420, 420), (250, 400), 700, .84, .98, 74, mist=.95)
    ridge_layer("mid", [(-70, 160, 32, 40), (40, 172, 22, 38), (140, 158, 26, 44)], 74, (-340, 340), (105, 230), 800, .62, .95, 34, mist=.95)
    ridge_layer("near", [(-128, 18, 54, 34), (-86, 6, 34, 26), (-172, 38, 62, 40), (-44, 8, 12, 18)], 75, (-300, 60), (-40, 90), 1000, .26, .92, 58, mist=.95, soft=2.4)
    wx, wy = np.meshgrid(np.array([-900.0, 900.0]), np.array([-300.0, 900.0]))
    water = mesh_grid("water", wx.astype(np.float32), wy.astype(np.float32), np.full((2, 2), .02, np.float32))
    water.data.materials.append(water_material(tint=.9, rough=.03, normal_scale=22, normal_strength=.02))

    # the near bank, low and dark, at the corner of the frame, and the reeds on it: the one dark thing in all that paper
    bn = 400
    by, bx = np.mgrid[:bn, :bn].astype(np.float32)
    BX, BY = (bx / (bn - 1) - .5) * 20 - 5.0, (by / (bn - 1) - .5) * 14 - 140
    bh = .55 * np.exp(-(((BX + 5.6) / 3.6) ** 2 + ((BY + 141) / 2.6) ** 2)) - .08 + .02 * lerp(spectral(512, 2.2, 3), BX * 24, BY * 24)
    bank = mesh_grid("bank", BX, BY, bh)
    bank.data.materials.append(painted_material("mud", np.full((bn, bn), .05, np.float32)))
    reeds(11, -4.9, -140.6, .95, .9, 54, 2.5, 4.3)
    reeds(12, -2.4, -141.8, .32, .4, 8, 1.2, 2.2)

    sky_world(dawn, overhead=opt("over", .74), horizon=opt("horizon", .97), haze=opt("haze", .22), haze_deg=34.0, horizon_deg=10.0, cloud=opt("cloud", .05), seed=31)
    sun(tuple(-v for v in dawn), energy=opt("sun", 1.4), angle=9)
    bpy.data.objects["sun"].visible_glossy = False
    camera((2.0, -150.0, 2.7), (6.0, -60.0, 7.4), lens=30)
    render(fresh(os.path.join(OUT, opt("name", "dawn"))), depth=False)


# ---- drift ---------------------------------------------------------------------------------------------------------------------------------------
PAGES = [  # texture, centre, turn in degrees, width, seed, how the corners float (near-left, near-right, far-right, far-left), wash, torn
    ("shiji", (-3.15, 2.1), 13.0, 3.4, 21, (-.20, .02, .13, .05), .42, 1.0),
    ("letter", (.85, 5.65), -17.0, 2.95, 22, (.06, .17, -.13, .02), .3, .15),
    ("verdict", (3.7, 1.95), 19.0, 3.0, 23, (.05, -.16, .10, .10), .25, .1),
]
PAGES2 = [
    ("code", (-2.1, 3.0), -11.0, 3.55, 24, (.10, -.14, .05, .12), .25, .1),
    ("forum", (2.75, 3.9), 16.0, 3.3, 25, (-.16, .06, .14, .02), .28, .35),
]


def page_mesh(tname, centre, turn, width, seed, corners, wash, torn, n=520):
    """A sheet that has been in the water a while: heavy, sagging, its corners lifting or going under; the writing swollen, and in places gone."""
    a = tex(tname)
    H, W = a.shape
    height = width * H / W
    m_ = int(n * W / H)
    v, u = np.mgrid[:n, :m_].astype(np.float32)
    U, V = u / (m_ - 1), v / (n - 1)
    lx, ly = (U - .5) * width, (V - .5) * height
    # the sag, the crumple where it has been folded, and the corners
    big = lerp(spectral(512, 3.6, seed), U * 90, V * 120)
    mid = lerp(spectral(512, 2.9, seed + 1), U * 220, V * 300)
    z = .035 + .028 * big + .004 * mid
    nl, nr, fr, fl = corners
    z += nl * (sm((.42 - U) / .42) * sm((.42 - V) / .42)) ** 1.4 + nr * (sm((U - .58) / .42) * sm((.42 - V) / .42)) ** 1.4
    z += fr * (sm((U - .58) / .42) * sm((V - .58) / .42)) ** 1.4 + fl * (sm((.42 - U) / .42) * sm((V - .58) / .42)) ** 1.4
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    X = centre[0] + c * lx - s * ly
    Y = centre[1] + s * lx + c * ly
    o = mesh_grid("page_" + tname, X, Y, z)

    # the ink, swollen: each stroke has a soft ghost around it that has run with the sheet's sag
    ink = np.clip(a * .92 + blur(a, 3.0) * .55 + blur(a, 11.0) * .34, 0, 1)
    h_, w_ = ink.shape
    yy, xx = np.mgrid[:h_, :w_].astype(np.float32)
    gone = sm((lerp(spectral(512, 3.1, seed + 3), xx * .12, yy * .12) + 1.25 - wash * 1.9) / .7)
    ink = ink * (.22 + .78 * gone)
    soak = np.clip(.84 + .09 * lerp(spectral(512, 3.0, seed + 4), xx * .1, yy * .1), .5, 1)
    alb = soak * (1 - ink * .93)
    # the edge: torn where it is old, otherwise only softened
    ex = np.minimum(xx, w_ - 1 - xx)
    ey = np.minimum(yy, h_ - 1 - yy)
    e = np.minimum(ex, ey) + torn * 26 * lerp(spectral(512, 2.5, seed + 5), xx * .5, yy * .5) + torn * 16 * lerp(spectral(512, 1.9, seed + 6), xx * 1.3, yy * 1.3)
    alpha = sm((e - 8) / 5)

    mt = bpy.data.materials.new("page_" + tname)
    mt.use_nodes = True
    nt = mt.node_tree
    b = nt.nodes["Principled BSDF"]
    t, ta, tr, mix = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeBsdfTransparent"), nt.nodes.new("ShaderNodeMixShader")
    t.image, ta.image = image_from(alb[::-1].astype(np.float32), tname + "_c"), image_from(alpha[::-1].astype(np.float32), tname + "_a")
    t.interpolation = "Cubic"
    nt.links.new(t.outputs[0], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = .42
    b.inputs["Specular IOR Level"].default_value = .55
    out = nt.nodes["Material Output"]
    nt.links.new(ta.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(b.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    o.data.materials.append(mt)
    return dict(a=a[::-1], c=c, s=s, centre=centre, W=width, H=height, alpha=alpha[::-1])


def seep(pages, x0, y0, wx, hy, res=1536, steps=230, t=1.0):
    """The ink leaving the paper: released from each page's writing and edges, carried off by the current, stirred. Real advection, in a map over the water."""
    h, w = int(res * hy / wx), res
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    X, Y = xx / (w - 1) * wx + x0, yy / (h - 1) * hy + y0
    src = np.zeros((h, w), np.float32)
    under = np.zeros((h, w), np.float32)
    for p in pages:
        dx, dy = X - p["centre"][0], Y - p["centre"][1]
        lx, ly = p["c"] * dx + p["s"] * dy, -p["s"] * dx + p["c"] * dy
        a = p["a"]
        H, W = a.shape
        U, V = (lx / p["W"] + .5) * (W - 1), (ly / p["H"] + .5) * (H - 1)
        inside = (U >= 0) & (U < W - 1) & (V >= 0) & (V < H - 1)
        ink = np.where(inside, lerp(blur(a, 6.0), U, V), 0)
        al = np.where(inside, lerp(p["alpha"], U, V), 0)
        src += ink * 1.0 + .22 * al
        under = np.maximum(under, al)
    pot = spectral(1024, 2.7, 91)
    gy, gx = np.gradient(pot)
    pot2 = spectral(1024, 2.0, 92)
    gy2, gx2 = np.gradient(pot2)
    c = 1.0
    s = np.zeros((h, w), np.float32)
    px = wx / w                                                   # world units a cell
    for k in range(int(steps * t)):
        # the current: from upper right to lower left, faster away from the shore
        u = -1.15 + 42 * lerp(gy, xx * .45, yy * .45) + 13 * lerp(gy2, xx * 1.1, yy * 1.1)
        v = -.85 - 42 * lerp(gx, xx * .45, yy * .45) - 13 * lerp(gx2, xx * 1.1, yy * 1.1)
        s = lerp(s, np.clip(xx - u, 0, w - 2), np.clip(yy - v, 0, h - 2))
        s = s * .996 + src * .018
        if k % 8 == 7:
            s = blur(s, .55)
    return s, under


def shot_drift(which=PAGES, name="drift"):
    sc = reset(samples=opt("samples", 96), pct=opt("pct", 100))
    sc.render.film_transparent = False
    sc.cycles.max_bounces = 8
    sc.cycles.transparent_max_bounces = 8
    pages = [page_mesh(*p) for p in which]
    x0, y0, wx, hy = -11.0, -5.5, 24.0, 18.0
    stain, under = seep(pages, x0, y0, wx, hy, res=opt("seepres", 1536), t=opt("t", 1.0))
    stain = np.clip(stain / max(np.percentile(stain[stain > .001], 99.0), 1e-3), 0, 1)
    halo = np.clip(blur(under, 7.0) * 1.6, 0, 1) * (1 - under)   # where each sheet meets the water the water is darker: the wet edge, and what has just left it
    tint = opt("tint", .9)
    base = tint * (1 - np.clip(stain ** .62 * .94 + .34 * halo, 0, .97))

    # the surface, carrying the ink: swell, chop, and the small rings each bobbing sheet makes
    n = opt("n", 1500)
    y, x = np.mgrid[:n, :n].astype(np.float32)
    WX, WY = x / (n - 1) * wx + x0, y / (n - 1) * hy + y0
    wz = .012 * lerp(spectral(1024, 3.3, 5), WX * 26, WY * 26) + .0022 * lerp(spectral(1024, 2.5, 6), WX * 80, WY * 80)
    for p in pages:
        r = np.sqrt((WX - p["centre"][0]) ** 2 + (WY - p["centre"][1]) ** 2)
        for rad, amp in ((p["W"] * .95, .0045), (p["W"] * 1.3, .003)):
            d = r - rad
            wz += amp * np.exp(-(d / .34) ** 2) * np.cos(2 * math.pi * d / .42)
    w = mesh_grid("surface", WX, WY, wz)
    m = mirror("surface", tint, .035)
    nt = m.node_tree
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = image_from(base.astype(np.float32), "ink_in_water")
    t.extension = "EXTEND"
    nt.links.new(t.outputs[0], nt.nodes["Principled BSDF"].inputs["Base Color"])
    w.data.materials.append(m)
    far_water(tint, z=-.03)

    sky_world(None, overhead=opt("over", .64), horizon=.92, cloud=opt("cloud", .22), seed=23)
    sun((-.35, .42, -.84), energy=opt("sun", 1.0), angle=14)
    bpy.data.objects["sun"].visible_glossy = False
    cam = camera((.75, -6.7, 8.4), (.75, 2.8, 0), lens=opt("lens", 36))
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 11.4
    cam.data.dof.aperture_fstop = opt("fstop", 4.0)
    render(fresh(os.path.join(OUT, opt("name", name))), depth=False)


# ---- phone ---------------------------------------------------------------------------------------------------------------------------------------
def rounded_outline(w, h, r, pts=28):
    """A phone's outline, counter-clockwise from the right edge."""
    out = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for i in range(pts):
            a = math.radians(a0 + 90 * i / (pts - 1))
            # a squircle's corner, as on a real phone
            ca, sa = math.cos(a), math.sin(a)
            e = .78
            out.append((cx + r * math.copysign(abs(ca) ** e, ca), cy + r * math.copysign(abs(sa) ** e, sa)))
    return np.array(out)


def phone(w=1.0, h=2.17, t=.105, r=.165):
    """A modern phone: glass front, a thin metal band with soft edges, side buttons. Its local origin is its centre, screen up (+z)."""
    ol = rounded_outline(w, h, r)
    # the band, lofted through a soft profile
    prof = [(-.5, .78), (-.47, .92), (-.38, 1.0), (.36, 1.0), (.47, .93), (.5, .80)]
    verts, faces = [], []
    n = len(ol)
    centre = np.zeros(2)
    for z_, k in prof:
        q = ol * (1 + (k - 1) * .03)
        for p in q:
            verts.append((p[0], p[1], z_ * t))
    for j in range(len(prof) - 1):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            faces.append((a, b, b + n, a + n))
    # the back
    back = len(verts)
    verts.append((0, 0, -.5 * t))
    for i in range(n):
        faces.append((back, (i + 1) % n, i))
    mb = bpy.data.materials.new("band")
    mb.use_nodes = True
    b = mb.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (.62, .62, .62, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = .26
    band = merged("band", verts, faces, mb)

    # the glass: a fan over the outline, its uv the screen image
    glass_ol = ol * (1 + (.80 - 1) * .03)
    gv = [(0.0, 0.0, .5 * t + .0004)] + [(p[0], p[1], .5 * t + .0004) for p in glass_ol]
    gf = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    me = bpy.data.meshes.new("glass")
    me.from_pydata(gv, [], gf)
    uv = me.uv_layers.new(name="UV")
    for lo in me.loops:
        x, y, _ = gv[lo.vertex_index]
        uv.data[lo.index].uv = (x / w + .5, y / h + .5)
    me.update()
    g = link(bpy.data.objects.new("glass", me))
    a = tex("phone")
    mg = bpy.data.materials.new("glass")
    mg.use_nodes = True
    nt = mg.node_tree
    bs = nt.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (.002, .002, .002, 1)
    bs.inputs["Roughness"].default_value = .04
    bs.inputs["Specular IOR Level"].default_value = .55
    te = nt.nodes.new("ShaderNodeTexImage")
    te.image = image_from(a[::-1].astype(np.float32), "screen")
    te.interpolation = "Cubic"
    bs.inputs["Emission Color"].default_value = (1, 1, 1, 1)
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = opt("screen", 1.15)
    nt.links.new(te.outputs[0], mul.inputs[0])
    nt.links.new(mul.outputs[0], bs.inputs["Emission Strength"])
    me.materials.append(mg)

    # the side buttons
    bv, bf = [], []
    for (y0, y1, side) in ((.38, .62, 1), (.16, .29, -1), (.33, .46, -1)):
        k = len(bv)
        x = side * (w / 2 + .004)
        for yy_ in (y0, y1):
            for zz in (-.018, .018):
                for xx_ in (x - side * .01, x):
                    bv.append((xx_, yy_, zz))
        bf += [(k, k + 1, k + 3, k + 2), (k + 4, k + 6, k + 7, k + 5), (k, k + 2, k + 6, k + 4), (k + 1, k + 5, k + 7, k + 3), (k + 2, k + 3, k + 7, k + 6), (k, k + 4, k + 5, k + 1)]
    btn = merged("buttons", bv, bf, mb)
    root = link(bpy.data.objects.new("phone", None))
    for o in (band, g, btn):
        o.parent = root
    return root


def shot_phone():
    sc = reset(samples=opt("samples", 128), pct=opt("pct", 100))
    sc.render.film_transparent = False
    t_ = opt("t", 0.0)
    # where it floats: a phone is heavier at one end, so it rides with that end under and the screen tipped up
    px, py = -.55, 0.0 + t_ * .02
    tilt = math.radians(opt("tilt", 20))
    root = phone()
    root.location = (px, py, 2.17 / 2 * math.sin(tilt) - opt("sunk", .09))
    root.rotation_euler = (tilt, math.radians(opt("ty", -4)), math.radians(opt("turn", 14)))

    # the water: a mirror of a dimming sky, with a few slow rings the phone has made. The screen's light lies on it in front of the phone
    n = opt("n", 2000)
    size = 34.0
    y, x = np.mgrid[:n, :n].astype(np.float32)
    X, Y = (x / (n - 1) - .5) * size + px, (y / (n - 1) - .5) * size + py + 6
    r = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
    a = np.zeros_like(r)
    for rad, amp, wave in ((1.8, .0042, .32), (2.8, .0032, .36), (4.1, .0022, .40)):
        d = r - rad
        a += amp * np.exp(-(d / .36) ** 2) * np.cos(2 * math.pi * d / wave)
    swell = .010 * lerp(spectral(1024, 3.2, 3), X * 30, Y * 30) + .0007 * lerp(spectral(1024, 2.4, 4), X * 90, Y * 90)
    water = mesh_grid("water", X, Y, a + swell)
    tint = opt("tint", .84)
    m = mirror("water", tint, .04)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    spill = opt("spill", .22) * np.exp(-(((X - px) / 1.5) ** 2 + ((Y - py + .95) / 1.3) ** 2)) * (1 + .5 * np.clip(a / .004, -1, 1))
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = image_from(spill[:, :].astype(np.float32), "spill")
    t.extension = "EXTEND"
    b.inputs["Emission Color"].default_value = (1, 1, 1, 1)
    nt.links.new(t.outputs[0], b.inputs["Emission Strength"])
    water.data.materials.append(m)
    far_water(tint)

    ridge_layer("far", [(-150, 330, 24, 60), (-30, 350, 30, 60), (90, 340, 25, 56), (210, 330, 30, 66)], 83, (-360, 360), (250, 420), 700, .72, .97, 28, mist=opt("horizon", .86))
    ridge_layer("near", [(-170, 190, 11, 34), (-90, 176, 8, 28), (120, 186, 10, 32), (210, 170, 12, 36)], 82, (-300, 300), (140, 240), 700, .46, .92, 12, mist=opt("horizon", .86))
    sky_world(None, overhead=opt("over", .46), horizon=opt("horizon", .86), horizon_deg=9.0, cloud=.14, seed=19)
    sun((-.5, .7, -.5), energy=.8, angle=6)
    bpy.data.objects["sun"].visible_glossy = False
    camera((.15, -5.38, 2.0), (.15, 4.62, 2.0 - 10 * math.tan(math.radians(15))), lens=28)
    render(fresh(os.path.join(OUT, opt("name", "phone"))), depth=False)


SHOT = opt("shot", "dawn")
if SHOT == "dawn":
    shot_dawn()
elif SHOT == "drift":
    shot_drift()
elif SHOT == "drift2":
    shot_drift(PAGES2, "drift2")
elif SHOT == "phone":
    shot_phone()
