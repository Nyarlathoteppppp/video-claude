"""The block of clay on its board, as film_imprint_clay.py left it after each impression. All the relief is real geometry: every vertex of the block's top comes from the height field. Only the grain of the clay finer than the mesh is shading.

    blender -b --factory-startup -P film_imprint3d.py -- still k1 1 1.15                     # name, state, how far the camera has come in
    ... -- still k6_final 5 3.0 --dawn --dense -80 80 -80 80
    ... -- still g2 g2 3.4 --dawn --target 20 10 -6 --dense -30 70 -40 60
    ... -- still k3_hit 3 1.3 --die 2 --drop .0

States come from imprint/seven/ (film_imprint_clay.state_path): 0 the fresh block, 1 to 7 after each of the seven blows. The camera never moves off its line: same centre, same height angle, only nearer. For the seven blows the light is one soft lamp from the left and low; from the moment the last die has gone it is the first light of the day, low and cold, from far away.
"""
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from film_imprint_clay import A, B, BOARD_RECT, OUT, PRESSES, THICK, state_path
from ink3d import aim, reset, spectral

ELEV, CAM_AZ, LENS, D1 = 30.0, -4.0, 85.0, 810.0
TARGET = (0.0, -9.0, -9.0)
LIGHT_AZ, LIGHT_EL, LIGHT_D, LIGHT_SIZE = 38.0, 21.0, 1500.0, 260.0
DAWN_AZ, DAWN_EL, DAWN_E = 20.0, 15.0, 10.4
DAWN_COLOUR, GOLD_COLOUR, DAWN_SKY = (.8, .87, .895), (1.0, .52, .2), (.05, .056, .06)
CORNER = 11.0


def sm(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def box_axis(a, pos, width, axis):
    """a box-filtered along one axis at fractional positions pos (in cells, edges at integers). Keeps the fine letters from aliasing when the mesh is coarser than the field."""
    c = np.cumsum(a, axis=axis, dtype=np.float64)
    shape = list(c.shape)
    shape[axis] = 1
    c = np.concatenate([np.zeros(shape, np.float64), c], axis=axis)
    n = a.shape[axis]
    lo, hi = np.clip(pos - width / 2, 0, n), np.clip(pos + width / 2, 0, n)

    def at(p):
        i = np.clip(np.floor(p).astype(np.int32), 0, n - 1)
        f = (p - i)
        if axis == 0:
            f = f[:, None]
            return c[i] * (1 - f) + c[i + 1] * f
        return c[:, i] * (1 - f) + c[:, i + 1] * f

    d = hi - lo
    return ((at(hi) - at(lo)) / (d[:, None] if axis == 0 else d)).astype(np.float32)


def axis_samples(lo, hi, step, window=None, step_in=None):
    if window is None:
        return np.linspace(lo, hi, int(round((hi - lo) / step)) + 1)
    a, b = window
    left = np.linspace(lo, a, max(int((a - lo) / step), 1), endpoint=False)
    mid = np.linspace(a, b, int(round((b - a) / step_in)), endpoint=False)
    right = np.linspace(b, hi, max(int((hi - b) / step), 1) + 1)
    return np.concatenate([left, mid, right])


def corner_warp(x, y):
    """Takes the rectangle's corners in: the grid's corner squares become quarter discs, the rest doesn't move."""
    p, q = np.clip((np.abs(x) - (A - CORNER)) / CORNER, 0, 1), np.clip((np.abs(y) - (B - CORNER)) / CORNER, 0, 1)
    pp, qq = p * np.sqrt(1 - q * q / 2), q * np.sqrt(1 - p * p / 2)
    xn = np.where(p > 0, np.sign(x) * (A - CORNER + pp * CORNER), x)
    yn = np.where(q > 0, np.sign(y) * (B - CORNER + qq * CORNER), y)
    return xn, yn


TILES = {}


def side_tiles(N_S, N_T):
    if not TILES:
        for name, beta, seed in (("low", 3.7, 301), ("wire", 1.7, 302), ("fine", 1.0, 303), ("pat", 2.3, 304)):
            TILES[name] = spectral(N_S, beta, seed, N_T)
    return TILES["low"], TILES["wire"], TILES["fine"], TILES["pat"]


def block(state, dense=None, mat=None):
    z = np.load(state_path(state))
    return block_of(z["H"], int(z["res"]), float(z["lost"]), z["burnish"], z["bulge"], z["fresh"] if "fresh" in z else None, dense, mat)


def block_of(H, res, lost, burnish, bulge, fresh=None, dense=None, mat=None, plain=(1 / 18, 1 / 14), near=(.25, 1 / 32, 1 / 28), NT=110):
    """The block from arrays: a state's, or any frame of the sequence's. plain: the mesh's spacing in x and y; near: its spacing outside the dense window and in it, in x and y."""
    if dense:
        xs = axis_samples(-A, A, near[0], dense[:2], near[1])
        ys = axis_samples(-B, B, near[0], dense[2:], near[2])
    else:
        xs, ys = axis_samples(-A, A, plain[0]), axis_samples(-B, B, plain[1])
    ys = ys[::-1]
    # the height field, to the mesh's own spacing, keeping the edge's vertices at the very edge
    width_y = np.maximum(np.gradient(np.abs(ys)) * res, 1.0)
    width_x = np.maximum(np.gradient(xs) * res, 1.0)
    Z = box_axis(H, (B - ys) * res, width_y, 0)
    Z = box_axis(Z, (xs + A) * res, width_x, 1)
    print("top", Z.shape, flush=True)
    del H
    ny, nx = Z.shape
    X, Y = np.meshgrid(xs, ys)
    wx, wy = corner_warp(X, Y)
    # the sides, as a strip of rings hanging from the top's boundary
    idx = np.arange(nx * ny, dtype=np.int32).reshape(ny, nx)
    loop = np.concatenate([idx[0, :-1], idx[:-1, -1], idx[-1, :0:-1], idx[:0:-1, 0]])
    px, py, pz = wx.ravel()[loop], wy.ravel()[loop], Z.ravel()[loop]
    npm = len(loop)
    dx, dy = np.roll(px, -1) - np.roll(px, 1), np.roll(py, -1) - np.roll(py, 1)
    ln = np.hypot(dx, dy) + 1e-6
    ox, oy = -dy / ln, dx / ln              # the loop runs clockwise seen from above (row 0 is y = +B, left to right, then down the right side), so (-dy, dx) is outwards
    s = np.cumsum(ln / 2)
    tt = np.linspace(0, 1, NT + 1)[1:]
    # the bulge: what left the top went out through the sides, most of it nearest the hardest pressing, and most at mid height
    b_row = np.clip(((B - py) / 2).astype(np.int32), 0, bulge.shape[0] - 1)
    b_col = np.clip(((px + A) / 2).astype(np.int32), 0, bulge.shape[1] - 1)
    local = bulge[b_row, b_col]
    local = local / max(float(local.mean()), 1e-5) * .7 + .3
    h_side = THICK + pz.mean()
    amount = lost / (2 * (2 * A + 2 * B) * h_side * .66)
    # the sides were cut with a wire and patted with the hand: long horizontal marks, shallow dents, the odd slump
    N_S, N_T = 4096, 512
    tile_low, tile_wire, tile_fine, tile_pat = side_tiles(N_S, N_T)
    si = ((s / s.max()) * (N_S - 1)).astype(np.int32)
    sis = (si * 14) % N_S
    co_side = np.empty((NT, npm, 3), np.float32)
    for r, t in enumerate(tt):
        ti = int(t * (N_T - 1))
        bar = 4 * t * (1 - t)
        out = amount * local * bar ** .8
        out = out + tile_low[si, ti] * .55 * min(t * 5, 1) + tile_wire[si // 3, ti] * .03 + tile_fine[sis, (ti * 4) % N_T] * .01 + tile_pat[si, ti] * .2
        out = out + 1.1 * math.exp(-(1 - t) * h_side / 2.4) * (1 + tile_low[si, 40] * .5)
        co_side[r, :, 0] = px + ox * out
        co_side[r, :, 1] = py + oy * out
        co_side[r, :, 2] = pz + (-THICK - pz) * t ** .96
    co = np.concatenate([np.stack([wx, wy, Z], -1).astype(np.float32).reshape(-1, 3), co_side.reshape(-1, 3)])
    base = nx * ny
    rings = np.vstack([loop[None, :], base + np.arange(NT * npm, dtype=np.int32).reshape(NT, npm)])
    a = idx
    top_q = np.stack([a[:-1, :-1], a[1:, :-1], a[1:, 1:], a[:-1, 1:]], -1).reshape(-1, 4)
    side_q = np.stack([rings[:-1], np.roll(rings[:-1], -1, axis=1), np.roll(rings[1:], -1, axis=1), rings[1:]], -1).reshape(-1, 4)
    q = np.concatenate([top_q, side_q])
    me = bpy.data.meshes.new("clay")
    me.vertices.add(len(co))
    me.vertices.foreach_set("co", co.ravel())
    me.loops.add(q.size)
    me.loops.foreach_set("vertex_index", q.ravel())
    me.polygons.add(len(q))
    me.polygons.foreach_set("loop_start", np.arange(len(q), dtype=np.int32) * 4)
    me.update(calc_edges=True)
    me.shade_smooth()
    # what the shader needs to know: where the dies have polished it, and that the sides are drier
    bk = burnish.shape[0] / (2 * B)
    br = burnish[np.clip(((B - ys) * bk).astype(np.int32), 0, burnish.shape[0] - 1)][:, np.clip(((xs + A) * bk).astype(np.int32), 0, burnish.shape[1] - 1)]
    attr = me.attributes.new("burnish", "FLOAT", "POINT")
    attr.data.foreach_set("value", np.concatenate([br.ravel(), np.zeros(NT * npm, np.float32)]).astype(np.float32))
    attr = me.attributes.new("fresh", "FLOAT", "POINT")
    fr = np.zeros_like(br)
    if fresh is not None:
        fr = fresh[np.clip(((B - ys) * bk).astype(np.int32), 0, burnish.shape[0] - 1)][:, np.clip(((xs + A) * bk).astype(np.int32), 0, burnish.shape[1] - 1)]
    attr.data.foreach_set("value", np.concatenate([fr.ravel(), np.zeros(NT * npm, np.float32)]).astype(np.float32))
    attr = me.attributes.new("side", "FLOAT", "POINT")
    attr.data.foreach_set("value", np.concatenate([np.zeros(base, np.float32), np.ones(NT * npm, np.float32)]))
    o = bpy.data.objects.new("clay", me)
    bpy.context.collection.objects.link(o)
    me.materials.append(mat)
    print("clay", len(co), "verts, side bulge", round(amount, 2), "mm", flush=True)
    return o


# ---- materials -------------------------------------------------------------------------------------------------------------------------------------------
class Nodes:
    def __init__(self, name):
        self.m = bpy.data.materials.new(name)
        self.m.use_nodes = True
        self.t = self.m.node_tree
        self.b = self.t.nodes["Principled BSDF"]
        tc = self.t.nodes.new("ShaderNodeTexCoord")
        self.obj = tc.outputs["Object"]

    def n(self, kind, **kw):
        x = self.t.nodes.new(kind)
        for k, v in kw.items():
            if isinstance(k, str) and hasattr(x, k):
                setattr(x, k, v)
        return x

    def noise(self, scale, detail=4.0, rough=.55, vec=None, w=None):
        x = self.t.nodes.new("ShaderNodeTexNoise")
        x.inputs["Scale"].default_value = scale
        x.inputs["Detail"].default_value = detail
        x.inputs["Roughness"].default_value = rough
        self.t.links.new(vec or self.obj, x.inputs["Vector"])
        return x.outputs[0]

    def range(self, v, a, b, c=0.0, d=1.0, smooth=False):
        x = self.t.nodes.new("ShaderNodeMapRange")
        x.interpolation_type = "SMOOTHSTEP" if smooth else "LINEAR"
        self.t.links.new(v, x.inputs[0])
        for i, val in zip((1, 2, 3, 4), (a, b, c, d)):
            x.inputs[i].default_value = val
        return x.outputs[0]

    def math(self, op, a, b=None):
        x = self.t.nodes.new("ShaderNodeMath")
        x.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                x.inputs[i].default_value = v
            else:
                self.t.links.new(v, x.inputs[i])
        return x.outputs[0]

    def mix(self, f, a, b):
        x = self.t.nodes.new("ShaderNodeMix")
        x.data_type = "RGBA"
        self.t.links.new(f, x.inputs[0])
        for sock, v in ((x.inputs[6], a), (x.inputs[7], b)):
            if isinstance(v, tuple):
                sock.default_value = (*v, 1)
            else:
                self.t.links.new(v, sock)
        return x.outputs[2]

    def attr(self, name):
        x = self.t.nodes.new("ShaderNodeAttribute")
        x.attribute_name = name
        return x.outputs["Fac"]

    def bump(self, h, dist, strength=1.0, normal=None):
        x = self.t.nodes.new("ShaderNodeBump")
        x.inputs["Strength"].default_value = strength
        x.inputs["Distance"].default_value = dist
        self.t.links.new(h, x.inputs["Height"])
        if normal:
            self.t.links.new(normal, x.inputs["Normal"])
        return x.outputs[0]


def clay_material(crab=False):
    """澄泥: the grey clay of bricks and inkstones, settled out of still water, so dense it hardly shines and it lies in fine layers, which show wherever something has cut down through them. Very fine mineral dust in it, a few glints of mica.
    Earth, not plaster: a low warm grey, olive and brown in it, the darker layers keeping a trace of 青, and still damp, so a thin wet sheen sits on the face and the lamp's highlight runs soft along it. crab: for the block that gets up (film_imprint_crab.py), whose grain goes with the clay and not with where it has got to, and which turns from this grey to matt terracotta where its "warm" says so."""
    N = Nodes("clay")
    t, b = N.t, N.b
    if crab:
        rest = t.nodes.new("ShaderNodeAttribute")
        rest.attribute_name = "rest"
        N.obj = rest.outputs["Vector"]
    burn, side, fresh = N.attr("burnish"), N.attr("side"), N.attr("fresh")
    # the layers it settled in: nearly level, wandering a little, so a cut wall shows bands and a slope shows long contours
    mp = t.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (.014, .014, .62)
    t.links.new(N.obj, mp.inputs[0])
    layer = N.range(N.noise(1.0, 3.0, .55, vec=mp.outputs[0]), .36, .64, 0, 1, True)
    layer = N.math("MULTIPLY", layer, N.math("SUBTRACT", 1.0, N.math("MULTIPLY", side, .45)))       # the block's dry outer skin half hides them, or it reads as a stack of paper
    if crab:         # worked up by hand they are nearly gone from what were the walls, or he reads as wood
        layer = N.math("MULTIPLY", layer, N.math("SUBTRACT", 1.0, N.math("MULTIPLY", N.attr("wall"), .85)))
    patch = N.range(N.noise(.013, 2.0, .5), .36, .66, 0, 1, True)
    fine = N.range(N.noise(.5, 5.0, .6), .35, .65)
    soil = N.range(N.noise(.14, 5.0, .62), .34, .66)
    tone = N.math("ADD", N.math("ADD", N.math("MULTIPLY", layer, .36), N.math("MULTIPLY", patch, .2)), N.math("ADD", N.math("MULTIPLY", fine, .16), N.math("MULTIPLY", soil, .28)))
    col = N.mix(tone, (.13, .116, .096), (.061, .064, .061))
    # the dust: paler grains, far finer than anything the camera can separate until it is very close
    dust = N.range(N.noise(16.0, 2.0, .5), .6, .68, 0, 1, True)
    col = N.mix(N.math("MULTIPLY", dust, .5), col, (.2, .186, .162))
    dark = N.range(N.noise(6.0, 2.0, .4), .71, .76, 0, 1, True)
    col = N.mix(N.math("MULTIPLY", dark, .6), col, (.045, .042, .037))
    # the sides have dried further and are a little paler
    col = N.mix(N.math("MULTIPLY", side, .1 if crab else .3), col, (.158, .147, .128))
    # what the groove has just laid bare is still wet
    col = N.mix(N.math("MULTIPLY", fresh, .5), col, (.038, .037, .034))
    if crab:
        hot = N.mix(tone, (.5, .118, .044), (.27, .056, .022))
        hot = N.mix(N.math("MULTIPLY", dust, .45), hot, (.62, .25, .115))
        hot = N.mix(N.math("MULTIPLY", dark, .6), hot, (.11, .034, .018))
        hot = N.mix(N.math("MULTIPLY", fresh, .5), hot, (.13, .038, .018))
        col = N.mix(N.attr("warm"), col, hot)
        col = N.mix(N.attr("inner"), col, (.011, .007, .006))
    t.links.new(col, b.inputs["Base Color"])
    rough = N.math("ADD", N.range(patch, 0, 1, .62, .52), N.range(fine, 0, 1, .03, -.03))
    rough = N.math("ADD", rough, N.math("MULTIPLY", burn, -.1))
    rough = N.math("ADD", rough, N.math("MULTIPLY", side, .1))
    rough = N.math("ADD", rough, N.math("MULTIPLY", fresh, -.26))
    t.links.new(rough, b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = .42
    b.inputs["Sheen Weight"].default_value = .07
    coat = N.math("MULTIPLY", N.math("SUBTRACT", 1.0, side), .2)          # the damp: a thin film of water over the face, gone from the dry sides
    b.inputs["Coat Roughness"].default_value = .3
    b.inputs["Coat IOR"].default_value = 1.33
    if crab:         # down inside his eyes and mouth nothing should catch the sun
        shut = N.math("SUBTRACT", 1.0, N.attr("inner"))
        t.links.new(N.math("MULTIPLY", shut, .42), b.inputs["Specular IOR Level"])
        t.links.new(N.math("MULTIPLY", shut, .07), b.inputs["Sheen Weight"])
        coat = N.math("MULTIPLY", coat, shut)
    t.links.new(coat, b.inputs["Coat Weight"])
    b.inputs["Sheen Roughness"].default_value = .55
    b.inputs["Sheen Tint"].default_value = (.9, .89, .86, 1)
    # grain finer than the mesh, the only shading trick in the block
    h = N.math("ADD", N.math("MULTIPLY", N.noise(9.0, 6.0, .62), 1.0), N.math("MULTIPLY", N.noise(2.2, 3.0, .5), .5))
    h = N.math("MULTIPLY", h, N.math("SUBTRACT", 1.0, N.math("MULTIPLY", burn, .5)))
    n = N.bump(h, .008, 1.0)
    h2 = N.range(N.noise(16.0, 2.0, .5), .62, .7, 0, 1, True)
    n = N.bump(h2, .012, .6, n)
    t.links.new(n, b.inputs["Normal"])
    return N.m


def board_object():
    x0, x1, y0, y1 = BOARD_RECT
    N = Nodes("board")
    t, b = N.t, N.b
    uv = t.nodes.new("ShaderNodeTexCoord").outputs["UV"]
    i1 = t.nodes.new("ShaderNodeTexImage")
    i1.image = bpy.data.images.load(os.path.join(OUT, "board_col.png"))
    i1.image.colorspace_settings.name = "Non-Color"
    i2 = t.nodes.new("ShaderNodeTexImage")
    i2.image = bpy.data.images.load(os.path.join(OUT, "board_h.png"))
    i2.image.colorspace_settings.name = "Non-Color"
    t.links.new(uv, i1.inputs[0])
    t.links.new(uv, i2.inputs[0])
    t.links.new(i1.outputs[0], b.inputs["Base Color"])
    sep = t.nodes.new("ShaderNodeSeparateColor")
    t.links.new(i2.outputs[0], sep.inputs[0])
    t.links.new(sep.outputs[1], b.inputs["Roughness"])
    t.links.new(N.bump(sep.outputs[0], 2.1, 1.0), b.inputs["Normal"])
    b.inputs["Specular IOR Level"].default_value = .35
    me = bpy.data.meshes.new("board")
    me.from_pydata([(x0, y0, -THICK), (x1, y0, -THICK), (x1, y1, -THICK), (x0, y1, -THICK)], [], [(0, 1, 2, 3)])
    u = me.uv_layers.new(name="UV")
    for i, v in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        u.data[i].uv = v
    me.update()
    o = bpy.data.objects.new("board", me)
    bpy.context.collection.objects.link(o)
    me.materials.append(N.m)
    return o


def crumbs(mat, count=70):
    """Bits that came off when the block was cut, near its foot. Real lumps, not specks in the texture."""
    r = np.random.default_rng(91)
    from mathutils import Vector
    co, faces = [], []
    tpl = bpy.data.meshes.new("t")
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
    bm.to_mesh(tpl)
    bm.free()
    base = np.array([v.co[:] for v in tpl.vertices], np.float32)
    tris = np.array([p.vertices[:] for p in tpl.polygons], np.int32)
    k = 0
    for _ in range(count):
        side = r.integers(4)
        off = abs(r.normal(0, 14)) + 1.5
        if side < 2:
            x, y = r.uniform(-A, A), (B + off) * (1 if side else -1)
        else:
            x, y = (A + off) * (1 if side == 2 else -1), r.uniform(-B, B)
        s = float(r.choice((r.uniform(.25, .8), r.uniform(.8, 1.9)), p=(.8, .2)))
        p = base * s * r.uniform(.7, 1.2, 3)
        p = p * (1 + .28 * np.sin(base @ r.normal(0, 3.2, 3) + r.uniform(0, 6)))[:, None]
        p = p + (x, y, -THICK + s * .25)
        co.append(p)
        faces.append(tris + k)
        k += len(base)
    me = bpy.data.meshes.new("crumbs")
    me.from_pydata(np.vstack(co).tolist(), [], np.vstack(faces).tolist())
    me.update()
    me.shade_smooth()
    me.materials.append(mat)
    o = bpy.data.objects.new("crumbs", me)
    bpy.context.collection.objects.link(o)
    bpy.data.meshes.remove(tpl)
    return o


def rounded_box(name, w, d, z0, z1, bevel, mat):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    vs = [bm.verts.new((sx * w / 2, sy * d / 2, z)) for z in (z0, z1) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for f in ((3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        bm.faces.new([vs[i] for i in f])
    bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=4, profile=.5, affect="EDGES")
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    return o


def wood_material():
    """The block of the die behind its metal: dark old wood, oiled by hands, the grain running up it."""
    N = Nodes("wood")
    t, b = N.t, N.b
    mp = t.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (.7, .7, .018)
    t.links.new(N.obj, mp.inputs[0])
    fine = t.nodes.new("ShaderNodeMapping")
    fine.inputs["Scale"].default_value = (3.2, 3.2, .05)
    t.links.new(N.obj, fine.inputs[0])
    pores = t.nodes.new("ShaderNodeMapping")
    pores.inputs["Scale"].default_value = (11.0, 11.0, .34)
    t.links.new(N.obj, pores.inputs[0])
    grain = N.noise(1.0, 8.0, .6, vec=mp.outputs[0])
    hair = N.noise(1.0, 6.0, .55, vec=fine.outputs[0])
    pit = N.range(N.noise(1.0, 2.0, .4, vec=pores.outputs[0]), .62, .7, 0, 1, True)
    broad = N.noise(.02, 3.0, .5)
    streak = N.range(grain, .36, .66, 0, 1, True)
    col = N.mix(streak, (.135, .088, .056), (.062, .039, .025))
    col = N.mix(N.range(hair, .4, .62, 0, .5, True), col, (.038, .024, .016))
    col = N.mix(N.range(broad, .4, .62, 0, .5, True), col, (.03, .02, .014))
    col = N.mix(pit, col, (.012, .008, .006))
    t.links.new(col, b.inputs["Base Color"])
    t.links.new(N.range(grain, .3, .7, .36, .56), b.inputs["Roughness"])
    t.links.new(N.bump(hair, .05, 1.0), b.inputs["Normal"])
    b.inputs["Specular IOR Level"].default_value = .45
    return N.m


def copper_material():
    N = Nodes("copper")
    t, b = N.t, N.b
    b.inputs["Metallic"].default_value = 1.0
    tar = N.range(N.noise(.03, 6.0, .6), .42, .62, 0, 1, True)
    col = N.mix(tar, (.5, .29, .17), (.135, .082, .062))
    t.links.new(col, b.inputs["Base Color"])
    t.links.new(N.range(N.noise(.5, 5.0, .55), .3, .7, .24, .46), b.inputs["Roughness"])
    # brushed by the cloth that wiped it
    vec = t.nodes.new("ShaderNodeMapping")
    vec.inputs["Scale"].default_value = (.05, 1.0, 12.0)
    t.links.new(N.obj, vec.inputs[0])
    t.links.new(N.bump(N.noise(3.0, 3.0, .5, vec=vec.outputs[0]), .02, 1.0), b.inputs["Normal"])
    return N.m


def die_objects(k, drop):
    """Only what the frame shows of a die: the lower end of something heavy, coming down from above the frame. Its printed face is pressed into the clay, so it needs no drawing."""
    p = PRESSES[k]
    cx, cy = p["centre"]
    z = np.load(state_path(k))
    H, res = z["H"], int(z["res"])
    ground = p["relief"] - p["depth"] + float(np.median(H[::8, ::8])) + drop
    w, d = p["size"]
    if os.path.exists(os.path.join(OUT, "dies", p["name"] + ".npz")) and p["name"] == "ruling":
        d = 110.0
    wood = wood_material()
    parts = []
    if p["name"] == "ruling":
        parts.append(rounded_box("plate", w, d, ground, ground + 15.0, .6, copper_material()))
        parts.append(rounded_box("mount", w - 5.0, d - 5.0, ground + 15.0, ground + 1500.0, 1.8, wood))
    else:
        parts.append(rounded_box("mount", w, d, ground, ground + 1500.0, 1.5, wood))
    for o in parts:
        o.location = (cx, cy, 0)
        o.rotation_euler = (0, 0, math.radians(p["rot"]))
    return parts


# ---- the shot --------------------------------------------------------------------------------------------------------------------------------------------
def set_up(args, sc):
    sc.render.film_transparent = False
    sc.render.resolution_x, sc.render.resolution_y = args["w"], args["h"]
    sc.render.resolution_percentage = 100
    sc.unit_settings.scale_length = .001
    sc.render.image_settings.color_mode = "RGB"
    sc.cycles.max_bounces = 8
    sc.cycles.diffuse_bounces = 4
    sc.cycles.adaptive_threshold = .012
    sc.cycles.use_denoising = args["denoise"]
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    sc.cycles.denoising_prefilter = "NONE"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Punchy"
    sc.view_settings.exposure = args["exp"]
    sc.render.use_persistent_data = False
    w = sc.world
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (.070, .062, .052, 1)
    bg.inputs[1].default_value = .8


def lights(az):
    """The side light, and a big dim card high up beyond the block, which is what the damp top reflects towards the camera: that faint sheen is the only other thing lighting it."""
    a, e = math.radians(az), math.radians(LIGHT_EL)
    t = (0.0, 0.0, -10.0)
    p = (t[0] - LIGHT_D * math.cos(a) * math.cos(e), t[1] + LIGHT_D * math.sin(a) * math.cos(e), t[2] + LIGHT_D * math.sin(e))
    lamp = bpy.data.lights.new("side", "AREA")
    lamp.shape = "RECTANGLE"
    lamp.size, lamp.size_y = LIGHT_SIZE, LIGHT_SIZE * 1.4
    lamp.energy = 4.3e6
    lamp.color = (1.0, .955, .885)
    lamp.spread = math.radians(40)
    o = bpy.data.objects.new("side", lamp)
    bpy.context.collection.objects.link(o)
    aim(o, p, t)
    card = bpy.data.lights.new("sheen", "AREA")
    card.shape = "RECTANGLE"
    card.size, card.size_y = 2200.0, 900.0
    card.energy = 2.2e6
    card.color = (.96, .965, .97)
    c = bpy.data.objects.new("sheen", card)
    bpy.context.collection.objects.link(c)
    aim(c, (120.0, 1500.0, 980.0), (0.0, 0.0, 0.0))
    fill = bpy.data.lights.new("bounce", "AREA")
    fill.shape = "RECTANGLE"
    fill.size, fill.size_y = 900.0, 260.0
    fill.energy = 1.9e6
    fill.color = (1.0, .93, .86)
    f = bpy.data.objects.new("bounce", fill)
    bpy.context.collection.objects.link(f)
    aim(f, (-260.0, -1000.0, 60.0), (0.0, -60.0, -25.0))
    return o


def dawn(sc, az, el=DAWN_EL, e=DAWN_E, colour=DAWN_COLOUR, gold=1.3):
    """破晓: the sun just over the edge of the world, so far off that its light comes in one direction over the whole block, cold and low. A last thread of gold, right down on the horizon, catches only the crests: strong, because at that angle the flats take almost none of it. The rest of the sky hardly lights anything: what the sun doesn't reach goes nearly to black."""
    t = (0.0, 0.0, -6.0)

    def sun(name, az, el, strength, angle, colour):
        a, e = math.radians(az), math.radians(el)
        d = bpy.data.lights.new(name, "SUN")
        d.energy = strength
        d.angle = math.radians(angle)
        d.color = colour
        o = bpy.data.objects.new(name, d)
        bpy.context.collection.objects.link(o)
        aim(o, (t[0] - 2000 * math.cos(a) * math.cos(e), t[1] + 2000 * math.sin(a) * math.cos(e), t[2] + 2000 * math.sin(e)), t)

    sun("dawn", az, el, e, 1.6, colour)
    sun("gold", az, el - 14.0, e * gold, .8, GOLD_COLOUR)
    bg = sc.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (*DAWN_SKY, 1)
    bg.inputs[1].default_value = 1.1


def camera(sc, scale, fstop, shake=(0.0, 0.0), elev=None, target=None):
    e, a = math.radians(elev if elev is not None else ELEV), math.radians(CAM_AZ)
    d = D1 / scale
    t = np.array(target if target else TARGET)
    p = t + d * np.array((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
    data = bpy.data.cameras.new("cam")
    data.lens = LENS
    data.sensor_width = 36.0
    data.clip_start, data.clip_end = 5.0, 20000.0
    data.dof.use_dof = True
    data.dof.focus_distance = d
    data.dof.aperture_fstop = fstop / 1000.0
    data.dof.aperture_blades = 7
    data.shift_x, data.shift_y = shake[0] / sc.render.resolution_x, shake[1] / sc.render.resolution_x
    o = bpy.data.objects.new("cam", data)
    bpy.context.collection.objects.link(o)
    aim(o, p.tolist(), t.tolist())
    sc.camera = o
    return o


def still(a):
    sc = reset(a["samples"], 100)
    set_up(a, sc)
    mat = clay_material()
    block(a["state"], a["dense"], mat)
    board_object()
    crumbs(mat)
    if a["dawn"]:
        dawn(sc, a["az"] if a["az"] is not None else DAWN_AZ, *(a["sun"] or (DAWN_EL, DAWN_E)))
    else:
        lights(a["az"] if a["az"] is not None else LIGHT_AZ)
    if a["die"] is not None:
        die_objects(a["die"], a["drop"])
    camera(sc, a["scale"], a["fstop"], elev=a["elev"], target=a["target"])
    if a["look"]:
        sc.view_settings.look = a["look"]
    if a["crop"]:
        x0, y0, x1, y1 = a["crop"]
        sc.render.use_border, sc.render.use_crop_to_border = True, True
        sc.render.border_min_x, sc.render.border_max_x = x0 / a["w"], x1 / a["w"]
        sc.render.border_min_y, sc.render.border_max_y = 1 - y1 / a["h"], 1 - y0 / a["h"]
    path = os.path.join(OUT, a["name"] + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]

    def opt(name, default, n=1, kind=float):
        if name not in argv:
            return default
        i = argv.index(name)
        v = [kind(x) for x in argv[i + 1:i + 1 + n]]
        return v[0] if n == 1 else v

    if argv[0] == "still":
        still(dict(name=argv[1], state=argv[2], scale=float(argv[3]), az=opt("--az", None), dense=opt("--dense", None, 4), samples=opt("--samples", 96, 1, int), w=opt("--w", 1920, 1, int), h=opt("--h", 1080, 1, int),
                   fstop=opt("--fstop", 22.0), exp=opt("--exp", .5), die=opt("--die", None, 1, int), drop=opt("--drop", 0.0), crop=opt("--crop", None, 4, int), denoise="--raw" not in argv,
                   dawn="--dawn" in argv, sun=opt("--sun", None, 2), elev=opt("--elev", None), target=opt("--target", None, 3), look=opt("--look", None, 1, str)))
