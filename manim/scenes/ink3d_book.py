"""序 · 书 and 十四 · 诗与合书: a real thread-bound copy of 《淮南子》 on a desk in moonlight through a lattice window. The film starts and ends here. At the start there is
only the old book. At the end there is one thing more: a sheet of new paper outside it, with the poem on it.

    python ink3d_book_paper.py tex                                  # the pages, the sheet and the desk, once
    Blender -b --factory-startup -P ink3d_book.py -- --shot=cover [--pct=100] [--samples=128] [--t=SECONDS]
    python ink3d_book_paper.py post cover
    Blender -b --factory-startup -P ink3d_book.py -- --shot=open|close --frames=A-B [--pct=100] [--samples=32] [--sub=3] [--step=N] [--dir=seq]    # one build, seq/SHOT_NNNN.png for each frame
                                                                                                                           # (or _s0, _s1, ... to be averaged, where things move fast)

Shots, in the film's order (t is seconds into the film for the opening; when the book closes, it is the same clock run backwards):

cover   t=0     First frame of the film and its thumbnail: the closed book, title slip the brightest thing in the picture, moon lattice on the desk. A very slow push in.
half    t=1.78  The cover lifts at the fore-edge and turns over to the right, bending as it goes; the first leaves start to follow it in a fan. Paper sound. (The moving camera passes this key at t=2.0.)
spread  t=4.8   The fan has landed. Two pages lie open, the whole woodblock layout in the moonlight, and the camera is coming down and over them. (It passes this key at t=3.9.)
column  t=5.6   Straight above the page, the column 昔者蒼頡作書而天雨粟鬼夜哭 stands where the film's subtitles live (x 1760-1860), so the book teaches how the film is read.
                The other characters begin to bleed (class Bleed in ink3d_book_paper.py, class Wash in ink_book.py): the wash goes on spreading until by t=7.2 the paper is the night sky of
                一 · 鬼夜哭, and the column hands over to the first subtitle.
sheet   end     十四: one sheet of xuan with a deckle edge on the desk, the corner of the open book beside it. We went in through the old book and come out outside it. The poem is written on it across the page,
                the title and then a couplet to a line (ink_book.py), so the viewer reads it before any subtitle speaks of it. On 这篇自传，还有一处荒诞 it is done; the last lines are still wet and catch the light.
seal    end     On 窗口一关 the empty seal comes down beside the signature, the only red in the film. Same camera as sheet, nothing else moves.
fading  end     Then the poem starts to go: from the first row the ink is drawn out as if by water, row by row (washed() in ink3d_book_paper.py, its t counts rows). Same camera.
closing end     On 它不存在我这里 the last row has gone, nothing on the sheet but the seal. Beside it the book closes by itself: the opening run backwards, top leaf first, the cover last, and
                its slip still says 淮南子. The frame is wider than end's, to hold the cover standing up, and pushes in to it.
end     end     On 只存在你那里 the moonlight narrows onto the sheet. Last frame: an old book, shut, a blank sheet, an empty seal. The poem is no longer on the screen, only in the viewer's
                memory, which is exactly what the line says. It holds, and the film ends.
sheet_blank, seal_only  Plates for compositing the writing and the washing: sheet's camera and light exactly, with the paper bare and with only the seal.
sheet_wet               The same again, with every stroke as wet as the wettest and the whole sheet catching the sky: what fresh ink looks like, for ink_book.py to let dry away.
open, close             The opening and the closing as they move, frame by frame: see ink3d_book_keys.py, and ink_book.py for what is done with them.

The cover and leaves are bendable grids and all the motion is a function of t: cover_u() and leaf_u() give how far each has turned, deform() bends it. The poem is written by revealing
sheet.png row by row, and sheet_ink.png says where it is wet.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from mathutils import Matrix, Vector

from ink3d import OUT3D, aim, blur, camera, mesh_grid, painted_material, render, reset, sm, spectral
from ink3d_book_keys import CLOSE_FRAMES, GLEAM_AT, GLEAM_POWER, GLEAM_SIZE, KEYS, MOON, SH, SHEET_C, SHEET_ROT, SW, close_book, close_path, close_state, column_key, needs_blur, open_book, open_keys, open_path, open_state, seq_time, shutter, window_axes

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


SHOT = opt("shot", "cover")
PCT, SAMPLES = opt("pct", 100), opt("samples", 128)
DIR = os.path.join(OUT3D, "book")
TEX = os.path.join(DIR, "tex")

# the book, in cm: spine at +x, fore-edge at -x
X_S, X_F = 7.25, -7.25
X_H = X_S - 1.1                  # the leaves turn on this line, just inside the stitching
L = X_H - X_F
Y0, Y1 = -11.5, 11.5
Z_BACK = .035                    # back cover
T_L = .55                        # the pages that stay down
N, DZ = 20, .03                  # the leaves that turn over, and their spacing
STRIP = .95                      # the stitched strip is pressed a little thinner
X_T = X_S - .92                  # the stitch line
HOLES = (-8.1, -2.9, 2.9, 8.1)
LEAF_SEED = 300
NS, NY = 150, 100


def h_block(s):
    """Top of the pages that stay down, by distance from the hinge: pinched at the stitching, rounded just after it."""
    return Z_BACK + T_L * (.93 + .07 * sm(s / 1.5)) + .05 * sm(s / 3.5) * (1 - .6 * s / L)


def z_closed(s, layer):
    return h_block(s) + (layer + 1) * DZ * (.55 + .45 * sm(s / 1.5))


def z_open(s, m):
    """Where leaf number m of the turned stack comes to rest (m = 0 lies lowest, on the cover): up over the stitched strip, then down to the desk."""
    rest = .085 + (m + 1) * DZ
    over = STRIP + .13 + m * .004
    w = sm((s - .35) / 7.5)
    bulge = .10 * sm(s / 2.4) * (1 - .55 * s / L)
    return over * (1 - w) + rest * w + bulge * (m + 1) / N


def deform(u, s, y, zc, zo, bend, curl, twist):
    """A sheet turning over the hinge. u is how far (0 closed, lying to the left; 1 open, lying to the right). s, y: 1D. zc, zo: heights (n_y × n_s) when closed and open.
    Near the start the outer edge leads, the way a thumb lifts it; near the end it trails, held back by the air."""
    ss, yy = np.meshgrid(s, y)
    q, yn = ss / s[-1], yy / Y1
    xc, xo = X_H - ss, X_H + ss
    if u <= 0:
        return xc, yy, zc
    if u >= 1:
        return xo, yy, zo
    a = math.pi * u + bend * math.sin(2 * math.pi * u) * q + curl * math.sin(math.pi * u) * (q - .3) + twist * math.sin(math.pi * u) * yn * q
    a = np.clip(a, 0, math.pi)
    ds = s[1] - s[0]
    x = X_H - np.cumsum(np.cos(a), 1) * ds
    z = zc[:, :1] + np.cumsum(np.sin(a), 1) * ds
    p = sm(u / .10)
    r = sm((u - .88) / .12)
    x = xc * (1 - p) + x * p
    z = zc * (1 - p) + z * p
    x = x * (1 - r) + xo * r
    z = z * (1 - r) + zo * r
    return x, yy, z


def cover_u(t):
    return float(sm((t - 1.2) / 1.35))


def leaf_u(k, t):
    """The leaves follow the cover in a fan, the top one first."""
    return float(sm((t - 1.48 - .068 * k) / 1.15))


# ---- materials ---------------------------------------------------------------------------------------------------------------------------------------
def mix_rgba(nt, fac, a, b):
    """The Mix node has float, vector and colour sockets of the same name: take the colour ones."""
    m = nt.nodes.new("ShaderNodeMix")
    m.data_type = "RGBA"
    ca, cb = [s for s in m.inputs if s.type == "RGBA"][:2]
    nt.links.new(fac, m.inputs[0])
    nt.links.new(a, ca)
    nt.links.new(b, cb)
    return [s for s in m.outputs if s.type == "RGBA"][0]


def tex(name, colorspace="sRGB"):
    im = bpy.data.images.load(os.path.join(TEX, name), check_existing=True)
    im.colorspace_settings.name = colorspace
    return im


def page_material(name, front, back, translucent=.16, rough=.82, bump=.05):
    """Two different pictures on the two faces of one sheet; a little light gets through."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = .25
    uf, ub = nt.nodes.new("ShaderNodeUVMap"), nt.nodes.new("ShaderNodeUVMap")
    uf.uv_map, ub.uv_map = "front", "back"
    tf, tb = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeTexImage")
    tf.image, tb.image = tex(front), tex(back)
    nt.links.new(uf.outputs[0], tf.inputs[0])
    nt.links.new(ub.outputs[0], tb.inputs[0])
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    colour = mix_rgba(nt, geo.outputs["Backfacing"], tf.outputs[0], tb.outputs[0])
    nt.links.new(colour, b.inputs["Base Color"])
    # paper fibres
    tc, noise_, bp = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeBump")
    noise_.inputs["Scale"].default_value = 9.0
    noise_.inputs["Detail"].default_value = 8
    bp.inputs["Strength"].default_value = bump
    nt.links.new(tc.outputs["Object"], noise_.inputs["Vector"])
    nt.links.new(noise_.outputs[0], bp.inputs["Height"])
    nt.links.new(bp.outputs[0], b.inputs["Normal"])
    # light through the sheet
    out = nt.nodes["Material Output"]
    tr, ms = nt.nodes.new("ShaderNodeBsdfTranslucent"), nt.nodes.new("ShaderNodeMixShader")
    ms.inputs[0].default_value = translucent
    nt.links.new(colour, tr.inputs[0])
    nt.links.new(b.outputs[0], ms.inputs[1])
    nt.links.new(tr.outputs[0], ms.inputs[2])
    nt.links.new(ms.outputs[0], out.inputs["Surface"])
    return m


def cover_material(picture):
    m = bpy.data.materials.new("cover")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = .62
    b.inputs["Specular IOR Level"].default_value = .4
    uf, ub = nt.nodes.new("ShaderNodeUVMap"), nt.nodes.new("ShaderNodeUVMap")
    uf.uv_map, ub.uv_map = "front", "back"
    tf, tb = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeTexImage")
    tf.image, tb.image = tex(picture), tex("lining.png")
    nt.links.new(uf.outputs[0], tf.inputs[0])
    nt.links.new(ub.outputs[0], tb.inputs[0])
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    colour = mix_rgba(nt, geo.outputs["Backfacing"], tf.outputs[0], tb.outputs[0])
    nt.links.new(colour, b.inputs["Base Color"])
    tbump, bp = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeBump")
    tbump.image = tex(picture.replace(".png", "_bump.png"), "Non-Color")
    nt.links.new(uf.outputs[0], tbump.inputs[0])
    nt.links.new(tbump.outputs[0], bp.inputs["Height"])
    bp.inputs["Strength"].default_value = .5
    bp.inputs["Distance"].default_value = .02
    nt.links.new(bp.outputs[0], b.inputs["Normal"])
    return m


def plain(name, value, rough=.85, spec=.3):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (value, value, value, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = spec
    return m


def desk_material():
    m = bpy.data.materials.new("desk")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Specular IOR Level"].default_value = .5
    uv = nt.nodes.new("ShaderNodeMapping")
    uv.inputs["Scale"].default_value = (2.0, 2.0, 1.0)
    uv.inputs["Location"].default_value = (.31, .17, 0.0)
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["UV"], uv.inputs[0])
    a, r_, h = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeTexImage")
    a.image, r_.image, h.image = tex("desk.png"), tex("desk_rough.png", "Non-Color"), tex("desk_bump.png", "Non-Color")
    a.image.colorspace_settings.name = "Non-Color"
    for t in (a, r_, h):
        nt.links.new(uv.outputs[0], t.inputs[0])
    nt.links.new(a.outputs[0], b.inputs["Base Color"])
    nt.links.new(r_.outputs[0], b.inputs["Roughness"])
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = .55
    bp.inputs["Distance"].default_value = .03
    nt.links.new(h.outputs[0], bp.inputs["Height"])
    nt.links.new(bp.outputs[0], b.inputs["Normal"])
    return m


# ---- the book ----------------------------------------------------------------------------------------------------------------------------------------
def grid_mesh(name, ns, ny):
    """A sheet whose vertices are set later, with UVs for both faces: u from the fold (left in the closed book's picture) etc."""
    me = bpy.data.meshes.new(name)
    n = ns * ny
    me.vertices.add(n)
    a = np.arange(n).reshape(ny, ns)
    quads = np.stack([a[:-1, :-1], a[1:, :-1], a[1:, 1:], a[:-1, 1:]], -1).ravel()      # wound so that the side facing up in the closed book is the front
    me.loops.add(len(quads))
    me.loops.foreach_set("vertex_index", quads.astype(np.int32))
    np_ = (ny - 1) * (ns - 1)
    me.polygons.add(np_)
    me.polygons.foreach_set("loop_start", np.arange(np_, dtype=np.int32) * 4)
    me.polygons.foreach_set("use_smooth", np.ones(np_, bool))
    me.update(calc_edges=True)
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    return o, quads


def set_uv(me, quads, us, vs, name):
    layer = me.uv_layers.new(name=name)
    uu = np.stack([us, vs], -1).reshape(-1, 2)
    layer.data.foreach_set("uv", uu[quads].ravel().astype(np.float32))


class Sheet:
    """A leaf (or the cover): a grid that deform() bends."""

    def __init__(self, name, material, length, y0, y1, zc, zo, bend, curl, twist, wave, seed, ns=NS, ny=NY, uv_front=None, uv_back=None):
        self.s, self.y = np.linspace(0, length, ns).astype(np.float32), np.linspace(y0, y1, ny).astype(np.float32)
        self.zc, self.zo, self.bend, self.curl, self.twist = zc, zo, bend, curl, twist
        self.o, quads = grid_mesh(name, ns, ny)
        ss, yy = np.meshgrid(self.s, self.y)
        uf = (1 - ss / length) if uv_front is None else uv_front(ss)
        ub = (ss / length) if uv_back is None else uv_back(ss)
        v = (yy - y0) / (y1 - y0)
        set_uv(self.o.data, quads, uf, v, "front")
        set_uv(self.o.data, quads, ub, v, "back")
        self.o.data.materials.append(material)
        # a sheet of old paper is never flat: it rolls a little, more toward its free edge
        i = (ss / length * 120).astype(np.int32)
        j = ((yy - y0) / (y1 - y0) * 200).astype(np.int32)
        n, own = spectral(256, 3.3, LEAF_SEED), spectral(256, 3.3, seed)      # a stack shares its waves, or the leaves below would poke through
        self.wave = (n[j % 256, i % 256] * wave + own[j % 256, i % 256] * .0025) * sm(ss / 3.5)

    def pose(self, u):
        ss, yy = np.meshgrid(self.s, self.y)
        zc = self.zc(ss) + self.wave * .55
        zo = self.zo(ss) + self.wave
        x, y, z = deform(u, self.s, self.y, zc, zo, self.bend, self.curl, self.twist)
        self.o.data.vertices.foreach_set("co", np.stack([x, y, z], -1).ravel().astype(np.float32))
        self.o.data.update()


def wall(name, xs, ys, z0, z1, dirx, diry, seed, layers=34, n_a=300, base=.38):
    """The edge of the stacked pages, as real strata: each leaf stands out a hair more or less than its neighbours, so the moon catches them."""
    r = np.random.default_rng(seed)
    a = np.linspace(0, 1, n_a)
    px, py = np.interp(a, [0, 1], xs), np.interp(a, [0, 1], ys)
    top = z1(px, py)
    b = np.linspace(0, 1, layers * 3)
    clump = blur(r.normal(0, 1, (layers, 8)).astype(np.float32), 1.3)[:, 0]
    strata = np.repeat(r.normal(0, 1, layers) * .7 + clump / (clump.std() + 1e-6) * .8, 3)
    drift = blur(spectral(layers * 3, 2.0, seed + 1, n_a), 1.2)
    drift = drift / (drift.std() + 1e-6)
    off = (strata[:, None] * .75 + drift * .45) * .016 + .018
    # rounding out to the sides: a stack bulges a little in the middle
    z = z0 + (top - z0)[None, :] * b[:, None]
    x = px[None, :] + dirx * off
    y = py[None, :] + diry * off
    o = mesh_grid(name, x.astype(np.float32), y.astype(np.float32), z.astype(np.float32))
    tone = base + np.clip(strata[:, None], -2, 2) * .10 + drift * .02 - .10 * (np.repeat(r.random(layers), 3)[:, None] > .82)
    o.data.materials.append(painted_material(name, np.clip(tone, .1, 1).astype(np.float32), rough=.9))
    return o


def tube(name, pts, radius, material):
    c = bpy.data.curves.new(name, "CURVE")
    c.dimensions = "3D"
    c.bevel_depth = radius
    c.bevel_resolution = 2
    c.resolution_u = 4
    sp = c.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, q in zip(sp.points, pts):
        p.co = (q[0], q[1], q[2], 1)
    c.materials.append(material)
    o = bpy.data.objects.new(name, c)
    bpy.context.collection.objects.link(o)
    return o


def stitching(z_top):
    """四眼订线: two strands of white silk along the stitch line, over the head and tail, and from each hole round the spine."""
    silk = plain("silk", .92, rough=.42, spec=.55)
    dark = plain("hole", .004, rough=.9)
    r = np.random.default_rng(55)
    rad, sep = .036, .082
    z = z_top + rad * .7
    for off, strand in ((-sep / 2, "a"), (sep / 2, "b")):
        # the long run
        pts = []
        pts.append((X_T + off, Y0 - .05, Z_BACK - .01))
        pts.append((X_T + off, Y0 - .055, z - .05))
        pts.append((X_T + off, Y0 + .03, z))
        for yy in np.linspace(Y0 + .3, Y1 - .3, 40):
            pts.append((X_T + off + .012 * math.sin(yy * 1.7 + off * 40), yy, z + .004 * math.sin(yy * 2.3)))
        pts.append((X_T + off, Y1 - .03, z))
        pts.append((X_T + off, Y1 + .055, z - .05))
        pts.append((X_T + off, Y1 + .05, Z_BACK - .01))
        tube("thread_long_" + strand, pts, rad, silk)
        # round the spine, at each hole and at the ends
        for yh in HOLES + (Y0 + .55, Y1 - .55):
            y_ = yh + off
            pts = [(X_T, y_, z), (X_S - .05, y_, z), (X_S + .03, y_, z - .04), (X_S + .055, y_, z - .12), (X_S + .05, y_, Z_BACK - .01)]
            tube(f"thread_{yh}_{strand}", pts, rad, silk)
    for yh in HOLES:
        bpy.ops.mesh.primitive_cylinder_add(radius=.06, depth=.006, location=(X_T, yh, z_top + .002))
        bpy.context.object.data.materials.append(dark)
    # a dark patch of cloth over each corner of the spine (包角)
    cloth = plain("corner", .10, rough=.7, spec=.3)
    for ya, yb in ((Y1 - 1.0, Y1 + .03), (Y0 - .03, Y0 + 1.0)):
        x_, y_ = np.meshgrid(np.linspace(X_S - .72, X_S + .03, 12), np.linspace(ya, yb, 12))
        zz = np.full_like(x_, z_top + .004) - .045 * sm((x_ - X_S + .02) / .05)
        o = mesh_grid("corner", x_.astype(np.float32), y_.astype(np.float32), zz.astype(np.float32), uv=False)
        o.data.materials.append(cloth)


def build():
    sheets = {}
    # desk
    x_, y_ = np.meshgrid(np.linspace(-120, 120, 2), np.linspace(-60, 60, 2))
    d = mesh_grid("desk", x_.astype(np.float32), y_.astype(np.float32), np.zeros((2, 2), np.float32))
    d.data.materials.append(desk_material())
    # back cover
    x_, y_ = np.meshgrid(np.linspace(X_F - .06, X_S + .03, 2), np.linspace(Y0 - .06, Y1 + .06, 2))
    b = mesh_grid("back", x_.astype(np.float32), y_.astype(np.float32), np.full((2, 2), Z_BACK - .005, np.float32))
    b.data.materials.append(plain("backc", .05, rough=.7))
    # the stack that stays down, and the strip
    sx = np.linspace(0, L, 260)
    yy = np.linspace(Y0, Y1, 8)
    xg, yg = np.meshgrid(X_H - sx, yy)
    top = mesh_grid("block", xg.astype(np.float32), yg.astype(np.float32), (h_block(X_H - xg) - .07).astype(np.float32))
    top.data.materials.append(plain("blocktop", .72, rough=.9))
    s_of = lambda x: np.maximum(X_H - x, 0)
    zfun = lambda px, py: np.where(px > X_H, h_block(0.0) + (N * .55 * DZ), h_block(s_of(px)))
    wall("fore", (X_F, X_F), (Y0, Y1), Z_BACK, zfun, -1, 0, 61)
    wall("head", (X_F, X_S), (Y1, Y1), Z_BACK, zfun, 0, 1, 62)
    wall("tail", (X_F, X_S), (Y0, Y0), Z_BACK, zfun, 0, -1, 63)
    wall("spine", (X_S, X_S), (Y0, Y1), Z_BACK, zfun, 1, 0, 64, base=.34)
    # the strip
    xg, yg = np.meshgrid(np.linspace(X_H, X_S + .02, 8), yy)
    z_strip = float(h_block(0.0) + (N + 1) * DZ * .55 + .03)
    st = mesh_grid("strip", xg.astype(np.float32), yg.astype(np.float32), np.full_like(xg, z_strip).astype(np.float32),)
    st.data.materials.append(plain("stripc", .055, rough=.62, spec=.4))
    stitching(z_strip)
    # the leaves. k = 0 is the top one, the first to turn. The last to turn shows its back on the right; the next one shows its front on the left.
    pool = ("page_a.png", "page_b.png", "page_c.png")
    for k in range(N + 1):
        layer = N - k
        m_ = k
        if k == N - 1:
            mat = page_material(f"leaf{k}", pool[k % 3], "page_R.png")
        elif k == N:
            mat = page_material(f"leaf{k}", "page_L.png", pool[k % 3])
        else:
            mat = page_material(f"leaf{k}", pool[k % 3], pool[(k + 1) % 3])
        r = np.random.default_rng(LEAF_SEED + k)
        dy = r.normal(0, .022)
        length = L + r.normal(0, .028)
        visible = k >= N - 1 or k < 3
        ns, ny = (NS, NY) if visible else (70, 34)
        sheets[k] = Sheet(f"leaf{k}", mat, length, Y0 + dy, Y1 + dy, (lambda s, l=layer: z_closed(s, l)), (lambda s, m=m_: z_open(s, m)),
                          bend=.95 + r.normal(0, .08), curl=.55, twist=.26 + r.normal(0, .05), wave=.024, seed=LEAF_SEED + k, ns=ns, ny=ny)
    # the cover, its picture spanning the 14.5 cm of the closed book
    cl = L + .06
    sheets["cover"] = Sheet("cover", cover_material("cover.png"), cl, Y0 - .05, Y1 + .05,
                            lambda s: z_closed(s, N + 1.4) + .03, lambda s: .045 + (.10 + STRIP - .045) * (1 - sm((s - .35) / 7.5)),
                            bend=.62, curl=.42, twist=.16, wave=.008, seed=LEAF_SEED + 99,
                            uv_front=lambda s: (X_H - s - X_F) / 14.5, uv_back=lambda s: 1 - (X_H - s - X_F) / 14.5)
    sol = sheets["cover"].o.modifiers.new("thick", "SOLIDIFY")
    sol.thickness, sol.offset = .045, -1
    return sheets


def window_verts(cols, rows, cw, ch, bar, offset, rect=None, dist=95.0):
    """The bars and the wall of a window, as quads. rect (x0, x1, y0, y1) is the opening in the wall if it is narrower than the lattice; the count of them depends only on cols and rows."""
    d, u, v = (Vector(a) for a in window_axes())
    c = Vector(offset) - d * dist
    me_v, me_f = [], []
    w, h = cols * cw, rows * ch
    x0, x1, y0, y1 = rect or (-w / 2, w / 2, -h / 2, h / 2)

    def box(cx, cy, a, b):
        i = len(me_v)
        p = c + u * cx + v * cy
        for sx_, sy_ in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            me_v.append(tuple(p + u * sx_ * a + v * sy_ * b))
        me_f.append((i, i + 1, i + 2, i + 3))

    for i in range(cols + 1):
        box(-w / 2 + i * cw, 0, bar / 2, h / 2)
    for j in range(rows + 1):
        box(0, -h / 2 + j * ch, w / 2, bar / 2)
    big = 400
    box(x0 - big, 0, big, h + big)           # the wall around the opening
    box(x1 + big, 0, big, h + big)
    box((x0 + x1) / 2, y1 + big, (x1 - x0) / 2, big)
    box((x0 + x1) / 2, y0 - big, (x1 - x0) / 2, big)
    return me_v, me_f


def window(**win):
    """The lattice the moon comes through, out of shot, in the moon's path. Only its shadow is seen. offset is the point on the desk at the centre of the window: with odd
    counts of cells, that is the middle of a cell. A cell of ch throws a shadow twice as long on the desk, at this low a moon."""
    me_v, me_f = window_verts(**win)
    me = bpy.data.meshes.new("window")
    me.from_pydata(me_v, [], me_f)
    o = bpy.data.objects.new("window", me)
    bpy.context.collection.objects.link(o)
    o.visible_camera = False
    o.visible_glossy = False
    return o


def moon_light(direction, energy, angle=.32):
    s = bpy.data.lights.new("moon", "SUN")
    s.energy, s.angle = energy, math.radians(angle)
    o = bpy.data.objects.new("moon", s)
    bpy.context.collection.objects.link(o)
    o.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
    return o


def fill(strength):
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)
    bg.inputs[1].default_value = strength


POEM_SHOTS = ("sheet", "seal", "fading", "sheet_blank", "seal_only", "sheet_wet")     # one camera, one light; what is on the paper is all that changes
SHEET_TEX = dict(sheet=("sheet.png", "sheet_ink.png", 1.0), seal=("sheet_seal.png", "sheet_ink.png", 1.0), fading=("sheet_fading.png", "sheet_fading_ink.png", 1.0),
                 sheet_blank=("sheet_blank.png", "sheet_ink.png", 0.0), seal_only=("sheet_seal_only.png", "sheet_ink.png", 0.0), sheet_wet=("sheet.png", "sheet_ink_full.png", 1.0),
                 closing=("sheet_seal_only.png", "sheet_ink.png", 0.0), end=("sheet_seal_only.png", "sheet_ink.png", 0.0))


# ---- the sheet ---------------------------------------------------------------------------------------------------------------------------------------
def sheet_materials(picture, ink_map, wet):
    """New paper: its deckle comes from the picture's alpha, and where the ink is still wet it is glossy and stands a hair proud of the sheet."""
    m = bpy.data.materials.new("sheet")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Specular IOR Level"].default_value = .5
    t, ink = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeTexImage")
    t.image, ink.image = tex(picture), tex(ink_map, "Non-Color")
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    r_ = nt.nodes.new("ShaderNodeMapRange")
    r_.inputs["To Min"].default_value, r_.inputs["To Max"].default_value = .8, .8 - .62 * wet
    nt.links.new(ink.outputs[0], r_.inputs["Value"])
    nt.links.new(r_.outputs[0], b.inputs["Roughness"])
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value, bp.inputs["Distance"].default_value = .6 * wet, .012
    nt.links.new(ink.outputs[0], bp.inputs["Height"])
    nt.links.new(bp.outputs[0], b.inputs["Normal"])
    tr, ms = nt.nodes.new("ShaderNodeBsdfTranslucent"), nt.nodes.new("ShaderNodeMixShader")
    ms.inputs[0].default_value = .12
    nt.links.new(t.outputs["Color"], tr.inputs[0])
    nt.links.new(b.outputs[0], ms.inputs[1])
    nt.links.new(tr.outputs[0], ms.inputs[2])
    alpha, none = nt.nodes.new("ShaderNodeMixShader"), nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(t.outputs["Alpha"], alpha.inputs[0])
    nt.links.new(none.outputs[0], alpha.inputs[1])
    nt.links.new(ms.outputs[0], alpha.inputs[2])
    nt.links.new(alpha.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    return m


def poem_sheet(picture, ink_map, wet):
    """One sheet lying close on the desk: never quite flat, and two corners still remembering the roll it came from."""
    u_, v_ = np.meshgrid(np.linspace(0, 1, 420), np.linspace(0, 1, 320))
    lx, ly = (u_ - .5) * SW, (v_ - .5) * SH
    n = spectral(256, 3.4, 77)
    fx, fy = u_ * 254.999, v_ * 254.999
    ix, iy = fx.astype(np.int32), fy.astype(np.int32)
    ax, ay = fx - ix, fy - iy
    wave = ((n[iy, ix] * (1 - ax) + n[iy, ix + 1] * ax) * (1 - ay) + (n[iy + 1, ix] * (1 - ax) + n[iy + 1, ix + 1] * ax) * ay) * .016
    e, f = np.minimum(u_, 1 - u_) * SW, np.minimum(v_, 1 - v_) * SH
    corner = lambda cx, cy, r, h_: h_ * np.exp(-((lx - cx) ** 2 + (ly - cy) ** 2) / r ** 2)
    z = .015 + (wave - wave.min()) + .02 * np.exp(-e / 1.4) + .012 * np.exp(-f / 1.0)
    z = z + corner(SW / 2, -SH / 2, 5.0, .14) + corner(-SW / 2, SH / 2, 4.0, .10) + corner(SW / 2, SH / 2, 2.8, .06)
    x = SHEET_C[0] + lx * math.cos(SHEET_ROT) - ly * math.sin(SHEET_ROT)
    y = SHEET_C[1] + lx * math.sin(SHEET_ROT) + ly * math.cos(SHEET_ROT)
    o = mesh_grid("sheet", x.astype(np.float32), y.astype(np.float32), z.astype(np.float32))
    o.data.materials.append(sheet_materials(picture, ink_map, wet))
    o.visible_shadow = False
    # its shadow is a soft grey line laid on the wood, so it can never go darker than the wood's own grain
    u_, v_ = np.meshgrid(np.linspace(-.5, .5, 2), np.linspace(-.5, .5, 2))
    lx, ly = u_ * (SW + 3.0), v_ * (SH + 3.0)
    x = SHEET_C[0] + lx * math.cos(SHEET_ROT) - ly * math.sin(SHEET_ROT)
    y = SHEET_C[1] + lx * math.sin(SHEET_ROT) + ly * math.cos(SHEET_ROT)
    o = mesh_grid("sheet_shadow", x.astype(np.float32), y.astype(np.float32), np.full((2, 2), .004, np.float32))
    m = bpy.data.materials.new("sheet_shadow")
    m.use_nodes = True
    nt = m.node_tree
    t, mix, none, dark = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeMixShader"), nt.nodes.new("ShaderNodeBsdfTransparent"), nt.nodes.new("ShaderNodeBsdfDiffuse")
    t.image = tex("sheet_shadow.png", "Non-Color")
    dark.inputs["Color"].default_value = (0, 0, 0, 1)
    nt.links.new(t.outputs["Color"], mix.inputs[0])
    nt.links.new(none.outputs[0], mix.inputs[1])
    nt.links.new(dark.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    o.data.materials.append(m)
    o.visible_shadow = False


def gleam(cam_pos, at, width, depth, strength):
    """What wet ink has to reflect: a soft pane of night sky in the mirror direction, seen only in reflections, so that a patch of the sheet catches it."""
    c = (2 * at[0] - cam_pos[0], 2 * at[1] - cam_pos[1], cam_pos[2])
    x_, y_ = np.meshgrid(np.linspace(c[0] - width / 2, c[0] + width / 2, 2), np.linspace(c[1] - depth / 2, c[1] + depth / 2, 2))
    o = mesh_grid("gleam", x_.astype(np.float32), y_.astype(np.float32), np.full((2, 2), c[2], np.float32), uv=False)
    m = bpy.data.materials.new("gleam")
    m.use_nodes = True
    nt = m.node_tree
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    o.data.materials.append(m)
    o.visible_camera = o.visible_diffuse = o.visible_transmission = o.visible_shadow = o.visible_volume_scatter = False
    return em


# ---- shots -------------------------------------------------------------------------------------------------------------------------------------------
COL_S = 1.2 + 2.5 * 1.12         # the column 昔者蒼頡 on the right-hand page, by distance from the hinge
TOP_CM = 3.4


def state(shot, t):
    if shot in ("half", "closing"):
        return cover_u(t), [leaf_u(k, t) for k in range(N)]
    if shot in ("spread", "column") + POEM_SHOTS:
        return 1.0, [1.0] * N
    return 0.0, [0.0] * N


def set_dof(cam, focus_at, fstop):
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = (Vector(focus_at) - cam.location).length
    cam.data.dof.aperture_fstop = fstop


def column_place():
    """The column's x, the top of its first character, and the page's height there, on the top leaf of the turned stack."""
    return X_H + COL_S, Y1 - TOP_CM - .13, float(z_open(np.array(COL_S), N - 1))


def shot_key(shot):
    if shot == "column":
        return column_key(*column_place())
    return KEYS["poem" if shot in POEM_SHOTS else shot]


def main():
    raw = SHOT + "_raw" if SHOT == "sheet" else SHOT      # sheet.png is the contact sheet
    for tail in (".png", "_z.png"):                       # render() skips what exists
        p = os.path.join(DIR, raw + tail)
        if os.path.exists(p):
            os.makedirs(os.path.join(DIR, "_old"), exist_ok=True)
            os.replace(p, os.path.join(DIR, "_old", raw + tail))
    t = opt("t", dict(cover=0.0, half=1.78, spread=4.8, column=7.4, closing=1.78).get(SHOT, 0.0))
    sc = reset(samples=SAMPLES, pct=PCT)
    sc.unit_settings.scale_length = .01
    sc.render.film_transparent = False
    sc.cycles.max_bounces = 8
    sheets = build()
    if SHOT in SHEET_TEX:
        poem_sheet(*SHEET_TEX[SHOT])
    cu, lu = state(SHOT, t)
    sheets["cover"].pose(cu)
    for k in range(N):
        sheets[k].pose(lu[k])
    sheets[N].pose(0.0)

    k = shot_key(SHOT)
    cam = camera(k["pos"], k["look"], lens=50, roll=k["roll"])
    if k["fstop"]:
        set_dof(cam, k["focus"], k["fstop"])
    else:
        cam.data.dof.use_dof = False
    if SHOT == "sheet_wet":      # all the ink as wet as the wettest, under a pane of sky as big as the sheet: what ink_book.py lets dry away
        gleam(k["pos"], SHEET_C, 70.0, 60.0, GLEAM_POWER)
    elif SHOT in POEM_SHOTS:
        gleam(k["pos"], GLEAM_AT, GLEAM_SIZE, GLEAM_SIZE, GLEAM_POWER)
    fill(k["dome"])
    moon_light(MOON, k["energy"], k["angle"])
    window(**k["win"])
    render(os.path.join(DIR, raw))


class Sequence:
    """One build, and then for each frame the camera, the moon's window, the light and the leaves set for that moment. Where things move fast a frame is several exposures across a half-open
    shutter, each with its share of the samples, averaged afterwards (ink_book.py), and that is the motion blur."""

    def __init__(self, shot):
        self.shot = shot
        self.sc = reset(samples=SAMPLES, pct=PCT)
        self.sc.unit_settings.scale_length = .01
        self.sc.render.film_transparent = False
        self.sc.render.use_persistent_data = True
        self.sc.cycles.max_bounces = 8
        self.sheets = build()
        self.seen = {}
        self.sheets[N].pose(0.0)
        self.gleam = None
        if shot == "open":
            col_x, top_y, z_page = column_place()
            self.path, self.keys = open_path(z_page, col_x, top_y), open_keys(z_page, col_x, top_y)
            first = open_state(self.path, self.keys, 0.0)
        else:
            poem_sheet(*SHEET_TEX["seal_only"])
            self.path = close_path()
            first = close_state(self.path, seq_time(shot, 0))
            self.gleam = gleam(KEYS["poem"]["pos"], GLEAM_AT, GLEAM_SIZE, GLEAM_SIZE, GLEAM_POWER)
        self.cam = camera((0, 0, 50), (0, 0, 0), lens=50)
        self.moon = moon_light(MOON, first["energy"], first["angle"])
        self.win = window(**first["win"])
        self.cam.data.dof.use_dof = True

    def set(self, t):
        st = open_state(self.path, self.keys, t) if self.shot == "open" else close_state(self.path, t)
        p, r, u, f = self.path.camera(t)
        self.cam.location = tuple(p)
        self.cam.rotation_euler = Matrix(np.stack([r, u, -f], 1).tolist()).to_euler()
        self.cam.data.dof.focus_distance = (Vector(st["focus"]) - self.cam.location).length
        self.cam.data.dof.aperture_fstop = st["fstop"]
        self.moon.data.energy, self.moon.data.angle = st["energy"], math.radians(st["angle"])
        fill(st["dome"])
        verts, _ = window_verts(**st["win"])
        self.win.data.vertices.foreach_set("co", np.array(verts, np.float32).ravel())
        self.win.data.update()
        if self.gleam is not None:
            self.gleam.inputs["Strength"].default_value = st["gleam"]
        clock = open_book(t) if self.shot == "open" else close_book(t)
        self.pose("cover", cover_u(clock))
        for k in range(N):
            self.pose(k, leaf_u(k, clock))

    def pose(self, k, u):
        if self.seen.get(k) != u:      # a leaf at rest is not sent to the renderer again
            self.sheets[k].pose(u)
            self.seen[k] = u

    def run(self, a, b, sub, step=1, where="seq"):
        os.makedirs(os.path.join(DIR, where), exist_ok=True)
        for n in range(a, b + 1, step):
            t = seq_time(self.shot, n)
            many = sub > 1 and needs_blur(t, self.shot)
            self.sc.cycles.samples = max(SAMPLES // sub, 8) if many else SAMPLES
            name = os.path.join(DIR, where, f"{self.shot}_{n:04d}")
            for i in range(sub if many else 1):
                out = f"{name}_s{i}" if many else name
                self.sc.cycles.seed = i       # each exposure its own noise, to average away; the same from frame to frame, so that it does not flicker
                if not os.path.exists(out + ".png"):
                    self.set(t + (i / (sub - 1) - .5) * shutter(t, self.shot) if many else t)
                    t0 = time.time()
                    render(out, depth=False)
                    print(f"frame {n} {out[-8:]} {time.time() - t0:.1f}s", flush=True)


main() if SHOT not in ("open", "close") else Sequence(SHOT).run(*(int(v) for v in opt("frames", f"0-{CLOSE_FRAMES - 1}").split("-")), opt("sub", 1), opt("step", 1), opt("dir", "seq"))
