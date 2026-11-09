"""Characters cut into stone, and the stone they are cut into, shared by the scenes where the writing is carved (ink3d_attn.py, ink3d_drip.py).

Blender's Python has no scipy or PIL, so the shapes of the characters are made once in the render environment:

    python ink3d_carve.py                      # distance fields -> ink/film_out/3d/glyphs/

Everything else runs inside Blender: rock() makes a fractured face over any grid, cut() takes a character out of it, strands() and bundle()
make ink threads out of separate hairs, so that the gaps in a thick one are real gaps with light coming through.
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GLYPHS = os.path.join(ROOT, "ink", "film_out", "3d", "glyphs")
TILE, PAD = 448, 56
_tiles = {}
# every character the two chapters need, by font
SETS = {"kaih": "举头望明月低思故乡家亲念的了在是有和人这中大为上个国不说们到地也子时道出而要于就下得可你年生自会那后能对着事其里所去行过十用发天如然作方成者多日都三小军二无同么经法当起与好看学进种将还分此心前面又定见只主没公从",
        "xing": "鬼哭当年缘底事早知今夜字成精",
        "mono": "0123456789+-.",
        "song": "太史公曰余读孔氏书想见其为人亲爱的今晚月亮很好你那边呢有人吗楼主还在关雎鸠河之洲本院判决如下明几时把酒问青天",
        "kai": "太史公曰余读孔氏书想见其为人亲爱的今晚月亮很好你那边呢有人吗楼主还在关雎鸠河之洲本院判决如下明几时把酒问青天"}


def tile_path(ch, kind):
    return os.path.join(GLYPHS, f"{kind}_{ord(ch):x}.npy")


def prep():
    from scipy.ndimage import distance_transform_edt
    from ink_film import glyph_mask
    os.makedirs(GLYPHS, exist_ok=True)
    for kind, chars in SETS.items():
        for ch in dict.fromkeys(chars):
            if not os.path.exists(tile_path(ch, kind)):
                m = glyph_mask(ch, kind, TILE, PAD) > .5
                np.save(tile_path(ch, kind), ((distance_transform_edt(~m) - distance_transform_edt(m)) / TILE).astype(np.float32))


def tile(ch, kind):
    k = (ch, kind)
    if k not in _tiles:
        _tiles[k] = np.load(tile_path(ch, kind))
    return _tiles[k]


def sample(t, x, y):
    """Bilinear lookup, clamped at the edges."""
    h, w = t.shape
    x, y = np.clip(x, 0, w - 1.001), np.clip(y, 0, h - 1.001)
    ix, iy = x.astype(np.int32), y.astype(np.int32)
    fx, fy = x - ix, y - iy
    return (t[iy, ix] * (1 - fx) + t[iy, ix + 1] * fx) * (1 - fy) + (t[iy + 1, ix] * (1 - fx) + t[iy + 1, ix + 1] * fx) * fy


def cut(H, A, u1, v1, ch, kind, cu, cv, size, depth=.16, slope=1.1, p=1.0, ink=.03, chip=None, stain=None, fresh=0.0, edge=.012, dust=0.0):
    """Takes a character out of the height field H (rows v1, columns u1) as a V-section cut, and inks the albedo A where it was cut.
    p: how much has been cut so far, from the top down. fresh: 1 for a new cut, pale stone with no ink in it yet. stain: an array the ink runs into.
    dust: how far the pale stone dust lies around a new cut, in metres."""
    half = size * (TILE + 2 * PAD) / TILE / 2
    ca, cb = np.searchsorted(u1, cu - half), np.searchsorted(u1, cu + half)
    ra, rb = np.searchsorted(v1, cv - half), np.searchsorted(v1, cv + half)
    if ca >= cb or ra >= rb or p <= 0:
        return
    U, V = np.meshgrid(u1[ca:cb], v1[ra:rb])
    T = tile(ch, kind)
    d = sample(T, (U - cu) / size * TILE + T.shape[1] / 2, (cv - V) / size * TILE + T.shape[0] / 2) * size
    if chip is not None:      # the edge of a chisel cut is not clean
        d = d + edge * sample_wrap(chip, U * 37, V * 37) + edge * .5 * sample_wrap(chip, U * 131 + 9, V * 131 + 3)
    done = smooth((p * 1.3 - np.clip((cv + size / 2 - V) / size, 0, 1)) / .3)
    g = np.clip(-d * slope, 0, depth) * done
    H[ra:rb, ca:cb] -= g
    inside = smooth(-d / (edge * 1.4 + 1e-4)) * done
    colour = ink + (.92 - ink) * fresh
    A[ra:rb, ca:cb] = A[ra:rb, ca:cb] * (1 - inside) + colour * inside
    if dust > 0:
        halo = smooth(1 - d / dust) * (1 - inside) * done * np.clip(.7 + .5 * sample_wrap(chip, U * 19, V * 19), 0, 1)
        A[ra:rb, ca:cb] = A[ra:rb, ca:cb] * (1 - halo) + .97 * halo
    if stain is not None:
        stain[ra:rb, ca:cb] = np.maximum(stain[ra:rb, ca:cb], inside * (1 - fresh))


def sample_wrap(t, x, y):
    h, w = t.shape
    ix, iy = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    fx, fy = x - ix, y - iy
    a, b, c, d = ix % w, (ix + 1) % w, iy % h, (iy + 1) % h
    return (t[c, a] * (1 - fx) + t[c, b] * fx) * (1 - fy) + (t[d, a] * (1 - fx) + t[d, b] * fx) * fy


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def worley(a, b, seed):
    """For points in cell coordinates: which jittered site is nearest, where it is, and how far the second nearest is beyond it (0 on a crack)."""
    r = np.random.default_rng(seed)
    M = 256
    jx, jy = r.random((M, M), np.float32), r.random((M, M), np.float32)
    ia, ib = np.floor(a).astype(np.int32), np.floor(b).astype(np.int32)
    f1, f2 = np.full(a.shape, 9.0, np.float32), np.full(a.shape, 9.0, np.float32)
    ca, cb = np.zeros(a.shape, np.int32), np.zeros(a.shape, np.int32)
    sa, sb = np.zeros(a.shape, np.float32), np.zeros(a.shape, np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            xx, yy = ia + dx, ib + dy
            px, py = xx + jx[yy % M, xx % M], yy + jy[yy % M, xx % M]
            d = np.hypot(a - px, b - py)
            near = d < f1
            f2 = np.where(near, f1, np.minimum(f2, d))
            f1 = np.where(near, d, f1)
            ca, cb = np.where(near, xx, ca), np.where(near, yy, cb)
            sa, sb = np.where(near, px, sa), np.where(near, py, sb)
    return ca % M, cb % M, sa, sb, f2 - f1


def rock(U, V, seed, beds=.3, joints=.16):
    """A weathered cliff face over the grid (metres), the same rock as the film's mountains: fluted from top to bottom by water, each smaller flute gathering on the
    crests of the larger, with the faint steps of the beds across it and a few hard joints. Returns the height and how deep in a crack each point sits."""
    from ink3d import blur, spectral
    r = np.random.default_rng(seed)
    t = blur(spectral(1024, 1.0, seed), 2.2)
    t /= t.std()
    warp = spectral(1024, 3.0, seed + 1)
    wu = sample_wrap(warp, U * 3.0, V * 1.5) * 2.0 + .4 * sample_wrap(spectral(1024, 2.2, seed + 5), U * 10, V * 7)
    H, w = np.zeros(U.shape, np.float32), np.ones(U.shape, np.float32)
    comes = np.clip(.62 + .55 * sample_wrap(warp, U * 9 + 300, V * 26 + 100), .1, 1.1)
    for lam, stretch, amp in ((16, 6, 2.3), (8, 7, 1.25), (4, 9, .7), (2, 11, .34), (1, 12, .15), (.5, 10, .05)):
        ox, oy = r.uniform(0, 1024, 2)
        v = 1 - np.sqrt(np.tanh(sample_wrap(t, (U + wu) * 5.0 / lam + ox, V * 5.0 / (lam * stretch) + oy) * .9) ** 2 + .012)
        v = v * v
        H += amp * v * w * (comes if lam > 3 else 1)
        w = np.clip(v * 1.9, 0, 1)
    bed = (V + U * .16 + sample_wrap(warp, U * 2, V * 2) * 1.1) / 5.3
    H += beds * ((bed - np.floor(bed)) ** 3 * 2 - 1)
    ca, cb, sa, sb, gap = worley(U / 6.5, (V + U * .16) / 3.0, seed + 2)
    crack = np.exp(-gap / .012) * smooth(sample_wrap(warp, U * 7, V * 7) * 1.3 - .1)      # joints: straight, and only here and there
    H -= joints * crack
    return H.astype(np.float32), crack.astype(np.float32)


def stains(A, U, V, H, crack, seed, amount=.3):
    """What rain leaves on a rock face: dark in the hollows, and dark streaks running straight down from the cracks and ledges, in patches."""
    from ink3d import blur, spectral
    fine = spectral(1024, 1.1, seed)
    wide = spectral(1024, 3.0, seed + 1)
    streak = sample_wrap(fine, U * 9.0, V * .5)
    patch = smooth(sample_wrap(wide, U * 3.2, V * 1.3) * .9 + .1)
    A -= amount * smooth(streak * .8 + .25) * patch
    A -= .38 * crack
    hollow = H - blur(H, 9)
    A += .34 * np.clip(hollow / (hollow.std() + 1e-6), -2.2, .7) * .5
    A += .07 * sample_wrap(fine, U * 40, V * 40)
    return np.clip(A, .03, 1)


def run_down(mask, keep=.986):
    """Ink running down the rock from where it was put: mask rows go upward with v, so it runs toward lower rows."""
    out = mask.copy()
    n = out.shape[0]
    for i in range(n - 2, -1, -1):
        out[i] = np.maximum(out[i], out[i + 1] * keep)
    return out


def smooth1d(r, n, sigma):
    """Noise along a line, std 1: a random walk seen through a window."""
    a = r.normal(size=n + int(sigma * 6))
    k = np.exp(-.5 * (np.arange(-int(sigma * 3), int(sigma * 3) + 1) / max(sigma, .5)) ** 2)
    v = np.convolve(a, k, "same")[int(sigma * 3):][:n]
    return v / (v.std() + 1e-6)


def bundle(ctrl, half, count, r0, dry, seed, prog=1.0, twist=1.0, n=96, thin=1.0, flat=.4, stray=.14):
    """An ink thread as count separate hairs following a cubic in (u, v, w), w out of the rock: a rope of nearly even width, pinched only where it is tied on at each end,
    never a leaf. half: half its width. It twists, so it narrows and widens as it turns; a few hairs stray from it. dry: how much of it has run out by the tail, in real gaps
    between the hairs. Returns (points n×3, radii n) per hair."""
    r = np.random.default_rng(seed)
    p = np.array(ctrl, np.float64)
    s = np.linspace(0, 1, n)
    q = s[:, None]
    c = (1 - q) ** 3 * p[0] + 3 * (1 - q) ** 2 * q * p[1] + 3 * (1 - q) * q ** 2 * p[2] + q ** 3 * p[3]
    T = np.gradient(c, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None] + 1e-9
    side = np.cross(T, [0, 0, 1.0])
    side /= np.linalg.norm(side, axis=1)[:, None] + 1e-9
    D = np.cross(T, side)
    width = half * (smooth(s / .1) * smooth((1 - s) / .1)) ** .6 * (.8 + .2 * smooth1d(r, n, 18)) + .002
    ang = r.uniform(0, 6.28) + twist * np.pi * s
    wide = np.cos(ang)[:, None] * side + np.sin(ang)[:, None] * D
    thick = -np.sin(ang)[:, None] * side + np.cos(ang)[:, None] * D
    dp = dry * smooth((s - .08) / .4) * smooth((1 - s) / .14)
    out = []
    for _ in range(count):
        o = r.uniform(-1, 1)
        e = r.normal(0, flat)
        loose = r.random() < stray
        k = 5.0 if loose else 1.0
        wob = smooth1d(r, n, 9) * .06 * k
        drift = smooth1d(r, n, 16) * .08 * k
        P = c + wide * ((o + wob) * width)[:, None] + thick * ((e + drift) * width)[:, None]
        press = smooth1d(r, n, 14) * .7 + r.normal(0, .55)
        on = smooth((press + 1.7 - 3.0 * dp * (.55 + .45 * abs(o))) / .4)
        R = r0 * r.uniform(.5, 1.2) * (.55 if loose else 1.0) * on * np.sin(np.pi * np.clip(s, .004, .996)) ** .25 * smooth((prog * 1.06 - s) / .05) * thin
        out.append((P, R.astype(np.float32)))
    return out


def strands(name, paths, mat, shadow=True):
    """Hairs, chips or rills as one curve object. paths: (points n×3 in the world, radii n)."""
    import bpy
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 1.0
    cu.bevel_resolution = 1
    cu.resolution_u = 1
    cu.fill_mode = "FULL"
    for P, R in paths:
        s = cu.splines.new("POLY")
        s.points.add(len(P) - 1)
        co = np.ones((len(P), 4), np.float32)
        co[:, :3] = P
        s.points.foreach_set("co", co.ravel())
        s.points.foreach_set("radius", R)
    o = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(o)
    cu.materials.append(mat)
    o.visible_shadow = shadow
    return o


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    prep()
    print(len(os.listdir(GLYPHS)), "glyphs in", GLYPHS)
