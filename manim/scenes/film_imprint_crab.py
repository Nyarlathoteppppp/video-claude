"""The block gets up. 94.4 to 102.4 s of the bridge clock: the brick with everything that was ever pressed into it stirs, gathers itself into the little crab from Claude Code's door and opens its eyes for the first time; yawns and stretches; turns to us and waves; stands. No light effects, no particles but real crumbs: only the clay moving, as in a stop-motion film, twelve poses a second.

    blender -b --factory-startup -P film_imprint_crab.py -- still c3 97.5                 # name, film time
    ... -- still c3_grey 97.5 --grey
    ... -- still c3_crop 97.5 --look-at 54 -49 62 3.4 --dense 58 118 -72 10

The skin is the whole surface of the block, a closed mesh that keeps its material coordinates. Every vertex knows two places: where it is on the block and where it will be on the crab. The block's section (160 by 40) has nearly the same perimeter as the crab's (98 by 98), so the skin rolls round without stretching along its length: the top goes over the crab's back, top and forehead, the front wall down its face. Claws are drawn out of the end walls and legs out of the bottom. Relief, the sim's height field state_g3 with the groove in it, is carried along the normal: nothing is painted on. Volume is measured on the closed mesh each frame and put right.
"""
import ast
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from film_imprint_clay import A, B, BOARD_RECT, BOARD_PX, OUT, THICK
from film_imprint3d import CORNER, Nodes, axis_samples, box_axis, camera, clay_material, corner_warp, dawn, set_up, sm
from ink3d import aim, reset, spectral

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = "g3"
U = 13.5             # mm to one of his squares across (a square is twice as tall as wide): fixes his size, and so his volume
ROUND = 8.5           # how round his body's edges are
KEEP_RELIEF = .8      # what is left of the marks' depth once they have been rolled round him


def clawd():
    """His real shape, from where the app's mark is kept."""
    with open(os.path.join(HERE, "ink_mac.py"), encoding="utf-8") as f:
        for node in ast.parse(f.read()).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "CLAWD":
                return ast.literal_eval(node.value)


G = clawd()
FIRST, LAST = G[0].index("#"), G[0].rindex("#") + 1
MID = (FIRST + LAST) / 2
BODY_ROWS = [i for i, r in enumerate(G) if r[FIRST] == "#"]
BODY_W, BODY_H, BODY_D = (LAST - FIRST) * U, len(BODY_ROWS) * 2 * U, 6 * U
EYES = [((c + .5 - MID) * U, (i + .5) * 2 * U) for i, r in enumerate(G) if i in BODY_ROWS for c in range(FIRST, LAST) if r[c] == "."]     # across, how far down from the top of him
CLAW_ROW = next(i for i, r in enumerate(G) if "#" in r[:FIRST])
CLAW_L = (FIRST - G[CLAW_ROW].index("#")) * U
LEG_ROW = next(i for i, r in enumerate(G) if i not in BODY_ROWS)
LEGS = [(c + .5 - MID) * U for c, x in enumerate(G[LEG_ROW]) if x == "#"]
LEG_H = 1.85 * U
FLOOR = -THICK
CLAW_Z = (FLOOR + BODY_H - CLAW_ROW * 2 * U - 2 * U, FLOOR + BODY_H - CLAW_ROW * 2 * U)     # how high on him, standing on the board with no legs yet
CLAW_Y = (-BODY_D / 2 + 9.0, -BODY_D / 2 + 9.0 + 2.3 * U)
LEG_Y = (-BODY_D / 2 + 5.0, BODY_D / 2 - 12.0)
PERIM = 2 * (BODY_D + BODY_H)
BENCH, BEYOND = (.0024, .0021, .0018), (.0024, .0021, .0018)
ON_BACK = 26.0        # how much of the block's top goes over onto his back; then his top; what is left comes down his forehead


def curve(*pts):
    """A move in poses: eased between one and the next, so it goes, stops, goes."""
    tt, vv = np.array([p[0] for p in pts]), np.array([p[1] for p in pts], np.float64)

    def f(t):
        t = np.asarray(t, np.float64)
        i = np.clip(np.searchsorted(tt, t, side="right") - 1, 0, len(tt) - 2)
        u = np.clip((t - tt[i]) / (tt[i + 1] - tt[i]), 0, 1)
        return vv[i] + (vv[i + 1] - vv[i]) * (u * u * (3 - 2 * u))
    return f


# The bridge clock moved on 2026-11-01: the time given on the command line is the new one (born 97.1 to 100.1, hello 102.9 to 104.3, cut 105.1). The curves and comments below stay in the old one, CLOCK_SHIFT earlier; every stretch keeps its length.
CLOCK_SHIFT = 2.7

# 94.4 to 97.4, born: it stirs once, as if something had breathed in it, and lies still. Then it draws in along its length (a small swell outwards before it goes), rolls up, the claws come, with two heaves it gets up off the board, right side a little ahead, and turns a little towards the sun. Last of all the eyes sink in, shut, and open for the first time.
STIR = curve((94.4, 0), (94.52, 1), (94.66, -.35), (94.82, 0))
GATHER = curve((94.86, 0), (94.92, -.02), (95.22, .17), (95.4, .2), (95.95, .82), (96.3, .97), (96.7, 1))
ROLL = curve((94.95, 0), (95.25, .08), (95.45, .1), (96.2, .62), (96.5, .72), (97.0, 1.025), (97.25, 1))
CLAW = curve((95.5, 0), (96.15, .52), (96.32, .5), (96.72, 1.13), (96.95, .97), (97.1, 1))
LIFT = curve((95.75, 0), (96.15, .3), (96.32, .27), (96.75, .68), (96.9, .62), (97.12, 1.07), (97.25, 1))
EYE_L, EYE_R = curve((96.98, 0), (97.18, 1)), curve((97.05, 0), (97.25, 1))
EYE_OPEN = curve((97.16, .05), (97.38, 1.06), (97.47, 1))
COLOUR = curve((94.85, -.3), (97.5, 1.5))
# 97.8 to 100.2, the yawn: a little sink first, then up and back; held; let go, a small bounce.
STRETCH = curve((97.8, 0), (98.1, -.12), (99.0, 1), (99.45, 1.04), (100.0, .08), (100.2, -.06), (100.36, 0), (100.7, .07), (101.52, .07), (101.92, -.07), (102.2, 0))
MOUTH = curve((97.9, 0), (98.2, .1), (98.45, .14), (99.0, 1), (99.5, .93), (99.95, .12), (100.1, 0))
SQUINT = curve((97.8, 0), (98.2, .85), (98.6, 1), (99.85, 1), (100.1, 0))
BLINK = curve((97.58, 0), (97.64, 1), (97.7, 1), (97.78, 0), (100.22, 0), (100.3, 1), (100.38, 1), (100.48, 0), (102.0, 0), (102.08, 1), (102.14, 1), (102.24, 0))
RAISE = curve((98.0, 0), (98.85, 1), (99.75, 1), (100.08, -.06), (100.22, 0))
# 100.2 to 101.6, hello: he turns to us, in two goes; one claw comes up from his side, stays up and waves twice. 101.6 to 102.4 it comes down and he stands.
YAW = curve((96.55, 0), (97.1, -20), (100.2, -20), (100.36, -11), (100.45, -12), (100.62, -2), (100.74, -4))
WAVE = curve((100.34, 0), (100.7, 1), (101.52, 1), (101.9, -.04), (102.06, 0))


def sway(t):
    """The wave, in degrees: twice over, its top at 100.9 and again at 101.3."""
    return 18.0 * float(sm((t - 100.6) / .12) * sm((101.58 - t) / .14)) * math.sin(2 * math.pi * (t - 100.8) / .4)


def shot(t):
    """The block's line of sight, coming down as it stands up until it is seen from just above its own eye line. From then on the camera stays put."""
    return dict(elev=float(curve((94.9, 30), (97.2, 15))(t)), scale=float(curve((94.9, 1.02), (97.2, .88))(t)), target=(0.0, float(curve((94.9, -9), (97.2, -6))(t)), float(curve((94.9, -9), (97.2, 36))(t))))


def light(t, grey=False):
    """The same dawn, the sun higher and swung round to the front of him: over his right shoulder as he faces us, so it reaches his face. And warmer as it climbs."""
    w = 0.0 if grey else float(sm((t - 93.4) / 4.0))
    return dict(az=-30.0, el=float(21.0 + 6.0 * sm((t - 93.4) / 9.0)), e=6.2 if grey else float(6.2 + 3.8 * sm((t - 94.9) / 2.6)), colour=(.79 + .17 * w, .87 + .03 * w, .91 - .1 * w))


def waves(p, length, seed, n=7):
    """Lumps of about that size, smooth, in three dimensions."""
    r = np.random.default_rng(seed)
    out = np.zeros(len(p), np.float32)
    for _ in range(n):
        d = r.normal(size=3)
        d = (d / np.linalg.norm(d)).astype(np.float32)
        out += np.sin((p @ d) * (2 * math.pi / (length * r.uniform(.7, 1.4))) + r.uniform(0, 6.28))
    return out / math.sqrt(n / 2)


def edge(d, soft):
    return sm(d / soft + .5)


def rrect(x, y, cx, cy, hw, hh, rad):
    """Signed distance from a rounded rectangle."""
    rad = np.minimum(rad, np.minimum(hw, hh))
    qx, qy = np.abs(x - cx) - hw + rad, np.abs(y - cy) - hh + rad
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rad


def round_of(sp):
    """Where a length along the block's section (0 at its top's back edge, on along the top, down the front, along the bottom, up the back) falls on his section. Goes round starting a little way down his back."""
    u = (np.asarray(sp, np.float64) * PERIM / 400.0 - ON_BACK * PERIM / 400.0) % PERIM
    D, H = BODY_D, BODY_H
    y = np.interp(u, (0, D, D + H, 2 * D + H, PERIM), (D / 2, -D / 2, -D / 2, D / 2, D / 2))
    z = np.interp(u, (0, D, D + H, 2 * D + H, PERIM), (FLOOR + H, FLOOR + H, FLOOR, FLOOR, FLOOR + H))
    return y.astype(np.float32), z.astype(np.float32)


def along(a, n):
    """a, smoothed by a running mean n samples long, keeping its ends."""
    n = max(int(n), 1) | 1
    return np.convolve(np.pad(a, n // 2, mode="edge"), np.ones(n) / n, mode="valid")


class Skin:
    """The block's whole surface as one closed mesh, and for each vertex where it is on the block and where it goes on the crab."""

    def __init__(self, dense, step, step_wall, step_under):
        z = np.load(os.path.join(OUT, f"state_{STATE}.npz"))
        H, res, lost, burnish, bulge = z["H"], int(z["res"]), float(z["lost"]), z["burnish"], z["bulge"]
        fresh = z["fresh"] if "fresh" in z else None
        xs = axis_samples(-A, A, step, dense[:2] if dense else None, 1 / 16)
        ys = axis_samples(-B, B, step, dense[2:] if dense else None, 1 / 16)[::-1]
        Z = box_axis(H, (B - ys) * res, np.maximum(np.gradient(np.abs(ys)) * res, 1.0), 0)
        Z = box_axis(Z, (xs + A) * res, np.maximum(np.gradient(xs) * res, 1.0), 1)
        del H
        self.xs, self.ys = xs, ys
        nx, ny = len(xs), len(ys)
        nz = int(round(THICK / step_wall))
        tt = np.linspace(0, 1, nz + 1) ** .96       # along a wall, top to bottom; flat height is -THICK times this
        yb = np.concatenate([np.linspace(-B, -B + 24, int(round(24 / step_wall)), endpoint=False), np.linspace(-B + 24, B, int(round((2 * B - 24) / step_under)) + 1)])     # the first of the underside comes round onto his face
        # one ring round the block's section: top (back to front), front wall, bottom (front to back), back wall
        ring_y = np.concatenate([ys, np.full(nz, -B), yb[1:], np.full(nz - 1, B)])
        ring_t = np.concatenate([np.zeros(ny), tt[1:], np.ones(len(yb) - 1), tt[nz - 1:0:-1]])
        ring_kind = np.concatenate([np.zeros(ny, np.int8), np.ones(nz, np.int8), np.full(len(yb) - 1, 2, np.int8), np.full(nz - 1, 3, np.int8)])
        ring_sp = np.concatenate([B - ys, 2 * B + ring_t[ny:ny + nz] * THICK, 2 * B + THICK + (yb[1:] + B), 4 * B + 2 * THICK - tt[nz - 1:0:-1] * THICK])
        np_ = len(ring_y)
        print("skin: ring", np_, "rings", nx, flush=True)
        X = np.repeat(xs[:, None], np_, 1).astype(np.float32)
        Y = np.repeat(ring_y[None, :], nx, 0).astype(np.float32)
        T = np.repeat(ring_t[None, :], nx, 0).astype(np.float32)
        KIND = np.repeat(ring_kind[None, :], nx, 0)
        SP = np.repeat(ring_sp[None, :].astype(np.float32), nx, 0)
        R = np.zeros_like(X)
        R[:, :ny] = Z.T.astype(np.float32)
        R[:, ny:ny + nz] = Z[-1, :, None].astype(np.float32) * (1 - T[:, ny:ny + nz])
        R[:, ny + nz + len(yb) - 1:] = Z[0, :, None].astype(np.float32) * (1 - T[:, ny + nz + len(yb) - 1:])
        # On the crab the marks go along the normal. The block's rim, and the walls, which only ever hung from the top's edge, must not: a rim rolled onto his face would be a crease across it, and every nick in the edge a stripe running down him. So near the old edges the marks give way to a smooth level, which dies away a little way down the walls.
        RIM = 7.0
        per = (len(xs) - 1) / (2 * A)
        Zf, Zb = along(Z[np.argmin(np.abs(ys + B - RIM))], 9 * per), along(Z[np.argmin(np.abs(ys - B + RIM))], 9 * per)
        Zl, Zr = along(Z[:, np.argmin(np.abs(xs + A - RIM))], 9 * per), along(Z[:, np.argmin(np.abs(xs - A + RIM))], 9 * per)
        wy = sm((B - np.abs(ys)) / RIM)[:, None]
        wx = sm((A - np.abs(xs)) / RIM)[None, :]
        Zc = Z * wy + np.where(ys[:, None] < 0, Zf[None, :], Zb[None, :]) * (1 - wy)
        Zc = Zc * wx + np.where(xs[None, :] < 0, Zl[:, None], Zr[:, None]) * (1 - wx)
        fade = np.exp(-T * THICK / RIM)
        C = np.zeros_like(R)
        C[:, :ny] = Zc.T.astype(np.float32)
        C[:, ny:ny + nz] = Zf[:, None].astype(np.float32) * fade[:, ny:ny + nz]
        C[:, ny + nz + len(yb) - 1:] = Zb[:, None].astype(np.float32) * fade[:, ny + nz + len(yb) - 1:]
        del Zc
        # the two end walls, as a grid inside the rings' last and first
        yi, ti = yb[1:-1], tt[1:nz]
        CY, CT = np.meshgrid(yi, ti)
        CY, CT = CY.astype(np.float32).ravel(), CT.astype(np.float32).ravel()
        n_cap = len(CY)
        ends = np.interp(yi, ys[::-1], Z[::-1, 0]), np.interp(yi, ys[::-1], Z[::-1, -1])
        nb_i = len(yi)
        cap_r = [np.outer(1 - ti, e).astype(np.float32).ravel() for e in ends]
        cap_c = [np.outer(np.exp(-ti * THICK / RIM), np.interp(yi, ys[::-1], e[::-1])).astype(np.float32).ravel() for e in (Zl, Zr)]
        # all vertices: tube, then one end, then the other
        self.nx, self.np_, self.ny, self.nz, self.nb = nx, np_, ny, nz, len(yb)
        self.mx = np.concatenate([X.ravel(), np.full(n_cap, -A, np.float32), np.full(n_cap, A, np.float32)])
        self.my = np.concatenate([Y.ravel(), CY, CY])
        self.t = np.concatenate([T.ravel(), CT, CT])
        self.kind = np.concatenate([KIND.ravel(), np.full(n_cap, 4, np.int8), np.full(n_cap, 5, np.int8)])
        self.r = np.concatenate([R.ravel(), cap_r[0], cap_r[1]])
        self.rq = np.concatenate([C.ravel(), cap_c[0], cap_c[1]])
        sp = np.concatenate([SP.ravel(), np.zeros(n_cap * 2, np.float32)])
        del X, Y, T, KIND, SP, R, C
        n = len(self.mx)
        self.mz = -THICK * self.t
        # on the block
        wx, wy = corner_warp(self.mx, self.my)
        ox, oy = np.zeros(n, np.float32), np.zeros(n, np.float32)
        ox[self.kind == 4], ox[self.kind == 5], oy[self.kind == 1], oy[self.kind == 3] = -1, 1, -1, 1
        cx, cy = np.abs(wx) - (A - CORNER), np.abs(wy) - (B - CORNER)
        corner = (cx > 0) & (cy > 0) & (self.kind != 0) & (self.kind != 2)
        ln = np.hypot(cx[corner], cy[corner]) + 1e-6
        ox[corner], oy[corner] = np.sign(wx[corner]) * cx[corner] / ln, np.sign(wy[corner]) * cy[corner] / ln
        walls = np.isin(self.kind, (1, 3, 4, 5))
        # the walls as they were cut and patted, and bulged by all that went out through them: the same recipe as the block's, film_imprint3d.block
        s = np.select([self.kind == 3, self.kind == 5, self.kind == 1, self.kind == 4], [self.mx + A, 2 * A + (B - self.my), 2 * A + 2 * B + (A - self.mx), 4 * A + 2 * B + (self.my + B)], 0).astype(np.float32)
        N_S, N_T = 4096, 512
        low, wire, fine, pat = spectral(N_S, 3.7, 301, N_T), spectral(N_S, 1.7, 302, N_T), spectral(N_S, 1.0, 303, N_T), spectral(N_S, 2.3, 304, N_T)
        pat = pat * .4                                       # in this frontal sun, at full strength they read as towelling
        si = (s / (4 * A + 4 * B) * (N_S - 1)).astype(np.int32)
        ti_ = (self.t * (N_T - 1)).astype(np.int32)
        local = bulge[np.clip(((B - wy) / 2).astype(np.int32), 0, bulge.shape[0] - 1), np.clip(((wx + A) / 2).astype(np.int32), 0, bulge.shape[1] - 1)]
        local = local / max(float(bulge.mean()), 1e-5) * .7 + .3
        h_side = THICK + float(Z.mean())
        amount = lost / (2 * (2 * A + 2 * B) * h_side * .66)
        bar = 4 * self.t * (1 - self.t)
        self.o = amount * local * np.maximum(bar, 0) ** .8 + low[si, ti_] * .55 * np.minimum(self.t * 5, 1) + wire[si // 3, ti_] * .03 + fine[(si * 14) % N_S, (ti_ * 4) % N_T] * .01 + pat[si, ti_] * .2
        self.o = (self.o + 1.1 * np.exp(-(1 - self.t) * h_side / 2.4) * (1 + low[si, 40] * .5)) * walls
        self.patted = (low[si, ti_] * .55 * np.minimum(self.t * 5, 1) * .3 + pat[si, ti_] * .05) * walls * sm((1 - self.t) * 7)      # what is left of them on him: he has drawn the bulge back in
        del low, wire, fine, pat, si, ti_, local, bar, s
        self.B = np.stack([wx + ox * self.o, wy + oy * self.o, self.mz], -1).astype(np.float32)
        self.out = np.stack([ox * self.o, oy * self.o], -1).astype(np.float32)
        # on the crab: first a box
        qx = self.mx * (BODY_W / (2 * A))
        qy, qz = round_of(sp)
        # the end walls, by a patch between the four edges they share with the tube
        caps = self.kind >= 4
        cy_, ct_ = self.my[caps], self.t[caps]
        S, Tt = (cy_ + B) / (2 * B), ct_

        def edge_at(a):
            return np.stack(round_of(a), -1)
        top, bot, front, back = edge_at(B - cy_), edge_at(2 * B + THICK + (cy_ + B)), edge_at(2 * B + Tt * THICK), edge_at(4 * B + 2 * THICK - Tt * THICK)
        c00, c10, c01, c11 = edge_at(np.array([2 * B]))[0], edge_at(np.array([0.0]))[0], edge_at(np.array([2 * B + THICK]))[0], edge_at(np.array([4 * B + THICK]))[0]
        s_, t_ = S[:, None], Tt[:, None]
        patch = (1 - t_) * top + t_ * bot + (1 - s_) * front + s_ * back - ((1 - s_) * (1 - t_) * c00 + s_ * (1 - t_) * c10 + (1 - s_) * t_ * c01 + s_ * t_ * c11)
        qy[caps], qz[caps] = np.clip(patch[:, 0], -BODY_D / 2, BODY_D / 2), np.clip(patch[:, 1], FLOOR, FLOOR + BODY_H)
        # then its edges rounded
        half = np.array((BODY_W / 2, BODY_D / 2, BODY_H / 2), np.float32)
        q = np.stack([qx, qy, qz - (FLOOR + BODY_H / 2)], -1).astype(np.float32)
        q[self.kind == 4, 0], q[self.kind == 5, 0] = -half[0], half[0]
        inner = np.clip(q, -(half - ROUND), half - ROUND)
        d = q - inner
        nrm = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
        self.n = nrm.astype(np.float32)
        self.Q = (inner + nrm * ROUND + (0, 0, FLOOR + BODY_H / 2)).astype(np.float32)
        Qx, Qy, Qz = self.Q[:, 0], self.Q[:, 1], self.Q[:, 2]
        # the parts he draws out of himself, in terms of where they are on him, so they go with the clay
        self.side = np.sign(Qx).astype(np.float32)
        on_end = sm((np.abs(Qx) - (BODY_W / 2 - 7.0)) / 7.0)
        self.w_claw = (edge(Qy - CLAW_Y[0], 3.6) * edge(CLAW_Y[1] - Qy, 3.6) * edge(Qz - CLAW_Z[0], 3.6) * edge(CLAW_Z[1] - Qz, 3.6) * on_end).astype(np.float32)
        under = sm((-self.n[:, 2] - .55) / .35)
        wl = np.zeros(n, np.float32)
        for xc in LEGS:
            wl = np.maximum(wl, edge(Qx - (xc - U / 2), 2.6) * edge((xc + U / 2) - Qx, 2.6))
        self.w_leg = (wl * edge(Qy - LEG_Y[0], 3.0) * edge(LEG_Y[1] - Qy, 3.0) * under).astype(np.float32)
        # when a claw rides up, the skin of his side slides with it, in its own plane: more stretched below it, bunched above it, and no sheet of clay hanging from where the claw used to be
        below = sm((Qz - FLOOR - ROUND) / (CLAW_Z[0] - FLOOR - ROUND))
        above = 1 - sm((Qz - CLAW_Z[1]) / (FLOOR + BODY_H - ROUND - CLAW_Z[1]))
        self.slide = (on_end * below * above * sm((Qy - (-BODY_D / 2 + 2.0)) / (CLAW_Y[0] - 2.5 - (-BODY_D / 2 + 2.0))) * sm(((BODY_D / 2 - 2.0) - Qy) / ((BODY_D / 2 - 2.0) - CLAW_Y[1] - 2.5))).astype(np.float32)
        self.front = sm((-self.n[:, 1] - .72) / .22).astype(np.float32)
        self.fz = (FLOOR + BODY_H) - Qz       # how far down his face
        # the calm round his eyes, so they read as eyes among the marks: not gone, only quieter
        calm = np.ones(n, np.float32) * 1e3
        for xc, zc in EYES:
            calm = np.minimum(calm, rrect(Qx, self.fz, xc, zc, U / 2, U, 1.5))
        self.keep = (1 - .78 * sm(1 - calm / 9.0) * self.front).astype(np.float32)
        # lumps from being worked by hand, thumb dents; a little lag from one lump of clay to the next
        self.rest = np.stack([self.mx + self.side * self.w_claw * CLAW_L / (BODY_W / (2 * A)), self.my, self.mz + self.r - self.w_leg * LEG_H], -1).astype(np.float32)
        final = self.Q + np.stack([self.side * self.w_claw * CLAW_L, 0 * Qx, (1 - self.w_leg) * LEG_H], -1)
        self.lump = (waves(final, 52.0, 11) * 1.3 + waves(final, 21.0, 12) * .26 + waves(final, 8.0, 13) * .07).astype(np.float32)
        # what were the walls and the underside get the same skin the top was born with, hairline cracks and grit, the block's own first state: looked at closely it has to be one clay all over
        z0 = np.load(os.path.join(OUT, "state_0.npz"))
        H0, r0 = z0["H"], int(z0["res"])
        H0 = H0[:H0.shape[0] // 4 * 4, :H0.shape[1] // 4 * 4].reshape(H0.shape[0] // 4, 4, H0.shape[1] // 4, 4).mean(axis=(1, 3))
        inset = 10.0
        wx0 = np.select([self.kind == 1, self.kind == 3, self.kind == 2, self.kind == 4, self.kind == 5], [self.mx, self.mx, self.mx, -A + inset + self.t * THICK, A - inset - self.t * THICK], 0)
        wy0 = np.select([self.kind == 1, self.kind == 3, self.kind == 2, self.kind == 4, self.kind == 5], [-B + inset + self.t * THICK, B - inset - self.t * THICK, self.my, self.my, self.my], 0)
        def fold(v, half):         # keeps inside the part that has been left alone, turning back at its edge
            span = 2 * (half - inset)
            q = (v + half - inset) % (2 * span)
            return np.minimum(q, 2 * span - q) - half + inset
        wx0, wy0 = fold(wx0, A), fold(wy0, B)
        k0 = r0 / 4
        sk = H0[np.clip(((B - wy0) * k0).astype(np.int32), 0, H0.shape[0] - 1), np.clip(((wx0 + A) * k0).astype(np.int32), 0, H0.shape[1] - 1)]
        on_wall = (self.kind != 0).astype(np.float32)
        away = np.select([self.kind == 2], [np.minimum(self.my + B, B - self.my) / 3.0], np.minimum(self.t, 1 - self.t) * THICK / 3.0)      # dying away where it meets the next face, so no seam
        away = np.minimum(away, np.where(self.kind >= 4, B - np.abs(self.my), A - np.abs(self.mx)) / 3.0)
        self.born = ((sk - float(np.median(H0))) * on_wall * sm(away)).astype(np.float32)
        del H0, z0, sk, wx0, wy0
        self.lag = waves(self.Q, 70.0, 14).astype(np.float32)
        self.patchy = (waves(self.Q, 34.0, 15) * .6 + waves(self.Q, 9.0, 16) * .3).astype(np.float32)
        self.final = final.astype(np.float32)
        bk = burnish.shape[0] / (2 * B)
        rows, cols = np.clip(((B - self.my) * bk).astype(np.int32), 0, burnish.shape[0] - 1), np.clip(((self.mx + A) * bk).astype(np.int32), 0, burnish.shape[1] - 1)
        on_top = self.kind == 0
        self.burnish = burnish[rows, cols] * on_top
        self.fresh = (fresh[rows, cols] if fresh is not None else np.zeros(n, np.float32)) * on_top
        self.is_side = (~on_top).astype(np.float32)
        self.faces(xs, ys, yb, yi, ti)
        print("skin:", n, "verts", len(self.quads), "quads", len(self.tris), "tris", flush=True)

    def faces(self, xs, ys, yb, yi, ti):
        nx, np_, ny, nz, nb = self.nx, self.np_, self.ny, self.nz, self.nb
        idx = np.arange(nx * np_, dtype=np.int32).reshape(nx, np_)
        nxt = np.roll(idx, -1, axis=1)
        tube = np.stack([idx[:-1], idx[1:], nxt[1:], nxt[:-1]], -1).reshape(-1, 4)
        quads, tris = [self.outward(tube, None)], []
        base = nx * np_
        ncols, nrows = len(yi), len(ti)
        for k, ring in enumerate((0, nx - 1)):
            start = base + k * ncols * nrows
            grid = start + np.arange(ncols * nrows, dtype=np.int32).reshape(nrows, ncols)
            # rows: front wall's vertex, the grid, back wall's vertex; last row the rings' bottom
            front = idx[ring, ny:ny + nz]                                  # t[1] to t[nz]
            back = idx[ring, ny + nz + nb - 1:][::-1]                       # t[1] to t[nz-1], reversed from the ring's order
            rows = [np.concatenate([[front[j]], grid[j], [back[j]]]) for j in range(nrows)]
            rows.append(idx[ring, ny + nz - 1:ny + nz + nb - 1])
            rows = np.array(rows, np.int32)
            q = np.stack([rows[:-1, :-1], rows[1:, :-1], rows[1:, 1:], rows[:-1, 1:]], -1).reshape(-1, 4)
            normal = (-1.0 if k == 0 else 1.0, 0.0, 0.0)
            quads.append(self.outward(q, normal))
            # the top of the end wall, zipped to the much finer edge of the top
            a, ya = idx[ring, :ny][::-1], ys[::-1]
            b, yb_ = rows[0], np.concatenate([[-B], yi, [B]])
            tr, i, j = [], 0, 0
            while i < ny - 1 or j < nb - 1:
                if j == nb - 1 or (i < ny - 1 and ya[i + 1] <= yb_[j + 1]):
                    tr.append((a[i], a[i + 1], b[j]))
                    i += 1
                else:
                    tr.append((a[i], b[j + 1], b[j]))
                    j += 1
            tris.append(self.outward(np.array(tr, np.int32), normal))
        self.quads, self.tris = np.concatenate(quads), np.concatenate(tris)

    def outward(self, f, normal):
        """Turns a patch's faces the right way out, judging by the block."""
        p = self.B
        a, b, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
        cr = np.cross(b - a, c - a)
        if normal is None:     # the tube: outwards is away from the block's middle
            mid = (a + b + c) / 3 - (0, 0, -THICK / 2)
            mid[:, 0] = 0
            sign = float((cr * mid).sum())
        else:
            sign = float((cr * np.array(normal)).sum())
        return f if sign > 0 else f[:, ::-1].copy()

    def volume(self, P):
        v = 0.0
        q = self.quads
        v += float((P[q[:, 0]] * np.cross(P[q[:, 1]], P[q[:, 2]])).sum()) + float((P[q[:, 0]] * np.cross(P[q[:, 2]], P[q[:, 3]])).sum())
        t = self.tris
        v += float((P[t[:, 0]] * np.cross(P[t[:, 1]], P[t[:, 2]])).sum())
        return v / 6


def frame(k, t, grey=False):
    """Everything about the clay at time t: where each vertex is, and what the shader needs."""
    n = len(k.mx)
    # what stage each lump of clay has got to
    ax = GATHER(t + .08 * k.lag + .05 * np.abs(k.mx) / A).astype(np.float32)
    ay = ROLL(t + .05 * k.lag - .08 * k.my / B).astype(np.float32)
    P = np.empty((n, 3), np.float32)
    P[:, 0] = k.B[:, 0] + (k.Q[:, 0] - k.B[:, 0]) * ax
    P[:, 1:] = k.B[:, 1:] + (k.Q[:, 1:] - k.B[:, 1:]) * ay[:, None]
    gone = np.clip(1 - ay, 0, 1)                                    # the bulge in the walls is drawn back in first
    P[:, :2] -= k.out * (gone - gone ** 3)[:, None]
    done = np.minimum(np.maximum(ax, 0), np.clip(ay, 0, 1))
    # the marks and the walls' own surface go round with it
    keep = 1 + (k.keep * KEEP_RELIEF - 1) * done
    P[:, 2] += k.r * (1 - done)
    P += k.n * ((k.rq * keep + k.patted + k.born + k.lump) * done)[:, None]
    # eyes, mouth: pressed into his face, along the normal, where they are on him
    Qx, fz = k.Q[:, 0], k.fz
    squint = float(SQUINT(t))
    e_open = float(EYE_OPEN(t)) * (1 - squint * .86) * (1 - .92 * float(BLINK(t)))
    press = np.zeros(n, np.float32)
    inner = np.zeros(n, np.float32)
    rim = np.zeros(n, np.float32)
    for (xc, zc), got in zip(EYES, (float(EYE_L(t)), float(EYE_R(t)))):
        if got <= 0:
            continue
        hh = max(U * e_open, 1.0)
        hw = U / 2 * (1 + .55 * squint)
        arc = (Qx - xc) ** 2 * (.02 * squint)            # squeezed shut, they are little arches
        d = rrect(Qx, fz + arc, xc, zc - .35 * U * squint, hw, hh, 1.6 + 1.2 * squint)
        press = np.maximum(press, sm(.5 - d / 1.5) * k.front * (6 + 20 * got) * (.35 + .65 * min(e_open, 1)))
        inner = np.maximum(inner, sm((1.0 - d) / 1.4) * k.front * got)
        rim = np.maximum(rim, np.exp(-(np.maximum(d, 0) / 2.6) ** 2) * (d > -1.5) * k.front * got * .9)
    m = float(MOUTH(t))
    if m > 0:
        hh = max(1.3 * U * m, .9)
        hw = U * (1.0 + 1.5 * m ** .6)
        d = rrect(Qx, fz, 0.0, 4.3 * U + hh, hw, hh, .92 * hh)
        d = d + (.5 * np.sin(Qx * .21) + .35 * np.sin(Qx * .57 + 1.0)) * k.front     # it split: its edge is not drawn with a ruler
        press = np.maximum(press, sm(.5 - d / 1.1) * k.front * (5 + 30 * m))
        inner = np.maximum(inner, sm((1.0 - d) / .8) * k.front * min(m * 5, 1))
        rim = np.maximum(rim, np.exp(-(np.maximum(d, 0) / 3.2) ** 2) * (d > -1.5) * k.front * (.8 + 1.0 * m))
    P -= k.n * (press - rim)[:, None]
    # the claws: when he raises them they reach out further, and ride up his sides. One goes up on its own to wave.
    both, hello = float(RAISE(t)), float(WAVE(t))
    right = k.side > 0
    firm = sm(k.w_claw * 3.0)                                # the claw itself stays a block: only its root gives
    far = 1 + .3 * max(both, 0) + .75 * max(hello, 0) * right
    ride = 1.8 * U * max(both, 0) + 3.2 * U * max(hello, 0) * right
    P[:, 0] += k.side * k.w_claw * CLAW_L * float(CLAW(t)) * far
    P[:, 2] += k.slide * ride
    # up off the board, right side a little ahead; the legs stay
    s = float(STRETCH(t))
    grown = LEG_H * .5 * max(s, 0) + LEG_H * .14 * min(s, 0)
    stand = LIFT(t + .11 * k.Q[:, 0] / (BODY_W / 2)).astype(np.float32) * LEG_H + grown
    P[:, 2] += stand * (1 - k.w_leg)
    zb = FLOOR + float(LIFT(t)) * LEG_H + grown
    # the claws go up, and shake; the waving one waves
    if abs(both) > 1e-4 or abs(hello) > 1e-4:
        shake = 7.0 * float(sm((t - 99.1) / .15) * sm((99.72 - t) / .15)) * math.sin(2 * math.pi * 6.0 * t)
        theta = np.radians((56 * both + shake) + (50 * hello + sway(t)) * right).astype(np.float32)
        px, pz = k.side * (BODY_W / 2 + (.9 + .3 * max(hello, 0) * right) * U), zb + (CLAW_Z[0] + CLAW_Z[1]) / 2 - FLOOR + ride
        rx, rz = P[:, 0] - px, P[:, 2] - pz
        c, sn = np.cos(theta * firm), np.sin(theta * firm) * k.side
        P[:, 0] = px + rx * c - rz * sn
        P[:, 2] = pz + rx * sn + rz * c
    # the stretch: up and narrower, then back
    h = P[:, 2] - zb
    above = np.maximum(h, 0)
    if abs(s) > 1e-4:
        grow = 1 + .2 * s
        thin = 1 / math.sqrt(grow)
        w = np.clip((P[:, 2] - FLOOR) / max(zb - FLOOR, 1.0), 0, 1)      # the legs lean in straight from the feet, which stay
        P[:, 0] *= 1 + (thin - 1) * w
        P[:, 1] *= 1 + (thin - 1) * w
        P[:, 2] = zb + above * grow + np.minimum(h, 0)
        phi = np.radians(14.0 * max(s, 0) * (float(RAISE(t)) if s > .2 else 1)) * np.clip(above / BODY_H, 0, 1.3) ** .9
        ry, rz = P[:, 1].copy(), above * grow
        P[:, 1] = ry * np.cos(phi) + rz * np.sin(phi)
        P[:, 2] = zb + rz * np.cos(phi) - ry * np.sin(phi) + np.minimum(h, 0)
    # his head a little on one side: as he yawns; away from the claw as he waves, and it rocks him
    tip = 3.0 * max(float(STRETCH(t)), 0) * max(both, 0) + hello * (-2.6 - .06 * sway(t))
    if abs(tip) > 1e-4:
        P[:, 0] += above * math.sin(math.radians(tip))
    # it stirs: lifts in the middle and lets go, before anything else
    st = float(STIR(t))
    if st != 0:
        hump = (1 - np.minimum((k.mx / A) ** 2 + (k.my / B) ** 2, 1)) * (.75 + .25 * k.lag)
        P[:, 2] = FLOOR + (P[:, 2] - FLOOR) * (1 + .15 * st * hump)
        P[:, :2] *= (1 - .012 * st * hump)[:, None]
    # and all of him turns on the spot
    psi = math.radians(float(YAW(t)))
    x, y = P[:, 0].copy(), P[:, 1].copy()
    P[:, 0], P[:, 1] = x * math.cos(psi) - y * math.sin(psi), x * math.sin(psi) + y * math.cos(psi)
    return P, press, inner, done


def warmth(k, t, grey):
    if grey:
        return np.zeros(len(k.mx), np.float32)
    T = float(COLOUR(t))
    high = np.clip((k.final[:, 2] - FLOOR) / (BODY_H + LEG_H), 0, 1)
    return sm((T - .25 * (1 - high) + .09 * k.patchy) / .8).astype(np.float32)


def put_mesh(k, P, attrs, mat):
    me = bpy.data.meshes.new("clay")
    me.vertices.add(len(P))
    me.vertices.foreach_set("co", P.ravel())
    q, tr = k.quads, k.tris
    me.loops.add(q.size + tr.size)
    me.loops.foreach_set("vertex_index", np.concatenate([q.ravel(), tr.ravel()]).astype(np.int32))
    me.polygons.add(len(q) + len(tr))
    me.polygons.foreach_set("loop_start", np.concatenate([np.arange(len(q), dtype=np.int32) * 4, q.size + np.arange(len(tr), dtype=np.int32) * 3]))
    me.update(calc_edges=True)
    me.shade_smooth()
    for name, a in attrs.items():
        if a.ndim == 2:
            x = me.attributes.new(name, "FLOAT_VECTOR", "POINT")
            x.data.foreach_set("vector", a.astype(np.float32).ravel())
        else:
            x = me.attributes.new(name, "FLOAT", "POINT")
            x.data.foreach_set("value", a.astype(np.float32))
    me.materials.append(mat)
    o = bpy.data.objects.new("clay", me)
    bpy.context.collection.objects.link(o)
    return o


def crumb_mesh(mat, pts, warm):
    """Real lumps. pts: x, y, z, size."""
    import bmesh
    r = np.random.default_rng(92)
    tpl = bpy.data.meshes.new("t")
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
    bm.to_mesh(tpl)
    bm.free()
    base = np.array([v.co[:] for v in tpl.vertices], np.float32)
    tris = np.array([p.vertices[:] for p in tpl.polygons], np.int32)
    co, faces = [], []
    for i, (x, y, z, s) in enumerate(pts):
        p = base * s * r.uniform(.65, 1.2, 3)
        p = p * (1 + .3 * np.sin(base @ r.normal(0, 3.2, 3) + r.uniform(0, 6)))[:, None] * (1 + .12 * np.sin(base @ r.normal(0, 8, 3)))[:, None]
        co.append(p + (x, y, z))
        faces.append(tris + i * len(base))
    co = np.vstack(co)
    me = bpy.data.meshes.new("crumbs")
    me.from_pydata(co.tolist(), [], np.vstack(faces).tolist())
    me.update()
    me.shade_smooth()
    me.materials.append(mat)
    for name, v in (("rest", co), ("warm", np.full(len(co), warm)), ("side", np.zeros(len(co))), ("wall", np.zeros(len(co))), ("burnish", np.zeros(len(co))), ("fresh", np.zeros(len(co))), ("inner", np.zeros(len(co)))):
        v = np.asarray(v, np.float32)
        x = me.attributes.new(name, "FLOAT_VECTOR" if v.ndim == 2 else "FLOAT", "POINT")
        x.data.foreach_set("vector" if v.ndim == 2 else "value", v.ravel())
    o = bpy.data.objects.new("crumbs", me)
    bpy.context.collection.objects.link(o)
    bpy.data.meshes.remove(tpl)
    return o


def debris(t, zb):
    """What has come off him by time t, on the board or on its way down. The same lumps at any t. All fall at the real rate."""
    r = np.random.default_rng(93)
    out = []
    # what fell where the block's edges used to stand, and what it left as it drew in
    for _ in range(90):
        side = r.integers(4)
        off = abs(r.normal(0, 14)) + 1.5
        x, y = (r.uniform(-A, A), (B + off) * (1 if side else -1)) if side < 2 else ((A + off) * (1 if side == 2 else -1), r.uniform(-B, B))
        s = float(r.choice((r.uniform(.25, .8), r.uniform(.8, 1.9)), p=(.8, .2)))
        out.append((x, y, FLOOR + s * .25, s))
    for _ in range(170):
        tland = r.uniform(94.55, 97.4)
        x, y = r.uniform(-A + 6, A - 6), r.uniform(-B + 3, B - 6)
        s = float(r.choice((r.uniform(.3, 1.0), r.uniform(1.0, 2.6)), p=(.78, .22)))
        if t >= tland:
            out.append((x, y, FLOOR + s * .3, s))
    # what he shakes off: from all over him at the height of the stretch, most from the claws
    for _ in range(210):
        t0 = r.uniform(98.25, 100.15)
        if r.random() < .4:
            x, y, z0 = r.choice((-1, 1)) * (BODY_W / 2 + r.uniform(3, CLAW_L)), r.uniform(CLAW_Y[0], CLAW_Y[1]), r.uniform(8, 30)
        else:
            x, y, z0 = r.uniform(-BODY_W / 2 - 4, BODY_W / 2 + 4), -BODY_D / 2 - r.uniform(5, 14), r.uniform(-12, 18)
        s = float(r.choice((r.uniform(.3, .9), r.uniform(.9, 1.7), r.uniform(1.7, 2.6)), p=(.62, .3, .08)))
        vx, vy = r.normal(0, 40), r.normal(-30, 30)
        dt = t - t0
        if dt < 0:
            continue
        fall = math.sqrt(max(2 * (z0 - FLOOR), 0) / 9800.0)
        if dt < fall:
            out.append((x + vx * dt, y + vy * dt, z0 - 4900.0 * dt * dt, s))
        else:
            out.append((x + vx * fall * 1.6 + r.normal(0, 7), y + vy * fall * 1.6 + r.normal(0, 16), FLOOR + s * .3, s))
    return out


def board_object(t):
    """The board as it was, and where the block stood, the thin grey film it has left on the wood, now it has got up."""
    x0, x1, y0, y1 = BOARD_RECT
    col, hr = np.load(os.path.join(OUT, "board_col.npy")), np.load(os.path.join(OUT, "board_h.npy"))
    ny, nx = col.shape[:2]
    yy, xx = np.mgrid[:ny, :nx]
    X, Y = (x0 + (xx + .5) / BOARD_PX).astype(np.float32), (y1 - (yy + .5) / BOARD_PX).astype(np.float32)
    p = np.stack([X, Y, 0 * X], -1).reshape(-1, 3)
    n1 = (waves(p, 26.0, 21) * .6 + waves(p, 7.0, 22) * .35 + waves(p, 2.2, 23) * .2).reshape(ny, nx)
    inside = np.hypot(np.maximum(np.abs(X) - (A - 11), 0), np.maximum(np.abs(Y) - (B - 11), 0)) - 11
    film = sm((-inside + n1 * 4.5) / 3.0) * sm((n1 + 1.3) / 1.2)
    lum = col.mean(-1)
    col = col * ((np.minimum(lum, .024) + np.maximum(lum - .024, 0) * .22) / lum)[..., None]         # in this higher sun the old dried slip would shine like sand
    new = col + (np.array((.034, .038, .04), np.float32) - col) * (film * .4)[..., None]
    off = sm((Y - 190) / 230)                                                                        # and the wood goes off into the dark
    new = new + (np.array(BENCH, np.float32) - new) * off[..., None]
    h = (hr[..., 0] + film * .22) * (1 - off)
    rough = (hr[..., 1] + film * .1) * (1 - off) + .95 * off
    N = Nodes("board")
    tn, b = N.t, N.b
    uv = tn.nodes.new("ShaderNodeTexCoord").outputs["UV"]

    def image(name, rgb):
        im = bpy.data.images.new(name, nx, ny, alpha=False, float_buffer=True)
        im.colorspace_settings.name = "Non-Color"
        a = np.ones((ny, nx, 4), np.float32)
        a[..., :3] = rgb
        im.pixels.foreach_set(a[::-1].ravel())
        im.update()
        node = tn.nodes.new("ShaderNodeTexImage")
        node.image = im
        tn.links.new(uv, node.inputs[0])
        return node
    i1 = image("board_col", new)
    i2 = image("board_h", np.dstack([(h + 2) / 3, rough, off]))
    tn.links.new(i1.outputs[0], b.inputs["Base Color"])
    sep = tn.nodes.new("ShaderNodeSeparateColor")
    tn.links.new(i2.outputs[0], sep.inputs[0])
    tn.links.new(sep.outputs[1], b.inputs["Roughness"])
    tn.links.new(N.bump(sep.outputs[0], 2.1, 1.0), b.inputs["Normal"])
    tn.links.new(N.range(sep.outputs[2], 0, 1, .35, .04), b.inputs["Specular IOR Level"])
    me = bpy.data.meshes.new("board")
    me.from_pydata([(x0, y0, FLOOR), (x1, y0, FLOOR), (x1, y1, FLOOR), (x0, y1, FLOOR)], [], [(0, 1, 2, 3)])
    u = me.uv_layers.new(name="UV")
    for i, v in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        u.data[i].uv = v
    me.update()
    o = bpy.data.objects.new("board", me)
    bpy.context.collection.objects.link(o)
    me.materials.append(N.m)
    # the bench goes on into the dark beyond the board, so there is no edge of the world behind him
    far = bpy.data.meshes.new("bench")
    far.from_pydata([(-9000, y1 - 1, FLOOR - .2), (9000, y1 - 1, FLOOR - .2), (9000, 9000, FLOOR - .2), (-9000, 9000, FLOOR - .2)], [], [(0, 1, 2, 3)])
    far.update()
    dark = bpy.data.materials.new("bench")
    dark.use_nodes = True
    dark.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*BENCH, 1)
    dark.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = .95
    dark.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = .04
    far.materials.append(dark)
    bpy.context.collection.objects.link(bpy.data.objects.new("bench", far))
    return o


def still(a):
    t = a["t"] - CLOCK_SHIFT
    sc = reset(a["samples"], 100)
    set_up(a, sc)
    mat = clay_material(crab=True)
    k = Skin(a["dense"], a["step"], a["step_wall"], a["step_under"])
    v0 = k.volume(k.B + np.stack([0 * k.r, 0 * k.r, k.r], -1))
    P, press, inner, done = frame(k, t, a["grey"])
    v1 = k.volume(P)
    c = (v0 / v1) ** (1 / 3)
    P[:, 0] *= c
    P[:, 1] *= c
    P[:, 2] = FLOOR + (P[:, 2] - FLOOR) * c
    print(f"volume: block {v0 / 1000:.1f} cm3, this pose {v1 / 1000:.1f} cm3, scaled {c:.4f}", flush=True)
    warm = warmth(k, t, a["grey"])
    put_mesh(k, P, dict(rest=k.rest, warm=warm, inner=inner, burnish=k.burnish, fresh=k.fresh, side=k.is_side * (1 - .85 * done), wall=k.is_side * done), mat)
    board_object(t)
    crumb_mesh(mat, debris(t, 0.0), float(np.median(warm)))
    li = light(t, a["grey"])
    dawn(sc, li["az"], li["el"], li["e"], li["colour"], gold=.22)
    # what is behind him: dark, as far as the camera goes, with no horizon; the sky still lights him as before
    tree = sc.world.node_tree
    bg = tree.nodes["Background"]
    far = tree.nodes.new("ShaderNodeBackground")
    far.inputs[0].default_value = (*BEYOND, 1)
    far.inputs[1].default_value = 1.0
    mix = tree.nodes.new("ShaderNodeMixShader")
    tree.links.new(tree.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"], mix.inputs[0])
    tree.links.new(bg.outputs[0], mix.inputs[1])
    tree.links.new(far.outputs[0], mix.inputs[2])
    tree.links.new(mix.outputs[0], tree.nodes["World Output"].inputs[0])
    s = shot(t)
    if a["elev"] is not None:
        s["elev"] = a["elev"]
    if a["look_at"]:
        x, y, z, s["scale"] = a["look_at"]
        s["target"] = (x, y, z)
    camera(sc, s["scale"], a["fstop"], elev=s["elev"], target=s["target"])
    path = os.path.join(OUT, "crab", a["name"] + ".png")
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
        draft = "--draft" in argv
        still(dict(name=argv[1], t=float(argv[2]), grey="--grey" in argv, samples=opt("--samples", 48 if draft else 256, 1, int), w=opt("--w", 960 if draft else 1920, 1, int), h=opt("--h", 540 if draft else 1080, 1, int),
                   fstop=opt("--fstop", 22.0), exp=opt("--exp", .5), denoise=True, dense=opt("--dense", None, 4), look_at=opt("--look-at", None, 4), elev=opt("--elev", None),
                   step=1 / 3 if draft else 1 / 7, step_wall=1 / 2.5 if draft else 1 / 5, step_under=1 / 1.5 if draft else 1 / 3))
