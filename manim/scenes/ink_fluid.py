"""Ink falling through still water, simulated: stable fluids (Stam 1999) on a small grid with the ink heavier than the water, the ink itself carried on a large grid by back-and-forth advection so
that its threads stay thin. Written once to a file, one sheet per frame of film; ink_one.py lays it in its water.

The sheet is a tank: its top wall is the water's skin, and drops go in just under it.
"""
import math

import numpy as np
from scipy.ndimage import laplace, zoom

from ink_grade import fft_noise, sm

DENSITY = 8.0       # what a sheet's 255 means: density = DENSITY * (a / 255) ** 2
SKIN = 0.075        # where the water's skin is, as a share of the sheet


def _periodic(f, xs, ys):
    n = f.shape[0]
    x0, y0 = np.floor(xs).astype(np.int32), np.floor(ys).astype(np.int32)
    ax, ay = xs - x0, ys - y0
    x0, y0, x1, y1 = x0 % n, y0 % n, (x0 + 1) % n, (y0 + 1) % n
    return (f[y0, x0] * (1 - ax) + f[y0, x1] * ax) * (1 - ay) + (f[y1, x0] * (1 - ax) + f[y1, x1] * ax) * ay


def simulate(path, steps, drops, nv=384, nd=1536, seed=3, weight=4.0, jet=1.2, curl=1.2, viscosity=1.5e-6, spread=6.0, pull=None, slack=None, warm=0.0, start=None, skin=True, stir=1.0, force=None):
    """steps: how much water time goes by between one stored sheet and the next. drops: (when, x, radius), when counted from the first sheet and may be before it, down to -warm.
    pull: pull(T, X, Y) -> (u, v) on the ink alone, the current that draws it out. slack: slack(T), 0 to 1, how far the water has stopped being driven by the ink's weight. spread: pixels² a second.
    start: the ink already there, nd square, in density. skin: False for open water, with no surface in the sheet. stir: how strong the slow unevenness in the water is.
    force: force(T, X, Y) -> (fu, fv) on the water itself, on its small grid: a jet stirred in this way rolls up into the mushroom a real one makes, which a pull on the ink alone never does."""
    r = nd // nv
    gy, gx = np.mgrid[:nv, :nv].astype(np.float32)
    fy, fx = np.fft.fftfreq(nv)[:, None].astype(np.float32), np.fft.rfftfreq(nv)[None].astype(np.float32)
    k2 = fx * fx + fy * fy
    k2[0, 0] = 1
    Gy, Gx = np.mgrid[:nd, :nd].astype(np.float32)
    X, Y = Gx / nd, Gy / nd
    wall_v = sm((gy / nv - SKIN + 0.05) / 0.03) if skin else 1.0
    wall_d = sm((Y - SKIN + 0.05) / 0.008) if skin else 1.0
    margin = sm((0.5 - np.abs(gx / nv - 0.5)) / 0.08) * sm((1 - gy / nv) / 0.08)
    grain = np.clip(1 + 0.5 * fft_noise(nd, nd, 2.4, seed + 1), 0.25, 2.2)
    ink = np.zeros((nd, nd), np.float32) if start is None else start.astype(np.float32)
    # what is in the water already: a slow unevenness, without which a bloom is a mirror image of itself
    sy, sx = np.gradient(fft_noise(nv, nv, 3.0, seed), 1 / nv)
    u, v = (0.03 * stir * sy).astype(np.float32), (-0.03 * stir * sx).astype(np.float32)
    drive_x, drive_y = 0.012 * stir * sy, -0.012 * stir * sx
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(steps), nd, nd))
    T, waiting = -warm, sorted(drops)
    moving = []
    for i, step in enumerate([warm] + list(steps)):
        if i:
            out[i - 1] = np.clip(np.sqrt(np.maximum(ink, 0) / DENSITY), 0, 1) * 255
        if step <= 0:
            continue
        subs = max(1, math.ceil(step * 60))
        dt = step / subs
        damp = np.exp(-(2 * np.pi * nv) ** 2 * k2 * viscosity * dt)
        for _ in range(subs):
            while waiting and waiting[0][0] <= T:
                t0, x0, rad = waiting.pop(0)
                R = np.hypot(X - x0, (Y - SKIN) * 0.8)
                ink = np.maximum(ink, 7.0 * sm((rad - R) / (0.4 * rad)) * grain)
                moving.append((t0, np.exp(-((gx / nv - x0) ** 2 + (gy / nv - SKIN - rad) ** 2) / (1.6 * rad) ** 2)))
            k = slack(T) if slack else 0.0
            rho = ink.reshape(nv, r, nv, r).mean((1, 3))
            v += dt * weight * (1 - k) * rho
            u += dt * drive_x
            v += dt * drive_y
            if force is not None:
                fu, fv = force(T, gx / nv, gy / nv)
                u += dt * fu
                v += dt * fv
            for t0, shape in moving:
                if T - t0 < 0.12:                        # what a drop brings in with it
                    v += dt / 0.12 * jet * shape
            if curl:                                     # keeps the curls turning that a coarse grid would smear away
                w = (np.roll(v, -1, 1) - np.roll(v, 1, 1) - np.roll(u, -1, 0) + np.roll(u, 1, 0)) * nv / 2
                a = np.abs(w)
                ax = (np.roll(a, -1, 1) - np.roll(a, 1, 1)) * nv / 2
                ay = (np.roll(a, -1, 0) - np.roll(a, 1, 0)) * nv / 2
                m = np.hypot(ax, ay) + 1e-3
                c = curl * dt / nv
                u += c * ay / m * w
                v -= c * ax / m * w
            xs, ys = gx - dt * u * nv, gy - dt * v * nv
            u, v = _periodic(u, xs, ys), _periodic(v, xs, ys)
            U, V = np.fft.rfft2(u), np.fft.rfft2(v)
            div = (fx * U + fy * V) / k2
            U, V = (U - fx * div) * damp, (V - fy * div) * damp
            calm = math.exp(-1.6 * k * dt)
            u = np.fft.irfft2(U, s=(nv, nv)).astype(np.float32) * wall_v * margin * calm
            v = np.fft.irfft2(V, s=(nv, nv)).astype(np.float32) * wall_v * margin * calm
            uu, vv = zoom(u, r, order=1), zoom(v, r, order=1)
            if pull is not None:
                pu, pv = pull(T, X, Y)
                uu, vv = uu + pu, vv + pv
            ax_, ay_ = dt * uu * nd, dt * vv * nd
            fwd = _periodic(ink, Gx - ax_, Gy - ay_)
            back = _periodic(fwd, Gx + ax_, Gy + ay_)
            ink = np.clip(_periodic(ink + 0.5 * (ink - back), Gx - ax_, Gy - ay_), 0, 8.0) * wall_d
            ink += spread * dt * laplace(ink)
            T += dt
        if i % 30 == 1:
            print(f"{i}/{len(steps)}  water {T:.2f} s  ink max {ink.max():.2f} mean {ink.mean():.4f}", flush=True)
    out.flush()
    print(path)
