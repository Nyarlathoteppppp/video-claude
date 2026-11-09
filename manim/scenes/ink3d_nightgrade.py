"""The night scenes' ink: ink_grade.py's grade, plus what it has no pass for. Used by ink3d_qian.py and ink3d_channel.py.

- Window light. NAME_w.png holds the windows alone, cut off by the mountains. A painter leaves paper for a light, so a lit window takes the ink away
  from its own pixels, and from the mist around it, wider the further off it is. That is what makes the mist glow.
- The moon. Painted here, because the mist would swallow a disc in the 3D scene; the scene only keeps its reflection on the water.
"""
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

from ink_grade import INK_K, PAPER, fft_noise, fog, grade, load, sm


GRADE_SKY = .9


def soft(a):
    return 1 - np.exp(-a)


def night_sky(h, w, top, hor, moon, seed):
    """The sky's ink: a grey wash, deep overhead and thin low down, with long cloud across it. Near the moon the cloud is lit and paler than the sky; away from it, a darker wash."""
    k = w / 1920
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    t = np.exp(-np.maximum(.62 * h - yy, 0) / (.34 * h))
    c = gaussian_filter(fft_noise(h, w, 3.2, seed + 31), (9 * k, 70 * k))
    c2 = gaussian_filter(fft_noise(h, w, 2.4, seed + 33), (3 * k, 34 * k))
    cloud = sm((c / c.std() + .5 * c2 / c2.std() + .3) / 1.6)
    near = 0 * t
    if moon is not None:
        near = np.exp(-np.hypot(xx - moon[0] * k, (yy - moon[1] * k) * 1.3) / (moon[2] * k * 3.4))
    return np.maximum(top * (1 - t) + hor * t - .85 * near * (.6 + .4 * cloud) + .22 * cloud * (1 - near), .04)


def night(name, rho=.03, mist_h=20.0, mist_d=1.2, lift=.2, gamma=1.0, contrast=1.0, seed=0, out=None,
          lights=(1.0, 1.0, 1.0, 1.0), lit_mist=1.0, far_keep=.55, moon=None, moon_glow=.5, aerial=0.0, sky_top=2.4, sky_hor=1.2, use_win=1.0, mist_floor=.3, moon_veil=.8, windows=None, horizon=None, reflect=.9, drop=(), grade_sky=GRADE_SKY, veil=None):
    """lights: how much the windows take away at four widths (the window's halo, the slope around it, the village's mist, the whole valley's mist).
    lit_mist: how much of that the mist itself takes on. far_keep: how much of a far window survives the mist in front of it (1 is all of it).
    drop: rectangles (x0, y0, x1, y1, pixels at full size) where no window is to show. grade_sky: the wash the sky goes through grade as; paler for mountains paler than that, or they get a dark line round them.
    veil: (y0, y1, amount), a band of mist across the frame that takes the far land into the sky.
    moon: (x, y, radius) in pixels at full size. aerial: how fast distance thins the ink by itself, above the valley mist. sky_top, sky_hor: the sky's ink overhead and at the skyline."""
    rgba = load(name + ".png")
    z = load(name + "_z.png")
    meta = json.load(open(name + ".json"))
    h, w = rgba.shape[:2]
    win = None
    if use_win and os.path.exists(name + "_w.png"):
        win = load(name + "_w.png")[..., :3].mean(-1)
        for x0, y0, x1, y1 in drop:
            f = w / 1920
            win[int(y0 * f):int(y1 * f), int(x0 * f):int(x1 * f)] = 0
    if aerial:
        rgba = rgba.copy()
        rgba[..., :3] = 1 - (1 - rgba[..., :3]) * np.exp(-aerial * z[..., 0] * meta["far"])[..., None]
    md = np.full((h, w), float(mist_d), np.float32)
    g = None
    if win is not None:
        s, hp = z[..., 0] * meta["far"], z[..., 1] * meta["zmax"]
        bank = fft_noise(h, w, 3.2, seed + 1)
        T = fog(s, hp + 6 * bank * (mist_h / 14), meta["cam_z"], rho * np.clip(1 + .45 * bank, .25, 2.2), mist_h)
        win_t = win * T ** (1 - far_keep)
        k = w / 1920        # the widths are for the full frame; drafts are smaller
        g = [gaussian_filter(win_t, .9), gaussian_filter(win_t, 5.0 * k), gaussian_filter(win, 24.0 * k), gaussian_filter(win, 80.0 * k)]
        g = [g[0] * 1.0, g[1] / .12, g[2] / .03, g[3] / .015]     # share of the pixels nearby that are window; 1 where the windows are thick
        print("window share", float(win.mean()))
        md = np.maximum(md - lit_mist * (.5 * soft(g[2]) + .8 * soft(g[3])), mist_floor)
    # the sky goes through grade as a wash about as dark as the mountains, so their edge dries as an edge between two washes and not as a line drawn on bare paper; it is replaced below
    im = grade(rgba, z, True, rho, mist_h, meta["cam_z"], gamma, contrast, grade_sky, seed, meta["far"], meta["zmax"], lift, md).astype(np.float32)
    tone = PAPER[None, None, :] * (1 + .028 * fft_noise(h, w, 2.6, seed + 7) + .012 * fft_noise(h, w, .4, seed + 8))[..., None]
    sky_density = night_sky(h, w, sky_top, sky_hor, moon, seed)
    n1, n2 = fft_noise(h, w, 1.9, seed + 3), fft_noise(h, w, 1.9, seed + 4)
    yy_, xx_ = np.mgrid[:h, :w].astype(np.float32)
    sky_a = map_coordinates(1 - rgba[..., 3], [yy_ + 1.1 * n1, xx_ + 1.1 * n2], order=1, mode="nearest")     # the same creep the ink has had
    # a wash, not a stipple: the tooth of the paper shows only faintly through it
    sky_im = PAPER[None, None, :] * np.exp(-(sky_density * (1 + .022 * fft_noise(h, w, .8, seed + 9)))[..., None] * INK_K)
    im = im * (1 - sky_a[..., None]) + sky_im * sky_a[..., None]
    if veil is not None:
        k_ = w / 1920
        y0, y1 = veil[0] * k_, veil[1] * k_
        c_ = gaussian_filter(fft_noise(h, w, 2.6, seed + 11), (6 * k_, 60 * k_))
        band = sm((yy_ + 16 * k_ * c_ / c_.std() - y0) / (.5 * (y1 - y0))) * sm((y1 - yy_ - 16 * k_ * c_ / c_.std()) / (.5 * (y1 - y0))) * veil[2]
        pale = PAPER[None, None, :] * np.exp(-sky_hor * INK_K)[None, None, :]
        im = im * (1 - band[..., None]) + pale * band[..., None]
    if moon is not None:
        k_ = w / 1920
        im = _moon(im, tone, sky_a, (moon[0] * k_, moon[1] * k_, moon[2] * k_), moon_glow, seed, moon_veil)
    if win is not None:
        take = lights[0] * .5 * soft(g[0]) + lights[1] * .5 * soft(g[1]) + lights[2] * .3 * soft(g[2]) + lights[3] * .25 * soft(g[3])
        im = np.minimum(im * np.exp(take)[..., None], tone)
        # the windows themselves: bare paper, a little unsteady at the edge like everything else
        n1, n2 = fft_noise(h, w, 1.9, seed + 3), fft_noise(h, w, 1.9, seed + 4)
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        lit = map_coordinates(np.clip(win_t * 1.15, 0, 1), [yy + .5 * n1, xx + .5 * n2], order=1, mode="nearest")[..., None]
        im = im * (1 - lit) + tone * lit
    if windows is not None or (horizon is not None and moon is not None):
        im = water_layer(im, tone, z, sky_a, meta["zmax"], windows, win, moon, horizon, reflect, seed)
    out = out or name + "_ink.png"
    Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).save(out)
    y = im.mean(-1) / 255
    print(f"{os.path.basename(out)}: mean {y.mean():.3f} (>=.25)  dead {(y < .08).mean():.3f} (<=.20)  bright {(y >= .5).mean():.3f} (>=.15)", flush=True)
    return out


def water_layer(im, tone, z, sky, zmax, windows, win, moon, horizon, gain, seed, moon_gain=.32):
    """What the water shows of the lights: below each window, straight down at the same x, a broken vertical streak, every one in the same river; and the moon's long column.
    It is a layer of its own, from the table of windows, so a reflection can go out with its window."""
    h, w = im.shape[:2]
    k = w / 1920
    water = (z[..., 1] * zmax < .42) & (sky < .5)
    water = gaussian_filter(water.astype(np.float32), .8)
    R = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    hz = (horizon or .52 * h / k) * k
    n_lit = 0
    if windows is not None and win is not None:
        T = np.load(windows)
        # the houses we can read stand in front of whatever is behind them: from the eaves to the water, as wide as their walls
        panes = [(T["sx"][i], T["w_px"][i] * .92, T["sy"][i] - 1.9 * T["h_px"][i], T["sy"][i] + 2 * T["h_px"][i], T["d"][i]) for i in range(int(T["n_hero"]))]
        for i, (sx, sy, rx, ry, wp, hp, b, d) in enumerate(zip(T["sx"], T["sy"], T["rx"], T["ry"], T["w_px"], T["h_px"], T["bright"], T["d"])):
            if i >= len(panes) and any(abs(sx - a) < b and c < sy < e and d > f for a, b, c, e, f in panes):
                continue    # that house's window is what shows there, not this one
            ix, iy, q = int(sx * k), int(sy * k), max(int(wp * k * .3), 1)
            if not (q <= ix < w - q and q <= iy < h - q) or win[iy - q:iy + q + 1, ix - q:ix + q + 1].max() < .1:
                continue    # out of the frame, gone out, or behind a hill
            half = max((wp * .44 if i < int(T["n_hero"]) else min(wp * .44, 3.2)) * k, .6)     # the ones we can read keep their width; the rest are lines, or close to they would weave into a plaid
            top = (ry - hp * .5) * k
            length = (hp * 1.2 + 30 + .75 * max(ry - hz / k, 0)) * k
            x0, x1, y0, y1 = int(rx * k - half), int(rx * k + half) + 1, int(top), int(top + length)
            x0, x1, y0, y1 = max(x0, 0), min(x1, w), max(y0, 0), min(y1, h)
            if x0 >= x1 or y0 >= y1:
                continue
            R[y0:y1, x0:x1] += float(b) * (.3 + .7 * np.exp(-d / 420)) * np.linspace(1, .25, y1 - y0, dtype=np.float32)[:, None]
            n_lit += 1
        print("reflections", n_lit)
    R = gaussian_filter(R, (2.5 * k, .5 * k))
    # each row of the water has moved a little sideways, in long slow waves, the more the nearer it is
    n = gaussian_filter(fft_noise(h, w, 1.6, seed + 51), (3.2 * k, 34 * k))
    n = n / n.std()
    amp = (.8 + 6.5 * np.clip((yy - hz) / (h - hz), 0, 1) ** 1.3) * k
    R = map_coordinates(R, [yy, xx + amp * n], order=1, mode="constant")
    m = gaussian_filter(fft_noise(h, w, 1.2, seed + 52), (1.1 * k, 24 * k))
    m = m / m.std()
    broken = .6 + .55 * sm((m + .4) / 1.3)
    a = 1 - np.exp(-gain * R * broken)
    if moon is not None:
        col = np.exp(-((xx - moon[0] * k) / (46 * k)) ** 2) * np.clip((yy - hz) / (60 * k), 0, 1)
        a = np.maximum(a, moon_gain * col * broken)
    a = (a * water)[..., None]
    return im * (1 - a) + tone * a


def _moon(im, tone, sky, moon, glow, seed, veil=1.0):
    """A full moon behind the mountains, with thin cloud across it and the ring cold air gives it. sky: 1 where it can be seen. veil: how much of the disc comes through."""
    h, w = im.shape[:2]
    mx, my, r = moon
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    d = np.hypot(xx - mx, yy - my)
    cloud = np.clip(fft_noise(h, w, 3.4, seed + 21) * .5 + .5, 0, 1)
    ring = glow * (.75 * np.exp(-np.maximum(d - r, 0) / (r * .9)) + .95 * np.exp(-np.maximum(d - r, 0) / (r * 4.5))) * (.72 + .5 * cloud)
    im = np.minimum(im * np.exp(ring * sky)[..., None], tone)
    n = fft_noise(h, w, 2.4, seed + 22)
    disc = np.clip((r - d + 1.5 * n) / 1.5, 0, 1) * sky
    face = .93 + .05 * n
    disc = (disc * face * veil)[..., None]
    return im * (1 - disc) + tone * disc
