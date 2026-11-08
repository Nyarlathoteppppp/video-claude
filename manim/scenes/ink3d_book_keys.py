"""The book's camera and moon as plain numbers: the approved key of every shot, and how the two moving shots get from one key to the next. ink3d_book.py reads this inside Blender and ink_book.py
reads it outside, to know where the sheet is on the frame at any moment, so it needs nothing but numpy and the subtitles' clock. All lengths in cm, all times in seconds of film time.

    open    The film's first 5.6 s: cover, a very slow push, then half, spread and column, while the book opens (t is the clock of cover_u and leaf_u).
    close   十四: from the poem's camera out to closing while the book shuts (the opening clock run backwards, top leaf first, the cover last), and on in to end, the moon narrowing to the sheet.
"""
import math

import numpy as np

from ink_subs import chapters

FPS = 30
MOON = (.78, .30, -.46)
SHEET_C, SHEET_ROT, SW, SH = (-24.5, -9.0), math.radians(-3.0), 30.0, 23.0
GLEAM_AT, GLEAM_SIZE, GLEAM_POWER = (-24.0, -14.0), 30.0, 5.0     # where the sky lies on the sheet: on the last lines written
LENS, SENSOR = 50.0, 36.0


def lattice(cw, ch, offset, cols=3, rows=3, bar=.95):
    return dict(cols=cols, rows=rows, cw=cw, ch=ch, bar=bar, offset=offset)


def key(pos, look, focus, fstop, win, energy, dome, angle=.32, roll=0.0):
    return dict(pos=pos, look=look, focus=focus, fstop=fstop, win=win, energy=energy, dome=dome, angle=angle, roll=roll)


KEYS = dict(
    cover=key((-10.2, -39.5, 35.0), (2.1, -.9, .5), (-3.5, 3.0, 1.3), .10, lattice(25.0, 12.5, (-1.2, 2.2, 1.0)), 9.5, .30, roll=math.radians(9)),
    half=key((-36.0, -47.0, 26.0), (.8, .5, 4.2), (-1.0, -3.0, 5.0), .22, lattice(36.0, 18.0, (-3.0, 3.0, 1.0)), 8.0, .30),
    spread=key((6.2, -23.0, 62.0), (7.2, -1.2, .4), (7.2, 0.0, .5), 2.0, lattice(44.0, 22.0, (5.5, -.5, 1.0)), 7.5, .20),
    # the whole sheet, x < 1650, and the corner of the open book beside it at the top right
    poem=key((-21.4, -26.0, 67.5), (-19.9, -10.2, .2), (-24.5, -9.0, .2), 2.0, lattice(44.0, 22.0, (-24.5, -9.0, .2)), 6.4, .30, angle=.9),
    # wider than end, to hold the cover standing up, and the moon still on both
    closing=key((-11.4, -47.2, 101.0), (-9.4, -.5, .5), (-16.0, -6.0, .5), 2.0, lattice(44.0, 22.0, (-22.0, -8.0, .2)), 6.4, .30, angle=.9),
    # the moon has narrowed to the sheet and the seal. The book keeps only the room's light
    end=key((-15.15, -41.0, 77.0), (-13.45, -5.0, .5), (-16.0, -6.0, .5), 2.0, lattice(26.0, 14.5, (-25.5, -9.0, .2), cols=1, rows=1, bar=.1), 5.8, .42, angle=1.6),
)


def column_key(col_x, top_y, z_page):
    """Straight above the page, so that the column at col_x, whose first character starts at top_y, stands where the film's subtitles live. 88 pixels to the cm: 13 characters at .886 cm fill the frame's height."""
    px = 88.0
    xc = col_x - (1806 - 960) / px
    yc = top_y - (540 - 34) / px
    h = 1920 / px * LENS / SENSOR
    return key((xc, yc, z_page + h), (xc, yc, z_page), None, None, lattice(120.0, 120.0, (0.0, 0.0, 0.0), cols=1, rows=1, bar=.1), 7.5, .20)


# ---- going smoothly from key to key -----------------------------------------------------------------------------------------------------------------
class Spline:
    """A cubic through the points with a continuous second derivative, and none at its two ends. ys may have several columns."""

    def __init__(self, xs, ys):
        self.x = xs = np.asarray(xs, np.float64)
        self.y = ys = np.asarray(ys, np.float64)
        n, h = len(xs), np.diff(xs)
        a, b = np.zeros((n, n)), np.zeros(ys.shape)
        a[0, 0] = a[-1, -1] = 1
        for i in range(1, n - 1):
            a[i, i - 1:i + 2] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
            b[i] = 6 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
        self.m = np.linalg.solve(a, b)

    def __call__(self, x):
        x = float(np.clip(x, self.x[0], self.x[-1]))
        i = int(np.clip(np.searchsorted(self.x, x) - 1, 0, len(self.x) - 2))
        h = self.x[i + 1] - self.x[i]
        p, q = (self.x[i + 1] - x) / h, (x - self.x[i]) / h
        return p * self.y[i] + q * self.y[i + 1] + ((p ** 3 - p) * self.m[i] + (q ** 3 - q) * self.m[i + 1]) * h * h / 6


class Monotone:
    """A curve through the points that never swings past them: a light must not overshoot where it is going. Its tangents are the harmonic mean of the slopes either side, flat at a turning point."""

    def __init__(self, xs, ys, v0=None, v1=0.0):
        self.x, self.y = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
        d = np.diff(self.y) / np.diff(self.x)
        self.m = np.zeros(len(xs))
        for i in range(1, len(xs) - 1):
            self.m[i] = 0 if d[i - 1] * d[i] <= 0 else 2 * d[i - 1] * d[i] / (d[i - 1] + d[i])
        self.m[0], self.m[-1] = d[0] if v0 is None else v0, v1

    def __call__(self, x):
        x = float(np.clip(x, self.x[0], self.x[-1]))
        i = int(np.clip(np.searchsorted(self.x, x) - 1, 0, len(self.x) - 2))
        h = self.x[i + 1] - self.x[i]
        q = (x - self.x[i]) / h
        return (2 * q ** 3 - 3 * q * q + 1) * self.y[i] + (q ** 3 - 2 * q * q + q) * h * self.m[i] + (3 * q * q - 2 * q ** 3) * self.y[i + 1] + (q ** 3 - q * q) * h * self.m[i + 1]

    def slope(self, x):
        i = np.clip(np.searchsorted(self.x, x) - 1, 0, len(self.x) - 2)
        h = self.x[i + 1] - self.x[i]
        q = (x - self.x[i]) / h
        return (6 * q * q - 6 * q) * (self.y[i] - self.y[i + 1]) / h + (3 * q * q - 4 * q + 1) * self.m[i] + (3 * q * q - 2 * q) * self.m[i + 1]


def timing(ts, ss, soft, creeping=False, dt=1 / 240):
    """How far along (s) at each moment: through the (t, s) given, never going back, with its speed smoothed over `soft` seconds so that nothing starts or stops with a jolt. It stops soft/2 after
    the last t. It sets off soft/2 before the first, from rest, or if creeping is already under way at the first t, at the first slope. Returns the table (grid, s)."""
    ts, ss = np.asarray(ts, np.float64), np.asarray(ss, np.float64)
    curve = Monotone(ts, ss, v0=None if creeping else 0.0)
    grid = np.arange(ts[0] - soft / 2, ts[-1] + soft / 2 + dt, dt)
    inside = (grid >= ts[0]) & (grid <= ts[-1])
    v = np.zeros_like(grid)
    v[inside] = curve.slope(grid[inside])
    v[grid < ts[0]] = curve.m[0]
    k = int(soft / 2 / dt)
    hann = .5 + .5 * np.cos(np.linspace(-math.pi, math.pi, 2 * k + 1))
    hann /= hann.sum()
    smooth = np.convolve(np.concatenate([np.full(k, v[0]), v, np.zeros(k)]), hann, "valid")
    s = np.concatenate([[0.0], np.cumsum(.5 * (smooth[1:] + smooth[:-1]) * dt)])
    s0 = np.interp(ts[0], grid, s) if creeping else 0.0          # s[0] is the start unless it was moving before the film began
    s = ss[0] + (s - s0) * (ss[-1] - ss[0]) / (s[-1] - s0)
    return grid, s


def cruise(s, t0, t1, up, down, dt=1 / 240):
    """How far along at each moment, for a camera with one long way to go: it creeps to s[1] by t0, leaves that speed over `up` seconds, holds one even speed through every key after it, and comes to rest over
    `down` seconds, at t1. The keys are only places on the way; the speed does not change at them. Returns the table (grid, s)."""
    creep = s[1] / t0
    V = (s[-1] - s[1] - creep * up / 2) / (t1 - t0 - up / 2 - down / 2)
    grid = np.arange(0.0, t1 + dt, dt)
    rise = .5 - .5 * np.cos(np.pi * np.clip((grid - t0) / up, 0, 1))
    fall = .5 - .5 * np.cos(np.pi * np.clip((t1 - grid) / down, 0, 1))
    v = (creep + (V - creep) * rise) * fall
    table = np.concatenate([[0.0], np.cumsum(.5 * (v[1:] + v[:-1]) * dt)])
    return grid, table * s[-1] / table[-1]


def camera_up(pos, look, roll):
    """The camera's up, the way Blender's to_track_quat('-Z', 'Y') makes it, then rolled. Straight down it is +y."""
    f = np.subtract(look, pos, dtype=np.float64)
    f /= np.linalg.norm(f)
    y = np.array((0.0, 0.0, 1.0)) - f[2] * f
    n = np.linalg.norm(y)
    y = np.array((0.0, 1.0, 0.0)) if n < 1e-6 else y / n
    x = np.cross(f, y)
    return -x * math.sin(roll) + y * math.cos(roll)


class Path:
    """A camera and its light going through keys, at the times given: the speed is set by `timing` and the shape by chord length, so the one cannot make the other ring."""

    def __init__(self, keys, times, soft, creeping=False):
        pos = np.array([k["pos"] for k in keys], np.float64)
        look = np.array([k["look"] for k in keys], np.float64)
        up = np.array([camera_up(k["pos"], k["look"], k["roll"]) for k in keys])
        chord = np.linalg.norm(np.diff(pos, axis=0), axis=1) + .25 * np.linalg.norm(np.diff(look, axis=0), axis=1) + .01
        self.s = np.concatenate([[0.0], np.cumsum(chord)])
        self.pos, self.look, self.up = Spline(self.s, pos), Spline(self.s, look), Spline(self.s, up)
        if soft is not None:
            self.grid, self.table = timing(times, self.s, soft, creeping)
        else:       # an even speed has to be even on the screen, not on the spline's parameter: measure the way itself, each step by how far off what it looks at the camera is, so that the page does not rush up as it comes near
            fine = np.linspace(0.0, self.s[-1], 6000)
            way = np.array([self.pos(x) for x in fine]), np.array([self.look(x) for x in fine])
            step = np.linalg.norm(np.diff(way[0], axis=0), axis=1) + .25 * np.linalg.norm(np.diff(way[1], axis=0), axis=1)
            far = np.linalg.norm(way[0] - way[1], axis=1)
            walked = np.concatenate([[0.0], np.cumsum(step / (.5 * (far[1:] + far[:-1])))])
            self.grid, table = cruise(np.interp(self.s, fine, walked), *times)
            self.table = np.interp(table, walked, fine)

    def along(self, t):
        return float(np.interp(t, self.grid, self.table))

    def camera(self, t):
        """pos, the three axes of the camera (right, up, forward), in the world."""
        s = self.along(t)
        p, f, u = self.pos(s), self.look(s) - self.pos(s), self.up(s)
        f = f / np.linalg.norm(f)
        r = np.cross(f, u)
        r /= np.linalg.norm(r)
        return p, r, np.cross(r, f), f

    def project(self, t, world, wide=1920):
        """Where points in the world land on the frame, in pixels."""
        p, r, u, f = self.camera(t)
        v = np.asarray(world, np.float64) - p
        d = v @ f
        k = LENS / SENSOR * wide / d
        return np.stack([wide / 2 + (v @ r) * k, wide * 9 / 32 - (v @ u) * k], -1)


def between(values, s_keys, s, log=False):
    """A value of the keys at path position s."""
    if log:
        return math.exp(Monotone(s_keys, np.log(values), v0=0.0)(s))
    return Monotone(s_keys, values, v0=0.0)(s)


# ---- open ---------------------------------------------------------------------------------------------------------------------------------------------
OPEN_S = 5.6                                   # the camera has landed over the column; from here on the frame is still and only the ink moves
OPEN_PUSH = .016                               # how far of the way to the book it creeps before it sets off
OPEN_CRUISE = (.75, OPEN_S, .9, 2.0)           # it leaves its creep at .75 s, so that it is under way by the time the cover lifts; up to speed over .9 s, and it takes 2 s to come to rest on OPEN_S: the page has to be read before the ink takes it


def mix_key(a, b, q, **over):
    """A waypoint between two keys, for a camera that has to swing wide to keep the standing cover in frame."""
    lerp = lambda x, y: tuple(np.add(np.multiply(x, 1 - q), np.multiply(y, q)).tolist())
    win = dict(a["win"], cw=a["win"]["cw"] + (b["win"]["cw"] - a["win"]["cw"]) * q, ch=a["win"]["ch"] + (b["win"]["ch"] - a["win"]["ch"]) * q, offset=lerp(a["win"]["offset"], b["win"]["offset"]))
    k = dict(pos=lerp(a["pos"], b["pos"]), look=lerp(a["look"], b["look"]), focus=lerp(a["focus"], b["focus"]), fstop=math.exp(math.log(a["fstop"]) * (1 - q) + math.log(b["fstop"]) * q), win=win,
             energy=a["energy"] + (b["energy"] - a["energy"]) * q, dome=a["dome"] + (b["dome"] - a["dome"]) * q, angle=a["angle"], roll=a["roll"] * (1 - q) + b["roll"] * q)
    k.update(over)
    return k


def open_book(t):
    """The opening's clock at film time t: the cover lifts on time and the fan lands a little ahead of the clock's own 3.92, while the camera still has room for it."""
    return float(t if t < 1.2 else 1.2 + (t - 1.2) * 1.08)


def open_keys(z_page, col_x, top_y):
    c, h, sp, co = KEYS["cover"], KEYS["half"], dict(KEYS["spread"], pos=(6.2, -26.0, 70.0), look=(7.2, -.6, 2.0)), column_key(col_x, top_y, z_page)      # a little higher than the still: the last leaves are still standing
    c2 = dict(c, pos=tuple(np.add(c["pos"], np.subtract(c["look"], c["pos"]) * OPEN_PUSH)))
    wide = mix_key(c, h, .5, pos=(-22.3, -53.0, 34.8), look=(.5, .2, 4.7))      # it swings out and back, so that the book does not grow as the cover comes up, and the cover's top stays in the frame
    rise = mix_key(h, sp, .5, pos=(-24.5, -46.5, 49.0), look=(4.2, 3.2, 5.0))   # the fan has moved to the right and stands tall
    return [c, c2, wide, h, rise, sp, co]


def open_path(z_page, col_x, top_y):
    return Path(open_keys(z_page, col_x, top_y), OPEN_CRUISE, None)


def open_state(path, keys, t):
    """Everything about the frame at t that is not the book. A cell of the lattice grows from key to key, and with three cells the column's one open pane is the middle one."""
    s = path.along(t)
    z = lambda f: float(between([f(k) for k in keys], path.s, s))
    win = dict(cols=3, rows=3, cw=z(lambda k: k["win"]["cw"]), ch=z(lambda k: k["win"]["ch"]), bar=z(lambda k: k["win"]["bar"]), offset=tuple(z(lambda k, i=i: k["win"]["offset"][i]) for i in range(3)))
    focus = tuple(z(lambda k, i=i: (k["focus"] or k["look"])[i]) for i in range(3))
    fstop = float(between([k["fstop"] or 60.0 for k in keys], path.s, s, log=True))        # the column has no depth of field at all
    return dict(win=win, energy=z(lambda k: k["energy"]), dome=z(lambda k: k["dome"]), angle=z(lambda k: k["angle"]), focus=focus, fstop=fstop, gleam=0.0)


# ---- close ---------------------------------------------------------------------------------------------------------------------------------------------
CLOSE_T0 = {name: a for name, _, a, _ in chapters()}["十四"] + 23.9      # the first real frame, the poem's camera, still. Frames are numbered from here and everything below goes by it, so the subtitles' clock can move without a new render
CLOSE_T1 = CLOSE_T0 + 8.3                      # the last: camera and moon have stopped, the rest of the film holds on the end still
CLOSE_FRAMES = int(round((CLOSE_T1 - CLOSE_T0) * FPS)) + 1
BOOK_T0, BOOK_T1 = CLOSE_T0 + 1.2, CLOSE_T0 + 3.75                       # the top leaf lifts once the camera has made room for it; the cover comes down just before 只存在你那里
BOOK_CLOCK0, BOOK_CLOCK1 = 3.95, 1.2           # the opening's clock, backwards: everything has landed; the cover has not started
CAM_TIMES = (CLOSE_T0 + .4, CLOSE_T0 + 2.4, CLOSE_T0 + 6.9)              # poem; closing, wide enough before the leaves stand up, and it stays about that wide while the cover comes over; end
MOON_T0, MOON_T1 = CLOSE_T0 + 3.8, CLOSE_T1 - .1                         # the light narrows on 只存在你那里


def close_book(t):
    """The opening's clock at film time t."""
    q = np.clip((t - BOOK_T0) / (BOOK_T1 - BOOK_T0), 0, 1)
    return float(BOOK_CLOCK0 + (BOOK_CLOCK1 - BOOK_CLOCK0) * q)


def close_path():
    return Path([KEYS["poem"], KEYS["closing"], KEYS["end"]], CAM_TIMES, .8)


def window_rect(win):
    """The opening of a lattice, in the plane of the window, about its centre."""
    return np.array((-win["cols"] * win["cw"] / 2, win["cols"] * win["cw"] / 2, -win["rows"] * win["ch"] / 2, win["rows"] * win["ch"] / 2))


def window_axes():
    d = np.array(MOON, np.float64)
    d /= np.linalg.norm(d)
    u = np.cross(d, (0.0, 0.0, 1.0))
    u /= np.linalg.norm(u)
    return d, u, np.cross(u, d)


def moon_narrow(t):
    q = np.clip((t - MOON_T0) / (MOON_T1 - MOON_T0), 0, 1)
    return float(q * q * q * (q * (6 * q - 15) + 10))


def close_state(path, t):
    """From poem to closing the lattice shifts a little. Then its bars stay where they are and the wall closes in round them, from all sides, until only the end's pane of light is left."""
    a, b, c = KEYS["poem"], KEYS["closing"], KEYS["end"]
    p, n = sm_((t - CAM_TIMES[0]) / (CAM_TIMES[1] - CAM_TIMES[0])), moon_narrow(t)
    offset = np.array(a["win"]["offset"]) * (1 - p) + np.array(b["win"]["offset"]) * p
    _, u, v = window_axes()
    delta = np.subtract(c["win"]["offset"], offset)
    r0, r1 = window_rect(b["win"]), window_rect(c["win"]) + (delta @ u, delta @ u, delta @ v, delta @ v)
    focus = np.array(a["focus"]) * (1 - p) + np.array(b["focus"]) * p
    return dict(win=dict(lattice(44.0, 22.0, tuple(offset)), rect=tuple(r0 + (r1 - r0) * n)), energy=a["energy"] + (c["energy"] - a["energy"]) * n, dome=a["dome"] + (c["dome"] - a["dome"]) * n,
                angle=a["angle"] + (c["angle"] - a["angle"]) * n, focus=tuple(focus), fstop=2.0, gleam=GLEAM_POWER * (1 - sm_((t - CLOSE_T0 - .2) / 1.5)))


def sm_(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def shutter(t, shot):
    """How long a moving frame is exposed for: half of it, as a film camera would, while the leaves fly. Once they have landed only the camera moves, over a page meant to be read, and a quarter keeps the characters characters."""
    if shot == "open":
        return (.5 - .25 * sm_((t - 3.4) / .5)) / FPS
    return .5 / FPS


def needs_blur(t, shot):
    """Whether anything moves fast enough at t for one exposure to strobe."""
    if shot == "open":
        return .9 <= t <= 5.0
    return CLOSE_T0 + .25 <= t <= BOOK_T1 + .1     # the camera sets off at CLOSE_T0


def seq_time(shot, n):
    return n / FPS if shot == "open" else CLOSE_T0 + n / FPS
