"""九 千窗: the tube. The wall of the cut is a sheet of windows and nothing else, and it rolls up. It is already a tube past the left of the first frame; the roll comes along it towards the camera, the two sides of the wall coming up and over and round under, until the top row meets the bottom row behind you. The camera backs off along the wall as it does and bends round towards it in one arc, on along it, close, with the three we know going by and the far side going to its point at the left of the frame. It never comes out to the axis. At last it turns to four windows on the near wall and comes to rest on them.

    Blender -b --factory-startup -P film_windows_tube.py -- way                                       the path, by the half second, and how it measures up
    Blender -b --factory-startup -P film_windows_tube.py -- seqplan                                   writes tube_low3/way.json: every frame of the shot
    Blender -b --factory-startup -P film_windows_tube.py -- t1_wall|t2_rising|t3_ring|t3b_pass|t4_mid|t5_near_window|K4a|K4a2|K4b|K4c|K4d|K4f|K4e [--pct=100] [--samples=128]
    Blender -b --factory-startup -P film_windows_tube.py -- f0123 --worker=1 [--pct=50] [--samples=24]      one frame of the shot, into that worker's plates, overwritten by its next

Plates as in film_windows_city.py, under the same names, and NAME_fog.npy, how much air is in each pixel. x along the axis, you look towards -x. How big a window comes at the edge of the frame on the axis: its height x 960 / RADIUS.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from mathutils import Matrix

from film_windows3d import Build, outline
from film_windows_city import ARGS, BAY, CUT_AT, CUT_FLOORS, CUT_COLS, CUT_LENS, CUT_PITCH, CUT_REVEAL, CUT_YAW, MARGIN, PLATES, SILL, STOREY, TEX, WIN_H, WIN_W, air, glow, lens_on, lit_window, mask_plate, only_the_windows, opt, plate, seen_width, set_up, shrunk, sky, surface
from film_windows_grid import GAIN, HALO, HALO_FAR, SMOG, SMOG_LEN, SMOG_SCATTER, SMOG_TINT, Lattice, Lens, halo_material
import json
import ink_subs

# ---------------------------------------------------------------- the shape

ROWS, ROW0 = 52, 6                      # floors of the wall that make the ring, and the first of them: the frame of the cut only sees floors 17 to 56
RADIUS = ROWS * STOREY / (2 * math.pi)
PLANE_Y, MID_Z = .12, CUT_FLOORS * STOREY / 2     # the wall of the cut, and the height of the axis
AXIS_Y = PLANE_Y - RADIUS
COL_FAR, COL_NEAR, TUBE_END = -3600, CUT_COLS + 60, -60000.0     # windows one by one as far as they can matter; and the same again, as light, for as far as it takes to come to a point
ROLL_LEN = 130.0                        # how long the roll is, from flat wall to closed ring
ROLL_AT = [40.0]                        # where it has got to: flat from here on. The frame of the cut starts past this


def _ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def on_surface(a, s, h=0.0):
    """x, y, z of the point a along the tube and s round it from the middle of the wall, h in from the surface: the sheet, bent round to a radius that is RADIUS where it has rolled and has no end to it where it hasn't."""
    a, s = np.asarray(a, float), np.asarray(s, float)
    q = s * _ease((ROLL_AT[0] - a) / ROLL_LEN) / RADIUS
    tiny = np.abs(q) < 1e-5
    safe = np.where(tiny, 1.0, q)
    sin_over, cos_over = np.where(tiny, 1.0, np.sin(q) / safe), np.where(tiny, q / 2, (1 - np.cos(q)) / safe)
    return a, PLANE_Y - s * cos_over - h * np.cos(q), MID_Z + s * sin_over - h * np.sin(q)


def T(a, h, z):
    """As a block's face has it: a along, h in, z up the unrolled wall."""
    s = z - MID_Z
    w = min(max((ROLL_AT[0] - a) / ROLL_LEN, 0.0), 1.0)
    q = s * w * w * (3 - 2 * w) / RADIUS
    if abs(q) < 1e-5:
        return a, PLANE_Y - s * q / 2 - h, MID_Z + s - h * q
    return a, PLANE_Y - s * (1 - math.cos(q)) / q - h * math.cos(q), MID_Z + s * math.sin(q) / q - h * math.sin(q)


def e_of(theta):
    return np.array((0.0, math.cos(theta), math.sin(theta)))


def u_of(theta):
    return np.array((0.0, -math.sin(theta), math.cos(theta)))


# ---------------------------------------------------------------- the way

FPS = 30
AXIS_DIR = np.array((-1.0, 0.0, 0.0))
H_CUT = PLANE_Y - CUT_AT[1]
FILM0 = ink_subs.at("同时在问", .4)      # the film's clock at the first frame
SHUTS, NEXT_LINE = ink_subs.at("窗一关"), ink_subs.at("昨夜")      # what the two windows are timed from
T_READ, T_RING = 1.0, 5.6               # still enough to read the three; the ring closed
T_MADE = 3.2                                    # by when a window has become a thing on a wall, and the air what it is going to be: well before the tube has closed
V_DRIFT = 1.2                           # metres a second, sliding back as the first frame has it; after that one speed, whatever gets you there
T_ARC, H_PASS, ARC_END = 2.6, 8.0, math.radians(176.0)  # the way you are going comes round from straight back to straight on, starting here, in one bend that brings you this close to the wall by the time the last move starts
T_BACK, BACK = 3.4, math.radians(7.0)   # how hard the look comes round to the far end, at its hardest
T_LOOK_UP, T_LOOK_DOWN = 9.9, 18.3      # the look comes through nothing without stopping there, and is turning to the wall as fast as it ever will by the first; keeps on to the second
T_END = SHUTS + 1.0 - FILM0             # and settles here, on K4a
T_PUSH, BRAKING = 8.0, 2.4              # from here it goes on to the window at the same speed, and comes to rest over this long: late enough that the wall is still going by the way it was while the look runs out, so that nothing comes back at the end
T_FLIP, T_SWAP = 3.4, .35               # the two windows, and the two under them, become what they will be, while others are still coming on round them: the dark ones going out first, then the right hand one coming on
K4_COL, K4_ROW = 73, 32                 # where the wall of the cut has a light window, a dark one, and under them next to nothing to lose
K4_ALONG, K4_FROM_WALL, K4_UP, K4_LOOK = 1.61, 5.5, .4, math.radians(102.0)     # the camera of K4, from the left window's lower left corner: along the wall, out from it, up it; and how far round it looks from along the axis
K4_Q = (WIN_W + MARGIN, WIN_H / 2)      # the point the last move is at: the gap between the two windows, half way up
T_FOUR = 13.5                           # from here on the way is its own again: it leaves off closing on the two, stays this far out, and comes to rest with the two and the two above them whole in the frame
K4_FOUR = (-.64, 8.3, -.36, math.radians(94.0))     # where it rests, from the middle of the four: along the wall, out from it, up it; and how far round it looks from along the axis. The foot of the frame stays clear for the subtitle
ROLL_END = 282.0 + ROLL_LEN + 40.0


def smooth(a, b, x):
    x = min(max((x - a) / (b - a), 0.0), 1.0)
    return x * x * (3 - 2 * x)


class View(Lens):
    """A camera with a say in which way is up."""

    def __init__(self, at, f, up, lens=CUT_LENS):
        self.at, self.lens, self.k = tuple(float(v) for v in at), lens, 1920 * lens / 36
        f = np.array(f, float) / np.linalg.norm(f)
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        self.f, self.r, self.u = tuple(f), tuple(r), tuple(np.cross(r, f))

    def see(self, p):
        v = [p[i] - self.at[i] for i in range(3)]
        d = sum(a * b for a, b in zip(v, self.f))
        if d < .5:
            return None
        return 960 + self.k * sum(a * b for a, b in zip(v, self.r)) / d, 540 - self.k * sum(a * b for a, b in zip(v, self.u)) / d, d

    def matrix(self):
        r, u, f = self.r, self.u, self.f
        return Matrix(((r[0], u[0], -f[0], self.at[0]), (r[1], u[1], -f[1], self.at[1]), (r[2], u[2], -f[2], self.at[2]), (0, 0, 0, 1)))


def from_yaw(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array((-math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p)))


def solve():
    """The whole way, frame by frame, from the cut."""
    n, m = int(round(T_PUSH * FPS)), int(round(T_END * FPS))

    def bend(end, v1):
        x, h, out = CUT_AT[0], H_CUT, []
        for i in range(n + 1):
            t = i / FPS
            out.append((x, h))
            v, psi = V_DRIFT + (v1 - V_DRIFT) * smooth(T_READ, T_READ + 1.4, t), ARC_END * smooth(T_ARC, end, t)
            x += v * math.cos(psi) / FPS
            h -= v * math.sin(psi) / FPS
        return out

    def bent(v1):
        lo, hi = T_ARC + .3, T_ARC + 8.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if bend(mid, v1)[-1][1] > H_PASS:
                lo = mid
            else:
                hi = mid
        return bend((lo + hi) / 2, v1)
    # which way you look: this is what the last move has to fit into, and it is the one thing it can't do slowly
    start = math.radians(90.0 - CUT_YAW)

    def turning(peak):
        look, out = start, []
        for i in range(m + 1):
            t = i / FPS
            out.append(look)
            look += (-BACK * smooth(T_READ, T_BACK, t) + (BACK + peak) * smooth(T_BACK, T_LOOK_UP, t)) * (1 - smooth(T_LOOK_DOWN, T_END, t)) / FPS
        return out
    lo, hi = 0.0, 2.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if turning(mid)[-1] < K4_LOOK:
            lo = mid
        else:
            hi = mid
    peak = (lo + hi) / 2
    looks = turning(peak)
    # the two windows
    a0 = K4_COL * BAY + MARGIN
    theta = (K4_ROW * STOREY + SILL + K4_Q[1] - MID_Z) / RADIUS
    e, u = e_of(theta), u_of(theta)
    q = np.array((a0 + K4_Q[0], AXIS_Y + RADIUS * math.cos(theta), MID_Z + RADIUS * math.sin(theta)))
    X1, Y1, Z1 = K4_ALONG - K4_Q[0], K4_FROM_WALL, K4_UP - K4_Q[1]

    def onward(path):
        """The rest of the way, from the end of the bend, and how far it is."""
        px, ph = path[-1]
        p0 = np.array((px, PLANE_Y - ph, MID_Z))
        P0, P1 = np.array((px - q[0], float((q - p0) @ e))), np.array((X1, Y1))
        go, stop, reach = np.array((path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])), np.array((-1.0, -.5)), np.linalg.norm(P1 - P0) * .4
        c1, c2 = P0 + go / np.linalg.norm(go) * reach, P1 - stop / np.linalg.norm(stop) * reach
        w = np.linspace(0.0, 1.0, 4001)[:, None]
        way = (1 - w) ** 3 * P0 + 3 * (1 - w) ** 2 * w * c1 + 3 * (1 - w) * w ** 2 * c2 + w ** 3 * P1
        return way, np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(way, axis=0), axis=1)))), float((p0 - q) @ u)
    lo, hi = 1.5, 3.5
    for _ in range(30):
        mid = (lo + hi) / 2
        path = bent(mid)
        if mid * (T_END - T_PUSH - BRAKING / 2) < onward(path)[1][-1]:
            lo = mid
        else:
            hi = mid
    v1 = (lo + hi) / 2
    path = bent(v1)
    way, along, Z0 = onward(path)
    frames = []

    def looking(t, look, tilt, pitch=0.0):
        s = e_of(tilt)
        return math.cos(pitch) * (math.cos(look) * AXIS_DIR + math.sin(look) * s) + math.sin(pitch) * np.array((0.0, 0.0, 1.0)), u_of(tilt)
    for i, (x, h) in enumerate(path):
        t = i / FPS
        k = smooth(T_READ, T_RING, t)
        f, up = looking(t, looks[i], 0.0, math.radians(CUT_PITCH) * (1 - k))
        frames.append((t, np.array((x, PLANE_Y - h, CUT_AT[2] + (MID_Z - CUT_AT[2]) * k)), f, up))
    gone = 0.0
    for i in range(n + 1, m + 1):
        t = i / FPS
        gone += v1 * (1 - smooth(T_END - BRAKING, T_END, t)) / FPS
        w = min(gone / along[-1], 1.0)
        X, Y = np.interp(gone, along, way[:, 0]), np.interp(gone, along, way[:, 1])
        tilt = theta * smooth(T_PUSH, T_END, t)
        at = q + np.array((X, 0.0, 0.0)) - e * Y + u * (Z0 + (Z1 - Z0) * smooth(0, 1, w))
        f, up = looking(t, looks[i], tilt)
        frames.append((t, at, f, up))
    # the four. Along the wall it goes on at the same speed and then brakes, as late as it can; out from the wall and up it, one curve that picks up what it was doing; the look goes on turning, and runs out when it gets there
    i0 = int(round(T_FOUR * FPS))
    theta4 = (K4_ROW * STOREY + SILL + (STOREY + WIN_H) / 2 - MID_Z) / RADIUS
    e4, u4 = e_of(theta4), u_of(theta4)
    q4 = np.array((a0 + K4_Q[0], AXIS_Y + RADIUS * math.cos(theta4), MID_Z + RADIUS * math.sin(theta4)))
    rest = q4 + np.array((K4_FOUR[0], 0.0, 0.0)) - e4 * K4_FOUR[1] + u4 * K4_FOUR[2]
    p0, ahead, back = frames[i0][1], frames[i0 + 1][1], frames[i0 - 1][1]
    v0, a00, span = (ahead - back) * FPS / 2, (ahead - 2 * p0 + back) * FPS * FPS, T_END - T_FOUR

    def run_out(start, by):
        return sum(by * (1 - smooth(start, T_END, i / FPS)) / FPS for i in range(i0 + 1, m + 1))
    lo, hi = T_FOUR, T_END
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p0[0] + run_out(mid, v0[0]) > rest[0] else (lo, mid)
    brake = (lo + hi) / 2
    lo, hi = T_FOUR, T_END
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if looks[i0] + run_out(mid, peak * .985) < K4_FOUR[3] else (lo, mid)
    ease = (lo + hi) / 2
    x, look = p0[0], looks[i0]
    for i in range(i0 + 1, m + 1):
        t = i / FPS
        x, look = x + v0[0] * (1 - smooth(brake, T_END, t)) / FPS, look + peak * (1 - .015 * smooth(T_FOUR, T_FOUR + 1.0, t)) * (1 - smooth(ease, T_END, t)) / FPS       # as the tilt adds to the turn
        s = (t - T_FOUR) / span
        d, v, a = rest - p0, v0 * span, a00 * span * span
        at = p0 + v * s + .5 * a * s ** 2 + (10 * d - 6 * v - 1.5 * a) * s ** 3 + (-15 * d + 8 * v + 1.5 * a) * s ** 4 + (6 * d - 3 * v - .5 * a) * s ** 5
        at[0] = x
        f, up = looking(t, look, theta * smooth(T_PUSH, T_END, t) + (theta4 - theta) * smooth(T_FOUR, T_END, t))
        frames[i] = (t, at, f, up)
    return frames, dict(peak=math.degrees(peak), speed=v1, t_push=T_PUSH, a0=a0, col=K4_COL, end=frames[-1][0], q=q, q4=q4, brake=brake, ease=ease)


def report(frames, info):
    print("radius", round(RADIUS, 2), "m,", ROWS, "floors round; a window at the edge of the frame on the axis", round(WIN_H * 960 / RADIUS, 1), "px high", flush=True)
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()}, flush=True)
    done, shuts, turned, grew, slowest, left, leaving = frames[-1], SHUTS - FILM0, 0.0, 0.0, 1e9, None, None
    a = min(frames, key=lambda p: abs(p[0] - shuts))
    print("settled", round(info["end"], 2), "s from the cut; when the line comes,", round(shuts, 2), "s: still", round(float(np.linalg.norm(a[1] - done[1])), 3), "m off where it stops, the window at", round(float(np.linalg.norm(done[1] - info["q"]) / np.linalg.norm(a[1] - info["q"])), 3), "of its size, turned", round(math.degrees(math.acos(min(1.0, float(a[2] @ done[2])))), 2), "deg short", flush=True)
    mark = dict(A=(77, 32), B=(76, 32), C=(77, 31), left=(73, 32), right=(74, 32))
    print("the four, from where the way changes: left to right and top to bottom of them all in the frame, how wide the top two come", flush=True)
    ROLL_AT[0] = ROLL_END
    for t, at, f, up in frames[int(T_FLIP * FPS)::15]:
        lens = View(at, f, up)
        seen = [lens.see(T(c * BAY + MARGIN + i * WIN_W, 0.0, (r * STOREY + SILL) + j * WIN_H)) for c in (K4_COL, K4_COL + 1) for r in (K4_ROW, K4_ROW + 1) for i in (0, 1) for j in (0, 1)]
        if all(seen):
            xs, ys = [a[0] for a in seen], [a[1] for a in seen]
            print(f"t {t:5.2f} x {min(xs):6.0f} {max(xs):6.0f} y {min(ys):6.0f} {max(ys):6.0f} top two {abs(seen[4][0] - seen[6][0]):4.0f} {abs(seen[12][0] - seen[14][0]):4.0f} wide, x {seen[4][0]:6.0f} {seen[14][0]:6.0f} y {seen[5][1]:6.0f} {seen[4][1]:6.0f}, depth {seen[0][2]:5.2f} {seen[12][2]:5.2f}", flush=True)
    ROLL_AT[0] = 40.0
    for n, (t, at, f, up) in enumerate(frames):
        lens = View(at, f, up)
        ahead = float(AXIS_DIR @ f)
        vanish = 960 + lens.k * float(AXIS_DIR @ np.array(lens.r)) / ahead if ahead > .05 else float("nan")
        if n:
            speed = float(np.linalg.norm(at - frames[n - 1][1])) * FPS
            turned = max(turned, math.degrees(math.acos(min(1.0, float(f @ frames[n - 1][2])))) * FPS)
            if t <= T_END - BRAKING:
                slowest = min(slowest, speed)
            if t > T_RING and left is not None and left >= 0 and not vanish >= 0:
                leaving = t
            left = vanish
            if t > T_PUSH:
                grew = max(grew, float(np.linalg.norm(frames[n - 1][1] - info["q"]) / np.linalg.norm(at - info["q"])) ** FPS)
        if n % 15 == 0 and n:
            b = frames[n - 15]
            where = []
            for name, (c, r) in mark.items():
                z = r * STOREY + SILL + WIN_H / 2
                th = (z - MID_Z) / RADIUS
                seen = lens.see((c * BAY + MARGIN + WIN_W / 2, AXIS_Y + RADIUS * math.cos(th), MID_Z + RADIUS * math.sin(th)))
                where.append(f"{name} {seen[0]:5.0f}" if seen else f"{name}     -")
            print(f"t {t:5.2f} x {at[0]:7.2f} from the wall {RADIUS - math.hypot(at[1] - AXIS_Y, at[2] - MID_Z):5.1f} speed {np.linalg.norm(at - b[1]) * FPS / 15:5.2f} m/s turning {math.degrees(math.acos(min(1.0, float(f @ b[2])))) * FPS / 15:5.1f} deg/s far point x {vanish:7.0f} ", *where, flush=True)
    print("speed along the way", round(info["speed"], 3), "m/s, slowest before it brakes", round(slowest, 2), "; turning at most", round(turned, 2), "deg/s; the window grows at most", round(grew, 3), "x/s; the far point leaves the frame at", leaving, "s", flush=True)
    plan = Tube(View(frames[0][1], frames[0][2], frames[0][3]))
    pics = plan.pictures_of_the_wall()
    for c in range(info["col"] - 5, info["col"] + 6):
        print("col", c, *[(f, plan.sizes[pics[c, f]]["kind"] if (c, f) in pics else "-") for f in (32, 31)], flush=True)


KEYS = dict(t1_wall=0.0, t2_rising=3.7, t3_ring=T_RING, t3b_pass=T_RING + 1.0, t4_mid=T_RING + 1.8)
CLICK, CLICKED = SHUTS + .5, SHUTS + 1.0
LAMP_OUT = ((SHUTS + 1.2, .88), (SHUTS + 2.0, .2), (SHUTS + 3.0, 0.0))     # the left window's level by the film's clock, from the click; a lamp is half gone long before it is gone
ENTER, BRIGHT, WORDS_FROM, WORDS_TO = NEXT_LINE + .15, NEXT_LINE + .4, NEXT_LINE + .4, NEXT_LINE + 1.0
with open(os.path.join(TEX, "manifest.json")) as f:
    STREAM = json.load(f).get("k4_stream")


def lamps(t):
    """What the two windows are doing by the film's clock: picture, level and for the left how far it has reddened, or None for only a frame."""
    if t < CLICK:
        left = ("k4_open", 1.0)
    elif t < LAMP_OUT[0][0]:
        left = ("k4_closing", 1 - .12 * smooth(CLICK, CLICKED, t))
    else:
        level = float(np.interp(t, *zip(*LAMP_OUT[:2])))
        if t > LAMP_OUT[1][0]:      # and the last of it slowly, for the eye makes most of that part
            level = LAMP_OUT[1][1] * max(1 - (t - LAMP_OUT[1][0]) / (LAMP_OUT[2][0] - LAMP_OUT[1][0]), 0.0) ** 2
        tint = min(1.0, (1 - level / LAMP_OUT[0][1]) / (1 - LAMP_OUT[1][1] / LAMP_OUT[0][1]))
        left = None if level < .0005 else ("k4_closing", round(level, 3), tuple(round(1 + (a - 1) * tint, 3) for a in EMBER_TINT))
    if t < ENTER:
        right = ("k4_typing", DIM)
    else:
        n = int((t - WORDS_FROM) / (WORDS_TO - WORDS_FROM) * 11) + 1 if t >= WORDS_FROM else 0
        right = ("k4_sent" if n >= 11 else f"k4_sent_{n:02d}", round(DIM + (1 - DIM) * smooth(ENTER, BRIGHT, t), 3))
    return left, right, *others(t)


def others(t):
    """The two above them, whatever those two are doing: the picture of the moment, the one on the left a stranger's question, the one on the right somebody's code."""
    now = min(max(int((t - FILM0 - STREAM["start"]) * STREAM["rate"]), 0), STREAM["ticks"] - 1)
    return f"k4_far_{now:03d}", f"k4_term_{now:03d}"


def focus_of(t, lens, info):
    """On the near wall: the three while you go past, then the two."""
    z = (K4_ROW - .5) * STOREY + SILL + WIN_H / 2
    depth = lambda p: sum((a - b) * c for a, b, c in zip(p, lens.at, lens.f))
    near = smooth(T_PUSH, T_END - 3.0, t)
    return max(depth(T(76.5 * BAY + MARGIN + WIN_W / 2, 0.0, z)), 4.0) ** (1 - near) * max(depth(tuple(info["q"] + (info["q4"] - info["q"]) * smooth(T_FOUR, T_END, t))), 4.0) ** near


def pose_of(key, frames, n=None):
    if n is not None:
        t, at, f, up = frames[min(n, len(frames) - 1)]
        if n == 0:
            return 0.0, View(CUT_AT, from_yaw(CUT_YAW, CUT_PITCH), (0, 0, 1))
        return n / FPS, View(at, f, up)
    if key in ("K4a", "K4a2", "K4b", "K4c", "K4d", "K4f", "K4e"):
        t, at, f, up = frames[-1]
    elif key == "t5_near_window":
        t, at, f, up = min(frames[int(T_PUSH * FPS):], key=lambda p: abs(np.linalg.norm(p[1] - frames[-1][1]) - 12.0))
    else:
        t, at, f, up = min(frames, key=lambda p: abs(p[0] - KEYS[key]))
    if key == "t1_wall":
        return 0.0, View(CUT_AT, from_yaw(CUT_YAW, CUT_PITCH), (0, 0, 1))
    return t, View(at, f, up)


# ---------------------------------------------------------------- what is on it

def quads(name, points, material, uv=None, state=None):
    """Very many four-cornered faces, from an array N x 4 x 3. state: three numbers a face, for the material to read."""
    n = len(points)
    me = bpy.data.meshes.new(name)
    me.vertices.add(4 * n)
    me.vertices.foreach_set("co", points.astype(np.float32).ravel())
    me.loops.add(4 * n)
    me.loops.foreach_set("vertex_index", np.arange(4 * n, dtype=np.int32))
    me.polygons.add(n)
    me.polygons.foreach_set("loop_start", np.arange(0, 4 * n, 4, dtype=np.int32))
    me.update(calc_edges=True)
    if uv is not None:
        me.uv_layers.new(name="UV").data.foreach_set("uv", uv.astype(np.float32).ravel())
    if state is not None:
        rgba = np.ones((n, 4, 4), np.float32)
        rgba[:, :, :3] = state[:, None, :]
        me.color_attributes.new("state", "FLOAT_COLOR", "POINT").data.foreach_set("color", rgba.ravel())
    me.materials.append(material)
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    return o


LIT, DARK, DARK_NEW = .96, .36, .5      # DARK_NEW: of those dark in the cut that have come on since, the wall of the cut having only three in ten of its own
AVERAGE = GAIN * (.97 * (1 - DARK) + .016 * DARK) * LIT     # a window too small to be anything: what the light, the dark and the unlit come to, between them. There are no black specks in it and no white ones
BY_ITSELF, TO_NOTHING = 24.0, 7.0       # how many pixels a window has to come to, its two sides taken together, to be seen for what it is, and how few before it is only that average
CLOSING, CLOSED = 4.0, 1.2              # how few pixels along the tube before it starts to spread into the wall around it, its light spread thinner to suit, and how few before it has met the next, all round: a mipmap, in the geometry. Nothing is left that can shimmer, or read as rays
DROPPED = .8                            # and from here on it is one surface of light
DUTY = WIN_W * WIN_H / (BAY * STOREY)


# what it is made of, by the time it has closed: a window is a thing on a wall, and the wall has a shine
PAGE, PAGE_WARM, SWING = .81, (1.0, .94, .836), .2      # how bright a white page is, about 218 as you see it, and a little warm; each window this much either side of that, for each one is somebody's own screen
SIT, TILT, SPIN = .04, math.radians(1.0), math.radians(.4)      # how far out of true each one hangs, once it has stopped having to line up with the cut: forward or back, tipped, turned
RINGS = ((.075, -.10), (.074, .070), (.070, .090), (.060, .105), (.048, .11), (.022, .11), (.014, .106), (.002, .09), (-.018, .05))   # a frame's section, wall to screen: how far out from the window's edge, how far off the wall
ACROSS = .2
RIM, RIM_FROM, RIM_TO = .22, 25.0, 80.0
GLASS_AT, FRAME_METAL, FRAME_ROUGH = .085, (.26, .225, .19), .22
WALL_DARK, WALL_ROUGH, WALL_SHEEN = (.0135, .0112, .0095), .13, .5      # very nearly black, half polished, the sheen is what shows
PANEL_A, PANEL_S, SEAM = 3 * BAY, 2 * STOREY, .008      # the wall's panels, joined where no window is
MAP_PX, MAP_BACK, MAP_FORE = 5.0, 240.0, 40.0           # what the windows come to on the wall is worked out on a map of it, this many points to the metre, this far down the tube from you and this far back
ON_THE_WALL = ((.3, .062), (.9, .011))                 # the light on the wall round a window, off the edge of it: how far, how much
IN_THE_WALL = ((.9, 0, .24), (1.8, 0, .06))       # and the streak of it in the polish, toward you, as a window would be seen in a floor if it could: how far behind the surface it seems, which of two softnesses, how much
SOFT = (.3, .7)
SHEEN = (.12, .5)                                       # how much of the polish shows that way, looking straight at it and looking along it
LIGHT_WARM = (1.0, .80, .58)
DARK_LIGHT = .05                                        # what a dark page throws, of what a white one does
FAR_LIGHT_WIDE = 11.0
FAR_LIGHT, FAR_LIGHT_AT, FAR_LIGHT_TINT = .8, 70.0, (.656, .55, .445)    # what the shine sees of the fog: you see it put on afterwards, it has to be somewhere for the glass and the metal and the polish to give back. The air on the way takes the blue out of it
GLINT, GLINT_AT, GLINT_WARM = .8, .35, (1.0, .86, .72)  # what the polish, the glass and the frames all see of a window: its light, standing this far out, as bright as this
BLUR_FAR, BLUR_CLOSE = 2.5, 8.0                                         # how many pixels across a point at the far end of the tube is, when the near wall is in focus


def luck(a0, z0, n=5):
    """Numbers from 0 to 1 that belong to the window at this place."""
    h, out = (int(round(a0 * 100)) * 73856093 ^ int(round(z0 * 100)) * 19349663) & 0xffffffff, []
    for _ in range(n):
        h = (h * 2654435761 + 12345) & 0xffffffff
        h ^= h >> 15
        out.append(h / 2 ** 32)
    return out


def sheet(name, p, material, uv):
    """A grid of points N x M x 3 as one surface, its faces sharing their corners so that a shine runs across it unbroken."""
    n, m = p.shape[:2]
    i = np.arange(n * m, dtype=np.int32).reshape(n, m)
    faces = np.stack((i[:-1, :-1], i[1:, :-1], i[1:, 1:], i[:-1, 1:]), axis=-1).reshape(-1, 4)
    me = bpy.data.meshes.new(name)
    me.vertices.add(n * m)
    me.vertices.foreach_set("co", p.astype(np.float32).ravel())
    me.loops.add(faces.size)
    me.loops.foreach_set("vertex_index", faces.ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", np.arange(0, faces.size, 4, dtype=np.int32))
    me.update(calc_edges=True)
    me.uv_layers.new(name="UV").data.foreach_set("uv", uv.reshape(-1, 2)[faces.ravel()].astype(np.float32).ravel())
    me.shade_smooth()
    me.materials.append(material)
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    return o


def blurred(x, sigma):
    h, w = x.shape
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None, :]
    return np.fft.irfft2(np.fft.rfft2(x) * np.exp(-2 * (np.pi * sigma) ** 2 * (fx * fx + fy * fy)), s=x.shape)


def polished_wall(made, lit_up, shine, on_it, span):
    """The wall as it was in the cut, going over to near black half polished stone: it holds very little light of its own, and gives back what it sees."""
    m = surface("night_wall", tuple(a + (b - a) * made for a, b in zip((.085, .070, .055), WALL_DARK)), .85, 0, shine + (WALL_SHEEN - shine) * made)
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Emission Color"].default_value = (*LIGHT_WARM, 1)

    def math_(op, a, b=None, clamp=False):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation, n.use_clamp = op, clamp
        for slot, v in zip(n.inputs, (a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                slot.default_value = v
            else:
                nt.links.new(v, slot)
        return n.outputs[0]
    uv = nt.nodes.new("ShaderNodeUVMap").outputs[0]
    split = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv, split.inputs[0])

    def from_seam(v, every, off=0.0):
        f = math_("FRACT", math_("DIVIDE", math_("SUBTRACT", v, off), every))
        return math_("MULTIPLY", math_("MINIMUM", f, math_("SUBTRACT", 1.0, f)), every)
    seam = math_("SUBTRACT", 1.0, math_("DIVIDE", math_("MINIMUM", from_seam(split.outputs[0], PANEL_A), from_seam(split.outputs[1], PANEL_S, .05)), SEAM, True))
    seam = math_("MAXIMUM", seam, 0.0)
    broad, fine = nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeTexNoise")
    for n, scale, detail in ((broad, .16, 6), (fine, 11.0, 4)):
        nt.links.new(uv, n.inputs["Vector"])
        n.inputs["Scale"].default_value, n.inputs["Detail"].default_value = scale, detail
    wander = math_("MULTIPLY", math_("SUBTRACT", broad.outputs[0], .5), 2.0)
    rough = math_("ADD", math_("ADD", WALL_ROUGH, math_("MULTIPLY", wander, .05)), math_("ADD", math_("MULTIPLY", math_("SUBTRACT", fine.outputs[0], .5), .04), math_("MULTIPLY", seam, .08)))
    nt.links.new(math_("ADD", .85 * (1 - made), math_("MULTIPLY", rough, made)), bsdf.inputs["Roughness"])
    tone = nt.nodes.new("ShaderNodeVectorMath")
    tone.operation = "SCALE"
    tone.inputs[0].default_value = bsdf.inputs["Base Color"].default_value[:3]
    nt.links.new(math_("MULTIPLY", math_("ADD", 1.0, math_("MULTIPLY", wander, .22 * made)), math_("SUBTRACT", 1.0, math_("MULTIPLY", seam, .2 * made))), tone.inputs["Scale"])
    nt.links.new(tone.outputs[0], bsdf.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = .01
    bump.inputs["Strength"].default_value = .35 * made
    nt.links.new(math_("SUBTRACT", math_("MULTIPLY", fine.outputs[0], .5), math_("MULTIPLY", seam, .3)), bump.inputs["Height"])
    nt.links.new(bump.outputs[0], bsdf.inputs["Normal"])
    where = nt.nodes.new("ShaderNodeCombineXYZ")
    (a_lo, a_hi), (s_lo, s_hi) = span
    nt.links.new(math_("DIVIDE", math_("SUBTRACT", split.outputs[0], a_lo), a_hi - a_lo), where.inputs[0])
    nt.links.new(math_("DIVIDE", math_("SUBTRACT", split.outputs[1], s_lo), s_hi - s_lo), where.inputs[1])
    light = nt.nodes.new("ShaderNodeTexImage")
    light.image, light.extension, light.interpolation = on_it, "CLIP", "Cubic"
    nt.links.new(where.outputs[0], light.inputs[0])
    rgb = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(light.outputs[0], rgb.inputs[0])
    nt.links.new(math_("ADD", lit_up, math_("MULTIPLY", rgb.outputs[0], made)), bsdf.inputs["Emission Strength"])
    # the streak: the surface lets through, by as much as it shines, what is laid out behind it
    side, clear, mix = nt.nodes.new("ShaderNodeLayerWeight"), nt.nodes.new("ShaderNodeBsdfTransparent"), nt.nodes.new("ShaderNodeMixShader")
    side.inputs[0].default_value = .5
    along = math_("MULTIPLY", side.outputs["Facing"], side.outputs["Facing"])
    amount = math_("MULTIPLY", math_("ADD", SHEEN[0], math_("MULTIPLY", along, SHEEN[1] - SHEEN[0])), math_("ADD", 1.0, math_("MULTIPLY", wander, .5)))
    nt.links.new(math_("MINIMUM", math_("MULTIPLY", amount, made), 1.0), mix.inputs[0])
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    nt.links.new(bsdf.outputs[0], mix.inputs[1])
    nt.links.new(clear.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


def behind_the_wall(name, on_it, channel, by):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    light, rgb, e, clear, add, out, by_ = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeSeparateXYZ"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeBsdfTransparent"), nt.nodes.new("ShaderNodeAddShader"), nt.nodes.new("ShaderNodeOutputMaterial"), nt.nodes.new("ShaderNodeMath")
    light.image, light.extension, light.interpolation = on_it, "CLIP", "Cubic"
    nt.links.new(nt.nodes.new("ShaderNodeUVMap").outputs[0], light.inputs[0])
    nt.links.new(light.outputs[0], rgb.inputs[0])
    by_.operation, by_.inputs[1].default_value = "MULTIPLY", by
    nt.links.new(rgb.outputs[channel], by_.inputs[0])
    e.inputs[0].default_value = (*LIGHT_WARM, 1)
    nt.links.new(by_.outputs[0], e.inputs[1])
    nt.links.new(e.outputs[0], add.inputs[0])
    nt.links.new(clear.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    return m


def soft_edge():
    """A rectangle with no edge to it: nothing that shines should show a line where a window's light stops."""
    if "soft_edge" in bpy.data.images:
        return bpy.data.images["soft_edge"]
    n = 64
    x = np.linspace(0, 1, n, dtype=np.float32)
    e = np.minimum(x, 1 - x) / .32
    e = np.clip(e, 0, 1)
    e = e * e * (3 - 2 * e)
    im = bpy.data.images.new("soft_edge", n, n, float_buffer=True, alpha=False)
    im.colorspace_settings.name = "Non-Color"
    px = np.ones((n, n, 4), np.float32)
    px[..., :3] = np.outer(e, e)[..., None]
    im.pixels.foreach_set(px.ravel())
    im.update()
    return im


def far_light(a, made):
    n, ring = 48, RADIUS + 6.0
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rim = np.stack((np.full(n, a - FAR_LIGHT_AT), AXIS_Y + ring * np.cos(th), MID_Z + ring * np.sin(th)), -1)
    mid = np.array((a - FAR_LIGHT_AT, AXIS_Y, MID_Z))
    m = halo_material(FAR_LIGHT_TINT, FAR_LIGHT * made)
    nt = m.node_tree
    where, off, far, sq, fall, by = nt.nodes.new("ShaderNodeNewGeometry"), nt.nodes.new("ShaderNodeVectorMath"), nt.nodes.new("ShaderNodeVectorMath"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeMath")
    off.operation, far.operation, sq.operation, fall.operation, by.operation = "SUBTRACT", "LENGTH", "MULTIPLY", "EXPONENT", "MULTIPLY"
    off.inputs[1].default_value = tuple(mid)
    nt.links.new(where.outputs["Position"], off.inputs[0])
    nt.links.new(off.outputs[0], far.inputs[0])
    nt.links.new(far.outputs["Value"], sq.inputs[0])
    nt.links.new(far.outputs["Value"], sq.inputs[1])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation, mul.inputs[1].default_value = "MULTIPLY", -1 / FAR_LIGHT_WIDE ** 2
    nt.links.new(sq.outputs[0], mul.inputs[0])
    nt.links.new(mul.outputs[0], fall.inputs[0])
    nt.links.new(fall.outputs[0], by.inputs[0])
    by.inputs[1].default_value = FAR_LIGHT * made
    nt.links.new(by.outputs[0], next(n for n in nt.nodes if n.type == "EMISSION").inputs[1])
    o = quads("halo_far_light", np.stack((rim, np.roll(rim, -1, axis=0), np.full((n, 3), mid), np.full((n, 3), mid)), axis=1), m)
    o.visible_camera = o.visible_diffuse = o.visible_transmission = o.visible_volume_scatter = o.visible_shadow = False
    o["halo"] = 1


def frame_metal(made):
    """Dark metal, that nothing near lights the edges of for long; far off it is only those edges, so they are given what they would catch."""
    if "frame_metal" in bpy.data.materials:
        return bpy.data.materials["frame_metal"]
    m = surface("frame_metal", FRAME_METAL, FRAME_ROUGH, 1.0, .5)
    nt = m.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    how_far, by = nt.nodes.new("ShaderNodeCameraData"), nt.nodes.new("ShaderNodeMapRange")
    by.interpolation_type = "SMOOTHSTEP"
    by.inputs["From Min"].default_value, by.inputs["From Max"].default_value = RIM_FROM, RIM_TO
    by.inputs["To Min"].default_value, by.inputs["To Max"].default_value = 0.0, RIM * made
    nt.links.new(how_far.outputs["View Distance"], by.inputs["Value"])
    b.inputs["Emission Color"].default_value = (1.0, .82, .6, 1)
    nt.links.new(by.outputs[0], b.inputs["Emission Strength"])
    return m


def pane_glass(made):
    """In front of a screen, what is in front of any screen: you see through it, and the flatter you look along it the more it is a mirror."""
    m = bpy.data.materials.new("pane_glass")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    fresnel, clear, shine, mix, out = nt.nodes.new("ShaderNodeFresnel"), nt.nodes.new("ShaderNodeBsdfTransparent"), nt.nodes.new("ShaderNodeBsdfAnisotropic"), nt.nodes.new("ShaderNodeMixShader"), nt.nodes.new("ShaderNodeOutputMaterial")
    fresnel.inputs["IOR"].default_value = 1.52
    shine.name = "shine"
    shine.inputs["Roughness"].default_value = .03
    by = nt.nodes.new("ShaderNodeMath")
    by.operation, by.inputs[1].default_value = "MULTIPLY", made
    nt.links.new(fresnel.outputs[0], by.inputs[0])
    nt.links.new(by.outputs[0], mix.inputs[0])
    nt.links.new(clear.outputs[0], mix.inputs[1])
    nt.links.new(shine.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


def page_material(name, tex, gain, level, warm=None, under=0.0):
    """A screen at a level of its own, and tinted if it is told to, down to what the glass of a dark one holds."""
    m = lit_window(name, os.path.join(TEX, tex + ".png"), gain, 0.0, "_m" not in tex)
    nt = m.node_tree
    shade = nt.nodes.new("ShaderNodeVectorMath")
    shade.operation = "MULTIPLY_ADD"
    nt.links.new(next(n for n in nt.nodes if n.type == "TEX_IMAGE").outputs[0], shade.inputs[0])
    shade.inputs[1].default_value = tuple(level * a for a in (warm or (1, 1, 1)))
    shade.inputs[2].default_value = tuple(under * a / gain for a in (1.0, .97, .93))
    nt.links.new(shade.outputs[0], next(n for n in nt.nodes if n.type == "EMISSION").inputs[0])
    return m


def see_through(m, by):
    """The same light, only so much of it there: what is behind it shows through the rest."""
    if by < 1:
        nt = m.node_tree
        e, o = next(n for n in nt.nodes if n.type == "EMISSION"), next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        clear, mix = nt.nodes.new("ShaderNodeBsdfTransparent"), nt.nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = by
        nt.links.new(clear.outputs[0], mix.inputs[1])
        nt.links.new(e.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], o.inputs[0])
    return m


def atlas_material():
    """A window out of the atlas, going over to the average as it gets small."""
    m = lit_window("atlas", os.path.join(TEX, "atlas.png"), GAIN, 0.0, False)
    m.cycles.emission_sampling = "NONE"
    nt = m.node_tree
    tile, e = next(n for n in nt.nodes if n.type == "TEX_IMAGE"), next(n for n in nt.nodes if n.type == "EMISSION")
    state, parts, spread = nt.nodes.new("ShaderNodeAttribute"), nt.nodes.new("ShaderNodeSeparateColor"), nt.nodes.new("ShaderNodeVectorMath")
    state.attribute_name = "state"
    nt.links.new(state.outputs["Color"], parts.inputs[0])
    there, small = nt.nodes.new("ShaderNodeMix"), nt.nodes.new("ShaderNodeMix")
    for mix in (there, small):
        mix.data_type = "RGBA"
    there.inputs[6].default_value = (0, 0, 0, 1)
    nt.links.new(tile.outputs[0], there.inputs[7])
    nt.links.new(parts.outputs[1], there.inputs[0])
    small.inputs[7].default_value = (AVERAGE / GAIN, AVERAGE / GAIN, AVERAGE / GAIN, 1)
    nt.links.new(there.outputs[2], small.inputs[6])
    nt.links.new(parts.outputs[0], small.inputs[0])
    spread.operation = "SCALE"
    nt.links.new(small.outputs[2], spread.inputs[0])
    nt.links.new(parts.outputs[2], spread.inputs["Scale"])
    nt.links.new(spread.outputs[0], e.inputs[0])
    return m


class Tube(Lattice):
    """The wall of the cut as a Lattice has it, and everything else on the same surface."""

    def __init__(self, lens, made=0.0):
        super().__init__(lens, None, [lens])
        self.made, self.metal, self.glass, self.screens = made, Build(), Build(), {}
        self.gain = GAIN + (PAGE - GAIN) * made
        self.warm = tuple(1 + (a - 1) * made for a in PAGE_WARM)
        self.halo_by = 1 - made
        self.glints = dict(light=Build(), dark=Build())
        self.lights, self.more_lights = None, None
        a = lens.at[0]
        a = math.floor(a * MAP_PX) / MAP_PX     # on the same points from frame to frame, or the light on the wall shimmers
        self.span = (a - MAP_BACK, a + MAP_FORE), (-ROWS * STOREY / 2, ROWS * STOREY / 2)
        with open(os.path.join(TEX, "manifest.json")) as f:
            info = json.load(f)
        self.tube, self.sizes = info["tube"], info["windows"]
        self.taken, self.stats = set(), {}
        self.shut_glass, self.shut_rim = Build(), Build()
        self.under_glass, self.under_rim = Build(), Build()

    def at(self, c, f):
        return (c - COL_FAR) * ROWS + (f - ROW0)

    def wall_of_the_tube(self, lit_up=0.0, shine=.2):
        n = ROWS * 10
        s = np.linspace(-ROWS * STOREY / 2, ROWS * STOREY / 2, n + 1)
        a = np.concatenate((np.arange(TUBE_END, -900.0, 1000.0), np.arange(-900.0, 900.0, 6.0), [900.0]))
        p = np.stack(on_surface(*np.meshgrid(a, s, indexing="ij")), axis=-1)
        if self.made:
            (a_lo, a_hi), (s_lo, s_hi) = self.span
            w, h = int((a_hi - a_lo) * MAP_PX), int((s_hi - s_lo) * MAP_PX)
            self.on_it = bpy.data.images.new("on_the_wall", w, h, float_buffer=True, alpha=False)
            self.on_it.colorspace_settings.name = "Non-Color"
            sheet("walls", p, polished_wall(self.made, lit_up, shine, self.on_it, self.span), np.stack(np.meshgrid(a, s, indexing="ij"), axis=-1))
            aa, ss = np.linspace(a_lo, a_hi, int((a_hi - a_lo) / 6) + 1), np.linspace(s_lo, s_hi, n + 1)
            uv = np.stack(np.meshgrid((aa - a_lo) / (a_hi - a_lo), (ss - s_lo) / (s_hi - s_lo), indexing="ij"), axis=-1)
            for i, (deep, soft, by) in enumerate(IN_THE_WALL):
                o = sheet(f"behind_{i}", np.stack(on_surface(*np.meshgrid(aa, ss, indexing="ij"), -deep), axis=-1), behind_the_wall(f"behind_{i}", self.on_it, 1 + soft, by * self.made), uv)
                o.visible_diffuse = o.visible_glossy = o.visible_transmission = o.visible_volume_scatter = o.visible_shadow = False
            return
        m = surface("night_wall", (.085, .070, .055), .85, 0, shine)
        bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Emission Color"].default_value = (1.0, .78, .55, 1)    # what all the windows round it come to, on a wall that faces all of them, near enough: it costs nothing and doesn't flicker
        bsdf.inputs["Emission Strength"].default_value = lit_up
        quads("walls", np.stack((p[:-1, :-1], p[1:, :-1], p[1:, 1:], p[:-1, 1:]), axis=2).reshape(-1, 4, 3), m)

    def wall_of_the_cut(self, as_cut):
        """The windows of the wall of the cut, whose pictures you have seen: in the first frame all of them as they were, afterwards those big enough to tell. Says what it has left for the rest to do."""
        plan = {}
        for (c, f), tex in self.pictures_of_the_wall().items():
            if not ROW0 <= f < ROW0 + ROWS or (c, f) in self.taken:
                continue
            a0, z0 = c * BAY + MARGIN, f * STOREY + SILL
            kind = self.sizes[tex]["kind"]
            c0, c1 = self.lens.see(T(a0, CUT_REVEAL, z0)), self.lens.see(T(a0 + WIN_W, CUT_REVEAL, z0 + WIN_H))
            seen = 0.0 if c0 is None or c1 is None else seen_width(c, f) if as_cut else max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1]))
            there = c0 is not None and c1 is not None and (as_cut or seen >= BY_ITSELF + 10)
            if there:
                self.on_the_wall += 1
                self.bays(T, a0, z0, tex + shrunk(seen), kind, max(seen, 18.0) if as_cut else seen, c0[2])
            plan[c, f] = (kind, there or as_cut)
        return plan

    def windows_of_the_tube(self, plan, came_on, beyond=1.0, averaged=1.0):
        """Everything else. came_on: how far those that were dark in the cut have come on; beyond: those past the end of the wall of the cut, of which there are none in the cut itself; averaged: how far the wall of the cut has gone over to what the far ones do."""
        t0 = time.time()
        cols = np.arange(COL_FAR, COL_NEAR, dtype=np.int64)
        cc, ff = np.repeat(cols, ROWS), np.tile(np.arange(ROW0, ROW0 + ROWS, dtype=np.int64), len(cols))
        aa, s0 = cc * BAY + MARGIN, ff * STOREY + SILL - MID_Z
        h = ((cc + 100000) * 73856093 + ff * 19349663) & 0xffffffff     # mixed all the way: a poorer hash shows as diagonals across the whole tube
        for shift, by in ((16, 0x85ebca6b), (13, 0xc2b2ae35), (16, 0x27d4eb2f), (15, 0x165667b1)):
            h = ((h ^ (h >> shift)) * by) & 0xffffffff
        h ^= h >> 16
        of_the_cut, turn = (cc >= 0) & (cc < CUT_COLS + 2), h / 2 ** 32 / LIT
        rising = np.where(turn < 1, np.clip((np.where(of_the_cut, came_on, beyond) * (1 + COMING_ON) - turn) / COMING_ON, 0, 1), 0.0)
        lit = rising > 0
        dark = ((h >> 3) * 2654435761 & 0xffffffff) / 2 ** 32 < np.where(of_the_cut, DARK_NEW, DARK)
        each = 1 + SWING * self.made * (2 * (((h >> 5) * 2246822519 & 0xffffffff) / 2 ** 32) - 1)
        gone = np.zeros(len(aa), bool)
        for (c, f), (kind, there) in plan.items():
            i = self.at(c, f)
            lit[i], rising[i], dark[i], gone[i] = True, 1.0, kind == "dark", there
        for c, f in self.taken:
            gone[self.at(c, f)] = True
        at, fw, rt, up = (np.array(v) for v in (self.lens.at, self.lens.f, self.lens.r, self.lens.u))

        def px(a, s, shape=False):
            p = np.stack(on_surface(a, s, CUT_REVEAL), axis=-1) - at
            d = np.maximum(p @ fw, .5)
            return self.lens.k * (p @ rt) / d, self.lens.k * (p @ up) / d, p @ fw
        x0, y0, d0 = px(aa, s0)
        x1, y1, d1 = px(aa + WIN_W, s0 + WIN_H)
        xa, ya, _ = px(aa, s0 + WIN_H / 2)
        xb, yb, _ = px(aa + WIN_W, s0 + WIN_H / 2)
        xm, ym, near = (x0 + x1) / 2, (y0 + y1) / 2, (d0 + d1) / 2
        seen, along = np.maximum(np.abs(x1 - x0), np.abs(y1 - y0)), np.hypot(xb - xa, yb - ya)
        reach = seen + self.slack
        shown = (d0 > .5) & (d1 > .5) & (np.abs(xm) < 960 + reach) & (np.abs(ym) < 540 + reach)
        # where the rows stop being windows and go on as lines
        ending = np.full(ROWS, min(ROLL_AT[0] - ROLL_LEN, at[0] - 40.0))
        ceiling, lines = ending.copy(), []
        for r in range(ROWS):
            mine = (ff == ROW0 + r) & (along < DROPPED) & (aa < ceiling[r])
            ending[r] = aa[mine].max() if mine.any() else COL_FAR * BAY
            s = (ROW0 + r) * STOREY + SILL - MID_Z - (STOREY - WIN_H) / 2
            lines.append(np.stack([np.stack(on_surface(p, q, CUT_REVEAL), axis=-1) for p, q in ((TUBE_END, s), (ending[r] + BAY, s), (ending[r] + BAY, s + STOREY), (TUBE_END, s + STOREY))]))
        shown &= aa > ending[ff - ROW0]
        quads("lit_lines", np.array(lines), glow("lines", (1.0, 1.0, 1.0), AVERAGE * DUTY * self.gain / GAIN))
        names = self.tube["names"]
        lights, darks = (np.array([i for i, n in enumerate(names) if self.sizes[n]["kind"] == w]) for w in ("light", "dark"))
        pick = np.where(dark, darks[(h >> 7) % len(darks)], lights[(h >> 7) % len(lights)])
        big = shown & lit & ~gone & (seen >= BY_ITSELF + 10)
        for i in np.nonzero(big)[0]:
            one = int(h[i] >> 7)
            if self.made and dark[i] and one % 5 < 2:       # a terminal
                name = f"code{one % 10}" + ("" if seen[i] >= 320 else "_m320" if seen[i] >= 70 else "_m160")
            elif self.made and not dark[i] and one % 9 == 0:        # somebody's question, with the answer on its way
                name = f"wait{one % 3}" + ("" if seen[i] >= 320 else "_m320" if seen[i] >= 70 else "_m160")
            elif seen[i] >= 160:
                pool = self.dark if dark[i] else self.light
                name = pool[one % len(pool)] + ("" if seen[i] >= 320 else "_m320")
            else:
                name = names[int(pick[i])] + ("_m320" if seen[i] >= 70 else "_m160")
            if rising[i] < 1:
                self.lamp(float(aa[i]), float(s0[i] + MID_Z), name, float(rising[i]), edge=.17 if dark[i] else .82)
            else:
                self.bays(T, float(aa[i]), float(s0[i] + MID_Z), name, "dark" if dark[i] else "light", float(seen[i]), float(near[i]))
        if beyond > 0:      # one nobody is in: only its frame, left where it was
            for i in np.nonzero(shown & ~lit & ~gone & (seen >= BY_ITSELF + 10))[0]:
                self.out(float(aa[i]), float(s0[i] + MID_Z))
        itself, nothing, shut, by = (a + (b - a) * self.made for a, b in ((BY_ITSELF, 10.0), (TO_NOTHING, 4.0), (CLOSING, 2.0), (2.5, 1.5)))        # what the air doesn't take is seen: a window stays one as long as it can
        fading = np.maximum(_ease((itself - np.sqrt(along * seen)) / (itself - nothing)), _ease((shut + by - along) / by)) * np.where(of_the_cut, averaged, float(beyond > 0))
        closing = _ease((shut - along) / (shut - CLOSED))
        small = shown & ~gone & (seen < BY_ITSELF + 10) & (lit | (fading > 0))
        if small.any():
            tile, across, down = pick[small], self.tube["cols"], self.tube["rows"]
            u0, v0 = (tile % across) / across, 1 - ((tile // across) + 1) / down
            du, dv = 1.0 / across, 1.0 / down
            uv = np.stack((np.stack((u0, v0), -1), np.stack((u0 + du, v0), -1), np.stack((u0 + du, v0 + dv), -1), np.stack((u0, v0 + dv), -1)), axis=1)
            c = closing[small]
            quads("lit_atlas", self.corners(aa[small], s0[small], longer=c * (BAY - WIN_W), taller=c * (STOREY - WIN_H)), atlas_material(), uv, np.stack((fading[small], rising[small], each[small] * self.gain / GAIN * WIN_W * WIN_H / ((WIN_W + c * (BAY - WIN_W)) * (WIN_H + c * (STOREY - WIN_H)))), -1))
            if self.made:       # and its frame, which at this size is one line round it
                quads("far_frames", self.corners(aa[small], s0[small], CUT_REVEAL - .01, -RINGS[0][0], -RINGS[0][0]), frame_metal(self.made))
        lamp = shown & lit & ~gone & (near < HALO_FAR) & (seen < BY_ITSELF + 10)
        if lamp.any():
            o = quads("halo_far", self.corners(aa[lamp], s0[lamp], .7, .25, .1), halo_material((1.0, .80, .58), HALO * self.halo_by))
            o.visible_camera = o.visible_glossy = o.visible_transmission = o.visible_shadow = False
            o["halo"] = 1
        if self.made:
            on = rising > 0
            for c, f in self.taken:
                on[self.at(c, f)] = False
            self.lights, self.more_lights = (aa[on], s0[on], (rising * each * self.gain * np.where(dark, DARK_LIGHT, 1.0))[on]), []
        edge = shown & (np.abs(xm) > 880) & (np.abs(xm) < 960) & (np.abs(ym) < 400)
        left, right = edge & (xm < 0), edge & (xm > 0)
        self.stats = dict(in_frame=int((shown & lit).sum()), big=int(big.sum()), small=int(small.sum()), as_lines=int(round(-TUBE_END / BAY * ROWS)), left_edge=[round(float(np.median(v[left])), 1) if left.any() else 0 for v in (np.abs(x1 - x0), np.abs(y1 - y0))], right_edge=[round(float(np.median(v[right])), 1) if right.any() else 0 for v in (np.abs(x1 - x0), np.abs(y1 - y0))], seconds=round(time.time() - t0, 1))

    def corners(self, a, s, h=CUT_REVEAL, ia=0.0, iz=0.0, longer=0.0, taller=0.0):
        pts = [(a + ia, s + iz - taller / 2), (a + WIN_W - ia + longer, s + iz - taller / 2), (a + WIN_W - ia + longer, s + WIN_H - iz + taller / 2), (a + ia, s + WIN_H - iz + taller / 2)]
        return np.stack([np.stack(on_surface(p, q, h), axis=-1) for p, q in pts], axis=1)

    def the_far_end(self):
        """The end of the lines, which you won't get to."""
        n = 96
        th = np.linspace(-math.pi, math.pi, n, endpoint=False)
        rim = np.stack((np.full(n, TUBE_END), AXIS_Y + RADIUS * np.cos(th), MID_Z + RADIUS * np.sin(th)), -1)
        mid = np.array((TUBE_END, AXIS_Y, MID_Z))
        quads("lit_tube_far", np.stack((rim, np.roll(rim, -1, axis=0), np.full((n, 3), mid), np.full((n, 3), mid)), axis=1), glow("tube_far", (1.0, 1.0, 1.0), AVERAGE * DUTY * self.gain / GAIN))

    def pane(self, a0, z0, rest=None):
        """The window as a thing: a frame of dark metal standing off the wall, its edges rounded so that they catch what light there is, and glass in front of the screen. Says how to find a point of it: along it, up it from its middle, and how far off the wall."""
        made = self.made
        u = luck(a0, rest if rest is not None else z0)
        ca, cz = a0 + WIN_W / 2, z0 + WIN_H / 2
        off, ta, tz, spin = SIT * made * (2 * u[0] - 1), TILT * made * (2 * u[1] - 1), TILT * made * (2 * u[2] - 1), SPIN * made * (2 * u[3] - 1)
        cs, sn = math.cos(spin), math.sin(spin)

        def at(a, b, h):
            return T(ca + a * cs - b * sn, h + off + ta * a + tz * b, cz + a * sn + b * cs)
        rings = [outline(WIN_W + 2 * e * made, WIN_H + 2 * e * made, max(.066 + e * made, .02), 5) for e, _ in RINGS]
        n = len(rings[0])
        pts = [at(a, b, CUT_REVEAL + (h - CUT_REVEAL) * made) for ring, (_, h) in zip(rings, RINGS) for a, b in ring]
        self.metal.add(pts, [[r * n + (i + 1) % n, r * n + i, (r + 1) * n + i, (r + 1) * n + (i + 1) % n] for r in range(len(RINGS) - 1) for i in range(n)])
        pane = outline(WIN_W + .008, WIN_H + .008, .07, 5)
        self.glass.add([at(a, b, GLASS_AT) for a, b in pane], [list(range(len(pane)))])
        return at

    def frame(self, a0, z0, rest=None):
        ca, cz = a0 + WIN_W / 2, z0 + WIN_H / 2
        inset, ring = outline(WIN_W - .028, WIN_H - .028, .05, 3), outline(WIN_W, WIN_H, .064, 3)
        m = len(inset)
        if self.made:
            whole = self.pane(a0, z0, rest)
            at = lambda a, b: whole(a, b, CUT_REVEAL)
        else:
            at = lambda a, b: T(ca + a, CUT_REVEAL, cz + b)
        return inset, [at(a, b) for a, b in inset], [at(a, b) for a, b in ring], [[i, (i + 1) % m, m + (i + 1) % m, m + i][::-1] for i in range(m)]

    def glint_at(self, b, a0, z0):
        b.add([T(a0 - .1, GLINT_AT, z0 - .1), T(a0 + WIN_W + .1, GLINT_AT, z0 - .1), T(a0 + WIN_W + .1, GLINT_AT, z0 + WIN_H + .1), T(a0 - .1, GLINT_AT, z0 + WIN_H + .1)], [(0, 1, 2, 3)], [(0, 0), (1, 0), (1, 1), (0, 1)])

    def glinting(self, b, name, level, warm=None):
        m = halo_material(tuple(a * c for a, c in zip(GLINT_WARM, warm or (1, 1, 1))), 1.0)
        nt = m.node_tree
        soft, by = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeMath")
        soft.image, soft.extension = soft_edge(), "EXTEND"
        nt.links.new(nt.nodes.new("ShaderNodeUVMap").outputs[0], soft.inputs[0])
        by.operation = "MULTIPLY"
        near = nt.nodes.new("ShaderNodeMapRange")       # for what is beside it: from across the tube it would only be a smudge in a dark screen
        near.interpolation_type = "SMOOTHSTEP"
        near.inputs["From Min"].default_value, near.inputs["From Max"].default_value = 3.0, 9.0
        near.inputs["To Min"].default_value, near.inputs["To Max"].default_value = level * self.made, 0.0
        nt.links.new(nt.nodes.new("ShaderNodeLightPath").outputs["Ray Length"], near.inputs["Value"])
        nt.links.new(near.outputs[0], by.inputs[1])
        nt.links.new(soft.outputs[0], by.inputs[0])
        nt.links.new(by.outputs[0], next(n for n in nt.nodes if n.type == "EMISSION").inputs[1])
        o = b.object("halo_glint_" + name, m)
        o.visible_camera = o.visible_diffuse = o.visible_transmission = o.visible_volume_scatter = o.visible_shadow = False
        o["halo"] = 1

    def light_on_the_wall(self):
        """What the windows come to on the wall round them, and in the polish of it, each by what it has on it: all of them, then the two that have their own lamp."""
        (a_lo, _), (s_lo, _) = self.span
        w, h = self.on_it.size
        lit = np.zeros((h, w), np.float32)
        a, s, by = self.lights
        a, s, by = np.concatenate((a, [v[0] for v in self.more_lights])), np.concatenate((s, [v[1] - MID_Z for v in self.more_lights])), np.concatenate((by, [v[2] for v in self.more_lights]))
        x0, x1 = np.rint((a - a_lo) * MAP_PX).astype(int), np.rint((a + WIN_W - a_lo) * MAP_PX).astype(int)
        y0, y1 = np.rint((s - s_lo) * MAP_PX).astype(int), np.rint((s + WIN_H - s_lo) * MAP_PX).astype(int)
        for i in np.nonzero((x1 > 0) & (x0 < w) & (y1 > 0) & (y0 < h))[0]:
            lit[max(y0[i], 0):y1[i], max(x0[i], 0):x1[i]] += by[i]
        px = np.ones((h, w, 4), np.float32)
        px[..., 0] = sum(k * blurred(lit, far * MAP_PX) for far, k in ON_THE_WALL)
        px[..., 1], px[..., 2] = blurred(lit, SOFT[0] * MAP_PX), blurred(lit, SOFT[1] * MAP_PX)
        self.on_it.pixels.foreach_set(px.ravel())
        self.on_it.update()

    def bays(self, T_, a0, z0, tex, kind, seen, d, k=1.0):
        if not self.made:
            return super().bays(T_, a0, z0, tex, kind, seen, d, k)
        self.count += 1
        if d < HALO_FAR and self.made < 1:
            self.halo.add([T(a0 + .25, .7, z0 + .1), T(a0 + WIN_W - .25, .7, z0 + .1), T(a0 + WIN_W - .25, .7, z0 + WIN_H - .1), T(a0 + .25, .7, z0 + WIN_H - .1)], [(0, 1, 2, 3)])
        self.glint_at(self.glints["dark" if kind == "dark" else "light"], a0, z0)
        inset, face, ring, sides = self.frame(a0, z0)
        level = 1 + SWING * self.made * (2 * luck(a0, z0)[4] - 1)
        self.screens.setdefault((tex, int(round(level * 100))), Build()).add(face, [list(range(len(face)))], [(a / WIN_W + .5, b / WIN_H + .5) for a, b in inset])
        self.edges[kind if kind == "dark" else "light"].add(face + ring, sides)

    def done(self, gain=GAIN, prefix="lit_"):
        super().done(gain, prefix)
        if not self.made:
            return
        for key in ("light", "dark"):
            bpy.data.materials["lit_edge_" + key]["words"] *= 1 - self.made
        for o in bpy.data.objects:      # the wall does this itself now
            if o.get("halo") and o.name.startswith("halo_near"):
                o.data.materials[0].node_tree.nodes["Emission"].inputs[1].default_value *= self.halo_by
        self.light_on_the_wall()
        for (tex, level), b in self.screens.items():
            b.object(f"lit_{tex}_{level}", page_material(f"{tex}_{level}", tex, self.gain, level / 100, self.warm))
        self.metal.object("frames", frame_metal(self.made)).data.shade_smooth()
        self.glass.object("panes", pane_glass(self.made))["glass"] = 1
        for kind, b in self.glints.items():
            self.glinting(b, kind, GLINT * (DARK_LIGHT if kind == "dark" else 1.0))

    def lamp(self, a0, z0, tex, level, warm=None, edge=.82):
        """One window at a level of its own, its picture, its rim and its light on the wall going down together, as a lamp's does, and redder if it is told to, down to what a shut one's glass holds."""
        inset, face, ring, sides = self.frame(a0, z0)
        m = page_material(tex, tex, self.gain, level, tuple(a * b for a, b in zip(warm or (1, 1, 1), self.warm)), OUT_GLASS * max(1 - level, 0))
        if self.more_lights is not None:
            self.more_lights.append((a0, z0, level * self.gain * (DARK_LIGHT if edge < .5 else 1.0)))
        pic, rim, light = Build(), Build(), Build()
        pic.add(face, [list(range(len(face)))], [(a / WIN_W + .5, b / WIN_H + .5) for a, b in inset])
        pic.object("lit_lamp_" + tex, m)
        rim.add(face + ring, sides)
        rim.object("lit_rim_" + tex, glow("rim_" + tex, tuple(a * b for a, b in zip((1.0, .97, .92), warm or (1, 1, 1))), max(edge * level * (1 - self.made), OUT_RIM)))
        light.add([T(a0 + .25, .7, z0 + .1), T(a0 + WIN_W - .25, .7, z0 + .1), T(a0 + WIN_W - .25, .7, z0 + WIN_H - .1), T(a0 + .25, .7, z0 + WIN_H - .1)], [(0, 1, 2, 3)])
        h = light.object("halo_" + tex, halo_material(tuple(a * b for a, b in zip((1.0, .80, .58), warm or (1, 1, 1))), HALO * level * self.halo_by))
        h.visible_camera = h.visible_glossy = h.visible_transmission = h.visible_shadow = False
        h["halo"] = 1
        if self.made:
            glint = Build()
            self.glint_at(glint, a0, z0)
            self.glinting(glint, tex, GLINT * level * (DARK_LIGHT if edge < .5 else 1.0), warm)

    def under(self, a0, z0, rest=None):
        """The same, for the two under the pair, which go when you get close: the foot of K4 has nothing in it for the subtitle."""
        inset, face, ring, sides = self.frame(a0, z0, rest)
        self.under_glass.add(face, [list(range(len(face)))])
        self.under_rim.add(face + ring, sides)

    def out(self, a0, z0):
        """What is left when the lamp has gone out: the frame, dark grey, and the glass, a little darker, holding no light of its own."""
        inset, face, ring, sides = self.frame(a0, z0)
        self.shut_glass.add(face, [list(range(len(face)))])
        self.shut_rim.add(face + ring, sides)

    def all_out(self, level, under):
        for glass, rim, name, by in ((self.shut_glass, self.shut_rim, "out", level), (self.under_glass, self.under_rim, "under", under)):
            glass.object(f"lit_{name}_glass", see_through(glow(f"{name}_glass", (1.0, .97, .93), OUT_GLASS), by))
            rim.object(f"lit_{name}_rim", see_through(glow(f"{name}_rim", (1.0, .97, .93), OUT_RIM), by))


OUT_GLASS, OUT_RIM = .007, .06
COMING_ON = .2                                  # how much of the time they take to come on one window takes, so that none of them does it in a frame
DIM, EMBER, EMBER_TINT = .62, .2, (1.0, .84, .66)      # the right hand window before anyone has pressed anything; the left half way out
FOG_LEN, FOG_DEEP, FOG_TINT = 75.0, (.02, .0135, .008), (.475, .29, .06)   # the air takes the far windows down slowly and warms what is left, and whatever is looked at in the glass and the wall is looked at through it
SPILL = .10                                     # how much of a window's light gets as far as the walls that face it, while the wall is still what it was in the cut; all of it when it is the wall it is going to be
WALL_GLOW = .0003                               # and the wall as a whole, so that where a window has gone there is still a wall to see
SHINE = .02                                     # the wall's sheen: seen as low along it as this, any more and it is a brown mirror of all the windows
K4 = dict(t5_near_window=(("k4_open", 1.0), ("k4_typing", DIM)), K4a=(("k4_closing", .88), ("k4_typing", DIM)), K4a2=(("k4_closing", EMBER, EMBER_TINT), ("k4_typing", DIM)), K4b=(None, ("k4_typing", DIM)), K4c=(None, ("k4_sent", 1.0)))
K4_AT = dict(K4d=SHUTS + .4, K4f=SHUTS + 1.4, K4e=NEXT_LINE + .6)        # all four lit; the left one going out; and the left one out, the one beside it just come on, by the film's clock
SEQ = os.path.join(PLATES, "..", "tube_low3")


def fog_at(t):
    """How far you can see and what colour the far is: what it was in the cut, and what the whole length of the tube wants, by the time you are looking along it."""
    k = smooth(T_READ, T_MADE, t)
    return SMOG_LEN ** (1 - k) * FOG_LEN ** k, tuple(a ** (1 - k) * b ** k for a, b in zip(SMOG, FOG_DEEP)), SMOG_SCATTER * (1 - k), tuple(a + (b - a) * k for a, b in zip(SMOG_TINT, FOG_TINT))


def spills(amount, far=1.0):
    """The windows' light on what faces them. Nothing is aimed at them: with this much of a tube lit, a ray that bounces hits one anyway. far: how much of it counts from across the tube, where in a dark screen's glass it would only be a smudge."""
    for m in bpy.data.materials:
        if m.name.startswith("lit_") and "words" in m:
            m.cycles.emission_sampling = "NONE"
            nt = m.node_tree
            mul = next(n for n in nt.nodes if n.type == "MATH" and n.operation == "MULTIPLY")
            if far == 1.0:
                mul.inputs[1].default_value = m["words"] * amount
                continue
            path, by = next((n for n in nt.nodes if n.type == "LIGHT_PATH"), None) or nt.nodes.new("ShaderNodeLightPath"), nt.nodes.new("ShaderNodeMapRange")
            by.interpolation_type = "SMOOTHSTEP"
            by.inputs["From Min"].default_value, by.inputs["From Max"].default_value = 8.0, 20.0
            by.inputs["To Min"].default_value, by.inputs["To Max"].default_value = m["words"] * amount, m["words"] * amount * far
            nt.links.new(path.outputs["Ray Length"], by.inputs["Value"])
            nt.links.new(by.outputs[0], mul.inputs[1])


def wisps(x, y, t, seed):
    """A very slow unevenness, about -1 to 1, tied to the far point: it is out there, it doesn't slide over the frame with the turn."""
    out = 0.0
    for i, (across, angle, drift, by) in enumerate(((1500.0, .4, .07, .5), (900.0, 1.9, -.11, .32), (560.0, 3.1, .15, .18))):
        p = 2 * math.pi * (x * math.cos(angle + seed) + y * math.sin(angle + seed)) / across + drift * t + seed * (i + 1) * 2.1
        q = 2 * math.pi * (-x * math.sin(angle + seed) + y * math.cos(angle + seed)) / (across * 1.3) - drift * .7 * t + seed
        out = out + by * np.sin(p + 1.1 * np.sin(q))
    return out


def fog_plate(path, lens, strength, pct, t=0.0):
    """How far each pixel looks off the length of the tube, in pixels of the lens, and how much of the air has come yet. What it comes to is put on afterwards, film_windows_ui._graded, so it can have its own curve."""
    w, h = int(1920 * pct / 100), int(1080 * pct / 100)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = (x + .5) * 100 / pct, (y + .5) * 100 / pct
    f, r, u = (np.array(v, np.float32) for v in (lens.f, lens.r, lens.u))
    d = f + ((x - 960) / lens.k)[..., None] * r + ((540 - y) / lens.k)[..., None] * u
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    along = d @ AXIS_DIR.astype(np.float32)
    off = lens.k * np.sqrt(np.maximum(1 - along * along, 0)) / np.maximum(along, 1e-4)
    ax = AXIS_DIR.astype(np.float32)
    deep = max(float(ax @ f), 1e-3)
    vx, vy = x - (960 + lens.k * float(ax @ r) / deep), y - (540 - lens.k * float(ax @ u) / deep)
    oy, oz = lens.at[1] - AXIS_Y, lens.at[2] - MID_Z        # and how far it is to the wall, along each pixel: the wall is a cylinder, so that needs no pass of its own
    a, b, c = d[..., 1] ** 2 + d[..., 2] ** 2, oy * d[..., 1] + oz * d[..., 2], oy * oy + oz * oz - RADIUS * RADIUS
    wall = (-b + np.sqrt(b * b - a * c)) / np.maximum(a, 1e-9)
    np.save(path, np.dstack((np.where(along > 1e-3, off, 1e9), np.full_like(off, strength), wisps(vx, vy, t, 0.0), wisps(vx, vy, t, 1.7), wall)).astype(np.float32))


def seq_plan(frames):
    """Every frame to the end of K4c, none of them standing still now: there is always somebody at work up there."""
    out = [dict(t=n / FPS, src=n) for n in range(int(round((NEXT_LINE + 1.2 - FILM0) * FPS)) + 1)]
    os.makedirs(SEQ, exist_ok=True)
    with open(os.path.join(SEQ, "way.json"), "w") as f:
        json.dump(dict(fps=FPS, film0=FILM0, end=T_END, line_gone=next(c["t1"] for c in ink_subs.cues() if c["text"].startswith("彼此")) - FILM0, frames=out), f)
    print(len(out), "frames,", len({o["src"] for o in out}), "to render", flush=True)


def main():
    key, pct, samples = ARGS[0], opt("pct", 100), opt("samples", 128)
    frames, info = solve()
    if key == "way":
        report(frames, info)
        return
    if key == "seqplan":
        seq_plan(frames)
        return
    t0 = time.time()
    seq = key[0] == "f" and key[1:].isdigit()
    if seq:
        t, lens = pose_of(key, frames, int(key[1:]))
        pair = lamps(FILM0 + t) if t >= T_FLIP else None
        base = os.path.join(SEQ, "plates", f"w{opt('worker', 0)}")
    else:
        t, lens = pose_of(key, frames)
        pair = lamps(K4_AT[key]) if key in K4_AT else K4[key] + others(FILM0 + t) if key in K4 else lamps(FILM0 + t) if t >= T_FLIP else None
        base = os.path.join(PLATES, key)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    print(key, "at", round(t, 2), "s after the cut, camera", [round(v, 2) for v in lens.at], "looking", [round(float(v), 3) for v in lens.f], flush=True)
    sc = set_up(samples, pct)
    came_on, rolled = smooth(T_READ, T_READ + 2.6, t), smooth(T_READ, T_RING, t)
    ROLL_AT[0] = 40.0 + (ROLL_END - 40.0) * rolled
    length, fog, scatter, tint = fog_at(t)
    sky(sc, 0, (.0014, .0011, .0009), fog)
    made = smooth(T_READ, T_MADE, t)
    city = Tube(lens, made)
    z0 = K4_ROW * STOREY + SILL
    four, was = {(c, K4_ROW - r) for c in (info["col"], info["col"] + 1) for r in (0, 1)}, city.pictures_of_the_wall()       # the two, and the two under them
    if pair:
        city.taken = four | {(c, K4_ROW + 1) for c in (info["col"], info["col"] + 1)}       # and now the two above
    else:           # what they are in the cut until then, and what is empty in the cut stays empty
        going = smooth(T_FLIP - T_SWAP, T_FLIP, t)
        city.taken = {at for at in four if at not in was or (going > 0 and at[0] != info["col"])}
    city.wall_of_the_tube(WALL_GLOW * rolled, .2 + (SHINE - .2) * rolled)
    plan = city.wall_of_the_cut(t < T_READ)
    frames_on = smooth(0, T_READ, t)
    city.windows_of_the_tube(plan, came_on, frames_on, smooth(T_READ, T_READ + 1.0, t))
    if pair:
        rise = smooth(T_FLIP, T_FLIP + T_SWAP, t)
        for i, (a0, what) in enumerate(zip((info["a0"], info["a0"] + BAY), pair)):
            if what:
                city.lamp(a0, z0, what[0], what[1] * (rise if i and key not in K4 else 1.0), *what[2:])
            else:
                city.out(a0, z0)
            city.under(a0, z0 - STOREY)
        for a0, tex, edge in zip((info["a0"], info["a0"] + BAY), pair[2:], (.82, .17)):
            city.lamp(a0, z0 + STOREY, tex, 1 + SWING * made * (2 * luck(a0, z0 + STOREY)[4] - 1), edge=edge)
    elif t > 0:
        for c, f in sorted(city.taken):
            a0, z = c * BAY + MARGIN, f * STOREY + SILL
            if (c, f) in was:
                city.lamp(a0, z, was[c, f], 1 - going, edge=.17 if city.sizes[was[c, f]]["kind"] == "dark" else .82)
            else:
                city.under(a0, z)
    city.the_far_end()
    city.all_out(frames_on, frames_on)
    city.done()
    if t > 0:
        spills(SPILL * rolled + (1 - SPILL * rolled) * made, 1 + (ACROSS - 1) * made)
    if t > 0 or seq:
        fog_plate(base + "_fog.npy", lens, rolled, pct, t)
    print(city.stats, "of the wall", city.on_the_wall, flush=True)
    if t == 0:
        air(SMOG_LEN, -1400.0, 1400.0, 2000.0, 140.0, SMOG, SMOG_TINT, SMOG_SCATTER)
    else:
        air(length, -700.0, 100.0, 160000.0, 420.0, fog, tint, scatter, -240.0)
    lens_on(sc, lens.at, lens.target(), CUT_LENS)
    sc.camera.matrix_world = lens.matrix()
    sc.camera.data.clip_start = .1
    sc.camera.data.clip_end = 400000.0
    if made:
        far_light(lens.at[0], made)
        sc.cycles.max_bounces, sc.cycles.glossy_bounces, sc.cycles.transparent_max_bounces = 8, 5, 16
        focus = focus_of(t, lens, info)
        sc.camera.data.dof.use_dof = True
        sc.camera.data.dof.focus_distance = focus
        blur = BLUR_FAR + (BLUR_CLOSE - BLUR_FAR) * smooth(10.0, 13.0, t)
        sc.camera.data.dof.aperture_fstop = (CUT_LENS / 1000) / max(blur * made * focus / lens.k, 1e-4)
    if seq and int(key[1:]) > 0:        # half a frame's worth of the camera moving, as a film camera has it
        n, cam = int(key[1:]), sc.camera
        cam.rotation_mode = "QUATERNION"
        sc.render.use_motion_blur, sc.render.motion_blur_shutter = True, .5
        for off in (-1, 0, 1):
            m = pose_of(key, frames, max(n + off, 1))[1].matrix()
            cam.location, cam.rotation_quaternion = m.to_translation(), m.to_quaternion()
            cam.keyframe_insert("location", frame=100 + off)
            cam.keyframe_insert("rotation_quaternion", frame=100 + off)
        sc.frame_set(100)
    print("built", round(time.time() - t0, 1), flush=True)
    plate(sc, base + "_room")
    print(key, "room", round(time.time() - t0, 1), flush=True)
    t1 = time.time()
    only_the_windows(sc)
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.samples = max(samples, 64)
    plate(sc, base + "_words")
    print(key, "words", round(time.time() - t1, 1), flush=True)
    if key == "t1_wall":
        mask_plate(sc, os.path.join(PLATES, key + "_mask"), ("lit_",))


if __name__ == "__main__":
    main()
