"""九 千窗: the way back. The camera comes back along its own axis through the cut, so that the wall only gets smaller where it stands; then it goes on back and turns as it goes, into the street K3 looks down, past the ends of the blocks and not over them. It goes as fast as what is nearest in the frame lets it, so that nothing grows or shrinks by more than a set amount in a second.

    Blender -b --factory-startup -P film_windows_move.py -- way                                   the path and the clock: seq/way.json, and how they measure up
    Blender -b --factory-startup -P film_windows_move.py -- street [--pct=50] [--samples=16]      the frames up to the cut
    Blender -b --factory-startup -P film_windows_move.py -- city [--pct=50] [--samples=16]        the frames from the cut on

Then film_windows_ui.py seq. --dir=seq_test keeps a trial apart.
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

from film_windows_city import ARGS, CUT_AT, CUT_LENS, CUT_PITCH, CUT_YAW, OUT, STOREY, cut_camera, lens_on, only_the_windows, opt, plate, set_up, sky, street_c3a
from film_windows_grid import AT, SMOG, SMOG_LEN, SMOG_SCATTER, SMOG_TINT, Lattice, Lens, air, blocks, the_city

FPS = 30
T_START, T_CUT, T_END = 10.5, 12.0, 20.0        # on the plan's clock for 九: the street a moment, the cut as the subtitle comes in, and the frame K3 as it goes
T_READ, T_CRUISE, T_SLOW, T_STOP = 13.2, 14.4, 17.9, 19.7   # still slow enough to read the three windows; up to pace; beginning to settle; there
T_TURN0, T_TURN1 = 13.4, 19.5                   # the turn into the street
RATE_SLOW = math.log(1.1)                       # how fast things shrink while you can read them, a second
BACK = (math.sin(math.radians(CUT_YAW)) * math.cos(math.radians(CUT_PITCH)), -math.cos(math.radians(CUT_YAW)) * math.cos(math.radians(CUT_PITCH)), -math.sin(math.radians(CUT_PITCH)))
# the way, as a Bézier from the cut to K3: leaving straight back along the axis, arriving down and across, east of the end of the row of blocks on the right
LEAVE, ARRIVE, ARRIVING = 60.0, 45.0, (.25, -.93, -.25)
GENTLE = .22                                    # seconds over which the speed is smoothed


def folder():
    return os.path.join(OUT, opt("dir", "seq"))


def smooth(a, b, x):
    x = min(max((x - a) / (b - a), 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Way:
    """The path by the metre."""

    def __init__(self, n=4000):
        A, D = np.array(CUT_AT, float), np.array(AT["K3"][0], float)
        B, C = A + LEAVE * np.array(BACK), D - ARRIVE * np.array(ARRIVING)
        u = np.linspace(0, 1, n)[:, None]
        self.points = (1 - u) ** 3 * A + 3 * (1 - u) ** 2 * u * B + 3 * (1 - u) * u ** 2 * C + u ** 3 * D
        self.metres = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))))
        self.length = self.metres[-1]

    def at(self, s):
        if s < 0:       # before the cut: straight on along the same axis, nearer the wall
            return tuple(np.array(CUT_AT) + s * np.array(BACK))
        return tuple(np.interp(s, self.metres, self.points[:, i]) for i in range(3))


def boxes():
    """Every block as x0, x1, y0, y1, 0, top."""
    out = np.array([(a, b, c, d, 0.0, n * STOREY) for a, b, c, d, n in blocks()])
    return out


def turn(t):
    k = smooth(T_TURN0, T_TURN1, t)
    return CUT_YAW + (AT["K3"][1] - CUT_YAW) * k, CUT_PITCH + (AT["K3"][2] - CUT_PITCH) * k


class Sight:
    """How far away what the camera is looking at is, by rays across the frame."""

    def __init__(self, cols=40, rows=23):
        self.box = boxes()
        px = 1920 * CUT_LENS / 36
        self.x = (np.linspace(0, 1920, cols) - 960) / px
        self.y = (540 - np.linspace(0, 1080, rows)) / px

    def depths(self, at, yaw, pitch):
        lens = Lens(at, yaw, pitch, CUT_LENS)
        f, r, u = np.array(lens.f), np.array(lens.r), np.array(lens.u)
        xx, yy = np.meshgrid(self.x, self.y)
        rays = f + xx[..., None] * r + yy[..., None] * u
        rays = rays.reshape(-1, 3)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        p = np.array(at)
        near = np.linalg.norm(self.box[:, [0, 2, 4]] - p, axis=1) < 900
        box = self.box[near]
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = (box[:, [0, 2, 4]][None] - p) / rays[:, None]
            t1 = (box[:, [1, 3, 5]][None] - p) / rays[:, None]
        lo, hi = np.minimum(t0, t1).max(axis=2), np.maximum(t0, t1).min(axis=2)
        hit = (hi >= np.maximum(lo, 0))
        d = np.where(hit, np.maximum(lo, 0), np.inf).min(axis=1)
        return d[np.isfinite(d)]

    def nearest(self, at, yaw, pitch):
        d = self.depths(at, yaw, pitch)
        return float(np.percentile(d, 12)) if len(d) else 400.0


def fly(way, sight, cruise):
    """Metres along the way for each frame from the cut, at this pace. The camera's speed is the pace times how far away what it sees is."""
    n = int(round((T_END - T_CUT) * FPS))
    s, out = 0.0, []
    for i in range(n + 1):
        t = T_CUT + i / FPS
        out.append((t, s))
        pace = RATE_SLOW + (cruise - RATE_SLOW) * smooth(T_READ, T_CRUISE, t)
        pace *= 1 - smooth(T_SLOW, T_STOP, t)
        yaw, pitch = turn(t)
        s = s + pace * sight.nearest(way.at(s), yaw, pitch) / FPS
    return out


def solve():
    way, sight = Way(), Sight()
    lo, hi = .1, 2.0
    for _ in range(14):
        mid = (lo + hi) / 2
        if fly(way, sight, mid)[-1][1] < way.length:
            lo = mid
        else:
            hi = mid
    cruise = (lo + hi) / 2
    run = fly(way, sight, cruise)
    # what comes into the frame comes in all at once, and the speed would jump with it: an eye forgives anything but that
    speed = np.diff([s for _, s in run])
    wide = int(GENTLE * FPS)
    kernel = np.exp(-.5 * (np.arange(-3 * wide, 3 * wide + 1) / wide) ** 2)
    calm = np.convolve(np.pad(speed, 3 * wide, mode="edge"), kernel / kernel.sum(), "valid")
    metres = np.concatenate(([0.0], np.cumsum(calm)))
    run = [(t, m) for (t, _), m in zip(run, metres)]
    k = way.length / run[-1][1]
    frames = []
    first = RATE_SLOW * sight.nearest(CUT_AT, CUT_YAW, CUT_PITCH)
    for i in range(int(round((T_CUT - T_START) * FPS))):
        t = T_START + i / FPS
        frames.append(dict(t=t, at=way.at(first * (t - T_CUT)), yaw=CUT_YAW, pitch=CUT_PITCH))
    cut = len(frames)
    for t, s in run:
        yaw, pitch = turn(t)
        frames.append(dict(t=t, at=way.at(min(s * k, way.length)), yaw=yaw, pitch=pitch))
    return dict(fps=FPS, cut=cut, t_cut=T_CUT, t_end=T_END, frames=frames, cruise=cruise, length=way.length), sight


def report(data, sight):
    """How it measures up: what the eye gets, frame by frame."""
    box = sight.box
    print("way", round(data["length"], 1), "m, pace at its fastest", round(math.exp(data["cruise"]), 3), "times a second", flush=True)
    frames = data["frames"]
    wall_at = np.array(CUT_AT) + 22.6 * -np.array(BACK)
    worst = 0
    table = []
    for n in range(1, len(frames)):
        a, b = frames[n - 1], frames[n]
        p, q = np.array(a["at"]), np.array(b["at"])
        speed = np.linalg.norm(q - p) * FPS
        clear = min(np.linalg.norm(np.maximum(np.maximum(bx[[0, 2, 4]] - q, 0), q - bx[[1, 3, 5]])) for bx in box)
        d = sight.depths(b["at"], b["yaw"], b["pitch"])
        near = float(np.percentile(d, 12)) if len(d) else 400.0
        rate = speed / near
        worst = max(worst, rate)
        table.append((b["t"], rate))
        spin = abs(b["yaw"] - a["yaw"]) * FPS
        wall = abs(math.log(np.linalg.norm(q - wall_at) / np.linalg.norm(p - wall_at))) * FPS
        if n % 15 == 0:
            print(f"t {b['t']:5.2f} at {q[0]:6.1f} {q[1]:6.1f} {q[2]:6.1f} yaw {b['yaw']:5.1f} pitch {b['pitch']:5.1f} speed {speed:5.1f} m/s nearest seen {near:6.1f} m clear of anything {clear:5.1f} m scale {math.exp(rate):4.2f}x/s wall {math.exp(wall):4.2f}x/s turn {spin:4.1f} deg/s", flush=True)
    print("fastest scale, by what is nearest in the frame", round(math.exp(worst), 3), "times a second, at", [round(t, 2) for t, r in table if r > worst * .97], flush=True)
    print("nearest the way comes to anything", round(min(min(np.linalg.norm(np.maximum(np.maximum(bx[[0, 2, 4]] - np.array(f["at"]), 0), np.array(f["at"]) - bx[[1, 3, 5]])) for bx in box) for f in frames), 2), "m", flush=True)


def build_city(sc, frames, cut):
    t0 = time.time()
    lenses = [Lens(f["at"], f["yaw"], f["pitch"], CUT_LENS) for f in frames[cut::8]] + [Lens(*AT["K3"])]
    sky(sc, 0, (.0014, .0011, .0009), SMOG)
    city = Lattice(Lens(*AT["C3b"]), Lens(*AT["K3"]), lenses)
    city.the_wall(True)
    the_city(city)
    city.done()
    air(SMOG_LEN, -1400.0, 1400.0, 2000.0, 140.0, SMOG, SMOG_TINT, SMOG_SCATTER)
    print(city.count, "windows", len(city.pictures), "pictures", round(time.time() - t0, 1), "s", flush=True)


def shoot(sc, frames, numbers, out, words_samples):
    cam = sc.camera
    from ink3d import aim
    for tail in ("room", "words"):
        t0 = time.time()
        if tail == "words":
            only_the_windows(sc)
            sc.cycles.use_denoising = False
            sc.cycles.use_adaptive_sampling = False
            sc.cycles.samples = words_samples
        for n in numbers:
            f = frames[n]
            lens = Lens(f["at"], f["yaw"], f["pitch"], CUT_LENS)
            aim(cam, f["at"], lens.target())
            plate(sc, os.path.join(out, f"f{n:04d}_{tail}"))
        print(tail, len(numbers), "frames", round((time.time() - t0) / max(len(numbers), 1), 2), "s each", flush=True)


def main():
    part, pct, samples = ARGS[0], opt("pct", 50), opt("samples", 16)
    base = folder()
    os.makedirs(os.path.join(base, "plates"), exist_ok=True)
    path = os.path.join(base, "way.json")
    if part == "way":
        data, sight = solve()
        data["width"], data["height"] = 1920 * pct // 100, 1080 * pct // 100
        with open(path, "w") as f:
            json.dump(data, f)
        report(data, sight)
        return
    with open(path) as f:
        data = json.load(f)
    frames, cut = data["frames"], data["cut"]
    every = opt("every", 1)
    sc = set_up(samples, pct)
    sc.render.use_persistent_data = True
    if part == "street":
        sky(sc)
        street_c3a()
        lens_on(sc, *cut_camera())
        shoot(sc, frames, range(0, cut, every), os.path.join(base, "plates"), max(samples, 16))
    else:
        build_city(sc, frames, cut)
        lens_on(sc, *cut_camera())
        shoot(sc, frames, range(cut, len(frames), every), os.path.join(base, "plates"), max(samples, 16))


if __name__ == "__main__":
    main()
