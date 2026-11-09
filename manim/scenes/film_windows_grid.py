"""九 千窗: the city of windows. The building has gone and its lattice has not. Past the cut the camera goes back and up, and the one wall turns out to be one of many, in rows and across them, the same bays and the same floors, with nothing in them but Claude's windows, and the air getting thick between you and the far ones.

    Blender -b --factory-startup -P film_windows_grid.py -- C3b [--pct=100] [--samples=128]     the other side of the cut: only the wall you were looking at
    Blender -b --factory-startup -P film_windows_grid.py -- K3a|K3|K2|K4a|K4b [--pct=100] [--samples=128]

Plates as in film_windows_city.py. x along the rows, y across them, z up, in metres. The camera leaves the wall at CUT_AT along the way it was looking, to AT["K3"].
"""
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
from mathutils import Matrix

import film_windows3d as w3
from film_windows3d import Build, outline
from film_windows_city import ARGS, BAY, CUT_AT, CUT_COLS, CUT_FLOORS, CUT_LENS, CUT_PITCH, CUT_REVEAL, CUT_YAW, MARGIN, PLATES, SILL, STOREY, SUBTITLE, TEX, WIN_H, WIN_W, air, cuboid, cut_plan, glow, halo_material, lens_on, lit_window, mask_plate, only_the_windows, opt, plate, seen_width, set_up, shrunk, sky, surface

GAIN = .95
SMOG, SMOG_TINT, SMOG_LEN, SMOG_SCATTER = (.026, .0145, .007), (.55, .30, .08), 240.0, 1 / 1600.0
HALO, HALO_FAR = 1.7, 240.0         # how bright the light of a window is on what is next to it; past this distance it doesn't count
LIT, DARK = .78, .12                # how many of the windows are lit, and of those how many a dark terminal
WALL_X = CUT_COLS * BAY + 24.0      # the wall of the cut: its windows, and the stair end
FAR = 1500.0
LIFT = dict(hero_k3=1.14, k4_closing=.88)      # the one you are meant to find, a touch above the rest; the one going, as it starts to fade

AT = dict(
    C3b=(CUT_AT, CUT_YAW, CUT_PITCH, CUT_LENS),
    K3=((381.7, -89.9, 113.9), 94.0, -7.0, 30.0),
    K2=((300.0, -52.0, 118.0), 90.0, 0.0, 35.0),
)
# 一念无存，一念生: two windows side by side, two floors under the one you found. The left is being shut, by the ordinary means; the place it leaves is wall, like any window not lit. The right has a stranger in it
K4_FLOOR, K4_SHUT, K4_NEW = 36, 5, 6
AT["K4a"] = AT["K4b"] = ((376.22, -98.75, 105.6), 78.0, 0.0, 30.0)
K4_SHRINK = .94
AT["K3a"] = (tuple((a + b) / 2 for a, b in zip(CUT_AT, AT["K3"][0])), (CUT_YAW + AT["K3"][1]) / 2, (CUT_PITCH + AT["K3"][2]) / 2, 30.0)


class Lens:
    """What a point looks like from here, in pixels of the frame, and how far along the axis."""

    def __init__(self, at, yaw, pitch, lens):
        y, p = math.radians(yaw), math.radians(pitch)
        self.at, self.lens, self.k = at, lens, 1920 * lens / 36
        self.f = (-math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p))
        self.r = (math.cos(y), math.sin(y), 0.0)
        r, f = self.r, self.f
        self.u = (r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0])

    def see(self, p):
        vx, vy, vz = p[0] - self.at[0], p[1] - self.at[1], p[2] - self.at[2]
        f, r, u = self.f, self.r, self.u
        d = vx * f[0] + vy * f[1] + vz * f[2]
        if d < .5:
            return None
        return 960 + self.k * (vx * r[0] + vy * r[1]) / d, 540 - self.k * (vx * u[0] + vy * u[1] + vz * u[2]) / d, d

    def target(self):
        return tuple(a + 40 * b for a, b in zip(self.at, self.f))


def blocks():
    """The rows of the city, seen from the camera's last place: x0, x1, y0, y1, floors. Near it, one at a time, by what each has to show from there; the rest by chance, lower than the eye, so that the horizon stays."""
    F = AT["K3"][0]
    out = []
    # on the left, close: the one whose east face holds the one window, and past its corner another, with a column you can read
    out.append((370.72 - 4 * BAY, 370.72, -94.14 - 7 * BAY, -94.14, 40))
    x1 = F[0] - 45.0
    y1 = F[1] - 4.0
    out.append((x1 - 4 * BAY, x1, y1 - 12 * BAY, y1, 46))
    # the left side of the street, from there on, coming down towards the far end
    rng = random.Random(201)
    d = 75.0
    floors = 44
    while d < FAR:
        n = rng.choice((6, 8, 10, 12))
        out.append((F[0] - d - n * BAY, F[0] - d, F[1] - 6.0 - rng.choice((8, 9, 10)) * BAY, F[1] - 6.0, max(floors, 24)))
        d += n * BAY + rng.uniform(16, 30)
        floors -= rng.randint(0, 3) if d < 500 else rng.randint(-1, 2)
    # the right side: steps down to the far end, below the eye, each showing its east face past the one before; over them the wall of the cut
    d = 40.0
    for lateral, wide, n, floors, gap in ((10.0, 4, 4, 36, 16.4), (7.5, 6, 6, 35, 19.6), (6.0, 6, 8, 37, 32.8), (5.0, 7, 10, 36, 36.0), (4.5, 8, 10, 34, 46.0)):
        out.append((F[0] - d - n * BAY, F[0] - d, F[1] + lateral, F[1] + lateral + wide * BAY, floors))
        d += n * BAY + gap
    while d < FAR:
        n, wide = rng.choice((8, 10, 12)), rng.choice((6, 8, 10))
        out.append((F[0] - d - n * BAY, F[0] - d, F[1] + 4.0, F[1] + 4.0 + wide * BAY, rng.randint(24, 34)))
        d += n * BAY + rng.uniform(26, 50)
    # the wall of the cut, and its row: lower to the west of it, and to the east
    out.append((0.0, WALL_X, 0.0, 61.2, CUT_FLOORS))
    x = -30.0
    while x > -FAR:
        n = rng.choice((14, 18, 24, 30))
        out.append((x - n * BAY, x, rng.uniform(-12, 4), 61.2, rng.randint(24, 36)))
        x -= n * BAY + rng.uniform(24, 36)
    x = WALL_X + 30.0
    while x < 900:
        n = rng.choice((14, 18, 24))
        out.append((x, x + n * BAY, 0.0, 61.2, rng.randint(36, 56)))
        x += n * BAY + rng.uniform(24, 36)
    # everything else, in rows either side, as far as the air lets you see
    for y in list(range(-860, -170, 88)) + list(range(92, 900, 88)):
        x = rng.uniform(380, 520)
        while x > -FAR:
            n, wide = rng.choice((10, 14, 18, 24)), rng.choice((5, 8, 12, 16))
            out.append((x - n * BAY, x, y, y + wide * BAY, rng.randint(14, 34)))
            x -= n * BAY + rng.uniform(22, 46)
    return out


class Lattice:
    """All the windows in sight, grouped by picture, and the walls they are on."""

    def __init__(self, lens, clear, way=None):
        """lens: where the wall of the cut is judged from. clear: the frame whose subtitle has to be left clear, or None. way: all the places the camera will be, if it is going to move."""
        with open(os.path.join(TEX, "manifest.json")) as f:
            info = json.load(f)
        self.sizes = info["windows"]
        self.light = [n for n in info["pool"] if self.sizes[n]["kind"] == "light"]
        self.dark = [n for n in info["pool"] if self.sizes[n]["kind"] == "dark"]
        self.lens, self.clear = lens, clear
        self.way, self.slack = way or [lens], 90.0 if way else 0.0
        self.pictures, self.edges = {}, dict(light=Build(), dark=Build())
        self.plain = dict(light=Build(), warm=Build(), dark=Build())
        self.halo, self.walls, self.roofs = Build(), Build(), Build()
        self.count = self.on_the_wall = 0

    def bays(self, T, a0, z0, tex, kind, seen, d, k=1.0):
        """One window, a0 along its wall, z0 up it. k: how far it has shrunk about its middle, on its way out."""
        ca, cz = a0 + WIN_W / 2, z0 + WIN_H / 2
        self.count += 1
        if d < HALO_FAR:        # the light it throws on the wall around it, from a little way out
            self.halo.add([T(a0 + .25, .7, z0 + .1), T(a0 + WIN_W - .25, .7, z0 + .1), T(a0 + WIN_W - .25, .7, z0 + WIN_H - .1), T(a0 + .25, .7, z0 + WIN_H - .1)], [(0, 1, 2, 3)])
        if seen < 6 and tex not in self.sizes:
            self.plain[kind].add([T(a0, CUT_REVEAL, z0), T(a0 + WIN_W, CUT_REVEAL, z0), T(a0 + WIN_W, CUT_REVEAL, z0 + WIN_H), T(a0, CUT_REVEAL, z0 + WIN_H)], [(0, 1, 2, 3)])
            return
        pic = self.pictures.setdefault(tex, Build())
        if seen < 18:
            pic.add([T(a0, CUT_REVEAL, z0), T(a0 + WIN_W, CUT_REVEAL, z0), T(a0 + WIN_W, CUT_REVEAL, z0 + WIN_H), T(a0, CUT_REVEAL, z0 + WIN_H)], [(0, 1, 2, 3)], [(0, 0), (1, 0), (1, 1), (0, 1)])
            return
        inset, ring = outline((WIN_W - .028) * k, (WIN_H - .028) * k, .05 * k, 3), outline(WIN_W * k, WIN_H * k, .064 * k, 3)
        m = len(inset)
        at = lambda a, b: T(ca + a, CUT_REVEAL, cz + b)
        pic.add([at(a, b) for a, b in inset], [list(range(m))], [(a / (WIN_W * k) + .5, b / (WIN_H * k) + .5) for a, b in inset])
        self.edges[kind if kind == "dark" else "light"].add([at(a, b) for a, b in inset] + [at(a, b) for a, b in ring], [[i, (i + 1) % m, m + (i + 1) % m, m + i][::-1] for i in range(m)])

    def sight(self, T, a0, z0, always):
        """How big the window comes to at its biggest and how near at its nearest, of all the places the camera will be. None if it is never in frame, or is only ever so edge-on that it is a line and would only shimmer."""
        best, near = None, 1e9
        middle, low, high = T(a0 + WIN_W / 2, 0, z0 + WIN_H / 2), T(a0, 0, z0), T(a0 + WIN_W, 0, z0 + WIN_H)
        for lens in self.way:
            s = lens.see(middle)
            if s is None:
                continue
            reach = lens.k * WIN_W / s[2] + self.slack
            if s[0] < -reach or s[0] > 1920 + reach or s[1] < -reach or s[1] > 1080 + reach:
                continue
            c0, c1 = lens.see(low), lens.see(high)
            if c0 is None or c1 is None or (abs(c1[0] - c0[0]) < 1.5 and not always):
                continue
            seen = max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1]) * 1.6)
            best = seen if best is None else max(best, seen)
            near = min(near, s[2])
        return None if best is None else (best, near)

    def face(self, T, length, floors, seed, special=None):
        """A wall's bays, each one lit or not by where it is, so that it stays as it is when the camera moves."""
        cols = int(length / BAY + 1e-6)
        off = (length - cols * BAY) / 2
        for c in range(cols):
            a0 = off + c * BAY + MARGIN
            for f in range(floors):
                z0 = f * STOREY + SILL
                tex = special.get((c, f)) if special else None
                if tex == "":
                    continue
                h = ((c * 73856093) ^ (f * 19349663) ^ (seed * 83492791)) & 0xffffffff
                h = ((h ^ (h >> 13)) * 1274126177) & 0xffffffff
                if tex is None and h / 2 ** 32 > LIT:
                    continue
                sight = self.sight(T, a0, z0, tex is not None)
                if sight is None:
                    continue
                seen, near = sight
                if self.clear and tex is None:
                    c0, c1 = self.clear.see(T(a0, 0, z0)), self.clear.see(T(a0 + WIN_W, 0, z0 + WIN_H))
                    if c0 and c1 and max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1]) * 1.6) >= 14:
                        x0, x1, y0, y1 = min(c0[0], c1[0]), max(c0[0], c1[0]), min(c0[1], c1[1]), max(c0[1], c1[1])
                        if x1 > SUBTITLE[0] and x0 < SUBTITLE[1] and y1 > SUBTITLE[2] and y0 < SUBTITLE[3]:
                            continue
                if tex is not None:
                    self.bays(T, a0, z0, tex, self.sizes[tex]["kind"], seen, near)
                    continue
                k = (h * 2654435761 & 0xffffffff) / 2 ** 32
                pool = self.dark if k < DARK else self.light
                kind = "dark" if pool is self.dark else "light"
                if seen < 6:
                    self.bays(T, a0, z0, "", "dark" if kind == "dark" else "warm" if k > .8 else "light", seen, near)
                else:
                    self.bays(T, a0, z0, pool[(h >> 7) % len(pool)] + shrunk(seen), kind, seen, near)

    def block(self, x0, x1, y0, y1, floors, seed, special=None):
        at = [lens.at for lens in self.way]
        H = floors * STOREY
        cuboid(self.walls, lambda a, b, c: (a, b, c), x0, x1, y0, y1, 0, H)
        if any(p[1] < y0 for p in at):
            self.face(lambda a, h, z: (x0 + a, y0 - h, z), x1 - x0, floors, seed * 4)
        if any(p[1] > y1 for p in at):
            self.face(lambda a, h, z: (x1 - a, y1 + h, z), x1 - x0, floors, seed * 4 + 1)
        if any(p[0] > x1 for p in at):
            self.face(lambda a, h, z: (x1 + h, y0 + a, z), y1 - y0, floors, seed * 4 + 2, special)
        if any(p[0] < x0 for p in at):
            self.face(lambda a, h, z: (x0 - h, y1 - a, z), y1 - y0, floors, seed * 4 + 3)

    def pictures_of_the_wall(self):
        """Which picture in which window of the wall of the cut: the same ones whichever way you look at it."""
        pool, rng, deck, out = sorted(self.light + self.dark), random.Random(92), [], {}
        known = dict(room_a="ask_a", room_b="ask_b", room_c="ask_c")
        for at, pic in cut_plan().items():
            if pic in known:
                out[at] = known[pic]
            else:
                if not deck:
                    deck = rng.sample(pool, len(pool))
                out[at] = deck.pop()
        return out

    def the_wall(self, as_cut):
        """The wall of the cut, the same windows in the same places with the same pictures in them as C3b, whichever way you look at it."""
        T = lambda a, h, z: (a, .12 - h, z)        # back from the wall's face by as much as they were in the wall
        cuboid(self.walls, lambda a, b, c: (a, b, c), 0, WALL_X, .12, 61.2, 0, CUT_FLOORS * STOREY)
        for (c, f), tex in self.pictures_of_the_wall().items():
            a0, z0 = c * BAY + MARGIN, f * STOREY + SILL
            c0, c1 = self.lens.see(T(a0, CUT_REVEAL, z0)), self.lens.see(T(a0 + WIN_W, CUT_REVEAL, z0 + WIN_H))
            if c0 is None or c1 is None:
                continue
            seen = seen_width(c, f) if as_cut else max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1]) * 1.6)
            self.on_the_wall += 1
            self.bays(T, a0, z0, tex + shrunk(seen), self.sizes[tex]["kind"], max(seen, 18.0) if as_cut else seen, c0[2])

    def done(self, gain=GAIN, prefix="lit_"):
        for name, b in self.pictures.items():
            m = lit_window(name, os.path.join(TEX, name + ".png"), gain * LIFT.get(name, 1.0), 0.0, "_m" not in name)
            m.cycles.emission_sampling = "NONE"
            b.object(prefix + name, m)
        for key, level, colour in (("light", .82, (1.0, .97, .92)), ("dark", .17, (1.0, .97, .92))):
            self.edges[key].object(prefix + "edge_" + key, glow("edge_" + key, colour, level * gain / GAIN))
        for key, level, colour in (("light", .80, (.93, .91, .87)), ("warm", .62, (1.0, .82, .60)), ("dark", .12, (1.0, .97, .92))):
            self.plain[key].object(prefix + "plain_" + key, glow("plain_" + key, colour, level * gain / GAIN))
        self.walls.object("walls", surface("night_wall", (.085, .070, .055), .85, 0, .2))
        h = self.halo.object("halo_near", halo_material((1.0, .80, .58), HALO))
        h.visible_camera = h.visible_glossy = h.visible_transmission = h.visible_shadow = False
        h["halo"] = 1


def the_ground():
    g = Build()
    g.add([(-3000, -3000, 0), (3000, -3000, 0), (3000, 3000, 0), (-3000, 3000, 0)], [(0, 1, 2, 3)])
    g.object("ground", surface("ground", (.03, .025, .02), .7))


def hand_of_windows(lens):
    """K2's windows, as near as your hand, out of the old set. Their place is the same to the camera as it ever was; what is behind them is now the street."""
    with open(os.path.join(TEX, "manifest.json")) as f:
        sizes = json.load(f)["windows"]
    wins = [w for w in w3.hand(sizes) if w["tex"] in ("hero", "ask_a", "ask_b", "ask_c", "k2_new")]
    screens = {w["tex"]: Build() for w in wins}
    rims, backs = dict(light=Build(), dark=Build()), dict(light=Build(), dark=Build())
    for w in wins:
        w["p"] = w3.place_at(w, w3.D2)
        w3.panel(w, sizes[w["tex"]], screens, rims, backs)
    M = Matrix.Translation((lens.at[0] - w3.D2, lens.at[1], lens.at[2])) @ Matrix.Rotation(math.pi / 2, 4, "Z")
    made = []
    for name, b in screens.items():
        m = lit_window(name, os.path.join(TEX, name + ".png"), GAIN, 0.0, True)
        m.cycles.emission_sampling = "NONE"
        made.append(b.object("lit_hand_" + name, m))
    for key, level in (("light", 1.0), ("dark", .3)):
        made.append(rims[key].object("lit_hand_rim_" + key, glow("rim_" + key, (1.0, .97, .92), level)))
        made.append(backs[key].object("hand_back_" + key, surface("back_" + key, (.02, .018, .016), .6)))
    for o in made:
        o.matrix_world = M


def the_city(city, key=None):
    """Every block but the wall of the cut, which city.the_wall has already put up."""
    row = blocks()
    wall = next(i for i, b in enumerate(row) if b[4] == CUT_FLOORS)
    for i, b in enumerate(row):
        if i == wall:       # the rest of the wall of the cut, which has only the one face you know
            city.face(lambda a, h, z: (WALL_X + h, a, z), 61.2, CUT_FLOORS, 7)
            continue
        special = {(6, 39): "ask_a", (6, 38): "hero_k3"} if i == 0 else None
        if i == 0 and key in ("K4a", "K4b"):
            special.update({(K4_SHUT, K4_FLOOR): "", (K4_NEW, K4_FLOOR): ""})
        city.block(*b, i + 11, special)
    if key in ("K4a", "K4b"):
        x1, y0 = row[0][1], row[0][2]
        T = lambda a, h, z: (x1 + h, y0 + a, z)
        z0 = K4_FLOOR * STOREY + SILL
        if key == "K4a":
            city.bays(T, K4_SHUT * BAY + MARGIN, z0, "k4_closing", "light", 900.0, 8.0, K4_SHRINK)
        city.bays(T, K4_NEW * BAY + MARGIN, z0, "k4_typing" if key == "K4a" else "k4_sent", "light", 900.0, 8.0)
    the_ground()


def main():
    key, pct, samples = ARGS[0], opt("pct", 100), opt("samples", 128)
    os.makedirs(PLATES, exist_ok=True)
    t0 = time.time()
    sc = set_up(samples, pct)
    at, yaw, pitch, focal = AT[key]
    lens = Lens(at, yaw, pitch, focal)
    sky(sc, 0, (.0014, .0011, .0009), SMOG)
    city = Lattice(lens, lens if key != "C3b" else None)
    city.the_wall(key == "C3b")
    if key != "C3b":
        the_city(city, key)
    city.done()
    print(city.count, "windows", len(city.pictures), "pictures, of the wall", city.on_the_wall, flush=True)
    air(SMOG_LEN, -1400.0, 1400.0, 2000.0, 140.0, SMOG, SMOG_TINT, SMOG_SCATTER)
    if key == "K2":
        hand_of_windows(lens)
    lens_on(sc, at, lens.target(), focal)
    if key == "K2":
        d = sc.camera.data.dof
        d.use_dof, d.focus_distance, d.aperture_fstop = True, 2.6, 16.0
        sc.camera.data.clip_start = .02
    print("built", round(time.time() - t0, 1), flush=True)
    plate(sc, os.path.join(PLATES, key + "_room"))
    print(key, "room", round(time.time() - t0, 1), flush=True)
    t1 = time.time()
    only_the_windows(sc)
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.samples = max(samples, 64)
    plate(sc, os.path.join(PLATES, key + "_words"))
    print(key, "words", round(time.time() - t1, 1), flush=True)
    if key == "C3b":
        mask_plate(sc, os.path.join(PLATES, key + "_mask"), ("lit_",))


if __name__ == "__main__":
    main()
