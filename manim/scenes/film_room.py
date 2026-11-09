"""Shots 03 to 06 of the bridge, put together from what film_room3d.py rendered: daylight and sun, the screen's light on the room, and the screen itself laid in where the camera sees it. The room is added up in linear light and turned into a picture; a screen is already light for the eye, so it goes in after that and keeps its white and its black.

    python film_room.py still T [T ...] [pct=100]                      # ink/film_out/film/room/keys/room_T.png
    python film_room.py seq [A B] [pct=50]                             # ink/film_out/film/room/seq3/room_NNNN.png, one for each frame from 93.4 to 107.0, the first numbered 0

The cut is hard: the room up to 96.4, then the frame the film opened on, the whole screen. From the first word the camera goes slowly in on where the words start.
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

import film_room_ui as ui
import ink_subs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "room")
PLATES = os.path.join(OUT, "plates")
FPS = 30
T0, T1, CUT = 90.4, 107.0, 96.4  # T0: where the air and the cloud started, as when the room ran six seconds. The room is now its last three
START = 93.4
POSES = 7                        # how many positions of the curtain were rendered, a second apart
CLOUD = .07                      # how much of the sun a thin cloud takes away over the room's six seconds: nothing happens in the room but this and the curtain

# the balance of the light in the room. glass: how much of the room the screen's glass gives back, white: how bright a white is on the screen, of what the picture can show
ROOM = dict(sky=1.5, sun=.7, glow=1.4, exposure=.96, glass=.035, white=1.0)
# close up: the screen fills the frame as the opening did. push: from when, to when, how much bigger; on: the point of the screen that stays where it is, in points, where the reply's text starts: the question may go out of the frame, the reply never
CLOSE = dict(sky=1.5, sun=.7, exposure=.96, glass=.022, push=(ui.REPLY, T1, 1.30), on=(ui.ring.WIN_X + ui.ring.WIN_W / 2 - (ui.page.BOX_W / 2 - ui.page.BOX_LEFT), ui.ring.WIN_Y + ui.WAIT_Y))


def sm(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def srgb_to_linear(a):
    return np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055) ** 2.4)


def linear_to_srgb(a):
    a = np.clip(a, 0, 1)
    return np.where(a <= .0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - .055)


def film(x):
    """Scene light to display light: a soft shoulder, so that the window burns out gently."""
    a, b, c, d, e = 2.51, .03, 2.43, .59, .14
    x = np.maximum(x, 0)
    return np.clip(x * (a * x + b) / (x * (c * x + d) + e), 0, 1)


def homography(src, dst):
    """The matrix taking the four src points to the four dst points."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append((x, y, 1, 0, 0, 0, -u * x, -u * y, -u))
        A.append((0, 0, 0, x, y, 1, -v * x, -v * y, -v))
    return np.linalg.svd(np.array(A, float))[2][-1].reshape(3, 3)


def load(name, w, h):
    a = np.load(os.path.join(PLATES, name + ".npy")).astype(np.float32)
    if a.shape[0] != h:
        a = np.stack([np.asarray(Image.fromarray(a[..., i]).resize((w, h), Image.BILINEAR)) for i in range(a.shape[2])], -1)
    return a


def plate(path_name):
    return os.path.exists(os.path.join(PLATES, path_name + ".npy"))


class Room:
    """The room from the shot's fixed camera."""

    def __init__(self, scale):
        with open(os.path.join(PLATES, "wide.json")) as f:
            self.j = json.load(f)
        self.scale = scale
        self.W, self.H = int(1920 * scale), int(1080 * scale)
        self.layers = {p: load(f"wide_{p}", self.W, self.H) for p in ["sky", "sun", "glow", "mask"] + [f"{q}_p{k}" for k in range(1, POSES) for q in ("sky", "sun")] if plate(f"wide_{p}")}
        k = self.W / self.j["width"]
        self.corners = np.array(self.j["corners"], float) * k
        self.pictures = {}
        self.xy = None

    def pose(self, what, x):
        """A light pass with the curtain where the air has it after x seconds. The steps are near enough, and soft enough, to pass from one to the next by adding."""
        at = lambda k: self.layers[f"{what}_p{k}" if k else what]
        if f"{what}_p1" not in self.layers:
            return self.layers[what]
        k = min(int(x), POSES - 2)
        u = float(np.clip(x - k, 0, 1))
        return at(k) * (1 - u) + at(k + 1) * u

    def setup(self, sw, sh):
        """Where each pixel of the frame comes from on a picture of the screen sw by sh."""
        Hm = homography(self.corners, np.array(((0, 0), (sw, 0), (sw, sh), (0, sh)), float))
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        self.xy = []
        for dy in (-.25, .25):
            for dx in (-.25, .25):
                X, Y = xx + .5 + dx, yy + .5 + dy
                d = Hm[2, 0] * X + Hm[2, 1] * Y + Hm[2, 2]
                self.xy.append(((Hm[0, 0] * X + Hm[0, 1] * Y + Hm[0, 2]) / d, (Hm[1, 0] * X + Hm[1, 1] * Y + Hm[1, 2]) / d))

    def screen(self, key):
        """The picture on the screen at key, as display light, seen from the camera."""
        if key in self.pictures:
            return self.pictures[key]
        pic = ui.picture(key)
        sh, sw = pic.shape[:2]
        if self.xy is None:
            self.setup(sw, sh)
        # the picture is brought down near what the frame will show of it, so the four samples are enough
        span = np.hypot(*(self.corners[1] - self.corners[0]))
        shrink = max(sw / span / 2.0, 1.0)
        small = np.asarray(Image.fromarray(pic).resize((int(sw / shrink), int(sh / shrink)), Image.LANCZOS), np.float32) / 255
        small = srgb_to_linear(small)
        acc = 0
        for X, Y in self.xy:
            acc = acc + np.stack([map_coordinates(small[..., i], [Y / shrink - .5, X / shrink - .5], order=1, mode="nearest") for i in range(3)], -1)
        if len(self.pictures) > 8:
            self.pictures.clear()
        self.pictures[key] = acc / 4
        return self.pictures[key]

    def frame(self, key, t):
        L = ROOM
        x = t - T0
        sun = L["sun"] * (1 - CLOUD * (.5 - .5 * math.cos(math.pi * min(x / (CUT - T0), 1))))
        base = self.pose("sky", x) * L["sky"] + self.pose("sun", x) * sun
        m = self.layers["mask"][..., :1]
        scene = base + self.layers["glow"] * L["glow"]
        y = room_tone(film(scene * L["exposure"]))
        shown = self.screen(key) * L["white"]
        back = film(base * L["exposure"]) * L["glass"]
        return y * (1 - m) + (shown + back - shown * back) * m


def room_tone(y):
    """The room's colour: a little less saturated than the camera's, a little warmth in its shadows."""
    lum = y.mean(-1, keepdims=True)
    y = lum + (y - lum) * .94
    return y + (1 - y) * np.array((.0010, .0006, 0), np.float32)


def bilinear(pic, px, py):
    """pic at every crossing of the columns px and the rows py."""
    ix, iy = np.clip(np.floor(px).astype(np.int32), 0, pic.shape[1] - 2), np.clip(np.floor(py).astype(np.int32), 0, pic.shape[0] - 2)
    fx, fy = (px - ix).astype(np.float32)[None, :, None], (py - iy).astype(np.float32)[:, None, None]
    top = pic[np.ix_(iy, ix)] * (1 - fx) + pic[np.ix_(iy, ix + 1)] * fx
    low = pic[np.ix_(iy + 1, ix)] * (1 - fx) + pic[np.ix_(iy + 1, ix + 1)] * fx
    return top * (1 - fy) + low * fy


class Close:
    """The screen square on: what is on it, and very faintly what its glass gives back of the room."""

    def __init__(self, scale):
        self.scale = scale
        self.W, self.H = int(1920 * scale), int(1080 * scale)
        L = CLOSE
        room = load("close_sky", self.W, self.H) * L["sky"] + load("close_sun", self.W, self.H) * L["sun"]
        self.back = film(gaussian_filter(room, (40 * scale, 40 * scale, 0)) * L["exposure"]) * L["glass"]
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        r = np.hypot((xx - self.W / 2) / self.W, (yy - self.H / 2) / self.W)
        self.falloff = (1 - .10 * r * r)[..., None]
        self.pictures = {}

    def picture(self, key):
        if key not in self.pictures:
            if len(self.pictures) > 8:
                self.pictures.clear()
            self.pictures[key] = srgb_to_linear(ui.picture(key).astype(np.float32) / 255).astype(np.float32)
        return self.pictures[key]

    def frame(self, key, t):
        pic = self.picture(key)
        L = CLOSE
        # an even pace in scale, starting and stopping with no easing
        a, b, most = L["push"]
        push = most ** float(np.clip((t - a) / (b - a), 0, 1))
        k = ui.ring.K * self.scale
        z = pic.shape[1] / ui.PW
        # each frame pixel from two by two samples of the panel, across and down
        fx, fy = L["on"]
        ax = fx + ((np.arange(2 * self.W) + .5) / 2 - fx * k) / (k * push)
        ay = fy + ((np.arange(2 * self.H) + .5) / 2 - fy * k) / (k * push)
        y = bilinear(pic, ax * z - .5, ay * z - .5)
        y = (y[0::2, 0::2] + y[1::2, 0::2] + y[0::2, 1::2] + y[1::2, 1::2]) / 4
        y = y * self.falloff
        return y + self.back - y * self.back


def finish(y, seed, grain):
    """Display light to the finished picture, with the grain any camera has."""
    p = linear_to_srgb(y)
    lum = p.mean(-1, keepdims=True)
    r = np.random.default_rng(seed)
    n = r.normal(0, 1, p.shape[:2]).astype(np.float32)
    if p.shape[0] >= 1000:
        n = gaussian_filter(n, .55) * 1.5
    p = p + n[..., None] * (grain * (1.2 - lum) + .0016)         # never less than enough to break up the steps of a very even white
    return np.uint8(np.clip(p, 0, 1) * 255 + .5)


SHOTS = {}


def shot(kind, scale):
    if kind not in SHOTS:
        SHOTS[kind] = kind(scale)
    return SHOTS[kind]


def at(t, scale=1.0):
    if t < CUT:
        f = finish(shot(Room, scale).frame(ui.state(t), t), int(t * FPS) + 7, .0065)
    else:
        f = finish(shot(Close, scale).frame(ui.state(t), t), int(t * FPS) + 7, .0035)
    full = f if scale == 1.0 else np.asarray(Image.fromarray(f).resize((1920, 1080), Image.LANCZOS))
    if t < CUT:
        # close up the screen says the subtitle's words itself, and the subtitle would lie on its box
        full = ink_subs.apply(full, t)
    if scale != 1.0:
        return np.asarray(Image.fromarray(full).resize((int(1920 * scale), int(1080 * scale)), Image.LANCZOS))
    return full


def main():
    args = [a for a in sys.argv[2:] if "=" not in a]
    opts = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
    what = sys.argv[1]
    if what == "still":
        scale = float(opts.get("pct", 100)) / 100
        out = os.path.join(OUT, "keys")
        os.makedirs(out, exist_ok=True)
        for a in args:
            t = float(a)
            Image.fromarray(at(t, scale)).save(os.path.join(out, f"room_{a}.png"))
        print(out)
    elif what == "seq":
        scale = float(opts.get("pct", 50)) / 100
        a, b = (float(args[0]), float(args[1])) if len(args) == 2 else (START, T1)
        out = os.path.join(OUT, "seq3" if scale < 1 else "seq_full")
        os.makedirs(out, exist_ok=True)
        for i in range(int(round((a - START) * FPS)), int(round((b - START) * FPS))):
            t = START + i / FPS
            path = os.path.join(out, f"room_{i:04d}.png")
            Image.fromarray(at(t, scale)).save(path, compress_level=1)
            if i % 30 == 0:
                print(i, flush=True)
        print(out)
    elif what == "zoom":
        # a corner rendered through a longer lens, to look at closely: python film_room.py zoom zoom_studio zoom_books
        out = os.path.join(OUT, "keys")
        for a in args:
            y = sum(load(f"{a}_{p}", 1920, 1080) * ROOM[p] for p in ("sky", "sun", "glow"))
            Image.fromarray(finish(room_tone(film(y * ROOM["exposure"])), 7, .002)).save(os.path.join(out, f"{a}.png"))
        print(out)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
