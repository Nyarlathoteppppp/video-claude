"""The ghost of 鬼夜哭: a hazy figure left in the ink. Head, shoulders, one long arm; the lower body goes into the wash. No ribs, no joints, no fingers, no face, only, at the end, a mouth that is a dark hole.

It is a distance field of a few soft limbs, so any pose between hanging and howling is one number away (Ghost(lift)), and a later slow head-lift is a slow change of that number. The ink does the rest.

    python film_myth_ghost.py shape 0.2 1.0     # the figures alone, flat, to judge the pose: goes to keys/ghost_shape.png
"""
import sys

import numpy as np
from PIL import Image

from ink_grade import fft_noise, sm

W, H = 1920, 1080
_fields = {}


def _fft(h, w, beta, seed):
    """Noise fields are slow to make and a moving figure asks for the same few on every frame."""
    key = (h, w, beta, seed)
    if key not in _fields:
        _fields[key] = fft_noise(h, w, beta, seed)
    return _fields[key]


# lift 0: head hanging, 1: thrown back, howling. Everything between is interpolated, in film pixels, for the figure in the frame k3_look and k4_cry share.
# Each entry is one joint at (hanging, starting to lift, howling).
LIFTS = (0.0, 0.22, 1.0)
POSE = dict(
    hip=((1210, 990), (1210, 990), (1216, 990)),        # the body narrows and bends away like smoke, it has no legs to read
    waist=((1104, 780), (1104, 780), (1120, 790)),
    chest=((1060, 660), (1060, 650), (1056, 624)),
    neck=((950, 512), (950, 496), (1040, 506)),         # the base of the neck
    skull=((838, 490), (836, 422), (1098, 330)),        # the centre of the head
    tilt=(-48, -20, 20),                                # degrees the face is above the horizontal. At the howl the long head stands nearly upright, in line with the neck, thrown back only a little
    open=(0.0, 0.0, 1.0),                               # how far the jaw has dropped
    shoulder=((962, 580), (962, 568), (996, 552)),
    elbow=((820, 640), (820, 630), (836, 616)),         # the long arm hangs forward, reaching down at what it sees; when the howl comes it has let go
    wrist=((664, 716), (664, 708), (690, 712)),
    tip=((590, 812), (590, 804), (640, 806)),
    back=((1056, 576), (1056, 566), (1104, 534)),       # the top of the bent back, only a curve in the outline
)

# form="shroud": the one figure of the close shots, the outline of the middle ghost of k4_wide made large. Round head, a slope of shoulder, wider and wider down to a hem that goes into the ink. Only the head and the sleeve move: at 0.22 the head is up and forward, looking; at 1 it has gone back, tilted to the sky, and the mouth opens.
SHROUD = dict(
    head=(48, 66),
    skull=((1036, 376), (1062, 340), (1098, 330)),      # at 0 it hangs, forward and down
    tilt=(-14, 12, 26),
    sleeve=(((1024, 468), (944, 500), (880, 546)), ((1024, 468), (944, 500), (880, 546)), ((1030, 470), (952, 512), (892, 566))),
)


def _at(lift, name):
    a = np.array(POSE[name], np.float64)
    if a.ndim == 1:
        return float(np.interp(lift, LIFTS, a))
    return np.array([np.interp(lift, LIFTS, a[:, i]) for i in range(2)])


def _seg(xx, yy, p, q, r0, r1):
    """The distance to a limb that goes from radius r0 at p to r1 at q."""
    px, py, qx, qy = p[0], p[1], q[0], q[1]
    dx, dy = qx - px, qy - py
    t = np.clip(((xx - px) * dx + (yy - py) * dy) / (dx * dx + dy * dy + 1e-6), 0, 1)
    return np.hypot(xx - px - t * dx, yy - py - t * dy) - (r0 + (r1 - r0) * t)


def _egg(xx, yy, c, ax, rx, ry):
    """A soft ellipse with its long axis ry along the unit vector ax. Close to a distance near its edge."""
    u = (xx - c[0]) * ax[0] + (yy - c[1]) * ax[1]
    v = -(xx - c[0]) * ax[1] + (yy - c[1]) * ax[0]
    q = np.hypot(u / ry, v / rx)
    return (q - 1) * np.minimum(rx, ry) * (0.5 + 0.5 * np.minimum(q, 1.5))


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


class Ghost:
    """One figure. scale and shift move it: scale is about the point at, in film pixels. sdf is negative inside."""

    def __init__(self, lift, w=W, h=H, scale=1.0, at=(0, 0), to=None, seed=3, flip=False, arm=1.0, lean=0.0, opens=True, form="full", behind=None, head=(64, 70), drift=0.0):
        self.lift, self.w, self.h, self.scale = lift, w, h, scale
        self.drift = drift      # a slow angle: the wander of every edge turns from one field of noise to another, so a figure that holds still is still not a printed shape. 0 is the key frames'
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        to = at if to is None else to
        self.flip = -1.0 if flip else 1.0
        self.arm, self.opens, self.form = arm, opens, form
        self.behind = behind      # y of the ridge it stands behind, one value per film column: nothing of it shows below that
        self.rot = np.array(((np.cos(lean), -np.sin(lean)), (np.sin(lean), np.cos(lean))))
        self.map = lambda p: self.rot @ ((np.array(p, np.float64) - np.array(at)) * scale * np.array((self.flip, 1.0))) + np.array(to)
        wob = 12.0 * scale + 1.0 if form == "phantom" else 9.0 * scale      # in mist the outline wanders further than it does close to
        xx = xx + wob * self._n(3.6, seed + 1)
        yy = yy + wob * self._n(3.6, seed + 2)
        self.head = SHROUD["head"] if form == "shroud" else head
        self.xx, self.yy = xx, yy
        self.parts = self._parts(lift)
        self.sdf = self._sdf()
        self.sdf_hair = self._hair()

    def _n(self, beta, seed):
        a = _fft(self.h, self.w, beta, seed)
        if not self.drift:
            return a
        return np.cos(self.drift) * a + np.sin(self.drift) * _fft(self.h, self.w, beta, seed + 5003)

    def _parts(self, lift):
        P = {k: self.map(_at(lift, k)) for k in ("hip", "waist", "chest", "neck", "skull", "shoulder", "elbow", "wrist", "tip", "back")}
        t = np.radians(_at(lift, "tilt"))
        if self.form == "shroud":
            P["skull"] = self.map([np.interp(lift, LIFTS, [p[i] for p in SHROUD["skull"]]) for i in range(2)])
            t = np.radians(np.interp(lift, LIFTS, SHROUD["tilt"]))
        P["face"] = self.rot @ np.array((-np.cos(t) * self.flip, -np.sin(t)))
        P["chin"] = self.rot @ np.array((-np.sin(t) * self.flip, np.cos(t)))
        P["open"] = _at(lift, "open")
        return P

    def _phantom_sdf(self):
        """Seen from far off, in mist: a piece of paper left in the ink. A short head with no neck, the body going down from it wider and wider, one short broad sleeve hanging forward."""
        P, s, xx, yy = self.parts, self.scale, self.xx, self.yy
        head = _egg(xx, yy, P["skull"], P["chin"], self.head[0] * s, self.head[1] * s)
        waist, foot = self.map((1072, 560)), self.map((1128, 900))
        d = _seg(xx, yy, P["skull"] + P["chin"] * 30 * s, waist, 62 * s, 96 * s)
        d = _smin(d, _seg(xx, yy, waist, foot, 96 * s, 150 * s), 20 * s)
        d = _smin(d, head, 28 * s)
        a, b, c = self.map((1030, 470)), self.map((952, 512)), self.map((892, 566))
        sleeve = _smin(_seg(xx, yy, a, b, 64 * s, 56 * s), _seg(xx, yy, b, c, 56 * s, 38 * s), 10 * s)
        self.sleeve_ends = a, c
        self.sdf_arm = sleeve
        self.sdf_keep = np.minimum(head, sleeve)
        return _smin(d, sleeve, 26 * s)

    def _shroud_sdf(self):
        """The far ghost's outline, near: what shows of the phantom is about this much inside its field, since mist eats its edge. The head goes down into the body like a drop, the top of the sleeve carries on the slope of the shoulder, and below the sleeve it keeps widening on both sides until it is lost."""
        P, s, xx, yy = self.parts, self.scale, self.xx, self.yy
        head = _egg(xx, yy, P["skull"], P["chin"], self.head[0] * s, self.head[1] * s)
        waist, foot = self.map((1072, 560)), self.map((1110, 900))
        d = _seg(xx, yy, P["skull"] + P["chin"] * 30 * s, waist, 50 * s, 84 * s)
        d = _smin(d, _seg(xx, yy, waist, foot, 84 * s, 190 * s), 20 * s)
        d = _smin(d, head, 28 * s)
        self.sdf_torso = d
        a, b, c = [self.map([np.interp(self.lift, LIFTS, [pose[j][i] for pose in SHROUD["sleeve"]]) for i in range(2)]) for j in range(3)]
        sleeve = _smin(_seg(xx, yy, a, b, 52 * s, 44 * s), _seg(xx, yy, b, c, 44 * s, 26 * s), 10 * s)
        self.sleeve_ends = a, c
        self.sdf_arm = sleeve
        self.sdf_keep = np.minimum(head, sleeve + 10 * s)
        return _smin(d, sleeve, 26 * s)

    def _sdf(self):
        if self.form == "phantom":
            return self._phantom_sdf()
        if self.form == "shroud":
            return self._shroud_sdf()
        P, s, xx, yy = self.parts, self.scale, self.xx, self.yy
        k = 16 * s
        d = _seg(xx, yy, P["hip"], P["waist"], 5 * s, 25 * s)
        d = _smin(d, _seg(xx, yy, P["waist"], P["chest"], 25 * s, 36 * s), k)
        d = _smin(d, _seg(xx, yy, P["chest"], P["back"], 32 * s, 20 * s), k)
        d = _smin(d, _seg(xx, yy, P["chest"], P["shoulder"], 32 * s, 18 * s), k)
        d = _smin(d, _seg(xx, yy, P["shoulder"], P["back"], 18 * s, 20 * s), k)
        # the middle of the shoulders, filled
        mid = (P["shoulder"] + P["back"] + P["chest"]) / 3
        d = _smin(d, np.hypot(xx - mid[0], yy - mid[1]) - 36 * s, k)
        # the neck rises out of the shoulders
        o = P["open"]
        d = _smin(d, _seg(xx, yy, P["shoulder"], P["neck"], 20 * s, (23 + 12 * o) * s), (14 + 14 * o) * s)
        d = _smin(d, _seg(xx, yy, P["back"], P["neck"], 20 * s, (23 + 12 * o) * s), (14 + 14 * o) * s)
        # the arm, thin and very long
        if self.arm > 0:
            along = lambda p: P["shoulder"] + (P[p] - P["shoulder"]) * self.arm
            arm = _seg(xx, yy, P["shoulder"], along("elbow"), 17 * s, 14 * s)
            arm = _smin(arm, _seg(xx, yy, along("elbow"), along("wrist"), 14 * s, 11 * s), 8 * s)
            arm = _smin(arm, _seg(xx, yy, along("wrist"), along("tip"), 12 * s, 3 * s), 6 * s)
            d = _smin(d, arm, 8 * s)
        else:
            arm = np.full_like(xx, 1e4)
        self.sdf_keep = arm
        self.sdf_arm = arm
        # the head: an egg, longer as the jaw drops, the neck going into it low and at the back
        head = _egg(xx, yy, P["skull"], P["chin"], (46 + 2 * o) * s, (56 + 20 * o) * s)
        jaw = _egg(xx, yy, P["skull"] + P["chin"] * (42 + 2 * o) * s + P["face"] * (14 - 14 * o) * s, P["chin"], (29 + 3 * o) * s, (32 + 0 * o) * s)
        head = _smin(head, jaw, 14 * s)
        self.sdf_keep = np.minimum(self.sdf_keep, head)
        # at the howl the neck is as wide as a long drop's waist, on the axis of the head, flowing into it and into the shoulders
        into = P["skull"] + P["chin"] * (26 - 16 * o) * s - P["face"] * (12 - 8 * o) * s
        d = _smin(d, _seg(xx, yy, P["neck"], into, (25 + 10 * o) * s, (21 + 9 * o) * s), (10 + 16 * o) * s)
        d = _smin(d, head, (8 + 18 * o) * s)
        return d

    def _hair(self):
        """A few long tapering locks hanging from the back of the head, straight down whatever the head does. They stay crisp where the body goes soft: pale lines in the ink, the only lines the figure has."""
        P, s, xx, yy = self.parts, self.scale, self.xx, self.yy
        o = P["open"]
        if self.form != "full":      # hair lifted out of the ink reads as scratches at any width tried, so this figure has none, as the far ghost has none
            return np.full_like(xx, 1e6)
        r = np.random.default_rng(29)
        root = P["skull"] - P["face"] * (38 * o - 20 * (1 - o)) * s + P["chin"] * 26 * (1 - o) * s
        d = np.full_like(xx, 1e6)
        for i in range(7):
            a = r.uniform(-0.3, 1)
            start = root + P["chin"] * a * 44 * s
            n = r.uniform(50, 170) * (1 + 0.7 * o) * s
            sway = r.uniform(-16, 34) * s
            pts = [start]
            for u in (0.33, 0.66, 1.0):
                pts.append(start + np.array((sway * u * u + r.uniform(-7, 7) * s * u, n * u)))
            w = r.uniform(3.0, 6.0) * s
            for q in range(3):
                d = np.minimum(d, _seg(xx, yy, pts[q], pts[q + 1], w * (1 - q / 3) + 0.5 * s, w * (1 - (q + 1) / 3) + 0.5 * s))
        return d

    def mouth_place(self):
        """Where the near figure's mouth is and which way it stands: on the axis of the head as it shows, in its lower third, whether or not it is open yet. It stays with the head as the outline wanders there, but an oval this size on the wandering grid gets a waist, so it is drawn on the straight one. Half way back to upright: slanted with the head it reads as the eye of something seen from the side."""
        P, s = self.parts, self.scale
        c = P["skull"] + P["chin"] * 52 * s - P["face"] * 10 * s
        ax = P["chin"] + np.array((0.0, 1.0))
        ix, iy = int(c[0]), int(c[1])
        return c - np.array((self.xx[iy, ix] - ix, self.yy[iy, ix] - iy)), ax / np.hypot(*ax)

    def mouth(self):
        """The hole, 0 to 1: nothing until the jaw starts to drop, and nothing on one that does not open. An upright oval on the long axis of the head, in its lower third, paper all round it: it can only be a mouth. Same wavering edge as everything else."""
        P, s = self.parts, self.scale
        o = P["open"]
        if o <= 0 or not self.opens:
            return np.zeros_like(self.xx)
        if self.form == "shroud":
            c, ax = self.mouth_place()
            yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
            e = _egg(xx, yy, c, ax, 12.4 * o * s, 29 * o * s)
            return sm(-e / (3.0 * s))
        far = self.form == "phantom"
        # on the axis of the head as it shows: the hair behind it carries the head's right edge out past the line of the skull, by this much
        c = P["skull"] + P["chin"] * (44 if far else 48) * o * s - P["face"] * (0 if far else 9) * o * s
        e = _egg(self.xx, self.yy, c, P["chin"], (9 if far else 9.5) * o * s, (24 if far else 20) * o * s)
        return sm(-e / (3.0 * s))

    def body(self, soft=4.0):
        return sm(-self.sdf / (soft * self.scale) + 0.5)

    def fade(self, y0=None, y1=None, seed=9, amount=70.0):
        """The lower body going: 1 above y0, 0 below y1, on an edge that wanders."""
        s = self.scale
        hip = self.parts["hip"]
        top, bottom = (430, 130) if self.form == "shroud" else (450, 240)      # the near figure's hem takes much longer to go, close to you can watch it going
        amount = 105.0 if self.form == "shroud" else amount
        y0 = hip[1] - top * s if y0 is None else y0
        y1 = hip[1] - bottom * s if y1 is None else y1
        n = self._n(3.0, seed)
        yy = np.mgrid[0:self.h, 0:self.w][0].astype(np.float32)
        out = 1 - sm((yy + amount * s * n - y0) / (y1 - y0))
        if self.form == "shroud":
            out = out * (1 - sm((yy - (y1 - 10 * s)) / (60 * s)))      # however far the edge wanders, none of it lower than the hem
        return out

    def fade_all(self, seed=9):
        """As fade(), but the long arm holds out longer than the body before it goes."""
        s = self.scale
        hip = self.parts["hip"]
        arm = self.fade(y0=hip[1] - 300 * s, y1=hip[1] - 70 * s, seed=seed + 40, amount=40.0)
        if self.form in ("phantom", "shroud"):
            # no line where the sleeve lies over the body, and it goes out into the mist before it reaches the ridge: a sleeve, not a leg standing on the hill
            a, c = self.sleeve_ends
            along = ((self.xx - a[0]) * (c[0] - a[0]) + (self.yy - a[1]) * (c[1] - a[1])) / ((c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2)
            arm = arm * (1 - sm((along - 0.35) / 0.75)) * sm(-self.sdf_arm / (34 * s))
            out = np.maximum(self.fade(seed=seed), arm)
            if self.form == "shroud":
                # the hem is far below the sleeve here, so it has to be told to go: out past the body it thins towards its tip, on a wandering edge
                loose = sm(self.sdf_torso / (30 * s))
                tip = 1 - sm((along + 0.12 * self._n(3.0, seed + 41) - 0.78) / 0.5)
                out = out * (1 - loose * (1 - tip))
        else:
            out = np.maximum(self.fade(seed=seed), arm * sm(-self.sdf_arm / (14 * s) + 1))
        if self.behind is not None:
            out = out * (1 - sm((np.mgrid[0:self.h, 0:self.w][0].astype(np.float32) - self.behind[None, :]) / 4.0))
        return out

    def _how(self, seed=9):
        """Along the outline, 0 where the edge is hard and 1 where it is lost."""
        how = sm((self._n(3.6, seed + 11) + 0.9) / 1.5)
        if self.form == "phantom":
            return 0.62 + 0.38 * how
        if self.form == "shroud":
            # going down the edge is more and more lost: no line left to follow by the hem
            yy = np.mgrid[0:self.h, 0:self.w][0].astype(np.float32)
            how = np.maximum(how, sm((yy - (self.parts["hip"][1] - 430 * self.scale)) / (190 * self.scale)))
        return how * (1 - 0.7 * sm((30 * self.scale - self.sdf_keep) / (30 * self.scale)))

    def reserve(self, seed=9):
        """What the running water cannot enter: where the figure's edge is to be hard, up to just inside it; where it is to be lost, only the core, so the ink runs on in and carve() can fade it out. And the hair."""
        s, h, w = self.scale, self.h, self.w
        if self.form == "phantom":
            return np.zeros((h, w), np.float32)      # the ink goes everywhere and the figure is lifted out of it at one strength: no hard patch of paper inside the mist
        wander = 3 * s * self._n(2.9, seed + 7)
        how = self._how(seed)
        r = sm((-self.sdf + wander - (3 + 36 * how) * s) / (8 * s))
        hair = sm((-self.sdf_hair + 0.5 * s) / (1.5 * s))
        return np.float32(np.maximum(r * sm((self.fade_all(seed) - 0.72) / 0.28), hair))

    def carve(self, seed=9):
        """Where the figure lifts the ink off, and how: `cut` is the shape itself, and `rim` the thinning of the ink in the narrow band around it. Works on a finished plate, so the pose can change without painting again."""
        s, h, w = self.scale, self.h, self.w
        wander = 4 * s * self._n(3.0, seed + 7)
        how = self._how(seed)
        soft = (3 + 40 * how) * s
        fade = self.fade_all(seed)
        self.faded = fade
        if self.form == "phantom":
            wander = wander + 2.0 * self._n(3.2, seed + 20)      # mist has no outline to follow, but it is not a blur either
        cut = sm((-self.sdf + wander) / soft + 0.45) * fade
        if self.form == "full":
            cut = np.maximum(cut, 0.86 * sm((-self.sdf_hair + 0.4 * s) / (2.0 * s)) * (0.75 + 0.25 * sm(1 - self.sdf_hair * 0 - 0)))
        width = (34 * (1 + 0.5 * self._n(3.2, seed + 12))) * s
        rim = np.exp(-(np.maximum(self.sdf - wander, 0) / np.maximum(width, 6 * s)) ** 2) * fade
        return np.float32(cut), np.float32(rim), np.float32(how)


def main():
    if sys.argv[1] == "shape":
        lifts = [float(v) for v in sys.argv[2:]] or [0.22, 1.0]
        import os
        from film_myth_wash import OUT
        tiles = []
        for lift in lifts:
            g = Ghost(lift)
            cut, rim, how = g.carve()
            a = 40 + 205 * cut + 0 * rim
            a = a * (1 - 0.85 * g.mouth())
            tiles.append(Image.fromarray(np.uint8(np.clip(a, 0, 255))).crop((400, 150, 1360, 1050)))
        sheet = Image.new("L", (960 * len(tiles), 900))
        for i, t in enumerate(tiles):
            sheet.paste(t, (i * 960, 0))
        os.makedirs(OUT, exist_ok=True)
        sheet.save(os.path.join(OUT, "ghost_shape.png"))


if __name__ == "__main__":
    main()
