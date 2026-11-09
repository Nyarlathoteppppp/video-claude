"""A brush, for what has to be written and cannot be set: the new character of 十三, ⿰礻⿱甶⿱人吅, the 礻 of 神, the 甶 of 鬼, 人, the 吅 of 器. Every stroke is designed by hand as a path of the brush with its pressure, the way a 楷书
stroke is made: the tip goes in against the direction, presses, travels lighter, presses again and comes back. A stroke is drawn hair by hair, so that it runs dry where a brush runs dry. No font is involved.

    python ink_brush.py test [SIZE]     # the whole character, ink/film_out/new_char/test.png
    python ink_brush.py strip [N]       # N frames of it being written, ink/film_out/new_char/strip.png
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from ink_grade import PAPER, fft_noise, sm
from ink_subs import _font, at

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "new_char")
HAIRS = 340


# ---------------------------------------------------------------- strokes, in a square a hundred wide: (x, y, pressure), where a pressure of 1 is the full body of the brush

def heng(x0, y0, x1, y1, w=1.0, rise=0.6):
    """横, rising as a running hand has it: the tip laid in at a slant, a press, away lighter and very slightly arched, a lift of the shoulder and a short press down at the end."""
    dx, dy = x1 - x0, y1 - y0
    return [(x0 - 0.5, y0 - 0.2, 0.1 * w), (x0 + 0.9, y0 + 0.3, 0.6 * w), (x0 + dx * 0.16, y0 + dy * 0.16, 0.42 * w), (x0 + dx * 0.5, y0 + dy * 0.5 - rise, 0.3 * w), (x0 + dx * 0.85, y0 + dy * 0.85 - rise * 0.4, 0.38 * w), (x1 - 0.2, y1 + 0.1, 0.56 * w), (x1 + 0.2, y1 + 0.9, 0.34 * w), (x1 - 0.6, y1 + 1.2, 0.06 * w)]


def shu(x0, y0, x1, y1, w=1.0, end="lu", bow=0.5):
    """竖: laid in from the left, a press, down with a slight waist and a slight bow, and either 垂露, a press that comes back up, or 悬针, away to nothing."""
    dx, dy = x1 - x0, y1 - y0
    a = [(x0 - 0.5, y0 - 0.5, 0.12 * w), (x0 + 0.3, y0 + 0.4, 0.66 * w), (x0 + dx * 0.14, y0 + dy * 0.14, 0.5 * w), (x0 + dx * 0.5 + bow, y0 + dy * 0.5, 0.4 * w)]
    if end == "lu":
        return a + [(x0 + dx * 0.85 + bow * 0.4, y0 + dy * 0.85, 0.44 * w), (x1 + 0.1, y1 - 0.6, 0.58 * w), (x1 - 0.1, y1 + 0.3, 0.3 * w), (x1 - 0.5, y1 - 0.2, 0.06 * w)]
    return a + [(x0 + dx * 0.78 + bow * 0.4, y0 + dy * 0.78, 0.42 * w), (x0 + dx * 0.92, y0 + dy * 0.92, 0.2 * w), (x1, y1, 0.015)]


def pie(x0, y0, x1, y1, w=1.0, bend=0.16):
    """撇: pressed in, then away and lighter all the way out to the tip, curving."""
    dx, dy = x1 - x0, y1 - y0
    mx, my = x0 + dx * 0.55 + dy * bend * 0.3, y0 + dy * 0.55 - dx * bend * 0.3
    return [(x0 + 0.5, y0 - 0.7, 0.14 * w), (x0 - 0.1, y0 + 0.3, 0.66 * w), (x0 + dx * 0.22, y0 + dy * 0.22, 0.56 * w), (mx, my, 0.42 * w), (x0 + dx * 0.82, y0 + dy * 0.85, 0.2 * w), (x1, y1, 0.015)]


def na(x0, y0, x1, y1, w=1.0):
    """捺: thin at the start, swelling, then the foot: pressed flat and drawn out level to a point."""
    dx, dy = x1 - x0, y1 - y0
    return [(x0 - 0.3, y0 - 0.4, 0.08 * w), (x0 + dx * 0.18, y0 + dy * 0.2, 0.22 * w), (x0 + dx * 0.45, y0 + dy * 0.52, 0.42 * w), (x0 + dx * 0.68, y0 + dy * 0.8, 0.74 * w), (x0 + dx * 0.79, y0 + dy * 0.93, 0.96 * w), (x0 + dx * 0.9, y0 + dy * 0.985, 0.5 * w), (x1, y1, 0.015)]


def dian(x0, y0, x1, y1, w=1.0):
    """点: touched down thin, pressed, and away: no more than a drop with a direction."""
    dx, dy = x1 - x0, y1 - y0
    return [(x0, y0, 0.05 * w), (x0 + dx * 0.4, y0 + dy * 0.42, 0.42 * w), (x0 + dx * 0.78, y0 + dy * 0.8, 0.8 * w), (x1, y1, 0.46 * w)]


def zhe(x0, y0, xc, yc, x1, y1, w=1.0, then="down"):
    """横折: along and rising, a lift, a small press on the shoulder, and down leaning in; one stroke of the brush."""
    dx, dy = xc - x0, yc - y0
    top = [(x0 - 0.5, y0 - 0.2, 0.1 * w), (x0 + 0.8, y0 + 0.3, 0.56 * w), (x0 + dx * 0.16, y0 + dy * 0.16, 0.38 * w), (x0 + dx * 0.55, y0 + dy * 0.55 - 0.3, 0.3 * w), (xc - 1.6, yc - 0.1, 0.34 * w), (xc - 0.1, yc + 0.1, 0.52 * w), (xc + 0.2, yc + 1.5, 0.6 * w)]
    if then == "down":
        ex, ey = x1 - xc, y1 - yc
        return top + [(xc + ex * 0.35 + 0.2, yc + ey * 0.35, 0.46 * w), (xc + ex * 0.8, yc + ey * 0.8, 0.42 * w), (x1, y1 - 0.2, 0.5 * w), (x1 - 0.5, y1 + 0.5, 0.1 * w)]
    return top + [((xc + x1) * 0.5 + 0.8, (yc + y1) * 0.5 - 0.4, 0.44 * w), (x1 + (xc - x1) * 0.2, y1 - (y1 - yc) * 0.18, 0.2 * w), (x1, y1, 0.015)]


def thread(*points):
    """牵丝: the hair the tip leaves on its way from one stroke to the next."""
    return [(x, y, 0.035) for x, y in points]


# in the order they are written, with how much ink the brush has: it is dipped before 礻, before 甶 and before 人
STROKES = (
    # 礻: tall and narrow on the left, leaning into the rest
    ("dian", dian(18.0, 8.0, 23.5, 15.5, 0.78), 1.0),
    ("zhe", zhe(9.0, 26.5, 30.0, 22.0, 7.0, 52.0, 1.0, "pie"), 0.86),
    ("shu", shu(20.0, 38.0, 21.0, 90.5, 1.08, "zhen", 0.8), 0.56),
    ("dian", dian(25.5, 42.0, 33.0, 50.5, 0.74), 0.6),
    # 甶
    ("pie", pie(64.0, 3.0, 58.5, 10.5, 0.66, 0.1), 1.0),
    ("shu", shu(47.5, 14.0, 49.5, 33.5, 0.7, "lu", 0.3), 0.95),
    ("zhe", zhe(48.0, 13.5, 78.0, 10.0, 75.5, 31.0, 0.84), 0.8),
    ("shu", shu(62.5, 12.5, 62.6, 32.0, 0.52, "lu", 0.1), 0.72),
    ("heng", heng(51.0, 22.8, 72.5, 20.8, 0.54, 0.2), 0.66),
    ("heng", heng(47.0, 34.5, 79.0, 31.5, 0.74, 0.3), 0.6),
    # 人: spreading over the two mouths
    ("pie", pie(61.0, 37.0, 37.0, 62.0, 1.1, 0.1), 1.0),
    ("na", na(58.0, 42.0, 97.0, 64.0, 1.12), 0.72),
    # 吅: small and close, tucked between the legs of 人
    ("shu", shu(45.5, 67.5, 46.6, 79.5, 0.62, "lu", 0.1), 0.95),
    ("zhe", zhe(46.0, 67.3, 58.5, 66.2, 57.2, 78.5, 0.66), 0.84),
    ("heng", heng(45.8, 80.8, 58.4, 79.6, 0.58, 0.1), 0.72),
    ("shu", shu(63.0, 66.5, 64.1, 78.5, 0.62, "lu", 0.1), 0.7),
    ("zhe", zhe(63.5, 66.3, 77.5, 65.2, 76.0, 77.5, 0.68), 0.62),
    ("heng", heng(63.3, 79.8, 77.3, 78.6, 0.6, 0.1), 0.5),
)


def spline(points, step):
    """A smooth path through the points, with its pressure, every step units. Catmull-Rom, in the three together."""
    p = np.array(points, np.float64)
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(p) - 2):
        a, b, c, d = p[i - 1], p[i], p[i + 1], p[i + 2]
        n = max(int(np.hypot(*(c - b)[:2]) / step), 2)
        for u in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * (2 * b + (c - a) * u + (2 * a - 5 * b + 4 * c - d) * u * u + (3 * b - a - 3 * c + d) * u ** 3))
    out.append(p[-2])
    q = np.array(out)
    q[:, 2] = np.clip(q[:, 2], 0.0, 1.3)
    return q


class Brush:
    """One brush at one size. The brush's mark on the paper is not a line across the stroke but a long oval leaning the way the hand holds it, its lean following the stroke a little: which is what cuts the head of a 横 at a slant and rounds its
    corners. Hairs are spread over that oval, and each draws its own line as long as it holds ink. Hairs lying side by side run out together, which is what makes a dry stroke streak rather than go grey."""

    def __init__(self, size=2400, body=8.6, seed=3, lean=-0.8, hairs=HAIRS):
        self.size = size
        self.k = size / 100.0
        self.body = body
        self.seed = seed
        self.lean = lean      # radians from level: the long way of the mark, when the brush is at rest
        self.hairs = hairs

    @staticmethod
    def _drift(r, n, m, c=1.0):
        a = gaussian_filter1d(r.normal(0, 1, (n, m)), 120 * c, axis=0, mode="nearest")
        return a / max(np.abs(a).max(), 1e-6)

    def stroke(self, path, index, upto=1.0, ink=1.0, step=0.05):
        """The stroke as far as upto (of its length), an image of its ink, 0 to 1. step: how finely it is followed, in the square's units: small characters don't need it as fine."""
        r = np.random.default_rng(self.seed * 1000 + index)
        q = spline(path, step)
        c = 0.05 / step                                          # every smoothing below is a length along the stroke
        # no hand holds a line quite to where it was meant to go, nor quite to one pressure
        q[:, :2] += 0.28 * self._drift(r, len(q), 2, c)
        q[:, 2] *= 1 + 0.1 * self._drift(r, len(q), 1, c)[:, 0]
        d = np.gradient(q[:, :2], axis=0)
        d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None]
        n = gaussian_filter1d(np.stack([-d[:, 1], d[:, 0]], 1), 26 * c, axis=0, mode="nearest")
        rest = np.array([np.cos(self.lean), np.sin(self.lean)])
        flip = np.sign((n * rest).sum(1, keepdims=True))
        flip[flip == 0] = 1
        axis = 0.62 * n * flip + 0.38 * rest
        axis /= np.maximum(np.hypot(axis[:, 0], axis[:, 1]), 1e-6)[:, None]
        cross = np.stack([-axis[:, 1], axis[:, 0]], 1)
        length = len(q)
        end = int(length * upto)
        im = Image.new("L", (self.size, self.size), 0)
        draw = ImageDraw.Draw(im)
        along = np.cumsum(np.hypot(*np.gradient(q[:, :2], axis=0).T))
        # the hairs, across the brush: how much each holds, those side by side alike
        U = np.sort(r.uniform(-1, 1, self.hairs))
        clump = gaussian_filter1d(r.normal(0, 1, self.hairs), self.hairs / 34, mode="wrap")
        clump = clump / max(np.abs(clump).max(), 1e-6)
        width = max(int(round(self.k * 0.13)), 1)
        for h in range(self.hairs):
            u = U[h]
            v = r.uniform(-1, 1) * np.sqrt(max(1 - u * u, 0))
            wander = gaussian_filter1d(r.normal(0, 1, length), 34 * c, mode="nearest")
            wander = wander / max(np.abs(wander).max(), 1e-6) * 0.05
            A = self.body * 0.5 * q[:, 2]
            ou, ov = A * (u + wander), A * 0.42 * v
            x = (q[:, 0] + axis[:, 0] * ou + cross[:, 0] * ov) * self.k
            y = (q[:, 1] + axis[:, 1] * ou + cross[:, 1] * ov) * self.k
            # what this hair still holds: the brush empties with the distance gone, faster where it goes lightly, and the outer hairs first
            streak = gaussian_filter1d(r.normal(0, 1, length), 220 * c, mode="nearest")
            streak = streak / max(np.abs(streak).max(), 1e-6)
            holds = ink + 0.2 - 0.016 * along * (1.25 - q[:, 2]) - 0.25 * u * u + 0.42 * clump[h] + 0.05 * streak
            on = holds > 0.25
            on[end:] = False
            i = 0
            while i < length:
                if not on[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < length and on[j + 1]:
                    j += 1
                if j - i >= 2:
                    stride = max((j - i) // 240, 1)
                    draw.line(list(zip(x[i:j + 1:stride].tolist(), y[i:j + 1:stride].tolist())), fill=255, width=width)
                i = j + 1
        return np.asarray(im, np.float32) / 255


W, H = 1920, 1080
PAGE = 2400                     # the sheet's pixels: a hundred units of the square to this many, then down to the frame
SCALE = 1070 / PAGE             # the frame's pixels to the sheet's
LEFT, TOP = 320, 10             # where the square's corner is in the frame: left of the subtitles' column
T_WRITE = at("我大概是") + 0.5
T_DONE = at("可说这句话") - 0.3
T_FROM = at("可说这句话")
SOURCES = (("神", 1.5, 24.0), ("鬼", 86.5, 14.0), ("人", 85.0, 40.0), ("器", 84.0, 74.0))       # the four old words, each by what it gave: in the square's units
INK_RGB = np.array((22, 21, 20), np.float32)


def _paper():
    """The same sheet as the end of the river's: nearly clean, with a little tone."""
    tone = 1 + 0.03 * fft_noise(H, W, 3.0, 91 + 7) + 0.014 * fft_noise(H, W, 0.5, 91 + 8)
    return PAPER * tone[..., None]


class Writing:
    """The character being written, stroke by stroke, the river's last ink on the brush. Each stroke is wetted by itself, on the part of the sheet it covers, so a frame costs only the stroke in hand."""

    def __init__(self):
        self.brush = Brush(PAGE)
        self.fibre = fft_noise(PAGE, PAGE, 1.6, 11) * 0.55 + fft_noise(PAGE, PAGE, 2.6, 12) * 0.45
        self.paper = _paper()
        self.done = np.zeros((PAGE, PAGE), np.float32)
        self.count = 0
        self.spans = self._spans()

    @staticmethod
    def _spans():
        """When each stroke is written: for as long as it is long, with the small lift between strokes the hand needs."""
        lengths = np.array([np.hypot(*np.diff(np.array(p)[:, :2], axis=0).T).sum() ** 0.6 + 2.2 for _, p, _ in STROKES])
        share = lengths / lengths.sum() * (T_DONE - T_WRITE)
        ends = T_WRITE + np.cumsum(share)
        return [(float(e - d), float(e - 0.09)) for e, d in zip(ends, share)]

    def _wet(self, a):
        """One stroke's ink, in the paper: on its own part of the sheet, and the whole of it gets its soak."""
        ys, xs = np.nonzero(a)
        if not len(ys):
            return None
        m = int(40 * PAGE / 2400)
        y0, y1, x0, x1 = max(ys.min() - m, 0), min(ys.max() + m, PAGE), max(xs.min() - m, 0), min(xs.max() + m, PAGE)
        k = PAGE / 2400
        part, fibre = a[y0:y1, x0:x1], self.fibre[y0:y1, x0:x1]
        edge = sm((gaussian_filter(part, 1.3 * k) - 0.44 + 0.06 * fibre) / 0.24)
        thick = gaussian_filter(part, 16 * k)
        soak = gaussian_filter(edge, 6 * k) * (0.5 + 0.28 * fibre)
        return (y0, y1, x0, x1), np.clip(edge * (0.86 + 0.2 * thick) + 0.14 * soak * (1 - edge), 0, 1.0)

    def _lay(self, into, wetted, tone=1.0):
        if wetted is not None:
            (y0, y1, x0, x1), d = wetted
            into[y0:y1, x0:x1] = 1 - (1 - into[y0:y1, x0:x1]) * (1 - np.clip(d * tone, 0, 1))

    def ink(self, t):
        """All the ink on the sheet at film time t."""
        whole = sum(1 for _, e in self.spans if e <= t)
        if whole < self.count:
            self.done[:] = 0
            self.count = 0
        for i in range(self.count, whole):
            _, path, load = STROKES[i]
            self._lay(self.done, self._wet(self.brush.stroke(path, i, ink=load)), 0.84 + 0.16 * load)
        self.count = whole
        d = self.done
        if whole < len(STROKES) and t >= self.spans[whole][0]:
            b, e = self.spans[whole]
            x = float(np.clip((t - b) / (e - b), 0, 1))
            d = d.copy()
            _, path, load = STROKES[whole]
            self._lay(d, self._wet(self.brush.stroke(path, whole, upto=x, ink=load)), 0.84 + 0.16 * load)
        return d

    def frame(self, t):
        d = self.ink(t)
        size = int(round(PAGE * SCALE))
        small = np.asarray(Image.fromarray(np.uint8(d * 255 + 0.5)).resize((size, size), Image.LANCZOS), np.float32) / 255
        f = self.paper.copy()
        x0, y0 = LEFT, max(TOP, 0)
        part = small[y0 - TOP:y0 - TOP + min(H - y0, size), :]
        f[y0:y0 + part.shape[0], x0:x0 + part.shape[1]] += (INK_RGB - f[y0:y0 + part.shape[0], x0:x0 + part.shape[1]]) * part[..., None]
        im = Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255)))
        for i, (c, ux, uy) in enumerate(SOURCES):
            k = float(sm((t - T_FROM - 0.2 - i * 0.55) / 0.5))
            if k > 0:
                self._note(im, c, LEFT + ux * size / 100, TOP + uy * size / 100, k)
        z = 1 + 0.035 * np.clip((t - T_WRITE) / (at("可说这句话") + 5.0 - T_WRITE), 0, 1)
        w, h = W / z, H / z
        cx, cy = 860, 500
        return np.asarray(im.resize((W, H), Image.BICUBIC, box=(cx - w * cx / W, cy - h * cy / H, cx + w * (W - cx) / W, cy + h * (H - cy) / H)))

    @staticmethod
    def _note(im, c, x, y, k):
        """One of the old words, small and paler, as a commentator would write it beside the character."""
        font = _font("kai", 66)
        layer = Image.new("L", (92, 92), 0)
        ImageDraw.Draw(layer).text((46, 46), c, font=font, fill=int(205 * k), anchor="mm")
        ink = Image.new("RGB", (92, 92), tuple(int(v) for v in INK_RGB))
        im.paste(ink, (int(x - 46), int(y - 46)), layer)


_writing = None


def write(t):
    """十三, the shot: wired in ink_cut by move="ink_brush:write"."""
    global _writing
    if _writing is None:
        _writing = Writing()
    return _writing.frame(t)


def strip(n=12, cols=4):
    os.makedirs(OUT, exist_ok=True)
    ts = np.linspace(T_WRITE - 0.2, T_FROM + 4.6, n)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 480, rows * 270))
    for i, t in enumerate(ts):
        f = write(float(t))
        sheet.paste(Image.fromarray(f).resize((480, 270), Image.LANCZOS), (i % cols * 480, i // cols * 270))
        print(i, round(float(t), 2), flush=True)
    Image.fromarray(f).save(os.path.join(OUT, "last.png"))
    sheet.save(os.path.join(OUT, "strip.png"))


def wet(a, size, seed=11):
    """Ink in paper: where the hairs lie close they close up into a body, where they do not they stay hairs; the edge takes the paper's roughness, the ink is a little darker where it was laid thick, and it soaks a little beyond itself."""
    k = size / 2400.0
    fibre = fft_noise(size, size, 1.6, seed) * 0.55 + fft_noise(size, size, 2.6, seed + 1) * 0.45
    body = gaussian_filter(a, 1.3 * k)
    edge = sm((body - 0.44 + 0.06 * fibre) / 0.24)
    thick = gaussian_filter(a, 16 * k)
    soak = gaussian_filter(edge, 6 * k) * (0.5 + 0.28 * fibre)
    return np.clip(edge * (0.74 + 0.28 * thick) + 0.14 * soak * (1 - edge), 0, 1.1)


def test(size=2400):
    os.makedirs(OUT, exist_ok=True)
    b = Brush(size)
    ink = np.zeros((size, size), np.float32)
    for i, (_, path, load) in enumerate(STROKES):
        ink = 1 - (1 - ink) * (1 - b.stroke(path, i, ink=load) * (0.84 + 0.16 * load))
    d = wet(ink, size)
    paper = np.array((241, 234, 219), np.float32)
    low = np.array((22, 21, 20), np.float32)
    f = paper + (low - paper) * np.clip(d, 0, 1)[..., None]
    Image.fromarray(np.uint8(f + 0.5)).save(os.path.join(OUT, "test.png"))
    print(os.path.join(OUT, "test.png"))


if __name__ == "__main__":
    if sys.argv[1] == "strip":
        strip(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    else:
        test(int(sys.argv[2]) if len(sys.argv) > 2 else 2400)
