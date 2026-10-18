"""AI 发展路线图 (episode 2). Storyboard: ai_history.scenes.md (v2 revision at the top overrides the rest).

Look (from the user's reference, 2026-10-18): warm near-black, one glowing main path (AI's momentum) whose head stays
near the centre, small serif labels at the nodes, serif titles, a representative picture per node shown as a soft,
warm-toned plate. No decorative lines (user, 2026-10-18): every line on screen means something (a school's strand,
a hand-over, an influence between two real nodes).

Style frame for review (vertical, 1080x1920, saves one PNG):
    $MANIM -r 1080,1920 -s --disable_caching ai_history.py StyleFrameV

The y axis is a subjective "AI 发展程度" index (labelled 示意): each milestone has a weight, the line is the smoothed
running sum. Weights below are provisional; the real table will live in ../../ai_history/milestones.csv (weight column).
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *  # noqa: F401,F403  (palette, cn, CN_FONT, TeX PATH)

CREAM = "#efe3c6"       # hairlines, labels
CORE = "#fff4d8"        # the main path's core
GLOW = "#f6d99c"        # bloom around the main path and the head
SPARK = "#c8473a"       # the few red accents
COLD = "#9fb3c8"        # setbacks / AI winters
SERIF = "Songti SC"
SONGTI = "/System/Library/Fonts/Supplemental/Songti.ttc"
TIMES_B = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
TIMES = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ai_history", "images")
EM = 0.01366            # Text em (one CJK advance) in scene units per font-size point (measured in alphago_mcts.py)

# (year, weight, short label[, span]) -- provisional, subjective; later from milestones.csv.
# A positive node ramps the line up over ~TAU years ending at its date (the dot sits at the date).
# A setback has a negative weight and an explicit span: the line slides down from `year` for `span` years, so a winter
# reads as a long, gentle slide, never a spike.
TAU = 1.7
MILESTONES = [
    (1943.5, 1, "神经元模型"), (1950.8, 2, "图灵测试"), (1956.6, 3, "达特茅斯会议"), (1958.5, 2, "感知机"),
    (1966.0, 1, "ELIZA"), (1969.0, -1.5, "神经网络遭质疑", 3), (1973.5, -4.0, "第一次寒冬", 6.5),
    (1985.0, 4, "专家系统", 5), (1986.8, 1.5, "反向传播"), (1987.0, -5.0, "第二次寒冬", 6),
    (1989.5, 1, "卷积网络"), (1997.4, 4, "深蓝"),
    (1997.9, 2, "LSTM"), (2006.5, 2, "深度学习"), (2009.5, 3, "ImageNet"), (2012.75, 8, "AlexNet"),
    (2014.5, 4, "GAN"), (2015.9, 5, "ResNet"), (2016.2, 8, "AlphaGo"), (2017.45, 10, "Transformer"),
    (2018.8, 6, "BERT"), (2019.1, 5, "GPT-2"), (2020.4, 9, "GPT-3"), (2020.9, 8, "AlphaFold 2"),
    (2021.5, 6, "DALL·E"), (2022.6, 7, "Stable Diffusion"), (2022.92, 15, "ChatGPT"), (2023.2, 14, "GPT-4"),
    # 2023-2026: provisional, short ramps (span 0.4 y); to be replaced by ../../ai_history/milestones.csv
    (2023.6, 6, "开源大模型", 0.4), (2024.15, 7, "Sora", 0.4), (2024.55, 9, "IMO 银牌水平", 0.4),
    (2024.7, 10, "o1", 0.4), (2024.78, 8, "诺贝尔奖", 0.4), (2025.05, 10, "DeepSeek-R1", 0.4),
    (2025.1, 5, "春晚机器人", 0.4), (2025.38, 9, "AlphaEvolve", 0.4), (2025.4, 7, "Codex", 0.4),
    (2025.55, 12, "IMO 金牌水平", 0.4), (2026.38, 12, "Erdős 猜想", 0.4), (2026.58, 10, "十项数学成果", 0.4),
    (2026.69, 15, "Navier–Stokes", 0.4), (2026.77, 10, "新一批数学成果", 0.4),
]


def ramp(m):
    """(start, end) years over which milestone m moves the line."""
    ty, w = m[0], m[1]
    if w < 0:
        return ty, ty + m[3]
    span = m[3] if len(m) > 3 else TAU
    return ty - span, ty


WINTERS = [ramp(m) for m in MILESTONES if m[1] < 0 and len(m) > 3 and m[3] >= 5]


def dev_index(t, ms=None, drift=0.02, t_start=1943):
    """Smoothed running sum of weights plus a slow drift, so the early decades still grow a little."""
    t = np.asarray(t, dtype=float)
    y = drift * np.clip(t - t_start, 0, None)
    for m in (MILESTONES if ms is None else ms):
        a, b = ramp(m)
        s = np.clip((t - a) / (b - a), 0, 1)
        y = y + m[1] * s * s * (3 - 2 * s)
    return y


MILESTONE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ai_history", "milestones.csv")


def date_to_year(d):
    """'1956', '1997-05' or '2026-09-08' -> fractional year (mid-month / mid-year when the day or month is missing)."""
    parts = [int(x) for x in d.split("-")]
    if len(parts) == 1:
        return parts[0] + 0.5
    if len(parts) == 2:
        return parts[0] + (parts[1] - 0.5) / 12
    return parts[0] + (parts[1] - 1) / 12 + (parts[2] - 0.5) / 365


def main_nodes():
    """[(year, axis label)] for tier=main rows; label unit shrinks with time: 1997 / 2024.07 / 05.20."""
    import csv
    if not os.path.exists(MILESTONE_CSV):
        return []
    out = []
    with open(MILESTONE_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["tier"] != "main" or r["weight"].strip().startswith("-"):
                continue
            d = r["date"]
            y = int(d[:4])
            lab = d[:4] if y < 2024 else (d[:7].replace("-", ".") if y < 2026 else d[5:10].replace("-", "."))
            out.append((date_to_year(d), lab))
    return sorted(out)


def csv_events():
    """Ramps from the verified table: (start, end, weight). Gains ramp in before their date (longer in the sparse early
    decades, shorter later); setbacks slide down from their date for span_years (capped at 6)."""
    import csv
    ev = []
    with open(MILESTONE_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t, w = date_to_year(r["date"]), float(r["weight"])
            if w < 0:
                ev.append((t, t + min(float(r["span_years"] or 3), 6.0), w))
            else:
                tau = 1.5 if t < 2000 else 0.8 if t < 2020 else 0.6
                ev.append((t - tau, t, w))
    return ev


def csv_index(t, ev):
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t)
    for a, b, w in ev:
        s = np.clip((t - a) / (b - a), 0, 1)
        y = y + w * s * s * (3 - 2 * s)
    return y


def winter_amount(t):
    """0..1: how deep inside an AI winter year t is (soft edges), for colouring the line cold."""
    v = 0.0
    for a, b in WINTERS:
        v = max(v, smoothstep((t - a) / 1.5) * (1 - smoothstep((t - b) / 1.5)))
    return v


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def rgb(hexcol):
    return np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255


def rgba_mob(rgb_arr, alpha, width):
    """Float RGB (H, W, 3) or a single colour, alpha (H, W), both 0-1 -> dithered RGBA ImageMobject `width` units wide."""
    rng = np.random.default_rng(11)
    if np.ndim(rgb_arr) == 1:
        rgb_arr = np.broadcast_to(rgb_arr, alpha.shape + (3,))
    img = np.dstack([rgb_arr, alpha])
    img8 = np.clip(np.round(img * 255 + rng.uniform(-0.5, 0.5, img.shape)), 0, 255).astype(np.uint8)
    m = ImageMobject(img8)
    m.width = width
    return m


def songti(px, weight="Black"):
    """Songti SC face from the system .ttc by style name (Light / Regular / Bold / Black)."""
    for i in range(16):
        try:
            f = ImageFont.truetype(SONGTI, px, index=i)
        except OSError:
            break
        fam, sty = f.getname()
        if fam.startswith("Songti SC") and sty.lower() == weight.lower():
            return f
    return ImageFont.truetype(SONGTI, px, index=0)


def background(W, H, focus_px, top_hex="#0b0907", bot_hex="#15110c", glow_hex="#f3cf8e", seed=2026):
    """One smooth plate: warm near-black gradient, faint low-frequency mottling, warm glow behind the head,
    sparse dust, vignette, fine grain. One image, no stacked layers (handbook: no banding)."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:H, 0:W].astype(float)
    v = ys / H
    top, bot = rgb(top_hex), rgb(bot_hex)
    img = top[None, None, :] * (1 - v[..., None]) + bot[None, None, :] * v[..., None]

    mott = gaussian_filter(rng.standard_normal((H // 8, W // 8)), 10)
    mott = np.kron(mott / (np.abs(mott).max() + 1e-9), np.ones((8, 8)))[:H, :W]
    img += gaussian_filter(mott, 8)[..., None] * np.array([0.016, 0.012, 0.008])

    fx, fy = focus_px
    d = np.hypot(xs - fx, ys - fy)
    img += ((0.075 * (1 - smoothstep(d / (0.75 * W))) ** 2) + 0.05 * (1 - smoothstep(d / (0.25 * W))) ** 2)[..., None] \
        * rgb(glow_hex)[None, None, :]

    dust = np.zeros((H, W))
    n = 500
    np.add.at(dust, (rng.integers(0, H, n), rng.integers(0, W, n)), rng.pareto(2.5, n) * 0.05 + 0.02)
    img += (gaussian_filter(dust, 0.8) * 2.2)[..., None] * rgb("#fff0d0")

    r2 = ((xs - W / 2) / (0.72 * W)) ** 2 + ((ys - fy) / (0.66 * H)) ** 2
    img *= (1 - 0.5 * smoothstep(r2 / 1.4))[..., None]
    img += rng.normal(0, 1.3 / 255, img.shape)
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)


class StyleFrameV(Scene):
    """One still: AI's path at 2017 (Transformer); the head sits near the centre, the camera follows it."""

    CREAM, CORE, GLOW, SPARK = CREAM, CORE, GLOW, SPARK
    LINE_OLD = "#6f6046"
    BG = ("#0b0907", "#15110c", "#f3cf8e")          # top, bottom, glow behind the head
    ERA = ("gpu_gtx1060_rear.jpg", 0.10, (0.5, 0.5), 9.0)   # heavy blur: the board carries a brand logo
    HUD = (("03", "06", "点燃"), "距上一个突破", "15 个月")     # AlphaGo 2016-03 -> Transformer 2017-06     # background plate: what AI ran on in this era (image, opacity)
    PLATE = True              # the node's picture (here: the Transformer paper) + "ChatGPT" with its T linked to it
    SNOW = False
    KEEPERS = 0

    HEAD_YEAR = 2017.5
    HEAD = np.array([-0.05, -0.25, 0])
    BACK_YEARS = 14.0
    KY = 0.047            # units per index point (the camera's vertical zoom for this section)

    def __init__(self, **kw):
        config.frame_width = config.frame_height * config.pixel_width / config.pixel_height
        super().__init__(**kw)

    def fs(self, px):
        return px / (EM * config.pixel_height / config.frame_height)

    def construct(self):
        fw, fh = config.frame_width, config.frame_height
        W, H = config.pixel_width, config.pixel_height
        ppu = H / fh
        to_px = lambda p: ((p[0] + fw / 2) * ppu, (fh / 2 - p[1]) * ppu)
        head = self.HEAD

        # world -> screen (camera follows the head)
        t1 = self.HEAD_YEAR
        t0 = t1 - self.BACK_YEARS - 5
        kx = (head[0] + fw / 2 + 0.2) / self.BACK_YEARS
        ky = self.KY
        sx = lambda t: head[0] + (t - t1) * kx
        sy = lambda t: head[1] + (dev_index(t) - dev_index(t1)) * ky
        ts = np.linspace(t0, t1, 1200)
        pts = np.array([[sx(t), sy(t), 0] for t in ts])

        self.add(ImageMobject(background(W, H, to_px(head), *self.BG)).set(width=fw))
        if self.ERA:
            self.add(self.era_plate(*self.ERA))
        if self.SNOW:
            self.add(self.snow(W, H), self.mist(W, H))

        # bloom along the path + around the head + a faint horizontal streak: one image, one colour
        mask = np.zeros((H, W))
        for p in pts:
            x, y = to_px(p)
            if 0 <= x < W and 0 <= y < H:
                mask[int(y), int(x)] += 1.0
        mask = np.minimum(mask, 1.0)
        g1, g2 = gaussian_filter(mask, 4), gaussian_filter(mask, 18)
        age_ramp = smoothstep((np.arange(W)[None, :] - 0.02 * W) / (0.55 * W))   # older part dimmer
        light = 1 - 0.78 * winter_amount(t1)          # expectation and funding: the light nearly goes out
        a = (0.5 * g1 / g1.max() + 0.28 * g2 / g2.max()) * (0.35 + 0.65 * age_ramp) * light
        hx, hy = to_px(head)
        ys, xs = np.mgrid[0:H, 0:W].astype(float)
        rr = np.hypot(xs - hx, ys - hy)
        a += light * (0.9 * np.exp(-(rr / 14) ** 2) + 0.45 * np.exp(-(rr / 60) ** 2) + 0.18 * np.exp(-(rr / 170) ** 2))
        a += light * 0.22 * np.exp(-((xs - hx) / 260) ** 2 - ((ys - hy) / 3.5) ** 2)
        self.add(rgba_mob(rgb(self.GLOW), np.clip(a, 0, 1), fw))

        # the path's crisp core, dim at the far end
        core = VMobject().set_points_smoothly(pts[::8].tolist() + [pts[-1].tolist()])
        cols = []
        for t in np.linspace(t0, t1, 60):
            age = smoothstep((t - t0) / (t1 - t0))
            c = interpolate_color(ManimColor(self.LINE_OLD), ManimColor(self.CORE), age)
            cols.append(interpolate_color(c, ManimColor(COLD), 0.85 * winter_amount(t)))
        core.set_stroke(color=cols, width=3.2)
        self.add(core)
        # nodes and serif labels (year left of the dot, name right of it, like the reference); only the big
        # nodes get labels, and a label is skipped if it would touch one already placed
        placed = []
        for ty, w, name, *m_extra in sorted(MILESTONES, key=lambda m: -m[1]):
            if not (t1 - self.BACK_YEARS < ty < t1 - 0.2):
                continue
            if w < 0:                       # a setback: no dot, a cold label under the sag
                a_, b_ = ramp((ty, w, name) + tuple(m_extra))
                tm = (a_ + b_) / 2
                p = np.array([sx(tm), sy(b_), 0])
                lab = Text(name, font=SERIF, font_size=self.fs(30), color=COLD).next_to(p, DOWN, 0.3).set_opacity(0.9)
                lab.shift(RIGHT * max(0, -fw / 2 + 0.3 - lab.get_left()[0]) + LEFT * max(0, lab.get_right()[0] - fw / 2 + 0.3))
                self.add(lab)
                continue
            p = np.array([sx(ty), sy(ty), 0])
            big = w >= 8
            self.add(Dot(p, radius=0.045 if big else 0.026, color=self.CORE).set_opacity(1 if big else 0.7).set_z_index(2))
            if not big or any(abs(p[1] - q[1]) < 0.3 and abs(p[0] - q[0]) < 1.6 for q in placed):
                continue
            self.add(Text(name, font=SERIF, font_size=self.fs(30), color=self.CREAM).next_to(p, RIGHT, 0.14).set_opacity(0.8))
            placed.append(p)
        self.add(Dot(head, radius=0.06, color="#fffaf0").set_z_index(3))

        # in a winter, a few faint warm points keep flickering in the dark: the researchers who did not stop
        if self.KEEPERS:
            rng = np.random.default_rng(4)
            for _ in range(self.KEEPERS):
                q = head + np.array([rng.uniform(-1.9, 1.9), rng.uniform(-2.2, 1.6), 0])
                d = 0.5
                yy, xx = np.mgrid[0:int(d * ppu), 0:int(d * ppu)].astype(float)
                c = d * ppu / 2
                a_k = rng.uniform(0.35, 0.7) * np.exp(-(np.hypot(xx - c, yy - c) / (0.12 * c)) ** 2) \
                    + 0.25 * np.exp(-(np.hypot(xx - c, yy - c) / (0.45 * c)) ** 2)
                self.add(rgba_mob(rgb("#f3cf8e"), np.clip(a_k, 0, 1), d).move_to(q))

        # representative picture, tied to its node by a hairline; "ChatGPT" with its T tied to the paper's title
        if self.PLATE:
            plate = self.paper_plate(width=1.55).rotate(-2.5 * DEGREES).move_to([1.15, 0.95, 0])
            corner = plate.get_corner(DL) + np.array([0.12, 0.14, 0])
            self.add(Line(head + np.array([0.05, 0.06, 0]), corner, stroke_width=1.2, color=self.CORE, stroke_opacity=0.55),
                     plate)
            word = Text("ChatGPT", font="Times New Roman", font_size=self.fs(64), color=self.CREAM,
                        t2c={"T": "#f2c14e"}).move_to([-0.55, 2.05, 0])
            t_glyph = word[-1]
            title_pt = plate.get_top() + np.array([-0.05, -0.32, 0])
            self.add(word, Line(t_glyph.get_bottom() + DOWN * 0.04, title_pt, stroke_width=1.2, color="#f2c14e",
                                stroke_opacity=0.7))

        self.add(self.year_mark(int(t1)).move_to([0.55, 3.15, 0]))
        if self.HUD:
            self.add(self.hud(*self.HUD))
        axis = Text("AI 发展（示意）", font=SERIF, font_size=self.fs(24), color=self.CREAM).rotate(PI / 2)
        self.add(axis.move_to([-fw / 2 + 0.16, -2.2, 0]).set_opacity(0.45))

    # ---------- pieces ----------
    def hud(self, chapter, label, value, at=None):
        """Top-left readout for important moments only: a small chapter marker, then one number with its label.
        chapter like ("03", "06", "加速"); value is the big number (e.g. "15 个月")."""
        fw, fh = config.frame_width, config.frame_height
        num, total, name = chapter
        tag = VGroup(Text(f"{num} / {total}", font="Times New Roman", font_size=self.fs(24), color="#f2c14e"),
                     Line(ORIGIN, RIGHT * 0.32, stroke_width=1, color=self.CREAM, stroke_opacity=0.5),
                     Text(name, font=SERIF, font_size=self.fs(24), color=self.CREAM).set_opacity(0.7)).arrange(RIGHT, buff=0.1)
        lab = Text(label, font=SERIF, font_size=self.fs(26), color=self.CREAM).set_opacity(0.7)
        val = Text(value, font=SERIF, font_size=self.fs(60), color="#f2c14e")
        g = VGroup(tag, lab, val).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        return g.move_to(at if at is not None else [-fw / 2 + 0.3 + g.width / 2, fh / 2 - 0.35 - g.height / 2, 0])

    def era_plate(self, name, opacity, focus=(0.5, 0.5), blur=4.0):
        """The era's machine as a ghostly print: greyscale, cover-cropped to the frame, softened, brightness becomes
        alpha in the palette's tint, edges fade out. Sources and licences: ai_history/images/LICENSES.md."""
        W, H = config.pixel_width, config.pixel_height
        im = Image.open(os.path.join(IMG_DIR, name)).convert("L")
        w0, h0 = im.size
        k = max(W / w0, H / h0)
        im = im.resize((int(w0 * k) + 1, int(h0 * k) + 1), Image.LANCZOS)
        x0 = int((im.size[0] - W) * focus[0])
        y0 = int((im.size[1] - H) * focus[1])
        # blurred enough that logos, labels and placards in the photo cannot be read: only the machine's shape stays
        L_ = gaussian_filter(np.asarray(im.crop((x0, y0, x0 + W, y0 + H)), dtype=float) / 255, blur)
        L_ = np.clip((L_ - np.percentile(L_, 5)) / (np.percentile(L_, 99) - np.percentile(L_, 5) + 1e-9), 0, 1)
        ys, xs = np.mgrid[0:H, 0:W].astype(float)
        r2 = ((xs - W / 2) / (0.62 * W)) ** 2 + ((ys - H / 2) / (0.55 * H)) ** 2
        alpha = opacity * L_ ** 1.3 * (1 - smoothstep(r2 / 1.2))
        return rgba_mob(rgb(self.CREAM), np.clip(alpha, 0, 1), config.frame_width)

    def snow(self, W, H, seed=8):
        """Sparse snow: soft flakes of a few sizes plus short motion streaks, one image."""
        rng = np.random.default_rng(seed)
        a = np.zeros((H, W))
        for n, sig, amp in [(260, 1.0, 0.9), (60, 2.4, 0.8), (14, 5.0, 0.7)]:
            m = np.zeros((H, W))
            np.add.at(m, (rng.integers(0, H, n), rng.integers(0, W, n)), amp)
            a += gaussian_filter(m, sig) * (2 * np.pi * sig ** 2)
        streak = np.zeros((H, W))
        np.add.at(streak, (rng.integers(0, H, 12), rng.integers(0, W, 12)), 1.0)
        a += gaussian_filter(streak, (3.5, 0.8)) * 22
        return rgba_mob(rgb("#e8eef7"), np.clip(a * 0.55, 0, 0.8), config.frame_width)

    def mist(self, W, H, seed=12):
        """Cold ground mist in the lower third (winters only): smooth, slightly uneven, one image."""
        rng = np.random.default_rng(seed)
        ys = np.arange(H)[:, None] / H
        tex = gaussian_filter(rng.standard_normal((H // 16, W // 16)), 3)
        tex = np.kron(tex / (np.abs(tex).max() + 1e-9), np.ones((16, 16)))[:H, :W]
        a = 0.2 * smoothstep((ys - 0.62) / 0.38) * (0.75 + 0.25 * gaussian_filter(tex, 10))
        return rgba_mob(rgb("#9fb6d0"), np.clip(a, 0, 1), config.frame_width)

    def year_mark(self, year):
        """The only clock on screen: the current year, large, light serif with a soft halo."""
        return self.title([str(year)], px=150, weight="Regular", tracking=0.08)

    def title(self, lines, px, weight="Black", tracking=0.0):
        """Serif title rendered once at delivery size: crisp text plus a soft warm halo (the reference's glow)."""
        f = songti(px, weight)
        lh = int(px * 1.35)
        gap = tracking * px
        width = lambda s: sum(f.getlength(ch) for ch in s) + gap * (len(s) - 1)
        wpx = int(max(width(s) for s in lines)) + 80
        hpx = lh * len(lines) + 60
        a = Image.new("L", (wpx, hpx), 0)
        d = ImageDraw.Draw(a)
        for k, s in enumerate(lines):
            x = (wpx - width(s)) / 2
            for ch in s:
                d.text((x, 30 + k * lh), ch, font=f, fill=255)
                x += f.getlength(ch) + gap
        A = np.asarray(a, dtype=float) / 255
        halo = gaussian_filter(A, 9)
        alpha = np.clip(0.93 * A + 0.45 * halo, 0, 1)
        col = rgb(self.CREAM)[None, None, :] * (A[..., None] > 0.02) + rgb(self.GLOW)[None, None, :] * (A[..., None] <= 0.02)
        ppu = config.pixel_height / config.frame_height
        return rgba_mob(col, alpha, wpx / ppu)

    def paper_plate(self, width, page_png="attention_p1.png", crop=(0.10, 0.06, 0.90, 0.68)):
        """A paper's real first page (rendered from the arXiv PDF into ai_history/images), cropped to title, authors and
        abstract: warm paper, crisp edge, light falling from the top left, a soft drop shadow. Falls back to a drawn
        page if the file is missing."""
        path = os.path.join(IMG_DIR, page_png)
        if os.path.exists(path):
            src = Image.open(path).convert("RGB")
            w0, h0 = src.size
            src = src.crop((int(crop[0] * w0), int(crop[1] * h0), int(crop[2] * w0), int(crop[3] * h0)))
            src = src.resize((900, int(900 * src.size[1] / src.size[0])), Image.LANCZOS)
            return self._plate_finish(np.asarray(src, dtype=float) / 255, width)
        Wp, Hp = 900, 1180
        img = Image.new("RGB", (Wp, Hp), (243, 239, 228))
        d = ImageDraw.Draw(img)
        tf = ImageFont.truetype(TIMES_B, 56)
        title = "Attention Is All You Need"
        d.text(((Wp - tf.getlength(title)) / 2, 120), title, font=tf, fill=(25, 25, 25))
        af = ImageFont.truetype(TIMES, 24)
        for k, s in enumerate(["Ashish Vaswani   Noam Shazeer   Niki Parmar   Jakob Uszkoreit",
                               "Llion Jones   Aidan N. Gomez   Łukasz Kaiser   Illia Polosukhin"]):
            d.text(((Wp - af.getlength(s)) / 2, 230 + k * 36), s, font=af, fill=(70, 70, 70))
        hf = ImageFont.truetype(TIMES_B, 32)
        d.text(((Wp - hf.getlength("Abstract")) / 2, 360), "Abstract", font=hf, fill=(25, 25, 25))
        rng = np.random.default_rng(5)
        y = 420
        for k in range(22):
            w = 640 if k in (7, 15, 21) else 700
            x0 = 100 + (40 if k in (0, 8, 16) else 0)
            d.rectangle([x0, y, 100 + w * rng.uniform(0.6, 1.0) if k in (7, 15, 21) else 800, y + 9], fill=(150, 146, 138))
            y += 30 if k not in (7, 15) else 54
        return self._plate_finish(np.asarray(img, dtype=float) / 255, width)

    def _plate_finish(self, A, width):
        Hp, Wp = A.shape[:2]
        A = 0.10 + 0.84 * A                                   # lift blacks, soften contrast
        A *= np.array([1.0, 0.93, 0.80])[None, None, :]       # warm tone
        ys, xs = np.mgrid[0:Hp, 0:Wp].astype(float)
        light = 1.0 - 0.28 * smoothstep((xs / Wp * 0.6 + ys / Hp * 0.8) / 1.4)   # brighter top left
        edge = np.minimum.reduce([xs, Wp - 1 - xs, ys, Hp - 1 - ys])
        A *= (light * (1 - 0.18 * (1 - smoothstep(edge / 120.0))))[..., None]   # gentle inner vignette
        pad = 90
        Wt, Ht = Wp + 2 * pad, Hp + 2 * pad
        rgb_t = np.zeros((Ht, Wt, 3))
        alpha = np.zeros((Ht, Wt))
        sh = np.zeros((Ht, Wt))
        sh[pad + 24:pad + 24 + Hp, pad + 14:pad + 14 + Wp] = 1
        alpha += 0.6 * gaussian_filter(sh, 26)                # shadow (black)
        rgb_t[pad:pad + Hp, pad:pad + Wp] = A
        page = np.zeros((Ht, Wt))
        page[pad:pad + Hp, pad:pad + Wp] = 1
        page = gaussian_filter(page, 0.8)                     # anti-aliased crisp edge
        alpha = page * 0.97 + (1 - page) * alpha
        rgb_t *= (page > 0.01)[..., None]
        return rgba_mob(rgb_t, np.clip(alpha, 0, 1), width * Wt / Wp)


class StyleFrameWinterV(StyleFrameV):
    """One still in the second AI winter: cold palette, the line sagging, snow, no picture."""

    HEAD_YEAR = 1991.0
    BACK_YEARS = 25.0
    KY = 0.24
    CREAM, CORE, GLOW, SPARK = "#c9d3df", "#e6edf6", "#9fb6d0", "#9fb3c8"
    LINE_OLD = "#4f5a68"
    BG = ("#07090c", "#0d1117", "#7f98b8")
    PLATE = False
    SNOW = True
    KEEPERS = 6
    HUD = None
    ERA = ("symbolics_lisp_chm.jpg", 0.12)


# ======================================================================================
# The relay of schools (user idea 2026-10-18): one school walks into a winter, another takes over the fire,
# and at the end they braid into one rope (AlphaGo = networks + trial-and-error + search; AlphaProof = + formal proof).
# Provisional subjective weights, same format as MILESTONES.
# ======================================================================================
# 规则派 = symbolism (符号主义); 学习派 = connectionism plus statistical learning (learning from data);
# 试错派 = behaviourism / reinforcement learning (行为主义).
SCHOOLS = {
    "规则派": dict(sub="教机器规则", color="#c9d3df", start=1955, ms=[
        (1956.6, 3, "达特茅斯会议"), (1959.5, 2, "通用问题求解器"), (1965.0, 1.5, "早期专家系统"),
        (1973.5, -3.5, "第一次寒冬", 6.5), (1983.0, 5, "专家系统热潮", 5), (1987.0, -5.5, "第二次寒冬", 6)]),
    "学习派": dict(sub="让机器从数据里学", color="#f2c98a", start=1943, ms=[
        (1943.5, 1, "神经元模型"), (1958.5, 2, "感知机"), (1969.0, -2, "神经网络遭质疑", 3),
        (1982.5, 1, "Hopfield 网络"), (1986.8, 2, "反向传播"), (1989.5, 1.2, "卷积网络")]),
    "试错派": dict(sub="在试错中学", color="#d9826b", start=1948, ms=[
        (1948.5, 0.5, "控制论"), (1959.0, 0.5, "跳棋程序"), (1988.5, 0.8, "时序差分学习"),
        (1992.3, 1.5, "TD-Gammon")]),
}


class StyleFrameRelayV(StyleFrameV):
    """One still at 1992: the rule school slides into the second winter and its light goes out; the learning school,
    quiet for years, now holds the fire; the trial-and-error school is a thin strand rising."""

    HEAD_YEAR = 1992.5
    HEAD = np.array([0.35, -0.2, 0])
    BACK_YEARS = 15.0
    KY = 0.46
    TORCH = "学习派"
    ERA = ("sun_e3000_cluster.jpg", 0.11)
    CREAM, CORE, GLOW, SPARK = "#c9d3df", "#fff4d8", "#f6d99c", "#c8473a"
    BG = ("#08090c", "#0f1116", "#8a7a5a")

    def construct(self):
        fw, fh = config.frame_width, config.frame_height
        W, H = config.pixel_width, config.pixel_height
        ppu = H / fh
        to_px = lambda p: ((p[0] + fw / 2) * ppu, (fh / 2 - p[1]) * ppu)
        head = self.HEAD
        t1, t0 = self.HEAD_YEAR, self.HEAD_YEAR - self.BACK_YEARS - 3
        kx = (head[0] + fw / 2 + 0.2) / self.BACK_YEARS
        idx = {k: (lambda t, v=v: dev_index(t, v["ms"], 0.012, v["start"])) for k, v in SCHOOLS.items()}
        y_ref = idx[self.TORCH](t1)
        sx = lambda t: head[0] + (t - t1) * kx
        sy = lambda k, t: head[1] + (idx[k](t) - y_ref) * self.KY

        self.add(ImageMobject(background(W, H, to_px(head), *self.BG)).set(width=fw))
        self.add(self.era_plate(*self.ERA))
        self.add(self.snow(W, H).set_opacity(0.6), self.mist(W, H))

        ts = np.linspace(t0, t1, 1000)
        ys, xs = np.mgrid[0:H, 0:W].astype(float)
        legend = VGroup()
        for k, v in SCHOOLS.items():
            pts = np.array([[sx(t), sy(k, t), 0] for t in ts])
            # how much light this strand carries over time (the torch moves from one school to another)
            if k == self.TORCH:
                light = 0.12 + 0.88 * smoothstep((ts - 1986.0) / 6.0)
            elif k == "规则派":
                light = 0.75 * smoothstep((ts - 1980.0) / 3.0) * (1 - smoothstep((ts - 1987.0) / 4.5)) + 0.05
            else:
                light = 0.08 + 0.12 * smoothstep((ts - 1988.0) / 4.0)
            mask = np.zeros((H, W))
            for p, l in zip(pts, light):
                x, y = to_px(p)
                if 0 <= x < W and 0 <= y < H:
                    mask[int(y), int(x)] = max(mask[int(y), int(x)], l)
            g1, g2 = gaussian_filter(mask, 4), gaussian_filter(mask, 16)
            a = 0.55 * g1 / max(g1.max(), 1e-9) * light.max() + 0.3 * g2 / max(g2.max(), 1e-9) * light.max()
            self.add(rgba_mob(rgb(v["color"]), np.clip(a, 0, 1), fw))
            core = VMobject().set_points_smoothly(pts[::8].tolist() + [pts[-1].tolist()])
            cols = [interpolate_color(ManimColor("#2a2c31"), ManimColor(v["color"]), 0.25 + 0.75 * l)
                    for l in light[:: max(1, len(light) // 50)]]
            core.set_stroke(color=cols, width=3.0 if k == self.TORCH else 2.2)
            self.add(core)
            legend.add(VGroup(Line(ORIGIN, RIGHT * 0.32, stroke_width=3, color=v["color"]),
                              Text(k, font=SERIF, font_size=self.fs(28), color=v["color"]),
                              Text(v["sub"], font=SERIF, font_size=self.fs(24), color=v["color"]).set_opacity(0.7))
                       .arrange(RIGHT, buff=0.14))
            # labelled nodes on this strand
            for ty, w, lab, *rest in v["ms"]:
                if not (t1 - self.BACK_YEARS + 1 < ty < t1 - 0.1):
                    continue
                p = np.array([sx(ty), sy(k, ty), 0])
                if w > 0:
                    self.add(Dot(p, radius=0.03, color=v["color"]).set_z_index(2))
                    if w < 1.5:
                        continue
                    self.add(Text(lab, font=SERIF, font_size=self.fs(28), color=v["color"])
                             .next_to(p, UP if k == "规则派" else RIGHT, 0.12).set_opacity(0.85))
                else:
                    a_, b_ = ramp((ty, w, lab, *rest))
                    q = np.array([sx(a_ + 1.4), sy(k, a_ + 1.4), 0])
                    self.add(Text(lab, font=SERIF, font_size=self.fs(28), color=COLD).next_to(q, LEFT, 0.25).set_opacity(0.9))

        # the hand-over: a spark arcs from the falling rule school to the network strand
        a_pt = np.array([sx(1988.3), sy("规则派", 1988.3), 0])
        b_pt = np.array([sx(1990.6), sy(self.TORCH, 1990.6), 0])
        arc = ArcBetweenPoints(a_pt, b_pt, angle=-0.9)
        # a comet: a trail that thickens and brightens towards its head, then the head itself
        trail = np.zeros((H, W))
        for u in np.linspace(0, 0.8, 500):
            x, y = to_px(arc.point_from_proportion(u))
            trail[int(y), int(x)] = max(trail[int(y), int(x)], (u / 0.8) ** 2)
        trail = 0.9 * gaussian_filter(trail, 1.6) / gaussian_filter(trail, 1.6).max() + 0.4 * gaussian_filter(trail, 8) / gaussian_filter(trail, 8).max()
        cx, cy = to_px(arc.point_from_proportion(0.8))
        rr = np.hypot(xs - cx, ys - cy)
        trail += np.exp(-(rr / 6) ** 2) + 0.45 * np.exp(-(rr / 26) ** 2)
        self.add(rgba_mob(rgb("#ffd88a"), np.clip(trail, 0, 1), fw))

        # the torch head
        hx, hy = to_px(head)
        rr = np.hypot(xs - hx, ys - hy)
        a = 0.9 * np.exp(-(rr / 14) ** 2) + 0.45 * np.exp(-(rr / 60) ** 2) + 0.16 * np.exp(-(rr / 170) ** 2)
        a += 0.2 * np.exp(-((xs - hx) / 240) ** 2 - ((ys - hy) / 3.5) ** 2)
        self.add(rgba_mob(rgb(self.GLOW), np.clip(a, 0, 1), fw))
        self.add(Dot(head, radius=0.06, color="#fffaf0").set_z_index(3))

        self.add(self.year_mark(int(t1)).move_to([0, 3.15, 0]))
        legend.arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to([0, 2.15, 0]).set_opacity(0.85)
        self.add(legend)
        axis = Text("AI 发展（示意）", font=SERIF, font_size=self.fs(24), color=self.CREAM).rotate(PI / 2)
        self.add(axis.move_to([-fw / 2 + 0.16, -2.4, 0]).set_opacity(0.45))


class StyleFrameFinaleV(StyleFrameV):
    """The last pull-back (2026-10-18): the whole line in one frame. Time is drawn compressed towards the past
    (x = -log of the time left to today), the way each big leap squeezed what came before; the tick spacing shows
    the squeeze. The two winters stay visible as small cold folds near the start."""

    NOW = 2026.795          # 2026-10-18
    WARP_C = 2.5            # years; smaller = stronger squeeze of the past (too small flattens the last months)
    HEIGHT_POW = 1.6        # drawn height = (momentum / today's)^p: early decades sink, the end stays steep

    def construct(self):
        fw, fh = config.frame_width, config.frame_height
        W, H = config.pixel_width, config.pixel_height
        ppu = H / fh
        to_px = lambda p: ((p[0] + fw / 2) * ppu, (fh / 2 - p[1]) * ppu)
        T, c, t_start = self.NOW, self.WARP_C, 1943.0
        L, R, B, TOP = -fw / 2 + 0.35, fw / 2 - 0.45, -2.35, 2.2
        u = lambda t: 1 - np.log1p((T - t) / c) / np.log1p((T - t_start) / c)
        sx = lambda t: L + (R - L) * u(t)
        ev = csv_events()
        y_max = float(csv_index(T, ev))
        sy = lambda t: B + (TOP - B) * (max(float(csv_index(t, ev)), 0.0) / y_max) ** self.HEIGHT_POW
        ts = np.linspace(t_start, T, 3000)
        pts = np.array([[sx(t), sy(t), 0] for t in ts])
        head = pts[-1]

        self.add(ImageMobject(background(W, H, to_px(head), *self.BG)).set(width=fw))

        # glow along the whole line, brighter towards today
        mask = np.zeros((H, W))
        for p, t in zip(pts, ts):
            x, y = to_px(p)
            if 0 <= x < W and 0 <= y < H:
                mask[int(y), int(x)] = max(mask[int(y), int(x)], 0.25 + 0.75 * smoothstep((t - 2005) / 21))
        g1, g2 = gaussian_filter(mask, 4), gaussian_filter(mask, 20)
        a = 0.55 * g1 / g1.max() + 0.35 * g2 / g2.max()
        hx, hy = to_px(head)
        ys, xs = np.mgrid[0:H, 0:W].astype(float)
        rr = np.hypot(xs - hx, ys - hy)
        a += 0.95 * np.exp(-(rr / 16) ** 2) + 0.5 * np.exp(-(rr / 70) ** 2) + 0.22 * np.exp(-(rr / 220) ** 2)
        a += 0.25 * np.exp(-((xs - hx) / 320) ** 2 - ((ys - hy) / 4.0) ** 2)
        self.add(rgba_mob(rgb(self.GLOW), np.clip(a, 0, 1), fw))

        core = VMobject().set_points_smoothly(pts[::10].tolist() + [pts[-1].tolist()])
        # Manim spreads a stroke gradient evenly across the curve's x extent, not along time, so sample the colours
        # at evenly spaced x and map each x back to its year through the inverse of the warp
        D = np.log1p((T - t_start) / c)
        cols = []
        for uu in np.linspace(u(t_start), 1.0, 80):
            t = T - c * np.expm1((1 - uu) * D)
            cc = interpolate_color(ManimColor(self.LINE_OLD), ManimColor(self.CORE), smoothstep((t - 1990) / 36))
            cols.append(interpolate_color(cc, ManimColor(COLD), 0.9 * winter_amount(t)))
        core.set_stroke(color=cols, width=3.4)
        self.add(core, Dot(head, radius=0.065, color="#fffaf0").set_z_index(3))

        # the squeezed time axis, ticked at every main node of the verified table; the labels change unit as the
        # breakthroughs crowd in: years, then year.month, then month.day in 2026, and today
        axis_y = B - 0.32
        ticks = VGroup(Line([L, axis_y, 0], [R, axis_y, 0], stroke_width=1, color=self.CREAM, stroke_opacity=0.3))
        nodes = main_nodes()
        x_today = sx(T)
        last_x, seen = [-9.0, -9.0], set()
        ticks.add(Text("10.18", font=SERIF, font_size=self.fs(28), color=self.CORE).next_to([x_today, axis_y, 0], DOWN, 0.08))
        for t, lab in nodes:
            x = sx(t)
            self.add(Dot([x, sy(t), 0], radius=0.03, color=self.CORE)
                     .set_opacity(0.35 + 0.65 * smoothstep((t - 1990) / 36)).set_z_index(2))
            ticks.add(Line([x, axis_y, 0], [x, axis_y + 0.07, 0], stroke_width=1.1, color=self.CREAM, stroke_opacity=0.6))
            if lab in seen:
                continue
            row = 0 if x - last_x[0] > 0.34 and x_today - x > 0.36 else 1
            if x - last_x[row] < 0.34:
                continue
            last_x[row] = x
            seen.add(lab)
            today = False
            ticks.add(Text(lab, font=SERIF, font_size=self.fs(24 if not today else 28),
                           color=self.CORE if today else self.CREAM).set_opacity(1 if today else 0.65)
                      .next_to([x, axis_y - 0.3 * row, 0], DOWN, 0.08))
        self.add(ticks)
        for a_, b_ in WINTERS:
            q = np.array([sx((a_ + b_) / 2), sy((a_ + b_) / 2), 0])
            self.add(Text("寒冬", font=SERIF, font_size=self.fs(24), color=COLD).next_to(q, UP, 0.16).set_opacity(0.85))

        self.add(self.title(["2026.10.18"], px=110, weight="Regular", tracking=0.04).move_to([0, 3.2, 0]))
        hud = self.hud(("06", "06", "现在"), "距上一个突破", "12 天")       # 2026-10-06 -> 2026-10-18
        self.add(hud.move_to([-fw / 2 + 0.3 + hud.width / 2, 2.0, 0]))
        axis = Text("AI 发展（示意）", font=SERIF, font_size=self.fs(24), color=self.CREAM).rotate(PI / 2)
        self.add(axis.move_to([-fw / 2 + 0.16, -1.2, 0]).set_opacity(0.45))
