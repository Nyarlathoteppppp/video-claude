"""Paper Great Wave: texture workshop and the 12-pose/sec Cycles wrapper.

From the repository root, with the render environment's Python:
    python3 manim/scenes/style_collage.py art
    Blender -b --factory-startup -P manim/scenes/style_collage_blender.py -- --frames=1,61,85,106,150 --pct=50 --samples=48
    Blender -b --factory-startup -P manim/scenes/style_collage_blender.py -- --samples=96
    python3 manim/scenes/styles_film.py clip collage

Only 60 canonical PNGs are needed by seq(), not 150 repeated renders. Preview
frames are separate, in frames/test/collage_fNNNN.png. No painted shadows.
"""
import math
import os
import sys

import cairo
import numpy as np
from scipy.ndimage import gaussian_filter
from styles_common import SEQ, hx, surface, text, seq

ART = os.path.join(SEQ, "collage_art")
SIZE = 2048


def collage(t):
    pose = min(59, max(0, math.floor(t * 12)))
    return surface(seq("collage", pose / 12))


def paper(col, seed, fibres=3800):
    """Dyed stock: cloudy pulp, short fibres, no lighting baked into the image."""
    r = np.random.default_rng(seed)
    n = r.normal(0, 1, (SIZE, SIZE)).astype(np.float32)
    cloud = gaussian_filter(n, 23)
    cloud /= max(float(cloud.std()), 1e-6)
    grain = gaussian_filter(n, 0.65)
    rgb = np.array(hx(col), np.float32)[None, None] * (1 + 0.006 * cloud + 0.018 * grain)[..., None]
    s = surface(np.clip(rgb, 0, 1))
    c = cairo.Context(s)
    for i in range(fibres):
        x, y = r.uniform(0, SIZE, 2)
        a = r.uniform(-0.6, 0.6) + (math.pi / 2 if i % 4 else 0)
        length = r.uniform(3, 23)
        c.set_source_rgba(*(hx("#F2E5CB") if i % 3 else hx("#4D4D43")), r.uniform(0.025, 0.085))
        c.set_line_width(r.uniform(0.45, 1.35))
        c.move_to(x, y)
        c.curve_to(x + 1, y + 3, x + math.cos(a) * length - 2, y + math.sin(a) * length,
                   x + math.cos(a) * length, y + math.sin(a) * length)
        c.stroke()
    return s, c


def stroke(c, pts, col, width=1, alpha=1):
    c.set_source_rgba(*hx(col), alpha)
    c.set_line_width(width)
    c.move_to(*pts[0])
    for p in pts[1:]:
        c.line_to(*p)
    c.stroke()


def newspaper(c, ink="#5F665F"):
    """Muted found print; the cropped type is texture, not a caption."""
    text(c, "海上来信", 74, 142, 52, ink, "Songti SC", True)
    text(c, "THE COASTAL REVIEW", 78, 208, 25, ink, "Futura")
    text(c, "潮汐 / 航线 / 纸上风景     VOL. 36 — KANAGAWA", 80, 260, 18, ink, "Hiragino Sans GB")
    stroke(c, [(72, 285), (1970, 285)], ink, 5)
    chinese = ["清晨的海面留下深蓝色的纹路。", "风从远处吹来，船沿着海岸前行。", "潮水抬起船头，也带来新的消息。",
               "纸张保存了光、盐和昨日的声音。", "远山很小，浪花在天空下展开。", "我们把海的形状剪下，留在桌上。"]
    english = ["The tide turns before dawn.", "Along the coast, a blue line", "records the passage of wind.", "A small boat crosses the bay;",
               "beyond it, the mountain waits.", "Paper remembers every fold.", "Soundings taken in calm water.", "Salt, ink and the morning sea."]
    for col in range(3):
        x = 78 + col * 650
        if col:
            stroke(c, [(x - 25, 320), (x - 25, 1990)], ink, 1.5, 0.6)
        text(c, ("潮水与远山", "NOTES AT SEA", "蓝色的航线", "WIND / WATER")[col], x, 355, 19, ink,
             "Songti SC" if col % 2 == 0 else "Futura")
        for row in range(29):
            y = 418 + row * 54
            if col % 2 == 0:
                text(c, chinese[(row + col) % len(chinese)], x, y, 29, ink, "Songti SC")
            else:
                text(c, english[(row + col) % len(english)], x, y, 27, ink, "Menlo")


def sea_chart(c, ink="#6C766D"):
    for x in range(80, SIZE, 256):
        stroke(c, [(x, 0), (x, SIZE)], ink, 1, 0.25)
    for y in range(80, SIZE, 256):
        stroke(c, [(0, y), (SIZE, y)], ink, 1, 0.25)
    for k in range(19):
        pts = []
        for u in np.linspace(-0.7, 4.4, 260):
            r = 180 + k * 57 + 35 * math.sin(u * 3 + k * 0.11) + 14 * math.sin(u * 7)
            pts.append((710 + r * math.cos(u), 940 + 0.73 * r * math.sin(u)))
        stroke(c, pts, ink, 2.1, 0.7)
    r = np.random.default_rng(62)
    for y in range(170, 1950, 164):
        for x in range(90, 1950, 188):
            text(c, str(int(r.integers(8, 98))), x + r.uniform(-32, 32), y, 26, ink, "Menlo", italic=True)
    text(c, "SAGAMI BAY", 1190, 290, 38, ink, "Futura")
    text(c, "相模湾 / 水深 单位：米", 1190, 345, 19, ink, "Hiragino Sans GB")
    cx, cy = 1600, 1590
    for a in range(0, 360, 15):
        th = math.radians(a)
        rr = 172 if a % 90 == 0 else 135
        stroke(c, [(cx, cy), (cx + rr * math.sin(th), cy - rr * math.cos(th))], ink, 2, 0.7)
    c.set_source_rgb(*hx(ink))
    c.set_line_width(2)
    c.arc(cx, cy, 115, 0, 2 * math.pi)
    c.stroke()
    text(c, "N", cx - 14, cy - 191, 34, ink, "Futura")


def music(c):
    ink = "#8A7B67"
    text(c, "ANDANTE - THE SEA", 85, 125, 22, ink, "Futura")
    r = np.random.default_rng(33)
    for y in range(235, 2000, 270):
        for k in range(5):
            stroke(c, [(65, y + k * 20), (1990, y + k * 20)], ink, 1.8, 0.65)
        for x in range(95, 1950, 95):
            yy = y + int(r.integers(0, 9)) * 10
            c.save()
            c.translate(x, yy)
            c.rotate(-0.25)
            c.scale(1.45, 1)
            c.arc(0, 0, 7, 0, 2 * math.pi)
            c.set_source_rgb(*hx(ink))
            c.fill()
            c.restore()
            stroke(c, [(x + 9, yy), (x + 9, yy - 61)], ink, 2)
            if x % 3 == 0:
                stroke(c, [(x + 9, yy - 61), (x + 28, yy - 49)], ink, 4)
        for x in range(440, 1950, 475):
            stroke(c, [(x, y), (x, y + 80)], ink, 2)


def art():
    os.makedirs(ART, exist_ok=True)
    stocks = [("indigo", "#052E58"), ("newspaper", "#A9C7D4"), ("chart", "#9CC0CF"),
              ("music", "#E5D6BB"), ("cream", "#EFEBDF"), ("halftone", "#295878"),
              ("graph", "#B7D2DA"), ("sky", "#CFBC9E"), ("desk", "#AAA092"),
              ("tape", "#D5BD8A"), ("stamp", "#F2EBDD"), ("boat", "#BEA579"),
              ("news_pale", "#DBCBAF"), ("chart_pale", "#CEC0A6"),
              ("graph_pale", "#E0D1B3"), ("dots_pale", "#D1C2A6"), ("warm", "#DDC8AA"),
              ("midtone", "#5F97B5"), ("foam", "#DCE9E8")]
    for seed, (name, col) in enumerate(stocks):
        s, c = paper(col, seed + 210, {"indigo": 7000, "cream": 11000}.get(name, 2600))
        if name in ("newspaper", "news_pale"):
            newspaper(c, "#7D7668" if name == "news_pale" else "#1E4C74")
        elif name in ("chart", "chart_pale"):
            sea_chart(c, "#898270" if name == "chart_pale" else "#245B80")
        elif name == "music":
            music(c)
        elif name in ("halftone", "dots_pale", "midtone"):
            c.set_source_rgba(*hx({"dots_pale": "#827A6D", "midtone": "#114673"}.get(name, "#092C53")),
                              {"dots_pale": 0.34, "midtone": 0.62}.get(name, 0.72))
            for y in range(0, SIZE, 17):
                for x in range((y // 17 % 2) * 8, SIZE, 17):
                    c.new_sub_path()
                    c.arc(x, y, 3.5, 0, 2 * math.pi)
            c.fill()
        elif name in ("graph", "graph_pale"):
            ink = "#A99B7E" if name == "graph_pale" else "#5D8EA9"
            for i in range(0, SIZE, 40):
                stroke(c, [(i, 0), (i, SIZE)], ink, 2 if i % 200 == 0 else 1, 0.65)
                stroke(c, [(0, i), (SIZE, i)], ink, 2 if i % 200 == 0 else 1, 0.65)
        elif name in ("desk", "boat"):
            r = np.random.default_rng(seed)
            for y in range(0, SIZE, 13 if name == "desk" else 110):
                pts = [(x, y + 4 * math.sin(x / 240 + y)) for x in range(0, SIZE + 20, 20)]
                stroke(c, pts, "#695B48", r.uniform(0.7, 2.2), 0.10 if name == "desk" else 0.32)
        elif name == "stamp":
            c.set_source_rgb(*hx("#D23832"))
            c.arc(1024, 930, 700, 0, 2 * math.pi)
            c.fill()
            c.set_line_width(9)
            c.rectangle(170, 170, 1708, 1708)
            c.stroke()
            text(c, "20", 240, 1750, 220, "#A8372D", "Futura")
            text(c, "海 / POST", 830, 1740, 120, "#A8372D", "Hiragino Sans GB")
            # Offset cancellation lines, not fake drop shadows.
            for j in range(4):
                stroke(c, [(x, 490 + j * 54 + 20 * math.sin(x / 110)) for x in range(1220, 2048, 8)],
                       "#3D4649", 7, 0.38)
        path = os.path.join(ART, name + ".png")
        s.write_to_png(path)
        print(path, flush=True)


if __name__ == "__main__":
    if sys.argv[1:] != ["art"]:
        raise SystemExit("Usage: python3 manim/scenes/style_collage.py art")
    art()
