"""The README pictures, rendered straight from the film renderer: a strip of key frames, an annotated frame, and a GIF
of the closing pull-back. Needs the same inputs as the renderer (images, beat table).

    python docs/make_img.py                 # all three -> docs/img/
    python docs/make_img.py anatomy         # just one
"""
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import cairo
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = os.path.join(HERE, "..", "manim", "scenes")
sys.path.insert(0, SCENES)
OUT = os.path.join(HERE, "img")
TMP = os.path.join(tempfile.gettempdir(), "video_claude_gif")
GOLD = (0.93, 0.78, 0.45)


def frame(t):
    import ai_history_film as F
    s = F.render_frame(t)
    a = np.ndarray((F.H, F.W, 4), np.uint8, s.get_data(), strides=(s.get_stride(), 4, 1))
    return Image.fromarray(a[:, :, [2, 1, 0]].copy())


def gif_frame(k_t):
    k, t = k_t
    frame(t).resize((360, 640), Image.LANCZOS).save(os.path.join(TMP, f"f{k:04d}.png"))


def strip():
    shots = [(3.0, "1943　起点"), (46, "第一次寒冬"), (127, "2012　AlexNet"), (196.5, "倒数 GPT-3.5"),
             (246, "2025　奥数金牌"), (309.5, "79 年 vs 1418 天")]
    w, h, gap, cap = 300, 533, 12, 46
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, len(shots) * (w + gap) + gap, h + cap + gap)
    c = cairo.Context(surf)
    c.set_source_rgb(0.07, 0.06, 0.05); c.paint()
    for i, (t, label) in enumerate(shots):
        im = frame(t).resize((w, h), Image.LANCZOS).convert("RGBA")
        b = np.array(im)[:, :, [2, 1, 0, 3]].copy()
        img = cairo.ImageSurface.create_for_data(memoryview(b), cairo.FORMAT_ARGB32, w, h, w * 4)
        x = gap + i * (w + gap)
        c.set_source_surface(img, x, gap); c.paint()
        c.select_font_face("Songti SC", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD); c.set_font_size(22)
        ext = c.text_extents(label)
        c.set_source_rgb(*GOLD); c.move_to(x + (w - ext.x_advance) / 2, gap + h + 32); c.show_text(label)
    p = os.path.join(OUT, "frames_strip.png")
    surf.write_to_png(p)
    Image.open(p).convert("RGB").save(p.replace(".png", ".jpg"), quality=88)
    os.remove(p)


def anatomy():
    t = 200.5
    im = frame(t).convert("RGBA")
    b = np.array(im)[:, :, [2, 1, 0, 3]].copy()
    surf = cairo.ImageSurface.create_for_data(memoryview(b), cairo.FORMAT_ARGB32, im.width, im.height, im.width * 4)
    c = cairo.Context(surf)
    marks = [  # number, box (x0, y0, x1, y1) or None, badge position
        (1, (35, 72, 540, 790), (520, 60)),          # info window
        (2, (725, 128, 1055, 325), (1040, 115)),     # clock and unit
        (3, (543, 348, 1016, 614), (1000, 340)),     # picture
        (4, None, (760, 790)),                       # head and impact ring
        (5, None, (300, 1040)),                      # echo arc
        (6, (694, 985, 1044, 1066), (1030, 975)),    # flash
        (7, (50, 1565, 715, 1622), (700, 1555)),     # time axis
        (8, (430, 1765, 955, 1848), (940, 1755)),    # background formula
    ]
    for n, box, (bx, by) in marks:
        if box:
            x0, y0, x1, y1 = box
            c.set_source_rgba(*GOLD, 0.9); c.set_line_width(3); c.set_dash([12, 8])
            c.rectangle(x0, y0, x1 - x0, y1 - y0); c.stroke(); c.set_dash([])
        c.set_source_rgb(*GOLD); c.arc(bx, by, 26, 0, 2 * math.pi); c.fill()
        c.set_source_rgb(0.08, 0.06, 0.04)
        c.select_font_face("Times New Roman", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD); c.set_font_size(34)
        e = c.text_extents(str(n))
        c.move_to(bx - e.x_bearing - e.width / 2, by - e.y_bearing - e.height / 2); c.show_text(str(n))
    surf.flush()
    a = np.ndarray((im.height, im.width, 4), np.uint8, surf.get_data(), strides=(surf.get_stride(), 4, 1))
    Image.fromarray(a[:, :, [2, 1, 0]].copy()).resize((720, 1280), Image.LANCZOS).save(os.path.join(OUT, "anatomy.jpg"), quality=88)


def gif(a=290.5, b=301.0, fps=10, out_fps=8, size="300:533", colors=96):
    os.makedirs(TMP, exist_ok=True)
    ts = [(k, a + k / fps) for k in range(int((b - a) * fps))]
    with Pool(8) as pool:
        pool.map(gif_frame, ts)
    pal = os.path.join(TMP, "pal.png")
    src = os.path.join(TMP, "f%04d.png")
    scale = f"fps={out_fps},scale={size}:flags=lanczos"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", src,
                    "-vf", f"{scale},palettegen=max_colors={colors}:stats_mode=diff", pal], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", src, "-i", pal,
                    "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle", "-loop", "0",
                    os.path.join(OUT, "ending.gif")], check=True)   # about 2.5 MB

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    what = sys.argv[1:] or ["strip", "anatomy", "gif"]
    for w in what:
        globals()[w]()
        print("done", w)
