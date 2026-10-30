"""The README pictures, rendered straight from the film renderers: for episode 2 a strip of key frames, an annotated
frame and a GIF of the closing pull-back; for episode 4 one frame per chapter stage and the short's formula menu.
Needs the same inputs as the renderers (images, beat tables).

    python docs/make_img.py                 # all four -> docs/img/
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


def to_image(s):
    a = np.ndarray((s.get_height(), s.get_width(), 4), np.uint8, s.get_data(), strides=(s.get_stride(), 4, 1))
    return Image.fromarray(a[:, :, [2, 1, 0]].copy())


def frame(t):
    import ai_history_film as F
    return to_image(F.render_frame(t))


def quant_frame(t, short=False):
    import quant_film as Q
    return to_image((Q.short_frame if short else Q.render_frame)(t))


def gif_frame(k_t):
    k, t = k_t
    frame(t).resize((360, 640), Image.LANCZOS).save(os.path.join(TMP, f"f{k:04d}.png"))


def sheet(shots, grab, w, h, cols, name):
    """Captioned frames in a grid -> docs/img/<name>.jpg."""
    gap, cap = 12, 46
    rows = (len(shots) + cols - 1) // cols
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, cols * (w + gap) + gap, rows * (h + cap) + gap)
    c = cairo.Context(surf)
    c.set_source_rgb(0.07, 0.06, 0.05); c.paint()
    for i, (t, label) in enumerate(shots):
        im = grab(t).resize((w, h), Image.LANCZOS).convert("RGBA")
        b = np.array(im)[:, :, [2, 1, 0, 3]].copy()
        img = cairo.ImageSurface.create_for_data(memoryview(b), cairo.FORMAT_ARGB32, w, h, w * 4)
        x, y = gap + (i % cols) * (w + gap), gap + (i // cols) * (h + cap)
        c.set_source_surface(img, x, y); c.paint()
        c.select_font_face("Songti SC", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD); c.set_font_size(22)
        ext = c.text_extents(label)
        c.set_source_rgb(*GOLD); c.move_to(x + (w - ext.x_advance) / 2, y + h + 32); c.show_text(label)
    p = os.path.join(OUT, name + ".png")
    surf.write_to_png(p)
    Image.open(p).convert("RGB").save(p.replace(".png", ".jpg"), quality=88)
    os.remove(p)


def strip():
    shots = [(3.0, "1943　起点"), (46, "第一次寒冬"), (127, "2012　AlexNet"), (196.5, "倒数 GPT-3.5"),
             (246, "2025　奥数金牌"), (309.5, "79 年 vs 1418 天")]
    sheet(shots, frame, 300, 533, len(shots), "frames_strip")


def quant():
    """Episode 4: each chapter's stage on its strongest beat, and the short's formula menu cropped 3:4 for its card."""
    shots = [(25.9, "终端：规则写成程序"), (36.1, "股票墙：5000 只一起看"), (104.8, "胶片机：回测与过拟合"),
             (118.3, "筹码桌：十次看运气"), (184.0, "撮合口：谁是对手"), (241.0, "公式墙：片尾卡点")]
    sheet(shots, quant_frame, 600, 338, 3, "quant_stages")
    menu = quant_frame(251.0, short=True).crop((0, 220, 1080, 1660)).resize((540, 720), Image.LANCZOS)
    menu.save(os.path.join(OUT, "quant_menu.jpg"), quality=88)


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
    what = sys.argv[1:] or ["strip", "anatomy", "gif", "quant"]
    for w in what:
        globals()[w]()
        print("done", w)
