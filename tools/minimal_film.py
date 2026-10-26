"""The idea behind manim/scenes/ai_history_film.py in about 80 lines: a glowing line that grows from 1943 to 2026.

Every frame is a pure function of time, render_frame(t): draw it with cairo (lines, dots, text), add the glow with
numpy/scipy (blur a thick copy of the line), and pipe the raw pixels to ffmpeg. Change the maths in points() and
render_frame() and you have your own film.

    python tools/minimal_film.py out.mp4 [--seconds 6]          # needs pycairo, numpy, scipy and ffmpeg on PATH
"""
import argparse
import math
import subprocess

import cairo
import numpy as np
from scipy.ndimage import gaussian_filter

W, H, FPS = 540, 960, 30
GOLD = (0.96, 0.82, 0.50)


def points(progress, n=240):
    """The line up to `progress` (0..1): flat for a long time, then steep."""
    xs = np.linspace(0, progress, max(2, int(n * progress)))
    return [(40 + x * (W - 80), H * 0.8 - (math.exp(5 * x) - 1) / (math.exp(5) - 1) * H * 0.55) for x in xs]


def glow_layer(pts, s=3):
    """Draw a thick copy of the line at 1/s size, blur it twice, scale it back up: a soft light under the line."""
    w, h = W // s, H // s
    mask = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
    c = cairo.Context(mask)
    c.scale(1 / s, 1 / s); c.set_line_width(10)
    c.move_to(*pts[0])
    for p in pts[1:]:
        c.line_to(*p)
    c.stroke(); mask.flush()
    a = np.ndarray((h, mask.get_stride()), np.uint8, mask.get_data())[:, :w] / 255.0
    g = np.clip(gaussian_filter(a, 3) * 0.9 + gaussian_filter(a, 10) * 1.4, 0, 1)
    g = np.pad(np.repeat(np.repeat(g, s, 0), s, 1), ((0, H - h * s), (0, W - w * s)), mode="edge")
    bgra = np.empty((H, W, 4), np.uint8)                       # cairo ARGB32 is BGRA in memory, premultiplied
    for i, ch in enumerate((GOLD[2], GOLD[1], GOLD[0], 1.0)):
        bgra[..., i] = g * ch * 255
    return cairo.ImageSurface.create_for_data(memoryview(bgra), cairo.FORMAT_ARGB32, W, H)


def render_frame(t, dur):
    p = min(1.0, t / (dur * 0.85))                             # how far the line has grown
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    c = cairo.Context(s)
    c.set_source_rgb(0.05, 0.04, 0.03); c.paint()             # 1. blank canvas
    pts = points(p)
    c.set_source_surface(glow_layer(pts)); c.paint()           # 2. the glow goes under the line
    c.set_source_rgb(*GOLD); c.set_line_width(2.5)             # 3. the line itself (cairo antialiases it)
    c.move_to(*pts[0])
    for q in pts[1:]:
        c.line_to(*q)
    c.stroke()
    x, y = pts[-1]                                             # 4. the head: three dots, faint to bright
    for r, alpha in ((22, 0.12), (10, 0.3), (4, 1.0)):
        c.set_source_rgba(1, 0.95, 0.8, alpha); c.arc(x, y, r, 0, 2 * math.pi); c.fill()
    c.select_font_face("Times New Roman"); c.set_font_size(64)    # 5. text: the clock
    c.set_source_rgb(0.95, 0.9, 0.8); c.move_to(W - 210, 110); c.show_text(str(int(1943 + p * (2026 - 1943))))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args()
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", a.out],
                          stdin=subprocess.PIPE)
    for n in range(int(a.seconds * FPS)):
        ff.stdin.write(bytes(render_frame(n / FPS, a.seconds).get_data()))   # one frame of raw pixels
    ff.stdin.close(); ff.wait()
    print(a.out)


if __name__ == "__main__":
    main()
