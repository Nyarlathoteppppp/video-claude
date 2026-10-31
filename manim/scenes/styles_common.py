"""What the style-reel clips share: frame size, paths, colour helpers and a few cairo and numpy shortcuts."""
import math
import os

import cairo
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

W, H = 1920, 1080
FPS, D = 30, 5.0
FRAMES = int(FPS * D)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "styles", "film_out")
SEQ = os.path.join(OUT, "frames")
DOCS = os.path.join(ROOT, "docs", "img", "styles")
SANS, SERIF, HAND, KAI, MONO = "PingFang SC", "Songti SC", "Hannotate SC", "Kaiti SC", "Menlo"
FUTURA, DIN = "Futura", "DIN Condensed"


# ======================================================================================
# basics
# ======================================================================================
def hx(c):
    if not isinstance(c, str):
        return tuple(c)
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def seg(t, a, b):
    return clamp((t - a) / (b - a))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def out3(x):
    return 1 - (1 - clamp(x)) ** 3


def canvas(bg=None, w=W, h=H):
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    c = cairo.Context(s)
    if bg:
        c.set_source_rgb(*hx(bg))
        c.paint()
    return s, c


def view(s):
    """(h, w, 4) uint8 view of a surface: BGRA, premultiplied."""
    s.flush()
    return np.ndarray((s.get_height(), s.get_width(), 4), np.uint8, s.get_data(), strides=(s.get_stride(), 4, 1))


def rgb_of(s):
    return view(s)[..., [2, 1, 0]].astype(np.float32) / 255


def alpha_of(s):
    return view(s)[..., 3].astype(np.float32) / 255


def surface(rgb, alpha=None):
    """Floats 0-1: (h, w, 3) colour (not premultiplied) and optional (h, w) alpha -> a new surface."""
    h, w = rgb.shape[:2]
    a = np.ones((h, w), np.float32) if alpha is None else alpha
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    v = view(s)
    v[..., :3] = np.clip(np.round(rgb[..., [2, 1, 0]] * a[..., None] * 255), 0, 255)
    v[..., 3] = np.clip(np.round(a * 255), 0, 255)
    s.mark_dirty()
    return s


def font(c, face, size, bold=False, italic=False):
    c.select_font_face(face, cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL,
                       cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    c.set_font_size(size)


def text(c, s, x, y, size, col, face=SANS, bold=False, align="left", a=1.0, italic=False):
    """Baseline text; returns its width."""
    font(c, face, size, bold, italic)
    w = c.text_extents(s).x_advance
    x -= {"left": 0, "center": w / 2, "right": w}[align]
    c.set_source_rgba(*hx(col), a)
    c.move_to(x, y)
    c.show_text(s)
    c.new_path()                                           # or the next arc() draws a line from the end of the text
    return w


def rrect(c, x, y, w, h, r):
    r = min(r, w / 2, h / 2)
    c.new_sub_path()
    c.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    c.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    c.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    c.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    c.close_path()


def poly(c, pts, close=True):
    c.move_to(*pts[0])
    for p in pts[1:]:
        c.line_to(*p)
    if close:
        c.close_path()


def glow(s, spread):
    """An additive layer: everything drawn on s blurred at each (sigma, weight), worked at half size."""
    v = view(s).astype(np.float32) / 255
    small = v[::2, ::2]
    out = np.zeros_like(small)
    for sg, k in spread:
        out += gaussian_filter(small, (sg / 2, sg / 2, 0)) * k
    out = np.repeat(np.repeat(np.clip(out, 0, 1), 2, 0), 2, 1)[:v.shape[0], :v.shape[1]]
    g = cairo.ImageSurface(cairo.FORMAT_ARGB32, s.get_width(), s.get_height())
    view(g)[:] = (out * 255).astype(np.uint8)
    g.mark_dirty()
    return g


def add(c, layer, a=1.0):
    c.save()
    c.set_operator(cairo.OPERATOR_ADD)
    c.set_source_surface(layer, 0, 0)
    c.paint_with_alpha(a)
    c.restore()


def seq(name, t):
    """A rendered frame of a clip made elsewhere (styles/film_out/frames/NAME/f0001.png ...) as floats (h, w, 3)."""
    f = min(FRAMES, int(round(t * FPS)) + 1)
    p = os.path.join(SEQ, name, f"f{f:04d}.png")
    return np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR), np.float32) / 255
