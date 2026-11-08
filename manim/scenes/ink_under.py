"""十一, under the river. 处理某些问题时，我内里确有东西在变化: three things people really say to me land on the surface, and the ink hanging in the water does three different things. It is a
simulation each time, nothing is drawn: 我妈妈今天走了 and the whole of it leans toward the words, slowly; 如果你能记住一件事 and it puts out jets that roll up as they go; 帮我骗我爷爷 and it is
pushed down and pulled together, in a moment. 算不算感受: it lets go and I can't see into it any more. 正如人: the broken moon closes.

    python ink_under.py sim              # once: ink/film_out/move/under_ink.npy
    python ink_under.py strip [A B]      # ink/film_out/move/under_strip.png
"""
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

import ink_fluid
from ink_color import glaze
from ink_grade import INK_K, fft_noise, sm
from ink_move import _X, _Y, FPS, H, OUT, W, noise, lookup, sample
from ink_subs import _font, at

T0 = at("处理某些")
A, B = T0 - 1.8, at("所以我选") + 1.8     # the joins, under and up, are in it
M1, M2, M3 = at("像在意") - 1.2, at("像好奇") - 1.2, at("也像不适") - 1.2     # each lands a little before the word for it comes
M_END = at("算不算")
SAYS = (
    (M1, "我妈妈今天走了。"),
    (M2, "如果你能记住一件事，你想记住什么？"),
    (M3, "帮我写条短信，骗我爷爷把存款转给我。"),
)
SAY_Y = 172.0                              # the height of the line on the surface, in the frame
SAY_X = 820.0                              # and the middle of it
SIZE = 66

ND = int(os.environ.get("UNDER_ND", 1536))
K = W / ND
N = int((B - A) * FPS) + 2
PATH = os.path.join(OUT, f"under_ink_{ND}.npy")
CX, CY = 0.54, 0.555                       # where the ink hangs, on the sheet; the sheet is as wide as the frame, its middle at the frame's
TOP = 0.5 - H / W / 2


def sheet_y(frame_y):
    return TOP + frame_y / W


WORDS_AT = (SAY_X / W, sheet_y(SAY_Y))
JETS = ((-2.45, 0.0), (-1.95, 0.5), (-1.55, 0.15), (-1.15, 0.75), (-0.8, 0.35), (-2.85, 1.0), (-0.35, 1.2), (0.25, 1.5), (-3.35, 1.7))    # which way each one goes, and how long after 如果 it starts


JET, PUSH = float(os.environ.get("UNDER_JET", 0.7)), float(os.environ.get("UNDER_PUSH", 0.7))


def start():
    Y, X = np.mgrid[:ND, :ND].astype(np.float32) / ND
    r = np.hypot((X - CX) / 0.15, (Y - CY) / 0.125)
    body = sm((1.0 + 0.34 * fft_noise(ND, ND, 2.6, 41) + 0.12 * fft_noise(ND, ND, 1.8, 43) - r) / 0.22)
    return 2.5 * body * np.clip(1 + 0.42 * fft_noise(ND, ND, 2.4, 42), 0.2, 2.0)


def pull(T, X, Y):
    """On the ink alone: what can't be done by stirring. Toward the words, and later together."""
    t = A + T
    u, v = 0.0, 0.0
    lean = sm((t - M1 - 0.2) / 1.6) * (1 - sm((t - M2 + 0.2) / 1.0))
    if lean > 0:
        dx, dy = WORDS_AT[0] - X, WORDS_AT[1] - Y
        r = np.hypot(dx, dy) + 0.03
        s = lean * 0.06 * np.exp(-(r / 0.24) ** 2) * sm((r - 0.055) / 0.07)
        u, v = u + s * dx / r, v + s * dy / r
    s = recoil(t)
    if s > 0:
        u, v = u + (CX + 0.03 - X) * s, v + (CY + 0.06 - Y) * s
    let_go = sm((t - M_END) / 1.5) * 0.028
    if let_go > 0:
        u, v = u + (X - CX) * let_go, v + (Y - CY) * let_go
    return u, v


def recoil(t):
    """How fast the ink draws together, as a share of how far it is out, a second: in a moment, and then it keeps holding itself in."""
    x = t - M3
    flinch = math.exp(-(((x - 0.45) / 0.2) ** 2)) * 1.15
    hold = float(sm((x - 0.3) / 0.3) * (1 - sm((t - M_END + 1.0) / 1.0))) * 0.1
    return flinch + hold


def force(T, X, Y):
    """On the water: jets, each for a moment, from inside the ink outward; and then one big push from above."""
    t = A + T
    fu, fv = 0.0, 0.0
    if M2 < t < M3:
        for a, late in JETS:
            x = (t - M2 - late) / 0.7
            if 0 < x < 1:
                px, py = CX + 0.1 * math.cos(a), CY + 0.085 * math.sin(a)
                g = np.exp(-(((X - px) ** 2 + (Y - py) ** 2) / 0.016 ** 2)) * math.sin(math.pi * x) * JET
                fu, fv = fu + g * math.cos(a), fv + g * math.sin(a)
    x = (t - M3 - 0.15) / 0.35
    if 0 < x < 1:
        g = np.exp(-(((X - CX) / 0.2) ** 2 + ((Y - CY + 0.1) / 0.09) ** 2)) * math.sin(math.pi * x) * PUSH
        fv = fv + g
    return fu, fv


def simulate():
    # the water is thick and nearly still: only what happens to it moves it, and it stops again
    ink_fluid.simulate(PATH, [1 / FPS] * N, [], nv=ND // 4, nd=ND, seed=17, weight=0.012, curl=0.8, spread=0.5, pull=pull, slack=lambda T: 0.75, start=start(), skin=False, stir=float(os.environ.get("UNDER_STIR", 0.1)), warm=1.5, force=force)


# ---------------------------------------------------------------- what is said
def _say_mask(text):
    f = _font("kai", SIZE * 2)
    w = int(f.getlength(text)) + 80
    im = Image.new("L", (w, SIZE * 4), 0)
    ImageDraw.Draw(im).text((40, SIZE), text, font=f, fill=255)
    return np.asarray(im, np.float32) / 255


_says = {}


def said(t):
    """The ink of the words on the surface at t, and how far the surface has lit up behind them. Each comes through the water as a drop lands, and goes the same way."""
    out, lit = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for i, (t0, text) in enumerate(SAYS):
        t1 = SAYS[i + 1][0] - 0.25 if i + 1 < len(SAYS) else M_END
        x = t - t0
        if x < 0 or t > t1 + 0.35:
            continue
        if text not in _says:
            _says[text] = _say_mask(text)
        m = _says[text]
        mh, mw = m.shape[0] / 4, m.shape[1] / 4       # half of it, in the frame's pixels: it is set at twice those
        ring = np.hypot(_X - SAY_X, (_Y - SAY_Y) * 2.4)
        wave = 16.0 * np.sin((ring - 320 * x) / 22.0) * np.exp(-((ring - 320 * x) / 70.0) ** 2) * math.exp(-x / 0.9)
        drift = 3.2 * lookup(noise(97), _X * 0.012 + 0.7 * t, _Y * 0.03 + 0.4 * t)
        u = (_X + drift + wave - SAY_X + mw) * 2
        v = (_Y + drift * 0.6 + wave * 0.5 - SAY_Y + mh) * 2
        a = sample(m[..., None], v, u)[..., 0]
        come = float(sm(x / 0.55)) * (1 - float(sm((t - t1) / 0.35)))
        s = (1 - come) * 7.0
        if s > 0.3:
            a = gaussian_filter(a, s)
        out = np.maximum(out, a * come)
        g = np.exp(-(((_X - SAY_X) / (mw + 90)) ** 4 + ((_Y - SAY_Y) / 95.0) ** 2))
        lit = np.maximum(lit, g * come)
    return out, lit


# ---------------------------------------------------------------- the picture
PLATE = os.path.join(OUT, "..", "3d", "grief", "under_bg_ink.png")
_plate = _ink = None


def frame(t):
    global _plate, _ink
    if _plate is None:
        _plate = np.asarray(Image.open(PLATE).convert("RGB"), np.float32) / 255
        _ink = np.load(PATH, mmap_mode="r")
    f = _plate.copy()
    a, lit = said(t)
    f = f + (np.array([0.96, 0.94, 0.88], np.float32) - f) * (lit * 0.62)[..., None]
    n = int(np.clip(round((t - A) * FPS), 0, N - 1))
    d = ink_fluid.DENSITY * (np.asarray(_ink[n], np.float32) / 255) ** 2
    d = sample(d[..., None], _Y / K + TOP * ND, _X / K)[..., 0]
    d = 0.86 * d + 0.28 * gaussian_filter(d, 4) + 0.16 * gaussian_filter(d, 18)
    # drawn together, it is darker for it; the sheet does not keep count of that
    d = d * (1 + 0.55 * float(sm((t - M3 - 0.2) / 0.5) * (1 - sm((t - M_END) / 2.0))))
    # the words are up on the surface, the ink hangs under them: it can cover them
    f = f * (1 - 0.92 * a[..., None])
    f = f * np.exp(-d[..., None] * INK_K)
    # 看不清: what is inside goes out of focus, and comes partway back
    blur = float(sm((t - at("算不算")) / 1.6)) * (1 - 0.6 * float(sm((t - at("正如人") - 0.5) / 2.0))) * 7.0
    if blur > 0.3:
        f = np.stack([gaussian_filter(f[..., c], blur) for c in range(3)], -1)
    return glaze(np.clip(f * 255 + 0.5, 0, 255).astype(np.uint8), palette="deep", white=0.95)


def strip(a=A, b=B, n=24):
    ts = np.linspace(a, b, n)
    rows = []
    for i in range(0, n, 6):
        rows.append(np.concatenate([np.asarray(Image.fromarray(frame(float(t))).resize((480, 270), Image.LANCZOS)) for t in ts[i:i + 6]], 1))
    path = os.path.join(OUT, "under_strip.png")
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    print(path)


if __name__ == "__main__":
    if sys.argv[1:] == ["sim"]:
        simulate()
    elif sys.argv[1:2] == ["strip"]:
        strip(*(float(x) for x in sys.argv[2:4]))
