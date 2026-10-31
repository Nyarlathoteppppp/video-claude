"""The Cycles plate of the outrun, played back off a tape: neon bloom, scanlines, colour bleed, tracking noise, the VCR's own text."""
import cairo
import numpy as np
from scipy.ndimage import gaussian_filter

from styles_common import FPS, H, W, add, canvas, glow, seq, surface


# The deck's own character generator: a 5 x 7 cell of square dots, no font in it. Digits and letters share their shapes, as on the chip.
GLYPHS = {
    "0": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": (".###.", "#...#", "....#", "..##.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "C": (".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    ".": (".", ".", ".", ".", ".", ".", "#"),
    ":": (".", ".", "#", ".", "#", ".", "."),
    " ": ("..", "..", "..", "..", "..", "..", ".."),
    ">": ("#......", "###....", "#####..", "#######", "#####..", "###....", "#......"),
}


def osd_width(s, cell):
    return sum(len(GLYPHS[ch][0]) + 1 for ch in s) * cell - cell


def osd(g, s, x, y, cell):
    """Set a line in the character generator, top left at (x, y): each dot one square of side `cell`."""
    for ch in s:
        rows = GLYPHS[ch]
        for j, row in enumerate(rows):
            for i, dot in enumerate(row):
                if dot == "#":
                    g.rectangle(x + i * cell, y + j * cell, cell, cell)
        x += (len(rows[0]) + 1) * cell
    g.fill()


def bloom(rgb):
    hot = np.clip(rgb[::2, ::2] - 0.5, 0, None) / 0.5
    b = gaussian_filter(hot, (3, 3, 0)) * 0.55 + gaussian_filter(hot, (11, 11, 0)) * 0.5 + gaussian_filter(hot, (38, 38, 0)) * 0.5
    b = np.repeat(np.repeat(b, 2, 0), 2, 1)[:rgb.shape[0], :rgb.shape[1]]
    return 1 - (1 - rgb) * (1 - np.clip(b, 0, 1))


def tape(rgb, t):
    """A VHS head: colour smeared and shifted sideways over a luma that stays nearly sharp, as on a real tape, a band of tracking noise
    wandering up the picture, scanlines over all of it."""
    f = int(round(t * FPS))
    rnd = np.random.default_rng(f + 5)
    out = rgb.copy()
    out[..., 0] = np.roll(rgb[..., 0], 3, axis=1)
    out[..., 2] = np.roll(rgb[..., 2], -3, axis=1)
    chroma = gaussian_filter(out, (0, 5, 0))
    luma = gaussian_filter(rgb.mean(axis=2, keepdims=True), (0, 0.9, 0))
    out = luma + (chroma - chroma.mean(axis=2, keepdims=True)) * 1.1
    y0 = int((1.25 - (t * 0.33) % 1.25) * H)
    for y in range(max(0, y0), min(H, y0 + 34)):
        out[y] = np.roll(out[y], int(rnd.normal(0, 30)), axis=0)
        speck = rnd.random(W) < 0.07
        out[y][speck] = 0.92
    wobble = (np.sin(np.arange(H) * 0.021 + t * 9) * 1.1).astype(int)
    if t > 2.84 and t < 3.05:                               # the moment the title arrives, the picture tears
        wobble += (rnd.normal(0, 16, H) * (np.arange(H) % 90 < 34)).astype(int)
    for y in np.nonzero(wobble)[0]:
        out[y] = np.roll(out[y], wobble[y], axis=0)
    out[::3] *= 0.9
    out += rnd.normal(0, 0.011, out.shape[:2])[..., None]
    return np.clip(out, 0, 1)


def synthwave(t):
    rgb = tape(bloom(seq("synthwave", t)), t)
    s = surface(rgb)
    c = cairo.Context(s)
    hud, g = canvas(None)
    frame = round(t * 30)
    clock = f"0:19:{24 + frame // 30:02d}:{frame % 30:02d}"
    g.set_antialias(cairo.ANTIALIAS_NONE)
    date = "OCT.30 1984"
    lines = (("PLAY", 86, 62, 6), (">", 242, 62, 6), ("SP", 86, H - 110, 5), (clock, 216, H - 110, 5), (date, W - 86 - osd_width(date, 5), H - 110, 5))
    for dx, colour in ((2, (0.05, 0.0, 0.12, 0.65)), (0, (0.96, 0.96, 0.96, 1.0))):      # a hard dark edge under the white, then the white
        g.set_source_rgba(*colour)
        for line, x, y, cell in lines:
            osd(g, line, x + dx, y + dx, cell)
    g.set_operator(cairo.OPERATOR_DEST_OUT)                  # the deck writes its text into the same scanlines as the picture
    g.set_source_rgba(0, 0, 0, 0.1)
    for y in range(0, H, 3):
        g.rectangle(0, y, W, 1)
    g.fill()
    add(c, glow(hud, ((2, 0.5), (7, 0.3))))
    c.set_source_surface(hud)
    c.paint()
    return s
