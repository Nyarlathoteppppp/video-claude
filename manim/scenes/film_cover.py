"""字成精's covers, made of the film's own pieces, at three times the frame's pixels and cut down, never blown up.

  A: the first frame's window (film_page.window, ink_mac's desktop and wallpaper): the greeting and the question typed in the box, the window
     smaller than the film's so the two sit close and the question reads at thumbnail size. It is the film's first frame come closer.
  B: the title page (film_title's last frame, hand/title_last.png): the poem, 字成精, the signature and the print. For 3:4 its ink is lifted off the
     page (the frame over film_hand.sheet(), the very paper, which it reproduces to a quarter of a level) and laid on a taller sheet of the same paper.

    python film_cover.py         # cover/cover_16x9_A.png, cover_3x4_A.png, cover_16x9_B.png, cover_3x4_B.png
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/render_work/宣传/manim/scenes"))
os.chdir(os.path.expanduser("~/render_work/宣传/manim/scenes"))
import numpy as np
from PIL import Image

import film_hand as hand
import film_page as ui
import film_ring as r
import ink_mac as mac
from ink_mac import H, PAD, W

HIGH = 3
OUT = os.path.join(r.FILM, "cover")
os.makedirs(OUT, exist_ok=True)


def compose(page, w, h, kz, cx, cy, crop_w, crop_h, out_size):
    """The desktop with a window of w by h points drawn kz frame pixels a point, its middle at (cx, cy) frame pixels; the crop_w by crop_h frame pixels
    round that middle, drawn at HIGH and brought down to out_size."""
    d = mac.desktop(app="Claude", clock=r.clock(0.0), show_dock=False)
    desk = np.asarray(Image.fromarray(d).resize((W * HIGH, H * HIGH), Image.LANCZOS), np.float32)
    layer = mac.snap(page, w + 2 * PAD, h + 2 * PAD, kz * HIGH).astype(np.float32)
    x, y = int(round((cx - (w / 2 + PAD) * kz) * HIGH)), int(round((cy - (h / 2 + PAD) * kz) * HIGH))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + layer.shape[1], desk.shape[1]), min(y + layer.shape[0], desk.shape[0])
    part = layer[y0 - y:y1 - y, x0 - x:x1 - x]
    desk[y0:y1, x0:x1] = desk[y0:y1, x0:x1] * (1 - part[..., 3:] / 255) + part[..., :3] * part[..., 3:] / 255
    box = [int(round(v * HIGH)) for v in (cx - crop_w / 2, cy - crop_h / 2, cx + crop_w / 2, cy + crop_h / 2)]
    assert box[0] >= 0 and box[1] >= 0 and box[2] <= W * HIGH and box[3] <= H * HIGH, box
    return Image.fromarray(np.uint8(np.clip(desk + .5, 0, 255))).crop(box).resize(out_size, Image.LANCZOS)


def cover_a():
    # 16:9: a window 560 by 300 points at the film's own 1.8 pixels a point, the crop a little wider than the window, clear of the menu bar
    page = ui.window(box=r.ASK, typing=True, greeting=True, w=560, h=300, top=r.TOP, box_w=480)
    compose(page, 560, 300, 1.8, 960, 560, 1150, 1150 * 9 / 16, (1920, 1080)).save(os.path.join(OUT, "cover_16x9_A.png"))
    # 3:4: a narrower, taller window, at 1.5 pixels a point so that it fits the frame's height with the forest round it
    page = ui.window(box=r.ASK, typing=True, greeting=True, w=450, h=440, top=r.TOP, box_w=410)
    compose(page, 450, 440, 1.5, 960, 562, 760, 760 * 4 / 3, (1080, 1440)).save(os.path.join(OUT, "cover_3x4_A.png"))


def cover_b():
    title = np.asarray(Image.open(os.path.join(hand.OUT, "title_last.png")).convert("RGB"), np.float32)
    Image.fromarray(np.uint8(title)).save(os.path.join(OUT, "cover_16x9_B.png"))
    paper = hand.sheet().astype(np.float32)[..., :3]
    ink = np.clip(title / np.maximum(paper, 1), 0, 1)                       # what the brush and the print did to the paper, as a multiplier
    dark = (ink.min(2) < .9)
    rows = np.nonzero(dark.any(1))[0]
    # the blocks of rows with ink: the poem's four, then the name with the signature and the print
    runs, start = [], rows[0]
    for a, b in zip(rows[:-1], rows[1:]):
        if b - a > 12:
            runs.append((start, a + 1))
            start = b
    runs.append((start, rows[-1] + 1))
    poem, name = (runs[0][0], runs[3][1]), (runs[4][0], runs[-1][1])
    def block(y0, y1):
        cols = np.nonzero(dark[y0:y1].any(0))[0]
        return ink[y0:y1, cols[0]:cols[-1] + 1]
    P, N = block(*poem), block(*name)
    print("runs", runs, "poem", P.shape, "name", N.shape, flush=True)
    w, h = 1080, 1440
    sheet = hand.sheet(w * 2, h * 2, seed=92).astype(np.float32)[..., :3]
    def lay(dst, m, x, y, s):
        m = np.stack([np.asarray(Image.fromarray(m[..., i], "F").resize((round(m.shape[1] * s), round(m.shape[0] * s)), Image.LANCZOS)) for i in range(3)], -1)
        dst[y:y + m.shape[0], x:x + m.shape[1]] *= np.clip(m, 0, 1)
        return m.shape
    s1 = 1840 / P.shape[1]                                     # the poem the width of the sheet less 120 px a side (at twice the size)
    s2 = s1 * 1.25                                             # the name a quarter larger than it stood under the poem
    hp, hn = P.shape[0] * s1, N.shape[0] * s2
    gap = 170
    top = (h * 2 - (hp + gap + hn)) / 2
    lay(sheet, P, round((w * 2 - P.shape[1] * s1) / 2), round(top), s1)
    lay(sheet, N, round((w * 2 - N.shape[1] * s2) / 2), round(top + hp + gap), s2)
    Image.fromarray(np.uint8(np.clip(sheet + .5, 0, 255))).resize((w, h), Image.LANCZOS).save(os.path.join(OUT, "cover_3x4_B.png"))


def sheet_of():
    """All four side by side, and as small as a phone shows them (180 px wide) and a desktop feed (360), to judge them at thumbnail size."""
    names = ("cover_16x9_A", "cover_16x9_B", "cover_3x4_A", "cover_3x4_B")
    ims = [Image.open(os.path.join(OUT, n + ".png")).convert("RGB") for n in names]
    rows = []
    for wpx in (180, 360):
        row = Image.new("RGB", (sum(round(wpx * (1 if i.width > i.height else .75)) + 20 for i in ims), round(wpx * 4 / 3 * .75) + 20), (40, 40, 40))
        x = 10
        for im in ims:
            tw = round(wpx * (1 if im.width > im.height else .75))
            t = im.resize((tw, round(tw * im.height / im.width)), Image.LANCZOS)
            row.paste(t, (x, 10))
            x += tw + 20
        rows.append(row)
    full = Image.new("RGB", (max(r_.width for r_ in rows), sum(r_.height for r_ in rows)), (40, 40, 40))
    y = 0
    for r_ in rows:
        full.paste(r_, (0, y))
        y += r_.height
    full.save(os.path.join(OUT, "thumbs.png"))


if __name__ == "__main__":
    cover_a()
    cover_b()
    sheet_of()
    print(OUT, flush=True)
