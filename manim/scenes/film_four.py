"""四's one cut away from the room: 深夜里没人回复的帖子. Her post again, the third time it is seen, and this time only its foot: what it was read, nobody's comment, and the box where a reply would go, empty, the caret in it blinking. Nothing moves and nobody comes. Then a hard cut to 五's window, where a reply is about to begin: its caret stands exactly where this one did, the same height and the same width.

    python film_four.py pair              # four/four_last.png, four/five_first.png (a proposal for 五's first frame) and four/pair.png
    python film_four.py clip              # four/four.mp4, the shot as the cut plays it (no subtitle burned), 1920 by 1080, 30 fps

The page is film_weibo's, untouched but for the caret, in the same Safari window on the same desktop. The camera is closer than 十一's: so close that a point of her page is as many pixels as a point of the Claude window is in film_ring's frame times the ratio of the two boxes' heights, and the carets come out the same size.
"""
import ast
import os
import subprocess
import sys

import numpy as np
from PIL import Image

import film_page as page
import film_ring as ring
import film_weibo as wb
import ink_mac as mac
import ink_subs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "four")
W, H, FPS = 1920, 1080, 30
BLINK = 1.06                        # the Mac's caret: on and off together, and how much of that it is on (film_room_ui)
ON = 1.15 / 2
CARET_PT = (1.3, 12.7)              # the caret in her box, in points of her page: as wide and as tall in the frame as 五's is in its first frame (learn/five_first.png)

# 五's window, as film_ring draws every Claude window: the question asked, the title in the bar, the reply not yet begun
FIVE_CLOCK = "10月30日 周五 20:13"


def five_ask():
    """The question 五 answers, read out of ink_learn without importing it."""
    with open(os.path.join(HERE, "ink_learn.py"), encoding="utf-8") as f:
        for node in ast.parse(f.read()).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ASK":
                return ast.literal_eval(node.value)
    raise ValueError("no ASK")


LEARN_FIRST = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "learn", "five_first.png")     # 五's real first frame, as its crew renders it


def five_first(caret=True):
    """The frame the cut lands in: 五's own first frame when its crew has written it (its window at film_weibo.PLACE, the reply's caret in the chat column), else the proposal: film_ring's window, the question sent, my reply about to start, its caret blinking where the words will come."""
    if caret and os.path.exists(LEARN_FIRST):
        return np.asarray(Image.open(LEARN_FIRST).convert("RGB").resize((W, H), Image.LANCZOS))
    ask = five_ask()
    body = page.window(page.you(ask) + page.para("", page.caret() if caret else ""), title=ask.rstrip("？?"), w=ring.WIN_W, h=ring.WIN_H, top=ring.TOP)
    return np.asarray(Image.fromarray(ring._frame(body, 2, FIVE_CLOCK)).resize((W, H), Image.LANCZOS))


def find_caret(f, tall=(20, 60)):
    """Where in the whole frame a caret stands: the dark upright between tall[0] and tall[1] pixels high, no more than four wide, with light either side of it all the way down, and nothing dark just above or below. Returns a region round it for caret_box."""
    a = 1 - f.astype(np.float32).mean(2) / 255
    dark = a > 0.5
    h, w = dark.shape
    best = None
    for x in range(4, w - 4):
        col = dark[:, x]
        if not col.any():
            continue
        edges = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
        for top, bottom in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
            n = bottom - top
            if not tall[0] <= n <= tall[1]:
                continue
            side = max(a[top:bottom, x - 4].mean(), a[top:bottom, x + 4].mean())
            ends = max(a[max(top - 4, 0), x - 1:x + 2].max(), a[min(bottom + 3, h - 1), x - 1:x + 2].max())
            wide = dark[top:bottom, x - 3:x + 4].mean(0)
            if side < 0.12 and ends < 0.3 and (wide > 0.5).sum() <= 4:
                score = side + ends
                if best is None or score < best[0]:
                    best = (score, x, top, bottom)
    if best is None:
        raise ValueError("no caret in the frame")
    _, x, top, bottom = best
    return (x - 30, top - 20, x + 30, bottom + 20)


def caret_box(f, region=(250, 200, 500, 420)):
    """Where a caret stands in a frame: the thin dark upright in the region (x0, y0, x1, y1), as left, top, right, bottom in frame pixels, to a tenth of a pixel by its darkness. region None: wherever find_caret finds it."""
    x0, y0, x1, y1 = region if region is not None else find_caret(f)
    a = 1 - f[y0:y1, x0:x1].astype(np.float32).mean(2) / 255
    tall = (a > 0.5).sum(0)
    cols = np.nonzero(tall > 12)[0]
    if not len(cols):
        raise ValueError("no caret")
    c0, c1 = cols.min() - 1, cols.max() + 2
    strip = a[:, c0:c1]
    rows = np.nonzero(strip.max(1) > 0.5)[0]
    w = strip.sum(1)
    xs = (strip * np.arange(c0, c1)[None]).sum() / strip.sum()
    return dict(x=x0 + xs, top=y0 + rows.min(), bottom=y0 + rows.max() + 1, width=float(w[rows].mean()))


class Post:
    """Her page in a Safari window on the desktop, scrolled down to its foot, seen through a camera at scale frame pixels a point whose frame puts the caret's middle at at. The window's bar just shows along the top of the frame, weibo.com in it, and under it the page starts at the line of what the post was read: what she wrote is scrolled away above. The window stands on the screen where the frame needs it, as a window can: nothing of the screen's edge or its menu bar is seen."""

    WIN_X, WIN_Y = 120.0, 40.0       # the window's top left on the screen, in points: far enough in that, framed on 五's caret (learn/five_first.png), the frame stays on the screen
    BAR_IN = 78                      # frame pixels of the window's bar that show along the top

    def __init__(self, at, scale):
        self.at, self.s = at, scale
        self.w, self.h = wb.PLACE[2:]
        self.screen = (W / mac.Z, H / mac.Z)
        self.box = (wb.MAIN_X + 20 + 1, wb.CARD_Y + wb.FOOT_Y + 56 + 1)          # the inside of her box, on the page
        caret_page_y = self.box[1] + (36 - CARET_PT[1]) / 2 + CARET_PT[1] / 2
        self.scroll = caret_page_y - (at[1] - self.BAR_IN) / scale              # how far down the page is scrolled
        cx, cy = self.caret_pt()
        top, bottom = cy - at[1] / scale, cy + (H - at[1]) / scale
        left, right = cx - at[0] / scale, cx + (W - at[0]) / scale
        assert top >= 22 and bottom <= self.screen[1] and left >= 0 and right <= self.screen[0], (top, bottom, left, right)
        assert self.WIN_Y + self.h >= bottom, "the window ends inside the frame"

    def caret_pt(self):
        """The caret's middle on the screen, in points."""
        return (self.WIN_X + self.box[0] + 11.4 + CARET_PT[0] / 2, self.WIN_Y + wb.BAR - self.scroll + self.box[1] + (36 - CARET_PT[1]) / 2 + CARET_PT[1] / 2)

    def html(self, caret):
        body = wb._body(0, 3)
        mark = f'<span style="position:absolute;left:11.4px;top:{(36 - CARET_PT[1]) / 2:.2f}px;width:{CARET_PT[0]}px;height:{CARET_PT[1]}px;background:#333"></span>' if caret else ""
        at = 'padding:9px 12px">发布你的评论'
        assert body.count(at) == 1
        body = body.replace(at, 'padding:9px 12px">' + mark + "发布你的评论")
        # this close, the hot list's card would come in at the frame's right edge cut through its numbers: in this shot only it is not there, and the page's grey shows
        side = f'<div style="position:absolute;left:{wb.SIDE_X}px;top:{wb.CARD_Y}px;width:{wb.SIDE_W}px;'
        assert body.count(side) == 1
        body = body.replace(side, side + "visibility:hidden;")
        inner = self.h - wb.BAR
        scrolled = f'<div style="position:absolute;left:0;top:{-self.scroll:.3f}px;width:{self.w}px;height:{inner + self.scroll:.3f}px">{body}</div>'
        win = f'<div style="position:absolute;left:{self.WIN_X - mac.PAD:.3f}px;top:{self.WIN_Y - mac.PAD:.3f}px">{mac.browser_html("weibo.com", self.w, self.h, wb.BAR)}<div style="position:absolute;left:{mac.PAD}px;top:{mac.PAD + wb.BAR}px;width:{self.w}px;height:{inner}px;border-radius:0 0 14px 14px;overflow:hidden">{scrolled}</div></div>'
        sw, sh = self.screen
        screen = f'<div style="position:absolute;left:0;top:0;width:{sw:.3f}px;height:{sh:.3f}px;overflow:hidden;background:url(file://{mac.wallpaper()}) center/cover">{win}</div>'
        cx, cy = self.caret_pt()
        ox, oy = self.at[0] / self.s - cx, self.at[1] / self.s - cy
        return f'<div style="position:absolute;left:{ox:.3f}px;top:{oy:.3f}px">{screen}</div>'

    def frame(self, caret=True):
        a = mac.snap(self.html(caret), W / self.s, H / self.s, 2 * self.s, "#000")[..., :3]
        return np.asarray(Image.fromarray(a).resize((W, H), Image.LANCZOS))


def matched():
    """Her page, framed on 五's caret."""
    first = five_first()
    c = caret_box(first, None if os.path.exists(LEARN_FIRST) else (250, 200, 500, 420))
    at = (c["x"], (c["top"] + c["bottom"]) / 2)
    scale = (c["bottom"] - c["top"]) / CARET_PT[1]
    post = Post(at, scale)
    # the page's caret lands on whole pixels its own way: measure where it came out and move the camera by what is left over, once
    mine = caret_box(post.frame(), (int(c["x"]) - 120, int(c["top"]) - 60, int(c["x"]) + 120, int(c["bottom"]) + 60))
    at = (at[0] + c["x"] - mine["x"], at[1] + (c["top"] + c["bottom"] - mine["top"] - mine["bottom"]) / 2)
    return Post(at, scale), c


def pair():
    os.makedirs(OUT, exist_ok=True)
    post, c = matched()
    a, b = post.frame(), five_first()
    mine = caret_box(a, (int(c["x"]) - 120, int(c["top"]) - 60, int(c["x"]) + 120, int(c["bottom"]) + 60))
    print("五's caret: x %.1f, y %d-%d, %.1f wide" % (c["x"], c["top"], c["bottom"], c["width"]), flush=True)
    print("四's caret: x %.1f, y %d-%d, %.1f wide (scale %.3f px a point)" % (mine["x"], mine["top"], mine["bottom"], mine["width"], post.s), flush=True)
    Image.fromarray(a).save(os.path.join(OUT, "four_last.png"))
    Image.fromarray(b).save(os.path.join(OUT, "five_first.png"))
    Image.fromarray(np.asarray(Image.fromarray(ink_subs.apply(a.copy(), ink_subs.at("深夜里", 2.0))))).save(os.path.join(OUT, "four_last_sub.png"))
    half = lambda x: np.asarray(Image.fromarray(x).resize((960, 540), Image.LANCZOS))
    Image.fromarray(np.concatenate([half(a), half(b)], 1)).save(os.path.join(OUT, "pair.png"))
    print(OUT, flush=True)


def clip():
    """The five seconds, the caret blinking as it does in a box the page has given focus to, timed so that a blink ends at the cut: 五's first frame has its caret coming on, and across the cut the caret keeps one rhythm."""
    os.makedirs(OUT, exist_ok=True)
    post, _ = matched()
    on, off = post.frame(True), post.frame(False)
    a = ink_subs.at("深夜里")
    n = int(round((next(c[3] for c in ink_subs.chapters() if c[0] == "四") - a) * FPS))
    path = os.path.join(OUT, "four.mp4")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for i in range(n):
        t = (i - n) / FPS                   # counted back from the cut, where 五's caret comes on
        enc.stdin.write((on if (t / BLINK) % 1 < ON else off).tobytes())
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("encoding failed")
    print(path, n, "frames", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "pair":
        pair()
    elif what == "clip":
        clip()
    else:
        raise SystemExit(__doc__)
