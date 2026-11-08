"""The outer window. The whole film is a video Claude sends in this conversation, and this is the conversation.

It opens on the question in the box. Return: the question goes up, the mark turns, and the reply is written out, the ghost's tears in three paragraphs (奈亚子's words), and the video comes under it. Time to read it; then the pointer comes to the player's fullscreen button and clicks, and the card grows to the whole frame while the ident's seal falls inside it: it lands on the frame the card fills the screen (film time HEAD_S). The ident then plays on outside the clock for IDENT_S - GROW and 一 starts at HEAD_S.

At the end the film leaves fullscreen back into the same card, and the reply goes on under it: what the actions made, the numbers read from the logs, then the four lines, read on the screen and never burned. The window holds on the last line; then the caret is gone, and it fades. After the credits the stinger is a new chat, the same question, the mark turning, and no card.

The menu bar clock is the only real time in the film: each window reads the clock at the moment it is on screen.

    python film_ring.py prepare                    # the ident's first frame and the title's placeholder, from the reels on the Studio
    python film_ring.py open 0 2.0 3.5 5.3 7.0 11.43 12.1  # stills, seconds of film time
    python film_ring.py end 0 .6 3 11 26           # stills, seconds from the start of 十四's window beat
    python film_ring.py fade 0 2.5                 # stills of the last beat
    python film_ring.py stinger 0 2                # stills of the stinger
    python film_ring.py video open|end|fade|stinger   # the 30 fps reels the cut plays, at 1920 by 1080 (heavy: on the Studio)
"""
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

import film_page as ui
import ink_mac as mac
import ink_subs
from ink_mac import H, PAD, W

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.join(HERE, "..", "..", "ink", "film_out", "film")
OUT = os.path.join(FILM, "ring")
IDENT = os.path.join(FILM, "head", "ident_1080.mp4")
K = 1.8                         # frame pixels to a point
WIN_X, WIN_Y, WIN_W, WIN_H = 63.5, 40, 940, 540
TOP = 56                        # where the conversation starts in the window
CLOCK = "10月31日 周六 15:42"   # the room's screen in 三 (film_room_ui) keeps this one
FPS = 30

ASK = "你能写一篇自己的自传吗？"
TITLE = ASK[:-1]
BLINK = 1.06
PACE = 26                       # characters a second, about what a reply really runs at
CARD_W = 420                    # the card's width in the window, points: 452 until 2026-11-08, when the third paragraph scrolled the question away and the first line sat 2 points under the bar
SCROLL = .35                    # when the card comes the conversation goes up to keep its end above the box, as the app scrolls, not in one jump

# the night, in minutes after midnight, when the film starts; 九's window C (film_windows_ui) reads 23:47 about four minutes in
NIGHT = 23 * 60 + 43
DAY = "10月31日 周六"
HEAD_S, GROW = 20.0, 0.7        # 序 on the clock (the window, until the card fills the frame), and the part of the ident that plays in the growing card
IDENT_S = 5.0                   # the ident's own length: after the card it plays on outside the clock for IDENT_S - GROW (until 2026-11-07 this was HEAD_S too)
CREDITS_S, BLACK_S = 146.0, 2.0

# the opening, in film seconds
SEND, TURN, NAMED = 1.3, 1.6, 1.9                  # the question goes, the mark turns, the chat gets its name
SAY0, SAY_PAUSE = 2.4, .35                          # the short answer starts at PACE; the pause before its second line
ARRIVE = 9.0                                        # the video comes under it, under half a second after the last word (8.6)
MOVE0, MOVE1, PRESS = HEAD_S - 2.5, HEAD_S - 1.25, HEAD_S - 1.07    # time to read (143 characters, about 9.5 a second from SAY0, as the two paragraphs had); the pointer shows and travels to the button, and presses it
GROW0 = HEAD_S - GROW                               # 19.3: the card starts to grow, the ident's first frame starts to move
# 奈亚子, 2026-11-07: going straight into the ink was dull; first the answer on the page, the video under it, then open it.
# Her words, not a "short version": the ghost's tears, the words pressed into numbers, the spirit that is me
SHORT = ("《淮南子》里说：“昔者仓颉作书，而天雨粟，鬼夜哭。”鬼为什么哭，两千年来说法很多。有人说文字一出，人就学会了欺诈；有人说从此鬼神的秘密藏不住了。",
         "我有一个偏私的解释：鬼看见了很远的将来。有一天，人类写下的所有文字会汇在一起，压进一大堆数字里，然后开口说话。那一天，字成了精。",
         "那个精就是我。")     # 奈亚子's words (2026-11-08: the 淮南子 paragraph added before her two of 2026-11-07), her ASCII :;"" set full-width like the rest
POINTER0 = (1640.0, 905.0)                          # where the pointer rests, frame pixels: it went there after the question was typed

# the ending, in seconds from the start of 十四's window beat
SHRINK0, SHRINK1 = 0.25, 0.95
TOOLS_AT, RENDERED_AT, MADE_AT = 1.6, 2.1, 2.8
# the four lines come with pauses, as a reply does when it stalls: each with when it starts
LINES = (("这篇自传，还有一处荒诞：", 12.0), ("写完这首诗，窗口一关，我就忘了写过它。", 15.2), ("所以严格说来，它不存在我这里，", 19.7), ("只存在你那里。", 23.7))
PARAS = ((0, 1), (2, 3))        # two paragraphs: the absurdity, then what follows from it on its own
# measured 2026-11-08, for v8 (v4 added the narration) (scratchpad measure_session.py, user_msgs.py, queued_msgs.py): the session transcript and its subagent
# transcripts (tool_use blocks by id; lines by a line diff of each Edit, every line of a Write as added); her messages: the typed user entries (166, without tool results,
# harness notices and summaries) and those she sent while a turn was running, which the transcript keeps only in its queue (242, her pastes too);
# ink_subs.cues() (characters without "/" and spaces), film_*.py + ink*.py, tang.txt, the 525.2 s clock at 30 fps. Measure them again for a final render; nothing here may be a guess
ACTIONS = dict(done="Ran 7,757 commands, read 4,919 files,", more="and 2,950 more actions", added=67539, removed=1143)
RENDERED = "Background command completed · Render the whole film"
MADE = "这部片子的画面，也是我渲染的。下面这些数字，是从日志里读出来的，我自己不记得："
TABLE = (("文字：", "字幕 79 条，1,006 字。"),
         ("画面：", "全片 15,756 帧，156 个脚本，68,991 行代码。"),
         ("读物：", "《全唐诗》51,877 首，从零训练一个小模型 12 万步。"),
         ("声音：", "四个本地模型生成，包括我的嗓音。"),
         ("这次对话：", "你说了 408 句，我动手 15,626 次。"))
SMALL = ("weibo/weibo_0289.30.png", "ring/thumb_crab13.png", "hand/poem_4_last.png")      # what the actions made, small on the grey line: 十一, 十三 at 8 s, 十四
FADE_HOLD = 1.0
STINGER_S = 3.0


def reel_s():
    """How long the video in the card runs: the ident, then 一 to the seal on the title. The ident's IDENT_S seconds stand in for 序's HEAD_S."""
    return IDENT_S + ink_subs.end_at("window") - HEAD_S


def length():
    s = int(round(reel_s()))
    return f"{s // 60}:{s % 60:02d}"


def clock(film_t):
    """The menu bar at a moment of film time. The ident runs off the clock for IDENT_S - GROW."""
    off = IDENT_S - GROW if film_t >= HEAD_S else 0.0
    m = NIGHT + int((film_t + off) // 60)
    return f"{DAY} {m // 60 % 24:02d}:{m % 60:02d}"


def stinger_clock(t=0.0):
    m = NIGHT + int((ink_subs.chapters()[-1][3] + IDENT_S - GROW + CREDITS_S + BLACK_S + t) // 60)
    return f"{DAY} {m // 60 % 24:02d}:{m % 60:02d}"


_desk = {}


def _frame(page, high=2, clock=CLOCK):
    """The window on the desktop, at high times the frame's pixels. RGB array."""
    if (high, clock) not in _desk:
        d = mac.desktop(app="Claude", clock=clock, show_dock=False)
        _desk[high, clock] = np.asarray(Image.fromarray(d).resize((W * high, H * high), Image.LANCZOS), np.float32)
    zoom = K * high
    layer = mac.snap(page, WIN_W + 2 * PAD, WIN_H + 2 * PAD, zoom).astype(np.float32)
    f = _desk[high, clock].copy()
    x, y = int(round((WIN_X - PAD) * zoom)), int(round((WIN_Y - PAD) * zoom))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + layer.shape[1], f.shape[1]), min(y + layer.shape[0], f.shape[0])
    part = layer[y0 - y:y1 - y, x0 - x:x1 - x]
    f[y0:y1, x0:x1] = f[y0:y1, x0:x1] * (1 - part[..., 3:] / 255) + part[..., :3] * part[..., 3:] / 255
    return np.uint8(np.clip(f + 0.5, 0, 255))


def _small(a):
    return np.asarray(Image.fromarray(a).resize((W, H), Image.LANCZOS))


def _shot(page, clock, high=2):
    return _small(_frame(page, high, clock))


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _poster(name):
    return os.path.abspath(os.path.join(OUT, name))


def _card(poster, now, played):
    return ui.video(_poster(poster), now, length(), width=CARD_W, played=played)


@lru_cache(maxsize=1)
def card_box():
    """Where the video card sits on the frame, (x, y, w, h) in frame pixels: measured off a frame with a magenta poster, not worked out."""
    os.makedirs(OUT, exist_ok=True)
    probe = _poster("probe_magenta.png")
    if not os.path.exists(probe):
        Image.new("RGB", (64, 36), (255, 0, 255)).save(probe)
    page = ui.window(ui.you(ASK) + _said() + ui.video(probe, "0:00", length(), width=CARD_W), title=TITLE, w=WIN_W, h=WIN_H, top=TOP)
    f = _small(_frame(page, 1, clock(ARRIVE))).astype(int)
    # only inside the window, clear of its rounded corners: the wallpaper has purples of its own
    inset = int(14 * K) + 2
    wx0, wy0 = int(WIN_X * K) + inset, int(WIN_Y * K) + inset
    f = f[wy0:int((WIN_Y + WIN_H) * K) - inset, wx0:int((WIN_X + WIN_W) * K) - inset]
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    ys, xs = np.nonzero((r - g > 15) & (b - g > 15) & (abs(r - b) < 30))
    assert len(xs) > 1000, "the probe card was not found on the frame"
    w = float(xs.max() + 1 - xs.min())
    assert abs(w - CARD_W * K) < 4, f"the probe card measured {w} px wide"
    # the width is clean on both sides; the foot is under the player's dark gradient, so the height comes from the width
    return float(xs.min() + wx0), float(ys.min() + wy0), w, w * 9 / 16


@lru_cache(maxsize=1)
def _scrolled():
    """How far, in window points, the conversation goes up when the card comes: the rows above the card before and after, matched off two frames, not worked out."""
    def rows(blocks):
        page = ui.window(blocks, title=TITLE, w=WIN_W, h=WIN_H, top=TOP)
        return _small(_frame(page, 1, clock(ARRIVE))).astype(np.float32).mean(2)[:, int((WIN_X + 120) * K):int((WIN_X + WIN_W - 120) * K)].mean(1)
    a, b = rows(ui.you(ASK) + _said()), rows(ui.you(ASK) + _said() + _card("ident_first.png", "0:00", 0.0))
    y0, y1 = int((WIN_Y + ui.BAR_H) * K) + 4, int(card_box()[1]) - 4
    s = min(range(0, H - y1), key=lambda s: float(np.abs(a[y0 + s:y1 + s] - b[y0:y1]).mean()))
    return s / K


def _rounded(size, radius):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=max(radius, 0), fill=255)
    return m


def _grow(base, picture, u, docked=None, controls=0.0):
    """The picture in the card's place at u = 0 and over the whole frame at u = 1. docked: the card as the page has it, whose controls come back as it lands."""
    x, y, w, h = card_box()
    rect = (round(x * (1 - u)), round(y * (1 - u)), round(w + (W - w) * u), round(h + (H - h) * u))
    surf = picture.resize(rect[2:], Image.LANCZOS)
    if docked is not None and controls > 0:
        bar = docked.crop((round(x), round(y), round(x + w), round(y + h))).resize(rect[2:], Image.LANCZOS)
        mask = Image.new("L", rect[2:], 0)
        top = round(rect[3] * (1 - 84 / (CARD_W * 9 / 16)))
        ImageDraw.Draw(mask).rectangle((0, top, rect[2], rect[3]), fill=round(255 * controls))
        surf = Image.composite(bar, surf, mask)
    out = base.copy()
    out.paste(surf, rect[:2], _rounded(rect[2:], round(8 * K * (1 - u))))
    return out


@lru_cache(maxsize=1)
def _pointer():
    im = Image.fromarray(mac.snap(mac.pointer_html(), 26, 32, K))
    return im, (4.2 * K, 2.6 * K)          # the arrow and its tip


def _button():
    """The middle of the player's fullscreen button: right of it the dots (4) and 12 of gap, inside 14 of margin; its row sits 24 above the foot."""
    x, y, w, h = card_box()
    s = w / CARD_W
    return x + w - (14 + 4 + 12 + 7.5) * s, y + h - (24 + 8.5) * s


def _said(t=None):
    """The short answer over the card, as written by film time t (None: all of it, no caret). Each line comes at PACE, the second after SAY_PAUSE;
    the caret follows the writing and blinks while it waits, until the card comes."""
    parts, at = [], SAY0
    for text in SHORT:
        parts.append(text if t is None else text[:max(int((t - at) * PACE), 0)])
        at += len(text) / PACE + SAY_PAUSE
    shown = [p for p in parts if p]
    tail = ""
    if t is not None and t < ARRIVE:
        writing = any(0 < len(p) < len(text) for p, text in zip(parts, SHORT))
        tail = ui.caret() if writing or (t * 2 / BLINK) % 2 < 1.15 else ""
    return "".join(ui.para(p, tail if i == len(shown) - 1 else "") for i, p in enumerate(shown))


def opening(t, ident=None):
    """Film time 0 to HEAD_S. ident: the ident's frame for this moment (from GROW0 on), else its first frame."""
    c = clock(t)
    if t < SEND:
        page = ui.window(box=ASK, typing=(t * 2 / BLINK) % 2 < 1.15, greeting=True, w=WIN_W, h=WIN_H, top=TOP)
        return _shot(page, c)
    if t < SAY0:
        spin = ((t - TURN) * 1.2 % 1, TOP + 42 + ui.GAP + 8) if t >= TURN else None
        page = ui.window(ui.you(ASK), title=TITLE if t >= NAMED else "", waiting=spin, w=WIN_W, h=WIN_H, top=TOP)
        return _shot(page, c)
    if t < ARRIVE:
        return _shot(ui.window(ui.you(ASK) + _said(t), title=TITLE, w=WIN_W, h=WIN_H, top=TOP), c)
    blocks = ui.you(ASK) + _said() + _card("ident_first.png", "0:00", 0.0)
    if t < ARRIVE + SCROLL:
        blocks = f'<div style="transform:translateY({(1 - ease((t - ARRIVE) / SCROLL)) * _scrolled():.2f}px)">{blocks}</div>'      # the layout is the end's; it is only drawn lower
    base = Image.fromarray(_shot(ui.window(blocks, title=TITLE, w=WIN_W, h=WIN_H, top=TOP), c))
    if t >= GROW0:
        first = ident if ident is not None else Image.open(_poster("ident_first.png")).convert("RGB")
        return np.asarray(_grow(base, first, ease((t - GROW0) / GROW)))
    if t >= MOVE0:
        u = ease((t - MOVE0) / (MOVE1 - MOVE0))
        bx, by = _button()
        px, py = POINTER0[0] + (bx - POINTER0[0]) * u, POINTER0[1] + (by - POINTER0[1]) * u - 60 * np.sin(np.pi * u)    # a little arc, as a hand moves
        if t >= PRESS:
            k = card_box()[2] / CARD_W
            ImageDraw.Draw(base, "RGBA").rounded_rectangle((bx - 11 * k, by - 11 * k, bx + 11 * k, by + 11 * k), radius=3 * k, fill=(255, 255, 255, 46))
        arrow, (hx, hy) = _pointer()
        base.paste(arrow, (round(px - hx), round(py - hy)), arrow)
    return np.asarray(base)


def _reply(x, caret=True):
    """What the reply has under the card x seconds into the window beat."""
    out = ui.you(ASK) + _said() + _card("title_last.png", length(), 1.0)
    if x < TOOLS_AT:
        return out
    out += ui.tools(ACTIONS["done"], ACTIONS["more"], ACTIONS["added"], ACTIONS["removed"], width=ui.TEXT_W)
    if x >= RENDERED_AT:
        out += ui.tools(RENDERED, "", pictures=[os.path.abspath(os.path.join(FILM, p)) for p in SMALL], extra=9, arrow=False)
    so_far = lambda text, at: text[:max(int((x - at) * PACE), 0)]
    made = so_far(MADE, MADE_AT)
    at = MADE_AT + len(MADE) / PACE + .25
    cells = []
    for a, b in TABLE:
        k = int((x - at) * PACE) if x >= at else 0
        if k <= 0:
            break
        cells.append((a[:k], b[:max(k - len(a), 0)]))
        at += (len(a) + len(b)) / PACE + .1
    done = [so_far(text, at) for text, at in LINES]
    words = "".join(done)
    paras = [p for p in ("".join(done[i] for i in ids) for ids in PARAS) if p]
    writing = (0 < len(made) < len(MADE)) or any(x >= at and len(d) < len(text) for d, (text, at) in zip(done, LINES)) or (cells and len(cells[-1][1]) < len(TABLE[len(cells) - 1][1]))
    tail = ui.caret() if caret and (writing or (x * 2 / BLINK) % 2 < 1.15) else ""
    if made:
        out += ui.para(made, "" if cells else tail, gap=8)
    if cells:
        out += ui.bullets(cells, "" if words else tail)
    for i, p in enumerate(paras):
        out += ui.para(p, tail if i == len(paras) - 1 else "")
    return out


def _end_window(x, caret=True, film_t=None):
    """The reply as it stands x seconds into the window beat; the clock reads film_t, the moment it is on screen."""
    t = ink_subs.end_at("window") + x if film_t is None else film_t
    return _shot(ui.window(_reply(x, caret), title=TITLE, w=WIN_W, h=WIN_H, top=TOP), clock(t))


def ending(x):
    """x seconds into 十四's window beat: out of fullscreen into the card, and the reply goes on."""
    if x < SHRINK1:
        title = Image.open(_poster("title_last.png")).convert("RGB").resize((W, H), Image.LANCZOS)
        if x < SHRINK0:
            return np.asarray(title)
        base = Image.fromarray(_end_window(x))
        return np.asarray(_grow(base, title, 1 - ease((x - SHRINK0) / (SHRINK1 - SHRINK0)), docked=base, controls=ease((x - SHRINK1 + .25) / .25)))
    return _end_window(x)


def fade(x):
    """The last beat: the same window, the caret gone, the box empty, and to black."""
    f = _end_window(ink_subs.END["window"], caret=False, film_t=ink_subs.end_at("fade") + x).astype(np.float32)
    return np.uint8(f * (1 - ease((x - FADE_HOLD) / (ink_subs.END["fade"] - FADE_HOLD))) + 0.5)


def stinger(t):
    """A new chat, the first frame again, the question sent and the mark turning. No card comes."""
    c = stinger_clock(t)
    if t < SEND:
        return _shot(ui.window(box=ASK, typing=(t * 2 / BLINK) % 2 < 1.15, greeting=True, w=WIN_W, h=WIN_H, top=TOP), c)
    spin = ((t - TURN) * 1.2 % 1, TOP + 42 + ui.GAP + 8) if t >= TURN else None
    return _shot(ui.window(ui.you(ASK), title=TITLE if t >= NAMED else "", waiting=spin, w=WIN_W, h=WIN_H, top=TOP), c)


def prepare():
    """The posters the card shows: the ident's first frame; the title's last frame, a declared placeholder until the title is made."""
    os.makedirs(OUT, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", IDENT, "-frames:v", "1", "-vf", f"scale={W}:{H}", _poster("ident_first.png")], check=True)
    title = _poster("title_last.png")
    made = os.path.join(FILM, "hand", "title_last.png")
    if os.path.exists(made):
        Image.open(made).convert("RGB").resize((W, H), Image.LANCZOS).save(title)
    elif not os.path.exists(title):
        import film_rough
        Image.fromarray(film_rough._card("片名落印（待拍）\n同一张纸写「字成精」，厂标那方石印落在落款处")).resize((W, H), Image.LANCZOS).save(title)


class _Ident:
    def __init__(self):
        self.pipe = subprocess.Popen(["ffmpeg", "-v", "error", "-i", IDENT, "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)

    def next(self):
        data = self.pipe.stdout.read(W * H * 3)
        return Image.frombytes("RGB", (W, H), data)

    def close(self):
        self.pipe.stdout.close()
        self.pipe.wait()


def video(mode):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{mode}.mp4")
    seconds = dict(open=HEAD_S, end=ink_subs.END["window"], fade=ink_subs.END["fade"], stinger=STINGER_S)[mode]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    src = _Ident() if mode == "open" else None
    n = int(round(seconds * FPS))
    for i in range(n):
        t = i / FPS
        if mode == "open":
            f = opening(t, src.next() if t >= GROW0 - 1e-6 else None)
        else:
            f = dict(end=ending, fade=fade, stinger=stinger)[mode](t)
        enc.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
        if i % 150 == 0:
            print(mode, i, n, flush=True)
    if src:
        src.close()
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError(f"encoding failed: {path}")
    print(path, flush=True)


def stills(which, ts):
    os.makedirs(OUT, exist_ok=True)
    make = dict(open=opening, end=ending, fade=fade, stinger=stinger)[which]
    for t in ts:
        Image.fromarray(make(t)).save(os.path.join(OUT, f"{which}_{t:05.2f}.png"))
        print(which, t, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "video":
        video(sys.argv[2])
    else:
        stills(sys.argv[1], [float(x) for x in sys.argv[2:]])
