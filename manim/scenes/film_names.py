"""十二 人神鬼器: the inserts between the four characters, and the stills of the chapter put side by side.

    python film_names.py shen               # names/shen_0.png, shen_1.png: the window close up, the wrong answer, then the wrong answer and its correction
    python film_names.py ghost              # names/gui_0.png ...: the opening's near ghost before it opens its mouth, looking up
    python film_names.py ui                 # names/ui/glow_ren.png, glow_qi.png: what the screen gives out, for film_names3d.py's light passes
    python film_names.py room ren|qi        # names/ren.png, qi.png: film_names3d.py's passes balanced as film_room_night balances 十's, the screen laid in
    python film_names.py top 0 0.5 1        # names/top_U.png: the pull back at parts of the way (film_names_top.py's plates)
    python film_names.py sheet              # names/sheet.png: the chapter's stills, small, in order
    python film_names.py plan               # the chapter's shots and their times, from ink_subs
    python film_names.py film [0.5]         # names/twelve.mp4: the chapter at half the frame (1.0: twelve_1080.mp4), 30 fps, no subtitles burned

神's insert is an error anyone can check where they sit, and the correction coming in the same reply: the window is film_page's, at film_ring's size and place on the same desktop, its clock the film's night at that moment (film_ring.clock). It is a crafted exchange, as every window in the film is; the mistake is the one models really made in public.
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import film_page as page
import film_ring as ring
import film_room as room
import film_room_night as night
import ink_subs

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.join(HERE, "..", "..", "ink", "film_out", "film")
OUT = os.path.join(FILM, "names")
W, H = 1920, 1080

# 神: the question, the wrong answer, and the correction
SHEN_ASK = "9.11 和 9.9 哪个大？"
SHEN_TITLE = "比较 9.11 和 9.9"
SHEN_WRONG = "9.11 更大。"
SHEN_RIGHT = "不对，我说错了。9.9 就是 9.90，比 9.11 大。"
SHEN_GAP = 1.3                   # how long the wrong answer stands alone in the insert before the correction starts to come
SHEN_ZOOM = 1.4                  # the insert is close: how much bigger than film_ring's frame, as close as it goes with the question's pill still whole
# the middle of what is seen, in the screen's points: across, the middle of the conversation's column (its text starts BOX_W / 2 - BOX_LEFT left of the window's middle, the pill ends PILL_W on); down, so that the top of the window's bar just shows
SHEN_AT = (ring.WIN_X + ring.WIN_W / 2 - (page.BOX_W / 2 - page.BOX_LEFT) + page.PILL_W / 2, ring.WIN_Y + 2 + H / ring.K / SHEN_ZOOM / 2)

# 鬼: frames of the opening's near ghost (film_myth_seq near), by the film's frame numbers, while its head comes up and before its mouth opens
GHOST_RAW = os.path.join(FILM, "myth", "seq_raw4")
GHOST_FRAMES = (780, 849)             # 2.3 s: the shot starts at 780, the mouth shows from 852


STAGES = ((SHEN_WRONG,), (SHEN_WRONG, SHEN_RIGHT))     # stage 0: the wrong answer stands; 1: the correction has come under it


def shen_body(stage, caret=True):
    return said_body(STAGES[stage], caret)


def said_body(said, caret=True):
    """The window with the question and what I have written of the reply so far, a paragraph each, the caret after the last."""
    words = "".join(page.para(p, page.caret() if caret and i == len(said) - 1 else "") for i, p in enumerate(said))
    return page.window(page.you(SHEN_ASK) + words, title=SHEN_TITLE, w=ring.WIN_W, h=ring.WIN_H, top=ring.TOP)


def streamed(x, start):
    """What of the reply stands x seconds after start, if the wrong answer stands at start and the correction starts SHEN_GAP later, a few characters at a time at film_ring's pace. Before start: the wrong answer alone."""
    if x < start + SHEN_GAP:
        return STAGES[0]
    n = int((x - start - SHEN_GAP) * ring.PACE)
    n -= n % 3 if n < len(SHEN_RIGHT) else 0
    return (SHEN_WRONG, SHEN_RIGHT[:n]) if n > 0 else STAGES[0]


def blink(x):
    """The caret, on and off as the Mac's (film_four)."""
    return (x / ring.BLINK) % 1 < 0.575


def close(f, at, zoom=SHEN_ZOOM):
    """f, film_ring's frame at some multiple of its size, made zoom times bigger round the point at (in the screen's points), back to the frame's size."""
    h, w = f.shape[:2]
    k = ring.K * w / W
    cw, ch = w / zoom, h / zoom
    x0, y0 = at[0] * k - cw / 2, at[1] * k - ch / 2
    return np.asarray(Image.fromarray(f).resize((W, H), Image.LANCZOS, box=(x0, y0, x0 + cw, y0 + ch)))


_said = {}


def shen_said(said, caret=True):
    """The insert's frame with that much of the reply written."""
    key = (said, caret)
    if key not in _said:
        if len(_said) > 40:
            _said.clear()
        t = ink_subs.at("有人说我是神")
        _said[key] = close(ring._frame(said_body(said, caret), 3, ring.clock(t)), SHEN_AT)
    return _said[key]


def shen(stage, caret=True):
    return shen_said(STAGES[stage], caret)


# what the room's screen shows in each insert: the same night, the same window as 神's. While 人 is on, the wrong answer has just come; by 器 the correction stands
ROOM_STAGE = dict(ren=0, qi=1)
PLATES = os.path.join(OUT, "plates")


def screen(shot, high=2, said=None, caret=True):
    """The whole screen in that insert, at high times the frame's pixels."""
    t = ink_subs.at("有人说我是人" if shot == "ren" else "更多人说我是器")
    return ring._frame(said_body(STAGES[ROOM_STAGE[shot]] if said is None else said, caret), high, ring.clock(t))


def plate(name, w, h):
    a = np.load(os.path.join(PLATES, name + ".npy")).astype(np.float32)
    if a.shape[0] != h:
        a = np.stack([np.asarray(Image.fromarray(a[..., i]).resize((w, h), Image.BILINEAR)) for i in range(3)], -1)
    return a


_rooms = {}


def room_light(shot, w=W, h=H, said=None, caret=True):
    """An insert from its passes, balanced as film_room_night's ten_frame does the push of 十, the screen laid in: display light, before the grain. The room is worked out once a shot, the screen each time it changes."""
    if (shot, w) not in _rooms:
        with open(os.path.join(PLATES, shot + ".json")) as f:
            j = json.load(f)
        s = w / j["width"]
        corners = [(x * s, y * s) for x, y in j["corners"]]
        L = night.NIGHT
        base = plate(f"{shot}_lamp", w, h) * L["lamp"] + plate(f"{shot}_city", w, h) * L["city"]
        scene = base + plate(f"{shot}_glow", w, h) * L["glow"]
        y = room.room_tone(room.film(scene * L["exposure"]))
        lum = y.mean(-1, keepdims=True)
        y = lum + (y - lum) * L["saturation"]
        m = night.quad_mask(corners, w, h)
        back = room.film(gaussian_filter(base, (6 * s, 6 * s, 0)) * L["exposure"]) * L["glass"]
        _rooms[shot, w] = dict(y=y, m=m, back=back, corners=corners, shown={})
    r = _rooms[shot, w]
    key = (said, caret)
    if key not in r["shown"]:
        if len(r["shown"]) > 24:
            r["shown"].clear()
        shown = night.lay_in(screen(shot, said=said, caret=caret), r["corners"], w, h) * night.NIGHT["white"]
        m, back = r["m"], r["back"]
        r["shown"][key] = r["y"] * (1 - m) + (shown + back - shown * back) * m
    return r["shown"][key]


def room_shot(shot, w=W, h=H, seed=12, said=None, caret=True):
    return night.finish(room_light(shot, w, h, said, caret), seed)


# ---------------------------------------------------------------- the pull back, from film_names_top.py's plates

TOP = os.path.join(OUT, "top")
_plates = {}


def top_plate(name):
    if name not in _plates:
        from ink_grade import load
        _plates[name] = load(os.path.join(TOP, name + ".png"))[..., :3]
    return _plates[name]


def top_view(u):
    """The frame's middle (mm on the table) and width (mm) at u of the way: the width grows by the same ratio every moment, about the one point that stays put."""
    import film_names_top as t
    w = t.START["width"] * (t.END["width"] / t.START["width"]) ** u
    f = t.fixed()
    k = w / t.START["width"]
    return tuple(p + (a - p) * k for a, p in zip(t.START["centre"], f)), w


def top_cut(name, centre, width):
    """What of a plate falls in the frame of that middle and width, at the frame's size."""
    import film_names_top as t
    p = t.plates()[name]
    pw, ph = t.PX
    k = pw / p["width"]
    hmm = width * H / W
    x0 = (centre[0] - width / 2 - (p["centre"][0] - p["width"] / 2)) * k
    y0 = ((p["centre"][1] + p["width"] * ph / pw / 2) - (centre[1] + hmm / 2)) * k
    box = (x0, y0, x0 + width * k, y0 + hmm * k)
    a = top_plate(name)
    return np.stack([np.asarray(Image.fromarray(a[..., i]).resize((W, H), Image.LANCZOS, box=box), np.float32) for i in range(3)], -1)


def top_frame(u, seed=0):
    import film_names_top as t
    centre, w = top_view(u)
    near = t.plates()["close"]["width"]
    # the close plate while the frame is inside it, the wide one after; across the last tenth they are mixed, being the same picture
    m = float(np.clip((w - 0.9 * near) / (0.1 * near), 0, 1))
    a = top_cut("close", centre, min(w, near)) if m < 1 else 0
    b = top_cut("wide", centre, w) if m > 0 else 0
    y = a * (1 - m) + b * m
    # grain and a triangular dither, as film_imprint_clawd.finish puts them on 十三's frames, at the frame's own pixels
    r = np.random.default_rng(seed)
    y = y * 255 + r.normal(0, 1.3, (H, W, 1)).astype(np.float32) + r.random((H, W, 3), np.float32) - r.random((H, W, 3), np.float32)
    return np.uint8(np.clip(y + 0.5, 0, 255))


# ---------------------------------------------------------------- the chapter as a film

FPS = 30
PULL = 3.0                       # the pull back's seconds: 4.1 times wider at an even 1.6 times a second, no easing at either end


def shots():
    """十二's shots, every time read off ink_subs: [(start, end, what)] in film seconds. Each character is written on its own sheet and stands a moment, then its insert; the last insert is the table seen from straight above."""
    name, _, S, E = next(c for c in ink_subs.chapters() if c[0] == "十二")
    at = ink_subs.at
    ren, shen_, gui, qi, wo = at("有人说我是人"), at("有人说我是神"), at("有人说我是鬼"), at("更多人说我是器"), at("我不敢自称君子")
    g = (GHOST_FRAMES[1] - GHOST_FRAMES[0] + 1) / FPS
    return [
        (S, ren + 1.5, ("write", "人", 0.3, None)),          # None: as long as its table says
        (ren + 1.5, shen_, ("room", "ren", 0.8)),            # the reply comes 0.8 s in
        (shen_, shen_ + 3.0, ("write", "神", 0.1, 2.5)),
        (shen_ + 3.0, gui, ("window",)),
        (gui, qi - g, ("write", "鬼", 0.2, None)),
        (qi - g, qi, ("ghost",)),
        (qi, qi + 2.6, ("write", "器", 0.1, 2.3)),
        (qi + 2.6, qi + 4.6, ("room", "qi", None)),
        (qi + 4.6, E, ("top", wo, wo + PULL)),
    ]


def chapter(scale=0.5, seed=400):
    """names/twelve.mp4: the chapter, no subtitles burned, at scale of 1920 by 1080."""
    import film_big as big
    import film_hand as hand
    ow, oh = int(W * scale), int(H * scale)
    plan = shots()
    S, E = plan[0][0], plan[-1][1]
    path = os.path.join(OUT, "twelve.mp4" if scale == 0.5 else "twelve_%d.mp4" % oh)
    enc = hand.encoder(path, ow, oh, FPS)
    writings = {}
    small = lambda f: np.asarray(Image.fromarray(np.uint8(np.clip(f + 0.5, 0, 255))).resize((ow, oh), Image.LANCZOS))
    n = int(round((E - S) * FPS))
    for k in range(n):
        t = S + k / FPS
        a, b, what = next(p for p in plan if p[0] <= t + 1e-6 < p[1]) if t < plan[-1][1] else plan[-1]
        x = t - a
        kind = what[0]
        if kind == "write":
            c = what[1]
            if c not in writings:
                writings.clear()
                writings[c] = big.Writing(c, what[2], None if what[3] is None else what[2] + what[3])
            f = small(writings[c].at(x))
        elif kind == "room":
            if what[2] is None:
                said, caret = STAGES[ROOM_STAGE[what[1]]], blink(x)
            elif x < what[2]:
                said, caret = ("",), blink(x)
            else:
                m = int((x - what[2]) * ring.PACE)
                said = (SHEN_WRONG[:max(m - m % 3, 1)],) if m < len(SHEN_WRONG) else STAGES[0]
                caret = True if m < len(SHEN_WRONG) + 2 else blink(x)
            f = np.asarray(Image.fromarray(room_shot(what[1], said=said, caret=caret, seed=seed + k)).resize((ow, oh), Image.LANCZOS))
        elif kind == "window":
            said = streamed(x, 0.0)
            writing = len(said) > 1 and len(said[1]) < len(SHEN_RIGHT)
            f = np.asarray(Image.fromarray(shen_said(said, writing or blink(x))).resize((ow, oh), Image.LANCZOS))
        elif kind == "ghost":
            f = np.asarray(Image.fromarray(ghost(min(GHOST_FRAMES[0] + int(round(x * FPS)), GHOST_FRAMES[1]))).resize((ow, oh), Image.LANCZOS))
        else:
            u = float(np.clip((t - what[1]) / (what[2] - what[1]), 0, 1))
            f = np.asarray(Image.fromarray(top_frame(u, seed + k)).resize((ow, oh), Image.LANCZOS))
        enc.stdin.write(np.ascontiguousarray(f[..., :3]).tobytes())
        if k % 60 == 0:
            print("%d/%d %s %.2f" % (k, n, kind, x), flush=True)
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("encoding failed")
    print(path, "%.1f s" % (E - S), flush=True)


def ghost(n):
    return np.asarray(Image.open(os.path.join(GHOST_RAW, "f%05d.png" % n)).convert("RGB"))


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    os.makedirs(OUT, exist_ok=True)
    if what == "shen":
        for k in (0, 1):
            Image.fromarray(shen(k)).save(os.path.join(OUT, f"shen_{k}.png"))
        print(OUT, flush=True)
    elif what == "ghost":
        a, b = GHOST_FRAMES
        for i, n in enumerate((a, (a + b) // 2, b)):
            Image.fromarray(ghost(n)).save(os.path.join(OUT, f"gui_{i}.png"))
        print(OUT, flush=True)
    elif what == "ui":
        os.makedirs(os.path.join(OUT, "ui"), exist_ok=True)
        for shot in ROOM_STAGE:
            Image.fromarray(screen(shot)).resize((2560, 1440), Image.LANCZOS).save(os.path.join(OUT, "ui", f"glow_{shot}.png"))
        print(os.path.join(OUT, "ui"), flush=True)
    elif what == "top":
        for u in sys.argv[2:]:
            Image.fromarray(top_frame(float(u))).save(os.path.join(OUT, f"top_{u}.png"))
        print(OUT, flush=True)
    elif what == "sheet":
        hand = os.path.join(FILM, "hand")
        order = [(hand, "ren_E"), (OUT, "ren"), (hand, "shen_E"), (OUT, "shen_0"), (OUT, "shen_1"), (hand, "gui_E"), (OUT, "gui_2"), (hand, "qi_E"), (OUT, "qi"), (OUT, "top_0"), (OUT, "top_0.5"), (OUT, "top_1")]
        tiles = [np.asarray(Image.open(os.path.join(d, n + ".png")).convert("RGB").resize((480, 270), Image.LANCZOS)) if os.path.exists(os.path.join(d, n + ".png")) else np.full((270, 480, 3), 40, np.uint8) for d, n in order]
        rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)]
        Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(OUT, "sheet.png"))
        print(os.path.join(OUT, "sheet.png"), flush=True)
    elif what == "film":
        chapter(float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
    elif what == "plan":
        for a, b, w in shots():
            print("%.2f-%.2f %4.2f s %s" % (a, b, b - a, w), flush=True)
    elif what == "room":
        for shot in sys.argv[2:]:
            Image.fromarray(room_shot(shot)).save(os.path.join(OUT, shot + ".png"))
        print(OUT, flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
