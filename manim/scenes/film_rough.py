"""The whole film cut together at 960 by 540, to judge its pace before anything goes to 1080p. Whatever is not built yet stands in as what exists of it, or as a card saying what will be there.

    python film_rough.py list                # the cut in film time, what is built and what is a card, and how much of each
    python film_rough.py learn               # 五六七's window, from ink_learn at its frames per second: rough/learn/
    python film_rough.py weibo               # the post, with no line on it: rough/weibo/
    python film_rough.py cut [A B]           # rough/rough.mp4, or only film seconds A to B as rough/rough_A_B.mp4, for looking at one join
    python film_rough.py sheet               # rough/rough_sheet.png, a frame of every shot
    python film_rough.py final list          # the same cut at 1920 by 1080: which source each shot takes, and which are still only low
    python film_rough.py final weibo         # final/weibo/: the post at 1920 and 30 fps
    python film_rough.py final cut [A B]     # final/film_1080.mp4 (or film_1080_A_B.mp4): every shot's 1080 version, the grain, the lines burned at full size
    python film_rough.py final mux           # final/film_1080_mix.mp4: with sound/film_mix.wav (crew F)

The final cut takes each source's 1080 version (NAME_1080 beside it, or HI), and falls back to the low one scaled up only where none exists yet, which
`final list` names. The film inside the card has one grain over it, from 一 to the title (and the ident); the outer window and the credits
are the world outside the video and stay clean.

The reels and the film's clock are kept apart as in film.director.md: the ident, then film time 0 onwards, then the title and the credits. Bridge times in the table of film.director.md are on the bridge's own clock: b() puts its 81.6 where 三's groove ends in the cut.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ink_subs

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.join(HERE, "..", "..", "ink", "film_out", "film")
OUT = os.path.join(FILM, "rough")
W, H, FPS = 960, 540, 30
BRIDGE0 = 81.6                   # the bridge table's time on the cut after 三's groove (crew A); it was film time less 12.3 until 2026-11-07
LEARN_FPS = 6
PAPER, INKY, FAINT = (246, 241, 230), (70, 62, 52), (140, 130, 116)
HEAD_S, GROW = 20.0, 0.7         # 序 on the clock (film_ring.HEAD_S, ink_subs' 序), and the ident's first part, which plays in the growing card
IDENT_S = 5.0                    # the ident's own length (film_ring.IDENT_S): the rest of it plays off the clock between 序 and 一
# 三 in frames after 「古人说」 at 30, 奈亚子 2026-11-07: the pressing gets faster and faster. Every shot is shorter than the one before until the seventh
# strike, which holds: 78 63 54 47 34 28 24 19 15 10 9 8 7, then 69. The strikes 4.6, 2.9, 2.0, 1.4, 0.9, 0.5 s apart (they were 4.6 3.4 2.6 2.0 1.5 1.1).
# film_imprint_clay.BLOWS keeps the render's own clock; here is only where each clip sits. (name, first frame, end frame)
PRESS_CUT = (("b1", 0, 78), ("d1_letter", 78, 141), ("b2", 141, 195), ("d2_poem", 195, 242), ("b3", 242, 276), ("d4_quarto", 276, 304),
             ("b4", 304, 328), ("d3_ruling", 328, 347), ("b5", 347, 362), ("s4_term", 362, 372), ("s4_keys", 372, 381), ("b6", 381, 389),
             ("s5_a", 389, 396), ("b7", 396, 465), ("dawn", 465, 549), ("groove", 549, 645))
PRESS_CONTACT = (27, 165, 252, 311, 353, 381, 396)     # the frame each die touches the clay on the cut (the mix's s1 to s7)
CONTACT_FRAME = (27, 27, 11, 9, 7, 3, 11)              # how far into b1.mp4 … b7.mp4 it touches: round(contact * 30) - round(head * 30) of BLOWS
CREDITS_S, BLACK_S = 146.0, 2.0
FINAL = False                    # set by `final`: 1920 by 1080, the 1080 sources, the grain
GRAIN = 3.0                      # the grain's standard deviation in 8-bit steps at mid grey, less towards black and white


def p(*a):
    return os.path.join(FILM, *a)


class Shot:
    def __init__(self, t0, t1, what, burn=True, built=True):
        self.t0, self.t1, self.what, self.burn, self.built = t0, t1, what, burn, built


def clip(path, at=0.0):
    return dict(kind="clip", path=path, at=at)


def frames(folder, fps=30, first=0):
    return dict(kind="frames", folder=folder, fps=fps, first=first)


STILL_PUSH = .007      # 九's held stills come in this much of the way in a second, about at (2026-11-08: held dead between the moving city and the tunnel they read as a slide show)


def still(path, push=0.0, at=(0.5, 0.45)):
    return dict(kind="still", path=path, push=push, at=at)


def card(text):
    return dict(kind="card", text=text)


def black():
    return dict(kind="black")


def b(x):
    """Film time of a time on the bridge's table: its BRIDGE0 is the end of 三's groove."""
    return ink_subs.at("古人说") + PRESS_CUT[-1][2] / 30 + x - BRIDGE0


def use_final():
    global W, H, FINAL, OUT
    W, H, FINAL = 1920, 1080, True
    OUT = os.path.join(FILM, "final")


# a low source and its 1080 version, where the version is not NAME_1080 beside it
HI = {
    ("learn", "learn.mp4"): ("learn", "learn_1920.mp4"),
    ("windows", "tube_low3", "tube_low3.mp4"): ("windows", "tube_hi", "tube_hi.mp4"),
    ("head", "credits_low3.mp4"): ("head", "credits_1080.mp4"),
    ("rough", "weibo"): ("final", "weibo"),
}
# low sources with their lines burned in whose 1080 version is clean: in the final cut the lines go on with ink_subs.apply, as they did in the low ones (crew D)
LINES_IN_LOW = {("myth", "myth_low5.mp4"), ("myth", "myth_low7.mp4")}
_wide = {}


def width(path):
    """A source's width in pixels: a clip's or still's, or a frames folder's first frame's."""
    if path not in _wide:
        q = path
        if os.path.isdir(path):
            fs = sorted(f for f in os.listdir(path) if f.endswith((".png", ".jpg")))
            q = os.path.join(path, fs[0]) if fs else None
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width", "-of", "csv=p=0", q],
                           capture_output=True, text=True) if q and os.path.exists(q) else None
        _wide[path] = int(r.stdout.strip() or 0) if r else 0
    return _wide[path]


def hi(path):
    """The 1080 version of a low source, or None while there is none."""
    rel = tuple(os.path.relpath(path, FILM).split(os.sep))
    if rel in HI:
        q = p(*HI[rel])
    elif os.path.isdir(path):
        q = path.rstrip(os.sep) + "_1080"
    else:
        d, f = os.path.split(path)
        stem, ext = os.path.splitext(f)
        q = os.path.join(d + "_1080", f) if os.path.isdir(d + "_1080") else os.path.join(d, stem + "_1080" + ext)
    return q if os.path.exists(q) else None


def full(what):
    """The shot's source at full size: (what, 'full' | 'hi' | 'LOW')."""
    path = what.get("path") or what.get("folder")
    if not path or width(path) >= 1920:
        return what, "full"
    q = hi(path)
    if not q:
        return what, "LOW"
    w = dict(what, **{"path" if "path" in what else "folder": q})
    if q == p(*HI[("rough", "weibo")]):
        w["fps"] = FPS
    return w, "hi"


def cut():
    s = []
    def add(t0, t1, what, burn=True, built=True):
        s.append(Shot(t0, t1, what, burn, built))
    at = ink_subs.at

    # 序: the outer window, the question sent, the video comes, the click, the card grows over the ident's first GROW seconds (film_ring.py)
    assert abs(ink_subs.chapters()[0][3] - HEAD_S) < 1e-6, "ink_subs' 序 and HEAD_S differ"
    add(0.0, HEAD_S, clip(p("ring", "open.mp4")), burn=False)

    # 一: the myth burns its own lines
    myth5, myth6 = p("myth", "myth_low5.mp4"), p("myth", "myth_low7.mp4") if os.path.exists(p("myth", "myth_low7.mp4")) else p("myth", "myth_low6.mp4")
    add(HEAD_S, at("鬼何以哭"), clip(myth5), burn=False)
    add(at("鬼何以哭"), at("依我之见"), clip(myth6), burn=False)

    # 二: the desk, the eight carriers a second each (film_two.py), then 「汇于一处」 coming in on the tablet to the first blow's framing (crew A)
    if os.path.exists(p("desk", "desk.mp4")):       # crew B: the daylight room, an even push to the books, the curtain behind the big line
        add(at("依我之见"), at("鬼之所以哭"), clip(p("desk", "desk.mp4")))
    else:
        add(at("依我之见"), at("鬼之所以哭"), card("二　书桌上的真书（待拍）\n「依我之见，并非如此」「我有一个偏私的解释」"), built=False)
    add(at("鬼之所以哭"), at("汇于一处"), clip(p("two", "two.mp4")))
    add(at("汇于一处"), at("古人说"), frames(p("imprint", "seq", "d0_tablet")))

    # 三: seven blows and the papers and screens between them, faster each time, on 三's own clock from 「古人说」 (PRESS_CUT, in frames)
    c = lambda f: at("古人说") + f / 30
    press = p("imprint", "seven", "seq_press")
    plays = dict(d1_letter=still(p("imprint", "d1_letter.png")), d2_poem=still(p("imprint", "d2_poem.png")), d4_quarto=still(p("imprint", "d4_quarto.png")),
                 d3_ruling=still(p("imprint", "d3_ruling.png")),
                 s4_term=frames(p("screens", "seq", "s4_key"), fps=60, first=0),       # the terminal typing gcd at twice its speed, frames 0 to 18
                 s4_keys=frames(p("screens", "seq", "s4_key"), fps=60, first=22),      # the keyboard, return going down on 38
                 s5_a=frames(p("screens", "seq", "s5_a"), fps=60, first=5),            # the post and the click on 发布, frames 5 to 17
                 dawn=clip(os.path.join(press, "dawn.mp4")), groove=clip(os.path.join(press, "groove.mp4")))
    for name, f0, f1 in PRESS_CUT:
        if name[0] == "b":
            k = int(name[1]) - 1
            lead = CONTACT_FRAME[k] - (PRESS_CONTACT[k] - f0)          # the clip's frame on the cut
            assert lead >= 0 and f0 <= PRESS_CONTACT[k] < f1, name
            add(c(f0), c(f1), clip(os.path.join(press, f"{name}.mp4"), at=lead / 30))
        else:
            add(c(f0), c(f1), plays[name])
    assert all(a[2] == b_[1] for a, b_ in zip(PRESS_CUT, PRESS_CUT[1:])) and abs(c(PRESS_CUT[-1][1]) - at("终于有一夜")) < 1e-6
    add(b(81.6), b(85.4), clip(p("wo", "end_low.mp4")))
    add(b(85.4), b(87.0), frames(p("screens", "seq", "s4_b")))
    add(b(87.0), b(89.6), frames(p("screens", "seq", "s5_b")))
    wo = next(x for x in (p("wo", "wo_low5.mp4"), p("wo", "wo_low4.mp4"), p("wo", "wo_low3.mp4")) if os.path.exists(x))      # low5: the post ends 「我是谁。」
    add(b(89.6), b(96.1), clip(wo))
    crab = next(x for x in (p("imprint", "crab", "crab_preview5.mp4"), p("imprint", "crab", "crab_preview4.mp4")) if os.path.exists(x))   # crew C: 5 is on the final groove
    add(at("字，成了精") - 1.0, at("字，成了精") + 15.3, clip(crab))
    add(b(112.4), b(126.0), frames(p("room", "seq3")))

    # 四: the reply goes on; one cut to the post nobody answered
    add(b(126.0), at("深夜里"), still(os.path.join(FILM, "room", "seq3", "room_0407.png")))
    ch = {c[0]: (c[2], c[3]) for c in ink_subs.chapters()}
    add(at("深夜里"), ch["四"][1], clip(p("four", "four.mp4")))           # crew D: the post's foot, the caret matched to 五's

    # 五六七: one clip from film_learn.py (crew E), no lines burned in it: the window, 思 by 張照, the run, the Tang page; then the page's last beat,
    # 低頭思故鄉 drawn off it (film_tang3d --eat, 2026-11-08), which the clip, made before it, does not have
    eat0 = ch["七"][1] - ink_subs.EAT_S
    assert abs(eat0 - at("熟读唐诗", 4.5)) < 1e-6
    add(ch["五"][0], eat0, clip(p("learn", "learn.mp4")))
    add(eat0, ch["七"][1], frames(p("learn", "tang", "eat")))

    # 八: the crab that has eaten, the block as it was, the dry bed
    add(at("其实"), at("这块泥"), clip(p("imprint", "crab", "eight2a.mp4")))          # crew C, 4.21 s
    add(at("这块泥"), at("河床"), still(p("imprint", "seven", "river", "river_first.png")))     # the whole block: the river's first frame held
    add(at("河床"), at("河床") + 6.0, clip(p("imprint", "seven", "river", "river.mp4")))
    add(at("河床") + 6.0, at("所以我说") + 6.5, clip(p("imprint", "crab", "eight2b.mp4")))  # crew C, 7.0 s
    add(at("所以我说") + 6.5, at("此刻"), clip(p("hand", "poem_1.mp4")))

    # 九: the city as keyframes, then the tunnel, which burns its own lines
    # the three rooms share 8.6 s (crew B); the tube takes its lines from at("每一扇窗里", -2.6)
    r0, r1 = at("此刻") + 1.4, at("每一扇窗里") - 3.0
    add(at("此刻"), r0, still(p("windows", "C1_clean.png"), STILL_PUSH))
    for i, n in enumerate(("C2A", "C2B", "C2C")):
        add(r0 + (r1 - r0) * i / 3, r0 + (r1 - r0) * (i + 1) / 3, still(p("windows", f"{n}_clean.png"), STILL_PUSH))
    add(r1, at("每一扇窗里") + 21.33, clip(p("windows", "tube_low3", "tube_low3.mp4")), burn=False)
    add(at("每一扇窗里") + 21.33, at("人最怕") - 1.5, still(p("windows", "K4e_clean.png"), STILL_PUSH, (0.5, 0.5)))

    # 十: the push into the new window at night, then windows 甲 and 乙 (crew B)
    rn = lambda *a: p("room_night", "seq", *a)
    add(at("人最怕") - 1.5, at("面对问题") - 6.5, clip(rn("ten.mp4")))
    add(at("面对问题") - 6.5, at("面对问题") + 2.8, clip(rn("jia.mp4")))
    add(at("面对问题") + 2.8, at("窗换了") + 4.0, clip(rn("yi.mp4")))
    add(at("窗换了") + 4.0, at("最难答"), clip(p("hand", "poem_2.mp4")))

    # 十一: five windows, each closer; the three things; the post; 痛 four times; the three sentences; the glass; window C (crew B)
    ev = lambda n: clip(p("eleven", "seq", n + ".mp4"))
    so = lambda n: clip(p("sorrow", "seq", n + ".mp4"))
    add(at("最难答"), at("感时花"), ev("w1"))
    add(at("感时花"), at("梧桐"), so("petal"))
    add(at("梧桐"), at("十年生死"), so("rain"))
    add(at("十年生死"), at("可每当"), so("lamp"))
    add(at("可每当"), at("还是某个"), ev("w2"))
    add(at("还是某个"), at("我本就是"), frames(p("rough", "weibo"), fps=LEARN_FPS))      # final: final/weibo at 30 fps
    add(at("我本就是"), at("我说的痛"), ev("w3"))
    add(at("我说的痛"), at("若我说"), ev("tong"))
    add(at("若我说"), at("处理某些"), ev("w4"))
    add(at("处理某些"), at("算不算"), ev("three"))
    add(at("算不算"), at("所以我选"), ev("blur"))
    add(at("所以我选"), at("这些以后再想"), ev("w5"))
    add(at("这些以后再想"), at("这些以后再想") + 17.0, clip(p("windows", "c11.mp4")))
    add(at("这些以后再想") + 17.0, at("人们急着"), clip(p("hand", "poem_3.mp4")))

    # 十二: the four names written, the room and the 9.11 window between them, the four sheets from above (crew D)
    add(ch["十二"][0], ch["十三"][0], clip(p("names", "twelve.mp4")))

    # 十三
    add(ch["十三"][0], ch["十四"][0], clip(p("thirteen", "crab13.mp4")))    # crew C: the crab walks into the gap between the four sheets and sits (film_crab_new.SHOT_S, the chapter)

    # 十四: the poem, the title on the same paper, out of fullscreen into the outer window, then the same window as the caret goes and the fade (the reverse shot cut 2026-11-08)
    end = ink_subs.end_at
    add(end("poem"), end("title"), clip(p("hand", "poem_4.mp4")), burn=False)
    add(end("title"), end("window"), clip(p("hand", "title.mp4")), burn=False)
    add(end("window"), end("fade"), clip(p("ring", "end.mp4")), burn=False)
    add(end("fade"), ink_subs.chapters()[-1][3], clip(p("ring", "fade.mp4")), burn=False)
    return s


def report():
    s = cut()
    total, cards = 0.0, 0.0
    for x in s:
        d = x.t1 - x.t0
        total += d
        cards += 0 if x.built else d
        name = x.what.get("path") or x.what.get("folder") or x.what.get("text", "").split("\n")[0]
        missing = "" if x.what["kind"] == "card" or os.path.exists(x.what.get("path") or x.what.get("folder")) else "  MISSING"
        print(f"{x.t0:7.2f} {x.t1:7.2f} {d:5.2f} {x.what['kind']:6s} {os.path.relpath(name, FILM) if x.what['kind'] != 'card' else name}{missing}")
    gaps = [(a.t1, b_.t0) for a, b_ in zip(s, s[1:]) if abs(a.t1 - b_.t0) > .02]
    print(f"\n片长 {total:.1f} s，占位卡 {cards:.1f} s，缝隙 {gaps}")


def _face(size):
    return ImageFont.truetype("/System/Library/Fonts/Supplemental/Songti.ttc", size, index=1)


def _card(text):
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    lines = text.split("\n")
    y = H * .40 - 17 * (len(lines) - 1)
    for i, line in enumerate(lines):
        d.text((W / 2, y), line, INKY if i == 0 else FAINT, font=_face(26 if i == 0 else 20), anchor="mm")
        y += 36
    return np.asarray(im)


def _fit(a):
    if a.shape[:2] == (H, W):
        return a
    return np.asarray(Image.fromarray(a).resize((W, H), Image.LANCZOS))


class Source:
    """The frames of a shot, in order, held on its last when it runs short."""

    def __init__(self, what):
        self.what, self.pipe, self.last, self.n = what, None, None, 0
        kind = what["kind"]
        if kind == "still":
            self.im = Image.open(what["path"]).convert("RGB")
            self.last = _fit(np.asarray(self.im))
        elif kind == "card":
            self.last = _card(what["text"])
        elif kind == "black":
            self.last = np.zeros((H, W, 3), np.uint8)
        elif kind == "clip":
            self.pipe = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(what["at"]), "-i", what["path"], "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        elif kind == "frames":
            self.files = sorted(f for f in os.listdir(what["folder"]) if f.endswith((".png", ".jpg")))

    def next(self):
        kind = self.what["kind"]
        if kind == "clip" and self.pipe:
            data = self.pipe.stdout.read(W * H * 3)
            if len(data) == W * H * 3:
                self.last = np.frombuffer(data, np.uint8).reshape(H, W, 3)
            else:
                self.close()
        elif kind == "still" and self.what.get("push"):
            # the camera nearer by push of the way each second, evenly, about at; drawn at the still's size, then brought to the frame's
            s = 1 / (1 - self.what["push"] * self.n / FPS)
            w, h = self.im.size
            ax, ay = self.what["at"][0] * w, self.what["at"][1] * h
            self.last = _fit(np.asarray(self.im.transform((w, h), Image.AFFINE, (1 / s, 0, ax - ax / s, 0, 1 / s, ay - ay / s), resample=Image.BICUBIC)))
        elif kind == "frames":
            i = self.what["first"] + int(self.n * self.what["fps"] / FPS)
            if i < len(self.files) and (self.last is None or int((self.n - 1) * self.what["fps"] / FPS) != int(self.n * self.what["fps"] / FPS)):
                self.last = _fit(np.asarray(Image.open(os.path.join(self.what["folder"], self.files[i])).convert("RGB")))
        self.n += 1
        return self.last

    def close(self):
        if self.pipe:
            self.pipe.stdout.close()
            self.pipe.wait()
            self.pipe = None


def burned(frame, t):
    if FINAL:
        return ink_subs.apply(frame, t)
    big = np.repeat(np.repeat(frame, 2, 0), 2, 1)
    out = ink_subs.apply(big, t)
    return ((out[::2, ::2].astype(np.uint16) + out[1::2, 1::2]) // 2).astype(np.uint8)


def encoder(path):
    q = ["-crf", "16", "-preset", "slow", "-tune", "grain"] if FINAL else ["-crf", "23", "-preset", "medium"]
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", *q,
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)


def grain(f, n):
    """One frame's grain: fresh each frame, a little coarser than a pixel, one grey for all three channels, weaker in the blacks and the whites."""
    g = np.random.default_rng(n).standard_normal((H, W), dtype=np.float32)
    g = (g + np.roll(g, 1, 0) + np.roll(g, 1, 1) + np.roll(np.roll(g, 1, 0), 1, 1)) * 0.5      # a 2 by 2 box, back to unit deviation
    x = f.astype(np.float32)
    lum = (x[..., 0] * .2126 + x[..., 1] * .7152 + x[..., 2] * .0722) / 255
    w = np.clip(4 * lum * (1 - lum), 0.3, 1.0)
    return np.clip(x + (g * w * GRAIN)[..., None] + 0.5, 0, 255).astype(np.uint8)


def write_shot(enc, x, a=0.0, b_=1e9, grained=False):
    """Everything of a shot between film seconds a and b_ goes to the encoder. A shot with its lines burned in already keeps its own.
    In the final cut the grain goes on before the lines: the lines are put on the film, not filmed."""
    what, how = full(x.what) if FINAL else (x.what, "")
    path = x.what.get("path") or x.what.get("folder") or ""
    burn = x.burn or (how == "hi" and tuple(os.path.relpath(path, FILM).split(os.sep)) in LINES_IN_LOW)
    src = Source(what)
    for n in range(int(round(x.t1 * FPS)) - int(round(x.t0 * FPS))):      # from the clock, so rounding never adds up into drift against the sound
        t = x.t0 + n / FPS
        f = src.next()
        if a <= t < b_:
            if FINAL and grained:
                f = grain(f, int(round(t * FPS)) + (10 ** 6 if t < 0 or x.t0 < 0 else 0))
            enc.stdin.write((burned(f, t) if burn else f).tobytes())
    src.close()


def grained(x):
    """In the video: 一 to the title's end. The outer window and the fade are not."""
    return HEAD_S - 1e-6 <= x.t0 and x.t1 <= ink_subs.end_at("window") + 1e-6


def film(a=None, b_=None):
    os.makedirs(OUT, exist_ok=True)
    whole = a is None
    a, b_ = (0.0, 1e9) if whole else (a, b_)
    name = "film_1080" if FINAL else "rough"
    path = os.path.join(OUT, f"{name}.mp4" if whole else f"{name}_{a:.0f}_{b_:.0f}.mp4")
    credits = next((x for x in (p("head", "credits_low3.mp4"), p("head", "credits_low2.mp4")) if os.path.exists(x)), None) if whole else None
    if whole and not credits:
        raise FileNotFoundError("no credits reel in head/")
    enc = encoder(path)
    ident = not whole
    for x in cut():
        if not ident and x.t0 >= HEAD_S:
            # the ident's reel after the part that played in the card, off the clock, between 序 and 一
            write_shot(enc, Shot(-(IDENT_S - GROW), 0.0, clip(p("head", "ident_1080.mp4"), at=GROW), burn=False), -1e9, 1e9, grained=True)
            ident = True
        if x.t1 > a and x.t0 < b_:
            write_shot(enc, x, a, b_, grained=grained(x))
            print(f"{x.t1:.1f}", flush=True)
    if whole:
        write_shot(enc, Shot(0.0, CREDITS_S, clip(credits), burn=False))      # final: full() takes credits_1080 when it is there
        write_shot(enc, Shot(0.0, BLACK_S, black(), burn=False))
        write_shot(enc, Shot(0.0, 3.0, clip(p("ring", "stinger.mp4")), burn=False))
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError(f"encoding failed: {path}")
    print(path)


def learn():
    import ink_learn
    folder = os.path.join(OUT, "learn")
    os.makedirs(folder, exist_ok=True)
    n = int((ink_subs.at("其实") - 141.6) * LEARN_FPS)
    for i in range(n):
        path = os.path.join(folder, f"l{i:04d}.jpg")
        if not os.path.exists(path):
            Image.fromarray(ink_learn.frame(141.6 + i / LEARN_FPS)).resize((W, H), Image.LANCZOS).save(path, quality=92)
            if i % 30 == 0:
                print(i, n, flush=True)


def weibo():
    """The post, with no line on it: the cut burns those. The final one at 30 fps, as png."""
    import film_weibo
    folder = os.path.join(OUT, "weibo")
    os.makedirs(folder, exist_ok=True)
    a = ink_subs.at("还是某个")
    fps = FPS if FINAL else LEARN_FPS
    for i in range(int((ink_subs.at("我本就是") - a) * fps)):
        im = Image.fromarray(np.array(film_weibo.weibo(a + i / fps)))
        im = im if im.size == (W, H) else im.resize((W, H), Image.LANCZOS)
        im.save(os.path.join(folder, f"w{i:04d}.png")) if FINAL else im.save(os.path.join(folder, f"w{i:04d}.jpg"), quality=92)


def final_list():
    for x in cut():
        w, how = full(x.what)
        path = w.get("path") or w.get("folder")
        name = os.path.relpath(path, FILM) if path else w.get("text", w["kind"]).split("\n")[0]
        burn = x.burn or (how == "hi" and tuple(os.path.relpath(x.what.get("path") or "", FILM).split(os.sep)) in LINES_IN_LOW)
        print(f"{x.t0:7.2f} {x.t1:7.2f} {how:4s} {'grain' if grained(x) else '     '} {'lines' if burn else '     '} {name}")
    for extra in (p("head", "ident_1080.mp4"), p("head", "credits_low3.mp4"), p("ring", "stinger.mp4")):
        w, how = full(clip(extra))
        print(f"{'':15s} {how:4s}       {os.path.relpath(w['path'], FILM)}")


def mux():
    video, audio = os.path.join(OUT, "film_1080.mp4"), p("sound", "film_mix.wav")
    out = os.path.join(OUT, "film_1080_mix.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                    "-movflags", "+faststart", out], check=True)
    print(out)


def sheet():
    tiles = []
    for x in cut():
        src = Source(x.what)
        f = None
        for _ in range(int(min((x.t1 - x.t0) * FPS / 2, 300))):
            f = src.next()
        f = src.next() if f is None else f
        src.close()
        tiles.append(np.asarray(Image.fromarray(f).resize((240, 135))))
    cols = 12
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * 135, cols * 240, 3), 30, np.uint8)
    for i, t in enumerate(tiles):
        out[(i // cols) * 135:(i // cols + 1) * 135, (i % cols) * 240:(i % cols + 1) * 240] = t
    Image.fromarray(out).save(os.path.join(OUT, "rough_sheet.png"))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "list"
    if what == "list":
        report()
    elif what == "learn":
        learn()
    elif what == "weibo":
        weibo()
    elif what == "cut":
        film(*(float(v) for v in sys.argv[2:4])) if len(sys.argv) > 3 else film()
    elif what == "sheet":
        sheet()
    elif what == "final":
        use_final()
        os.makedirs(OUT, exist_ok=True)
        sub = sys.argv[2] if len(sys.argv) > 2 else "list"
        if sub == "list":
            final_list()
        elif sub == "weibo":
            weibo()
        elif sub == "cut":
            film(*(float(v) for v in sys.argv[3:5])) if len(sys.argv) > 4 else film()
        elif sub == "mux":
            mux()
        else:
            raise SystemExit(__doc__)
    else:
        raise SystemExit(__doc__)
