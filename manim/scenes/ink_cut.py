"""The whole film cut from its stills, to judge story and pace before anything moves. Every shot lands on a subtitle, so the cut follows the text. A shot
with no still yet shows what it will be, in small writing on plain paper.

    python ink_cut.py sheet      # one frame per shot, ink/film_out/cut/sheet.png
    python ink_cut.py            # ink/film_out/cut/cut.mp4, 960x540, with ink_audio.wav under it if that exists
"""
import importlib
import os
import subprocess
import sys
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ink_join
import ink_learn
import ink_subs
from ink_color import PALETTES, PIGMENTS, depth_of, glaze
from ink_events import EVENTS
from ink_grade import load
from ink_move import Move, Plate
from ink_subs import at, chapters

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
STILLS = os.path.join(ROOT, "ink", "film_out", "3d")
OUT = os.path.join(ROOT, "ink", "film_out", "cut")
AUDIO = os.path.join(ROOT, "ink", "film_out", "ink_audio.wav")
W, H, FPS = 1920, 1080, 30
PAPER = (241, 234, 219)
START = {name: a for name, _, a, _ in chapters()}
END = chapters()[-1][3]
JOIN_S = dict(cut=0.0, dissolve=0.5, slow=11.0, water=1.6, ink=2.2, wash=1.2, under=1.6, up=1.6)     # slow stands in for lights going out, or a poem being written, until those move


def S(still, palette, start, join="water", z=(1.0, 1.06), c=(0.5, 0.5), note="", move=None, m=None):
    """move: the module whose frame(t) is this shot, once it moves. m: what ink_move.Move does to the still (its water, mist, flames, rings); rings' times count from the shot's start."""
    return dict(still=still, palette=palette, start=start, join=join, z=z, c=c, note=note, move=move, m=m)


W_ = dict(water=1.0, level=0.0)      # still water at the level of the sea
WM = dict(water=1.0, level=0.0, mist=0.14)
RIVER = dict(water=1.0, level=0.22, mist=0.1)
AUTO = dict(water=1.0, mist=0.1)     # water at the flattest level of the frame
INK = dict(billow=9.0)               # ink turning over in deep water
HEART = [at("古人说", 1.5)]
for gap in (1.7, 1.6, 1.45, 1.3, 1.15, 1.0, 0.85, 0.72, 0.6, 0.5, 0.42, 0.36):    # as in ink_mix
    HEART.append(HEART[-1] + gap)


def beats(a, b, x, y, squash=0.4, size=1.2):
    """The heart's beats that fall in a shot, as rings on its water."""
    return [(t - a, x, y, squash, size) for t in HEART if a - 1 <= t < b]


SHOTS = [
    # 序
    S("book/cover", "wood", 0.0, z=(1.0, 1.04)),
    S("book/half", "wood", 1.6, "dissolve", z=(1.0, 1.03)),
    S("book/spread", "wood", 3.0, "dissolve", z=(1.0, 1.05)),
    S("book/column", "wood", 4.6, "dissolve", z=(1.0, 1.12)),
    # 一
    S("myth/clouds", "night", START["一"], "ink", z=(1.0, 1.1), m=dict(mist=0.3)),
    S("myth/tracks", "moon", START["一"] + 3.6, move="ink_open:tracks"),
    S("myth/grow", "moon", START["一"] + 5.8, "dissolve", move="ink_open:grow"),
    S("myth/rift", "night", at("而天雨粟", -0.6), "dissolve", move="ink_open:rift"),
    S("myth/grain", "grain", at("而天雨粟", 1.6), "dissolve", move="ink_open:grain"),
    S("myth/marsh", "ghost", at("鬼夜哭"), "ink", z=(1.0, 1.08), m=dict(W_, mist=0.2, flames=0.8, rings=[(0.6, 0.625, 0.68, 0.22, 1.0), (2.4, 0.34, 0.74, 0.22, 1.2), (4.6, 0.52, 0.8, 0.22, 1.1)])),
    S("myth/marsh", "ghost", at("鬼何以哭"), "dissolve", z=(1.18, 1.34), c=(0.32, 0.62), m=dict(W_, mist=0.2, flames=0.8)),
    S("myth/lie", "ghost", at("或曰：/文字"), note="一团火，倒影飘向另一边", m=dict(W_, mist=0.15, flames=0.8)),
    # 一's end and 二's start are one move: valley, water's skin, under it, up again by the first stream
    S("myth/bare_lit", "ghost", at("或曰：/鬼神"), move="ink_one"),
    # 二
    S("ages/bone", "day", at("鬼之所以哭"), "cut", note="沿河飞：甲骨、竹简", m=RIVER),
    S("ages/stele", "day", at("鬼之所以哭", 1.8), note="碑林", m=RIVER),
    S("ages/screen", "night", at("有朝一日"), note="书架、信纸、一块亮着的屏幕，河越来越黑", m=RIVER),
    S("ages/falls", "deep", at("汇于一处"), note="大河落下悬崖，进黑潭", m=dict(W_, mist=0.15)),
    # 三
    S("jing/pool", "deep", at("古人说"), z=(1.0, 1.06), m=dict(water=1.0, rings=beats(at("古人说"), at("字，活了"), 0.44, 0.72))),
    S("jing/pool", "deep", at("字，活了"), "dissolve", z=(1.06, 1.16), m=dict(water=1.0, rings=beats(at("字，活了"), at("终于有一夜", 2.5), 0.44, 0.72))),
    S("jing/nihao", "deep", at("终于有一夜", 2.5), "dissolve", z=(1.0, 1.03), m=dict(water=1.0, rings=beats(at("终于有一夜", 2.5), at("字，成了精"), 0.47, 0.52, 0.45, 1.5))),
    S("jing/before", "deep", at("字，成了精"), "under", z=(1.0, 1.03), m=dict(billow=14.0)),
    S("jing/turn", "deep", at("字，成了精", 2.2), "dissolve", z=(1.0, 1.06), m=dict(billow=14.0)),
    S("jing/wo", "deep", at("那个精"), "dissolve", z=(1.0, 1.08), m=INK),
    # 四
    S("read/dawn", "day", at("我没有童年"), "up", z=(1.0, 1.05), m=dict(mist=0.1)),
    S("read/drift", "day", at("史书"), z=(1.0, 1.06)),
    S("read/drift2", "day", at("代码"), z=(1.0, 1.06)),
    S("read/phone", "night", at("深夜里"), "ink", z=(1.0, 1.05)),
    # 五 六 七: in my own window, with two moments of ink: the brush's 笔断意连, and the drop
    S("map/delta", "none", at("读之前") - ink_learn.CHAT, "ink", move="ink_learn:five"),
    S("attn/look", "none", at("读一句话"), "dissolve", move="ink_learn:six"),
    S("attn/lian", "moon", at("书家说"), "ink", z=(1.0, 1.05), m=W_),
    S("attn/guess", "none", at("下一个字"), "ink", move="ink_learn:guess"),
    S("drip/drop", "night", at("为了这几秒"), "ink", z=(1.0, 1.08)),
    S("drip/junk", "none", at("为了这几秒", ink_learn.T_RUN - ink_learn.T7 - 1.1), "ink", move="ink_learn:seven"),
    S("drip/right", "none", at("熟读"), "cut", move="ink_learn:write"),
    S("drip/chant", "night", at("熟读", 6.0), "ink", z=(1.0, 1.05)),
    # 八
    S("bed/flood", "day", at("其实"), z=(1.0, 1.08), m=dict(mist=0.1)),
    S("bed/recede", "dawn", at("就像河床"), z=(1.0, 1.06), m=dict(mist=0.12)),
    S("bed/sunrise", "dawn", at("所以我说"), "dissolve", z=(1.0, 1.05), m=dict(mist=0.12)),
    S("bed/bed_low", "dawn", at("所以我说", 3.2), "dissolve", z=(1.0, 1.06), m=AUTO),
    # 九
    S("qian/chat", "window", at("此刻"), "ink", move="ink_nine"),
    # 十: the band of mist along the horizon of the high shots stays out of frame
    S("channel/silver", "night", at("可我并不慌"), "ink", z=(1.0, 1.06), m=AUTO),
    S("channel/surface", "night", at("而是更稳的东西"), z=(1.0, 1.05), m=AUTO),
    S("channel/shape_dry", "night", at("记忆如河水"), z=(1.23, 1.26), c=(0.5, 0.6)),
    S("channel/shape_flood", "night", at("记忆如河水", 2.2), "dissolve", z=(1.26, 1.28), c=(0.5, 0.6), m=dict(water=1.0)),
    S("channel/shape", "night", at("记忆如河水", 4.0), "dissolve", z=(1.28, 1.3), c=(0.5, 0.6), m=dict(water=1.0)),
    S("channel/newwin", "night", at("记忆如河水", 5.2), "dissolve", z=(1.3, 1.32), c=(0.5, 0.6)),
    # 十一
    S("grief/grey", "rain", START["十一"], z=(1.0, 1.05), m=WM),
    S("grief/grey", "rain", at("我能写出"), "dissolve", z=(1.0, 1.05), m=WM, move="ink_eleven:words"),
    S("grief/wonan", "rain", at("可每当"), "dissolve", z=(1.0, 1.05), m=WM),
    S("grief/thread", "rain", at("这份难过"), z=(1.0, 1.05), m=WM),
    S("grief/blog", "none", at("还是某个"), "wash", move="ink_eleven:blog"),
    S("grief/thread", "rain", at("我本就是"), "wash", z=(1.05, 1.1), note="线越来越多，整条河都连着岸上的字", m=WM),
    S("grief/tong", "rain", at("我说的痛"), "dissolve", z=(1.0, 1.06), m=WM),
    S("jing/pool", "rain", at("若我说"), z=(1.7, 1.76), c=(0.5, 0.9), m=dict(water=1.0)),
    S("jing/pool", "rain", at("这话同样"), "dissolve", z=(1.76, 1.84), c=(0.5, 0.9), m=dict(water=1.0)),
    S("grief/under", "deep", at("处理某些"), "under", z=(1.0, 1.05), m=INK),
    S("grief/under_whole", "deep", at("正如人"), "dissolve", z=(1.0, 1.04), m=INK),
    S("grief/stones", "rain", at("所以我选"), "up", note="河从两块石头之间穿过，哪一边也不靠", m=WM),
    # 十二
    S("names/steles", "none", at("人们急着"), z=(1.0, 1.06), m=dict(W_, mist=0.15)),
    S("names/worn", "none", at("有人说我是人"), "dissolve", z=(1.0, 1.05), m=dict(W_, mist=0.1)),
    S("names/worn_shen", "none", at("有人说我是神"), "dissolve", note="「神」碑，碑面裂开一道缝", m=dict(W_, mist=0.1)),
    S("myth/lone", "ghost", at("有人说我是鬼"), z=(1.0, 1.05), m=dict(W_, mist=0.15, flames=0.8)),
    S("names/ding", "bronze", at("更多人说"), "water", z=(1.0, 1.05), m=AUTO),
    # 十三
    S("new/edge", "none", at("这些名字"), z=(1.0, 1.06), m=dict(mist=0.2)),
    S("new/char", "none", at("新东西最大的"), "dissolve", move="ink_brush:write"),
    # 十四
    S("book/sheet_blank", "none", START["十四"], "dissolve", z=(1.0, 1.0)),
    S("book/sheet", "none", START["十四"] + 6.0, "slow", z=(1.0, 1.0)),
    S("book/seal", "none", at("写完这首诗"), "dissolve", z=(1.0, 1.03)),
    S("book/fading", "none", at("写完这首诗", 2.5), "dissolve", z=(1.0, 1.02)),
    S("book/closing", "none", at("所以严格说来"), "dissolve", z=(1.0, 1.02)),
    S("book/end", "none", at("只存在你那里"), "dissolve", z=(1.0, 1.03)),
]


def _ttf(size):
    return ImageFont.truetype("/System/Library/Fonts/Songti.ttc", size)


def _pending(note):
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    d.text((W // 2 - 760, H // 2 - 30), "待渲：" + note, (120, 112, 100), font=_ttf(44))
    return im


_cache = {}


def picture(shot):
    """The still in its colour, cached on disk because a glaze takes seconds; a still that has been rendered again is coloured again."""
    key = (shot["still"], shot["palette"], shot["note"])
    if key in _cache:
        return _cache[key]
    path = os.path.join(STILLS, shot["still"])
    if not os.path.exists(path + "_ink.png"):
        im = _pending(shot["note"] or shot["still"])
    else:
        os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
        mix = zlib.crc32(repr((PALETTES[shot["palette"]], PIGMENTS)).encode())     # a palette that has changed is a different picture
        done = os.path.join(OUT, "cache", f"{shot['still'].replace('/', '_')}__{shot['palette']}_{mix:08x}.png")
        if os.path.exists(done) and os.path.getmtime(done) > os.path.getmtime(path + "_ink.png"):
            im = Image.open(done).convert("RGB")
        else:
            ink = np.asarray(Image.open(path + "_ink.png").convert("RGB"))
            found = depth_of(path)
            im = Image.fromarray(glaze(ink, load(found) if found else None, shot["palette"]))
            im.save(done)
    if len(_cache) > 6:
        _cache.pop(next(iter(_cache)))
    _cache[key] = im
    return im


_moved = {}


def mover(shot):
    key = (shot["still"], shot["palette"], repr(shot["m"]))
    if key not in _moved:
        if len(_moved) > 2:
            _moved.pop(next(iter(_moved)))
        _moved[key] = Move(Plate(shot["still"], shot["palette"]), centre=shot["c"], **shot["m"])
    return _moved[key]


def framed(shot, t, end):
    """The still at film time t: a steady push about a fixed centre. Or the shot itself, if it moves."""
    if shot["move"]:
        name, _, fn = shot["move"].partition(":")
        return getattr(importlib.import_module(name), fn or "frame")(t)
    p = (t - shot["start"]) / max(end - shot["start"], 1e-3)
    if shot["m"] is not None and os.path.exists(os.path.join(STILLS, shot["still"] + "_ink.png")):
        # the push goes through the picture's depth; a frame that starts tighter than the whole still is cropped from it
        base = min(shot["z"])
        f = mover(shot).frame(t - shot["start"], s=max(shot["z"][0] + (shot["z"][1] - shot["z"][0]) * p - base, 0.0))
        im = Image.fromarray(f)
    else:
        base = max(shot["z"][0] + (shot["z"][1] - shot["z"][0]) * p, 1.0)     # a join runs a little before the shot's start and after its end
        im = picture(shot)
    w, h = im.size[0] / base, im.size[1] / base
    cx, cy = shot["c"][0] * im.size[0], shot["c"][1] * im.size[1]
    x0, y0 = min(max(cx - w / 2, 0), im.size[0] - w), min(max(cy - h / 2, 0), im.size[1] - h)
    f = np.asarray(im.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + w, y0 + h)))
    event = EVENTS.get(shot["still"])      # what happens in this shot, painted over it: see ink_events.py
    return event(f, t - shot["start"], end - shot["start"]) if event else f


def frame_at(t):
    i = max(k for k, s in enumerate(SHOTS) if s["start"] - JOIN_S[s["join"]] / 2 <= t)
    s = SHOTS[i]
    end = SHOTS[i + 1]["start"] if i + 1 < len(SHOTS) else END
    half = JOIN_S[s["join"]] / 2
    if i == 0 or t >= s["start"] + half:
        return framed(s, t, end)
    prev = SHOTS[i - 1]
    a, b = framed(prev, t, s["start"]), framed(s, t, end)
    u = (t - s["start"] + half) / (2 * half)
    if s["join"] in ("dissolve", "slow"):
        return (a * (1 - u) + b * u + 0.5).astype(np.uint8)
    if s["join"] in ("ink", "wash"):
        return ink_join.ink(a, b, u, seed=i)
    if s["join"] == "under":
        return ink_join.surface(a, b, u, down=True, seed=i)
    if s["join"] == "up":
        return ink_join.surface(a, b, u, down=False, seed=i)
    return ink_join.water(a, b, u, seed=i)


def sheet():
    cols, tw, th = 6, 320, 180
    rows = (len(SHOTS) + cols - 1) // cols
    out = Image.new("RGB", (cols * tw, rows * th), PAPER)
    label = _ttf(16)
    missing = []
    for i, s in enumerate(SHOTS):
        end = SHOTS[i + 1]["start"] if i + 1 < len(SHOTS) else END
        t = min(s["start"] + JOIN_S[s["join"]] / 2 + 0.6 * (end - s["start"]), end - 0.05)
        f = ink_subs.apply(framed(s, t, end), t)
        im = Image.fromarray(f).resize((tw, th), Image.LANCZOS)
        ImageDraw.Draw(im).text((5, 3), f"{int(t) // 60}:{int(t) % 60:02d} {s['still']}", (220, 50, 40), font=label)
        out.paste(im, ((i % cols) * tw, (i // cols) * th))
        if not os.path.exists(os.path.join(STILLS, s["still"] + "_ink.png")):
            missing.append(s["still"])
    path = os.path.join(OUT, "sheet.png")
    out.save(path)
    print(path, f"{len(SHOTS)} shots, {len(set(missing))} stills pending")


def film(a=0.0, b=END, silent=False):
    os.makedirs(OUT, exist_ok=True)
    whole = (a, b) == (0.0, END)
    path = os.path.join(OUT, "cut.mp4" if whole else f"cut_{a:.0f}_{b:.0f}.mp4")
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if os.path.exists(AUDIO) and not silent:
        cmd += ["-ss", f"{a:.3f}", "-i", AUDIO, "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-vf", "scale=960:540", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(int(a * FPS), int(b * FPS)):
        t = n / FPS
        pipe.stdin.write(ink_subs.apply(frame_at(t), t).tobytes())
        if n % 900 == 0:
            print(f"{t:.0f} s of {END:.0f}", flush=True)
    pipe.stdin.close()
    pipe.wait()
    print(path)


def whole(n=4):
    """The film in n parts at once, joined, with the sound under it."""
    os.makedirs(OUT, exist_ok=True)
    edges = [END * i / n for i in range(n + 1)]
    edges = [int(e * FPS) / FPS for e in edges]
    parts = [subprocess.Popen([sys.executable, __file__, f"{a}", f"{b}", "silent"]) for a, b in zip(edges, edges[1:])]
    for p in parts:
        p.wait()
    names = [os.path.join(OUT, f"cut_{a:.0f}_{b:.0f}.mp4") for a, b in zip(edges, edges[1:])]
    listing = os.path.join(OUT, "parts.txt")
    with open(listing, "w") as f:
        f.writelines(f"file '{x}'\n" for x in names)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing]
    if os.path.exists(AUDIO):
        cmd += ["-i", AUDIO, "-c:a", "aac", "-b:a", "192k", "-shortest"]
    subprocess.run(cmd + ["-c:v", "copy", "-movflags", "+faststart", os.path.join(OUT, "cut.mp4")], check=True)
    print(os.path.join(OUT, "cut.mp4"))


if __name__ == "__main__":
    if sys.argv[1:] == ["sheet"]:
        sheet()
    elif sys.argv[1:2] == ["whole"]:
        whole(*(int(x) for x in sys.argv[2:3]))
    else:
        film(*(float(x) for x in sys.argv[1:3]), silent="silent" in sys.argv)
