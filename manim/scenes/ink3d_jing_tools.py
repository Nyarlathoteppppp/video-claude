"""Chapters 三 (成精) and 四 (我没有童年): what happens outside Blender. Runs in the render environment.

    python ink3d_jing_tools.py tex                    # the glyphs, the digits on the water and the things afloat -> ink/film_out/3d/tex_jr/*.npy
    python ink3d_jing_tools.py grade jing/pool ...    # NAME.png -> NAME_ink.png, with each shot's own ink settings
    python ink3d_jing_tools.py sheet jing             # ink/film_out/3d/jing/sheet.png
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from ink_film import OUT, font, glyph_mask
from ink_grade import FLOOR, INK_K, PAPER, fft_noise, grade, load, sm
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label, map_coordinates, zoom

ROOT3D = os.path.join(OUT, "3d")
TEX = os.path.join(ROOT3D, "tex_jr")


def save(name, a):
    os.makedirs(TEX, exist_ok=True)
    np.save(os.path.join(TEX, name + ".npy"), np.asarray(a, np.float32))


def digits_tile(px=4096, line=64, seed=5):
    """Rows of small numbers, the same model weights everywhere, each number a little brighter or dimmer than the next."""
    r = np.random.default_rng(seed)
    im = Image.new("L", (px, px), 0)
    d = ImageDraw.Draw(im)
    f = font("mono", 50)
    for y in range(0, px, line):
        x = int(r.uniform(-80, 0))
        while x < px:
            s = f"{r.normal(0, .7):+.2f}" if r.random() < .8 else f"{r.normal(0, .7):+.3f}"
            d.text((x, y + 2), s, int(r.uniform(90, 255)), font=f)
            x += int(f.getlength(s)) + 24
    return np.asarray(im, np.float32) / 255


def page_shiji(w=1500, h=2100):
    """One page of a history in woodblock print: vertical columns, a border, 太史公曰 large enough to catch."""
    r = np.random.default_rng(1)
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = 110, 150, w - 110, h - 170
    d.rectangle([x0, y0, x1, y1], outline=230, width=7)
    d.rectangle([x0 + 14, y0 + 14, x1 - 14, y1 - 14], outline=190, width=2)
    body = ("太史公曰余读孔氏书想见其为人适鲁观仲尼庙堂车服礼器诸生以时习礼其家余祗回留之不能去云天下君王至于贤人众矣当时则荣没则已焉孔子布衣传十余世学者宗之"
            "自天子王侯中国言六艺者折中于夫子可谓至圣矣太史公曰余登箕山其上盖有许由冢云孔子序列古之仁圣贤人如吴太伯伯夷之伦详矣余以所闻由光义至高其文辞不少概见何哉")
    n = 8
    cw = (x1 - x0 - 28) / n
    k = 0
    for c in range(n):
        x = x1 - 14 - cw * (c + 1)
        d.line([(x, y0 + 14), (x, y1 - 14)], fill=120, width=2)
        size = 186 if c == 0 else 92
        f = font("song", size)
        y = y0 + 40
        count = 4 if c == 0 else 18
        for i in range(count):
            ch = body[k % len(body)]
            k += 1
            d.text((x + (cw - size) / 2, y), ch, int(r.uniform(205, 255)), font=f)
            y += size * (1.16 if c == 0 else 1.06)
            if y > y1 - 110:
                break
    return np.asarray(im, np.float32) / 255


def page_letter(w=1500, h=2000):
    """A letter written by hand on ruled paper: the two lines that matter large, and the rest running on below them."""
    r = np.random.default_rng(2)
    im = Image.new("L", (w, h), 0)
    lines = [("亲爱的，", 230, 110), ("今晚月亮很好。", 208, 150), ("我站在院子里看了很久，", 118, 110), ("想告诉你，又不知道", 118, 110), ("从哪里说起。", 118, 110), ("你那边也能看见吗？", 118, 110)]
    y = 130
    ys = []
    for s_, size, x in lines:
        f = font("xing", size)
        px = x
        for ch in s_:
            o = Image.new("L", (size + 50, size + 70), 0)
            ImageDraw.Draw(o).text((16, 16), ch, int(r.uniform(215, 255)), font=f)
            o = o.rotate(float(r.normal(0, 2.4)), resample=Image.BICUBIC)
            im.paste(o, (int(px), int(y + r.normal(0, 4))), o)
            px += f.getlength(ch) * r.uniform(.88, .99)
        y += size + 62
        ys.append(y)
    a = np.asarray(im, np.float32) / 255
    for yy_ in ys:     # the writing paper's ruled lines
        a[yy_:yy_ + 3, 90:w - 90] = np.maximum(a[yy_:yy_ + 3, 90:w - 90], .30)
    return a


def page_verdict(w=1500, h=2100):
    """A court judgment: a printed title, printed body, and a round stamp, which in this world is ink."""
    r = np.random.default_rng(3)
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    f = font("song", 196)
    t = "判决书"
    d.text(((w - f.getlength("  ".join(t))) / 2, 130), "  ".join(t), 255, font=f)
    f2 = font("song", 64)
    body = ["本院认为，被告人的行为已构成", "犯罪，公诉机关指控的事实清楚，", "证据确实、充分。判决如下：", "", "一、被告人有期徒刑三年。", "二、如不服本判决，可在接到判决书", "之次日起十日内提出上诉。"]
    y = 410
    for s in body:
        d.text((130, y), s, int(r.uniform(225, 255)), font=f2)
        y += 104
    f3 = font("song", 66)
    d.text((w - 620, y + 90), "审判长    （署名）", 225, font=f3)
    d.text((w - 620, y + 190), "二〇一九年三月七日", 225, font=f3)
    a = np.asarray(im, np.float32) / 255
    # the stamp: a ring and a star, printed unevenly over the date
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    cx, cy, R = w - 400, y + 200, 205
    rr = np.hypot(xx - cx, yy - cy)
    ring = (np.abs(rr - R) < 11).astype(np.float32)
    ang = np.arctan2(yy - cy, xx - cx)
    star = (rr < 62 * (1 + .55 * np.cos(ang * 5 + math.pi / 2))).astype(np.float32)
    t2 = Image.new("L", (w, h), 0)
    td = ImageDraw.Draw(t2)
    fs = font("song", 54)
    for i, ch in enumerate("某某市人民法院"):
        an = math.radians(-150 + i * 20)
        ox, oy = cx + (R - 62) * math.cos(an) - 27, cy + (R - 62) * math.sin(an) - 27
        o = Image.new("L", (80, 80), 0)
        ImageDraw.Draw(o).text((13, 8), ch, 255, font=fs)
        o = o.rotate(-math.degrees(an) - 90, resample=Image.BICUBIC)
        t2.paste(o, (int(ox - 13), int(oy - 8)), o)
    st = np.maximum(np.maximum(ring, star), np.asarray(t2, np.float32) / 255)
    wear = np.clip(.72 + .34 * fft_noise(h, w, 1.1, 8), 0, 1)
    return np.maximum(a * .85, st * wear)


def page_code(w=1500, h=2100):
    """A sheet of printed code, tractor-feed: line numbers, indentation, braces. Anyone who has seen code knows it from across a room."""
    r = np.random.default_rng(4)
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    lines = ["def read(page):", "    for line in page:", "        if not line:", "            continue", "        words = line.split()", "        for w in words:", "            seen[w] += 1", "    return seen", "",
             "if __name__ == '__main__':", "    # TODO: why does this work", "    while True:", "        text = next(river)", "        read(text)", "", "# fixed the bug. do not touch", "x = x + 1"]
    f = font("mono", 66)
    fn = font("mono", 46)
    y = 150
    for i, s_ in enumerate(lines):
        d.text((150, y + 12), f"{i + 1:3d}", 140, font=fn)
        d.text((285, y), s_, int(r.uniform(205, 255)), font=f)
        y += 106
    for y_ in range(70, h - 40, 108):                                     # the feed holes down both sides
        for x_ in (46, w - 46):
            d.ellipse([x_ - 18, y_ - 18, x_ + 18, y_ + 18], fill=95)
    return np.asarray(im, np.float32) / 255


def page_forum(w=1500, h=2100):
    """A forum thread, printed: people quoting each other inside quotes, shouting. The argument is in the shape of the page."""
    r = np.random.default_rng(5)
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    posts = [("楼主", "你们根本不懂！！！", 0, 118), ("2楼", "笑死，你先读读原文", 40, 92), ("3楼", "回复2楼：你才该读！", 90, 104), ("4楼", "吵什么吵", 20, 84), ("5楼", "回复3楼：回复2楼：呵呵", 130, 92),
             ("楼主", "删了吧，没意思", 0, 110), ("7楼", "别删！我还没骂完！！", 70, 110), ("8楼", "……", 30, 84)]
    y = 90
    for who, s_, ind, size in posts:
        x = 90 + ind
        fw, f = font("song", 54), font("kaib", size)
        box_h = size + 120
        d.rectangle([x, y, w - 80, y + box_h], outline=170, width=4)
        d.text((x + 24, y + 12), who, 190, font=fw)
        d.text((x + 24, y + 76), s_, int(r.uniform(225, 255)), font=f)
        y += box_h + 38
    return np.asarray(im, np.float32) / 255


def phone_font(size):
    """A phone speaks in its system's sans, not in a brush. Hei is on every Mac."""
    for path in ("/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return font("kaib", size)


def phone_screen(w=1242, h=2688):
    """A chat in dark mode with one message in it, and nothing after: one lit bubble that lights the glass round it. The rest is the screen's own dim glow. Black round the edge is the bezel."""
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    bz = 34
    R = 118
    dx = np.maximum(np.abs(xx - w / 2) - (w / 2 - bz - R), 0)
    dy = np.maximum(np.abs(yy - h / 2) - (h / 2 - bz - R), 0)
    screen = sm((R - np.hypot(dx, dy)) / 2.0)
    by0, by1 = 930, 1270
    v = (.052 + .13 * np.exp(-((yy - (by0 + by1) / 2) / 760) ** 2)) * screen
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    f = phone_font(190)
    s = "有人吗？"
    tw = int(f.getlength(s))
    bx1 = w - 70
    bx0 = bx1 - tw - 160
    d.rounded_rectangle([bx0, by0, bx1, by1], 96, fill=255)                                 # the bubble
    d.polygon([(bx1 - 120, by1 - 40), (bx1 + 16, by1 + 34), (bx1 - 40, by1 - 90)], fill=255)  # its tail
    bubble = np.asarray(im, np.float32) / 255
    fld = Image.new("L", (w, h), 0)
    ImageDraw.Draw(fld).rounded_rectangle([96, h - 280, w - 96, h - 148], 66, outline=255, width=5)  # the empty field
    isl = Image.new("L", (w, h), 0)
    ImageDraw.Draw(isl).rounded_rectangle([w // 2 - 190, 76, w // 2 + 190, 186], 55, fill=255)      # the camera island
    tx = Image.new("L", (w, h), 0)
    ImageDraw.Draw(tx).text((bx0 + 80, by0 + 66), s, 255, font=f)
    t = np.asarray(tx, np.float32) / 255
    out = v * (1 - bubble) + .98 * bubble * (1 - t) + .012 * t
    out = np.maximum(out, np.asarray(fld, np.float32) / 255 * .10 * screen)
    out = out * (1 - np.asarray(isl, np.float32) / 255)
    return out


def cmd_tex():
    save("glyph_wo", glyph_mask("我", "kaib", 900, 40))
    save("glyph_jing", glyph_mask("精", "kaib", 900, 40))
    save("digits", digits_tile())
    f = font("kaib", 300)
    w = int(f.getlength("你好")) + 80
    im = Image.new("L", (w, 380), 0)
    ImageDraw.Draw(im).text((40, 30), "你好", 255, font=f)
    save("nihao", np.asarray(im, np.float32) / 255)
    save("shiji", page_shiji())
    save("letter", page_letter())
    save("verdict", page_verdict())
    save("code", page_code())
    save("forum", page_forum())
    save("phone", phone_screen())
    print("textures in", TEX)


# ---- the last step: each shot's ink ---------------------------------------------------------------------------------------------------------
def shafts(h, w, seed=3, x0=.46, spread=1.25, count=26):
    """Light falling through water, seen from below: soft rays that fan out from a point above the frame, brightest near the top."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    ox, oy = x0 * w, -h * .55
    ang = np.arctan2(xx - ox, yy - oy)
    v = np.zeros((h, w), np.float32)
    for _ in range(count):
        a0 = r.uniform(-spread / 2, spread / 2)
        wd = r.uniform(.006, .035)
        v += r.uniform(.2, 1) * np.exp(-((ang - a0) / wd) ** 2)
    v = gaussian_filter(v, 2.0)
    return v / v.max()


def water_from_below(h, w, seed=4, cx=.45, lo=.34):
    """What you see looking up from the bottom of the pool: the surface far overhead, a shaking ceiling of light, and rays fanning down from it."""
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    u, v = xx / w, yy / h
    # the broad light: strongest where the moon comes through, fading to ink in the corners and the deep
    g = np.exp(-(((u - cx) / .36) ** 2 + ((v - .22) / .66) ** 2))
    # the rays
    ray = shafts(h, w, seed, cx, 1.5, 34)
    # the ceiling, in perspective: the underside of the surface, soft bright cells squeezed toward the top of the frame
    t = 1 / (v + .16)
    X = (u - cx) * t * 48
    Y = t * 30
    n = fft_noise(1024, 1024, 3.0, seed + 3)
    n2 = fft_noise(1024, 1024, 2.6, seed + 5)
    net = map_coordinates(n, [Y % 1024, X % 1024], order=1, mode="wrap") + .45 * map_coordinates(n2, [(Y * 2.1) % 1024, (X * 2.3) % 1024], order=1, mode="wrap")
    net = np.clip(1 - np.abs(net) * 1.25, 0, 1) ** 2.2
    net = gaussian_filter(net, h / 1080 * 1.6)
    ceil = np.exp(-(v / .11) ** 2)
    L = lo + g ** .8 * (.62 + .42 * ray) + ceil * (.10 + .52 * net) * np.exp(-((u - cx) / .60) ** 2)
    return np.clip(L, 0, 1.0), ray


def comp_ink(shot, back=None, back_keep=.7, ray_over=.14, lo=.34, probe=None, k=.4, ink=.28):
    """Fore plate over back plate over the painted light, with water between them: the back plate is paler for the water in front of it.
    probe: instead of a Cycles plate, what the ink lets through from ink3d_jing.py --shot=probe, raised to k (.3: dens 2.1) and flat ink of grey ink. Thin ink lit by Cycles glows as bright as the light behind
    it and the turn disappears; lines of ink need to stay darker than the water."""
    if probe:
        T = np.load(os.path.join(ROOT3D, "jing", probe + ".npy")) ** k
        T = np.clip(zoom(T, (1080 / T.shape[0], 1920 / T.shape[1]), order=3), 0, 1)
        fg = np.dstack([np.full_like(T, ink)] * 3 + [1 - T])
    else:
        fg = load(os.path.join(ROOT3D, shot + ".png"))
    h, w = fg.shape[:2]
    L, ray = water_from_below(h, w, lo=lo)
    if back and os.path.exists(os.path.join(ROOT3D, back + ".png")):
        c = load(os.path.join(ROOT3D, back + ".png"))
        if c.shape[:2] != (h, w):
            c = zoom(c, (h / c.shape[0], w / c.shape[1], 1), order=1)
        a = c[..., 3] * back_keep
        L = L * (1 - a) + c[..., :3].mean(-1) * a
    a = fg[..., 3]
    L = L * (1 - a) + fg[..., :3].mean(-1) * a
    L = np.clip(L + ray_over * ray * np.clip(1.1 - np.linspace(0, 1, h)[:, None], 0, 1) * (1 - L), 0, 1)
    return np.dstack([L, L, L, np.ones_like(L)])


# the film's rule for every still, night included: mean brightness >= .30, under .08 at most 15%, at least 20% of the frame at .5 or above
SHOTS = {"jing/pool": dict(gamma=1.15), "jing/nihao": dict(gamma=1.3),
         "jing/before": dict(gamma=2.0, lo=.5), "jing/turn": dict(gamma=2.0, lo=.5, probe="probe_1.4_1_2", k=.3, ink=.3), "jing/wo": dict(gamma=1.5, back="jing/back"),
         "read/dawn": dict(gamma=1.1), "read/drift": dict(gamma=1.5), "read/drift2": dict(gamma=1.6), "read/phone": dict(gamma=1.35, clean=[(613, 455), (860, 432), (978, 785), (652, 790)], waterline=[(660, 779), (966, 777)])}
COMPOSED = ("jing/wo", "jing/before", "jing/turn")


def measure(path):
    """The rule, in numbers."""
    a = np.asarray(Image.open(path).convert("L"), np.float32) / 255
    ok = a.mean() >= .30 and (a < .08).mean() <= .15 and (a >= .5).mean() >= .20
    print(f"{os.path.relpath(path, ROOT3D):28s} mean {a.mean():.3f}  under.08 {(a < .08).mean():.3f}  over.5 {(a >= .5).mean():.3f}  {'ok' if ok else 'FAIL'}")


def clean_over(im, rgba, poly, waterline, gamma, water_gamma):
    """The phone is the one thing in this world that is not ink: no fibres under it, no dried edge. Its silhouette comes from its own black bezel, found inside poly (frame pixels) and closed along
    waterline, where the bezel goes under. Only a very fine grain is left on it, so it is in the same picture as the paper."""
    h, w = im.shape[:2]
    L = rgba[..., :3].mean(-1)
    rough = Image.new("L", (w, h), 0)
    ImageDraw.Draw(rough).polygon([tuple(p) for p in poly], fill=255)
    line = Image.new("L", (w, h), 0)
    ImageDraw.Draw(line).line([tuple(p) for p in waterline], fill=255, width=2)
    bezel = ((L < .22) & (np.asarray(rough) > 0)) | (np.asarray(line) > 0)
    solid = binary_fill_holes(binary_closing(bezel, iterations=2))
    lab, n = label(solid)
    solid = lab == 1 + np.argmax(np.bincount(lab.ravel())[1:])
    m = gaussian_filter(binary_dilation(solid, iterations=3).astype(np.float32), .7)[..., None]   # three pixels of plain water round it, where the grade's fibres had dragged its black out
    D = np.clip(1 - L, 0, 1) ** np.where(solid, gamma, water_gamma) * 3.1 * (1 + .03 * fft_noise(h, w, .7, 41))
    clean = np.maximum(PAPER[None, None, :] * np.exp(-D[..., None] * INK_K[None, None, :]), FLOOR[None, None, :])
    return np.clip(im * (1 - m) + clean * m, 0, 255).astype(np.uint8)


def cmd_grade(shot, **over):
    """shot may carry a test suffix, jing/wo@t14: same settings as jing/wo. A number in --back_keep etc. overrides the settings; back itself stays as set."""
    name = os.path.join(ROOT3D, shot.replace("@", "_"))
    base = shot.split("@")[0]
    st = dict(SHOTS[base])
    st.update(over)
    comp = {k: st.pop(k) for k in list(st) if k in ("back", "back_keep", "ray_over", "lo", "probe", "k", "ink")}
    clean = st.pop("clean", None), st.pop("waterline", None), st.pop("clean_gamma", 1.7)
    meta = json.load(open(name + ".json"))
    rgba = comp_ink(shot.replace("@", "_"), **comp) if base in COMPOSED else load(name + ".png")
    z = load(name + "_z.png") if os.path.exists(name + "_z.png") else np.zeros_like(rgba)
    im = grade(rgba, z, cam_z=meta["cam_z"], far=meta["far"], zmax=meta["zmax"], **st)
    if clean[0]:
        im = clean_over(im, rgba, *clean, st["gamma"])
    Image.fromarray(im).save(name + "_ink.png")
    measure(name + "_ink.png")


ORDER = {"jing": ["pool", "nihao", "before", "turn", "wo"], "read": ["dawn", "drift", "drift2", "phone"]}


def cmd_sheet(chapter):
    d = os.path.join(ROOT3D, chapter)
    names = [n + "_ink.png" for n in ORDER[chapter] if os.path.exists(os.path.join(d, n + "_ink.png"))]
    tw = 960
    rows_ = math.ceil(len(names) / 2)
    s = Image.new("RGB", (2 * tw, rows_ * tw * 9 // 16), (30, 30, 30))
    for i, n in enumerate(names):
        s.paste(Image.open(os.path.join(d, n)).resize((tw, tw * 9 // 16), Image.LANCZOS), ((i % 2) * tw, (i // 2) * tw * 9 // 16))
        measure(os.path.join(d, n))
    s.save(os.path.join(d, "sheet.png"))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "tex":
        cmd_tex()
    elif c == "grade":
        over = {a[2:].split("=")[0]: float(a.split("=")[1]) for a in sys.argv[2:] if a.startswith("--")}
        for s in sys.argv[2:]:
            if not s.startswith("--"):
                cmd_grade(s, **over)
    elif c == "sheet":
        cmd_sheet(sys.argv[2])
