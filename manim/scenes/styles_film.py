"""The style reel: thirteen 5-second clips, one per entry of 风格库.md, each on the subject that shows that style best, opening
on a wall of all of them at once, then one by one with a transition made for each pair and a split-flap counter down the
right edge, and closing on the wall again. Each clip is a function of t in
style_SLUG.py; 等轴 2.5D, 3D 渲染 and 液态流动 have no module and only get their finish here, over frames from styles_blender.py.
A frame is a pure function of t, as in the other films.

    PY=python3        # a Python with pycairo, numpy, scipy, Pillow (and ffmpeg on PATH)
    $PY styles_film.py clip [SLUG ...] [--jobs 18]    # clips -> styles/film_out/SLUG.mp4 (all if none named)
    $PY styles_film.py snap SLUG t [t ...]            # frames -> styles/film_out/snaps/SLUG_t.png, to look at
    $PY styles_film.py stills [--allow-missing]       # one chosen frame per clip + overview -> docs/img/styles/
    $PY styles_film.py reel [--allow-missing]         # clips -> styles/film_out/reel.mp4
"""
import argparse
import importlib
import math
import os
import subprocess
import sys
from functools import lru_cache

import cairo
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from styles_common import *


# ======================================================================================
# the Blender clips with no module of their own: their finish over the rendered frames
# ======================================================================================
def port(t):
    return surface(seq("port", t))


def letterpress(t):
    """The plate holds the blank sheet for 26 frames: shown in 9, and the time saved rests on the printed page, pushing in."""
    g = t * FPS
    rgb = seq("letterpress", round(float(np.interp(g, (0, 80, 89, 132), (0, 80, 106, 149)))) / FPS)
    if g <= 132:
        return surface(rgb)
    z = 0.01 * smooth(seg(g, 132, 149))                     # a crop this small keeps the type crisp
    box = (W * z, H * z, W * (1 - z), H * (1 - z))
    page = Image.fromarray((rgb * 255 + 0.5).astype(np.uint8)).resize((W, H), Image.BICUBIC, box=box)
    return surface(np.asarray(page, np.float32) / 255)


def splash(t):
    return surface(seq("splash", t))


STYLES = [("handdrawn", "逐帧手绘"), ("port", "等轴 2.5D"), ("flat", "扁平矢量"), ("line", "线条动画"), ("letterpress", "3D 渲染"),
          ("morph", "形变动画"), ("sticker", "贴纸风格"), ("cyberhud", "赛博朋克 HUD"), ("collage", "拼贴剪贴"),
          ("glass", "弥散渐变 + 玻璃拟态"), ("synthwave", "复古 synthwave"), ("pixel", "像素"), ("splash", "液态流动")]
BY_SLUG = {"port": port, "letterpress": letterpress, "splash": splash}


def clip_fn(slug):
    """A clip's frame function: SLUG(t) from style_SLUG.py if it has a module of its own, otherwise one of those above."""
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"style_{slug}.py")):
        return getattr(importlib.import_module(f"style_{slug}"), slug)
    return BY_SLUG[slug]


# ======================================================================================
# rendering
# ======================================================================================
def frame_bytes(surf):
    return bytes(view(surf))


def part(job):
    slug, a, b, path = job
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "15", "-preset", "medium",
                          "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    fn = clip_fn(slug)
    for f in range(a, b):
        p.stdin.write(frame_bytes(fn(f / FPS)))
    p.stdin.close()
    p.wait()


def cmd_clip(slugs, jobs):
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    for slug in slugs:
        size = FRAMES // jobs + 1
        tasks = [(slug, a, min(FRAMES, a + size), os.path.join(OUT, f".{slug}_{a:04d}.mp4")) for a in range(0, FRAMES, size)]
        with Pool(jobs) as pool:
            pool.map(part, tasks)
        lst = os.path.join(OUT, f".{slug}.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{os.path.basename(t[3])}'\n" for t in tasks)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                        os.path.join(OUT, slug + ".mp4")], check=True)
        for t in tasks:
            os.remove(t[3])
        os.remove(lst)
        print("clip", slug, flush=True)


def cmd_snap(slug, times):
    d = os.path.join(OUT, "snaps")
    os.makedirs(d, exist_ok=True)
    for t in times:
        p = os.path.join(d, f"{slug}_{t:05.2f}.png")
        clip_fn(slug)(t).write_to_png(p)
        print(p, flush=True)


# Editable MP4 sampling times, in seconds; unspecified styles use 3.5 s.
STILL_TIMES = {
    "handdrawn": 2.93, "port": 3.0, "flat": 0.47, "line": 3.6,     # koi through its gold whirl; lift-off, cloud on both sides; both wings of the attractor
    "letterpress": 4.0, "morph": 4.67, "sticker": 4.6, "cyberhud": 3.4,     # 山 with its seal
    "collage": 1.5, "glass": 3.2, "synthwave": 0.4, "pixel": 3.0,     # the paper wave before it falls; red sun half behind frost; car under the sun, before any tracking band; junk inside the sun
    "splash": 3.3,  # the crown at its brightest, spray and threads all out
}
INK, CREAM, MUTED, GOLD = "#0F0E0C", "#F4EFE4", "#A39C8E", "#C9A66B"
NUMERALS = "零一二三四五六七八九十"


def _numeral(n):
    return NUMERALS[n] if n <= 10 else "十" + NUMERALS[n - 10]


def _film_clips(allow_missing=False):
    clips, missing = [], []
    for slug, name in STYLES:
        path = os.path.join(OUT, slug + ".mp4")
        if os.path.isfile(path):
            clips.append((slug, name, path))
        else:
            missing.append(path)
    if missing and not allow_missing:
        raise RuntimeError("Missing clip file(s):\n" + "\n".join(missing))
    for path in missing:
        print("Skipping missing clip: " + path, file=sys.stderr)
    if not clips:
        raise RuntimeError("No finished clips found in " + OUT)
    return clips


class _FilmReader:
    """Bounded rawvideo pipe: one Cairo surface per read, never a clip cache. A seek can land a frame short of the
    end of a clip, so the last frame is held for up to three reads rather than failing a whole reel over it."""
    def __init__(self, path, w=W, h=H, frames=FRAMES, start=0.0, flags="lanczos"):
        self.path, self.w, self.h = path, w, h
        self.remaining, self.closed, self.held = frames, False, None
        self.proc = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "fatal", "-nostdin",
            "-threads", "1", "-ss", f"{start:.9f}", "-i", path,
            "-map", "0:v:0", "-an", "-sn", "-dn", "-filter_threads", "1",
            "-vf", f"fps={FPS},scale={w}:{h}:flags={flags},setsar=1",
            "-frames:v", str(frames), "-threads", "1", "-f", "rawvideo",
            "-pix_fmt", "bgra", "pipe:1"], stdout=subprocess.PIPE, bufsize=0)

    def read(self):
        if self.closed or self.remaining <= 0:
            raise RuntimeError("Read past requested frames: " + self.path)
        surf, _ = canvas(None, self.w, self.h)
        surf.flush()
        data = memoryview(surf.get_data()).cast("B")
        offset = 0
        while offset < len(data):
            n = self.proc.stdout.readinto(data[offset:])
            if not n:
                if self.held is None or self.remaining > 3:
                    raise RuntimeError(f"Incomplete video frame in {self.path} "
                                       f"({self.remaining} requested frames still unread)")
                data[:] = self.held
                break
            offset += n
        surf.mark_dirty()
        self.remaining -= 1
        if self.remaining <= 4:
            self.held = bytes(data)
        return surf

    def close(self, abort=False):
        if self.closed:
            return
        self.closed = True
        if (abort or self.remaining) and self.proc.poll() is None:
            self.proc.terminate()
        self.proc.stdout.close()
        rc = self.proc.wait()
        if rc and not abort and not self.remaining:
            raise RuntimeError(f"ffmpeg decoder exited {rc}: {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, typ, value, tb):
        self.close(abort=typ is not None)


def _film_jpeg(surf, path):
    Image.frombytes("RGBA", (surf.get_width(), surf.get_height()),
                    frame_bytes(surf), "raw", "BGRA").convert("RGB").save(
                        path, "JPEG", quality=86, subsampling=0)


def _fine(c):
    """Unhinted type: tracked and turned letters would otherwise sit unevenly."""
    fo = cairo.FontOptions()
    fo.set_antialias(cairo.ANTIALIAS_GRAY)
    fo.set_hint_style(cairo.HINT_STYLE_NONE)
    fo.set_hint_metrics(cairo.HINT_METRICS_OFF)
    c.set_font_options(fo)


def _tracked(c, s, x, y, size, col, face=SERIF, bold=False, track=0.0, align="left", a=1.0):
    """Baseline text with letter-spacing, in ems, which cairo's toy API doesn't offer. Returns the width."""
    font(c, face, size, bold)
    adv = [c.text_extents(ch).x_advance + track * size for ch in s]
    total = sum(adv) - track * size
    x -= {"left": 0, "center": total / 2, "right": total}[align]
    c.set_source_rgba(*hx(col), a)
    for ch, w in zip(s, adv):
        c.move_to(x, y)
        c.show_text(ch)
        x += w
    c.new_path()
    return total


def _vertical(c, s, cx, y, size, col, a=1.0):
    """A name set top to bottom, as Chinese is: characters upright, Latin runs turned a quarter clockwise. Returns the end."""
    c.set_source_rgba(*hx(col), a)
    run = ""
    for ch in s + " ":
        if ch.isascii() and ch not in " +":
            run += ch
            continue
        if run:
            font(c, FUTURA, size * 0.8)
            w = c.text_extents(run).x_advance + (len(run) - 1) * size * 0.06
            c.save()
            c.translate(cx - size * 0.27, y + size * 0.1)
            c.rotate(math.pi / 2)
            x = 0.0
            for r in run:
                c.move_to(x, 0)
                c.show_text(r)
                x += c.text_extents(r).x_advance + size * 0.06
            c.restore()
            y += w + size * 0.34
            run = ""
        if ch == " ":
            y += size * 0.16
        else:
            font(c, SERIF, size)
            e = c.text_extents(ch)
            c.move_to(cx - e.x_advance / 2, y + size * 0.86)
            c.show_text(ch)
            y += size * 1.2
    c.new_path()
    return y


def cmd_stills(allow_missing=False):
    clips = _film_clips(allow_missing)
    os.makedirs(DOCS, exist_ok=True)
    rows = (len(clips) + 3) // 4
    sheet, c = canvas("#F4EFE4", 1600, 56 + rows * 264 + 8)
    c.set_antialias(cairo.ANTIALIAS_BEST)
    _fine(c)
    for i, (slug, name, path) in enumerate(clips):
        # Align the seek to the source's 30 fps frame grid.
        t = min(FRAMES - 1, max(0, round(STILL_TIMES.get(slug, 3.5) * FPS))) / FPS
        with _FilmReader(path, 1600, 900, frames=1, start=t) as reader:
            still = reader.read()
        dest = os.path.join(DOCS, slug + ".jpg")
        _film_jpeg(still, dest)
        x, y = 40 + (i % 4) * 384, 40 + (i // 4) * 264
        c.save()
        c.translate(x, y)
        c.scale(368 / 1600, 207 / 900)
        c.set_source_surface(still)
        c.paint()
        c.restore()
        _tracked(c, f"{i + 1:02d}", x, y + 236, 21, "#8B8375", DIN, True, 0.06)
        _tracked(c, name, x + 34, y + 236, 20, "#1C1A17", SERIF, False, 0.06)
        print(f"{dest}  (t={t:.3f}s)", flush=True)
    i = len(clips)
    if i % 4:
        x, y = 40 + (i % 4) * 384, 40 + (i // 4) * 264
        _tracked(c, "风格库", x - 2, y + 158, 58, "#1C1A17", SERIF, True, 0.14)      # on the column's left edge, seated on the tile's bottom edge
        _tracked(c, f"{_numeral(i)}种画法", x, y + 205, 24, "#5F594E", SERIF, False, 0.12)
    dest = os.path.join(DOCS, "overview.jpg")
    _film_jpeg(sheet, dest)
    print(dest, flush=True)


# ------------------------------------------------------------------ the reel's own furniture
LABEL_X, LABEL_Y = W - 52, 132                                # the column on the right edge is the one clear of every clip's own HUD
FLAP_W, FLAP_H = 26, 40


@lru_cache(maxsize=2)
def _label_shade(peak=0.62, power=1.6):
    """A soft falloff against the frame's edge: painted, it darkens footage for cream type; as a mask, it clears a page's margin."""
    x = np.arange(W - 190, W, dtype=np.float32)[None, :]
    y = np.arange(0, 680, dtype=np.float32)[:, None]
    ramp = np.clip((x - (W - 190)) / 190, 0, 1) ** power
    edge = np.clip((y - 60) / 90, 0, 1) * np.clip((660 - y) / 110, 0, 1)
    return surface(np.zeros((680, 190, 3), np.float32), ramp * (edge * edge * (3 - 2 * edge)) * peak)


def _halo(s, sigma, strength, dark):
    """A tight halo around the type on s: cream behind ink, black behind cream."""
    v = view(s).astype(np.float32) / 255
    halo = np.clip(gaussian_filter(v[..., 3], sigma) * 1.5, 0, 1) * strength
    alpha = v[..., 3] + (1 - v[..., 3]) * halo
    tint = np.array(hx(CREAM) if dark else (0.0, 0.0, 0.0), np.float32)[::-1]     # BGR, as the surface stores it
    out = view(s)
    out[..., :3] = np.round(np.clip(v[..., :3] + ((1 - v[..., 3]) * halo)[..., None] * tint, 0, 1) * 255).astype(np.uint8)
    out[..., 3] = np.round(alpha * 255).astype(np.uint8)
    s.mark_dirty()
    return s


@lru_cache(maxsize=32)
def _label_name(name, total, dark=False):
    """The name, its hairline and the total, with a tight halo: cream on dark footage, ink on a white page.
    Pre-rendered: the blur is too dear per frame."""
    s, c = canvas(None, 190, 680)
    c.set_antialias(cairo.ANTIALIAS_BEST)
    _fine(c)
    cx = 190 - 52
    ink = "#1A1815" if dark else CREAM
    _tracked(c, f"{total:02d}", cx, LABEL_Y + 64, 20, ink, DIN, True, 0.08, "center", 0.72)
    c.set_source_rgba(*hx(ink), 0.5)
    c.rectangle(cx - 0.5, LABEL_Y + 74, 1, 30)
    c.fill()
    _vertical(c, name, cx, LABEL_Y + 112, 30, ink)
    return _halo(s, 2.4, 0.5 if dark else 0.72, dark)


class _Tone:
    """The ground under some type, followed from frame to frame so the type never flips in one."""
    ink = paper = None

    def follow(self, target, ground):
        self.ink = target if self.ink is None else self.ink + (target - self.ink) * 0.18
        self.paper = ground if self.paper is None else self.paper + (ground - self.paper) * 0.18


def _ground(pixels):
    """How far a patch of picture asks for dark type (0 on footage, 1 on a page) and the colour of its paper.
    Judged by the paper rather than the mean, so a densely printed page still counts as a page."""
    ground = pixels.astype(np.float32) / 255
    paper = np.percentile(ground.reshape(-1, 3), 85, axis=0)[::-1]           # stored BGR
    return smooth((np.percentile(ground.mean(axis=2), 60) - 0.62) / 0.08), paper


def _flap_half(c, x, y, digit, upper, fold=1.0, shade=0.0):
    """Half of one split-flap leaf; fold squashes it about the hinge as it turns."""
    if fold <= 0.01:
        return
    c.save()
    c.translate(x, y + FLAP_H / 2)
    c.scale(1, fold)
    if upper:
        c.rectangle(0, -FLAP_H / 2, FLAP_W, FLAP_H / 2 - 0.6)
    else:
        c.rectangle(0, 0.6, FLAP_W, FLAP_H / 2 - 0.6)
    c.clip()
    g = cairo.LinearGradient(0, -FLAP_H / 2, 0, FLAP_H / 2)
    for pos, col in ((0.0, "#3A362F"), (0.49, "#24211D"), (0.51, "#1B1916"), (1.0, "#11100E")):
        g.add_color_stop_rgb(pos, *hx(col))
    rrect(c, 0, -FLAP_H / 2, FLAP_W, FLAP_H, 3)
    c.set_source(g)
    c.fill()
    font(c, DIN, 36, True)
    e = c.text_extents(digit)
    c.set_source_rgb(*hx(CREAM))
    c.move_to(FLAP_W / 2 - e.width / 2 - e.x_bearing, -(e.y_bearing + e.height / 2))
    c.show_text(digit)
    c.new_path()
    if shade > 0:
        c.set_source_rgba(0, 0, 0, shade)
        c.paint()
    c.restore()


def _flap(c, x, y, old, new, phase):
    """One leaf: the old top falls over the hinge, the new bottom comes down after it."""
    c.set_source_rgba(0, 0, 0, 0.4)
    rrect(c, x - 1, y - 1, FLAP_W + 2, FLAP_H + 3, 4)
    c.fill()
    if old == new or phase >= 1:
        _flap_half(c, x, y, new, True)
        _flap_half(c, x, y, new, False)
    else:
        _flap_half(c, x, y, new, True)
        _flap_half(c, x, y, old, False)
        if phase < 0.5:
            _flap_half(c, x, y, old, True, 1 - phase * 2, phase * 1.1)
        else:
            _flap_half(c, x, y, new, False, phase * 2 - 1, (1 - phase) * 1.1)
    for px in (x - 1.5, x + FLAP_W - 0.5):                    # the pins it turns on
        c.set_source_rgb(*hx("#0A0908"))
        c.rectangle(px, y + FLAP_H / 2 - 3, 2, 6)
        c.fill()


def _reel_label(surf, number, total, name, previous=None, phase=1.0, previous_name=None, a=1.0, ink=0.0, paper=(1.0, 1.0, 1.0)):
    """The counter and the style's name, down the right edge. Only a digit that changes turns over."""
    c = cairo.Context(surf)
    c.set_antialias(cairo.ANTIALIAS_BEST)
    _fine(c)
    if a < 1:
        c.rectangle(W - 190, 0, 190, 680)
        c.clip()
        c.push_group()
    if ink < 1:
        c.set_source_surface(_label_shade(), W - 190, 0)
        c.paint_with_alpha(1 - ink)
    if ink > 0:                                                  # the page's own paper, so its print falls away towards the margin
        c.set_source_rgba(*paper, ink)
        c.mask_surface(_label_shade(0.9, 1.0), W - 190, 0)

    def lay(label, alpha):
        for dark, k in ((False, 1 - ink), (True, ink)):
            if k * alpha > 0.004:
                c.set_source_surface(_label_name(label, total, dark), W - 190, 0)
                c.paint_with_alpha(k * alpha)

    old = f"{number if previous is None else previous:02d}"
    new = f"{number:02d}"
    x = LABEL_X - FLAP_W - 1
    turn = smooth(phase) if phase < 1 else 1.0
    _flap(c, x, LABEL_Y, old[0], new[0], turn)
    _flap(c, x + FLAP_W + 2, LABEL_Y, old[1], new[1], turn)
    if previous_name and previous_name != name and phase < 1:
        lay(previous_name, 1 - smooth(phase / 0.45))
        lay(name, smooth((phase - 0.55) / 0.45))
    else:
        lay(name, 1.0)
    if a < 1:
        c.pop_group_to_source()
        c.paint_with_alpha(a)


# ------------------------------------------------------------------ the wall: every clip in its own window, opening the reel and closing it
CELL_W, CELL_H, FOOT_H = W // 4, H // 4, 84
TILE_FLAGS = "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp"
WALL_OVERLAP, WALL_PUSH = 12, 30                              # the push into the first window starts over the wall's last 12 frames


def _wall_cell(i, n):
    """Window i of n: four to a row, the rows centred, with a cell left over for the title whenever the grid has room."""
    rows = min(4, n // 4 + 1)
    return (i % 4) * CELL_W, (i // 4) * CELL_H + (H - rows * CELL_H) // 2


@lru_cache(maxsize=2)
def _tile_foot(peak):
    """A soft falloff from a window's lower left corner for its caption to sit on: no bar, no edge."""
    x = np.arange(CELL_W, dtype=np.float32)[None, :]
    y = np.arange(FOOT_H, dtype=np.float32)[:, None] / (FOOT_H - 1)
    along = np.clip((430 - x) / 300, 0, 1)
    return surface(np.zeros((FOOT_H, CELL_W, 3), np.float32), (y * y * (3 - 2 * y)) * (along * along * (3 - 2 * along)) * peak)


@lru_cache(maxsize=32)
def _tile_label(number, name, dark=False):
    """A window's number and style, in the DIN and Songti of the column on the right edge: cream on footage, ink on a page.
    Pre-rendered: the blur is too dear per frame."""
    s, c = canvas(None, CELL_W, FOOT_H)
    c.set_antialias(cairo.ANTIALIAS_BEST)
    _fine(c)
    base = FOOT_H - 17
    width = _tracked(c, f"{number:02d}", 15, base, 24, "#3E3A32" if dark else "#D9B87C", DIN, True, 0.06)
    _tracked(c, name, 15 + width + 11, base, 21, "#1A1815" if dark else CREAM, SERIF, False, 0.07)
    return _halo(s, 2.0, 0.5 if dark else 0.7, dark)


@lru_cache(maxsize=32)
def _tile_cloud(number, name):
    """Where a window goes out of focus for its caption: a soft cloud round the type, in the caption's own shape."""
    alpha = view(_tile_label(number, name))[..., 3].astype(np.float32) / 255
    return np.clip(gaussian_filter(alpha, 12) * 8, 0, 1)[..., None]


@lru_cache(maxsize=32)
def _tile_cloud_mask(number, name):
    return surface(np.zeros((FOOT_H, CELL_W, 3), np.float32), _tile_cloud(number, name)[..., 0])


class _CaptionTone:
    """The ground right under a window's caption, read afresh every frame: which type it takes and how much help that type needs.
    The choice has a dead band and a minimum hold, so a ground wandering through the middle can't make the type flicker;
    while it stays in the middle the foot does the work instead."""
    TO_INK, TO_CREAM, HOLD, FADE = 0.72, 0.26, 30, 4
    CREAM_OVER, INK_OVER = 0.42, 0.58                            # the brightest ground cream reads on, the darkest ink reads on
    LUMA = np.array((0.114, 0.587, 0.299), np.float32)            # stored BGR
    dark = None

    def follow(self, paper, deep, middle, light):
        now = np.array((deep, middle, light))
        if self.dark is None:                                    # right from a window's first frame, with no fade to get there
            self.dark, self.since, self.levels, self.paper = middle > 0.65, 0, now, paper
            self.ink = float(self.dark)
        else:
            self.levels = self.levels + (now - self.levels) * (0.6 if abs(middle - self.levels[1]) > 0.2 else 0.3)   # a page landing can't wait
            self.paper = self.paper + (paper - self.paper) * 0.18
            self.since += 1
            if self.since >= self.HOLD and (self.levels[1] < self.TO_CREAM if self.dark else self.levels[1] > self.TO_INK):
                self.dark, self.since = not self.dark, 0
            self.ink += max(-1 / self.FADE, min(1 / self.FADE, float(self.dark) - self.ink))

    def help(self):
        """(foot weight, paper to wash with, extra shade for cream, extra wash for ink): enough to clear the type's own limit."""
        deep, middle, light = self.levels
        tint = float(self.paper @ self.LUMA[::-1])
        paper = self.paper
        if tint < 0.8:                                            # paper that dull can't lift ink: bring it towards cream, keeping its hue
            paper = paper + (np.array(hx(CREAM), np.float32) - paper) * (0.8 - tint) / (0.94 - tint)
            tint = 0.8
        shade = 1 - self.CREAM_OVER / (max(light, 0.01) * 0.7)      # the foot itself takes off at least 30% where the type sits
        washed = deep + (tint - deep) * 0.4
        wash = (self.INK_OVER - washed) / max(tint - washed, 0.01)
        return smooth((middle - 0.35) / 0.2), paper, min(0.85, max(0.0, shade)), min(0.85, max(0.0, wash))


def _tile_ground(tile, number, name):
    """What a caption would stand on in this picture: the paper there, and the out-of-focus ground at three levels."""
    foot = view(tile)[CELL_H - FOOT_H::2, ::2, :3].astype(np.float32)
    cloud = _tile_cloud(number, name)[::2, ::2]
    under = cloud[..., 0] > 0.5
    soft = foot + (gaussian_filter(foot, (2.25, 2.25, 0)) - foot) * cloud
    return (np.percentile(foot[under], 85, axis=0)[::-1] / 255, *(np.percentile(soft[under] @ _CaptionTone.LUMA, (20, 50, 80)) / 255))


def _tile_frost(tile, number, name, a=1.0):
    """Take a window's foot out of focus under its caption, so the clip's own small print can't read through the type."""
    foot = view(tile)[CELL_H - FOOT_H:]
    picture = foot[..., :3].astype(np.float32)
    foot[..., :3] = np.round(picture + (gaussian_filter(picture, (4.5, 4.5, 0)) - picture) * _tile_cloud(number, name) * a).astype(np.uint8)
    tile.mark_dirty()


class _Window:
    """A clip playing in the wall, read a few frames ahead so its caption is dressed before the ground changes under it."""
    LEAD = 5

    def __init__(self, reader, frames):
        self.reader, self.left, self.queue = reader, frames, []

    def read(self):
        """This frame, and the one its caption should already be judged by."""
        while self.left and len(self.queue) <= self.LEAD:
            self.queue.append(self.reader.read())
            self.left -= 1
        ahead = self.queue[-1]
        return self.queue.pop(0), ahead


def _wall_windows(stack, clips, frames=FRAMES, start=0.0):
    return [_Window(stack.enter_context(_FilmReader(path, CELL_W, CELL_H, frames=frames, start=start, flags=TILE_FLAGS)), frames)
            for _, _, path in clips]


def _tile_caption(c, x, y, number, name, tone, a=1.0):
    if a <= 0.004:
        return
    ink, top = smooth(tone.ink), y + CELL_H - FOOT_H
    weight, paper, shade, wash = tone.help()
    if ink < 1:
        c.set_source_surface(_tile_foot(0.78), x, top)
        c.paint_with_alpha((1 - ink) * a * (0.795 + 0.205 * weight))
        if shade > 0:
            c.set_source_rgba(0, 0, 0, (1 - ink) * a * shade)
            c.mask_surface(_tile_cloud_mask(number, name), x, top)
    if ink > 0:                                                  # the page's own paper, so its print falls away under the caption
        c.set_source_rgba(*paper, ink * a)
        c.mask_surface(_tile_foot(0.9), x, top)
        if wash > 0:
            c.set_source_rgba(*paper, ink * a * wash)
            c.mask_surface(_tile_cloud_mask(number, name), x, top)
    for dark, k in ((False, 1 - ink), (True, ink)):
        if k * a > 0.004:
            c.set_source_surface(_tile_label(number, name, dark), x, top)
            c.paint_with_alpha(k * a)


def _wall_title(c, n, a=1.0):
    """The title, in the cells after the last window."""
    if a <= 0.004 or n >= min(4, n // 4 + 1) * 4:
        return
    x, y = _wall_cell(n, n)
    x += 40
    width = _tracked(c, "风格库", x, y + 150, 66, CREAM, SERIF, True, 0.16, a=a)
    c.set_source_rgba(*hx(GOLD), 0.75 * a)
    c.rectangle(x + width + 34, y + 126.5, max(0, min(x + 876, W - 44) - x - width - 34), 1)
    c.fill()
    _tracked(c, f"{_numeral(n)}种画法", x + 2, y + 196, 24, CREAM, SERIF, False, 0.16, a=a)
    _tracked(c, "STYLE STUDIES", x + 2, y + 228, 14, MUTED, FUTURA, False, 0.34, a=a)


def _wall_frame(clips, windows, tones, appear=None, caption=None, title=1.0):
    """One frame of the wall, the same at the reel's opening and its close."""
    n = len(clips)
    surf, c = canvas(INK)
    c.set_antialias(cairo.ANTIALIAS_BEST)
    _fine(c)
    for i, (window, (_, name, _), tone) in enumerate(zip(windows, clips, tones)):
        x, y = _wall_cell(i, n)
        tile, ahead = window.read()
        tone.follow(*_tile_ground(ahead, i + 1, name))
        shown = 1.0 if caption is None else caption(i)
        if shown > 0.004:
            _tile_frost(tile, i + 1, name, shown)
        c.save()
        c.rectangle(x, y, CELL_W, CELL_H)
        c.clip()
        c.set_source_surface(tile, x, y)
        c.paint_with_alpha(1.0 if appear is None else appear(i))
        c.restore()
        _tile_caption(c, x, y, i + 1, name, tone, shown)
    c.set_source_rgb(*hx(INK))                                   # hairline gutters
    top = _wall_cell(0, n)[1]
    for gx in range(CELL_W, W, CELL_W):
        c.rectangle(gx - 1, 0, 2, H)
    for gy in range(top + CELL_H, H, CELL_H):
        c.rectangle(0, gy - 1, W, 2)
    c.fill()
    _wall_title(c, n, title)
    return surf


def _reel_opening(clips, first_clip, emit, label):
    """Every clip at once, start to finish, on the wall from the very first frame; then the camera pushes into the first window.
    Its clip plays out while the camera gathers way, then dissolves to its own frame 0 at the fastest part of the move,
    and the push lands on frame 0, so the clip starts over with no cut to see. Returns with frame 0 shown."""
    from contextlib import ExitStack
    n = len(clips)
    tones = [_CaptionTone() for _ in clips]
    first = first_clip.read()
    x0, y0 = _wall_cell(0, n)
    ax, ay = x0 * 4 / 3, y0 * 4 / 3                              # the one point the push leaves still; at 4x the window fills the frame
    with ExitStack() as stack:
        windows = _wall_windows(stack, clips)
        for _ in range(FRAMES - WALL_OVERLAP):
            emit(_wall_frame(clips, windows, tones))
        ending = stack.enter_context(_FilmReader(clips[0][2], frames=WALL_OVERLAP, start=(FRAMES - WALL_OVERLAP) / FPS))
        for j in range(WALL_PUSH):
            if j < WALL_OVERLAP:
                wall = _wall_frame(clips, windows, tones)
                late = ending.read()
            p = (j + 1) / WALL_PUSH
            k = 4 ** smooth(p)
            tx, ty = ax * (1 - k), ay * (1 - k)
            surf, c = canvas(INK)
            c.save()
            c.translate(tx, ty)
            c.scale(k, k)
            c.set_source_surface(wall)
            c.get_source().set_filter(cairo.FILTER_BILINEAR)
            c.paint()
            c.restore()
            c.set_source_rgba(*hx(INK), 0.55 * smooth(p / 0.45))     # the others fall back as they leave
            c.paint()
            sx, sy = tx + x0 * k, ty + y0 * k                        # the first window, from here on the full-size picture
            seam = 1 - smooth(p)
            c.save()
            c.rectangle(sx, sy, (CELL_W - seam) * k, (CELL_H - seam) * k)
            c.clip()
            c.save()
            c.translate(sx, sy)
            c.scale(k / 4, k / 4)
            for picture, a in ((late, 1.0), (first, smooth((p - 0.42) / 0.24))):
                if a > 0:
                    c.set_source_surface(picture)
                    c.get_source().set_filter(cairo.FILTER_GOOD)
                    c.paint_with_alpha(a)
            c.restore()
            handover = 1 - smooth(p / 0.3)                           # the wall's own window hands over to it, and takes its caption along
            if handover > 0:
                c.translate(tx, ty)
                c.scale(k, k)
                c.set_source_surface(wall)
                c.get_source().set_filter(cairo.FILTER_BILINEAR)
                c.paint_with_alpha(handover)
            c.restore()
            a = smooth((j + 1 - (WALL_PUSH - 8)) / 8)
            if a > 0:
                label(surf, 1, n, clips[0][1], a=a)
            emit(surf)


def _reel_outro(clips, emit, last):
    """The last clip shrinks into its place on a wall of all of them, every one still playing."""
    from contextlib import ExitStack
    n = len(clips)
    home = _wall_cell(n - 1, n)
    tones = [_CaptionTone() for _ in clips]
    with ExitStack() as stack:
        # Finishing by 4.4 s leaves the seek room at the end of every clip.
        windows = _wall_windows(stack, clips, frames=90, start=1.4)
        for frame in range(90):
            surf = _wall_frame(clips, windows, tones,
                               appear=lambda i: smooth((frame - 2 - (n - 1 - i) * 0.9) / 9),
                               caption=lambda i: smooth((frame - 14 - (n - 1 - i) * 0.5) / 8),
                               title=smooth((frame - 26) / 14))
            c = cairo.Context(surf)
            if frame < 18:                                     # the last frame of the last clip, on its way to its tile
                e = smooth(frame / 15)
                k = 1 - e * 0.75
                c.save()
                c.translate(home[0] * e, home[1] * e)
                c.scale(k, k)
                c.set_source_surface(last)
                c.paint_with_alpha(1 - smooth((frame - 14) / 3.5))
                c.restore()
            if frame >= 78:                                    # and out
                c.set_source_rgba(*hx(INK), smooth((frame - 78) / 11))
                c.paint()
            emit(surf)


def cmd_reel(allow_missing=False):
    from contextlib import ExitStack
    import tempfile
    from styles_transitions import TRANSITIONS, TRANSITION_FRAMES, crossfade

    clips = _film_clips(allow_missing)
    transitions = {(a, b): fn for a, b, fn in TRANSITIONS}
    overlap = TRANSITION_FRAMES
    fd, temporary = tempfile.mkstemp(prefix=".reel-", suffix=".mp4", dir=OUT)
    os.close(fd)
    encoder = None
    written = 0
    try:
        encoder = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}",
            "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", temporary], stdin=subprocess.PIPE)

        def emit(surf):
            nonlocal written
            encoder.stdin.write(frame_bytes(surf))
            written += 1

        tone = _Tone()

        def label(surf, *args, **kw):
            tone.follow(*_ground(view(surf)[120:640:4, W - 96:W - 12:4, :3]))
            _reel_label(surf, *args, ink=tone.ink, paper=tone.paper, **kw)

        last = None
        with ExitStack() as stack:
            current = stack.enter_context(_FilmReader(clips[0][2]))
            _reel_opening(clips, current, emit, label)
            for i, (slug, name, _) in enumerate(clips):
                has_next = i + 1 < len(clips)
                # Incoming frames 0..11 already played during the last overlap.
                start = overlap if i else 1                     # the opening landed on the first clip's frame 0
                stop = FRAMES - overlap if has_next else FRAMES
                for frame in range(start, stop):
                    surf = current.read()
                    if frame == stop - 1 and not has_next:
                        last = surface(rgb_of(surf))        # clean of the label, for the outro
                    label(surf, i + 1, len(clips), name)
                    emit(surf)
                if has_next:
                    next_slug, next_name, path = clips[i + 1]
                    incoming = stack.enter_context(_FilmReader(path))
                    transition = transitions.get((slug, next_slug), crossfade)
                    for frame in range(overlap):
                        phase = frame / (overlap - 1)
                        a = rgb_of(current.read())
                        b = rgb_of(incoming.read())
                        surf = surface(transition(a, b, phase))
                        label(surf, i + 2, len(clips), next_name, previous=i + 1, phase=phase, previous_name=name)
                        emit(surf)
                    del a, b
                    current.close()
                    current = incoming
                print(f"reel {i + 1:02d}/{len(clips):02d} {slug}", flush=True)
        _reel_outro(clips, emit, last)
        expected = FRAMES - WALL_OVERLAP + WALL_PUSH - 1 + len(clips) * FRAMES - (len(clips) - 1) * overlap + 90
        if written != expected:
            raise RuntimeError(f"Reel frame count: {written}, expected {expected}")
        encoder.stdin.close()
        rc = encoder.wait()
        if rc:
            raise RuntimeError(f"ffmpeg reel encoder exited {rc}")
        dest = os.path.join(OUT, "reel.mp4")
        os.replace(temporary, dest)
        print(f"{dest}  ({written} frames, {written / FPS:.2f}s)", flush=True)
    finally:
        if encoder is not None:
            if encoder.poll() is None:
                encoder.terminate()
            try:
                if not encoder.stdin.closed:
                    encoder.stdin.close()
            except BrokenPipeError:
                pass
            encoder.wait()
        if os.path.exists(temporary):
            os.remove(temporary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("clip")
    p.add_argument("slugs", nargs="*")
    p.add_argument("--jobs", type=int, default=min(18, os.cpu_count() or 4))
    p = sub.add_parser("snap")
    p.add_argument("slug")
    p.add_argument("times", nargs="+", type=float)
    p = sub.add_parser("stills")
    p.add_argument("--allow-missing", action="store_true")
    p = sub.add_parser("reel")
    p.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    if a.cmd == "clip":
        cmd_clip(a.slugs or [s for s, _ in STYLES], a.jobs)
    elif a.cmd == "snap":
        cmd_snap(a.slug, a.times)
    else:
        try:
            if a.cmd == "stills":
                cmd_stills(a.allow_missing)
            elif a.cmd == "reel":
                cmd_reel(a.allow_missing)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            sys.exit(str(exc))
