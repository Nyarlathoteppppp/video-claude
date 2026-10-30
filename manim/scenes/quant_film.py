"""一口气教你学会量化交易: the film renderer for episode 4 (landscape 1920x1080, 30 fps). Storyboard: quant.scenes.md.

Subtitles, their times and the small notes are read from the storyboard's table, so the storyboard stays the one source
of truth; `check` fails if a subtitle cannot be read in its slot or a row misses the beat grid. The look, the beat grid
and every drawing helper live in quant_kit.py. This file holds the hero shots (the hook, the title, the chapter pages,
Simons, money into compute, the finale and the end), the lineup, subtitles, frames, sound and the commands. The other
shots live in quant_shots_a.py (#4-#17), quant_shots_b.py (#20-#33) and quant_shots_c.py (#35-#46, #52); each exports
SHOTS = [(first row, last row, draw)] with draw(ctx, t) painting rows first..last.

    PY=python3        # a Python with pycairo, numpy, scipy, Pillow (and ffmpeg on PATH)
    $PY quant_film.py check                                   # subtitles vs reading time, beat grid, inputs
    $PY quant_film.py snap 3 14 30 50                         # PNGs + contact sheet -> quant/film_out/snaps/
    $PY quant_film.py audio                                   # music + effects -> quant/film_out/film_audio.wav
    $PY quant_film.py render 0 267.49 --jobs 18 --name rough_v5
    $PY quant_film.py short --jobs 18                         # the 9:16 short «选择你的量化公式»
    $PY quant_film.py snap --short 233 240                    # its frames (film times)

Inputs not in the repo: quant/bgm/aizo.wav and .beats.csv (from #56 to the end; manim/sfx/analyze_bgm.py makes the
.beats.csv), quant/memes/*.jpg, manim/sfx/*.wav (make_sfx.py). quant/formulas/*.png are in it (make_formulas.py).
"""
import argparse
import importlib
import json
import math
import os
import re
import subprocess
import sys

from quant_kit import *  # noqa: F401,F403  (the kit is this film's vocabulary)
from scipy.io import wavfile
from scipy.signal import resample_poly

SHOT_MODULES = []
for _m in ("quant_shots_a", "quant_shots_b", "quant_shots_c"):
    try:
        SHOT_MODULES.append(importlib.import_module(_m))
    except ModuleNotFoundError as e:                          # a module not written yet: its rows stay empty
        if e.name != _m:
            raise


# ======================================================================================
# #1-#2 the hook (effects only) and #3 the title
# ======================================================================================
def hud_rings(ctx, cx, cy, r, t, a, col="amber"):
    """Thin rotating arc segments around a portrait: the panel's instrument look."""
    for i, (rr, n, sp) in enumerate(((r + 22, 3, 0.25), (r + 40, 5, -0.15), (r + 58, 2, 0.4))):
        for j in range(n):
            u0 = t * sp + j * 2 * math.pi / n + i
            ctx.arc(cx, cy, rr, u0, u0 + 0.9 / n * math.pi)
            stroke(ctx, col, 2.0 if i else 3.0, (0.55 - 0.15 * i) * a)
            ctx.new_path()


def scene_hook(ctx, t):
    """#1: the investor-panel cover, complete from frame 0. #2: Liang's path lights node by node into the question."""
    k2 = sm((t - A(2)) / 0.7)
    out = sm((t - (A(3) - 0.45)) / 0.45)                      # everything rushes into the title
    ctx.save()
    ctx.translate(W / 2, H / 2); ctx.scale(1 + 0.6 * out, 1 + 0.6 * out); ctx.translate(-W / 2, -H / 2)
    a_all = 1 - out
    # the portrait side
    px, py, pr = mix(560, 300, k2), mix(560, 400, k2), mix(220, 150, k2)
    hud_rings(ctx, px, py, pr, t, a_all)
    portrait(ctx, "liang_shush.jpg", px, py, pr, a_all, "amber", focus=(0.5, 0.36), zoom=1.15)
    text(ctx, "幻方量化 · 创始人", px, py - pr - 132, 30, "gold", a_all * (1 - k2), bold=True, align="center")
    glow_text(ctx, "梁文锋", px, py - pr - 50, 64, "ink", a_all, align="center", glow=0.5)
    # the number side (#1)
    a1 = a_all * (1 - k2)
    if a1 > 0.002:
        cx = 1330
        text(ctx, "幻方量化 2025 年产品平均收益", cx, 318, 38, "ink", a1, align="center")
        pulse = 0.7 + 0.6 * kick(t, 0.0, 0.9) + 0.15 * math.sin(t * 3)
        num(ctx, "+56%", cx, 640, 300, "red", a1, align="center", glow=pulse)
        sx = mix(900, 1760, ((t - 0.2) / 1.6) % 1.0) if t < 3.4 else -999
        glow_at(ctx, sx, 540, "white", 1.6, 0.25 * a1)
        x0, x1, y = 1000, 1660, 740
        ctx.rectangle(x0, y, x1 - x0, 14); fill(ctx, "grid", a1)
        ctx.rectangle(x0, y, (x1 - x0) * 0.5655, 14); fill(ctx, "red", a1)
        for i in range(11):
            ctx.move_to(x0 + (x1 - x0) * i / 10, y + 20); ctx.line_to(x0 + (x1 - x0) * i / 10, y + 28)
            stroke(ctx, "dim", 1.5, a1)
    # the path (#2)
    if k2 > 0.002:
        y = 400
        x0, x1 = 560, 1760
        nodes = [(0.05, "2008", "开始做量化"), (0.45, "2015", "成立幻方量化"), (0.88, "2023", "孵化 DeepSeek")]
        reach = ease_out((t - A(2) - 0.3) / 2.2)
        gline(ctx, [(x0, y), (x0 + (x1 - x0) * reach, y)], "cyan", 2.2, a_all * k2)
        for i, (u, yr, what) in enumerate(nodes):
            t0 = A(2) + 0.4 + 0.75 * i
            k = on(t, t0, 0.25)
            if k <= 0:
                continue
            x = x0 + (x1 - x0) * u
            dot(ctx, x, y, 9, "amber" if i == 2 else "cyan", a_all * k, 1 + 2 * kick(t, t0, 0.5))
            num(ctx, yr, x, y - 34, 44, "ink", a_all * k, align="center", glow=0.4)
            text(ctx, what, x, y + 58, 32, "amber" if i == 2 else "ink", a_all * k, bold=True, align="center")
        tq = A(2) + 2.9
        kq = on(t, tq, 0.3)
        if kq > 0:
            glow_text(ctx, "?", 1160, 820, 300, "amber", a_all * kq, align="center", glow=1.2,
                      scale=pop((t - tq) / 0.35))
    ctx.restore()


TITLE_FORMULAS = [("bachelier", 340, 230), ("kelly", 1580, 230), ("sharpe", 300, 800), ("black_scholes", 1600, 800),
                  ("fama_french", 960, 150)]


def scene_title(ctx, t):
    """#3: the title lands on the first downbeat; a formula lights up on each beat after it."""
    t0 = A(3)
    surface(ctx, lambda x, y: 0.26 * math.sin(2.4 * x + 0.8 * (t - t0)) * math.cos(1.7 * y), 960, 640, 560,
            0.6 + 0.04 * (t - t0), 1.05, a=0.5, prog=min(1.0, (t - t0) / 1.4 + 0.2), n=18)
    for i, (name, x, y) in enumerate(TITLE_FORMULAS):
        tb = at(3, i + 1)
        k = on(t, tb, 0.2)
        if k > 0:
            formula(ctx, name, x, y, 470 if name != "fama_french" else 760, 0.55 * k,
                    lit={1: 0.4 + 0.6 * kick(t, tb, 0.9)}, scale=pop((t - tb) / 0.3))
    s = 1 + 0.3 * (1 - ease_out((t - t0) / 0.28))
    glow_text(ctx, "一口气教你学会", 960, 515, 96, "ink", 1.0, align="center", glow=0.6, scale=s)
    glow_text(ctx, "量化交易", 960, 690, 156, "amber", 1.0, align="center", glow=0.9 + 0.8 * kick(t, t0, 0.6), scale=s)


# ======================================================================================
# chapter pages (#11 #19 #25 #34) and the step track on top
# ======================================================================================
STEPS = ["找规律", "回测", "下注", "下单"]
CHAPTERS = {11: 0, 19: 1, 25: 2, 34: 3}
ROAD = (330, 1590, 440)                       # the chapter pages' four-step road: x0, x1, y


def step_track(ctx, cur, x0, x1, y, a=1.0, size=30, t=0.0, lit_at=None):
    """The four steps as nodes on a line: done ones cyan, the current one amber (pulsing from lit_at), later ones dim."""
    if a <= 0.002:
        return
    xs = [x0 + (x1 - x0) * i / 3 for i in range(4)]
    gline(ctx, [(xs[0], y), (xs[-1], y)], "dim", 1.5, a, glow=0)
    if cur > 0:
        gline(ctx, [(xs[0], y), (xs[cur], y)], "cyan", 2.0, a, 0.8)
    for i, x in enumerate(xs):
        col = "cyan" if i < cur else ("amber" if i == cur else "dim")
        g = 1 + (2 * kick(t, lit_at, 0.6) if (i == cur and lit_at is not None) else 0)
        dot(ctx, x, y, size * 0.28, col, a, g if i <= cur else 0)
        text(ctx, STEPS[i], x, y + size * 1.45, size, col if i <= cur else "dim", a, bold=i == cur, align="center")


def scene_chapter(ctx, t, n):
    """The same four-step road on every chapter page, so the film's spine stays in view. #11: the road is laid out one
    step per beat, step one lights on b4. #19 #25 #34: the steps before are done (cyan); a light runs along the road
    from the last step and lights this one on the b2 downbeat."""
    i = CHAPTERS[n]
    x0, x1, y = ROAD
    xs = [x0 + (x1 - x0) * j / 3 for j in range(4)]
    lit = at(n, 4) if i == 0 else at(n, 2)
    show = [on(t, at(n, j), 0.2) for j in range(4)] if i == 0 else [on(t, A(n), 0.25)] * 4
    reach = ease_out((t - A(n)) / (3 * BEAT)) if i == 0 else 1.0
    gline(ctx, [(xs[0], y), (xs[0] + (xs[3] - xs[0]) * reach, y)], "dim", 2, show[0], glow=0)
    if i > 0:
        gline(ctx, [(xs[0], y), (xs[i - 1], y)], "cyan", 3, show[0], 0.8)
        u = smooth((t - A(n) - 0.1) / (lit - A(n) - 0.1))
        if u > 0:
            hx_ = xs[i - 1] + (xs[i] - xs[i - 1]) * u
            gline(ctx, [(xs[i - 1], y), (hx_, y)], "amber", 4, 1.0, 1.2)
            glow_at(ctx, hx_, y, "amber", 1.4, 0.7 * (1 - kick(t, lit, 0.3)) if t < lit + 0.3 else 0.0)
    for j, x in enumerate(xs):
        if show[j] <= 0.002:
            continue
        now = j == i and t >= lit
        col = "cyan" if j < i else ("amber" if now else "dim")
        if now:
            glow_at(ctx, x, y, "amber", 3.2, 0.55 * kick(t, lit, 0.7) + 0.12)
        dot(ctx, x, y, 22 if now else 14, col, show[j], 2.5 if now else (0.8 if j < i else 0))
        if now:
            glow_text(ctx, STEPS[j], x, y + 120, 84, "amber", 1.0, align="center", glow=0.8 + kick(t, lit, 0.6),
                      scale=pop((t - lit) / 0.3))
            text(ctx, f"第{'一二三四'[j]}步", x, y - 60, 34, "ink2", on(t, lit, 0.25), align="center")
        else:
            text(ctx, STEPS[j], x, y + 100, 56, col if j < i else ("ink2" if j == i else "dim"), show[j],
                 bold=j < i, align="center")


def draw_top_steps(ctx, t):
    """The small step track at the top while a step is being explained (not over the chapter pages)."""
    spans = [(B(11), A(19), 0), (B(19), A(25), 1), (B(25), A(34), 2), (B(34), A(41), 3)]
    for a, b, i in spans:
        k = win(t, a, b, 0.3, 0.3)
        if k > 0.002:
            step_track(ctx, i, 700, 1220, 36, 0.9 * k, 22)


# ======================================================================================
# #18 Simons
# ======================================================================================
def medallion_card(ctx, x, y, a=1.0, k66=1.0, k39=1.0, t=0.0, t66=None, scale=1.0):
    """大奖章基金: the fund's record, gross and net side by side (the net one at 80 % of the gross one's size)."""
    if a <= 0.002:
        return
    ctx.save(); ctx.translate(x, y); ctx.scale(scale, scale)
    w, h = 800, 300
    panel(ctx, 0, 0, w, h, a, edge="gold", glow=0.4 * a * (kick(t, t66, 0.6) if t66 else 0))
    text(ctx, "大奖章基金", 34, 62, 40, "gold", a, bold=True)
    text(ctx, "1988–2018 · 年化收益", 34, 108, 26, "ink2", a)
    text(ctx, "费前", 34, 250, 28, "ink2", a * k66)
    num(ctx, "≈66%", 104, 262, 120, "gold", a * k66, glow=0.8 + kick(t, t66, 0.7) if t66 else 0.8,
        scale=pop(k66) if k66 < 1 else 1.0)
    text(ctx, "费后", 440, 250, 28, "ink2", a * k39)
    num(ctx, "≈39%", 510, 255, 96, "gold", a * k39, glow=0.7)
    ctx.restore()


def scene_simons(ctx, t):
    """#18 (starts on a half bar, downbeats on beats 2 and 6): the legend's title block on b0, his torus on the b2
    downbeat, the Chern–Simons form lighting on b3-b4, the Medallion card on b4 and its record slamming in on the b6
    downbeat (gross 66 % first, net 39 % half a beat later)."""
    n = 18
    t0 = A(n)
    u = t - t0
    # right: the mathematician's torus and his formula
    ka = on(t, at(n, 2), 0.4)
    torus(ctx, 1390, 380, 230, 92, 0.3 + 0.12 * u, 1.12, ka, prog=min(1.0, (t - at(n, 2)) / 1.2) if ka > 0 else 0)
    kf = on(t, at(n, 3), 0.3)
    formula(ctx, "chern_simons", 1390, 700, 780, kf, lit={1: on(t, at(n, 3.5), 0.2), 2: on(t, at(n, 4), 0.2)},
            col="ink", lit_col="amber")
    # left: who he is
    x = 150
    k0 = on(t, t0, 0.2)
    glow_text(ctx, "吉姆·西蒙斯", x, 285, 76, "amber", k0, glow=0.8 + kick(t, t0, 0.6),
              scale=1 + 0.15 * (1 - ease_out(u / 0.22)))
    text(ctx, "文艺复兴科技 创始人", x, 370, 36, "ink", on(t, at(n, 1.5), 0.3), bold=True)
    # the record
    kc = on(t, at(n, 4), 0.3)
    if kc > 0:
        medallion_card(ctx, x, 470 + 40 * (1 - kc), kc, on(t, at(n, 6), 0.25), on(t, at(n, 6.5), 0.25), t, at(n, 6))


# ======================================================================================
# #47-#50 the firm's money into compute, DeepSeek, the meme
# ======================================================================================
LEDGER_L, LEDGER_R = (140, 300, 600, 360), (1180, 300, 600, 360)


def ledgers(ctx, t, a, fees=0.0):
    """Two separate books: the clients' fund (their gains and losses stay theirs) and the firm (fees come in)."""
    x, y, w, h = LEDGER_L
    panel(ctx, x, y, w, h, a, edge="ice")
    text(ctx, "客户的基金", x + 30, y + 110, 46, "ink", a, bold=True)
    ys = equity(470, 60, 0.15, 1.0)
    gline(ctx, curve_pts(ys, x + 30, y + 210, w - 60, 110), "ice", 2.2, a * 0.8, 0.6)
    x, y, w, h = LEDGER_R
    panel(ctx, x, y, w, h, a, edge="gold")
    text(ctx, "幻方（公司）", x + 30, y + 110, 46, "ink", a, bold=True)
    for i in range(3):                                        # fee slips crossing over, one per beat
        k = ease_out(fees - i)
        if k <= 0:
            continue
        sx, sy = LEDGER_L[0] + LEDGER_L[2] - 40, 480 + i * 40
        ex, ey = LEDGER_R[0] + 40 + i * 70, 590
        px, py = sx + (ex - sx) * k, sy + (ey - sy) * k - 120 * math.sin(math.pi * k)
        rrect(ctx, px - 32, py - 18, 64, 36, 6); fill(ctx, "gold", a * 0.9)
        text(ctx, "费", px, py + 10, 24, "bg", a, bold=True, align="center")


def scene_firm(ctx, t):
    """#47: the two books, fees crossing over on beats 1-3, then the firm's own money gathers in the centre."""
    n = 47
    b = bt(t, n)
    gather = smooth((b - 4) / 3.5)
    ledgers(ctx, t, on(t, A(n), 0.3) * (1 - 0.7 * gather), fees=max(0.0, (t - at(n, 1)) / 0.6) if b > 1 else 0.0)
    if b > 4:
        k = on(t, at(n, 4), 0.5)
        cx, cy = mix(1480, 960, ease_out((t - at(n, 4)) / 1.2)), mix(600, 470, ease_out((t - at(n, 4)) / 1.2))
        s = 1 + 0.25 * gather + 0.05 * math.sin(t * 20) * gather
        glow_at(ctx, cx, cy, "gold", 2.5 + 3 * gather, 0.5 * k)
        ctx.save(); ctx.translate(cx, cy); ctx.scale(s, s)
        rrect(ctx, -130, -70, 260, 140, 14); fill(ctx, "gold", k)
        text(ctx, "公司资金", 0, 14, 40, "bg", k, bold=True, align="center")
        ctx.restore()
        r = rng(47)
        for i in range(140):                                  # light sucked in from the dark: the riser
            ang = r.uniform(0, 2 * math.pi)
            ph = r.random()
            d = 900 * (1 - ((t * 0.6 + ph) % 1.0) ** 0.7)
            if d > 140:
                dot(ctx, cx + d * math.cos(ang), cy + d * 0.6 * math.sin(ang), 2.2, "gold", 0.8 * gather, 0.4)


GPU_N = 100
WALL = (600, 110, 720)                                        # x, y, size of the 100 x 100 wall when fully seen


@lru_cache(None)
def burst_particles(n=2200):
    """Each particle: the wall cell it lands on (x, y in 0-1), a sideways swirl, a launch delay."""
    r = rng(48)
    return r.random(n), r.random(n), r.uniform(-1, 1, n), r.uniform(0.0, 0.18, n)


def gpu_wall(ctx, t, x, y, size, lit_from, a=1.0, cx=0.2, cy=0.5):
    """The 100 x 100 GPU wall; cells light up in a wave that starts at time lit_from from (cx, cy) in wall units."""
    jj, ii = np.mgrid[0:GPU_N, 0:GPU_N] / (GPU_N - 1)
    dist = np.hypot(ii - cx, jj - cy)
    lit = np.clip((t - lit_from) * 0.55 - dist, 0, 1) ** 0.6 if t > lit_from else np.zeros((GPU_N, GPU_N))
    _, ph = cell_values(GPU_N, GPU_N, 48)
    shimmer = 0.75 + 0.25 * np.sin(t * 6 + ph * 20)
    base = hx("dim")[None, None] * 0.55
    hot = mix(hx("gold"), hx("ice"), np.clip((t - lit_from - 0.6) / 1.2, 0, 1))[None, None] * shimmer[..., None]
    rgb = mix(base, hot, lit[..., None])
    step = 8
    img = np.repeat(np.repeat(rgb, step, 0), step, 1)
    m = np.zeros((step, step)); m[:6, :6] = 1
    s = surf_rgb(img, np.tile(m, (GPU_N, GPU_N)))
    k = size / (GPU_N * step)
    ctx.save(); ctx.translate(x, y); ctx.scale(k, k)
    ctx.set_source_surface(s, 0, 0); ctx.get_source().set_filter(cairo.FILTER_GOOD); ctx.paint_with_alpha(a)
    ctx.restore()
    if t > lit_from:
        glow_at(ctx, x + size / 2, y + size / 2, "ice", size / 180, 0.18 * a * min(1.0, (t - lit_from) / 2))
    return float(lit.mean())


def scene_compute(ctx, t):
    """#48: the hit. The firm's block bursts into gold light that streams into the GPU wall, which lights up in a wave
    while the camera pulls back. #49: the wall feeds tokens into the word DeepSeek; a dated news box on the side."""
    t0 = A(48)
    u = t - t0
    wx, wy, ws = WALL
    cx, cy = wx + ws / 2, wy + ws / 2
    lit_from = t0 + 0.35
    if t < A(49):
        zoom = mix(3.0, 1.0, smooth((u - 0.4) / 2.8))           # start close on the centre, pull back to all 10,000
        ctx.save()
        ctx.translate(cx, cy); ctx.scale(zoom, zoom); ctx.translate(-cx, -cy)
        frac = gpu_wall(ctx, t, wx, wy, ws, lit_from, 1.0, 0.5, 0.5)
        ctx.restore()
        tx, ty, sw, de = burst_particles()                     # the block's light, each speck landing on its cell
        arrive = 0.35 + np.hypot(tx - 0.5, ty - 0.5) / 0.55     # ... just as the wave lights that cell
        prog = np.clip((u - de) / (arrive - de), 0, 1)
        ctx.save(); ctx.set_operator(cairo.OPERATOR_ADD)
        for i in np.nonzero((u > de) & (prog < 1))[0]:
            k = ease_out(float(prog[i]))
            ex, ey = cx + (wx + tx[i] * ws - cx) * zoom, cy + (wy + ty[i] * ws - cy) * zoom
            dx, dy = ex - cx, ey - cy
            sw_ = sw[i] * 0.35 * math.sin(math.pi * k)
            x = cx + dx * k - dy * sw_
            y = cy + dy * k + dx * sw_
            col = mix(hx("gold"), hx("ice"), k)
            ctx.arc(x, y, 2.6, 0, 2 * math.pi); ctx.set_source_rgba(*col, 0.85); ctx.fill()
        ctx.restore()
        kb = 1 - sm(u / 0.3)                                   # the block itself, bursting
        if kb > 0:
            glow_at(ctx, cx, cy, "gold", 5 * (1 + u * 5), 0.9 * kb)
            rrect(ctx, cx - 130 * (1 + u * 3), cy - 70 * (1 + u * 3), 260 * (1 + u * 3), 140 * (1 + u * 3), 14)
            fill(ctx, "gold", kb ** 2)
        kl = on(t, at(48, 3), 0.3)
        if kl > 0:
            panel(ctx, 60, 150, 500, 230, kl, edge="ice")
            text(ctx, "萤火二号", 90, 262, 48, "ink", kl, bold=True)
            num(ctx, f"{int(10000 * min(1.0, frac * 1.15)):,}", 90, 350, 74, "ice", kl, glow=0.8)
            text(ctx, "张 A100", 330, 346, 30, "ink2", kl)
        return
    # #49: tokens into DeepSeek
    t1 = A(49)
    v = t - t1
    sh = smooth(v / 0.8)
    x, y, s = mix(wx, 140, sh), mix(wy, 260, sh), mix(ws, 520, sh)
    gpu_wall(ctx, t, x, y, s, lit_from, 0.9)
    r = rng(49)
    toks = ["量", "化", "the", "AI", "模", "型", "▁deep", "token", "算", "力", "思", "考", "▁seek", "数", "据", "推", "理"]
    cx, cy = 1300, 540
    kw = on(t, at(49, 2), 0.35)
    for i in range(70):
        ph = r.random()
        k = ((v * 0.55 + ph) % 1.0)
        if v < 0.3:
            continue
        sx, sy = x + s, y + r.uniform(0.1, 0.9) * s
        ex, ey = cx + r.normal(0, 140), cy + r.normal(0, 40)
        px, py = sx + (ex - sx) * k, sy + (ey - sy) * k - 80 * math.sin(math.pi * k) * r.uniform(-1, 1)
        text(ctx, toks[i % len(toks)], px, py, 22, "ice", 0.9 * (1 - k ** 3), font=MONO)
    glow_text(ctx, "DeepSeek", cx, cy + 50, 150, "ink", kw, font=LATIN, align="center", glow=1.0 + kick(t, at(49, 2), 0.6),
              scale=pop((t - at(49, 2)) / 0.35))
    kn = on(t, at(49, 4), 0.3)
    if kn > 0:
        panel(ctx, 1020, 720, 600, 130, kn, edge="dim")
        text(ctx, "英伟达单日市值减少近 6000 亿美元", 1046, 795, 30, "ink", kn)
        ctx.rectangle(1560, 745, 30, 30 + 60 * kn); fill(ctx, "green", kn)


def scene_uncle(ctx, t):
    """#50: the shush meme slams onto the wall on the b0 downbeat; '百亿补贴' pops up on b2 and turns into '算力' on
    the b4 downbeat."""
    n = 50
    gpu_wall(ctx, t, 140, 260, 520, A(48) + 0.35, 0.5 * (1 - on(t, A(n), 0.3)) + 0.25)
    gpu_wall(ctx, t, 260, 60, 1400, A(48) + 0.35, 0.22 * on(t, A(n), 0.3))
    k = pop((t - A(n)) / 0.3)
    sticker(ctx, "liang_shush.jpg", 700, 500, 520, on(t, A(n), 0.1), rot=-0.05, scale=k)
    kb = on(t, at(n, 2), 0.2)
    if kb > 0:
        ctx.save(); ctx.translate(1330, 330); ctx.scale(pop((t - at(n, 2)) / 0.3), pop((t - at(n, 2)) / 0.3))
        ctx.rotate(0.06)
        rrect(ctx, -230, -70, 460, 140, 70); fill(ctx, "red", kb, keep=True); stroke(ctx, "gold", 5, kb)
        glow_text(ctx, "百亿补贴", 0, 26, 74, "gold", kb, align="center", glow=0.5)
        ctx.restore()
    ka = on(t, at(n, 4), 0.3)
    if ka > 0:
        arrow(ctx, 1330, 420, 1330, 420 + 150 * ka, "ice", 5, ka)
        glow_text(ctx, "算力", 1330, 720, 150, "ice", ka, align="center", glow=1.0 + kick(t, at(n, 4), 0.6),
                  scale=pop((t - at(n, 4)) / 0.35))


def scene_anime(ctx, t):
    """#53: the anime meme fills the frame (no music yet); at the very end it squeezes into a line, and #56 lands
    with AIZO."""
    n = 53
    k = pop((t - A(n)) / 0.3)
    sq = 1 - ease_in((t - (B(n) - 0.35)) / 0.35)
    ctx.save(); ctx.translate(W / 2, 470); ctx.scale(1, max(0.01, sq)); ctx.translate(-W / 2, -470)
    sticker(ctx, "laoshi_zhuanqian.jpg", W / 2, 470, 1180, on(t, A(n), 0.1), scale=k * (1 + 0.03 * (t - A(n))))
    ctx.restore()
    if sq < 1:
        gline(ctx, [(W / 2 - 900 * (1 - sq), 470), (W / 2 + 900 * (1 - sq), 470)], "ice", 4, 1 - sq, 1.5)


# ======================================================================================
# on AIZO: #56 the closing lines, #54 the battle part, #51 the recap, #55 the punchline, #57 the disclaimer
# ======================================================================================
FINALE = ["bachelier", "markowitz", "kelly", "sharpe", "black_scholes", "SIMONS", "grinold", "fama_french"]
FINALE_LIT = {"bachelier": [1, 2], "markowitz": [2, 1], "kelly": [1, 2, 3, 4], "sharpe": [1, 2], "black_scholes": [1, 2],
              "grinold": [2, 3, 1], "fama_french": [2, 3, 4, 1]}


def finale_hits():
    """One item per AIZO beat through #54; the last one (法玛–弗伦奇) hands over to #51's first step."""
    bs = [f for f, _ in beats() if A(54) - 0.05 <= f < B(54) - 0.05]
    return bs[:len(FINALE)]


def speed_lines(ctx, t, a, col="ice"):
    r = rng(int(t * 15))
    ctx.save(); ctx.set_operator(cairo.OPERATOR_ADD)
    for _ in range(70):
        ang = r.uniform(0, 2 * math.pi)
        r0 = r.uniform(420, 760)
        r1 = r0 + r.uniform(200, 700)
        ctx.move_to(W / 2 + r0 * math.cos(ang), 470 + r0 * math.sin(ang) * 0.7)
        ctx.line_to(W / 2 + r1 * math.cos(ang), 470 + r1 * math.sin(ang) * 0.7)
        stroke(ctx, col, r.uniform(1.5, 4), 0.16 * a)
    ctx.restore()


def finale_item(ctx, t, name, age, a, scale, dx):
    ctx.save(); ctx.translate(W / 2 + dx, 470); ctx.scale(scale, scale)
    if name in FORMULA_INFO:
        order = FINALE_LIT[name]
        lit = {term: on(age, 0.08 + 0.55 * BEAT * j / max(1, len(order) - 1), 0.08) for j, term in enumerate(order)}
        formula_card(ctx, name, 0, 0, 1000 if name != "fama_french" else 1300, a, lit)
    elif name == "SIMONS":
        glow_text(ctx, "吉姆·西蒙斯", -400, -190, 70, "amber", a, glow=0.8)
        label(ctx, "RENAISSANCE TECHNOLOGIES → MEDALLION FUND", -400, -130, 20, "ink2", a)
        medallion_card(ctx, -400, -90, a, 1.0, on(age, 0.25, 0.1), age + 10, None)
    ctx.restore()


def scene_finale(ctx, t):
    """#54: an item slams in on every beat from alternating sides (the #51 recap assembles the four steps)."""
    hits = finale_hits()
    speed_lines(ctx, t, 1.0)
    for i, name in enumerate(FINALE):
        if i >= len(hits) or t < hits[i]:
            break
        nxt = hits[i + 1] if i + 1 < len(hits) else B(54)
        if t >= nxt:
            continue
        age = t - hits[i]
        k = ease_out(age / 0.13)
        dx = (1 - k) * (900 if i % 2 else -900)
        finale_item(ctx, t, name, age, 1.0, 1.15 - 0.15 * k, dx)


def scene_learned(ctx, t):
    """#55, where AIZO turns quiet after the recap: the big line, then the 理论上 stamp two beats later."""
    k = on(t, A(55), 0.25)
    glow_text(ctx, "你已经完全学会了量化交易了", W / 2, 500, 86, "ink", k, align="center", glow=0.8,
              scale=pop((t - A(55)) / 0.3))
    stamp(ctx, "理论上", 1600, 360, 40, "red", 1.0, (t - A(55) - 2 * FINALE_BEAT) / 0.5, rot=0.12)


RECAP = [("找规律", "fama_french"), ("回测", "sharpe"), ("下注", "kelly"), ("下单", None)]


def recap_hits():
    """AIZO's beats under #51 (its beats 16-31): the steps slam in on b0 b2 b4 b6, the nodes on b8-b11, 就这四步 on b12."""
    return [f for f, _ in beats() if A(51) - 0.05 <= f < B(51) - 0.05]


def order_streaks(ctx, age, a=1.0):
    """下单: orders streak into the exchange gate (local coordinates, gate at x = 520)."""
    r = rng(34)
    gx = 520
    gline(ctx, [(gx, 420), (gx, 700)], "ice", 3, a, 1.2)
    for j in range(18):
        y = 430 + 260 * j / 17
        sp, ph = r.uniform(0.8, 1.6), r.random()
        x = -700 + ((age * sp * 1.6 + ph) % 1.0) * (gx + 700)
        col = "cyan" if j % 3 else "amber"
        gline(ctx, [(max(-700, x - 80 - 200 * sp), y), (x, y)], col, 2.2, 0.85 * a, 0.9, core=False)
        dot(ctx, x, y, 4, col, a, 0.8)


def scene_recap(ctx, t):
    """#51, on AIZO right after the battle part: the four steps slam back in from alternating sides, two beats each, each over its formula (下单
    over the order streaks); on b8-b11 the road assembles node by node and 就这四步 (#11's line, coming
    right after #55's 完全学会) lands on the b12 downbeat."""
    hits = recap_hits()
    if len(hits) < 13:
        return
    speed_lines(ctx, t, 1.0, "amber")
    for i, (name, f) in enumerate(RECAP):
        t0, t1 = hits[2 * i], hits[2 * i + 2]
        if not t0 <= t < t1:
            continue
        age = t - t0
        k = ease_out(age / 0.13)
        s = 1.15 - 0.15 * k
        ctx.save(); ctx.translate(W / 2 + (1 - k) * (900 if i % 2 else -900), 0)
        if f:
            ctx.save(); ctx.translate(0, 590); ctx.scale(s, s)
            formula_card(ctx, f, 0, 0, 1200 if f == "fama_french" else 1000, 1.0,
                         {term: on(age, 0.08 + 0.12 * j, 0.08) for j, term in enumerate(FINALE_LIT[f])})
            ctx.restore()
        else:
            order_streaks(ctx, age)
        text(ctx, f"第{'一二三四'[i]}步", 0, 170, 40, "ink2", 1.0, align="center")
        glow_text(ctx, name, 0, 300, 120, "amber", 1.0, align="center", glow=1.0 + kick(t, t0, 0.5), scale=s)
        ctx.restore()
    if t >= hits[8]:
        cur = min(3, sum(t >= h for h in hits[8:12]) - 1)
        step_track(ctx, cur, 360, 1560, 440, 1.0, 64, t, hits[8 + cur])
        glow_text(ctx, "就这四步", W / 2, 760, 110, "amber", on(t, hits[12], 0.2), align="center", glow=1.2,
                  scale=pop((t - hits[12]) / 0.3))


def scene_end(ctx, t):
    """#56: the closing lines, appearing with AIZO's downbeat; #57 (after #51 and #55): the disclaimer."""
    if t < A(57):
        k1 = on(t, A(56), 0.06)                          # lands on the frame AIZO comes in on
        k2 = on(t, A(56) + 4 * FINALE_BEAT, 0.25)
        glow_text(ctx, "来不及为古法交易哀悼了，", W / 2, 440, 76, "ink", k1, align="center", glow=0.5 + kick(t, A(56), 0.5),
                  scale=pop((t - A(56)) / 0.3))
        glow_text(ctx, "拿着量化去挑战华尔街吧", W / 2, 600, 100, "amber", k2, align="center", glow=1.0,
                  scale=pop((t - A(56) - 4 * FINALE_BEAT) / 0.35))
    else:
        k = on(t, A(57), 0.4)
        glow_text(ctx, "仅供科普，不构成投资建议。", W / 2, 500, 66, "ink", k, align="center", glow=0.3)
        text(ctx, "资料来源放简介", W / 2, 590, 28, "dim", k, align="center")


# ======================================================================================
# the lineup
# ======================================================================================
@lru_cache(None)
def scene_lineup():
    """(start, end, draw): every shot in order; neighbours cross-fade over XF seconds."""
    ch = lambda n: (A(n), B(n), lambda c, t: scene_chapter(c, t, n))  # noqa: E731
    out = [(0.0, A(3), scene_hook), (A(3), B(3), scene_title), ch(11), (A(18), B(18), scene_simons), ch(19), ch(25),
           ch(34), (A(47), A(48), scene_firm), (A(48), A(50), scene_compute), (A(50), B(50), scene_uncle),
           (A(53), B(53), scene_anime), (A(54), B(54), scene_finale), (A(55), B(55), scene_learned),
           (A(56), B(56), scene_end), (A(51), B(51), scene_recap), (A(57), END, scene_end)]
    for mod in SHOT_MODULES:
        out += [(A(a), B(b), fn) for a, b, fn in mod.SHOTS]
    return sorted(out, key=lambda s: s[0])


def covered_rows():
    spans = [(a, b) for a, b, _ in scene_lineup()]
    return {n for n in rows() if any(a - 1e-3 <= A(n) and B(n) <= b + 1e-3 for a, b in spans)}


XF = 0.16


# ======================================================================================
# subtitles, notes
# ======================================================================================
def draw_note(ctx, t):
    for r in rows().values():
        if not r["note"] or not (r["a"] <= t < r["b"]):
            continue
        k = sm((t - r["a"] - 0.3) / 0.4) * (1 - sm((t - r["b"] + 0.3) / 0.3))
        text(ctx, r["note"], NOTE_X, 912, NOTE_SIZE, "ink2", 0.85 * k, align="right", halo=0.9)


def sub_runs(s):
    """'用{夏普比率}打分' -> [(text, colour), ...]; {!..} is red, {..} amber."""
    out = []
    for m in re.finditer(r"\{(!?)([^}]*)\}|([^{]+)", s):
        col = "ink" if m[3] else ("red" if m[1] else "amber")
        out.append((m[3] or m[2], col))
    return out


@lru_cache(maxsize=256)
def subtitle_surface(s):
    """One line, centred, with a soft dark halo so it reads over any light."""
    tmp = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 4, 4))
    pieces, x = [], 0.0
    for chunk, col in sub_runs(s):
        for i, part in enumerate(chunk.split(" ")):
            if i:
                x += SUB_SIZE * 0.22
            for piece, f in runs(part, SANS):
                set_font(tmp, f, SUB_SIZE, True)
                pieces.append((x, piece, col, f)); x += tmp.text_extents(piece).x_advance
    pad = 40
    w, h = int(x + 2 * pad), int(SUB_SIZE * 1.6 + 2 * pad)
    m = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    c = cairo.Context(m)
    for px, part, _, f in pieces:
        set_font(c, f, SUB_SIZE, True); c.move_to(pad + px, pad + SUB_SIZE); c.show_text(part)
    halo = np.clip(gaussian_filter(surf_alpha(m), 6) * 2.4, 0, 0.9)
    out = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    c = cairo.Context(out)
    c.set_source_surface(surf_from(hx("bg"), halo, dither=False), 0, 0); c.paint()
    for px, part, col, f in pieces:
        set_font(c, f, SUB_SIZE, True); c.set_source_rgb(*hx(col)); c.move_to(pad + px, pad + SUB_SIZE); c.show_text(part)
    return out, w, pad + SUB_SIZE


def draw_subtitle(ctx, t):
    for r in rows().values():
        if not r["sub"]:
            continue
        a, b = shown(r)
        if not (a <= t < b + 0.2):
            continue
        k = sm((t - a) / 0.12) * (1 - sm((t - b) / 0.2))
        surf, w, base = subtitle_surface(r["sub"])
        ctx.set_source_surface(surf, W / 2 - w / 2, SUB_Y - base)
        ctx.paint_with_alpha(k)


# ======================================================================================
# frames
# ======================================================================================
def shake_at(t):
    s = 0.0
    for at_, px in [(A(3), 10), (at(6, 1), 8), (A(48), 14), (A(50), 6)] + [(h, 7) for h in finale_hits()] + \
            [(h, 9) for h in recap_hits()[:13:2]]:
        age = t - at_
        if 0 <= age < 0.35:
            s = max(s, px * (1 - age / 0.35))
    return s


def flash_at(t):
    f = 0.0
    for at_, k in [(A(3), 0.55), (A(48), 0.7), (A(55), 0.3)] + [(A(n), 0.18) for n in CHAPTERS] + \
            [(h, 0.25) for h in finale_hits()] + [(h, 0.3) for h in recap_hits()[:13:2]]:
        age = t - at_
        if 0 <= age < 0.4:
            f = max(f, k * math.exp(-age / 0.08))
    return f


def render_frame(t, clean=False):
    """clean: the shots alone, without the overlays (notes, subtitles, flashes, shake), for finding entrances."""
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(s)
    ctx.set_source_surface(background(), 0, 0); ctx.paint()
    for a, b, draw in scene_lineup():
        if not (a - XF <= t < b + XF):
            continue
        k = sm((t - a + XF) / (2 * XF)) * (1 - sm((t - b + XF) / (2 * XF)))
        if a == 0.0:
            k = 1 - sm((t - b + XF) / (2 * XF))
        if k <= 0.002:
            continue
        ctx.push_group()
        draw(ctx, t)
        ctx.pop_group_to_source(); ctx.paint_with_alpha(k)
    if clean:
        return s
    draw_top_steps(ctx, t)
    if at(47, 4) <= t < A(48):                                # the dip before the hit
        ctx.set_source_rgba(*hx("bg"), 0.35 * sm((t - at(47, 4)) / (A(48) - at(47, 4)))); ctx.paint()
    draw_note(ctx, t)
    draw_subtitle(ctx, t)
    if t > END - 1.2:
        ctx.set_source_rgba(0, 0, 0, sm((t - END + 1.2) / 1.2)); ctx.paint()
    f = flash_at(t)
    if f > 0.01:
        ctx.save(); ctx.set_operator(cairo.OPERATOR_ADD); ctx.set_source_rgba(0.85, 0.92, 1.0, f); ctx.paint()
        ctx.restore()
    sh = shake_at(t)
    if sh > 0.1:
        out = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        c = cairo.Context(out)
        c.translate(W / 2 + sh * math.sin(t * 91), H / 2 + sh * math.cos(t * 77)); c.scale(1.02, 1.02)
        c.translate(-W / 2, -H / 2)
        c.set_source_surface(s, 0, 0); c.paint()
        return out
    return s


def frame_bytes(t):
    s = render_frame(t)
    s.flush()
    return bytes(s.get_data())


# ======================================================================================
# sound: effects under the hook, the title and the body; AIZO from #56, the only song
# ======================================================================================
def sfx_at():
    """(time, effect, extra dB) placed by hand: the hits the picture is built around."""
    out = [(0.0, "boom", -4.0), (A(2) + 2.9, "question", 0.0), (HOOK - 1.8, "riser", 0.0), (A(3), "boom", 0.0),
           (A(53), "whoosh", 0.0), (A(55), "success", -2.0), (A(4), "whoosh", -4.0), (at(4, 1), "boom", -8.0),
           (at(6, 1), "stone_big", 0.0), (at(18, 6), "chime", -4.0), (at(47, 4), "riser", -6.0),
           (A(48) - 1.8, "riser", 0.0), (A(48), "boom", 2.0), (A(50), "pop", 0.0), (A(56) - 1.8, "riser", 0.0), (A(56), "boom", 0.0)]
    out += [(A(2) + 0.4 + 0.75 * i, "tick", 4.0) for i in range(3)]
    out += [(at(3, i + 1), "tick", 2.0) for i in range(5)]                 # the title's formulas light up
    out += [(at(11, j), "tick", 4.0) for j in range(4)]                     # the road laid out step by step
    out += [(at(n, 4 if i == 0 else 2), "chime", -6.0) for n, i in CHAPTERS.items()]
    out += [(A(n), "whoosh_soft", 0.0) for n in CHAPTERS if n != 11]
    chapter_starts = {A(n) for n in CHAPTERS}
    out += [(a, "whoosh_soft", -3.0) for a, _, _ in scene_lineup() if A(4) < a < A(53) and a not in chapter_starts]
    out += [(h, "stone", -8.0) for h in finale_hits()]
    out += [(h, "stone", -6.0) for h in recap_hits()[:12]]
    return out


def pulses():
    """A soft low thump on every downbeat of the body, where there is no music to carry the bar."""
    return [(f, "pulse", 0.0) for f, k in beats() if k == 1 and A(4) + 0.1 < f < A(56) - 2.0]


def _grey(t):
    s = render_frame(t, clean=True)
    a = np.ndarray((H, W, 4), np.uint8, s.get_data(), strides=(s.get_stride(), 4, 1))
    return a[::6, ::6, :3].astype(np.float32).mean(axis=2)


def _change(f):
    """How much the picture changes across beat f, and just before it (mean |grey difference| per pixel, 0-255)."""
    p, q, r = _grey(f - 0.32), _grey(f - 0.04), _grey(f + 0.26)
    return float(np.abs(r - q).mean()), float(np.abs(q - p).mean())


@lru_cache(None)
def entrance_cues():
    """(time, effect, extra dB) for the body's beats where something new enters the picture, found by rendering
    around each beat: a bigger entrance gets a pop, a small one a tick (not two ticks in a row)."""
    from multiprocessing import Pool
    fixed = [a for a, _, _ in sfx_at()] + [A(n) for n in rows()]
    cand = [f for f, _ in beats() if A(4) + 0.3 < f < A(56) - 0.3 and min(abs(f - x) for x in fixed) > 0.12]
    with Pool(min(18, os.cpu_count() or 4)) as pool:
        change = pool.map(_change, cand)
    out, last = [], -9.0
    for f, (after, before) in zip(cand, change):
        if after < 1.2 or after < 2.2 * before:
            continue
        if after >= 2.5:
            out.append((f, "pop", 4.0)); last = f
        elif f - last > 1.5 * BEAT:
            out.append((f, "tick", 2.0)); last = f
    return out


def load_wav(path, sr):
    fs, x = wavfile.read(path)
    x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else (2 ** 31 if x.dtype == np.int32 else 1.0))
    x = x if x.ndim == 2 else np.repeat(x[:, None], 2, axis=1)
    if fs != sr:
        g = math.gcd(sr, fs)
        x = resample_poly(x, sr // g, fs // g, axis=0).astype(np.float32)
    return x[:, :2]


def cmd_audio():
    sr = 48000
    n = int(END * sr)
    y = np.zeros((n, 2), np.float32)                          # music
    fx = np.zeros((n, 2), np.float32)                         # effects and the ambient bed

    def add(at_, clip, db=0.0, buf=y):
        k = int(round(at_ * sr))
        m = min(len(clip), n - k)
        if m > 0:
            buf[k:k + m] += clip[:m] * 10 ** (db / 20)

    def piece(x, s0, dur, fade_in=0.02, fade_out=0.02):
        x = x[int(s0 * sr):int((s0 + dur) * sr)].copy()
        i, o = int(fade_in * sr), int(fade_out * sr)
        x[:i] *= np.linspace(0, 1, i)[:, None]
        x[-o:] *= np.linspace(1, 0, o)[:, None]
        return x

    add(A(56), piece(load_wav(FINALE_SONG, sr), FINALE_FROM, END - A(56)), FINALE_GAIN)
    gains = json.load(open(os.path.join(SFX, "gains.json")))
    clips = {}
    for at_, name, extra in sfx_at() + pulses() + entrance_cues():
        if name not in clips:
            clips[name] = load_wav(os.path.join(SFX, name + ".wav"), sr)
        add(at_, clips[name], gains.get(name, 0.0) + extra, fx)
    amb = load_wav(os.path.join(SFX, "ambient.wav"), sr)      # the room under everything without music
    add(0.0, piece(np.tile(amb, (int(A(56) / 32) + 3, 1)), 0.0, A(56) + 0.5, 0.01, 1.0), gains["ambient"] + 6.0, fx)
    up = 10 ** (FX_LIFT / 20)                                 # lift the effects outside the music, 0.4 s ramps
    knots = [(0.0, up), (A(56) - 0.4, up), (A(56), 1.0), (END, 1.0)]
    fx *= np.interp(np.arange(n) / sr, *zip(*knots)).astype(np.float32)[:, None]
    from scipy.ndimage import minimum_filter1d, maximum_filter1d, uniform_filter1d
    w = int(0.04 * sr)                                        # min-then-mean gain never lets a peak past the ceiling
    g = np.minimum(1.0, FX_CEIL / np.maximum(maximum_filter1d(np.abs(fx).max(axis=1), w), 1e-9))
    fx *= uniform_filter1d(minimum_filter1d(g, w), w)[:, None]
    y += fx
    fade = int(3.0 * sr)
    y[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 2
    peak = float(np.abs(y).max())
    if peak > 0.89:                                     # -1 dBFS headroom so the AAC encode does not clip
        y *= 0.89 / peak
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "film_audio.wav")
    wavfile.write(path, sr, (y * 32767).astype(np.int16))
    cues = entrance_cues()
    print(path, f"{len(y) / sr:.2f}s", f"mix peak {20 * math.log10(max(peak, 1e-9)):.1f} dBFS before trim;",
          f"{len(cues)} entrance cues ({sum(c[1] == 'pop' for c in cues)} pops), {len(pulses())} pulses")
    return path


# ======================================================================================
# commands
# ======================================================================================
def cmd_check():
    r = rows()
    grid = [f for f, k in beats() if k in (1, 3)]
    bad = 0
    prev = 0.0
    for n in sorted(r, key=lambda n: r[n]["a"]):             # rows keep their v2 numbers: check in time order
        x = r[n]
        msgs = []
        slot = x["b"] - x["a"]
        room = need(x["sub"]) + 0.12 + (0 if n in QUICK else 0.6 + LOOK)
        if x["sub"] and room > slot + 1e-3:
            msgs.append(f"needs {room:.2f}s with picture time, has {slot:.2f}s")
        if x["sub"] and len(plain(x["sub"]).replace(" ", "")) > 18:
            msgs.append("over 18 characters")
        if n not in (1, 2) and min(abs(x["a"] - g) for g in grid) > 0.06:
            msgs.append("starts off the bar / half bar")
        if x["a"] < prev - 1e-3:
            msgs.append("overlaps the previous line")
        prev = x["b"]
        bad += bool(msgs)
        if msgs:
            print(f"#{n} {x['sub']}: " + "; ".join(msgs))
    empty = sorted(set(r) - covered_rows())
    print(f"{len(r)} rows, {bad} with problems; film {END:.2f}s; shots missing for rows {empty or 'none'}")
    missing = [p for p in [os.path.join(MEMES, "liang_shush.jpg"), os.path.join(MEMES, "laoshi_zhuanqian.jpg"),
                           FINALE_SONG, FINALE_BEATS] +
               [os.path.join(FORMULAS, n + ".png") for n in FORMULA_INFO] if not os.path.exists(p)]
    if missing:
        print("missing inputs:", missing)
    return bad == 0 and not missing


def cmd_snap(times, short=False):
    from PIL import Image as _Image
    d = os.path.join(OUT, "snaps")
    os.makedirs(d, exist_ok=True)
    (tw, th), cols = ((270, 480), 6) if short else ((640, 360), 3)
    thumbs = []
    for t in times:
        p = os.path.join(d, f"{'s' if short else 't'}{t:07.2f}.png")
        (short_frame if short else render_frame)(t).write_to_png(p)
        im = _Image.open(p).convert("RGB")
        im.thumbnail((tw, th))
        thumbs.append((t, im))
    sheet = _Image.new("RGB", (cols * (tw + 10), ((len(thumbs) + cols - 1) // cols) * (th + 10)), (40, 40, 40))
    for i, (t, im) in enumerate(thumbs):
        sheet.paste(im, ((i % cols) * (tw + 10) + 5, (i // cols) * (th + 10) + 5))
    path = os.path.join(d, "contact.png")
    sheet.save(path)
    print(path)


def _render_chunk(args):
    a, b, path, short = args
    w, h, fb = (SHORT_W, SHORT_H, short_bytes) if short else (W, H, frame_bytes)
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{w}x{h}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                          "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for n in range(int(round(a * FPS)), int(round(b * FPS))):
        p.stdin.write(fb(n / FPS))
    p.stdin.close()
    p.wait()
    return path


def render_video(a, b, jobs, name, short=False):
    """Frames a..b in parallel chunks, joined (no sound) into chunks/<name>_video.mp4."""
    from multiprocessing import Pool
    os.makedirs(os.path.join(OUT, "chunks"), exist_ok=True)
    edges = [a + (b - a) * i / jobs for i in range(jobs + 1)]
    edges = [round(e * FPS) / FPS for e in edges]
    tasks = [(edges[i], edges[i + 1], os.path.join(OUT, "chunks", f"{name}_{i:03d}.mp4"), short) for i in range(jobs)]
    with Pool(jobs) as pool:
        parts = pool.map(_render_chunk, tasks)
    lst = os.path.join(OUT, "chunks", f"{name}.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    video = os.path.join(OUT, "chunks", f"{name}_video.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", video],
                   check=True)
    for p in parts + [lst]:
        os.remove(p)
    return video


def cmd_render(a, b, jobs, name):
    assert cmd_check(), "fix the storyboard first"
    video = render_video(a, b, jobs, name)
    audio = cmd_audio()
    final = os.path.join(OUT, f"{name}.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i",
                    audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    final], check=True)
    os.remove(video)
    print(final)


# ======================================================================================
# the vertical short «选择你的量化公式» (1080 x 1920): the ending with the film frame at 1080 wide between a title and a
# question; #54's items get numbers, #51's recap gives way to a numbered menu of them, and #55's 理论上 is redrawn
# at the short's own size to end on
# ======================================================================================
SHORT_W, SHORT_H, SHORT_Y = 1080, 1920, 648       # the frame shrinks to 1080 x 608; its top lands on the room's grid
SHORT_K = SHORT_W / W
SHORT_FEATHER = 40                                # px at the frame's top and bottom that fade into the room
SHORT_CTA = SHORT_Y + 608 + 128                   # baseline of the question under the frame
SHORT_MENU = {"bachelier": "随机游走", "markowitz": "投资组合理论", "kelly": "凯利公式", "sharpe": "夏普比率",
              "black_scholes": "期权定价公式", "SIMONS": "西蒙斯的公式", "grinold": "主动管理基本定律",
              "fama_french": "三因子模型"}


def short_span():
    """Two seconds of the meme first (it is up, so the first frame is the cover), then AIZO with #56; out on #55,
    before #57 starts to cross in."""
    return round((A(56) - 2.0) * FPS) / FPS, round((A(57) - XF - 0.05) * FPS) / FPS


@lru_cache(None)
def short_beats():
    """AIZO beats from #51: the menu's rows land on 0-7, the question turns and the cursor starts on 8."""
    return tuple(f for f, _ in beats() if f >= A(51) - 0.05)


@lru_cache(None)
def short_room():
    """The film's room continued around the frame (shrunk with it, so the seam is invisible), the kicker, the hedge."""
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, SHORT_W, SHORT_H)
    ctx = cairo.Context(s)
    ctx.save(); ctx.scale(SHORT_K, SHORT_K)
    ctx.set_source_surface(background(W, math.ceil(SHORT_H / SHORT_K), round(SHORT_Y / SHORT_K)), 0, 0)
    ctx.get_source().set_filter(cairo.FILTER_GOOD); ctx.paint()
    ctx.restore()
    text(ctx, "一口气教你学会量化交易", SHORT_W / 2, 400, 34, "ink2", 0.85, align="center", spacing=0.12)
    text(ctx, "仅供科普，不构成投资建议", SHORT_W / 2, SHORT_CTA + 82, 28, "dim", align="center")
    return s


@lru_cache(None)
def short_mask():
    """The frame's alpha: 1 inside, easing to 0 over SHORT_FEATHER px at its top and bottom."""
    y = np.arange(round(H * SHORT_K), dtype=float)
    ramp = np.minimum(smooth(y / SHORT_FEATHER), smooth(y[::-1] / SHORT_FEATHER))
    return surf_from(np.ones(3), np.tile(ramp[:, None], (1, SHORT_W)), dither=False)


def two_tone(ctx, parts, cx, y, size, glow):
    """One glowing line centred on cx, each (text, colour) part in its own colour."""
    x = cx - sum(text_width(ctx, s, size, SANS, True) for s, _ in parts) / 2
    for s, col in parts:
        glow_text(ctx, s, x, y, size, col, glow=glow)
        x += text_width(ctx, s, size, SANS, True)


def short_badge(ctx, t):
    """#54: each item's number beside it, landing with its beat."""
    hits = finale_hits()
    if not hits or not hits[0] <= t < A(51):
        return
    i = max(j for j, h in enumerate(hits) if h <= t)
    num(ctx, str(i + 1), 84, SHORT_Y + 470 * SHORT_K + 30, 92, "amber", 1.0, align="center",
        glow=0.8 + kick(t, hits[i], 0.4), scale=pop((t - hits[i]) / 0.25))


def short_menu(ctx, t, a):
    """#51's slot: the eight items as numbered rows, one per beat from the right; from beat 8 a cursor runs down."""
    bs = short_beats()
    for i, name in enumerate(FINALE):
        age = t - bs[i]
        if age < 0:
            break
        k = ease_out(age / 0.13)
        al = a * min(1.0, age / 0.06)
        cur = bs[8 + i] <= t < bs[9 + i]
        y = 690 + 84 * i
        ctx.save(); ctx.translate((1 - k) * 420, 0)
        rrect(ctx, 60, y - 38, 960, 76, 12)
        fill(ctx, "amber" if cur else "glass", (0.14 if cur else 0.55) * al, keep=True)
        stroke(ctx, "amber" if cur else "grid", 2.5 if cur else 1.2, al)
        num(ctx, str(i + 1), 110, y + 20, 56, "amber", al, align="center",
            glow=0.5 + 0.6 * kick(t, bs[8], 0.6) + 0.4 * cur)
        glow_text(ctx, SHORT_MENU[name], 170, y + 2, 36, "ink", al, glow=0.3 + 0.2 * cur)
        sub = "文艺复兴 · 大奖章基金" if name == "SIMONS" else " · ".join(FORMULA_INFO[name][:2])
        text(ctx, sub, 170, y + 30, 20, "ink2", al)
        if name == "SIMONS":                          # the one formula nobody has published
            glow_text(ctx, "？", 960, y + 22, 60, "amber", al, align="center", glow=0.9)
        else:                                         # as large as the row allows: 66 px tall, clear of the name
            room = 1000 - 210 - max(text_width(ctx, SHORT_MENU[name], 36, SANS, True), text_width(ctx, sub, 20))
            _, fh = formula_size(name, 400)
            fw = min(room, 400 * 66 / fh)
            formula(ctx, name, 1000 - fw / 2, y, fw, al, glow=0.4)
        ctx.restore()


def short_learned(ctx, t):
    """#55 where the menu was, the stamp big enough to read on a phone, landing two beats after the line."""
    glow_text(ctx, "你已经完全学会了量化交易了", SHORT_W / 2, 990, 66, "ink", on(t, A(55), 0.25), align="center",
              glow=0.8, scale=pop((t - A(55)) / 0.3))
    stamp(ctx, "理论上", 880, 885, 52, "red", 1.0, (t - A(55) - 2 * FINALE_BEAT) / 0.5, rot=0.12)


def short_frame(t):
    """One frame of the short: the room, the film frame up to #51, the menu, #55, the title, the question."""
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, SHORT_W, SHORT_H)
    ctx = cairo.Context(s)
    ctx.set_source_surface(short_room(), 0, 0); ctx.paint()
    fa = 1 - sm((t - A(51)) / 0.15)
    f = flash_at(t) * (fa if t < A(55) - XF else 1.0)
    if f > 0.01:                                      # the frame's flashes light the room too: no lit box
        ctx.save(); ctx.set_operator(cairo.OPERATOR_ADD); ctx.set_source_rgba(0.85, 0.92, 1.0, f); ctx.paint()
        ctx.restore()
    if fa > 0.002:
        ctx.push_group()
        ctx.save(); ctx.translate(0, SHORT_Y); ctx.scale(SHORT_K, SHORT_K)
        ctx.set_source_surface(render_frame(t), 0, 0); ctx.get_source().set_filter(cairo.FILTER_GOOD); ctx.paint()
        ctx.restore()
        ctx.set_operator(cairo.OPERATOR_DEST_IN); ctx.set_source_surface(short_mask(), 0, SHORT_Y); ctx.paint()
        ctx.pop_group_to_source(); ctx.paint_with_alpha(fa)
    if A(51) <= t:
        short_menu(ctx, t, 1 - sm((t - A(55)) / 0.25))
    if A(55) <= t:
        short_learned(ctx, t)
    short_badge(ctx, t)
    two_tone(ctx, [("选择你的", "ink"), ("量化公式", "amber")], SHORT_W / 2, 528, 112, 0.8 + 0.6 * kick(t, A(56), 0.6))
    turn = short_beats()[8]
    parts = [("你选几号？", "ink")] if t >= turn else [("你选哪个公式？", "ink")]
    k = pop((t - turn) / 0.3) if t >= turn else 1.0
    ctx.save(); ctx.translate(SHORT_W / 2, SHORT_CTA); ctx.scale(k, k)
    two_tone(ctx, parts + [("评论区见", "amber")], 0, 0, 54, 0.35 + kick(t, turn, 0.5))
    ctx.restore()
    return s


def short_bytes(t):
    s = short_frame(t)
    s.flush()
    return bytes(s.get_data())


def cmd_short(jobs, name):
    assert cmd_check(), "fix the storyboard first"
    a, b = short_span()
    video = render_video(a, b, jobs, name, short=True)
    audio = cmd_audio()
    final = os.path.join(OUT, f"{name}.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i",
                    audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af",
                    f"afade=t=in:d=0.03,afade=t=out:st={b - a - 1.0:.3f}:d=1.0", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", final], check=True)
    os.remove(video)
    print(final)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("snap"); s1.add_argument("times", type=float, nargs="+")
    s1.add_argument("--short", action="store_true")
    s2 = sub.add_parser("render"); s2.add_argument("a", type=float); s2.add_argument("b", type=float)
    s2.add_argument("--jobs", type=int, default=8); s2.add_argument("--name", default="preview")
    s3 = sub.add_parser("short"); s3.add_argument("--jobs", type=int, default=8)
    s3.add_argument("--name", default="short_select_formula")
    sub.add_parser("audio")
    sub.add_parser("check")
    args = ap.parse_args()
    if args.cmd == "snap":
        cmd_snap(args.times, args.short)
    elif args.cmd == "render":
        cmd_render(args.a, args.b, args.jobs, args.name)
    elif args.cmd == "short":
        cmd_short(args.jobs, args.name)
    elif args.cmd == "check":
        sys.exit(0 if cmd_check() else 1)
    else:
        cmd_audio()
