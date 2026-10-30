"""The App hands off to a cyan terminal, then a stock wall (#4–#17, except #11).

#4–#6 and their drawing helpers are preserved. #7–#10 share one terminal;
#12–#17 magnify, score, barcode and rank the same 100×50 wall. Frames are
stateless; expensive market geometry and wall exposures are cached.
"""
import math
from functools import lru_cache

import numpy as np

from quant_kit import *  # noqa: F401,F403


# ======================================================================================
# preserved entrance helpers and cached stock-wall exposures
# ======================================================================================
def _safe(draw):
    """Contain glows, stamp overshoot and the renderer's out-of-range cross-fade calls."""
    def wrapped(ctx, t):
        ctx.save()
        try:
            ctx.rectangle(60, 90, 1800, 780); ctx.clip()
            draw(ctx, t)
        finally:
            ctx.restore()
    wrapped.__name__ = draw.__name__
    wrapped.__doc__ = draw.__doc__
    return wrapped


def _ripple(ctx, x, y, t, t0, col="amber", a=1.0):
    age = t - t0
    if not 0 <= age <= 0.85:
        return
    for delay in (0.0, 0.14):
        u = (age - delay) / 0.7
        if 0 <= u <= 1:
            ctx.new_path(); ctx.arc(x, y, 10 + 90 * ease_out(u), 0, 2 * math.pi)
            stroke(ctx, col, 2.5, a * (1 - u))
    dot(ctx, x, y, 5, col, a * kick(t, t0, 0.65), 1.0)


def _app(ctx, cx, top, scale=1.0, a=1.0, upto=26, record=False, captions=True, short_captions=False):
    """One amber App, reused without people or a hand cursor; local dimensions are 340×520."""
    ctx.save(); ctx.translate(cx, top); ctx.scale(scale, scale)
    panel(ctx, -170, 0, 340, 520, a, edge="amber", r=32, corners=False, fill_a=0.9)
    rrect(ctx, -42, 16, 84, 6, 3); fill(ctx, "amber", 0.45 * a)
    if captions or short_captions:
        text(ctx, "交易记录" if record else "单股窗口", -136, 70, 26, "amber", a, bold=True)
    if captions:
        label(ctx, "RECORD" if record else "TRADING APP", 136, 68, 18, "ink2", a,
              align="right", spacing=0.12)
    gline(ctx, [(-136, 91), (136, 91)], "dim", 1, 0.5 * a, glow=0)
    candles(ctx, walk(9, 26, 0.05, 1.2), -136, 120, 272, 230, upto=upto, a=0.8 * a, mono="amber")
    for yy in (120, 235, 350):
        gline(ctx, [(-136, yy), (136, yy)], "dim", 1, 0.3 * a, glow=0)
    if captions:
        label(ctx, "LOCAL VIEW", -136, 390, 18, "ink2", a, spacing=0.12)
    for x, s, col in ((-136, "买入", "amber"), (16, "卖出", "dim")):
        rrect(ctx, x, 416, 120, 56, 12); fill(ctx, col, 0.18 * a, keep=True)
        stroke(ctx, col, 1.8, 0.8 * a)
        if captions or short_captions:
            text(ctx, s, x + 60, 453, 30, col, a, bold=True, align="center")
    gline(ctx, [(-42, 498), (42, 498)], "dim", 3, 0.6 * a, glow=0)
    ctx.restore()


@lru_cache(None)
def _wall_textures():
    rgb, alpha = wall_image(100, 50, 6, 2, 9)
    return surf_rgb(rgb, alpha), surf_rgb(np.clip(rgb * 1.8, 0, 1), alpha)


def _stock_wall(ctx, x, y, w, h, t, a=1.0, cols=100.0, rows_=50.0, scan=False, scan_speed=0.14):
    """Crop/zoom the cached 100×50 texture. A narrow band alone gets a brighter cached exposure."""
    base, bright = _wall_textures()
    cols, rows_ = max(1.0, min(100.0, cols)), max(1.0, min(50.0, rows_))
    phase = (max(0.0, t) * scan_speed) % 1.0
    ctx.save(); ctx.rectangle(x, y, w, h); ctx.clip()
    ctx.translate(x, y); ctx.scale(w / (cols * 8), h / (rows_ * 8))
    ctx.set_source_surface(base, 0, 0); ctx.get_source().set_filter(cairo.FILTER_NEAREST)
    ctx.paint_with_alpha(a)
    if scan:
        sx = phase * cols * 8
        ctx.save(); ctx.rectangle(sx - 12, 0, 24, rows_ * 8); ctx.clip()
        ctx.set_source_surface(bright, 0, 0); ctx.get_source().set_filter(cairo.FILTER_NEAREST)
        ctx.paint_with_alpha(0.85 * a)
        ctx.rectangle(sx - 12, 0, 24, rows_ * 8); fill(ctx, "cyan", 0.12 * a)
        ctx.restore()
    ctx.restore()
    if scan:
        sx = x + phase * w
        gline(ctx, [(sx, y), (sx, y + h)], "cyan", 2.2, 0.8 * a, glow=0.8)


# ======================================================================================
# #4–#6: preserved entrance
# ======================================================================================
@_safe
def shot_old_way(ctx, t):
    """#4-#6 in one fixed frame, so the eye stays on the App while the three quick lines run (three layouts in a row
    tired the owner). #4: the App and a rule terminal; VS slams in on b1, all 16 terminal cells light on b2.
    #5: the terminal steps back, the chart draws, 感觉要涨 pops up on b1, the buy button is tapped on b2.
    #6: the 古法交易 stamp lands on the App on b1.
    """
    cx, top, sc = 600, 150, 1.22
    back = on(t, A(5), 0.3)
    _app(ctx, cx, top, sc, upto=12 + 14 * ease_out(max(0.0, t - A(5)) / (1.5 * BEAT)), captions=False)
    bx, by = cx - 76 * sc, top + 444 * sc                       # the buy button's centre
    text(ctx, "买入", bx, by + 12, 36, "amber", bold=True, align="center")
    # the computer: a rule terminal whose 16 cells all work at once
    kt = on(t, A(4), 0.25) * (1 - 0.8 * back)
    hit, lit = kick(t, at(4, 2), 0.5), on(t, at(4, 2), 0.15)
    panel(ctx, 1080, 250, 640, 450, kt, edge="cyan", fill_a=0.8, glow=0.6 * hit)
    for j in range(4):
        for i in range(4):
            x, y = 1104 + i * 151, 274 + j * 103
            ctx.rectangle(x, y, 139, 91)
            fill(ctx, "cyan", (0.04 + 0.12 * lit + 0.25 * hit) * kt, keep=True)
            stroke(ctx, "cyan", 1.4, (0.35 + 0.5 * lit) * kt)
            for k in range(2):
                yy = y + 32 + 26 * k
                gline(ctx, [(x + 18, yy), (x + 18 + (90 - 30 * k) * lit, yy)], "cyan", 1.4, 0.7 * kt * lit, glow=0)
    kv = on(t, at(4, 1), 0.15) * (1 - back)
    glow_text(ctx, "VS", 943, 520, 130, "ink", kv, align="center", glow=0.8 + kick(t, at(4, 1), 0.5),
              scale=pop((t - at(4, 1)) / 0.3), font=LATIN)
    # you: a hunch pops up, a tap
    kb = on(t, at(5, 1), 0.22)
    if kb > 0:
        x, y = 880 + 30 * (1 - kb), 330
        panel(ctx, x, y, 430, 100, kb, edge="amber", r=24, corners=False, fill_a=0.94)
        dot(ctx, x + 36, y + 50, 5, "amber", kb, 0.6)
        text(ctx, "感觉要涨", x + 66, y + 64, 40, "amber", kb, bold=True)
    _ripple(ctx, bx, by, t, at(5, 2))
    glow_at(ctx, cx, 470, "red", 2.2, 0.3 * kick(t, at(6, 1), 0.55))
    stamp(ctx, "古法交易", cx + 20, 480, 96, "red", 1.0, (t - at(6, 1)) / 0.6, rot=-0.12)


# ======================================================================================
# shared stages: terminal, wall detail, barcode and score ordering
# ======================================================================================
def _frame(draw):
    """Reserve the chapter track, subtitles and notes without changing the old entrance crop."""
    def wrapped(ctx, t):
        ctx.save()
        try:
            ctx.rectangle(60, 95, 1800, 830); ctx.clip()
            draw(ctx, t)
        finally:
            ctx.restore()
    wrapped.__name__ = draw.__name__
    wrapped.__doc__ = draw.__doc__
    return wrapped


def _slip(ctx, x, y, a=1.0, w=78, h=110):
    """A wordless paper order, with a folded corner and perforated tear edge."""
    if a <= 0.002:
        return
    pts = [(x, y), (x + w - 14, y), (x + w, y + 14), (x + w, y + h)]
    pts += [(x + w - i * w / 12, y + h - (5 if i % 2 else 0)) for i in range(1, 13)]
    path(ctx, pts, close=True); fill(ctx, "glass", 0.97 * a, keep=True); stroke(ctx, "cyan", 1.8, a)
    gline(ctx, [(x + w - 14, y), (x + w - 14, y + 14), (x + w, y + 14)], "cyan", 1, a, glow=0)
    for i in range(3):
        gline(ctx, [(x + 12, y + 32 + i * 15), (x + w - 16 - i * 8, y + 32 + i * 15)],
              "ice", 1.3, 0.35 * a, glow=0)


@lru_cache(None)
def _terminal_prices():
    """A lightly smoothed seeded price walk, sampled continuously as the terminal scrolls."""
    return gaussian_filter(walk(707, 2048, 0.0, 0.9)[:, 3], 0.55)


def _terminal(ctx, t, cx=960, cy=510, scale=1.35, a=1.0, chart=1.0, words=0.0,
              locked=1.0, flare=0.0, legacy=0.0, tickets=()):
    """The inherited 640×450 terminal, transformed as one object including its paper outlet."""
    ctx.save(); ctx.translate(cx - 320 * scale, cy - 225 * scale); ctx.scale(scale, scale)
    panel(ctx, 0, 0, 640, 450, a, edge="cyan", fill_a=0.8, glow=0.35 * flare)
    for j in range(4):
        for i in range(4):
            x, y = 24 + i * 151, 24 + j * 103
            ctx.rectangle(x, y, 139, 91)
            fill(ctx, "cyan", 0.16 * a * legacy, keep=True); stroke(ctx, "cyan", 1.4, 0.85 * a * legacy)
            for k in range(2):
                yy = y + 32 + 26 * k
                gline(ctx, [(x + 18, yy), (x + 108 - 30 * k, yy)], "cyan", 1.4,
                      0.7 * a * legacy, glow=0)
    for yy, word in ((150, "卖"), (310, "买")):
        gline(ctx, [(48, yy), (588, yy)], "cyan", 1.5 + 2.0 * locked + 7 * flare,
              chart * a * min(1.0, 0.3 + 0.65 * locked + flare),
              glow=0.3 + 3 * flare, dash=[10, 9])
        for xx in (48, 588):
            glow_at(ctx, xx, yy, "ice", 1.6, 0.95 * chart * a * flare)
        glow_text(ctx, word, 78, yy - 15, 38, "cyan", words * a, glow=0.45)
        for xx, direction in ((48, 1), (588, -1)):
            gline(ctx, [(xx, yy - 13), (xx, yy + 13), (xx + 13 * direction, yy + 13)],
                  "ice", 2, a * chart * locked, glow=0.4 + flare)
    u = np.linspace(0, 1, 121)
    series = _terminal_prices()
    sample = np.arange(len(u)) + max(0.0, (t - A(7)) / BEAT) * 6
    prices = np.interp(sample, np.arange(len(series)), series)
    prices = prices - prices.mean()
    ys = 230 - 70 * prices / max(float(np.abs(prices).max()), 1e-6)
    touch_flash = 0.0
    if A(8) <= t <= B(8):
        b = bt(t, 8)
        hits = (0.8, 4.0, 4.25, 4.5)
        touch = max(math.exp(-((b - hit) / 0.18) ** 2) for hit in hits)
        ys = mix(ys, 310, touch * u ** 12)
        touch_flash = max(kick(t, at(8, hit), 0.75) for hit in hits)
    pts = [(48 + 540 * float(q), float(y)) for q, y in zip(u, ys)]
    gline(ctx, pts, "ice", 2.3, a * chart, glow=0.6)
    dot(ctx, *pts[-1], 4, "cyan", a * chart, 0.7)
    glow_at(ctx, 588, 310, "cyan", 1.6, a * touch_flash)
    dot(ctx, 588, 310, 8, "ice", a * touch_flash, 1.5)
    gline(ctx, [(640, 112), (640, 385)], "cyan", 3, a * chart, glow=0.3)
    ctx.save(); ctx.rectangle(640, -20, 650, 540); ctx.clip()
    for x, y, opacity, angle in tickets:
        ctx.save(); ctx.translate(x + 240 / 2.7, y + 320 / 2.7); ctx.rotate(angle)
        _slip(ctx, -240 / 2.7, -320 / 2.7, a * opacity, w=240 / 1.35, h=320 / 1.35)
        ctx.restore()
    ctx.restore(); ctx.restore()


def _orders(t):
    """Slide a full-size order into the right third, then replace it with a three-slip fan."""
    burst = on(t, at(8, 4), 0.25)
    first = on(t, at(8, 0.8), 0.6)
    out = [(mix(640 - 240 / 1.35, 770, first), 139, first * (1 - burst), 0.0)]
    for i, beat in enumerate((4.0, 4.18, 4.36)):
        k = on(t, at(8, beat), 0.5)
        out.append((mix(640 - 240 / 1.35, 700 + 70 * i, k),
                    112 + 27 * abs(i - 1), k, (i - 1) * 0.13 * k))
    return tuple(out)


def _intro_wall(ctx, t, a=0.8, scale=1.0, sweep=None):
    """One fixed-centre stock mural, including an optional non-repeating scan."""
    x, y, w, h = 960 - 800 * scale, 550 - 300 * scale, 1600 * scale, 600 * scale
    _stock_wall(ctx, x, y, w, h, 0, a=a)
    if sweep is not None and 0 <= sweep <= 1:
        sx = x + w * sweep
        ctx.save(); ctx.rectangle(sx - 18, y, 36, h); ctx.clip()
        _stock_wall(ctx, x, y, w, h, 0, a=0.8)
        ctx.restore()
        gline(ctx, [(sx, y), (sx, y + h)], "cyan", 3, 0.8, glow=0.8)


def _card(ctx, name, lit=None):
    """Chinese-only formula authority in a reserved strip above the wall."""
    width = {"macd": 650, "factor": 720, "fama_french": 1180}[name]
    _, fh = formula_size(name, width)
    if name != "factor" and fh > 115:
        width *= 115 / fh
        _, fh = formula_size(name, width)
    cx = 1320 if name != "fama_french" else 960
    formula_card(ctx, name, cx, 110 + (fh + 128) / 2, width, lit=lit, en=False)


@lru_cache(None)
def _tile_data():
    """Candles and both genuine exponential averages share one price axis."""
    data = walk(12, 96, 0.02, 0.9)
    fast, slow = ema(data[:, 3], 12), ema(data[:, 3], 26)
    lo, hi = float(data[:, 2].min()), float(data[:, 1].max())
    p = curve_pts(fast, -330, -130, 660, 250, lo=lo, hi=hi)
    q = curve_pts(slow, -330, -130, 660, 250, lo=lo, hi=hi)
    i = 30 + int(np.argmax(np.abs(fast[30:] - slow[30:])))
    return data, p, q, i


@lru_cache(None)
def _wall_scores():
    """Stable scores and all 5000 global rank destinations, brightest first."""
    scores = rng(940).uniform(0.05, 0.98, (50, 100))
    order = np.argsort(-scores, axis=None).reshape(50, 100)
    ranks = np.empty(5000, dtype=int)
    ranks[order.ravel()] = np.arange(5000)
    return scores, ranks.reshape(50, 100), order


@lru_cache(None)
def _score_texture(barcode=False, ranked=False):
    """A cached 5000-tile exposure; minified barcodes retain separated hairlines."""
    scores, _, order = _wall_scores()
    if ranked:
        scores = scores.ravel()[order]
    levels = np.floor(scores * 11) / 11
    if ranked:
        levels = np.broadcast_to((1 - np.arange(50)[:, None] / 49) ** 1.6, (50, 100))
        colour = mix(hx("cyan")[None, None], hx("ice")[None, None], 0.65 * levels[..., None])
        exposure = colour * (0.02 + 0.98 * levels[..., None])
    else:
        exposure = hx("cyan")[None, None] * (0.1 + 0.9 * levels[..., None])
    rgb = np.repeat(np.repeat(exposure, 12, 0), 24, 1)
    mask = np.ones((12, 24))
    mask[10:, :] = 0; mask[:, 22:] = 0
    if barcode:
        mask[:10, :22] = np.tile([1.0, 0.35 if ranked else 0.12], 11)[None, :]
    return surf_rgb(rgb, np.tile(mask, (50, 100)))


@lru_cache(None)
def _barcode_detail():
    """2048 deterministic factor stripes on the magnified tile, not thousands of labels."""
    values = rng(140).uniform(0.12, 1.0, 2048)
    alpha = np.repeat(values, 2)[None, :] * np.ones((160, 1))
    alpha[:, 1::2] *= 0.08
    return surf_from(hx("cyan"), alpha)


def _stretch(ctx, surface, x, y, w, h, a):
    """Fit a cached texture to a rectangle without allocating a frame-sized surface."""
    if a <= 0.002:
        return
    ctx.save(); ctx.translate(x, y); ctx.scale(w / surface.get_width(), h / surface.get_height())
    ctx.set_source_surface(surface, 0, 0); ctx.get_source().set_filter(cairo.FILTER_NEAREST)
    ctx.paint_with_alpha(a); ctx.restore()


def _factor_stage(ctx, detail=0.0, score=0.0, barcode=0.0, wall_a=0.15, neighbours=0.0,
                  tile_scale=1.0, chart=1.0, tile_width=760, line_draw=1.0,
                  band=0.0, collapse=0.0, lift=0.0):
    """The same wall, with a bounded optical detail view of its central tile."""
    _stock_wall(ctx, 160, 350, 1600, 500, 0, a=wall_a * (1 - score))
    _stretch(ctx, _score_texture(False), 160, 350, 1600, 500, wall_a * score * (1 - barcode))
    _stretch(ctx, _score_texture(True), 160, 350, 1600, 500, wall_a * barcode)
    if detail <= 0.002:
        return
    ctx.save(); ctx.rectangle(160, 350, 1600, 500); ctx.clip()
    ctx.translate(960, 600); ctx.scale(tile_scale * tile_width / 760, tile_scale)
    for dx, level in ((-810, 0.27), (810, 0.88)):
        panel(ctx, dx - 380, -170, 760, 340, detail * neighbours, edge="cyan", fill_a=0.95)
        ctx.rectangle(dx - 378, -168, 756, 336); fill(ctx, "cyan", detail * neighbours * level * 0.28)
        if barcode:
            _stretch(ctx, _barcode_detail(), dx - 350, -135, 700, 270, detail * neighbours * barcode * level)
    panel(ctx, -380, -170, 760, 340, detail, edge="cyan", fill_a=0.98)
    ctx.rectangle(-378, -168, 756, 336); fill(ctx, "cyan", detail * score * 0.22)
    data, fast, slow, _ = _tile_data()
    ca = detail * chart * (1 - barcode) * (1 - 0.8 * score)
    candles(ctx, data, -330, -130, 660, 250, a=0.45 * ca)
    count = max(2, int(len(fast) * line_draw))
    if band:
        _gap_band(ctx, count, collapse, lift, tile_width / 760, tile_scale, ca * band)
    gline(ctx, fast[:count], "cyan", 3.5, ca, glow=0.55)
    gline(ctx, slow[:count], "amber", 3.5, ca, glow=0.45)
    bw, bh = mix(210, 700, barcode), mix(6, 270, barcode)
    by = mix(70, 0, barcode)
    _stretch(ctx, _barcode_detail(), -bw / 2, by - bh / 2, bw, bh, detail * barcode)
    ctx.restore()


def _gap_band(ctx, count, collapse, lift, xscale, scale, a):
    """Squash the actual EMA ribbon into a horizontal bar before lifting that same light out."""
    _, fast, slow, i = _tile_data()
    centre = (fast[i][1] + slow[i][1]) / 2
    shrink = mix(1.0, 210 / (660 * xscale * scale), lift)
    cy = mix(centre, -200 / scale, lift)
    edges = []
    for points, sign in ((fast[:count], -1), (slow[:count], 1)):
        edges.append([(x * shrink, mix(mix(y, centre + sign * 3, collapse),
                                      -200 / scale + sign * 3, lift)) for x, y in points])
    path(ctx, edges[0] + edges[1][::-1], close=True)
    fill(ctx, "cyan", a * (0.24 + 0.5 * collapse))
    for edge in edges:
        gline(ctx, edge, "cyan", 1.5, a * (0.35 + 0.55 * collapse), glow=0.55)
    left, right = fast[0][0] * shrink, fast[count - 1][0] * shrink
    gline(ctx, [(left, cy), (right, cy)], "cyan", 12, a * collapse ** 2, glow=1.2)
    for x in (left, right):
        dot(ctx, x, cy, 4.5, "ice", a * collapse, 0.8)


def _dif_bar(ctx, pull=1.0, drop=0.0, a=1.0, flare=0.0, length=210):
    """Extract the measured EMA gap as one bar, then return it as the tile's score."""
    _, fast, slow, i = _tile_data()
    p = [(960 + fast[i][0], 600 + fast[i][1]), (960 + slow[i][0], 600 + slow[i][1])]
    target = [(960 - length / 2, mix(400, 670, drop)), (960 + length / 2, mix(400, 670, drop))]
    pts = lerp_pts(p, target, pull)
    gline(ctx, pts, "cyan", mix(4, 12, pull), a, glow=0.8 + flare)
    for x, y in pts:
        dot(ctx, x, y, 4.5, "ice", a, 0.6 + flare)


@lru_cache(None)
def _weak_geometry():
    """A slight slow drift under non-periodic seeded market disturbance."""
    u = np.linspace(0, 1, 241)
    drift = gaussian_filter(walk(151, len(u), 0.0, 0.15)[:, 3], 10)
    signal = 22 * (u - 0.5) + 3 * (drift - drift.mean())
    shocks = gaussian_filter(rng(150).normal(0, 58, 2048), 0.55)
    walk_noise = walk(152, 2048, 0.0, 3.0)[:, 3]
    noise = shocks + 2 * (walk_noise - gaussian_filter(walk_noise, 18))
    return u, signal, noise


def _weak_stripe(ctx, clock, a=1.0, spread=1.0):
    """Reveal the cyan drift with a circular magnifier, without regularising the noise."""
    u, signal, noise = _weak_geometry()
    xs = 440 + 1040 * u
    sample = np.arange(len(u)) + max(0.0, clock) * 8
    wave = np.interp(sample, np.arange(len(noise)), noise) + signal
    ctx.save(); ctx.rectangle(430, 490, 1060, 260); ctx.clip()
    gline(ctx, [(float(x), float(600 + spread * y)) for x, y in zip(xs, wave)],
          "violet", 1.4, 0.42 * a, glow=0.1)
    gline(ctx, [(float(x), float(600 + spread * y)) for x, y in zip(xs, signal)],
          "cyan", 2.8, 0.7 * a, glow=0.6)
    lens = a * spread
    ctx.save(); ctx.arc(1120, 600, 95, 0, 2 * math.pi); ctx.clip()
    ctx.rectangle(1025, 505, 190, 190); fill(ctx, "glass", 0.8 * lens)
    part = slice(135, 181)
    lx = np.linspace(1025, 1215, 46)
    baseline = float(signal[part].mean())
    gline(ctx, [(float(x), float(600 + 0.75 * y)) for x, y in zip(lx, wave[part])],
          "violet", 1.3, 0.28 * lens, glow=0)
    gline(ctx, [(float(x), float(600 + 3.5 * (y - baseline))) for x, y in zip(lx, signal[part])],
          "cyan", 3.5, 0.95 * lens, glow=0.8)
    ctx.restore()
    ctx.arc(1120, 600, 95, 0, 2 * math.pi); stroke(ctx, "cyan", 1.8, 0.65 * lens)
    gline(ctx, [(1188, 668), (1240, 720)], "cyan", 3, 0.6 * lens, glow=0.3)
    ctx.restore()


def _rank_wall(ctx, k, a=1.0):
    """Move every barcoded tile to its global score rank, filling bright rows from the top."""
    if k <= 0 or k >= 1:
        _stretch(ctx, _score_texture(True, k >= 1), 160, 350, 1600, 500, a)
        return
    scores, ranks, _ = _wall_scores()
    levels = np.floor(scores * 11).astype(int)
    settle = sm((k - 0.72) / 0.28)
    for level in range(11):
        rr, cc = np.where(levels == level)
        ctx.new_path()
        for row, col in zip(rr, cc):
            rank = int(ranks[row, col])
            x = 160 + mix(float(col), float(rank % 100), k) * 16
            y = 350 + mix(float(row), float(rank // 100), k) * 10
            for offset in range(0, 14, 2):
                ctx.rectangle(x + offset, y, 0.85, 8.3)
        fill(ctx, "cyan", a * (1 - settle) * (0.1 + 0.9 * level / 11))
    _stretch(ctx, _score_texture(True, True), 160, 350, 1600, 500, a * settle)


# ======================================================================================
# #7–#10: the terminal takes over, works, and reveals its stock universe
# ======================================================================================
@_frame
def shot_rules(ctx, t):
    """#7: inherit the dim right terminal exactly; b2 locks both price thresholds."""
    move = sm((t - A(7)) / 1.25)
    fade = 1 - sm((t - A(7)) / 0.48)
    hit = kick(t, at(7, 2), 1.2)
    locked = on(t, at(7, 2), 0.12)
    chart = sm((t - A(7) - 0.3) / 0.8)
    words = chart
    _terminal(ctx, t, mix(1400, 960, move), mix(475, 510, move), mix(1, 1.35, move),
              a=mix(0.2, 1, move), chart=chart, words=words, legacy=1 - chart,
              locked=locked, flare=hit)
    if fade > 0:
        ctx.save(); ctx.push_group()
        _app(ctx, 600, 150, 1.22, captions=False)
        text(ctx, "买入", 600 - 76 * 1.22, 150 + 444 * 1.22 + 12, 36, "amber", bold=True, align="center")
        panel(ctx, 880, 330, 430, 100, edge="amber", r=24, corners=False, fill_a=0.94)
        dot(ctx, 916, 380, 5, "amber", glow=0.6)
        text(ctx, "感觉要涨", 946, 394, 40, "amber", bold=True)
        stamp(ctx, "古法交易", 620, 480, 96, "red", 1.0, 1.0, rot=-0.12)
        ctx.pop_group_to_source(); ctx.paint_with_alpha(fade); ctx.restore()
    glow_at(ctx, 960, 510, "cyan", 3.0, 0.55 * hit)


@_frame
def shot_pipeline(ctx, t):
    """#8: the same price touches buy; b4 releases three wordless paper orders."""
    shift = sm((t - A(8)) / 0.55)
    _terminal(ctx, t, cx=mix(960, 760, shift), words=1.0, tickets=_orders(t))


@_frame
def shot_stock_wall(ctx, t):
    """#9: a steady pullback opens the whole 100×50 wall on b4; it scans once."""
    u = float(np.clip((t - A(9)) / (4 * BEAT), 0, 1))
    zoom = 1.35 * (0.38 / 1.35) ** u
    wall_scale = 1.8 ** (1 - u)
    reveal = on(t, A(9), 1.0)
    phase = (t - at(9, 4)) / (2.4 * BEAT)
    _intro_wall(ctx, t, a=0.8 * reveal, scale=wall_scale,
                sweep=phase if 0 <= phase <= 1 else None)
    terminal_a = 1 - on(t, at(9, 4), 0.45)
    _terminal(ctx, B(8), cx=mix(760, 960, u), scale=zoom, a=terminal_a,
              words=1 - on(t, A(9), 0.35), tickets=_orders(B(8)))
    hit = kick(t, at(9, 4), 0.55)
    glow_at(ctx, 960, 530, "cyan", 4.2, 0.6 * hit)
    count = on(t, at(9, 4), 0.13) * (1 - on(t, B(9) - 0.3, 0.3))
    num(ctx, "5000", 960, 215, 104, "cyan", count, align="center", glow=0.7 + hit)


@_frame
def shot_same_time(ctx, t):
    """#10: the wall stops; its single return receipt flips to a question on b2."""
    dim = on(t, A(10), 0.7)
    _intro_wall(ctx, t, a=mix(0.8, 0.12, dim))
    enter = on(t, A(10), 0.75)
    flip = float(np.clip((t - at(10, 2) + 0.18) / 0.36, 0, 1))
    sx = max(0.025, abs(math.cos(math.pi * flip)))
    ctx.save(); ctx.translate(960, 550); ctx.scale(sx, 1)
    _slip(ctx, -145, -195, enter, w=290, h=390)
    if flip < 0.5:
        pts = [(float(x), float(y)) for x, y in zip(np.linspace(-110, 110, 45),
                                                   -25 - 35 * np.sin(np.linspace(0, 9, 45)))]
        gline(ctx, pts[:23], "red", 2.5, enter, glow=0.2)
        gline(ctx, pts[22:], "green", 2.5, enter, glow=0.2)
    else:
        glow_text(ctx, "？", 0, 70, 180, "ice", enter, align="center", glow=0.8)
    ctx.restore()
    glow_at(ctx, 960, 550, "ice", 2.8, 0.6 * kick(t, at(10, 2), 0.45))


# ======================================================================================
# #12–#17: one wall tile becomes a score, many stripes, then a ranked stock wall
# ======================================================================================
@_frame
def shot_macd(ctx, t):
    """#12: magnify a candle tile; b2 extracts its EMA gap and lights DIF."""
    push = sm((t - A(12)) / 1.2)
    collapse = sm((t - at(12, 2)) / 0.32)
    lift = sm((t - at(12, 2) - 0.32) / 0.6)
    hit = kick(t, at(12, 2), 0.75)
    _factor_stage(ctx, detail=1, tile_scale=mix(0.85, 1, push), tile_width=760 * 1100 / 660,
                  line_draw=on(t, A(12), 1.1), band=1, collapse=collapse, lift=lift)
    _card(ctx, "macd", {1: (on(t, at(12, 2), 0.1), "cyan"),
                         2: (0.35, "cyan"), 3: (0.35, "amber")})
    _, fast, slow, i = _tile_data()
    gap_y = mix(600 + (fast[i][1] + slow[i][1]) / 2, 400, lift)
    glow_at(ctx, 960, gap_y, "cyan", 1.8, 0.6 * hit)


@_frame
def shot_factor_score(ctx, t):
    """#13: return the bright bar to its tile; b4 reveals neighbouring score brightness."""
    drop = sm((t - A(13)) / 1.05)
    neighbours = on(t, at(13, 4), 0.38)
    hit = kick(t, at(13, 4), 0.55)
    pulse = float(np.clip((t - A(13)) / (4 * BEAT), 0, 1))
    length = 210 * (1 - 0.4 * math.sin(math.pi * pulse) ** 2)
    _factor_stage(ctx, detail=1, score=drop * length / 210, neighbours=neighbours,
                  wall_a=0.15 + 0.13 * neighbours, tile_width=mix(760 * 1100 / 660, 760, drop))
    _dif_bar(ctx, drop=drop, flare=0.25 * hit, length=length)
    _card(ctx, "macd", {1: (0.8, "cyan")})
    glow_text(ctx, "因子", 960, 905, 66, "cyan", on(t, A(13), 0.5), align="center", glow=0.55)
    glow_at(ctx, 1650, 600, "cyan", 2.2, 0.6 * hit)


@_frame
def shot_factor_matrix(ctx, t):
    """#14: one score splits into 2048 stripes; b2 opens the barcoded mural."""
    split = sm((t - A(14)) / (2 * BEAT))
    pull = sm((t - at(14, 2)) / (2.8 * BEAT))
    detail = 1 - pull
    _factor_stage(ctx, detail=detail, score=1, barcode=split, neighbours=1,
                  wall_a=mix(0.28, 0.8, pull), tile_scale=1 / (1 + 0.8 * pull))
    _dif_bar(ctx, drop=1, a=1 - split)
    _card(ctx, "macd", {1: (0.8 * (1 - split), "cyan")})
    glow_text(ctx, "几千", 960, 905, 66, "cyan", split, align="center", glow=0.6)
    glow_at(ctx, 960, 600, "cyan", 3.0, 0.65 * kick(t, at(14, 2), 0.5))


@_frame
def shot_weak_factor(ctx, t):
    """#15: magnify one stripe's weak signal; b4 freezes the noisy readout at 51%."""
    lift = on(t, A(15), 1.1)
    _factor_stage(ctx, barcode=1, score=1, wall_a=mix(0.8, 0.12, lift))
    gline(ctx, [(952, mix(592, 472, lift)), (952, mix(600, 728, lift))], "cyan", 2,
          1 - lift, glow=0.8)
    clock = (min(t, at(15, 4)) - A(15)) / BEAT
    _weak_stripe(ctx, clock, a=lift, spread=lift)
    frozen = on(t, at(15, 4), 0.1)
    _card(ctx, "factor", {1: (0.5, "cyan"), 2: (0.5 * frozen, "violet")})
    num(ctx, "51%", 960, 905, 88, "cyan", frozen, align="center", glow=0.8)
    gline(ctx, [(440, 490), (440, 750), (1480, 750), (1480, 490)], "ice", 1,
          0.35 * frozen, glow=0.2)
    glow_at(ctx, 960, 600, "ice", 2.5, 0.5 * kick(t, at(15, 4), 0.4))


@_frame
def shot_combine(ctx, t):
    """#16: identical stripes stack and fall; b4 twists the survivors into one thick line."""
    open_ = on(t, A(16), 0.85)
    merge = on(t, at(16, 4), 0.65)
    hit = kick(t, at(16, 4), 0.5)
    _factor_stage(ctx, barcode=1, score=1, wall_a=0.12)
    _weak_stripe(ctx, 4, a=1 - open_)
    gline(ctx, [(440, 490), (440, 750), (1480, 750), (1480, 490)], "ice", 1,
          0.35 * (1 - open_), glow=0.2)
    for i in range(18):
        group, duplicate = divmod(i, 3)
        y0 = 500 + group * 42 + duplicate * 7
        stack = on(t, at(16, 0.8 + group * 0.16), 0.6)
        fall = on(t, at(16, 2 + group * 0.12), 0.8) if duplicate else 0
        alpha = open_ * (1 - fall) * (1 - 0.7 * merge)
        pts = []
        for j in range(81):
            u = j / 80
            y = mix(y0, 500 + group * 42, stack) + 260 * fall
            y += 10 * math.sin(u * 19 + group) * (1 - merge)
            twist = 22 * math.sin(u * 14 + group * math.pi / 3) * merge * (1 - merge)
            pts.append((440 + u * 1040, mix(y, 600, merge) + twist))
        gline(ctx, pts, "cyan" if not duplicate else "ice", 1.2, alpha, glow=0.25)
    gline(ctx, [(440, 600), (1480, 600)], "cyan", 12, merge, glow=0.8 + hit)
    residual = [(440 + i * 13, 800 + 8 * math.sin(i * 1.9)) for i in range(81)]
    gline(ctx, residual, "violet", 1.3, 0.35 * merge, glow=0.2)
    _card(ctx, "factor", {1: (0.5 + 0.5 * merge, "cyan"), 2: (0.5, "violet")})
    glow_at(ctx, 960, 600, "cyan", 3.2, 0.65 * hit)


@_frame
def shot_three_factor(ctx, t):
    """#17: three formula strands merge; b4 reorders all 5000 stocks by score."""
    separate = on(t, A(17), 0.65)
    gather = on(t, at(17, 2.6), 0.7)
    reveal = on(t, at(17, 4), 0.45)
    sort = sm((t - at(17, 4)) / BEAT)
    _factor_stage(ctx, barcode=1, score=1, wall_a=0.12 * (1 - reveal))
    _rank_wall(ctx, sort, a=0.85 * reveal)
    gline(ctx, [(440, 600), (1480, 600)], "cyan", 12, 1 - separate, glow=0.8)
    lit = {}
    for i, (col, term) in enumerate((("cyan", 2), ("ice", 3), ("amber", 4))):
        active = on(t, at(17, i), 0.4)
        lit[term] = (active * 0.85, col)
        pts = []
        for j in range(81):
            u = j / 80
            y = 600 + (i - 1) * 65 * separate * (1 - gather)
            y += 7 * math.sin(u * 15 + i) * separate * (1 - gather)
            pts.append((440 + 1040 * u, y))
        gline(ctx, pts, col, mix(2.4, 8, gather), separate * (0.12 + 0.72 * active) * (1 - reveal), glow=0.6)
    lit[1] = (gather * 0.8, "cyan")
    residual = [(440 + i * 13, 800 + 8 * math.sin(i * 1.9)) for i in range(81)]
    gline(ctx, residual, "violet", 1.3, 0.35 + 0.15 * gather, glow=0.2)
    _card(ctx, "fama_french", lit)
    glow_text(ctx, "多因子", 960, 905, 66, "cyan", reveal, align="center", glow=0.8)
    glow_at(ctx, 960, 390, "cyan", 3.5, 0.75 * kick(t, at(17, 4), 0.55))


SHOTS = [
    (4, 6, shot_old_way),
    (7, 7, shot_rules),
    (8, 8, shot_pipeline),
    (9, 9, shot_stock_wall),
    (10, 10, shot_same_time),
    (12, 12, shot_macd),
    (13, 13, shot_factor_score),
    (14, 14, shot_factor_matrix),
    (15, 15, shot_weak_factor),
    (16, 16, shot_combine),
    (17, 17, shot_three_factor),
]
