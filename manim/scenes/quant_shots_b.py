"""Film-strip backtests (#20–#24) and one felt betting table (#26–#33).

Pure-time Cairo shots for quant_film.py; phone-sized subjects and a separate formula rail.
"""
from quant_kit import *  # noqa: F401,F403


# ======================================================================================
# Shared objects
# ======================================================================================
def _partial(pts, progress):
    """Cut a polyline, retaining an interpolated endpoint."""
    m = min(1.0, max(0.0, progress)) * (len(pts) - 1)
    i = int(m)
    out = list(pts[:i + 1])
    if i < len(pts) - 1 and m > i:
        out += lerp_pts([pts[i]], [pts[i + 1]], m - i)
    return out


def _authority(ctx, name, lit=None, a=1.0):
    """Reserve a full-size formula rail, with a compact Chinese credit inside the card."""
    width, cx = 600, 1510
    _, fh = formula_size(name, width)
    cy = 105 + (fh + 70) / 2
    formula_card(ctx, name, cx, cy, width, a=a, lit=lit or {}, info=False, en=False)
    who, year, _ = FORMULA_INFO[name]
    credit = f"{who} · {year}" if year else who
    text(ctx, credit, cx, cy + fh / 2 + 28, 30, "amber", a, bold=True, align="center")


def _ellipse(ctx, x, y, rx, ry):
    """Append an ellipse without leaving a transformed context."""
    ctx.save(); ctx.translate(x, y); ctx.scale(rx, ry)
    ctx.new_sub_path(); ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()


def _reel(ctx, x, y, r, angle=0.0, a=1.0):
    """A round, six-spoked film reel, also the coin's match-cut silhouette."""
    ctx.save(); ctx.translate(x, y); ctx.rotate(angle)
    ctx.arc(0, 0, r, 0, 2 * math.pi)
    fill(ctx, "bg2", a, keep=True); stroke(ctx, "ice", 3, 0.55 * a)
    ctx.arc(0, 0, r * 0.87, 0, 2 * math.pi); stroke(ctx, "amber", 6, 0.26 * a)
    for u in np.linspace(0, 2 * math.pi, 6, endpoint=False):
        ctx.arc(r * 0.52 * math.cos(u), r * 0.52 * math.sin(u), r * 0.22, 0, 2 * math.pi)
        fill(ctx, "bg", a, keep=True); stroke(ctx, "ink2", 1.5, 0.4 * a)
    ctx.arc(0, 0, r * 0.13, 0, 2 * math.pi)
    fill(ctx, "dim", a, keep=True); stroke(ctx, "ice", 2, 0.7 * a)
    ctx.restore()


# ======================================================================================
# The projector: 192 px frames, real transparent sprockets, a fixed cyan gate
# ======================================================================================
@lru_cache(None)
def _film_texture(seed):
    """Cache sixteen printed frames; the two perforated margins are transparent."""
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 3072, 220)
    c = cairo.Context(surf)
    c.rectangle(0, 0, 3072, 220); fill(c, "#302831", 0.97)
    for i in range(16):
        x = i * 192
        c.rectangle(x + 6, 35, 180, 150); fill(c, "#141c26", 1)
        candles(c, walk(seed + i, 9, 0.08, 0.9), x + 17, 48, 158, 124, a=0.7)
        gline(c, [(x, 34), (x, 186)], "amber", 2, 0.43, glow=0)
        gline(c, [(x + 6, 35), (x + 186, 35), (x + 186, 185), (x + 6, 185)],
              "ink2", 1, 0.25, glow=0, close=True)
    for x in range(10, 3072, 32):
        for y in (8, 194):
            rrect(c, x, y, 17, 18, 3)
            c.set_operator(cairo.OPERATOR_CLEAR); c.fill(); c.set_operator(cairo.OPERATOR_OVER)
    for y in (1, 30, 190, 219):
        gline(c, [(0, y), (3072, y)], "amber", 1, 0.35, glow=0)
    return surf


def _film_strip(ctx, y=395, offset=640, width=1040, height=240, a=1.0, seed=200, splice=None):
    """Keep large perforated frames in the left stage, clear of the formula rail."""
    if a <= 0.002:
        return
    scale = height / 220
    left = 620 - width / 2
    ctx.save(); ctx.rectangle(left, y - height / 2, width, height); ctx.clip()
    origin = 620 - 96 * scale - offset
    period = 3072 * scale
    origin -= math.ceil((origin - left) / period) * period
    while origin < left + width:
        paint_img(ctx, _film_texture(seed), origin, y - height / 2, a=a, scale=scale)
        origin += period
    if splice is not None:
        ctx.rectangle(splice, y - height / 2 + 32 * scale, 2300, 156 * scale)
        fill(ctx, "#24242c", a)
        ctx.rectangle(splice - 10, y - height / 2, 20, height)
        fill(ctx, "ice", 0.16 * a)
        gline(ctx, [(splice - 7, y - height / 2 + 3), (splice + 7, y + height / 2 - 3)],
              "amber", 2, 0.7 * a, glow=0)
    ctx.restore()


def _film_stage(ctx, y=395, offset=640, width=1040, height=240, a=1.0,
                clamp=1.0, gate=1.0, seed=200, splice=None):
    """One substantial projector strip, with its fixed read head at the stage centre."""
    reel_a = 0.28 * a * sm((width - 300) / 700)
    _reel(ctx, 145, y, height * 0.43, -offset / 145, reel_a)
    _reel(ctx, 1025, y, height * 0.43, -offset / 145, reel_a)
    _film_strip(ctx, y, offset, width, height, a, seed, splice)
    if gate * a <= 0.002:
        return
    half = height / 2
    glow_at(ctx, 620, y, "cyan", 1.65 * height / 220, 0.13 * a * gate)
    gline(ctx, [(620, y - half + 31), (620, y + half - 31)], "cyan", 3, 0.8 * a * gate)
    gap = 12 * (1 - clamp)
    for sign in (-1, 1):
        yy = y + sign * (half + 10 + gap)
        rrect(ctx, 585, yy - 7, 70, 14, 4)
        fill(ctx, "bg2", a * gate, keep=True); stroke(ctx, "cyan", 2.5, a * gate)
        gline(ctx, [(595, yy), (645, yy)], "cyan", 3, a * gate)


@lru_cache(None)
def _history_values(beautiful=False):
    """Keep one deterministic equity recording beneath the read head."""
    u = np.linspace(0, 1, 121)
    if beautiful:
        return 0.18 + 0.63 * u + 0.035 * np.sin(11 * math.pi * u) * np.sin(math.pi * u)
    v = equity(201, 121, 0.25, 0.8)
    return 0.12 + 0.76 * (v - v.min()) / (v.max() - v.min())


def _record_pts(progress=1.0, beautiful=False):
    """Use a 300 px recording area beneath the film, ending under the fixed read head."""
    values = _history_values(beautiful)
    u = np.linspace(0, 1, len(values))
    pts = [(620 - 500 * (progress - float(x)), 900 - 300 * float(v)) for x, v in zip(u, values)]
    return _partial(pts, progress)


@lru_cache(None)
def _wobble_widths():
    """Measure local return dispersion in screen pixels, rather than using a constant stroke."""
    ys = 300 * _history_values()
    return tuple(float(np.clip(12 + 2.1 * np.std(np.diff(ys[max(0, i - 6):min(len(ys), i + 7)])),
                               16, 43)) for i in range(len(ys)))


def _record(ctx, progress=1.0, beautiful=False, a=1.0, band=0.0, flash=0.0):
    """Fill a variable-width violet ribbon around the equity trace, with a beat-five flash."""
    pts = _record_pts(progress, beautiful)
    if band > 0:
        widths = _wobble_widths()
        upper = _partial([(x, y - widths[i]) for i, (x, y) in enumerate(pts)], band)
        lower = _partial([(x, y + widths[i]) for i, (x, y) in enumerate(pts)], band)
        path(ctx, upper + list(reversed(lower)), close=True)
        fill(ctx, "violet", a * (0.48 + 0.28 * flash))
        for edge in (upper, lower):
            gline(ctx, edge, "violet", 2.3, a * (0.8 + 0.2 * flash), glow=0.35 + flash)
    gline(ctx, pts, "amber", 4.8, a, glow=0.7)
    if progress > 0:
        head(ctx, pts, "amber", a, r=6)
        gline(ctx, [(620, 535), (620, pts[-1][1] - 12)], "cyan", 1, 0.15 * a, glow=0)


@lru_cache(None)
def _comparison():
    """Equal return, with period-return Sharpe ratios exactly 0.4 and 2.1."""
    noise = rng(220).normal(0, 1, 96)
    noise += 1.8 * np.sin(np.linspace(0, 6 * math.pi, 96))
    noise = (noise - noise.mean()) / noise.std()
    paths = tuple(np.r_[0.0, np.cumsum(0.02 + 0.02 / score * noise)] for score in (0.4, 2.1))
    return paths, min(float(p.min()) for p in paths) - 0.1, max(float(p.max()) for p in paths) + 0.1


def _comparison_strip(ctx, index, y, height, a=1.0, align=1.0, bright=1.0,
                      print_a=0.46, path_a=1.0, gate=0.55):
    """Print one of the equal-endpoint records on the same film stock."""
    paths, lo, hi = _comparison()
    _film_stage(ctx, y=y, height=height, a=print_a * a, gate=gate, seed=200 + index * 20)
    end_x = mix(950 if index == 0 else 1030, 990, align)
    pts = curve_pts(paths[index], 175, y - height * 0.30, end_x - 175, height * 0.60, lo=lo, hi=hi)
    col = "violet" if index == 0 else "cyan"
    gline(ctx, pts, col, 3.3, a * bright * path_a, glow=0.7)
    head(ctx, pts, col, a * bright * path_a, r=5)
    gline(ctx, [(end_x, y - height * 0.37), (end_x, y + height * 0.37)],
          col, 1.5, a * path_a * (0.25 + 0.5 * align), glow=0)


@lru_cache(None)
def _discard_pile():
    """Four ordered fans of twenty-five distinct grey, perforated equity-film ribbons."""
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, 920)
    c = cairo.Context(surf)
    c.rectangle(80, 220, 1060, 680); c.clip()
    r = rng(230)
    for stack in range(4):
        for j in range(25):
            x = 103 + stack * 250 + j * 1.25
            y = 330 + j * 21
            c.save(); c.translate(x, y); c.rotate((stack - 1.5) * 0.012 + (j - 12) * 0.0009)
            rrect(c, 0, 0, 218, 18, 2)
            fill(c, "#101923", 0.98, keep=True); stroke(c, "ink2", 1, 0.62)
            for xx in range(6, 214, 13):
                for yy in (2, 14):
                    c.rectangle(xx, yy, 5, 2); fill(c, "bg", 1)
            for xx in (72, 145):
                gline(c, [(xx, 5), (xx, 13)], "ink2", 0.8, 0.45, glow=0)
            values = np.cumsum(r.normal(0.1, 0.75, 28))
            pts = curve_pts(values, 7, 5, 204, 8)
            gline(c, pts, "ink", 1.35, 0.52, glow=0, core=False)
            c.restore()
    return surf


def _shutter(ctx, opening, a=1.0):
    """Slide two blades clear of the separated ribbon fans without a fog layer."""
    paint_img(ctx, _discard_pile(), 0, 0, a=a)
    for sign in (-1, 1):
        x = 80 - 540 * opening if sign < 0 else 610 + 540 * opening
        ctx.save(); ctx.rectangle(80, 220, 1060, 680); ctx.clip()
        ctx.rectangle(x, 220, 530, 680); fill(ctx, "#080e18", a)
        for y in range(235, 900, 22):
            gline(ctx, [(x + 5, y), (x + 525, y)], "ink2", 1, 0.07 * a, glow=0)
        edge_x = x + 530 if sign < 0 else x
        gline(ctx, [(edge_x, 220), (edge_x, 900)], "ice", 2, 0.22 * a, glow=0)
        ctx.restore()


def shot_backtest(ctx, t):
    """#20: centre the projector and equity recording; clamp forward playback on beat three."""
    b = bt(t, 20)
    width = mix(192 * 160 / 220, 1040, sm(b / 1.8))
    height = mix(160, 240, sm(b / 1.5))
    offset = -480 * sm(b / 2) if b < 2 else -480 + 280 * (b - 2)
    clamp = on(t, at(20, 2), 0.12)
    ctx.save(); ctx.translate(340, 0)
    _film_stage(ctx, offset=offset, width=width, height=height, clamp=clamp,
                gate=sm(b / 1.5))
    _record(ctx, min(1.0, max(0.0, (b - 2) / 4)), a=clamp)
    glow_at(ctx, 620, 395, "cyan", 1.8, 0.55 * kick(t, at(20, 2)))
    ctx.restore()


def shot_sharpe(ctx, t):
    """#21: ease the stopped projector left on beat one; light return and wobble on beat five."""
    b = bt(t, 21)
    ctx.save(); ctx.translate(340 * (1 - sm(b)), 0)
    _film_stage(ctx)
    _record(ctx, band=sm(b / 3.8), flash=kick(t, at(21, 4), 0.9))
    ctx.restore()
    lit = on(t, at(21, 4), 0.12)
    _authority(ctx, "sharpe", {1: (lit, "amber"), 2: (lit, "violet")}, a=sm(b / 0.7))
    glow_at(ctx, 1510, 270, "violet", 1.6, 0.35 * kick(t, at(21, 4)))


def shot_stability(ctx, t):
    """#22: split the same stock into two records; align ends and favour calm on beat three."""
    b = bt(t, 22)
    split = sm(b / 1.7)
    aligned = on(t, at(22, 2), 0.14)
    _authority(ctx, "sharpe", {1: (0.5, "amber"), 2: (aligned, "violet")})
    _record(ctx, a=1 - split, band=1)
    for i, target in enumerate((350, 660)):
        y = mix(395, target, split)
        a = 1.0 if i == 0 else split
        bright = 0.65 if i == 0 else 0.55 + 0.45 * aligned
        _comparison_strip(ctx, i, y, 240, a, aligned, bright,
                          print_a=mix(1, 0.46, split), path_a=split, gate=mix(1, 0.55, split))
        num(ctx, "0.4" if i == 0 else "2.1", 1065, y + 24, 72,
            "violet" if i == 0 else "cyan", aligned, align="center", glow=0.5)
    glow_at(ctx, 990, 620, "cyan", 1.5, 0.48 * kick(t, at(22, 2)))


def shot_selection(ctx, t):
    """#23: keep the beautiful strip; beat five pulls its shutter off one hundred rejects."""
    b = bt(t, 23)
    merge = sm(b / 1.6)
    opening = on(t, at(23, 4), 0.28)
    _shutter(ctx, opening)
    _comparison_strip(ctx, 0, 350, 240, a=1 - merge, bright=0.65)
    _comparison_strip(ctx, 1, mix(660, 395, merge), 240,
                      print_a=mix(0.46, 1, merge), path_a=1 - merge, gate=mix(0.55, 1, merge))
    _record(ctx, beautiful=True, a=merge)
    _authority(ctx, "sharpe")
    num(ctx, "100", 1510, 774, 140, "amber", opening, align="center",
        glow=0.6 + kick(t, at(23, 4)))


def shot_overfit(ctx, t):
    """#24: run across a splice into blank future film; snap green on beat five, then wind up."""
    b = bt(t, 24)
    wind = sm((b - 6) / 2)
    ctx.save(); ctx.push_group()
    paint_img(ctx, _discard_pile(), 0, 0, a=1 - sm(b / 1.2))
    splice = 1090 - 117.5 * min(max(b, 0), 4) - 150 * max(b - 4, 0)
    _film_stage(ctx, offset=640 + 117.5 * b, seed=220, splice=splice)
    _record(ctx, beautiful=True)
    if b >= 4:
        start = _record_pts(1, True)[-1]
        fall = [start, (650, start[1] + 165), (710, 882), (805, 844), (930, 897), (1090, 861)]
        pts = _partial(fall, min(1.0, 0.36 + 0.64 * sm((b - 4) / 1.5)))
        gline(ctx, pts, "green", 4.5, glow=0.8)
        head(ctx, pts, "green", r=7)
        glow_at(ctx, 650, start[1] + 165, "green", 1.4, 0.45 * kick(t, at(24, 4)))
    ctx.pop_group_to_source(); ctx.paint_with_alpha(1 - wind); ctx.restore()
    _reel(ctx, 960, 550, 108, angle=b * 0.65, a=wind)
    _authority(ctx, "sharpe")


# ======================================================================================
# The felt table: all coins, chips, cards and windows inhabit this same slab
# ======================================================================================
@lru_cache(None)
def _felt():
    """Cache radial teal light, fine felt grain and directional fibres."""
    yy, xx = np.mgrid[0:600, 0:1560].astype(float)
    ry = np.where(yy < 300, 170, 300)
    radius = ((xx - 780) / 780) ** 2 + ((yy - 300) / ry) ** 2
    light = np.exp(-(((xx - 700) / 650) ** 2 + ((yy - 320) / 280) ** 2) * 1.5)
    grain = gaussian_filter(rng(260).normal(0, 1, (600, 1560)), (0.45, 1.1))
    fibre = rng(261).normal(0, 1, (600, 1560))
    rgb = mix(hx("#071b22"), hx("#164449"), light[..., None])
    rgb += grain[..., None] * 0.012 + fibre[..., None] * 0.0025
    return surf_from(np.clip(rgb, 0, 1), np.clip((1 - radius) * 120, 0, 1))


@lru_cache(None)
def _table_rim(inset=0, drop=0):
    """Keep the padded far rail below the formula header without moving the bankroll."""
    return tuple((960 + (780 - inset) * math.cos(u),
                  570 + drop + (170 - inset if math.sin(u) < 0 else 300 - inset) * math.sin(u))
                 for u in np.linspace(0, 2 * math.pi, 161))


def _table(ctx):
    """One radial-lit felt table, padded leather rail and printed central betting circle."""
    path(ctx, _table_rim(drop=14), close=True); fill(ctx, "#030b12", 0.98)
    gline(ctx, _table_rim(drop=9), "#030a10", 30, 1, glow=0, close=True, core=False)
    paint_img(ctx, _felt(), 180, 270)
    gline(ctx, _table_rim(), "#172b35", 25, 1, glow=0, close=True, core=False)
    gline(ctx, _table_rim(inset=3, drop=-3), "ice", 2.5, 0.3, glow=0.3, close=True, core=False)
    gline(ctx, _table_rim(inset=15), "gold", 1.4, 0.24, glow=0, close=True, core=False)
    _ellipse(ctx, 960, 550, 144, 71); stroke(ctx, "gold", 2, 0.3)
    _ellipse(ctx, 960, 550, 137, 66); stroke(ctx, "gold", 0.8, 0.15)


@lru_cache(None)
def _disc_art(coin=False, shadow=True):
    """Cache a gold disc: shaded side band, cream inserts, inlay, specular and contact shadow."""
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 272, 192)
    c = cairo.Context(surf); c.translate(136, 78)
    if shadow:
        for rx, ry, alpha in ((112, 48, 0.09), (105, 42, 0.17), (98, 35, 0.22)):
            _ellipse(c, 1, 29, rx, ry); fill(c, "#020a0c", alpha)
    front = [(100 * math.cos(u), 42 * math.sin(u)) for u in np.linspace(0, math.pi, 41)]
    path(c, front + [(x, y + 20) for x, y in reversed(front)], close=True)
    side = cairo.LinearGradient(0, 0, 0, 63)
    side.add_color_stop_rgb(0, *hx("#a66b26"))
    side.add_color_stop_rgb(0.5, *hx("#89521b"))
    side.add_color_stop_rgb(1, *hx("#4d2b10"))
    c.set_source(side); c.fill()
    inserts = tuple(math.pi / 8 + k * math.pi / 4 for k in range(8))
    for u in inserts:
        if math.sin(u) > 0 and not coin:
            p = [(100 * math.cos(v), 42 * math.sin(v)) for v in (u - 0.075, u + 0.075)]
            path(c, p + [(p[1][0], p[1][1] + 20), (p[0][0], p[0][1] + 20)], close=True)
            fill(c, "#c8b58d", 0.85)
    gline(c, [(x, y + 20) for x, y in front], "#e2b66c", 1.4, 0.7, glow=0)
    _ellipse(c, 0, 0, 100, 42)
    face = cairo.RadialGradient(-32, -19, 2, 4, 9, 125)
    face.add_color_stop_rgb(0, *hx("#ffe3a0"))
    face.add_color_stop_rgb(0.48, *hx("#dfa647"))
    face.add_color_stop_rgb(1, *hx("#9b6224"))
    c.set_source(face); c.fill_preserve(); stroke(c, "#f6dda2", 2, 0.85)
    if coin:
        for u in np.linspace(0, 2 * math.pi, 36, endpoint=False):
            gline(c, [(91 * math.cos(u), 38.2 * math.sin(u)), (98 * math.cos(u), 41.1 * math.sin(u))],
                  "#fff0c9", 1.3, 0.65, glow=0)
    else:
        for u in inserts:
            path(c, [(rr * math.cos(v), 0.42 * rr * math.sin(v))
                     for rr, v in ((98, u - 0.075), (98, u + 0.075), (80, u + 0.095), (80, u - 0.095))],
                 close=True)
            fill(c, "#f4e9cd", 0.98)
    for r, alpha in ((75, 0.8), (70, 0.35)):
        _ellipse(c, 0, 0, r, 0.42 * r); stroke(c, "#fff0c7", 1.8, alpha)
    _ellipse(c, 0, 0, 54, 22.7)
    inlay = cairo.RadialGradient(-15, -8, 1, 7, 9, 64)
    inlay.add_color_stop_rgb(0, *hx("#efc675"))
    inlay.add_color_stop_rgb(1, *hx("#a36825"))
    c.set_source(inlay); c.fill_preserve(); stroke(c, "#fff0cc", 1.6, 0.65)
    _ellipse(c, 0, 0, 43, 18); stroke(c, "#82531c", 1, 0.4)
    c.save(); _ellipse(c, 0, 0, 99, 41.5); c.clip()
    shine = cairo.LinearGradient(-75, -35, 65, 28)
    for stop, alpha in ((0, 0), (0.25, 0.26), (0.42, 0.03), (0.58, 0.16), (1, 0)):
        shine.add_color_stop_rgba(stop, 1, 0.97, 0.87, alpha)
    c.rectangle(-102, -44, 204, 88); c.set_source(shine); c.fill(); c.restore()
    gline(c, [(99 * math.cos(u), 41.6 * math.sin(u)) for u in np.linspace(3.5, 5.7, 35)],
          "#fff3cf", 2.8, 0.92, glow=0, core=False)
    return surf


def _chip(ctx, x, y, r=29, a=1.0, sx=1.0, ry=0.42, shadow=True, coin=False):
    """All money has one amber body; motion, never face colour, encodes the result."""
    if a <= 0.002 or r <= 0.1:
        return
    ctx.save(); ctx.translate(x, y)
    ctx.scale(r / 100 * max(0.025, sx), r / 100 * ry / 0.42)
    paint_img(ctx, _disc_art(coin, shadow), -136, -78, a=a)
    ctx.restore()


def _rim_arc(ctx, x, y, r, fraction, a=1.0):
    """Keep the probability arc on the perspective coin rim."""
    pts = [(x + r * math.cos(u), y + 0.42 * r * math.sin(u))
           for u in np.linspace(-math.pi / 2, -math.pi / 2 + 2 * math.pi * fraction, 70)]
    gline(ctx, pts, "red", 5, a, glow=0.7)
    if pts:
        head(ctx, pts, "red", a, r=4)


def _coin(ctx, phase=0.0, a=1.0, arc=False, blur=0.0, r=92, ry=0.42):
    """A reeded gold coin flicks in its printed circle, with a restrained spin streak."""
    if blur > 0:
        for off in (-11, 0, 11):
            gline(ctx, [(868, 550 + off), (1052, 550 - off)], "gold", 2.5, 0.26 * a * blur, glow=0.4)
    sx = max(0.04, abs(math.cos(math.pi * phase)))
    _chip(ctx, 960, 550, r, a=a, sx=sx, ry=ry, coin=True)
    if arc:
        ctx.save(); ctx.translate(960, 550); ctx.scale(sx, 1); ctx.translate(-960, -550)
        _rim_arc(ctx, 960, 550, r + 14, 0.51, a)
        ctx.restore()


OWN_X, OWN_BASE, OWN_R = 760, 775, 52
START_CHIPS, TRAY_CHIPS = 8, 36
ZERO_Y = OWN_BASE - (START_CHIPS - 1) * OWN_R * 0.2
TEN_ROUNDS = tuple((i * 0.5, outcome) for i, outcome in enumerate((1, -1, -1, 1, -1, -1, 1, -1, 1, -1)))


@lru_cache(None)
def _stack_jitter(seed):
    """Cache slight horizontal registration errors, not changing per-frame randomness."""
    return tuple(float(x) for x in rng(seed).uniform(-1.7, 1.7, 64))


def _stack_top(count, base=OWN_BASE, r=OWN_R):
    """Return the top face centre at the same thickness used to draw the stack."""
    return base - max(0, count - 1) * 0.2 * r


def _stack(ctx, x, base, count, r=OWN_R, a=1.0, seed=270):
    """Build a tower from thick chips, interpolating only a partial top chip."""
    count = max(0.0, float(count))
    whole = int(math.floor(count))
    for i in range(int(math.ceil(count))):
        partial = count - whole if i == whole else 1.0
        y = base - min(i, max(0, count - 1)) * 0.2 * r
        _chip(ctx, x + _stack_jitter(seed)[i % 64], y, r, a=a * sm(partial), shadow=i == 0)


def _tray_slot(total):
    """Locate the last occupied chip in the other side's six-column money tray."""
    i = max(0, int(math.ceil(total)) - 1)
    column, level = (5, i - 30) if i >= 30 else (i // 6, i % 6)
    return 850 + column * 55, 450 - level * 27 * 0.2


def _tray(ctx, total=TRAY_CHIPS):
    """The far-rim reserve is real money, dimmer and smaller in perspective."""
    rrect(ctx, 808, 410, 363, 65, 16)
    fill(ctx, "#09151c", 0.92, keep=True); stroke(ctx, "ink2", 1.5, 0.32)
    gline(ctx, [(824, 467), (1156, 467)], "gold", 1.5, 0.24, glow=0)
    for i in range(6):
        count = max(0, total - i * 6) if i == 5 else min(6, max(0, total - i * 6))
        _stack(ctx, 850 + 55 * i, 450, count, r=27, a=0.58, seed=280 + i)


def _zero_mark(ctx, count, col=None, lit=0.0):
    """A dashed starting height and a coloured gap show the balance without a graph."""
    for x0, x1 in ((630, 700), (822, 886)):
        gline(ctx, [(x0, ZERO_Y), (x1, ZERO_Y)], "ink", 1.8, 0.56, glow=0, dash=[7, 7])
    if col is None or lit <= 0:
        return
    top = _stack_top(count)
    if abs(top - ZERO_Y) < 1:
        return
    ctx.rectangle(829, min(top, ZERO_Y), 43, abs(top - ZERO_Y)); fill(ctx, col, 0.22 * lit)
    gline(ctx, [(826, ZERO_Y), (861, ZERO_Y), (861, top), (826, top)], col, 3.5, lit, glow=0.8)
    glow_at(ctx, 852, (top + ZERO_Y) / 2, col, 0.8, 0.25 * lit)


def _game_money(ctx, count=START_CHIPS, reserve=TRAY_CHIPS, col=None, lit=0.0):
    """Your near-side tower, warm ownership light and fixed starting-height marker."""
    _tray(ctx, reserve)
    glow_at(ctx, OWN_X, OWN_BASE + 10, "amber", 1.3, 0.25)
    _ellipse(ctx, OWN_X, OWN_BASE + 15, 82, 27); stroke(ctx, "amber", 2, 0.4)
    _stack(ctx, OWN_X, OWN_BASE, count)
    _zero_mark(ctx, count, col, lit)


def _transfer_point(start, end, u, lane=0.0):
    """Curve a sliding chip around the coin so both money endpoints stay legible."""
    return (mix(start[0], end[0], u) + (95 + lane) * math.sin(math.pi * u),
            mix(start[1], end[1], u) + 10 * math.sin(math.pi * u))


def _round_state(b, events, travel=0.2, delay=0.06):
    """Conserve the two balances and the one in-flight chip for every timed round."""
    own, reserve, phase = START_CHIPS, TRAY_CHIPS, 0.0
    moving, pops = [], []
    for beat, outcome in events:
        dt = (b - beat) * BEAT
        if dt < 0:
            continue
        phase = sm(dt / 0.14)
        if outcome > 0:
            start = _tray_slot(reserve)
            end = (OWN_X, _stack_top(own + 1))
            if dt >= delay:
                reserve -= 1
            if dt >= delay + travel:
                own += 1
        else:
            start = (OWN_X, _stack_top(own))
            end = _tray_slot(reserve + 1)
            if dt >= delay:
                own -= 1
            if dt >= delay + travel:
                reserve += 1
        if delay <= dt < delay + travel:
            u = sm((dt - delay) / travel)
            radius = mix(27, OWN_R, u) if outcome > 0 else mix(OWN_R, 27, u)
            moving.append((_transfer_point(start, end, u), radius))
        age = dt - delay - travel
        if 0 <= age < 0.36:
            pops.append((outcome, age, _stack_top(own)))
    return own, reserve, phase, moving, pops


def _play_rounds(ctx, b, events, travel=0.2, delay=0.06, col=None, lit=0.0, arc=False):
    """Flick the coin, transfer one gold unit, and briefly print its signed cash change."""
    own, reserve, phase, moving, pops = _round_state(b, events, travel, delay)
    _game_money(ctx, own, reserve, col, lit)
    _coin(ctx, phase, arc=arc)
    for (x, y), radius in moving:
        _chip(ctx, x, y, radius)
    for outcome, age, y in pops:
        alpha = sm(age / 0.04) * (1 - sm((age - 0.18) / 0.18))
        text(ctx, "+1" if outcome > 0 else "−1", OWN_X + 105, y - 20 - 26 * age / 0.36,
             47, "red" if outcome > 0 else "green", alpha, font=DIGITS, bold=True, align="center")


def _million_level(b):
    """Compress the million-round scale into a tower, retaining luck after the reveal."""
    if b < 2:
        u = sm(b / 2)
        return 6 + 15 * u + (0.8 * math.sin(22 * b) + 0.45 * math.sin(39 * b)) * math.sin(math.pi * u)
    d = b - 2
    return 21 + 0.8 * math.sin(7 * d) + 0.45 * math.sin(17 * d)


def _traffic(ctx, b, count, a=1.0):
    """Two directional money ribbons accelerate over the felt, never showering coins."""
    speed = mix(1.2, 4.5, sm(b / 2))
    far = _tray_slot(TRAY_CHIPS - 12)
    near = (OWN_X, _stack_top(count))
    for incoming in (True, False):
        start, end = (far, near) if incoming else (near, far)
        lane = -25 if incoming else 38
        points = [_transfer_point(start, end, u, lane) for u in np.linspace(0, 1, 40)]
        gline(ctx, points, "amber", 8, 0.12 * a, glow=0.3, core=False)
        for i in range(9):
            u = (b * BEAT * speed + i / 9) % 1
            x, y = _transfer_point(start, end, u, lane)
            r = mix(17, 31, u) if incoming else mix(31, 17, u)
            _chip(ctx, x, y, r, a=0.7 * a, shadow=False)


def _cost_origin():
    """Start costs at the preceding million-round frame, not at a new diagram."""
    return _million_level((A(29) - A(28)) / BEAT)


def _cost_level(b):
    """Remove the gross margin, then at most two chips more, leaving an upright bankroll."""
    start = _cost_origin()
    target = START_CHIPS - min(2, (start - START_CHIPS) / 2)
    return mix(start, target, sm(b / 4))


def _scraper(ctx, x, y=659, depth=0.0, a=1.0, length=156, vertical=False):
    """A bright metal bevel cuts the tower, or makes the next assumption's clean sweep."""
    ctx.save(); ctx.push_group(); ctx.save(); ctx.translate(x, y)
    ctx.rotate((math.pi / 2 if vertical else 0) - 0.08 - 0.08 * depth)
    rrect(ctx, -length / 2, -19, length, 28, 4)
    metal = cairo.LinearGradient(0, -19, 0, 9)
    metal.add_color_stop_rgb(0, *hx("#5c7483"))
    metal.add_color_stop_rgb(0.45, *hx("#e4f3fc"))
    metal.add_color_stop_rgb(0.66, *hx("#91a8b7"))
    metal.add_color_stop_rgb(1, *hx("#f4fcff"))
    ctx.set_source(metal); ctx.fill_preserve(); stroke(ctx, "ice", 1.4, 1)
    gline(ctx, [(-length / 2 + 2, 8), (length / 2 - 2, 8)], "white", 3, 1, glow=0.55)
    rrect(ctx, -15, -72, 30, 57, 7); fill(ctx, "#122430", 1, keep=True); stroke(ctx, "ink2", 1.3, 0.6)
    ctx.restore(); ctx.pop_group_to_source(); ctx.paint_with_alpha(a); ctx.restore()


def _bankroll(ctx, push=0.0, a=1.0, deal=1.0, selected=True):
    """Exactly fifty equal gold chips; only one can leave the five-by-ten bankroll."""
    for row in range(5):
        for col in range(10):
            if row == 0 and col == 4 and not selected:
                continue
            x, y = 600 + 80 * col, 500 + 54 * row
            y = mix(887, y, deal)
            if row == 0 and col == 4:
                x, y = mix(x, 960, push), mix(y, 410, push)
            _chip(ctx, x, y, 25, a=a)


CARD_RANKS = (2, 10, 9, 10, 10)


def _playing_card(ctx, i, x, y, reveal=1.0, chart=0.0, window=0.0, sx=1.0, a=1.0):
    """Poker faces contain only rank and pips; the same paper becomes a quote window."""
    face = reveal >= 0.5
    flip = max(0.035, abs(1 - 2 * reveal)) * max(0.035, sx)
    w, h = mix(112, 210, window), mix(172, 202, window)
    ctx.save(); ctx.translate(x, y); ctx.scale(flip, 1)
    rrect(ctx, -w / 2, -h / 2, w, h, 9)
    fill(ctx, "#bac5ce" if face and window < 0.5 else "bg2", 0.94 * a, keep=True)
    stroke(ctx, "cyan" if window > 0.5 or not face else "ink", 2, 0.7 * a)
    if not face:
        for d in (-42, -14, 14, 42):
            gline(ctx, [(-40, d - 14), (40, d + 14)], "cyan", 1, 0.32 * a, glow=0)
    else:
        ink = a * (1 - chart)
        text(ctx, str(CARD_RANKS[i]), -w / 2 + 10, -h / 2 + 28, 24, "bg", ink, font=DIGITS, bold=True)
        rank = CARD_RANKS[i]
        pips = [(0, -20), (0, 45)] if rank == 2 else [
            (xx, yy) for xx in (-24, 24) for yy in (-34, -3, 28, 59)]
        if rank == 9:
            pips += [(0, 13)]
        elif rank == 10:
            pips += [(0, -18), (0, 44)]
        for xx, yy in pips:
            path(ctx, [(xx, yy - 8), (xx + 5, yy), (xx, yy + 8), (xx - 5, yy)], close=True)
            fill(ctx, "red" if i % 2 == 0 else "bg", ink)
        if chart > 0:
            rrect(ctx, -w / 2 + 7, -h / 2 + 7, w - 14, h - 14, 4); fill(ctx, "bg2", chart * a)
            candles(ctx, walk(320 + i, 12, 0.02, 0.8), -w / 2 + 14, -h / 2 + 20,
                    w - 28, h - 40, a=chart * a)
        if window > 0:
            gline(ctx, [(-w / 2 + 10, -h / 2 + 12), (w / 2 - 10, -h / 2 + 12)],
                  "cyan", 2, window * a, glow=0.25)
    ctx.restore()


def _quote_row(ctx, a=1.0):
    """The five former playing cards, now unlabelled market windows."""
    for i in range(5):
        _playing_card(ctx, i, 450 + 255 * i, 590, chart=1, window=1, a=a)


# ======================================================================================
# Edge and the realised sequence
# ======================================================================================
def shot_coin_edge(ctx, t):
    """#26: land at 51 percent; beats four and five teach one unit in, then one unit out."""
    b = bt(t, 26)
    _table(ctx)
    _authority(ctx, "edge", {1: (on(t, at(26, 2), 0.15), "red"), 2: (0.35, "violet")})
    if b < 2:
        _game_money(ctx)
        morph = sm(b / 0.8)
        phase = max(0, b) * 2.5
        ctx.save(); ctx.translate(960, 550); ctx.scale(max(0.04, abs(math.cos(math.pi * phase))), 1)
        ctx.translate(-960, -550)
        _reel(ctx, 960, 550, 108, angle=5.2 + phase * math.pi, a=1 - morph)
        ctx.restore()
        _coin(ctx, phase, a=morph, r=mix(108, 92, morph), ry=mix(1.0, 0.42, morph))
    else:
        _play_rounds(ctx, b, ((3, 1), (4, -1)), travel=0.38, delay=0.1, arc=True)
    num(ctx, "51%", 960, 837, 104, "red", on(t, at(26, 2), 0.12), align="center")
    glow_at(ctx, 960, 550, "gold", 2.1, 0.5 * kick(t, at(26, 2)))


def shot_ten_flips(ctx, t):
    """#27: ten half-beat transfers, four wins and six losses, end two chips below the start."""
    b = bt(t, 27)
    _table(ctx)
    _authority(ctx, "edge", {1: (0.4, "red"), 2: (on(t, at(27, 4), 0.12), "violet")})
    _play_rounds(ctx, b, TEN_ROUNDS, col="green", lit=on(t, at(27, 4), 0.12))
    num(ctx, "−2 元", 960, 837, 96, "green", on(t, at(27, 4), 0.15), align="center")
    glow_at(ctx, 852, ZERO_Y + OWN_R * 0.2, "green", 1.2, 0.35 * kick(t, at(27, 4)))


def shot_million(ctx, t):
    """#28: accelerate the same cash transfers; beat three reveals a positive, still trembling margin."""
    b = bt(t, 28)
    _table(ctx)
    _authority(ctx, "edge", {1: (on(t, at(28, 2), 0.12), "red"), 2: (0.7, "violet")})
    level = _million_level(b)
    lit = on(t, at(28, 2), 0.12) if level >= START_CHIPS else 1 - sm(b / 0.5)
    _game_money(ctx, level, START_CHIPS + TRAY_CHIPS - level,
                "red" if level >= START_CHIPS else "green", lit)
    rush = sm(b / 0.4) * (1 - sm((b - 2) / 0.45))
    _coin(ctx, phase=b * mix(6, 10, sm(b / 2)) if b < 2 else 0, blur=rush)
    _traffic(ctx, b, level, a=rush)
    if b < 2:
        count = int(round(mix(10, 1000000, sm(b / 1.85))))
        text(ctx, f"{count:,}", 960, 837, 88, "ink", font=DIGITS, bold=True, align="center")
    else:
        num(ctx, "+2 万", 960, 837, 96, "red", align="center", glow=0.6)
    glow_at(ctx, 852, (_stack_top(level) + ZERO_Y) / 2, "red", 1.2, 0.3 * kick(t, at(28, 2)))


def shot_costs(ctx, t):
    """#29: shear the actual gold tower past its starting height; beat five exposes the negative gap."""
    b = bt(t, 29)
    _table(ctx)
    _authority(ctx, "edge", {1: (0.35, "red"), 2: (0.55, "violet")})
    level = _cost_level(b)
    deep = on(t, at(29, 4), 0.12)
    _game_money(ctx, level, START_CHIPS + TRAY_CHIPS - _cost_origin(),
                "red" if level > START_CHIPS else "green", 0.7 if level > START_CHIPS else deep)
    _coin(ctx)
    exit_k = sm((b - 4) / 0.9)
    x = mix(645, 835, sm(b / 4)) + 1275 * exit_k
    blade_a = 1 - sm((b - 4.1) / 0.7)
    _scraper(ctx, x, y=_stack_top(level), depth=sm(b / 4), a=blade_a)
    num(ctx, "−1 万", 1210, 860, 96, "green", deep, align="center")
    glow_at(ctx, 852, (_stack_top(level) + ZERO_Y) / 2, "green", 1.2, 0.4 * kick(t, at(29, 4)))


# ======================================================================================
# New, pre-cost bankroll assumption; cards; independent breadth
# ======================================================================================
def shot_kelly(ctx, t):
    """#30: visibly clear the cost example; deal fifty equal chips, push exactly one on beat five."""
    b = bt(t, 30)
    _table(ctx)
    sweep = sm((b - 0.35) / 0.85)
    edge = mix(170, 1760, sweep)
    ctx.save(); ctx.rectangle(edge, 395, 1800, 430); ctx.clip()
    _game_money(ctx, _cost_level(6), START_CHIPS + TRAY_CHIPS - _cost_origin(), "green", 1)
    _coin(ctx)
    ctx.restore()
    if b < 0.35:
        exit_k = sm(b / 0.35)
        _scraper(ctx, mix(1045, 1810, exit_k), y=_stack_top(_cost_level(6)), depth=1, a=1 - exit_k)
    else:
        _scraper(ctx, edge, y=610, depth=1, a=1 - sm((b - 1.1) / 0.3), length=470, vertical=True)
    deal = sm((b - 1.3) / 1.5)
    pushed = on(t, at(30, 4), 0.2)
    _authority(ctx, "kelly", {1: (deal, "cyan"), 2: (deal, "cyan"),
                            3: (deal, "gold"), 4: (pushed, "gold")})
    ctx.save(); ctx.rectangle(230, 305, 1460, 550); ctx.clip()
    _bankroll(ctx, pushed, a=deal, deal=deal)
    ctx.restore()
    num(ctx, "2%", 960, 837, 104, "gold", pushed, align="center")
    glow_at(ctx, 960, 410, "gold", 1.3, 0.5 * kick(t, at(30, 4)))


def shot_thorp(ctx, t):
    """#31: pips, not labels; accumulating high cards move the selected chip farther forward."""
    b = bt(t, 31)
    _table(ctx)
    _authority(ctx, "kelly", {1: (0.7, "cyan"), 4: (0.65, "gold")})
    _bankroll(ctx, push=1, a=mix(1.0, 0.14, sm(b / 1.1)), selected=False)
    high = 0.0
    for i, rank in enumerate(CARD_RANKS):
        reveal = sm((b - i) / 0.4)
        fan = sm(b / 1.1)
        x = mix(1900 + 7 * i, mix(730, 730 + 115 * i, fan), sm(b / 0.9))
        _playing_card(ctx, i, x, 590, reveal=reveal, a=sm(b / 0.25))
        if rank >= 9:
            high += sm((reveal - 0.5) * 2)
    _chip(ctx, 960, 410 + 15 * high, 25)
    glow_text(ctx, "索普", 370, 235, 72, "gold", a=sm(b / 0.6), align="center", glow=0.4)
    glow_at(ctx, 960, 470, "gold", 1.2, 0.4 * kick(t, at(31, 4)))


def shot_wall_street(ctx, t):
    """#32: card pips become candle prints; beat three flips the entire deck into quotes."""
    b = bt(t, 32)
    _table(ctx)
    _authority(ctx, "kelly", {1: (0.55, "cyan"), 4: (on(t, at(32, 2), 0.12), "cyan")})
    _bankroll(ctx, push=1, a=0.14 * (1 - sm(b / 1.6)), selected=False)
    _chip(ctx, 960, 470, 25, a=1 - sm(b / 1.6))
    chart = sm(b / 1.6)
    flip = on(t, at(32, 2), 0.32)
    for i in range(5):
        x = mix(730 + 115 * i, 450 + 255 * i, flip)
        _playing_card(ctx, i, x, 590, chart=chart, window=flip,
                      sx=max(0.035, abs(1 - 2 * flip)))
    num(ctx, "1969", 960, 837, 104, "cyan", on(t, at(32, 2), 0.12), align="center")
    glow_at(ctx, 960, 590, "cyan", 2.0, 0.38 * kick(t, at(32, 2)))


def shot_grinold(ctx, t):
    """#33: beats one, three and five step both the chip grid and its light column to 1, 2, 3."""
    b = bt(t, 33)
    _table(ctx)
    _quote_row(ctx, a=0.18 * (1 - sm(b / 0.6)))
    four = on(t, at(33, 2), 0.18)
    nine = on(t, at(33, 4), 0.18)
    side = 1 + four + nine
    lit = on(t, at(33, 4), 0.12)
    _authority(ctx, "grinold", {1: (lit, "cyan"), 2: (0.35 + 0.65 * lit, "cyan"),
                              3: (lit, "cyan")}, a=sm(b / 0.7))
    top, base = 620 - 90 * side, 620
    gradient = cairo.LinearGradient(0, base, 0, top)
    gradient.add_color_stop_rgba(0, *hx("cyan"), 0.12 + 0.08 * side)
    gradient.add_color_stop_rgba(1, *hx("cyan"), 0.25 + 0.2 * side)
    ctx.rectangle(905, top, 110, base - top); ctx.set_source(gradient); ctx.fill()
    gline(ctx, [(905, base), (905, top), (1015, top), (1015, base)],
          "cyan", 3, 0.45 + 0.16 * side, glow=0.5)
    gline(ctx, [(905, top), (1015, top)], "ice", 5, 0.55 + 0.14 * side, glow=0.9)
    for level in range(1, 4):
        if side >= level:
            gline(ctx, [(909, base - 90 * level), (1011, base - 90 * level)],
                  "ice", 2, 0.35 + 0.15 * side, glow=0.3)
    glow_at(ctx, 960, top, "cyan", 0.6 + 0.2 * side, 0.17 * side)
    glow_at(ctx, 960, 760, "cyan", 0.8 * side, 0.2 * side)
    for count, alpha in ((1, 1 - four), (2, four * (1 - nine)), (3, nine)):
        for row in range(count):
            for col in range(count):
                x = 960 + (col - (count - 1) / 2) * 110
                y = 760 + (row - (count - 1) / 2) * 70
                a = alpha * (0.88 + 0.06 * ((row + 2 * col) % 3))
                glow_at(ctx, x, y, "cyan", 0.45, 0.35 * a)
                _chip(ctx, x, y, 42, a=a)
    glow_at(ctx, 960, top, "cyan", 1.6, 0.45 * kick(t, at(33, 4)))


SHOTS = [
    (20, 20, shot_backtest),
    (21, 21, shot_sharpe),
    (22, 22, shot_stability),
    (23, 23, shot_selection),
    (24, 24, shot_overfit),
    (26, 26, shot_coin_edge),
    (27, 27, shot_ten_flips),
    (28, 28, shot_million),
    (29, 29, shot_costs),
    (30, 30, shot_kelly),
    (31, 31, shot_thorp),
    (32, 32, shot_wall_street),
    (33, 33, shot_grinold),
]
