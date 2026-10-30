"""Order desk, holdings and counterparties (#35–#46 of quant.scenes.md).

One horizontal matching slot joins wordless, complementary order halves. Amber is
always the phone and cyan the program; tickets identify orders, never money flows.
All frames are pure functions of film time. Local beat indices are zero-based.
"""
from quant_kit import *  # noqa: F401,F403


def _u(b, start, duration=1.0):
    """A clamped, beat-relative mechanical transition."""
    return sm((b - start) / duration)


def _oval(ctx, x, y, rx, ry, col, a=1.0, lw=3):
    """An unlabelled dial, lens or hand-drawn search ring."""
    pts = [(x + rx * math.cos(v), y + ry * math.sin(v))
           for v in np.linspace(0, 2 * math.pi, 81)]
    gline(ctx, pts, col, lw, a, 0.6)


@lru_cache(None)
def _teeth():
    """Both halves share this exact boundary, including its four interlocking teeth."""
    return ((0, -58), (0, -42), (12, -42), (12, -22), (-12, -22),
            (-12, -2), (12, -2), (12, 18), (-12, 18), (-12, 38),
            (0, 38), (0, 58))


@lru_cache(None)
def _wear():
    """Fixed scratches make old paper recognisable without dates or colour coding."""
    r = rng(3906)
    return tuple((float(r.uniform(-145, 135)), float(r.uniform(-45, 45)),
                  float(r.uniform(8, 24))) for _ in range(17))


def _half_path(ctx, hand, buy=True):
    """Trace a half-ticket; the outside corner also distinguishes the two order sides."""
    if hand < 0:
        outer = [(-160, -44), (-146, -58)] if buy else [(-160, -58)]
        bottom = [(-160, 58)] if buy else [(-146, 58), (-160, 44)]
        path(ctx, outer + list(_teeth()) + bottom, close=True)
    else:
        outer = [(146, -58), (160, -44)] if buy else [(160, -58)]
        bottom = [(160, 58)] if buy else [(160, 44), (146, 58)]
        path(ctx, [_teeth()[0]] + outer + bottom + list(reversed(_teeth())), close=True)


def _source(ctx, x, y, shape, col, a):
    """A colour-and-shape maker's punch, never an account label."""
    if shape == "square":
        rrect(ctx, x - 8, y - 8, 16, 16, 2); stroke(ctx, col, 2, a)
    elif shape == "diamond":
        gline(ctx, [(x, y - 10), (x + 9, y), (x, y + 10), (x - 9, y)],
              col, 2, a, 0, close=True)
    elif shape == "double":
        for dx in (-5, 5):
            gline(ctx, [(x + dx, y - 10), (x + dx, y + 10)], col, 2, a, 0)
    elif shape == "circle":
        _oval(ctx, x, y, 8, 8, col, a, 2)


def _half(ctx, x, y, hand, col, a=1.0, w=360, h=140, buy=True,
          finish="fresh", source=None, gap=False):
    """Draw a wordless half with its toothed contact edge at (x, y)."""
    if a <= 0.002:
        return
    ctx.save(); ctx.translate(x, y); ctx.scale(w / 320, h / 116)
    _half_path(ctx, hand, buy)
    if gap:
        ctx.set_dash([7, 9]); stroke(ctx, "dim", 1.5, a)
    else:
        fill(ctx, "bg2" if finish == "worn" else "glass", 0.98 * a, keep=True)
        fill(ctx, "ink2" if finish == "worn" else col,
             (0.13 if finish == "worn" else 0.22) * a, keep=True)
        stroke(ctx, "ink2" if finish == "worn" else col,
               2.3 if finish == "worn" else 3, a * 0.95)
        ctx.save(); _half_path(ctx, hand, buy); ctx.clip()
        if finish == "worn":
            for xx, yy, length in _wear():
                gline(ctx, [(xx, yy), (xx + length, yy - 3)], "ink2", 1.4, a * 0.48, 0)
            gline(ctx, [(hand * 144, -45), (hand * 139, 39)], "ink2", 3, a * 0.3, 0)
        else:
            gline(ctx, [(hand * 148, -43), (hand * 148, 43)], col, 3, a * 0.85, 0)
            gline(ctx, [(hand * 143, -50), (hand * 34, -50)], "white", 3.5, a * 0.8, 0)
            gline(ctx, [(hand * 139, -43), (hand * 59, -15)], "ice", 8, a * 0.12, 0)
        ctx.restore()
        if finish == "worn":
            for yy in (-35, 3, 39):
                gline(ctx, [(hand * 159, yy - 4), (hand * 153, yy), (hand * 159, yy + 5)],
                      "bg2", 4.5, a, 0)
        if source:
            _source(ctx, hand * 119, -8, source, col, a * 0.85)
            dy = 1 if buy else -1
            gline(ctx, [(hand * 119 - 8, 26 + 4 * dy), (hand * 119, 26 - 4 * dy),
                        (hand * 119 + 8, 26 + 4 * dy)], col, 2, a * 0.65, 0)
    ctx.restore()


def _ticket(ctx, x, y, col="ice", a=1.0, w=360, h=140, finish="fresh"):
    """A complete ticket retains the same vertical join as every matched pair."""
    _half(ctx, x, y, -1, col, a, w, h, True, finish)
    _half(ctx, x, y, 1, col, a, w, h, False, finish)


def _phone(ctx, t, a):
    """A filled 260×520 trading screen; shell, live candles and volume dim as one object."""
    if a <= 0.002:
        return
    ctx.save(); ctx.push_group()
    rrect(ctx, 120, 260, 260, 520, 32); fill(ctx, "glass", 1, keep=True)
    stroke(ctx, "amber", 3.5, 1)
    rrect(ctx, 136, 282, 228, 470, 22); fill(ctx, "bg2", 1, keep=True)
    stroke(ctx, "amber", 1.3, 0.5)
    ctx.save(); rrect(ctx, 136, 282, 228, 470, 22); ctx.clip()
    for yy in (361, 426, 491, 556, 621):
        gline(ctx, [(151, yy), (349, yy)], "dim", 1, 0.35, 0)
    data = walk(350, 20, 0.03, 0.85).copy()
    data[-1, 3] = data[-1, 0] + 0.75 * math.sin(t * 2.1)
    data[-1, 1] = max(data[-1, 1], data[-1, 0] + 0.85)
    data[-1, 2] = min(data[-1, 2], data[-1, 0] - 0.85)
    Y = candles(ctx, data, 151, 341, 198, 284, mono="amber", a=0.95)
    yy = Y(data[-1, 3])
    gline(ctx, [(151, yy), (349, yy)], "amber", 1.2, 0.45, 0, dash=[3, 5])
    dot(ctx, 344, yy, 3, "amber", 0.85, 0.3)
    for i, (o, hi, lo, close) in enumerate(data):
        height = 13 + 34 * min(1, abs(close - o) / 1.5)
        ctx.rectangle(152 + 9.8 * i, 716 - height, 6, height)
        fill(ctx, "amber", 0.35 + 0.25 * (i == 19))
    ctx.restore()
    rrect(ctx, 212, 295, 76, 7, 3); fill(ctx, "amber", 0.55)
    rrect(ctx, 213, 761, 74, 5, 2); fill(ctx, "amber", 0.65)
    rrect(ctx, 368, 475, 17, 90, 7); fill(ctx, "bg", 1, keep=True)
    stroke(ctx, "amber", 2.5, 1)
    screen = ctx.pop_group(); ctx.set_source(screen); ctx.paint_with_alpha(a); ctx.restore()


def _stage(ctx, t, phone=0.25, program=0.25, seam=0.35, halt=0.0):
    """The enlarged desk keeps its seam at y=520 and the jam warning behind its centre."""
    rrect(ctx, 380, 505, 1160, 30, 15); fill(ctx, "bg", 0.92)
    gline(ctx, [(380, 520), (1540, 520)], "ice", 2.5, seam * 0.5, 0.4)
    for yy in (424, 616):
        gline(ctx, [(710, yy), (1210, yy)], "dim", 2, seam, 0)
    for xx in (710, 1210):
        for yy, end in ((424, 440), (616, 600)):
            gline(ctx, [(xx, yy), (xx, end)], "dim", 1.5, seam * 0.4, 0)
    _phone(ctx, t, phone)
    col = "red" if halt > 0.5 else "cyan"
    lamp = program * (1 - 0.8 * halt)
    gline(ctx, [(1540, 475), (1540, 565)], col, 5, lamp, 0.7)
    _oval(ctx, 1660, 520, 58, 58, col, lamp * 0.55, 2)
    _source(ctx, 1660, 520, "square", col, lamp)
    glow_at(ctx, 1660, 520, col, 1.25, lamp * 0.35)
    dot(ctx, 1660, 520, 11, col, lamp, 0.8)
    warning = program * halt
    gline(ctx, [(960, 441), (960, 477)], "red", 3, warning * 0.55, 0)
    _oval(ctx, 960, 414, 27, 27, "red", warning * 0.7, 3)
    dot(ctx, 960, 414, 15, "red", warning, 1)


def _jam(ctx, a):
    """Two oversized tickets overlap at the centre, in front of the red warning lamp."""
    for x, yy, angle in ((945, 511, -0.06), (985, 537, 0.045)):
        ctx.save(); ctx.translate(x, yy); ctx.rotate(angle)
        _ticket(ctx, 0, 0, "cyan", a)
        ctx.restore()


def _contact(ctx, t, n, beat, y=520):
    """Only the shared tooth edge flashes: two orders have become one trade."""
    hit = kick(t, at(n, beat), 0.32)
    glow_at(ctx, 960, y, "ice", 1.5, hit)
    gline(ctx, [(960 + x * 360 / 320, y + yy * 140 / 116) for x, yy in _teeth()],
          "ice", 4, hit, 1.5)


def _padlock(ctx, x, y, closed, a=1.0, col="ice"):
    """A 40-pixel lock body with a clearly raised or fully seated shackle."""
    if a <= 0.002:
        return
    ctx.save(); ctx.translate(x, y); ctx.scale(4 / 3, 4 / 3)
    ctx.save(); ctx.translate(-10, -4); ctx.rotate(-1.05 * (1 - closed))
    ctx.move_to(0, 4); ctx.line_to(0, -9)
    ctx.arc(10, -9, 10, math.pi, 2 * math.pi); ctx.line_to(20, 4)
    stroke(ctx, col, 3, a); ctx.restore()
    rrect(ctx, -15, 0, 30, 24, 4); fill(ctx, "glass", a, keep=True)
    stroke(ctx, col, 2.5, a)
    ctx.arc(0, 10, 2.5, 0, 2 * math.pi); fill(ctx, col, a)
    gline(ctx, [(0, 11), (0, 16)], col, 2, a, 0)
    ctx.restore()


def _tray_shell(ctx, a=1.0, morph=1.0):
    """A 1200-pixel ten-slot drawer, morphed from the search field's rounded shell."""
    x, y = mix(425, 360, morph), mix(390, 592, morph)
    w, h = mix(1070, 1200, morph), mix(140, 278, morph)
    rrect(ctx, x, y, w, h, mix(70, 20, morph)); fill(ctx, "glass", 0.94 * a, keep=True)
    stroke(ctx, "ice", 2, a * 0.65)
    for i in range(10):
        xx = 380 + i * 116
        rrect(ctx, xx, 608, 108, 244, 8); fill(ctx, "bg", a * morph * 0.9, keep=True)
        stroke(ctx, "dim", 1.5, a * morph * 0.8)


def _holding(ctx, i, worn, closed, a=1.0, x=None, y=686):
    """Larger matte/worn or glossy tickets carry readable open/closed padlocks."""
    x = 434 + i * 116 if x is None else x
    _ticket(ctx, x, y, "ice", a, 94, 145, "worn" if worn else "fresh")
    _padlock(ctx, x, y + 100, closed, a, "ink2" if worn else "ice")


def _inventory(ctx, a=1.0, after_turn=False):
    """The exact settled drawer state is reused across both adjacent cuts."""
    _tray_shell(ctx, a)
    for i in range(10):
        worn = i in (4, 5) if after_turn else i < 6
        _holding(ctx, i, worn, 0 if worn else 1, a)


def _stopwatch(ctx, seconds, a=1.0, readout=True):
    """An unnumbered stopwatch with the shot's sole numeric readout."""
    _oval(ctx, 960, 250, 67, 67, "amber", a * 0.65)
    rrect(ctx, 947, 161, 26, 12, 3); fill(ctx, "amber", a * 0.8)
    for i in range(12):
        v = i * math.pi / 6
        gline(ctx, [(960 + 57 * math.sin(v), 250 - 57 * math.cos(v)),
                    (960 + 63 * math.sin(v), 250 - 63 * math.cos(v))], "amber", 2, a * 0.6, 0)
    v = seconds / 3 * 2 * math.pi
    gline(ctx, [(960, 250), (960 + 48 * math.sin(v), 250 - 48 * math.cos(v))], "amber", 3, a)
    if readout:
        num(ctx, f"{seconds:.1f} s", 960, 410, 82, "amber", a, align="center")


def _time_ruler(ctx, exponent, a=1.0):
    """Six tenfold temporal enlargements, not a camera zoom; no extra numeric labels."""
    if a <= 0.002:
        return
    x0, x1, yy = 460, 1460, 754
    gline(ctx, [(x0, yy), (x1, yy)], "dim", 2, a * 0.6, 0)
    spacing = 40 * 10 ** (exponent % 1)
    for i in range(-13, 14):
        xx = 960 + i * spacing
        if x0 <= xx <= x1:
            height = 23 if i % 5 == 0 else 12
            gline(ctx, [(xx, yy - height), (xx, yy + height)], "ice", 1.5, a * 0.65, 0)
    # A subpixel interval only resolves at the last (millionfold) ruler scale.
    gap = 12 * 10 ** (exponent - 6)
    for xx, col in ((960 - gap / 2, "cyan"), (960 + gap / 2, "ice")):
        gline(ctx, [(xx, yy - 37), (xx, yy + 37)], col, 2.5, a, 0.5)


def _search(ctx, b, a=1.0, shell=True, letters=True):
    """One query only; its existing T+0 glyphs enlarge rather than being duplicated."""
    if a <= 0.002:
        return
    if shell:
        rrect(ctx, 425, 390, 1070, 140, 70); fill(ctx, "glass", a * 0.98, keep=True)
        stroke(ctx, "ice", 2, a * 0.6)
    _oval(ctx, 480, 457, 18, 18, "ice", a * 0.7, 2.5)
    gline(ctx, [(493, 470), (508, 485)], "ice", 3, a * 0.7, 0)
    query, prefix = "量化为什么能 T+0？", "量化为什么能 "
    s = typed(query, _u(b, 0.05, 1.65))
    z = _u(b, 2, 1.45)
    x, baseline, size = 550, 480, 52
    pre = s[:len(prefix)]
    if letters:
        text(ctx, pre, x, baseline, size, "ink", a * (1 - 0.55 * z), bold=True)
    start = x + text_width(ctx, prefix, size, bold=True)
    tw = text_width(ctx, "T+0", size, bold=True)
    cx = start + tw / 2 + (tw * 0.425 + 8) * z
    token = s[len(prefix):len(prefix) + 3]
    if letters:
        ctx.save(); ctx.translate(cx, baseline); ctx.scale(1 + 0.85 * z, 1 + 0.85 * z)
        text(ctx, token, 0, 0, size, "amber" if z > 0 else "ink", a, bold=True, align="center")
        ctx.restore()
        text(ctx, s[len(prefix) + 3:], cx + tw / 2 * (1 + 0.85 * z) + 12,
             baseline, size, "ink", a * (1 - 0.55 * z), bold=True)
    ring = _u(b, 2, 0.15)
    _oval(ctx, cx, 455 - 12 * z, tw / 2 * (1 + 0.85 * z) + 22,
          45 + 21 * z, "amber", a * ring)


def _price(ctx, points, progress, col, a=0.55):
    """Price stays separate from order routes and never becomes a value-flow arrow."""
    if a <= 0.002:
        return
    reach = max(0.0, min(1.0, progress)) * (len(points) - 1)
    count = int(reach)
    pts = list(points[:count + 1])
    if count < len(points) - 1:
        p, q, k = points[count], points[count + 1], reach - count
        pts.append((mix(p[0], q[0], k), mix(p[1], q[1], k)))
    gline(ctx, pts, col, 3, a, 0.5)
    if pts:
        dot(ctx, *pts[-1], 4, col, a, 0.5)


@lru_cache(None)
def _rally():
    """The rally's endpoint becomes the next shot's falling price trace."""
    return ((650, 360), (735, 351), (805, 363), (885, 325),
            (975, 334), (1055, 281), (1150, 235), (1250, 248))


@lru_cache(None)
def _fall():
    """A falling trace leaves room to continue lower after the trade has locked."""
    return ((650, 248), (735, 259), (805, 246), (885, 285),
            (975, 279), (1055, 321), (1150, 346), (1250, 365))


def _alternatives(ctx, a=0.4, buy=False):
    """Three readable but subordinate alternatives wait beside the program's half."""
    for x, col, shape in ((1180, "violet", "double"), (1340, "ice", "circle"), (1500, "gold", "diamond")):
        _half(ctx, x, 675, 1, col, a, 260, 110, buy, source=shape)


def _joined(ctx, y=520, a=1.0, reverse=False, left="amber", left_source="circle"):
    """A completed trade remains a pair, not a payment travelling to either source."""
    _half(ctx, 960, y, -1, left, a, buy=not reverse, source=left_source)
    _half(ctx, 960, y, 1, "cyan", a, buy=reverse, source="square")


def _queues(ctx, opened, a=1.0):
    """Four populated lanes fill both sides of a 1300×300 panel; amber is only one lane."""
    rrect(ctx, 310, 600, 1300, 300, 16); fill(ctx, "bg", a * 0.95, keep=True)
    stroke(ctx, "dim", 2, a * 0.8)
    ctx.save(); rrect(ctx, 310, 600, 1300, 300, 16); ctx.clip()
    sources = (("violet", "double", "cyan", "square"),
               ("ice", "circle", "gold", "diamond"),
               ("gold", "diamond", "violet", "double"),
               ("amber", "circle", "ice", "circle"))
    for row, (left, left_shape, right, right_shape) in enumerate(sources):
        yy = 640 + row * 72
        gline(ctx, [(326, yy), (1594, yy)], "dim", 1.5, a * 0.4, 0)
        _source(ctx, 342, yy, left_shape, left, a * 0.85)
        _source(ctx, 1580, yy, right_shape, right, a * 0.85)
        for j in range(4):
            alpha = a * (0.74 - 0.07 * j)
            _half(ctx, 850 - j * 125, yy, -1, left, alpha,
                  180, 60, True, source=left_shape)
            _half(ctx, 1070 + j * 125, yy, 1, right, alpha,
                  180, 60, False, source=right_shape)
    cover_x = 310 + 1324 * opened
    rrect(ctx, cover_x, 600, 1300, 300, 16); fill(ctx, "glass", a, keep=True)
    stroke(ctx, "dim", 2, a * 0.7)
    for j in range(3):
        gline(ctx, [(cover_x + 638 + 12 * j, 730), (cover_x + 638 + 12 * j, 772)],
              "ice", 2.5, a * 0.5, 0)
    ctx.restore()


# #35–#38: the same order outlet, time magnification, a jam, then a query.
def shot_seconds(ctx, t):
    """#35: the chip becomes a hesitant ticket; beat 5 releases it and stops the watch."""
    b = bt(t, 35)
    _stage(ctx, t, phone=0.8, program=0.12)
    release = on(t, at(35, 4), 0.32)
    hesitation = 22 * math.sin(max(0, b) * math.pi * 1.6) * (1 - release)
    x = mix(330 + hesitation, 960, release)
    morph = _u(b, 0, 0.85)
    _stopwatch(ctx, 3.0 if b >= 4 else min(2.9, max(0, b) * 0.75))
    ctx.save(); ctx.rectangle(380, 420, 1160, 220); ctx.clip()
    _ticket(ctx, x, 520, "amber", morph)
    if morph < 1:
        ctx.arc(400, 520, 43, 0, 2 * math.pi); fill(ctx, "glass", 1 - morph, keep=True)
        stroke(ctx, "amber", 4, 1 - morph)
        _oval(ctx, 400, 520, 31, 31, "amber", 1 - morph, 2)
    ctx.restore()
    glow_at(ctx, 380, 520, "amber", 0.7, kick(t, at(35, 4), 0.25) * 0.5)


def shot_microseconds(ctx, t):
    """#36: beat 3 fires from the cyan exit; six decimal ruler enlargements resolve the delay."""
    b = bt(t, 36)
    _stage(ctx, t, phone=mix(0.8, 0.18, _u(b, 0, 1)), program=mix(0.12, 0.9, _u(b, 0, 1)))
    old = 1 - _u(b, 0, 0.8)
    _ticket(ctx, 960, 520, "amber", old)
    _stopwatch(ctx, 3.0, old, readout=False)
    fire = on(t, at(36, 2), 0.045)
    ctx.save(); ctx.rectangle(650, 420, 890, 220); ctx.clip()
    _ticket(ctx, mix(1720, 960, fire), 520, "cyan", _u(b, 0.5, 0.8))
    ctx.restore()
    glow_at(ctx, 1540, 520, "cyan", 1, kick(t, at(36, 2), 0.25))
    exponent = 6 * _u(b, 2.12, 3.0)
    _time_ruler(ctx, exponent, _u(b, 1.2, 0.8))
    glow_text(ctx, "微秒", 960, 345, 86, "cyan", _u(b, 2, 0.7), align="center")


def shot_even_orders(ctx, t):
    """#37: identical, evenly spaced sends; beat 5 overlaps two tickets and halts the exit."""
    b = bt(t, 37)
    halt = on(t, at(37, 4), 0.08)
    _stage(ctx, t, phone=0.18, program=0.9, halt=halt)
    _time_ruler(ctx, 6, 1 - _u(b, 0, 0.8))
    _ticket(ctx, mix(960, 620, _u(b, 0, 0.8)), 520, "cyan", 1 - _u(b, 0, 0.8))
    ctx.save(); ctx.rectangle(380, 420, 1160, 220); ctx.clip()
    for i in range(4):
        age = b - i
        if 0 <= age <= 2.2:
            x = 1720 - 700 * age
            _ticket(ctx, x, 520, "cyan", (1 - _u(age, 1.6, 0.6)) * (1 - halt))
    ctx.restore()
    _jam(ctx, halt)
    glow_at(ctx, 960, 520, "red", 1.5, kick(t, at(37, 4), 0.3))


def shot_t0_search(ctx, t):
    """#38: type the complete query; beat 3 circles and enlarges its existing T+0."""
    b = bt(t, 38)
    fade = 1 - _u(b, 0, 0.9)
    _stage(ctx, t, phone=0.18, program=mix(0.9, 0.15, 1 - fade), seam=0.18, halt=fade)
    _jam(ctx, fade)
    _search(ctx, b, _u(b, 0, 0.6))


# #39–#40: one drawer, six worn unlocked tickets and four fresh tickets.
def shot_inventory(ctx, t):
    """#39: the query shell becomes ten slots; beat 5 shuts all four new padlocks."""
    b = bt(t, 39)
    _stage(ctx, t, phone=0.18, program=0.15, seam=0.18)
    morph = _u(b, 0, 1.6)
    _tray_shell(ctx, morph=morph)
    _search(ctx, 6, 1 - _u(b, 0, 0.65), shell=False, letters=False)
    closed = on(t, at(39, 4), 0.09)
    for i in range(10):
        x = mix(960, 434 + i * 116, morph)
        y = mix(520, 686, morph)
        _holding(ctx, i, i < 6, 0 if i < 6 else closed, morph, x, y)
        if i >= 6:
            glow_at(ctx, x, y + 110, "ice", 0.55, kick(t, at(39, 4), 0.25) * 0.65)
    glow_text(ctx, "底仓", 960, 345, 86, "ice", _u(b, 3.4, 0.6), align="center")


def shot_inventory_turn(ctx, t):
    """#40: sell four worn tickets on a red rise; four locked fresh ones refill by beat 5."""
    b = bt(t, 40)
    _stage(ctx, t, phone=0.18, program=0.15, seam=0.18)
    _tray_shell(ctx)
    sell = _u(b, 0, 1.65)
    buy = _u(b, 2, 2)
    up = ((650, 354), (770, 344), (875, 314), (970, 272))
    dip = ((970, 272), (1070, 300), (1180, 330), (1270, 348))
    _price(ctx, up, _u(b, 0, 1.1), "red")
    _price(ctx, dip, _u(b, 1.6, 1.5), "green")
    for i in range(10):
        if i < 4:
            if sell < 1:
                _holding(ctx, i, True, 0, 1 - _u(b, 1.1, 0.55),
                         mix(434 + i * 116, 380, sell), mix(686, 520, sell))
            if b >= 2:
                _holding(ctx, i, False, 1, _u(b, 2, 0.5),
                         mix(1540, 434 + i * 116, buy), mix(520, 686, buy))
        else:
            _holding(ctx, i, i < 6, 0 if i < 6 else 1)
    glow_at(ctx, 960, 705, "ice", 2.4, kick(t, at(40, 4), 0.3) * 0.55)


# #41–#46: complementary teeth answer only who supplied the other order.
def shot_bill_reveal(ctx, t):
    """#41: a drawer receipt rises; beat 3 flips it into one half and a toothed absence."""
    b = bt(t, 41)
    _stage(ctx, t, phone=0.18, program=0.15)
    _inventory(ctx, mix(1, 0.28, _u(b, 0, 1.7)), after_turn=True)
    lift = _u(b, 0, 1.7)
    y = mix(686, 520, lift)
    flip = on(t, at(41, 2), 0.28)
    ctx.save(); ctx.translate(960, y); ctx.scale(max(0.015, abs(math.cos(math.pi * flip))), 1)
    if flip < 0.5:
        _ticket(ctx, 0, 0, "ice", lift)
    else:
        _half(ctx, 0, 0, -1, "amber", source="circle")
        _half(ctx, 0, 0, 1, "dim", a=0.65, buy=False, gap=True)
    ctx.restore()
    glow_text(ctx, "？", 960, 345, 102, "amber", _u(b, 2, 0.45), align="center")


def shot_matching(ctx, t):
    """#42: the opposite half reaches the same price height and locks on beat 3."""
    b = bt(t, 42)
    _stage(ctx, t, phone=0.25, program=mix(0.15, 0.7, _u(b, 0, 1)))
    _inventory(ctx, 0.28 * (1 - _u(b, 0, 1)), after_turn=True)
    _half(ctx, 960, 520, -1, "amber", source="circle")
    enter = _u(b, 0, 2)
    _half(ctx, mix(1540, 960, enter), mix(568, 520, enter), 1, "cyan", buy=False, source="square")
    _half(ctx, 960, 520, 1, "dim", a=0.65 * (1 - enter), buy=False, gap=True)
    _contact(ctx, t, 42, 2)


def shot_chasing(ctx, t):
    """#43: beat 5 identifies the cyan selling half; three alternative sources remain dim."""
    b = bt(t, 43)
    _stage(ctx, t, phone=mix(0.25, 0.7, _u(b, 0, 1)), program=0.7)
    old = 1 - _u(b, 0, 0.8)
    _joined(ctx, mix(520, 570, 1 - old), old)
    _price(ctx, _rally(), _u(b, 0, 2.1), "red")
    _alternatives(ctx, 0.4 * _u(b, 0, 1))
    left = _u(b, 0.7, 2)
    right = _u(b, 2.6, 1.4)
    _half(ctx, mix(380, 960, left), mix(555, 520, left), -1, "amber",
          _u(b, 0.7, 0.4), source="circle")
    _half(ctx, mix(1540, 960, right), mix(475, 520, right), 1, "cyan",
          _u(b, 2.6, 0.4), buy=False, source="square")
    _contact(ctx, t, 43, 4)
    glow_text(ctx, "可能", 960, 830, 80, "ice", _u(b, 4, 0.55), align="center")


def shot_panic_selling(ctx, t):
    """#44: beat 5 joins phone sell to program buy; the separate price trace keeps falling."""
    b = bt(t, 44)
    _stage(ctx, t, phone=0.7, program=0.7)
    old = 1 - _u(b, 0, 0.8)
    _joined(ctx, mix(520, 570, 1 - old), old)
    pts = lerp_pts(_rally(), _fall(), _u(b, 0, 1.8))
    _price(ctx, pts, 1, mix(hx("red"), hx("green"), _u(b, 0, 0.7)))
    _price(ctx, ((1250, 365), (1295, 385), (1340, 378), (1400, 425)),
           _u(b, 4.15, 2.4), "green")
    _alternatives(ctx, buy=True)
    left = _u(b, 0.7, 2)
    right = _u(b, 2.6, 1.4)
    _half(ctx, mix(380, 960, left), mix(475, 520, left), -1, "amber",
          _u(b, 0.7, 0.4), buy=False, source="circle")
    _half(ctx, mix(1540, 960, right), mix(555, 520, right), 1, "cyan",
          _u(b, 2.6, 0.4), buy=True, source="square")
    _contact(ctx, t, 44, 4)
    glow_text(ctx, "可能", 960, 830, 80, "ice", align="center")


def shot_open_desk(ctx, t):
    """#45: beat 3 slides the cover away, exposing many sources sharing the same slot."""
    b = bt(t, 45)
    _stage(ctx, t, phone=mix(0.7, 0.25, _u(b, 0, 1.5)), program=0.7)
    old = 1 - _u(b, 0, 1)
    _joined(ctx, a=old, reverse=True)
    _price(ctx, _fall(), 1, "green", 0.55 * old)
    _price(ctx, ((1250, 365), (1295, 385), (1340, 378), (1400, 425)), 1, "green", 0.55 * old)
    opened = on(t, at(45, 2), 0.35)
    _alternatives(ctx, 0.4 * (1 - opened), buy=True)
    _queues(ctx, opened, _u(b, 0, 1))
    gline(ctx, [(310 + 1324 * opened, 601), (310 + 1324 * opened, 899)],
          "ice", 2, kick(t, at(45, 2), 0.25) * 0.45, 0.5)


def shot_other_counterparties(ctx, t):
    """#46: fund and program join without the dimmed phone; beat 5 locks a second pair."""
    b = bt(t, 46)
    _stage(ctx, t, phone=mix(0.25, 0.015, _u(b, 0, 1.2)), program=0.7)
    _queues(ctx, 1, mix(1, 0.32, _u(b, 0, 1.5)))
    first = _u(b, 0.3, 1.7)
    settle = _u(b, 2.45, 0.95)
    y = mix(520, 740, settle)
    visible = _u(b, 0.3, 0.35) * mix(1, 0.5, settle)
    w, h = mix(180, 360, first), mix(60, 140, first)
    _half(ctx, mix(850, 960, first), mix(640, y, first), -1, "violet",
          visible, w, h, buy=True, source="double")
    _half(ctx, mix(1070, 960, first), mix(640, y, first), 1, "cyan",
          visible, w, h, buy=False, source="square")
    second = _u(b, 2.6, 1.4)
    visible = _u(b, 2.6, 0.35)
    w, h = mix(180, 360, second), mix(60, 140, second)
    _half(ctx, mix(850, 960, second), mix(640, 520, second), -1, "violet",
          visible, w, h, buy=False, source="double")
    _half(ctx, mix(1070, 960, second), mix(640, 520, second), 1, "cyan",
          visible, w, h, buy=True, source="square")
    _contact(ctx, t, 46, 4)


SHOTS = [
    (35, 35, shot_seconds),
    (36, 36, shot_microseconds),
    (37, 37, shot_even_orders),
    (38, 38, shot_t0_search),
    (39, 39, shot_inventory),
    (40, 40, shot_inventory_turn),
    (41, 41, shot_bill_reveal),
    (42, 42, shot_matching),
    (43, 43, shot_chasing),
    (44, 44, shot_panic_selling),
    (45, 45, shot_open_desk),
    (46, 46, shot_other_counterparties),
]
