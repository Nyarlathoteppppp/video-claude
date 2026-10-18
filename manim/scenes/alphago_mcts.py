"""AlphaGo 是怎么“想”棋的：蒙特卡洛树搜索 — public explainer, two cuts in one file.

    AlphaGoMCTS       16:9  (-r 1920,1080 --frame_rate 30)
    AlphaGoMCTSShort  9:16  (-r 1080,1920 --frame_rate 30; the frame is 4.5 units wide)

Storyboard: alphago_mcts.scenes.md ("Update" and "Fact wording" sections override the early beats).
There is no voiceover: the on-screen text carries the story.

Reading time is enforced in code (Base.play): every caption / text card is a "track" that must stay fully visible
(fade-in finished -> fade-out started) for  reading_time(text) + 0.5 x (animation time under it) + aha hold,
one reading target at a time. Every track is written to <scene>.captions.csv and asserted.
Sound effects come from manim/sfx through style.sfx; Base.sfx enforces the README mixing rules.
The optional voice(key) hooks from log_trick.py are kept; they do nothing without WAVs in manim/voice/<id>/.

Every number on screen is computed below (tic-tac-toe MCTS, random play-outs, the pi darts) or comes from the
storyboard's Facts list (250 moves, 150 moves, 10^360, 10^80, 2016, game 2, move 37, 1/10000, 1946, 2017).
"""
import sys
import os
import csv
import json
import wave
import math
import random
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *  # noqa: F401,F403  (palette, cn, reading_time, sfx, SFX_DIR, TeX PATH)
from style import sfx as style_sfx, reading_time, SFX_DIR

import numpy as np
from PIL import Image, ImageDraw
from manim.animation.animation import Wait
from manim.utils.rate_functions import ease_out_cubic, rush_into, there_and_back

# ======================================================================================
# colour = meaning (4 meaning colours + red for the threat only)
# ======================================================================================
X_COL = OPEN_COL        # X, the agent, AlphaGo's accent
O_COL = V_COL           # O, the opponent
WIN_COL = Q_COL         # win rate / success fraction
BONUS_COL = R_COL       # curiosity bonus
THREAT_COL = NEG_COL    # the threat only
DRAW_COL = DIM
BOARD_FILL, BOARD_EDGE, BOARD_LINE = "#272b33", "#3a404b", "#5b6270"
STONE_B, STONE_B_RIM, STONE_W, STONE_W_RIM = "#0b0c0e", "#7a818c", "#ebe7dd", "#a9adb5"
CARD_FACE, CARD_BACK = "#ebe7dd", "#2f3540"
DIM_OP = 0.25           # opacity factor for "not the current focus"

# ======================================================================================
# the toy problem: same UCT MCTS as scratchpad/ttt/mcts.py (same seed, same order of random calls)
# ======================================================================================
LINES3 = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
TOY_BOARD, TOY_PLAYER = "X.O.O....", "X"
CANDS = (1, 3, 5, 6, 7, 8)
UCT_C = math.sqrt(2)
MCTS_SEED, MCTS_ITERS = 7, 2000
TABLE = {20: (3, 2, 4, 4, 3, 4), 50: (6, 9, 7, 14, 7, 7), 300: (25, 54, 35, 125, 34, 27),
         2000: (63, 118, 87, 1584, 83, 65)}
DEMO_SEED = 145          # beat 3 play-outs: three after move 6, then three after move 1
PI_SEED = 2016           # darts
PI_CHECKS = (10, 100, 1000, 5000)
SCORE_X = {"X": 1.0, "D": 0.5, "O": 0.0}


def winner(b):
    for a, c, d in LINES3:
        if b[a] != "." and b[a] == b[c] == b[d]:
            return b[a]
    return "D" if "." not in b else None


def legal(b):
    return [i for i, v in enumerate(b) if v == "."]


def put(b, i, p):
    return b[:i] + p + b[i + 1:]


def other(p):
    return "O" if p == "X" else "X"


def rollout(b, p, rng, trace=None):
    while winner(b) is None:
        i = rng.choice(legal(b))
        b = put(b, i, p)
        if trace is not None:
            trace.append((i, p))
        p = other(p)
    return winner(b)


class TNode:
    def __init__(self, b, p, parent=None, move=None):
        self.b, self.p, self.parent, self.move = b, p, parent, move
        self.kids = []
        self.untried = legal(b) if winner(b) is None else []
        self.N = 0
        self.W = 0.0


def uct_pick(n, c=UCT_C):
    return max(n.kids, key=lambda k: k.W / k.N + c * math.sqrt(math.log(n.N) / k.N))


def freeze(root):
    """path (tuple of moves) -> (N, W, mover); W counts for the player who moved into the node."""
    out, stack = {}, [(root, ())]
    while stack:
        n, key = stack.pop()
        out[key] = (n.N, n.W, other(n.p))
        for k in n.kids:
            stack.append((k, key + (k.move,)))
    return out


def run_mcts(board, player, iters, seed, snap_at):
    rng = random.Random(seed)
    root = TNode(board, player)
    snaps, log = {}, []
    for it in range(1, iters + 1):
        n, path = root, []
        while not n.untried and n.kids:
            n = uct_pick(n)
            path.append(n.move)
        n_sel = len(path)
        expanded = None
        if n.untried:
            i = rng.choice(n.untried)
            n.untried.remove(i)
            k = TNode(put(n.b, i, n.p), other(n.p), n, i)
            n.kids.append(k)
            n = k
            path.append(i)
            expanded = i
        trace = []
        res = rollout(n.b, n.p, rng, trace)
        m = n
        while m:      # W from the perspective of the player who moved into m (= other(m.p))
            m.N += 1
            mover = other(m.p)
            m.W += 1 if res == mover else (0.5 if res == "D" else 0)
            m = m.parent
        log.append(dict(it=it, path=tuple(path), n_sel=n_sel, expanded=expanded, trace=trace, res=res))
        if it in snap_at:
            snaps[it] = freeze(root)
    return root, snaps, log


def x_wins(rec):
    """Wins seen from X's side (draw = half) for a frozen node record (N, W, mover)."""
    N, W, mover = rec
    return W if mover == "X" else N - W


def toy_data():
    snap_at = {6, 7, 8, 9, 10, 11, 12, 20, 50, 300, 2000}
    root, snaps, log = run_mcts(TOY_BOARD, TOY_PLAYER, MCTS_ITERS, MCTS_SEED, snap_at)
    counts = {it: tuple(snaps[it][(m,)][0] for m in CANDS) for it in TABLE}
    for it, row in TABLE.items():
        assert counts[it] == row, (it, counts[it])                       # storyboard table, exactly
    assert max(counts[20]) / 20 <= 0.25                                   # early: no move above 25 %
    c = counts[2000]
    i6 = CANDS.index(6)
    assert c[i6] == max(c) and c[i6] / 2000 > 0.70                        # late: move 6 > 70 %
    share6 = c[i6] / 2000
    assert round(100 * share6) == 79
    wr = {m: snaps[2000][(m,)][1] / snaps[2000][(m,)][0] for m in CANDS}
    assert 0.45 < wr[6] < 0.50 and round(wr[6], 2) == 0.48               # move 6 ~ a draw
    assert all(round(wr[m], 2) <= 0.22 for m in CANDS if m != 6)         # the others <= 0.22
    order = [e["path"][0] for e in log[:6]]                               # expansion order at the root
    assert sorted(order) == list(CANDS) and all(e["n_sel"] == 0 for e in log[:6])
    for e in log[6:12]:                                                   # rounds 7-12: one step down + one new
        assert e["n_sel"] == 1 and e["expanded"] is not None and len(e["path"]) == 2
    tally = {}
    for e in log[:12]:                                                    # replay the backups in X's view
        for j in range(1, len(e["path"]) + 1):
            t = tally.setdefault(e["path"][:j], [0, 0.0])
            t[0] += 1
            t[1] += SCORE_X[e["res"]]
        if e["it"] in snaps:
            for k, (n, w) in tally.items():
                assert snaps[e["it"]][k][0] == n and abs(x_wins(snaps[e["it"]][k]) - w) < 1e-9
    bars = {}                                                             # beat 5: the two UCT terms per move
    for it in (9, 10, 11, 12):
        s = snaps[it]
        Nr = s[()][0]
        bars[it] = {m: (s[(m,)][1] / s[(m,)][0], UCT_C * math.sqrt(math.log(Nr) / s[(m,)][0])) for m in CANDS}
    for it in (9, 10, 11):                                                # the tallest bar is picked next
        best = max(order, key=lambda m: sum(bars[it][m]))
        assert log[it]["path"][0] == best, (it, best)
    return dict(snaps=snaps, log=log, order=order, counts=counts, wr=wr, share6=share6, bars=bars)


def demo_games():
    rng = random.Random(DEMO_SEED)
    games = {}
    for m in (6, 1):
        games[m] = []
        for _ in range(3):
            tr = []
            res = rollout(put(TOY_BOARD, m, "X"), "O", rng, tr)
            games[m].append((tr, res))
    g1 = games[1][0]
    assert g1[0][0][0] != 6 and g1[1] == "X"          # after X plays 1, O misses its win at 6 and X wins
    assert all(tr[0] == (6, "O") and r == "O" for tr, r in games[1][1:])   # when O does see it, X loses
    assert sum(SCORE_X[r] for _, r in games[6]) > sum(SCORE_X[r] for _, r in games[1])
    return games


def pi_data():
    rng = random.Random(PI_SEED)
    pts, inside, est = [], 0, {}
    for n in range(1, PI_CHECKS[-1] + 1):
        x, y = rng.random(), rng.random()
        hit = x * x + y * y <= 1.0
        inside += hit
        pts.append((x, y, hit))
        if n in PI_CHECKS:
            est[n] = (inside, 4 * inside / n)
    errs = [abs(est[n][1] - math.pi) for n in PI_CHECKS]
    assert errs[-1] < 0.05                                       # 5000 points: within 0.05 of pi
    assert all(a > b for a, b in zip(errs, errs[1:]))            # each checkpoint closer than the last
    assert f"{est[5000][1]:.2f}" == "3.15" or f"{est[5000][1]:.2f}" == "3.14"
    return pts, est


# ======================================================================================
# Go board (abstract): 18 + 18 stones of an opening, then move 37 (black, AlphaGo) as a fifth-line shoulder hit
# ======================================================================================
M37 = (14, 8)        # (column, row) with row 0 at the top; abstract position, not the game record
OPENING_B = [(3, 3), (2, 5), (5, 2), (15, 15), (13, 16), (16, 12), (15, 2), (16, 5), (3, 15), (2, 12),
             (5, 16), (9, 2), (9, 16), (12, 2), (2, 9), (6, 16), (16, 16), (11, 16)]
OPENING_W = [(16, 3), (13, 3), (14, 4), (15, 9), (16, 8), (3, 5), (4, 3), (3, 13), (4, 15), (14, 15),
             (15, 14), (10, 3), (2, 15), (8, 15), (3, 10), (5, 4), (13, 14), (16, 14)]
assert len(OPENING_B) == len(OPENING_W) == 18 and len(set(OPENING_B + OPENING_W)) == 36
assert M37 not in OPENING_B + OPENING_W and (15, 9) in OPENING_W[:10]   # the white stone it leans on


N_OPEN = 10          # 10 + 10 stones on the board before the move (abstract)


def opening_stones():
    st = []
    for i in range(N_OPEN):
        st += [(OPENING_B[i], "B"), (OPENING_W[i], "W")]
    return st


# ======================================================================================
# drawing helpers (sizes in scene units; `sw` scales the absolute stroke widths)
# ======================================================================================
def make_stone(color, r, sw=1.0):
    if color == "B":
        body = Circle(radius=r).set_fill(STONE_B, 1).set_stroke(STONE_B_RIM, 1.3 * sw, opacity=0.9)
        shine = Circle(radius=r * 0.32).set_fill(WHITE, 0.10).set_stroke(width=0).move_to([-0.35 * r, 0.38 * r, 0])
        return VGroup(body, shine)
    return VGroup(Circle(radius=r).set_fill(STONE_W, 1).set_stroke(STONE_W_RIM, 1.0 * sw))


def _rgba_image(A, color, half_w, half_h, seed):
    rng = np.random.default_rng(seed)
    A8 = np.clip(np.round(A * 255 + rng.uniform(-0.5, 0.5, A.shape)), 0, 255).astype(np.uint8)
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    img = np.zeros(A.shape + (4,), dtype=np.uint8)
    img[..., 0], img[..., 1], img[..., 2], img[..., 3] = r, g, b, A8
    m = ImageMobject(img)
    m.width = 2 * half_w
    return m


def glow_image(r, color, ppu, peak=0.42, n=7, seed=3):
    """One smooth radial glow (same colour, outer radius and centre opacity as the old 7-disc stack), dithered."""
    R = r * (1 + 1.6 * (n - 1) / n)
    a0 = 1 - (1 - peak / n) ** n
    half = R * 1.06
    W = max(16, int(round(2 * half * ppu)))
    xs = (np.arange(W) + 0.5) / W * 2 * half - half
    X, Y = np.meshgrid(xs, xs)
    rho = np.hypot(X, Y)
    A = a0 * (1 - smoothstep((rho - 0.4 * r) / (R - 0.4 * r)))
    return _rgba_image(A, color, half, half, seed)


def box_glow_image(w, h, cr, color, ppu, peak=0.27, sigma=0.06, seed=5):
    """A soft glow around a rounded rectangle (signed distance falloff), dithered."""
    pad = 4 * sigma
    hw, hh = w / 2 + pad, h / 2 + pad
    Wp, Hp = max(16, int(round(2 * hw * ppu))), max(16, int(round(2 * hh * ppu)))
    X, Y = np.meshgrid((np.arange(Wp) + 0.5) / Wp * 2 * hw - hw, (np.arange(Hp) + 0.5) / Hp * 2 * hh - hh)
    qx, qy = np.abs(X) - (w / 2 - cr), np.abs(Y) - (h / 2 - cr)
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - cr
    A = np.where(d > 0, peak * np.exp(-(d / sigma) ** 2), peak)
    return _rgba_image(A, color, hw, hh, seed)


class GoBoard(VGroup):
    MARGIN = 0.045

    def __init__(self, size, stones=(), sw=1.0, **kw):
        super().__init__(**kw)
        step = size * (1 - 2 * self.MARGIN) / 18
        self.panel = RoundedRectangle(width=size, height=size, corner_radius=size * 0.025)
        self.panel.set_fill(BOARD_FILL, 1).set_stroke(BOARD_EDGE, 1.2 * sw)
        self.hlines = VGroup(*[Line([-9 * step, (i - 9) * step, 0], [9 * step, (i - 9) * step, 0]) for i in range(19)])
        self.vlines = VGroup(*[Line([(i - 9) * step, -9 * step, 0], [(i - 9) * step, 9 * step, 0]) for i in range(19)])
        VGroup(self.hlines, self.vlines).set_stroke(BOARD_LINE, 1.1 * sw)
        self.hoshi = VGroup(*[Dot([(c - 9) * step, (9 - r) * step, 0], radius=step * 0.12, color=BOARD_LINE)
                              for c in (3, 9, 15) for r in (3, 9, 15)])
        self.stones = VGroup()
        for (c, r), col in stones:
            self.stones.add(make_stone(col, step * 0.47, sw).move_to([(c - 9) * step, (9 - r) * step, 0]))
        self.add(self.panel, self.hlines, self.vlines, self.hoshi, self.stones)

    @property
    def step(self):
        return self.panel.width * (1 - 2 * self.MARGIN) / 18

    def pt(self, c, r):
        return self.panel.get_center() + np.array([(c - 9) * self.step, (9 - r) * self.step, 0])


class TTTBoard(VGroup):
    """Tic-tac-toe board on a rounded panel; marks are submobjects (self.marks: cell -> mobject)."""
    INSET = 0.90

    def __init__(self, cells=".........", size=3.0, **kw):
        super().__init__(**kw)
        cs = size * self.INSET / 3
        self.panel = RoundedRectangle(width=size, height=size, corner_radius=size * 0.09)
        self.panel.set_fill(PANEL, 1).set_stroke(GRID, max(0.8, 1.8 * size / 3))
        g = VGroup(
            Line([-cs / 2, 1.5 * cs, 0], [-cs / 2, -1.5 * cs, 0]), Line([cs / 2, 1.5 * cs, 0], [cs / 2, -1.5 * cs, 0]),
            Line([-1.5 * cs, cs / 2, 0], [1.5 * cs, cs / 2, 0]), Line([-1.5 * cs, -cs / 2, 0], [1.5 * cs, -cs / 2, 0]))
        g.set_stroke(LINE, max(0.7, 2.6 * cs))
        self.grid = g
        self.marks = {}
        self.add(self.panel, self.grid)
        for i, v in enumerate(cells):
            if v in "XO":
                m = self.mark(i, v)
                self.marks[i] = m
                self.add(m)

    @property
    def cs(self):
        return self.panel.width * self.INSET / 3

    def cell(self, i):
        r, c = divmod(i, 3)
        return self.panel.get_center() + np.array([(c - 1) * self.cs, (1 - r) * self.cs, 0])

    def mark(self, i, p, opacity=1.0, scale=1.0):
        cs = self.cs * scale
        w = max(0.9, 7.0 * cs)
        if p == "X":
            a = 0.26 * cs
            m = VGroup(Line([-a, a, 0], [a, -a, 0]), Line([-a, -a, 0], [a, a, 0]))
            m.set_stroke(X_COL, w, opacity=opacity)
            for ln in m:
                ln.set_cap_style(CapStyleType.ROUND)
        else:
            m = Circle(radius=0.28 * cs).set_stroke(O_COL, w, opacity=opacity).set_fill(opacity=0)
        return m.move_to(self.cell(i))


def card(w=0.8, h=1.12, face=True, rank="A", suit="♠"):
    body = RoundedRectangle(width=w, height=h, corner_radius=w * 0.11)
    if face:
        body.set_fill(CARD_FACE, 1).set_stroke("#bdb8ad", 1.0)
        s = Text(suit, font="Helvetica Neue", font_size=w * 60, color="#1b1d22").move_to(body)
        r = Text(rank, font="Helvetica Neue", font_size=w * 30, color="#1b1d22")
        r.move_to(body.get_corner(UL) + np.array([0.17 * w, -0.15 * h, 0]))
        return VGroup(body, s, r)
    body.set_fill(CARD_BACK, 1).set_stroke("#4d5562", 1.0)
    inner = RoundedRectangle(width=w * 0.72, height=h * 0.78, corner_radius=w * 0.07)
    inner.set_stroke("#5c6574", 1.0).set_fill(opacity=0).move_to(body)
    return VGroup(body, inner)


def dice(size=0.8, face=5, color=TEXT, sw=1.0):
    body = RoundedRectangle(width=size, height=size, corner_radius=size * 0.2)
    body.set_fill(PANEL, 1).set_stroke(color, 2.2 * sw)
    q = size * 0.26
    spots = {3: [(-1, 1), (0, 0), (1, -1)], 5: [(-1, 1), (1, 1), (0, 0), (-1, -1), (1, -1)]}[face]
    return VGroup(body, VGroup(*[Dot([a * q, b * q, 0], radius=size * 0.07, color=color) for a, b in spots]))


def clock(r=0.4, color=TEXT, sw=1.0):
    face = Circle(radius=r).set_stroke(color, 2.4 * sw).set_fill(opacity=0)
    ticks = VGroup(*[Line(UP * r * 0.8, UP * r * 0.92).rotate(-k * PI / 6, about_point=ORIGIN).set_stroke(color, 1.6 * sw)
                     for k in range(12)])
    hour = Line(ORIGIN, UP * r * 0.48).rotate(-PI / 3, about_point=ORIGIN).set_stroke(color, 3 * sw)
    minute = Line(ORIGIN, UP * r * 0.74).set_stroke(color, 2.4 * sw)
    return VGroup(face, ticks, hour, minute)


def net_icon(w, h, color, sw=1.0):
    cols = (3, 4, 4, 2)
    xs = np.linspace(-w / 2, w / 2, len(cols))
    layers = [[np.array([x, y, 0]) for y in np.linspace(-h / 2, h / 2, k)] for x, k in zip(xs, cols)]
    wires = VGroup(*[Line(a, b) for L1, L2 in zip(layers, layers[1:]) for a in L1 for b in L2])
    wires.set_stroke(color, 0.9 * sw, opacity=0.45)
    dots = VGroup(*[Dot(p, radius=0.05 * max(sw, 0.55), color=color) for L in layers for p in L])
    return VGroup(wires, dots)


def gauge(r=0.36, frac=0.68, color=WIN_COL, sw=1.0):
    track = Arc(radius=r, start_angle=PI, angle=-PI).set_stroke(GRID, 6 * sw)
    val = Arc(radius=r, start_angle=PI, angle=-PI * frac).set_stroke(color, 6 * sw)
    ang = PI - PI * frac
    needle = Line(ORIGIN, r * 0.82 * np.array([math.cos(ang), math.sin(ang), 0])).set_stroke(TEXT, 2.6 * sw)
    return VGroup(track, val, needle, Dot(radius=0.045 * max(sw, 0.6), color=TEXT))


def dot_field_image(n=100, px=1000, lit=(37, 62)):
    """A 100 x 100 field of grey dots (10 000), transparent background; the lit cell is left empty."""
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pitch = px / n
    rad = pitch * 0.26
    for i in range(n):
        for j in range(n):
            if (i, j) != lit:
                cx, cy = (j + 0.5) * pitch, (i + 0.5) * pitch
                d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(142, 147, 156, 150))
    return np.array(img)


def points_image(pts, px, rad_px, upto, start=0):
    """Darts as an RGBA image of the unit square (inside the quarter circle teal, outside grey)."""
    ss = 2
    img = Image.new("RGBA", (px * ss, px * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = rad_px * ss
    for x, y, hit in pts[start:upto]:
        cx, cy = x * px * ss, (1 - y) * px * ss
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(92, 208, 179, 235) if hit else (142, 147, 156, 170))
    return np.array(img.resize((px, px), Image.LANCZOS))


def fan(start, ends, color, width, opacity):
    v = VMobject()
    for e in ends:
        v.start_new_path(np.array(start, dtype=float))
        v.add_line_to(np.array(e, dtype=float))
    return v.set_stroke(color, width, opacity=opacity).set_fill(opacity=0)


def waterfill(vals, total, minw):
    n = len(vals)
    if n == 0:
        return []
    if n * minw >= total - 1e-9:
        return [total / n] * n
    fixed = set()
    while True:
        free = total - minw * len(fixed)
        s = sum(v for i, v in enumerate(vals) if i not in fixed)
        ws = [minw if i in fixed else free * v / s for i, v in enumerate(vals)]
        low = {i for i, w in enumerate(ws) if i not in fixed and w < minw}
        if not low:
            return ws
        fixed |= low


def fade_ramp(g, factor, run_time=None):
    """fade_to for a ramped fan level: scales every gradient stop, keeping the ramp."""
    lines = list(g)
    if not hasattr(g, "_base_rgbas"):
        g._base_rgbas = [ln.get_stroke_rgbas().copy() for ln in lines]
    start = [ln.get_stroke_rgbas().copy() for ln in lines]

    def upd(_, a):
        for ln, b0, s0 in zip(lines, g._base_rgbas, start):
            r = s0.copy()
            r[:, 3] = s0[:, 3] + (b0[:, 3] * factor - s0[:, 3]) * a
            ln.stroke_rgbas = r
    kw = {} if run_time is None else {"run_time": run_time}
    return UpdateFromAlphaFunc(g, upd, rate_func=smooth, **kw)


def fade_level(m, factor, run_time=None):
    return fade_ramp(m, factor, run_time) if getattr(m, "_ramp", False) else fade_to(m, factor, run_time)


def fade_to(mob, factor, run_time=None):
    """Multiplicative opacity change relative to the first-seen opacity (keeps translucent fills)."""
    fam = [m for m in mob.get_family() if isinstance(m, VMobject)]
    for m in fam:
        if not hasattr(m, "_base_op"):
            m._base_op = (m.get_fill_opacity(), m.get_stroke_opacity())
    start = [(m.get_fill_opacity(), m.get_stroke_opacity()) for m in fam]

    def upd(_, a):
        for m, (f0, s0) in zip(fam, start):
            fb, sb = m._base_op
            m.set_fill(opacity=f0 + (fb * factor - f0) * a, family=False)
            m.set_stroke(opacity=s0 + (sb * factor - s0) * a, family=False)
    kw = {} if run_time is None else {"run_time": run_time}
    return UpdateFromAlphaFunc(mob, upd, rate_func=smooth, **kw)


def small_tree_icon(s=1.0, sw=1.0):
    pts = {0: (0, 0.45), 1: (-0.38, 0.0), 2: (0.0, 0.0), 3: (0.38, 0.0), 4: (-0.06, -0.45), 5: (0.2, -0.45)}
    edges = [(0, 1), (0, 2), (0, 3), (2, 4), (2, 5)]
    g = VGroup(*[Line([*pts[a], 0], [*pts[b], 0]).set_stroke(X_COL if a == 0 else O_COL, 3.5 * sw) for a, b in edges])
    g.add(*[Dot([*p, 0], radius=0.065, color=TEXT) for p in pts.values()])
    return g.scale(s)


def check_glyph(w, color, sw):
    """A drawn tick (no font): w = overall width."""
    m = VMobject().set_points_as_corners([[-0.5 * w, 0.02 * w, 0], [-0.16 * w, -0.32 * w, 0], [0.5 * w, 0.38 * w, 0]])
    m.set_stroke(color, sw).set_fill(opacity=0)
    m.set_cap_style(CapStyleType.ROUND)
    m.joint_type = LineJointType.ROUND
    return m


def cross_glyph(w, color, sw, opacity=1.0):
    a = 0.5 * w
    g = VGroup(Line([-a, a, 0], [a, -a, 0]), Line([-a, -a, 0], [a, a, 0])).set_stroke(color, sw, opacity=opacity)
    for ln in g:
        ln.set_cap_style(CapStyleType.ROUND)
    return g


def x_glyph(w, sw):
    return cross_glyph(w, X_COL, sw)


def o_glyph(w, sw):
    return Circle(radius=0.5 * w).set_stroke(O_COL, sw).set_fill(opacity=0)


def mini_card(w, h, face):
    """A card without any glyphs (abstract, no text to read)."""
    body = RoundedRectangle(width=w, height=h, corner_radius=w * 0.12)
    if face:
        return VGroup(body.set_fill(CARD_FACE, 1).set_stroke("#bdb8ad", 0.9))
    body.set_fill(CARD_BACK, 1).set_stroke("#4d5562", 0.9)
    inner = RoundedRectangle(width=w * 0.7, height=h * 0.76, corner_radius=w * 0.08).set_stroke("#5c6574", 0.8)
    return VGroup(body, inner.set_fill(opacity=0).move_to(body))


GAME_SEED, GAME_P = 1946, 0.3    # the 100 illustrative solitaire games (labelled 示意; not a real win rate)
FACT52 = math.factorial(52)
assert 8.0e67 <= FACT52 < 8.1e67 and round(FACT52 / 1e67, 2) == 8.07      # "约 8×10^67 种洗法"


def solitaire_games():
    rng = random.Random(GAME_SEED)
    res = [rng.random() < GAME_P for _ in range(100)]
    cum = [0]
    for ok in res:
        cum.append(cum[-1] + ok)
    assert 0 < cum[-1] < 100
    return res, cum


# ======================================================================================
# the MCTS tree drawn from a frozen snapshot
# ======================================================================================
def tree_layout(snap, cfg):
    total = snap[()][0]
    thr = max(cfg["thr_min"], cfg["thr_frac"] * total)
    kids1 = [(m,) for m in CANDS if (m,) in snap]
    W = cfg["x1"] - cfg["x0"]
    ws = waterfill([snap[k][0] for k in kids1], W, cfg["min_slot"]) if cfg["tilt"] else [W / len(kids1)] * len(kids1)
    pos = {(): np.array([cfg["root"][0], cfg["root"][1], 0.0])}
    span = {}
    x = cfg["x0"]
    for k, w in zip(kids1, ws):
        span[k] = (x, x + w)
        pos[k] = np.array([x + w / 2, cfg["child_y"], 0.0])
        x += w

    def rec(key, depth):
        if depth > cfg["max_depth"]:
            return
        kids = [key + (m,) for m in range(9) if key + (m,) in snap and snap[key + (m,)][0] >= thr]
        if not kids:
            return
        a, b = span[key]
        xx = a
        for k, w in zip(kids, waterfill([snap[k][0] for k in kids], b - a, cfg["min_span"])):
            span[k] = (xx, xx + w)
            pos[k] = np.array([xx + w / 2, cfg["levels"][depth], 0.0])
            xx += w
            rec(k, depth + 1)

    for k in kids1:
        rec(k, 2)
    return pos, total


def tree_anchor(key, p, cfg, top):
    if key == ():
        return p + DOWN * cfg["root_size"] / 2
    if len(key) == 1:
        return p + (UP if top else DOWN) * cfg["child_size"] / 2
    return p


def tree_edge(key, pos, snap, total, cfg):
    a = tree_anchor(key[:-1], pos[key[:-1]], cfg, top=False)
    b = tree_anchor(key, pos[key], cfg, top=True)
    w = float(np.clip(cfg["ew"] * snap[key][0] / total, cfg["ew_min"], cfg["ew"]))
    col = X_COL if snap[key][2] == "X" else O_COL
    return Line(a, b, stroke_width=w, color=col, stroke_opacity=0.88).set_z_index(1)


def tree_dot(key, pos, snap, total, cfg):
    r = float(np.clip(cfg["rmax"] * 1.25 * math.sqrt(snap[key][0] / total), cfg["rmin"], cfg["rmax"]))
    col = X_COL if snap[key][2] == "X" else O_COL
    return Circle(radius=r).set_fill(col, 1).set_stroke(BG, 1.5 * cfg["sw"]).move_to(pos[key]).set_z_index(3)


def tree_icon(snap, cfg, root_cells, sw=1.0):
    pos, total = tree_layout(snap, cfg)
    g = VGroup(*[tree_edge(k, pos, snap, total, cfg) for k in pos if k])
    g.add(*[tree_dot(k, pos, snap, total, cfg) for k in pos if len(k) >= 2])
    g.add(TTTBoard(root_cells, size=cfg["root_size"]).move_to(pos[()]).set_z_index(5))
    g.add(*[TTTBoard(put(TOY_BOARD, m, "X"), size=cfg["child_size"]).move_to(pos[(m,)]).set_z_index(5) for m in CANDS])
    return g


def teaser_tree(snap, root_pt, x0, x1, levels, sw=1.0, ew=16, glow_col=WIN_COL):
    """The real 2000-round tree, drawn as bare branches growing out of a point (schematic use: label it)."""
    cfg = dict(x0=x0, x1=x1, root=root_pt[:2], root_size=0.0, child_y=levels[1], child_size=0.0,
               levels={d: levels[d] for d in range(2, len(levels))}, max_depth=len(levels) - 1, tilt=True,
               min_slot=(x1 - x0) / 9, min_span=(x1 - x0) / 30, thr_frac=0.02, thr_min=2, ew=ew, ew_min=1.2 * sw,
               rmin=0.04 * sw ** 0.5, rmax=0.09 * sw ** 0.5, sw=sw)
    pos, total = tree_layout(snap, cfg)
    edges, dots, hot = VGroup(), VGroup(), VGroup()
    for k in pos:
        if not k:
            continue
        e = tree_edge(k, pos, snap, total, cfg)
        on = k[0] == 6
        e.set_stroke(glow_col if on else GREY_B, opacity=0.95 if on else 0.55)
        (hot if on else edges).add(e)
        d = tree_dot(k, pos, snap, total, cfg).set_fill(glow_col if on else GREY_B, 1)
        dots.add(d)
    glow = VGroup()                     # no translucent tube around the trunk (it read as a hard-edged band)
    g = VGroup(glow, edges, hot, dots)
    g.glow, g.edges, g.hot, g.dots = glow, edges, hot, dots
    return g


def dim_board(go, factor):
    """Dim a board: lines and star points by opacity, stones by colour (they stay opaque, nothing shows through)."""
    dim_now(VGroup(go.hlines, go.vlines, go.hoshi), factor)
    for stone in go.stones:
        body = stone[0]
        body.set_fill(interpolate_color(body.get_fill_color(), ManimColor(BOARD_FILL), 1 - factor), opacity=1)
        body.set_stroke(interpolate_color(body.get_stroke_color(), ManimColor(BOARD_FILL), 1 - factor))
        for extra in stone[1:]:
            extra.set_fill(opacity=extra.get_fill_opacity() * factor)
    return go


def dim_now(mob, factor):
    for m in mob.get_family():
        if isinstance(m, VMobject):
            m._base_op = (m.get_fill_opacity(), m.get_stroke_opacity())
            m.set_fill(opacity=m.get_fill_opacity() * factor, family=False)
            m.set_stroke(opacity=m.get_stroke_opacity() * factor, family=False)
    return mob


CLEAN = os.environ.get("ALPHAGO_CLEAN") == "1"     # stills for covers: captions and tags invisible
EM = 0.01366        # Text / MathTex em (one CJK advance) in scene units per font-size point


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def gradient_image(alpha_fn, x0, x1, y0, y1, ppu, color=BG, seed=7):
    """One smooth BG-coloured overlay (RGBA) with ~1/255 alpha dither, instead of stacked translucent rectangles."""
    W, H = max(2, int(round((x1 - x0) * ppu))), max(2, int(round((y1 - y0) * ppu)))
    xs = x0 + (np.arange(W) + 0.5) / W * (x1 - x0)
    ys = y1 - (np.arange(H) + 0.5) / H * (y1 - y0)
    X, Y = np.meshgrid(xs, ys)
    A = np.clip(alpha_fn(X, Y), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    A8 = np.clip(np.round(A * 255 + rng.uniform(-0.5, 0.5, A.shape)), 0, 255).astype(np.uint8)
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    img = np.zeros((H, W, 4), dtype=np.uint8)
    img[..., 0], img[..., 1], img[..., 2], img[..., 3] = r, g, b, A8
    m = ImageMobject(img)
    m.width = x1 - x0
    m.move_to([(x0 + x1) / 2, (y0 + y1) / 2, 0])
    return m


def set_fan(m, a, lag=0.0):
    S, E = m._S, m._E
    if lag > 0:
        prog = np.clip((a - lag * m._order) / (1 - lag), 0.002, 1.0)[:, None]
    else:
        prog = max(a, 0.002)
    P = S + (E - S) * prog
    if getattr(m, "_ramp", False):            # one Line per branch (each carries its own opacity ramp)
        for ln, s_, p_ in zip(m, S, P):
            ln.set_points_as_corners([s_, p_])
        return
    pts = np.empty((len(P) * 4, 3))
    pts[0::4], pts[1::4], pts[2::4], pts[3::4] = S, S + (P - S) / 3, S + 2 * (P - S) / 3, P
    m.set_points(pts)


def fan_grow(m, rt, lag=0.6):
    return UpdateFromAlphaFunc(m, lambda mob, a: set_fan(mob, ease_out_cubic(a), lag), run_time=rt)


def fractal_fan(origin, a0, a1, counts=(250, 3, 2), lengths=(2.5, 1.0, 0.45), spreads=(0.0, 0.34, 0.26),
                colors=(X_COL, O_COL, X_COL), widths=(0.8, 0.5, 0.42), ops=(0.2, 0.11, 0.08), seed=0, jitter=0.15):
    """Hairline fan out of a point; every tip fans again (3 levels). Returns (level mobjects, level-1 ends, angles)."""
    rng = np.random.default_rng(seed)
    origin = np.array(origin, dtype=float)
    angs = np.linspace(a0, a1, counts[0])
    S = np.repeat(origin[None, :], counts[0], axis=0)
    L = lengths[0] * (1 + jitter * rng.uniform(-1, 1, counts[0]))
    E = S + np.stack([np.cos(angs) * L, np.sin(angs) * L, np.zeros_like(angs)], axis=1)
    levels = [(S, E, angs)]
    for k in (1, 2):
        Sp, Ep, ap = levels[-1]
        offs = np.linspace(-spreads[k], spreads[k], counts[k])
        A = (ap[:, None] + offs[None, :]).ravel()
        Sn = np.repeat(Ep, counts[k], axis=0)
        Lk = lengths[k] * (1 + jitter * rng.uniform(-1, 1, len(A)))
        En = Sn + np.stack([np.cos(A) * Lk, np.sin(A) * Lk, np.zeros_like(A)], axis=1)
        levels.append((Sn, En, A))
    mobs = []
    for k, (S_, E_, A_) in enumerate(levels):
        if k == 0:
            # the first level converges on one point: ramp every line from ~0.05 at the root to full at 30 %
            stops = 11
            opr = [ops[0] * min(1.0, 0.25 + 0.75 * (j / (stops - 1)) / 0.3) for j in range(stops)]
            m = VGroup()
            for s_, e_ in zip(S_, E_):
                ln = Line(s_, e_).set_fill(opacity=0)
                ln.set_stroke(color=[colors[0]] * stops, width=widths[0], opacity=opr)
                ln.set_sheen_direction((e_ - s_) / np.linalg.norm(e_ - s_))
                m.add(ln)
            m._ramp = True
        else:
            m = VMobject().set_fill(opacity=0).set_stroke(colors[k], widths[k], opacity=ops[k])
        m._S, m._E = S_, E_
        m._order = np.argsort(np.argsort(A_, kind="stable")) / max(1, len(A_) - 1)
        set_fan(m, 1.0)
        mobs.append(m)
    return mobs, E, angs


# ======================================================================================
# shared scene base: text tracks with enforced reading time, rule-checked sound effects, voice hooks
# ======================================================================================
SFX_TIER = dict(stone_big="accent", boom="accent", success="accent", chime="accent",
                stone="event", question="event", whoosh="event", riser="event", dice="event", shuffle="event",
                pop="small", card="small", tick="small", rain="bed", whoosh_soft="bed", ambient="bed")


def cjk_units(text):
    """CJK characters + 5/8 per other visible character (so reading_time = max(2, 1 + n/5))."""
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    oth = sum(1 for ch in text if not ch.isspace() and not ("一" <= ch <= "鿿") and ch not in "，。：；、！？“”（）,.:;!?()")
    return cjk + oth * 5 / 8


class Track:
    def __init__(self, text, mob, aha, slack):
        self.text, self.mob, self.aha, self.slack = text, mob, aha, slack
        self.base = reading_time(text)
        self.watch = 0.0
        self.t_on = None
        self.t_off = None

    @property
    def required(self):
        return self.base + self.watch + self.aha


class Base(Scene):
    SCENE_ID = ""
    VOICE_JSON = ""
    T_SIZE = {"title": 46, "cap": 30, "label": 22}     # the three running type sizes (CJK)
    N_SIZE = {"title": 62, "cap": 40, "label": 30}     # matching sizes for numbers (MathTex / DecimalNumber)
    REF_PPU = 1920 / (8.0 * 16 / 9)                    # pixels per unit of the delivery resolution
    PX_FLOOR, TAG_FLOOR = 24, 20                       # legibility floor (px at the delivery resolution)
    SW = 1.0
    CAP_Y = -3.5
    CAP_MAXW = 12.6
    MAX_OVERHOLD = 1.0

    def __init__(self, **kw):
        # Manim 0.22's -r only sets pixels; derive the frame width from the pixel aspect (vertical cut: 4.5 units)
        config.frame_width = config.frame_height * config.pixel_width / config.pixel_height
        super().__init__(**kw)

    def init_common(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.here = here
        self.voice_dir = os.path.join(here, "..", "voice", self.SCENE_ID)
        with open(os.path.join(here, self.VOICE_JSON), encoding="utf-8") as fh:
            self.voice_keys = [ln["key"] for ln in json.load(fh)["lines"]]
        self.voice_state = {"end": 0.0}
        self.tracks_open, self.pend_on, self.pend_off, self.rows = [], [], [], []
        self.track_of = {}
        self.cap = None
        self.sfx_log, self.sfx_skipped = [], []
        self.marks = []
        self.min_px = 1e9

    # ---------- type (every text goes through a size check) ----------
    def px(self, f):
        return f * EM * self.REF_PPU

    def floor_ok(self, f, s):
        p = self.px(f)
        floor = self.TAG_FLOOR if s.strip() == "示意" else self.PX_FLOOR
        assert p >= floor - 1e-6, (s, round(p, 1), floor)
        self.min_px = min(self.min_px, p)

    def T(self, s, kind="cap", color=TEXT, **kw):
        assert "?" not in s, "use the full-width ？ of the CJK font"
        f = self.T_SIZE[kind] * kw.pop("k", 1.0)
        self.floor_ok(f, s)
        return cn(s, size=f, color=color, **kw)

    def H(self, s, px, color=TEXT, weight=NORMAL, **kw):
        """Hero text sized in delivery pixels (one CJK advance = px)."""
        assert "?" not in s, "use the full-width ？ of the CJK font"
        f = px / (EM * self.REF_PPU)
        self.floor_ok(f, s)
        return cn(s, size=f, color=color, weight=weight, **kw)

    def HN(self, s, px, color=TEXT):
        f = px / (EM * self.REF_PPU)
        self.floor_ok(f, s)
        return MathTex(s, font_size=f, color=color)

    def N(self, s, kind="cap", color=TEXT):
        self.floor_ok(self.N_SIZE[kind], s)
        return MathTex(s, font_size=self.N_SIZE[kind], color=color)

    def num(self, v, kind="cap", decimals=0, color=TEXT, px=None):
        f = self.N_SIZE[kind] if px is None else px / (EM * self.REF_PPU)
        self.floor_ok(f, str(v))
        if decimals == 0:
            return Integer(v, font_size=f, color=color, group_with_commas=False)
        return DecimalNumber(v, num_decimal_places=decimals, font_size=f, color=color, group_with_commas=False)

    def markup(self, s, kind="cap", color=TEXT):
        """CJK text with 10^n written as a real superscript."""
        self.floor_ok(self.T_SIZE[kind] * 0.6, s)                      # the superscript digits
        m = re.sub(r"10\^(\d+)", r"10<sup>\1</sup>", s)
        return MarkupText(m, font=CN_FONT, font_size=self.T_SIZE[kind], color=color)

    # ---------- optional narration (same mechanism as log_trick.py) ----------
    def voice(self, key, gap=0.15):
        assert key in self.voice_keys, key
        path = os.path.join(self.voice_dir, key + ".wav")
        if not os.path.exists(path):
            return
        with wave.open(path) as w:
            dur = w.getnframes() / w.getframerate()
        remain = self.voice_state["end"] - self.time
        if remain > 0:
            self.wait(remain + gap)
        self.add_sound(path)
        self.voice_state["end"] = self.time + dur

    def voice_flush(self, tail=0.3):
        remain = self.voice_state["end"] - self.time
        if remain > 0:
            self.wait(remain + tail)

    def mark(self, name):
        self.marks.append((name, self.time))
        print(f"[MARK] {name} {self.time:.2f}", file=sys.stderr)

    def frame_mark(self, name):
        print(f"[FRAME] {name} {self.renderer.num_plays}", file=sys.stderr)

    # ---------- text tracks ----------
    def show(self, mob, text, aha=0.0, f_end=1.0, slack=False):
        """mob (a caption or text card) fades in during the next play; its fade-in ends at fraction f_end."""
        tr = Track(text, mob, aha, slack)
        self.pend_on.append((tr, f_end))
        self.track_of[id(mob)] = tr
        return mob

    def retire(self, *mobs):
        for m in mobs:
            tr = self.track_of.pop(id(m), None)
            if tr is not None:
                self.pend_off.append(tr)

    def hide(self, *mobs, shift=UP * 0.12):
        self.retire(*mobs)
        return [FadeOut(m, shift=shift) for m in mobs]

    def _settle(self, tracks):
        rem = max([tr.t_on + tr.required - self.time for tr in tracks], default=0.0)
        if rem > 1e-4:
            self.wait(rem + 1.0 / config.frame_rate)       # waits are whole frames: round up

    def settle(self):
        self._settle(self.tracks_open)

    def play(self, *anims, watch=True, **kw):
        if anims and all(isinstance(a, Wait) for a in anims):
            return super().play(*anims, **kw)
        assert len(self.pend_on) <= 1, [t.text for t, _ in self.pend_on]
        need = list(self.tracks_open) if self.pend_on else list(self.pend_off)   # one reading target at a time
        self._settle(need)
        for tr in need:                       # nothing new appears (or leaves) before its reader is done
            assert self.time >= tr.t_on + tr.required - 1e-3, (tr.text, self.time - tr.t_on, tr.required)
        t0 = self.time
        super().play(*anims, **kw)
        t1 = self.time
        leaving = {id(tr) for tr in self.pend_off}
        for tr in self.tracks_open:
            if watch and id(tr) not in leaving:
                tr.watch += 0.5 * max(0.0, t1 - max(t0, tr.t_on))
        for tr in self.pend_off:
            tr.t_off = t0
            self.tracks_open.remove(tr)
            self.rows.append(tr)
        for tr, f in self.pend_on:
            tr.t_on = t0 + (t1 - t0) * f
            if watch:
                tr.watch += 0.5 * (t1 - tr.t_on)
            self.tracks_open.append(tr)
        self.pend_on, self.pend_off = [], []

    # ---------- captions: one fixed band, one style ----------
    def cap_mob(self, text, kind="cap"):
        lines = text.split("\n")
        assert len(lines) <= 2 and all(len(re.sub(r"\s|\^", "", ln)) <= 18 for ln in lines), text
        g = VGroup(*[self.markup(ln, kind) if "^" in ln else self.T(ln, kind) for ln in lines])
        g.arrange(DOWN, buff=0.14)
        assert g.width <= self.CAP_MAXW, (text, g.width)
        g.move_to([0, self.CAP_Y, 0]).set_z_index(20)
        return self.clean(g)

    def clean(self, mob):
        """Caption-like text: invisible when rendering clean stills."""
        if CLEAN:
            mob.set_opacity(0)
        return mob

    def say(self, text, aha=0.0, kind="cap"):
        self.play(*self.say_anims(text, aha=aha, kind=kind, rt=0.36), run_time=0.36, watch=False)

    def say_anims(self, text, aha=0.0, rt=0.7, kind="cap", mob=None):
        """Caption swap (old one out, new one in). mob: a pre-built caption placed elsewhere (hero panel / top)."""
        new = self.cap_mob(text, kind) if mob is None else self.clean(mob)
        old, self.cap = self.cap, new
        f = min(1.0, 0.36 / rt)
        anims = []
        if old is not None:
            self.retire(old)
            anims.append(FadeOut(old, shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0, 0.5 * f)))
        anims.append(FadeIn(new, shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0.45 * f if old else 0, f)))
        self.show(new, text.replace("\n", "").replace("^", ""), aha=aha, f_end=f)
        return anims

    def unsay_anims(self):
        old, self.cap = self.cap, None
        if old is None:
            return []
        self.retire(old)
        return [FadeOut(old, shift=UP * 0.1)]

    # ---------- sound effects (style.sfx + README rules) ----------
    def sfx(self, name, offset=0.0, extra_db=0.0):
        assert extra_db <= 0.0, "never raise a gain above gains.json"
        t = self.time + offset
        tier = SFX_TIER[name]
        log = self.sfx_log
        if tier == "event" and any(tr == "event" and abs(t - u) < 0.8 for u, _, tr in log):
            self.sfx_skipped.append((round(t, 2), name, "event < 0.8 s"))
            return
        if tier == "small" and any(tr == "accent" and abs(t - u) < 0.5 for u, _, tr in log):
            self.sfx_skipped.append((round(t, 2), name, "near accent"))
            return
        if name == "tick" and any(n == "tick" and abs(t - u) < 0.25 for u, n, _ in log):
            self.sfx_skipped.append((round(t, 2), name, "tick > 4/s"))
            return
        style_sfx(self, name, offset, extra_db)
        log.append((t, name, tier))
        print(f"[CUE] {name} {t:.3f}", file=sys.stderr)

    def check_sfx(self):
        acc = sorted(u for u, _, tr in self.sfx_log if tr == "accent")
        ev = sorted(u for u, _, tr in self.sfx_log if tr == "event")
        assert len(acc) <= 5, acc
        assert all(b - a >= 20.0 for a, b in zip(acc, acc[1:])), acc
        assert all(b - a >= 0.8 for a, b in zip(ev, ev[1:])), ev
        assert not any(tr == "small" and any(abs(u - a) < 0.5 for a in acc) for u, _, tr in self.sfx_log)
        counts = {}
        for _, n, tr in self.sfx_log:
            counts.setdefault(tr, {}).setdefault(n, 0)
            counts[tr][n] += 1
        print("[SFX] " + json.dumps(counts, ensure_ascii=False), file=sys.stderr)
        print("[SFX] accents at " + ", ".join(f"{a:.1f}" for a in acc), file=sys.stderr)
        if self.sfx_skipped:
            print("[SFX] skipped " + str(self.sfx_skipped), file=sys.stderr)

    def ambient_bed(self):
        sys.path.insert(0, SFX_DIR)
        from make_sfx import write_bed
        with open(os.path.join(SFX_DIR, "gains.json")) as fh:
            gain = json.load(fh)["ambient"]
        bed = os.path.join(SFX_DIR, f"bed_{type(self).__name__}.wav")
        write_bed(self.time, bed)
        self.add_sound(bed, time_offset=-self.time, gain=gain)

    # ---------- the end: close tracks, write the CSV, assert ----------
    def finish(self):
        assert not self.pend_on and not self.pend_off
        for tr in list(self.tracks_open):
            tr.t_off = self.time
            self.rows.append(tr)
        self.tracks_open = []
        path = os.path.join(self.here, f"{self.SCENE_ID}.captions.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["text", "t_on", "t_off", "n", "required", "actual"])
            for tr in sorted(self.rows, key=lambda r: r.t_on):
                w.writerow([tr.text, f"{tr.t_on:.2f}", f"{tr.t_off:.2f}", f"{cjk_units(tr.text):.1f}",
                            f"{tr.required:.2f}", f"{tr.t_off - tr.t_on:.2f}"])
        over = [(tr.text, round(tr.t_off - tr.t_on - tr.required, 2)) for tr in self.rows
                if not tr.slack and tr.t_off - tr.t_on - tr.required > self.MAX_OVERHOLD + 1e-3]
        for tr in self.rows:
            assert tr.t_off - tr.t_on >= tr.required - 1e-3, (tr.text, tr.t_off - tr.t_on, tr.required)
        assert not over, ("captions held > 1 s past their need", over)
        print(f"[REPORT] {len(self.rows)} text tracks -> {path}; smallest text {self.min_px:.1f} px", file=sys.stderr)
        self.check_sfx()
        self.ambient_bed()
        print("[REPORT] duration %.2f s" % self.time, file=sys.stderr)
        for (a, t0), (b, t1) in zip(self.marks, self.marks[1:]):
            print(f"[REPORT] {a:10s} {t0:7.2f} -> {t1:7.2f}  ({t1 - t0:5.1f} s)", file=sys.stderr)

    # ---------- shared moves ----------
    def slam(self, st, p, step, halo, shake=None, drop=0.32, n_rings=1, extra=None, rt=0.9):
        """The move-37 stone slams down: stone_big lands on the touch-down frame, one ripple ring spreads.
        extra: a function returning animations for the ripple play (called after the drop, so any text it shows
        is registered for the play it really fades in with)."""
        st.save_state()
        st.scale(2.6).set_opacity(0).shift(UP * 0.25)
        self.sfx("stone_big", offset=drop)
        self.play(Restore(st), run_time=drop, rate_func=rush_into, watch=False)
        extra = extra() if extra is not None else []
        rings = VGroup(*[Circle(radius=step * 0.5).set_stroke(X_COL, 3.5 * self.SW).move_to(p).set_z_index(6)
                         for _ in range(n_rings)])
        self.add(rings)
        anims = [LaggedStart(*[r.animate(rate_func=ease_out_cubic).scale(7).set_stroke(opacity=0) for r in rings],
                             lag_ratio=0.3), FadeIn(halo, scale=0.6)]
        if shake is not None:
            anims.append(VGroup(shake, st).animate(rate_func=there_and_back).shift(DOWN * 0.04))
        self.play(*anims, *extra, run_time=rt, watch=False)
        self.remove(rings)

    def ghost_marks(self, board, trace, opacity=0.62, scale=0.92, z=7):
        return [board.mark(i, p, opacity=opacity, scale=scale).set_z_index(z) for i, p in trace]

    def chip(self, res, h=0.40):
        txt, col = {"X": ("赢", WIN_COL), "D": ("平", DRAW_COL), "O": ("输", THREAT_COL)}[res]
        box = RoundedRectangle(width=h * 1.55, height=h, corner_radius=h * 0.32)
        box.set_fill(col, 0.18).set_stroke(col, 1.6 * self.SW)
        return VGroup(box, self.T(txt, "label", col).move_to(box))

    def stamp(self, res, r):
        txt, col = {"X": ("赢", WIN_COL), "D": ("平", DRAW_COL), "O": ("输", THREAT_COL)}[res]
        disc = Circle(radius=r).set_fill(BG, 0.9).set_stroke(col, 3 * self.SW)
        return VGroup(disc, self.T(txt, "title", col, weight=MEDIUM).move_to(disc))

    def counter(self, n, w, centre, z=6):
        r1 = VGroup(self.T("试", "label"), self.num(n, "label"))
        r2 = VGroup(self.T("赢", "label", WIN_COL), self.num(w, "label", 1, WIN_COL))
        for r in (r1, r2):
            r.arrange(RIGHT, buff=0.1)
        g = VGroup(r1, r2).arrange(DOWN, buff=0.06, aligned_edge=LEFT).move_to(centre).set_z_index(z)
        g.vals = [n, w]
        return g

    def tick_anims(self, ctr, n, w):
        anims = []
        for row, val in ((0, n), (1, w)):
            d = ctr[row][1]
            if ctr.vals[row] != val:
                anims.append(ChangeDecimalToValue(d, val))
            else:
                anims.append(Indicate(d, color=d.get_color(), scale_factor=1.2))
        ctr.vals = [n, w]
        return anims

    def focus_ctrs(self, keys, rt=None):
        """Counters: the active branch at full strength, every other one dimmed to 0.3."""
        return [fade_to(c, 1.0 if k in keys else 0.3, run_time=rt) for k, c in self.ctrs.items()]

    def hero_lines(self, text, px, color=TEXT, weight=NORMAL, buff=0.16, aligned_edge=ORIGIN):
        """Caption-rule text (<= 2 lines, <= 18 characters each) set at a hero size."""
        lines = text.split("\n")
        assert len(lines) <= 2 and all(len(re.sub(r"\s", "", ln)) <= 18 for ln in lines), text
        g = VGroup(*[self.H(ln, px, color, weight) for ln in lines])
        return g.arrange(DOWN, buff=buff, aligned_edge=aligned_edge)

    def legend(self, kind="label", k=1.0, gw=0.3, sw=5.0, horizontal=False):
        """Who is who: orange X = us, blue O = the opponent."""
        r1 = VGroup(x_glyph(gw, sw), self.T("X：我们", kind, k=k)).arrange(RIGHT, buff=0.16)
        r2 = VGroup(o_glyph(gw, sw), self.T("O：对手", kind, k=k)).arrange(RIGHT, buff=0.16)
        if horizontal:
            return VGroup(r1, r2).arrange(RIGHT, buff=0.45)
        return VGroup(r1, r2).arrange(DOWN, buff=0.22, aligned_edge=LEFT)

    def turn_badge(self, kind="title", gw=0.42, sw=7.0):
        t = self.T("轮到 X", kind, X_COL, weight=MEDIUM)
        inner = VGroup(x_glyph(gw, sw), t).arrange(RIGHT, buff=0.2)
        h = inner.height + 0.3
        pill = RoundedRectangle(width=inner.width + 0.5, height=h, corner_radius=h / 2)
        pill.set_fill(X_COL, 0.14).set_stroke(X_COL, 2.2 * self.SW)
        return VGroup(pill, inner.move_to(pill))

    def callout(self, ctr, row, text, color, left, kind="cap"):
        """A text note for one counter row with a thin leader line; returns (mobject, animations); it is a track."""
        t = self.T(text, kind, color).move_to(left, aligned_edge=LEFT).set_z_index(12)
        tip = ctr[row].get_right() + RIGHT * 0.1
        lead = Line(t.get_left() + LEFT * 0.12, tip).set_stroke(color, 1.4 * self.SW, opacity=0.85).set_z_index(12)
        dot = Dot(tip, radius=0.035, color=color).set_z_index(12)
        g = VGroup(t, lead, dot)
        self.show(t, text)
        return g, [FadeIn(t, shift=LEFT * 0.08), Create(lead), FadeIn(dot)]

    def step_badges(self, r=0.28, kind="cap", buff=0.42):
        """The four steps 选 → 长 → 试 → 传 as a compact row (to light the step a network upgrades)."""
        row = VGroup()
        for ch in "选长试传":
            b = Circle(radius=r).set_fill(PANEL, 1).set_stroke(GREY_B, 1.6 * self.SW)
            row.add(VGroup(b, self.T(ch, kind, weight=MEDIUM).move_to(b)))
        row.arrange(RIGHT, buff=buff)
        arrows = VGroup(*[Arrow(row[k][0].get_right(), row[k + 1][0].get_left(), buff=0.05, stroke_width=2 * self.SW,
                                color=GRID, max_tip_length_to_length_ratio=0.3) for k in range(3)])
        return VGroup(row, arrows)

    def badge_light(self, badges, k):
        anims = []
        for j, b in enumerate(badges[0]):
            on = j == k
            anims += [b[0].animate.set_fill(TEXT if on else PANEL, 1).set_stroke(TEXT if on else GREY_B, 1.6 * self.SW),
                      b[1].animate.set_color(BG if on else TEXT)]
        return anims

    # ---------- slow, readable play-outs (round 5) ----------
    def drop_read_cap(self):
        """During long play-outs a caption leaves as soon as it has been read (no over-hold)."""
        if self.cap is None:
            return []
        tr = self.track_of.get(id(self.cap))
        if tr is None or tr.t_on is None or self.time < tr.t_on + tr.required - 1e-3:
            return []
        return self.unsay_anims()

    def hold(self, t):
        """A static hold of t seconds on a finished board; a caption that is read meanwhile leaves."""
        t_end = self.time + t
        tr = self.track_of.get(id(self.cap)) if self.cap is not None else None
        if tr is not None and tr.t_on is not None and tr.t_on + tr.required < t_end - 0.4:
            due = tr.t_on + tr.required
            if due > self.time:
                self.wait(due - self.time + 1.0 / config.frame_rate)
            self.play(*self.unsay_anims(), run_time=0.3, watch=False)
        if t_end - self.time > 1e-3:
            self.wait(t_end - self.time)

    def play_moves(self, board, trace, per=0.45, z=11, extra=None):
        """The seeded random game, one move per beat in the X / O colours."""
        marks = []
        for j, (i, q) in enumerate(trace):
            m = board.mark(i, q).set_z_index(z)
            self.sfx("tick", offset=0.05, extra_db=-6.0)
            more = extra if (extra is not None and j == 0) else []
            self.play(FadeIn(m, scale=1.4, rate_func=squish_rate_func(smooth, 0, 0.55)), *more, *self.drop_read_cap(),
                      run_time=per)
            marks.append(m)
        return marks

    def win_capsules(self, board, cells, res, sw):
        """Capsules around the winning three on the real final board (asserted to be a line of the winner)."""
        assert winner(cells) == res, (cells, res)
        lines = [L for L in LINES3 if cells[L[0]] != "." and cells[L[0]] == cells[L[1]] == cells[L[2]]]
        caps = VGroup()
        if res == "D":
            assert not lines
            return caps
        assert lines and all(all(cells[c] == res for c in L) and L in LINES3 for L in lines), (cells, lines)
        col = WIN_COL if res == "X" else THREAT_COL
        for L in lines:
            a, c = board.cell(L[0]), board.cell(L[2])
            v = c - a
            w = board.cs * 0.8
            cap = RoundedRectangle(width=np.linalg.norm(v) + w, height=w, corner_radius=0.499 * w)
            cap.set_stroke(col, sw).set_fill(col, 0.10)
            caps.add(cap.rotate(math.atan2(v[1], v[0])).move_to((a + c) / 2))
        return caps

    def end_game(self, board, cells, res, stamp_pos, r, sw, z=12, cap_anims=None):
        """Capsule (or none for a draw), then the result stamp, then a hold on the full final board."""
        caps = self.win_capsules(board, cells, res, sw).set_z_index(z - 1)
        if len(caps):
            self.play(*[Create(c) for c in caps], *self.drop_read_cap(), run_time=0.5)
        sp = self.stamp(res, r).move_to(stamp_pos).set_z_index(z)
        self.sfx("pop", offset=0.1)
        self.play(FadeIn(sp, scale=1.4), *(cap_anims or self.drop_read_cap()), run_time=0.35)
        self.hold(1.2)
        return caps, sp

    def teaser_for(self, p37, x0, x1, levels, sw, ew):
        return teaser_tree(self.D["snaps"][2000], p37, x0, x1, levels, sw=sw, ew=ew)


# ======================================================================================
# 16:9 master
# ======================================================================================
class AlphaGoMCTS(Base):
    SCENE_ID = "alphago_mcts"
    VOICE_JSON = "alphago_mcts.voice.json"

    ROOT = np.array([-2.0, 2.28, 0.0])
    ROOT_SIZE = 1.15
    CHILD_Y = 0.45
    CHILD_SIZE = 0.92
    TX0, TX1 = -6.85, 2.85
    GC_Y, GC_SIZE = -1.42, 0.56
    PX = 5.12                      # right panel centre, spans [3.35, 6.9]
    BAR_BASE = -2.8
    BAR_W = 0.44
    HERO_X0, HERO_X1 = 1.85, 6.45  # the hero text column (62 % .. 95 % of the width)

    def construct(self):
        assert abs(config.frame_width - 14.222) < 0.05, "AlphaGoMCTS is the 16:9 cut (-r 1920,1080)"
        self.init_common()
        self.D = toy_data()
        self.games = demo_games()
        self.pts, self.pi_est = pi_data()
        self.tree_cfg = dict(x0=self.TX0, x1=self.TX1, root=self.ROOT[:2], root_size=self.ROOT_SIZE,
                             child_y=self.CHILD_Y, child_size=0.86,
                             levels={2: -0.62, 3: -1.28, 4: -1.9, 5: -2.48}, max_depth=5, tilt=True,
                             min_slot=1.1, min_span=0.27, thr_frac=0.02, thr_min=2, ew=20, ew_min=1.4,
                             rmin=0.05, rmax=0.12, sw=1.0)
        self.beat0()
        self.beat1()
        self.beat2()
        self.beat3()
        self.beat4()
        self.beat5()
        self.beat6()
        self.beat7()
        self.beat8()
        self.mark("end")
        self.finish()

    # ------------------------------------------------------------------ the hero board (opening, callback, cover)
    def hero_board(self, dim=0.5, bottom_fade=False):
        """The board fills the left ~62 % and bleeds off the left / top / bottom edges; the move-37 point sits at
        36 % width, 52 % height; one smooth gradient fades it into the right text panel."""
        size = 10.5
        step = size * (1 - 2 * GoBoard.MARGIN) / 18
        Wf = config.frame_width
        target = np.array([0.36 * Wf - Wf / 2, 4.0 - 0.52 * 8.0, 0])
        go = GoBoard(size, opening_stones(), self.SW)
        go.move_to(target - np.array([(M37[0] - 9) * step, (9 - M37[1]) * step, 0]))
        dim_board(go, dim)
        p37 = go.pt(*M37)
        assert np.linalg.norm(p37 - target) < 1e-6
        assert go.get_left()[0] < -Wf / 2 and go.get_top()[1] > 4 and go.get_bottom()[1] < -4
        edge = go.panel.get_right()[0]
        st = make_stone("B", go.step * 0.47).move_to(p37).set_z_index(5)
        halo = glow_image(go.step * 0.62, X_COL, config.pixel_width / Wf).move_to(p37).set_z_index(4)
        xa, xb = edge - 1.6, edge - 0.15                    # fully BG before the panel edge

        def alpha(X, Y):
            a = smoothstep((X - xa) / (xb - xa))
            if bottom_fade:
                a = np.maximum(a, 0.92 * smoothstep((-2.45 - Y) / 1.0))
            return a

        ppu = config.pixel_width / Wf
        grad = gradient_image(alpha, -Wf / 2 - 0.05, Wf / 2 + 0.05, -4.05, 4.05, ppu).set_z_index(8)
        return go, p37, st, halo, grad

    def hero_place(self, m, y):
        m.move_to([self.HERO_X0 + m.width / 2, y, 0])
        assert m.get_right()[0] <= self.HERO_X1 + 1e-6, m.width
        return m.set_z_index(10)

    def hero_teaser(self, p37):
        return self.teaser_for(p37, p37[0] - 3.4, p37[0] + 3.35, [p37[1], 0.72, 1.45, 2.1, 2.7, 3.25], sw=self.SW,
                               ew=24)

    # ------------------------------------------------------------------ process strip
    def make_strip(self):
        names = ("① 为什么难", "② 随便下完", "③ 一棵树", "④ 选哪条", "⑤ 越长越偏", "⑥ AlphaGo")
        bw, gap = 2.06, 0.15
        self.strip_boxes, self.strip_texts = VGroup(), VGroup()
        for i, s in enumerate(names):
            x = -(6 * bw + 5 * gap) / 2 + bw / 2 + i * (bw + gap)
            b = RoundedRectangle(width=bw, height=0.46, corner_radius=0.12).move_to([x, 3.58, 0])
            b.set_fill(PANEL, 0.95).set_stroke(GRID, 1.2)
            self.strip_boxes.add(b)
            self.strip_texts.add(self.T(s, "label", DIM).move_to(b))
        self.strip_hl = RoundedRectangle(width=bw, height=0.46, corner_radius=0.12)
        self.strip_hl.set_fill(TEXT, 0.08).set_stroke(TEXT, 1.6).move_to(self.strip_boxes[0])
        self.strip_texts[0].set_color(TEXT)
        self.strip = VGroup(self.strip_boxes, self.strip_hl, self.strip_texts).set_z_index(15)

    def set_stage(self, i):
        anims = [self.strip_hl.animate.move_to(self.strip_boxes[i])]
        for j, t in enumerate(self.strip_texts):
            anims.append(t.animate.set_color(TEXT if j == i else (GREY_B if j < i else DIM)))
        return anims

    # ------------------------------------------------------------------ beat 0: move 37 hook, then the history
    def beat0(self):
        self.mark("beat0")
        go, p37, st, halo, grad = self.hero_board(0.5)
        self.add(go, grad)                                        # frame 0 = the cover composition
        self.frame_mark("board_before")

        self.voice("h01")
        tag = self.clean(self.H("2016 · AlphaGo vs 李世石 · 第37手", 26, DIM))
        q1 = self.H("这是失误吗？", 96, TEXT, SEMIBOLD, t2c={"？": THREAT_COL})
        self.hero_place(VGroup(tag, q1).arrange(DOWN, buff=0.38, aligned_edge=LEFT), 0.0)
        tag.set_z_index(10)

        def hook():
            self.show(q1, "这是失误吗？")
            return [FadeIn(tag, shift=LEFT * 0.1), FadeIn(q1, shift=LEFT * 0.15)]

        self.slam(st, p37, go.step, halo, shake=go, drop=0.3, rt=0.7, extra=hook)
        self.frame_mark("stone_landed")
        self.play(q1[-1].animate(rate_func=there_and_back).scale(1.3), halo.animate(rate_func=there_and_back).scale(1.3),
                  run_time=0.8, watch=False)

        # Lee Sedol took more than ten minutes to answer: one clock sweep around the stone
        self.voice("h02")
        q2 = VGroup(self.H("李世石", 72), self.H("十几分钟后才回应", 72)).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        self.hero_place(q2, 0.0)
        sweep = Arc(radius=go.step * 1.05, start_angle=PI / 2, angle=-TAU, arc_center=p37)
        sweep.set_stroke(TEXT, 2.4, opacity=0.85).set_z_index(6)
        hand = Line(p37, p37 + UP * go.step * 1.05).set_stroke(TEXT, 2.4, opacity=0.85).set_z_index(6)
        self.retire(q1)
        self.show(q2, "李世石十几分钟后才回应")
        tag_to = tag.copy().next_to(q2, UP, buff=0.38).align_to(q2, LEFT)
        assert tag_to.get_bottom()[1] > q2.get_top()[1] + 0.2
        self.play(FadeTransform(q1, q2), tag.animate.move_to(tag_to), run_time=0.55)
        self.play(Create(sweep), Rotate(hand, angle=-TAU, about_point=p37), run_time=2.0, watch=False)
        self.play(FadeOut(sweep), FadeOut(hand), run_time=0.3, watch=False)

        # the odds roll: 1/10 -> 1/100 -> 1/1000 -> 1/10000
        lab = self.H("人类会下这一手的概率", 40)
        approx = self.HN(r"\approx 1/", 160)
        den = self.num(10, px=160)
        probe = VGroup(approx.copy(), self.num(10000, px=160)).arrange(RIGHT, buff=0.08)
        foot = self.clean(self.H("AlphaGo 自己的估计", 24, DIM))
        self.hero_place(VGroup(lab, probe, foot).arrange(DOWN, buff=0.32, aligned_edge=LEFT), 0.0)
        approx.move_to(probe[0])
        den.next_to(approx, RIGHT, buff=0.08).align_to(probe[1], LEFT).set_z_index(10)
        den.add_updater(lambda m: m.set_z_index(10))       # rebuilt digits would otherwise drop under the gradient
        approx.set_z_index(10)
        foot.set_z_index(10)
        odds = VGroup(lab, approx, den)
        self.retire(q2)
        self.show(odds, "人类会下这一手的概率 ≈ 1/10000")
        self.play(FadeOut(q2, shift=UP * 0.1), FadeOut(tag), FadeIn(lab, shift=UP * 0.1), FadeIn(approx, shift=UP * 0.1),
                  FadeIn(den, shift=UP * 0.1), FadeIn(foot), run_time=0.55)
        for v in (100, 1000, 10000):
            self.sfx("tick", offset=0.3)
            self.play(ChangeDecimalToValue(den, v), run_time=0.32)
            self.wait(0.1)
        self.play(den.animate(rate_func=there_and_back).set_color(X_COL), run_time=0.55, watch=False)

        # teaser: the lopsided tree erupts out of the stone, then folds back into it
        tree = self.hero_teaser(p37).set_z_index(9)
        tlab = self.clean(self.T("示意", "label", DIM, k=20 / self.px(self.T_SIZE["label"])))
        tlab.move_to([p37[0] - 3.1, 3.55, 0]).set_z_index(9)
        t1 = self.H("它在想什么？", 84, TEXT, MEDIUM)
        t2 = self.H("看完这棵树你就懂了", 44, GREY_B)
        teaser = VGroup(t1, t2).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        self.hero_place(teaser, 0.0)
        self.voice("h03")
        self.retire(odds)
        self.show(teaser, "它在想什么？看完这棵树你就懂了", f_end=0.5)
        self.sfx("whoosh_soft", offset=0.2)
        self.play(FadeOut(odds, shift=UP * 0.1), FadeOut(foot),
                  LaggedStart(*[GrowFromPoint(e, p37) for e in [*tree.edges, *tree.hot]], lag_ratio=0.02),
                  LaggedStart(*[FadeIn(d, scale=0.3) for d in tree.dots], lag_ratio=0.02), FadeIn(tree.glow),
                  FadeIn(tlab), FadeIn(teaser, shift=LEFT * 0.12, rate_func=squish_rate_func(smooth, 0, 0.5)),
                  run_time=1.2)
        self.frame_mark("teaser_tree")
        self.play(*[ShowPassingFlash(e.copy().set_stroke(WHITE, e.stroke_width + 2, opacity=0.9), time_width=0.6)
                    for e in tree.hot], halo.animate(rate_func=there_and_back).scale(1.4), run_time=1.0, watch=False)
        self.settle()
        out = self.stakes(go, p37, st, halo, grad, tree, tlab, teaser)
        self.history(out, p37)

    # ------------------------------------------------------------------ beat 0b: what was at stake (hero panel)
    def panel_cap(self, text):
        """Caption inside the right hero panel (the board fills the left): 42 px, top edge fixed."""
        g = self.hero_lines(text, 42, aligned_edge=LEFT)
        self.hero_place(g, 0.0)
        g.shift(UP * (-1.42 - g.get_top()[1]))
        return g.set_z_index(20)

    def stakes(self, go, p37, st, halo, grad, tree, tlab, teaser):
        self.mark("stakes")
        X0 = self.HERO_X0
        # a. the opponent: 18 world titles counted up live (blue = the opponent, like O in the game later)
        l1 = self.H("对手：李世石", 56)
        w1, w2 = self.H("拿过", 48), self.H("个世界冠军", 48)
        n18 = self.num(18, color=O_COL, px=230)
        row = VGroup(w1, n18, w2).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
        lee = VGroup(l1, row).arrange(DOWN, buff=0.34, aligned_edge=LEFT)
        self.hero_place(lee, 0.0)
        n18.set_value(1)
        n18.add_updater(lambda m: m.set_z_index(10))       # rebuilt digits would otherwise drop under the gradient
        w2.add_updater(lambda m: m.next_to(n18, RIGHT, buff=0.2).align_to(w1, DOWN))
        self.retire(teaser)
        self.show(lee, "对手：李世石拿过 18 个世界冠军", f_end=0.7)
        self.play(tree.animate(rate_func=rush_into).scale(0.02, about_point=p37).set_opacity(0), FadeOut(tlab),
                  FadeOut(teaser, shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0, 0.45)),
                  FadeIn(lee, shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0.35, 1)), run_time=0.8)
        self.remove(tree)
        self.sfx("tick", offset=1.15)
        self.play(ChangeDecimalToValue(n18, 18, rate_func=ease_out_cubic), run_time=1.2)
        self.play(n18.animate(rate_func=there_and_back).scale(1.12), run_time=0.45)

        # b. a timeline in the panel: 1997, chess
        AX = 0.1

        def yx(year):
            return 2.05 + (year - 1997) * 4.25 / 29

        w2.clear_updaters()
        lee.generate_target()
        lee.target.scale(0.56).set_opacity(0.6)
        lee.target.move_to([X0 + lee.target.width / 2, 3.4 - lee.target.height / 2, 0])
        d97 = Dot([yx(1997), AX, 0], radius=0.085, color=TEXT).set_z_index(10)
        y97 = self.N("1997", "label").next_to(d97, UP, buff=0.16).set_z_index(10)
        l97 = VGroup(self.H("电脑赢了", 28, DIM), self.H("国际象棋世界冠军", 28, DIM)).arrange(DOWN, buff=0.08,
                                                                                  aligned_edge=LEFT)
        l97.next_to(d97, DOWN, buff=0.2).align_to(d97, LEFT).set_z_index(10)
        self.retire(lee)
        self.sfx("pop", offset=0.45)
        self.play(MoveToTarget(lee), FadeIn(d97, scale=0.4), FadeIn(y97, shift=DOWN * 0.1), FadeIn(l97, shift=UP * 0.1),
                  *self.say_anims("1997 年，电脑就赢了国际象棋世界冠军", rt=0.8,
                                  mob=self.panel_cap("1997 年，电脑就赢了\n国际象棋世界冠军")), run_time=0.8)

        # c. Go was thought to be ten years away
        axis = Line([yx(1997), AX, 0], [yx(2016), AX, 0]).set_stroke(GREY_B, 2.5).set_z_index(9)
        d16 = Dot([yx(2016), AX, 0], radius=0.085, color=TEXT).set_z_index(10)
        y16 = self.N("2016", "label").next_to(d16, UP, buff=0.16).set_z_index(10)
        p16 = d16.get_center()
        arrow = Arrow(p16, [yx(2026), AX, 0], buff=0, stroke_width=3.5, color=GREY_B,
                      max_tip_length_to_length_ratio=0.16).set_z_index(9)
        l10 = self.H("专家预计：还要十年", 28, DIM)
        l10.next_to(arrow, DOWN, buff=0.2).align_to([self.HERO_X1, 0, 0], RIGHT).set_z_index(10)
        assert l10.get_left()[0] > l97.get_right()[0] + 0.3, (l10.get_left(), l97.get_right())
        self.play(Create(axis), *self.say_anims("可围棋难得多：此前普遍认为，电脑赢职业棋手至少还要十年", rt=0.9,
                                                mob=self.panel_cap("可围棋难得多：此前普遍认为，\n电脑赢职业棋手至少还要十年")),
                  run_time=0.9)
        self.play(FadeIn(d16, scale=0.4), FadeIn(y16, shift=DOWN * 0.1), run_time=0.4)
        self.sfx("whoosh_soft", offset=0.1)
        self.play(GrowArrow(arrow), FadeIn(l10, shift=LEFT * 0.1, rate_func=squish_rate_func(smooth, 0.4, 1)),
                  run_time=0.9)

        # d. the arrow snaps back to 2016: 4 : 1
        f = 180 / (EM * self.REF_PPU)
        self.floor_ok(f, "4 : 1")
        score = MathTex("4", ":", "1", font_size=f)
        score[0].set_color(X_COL)
        score[1].set_color(DIM)
        score[2].set_color(O_COL)
        score.arrange(RIGHT, buff=0.34).move_to([yx(2016), AX + 1.38, 0]).set_z_index(10)
        assert score.get_top()[1] < lee.get_bottom()[1] - 0.12 and score.get_bottom()[1] > y16.get_top()[1] + 0.1
        strike = Line(l10.get_left() + LEFT * 0.06, l10.get_right() + RIGHT * 0.06).set_stroke(GREY_B, 2.2)
        strike.set_z_index(11)
        self.sfx("whoosh", offset=0.05)
        self.play(arrow.animate(rate_func=rush_into).scale(0.02, about_point=p16).set_opacity(0), Create(strike),
                  fade_to(l10, 0.55),
                  *self.say_anims("2016 年 3 月，首尔：AlphaGo 4 : 1 获胜", rt=0.6,
                                  mob=self.panel_cap("2016 年 3 月，首尔：\nAlphaGo 4 : 1 获胜")), run_time=0.6)
        self.remove(arrow)
        self.sfx("stone", offset=0.25)
        self.play(FadeIn(score, scale=1.6), Flash(p16, color=X_COL, line_length=0.16, flash_radius=0.3), run_time=0.55)

        # e. back to the stone: move 37 of game 2
        timeline = VGroup(d97, y97, l97, axis, d16, y16, l10, strike)
        ring = Circle(radius=go.step * 0.5).set_stroke(X_COL, 3.5).move_to(p37).set_z_index(6)
        self.add(ring)
        self.play(fade_to(timeline, 0.4), fade_to(score, 0.6), fade_to(lee, 0.6),
                  ring.animate(rate_func=ease_out_cubic).scale(6).set_stroke(opacity=0),
                  halo.animate(rate_func=there_and_back).scale(1.45),
                  *self.say_anims("第二局的第 37 手，是它最出人意料的一步", rt=0.9,
                                  mob=self.panel_cap("第二局的第 37 手，\n是它最出人意料的一步")), run_time=0.9)
        self.remove(ring)
        return [go, st, halo, grad, lee, timeline, score]

    # ------------------------------------------------------------------ history: solitaire -> Monte Carlo -> pi
    def history(self, fade_list, p37):
        cards = VGroup(*[card(0.95, 1.33, True, r, s) for r, s in
                         (("A", "♠"), ("7", "♣"), ("K", "♠"), ("3", "♣"), ("Q", "♠"))])
        fan_c = np.array([0.0, -0.1, 0])
        for cd in cards:
            cd.move_to(fan_c)
        self.sfx("shuffle", offset=0.45)
        cards.save_state()
        for cd in cards:
            cd.scale(0.1).move_to(p37).set_opacity(0)
        self.play(Restore(cards), *[FadeOut(m) for m in fade_list],
                  *self.say_anims("答案，要从一副纸牌说起", rt=0.8), run_time=0.8)
        self.play(*[cd.animate.rotate((2 - k) * 0.22, about_point=fan_c + DOWN * 1.6) for k, cd in enumerate(cards)],
                  run_time=0.6, rate_func=ease_out_cubic)
        self.mark("history")

        # a. Ulam, 1946: one solitaire layout
        self.voice("h04")
        rr = random.Random(1946)
        piles, tops = VGroup(), []
        for col in range(7):
            pile = VGroup()
            for k in range(col + 1):
                face = k == col
                c = card(0.78, 1.09, face, "A23456789TJQK"[rr.randrange(13)].replace("T", "10"), "♠♣"[rr.randrange(2)])
                c.move_to([-3.2 + col * 0.92, 0.75 - k * 0.3, 0])
                pile.add(c)
                if face:
                    tops.append(c)
            piles.add(pile)
        self.sfx("card", offset=0.5)
        rest = [VGroup(*[c for c in piles[col] if c is not tops[col]]) for col in range(7)]
        self.play(*self.say_anims("1946 年，数学家乌拉姆\n养病时，玩一种单人纸牌", rt=1.1),
                  *[ReplacementTransform(cards[k], tops[k + 1]) for k in range(5)], FadeIn(tops[0]), FadeIn(tops[6]),
                  LaggedStart(*[FadeIn(r, shift=DOWN * 0.1) for r in rest], lag_ratio=0.1), run_time=1.1)
        self.add(piles)

        # b. the same family as the solitaire that came with every PC: all four piles A -> K = a win
        FX, FY = [-0.44 + 0.92 * j for j in range(4)], 2.45
        slots = VGroup(*[RoundedRectangle(width=0.78, height=1.09, corner_radius=0.086).set_stroke(GREY_B, 1.6, opacity=0.8)
                         .set_fill(opacity=0).move_to([x, FY, 0]) for x in FX])
        found, fly = VGroup(), []
        k = 0
        for r, rank in enumerate(["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]):
            for s, suit in enumerate("♠♥♦♣"):
                c = card(0.78, 1.09, True, rank, suit).move_to([FX[s], FY + 0.012 * r, 0])
                src = piles[k % 7][-1].get_center()
                fly.append(FadeIn(c, shift=c.get_center() - src))
                found.add(c)
                k += 1
        tab_cards = [c for col in piles for c in col]
        self.play(FadeIn(slots), *self.say_anims("和电脑里自带的『纸牌』接龙是一类：\n把牌全部接完就算赢", rt=0.6), run_time=0.6)
        self.sfx("card", offset=0.1)
        self.play(LaggedStart(*fly, lag_ratio=0.15),
                  LaggedStart(*[FadeOut(c, scale=0.9) for c in reversed(tab_cards)], lag_ratio=0.12), run_time=3.2)
        ok = check_glyph(0.62, WIN_COL, 7).move_to([FX[-1] + 1.05, FY, 0])
        self.sfx("pop", offset=0.2)
        self.play(Create(ok), run_time=0.4)

        # c. whether it comes out is mostly the luck of the shuffle: shuffle, then two deals
        dc = np.array([0.0, 0.7, 0])
        deck = VGroup(*[mini_card(0.95, 1.33, False).move_to(dc + UP * 0.014 * i) for i in range(24)])
        self.play(FadeOut(VGroup(found, slots, ok), target_position=dc, scale=0.6),
                  FadeIn(deck, scale=1.3, rate_func=squish_rate_func(smooth, 0.4, 1)),
                  *self.say_anims("能不能接完，很大程度看洗牌的运气", rt=0.8), run_time=0.8)
        self.sfx("shuffle", offset=0.1)
        for rep in range(2):
            a_half, b_half = list(deck[:12]), list(deck[12:])
            self.play(VGroup(*a_half).animate.shift(LEFT * 1.05 + DOWN * 0.06).rotate(0.06),
                      VGroup(*b_half).animate.shift(RIGHT * 1.05 + UP * 0.06).rotate(-0.06), run_time=0.28)
            order = [c for pair in zip(a_half, b_half) for c in pair]
            deck.submobjects = order
            back = {id(c): (-0.06 if c in a_half else 0.06) for c in order}
            self.play(LaggedStart(*[c.animate.rotate(back[id(c)]).move_to(dc + UP * 0.014 * i)
                                    for i, c in enumerate(order)], lag_ratio=0.08), run_time=0.5)
        PW, PH = 5.0, 3.5
        centres = [np.array([-3.45, 0.55, 0]), np.array([3.45, 0.55, 0])]
        panels, tabs, fslots = [], [], []
        for c0 in centres:
            box = RoundedRectangle(width=PW, height=PH, corner_radius=0.2).set_fill(PANEL, 0.9).set_stroke(GRID, 1.4)
            box.move_to(c0)
            fs = VGroup(*[RoundedRectangle(width=0.4, height=0.56, corner_radius=0.05).set_stroke(GREY_B, 1.2)
                          .set_fill(opacity=0).move_to(c0 + np.array([0.3 + j * 0.55, 1.05, 0])) for j in range(4)])
            tab = VGroup()
            for i in range(7):
                for kk in range(i + 1):
                    tab.add(mini_card(0.4, 0.56, kk == i).move_to(c0 + np.array([-1.8 + i * 0.6, 0.2 - kk * 0.12, 0])))
            panels.append(box)
            tabs.append(tab)
            fslots.append(fs)
        deal = []
        for tab in tabs:
            for m in tab:
                deal.append(FadeIn(m, shift=m.get_center() - dc))
        self.play(FadeOut(deck, scale=0.5), FadeIn(panels[0]), FadeIn(panels[1]), FadeIn(fslots[0]), FadeIn(fslots[1]),
                  LaggedStart(*deal, lag_ratio=0.02), run_time=1.0)
        heights = ([13, 13, 13, 13], [2, 1, 0, 2])          # deal A comes out, deal B gets stuck
        grow, gone = [], []
        for p_i in (0, 1):
            fs, tab = fslots[p_i], tabs[p_i]
            stack = VGroup()
            for j, hgt in enumerate(heights[p_i]):
                for q in range(hgt):
                    m = mini_card(0.4, 0.56, True).move_to(fs[j].get_center() + UP * 0.01 * q)
                    src = tab[(j * 13 + q) % len(tab)].get_center()
                    grow.append((p_i, FadeIn(m, shift=m.get_center() - src)))
                    stack.add(m)
            n_go = len(tab) if p_i == 0 else 5
            gone += [FadeOut(m, scale=0.9) for m in list(tab)[::-1][:n_go]]
            tabs[p_i] = (tab, stack)
        self.play(LaggedStart(*[a for p_i, a in grow if p_i == 0], lag_ratio=0.12),
                  LaggedStart(*[a for p_i, a in grow if p_i == 1], lag_ratio=0.6, rate_func=squish_rate_func(linear, 0, 0.35)),
                  LaggedStart(*gone, lag_ratio=0.1), run_time=1.8)
        stuck = tabs[1][0][:-5]
        self.play(Wiggle(stuck, scale_value=1.02, rotation_angle=0.015 * TAU), run_time=0.6)
        yes = check_glyph(0.8, WIN_COL, 9).move_to(centres[0] + np.array([-1.45, 1.05, 0])).set_z_index(5)
        no = cross_glyph(0.56, GREY_B, 8).move_to(centres[1] + np.array([-1.45, 1.05, 0])).set_z_index(5)
        self.sfx("pop", offset=0.2)
        self.play(Create(yes), Create(no), run_time=0.5)

        # d. the question
        deals = VGroup(panels[0], fslots[0], tabs[0][1], panels[1], fslots[1], tabs[1][0], tabs[1][1], yes, no)
        qm = self.H("？", 180, WIN_COL, MEDIUM).move_to([0, 0.55, 0])
        self.play(fade_to(deals, 0.35), FadeIn(qm, scale=0.5),
                  *self.say_anims("他想知道：随便洗一次，\n能玩通的概率是多少？", rt=0.7), run_time=0.7)

        # e. too many shuffles to count: 52 cards, 52! orders, built up card by card
        row = VGroup(*[mini_card(0.36, 0.5, False) for _ in range(52)])
        row.arrange_in_grid(rows=2, cols=26, buff=(0.09, 0.12)).move_to([0, 2.55, 0])
        assert row.width < 13.0
        kt = ValueTracker(1)
        n_cards = self.num(52, "title")
        crow = VGroup(n_cards, self.T("张牌", "cap")).arrange(RIGHT, buff=0.16, aligned_edge=DOWN).move_to([0, 1.45, 0])
        n_cards.set_value(1)
        n_cards.add_updater(lambda m: m.set_value(int(kt.get_value())))
        crow[1].add_updater(lambda m: m.next_to(n_cards, RIGHT, buff=0.16).align_to(crow[1], DOWN))
        self.floor_ok(22, "52!")
        facts = [Text(str(math.factorial(j)), font="Helvetica Neue", font_size=22, color=TEXT).move_to([0, 0.5, 0])
                 for j in range(1, 53)]
        assert facts[-1].width < 12.6 and len(str(FACT52)) == 68
        holder = VGroup(facts[0])
        holder.add_updater(lambda m: setattr(m, "submobjects", [facts[min(51, int(kt.get_value()) - 1)]]))
        orders = self.T("种洗法", "cap", DIM).move_to([0, -0.2, 0])
        self.play(FadeOut(VGroup(deals, qm), shift=UP * 0.2), FadeIn(crow), FadeIn(holder), FadeIn(orders),
                  FadeIn(row[0]),
                  *self.say_anims("可 52 张牌有约 8×10^67 种洗法，\n一种种算不完", rt=0.7), run_time=0.7)
        self.sfx("card", offset=0.1)
        self.play(kt.animate(rate_func=linear).set_value(52),
                  LaggedStart(*[FadeIn(c, shift=DOWN * 0.15) for c in row[1:]], lag_ratio=0.5), run_time=2.6)
        holder.clear_updaters()
        n_cards.clear_updaters()
        crow[1].clear_updaters()
        f = 120 / (EM * self.REF_PPU)
        self.floor_ok(f, "8x10^67")
        big = MathTex(r"\approx 8\times 10^{67}", font_size=f, color=TEXT).move_to([0, 0.45, 0])
        orders2 = self.T("种洗法", "cap").next_to(big, RIGHT, buff=0.3).align_to(big, DOWN)
        self.sfx("pop", offset=0.4)
        self.play(FadeTransform(holder, big), ReplacementTransform(orders, orders2), run_time=0.8)

        # f. Ulam's idea: just play 100 games and count (illustrative grid, tagged 示意)
        res, cum = solitaire_games()
        n_ok = cum[-1]
        S = 4.7
        sq_c = np.array([-2.75, -0.35, 0])
        square = Square(S).set_stroke(TEXT, 2.0).move_to(sq_c)
        cs = S / 10
        o = square.get_corner(DL)
        inner = VGroup(*[Line(o + RIGHT * cs * i, o + RIGHT * cs * i + UP * S) for i in range(1, 10)],
                       *[Line(o + UP * cs * i, o + UP * cs * i + RIGHT * S) for i in range(1, 10)]).set_stroke(GRID, 1.0)
        marks = VGroup()
        for i, win in enumerate(res):
            cen = o + np.array([(i % 10 + 0.5) * cs, (9.5 - i // 10) * cs, 0])
            marks.add((check_glyph(cs * 0.52, WIN_COL, 3.4) if win else cross_glyph(cs * 0.36, GREY_B, 2.6, 0.75))
                      .move_to(cen))
        ctr_n = self.num(n_ok, "title", color=WIN_COL)
        grow_row = VGroup(self.T("成功", "cap", WIN_COL), ctr_n, self.T("/ 100", "cap", DIM))
        grow_row.arrange(RIGHT, buff=0.18, aligned_edge=DOWN).move_to([3.45, 0.85, 0])
        ctr_n.set_value(0)
        tag = self.clean(self.T("示意", "label", DIM, k=20 / self.px(self.T_SIZE["label"])))
        tag.next_to(square, UP, buff=0.12).align_to(square, LEFT)
        self.play(FadeOut(VGroup(row, crow, big, orders2), shift=UP * 0.2), Create(square),
                  Create(inner, lag_ratio=0.05), FadeIn(grow_row), FadeIn(tag),
                  *self.say_anims("他想到：不如真的洗 100 次、\n玩 100 局，数数成功几局", rt=0.8), run_time=0.8)
        gt = ValueTracker(0)
        ctr_n.add_updater(lambda m: m.set_value(cum[min(100, int(gt.get_value()))]))
        grow_row[2].add_updater(lambda m: m.next_to(ctr_n, RIGHT, buff=0.18).align_to(grow_row[0], DOWN))
        self.sfx("card", offset=0.1)
        self.play(LaggedStart(*[FadeIn(m, scale=1.4) for m in marks], lag_ratio=0.5),
                  gt.animate(rate_func=linear).set_value(101), run_time=3.0)
        ctr_n.clear_updaters()
        grow_row[2].clear_updaters()
        assert ctr_n.get_value() == n_ok

        # g. this is the Monte Carlo method; the same counting estimates pi
        self.voice("h05")
        quarter = VMobject()
        quarter.set_points_as_corners([o, o + RIGHT * S])
        quarter.append_points(Arc(radius=S, start_angle=0, angle=PI / 2, arc_center=o).points)
        quarter.add_line_to(o)
        quarter.set_fill(WIN_COL, 0.07).set_stroke(WIN_COL, 2.2)
        rx = 3.45
        num_n = self.num(0, "cap")
        num_k = self.num(0, "cap", color=WIN_COL)
        row_n = VGroup(self.T("撒了", "label"), num_n, self.T("个点", "label")).arrange(RIGHT, buff=0.16)
        row_k = VGroup(self.T("落进圆里", "label", WIN_COL), num_k, self.T("个", "label", WIN_COL)).arrange(RIGHT, buff=0.16)
        VGroup(row_n, row_k).arrange(DOWN, buff=0.3, aligned_edge=LEFT).move_to([rx, 0.85, 0])
        est = self.num(self.pi_est[10][1], "title", decimals=3)
        est_row = VGroup(self.T("估计圆周率", "label", DIM), est).arrange(RIGHT, buff=0.25).move_to([rx, -0.55, 0])
        truth = self.T("真正的圆周率 3.14159…", "label", DIM).move_to([rx, -1.55, 0])
        self.play(LaggedStart(*[FadeOut(m, scale=0.4) for m in marks], lag_ratio=0.01), FadeOut(inner), FadeOut(tag),
                  DrawBorderThenFill(quarter), FadeTransform(grow_row, VGroup(row_n, row_k)),
                  *self.say_anims("这就是蒙特卡洛方法：\n算不出来，就随机试很多次", rt=1.0), run_time=1.0)
        row_n[2].add_updater(lambda m: m.next_to(num_n, RIGHT, buff=0.16))
        row_k[2].add_updater(lambda m: m.next_to(num_k, RIGHT, buff=0.16))

        def to_scene(x, y):
            return o + np.array([x * S, y * S, 0])

        dots = VGroup(*[Dot(to_scene(x, y), radius=0.045, color=WIN_COL if h else DIM).set_z_index(3)
                        for x, y, h in self.pts[:100]])
        self.play(LaggedStart(*[FadeIn(d, shift=DOWN * 0.35, scale=1.6) for d in dots[:10]], lag_ratio=0.2),
                  ChangeDecimalToValue(num_n, 10), ChangeDecimalToValue(num_k, self.pi_est[10][0]),
                  FadeIn(est_row, shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0.5, 1)), FadeIn(truth),
                  run_time=0.8)
        self.sfx("rain", offset=0.0)
        self.play(LaggedStart(*[FadeIn(d, scale=1.6) for d in dots[10:100]], lag_ratio=0.03),
                  ChangeDecimalToValue(num_n, 100), ChangeDecimalToValue(num_k, self.pi_est[100][0]),
                  ChangeDecimalToValue(est, self.pi_est[100][1]), run_time=0.7)
        steps = [(100, 400), (400, 700), (700, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000)]
        base100 = ImageMobject(points_image(self.pts, 640, 3.6, 100, 10)).set_z_index(2)
        base100.height = S
        base100.move_to(sq_c)
        self.remove(*dots[10:])
        self.add(base100)
        imgs = []
        k_run = self.pi_est[100][0]
        for a, b in steps:
            im = ImageMobject(points_image(self.pts, 640, 2.6 if b <= 1000 else 2.0, b, a)).set_z_index(2)
            im.height = S
            im.move_to(sq_c)
            imgs.append(im)
            k_run = sum(1 for x, y, h in self.pts[:b] if h)
            self.play(FadeIn(im), ChangeDecimalToValue(num_n, b), ChangeDecimalToValue(num_k, k_run),
                      ChangeDecimalToValue(est, 4 * k_run / b), run_time=0.25, rate_func=linear)
        assert k_run == self.pi_est[5000][0]
        # why x 4: the quarter disc covers pi/4 of the square, so the share of dots inside x 4 ~ pi
        n_in, est5 = self.pi_est[5000]
        assert n_in == 3933 and f"{4 * n_in / 5000:.3f}" == "3.146" and abs(est5 - 4 * n_in / 5000) < 1e-12
        sq_fill = Square(S).set_fill(GREY_B, 0.16).set_stroke(width=0).move_to(sq_c).set_z_index(1)
        q_fill = quarter.copy().set_fill(WIN_COL, 0.34).set_stroke(width=0).set_z_index(1.5)
        cen = o + np.array([0.424 * S, 0.424 * S, 0])
        q_lab = MathTex(r"\pi/4", font_size=self.N_SIZE["title"], color=TEXT).move_to(cen).set_z_index(4)
        self.floor_ok(self.N_SIZE["title"], "pi/4")
        dim_ims = [im.animate.set_opacity(0.22) for im in (base100, *imgs)] + [dots[:10].animate.set_opacity(0.22)]
        self.play(*dim_ims, FadeIn(sq_fill), run_time=0.6, watch=False)
        self.play(FadeIn(q_fill), Write(q_lab), *self.say_anims("四分之一圆占正方形面积的 π/4", rt=0.9), run_time=0.9)
        ar = MathTex("3933", r"\div", "5000", r"\times", "4", "=", "3.146", font_size=self.N_SIZE["cap"])
        ar[0].set_color(WIN_COL)
        ar[6].set_color(WIN_COL)
        self.floor_ok(self.N_SIZE["cap"], "3933/5000x4")
        ar.move_to([rx, -1.12, 0])
        assert ar.get_left()[0] > square.get_right()[0] + 0.3 and ar.get_right()[0] < 6.9
        self.play(TransformFromCopy(num_k, ar[0]), TransformFromCopy(num_n, ar[2]), FadeIn(ar[1]), FadeIn(ar[3:5]),
                  truth.animate.shift(DOWN * 0.5),
                  *self.say_anims("所以：落进圆里的比例 × 4 ≈ π", rt=0.9), run_time=0.9)
        self.play(FadeIn(ar[5:], shift=LEFT * 0.1), Indicate(est, color=WIN_COL, scale_factor=1.1), run_time=0.7)
        self.pi_extra = Group(sq_fill, q_fill, q_lab, ar)
        big = self.N(r"\approx 3.14", "title", WIN_COL).move_to(est, aligned_edge=LEFT)
        assert f"{self.pi_est[5000][1]:.2f}" in ("3.14", "3.15")
        self.sfx("chime", offset=0.6)
        undim = [im.animate.set_opacity(1.0) for im in (base100, *imgs)] + [dots[:10].animate.set_opacity(1.0)]
        self.play(FadeTransform(est, big), *undim, FadeOut(Group(sq_fill, q_fill, q_lab)),
                  *self.say_anims("随机撒点：撒得越多，越接近 3.14", aha=1.5, rt=0.9), run_time=0.9)
        self.play(Circumscribe(big, color=WIN_COL, buff=0.12, time_width=0.6), run_time=0.8)

        # Monte Carlo
        self.voice("h06")
        pi_all = Group(square, quarter, dots[:10], base100, *imgs, row_n, row_k, est_row[0], big, truth, self.pi_extra[3])
        d1 = dice(0.8, 5, sw=self.SW).rotate(0.18).move_to([-3.9, 0.55, 0])
        d2 = dice(0.8, 3, sw=self.SW).rotate(-0.14).move_to([-2.9, 0.35, 0])
        dd = VGroup(d1, d2)
        mc1 = self.T("后来用于核武器研究，", "cap")
        name2 = self.T("蒙特卡洛", "title", weight=MEDIUM)
        mc2 = VGroup(self.T("代号：", "cap"), name2, self.T("（一座赌城）", "cap", DIM)).arrange(RIGHT, buff=0.1,
                                                                                           aligned_edge=DOWN)
        card_mc = VGroup(mc1, mc2).arrange(DOWN, buff=0.32, aligned_edge=LEFT).move_to([1.1, 0.45, 0])
        self.show(card_mc, "后来用于核武器研究，代号：蒙特卡洛（一座赌城）")
        self.sfx("dice", offset=0.55)
        self.play(FadeOut(pi_all), *self.unsay_anims(), FadeIn(d1, shift=DOWN * 0.5), FadeIn(d2, shift=DOWN * 0.5),
                  LaggedStart(FadeIn(mc1, shift=UP * 0.1), FadeIn(mc2, shift=UP * 0.1), lag_ratio=0.3),
                  run_time=0.8, rate_func=ease_out_cubic)

        # random trials + a search tree = MCTS (5 s); AlphaGo uses it
        self.voice("h07")
        ti = small_tree_icon(0.9, self.SW)
        a1 = self.T("随机试验", "cap")
        a2 = self.T("+", "cap", DIM)
        a3 = self.T("搜索树", "cap")
        a4 = self.T("=", "cap", DIM)
        a5 = self.T("蒙特卡洛树搜索", "title", weight=MEDIUM)
        dd_t = dd.copy().scale(0.62)
        eq = VGroup(VGroup(dd_t, a1).arrange(DOWN, buff=0.2), a2, VGroup(ti, a3).arrange(DOWN, buff=0.2), a4, a5)
        eq.arrange(RIGHT, buff=0.45).move_to([0, 0.5, 0])
        a1.align_to(a3, DOWN)
        self.retire(card_mc)
        self.show(eq, "随机试验 + 搜索树 = 蒙特卡洛树搜索", f_end=0.7)
        self.sfx("whoosh_soft", offset=0.3)
        sq = dict(rate_func=squish_rate_func(smooth, 0, 0.7))
        self.play(FadeOut(mc1), FadeOut(mc2[0]), FadeOut(mc2[2]), ReplacementTransform(dd, dd_t, **sq),
                  ReplacementTransform(name2, a5[:4], **sq), FadeIn(a5[4:], shift=LEFT * 0.2, **sq), FadeIn(a1, **sq),
                  FadeIn(a2, **sq), FadeIn(a4, **sq), Create(ti, lag_ratio=0.2, **sq), FadeIn(a3, **sq), run_time=1.0)
        uses = self.T("AlphaGo 用的正是它", "cap").move_to([0, -0.85, 0])
        self.retire(eq)
        self.show(uses, "AlphaGo 用的正是它")
        self.play(FadeOut(VGroup(dd_t, a1, a2, ti, a3, a4)), a5.animate.move_to([0, 0.55, 0]),
                  FadeIn(uses, shift=UP * 0.12), run_time=0.6)
        self.b0_end = (a5, uses)

    # ------------------------------------------------------------------ beat 1: why brute force fails
    def beat1(self):
        self.mark("beat1")
        self.make_strip()
        go = GoBoard(3.4, opening_stones(), 0.7).move_to([-5.2, 0.15, 0]).set_z_index(5)
        p = go.pt(*M37)
        st = make_stone("B", go.step * 0.47, 0.7).move_to(p).set_z_index(7)
        halo = glow_image(go.step * 0.6, X_COL, config.pixel_width / config.frame_width).move_to(p).set_z_index(6)
        self.go, self.go_st = go, (st, halo)
        a5, uses = self.b0_end
        self.voice("b01")
        self.sfx("whoosh", offset=0.2)
        self.retire(uses)
        self.play(FadeOut(a5, shift=UP * 0.3), FadeOut(uses, shift=UP * 0.3), FadeIn(self.strip, shift=DOWN * 0.2),
                  FadeIn(go, shift=RIGHT * 0.5), FadeIn(halo, shift=RIGHT * 0.5), FadeIn(st, shift=RIGHT * 0.5),
                  *self.say_anims("最笨的办法：把所有下法都算一遍", rt=0.8), run_time=0.8, watch=False)
        levels, ends1, angs = fractal_fan(p, -1.12, 1.12, lengths=(2.5, 1.0, 0.45), seed=250)
        for m in levels:
            m.set_z_index(6)
            set_fan(m, 0.0)
        lab1 = self.T("每一步约 250 种走法", "label").move_to([-2.4, 2.95, 0])
        cnt = self.N("250", "title").move_to([3.75, 1.85, 0])
        self.add(*levels)
        self.sfx("pop", offset=0.2)
        self.play(fan_grow(levels[0], 1.1, lag=0.5), FadeIn(lab1, shift=DOWN * 0.1, rate_func=squish_rate_func(smooth, 0.4, 1)),
                  Write(cnt, rate_func=squish_rate_func(smooth, 0.3, 1)), run_time=1.1)
        cnt2 = MathTex(r"250", r"\times", r"250", font_size=self.N_SIZE["title"]).move_to(cnt)
        self.sfx("pop", offset=0.3)
        self.play(fan_grow(levels[1], 1.3, lag=0.7), TransformMatchingTex(cnt, cnt2), run_time=1.3)
        cnt3 = MathTex(r"250", r"\times", r"250", r"\times", r"\cdots", r"\times", r"250",
                       font_size=self.N_SIZE["title"]).move_to(cnt)
        br = Brace(cnt3, DOWN, buff=0.15, color=DIM)
        br_lab = self.T("一盘约 150 手", "label", DIM).next_to(br, DOWN, buff=0.12)
        self.play(fan_grow(levels[2], 1.1, lag=0.7), TransformMatchingTex(cnt2, cnt3), run_time=1.1)
        self.frame_mark("fan_beat1")
        self.play(GrowFromCenter(br), FadeIn(br_lab, shift=UP * 0.1), run_time=0.5)
        total = self.N(r"\approx 10^{360}", "title").next_to(br_lab, DOWN, buff=0.45)
        self.play(Write(total), *self.say_anims("粗略一算：约 10^360 种下法", rt=0.9), run_time=0.9)

        # 10^80 vs 10^360 as strings of zeros
        self.voice("b02")
        tree_all = VGroup(*levels)
        self.play(*[fade_level(m, DIM_OP, run_time=0.7) for m in levels], FadeOut(lab1), FadeOut(VGroup(cnt3, br, br_lab)),
                  total.animate.scale(0.62).move_to([6.0, 0.62, 0]), run_time=0.7)
        x0 = -1.15

        def zrow(s):
            assert self.px(21) >= self.PX_FLOOR
            return Text(s, font="Helvetica Neue", font_size=21, color=DIM)

        la = VGroup(self.T("宇宙里所有的原子", "label"), self.N(r"\approx 10^{80}", "label"))
        la.arrange(RIGHT, buff=0.25).move_to([0, 2.55, 0]).align_to([x0, 0, 0], LEFT)
        rows_a = VGroup(zrow("1" + "0" * 40), zrow("0" * 40)).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        rows_a.next_to(la, DOWN, buff=0.22).align_to(la, LEFT).set_color(WIN_COL)
        rows_a[0][0].set_color(TEXT)
        lb = VGroup(self.T("一盘围棋的下法", "label"), self.N(r"\approx 10^{360}", "label"))
        lb.arrange(RIGHT, buff=0.25).next_to(rows_a, DOWN, buff=0.42).align_to(la, LEFT)
        rows_b = VGroup(zrow("1" + "0" * 40), *[zrow("0" * 40) for _ in range(8)])
        rows_b.arrange(DOWN, buff=0.12, aligned_edge=LEFT).next_to(lb, DOWN, buff=0.22).align_to(la, LEFT)
        rows_b[0][0].set_color(TEXT)
        assert rows_b.get_bottom()[1] > -3.0 and rows_a.get_right()[0] < 6.9
        self.play(FadeIn(la, shift=UP * 0.1), run_time=0.45)
        self.play(AddTextLetterByLetter(rows_a[0], time_per_char=0.012), run_time=0.5)
        self.play(AddTextLetterByLetter(rows_a[1], time_per_char=0.012), run_time=0.5)
        self.play(ReplacementTransform(total, lb[1]), FadeIn(lb[0]), run_time=0.6)
        self.sfx("boom", offset=2.4)
        self.play(LaggedStart(*[AddTextLetterByLetter(r, time_per_char=0.006) for r in rows_b], lag_ratio=0.5),
                  run_time=2.4)
        self.say("算到宇宙毁灭也算不完", aha=1.5)
        self.say("那 AlphaGo 是怎么想的？")
        self.beat1_stuff = VGroup(tree_all, la, rows_a, lb, rows_b)

    # ------------------------------------------------------------------ beat 2: a small game
    def beat2(self):
        self.mark("beat2")
        self.voice("b03")
        centre = np.array([-2.35, -0.15, 0])
        ttt = TTTBoard(".........", size=3.9).move_to(centre)
        gcopy = self.go.copy()
        self.remove(self.go)
        self.add(gcopy)
        hl_t, vl_t = [ttt.grid[2], ttt.grid[3]], [ttt.grid[0], ttt.grid[1]]
        anims = [ReplacementTransform(gcopy.panel, ttt.panel)]
        anims += [Transform(ln, (hl_t[1] if i < 9.5 else hl_t[0]).copy()) for i, ln in enumerate(gcopy.hlines)]
        anims += [Transform(ln, (vl_t[0] if i < 9.5 else vl_t[1]).copy()) for i, ln in enumerate(gcopy.vlines)]
        anims += [FadeOut(gcopy.stones, scale=0.4), FadeOut(gcopy.hoshi), FadeOut(self.go_st[0], scale=0.4),
                  FadeOut(self.go_st[1], scale=0.4)]
        self.sfx("whoosh_soft", offset=0.2)
        self.play(FadeOut(self.beat1_stuff), *anims, *self.say_anims("先看个小例子：井字棋", rt=1.2), run_time=1.2)
        self.remove(gcopy.hlines, gcopy.vlines)
        self.add(ttt)
        marks = [ttt.mark(0, "X"), ttt.mark(2, "O"), ttt.mark(4, "O")]
        # who is who (stays for the whole search section)
        leg = self.legend("label", k=1.15).set_z_index(15)
        leg.move_to([-6.6 + leg.width / 2, 0.0, 0])
        assert leg.get_right()[0] < ttt.panel.get_left()[0] - 0.15
        self.sfx("stone", offset=0.3)
        self.play(LaggedStart(*[Create(m) for m in marks], lag_ratio=0.35),
                  LaggedStart(FadeIn(leg[0], shift=RIGHT * 0.1), FadeIn(leg[1], shift=RIGHT * 0.1), lag_ratio=0.5,
                              rate_func=squish_rate_func(linear, 0.3, 1)), run_time=0.9)
        self.remove(*marks)
        for i, m in zip((0, 2, 4), marks):
            ttt.marks[i] = m
            ttt.add(m)
        self.ttt, self.leg = ttt, leg
        # whose turn: a big badge pulses once, then docks above the legend
        badge = self.turn_badge("title").move_to([2.1, -0.15, 0]).set_z_index(15)
        self.sfx("pop", offset=0.3)
        self.play(FadeIn(badge, scale=0.6), *self.say_anims("现在轮到 X 下", rt=0.6), run_time=0.6)
        self.play(badge.animate(rate_func=there_and_back).scale(1.12), run_time=0.6)
        dock = badge.copy().scale(0.52)
        dock.move_to([-6.6 + dock.width / 2, leg.get_top()[1] + 0.28 + dock.height / 2, 0])
        self.play(badge.animate.scale(0.52).move_to(dock), run_time=0.6)
        self.badge = badge
        thr = Line(ttt.cell(2) + (ttt.cell(2) - ttt.cell(6)) * 0.12, ttt.cell(6) + (ttt.cell(6) - ttt.cell(2)) * 0.12)
        thr.set_stroke(THREAT_COL, 12, opacity=0.4).set_z_index(1)
        hot = RoundedRectangle(width=ttt.cs * 0.9, height=ttt.cs * 0.9, corner_radius=0.12)
        hot.set_stroke(THREAT_COL, 3).set_fill(THREAT_COL, 0.12).move_to(ttt.cell(6)).set_z_index(2)
        self.play(Create(thr), FadeIn(hot), *self.say_anims("O 已经斜着连了两个：\nX 不堵，O 下一步就赢", rt=0.8),
                  run_time=0.8)
        self.ghosts = {m: ttt.mark(m, "X", opacity=0.3).set_z_index(3) for m in CANDS}
        self.play(LaggedStart(*[FadeIn(g, scale=0.6) for g in self.ghosts.values()], lag_ratio=0.12),
                  *self.say_anims("X 有 6 个空位可选，哪一步最好？", aha=1.0, rt=0.7), run_time=0.7)
        self.threat = VGroup(thr, hot)

    # ------------------------------------------------------------------ beat 3: just play it out
    def beat3(self):
        self.mark("beat3")
        ttt = self.ttt
        self.voice("b04")
        rows_y = {6: 1.25, 1: -0.9}
        icons, slots = {}, {}
        for m, y in rows_y.items():
            icons[m] = TTTBoard(put(TOY_BOARD, m, "X"), size=1.0).move_to([1.35, y, 0])
            slots[m] = [np.array([2.7 + k * 1.0, y, 0]) for k in range(3)]
        self.play(*self.set_stage(1), fade_to(self.threat, 0.4),
                  FadeIn(icons[6]), FadeIn(icons[1]), *self.say_anims("想法一：随便下完，看输赢", rt=0.8),
                  run_time=0.8)
        chips, thumbs = VGroup(), {6: VGroup(), 1: VGroup()}
        ghosts = VGroup(*self.ghosts.values())
        stamp_at = np.array([ttt.panel.get_x(), ttt.panel.get_top()[1] + 0.62, 0])
        assert stamp_at[1] + 0.45 < 3.3
        TH = 0.82

        def to_thumb(m, k, res, trial, gm, caps, sp, more=()):
            """The finished game shrinks next to its tally (board + capsule) and the stamp becomes the chip."""
            src = VGroup(ttt.panel, ttt.grid, *[ttt.marks[i] for i in (0, 2, 4)]).copy()
            src.add(trial.copy(), *[g.copy() for g in gm], *[c.copy() for c in caps])
            src.set_z_index(8)
            self.add(src)
            self.remove(trial, *gm, *caps)
            ch = self.chip(res, h=0.34).move_to(slots[m][k] + DOWN * 0.52).set_z_index(9)
            self.sfx("tick", offset=0.4)
            self.play(src.animate.scale(TH / ttt.panel.width).move_to(slots[m][k] + UP * 0.2), ReplacementTransform(sp, ch),
                      fade_to(ghosts, 1.0), *more, run_time=0.6)
            thumbs[m].add(src)
            chips.add(ch)

        def play_out(m, k, trace, res, cap=None, more=()):
            trial = ttt.mark(m, "X").set_z_index(6)
            self.play(FadeIn(trial, scale=1.3), fade_to(ghosts, 0.0), *(cap or self.drop_read_cap()), *more, run_time=0.4)
            gm = self.play_moves(ttt, trace, z=7)
            cells = put(TOY_BOARD, m, "X")
            for i, q in trace:
                cells = put(cells, i, q)
            caps, sp = self.end_game(ttt, cells, res, stamp_at, 0.45, 6)
            to_thumb(m, k, res, trial, gm, caps, sp)

        for k, (trace, res) in enumerate(self.games[6]):
            play_out(6, k, trace, res, cap=self.say_anims("多下几盘，赢得多的那步更好", rt=0.4) if k == 1 else None)
        for k in (1, 2):                    # after move 1: O sees its win and takes it
            tr, r = self.games[1][k]
            play_out(1, k - 1, tr, r, more=[FadeOut(thumbs[6], scale=0.8)] if k == 1 else [])
        # the flaw, live: here O plays at random and gives the game away
        self.voice("b05")
        trace, res = self.games[1][0]
        assert trace[0][0] != 6 and res == "X"
        trial = ttt.mark(1, "X").set_z_index(6)
        self.play(FadeIn(trial, scale=1.3), fade_to(ghosts, 0.0), fade_to(self.threat, 2.4),
                  fade_to(VGroup(chips[:3], icons[6]), DIM_OP), *self.drop_read_cap(), run_time=0.5)
        win_here = self.T("O 下这里就赢", "label", THREAT_COL).next_to(ttt, DOWN, buff=0.14).align_to(ttt.panel, LEFT)
        self.play(FadeIn(win_here, shift=UP * 0.1), Indicate(self.threat[1], color=THREAT_COL), run_time=0.7)
        dumb = self.T("对手却乱下，白送一局", "label", THREAT_COL).move_to(win_here, aligned_edge=LEFT)
        gm = self.play_moves(ttt, trace[:1], z=7, extra=[FadeTransform(win_here, dumb)])
        gm += self.play_moves(ttt, trace[1:], z=7)
        cells = put(TOY_BOARD, 1, "X")
        for i, q in trace:
            cells = put(cells, i, q)
        caps, sp = self.end_game(ttt, cells, res, stamp_at, 0.45, 6,
                                 cap_anims=self.say_anims("这局赢了，可对手明明能赢", rt=0.35))
        to_thumb(1, 2, res, trial, gm, caps, sp,
                 more=[FadeOut(dumb), fade_to(self.threat, 0.4), fade_to(VGroup(chips[:3], icons[6]), 1.0)])
        self.play(FadeOut(thumbs[1], scale=0.8), *self.say_anims("对手被当成乱下的傻瓜：太粗糙", rt=0.5), run_time=0.5)
        self.say("要把注意力放在有希望的地方")
        self.say("可怎么知道哪里有希望？")
        self.play(FadeOut(VGroup(chips, icons[6], icons[1])), FadeOut(self.threat), run_time=0.6, watch=False)

    # ------------------------------------------------------------------ beat 4: a tree that remembers
    def slot_x(self, m):
        w = (self.TX1 - self.TX0) / 6
        return self.TX0 + (CANDS.index(m) + 0.5) * w

    def beat4(self):
        self.mark("beat4")
        D = self.D
        ttt = self.ttt
        self.voice("b06")
        ghost_g = VGroup(*self.ghosts.values())
        f = self.ROOT_SIZE / ttt.panel.width
        self.sfx("whoosh_soft", offset=0.2)
        leg_to = self.leg.copy()
        leg_to.move_to([-6.85 + leg_to.width / 2, 2.32, 0])
        now_lab = self.T("现在的局面（轮到 X）", "label", DIM, t2c={"轮到 X": X_COL})
        now_lab.move_to([self.ROOT[0] + self.ROOT_SIZE / 2 + 0.3 + now_lab.width / 2, self.ROOT[1], 0])
        badge_to = self.badge.copy().move_to(now_lab.get_right() + LEFT * self.badge.width / 2)
        self.play(*self.set_stage(2), VGroup(ttt, ghost_g).animate.scale(f, scale_stroke=True).move_to(self.ROOT),
                  self.leg.animate.move_to(leg_to), self.badge.animate.move_to(badge_to),
                  *self.say_anims("想法二：画一棵树，记住每次结果", rt=1.1), run_time=1.1)
        self.root = ttt
        ttt.set_z_index(5)
        rlab = VGroup(self.T("第", "label", DIM), self.num(0, "cap"), self.T("轮", "label", DIM)).arrange(RIGHT, buff=0.12)
        rlab.move_to([-4.5 + rlab.width / 2, self.ROOT[1], 0])
        rlab[2].add_updater(lambda m, n=rlab[1]: m.next_to(n, RIGHT, buff=0.12))
        probe = VGroup(self.T("第", "label"), self.num(2000, "cap"), self.T("轮", "label")).arrange(RIGHT, buff=0.12)
        assert rlab.get_left()[0] > leg_to.get_right()[0] + 0.3 and now_lab.get_right()[0] < 3.2
        assert -4.5 + probe.width < self.ROOT[0] - self.ROOT_SIZE / 2 - 0.2, probe.width
        self.round_num, self.round_lab = rlab[1], rlab
        self.now_lab = now_lab
        self.play(FadeIn(rlab), FadeTransform(self.badge, now_lab), run_time=0.5)
        self.kids, self.edges, self.ctrs = {}, {}, {}
        root_bottom = ttt.get_bottom()
        note, thumb = None, None
        first_draw = next(e["it"] for e in D["log"][:6] if e["res"] == "D")
        for e in D["log"][:6]:
            m = e["path"][0]
            p = np.array([self.slot_x(m), self.CHILD_Y, 0])
            kid = TTTBoard(TOY_BOARD, size=self.CHILD_SIZE).move_to(p).set_z_index(5)
            xm = kid.mark(m, "X").set_z_index(6)
            edge = Line(root_bottom, p + UP * self.CHILD_SIZE / 2, stroke_width=20 / 6, color=X_COL,
                        stroke_opacity=0.88).set_z_index(1)
            self.sfx("pop", offset=0.4)
            out = []
            if note is not None:
                self.retire(note[0])
                out, note = [FadeOut(note)], None
            if thumb is not None:                        # the previous round's finished board leaves now
                out, thumb = out + [FadeOut(thumb, scale=0.8)], None
            self.play(TransformFromCopy(VGroup(ttt.panel, ttt.grid, *[ttt.marks[i] for i in (0, 2, 4)]), kid),
                      ReplacementTransform(self.ghosts[m], xm), GrowFromPoint(edge, root_bottom),
                      ChangeDecimalToValue(self.round_num, e["it"]), *self.focus_ctrs(set(), rt=0.42), *out,
                      run_time=0.42)
            self.remove(xm)
            kid.add(xm)
            kid.marks[m] = xm
            ch = self.chip(e["res"], h=0.34).move_to(p + DOWN * 0.95).set_z_index(9)
            ctr = self.counter(1, SCORE_X[e["res"]], p + DOWN * 0.98)
            if e["it"] <= 3:        # rounds 1-3: a slow, readable play-out on an enlarged copy of the new leaf
                ZS = 2.0
                zb = TTTBoard(TOY_BOARD, size=ZS).move_to([p[0] + 1.35, -1.45, 0])
                zb.add(zb.mark(m, "X"))
                zb.set_z_index(10)
                link = DashedLine(kid.panel.get_corner(DR), zb.panel.get_corner(UL), color=GREY_B,
                                  stroke_width=1.6).set_z_index(9)
                self.sfx("whoosh_soft", offset=0.1)
                self.play(TransformFromCopy(kid, zb), Create(link), *self.drop_read_cap(), run_time=0.6)
                gm = self.play_moves(zb, e["trace"], z=11)
                cells = put(TOY_BOARD, m, "X")
                for i, q in e["trace"]:
                    cells = put(cells, i, q)
                caps, sp = self.end_game(zb, cells, e["res"], zb.get_right() + RIGHT * 0.55, 0.42, 5)
                thumb = VGroup(zb, *gm, *caps)
                self.play(thumb.animate.scale(0.55 / ZS).move_to(p + RIGHT * 0.78), FadeOut(link),
                          ReplacementTransform(sp, ch), run_time=0.6)
                self.sfx("tick", offset=0.2)
                self.play(FadeTransform(ch, ctr), run_time=0.4)
            else:
                gm = [kid.mark(i, q, opacity=0.6, scale=0.9).set_z_index(7) for i, q in e["trace"]]
                self.play(LaggedStart(*[FadeIn(g, scale=1.3) for g in gm], lag_ratio=0.6), FadeIn(ch, scale=0.6),
                          run_time=0.3)
                self.play(FadeOut(VGroup(*gm)), FadeTransform(ch, ctr), run_time=0.25)
            self.kids[(m,)], self.edges[(m,)], self.ctrs[(m,)] = kid, edge, ctr
            # what a counter means: its own reading step, one row at a time (round 1 is a loss, round 3 the first draw)
            if e["it"] == 1:
                assert ctr.vals[0] == 1
                note, anims = self.callout(ctr, 0, "试 1 = 这一步模拟过 1 盘", TEXT, [p[0] + 1.25, -1.62, 0])
                self.play(*anims, *self.unsay_anims(), run_time=0.6)
            elif e["it"] == first_draw:
                assert ctr.vals[1] == 0.5
                note, anims = self.callout(ctr, 1, "赢 0.5 = 平局，算半盘", WIN_COL, [p[0] + 1.25, -1.62, 0])
                self.play(*anims, run_time=0.6)
        if note is not None:
            self.retire(note[0])
            self.play(FadeOut(note), run_time=0.35)
        self.half_note = None
        steps = [("选", "挑最有希望的路", "选择"), ("长", "往前多想一步", "扩展"),
                 ("试", "随便下完看输赢", "模拟"), ("传", "把输赢记回去", "回传")]
        self.step_rows = VGroup()
        for k, (big, txt, small) in enumerate(steps):
            y = 1.55 - k * 0.98
            badge = Circle(radius=0.36).set_fill(PANEL, 1).set_stroke(GREY_B, 1.6).move_to([3.78, y, 0])
            bt = self.T(big, "cap", weight=MEDIUM).move_to(badge)
            t = self.T(txt, "label").next_to(badge, RIGHT, buff=0.26).shift(UP * 0.13)
            s = self.T(small, "label", DIM, k=0.8).next_to(t, DOWN, buff=0.06).align_to(t, LEFT)
            self.step_rows.add(VGroup(badge, bt, t, s))
        arrows = VGroup(*[Arrow([3.78, 1.55 - k * 0.98 - 0.38, 0], [3.78, 1.55 - (k + 1) * 0.98 + 0.38, 0], buff=0,
                                stroke_width=2.5, color=GRID, max_tip_length_to_length_ratio=0.35) for k in range(3)])
        loop = CurvedArrow([3.3, 1.55 - 3 * 0.98, 0], [3.3, 1.55, 0], angle=-PI * 0.55, color=GRID, stroke_width=2.5)
        self.step_arrows = VGroup(arrows, loop)
        self.play(LaggedStart(*[FadeIn(r, shift=LEFT * 0.2) for r in self.step_rows], lag_ratio=0.15),
                  FadeIn(self.step_arrows), *self.focus_ctrs(set(), rt=1.2),
                  *self.say_anims("每一轮做四件事", rt=1.2), run_time=1.2)
        self.gkids, self.gedges, self.gctrs = {}, {}, {}
        self.iteration_slow(D["log"][6])
        self.say("一轮又一轮，树就一点点长大")
        for e in D["log"][7:9]:
            self.iteration_fast(e)
        self.play(*self.set_step(None), run_time=0.3)
        self.say("但还有个问题：先试哪一步？")

    def set_step(self, k):
        anims = []
        for j, row in enumerate(self.step_rows):
            if row not in self.mobjects:
                continue
            on = j == k
            anims += [row[0].animate.set_fill(TEXT if on else PANEL, 1).set_stroke(TEXT if on else GREY_B, 1.6),
                      row[1].animate.set_color(BG if on else TEXT),
                      row[2].animate.set_color(TEXT if on or k is None else DIM)]
        return anims

    def iteration_slow(self, e):
        m, g = e["path"]
        kid = self.kids[(m,)]
        snap = self.D["snaps"][e["it"]]
        self.play(ChangeDecimalToValue(self.round_num, e["it"]), run_time=0.35)
        # 1 select
        self.voice("b07")
        cur = Dot(self.root.get_bottom(), radius=0.09, color=WHITE).set_z_index(12)
        ring_root = SurroundingRectangle(self.root.panel, buff=0.06, corner_radius=0.12, color=WHITE, stroke_width=2.5)
        best = [k for k in CANDS if self.ctrs[(k,)].vals[1] == max(c.vals[1] for c in self.ctrs.values())]
        self.play(*self.set_step(0), FadeIn(ring_root), *self.focus_ctrs({(k,) for k in best}, rt=0.75),
                  *self.say_anims("选：挑最有希望的那条路往下走", rt=0.75), run_time=0.75)
        boxes = VGroup(*[SurroundingRectangle(self.ctrs[(k,)][1], buff=0.06, color=WIN_COL, stroke_width=2,
                                              corner_radius=0.06) for k in best])
        self.play(Create(boxes), run_time=0.5)
        ring_kid = SurroundingRectangle(kid.panel, buff=0.06, corner_radius=0.1, color=WHITE, stroke_width=2.5)
        flash = self.edges[(m,)].copy().set_stroke(WHITE, 5, opacity=1)
        self.add(cur)
        self.play(MoveAlongPath(cur, Line(self.root.get_bottom(), kid.get_top())),
                  ShowPassingFlash(flash, time_width=0.6), *self.focus_ctrs({(m,)}, rt=0.8), run_time=0.8)
        self.play(ReplacementTransform(ring_root, ring_kid), FadeOut(boxes), run_time=0.35)
        # 2 expand
        self.voice("b08")
        gp = np.array([kid.get_x(), self.GC_Y, 0])
        gk = TTTBoard(put(put(TOY_BOARD, m, "X"), g, "O"), size=self.GC_SIZE).move_to(gp).set_z_index(5)
        ge = Line(kid.get_bottom() + DOWN * 0.62, gk.get_top(), stroke_width=20 / 7, color=O_COL,
                  stroke_opacity=0.88).set_z_index(1)
        self.play(*self.set_step(1), *self.say_anims("长：往前多想一步，长出新分支", rt=0.6), run_time=0.6)
        self.sfx("pop", offset=0.4)
        self.play(GrowFromPoint(gk, ge.get_start()), Create(ge), MoveAlongPath(cur, Line(kid.get_top(), gk.get_top())),
                  run_time=0.8)
        self.gkids[e["path"]], self.gedges[e["path"]] = gk, ge
        # 3 simulate on a zoomed copy of the new board
        self.voice("b09")
        cells = put(put(TOY_BOARD, m, "X"), g, "O")
        zc, zs = np.array([-3.9, -1.85, 0]), 1.75
        zb = TTTBoard(cells, size=zs).move_to(zc).set_z_index(10)
        self.play(*self.set_step(2), *self.say_anims("试：从这里随便下完，看输赢", rt=0.6), run_time=0.6)
        link = DashedLine(gk.get_right(), zb.get_left(), color=GREY_B, stroke_width=1.6).set_z_index(9)
        self.sfx("whoosh_soft", offset=0.1)
        self.play(TransformFromCopy(gk, zb), Create(link), run_time=0.6)
        gm = self.play_moves(zb, e["trace"], z=11)
        for i, q in e["trace"]:
            cells = put(cells, i, q)
        caps, sp = self.end_game(zb, cells, e["res"], zb.get_right() + RIGHT * 0.55, 0.42, 5)
        ch = self.chip(e["res"], h=0.34).next_to(gk, RIGHT, buff=0.15).set_z_index(12)
        self.slow_thumb = VGroup(zb, *gm, *caps)          # stays beside the leaf through 传
        self.play(self.slow_thumb.animate.scale(0.42 / zs).move_to(gk.get_center() + LEFT * 0.55), FadeOut(link),
                  ReplacementTransform(sp, ch), run_time=0.6)
        # 4 back up
        self.voice("b10")
        self.play(*self.set_step(3), *self.say_anims("传：把输赢沿着来时的路传回去", rt=0.6), run_time=0.6)
        gctr = self.counter(1, x_wins(snap[e["path"]]), gp + DOWN * 0.68)
        self.play(ch.animate.move_to(gk.get_center()), run_time=0.3)
        self.sfx("tick", offset=0.2)
        self.play(FadeIn(gctr, shift=UP * 0.1), Flash(gk.get_center(), color=TEXT, line_length=0.15, flash_radius=0.4),
                  run_time=0.45)
        self.play(ch.animate.move_to(kid.get_center()), MoveAlongPath(cur, Line(gk.get_top(), kid.get_bottom())),
                  run_time=0.5)
        rec = snap[(m,)]
        ctr = self.ctrs[(m,)]
        both = SurroundingRectangle(ctr, buff=0.08, color=TEXT, stroke_width=2, corner_radius=0.08).set_z_index(7)
        self.sfx("tick", offset=0.3)
        self.play(*self.tick_anims(ctr, rec[0], x_wins(rec)), Create(both), run_time=0.55)
        self.say("每一步记一笔：试了几盘、赢了几盘")
        self.play(ch.animate.move_to(self.root.get_center()).scale(0.6).set_opacity(0),
                  MoveAlongPath(cur, Line(kid.get_top(), self.root.get_bottom())), FadeOut(both), run_time=0.6)
        self.play(FadeOut(VGroup(cur, ring_kid)), FadeOut(gctr), *self.focus_ctrs(set(), rt=0.3), run_time=0.3,
                  watch=False)
        self.remove(ch)

    def iteration_fast(self, e):
        m, g = e["path"]
        kid = self.kids[(m,)]
        snap = self.D["snaps"][e["it"]]
        cur = Dot(self.root.get_bottom(), radius=0.09, color=WHITE).set_z_index(12)
        gone = []
        if getattr(self, "slow_thumb", None) is not None:      # the slow round's board leaves when the next round starts
            gone, self.slow_thumb = [FadeOut(self.slow_thumb, scale=0.8)], None
        self.play(ChangeDecimalToValue(self.round_num, e["it"]), *self.set_step(0), FadeIn(cur),
                  *self.focus_ctrs({(m,)}, rt=0.3), *gone, run_time=0.3)
        self.play(MoveAlongPath(cur, Line(self.root.get_bottom(), kid.get_top())), run_time=0.35)
        gp = np.array([kid.get_x(), self.GC_Y, 0])
        gk = TTTBoard(put(put(TOY_BOARD, m, "X"), g, "O"), size=self.GC_SIZE).move_to(gp).set_z_index(5)
        ge = Line(kid.get_bottom() + DOWN * 0.62, gk.get_top(), stroke_width=20 / 8, color=O_COL,
                  stroke_opacity=0.88).set_z_index(1)
        self.sfx("pop", offset=0.2)
        self.play(*self.set_step(1), GrowFromPoint(gk, ge.get_start()), Create(ge),
                  MoveAlongPath(cur, Line(kid.get_top(), gk.get_top())), run_time=0.4)
        gm = [gk.mark(i, q, opacity=0.65, scale=0.9).set_z_index(7) for i, q in e["trace"]]
        ch = self.chip(e["res"], h=0.30).next_to(gk, RIGHT, buff=0.12).set_z_index(12)
        self.play(*self.set_step(2), LaggedStart(*[FadeIn(q, scale=1.3) for q in gm], lag_ratio=0.7),
                  FadeIn(ch, scale=0.6), run_time=0.45)
        rec = snap[(m,)]
        self.play(*self.set_step(3), FadeOut(VGroup(*gm)), ch.animate.move_to(kid.get_center()),
                  MoveAlongPath(cur, Line(gk.get_top(), kid.get_bottom())), run_time=0.4)
        self.sfx("tick", offset=0.2)
        self.play(*self.tick_anims(self.ctrs[(m,)], rec[0], x_wins(rec)), FadeOut(ch), FadeOut(cur), run_time=0.4)
        self.gkids[e["path"]], self.gedges[e["path"]] = gk, ge

    # ------------------------------------------------------------------ beat 5: whom to try next
    def bar_mobs(self, it, scale):
        out = {}
        for m in CANDS:
            wr, bonus = self.D["bars"][it][m]
            x = self.slot_x(m)
            hw = max(wr * scale, 0.001)
            teal = Rectangle(width=self.BAR_W, height=hw).set_fill(WIN_COL, 0.9).set_stroke(width=0)
            teal.move_to([x, self.BAR_BASE + hw / 2, 0])
            yb = Rectangle(width=self.BAR_W, height=bonus * scale).set_fill(BONUS_COL, 0.9).set_stroke(width=0)
            yb.move_to([x, self.BAR_BASE + hw + bonus * scale / 2, 0])
            out[m] = (teal, yb)
        return out

    def beat5(self):
        self.mark("beat5")
        D = self.D
        self.voice("b12")
        gstuff = VGroup(*self.gkids.values(), *self.gedges.values())
        if self.half_note is not None:
            gstuff.add(self.half_note)
        sel = self.step_rows[0]
        self.play(*self.set_stage(3), FadeOut(gstuff), FadeOut(self.step_rows[1:]), FadeOut(self.step_arrows),
                  sel.animate.move_to([self.PX - 0.25, 2.5, 0]), *self.focus_ctrs({(m,) for m in CANDS},
                                                                                              rt=0.8), *self.say_anims("“最有希望”怎么判断？", rt=0.8), run_time=0.8)
        self.play(*self.set_step(0), run_time=0.3)
        scale = 1.55 / max(sum(v) for it in (9, 10, 11, 12) for v in D["bars"][it].values())
        base = Line([self.TX0 + 0.2, self.BAR_BASE, 0], [self.TX1 - 0.2, self.BAR_BASE, 0], color=GRID, stroke_width=1.6)
        bars = self.bar_mobs(9, scale)

        def legend(color, a, b, y):
            sw_ = RoundedRectangle(width=0.3, height=0.3, corner_radius=0.05).set_fill(color, 0.9).set_stroke(width=0)
            txt = VGroup(self.T(a, "cap", color), self.T(b, "label", DIM)).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            return VGroup(sw_, txt).arrange(RIGHT, buff=0.25).move_to([self.PX, y, 0]).align_to([3.55, 0, 0], LEFT)

        leg1 = legend(WIN_COL, "胜率", "赢得多，多试试", 1.2)
        leg2 = legend(BONUS_COL, "好奇心加分", "试得少，加分多", -0.05)
        self.sfx("pop", offset=0.6)
        self.play(Create(base), FadeIn(leg1, shift=LEFT * 0.15),
                  *[TransformFromCopy(self.ctrs[(m,)][1][1], bars[m][0]) for m in CANDS],
                  *self.say_anims("一看胜率：赢的盘数 ÷ 试的盘数", rt=1.1), run_time=1.1)
        self.voice("b13")
        self.play(FadeIn(leg2, shift=LEFT * 0.15), *[GrowFromEdge(bars[m][1], DOWN) for m in CANDS],
                  *self.focus_ctrs(set(), rt=1.1), *self.say_anims("二看好奇心：试得少，多加分", rt=1.1), run_time=1.1)
        k6 = self.kids[(6,)]
        ring6 = SurroundingRectangle(VGroup(*bars[6]), buff=0.08, color=BONUS_COL, stroke_width=2.5, corner_radius=0.06)
        lab6 = self.T("只试过 1 盘", "label", BONUS_COL).next_to(ring6, DOWN, buff=0.06)
        ringk = SurroundingRectangle(k6.panel, buff=0.06, color=BONUS_COL, stroke_width=2.5, corner_radius=0.1)
        self.play(Create(ring6), Create(ringk), FadeIn(lab6, shift=UP * 0.1), *self.focus_ctrs({(6,)}, rt=0.7),
                  *self.say_anims("没怎么试过的，说不定是宝藏", rt=0.7), run_time=0.7)
        self.play(FadeOut(ring6), FadeOut(ringk), FadeOut(lab6), run_time=0.35)
        self.say("总分最高的先试，一样高就轮流")
        cur_bars = bars
        for it in (10, 11, 12):                     # each round picks the tallest bar (asserted in toy_data)
            e = D["log"][it - 1]
            m = e["path"][0]
            tall = max(sum(D["bars"][it - 1][q]) for q in CANDS)
            tops = [q for q in CANDS if abs(sum(D["bars"][it - 1][q]) - tall) < 1e-9]
            hl = VGroup(*[SurroundingRectangle(VGroup(*cur_bars[q]), buff=0.06, color=WHITE, stroke_width=2)
                          for q in tops])
            cur = Dot(self.root.get_bottom(), radius=0.09, color=WHITE).set_z_index(12)
            kid = self.kids[(m,)]
            self.play(Create(hl), ChangeDecimalToValue(self.round_num, it), FadeIn(cur),
                      *self.focus_ctrs({(m,)}, rt=0.4), run_time=0.4)
            self.play(MoveAlongPath(cur, Line(self.root.get_bottom(), kid.get_top())), run_time=0.4)
            ch = self.chip(e["res"], h=0.32).next_to(kid, RIGHT, buff=0.08).set_z_index(12)
            self.play(FadeIn(ch, scale=0.6), run_time=0.25)
            rec = D["snaps"][it][(m,)]
            new_bars = self.bar_mobs(it, scale)
            self.sfx("tick", offset=0.2)
            self.play(*self.tick_anims(self.ctrs[(m,)], rec[0], x_wins(rec)), FadeOut(ch), FadeOut(cur), FadeOut(hl),
                      *[Transform(cur_bars[q][j], new_bars[q][j]) for q in CANDS for j in (0, 1)], run_time=0.6)
        self.b5_stuff = VGroup(base, leg1, leg2, sel, *[b for q in CANDS for b in cur_bars[q]])

    # ------------------------------------------------------------------ beat 6: the tree tilts
    def chart_bar(self, frac, y, L):
        return Rectangle(width=max(0.02, frac * L), height=0.42).set_fill(X_COL, 0.85).set_stroke(width=0).move_to(
            [4.08, y, 0], aligned_edge=LEFT)

    def tree_morph(self, snap):
        cfg = self.tree_cfg
        pos, total = tree_layout(snap, cfg)
        anims = []
        for m in CANDS:
            kid = self.kids[(m,)]
            anims.append(kid.animate.scale(cfg["child_size"] / kid.panel.width, scale_stroke=True).move_to(pos[(m,)]))
        new_dots = {k: tree_dot(k, pos, snap, total, cfg) for k in pos if len(k) >= 2}
        new_edges = {k: tree_edge(k, pos, snap, total, cfg) for k in pos if len(k) >= 1}
        old_pos = self.tree_pos
        for k, d in new_dots.items():
            if k in self.tdots:
                anims.append(Transform(self.tdots[k], d))
            else:
                self.tdots[k] = d
                anims.append(GrowFromPoint(d, tree_anchor(k[:-1], old_pos.get(k[:-1], pos[k[:-1]]), cfg, top=False)))
        for k in [k for k in self.tdots if k not in new_dots]:
            anims.append(FadeOut(self.tdots.pop(k)))
        for k, ed in new_edges.items():
            if k in self.edges:
                anims.append(Transform(self.edges[k], ed))
            else:
                self.edges[k] = ed
                anims.append(GrowFromPoint(ed, tree_anchor(k[:-1], old_pos.get(k[:-1], pos[k[:-1]]), cfg, top=False)))
        for k in [k for k in self.edges if k not in new_edges]:
            anims.append(FadeOut(self.edges.pop(k)))
        self.tree_pos = pos
        return anims

    def beat6(self):
        self.mark("beat6")
        D = self.D
        snaps = D["snaps"]
        self.voice("b14")

        def wr(it, m):
            rec = snaps[it][(m,)]
            return x_wins(rec) / rec[0]

        # what the captions say, checked against the computed tree
        others = [m for m in CANDS if m != 6]
        assert max(snaps[20][(m,)][0] for m in CANDS) <= 2 * min(snaps[20][(m,)][0] for m in CANDS)
        assert all(round(100 * wr(50, m)) <= 25 for m in others) and round(100 * wr(50, 6)) >= 40
        assert all(max(CANDS, key=lambda m: wr(it, m)) == 6 for it in (50, 300, 2000))
        assert round(100 * wr(2000, 6)) == 48 and snaps[2000][(6,)][0] == 1584 and round(100 * D["share6"]) == 79
        BASE, HB, BW = -2.85, 2.2, 0.62
        cols = {}
        for m in CANDS:
            x = self.slot_x(m)
            n0 = snaps[12][(m,)][0]
            bar = Rectangle(width=BW, height=max(0.02, HB * n0 / 12)).set_fill(X_COL, 0.85).set_stroke(width=0)
            bar.move_to([x, BASE, 0], aligned_edge=DOWN).set_z_index(2)
            nlab = VGroup(self.T("试", "label"), self.num(n0, "label")).arrange(RIGHT, buff=0.08)
            nlab.next_to(bar, UP, buff=0.08)
            wnum = self.num(round(100 * wr(12, m)), "label", color=WIN_COL)
            wlab = VGroup(self.T("胜率", "label", WIN_COL), wnum, self.T("%", "label", WIN_COL)).arrange(RIGHT, buff=0.06)
            wlab.move_to([x, self.CHILD_Y - self.CHILD_SIZE / 2 - 0.3, 0])
            assert wlab.width < (self.TX1 - self.TX0) / 6 - 0.08
            cols[m] = (bar, nlab, wlab)
        base = Line([self.TX0 + 0.2, BASE, 0], [self.TX1 - 0.2, BASE, 0], color=GRID, stroke_width=1.6)

        def edge_to(it, m):
            w = float(np.clip(20 * snaps[it][(m,)][0] / it, 1.4, 20))
            return Line(self.root.get_bottom(), self.kids[(m,)].get_top(), stroke_width=w, color=X_COL,
                        stroke_opacity=0.88).set_z_index(1)

        self.play(*self.set_stage(4), FadeOut(self.b5_stuff), Create(base),
                  *[TransformFromCopy(self.ctrs[(m,)][0][1], cols[m][1][1]) for m in CANDS],
                  *[FadeIn(cols[m][1][0]) for m in CANDS],
                  *[FadeTransform(self.ctrs[(m,)][1].copy(), cols[m][2]) for m in CANDS],
                  *[GrowFromEdge(cols[m][0], DOWN) for m in CANDS], FadeOut(VGroup(*self.ctrs.values())),
                  *self.say_anims("一开始：每一步都试得差不多", rt=1.0), run_time=1.0)
        self.add(*[cols[m][1] for m in CANDS])           # the label groups themselves must be in the scene for updaters
        for m in CANDS:
            cols[m][1].add_updater(lambda mob, b=cols[m][0]: mob.next_to(b, UP, buff=0.08))
            cols[m][2][2].add_updater(lambda mob, w=cols[m][2]: mob.next_to(w[1], RIGHT, buff=0.06))

        def step_to(it, rt, extra=()):
            anims = []
            for m in CANDS:
                bar, nlab, wlab = cols[m]
                n = snaps[it][(m,)][0]
                new = Rectangle(width=BW, height=max(0.02, HB * n / it)).set_fill(X_COL, 0.85).set_stroke(width=0)
                new.move_to([self.slot_x(m), BASE, 0], aligned_edge=DOWN).set_z_index(2)
                anims += [Transform(bar, new), ChangeDecimalToValue(nlab[1], n),
                          ChangeDecimalToValue(wlab[1], round(100 * wr(it, m))), Transform(self.edges[(m,)], edge_to(it, m))]
            anims.append(ChangeDecimalToValue(self.round_num, it))
            self.play(*anims, *extra, run_time=rt)
            for m in CANDS:
                assert cols[m][1][1].get_value() == snaps[it][(m,)][0]
                assert cols[m][2][1].get_value() == round(100 * wr(it, m))

        step_to(20, 1.2)
        # 50: the other five never block the diagonal, so O wins next move: almost every game is lost
        self.voice("b15")
        k1 = self.kids[(1,)]
        red_o = k1.mark(6, "O").set_stroke(THREAT_COL).set_z_index(8)
        red_line = Line(k1.cell(2) + (k1.cell(2) - k1.cell(6)) * 0.12, k1.cell(6) + (k1.cell(6) - k1.cell(2)) * 0.12)
        red_line.set_stroke(THREAT_COL, 3, opacity=0.8).set_z_index(8)
        step_to(50, 1.0, extra=self.say_anims("可其他 5 步都没堵住斜线：\nO 下一步就赢，几乎盘盘输", rt=1.0))
        self.play(FadeIn(red_o, scale=1.5), Create(red_line), run_time=0.45)
        self.play(*[Indicate(cols[m][2], color=THREAT_COL, scale_factor=1.08) for m in others], run_time=0.8)
        self.play(FadeOut(red_o), FadeOut(red_line), run_time=0.4)
        # 300: the bottom-left block mostly ends in a draw: the best win rate
        hl6 = SurroundingRectangle(cols[6][2], buff=0.07, color=WIN_COL, stroke_width=2.4, corner_radius=0.08)
        step_to(300, 1.2, extra=self.say_anims("左下角堵住了：大多下成平局，胜率最高", rt=1.2))
        self.play(Create(hl6), Indicate(cols[6][2], color=WIN_COL, scale_factor=1.12), run_time=0.6)
        # 2000: higher win rate -> bigger score -> tried more -> tried even more
        self.settle()                                   # the riser climbs with the last step, success lands on 79 %
        self.sfx("riser", offset=0.0)
        self.sfx("success", offset=1.8)
        step_to(2000, 2.0, extra=self.say_anims("胜率高 → 加分高 → 试得更多 → 越试越多", rt=2.0))
        bar6 = cols[6][0]
        glow6 = box_glow_image(BW + 0.12, bar6.height + 0.12, 0.08, X_COL, config.pixel_width / config.frame_width,
                               peak=0.32, sigma=0.09).move_to(bar6).set_z_index(1)
        pct = self.T(f"{round(100 * D['share6'])}%", "label", X_COL, weight=MEDIUM)
        pct.next_to(bar6, RIGHT, buff=0.1).align_to(bar6, UP).set_z_index(3)
        assert pct.get_right()[0] < self.slot_x(7) - BW / 2 - 0.05
        dim_rest = [fade_to(VGroup(*cols[m], self.kids[(m,)]), 0.45) for m in others]
        k6 = self.kids[(6,)]
        ring = SurroundingRectangle(k6.panel, buff=0.07, color=X_COL, stroke_width=3, corner_radius=0.1)
        self.play(FadeIn(glow6), FadeIn(pct, scale=1.3), Create(ring), *dim_rest,
                  ShowPassingFlash(self.edges[(6,)].copy().set_stroke(WHITE, self.edges[(6,)].stroke_width + 3, opacity=0.9),
                                   time_width=0.7), run_time=0.8)
        self.frame_mark("bars_79")
        self.play(Indicate(VGroup(bar6, pct), color=X_COL, scale_factor=1.04),
                  *self.say_anims("井字棋里，没人教它堵——它自己找到了", aha=2.0, rt=0.7), run_time=0.7)
        for m in CANDS:
            cols[m][1].clear_updaters()
            cols[m][2][2].clear_updaters()
        # deeper: the branch that is tried most is also thought through further
        legend = VGroup(
            VGroup(Dot(radius=0.07, color=X_COL), self.T("橙点：X 的下一步", "label", t2c={"橙点": X_COL})).arrange(RIGHT, buff=0.15),
            VGroup(Dot(radius=0.07, color=O_COL), self.T("蓝点：O 的下一步", "label", t2c={"蓝点": O_COL})).arrange(RIGHT, buff=0.15))
        legend.arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([5.0, 1.15, 0])
        assert legend.get_left()[0] > self.TX1 + 0.3
        self.tdots = {}
        self.tree_pos = {(): self.ROOT.copy(), **{(m,): self.kids[(m,)].get_center() for m in CANDS}}
        colstuff = Group(base, glow6, pct, hl6, *[x for m in CANDS for x in cols[m]])
        self.settle()                                   # the aha is read with the columns still up
        self.play(FadeOut(Group(*colstuff), shift=DOWN * 0.15), FadeOut(ring),
                  *[fade_to(self.kids[(m,)], 1.0) for m in others], run_time=0.6, watch=False)
        self.play(*self.tree_morph(snaps[300]), FadeIn(legend, shift=LEFT * 0.1),
                  *self.say_anims("试得多的分支，就往下想得更深：\nX 堵住后，O 怎么应，X 再怎么下", rt=1.2), run_time=1.2)
        self.play(*self.tree_morph(snaps[2000]), run_time=1.4)
        boards = {}
        for key in ((6, 3), (6, 3, 5)):
            cells = TOY_BOARD
            for j, mv in enumerate(key):
                cells = put(cells, mv, "X" if j % 2 == 0 else "O")
            boards[key] = TTTBoard(cells, size=0.5).move_to(self.tree_pos[key]).set_z_index(5)
        assert snaps[2000][(6, 3)][0] == max(snaps[2000][k][0] for k in snaps[2000] if len(k) == 2 and k[0] == 6)
        assert snaps[2000][(6, 3, 5)][0] == max(snaps[2000][k][0] for k in snaps[2000] if len(k) == 3 and k[:2] == (6, 3))
        self.play(*[ReplacementTransform(self.tdots[k], b) for k, b in boards.items()], run_time=0.7)
        for k, b in boards.items():
            self.tdots[k] = b
        # the payoff: the real board
        self.voice("b16")
        root = self.root
        k6 = self.kids[(6,)]
        big = TTTBoard(TOY_BOARD, size=3.0).move_to([5.0, -0.55, 0])
        assert big.get_left()[0] > self.TX1 + 0.3 and big.get_bottom()[1] > -3.0
        ring = SurroundingRectangle(k6.panel, buff=0.07, color=X_COL, stroke_width=3, corner_radius=0.1)
        arrow = CurvedArrow(k6.get_right() + RIGHT * 0.08, big.cell(6) + LEFT * 0.42, angle=0.5, color=X_COL,
                            stroke_width=2.2, tip_length=0.18).set_z_index(9)
        self.play(FadeOut(legend), Create(ring),
                  TransformFromCopy(VGroup(root.panel, root.grid, *[root.marks[i] for i in (0, 2, 4)]),
                                    VGroup(big.panel, big.grid, *[big.marks[i] for i in (0, 2, 4)])),
                  *self.say_anims("最后：下试得最多的那一步", rt=0.9), run_time=0.9)
        self.add(big)
        thr = Line(big.cell(2) + (big.cell(2) - big.cell(6)) * 0.14, big.cell(6) + (big.cell(6) - big.cell(2)) * 0.14)
        thr.set_stroke(THREAT_COL, 10, opacity=0.55).set_z_index(1)
        self.play(Create(thr), Create(arrow), run_time=0.6)
        x6 = big.mark(6, "X").set_z_index(8)
        x6s = root.mark(6, "X").set_z_index(8)
        mid = (big.cell(4) + big.cell(6)) / 2
        p1 = Line(thr.get_start(), mid).set_stroke(THREAT_COL, 10, opacity=0.55).set_z_index(1)
        p2 = Line(mid, thr.get_end()).set_stroke(THREAT_COL, 10, opacity=0.55).set_z_index(1)
        self.remove(thr)
        self.add(p1, p2)
        self.sfx("stone", offset=0.25)
        self.play(FadeIn(x6, scale=1.7), FadeIn(x6s, scale=1.6), FadeOut(self.now_lab),
                  p1.animate.shift(UP * 0.12 + RIGHT * 0.1).rotate(0.25).set_stroke(opacity=0),
                  p2.animate.shift(DOWN * 0.18 + LEFT * 0.1).rotate(-0.35).set_stroke(opacity=0),
                  *self.say_anims("正是唯一不会输的那一步", aha=1.0, rt=0.8), run_time=0.8)
        self.remove(p1, p2, x6, x6s)
        big.marks[6] = x6
        big.add(x6)
        root.marks[6] = x6s
        root.add(x6s)
        self.say("可围棋大得多——怎么办？")
        self.root = big                                  # the real board becomes the Go board in beat 7
        self.b6_stuff = VGroup(root, ring, arrow)

    # ------------------------------------------------------------------ beat 7: AlphaGo's two instincts
    def beat7(self):
        self.mark("beat7")
        self.voice("b17")
        tree_bits = VGroup(*self.kids.values(), *self.edges.values(), *self.tdots.values())
        bsize = 6.0
        go = GoBoard(bsize, opening_stones(), 0.8).move_to([-3.75, -0.25, 0]).set_z_index(5)
        p = go.pt(*M37)
        self.sfx("whoosh", offset=0.2)
        self.play(*self.set_stage(5), FadeOut(self.b6_stuff), FadeOut(tree_bits), FadeOut(self.round_lab), FadeOut(self.leg),
                  run_time=0.7, watch=False)
        # the 3x3 root grows into a big 19x19 board (the grid multiplies)
        root = self.root
        self.play(root.animate.scale(bsize / root.panel.width, scale_stroke=True).move_to(go), run_time=0.7)
        src = [root.grid[2], root.grid[3], root.grid[0], root.grid[1]]
        anims = [ReplacementTransform(root.panel, go.panel), FadeOut(VGroup(*root.marks.values()))]
        anims += [TransformFromCopy(src[0 if i >= 9.5 else 1], ln) for i, ln in enumerate(go.hlines)]
        anims += [TransformFromCopy(src[2 if i < 9.5 else 3], ln) for i, ln in enumerate(go.vlines)]
        anims += [FadeIn(go.hoshi), FadeIn(go.stones, scale=0.6), FadeOut(root.grid)]
        self.play(*anims, run_time=0.9)
        self.remove(root)
        self.add(go)
        levels, ends, angs = fractal_fan(p, -1.0, 1.0, lengths=(2.3, 0.95, 0.42), seed=17)
        for m in levels:
            m.set_z_index(6)
            set_fan(m, 0.0)
        self.add(*levels)
        self.sfx("pop", offset=0.3)
        self.play(fan_grow(levels[0], 1.0, lag=0.5), *self.say_anims("回到围棋：每步两百多种，树太大", rt=1.0), run_time=1.0)
        self.play(fan_grow(levels[1], 1.0, lag=0.7), run_time=1.0)
        self.play(fan_grow(levels[2], 0.8, lag=0.7), run_time=0.8)
        self.frame_mark("fan_beat7")
        pol = self.net_box("策略网络", X_COL).move_to([5.0, 1.3, 0])
        val = self.net_box("价值网络", WIN_COL).move_to([5.0, -1.05, 0])
        badges = self.step_badges().move_to([5.0, 2.68, 0]).set_z_index(10)
        assert badges.get_bottom()[1] > pol.box.get_top()[1] + 0.25 and badges.get_right()[0] < 6.9
        self.play(FadeIn(VGroup(pol.box, pol.icon, pol.title, val.box, val.icon, val.title), shift=LEFT * 0.2),
                  FadeIn(badges, shift=DOWN * 0.1), *self.say_anims("AlphaGo 加了两样直觉", rt=0.8), run_time=0.8)
        self.add(pol, val)
        # policy network: 4-5 bright branches, the rest fade to 0.06
        self.voice("b18")
        keep = [38, 92, 125, 160, 212]
        bright = VGroup(*[Line(p, ends[i]).set_stroke(X_COL, 3.6, opacity=0.95) for i in keep]).set_z_index(7)
        tips = VGroup(*[Dot(ends[i], radius=0.075, color=X_COL) for i in keep]).set_z_index(8)
        beam = DashedLine(pol.get_left() + LEFT * 0.1, ends[keep[0]] + RIGHT * 0.25, color=X_COL, stroke_width=2.2,
                          dash_length=0.12).set_z_index(6)
        self.play(fade_to(val, 0.45), pol.glow.animate.set_opacity(1.0), Create(beam),
                  *self.say_anims("策略网络：看过 3000 万个\n人类高手的局面，学会猜高手会下哪", rt=0.8), run_time=0.8)
        self.sfx("whoosh_soft", offset=0.2)
        self.play(*self.badge_light(badges, 0),
                  *[fade_level(m, min(1.0, 0.06 / op), run_time=1.0) for m, op in zip(levels, (0.2, 0.11, 0.08))],
                  Create(bright), FadeIn(tips, scale=0.5),
                  *self.say_anims("它升级了「选」：好奇心加分优先给\n看起来像好棋的走法 → 树变窄", rt=1.0), run_time=1.0)
        self.frame_mark("pruned_beat7")
        # value network: a glance replaces the long play-out
        self.voice("b19")
        leaf = ends[keep[2]]
        rng = random.Random(19)
        wpts, x, y = [leaf], leaf[0], leaf[1]
        while x < 2.9:
            x += 0.16
            y += rng.uniform(-0.24, 0.2)
            wpts.append(np.array([x, y, 0]))
        walk = DashedVMobject(VMobject().set_points_as_corners(wpts).set_stroke(GREY_B, 2.2), num_dashes=36)
        walk.set_z_index(7)
        self.play(beam.animate.put_start_and_end_on(val.get_left() + LEFT * 0.1, leaf + RIGHT * 0.15),
                  fade_to(pol, 0.45), pol.glow.animate.set_opacity(0), fade_to(val, 1.0),
                  val.glow.animate.set_opacity(1.0), Create(walk), *self.badge_light(badges, None),
                  *self.say_anims("价值网络：从大量自我对弈里学会，\n看一眼局面就估出谁赢面大", rt=1.1), run_time=1.1)
        dial = gauge(0.34, 0.68, sw=self.SW).move_to(leaf + UP * 0.42).set_z_index(9)
        note = self.T("2016 版：价值网络 + 快速试下一起估", "label", DIM).next_to(val, DOWN, buff=0.25)
        note.align_to([6.85, 0, 0], RIGHT)
        assert note.get_right()[0] <= 6.9 and note.get_left()[0] > 1.5
        self.sfx("pop", offset=0.5)
        self.play(*self.badge_light(badges, 2), FadeOut(walk), FadeIn(dial, scale=0.5), FadeIn(note),
                  *self.say_anims("它升级了「试」：不用每次都随便下完，\n一眼估胜率 → 能想得更深", rt=0.8), run_time=0.8)
        # narrow and deep (schematic)
        self.voice("b20")
        rng = random.Random(20)
        deep = VGroup()
        level = [ends[i] for i in keep]
        cols = [O_COL, X_COL, O_COL, X_COL]
        for d in range(4):
            nxt = []
            for q0 in level:
                k = 2 if (d == 0 and rng.random() < 0.6) else 1
                for j in range(k):
                    q = q0 + np.array([0.42, (j - (k - 1) / 2) * 0.36 + rng.uniform(-0.1, 0.1), 0])
                    deep.add(Line(q0, q).set_stroke(cols[d], 2.8, opacity=0.9))
                    deep.add(Dot(q, radius=0.055, color=cols[d]))
                    nxt.append(q)
            level = nxt
        deep.set_z_index(8)
        tag = self.T("示意", "label", DIM, k=20 / self.px(self.T_SIZE["label"])).move_to([1.2, 2.45, 0])
        assert deep.get_right()[0] < 3.1
        self.play(beam.animate.set_stroke(opacity=0), FadeOut(dial), FadeOut(note), fade_to(val, 0.45),
                  val.glow.animate.set_opacity(0), run_time=0.45, watch=False)
        self.sfx("pop", offset=0.4)
        self.play(LaggedStart(*[Create(m) if isinstance(m, Line) else FadeIn(m, scale=0.5) for m in deep],
                              lag_ratio=0.04), FadeIn(tag), *self.say_anims("树变得又窄又深", rt=1.8), run_time=1.8)
        zero = self.net_box("AlphaGo Zero", TEXT, w=4.1, h=0.95).move_to([4.45, 0.15, 0])
        assert zero.box.get_right()[0] < 6.75 and zero.box.get_left()[0] > deep.get_right()[0] + 0.2, (
            zero.box.get_left(), zero.box.get_right())

        loop = VGroup(CurvedArrow([3.9, 2.0, 0], [5.1, 2.0, 0], angle=-PI * 0.7, color=GREY_B, stroke_width=2.5),
                      CurvedArrow([5.1, 1.7, 0], [3.9, 1.7, 0], angle=-PI * 0.7, color=GREY_B, stroke_width=2.5))
        bst = make_stone("B", 0.13).move_to([4.5, 2.4, 0])
        wst = make_stone("W", 0.13).move_to([4.5, 1.3, 0])
        yr = self.T("2017", "label", DIM).next_to(zero, DOWN, buff=0.2)
        self.sfx("whoosh_soft", offset=0.4)
        self.play(ReplacementTransform(VGroup(pol.box, pol.icon, pol.title, val.box, val.icon, val.title),
                                       VGroup(zero.box, zero.icon, zero.title)), FadeOut(pol.glow), FadeOut(val.glow),
                  FadeOut(badges), run_time=0.9)
        self.add(zero)
        self.play(Create(loop), FadeIn(bst), FadeIn(wst), FadeIn(yr),
                  *self.say_anims("AlphaGo Zero 不看人类棋谱\n自己跟自己下，练得更强", rt=0.8), run_time=0.8)
        self.b7_stuff = Group(go, *levels, bright, tips, deep, tag, zero, loop, bst, wst, beam, yr)

    def net_box(self, title, color, w=3.1, h=1.2):
        sw = self.SW
        box = RoundedRectangle(width=w, height=h, corner_radius=0.2 * h).set_fill(PANEL, 1).set_stroke(color, 2.2 * sw)
        t = self.T(title, "cap", color, weight=MEDIUM)
        icon = net_icon(h * 0.9, h * 0.55, color, sw)
        inner = VGroup(icon, t).arrange(RIGHT, buff=0.3)
        if inner.width + 0.6 > w:
            w = inner.width + 0.6
            box = RoundedRectangle(width=w, height=h, corner_radius=0.2 * h).set_fill(PANEL, 1).set_stroke(color, 2.2 * sw)
        glow = box_glow_image(w, h, 0.2 * h, color, config.pixel_width / config.frame_width).move_to(box)
        glow.set_opacity(0)
        inner.move_to(box)
        g = Group(glow, box, icon, t)
        g.glow, g.box, g.icon, g.title = glow, box, icon, t
        return g

    # ------------------------------------------------------------------ beat 8: move 37 again, final card, question
    def beat8(self):
        self.mark("beat8")
        self.voice("b21")
        go, q37, st, halo, grad = self.hero_board(0.7, bottom_fade=True)
        self.sfx("whoosh", offset=0.3)
        self.play(FadeOut(self.b7_stuff), FadeOut(self.strip), FadeIn(go, scale=0.97), FadeIn(grad),
                  *self.say_anims("回到第二局的第 37 手", rt=0.9), run_time=0.9)
        rootp = np.array([q37[0] - 1.6, 3.35, 0])
        rdot = Dot(rootp, radius=0.08, color=TEXT).set_z_index(9)
        cands = [(7, 9), (10, 9), (17, 11), (11, 5), (9, 12), (12, 11)]
        branches = VGroup(*[CubicBezier(rootp, rootp + DOWN * 0.9, go.pt(*c) + UP * 1.1, go.pt(*c))
                            .set_stroke(GREY_B, 1.8, opacity=0.6) for c in cands]).set_z_index(9)
        main = CubicBezier(rootp, rootp + DOWN * 0.9, q37 + UP * 1.3, q37).set_stroke(X_COL, 1.2, opacity=0.6)
        main.set_z_index(9)
        tips = VGroup(*[Dot(go.pt(*c), radius=0.06, color=GREY_B) for c in cands]).set_z_index(9)
        for c in cands:
            assert -7.0 < go.pt(*c)[0] < 1.0 and -2.6 < go.pt(*c)[1] < 3.3, c
        tag = self.T("示意", "label", DIM, k=20 / self.px(self.T_SIZE["label"])).next_to(rdot, LEFT, buff=0.15)
        tag.set_z_index(10)

        # why move 37 was found: the curiosity bonus (schematic bars as in the 选 step, tagged 示意)
        S, BW, BY = 2.6, 0.5, -1.45
        xs = [2.75, 4.1, 5.45]
        state = {"wr": [0.48, 0.45, 0.0], "bn": [0.10, 0.12, 0.02]}

        def stack(j, wr, bn):
            t = Rectangle(width=BW, height=max(0.002, wr * S)).set_fill(WIN_COL, 0.9).set_stroke(width=0)
            t.move_to([xs[j], BY, 0], aligned_edge=DOWN)
            y = Rectangle(width=BW, height=max(0.002, bn * S)).set_fill(BONUS_COL, 0.9).set_stroke(width=0)
            y.move_to(t.get_top(), aligned_edge=DOWN)
            return VGroup(t, y).set_z_index(10)

        bars = [stack(j, state["wr"][j], state["bn"][j]) for j in range(3)]

        def bars_to(wr, bn):
            out = []
            for j in range(3):
                new = stack(j, wr[j], bn[j])
                out += [Transform(bars[j][0], new[0]), Transform(bars[j][1], new[1])]
            state["wr"], state["bn"] = list(wr), list(bn)
            return out

        labs = VGroup(self.T("走法 A", "label", DIM), self.T("走法 B", "label", DIM), self.T("第 37 手", "label", X_COL))
        for j, lb in enumerate(labs):
            lb.move_to([xs[j], BY - 0.28, 0])
        base = Line([xs[0] - 0.55, BY, 0], [xs[2] + 0.55, BY, 0], color=GRID, stroke_width=1.6)
        key = VGroup(
            VGroup(Square(0.22).set_fill(WIN_COL, 0.9).set_stroke(width=0), self.T("胜率", "label", WIN_COL)).arrange(RIGHT, buff=0.12),
            VGroup(Square(0.22).set_fill(BONUS_COL, 0.9).set_stroke(width=0), self.T("好奇心加分", "label", BONUS_COL))
            .arrange(RIGHT, buff=0.12)).arrange(RIGHT, buff=0.4)
        self.hero_place(key, 2.95)
        n_tot = self.num(1000, "label")
        tot = VGroup(self.T("总共试了", "label", DIM), n_tot, self.T("盘", "label", DIM)).arrange(RIGHT, buff=0.12)
        self.hero_place(tot, 2.35)
        tot[2].add_updater(lambda m: m.next_to(n_tot, RIGHT, buff=0.12))
        n_tot.add_updater(lambda m: m.set_z_index(10))
        ptag = self.T("示意", "label", DIM, k=20 / self.px(self.T_SIZE["label"])).move_to([self.HERO_X1 - 0.25, 3.45, 0])
        echo = self.HN(r"\approx 1/10000", 32, DIM).next_to(labs[2], DOWN, buff=0.14)
        panel = VGroup(base, labs, key, tot, ptag, echo).set_z_index(10)
        assert echo.get_bottom()[1] > -2.9 and max(m.get_right()[0] for m in panel) <= self.HERO_X1 + 0.05
        # a. the policy network: humans almost never play here, about 1 in 10000; the branch is hair-thin
        self.sfx("pop", offset=0.5)
        self.play(FadeIn(rdot), FadeIn(tag), LaggedStart(*[Create(b) for b in branches], lag_ratio=0.1), FadeIn(tips),
                  Create(main), FadeIn(VGroup(base, labs, key, tot, ptag)), *[FadeIn(b) for b in bars],
                  *self.say_anims("策略网络觉得：\n人类几乎不会下这里（约万分之一）", rt=1.1), run_time=1.1)
        self.play(FadeIn(echo, shift=UP * 0.1), Indicate(bars[2], color=BONUS_COL, scale_factor=1.3), run_time=0.6)
        self.play(Indicate(echo, color=X_COL, scale_factor=1.15), run_time=0.6)
        # b. the more games in total, the bigger the curiosity bonus: even an unusual move gets tried
        self.sfx("riser", offset=0.6)
        self.play(ChangeDecimalToValue(n_tot, 20000, rate_func=linear), *bars_to([0.48, 0.45, 0.0], [0.06, 0.07, 0.62]),
                  *self.say_anims("可试的总盘数越多，好奇心加分越大：\n冷门的一步也会被试到", rt=2.4), run_time=2.4)
        tried = SurroundingRectangle(VGroup(bars[2], labs[2]), buff=0.08, color=WHITE, stroke_width=2, corner_radius=0.06)
        self.play(Create(tried), ShowPassingFlash(main.copy().set_stroke(WHITE, 4, opacity=0.9), time_width=0.6),
                  run_time=0.8)
        # c. tried: locally it looks like a loss, but played forward the win rate is higher
        patch = RoundedRectangle(width=3.2 * go.step, height=3.2 * go.step, corner_radius=0.12).move_to(q37)
        patch.set_fill(BG, 0.55).set_stroke(GREY_B, 1.6, opacity=0.8).set_z_index(8.5)
        local = self.H("看似吃亏", 28, TEXT).next_to(patch, LEFT, buff=0.15).set_z_index(10)
        spread = glow_image(3.6, WIN_COL, config.pixel_width / config.frame_width, peak=0.24).move_to(q37).set_z_index(7.5)
        whole = self.H("全局", 28, TEXT).move_to(q37 + np.array([-2.6, -1.9, 0])).set_z_index(10)
        self.play(FadeOut(tried), FadeIn(patch), FadeIn(local, shift=RIGHT * 0.1),
                  *self.say_anims("一试才发现：眼前局部看似吃亏，\n往后推演，胜率反而更高", rt=0.8), run_time=0.8)
        spread.save_state()
        spread.scale(0.12).set_opacity(0)
        self.play(Restore(spread, rate_func=ease_out_cubic), FadeOut(patch), FadeOut(local), FadeIn(whole),
                  *bars_to([0.48, 0.45, 0.64], [0.06, 0.07, 0.32]), run_time=1.6)
        # d. higher win rate -> tried more -> this branch grows thicker
        self.play(main.animate.set_stroke(width=11, opacity=0.9), fade_to(branches, 0.5), FadeOut(spread), FadeOut(whole),
                  *bars_to([0.48, 0.45, 0.64], [0.05, 0.06, 0.10]), fade_to(VGroup(bars[0], bars[1]), 0.5),
                  *self.say_anims("胜率高 → 试得更多 → 这根树枝越长越粗", rt=1.2), run_time=1.2)
        m37 = self.H("第 37 手", 96, TEXT, SEMIBOLD)
        hum = VGroup(self.H("人类：约", 40, DIM), self.HN(r"1/10000", 40, DIM)).arrange(RIGHT, buff=0.12)
        self.hero_place(VGroup(m37, hum).arrange(DOWN, buff=0.4, aligned_edge=LEFT), 0.0)
        hum.set_z_index(10)
        n_tot.clear_updaters()
        tot[2].clear_updaters()
        self.play(FadeOut(Group(panel, *bars), shift=UP * 0.1), FadeIn(m37, shift=LEFT * 0.1),
                  ShowPassingFlash(main.copy().set_stroke(WHITE, 13, opacity=0.9), time_width=0.6),
                  *self.say_anims("它总下树里试得最多的那一步", rt=1.0), run_time=1.0)
        self.play(FadeIn(hum), *self.say_anims("人类几乎不会下的地方，它的树最粗", rt=0.9), run_time=0.9)
        self.settle()
        self.slam(st, q37, go.step, halo, rt=0.7, extra=lambda: self.say_anims("所以它敢下那步万分之一的棋", aha=2.0,
                                                                              rt=0.7))
        # final card
        self.voice("b22")
        cfg = dict(self.tree_cfg)
        cfg.update(x0=-3.6, x1=3.6, root=(0, 2.75), root_size=0.75, child_y=1.85, child_size=0.55,
                   levels={2: 1.2, 3: 0.75, 4: 0.33, 5: -0.05}, min_slot=0.75, min_span=0.2, ew=14, rmin=0.04,
                   rmax=0.09)
        icon = tree_icon(self.D["snaps"][2000], cfg, put(TOY_BOARD, 6, "X"))
        l1 = self.T("AlphaGo 的思考：", "cap", DIM)
        l2 = self.T("一棵会自己长偏的树", "title", weight=MEDIUM)
        steps = VGroup()
        for k, ch in enumerate("选长试传"):
            badge = Circle(radius=0.32).set_fill(PANEL, 1).set_stroke(GREY_B, 1.6)
            steps.add(VGroup(badge, self.T(ch, "cap", weight=MEDIUM).move_to(badge)))
            if k < 3:
                steps.add(self.T("→", "cap", DIM))
        steps.arrange(RIGHT, buff=0.25)
        row = VGroup(self.T("反复：", "cap", DIM), steps).arrange(RIGHT, buff=0.2)
        endcard = VGroup(VGroup(l1, l2).arrange(DOWN, buff=0.3), row).arrange(DOWN, buff=0.45).move_to([0, -1.6, 0])
        self.show(endcard, "AlphaGo 的思考：一棵会自己长偏的树 反复：选 → 长 → 试 → 传", f_end=0.6)
        self.play(FadeOut(Group(go, st, halo, rdot, tag, branches, main, tips, m37, hum)), FadeOut(grad),
                  *self.unsay_anims(), LaggedStart(*[FadeIn(m) for m in icon], lag_ratio=0.01),
                  LaggedStart(FadeIn(l1, shift=UP * 0.1), FadeIn(l2, shift=UP * 0.15),
                              *[FadeIn(m, shift=UP * 0.1) for m in row], lag_ratio=0.12,
                              rate_func=squish_rate_func(linear, 0, 0.6)), run_time=1.2)
        # the comment driver
        qq = VGroup(self.H("这算思考，", 72, TEXT, MEDIUM), self.H("还是只是统计？", 72, TEXT, MEDIUM))
        qq.arrange(DOWN, buff=0.25).move_to([0, 0.1, 0])
        self.retire(endcard)
        self.show(qq, "这算思考，还是只是统计？", slack=True)
        self.play(FadeOut(VGroup(icon, endcard), shift=UP * 0.15), FadeIn(qq, shift=UP * 0.15), run_time=0.7)
        self.settle()
        self.wait(1.5)
        self.voice_flush()


# ======================================================================================
# 9:16 Xiaohongshu cut (frame 4.5 x 8; bottom 15 % = y < -2.8 stays free)
# ======================================================================================
class AlphaGoMCTSShort(Base):
    SCENE_ID = "alphago_mcts_short"
    VOICE_JSON = "alphago_mcts_short.voice.json"
    T_SIZE = {"title": 30, "cap": 21, "label": 15}     # caption 21 -> about 69 px at 1080 wide
    N_SIZE = {"title": 40, "cap": 28, "label": 21}
    REF_PPU = 1080 / 4.5
    PX_FLOOR, TAG_FLOOR = 32, 32
    SW = 0.6
    CAP_Y = -2.3
    CAP_MAXW = 4.2
    SLOT = 0.7
    ROOT = np.array([0.0, 2.5, 0.0])
    ROOT_SIZE = 0.95
    CHILD_Y = 1.42
    CHILD_SIZE = 0.56
    BAR_BASE = -1.75
    HERO_Y = -2.05

    def construct(self):
        assert abs(config.frame_width - 4.5) < 0.02, "AlphaGoMCTSShort is the vertical cut (-r 1080,1920)"
        self.init_common()
        self.D = toy_data()
        self.pts, self.pi_est = pi_data()
        self.tree_cfg = dict(x0=-2.1, x1=2.1, root=self.ROOT[:2], root_size=self.ROOT_SIZE, child_y=self.CHILD_Y,
                             child_size=self.CHILD_SIZE, levels={2: 0.78, 3: 0.32, 4: -0.12, 5: -0.52}, max_depth=4,
                             tilt=False, min_slot=0.7, min_span=0.2, thr_frac=0.04, thr_min=2, ew=12, ew_min=1.0,
                             rmin=0.035, rmax=0.075, sw=0.6)
        self.v_open()
        self.v_toy()
        self.v_steps()
        self.v_tilt()
        self.v_go()
        self.v_end()
        self.mark("end")
        self.finish()

    def slot_x(self, m):
        return -2.1 + (CANDS.index(m) + 0.5) * self.SLOT

    def open_board(self):
        """The vertical crop: the board fills the frame (its edge lines stay off-screen), the move-37 point at
        50 % width / 45 % height; one smooth gradient darkens the top and bottom for the text."""
        size = 11.5
        step = size * (1 - 2 * GoBoard.MARGIN) / 18
        target = np.array([0.0, 4.0 - 0.45 * 8.0, 0])
        go = GoBoard(size, opening_stones(), 0.8)
        go.move_to(target - np.array([(M37[0] - 9) * step, (9 - M37[1]) * step, 0]))
        dim_board(go, 0.5)
        p37 = go.pt(*M37)
        assert np.linalg.norm(p37 - target) < 1e-6
        assert p37[0] + 4 * go.step > 2.25 + 0.02, "the last grid line must stay off-screen"
        st = make_stone("B", go.step * 0.47, 0.8).move_to(p37).set_z_index(5)
        halo = glow_image(go.step * 0.62, X_COL, config.pixel_width / config.frame_width).move_to(p37).set_z_index(4)

        def alpha(X, Y):
            return np.maximum(0.86 * smoothstep((Y - 1.75) / 1.45), 0.9 * smoothstep((-1.35 - Y) / 1.3))

        ppu = config.pixel_width / config.frame_width
        grad = gradient_image(alpha, -2.3, 2.3, -4.05, 4.05, ppu).set_z_index(8)
        return go, p37, st, halo, grad

    def v_teaser(self, p37):
        return self.teaser_for(p37, -2.1, 2.1, [p37[1], 1.1, 1.7, 2.2, 2.6, 2.95], sw=0.7, ew=16)

    def v_open(self):
        self.mark("open")
        go, p37, st, halo, grad = self.open_board()
        self.add(go, grad)
        self.frame_mark("board_before")
        self.voice("s00")
        hook = VGroup(self.H("人类万分之一", 84, TEXT, MEDIUM), self.H("才会下的一手", 84, TEXT, MEDIUM))
        hook.arrange(DOWN, buff=0.16).move_to([0, 2.62, 0]).set_z_index(20)
        vtag = self.clean(self.H("2016 · AlphaGo vs 李世石 · 第37手", 36, DIM)).next_to(hook, UP, buff=0.22)
        vfoot = self.clean(self.H("AlphaGo 自己的估计", 34, DIM)).next_to(hook, DOWN, buff=0.18)
        VGroup(vtag, vfoot).set_z_index(20)
        assert vtag.get_top()[1] < 3.9 and vtag.width < 4.3

        def show_hook():
            self.show(hook, "人类万分之一才会下的一手")
            return [FadeIn(self.clean(hook), shift=DOWN * 0.1), FadeIn(vtag), FadeIn(vfoot)]

        self.slam(st, p37, go.step, halo, shake=go, drop=0.2, rt=0.7, extra=show_hook)
        self.frame_mark("stone_landed")
        self.play(halo.animate(rate_func=there_and_back).scale(1.35), run_time=0.9, watch=False)
        # stakes: three hero lines in the top third (the 18 counts up in blue = the opponent)
        TOP = 2.62
        n18 = self.num(18, color=O_COL, px=150)
        s1a = VGroup(self.H("对手是拿过", 64), n18, self.H("个", 64)).arrange(RIGHT, buff=0.12, aligned_edge=DOWN)
        s1 = VGroup(s1a, self.H("世界冠军的李世石", 64)).arrange(DOWN, buff=0.16).move_to([0, TOP, 0]).set_z_index(20)
        assert s1.width <= 4.3
        n18.set_value(1)
        n18.add_updater(lambda m: m.set_z_index(20))
        s1a[2].add_updater(lambda m: m.next_to(n18, RIGHT, buff=0.12).align_to(s1a[0], DOWN))
        self.retire(hook)
        self.show(s1, "对手是拿过 18 个世界冠军的李世石", f_end=0.6)
        self.play(FadeOut(hook, shift=UP * 0.1), FadeOut(vtag), FadeOut(vfoot), FadeIn(self.clean(s1), shift=UP * 0.1),
                  run_time=0.5)
        self.sfx("tick", offset=0.9)
        self.play(ChangeDecimalToValue(n18, 18, rate_func=ease_out_cubic), run_time=1.0)
        s1a[2].clear_updaters()
        s2 = self.hero_lines("此前普遍认为：\n电脑赢职业棋手还要十年", 64).move_to([0, TOP, 0]).set_z_index(20)
        assert s2.width <= 4.3
        self.retire(s1)
        self.show(s2, "此前普遍认为：电脑赢职业棋手还要十年", f_end=0.6)
        self.play(FadeOut(s1, shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0, 0.5)),
                  FadeIn(self.clean(s2), shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0.25, 1)), run_time=0.5)
        s3 = self.hero_lines("结果 AlphaGo 4 : 1 获胜", 64).move_to([0, TOP, 0]).set_z_index(20)
        f = 150 / (EM * self.REF_PPU)
        self.floor_ok(f, "4 : 1")
        score = MathTex("4", ":", "1", font_size=f)
        score[0].set_color(X_COL)
        score[1].set_color(DIM)
        score[2].set_color(O_COL)
        score.arrange(RIGHT, buff=0.26).move_to([0, self.HERO_Y, 0]).set_z_index(20)
        assert score.get_bottom()[1] > -2.8
        self.retire(s2)
        self.show(s3, "结果 AlphaGo 4 : 1 获胜", f_end=0.6)
        self.sfx("stone", offset=0.3)
        self.play(FadeOut(s2, shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0, 0.5)),
                  FadeIn(self.clean(s3), shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0.25, 1)), FadeIn(score, scale=1.5),
                  halo.animate(rate_func=there_and_back).scale(1.3), run_time=0.6)
        # the teaser tree erupts upward out of the stone
        tree = self.v_teaser(p37).set_z_index(9)
        tlab = self.clean(self.T("示意", "label", DIM).move_to([1.7, 3.3, 0]).set_z_index(9))
        c1 = self.clean(self.H("AI 的脑子里，在长一棵树", 72, TEXT, MEDIUM).move_to([0, self.HERO_Y, 0])
                        .set_z_index(20))
        assert c1.width <= 4.25
        self.retire(s3)
        self.show(c1, "AI 的脑子里，在长一棵树", f_end=0.45)
        self.sfx("whoosh_soft", offset=0.2)
        self.play(FadeOut(s3, shift=UP * 0.1), FadeOut(score, rate_func=squish_rate_func(smooth, 0, 0.4)),
                  LaggedStart(*[GrowFromPoint(e, p37) for e in [*tree.edges, *tree.hot]], lag_ratio=0.02),
                  LaggedStart(*[FadeIn(d, scale=0.3) for d in tree.dots], lag_ratio=0.02), FadeIn(tree.glow),
                  FadeIn(tlab), FadeIn(c1, shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0.3, 0.7)), run_time=1.1)
        self.frame_mark("teaser_tree")
        self.play(*[ShowPassingFlash(e.copy().set_stroke(WHITE, e.stroke_width + 1.5, opacity=0.9), time_width=0.6)
                    for e in tree.hot], run_time=0.5, watch=False)
        self.settle()
        # the secret: 1946 solitaire. The tree folds back, a deck shuffles out of the stone for ~2 s
        dc = np.array([0, 0.95, 0])
        deck = VGroup(*[mini_card(0.9, 1.26, False).move_to(dc + UP * 0.014 * i) for i in range(20)])
        c2 = self.hero_lines("秘诀来自 1946 年的\n纸牌接龙：洗法多到算不完", 64, weight=MEDIUM)
        c2.move_to([0, self.HERO_Y + 0.05, 0]).set_z_index(20)
        assert c2.width <= 4.3 and c2.get_bottom()[1] > -2.8, (c2.width, c2.get_bottom())
        deck.save_state()
        deck.scale(0.1).move_to(p37).set_opacity(0)
        self.retire(c1)
        self.show(c2, "秘诀来自 1946 年的纸牌接龙：洗法多到算不完", f_end=0.6)
        self.sfx("shuffle", offset=0.45)
        self.play(tree.animate(rate_func=squish_rate_func(rush_into, 0, 0.6)).scale(0.02, about_point=p37).set_opacity(0),
                  FadeOut(tlab), FadeOut(go), FadeOut(st), FadeOut(halo), FadeOut(grad),
                  Restore(deck, rate_func=squish_rate_func(smooth, 0.35, 1)), FadeOut(c1, shift=UP * 0.1),
                  FadeIn(self.clean(c2), shift=UP * 0.1), run_time=0.75)
        self.remove(tree)
        for rep in range(2):
            a_half, b_half = list(deck[:10]), list(deck[10:])
            self.play(VGroup(*a_half).animate.shift(LEFT * 0.75 + DOWN * 0.05),
                      VGroup(*b_half).animate.shift(RIGHT * 0.75 + UP * 0.05), run_time=0.24)
            order = [c for pair in zip(a_half, b_half) for c in pair]
            deck.submobjects = order
            self.play(LaggedStart(*[c.animate.move_to(dc + UP * 0.014 * i) for i, c in enumerate(order)],
                                  lag_ratio=0.08), run_time=0.42)
        # play 100 games and count (illustrative, tagged 示意)
        res, cum = solitaire_games()
        S = 2.9
        square = Square(S).set_stroke(TEXT, 1.4).move_to(dc)
        o = square.get_corner(DL)
        cs = S / 10
        inner = VGroup(*[Line(o + RIGHT * cs * i, o + RIGHT * cs * i + UP * S) for i in range(1, 10)],
                       *[Line(o + UP * cs * i, o + UP * cs * i + RIGHT * S) for i in range(1, 10)]).set_stroke(GRID, 0.8)
        marks = VGroup()
        for i, win in enumerate(res):
            cen = o + np.array([(i % 10 + 0.5) * cs, (9.5 - i // 10) * cs, 0])
            marks.add((check_glyph(cs * 0.52, WIN_COL, 2.4) if win else cross_glyph(cs * 0.36, GREY_B, 1.8, 0.75))
                      .move_to(cen))
        ctr_n = self.num(cum[-1], "title", color=WIN_COL)
        crow = VGroup(self.T("成功", "cap", WIN_COL), ctr_n, self.T("/ 100", "cap", DIM))
        crow.arrange(RIGHT, buff=0.12, aligned_edge=DOWN).move_to([0, -0.95, 0])
        ctr_n.set_value(0)
        tag = self.clean(self.T("示意", "label", DIM)).next_to(square, UP, buff=0.08).align_to(square, RIGHT)
        c3 = self.hero_lines("不如随机玩 100 局，\n数数成功几局", 64, weight=MEDIUM)
        c3.move_to([0, self.HERO_Y + 0.05, 0]).set_z_index(20)
        self.retire(c2)
        self.show(c3, "不如随机玩 100 局，数数成功几局", f_end=0.6)
        self.play(FadeOut(deck, scale=0.6), Create(square), Create(inner, lag_ratio=0.05), FadeIn(crow), FadeIn(tag),
                  FadeOut(c2, shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0, 0.5)),
                  FadeIn(self.clean(c3), shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0.3, 1)), run_time=0.6)
        gt = ValueTracker(0)
        ctr_n.add_updater(lambda m: m.set_value(cum[min(100, int(gt.get_value()))]))
        crow[2].add_updater(lambda m: m.next_to(ctr_n, RIGHT, buff=0.12).align_to(crow[0], DOWN))
        self.sfx("card", offset=0.1)
        self.play(LaggedStart(*[FadeIn(m, scale=1.4) for m in marks], lag_ratio=0.5),
                  gt.animate(rate_func=linear).set_value(101), run_time=1.8)
        ctr_n.clear_updaters()
        crow[2].clear_updaters()
        # the same counting gives pi: the grid becomes the dart square
        quarter = VMobject()
        quarter.set_points_as_corners([o, o + RIGHT * S])
        quarter.append_points(Arc(radius=S, start_angle=0, angle=PI / 2, arc_center=o).points)
        quarter.add_line_to(o)
        quarter.set_fill(WIN_COL, 0.07).set_stroke(WIN_COL, 1.6)
        est = self.num(self.pi_est[10][1], "title", decimals=3)
        est_row = VGroup(self.T("圆周率 ≈", "label", DIM), est).arrange(RIGHT, buff=0.15).move_to([0, -0.95, 0])
        c4 = self.hero_lines("落进圆里的比例 × 4 ≈ π", 64, weight=MEDIUM).move_to([0, self.HERO_Y + 0.05, 0]).set_z_index(20)
        assert c4.width <= 4.3
        self.retire(c3)
        self.show(c4, "落进圆里的比例 × 4 ≈ π", f_end=0.6)
        self.play(LaggedStart(*[FadeOut(m, scale=0.4) for m in marks], lag_ratio=0.01), FadeOut(inner), FadeOut(tag),
                  DrawBorderThenFill(quarter), FadeTransform(crow, est_row),
                  FadeOut(c3, shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0, 0.5)),
                  FadeIn(self.clean(c4), shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0.3, 1)), run_time=0.6)
        imgs, prev = [], 0
        self.sfx("rain", offset=0.0)
        for b in PI_CHECKS[1:]:
            im = ImageMobject(points_image(self.pts, 600, 3.4 if b <= 100 else (2.2 if b <= 1000 else 1.6), b, prev))
            im.height = S
            im.move_to(dc).set_z_index(2)
            imgs.append(im)
            anims = [FadeIn(im)]
            anims.append(ChangeDecimalToValue(est, self.pi_est[b][1]))
            self.play(*anims, run_time=0.3, rate_func=linear)
            prev = b
        big = self.N(r"3.14", "title", WIN_COL).move_to(est, aligned_edge=LEFT)
        self.play(FadeTransform(est, big), est_row[0].animate.set_color(WIN_COL), run_time=0.45)
        self.v_open_end = Group(square, quarter, *imgs, est_row[0], big)
        self.v_c2 = c4

    def v_toy(self):
        self.mark("toy")
        self.voice("s01")
        strip = self.T("AlphaGo 在想什么？", "cap", weight=MEDIUM).move_to([0, 3.52, 0])
        ul = Line([-0.9, 3.25, 0], [0.9, 3.25, 0]).set_stroke(GRID, 1.5)
        self.title_strip = VGroup(strip, ul).set_z_index(15)
        ttt = TTTBoard(TOY_BOARD, size=3.0).move_to([0, 0.95, 0])
        self.sfx("whoosh_soft", offset=0.2)
        self.retire(self.v_c2)
        leg = self.legend("label", gw=0.2, sw=3.2, horizontal=True).move_to([0, -0.92, 0]).set_z_index(15)
        badge = self.turn_badge("cap", gw=0.26, sw=4.0).move_to([0, -1.56, 0]).set_z_index(15)
        assert badge.get_bottom()[1] > self.CAP_Y + 0.25 and badge.get_top()[1] < leg.get_bottom()[1] - 0.08
        self.sfx("pop", offset=0.5)
        self.play(FadeOut(self.v_open_end), FadeOut(self.v_c2, rate_func=squish_rate_func(smooth, 0, 0.3)),
                  FadeIn(self.title_strip, shift=DOWN * 0.1),
                  FadeIn(ttt, scale=0.9), FadeIn(leg, shift=UP * 0.1, rate_func=squish_rate_func(smooth, 0.3, 1)),
                  FadeIn(badge, scale=0.6, rate_func=squish_rate_func(smooth, 0.45, 1)),     # after the old caption left
                  *self.say_anims("现在轮到 X 下（我们）", rt=0.8), run_time=0.8)
        self.play(badge.animate(rate_func=there_and_back).scale(1.12), run_time=0.4)
        dock = badge.copy().scale(0.62)
        dock.move_to([2.12 - dock.width / 2, 2.8, 0])
        assert dock.get_bottom()[1] > ttt.get_top()[1] + 0.05
        self.play(badge.animate.scale(0.62).move_to(dock), run_time=0.45)
        self.ttt, self.leg, self.badge = ttt, leg, badge
        thr = Line(ttt.cell(2) + (ttt.cell(2) - ttt.cell(6)) * 0.12, ttt.cell(6) + (ttt.cell(6) - ttt.cell(2)) * 0.12)
        thr.set_stroke(THREAT_COL, 8, opacity=0.4).set_z_index(1)
        hot = RoundedRectangle(width=ttt.cs * 0.9, height=ttt.cs * 0.9, corner_radius=0.08)
        hot.set_stroke(THREAT_COL, 2).set_fill(THREAT_COL, 0.12).move_to(ttt.cell(6)).set_z_index(2)
        self.play(Create(thr), FadeIn(hot), *self.say_anims("O 斜着连了两个：不堵就输", rt=0.8), run_time=0.8)
        self.ghosts = {m: ttt.mark(m, "X", opacity=0.3).set_z_index(3) for m in CANDS}
        self.play(LaggedStart(*[FadeIn(g, scale=0.6) for g in self.ghosts.values()], lag_ratio=0.12), run_time=0.5)
        self.threat = VGroup(thr, hot)

    def v_steps(self):
        self.mark("steps")
        self.voice("s02")
        D = self.D
        ttt = self.ttt
        f = self.ROOT_SIZE / ttt.panel.width
        leg_to = self.leg.copy().arrange(DOWN, buff=0.16, aligned_edge=LEFT)
        leg_to.move_to([-2.12 + leg_to.width / 2, self.ROOT[1], 0])
        badge_to = self.badge.copy().move_to([self.ROOT[0] + self.ROOT_SIZE / 2 + 0.14 + self.badge.width / 2,
                                              self.ROOT[1], 0])
        assert leg_to.get_right()[0] < self.ROOT[0] - self.ROOT_SIZE / 2 - 0.08 and badge_to.get_right()[0] < 2.2
        self.play(FadeOut(self.threat), VGroup(ttt, *self.ghosts.values()).animate.scale(f, scale_stroke=True)
                  .move_to(self.ROOT), Transform(self.leg, leg_to), self.badge.animate.move_to(badge_to), run_time=0.7)
        self.root = ttt
        ttt.set_z_index(5)
        badges = VGroup()
        for k, (ch, small) in enumerate(zip("选长试传", ("选择", "扩展", "模拟", "回传"))):
            b = Circle(radius=0.34).set_fill(PANEL, 1).set_stroke(GREY_B, 1.2).move_to([-1.56 + k * 1.04, -0.95, 0])
            badges.add(VGroup(b, self.T(ch, "cap", weight=MEDIUM).move_to(b),
                              self.T(small, "label", DIM).next_to(b, DOWN, buff=0.1)))
        arrows = VGroup(*[Arrow(badges[k][0].get_right(), badges[k + 1][0].get_left(), buff=0.06, stroke_width=2,
                                color=GRID, max_tip_length_to_length_ratio=0.3) for k in range(3)])
        self.badges = badges
        self.kids, self.edges, self.ctrs = {}, {}, {}
        rb = ttt.get_bottom()
        anims = []
        for e in D["log"][:6]:
            m = e["path"][0]
            p = np.array([self.slot_x(m), self.CHILD_Y, 0])
            kid = TTTBoard(TOY_BOARD, size=self.CHILD_SIZE).move_to(p).set_z_index(5)
            xm = kid.mark(m, "X").set_z_index(6)
            edge = Line(rb, p + UP * self.CHILD_SIZE / 2, stroke_width=12 / 6, color=X_COL,
                        stroke_opacity=0.88).set_z_index(1)
            ctr = self.counter(1, SCORE_X[e["res"]], p + DOWN * 0.56)
            dim_now(ctr, 0.3)
            anims.append(AnimationGroup(
                TransformFromCopy(VGroup(ttt.panel, ttt.grid, *[ttt.marks[i] for i in (0, 2, 4)]), kid),
                ReplacementTransform(self.ghosts[m], xm), GrowFromPoint(edge, rb), FadeIn(ctr, shift=UP * 0.05)))
            self.kids[(m,)], self.edges[(m,)], self.ctrs[(m,)] = kid, edge, ctr
            kid.marks[m] = xm
            kid._xm = xm
        self.sfx("pop", offset=0.3)
        self.play(LaggedStart(*anims, lag_ratio=0.2), FadeIn(badges, lag_ratio=0.15), FadeIn(arrows), run_time=0.85)
        for kid in self.kids.values():
            self.remove(kid._xm)
            kid.add(kid._xm)

        def light(k):
            return [b[0].animate.set_fill(TEXT if j == k else PANEL, 1).set_stroke(TEXT if j == k else GREY_B, 1.2)
                    for j, b in enumerate(badges)] + [b[1].animate.set_color(BG if j == k else TEXT)
                                                      for j, b in enumerate(badges)]

        e = D["log"][6]
        m, g = e["path"]
        kid = self.kids[(m,)]
        cur = Dot(rb, radius=0.06, color=WHITE).set_z_index(12)
        self.add(cur)
        self.play(*light(0), MoveAlongPath(cur, Line(rb, kid.get_top())), *self.focus_ctrs({(m,)}, rt=0.8),
                  *self.say_anims("选：挑最有希望的路", rt=0.8), run_time=0.8)
        gp = np.array([kid.get_x(), 0.35, 0])
        gk = TTTBoard(put(put(TOY_BOARD, m, "X"), g, "O"), size=0.42).move_to(gp).set_z_index(5)
        ge = Line(kid.get_bottom() + DOWN * 0.42, gk.get_top(), stroke_width=12 / 7, color=O_COL).set_z_index(1)
        self.sfx("pop", offset=0.3)
        self.play(*light(1), GrowFromPoint(gk, ge.get_start()), Create(ge),
                  MoveAlongPath(cur, Line(kid.get_top(), gk.get_top())), *self.say_anims("长：往前多想一步", rt=0.8),
                  run_time=0.8)
        cells = put(put(TOY_BOARD, m, "X"), g, "O")
        zs = 1.0                                   # about 2.4 x the leaf
        zb = TTTBoard(cells, size=zs).move_to([0.55, 0.15, 0]).set_z_index(10)
        link = DashedLine(gk.get_right(), zb.get_left(), color=GREY_B, stroke_width=1.0).set_z_index(9)
        self.play(*light(2), TransformFromCopy(gk, zb), Create(link), *self.say_anims("试：随便把棋下完", rt=0.7),
                  run_time=0.7)
        gm = self.play_moves(zb, e["trace"], z=11)
        for i, q in e["trace"]:
            cells = put(cells, i, q)
        caps, sp = self.end_game(zb, cells, e["res"], zb.get_right() + RIGHT * 0.42, 0.3, 3.0)
        ch = self.chip(e["res"], h=0.28).next_to(gk, RIGHT, buff=0.08).set_z_index(12)
        thumb = VGroup(zb, *gm, *caps)            # stays beside the leaf through 传
        self.play(thumb.animate.scale(0.3 / zs).next_to(gk, DOWN, buff=0.08), FadeOut(link), ReplacementTransform(sp, ch),
                  run_time=0.5)
        assert thumb.get_bottom()[1] > -0.55
        rec = D["snaps"][e["it"]][(m,)]
        self.play(*light(3), ch.animate.move_to(kid.get_center()),
                  MoveAlongPath(cur, Line(gk.get_top(), kid.get_bottom())), *self.say_anims("传：把输赢记回去", rt=0.7),
                  run_time=0.7)
        self.sfx("tick", offset=0.2)
        self.play(*self.tick_anims(self.ctrs[(m,)], rec[0], x_wins(rec)), FadeOut(ch), FadeOut(cur), run_time=0.5)
        self.v_steps_extra = VGroup(gk, ge, arrows, thumb)

    def v_tilt(self):
        self.mark("tilt")
        self.voice("s03")
        D = self.D
        snaps = D["snaps"]
        L = 1.0
        head_y = -1.5
        base = Line([-2.1, head_y, 0], [2.1, head_y, 0]).set_stroke(GRID, 1.2)
        counts = {it: [snaps[it][(m,)][0] for m in CANDS] for it in (7, 20, 50, 300, 2000)}

        def wr(it, m):
            rec = snaps[it][(m,)]
            return round(100 * x_wins(rec) / rec[0])

        assert all(wr(50, m) <= 25 for m in CANDS if m != 6) and all(max(CANDS, key=lambda q: wr(it, q)) == 6
                                                                    for it in (300, 2000))

        def bar(m, frac):
            h = max(0.012, frac * L)
            return Rectangle(width=0.36, height=h).set_fill(X_COL, 0.85).set_stroke(width=0).move_to(
                [self.slot_x(m), head_y + h / 2, 0])

        bars = {m: bar(m, counts[7][k] / 7) for k, m in enumerate(CANDS)}
        nums = {m: self.num(counts[7][k], "label").next_to(bars[m], UP, buff=0.06) for k, m in enumerate(CANDS)}
        rds = {}
        for m in CANDS:                    # win rate under each branch's bar (teal = 胜率)
            r = VGroup(self.num(wr(7, m), "label", color=WIN_COL), self.T("%", "label", WIN_COL)).arrange(RIGHT, buff=0.03)
            rds[m] = r.move_to([self.slot_x(m), head_y - 0.2, 0])
            assert r.width < self.SLOT - 0.06
        rlab = VGroup(self.T("第", "label", DIM), self.num(7, "cap"), self.T("轮", "label", DIM)).arrange(RIGHT, buff=0.08)
        rlab.move_to([1.0 + rlab.width / 2, 2.97, 0])
        probe = VGroup(self.T("第", "label"), self.num(2000, "cap"), self.T("轮", "label")).arrange(RIGHT, buff=0.08)
        assert 1.0 + probe.width < 2.2, probe.width
        rlab[2].add_updater(lambda m, n=rlab[1]: m.next_to(n, RIGHT, buff=0.08))
        self.round_num = rlab[1]
        self.play(FadeOut(self.badges), FadeOut(self.v_steps_extra), Create(base), FadeIn(rlab),
                  *[TransformFromCopy(self.ctrs[(m,)][0][1], nums[m]) for m in CANDS],
                  *[FadeTransform(self.ctrs[(m,)][1].copy(), rds[m]) for m in CANDS],
                  *[GrowFromEdge(bars[m], DOWN) for m in CANDS], *[FadeOut(c) for c in self.ctrs.values()],
                  *self.say_anims("一开始，每一步试得差不多", rt=0.9), run_time=0.9)
        for m in CANDS:
            nums[m].add_updater(lambda mob, b=bars[m]: mob.next_to(b, UP, buff=0.06))
            rds[m][1].add_updater(lambda mob, r=rds[m]: mob.next_to(r[0], RIGHT, buff=0.03))
        self.tdots = {}
        self.tree_pos = {(): self.ROOT.copy(), **{(m,): self.kids[(m,)].get_center() for m in CANDS}}

        def step_to(it, rt, extra=()):
            snap = snaps[it]
            anims = AlphaGoMCTS.tree_morph(self, snap)
            for k, m in enumerate(CANDS):
                anims += [Transform(bars[m], bar(m, counts[it][k] / it)), ChangeDecimalToValue(nums[m], counts[it][k]),
                          ChangeDecimalToValue(rds[m][0], wr(it, m))]
            anims.append(ChangeDecimalToValue(self.round_num, it))
            self.play(*anims, *extra, run_time=rt)
            assert all(rds[m][0].get_value() == wr(it, m) for m in CANDS)

        step_to(20, 0.85)
        step_to(50, 0.8, extra=self.say_anims("其他 5 步没堵住：几乎盘盘输", rt=0.8))
        step_to(300, 0.85)
        hl = SurroundingRectangle(rds[6], buff=0.05, color=WIN_COL, stroke_width=2.0, corner_radius=0.05)
        self.play(Create(hl), run_time=0.4)
        self.settle()
        self.sfx("riser", offset=0.0)
        self.sfx("success", offset=1.8)
        step_to(2000, 2.0, extra=self.say_anims("左下角堵住了：\n胜率最高，越试越多", rt=2.0))
        self.voice("s04")
        ring = SurroundingRectangle(self.kids[(6,)].panel, buff=0.05, color=X_COL, stroke_width=2.2, corner_radius=0.06)
        dim_rest = [fade_to(VGroup(bars[m], nums[m], rds[m], self.kids[(m,)]), 0.4) for m in CANDS if m != 6]
        self.play(*dim_rest, Create(ring), Indicate(rds[6], color=WIN_COL, scale_factor=1.15), run_time=0.8)
        self.frame_mark("bars_79")
        for m in CANDS:
            nums[m].clear_updaters()
            rds[m][1].clear_updaters()
        x6 = self.root.mark(6, "X").set_z_index(8)
        self.sfx("stone", offset=0.3)
        self.play(FadeIn(x6, scale=1.6), FadeOut(self.badge), run_time=0.5)        # X has moved
        self.remove(x6)
        self.root.add(x6)
        self.v_tilt_stuff = VGroup(base, rlab, ring, hl, *bars.values(), *nums.values(), *rds.values())

    def v_go(self):
        self.mark("go")
        self.voice("s05")
        tree_bits = VGroup(*self.kids.values(), *self.edges.values(), *self.tdots.values())
        go = GoBoard(2.9, opening_stones(), 0.5).move_to([0, 1.55, 0]).set_z_index(5)
        p = go.pt(*M37)
        self.sfx("whoosh", offset=0.7)          # >= 0.8 s after the stone of X's move
        self.play(FadeOut(self.v_tilt_stuff), FadeOut(tree_bits), FadeOut(self.root), FadeOut(self.leg),
                  FadeIn(go, scale=0.9),
                  *self.unsay_anims(), run_time=0.8)
        levels, ends, angs = fractal_fan(p, -PI / 2 - 0.87, -PI / 2 + 0.87, lengths=(1.1, 0.45, 0.2),
                                         widths=(0.6, 0.45, 0.35), seed=5)
        for m in levels:
            m.set_z_index(6)
            set_fan(m, 0.0)
        self.add(*levels)
        pol = AlphaGoMCTS.net_box(self, "策略网络", X_COL, w=3.4, h=0.66).move_to([0, -0.5, 0])
        val = AlphaGoMCTS.net_box(self, "价值网络", WIN_COL, w=3.4, h=0.66).move_to([0, -1.32, 0])
        assert val.box.get_bottom()[1] > self.CAP_Y + 0.34 + 0.15 and pol.box.get_top()[1] < -0.05
        keep = [38, 92, 125, 160, 212]
        bright = VGroup(*[Line(p, ends[i]).set_stroke(X_COL, 2.4, opacity=0.95) for i in keep]).set_z_index(7)
        tips = VGroup(*[Dot(ends[i], radius=0.045, color=X_COL) for i in keep]).set_z_index(8)
        self.sfx("pop", offset=0.3)
        self.play(fan_grow(levels[0], 0.55, lag=0.5), run_time=0.55)
        self.play(fan_grow(levels[1], 0.6, lag=0.7), fan_grow(levels[2], 0.6, lag=0.7), FadeIn(pol, shift=UP * 0.1),
                  *self.say_anims("棋感 = 学过人类高手棋谱：\n升级「选」", rt=0.6), run_time=0.6)
        self.sfx("whoosh_soft", offset=0.2)
        self.play(*[fade_level(m, min(1.0, 0.06 / op), run_time=0.7) for m, op in zip(levels, (0.2, 0.11, 0.08))],
                  Create(bright),
                  FadeIn(tips, scale=0.5), pol.glow.animate.set_opacity(1.0), run_time=0.7)
        dial = gauge(0.2, 0.68, sw=self.SW).move_to(ends[keep[2]] + DOWN * 0.28).set_z_index(9)
        self.play(FadeIn(val, shift=UP * 0.1), fade_to(pol, 0.45), *self.say_anims("大局观 = 一眼估胜率：\n升级「试」", rt=0.45),
                  run_time=0.45)
        self.sfx("pop", offset=0.25)
        self.play(FadeIn(dial, scale=0.5), val.glow.animate.set_opacity(1.0), run_time=0.4)
        self.v_go_stuff = Group(go, *levels, bright, tips, pol, val, dial)

    def v_end(self):
        self.mark("end_card")
        self.voice("s06")
        go, q37, st, halo, grad = self.open_board()
        self.sfx("whoosh", offset=0.2)
        rootp = np.array([0, 2.95, 0])
        rdot = Dot(rootp, radius=0.06, color=TEXT).set_z_index(9)
        tag = self.T("示意", "label", DIM).next_to(rdot, RIGHT, buff=0.1).set_z_index(9)
        self.play(FadeOut(self.v_go_stuff), FadeIn(go, scale=0.97), FadeIn(grad), FadeIn(rdot), FadeIn(tag),
                  *self.say_anims("策略网络觉得人类\n几乎不会下（约万分之一）", rt=0.9), run_time=0.9, watch=False)
        cands = [(12, 11), (16, 6), (12, 6), (11, 9), (16, 11)]
        for c in cands:
            assert -2.1 < go.pt(*c)[0] < 2.1 and go.pt(*c)[1] > self.CAP_Y + 0.34 + 0.4, c
        branches = VGroup(*[CubicBezier(rootp, rootp + DOWN * 0.7, go.pt(*c) + UP * 0.9, go.pt(*c))
                            .set_stroke(GREY_B, 1.4, opacity=0.65) for c in cands]).set_z_index(9)
        main = CubicBezier(rootp, rootp + DOWN * 0.7, q37 + UP * 1.0, q37).set_stroke(X_COL, 0.9, opacity=0.6).set_z_index(9)
        tips = VGroup(*[Dot(go.pt(*c), radius=0.06, color=GREY_B) for c in cands]).set_z_index(9)
        self.play(LaggedStart(*[Create(b) for b in branches], lag_ratio=0.1), FadeIn(tips), Create(main), run_time=0.8)
        self.play(main.animate.set_stroke(width=7, opacity=0.9), fade_to(branches, 0.5),
                  *self.say_anims("但好奇心加分让它去试：\n一试，胜率反而更高", rt=1.4), run_time=1.4)
        self.play(ShowPassingFlash(main.copy().set_stroke(WHITE, 8, opacity=0.9), time_width=0.6),
                  *self.say_anims("它总下试得最多的那一步：\n第 37 手就是这样选出来的", rt=0.7), run_time=0.7)
        self.slam(st, q37, go.step, halo, rt=0.6)
        # the comment driver, then loop back to the opening composition (no end card in the short cut: length)
        qq = VGroup(self.H("这算思考，", 84, TEXT, MEDIUM), self.H("还是只是统计？", 84, TEXT, MEDIUM))
        qq.arrange(DOWN, buff=0.2).move_to([0, 0.4, 0]).set_z_index(20)
        self.show(qq, "这算思考，还是只是统计？", slack=True, f_end=0.55)
        self.play(FadeOut(Group(go, st, halo, rdot, tag, branches, main, tips)), FadeOut(grad), *self.unsay_anims(),
                  FadeIn(qq, shift=UP * 0.15, rate_func=squish_rate_func(smooth, 0, 0.55)), run_time=0.8)
        self.settle()
        go2, p2, st2, halo2, grad2 = self.open_board()
        self.retire(qq)
        self.play(FadeOut(qq), FadeOut(self.title_strip), FadeIn(go2), FadeIn(st2), FadeIn(halo2), FadeIn(grad2),
                  run_time=0.8, watch=False)
        self.wait(0.4)


# ======================================================================================
# 4K cover plates: the full teaser tree out of the stone, no text (render with -s at 3840x2160 / 2160x3840)
# ======================================================================================
class CoverPlate16(AlphaGoMCTS):
    def construct(self):
        self.init_common()
        self.D = toy_data()
        go, p37, st, halo, grad = self.hero_board(0.5)
        tree = self.hero_teaser(p37).set_z_index(9)
        self.add(go, grad, halo, st, tree)
        px = ((p37[0] + config.frame_width / 2) * config.pixel_width / config.frame_width,
              (4.0 - p37[1]) * config.pixel_height / 8.0)
        print(f"[COVER] stone centre px = ({px[0]:.1f}, {px[1]:.1f}) of {config.pixel_width}x{config.pixel_height}",
              file=sys.stderr)


class CoverPlate9(AlphaGoMCTSShort):
    def construct(self):
        self.init_common()
        self.D = toy_data()
        go, p37, st, halo, grad = self.open_board()
        tree = self.v_teaser(p37).set_z_index(9)
        self.add(go, grad, halo, st, tree)
        px = ((p37[0] + config.frame_width / 2) * config.pixel_width / config.frame_width,
              (4.0 - p37[1]) * config.pixel_height / 8.0)
        print(f"[COVER] stone centre px = ({px[0]:.1f}, {px[1]:.1f}) of {config.pixel_width}x{config.pixel_height}",
              file=sys.stderr)
