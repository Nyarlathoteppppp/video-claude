"""Shared look for every COMP90054 Manim clip (same colours as the website's js/ui/scene.js).

LaTeX (BasicTeX + dvisvgm) is installed: use MathTex for formulas (split into substrings so each
part can be coloured and transformed) and DecimalNumber for changing numbers. Chinese text must
stay in Text(font=CN_FONT) because the LaTeX setup has no CJK fonts.
"""
import math
import os

# LaTeX lives in /Library/TeX/texbin, which only login shells have on PATH (tip from the 90051 session)
if '/Library/TeX/texbin' not in os.environ.get('PATH', ''):
    os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ.get('PATH', '')

from manim import *

config.background_color = "#14161b"

BG = "#14161b"
PANEL = "#1b1e25"
GRID = "#2c313b"
LINE = "#4a505c"
TEXT = "#ece9e1"
DIM = "#8e939c"

V_COL = "#58C4DD"      # state value V, path cost g, closed list
Q_COL = "#5CD0B3"      # action value Q
R_COL = "#F4D345"      # reward r
GAMMA_COL = "#83C167"  # discount gamma, f = g + h
G_COL = "#F0AC5F"      # return G, heuristic h
ALPHA_COL = "#9A72AC"  # learning rate alpha, weights w
NEG_COL = "#FC6255"    # TD error delta, negative values
TARGET_COL = "#D147BD" # targets
OPEN_COL = "#FF862F"   # policy / open list

CN_FONT = "PingFang SC"


def cn(s, size=28, color=TEXT, **kw):
    """Chinese (or mixed) text."""
    return Text(s, font=CN_FONT, font_size=size, color=color, **kw)


def mono(s, size=26, color=TEXT, **kw):
    """Formula-like text in a clean Latin font (Unicode maths such as γ, Σ, ←, ⁺)."""
    return Text(s, font="Helvetica Neue", font_size=size, color=color, **kw)


def num(x, digits=3, size=24, color=TEXT):
    """A number as Text (rebuild it to change the value; DecimalNumber needs LaTeX)."""
    return Text(f"{x:.{digits}f}".replace("-", "−"), font="Helvetica Neue", font_size=size, color=color)


def caption(scene, s, old=None, size=26):
    """Bottom caption that replaces the previous one. Returns the new mobject."""
    new = cn(s, size=size, color=TEXT).to_edge(DOWN, buff=0.35)
    if old is None:
        scene.play(FadeIn(new, shift=UP * 0.15), run_time=0.6)
    else:
        scene.play(FadeOut(old, shift=UP * 0.15), FadeIn(new, shift=UP * 0.15), run_time=0.6)
    return new


def tex(*parts, size=40, colors=None, **kw):
    """MathTex split into parts; colors maps part index -> colour, e.g. tex(r"V(s)", r"=", r"\\max_a", colors={0: V_COL})."""
    m = MathTex(*parts, font_size=size, color=TEXT, **kw)
    for i, c in (colors or {}).items():
        m[i].set_color(c)
    return m


# ---------------------------------------------------------------- reading time for captions
def reading_time(s):
    """Seconds a viewer needs to read caption s while also watching the picture.
    1 s to notice it, 5 CJK characters per second, 8 Latin letters / digits / symbols per second, at least 2 s."""
    cjk = sum(1 for ch in s if "一" <= ch <= "鿿")
    other = sum(1 for ch in s if not ch.isspace() and not ("一" <= ch <= "鿿") and ch not in "，。：；、！？“”（）,.:;!?()")
    return max(2.0, 1.0 + cjk / 5 + other / 8)


class ReadingClock:
    """Caption timing with split attention: while something moves under a caption the viewer is watching,
    not reading, so animation time counts only SPLIT (half) towards reading; static holds count in full.
    Rule per caption: time on screen >= reading_time(text) + SPLIT * animation time under it.

        clock.shown(self, text)   # when the caption appears (before the play that fades it in)
        clock.hold(self, 1.0)     # a static pause under the caption (counts in full)
        clock.ready(self)         # before the caption is replaced / faded: waits for whatever is missing
        clock.report()            # at the end: prints every caption's numbers and asserts the rule
    Plain self.wait() between shown() and ready() is counted as animation (conservative)."""

    SPLIT = 0.5

    def __init__(self):
        self.text = None
        self.t0 = 0.0
        self.still = 0.0
        self.log = []

    def shown(self, scene, text):
        if self.text is not None:
            self.ready(scene)
        self.text, self.t0, self.still = text, scene.time, 0.0

    def hold(self, scene, seconds):
        t = scene.time
        scene.wait(seconds)
        self.still += scene.time - t

    def ready(self, scene):
        if self.text is None:
            return
        elapsed = scene.time - self.t0
        moving = max(0.0, elapsed - self.still)
        need = reading_time(self.text) + self.SPLIT * moving
        if need - elapsed > 1e-6:
            # Manim freezes int(duration * fps) frames, so round up to whole frames (+ half a frame against truncation)
            fps = config.frame_rate
            scene.wait((math.ceil((need - elapsed) * fps - 1e-6) + 0.5) / fps)
        self.log.append((self.text, reading_time(self.text), moving, need, scene.time - self.t0))
        self.text = None

    def report(self):
        print("[CAPTIONS] reading  anim  need  on-screen  text")
        for text, read, moving, need, total in self.log:
            print(f"[CAPTIONS] {read:5.1f} {moving:5.1f} {need:5.1f} {total:6.1f}   {text[:40]}")
            assert total >= need - 1e-6, (text, total, need)


# ---------------------------------------------------------------- shared sound effects (manim/sfx, licence-free)
SFX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sfx")


def sfx(scene, name, offset=0.0, extra_db=0.0):
    """Add a sound from manim/sfx; offset (s from now) lands the hit on the frame where the motion lands.
    Mixing rules: manim/sfx/README.md (few accents, never stacked, small sounds on groups)."""
    import json
    with open(os.path.join(SFX_DIR, "gains.json")) as fh:
        gain = json.load(fh)[name]
    scene.add_sound(os.path.join(SFX_DIR, name + ".wav"), time_offset=offset, gain=gain + extra_db)
