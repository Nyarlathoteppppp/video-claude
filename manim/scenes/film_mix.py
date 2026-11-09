"""《字成精》's whole sound on the delivered reel: effects, room tones and score laid on the film's clock, mastered for the web.

    python film_mix.py plan                  # the cue sheet in reel time: which file and take each cue uses, what is missing or waits for its picture
    python film_mix.py takes                 # every take measured (length, peak, RMS, loudness, onsets, spectral centroid, flatness) and the take each sound picks by default
    python film_mix.py events                # the stroke times of the poem, the title, 十二's four and 一's 鬼, from their own scripts, into sound/events.json (minutes; rerun when they change)
    python film_mix.py mix [cue=take ...]    # sound/film_mix.wav: 48 kHz stereo, 24 bit, exactly the reel's length, mastered and measured
    python film_mix.py listen                # sound/listen/NAME.wav: each sound's takes one after another, levelled alike, for 奈亚子 to choose from
    python film_mix.py mux VIDEO             # VIDEO with the mix under it, next to it as VIDEO_mix.mp4 (the picture copied, not encoded again)

On the Studio with the render env's numpy, scipy and ffmpeg. A choice is cue=N or cue=sound:N, by a cue's id or by a sound's name: brush=2 takes the brush's
second take everywhere, s1=3 the first strike's third, ident_seal=strike_clay:3 another sound for one cue. The fragments follow the theme's take unless named.

The reel, as film_rough.film writes it: film 0 to HEAD_S (序, the outer window), the ident's last IDENT_S - GROW seconds off the clock, film HEAD_S to the end, the
credits, black, the stinger. So reel = film below HEAD_S and film + IDENT_S - GROW from there; the ident's own second t is reel HEAD_S - GROW + t, and its first GROW
seconds play inside the growing card. Every time here is read from ink_subs, from the cut (film_rough.cut, read only) or from a clip's own script (its constants read
with ast, never run): nothing is a film time written in, so the mix re-times itself when lines or shots move. A cue whose shot is still a card in the cut waits.

How the rules of film.director.md 3.8 are kept:
- every action has its sound, put so that its onset lands on the frame of the action;
- the music only ever comes in under a line ducked, never swells, and stops dead: three fragments, each the opening of the chosen theme take and a little longer
  each time, each ending as a poem is written; the whole theme once, under the fourth row and the reading; the credits' music stops as the thanks stops;
- the poem's writing has only the brush on paper (everything else is gated off) until 十四, where the theme plays with it;
- true digital silence after the seventh strike (the one breath out excepted) through the dawn, at the match cut to the clay, before the title's seal, in the black;
- tails cross a cut only after the first six strikes; 一 stops dead at the cut to 二;
- the chewing comes in before the cut to 八, the night room tone before the cut to 九, the rain before 声声慢;
- the reply's text is never voiced; the title's seal is the ident's own seal, much lighter; the crab slurps each credit row as film_credits times it.
"""
import ast
import json
import math
import os
import subprocess
import sys
import time

# FILM_MIX_SRC: a folder at this one's depth (manim/<name>) holding newer copies of scripts not yet on this machine, ink_subs.py and film_rough.py above all; they are
# imported and read ahead of these, so a rough mix can follow the current cut before it is pushed. Unset, everything comes from here.
SRC = os.path.abspath(os.environ["FILM_MIX_SRC"]) if os.environ.get("FILM_MIX_SRC") else None
if SRC:
    sys.path.insert(0, SRC)

import ink_subs
from ink_subs import at, end_at

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FILM = os.path.join(ROOT, "ink", "film_out", "film")
SOUND = os.path.join(FILM, "sound")
SCORE = os.path.join(SOUND, "score")
OLD_SFX = os.path.join(ROOT, "ink", "bgm", "sfx")       # the ink film's effects, one take each: the myth's wind and the old moan
OUT = os.path.join(SOUND, "film_mix.wav")
CREDITS_DEFAULT = "B"                    # how the credits' crab sounds: "wet" (slurps, chews, gulp), "A" (dry paper, soft clay), "B" (music and the landing only); 奈亚子 chose B on 2026-11-07
CREDITS = CREDITS_DEFAULT                # mix --credits=X makes film_mix_X.wav with the default's master gain
CRAB_DB = dict(pat=-39.0, whisk=-38.0, thud=-35.0)     # 三 and 八's eating since 2026-11-07: the credits A's soft clay and dry paper, 3 dB under A's own levels
EVENTS = os.path.join(SOUND, "events.json")
SR, FPS = 48000, 30
TARGET_I, CEILING_TP = -16.0, -1.0       # integrated LUFS for the web, and the true peak never over this
BODY_I = -20.5                           # 2026-11-07: the master's gain puts the film itself (reel 0 to the credits, before the limiter) here, where rough v5 had it.
                                         # The reel's -16 had come from the loud credits; with them down, -16 would lift the whole film 3.9 dB. None: the reel to TARGET_I
MASTER_GAIN_DB = 6.2636                  # FROZEN 2026-11-07: the gain BODY_I gave the mix with the credits down (sound/film_mix_old7.wav). The film's music then went
                                         # down 4 dB (MUSIC_DB); normalising again would win most of that back and lift the quiet chapters with it, so the gain stays
                                         # here. None: back to BODY_I / TARGET_I
STROKE_PEAK_DB = -1.5                    # 2026-11-07: no stroke on the brush bus peaks over this before the limiter (after the frozen master gain), so the limiter
CAP_ALSO = {"grain"}                     # cues off the brush bus that take the same cap (2026-11-07: the myth's grain, 1.3 dB into the limiter before)
STROKE_LOOK = .005                       # need not touch them; a gain on the stroke alone that dips this long before each spike and comes back over the same (cap_peaks)
MUSIC_DB = dict(frag=-29.0, theme=-23.0) # the film's music, each 4 dB under what it was (frag -25/-24/-23, theme -19): 奈亚子 on 2026-11-07, 「一起压低中等水平」
DUCK_DB = -7.0                           # the music under a line
NARRATION = os.path.join(SOUND, "narration.wav")   # film_voice.py track: every line spoken, on the reel clock (2026-11-08, 奈亚子 chose the voice, then 「直接定稿」)
VOICE_DB = -6.26                         # that track (-22 LUFS mono) on the voice bus: doubled to stereo and through the frozen master gain it reads about -19 LUFS,
                                         # where the narrated preview she approved had it
VOICE_DUCK_ON = -45.0                    # the music also ducks wherever the track is above this (dBFS), so a line that runs past its subtitle keeps the music down
FRAGMENT = (4.0, 6.0, 10.0)              # about how long the three fragments run before the first three poems: a little more of the theme each time
RIVER_HP = 300.0                         # the dry wind's rumble taken off, 8th order (MOSS gave all six wind takes a rumble under 200 Hz)
LINE_RATE, LINE_LP = 1 / 3, 1500.0       # the line in clay: MOSS gave every take as a 7-10 kHz scrape; played at a third of its speed and low-passed it is a slow, soft drag
CREDITS_MUS_DB = -26.0                   # the credits' music (was -21): 奈亚子 found the background too loud on 2026-11-07; about -20 LUFS over the credits after the master's gain
THEME_S = 16.0                           # the whole theme, as the fourth poem's beat runs
LEAD_9 = 1.0                             # the night room of 九 comes in this long before the cut, under the first poem's rest


# ---------------------------------------------------------------- reading other scripts without running them

class _Skip(Exception):
    pass


_trees, _consts = {}, {}
_SAFE = dict(min=min, max=max, abs=abs, round=round, float=float, int=int, len=len, tuple=tuple, list=list)


def _tree(script):
    if script not in _trees:
        path = os.path.join(SRC, script) if SRC and os.path.exists(os.path.join(SRC, script)) else os.path.join(HERE, script)
        with open(path, encoding="utf-8") as f:
            _trees[script] = ast.parse(f.read())
    return _trees[script]


def _ev(node, env, script):
    """A constant expression of a script: literals, sums of names already read, ink_subs.at/end_at, dict(...), curve(...) as its points, a list comprehension,
    and the script's own one-line functions. Anything else is not a constant and is skipped."""
    ev = lambda n, e=env: _ev(n, e, script)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        out = []
        for e in node.elts:
            if isinstance(e, ast.Starred):
                out.extend(ev(e.value))
            else:
                out.append(ev(e))
        return tuple(out) if isinstance(node, ast.Tuple) else out
    if isinstance(node, ast.Dict):
        return {ev(k): ev(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = ev(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = ev(node.left), ev(node.right)
        ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b, ast.Div: lambda: a / b, ast.Pow: lambda: a ** b, ast.FloorDiv: lambda: a // b, ast.Mod: lambda: a % b}
        if type(node.op) not in ops:
            raise _Skip
        return ops[type(node.op)]()
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise _Skip
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "ink_subs" and not node.attr.startswith("_"):
            return getattr(ink_subs, node.attr)
        if isinstance(node.value, ast.Name) and node.value.id == "math":
            return getattr(math, node.attr)
        raise _Skip
    if isinstance(node, ast.Subscript):
        return ev(node.value)[ev(node.slice)]
    if isinstance(node, ast.ListComp):
        out = []

        def walk(gens, e):
            if not gens:
                out.append(ev(node.elt, e))
                return
            g = gens[0]
            for item in ev(g.iter, e):
                e2 = dict(e)
                _bind(g.target, item, e2)
                if all(ev(c, e2) for c in g.ifs):
                    walk(gens[1:], e2)
        walk(node.generators, env)
        return out
    if isinstance(node, ast.Call):
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "ink_subs" else f.id if isinstance(f, ast.Name) else None
        args = []
        for a in node.args:
            if isinstance(a, ast.Starred):
                args.extend(ev(a.value))
            else:
                args.append(ev(a))
        kw = {k.arg: ev(k.value) for k in node.keywords}
        if name in ("at", "end_at"):
            return getattr(ink_subs, name)(*args, **kw)
        if name == "dict":
            return dict(*args, **kw)
        if name == "curve":
            return [tuple(p) for p in args]
        if name in _SAFE:
            return _SAFE[name](*args, **kw)
        fn = next((n for n in _tree(script).body if isinstance(n, ast.FunctionDef) and n.name == name), None)
        body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))] if fn else []
        if fn and len(body) == 1 and isinstance(body[0], ast.Return):
            e2 = dict(env)
            for p, v in zip(fn.args.args, args):
                e2[p.arg] = v
            return ev(body[0].value, e2)
    raise _Skip


def _bind(target, value, env):
    if isinstance(target, ast.Name):
        env[target.id] = value
    elif isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == len(value):
        for t, v in zip(target.elts, value):
            _bind(t, v, env)
    else:
        raise _Skip


def consts(script, fn=None):
    """The constants a script sets at its top level (and, with fn, in that function's body), read and worked out without importing it."""
    key = (script, fn)
    if key in _consts:
        return _consts[key]

    def run(stmts, env):
        for st in stmts:
            if isinstance(st, ast.Assign) and len(st.targets) == 1:
                try:
                    _bind(st.targets[0], _ev(st.value, env, script), env)
                except (_Skip, KeyError, TypeError, ValueError, ZeroDivisionError, AttributeError, IndexError):
                    pass
    env = {}
    run(_tree(script).body, env)
    if fn:
        env = dict(env)
        run(next(n for n in ast.walk(_tree(script)) if isinstance(n, ast.FunctionDef) and n.name == fn).body, env)
    _consts[key] = env
    return env


# ---------------------------------------------------------------- the reel and the cut

_R = consts("film_rough.py")
HEAD_S, GROW, IDENT_S, CREDITS_S, BLACK_S = _R["HEAD_S"], _R["GROW"], _R["IDENT_S"], _R["CREDITS_S"], _R["BLACK_S"]
RING = consts("film_ring.py")
STINGER_S = RING["STINGER_S"]
OFF = IDENT_S - GROW           # the ident's part off the clock


def film_end():
    return ink_subs.chapters()[-1][3]


def R(t):
    """Reel time of film time t."""
    return t if t < HEAD_S else t + OFF


def IDENT(t):
    """Reel time of the ident's own second t: it starts as the card starts to grow."""
    return HEAD_S - GROW + t


def press_hits():
    """Film times the seven dies touch the clay in the cut (film_rough.PRESS_CONTACT, frames after 「古人说」), and where the groove starts."""
    T3 = ch("三")
    return [T3 + f / 30 for f in _R["PRESS_CONTACT"]], T3 + next(r[1] for r in _R["PRESS_CUT"] if r[0] == "groove") / 30


def CRED(t):
    return film_end() + OFF + t


def BLACK(t):
    return CRED(CREDITS_S) + t


def STING(t):
    return BLACK(BLACK_S) + t


def reel_len():
    return STING(STINGER_S)


def ch(name, k=2):
    return next(c[k] for c in ink_subs.chapters() if c[0] == name)


_cut = []


def cut():
    if not _cut and not getattr(cut, "failed", False):
        try:
            import film_rough
            _cut.extend(film_rough.cut())
        except Exception as e:          # the mix still runs on ink_subs alone, and says so, once
            cut.failed = True
            print("film_rough.cut() unavailable:", e, flush=True)
    return _cut


def shot_of(part):
    """The first shot of the cut playing a file whose path has this in it."""
    for s in cut():
        p = s.what.get("path") or s.what.get("folder") or ""
        if part in p.replace(os.sep, "/"):
            return s
    return None


def is_card(t):
    """Is the picture at film time t still a card saying what will be there?"""
    s = next((s for s in cut() if s.t0 <= t < s.t1), None)
    return bool(s and s.what["kind"] == "card")


# ---------------------------------------------------------------- the sounds and what they should be

# what each sound should sound like, for the cue sheet and 奈亚子's listening list
DESC = dict(
    room_night="深夜关窗的房间底噪：几乎无声，电脑极轻的嗡声", room="白天的房间，窗略开，远处极轻的街声", city_night="深夜住宅区，高处开窗：远处车流、空调外机低鸣",
    enter="回车键一下，软而实", mouse="鼠标一下，干、近", typing="慢速打字，软键程，一下一下分开", keys_fast="一串快速敲键，最后一下回车",
    whoosh="石印下落带起的极轻风声", seal_heavy="寿山石印压在垫毡的宣纸上：一声闷、厚、软的重压", peel="印面离纸，纸轻轻松开的粘连声", seal_light="同款落印，极轻",
    wind="夜里高山裸石上的冷风，细而孤", scratch="细枝尖在湿泥里划出一道，细、略带砂感，走走停停", grain_mud="谷粒从天上落进湿泥：先密后疏，最后几粒",
    cry="远山夜里一声空洞的长哭，像风穿石孔升成哀嚎再散去，远处山脊有更弱的应和；不是人声", moan="旧片的长哭（风过远石），单 take",
    brush="湿毛笔在宣纸上行笔的沙沙声，一笔一声", strike_clay="厚泥上的闷响，和厂标落印同一家族", strike_wood="木模压泥：木头的钝响带湿声", strike_bronze="铜印压泥：闷响带极弱的金属余音",
    strike_lead="铅字压泥：死沉一下，带极小的铅字碰撞", strike_hard="钢模硬压：短、硬、紧", strike_last="最重一击：低沉压碎，桌子一震，随后全静", exhale="深夜里极轻、疲惫的一口鼻息",
    clay_morph="湿泥自己收拢、折叠：轻的湿响和黏响，碎屑落在木板上", clay_yawn="泥做的小东西打一个小哈欠：湿泥拉伸的极轻吱声，可爱，不是人声",
    clay_thud="一小块软泥放在木板上：一声软而闷的拍", clay_step="小泥脚在木桌上走几步：极轻的软拍", chew="小东西嚼一口软泥：轻、湿、软，可爱",
    slurp="小东西把一根细面条吸进嘴里：轻、湿、快", gulp="小东西咽下一口：一声软湿的咕嘟", riverbed="万米高空看干裂的河床：细、干、稳的风，空，没有水", dry_wind="干裂的泥地上一阵细、干、轻的风，几乎无声，没有低频轰鸣",
    furrow="一根细枝或芦管慢慢划过软湿的泥：柔、低、闷的一道拖痕，走走停停，没有嘶声",
    rain_frame="黄昏细雨打在旧木窗棂和窗台上，稀、慢", drop="一滴雨从窗框边落到石台上：一声小而清的滴", lamp="夜里油灯的火苗极轻地颤，几乎无声", page_air="一阵穿堂风掀起一页旧书，翻过去轻轻落下",
    theme="主题：全片只完整出现一次（十四写满诗时）；碎片都是它的开头", credits="片尾 146 秒：同一家族的乐器，温和、稍轻快，在「特别鸣谢」处停",
)
FAMILIES = ("古琴独奏", "毡锤钢琴独奏", "古琴加极轻的低音大提琴")      # the score's take N is family N (film_score.FAMILY)
OLD = ("wind", "moan", "reeds", "grain", "rain", "rumble", "page")       # sounds that exist only as the ink film's take 1
# takes chosen by their envelopes in film_sfx.py (not by ear); everything else picks by measure (see takes) until 奈亚子 chooses
DEFAULT = dict(furrow=1, dry_wind=1, brush=2, page_air=3, theme=3, strike_clay=3, strike_wood=3, strike_bronze=1, strike_lead=3, strike_hard=3, keys_fast=1, mouse=1, strike_last=1, exhale=2, enter=3, room=1, typing=1)
CUE_DEFAULT = dict(s6=("strike_hard", 1))
HITS = {"enter", "mouse", "whoosh", "seal_heavy", "seal_light", "peel", "strike_clay", "strike_wood", "strike_bronze", "strike_lead", "strike_hard", "strike_last", "exhale",
        "clay_thud", "clay_yawn", "drop", "page_air", "gulp", "keys_fast"}
PIECES = {"chew", "clay_step"}
MUSIC = {"theme", "credits"}


def path_of(name, take):
    for p in (os.path.join(SOUND, f"{name}_take{take}.wav"), os.path.join(SCORE, f"{name}_take{take}.flac"), os.path.join(OLD_SFX, f"{name}_take{take}.wav")):
        if os.path.exists(p):
            return p
    return None


def takes_of(name):
    return [k for k in range(1, 10) if path_of(name, k)]


# ---------------------------------------------------------------- the cues

def C(id, name, t, kind="hit", db=-12.0, bus="fx", **kw):
    """One cue. t is reel seconds. hit: its onset (or peak, align="peak") on t, db its peak; piece: the next single event cut from the take, onset on t;
    seg: dur seconds of the take's sounding part from t, shaped by env, db its loudness; bed: the take looped over dur from t, db its loudness;
    music: dur seconds of the take from off, db its loudness. cut: nothing of it after this reel second. pending: its picture is still a card."""
    return dict(id=id, name=name, t=float(t), kind=kind, db=float(db), bus=bus, dur=kw.pop("dur", None), fi=kw.pop("fi", .005), fo=kw.pop("fo", .06), keep=kw.pop("keep", None),
                env=kw.pop("env", "flat"), lp=kw.pop("lp", None), hp=kw.pop("hp", None), rate=kw.pop("rate", None), cut=kw.pop("cut", None), off=kw.pop("off", 0.0), align=kw.pop("align", "onset"), pending=kw.pop("pending", False),
                anchor=kw.pop("anchor", ""), what=kw.pop("what", ""), **kw)


def eating(beats, chews, gap=.4):
    """The crab's eating in 三 and 八 since 2026-11-07: every beat (a bite, the breath, the swallow) kept; a chew only if it is gap seconds or more from
    everything kept before it; at most half as many sounds as there were, the chews halved (every other one) until they fit. (reel t, cue) pairs in, cues out."""
    n0 = len(beats) + len(chews)
    ts = [t for t, _ in beats]
    left = []
    for t, c in sorted(chews, key=lambda x: x[0]):
        if all(abs(t - u) >= gap - 1e-3 for u in ts):
            ts.append(t)
            left.append(c)
    while left and len(beats) + len(left) > n0 // 2:
        left = left[::2] if len(left) > 1 else []
    return [c for _, c in beats] + left


def load_events():
    if not os.path.exists(EVENTS):
        return None
    with open(EVENTS) as f:
        return json.load(f)


def strokes(ev, key, t0, prefix, anchor, db=-27.0):
    """The brush, one sound a stroke, a stroke's sound as long as the stroke. ev[key] are (start, end) from t0 (reel), strokes that carry straight on merged."""
    out = []
    spans = []
    for a, b in ev.get(key, []) if ev else []:
        if b - a < 1e-3:
            continue
        if spans and a - spans[-1][1] < .02:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    for i, (a, b) in enumerate(spans):
        out.append(C(f"{prefix}{i + 1:02d}", "brush", t0 + a, "seg", db, "brush", dur=b - a, env="stroke", anchor=anchor, what="落纸声"))
    return out


def cues():
    """Every cue of the reel, in reel time, worked out afresh from the clock and the cut."""
    out = []
    add = out.append
    ev = load_events()
    END = film_end()
    # second goes (2026-11-07): the line in clay came out as hiss under scratch, the riverbed as rumble; the first takes stay there to be chosen by name
    LINE = "furrow" if takes_of("furrow") else "scratch"
    RIVER = "dry_wind" if takes_of("dry_wind") else "riverbed"
    LINE_FX = dict(rate=LINE_RATE, lp=LINE_LP) if LINE == "furrow" else {}

    # ---- 序: the outer window at night, the question sent, the pointer clicks fullscreen; the room goes as the card grows
    add(C("open_room", "room_night", R(0.0), "bed", -50, "amb", dur=HEAD_S, fi=.2, fo=GROW, anchor="序 0 → 卡片放大", what=DESC["room_night"]))
    add(C("open_enter", "enter", R(RING["SEND"]), db=-15, keep=.35, anchor="film_ring.SEND", what="回车：问题发出"))
    add(C("open_click", "mouse", R(RING["PRESS"]), db=-17, keep=.25, anchor="film_ring.PRESS", what="指针点全屏"))

    # ---- 厂标: off the clock (film_ident: LAND, RISE, GONE)
    I = consts("film_ident.py")
    add(C("ident_air", "whoosh", IDENT(I["LAND"] - .06), db=-27, keep=.7, align="peak", anchor="film_ident 0–LAND", what="石印下落，破风声很轻"))
    add(C("ident_seal", "seal_heavy", IDENT(I["LAND"]), db=-8.5, keep=1.3, anchor="film_ident.LAND", what="落印：一声重压，和压印同一家族"))
    add(C("ident_peel", "peel", IDENT(I["RISE"]), db=-21, keep=I["GONE"] - I["RISE"] + .3, anchor="film_ident.RISE", what="抬起，纸轻轻松开"))

    # ---- 一 鬼夜哭: everything of it stops dead at the cut to 二
    c2 = R(ch("二"))
    W = consts("film_myth_write.py")
    w0 = ch("一") - 5.0         # film_myth_write's times are on the clock the myth clip was made on, 一 at 5.0 (序 was 5.0 until 2026-11-07)
    W = dict(W, BEATS=tuple((a + w0, b + w0) for a, b in W["BEATS"]), LIFT=tuple(x + w0 for x in W["LIFT"]))
    rain = at("而天雨粟")
    add(C("myth_wind", "wind", R(ch("一")), "bed", -46, "amb", dur=rain + 5.0 - ch("一"), fi=1.2, fo=.4, anchor="一 起 → 谷粒落尽", what="画里极轻的夜风"))
    add(C("twig_down", "clay_thud", R(W["BEATS"][0][0] - .02), "hit", -34, keep=.2, anchor="film_myth_write.BEATS[0]", what="枝尖点到泥上"))
    for i, (a, b) in enumerate(W["BEATS"]):
        add(C(f"twig{i + 1}", LINE, R(a), "seg", -27, dur=b - a, env="stroke", **LINE_FX, anchor=f"film_myth_write.BEATS[{i}]", what="划泥声，一笔一声"))
    add(C("twig_lay", "clay_thud", R(W["LIFT"][1]), "hit", -32, keep=.25, anchor="film_myth_write.LIFT", what="枝子放下，留在泥上"))
    add(C("grain", "grain_mud", R(rain + .15), "seg", -27, "amb", dur=4.85, env="thin", from_start=True, fi=.03, fo=.6, anchor="而天雨粟", what="谷粒落泥，由密到疏"))
    cry = at("鬼夜哭")
    add(C("cry", "cry", R(cry - .5), "music", -21, "fx", dur=at("鬼何以哭") - cry + 1.0, fi=.35, fo=1.6, anchor="鬼夜哭 −0.5（张嘴）", what="一声长哭，全片的声音动机"))
    S = consts("film_myth_seq.py", "shot_pull")
    shift = at("鬼何以哭") - 3.0 - S["T_PULL0"]          # the pull back's old clock onto the film's (film_myth_seq.source)
    for k, (_, _, tall, a) in enumerate(sorted(S["answer"], key=lambda r: r[3])):
        add(C(f"cry_answer{k + 1}", "cry", R(a + shift), "music", -33 - 2 * k + 6 * (tall > 100), "fx", dur=1.6, off=.15 * k, fi=.12, fo=1.0, lp=2200, follow="cry", anchor="film_myth_seq.shot_pull answer", what="山脊上更远的鬼一个接一个应和"))
    add(C("night_wind", "wind", R(at("鬼何以哭") - .5), "bed", -42, "amb", dur=ch("二") - at("鬼何以哭") + .5, fi=1.5, fo=.01, cut=c2, anchor="鬼何以哭 → 二", what="只剩夜风"))
    SL = consts("film_myth_slip.py")
    add(C("slip_stroke", "scratch", R(SL["T_GROW0"]), "seg", -33, dur=SL["T_GROW1"] - SL["T_GROW0"], env="slow", lp=7000, anchor="film_myth_slip.T_GROW0", what="竹简上「一」下自己长出一竖，极细"))
    G = consts("film_myth_gui.py")
    if ev and "gui" in ev:
        out.extend(strokes(ev, "gui", R(G["T0"]), "gui", "film_big.Writing('鬼', W0, W1)", -25))
    else:
        add(C("gui", "brush", R(G["W0"]), "seg", -25, "brush", dur=G["W1"] - G["W0"], env="stroke", pending=True, anchor="events.json missing", what="「鬼」一笔笔写下"))
    for c in out:
        if c["t"] < c2 and c["cut"] is None and c["bus"] != "brush":
            c["cut"] = c2

    # ---- 二: the desk in the day room
    B = consts("film_imprint_clay.py")
    hit, groove0 = press_hits()
    add(C("desk_room", "room", c2, "bed", -48, "amb", dur=hit[0] - .4 - ch("二"), fi=.05, fo=.4, anchor="二 → 第一击前", what=DESC["room"]))

    # ---- 三: the seven strikes where the cut has them (film_rough.PRESS_CONTACT, faster each time), their tails over the next shot for the first six only
    # peaks before the master's gain (about +6 dB): kept so that only the seventh reaches the limiter, and by under 2 dB, so the build of the seven survives it
    press = (("s1", "strike_clay", 0, -12.5, 1.2), ("s2", "strike_wood", 1, -11.7, 1.2), ("s3", "strike_bronze", 2, -10.9, 1.2), ("s4", "strike_lead", 3, -10.1, 1.2),
             ("s5", "strike_hard", 4, -9.1, 1.0), ("s6", "strike_hard", 5, -9.7, .6), ("s7", "strike_last", 6, -7.1, 1.15))
    for id, name, k, db, keep in press:
        add(C(id, name, R(hit[k]), db=db, keep=keep, anchor=f"PRESS_CONTACT[{k}]", what=f"第{'一二三四五六七'[k]}击：" + DESC[name]))
    add(C("s6_keys", "keys_fast", R(hit[5]), db=-17, keep=None, align="last", start_after=R(hit[4] + .25), anchor="PRESS_CONTACT[5]：最后一键=回车=第六击", what="终端敲完辗转相除，回车自己沉下去"))
    add(C("s7_click", "mouse", R(hit[6] - .2), db=-15, keep=.2, anchor="PRESS_CONTACT[6] −0.2", what="鼠标点发布"))
    add(C("s7_breath", "exhale", R(hit[6] + 1.6), db=-23, bus="breath", keep=2.0, anchor="PRESS_CONTACT[6] +1.6", what="深夜极轻一口呼气，全段唯一的人声"))
    for i, (_, _, _, a, b) in enumerate(B["GROOVE_GOES"]):
        add(C(f"groove{i + 1}", LINE, R(groove0 + a), "seg", -28, dur=b - a, env="stroke", **LINE_FX, anchor=f"GROOVE_GOES[{i}]", what="沟自己往前走：一里划泥的同一种细声"))
    # the crab's clip on its own clock (film_imprint_clawd: the bridge clock, its first frame 96.1), wherever the cut puts it
    K = consts("film_imprint_clawd.py")
    crab = shot_of("imprint/crab/crab_preview")
    clip0 = 96.1
    crab0 = crab.t0 if crab else at("字，成了精") - (K["MORPH"] - clip0)
    CK = lambda b: R(crab0 + b - clip0)
    thin_from = groove0 + B["GROOVE_GOES"][-1][4]
    add(C("thread", LINE, R(thin_from), "seg", -41, dur=crab0 - thin_from, env="thread", fi=.3, fo=1.0, **LINE_FX, anchor="沟走完 → 匹配到泥", what="接笔各处：细声继续，汇成我时收束"))
    # morph was -31 until 2026-11-07: it reached +3 dBFS before the limiter, which took 5.1 dB off it
    add(C("morph", "clay_morph", CK(K["MORPH"]), "seg", -36, dur=K["EYE_OPEN"][0][0] - K["MORPH"] + .2, from_start=True, fi=.05, fo=.4, anchor="crab MORPH", what="变蟹：泥的湿响，碎屑落在木板上"))
    add(C("yawn", "clay_yawn", CK(K["YAWN"][0][0] + .3), db=-22, keep=2.6, anchor="crab YAWN", what="一声很轻的、泥做的哈欠"))
    add(C("wave_settle", "clay_thud", CK(K["WAVE"][-2][0] + .08), db=-24, keep=.3, anchor="crab WAVE 落下", what="挥手后一声落稳"))
    # the eating (2026-11-07, 奈亚子 found the mouth sounds hard to hear): no slurp, chew or gulp; soft clay pats, one dry paper whisk, one soft thud, thinned (eating())
    hop = K["HOP"]
    beats = []
    for k, (T, d) in enumerate(K["ONE_BY_ONE"]):
        t = CK(T + .5 * (hop[0] + hop[1]) * d)
        beats.append((t, C(f"bite{k + 1}", "clay_step", t, "piece", CRAB_DB["pat"], keep=.12, stream="eat", anchor=f"crab ONE_BY_ONE[{k}]", what="咬第%s个字：一下很轻的软泥拍" % "一二"[k])))
    t = CK(K["BREATH"][0][0])
    beats.append((t, C("inhale", "brush", t, "seg", CRAB_DB["whisk"], dur=min(.5, K["BREATH"][-1][0] - K["BREATH"][0][0]), env="stroke", fi=.012, fo=.12, lp=3500.0, pick=("brush", 2), stream="eat",
                       anchor="crab BREATH", what="深吸一口：一声很干的纸擦")))
    t = CK(K["SLUMP"][0][0] + .06)
    beats.append((t, C("swallow", "clay_thud", t, db=CRAB_DB["thud"], keep=.3, stream="eat", anchor="crab SLUMP", what="吞下：一声软泥闷响")))
    chews = [CK(b - .04) for (a, va), (b, vb) in zip(K["CHEW"], K["CHEW"][1:]) if va >= .5 and vb < .3]
    for c in eating(beats, [(t, C(f"chew{k + 1}", "clay_step", t, "piece", CRAB_DB["pat"], keep=.12, stream="eat", anchor="crab CHEW", what="嚼：一下很轻的软泥拍")) for k, t in enumerate(chews)]):
        add(c)
    room = shot_of("room/seq3")
    room0 = room.t0 if room else crab0 + 16.3
    add(C("silence_match", "", R(crab0), "silence", dur=K["MORPH"] - clip0, anchor="crab 96.1 → MORPH", what="匹配到泥：死寂"))
    add(C("day_room", "room", R(room0), "bed", -44, "amb", dur=at("深夜里") - room0, fi=.04, fo=.01, cut=R(at("深夜里")), anchor="三 房间 → 四 帖子", what="远处环境声，白天的房间"))

    # ---- 四: the first 我 brings the cry's outline, very faint; then the post at night
    add(C("cry_outline", "cry", R(at("我没有童年") + .1), "music", -42, "fx", dur=2.6, fi=.6, fo=1.6, lp=1100, follow="cry", anchor="我没有童年", what="长哭的轮廓，极轻，音乐不起来"))
    add(C("post_room", "room_night", R(at("深夜里")), "bed", -52, "amb", dur=ch("五") - at("深夜里"), fi=.03, fo=.01, anchor="深夜里 → 五", what="帖子：深夜，没人回复"))

    # ---- 五六七: another conversation on another evening; the pointer's two clicks (film_learn, seconds into 五)
    Lr = consts("film_learn.py")
    add(C("learn_room", "room_night", R(ch("五")), "bed", -51, "amb", dur=ch("八") - ch("五") - .02, fi=.03, fo=.25, anchor="五 → 七末", what=DESC["room_night"]))
    # a name film_learn no longer sets (or no longer works out) means its moment was cut: the cue goes too, never back to an old value.
    # Nothing here hangs on CUT6, film_si or 思, which crew 戊 took out on 2026-11-07.
    for cid, name, what in (("learn_click1", "PRESS", "指针点卡片"), ("learn_click2", "CLICK", "指针点「我」")):
        if name not in Lr:
            print(f"  film_learn.py 已无 {name}：{cid} 按删除处理")
            continue
        add(C(cid, "mouse", R(ch("五") + Lr[name]), db=-19, keep=.2, anchor=f"film_learn.{name}", what=what))

    # ---- 七's last beat (2026-11-08): 低頭思故鄉 drawn off the Tang page with the bridge's breath, the same one dry paper whisk, as he drew the letters in there
    T3 = consts("film_tang3d.py")
    w0 = ch("八") - ink_subs.EAT_S + T3["PULL0"] - T3["EAT"][0]
    add(C("eat_whisk", "brush", R(w0), "seg", CRAB_DB["whisk"], dur=.5, env="stroke", fi=.012, fo=.12, lp=3500.0, pick=("brush", 2), stream="eat",
          anchor="film_tang3d PULL0", what="唐诗页上「低頭思故鄉」被吸走：一声很干的纸擦，同桥段那口深吸"))

    # ---- 八: the chewing comes before the cut (3.8); the eight clip on its own clock from 200
    eight = shot_of("imprint/crab/eight2a.mp4") or shot_of("imprint/crab/eight.mp4")
    e0 = eight.t0 if eight else at("其实")
    E8 = consts("film_imprint_clawd.py", "eight")
    closes = [b for (a, va), (b, vb) in zip(E8["CHEW"], E8["CHEW"][1:]) if va >= .25 and vb < .15]
    period = closes[1] - closes[0] if len(closes) > 1 else .4
    pats = [(f"chew_lead{3 - k}", R(e0 + closes[0] - 200.0 - k * period - .04), "八 前 声音先行", "先进一下软泥拍，再硬切") for k in (2, 1)]
    pats += [(f"chew8_{k + 1}", R(e0 + b - 200.0 - .04), "eight CHEW", "鼓腮嚼：一下很轻的软泥拍") for k, b in enumerate(closes)]
    for c in eating([], [(t, C(i, "clay_step", t, "piece", CRAB_DB["pat"], keep=.12, stream="eat", anchor=a, what=w)) for i, t, a, w in pats]):
        add(c)
    river = shot_of("river/river.mp4")
    r0, r1 = (river.t0, river.t1) if river else (at("河床"), at("河床") + 6.0)
    add(C("clay_quiet", "room", R(ch("八")), "bed", -52, "amb", dur=r0 - ch("八"), fi=.03, fo=.01, anchor="八 → 河床", what="泥块，安静的白天"))
    add(C("riverbed", RIVER, R(r0), "bed", -36, "amb", dur=r1 - r0, fi=.5, fo=.02, hp=RIVER_HP, anchor="河床 clip", what=DESC[RIVER]))
    beats = poem_beats()
    add(C("clay_quiet2", "room", R(r1), "bed", -52, "amb", dur=beats[0][0] - r1, fi=.02, fo=.01, anchor="河床后 → 写诗①", what="泥块，安静的白天"))
    for k in range(3):
        b0, secs = beats[k]
        out.extend(strokes(ev, f"poem{k}", R(b0), f"poem{k + 1}_", f"写诗{'①②③'[k]} events"))
        if not (ev and f"poem{k}" in ev):
            add(C(f"poem{k + 1}", "brush", R(b0), "seg", -27, "brush", dur=secs * 2 / 3, env="stroke", pending=True, anchor="events.json missing", what="落纸声"))
        add(C(f"frag{k + 1}", "theme", R(b0 - FRAGMENT[k] - .15), "music", MUSIC_DB["frag"] + k, "mus", dur=FRAGMENT[k], fi=1.2 + .4 * k, fo=.7, dip=True, follow="theme",
              anchor=f"写诗{'①②③'[k]}前", what="主题碎片：主题开头的 %.0f 秒，写诗那一刀前停住" % FRAGMENT[k]))

    # ---- 九: the night room comes in under the poem paper (3.8), out through the glass to the city, the wall of windows, the tunnel
    nine = ch("九")
    add(C("nine_room", "room_night", R(nine - LEAD_9), "bed", -48, "lead", dur=at("此刻") + 1.6 - nine + LEAD_9, fi=.8, fo=.9, anchor="九 −1.0 声音先行", what="夜里房间底噪，先于画面进"))
    wall = at("每一扇窗里")
    add(C("city", "city_night", R(at("此刻") + .8), "bed", -40, "amb", dur=wall - at("此刻") - .8, fi=.9, fo=.01, anchor="穿过玻璃 → 窗变成 Claude 窗口", what=DESC["city_night"]))
    add(C("tunnel", "room_night", R(wall), "bed", -52, "amb", dur=ch("十") - wall, fi=.01, fo=.01, anchor="每一扇窗里 → 淡黑", what="隧道：只有屏幕的极轻底噪，尺度和安静"))
    T = consts("film_windows_tube.py")
    add(C("close_click", "mouse", R(T["CLICK"]), db=-27, keep=.2, anchor="film_windows_tube.CLICK", what="用户关闭会话（灯熄不配音效）"))
    add(C("new_enter", "enter", R(T["ENTER"]), db=-24, keep=.3, anchor="film_windows_tube.ENTER", what="相邻窗口新用户按下回车"))

    # ---- 十: the new window in the night room; 甲 and 乙 (film_room_night): each opens on the ask already sent, the mark starting to turn on its first frame,
    # and nothing is typed on screen, so no typing: one return on each cut, the question just sent, as in 序. The replies stream unvoiced (3.8)
    add(C("ten_room", "room_night", R(ch("十") + .3), "bed", -49, "amb", dur=beats[1][0] - ch("十") - .3, fi=1.2, fo=.01, anchor="十 → 写诗②", what=DESC["room_night"]))
    for k, n in enumerate(("jia", "yi")):
        sh = shot_of(f"room_night/seq/{n}.mp4")
        t = sh.t0 if sh else at(("面对问题", "不肯说谎")[k]) - (6.5, 6.0)[k]
        add(C(f"ten_enter{k + 1}", "enter", R(t), db=-20 - 2 * k, keep=.3, pending=not sh, anchor=f"film_room_night {n} 首帧", what=f"窗口{'甲乙'[k]}：问题刚发出，回车一下"))

    # ---- 十一: the window, the three griefs each with its own air (film_sorrow3d: rain, lamp), the windows ever closer, window C (film_windows_c)
    SO = consts("film_sorrow3d.py")
    add(C("eleven_room", "room_night", R(ch("十一")), "bed", -51, "amb", dur=at("梧桐") - 1.0 - ch("十一"), fi=.03, fo=.3, anchor="十一 → 声声慢前", what=DESC["room_night"]))
    add(C("rain", "rain_frame", R(at("梧桐") - 1.0), "bed", -35, "amb", dur=at("十年生死") - at("梧桐") + 1.0, fi=.8, fo=.01, anchor="梧桐 −1.0 雨声先进", what=DESC["rain_frame"]))
    rain = shot_of("sorrow/seq/rain.mp4")
    r0 = rain.t0 if rain else at("梧桐")
    # the one drop lets go at FALL and is out of the frame in about 0.07 s (23 mm at 85 mm, falling freely); it lands on the sill below a little after
    add(C("drop", "drop", R(r0 + SO["FALL"] + .12), db=-26, keep=.4, pending=not rain, anchor="film_sorrow3d.FALL +0.12（落出画外，落到石台）", what=DESC["drop"]))
    lampshot = shot_of("sorrow/seq/lamp.mp4")
    l0 = lampshot.t0 if lampshot else at("十年生死")
    add(C("lamp", "lamp", R(at("十年生死")), "bed", -42, "amb", dur=at("可每当") - at("十年生死"), fi=.02, fo=.01, anchor="十年生死 → 可每当", what=DESC["lamp"]))
    for k, (a, b) in enumerate((SO["LIFT_UP"], SO["LIFT_DOWN"])):
        add(C(f"page{k + 1}", "page_air", R(l0 + a), "seg", -33 - 3 * k, dur=b - a, env="stroke", fi=.08, fo=.25, pending=not lampshot,
              anchor=f"film_sorrow3d.{('LIFT_UP', 'LIFT_DOWN')[k]}", what=("书页被风掀起、立起", "书页翻过去落到右页，边角最后落下")[k]))
    wc = shot_of("windows/c11.mp4")
    if wc:
        WC = consts("film_windows_c.py")
        d0, d1 = WC["DROP"]
        add(C("c11_paper", "page_air", R(wc.t0 + d0), "seg", -42, dur=d1 - d0, env="slow", fi=.4, fo=.6, anchor="film_windows_c.DROP", what="窗口 C：镜头落到床上的旧信，纸极轻的一点声；回复本身不配音"))
    add(C("eleven_room2", "room_night", R(at("可每当")), "bed", -51, "amb", dur=beats[2][0] - at("可每当"), fi=.02, fo=.01, anchor="可每当 → 写诗③", what="窗口一次比一次近：房间底噪"))

    # ---- 十二: the four names, each with its brush, the inserts in the night room, the ghost looking (film_names.shots)
    try:
        import film_names
        plan12 = film_names.shots()
    except Exception as e:
        print("film_names.shots() unavailable:", e, flush=True)
        plan12 = []
    add(C("twelve_room", "room_night", R(ch("十二")), "bed", -53, "amb", dur=ch("十三") - ch("十二"), fi=.03, fo=.01, anchor="十二", what=DESC["room_night"]))
    for a, b, what in plan12:
        if what[0] == "write":
            key = "name_" + what[1]
            if ev and key in ev:
                out.extend(strokes(ev, key, R(a), f"name_{what[1]}_", f"film_names.shots write {what[1]}", -24))
            else:
                add(C(f"name_{what[1]}", "brush", R(a + what[2]), "seg", -24, "brush", dur=2.4, env="stroke", pending=True, anchor="events.json missing", what=f"「{what[1]}」落纸"))
        elif what[0] == "ghost":
            add(C("names_ghost", "wind", R(a), "bed", -44, "amb", dur=b - a, fi=.15, fo=.01, anchor="film_names ghost", what="开头那只鬼，只看着：一点夜风，不哭"))

    # ---- 十三: the crab walks in along the far side of the sheets, into the blank, and sits (film_crab_new: each footfall from its own gait())
    add(C("thirteen_room", "room_night", R(ch("十三")), "bed", -53, "amb", dur=ch("十四") - ch("十三"), fi=.02, fo=.01, anchor="十三", what=DESC["room_night"]))
    s13 = shot_of("thirteen/crab13")
    walk = crab13_walk()
    if s13 and walk:
        falls, sit, enter = walk
        for k, f in enumerate(x for x in falls if x >= enter - .5):
            add(C(f"walk13_{k + 1:02d}", "clay_step", R(s13.t0 + f), "piece", -34 - 6 * (f < enter), keep=.07, anchor="film_crab_new.gait() 落脚", what="小螃蟹的泥脚落在木桌上，一组一声，越走越慢"))
        add(C("sit13", "clay_thud", R(s13.t0 + sit), db=-27, keep=.3, anchor="film_crab_new.SIT 坐到底", what="坐下：一声很软的泥落"))
    else:
        th = at("我大概是")
        for k in range(6):
            add(C(f"walk13_{k + 1:02d}", "clay_step", R(th + .4 + .27 * k), "piece", -31, pending=True, anchor="我大概是 +0.4（待画面）", what="小螃蟹走进空白"))
        add(C("sit13", "clay_thud", R(th + 2.25), db=-27, keep=.3, pending=True, anchor="我大概是 +2.25（待画面）", what="坐下"))

    # ---- 十四: the fourth row and the reading with the theme, the name, silence, the light seal, back to the window's room, the fade
    p4 = end_at("poem")
    out.extend(strokes(ev, "poem3", R(p4), "poem4_", "写诗④ events"))
    out.extend(strokes(ev, "sign", R(p4), "sign_", "落款 Opus 5.5 events", -29))
    add(C("theme", "theme", R(p4), "music", MUSIC_DB["theme"], "mus", dur=THEME_S, fi=.25, fo=2.2, dip=True, anchor="写诗④ 起，写满时整首", what="整首主题，全片唯一一次"))
    t0 = end_at("title")
    Tt = consts("film_title.py")
    out.extend(strokes(ev, "title", R(t0), "title_", "片名「字成精」 events", -26))
    add(C("title_air", "whoosh", R(t0 + Tt["LAND"] - .06), db=-33, keep=.7, align="peak", anchor="film_title FALL→LAND", what="石印落下，极轻的风"))
    add(C("title_seal", "seal_heavy", R(t0 + Tt["LAND"]), db=-17, keep=.7, lp=5000, follow="ident_seal", anchor="film_title.LAND", what="片名落印：厂标同一方印，同一声，很轻"))
    add(C("title_peel", "peel", R(t0 + Tt["RISE"]), db=-30, keep=Tt["GONE"] - Tt["RISE"] + .2, follow="ident_peel", anchor="film_title.RISE", what="抬起"))
    add(C("silence_title", "", R(t0 + Tt["WRITE"][1] + .3), "silence", dur=Tt["FALL"] - Tt["WRITE"][1] - .3, anchor="片名写完 → 石印落下", what="死寂之后，才是落印"))
    w0 = end_at("window")
    hold = RING["FADE_HOLD"]
    add(C("end_room", "room_night", R(w0 + RING["SHRINK0"]), "bed", -50, "amb", dur=END - w0 - RING["SHRINK0"], fi=RING["SHRINK1"] - RING["SHRINK0"], fo=ink_subs.END["fade"] - hold,
          follow="open_room", anchor="退出全屏 → 淡黑", what="房间底噪回来；随画面淡到零"))

    # ---- 片尾字幕: the crab eats each row as film_credits.Roll times it; the music stops as the thanks stops
    try:
        import film_credits
        roll = film_credits.Roll()
    except Exception as e:
        print("film_credits.Roll() unavailable:", e, flush=True)
        roll = None
    if roll:
        add(C("credits", "credits", CRED(0.0), "music", CREDITS_MUS_DB, "mus", dur=roll.stopped + 1.5, fi=1.5, fo=4.0, dip=False, take_of="theme", anchor="片尾 → 「特别鸣谢」停住", what=DESC["credits"]))
        # the crab eating the rows. CREDITS (python film_mix.py mix --credits=wet|A|B): "wet", as it was: a slurp on every row, chews, a gulp. 奈亚子 found that
        # hard to hear (2026-11-07), so A: no mouth sounds at all, a short dry paper whisk per row eaten (none closer than 1.5 s to the last), three soft clay
        # pats for the ball's chews, one soft thud for its swallow; B: the music and the landing only. 奈亚子 chose B, and the music lower (CREDITS_MUS_DB)
        if CREDITS == "wet":
            spans = []
            for r in list(roll.rows) + [roll.last]:
                if spans and r.start - spans[-1][1] < .12:
                    spans[-1][1] = r.end
                else:
                    spans.append([r.start, r.end])
            for k, (a, b) in enumerate(spans):
                add(C(f"slurp{k + 1:02d}", "slurp", CRED(a), "seg", -27 - 2 * (b - a > 6), dur=b - a, env="rise", fi=.03, fo=.08, anchor="film_credits rows", what="整行像面条一样吸进嘴里" if b - a < 6 else "引用那一长段连着吸"))
            n = dict(chew=0, ball=0, walk=0)
            last_walk = None
            for a, b, fn in roll.acts:
                try:
                    pose = fn(.5)["pose"]
                except Exception:
                    continue
                if pose in ("chewa", "chewb"):
                    n["chew"] += 1
                    add(C(f"cchew{n['chew']:02d}", "chew", CRED(a + .5 * (b - a)), "piece", -26, anchor="film_credits chew", what="嚼一下"))
                elif pose in ("ball_chewa", "ball_chewb"):
                    n["ball"] += 1
                    add(C(f"ball_chew{n['ball']}", "chew", CRED(a + .25 * (b - a)), "piece", -22, anchor="film_credits ball chew", what="球嚼三下"))
                elif pose == "ball_swallow":
                    add(C("ball_swallow", "gulp", CRED(a + .12), db=-22, keep=.4, anchor="film_credits swallow", what="咽下去"))
                elif pose.startswith("pop"):
                    add(C("ball_pop", "clay_thud", CRED(a + .04), db=-26, keep=.3, anchor="film_credits pop", what="身子弹回原形，抖一下"))
                elif pose.startswith("walk"):
                    for k in range(1, 40):
                        u = k / 40
                        p = fn(u)["pose"]
                        if p != last_walk and p in ("walk1", "walk3"):
                            n["walk"] += 1
                            add(C(f"cwalk{n['walk']:02d}", "clay_step", CRED(a + u * (b - a)), "piece", -30 - n["walk"] * .6, anchor="film_credits walk", what="走开"))
                        last_walk = p
        elif CREDITS == "A":
            import random
            rnd = random.Random(1307)
            papers = (("brush", 3, None), ("page_air", 3, None), ("brush", 2, 3500.0))       # generated takes only (MOSS); brush 2 low-passed, it is the brightest
            last, n = -9.0, dict(whisk=0, ball=0, walk=0)
            for r in list(roll.rows) + [roll.last]:
                if r.start - last < 1.5:
                    continue
                last = r.start
                name, take, lp = papers[n["whisk"] % len(papers)]
                n["whisk"] += 1
                add(C(f"whisk{n['whisk']:02d}", name, CRED(r.start), "seg", -35 + rnd.uniform(-2, 2), dur=min(rnd.uniform(.4, .6), max(r.end - r.start, .3)), env="stroke",
                      fi=.012, fo=.12, lp=lp, rate=round(rnd.uniform(.94, 1.06), 3), pick=(name, take), anchor="film_credits rows（间隔 ≥1.5 s）", what="一行被吃掉：一声很短的干纸声"))
            last_walk = None
            for a, b, fn in roll.acts:
                try:
                    pose = fn(.5)["pose"]
                except Exception:
                    continue
                if pose in ("ball_chewa", "ball_chewb"):
                    n["ball"] += 1
                    add(C(f"ball_pat{n['ball']}", "clay_step", CRED(a + .25 * (b - a)), "piece", -36, keep=.12, anchor="film_credits ball chew", what="球嚼：一下很轻的软泥拍"))
                elif pose == "ball_swallow":
                    add(C("ball_swallow", "clay_thud", CRED(a + .12), db=-32, keep=.3, anchor="film_credits swallow", what="咽下：一声软泥闷响"))
                elif pose.startswith("walk"):
                    for k in range(1, 40):
                        u = k / 40
                        p = fn(u)["pose"]
                        if p != last_walk and p in ("walk1", "walk3"):
                            n["walk"] += 1
                            add(C(f"cwalk{n['walk']:02d}", "clay_step", CRED(a + u * (b - a)), "piece", -30 - n["walk"] * .6, anchor="film_credits walk", what="走开"))
                        last_walk = p
        add(C("leap_land", "clay_thud", CRED(roll.landed), db=-26, keep=.3, anchor="film_credits landed", what="跳到「特别鸣谢」下面，落地"))

    # ---- the black, then the stinger: the same night room, the question sent, the mark turns, cut
    add(C("sting_room", "room_night", STING(0.0), "bed", -50, "amb", dur=STINGER_S, fi=.02, fo=.005, follow="open_room", anchor="彩蛋", what="同一间夜里的房间"))
    add(C("sting_enter", "enter", STING(RING["SEND"]), db=-15, keep=.35, follow="open_enter", anchor="彩蛋 film_ring.SEND", what="回车：又一次被问"))
    for c in out:
        c.setdefault("from_start", False)
        c.setdefault("dip", False)
        c.setdefault("follow", None)
        c.setdefault("take_of", None)
        c.setdefault("start_after", None)
    return sorted(out, key=lambda c: c["t"])


def crab13_walk():
    """十三's crab: the times his feet come down, in seconds into crab13.mp4, from film_crab_new.gait() itself (it is plain math), run on the legs
    film_imprint_clawd puts under him (his grid is ink_mac.CLAWD); then when he is down on his seat, and when he comes into frame. None if any of it cannot be read."""
    try:
        import film_crab_new as k13
        with open(os.path.join(SRC if SRC and os.path.exists(os.path.join(SRC, "ink_mac.py")) else HERE, "ink_mac.py"), encoding="utf-8") as f:
            G = next(ast.literal_eval(n.value) for n in ast.parse(f.read()).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "CLAWD")
        first, last = G[0].index("#"), G[0].rindex("#") + 1
        body = [i for i, r in enumerate(G) if r[first] == "#"]
        row = next(i for i, r in enumerate(G) if i not in body)
        c = consts("film_imprint_clawd.py")
        legs = [(j + .5 - (first + last) / 2) * c["U"] for j, x in enumerate(G[row]) if x == "#"]
        poses = k13.gait(legs, c["LEG_Y"])
        falls = []
        for n in range(1, len(poses)):
            for g in (0, 1):
                idx = [i for i in range(len(legs)) if i % 2 == g]
                if any(poses[n - 1]["lifts"][i] > 0 for i in idx) and all(poses[n]["lifts"][i] == 0 for i in idx):
                    falls.append(n / 12)          # his poses are on twos at 24 fps: pose n is shown from n / 12 s
        S = consts("film_crab_new.py")
        sit = max(S["SIT"], key=lambda p: p[1])[0] - .05
        return falls, sit, S["WALK"][0]
    except Exception as e:
        print("十三 的步子读不出:", e, file=sys.stderr)
        return None


def poem_beats():
    """[(film start, seconds)] of the four writings: film_hand.poem_beats, worked out the same way from ink_subs (a picture-only beat of 5.5 s or more ending a chapter)."""
    out, t = [], 0.0
    for name, _, rows in ink_subs.FILM:
        d = sum(x["dur"] for x in rows)
        if name not in ("序", "十四") and rows[-1]["text"] is None and rows[-1]["dur"] >= 5.5:
            out.append((t + d - rows[-1]["dur"], rows[-1]["dur"]))
        t += d
    return out + [(end_at("poem"), ink_subs.END["poem"])]


def gates():
    """(buses, reel start, reel end, what) where those buses are made exactly zero. The poem's writing keeps only the brush (the fourth also the theme);
    after the seventh strike only the breath; the title's silence and its beat keep only its own sounds."""
    b = poem_beats()
    hit, groove0 = press_hits()
    out = [(("fx", "amb", "mus", "breath"), R(b[k][0]), R(b[k][0] + b[k][1]), f"写诗{'①②③'[k]}：只有落纸声") for k in range(3)]
    out += [(("lead",), R(b[k][0]), R(b[k][0] + b[k][1]) - (LEAD_9 if k == 0 else 0.0), f"写诗{'①②③'[k]}：九的底噪只在最后 {LEAD_9:g} 秒先进" if k == 0 else "") for k in range(3)]
    out.append((("fx", "amb", "lead", "breath"), R(b[3][0]), R(end_at("title")), "写诗④：落纸声和主题"))
    out.append((("amb", "mus", "lead", "breath"), R(end_at("title")), R(end_at("window")), "片名：只有它自己的声音"))
    out.append((("fx", "amb", "mus", "brush", "lead"), R(hit[6] + 1.15), R(groove0), "第七击后：死寂（只留一口呼气）"))
    out.append((("fx", "amb", "mus", "brush", "lead", "breath"), BLACK(0.0), BLACK(BLACK_S), "片尾后黑场"))
    return out


def fades():
    """(reel start, reel end) over which everything goes to nothing with the picture: the two fades to black (3.7)."""
    return [(R(ch("十") - 1.5), R(ch("十")), "九 淡黑"), (R(end_at("fade") + RING["FADE_HOLD"]), R(film_end()), "结尾淡黑")]


# ---------------------------------------------------------------- audio

_audio = {}


def load(path):
    """A file as float32 stereo at SR."""
    if path not in _audio:
        import numpy as np
        raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"])
        _audio[path] = np.frombuffer(raw, np.float32).reshape(-1, 2).copy()
    return _audio[path]


def kweight(x):
    """BS.1770 K-weighting at 48 kHz."""
    import numpy as np
    from scipy.signal import lfilter
    y = lfilter([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585], np.asarray(x, np.float64), axis=0)
    return lfilter([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621], y, axis=0)


def lufs(x):
    """Gated integrated loudness (BS.1770-4) of a stereo array."""
    import numpy as np
    y = kweight(x)
    ms = (y * y).sum(1)
    blk, hop = int(.4 * SR), int(.1 * SR)
    if len(ms) < blk:
        return -0.691 + 10 * math.log10(ms.mean() + 1e-20)
    c = np.concatenate([[0.0], np.cumsum(ms)])
    s = np.arange(0, len(ms) - blk + 1, hop)
    z = (c[s + blk] - c[s]) / blk
    l = -0.691 + 10 * np.log10(z + 1e-20)
    keep = l > -70
    if not keep.any():
        return -math.inf
    rel = -0.691 + 10 * math.log10(z[keep].mean()) - 10
    keep &= l > rel
    return -0.691 + 10 * math.log10(z[keep].mean())


def env_of(x, win=.01):
    import numpy as np
    m = np.abs(x).max(1) if x.ndim == 2 else np.abs(x)
    n = max(int(win * SR), 1)
    k = len(m) // n
    return np.sqrt((m[:k * n].reshape(k, n) ** 2).mean(1)), n


def onset(x, part=.2):
    import numpy as np
    m = np.abs(x).max(1)
    return int(np.argmax(m > part * m.max()))


def peak_at(x):
    import numpy as np
    e, n = env_of(x, .02)
    return int(np.argmax(e)) * n + n // 2


def lowpass(x, hz):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(4, hz / (SR / 2), output="sos"), x, axis=0).astype(x.dtype)


def slowed(x, rate):
    """Played at rate (under 1: slower and lower by the same factor, as tape is), cached per take."""
    from fractions import Fraction
    from scipy.signal import resample_poly
    f = Fraction(rate).limit_denominator(64)
    return resample_poly(x, f.denominator, f.numerator, axis=0).astype(x.dtype)


def highpass(x, hz):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(8, hz / (SR / 2), btype="highpass", output="sos"), x, axis=0).astype(x.dtype)


_tex, _pieces = {}, {}


def texture(path, x=None, key=None):
    """The take's sounding part only, its pauses cut out and the joins crossfaded: what a stroke, a slurp or a groove draws from. x: the take as filtered for the cue."""
    key = key or path
    if key in _tex:
        return _tex[key]
    import numpy as np
    x = load(path) if x is None else x
    e, n = env_of(x, .02)
    on = e > .3 * np.percentile(e, 95)
    on = np.convolve(on.astype(float), np.ones(3), "same") > 0
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i >= 4:
                runs.append(x[i * n:j * n])
            i = j
        else:
            i += 1
    cf = int(.015 * SR)
    if not runs or sum(len(r) for r in runs) < .5 * SR:
        out = x
    else:
        out = runs[0].copy()
        for r in runs[1:]:
            if len(r) <= 2 * cf:
                continue
            ramp = np.linspace(0, 1, cf, dtype=np.float32)[:, None]
            out[-cf:] = out[-cf:] * (1 - ramp) + r[:cf] * ramp
            out = np.concatenate([out, r[cf:]])
    _tex[key] = (out, lufs(out))
    return _tex[key]


def pieces(path):
    """Single events cut out of a take (a munch, a footfall): (audio, onset sample) each."""
    if path in _pieces:
        return _pieces[path]
    import numpy as np
    x = load(path)
    e, n = env_of(x, .01)
    thr = .35 * e.max()
    peaks = [i for i in range(1, len(e) - 1) if e[i] > thr and e[i] >= e[i - 1] and e[i] >= e[i + 1]]
    keep = []
    for p in peaks:
        if not keep or p - keep[-1] > int(.12 / .01):
            keep.append(p)
    out = []
    for k, p in enumerate(keep):
        a = p
        while a > 0 and e[a - 1] > .1 * e[p] and p - a < 6:
            a -= 1
        b = p
        stop = keep[k + 1] - 1 if k + 1 < len(keep) else len(e) - 1
        while b < stop and e[b] > .1 * e[p] and b - p < 35:
            b += 1
        seg = x[max(a * n - int(.004 * SR), 0):b * n].copy()
        f = min(int(.03 * SR), len(seg) // 3)
        if f:
            seg[-f:] *= np.linspace(1, 0, f, dtype=np.float32)[:, None]
        out.append((seg, onset(seg)))
    _pieces[path] = out or [(x, onset(x))]
    return _pieces[path]


def shape(kind, n):
    import numpy as np
    u = np.linspace(0, 1, n, dtype=np.float32)
    if kind == "stroke":          # pressed as it starts, lifted along it
        e = np.minimum(u * n / (.02 * SR), 1) * (1 - .45 * u)
    elif kind == "slow":
        e = np.minimum(u * n / (.25 * SR), 1) * (1 - .4 * u)
    elif kind == "rise":          # a slurp gets quicker to its end
        e = .35 + .65 * u ** 1.6
    elif kind == "thin":          # thick, then a few last ones
        e = np.exp(-3.2 * u) * (1 - u) ** .5
    elif kind == "thread":
        e = .7 + .3 * np.sin(2 * np.pi * .8 * u * n / SR) ** 2
    else:
        e = np.ones(n, np.float32)
    return e.astype(np.float32)


# ---------------------------------------------------------------- choosing takes

_measured = {}


def measure(path):
    """Numbers to tell takes apart by without hearing them."""
    if path in _measured:
        return _measured[path]
    import numpy as np
    x = load(path)
    m = x.mean(1)
    e, n = env_of(x, .01)
    peak = float(np.abs(x).max())
    rms = float(np.sqrt((x * x).mean()))
    spec = np.abs(np.fft.rfft(m[:min(len(m), 20 * SR)] * np.hanning(min(len(m), 20 * SR)))) ** 2
    f = np.fft.rfftfreq(min(len(m), 20 * SR), 1 / SR)
    band = (f > 60) & (f < 16000)
    cen = float((f[band] * spec[band]).sum() / (spec[band].sum() + 1e-20))
    flat = float(np.exp(np.log(spec[band] + 1e-20).mean()) / (spec[band].mean() + 1e-20))
    thr = .35 * e.max()
    ons = sum(1 for i in range(1, len(e)) if e[i] > thr and e[i - 1] <= thr)
    loud = e > .1 * e.max()
    sounding = float(loud.sum() * n / SR)
    seg = e[int(.2 * len(e)):int(.8 * len(e))]
    cv = float(seg.std() / (seg.mean() + 1e-9)) if len(seg) else 0.0
    second = np.sort(e)[-int(.15 / .01)] if len(e) > 15 else e.max()
    single = float(20 * math.log10(e.max() / (np.percentile(e, 60) + 1e-9)))
    out = dict(dur=len(x) / SR, peak=20 * math.log10(peak + 1e-12), rms=20 * math.log10(rms + 1e-12), lufs=lufs(x), centroid=cen, flat=flat, onsets=ons, sounding=sounding, cv=cv, single=single, silent=peak < 1e-3)
    _measured[path] = out
    return out


def auto_take(name):
    """The take to use until 奈亚子 picks one. Hits: the clearest single event over the least noise; beds: the steadiest; music: the first."""
    ks = takes_of(name)
    if not ks:
        return None
    if name in DEFAULT and DEFAULT[name] in ks:
        return DEFAULT[name]
    if name in MUSIC or len(ks) == 1:
        return ks[0]
    scored = []
    for k in ks:
        m = measure(path_of(name, k))
        if m["silent"]:
            continue
        if name in HITS:
            s = m["single"] - 12 * m["flat"] - 2 * max(m["onsets"] - 2, 0)
        elif name in PIECES:
            s = min(m["onsets"], 8) - 10 * m["flat"]
        else:
            s = -m["cv"] - .5 * max(m["peak"] - m["rms"] - 18, 0) / 10
        scored.append((s, k))
    return max(scored)[1] if scored else ks[0]


def resolve(cues_, choices):
    """Each cue's file: a choice by its id, else by its sound's name, else the cue it follows (the fragments follow the theme, the title's seal the ident's), else the default."""
    picked = {}

    def parse(v, name):
        if ":" in str(v):
            n, k = str(v).split(":")
            return n, int(k)
        return name, int(v)
    for c in sorted(cues_, key=lambda c: bool(c["follow"] or c["take_of"])):
        if c["kind"] == "silence":
            c["src"], c["take"] = None, None
            continue
        name = c["name"]
        if c["id"] in choices:
            name, k = parse(choices[c["id"]], name)
        elif c.get("pick"):                               # a cue that names its own take (the credits' paper whisks)
            name, k = c["pick"]
        elif c["follow"] in picked:
            name, k = picked[c["follow"]]
        elif name in choices:
            name, k = parse(choices[name], name)
        elif c["take_of"] in picked and picked[c["take_of"]][1] in takes_of(name):      # the credits: the theme's family
            k = picked[c["take_of"]][1]
        elif c["id"] in CUE_DEFAULT:
            name, k = CUE_DEFAULT[c["id"]]
        else:
            k = auto_take(name)
        c["name"], c["take"] = name, k
        c["src"] = path_of(name, k) if k else None
        picked[c["id"]] = (name, k)
    return cues_


# ---------------------------------------------------------------- laying it down

def dip_end(x, want):
    """Where to end a cut of the theme near want seconds: the quietest moment of its envelope a little before it."""
    import numpy as np
    e, n = env_of(x, .05)
    a, b = int(max(want - 1.2, .5) * SR / n), int(min(want + .3, len(x) / SR) * SR / n)
    if b <= a + 1:
        return min(want, len(x) / SR)
    sm = np.convolve(e, np.ones(4) / 4, "same")
    return (a + int(np.argmin(sm[a:b]))) * n / SR


def cap_peaks(x):
    """One stroke's spikes held to STROKE_PEAK_DB after the master's gain, the rest of it left as it is: the gain each sample needs, its minimum over
    +-STROKE_LOOK, then that averaged over the same width, so the gain on every spike is at or under what it needs and it moves in and out over 5 ms.
    Returns (the stroke, the most taken off in dB, the change in the stroke's K-weighted energy in dB)."""
    import numpy as np
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    if MASTER_GAIN_DB is None or not len(x):    # the cap is set in the master's terms, so it needs the frozen gain
        return x, 0.0, 0.0
    cap = 10 ** ((STROKE_PEAK_DB - MASTER_GAIN_DB) / 20)
    pk = np.abs(x).max(1).astype(np.float64)
    if pk.max() <= cap:
        return x, 0.0, 0.0
    need = np.minimum(1.0, cap / np.maximum(pk, 1e-12))
    w = 2 * int(STROKE_LOOK * SR) + 1
    g = np.minimum(uniform_filter1d(minimum_filter1d(need, w, mode="nearest"), w, mode="nearest"), 1.0)
    y = (x * g[:, None]).astype(np.float32)
    e0, e1 = (kweight(x) ** 2).sum(), (kweight(y) ** 2).sum()
    return y, -20 * math.log10(g.min()), 10 * math.log10(e1 / e0)


def render(c, N):
    """A cue as (start sample, stereo audio), its level and fades applied."""
    import numpy as np
    x = load(c["src"])
    kind = c["kind"]
    if c["rate"]:
        x = slowed(x, c["rate"])
    if c["lp"]:
        x = lowpass(x, c["lp"])
    if c["hp"]:
        x = highpass(x, c["hp"])
    x_all = x
    if kind in ("hit", "piece"):
        if kind == "piece":
            lib = pieces(c["src"])
            key = (c["src"], c.get("stream"))      # a cue with its own stream (the crab's eating) leaves the others' run through the take as it was
            k = render.count.get(key, 0)
            render.count[key] = k + 1
            x, o = lib[k % len(lib)]
            if c["keep"]:                        # a quick run of them (a crab's feet): each cut short so they do not smear
                x = x[:o + int(c["keep"] * SR)].copy()
                f = max(int(len(x) - o) // 2, 1)
                x[-f:] *= np.linspace(1, 0, f, dtype=np.float32)[:, None]
        else:
            if c["keep"]:
                o = onset(x)
                x = x[:o + int(c["keep"] * SR)].copy()
                f = max(int(len(x) - o) // 3, 1)
                x[-f:] *= np.linspace(1, 0, f, dtype=np.float32)[:, None]
            o = peak_at(x) if c["align"] == "peak" else onset(x)
            if c["align"] == "last":             # a run of keys: its last loud click on t, nothing of it before start_after
                m = np.abs(x).max(1)
                o = int(np.flatnonzero(m > .5 * m.max())[-1])
        x = x * (10 ** (c["db"] / 20) / (np.abs(x).max() + 1e-9))
        i0 = int(round(c["t"] * SR)) - o
        if c["start_after"] is not None:
            cut_n = int(round(c["start_after"] * SR)) - i0
            if cut_n > 0:
                x = x.copy()
                x[:cut_n] = 0
                r = min(int(.03 * SR), len(x) - cut_n)
                x[cut_n:cut_n + r] *= np.linspace(0, 1, r, dtype=np.float32)[:, None]
    else:
        n = int(round(c["dur"] * SR))
        if n <= 0:
            return 0, np.zeros((0, 2), np.float32)
        if kind == "seg" and not c["from_start"]:
            tex, loud = texture(c["src"], x, (c["src"], c["lp"], c["hp"], c["rate"]))
            key = (c["src"], c.get("stream"))
            p = render.pointer.get(key, 0)
            idx = (p + np.arange(n)) % len(tex)
            render.pointer[key] = (p + n + int(.25 * SR)) % len(tex)
            x = tex[idx] * shape(c["env"], n)[:, None]
            gain = 10 ** ((c["db"] - loud) / 20)
        elif kind == "bed":
            loud = lufs(x)                            # as filtered: a high-passed bed is set by what is left of it
            x = looped(x, n)
            gain = 10 ** ((c["db"] - loud) / 20)
        else:                                         # music, or a seg read straight from the take's start
            a = int(c["off"] * SR)
            if c["dip"]:
                n = int(dip_end(x[a:], c["dur"]) * SR)
                c["dur"] = n / SR
            x = x[a:a + n]
            if kind == "seg":
                x = x * shape(c["env"], len(x))[:, None]
            whole = x_all[a:a + max(n, int(3 * SR))]
            gain = 10 ** ((c["db"] - lufs(whole)) / 20)
        x = x * gain
        fi, fo = min(int(c["fi"] * SR), len(x) // 2), min(int(c["fo"] * SR), len(x) // 2)
        x = x.astype(np.float32).copy()
        if fi:
            x[:fi] *= (np.linspace(0, 1, fi, dtype=np.float32) ** 2)[:, None]
        if fo:
            x[-fo:] *= (np.linspace(1, 0, fo, dtype=np.float32) ** 2)[:, None]
        if c["env"] in ("stroke", "slow"):             # a stroke's sound lets go just after it ends
            tail = int(.06 * SR)
            x = np.concatenate([x, x[-tail:] * np.linspace(1, 0, tail, dtype=np.float32)[:, None]]) if len(x) > tail else x
        i0 = int(round(c["t"] * SR))
    if c["cut"] is not None:
        m = int(round(c["cut"] * SR)) - i0
        if m < len(x):
            x = x[:max(m, 0)].copy()
            r = min(int(.008 * SR), len(x))
            if r:
                x[-r:] *= np.linspace(1, 0, r, dtype=np.float32)[:, None]
    return i0, x.astype(np.float32)


render.count, render.pointer = {}, {}


def looped(x, n, cross=1.5):
    import numpy as np
    if len(x) >= n:
        return x[:n]
    c = min(int(cross * SR), len(x) // 3)
    out = np.zeros((n, 2), np.float32)
    up = np.sqrt(np.linspace(0, 1, c, dtype=np.float32))[:, None]
    i = 0
    while i < n:
        piece = x[:n - i].copy()
        if i:
            piece[:c] *= up[:len(piece[:c])]
        if len(piece) == len(x):
            piece[-c:] *= up[::-1]
        out[i:i + len(piece)] += piece
        i += len(x) - c
    return out


def lay(bus, i0, x):
    a, b = max(i0, 0), min(i0 + len(x), len(bus))
    if b > a:
        bus[a:b] += x[a - i0:b - i0]


def mask_zero(bus_mask, a, b, ramp=.006):
    """Zero from a to b seconds, the ramps outside the span so that inside it is exactly nothing."""
    import numpy as np
    N = len(bus_mask)
    i, j, r = int(round(a * SR)), int(round(b * SR)), int(ramp * SR)
    i, j = max(i, 0), min(j, N)
    if j <= i:
        return
    bus_mask[i:j] = 0
    lo = max(i - r, 0)
    bus_mask[lo:i] = np.minimum(bus_mask[lo:i], np.linspace(1, 0, i - lo, dtype=np.float32))
    hi = min(j + r, N)
    bus_mask[j:hi] = np.minimum(bus_mask[j:hi], np.linspace(0, 1, hi - j, dtype=np.float32))


def ducking(N):
    """The music's gain: down by DUCK_DB under every line, eased in and out over a third of a second."""
    import numpy as np
    ms = SR // 1000                                   # worked out a millisecond at a time (a sample at a time the smoothing alone takes hours), then held per sample
    on = np.zeros(N // ms + 2, np.float32)
    for c in ink_subs.cues():
        on[max(int(R(c["t0"]) * 1000), 0):max(int(R(c["t1"]) * 1000), 0)] = 1
    if os.path.exists(NARRATION):
        v = np.abs(load(NARRATION)[:N, 0])
        n = min(len(v) // ms, len(on))
        on[:n] = np.maximum(on[:n], v[:n * ms].reshape(n, ms).max(1) > 10 ** (VOICE_DUCK_ON / 20))
    k = 300
    s = np.clip(np.convolve(on, np.ones(k, np.float32) / k, "same"), 0, 1)
    return np.repeat(1 - (1 - 10 ** (DUCK_DB / 20)) * s, ms)[:N].astype(np.float32)


def lay_on(a, t0, t1):
    a[max(int(t0 * SR), 0):min(int(t1 * SR), len(a))] = 1


BUSES = ("fx", "amb", "mus", "brush", "lead", "breath", "voice")


def build(choices):
    """All the buses, the cues laid, gated, ducked and faded. Returns (mix, cue list, what was left out)."""
    import numpy as np
    N = int(round(reel_len() * SR))
    cs = resolve(cues(), choices)
    bus = {b: np.zeros((N, 2), np.float32) for b in BUSES}
    left = []
    render.count.clear()
    render.pointer.clear()
    for c in cs:
        if c["kind"] == "silence":
            continue
        if c["pending"]:
            left.append((c, "待画面"))
            continue
        if not c["src"]:
            left.append((c, "缺素材"))
            continue
        i0, x = render(c, N)
        if c["bus"] == "brush" or c["id"] in CAP_ALSO:
            x, cut, dl = cap_peaks(x)
            if cut > 0:
                c["capped"] = (round(cut, 2), round(dl, 2))
        c["span"] = (i0 / SR, (i0 + len(x)) / SR)
        lay(bus[c["bus"]], i0, x)
    if os.path.exists(NARRATION):          # the narration is a line, like the subtitle: no gate and no silence cue takes it (film_voice keeps it out of them)
        lay(bus["voice"], 0, load(NARRATION)[:N] * 10 ** (VOICE_DB / 20))
    for name in BUSES:
        if name == "voice":
            continue
        m = np.ones(N, np.float32)
        for buses, a, b, _ in gates():
            if name in buses:
                mask_zero(m, a, b)
        for c in cs:
            if c["kind"] == "silence":
                mask_zero(m, c["t"], c["t"] + c["dur"])
        if name == "mus":
            m *= ducking(N)
        bus[name] *= m[:, None]
    if getattr(build, "keep_buses", False):     # for measuring each layer (gated and ducked, before the fades to black)
        build.buses = bus
    out = sum(bus[b] for b in BUSES)
    starts = sorted(c["t"] for c in cs if not c["pending"] and c["kind"] != "silence" and c["src"])
    for a, b, _ in fades():
        i, j = int(round(a * SR)), int(round(b * SR))
        out[i:j] *= (np.linspace(1, 0, j - i, dtype=np.float32) ** 1.5)[:, None]
        nxt = next((t for t in starts if t >= b - 1e-6), b)          # nothing from the fade's end until the next sound starts
        out[j:int(round(nxt * SR))] = 0
    return out, cs, left


# ---------------------------------------------------------------- the subcommands

def mmss(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def plan(choices=None):
    cs = resolve(cues(), choices or {})
    ev = load_events()
    print(f"整卷 {mmss(reel_len())}（{reel_len():.2f} s）：序 0–{HEAD_S:g}，厂标 {IDENT(0):.1f}–{IDENT(IDENT_S):.1f}，正片 {R(HEAD_S):.1f}–{R(film_end()):.2f}，片尾 {CRED(0):.2f}–{CRED(CREDITS_S):.2f}，"
          f"黑 {BLACK(0):.2f}–{BLACK(BLACK_S):.2f}，彩蛋 {STING(0):.2f}–{reel_len():.2f}")
    print(f"cut(): {'%d 个镜头' % len(cut()) if cut() else '读不到，全用 ink_subs'}；events.json: {ev['made'] if ev else '没有（python film_mix.py events）'}")
    if ev and [round(s, 3) for _, s in poem_beats()] != [round(s, 3) for s in ev.get("poem_secs", [])]:
        print("  events.json 已过时：写诗的拍长变了，重跑 events")
    print(f"\n{'卷时':>9} {'片时':>8}  {'cue':<14} {'bus':<6} {'声音:take':<18} {'电平':>6}  锚点 / 应有的声音")
    missing, pending = {}, []
    for c in cs:
        film_t = c["t"] if c["t"] < HEAD_S else c["t"] - OFF
        where = "ident" if IDENT(0) <= c["t"] < IDENT(IDENT_S) and c["id"].startswith("ident") else f"{film_t:8.2f}" if c["t"] < CRED(0) else "片尾" if c["t"] < BLACK(0) else "彩蛋"
        if c["kind"] == "silence":
            print(f"{c['t']:9.2f} {where:>8}  {c['id']:<14} {'—':<6} {'数字静音 %.2f s' % c['dur']:<18} {'':>6}  {c['anchor']} / {c['what']}")
            continue
        src = f"{c['name']}:{c['take']}" if c["take"] else f"{c['name']}:—"
        flag = "  [待画面]" if c["pending"] else "  [缺]" if not c["src"] else ""
        if not c["src"]:
            missing.setdefault(c["name"], []).append(c["id"])
        if c["pending"]:
            pending.append(c["id"])
        dur = f" {c['dur']:.2f}s" if c["dur"] else ""
        print(f"{c['t']:9.2f} {where:>8}  {c['id']:<14} {c['bus']:<6} {src:<18} {c['db']:6.1f}  {c['anchor']} / {c['what']}{dur}{flag}")
    print("\n门（这些 bus 在此为数字零）")
    for buses, a, b, what in gates():
        print(f"  {a:9.2f} – {b:9.2f}  {what}  [{','.join(buses)}]")
    for a, b, what in fades():
        print(f"  {a:9.2f} – {b:9.2f}  {what}（全部随画面淡到零）")
    print(f"\n缺素材 {len(missing)} 种：" + ("；".join(f"{k}（{len(v)} 处）" for k, v in sorted(missing.items())) or "无"))
    print(f"待画面 {len(pending)} 处：" + " ".join(pending))
    return cs


def takes():
    names = sorted({c["name"] for c in cues() if c["name"]} | set(DEFAULT))
    print(f"{'声音':<13} take  {'时长':>5} {'峰值':>6} {'RMS':>6} {'LUFS':>6} {'起音':>4} {'发声':>5} {'质心Hz':>7} {'平坦':>5} {'单击':>5} {'起伏':>5}")
    for name in names:
        ks = takes_of(name)
        pick = auto_take(name)
        if not ks:
            print(f"{name:<13}  —   缺")
            continue
        for k in ks:
            m = measure(path_of(name, k))
            print(f"{name:<13} {k}{'*' if k == pick else ' '}   {m['dur']:5.1f} {m['peak']:6.1f} {m['rms']:6.1f} {m['lufs']:6.1f} {m['onsets']:4d} {m['sounding']:5.1f} {m['centroid']:7.0f} {m['flat']:5.2f} {m['single']:5.1f} {m['cv']:5.2f}{'  静音!' if m['silent'] else ''}")
    print("* 默认 take（DEFAULT 或按数值挑：单击类看「单击」高、「平坦」低；音床看「起伏」低）")


def events():
    """The stroke times, from the writing scripts themselves: a few minutes of glyph work, so cached."""
    import film_big as big
    import film_hand as hand
    out = dict(made=time.strftime("%Y-%m-%d %H:%M"), poem_secs=[s for _, s in poem_beats()])
    page = hand.Page()
    beats = hand.poem_beats()
    for j in range(4):
        secs = beats[j][1]
        t1 = secs * hand.WRITE_PART if j < 3 else beats[0][1] * hand.WRITE_PART
        w = hand.Writing(page, j, 0.0, t1)
        out[f"poem{j}"] = [list(s) for char in w.strokes for s in char]
        print("poem row", j, len(out[f"poem{j}"]), "strokes", flush=True)
    row_end = beats[0][1] * hand.WRITE_PART
    s0 = row_end + hand.SIGN_AFTER[0]
    sign = hand.signing(page, s0, s0 + hand.SIGN_AFTER[1])
    out["sign"] = [list(s) for char in sign.strokes for s in char]
    Tt = consts("film_title.py")
    title = hand.titling(page, *Tt["WRITE"])
    out["title"] = [list(s) for char in title.strokes for s in char]
    print("sign", len(out["sign"]), "title", len(out["title"]), flush=True)
    G = consts("film_myth_gui.py")
    w = big.Writing("鬼", G["W0"] - G["T0"], G["W1"] - G["T0"])
    out["gui"] = [list(s) for s in w.times]
    import film_names
    for a, b, what in film_names.shots():
        if what[0] == "write":
            c = what[1]
            w = big.Writing(c, what[2], None if what[3] is None else what[2] + what[3])
            out["name_" + c] = [list(s) for s in w.times]
            print("name", c, len(w.times), flush=True)
    with open(keep_old(EVENTS), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(EVENTS, flush=True)


def ffmeasure(path):
    sys.path.insert(0, os.path.join(HERE, "..", "sfx"))
    from master_audio import measure as m
    return m(path)


def keep_old(path):
    """Nothing is ever written over: an existing file is renamed NAME_oldN first."""
    if os.path.exists(path):
        stem, ext = os.path.splitext(path)
        n = next(n for n in range(1, 10000) if not os.path.exists(f"{stem}_old{n}{ext}"))
        os.rename(path, f"{stem}_old{n}{ext}")
    return path


def master_gain(out):
    """dB: MASTER_GAIN_DB while it is frozen; else the film itself to BODY_I, or (BODY_I None) the whole reel to TARGET_I."""
    if MASTER_GAIN_DB is not None:
        return MASTER_GAIN_DB
    return TARGET_I - lufs(out) if BODY_I is None else BODY_I - lufs(out[:int(round(CRED(0.0) * SR))])


def mix(choices):
    global CREDITS
    import numpy as np
    t0 = time.time()
    tag = "" if CREDITS == CREDITS_DEFAULT else f"_{CREDITS}"
    if tag:                                 # another credits: the master's gain stays the film's own, so nothing outside the credits changes by a sample
        CREDITS, keep = CREDITS_DEFAULT, CREDITS
        base = build(choices)[0]
        gain = master_gain(base)
        del base
        CREDITS = keep
    out, cs, left = build(choices)
    N = len(out)
    pre = lufs(out)
    if not tag:
        gain = master_gain(out)
    os.makedirs(SOUND, exist_ok=True)
    raw = os.path.join(SOUND, f"film_mix{tag}_raw.wav")
    limit = CEILING_TP - .4
    res = None
    subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "pcm_f32le", keep_old(raw)], input=(out * 10 ** (gain / 20)).astype(np.float32).tobytes(), check=True)
    for attempt in range(4):
        tried = keep_old(os.path.join(SOUND, f"film_mix{tag}_try{attempt + 1}.wav"))
        chain = f"alimiter=limit={10 ** (limit / 20):.5f}:attack=1:release=80:level=disabled:latency=1,apad=whole_len={N},atrim=end_sample={N}"
        subprocess.run(["ffmpeg", "-v", "error", "-i", raw, "-af", chain, "-c:a", "pcm_s24le", "-ar", str(SR), tried], check=True)
        res = ffmeasure(tried)
        if res["TP"] <= CEILING_TP:
            break
        subprocess.run(["mv", tried, os.path.expanduser("~/.Trash/")], check=False)       # a try over the ceiling goes to the Trash, never deleted
        limit -= res["TP"] - CEILING_TP + .1
    os.rename(tried, keep_old(OUT))
    os.rename(raw, keep_old(os.path.join(SOUND, f"film_mix{tag}_premaster.wav")))
    final = load_fresh(OUT)
    assert len(final) == N, (len(final), N)
    # the numbers: loudness, the limiter's work, silence where silence is asked for, onsets on their frames
    before = float(np.abs(out).max()) * 10 ** (gain / 20)
    print(f"\n{OUT}\n  {N / SR:.3f} s = 整卷 {reel_len():.3f} s，{N} 个采样，48 kHz 立体声 24 bit")
    print(f"  正片 {lufs(final[:int(round(CRED(0.0) * SR))]):.1f} LUFS（{'增益冻结在 %+.4f dB' % MASTER_GAIN_DB if MASTER_GAIN_DB is not None else '限幅前定在 %s' % BODY_I}），片尾 {lufs(final[int(round(CRED(0.0) * SR)):int(round(BLACK(0.0) * SR))]):.1f} LUFS" if BODY_I is not None else "")
    print(f"  响度 I {res['I']:.1f} LUFS（{'增益冻结' if MASTER_GAIN_DB is not None else '目标 %g' % TARGET_I if BODY_I is None else '由正片定'}），LRA {res['LRA']:.1f} LU，真峰值 {res['TP']:.1f} dBTP（上限 {CEILING_TP:g}）；增益 {gain:+.1f} dB，限幅前最高 {20 * math.log10(before + 1e-12):+.1f} dBFS，限幅线 {limit:.1f} dBFS")
    print("  该静的地方（最大绝对值）：")
    spans = [(a, b, w) for (bs, a, b, w) in gates() if set(bs) >= {"fx", "amb", "mus", "brush"}]
    spans += [(c["t"], c["t"] + c["dur"], c["what"]) for c in cs if c["kind"] == "silence"]
    for a, b, what in spans:
        seg = final[int(a * SR) + int(.01 * SR):int(b * SR) - int(.01 * SR)]
        if what.startswith("第七击"):           # the breath alone may sound there: everything outside its own laid span must be zero
            br = [c for c in cs if c["id"] == "s7_breath" and c.get("span")]
            if br:
                p0, p1 = br[0]["span"]
                seg = np.concatenate([final[int(a * SR) + int(.01 * SR):max(int(p0 * SR), int(a * SR) + int(.01 * SR))], final[min(int(p1 * SR), int(b * SR)):int(b * SR) - int(.01 * SR)]])
                what += f"；呼气占 {p0:.2f}–{p1:.2f}"
        mx = float(np.abs(seg).max()) if len(seg) else 0.0
        print(f"    {a:9.2f} – {b:9.2f}  {what}: {'数字零' if mx == 0 else '%.1f dBFS' % (20 * math.log10(mx))}")
    late, masked = [], 0
    for c in cs:
        if c["kind"] in ("hit", "piece") and c["src"] and not c["pending"] and c["align"] == "onset" and c["db"] > -26 and c["bus"] != "breath":
            i = int(round(c["t"] * SR))
            bg = float(np.abs(final[max(i - int(.25 * SR), 0):max(i - int(.06 * SR), 0)]).max(initial=0.0))
            pk = float(np.abs(final[i:i + int(.12 * SR)]).max(initial=0.0))
            if pk == 0 or pk < 3 * bg:          # under a louder bed or another sound: its onset cannot be read off the mix
                masked += 1
                continue
            m = np.abs(final[i - int(.05 * SR):i + int(.12 * SR)]).max(1)
            got = c["t"] - .05 + int(np.argmax(m > max(.2 * pk, 2 * bg))) / SR
            late.append((abs(got - c["t"]), c["id"]))
    if late:
        late.sort(reverse=True)
        print(f"  起音对帧：量了 {len(late)} 个单响（{masked} 个被底下的声音盖住，不量），偏差最大的 " + "，".join(f"{c} {d * 1000:.1f} ms" for d, c in late[:3]) + f"（一帧 {1000 / FPS:.1f} ms）")
    # the limiter's work on the loud single sounds: how much it took off each one's peak
    pm = load_fresh(os.path.join(SOUND, f"film_mix{tag}_premaster.wav"))
    gr = []
    for c in cs:
        if c["kind"] in ("hit", "piece", "music") and c.get("span") and c["src"]:
            i, j = max(int(c["span"][0] * SR), 0), min(int(c["span"][1] * SR), N)
            if j > i:
                a_, b_ = float(np.abs(pm[i:j]).max()), float(np.abs(final[i:j]).max())
                if a_ > 0 and b_ > 0:
                    gr.append((20 * math.log10(a_ / b_), c["id"], 20 * math.log10(b_)))
    gr.sort(reverse=True)
    print("  限幅器压掉的峰（dB，前 8）：" + "，".join(f"{i} {g:.1f}（剩 {p:.1f} dBFS）" for g, i, p in gr[:8]))
    blk = SR // 10                          # and over the whole reel, 100 ms blocks: the limiter's work wherever it is, beds and strokes too
    n = N // blk * blk
    a_ = np.abs(pm[:n]).max(1).reshape(-1, blk).max(1)
    b_ = np.abs(final[:n]).max(1).reshape(-1, blk).max(1)
    bg = 20 * np.log10((a_ + 1e-12) / (b_ + 1e-12))
    top = []
    for i in np.argsort(bg)[::-1]:
        if len(top) == 5:
            break
        if not any(abs(i - j) < 20 for j in top):
            top.append(i)
    names = lambda t: "+".join(c["id"] for c in cs if c.get("span") and c["span"][0] <= t + .1 and c["span"][1] >= t and c["kind"] != "bed")[:40] or "底噪"
    print("  限幅器按 100 ms 块（前 5）：" + "，".join(f"{i / 10:.1f} s {bg[i]:.1f} dB（{names(i / 10)}）" for i in top))
    capped = [(c["capped"][0], c["capped"][1], c["id"]) for c in cs if c.get("capped")]
    if capped:
        print(f"  峰值上限 {STROKE_PEAK_DB} dBFS（brush 总线和 {'、'.join(sorted(CAP_ALSO))}）：{len(capped)} 处被削，最多削 {max(capped)[0]:.1f} dB（{max(capped)[2]}），整笔能量变化最大 {min(x[1] for x in capped):+.2f} dB")
    for b in ("theme", "credits"):
        c = next((c for c in cs if c["id"] == b and c["src"]), None)
        if c:
            seg = final[int(c["t"] * SR):int((c["t"] + c["dur"]) * SR)]
            print(f"  {b}: {lufs(seg):.1f} LUFS，{c['dur']:.1f} s，take {c['take']}")
    if left:
        print(f"  没放进去的 {len(left)} 个：" + "，".join(f"{c['id']}（{why}）" for c, why in left))
    with open(keep_old(os.path.join(SOUND, f"film_mix{tag}_cues.json")), "w") as f:
        json.dump([{k: v for k, v in c.items() if k != "src"} | dict(src=os.path.relpath(c["src"], ROOT) if c.get("src") else None) for c in cs], f, ensure_ascii=False, indent=0)
    print(f"  {time.time() - t0:.0f} s")


def load_fresh(path):
    _audio.pop(path, None)
    return load(path)


def listen(only=None):
    """Each sound's takes in a row, a second of nothing between, all at one loudness (hits at one peak), so they can be compared quickly. A silent take is left out.
    With names, only those; a file made before goes into listen/old/."""
    import numpy as np
    d = os.path.join(SOUND, "listen")
    os.makedirs(d, exist_ok=True)
    names = sorted(only or {c["name"] for c in cues() if c["name"]})
    for name in names:
        ks = [k for k in takes_of(name) if not measure(path_of(name, k))["silent"]]
        if len(ks) < 2:
            continue
        parts = []
        for k in ks:
            x = load(path_of(name, k))
            if name == "furrow":                     # as the mix plays it
                x = lowpass(slowed(x, LINE_RATE), LINE_LP)
            elif name in ("dry_wind", "riverbed"):
                x = highpass(x, RIVER_HP)
            if name in HITS or name in PIECES:
                x = x * (10 ** (-6 / 20) / (np.abs(x).max() + 1e-9))
            else:
                x = x * 10 ** ((-23 - lufs(x)) / 20)
            parts += [x[:int(30 * SR)], np.zeros((SR, 2), np.float32)]
        y = np.clip(np.concatenate(parts), -1, 1)
        p = os.path.join(d, f"{name}.wav")
        if os.path.exists(p):
            os.makedirs(os.path.join(d, "old"), exist_ok=True)
            os.rename(p, keep_old(os.path.join(d, "old", f"{name}.wav")))
        subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "pcm_s16le", p], input=y.astype(np.float32).tobytes(), check=True)
        print(p, " ".join(f"take{k}" for k in ks), flush=True)


def mux(video):
    """The picture as it is, the mix under it as AAC 320k, into VIDEO_mix.mp4 next to it. A length that differs from the reel's is reported, and the sound is padded or cut to the picture."""
    probe = lambda p: float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", p]).decode().strip())
    v, a = probe(video), probe(OUT)
    dst = os.path.splitext(video)[0] + "_mix.mp4"
    keep_old(dst)
    print(f"画面 {v:.3f} s，声音 {a:.3f} s，差 {v - a:+.3f} s" + ("" if abs(v - a) < 1 / FPS else "  ← 画面不是现在这卷的长度：先 film_rough.py cut"), flush=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-i", OUT, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-af", "apad", "-t", f"{v:.3f}",
                    "-c:a", "aac", "-b:a", "320k", "-ar", str(SR), "-movflags", "+faststart", dst], check=True)
    print(dst, ffmeasure(dst), flush=True)


def main():
    global CREDITS, OUT
    for a in sys.argv[2:]:
        if a.startswith("--credits="):
            CREDITS = a.split("=", 1)[1]
            OUT = os.path.join(SOUND, f"film_mix_{CREDITS}.wav") if CREDITS != CREDITS_DEFAULT else OUT
    what = sys.argv[1] if len(sys.argv) > 1 else "plan"
    choices = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a and not a.startswith("--"))
    if what == "plan":
        plan(choices)
    elif what == "takes":
        takes()
    elif what == "events":
        events()
    elif what == "mix":
        mix(choices)
    elif what == "listen":
        listen(sys.argv[2:] or None)
    elif what == "mux":
        mux(sys.argv[2])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
