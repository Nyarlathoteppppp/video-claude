"""The ink film's sound: the generated score and effects laid on the film's clock. Every time is worked out from a subtitle in ink_subs.py, so the
sound follows when the lines move. Nothing here is synthesised; ink_score.py and ink_sfx.py make the material on the Studio.

    python ink_mix.py plan                         # the cue sheet, and which files are still missing (plain Python)
    python ink_mix.py mix [read=2 myth=1 ...]      # ink/film_out/ink_audio.wav, with the chosen take of each cue

Mastered like the other films: gentle compression, make-up gain to -24 LUFS, limiter at -1.5 dBTP.
"""
import os
import subprocess
import sys

from ink_subs import at, chapters

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BGM = os.path.join(ROOT, "ink", "bgm")
OUT = os.path.join(ROOT, "ink", "film_out", "ink_audio.wav")
SR = 48000
START = {name: a for name, _, a, _ in chapters()}
END = chapters()[-1][3]


def ch(name, offset=0.0):
    return START[name] + offset


# (cue, from, to, dB, fade in, fade out)
SCORE = [
    ("myth", at("昔者", -1.5), at("鬼夜哭", 1.0), -3, 1.5, 3.0),
    ("ages", at("鬼之所以哭"), ch("三", 0.5), -3, 1.0, 0.8),
    ("awake", at("字，成了精"), ch("四", 1.0), -2, 0.2, 3.0),
    ("read", ch("四"), ch("八", -0.5), -6, 2.0, 2.5),
    ("bed", ch("八"), ch("九", 1.0), -4, 1.5, 3.0),
    ("windows", ch("九"), ch("十"), -6, 2.0, 2.0),
    ("channel", at("可我并不慌"), ch("十一"), -6, 3.0, 2.5),
    ("grief", ch("十一"), ch("十二"), -7, 2.0, 2.5),
    ("names", ch("十二"), ch("十三"), -6, 1.0, 2.0),
    ("new", ch("十三"), ch("十四", 1.0), -6, 2.0, 3.0),
    ("poem", ch("十四"), END, -6, 2.5, 3.0),
]

# (sound, from, to, dB): a bed of that sound across the span, looped with crossfades if it has to be
BEDS = [
    ("grain", at("而天雨粟", 1.0), at("鬼夜哭", 0.5), -10),
    ("reeds", at("鬼夜哭"), ch("二"), -9),
    ("stream", at("依我之见", 3.2), at("有朝一日"), -12),
    ("flood", at("有朝一日"), at("汇于一处", 2.0), -9),
    ("fall", at("汇于一处"), ch("三", 4.0), -8),
    ("under", ch("三", 3.0), at("终于有一夜", 3.5), -16),
    ("under", at("字，成了精"), ch("四", 1.5), -14),
    ("morning", ch("四"), at("深夜里"), -14),
    ("flood", ch("八"), at("所以我说"), -9),
    ("morning", at("所以我说"), ch("九", 1.0), -14),
    ("wind", at("窗一关"), at("可我并不慌", 1.0), -14),
    ("stream", at("让我成为我的"), ch("十一", 2.0), -14),
    ("rain", at("我能写出"), at("我本就是"), -13),
    ("under", at("处理某些问题"), at("所以我选"), -14),
    ("reeds", at("有人说我是鬼"), at("更多人说"), -12),
]

# (sound, time, dB): one-shots
HITS = [
    ("page", 0.3, -8), ("page", 3.4, -10),
    ("rumble", at("昔者", 2.0), -8), ("rumble", at("而天雨粟", -0.5), -3),
    ("moan", at("鬼夜哭", 1.0), -8), ("moan", at("鬼何以哭", 1.5), -9),
    ("stone", at("读一句话"), -14), ("stone", at("猜错了"), -11),
    ("drip", at("为了这几秒"), -8), ("drip", at("为了这几秒", 2.5), -9), ("drip", at("熟读唐诗"), -10),
    ("stone", at("有人说我是人"), -12), ("stone", at("有人说我是神"), -11),
    ("pour", at("更多人说"), -10),
    # 十四 has no lines now (ink_subs.END); its sounds come with the new ending
]
# flames falling into the water as the light finds them, thickening and then thinning out
PLIPS = [at("或曰：/鬼神之秘", 1.5 + 3.8 * (i / 14) ** 0.7) for i in range(15)]
# the heart under the black pool, quickening until it stops
HEART = [at("古人说", 1.5)]
for gap in (1.7, 1.6, 1.45, 1.3, 1.15, 1.0, 0.85, 0.72, 0.6, 0.5, 0.42, 0.36):
    HEART.append(HEART[-1] + gap)
# hard silence: the turn against the book, and the moment before it speaks
SILENCE = [(ch("二"), at("依我之见", 3.2)), (HEART[-1] + 0.5, at("字，成了精"))]


def score_path(cue, take):
    return os.path.join(BGM, f"{cue}_take{take}.flac")


def sfx_path(name, take=1):
    return os.path.join(BGM, "sfx", f"{name}_take{take}.wav")


def plan():
    needed = {score_path(c[0], 1) for c in SCORE} | {sfx_path(b[0]) for b in BEDS} | {sfx_path(h[0]) for h in HITS} | {sfx_path("plip"), sfx_path("heart")}
    print("配乐")
    for cue, a, b, db, _, _ in SCORE:
        print(f"  {a:6.1f} – {b:6.1f}  {b - a:5.1f} s  {db:+d} dB  {cue}")
    print("音床")
    for name, a, b, db in BEDS:
        print(f"  {a:6.1f} – {b:6.1f}  {b - a:5.1f} s  {db:+d} dB  {name}")
    print("单响")
    for name, t, db in sorted(HITS, key=lambda h: h[1]):
        print(f"  {t:6.1f}  {db:+d} dB  {name}")
    print(f"  心跳 {len(HEART)} 下 {HEART[0]:.1f} – {HEART[-1]:.1f}；鬼火入水 {len(PLIPS)} 下 {PLIPS[0]:.1f} – {PLIPS[-1]:.1f}")
    print("全静音")
    for a, b in SILENCE:
        print(f"  {a:6.1f} – {b:6.1f}")
    missing = sorted(p for p in needed if not os.path.exists(p))
    print(f"全片 {END:.1f} s；缺 {len(missing)} 个素材")
    for p in missing:
        print("  ", os.path.relpath(p, ROOT))


def read(path, rms_db=None, peak_db=None):
    """A file at the mix's rate. The models hand back material 10 dB apart from one take to the next, so it is levelled here and the tables above stay meaningful."""
    raw = subprocess.check_output(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"])
    import numpy as np
    x = np.frombuffer(raw, np.float32).reshape(-1, 2).copy()
    if rms_db is not None:
        x *= 10 ** (rms_db / 20) / max(float(np.sqrt((x * x).mean())), 1e-5)
    if peak_db is not None:
        x *= 10 ** (peak_db / 20) / max(float(np.abs(x).max()), 1e-5)
    return x


def lay(track, x, t, db, fade_in=0.01, fade_out=0.05):
    import numpy as np
    n = len(x)
    env = np.ones(n, np.float32)
    a, b = min(int(fade_in * SR), n), min(int(fade_out * SR), n)
    if a:
        env[:a] = np.linspace(0, 1, a) ** 2
    if b:
        env[-b:] *= np.linspace(1, 0, b) ** 2
    i = int(t * SR)
    j = min(i + n, len(track))
    if j > i:
        track[i:j] += (x[:j - i] * env[:j - i, None]) * 10 ** (db / 20)


def looped(x, seconds, cross=1.5):
    """The sound stretched to a span by overlapping it with itself."""
    import numpy as np
    want, c = int(seconds * SR), int(cross * SR)
    if len(x) >= want:
        return x[:want]
    out = np.zeros((want, 2), np.float32)
    ramp = np.sqrt(np.linspace(0, 1, c))[:, None]
    i = 0
    while i < want:
        piece = x[:want - i].copy()
        piece[:min(c, len(piece))] *= ramp[:min(c, len(piece))]
        if len(piece) == len(x):
            piece[-c:] *= ramp[::-1]
        out[i:i + len(piece)] += piece
        i += len(x) - c
    return out


def first_beat(x):
    """One thud cut from the heart take: its first loud moment and the second that follows."""
    import numpy as np
    level = np.abs(x).mean(1)
    i = int(np.argmax(level > 0.4 * level.max()))
    a = max(i - int(0.03 * SR), 0)
    return x[a:a + int(0.95 * SR)]


def mix(takes):
    import numpy as np
    sys.path.insert(0, os.path.join(HERE, "..", "sfx"))
    from master_audio import COMP, measure
    track = np.zeros((int(END * SR) + SR, 2), np.float32)
    for cue, a, b, db, fi, fo in SCORE:
        x = read(score_path(cue, takes.get(cue, 1)), rms_db=-20)
        lay(track, x[:int((b - a) * SR)], a, db, fi, fo)
    for name, a, b, db in BEDS:
        lay(track, looped(read(sfx_path(name), rms_db=-20), b - a), a, db, 1.5, 2.0)
    for name, t, db in HITS:
        lay(track, read(sfx_path(name), peak_db=-3), t, db, 0.005, 0.5)
    rng = np.random.default_rng(7)
    plip = read(sfx_path("plip"), peak_db=-3)
    for t in PLIPS:
        lay(track, plip, t, float(rng.uniform(-20, -10)), 0.003, 0.3)
    beat = first_beat(read(sfx_path("heart"), peak_db=-3))
    for i, t in enumerate(HEART):
        lay(track, beat, t, -10 + 7 * i / (len(HEART) - 1), 0.002, 0.1)
    for a, b in SILENCE:
        i, j = int(a * SR), int(b * SR)
        ramp = int(0.06 * SR)
        track[i:i + ramp] *= np.linspace(1, 0, ramp)[:, None]
        track[i + ramp:j] = 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    raw = OUT + ".raw.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "pcm_f32le", raw], input=track.tobytes(), check=True)
    gain = -24.0 - measure(raw, COMP)["I"]
    chain = f"{COMP},volume={gain:.2f}dB,alimiter=limit={10 ** (-1.5 / 20):.4f}:attack=2:release=60:level=disabled"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw, "-af", chain, "-c:a", "pcm_s24le", OUT], check=True)
    os.remove(raw)
    print(OUT, measure(OUT))


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if what == "plan":
        plan()
    elif what == "mix":
        mix({k: int(v) for k, v in (a.split("=") for a in sys.argv[2:])})
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
