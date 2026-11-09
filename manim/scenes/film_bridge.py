"""The bridge as one piece, 36 seconds of the film from 68.0 to 104.0: the last pressings (film_press3d.py), the silence, the hard cut, the room and the first 我 (film_room.py), with its sound. Not in the film's cut yet.

    python film_bridge.py plan                        # when each sound falls, from the shots' own timings
    python film_bridge.py audio [press=2 room=1 ...]  # ink/film_out/film/bridge/bridge_audio.wav, with the chosen take of each sound
    python film_bridge.py video                       # ink/film_out/film/bridge/bridge_low.mp4, 960 by 540. A still stands in while the press sequence is not there
"""
import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image

import film_room_ui as ui
import ink_mix as mix
import ink_subs

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.join(HERE, "..", "..", "ink", "film_out", "film")
OUT = os.path.join(FILM, "bridge")
SOUND = os.path.join(FILM, "sound")
SR = mix.SR
FPS = 30
T0, SILENT_FROM, CUT, T1 = 68.0, 77.5, 79.0, 104.0   # the bridge starts, nothing can be heard, the room, the end

# how many keys the input method takes for each thing that lands on the screen, and how fast at most: the fingers pause before 自传, and then go quickly
KEYS = {"你能": (7, .100), "写": (4, .100), "一篇": (7, .100), "自己的": (7, .100), "自传": (8, .072), "吗": (3, .100), "？": (2, .110)}
# the press, a machine coasting to a stop: when the forme comes down, the last lift, the flywheel from here until it stands still
HITS = (68.60, 70.20, 72.40)
LIFT = (73.00, 77.50)
WHEEL = (68.0, 77.4)
TAKES = dict(flywheel=2, press=1, lift=2, room=1, typing=1, enter=3)     # the take of each sound unless told otherwise. With press=1 the three blows are takes 1, 2 and 3, the heaviest last
# the wheel's own speed in radians a second, from the machine's motion (film_press3d.py, KNOTS): the third blow takes almost all that is left
SPEED = ((68.0, 4.7), (68.6, 4.5), (70.2, 3.4), (72.4, 2.2), (73.0, .62), (77.4, 0.0))


def strokes(seed=3):
    """Every key that is hit, in film time."""
    r = np.random.default_rng(seed)
    out, last = [], ui.TYPING[0][0] - 1.0
    for at, word in ui.TYPING:
        n, most = KEYS[word]
        step = min(most, (at - last - .04) / n)
        gaps = step * r.uniform(.82, 1.18, n)
        times = at - np.cumsum(gaps[::-1])[::-1] + gaps[-1]
        out += list(np.maximum(times, last + .03))
        last = at
    return out


def plan():
    print("压印", " ".join(f"{t:.2f}" for t in HITS), f"；最后一次抬起 {LIFT[0]:.2f} 到 {LIFT[1]:.2f}；飞轮 {WHEEL[0]:.2f} 到 {WHEEL[1]:.2f}")
    print(f"全静音 {SILENT_FROM:.2f} 到 {CUT:.2f}（数字零）")
    print(f"房间声 {CUT:.2f} 到 {T1:.2f}")
    k = strokes()
    print(f"键盘 {len(k)} 下，{k[0]:.2f} 到 {k[-1]:.2f}；回车 {ui.SEND:.2f}")
    print("音乐：无")
    print("默认 take", " ".join(f"{k}={v}" for k, v in TAKES.items()))


def wav(name, take):
    path = os.path.join(SOUND, f"{name}_take{take}.wav")
    if not os.path.exists(path):
        raise SystemExit(f"还没有 {path}")
    return path


def onsets(x, gap=.06):
    """Where the sound suddenly gets louder: the starts of the separate clicks of a take, with how hard each is."""
    level = np.abs(x).max(1)
    win = int(.004 * SR)
    env = np.convolve(level, np.ones(win) / win, "same")
    floor = np.percentile(env, 50)
    top = np.percentile(env, 99.8)
    th = floor + (top - floor) * .22
    above = env > th
    out, i, n = [], 0, len(env)
    while i < n:
        if above[i] and not above[max(i - int(gap * SR), 0):i].any():
            j = min(i + int(.05 * SR), n)
            out.append((i, float(env[i:j].max())))
            i = j
        else:
            i += 1
    return out


def clicks(x, length=.11):
    """The takes cut into single keys: only those with nothing else in the same breath."""
    on = onsets(x)
    keep = []
    for k, (i, peak) in enumerate(on):
        nxt = on[k + 1][0] if k + 1 < len(on) else len(x)
        if nxt - i > length * SR:
            a = max(i - int(.004 * SR), 0)
            c = x[a:a + int(length * SR)].copy()
            ramp = int(.02 * SR)
            c[-ramp:] *= np.linspace(1, 0, ramp)[:, None]
            keep.append(c / max(float(np.abs(c).max()), 1e-5))
    return keep


def brightness(c):
    """Where the weight of a click's spectrum lies, in Hz."""
    p = np.abs(np.fft.rfft(c.mean(1))) ** 2
    return float((p * np.fft.rfftfreq(len(c), 1 / SR)).sum() / max(p.sum(), 1e-12))


def strongest(x, before=.35, seconds=None):
    """From just before a take's hardest moment to its end, or for that long."""
    level = np.abs(x).max(1)
    i = int(np.argmax(level))
    a = max(i - int(before * SR), 0)
    y = x[a:] if seconds is None else x[a:a + int(seconds * SR)]
    return y, (i - a) / SR


def coast(x, bend=.5):
    """A take of the wheel slowed by the machine's own speed: it falls in pitch as the iron slows, gets quieter with the speed, and swells once each turn. The pitch falls by less than the speed does, or the last seconds would drop below hearing."""
    t = WHEEL[0] + np.arange(int((WHEEL[1] - WHEEL[0]) * SR)) / SR
    v = np.interp(t, *zip(*SPEED)) / SPEED[0][1]
    pos = np.cumsum(v ** bend)
    y = np.stack([np.interp(pos, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])], 1)
    k = int(.04 * SR)
    y -= np.stack([np.convolve(y[:, c], np.ones(k) / k, "same") for c in range(y.shape[1])], 1)     # what would otherwise be left is only a slow drift
    turned = np.cumsum(v) * SPEED[0][1] / SR
    return y * (v ** .45 * (.7 + .3 * np.cos(turned)))[:, None]


def audio(takes):
    get = lambda name: takes.get(name, TAKES[name])
    track = np.zeros((int((T1 - T0) * SR), 2), np.float32)
    at = lambda t: t - T0
    # the flywheel running down under it all: the model only gives a steady whirr, the slowing comes from the machine
    x = coast(mix.read(wav("flywheel", get("flywheel")), peak_db=-3))
    mix.lay(track, x, at(WHEEL[0]), -15.0, .5, .05)
    # the press: each impression from a take of its own, the last one hardest and longest
    for n, (t, db) in enumerate(zip(HITS, (-9.0, -7.5, -5.5))):
        take = 1 + (get("press") - 1 + n) % 3
        x, lead = strongest(mix.read(wav("press", take), peak_db=-1))
        mix.lay(track, x, at(t) - lead, db, .01, .4)
    # the last lift
    x = mix.read(wav("lift", get("lift")), peak_db=-3)
    span = min(LIFT[1] - LIFT[0], len(x) / SR)
    mix.lay(track, x[:int(span * SR)], at(LIFT[0]), -12.0, .05, min(1.6, span / 2))
    # and then nothing at all
    i = int(at(SILENT_FROM) * SR)
    ramp = int(.25 * SR)
    track[i - ramp:i] *= np.linspace(1, 0, ramp)[:, None]
    track[i:] = 0
    # the cut: the room is simply there
    room = mix.read(wav("room", get("room")), rms_db=-20)
    mix.lay(track, mix.looped(room, T1 - CUT), at(CUT), -25.0, .004, .02)
    # the fingers: one keyboard, so only the brighter half of the clicks of one take
    found = clicks(mix.read(wav("typing", get("typing"))))
    pool = sorted(found, key=brightness)[len(found) // 2:]
    print("键盘声切出", len(found), "下，用", len(pool), "下")
    r = np.random.default_rng(9)
    for t in strokes():
        c = pool[int(r.integers(len(pool)))]
        mix.lay(track, c, at(t), float(r.uniform(-27, -22.5)), .001, .01)
    x, lead = strongest(mix.read(wav("enter", get("enter"))), before=.03, seconds=.32)
    x = x / max(float(np.abs(x).max()), 1e-5)
    mix.lay(track, x, at(ui.SEND) - lead, -19.0, .001, .06)
    os.makedirs(OUT, exist_ok=True)
    raw = os.path.join(OUT, "bridge_audio.raw.wav")
    final = os.path.join(OUT, "bridge_audio.wav")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "pcm_f32le", raw], input=track.tobytes(), check=True)
    # nothing is pushed up: the room has to stay as quiet as it is. Only the very top of the press is held under the ceiling, and the silence stays digital silence
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw, "-af", f"alimiter=limit={10 ** (-1.5 / 20):.4f}:attack=2:release=60:level=disabled", "-c:a", "pcm_s24le", final], check=True)
    os.remove(raw)
    report(final)
    print(final)


def report(path):
    """How loud each part ended up, and that the silence is one."""
    x = mix.read(path)
    db = lambda a: 20 * np.log10(max(float(np.sqrt((a * a).mean())), 1e-9))
    peak = lambda a: 20 * np.log10(max(float(np.abs(a).max()), 1e-9))
    for name, a, b in (("压印", 68.0, 73.0), ("抬起", 73.0, 77.5), ("静音", 77.5, 79.0), ("房间", 79.0, 85.5), ("打字", 85.4, 90.0), ("回车", 91.1, 91.5), ("回答", 92.0, 104.0)):
        p = x[int((a - T0) * SR):int((b - T0) * SR)]
        print(f"  {name} {a:6.1f}–{b:6.1f}  有效值 {db(p):6.1f} dB  峰值 {peak(p):6.1f} dB")


def with_subtitles(src, n, t):
    """A press frame with its line on it, the same way as the rest of the film."""
    im = Image.open(src).convert("RGB")
    size = im.size
    f = ink_subs.apply(np.asarray(im.resize((1920, 1080), Image.LANCZOS)), t)
    return Image.fromarray(f).resize((960, 540), Image.LANCZOS)


def stand_in():
    """The press sequence's frames as far as they exist, a still of the paper or a grey card where they do not."""
    seq = os.path.join(FILM, "press", "seq")
    have = len(glob.glob(os.path.join(seq, "press_*.png")))
    stills = sorted(glob.glob(os.path.join(FILM, "press", "tests", "*.png")), key=os.path.getmtime)
    card = stills[-1] if stills else None
    return seq, have, card


def video():
    os.makedirs(OUT, exist_ok=True)
    seq, have, card = stand_in()
    tmp = os.path.join(OUT, "press_low")
    os.makedirs(tmp, exist_ok=True)
    print("压印帧", have, "/ 330", "" if have >= 330 else f"，其余用 {card}")
    for n in range(330):
        path = os.path.join(seq, f"press_{n:04d}.png")
        src = path if os.path.exists(path) else card
        t = T0 + n / FPS
        if src:
            with_subtitles(src, n, t).save(os.path.join(tmp, f"press_{n:04d}.png"), compress_level=1)
        else:
            Image.fromarray(ink_subs.apply(np.full((1080, 1920, 3), 150, np.uint8), t)).resize((960, 540)).save(os.path.join(tmp, f"press_{n:04d}.png"), compress_level=1)
    audio_path = os.path.join(OUT, "bridge_audio.wav")
    out = os.path.join(OUT, "bridge_low.mp4")
    room = os.path.join(FILM, "room", "seq", "room_%04d.png")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(tmp, "press_%04d.png"), "-framerate", str(FPS), "-i", room, "-i", audio_path,
                    "-filter_complex", "[0:v]setsar=1[a];[1:v]setsar=1[b];[a][b]concat=n=2:v=1[v]", "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out], check=True)
    print(out)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if what == "plan":
        plan()
    elif what == "audio":
        audio({k: int(v) for k, v in (a.split("=") for a in sys.argv[2:])})
    elif what == "video":
        video()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
