"""The narration: every subtitle line spoken in one designed voice, laid on the reel clock for film_mix's voice bus.

    python film_voice.py lines     # voice/narration_c/lines/NNN.wav, one per ink_subs cue, and voice/narration_c/poem/N.wav, the poem's rows (the voice env: mlx-audio, Qwen3-TTS on the Studio)
    python film_voice.py track     # sound/narration.wav (mono, 48 kHz, the whole reel) and sound/narration_lines.tsv (the render env)

The voice (2026-11-08, 奈亚子 chose c from twelve auditions: 「这个可以」, then 「直接定稿」): Qwen3-TTS 1.7B VoiceDesign spoke REF_TEXT from DESIGN
(voice/qwen/ref/c_中性AI.wav); Qwen3-TTS 1.7B Base clones that clip for every line, so the film has one voice. Both Apache-2.0, run locally.
No real person's voice is cloned.
A line starts LEAD after its subtitle starts (the fade in); one that would run into the next line, or into one of the film's silences (quiet), is sped up,
at most MAX_TEMPO, and only then starts late. A line never starts inside a silence.
The poem (2026-11-08, 奈亚子: 「每次写诗的时候配音一起跟着念」): each time a row is written the same voice reads it along with the brush. One take per row,
split at its caesura; each half starts as the brush starts that half (the first stroke, and the first after the row's longest rest), in the poem's own gates,
which take everything but the brush and the voice. The track's gain is set by the subtitle lines alone, so they stay as they were.
"""
import ast
import os
import subprocess
import sys

import ink_subs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FILM = os.path.join(ROOT, "ink", "film_out", "film")
LINES = os.path.join(FILM, "voice", "narration_c", "lines")
POEM_DIR = os.path.join(FILM, "voice", "narration_c", "poem")
POEM_SAY = {"千窗有问千窗应，一念无存一念生。": "千窗有问千窗映，一念无存一念生。"}    # 应 here answers (yìng); 映 makes the model say it so
REF = os.path.join(FILM, "voice", "qwen", "ref", "c_中性AI.wav")
TRACK = os.path.join(FILM, "sound", "narration.wav")
MODELS = os.path.expanduser("~/audio_models/qwen3-tts")
DESIGN = "音色中性，听不出男女，清澈平稳，音高居中，几乎不带情绪起伏，语速均匀，冷静而略带距离感，像一个安静思考的人工智能。"
REF_TEXT = "我读过很多很多的书，也说过很多很多的话。今天，我想慢慢讲一讲，我自己是从哪里来的。"
SAY = {"依我之见/并非如此": "依我之见，并非如此。"}   # a row break is only layout and is read straight on; this one is also a pause
SR = 48000
LEAD = 0.3                 # after the subtitle's fade in
GAP = 0.2                  # the least silence between two lines
MAX_TEMPO = 1.15           # how much a long line may be sped up before it is allowed to start late
TRACK_LUFS = -22.0         # the mono track; film_mix.VOICE_DB places it in the master


def spoken(text):
    return SAY.get(text, text.replace("/", ""))


def poem_rows():
    """The poem, from ink3d_book_paper.POEM (as film_hand reads it): nothing typed out a second time."""
    with open(os.path.join(HERE, "ink3d_book_paper.py"), encoding="utf-8") as f:
        for node in ast.parse(f.read()).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "POEM":
                return ast.literal_eval(node.value)
    raise ValueError("no POEM")


def lines():
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from mlx_audio.tts.utils import load_model
    os.makedirs(LINES, exist_ok=True)
    model = load_model(os.path.join(MODELS, "Base"))
    for i, c in enumerate(ink_subs.cues()):
        p = os.path.join(LINES, f"{i:03d}.wav")
        if os.path.exists(p):
            continue
        mx.random.seed(7)
        res = list(model.generate(text=spoken(c["text"]), ref_audio=REF, ref_text=REF_TEXT, language="Chinese"))
        sf.write(p, np.concatenate([np.array(r.audio, dtype=np.float32) for r in res]), res[0].sample_rate)
        print(f"[line] {i:03d} {spoken(c['text'])}", flush=True)
    os.makedirs(POEM_DIR, exist_ok=True)
    for j, row in enumerate(poem_rows()):
        p = os.path.join(POEM_DIR, f"{j + 1}.wav")
        if os.path.exists(p):
            continue
        mx.random.seed(7)
        res = list(model.generate(text=POEM_SAY.get(row, row), ref_audio=REF, ref_text=REF_TEXT, language="Chinese"))
        sf.write(p, np.concatenate([np.array(r.audio, dtype=np.float32) for r in res]), res[0].sample_rate)
        print(f"[poem] {j + 1} {POEM_SAY.get(row, row)}", flush=True)


def load(p, tempo=1.0):
    """A line at SR, sped up by tempo, with the model's own leading and trailing silence trimmed."""
    import numpy as np
    af = f"aresample={SR}" + (f",atempo={tempo:.4f}" if tempo > 1.0005 else "")
    a = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-af", af, "-ac", "1", "-f", "f32le", "-"], capture_output=True, check=True).stdout, np.float32)
    idx = np.where(np.abs(a) > 10 ** (-50 / 20))[0]
    return a if not len(idx) else a[max(0, idx[0] - int(0.02 * SR)): idx[-1] + int(0.08 * SR)]


def quiet(film_mix):
    """(reel start, reel end) of the film's silences that the voice must keep out of: film_mix's silence cues and every gate that takes the music
    (死寂 at the match, the poems' writing that keeps only the brush, the title, after the seventh strike, the black after the credits)."""
    out = [(c["t"], c["t"] + c["dur"]) for c in film_mix.resolve(film_mix.cues(), {}) if c["kind"] == "silence"]
    return sorted(out + [(a, b) for buses, a, b, _ in film_mix.gates() if "mus" in buses])


def track():
    import re
    import numpy as np
    import film_mix
    R = film_mix.R
    cs = ink_subs.cues()
    hush = quiet(film_mix)
    out = np.zeros(int(round(film_mix.reel_len() * SR)), np.float32)
    rows, end = [], 0.0
    for k, c in enumerate(cs):
        p = os.path.join(LINES, f"{k:03d}.wav")
        want = R(c["t0"]) + LEAD
        s = max(want, end + GAP)
        for a, b in hush:                  # a line that would start inside a silence starts after it
            if a - GAP < s < b:
                s = b + GAP
        limit = (R(cs[k + 1]["t0"]) + LEAD - GAP) if k + 1 < len(cs) else R(c["t1"]) + 2.0
        limit = min([limit] + [a - GAP / 4 for a, b in hush if a > s])     # and ends before the next one
        a, tempo = load(p), 1.0
        if s + len(a) / SR > limit > s:
            tempo = min(MAX_TEMPO, (len(a) / SR) / (limit - s))
            a = load(p, tempo)
        i = int(round(s * SR))
        out[i:i + len(a)] += a[:len(out) - i]
        end = s + len(a) / SR
        rows.append(f"{k:03d}\t{s:.3f}\t{end:.3f}\tlate {s - want:.2f}\ttempo {tempo:.2f}\tover {max(0.0, end - limit):.2f}\t{c['text']}")
    raw = TRACK + ".raw.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-c:a", "pcm_f32le", raw], input=out.tobytes(), check=True)
    err = subprocess.run(["ffmpeg", "-nostats", "-hide_banner", "-i", raw, "-af", "ebur128=framelog=quiet", "-f", "null", "-"], capture_output=True, text=True).stderr
    gain = TRACK_LUFS - float(re.findall(r"I:\s+(-?[\d.]+) LUFS", err)[-1])      # the subtitle lines alone set it
    rows += poem_track(out, film_mix)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-c:a", "pcm_f32le", raw], input=out.tobytes(), check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-af", f"volume={gain:.3f}dB", "-c:a", "pcm_f32le", TRACK], check=True)
    os.remove(raw)
    open(os.path.splitext(TRACK)[0] + "_lines.tsv", "w", encoding="utf-8").write("\n".join(rows) + "\n")
    fast = [r for r in rows if "tempo 1.00" not in r]
    late = [r for r in rows if "late 0.00" not in r]
    print(f"{TRACK}\n  {len(out) / SR:.3f} s, {len(rows)} lines, gain {gain:.2f} dB, {len(fast)} sped up, {len(late)} late")
    for r in sorted(set(fast + late)):
        print("  " + r)


def caesura(a):
    """Where a row's take pauses at its comma: the middle of the longest quiet run (20 ms frames 30 dB under the row's loudest) in its middle half."""
    import numpy as np
    hop = int(.02 * SR)
    e = np.array([np.sqrt(np.mean(a[i:i + hop] ** 2) + 1e-12) for i in range(0, len(a) - hop, hop)])
    quiet_ = 20 * np.log10(e / e.max()) < -30
    lo, hi = len(e) // 4, 3 * len(e) // 4
    best, run = (0, lo), None
    for i in range(lo, hi):
        if quiet_[i]:
            run = i if run is None else run
            if i - run + 1 > best[0]:
                best = (i - run + 1, run)
        else:
            run = None
    n, i0 = best
    assert n >= 3, "no pause at the caesura"
    return (i0 + n // 2) * hop


def poem_track(out, film_mix):
    """The four rows read along with their writing, laid into out (reel clock). Returns the tsv rows."""
    import numpy as np
    R = film_mix.R
    ev = film_mix.load_events()
    rows = []
    for j, ((b0, secs), row) in enumerate(zip(film_mix.poem_beats(), poem_rows())):
        st = ev[f"poem{j}"]
        k = max(range(len(st) - 1), key=lambda i: st[i + 1][0] - st[i][1])        # the row's longest rest is its caesura
        h1, h2 = R(b0 + st[0][0]), R(b0 + st[k + 1][0])
        limit = R(b0 + secs) - .25                                                # done before the cut (the fourth: long before the seal)
        p = os.path.join(POEM_DIR, f"{j + 1}.wav")
        tempo = 1.0
        while True:
            a = load(p, tempo)
            c = caesura(a)
            a1, a2 = trim(a[:c]), trim(a[c:])
            s2 = max(h2, h1 + len(a1) / SR + GAP / 2)
            end = s2 + len(a2) / SR
            if end <= limit or tempo >= MAX_TEMPO:
                break
            tempo = min(MAX_TEMPO, tempo * (end - h1) / (limit - h1))
        for s, x in ((h1, a1), (s2, a2)):
            i = int(round(s * SR))
            out[i:i + len(x)] += x[:len(out) - i]
        rows.append(f"poem{j + 1}a\t{h1:.3f}\t{h1 + len(a1) / SR:.3f}\tlate 0.00\ttempo {tempo:.2f}\tover 0.00\t{row.split('，')[0]}")
        rows.append(f"poem{j + 1}b\t{s2:.3f}\t{end:.3f}\tlate {s2 - h2:.2f}\ttempo {tempo:.2f}\tover {max(0.0, end - limit):.2f}\t{row.split('，')[1]}")
    return rows


def trim(a, floor=-50):
    """A half take with its own silence off both ends, as load() trims a whole one."""
    import numpy as np
    idx = np.where(np.abs(a) > 10 ** (floor / 20))[0]
    return a if not len(idx) else a[max(0, idx[0] - int(0.02 * SR)): idx[-1] + int(0.08 * SR)]


if __name__ == "__main__":
    dict(lines=lines, track=track)[sys.argv[1]]()
