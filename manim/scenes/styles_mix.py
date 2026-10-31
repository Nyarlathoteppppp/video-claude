"""Lays each take from styles_score.py under the reel: measures its real beat, stretches it so 8 beats last one cut (4.6 s),
slides it so its biggest entrance lands as the wall opens onto the first clip (5.57 s), masters it to -18 LUFS and muxes it.

    PY=python3        # numpy, scipy, ffmpeg on PATH
    $PY styles_mix.py        # styles/film_out/bgm/take*.flac + reel.mp4 -> styles/film_out/reel_takeN.mp4
"""
import glob
import json
import os
import re
import subprocess

import numpy as np

from styles_common import OUT

SR = 22050
HOP = 256
CUT = 138 / 30          # the reel's clips start this far apart
BEAT = CUT / 8
LAND = 167 / 30         # the push into the first window lands here
LENGTH = 2063 / 30
FADE = 2.5
LUFS = -18              # -14 was too loud for music with nothing over it
NUDGE = 0.03            # the beat finder lands a little before the kick; on take 2 every cut measured about 40 ms early


def pcm(path, *filters):
    cmd = ["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR)]
    if filters:
        cmd += ["-af", ",".join(filters)]
    raw = subprocess.run(cmd + ["-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def flux(x):
    """Onset strength: how much new energy each hop brings, per log-spaced band."""
    n = 1024
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::HOP] * np.hanning(n)
    mag = np.log1p(40 * np.abs(np.fft.rfft(frames, axis=1)))
    return np.clip(np.diff(mag, axis=0), 0, None).mean(axis=1)


def grid(env, lo=96.0, hi=112.0):
    """The steadiest beat near the tempo asked for: (seconds per beat, phase in seconds, how much the onsets prefer that grid)."""
    rate = SR / HOP
    env = env - env.mean()
    best = (0.0, 0.0, 0.0)
    for bpm in np.arange(lo, hi, 0.02):
        period = 60 / bpm * rate
        steps = int(len(env) / period)
        at = np.arange(steps) * period
        scores = [env[np.minimum((at + p).astype(int), len(env) - 1)].mean() for p in np.arange(0, period, 2)]
        k = int(np.argmax(scores))
        if scores[k] > best[0]:
            best = (scores[k], 60 / bpm, k * 2 / rate)
    return best[1], best[2], best[0] / (env.std() + 1e-9)


def entrance(x, phase):
    """The bar line after which the music gets loudest, in seconds into the stretched take; it must leave LAND seconds before it and the rest of the reel after."""
    bars = np.arange(phase + 4 * BEAT * np.ceil((LAND - phase) / (4 * BEAT)), len(x) / SR - LENGTH + LAND, 4 * BEAT)
    def level(a, b):
        part = x[int(a * SR):int(b * SR)]
        return 20 * np.log10(np.sqrt((part ** 2).mean()) + 1e-6)
    jumps = [level(b, b + 8 * BEAT) - level(max(0, b - 8 * BEAT), b) for b in bars]
    k = int(np.argmax(jumps))
    return float(bars[k]), float(jumps[k])


def bar_phase(x, phase):
    """Which of the four beats is the one: the one with the most low end arriving on it."""
    low = pcm_low(x)
    env = flux(low)
    rate = SR / HOP
    scores = []
    for b in range(4):
        at = (phase + b * BEAT + np.arange(int(len(env) / rate / (4 * BEAT))) * 4 * BEAT) * rate
        scores.append(env[np.minimum(at.astype(int), len(env) - 1)].mean())
    return phase + int(np.argmax(scores)) * BEAT


def pcm_low(x):
    spectrum = np.fft.rfft(x)
    spectrum[np.fft.rfftfreq(len(x), 1 / SR) > 150] = 0
    return np.fft.irfft(spectrum, len(x)).astype(np.float32)


def loudness(path):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", path, "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return json.loads(re.search(r"\{[^{}]+\}", err, re.S).group(0))


def main():
    reel = os.path.join(OUT, "reel.mp4")
    for take in sorted(glob.glob(os.path.join(OUT, "bgm", "take*.flac"))):
        name = os.path.basename(take)[:-5]
        beat, _, sharp = grid(flux(pcm(take)))
        tempo = beat / BEAT                                   # atempo > 1 speeds the take up
        stretch = f"atempo={tempo:.5f}"
        x = pcm(take, stretch)
        _, phase, _ = grid(flux(x), 60 / BEAT - 0.05, 60 / BEAT + 0.05)
        phase = bar_phase(x, phase)
        drop, jump = entrance(x, phase)
        start = drop - LAND + NUDGE
        end = start + LENGTH
        if end > len(x) / SR:
            print(name, f"too short: needs {end:.1f} s, has {len(x) / SR:.1f} s", flush=True)
            continue
        chain = (f"{stretch},atrim=start={start:.4f}:end={end:.4f},asetpts=PTS-STARTPTS,"
                 f"afade=t=in:d=0.03,afade=t=out:st={LENGTH - FADE:.4f}:d={FADE}")
        m = loudness_of_chain(take, chain)
        master = (f"{chain},loudnorm=I={LUFS}:TP=-1.5:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
                  f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true,aresample=48000")
        dest = os.path.join(OUT, f"reel_{name}.mp4")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", reel, "-i", take, "-filter_complex", f"[1:a]{master}[a]",
                        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", "-shortest", dest], check=True)
        got = loudness(dest)
        print(f"{name}: {60 / beat:.2f} BPM measured, stretched x{tempo:.4f}, beat sharpness {sharp:.2f}, entrance +{jump:.1f} dB at {drop:.2f} s, "
              f"start {start:.2f} s, {float(got['input_i']):.1f} LUFS, peak {float(got['input_tp']):.1f} dB -> {dest}", flush=True)


def loudness_of_chain(take, chain):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", take, "-af", f"{chain},loudnorm=I={LUFS}:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return json.loads(re.search(r"\{[^{}]+\}", err, re.S).group(0))


if __name__ == "__main__":
    main()
