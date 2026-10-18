"""Beat, bar and section map of a music track, for cutting a Manim video to its BGM (numpy + scipy + ffmpeg only).

    python analyze_bgm.py song.mp3 [--bpm-hint 90] [--beats-per-bar 4] [--start 0] [--end 0]

Writes next to the audio:
  <name>.beats.csv     every beat: t, bar, beat_in_bar (1 = downbeat), strength (onset, 0-1), loud_db (beat RMS)
  <name>.sections.csv  section boundaries (on downbeats): t_start, t_end, bars, loud_db (mean), label (quiet / mid / loud)
and prints the tempo, the bar grid and an energy strip per bar, so a section can be picked by eye.

Method: log-mel spectral flux onset envelope -> tempo from its autocorrelation (log-normal prior around the hint)
-> dynamic-programming beat tracker (Ellis 2007) -> downbeat phase = the beat-in-bar position with the strongest
low-band onsets -> section boundaries = peaks of a checkerboard novelty curve on the bar-level self-similarity matrix.
Times are in seconds from the start of the file (or from --start if given), on a 512-sample hop at 22050 Hz (23 ms).
"""
import argparse
import csv
import os
import subprocess

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, get_window

SR = 22050
N_FFT = 2048
HOP = 512


def load(path, start=0.0, end=0.0):
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(start)]
    if end > start:
        cmd += ["-to", str(end)]
    cmd += ["-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def mel_filterbank(n_mels=64, fmin=30.0, fmax=8000.0):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, freqs.size))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb, pts[1:-1]


def spectrogram(y):
    win = get_window("hann", N_FFT)
    pad = np.pad(y, N_FFT // 2)
    n = 1 + (pad.size - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(pad, (n, N_FFT), (pad.strides[0] * HOP, pad.strides[0]))
    return np.abs(np.fft.rfft(frames * win, axis=1))  # (frames, bins)


def onset_envelope(logmel, band=None):
    m = logmel if band is None else logmel[:, band]
    flux = np.maximum(np.diff(m, axis=0, prepend=m[:1]), 0).mean(axis=1)
    flux -= uniform_filter1d(flux, 16)  # remove slow loudness drift
    flux = np.maximum(flux, 0)
    return flux / (flux.max() + 1e-9)


def estimate_tempo(env, hint):
    fps = SR / HOP
    ac = np.correlate(env - env.mean(), env - env.mean(), "full")[env.size - 1:]
    lags = np.arange(ac.size)
    bpm = np.where(lags > 0, 60 * fps / np.maximum(lags, 1), 0)
    ok = (bpm >= 50) & (bpm <= 200)
    prior = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1e-9) / hint) / 0.9) ** 2)
    score = np.where(ok, ac * prior, -np.inf)
    lag = int(np.argmax(score))
    # parabolic refinement of the peak for sub-frame period
    if 1 <= lag < ac.size - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        lag = lag + 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    return 60 * fps / lag, lag


def track_beats(env, period, tightness=100.0):
    """Ellis (2007) dynamic programming: maximise onset strength at beats + penalty on deviating from the period."""
    n = env.size
    score = env.copy()
    back = -np.ones(n, dtype=int)
    lo, hi = int(round(period / 2)), int(round(period * 2))
    offs = np.arange(-hi, -lo + 1)
    pen = -tightness * np.log(-offs / period) ** 2
    for t in range(hi, n):
        cand = score[t + offs] + pen
        k = int(np.argmax(cand))
        score[t] = env[t] + cand[k]
        back[t] = t + offs[k]
    # end on the best-scoring frame in the last period
    t = int(np.argmax(score[n - hi:])) + n - hi
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    beats = np.array(beats[::-1])
    # trim weak beats at the edges (silence before the music starts / after it ends)
    thr = 0.1 * np.median(env[beats])
    keep = np.where(uniform_filter1d(env[beats], 3) > thr)[0]
    return beats[keep[0]:keep[-1] + 1] if keep.size else beats


def checkerboard_novelty(feat, half):
    # z-score each dimension across bars, then a Gaussian similarity on Euclidean distance: unlike cosine
    # similarity this also sees a pure loudness step (same instruments, louder), common between verse and chorus
    f = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-6)
    d2 = ((f[:, None, :] - f[None, :, :]) ** 2).sum(axis=2)
    ssm = np.exp(-d2 / (np.median(d2[d2 > 0]) + 1e-9))
    k = np.ones((2 * half, 2 * half))
    k[:half, half:] = k[half:, :half] = -1
    g = np.exp(-0.5 * (np.linspace(-2, 2, 2 * half)) ** 2)
    k *= np.outer(g, g)
    pad = np.pad(ssm, half, mode="edge")
    return np.array([(pad[i:i + 2 * half, i:i + 2 * half] * k).sum() for i in range(len(f))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--bpm-hint", type=float, default=100.0)
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--section-bars", type=int, default=4, help="half-width of the novelty kernel, in bars")
    a = ap.parse_args()

    y = load(a.audio, a.start, a.end)
    S = spectrogram(y)
    fb, centres = mel_filterbank()
    logmel = np.log1p(100 * (S ** 2) @ fb.T)
    env = onset_envelope(logmel)
    low = onset_envelope(logmel, centres < 200)  # kick / bass: where the downbeat usually is
    fps = SR / HOP

    bpm, period = estimate_tempo(env, a.bpm_hint)
    beats = track_beats(env, period)
    bt = beats / fps
    rms = np.sqrt(uniform_filter1d(y ** 2, HOP)[::HOP][: env.size] + 1e-12)
    loud = 20 * np.log10(np.array([rms[b:b + int(period)].mean() for b in beats]) + 1e-9)

    bpb = a.beats_per_bar
    phase = int(np.argmax([low[beats[p::bpb]].mean() + 0.5 * env[beats[p::bpb]].mean() for p in range(bpb)]))
    pos = (np.arange(beats.size) - phase) % bpb + 1           # 1 = downbeat
    bar = (np.arange(beats.size) - phase) // bpb + 1          # bar 0 = pickup beats before the first downbeat

    # bar-level features: mean log-mel + loudness, then novelty peaks on downbeats
    down = np.where(pos == 1)[0]
    bars_t = bt[down]
    edges = list(beats[down]) + [env.size]
    bar_loud = np.array([loud[i:i + bpb].mean() for i in down])
    feat = np.array([np.r_[logmel[s:e].mean(axis=0), logmel[s:e].std(axis=0)] for s, e in zip(edges[:-1], edges[1:])])
    feat = np.c_[feat, np.repeat(bar_loud[:, None], 16, axis=1) / 3.0]  # loudness weighted like ~16 mel bands
    nov = checkerboard_novelty(feat, a.section_bars)
    nov = np.maximum(nov, 0) / (nov.max() + 1e-9)
    pk, _ = find_peaks(nov, height=0.25, distance=a.section_bars)
    bounds = [0] + [int(p) for p in pk if 0 < p < len(down)] + [len(down)]

    base = os.path.splitext(a.audio)[0]
    with open(base + ".beats.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "bar", "beat_in_bar", "strength", "loud_db"])
        for t, b, p, s, l in zip(bt, bar, pos, env[beats], loud):
            w.writerow([f"{t + a.start:.3f}", b, p, f"{s:.3f}", f"{l:.1f}"])

    lo_q, hi_q = np.percentile(bar_loud, [33, 67])
    rows = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        t0 = bars_t[s]
        t1 = bars_t[e] if e < len(down) else bt[-1] + period / fps
        L = bar_loud[s:e].mean()
        label = "quiet" if L < lo_q else "loud" if L > hi_q else "mid"
        rows.append((t0 + a.start, t1 + a.start, e - s, L, label))
    with open(base + ".sections.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_start", "t_end", "bars", "loud_db", "label"])
        for r in rows:
            w.writerow([f"{r[0]:.3f}", f"{r[1]:.3f}", r[2], f"{r[3]:.1f}", r[4]])

    dur = y.size / SR
    print(f"{os.path.basename(a.audio)}: {dur:.1f} s, tempo {bpm:.1f} BPM (beat {60 / bpm:.3f} s, "
          f"bar {bpb * 60 / bpm:.2f} s), {beats.size} beats, {len(down)} bars, first downbeat {bars_t[0] + a.start:.2f} s")
    print("sections:")
    for t0, t1, n, L, label in rows:
        print(f"  {t0:7.2f} – {t1:7.2f} s  {n:3d} bars  {L:6.1f} dB  {label}")
    # one character per bar: loudness relative to the track (" .:-=+*#%@")
    ramp = " .:-=+*#%@"
    q = np.clip((bar_loud - bar_loud.min()) / (np.ptp(bar_loud) + 1e-9) * (len(ramp) - 1), 0, len(ramp) - 1).astype(int)
    strip = "".join(ramp[i] for i in q)
    print("energy per bar (| = section boundary):")
    marks = set(bounds[1:-1])
    print("  " + "".join(("|" if i in marks else "") + c for i, c in enumerate(strip)))
    print(f"wrote {base}.beats.csv and {base}.sections.csv")


if __name__ == "__main__":
    main()
