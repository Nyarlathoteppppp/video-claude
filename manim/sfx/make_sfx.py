"""Synthesise the sound effects used by the public videos (numpy + scipy only, run with the manim venv).

Every sound is generated from sine waves and seeded noise in this file, so the results carry no third-party
licence. Output: 48 kHz mono 16-bit WAVs next to this script, peaks normalised to -3 dBFS (mix with the
`gain=` argument of Scene.add_sound; the balanced values are written to gains.json).

    manim/.venv/bin/python manim/sfx/make_sfx.py            # writes *.wav + preview.wav (all sounds in a row)
"""
import os
import wave

import numpy as np
from scipy import signal

SR = 48000
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(90054)


def t_axis(dur):
    return np.arange(int(dur * SR)) / SR


def env_exp(dur, tau, attack=0.002):
    t = t_axis(dur)
    a = np.clip(t / attack, 0, 1) if attack > 0 else 1.0
    return a * np.exp(-t / tau)


def bandpass(x, lo, hi, order=4):
    sos = signal.butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def lowpass(x, fc, order=4):
    return signal.sosfilt(signal.butter(order, fc, fs=SR, output="sos"), x)


def highpass(x, fc, order=4):
    return signal.sosfilt(signal.butter(order, fc, btype="high", fs=SR, output="sos"), x)


def glide(f0, f1, dur, curve="exp"):
    """Sine whose frequency moves from f0 to f1 (phase-continuous)."""
    t = t_axis(dur)
    if curve == "exp":
        f = f0 * (f1 / f0) ** (t / dur)
    else:
        f = f0 + (f1 - f0) * t / dur
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def saturate(x, drive=3.0):
    """Soft clipping adds harmonics, so a sub-bass boom is still heard on phone / laptop speakers."""
    peak = np.max(np.abs(x))
    return np.zeros_like(x) if peak == 0 else np.tanh(drive * x / peak) / np.tanh(drive)


def reverb(x, tail=0.6, mix=0.18, bright=5000):
    """Cheap room: convolve with exponentially decaying filtered noise."""
    ir = rng.standard_normal(int(tail * SR)) * np.exp(-t_axis(tail) / (tail / 6))
    ir = lowpass(ir, bright)
    ir /= np.sqrt(np.sum(ir ** 2))
    x = fade_out(x, 0.01)                       # no hard stop of the dry signal inside the tail
    wet = signal.fftconvolve(x, ir)[: len(x) + len(ir)]
    dry = np.concatenate([x, np.zeros(len(wet) - len(x))])
    return (1 - mix) * dry + mix * wet


def fade_out(x, dur=0.01):
    n = min(len(x), int(dur * SR))
    x = x.copy()
    if n > 0:
        x[-n:] *= np.linspace(1, 0, n)
    return x


def place(total, items):
    """Mix (start_seconds, clip) pairs into one buffer of `total` seconds."""
    out = np.zeros(int(total * SR))
    for t0, clip in items:
        i = int(t0 * SR)
        assert 0 <= i < len(out), (t0, total)
        j = min(len(out), i + len(clip))
        out[i:j] += clip[: j - i]
    return out


# ---------------------------------------------------------------- the sounds

def stone():
    """Go stone on a wooden board: sharp click + short stone ring + wooden body."""
    dur = 0.35
    click = bandpass(rng.standard_normal(int(0.004 * SR)), 1200, 7000) * np.hanning(int(0.004 * SR))
    ring = sum(a * np.sin(2 * np.pi * f * t_axis(dur)) * env_exp(dur, tau, 0.0005)
               for f, a, tau in [(2350, 0.35, 0.018), (3720, 0.22, 0.012), (5150, 0.12, 0.008)])
    body = sum(a * np.sin(2 * np.pi * f * t_axis(dur)) * env_exp(dur, tau, 0.001)
               for f, a, tau in [(190, 1.0, 0.05), (410, 0.6, 0.035), (870, 0.35, 0.022), (1450, 0.2, 0.015)])
    x = place(dur, [(0, click * 0.9), (0, ring), (0.001, body)])
    return reverb(x, tail=0.35, mix=0.12)


def stone_big():
    """The move-37 slam: the stone plus a low boom and a longer room."""
    dur = 1.6
    boom = glide(90, 38, dur) * env_exp(dur, 0.32, 0.004) * 0.9
    thump = lowpass(rng.standard_normal(int(dur * SR)), 260) * env_exp(dur, 0.06, 0.001) * 2.5
    low = saturate(boom + thump, 4.0) * 0.8
    x = place(dur, [(0, stone() * 1.2), (0, low)])
    return reverb(x, tail=1.4, mix=0.22, bright=3500)


def tick():
    """Small UI tick for counters and single points."""
    dur = 0.06
    t = t_axis(dur)
    return (np.sin(2 * np.pi * 1850 * t) * 0.7 + np.sin(2 * np.pi * 3700 * t) * 0.3) * env_exp(dur, 0.009, 0.0008)


def pop():
    """Soft bubble pop for a new branch / node / mini board appearing."""
    dur = 0.12
    return glide(380, 980, dur) * env_exp(dur, 0.035, 0.004) * np.hanning(int(dur * SR)) ** 0.3


def card():
    """One playing card snapped onto a table."""
    dur = 0.09
    n = int(dur * SR)
    snap = bandpass(rng.standard_normal(n), 1800, 7000) * env_exp(dur, 0.012, 0.0005)
    slap = lowpass(rng.standard_normal(n), 900) * env_exp(dur, 0.02, 0.001) * 0.6
    return snap + slap


def shuffle():
    """Cards dealt in a quick run (about 0.7 s)."""
    times = np.cumsum([0] + list(rng.uniform(0.045, 0.085, 9)))
    return place(times[-1] + 0.12, [(t0, card() * rng.uniform(0.6, 1.0)) for t0 in times])


def rain():
    """Random points landing: sparse ticks that get denser (2.5 s), very quiet in the mix."""
    dur = 2.5
    items = []
    t0 = 0.0
    while t0 < dur - 0.08:
        gap = 0.16 * (1 - t0 / dur) ** 2 + 0.012
        f = rng.uniform(2200, 4200)
        d = 0.03
        items.append((t0, np.sin(2 * np.pi * f * t_axis(d)) * env_exp(d, 0.006, 0.0005) * rng.uniform(0.35, 0.8)))
        t0 += rng.exponential(gap)
    x = place(dur, items)
    return x * np.clip(np.linspace(0.6, 1.0, len(x)), 0, 1)


def whoosh(dur=0.55, f0=250, f1=4000):
    """Transition swish: noise with a band sweeping up, swelling in and out (STFT mask)."""
    n = int(dur * SR)
    noise = rng.standard_normal(n + 2048)
    f, tt, Z = signal.stft(noise, SR, nperseg=1024)
    centre = f0 * (f1 / f0) ** np.clip(tt / dur, 0, 1)
    mask = np.exp(-0.5 * (np.log(np.maximum(f[:, None], 1) / centre[None, :]) / 0.45) ** 2)
    _, y = signal.istft(Z * mask, SR, nperseg=1024)
    y = y[:n]
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.6
    return y * env


def whoosh_soft():
    """Quieter, lower whoosh for slides between beats."""
    return lowpass(whoosh(0.7, 180, 1800), 2500)


def bell(freq, dur=1.6, tau=0.55):
    """Bell-ish tone: fundamental + slightly inharmonic partials."""
    t = t_axis(dur)
    parts = [(1.0, 1.0, tau), (2.0, 0.35, tau * 0.6), (2.76, 0.18, tau * 0.4), (5.4, 0.06, tau * 0.25)]
    return fade_out(sum(a * np.sin(2 * np.pi * freq * r * t) * env_exp(dur, tt, 0.003) for r, a, tt in parts), 0.04)


def chime():
    """The 'aha' moment: a soft major-sixth bell dyad with room."""
    x = place(1.8, [(0, bell(784)), (0.06, bell(1318.5) * 0.6)])
    return reverb(x, tail=1.2, mix=0.25)


def success():
    """Rising three-note arpeggio (pentatonic) for the 79 % reveal."""
    notes = [659.3, 784.0, 1046.5]
    x = place(2.0, [(i * 0.11, bell(f, 1.6, 0.5) * (0.8 + 0.1 * i)) for i, f in enumerate(notes)])
    return reverb(x, tail=1.2, mix=0.25)


def question():
    """'Is that a mistake?': two short muted tones, the second higher (a raised eyebrow)."""
    d = 0.22
    a = fade_out(np.sin(2 * np.pi * 330 * t_axis(d)) * env_exp(d, 0.08, 0.01), 0.03)
    b = fade_out(np.sin(2 * np.pi * 494 * t_axis(d)) * env_exp(d, 0.1, 0.01), 0.03)
    return reverb(place(0.6, [(0, a), (0.18, b)]), tail=0.5, mix=0.2)


def boom():
    """Deep impact for the 10^360 vs 10^80 reveal."""
    dur = 2.2
    x = glide(70, 30, dur) * env_exp(dur, 0.6, 0.01)
    x += lowpass(rng.standard_normal(int(dur * SR)), 180) * env_exp(dur, 0.12, 0.002) * 2.0
    hit = bandpass(rng.standard_normal(int(dur * SR)), 150, 900) * env_exp(dur, 0.06, 0.001) * 2.2
    body = glide(150, 70, dur) * env_exp(dur, 0.35, 0.01) * 0.35        # octave layer that phone speakers can play
    x = saturate(x, 4.0) * 0.45 + hit + body
    return reverb(x, tail=1.6, mix=0.25, bright=1500)


def riser(dur=1.8):
    """Tension riser for fast-forward: rising noise band + rising tone, ends abruptly."""
    n = int(dur * SR)
    tone = glide(220, 880, dur) * 0.25
    x = whoosh(dur, 300, 6000)[:n]
    env = (np.linspace(0, 1, n) ** 2.2)
    return fade_out((x + tone) * env, 0.02)


def dice():
    """Two dice rolled on a table (clacks getting closer together), for the Monte Carlo naming."""
    times, t0, gap = [], 0.0, 0.11
    while gap > 0.018 and t0 < 0.9:
        times.append(t0)
        t0 += gap
        gap *= 0.8

    def clack():
        d = 0.05
        f = rng.uniform(1500, 2600)
        return (np.sin(2 * np.pi * f * t_axis(d)) * 0.6 + bandpass(rng.standard_normal(int(d * SR)), 2500, 8000) * 0.4) \
            * env_exp(d, 0.008, 0.0005)

    return reverb(place(t0 + 0.1, [(tt, clack() * rng.uniform(0.5, 1.0)) for tt in times]), tail=0.4, mix=0.15)


def ambient(dur=32.0):
    """Optional quiet pad bed, seamlessly loopable: every frequency has a whole number of cycles in `dur`."""
    t = t_axis(dur)
    chord = [110.0, 164.8125, 220.0, 277.1875, 329.625]          # A2 E3 A3 C#4 E4, snapped to 1/32 Hz
    x = np.zeros_like(t)
    for i, f in enumerate(chord):
        for det in (-0.09375, 0.09375):                           # ±3/32 Hz detune → slow beating, still loops
            lfo = 0.6 + 0.4 * np.sin(2 * np.pi * (i + 1) / dur * t + i)
            x += np.sin(2 * np.pi * (f + det) * t + rng.uniform(0, 2 * np.pi)) * lfo / (1 + 0.6 * i)
    return lowpass(np.tile(x, 2), 1400)[len(t):]                  # filter on two copies, keep the steady-state second one


def write_bed(total, path, fade_in=2.5, fade_out_s=4.0):
    """Ambient bed of exactly `total` seconds (the 32 s loop tiled, faded in and out), peak -3 dBFS.

    Call at the end of construct() and add it at t = 0:
        write_bed(self.time, bed); self.add_sound(bed, time_offset=-self.time, gain=SFX_GAIN["ambient"])
    """
    loop = ambient()
    n = int(total * SR)
    x = np.tile(loop, n // len(loop) + 1)[:n]
    ni, no = min(n, int(fade_in * SR)), min(n, int(fade_out_s * SR))
    x[:ni] *= np.sin(np.linspace(0, np.pi / 2, ni)) ** 2
    x[n - no:] *= np.cos(np.linspace(0, np.pi / 2, no)) ** 2
    ref = np.max(np.abs(loop - np.mean(loop)))                    # same scale as ambient.wav, so gains.json applies
    return write_wav(path, x, scale_ref=ref, loop=True)


SOUNDS = {
    "stone": stone, "stone_big": stone_big, "tick": tick, "pop": pop, "card": card, "shuffle": shuffle,
    "rain": rain, "whoosh": whoosh, "whoosh_soft": whoosh_soft, "chime": chime, "success": success,
    "question": question, "boom": boom, "riser": riser, "dice": dice, "ambient": ambient,
}

# Loudness tiers: the loudest 50 ms of each sound should land here after its gain (dBFS). Accents (aha / slam /
# boom) sit highest, ordinary events below, tiny UI sounds lower, beds lowest. gains.json holds the resulting
# Scene.add_sound(gain=...) values.
# Kept deliberately low: the effects support the picture and captions, they must not take the foreground.
TARGET = {
    "stone_big": -21, "boom": -20, "success": -23, "chime": -24,
    "stone": -27, "question": -28, "whoosh": -30, "riser": -29, "dice": -30, "shuffle": -30,
    "pop": -33, "card": -33, "tick": -35,
    "rain": -37, "whoosh_soft": -35, "ambient": -41,
}


def short_term_db(x, win=0.05):
    n = int(win * SR)
    if len(x) <= n:
        return 20 * np.log10(np.sqrt(np.mean(x ** 2)))
    return 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(n) / n, "valid").max()))


def write_wav(path, x, peak_db=-3.0, loop=False, normalise=True, scale_ref=None):
    """loop=True: no edge fades (the file is tiled back to back). normalise=False: write the samples as they are."""
    x = np.asarray(x, dtype=np.float64)
    assert len(x) > 0 and np.all(np.isfinite(x)), path
    if normalise:
        x = x - np.mean(x)
    if not loop:
        n = int(0.0005 * SR)                                      # 0.5 ms taper: no step from silence, transients kept
        x[:n] *= np.linspace(0, 1, n)
        x = fade_out(x, 0.005)
    if normalise:
        peak = scale_ref if scale_ref is not None else np.max(np.abs(x))
        assert peak > 0, path
        x = x / peak * 10 ** (peak_db / 20)
    assert np.max(np.abs(x)) <= 1.0, path
    pcm = np.round(x * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return x


def main():
    import json
    preview, gains = [], {}
    for name, fn in SOUNDS.items():
        x = write_wav(os.path.join(HERE, name + ".wav"), fn(), loop=(name == "ambient"))
        st = short_term_db(x)
        gains[name] = round(TARGET[name] - st, 1)
        spec = np.abs(np.fft.rfft(x))
        centroid = (spec * np.fft.rfftfreq(len(x), 1 / SR)).sum() / spec.sum()
        phone = short_term_db(bandpass(x, 250, 8000)) - st          # level left on a phone speaker (rough)
        print(f"{name:12s} {len(x) / SR:5.2f}s  loudest-50ms {st:6.1f} dBFS  gain {gains[name]:+6.1f} dB"
              f"  centroid {centroid:6.0f} Hz  phone-band {phone:+5.1f} dB")
        if name != "ambient":
            preview += [x * 10 ** (gains[name] / 20), np.zeros(int(0.45 * SR))]
    with open(os.path.join(HERE, "gains.json"), "w") as fh:
        json.dump(gains, fh, indent=1)
    # preview.wav: every effect in SOUNDS order at its real mix level (no normalisation)
    write_wav(os.path.join(HERE, "preview.wav"), np.concatenate(preview), normalise=False)


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["bed"]:                                  # make_sfx.py bed <seconds> <out.wav>
        write_bed(float(sys.argv[2]), sys.argv[3])
    else:
        main()
