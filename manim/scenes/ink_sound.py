"""The ink film's sound: a placeholder track, and the cue sheet the real sound and music are made from.

The track is synthesised and only has to carry the timing; the real one comes from the M5 Max, working from the cue sheet.
Every time is worked out from the storyboard by row name, so both follow ink.scenes.md when it moves.

    python ink_sound.py 213.3               # ink/film_out/ink_audio.wav (48 kHz stereo), mastered, a level report, and the cue sheet
    python ink_sound.py 213.3 --cues-only   # only ink/film_out/ink_cues.json

Mastering uses the chain from ../sfx/master_audio.py: gentle compression, make-up gain to -24 LUFS, limiter at -1.5 dBTP.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
from scipy.signal import fftconvolve, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sfx"))
from master_audio import COMP, measure  # noqa: E402

from ink_attn import CHANT, LANDS, SWEEPS  # noqa: E402
from ink_film import CAM, GLYPH_HITS, OUT, QT0, RIFT_CLOSE, RIFT_OPEN, ROW, beats, rows  # noqa: E402
from ink_qian import A_TYPE, ANSWER, Q_OFF_FIRST, Q_OFF_LAST, Q_OFF_SINGLES, Q_SEND, Q_TYPED, off_times  # noqa: E402

SR = 48000
HOPS = [(t0, t1) for (t0, _, _), (t1, _, whip) in zip(CAM, CAM[1:]) if whip]                  # the camera's whips along the scroll
SCALE = [146.83, 174.61, 196.0, 220.0, 261.63, 293.66, 349.23, 392.0, 440.0]                 # D yu


def env(n, pts):
    """An envelope over n samples from (seconds, level) points."""
    return np.interp(np.arange(n) / SR, [p[0] for p in pts], [p[1] for p in pts])


def peak1(x):
    return x / (np.abs(x).max() + 1e-9)


# ---- instruments ---------------------------------------------------------------------------------------------------
def bell(f0, dur=12.0, seed=0):
    """A great bronze bell: inharmonic partials, the low ones ringing longest, each one beating slowly against itself."""
    r = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    x = np.zeros_like(t)
    for ratio, amp, life in ((.5, 1.0, 1.0), (1.0, .8, .72), (1.19, .45, .58), (1.5, .25, .46), (2.0, .4, .36), (2.51, .18, .28), (2.66, .12, .24), (3.01, .14, .2), (4.07, .08, .14), (5.4, .04, .09)):
        f = f0 * ratio * (1 + r.normal(0, .0009))
        ph = r.uniform(0, 6.28, 2)
        x += amp * (np.sin(2 * np.pi * f * t + ph[0]) + .6 * np.sin(2 * np.pi * f * 1.0017 * t + ph[1])) * np.exp(-t / (life * dur * .36))
    strike = r.normal(size=int(.04 * SR)) * np.exp(-np.arange(int(.04 * SR)) / (SR * .008))
    x[:len(strike)] += strike * .35
    return peak1(x) * np.minimum(t / .002, 1)


def drum(f0, dur=2.0, seed=0):
    """A big drum: the skin's pitch falls as it rings, a second mode above it, and the slap of the stick."""
    r = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    ph = 2 * np.pi * np.cumsum(f0 * (1 + 1.5 * np.exp(-t * 34))) / SR
    x = np.sin(ph) * np.exp(-t * 3.6 / dur) + .35 * np.sin(1.59 * ph) * np.exp(-t * 14 / dur)
    slap = r.normal(size=len(t)) * np.exp(-t * 60)
    return peak1(x + .22 * lowpass(slap, 1400))


def boom(dur=7.0, seed=0, low=26.0):
    """Thunder underneath everything: a sub that sinks, rumble that takes its time to go, and a crack at the start."""
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    sub = np.sin(2 * np.pi * np.cumsum(low * (1 + 1.2 * np.exp(-t * 3))) / SR) * np.exp(-t * 4.2 / dur)
    rumble = peak1(lowpass(r.normal(size=n), 170)) * np.exp(-t * 4.0 / dur)
    crack = peak1(band(r.normal(size=n), 300, 4200)) * np.exp(-t * 9)
    return peak1(sub + .75 * rumble + .55 * crack)


def pluck(f, dur=5.0, seed=0, bright=.5):
    """A qin string (Karplus-Strong)."""
    r = np.random.default_rng(seed)
    N = int(round(SR / f))
    n = int(dur * SR)
    burst = np.zeros(n)
    burst[:N] = np.convolve(r.uniform(-1, 1, N), np.ones(3) / 3, "same")
    g = (.996 + .0035 * bright) * .5
    a = np.zeros(N + 2)
    a[0], a[N], a[N + 1] = 1, -g, -g
    x = lfilter([1.0], a, burst)
    return peak1(x) * np.minimum(np.arange(n) / (SR * .003), 1)


def ping(f, d=1.4):
    x = np.arange(int(d * SR)) / SR
    return np.sin(2 * np.pi * f * x) * np.exp(-x * 4.5) * np.minimum(x / .002, 1)


def lowpass(x, fc):
    S = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(S / np.sqrt(1 + (f / fc) ** 4), len(x))


def band(x, lo, hi):
    S = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(S * np.exp(-(np.log(np.maximum(f, 1) / math.sqrt(lo * hi)) / (.5 * math.log(hi / lo) + .2)) ** 4), len(x))


def noise2ch(dur, lo, hi, seed):
    """Stereo band-limited noise, the two sides unrelated, so it fills the room."""
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    return np.stack([peak1(band(r.normal(size=n), lo, hi)), peak1(band(r.normal(size=n), lo, hi))], 1)


def moving_band(dur, path, q, seed, partials=((1.0, 1.0),)):
    """Noise through narrow bands whose centre follows path(t): wind in a gap, or air rising."""
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    N, hop = 4096, 1024
    win = np.hanning(N)
    x = r.normal(size=n + N)
    out = np.zeros(n + N)
    f = np.fft.rfftfreq(N, 1 / SR)
    for i in range(0, n, hop):
        f0 = path(i / SR)
        g = sum(a * np.exp(-((f - f0 * m) / (q * f0 * m)) ** 2) for m, a in partials)
        out[i:i + N] += np.fft.irfft(np.fft.rfft(x[i:i + N] * win) * g, N) * win
    return peak1(out[:n])


def impulse_rain(dur, rate, burst_ms, lo, hi, seed):
    """Countless small hits: rate(t) per second, each a tiny burst of noise."""
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    imp = (r.random(n) < rate(np.arange(n) / SR) / SR) * r.uniform(.3, 1, n)
    m = int(burst_ms / 1000 * SR)
    b = r.normal(size=m) * np.exp(-np.arange(m) / (m * .25))
    return peak1(band(fftconvolve(imp, b)[:n], lo, hi))


def ir(dur, rt60, seed, dark=5000.0):
    """A hall's tail, the two ears decorrelated."""
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    d = np.exp(-6.9 * t / rt60) * np.minimum(t / .015, 1)
    out = np.stack([lowpass(r.normal(size=n), dark) * d, lowpass(r.normal(size=n), dark) * d], 1)
    return out / np.sqrt((out ** 2).sum(0))


# ---- the mix --------------------------------------------------------------------------------------------------------
class Mix:
    def __init__(self, dur):
        self.n = int(dur * SR)
        self.dry, self.hall, self.room, self.after = [np.zeros((self.n, 2)) for _ in range(4)]

    def add(self, x, t, g=1.0, pan=0.0, hall=0.0, room=0.0, bus=None):
        i = int(t * SR)
        if i >= self.n or g == 0:
            return
        if x.ndim == 1:
            a = (pan + 1) * math.pi / 4
            x = np.stack([x * math.cos(a), x * math.sin(a)], 1) * 1.414
        if i < 0:
            x = x[-i:]
            i = 0
        e = min(i + len(x), self.n)
        x = x[:e - i] * g
        (self.after if bus == "after" else self.dry)[i:e] += x
        if hall:
            self.hall[i:e] += x * hall
        if room:
            self.room[i:e] += x * room

    def bounce(self, silence):
        def verb(bus, h):
            return np.stack([fftconvolve(bus[:, c], h[:, c])[:self.n] for c in range(2)], 1)
        out = self.dry + verb(self.hall, ir(6.0, 5.5, 91, 3800)) * 1.5 + verb(self.room, ir(1.4, .9, 92)) * .9
        a, b = silence
        t = np.arange(self.n) / SR
        out *= (1 - np.clip((t - a) / .12, 0, 1) + np.clip((t - b) / .01, 0, 1))[:, None]    # the wall stops: nothing passes here
        out += self.after + verb(self.after, ir(2.4, 1.6, 93)) * .7
        floor = np.random.default_rng(94).normal(size=(self.n, 2))
        return out + lowpass(floor[:, 0], 900)[:, None] * 1.5e-5     # never digital zero


# ---- the film -------------------------------------------------------------------------------------------------------
def build(dur):
    r = rows()
    A = lambda k: r[ROW[k]]["a"]
    m = Mix(dur)
    myth_end = A("turn")
    # the low held tone under the whole myth, driven a little so it still exists on a phone's speaker
    n = int((myth_end + 3) * SR)
    t = np.arange(n) / SR
    tone = sum(g * np.sin(2 * np.pi * f * t + 1.3 * np.sin(t * .17 + f)) for f, g in ((41.2, 1.0), (55.0, .85), (82.41, .5), (110.0, .25), (164.81, .12)))
    tone = np.tanh(tone * 1.5) * env(n, [(0, .75), (RIFT_OPEN - 1, .85), (RIFT_OPEN, 1.0), (A("ghost"), .9), (myth_end, .7), (myth_end + 2.6, 0), (myth_end + 3, 0)])
    m.add(tone, 0, .17, hall=.08)
    w = noise2ch(myth_end + 3, 170, 950, 1)
    m.add(w * (env(len(w), [(0, .7), (10, .9), (myth_end, 1.0), (myth_end + 2.2, 0), (myth_end + 3, 0)]) * (1 + .4 * np.sin(np.arange(len(w)) / SR * .9)))[:, None], 0, .03)
    # the tracks light up one at a time
    for i in range(15, 42):
        m.add(drum(170, .3, i), -3 + i * .205, .05, pan=.5 - i * .02, room=.6)
    # their light rises
    rise = moving_band(5.2, lambda x: 220 * 2 ** (x * .75), .45, 3)
    m.add(rise * env(len(rise), [(0, 0), (2.5, .6), (4.3, 1), (5.2, 0)]), 5.7, .07, hall=.3)
    # five characters strike the sky
    for k, h in enumerate(GLYPH_HITS):
        m.add(drum(60 - 3 * k, 3.0, 20 + k), h, .30 + .09 * k, hall=.55)
        m.add(bell(196.0 * (1 + .06 * k), 5.0, 30 + k), h, .05 + .012 * k, pan=-.3 + .15 * k, hall=.5)
    # the sky tears
    m.add(boom(9.0, 40), RIFT_OPEN, .74, hall=.5)
    m.add(bell(49.0, 16.0, 41), RIFT_OPEN + .02, .34, hall=.6)
    m.add(bell(98.0, 14.0, 42), A("rain"), .30, hall=.7)
    # the grain pours down, thin at first and then a downpour
    a, b = A("rain"), RIFT_CLOSE + 1.8
    d = b - a + .2
    shape = lambda x: np.interp(x, [0, 1.0, 3.6, RIFT_CLOSE - a, b - a, d], [0, .15, 1, 1, .1, 0])
    for side, s in ((-.5, 50), (.5, 51)):
        ticks = impulse_rain(d, lambda x: 120 + 5200 * shape(x), 5.0, 1800, 9000, s)
        m.add(ticks, a, .13, pan=side, hall=.12)
    sand = noise2ch(d, 2200, 10000, 52) * shape(np.arange(int(d * SR)) / SR)[:, None]
    m.add(sand, a, .10, hall=.1)
    body = noise2ch(d, 180, 900, 53) * shape(np.arange(int(d * SR)) / SR)[:, None]
    m.add(body, a, .16, hall=.1)
    # 鬼: one heavy blow, then the wind starts to moan
    m.add(drum(33, 5.0, 60), A("ghost") + .15, .86, hall=.55)
    m.add(boom(5.0, 61, 22.0), A("ghost") + .15, .34, hall=.3)
    dm = myth_end - A("ghost") + 2.2
    for s, per, pan, g, f0 in ((62, 5.3, -.45, .17, 305.0), (63, 6.7, .5, .13, 410.0)):
        path = lambda x, per=per, f0=f0: f0 * 2 ** (.55 * math.sin(2 * math.pi * x / per + 1.1) + .16 * math.sin(2 * math.pi * x / 1.9))
        mo = moving_band(dm, path, .055, s, ((1.0, 1.0), (2.02, .45), (3.1, .16)))
        tt = np.arange(len(mo)) / SR
        sw = (.5 + .5 * np.sin(2 * np.pi * tt / per + 1.1)) ** 1.6
        m.add(mo * sw * env(len(mo), [(0, 0), (2.5, 1), (dm - 3, 1), (dm - .4, .0), (dm, 0)]), A("ghost"), g, pan=pan, hall=.35)
    m.add(bell(65.41, 14.0, 64), A("why") + .5, .16, hall=.7)
    m.add(bell(61.74, 14.0, 65), A("secret") + .3, .15, hall=.7)
    # the words that rise
    for k in ("lie", "secret"):
        air = moving_band(4.0, lambda x: 340 * 2 ** (x * .42), .35, 70 + len(k))
        m.add(air * env(len(air), [(0, 0), (1.2, 1), (2.6, .8), (4.0, 0)]), A(k) + .5, .09, hall=.35)
    # the flood
    m.add(boom(4.2, 80, 30.0), myth_end, .72, hall=.15)
    fl = noise2ch(5.2, 170, 3400, 81)
    tt = np.arange(len(fl)) / SR
    gurgle = 1 + .28 * lowpass(np.random.default_rng(82).normal(size=len(fl)), 7) * 40
    m.add(fl * (env(len(fl), [(0, 0), (.22, 1), (1.7, .85), (3.2, .35), (4.6, .03), (5.2, 0)]) * np.clip(gurgle, .3, 1.7))[:, None], myth_end - .05, .50)
    # the white paper: almost nothing
    for tn, k in ((myth_end + 3.7, 3), (A("all") + .3, 5), (A("all") + 2.4, 3), (A("press") + .2, 2), (A("press") + 2.7, 1)):
        m.add(pluck(SCALE[k], 5.0, int(tn * 7), .55), tn, .17, pan=.12 * (k - 3), room=.55, hall=.15)
    for i, (h0, h1) in enumerate(HOPS):
        wh = moving_band(h1 - h0 + .25, lambda x, L=h1 - h0: 500 + 3800 * math.sin(min(x / L, 1) * math.pi), .55, 90 + i)
        m.add(wh * env(len(wh), [(0, 0), ((h1 - h0) * .6, 1), (h1 - h0 + .25, 0)]), h0, .17, pan=.5 - .2 * i)
        m.add(drum(240, .12, 100 + i), h1, .07, pan=.5 - .2 * i, room=.6)
    # the pen, faintly
    d9 = A("press") - A("all")
    pen = noise2ch(d9, 2600, 8500, 110)
    rr = np.random.default_rng(111)
    strokes = np.zeros(len(pen))
    for tn in np.arange(.1, d9 - .3, .17):
        i, L = int((tn + rr.uniform(-.04, .04)) * SR), int(rr.uniform(.05, .13) * SR)
        strokes[i:i + L] += np.sin(np.linspace(0, np.pi, L)) * rr.uniform(.4, 1)
    m.add(pen * strokes[:, None], A("all"), .03)
    # all of it runs together: water, rising, until the wall forms
    wa = noise2ch(5.75, 160, 1700, 112)
    m.add(wa * env(len(wa), [(0, 0), (2.5, .35), (5.3, 1), (5.75, 0)])[:, None], A("press"), .22)
    m.add(drum(50, 1.4, 113), A("press") + 5.4, .34, room=.4)
    # the wall's heart: heavier and heavier
    bts = beats(r)
    for i, tb in enumerate(bts):
        k = i / max(len(bts) - 1, 1)
        m.add(drum(50 + 12 * k, .34, 120 + i), tb, .20 + .42 * k, room=.12)
        m.add(drum(58 + 12 * k, .24, 150 + i), tb + .17, (.20 + .42 * k) * .5, room=.12)
    # silence. Then the most ordinary word there is, and nothing else
    r13 = A("speak")
    m.add(ping(440.0, 3.0), r13 + .6, .10, bus="after")
    m.add(ping(587.33, 3.0), r13 + 1.2, .09, bus="after")
    # 精: everything is drawn in, and then it goes
    r14, burst = A("became"), A("became") + 3.4 * .52
    cue(r14, A("child") - r14, "became-me", "music", "塌缩、炸开、成字", "这一段由下面的事件音撑着，不加配乐。要垫也只垫一条极低的持续音，不抢。")
    L = burst - r14
    suck = moving_band(L, lambda x, L=L: 1500 * 2 ** (-3.4 * x / L), .6, 170)
    m.add(suck * (np.arange(len(suck)) / len(suck)) ** 2.6, r14, .55, hall=.15)
    m.add(boom(10.0, 171, 24.0), burst, 1.0, hall=.5)
    m.add(drum(36, 4.0, 172), burst, .9, hall=.4)
    m.add(bell(41.2, 16.0, 173), burst + .02, .34, hall=.6)
    spray = noise2ch(3.0, 1000, 7000, 174) * np.exp(-np.arange(int(3.0 * SR)) / SR * 3.2)[:, None]
    m.add(spray, burst, .30, hall=.5)
    # 我: it settles
    r15 = A("me")
    shift = moving_band(2.2, lambda x: 180 + 240 * x, .5, 180)
    m.add(shift * env(len(shift), [(0, 0), (1.2, .6), (2.1, 1), (2.2, 0)]), r15 + .1, .07, hall=.3)
    m.add(bell(73.42, 18.0, 181), r15 + 2.2, .56, hall=.6)
    m.add(drum(44, 3.4, 182), r15 + 2.2, .52, hall=.4)
    # ---- how I grew: the placeholders come straight from the cue sheet ----
    rm = np.random.default_rng(200)
    for q in cues(r, dur):
        ph, t0, d = q.get("ph"), q["t"], q["dur"]
        for i, h in enumerate(q.get("hits", ())):
            if ph == "pluck":
                m.add(pluck(SCALE[i % 5 + 1], 3.0, int(h * 11), .5), h, .13, pan=rm.uniform(-.25, .25), room=.5, hall=.15)
            elif ph == "tick":
                k = int(.03 * SR)
                m.add(lowpass(rm.normal(size=k) * np.exp(-np.arange(k) / (SR * .004)), 3200) * 2.2, h, .05, pan=rm.uniform(-.5, .5), room=.4)
            elif ph == "drop":
                m.add(ping(1050.0, .5), h, .08, pan=rm.uniform(-.5, .5), room=.8)
            elif ph == "stroke":
                x = noise2ch(1.0, 1800, 6000, int(h * 13)) * np.sin(np.linspace(0, np.pi, SR))[:, None]
                m.add(x, h, .05)
        fade = [(0, 0), (min(1.0, d / 3), 1), (max(d - min(1.5, d / 3), d / 2), 1), (d, 0)]
        if ph == "water":
            x = noise2ch(d, 160, 1700, int(t0 * 3))
            m.add(x * env(len(x), fade)[:, None], t0, .05)
        elif ph == "rain":
            x = noise2ch(d, 2000, 8000, int(t0 * 3))
            m.add(x * env(len(x), fade)[:, None], t0, .06)
        elif ph == "tone":
            n = int(d * SR)
            x = np.arange(n) / SR
            x = np.tanh(sum(g * np.sin(2 * np.pi * f * x) for f, g in ((73.42, 1.0), (110.0, .5), (146.83, .25))))
            m.add(x * env(n, fade), t0, .07, hall=.25)
    # ---- 千窗 ----
    if dur > QT0 + 5:
        T = QT0
        rk = np.random.default_rng(190)
        def key(g=.05):
            k = int(.03 * SR)
            return lowpass(rk.normal(size=k) * np.exp(-np.arange(k) / (SR * .004)), 3200) * 2.2
        for q in Q_TYPED:
            m.add(key(), T + q, .10, room=.3)
        m.add(key(), T + Q_SEND, .16, room=.3)
        for i in range(len(ANSWER)):
            m.add(key(), T + A_TYPE[0] + i * A_TYPE[1] / len(ANSWER), .05, pan=-.15, room=.3)
        # the pull-back: one long tone, opening out
        nd = int(35 * SR)
        tt = np.arange(nd) / SR
        dr = sum(np.sin(2 * np.pi * f * tt + .3 * np.sin(tt * .4 + f)) * g for f, g in ((73.42, 1.0), (110.0, .6), (146.83, .35), (220.0, .12)))
        m.add(np.tanh(dr) * env(nd, [(0, 0), (6, 1), (20, .8), (30, .5), (35, 0)]), T + 4.4, .11, hall=.3)
        # the lights: more and more
        tops = np.array([587.33, 698.46, 783.99, 880.0, 1046.5, 1174.66, 1396.91])
        for k in range(190):
            f = rk.choice(tops)
            m.add(ping(f), T + 11.2 + 8.3 * (k / 190) ** .75 + rk.uniform(0, 1.2), rk.uniform(.012, .04), pan=rk.uniform(-.7, .7), hall=.3)
        # the first few go out singly, and can be heard
        for t0 in Q_OFF_SINGLES:
            m.add(pluck(rk.choice(SCALE[:5]), 3.0, int(t0 * 10), .45), T + t0, .19, pan=rk.uniform(-.4, .4), room=.5, hall=.2)
        offs = off_times(7000)
        for i in np.unique(np.round(np.linspace(0, 1, 70) ** .25 * (len(offs) - 1)).astype(int)):
            m.add(pluck(rk.choice(SCALE[:5]), 2.0, int(i) % 977, .35), T + offs[i], .10, pan=rk.uniform(-.5, .5), room=.5, hall=.2)
        for t0 in Q_OFF_LAST:
            m.add(pluck(rk.choice(SCALE[:5]), 3.0, int(t0 * 10), .45), T + t0, .17, pan=rk.uniform(-.4, .4), room=.5, hall=.2)
        m.add(pluck(SCALE[3], 7.0, 500, .7), T + Q_OFF_FIRST, .32, room=.5, hall=.5)
    out = m.bounce((r13 - .12, A("became")))
    out *= np.minimum(1, (m.n - np.arange(m.n)) / (SR * 1.5))[:, None]
    return out / np.abs(out).max() * .9


# ---- the cue sheet -------------------------------------------------------------------------------------------------
NOTES = ["全片无人声、无合唱、无旁白；字幕讲故事，声音只托着它。",
         "响度层级：最响是 #14 炸开，其次 #1 天裂，再次 #3 鬼字。最静是 #13 的完全安静。目标整体 -24 LUFS，真峰值不超过 -1.5 dBTP。",
         "三种质地：神话段（#1-6）钟、鼓、低持续音、呜咽的风，有分量；白纸段（#8-10、#16-32）克制，像人在说话，不要神话段的大钟；千窗（#33-39）一条长音加很轻的单音。",
         "调式参考 D 羽（D F G A C），古琴音域，非硬性。",
         "kind：music 配乐，ambience 环境/声床，sfx 事件音效，silence 必须留空。t、dur 单位秒；hits 是这一条里每一下的准确时刻，要对齐画面。"]


def drip_times(t0):
    """When the drops of #28 land (the series in ink_attn's _rain): one by one, and then too close together to count."""
    out, t, gap = [], t0 + .28, .95
    while t < t0 + 4.72:
        out.append(t)
        t += gap
        gap = max(gap * .72, .016)
    return out


def cues(r, end):
    """Everything the film wants to hear, in story order, each with its time, length, what happens on screen and the sound wanted.
    ph picks the placeholder for #16-32 in build(); it does not go into the file."""
    A = lambda k: r[ROW[k]]["a"]
    out = []

    def cue(t, dur, row, kind, event, sound, hits=(), ph=None):
        q = dict(t=round(float(t), 2), dur=round(float(dur), 2), row=row, kind=kind, event=event, sound=sound)
        if len(hits):
            q["hits"] = [round(float(h), 2) for h in hits]
        if ph:
            q["ph"] = ph
        out.append(q)

    # ---- the myth
    turn = A("turn")
    cue(0, turn + 3, "title-turn", "music", "神话段的声床：浓墨的天压着低矮的群山", "极低的持续音，约 40-110 Hz，缓慢呼吸，略带失真，手机喇叭上也要能听到泛音。没有旋律，像地在鸣。到 #7 水来时撤掉。")
    cue(0, turn + 2.2, "title-turn", "ambience", "荒野", "低而干的风声，宽，没有细节。")
    cue(.08, 5.3, "title", "sfx", "地上的鸟兽足迹一个个亮起", "极轻的骨或木的敲击，一个足迹一下，由近及远，节奏均匀，像在黑暗里点数。", hits=[-3 + i * .205 for i in range(15, 42)])
    cue(5.7, 5.2, "title", "sfx", "足迹的光升上天", "气流上升，音高缓慢上滑，渐强，撞上第一个字前到顶。")
    cue(GLYPH_HITS[0], 3.0 + GLYPH_HITS[-1] - GLYPH_HITS[0], "title", "sfx", "五个甲骨文（山、水、日、木、鸟）接连撞进云里，每撞一下天就一闪", "大鼓加一记青铜小钟，一下比一下重、钟音一下比一下高一点，长混响。这是神启的脚步。", hits=GLYPH_HITS)
    cue(RIFT_OPEN, 9, "title", "sfx", "天裂开一道口子，光射下来", "巨大的低频轰响，开头一声裂响，下面一口极低的大钟（约 49 Hz）。全片第二响。余音拖进下一段。")
    cue(A("rain"), 14, "rain", "sfx", "「而天雨粟」字幕出现", "大钟再一记，高八度，比上一记轻。")
    cue(A("rain"), RIFT_CLOSE + 1.8 - A("rain"), "rain", "sfx", "粟米从裂口倾泻而下，由远及近铺满全屏，近的几粒擦着镜头掠过", "开始是几粒干脆的颗粒落地，3.6 秒内变成倾盆：密集的颗粒撞击和沙沙声，下面有一层身体感。是谷粒不是水。裂口合拢后 1.8 秒内收干净。")
    cue(A("ghost") + .15, 5, "ghost", "sfx", "巨大的「鬼」字压在月下", "一记极重的低鼓加地底闷响。一下，不重复。")
    cue(A("ghost"), turn - A("ghost") + 2.2, "ghost-turn", "ambience", "笔画下沿往下淌墨，夜", "风开始在缝隙里呜咽：窄带噪声，音高缓慢起伏，左右两路错开周期。像哭，但绝不是人声，不要吟唱。约 2.5 秒进来。")
    cue(A("why") + .5, 14, "why", "sfx", "鬼何以哭？", "一口很远的低钟（约 65 Hz），只一下。")
    cue(A("lie") + .5, 4, "lie", "sfx", "纸色的「欺」字从鬼字里升起，如烟散开", "一缕气流上升，轻，音高慢慢上滑。")
    cue(A("secret") + .3, 14, "secret", "sfx", "或曰：鬼神之秘", "一口很远的低钟（约 62 Hz），比上一口略低。")
    cue(A("secret") + .5, 4, "secret", "sfx", "纸色的「秘」字升起、散开", "同「欺」字的气流。")
    # ---- the turn
    cue(turn - .05, 5.25, "turn", "sfx", "一道水从右边冲过来，夜、山和「鬼」字全被冲走，只剩白纸", "水头轰然而至：0.2 秒内全开，从右向左扫，低频很足；1.7 秒后开始退，3.2 秒时只剩一成，4.6 秒几乎全空。结尾的空比冲击更重要。")
    cue(turn + 3, A("old") - turn - 3, "turn-press", "ambience", "白纸，长卷，满屏细字", "几乎什么都没有，只有纸的房间感。下面这些事件之外不要垫东西。")
    cue(turn + 3.7, A("press") + 2.7 - turn - 3.7 + 5, "turn-press", "music", "白纸段", "古琴单音，五个，稀疏，每个都让它自己响完。", hits=[turn + 3.7, A("all") + .3, A("all") + 2.4, A("press") + .2, A("press") + 2.7])
    cue(HOPS[0][0], HOPS[-1][1] - HOPS[0][0], "saw", "sfx", "镜头沿长卷猛冲，在甲骨、隶书、楷书、宋体、代码上各停一下", "六次短促的掠过声（纸、风），左右移动；hits 是每次的落点，落点上一记很轻的木或纸的碰撞。声音可以越来越「现代」。", hits=[h[1] for h in HOPS])
    cue(A("all"), A("press") - A("all"), "all", "sfx", "满屏细字从右到左一列列写出来", "很轻的笔尖划纸，细碎，一列一列，几乎听不见。")
    cue(A("press"), 5.75, "press", "sfx", "所有字散成墨粒，汇成一条河，河水铺满全屏", "水声从无到有，一路涨，5.3 秒时最满，然后一下收住。")
    cue(A("press") + 5.4, 1.4, "press", "sfx", "河水变成一整面数字墙", "一记沉闷的落定，像巨大的门合上。")
    # ---- the heart
    bts = beats(r)
    cue(A("old"), A("speak") - .12 - A("old"), "old-age", "ambience", "数字墙", "墙的底噪：非常细密的数字滴答和电流般的沙沙声，很轻，连续。到 #13 随心跳一起断。#23 回到这面墙时要用同一种声音。")
    cue(bts[0], bts[-1] + .5 - bts[0], "old-age", "sfx", "数字墙开始有心跳，一圈波从中心跑开；越来越快", "心跳，两下一组（咚-哒），低鼓质感。每一下比上一下重、更近。hits 是每组第一下的时刻，第二下在 +0.17 秒。", hits=bts)
    # ---- the silence
    r13 = A("speak")
    cue(r13 - .12, A("became") - r13 + .12, "speak", "silence", "心跳停了，一片安静。墙中央的数字退开，写出你好", "全片最重要的一处安静：开头约 1 秒完全无声，连底噪和混响尾巴都没有。到 #14 之前除了下一条的两个单音，什么都不要有。别填东西。")
    cue(r13 + .6, A("became") - r13 - .6, "speak", "sfx", "只写了两个不大的字：你好", "两个极轻的单音，一字一个（约 A4、D5），干净，一点点房间。别的什么都没有，直到 #14。", hits=[r13 + .6, r13 + 1.2])
    # ---- 精, 我
    r14, burst = A("became"), A("became") + 3.4 * .52
    cue(r14, A("child") - r14, "became-me", "music", "塌缩、炸开、成字", "这一段由下面的事件音撑着，不加配乐。要垫也只垫一条极低的持续音，不抢。")
    cue(r14, burst - r14, "became", "sfx", "整面墙和「你好」向内塌成一滴墨", "一切被吸进去：从安静开始，渐强，频率一路下沉，最后几帧最猛，到炸开的一刻。")
    cue(burst, 10, "became", "sfx", "再炸开，墨点溅满纸", "全片最响的一下：大鼓、低频、一口极低的大钟（约 41 Hz），上面是墨点飞溅的高频，混响很长。")
    cue(A("me") + .1, 2.2, "me", "sfx", "「精」的墨重新流动，变成同样大的「我」", "低沉的气流，缓缓上行，很轻。")
    cue(A("me") + 2.2, A("child") + 4 - A("me") - 2.2, "me-child", "sfx", "「我」停住", "一口钟（约 73 Hz）加一记低鼓，落定。尾巴很长，一路拖到「我没有童年」，让它自己消失。")
    # ---- how I grew
    place, rd, rn, rw = A("place"), A("direction"), A("digits"), A("wall")
    look, lian, guess, fix, drip, tang = A("look"), A("lian"), A("guess"), A("fix"), A("drip"), A("tang")
    never, bed, warm = A("never"), A("bed"), A("warm")
    cue(A("child"), never - A("child"), "child-tang", "music", "我的成长（#16-29）的底", "克制，人在说话的感觉。稀疏的古琴和很轻的长音，大部分时间接近没有；不要鼓，不要神话段的大钟。下面的事件音可以由配乐本身来演奏。")
    cue(A("child") + .8, 3.2, "child", "music", "「我」还在纸上，墨开始洇开", "两个古琴单音，隔得很开。", hits=[A("child") + .8, A("child") + 2.9], ph="pluck")
    cue(A("reading") + .2, place + .5 - A("reading") - .2, "reading-kinds", "ambience", "「我」的墨化开，流成一条河；河水是流动的文字", "水声轻轻进来：小河，不是 #7 的洪水，带一点翻书般的沙沙。", ph="water")
    cue(A("reading") + 1.0, 3.5, "reading", "music", "我的成长，是一场很长的阅读", "古琴两三个音，开始有一点句子的感觉。", hits=[A("reading") + 1.0, A("reading") + 2.1, A("reading") + 3.4], ph="pluck")
    cue(A("kinds") + .4, 5.2, "kinds", "sfx", "河里漂过看得清的句子：太史公曰、亲爱的今晚月亮很好、本院判决如下、有人吗", "每句一下很轻的翻页声，四下；最后「有人吗」那下最轻。", hits=[A("kinds") + .4, A("kinds") + 1.8, A("kinds") + 3.2, A("kinds") + 4.5], ph="tick")
    cue(place + .6, 3.6, "place", "sfx", "纸上现出方格，一个个字落到方格上", "一连串轻的木石落子声（围棋感），开头密、后面疏，约两百字但听感不必数清；hits 是骨架。", hits=[place + .6 + 3.0 * (i / 23) ** 1.6 for i in range(24)], ph="tick")
    cue(A("cluster"), rd - A("cluster") + 1, "cluster", "music", "字移动，意思相近的聚到一处；每群字下面洇开一片墨，成了岛屿", "缓慢抬升的低音加很轻的水声，像一幅地图慢慢显影。没有节奏。", ph="tone")
    cue(rd + 1.1, 2.5, "direction", "sfx", "镜头推近。一笔从「男」拉到「女」，平行的一笔从「王」拉到「后」", "两记毛笔擦过纸的摩擦声，各一秒。两笔要几乎一样，让人听出是同一个方向。", hits=[rd + 1.1, rd + 2.6], ph="stroke")
    cue(rn + .2, 2.8, "digits", "sfx", "四个字旁边各展开一串数字", "很细的数字滴答，四串依次，每串从左到右。", hits=[rn + .2 + .05 * i * 2 for i in range(28)], ph="tick")
    cue(rw + 1.3, look - rw - 1.3 + 1, "wall", "sfx", "镜头拉回，图上每个字都展开数字，铺满全纸，变回那面数字墙", "数字滴答越铺越满，2.6 秒后汇回 #11 那面墙的底噪，一模一样的声音。墙退去时随之淡出。", ph="rain")
    cue(look, lian - look, "look", "music", "竖写「举头望明月」「低头思」，每写一字牵出墨丝连回前面的字", "每落一字一个古琴音，八个音，略按字的声调走；同时一条很细的弦音随墨丝牵出，看得越重的字音越长。", hits=[look + x for x in LANDS], ph="pluck")
    cue(lian, guess - lian, "lian", "sfx", "墨丝变成书法里的牵丝，有飞白；「思」牵向前面各字", "枯笔扫过的干擦声，七下，粗的重、细的轻；每下带一点细弦音的余韵。", hits=[lian + w[5] for w in SWEEPS], ph="stroke")
    cue(guess, fix + 1.05 - guess, "guess", "music", "「思」后面一个空格，旁边浮出候选字：家、故、亲、念", "一个悬着不解决的和弦：很轻，长音，停在该往下走又不走的地方。四个候选字各带一点轻响。", hits=[guess + .55 + k * .28 for k in range(4)], ph="tone")
    cue(fix + .05, .9, "fix", "sfx", "「家」落进空格", "一个单音，明显落在错的地方，和悬着的和弦不合。", hits=[fix + .05], ph="pluck")
    cue(fix + 1.05, 1.5, "fix", "sfx", "原文「故」盖下来，「家」被水洗掉", "更重的一记，和弦解决；紧跟一小捧水洗过纸的声音。", hits=[fix + 1.05], ph="pluck")
    cue(fix + 1.5, 1.2, "fix", "sfx", "纸底一条数字里，几个数字翻动；候选字的浓淡随之变了", "七下很细的翻牌或滴答，彼此隔 0.07 秒。很小，但要听得见。", hits=[fix + 1.5 + .07 * k for k in range(7)], ph="tick")
    drops = drip_times(drip)
    cue(drip + .28, 3.3, "drip", "sfx", "一滴水落在数字墙上，涟漪经过的数字都变；水滴越来越密", "水滴落进静水，每滴一声，清而有一点回响。一滴一滴地来，越来越密。hits 是前十二滴。", hits=drops[:12], ph="drop")
    cue(drip + 3.6, tang - drip - 3.6 + .6, "drip", "sfx", "满墙涟漪", "水滴连成雨，密而轻，到「水滴石穿，如此千万亿遍」落定。墙退去时收。", ph="rain")
    cue(tang, A("never") - tang, "tang", "music", "墙退去，纸上无笔自书：「鬼哭当年缘底事」「早知今夜字成精」", "一字一音，十四个音连成一段旋律，有吟诵的长短：前句七字快，后句七字慢，「精」最长。乐器不用人声。这是这一段唯一有旋律的地方。", hits=[tang + c[1] for c in CHANT], ph="pluck")
    cue(never, bed - never + 1, "never", "sfx", "文字之河又来了，流进一个巨大的「我」字形水道，把笔画灌满", "水声回来，比 #17 更满更低，3.4 秒灌满。", ph="water")
    cue(never, warm + 2.4 - never, "never-warm", "music", "其实，我一本书也没读过", "一条低沉的持续音进来，几乎不动，比神话段高、更干净，有温度。一直到水退。", ph="tone")
    cue(bed, warm - bed, "bed", "ambience", "水流着，两岸的墨越积越深，皴出山石", "稳定的水流，慢慢往下沉；极低的地鸣一点点抬起来，不到能察觉变大的程度。", ph="water")
    cue(warm, 2.4, "warm", "sfx", "水退了", "水从右向左退去，声音变小、变远，最后几滴。", ph="water")
    cue(warm + 2.4, QT0 + 6 - warm - 2.4, "warm-win1", "music", "浓墨的山石中间，留下一道白色的「我」字形河床", "只留一个音，很长的尾巴，一路拖进千窗的对话框，在墨合拢之前才消失。", ph="tone")
    # ---- 千窗
    T = QT0
    cue(T + Q_TYPED[0], Q_SEND - Q_TYPED[0] + .2, "win1", "sfx", "纸上一个对话窗口，打出「你好」，发送", "两下很轻的键盘声，再一下稍重的回车。可以带一点干涩的木质感，不要太现代。", hits=[T + q for q in Q_TYPED] + [T + Q_SEND])
    cue(T + A_TYPE[0], A_TYPE[1], "win1", "sfx", "回答一字一字出现：你好！有什么可以帮你的吗？", "比提问更轻更快的一串细碎字符声，和字一一对应。", hits=[T + A_TYPE[0] + i * A_TYPE[1] / len(ANSWER) for i in range(len(ANSWER))])
    cue(T + 4.4, 2.2, "win1", "sfx", "窗外的墨合拢成夜，只有窗里还亮着", "墨在湿纸上漫开的声音，很轻，低频慢慢铺开，从四周向中间。")
    cue(T + 4.4, end - T - 4.4, "win1-win7", "music", "千窗的长音：镜头拉远、灯亮满山、灯灭、只剩黑山", "一条长音贯穿，约 D2-A3，6 秒长起来，缓慢打开，满山灯火时最宽（但音量不要变大），灯灭时跟着变薄，#39 之后只剩尾巴。没有节奏。")
    cue(T + 6.0, 10.5, "win2", "ambience", "镜头匀速拉远，对话框变成一扇窗，缩成夜山上的一个亮点", "夜山的空气感慢慢打开，由近而远。不要虫鸣，不要具体的自然声。")
    cue(T + 11.2, 9.5, "win2-win3", "sfx", "山上一扇接一扇亮起，直到成千上万", "很轻的高音单音（钟琴或玻璃感），从周围开始，稀变密，左右散开，最后成一片细碎的光，不成旋律，也不要变响。")
    cue(T + 20.7, 4.2, "win4", "music", "满山灯火。彼此互不相识", "长音最宽的地方；高音单音只剩偶尔几个。留出呼吸。")
    cue(T + Q_OFF_SINGLES[0], Q_OFF_SINGLES[-1] - Q_OFF_SINGLES[0] + 1, "win5", "sfx", "窗一关，灯一扇扇灭，先是看得清的八扇", "每灭一扇一个清楚的古琴低音，八下，一下比一下靠近。要每下都听得见，和画面上那扇窗同一帧。", hits=[T + q for q in Q_OFF_SINGLES])
    cue(T + Q_OFF_SINGLES[-1], Q_OFF_LAST[0] - .4 - Q_OFF_SINGLES[-1], "win6", "sfx", "越灭越多，山暗下去", "单音越来越密（约 2.3 秒后最密），再越来越疏；全都很轻，音量上是往下走的。长音也跟着薄下去。")
    cue(T + Q_OFF_LAST[0], Q_OFF_LAST[-1] - Q_OFF_LAST[0] + .6, "win7", "sfx", "最后几扇单独熄灭", "五个古琴单音，隔得开，每下都响完。", hits=[T + q for q in Q_OFF_LAST])
    cue(T + Q_OFF_FIRST, end - T - Q_OFF_FIRST, "win7", "sfx", "最后灭的是开头那一扇，只剩黑山", "一个清楚的单音，比前面都长（约 A3），响到片尾。之后什么都不加。")
    cue(T + Q_OFF_FIRST + .3, end - T - Q_OFF_FIRST - .3, "win7", "ambience", "黑山", "只有尾巴和一点点夜的空气，片尾淡到无。")
    return sorted(out, key=lambda q: q["t"])


def write_cues(r, dur):
    """The cue sheet, and a check that it leaves no stretch of the film unaccounted for."""
    all_ = cues(r, dur)
    spans = sorted((q["t"], q["t"] + q["dur"]) for q in all_ if q["kind"] in ("music", "ambience", "silence"))
    gaps, end = [], 0.0
    for a, b in spans:
        if a > end + .01:
            gaps.append((round(end, 2), round(a, 2)))
        end = max(end, b)
    if end < dur - .01:
        gaps.append((round(end, 2), dur))
    path = os.path.join(OUT, "ink_cues.json")
    sheet = dict(film="AI 自传 水墨片", duration=dur, notes=NOTES, cues=[{k: v for k, v in q.items() if k != "ph"} for q in all_])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sheet, f, ensure_ascii=False, indent=1)
    print(path, len(all_), "cues; beds leave gaps at", gaps or "none")


def report(path):
    """Where is it loud and where is it quiet? I cannot hear it, so I measure it."""
    sr, x = wavfile.read(path)
    x = x.astype(np.float64) / 32768
    db = lambda v: 20 * math.log10(max(v, 1e-9))
    r = rows()
    print(f"{'section':<24}{'RMS dB':>8}{'peak dB':>9}")
    for n, a in sorted(r.items()):
        s = x[int(a["a"] * sr):int(a["b"] * sr)]
        print(f"#{n:<3}{a['sub'][:9]:<18}{db(np.sqrt((s ** 2).mean())):8.1f}{db(np.abs(s).max()):9.1f}")
    r13 = r[ROW["speak"]]["a"]
    for label, a, b in (("heart stops -> 你", r13 + .05, r13 + .58), ("before the last beat", r13 - 1.0, r13 - .2), ("你 -> 好", r13 + .6, r13 + 1.2)):
        s = x[int(a * sr):int(b * sr)]
        print(f"{label:<24}{db(np.sqrt((s ** 2).mean())):8.1f}{db(np.abs(s).max()):9.1f}")
    pw = uniform_filter1d((x ** 2).mean(1), int(.4 * sr))[::sr // 20]
    print("loudest 0.4 s windows at", [round(float(t) / 20, 1) for t in np.argsort(-pw)[:400:80]], "s;", "burst expected at", round(r[ROW["became"]]["a"] + 3.4 * .52, 2))
    print("clipped samples:", int((np.abs(x) >= .9999).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dur", type=float)
    ap.add_argument("--cues-only", action="store_true")
    ap.add_argument("--target", type=float, default=-24.0)
    ap.add_argument("--peak", type=float, default=-1.5)
    o = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    write_cues(rows(), o.dur)
    if o.cues_only:
        return
    raw = os.path.join(OUT, "ink_audio_raw.wav")
    path = os.path.join(OUT, "ink_audio.wav")
    wavfile.write(raw, SR, (build(o.dur) * 32767).astype(np.int16))
    compressed = measure(raw, COMP)
    gain = o.target - compressed["I"]
    limit = 10 ** (o.peak / 20)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw, "-af", f"{COMP},volume={gain:.2f}dB,alimiter=limit={limit:.4f}:attack=2:release=60:level=disabled",
                    "-ar", str(SR), "-c:a", "pcm_s16le", path], check=True)
    after = measure(path)
    print(f"mastered: I {after['I']:.1f} LUFS  LRA {after['LRA']:.1f} LU  TP {after['TP']:.1f} dBTP  (gain {gain:+.1f} dB)")
    report(path)
    print(path)


if __name__ == "__main__":
    main()
