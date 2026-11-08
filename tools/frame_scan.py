"""Scan the final reel's per-frame luma (96x54 gray raw) for flashes and cuts, and set the cuts against the shot list.

    python -I frame_scan.py luma96.raw final_list.txt
"""
import re
import sys

import numpy as np

W, H, FPS = 96, 54, 30
HEAD, OFF = 20.0, 4.3            # reel = film below HEAD_S, film + (HEAD_S - GROW) from there
CREDITS0 = 529.5


def reel(t):
    return t if t < HEAD else t + OFF


raw = np.fromfile(sys.argv[1], np.uint8)
n = raw.size // (W * H)
f = raw[: n * W * H].reshape(n, H, W).astype(np.float32)
print("frames", n, "seconds", n / FPS)
mean = f.mean(axis=(1, 2))
d = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))        # d[i] = |f[i+1] - f[i]|

# expected cuts: every shot start in the list, as reel frames
exp = []
for line in open(sys.argv[2], encoding="utf-8"):
    m = re.match(r"\s*([\d.]+)\s+([\d.]+)\s+\S+", line)
    if m:
        exp.append(reel(float(m.group(1))))
exp = sorted(set(round(x, 3) for x in exp))
exp += [CREDITS0, 675.5, 677.5]
exp_frames = [int(round(x * FPS)) for x in exp]

# flashes: a frame unlike both neighbours while the neighbours are alike
flash = []
for i in range(1, n - 1):
    a, b, c = d[i - 1], d[i], abs(f[i + 1] - f[i - 1]).mean()
    if a > 12 and b > 12 and c < 0.35 * min(a, b):
        flash.append((i, round(i / FPS, 3), round(a, 1), round(b, 1), round(float(c), 1)))
print("flashes (frame, s, d_in, d_out, d_skip):", len(flash))
for x in flash[:40]:
    print("  ", x)

# short shots: two big changes within 3 frames
cuts = [i + 1 for i in range(n - 1) if d[i] > 18]
print("big changes:", len(cuts))
unexpected = [c for c in cuts if min(abs(c - e) for e in exp_frames) > 2]
print("big changes not at a listed cut:", len(unexpected))
for c in unexpected[:60]:
    print("   frame %d  %.3f s  d %.1f  luma %.1f -> %.1f" % (c, c / FPS, d[c - 1], mean[c - 1], mean[c]))
short = [(a, b) for a, b in zip(cuts, cuts[1:]) if 0 < b - a <= 3]
print("changes 3 frames or less apart:", short[:30])

# listed cuts with no visible change (fine for matched cuts, worth a look otherwise)
quiet = [(e, round(e / FPS, 2), round(float(d[e - 1]), 1)) for e in exp_frames if 0 < e < n and d[e - 1] < 2]
print("listed cuts with |d| < 2:", quiet)

# black runs (mean luma < 4) of 3 frames or more
blk = mean < 4
runs, i = [], 0
while i < n:
    if blk[i]:
        j = i
        while j < n and blk[j]:
            j += 1
        if j - i >= 3:
            runs.append((round(i / FPS, 2), round(j / FPS, 2)))
        i = j
    else:
        i += 1
print("black runs:", runs)
