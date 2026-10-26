"""How well do the visual hits land on the music? Finds frames where the picture jumps (bursts, flashes, cuts) and measures
each one's distance to the nearest beat of the song as it plays in the finished video.

    python beatcheck.py film.mp4 --cold 3.8667 --body-from 1.366667     # cut points on frame boundaries (41/30 s)
"""
import argparse
import csv
import os
import subprocess

import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
BEATS = os.path.normpath(os.path.join(HERE, "..", "..", "ai_history", "bgm", "bgm_v3.beats.csv"))
COLD_BEATS = [0.0, 0.975, 1.951, 2.903]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--cold", type=float, default=3.8667)
    ap.add_argument("--body-from", type=float, default=1.366667)
    a = ap.parse_args()
    w, h, fps = 96, 96, 30
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", a.video, "-vf", f"scale={w}:{h},format=gray", "-r", str(fps),
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(float)
    diff = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))
    t = (np.arange(len(diff)) + 1) / fps
    pk, _ = find_peaks(diff, height=np.percentile(diff, 97), distance=int(0.4 * fps))
    with open(BEATS) as fh:
        song = np.array([float(r["t"]) for r in csv.DictReader(fh)])
    beats = np.concatenate([COLD_BEATS, a.cold + song[song >= a.body_from] - a.body_from])
    half = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2]))
    d = np.array([np.min(np.abs(half - t[p])) for p in pk])
    print(f"{len(pk)} visual hits; within 1 frame of a (half-)beat: {np.mean(d <= 1.01 / fps) * 100:.0f}%, "
          f"within 2 frames: {np.mean(d <= 2.01 / fps) * 100:.0f}%, median offset {np.median(d) * 1000:.0f} ms")
    worst = sorted(zip(d, t[pk]), reverse=True)[:5]
    print("largest offsets (ms @ s):", [(round(x * 1000), round(y, 2)) for x, y in worst])


if __name__ == "__main__":
    main()
