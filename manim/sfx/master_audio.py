"""Master the sound-effects-only audio of a rendered video (the raw Manim mix is very quiet: about -40 LUFS).

Chain: gentle compression (narrows the range so small sounds stay audible and accents don't jump out)
-> make-up gain to the target loudness -> peak limiter. The video stream is copied, not re-encoded.

    python3 master_audio.py in.mp4 out.mp4 [--target -24] [--peak -1.5]

Prints the before/after integrated loudness (LUFS), loudness range (LU) and true peak (dBTP), and fails if the
result misses the target by more than 1 LU or exceeds the peak ceiling.
"""
import argparse
import re
import subprocess

COMP = "acompressor=threshold=-34dB:ratio=2.5:attack=5:release=250:knee=6"


def measure(path, pre=""):
    chain = (pre + "," if pre else "") + "ebur128=peak=true"
    out = subprocess.run(["ffmpeg", "-nostats", "-i", path, "-af", chain, "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    summary = out[out.rfind("Summary:"):]
    get = lambda key: float(re.search(key + r":\s+(-?[\d.]+|-inf)", summary).group(1))
    return {"I": get("I"), "LRA": get("LRA"), "TP": get("Peak")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--target", type=float, default=-24.0)
    ap.add_argument("--peak", type=float, default=-1.5)
    a = ap.parse_args()

    before = measure(a.src)
    compressed = measure(a.src, COMP)
    gain = a.target - compressed["I"]
    limit = 10 ** (a.peak / 20)
    chain = f"{COMP},volume={gain:.2f}dB,alimiter=limit={limit:.4f}:attack=2:release=60:level=disabled"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.src, "-map", "0:v", "-map", "0:a",
                    "-c:v", "copy", "-af", chain, "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-movflags", "+faststart", a.dst], check=True)
    after = measure(a.dst)
    print(f"before  I {before['I']:6.1f} LUFS  LRA {before['LRA']:4.1f} LU  TP {before['TP']:6.1f} dBTP")
    print(f"after   I {after['I']:6.1f} LUFS  LRA {after['LRA']:4.1f} LU  TP {after['TP']:6.1f} dBTP  (gain {gain:+.1f} dB)")
    assert abs(after["I"] - a.target) <= 1.0, after
    assert after["TP"] <= a.peak + 0.5, after      # AAC re-encode can overshoot the limiter slightly


if __name__ == "__main__":
    main()
