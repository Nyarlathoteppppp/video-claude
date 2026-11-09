"""Looking at the events layer: frames and strips of a shot through ink_cut.framed, and how long the layer takes.

    python ink_events_lab.py strip drip/junk [N=6] [k=0] [--a=FRAC --b=FRAC]     # ink/film_out/move/events/drip_junk.png: N frames, 640 wide; k: which use of the still
    python ink_events_lab.py at drip/junk 1.5 [k]                                # the full frame t seconds into the shot
    python ink_events_lab.py zoom drip/junk 1.5 x0 y0 x1 y1 [k]                  # a crop of it, doubled
    python ink_events_lab.py time drip/junk [k]                                  # seconds a frame, with and without the layer
    python ink_events_lab.py table                                               # every shot this layer touches: start, length
"""
import os
import sys
import time

import numpy as np
from PIL import Image

import ink_cut
import ink_events
from ink_cut import END, SHOTS, framed

OUT = os.path.join(ink_cut.ROOT, "ink", "film_out", "move", "events")


def shot(still, k=0):
    hits = [i for i, s in enumerate(SHOTS) if s["still"] == still]
    i = hits[k]
    s = SHOTS[i]
    return s, (SHOTS[i + 1]["start"] if i + 1 < len(SHOTS) else END)


def name(still, tag=""):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, still.replace("/", "_") + tag + ".png")


def strip(still, n=6, k=0, a=0.0, b=1.0, cols=3, tw=640):
    s, end = shot(still, k)
    dur = end - s["start"]
    th = tw * 9 // 16
    rows = (n + cols - 1) // cols
    out = Image.new("RGB", (cols * tw, rows * th))
    for i in range(n):
        t = dur * (a + (b - a) * i / max(n - 1, 1))
        t = min(t, dur - 0.02)
        f = framed(s, s["start"] + t, end)
        out.paste(Image.fromarray(f).resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th))
    p = name(still, f"_k{k}" if k else "")
    out.save(p)
    print(p, f"dur {dur:.2f}")


def at(still, t, k=0):
    s, end = shot(still, k)
    p = name(still, f"_at{t:.2f}")
    Image.fromarray(framed(s, s["start"] + t, end)).save(p)
    print(p)


def zoom(still, t, x0, y0, x1, y1, k=0):
    s, end = shot(still, k)
    p = name(still, f"_z{t:.2f}")
    Image.fromarray(framed(s, s["start"] + t, end)).crop((x0, y0, x1, y1)).resize(((x1 - x0) * 2, (y1 - y0) * 2), Image.LANCZOS).save(p)
    print(p)


def timing(still, k=0):
    s, end = shot(still, k)
    dur = end - s["start"]
    ev = ink_events.EVENTS.get(still)
    framed(s, s["start"] + dur / 2, end)      # builds what is cached
    ts = [dur * x for x in (0.2, 0.4, 0.6, 0.8, 0.95)]
    a = time.perf_counter()
    fs = []
    ink_events.EVENTS.pop(still, None)
    for t in ts:
        fs.append(framed(s, s["start"] + t, end))
    base = (time.perf_counter() - a) / len(ts)
    extra = 0.0
    if ev:
        ink_events.EVENTS[still] = ev
        a = time.perf_counter()
        for t, f in zip(ts, fs):
            ev(f, t, dur)
        extra = (time.perf_counter() - a) / len(ts)
    print(f"{still}: base {base:.3f} s, layer {extra:.3f} s a frame")


def table():
    for i, s in enumerate(SHOTS):
        end = SHOTS[i + 1]["start"] if i + 1 < len(SHOTS) else END
        print(f"{i:3d} {s['still']:24s} {s['start']:7.2f} {end - s['start']:6.2f} z={s['z']} c={s['c']} join={s['join']} m={'y' if s['m'] is not None else 'n'} ev={'y' if s['still'] in ink_events.EVENTS else 'n'}")


if __name__ == "__main__":
    what = sys.argv[1]
    args = [x for x in sys.argv[2:] if not x.startswith("--")]
    opts = dict(x[2:].split("=") for x in sys.argv[2:] if x.startswith("--"))
    if what == "strip":
        strip(args[0], *(int(x) for x in args[1:3]), a=float(opts.get("a", 0)), b=float(opts.get("b", 1)), cols=int(opts.get("cols", 3)), tw=int(opts.get("tw", 640)))
    elif what == "at":
        at(args[0], float(args[1]), *(int(x) for x in args[2:3]))
    elif what == "zoom":
        zoom(args[0], float(args[1]), *(int(x) for x in args[2:7]))
    elif what == "time":
        timing(args[0], *(int(x) for x in args[1:2]))
    elif what == "table":
        table()
    else:
        raise SystemExit(__doc__)
