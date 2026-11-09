"""The 1080p pass of shots already approved at 960 by 540: the same frames, only bigger. Nothing here sets a time, a camera, a light or a look; each part calls the code that made the low version, at full size, and writes beside it under a _1080 name, never over anything.

    python film_hi1080.py room            # room/seq3_1080/room_NNNN.png: 三's room, film_room.py's frames 0 to 407 at full size (frames already there are kept)
    python film_hi1080.py myth5           # myth/myth_low5_1080.mp4: film_myth_seq.py's burn of myth_low5 (frames 150 to 1199), from the same raw frames, no lines burned
    python film_hi1080.py myth7           # myth/myth_low7_1080.mp4: its burn7 (鬼何以哭 to 依我之见), the same way
    python film_hi1080.py encode petal    # sorrow/seq/petal_1080.mp4 from sorrow/seq/petal_1080/ (film_sorrow3d_hi.py), as crew B encoded the low one
    python film_hi1080.py rainfix         # sorrow/seq/rain_1080_fix/: the rain's top right corner mended (see RAIN_BOX); then rainencode, rainsheets
    python film_hi1080.py sheet NAME      # ~/render_work/crewD/check_NAME.png: low and hi side by side at matching times, and how far apart they are

The myth's raw frames were always rendered at 1920 by 1080 (film_myth_seq.py: "each shot renders in its own process at film size"); burn put the lines on and made them small. So its 1080 version is those raw frames encoded, no simulation run again.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FILM = os.path.join(HERE, "..", "..", "ink", "film_out", "film")
W, H, FPS = 1920, 1080, 30
CHECK = os.path.expanduser("~/render_work/crewD")


def encoder(path, crf, preset="medium", faststart=False):
    assert not os.path.exists(path), path + " is there already"
    cmd = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]
    return subprocess.Popen(cmd + (["-movflags", "+faststart"] if faststart else []) + [path], stdin=subprocess.PIPE)


def room():
    """film_room.at(t, 1.0) without its ink_subs.apply: the room's own times before its cut carry no line now, and the cut burns the film's own."""
    import film_room as R
    out = os.path.join(FILM, "room", "seq3_1080")
    os.makedirs(out, exist_ok=True)
    n = int(round((R.T1 - R.START) * R.FPS))
    for i in range(n):
        path = os.path.join(out, f"room_{i:04d}.png")
        if os.path.exists(path):
            continue
        t = R.START + i / R.FPS
        if t < R.CUT:
            f = R.finish(R.shot(R.Room, 1.0).frame(R.ui.state(t), t), int(t * R.FPS) + 7, .0065)
        else:
            f = R.finish(R.shot(R.Close, 1.0).frame(R.ui.state(t), t), int(t * R.FPS) + 7, .0035)
        assert f.shape == (H, W, 3)
        Image.fromarray(f).save(path, compress_level=1)
        if i % 30 == 0:
            print("room", i, n, flush=True)
    print(out, n, "frames", flush=True)


def myth(which):
    import film_myth_seq as M
    import ink_subs
    if which == "5":
        first, last, name = 150, int(round(ink_subs.at("或曰") * FPS)), "myth_low5"
    else:
        first, last, name = int(round(ink_subs.at("鬼何以哭") * FPS)), int(round(ink_subs.at("依我之见") * FPS)), "myth_low7"
    path = os.path.join(M.MYTH, name + "_1080.mp4")
    enc = encoder(path, 16, "slow", True)          # as burn encoded the low one
    prev_path, prev = None, None
    for n in range(first, last):
        src = M.source(n)
        if src != prev_path:
            prev, prev_path = np.asarray(Image.open(src).convert("RGB"), np.uint8), src
        assert prev.shape == (H, W, 3), src
        enc.stdin.write(prev.tobytes())
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("encoding failed")
    print(path, last - first, "frames", flush=True)


def encode(shot):
    """As crew B's tools/seqs.sh encodes the low ones: 30 fps, the first 135 frames, crf 17."""
    s = os.path.join(FILM, "sorrow", "seq")
    path = os.path.join(s, f"{shot}_1080.mp4")
    assert not os.path.exists(path), path
    subprocess.run(["ffmpeg", "-loglevel", "error", "-framerate", "30", "-i", os.path.join(s, f"{shot}_1080", "%04d.png"), "-frames:v", "135", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", path], check=True)
    print(path, flush=True)


# ---------------------------------------------------------------- the rain's corner

# Top right of the rain shot, the dark beyond the lattice and the soft bar under it: nothing there moves (the camera, the lattice and the streaks are still; only the drops move, far to the left), but at 96 samples the denoiser lays the dark's lower edge down a little differently each frame, often as a stepped, grainy band, and at full size it flickers.
# In the frames where it came out clean the edge is the same, so each frame's corner becomes the median of the five clean frames nearest it, under a feathered mask: full right of x 1660 and above y 230, nothing left of 1600 or below 290 (the drops never come right of x 1211, the bead on the bar stands at x 1480 to 1500).
RAIN_BOX = (1560, 320)                    # the part of the frame the mask lies in: x from 1560 to the edge, y from the top to 320


def rain_mask():
    x0, y1 = RAIN_BOX
    xx = np.arange(W - x0)[None, :] + x0
    yy = np.arange(y1)[:, None]
    sm = lambda u: np.clip(u, 0, 1) ** 2 * (3 - 2 * np.clip(u, 0, 1))
    return (sm((xx - 1600) / 60.0) * sm((290 - yy) / 60.0)).astype(np.float32)


def rain_rough(corner):
    """How grainy the dark's edge is: the mean size of the Laplacian over the band it lies in. Clean frames come out at 0.5 to 0.57, the stepped ones at 0.7 to 2."""
    from scipy.ndimage import laplace
    return float(np.abs(laplace(corner.mean(-1)[100:260, 1650 - RAIN_BOX[0]:])).mean())


def rain_fix():
    """sorrow/seq/rain_1080_fix/NNNN.png, every frame with its corner mended; the frames as rendered stay in rain_1080/."""
    s = os.path.join(FILM, "sorrow", "seq")
    src, out = os.path.join(s, "rain_1080"), os.path.join(s, "rain_1080_fix")
    assert not os.path.exists(out), out + " is there already"
    os.makedirs(out)
    x0, y1 = RAIN_BOX
    frames = [np.asarray(Image.open(os.path.join(src, f"{i:04d}.png")).convert("RGB")) for i in range(135)]
    corners = np.stack([f[:y1, x0:].astype(np.float32) for f in frames])
    rough = np.array([rain_rough(c) for c in corners])
    clean = np.nonzero(rough < 0.6)[0]
    print("clean frames", len(clean), clean.tolist(), flush=True)
    m = rain_mask()[..., None]
    after = []
    for i, f in enumerate(frames):
        near = clean[np.argsort(np.abs(clean - i), kind="stable")[:5]]
        plate = np.median(corners[near], 0)
        y = f.copy()
        y[:y1, x0:] = np.uint8(np.clip(corners[i] * (1 - m) + plate * m + .5, 0, 255))
        after.append(rain_rough(y[:y1, x0:].astype(np.float32)))
        Image.fromarray(y).save(os.path.join(out, f"{i:04d}.png"), compress_level=1)
    print("roughness before: max %.2f, %d frames above 0.6; after: max %.2f, %d above 0.6" % (rough.max(), (rough >= .6).sum(), max(after), sum(a >= .6 for a in after)), flush=True)
    print(out, flush=True)


def rain_encode():
    """sorrow/seq/rain_1080.mp4 again from the mended frames, the one before it kept as rain_1080_old1.mp4."""
    s = os.path.join(FILM, "sorrow", "seq")
    path, old = os.path.join(s, "rain_1080.mp4"), os.path.join(s, "rain_1080_old1.mp4")
    assert not os.path.exists(old), old
    os.rename(path, old)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-framerate", "30", "-i", os.path.join(s, "rain_1080_fix", "%04d.png"), "-frames:v", "135", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", path], check=True)
    print(path, flush=True)


def rain_sheets():
    """~/render_work/crewD/rain_corner_fix.png: the corner at 100 %, frames 58 to 63, as rendered (left) and mended (right). rain_mask_check.png: the mask over the frames where the drop is furthest right (0) and the bead on the bar shows (120), half size, and the bead at 100 %."""
    s = os.path.join(FILM, "sorrow", "seq")
    x0, y1 = RAIN_BOX
    rows = []
    for i in range(58, 64):
        a = np.asarray(Image.open(os.path.join(s, "rain_1080", f"{i:04d}.png")).convert("RGB"))[:y1, x0:]
        b = np.asarray(Image.open(os.path.join(s, "rain_1080_fix", f"{i:04d}.png")).convert("RGB"))[:y1, x0:]
        tile = Image.fromarray(np.concatenate([a, np.full((y1, 8, 3), 255, np.uint8), b], 1))
        ImageDraw.Draw(tile).text((6, 4), f"{i} before", fill=(255, 255, 0))
        ImageDraw.Draw(tile).text((W - x0 + 14, 4), f"{i} after", fill=(255, 255, 0))
        rows.append(np.asarray(tile))
    os.makedirs(CHECK, exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(CHECK, "rain_corner_fix.png"))
    full = np.zeros((H, W), np.float32)
    full[:y1, x0:] = rain_mask()
    shown = []
    for i in (0, 120):
        f = np.asarray(Image.open(os.path.join(s, "rain_1080", f"{i:04d}.png")).convert("RGB")).astype(np.float32)
        f = f * (1 - .6 * full[..., None]) + np.array([255, 40, 40], np.float32) * .6 * full[..., None]
        shown.append(np.asarray(Image.fromarray(np.uint8(f)).resize((W // 2, H // 2), Image.LANCZOS)))
    bead = np.asarray(Image.fromarray(np.uint8(f)).crop((1380, 180, 1920, 480)))          # frame 120 at 100 %: the bead and the mask's feathered edge
    Image.fromarray(np.concatenate([np.concatenate(shown, 1), np.pad(bead, ((0, H // 2 - bead.shape[0]), (0, W - bead.shape[1]), (0, 0)))], 0)).save(os.path.join(CHECK, "rain_mask_check.png"))
    print(os.path.join(CHECK, "rain_corner_fix.png"), os.path.join(CHECK, "rain_mask_check.png"), flush=True)


# ---------------------------------------------------------------- checking

def video_frame(path, k, size):
    """Frame k of a clip, at size."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{k}),scale={size[0]}:{size[1]}:flags=lanczos", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(size[1], size[0], 3)


def frame_of(src, k, size):
    if os.path.isdir(src):
        return np.asarray(Image.open(os.path.join(src, f"room_{k:04d}.png")).convert("RGB").resize(size, Image.LANCZOS))
    return video_frame(src, k, size)


PAIRS = dict(
    twelve=("names/twelve.mp4", "names/twelve_1080.mp4", [0, 100, 200, 300, 380, 520, 620, 700, 760, 850, 989]),
    room=("room/seq3", "room/seq3_1080", [0, 45, 89, 90, 200, 300, 407]),
    petal=("sorrow/seq/petal.mp4", "sorrow/seq/petal_1080.mp4", [0, 30, 60, 90, 110, 134]),
    rain=("sorrow/seq/rain.mp4", "sorrow/seq/rain_1080.mp4", [0, 30, 60, 90, 110, 134]),
    lamp=("sorrow/seq/lamp.mp4", "sorrow/seq/lamp_1080.mp4", [0, 30, 60, 90, 110, 134]),
    myth5=("myth/myth_low5.mp4", "myth/myth_low5_1080.mp4", [0, 150, 300, 450, 600, 750, 824, 825, 914]),
    myth7=("myth/myth_low7.mp4", "myth/myth_low7_1080.mp4", [0, 80, 134, 135, 250, 350, 464]),
)


def sheet(name):
    """Each row: the low frame, the hi one made as small, and their difference ten times over; the mean difference over the frame (and over its top four fifths, where no line is burned) printed on it."""
    low, hi, ks = PAIRS[name]
    low, hi = os.path.join(FILM, low), os.path.join(FILM, hi)
    size = (480, 270)
    rows = []
    for k in ks:
        a = frame_of(low, k, size).astype(np.int16)
        b = frame_of(hi, k, size).astype(np.int16)
        d = np.abs(a - b)
        top = d[: size[1] * 4 // 5].mean()
        diff = Image.fromarray(np.uint8(np.clip(d * 10, 0, 255)))
        ImageDraw.Draw(diff).text((6, 4), f"#{k}  t {k / FPS:.2f}  mean {d.mean():.2f}  top {top:.2f}", fill=(255, 255, 0))
        rows.append(np.concatenate([a.astype(np.uint8), b.astype(np.uint8), np.asarray(diff)], 1))
        print(f"{name} #{k} t {k / FPS:.2f}: mean |low-hi| {d.mean():.2f}, top four fifths {top:.2f}", flush=True)
    os.makedirs(CHECK, exist_ok=True)
    path = os.path.join(CHECK, f"check_{name}.png")
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    print(path, flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "room":
        room()
    elif what in ("myth5", "myth7"):
        myth(what[-1])
    elif what == "encode":
        encode(sys.argv[2])
    elif what == "rainfix":
        rain_fix()
    elif what == "rainencode":
        rain_encode()
    elif what == "rainsheets":
        rain_sheets()
    elif what == "sheet":
        for n in sys.argv[2:]:
            sheet(n)
    else:
        raise SystemExit(__doc__)
