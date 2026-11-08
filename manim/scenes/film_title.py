"""十四's title beat: the poem has been read, and on the same sheet, under it, the film's name is written stroke by stroke: 字成精. Then the studio's stone, the one the ident brought down, comes down beside the signature from above the camera, presses, lifts and is gone, and leaves its print in cinnabar: the film's last strike, and the one red thing on paper in the film. The poem and the name are film_hand.py's; the stone, its fall and the print are film_ident.py's, the print read off the very maps the ident printed with.

    python film_title.py stills                 # hand/title_still_N.png: the poem's four states as they will stand, and hand/title_last.png, the last frame (1920 by 1080)
    python film_title.py print                  # hand/title_print.png: the print alone at three times the size it has on the sheet
    python film_title.py plan                   # hand/title3d/plan.json, for film_title3d.py: the seal's place and the stone's times
    python film_title.py test 132 145 160       # hand/title_test_N.png: frames of the beat from whatever title3d passes there are
    python film_title.py film                   # hand/title.mp4, the beat (ink_subs.END["title"] long, 1920 by 1080, 30 fps), and hand/title_last.png, its last frame

The camera looks straight down on the sheet and does not move. The frame is 48 cm of paper across, so the stone's 3 cm face prints SEAL pixels wide.
"""
import json
import os
import struct
import sys
import zlib

import numpy as np
from PIL import Image

import film_hand as hand
from film_hand import H, INK_RGB, OUT, W, save
from ink_grade import sm

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "head", "maps")
# the ident's maps: what they cover in cm (x0, x1, y0, y1), how many samples a cm, and the stone's side. Kept in step with film_ident.py by hand, which imports Blender's world
AREA, PX, SIDE = (-5.0, 5.0, -6.5, 3.5), 400, 3.0
MARGIN = 0.3                       # cm round the stone's face that the paste can creep into
HALO = np.array((250, 228, 180), np.float32)

# the beat, in seconds from its start (ink_subs.end_at("title")), ink_subs.END["title"] long: the name is written, a silence, and the stone. The stone's times are film_ident's pose and sunk in the same family: it lands, the paper gives and springs a little, it rests, it lifts and is gone
WRITE = (0.4, 3.2)
FALL, LAND, RISE, GONE = 3.9, 4.6, 5.1, 5.9
FPS = 30
TITLE3D = os.path.join(OUT, "title3d")          # film_title3d.py's passes, and what it is told


def read_png16(path):
    """A 16-bit RGB PNG as film_press_maps.png16 writes it (every row unfiltered), to 0-1 floats."""
    with open(path, "rb") as f:
        data = f.read()
    pos, idat, w, h = 8, [], 0, 0
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, colour = struct.unpack(">IIBB", body[:10])
            assert depth == 16 and colour == 2, (depth, colour)
        elif kind == b"IDAT":
            idat.append(body)
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(h, 1 + w * 6)
    assert not raw[:, 0].any(), "filtered rows"
    return np.frombuffer(raw[:, 1:].tobytes(), ">u2").reshape(h, w, 3).astype(np.float32) / 65535


_print = {}


def stone_print(side):
    """The ident's print, the stone's face and a little round it, at side pixels for the face: how much red, its colour, and the oil's halo round it. Returns (alpha, colour, halo, pad), pad the pixels added on each side."""
    if side in _print:
        return _print[side]
    x0, x1, y0, y1 = AREA
    half = SIDE / 2 + MARGIN
    c0, c1 = int(round((-half - x0) * PX)), int(round((half - x0) * PX))
    r0, r1 = int(round((y1 - half) * PX)), int(round((y1 + half) * PX))
    cover = read_png16(os.path.join(MAPS, "ink_cover.png"))[r0:r1, c0:c1]
    relief = read_png16(os.path.join(MAPS, "relief.png"))[r0:r1, c0:c1]
    colour = np.asarray(Image.open(os.path.join(MAPS, "ink_colour.png")).convert("RGB"), np.float32)[r0:r1, c0:c1]
    n = int(round(side * (SIDE + 2 * MARGIN) / SIDE))
    small = lambda a: np.asarray(Image.fromarray(np.float32(a), "F").resize((n, n), Image.LANCZOS), np.float32)
    red = np.clip(small(cover[..., 0]), 0, 1)
    halo = np.clip(small(relief[..., 2]), 0, 1)
    rgb = np.stack([small(colour[..., i]) for i in range(3)], -1)
    # under thin red the colour map is still the deep middle of the paste: weight it by the red, so the edge goes out to paper and not to a dark rim
    _print[side] = red, np.clip(rgb, 0, 255), halo, (n - side) // 2
    return _print[side]


def press_print(f, x, y, side, amount=1.0):
    """The stone's print into the frame f, its face's top left corner at x, y. amount: how much of it is there (0 to 1)."""
    red, rgb, halo, pad = stone_print(side)
    n = red.shape[0]
    X, Y = x - pad, y - pad
    part = f[Y:Y + n, X:X + n]
    part = part * (1 - 0.05 * halo[..., None] * amount) + HALO * 0.05 * halo[..., None] * amount
    part += (rgb - part) * (red * amount)[..., None]
    f[Y:Y + n, X:X + n] = part
    return f


def last(page=None):
    """The frame the beat ends on: the poem, the signature, the name, and the print beside the signature."""
    page = page or hand.Page()
    f = page.titled()
    x, y, s = page.seal_box()
    return press_print(f, x, y, s)


def plan():
    """What film_title3d.py needs to know and cannot work out in Blender: where the seal's place is in the frame, how many frame pixels a cm of the sheet is, and the stone's times."""
    x, y, s = hand.Page().seal_box()
    os.makedirs(TITLE3D, exist_ok=True)
    with open(os.path.join(TITLE3D, "plan.json"), "w") as f:
        json.dump(dict(seal=[int(x), int(y), int(s)], px_cm=s / SIDE, w=W, h=H, fps=FPS, fall=FALL, land=LAND, rise=RISE, gone=GONE), f, indent=1)
    print(os.path.join(TITLE3D, "plan.json"), flush=True)


# ---------------------------------------------------------------- the beat as a film, film_title3d.py's light over the sheet

def _linear(a):
    a = np.clip(a, 0, 1)
    return np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055) ** 2.4)


def _display(a):
    a = np.clip(a, 0, 1)
    return np.where(a <= .0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - .055)


def light(name):
    """A pass of film_title3d.py's at the frame's size, or None if it was not rendered."""
    path = os.path.join(TITLE3D, name + ".npy")
    if not os.path.exists(path):
        return None
    a = np.load(path).astype(np.float32)
    if a.shape[0] != H:
        if a.ndim == 2:
            a = np.asarray(Image.fromarray(a, "F").resize((W, H), Image.BILINEAR), np.float32)
        else:
            a = np.stack([np.asarray(Image.fromarray(a[..., i], "F").resize((W, H), Image.BILINEAR), np.float32) for i in range(3)], -1)
    return a


class Beat:
    """The beat's frames: at(t) for t from 0 to its length. The sheet is film_hand's, the name written on it by titling; the light on it is the stone's render over the bare sheet's, so where nothing happens it is the sheet exactly."""

    def __init__(self):
        self.page = hand.Page()
        self.writing = hand.titling(self.page, *WRITE)
        self.titled = self.page.titled()
        x, y, s = self.page.seal_box()
        self.printed = press_print(self.titled.copy(), x, y, s)
        self.paper = _linear(self.page.paper / 255)
        self.flat = np.maximum(light("flat"), 1e-4)
        self.left = light("left")

    def at(self, t):
        f = round(t * FPS)
        if t < WRITE[1] + 0.2:
            base = self.writing.at(t)
        else:
            base = self.printed if t >= LAND else self.titled
        if t < FALL - 0.5 / FPS:
            return base
        lit = light("f%04d" % f) if t < GONE else self.left
        if lit is None:
            raise FileNotFoundError("title3d pass for frame %d" % f)
        cover = light("a%04d" % f) if t < GONE else None
        under = _linear(base / 255)
        if cover is not None:
            a = np.clip(cover, 0, 1)[..., None]
            under = under * (1 - a) + self.paper * a
        return _display(under * (lit / self.flat)) * 255


def film():
    """hand/title.mp4, the whole beat at 1920 by 1080, and hand/title_last.png, its last frame (the stone gone, the print left)."""
    import ink_subs
    beat = Beat()
    secs = ink_subs.END["title"]
    path = os.path.join(OUT, "title.mp4")
    enc = hand.encoder(path, W, H, FPS)
    last = None
    for k in range(int(round(secs * FPS))):
        last = np.uint8(np.clip(beat.at(k / FPS) + 0.5, 0, 255))
        enc.stdin.write(last.tobytes())
        if k % 30 == 0:
            print(k, flush=True)
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError("encoding failed")
    for name in ("title_last",):
        old = os.path.join(OUT, name + ".png")
        if os.path.exists(old):
            n = next(n for n in range(1, 1000) if not os.path.exists(os.path.join(OUT, f"{name}_old{n}.png")))
            os.rename(old, os.path.join(OUT, f"{name}_old{n}.png"))         # never over an older one
    Image.fromarray(last).save(os.path.join(OUT, "title_last.png"))
    print(path, "%.1f s" % secs, flush=True)


def test(frames):
    """Single frames of the beat, as hand/title_test_N.png, from whatever passes there are."""
    beat = Beat()
    for f in frames:
        Image.fromarray(np.uint8(np.clip(beat.at(f / FPS) + 0.5, 0, 255))).save(os.path.join(OUT, "title_test_%d.png" % f))
    print(OUT, flush=True)


def stills():
    page = hand.Page()
    for n in (1, 2, 3):
        save(page.still(n), "title_still_%d" % n)
    save(page.finished(), "title_still_4")
    save(page.titled(), "title_still_5")
    save(last(page), "title_last")


def print_alone():
    page = hand.Page()
    x, y, s = page.seal_box()
    f = press_print(page.paper.copy(), x, y, s)
    pad = 40
    crop = np.uint8(np.clip(f[y - pad:y + s + pad, x - pad:x + s + pad] + 0.5, 0, 255))
    os.makedirs(OUT, exist_ok=True)
    Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3), Image.LANCZOS).save(os.path.join(OUT, "title_print.png"))
    print(os.path.join(OUT, "title_print.png"), flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "stills":
        stills()
    elif what == "print":
        print_alone()
    elif what == "plan":
        plan()
    elif what == "film":
        film()
    elif what == "test":
        test([int(v) for v in sys.argv[2:]])
    else:
        raise SystemExit(__doc__)
