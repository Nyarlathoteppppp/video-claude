"""The block of blue-grey clay of 三, 澄泥, and the seven dies pressed into it, in millimetres: world x to the right, y away from the camera, z up, the clay's untouched top at z = 0.

    python film_imprint_clay.py bake [--res 32] [--from k]      # state_0.npz .. state_7.npz, one per impression, and a hillshade of each, into ink/film_out/film/imprint/seven/. --from k carries on from state_k.npz
    python film_imprint_clay.py groove                          # state_g1.npz .. state_g3.npz: no die, the groove that goes on by itself out of the last state, the vision's stroke
    python film_imprint_clay.py board [--clean]                 # the old wooden board's textures; --clean: as arrays with no clay on it, for film_imprint_clawd.py
    python film_imprint_clay.py glyphs                          # crab/clay_glyphs.npz: the letters of clay at the little crab's feet, from this machine's fonts
    python film_imprint_clay.py dies                            # seq_h/die_k.npz: each die's face, for film_imprint3d.py's meshes
    python film_imprint_clay.py frames --blow k                 # seq_h/bk/: the height field frame by frame through the k-th blow (1 to 7), and how the die moves, for film_imprint_seq.py
    python film_imprint_clay.py groove --frames                 # also seq_h/g/: the groove, frame by frame
    python film_imprint_clay.py sheet name                      # seven/seq_press/sheet_name.png: eight frames of a segment

Everything happens to a height field and nothing else. A die is a surface coming down: wherever the clay stands above it, the clay is pushed down to it, and that volume goes somewhere. Part of it runs out to the sides
of each stroke and stands up as a lip, part swells the ground around, and it flows sideways as it goes, which is what drags and bends the marks already there. Old grooves under a new die are not erased: what is pressed out of
their walls fills them, so they close up and shallow, and what stands proud is cut. The spreading is done with cosine transforms, so its reflection at the block's edges is exact and no volume is lost: bake prints the books.
What the block loses from its top by sagging under a heavy die goes out through its sides, as a bulge film_imprint3d.py gives the sides. Where the strain is too much for it the clay tears: cracks walked across the height field, cut into it, their volume in their own lips.

Importing this needs only numpy. film_imprint3d.py renders what this bakes; 八「河床」 comes back to the same last state, state_path(LAST).
Everything of the seven blows lives in seven/ (states, seq_h, stills), so that the five-blow states still in imprint/ stay as they were for anyone reading them until they move over.
Times here are seconds from the start of 三 (T3): the cut list adds 三's start from ink_subs.
"""
import glob
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "ink", "film_out", "film", "imprint")
DIES = os.path.join(OUT, "dies")
SEVEN = os.path.join(OUT, "seven")


def state_path(k):
    """Where state k of the seven blows is: 0 the untouched block, 1..7 after each blow, g1..g3 the groove, wo the 我."""
    return os.path.join(SEVEN, f"state_{k}.npz")

A, B, THICK = 120.0, 80.0, 40.0     # half width, half depth, height of the block
EDGE_R = 4.0                        # the hand-cut top edge's radius
WORKERS = 12

# ---- the seven dies -----------------------------------------------------------------------------------------------------------------------------------
# In the order of 七击, oldest first. The tablet's, the letter's, the poem's, the quarto's and the ruling's faces are cut from the carriers film_docs.py writes (dies/NAME.npz); the
# code's and the post's are drawn here.
LETTER = ("疾患以來漸就衰損", "汝輩稚小家貧每役", "柴水之勞何時可免", "念之在心若何可言")                             # 陶渊明《与子俨等疏》, right to left
POEM = ("國破山河在", "城春草木深", "感時花濺淚", "恨別鳥驚心", "烽火連三月", "家書抵萬金", "白頭搔更短", "渾欲不勝簪")      # 杜甫《春望》
RULING = ("平政院裁決書", "主文", "教育部之處分取消之")                                                          # placeholder until the real die comes: only the main text, which is checked
CODE = ("def gcd(a, b):", "    while b:", "        a, b = b, a % b", "    return a", "print(gcd(1071, 462))")      # 辗转相除法, gcd(1071, 462) = 21
POST = ("今天又说肚子疼，没有去上学。", "其实不疼。", "可是说不出是为什么。", "写在这里，反正也没有人看。")             # film_weibo.py's post, without its lines that say 我: that word belongs to 三's end
POST_CHARS, POST_EM, POST_LEAD = 9, 22.0, 26.0                                                                      # set solid across the plate, big and close: nothing of the plate's face goes unprinted
for _text in (LETTER, POEM, RULING, POST):
    assert not any("我" in s for s in _text)

# size (x, y) of the printing face; centre on the block, rotation in degrees (anticlockwise seen from above); how far the lowest part of the face goes below the surface there; relief: strokes above the die's ground
# sheet: where the face comes from on its carrier: "run" a column of it, "line" a line of it, a number the margin round everything written; crest: how much of the carrier's stroke profile is kept
# shape: the face's departure from flat. bend mm at the ends of x and y, tilt mm per 100 mm about (tx, ty). rock: the plate lands on one end, rolls and skids over as the other comes down
# flow, shear: how far the marks already there are carried, by the volume this die drives out, and by the clay's own uneven give (amount in mm, two scales). collapse: how much of the old relief is squeezed out of them
# sag: how much the block's top goes down under this die, as a whole. dish: the wide hollow around the die. relax, slump: what the wet clay does once the die has gone. tears: cracks at the worst of the strain (count, widest, longest, in the block's rim)
PRESSES = [
    dict(name="tablet", size=(204.0, 40.0), sheet="line", crest=0.0, centre=(-4.0, 46.0), rot=-1.5, depth=2.5, relief=4.0, grain=.02, grain_scale=(2.0, 2.0), uneven=.05, sag=.08, dish=.22, burr=.05, flow=2.4, relax=((.05, 2.2), (.02, .7)), slump=3.6),
    dict(name="letter", size=(70.0, 110.0), sheet="run", crest=.45, centre=(-62.0, -14.0), rot=0.0, depth=2.7, relief=4.6, grain=.03, grain_scale=(.45, 5.0), uneven=.1, sag=.12, dish=.3, burr=.06, flow=3.0, shear=((.8, 9.0), (.9, 26.0)), collapse=.15, relax=((.07, 2.2), (.02, .7)), slump=3.2),
    dict(name="poem", size=(150.0, 94.0), sheet=8.0, crest=.4, centre=(27.0, -4.0), rot=-12.0, depth=2.9, relief=4.0, grain=.04, grain_scale=(6.0, .5), uneven=.12, sag=.2, dish=.45, burr=.07, flow=7.0, shear=((1.4, 9.0), (1.6, 26.0)), collapse=.3, relax=((.16, 2.2), (.05, .7)), slump=2.6, tears=(24, .3, 9.0, 0)),
    dict(name="quarto", size=(118.0, 96.0), sheet=2.5, crest=.5, centre=(54.0, 22.0), rot=4.0, depth=2.3, relief=3.2, grain=.003, grain_scale=(2.0, 2.0), uneven=.03, sag=.18, dish=.35, burr=.035, flow=6.0, shear=((1.2, 9.0), (1.4, 26.0)), collapse=.35, relax=((.12, 2.2), (.04, .7)), slump=2.5, tears=(18, .28, 8.0, 0)),
    dict(name="ruling", size=(150.0, 108.0), sheet=3.0, crest=.7, text_level=.5, centre=(-7.0, 0.0), rot=6.0, depth=3.6, relief=4.2, grain=.004, grain_scale=(2.0, 2.0), uneven=.03, sag=.35, dish=.6, burr=.03, flow=8.0, shear=((2.0, 10.0), (2.2, 30.0)), collapse=.5, relax=((.2, 2.2), (.06, .7)), slump=2.2, tears=(70, .4, 14.0, 10)),
    dict(name="code", size=(216.0, 36.0), centre=(5.0, -31.0), rot=-3.0, depth=.6, relief=.84, conform=4.0, shoulder=.08, grain=.002, grain_scale=(9.0, .3), uneven=.012, sag=.12, dish=.15, burr=.004, flow=2.5, shear=((.3, 8.0), (.3, 20.0)), collapse=.3),
    dict(name="post", size=(206.0, 134.0), centre=(10.0, -6.0), rot=2.5, depth=6.5, relief=3.0, grain=.006, grain_scale=(3.0, 3.0), uneven=.05, waver=.55, bend=(7.0, 7.0), low=(-6.0, -6.0), dents=((-80.0, 41.0, 17.0, 7.5), (30.0, 34.0, 13.0, 6.5), (66.0, -32.0, 34.0, 8.5)),
         rock=dict(n=12, fine=5, skid=(3.0, 4.0), spin=4.5, pivot=(-130.0, 95.0), tilt=((3.6, 2.2), (-2.8, -1.8)), sink=(.5, 1.0, .72), grip=.8), ghost=.45, squeeze=.7, sag=2.2, dish=3.0, burr=.03, flow=11.0, shear=((3.0, 11.0), (3.4, 34.0)), collapse=.5,
         relax=((2.0, 14.0), (.8, 5.0), (.25, 1.8), (.07, .6)), slump=1.8, tears=(420, .55, 24.0, 46)),
]
LAST = len(PRESSES)


def sm(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def spectral(n, beta, seed, m=None):
    """Tileable noise with a 1/f^beta spectrum, mean 0, std 1."""
    from scipy.fft import irfft2
    m = m or n
    r = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(m)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1
    s = (r.standard_normal(f.shape, dtype=np.float32) + 1j * r.standard_normal(f.shape, dtype=np.float32)) / f ** (beta / 2)
    s[0, 0] = 0
    v = irfft2(s, s=(n, m), workers=WORKERS)
    return ((v - v.mean()) / v.std()).astype(np.float32)


def blobs(ny, nx, res, scale_mm, seed):
    """Smooth noise whose features are about scale_mm across and which has nothing finer, std 1. A 1/f spectrum can't do this: it stays rough all the way down, and whatever a die's unevenness is made of, the clay takes."""
    from scipy.fft import irfft2
    r = np.random.default_rng(seed)
    n, m = 2 * ny, 2 * nx
    fy, fx = np.fft.fftfreq(n, 1 / res)[:, None], np.fft.rfftfreq(m, 1 / res)[None, :]
    k = np.exp(-((fx * fx + fy * fy) * (scale_mm * 1.2) ** 2)).astype(np.float32)
    s = (r.standard_normal(k.shape, dtype=np.float32) + 1j * r.standard_normal(k.shape, dtype=np.float32)) * k
    v = irfft2(s, s=(n, m), workers=WORKERS)[:ny, :nx]
    return ((v - v.mean()) / v.std()).astype(np.float32)


def font(key, px):
    from PIL import ImageFont
    name, index = dict(kai=("Kaiti.ttc", 2), kaib=("Kaiti.ttc", 4), kaik=("Kaiti.ttc", 6), song=("Songti.ttc", 7), songb=("Songti.ttc", 2), mono=("Menlo.ttc", 0), ping=("PingFang.ttc", 11))[key]
    paths = [f"/System/Library/Fonts/Supplemental/{name}", f"/System/Library/Fonts/{name}"] + glob.glob(f"/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/{name}")
    return ImageFont.truetype(next(p for p in paths if os.path.exists(p)), int(round(px)), index=index)


# ---- the faces of the dies ----------------------------------------------------------------------------------------------------------------------------
def glyph(canvas, ch, f, x, y, res, rot=0.0, anchor="mm"):
    """Pastes one character at (x, y) in mm from the die's top left, turned a little."""
    from PIL import Image, ImageChops, ImageDraw
    s = int(f.size * 1.7) | 1
    t = Image.new("L", (s, s), 0)
    ImageDraw.Draw(t).text((s / 2, s / 2), ch, 255, font=f, anchor=anchor)
    if rot:
        t = t.rotate(rot, resample=Image.BICUBIC)
    l, u = int(x * res - s / 2), int(y * res - s / 2)
    canvas.paste(ImageChops.lighter(canvas.crop((l, u, l + s, u + s)), t), (l, u))


def placeholder_mask(spec, res):
    """Until the real faces come from the document shots: the same words, in the fonts nearest to them. Returns a 0-1 array, row 0 at the die's far edge (+y), the words reading the right way round on the clay."""
    from PIL import Image, ImageDraw
    w, h = spec["size"]
    im = Image.new("L", (int(w * res), int(h * res)), 0)
    r = np.random.default_rng(hash_name(spec["name"]))
    name = spec["name"]
    if name == "letter":
        em, step = 12.4, 13.0
        f = font("kaik", em * res)
        for c, column in enumerate(LETTER):
            x = w - 13.0 - c * 14.8
            for i, ch in enumerate(column):
                glyph(im, ch, f, x + r.normal(0, .35), 7.5 + i * step + r.normal(0, .3), res, rot=float(r.normal(0, 1.4)))
    elif name == "poem":
        em, step = 12.6, 14.6
        f = font("kaib", em * res)
        d = ImageDraw.Draw(im)
        d.rectangle([2.2 * res, 2.2 * res, (w - 2.2) * res, (h - 2.2) * res], outline=255, width=int(.8 * res))
        for c, column in enumerate(POEM):
            x = w - 12.0 - c * 14.3
            for i, ch in enumerate(column):
                glyph(im, ch, f, x + r.normal(0, .25), 9.5 + i * step + r.normal(0, .25), res, rot=float(r.normal(0, 1.0)))
    elif name == "ruling":
        d = ImageDraw.Draw(im)
        d.rectangle([3 * res, 3 * res, (w - 3) * res, (h - 3) * res], outline=255, width=int(1.3 * res))
        d.rectangle([5 * res, 5 * res, (w - 5) * res, (h - 5) * res], outline=255, width=int(.45 * res))
        n = 13
        pitch = (w - 10) / n
        for c in range(1, n):
            x = (5 + c * pitch) * res
            d.line([x, 5 * res, x, (h - 5) * res], fill=255, width=max(int(.55 * res), 1))
        f = font("songb", 8.2 * res)
        for c in range(n):
            column = RULING[c % len(RULING)]
            x = w - 5 - (c + .5) * pitch
            for i, ch in enumerate(column):
                glyph(im, ch, f, x, 12 + i * 9.6, res)
    elif name == "code":
        f = font("mono", .66 * res)
        lead = .88
        d = ImageDraw.Draw(im)
        for row in range(int(h / lead) - 1):
            k = row
            x = 1.4
            while x < w:
                line = CODE[k % len(CODE)]
                d.text((x * res, (1.0 + row * lead) * res), line, 255, font=f, anchor="ls")
                x += f.getlength(line) / res + .7
                k += 1
    elif name == "post":
        f = font("ping", POST_EM * res)
        text = "".join(POST)
        for i, ch in enumerate(text):
            row, col = divmod(i, POST_CHARS)
            glyph(im, ch, f, 4 + (col + .5) * POST_EM, 16 + row * POST_LEAD, res)
    a = np.asarray(im, np.float32) / 255
    if name == "letter":
        from scipy.ndimage import gaussian_filter
        a = gaussian_filter(a, .05 * res)
    return a


def hash_name(name):
    return sum(ord(c) * (i + 1) for i, c in enumerate(name))


def sheet_face(path, spec, res):
    """A piece of the sheet the document shots photograph, as a die's printing face from 0 to 1: its own characters, the same strokes, at the scale this die is cut.
    The letter's die is centred on the column that must still read, and leaves off the characters its edge would cut through. The others take all that is written on the sheet."""
    from scipy.ndimage import zoom
    import json
    z = np.load(path)
    height, px = z["height"], 1 / float(z["mm_per_px"])
    w, h = spec["size"]
    if spec["sheet"] in ("run", "line"):
        run = np.array(json.loads(str(z["keys"]))["run"])
        y0, y1 = run[:, 1].min(), run[:, 3].max()
        scale = (h - 5.0) / (y1 - y0)
        if spec["sheet"] == "line":         # a whole line of signs, as large as the face holds it
            scale = min(scale, (w - 6.0) / (run[:, 2].max() - run[:, 0].min()))
        cx, cy = (run[:, 0].min() + run[:, 2].max()) / 2, (y0 + y1) / 2
        x0, x1, y0, y1 = cx - w / 2 / scale, cx + w / 2 / scale, cy - h / 2 / scale, cy + h / 2 / scale
        keep = np.zeros_like(height)
        for c in json.loads(str(z["chars"])):
            a, b, e, f = c["box"]
            if a >= x0 and b >= y0 and e <= x1 and f <= y1:
                l, u, r, d = int((a - .4) * px), int((b - .4) * px), int((e + .4) * px), int((f + .4) * px)
                keep[u:d, l:r] = height[u:d, l:r]
        height = keep
    else:
        ys, xs = np.nonzero(height > 0)
        m = spec["sheet"]
        x0, x1, y0, y1 = xs.min() / px - m, xs.max() / px + m, ys.min() / px - m, ys.max() / px + m
        scale = w / (x1 - x0)
    print(f"  sheet ({x0:.1f}, {y0:.1f}) to ({x1:.1f}, {y1:.1f}), scale {scale:.3f}, die {(x1 - x0) * scale:.0f} x {(y1 - y0) * scale:.0f} mm", flush=True)
    # the sheet's height slopes out from each stroke by the same amount whatever the scale. Taking only its upper part keeps fine strokes from running together when the die is cut smaller than the sheet
    a = sm((height / float(height.max()) - spec["crest"]) / (1 - spec["crest"]))
    # small type stands lower on its plate than the rules do: a character 5 mm high can't go as deep as a cut across the whole plate
    if "text_level" in spec:
        level = np.ones_like(a)
        for c in json.loads(str(z["chars"])):
            l, u, r, d = [int(v * px) for v in c["box"]]
            level[max(u - 8, 0):d + 8, max(l - 8, 0):r + 8] = spec["text_level"]
        seal = json.loads(str(z["keys"])).get("seal")
        if seal:
            l, u, r, d = [int(v * px) for v in seal]
            level[max(u, 0):d, max(l, 0):r] = spec["text_level"]
        a = a * level
    a = a[max(int(y0 * px), 0):int(y1 * px), max(int(x0 * px), 0):int(x1 * px)]
    return np.clip(zoom(a, res * scale / px, order=1), 0, 1).astype(np.float32)


def die_face(spec, res):
    """The die's face as relief above its strokes, in mm, on its own grid: 0 on the printing surface, spec["relief"] on the ground, and the die's own material in both: the grain of its wood, the marks of the chisel."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    name, r_h = spec["name"], spec["relief"]
    path = os.path.join(DIES, name + ".npz")
    w, h = spec["size"]
    if os.path.exists(path) and "sheet" in spec:
        face = sheet_face(path, spec, res)
        print(name, "real face from the document", flush=True)
    else:
        face = np.clip(distance_transform_edt(placeholder_mask(spec, res) >= .5) / res / spec.get("shoulder", .3), 0, 1).astype(np.float32)
        print(name, "placeholder face", flush=True)
    if spec.get("fatten"):
        from scipy.ndimage import maximum_filter
        face = maximum_filter(face, size=int(2 * spec["fatten"] * res) | 1)
    ny, nx = face.shape
    rel = r_h * (1 - face)
    n = max(nx, ny)
    seed = hash_name(name)
    if spec["uneven"]:
        rel = rel + (blobs(ny, nx, res, 22.0, seed + 1) * spec["uneven"] + spec["uneven"])
    # the material itself, on faces and ground alike
    if spec["grain"]:
        sx, sy = spec["grain_scale"]
        tile = spectral(4096, 1.4, seed + 2)
        yy, xx = np.mgrid[:ny, :nx]
        gr = tile[(yy * sy / res * 8).astype(np.int32) % 4096, (xx * sx / res * 8).astype(np.int32) % 4096]
        rel = rel + gaussian_filter(gr, .03 * res) * spec["grain"]
    # the ground of a carved block keeps the cutter's stray ridges, thin and along the grain. A few stand tall enough to print.
    if name in ("letter", "poem"):
        from PIL import Image, ImageDraw
        rr = np.random.default_rng(seed + 9)
        im = Image.new("L", (nx, ny), 0)
        d = ImageDraw.Draw(im)
        along = math.pi / 2 if name == "letter" else 0.0
        for _ in range(int(w * h / 200)):
            x0, y0, L, a = rr.uniform(0, nx), rr.uniform(0, ny), rr.uniform(1.5, 7.0) * res, along + rr.normal(0, .3)
            d.line([x0, y0, x0 + L * math.cos(a), y0 + L * math.sin(a)], fill=int(rr.uniform(70, 255)), width=max(int(rr.uniform(.2, .45) * res), 1))
        g = gaussian_filter(np.asarray(im, np.float32) / 255, .05 * res)
        rel = rel + sm(g / .35) * np.minimum(spec["depth"] - .34 * g - rel, 0)
    # a plate that has been dropped and hammered flat more than once doesn't touch evenly
    if spec.get("waver"):
        rel = rel + (blobs(ny, nx, res, 42.0, seed + 4) * .85 + blobs(ny, nx, res, 18.0, seed + 5) * .42) * spec["waver"]
    for dx, dy, rad, height in spec.get("dents", ()):
        yy, xx = np.ogrid[:ny, :nx]
        rel = rel + height * np.exp(-((((xx + .5) / res - w / 2 - dx) ** 2 + (h / 2 - (yy + .5) / res - dy) ** 2) / (rad * rad))).astype(np.float32)
    # the edge of the die: a small chamfer on all four sides
    yy, xx = np.ogrid[:ny, :nx]
    to_edge = np.minimum.reduce([(xx + .5) / res * np.ones((ny, nx), np.float32), (nx - xx - .5) / res * np.ones((ny, nx), np.float32), (yy + .5) / res * np.ones((ny, nx), np.float32), (ny - yy - .5) / res * np.ones((ny, nx), np.float32)])
    rel = rel + np.clip(.7 - to_edge, 0, .7) * 1.4
    return rel.astype(np.float32)


# ---- the physics ------------------------------------------------------------------------------------------------------------------------------------
class Field:
    """Transforms for the block's grid, reflecting at its edges. spread() moves volume outwards without losing any of it."""

    def __init__(self, ny, nx, res):
        self.res = res
        fy = np.arange(ny, dtype=np.float32)[:, None] / (2 * ny / res)
        fx = np.arange(nx, dtype=np.float32)[None, :] / (2 * nx / res)
        self.f2 = fy * fy + fx * fx

    def kernel(self, sigmas, weights):
        return sum(w * np.exp(-2 * np.pi ** 2 * s * s * self.f2) for s, w in zip(sigmas, weights)).astype(np.float32)

    def potential(self, sigmas, weights):
        """What a volume spread by this kernel moved through on its way: its gradient is the sideways flow."""
        return sum(w * s * s / 2 * np.exp(-2 * np.pi ** 2 * s * s * self.f2) for s, w in zip(sigmas, weights)).astype(np.float32)

    def apply(self, a, k):
        from scipy.fft import dctn, idctn
        return idctn(dctn(a, type=2, norm="ortho", workers=WORKERS) * k, type=2, norm="ortho", workers=WORKERS).astype(np.float32)

    def blur(self, a, sigma):
        return self.apply(a, np.exp(-2 * np.pi ** 2 * sigma * sigma * self.f2).astype(np.float32))


def bilerp(a, rows, cols):
    """a at fractional (rows, cols), clamped to its edges. Much faster than map_coordinates on this size of grid."""
    ny, nx = a.shape
    r0, c0 = np.clip(np.floor(rows).astype(np.int32), 0, ny - 2), np.clip(np.floor(cols).astype(np.int32), 0, nx - 2)
    fr, fc = np.clip(rows - r0, 0, 1).astype(np.float32), np.clip(cols - c0, 0, 1).astype(np.float32)
    return (a[r0, c0] * (1 - fr) + a[r0 + 1, c0] * fr) * (1 - fc) + (a[r0, c0 + 1] * (1 - fr) + a[r0 + 1, c0 + 1] * fr) * fc


def virgin(res):
    """The top of the block before anything happens: smoothed by hand with a wooden rib, a little uneven, grit in it, a few air pits. Its edges come down in a soft round to where the sides begin."""
    nx, ny = int(2 * A * res), int(2 * B * res)
    n = max(nx, ny)
    yy, xx = np.mgrid[:ny, :nx]
    X, Y = (xx + .5) / res - A, B - (yy + .5) / res
    H = blobs(ny, nx, res, 55.0, 1) * .2 + blobs(ny, nx, res, 17.0, 9) * .055 + blobs(ny, nx, res, 5.0, 10) * .018 + blobs(ny, nx, res, 1.5, 11) * .006 + blobs(ny, nx, res, .6, 12) * .003
    # the rib's strokes: long, shallow, overlapping, in two main directions
    tile, bow = blobs(4096, 4096, 1, 5.0, 2), spectral(1024, 5.0, 8)
    for ang, amp, seed_shift, sx, sy in ((-7.0, .012, 0, .09, 1.7), (11.0, .007, 977, .14, 2.6)):
        a = math.radians(ang)
        u, v = X * math.cos(a) + Y * math.sin(a), -X * math.sin(a) + Y * math.cos(a)
        v = v + bilerp(bow, (Y * .4 + seed_shift) % 1022, (X * .4 + seed_shift) % 1022) * 7.0
        band = bilerp(tile, (v * sy + seed_shift) % 4094, (u * sx + seed_shift) % 4094)
        H = H + band * amp
    H = H + spectral(n, 1.15, 3)[:ny, :nx] * .0012
    # air pits, and grit dragged along by the rib
    r = np.random.default_rng(5)
    for _ in range(900):
        cx, cy, rad = r.uniform(-A, A), r.uniform(-B, B), float(r.choice((r.uniform(.05, .14), r.uniform(.14, .34)), p=(.8, .2)))
        c, ro = int((cx + A) * res), int((B - cy) * res)
        s = max(int(rad * res * 2), 2)
        t, b, l, m = max(ro - s, 0), min(ro + s + 1, ny), max(c - s, 0), min(c + s + 1, nx)
        d = np.hypot(yy[t:b, l:m] - ro, xx[t:b, l:m] - c) / (rad * res)
        H[t:b, l:m] -= (sm(1 - d) * rad * r.uniform(.35, .7)).astype(np.float32)
    # the rounded edge, with hand-cut unevenness
    wob = spectral(4096, 2.6, 6, 8)[:, 0]
    d = np.minimum(A - np.abs(X), B - np.abs(Y))
    corner = np.hypot(np.clip(A - np.abs(X), 0, 8), np.clip(B - np.abs(Y), 0, 8))
    d = np.where((A - np.abs(X) < 8) & (B - np.abs(Y) < 8), np.minimum(d, corner * .94), d)
    R = EDGE_R * (1 + .22 * wob[((X + Y) * 6).astype(np.int32) % 4096])
    t = np.clip(d / R, 0, 1)
    H = H - R * (1 - np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1)))
    return crazing(H, res, 70).astype(np.float32)


def die_surface(face, spec, X, Y, die_res, dx=0.0, dy=0.0, spin=0.0, tilt=None):
    """The die's face over the block's grid, as height above its own lowest point, wherever it now is. Also which part of the grid it covers."""
    h, w = face.shape[0] / die_res, face.shape[1] / die_res
    cx, cy = spec["centre"]
    if spin:
        px, py = spec["rock"]["pivot"]
        c, n = math.cos(math.radians(spin)), math.sin(math.radians(spin))
        cx, cy = px + (cx - px) * c - (cy - py) * n, py + (cx - px) * n + (cy - py) * c
    cx, cy = cx + dx, cy + dy
    a = math.radians(spec["rot"] + spin)
    lx, ly = (X - cx) * math.cos(a) + (Y - cy) * math.sin(a), -(X - cx) * math.sin(a) + (Y - cy) * math.cos(a)
    inside = (np.abs(lx) < w / 2) & (np.abs(ly) < h / 2)
    rel = bilerp(face, (h / 2 - ly) * die_res - .5, (lx + w / 2) * die_res - .5)
    # the plate is domed, and doesn't come down square, so its pressure is nothing like even
    if "bend" in spec:
        tx, ty = spec.get("low", (0.0, 0.0))
        bx, by = spec["bend"]
        rel = rel + (bx * ((lx - tx) / (w / 2)) ** 2 + by * ((ly - ty) / (h / 2)) ** 2).astype(np.float32)
        if tilt:
            rel = rel + ((lx - tx) * tilt[0] / 100 + (ly - ty) * tilt[1] / 100).astype(np.float32)
    rel = rel - float(rel[inside].min())
    outside = np.hypot(np.maximum(np.abs(lx) - w / 2, 0), np.maximum(np.abs(ly) - h / 2, 0))
    return (rel + outside * 8.0).astype(np.float32), inside


def poses(spec):
    """Where a rocking plate is at each moment of its landing, as it skids, turns, and rolls from its first corner to its last. It doesn't go smoothly: it sticks and lets go."""
    R = spec["rock"]
    out = []
    T = R["n"] * R["fine"]
    for i in range(T):
        t = (i + .5) / T
        e = float(np.clip(sm(t) + .035 * math.sin(t * 2 * math.pi * 3.3 + .8), 0, 1))
        (ax, ay), (bx, by) = R["tilt"]
        out.append(dict(dx=R["skid"][0] * e, dy=R["skid"][1] * e, spin=R["spin"] * e, tilt=(ax + (bx - ax) * e, ay + (by - ay) * e), depth=spec["depth"] * float(np.interp(t, (0, .45, 1), R["sink"]))))
    return out


def swirl(ny, nx, res, pairs, seed):
    """A displacement that moves nothing in or out, which is what uneven give in a soft body comes to: for each (amount in mm, scale in mm), the curl of smooth noise, at that amount. Returns it along columns and rows."""
    U, V = 0.0, 0.0
    for i, (amount, scale) in enumerate(pairs):
        gr, gc = np.gradient(blobs(ny, nx, res, scale, seed + i), 1.0 / res)
        norm = amount / float(np.quantile(np.hypot(gr[::8, ::8], gc[::8, ::8]), .98))
        U, V = U + gr * norm, V - gc * norm
    return U, V


def walk_cracks(im, seeds, res, steer, width_max, r):
    """Walks each crack across the block from its seed, unsteadily, thinning as it goes, now and then forking, and draws it into im, brighter the wider."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    ny, nx = im.size[1], im.size[0]
    k = steer.shape[0] / ny
    stack = list(seeds)
    while stack:
        x, y, th, w, L, forks = stack.pop()
        s, ds = 0.0, .2
        px, py = x, y
        kink, run = 0.0, 0.0
        while s < L:
            row, col = int(np.clip(py * k, 0, steer.shape[0] - 1)), int(np.clip(px * k, 0, steer.shape[1] - 1))
            if run <= 0:        # clay doesn't tear in curves: it goes straight, then it finds another way
                kink, run = r.normal(0, .62), r.uniform(.3, 1.6)
            a = th + float(steer[row, col]) * .6 + kink
            run -= ds
            qx, qy = px + math.cos(a) * ds * res, py + math.sin(a) * ds * res
            wn = w * (1 - s / L) ** .65
            d.line([px, py, qx, qy], fill=int(np.clip(255 * (wn / width_max) ** .5, 40, 255)), width=max(int(round(wn * res)), 1))
            if forks and r.random() < ds * .07:
                stack.append((qx, qy, a + r.choice((-1, 1)) * r.uniform(.4, .95), wn * .62, (L - s) * r.uniform(.35, .7), forks - 1))
            px, py, s = qx, qy, s + ds


def crazing(H, res, seed):
    """It has dried on top before anyone came to it: hairlines, too fine to see from across the room, in some places and not in others. The first thing up close that says earth and not felt."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    ny, nx = H.shape
    r = np.random.default_rng(seed)
    steer = blobs(ny // 8, nx // 8, res / 8, 18.0, seed + 1)
    im = Image.new("L", (nx, ny), 0)
    seeds = [(float(r.uniform(0, nx)), float(r.uniform(0, ny)), float(r.uniform(-math.pi, math.pi)), float(r.uniform(.04, .09)), float(r.uniform(5.0, 26.0)), 2) for _ in range(260)]
    walk_cracks(im, seeds, res, steer, .09, r)
    c = gaussian_filter(np.asarray(im, np.float32) / 255, .02 * res)
    return H - c * .1 * (.25 + .75 * sm((blobs(ny, nx, res, 30.0, seed + 2) + .5) / 1.2))


def tear(H, F, spec, field, res):
    """The clay gives way where the strain was worst: from the top of each wall the die drove down, mostly away across the ground, some along the wall, and from the rim of the block inwards where the block was squeezed out sideways."""
    from PIL import Image
    n, width, length, rim = spec["tears"]
    ny, nx = H.shape
    seed = hash_name(spec["name"]) + 90
    r = np.random.default_rng(seed)
    k = res // 2
    gr, gc = np.gradient(field.blur(H, .9)[::k, ::k], k / res)
    mag = np.hypot(gr, gc) * np.maximum(field.blur(F, 2.2)[::k, ::k], .03)
    p = (mag ** 2 / float((mag ** 2).sum())).ravel()
    at = r.choice(p.size, n, p=p)
    seeds = []
    for i in at:
        row, col = np.unravel_index(i, mag.shape)
        out = math.atan2(gr[row, col], gc[row, col])
        kind = r.random()
        th = out + (math.pi if r.random() < .5 else 0) + r.normal(0, .35) if kind < .72 else out + math.pi / 2 * r.choice((-1, 1)) + r.normal(0, .25)
        seeds.append((col * k + k / 2, row * k + k / 2, th, width * float(r.uniform(.3, 1) ** 1.6), length * float(r.uniform(.12, 1) ** 1.8), 2))
    for _ in range(rim):
        s = r.uniform(0, 2 * (nx + ny))
        if s < nx:
            x, y, th = s, 3.0, math.pi / 2
        elif s < nx + ny:
            x, y, th = nx - 3.0, s - nx, math.pi
        elif s < 2 * nx + ny:
            x, y, th = s - nx - ny, ny - 3.0, -math.pi / 2
        else:
            x, y, th = 3.0, s - 2 * nx - ny, 0.0
        seeds.append((x, y, th + r.normal(0, .3), width * float(r.uniform(.45, 1)), length * float(r.uniform(.25, .8)), 1))
    steer = blobs(ny // 8, nx // 8, res / 8, 7.0, seed + 1)
    im = Image.new("L", (nx, ny), 0)
    walk_cracks(im, seeds, res, steer, width, r)
    c = field.blur(np.asarray(im, np.float32) / 255, .035).astype(np.float32)
    cut = c * width * 2.6
    lips = field.kernel((.2, .55, 1.8), (.45, .35, .2))
    print(f"  {len(seeds)} tears, {float(cut.sum()) / res ** 2:.0f} mm3 opened", flush=True)
    return H - cut + field.apply(cut, lips)


def slump(H, res, weak, seed, field):
    """Walls too steep for wet clay to hold give way, each bit of wall by its own strength, and come to rest at their foot. weak is the slope (rise over run) the weaker parts hold."""
    ny, nx = H.shape
    for grid, iters in ((8, 26), (16, 26)):
        k = res // grid
        my, mx = ny // k, nx // k
        h0 = H[:my * k, :mx * k].reshape(my, k, mx, k).mean(axis=(1, 3))
        h = h0.copy()
        strong = blobs(my, mx, grid, 3.0, seed + grid) * .7 + blobs(my, mx, grid, .9, seed + grid + 1) * .5
        hold = (weak * (1 + np.maximum(strong, -.8) * .75) / grid).astype(np.float32)
        hx, hy = (hold[:, 1:] + hold[:, :-1]) / 2, (hold[1:] + hold[:-1]) / 2
        for _ in range(iters):
            d = h[:, 1:] - h[:, :-1]
            e = np.sign(d) * np.maximum(np.abs(d) - hx, 0) * .2
            h[:, 1:] -= e
            h[:, :-1] += e
            d = h[1:] - h[:-1]
            e = np.sign(d) * np.maximum(np.abs(d) - hy, 0) * .2
            h[1:] -= e
            h[:-1] += e
        yy, xx = np.ogrid[:ny, :nx]
        H = H + field.blur(bilerp((h - h0).astype(np.float32), np.broadcast_to((yy + .5) / k - .5, (ny, nx)), np.broadcast_to((xx + .5) / k - .5, (ny, nx))), .6 * k / res)
    return H


def grit(ny, nx, res):
    """The clay's grain: what makes a dry surface look dry at any distance. The dies don't move it: it is what the material is made of, so it goes on every state as it is saved."""
    closed = .4 + .6 * sm((blobs(ny, nx, res, 26.0, 34) + .7) / 1.4)        # where the rib went over it most the surface has closed up; left alone evenly it reads as felt
    return (blobs(ny, nx, res, .07, 30) * .0026 + blobs(ny, nx, res, .16, 31) * .0058 + blobs(ny, nx, res, .42, 32) * .0052) * closed + blobs(ny, nx, res, 1.3, 33) * .0036


def press(H, spec, field, res, die_res, rec=None):
    """One impression. Returns the new surface, where the die touched, and the volume the block lost through its sides by sagging. rec, if there is one, is shown each stage on the way: it changes nothing."""
    t0 = time.time()
    ny, nx = H.shape
    v0 = float(H.sum()) / res ** 2
    seed = hash_name(spec["name"])
    face = die_face(spec, die_res)
    yy, xx = np.mgrid[:ny, :nx]
    X, Y = (xx + .5) / res - A, B - (yy + .5) / res
    rock = poses(spec) if "rock" in spec else None
    peak = max(rock, key=lambda q: q["depth"]) if rock else dict(dx=0.0, dy=0.0, spin=0.0, tilt=None, depth=spec["depth"])
    rel, inside = die_surface(face, spec, X, Y, die_res, peak["dx"], peak["dy"], peak["spin"], peak["tilt"])
    base = float(np.median(H[inside]))      # how deep it goes is measured from where this part of the surface now is, which has sunk a little with each die
    die_base = base                       # the die's motion track needs one height, even when its thin face conforms to the surface
    if "conform" in spec:                   # a thin plate is not rigid: it bends over what it lands on, into the hollows as well
        base = field.blur(H, spec["conform"])
    Z = rel - peak["depth"] + base
    # how volume leaves a stroke: most of it runs out as a lip right beside it, the rest swells the ground further off
    sigmas, weights = (.14, .36, .9, 2.4, 7.0, 24.0), (.12, .24, .22, .16, .14, .12)
    K, P = field.kernel(sigmas, weights), field.potential(sigmas, weights)
    # first what the pressure does to what is already there. Old marks under the die are squeezed shallower, some much more than others
    zone = field.blur(inside.astype(np.float32), 7.0)
    H_pre = H
    if spec.get("collapse"):
        c = spec["collapse"] * zone * sm((blobs(ny, nx, res, 15.0, seed + 20) + 1.1) / 1.5)
        H = H - (H - field.blur(H, 5.0)) * c
        if "squeeze" in spec:           # the finest marks go first
            H = H - (H - field.blur(H, .5)) * spec["squeeze"] * zone
        H = H + (v0 - float(H.sum()) / res ** 2) / (2 * A * 2 * B)
        del c
    # then the sideways flow, worked out from the whole impression at once: it is what bends the older marks round the new ones. On top of that the clay doesn't give evenly: it shears, by far more than the mean flow, and that is what really pushes old letters out of true
    over0 = np.maximum(H - Z, 0)
    phi = field.apply(over0, P)
    gy, gx = np.gradient(phi, 1.0 / res)        # gy is along the rows, so towards -y
    del phi
    flow = spec.get("flow", 1.0) / 2.2          # the layer that moves is about 2.2 mm deep, shallower the harder the die
    reach = 3.0 + spec.get("flow", 1.0) * .6
    U, V = np.clip(-gx * flow, -reach, reach), np.clip(-gy * flow, -reach, reach)
    if "shear" in spec:
        u, v = swirl(ny, nx, res, spec["shear"], seed + 21)
        U, V = U + u * zone, V + v * zone
        del u, v
    keep = sm(np.minimum(A - np.abs(X), B - np.abs(Y)) / 9.0)      # the block's rounded rim stays at the rim
    U, V = U * keep, V * keep
    print(f"  flow up to {float(np.hypot(U, V).max()):.2f} mm", flush=True)
    if rec:
        rec.begin(H_pre, H, U, V, K, field, yy, xx, die_base)
        if not rock:
            rec.descend(Z, spec["depth"])
    H = bilerp(H, yy - V * res, xx - U * res)
    H = H + (v0 - float(H.sum()) / res ** 2) / (2 * A * 2 * B)      # this resampling isn't conservative by itself. Its small error goes back evenly
    del U, V, gx, gy, over0
    # then the die comes down, and the clay answers it
    forced = 0.0
    if rock:
        # a plate landing askew: where it has hold of the clay it takes the clay along as it skids, tearing it from what it hasn't got hold of, and what it has just printed is dragged out of shape by the part of it coming down next
        grip = spec["rock"]["grip"] * sm((blobs(ny, nx, res, 15.0, seed + 12) + .9) / 1.5)
        touched = np.zeros((ny, nx), bool)
        px, py = spec["rock"]["pivot"]
        fine = spec["rock"]["fine"]
        memory = field.blur(H, .55) - field.blur(H, 4.0)         # what it presses out isn't quite gone: clay remembers, and what was pressed out of it comes back a little when the weight is off
        X2, Y2 = X[::2, ::2], Y[::2, ::2]
        rows2, cols2 = yy / 2, xx / 2
        last = Z_last = None
        for g in range(0, len(rock), fine):
            sweep = None
            for q in rock[g:g + fine]:
                r2, _ = die_surface(face, spec, X2, Y2, die_res, q["dx"], q["dy"], q["spin"], q["tilt"])
                r2 = r2 - q["depth"]
                sweep = r2 if sweep is None else np.minimum(sweep, r2)
            Z = bilerp(sweep, rows2, cols2) + base
            q = rock[g + fine - 1]
            if rec and g // fine < 2:
                rec.landing(g // fine, Z)
            if last is not None:
                hold = field.blur((H > Z_last - .05).astype(np.float32), .4) * grip
                turn = math.radians(q["spin"] - last["spin"])
                dX, dY = (q["dx"] - last["dx"]) - turn * (Y - py), (q["dy"] - last["dy"]) + turn * (X - px)
                v = float(H.sum())
                rows, cols = yy + dY * hold * res, xx - dX * hold * res
                H = bilerp(H, rows, cols)
                H = H + (v - float(H.sum())) / H.size
                memory = bilerp(memory, rows, cols)
                del rows, cols, hold, dX, dY
            for _ in range(3):
                over = np.maximum(H - Z, 0)
                H = H - over + field.apply(over, K)
            forced += float(np.maximum(H - Z, 0).sum()) / res ** 2
            H = np.minimum(H, Z)
            touched |= H > Z - .03
            Z_last, last = Z, q
            if rec and g // fine >= 2:
                rec.put(f"g{g // fine}", H)
        contact = touched
        where = field.blur(contact.astype(np.float32), 1.0) * (.35 + .9 * sm((blobs(ny, nx, res, 12.0, seed + 13) + 1.0) / 2.0))
        gr, gc = np.gradient(field.blur(H, .5), 1.0 / res)
        H = H + memory * where * sm((1.3 - np.hypot(gr, gc)) / .9) * spec["ghost"]        # never on a wall: only ground can remember
        del gr, gc
        del grip, touched, Z_last, memory, where, X2, Y2, rows2, cols2, sweep
    else:
        STEPS, EACH, SETTLE = 8, 2, 8
        for i in range(STEPS):
            z = Z + (1 - (i + 1) / STEPS) * spec["depth"] * 1.1
            for _ in range(EACH):
                over = np.maximum(H - z, 0)
                H = H - over + field.apply(over, K)
        for _ in range(SETTLE):
            over = np.maximum(H - Z, 0)
            if float(over.max()) < .003:
                break
            H = H - over + field.apply(over, K)
        forced = float(np.maximum(H - Z, 0).sum()) / res ** 2       # what the die still has under it after flowing as far as a surface can: a wide die drives this out through the block's sides
        H = np.minimum(H, Z)
        contact = (H > Z - .03) & inside
    del face, rel, over, Z
    if rec:
        rec.put("s0", H)
    print(f"  contact {contact.sum() / res ** 2:.0f} mm2, driven through the sides {forced:.0f} mm3", flush=True)
    F = field.blur(contact.astype(np.float32), 1.5)
    # the wide hollow under the die, its volume in the swell around it
    if spec["dish"]:
        hollow = field.blur(F, 14.0) - field.blur(F, 46.0)
        H = H - hollow * (spec["dish"] / max(float(hollow.max()), 1e-3))
    # the block as a whole settles, most of all under the die. This volume leaves through the sides
    wide = field.blur(F, 60.0)
    wide = .35 + .65 * wide / max(float(wide.mean()), 1e-4)
    wide = wide / float(wide.mean())
    H = H - wide * spec["sag"]
    lost = float(spec["sag"]) * (2 * A) * (2 * B) + forced
    v1 = float(H.sum())
    if rec:
        rec.put("s1", H)
    # where the strain was too much, it tears
    if "tears" in spec:
        H = tear(H, F, spec, field, res)
    if rec:
        rec.put("s2", H)
    # the die comes away: clay clings to its edges and stands up in a fine burr
    edge = np.abs(np.gradient(F, axis=0)) + np.abs(np.gradient(F, axis=1))
    edge = sm(edge * res * .9)
    H = H + edge * (blobs(ny, nx, res, .35, seed + 40) * .6 + .3) * spec["burr"]
    if rec:
        rec.put("s3", H)
    # the wet clay moves a little after the die has gone: no edge stays where the die cut it, and the steepest walls give way
    if spec.get("relax"):
        u, v = swirl(ny, nx, res, spec["relax"], seed + 41)
        keep = sm(np.minimum(A - np.abs(X), B - np.abs(Y)) / 9.0)
        if rec:
            rec.relax(H, u * keep, v * keep)
        H = bilerp(H, yy - v * keep * res, xx - u * keep * res)
        del u, v, keep
    if rec:
        rec.put("s4", H)
    if spec.get("slump"):
        H = slump(H, res, spec["slump"], seed + 50, field)
    if rec:
        rec.put("s5", H)
    # wet clay relaxes: sharp crests round off a little
    H = H + (field.blur(H, .045) - H) * .55
    H = H + (v1 - float(H.sum())) / H.size
    if rec:
        rec.put("s6", H)
    print(f"{spec['name']}: {time.time() - t0:.0f}s, deepest now {float(H.min()):.2f}, highest {float(H.max()):.2f}, mean {float(H.mean()):.3f}", flush=True)
    return H.astype(np.float32), F.astype(np.float32), lost, wide


# ---- the sequence ---------------------------------------------------------------------------------------------------------------------------------------
FPS = 30
SEQ_H = os.path.join(SEVEN, "seq_h")
# In seconds from the start of 三 (T3). press: frames from first touch to full depth. lift: when it starts to rise. gone: out of the frame. settled: when the clay has stopped moving. fall: how long it falls for. head to tail: the frames made.
# cut: up to here the low shot of its face (the first two only). Contacts 0.9, then 4.6, 3.4, 2.6, 2.0, 1.5, 1.1 apart: 「古人说」 lands on the first, 「字，活了三千年」 on the second.
BLOWS = [
    dict(contact=0.9, press=13, lift=1.72, gone=2.5, settled=2.38, fall=.26, head=0.0, cut=0.64, tail=3.0, shake=1.5, scale=1.0, tilt=30.0, low=(470.0, 2.0, 32.0, 52.0)),
    dict(contact=5.5, press=9, lift=5.92, gone=6.5, settled=6.45, fall=.22, head=4.6, cut=5.28, tail=7.1, shake=2.2, scale=1.1, tilt=30.0, low=(900.0, 2.0, -5.5, 66.0)),       # the low shot from twice as far, the face's column high in the frame where d1_letter's is now (0.05 to 0.38 of the height); it was 446 and 63.5
    dict(contact=8.9, press=3, lift=9.16, gone=9.5, settled=9.46, fall=.12, head=8.55, tail=9.7, shake=3.0, scale=1.2),
    dict(contact=11.5, press=3, lift=11.7, gone=12.05, settled=12.0, fall=.1, head=11.2, tail=12.3, shake=2.6, scale=1.32),
    dict(contact=13.5, press=2, lift=13.58, gone=13.8, settled=13.76, fall=.07, head=13.25, tail=13.8, shake=3.4, scale=1.45),
    dict(contact=15.0, press=2, lift=15.07, gone=15.26, settled=15.22, fall=.06, head=14.9, tail=15.6, shake=1.6, scale=1.58),
    dict(contact=16.1, press=12, lift=16.6, gone=18.6, settled=18.4, fall=.16, head=15.75, tail=18.65, shake=6.0, scale=1.72, power=2.3, rise=150.0),
]
assert len(BLOWS) == LAST and all(abs(b["contact"] - a["contact"] - d) < 1e-6 for a, b, d in zip(BLOWS, BLOWS[1:], (4.6, 3.4, 2.6, 2.0, 1.5, 1.1)))


def small2(h):
    ny, nx = h.shape
    return h[:ny // 2 * 2, :nx // 2 * 2].reshape(ny // 2, 2, nx // 2, 2).mean(axis=(1, 3)).astype(np.float32)


class Recorder:
    """Sees what a press goes through, and keeps it at half the field's resolution. Up to where the die has gone right down, each frame is worked out afresh from the start, as far as the die has come by then: the old relief squeezed that far, carried that far, cut to where the die now is. Then it is the press's own stages. What it keeps is the height field, never a picture."""

    def __init__(self, k, grain, res):
        self.k, self.spec, self.blow, self.grain, self.res = k, PRESSES[k], BLOWS[k], grain, res
        self.snap = {}
        self.fc, self.fl, self.fs = round(self.blow["contact"] * FPS), round(self.blow["lift"] * FPS), round(self.blow["settled"] * FPS)
        post = self.fs - self.fl
        self.n_relax = max(int((post - 4) * .62), 1)
        self.n_slump = max(post - 4 - self.n_relax, 1)

    def put(self, name, h):
        self.snap[name] = small2(h + self.grain)

    def begin(self, H_pre, H_col, U, V, K, field, yy, xx, base):
        self.pre, self.col, self.U, self.V, self.K, self.field, self.yy, self.xx, self.base = H_pre, H_col, U, V, K, field, yy, xx, base

    def partway(self, a, z):
        h = self.pre + (self.col - self.pre) * a
        h = bilerp(h, self.yy - self.V * a * self.res, self.xx - self.U * a * self.res)
        for _ in range(3):
            over = np.maximum(h - z, 0)
            h = h - over + self.field.apply(over, self.K)
        return np.minimum(h, z)

    def descend(self, Z, depth):
        n = self.blow["press"]
        for i in range(1, n):
            t0 = time.time()
            a = i / n
            self.put(f"p{i}", self.partway(a, Z + (1 - a) * depth))
            print(f"  frame {i} of {n}, {time.time() - t0:.0f}s", flush=True)

    def landing(self, g, Z):
        self.put(f"g{g}", self.partway((g + 1) / 3, Z))

    def relax(self, H, u, v):
        for j in range(1, self.n_relax + 1):
            b = float(sm(j / self.n_relax))
            self.put(f"r{j}", bilerp(H, self.yy - v * b * self.res, self.xx - u * b * self.res))

    def write(self, F, wide, zb, zr):
        """Lays what it has seen out frame by frame, and writes the track the die takes. F, wide: where the die touched and how the block sank, from the press."""
        import json
        k, spec, bl = self.k, self.spec, self.blow
        folder = os.path.join(SEQ_H, f"b{k + 1}")
        os.makedirs(folder, exist_ok=True)
        z0, z1 = np.load(state_path(k)), np.load(state_path(k + 1))
        np.save(os.path.join(folder, "before.npy"), small2(z0["H"]))
        np.save(os.path.join(folder, "after.npy"), small2(z1["H"]))
        np.savez(os.path.join(folder, "side.npz"), bulge0=z0["bulge"], bulge1=z1["bulge"], burnish0=z0["burnish"], burnish1=z1["burnish"], lost0=float(z0["lost"]), lost1=float(z1["lost"]))
        S = self.snap
        rock = "rock" in spec
        down = [S[f"g{g}"] for g in range(spec["rock"]["n"])] if rock else [S[f"p{i}"] for i in range(1, bl["press"])] + [S["s0"]]
        seq = {}
        f = self.fc
        for h in down:
            seq[f] = h
            f += 1

        def between(a, b, n, at):
            for i in range(1, n + 1):
                seq[at + i - 1] = a + (b - a) * (i / n)
            return at + n

        room = self.fl - f
        n1 = max(room // 2, 1)
        f = between(S["s0"], S["s1"], n1, f)
        f = between(S["s1"], S["s2"], max(room - n1, 1), f)
        f = max(f, self.fl)
        f = between(S["s2"], S["s3"], 2, f)
        if spec.get("relax"):
            for j in range(1, self.n_relax + 1):
                seq[f] = S[f"r{j}"]
                f += 1
        else:
            # The code die has no sideways relaxation; retain its surface for the same interval.
            f = between(S["s3"], S["s4"], self.n_relax, f)
        f = between(S["s4"], S["s5"], self.n_slump, f)
        last = between(S["s5"], S["s6"], 2, f)
        wide2, F2 = small2(wide), small2(F)
        sag = max(spec["sag"] * .35, .1)
        back = max(spec["depth"] * .05, .03)
        for f in sorted(seq):
            t = f / FPS
            tau = t - bl["contact"]
            fade = 1 - float(sm((t - (bl["settled"] - .12)) / .12))
            h = seq[f] - wide2 * (sag * math.exp(-tau / .09) * math.cos(2 * math.pi * tau / .16) * fade)      # the whole block gives under the blow and comes back
            if t >= bl["lift"]:
                h = h - F2 * (back * math.exp(-(t - bl["lift"]) / .14) * fade)                               # what was held down comes up a little when the weight is off
            np.save(os.path.join(folder, f"h{f:05d}.npy"), h.astype(np.float32))
        # the track of the die
        fh, ft = round(bl["head"] * FPS), round(bl["tail"] * FPS)
        fcut = round(bl["cut"] * FPS) if "cut" in bl else None
        cx, cy = spec["centre"]
        depth = spec["depth"]
        high = 300.0 / bl["scale"]
        rise = bl.get("rise", 300.0 / bl["scale"])
        frames = []
        n = spec["rock"]["n"] if rock else bl["press"]
        for f in range(fh, ft + 1):
            t = f / FPS
            pose = dict(x=cx, y=cy, a=spec["rot"], sx=0.0, sy=0.0, tx=0.0, ty=0.0, tilt=0.0, up=0.0)
            if rock:
                q = zr[int(np.clip(f - self.fc, 0, n - 1))]
                pose.update(x=q["x"], y=q["y"], a=q["a"], sx=q["sx"], sy=q["sy"], tx=q["tx"], ty=q["ty"])
                z = q["z"]
            else:
                z = zb
            if f < self.fc:
                pose["up"] = high * ((self.fc - f) / (bl["fall"] * FPS)) ** 2 + (0.0 if rock else depth)
            elif f < self.fc + n and not rock:
                pose["up"] = (1 - (f - self.fc + 1) / n) * depth
            if t > bl["lift"]:
                pose["up"] = rise * min((t - bl["lift"]) / (bl["gone"] - bl["lift"]), 1.2) ** bl.get("power", 1.8)
            cam = "std"
            if fcut and f < fcut:
                cam = "low"
                D, cz, zt, zair = bl["low"]
                v = float(sm((t - (bl["cut"] - .14)) / .14))
                wob = math.sin((t - bl["head"]) * 3.1) * .8
                pose["tilt"] = (bl["tilt"] + wob) * (1 - v) + 3.0 * v
                pose["up"] = zair - z + wob * 1.5 + 8.0 * v
            elif fcut and f < self.fc:
                pose["tilt"] = 3.0 * (self.fc - f) / (self.fc - fcut)
            tau = t - bl["contact"]
            A_ = bl["shake"] * math.exp(-tau / .1) if tau >= 0 else 0.0
            shake = [A_ * math.sin(2 * math.pi * 19 * tau + .6), A_ * 1.3 * math.sin(2 * math.pi * 23 * tau)]
            h = "before.npy" if f < self.fc else (f"h{f:05d}.npy" if f in seq else "after.npy")
            frames.append(dict(f=f, t=round(t, 3), h=h, cam=cam, die=dict(z=z, **pose), shake=shake, w=float(sm((f - self.fc) / max(self.fl - self.fc, 1)))))
        with open(os.path.join(folder, "manifest.json"), "w") as fp:
            json.dump(dict(blow=k + 1, name=spec["name"], contact=self.fc, cut=fcut, scale=bl["scale"], res=self.res // 2, low=bl.get("low"), frames=frames), fp)
        print(f"b{k + 1}: {len(seq)} frames of clay, {len(frames)} frames in all", flush=True)


def dies():
    """Each die's face, for the meshes: its relief at 8 per mm, all its departures from flat in it, its lowest point at 0. Where it comes down."""
    os.makedirs(SEQ_H, exist_ok=True)
    die_res = 48
    for k, spec in enumerate(PRESSES):
        face = die_face(spec, die_res)
        ny, nx = face.shape
        w, h = spec["size"]
        yy, xx = np.ogrid[:ny, :nx]
        lx, ly = (xx + .5) / die_res - w / 2, h / 2 - (yy + .5) / die_res
        cav = np.clip(face / spec["relief"], 0, 1)
        if "bend" in spec:
            tx, ty = spec.get("low", (0.0, 0.0))
            bx, by = spec["bend"]
            face = face + (bx * ((lx - tx) / (w / 2)) ** 2 + by * ((ly - ty) / (h / 2)) ** 2).astype(np.float32)
        step = 6
        small = face[:ny // step * step, :nx // step * step].reshape(ny // step, step, nx // step, step).mean(axis=(1, 3))
        cavs = cav[:ny // step * step, :nx // step * step].reshape(ny // step, step, nx // step, step).mean(axis=(1, 3))
        z = np.load(state_path(k))["H"]
        res = int(np.load(state_path(k))["res"])
        X, Y = (np.arange(z.shape[1]) + .5) / res - A, B - (np.arange(z.shape[0]) + .5) / res
        cx, cy = spec["centre"]
        a = math.radians(spec["rot"])
        XX, YY = np.meshgrid(X[::8], Y[::8])
        qx, qy = (XX - cx) * math.cos(a) + (YY - cy) * math.sin(a), -(XX - cx) * math.sin(a) + (YY - cy) * math.cos(a)
        base = float(np.median(z[::8, ::8][(np.abs(qx) < w / 2) & (np.abs(qy) < h / 2)]))
        np.savez(os.path.join(SEQ_H, f"die_{k}.npz"), face=(small - small.min()).astype(np.float32), cav=cavs.astype(np.float32), res=die_res / step, w=w, h=h, relief=spec["relief"], depth=spec["depth"], base=base)
        print(spec["name"], small.shape, "base", round(base, 3), flush=True)
    print("finished", flush=True)


def blow_frames(k):
    """Runs the k-th press (0 to 6) again from the state before it, showing a Recorder everything, and checks it comes out at the state already on file."""
    z = np.load(state_path(k))
    res = int(z["res"])
    ny, nx = z["H"].shape
    grain = grit(ny, nx, res)
    H = z["H"] - grain
    field = Field(ny, nx, res)
    spec = PRESSES[k]
    rec = Recorder(k, grain, res)
    t0 = time.time()
    H1, F, lost, wide = press(H, spec, field, res, int(min(res * 1.5, 48)), rec=rec)
    off = float(np.abs(H1 + grain - np.load(state_path(k + 1))["H"]).max())
    print(f"press again, {time.time() - t0:.0f}s: at most {off:.4f} mm off the state on file", flush=True)
    d = np.load(os.path.join(SEQ_H, f"die_{k}.npz"))
    face, die_step = d["face"], float(d["res"])
    zr = []
    if "rock" in spec:
        R = spec["rock"]
        w, h = spec["size"]
        gy, gx = face.shape
        ly = h / 2 - (np.arange(gy)[:, None] + .5) / die_step
        lx = (np.arange(gx)[None, :] + .5) / die_step - w / 2
        tx, ty = spec.get("low", (0.0, 0.0))
        rk = poses(spec)
        base = rec.base
        for g in range(R["n"]):
            q = rk[g * R["fine"] + R["fine"] - 1]
            sx, sy = q["tilt"][0] / 100, q["tilt"][1] / 100
            lowest = float((face + (lx - tx) * sx + (ly - ty) * sy).min())
            cx, cy = spec["centre"]
            px, py = R["pivot"]
            c, n_ = math.cos(math.radians(q["spin"])), math.sin(math.radians(q["spin"]))
            x, y = px + (cx - px) * c - (cy - py) * n_ + q["dx"], py + (cx - px) * n_ + (cy - py) * c + q["dy"]
            zr.append(dict(x=x, y=y, a=spec["rot"] + q["spin"], sx=sx, sy=sy, tx=tx, ty=ty, z=base - q["depth"] - lowest))
    rec.write(F, wide, rec.base - spec["depth"], zr)
    print("finished", flush=True)


def sheet(name):
    """Eight frames of a segment, with their film times, for looking at."""
    from PIL import Image, ImageDraw
    folder = os.path.join(SEVEN, "seq_press", name)
    files = sorted(glob.glob(os.path.join(folder, "f*.png")))
    pick = [files[int(round(i * (len(files) - 1) / 7))] for i in range(8)]
    keys = {f"b{k + 1}": tuple(round((bl["contact"] + dt) * FPS) for dt in ((-.5, -.12, 0, .1, .3, .6, 1.0) if k < 2 else (-.2, -.05, 0, .05, .12, .25, .5))) for k, bl in enumerate(BLOWS)}
    if name in keys:
        have = {int(os.path.basename(p)[1:6]): p for p in files}
        pick = [have[min(have, key=lambda f: abs(f - want))] for want in keys[name]] + [files[-1]]
    W, H = 480, 270
    out = Image.new("RGB", (W * 4, H * 2), (10, 10, 10))
    for i, p in enumerate(pick):
        im = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
        f = int(os.path.basename(p)[1:6])
        ImageDraw.Draw(im).text((8, 6), f"{f / FPS:.2f} s  f{f}", fill=(235, 235, 235))
        out.paste(im, ((i % 4) * W, (i // 4) * H))
    path = os.path.join(SEVEN, "seq_press", f"sheet_{name}.png")
    out.save(path)
    print("wrote", path, flush=True)


def hillshade(H, path, az=200.0):
    """Lit low from the left, as the shot is, to read at a glance what has been pressed."""
    from PIL import Image
    k = max(H.shape[1] // 1600, 1)
    h = H[::k, ::k]
    gy, gx = np.gradient(h, k)
    s = .58 + (gx * .8 + gy * .5) * 1.7
    Image.fromarray((np.clip(s, 0, 1) * 255).astype(np.uint8)).save(path)


def bake(res, start=0):
    os.makedirs(SEVEN, exist_ok=True)
    os.makedirs(DIES, exist_ok=True)
    die_res = int(min(res * 1.5, 48))
    if start:
        z = np.load(state_path(start))
        H, res = z["H"], int(z["res"])
        H = H - grit(*H.shape, res)
        burnish = np.repeat(np.repeat(z["burnish"], 2, axis=0), 2, axis=1)[:H.shape[0], :H.shape[1]] if res > 16 else z["burnish"]
        bulge, total_lost, v0 = z["bulge"], float(z["lost"]), float(z["v0"])
    else:
        H = virgin(res)
        burnish = np.zeros_like(H)
        bulge = np.zeros_like(H[::res * 2, ::res * 2])      # sides' bulge, by where the volume went, at 2 mm a sample
        total_lost = 0.0
        v0 = float(H.sum()) / res ** 2
    ny, nx = H.shape
    field = Field(ny, nx, res)
    grain = grit(ny, nx, res)

    def save(k):
        np.savez(state_path(k), H=H + grain, burnish=burnish[::2, ::2] if res > 16 else burnish, bulge=bulge, res=res, lost=total_lost, v0=v0)
        hillshade(H, state_path(k)[:-4] + ".png")

    if not start:
        save(0)
    for k, spec in enumerate(PRESSES):
        if k < start:
            continue
        H, F, lost, wide = press(H, spec, field, res, die_res)
        burnish = np.maximum(burnish * .92, F)
        total_lost += lost
        bulge = bulge + (wide[::res * 2, ::res * 2] * lost / (2 * A * 2 * B)).astype(np.float32)
        v = float(H.sum()) / res ** 2
        print(f"after {k + 1}: top lost {v0 - v:.0f} mm3, went out through the sides {total_lost:.0f} mm3 (unaccounted {(v0 - v - total_lost):+.1f} mm3)", flush=True)
        save(k + 1)
    print("finished", flush=True)


# ---- the groove ---------------------------------------------------------------------------------------------------------------------------------------
# Where the character of the white page lies on the clay: how many mm the frame takes seen from straight above, and the point in its middle. The cut from film_wo.py's last frame lands on it, so the clay's stroke is that stroke, the same size in the frame
WO_WIDE, WO_AT = 225.0, (0.0, 0.0)
GROOVE_STEP = .1                                                     # mm
GROOVE_CUTS = ((0, .3, "g1"), (0, .62, "g2"), (0, 1.0, "g3"))        # stroke, how far along it: the three moments kept
GROOVE_WINDOW = (-58.0, 22.0, -18.0, 62.0)                           # the part of the block the groove and all it pushes aside stay inside
GROOVE_FRAMES = (636, 732)                                           # 21.2 to 24.4 s after 三 starts: from 「终于有一夜」
# (stroke, from, to, start, end): it goes and stops. Seconds after GROOVE_FRAMES[0]. What is left out of the table it just stands.
GROOVE_GOES = ((0, 0.0, .22, .12, .52), (0, .22, .44, .72, 1.12), (0, .44, .72, 1.34, 1.84), (0, .72, 1.0, 2.06, 2.72))


def wo_masks():
    """What each stroke of 我 covers, as film_wo.py's brush left it: a list of them at SS times the frame, and the frame pixels of their square's corner and its side."""
    path = os.path.join(OUT, "crab", "wo_masks.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["ink"], z["masks"], tuple(int(v) for v in z["place"]), int(z["ss"])
    from film_wo import SS, written
    ink, masks, place = written()
    ink, masks = (ink * 255).astype(np.uint8), (np.stack(masks) * 255).astype(np.uint8)
    np.savez(path, ink=ink, masks=masks, place=np.array(place), ss=SS)
    return ink, masks, tuple(place), SS


def wo_mm(px, py):
    q = WO_WIDE / 1920
    return WO_AT[0] + (np.asarray(px) - 960) * q, WO_AT[1] - (np.asarray(py) - 540) * q


def wo_groove():
    """The first stroke of 我 in mm on the block at GROOVE_STEP along it, in the middle of what the brush covered, with how wide it was there."""
    from film_wo import WO
    from ink_brush import spline
    from scipy.ndimage import distance_transform_edt, gaussian_filter1d, map_coordinates
    ink, masks, (x0, y0, frame), ss = wo_masks()
    m = masks[0] > 127
    edt = distance_transform_edt(m).astype(np.float32)
    k = frame / 100 * ss
    q = spline([p[:3] for p in WO["marks"][0][0]], .1)[:, :2] * k
    # onto the middle of the stroke as it came out: uphill across it, a few times
    for _ in range(6):
        d = np.gradient(q, axis=0)
        d = d / (np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6)
        n = np.stack([-d[:, 1], d[:, 0]], 1)
        best = np.stack([map_coordinates(edt, [q[:, 1] + n[:, 1] * o, q[:, 0] + n[:, 0] * o], order=1) for o in (-3.0, 0.0, 3.0)], 1)
        q = q + n * (3.0 * (np.argmax(best, 1) - 1))[:, None] * .5
        q = gaussian_filter1d(q, 6.0, axis=0, mode="nearest")
    wide = map_coordinates(edt, [q[:, 1], q[:, 0]], order=1)
    x, y = wo_mm(x0 + q[:, 0] / ss, y0 + q[:, 1] / ss)
    mm = WO_WIDE / 1920 / ss
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    at = np.arange(0, s[-1], GROOVE_STEP)
    hw = np.interp(at, s, gaussian_filter1d(wide, 8.0, mode="nearest") * mm)
    return [dict(x=np.interp(at, s, x), y=np.interp(at, s, y), hw=np.clip(hw * 1.1, .35, 4.6), stops=(.22, .44, .72))]


def groove_timetable(lengths):
    """For each frame in which the groove is on the move: which stroke it has got to, how far along it, the film frame."""
    out = []
    for f in range(GROOVE_FRAMES[0], GROOVE_FRAMES[1] + 1):
        t = (f - GROOVE_FRAMES[0]) / FPS
        for k, a, b, t0, t1 in GROOVE_GOES:
            if t0 < t <= t1 + 1 / FPS:
                u = min((t - t0) / (t1 - t0), 1.0)
                u = u * u * (3 - 2 * u) * .35 + u * .65            # it doesn't start or stop dead
                out.append((k, int((a + (b - a) * u) * (lengths[k] - 1)), f))
                break
    return out


class Plough:
    """A blunt thing nobody can see, going along under the surface. Whatever stands above it is driven out to both sides and ahead of it, each step of the way, so the walls of what it leaves stand up as lips and a little of the clay goes on before it."""
    TILE = 768

    def __init__(self, H, res, seed):
        from scipy.fft import rfft2
        self.H, self.res = H, res
        self.fresh = np.zeros_like(H)
        ny, nx = H.shape
        self.lump = blobs(ny, nx, res, .45, seed) * .7 + blobs(ny, nx, res, 1.6, seed + 2) * .7
        self.crumb = blobs(ny, nx, res, .16, seed + 3) * .3 + blobs(ny, nx, res, .4, seed + 4) * .3 + blobs(ny, nx, res, 1.1, seed + 5) * .55 + blobs(ny, nx, res, 3.0, seed + 6) * .7 + blobs(ny, nx, res, 8.0, seed + 7) * .6
        self.r = np.random.default_rng(seed + 1)
        n = self.TILE
        fy, fx = np.fft.fftfreq(n, 1 / res)[:, None], np.fft.rfftfreq(n, 1 / res)[None, :]
        f2 = fy * fy + fx * fx
        self.K = sum(w * np.exp(-2 * np.pi ** 2 * s * s * f2) for s, w in zip((.45, .9, 1.8, 3.5), (.06, .3, .38, .26))).astype(np.complex64)
        self.moved = 0.0
        self.grain = None

    def settled(self):
        """What it looks like a moment after it has passed: the knife edges it cut sag and go blunt, and what it has laid bare is as rough as the rest of the clay. Worked out afresh for each moment; it never feeds back into the plough, so a frame later everything still lines up."""
        from scipy.ndimage import gaussian_filter
        H, res = self.H, self.res
        rows, cols = np.nonzero(self.fresh > .01)
        pad = int(9 * res)
        r0, r1, c0, c1 = max(rows.min() - pad, 0), min(rows.max() + pad, H.shape[0]), max(cols.min() - pad, 0), min(cols.max() + pad, H.shape[1])
        win, f = H[r0:r1, c0:c1], self.fresh[r0:r1, c0:c1]
        near = np.clip(gaussian_filter(f, .9 * res) * 2.6, 0, 1)
        if self.grain is None:
            self.grain = grit(*H.shape, res)
        rough = self.grain[r0:r1, c0:c1]
        out = H.copy()
        soft = win + near * (gaussian_filter(win, .13 * res) - win)
        out[r0:r1, c0:c1] = soft + near * rough * .75
        return out

    def spread(self, r0, c0, over, weight=True):
        """Puts the volume in over, a window of the field whose top left is (r0, c0), back round about where it came from."""
        from scipy.fft import irfft2, rfft2
        n, H = self.TILE, self.H
        total = float(over.sum())
        if total <= 0:
            return
        my, mx = over.shape
        tr, tc = int(np.clip(r0 + my // 2 - n // 2, 0, H.shape[0] - n)), int(np.clip(c0 + mx // 2 - n // 2, 0, H.shape[1] - n))
        tile = np.zeros((n, n), np.float32)
        tile[r0 - tr:r0 - tr + my, c0 - tc:c0 - tc + mx] = over
        if weight:
            tile = tile * (1 + np.clip(self.lump[tr:tr + n, tc:tc + n], -1.4, 2) * .5)
            tile = np.maximum(tile, 0)
        back = irfft2(rfft2(tile) * self.K, s=(n, n))
        H[tr:tr + n, tc:tc + n] += back * (total / float(back.sum()))        # (it wraps at the tile's edge, far out where almost nothing reaches)

    def step(self, x, y, dx, dy, hw, dep, ground):
        """One step of the plough at (x, y) in mm, heading (dx, dy)."""
        H, res = self.H, self.res
        W = int((hw * 2.4 + .4) * res)
        rc, cc = int((B - y) * res), int((x + A) * res)
        r0, c0 = rc - W, cc - W
        yy, xx = np.ogrid[r0:rc + W, c0:cc + W]
        X, Y = (xx + .5) / res - A - x, B - (yy + .5) / res - y
        a, b = X * dx + Y * dy, -X * dy + Y * dx
        # the clay it goes through isn't even: harder here, softer there, so the walls it leaves aren't turned on a lathe
        c = np.clip(self.crumb[r0:rc + W, c0:cc + W], -2, 2)
        q = (a / (hw * 1.15)) ** 2 + (b / (hw * (1 + .13 * c))) ** 2
        Z = ground - dep + dep * 1.25 * q ** .85 + c * dep * .09
        win = H[r0:rc + W, c0:cc + W]
        over = np.maximum(win - Z, 0).astype(np.float32)
        win -= over
        self.moved += float(over.sum()) / res ** 2
        f = self.fresh[r0:rc + W, c0:cc + W]
        np.maximum(f, sm(1.25 - np.sqrt(q)).astype(np.float32), out=f)
        self.spread(r0, c0, over)

    def shove(self, x, y, dx, dy, hw, amount):
        """Ahead of it and beside it the clay is pushed aside, and whatever is written there goes with it."""
        H, res = self.H, self.res
        W = int(12 * res)
        rc, cc = int((B - y) * res), int((x + A) * res)
        r0, c0, r1, c1 = max(rc - W, 0), max(cc - W, 0), min(rc + W, H.shape[0]), min(cc + W, H.shape[1])
        yy, xx = np.mgrid[r0:r1, c0:c1]
        X, Y = (xx + .5) / res - A - x, B - (yy + .5) / res - y
        a, b = X * dx + Y * dy, -X * dy + Y * dx
        m = amount * np.sign(b) * (1 - np.exp(-(b / hw) ** 2)) * np.exp(-np.abs(b) / 2.6) * np.exp(-((a - 2.0) / 5.0) ** 2)
        win = H[r0:r1, c0:c1]
        v = float(win.sum())
        new = bilerp(win, yy - r0 + m * dx * res, xx - c0 + m * dy * res)
        H[r0:r1, c0:c1] = new + (v - float(new.sum())) / new.size

    def nick(self, x, y, dx, dy, hw):
        """The lip, stretched, splits: a short crack going off from it, jagged, into the ground either side."""
        from PIL import Image, ImageDraw
        from scipy.ndimage import gaussian_filter
        H, res, r = self.H, self.res, self.r
        side = r.choice((-1, 1))
        th = math.atan2(dy, dx) + side * math.pi / 2 + r.normal(0, .55)
        px, py = x - dy * side * hw * 1.1, y + dx * side * hw * 1.1
        L, width, dep = r.uniform(.6, 2.8), r.uniform(.06, .15), r.uniform(.14, .34)
        W = int(5.5 * res)
        rc, cc = int((B - py) * res), int((px + A) * res)
        im = Image.new("L", (2 * W, 2 * W), 0)
        d = ImageDraw.Draw(im)
        s, cx, cy = 0.0, float(W), float(W)
        while s < L:
            run = r.uniform(.2, .7)
            a = th + r.normal(0, .6)
            qx, qy = cx + math.cos(a) * run * res, cy - math.sin(a) * run * res
            d.line([cx, cy, qx, qy], fill=int(255 * (1 - s / L) ** .6), width=max(int(width * res * (1 - s / L) ** .6), 1))
            cx, cy, s = qx, qy, s + run
        c = gaussian_filter(np.asarray(im, np.float32) / 255, .035 * res)
        over = c * dep
        H[rc - W:rc + W, cc - W:cc + W] -= over
        self.spread(rc - W, cc - W, over, weight=False)


def groove(frames=False):
    """No die, no hand. Out of the side of what is already there the clay moves by itself, along the stroke the ghost saw: first the wedge, then, up and to the right, the long stroke, which stops. It goes in the beats the vision goes in."""
    from scipy.ndimage import gaussian_filter1d
    z = np.load(state_path(LAST))
    H, res = z["H"].copy(), int(z["res"])
    burnish, bulge, lost, v0 = z["burnish"], z["bulge"], float(z["lost"]), float(z["v0"])
    v_start = float(H.sum())
    field = Field(*H.shape, res)
    smooth = field.blur(H, 2.6)
    plough = Plough(H, res, 501)
    t0 = time.time()
    cuts = list(GROOVE_CUTS)
    strokes = wo_groove()
    np.savez(os.path.join(OUT, "crab", "wo_first.npz"), **{c: strokes[0][c] for c in "xy"}, hw=strokes[0]["hw"])
    todo = groove_timetable([len(st["x"]) for st in strokes]) if frames else []
    made = []
    if frames:
        folder = os.path.join(SEQ_H, "g")
        os.makedirs(folder, exist_ok=True)
        wx0, wx1, wy0, wy1 = GROOVE_WINDOW
        r0, r1, c0, c1 = int((B - wy1) * res), int((B - wy0) * res), int((wx0 + A) * res), int((wx1 + A) * res)

        def film_frame(f, done):
            name = f"h{f:05d}.npy"
            np.save(os.path.join(folder, name), done[r0:r1, c0:c1].astype(np.float32))
            np.save(os.path.join(folder, f"fresh{f:05d}.npy"), plough.fresh[r0:r1:2, c0:c1:2].astype(np.float32))
            return name
    for k, st in enumerate(strokes):
        x, y = st["x"], st["y"]
        hw = st["hw"]
        n = len(x)
        # nothing is steering it: it wanders off its line by a little, more than the brush did
        wr = np.random.default_rng(600 + k)
        along = np.arange(n) * GROOVE_STEP
        x = x + np.interp(along, np.arange(0, along[-1] + 6, 3.0), wr.normal(0, .2, len(np.arange(0, along[-1] + 6, 3.0))))
        y = y + np.interp(along, np.arange(0, along[-1] + 6, 3.0), wr.normal(0, .2, len(np.arange(0, along[-1] + 6, 3.0))))
        stops = [int(at * n) for at in st["stops"]]
        # it meets harder and softer going, and each time it sets off again it snatches, not evenly
        going = 1 + .1 * gaussian_filter1d(wr.normal(0, 1, n), 16.0) * math.sqrt(2 * math.sqrt(math.pi) * 16.0)
        snatch = gaussian_filter1d(wr.normal(0, 1, n), 2.5) * math.sqrt(2 * math.sqrt(math.pi) * 2.5)
        dx, dy = np.gradient(x), np.gradient(y)
        ln = np.hypot(dx, dy) + 1e-6
        dx, dy = dx / ln, dy / ln
        s = np.cumsum(ln)
        ground = bilerp(smooth, (B - y) * res, (x + A) * res)
        next_nick, next_shove = float(plough.r.uniform(.8, 3.0)), .3
        for i in range(n):
            since = min((s[i] - s[j] for j in stops if j <= i), default=99.0)
            dep = hw[i] * 1.5 * going[i] * (1 + .09 * snatch[i] * math.exp(-since / 1.3))
            plough.step(float(x[i]), float(y[i]), float(dx[i]), float(dy[i]), float(hw[i]), float(dep), float(ground[i]))
            if s[i] > next_shove:
                plough.shove(float(x[i]), float(y[i]), float(dx[i]), float(dy[i]), float(hw[i]), .06)
                next_shove = s[i] + .3
            if s[i] > next_nick:
                plough.nick(float(x[i]), float(y[i]), float(dx[i]), float(dy[i]), float(hw[i]))
                next_nick = s[i] + float(plough.r.uniform(.9, 3.6))
            if todo and todo[0][:2] <= (k, i):
                fs = []
                while todo and todo[0][:2] <= (k, i):
                    fs.append(todo.pop(0)[2])
                name = film_frame(fs[-1], plough.settled())
                made += [dict(f=int(f), h=name) for f in fs]
            while cuts and cuts[0][0] == k and i >= int(cuts[0][1] * (n - 1)):
                name = cuts.pop(0)[2]
                done = plough.settled()
                np.savez(state_path(name), H=done, burnish=burnish, bulge=bulge, res=res, lost=lost, v0=v0, fresh=plough.fresh[::2, ::2])
                hillshade(done, state_path(name)[:-4] + ".png")
                print(f"{name}: at ({x[i]:.1f}, {y[i]:.1f}), {plough.moved:.0f} mm3 moved so far, volume off by {(float(H.sum()) - v_start) / res ** 2:+.2f} mm3, {time.time() - t0:.0f}s", flush=True)
    if frames:
        import json
        name = made[-1]["h"] if made else None
        all_frames = {m["f"]: m["h"] for m in made}
        out = []
        last = None
        for f in range(GROOVE_FRAMES[0], GROOVE_FRAMES[1] + 1):
            last = all_frames.get(f, last)
            out.append(dict(f=f, h=last))
        with open(os.path.join(folder, "manifest.json"), "w") as fp:
            json.dump(dict(window=GROOVE_WINDOW, res=res, frames=out), fp)
        print(len(made), "groove frames", flush=True)
    print("finished", flush=True)


def wo_clay():
    """The whole of 我 in the clay, as the cut at 96.1 finds it: the groove's first stroke and the rest, each stroke's outline the brush's own and far deeper than anything the dies left, what came out of it standing up along its lips. state_wo, and wo_label, which stroke each place belongs to."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, zoom
    ink, masks, (x0, y0, frame), ss = wo_masks()
    z = np.load(state_path("g3"))
    H, res = z["H"].copy(), int(z["res"])
    burnish, bulge, lost, v0 = z["burnish"], z["bulge"], float(z["lost"]), float(z["v0"])
    q = WO_WIDE / 1920
    x_lo, y_hi = wo_mm(x0, y0)
    x_hi, y_lo = wo_mm(x0 + frame, y0 + frame)
    pad = 14.0
    r0, r1, c0, c1 = max(int((B - y_hi - pad) * res), 0), min(int((B - y_lo + pad) * res), H.shape[0]), max(int((x_lo - pad + A) * res), 0), min(int((x_hi + pad + A) * res), H.shape[1])
    rows, cols = np.mgrid[r0:r1, c0:c1].astype(np.float32)
    px, py = (((cols + .5) / res - A - WO_AT[0]) / q + 960 - x0) * ss, ((WO_AT[1] - (B - (rows + .5) / res)) / q + 540 - y0) * ss
    del rows, cols
    inked = map_coordinates(np.float32(ink) / 255, [py, px], order=1, mode="constant") > .5
    each = [map_coordinates(np.float32(m) / 255, [py[::4, ::4], px[::4, ::4]], order=1, mode="constant") > .5 for m in masks]
    del px, py
    # which stroke each place belongs to: the nearest
    label = np.argmin(np.stack([distance_transform_edt(~m).astype(np.float32) for m in each]), axis=0).astype(np.uint8)
    os.makedirs(os.path.join(OUT, "crab"), exist_ok=True)
    np.savez(os.path.join(OUT, "crab", "wo_label.npz"), label=label, rc=np.array((r0, c0)), step=4, res=res)
    d = distance_transform_edt(inked).astype(np.float32) / res
    win = H[r0:r1, c0:c1]
    ground = gaussian_filter(small2(small2(win)), 1.0 * res / 4)
    ground = zoom(ground, 4, order=1)[:win.shape[0], :win.shape[1]]
    pad_r, pad_c = win.shape[0] - ground.shape[0], win.shape[1] - ground.shape[1]
    if pad_r or pad_c:
        ground = np.pad(ground, ((0, pad_r), (0, pad_c)), mode="edge")
    plough = Plough(H, res, 777)
    rough = plough.crumb[r0:r1, c0:c1]
    DEEP = 9.0
    Z = ground - DEEP * (1 - np.exp(-(d / 1.7) ** 1.25)) - 1.1 * sm(d / .3) + np.clip(rough, -2, 2) * .28 * sm(d / 1.2)
    Z = Z.astype(np.float32)
    over = np.maximum(win - Z, 0) * inked
    moved = float(over.sum()) / res ** 2
    win -= over
    # what came out of it lies along its lips, lumpy, more where the stroke is broad
    coarse = small2(small2(over))
    lump = np.clip(small2(small2(plough.lump[r0:r1, c0:c1])), -1.4, 2)
    back = sum(w * gaussian_filter(coarse * (1 + .5 * lump), sg * res / 4) for sg, w in ((.8, .3), (1.8, .45), (3.8, .25)))
    back = zoom(back, 4, order=1)[:win.shape[0], :win.shape[1]]
    back = np.pad(back, ((0, win.shape[0] - back.shape[0]), (0, win.shape[1] - back.shape[1])), mode="edge")
    rim_top = float(np.percentile(back[~inked & (d == 0)], 99.9))
    win += back * (moved * res ** 2 / float(back.sum()))
    win[inked] = np.minimum(win, Z)[inked]
    # the lips, stretched, split here and there
    wr = np.random.default_rng(778)
    edge = np.argwhere(inked[1:-1:8, 1:-1:8] & ~(np.roll(inked, 8, 0) & np.roll(inked, -8, 0) & np.roll(inked, 8, 1) & np.roll(inked, -8, 1))[1:-1:8, 1:-1:8])
    for i in wr.choice(len(edge), 70, replace=False):
        r, c = edge[i] * 8 + 1 + (r0, c0)
        th = wr.uniform(0, 6.28)
        plough.nick((c + .5) / res - A, B - (r + .5) / res, math.cos(th), math.sin(th), 1.2)
    plough.fresh[r0:r1, c0:c1] = np.maximum(plough.fresh[r0:r1, c0:c1], sm(d / .6))
    done = plough.settled()
    np.savez(state_path("wo"), H=done, burnish=burnish, bulge=bulge, res=res, lost=lost, v0=v0, fresh=plough.fresh[::2, ::2])
    hillshade(done, state_path("wo")[:-4] + ".png")
    print(f"wo: {moved:.0f} mm3 moved, most it stands up about {rim_top * moved * res ** 2 / float(back.sum()):.1f} mm, stroke bottoms at {float(Z[inked].min()):.1f}", flush=True)


# ---- the little crab's food ------------------------------------------------------------------------------------------------------------------------------
# key, signs, font, face in it, how much the outline is softened and how much clay is added to it (mm), how close several signs stand. Each is judged by eye: the first thing it must do is read
GLYPHS = (("zi", "字", "STHeiti Medium.ttc", 0, .25, .3, 1.0), ("a", "A", "Arial Black.ttf", 0, .3, .1, 1.0), ("wedge", "\U00012000", "NotoSansCuneiform-Regular.ttf", 0, .3, .2, 1.0), ("shan", "山", "STHeiti Medium.ttc", 0, .35, .55, 1.0), ("omega", "\u03a9", "Georgia Bold.ttf", 0, .3, .25, 1.0), ("code", "</>", "Menlo.ttc", 1, .45, .8, .88), ("wen", "文", "STHeiti Medium.ttc", 0, .35, .5, 1.0))
GLYPH_RES = 16      # per mm


def glyphs(height=16.2):
    """The letters of clay that lie at his feet: the distance from each one's outline, at its size, read from this machine's fonts (they stay out of the repo). film_imprint_clawd.py makes them solid."""
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    out, names = {}, []
    for key, ch, name, index, soft, fat, close in GLYPHS:
        path = next(p for p in (f"/System/Library/Fonts/Supplemental/{name}", f"/System/Library/Fonts/{name}") if os.path.exists(p))
        big = 900
        f = ImageFont.truetype(path, big, index=index)
        im = Image.new("L", (big * 2 * len(ch), big * 2), 0)
        for i, c in enumerate(ch):      # several signs stand close enough to be pressed into one lump
            ImageDraw.Draw(im).text((big // 3 + int(i * close * f.getlength(c)), big // 3), c, fill=255, font=f)
        box = im.getbbox()
        im = im.crop(box)
        px = height * GLYPH_RES
        k = min(px / im.height, px * (1.25 if len(ch) == 1 else 2.2) / im.width)
        im = im.resize((max(int(im.width * k), 1), max(int(im.height * k), 1)), Image.LANCZOS)
        pad = 6 * GLYPH_RES
        a = np.zeros((im.height + 2 * pad, im.width + 2 * pad), bool)
        a[pad:pad + im.height, pad:pad + im.width] = np.asarray(im) > 110
        out[key] = (gaussian_filter((distance_transform_edt(~a) - distance_transform_edt(a)) / GLYPH_RES, soft * GLYPH_RES) - fat).astype(np.float32)
        names.append(key)
        print(key, a.shape, flush=True)
    os.makedirs(os.path.join(OUT, "crab"), exist_ok=True)
    np.savez(os.path.join(OUT, "crab", "clay_glyphs.npz"), names=np.array(names), res=GLYPH_RES, **out)
    print("finished", flush=True)


# ---- the board ---------------------------------------------------------------------------------------------------------------------------------------
def png16(path, a):
    """A 16-bit RGB PNG of linear values, which neither PIL nor Blender's image API will write for us as fast."""
    import struct
    import zlib
    h, w = a.shape[:2]
    data = (np.clip(a, 0, 1) * 65535 + .5).astype(">u2")
    raw = b"".join(b"\0" + data[i].tobytes() for i in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


BOARD_RECT = (-520.0, 520.0, -320.0, 480.0)
BOARD_PX = 5.0


def board(clean=False):
    """An old workshop board: dark, oiled by decades of hands, its grain standing a little proud, scratched, with dried slip on it round where the block sits, and a dark halo where the clay's water has gone into the wood. Colour and height (linear) as 16-bit PNG. clean: the same wood with no clay on it, for the little crab, as arrays."""
    from PIL import Image
    x0, x1, y0, y1 = BOARD_RECT
    nx, ny = int((x1 - x0) * BOARD_PX), int((y1 - y0) * BOARD_PX)
    n = max(nx, ny)
    r = np.random.default_rng(71)
    yy, xx = np.mgrid[:ny, :nx]
    X, Y = x0 + (xx + .5) / BOARD_PX, y1 - (yy + .5) / BOARD_PX
    warp = spectral(1024, 3.0, 72)
    u = X * .012 + 7
    v = (Y + bilerp(warp, (Y * .25) % 1022, (X * .05) % 1022) * 16) * .19
    v = v + .9 * np.sin(v * .31 + 1.0) + .5 * np.sin(v * .83 + bilerp(warp, (X * .02) % 1022, (Y * .02) % 1022) * 2)
    rings = np.sin(v * 2 * np.pi * .55)
    late = sm((rings + .3) / 1.1) * (.55 + .45 * sm((bilerp(warp, (Y * .9) % 1022, (X * .05) % 1022) + 1) / 2))
    streak = spectral(4096, 1.3, 73)
    s = bilerp(streak, (Y * 3.2) % 4094, (X * .22) % 4094)
    pore = spectral(4096, .8, 74)
    p = bilerp(pore, (Y * 5) % 4094, (X * .6) % 4094)
    dark, pale = np.array((.0065, .0042, .0029), np.float32), np.array((.027, .017, .0105), np.float32)
    mix = np.clip(.16 + late * .22 + s * .13, 0, 1)[..., None]
    col = dark + (pale - dark) * mix
    col = col * (1 + p[..., None] * .10)
    h = late * .10 + s * .05 + p * .015
    # the seam between two planks, not far behind the block
    seam = sm(1 - np.abs(Y - 138.0 - bilerp(warp, (X * .01) % 1022, np.full_like(X, 5.0)) * 1.2) / 1.1)
    col = col * (1 - seam[..., None] * .9)
    h = h - seam * 1.4
    # scratches, knife cuts, dents from dropped tools
    from PIL import ImageDraw
    im = Image.new("L", (nx, ny), 0)
    d = ImageDraw.Draw(im)
    for _ in range(300):
        cx, cy, L, a = r.uniform(0, nx), r.uniform(0, ny), r.uniform(8, 110) * BOARD_PX, r.normal(0, .7)
        d.line([cx, cy, cx + L * math.cos(a), cy + L * math.sin(a)], fill=int(r.uniform(50, 220)), width=int(r.integers(1, 3)))
    cuts = np.asarray(im, np.float32) / 255
    col = col + (np.array((.05, .037, .027), np.float32) - col) * (cuts * .33)[..., None]
    h = h - cuts * .22
    if clean:
        np.save(os.path.join(OUT, "board_clean_col.npy"), col.astype(np.float32))
        np.save(os.path.join(OUT, "board_clean_h.npy"), np.dstack([h, .62 + .10 * sm((s + .5) / 1.5)]).astype(np.float32))
        print("clean board", col.shape, flush=True)
        return
    # the clay's own marks: a wet halo round the block, thin dried slip beyond it
    d_block = np.hypot(np.maximum(np.abs(X) - A, 0), np.maximum(np.abs(Y) - B, 0))
    n1, n2 = spectral(n, 2.5, 75)[:ny, :nx], spectral(n, 1.6, 76)[:ny, :nx]
    wet = sm(1 - (d_block + n1 * 3) / 7) * (d_block > 0)
    col = col * (1 - wet * .55)[..., None]
    slip = sm((n1 * .85 + n2 * .22 + 1.8 * np.exp(-d_block / 14) - 1.9) / .22) * sm((n2 + .4) / .7)
    col = col + (np.array((.17, .185, .19), np.float32) - col) * (slip * .3)[..., None]
    h = h + slip * .16
    rough = .62 + .10 * sm((s + .5) / 1.5) + .16 * slip - .22 * wet
    col = col * (.2 + .8 * np.exp(-(d_block / 130) ** 2))[..., None]      # the pool of the lamp: the old wood goes off into the dark beyond the block
    png16(os.path.join(OUT, "board_col.png"), col)
    png16(os.path.join(OUT, "board_h.png"), np.dstack([(h + 2) / 3, rough, np.zeros_like(h)]))
    print("board", col.shape, flush=True)


if __name__ == "__main__":
    def arg(name, default):
        return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
    if sys.argv[1] == "bake":
        bake(arg("--res", 32), arg("--from", 0))
    elif sys.argv[1] == "groove":
        groove("--frames" in sys.argv)
    elif sys.argv[1] == "wo":
        wo_clay()
    elif sys.argv[1] == "dies":
        dies()
    elif sys.argv[1] == "frames":
        blow_frames(arg("--blow", 1) - 1)
    elif sys.argv[1] == "sheet":
        sheet(sys.argv[2])
    elif sys.argv[1] == "board":
        os.makedirs(OUT, exist_ok=True)
        board("--clean" in sys.argv)
    elif sys.argv[1] == "glyphs":
        glyphs()
