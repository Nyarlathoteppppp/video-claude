"""The printing house round the press, the textures of it: the floor's old boards, the rough plaster, the printed sheets drying on the line, the labels on the type case. film_press_machine.py builds the room and reads them.

    python film_press_room.py bake          # into ink/film_out/film/press/maps/

Baked once, since Blender's own Python has neither PIL nor scipy. Importing this needs only numpy. Sizes in the units of the scene, 1 unit a millimetre of the model, and the model is at about 1:3.5: the machine stands 512 units tall for 1.8 m.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import film_press_maps as M  # noqa: E402

FLOOR_Z, WALL_Y = -300.0, 420.0
FLOOR_RECT = (-1600.0, 800.0, -800.0, WALL_Y)       # x0, x1, y0, y1: what the wide shot can see of the floor
FLOOR_PX = 3.0
FLOOR_H = 8.0                                       # the height map spans this many units
WALL_RECT = (-1600.0, 800.0, FLOOR_Z, 700.0)        # x0, x1, z0, z1
WALL_PX = 2.5
WALL_H = 24.0
SHEET_W, SHEET_L = 118.0, 150.0                     # a drying sheet's front half
LABELS = ("初号", "小初", "一号", "小一", "二号", "小二", "三号", "小三", "四号", "小四", "五号", "小五", "六号", "小六", "七号", "八号", "正楷", "仿宋")
LABEL_ROWS, LABEL_COLS = 5, 4
# What the sheets on the line have printed on them: lines already used and checked on the big sheet, a heading and a body. Nothing here says 我.
PAGES = [
    dict(title="烽火连三月", font="song", vertical=True, lines=["烽火连三月，家书抵万金", "念天地之悠悠，独怆然而涕下", "不以物喜，不以己悲", "十年生死两茫茫", "问君能有几多愁", "此恨绵绵无绝期"]),
    dict(title="So long lives this", font="baskr", lines=["So long lives this, and this gives life to thee.", "", "Dear Anna, the rain has not stopped since you left.", "Please keep the receipt.", "The minutes of the last meeting are enclosed.", "Yours, as ever,"]),
    dict(title="人固有一死", font="song", lines=["本院认为，被告人的行为已构成盗窃罪，", "依法应予惩处。判决如下：", "被告人犯盗窃罪，判处有期徒刑一年。", "如不服本判决，可在接到判决书的第二日起", "十日内提出上诉。", "审判长、审判员、书记员。"]),
    dict(title=M.HOMER[:11], font="bask", lines=[M.HOMER, "", M.HOMER, "", M.HOMER]),
    dict(title="见字如面", font="kai", lines=["你走以后，家里一切都好，勿念。", "今天下雨了，想起你说过的那家面馆。", "天冷了，记得加衣。", "窗台的花还开着。", "等你回来，再一起去看海。"]),
]
for _p in PAGES:
    assert "我" not in _p["title"] and not any("我" in s for s in _p["lines"])


def maps_path(name, root=None):
    return os.path.join(root or M.MAPS, name)


def srgb(a):
    a = np.clip(a, 0, 1)
    return np.where(a < .0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - .055)


def save_rgb(path, col):
    from PIL import Image
    Image.fromarray((srgb(col) * 255 + .5).astype(np.uint8)).save(path, compress_level=1)


def resample(tile, r, c):
    from scipy.ndimage import map_coordinates
    return map_coordinates(tile, np.stack([r, c]), order=1, mode="grid-wrap").astype(np.float32)


# ---- the floor -------------------------------------------------------------------------------------------------------------------------------------------
def floor():
    """Wide boards running from the front of the shop to the back wall, about 28 cm each, in lengths with butt joints, nailed. Old softwood gone dark: the soft part of each ring worn away, proud grain, edges rounded by two hundred years of boots, a paler path where the pressman stands, ink ground in round the machine."""
    x0, x1, y0, y1 = FLOOR_RECT
    px = FLOOR_PX
    nx, ny = int((x1 - x0) * px), int((y1 - y0) * px)
    r = np.random.default_rng(21)
    col_dark, col_pale = np.array((.034, .025, .018), np.float32), np.array((.135, .104, .078), np.float32)
    streaks = M.spectral(2048, 1.5, 31)
    warp = M.spectral(1024, 3.0, 32)
    pores = M.spectral(2048, .7, 33)
    col = np.zeros((ny, nx, 3), np.float32)
    h = np.zeros((ny, nx), np.float32)
    gap = np.zeros((ny, nx), np.float32)
    xs = x0 + (np.arange(nx) + .5) / px
    ys = y1 - (np.arange(ny) + .5) / px
    x = x0 - r.uniform(0, 40)
    while x < x1:
        w = r.uniform(70, 92)
        l, a = max(int((x - x0) * px), 0), min(int((x + w - x0) * px) + 1, nx)
        y = y0 - r.uniform(0, 900)
        while y < y1 and l < a:
            L = r.uniform(620, 1350)
            t, b = max(int((y1 - y - L) * px), 0), min(int((y1 - y) * px) + 1, ny)
            if t < b:
                X, Y = np.meshgrid(xs[l:a], ys[t:b])
                u, v = (X - x) / w, Y - y
                # growth rings: flat-sawn, a cathedral arch round a pith off to one side
                u0, v0 = r.uniform(-1.4, 2.4), r.uniform(0, L)
                ox, oy = r.uniform(0, 1024, 2)
                bend = resample(warp, v * .018 + oy, u * 22 + ox)
                rr = np.hypot(u - u0, (v - v0) * r.uniform(.0009, .0024)) + bend * .035
                rings = np.sin(rr * r.uniform(55, 95) * 2 * np.pi / 2)
                late = M.sm((rings + .15) / .7)                 # the hard late wood stands proud
                s = resample(streaks, v * .16 + r.uniform(0, 2048), X * 2.6 + r.uniform(0, 2048))
                p = resample(pores, v * .5 + r.uniform(0, 2048), X * 3.0 + r.uniform(0, 2048))
                tone = r.uniform(.55, 1.15)
                mix = np.clip(.34 * tone + late * .22 + s * .10, 0, 1)[..., None]
                plank = col_dark + (col_pale - col_dark) * mix
                plank = plank * (1 + p[..., None] * .10)
                to_side = np.minimum(u, 1 - u) * w
                to_end = np.minimum(v, L - v)
                edge = np.minimum(to_side, to_end)
                # worn edges, and cupped a little across the width, each board on its own
                hh = late * .12 + s * .05 + p * .02 - M.sm(1 - edge / 2.2) * .9 + (((u - .5) * 2) ** 2) * r.uniform(-.35, .35) + r.uniform(-.25, .25)
                # a knot or two
                for _ in range(int(r.integers(0, 3))):
                    ku, kv = r.uniform(.2, .8), r.uniform(.1, .9) * L
                    d = np.hypot((u - ku) * w / r.uniform(2.2, 4), (v - kv) / r.uniform(4, 7))
                    k = M.sm(1 - d)[..., None]
                    plank = plank * (1 - k * .72)
                    hh = hh + k[..., 0] * .1
                # nails: two at each joist, in a line right across the floor
                for jy in np.arange(np.ceil(y / 150) * 150, y + L, 150):
                    for ku in (.13, .87):
                        d = np.hypot((u - ku) * w, v - (jy - y))
                        k = M.sm(1 - d / 1.25)
                        plank = plank * (1 - k[..., None] * .8)
                        hh = hh - k * .25
                col[t:b, l:a] = plank
                h[t:b, l:a] = hh
                gap[t:b, l:a] = np.maximum(gap[t:b, l:a], M.sm(1 - edge / .9))
            y += L
        x += w
    # the wear of feet: paler, the grain knocked back, greyed by dust in the worn part
    X, Y = np.meshgrid(xs, ys)
    n1, n2 = M.spectral(max(nx, ny), 2.4, 34)[:ny, :nx], M.spectral(max(nx, ny), 1.6, 35)[:ny, :nx]
    path = np.exp(-(((X - 70) / 430) ** 2 + ((Y + 300) / 230) ** 2)) + .55 * np.exp(-(((X + 480) / 500) ** 2 + ((Y + 200) / 330) ** 2))
    worn = np.clip(path * (.75 + .25 * n1) * .75, 0, 1)
    dust = M.sm((n1 * .6 + n2 * .35 + .2) / 1.5) * .5 * (.45 + .55 * M.sm((Y - 120) / 260) + worn * .4)
    grey = col.mean(-1, keepdims=True)
    col = col + (grey * 1.5 + .012 - col) * (worn * .5)[..., None]
    col = col + (np.array((.115, .107, .096), np.float32) - col) * (dust * .34)[..., None]
    h = h * (1 - .45 * worn)
    # ink: ground into the boards under and round the machine, and dropped from the roller as far as it was carried
    near = np.exp(-(((X - 20) / 330) ** 2 + ((Y + 10) / 280) ** 2))
    stain = M.sm((M.spectral(max(nx, ny), 2.3, 36)[:ny, :nx] * .8 + near * 1.9 - 1.5) / .45)
    drops = np.zeros((ny, nx), np.float32)
    for _ in range(420):
        cx, cy = r.normal(40, 340), r.normal(-90, 290)
        rad = float(r.choice((r.uniform(.6, 1.6), r.uniform(1.4, 3.6), r.uniform(3, 8)), p=(.72, .23, .05))) * px
        if not (x0 < cx < x1 and y0 < cy < y1):
            continue
        c, ro = (cx - x0) * px, (y1 - cy) * px
        t, b, l, a = max(int(ro - rad * 2), 0), min(int(ro + rad * 2) + 1, ny), max(int(c - rad * 2), 0), min(int(c + rad * 2) + 1, nx)
        if t >= b or l >= a:
            continue
        yy, xx = np.mgrid[t:b, l:a]
        ang = np.arctan2(yy - ro, xx - c)
        d = np.hypot(yy - ro, xx - c) / (rad * (1 + .22 * np.sin(ang * r.integers(3, 7) + r.uniform(0, 6))))
        drops[t:b, l:a] = np.maximum(drops[t:b, l:a], M.sm((1 - d) / .25) * r.uniform(.6, 1))
    ink = np.clip(np.maximum(stain * .85, drops), 0, 1) * (1 - gap * .5)
    col = col + (np.array((.0055, .0055, .006), np.float32) - col) * ink[..., None]
    # everything collects dirt along the back wall
    col = col * (1 - .5 * M.sm((Y - 340) / 80))[..., None]
    col = col * (1 - gap[..., None] * .92)
    h = h - gap * 1.2
    rough = .80 + .05 * np.clip(n2, -1, 1) - .10 * worn - .24 * ink
    save_rgb(maps_path("floor_col.png"), col)
    M.png16(maps_path("floor_hr.png"), np.stack([h / FLOOR_H + .5, rough, ink], -1))
    print("floor", col.shape, flush=True)


# ---- the wall --------------------------------------------------------------------------------------------------------------------------------------------
def wall():
    """Lime plaster put on by hand a long time ago: uneven at every scale, patched, cracked, fallen away here and there down to the rough coat, sooty at the top and damp at the foot. Grey, because nothing but the paper and the light should be warm."""
    from scipy.ndimage import gaussian_filter
    x0, x1, z0, z1 = WALL_RECT
    px = WALL_PX
    nx, ny = int((x1 - x0) * px), int((z1 - z0) * px)
    n = max(nx, ny)
    r = np.random.default_rng(55)
    noise = lambda beta, seed: M.spectral(n, beta, seed)[:ny, :nx]
    big, mid, fine, grit = noise(3.2, 56), noise(2.3, 57), noise(1.6, 58), noise(.8, 59)
    # strokes of the float: long shallow scoops, each its own direction
    scoops = np.zeros((ny, nx), np.float32)
    for _ in range(1400):
        cy, cx, a = r.uniform(0, ny), r.uniform(0, nx), r.uniform(0, np.pi)
        L, W_ = r.uniform(40, 170) * px, r.uniform(9, 26) * px
        t, b, l, m = max(int(cy - L), 0), min(int(cy + L) + 1, ny), max(int(cx - L), 0), min(int(cx + L) + 1, nx)
        if t >= b or l >= m:
            continue
        yy, xx = np.mgrid[t:b, l:m]
        u, v = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a), -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
        d = (u / L) ** 2 + (v / W_) ** 2
        scoops[t:b, l:m] += (np.exp(-d * 3) * (d < 1)).astype(np.float32) * r.uniform(-1, 1)
    scoops = gaussian_filter(scoops, 1.2)
    h = big * 3.2 + mid * 1.1 + scoops * .9 + fine * .34 + grit * .10
    # patches, each mixed and floated on another day: a slightly different grey, a step at the edge
    patches = M.sm((noise(3.0, 60) - .9) / .08)
    h = h + patches * .9
    # where it has come away: the rough coat behind, a clear edge
    loss_field = noise(2.8, 61) + noise(1.5, 62) * .14
    Z = z1 - (np.arange(ny)[:, None] + .5) / px
    X = x0 + (np.arange(nx)[None, :] + .5) / px
    low = np.clip((-60 - Z) / 260, 0, 1) * .5
    loss = M.sm((loss_field + low - 2.45) / .05)
    h = h - loss * 3.5 + loss * noise(1.1, 63) * .7
    # cracks: the zero lines of one field, where another lets them run
    c_f = noise(2.5, 64)
    crack = np.exp(-(c_f / .012) ** 2) * M.sm((noise(2.8, 65) - .55) / .25)
    crack = gaussian_filter(crack.astype(np.float32), .6)
    h = h - crack * 1.6
    base = np.array((.215, .205, .185), np.float32)
    col = base[None, None] * (1 + mid[..., None] * .10 + big[..., None] * .13 + grit[..., None] * .025 + fine[..., None] * .05)
    col = col + (np.array((.265, .252, .222), np.float32) - col) * (patches * .55)[..., None]
    col = col + (np.array((.15, .138, .12), np.float32) - col) * loss[..., None] * .8
    col = col * (1 - .72 * crack)[..., None]
    # soot from above, a tide of damp and old dirt from the floor, a few long runs down the wall
    soot = M.sm((Z - 180) / 420) * (.65 + .35 * np.clip(big, -1, 1))
    damp = M.sm((-120 - Z + mid * 40) / 150)
    col = col * (1 - .62 * soot)[..., None] * (1 - .55 * damp)[..., None]
    # long faint runs down from the top of the wall, each of its own width and length, and never a straight line
    wander = gaussian_filter(M.spectral(n, 2.6, 68)[:ny, :nx], (10, 2))
    drip = gaussian_filter(r.normal(size=(1, nx)).astype(np.float32), (0, 16.0))
    drip = (drip - drip.mean()) / drip.std()
    reach = M.sm((Z - (250 + 180 * gaussian_filter(r.normal(size=(1, nx)).astype(np.float32), (0, 30.0)) * 12)) / 160)
    col = col * (1 - .11 * M.sm((drip + wander * .8 - 1.0) / .9) * reach * (1 - loss))[..., None]
    rough = np.full_like(h, .93)
    save_rgb(maps_path("wall_col.png"), col)
    M.png16(maps_path("wall_hr.png"), np.stack([h / WALL_H + .5, rough, loss], -1))
    print("wall", col.shape, flush=True)


# ---- the printed sheets ----------------------------------------------------------------------------------------------------------------------------------
def page(k, spec, wpx=1000):
    """One sheet off the press: a heading and a body, set large, as a broadside is, so that even small in the frame it reads as print. The sheet's paper isn't quite even."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import gaussian_filter
    hpx = int(wpx * SHEET_L / SHEET_W)
    scale = wpx / SHEET_W
    im = Image.new("L", (wpx, hpx), 0)
    d = ImageDraw.Draw(im)
    r = np.random.default_rng(300 + k)
    margin = 13 * scale
    title, body = M.font(spec["font"], 13.5 * scale), M.font(spec["font"], 6.1 * scale)
    if spec.get("vertical"):
        x = wpx - margin - 6 * scale
        y = margin + 8 * scale
        for ch in spec["title"]:
            d.text((x, y), ch, 255, font=title, anchor="mm")
            y += 15.5 * scale
        x -= 19 * scale
        for line in spec["lines"]:
            y = margin + 4 * scale
            for ch in line:
                if ch in "，。":
                    y += 3.5 * scale
                    continue
                d.text((x, y), ch, 255, font=body, anchor="mm")
                y += 7.0 * scale
            x -= 13.5 * scale
    else:
        y = margin + 12 * scale
        d.text((wpx / 2, y), spec["title"], 255, font=title, anchor="ms")
        y += 18 * scale
        for line in spec["lines"]:
            y += 11.5 * scale
            if line:
                f = body
                while f.getlength(line) > wpx - 2 * margin:
                    f = f.font_variant(size=f.size - 1)
                d.text((margin, y), line, 255, font=f, anchor="ls")
    a = np.asarray(im, np.float32) / 255
    # hand-inked: a little uneven, a little starved at places
    a = a * (.86 + .14 * M.sm((M.spectral(max(wpx, hpx), 2.2, 310 + k)[:hpx, :wpx] + .4) / 1.4))
    cloud = M.spectral(max(wpx, hpx), 2.8, 320 + k)[:hpx, :wpx]
    fibre = M.spectral(max(wpx, hpx), 1.1, 330 + k)[:hpx, :wpx]
    paper = np.array((.26, .24, .195), np.float32)[None, None] * (1 + cloud[..., None] * .03 + fibre[..., None] * .025)
    col = paper + (np.array((.012, .012, .014), np.float32) - paper) * a[..., None]
    save_rgb(maps_path(f"dry_{k}.png"), col)


def labels():
    """The cards in the type case's brass holders, one to a drawer, written in by hand long ago."""
    from PIL import Image, ImageDraw
    cw, ch = 256, 96
    im = Image.new("L", (cw * LABEL_COLS, ch * LABEL_ROWS), 0)
    d = ImageDraw.Draw(im)
    f = M.font("kai", 62)
    for i, text in enumerate(LABELS):
        d.text((cw * (i % LABEL_COLS) + cw / 2, ch * (i // LABEL_COLS) + ch / 2), text, 255, font=f, anchor="mm")
    a = np.asarray(im, np.float32)[..., None] / 255
    paper = np.array((.13, .12, .095), np.float32)[None, None]
    save_rgb(maps_path("case_labels.png"), paper + (np.array((.02, .02, .025), np.float32) - paper) * a)


def label_uv(i):
    """(u0, v0, u1, v1) of drawer i's label in the atlas."""
    c, ro = i % LABEL_COLS, i // LABEL_COLS
    return c / LABEL_COLS, 1 - (ro + 1) / LABEL_ROWS, (c + 1) / LABEL_COLS, 1 - ro / LABEL_ROWS


def bake():
    os.makedirs(M.MAPS, exist_ok=True)
    floor()
    wall()
    for k, spec in enumerate(PAGES):
        page(k, spec)
    labels()
    print("finished", flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "bake":
        bake()
    elif what in ("floor", "wall", "labels"):
        globals()[what]()
    elif what == "pages":
        for k, spec in enumerate(PAGES):
            page(k, spec)
