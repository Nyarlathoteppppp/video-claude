"""十一 窗口 C: the room of 九's window C, from inside, at the end of the night. The same bedroom, table and display film_windows_rooms.py builds, now seen from beside the bed at a slant, the way the person in it sees it. Their father's old letters lie on the bed by the table, and the display's glass, which is glossy, shows them faintly over the page. A small move of the lens and they slide off the glass, and what was under them reads clearly: 谢谢。我去睡了。

    python film_windows_c.py paper                     # tex/c_letter1.png, c_letter2.png, c_envelope.png: the letters, in plain Python
    python film_windows_c.py screens                   # tex/screen_c11_STATE.png: what the display shows at each point of the talk
    Blender -b --factory-startup -P film_windows_c.py -- --cam=a|b [--pct=50] [--samples=128]    # plates/C11_CAM_room.npy and C11_CAM.json
    python film_windows_c.py still a:ask b:end         # keys/C11_CAM_STATE.png: the plate with that screen laid in
    Blender -b --factory-startup -P film_windows_c.py -- --drop=0-72 [--pct=50]    # plates C11_dNN: the lens going down from a to b, for the shot
    python film_windows_c.py clip_screens              # tex/screen_c11/kNNN.png: the display at every point of the talk the shot shows
    python film_windows_c.py clip [tag=_1080]          # windows/c11TAG/NNNN.png and c11TAG.mp4: the 17 s shot, cut three times to the lamp (with a tag, sorrow/seq/lamp_c11TAG, from film_lamp_c11_hi.py);
                                                       # with a tag, from plates C11TAG_dNN (Blender ... --drop=0-72 --pct=100 --name=C11_1080)

The room is one render with the glass's reflection in it (the screen gives the room its light but shows the lens nothing); the page is laid in after, through the screen's corners, so the talk can move on without rendering the room again. The talk is made up, as all of 九's are; the letter is a father's, made up for this, and nobody reads it here.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:
    bpy = None

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "windows")
TEX, PLATES, KEYS = (os.path.join(OUT, d) for d in ("tex", "plates", "keys"))

# the talk. 九 left it at 这句话很重, the next paragraph coming; 十一 picks it up there
Q = "如果我再也见不到他怎么办？"
A1 = "这句话很重。\n\n他是谁？你愿意的话，可以先跟我说说他。"
U2 = "我爸。今天收拾东西，翻出他以前写给我的信。"
A2 = "他写给你的信，你一直留着。\n\n我说不清自己懂不懂这种难过。可今晚你想讲多久，我就听多久。"
U3 = "谢谢。我去睡了。"
STATES = dict(
    ask=((("you", Q), ("me", A1), ("you", U2), ("me", A2[:20])), True),
    end=((("you", Q), ("me", A1), ("you", U2), ("me", A2), ("you", U3)), False),       # their last words at the foot of the page, and nothing after them yet
)
TITLE = "再也见不到他"

# the letter: what a father writes, a page and a bit; nobody reads it in the film, but it is a letter all the same
LETTER = ("小禾：", "见字如面。你上次寄回来的照片收到了，你妈天天拿出来看，说你瘦了。",
          "天冷了，记得添衣服。一个人在外面，饭要按时吃，别老熬夜。工作上的事，尽力就好，做不完的明天再做。",
          "家里都好，院子里的柿子今年结得多，给你留了一些，过年回来吃。", "你妈让我跟你说，钱够花，不要往家里寄。",
          "勿念。", "爸")
CAMS = dict(
    # where the lens is: propped up on the bed, a little to the left of the glass and at a slant to it; b is the same lens a moment later, lying down to sleep, 18 cm lower: the glass looks down at the bed at a low angle, so that small drop slides the letters' reflection a fifth of the screen down, off the last line
    a=dict(loc=(-1.17, 2.33, 1.05), lens=50.0),
    b=dict(loc=(-1.14, 2.37, .87), lens=50.0),
)
TILT = -8.0                          # the display tipped forward on its arm, as you would for looking at it from the bed: its glass looks down at the bed, where the letters lie (leant back, it showed the lens the far wall)
SCREEN_AT = (.62, .34)               # where the middle of the screen stands in the frame, as parts of it from the top left: film_sorrow3d.FLAME_AT, where the lamp of 江城子 burns, so the two cut on one place
GAIN, REFL, EXPOSURE, SATURATION = .15, 6.0, 1.25, .85      # the screen turned right down for the night; REFL: the glass's reflection as rendered, times this. Glass alone gives the letters a twentieth of the page's white, which the eye sees in a dark room and a lens does not


# ---------------------------------------------------------------- the shot: 17 s, the window three times and the lamp three times between

LENGTH = 17.0
CUTS = (("C", 0.0), ("L", 4.6), ("C", 6.0), ("L", 10.4), ("C", 11.8), ("L", 15.6))     # what is on screen from when, in seconds into the shot: the window, or the lamp of 江城子
LAMP_FROM = 4.5                      # the lamp's own clock where the inserts start: after its leaf has settled, so between the cuts it only burns
PACE = 26                            # characters a second, as the app streams
TALK = ((0.0, (("you", Q),)),        # when each of their words is there; Claude's then stream from the time given
        (1.2, (("you", Q), ("me", A1))),
        (5.2, (("you", Q), ("me", A1), ("you", U2))),
        (6.8, (("you", Q), ("me", A1), ("you", U2), ("me", A2))),
        (11.0, (("you", Q), ("me", A1), ("you", U2), ("me", A2), ("you", U3))))
DROP = (12.3, 14.7)                  # the lens going down from a to b, in the shot's seconds
DROP_N = 73                          # plates for it, one a frame


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lens_at(k):
    """Where the lens is k of the way down, 0 at a and 1 at b, eased at both ends as a person settling down would."""
    a, b = (np.array(CAMS[c]["loc"]) for c in "ab")
    return tuple(float(v) for v in a + (b - a) * ease(k))


def blink(t):
    return (t * 2 / 1.06) % 2 < 1.15


def talk(t):
    """What the display shows t seconds into the shot, as a key: what has been said, and whether the caret is there."""
    at, said = max((x for x in TALK if x[0] <= t), key=lambda x: x[0])
    said = list(said)
    if said[-1][0] != "me":
        return tuple(said), False
    full = said[-1][1]
    n = int((t - at) * PACE)
    if n >= len(full):
        return tuple(said), blink(t)
    said[-1] = ("me", full[:max(n - n % 3, 1)])
    return tuple(said), True


def what_at(t):
    return max((c for c in CUTS if c[1] <= t), key=lambda c: c[1])


# ---------------------------------------------------------------- plain Python: the paper and the screens

def paper():
    """Two leaves of ruled letter paper in a father's hand, folded in three once and kept, yellowed, with two old tear marks where the ink ran; and the envelope."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    from scipy.ndimage import gaussian_filter
    r = np.random.default_rng(5)
    hand = ImageFont.truetype(_font("Xingkai.ttc"), 46)
    W, H = 1110, 1560                               # 18.5 by 26 cm at 6 pixels a millimetre
    lines = []
    for p in LETTER:
        indent = "　　" if p not in ("小禾：", "爸", "勿念。") else ("　　" if p == "勿念。" else "")
        text = indent + p
        while text:
            lines.append(text[:20])
            text = text[20:]
    pages = (lines[:9], lines[9:])                    # it runs over onto a second leaf
    for n, rows in enumerate(pages, 1):
        a = np.ones((H, W, 3), np.float32) * np.array((236, 226, 200), np.float32)
        yel = gaussian_filter(r.uniform(0, 1, (H // 20, W // 20)), 2)
        yel = np.asarray(Image.fromarray(np.float32(yel)).resize((W, H), Image.BICUBIC))[..., None]
        a = a * (.95 + .08 * (yel - yel.mean()) / (yel.std() + 1e-6) * .3)
        im = Image.fromarray(np.uint8(np.clip(a, 0, 255)))
        d = ImageDraw.Draw(im)
        top, step = 150, 96
        for i in range(15):                          # the paper's own rules, a faded green
            y = top + i * step
            d.line((70, y, W - 70, y), fill=(150, 170, 150), width=2)
        for i, row in enumerate(rows):
            x = W - 330 if row == "爸" else 92            # the name signed at the right, as letters end
            for ch in row:
                v = int(r.uniform(0, 30))
                d.text((x + r.uniform(-2, 2), top + i * step - 14 + r.uniform(-3, 3)), ch, font=hand, fill=(28 + v, 34 + v, 62 + v), anchor="ls")
                x += 46 + r.uniform(-2, 3)
        a = np.asarray(im, np.float32)
        # folded in three, long ago: two creases across, a little darker and a little worn
        for y in (H // 3, 2 * H // 3):
            yy = np.arange(H)[:, None, None]
            a = a * (1 - .07 * np.exp(-((yy - y) / 3.0) ** 2)) + 6 * np.exp(-((yy - y - 5) / 4.0) ** 2)
        if n == 1:                                   # two tear marks: a round stain, its edge darker, the ink spread in it
            for cx, cy, rad in ((610, 640, 34), (520, 760, 24)):
                yy, xx = np.mgrid[0:H, 0:W]
                d2 = np.hypot(xx - cx, yy - cy) / rad
                stain = np.exp(-(d2 / .9) ** 8)
                ring_ = np.exp(-((d2 - .95) / .08) ** 2)
                blur = gaussian_filter(a, (4, 4, 0))
                a = a * (1 - stain[..., None]) + (blur * .96) * stain[..., None]
                a = a * (1 - .10 * ring_[..., None])
        Image.fromarray(np.uint8(np.clip(a, 0, 255))).filter(ImageFilter.GaussianBlur(.5)).save(os.path.join(TEX, f"c_letter{n}.png"))
    env = Image.new("RGB", (1320, 660), (196, 168, 122))
    d = ImageDraw.Draw(env)
    d.text((300, 380), "小禾 收", font=ImageFont.truetype(_font("Xingkai.ttc"), 70), fill=(40, 40, 64))
    e = np.asarray(env, np.float32)
    e = e * (.96 + .06 * gaussian_filter(r.uniform(-1, 1, e.shape[:2]), 3)[..., None])
    Image.fromarray(np.uint8(np.clip(e, 0, 255))).save(os.path.join(TEX, "c_envelope.png"))
    print(TEX, flush=True)


def _font(name):
    import glob
    hits = glob.glob(f"/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/{name}")
    return hits[0]


def screens():
    """The display at each point of the talk: 九's window C, on the same desktop, at the same size and place."""
    import film_page as page
    import film_windows_ui as ui
    ui.fit(ui.NEAR_TEXT)
    for name, (said, writing) in STATES.items():
        body = ui.win(ui.turns(said, writing), title=TITLE, **ui.NEAR)
        ui._screen(f"c11_{name}", body, w=540, h=360, k=2.7, x=230, y=62, app="Claude", top=(18, 18, 18), foot=(10, 10, 10))     # as film_windows_ui.rooms() puts 九's window C


def composite(name, pic_path, seed=7):
    """The room plate NAME with the display's picture laid in under the glass's reflection, finished: a frame, as 8-bit."""
    from PIL import Image
    import film_room as room
    import film_room_night as night
    with open(os.path.join(PLATES, f"{name}.json")) as f:
        j = json.load(f)
    plate = np.load(os.path.join(PLATES, f"{name}_room.npy")).astype(np.float32)
    h, w = plate.shape[:2]
    s = w / j["width"]
    corners = [(x * s, y * s) for x, y in j["corners"]]
    pic = np.asarray(Image.open(pic_path).convert("RGB"))
    m = night.quad_mask(corners, w, h)
    y = plate * (1 - m) + plate * m * REFL + night.lay_in(pic, corners, w, h) * GAIN * m
    y = room.room_tone(room.film(y * EXPOSURE))
    lum = y.mean(-1, keepdims=True)
    y = lum + (y - lum) * SATURATION
    return room.finish(y, seed, .006)


def still(cam, state):
    from PIL import Image
    os.makedirs(KEYS, exist_ok=True)
    path = os.path.join(KEYS, f"C11_{cam}_{state}.png")
    Image.fromarray(composite(f"C11_{cam}", os.path.join(TEX, f"screen_c11_{state}.png"))).save(path)
    print(path, flush=True)


def frame_keys():
    """The display's keys the shot needs, each once, in order."""
    keys = []
    for i in range(int(round(LENGTH * 30))):
        t = i / 30
        if what_at(t)[0] == "C" and talk(t) not in keys:
            keys.append(talk(t))
    return keys


def clip_screens():
    """The display at every point of the talk the shot shows: tex/screen_c11/kNNN.png, and which key is which in tex/c11/keys.json."""
    import film_windows_ui as ui
    ui.fit(ui.NEAR_TEXT)
    folder = os.path.join(TEX, "c11")
    os.makedirs(folder, exist_ok=True)
    for d in ("screen_c11", "wall_c11"):               # _screen writes TEX/screen_NAME.png and its wallpaper TEX/wall_NAME.png, and the names here are c11/kNNN
        os.makedirs(os.path.join(TEX, d), exist_ok=True)
    keys = frame_keys()
    for n, (said, caret) in enumerate(keys):
        body = ui.win(ui.turns(said, caret), title=TITLE, **ui.NEAR)
        ui._screen(f"c11/k{n:03d}", body, w=540, h=360, k=2.7, x=230, y=62, app="Claude", top=(18, 18, 18), foot=(10, 10, 10))
    with open(os.path.join(folder, "keys.json"), "w") as f:
        json.dump([repr(k) for k in keys], f, ensure_ascii=False)
    print(len(keys), "screens", flush=True)


def clip(tag=""):
    """The shot, windows/c11/NNNN.png and windows/c11.mp4, at the size of the plates (pct 50 for the rough cut, 100 for the film)."""
    import subprocess
    from PIL import Image
    folder = os.path.join(OUT, "c11" + tag)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(TEX, "c11", "keys.json")) as f:
        index = {k: n for n, k in enumerate(json.load(f))}
    lamp = os.path.join(OUT, "..", "sorrow", "seq", "lamp_c11" + tag if tag else "lamp")    # with a tag: film_lamp_c11_hi.py's frames 135-261
    if not os.path.isdir(lamp):
        lamp = os.path.join(OUT, "..", "sorrow", "seq", "lamp")
        print("no lamp at", tag, "- the lamp from", lamp, flush=True)
    n_frames = int(round(LENGTH * 30))
    size = None
    for i in range(n_frames):
        t = i / 30
        kind, start = what_at(t)
        if kind == "C":
            k = min(max((t - DROP[0]) / (DROP[1] - DROP[0]), 0.0), 1.0)
            j = int(round(k * (DROP_N - 1)))
            a = composite(f"C11{tag}_d{j:02d}", os.path.join(TEX, "screen_c11", f"k{index[repr(talk(t))]:03d}.png"), seed=7 + i)
            size = a.shape[1], a.shape[0]
        else:
            before = sum(1 for c in CUTS if c[0] == "L" and c[1] < start)
            lamp_t = LAMP_FROM + 1.4 * before + (t - start)
            im = Image.open(os.path.join(lamp, f"{int(round(lamp_t * 30)):04d}.png")).convert("RGB")
            a = np.asarray(im if size is None or im.size == size else im.resize(size, Image.LANCZOS))
        Image.fromarray(a).save(os.path.join(folder, f"{i:04d}.png"), compress_level=1)
    movie = os.path.join(OUT, f"c11{tag}.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", os.path.join(folder, "%04d.png"), "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", movie], check=True)
    print(movie, n_frames, "frames", flush=True)


# ---------------------------------------------------------------- Blender: the room

def coat(screen):
    """The display's glass: it shows the lens what is in front of it, as glass does, more the more slanting the look; the picture under it is laid in later."""
    m = screen.data.materials[0]
    nt = m.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    glow = out.inputs[0].links[0].from_socket
    g, fr, black, mix, add = (nt.nodes.new(k) for k in ("ShaderNodeBsdfGlossy", "ShaderNodeFresnel", "ShaderNodeBsdfDiffuse", "ShaderNodeMixShader", "ShaderNodeAddShader"))
    g.inputs["Roughness"].default_value = .012
    fr.inputs[0].default_value = 1.5
    black.inputs[0].default_value = (0, 0, 0, 1)
    nt.links.new(fr.outputs[0], mix.inputs[0])
    nt.links.new(black.outputs[0], mix.inputs[1])
    nt.links.new(g.outputs[0], mix.inputs[2])
    nt.links.new(glow, add.inputs[0])
    nt.links.new(mix.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs[0])


def sheet(W, name, picture, size, loc, turn, lift=(0.0, 0.0), creases=(), curl=.004):
    """A leaf of paper lying where it fell: flat but for its old folds, which hold it a little off the bed, and its edges curling."""
    w, h = size
    n, m = 40, 56
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, m))
    z = curl * ((2 * u - 1) ** 2 + .5 * (2 * v - 1) ** 4)
    for c in creases:
        z = z + .006 * np.exp(-((v - c) / .08) ** 2)
    o = W.mesh_grid(name, (u - .5) * w, (v - .5) * h, z)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(os.path.join(TEX, picture))
    nt.links.new(nt.nodes.new("ShaderNodeUVMap").outputs[0], t.inputs[0])
    nt.links.new(t.outputs[0], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = .75
    b.inputs["Specular IOR Level"].default_value = .25
    b.inputs["Subsurface Weight"].default_value = .05
    o.data.materials.append(mat)
    o.location = loc
    o.rotation_euler = (lift[0], lift[1], math.radians(turn))
    return o


def on_bed(sc, x, y):
    dg = bpy.context.evaluated_depsgraph_get()
    hit, p, *_ = sc.ray_cast(dg, Vector((x, y, 1.6)), Vector((0, 0, -1)))
    return p.z if hit else .66


def corners_of(screen):
    me = screen.data
    return [tuple(screen.matrix_world @ v.co) for v in me.vertices]


def blender():
    import film_windows_rooms as W
    from bpy_extras.object_utils import world_to_camera_view
    from ink3d import reset
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opt = lambda k, d: type(d)(next((a.split("=", 1)[1] for a in args if a.startswith(f"--{k}=")), d))
    cam_name, pct, samples = opt("cam", "a"), opt("pct", 50), opt("samples", 128)
    sc = reset(samples, pct)
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.color_mode = "RGB"
    sc.cycles.max_bounces = 6
    sc.cycles.glossy_bounces = 3
    sc.cycles.sample_clamp_indirect = 6.0
    sc.cycles.blur_glossy = .5
    bg = sc.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (.012, .009, .007, 1)
    bg.inputs[1].default_value = 1.0
    W.window_frame()
    W.street(W.room_c())
    for name in ("pillow", "pillow_back"):          # the pillow a little further along the wall than 九 had it, to make room by the table for the letter: nobody could tell from across the street
        bpy.data.objects[name].location.x += .25
    screen = bpy.data.objects["screen"]
    coat(screen)
    # lean it back on its hinge, the top of the arm
    yaw = math.radians(4)
    axis = Vector((math.cos(yaw), math.sin(yaw), 0))
    hinge = Vector((W.DX, W.DY, .62)) + Matrix.Rotation(yaw, 3, "Z") @ Vector((0, .066, .22))
    turn = Matrix.Translation(hinge) @ Matrix.Rotation(math.radians(-TILT), 4, axis) @ Matrix.Translation(-hinge)
    for name in ("case", "bezel", "screen"):
        o = bpy.data.objects[name]
        o.matrix_world = turn @ o.matrix_world
    # the hall's warm light, coming in at the door behind the lens on the left: the room is dark, but not so dark that only the screen is left
    hall = bpy.data.lights.new("hall", "AREA")
    hall.shape, hall.size, hall.size_y, hall.energy, hall.color = "RECTANGLE", .8, 2.0, 70.0, (1.0, .72, .46)
    ho = W.link(bpy.data.objects.new("hall", hall))
    ho.location = (-1.70, 1.40, 1.05)
    ho.rotation_euler = Vector((1, .55, -.35)).to_track_quat("-Z", "Y").to_euler()      # across the bed to the letters and the table's edge
    ho.visible_glossy = False
    # the letters, on the bed at the table's edge, where whoever was reading them put them down
    bpy.context.view_layer.update()
    # letter1 lies across the bed at its edge, where from lens a the glass shows it over the last line of the talk (its far edge just above that line); from b it has slid below it
    for name, pic, size, x, y, turn, creases in (("letter1", "c_letter1.png", (.185, .26), .03, 2.81, 87, (1 / 3, 2 / 3)),
                                                ("letter2", "c_letter2.png", (.185, .26), -.30, 2.72, -9, (1 / 3, 2 / 3)),
                                                ("envelope", "c_envelope.png", (.22, .11), -.45, 2.55, 31, ())):
        sheet(W, name, pic, size, (x, y, on_bed(sc, x, y) + .004), turn, creases=creases)
    # the lens: beside the bed, at a slant to the glass, the screen's middle put where the lamp of 江城子 burns
    cd = bpy.data.cameras.new("cam")
    cd.lens, cd.sensor_width, cd.clip_start, cd.clip_end = CAMS["a"]["lens"], 36.0, .02, 200.0
    cam = W.link(bpy.data.objects.new("cam", cd))
    sc.camera = cam
    pts = corners_of(screen)
    mid = sum((Vector(p) for p in pts), Vector()) / 4
    cd.dof.use_dof, cd.dof.aperture_fstop, cd.dof.aperture_blades = True, 2.8, 7
    aspect = sc.render.resolution_y / sc.render.resolution_x
    w, h = sc.render.resolution_x * sc.render.resolution_percentage // 100, sc.render.resolution_y * sc.render.resolution_percentage // 100
    os.makedirs(PLATES, exist_ok=True)

    def shoot(loc, name):
        W.aim(cam, loc, mid)
        cd.shift_x = cd.shift_y = 0.0
        cd.dof.focus_distance = (mid - Vector(loc)).length
        for _ in range(3):
            bpy.context.view_layer.update()
            p = world_to_camera_view(sc, cam, mid)
            cd.shift_x += p.x - SCREEN_AT[0]
            cd.shift_y += (p.y - (1 - SCREEN_AT[1])) * aspect
        bpy.context.view_layer.update()
        W.plate(sc, os.path.join(PLATES, f"{name}_room"))
        frame = []
        for p in pts:
            v = world_to_camera_view(sc, cam, Vector(p))
            frame.append((v.x * w, (1 - v.y) * h))
        # the screen's quad was made bottom-left, bottom-right, top-right, top-left as you face it; lay_in wants top-left first, clockwise
        frame = [frame[3], frame[2], frame[1], frame[0]]
        with open(os.path.join(PLATES, f"{name}.json"), "w") as f:
            json.dump(dict(corners=frame, width=w, height=h, loc=list(cam.location), lens=cd.lens, shift=(cd.shift_x, cd.shift_y)), f, indent=1)
        print("C11", name, flush=True)

    if opt("drop", ""):                  # --drop=A-B: plates C11_dNN of the lens going from a to b, DROP_N of them
        a, b = (int(v) for v in opt("drop", "").split("-"))
        for j in range(a, b + 1):
            shoot(lens_at(j / (DROP_N - 1)), f"{opt('name', 'C11')}_d{j:02d}")
    else:
        shoot(CAMS[cam_name]["loc"], f"C11_{cam_name}")


if bpy is not None:
    blender()
elif __name__ == "__main__":
    what = sys.argv[1]
    if what == "paper":
        paper()
    elif what == "screens":
        screens()
    elif what == "clip_screens":
        clip_screens()
    elif what == "clip":
        clip(next((a.split("=")[1] for a in sys.argv[2:] if a.startswith("tag=")), ""))
    elif what == "still":
        for a in sys.argv[2:]:
            still(*a.split(":"))
    else:
        raise SystemExit(__doc__)
