"""十一 这份难过是谁的: the three sorrows, each one thing changing, nobody and no words of the poem on screen.

    petal   感时花溅泪，恨别鸟惊心: one apricot petal comes down on the 家书 of 三, the same sheet on the same old table under the same low light, by 念之在心
    rain    梧桐更兼细雨，到黄昏、点点滴滴: under a bar of an old lattice window, at dusk in the rain, two drops run together into one and it falls out of the frame
    lamp    十年生死两茫茫: a lamp that has not gone out, an oil lamp by an open book at night, and a leaf of the book the air lifts and turns over

    Blender -b --factory-startup -P film_sorrow3d.py -- --shot=petal|rain|lamp [--t=SECONDS] [--pct=100] [--samples=256]
    Blender -b --factory-startup -P film_sorrow3d.py -- --shot=petal --frames=0-134 [--pct=50] [--samples=96]

Stills go to ink/film_out/film/sorrow/SHOT_T.png, frames to sorrow/seq/SHOT/NNNN.png. Everything moving is a function of t, seconds into the shot, so a still at t is a frame of the shot.
The letter and its table are film_docs3d.py's, built by its own functions, so the sheet is the same sheet. The book by the lamp is set in Songti TC from the same 與子儼等疏, the one text the film has read off its source page.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
try:
    import bpy
    from mathutils import Matrix, Vector
    from ink3d import ROOT, aim, mesh_grid, reset
except ImportError:                    # plain Python: only `pages`, the book's leaves, which need PIL and film_docs
    bpy = None
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


SHOT = opt("shot", "petal")
PCT, SAMPLES = opt("pct", 100), opt("samples", 256)
T = opt("t", -1.0)
FRAMES = opt("frames", "")
FPS = 30
OUT = os.path.join(ROOT, "ink", "film_out", "film", "sorrow")
LENGTH = dict(petal=4.5, rain=4.5, lamp=4.5)            # each as long as its line


def docs3d():
    """film_docs3d.py's functions."""
    import film_docs3d                 # its main() runs only as a script
    return film_docs3d


def sm(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def surface(name, colour, rough=.5, metal=0.0, spec=.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*colour, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["Specular IOR Level"].default_value = spec
    return m


def link(o):
    bpy.context.collection.objects.link(o)
    return o


def lathe(name, profile, loc, mat, n=96):
    p = np.array(profile, np.float32)
    a = np.linspace(0, 2 * math.pi, n + 1)
    r = p[:, 0][:, None]
    o = mesh_grid(name, loc[0] + r * np.cos(a), loc[1] + r * np.sin(a), loc[2] + np.broadcast_to(p[:, 1][:, None], (len(p), n + 1)))
    o.data.materials.append(mat)
    return o


def box(name, x0, x1, y0, y1, z0, z1, mat, matrix=None):
    bpy.ops.mesh.primitive_cube_add(size=1)
    o = bpy.context.object
    o.name = name
    o.scale = (x1 - x0, y1 - y0, z1 - z0)
    o.location = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    if matrix is not None:
        bpy.context.view_layer.update()
        o.matrix_world = matrix @ o.matrix_world
    o.data.materials.append(mat)
    return o


def camera(loc, target, lens, fstop, focus=None, roll=0.0):
    c = bpy.data.cameras.new("cam")
    c.lens, c.sensor_width, c.clip_start, c.clip_end = lens, 36.0, .005, 100.0
    c.dof.use_dof = True
    c.dof.focus_distance = focus if focus is not None else (Vector(target) - Vector(loc)).length
    c.dof.aperture_fstop = fstop
    c.dof.aperture_blades = 7
    o = link(bpy.data.objects.new("cam", c))
    aim(o, loc, target, roll)
    bpy.context.scene.camera = o
    return o


def place(cam, point, at):
    """Shift the lens so that a point of the world lands at a place in the frame, as parts of the frame from its top left."""
    from bpy_extras.object_utils import world_to_camera_view
    sc = bpy.context.scene
    aspect = sc.render.resolution_y / sc.render.resolution_x
    for _ in range(3):
        bpy.context.view_layer.update()
        p = world_to_camera_view(sc, cam, point)
        cam.data.shift_x += p.x - at[0]
        cam.data.shift_y += (p.y - (1 - at[1])) * aspect


# ---------------------------------------------------------------- 春望: the petal on the letter

def petal_mesh(name, L=.0165, W=.0122, seed=4):
    """An apricot petal: broadest two thirds of the way out and narrowing to a short claw at its foot, the tip with a small notch, cupped, its edge a little wavy; white with a blush at the foot. Its veins are only just there: drawn hard, the petal read as a shell."""
    r = np.random.default_rng(seed)
    n, m = 48, 40
    u = np.linspace(0, 1, n)[:, None]
    v = np.linspace(-1, 1, m)[None, :]
    edge = 1 + .035 * np.sin(u * 19 + r.uniform(0, 6)) * u + .02 * np.sin(u * 41 + r.uniform(0, 6)) * u      # no petal's edge is a clean curve
    half = W / 2 * np.sin(np.pi * np.clip(u, 0, 1) ** 1.5) ** .5 * (1 - .55 * np.exp(-(u / .12) ** 2)) * edge      # broadest well out, narrowing to a short claw
    notch = 1 - .11 * np.exp(-(v / .13) ** 2) * u ** 10
    x = u * L * notch
    y = v * half
    wave = .00018 * np.sin(v * 7 + r.uniform(0, 6)) * u ** 2
    crinkle = .00003 * np.sin(u * 23 + v * 9 + r.uniform(0, 6)) + .000012 * np.sin(u * 7 - v * 17 + r.uniform(0, 6))      # barely: any more and it reads as a pleated shell
    z = .0009 * v ** 2 * np.sin(np.pi * u ** .8) + .0007 * u ** 2.2 + wave + crinkle
    o = mesh_grid(name, x, y, z)
    s = o.modifiers.new("thin", "SOLIDIFY")
    s.thickness = .00006
    m_ = bpy.data.materials.new("petal")
    m_.use_nodes = True
    nt = m_.node_tree
    b = nt.nodes["Principled BSDF"]
    uv = nt.nodes.new("ShaderNodeUVMap")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs[0], sep.inputs[0])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = .0, .55
    ramp.color_ramp.elements[0].color = (.86, .50, .55, 1)        # the foot, pink
    ramp.color_ramp.elements[1].color = (.95, .84, .84, 1)        # the rest white with a blush in it
    nt.links.new(sep.outputs[1], ramp.inputs[0])                   # the mesh's v runs along the petal from its foot
    nt.links.new(ramp.outputs[0], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = .55
    b.inputs["Subsurface Weight"].default_value = .25
    b.inputs["Subsurface Radius"].default_value = (.002, .0015, .0012)
    b.inputs["Transmission Weight"].default_value = .35
    # its veins, fine and running from the foot
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "X"
    wave.inputs["Scale"].default_value = 44
    wave.inputs["Distortion"].default_value = 5.0
    wave.inputs["Detail"].default_value = 4.0
    nt.links.new(uv.outputs[0], wave.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .06
    nt.links.new(wave.outputs[1], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    o.data.materials.append(m_)
    return o


PETAL_REST = (159.0, 151.0)       # where on the letter it comes to rest, mm from the sheet's top left: just right of 在心, half over the column before it
PETAL_FALL = 3.2                  # seconds it takes to come down into the frame and settle


def petal_at(o, sheet, t):
    """Where the petal is t seconds into the shot: it comes in from the top of the frame, side-slipping and turning as a light thing falls, and lies down."""
    rest = sheet.at(*PETAL_REST) + Vector((0, 0, .00035))
    k = min(max(t / PETAL_FALL, 0.0), 1.0)
    down = 1 - (1 - k) ** 1.6
    h = .11 * (1 - down)
    sway = .016 * math.sin(2.6 * t + .4) * (1 - k) ** .7
    drift = Vector((-.022 * (1 - down) + sway, .012 * (1 - down), h))
    o.location = rest + drift
    tilt = .9 * (1 - k) ** 1.2
    o.rotation_euler = (tilt * math.sin(3.1 * t), tilt * .6 * math.cos(2.3 * t + .5), math.radians(28) + 1.1 * (1 - k) + .2 * math.sin(1.7 * t))


def shot_petal():
    D = docs3d()
    D.table()
    sheet = D.Sheet("letter")
    D.lights(D.POWER["letter"], .07)
    p = petal_mesh("petal")
    target = sheet.at(148.0, 160.0)                              # 念之在心 and the petal, the petal right of the middle; 若何可言 runs on down
    loc = D.tilted(sheet, target, 82, 85, 31)
    cam = D.shoot(loc, target, sheet.up, 85, stop=4.0, shift_y=-.02)
    cam.data.dof.focus_distance = (sheet.at(*PETAL_REST) - cam.location).length
    return lambda t: petal_at(p, sheet, t)


# ---------------------------------------------------------------- 声声慢: two drops under the lattice

def wood(name, colour, rough):
    m = surface(name, colour, rough, 0, .5)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc, mp, n, ramp, bump = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeMapping"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeValToRGB"), nt.nodes.new("ShaderNodeBump")
    mp.inputs["Scale"].default_value = (6, 160, 160)           # grain runs along x
    n.inputs["Scale"].default_value = 1.0
    n.inputs["Detail"].default_value = 10
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs[0], n.inputs["Vector"])
    ramp.color_ramp.elements[0].color = tuple(c * .65 for c in colour) + (1,)
    ramp.color_ramp.elements[1].color = tuple(min(c * 1.25, 1) for c in colour) + (1,)
    nt.links.new(n.outputs[0], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], b.inputs["Base Color"])
    bump.inputs["Strength"].default_value = .35
    bump.inputs["Distance"].default_value = .0004
    nt.links.new(n.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    return m


BAR_Z = -.011                     # the underside of the bar the drops hang from
DROPS = ((-.0058, .0023), (.0052, .0020))   # where the two hang along the bar and how big
MEET, FALL = 2.1, 3.0             # when they have run together, when the one drop lets go


def water():
    m = bpy.data.materials.new("water")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (1, 1, 1, 1)
    b.inputs["Roughness"].default_value = 0.0
    b.inputs["IOR"].default_value = 1.333
    b.inputs["Transmission Weight"].default_value = 1.0
    return m


MM = .001                          # the drops are built in millimetres in an object scaled down: Blender will not make metaballs much under a centimetre
BALL = 1.75                        # a lone ball's surface is about 1.14 of its radius across, so a drop of radius r wants a ball of r times this


def drops_at(mb, t):
    """The two drops as metaballs, which run together by themselves when they come close. Each is a body and a neck to the wood; they creep along the bar to each other, touch, become one that swells and lengthens, and it lets go and falls out of the frame, leaving a little behind."""
    mb.elements.clear()
    bar, y = BAR_Z / MM, .0085 / MM

    def ball(x, z, r):
        e = mb.elements.new()
        e.type, e.co, e.radius = "BALL", (x, y, z), r * BALL

    k = sm(t / MEET)
    (xa, ra), (xb, rb) = ((x / MM, r / MM) for x, r in DROPS)
    mid = (xa * ra ** 3 + xb * rb ** 3) / (ra ** 3 + rb ** 3)
    swell = sm((t - MEET) / (FALL - MEET))
    fall = max(t - FALL, 0.0)
    drop = 4900.0 * fall * fall                       # mm, as anything falls
    for x0, r in ((xa, ra), (xb, rb)):
        x = x0 + (mid - x0) * k
        sag = r * (.45 + .55 * swell)                 # how far below the wood its middle hangs: clinging flat at first, hanging once it has swollen
        if fall == 0:
            ball(x, bar + .2 * r, r * (.78 - .16 * swell))      # the neck, where it holds on: broad while it clings
            ball(x, bar - sag - .55 * r * swell, r * .92)
        ball(x, bar - sag - 1.2 * r * swell - drop, r * (.8 + .1 * swell))
    if fall > 0:
        ball(mid, bar + .1, .55)                      # what is left on the wood
    for x in (-24.0, -17.5, 15.0, 21.5, 28.0):        # the wet line along the bar's foot, the start of the next ones
        ball(x, bar + .15, .55 + .2 * ((x * 7) % 3) / 3)


def leaf_mesh(name, size, lobes=3):
    """A 梧桐 leaf: broad, palmate, three or five shallow lobes, a stalk; seen far out of focus it only has to have the shape."""
    n, m = 60, 60
    th = np.linspace(0, 2 * math.pi, m)[None, :]
    rr = np.linspace(0, 1, n)[:, None]
    lobe = 1 + .28 * np.cos(lobes * (th - math.pi / 2)) - .18 * np.exp(-((th - 3 * math.pi / 2) / .35) ** 2)
    x = size * rr * lobe * np.cos(th)
    y = size * rr * lobe * np.sin(th) * .95
    z = -.08 * size * (rr * lobe) ** 2 + .02 * size * np.sin(th * 3) * rr
    return mesh_grid(name, x, y, z)


def shot_rain():
    sc = bpy.context.scene
    old = wood("lattice", (.11, .065, .038), .28)
    # the lattice: a bar across, uprights, and the bars behind them, square to the lens
    box("bar", -.30, .30, -.012, .012, BAR_Z, BAR_Z + .022, old)
    box("bar_up", -.30, .30, -.012, .012, .108, .13, old)
    for x in (-.25, -.1, .027, .17):            # the drops hang between two of them, the one at .027 standing in the frame's right edge: the bar and it make the lattice's corner
        box("upright", x - .009, x + .009, -.012, .012, -.25, .25, old)
    box("bar_low", -.30, .30, -.012, .012, -.142, -.12, old)
    mbd = bpy.data.metaballs.new("drops")
    mbd.resolution, mbd.render_resolution, mbd.threshold = .25, .06, .6
    mbo = link(bpy.data.objects.new("drops", mbd))
    mbo.scale = (MM, MM, MM)
    mbd.materials.append(water())
    # outside: a 梧桐 in the rain, the dusk behind it
    leaf = surface("leaf", (.035, .055, .025), .25, 0, .6)
    r = random.Random(5)
    for i in range(16):
        o = leaf_mesh(f"leaf{i}", r.uniform(.11, .16), r.choice((3, 3, 5)))
        o.data.materials.append(leaf)
        o.location = (r.uniform(-.9, .9), r.uniform(.5, 1.6), r.uniform(-.5, .6)) if i < 9 else (r.uniform(-1.4, 1.4), r.uniform(1.6, 3.2), r.uniform(-.6, .9))     # some near enough to be dark soft shapes, not a wash
        o.rotation_euler = (math.radians(r.uniform(50, 110)), math.radians(r.uniform(-30, 30)), math.radians(r.uniform(-40, 40)))
    # the sky at dusk in rain: warm and light low down where the sun went, greying upwards; below the horizon the dark of the street. No lamp out there: anything bright in a drop would be a lamp's shape
    nt = sc.world.node_tree
    bg = nt.nodes["Background"]
    tc, sep, ramp = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeSeparateXYZ"), nt.nodes.new("ShaderNodeValToRGB")
    half = nt.nodes.new("ShaderNodeMath")
    half.operation, half.inputs[1].default_value, half.inputs[2].default_value = "MULTIPLY_ADD", .5, .5      # the direction's height, -1 to 1, as 0 to 1: the horizon at a half
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs[2], half.inputs[0])
    nt.links.new(half.outputs[0], ramp.inputs[0])
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = .47, (.035, .03, .028, 1)
    els[1].position, els[1].color = .52, (.62, .50, .40, 1)
    els.new(.62).color = (.44, .41, .39, 1)
    els.new(.85).color = (.30, .31, .32, 1)
    nt.links.new(ramp.outputs[0], bg.inputs[0])
    bg.inputs[1].default_value = .8
    lamp = bpy.data.lights.new("room", "POINT")
    lamp.shadow_soft_size, lamp.energy, lamp.color = .12, 6.0, (1.0, .74, .50)
    lo = link(bpy.data.objects.new("room", lamp))
    lo.location = (-.35, -.9, .25)
    # rain: thin streaks between the lattice and the tree
    rain = bpy.data.materials.new("rain")
    rain.use_nodes = True
    rb = rain.node_tree.nodes["Principled BSDF"]
    rb.inputs["Base Color"].default_value = (1, 1, 1, 1)
    rb.inputs["Transmission Weight"].default_value = 1.0
    rb.inputs["Roughness"].default_value = .05
    for i in range(160):
        x, y, z = r.uniform(-.5, .5), r.uniform(.12, 1.2), r.uniform(-.4, .4)
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=.0005, depth=r.uniform(.05, .11), location=(x, y, z))
        bpy.context.object.data.materials.append(rain)
    camera((.003, -.122, BAR_Z - .012), (.0, .0085, BAR_Z - .0074), 85.0, 5.6)      # the bar's foot a fifth of the way down, the drops large, the upright in at the right
    sc.render.motion_blur_shutter = .5
    return lambda t: drops_at(mbd, t)


# ---------------------------------------------------------------- 江城子: the lamp that has not gone out, and the book

BOOK_W, BOOK_H = .155, .235          # one half of the open book, a leaf folded at the fore-edge, as old books are
LAMP_AT = (.16, .07)                 # the lamp on the table, right of the book, its wick reaching back over it


COLS, ROWS = 7, 13                   # a half-leaf: seven columns of thirteen, as large as a lamp-lit page is printed
# The letter's opening, up to where film_docs.LETTER takes it up (少學琴書), as zh.wikisource.org/wiki/與子儼等疏 has it (陶淵明集; read 2026-11-08), stops left out.
# Only its last half-leaf is on the table: the right page before the leaf turns. (That page spells 閑 where film_docs.LETTER has 閒; both editions exist, the book keeps LETTER's.)
LETTER_HEAD = ("告儼俟份佚佟天地賦命生必有死自古賢聖誰能獨免子夏有言死生有命富貴在天四友之人親受音旨發斯談者將非窮達不可妄求壽夭永無外請故耶"
               "吾年過五十少而窮苦每以家弊東西遊走性剛才拙與物多忤自量為己必貽俗患僶俛辭世使汝等幼而饑寒余嘗感孺仲賢妻之言敗絮自擁何慚兒子此既一事矣但恨鄰靡二仲室無萊婦抱茲苦心良獨內愧")


def book_page(path, columns, seed, px=2200):
    """A half-leaf of a block-printed book: a single-line frame, column rules, the characters in Songti TC, the ink a little uneven from character to character, the paper a little uneven. columns: the text of each column, right to left."""
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    r = np.random.default_rng(seed)
    w, h = int(px * BOOK_W / BOOK_H), px
    im = Image.new("RGB", (w, h), (226, 212, 184))
    d = ImageDraw.Draw(im)
    ink = (38, 30, 24)
    m = int(w * .11), int(h * .09)
    box_ = (m[0], m[1], w - m[0], h - int(h * .07))
    d.rectangle(box_, outline=ink, width=int(w * .006))
    cw = (box_[2] - box_[0]) / COLS
    for i in range(1, COLS):
        x = box_[0] + i * cw
        d.line((x, box_[1], x, box_[3]), fill=(70, 58, 46), width=max(1, int(w * .0015)))
    f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Songti.ttc", int(cw * .70), index=7)
    for c, text in enumerate(columns):
        x = box_[2] - (c + .5) * cw
        for k, ch in enumerate(text):
            if ch == " ":
                continue
            y = box_[1] + (k + .55) * (box_[3] - box_[1]) / ROWS
            v = int(r.uniform(0, 26))
            d.text((x, y), ch, fill=(ink[0] + v, ink[1] + v, ink[2] + v), font=f, anchor="mm")
    a = np.asarray(im, np.float32)
    coarse = np.asarray(Image.fromarray(np.uint8(r.uniform(0, 255, (h // 60, w // 60)))).resize((w, h), Image.BICUBIC), np.float32)[..., None] / 255
    fine = r.normal(0, 1, (h, w, 1)).astype(np.float32)
    a = a * (.975 + .05 * coarse) + 3.0 * fine
    Image.fromarray(np.uint8(np.clip(a, 0, 255))).filter(ImageFilter.GaussianBlur(.7)).save(path)


def book_columns(text):
    """The passage of 與子儼等疏 the film has, column after column. It is from the middle of the letter, so it has no title over it: the book is open somewhere inside it."""
    cols = [text[i:i + ROWS] for i in range(0, len(text), ROWS)]
    return [cols[i:i + COLS] for i in range(0, len(cols), COLS)]


def leaf_surface(name, picture, back=None):
    """Paper with a page printed on it. With back, the leaf that turns: back is printed on its other face, the right way round when that face is up on the right
    stack (its picture mirrored across the leaf), and the lamp shining through the leaf comes through both pages' ink."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(picture)
    uv = nt.nodes.new("ShaderNodeUVMap")
    nt.links.new(uv.outputs[0], t.inputs[0])
    face, through = t.outputs[0], t.outputs[0]
    if back:
        flip = nt.nodes.new("ShaderNodeMapping")
        flip.inputs["Scale"].default_value = (-1, 1, 1)
        flip.inputs["Location"].default_value = (1, 0, 0)
        nt.links.new(uv.outputs[0], flip.inputs[0])
        tb = nt.nodes.new("ShaderNodeTexImage")
        tb.image = bpy.data.images.load(back)
        nt.links.new(flip.outputs[0], tb.inputs[0])
        side = nt.nodes.new("ShaderNodeMix")
        side.data_type = "RGBA"
        nt.links.new(nt.nodes.new("ShaderNodeNewGeometry").outputs["Backfacing"], side.inputs[0])
        nt.links.new(t.outputs[0], side.inputs[6])
        nt.links.new(tb.outputs[0], side.inputs[7])
        both = nt.nodes.new("ShaderNodeMix")
        both.data_type, both.blend_type = "RGBA", "MULTIPLY"
        both.inputs[0].default_value = 1.0
        nt.links.new(t.outputs[0], both.inputs[6])
        nt.links.new(tb.outputs[0], both.inputs[7])
        face, through = side.outputs[2], both.outputs[2]
    nt.links.new(face, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = .8
    b.inputs["Specular IOR Level"].default_value = .2
    b.inputs["Subsurface Weight"].default_value = .05
    tr = nt.nodes.new("ShaderNodeBsdfTranslucent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = .18                             # thin paper: the lamp shows through a lifted leaf
    nt.links.new(through, tr.inputs[0])
    nt.links.new(b.outputs[0], mix.inputs[1])
    nt.links.new(tr.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], nt.nodes["Material Output"].inputs[0])
    return m


def sheet_grid(name, w, h, bend, n=60, m=40):
    """A leaf from the spine (x = 0) outwards: bend(x) gives its height and how far it has come back towards the spine."""
    xs = np.linspace(0, 1, n)[None, :]
    ys = np.linspace(0, 1, m)[:, None]
    X, Z = bend(xs * w)
    return mesh_grid(name, np.broadcast_to(X, (m, n)), np.broadcast_to(ys * h, (m, n)), np.broadcast_to(Z, (m, n)))


# The leaf turns over (until 2026-11-08 it rose and fell back on its own stack; 奈亚子: 「这个翻页没有做好，没有翻过去」).
# Its angle about the spine, degrees: 0 lying on the left stack, 180 on the right. The air gets under the fore-edge first, the leaf goes up, hangs a moment
# just past standing, and falls over onto the right stack, slowing on the air under it.
TURN = ((0, 0), (.4, 0), (1.1, 10), (2.1, 88), (2.5, 100), (3.2, 168), (3.6, 180), (4.5, 180))
# How much more the fore-edge is turned than the spine, degrees (the bend grows along the leaf): first it leads, lifted by the air under it; then it trails,
# held back by the air, most as the leaf falls, so the fore-edge comes down last on a cushion of air.
CURL = ((0, 0), (.4, 0), (1.1, 35), (1.7, 15), (2.1, -8), (2.5, -5), (3.2, -35), (3.6, -38), (4.15, 0), (4.5, 0))
LIFT_UP, LIFT_DOWN = (.4, 2.1), (2.5, 4.1)     # the sound (film_mix): the air lifting the leaf, and the leaf falling over and settling
LIE = .0006                                    # how far the turning leaf lies over the stack it rests on
FLAME_AT = (.62, .34)                # where the flame stands in the frame, as parts of its width and height from the top left: where the screen of window C is, so the two cut on one place


def through(keys, t):
    """A value through keys [(t, v)] smoothly, never overshooting one (a monotone cubic, Fritsch and Carlson); level before the first and after the last."""
    x = np.array([k[0] for k in keys], float)
    y = np.array([k[1] for k in keys], float)
    if t <= x[0]:
        return float(y[0])
    if t >= x[-1]:
        return float(y[-1])
    h, d = np.diff(x), np.diff(y) / np.diff(x)
    m = np.zeros_like(y)
    for i in range(1, len(x) - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    i = int(np.searchsorted(x, t)) - 1
    s = (t - x[i]) / h[i]
    return float((2 * s ** 3 - 3 * s ** 2 + 1) * y[i] + (s ** 3 - 2 * s ** 2 + s) * h[i] * m[i] + (3 * s ** 2 - 2 * s ** 3) * y[i + 1] + (s ** 3 - s ** 2) * h[i] * m[i + 1])


def gust(t):
    """The air at the flame, 0 to 1: it comes as the leaf lifts, holds while it goes up, dies away as it falls; then a small breath as the leaf lands and
    pushes the air out from under it, across the right page towards the lamp."""
    return sm((t - .25) / .6) * (1 - sm((t - 2.2) / 1.0)) + .5 * math.exp(-((t - 3.65) / .18) ** 2)


def stack_top(q):
    """How high the top leaf of either stack lies, q of the way from the spine to the fore-edge."""
    q = np.clip(q, 0, 1)
    return .012 + .006 * np.sin(np.pi * q) ** .5 - .004 * q ** 2


def turning_leaf(t, n=60):
    """The turning leaf's section at t, spine to fore-edge: (x, z) of n points, x negative over the left stack. Bent along its length without stretching;
    it starts lying on the left stack and ends lying on the right, and never goes into either."""
    s = np.linspace(0, BOOK_W, n)
    q = s / BOOK_W
    th = math.radians(through(TURN, t)) + math.radians(through(CURL, t)) * q ** 1.6
    mid = (th[1:] + th[:-1]) / 2
    ds = BOOK_W / (n - 1)
    x = np.concatenate([[0.0], np.cumsum(-np.cos(mid) * ds)])
    z = stack_top(0) + LIE + np.concatenate([[0.0], np.cumsum(np.sin(mid) * ds)])
    w = sm((t - .4) / .6)                      # off the left stack's curve into the air
    x, z = (1 - w) * -s + w * x, (1 - w) * (stack_top(q) + LIE) + w * z
    w = sm((t - 3.7) / .5)                     # down into the right stack's curve
    x, z = (1 - w) * x + w * s, (1 - w) * z + w * (stack_top(q) + LIE)
    return x, np.maximum(z, stack_top(np.abs(x) / BOOK_W) + LIE)


def flame_material():
    """An oil flame: blue and faint at its root, a yellow-white core, orange towards the tip; its edge thins to nothing, so it has no outline."""
    m = bpy.data.materials.new("flame")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    n = nt.nodes.new
    o, e, tc, sep, ramp = n("ShaderNodeOutputMaterial"), n("ShaderNodeEmission"), n("ShaderNodeTexCoord"), n("ShaderNodeSeparateXYZ"), n("ShaderNodeValToRGB")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs[2], ramp.inputs[0])
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = .0, (.06, .10, .45, 1)
    els[1].position, els[1].color = .16, (1.0, .55, .18, 1)
    for at, c in ((.30, (1.0, .80, .45, 1)), (.60, (1.0, .62, .22, 1)), (1.0, (.85, .28, .05, 1))):
        els.new(at).color = c
    nt.links.new(ramp.outputs[0], e.inputs[0])
    e.inputs[1].default_value = 1.25
    # how much of the flame is there: none at its silhouette, less at the root and the tip
    lw, inv, pw, prof, mul = n("ShaderNodeLayerWeight"), n("ShaderNodeMath"), n("ShaderNodeMath"), n("ShaderNodeFloatCurve"), n("ShaderNodeMath")
    lw.inputs["Blend"].default_value = .5
    inv.operation, inv.inputs[0].default_value = "SUBTRACT", 1.0
    nt.links.new(lw.outputs["Facing"], inv.inputs[1])
    pw.operation, pw.inputs[1].default_value = "POWER", 1.6
    nt.links.new(inv.outputs[0], pw.inputs[0])
    c = prof.mapping.curves[0]
    c.points[0].location, c.points[1].location = (0, .25), (1, 0)
    c.points.new(.2, .9)
    c.points.new(.7, .8)
    prof.mapping.update()
    nt.links.new(sep.outputs[2], prof.inputs["Value"])
    mul.operation = "MULTIPLY"
    nt.links.new(pw.outputs[0], mul.inputs[0])
    nt.links.new(prof.outputs[0], mul.inputs[1])
    tr, mix, cam, gate = n("ShaderNodeBsdfTransparent"), n("ShaderNodeMixShader"), n("ShaderNodeLightPath"), n("ShaderNodeMath")
    # only the camera sees it; the light inside lights the room, smoothly
    gate.operation = "MULTIPLY"
    nt.links.new(mul.outputs[0], gate.inputs[0])
    nt.links.new(cam.outputs["Is Camera Ray"], gate.inputs[1])
    nt.links.new(gate.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(e.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], o.inputs[0])
    return m


def flame_at(o, light, t):
    """It stands almost still, and leans a little away from the book and shortens when the air comes that turns the leaf; it does not go out."""
    k = gust(t)
    flick = .04 * math.sin(23 * t) + .03 * math.sin(37 * t + 1.3)
    o.scale = (1, 1, 1 - .18 * k + flick)
    o.rotation_euler = (0, .14 * k + .02 * math.sin(11 * t), 0)
    light.energy = 2.6 * (1 - .12 * k + flick * .6)


def shot_lamp():
    D = docs3d()
    D.table()
    sc = bpy.context.scene
    # the book: two stacks of leaves, the top ones curved up from the spine, then the one the air turns, lying on the left stack
    left, right, lifted, back = (os.path.join(OUT, "tex", f"leaf_{n}.png") for n in ("left", "right", "lifted", "back"))
    paper = surface("block", (.80, .74, .61), .85, 0, .2)
    def left_side(o):
        """A leaf built out along -x: its faces turned up again and its picture the right way round."""
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(o.data)
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(o.data)
        bm.free()
        uv = o.data.uv_layers[0].data
        a = np.empty(len(uv) * 2, np.float32)
        uv.foreach_get("uv", a)
        a[0::2] = 1 - a[0::2]
        uv.foreach_set("uv", a)

    for side, pic in ((1, right), (-1, left)):
        def bend(x, side=side):
            return side * x, stack_top(x / BOOK_W)
        o = sheet_grid(f"leaf_{side}", BOOK_W, BOOK_H, bend)
        if side < 0:
            left_side(o)
        o.data.materials.append(leaf_surface(f"leaf_{side}", pic))
        box(f"block_{side}", min(0, side * BOOK_W), max(0, side * BOOK_W), 0, BOOK_H, 0, .011, paper)
    # the top leaf of the left stack, the one being read, is the one the air turns: its other face lands up on the right stack
    lift_leaf = sheet_grid("lifted", BOOK_W, BOOK_H, lambda x: (-x, stack_top(x / BOOK_W) + LIE), 60, 40)
    left_side(lift_leaf)
    lift_leaf.data.materials.append(leaf_surface("lifted", lifted, back))
    cover = surface("cover", (.17, .19, .24), .7, 0, .3)          # indigo paper covers, faded
    box("cover_r", 0, BOOK_W + .002, -.002, BOOK_H + .002, -.0015, 0, cover)
    box("cover_l", -BOOK_W - .002, 0, -.002, BOOK_H + .002, -.0015, 0, cover)
    root = bpy.data.objects.new("book", None)
    link(root)
    for o in list(bpy.data.objects):
        if o.name.startswith(("leaf_", "block_", "lifted", "cover_")):
            o.parent = root
    root.location = (-.06, -.13, 0)
    root.rotation_euler = (0, 0, math.radians(-8))
    # the lamp: a celadon dish on a stem, oil in it, a wick of twisted cotton over the lip, the flame at its end
    glaze = surface("black_glaze", (.030, .022, .016), .14, 0, .5)      # an old dark-glazed lamp: the flame runs along it in highlights
    lx, ly = LAMP_AT
    lathe("lamp", [(.0, .0), (.042, .0), (.043, .004), (.036, .010), (.012, .016), (.010, .070), (.013, .080), (.050, .086), (.056, .096), (.054, .098), (.046, .088), (.0, .086)], (lx, ly, 0), glaze)
    oil = surface("oil", (.003, .002, .001), .02, 0, .5)            # nearly black but for what it reflects: lit from 5 cm, anything paler reads as a saucer of milk
    lathe("oil", [(.0, .0925), (.051, .0925)], (lx, ly, 0), oil)
    wick = surface("wick", (.015, .012, .010), .95, 0, .1)
    end = Vector((lx - .055, ly - .006, .1005))
    pts = [Vector((lx - .018, ly - .002, .0935)), Vector((lx - .04, ly - .004, .095)), Vector((lx - .053, ly - .006, .0985)), end]
    c = bpy.data.curves.new("wick", "CURVE")
    c.dimensions, c.bevel_depth = "3D", .0016
    s = c.splines.new("POLY")
    s.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        s.points[i].co = (*p, 1)
    c.materials.append(wick)
    link(bpy.data.objects.new("wick", c))
    char = surface("char", (.02, .02, .02), .9)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=.0019, location=end + Vector((0, 0, .001)))
    bpy.context.object.data.materials.append(char)
    # the flame: a teardrop of light standing on the wick's end
    n, m_ = 24, 32
    zz = np.linspace(0, 1, n)[:, None]
    th = np.linspace(0, 2 * math.pi, m_)[None, :]
    rr = .0034 * np.sin(np.pi * zz ** .55) ** 1.2
    fl = mesh_grid("flame", rr * np.cos(th), rr * np.sin(th), np.broadcast_to(.031 * zz, (n, m_)))
    fl.data.materials.append(flame_material())
    fl.location = end + Vector((0, 0, .0015))
    light = bpy.data.lights.new("flame", "POINT")
    light.shadow_soft_size, light.color = .004, (1.0, .66, .32)
    lo = link(bpy.data.objects.new("flame_light", light))
    lo.parent = fl
    lo.location = (0, 0, .007)
    bg = sc.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (.010, .008, .006, 1)          # a dark room; what little else there is
    bg.inputs[1].default_value = 1.0
    cam = camera((.03, -.52, .30), (.0, -.02, .05), 50.0, 3.5)
    cam.data.dof.focus_distance = (fl.location - cam.location).length
    place(cam, fl.location + Vector((0, 0, .008)), FLAME_AT)

    def at_t(t):
        X, Z = turning_leaf(t)
        me = lift_leaf.data
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        co[:, 0] = np.tile(X, 40)
        co[:, 2] = np.tile(Z, 40)
        me.vertices.foreach_set("co", co.ravel())
        me.update()
        flame_at(fl, light, t)
    return at_t


def pages():
    """The four half-leaves, in plain Python. The book reads from the right. Before the turn: on the right the end of the letter's opening (良獨內愧),
    on the left film_docs.LETTER's first half-leaf (少學琴書), the top leaf of the left stack. The air turns that leaf: its other face, LETTER's second
    (念之在心，若何可言), comes down on the right, and the left now shows what lay under it, the third, where the letter ends (吾復何言)."""
    import film_docs
    folder = os.path.join(OUT, "tex")
    os.makedirs(folder, exist_ok=True)
    first, second, third = book_columns(film_docs.LETTER)
    head = book_columns(LETTER_HEAD[len(LETTER_HEAD) - COLS * ROWS:])[0]
    for n, cols, seed in (("right", head, 1), ("lifted", first, 2), ("back", second, 4), ("left", third, 3)):
        book_page(os.path.join(folder, f"leaf_{n}.png"), cols, seed)
    print(folder, flush=True)


# ---------------------------------------------------------------- render

def main():
    sc = reset(SAMPLES, PCT)
    sc.render.film_transparent = False
    sc.render.image_settings.color_mode = "RGB"
    sc.view_settings.view_transform = "AgX"
    sc.cycles.max_bounces = 12
    sc.cycles.transmission_bounces = 12
    sc.cycles.transparent_max_bounces = 16
    sc.cycles.caustics_refractive = False
    sc.cycles.sample_clamp_indirect = 6.0
    sc.cycles.adaptive_threshold = .008
    move = dict(petal=shot_petal, rain=shot_rain, lamp=shot_lamp)[SHOT]()
    os.makedirs(OUT, exist_ok=True)
    if FRAMES:
        a, b = (int(v) for v in FRAMES.split("-"))
        folder = os.path.join(OUT, "seq", SHOT)
        os.makedirs(folder, exist_ok=True)
        for i in range(a, b + 1):
            move(i / FPS)
            sc.render.filepath = os.path.join(folder, f"{i:04d}.png")
            bpy.ops.render.render(write_still=True)
            print("frame", i, flush=True)
        return
    t = T if T >= 0 else LENGTH[SHOT] * .8
    move(t)
    sc.render.filepath = os.path.join(OUT, f"{SHOT}_{t:.1f}" + ("" if PCT == 100 else f"_p{PCT}") + ".png")
    bpy.ops.render.render(write_still=True)
    print("wrote", sc.render.filepath, flush=True)


if bpy is None:
    pages()
else:
    main()
