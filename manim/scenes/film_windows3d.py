"""九 千窗 in Blender: glass panels with the real windows on their faces, in real fog, colour straight out. One set of windows for K1 to K3, the camera only going back along its axis, fixed on the one window.

    Blender -b --factory-startup -P film_windows3d.py -- KEY[,KEY...] PLACE [--pct 100] [--samples 256] [--suffix NAME]

KEY is K2 or K3, PLACE is A (warm and dark, the screens lighting the fog and each other). Output render/KEYPLACE.png, under ink/film_out/film/windows/. One Blender unit is a metre, and a window at scale 1 is a millimetre to its point.
"""
import json
import math
import os
import random
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ink3d import aim, reset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "windows")
TEX = os.path.join(OUT, "tex")
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
LENS = 35.0
PX = 1920 * LENS / 36.0            # frame pixels across a metre a metre away

# the camera's distance from the one window along its axis: the same log rate, about 1.42 times a second, from K1 at 210.5 s, where the window is the size film_ring has it
D1, D2, D3 = 1.037, 4.18, 23.9
TILT = 0.0
RADIUS, CHAMFER, THICK = .014, .0025, .012
AXIS = (0.0, math.cos(math.radians(TILT)), -math.sin(math.radians(TILT)))
UP = (0.0, math.sin(math.radians(TILT)), math.cos(math.radians(TILT)))
HERO_LIFT = .3                      # how much brighter the one window is than the rest by K3, so that you can find it again
# how many windows each band of depth from the camera of K3 has in its frame: nearer than these they are set out by hand
BANDS = ((12, 20, 10), (20, 32, 26), (32, 55, 70), (55, 100, 190), (100, 180, 420), (180, 340, 900))
CLEAR = (-130.0, -80.0, 130.0, 80.0)        # the gap around the one window in K3, in pixels from the middle
FOOT = (-540.0, -560.0, 540.0, -395.0)      # the subtitle's

SHOTS = dict(
    K1=dict(d=D1, focus=D1, fstop=8.0),
    K2=dict(d=D2, focus=2.2, fstop=4.0),
    K3=dict(d=D3, focus=D3, fstop=.45),
)
FOG_FROM, FOG_LEN = .55, 2.7        # the fog goes back with the camera: it starts this many camera-distances in front of it, behind what you are reading, and takes this many to lose all but a third of what is behind it
# the place. scatter: how much of the screens' light the fog turns towards you, per m. spill: how much more light a screen throws than you see on it
PLACES = dict(
    # warm: the air takes a little more blue than red, so that what is far goes warm, not grey
    A=dict(colour=(.0194, .0159, .0130), warm=(.26, .13, 0.0), screen=.95, diffuse=.07, ior=1.5, rim=(1.0, .30), back=(.22, .04),
           K1=dict(scatter=.02, spill=5.0), K2=dict(scatter=.02, spill=5.0), K3=dict(scatter=.01, spill=4.0)),
)


def camera_at(d):
    return tuple(-d * a for a in AXIS)


def leaving(d):
    """How far the camera has come from K2 to K3, eased at both ends: 0 at K2 and before it."""
    s = max(min(math.log(max(d, D2) / D2) / math.log(D3 / D2), 1.0), 0.0)
    return s * s * (3 - 2 * s)


def stand(name, sizes, width, cx, cy, d, d_cam, yaw, pitch, n):
    """A window width metres across, its centre at this pixel of the frame seen from the camera d_cam back from the one window, d metres from that camera along its axis."""
    cam = camera_at(d_cam)
    r, u = (cx - 960) / PX * d, (540 - cy) / PX * d
    p = (cam[0] + r, cam[1] + AXIS[1] * d + UP[1] * u, cam[2] + AXIS[2] * d + UP[2] * u)
    return dict(tex=name, p=p, yaw=yaw, pitch=TILT + pitch, s=width / (sizes[name]["w"] * .001), n=n)


def hand(sizes):
    """The windows set out by eye: the one window, then the three you can read in K2 and a few behind them, then the few close to the camera in K3, all dark ones, which frame the picture and don't pull at the eye.

    All that is in the frame of K2 closes up on the one window by K3, into a box 340 by 190 pixels, whatever its depth: going back alone would pile K2's windows onto it. So they each have a place of their own in K3, and they drift there, sideways and only by a metre or two, as the camera goes back."""
    w = [dict(tex="hero", p=(0.0, -.01 * UP[1], -.01 * UP[2]), yaw=0.0, pitch=TILT, s=1.0, n=8)]      # its middle is a little under the middle of the screen it is on
    for name, width, cx, cy, d, yaw, pitch, x3, y3 in (
        ("ask_b", .81, 370, 270, 2.2, 4.0, -.5, 765, 415),
        ("ask_a", .90, 1530, 270, 2.3, -4.0, -.5, 1165, 420),
        ("ask_c", .81, 1520, 770, 2.2, -3.0, .5, 1180, 665),
        ("k2_new", .80, 380, 770, 3.0, 3.5, .5, 740, 655),
        ("chat01", .9, 900, 165, 8.0, 2.0, -1.0, 945, 395),
        ("chat66", .9, 760, 800, 7.5, 3.0, .5, 850, 720),
        ("chat33", .9, 1060, 880, 10.0, -2.0, 1.0, 1045, 745),
        ("new1", .9, 1140, 330, 12.0, -5.0, 0.0, 1300, 535),
    ):
        win = stand(name, sizes, width, cx, cy, d, D2, yaw, pitch, 8 if d < 5 else 5)
        win["far"] = stand(name, sizes, width, x3, y3, d + D3 - D2, D3, yaw, pitch, 5)["p"]
        w.append(win)
    for name, width, cx, cy, d, yaw, pitch in (
        ("chat37", .85, 165, 830, 6.0, 7.0, 1.0),
        ("chat12", .9, 1745, 180, 6.8, -7.0, -2.0),
        ("chat45", .9, 330, 150, 9.5, 5.0, -3.0),
        ("chat20", .85, 1800, 770, 9.0, -6.0, 2.0),
        ("new3", .85, 70, 455, 11.0, 6.0, 0.0),
    ):
        w.append(stand(name, sizes, width, cx, cy, d, D3, yaw, pitch, 6))
    return w


def place_at(win, d):
    """Where a window is with the camera d back."""
    if "far" not in win:
        return win["p"]
    s = leaving(d)
    return tuple(a + (b - a) * s for a, b in zip(win["p"], win["far"]))


def in_frame(p, d_cam, margin=0.0):
    """Whether a point is in the frame of the camera d_cam back, and how far along its axis."""
    cam = camera_at(d_cam)
    v = (p[0] - cam[0], p[1] - cam[1], p[2] - cam[2])
    depth = v[1] * AXIS[1] + v[2] * AXIS[2]
    up = v[1] * UP[1] + v[2] * UP[2]
    return depth > 0 and abs(v[0]) < .5143 * depth + margin and abs(up) < .2893 * depth + margin, depth


def seen_rect(p, width, height, d_cam):
    """What a window takes up in the frame of the camera d_cam back: x0, y0, x1, y1 in pixels from the middle."""
    cam = camera_at(d_cam)
    v = (p[0] - cam[0], p[1] - cam[1], p[2] - cam[2])
    depth = v[1] * AXIS[1] + v[2] * AXIS[2]
    up = v[1] * UP[1] + v[2] * UP[2]
    k = PX / depth
    return (v[0] - width / 2) * k, (up - height / 2) * k, (v[0] + width / 2) * k, (up + height / 2) * k


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def crowded(r, taken, share=.015, gap=.12):
    """Whether a window would lie over one already there, or stop so close to it that the two seem to touch."""
    area = (r[2] - r[0]) * (r[3] - r[1])
    g = gap * (r[3] - r[1])
    for t in taken:
        w, h = min(r[2] + g, t[2]) - max(r[0] - g, t[0]), min(r[3] + g, t[3]) - max(r[1] - g, t[1])
        if w > 0 and h > 0 and w * h > share * min(area, (t[2] - t[0]) * (t[3] - t[1])):
            return True
    return False


def field(pool, sizes, placed):
    """All the others, in a volume the camera of K3 stands inside: above, below and to each side of it, thinning out with distance until all that is left of a window is a little of its screen's warm white through the fog.

    Nothing close around the one window, nothing on the way the camera goes, nothing in front of the windows of K2. Windows of about the same depth don't lie over each other, as nobody piles them up; one far behind another may, as that is how you see the depth. Near the camera they are all dark ones."""
    rng = random.Random(4019)
    dark = [n for n in pool if sizes[n]["kind"] == "dark"]
    decks = {}

    def top(names):
        deck = decks.setdefault(names[0], [])
        if not deck:
            deck.extend(names)
            rng.shuffle(deck)
        return deck

    out = []
    taken = []
    for w in placed:
        size = sizes[w["tex"]]
        p = w.get("far", w["p"])
        x0, y0, x1, y1 = seen_rect(p, size["w"] * .001 * w["s"], size["h"] * .001 * w["s"], D3)
        taken.append((x0 - 14, y0 - 14, x1 + 14, y1 + 14, in_frame(p, D3)[1]))      # these turn, and a turned window is wider than its plain rectangle
    for z0, z1, count in BANDS:
        made = tries = 0
        while made < count and tries < count * 60:
            tries += 1
            z = z0 * (z1 / z0) ** rng.random()
            deck = top(dark if z < 20 else pool)
            name = deck[-1]
            width = rng.uniform(.75, 1.05)
            height = width * sizes[name]["h"] / sizes[name]["w"]
            hw, hh = width / 2 * PX / z, height / 2 * PX / z
            if z < 55:
                cx, cy = rng.uniform(-hw, 1920 + hw), rng.uniform(-hh, 1080 + hh)
            else:       # the volume is long, not wide: past the middle distance it draws in towards the far end of the axis, behind the one window
                spread = (55 / z) ** .55
                cx, cy = rng.gauss(960, 640 * spread), rng.gauss(540, 380 * spread)
            r = (cx - 960 - hw, 540 - cy - hh, cx - 960 + hw, 540 - cy + hh)
            if z < 120 and overlaps(r, CLEAR) or z < 100 and overlaps(r, FOOT):
                continue
            x, y = (cx - 960) / PX * z, (540 - cy) / PX * z
            if z < D3 and abs(x) - width / 2 < .75 and abs(y) - height / 2 < .45:
                continue
            win = stand(name, sizes, width, cx, cy, z, D3, rng.uniform(-9, 9), rng.uniform(-5, 5), 6 if z < 20 else 4 if z < 60 else 2)
            seen, depth = in_frame(win["p"], D2, .9)
            if seen and depth < 16.0:
                continue
            if z < 55:
                inside = (min(r[2], 960) - max(r[0], -960)) * (min(r[3], 540) - max(r[1], -540))
                if inside < .35 * (r[2] - r[0]) * (r[3] - r[1]):        # no slivers along the edges of the frame
                    continue
            like = [t for t in taken if .5 < t[4] / z < 2]
            if crowded(r, like) if z < 55 else crowded(r, like, .25, 0.0):
                continue
            taken.append(r + (z,))
            deck.pop()
            out.append(win)
            made += 1
    return out


def outline(w, h, r, n):
    """A rounded rectangle, anticlockwise from its right edge as you face it."""
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


class Build:
    """Vertices, faces and UVs gathered for one object."""

    def __init__(self):
        self.v, self.f, self.uv = [], [], []

    def add(self, pts, loops=None, uv=None):
        base = len(self.v)
        self.v.extend(pts)
        for i, loop in enumerate(loops):
            self.f.append([base + j for j in loop])
            if uv:
                self.uv.extend(uv[j] for j in loop)

    def object(self, name, material):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        if self.uv:
            layer = me.uv_layers.new(name="UV")
            flat = [c for p in self.uv for c in p]
            layer.data.foreach_set("uv", flat)
        me.materials.append(material)
        me.update()
        o = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(o)
        return o


def frame_of(win, size):
    """The window's axes in the world: its right, its up, and the way it faces."""
    yaw, pitch = math.radians(win["yaw"]), math.radians(win["pitch"])
    u = (math.cos(yaw), math.sin(yaw), 0.0)
    n0 = (math.sin(yaw), -math.cos(yaw), 0.0)
    v = (-n0[0] * math.sin(pitch), -n0[1] * math.sin(pitch), math.cos(pitch))
    n = (n0[0] * math.cos(pitch), n0[1] * math.cos(pitch), math.sin(pitch))
    p = win["p"]
    return lambda a, b, c: (p[0] + u[0] * a + v[0] * b + n[0] * c, p[1] + u[1] * a + v[1] * b + n[1] * c, p[2] + u[2] * a + v[2] * b + n[2] * c)


def panel(win, size, screens, rims, backs):
    """One window: its screen under the glass, a thin chamfer that catches the light, a side and a back."""
    w, h = size["w"] * .001 * win["s"], size["h"] * .001 * win["s"]
    c, t, r, n = CHAMFER * win["s"], THICK * win["s"], RADIUS * win["s"], win["n"]
    at = frame_of(win, size)
    face = outline(w - 2 * c, h - 2 * c, max(r - c, .001), n)
    edge = outline(w, h, r, n)
    m = len(face)
    screens[win["tex"]].add([at(a, b, 0.0) for a, b in face], [list(range(m))], [(a / w + .5, b / h + .5) for a, b in face])
    theme = size["kind"]
    ring = [at(a, b, 0.0) for a, b in face] + [at(a, b, -c) for a, b in edge] + [at(a, b, -t) for a, b in edge]
    loops = [[i, (i + 1) % m, m + (i + 1) % m, m + i][::-1] for i in range(m)] + [[m + i, m + (i + 1) % m, 2 * m + (i + 1) % m, 2 * m + i][::-1] for i in range(m)]
    rims[theme].add(ring, loops)
    backs[theme].add([at(a, b, -t) for a, b in edge], [list(range(m))[::-1]])


def nodes(mat):
    mat.use_nodes = True
    t = mat.node_tree
    t.nodes.clear()
    return t.nodes, t.links, t.nodes.new("ShaderNodeOutputMaterial")


def screen_material(name, place, spill, gain=1.0):
    """The picture of the window as it is, through a thin coat of glass. It throws more light at the fog and at other windows than you see on it, and it takes a little from them."""
    mat = bpy.data.materials.new("screen_" + name)
    n, l, out = nodes(mat)
    tex = n.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.join(TEX, name + ".png"))
    tex.image.colorspace_settings.name = "sRGB"
    tex.interpolation = "Cubic" if name in ("hero", "ask_a", "ask_b", "ask_c", "k2_new") else "Linear"
    tex.extension = "EXTEND"
    uv = n.new("ShaderNodeUVMap")
    uv.uv_map = "UV"
    l.new(uv.outputs[0], tex.inputs[0])
    path = n.new("ShaderNodeLightPath")
    seen = n.new("ShaderNodeMath")
    seen.operation = "MAXIMUM"
    l.new(path.outputs["Is Camera Ray"], seen.inputs[0])
    l.new(path.outputs["Is Glossy Ray"], seen.inputs[1])
    strength = n.new("ShaderNodeMapRange")
    strength.inputs["From Min"].default_value, strength.inputs["From Max"].default_value = 0.0, 1.0
    strength.inputs["To Min"].default_value = place["screen"] * spill * gain
    strength.inputs["To Max"].default_value = place["screen"] * gain
    l.new(seen.outputs[0], strength.inputs["Value"])
    light = n.new("ShaderNodeEmission")
    l.new(tex.outputs[0], light.inputs[0])
    l.new(strength.outputs[0], light.inputs[1])
    matte = n.new("ShaderNodeBsdfDiffuse")
    matte.inputs[0].default_value = (place["diffuse"],) * 3 + (1,)
    add = n.new("ShaderNodeAddShader")
    l.new(light.outputs[0], add.inputs[0])
    l.new(matte.outputs[0], add.inputs[1])
    coat = n.new("ShaderNodeBsdfGlossy") if "ShaderNodeBsdfGlossy" in dir(bpy.types) else n.new("ShaderNodeBsdfAnisotropic")
    coat.inputs["Roughness"].default_value = .04
    fres = n.new("ShaderNodeFresnel")
    fres.inputs[0].default_value = place["ior"]
    mix = n.new("ShaderNodeMixShader")
    l.new(fres.outputs[0], mix.inputs[0])
    l.new(add.outputs[0], mix.inputs[1])
    l.new(coat.outputs[0], mix.inputs[2])
    l.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def rim_material(name, level, spill=1.0, tint=(1.0, .97, .92)):
    """The glass at the edge of a panel: the screen's light comes out of it, and it reflects."""
    mat = bpy.data.materials.new(name)
    n, l, out = nodes(mat)
    glow = n.new("ShaderNodeEmission")
    glow.inputs[0].default_value = tint + (1,)
    path = n.new("ShaderNodeLightPath")
    seen = n.new("ShaderNodeMath")
    seen.operation = "MAXIMUM"
    l.new(path.outputs["Is Camera Ray"], seen.inputs[0])
    l.new(path.outputs["Is Glossy Ray"], seen.inputs[1])
    strength = n.new("ShaderNodeMapRange")
    strength.inputs["From Min"].default_value, strength.inputs["From Max"].default_value = 0.0, 1.0
    strength.inputs["To Min"].default_value, strength.inputs["To Max"].default_value = level * spill, level
    l.new(seen.outputs[0], strength.inputs["Value"])
    l.new(strength.outputs[0], glow.inputs[1])
    gloss = n.new("ShaderNodeBsdfGlossy") if "ShaderNodeBsdfGlossy" in dir(bpy.types) else n.new("ShaderNodeBsdfAnisotropic")
    gloss.inputs["Color"].default_value = (.16, .16, .16, 1)        # glass, not a mirror: a mirror of a thousand small bright panels is a line of dots
    gloss.inputs["Roughness"].default_value = .12
    add = n.new("ShaderNodeAddShader")
    l.new(glow.outputs[0], add.inputs[0])
    l.new(gloss.outputs[0], add.inputs[1])
    l.new(add.outputs[0], out.inputs["Surface"])
    return mat


def fog_box(colour, warm, shot, d, scatter=True):
    """The air: it hides what is far with its own colour, with no noise, and turns some of the screens' light towards you."""
    mat = bpy.data.materials.new("fog")
    n, l, out = nodes(mat)
    sigma = 1.0 / (FOG_LEN * d)
    absorb = n.new("ShaderNodeVolumeAbsorption")
    absorb.inputs["Color"].default_value = warm + (1,)
    absorb.inputs["Density"].default_value = sigma if scatter else sigma + shot["scatter"]
    own = n.new("ShaderNodeEmission")
    own.inputs[0].default_value = tuple(c * (1 - a) for c, a in zip(colour, warm)) + (1,)
    own.inputs[1].default_value = sigma
    add = n.new("ShaderNodeAddShader")
    l.new(absorb.outputs[0], add.inputs[0])
    l.new(own.outputs[0], add.inputs[1])
    last = add
    if scatter and shot["scatter"]:
        turned = n.new("ShaderNodeVolumeScatter")
        turned.inputs["Color"].default_value = (1.0, .86, .70, 1)
        turned.inputs["Density"].default_value = shot["scatter"]
        turned.inputs["Anisotropy"].default_value = .35
        more = n.new("ShaderNodeAddShader")
        l.new(add.outputs[0], more.inputs[0])
        l.new(turned.outputs[0], more.inputs[1])
        last = more
    l.new(last.outputs[0], out.inputs["Volume"])
    me = bpy.data.meshes.new("fog")
    x0, x1, y0, y1, z0, z1 = -300.0, 300.0, camera_at(d)[1] + FOG_FROM * d * AXIS[1], 420.0, -180.0, 180.0
    v = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    me.from_pydata(v, [], [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    me.materials.append(mat)
    me.update()
    o = bpy.data.objects.new("fog", me)
    bpy.context.collection.objects.link(o)
    o.visible_shadow = True
    return o


def build(place_name, key, pct, samples, scatter):
    place, shot = PLACES[place_name], SHOTS[key]
    air = place[key]
    sc = reset(samples=samples, pct=pct)
    sc.render.film_transparent = False
    sc.render.image_settings.color_mode = "RGB"
    sc.cycles.max_bounces = 4
    sc.cycles.diffuse_bounces = 2
    sc.cycles.glossy_bounces = 2
    sc.cycles.transparent_max_bounces = 24
    sc.cycles.volume_bounces = 0
    sc.cycles.sample_clamp_indirect = 4.0
    sc.cycles.adaptive_threshold = .006
    sc.cycles.adaptive_min_samples = min(64, samples)
    sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = place["colour"] + (1,)
    sc.world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    with open(os.path.join(TEX, "manifest.json")) as f:
        info = json.load(f)
    sizes = info["windows"]
    near = hand(sizes)
    wins = near + field(info["pool"], sizes, near)
    screens = {w["tex"]: Build() for w in wins}
    rims = dict(light=Build(), dark=Build())
    backs = dict(light=Build(), dark=Build())
    for w in wins:
        w["p"] = place_at(w, shot["d"])
        panel(w, sizes[w["tex"]], screens, rims, backs)
    print(len(wins), "windows,", len(screens), "pictures", flush=True)
    for name, b in screens.items():
        b.object("screen_" + name, screen_material(name, place, air["spill"], 1 + HERO_LIFT * leaving(shot["d"]) if name == "hero" else 1.0))
    for i, theme in enumerate(("light", "dark")):
        rims[theme].object("rim_" + theme, rim_material("rim_" + theme, place["rim"][i], air["spill"]))
        backs[theme].object("back_" + theme, rim_material("back_" + theme, place["back"][i], air["spill"]))
    fog_box(place["colour"], place["warm"], air, shot["d"], scatter)
    c = bpy.data.cameras.new("cam")
    c.lens, c.clip_start, c.clip_end = LENS, .02, 800.0
    c.dof.use_dof = True
    c.dof.focus_distance = shot["focus"]
    c.dof.aperture_fstop = shot["fstop"]
    o = bpy.data.objects.new("cam", c)
    bpy.context.collection.objects.link(o)
    sc.camera = o
    aim(o, camera_at(shot["d"]), AXIS)
    return sc


def main():
    """Two renders. The first has the fog hide what is behind it but turn no light towards you, and is not denoised, so that every word stays as it is. The second, smaller, has the fog lit by the screens, and only the light it adds is kept: the denoiser eats words, and that light has nothing fine in it."""
    keys, place = ARGS[0].split(","), ARGS[1]
    pct = int(ARGS[ARGS.index("--pct") + 1]) if "--pct" in ARGS else 100
    samples = int(ARGS[ARGS.index("--samples") + 1]) if "--samples" in ARGS else 256
    suffix = ARGS[ARGS.index("--suffix") + 1] if "--suffix" in ARGS else ""
    os.makedirs(os.path.join(OUT, "render"), exist_ok=True)
    for key in keys:
        base = os.path.join(OUT, "render", f"{key}{place}{suffix}")
        t1 = time.time()
        sc = build(place, key, pct, samples, False)
        sc.cycles.use_denoising = False
        sc.render.filepath = base + "_n.png"
        bpy.ops.render.render(write_still=True)
        if PLACES[place][key]["scatter"]:
            sc = build(place, key, pct // 2, samples, True)
            sc.render.filepath = base + "_d.png"
            bpy.ops.render.render(write_still=True)
        print(key, place, "render seconds", round(time.time() - t1, 1), flush=True)


if __name__ == "__main__":
    main()
