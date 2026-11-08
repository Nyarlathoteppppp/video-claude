"""What the film's Blender scenes share: the render setup, an ink-painted terrain, and the depth pass that ink_grade.py turns into mist.
Imported inside Blender:

    Blender -b --factory-startup -P manim/scenes/ink3d_SCENE.py -- [args]

The scene renders plain grey light; the ink, the paper and the mist all come afterwards from ink_grade.py, so every scene ends up in the
same world. Each frame is NAME.png (RGBA, 16 bit, sky left transparent) and NAME_z.png (red: distance from the camera / FAR, green: height / ZMAX).
"""
import json
import math
import os

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT3D = os.path.join(ROOT, "ink", "film_out", "3d")
W, H, FPS = 1920, 1080, 30
FAR, ZMAX = 400.0, 100.0


def reset(samples=64, pct=100):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    sc.render.engine = "CYCLES"
    sc.cycles.device = "GPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.resolution_percentage = pct
    sc.render.fps = FPS
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "16"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    w = bpy.data.worlds.new("sky")
    w.use_nodes = True
    sc.world = w
    return sc


def world_light(strength=.35):
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)
    bg.inputs[1].default_value = strength


def sun(direction, energy=3.0, angle=4.0):
    """A sun from direction (x, y, z). Mind the horizontal part: it ends up mirrored, shining along (-x, -y, z). Scenes are lit by eye against this."""
    d = bpy.data.lights.new("sun", "SUN")
    d.energy, d.angle = energy, math.radians(angle)
    o = bpy.data.objects.new("sun", d)
    bpy.context.collection.objects.link(o)
    v = np.array(direction, float)
    v /= np.linalg.norm(v)
    o.rotation_euler = (math.acos(-v[2]), 0, math.atan2(-v[0], v[1]) + math.pi)
    return o


def camera(loc, target, lens=35.0, far=FAR, roll=0.0):
    c = bpy.data.cameras.new("cam")
    c.lens, c.clip_start, c.clip_end = lens, .01, far * 2
    o = bpy.data.objects.new("cam", c)
    bpy.context.collection.objects.link(o)
    bpy.context.scene.camera = o
    aim(o, loc, target, roll)
    return o


def aim(o, loc, target, roll=0.0):
    from mathutils import Vector
    o.location = loc
    d = Vector(target) - Vector(loc)
    o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    o.rotation_euler.rotate_axis("Z", roll)


def mesh_grid(name, x, y, z, uv=True):
    """A mesh from three n×m arrays, fast."""
    n, m = z.shape
    me = bpy.data.meshes.new(name)
    me.vertices.add(n * m)
    me.vertices.foreach_set("co", np.stack([x, y, z], -1).ravel().astype(np.float32))
    a = np.arange(n * m).reshape(n, m)
    quads = np.stack([a[:-1, :-1], a[:-1, 1:], a[1:, 1:], a[1:, :-1]], -1).ravel()
    me.loops.add(len(quads))
    me.loops.foreach_set("vertex_index", quads.astype(np.int32))
    np_ = (n - 1) * (m - 1)
    me.polygons.add(np_)
    me.polygons.foreach_set("loop_start", np.arange(np_, dtype=np.int32) * 4)
    me.polygons.foreach_set("use_smooth", np.ones(np_, bool))
    if uv:
        u = me.uv_layers.new(name="UV")
        uu = np.stack([np.broadcast_to(np.linspace(0, 1, m), (n, m)), np.broadcast_to(np.linspace(0, 1, n)[:, None], (n, m))], -1)
        u.data.foreach_set("uv", uu.reshape(-1, 2)[quads].ravel().astype(np.float32))
    me.update(calc_edges=True)
    me.validate()
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    return o


def spectral(n, beta, seed, m=None):
    """Tileable noise with a 1/f^beta spectrum, mean 0, std 1."""
    m = m or n
    r = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(m)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1
    s = (r.normal(size=f.shape) + 1j * r.normal(size=f.shape)) / f ** (beta / 2)
    s[0, 0] = 0
    v = np.fft.irfft2(s, s=(n, m))
    return ((v - v.mean()) / v.std()).astype(np.float32)


def lerp(tile, x, y):
    h, w = tile.shape
    ix, iy = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    fx, fy = x - ix, y - iy
    a, b, c, d = ix % w, (ix + 1) % w, iy % h, (iy + 1) % h
    return (tile[c, a] * (1 - fx) + tile[c, b] * fx) * (1 - fy) + (tile[d, a] * (1 - fx) + tile[d, b] * fx) * fy


def blur(a, s):
    """Gaussian blur through the FFT (Blender's Python has no scipy)."""
    n, m = a.shape
    fy, fx = np.fft.fftfreq(n)[:, None], np.fft.rfftfreq(m)[None, :]
    return np.fft.irfft2(np.fft.rfft2(a) * np.exp(-2 * (math.pi * s) ** 2 * (fx * fx + fy * fy)), s=(n, m)).astype(np.float32)


def sm(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ridged(n, seed, base=4.0, octaves=7, warp=.05):
    """Sharp mountain ridges: folded noise, each octave weighted by the one above, so detail gathers on the crests."""
    r = np.random.default_rng(seed)
    tile = spectral(1024, 1.0, seed)
    tile = blur(tile, 2.2)
    tile /= tile.std()
    y, x = np.mgrid[:n, :n].astype(np.float32) / n
    wx, wy = lerp(spectral(512, 3.2, seed + 1), x * 512, y * 512) * warp, lerp(spectral(512, 3.2, seed + 2), x * 512, y * 512) * warp
    x, y = x + wx, y + wy
    h, w, f, a = np.zeros((n, n), np.float32), np.ones((n, n), np.float32), base, 1.0
    for _ in range(octaves):
        ox, oy = r.uniform(0, 1024, 2)
        v = 1 - np.abs(np.tanh(lerp(tile, x * f * 18 + ox, y * f * 18 + oy) * .9))
        v = v * v
        h += a * v * w
        w = np.clip(v * 1.8, 0, 1)
        f *= 2.03
        a *= .52
    return h / h.max()


def massif(X, Y, peaks, seed):
    """Mountains placed by hand, as a painter composes them. peaks: (x, y, height, radius). Each has a warped outline and steep sides."""
    n = X.shape[0]
    y, x = np.mgrid[:n, :n].astype(np.float32) * (1024 / n)
    wx, wy = lerp(spectral(1024, 3.0, seed), x, y), lerp(spectral(1024, 3.0, seed + 1), x, y)
    span = float(X.max() - X.min())
    px, py = X + wx * span * .035, Y + wy * span * .035
    h = np.zeros_like(X)
    for i, (cx, cy, top, rad) in enumerate(peaks):
        a = np.arctan2(py - cy, px - cx)
        lobes = 1 + .24 * np.sin(a * 3 + i * 1.7) + .13 * np.sin(a * 5 + i * 2.9) + .07 * np.sin(a * 9 + i)
        d = np.sqrt((px - cx) ** 2 + (py - cy) ** 2) / (rad * lobes)
        h = np.maximum(h, top * np.exp(-d ** 1.55))
    return h


def erode(h, cell, seed, wavelength=26.0, octaves=7, gain=.52, strength=.34):
    """Gullies that follow the slope and branch as they go: at each octave, stripes running downhill are cut into the surface, the
    direction taken from the slope so far, so finer gullies run into coarser ones. Stripes from four nearby cells are blended so they never break."""
    n = h.shape[0]
    r = np.random.default_rng(seed)
    y, x = np.mgrid[:n, :n].astype(np.float32) * cell
    amp = strength * wavelength
    wl = wavelength
    for _ in range(octaves):
        gy, gx = np.gradient(blur(h, max(wl / cell / 6, .7)), cell)
        m = np.sqrt(gx * gx + gy * gy)
        ux, uy = gx / (m + 1e-5), gy / (m + 1e-5)
        cell_size = wl * 1.6
        fx, fy = x / cell_size, y / cell_size
        ix, iy = np.floor(fx).astype(np.int32), np.floor(fy).astype(np.int32)
        tx, ty = sm(fx - ix), sm(fy - iy)
        rnd = r.uniform(0, 2 * math.pi, (257, 257)).astype(np.float32)
        out = np.zeros_like(h)
        for dx, dy, w in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)), (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
            cxx, cyy = (ix + dx) * cell_size, (iy + dy) * cell_size
            perp = (x - cxx) * -uy + (y - cyy) * ux
            ph = perp / wl * 2 * math.pi + rnd[(iy + dy) % 257, (ix + dx) % 257]
            out += w * (1 - np.abs(np.sin(ph * .5))) * 2 - w * .72        # sharp ridges, broad gullies
        weight = sm(m / .35) * (np.maximum(h, 0) / (h.max() + 1e-6)) ** .35
        h = h + amp * out * weight
        amp *= gain
        wl *= gain
    return h


def strokes_along_slope(h, seed, steps=14, length=1.0):
    """Line integral convolution of noise down the slope: brush strokes that run the way water would, as in 披麻皴."""
    n = h.shape[0]
    noise_ = spectral(n, .3, seed)
    g = blur(h, 1.2)
    gy, gx = np.gradient(g)
    m = np.sqrt(gx * gx + gy * gy) + 1e-6
    ux, uy = gx / m, gy / m
    y, x = np.mgrid[:n, :n].astype(np.float32)
    out = noise_.copy()
    px, py, qx, qy = x.copy(), y.copy(), x.copy(), y.copy()
    for _ in range(steps):
        px, py = px + lerp(ux, px, py) * length, py + lerp(uy, px, py) * length
        qx, qy = qx - lerp(ux, qx, qy) * length, qy - lerp(uy, qx, qy) * length
        out += lerp(noise_, px, py) + lerp(noise_, qx, qy)
    return out / out.std()


def ink_albedo(h, seed, cell):
    """The rock's paint, from its shape: dark in the gullies and shadowed folds, brush strokes down the slopes, moss dots along the crests."""
    n = h.shape[0]
    fine = h - blur(h, 3)
    mid = h - blur(h, 14)
    wide = h - blur(h, 50)
    norm = lambda a: a / (a.std() + 1e-6)
    cav = np.clip(-norm(fine) * .5 - norm(mid) * .7 - norm(wide) * .3, 0, 3)
    gy, gx = np.gradient(h / cell)
    slope = np.sqrt(gx * gx + gy * gy)
    st = strokes_along_slope(h, seed)
    a = .78 - .17 * cav - .12 * np.clip(st, -1.6, 1.6) * (.45 + .55 * sm(slope / 1.5)) - .1 * sm(slope / 2.2)
    # moss dots (点苔) on the crests
    r = np.random.default_rng(seed + 9)
    crest = np.clip(norm(mid), 0, 3) / 3
    dots = np.zeros((n, n), np.float32)
    k = int(n * n * .0009)
    yy, xx = r.integers(0, n, k), r.integers(0, n, k)
    keep = r.random(k) < crest[yy, xx] * 1.4
    dots[yy[keep], xx[keep]] = 1
    dots = np.clip(blur(dots, .9) * 9, 0, 1)
    return np.clip(a - .5 * dots, .05, 1).astype(np.float32)


def image_from(a, name):
    """A float array in 0-1 as a Blender image (data, no colour conversion)."""
    h, w = a.shape
    im = bpy.data.images.new(name, w, h, float_buffer=True, alpha=False)
    im.colorspace_settings.name = "Non-Color"
    px = np.ones((h, w, 4), np.float32)
    px[..., 0] = px[..., 1] = px[..., 2] = a
    im.pixels.foreach_set(px.ravel())
    im.pack()
    return im


def painted_material(name, albedo, rough=.92):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = image_from(albedo, name + "_paint")
    t.interpolation = "Cubic"
    nt.links.new(t.outputs[0], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = .2
    return m


def grey_material(name, value=.6, rough=.9):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (value, value, value, 1)
    b.inputs["Roughness"].default_value = rough
    return m


def water_material(name="water", tint=.86, rough=.05, normal_scale=120.0, normal_strength=.15):
    """Still water as a mirror with fine ripples: what shows on it is what it reflects, the sky's bare paper and the mountains' ink."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (tint, tint, tint, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = rough
    tc, n1, bump = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeBump")
    n1.inputs["Scale"].default_value = normal_scale
    n1.inputs["Detail"].default_value = 6
    bump.inputs["Strength"].default_value = normal_strength
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    nt.links.new(n1.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], b.inputs["Normal"])
    return m


def depth_material():
    m = bpy.data.materials.new("depth")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    c, d, e, o = nt.nodes.new("ShaderNodeCameraData"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeOutputMaterial")
    g, sep, hz, comb = nt.nodes.new("ShaderNodeNewGeometry"), nt.nodes.new("ShaderNodeSeparateXYZ"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeCombineXYZ")
    d.operation = hz.operation = "DIVIDE"
    d.inputs[1].default_value = FAR
    hz.inputs[1].default_value = ZMAX
    nt.links.new(c.outputs["View Distance"], d.inputs[0])
    nt.links.new(g.outputs["Position"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], hz.inputs[0])
    nt.links.new(d.outputs[0], comb.inputs["X"])
    nt.links.new(hz.outputs[0], comb.inputs["Y"])
    nt.links.new(comb.outputs[0], e.inputs["Color"])
    nt.links.new(e.outputs[0], o.inputs["Surface"])
    return m


def render(name, depth=True):
    """Renders the frame and, on top, the distance from the camera. Skips what is already on disk."""
    sc = bpy.context.scene
    os.makedirs(os.path.dirname(name), exist_ok=True)
    bpy.context.view_layer.update()         # or the camera's matrix still reads as at the origin
    with open(name + ".json", "w") as f:
        json.dump(dict(cam_z=sc.camera.matrix_world.translation.z, far=FAR, zmax=ZMAX), f)
    if not os.path.exists(name + ".png"):
        sc.render.filepath = name + ".png"
        bpy.ops.render.render(write_still=True)
    if depth and not os.path.exists(name + "_z.png"):
        vl = bpy.context.view_layer
        keep = (sc.cycles.samples, sc.cycles.use_denoising, sc.view_settings.view_transform, sc.render.filter_size if hasattr(sc.render, "filter_size") else None)
        sc.cycles.samples, sc.cycles.use_denoising, sc.view_settings.view_transform = 2, False, "Raw"
        vl.material_override = depth_material()
        sc.render.filepath = name + "_z.png"
        bpy.ops.render.render(write_still=True)
        vl.material_override = None
        sc.cycles.samples, sc.cycles.use_denoising, sc.view_settings.view_transform = keep[0], keep[1], keep[2]
