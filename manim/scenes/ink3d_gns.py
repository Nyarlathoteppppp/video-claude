"""Blender helpers shared by chapters 11-13 of the ink film: meshes with a colour on every vertex, water you can see through, stone slabs, and a grid
laid on the water plane so that every vertex lands on one pixel of the frame."""
import math
import os

import bpy
import numpy as np

from ink3d import OUT3D, W, H, blur, lerp, sm, spectral

TEX = os.path.join(OUT3D, "_gns")


def tex(name):
    return np.load(os.path.join(TEX, name + ".npy")).astype(np.float32)


def opts(argv):
    """--key=value and --flag arguments after the '--'."""
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    out = {}
    for a in args:
        if a.startswith("--"):
            k, _, v = a[2:].partition("=")
            out[k] = v if v else True
    return out


def resample(a, ny, nx):
    """Bilinear resize of a 2D array."""
    h, w = a.shape
    y = np.linspace(0, h - 1.001, ny, dtype=np.float32)
    x = np.linspace(0, w - 1.001, nx, dtype=np.float32)
    iy, ix = y.astype(np.int32), x.astype(np.int32)
    fy, fx = (y - iy)[:, None], (x - ix)[None, :]
    a00, a01 = a[iy][:, ix], a[iy][:, ix + 1]
    a10, a11 = a[iy + 1][:, ix], a[iy + 1][:, ix + 1]
    return (a00 * (1 - fx) + a01 * fx) * (1 - fy) + (a10 * (1 - fx) + a11 * fx) * fy


def sample(a, u, v):
    """Bilinear lookup in a 2D array at u (0-1 across), v (0-1 down), clamped."""
    h, w = a.shape
    x, y = np.clip(u, 0, 1) * (w - 1.001), np.clip(v, 0, 1) * (h - 1.001)
    ix, iy = x.astype(np.int32), y.astype(np.int32)
    fx, fy = x - ix, y - iy
    return (a[iy, ix] * (1 - fx) + a[iy, ix + 1] * fx) * (1 - fy) + (a[iy + 1, ix] * (1 - fx) + a[iy + 1, ix + 1] * fx) * fy


def mesh_arrays(name, x, y, z, color=None, keep=None, wrap=False):
    """A mesh from n×m arrays, with an optional value per vertex (the colour attribute "C") and an optional mask of faces to keep (per vertex: a quad stays
    if any of its corners is kept). wrap joins the last column to the first."""
    n, m = z.shape
    me = bpy.data.meshes.new(name)
    a = np.arange(n * m).reshape(n, m)
    if wrap:
        q = np.stack([a[:-1, :], np.roll(a[:-1, :], -1, 1), np.roll(a[1:, :], -1, 1), a[1:, :]], -1).reshape(-1, 4)
    else:
        q = np.stack([a[:-1, :-1], a[:-1, 1:], a[1:, 1:], a[1:, :-1]], -1).reshape(-1, 4)
    if keep is not None:
        q = q[keep.ravel()[q].any(1)]
    co = np.stack([x, y, z], -1).reshape(-1, 3).astype(np.float32)
    used, inv = np.unique(q, return_inverse=True)
    q2 = inv.reshape(-1, 4)
    me.vertices.add(len(used))
    me.vertices.foreach_set("co", co[used].ravel())
    me.loops.add(q2.size)
    me.loops.foreach_set("vertex_index", q2.ravel().astype(np.int32))
    me.polygons.add(len(q2))
    me.polygons.foreach_set("loop_start", np.arange(len(q2), dtype=np.int32) * 4)
    me.polygons.foreach_set("use_smooth", np.ones(len(q2), bool))
    if color is not None:
        c = me.color_attributes.new("C", "FLOAT_COLOR", "POINT")
        v = color.ravel()[used].astype(np.float32)
        rgba = np.ones((len(used), 4), np.float32)
        rgba[:, 0] = rgba[:, 1] = rgba[:, 2] = v
        c.data.foreach_set("color", rgba.ravel())
    me.update(calc_edges=True)
    me.validate()
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    return o


def attr_material(name, rough=.9, metallic=0.0, spec=.3, rough_attr=False):
    """Base colour read from the vertices. rough_attr: roughness falls as the colour darkens, for worn metal."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "C"
    nt.links.new(at.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Specular IOR Level"].default_value = spec
    return m


def glass_material(name="water_glass", rough=0.0, ior=1.33, tint=1.0):
    """Water. Shadow rays pass straight through, or a film of water would turn everything under it black."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (tint, tint, tint, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["IOR"].default_value = ior
    b.inputs["Transmission Weight"].default_value = 1.0
    lp, tr, mix = nt.nodes.new("ShaderNodeLightPath"), nt.nodes.new("ShaderNodeBsdfTransparent"), nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Shadow Ray"], mix.inputs[0])
    nt.links.new(b.outputs[0], mix.inputs[1])
    nt.links.new(tr.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    return m


def mirror_attr_material(name="ink_water", tint=.86, rough=.04):
    """Water as a mirror, with ink floating in it: where the colour attribute C is above 0 the surface turns to matt ink."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    out = nt.nodes["Material Output"]
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (tint, tint, tint, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = rough
    d = nt.nodes.new("ShaderNodeBsdfDiffuse")
    d.inputs["Color"].default_value = (.012, .012, .012, 1)
    mix = nt.nodes.new("ShaderNodeMixShader")
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "C"
    nt.links.new(at.outputs["Fac"], mix.inputs[0])
    nt.links.new(b.outputs[0], mix.inputs[1])
    nt.links.new(d.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


def camera_rays(loc, target, lens, w=W, h=H):
    """Unit directions and the camera axes, for ground-plane grids."""
    loc, target = np.array(loc, float), np.array(target, float)
    f = target - loc
    f /= np.linalg.norm(f)
    right = np.cross(f, (0, 0, 1))
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return f, right, up, lens / 36.0 * w


def screen_plane(loc, target, lens, z=0.0, res=1.0, margin=.06, far=900.0):
    """A grid on the plane at height z, one vertex to every 1/res pixels of the frame, reaching a little past its edges. Returns X, Y."""
    f, right, up, fp = camera_rays(loc, target, lens)
    nx, ny = int(W * res * (1 + 2 * margin)), int(H * res * (1 + 2 * margin))
    sx = np.linspace(-W / 2 * (1 + 2 * margin), W / 2 * (1 + 2 * margin), nx, dtype=np.float32)
    # rows from the bottom of the frame up to just under the horizon
    pitch = math.asin(f[2])
    y_h = math.tan(-pitch) * fp
    y_bot, y_top = -H / 2 * (1 + 2 * margin), H / 2 * (1 + 2 * margin)
    y_top = min(y_top, y_h - .3)
    sy = np.linspace(y_bot, y_top, ny, dtype=np.float32)
    sxx, syy = np.meshgrid(sx, sy)
    d = (f[None, None, :] * fp + right[None, None, :] * sxx[..., None] + up[None, None, :] * syy[..., None])
    t = (z - loc[2]) / d[..., 2]
    t = np.where(t > 0, t, 1e9)
    X, Y = loc[0] + t * d[..., 0], loc[1] + t * d[..., 1]
    dist = np.sqrt((X - loc[0]) ** 2 + (Y - loc[1]) ** 2)
    s = np.minimum(1.0, far / (dist + 1e-3))
    return (loc[0] + (X - loc[0]) * s).astype(np.float32), (loc[1] + (Y - loc[1]) * s).astype(np.float32)


def stele(name, loc, turn, width, height, thick, carve=None, carve_depth=.09, seed=0, res=1.0, wear=0.0, crack=None, carve_map=None, tone=1.0):
    """A stone slab, weathered, standing at loc, facing along +(-sin(turn), -cos(turn)) (turn=0 faces -y).
    carve: name of a depth map. A pillow of two grids meeting at a worn outline. wear: 0-1, how far the water has rubbed the character out from the bottom up."""
    nv, nu = int(1800 * res), int(1000 * res)
    v = np.linspace(0, 1, nv, dtype=np.float32)[:, None]          # bottom to top
    u = np.linspace(0, 1, nu, dtype=np.float32)[None, :]
    r = np.random.default_rng(seed)
    n1 = spectral(1024, 3.2, seed + 1)
    n2 = spectral(1024, 2.9, seed + 2)
    n3 = spectral(1024, 2.5, seed + 3)
    U, V = np.meshgrid(u[0], v[:, 0])
    sc = 1024
    big = lerp(n1, U * sc * .55, V * sc)
    mid = lerp(n2, U * sc * .9, V * sc * 1.7)
    fine = lerp(n3, U * sc * 1.5, V * sc * 2.6)
    # the outline: an old slab, its corners knocked off, its top rounded a little
    du = np.abs(2 * U - 1)
    top = height * (1 - .085 * du ** 3.5) + .34 * lerp(n1, U * sc, np.full_like(U, 300.0)) * (width / 6) * .35
    z = V * top
    side = 1 + .028 * lerp(n1, np.full_like(V, 700.0), V * sc) + .010 * lerp(n2, np.full_like(V, 100.0), V * sc * 2)
    x = (U - .5) * width * side * (1 + .03 * (1 - V))
    # pillow profile: flat over the face, rounding off near the outline
    bu = np.minimum(U, 1 - U) * width
    bv = np.minimum(V * top, top - V * top)
    b = np.minimum(bu, bv)
    prof = sm(b / (thick * .55)) ** .55
    # the surface
    edge_chip = np.exp(-b / (thick * .55))
    rock = .12 * big + .030 * mid + .0035 * fine + edge_chip * (.16 * np.clip(mid, -2, 2) + .012 * fine)
    # weather runs down: vertical streaks
    streak = lerp(spectral(1024, 2.2, seed + 4), U * sc * 3.0, V * 34)
    rock = rock + .010 * streak
    # pits
    pits = blur(np.clip(lerp(spectral(1024, 1.8, seed + 5), U * sc * 2.2, V * sc * 3.4) - 1.6, 0, 1), 1.4) * .13
    rock = rock - pits
    d = np.zeros_like(U)
    groove = np.zeros_like(U)
    if carve_map is not None or carve:
        cm = carve_map if carve_map is not None else tex(carve)
        cm = sample(cm, U, 1 - V)
        keep = 1.0
        if wear > 0:
            keep = sm((V - wear + .12 * big) / .14)      # rubbed out from the bottom up
        cn = 1 + .22 * mid
        d = cm * carve_depth * height * .35 * cn * keep
        groove = cm * keep
    y_front = -(thick / 2) * prof - rock * (.8 + .4 * prof) + d
    albedo = (.60 + .10 * big + .05 * mid + .02 * fine - .10 * np.clip(streak, -1, 2) * .35)
    # rain stains, darkest near the foot and under the top edge
    stain = np.clip(lerp(spectral(1024, 2.2, seed + 6), U * sc * 2.4, V * 90) + .2, 0, 2)
    albedo = albedo - .12 * stain * (.5 + .5 * (1 - V))
    # lichen blotches
    lich = sm((lerp(spectral(1024, 2.6, seed + 7), U * sc * .9, V * sc * 1.6) - 1.0) / .25)
    albedo = albedo + .14 * lich - .10 * sm((lerp(spectral(1024, 2.6, seed + 8), U * sc * .9, V * sc * 1.6) - 1.3) / .2)
    albedo = albedo * tone - .50 * tone * groove ** .8    # soot and damp in the cuts
    albedo = albedo - .22 * edge_chip * np.clip(-mid, 0, 2) * .5
    albedo = np.clip(albedo, .03, .92)
    if crack is not None:
        depth, dark = crack(U, V, width, top)
        y_front = y_front + depth
        albedo = np.clip(albedo - dark, .01, 1)
    t = turn
    ca, sa = math.cos(t), math.sin(t)
    def place(px, py, pz):
        return loc[0] + px * ca - py * sa, loc[1] + px * sa + py * ca, loc[2] + pz
    X, Y, Z = place(x, y_front, z)
    front = mesh_arrays(name, X, Y, Z, albedo)
    y_back = (thick / 2) * prof + rock * .5
    Xb, Yb, Zb = place(x[:, ::-1], y_back[:, ::-1], z[:, ::-1])
    back = mesh_arrays(name + "_b", Xb, Yb, Zb, albedo[:, ::-1])
    mat = attr_material("stone_" + name, rough=.86, spec=.25)
    front.data.materials.append(mat)
    back.data.materials.append(mat)
    return front, back, place


def ground_mound(name, cx, cy, size, n, rise, seed, base_z=0.0, slope_to=None):
    """A low bank of rock and mud: a mound rising out of the water with broken stones on it. Returns the object and the height function's arrays."""
    y, x = np.mgrid[:n, :n].astype(np.float32)
    X, Y = cx + (x / (n - 1) - .5) * size, cy + (y / (n - 1) - .5) * size
    s = n / 1024
    b1 = lerp(spectral(1024, 3.0, seed), x / s, y / s)
    b2 = lerp(spectral(1024, 2.0, seed + 1), x / s, y / s)
    b3 = lerp(spectral(1024, 1.2, seed + 2), x / s, y / s)
    rr = np.sqrt(((X - cx) / (size * .5)) ** 2 + ((Y - cy) / (size * .5)) ** 2)
    env = 1 - sm((rr - .35) / .6)
    h = base_z + rise * env * (1 + .5 * b1) + .18 * rise * env * np.clip(b2, -2, 2) + .05 * rise * env * b3
    return X, Y, h, env


def world_split(diffuse=.4, shiny=1.1, zenith=None, cloud=.0):
    """The sky as two things: dim light on matt surfaces, and bare paper in anything that mirrors or lets light through. So water reads as paper with the ink of
    what stands over it, as a painter leaves water white. zenith: how bright the mirrored sky is overhead, below shiny, so that ripples tilting toward it show dark;
    cloud: how much it is mottled. The world is left out of light sampling, or shadow rays would see the wrong one."""
    sc = bpy.context.scene
    nt = sc.world.node_tree
    bg = nt.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)
    lp = nt.nodes.new("ShaderNodeLightPath")
    mx, mr = nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeMapRange")
    mx.operation = "MAXIMUM"
    nt.links.new(lp.outputs["Is Glossy Ray"], mx.inputs[0])
    nt.links.new(lp.outputs["Is Transmission Ray"], mx.inputs[1])
    nt.links.new(mx.outputs[0], mr.inputs["Value"])
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = diffuse, shiny
    if zenith is not None:
        tc, sep, gr = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeSeparateXYZ"), nt.nodes.new("ShaderNodeMapRange")
        nt.links.new(tc.outputs["Generated"], sep.inputs[0])
        nt.links.new(sep.outputs["Z"], gr.inputs["Value"])
        gr.interpolation_type = "SMOOTHSTEP"
        gr.inputs["From Min"].default_value, gr.inputs["From Max"].default_value = .10, .62
        gr.inputs["To Min"].default_value, gr.inputs["To Max"].default_value = shiny, zenith
        out = gr.outputs[0]
        if cloud:
            no, ml, ad = nt.nodes.new("ShaderNodeTexNoise"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeMath")
            no.inputs["Scale"].default_value = 3.0
            no.inputs["Detail"].default_value = 6
            nt.links.new(tc.outputs["Generated"], no.inputs["Vector"])
            ad.operation, ml.operation = "ADD", "MULTIPLY"
            nt.links.new(no.outputs[0], ad.inputs[0])
            ad.inputs[1].default_value = .5
            nt.links.new(ad.outputs[0], ml.inputs[0])
            nt.links.new(out, ml.inputs[1])
            out = ml.outputs[0]
        nt.links.new(out, mr.inputs["To Max"])
    nt.links.new(mr.outputs[0], bg.inputs[1])
    sc.world.cycles.sampling_method = "NONE"


def glass_bounces():
    sc = bpy.context.scene
    sc.cycles.max_bounces = 16
    sc.cycles.glossy_bounces = 10
    sc.cycles.transmission_bounces = 14
    sc.cycles.transparent_max_bounces = 12
    sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False


def border(spec):
    """Render only part of the frame, for looking at detail at full size: x0,y0,x1,y1 as fractions, y from the top."""
    x0, y0, x1, y1 = [float(v) for v in spec.split(",")]
    r = bpy.context.scene.render
    r.use_border, r.use_crop_to_border = True, False
    r.border_min_x, r.border_max_x, r.border_min_y, r.border_max_y = x0, x1, 1 - y1, 1 - y0


def land(name, cx, cy, sx, sy, peaks, seed, n=1400, wavelength=30.0, strength=.22, base=-1.0, tone=1.0, mist=None):
    """Rock from hand-placed peaks, cut by gullies and painted from its own shape: the same treatment at any scale, as a painter uses the same strokes for a stone
    and a mountain. sx, sy: width and depth of the patch, stretched from a square grid. tone scales the paint, below 1 for darker rock."""
    from ink3d import erode, ink_albedo, massif, mesh_grid, painted_material
    y, x = np.mgrid[:n, :n].astype(np.float32)
    s = max(sx, sy)
    X, Y = cx + (x / (n - 1) - .5) * s, cy + (y / (n - 1) - .5) * s
    h = massif(X, Y, peaks, seed)
    h = erode(h, s / n, seed + 1, wavelength=wavelength, octaves=7, gain=.5, strength=strength)
    h = np.maximum(h, 0)
    h = h + base * (1 - sm(h / (.06 * max(p[2] for p in peaks) + 1e-3)))
    o = mesh_grid(name, X, Y, h)
    m = painted_material(name + "_rock", np.clip(ink_albedo(h, seed + 2, s / n) * tone, .02, 1))
    if mist:
        aerial(m, **mist)
    o.data.materials.append(m)
    return o


def river(cam, target, lens, level=0.0, rings=(), phase=0.0, res=.75, seed=41, chop=.012, rough=.03, far=1000.0):
    """The river on a grid laid to the frame, so every ripple is real geometry. rings: (x, y, strength, spacing, reach), a point where something strikes the water;
    phase moves the waves outward as time goes on."""
    from ink3d import water_material
    X, Y = screen_plane(cam, target, lens, z=level, res=res, far=far)
    z = np.zeros_like(X)
    for (x0, y0, a, sp, reach) in rings:
        d = np.hypot(X - x0, Y - y0)
        z += a * np.sin((d - phase) * 2 * math.pi / sp) * np.exp(-d / reach) * sm((d - .05) / (sp * .5))
    n = spectral(1024, 2.2, seed)
    z += chop * lerp(n, (X % 400) * 2.56, (Y % 400) * 2.56)
    o = mesh_arrays("river", X, Y, level + z, None)
    o.data.materials.append(water_material(rough=rough, normal_scale=9, normal_strength=.018))
    return o


def aerial(mat, cam, k=.01, s0=40.0, H=40.0, scale=.018):
    """Mist inside the material: the farther a point is from the camera and the lower it lies, the more it turns to bare paper, in rough banks. Because it is the
    surface's own colour, what the water mirrors is misted too, which a fog added in post cannot do."""
    nt = mat.node_tree
    out = nt.nodes["Material Output"]
    src = out.inputs["Surface"].links[0].from_socket
    geo, sep = nt.nodes.new("ShaderNodeNewGeometry"), nt.nodes.new("ShaderNodeSeparateXYZ")
    d = nt.nodes.new("ShaderNodeVectorMath")
    d.operation = "DISTANCE"
    d.inputs[1].default_value = cam
    nt.links.new(geo.outputs["Position"], d.inputs[0])
    nt.links.new(geo.outputs["Position"], sep.inputs[0])

    def op(name, a, b):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = name
        for i, v in enumerate((a, b)):
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                nt.links.new(v, n.inputs[i])
        return n.outputs[0]

    far = op("MAXIMUM", op("SUBTRACT", d.outputs["Value"], s0), 0.0)
    low = op("EXPONENT", op("DIVIDE", op("MULTIPLY", sep.outputs["Z"], -1.0), H), 0.0)
    no = nt.nodes.new("ShaderNodeTexNoise")
    no.inputs["Scale"].default_value = scale
    no.inputs["Detail"].default_value = 5
    nt.links.new(geo.outputs["Position"], no.inputs["Vector"])
    bank = op("ADD", op("MULTIPLY", no.outputs[0], 2.0), 0.0)
    T = op("EXPONENT", op("MULTIPLY", op("MULTIPLY", op("MULTIPLY", far, k), low), op("MULTIPLY", bank, -1.0)), 0.0)
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (1, 1, 1, 1)
    e.inputs["Strength"].default_value = 1.0
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(T, mix.inputs[0])
    nt.links.new(e.outputs[0], mix.inputs[1])
    nt.links.new(src, mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return mat


def boulder(name, x, y, rx, ry, h, seed, n=800, level=0.0, tone=1.0, carve=None):
    """A river stone: a dome worn smooth by water, with a few planes and cracks left in it, wet and dark at the waterline, dry and pale on top. carve(X, Y, z) gives how
    far to cut in at each vertex and how much darker the cut is."""
    v, u = np.mgrid[:n, :n].astype(np.float32)
    X, Y = x + (u / (n - 1) * 2 - 1) * rx * 1.3, y + (v / (n - 1) * 2 - 1) * ry * 1.3
    s = n / 1024
    warp = lerp(blur(spectral(1024, 3.2, seed), 14.0), u / s, v / s)     # coarse only: any fine noise here is magnified into needles where the dome meets the water
    d = np.sqrt(((X - x) / rx) ** 2 + ((Y - y) / ry) ** 2) * (1 + .20 * warp)
    dome = np.clip(1 - d * d, 0, 1) ** .72
    big = lerp(spectral(1024, 3.4, seed + 1), u / s, v / s)
    plane = np.abs(lerp(spectral(1024, 3.2, seed + 2), u / s, v / s))
    mid = lerp(spectral(1024, 3.0, seed + 3), u / s, v / s)
    fine = lerp(spectral(1024, 2.6, seed + 4), u / s, v / s)
    z = level - .5 + (h + .5) * dome * (1 + .14 * big) + .05 * h * dome * plane + .008 * h * dome * mid + .0012 * h * dome * fine
    crack = np.exp(-(plane / .035) ** 2)
    z = z - .02 * h * crack * dome
    wet = 1 - sm((z - level) / (.32 * h))
    alb = (.56 + .10 * big + .06 * mid + .03 * fine - .30 * crack) * tone * (1 - .55 * wet)
    if carve:
        cut, dark = carve(X, Y, z)
        z, alb = z - cut, alb * (1 - dark)
    o = mesh_arrays(name, X, Y, z, np.clip(alb, .02, .9))
    o.data.materials.append(attr_material(name + "_m", rough=.5, spec=.5))
    return o


def film_material(name="film", rough=.03, ior=1.42, sheen=.10):
    """A thin run of water over something dark. A clear film barely shows: what the eye picks up is the sky lying along it, so part of the surface always mirrors, and
    more of it toward the edges of each run, where it turns away from you."""
    m = glass_material(name, rough=.012, ior=ior)
    nt = m.node_tree
    mix_shadow = nt.nodes["Material Output"].inputs["Surface"].links[0].from_node
    b = nt.nodes["Principled BSDF"]
    gl, mx, lw, mr, pw = nt.nodes.new("ShaderNodeBsdfGlossy"), nt.nodes.new("ShaderNodeMixShader"), nt.nodes.new("ShaderNodeLayerWeight"), nt.nodes.new("ShaderNodeMapRange"), nt.nodes.new("ShaderNodeMath")
    gl.inputs["Color"].default_value = (1, 1, 1, 1)
    gl.inputs["Roughness"].default_value = rough
    lw.inputs["Blend"].default_value = .5
    pw.operation = "POWER"
    pw.inputs[1].default_value = 2.0
    nt.links.new(lw.outputs["Facing"], pw.inputs[0])
    nt.links.new(pw.outputs[0], mr.inputs["Value"])
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = sheen, .9
    nt.links.new(mr.outputs[0], mx.inputs[0])
    nt.links.new(b.outputs[0], mx.inputs[1])
    nt.links.new(gl.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], mix_shadow.inputs[1])
    return m


def drawn_glass(name="drawn", rough=.0, ior=1.33, edge=.8, power=7.0):
    """Falling water. Against a pale sky real glass nearly vanishes, so its limbs are darkened as a painter would line them."""
    m = glass_material(name, rough=rough, ior=ior)
    nt = m.node_tree
    mix_shadow = nt.nodes["Material Output"].inputs["Surface"].links[0].from_node
    b = nt.nodes["Principled BSDF"]
    d, mx, lw, pw, ml = nt.nodes.new("ShaderNodeBsdfDiffuse"), nt.nodes.new("ShaderNodeMixShader"), nt.nodes.new("ShaderNodeLayerWeight"), nt.nodes.new("ShaderNodeMath"), nt.nodes.new("ShaderNodeMath")
    d.inputs["Color"].default_value = (.02, .02, .02, 1)
    lw.inputs["Blend"].default_value = .5
    pw.operation, ml.operation = "POWER", "MULTIPLY"
    pw.inputs[1].default_value = power
    ml.inputs[1].default_value = edge
    nt.links.new(lw.outputs["Facing"], pw.inputs[0])
    nt.links.new(pw.outputs[0], ml.inputs[0])
    nt.links.new(ml.outputs[0], mx.inputs[0])
    nt.links.new(b.outputs[0], mx.inputs[1])
    nt.links.new(d.outputs[0], mx.inputs[2])
    nt.links.new(mx.outputs[0], mix_shadow.inputs[1])
    return m


def water_ink(cam, target, lens, level=0.0, rings=(), glyphs=(), res=.8, seed=41, chop=.003, rough=.02, far=1000.0, tint=.9, warp=None):
    """The river with characters in it. rings: (x, y, strength, spacing, radius, width), a wave packet travelling out from where a drop struck, flat behind it so the
    character in its middle stays legible. glyphs: (mask, x, y, sx, sy, angle, strength, blur), ink lying on or under the surface: the mirror gives way to matt ink as
    far as its strength, so a faint one is still partly water. Each is pushed about a little by the ripples, as ink on moving water is; warp(X, Y) gives a further push."""
    X, Y = screen_plane(cam, target, lens, z=level, res=res, far=far)
    z = np.zeros_like(X)
    for (x0, y0, a, sp, R, w_) in rings:
        reach = R + 2.6 * w_
        near = (np.abs(X - x0) < reach) & (np.abs(Y - y0) < reach)
        d = np.hypot(X[near] - x0, Y[near] - y0)
        z[near] += a * np.sin((d - R) * 2 * math.pi / sp) * np.exp(-((d - R) / w_) ** 2) * sm(d / (sp * .6))
    n = spectral(1024, 2.6, seed)
    z += chop * lerp(n, (X % 400) * 2.56, (Y % 400) * 2.56)
    wx = lerp(spectral(1024, 3.0, seed + 1), (X % 200) * 5.12, (Y % 200) * 5.12)
    wy = lerp(spectral(1024, 3.0, seed + 2), (X % 200) * 5.12, (Y % 200) * 5.12)
    C = np.zeros_like(X)
    px, py = warp(X, Y) if warp else (0.0, 0.0)
    for (mask, x0, y0, sx, sy, ang, strength, bl) in glyphs:
        m = blur(mask, bl) if bl else mask
        ca, sa = math.cos(ang), math.sin(ang)
        dx, dy = X + px + .05 * wx - x0, Y + py + .05 * wy - y0
        u, v = (dx * ca + dy * sa) / sx + .5, .5 - (-dx * sa + dy * ca) / sy
        ins = (u > 0) & (u < 1) & (v > 0) & (v < 1)
        if ins.any():
            C[ins] = np.maximum(C[ins], sample(m, u[ins], v[ins]) * strength)
    o = mesh_arrays("river", X, Y, level + z, np.clip(C, 0, 1))
    o.data.materials.append(mirror_attr_material(tint=tint, rough=rough))
    return o


def ink_image(name, r, g=None, b=None):
    """A float image for the shader from up to three frame-sized arrays, top row first."""
    h, w = r.shape
    im = bpy.data.images.new(name, w, h, float_buffer=True, alpha=True)
    im.colorspace_settings.name = "Non-Color"         # before the pixels: changing it afterwards throws them away
    px = np.ones((h, w, 4), np.float32)
    px[..., 0] = r[::-1]
    px[..., 1] = (g if g is not None else r)[::-1]
    px[..., 2] = (b if b is not None else r)[::-1]
    im.pixels.foreach_set(px.ravel())
    im.pack()
    return im


def tube(name, pts, radius, mat, nq=10):
    """A line through 3D points, radius per point. pts: n×3."""
    pts = np.asarray(pts, np.float32)
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-6
    a = np.cross(t, (0, 0, 1))
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-6
    b = np.cross(t, a)
    q = np.linspace(0, 2 * math.pi, nq, dtype=np.float32)
    rr = np.asarray(radius, np.float32)[:, None]
    P = pts[:, None, :] + rr[..., None] * (a[:, None, :] * np.cos(q)[None, :, None] + b[:, None, :] * np.sin(q)[None, :, None])
    o = mesh_arrays(name, P[..., 0], P[..., 1], P[..., 2], None)
    o.data.materials.append(mat)
    return o


def matte(name, value, shadow=True):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (value, value, value, 1)
    b.inputs["Roughness"].default_value = 1.0
    b.inputs["Specular IOR Level"].default_value = .0
    return m


def project(cam, target, lens, X, Y, Z):
    """World points to pixels of the full frame, and how far they are from the camera."""
    f, right, up, fp = camera_rays(cam, target, lens)
    dx, dy, dz = X - cam[0], Y - cam[1], Z - cam[2]
    zc = dx * f[0] + dy * f[1] + dz * f[2]
    xc = dx * right[0] + dy * right[1] + dz * right[2]
    yc = dx * up[0] + dy * up[1] + dz * up[2]
    return W / 2 + xc / zc * fp, H / 2 - yc / zc * fp, np.sqrt(dx * dx + dy * dy + dz * dz)


def sphere_cloud(name, drops, mat):
    """Many small spheres as one mesh. drops: (x, y, z, r). mat: a material or the name of one."""
    uv_n, vn = 10, 7
    verts, faces = [], []
    for (x, y, z, r) in drops:
        base = len(verts)
        for j in range(vn):
            ph = math.pi * j / (vn - 1)
            for i in range(uv_n):
                th = 2 * math.pi * i / uv_n
                verts.append((x + r * math.sin(ph) * math.cos(th), y + r * math.sin(ph) * math.sin(th), z + r * 1.15 * math.cos(ph)))
        for j in range(vn - 1):
            for i in range(uv_n):
                a, c = base + j * uv_n + i, base + j * uv_n + (i + 1) % uv_n
                faces.append((a, a + uv_n, c + uv_n, c))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(bpy.data.materials[mat] if isinstance(mat, str) else mat)
    return o
