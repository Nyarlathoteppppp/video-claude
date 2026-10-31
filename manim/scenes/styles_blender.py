"""The four GPU clips of the style reel, built from code in Blender and rendered with Cycles on the Metal GPU, 5 s each:
a drop of liquid gold hitting a pool of it (液态流动, a Mantaflow simulation), a forme of movable type (3D 渲染),
a container port as a table model (等轴 2.5D) and a flight over a neon valley towards a striped sun (复古 synthwave).
styles_film.py titles the frames and cuts them into the reel.

    Blender -b --factory-startup -P manim/scenes/styles_blender.py -- SCENE [options]

SCENE is splash, letterpress, port or synthwave. Frames go to styles/film_out/frames/SCENE/f0001.png ... f0150.png;
frames already there are kept, so a stopped render picks up where it left off.
    --frames=10,60,90   just these, small and fast, into styles/film_out/frames/test/ (for choosing and checking)
    --fast              fewer samples, half size
    --res=N             splash: the simulation grid (the bake is cached per N in ~/render_work/cache)
    --rebake            splash: simulate again even if a cache exists
"""
import gzip
import json
import math
import os
import random
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector, noise

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
SCENE = ARGS[0] if ARGS else "splash"
FAST = "--fast" in ARGS
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAMES = os.path.join(ROOT, "styles", "film_out", "frames")
FPS, N = 30, 150
SONG = "/System/Library/Fonts/Supplemental/Songti.ttc"
FUTURA = "/System/Library/Fonts/Supplemental/Futura.ttc"
MENLO = "/System/Library/Fonts/Menlo.ttc"


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


# ======================================================================================
# scene plumbing
# ======================================================================================
def reset(samples, bounces=8, glossy=6):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    sc.cycles.device = "GPU"
    sc.cycles.samples = 48 if FAST else samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.01
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.max_bounces, sc.cycles.glossy_bounces = bounces, glossy
    sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.resolution_percentage = 50 if FAST else 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, N
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Punchy"
    except TypeError:
        pass
    return sc


def shoot(sc, name):
    """Render the clip's frames (skipping ones already on disk), or with --frames only the listed ones, smaller."""
    test = [int(f) for f in opt("frames", "").split(",") if f]
    if test:
        out = os.path.join(FRAMES, "test")
        sc.render.resolution_percentage = opt("pct", 40)
        sc.cycles.samples = min(sc.cycles.samples, 64)
    else:
        out = os.path.join(FRAMES, name)
    os.makedirs(out, exist_ok=True)
    for f in test or range(1, N + 1):
        path = os.path.join(out, f"{name}_f{f:04d}.png" if test else f"f{f:04d}.png")
        if not test and os.path.exists(path):
            continue
        t0 = time.time()
        sc.frame_set(f)
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f"{name} frame {f} in {time.time() - t0:.1f} s", flush=True)


def link(o):
    bpy.context.collection.objects.link(o)
    return o


def aim(o, loc, target, up="Y", track="-Z"):
    o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat(track, up).to_euler()


def key(o, path, frames_values, interp="BEZIER"):
    """Keyframe o.<path> at (frame, value) pairs."""
    for f, v in frames_values:
        setattr(o, path, v)
        o.keyframe_insert(path, frame=f)
    ad = o.animation_data
    if ad and ad.action:
        for fc in all_fcurves(ad.action):
            for k in fc.keyframe_points:
                k.interpolation = interp


def all_fcurves(action):
    """F-curves of an action, before and after the layered actions of Blender 4.4+."""
    if hasattr(action, "fcurves") and len(action.fcurves):
        return list(action.fcurves)
    out = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in strip.channelbags:
                out += list(bag.fcurves)
    return out


def material(name, color, metallic=0.0, rough=0.5, **extra):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Roughness"].default_value = rough
    for k, v in extra.items():
        p.inputs[k].default_value = v
    return m


def emission(name, color, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1)
    e.inputs["Strength"].default_value = strength
    nt.links.new(e.outputs["Emission"], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    return m


def hexc(h):
    """sRGB hex -> linear rgb, for colours picked by eye."""
    h = h.lstrip("#")
    return tuple(((int(h[i:i + 2], 16) / 255 + 0.055) / 1.055) ** 2.4 for i in (0, 2, 4))


def cube(name, size, loc, mat=None, bevel=0.0, segs=3):
    """A box as its own mesh (bmesh, no operators), bevelled in the mesh itself."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))
    if bevel:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel, segments=segs, affect="EDGES", profile=0.5)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = bevel > 0
    o = link(bpy.data.objects.new(name, me))
    o.location = loc
    if mat:
        me.materials.append(mat)
    return o


def instance(name, mesh, loc, mat=None):
    o = link(bpy.data.objects.new(name, mesh))
    o.location = loc
    if mat:
        o.material_slots[0].link = "OBJECT"
        o.material_slots[0].material = mat
    return o


def area(name, loc, target, size, energy, color, shape="SQUARE", size_y=None, seen=False):
    d = bpy.data.lights.new(name, "AREA")
    d.energy, d.color, d.shape, d.size = energy, color, shape, size
    if size_y:
        d.size_y = size_y
    o = link(bpy.data.objects.new(name, d))
    aim(o, loc, target)
    o.visible_camera = seen
    return o


def softbox(name, loc, target, w, h, strength, color, seen=False, falloff=True):
    """An emissive card, bright in the middle and falling off to its edges like a studio softbox in a reflection.
    Unless `seen`, only reflections see it, never the camera directly."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([(-w / 2, -h / 2, 0), (w / 2, -h / 2, 0), (w / 2, h / 2, 0), (-w / 2, h / 2, 0)], [], [(0, 1, 2, 3)])
    me.uv_layers.new()
    for i, uv in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        me.uv_layers[0].data[i].uv = uv
    o = link(bpy.data.objects.new(name, me))
    aim(o, loc, target, track="Z")
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    emi = nt.nodes.new("ShaderNodeEmission")
    emi.inputs["Color"].default_value = (*color, 1)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if falloff:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sub = nt.nodes.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"
        sub.inputs[1].default_value = (0.5, 0.5, 0)
        mul = nt.nodes.new("ShaderNodeVectorMath"); mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = (2.0, 2.0, 0)
        grad = nt.nodes.new("ShaderNodeTexGradient"); grad.gradient_type = "QUADRATIC_SPHERE"
        k = nt.nodes.new("ShaderNodeMath"); k.operation = "MULTIPLY"; k.inputs[1].default_value = strength
        nt.links.new(tc.outputs["UV"], sub.inputs[0])
        nt.links.new(sub.outputs["Vector"], mul.inputs[0])
        nt.links.new(mul.outputs["Vector"], grad.inputs["Vector"])
        nt.links.new(grad.outputs["Fac"], k.inputs[0])
        nt.links.new(k.outputs["Value"], emi.inputs["Strength"])
    else:
        emi.inputs["Strength"].default_value = strength
    nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])
    me.materials.append(m)
    if not seen:
        o.visible_camera = False
    o.visible_shadow = False
    return o


def camera(loc, target, lens, fstop=None, focus=None, ortho=None):
    d = bpy.data.cameras.new("cam")
    d.lens = lens
    d.sensor_width = 36
    if ortho:
        d.type, d.ortho_scale = "ORTHO", ortho
    o = link(bpy.data.objects.new("cam", d))
    aim(o, loc, target)
    if fstop:
        d.dof.use_dof = True
        d.dof.aperture_fstop = fstop
        if focus is not None:
            d.dof.focus_object = focus
        else:
            d.dof.focus_distance = (Vector(target) - Vector(loc)).length
    d.clip_start = 0.001
    d.clip_end = 2000
    bpy.context.scene.camera = o
    return o


def empty(name, loc):
    o = link(bpy.data.objects.new(name, None))
    o.location = loc
    return o


def track(cam, target):
    """Keep the camera pointed at an empty, however its location is keyed."""
    c = cam.constraints.new("TRACK_TO")
    c.target = target
    c.track_axis, c.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"


def world_flat(color, light=None, strength=1.0):
    """The camera sees a flat colour; reflections and lighting see `light` (default: the same colour)."""
    w = bpy.context.scene.world or bpy.data.worlds.new("w")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    seen = nt.nodes.new("ShaderNodeBackground"); seen.inputs["Color"].default_value = (*color, 1)
    lit = nt.nodes.new("ShaderNodeBackground"); lit.inputs["Color"].default_value = (*(light or color), 1)
    lit.inputs["Strength"].default_value = strength
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(lit.outputs["Background"], mix.inputs[1])
    nt.links.new(seen.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], nt.nodes.new("ShaderNodeOutputWorld").inputs["Surface"])


def text_mesh(body, font, size, depth, bevel=0.0, align=("CENTER", "CENTER")):
    """A text object turned into a mesh: returns the mesh data (None if the font has no outline for it)."""
    cu = bpy.data.curves.new("t", "FONT")
    cu.body = body
    cu.font = font
    cu.size = size
    cu.extrude = depth / 2
    cu.bevel_depth = bevel
    cu.bevel_resolution = 2 if bevel else 0
    cu.align_x, cu.align_y = align
    o = link(bpy.data.objects.new("t", cu))
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
    bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.curves.remove(cu)
    if not len(me.vertices):
        return None
    for p in me.polygons:
        p.use_smooth = False
    return me


# ======================================================================================
# 液态流动: liquid gold
# ======================================================================================
def flow(o, velocity=None):
    m = o.modifiers.new("fluid", "FLUID")
    m.fluid_type = "FLOW"
    fs = m.flow_settings
    fs.flow_type, fs.flow_behavior, fs.flow_source = "LIQUID", "GEOMETRY", "MESH"
    if velocity:
        fs.use_initial_velocity = True
        fs.velocity_coord = velocity
    o.hide_render = True


def bobj_path(cache, f):
    return os.path.join(cache, "mesh", f"fluid_mesh_{f:04d}.bobj.gz")


def read_bobj(path):
    """Mantaflow's mesh format: vertices in a unit cube around 0 (by the domain's longest side), normals, triangles."""
    with gzip.open(path, "rb") as fh:
        raw = fh.read()
    nv = int.from_bytes(raw[:4], "little")
    at = 4
    co = np.frombuffer(raw, "<f4", nv * 3, at).reshape(-1, 3)
    at += nv * 12
    nn = int.from_bytes(raw[at:at + 4], "little")
    at += 4
    normals = np.frombuffer(raw, "<f4", nn * 3, at).reshape(-1, 3)
    at += nn * 12
    nt = int.from_bytes(raw[at:at + 4], "little")
    at += 4
    return co, np.frombuffer(raw, "<i4", nt * 3, at), normals


def splash():
    """A 12 mm drop of liquid gold hits a shallow pool at 3.5 m/s, simulated with Mantaflow at real size and played about
    30x slower than life and slowing: the crown rises until its rim tears into lace and threads of gold fly off it. Dark
    studio, long strip lights, the lens low over the pool."""
    sc = reset(256, 12, 8)
    res = opt("res", 300)
    dx, dz, pool, r = opt("dx", 0.15), opt("dz", 0.08), opt("pool", 0.010), 0.006    # a shallow pool stands the crown up
    hw = dx / 2
    slow = opt("slow", 2.0)                                # simulated frames per frame of the base 30x slow motion
    off = (0.0008, 0.0005)                                 # a dead-centre hit would print the grid's symmetry on the crown
    unit = max(dx, dz)                                     # the cache's meshes span the box's longest side
    sim = opt("sim", 240)
    cache = os.path.expanduser(f"~/render_work/cache/crown_{res}{opt('tag', '_thin')}")
    if "--rebake" in ARGS or not os.path.exists(bobj_path(cache, sim)):
        dom = cube("domain", (dx, dx, dz), (0, 0, dz / 2))
        m = dom.modifiers.new("fluid", "FLUID")
        m.fluid_type = "DOMAIN"
        ds = m.domain_settings
        ds.domain_type = "LIQUID"
        ds.resolution_max = res
        ds.use_mesh = True
        ds.mesh_scale = 2
        ds.use_fractions = True
        ds.time_scale = 0.0167 / slow
        ds.cfl_condition = 2.0
        ds.timesteps_max, ds.timesteps_min = 16, 1
        ds.cache_type = "ALL"
        ds.cache_frame_start, ds.cache_frame_end = 1, sim
        ds.cache_directory = cache
        pl = cube("pool", (dx * 1.1, dx * 1.1, pool), (0, 0, pool / 2))
        flow(pl)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(*off, pool + r + 0.002), segments=48, ring_count=24)
        drop = bpy.context.object
        flow(drop, (0.0, 0.0, -opt("v", 3.5)))
        t0 = time.time()
        with bpy.context.temp_override(scene=sc, object=dom, active_object=dom, selected_objects=[dom],
                                       selected_editable_objects=[dom]):
            bpy.ops.fluid.bake_all()
        print(f"baked res {res} in {time.time() - t0:.0f} s", flush=True)
        if "--bakeonly" in ARGS:
            return
        for o in (dom, pl, drop):
            bpy.data.objects.remove(o, do_unlink=True)

    gold = material("gold", (1.0, 0.6, 0.11), metallic=1.0, rough=0.035, **{"Specular Tint": (1.0, 0.7, 0.2, 1)})
    # seen by the lens the pool is no mirror but a dark, faintly rough sheen: a crown's reflection melts away within its own
    # height and leaves a glow. Seen by the crown it is still gold, or the walls would mirror a black floor.
    nt = gold.node_tree
    metal = nt.nodes["Principled BSDF"]
    # only the bounce the lens sees is tinted full gold: metal mirroring metal would multiply its way to burning red
    tint = nt.nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.inputs["A"].default_value = (1.0, 0.95, 0.85, 1)
    tint.inputs["B"].default_value = metal.inputs["Base Color"].default_value
    nt.links.new(nt.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"], tint.inputs["Factor"])
    nt.links.new(tint.outputs["Result"], metal.inputs["Base Color"])
    height = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"], height.inputs[0])
    shore = nt.nodes.new("ShaderNodeMapRange")
    shore.interpolation_type = "SMOOTHSTEP"
    shore.inputs["From Min"].default_value, shore.inputs["From Max"].default_value = pool + 0.0008, pool + 0.0036
    shore.inputs["To Min"].default_value, shore.inputs["To Max"].default_value = 1.0, 0.0
    nt.links.new(height.outputs["Z"], shore.inputs["Value"])
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value, rough.inputs["To Max"].default_value = 0.035, 0.3
    nt.links.new(rough.outputs[0], metal.inputs["Roughness"])
    grain = nt.nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value, grain.inputs["Detail"].default_value = opt("grain", 420.0), 3.0
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], grain.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Distance"].default_value = opt("bump", 0.00008)
    nt.links.new(grain.outputs[0], bump.inputs["Height"])
    sheen = nt.nodes.new("ShaderNodeBsdfAnisotropic")
    # and it dies away with distance from the crown, so that the pool has no far edge and no horizon, only darkness
    nearer = nt.nodes.new("ShaderNodeMath")
    nearer.operation = "MINIMUM"
    nearer.inputs[1].default_value = 0.0
    nt.links.new(height.outputs["Y"], nearer.inputs[0])
    flat = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(height.outputs["X"], flat.inputs["X"])
    nt.links.new(nearer.outputs[0], flat.inputs["Y"])
    away = nt.nodes.new("ShaderNodeVectorMath")
    away.operation = "LENGTH"
    nt.links.new(flat.outputs[0], away.inputs[0])
    fade = nt.nodes.new("ShaderNodeMapRange")
    fade.interpolation_type = "SMOOTHSTEP"
    fade.inputs["From Min"].default_value, fade.inputs["From Max"].default_value = opt("fadein", 0.03), opt("fadeout", 0.14)
    fade.inputs["To Min"].default_value, fade.inputs["To Max"].default_value = 1.0, 0.0
    nt.links.new(away.outputs["Value"], fade.inputs["Value"])
    # perspective squeezes all the far side into a few rows, so there it fades by the angle of view, not by distance
    view = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(nt.nodes.new("ShaderNodeNewGeometry").outputs["Incoming"], view.inputs[0])
    far = nt.nodes.new("ShaderNodeMapRange")
    far.interpolation_type = "SMOOTHSTEP"
    far.inputs["From Min"].default_value, far.inputs["From Max"].default_value = opt("far0", 0.012), opt("far1", 0.085)
    nt.links.new(view.outputs["Z"], far.inputs["Value"])
    fading = nt.nodes.new("ShaderNodeMath")
    fading.operation = "MULTIPLY"
    nt.links.new(fade.outputs[0], fading.inputs[0])
    nt.links.new(far.outputs[0], fading.inputs[1])
    dim = nt.nodes.new("ShaderNodeMix")
    dim.data_type = "RGBA"
    dim.inputs["A"].default_value = (0, 0, 0, 1)
    dim.inputs["B"].default_value = (*(opt("sheen", 0.32) * c for c in (1.0, 0.75, 0.4)), 1)
    nt.links.new(fading.outputs[0], dim.inputs["Factor"])
    nt.links.new(dim.outputs["Result"], sheen.inputs["Color"])
    sheen.inputs["Roughness"].default_value = opt("poolrough", 0.22)
    nt.links.new(bump.outputs[0], sheen.inputs["Normal"])
    direct = nt.nodes.new("ShaderNodeMath")
    direct.operation = "MULTIPLY"
    nt.links.new(shore.outputs[0], direct.inputs[0])
    nt.links.new(nt.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"], direct.inputs[1])
    nt.links.new(direct.outputs[0], rough.inputs["Value"])    # the crown and drop see a clean mirror: no seam at their horizon
    both = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(direct.outputs[0], both.inputs[0])
    nt.links.new(metal.outputs[0], both.inputs[1])
    nt.links.new(sheen.outputs[0], both.inputs[2])
    nt.links.new(both.outputs[0], next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL").inputs["Surface"])
    centre = np.array((0.0, 0.0, dz / 2), np.float32)
    # the drop falls for 0.4 s, then the crown plays at a whole number of meshes per frame, as meshes cannot be
    # interpolated and an uneven cadence judders: quick through the impact, slowing as the crown opens
    pre = 12
    speed = np.concatenate([np.full(int(p.split(":")[1]), float(p.split(":")[0])) for p in opt("ramp", "3:8,2:12,1:117").split(",")])
    speed = np.convolve(np.pad(speed, 4, mode="edge"), np.ones(9) / 9, "valid")
    table = np.concatenate((np.ones(pre + 1), 1 + np.cumsum(speed)))
    table = np.pad(table, (0, max(0, N - len(table))), mode="edge")

    def sim_frame(f):
        return f if "--noramp" in ARGS else min(int(round(table[min(f, N) - 1])), sim)

    levels = os.path.join(cache, "waterline.json")         # the walls' water line, by simulated frame
    line = {}
    if os.path.exists(levels):
        with open(levels) as fh:
            line = json.load(fh)
    if any(str(sim_frame(f)) not in line for f in range(1, N + 1)):
        for f in range(1, N + 1):
            k = str(sim_frame(f))
            if k not in line:
                co, _, normals = read_bobj(bobj_path(cache, int(k)))
                co = co * unit + centre
                edge = (np.maximum(np.abs(co[:, 0]), np.abs(co[:, 1])) > dx / 2 - 0.012) & (normals[:, 2] > 0.9)
                line[k] = float(np.median(co[edge, 2])) if edge.any() else pool
        with open(levels, "w") as fh:
            json.dump(line, fh)

    def ripple(rad, k):
        """The crown's rings, running out across the still pool, as a height over the water line."""
        run = max(0.0, k / slow - 40)
        z = rad * 0.0
        for delay, amp in ((0, 0.00005), (26, 0.00004), (55, 0.00003)):
            if run > delay:
                front = 0.046 + opt("ring", 0.00075) * (run - delay)
                z += amp * min(1.0, (run - delay) / 16) * np.cos((rad - front) / 0.002) * np.exp(-((rad - front) / 0.016) ** 2)
        return z

    def settle(co, normals, level):
        """The pool's own surface low-passed as a height field: a grazing reflection turns FLIP's grid chatter to tiles."""
        g, sigma = 0.0005, opt("settle", 0.0024)
        n = int(dx / g) + 2
        pick = (np.abs(co[:, 2] - level) < 0.007) & (normals[:, 2] > 0.45)
        x, y = ((co[:, 0] + hw) / g).clip(0, n - 1.001), ((co[:, 1] + hw) / g).clip(0, n - 1.001)
        ix, iy = x.astype(np.int32), y.astype(np.int32)
        cell = ix[pick] * n + iy[pick]
        height = np.bincount(cell, weights=co[pick, 2] - level, minlength=n * n).reshape(n, n)
        weight = np.bincount(cell, minlength=n * n).reshape(n, n).astype(np.float64)
        fy, fx = np.fft.rfftfreq(n), np.fft.fftfreq(n)[:, None]
        blur = np.exp(-2 * (math.pi * sigma / g) ** 2 * (fx ** 2 + fy ** 2))
        height = np.fft.irfft2(np.fft.rfft2(height) * blur, s=(n, n))
        weight = np.fft.irfft2(np.fft.rfft2(weight) * blur, s=(n, n))
        field = height / np.maximum(weight, 1e-3)
        fx, fy = (x - ix)[pick], (y - iy)[pick]
        ix, iy = ix[pick], iy[pick]
        low = (field[ix, iy] * (1 - fx) * (1 - fy) + field[ix + 1, iy] * fx * (1 - fy)
               + field[ix, iy + 1] * (1 - fx) * fy + field[ix + 1, iy + 1] * fx * fy)
        keep = np.clip(1 - (np.abs(co[pick, 2] - level) - 0.005) / 0.002, 0, 1) * np.clip((normals[pick, 2] - 0.45) / 0.25, 0, 1)
        co[pick, 2] = co[pick, 2] * (1 - keep) + (level + low) * keep

    def loose_drops(co, tri, level):
        """Drops of a few cells come out of the grid as spiky scraps, and smoothing shrinks them: take them out of the
        mesh and hand back each one's centre and the radius of a ball of the same volume."""
        up = co[:, 2] > level + 0.002
        edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
        inside = edges[up[edges[:, 0]] & up[edges[:, 1]]]
        label = np.arange(len(co))
        while True:                                        # connected pieces, by hooking and pointer jumping
            a, b = label[inside[:, 0]], label[inside[:, 1]]
            least = np.minimum(a, b)
            new = label.copy()
            np.minimum.at(new, a, least)
            np.minimum.at(new, b, least)
            new = new[new[new]]
            if np.array_equal(new, label):
                break
            label = new
        where = np.flatnonzero(up)
        ids, piece = np.unique(label[where], return_inverse=True)
        lo, hi = np.full((len(ids), 3), 9.0), np.full((len(ids), 3), -9.0)
        for c in range(3):
            np.minimum.at(lo[:, c], piece, co[where, c])
            np.maximum.at(hi[:, c], piece, co[where, c])
        small = (hi - lo).max(axis=1) < opt("drop", 0.0035)
        of = np.full(len(co), -1)
        of[where] = piece
        shore = edges[up[edges[:, 0]] != up[edges[:, 1]]]
        small[of[shore[up[shore[:, 0]], 0]]] = False       # still joined to what lies under the cut
        small[of[shore[up[shore[:, 1]], 1]]] = False
        first = of[tri[:, 0]]
        gone = (first >= 0) & small[np.maximum(first, 0)]
        mid = (lo + hi) / 2
        p = co[tri[gone]].astype(np.float64) - mid[first[gone]][:, None, :]
        volume = np.abs(np.bincount(first[gone], np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])) / 6, minlength=len(ids)))
        rad = np.clip(np.cbrt(3 * volume / (4 * math.pi)), 0.00025, 0.0016)
        rad *= np.clip((hw - 0.004 - np.hypot(mid[:, 0], mid[:, 1])) / 0.012, 0, 1)    # and what reaches the box's walls has shrunk away
        return tri[~gone], mid[small], rad[small]

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0)
    ball = bpy.context.object
    ball_co = np.array([v.co for v in ball.data.vertices], np.float32)
    ball_tri = np.array([p.vertices for p in ball.data.polygons], np.int32)
    bpy.data.objects.remove(ball, do_unlink=True)

    def drops_mesh(mid, rad):
        """Each loose drop, flying out and up from the crown's foot: a round head drawing a thread behind it."""
        n = len(mid)
        way = np.stack((mid[:, 0], mid[:, 1], np.maximum(mid[:, 2] - pool, 0.002) * 1.2), -1)
        way = way / np.maximum(np.linalg.norm(way, axis=1, keepdims=True), 1e-6)
        along = ball_co @ way.T                            # (ball vertices, drops)
        stretch = np.clip(0.0012 / np.maximum(rad, 1e-5) + 1.5, 2.0, opt("thread", 5.0))
        off_axis = ball_co[None] - way[:, None, :] * along.T[:, :, None]
        taper = 0.22 + 0.78 * ((1 + along.T) / 2) ** 1.5
        co = (mid[:, None, :] + rad[:, None, None] * (off_axis * taper[:, :, None] + way[:, None, :] * (along.T * stretch[:, None])[:, :, None])).reshape(-1, 3)
        speed = np.repeat(way * opt("fling", 0.06), len(ball_co), axis=0)
        tri = (ball_tri[None] + (np.arange(n, dtype=np.int32) * len(ball_co))[:, None, None]).reshape(-1)
        me = bpy.data.meshes.new("drops")
        me.vertices.add(len(co))
        me.vertices.foreach_set("co", co.astype(np.float32).ravel())
        me.loops.add(len(tri))
        me.loops.foreach_set("vertex_index", tri)
        me.polygons.add(len(tri) // 3)
        me.polygons.foreach_set("loop_start", np.arange(0, len(tri), 3, dtype=np.int32))
        me.update(calc_edges=True)
        me.shade_smooth()
        me.materials.append(gold)
        me.attributes.new("velocity", "FLOAT_VECTOR", "POINT").data.foreach_set("vector", speed.astype(np.float32).ravel())
        return me

    def frame_mesh(f):
        """A fresh scene cannot replay a bake, so the cache's own mesh files are read straight in."""
        k = sim_frame(f)
        if f <= pre:                                       # still falling: the pool is a plain mirror under a plain sphere
            loose[:] = np.zeros((0, 3)), np.zeros(0)
            me = bpy.data.meshes.new("liquid")
            me.from_pydata([((hw - 0.0025) * math.cos(a), (hw - 0.0025) * math.sin(a), line[str(k)] - 0.00002) for a in np.linspace(0, 2 * math.pi, 360, endpoint=False)],
                           [], [tuple(range(360))])
            me.materials.append(gold)
            return me
        co, tris, normals = read_bobj(bobj_path(cache, k))
        co = co * unit + centre
        level = line[str(k)]
        settle(co, normals, level)
        # the box's walls must never show: towards them the surface settles flat, and the mesh is cut off in a circle
        # where the open pool takes over at the same height
        rad = np.hypot(co[:, 0], co[:, 1])
        u = np.clip((rad - (hw - 0.014)) / 0.010, 0, 1)
        near = u * u * (3 - 2 * u)
        near *= np.clip(1 - (np.abs(co[:, 2] - level) - 0.003) / 0.006, 0, 1)
        near = np.where((rad > hw - 0.0035) & (np.abs(co[:, 2] - level) < 0.006), 1.0, near)    # a rim passing overhead stays up
        # and out past the disturbance the pool has not been touched yet: no grid chatter on the mirror
        reach = np.clip((rad - (0.016 + 0.00085 * k / slow)) / 0.008, 0, 1)
        reach = (reach * reach * (3 - 2 * reach)) * np.clip(1 - (np.abs(co[:, 2] - level) - 0.0015) / 0.003, 0, 1)
        near = np.maximum(near, reach)
        co[:, 2] = co[:, 2] * (1 - near) + (level + ripple(rad, k)) * near
        tri = tris.reshape(-1, 3)
        keep = rad[tri].min(axis=1) < hw - 0.0008
        tri, mid, size = loose_drops(co, tri[keep], level)
        loose[:] = mid, size
        me = bpy.data.meshes.new("liquid")
        me.vertices.add(len(co))
        me.vertices.foreach_set("co", co.ravel())
        me.loops.add(tri.size)
        me.loops.foreach_set("vertex_index", tri.ravel())
        me.polygons.add(len(tri))
        me.polygons.foreach_set("loop_start", np.arange(0, tri.size, 3, dtype=np.int32))
        me.update(calc_edges=True)
        me.shade_smooth()
        me.materials.append(gold)
        low[:] = np.flatnonzero(co[:, 2] < level + 0.006).tolist()
        # meshes share no vertices from frame to frame, so motion blur gets the crown's measured spread: rim out, tips up
        rate = (table[min(f, N - 1)] - table[min(f, N - 1) - 1]) * FPS
        lift = np.clip((co[:, 2] - level - 0.001) / 0.002, 0, 1)
        out = lift * rad / max(0.0032 + 0.00039 * k, 0.01) * 0.00039 * rate / np.maximum(rad, 1e-4)
        v = np.stack((co[:, 0] * out, co[:, 1] * out, lift * (co[:, 2] - level) / max(0.0002 * k, 0.004) * 0.0002 * rate), -1)
        me.attributes.new("velocity", "FLOAT_VECTOR", "POINT").data.foreach_set("vector", v.astype(np.float32).ravel())
        return me

    low, loose = [], [np.zeros((0, 3)), np.zeros(0)]
    liquid = link(bpy.data.objects.new("liquid", frame_mesh(1)))
    drops = link(bpy.data.objects.new("drops", drops_mesh(*loose)))
    sm = liquid.modifiers.new("smooth", "SMOOTH")          # the grid's chatter is not the liquid's surface
    sm.factor, sm.iterations = 0.8, opt("smooth", 60)
    foot = liquid.vertex_groups.new(name="foot")           # and the crown's foot, seen at a grazing angle, shows its steps longest
    calm = liquid.modifiers.new("calm", "SMOOTH")
    calm.factor, calm.iterations, calm.vertex_group = 0.8, opt("calm", 120), "foot"

    # the open pool past the simulated box, with the crown's ripples running out across it
    rr, aa = np.meshgrid(np.geomspace(hw - 0.0025, 30.0, 300), np.linspace(0, 2 * math.pi, 360, endpoint=False), indexing="ij")
    rn, an = rr.shape
    me = bpy.data.meshes.new("open pool")
    me.from_pydata(np.stack((rr * np.cos(aa), rr * np.sin(aa), rr * 0 + pool), -1).reshape(-1, 3).tolist(), [],
                   [(i * an + j, (i + 1) * an + j, (i + 1) * an + (j + 1) % an, i * an + (j + 1) % an) for i in range(rn - 1) for j in range(an)])
    me.shade_smooth()
    me.materials.append(gold)
    far = link(bpy.data.objects.new("open pool", me))

    def open_pool(f):
        k = sim_frame(f)
        co = np.stack((rr * np.cos(aa), rr * np.sin(aa), ripple(rr, k) + line[str(k)] - 0.00002), -1)
        far.data.vertices.foreach_set("co", co.ravel())
        far.data.update()

    def load_frame(scene, depsgraph=None):
        old = liquid.data
        liquid.data = frame_mesh(scene.frame_current)
        sm.show_render = calm.show_render = scene.frame_current > pre     # smoothing would pull the still pool's edge away from the open pool
        foot.add(low, 1.0, "REPLACE")
        bpy.data.meshes.remove(old)
        old = drops.data
        drops.data = drops_mesh(*loose)
        bpy.data.meshes.remove(old)
        open_pool(scene.frame_current)
        shape_drop(scene.frame_current)
        liquid.visible_glossy = far.visible_glossy = scene.frame_current > pre    # the falling drop mirrors only the room: no pool edge across its belly

    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=128, ring_count=96, location=(*off, pool + r + 0.016))
    drop = bpy.context.object
    bpy.ops.object.shade_smooth()
    # the drop alone is a little satin, or it shows a hard copy of every lamp and of the pool's edge
    drop.data.materials.append(material("gold drop", (1.0, 0.6, 0.11), metallic=1.0, rough=opt("droprough", 0.11), **{"Specular Tint": (1.0, 0.7, 0.2, 1)}))
    drop_co = np.empty(len(drop.data.vertices) * 3, np.float32)
    drop.data.vertices.foreach_get("co", drop_co)
    drop_co = drop_co.reshape(-1, 3)

    def shape_drop(f):
        """Pulled to a point at first, it rounds up into the ball that the simulation starts from."""
        tail = 1.2 * (1 - min((f - 1) / pre, 1.0)) ** 1.6
        top = np.maximum(drop_co[:, 2] / r, 0)
        co = drop_co.copy()
        co[:, :2] *= (1 - 0.4 * tail * top ** 2)[:, None]
        co[:, 2] += r * tail * top ** 4
        drop.data.vertices.foreach_set("co", co.ravel())
        drop.data.update()

    shape_drop(1)
    bpy.app.handlers.frame_change_pre.append(load_frame)
    for f in range(1, pre + 2):
        u = (f - 1) / pre
        drop.location.z = pool + r + 0.002 + opt("fall", 0.009) * (1 - u) ** 1.8    # accelerating into the simulation's speed
        drop.keyframe_insert("location", index=2, frame=f)
        drop.hide_render = f > pre
        drop.keyframe_insert("hide_render", frame=f)

    me = bpy.data.meshes.new("floor")                      # under the crater, should the film tear through
    me.from_pydata([(-hw, -hw, 0.0002), (hw, -hw, 0.0002), (hw, hw, 0.0002), (-hw, hw, 0.0002)], [], [(0, 1, 2, 3)])
    me.materials.append(gold)
    link(bpy.data.objects.new("floor", me)).visible_glossy = False
    sc.view_settings.view_transform = "Khronos PBR Neutral"     # keeps gold's hue right up to the highlight, where Standard clips to lemon
    sc.view_settings.look = "None"
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = opt("shutter", 0.5)
    w = bpy.data.worlds.new("gold room")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    # what the metal mirrors: long soft bands, which liquid metal draws out into flowing lines. They hang only on the lens's
    # side of the room: the walls facing the lens mirror them and shine, while the pool, grazed, mirrors the dark far side.
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = -1.0, 1.0
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    grad = nt.nodes.new("ShaderNodeValToRGB")
    els = grad.color_ramp.elements
    els[0].position, els[0].color = 0.0, (1.0, 0.78, 0.42, 1)
    els.new(0.18).color = (0.6, 0.43, 0.2, 1)                  # below the horizon only the falling drop can see the room: a glow low in its belly
    els.new(0.38).color = (0.1, 0.065, 0.03, 1)
    els[1].position, els[1].color = 1.0, (0.6, 0.46, 0.26, 1)
    els.new(0.45).color = (0.25, 0.16, 0.07, 1)
    els.new(0.5).color = (0.42, 0.31, 0.15, 1)
    els.new(0.54).color = (0.75, 0.6, 0.36, 1)
    els.new(0.62).color = (0.28, 0.2, 0.1, 1)
    els.new(0.7).color = (1.0, 0.82, 0.55, 1)
    els.new(0.8).color = (0.5, 0.38, 0.2, 1)
    els.new(0.9).color = (0.85, 0.68, 0.42, 1)
    nt.links.new(mr.outputs[0], grad.inputs["Fac"])
    fill = nt.nodes.new("ShaderNodeBackground")
    side = nt.nodes.new("ShaderNodeMapRange")
    side.interpolation_type = "SMOOTHSTEP"
    side.inputs["From Min"].default_value, side.inputs["From Max"].default_value = -0.35, 0.7
    side.inputs["To Min"].default_value, side.inputs["To Max"].default_value = opt("fill", 1.0), 0.015
    nt.links.new(sep.outputs["Y"], side.inputs["Value"])
    nt.links.new(side.outputs[0], fill.inputs["Strength"])
    nt.links.new(grad.outputs["Color"], fill.inputs["Color"])
    # what the lens sees: near black, with a little warmth low behind the crown
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = (0.0, 1.0, 0.02)
    nt.links.new(tc.outputs["Generated"], dot.inputs[0])
    power = nt.nodes.new("ShaderNodeMath")
    power.operation = "POWER"
    power.use_clamp = True
    power.inputs[1].default_value = opt("glowfall", 90.0)
    nt.links.new(dot.outputs["Value"], power.inputs[0])
    tint = nt.nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.inputs["A"].default_value = (0.002, 0.0013, 0.0006, 1)
    tint.inputs["B"].default_value = (*(opt("glow", 0.003) * c for c in (1.0, 0.5, 0.1)), 1)
    nt.links.new(power.outputs[0], tint.inputs["Factor"])
    dark = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(tint.outputs["Result"], dark.inputs["Color"])
    seen = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(nt.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"], seen.inputs[0])
    nt.links.new(fill.outputs[0], seen.inputs[1])
    nt.links.new(dark.outputs[0], seen.inputs[2])
    nt.links.new(seen.outputs[0], nt.nodes.new("ShaderNodeOutputWorld").inputs["Surface"])

    def strip(name, loc, w, h, strength, color):
        """A long soft tube of light: on metal it draws a thin line that runs along the liquid, never a blob."""
        o = softbox(name, loc, (0, 0, pool + 0.012), w, h, strength, color)
        scale = next(n for n in o.active_material.node_tree.nodes if n.type == "VECT_MATH" and n.operation == "MULTIPLY")
        scale.inputs[1].default_value = (2.0, 0.35, 0.0) if h > w else (0.35, 2.0, 0.0)
        return o

    strip("top", (0.0, -0.14, 0.55), 3.0, opt("topw", 0.3), opt("top", 1.5), (1.0, 0.9, 0.72))
    strip("side", (-0.46, -0.3, 0.2), opt("sidew", 0.1), 2.0, opt("side", 2.6), (1.0, 0.9, 0.72))
    strip("back", (0.08, 0.85, 0.11), 2.4, 0.05, opt("back", 0.2), (1.0, 0.72, 0.36))
    focus = empty("focus", (0, 0, pool + r))
    cam = camera((0, -0.2, pool + 0.02), (0, 0, pool), 105, opt("fstop", 50.0), focus)
    aim_at = empty("aim", (0, 0, pool))
    track(cam, aim_at)
    # low over the pool; the point of impact sits on the lower third, the crown nearly filling the frame's width
    #            frame  dist   aim z   azimuth elevation  focus towards lens
    stops = ((1,        0.270, 0.0087, -14.0,  3.0,       0.0),
             (pre + 1,  0.190, 0.0068, -12.0,  3.0,       0.0),
             (30,       0.170, 0.0085, -9.0,   4.0,       0.02),
             (54,       0.225, 0.0095, -6.0,   3.2,       0.018),
             (100,      0.330, 0.0140, 0.0,    3.6,       0.026),
             (N,        0.420, 0.0180, 7.0,    4.0,       0.030))
    path = [np.convolve(np.pad(np.interp(np.arange(1, N + 1), [q[0] for q in stops], [q[i] for q in stops]), 10, mode="edge"),
                        np.ones(21) / 21, "valid") for i in range(1, 6)]
    for f in range(1, N + 1):
        d, z, az, el, front = (float(p[f - 1]) for p in path)
        az, el = math.radians(az), math.radians(el)
        cam.location = (d * math.cos(el) * math.sin(az), -d * math.cos(el) * math.cos(az), pool + z + d * math.sin(el))
        cam.keyframe_insert("location", frame=f)
        aim_at.location.z = pool + z
        aim_at.keyframe_insert("location", index=2, frame=f)
        focus.location = (front * math.sin(az), -front * math.cos(az), pool + z)
        focus.keyframe_insert("location", frame=f)
    shoot(sc, "splash")


# ======================================================================================
# 3D 渲染: movable type
# ======================================================================================
PASSAGE = ("板印书籍唐人尚未盛为之自冯瀛王始印五经已后典籍皆为板本庆历中有布衣毕昇又为活板其法用胶泥刻字薄如钱唇"
           "每字为一印火烧令坚先设一铁板其上以松脂蜡和纸灰之类冒之欲印则以一铁范置铁板上乃密布字印满铁范为一板持就火炀之"
           "药稍熔则以一平板按其面则字平如砥若止印三二本未为简易若印数十百千本则极为神速常作二铁板一板印刷一板已自布字"
           "此印者才毕则第二板已具更互用之瞬息可就每一字皆有数印如之也等字每字有二十余印以备一板内有重复者不用则以纸贴之"
           "每韵为一帖木格贮之有奇字素无备者旋刻之以草火烧瞬息可成不以木为之者木理有疏密沾水则高下不平兼与药相粘不可取"
           "不若燔土用讫再火令药熔以手拂之其印自落殊不沾污昇死其印为予群从所得至今保藏")


def letterpress():
    """A forme of lead type set with Shen Kuo's account of Bi Sheng's movable type, every face cut in mirror image. Macro
    on the backwards faces; a roller inks them; a hinged tympan with a sheet in it slams down and swings open like a page,
    and what was backwards on the metal reads true on the paper, with 活 in cinnabar at its heart."""
    sc = reset(256, 10, 6)
    rnd = random.Random(11)
    font = bpy.data.fonts.load(SONG)
    pitch, body, high = 0.0095, 0.0094, 0.024
    cols, rows = 34, 30
    def type_metal(name, stops, metal, rough, scale):
        """Noise per piece of type (Object Info random as W) driving colour, metalness and roughness."""
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        p = nt.nodes["Principled BSDF"]
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.noise_dimensions = "4D"
        nz.inputs["Scale"].default_value = scale
        nz.inputs["Detail"].default_value = 6
        nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], nz.inputs["Vector"])
        k = nt.nodes.new("ShaderNodeMath")
        k.operation = "MULTIPLY"
        k.inputs[1].default_value = 60.0
        nt.links.new(nt.nodes.new("ShaderNodeObjectInfo").outputs["Random"], k.inputs[0])
        nt.links.new(k.outputs["Value"], nz.inputs["W"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        for el_, (pos, col) in zip(ramp.color_ramp.elements, stops):
            el_.position, el_.color = pos, (*hexc(col), 1)
        nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], p.inputs["Base Color"])
        for socket, (a_, b_) in (("Metallic", metal), ("Roughness", rough)):
            rr = nt.nodes.new("ShaderNodeMapRange")
            rr.inputs["From Min"].default_value, rr.inputs["From Max"].default_value = stops[0][0], stops[1][0]
            rr.inputs["To Min"].default_value, rr.inputs["To Max"].default_value = a_, b_
            nt.links.new(nz.outputs["Fac"], rr.inputs["Value"])
            nt.links.new(rr.outputs["Result"], p.inputs[socket])
        return m

    # the shoulders: dull, oxidised. The faces: polished by use, finely grained, wiped clean.
    lead = type_metal("lead", ((0.35, "#474a49"), (0.65, "#5c5f5d")), (1.0, 1.0), (0.46, 0.58), 5200.0)
    ink = type_metal("ink", ((0.38, "#9a9b96"), (0.62, "#c0c0ba")), (1.0, 1.0), (0.2, 0.32), 5200.0)

    def stained(m, distance, clean_from, clean_at, inside=False):
        """Old ink stays where no cloth can reach: black at the foot of a stroke and in the crook between two, a little
        unevenly. With `inside`, on the face itself, along its edges and in its corners."""
        nt = m.node_tree
        pb = nt.nodes["Principled BSDF"]
        ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
        ao.samples, ao.inside = 12, inside
        ao.inputs["Distance"].default_value = distance
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.noise_dimensions = "4D"
        nz.inputs["Scale"].default_value = 1400.0
        nz.inputs["Detail"].default_value = 3
        nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], nz.inputs["Vector"])
        nt.links.new(nt.nodes["Noise Texture"].inputs["W"].links[0].from_socket, nz.inputs["W"])
        off = nt.nodes.new("ShaderNodeMath")
        off.operation = "SUBTRACT"
        off.inputs[1].default_value = 0.5
        nt.links.new(nz.outputs["Fac"], off.inputs[0])
        patchy = nt.nodes.new("ShaderNodeMath")
        patchy.operation = "MULTIPLY_ADD"
        patchy.inputs[1].default_value = 0.5
        nt.links.new(off.outputs[0], patchy.inputs[0])
        nt.links.new(ao.outputs["AO"], patchy.inputs[2])
        kept = nt.nodes.new("ShaderNodeMapRange")
        kept.interpolation_type = "SMOOTHSTEP"
        kept.inputs["From Min"].default_value, kept.inputs["From Max"].default_value = clean_from, clean_at
        kept.inputs["To Min"].default_value, kept.inputs["To Max"].default_value = 1.0, 0.0
        nt.links.new(patchy.outputs[0], kept.inputs["Value"])
        tone = nt.nodes.new("ShaderNodeMix")
        tone.data_type = "RGBA"
        nt.links.new(kept.outputs[0], tone.inputs[0])
        nt.links.new(pb.inputs["Base Color"].links[0].from_socket, tone.inputs[6])
        tone.inputs[7].default_value = (*hexc("#070707"), 1)
        nt.links.new(tone.outputs[2], pb.inputs["Base Color"])
        for socket, value in (("Metallic", 0.0), ("Roughness", 0.5)):
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "FLOAT"
            nt.links.new(kept.outputs[0], mix.inputs[0])
            nt.links.new(pb.inputs[socket].links[0].from_socket, mix.inputs[2])
            mix.inputs[3].default_value = value
            nt.links.new(mix.outputs[0], pb.inputs[socket])

    stained(lead, 0.0012, 0.68, 0.98)
    stained(ink, 0.0003, 0.5, 0.82, inside=True)
    cinnabar = material("cinnabar", hexc("#980F07"), rough=0.42, **{"Specular IOR Level": 0.16})   # the strip must not wash it pink
    chase = material("chase", hexc("#2a2826"), metallic=1.0, rough=0.55)
    wood = material("wood", hexc("#1a140e"), rough=0.7)

    bm = bmesh.new()                                       # one bevelled body shared by every piece of type
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * body, v.co.y * body, (v.co.z + 0.5) * high))
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=0.00025, segments=2, affect="EDGES", profile=0.5)
    shank = bpy.data.meshes.new("shank")
    bm.to_mesh(shank)
    bm.free()
    for poly in shank.polygons:
        poly.use_smooth = True
    shank.materials.append(lead)
    faces = {}

    no_glyph = len(text_mesh("\U000F0001", font, 0.0084, 0.0005, 0.00004).vertices)    # what a missing character looks like
    # The Black face lacks a few characters the Bold one has. Blender opens only a collection's first face, so hand it a copy
    # whose first entry points at Bold; a heavier bevel brings its strokes up to the weight of the rest.
    raw = bytearray(open(SONG, "rb").read())
    raw[12:16] = raw[16:20]
    bold_path = os.path.join(bpy.app.tempdir, "songti_bold.ttc")
    with open(bold_path, "wb") as fh:
        fh.write(raw)
    del raw
    bold = bpy.data.fonts.load(bold_path)

    def face(ch):
        """The mirrored, raised face for one character: lead walls, inked top."""
        if ch not in faces:
            me = text_mesh(ch, font, 0.0084, 0.0005, 0.00004)
            if me and len(me.vertices) == no_glyph:
                me = text_mesh(ch, bold, 0.0102, 0.0005, 0.00009)   # the Bold face also draws it small
                if me and len(me.vertices) == len(text_mesh("\U000F0001", bold, 0.0102, 0.0005, 0.00009).vertices):
                    me = None                                  # the font's empty box must never reach the screen
            if me:
                me.materials.append(lead)
                me.materials.append(ink)
                for poly in me.polygons:
                    poly.material_index = 1 if poly.normal.z > 0.6 else 0
            faces[ch] = me
        return faces[ch]

    forme = empty("forme", (0, 0, 0))
    forme.rotation_euler.z = math.radians(-24)
    # The column stands empty where the text ends; 315 = 21 x 15 and a column is 30, so the 21 columns of about 13 rows the
    # last frame sees never show the same run twice.
    text = PASSAGE + " " * (315 - len(PASSAGE))
    start = text.index("活") - (cols // 2 * rows + rows // 2)      # so that 活 lands in the middle of the forme
    hero = None
    missing = set()
    printed = []
    for c in range(cols):                                   # mirrored: the first column sits on the left
        for r in range(rows):
            x = (c - (cols - 1) / 2) * pitch
            y = ((rows - 1) / 2 - r) * pitch
            ch = text[(start + c * rows + r) % len(text)]
            spacer = ch == " "
            lift = -0.004 if spacer else rnd.uniform(0.0, 0.0004)   # spacers sit lower and carry no face
            blk = instance(f"type{c}_{r}", shank, (x, y, lift))
            blk.parent = forme
            blk.rotation_euler.z = math.radians(rnd.uniform(-0.4, 0.4))
            if spacer:
                continue
            if face(ch):
                f = instance(f"face{c}_{r}", face(ch), (0, 0, high + 0.00025))
                f.scale.x = -1                                 # type is cut backwards
                f.parent = blk
                bite = 1.0 - 0.07 * (1.0 - lift / 0.0004)      # the tallest pieces print hardest
                printed.append((x, y, ch, blk.rotation_euler.z, bite, c == cols // 2 and r == rows // 2 and ch == "活"))
            else:
                missing.add(ch)
            if ch == "活" and c == cols // 2 and r == rows // 2:
                hero = blk
                f.data = f.data.copy()                         # the one piece with fresh ink on it
                f.data.materials[1] = cinnabar
    print("no outline for:", "".join(sorted(missing)), flush=True)
    cube("table", (2.0, 2.0, 0.02), (0, 0, -0.01), wood)
    rot = forme.rotation_euler.to_matrix()
    hp = rot @ hero.location
    tip = high + 0.0005

    # the roller's front edge, in forme coordinates, is what wets the faces
    ax = rot @ Vector((1, 0, 0))
    nt = ink.node_tree
    pb = nt.nodes["Principled BSDF"]
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = tuple(ax)
    nt.links.new(nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"], dot.inputs[0])
    front = nt.nodes.new("ShaderNodeValue")
    behind = nt.nodes.new("ShaderNodeMath")
    behind.operation = "LESS_THAN"
    nt.links.new(dot.outputs["Value"], behind.inputs[0])
    nt.links.new(front.outputs[0], behind.inputs[1])
    tone = nt.nodes.new("ShaderNodeMix")
    tone.data_type = "RGBA"
    nt.links.new(behind.outputs[0], tone.inputs[0])
    nt.links.new(pb.inputs["Base Color"].links[0].from_socket, tone.inputs[6])
    tone.inputs[7].default_value = (*hexc("#050506"), 1)
    nt.links.new(tone.outputs[2], pb.inputs["Base Color"])
    level = nt.nodes.new("ShaderNodeMath")
    level.operation = "SUBTRACT"
    level.inputs[0].default_value = 0.5
    scaled = nt.nodes.new("ShaderNodeMath")
    scaled.operation = "MULTIPLY"
    scaled.inputs[1].default_value = 0.34
    nt.links.new(behind.outputs[0], scaled.inputs[0])
    nt.links.new(scaled.outputs[0], level.inputs[1])
    nt.links.new(level.outputs[0], pb.inputs["Specular IOR Level"])    # black ink in a raking light must stay black
    for socket, wet_value in (("Metallic", 0.0), ("Roughness", 0.26)):
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "FLOAT"
        nt.links.new(behind.outputs[0], mix.inputs[0])
        nt.links.new(pb.inputs[socket].links[0].from_socket, mix.inputs[2])
        mix.inputs[3].default_value = wet_value
        nt.links.new(mix.outputs[0], pb.inputs[socket])

    def rod(name, radius, length, loc, mat):
        """A cylinder along the forme's y axis."""
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=128, radius1=radius, radius2=radius, depth=length)
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, "X"))
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        for poly in me.polygons:
            poly.use_smooth = True
        me.materials.append(mat)
        o = link(bpy.data.objects.new(name, me))
        o.location = loc
        o.parent = forme
        return o

    tack = material("tack", hexc("#050506"), rough=0.14)        # the orange-peel skin of tacky ink
    tn = tack.node_tree
    nz = tn.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 240.0
    nz.inputs["Detail"].default_value = 4
    bump = tn.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.0006
    tn.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    tn.links.new(bump.outputs["Normal"], tn.nodes["Principled BSDF"].inputs["Normal"])
    radius = 0.018
    roller = rod("roller", radius, 0.42, (-0.3, 0, tip + radius), tack)

    # the tympan: paper in a wooden frame, hinged along the forme's right edge, its underside at the hinge line
    x0 = cols * pitch / 2 + 0.012
    y0 = rows * pitch / 2 + 0.015
    zh = high + 0.0012
    hinge = empty("hinge", (x0, 0, zh))
    hinge.parent = forme
    paper = material("paper", hexc("#EFE8D8"), rough=0.86)
    pn = paper.node_tree
    nz = pn.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 1400.0
    nz.inputs["Detail"].default_value = 8
    bump = pn.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.2
    bump.inputs["Distance"].default_value = 0.0002
    pn.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    pn.links.new(bump.outputs["Normal"], pn.nodes["Principled BSDF"].inputs["Normal"])
    sheet = cube("sheet", (2 * x0, 2 * y0, 0.002), (-x0, 0, 0.001), paper)
    sheet.parent = hinge
    bar, lift = 0.016, 0.005
    for name, size, loc in (("t0", (bar, 2 * y0 + 2 * bar, 0.008), (-2 * x0 - bar / 2, 0, lift)),
                            ("t1", (2 * x0, bar, 0.008), (-x0, y0 + bar / 2, lift)),
                            ("t2", (2 * x0, bar, 0.008), (-x0, -y0 - bar / 2, lift))):
        cube(name, size, loc, wood, bevel=0.001).parent = hinge
    brass = material("brass", hexc("#B08A45"), metallic=1.0, rough=0.3)
    for y in (-0.13, 0.0, 0.13):
        rod(f"barrel{y}", 0.0048, 0.05, (x0, y, zh + 0.002), brass)
    bed_top = zh - 0.0095
    bed = cube("bed", (0.44, 0.46, bed_top), (x0 + 0.222, 0, bed_top / 2), wood)
    bed.parent = forme

    # the print: the same mirrored faces rendered flat from above once, each with how hard it bit, then put through what a
    # press does to paper. Everything fine (the torn edge, the pores) is left to the shader, so no zoom finds a pixel.
    ppc = 180                                                  # mask pixels per piece of type
    mask = bpy.data.scenes.new("mask")
    mask.render.engine = "CYCLES"
    mask.cycles.device = "GPU"
    mask.cycles.samples, mask.cycles.use_adaptive_sampling, mask.cycles.use_denoising = 16, False, False
    mask.cycles.max_bounces = 0
    mask.cycles.pixel_filter_type = "BOX"
    mask.view_settings.view_transform = "Raw"
    mask.render.resolution_x, mask.render.resolution_y = round(2 * x0 / pitch * ppc), round(2 * y0 / pitch * ppc)
    mask.render.resolution_percentage = 100
    mask.render.image_settings.file_format = "OPEN_EXR"
    mask.render.filepath = os.path.join(bpy.app.tempdir, "letterpress_mask.exr")
    flat = bpy.data.materials.new("flat")
    flat.use_nodes = True
    nt = flat.node_tree
    nt.nodes.clear()
    glow = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(nt.nodes.new("ShaderNodeObjectInfo").outputs["Color"], glow.inputs["Color"])
    nt.links.new(glow.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    print_of_hero = None
    for x, y, ch, turn, bite, is_hero in printed:
        g = bpy.data.objects.new("glyph", face(ch))
        mask.collection.objects.link(g)
        g.location = (x - x0, y, 0)
        g.rotation_euler.z = turn
        g.scale.x = -1
        g.color = (1.0, float(is_hero), bite, 1.0)
        for slot in g.material_slots:
            slot.link = "OBJECT"
            slot.material = flat
        if is_hero:
            print_of_hero = empty("print_of_hero", (x - x0, y, 0))
            print_of_hero.parent = hinge
    above = bpy.data.objects.new("above", bpy.data.cameras.new("above"))
    above.data.type = "ORTHO"
    above.data.ortho_scale = 2 * max(x0, y0)
    above.location = (-x0, 0, 1)
    mask.collection.objects.link(above)
    mask.camera = above
    bpy.ops.render.render(write_still=True, scene=mask.name)
    shot = bpy.data.images.load(mask.render.filepath)
    w, h = shot.size
    px = np.empty(w * h * 4, np.float32)
    shot.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)

    def soft(a, sigma):
        """Three box passes each way: as good as a Gaussian of sigma metres."""
        r = max(1, round(sigma * ppc / pitch))
        for _ in range(3):
            for axis in (0, 1):
                c = np.cumsum(np.pad(a, [(r + 1, r) if i == axis else (0, 0) for i in (0, 1)], mode="edge"), axis=axis, dtype=np.float64)
                hi, lo = [slice(None)] * 2, [slice(None)] * 2
                hi[axis], lo[axis] = slice(2 * r + 1, None), slice(None, -(2 * r + 1))
                a = ((c[tuple(hi)] - c[tuple(lo)]) / (2 * r + 1)).astype(np.float32)
        return a

    cover = px[..., 0]
    heart = np.clip((soft(cover, 0.00032) - 0.7) / 0.27, 0, 1)   # the press squeezes ink out to a stroke's edge, thinner at its heart
    bitten = np.clip(soft(px[..., 2], 0.00025) / np.maximum(soft(cover, 0.00025), 0.02), 0, 1)
    sunk = np.clip(soft(cover, 0.0001) * 1.25, 0, 1)
    lip = soft(cover, 0.0008)                                  # paper pushed aside stands a little proud round the strokes
    plate = np.dstack((soft(cover, 0.00011),                   # R where the ink reaches, before the paper's tooth
                       0.5 + 0.5 * (0.1 * lip - sunk),          # G height, 0.5 being untouched paper
                       bitten * (1 - 0.05 * heart),             # B how much ink
                       soft(px[..., 1], 0.00011)))              # A cinnabar
    sheet_print = bpy.data.images.new("sheet_print", w, h, alpha=True, float_buffer=True)
    sheet_print.colorspace_settings.name = "Non-Color"
    sheet_print.alpha_mode = "CHANNEL_PACKED"
    sheet_print.pixels.foreach_set(plate.ravel())
    sheet_print.update()
    bpy.data.images.remove(shot)
    bpy.data.scenes.remove(mask)
    del px, cover, heart, bitten, sunk, lip, plate

    inked = paper.copy()
    nt = inked.node_tree
    pb = nt.nodes["Principled BSDF"]

    def calc(kind, a, b):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = kind
        for sock, v in zip(n.inputs, (a, b)):
            if isinstance(v, float):
                sock.default_value = v
            else:
                nt.links.new(v, sock)
        return n.outputs[0]

    def smooth(v, lo, hi):
        n = nt.nodes.new("ShaderNodeMapRange")
        n.interpolation_type = "SMOOTHSTEP"
        n.inputs["From Min"].default_value, n.inputs["From Max"].default_value = lo, hi
        nt.links.new(v, n.inputs["Value"])
        return n.outputs[0]

    def grain(scale, detail, rough):
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
        return n.outputs["Fac"]

    tc = nt.nodes.new("ShaderNodeTexCoord")
    tx = nt.nodes.new("ShaderNodeTexImage")
    tx.image = sheet_print
    nt.links.new(tc.outputs["UV"], tx.inputs["Vector"])
    chan = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(tx.outputs["Color"], chan.inputs[0])
    struck = nt.nodes.new("ShaderNodeValue")                   # nothing on the paper until the press lands
    edge = calc("ADD", chan.outputs[0], calc("MULTIPLY", calc("SUBTRACT", grain(3000.0, 5.0, 0.62), 0.5), 0.7))
    pores = smooth(grain(9000.0, 2.0, 0.5), 0.6, 0.7)
    amount = calc("MULTIPLY", smooth(edge, 0.44, 0.58), calc("SUBTRACT", 1.0, calc("MULTIPLY", pores, 0.25)))
    amount = calc("MULTIPLY", amount, struck.outputs[0])

    def mix_colour(fac, a, b):
        n = nt.nodes.new("ShaderNodeMix")
        n.data_type = "RGBA"
        nt.links.new(fac, n.inputs[0])
        for sock, v in ((n.inputs[6], a), (n.inputs[7], b)):
            if isinstance(v, str):
                sock.default_value = (*hexc(v), 1)
            else:
                nt.links.new(v, sock)
        return n.outputs[2]

    # the amount of ink changes the black, not how much paper shows through it: that would turn every stroke grey
    black = mix_colour(smooth(chan.outputs[2], 0.88, 1.0), "#2a231d", "#090706")
    ink_colour = mix_colour(smooth(tx.outputs["Alpha"], 0.35, 0.6), black, "#A5130A")
    sheet_colour = mix_colour(smooth(grain(700.0, 6.0, 0.6), 0.35, 0.65), "#EFE7D6", "#F6F0E2")
    nt.links.new(mix_colour(amount, sheet_colour, ink_colour), pb.inputs["Base Color"])
    sheen = nt.nodes.new("ShaderNodeMix")
    sheen.data_type = "FLOAT"
    nt.links.new(amount, sheen.inputs[0])
    sheen.inputs[2].default_value, sheen.inputs[3].default_value = 0.86, 0.55
    nt.links.new(sheen.outputs[0], pb.inputs["Roughness"])
    gloss = nt.nodes.new("ShaderNodeMix")
    gloss.data_type = "FLOAT"
    nt.links.new(amount, gloss.inputs[0])
    gloss.inputs[2].default_value, gloss.inputs[3].default_value = 0.5, 0.22
    nt.links.new(gloss.outputs[0], pb.inputs["Specular IOR Level"])
    sink = nt.nodes.new("ShaderNodeBump")
    sink.inputs["Distance"].default_value = 0.00056            # 0.28 mm, the height map runs over half of its range
    nt.links.new(calc("MULTIPLY", calc("SUBTRACT", chan.outputs[1], 0.5), struck.outputs[0]), sink.inputs["Height"])
    nt.links.new(pb.inputs["Normal"].links[0].from_socket, sink.inputs["Normal"])
    nt.links.new(sink.outputs["Normal"], pb.inputs["Normal"])
    me = bpy.data.meshes.new("page")
    me.from_pydata([(-2 * x0, -y0, 0), (-2 * x0, y0, 0), (0, y0, 0), (0, -y0, 0)], [], [(0, 1, 2, 3)])
    me.uv_layers.new()
    for i, uv in enumerate(((0, 0), (0, 1), (1, 1), (1, 0))):
        me.uv_layers[0].data[i].uv = uv
    me.materials.append(inked)
    page = link(bpy.data.objects.new("page", me))              # a skin on the sheet's underside, facing the forme
    page.location = (0, 0, -0.00002)
    page.parent = hinge

    SLAM, OPEN, LAND = 84, 96, 140
    for f in range(1, N + 1):
        u = min(1.0, max(0.0, (f - 12) / 54))
        rx = -0.3 + 0.6 * u
        roller.hide_render = f > 66                            # rolled clear before the tympan comes down
        roller.keyframe_insert("hide_render", frame=f)
        roller.location.x = rx
        roller.rotation_euler.y = rx / radius
        roller.keyframe_insert("location", index=0, frame=f)
        roller.keyframe_insert("rotation_euler", index=1, frame=f)
        front.outputs[0].default_value = rx
        front.outputs[0].keyframe_insert("default_value", frame=f)
        if f < 62:
            th = 115.0
        elif f < SLAM:
            th = 115.0 * (1 - ((f - 62) / (SLAM - 62)) ** 2.6)
        elif f < OPEN:
            th = 1.4 * abs(math.sin(math.pi * (f - SLAM) / 5)) * math.exp(-(f - SLAM) / 2.5)
        elif f < LAND:
            w = (f - OPEN) / (LAND - OPEN)
            th = 180.0 * w * w * (3 - 2 * w)
        else:
            th = 180.0 - 1.1 * abs(math.sin(math.pi * (f - LAND) / 5)) * math.exp(-(f - LAND) / 2.5)
        hinge.rotation_euler.y = math.radians(th)
        hinge.keyframe_insert("rotation_euler", index=1, frame=f)
        struck.outputs[0].default_value = 0.0 if f < SLAM else 1.0
        struck.outputs[0].keyframe_insert("default_value", frame=f)

    world_flat(hexc("#0c0a08"), (0.03, 0.027, 0.024))
    el = math.radians(36)
    strip = softbox("strip", (hp.x + 0.02, hp.y + math.cos(el) * 0.34, tip + math.sin(el) * 0.34), (hp.x, hp.y, tip), 1.0, 0.24, 13.0,
                    (1.0, 0.87, 0.7))                          # lies in the mirror direction of the camera's view
    sky = softbox("sky", (hp.x + 0.1, hp.y + 0.08, 0.5), (hp.x, hp.y, tip), 0.7, 0.45, 3.0, (1.0, 0.92, 0.82))
    warm = area("warm", (hp.x - 0.24, hp.y + 0.08, 0.2), (hp.x, hp.y, tip), 0.2, 11.0, (1.0, 0.72, 0.45), "RECTANGLE", 0.05)
    cool = area("cool", (hp.x + 0.26, hp.y - 0.16, 0.16), (hp.x, hp.y, tip), 0.25, 3.5, (0.72, 0.84, 1.0))
    read_at = rot @ Vector((2 * x0 - hero.location.x, hero.location.y, zh))   # where 活 ends up, reading true
    area("page", read_at + Vector((-0.12, 0.05, 0.75)), read_at, 0.9, 11.0, (1.0, 0.9, 0.76), "RECTANGLE", 0.6)
    # The forme's lights bow out as the page comes up: an area light throws nothing behind its own plane, and cool hangs right
    # over the page, so its plane would cut the page in two. One far, low, even light takes over and shows up the impression.
    across, up_page = rot @ Vector((1, 0, 0)), rot @ Vector((0, 1, 0))
    rake_el = math.radians(26)
    rake = area("rake", read_at + (-0.55 * across + 0.83 * up_page) * 1.5 * math.cos(rake_el) + Vector((0, 0, 1.5 * math.sin(rake_el))),
                read_at, 0.7, 75.0, (1.0, 0.88, 0.72), "RECTANGLE", 0.5)
    for f, k in ((OPEN, 1.0), (118, 0.0)):
        for target, attr, full in ((warm.data, "energy", 11.0), (cool.data, "energy", 3.5),
                                   (strip.data.materials[0].node_tree.nodes["Math"].inputs[1], "default_value", 13.0),
                                   (sky.data.materials[0].node_tree.nodes["Math"].inputs[1], "default_value", 3.0)):
            setattr(target, attr, full * k)
            target.keyframe_insert(attr, frame=f)
    for f, e in ((OPEN, 0.0), (118, 75.0)):
        rake.data.energy = e
        rake.data.keyframe_insert("energy", frame=f)

    near = rot @ Vector((-0.0095 * 3, -0.0095 * 5, 0))
    p0 = Vector((hp.x + near.x, hp.y + near.y, tip))
    focus = empty("focus", p0)                                 # starts on a row nearer the lens
    key(focus, "location", [(1, p0), (30, p0), (62, Vector((hp.x, hp.y, tip)))])
    follow = focus.constraints.new("COPY_LOCATION")            # then it rides on the print of 活 through the swing
    follow.target = print_of_hero
    follow.influence = 0.0
    follow.keyframe_insert("influence", frame=SLAM)
    follow.influence = 1.0
    follow.keyframe_insert("influence", frame=OPEN)
    cam = camera((hp.x, hp.y - 0.3 * math.cos(el), tip + 0.3 * math.sin(el)), (hp.x, hp.y, tip), 100, 24, focus)
    aim_at = empty("aim", p0)
    track(cam, aim_at)
    end_el, end_d = math.radians(62), 0.54
    along = rot @ Vector((0, -1, 0))                           # at the end the camera stands square to the page
    swing_at = rot @ Vector((x0 - 0.1, 0, 0.12))
    frames = ((1, hp + Vector((-0.022, -0.36 * math.cos(el), 0.36 * math.sin(el))), p0.lerp(hp, 0.5)),
              (62, hp + Vector((0.006, -0.24 * math.cos(el), 0.24 * math.sin(el))), hp),
              (SLAM, hp + Vector((0.006, -0.26 * math.cos(el), 0.26 * math.sin(el))), hp),
              (OPEN, hp + Vector((-0.04, -0.34 * math.cos(el), 0.34 * math.sin(el))), hp + Vector((0.03, 0, 0))),
              (118, swing_at + Vector((-0.22, -0.44, 0.34)), swing_at),
              (LAND, read_at + along * end_d * math.cos(end_el) + Vector((0, 0, end_d * math.sin(end_el))), read_at),
              (N, read_at + along * (end_d - 0.025) * math.cos(end_el) + Vector((0, 0, (end_d - 0.025) * math.sin(end_el))), read_at))
    for f, at, target in frames:
        cam.location = at
        cam.keyframe_insert("location", frame=f)
        aim_at.location = target
        aim_at.keyframe_insert("location", frame=f)
    for f, dz in ((SLAM, 0.0), (SLAM + 1, -0.0035), (SLAM + 4, 0.0016), (SLAM + 8, 0.0)):
        aim_at.location = hp + Vector((0, 0, dz))
        aim_at.keyframe_insert("location", frame=f)
    shoot(sc, "letterpress")


# ======================================================================================
# 等轴 2.5D: a container port as a table model
# ======================================================================================
def beam(name, p0, p1, thick, mat):
    """A square bar from p0 to p1."""
    p0, p1 = Vector(p0), Vector(p1)
    o = cube(name, (thick, thick, (p1 - p0).length), (p0 + p1) / 2, mat)
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = (p1 - p0).to_track_quat("Z", "Y")
    return o


class Solid:
    """Boxes and bars merged into one mesh of one material: a whole crane costs one object."""

    def __init__(self):
        self.v, self.f = [], []

    def _hex(self, c):
        k = len(self.v)
        self.v += c
        self.f += [(k, k + 3, k + 2, k + 1), (k + 4, k + 5, k + 6, k + 7), (k, k + 1, k + 5, k + 4),
                   (k + 1, k + 2, k + 6, k + 5), (k + 2, k + 3, k + 7, k + 6), (k + 3, k, k + 4, k + 7)]

    def box(self, x0, x1, y0, y1, z0, z1):
        self._hex([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])

    def bar(self, a, b, t):
        a, b = np.array(a, float), np.array(b, float)
        d = (b - a) / np.linalg.norm(b - a)
        up = np.array((0, 0, 1.0)) if abs(d[2]) < 0.9 else np.array((1.0, 0, 0))
        s = np.cross(d, up)
        s = s / np.linalg.norm(s) * t / 2
        w = np.cross(s, d)
        w = w / np.linalg.norm(w) * t / 2
        self._hex([tuple(p + q) for p in (a, b) for q in (-s - w, s - w, s + w, -s + w)])

    def done(self, name, mat, parent=None):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        me.update()
        me.materials.append(mat)
        o = link(bpy.data.objects.new(name, me))
        o.parent = parent
        return o


def math_nodes(nt):
    """Shader arithmetic in one line: op("MULTIPLY", a, b), where each of a and b is a socket or a number."""
    def op(kind, a, b=None, clamp=False):
        n = nt.nodes.new("ShaderNodeMath")
        n.operation = kind
        n.use_clamp = clamp
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                nt.links.new(v, n.inputs[i])
        return n.outputs[0]
    return op


LABEL_CELLS, LABEL_W, LABEL_H = 12, 2048, 1474                # two columns of six walls, each 1024 x 239, and a blank strip on top


def container_labels():
    """The markings of six invented lines, drawn once into one image: a mark, a name, and the box number high in the corner."""
    import blf
    import imbuf
    fonts = "/System/Library/Fonts/Supplemental/"
    lines = (("XINGHE", "Futura.ttc", "★", "XHLU", "星河航运"), ("TIDEMARK", "Arial Black.ttf", "▲", "TDMU", "CONTAINER LINE"), ("KESTREL", "Impact.ttf", "◆", "KSTU", "SEA FREIGHT"),
             ("OSPREY", "Rockwell.ttc", "●", "OSPU", "OCEAN CARRIERS"), ("LANTERN", "DIN Condensed Bold.ttf", "◆", "LNTU", "SHIPPING CO."), ("HAIYAN", "GillSans.ttc", "★", "HYNU", "海燕航运"))
    rnd = random.Random(8)
    din = blf.load(fonts + "DIN Condensed Bold.ttf")
    ib = imbuf.new((LABEL_W, LABEL_H))

    def put(f, text, size, x, y):
        blf.size(f, size)
        blf.color(f, 1, 1, 1, 1)
        blf.position(f, x, y, 0)
        with blf.bind_imbuf(f, ib, display_name="sRGB"):
            blf.draw_buffer(f, text)
    for cell in range(LABEL_CELLS):
        name, face, mark, owner, tag = lines[cell % 6]
        x0, y0 = (cell % 2) * 1024, (cell // 2) * 239
        f = blf.load(fonts + face)
        text = mark + " " + name
        blf.size(f, 100)
        w, h = blf.dimensions(f, text)
        reach = 620 if cell < 6 else 800                      # half the boxes wear the name small and to the side, half across the wall
        size = min(100 * reach / w, 100 * 126 / h)
        put(f, text, size, x0 + 44, y0 + 66)
        put(f, tag, 34, x0 + 48, y0 + 22)
        number = "%s %06d %d" % (owner, rnd.randrange(10 ** 6), rnd.randrange(10))
        blf.size(din, 42)
        put(din, number, 42, x0 + 1024 - 40 - blf.dimensions(din, number)[0], y0 + 239 - 56)
        put(din, rnd.choice(("45G1", "42G1", "22G1", "45R1")), 32, x0 + 1024 - 104, y0 + 239 - 92)
    path = os.path.join(bpy.app.tempdir, "port_labels.png")
    imbuf.write(ib, filepath=path)
    im = bpy.data.images.load(path)
    im.colorspace_settings.name = "Non-Color"
    im.pack()
    return im


def box_material():
    """A steel box: corrugated walls with a line's name and number, doors and lock rods on the end, corner castings, grime and
    rust. Face size lives in the attribute's alpha."""
    m = bpy.data.materials.new("box")
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    p.inputs["Roughness"].default_value = 0.5
    op = math_nodes(nt)

    def mix(fac, a, b):
        n = nt.nodes.new("ShaderNodeMix")
        n.data_type = "RGBA"
        nt.links.new(fac, n.inputs[0])
        for i, c in ((6, a), (7, b)):
            if isinstance(c, tuple):
                n.inputs[i].default_value = (*c, 1)
            else:
                nt.links.new(c, n.inputs[i])
        return n.outputs[2]

    def near(x, c, w):
        return op("LESS_THAN", op("ABSOLUTE", op("SUBTRACT", x, c)), w)

    def noise_of(vector, scale, detail):
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        nt.links.new(vector, n.inputs["Vector"])
        return n.outputs["Fac"]
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "paint"
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs[0])
    u, v, fw = sep.outputs["X"], sep.outputs["Y"], attr.outputs["Alpha"]
    e = 0.0042
    inside = op("MULTIPLY", op("MULTIPLY", op("GREATER_THAN", u, e), op("LESS_THAN", u, op("SUBTRACT", fw, e))),
                op("MULTIPLY", op("GREATER_THAN", v, e), op("LESS_THAN", v, 0.07 - e)))
    ribs = op("MULTIPLY", op("ADD", op("SINE", op("MULTIPLY", u, 2 * math.pi / 0.0115)), 1.0), 0.5)
    long_side = op("GREATER_THAN", fw, 0.1)
    end = op("LESS_THAN", fw, 0.1)
    # the door end: two leaves with a seam down the middle, four lock rods, a handle on each
    leaf = op("MULTIPLY", op("ADD", op("SINE", op("MULTIPLY", u, 2 * math.pi / 0.0068)), 1.0), 0.5)
    mid = op("ABSOLUTE", op("SUBTRACT", u, 0.035))
    seam = op("MULTIPLY", end, op("LESS_THAN", mid, 0.0005))
    rods = op("ADD", near(mid, 0.0115, 0.0007), near(mid, 0.0255, 0.0007))
    grips = op("ADD", near(mid, 0.0145, 0.0035), near(mid, 0.0285, 0.0035))
    hardware = op("MULTIPLY", end, op("MAXIMUM", rods, op("MULTIPLY", near(v, 0.031, 0.001), grips)))
    tone = op("ADD", op("MULTIPLY", long_side, op("ADD", 0.72, op("MULTIPLY", ribs, 0.28))), op("MULTIPLY", end, op("ADD", 0.84, op("MULTIPLY", leaf, 0.16))))
    frame = op("ADD", 0.58, op("MULTIPLY", inside, 0.42))
    # weather: long streaks down the walls, rust low down
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    streak = nt.nodes.new("ShaderNodeMapping")
    streak.inputs["Scale"].default_value = (16.0, 16.0, 1.5)
    nt.links.new(geo.outputs["Position"], streak.inputs["Vector"])
    grime = nt.nodes.new("ShaderNodeMapRange")
    grime.inputs["From Min"].default_value, grime.inputs["From Max"].default_value = 0.34, 0.66
    grime.inputs["To Min"].default_value, grime.inputs["To Max"].default_value = 0.8, 1.06
    nt.links.new(noise_of(streak.outputs[0], 7.0, 5), grime.inputs["Value"])
    rust = nt.nodes.new("ShaderNodeMapRange")
    rust.inputs["From Min"].default_value, rust.inputs["From Max"].default_value = 0.6, 0.7
    rust.inputs["To Max"].default_value = 0.7
    nt.links.new(noise_of(geo.outputs["Position"], 70.0, 4), rust.inputs["Value"])
    low = op("SUBTRACT", 1.0, op("DIVIDE", v, 0.09), clamp=True)
    # markings, dark on the pale boxes and white on the rest
    paint_ = attr.outputs["Color"]
    lum = nt.nodes.new("ShaderNodeRGBToBW")
    nt.links.new(paint_, lum.inputs[0])
    ink = mix(op("GREATER_THAN", lum.outputs[0], 0.29), (0.86, 0.86, 0.82), (0.02, 0.022, 0.026))
    label_uv = nt.nodes.new("ShaderNodeUVMap")
    label_uv.uv_map = "label"
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = container_labels()
    tex.interpolation = "Cubic"
    tex.extension = "CLIP"
    nt.links.new(label_uv.outputs["UV"], tex.inputs["Vector"])
    # corner castings, each with its slot
    du, dv = op("MINIMUM", u, op("SUBTRACT", fw, u)), op("MINIMUM", v, op("SUBTRACT", 0.07, v))
    casting = op("MULTIPLY", op("LESS_THAN", du, 0.0058), op("LESS_THAN", dv, 0.0058))
    slot = op("MULTIPLY", near(du, 0.0029, 0.0014), near(dv, 0.0029, 0.0018))
    base = mix(op("MULTIPLY", rust.outputs[0], low), paint_, (0.13, 0.05, 0.022))
    base = mix(op("MULTIPLY", tex.outputs["Alpha"], 0.96), base, ink)
    base = mix(hardware, base, (0.34, 0.34, 0.31))
    base = mix(casting, base, (0.3, 0.3, 0.28))
    base = mix(slot, base, (0.008, 0.008, 0.008))
    gain = op("MULTIPLY", op("MULTIPLY", tone, frame), op("MULTIPLY", grime.outputs[0], op("SUBTRACT", 1.0, op("MULTIPLY", seam, 0.7))))
    scale = nt.nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    nt.links.new(base, scale.inputs[0])
    nt.links.new(gain, scale.inputs["Scale"])
    nt.links.new(scale.outputs[0], p.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.5
    bump.inputs["Distance"].default_value = 0.003
    nt.links.new(op("ADD", op("MULTIPLY", long_side, ribs), op("MULTIPLY", end, op("MULTIPLY", leaf, 0.6))), bump.inputs["Height"])
    nt.links.new(bump.outputs[0], p.inputs["Normal"])
    return m


class Boxes:
    """Every container in one mesh, each with its own paint and markings: thousands, for the cost of one object."""
    L, W, H = 0.3, 0.07, 0.07
    pick = random.Random(23)
    BLANK = [(0.5, 0.995)] * 4

    def __init__(self, mat):
        self.mat = mat
        self.v, self.f, self.uv, self.col, self.label = [], [], [], [], []

    def add(self, x, y, z, col, shade=1.0, cell=None):
        L, W, H = self.L, self.W, self.H
        x0, x1, y0, y1, z0, z1 = x - L / 2, x + L / 2, y - W / 2, y + W / 2, z, z + H
        k = len(self.v)
        self.v += [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        c = (col[0] * shade, col[1] * shade, col[2] * shade)
        cell = self.pick.randrange(LABEL_CELLS + 2) if cell is None else cell     # a few boxes are unmarked
        wall = self.BLANK
        if cell < LABEL_CELLS:
            a, b, d = (cell % 2) / 2, (cell // 2) * 239 / LABEL_H, 239 / LABEL_H
            wall = [(a, b), (a + 0.5, b), (a + 0.5, b + d), (a, b + d)]
        for q, fw in (((0, 1, 5, 4), L), ((1, 2, 6, 5), W), ((2, 3, 7, 6), L), ((3, 0, 4, 7), W), ((4, 5, 6, 7), L)):
            self.f.append(tuple(k + i for i in q))
            self.uv += [(0, 0), (fw, 0), (fw, 0.07), (0, 0.07)]
            self.col += [(*c, fw)] * 4
            self.label += wall if fw == L and q[0] != 4 else self.BLANK

    def done(self, name, parent=None):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        me.update()
        me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.array(self.uv, np.float32).ravel())
        me.uv_layers.new(name="label").data.foreach_set("uv", np.array(self.label, np.float32).ravel())
        me.color_attributes.new("paint", "FLOAT_COLOR", "CORNER").data.foreach_set("color", np.array(self.col, np.float32).ravel())
        me.materials.append(self.mat)
        o = link(bpy.data.objects.new(name, me))
        o.parent = parent
        return o


def sea_material():
    """Swell and ripples, with the sun's glitter gathered into one broad road across the harbour. An orthographic camera would
    never form that road from real highlights, so it is laid down by position, the way the eye expects it."""
    m = bpy.data.materials.new("sea")
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    p.inputs["Roughness"].default_value = 0.2
    p.inputs["Specular IOR Level"].default_value = 0.6
    op = math_nodes(nt)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    turn = nt.nodes.new("ShaderNodeMapping")                  # ripples lie across the line of sight
    turn.inputs["Rotation"].default_value = (0, 0, -math.pi / 4)
    nt.links.new(tc.outputs["Object"], turn.inputs["Vector"])
    scale = nt.nodes.new("ShaderNodeMapping")
    scale.inputs["Scale"].default_value = (1.0, 3.0, 1.0)
    nt.links.new(turn.outputs[0], scale.inputs["Vector"])
    swell = nt.nodes.new("ShaderNodeTexNoise")
    swell.noise_dimensions = "4D"
    swell.inputs["Scale"].default_value = 9.0
    swell.inputs["Detail"].default_value = 5
    nt.links.new(scale.outputs[0], swell.inputs["Vector"])
    ripple = nt.nodes.new("ShaderNodeBump")
    ripple.inputs["Strength"].default_value = 0.5
    ripple.inputs["Distance"].default_value = 0.03
    nt.links.new(swell.outputs["Fac"], ripple.inputs["Height"])
    nt.links.new(ripple.outputs[0], p.inputs["Normal"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    r = ramp.color_ramp
    r.elements[0].position, r.elements[0].color = 0.36, (*hexc("#063D54"), 1)
    r.elements[1].position, r.elements[1].color = 0.66, (*hexc("#0E7F8F"), 1)
    nt.links.new(swell.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], p.inputs["Base Color"])
    # the road: how far this point lies from a line running down the screen, leaning a little to the right
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    across = op("MULTIPLY", op("ADD", op("MULTIPLY", sep.outputs["X"], 0.75), op("MULTIPLY", sep.outputs["Y"], 1.25)), 0.7071)
    q = op("DIVIDE", op("ADD", op("ADD", across, 2.85), op("MULTIPLY", op("SUBTRACT", swell.outputs["Fac"], 0.5), 1.1)), 0.72)
    road = op("EXPONENT", op("MULTIPLY", op("MULTIPLY", q, q), -1.0))
    glint = nt.nodes.new("ShaderNodeTexNoise")
    glint.noise_dimensions = "4D"
    glint.inputs["Scale"].default_value = 48.0
    glint.inputs["Detail"].default_value = 2
    nt.links.new(scale.outputs[0], glint.inputs["Vector"])
    spark = nt.nodes.new("ShaderNodeMapRange")
    spark.inputs["From Min"].default_value, spark.inputs["From Max"].default_value = 0.76, 0.8
    spark.inputs["To Max"].default_value = 2.6
    nt.links.new(op("ADD", glint.outputs["Fac"], op("MULTIPLY", road, 0.15)), spark.inputs["Value"])
    p.inputs["Emission Color"].default_value = (*hexc("#FFE0B4"), 1)
    glow = op("ADD", op("MULTIPLY", spark.outputs[0], op("GREATER_THAN", road, 0.04)), op("MULTIPLY", road, 0.16))
    nt.links.new(glow, p.inputs["Emission Strength"])
    for node, w in ((swell, 1.4), (glint, 6.0)):              # glitter twinkles; swell only breathes
        node.inputs["W"].default_value = 0.0
        node.inputs["W"].keyframe_insert("default_value", frame=1)
        node.inputs["W"].default_value = w
        node.inputs["W"].keyframe_insert("default_value", frame=N)
    action = nt.animation_data.action
    for fc in all_fcurves(action):
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"
    return m


def port():
    """A container port as a sunlit table model, at true isometric angles through an orthographic camera. It opens tight
    on one orange box hanging from a crane, follows it ashore on to a truck, and backs off until the whole port is there."""
    sc = reset(96, 6, 3)
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    rnd = random.Random(4)
    mat = {k: material(k, hexc(v), rough=r, metallic=me) for k, v, r, me in (
        ("asphalt", "#4C5054", 0.75, 0), ("apron", "#8C8A84", 0.8, 0), ("quay", "#A9A69C", 0.8, 0), ("line", "#F2F0E8", 0.6, 0), ("amber", "#F2B01E", 0.5, 0),
        ("crane", "#14508F", 0.42, 0.1), ("trim", "#EDEBE4", 0.5, 0), ("dark", "#1E2226", 0.6, 0), ("steel", "#70777C", 0.4, 0.5), ("rail", "#6A6E72", 0.4, 0.6),
        ("hull", "#101E33", 0.32, 0.1), ("boot", "#8E2A20", 0.5, 0), ("stripe", "#EDEBE4", 0.5, 0), ("deck", "#3B4347", 0.7, 0), ("white", "#F0EEE6", 0.45, 0),
        ("glass", "#16303E", 0.12, 0.6), ("funnel", "#C93A2A", 0.5, 0), ("roof_a", "#2E7B74", 0.6, 0), ("roof_b", "#C8553D", 0.6, 0), ("concrete", "#B8B4A8", 0.8, 0),
        ("tree", "#2F6B45", 0.8, 0), ("tank", "#E8E6DE", 0.35, 0.2), ("spread", "#E8A81A", 0.45, 0.1), ("bed", "#2E3A40", 0.8, 0), ("foam", "#F4F6F2", 0.9, 0), ("lifeboat", "#F2661A", 0.4, 0))}
    box_mat = box_material()
    PAINT = [hexc(c) for c in ("#1E62B0", "#143A6E", "#2C8A55", "#AC3A28", "#E2AE2C", "#178E94", "#8E999E", "#E6E4DA", "#AE3A72", "#1E62B0", "#E2AE2C", "#2C8A55")]
    HERO = hexc("#FF4F12")
    L, Wc, Hc = Boxes.L, Boxes.W, Boxes.H

    def paint():
        return rnd.choice(PAINT), rnd.uniform(0.78, 1.08)

    edge = -0.9                                               # the quay line: land behind it, water in front
    cube("land", (20.0, 12, 0.5), (0, edge + 6, -0.25), mat["asphalt"])
    cube("seabed", (20.0, 10, 0.2), (0, edge - 5, -0.3), mat["bed"])
    cube("water", (20.0, 10, 0.02), (0, edge - 5, -0.09), sea_material())
    ground = Solid()                                          # the painted ground, all one mesh
    ground.box(-10, 10, -0.9, -0.02, 0.0, 0.003)
    ground.done("apron", mat["apron"])
    paintwork = Solid()
    for x in np.arange(-9.9, 10, 0.4):
        paintwork.box(x, x + 0.2, -0.105, -0.095, 0.003, 0.0045)
        paintwork.box(x, x + 0.2, -0.355, -0.345, 0.003, 0.0045)
    for y in (0.08, 0.84, 1.6, 2.36, 3.12, 3.88):             # lanes between the blocks
        for x in np.arange(-9.9, 10, 0.3):
            paintwork.box(x, x + 0.14, y - 0.005, y + 0.005, 0.0, 0.0015)
    paintwork.done("paint", mat["line"])
    hazard = Solid()
    for x in np.arange(-9.9, 10, 0.1):
        hazard.box(x, x + 0.05, -0.9, -0.86, 0.003, 0.0055)
    hazard.done("hazard", mat["amber"])
    steel = Solid()
    for y in (-0.76, -0.14):                                  # crane rails
        steel.box(-10, 10, y - 0.008, y + 0.008, 0.003, 0.012)
    for y in (4.42, 4.52):                                    # the railway
        steel.box(-10, 10, y - 0.007, y + 0.007, 0.007, 0.018)
    steel.done("rails", mat["rail"])
    dark = Solid()
    for x in np.arange(-9.95, 10, 0.08):
        dark.box(x, x + 0.03, 4.38, 4.56, 0.0, 0.007)
    for x in np.arange(-9.8, 10, 0.5):                         # fenders along the quay face
        dark.box(x, x + 0.08, -0.915, -0.9, -0.09, -0.01)
    dark.done("sleepers", mat["dark"])
    bollards = Solid()
    for x in np.arange(-9.7, 10, 0.5):
        bollards.box(x, x + 0.018, -0.885, -0.867, 0.003, 0.02)
    bollards.done("bollards", mat["dark"])

    yard = Boxes(box_mat)                                     # the yard
    for y0 in (0.14, 0.9, 1.66, 2.42, 3.18):
        for x0 in np.arange(-9.6, 9.6, 2.16):
            for col in range(6):
                x = x0 + col * (L + 0.018)
                for row in range(8):
                    y = y0 + row * (Wc + 0.005) + 0.04
                    if rnd.random() < 0.06:
                        continue
                    h = 0.5 + 0.5 * noise.noise(Vector((x * 0.9, y * 1.7, 3.1)))
                    for tier in range(1 + int(h * 5.4)):
                        c, s = paint()
                        yard.add(x, y, tier * Hc, c, s)
    yard.done("yard")
    # zone letters on the ground, big enough to read when the port is small
    for i, (x, y, word) in enumerate(((-2.75, 0.03, "A"), (-0.55, 0.03, "B"), (1.6, 0.03, "C"), (-4.9, 0.03, "D"), (3.75, 0.03, "E"))):
        tm = text_mesh(word, bpy.data.fonts.load(FUTURA, check_existing=True), 0.11, 0.0)
        o = link(bpy.data.objects.new("zone", tm))
        o.location = (x, y, 0.004)
        tm.materials.append(mat["line"])

    ship_y, beam_w = -2.0, 0.84                               # the ship: a real hull, flared at the bow, bow towards -x
    stations = [(-2.92, 0.0), (-2.89, 0.06), (-2.84, 0.12), (-2.77, 0.19), (-2.68, 0.26), (-2.56, 0.32), (-2.4, 0.37), (-2.2, 0.4), (-1.8, 0.42),
                (2.05, 0.42), (2.18, 0.418), (2.27, 0.405), (2.33, 0.38), (2.37, 0.345), (2.39, 0.3)]
    hv, hf = [], []
    levels = [(-0.09, 0.8), (-0.02, 0.86), (0.0, 0.88), (0.17, 1.0), (0.19, 1.0)]
    for x, b in stations:
        for z, k in levels:
            hv.append((x, ship_y - b * k, z))
        for z, k in reversed(levels):
            hv.append((x, ship_y + b * k, z))
    ring = 2 * len(levels)
    for s in range(len(stations) - 1):
        for i in range(ring - 1):
            hf.append((s * ring + i, s * ring + i + 1, (s + 1) * ring + i + 1, (s + 1) * ring + i))
    hf.append(tuple(range(ring - 1, -1, -1)))
    hf.append(tuple((len(stations) - 1) * ring + i for i in range(ring)))
    me = bpy.data.meshes.new("hull")
    me.from_pydata(hv, [], hf)
    me.update()
    for m_ in (mat["boot"], mat["stripe"], mat["hull"], mat["deck"]):
        me.materials.append(m_)
    for f, p in enumerate(me.polygons):
        z = p.center.z
        n = p.normal.z
        p.material_index = 3 if n > 0.9 else 0 if z < -0.02 else 1 if z < 0.0 else 2
    link(bpy.data.objects.new("hull", me))
    tm = text_mesh("XINGHE", bpy.data.fonts.load(FUTURA, check_existing=True), 0.15, 0.0)
    o = link(bpy.data.objects.new("name", tm))
    tm.materials.append(mat["line"])
    o.rotation_euler = (math.pi / 2, 0, 0)
    o.location = (-1.1, ship_y - beam_w / 2 - 0.0035, 0.09)
    deck = 0.19
    # the accommodation block, seen mostly from astern: decks stepped back, each with its row of windows and its walkway
    house, edges, panes, x0, x1 = Solid(), Solid(), Solid(), 1.86, 2.22
    z = deck + 0.05
    house.box(x0 - 0.02, x1 + 0.02, ship_y - 0.34, ship_y + 0.34, deck, z)
    for k in range(6):
        a, b, w, roof = x0 + k * 0.004, x1 - k * 0.002, 0.32 - k * 0.012, z + 0.058
        house.box(a, b, ship_y - w, ship_y + w, z, roof - 0.006)
        edges.box(a - 0.008, b + 0.008, ship_y - w - 0.008, ship_y + w + 0.008, roof - 0.006, roof)
        edges.box(a - 0.008, b + 0.008, ship_y - w - 0.0085, ship_y - w - 0.0065, roof + 0.012, roof + 0.0145)
        edges.box(b + 0.0065, b + 0.0085, ship_y - w - 0.008, ship_y + w + 0.008, roof + 0.012, roof + 0.0145)
        for yy in np.arange(-w + 0.016, w - 0.01, 0.032):     # each window a bar through the block, so it shows at both ends
            panes.box(a - 0.001, b + 0.001, ship_y + yy - 0.0105, ship_y + yy + 0.0105, z + 0.016, z + 0.038)
        for xx in np.arange(a + 0.016, b - 0.01, 0.032):
            panes.box(xx - 0.0105, xx + 0.0105, ship_y - w - 0.001, ship_y + w + 0.001, z + 0.016, z + 0.038)
        z = roof
    # the bridge: glass all the way across, and wings reaching out past the ship's side, each held up on a bracket
    fore, back, core_back, span = 1.89, 1.99, 2.12, 0.46
    house.box(fore, core_back, ship_y - 0.2, ship_y + 0.2, z, z + 0.056)
    house.box(fore, back, ship_y - span, ship_y + span, z, z + 0.056)
    edges.box(fore - 0.006, back + 0.006, ship_y - span - 0.006, ship_y + span + 0.006, z - 0.006, z)
    edges.box(fore - 0.008, back + 0.008, ship_y - span - 0.008, ship_y + span + 0.008, z + 0.056, z + 0.063)
    edges.box(fore - 0.008, core_back + 0.008, ship_y - 0.208, ship_y + 0.208, z + 0.056, z + 0.063)
    g0, g1 = z + 0.018, z + 0.046
    panes.box(fore - 0.001, fore + 0.002, ship_y - span + 0.004, ship_y + span - 0.004, g0, g1)
    panes.box(fore + 0.006, back - 0.006, ship_y - span - 0.001, ship_y + span + 0.001, g0, g1)
    panes.box(core_back - 0.002, core_back + 0.001, ship_y - 0.19, ship_y + 0.19, g0, g1)
    for side in (-1, 1):
        panes.box(back - 0.002, back + 0.001, ship_y + side * 0.33 - 0.125, ship_y + side * 0.33 + 0.125, g0, g1)
        edges.bar((1.94, ship_y + side * 0.28, z - 0.11), (1.94, ship_y + side * 0.45, z - 0.005), 0.012)
        edges.bar((1.94, ship_y + side * 0.28, z - 0.11), (1.94, ship_y + side * 0.28, z - 0.005), 0.012)
        for yy in np.arange(0.215, 0.45, 0.029):              # mullions
            edges.box(back + 0.0005, back + 0.002, ship_y + side * yy - 0.0013, ship_y + side * yy + 0.0013, g0, g1)
    for yy in np.arange(-0.18, 0.185, 0.029):
        edges.box(core_back + 0.0005, core_back + 0.002, ship_y + yy - 0.0013, ship_y + yy + 0.0013, g0, g1)
    # lifeboats in their davits, halfway up each side
    for side in (-1, 1):
        y_ = ship_y + side * 0.365
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1.0, location=(2.03, y_, deck + 0.175), scale=(0.07, 0.022, 0.021))
        bpy.ops.object.shade_smooth()
        bpy.context.object.data.materials.append(mat["lifeboat"])
        for xx in (1.98, 2.08):
            edges.bar((xx, ship_y + side * 0.29, deck + 0.21), (xx, y_, deck + 0.225), 0.006)
            edges.bar((xx, y_, deck + 0.225), (xx, y_, deck + 0.195), 0.003)
    house.done("house", mat["white"])
    edges.done("walkways", mat["trim"])
    panes.done("windows", mat["glass"])
    # the signal mast, with radar scanners turning on it
    bridge_roof = z + 0.063
    mast = Solid()
    mast.box(1.95, 1.975, ship_y - 0.012, ship_y + 0.012, bridge_roof, bridge_roof + 0.15)
    mast.box(1.945, 1.98, ship_y - 0.1, ship_y + 0.1, bridge_roof + 0.095, bridge_roof + 0.101)
    for yy, top_ in ((-0.09, 0.2), (0.09, 0.18), (-0.04, 0.19), (0.05, 0.17)):
        mast.box(1.961, 1.964, ship_y + yy - 0.0015, ship_y + yy + 0.0015, bridge_roof + 0.095, bridge_roof + top_)
    mast.box(2.05, 2.066, ship_y - 0.008, ship_y + 0.008, bridge_roof, bridge_roof + 0.05)
    mast.done("mast", mat["trim"])
    for xx, yy, zz, half_, turns in ((1.9625, 0.0, 0.156, 0.052, 3.0), (2.058, 0.0, 0.052, 0.038, -4.0)):
        pivot = empty("radar", (xx, ship_y + yy, bridge_roof + zz))
        scan = Solid()
        scan.box(-0.004, 0.004, -half_, half_, 0.0, 0.007)
        scan.done("scanner", mat["white"], pivot)
        key(pivot, "rotation_euler", [(1, (0, 0, 0)), (N, (0, 0, 2 * math.pi * turns))], "LINEAR")
    # the funnel: a raked, rounded casing in the line's colours, sooty at the top, with twin exhausts
    rungs = ((0.0, "funnel"), (0.33, "funnel"), (0.62, "trim"), (0.8, "funnel"), (0.9, "dark"), (1.0, "dark"))
    fv, ff = [], []
    high, ring_n = 0.6, 28

    def funnel_at(u):
        return 2.285 + 0.03 * u, 0.062 - 0.012 * u, 0.098 - 0.02 * u
    for i, (u, _) in enumerate(rungs):
        cx, a, b = funnel_at(u)
        for j in range(ring_n):
            th = 2 * math.pi * j / ring_n
            c_, s_ = math.cos(th), math.sin(th)
            fv.append((cx + a * math.copysign(abs(c_) ** 0.5, c_), ship_y + b * math.copysign(abs(s_) ** 0.5, s_), deck + high * u))
    for i in range(len(rungs) - 1):
        for j in range(ring_n):
            ff.append((i * ring_n + j, i * ring_n + (j + 1) % ring_n, (i + 1) * ring_n + (j + 1) % ring_n, (i + 1) * ring_n + j))
    ff.append(tuple((len(rungs) - 1) * ring_n + j for j in range(ring_n)))
    me = bpy.data.meshes.new("funnel")
    me.from_pydata(fv, [], ff)
    me.update()
    for name_ in ("funnel", "trim", "dark"):
        me.materials.append(mat[name_])
    for f_, poly in enumerate(me.polygons):
        poly.use_smooth = True
        poly.material_index = ("funnel", "trim", "dark").index(rungs[min(f_ // ring_n, len(rungs) - 2)][1])
    link(bpy.data.objects.new("funnel", me))
    pipes = Solid()
    tip_x = funnel_at(1.0)[0]
    for yy in (-0.03, 0.03):
        pipes.bar((tip_x, ship_y + yy, deck + high - 0.02), (tip_x + 0.004, ship_y + yy, deck + high + 0.05), 0.022)
    pipes.done("exhausts", mat["steel"])
    # the line's mark on the funnel's white band
    cx, a, b = funnel_at(0.71)
    heiti = bpy.data.fonts.load("/System/Library/Fonts/STHeiti Medium.ttc", check_existing=True)
    for where, turn in (((cx + a + 0.0015, ship_y, deck + high * 0.71), (math.pi / 2, 0, math.pi / 2)), ((cx, ship_y - b - 0.0015, deck + high * 0.71), (math.pi / 2, 0, 0))):
        tm = text_mesh("星", heiti, 0.082, 0.0)
        tm.materials.append(mat["funnel"])
        o = link(bpy.data.objects.new("funnel mark", tm))
        o.location, o.rotation_euler = where, turn
    bays, b_hero = 12, 5
    sx = -2.12 + b_hero * (L + 0.03)
    stack = Boxes(box_mat)
    lash = Solid()
    for bay in range(bays):
        x = -2.12 + bay * (L + 0.03)
        if bay:
            lash.box(x - L / 2 - 0.022, x - L / 2 - 0.008, ship_y - 0.36, ship_y + 0.36, deck, deck + 0.17)
        for row in range(9):
            y = ship_y + (row - 4) * (Wc + 0.012)
            hero_col = bay == b_hero and row == 4
            tiers = 3 if hero_col else rnd.choice((2, 3, 3, 4, 4, 5))
            for tier in range(tiers):
                c, s = paint()
                stack.add(x, y, deck + tier * Hc, c, s)
    stack.done("deck load")
    lash.done("lashing", mat["steel"])

    # the ship-to-shore cranes
    rail_a, rail_b, boom_z = -0.76, -0.14, 1.32

    def sts(cx):
        g = empty("crane", (cx, 0, 0))
        blue, white = Solid(), Solid()
        for y in (rail_a, rail_b):
            for dx in (-0.2, 0.2):
                blue.bar((dx, y, 0.03), (dx, y, boom_z), 0.04)
            white.box(-0.26, 0.26, y - 0.024, y + 0.024, 0.012, 0.05)
            blue.box(-0.22, 0.22, y - 0.02, y + 0.02, boom_z - 0.32, boom_z - 0.28)
            blue.box(-0.22, 0.22, y - 0.016, y + 0.016, 0.48, 0.51)
            blue.bar((-0.2, y, 0.05), (0.2, y, 0.48), 0.012)
            blue.bar((0.2, y, 0.05), (-0.2, y, 0.48), 0.012)
        for dx in (-0.2, 0.2):
            blue.bar((dx, rail_a, 0.5), (dx, rail_b, boom_z - 0.3), 0.02)
            blue.bar((dx, rail_a, boom_z - 0.3), (dx, rail_b, 0.5), 0.02)
        # the boom: a girder, with a lattice along each side
        y0, y1 = -2.6, 0.38
        white.box(-0.055, 0.055, y0, y1, boom_z - 0.03, boom_z + 0.035)
        for dx in (-0.055, 0.055):
            blue.box(dx - 0.006, dx + 0.006, y0, y1, boom_z + 0.035, boom_z + 0.06)
            for k in range(30):
                a, b = y0 + k * 0.1, y0 + (k + 1) * 0.1
                blue.bar((dx, a, boom_z + 0.06), (dx, b, boom_z + 0.035), 0.006)
        apex = (0, -0.47, boom_z + 0.6)
        for dx in (-0.1, 0.1):
            blue.bar((dx, -0.56, boom_z), (0, -0.47, boom_z + 0.6), 0.034)
            blue.bar((dx, -0.38, boom_z), (0, -0.47, boom_z + 0.6), 0.034)
        for tip in ((0, -2.4, boom_z + 0.04), (0, -1.5, boom_z + 0.04), (0, 0.32, boom_z + 0.04)):
            white.bar(apex, tip, 0.012)
        cab = Solid()
        cab.box(-0.11, 0.11, -0.17, 0.09, boom_z + 0.04, boom_z + 0.15)
        blue.done("crane steel", mat["crane"], g)
        white.done("crane trim", mat["trim"], g)
        cab.done("machine house", mat["trim"], g)
        tro = Solid()
        tro.box(-0.08, 0.08, -0.07, 0.07, -0.05, 0.0)
        t = tro.done("trolley", mat["dark"], g)
        t.location = (0, -1.6, boom_z - 0.03)
        return g, t

    cranes = [sts(cx) for cx in (sx - 3.0, sx - 1.5, sx, sx + 1.5, sx + 3.0)]
    # the three others keep working, slowly
    for (g, t), a, b, h in ((cranes[1], -2.2, -1.1, 0), (cranes[3], -1.3, -2.35, 0.3), (cranes[0], -1.2, -0.5, 0.6), (cranes[4], -2.1, -1.2, 0.2)):
        key(t, "location", [(1, Vector((0, a, boom_z - 0.03))), (N, Vector((0, b, boom_z - 0.03)))])
        s = Solid()
        s.box(-0.16, 0.16, -0.04, 0.04, -0.012, 0.0)
        sp = s.done("spreader", mat["spread"], t)
        sp.location = (0, 0, -0.42 - h * 0.3)
        r = Solid()
        r.box(-0.003, 0.003, -0.003, 0.003, -0.42 - h * 0.3, 0)
        r.done("rope", mat["dark"], t)
    hero_crane, tro = cranes[2]
    tz = boom_z - 0.08                                        # the rope leaves the trolley here

    # the move, integrated so that the load really swings as the trolley starts and stops
    truck_y, lane_z = -0.48, 0.05 + Hc + 0.014
    top = deck + 3 * Hc + 0.03

    def ease(a, b, f, f0, f1):
        u = min(1.0, max(0.0, (f - f0) / (f1 - f0)))
        u = u * u * u * (u * (6 * u - 15) + 10)
        return a + (b - a) * u
    theta, omega, last_v, swing = 0.0, 0.0, 0.0, {}
    ys, zs = {}, {}
    for f in range(1, N + 1):
        y = ease(ship_y, truck_y, f, 8, 66)
        z = ease(top + 0.1, top + 0.4, f, 1, 16) if f < 70 else ease(top + 0.4, lane_z, f, 70, 106) if f < 112 else ease(lane_z, lane_z + 0.5, f, 112, 146)
        ys[f], zs[f] = y, z
        for _ in range(8):                                    # a pendulum driven by the trolley's acceleration
            v = (ease(ship_y, truck_y, f + 0.5, 8, 66) - ease(ship_y, truck_y, f - 0.5, 8, 66)) * FPS
            acc = (v - last_v) * FPS / 8
            last_v = last_v + (v - last_v) / 8
            omega += (-15.0 * theta - 0.55 * acc * 0.42 - 1.05 * omega) / FPS / 8
            theta += omega / FPS / 8
        swing[f] = theta
    rope = Solid()
    for dx in (-0.09, 0.09):
        for dy in (-0.02, 0.02):
            rope.box(dx + dy * 0.3 - 0.0028, dx + dy * 0.3 + 0.0028, dy - 0.0028, dy + 0.0028, -1.0, 0.0)
    cable = rope.done("hoist", mat["dark"], hero_crane)
    cable.rotation_mode = "XYZ"
    sp = Solid()
    sp.box(-L / 2 - 0.012, L / 2 + 0.012, -Wc / 2 - 0.008, Wc / 2 + 0.008, 0.0, 0.012)
    sp.box(-0.1, 0.1, -0.012, 0.012, 0.012, 0.034)
    spreader = sp.done("spreader", mat["dark"], hero_crane)
    spreader.rotation_mode = "XYZ"
    hero = Boxes(box_mat)
    hero.add(0, 0, -Hc, HERO, cell=0)
    load = hero.done("hero box", spreader)
    for f in range(1, N + 1):
        y, z, th = ys[f], zs[f], swing[f]
        length = tz - z
        tro.location = (0, y, boom_z - 0.03)
        tro.keyframe_insert("location", frame=f)
        spreader.location = (0, y + length * math.sin(th), tz - length * math.cos(th))
        spreader.rotation_euler = (th * 0.6, 0, 0)
        spreader.keyframe_insert("location", frame=f)
        spreader.keyframe_insert("rotation_euler", frame=f)
        cable.location = (0, y, tz)
        cable.rotation_euler = (th, 0, 0)
        cable.scale = (1, 1, length)
        cable.keyframe_insert("location", frame=f)
        cable.keyframe_insert("rotation_euler", frame=f)
        cable.keyframe_insert("scale", frame=f)

    def truck(x, y, cargo=None, shade=1.0):
        t = empty("truck", (x, y, 0))
        body = Solid()
        body.box(-0.2, 0.17, -0.032, 0.032, 0.022, 0.036)
        body.done("chassis", mat["dark"], t)
        wh = Solid()
        for wx in (-0.16, -0.11, 0.12, 0.2):
            wh.box(wx - 0.018, wx + 0.018, -0.04, 0.04, 0.0, 0.03)
        wh.done("wheels", mat["dark"], t)
        cab = Solid()
        cab.box(0.17, 0.25, -0.036, 0.036, 0.02, 0.085)
        cab.done("cab", mat["white"], t)
        win = Solid()
        win.box(0.215, 0.2535, -0.0365, 0.0365, 0.055, 0.08)
        win.done("cab window", mat["glass"], t)
        if cargo:
            b = Boxes(box_mat)
            b.add(-0.02, 0, 0.036, cargo)
            b.done("cargo", t)
        return t

    t = truck(sx, truck_y)
    key(t, "location", [(1, Vector((sx, truck_y, 0))), (116, Vector((sx, truck_y, 0))), (N, Vector((sx + 1.5, truck_y, 0)))])
    landed = Boxes(box_mat)
    landed.add(-0.02, 0, 0.036, HERO, cell=0)
    landed = landed.done("hero on truck", t)
    landed.hide_render = True
    landed.keyframe_insert("hide_render", frame=1)
    landed.hide_render = False
    landed.keyframe_insert("hide_render", frame=108)
    load.hide_render = False
    load.keyframe_insert("hide_render", frame=1)
    load.hide_render = True
    load.keyframe_insert("hide_render", frame=108)
    for x, y, d in ((-3.6, -0.23, 1.4), (-1.0, -0.23, 1.25), (1.2, -0.23, 1.5), (4.0, -0.23, 1.1), (-5.5, -0.23, 1.3),
                    (5.6, -0.48, -1.0), (2.6, -0.48, -1.2), (-3.0, 0.08, 0.9), (0.7, 0.84, -0.8), (-1.7, 1.6, 0.8), (2.5, 2.36, -0.7), (-4.4, 3.12, 0.7), (3.7, 3.88, -0.7)):
        tr = truck(x, y, rnd.choice(PAINT))
        if d < 0:
            tr.rotation_euler = (0, 0, math.pi)
        key(tr, "location", [(1, Vector((x, y, 0))), (N, Vector((x + d, y, 0)))], "LINEAR")

    # rubber-tyred gantries straddling the stacks
    for x0, y0, dx in ((-2.0, 0.14, 0.32), (0.4, 0.9, -0.28), (2.6, 1.66, 0.3), (-3.6, 2.42, -0.26), (0.8, 3.18, 0.28)):
        g = empty("rtg", (x0, y0 + 0.31, 0))
        s = Solid()
        for dy in (-0.33, 0.33):
            for ddx in (-0.13, 0.13):
                s.bar((ddx, dy, 0.02), (ddx, dy, 0.44), 0.024)
            s.box(-0.17, 0.17, dy - 0.016, dy + 0.016, 0.0, 0.03)
            s.box(-0.14, 0.14, dy - 0.01, dy + 0.01, 0.22, 0.24)
        for ddx in (-0.13, 0.13):
            s.box(ddx - 0.016, ddx + 0.016, -0.37, 0.37, 0.44, 0.475)
        s.done("gantry", mat["spread"], g)
        c = Solid()
        c.box(-0.06, 0.06, -0.06, 0.06, 0.43, 0.49)
        c.done("gantry cab", mat["dark"], g)
        key(g, "location", [(1, Vector((x0, y0 + 0.31, 0))), (N, Vector((x0 + dx, y0 + 0.31, 0)))], "LINEAR")

    train = empty("train", (0, 4.47, 0))
    b = Boxes(box_mat)
    fl = Solid()
    for i in range(46):
        x = -i * (L + 0.035)
        fl.box(x - L / 2 - 0.012, x + L / 2 + 0.012, -0.045, 0.045, 0.022, 0.04)
        c, s = paint()
        b.add(x, 0, 0.04, c, s)
    b.done("wagons", train)
    fl.done("flats", mat["dark"], train)
    loco = Solid()
    loco.box(0.2, 0.58, -0.05, 0.05, 0.022, 0.125)
    loco.done("loco", mat["funnel"], train)
    key(train, "location", [(1, Vector((2.0, 4.47, 0))), (N, Vector((3.6, 4.47, 0)))], "LINEAR")

    # the back of the port: long sheds with sawtooth roofs, tanks, trees
    for i, x in enumerate(np.arange(-9, 9.5, 1.5)):
        a, b = x, x + 1.2
        wall = Solid()
        wall.box(a, b, 4.8, 5.8, 0, 0.16)
        wall.done("shed", mat["concrete"])
        pts = []
        faces = []
        for k in range(6):
            yy, step = 4.8 + k * (1.0 / 6), 1.0 / 6
            n = len(pts)
            pts += [(a, yy, 0.16), (b, yy, 0.16), (b, yy, 0.24), (a, yy, 0.24), (b, yy + step, 0.16), (a, yy + step, 0.16)]
            faces += [(n, n + 1, n + 2, n + 3), (n + 3, n + 2, n + 4, n + 5)]
        me = bpy.data.meshes.new("roof")
        me.from_pydata(pts, [], faces)
        me.update()
        me.materials.append(mat["roof_a" if i % 3 else "roof_b"])
        link(bpy.data.objects.new("roof", me))
    for x, y, r in ((6.4, 6.3, 0.34), (7.2, 6.3, 0.34), (6.8, 7.0, 0.34), (-6.6, 6.3, 0.3), (-5.9, 6.3, 0.3)):
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=r, depth=0.34, location=(x, y, 0.17))
        o = bpy.context.object
        bpy.ops.object.shade_smooth()
        o.data.materials.append(mat["tank"])
    for _ in range(90):
        x, y = rnd.uniform(-9.5, 9.5), rnd.uniform(5.95, 7.8)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=rnd.uniform(0.07, 0.13), location=(x, y, 0.1))
        o = bpy.context.object
        bpy.ops.object.shade_smooth()
        o.data.materials.append(mat["tree"])

    # masts of light, and a tug drawing its wake across the harbour
    masts = Solid()
    for x in np.arange(-9, 9.5, 1.08):
        masts.bar((x, 0.03, 0), (x, 0.03, 0.56), 0.012)
        masts.box(x - 0.05, x + 0.05, 0.015, 0.045, 0.56, 0.575)
    masts.done("masts", mat["steel"])

    def boat(x, y, heading, hull_mat, length=0.34, beam_=0.11):
        g = empty("boat", (x, y, -0.08))
        pts = [(-length / 2, -beam_ / 2), (length * 0.2, -beam_ / 2), (length / 2, 0), (length * 0.2, beam_ / 2), (-length / 2, beam_ / 2)]
        s = bmesh.new()
        f = s.faces.new([s.verts.new((a, b, 0)) for a, b in pts])
        ext = bmesh.ops.extrude_face_region(s, geom=[f])
        for v in ext["geom"]:
            if isinstance(v, bmesh.types.BMVert):
                v.co.z += 0.05
        bmesh.ops.recalc_face_normals(s, faces=s.faces)
        me = bpy.data.meshes.new("boat")
        s.to_mesh(me)
        s.free()
        me.materials.append(hull_mat)
        link(bpy.data.objects.new("boat", me)).parent = g
        c = Solid()
        c.box(-length * 0.22, length * 0.06, -beam_ * 0.32, beam_ * 0.32, 0.05, 0.09)
        c.box(-length * 0.12, length * 0.05, -beam_ * 0.24, beam_ * 0.24, 0.09, 0.122)
        c.box(-length * 0.14, length * 0.07, -beam_ * 0.27, beam_ * 0.27, 0.122, 0.127)
        c.done("wheelhouse", mat["white"], g)
        w = Solid()
        w.box(-length * 0.121, length * 0.051, -beam_ * 0.241, beam_ * 0.241, 0.099, 0.114)
        w.done("wheelhouse glass", mat["glass"], g)
        t = Solid()
        t.box(-length * 0.44, length * 0.46, -beam_ * 0.53, beam_ * 0.53, 0.0, 0.012)
        t.box(-length * 0.34, -length * 0.26, -0.012, 0.012, 0.05, 0.1)
        t.done("fender and stack", mat["dark"], g)
        g.rotation_euler = (0, 0, heading)
        return g

    def wake(g, length, width):
        """White water: a churned strip down the middle, two arms feathering out behind, a moustache at the bow. The shape
        travels with the boat; the lace is in world space, so it stays on the water where it was laid."""
        me = bpy.data.meshes.new("wake")
        me.from_pydata([(-length, -width * 1.4, 0.002), (0.22, -width * 1.4, 0.002), (0.22, width * 1.4, 0.002), (-length, width * 1.4, 0.002)], [], [(0, 1, 2, 3)])
        me.update()
        m = bpy.data.materials.new("wake")
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        op = math_nodes(nt)
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["Object"], sep.inputs[0])
        x, y = sep.outputs["X"], op("ABSOLUTE", sep.outputs["Y"])
        s = op("DIVIDE", x, -length, clamp=True)
        ahead = op("SUBTRACT", 1.0, s)
        off = op("ABSOLUTE", op("SUBTRACT", y, op("ADD", 0.05, op("MULTIPLY", width, op("POWER", s, 0.7)))))
        arms = op("SUBTRACT", 1.0, op("DIVIDE", off, op("ADD", 0.014, op("MULTIPLY", 0.085, s))), clamp=True)
        arms = op("MULTIPLY", op("MULTIPLY", arms, op("POWER", ahead, 0.8)), op("LESS_THAN", x, -0.06))
        strip = op("SUBTRACT", 1.0, op("DIVIDE", y, op("ADD", 0.036, op("MULTIPLY", 0.085, s))), clamp=True)
        strip = op("MULTIPLY", op("MULTIPLY", strip, op("SUBTRACT", 1.0, op("DIVIDE", s, 0.85), clamp=True)), op("MULTIPLY", 1.3, op("LESS_THAN", x, -0.12)))
        hull_side = op("MULTIPLY", 0.055, op("DIVIDE", op("SUBTRACT", 0.17, x), 0.102, clamp=True))
        bow = op("SUBTRACT", 1.0, op("DIVIDE", op("ABSOLUTE", op("SUBTRACT", y, op("ADD", hull_side, 0.005))), 0.013), clamp=True)
        bow = op("MULTIPLY", bow, op("MULTIPLY", op("GREATER_THAN", x, -0.2), op("LESS_THAN", x, 0.2)))
        foam = op("MAXIMUM", op("MAXIMUM", arms, strip), bow)
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        def froth(scale, detail, twist):
            n = nt.nodes.new("ShaderNodeTexNoise")
            n.inputs["Scale"].default_value = scale
            n.inputs["Detail"].default_value = detail
            n.inputs["Roughness"].default_value = 0.62
            n.inputs["Distortion"].default_value = twist
            nt.links.new(geo.outputs["Position"], n.inputs["Vector"])
            return n.outputs["Fac"]
        lace = op("ADD", op("MULTIPLY", froth(62.0, 6, 0.5), 0.7), op("MULTIPLY", froth(20.0, 3, 0.3), 0.3))
        bright = op("DIVIDE", op("SUBTRACT", op("ADD", op("MULTIPLY", foam, 1.2), op("MULTIPLY", op("SUBTRACT", lace, 0.5), 1.3)), 0.45), 0.16, clamp=True)
        # round the white water, a wider milky stain of aerated sea
        milk = op("SUBTRACT", 1.0, op("DIVIDE", y, op("ADD", 0.07, op("MULTIPLY", width, s))), clamp=True)
        milk = op("MULTIPLY", op("MULTIPLY", milk, ahead), op("LESS_THAN", x, -0.04))
        cover = op("MAXIMUM", op("MULTIPLY", bright, op("ADD", 0.6, op("MULTIPLY", foam, 0.4))), op("MULTIPLY", milk, 0.26))
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(cover, mix.inputs[0])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        white = nt.nodes.new("ShaderNodeBsdfDiffuse")
        white.inputs["Color"].default_value = (0.95, 0.97, 0.96, 1)
        lit = nt.nodes.new("ShaderNodeEmission")
        lit.inputs["Strength"].default_value = 0.22
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(white.outputs[0], add.inputs[0])
        nt.links.new(lit.outputs[0], add.inputs[1])
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(add.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
        me.materials.append(m)
        o = link(bpy.data.objects.new("wake", me))
        o.parent = g
        o.visible_shadow = False
        return o

    for x, y, a, h, spd in ((-4.2, -4.0, 2.0, 0.42, 0.95), (2.6, -4.6, -1.5, 3.3, 0.7), (-0.4, -3.15, 0.0, 0.0, 0.8)):
        g = boat(x, y, h, mat["boot"] if a > 0 else mat["amber"])
        wake(g, 1.6, 0.4)
        d = Vector((math.cos(h), math.sin(h), 0)) * spd
        key(g, "location", [(1, Vector((x, y, -0.08))), (N, Vector((x, y, -0.08)) + d)], "LINEAR")

    world_flat(hexc("#8CB2D6"), hexc("#94B8DC"), 0.62)
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy, sun.color, sun.angle = 5.2, (1.0, 0.8, 0.55), math.radians(1.2)
    so = link(bpy.data.objects.new("sun", sun))
    so.rotation_euler = (Vector((-0.74, -0.46, 0.46))).to_track_quat("Z", "Y").to_euler()

    # the camera backs off at a steady rate of scale, its centre sliding from the box to the middle of the port
    d = 20
    cam = camera((d, -d, d), (0, 0, 0), 50, ortho=2.5)
    cam.rotation_euler = (math.radians(54.736), 0, math.radians(45))
    view = Vector((1, -1, 1)) * d
    for f in range(1, N + 1):
        y, z = ys[f], zs[f]
        u = min(1.0, max(0.0, (f - 16) / 118))
        u = u * u * (3 - 2 * u)
        scale = 2.5 * (8.4 / 2.5) ** u
        box_at = Vector((sx, y, z - 0.04))
        centre = box_at * (1 - u) + Vector((0.1, 0.25, 0.12)) * u
        cam.data.ortho_scale = scale
        cam.location = centre + view
        cam.data.keyframe_insert("ortho_scale", frame=f)
        cam.keyframe_insert("location", frame=f)
    shoot(sc, "port")


# ======================================================================================
# 复古 synthwave
# ======================================================================================
def palm_mesh():
    """A palm in silhouette: a leaning trunk and a crown of drooping, feathered fronds."""
    rnd = random.Random(7)
    verts, faces = [], []
    rings = 12
    for s in range(rings + 1):
        u = s / rings
        cx, z, r = 0.1 * u * u, u, 0.024 - 0.012 * u
        for i in range(8):
            a = 2 * math.pi * i / 8
            verts.append((cx + r * math.cos(a), r * math.sin(a), z))
    for s in range(rings):
        for i in range(8):
            j = (i + 1) % 8
            faces.append((s * 8 + i, s * 8 + j, (s + 1) * 8 + j, (s + 1) * 8 + i))
    top = np.array((0.1, 0.0, 1.0))
    for k in range(13):
        az = 2 * math.pi * k / 13 + rnd.uniform(-0.1, 0.1)
        rise = rnd.uniform(0.12, 0.34)
        length = rnd.uniform(0.3, 0.42)
        d = np.array((math.cos(az), math.sin(az), 0.0))
        side = np.array((-math.sin(az), math.cos(az), 0.0))
        steps = 14
        for i in range(steps):
            u0, u1 = i / steps, (i + 1) / steps
            p0 = top + d * length * u0 + np.array((0, 0, rise * u0 - (rise + 0.26) * u0 * u0))
            p1 = top + d * length * u1 + np.array((0, 0, rise * u1 - (rise + 0.26) * u1 * u1))
            w = 0.05 * math.sin(math.pi * min(1.0, u1 + 0.1)) ** 0.7
            for sgn in (-1, 1):                               # a leaflet each side, pointing out and down
                n = len(verts)
                tip = p1 + side * sgn * w + d * 0.012 + np.array((0, 0, -0.016))
                verts += [tuple(p0), tuple(p1), tuple(tip)]
                faces.append((n, n + 1, n + 2))
    me = bpy.data.meshes.new("palm")
    me.from_pydata(verts, [], faces)
    me.update()
    return me


def synthwave():
    """Outrun: flat out down a mirror-black road between palms into a striped sun. The car pulls away with its lights
    streaking, and chrome 1984 rises off the horizon above it. The VHS treatment is laid over in style_synthwave.py."""
    sc = reset(96, 8, 6)
    sc.cycles.adaptive_threshold = 0.02
    sc.view_settings.view_transform = "Standard"           # the neon palette should come out as picked, not washed
    sc.view_settings.look = "None"
    rnd = random.Random(19)
    speed = 46.0

    def cam_y(t):
        return -6 + speed * t

    sp, nx, ny = 3.2, 44, 320
    bm = bmesh.new()
    grid = []
    for j in range(ny + 1):
        row = []
        for i in range(nx + 1):
            x, y = (i - nx / 2) * sp, j * sp - 30
            ax = abs(x)
            wall = max(0.0, min(1.0, (ax - 15) / 34)) ** 1.5
            h = wall * (14 + 26 * (noise.noise(Vector((x * 0.024, y * 0.024, 0.3))) * 0.5 + 0.5)
                        + 8 * noise.noise(Vector((x * 0.08, y * 0.08, 1.7))))
            row.append(bm.verts.new((x, y, max(0.0, h))))
        grid.append(row)
    for j in range(ny):
        for i in range(nx):
            bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
    me = bpy.data.meshes.new("terrain")
    bm.to_mesh(me)
    bm.free()
    solid = link(bpy.data.objects.new("terrain", me))
    solid.data.materials.append(material("night", hexc("#0A0314"), rough=0.3, **{"Specular IOR Level": 0.8}))
    wire = link(bpy.data.objects.new("wire", me.copy()))
    wire.data.materials.clear()
    wm = wire.modifiers.new("wire", "WIREFRAME")
    wm.thickness, wm.use_replace, wm.use_even_offset = 0.055, True, True
    wire.location.z = 0.012
    neon = bpy.data.materials.new("neon")                    # bright near, sinking into the haze
    neon.use_nodes = True
    nt = neon.node_tree
    nt.nodes.clear()
    cd = nt.nodes.new("ShaderNodeCameraData")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 6.0, 320.0
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 1.5, 0.1
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*hexc("#D640D8"), 1)
    nt.links.new(cd.outputs["View Distance"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], em.inputs["Strength"])
    nt.links.new(em.outputs["Emission"], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    wire.data.materials.append(neon)

    # the road: polished black, with the sun running down it
    road_mat = bpy.data.materials.new("road")
    road_mat.use_nodes = True
    nt = road_mat.node_tree
    p = nt.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*hexc("#05030A"), 1)
    p.inputs["Metallic"].default_value = 0.85
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.5
    nz.inputs["Detail"].default_value = 6
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["From Min"].default_value, rr.inputs["From Max"].default_value = 0.35, 0.65
    rr.inputs["To Min"].default_value, rr.inputs["To Max"].default_value = 0.02, 0.14
    nt.links.new(nz.outputs["Fac"], rr.inputs["Value"])
    nt.links.new(rr.outputs[0], p.inputs["Roughness"])
    stretch = nt.nodes.new("ShaderNodeMapping")
    stretch.inputs["Scale"].default_value = (3.0, 0.1, 1.0)
    nt.links.new(tc.outputs["Object"], stretch.inputs["Vector"])
    streak = nt.nodes.new("ShaderNodeTexNoise")
    streak.inputs["Scale"].default_value = 1.0
    streak.inputs["Detail"].default_value = 4
    nt.links.new(stretch.outputs[0], streak.inputs["Vector"])
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.07
    bp.inputs["Distance"].default_value = 0.05
    nt.links.new(streak.outputs["Fac"], bp.inputs["Height"])
    nt.links.new(bp.outputs[0], p.inputs["Normal"])
    road = Solid()
    road.box(-7.0, 7.0, -40, ny * sp, 0.0, 0.07)
    road.done("road", road_mat)
    cyan = emission("road line", hexc("#2DE2E6"), 1.6)
    lines = Solid()
    for x in (-6.8, 6.8):
        lines.box(x - 0.07, x + 0.07, -40, ny * sp, 0.07, 0.08)
    for y in np.arange(-40, ny * sp, 9.0):
        lines.box(-0.05, 0.05, y, y + 3.6, 0.07, 0.08)
    lines.done("road lines", cyan)

    # palms, line astern on both sides
    palm = palm_mesh()
    palm.materials.append(material("palm", hexc("#07020E"), rough=0.7))
    for side in (-1, 1):
        y = -24.0
        while y < 900:
            s = rnd.uniform(9, 14)
            o = link(bpy.data.objects.new("palm", palm))
            o.location = (side * rnd.uniform(9.5, 12.5), y, 0)
            o.scale = (s, s, s)
            o.rotation_euler = (0, 0, (math.pi if side < 0 else 0) + rnd.uniform(-0.6, 0.6))   # leaning a little away from the road
            y += rnd.uniform(15, 27)

    sky = empty("sky", (0, 0, 0))
    sun = bpy.data.materials.new("sun")                      # vertical gradient, sliced by gaps that drift downwards and widen
    sun.use_nodes = True
    nt = sun.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*hexc("#FF1F8E"), 1)
    ramp.color_ramp.elements[1].color = (*hexc("#FFE88A"), 1)
    ramp.color_ramp.elements.new(0.55).color = (*hexc("#FF8A3D"), 1)
    mp = nt.nodes.new("ShaderNodeMapRange")
    mp.inputs["From Min"].default_value, mp.inputs["From Max"].default_value = -1.0, 1.0
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.15
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 9.0
    drift = nt.nodes.new("ShaderNodeMath"); drift.operation = "ADD"
    drift.inputs[1].default_value = 0.0
    drift.inputs[1].keyframe_insert("default_value", frame=1)
    drift.inputs[1].default_value = 3.0
    drift.inputs[1].keyframe_insert("default_value", frame=N)
    frac = nt.nodes.new("ShaderNodeMath"); frac.operation = "FRACT"
    depth = nt.nodes.new("ShaderNodeMapRange")
    depth.inputs["From Min"].default_value, depth.inputs["From Max"].default_value = 0.42, -0.45
    depth.inputs["To Min"].default_value, depth.inputs["To Max"].default_value = 0.04, 0.66
    lt = nt.nodes.new("ShaderNodeMath"); lt.operation = "LESS_THAN"
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    nt.links.new(sep.outputs["Y"], mp.inputs["Value"])
    nt.links.new(mp.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
    nt.links.new(sep.outputs["Y"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], drift.inputs[0])
    nt.links.new(drift.outputs["Value"], frac.inputs[0])
    nt.links.new(sep.outputs["Y"], depth.inputs["Value"])
    nt.links.new(frac.outputs["Value"], lt.inputs[0])
    nt.links.new(depth.outputs["Result"], lt.inputs[1])
    nt.links.new(lt.outputs["Value"], mix.inputs["Fac"])
    nt.links.new(em.outputs["Emission"], mix.inputs[1])
    nt.links.new(tr.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    for fc in all_fcurves(nt.animation_data.action):
        for k in fc.keyframe_points:
            k.interpolation = "LINEAR"
    bpy.ops.mesh.primitive_circle_add(vertices=160, radius=1.0, fill_type="NGON", location=(0, 1300, 118), rotation=(math.radians(90), 0, 0))
    s = bpy.context.object
    s.scale = (270, 270, 270)
    s.data.materials.append(sun)
    s.parent = sky

    w = bpy.context.scene.world or bpy.data.worlds.new("w")  # sky: deep violet overhead, hot pink at the horizon
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (*hexc("#FF4D94"), 1)
    els[1].position, els[1].color = 0.32, (*hexc("#120A30"), 1)
    els.new(0.09).color = (*hexc("#6A1A78"), 1)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.95
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], nt.nodes.new("ShaderNodeOutputWorld").inputs["Surface"])
    star = emission("star", (1.0, 0.9, 1.0), 6.0)
    for _ in range(260):
        a, e = rnd.uniform(-1.1, 1.1), rnd.uniform(0.1, 0.62)
        d = 1600
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=rnd.uniform(1.1, 3.2),
                                              location=(d * math.sin(a), d * math.cos(a) * math.cos(e), d * math.sin(e)))
        bpy.context.object.data.materials.append(star)
        bpy.context.object.parent = sky

    chrome = material("chrome", (0.95, 0.95, 0.98), metallic=1.0, rough=0.14)
    rig = empty("rig", (0, 0, 0))                             # the title and everything that lights it travel with the camera
    # Blender opens only a collection's first face, so hand it a copy whose first entry points at Futura Condensed ExtraBold
    raw = bytearray(open(FUTURA, "rb").read())
    raw[12:16] = raw[28:32]
    heavy_path = os.path.join(bpy.app.tempdir, "futura_condensed_extrabold.ttc")
    with open(heavy_path, "wb") as fh:
        fh.write(raw)
    del raw
    font = bpy.data.fonts.load(heavy_path)
    title_size = opt("tsize", 5.2)
    me = text_mesh("1984", font, title_size, 0.7, 0.1)
    keyline = text_mesh("1984", font, title_size, 0.3, 0.27)
    xs, ys = [v.co.x for v in me.vertices], [v.co.y for v in me.vertices]
    tw, th = max(xs) - min(xs), max(ys) - min(ys)
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    for m in (me, keyline):                                   # centred on the figures themselves, leaning into the road
        for v in m.vertices:
            v.co.x += 0.2 * (v.co.y - cy) - cx
            v.co.y -= cy
        m.update()
    print(f"title {tw:.2f} x {th:.2f}", flush=True)
    face = bpy.data.materials.new("chrome face")              # the eighties airbrush: sky down to a hard horizon, dark ground up from gold
    face.use_nodes = True
    nt = face.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    up = nt.nodes.new("ShaderNodeMapRange")
    up.inputs["From Min"].default_value, up.inputs["From Max"].default_value = -th / 2, th / 2
    nt.links.new(sep.outputs["Y"], up.inputs["Value"])
    horizon = up.outputs[0]
    for rate, phase, amp in ((1.9, 0.0, 0.012), (4.7, 1.0, 0.006)):      # the horizon wanders a little, as a reflected skyline does
        a = nt.nodes.new("ShaderNodeMath"); a.operation = "MULTIPLY"; a.inputs[1].default_value = rate
        b = nt.nodes.new("ShaderNodeMath"); b.operation = "ADD"; b.inputs[1].default_value = phase
        c = nt.nodes.new("ShaderNodeMath"); c.operation = "SINE"
        d = nt.nodes.new("ShaderNodeMath"); d.operation = "MULTIPLY"; d.inputs[1].default_value = amp
        e = nt.nodes.new("ShaderNodeMath"); e.operation = "ADD"
        nt.links.new(sep.outputs["X"], a.inputs[0])
        nt.links.new(a.outputs[0], b.inputs[0])
        nt.links.new(b.outputs[0], c.inputs[0])
        nt.links.new(c.outputs[0], d.inputs[0])
        nt.links.new(horizon, e.inputs[0])
        nt.links.new(d.outputs[0], e.inputs[1])
        horizon = e.outputs[0]
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (*hexc("#FFD98A"), 1)
    els[1].position, els[1].color = 1.0, (*hexc("#0C2F96"), 1)
    for pos, col in ((0.13, "#F2A030"), (0.32, "#B4501A"), (0.497, "#2A0C12"), (0.503, "#FFFFFF"), (0.6, "#CFE8FF"), (0.73, "#4E9CF2"), (0.87, "#1E5CD8")):
        els.new(pos).color = (*hexc(col), 1)
    nt.links.new(horizon, ramp.inputs["Fac"])
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    me.materials.append(face)
    bevel = material("chrome bevel", (0.95, 0.95, 0.98), metallic=1.0, rough=0.14, **{"Emission Strength": 0.5})     # true metal over a lit edge: white above, pink below
    nt = bevel.node_tree
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Normal"], sep.inputs[0])
    tilt = nt.nodes.new("ShaderNodeMapRange")
    tilt.inputs["From Min"].default_value, tilt.inputs["From Max"].default_value = -0.9, 0.9
    nt.links.new(sep.outputs["Y"], tilt.inputs["Value"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (*hexc("#FF3D9A"), 1)
    els[1].position, els[1].color = 1.0, (*hexc("#FFFFFF"), 1)
    els.new(0.5).color = (*hexc("#8A5CFF"), 1)
    nt.links.new(tilt.outputs[0], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Emission Color"])
    me.materials.append(bevel)
    for pg in me.polygons:
        pg.material_index = 0 if pg.normal.z > 0.999 else 1
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.set_sharp_from_angle(angle=math.radians(40))
    me.update()
    title = link(bpy.data.objects.new("1984", me))
    title.rotation_euler.x = math.radians(90)
    title.parent = rig
    keyline.materials.append(emission("title keyline", hexc("#170838"), 1.0))
    kl = link(bpy.data.objects.new("1984 keyline", keyline))
    kl.location.z = -0.1
    kl.parent = title
    kl.visible_shadow = False
    glint_mat = bpy.data.materials.new("glint")               # four thin arms, white-hot at the middle and gone by the tips
    glint_mat.use_nodes = True
    nt = glint_mat.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ln = nt.nodes.new("ShaderNodeVectorMath"); ln.operation = "LENGTH"
    nt.links.new(tc.outputs["Object"], ln.inputs[0])
    fall = nt.nodes.new("ShaderNodeMapRange")
    fall.inputs["From Min"].default_value, fall.inputs["From Max"].default_value = 0.0, 1.0
    fall.inputs["To Min"].default_value, fall.inputs["To Max"].default_value = 7.0, 0.0
    fall.interpolation_type = "SMOOTHERSTEP"
    nt.links.new(ln.outputs["Value"], fall.inputs["Value"])
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 0.97, 0.92, 1)
    nt.links.new(fall.outputs[0], em.inputs["Strength"])
    ad = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(em.outputs[0], ad.inputs[0])
    nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], ad.inputs[1])
    nt.links.new(ad.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    glints = []
    for name, gx, gy, reach, peaks in (("glint a", 0.385 * tw, 0.47 * th, 1.25, (3.42, 4.72)), ("glint b", -0.19 * tw, 0.44 * th, 1.15, (4.06,))):
        w = 0.05
        gm = bpy.data.meshes.new(name)
        gm.from_pydata([(0, 1, 0), (w, w, 0), (1, 0, 0), (w, -w, 0), (0, -1, 0), (-w, -w, 0), (-1, 0, 0), (-w, w, 0)], [], [tuple(range(8))])
        gm.materials.append(glint_mat)
        g = link(bpy.data.objects.new(name, gm))
        g.parent = title
        g.location = (gx, gy, 0.47)
        g.visible_shadow = False
        glints.append((g, reach, peaks))
    # behind the camera, what the letters' flat faces mirror: warm sky over a dark horizon over a violet floor
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, -34, 15), rotation=(math.radians(90), 0, 0))
    card = bpy.context.object
    card.scale = (120, 18, 1)
    card.parent = rig
    m = bpy.data.materials.new("mirror card")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (*hexc("#FF4D94"), 1)
    els[1].position, els[1].color = 1.0, (*hexc("#FFF4E0"), 1)
    for pos, col in ((0.3, "#7A3CFF"), (0.46, "#1B1035"), (0.5, "#07040F"), (0.54, "#3D5BFF"), (0.72, "#9FE3FF")):
        els.new(pos).color = (*hexc(col), 1)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = 1.2
    nt.links.new(tc.outputs["UV"], sep.inputs[0])
    nt.links.new(sep.outputs["Y"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    card.data.materials.append(m)
    card.visible_shadow = False
    def linked_only(box, receivers):
        """Light only `receivers`. Anything else that looks at the card would see a black hole, since its light is withheld, so the
        card is see-through as well: a reflection in the road passes straight through it to the sky."""
        box.light_linking.receiver_collection = receivers
        nt = box.data.materials[0].node_tree
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        emi = next(n for n in nt.nodes if n.type == "EMISSION")
        both = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(emi.outputs[0], both.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], both.inputs[1])
        nt.links.new(both.outputs[0], out.inputs["Surface"])

    only_title = bpy.data.collections.new("only title")
    only_title.objects.link(title)
    for box in (softbox("rim", (0, 30, 22), (0, 34, 6), 30, 4, 6.0, (1.0, 0.82, 0.7)),
                softbox("violet", (17, 26, 5), (0, 34, 6), 3, 14, 6.0, (0.55, 0.35, 1.0)),
                softbox("pink", (-17, 24, 4), (0, 34, 6), 3, 12, 5.0, (1.0, 0.3, 0.6))):
        box.parent = rig
        linked_only(box, only_title)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from style_synthwave_car import outrun_car
    car, lamp_strength, flames, spokes = outrun_car(chrome, opt("paint", "#C8102E"))
    only_car = bpy.data.collections.new("only car")
    no_plate = bpy.data.collections.new("all but the plate")  # the plate stays out of the car's lights and keeps to its own small lamp
    for o in car.children_recursive:
        (no_plate if o.get("plate") else only_car).objects.link(o)
    for entry in no_plate.collection_objects:
        entry.light_linking.link_state = "EXCLUDE"
    # the car's own lights: a sheen on its deck, and the magenta glow on the road beneath it
    for box in (softbox("deck", (0, -3.5, 5.0), (0, -1.5, 0.8), 6, 3, 2.2, (1.0, 0.55, 0.85)),
                softbox("tailside", (-5, -4, 1.4), (0, -2, 0.6), 2, 2.2, 2.6, (0.4, 0.9, 1.0)),
                softbox("tailside2", (5, -4, 1.4), (0, -2, 0.6), 2, 2.2, 2.6, (1.0, 0.3, 0.7))):
        box.parent = car
        linked_only(box, only_car)
    under = bpy.data.lights.new("under", "POINT")
    under.energy, under.color, under.shadow_soft_size = 12.0, (1.0, 0.18, 0.75), 0.5
    ul = link(bpy.data.objects.new("under", under))
    ul.location = (0, 0, 0.12)
    ul.parent = car
    ul.visible_glossy = False
    red = bpy.data.lights.new("red", "POINT")
    red.energy, red.color, red.shadow_soft_size = 10.0, (1.0, 0.05, 0.12), 0.3
    rl = link(bpy.data.objects.new("red", red))
    rl.location = (0, -2.7, 0.6)
    rl.parent = car
    rl.visible_glossy = False
    rl.light_linking.receiver_collection = no_plate
    for dx in (-0.9, 0.9):                                    # just enough on the tyres to set them on the road
        low = bpy.data.lights.new("tyre light", "POINT")
        low.energy, low.color, low.shadow_soft_size = 45.0, (1.0, 0.5, 0.65), 0.2
        lo = link(bpy.data.objects.new("tyre light", low))
        lo.location = (dx, -3.5, 0.3)
        lo.parent = car
        lo.visible_glossy = False
        lo.light_linking.receiver_collection = no_plate

    # the tail lights smear into two ribbons as the car goes away
    trail = bpy.data.materials.new("trail")
    trail.use_nodes = True
    nt = trail.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    fade = nt.nodes.new("ShaderNodeMapRange")
    fade.inputs["From Min"].default_value, fade.inputs["From Max"].default_value = -1.0, 0.0
    fade.inputs["To Max"].default_value = 2.4
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*hexc("#FF1436"), 1)
    add = nt.nodes.new("ShaderNodeAddShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    nt.links.new(sep.outputs["Y"], fade.inputs["Value"])
    nt.links.new(fade.outputs[0], em.inputs["Strength"])
    nt.links.new(tr.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
    ribbons = []
    for dx in (-0.66, 0.66):
        s = Solid()
        s.box(-0.05, 0.05, -1.0, 0.0, -0.012, 0.012)
        o = s.done("ribbon", trail, car)
        o.location = (dx, -2.2, 0.72)
        ribbons.append(o)

    def smooth(a, b, x):
        u = min(1.0, max(0.0, (x - a) / (b - a)))
        return u * u * (3 - 2 * u)

    def overshoot(x):
        if x <= 0:
            return 0.0
        return 1 - math.exp(-5.5 * x) * math.cos(9 * x)

    def lane(t):                                             # starts in the right-hand lane, swings to the crown of the road
        return 1.1 * (1 - smooth(0.25, 2.35, t))

    def pop(t):                                              # three bangs from the pipes as it goes
        return max(math.exp(-((t - c) / w) ** 2) * h for c, w, h in ((2.52, 0.045, 1.0), (2.68, 0.035, 0.8), (2.82, 0.03, 0.55)))

    rock = random.Random(84)
    cam = camera((0, -6, 1.5), (0, 90, 3.6), 32)
    for f in range(1, N + 1):
        t = (f - 1) / FPS
        y = cam_y(t)
        off = 1 - smooth(0.5, 2.6, t)
        sway = -0.6 * off + 0.14 * math.sin(0.9 * t + 0.4)
        kick = math.exp(-((t - 2.52) / 0.2) ** 2)              # the launch comes up through the camera
        cam.location = (sway, y, 1.38 + 0.03 * math.sin(2.2 * t) + 0.012 * kick * math.sin(70 * t))
        look = Vector((sway * 0.3 + 7.0 * off, y + 90, 2.6)) - cam.location
        q = look.to_track_quat("-Z", "Y")
        q.rotate(Matrix.Rotation(math.radians(1.2) * math.sin(0.9 * t + 1.0), 4, look))
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = q
        gone = smooth(2.46, 5.0, t) ** 1.6
        gap = 7.2 + 78 * gone - 0.45 * smooth(1.9, 2.42, t) * (1 - smooth(2.46, 2.8, t))   # it sags back as it drops a gear
        cx = lane(t)
        vx = (lane(t + 0.02) - lane(t - 0.02)) / 0.04
        squat = 0.03 * math.exp(-((t - 2.62) / 0.24) ** 2)
        car.location = (cx, y + gap, 0.07 + 0.006 * math.sin(26 * t) - 0.012 * kick)
        car.rotation_euler = (squat, -0.06 * vx, -0.17 * vx + 0.01 * math.sin(1.7 * t))
        car.keyframe_insert("location", frame=f)
        car.keyframe_insert("rotation_euler", frame=f)
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        bang = pop(t)
        lamp_strength.default_value = 3.0 * (1 + 0.8 * smooth(2.4, 2.56, t) * (1 - smooth(2.7, 3.6, t)))
        lamp_strength.keyframe_insert("default_value", frame=f)
        for o in flames:
            width = rock.uniform(0.65, 0.95)
            o.scale = (width, (0.3 + 0.7 * bang) * rock.uniform(0.7, 1.05), width)
            o.rotation_euler.y = rock.uniform(0, 2 * math.pi)  # a different tongue every frame
            o.hide_render = bang < 0.2
            o.keyframe_insert("scale", frame=f)
            o.keyframe_insert("rotation_euler", index=1, frame=f)
            o.keyframe_insert("hide_render", frame=f)
        for o in spokes:                                     # a turn per frame would strobe: what a camera shows is the alias
            o.rotation_euler.x = 0.31 * f
            o.keyframe_insert("rotation_euler", index=0, frame=f)
        length = max(0.001, 18 * gone)
        for o in ribbons:
            o.scale = (1, length, 1)
            o.keyframe_insert("scale", frame=f)
        sky.location = (0, y, 0)
        sky.keyframe_insert("location", frame=f)
        rig.location = (0, y + 6 - 12 * smooth(2.9, 5.0, t), 0)
        rig.keyframe_insert("location", frame=f)
        rise = overshoot(t - 2.9)
        title.location = (0, 34, -9 + opt("tlift", 14.5) * rise)
        title.keyframe_insert("location", frame=f)
        for g, reach, peaks in glints:
            flash = reach * max(math.exp(-((t - c) / 0.17) ** 2) for c in peaks)
            g.scale = (flash, flash, flash)
            g.rotation_euler.z = 0.2 * (t - peaks[0])
            g.hide_render = flash < 0.03
            g.keyframe_insert("scale", frame=f)
            g.keyframe_insert("rotation_euler", index=2, frame=f)
            g.keyframe_insert("hide_render", frame=f)
    shoot(sc, "synthwave")


globals()[SCENE]()
print("done", SCENE)
