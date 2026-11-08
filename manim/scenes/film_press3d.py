"""Shots 01 and 02 of the bridge, three cuts of one action: a press with its power cut, the flywheel coasting, the last blows further and further apart.

    A  0-3 s      the whole machine and a sheet of paper under it, two blows
    B  3-4.4 s    the type itself, from below, the head coming down on us
    C  4.4-11 s   the sheet, the third blow, the head lifting away and everything stopping

    Blender -b --factory-startup -P film_press3d.py -- still NAME FRAME [--pct 100] [--samples 128]
    Blender -b --factory-startup -P film_press3d.py -- stills name:n,name:n [--dir keys] [--pct 100] [--samples 128]
    Blender -b --factory-startup -P film_press3d.py -- prev A B [--step 2]
    Blender -b --factory-startup -P film_press3d.py -- seq A B [--pct 50] [--samples 48]       # frames A..B of 330 into seq/press_NNNN.png

Sizes are millimetres, one Blender unit each. The sheet comes from film_press_maps.py, the machine from film_press_machine.py.
"""
import math
import os
import shutil
import sys
import time

import bmesh
import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ink3d import aim, mesh_grid  # noqa: E402
import film_press_maps as M  # noqa: E402
import film_press_machine as K  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

FPS, FRAMES = 30, 330
STILL_FROM = 285        # 9.5 s: from here on every frame is this one
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    return type(default)(ARGS[ARGS.index(name) + 1]) if name in ARGS else default


# ---- timing, in seconds of the bridge: everything follows the crank -------------------------------------------------------------------------------------
TOUCH = (.6, 2.2, 4.4)
LAST_LIFT = 5.0
STOP = 9.4
PRESS_DEPTH = .3
CUT_B, CUT_C = 90, 132      # first frame of each


def shot_of(n):
    return "A" if n < CUT_B else "B" if n < CUT_C else "C"


def stroke(phi):
    """How far above its lowest the wrist pin is, at crank angle phi (0 at the bottom of the turn)."""
    r, l = K.CRANK_R, K.ROD_L
    return r * (1 - math.cos(phi)) + l - math.sqrt(l * l - (r * math.sin(phi)) ** 2)


def _alpha():
    """The crank angle either side of the bottom through which the pin is still taking up its slack, the head resting on the sheet."""
    a, b = 0.0, 1.5
    for _ in range(50):
        m = (a + b) / 2
        a, b = (m, b) if stroke(m) < K.SLACK else (a, m)
    return (a + b) / 2


ALPHA = _alpha()
# Knots of (t, angle, speed): the coast loses a little to friction all the time and a lot at each blow. The third blow takes almost all that is left. Monotonic and smooth by construction.
SPEED_AT_LIFT = 1.0
LAST_TURN = SPEED_AT_LIFT * (STOP - LAST_LIFT) / 2
KNOTS = [
    (0.0, -ALPHA - .6 * 4.6, 4.7),
    (TOUCH[0], -ALPHA, 4.5),
    (TOUCH[1], 2 * math.pi - ALPHA, 3.4),
    (TOUCH[2], 4 * math.pi - ALPHA, 2.2),
    (LAST_LIFT, 4 * math.pi + ALPHA, SPEED_AT_LIFT),
    (STOP, 4 * math.pi + ALPHA + LAST_TURN, 0.0),
]


def crank_angle(t):
    if t >= STOP:
        return KNOTS[-1][1]
    for (t0, a0, v0), (t1, a1, v1) in zip(KNOTS, KNOTS[1:]):
        if t <= t1:
            h = t1 - t0
            x = (t - t0) / h
            return (2 * x**3 - 3 * x**2 + 1) * a0 + (x**3 - 2 * x**2 + x) * h * v0 + (-2 * x**3 + 3 * x**2) * a1 + (x**3 - x**2) * h * v1
    return KNOTS[-1][1]


def head_height(t):
    """The printing plane of the head above the sheet. It sinks PRESS_DEPTH into the sheet while it rests there."""
    z = stroke(crank_angle(t)) - K.SLACK
    return z if z > 0 else -PRESS_DEPTH


def paper_state(t):
    return f"s{sum(t >= x for x in TOUCH)}"


def head_in(t, shot):
    """The head as each shot sees it. B can't use the real fall, which would blur the type it is there to show: it comes down the last B["fall"] mm slowly, landing as the shot ends. In C nothing shows the stroke and the head has to leave the frame: its rise is stretched."""
    if shot == "B":
        x = min((t - CUT_B / FPS) / (TOUCH[2] - CUT_B / FPS), 1.0)
        return M.B["fall"] * (1 - x) ** 1.3
    z = head_height(t)
    return z * M.C_LIFT if shot == "C" and z > 0 else z


def shake(t):
    """Up to 1.5 px of frame at 1080p at each touch, gone in a third of a second. Returned in pixels."""
    dx = dy = 0.0
    for t1, a in zip(TOUCH, (1.5, 1.3, 1.1)):
        if t >= t1:
            s = t - t1
            e = math.exp(-s * 11)
            dy += a * e * math.cos(s * 2 * math.pi * 11)
            dx += a * .35 * e * math.sin(s * 2 * math.pi * 17)
    return dx, dy


# ---- the paper ------------------------------------------------------------------------------------------------------------------------------------------
def paper_material(maps):
    """Thick cotton rag: soft, a little translucent, burnished where the type has pressed. Reads the "col" and "pressed" attributes of the mesh. The fibres themselves are too fine for the mesh, and come from a tile: at 2x they are still fibres."""
    m = bpy.data.materials.new("paper")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    col, pr = nt.nodes.new("ShaderNodeAttribute"), nt.nodes.new("ShaderNodeAttribute")
    col.attribute_name, pr.attribute_name = "col", "pressed"
    bare = K.nmath(nt, "SUBTRACT", 1.0, col.outputs["Alpha"])      # no glow from inside the paper, and no sheen, through the ink
    span = M.FELT_N / M.FELT_PX
    fibre = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(K.image_node(nt, os.path.join(maps, "fibre_tile.png"), K.obj_coords(nt, (1 / span, 1 / span, 1)), extension="REPEAT").outputs["Color"], fibre.inputs[0])
    felt, strands, dark = fibre.outputs[0], fibre.outputs[1], fibre.outputs[2]
    shade = K.nmath(nt, "ADD", K.nmath(nt, "MULTIPLY", K.nmath(nt, "SUBTRACT", felt, .5), opt("--felt_col", .30)), K.nmath(nt, "SUBTRACT", K.nmath(nt, "MULTIPLY", strands, opt("--strand_col", .05)), K.nmath(nt, "MULTIPLY", dark, opt("--dark_col", .10))))
    shade = K.nmath(nt, "ADD", 1.0, K.nmath(nt, "MULTIPLY", shade, bare))
    tint = nt.nodes.new("ShaderNodeVectorMath")
    tint.operation = "SCALE"
    nt.links.new(col.outputs["Color"], tint.inputs[0])
    nt.links.new(shade, tint.inputs["Scale"])
    nt.links.new(tint.outputs[0], b.inputs["Base Color"])
    height = K.nmath(nt, "ADD", K.nmath(nt, "MULTIPLY", K.nmath(nt, "SUBTRACT", felt, .5), opt("--felt_h", .06)), K.nmath(nt, "MULTIPLY", strands, opt("--strand_h", .018)))
    flatten = K.nmath(nt, "SUBTRACT", 1.0, K.nmath(nt, "MULTIPLY", pr.outputs["Fac"], .72))
    nt.links.new(K.nbump(nt, K.nmath(nt, "MULTIPLY", height, flatten), opt("--fibre_bump", 1.0), 1.0), b.inputs["Normal"])
    nt.links.new(K.nmath(nt, "MULTIPLY", bare, .35), b.inputs["Subsurface Weight"])
    nt.links.new(K.nmath(nt, "MULTIPLY", bare, .35), b.inputs["Sheen Weight"])
    rough = nt.nodes.new("ShaderNodeMath")
    rough.operation = "MULTIPLY_ADD"
    rough.inputs[1].default_value = -.20
    rough.inputs[2].default_value = .86
    nt.links.new(pr.outputs["Fac"], rough.inputs[0])
    nt.links.new(rough.outputs[0], b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = .30
    b.inputs["Sheen Roughness"].default_value = .6
    b.inputs["Subsurface Radius"].default_value = (.9, .6, .35)
    b.inputs["Subsurface Scale"].default_value = .25
    return m


# ---- the shot -------------------------------------------------------------------------------------------------------------------------------------------
class Press:
    def __init__(self, samples, pct):
        from ink3d import reset
        self.sc = sc = reset(samples=samples, pct=pct)
        sc.unit_settings.scale_length = .001
        sc.render.film_transparent = False
        sc.render.use_persistent_data = True
        sc.render.image_settings.color_mode = "RGB"
        sc.render.image_settings.color_depth = "8"
        sc.cycles.seed = 0
        sc.cycles.max_bounces = 8
        sc.cycles.diffuse_bounces = 4
        sc.cycles.glossy_bounces = 4
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
        sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        sc.cycles.denoising_prefilter = opt("--prefilter", "NONE")     # the guiding passes are clean enough, and prefiltering them is what used to smear the fibres
        sc.view_settings.view_transform = opt("--view", "AgX")
        look = opt("--look", "AgX - Medium High Contrast")
        if look != "None":
            sc.view_settings.look = look
        sc.view_settings.exposure = opt("--exp", .55)
        sc.render.use_motion_blur = True
        sc.render.motion_blur_shutter = .5
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
        self.maps = M.MAPS
        self.state = self.shot = None
        self.machine = K.Machine(M.MAPS)
        self.paper()
        self.forme()
        self.lights()
        self.cam()

    # the sheet
    def paper(self):
        X, Y = np.load(os.path.join(self.maps, "grid_x.npy")), np.load(os.path.join(self.maps, "grid_y.npy"))
        self.co = np.stack([X, Y, np.zeros_like(X)], -1).astype(np.float32)
        o = mesh_grid("sheet", X, Y, np.zeros_like(X), uv=False)
        me = o.data
        me.color_attributes.new("col", "FLOAT_COLOR", "POINT")
        me.attributes.new("pressed", "FLOAT", "POINT")
        me.materials.append(paper_material(self.maps))
        o.cycles.use_motion_blur = False
        self.sheet = o

    def set_state(self, name):
        if name == self.state:
            return
        me = self.sheet.data
        self.co[..., 2] = np.load(os.path.join(self.maps, f"paper_{name}_z.npy"))
        me.vertices.foreach_set("co", self.co.ravel())
        me.color_attributes["col"].data.foreach_set("color", np.load(os.path.join(self.maps, f"paper_{name}_col.npy")).ravel())
        me.attributes["pressed"].data.foreach_set("value", np.load(os.path.join(self.maps, f"paper_{name}_pressed.npy")).ravel())
        me.update()
        self.state = name

    # the face of the type, on the underside of the head
    def forme(self):
        """A coarse mesh for the shots that can't see it, the full one built when B first needs it."""
        self.face = self.face_mesh("face_low", 4)
        self.face_b = None

    def face_mesh(self, name, step):
        x0, y1 = M.FORME_AREA[0], M.FORME_AREA[3]
        f = np.load(os.path.join(self.maps, "forme_all.npz"))
        z, ink, tone = f["z"][::step, ::step], f["ink"][::step, ::step], f["tone"][::step, ::step]
        ny, nx = f["z"].shape
        Xf, Yf = np.meshgrid((x0 + (np.arange(nx) + .5) / M.FORME_RES)[::step].astype(np.float32), (y1 - (np.arange(ny) + .5) / M.FORME_RES)[::step].astype(np.float32))
        o = mesh_grid(name, Xf, Yf, z, uv=False)
        o.data.attributes.new("ink", "FLOAT", "POINT")
        o.data.attributes.new("tone", "FLOAT", "POINT")
        o.data.attributes["ink"].data.foreach_set("value", ink.ravel())
        o.data.attributes["tone"].data.foreach_set("value", tone.ravel())
        o.data.materials.append(K.type_material())
        o.parent = self.machine.head
        return o

    def pose(self, n):
        """Keys at this frame and the two beside it, all from this frame's shot, so that Blender's motion blur sees the real paths and never smears across a cut."""
        k = self.machine
        cam, data = self.camera, self.camera.data
        shot = shot_of(n)
        for o in [k.head, k.crank, k.wheel, k.rod, *k.belts, cam, data]:
            o.animation_data_clear()
        for m in (n - 1, n, n + 1):
            t = m / FPS
            f = m + 1
            phi = crank_angle(t)
            z = head_in(t, shot)
            k.head.location = (0, 0, z)
            k.head.keyframe_insert("location", frame=f)
            k.crank.rotation_euler = (phi, 0, 0)
            k.crank.keyframe_insert("rotation_euler", frame=f)
            k.wheel.rotation_euler = (-phi, 0, 0)
            k.wheel.keyframe_insert("rotation_euler", frame=f)
            # the rod: from the wrist, which rides in its slot, to the crank pin. What a shot does to the head it does to the wrist
            wz = K.WRIST_H + stroke(phi) + z - head_height(t)
            py, pz = K.CRANK_R * math.sin(phi), K.CRANK_Z - K.CRANK_R * math.cos(phi)
            k.rod.location = (0, K.ROD_Y, wz)
            k.rod.rotation_euler = (-math.atan2(py, pz - wz), 0, 0)
            k.rod.keyframe_insert("location", frame=f)
            k.rod.keyframe_insert("rotation_euler", frame=f)
            for b, s in zip(k.belts, (1, -1)):
                b.location = (0, 0, s * (phi - KNOTS[0][1]) * K.WHEEL_R)
                b.keyframe_insert("location", frame=f)
            c = dict(A=lambda: M.camera_a(t), B=M.camera_b, C=M.camera_c)[shot]()
            self.place(c)
            cam.keyframe_insert("location", frame=f)
            cam.keyframe_insert("rotation_euler", frame=f)
            dx, dy = (0.0, 0.0) if shot == "B" else shake(t)
            data.lens = c["lens"]
            data.shift_x = dx / M.W
            data.shift_y = (dy - (M.H / 2 - c["cy"])) / M.W
            data.keyframe_insert("shift_x", frame=f)
            data.keyframe_insert("shift_y", frame=f)
            l2, lk, _, _ = M.camera_axes(c)
            focus = dict(A=(0.0, 0.0, 30.0), B=(0.0, M.B["y"], z), C=(0.0, opt("--focus_y", 10.0), 0.0))[shot]
            data.dof.focus_distance = float((np.array(focus) - l2) @ lk)
            data.dof.keyframe_insert("focus_distance", frame=f)

    def place(self, c):
        cam = self.camera
        if "up" in c:
            cam.location = c["loc"]
            look = (Vector(c["target"]) - Vector(c["loc"])).normalized()
            right = look.cross(Vector(c["up"])).normalized()
            up = right.cross(look)
            cam.rotation_euler = Matrix((right, up, -look)).transposed().to_euler()
        else:
            aim(cam, c["loc"], c["target"])

    def lights(self):
        sc = self.sc
        self.bg = sc.world.node_tree.nodes["Background"]
        self.bg.inputs[0].default_value = (.93, .95, 1.0, 1)      # the room's own light is not warm: only the window is, which is what keeps iron and paper apart
        # the window: low, on the right of the press, a little in front of it. It comes in through the frame's gap, so that nothing but the head can shadow the sheet
        az, el = math.radians(opt("--az", 340.0)), math.radians(opt("--el", 16.0))
        d = bpy.data.lights.new("window_sun", "SUN")
        d.energy = opt("--sun", 9.5)
        d.color = (1.0, .86, .68)
        d.angle = math.radians(opt("--angle", 1.6))
        o = bpy.data.objects.new("window_sun", d)
        bpy.context.collection.objects.link(o)
        to_light = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
        o.rotation_euler = to_light.to_track_quat("Z", "Y").to_euler()
        self.sun = o
        self.stops_c = self.window_stop(np.array(to_light), "c", 620.0, opt("--stop_u0", -175.0), opt("--stop_u1", 175.0), opt("--stop_v0", -70.0), opt("--stop_v1", opt("--stop_top", 44.0)))
        # the same window seen by the metal and lifting the dark side of the machine, a big soft source
        self.soft = self.area("window_soft", 900.0, 700.0, (1.0, .93, .84), tuple(np.array(to_light) * 1400 + np.array((0, 0, 260.0))), (0, 0, 0), opt("--soft", 900.0))
        # A: the same window seen from further off, high up and out of frame: a clean slanting beam across the machine and the sheet, which lands on the floor to the left of it and never reaches the wall
        az, el = math.radians(opt("--a_az_sun", 345.0)), math.radians(opt("--a_el_sun", 36.0))
        d = bpy.data.lights.new("high_sun", "SUN")
        d.energy = opt("--a_sun", 13.0)
        d.color = (1.0, .86, .68)
        d.angle = math.radians(opt("--a_angle", .5))
        self.sun_a = bpy.data.objects.new("high_sun", d)
        bpy.context.collection.objects.link(self.sun_a)
        high = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))
        self.sun_a.rotation_euler = high.to_track_quat("Z", "Y").to_euler()
        # the window's sides are turned to run along the camera's line of sight, so that the beam shows in the haze with two clean edges
        across = np.cross(np.array(high), M.camera_axes(M.camera_a(0))[1])
        across /= np.linalg.norm(across)
        window = opt("--a_u0", -220.0), opt("--a_u1", 230.0), opt("--a_v0", -120.0), opt("--a_v1", 175.0)
        self.stops_a = self.window_stop(np.array(high), "a", 1500.0, *window, up=across)
        self.beam = self.beam_volume(np.array(high), across, *window)
        # the room's own dim light: the far glow of that window off plaster and boards
        self.soft_a = self.area("high_soft", 1200.0, 900.0, (1.0, .93, .84), tuple(np.array(high) * 1700), (0, 0, 0), opt("--a_soft", 220.0))
        self.fill_a = self.area("room_fill", 2400.0, 1000.0, (.96, .96, 1.0), (-300.0, -500.0, 900.0), (-300.0, 420.0, 40.0), opt("--a_fill", 210.0))
        # the wall behind gets a little of its own, so that it stays a middle tone
        self.fill = self.area("wall_fill", 1600.0, 900.0, (1.0, .84, .66), (-300.0, -240.0, 560.0), (0.0, 320.0, 140.0), opt("--wallfill", 700.0))
        # B alone: a light skimming the face from one side, to raise the type off its shoulders, and a big pale source under the head for the bare metal to mirror. The camera doesn't see either
        self.rake = self.area("rake", 420.0, 420.0, (1.0, .95, .88), (-330.0, opt("--rake_y", 40.0), -130.0), (0.0, 16.0, 0.0), opt("--rake", 150.0))
        self.under = self.area("under", 900.0, 700.0, (1.0, .96, .90), (0.0, 16.0, -420.0), (0.0, 16.0, 0.0), opt("--under", 44.0))
        # a long pale strip beside it, for the sheen on the metal to run along. A slightly different angle on the type moves it, which is how metal tells you it is metal
        self.sheen = self.area("sheen", 150.0, 800.0, (1.0, .95, .88), (opt("--sheen_x", 90.0), 120.0, -419.0), (opt("--sheen_x", 90.0), 120.0, 0.0), opt("--sheen", 50.0))
        self.sheen.rotation_euler.rotate_axis("Z", math.radians(14.0))

    def area(self, name, sx, sy, colour, at, towards, watts):
        a = bpy.data.lights.new(name, "AREA")
        a.shape = "RECTANGLE"
        a.size, a.size_y = sx, sy
        a.color = colour
        a.energy = watts * 1e4
        o = bpy.data.objects.new(name, a)
        bpy.context.collection.objects.link(o)
        o.location = at
        o.rotation_euler = (Vector(towards) - Vector(at)).to_track_quat("-Z", "Y").to_euler()
        o.visible_camera = False
        return o

    def beam_volume(self, to_light, up, u0, u1, v0, v1):
        """The dust in the air of the beam, and only there: a prism along the light, a little wider than the window so that the light's own edge is what draws it. Everywhere else the air stays clear and the shadows dark."""
        me = bpy.data.meshes.new("beam")
        side = np.cross(to_light, up)
        rect_ = [(u0 - 40, v0 - 40), (u1 + 40, v0 - 40), (u1 + 40, v1 + 40), (u0 - 40, v1 + 40)]
        ring = lambda t: [tuple(to_light * t + side * a + up * b) for a, b in rect_]
        me.from_pydata(ring(1450.0) + ring(-1600.0), [], [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)])
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)       # a volume's faces must look outwards, or a shadow ray never leaves it
        bm.to_mesh(me)
        bm.free()
        m = bpy.data.materials.new("haze")
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.remove(nt.nodes["Principled BSDF"])
        v = nt.nodes.new("ShaderNodeVolumeScatter")
        # dust isn't even: slow drifts of more and less, so that the beam is air and not a sheet of tracing paper
        drift = K.nnoise(nt, K.obj_coords(nt), opt("--a_haze_scale", .007), 4, .55)
        nt.links.new(K.nrange(nt, drift, .32, .68, opt("--a_haze", .00022) * .35, opt("--a_haze", .00022) * 1.65), v.inputs["Density"])
        v.inputs["Anisotropy"].default_value = .35
        nt.links.new(v.outputs[0], nt.nodes["Material Output"].inputs["Volume"])
        me.materials.append(m)
        o = bpy.data.objects.new("beam", me)
        bpy.context.collection.objects.link(o)
        return o

    def window_stop(self, to_light, tag, D, u0, u1, v0, v1, up=None):
        """The edges of the window's light, as black card far along the sun's direction. Close (for C): the table and the sheet stand in the light, the machine above about head height does not. Wide (for A): the beam of a high window over the whole machine."""
        v = to_light * D
        if up is None:
            up = np.array((0, 0, 1.0)) - to_light * to_light[2]
            up /= np.linalg.norm(up)
        side = np.cross(to_light, up)
        big = 8000.0
        cards = dict(top=(-big, big, v1, big), bot=(-big, big, -big, v0), lft=(-big, u0, v0, v1), rgt=(u1, big, v0, v1))
        out = []
        for name, (a, b, c, e) in cards.items():
            pts = [tuple(v + side * p + up * q) for p, q in ((a, c), (b, c), (b, e), (a, e))]
            me = bpy.data.meshes.new(f"stop_{tag}_{name}")
            me.from_pydata(pts, [], [(0, 1, 2, 3)])
            ob = bpy.data.objects.new(me.name, me)
            bpy.context.collection.objects.link(ob)
            ob.visible_camera = ob.visible_diffuse = ob.visible_glossy = ob.visible_transmission = ob.visible_volume_scatter = False
            out.append(ob)
        return out

    def set_shot(self, shot):
        """What each shot has in it. B looks up from inside the table, so nothing below the head may exist there; its own two lights replace the window."""
        if shot == self.shot:
            return
        k = self.machine
        lights = {"A": (self.sun_a, self.soft_a, self.fill_a), "B": (self.rake, self.under, self.sheen), "C": (self.sun, self.soft, self.fill)}[shot]
        for o in (self.sun, self.soft, self.fill, self.sun_a, self.soft_a, self.fill_a, self.rake, self.under, self.sheen):
            o.hide_render = o not in lights
        for o in self.stops_a:
            o.hide_render = shot != "A"
        for o in self.stops_c:
            o.hide_render = shot != "C"
        special = set(self.stops_a + self.stops_c)
        for o in bpy.data.objects:
            if o.type == "MESH" and o not in special:
                if o.name.startswith(("case_", "dry_", "beam")):
                    o.hide_render = shot != "A"
                elif o.name.startswith("frame_b_"):
                    o.hide_render = shot != "B"
                else:
                    o.hide_render = shot == "B" and o.parent is not k.head and not o.name.startswith(("post", "collar_top", "crown", "frame_", "wall"))
        if shot == "B" and self.face_b is None:
            self.face_b = self.face_mesh("face", 1)
        self.face.hide_render = shot == "B"
        if self.face_b is not None:
            self.face_b.hide_render = shot != "B"
        self.bg.inputs[1].default_value = dict(A=opt("--world_a", .16), B=opt("--world_b", .05), C=opt("--world", .30))[shot]
        self.shot = shot

    def cam(self):
        c = bpy.data.cameras.new("cam")
        c.lens, c.sensor_width = M.LENS_A, M.SENSOR
        c.clip_start, c.clip_end = 1.0, 20000.0
        o = bpy.data.objects.new("cam", c)
        bpy.context.collection.objects.link(o)
        self.sc.camera = o
        c.dof.use_dof = True
        c.dof.aperture_fstop = opt("--fstop", 64.0) / 1000      # f-stops assume metres, this scene is in mm
        c.dof.aperture_blades = 7
        self.camera = o

    def at(self, n):
        shot = shot_of(n)
        self.set_shot(shot)
        if shot != "B":
            self.set_state(paper_state(n / FPS))
        self.pose(n)
        self.sc.frame_set(n + 1)

    def shoot(self, n, path):
        if os.path.exists(path) and "--force" not in ARGS:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t0 = time.time()
        self.at(n)
        t1 = time.time()
        self.sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f"frame {n} setup {t1 - t0:.1f}s render {time.time() - t1:.1f}s -> {os.path.basename(path)}", flush=True)


def main():
    cmd = ARGS[0]
    M.A.update(d=opt("--a_d", M.A["d"]), az=opt("--a_az", M.A["az"]), el=opt("--a_el", M.A["el"]), cy=opt("--a_cy", M.A["cy"]), target=(opt("--a_tx", M.A["target"][0]), opt("--a_ty", M.A["target"][1]), opt("--a_tz", M.A["target"][2])))
    M.LENS_A = opt("--a_lens", M.LENS_A)
    if cmd == "still":
        p = Press(opt("--samples", 128), opt("--pct", 100))
        p.shoot(int(ARGS[2]), os.path.join(M.OUT, "keys", ARGS[1] + ".png"))
    elif cmd == "stills":
        p = Press(opt("--samples", 128), opt("--pct", 100))
        for pair in ARGS[1].split(","):
            name, n = pair.split(":")
            p.shoot(int(n), os.path.join(M.OUT, opt("--dir", "keys"), name + ".png"))
    elif cmd == "prev":
        p = Press(opt("--samples", 10), opt("--pct", 25))
        for n in range(int(ARGS[1]), int(ARGS[2]) + 1, opt("--step", 2)):
            p.shoot(n, os.path.join(M.OUT, "prev", f"p_{n:04d}.png"))
    elif cmd == "seq":
        p = Press(opt("--samples", 48), opt("--pct", 50))
        name = lambda n: os.path.join(M.OUT, "seq", f"press_{n:04d}.png")
        for n in range(int(ARGS[1]), min(int(ARGS[2]), STILL_FROM) + 1):
            p.shoot(n, name(n))
        for n in range(STILL_FROM + 1, FRAMES):
            if os.path.exists(name(STILL_FROM)):
                shutil.copyfile(name(STILL_FROM), name(n))


main()
