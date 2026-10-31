"""The Great Wave, assembled from real, shallow paper solids on a warm desk.

Prepare textures with the *render Python*, not Blender's Python:
    python3 manim/scenes/style_collage.py art
    Blender -b --factory-startup -P manim/scenes/style_collage_blender.py -- --frames=1,61,85,106,150 --pct=50 --samples=48
    Blender -b --factory-startup -P manim/scenes/style_collage_blender.py -- --samples=96

All outlines below are authored cubic Beziers (picture coordinates, Y up).
Concave filling/extrusion uses Blender's 2D Curve fill, evaluated to a mesh;
there is no triangle fan across the wave's hollow. The hollow is an open
concavity, not a painted sky patch. All paper pieces, including individual
flying scraps and their fibrous cores, end up in ONE animated mesh.

Full renders: frames/collage/fNNNN.png, 60 canonical seq() indices.
--frames: ordinary 30 fps timeline indices, frames/test/collage_fNNNN.png.
Preview files never contaminate the production sequence. --overwrite replaces
existing production poses; otherwise a stopped render resumes.
"""
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRAMES = os.path.join(ROOT, "styles", "film_out", "frames")
ART = os.path.join(FRAMES, "collage_art")
FPS, N = 30, 150


def opt(name, default):
    for a in ARGS:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


def link(o):
    bpy.context.collection.objects.link(o)
    return o


def lin(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(v / 12.92 if v < 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c)


def seg(t, a, b):
    return min(1.0, max(0.0, (t - a) / (b - a)))


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def path(start, *commands):
    """Closed path: pairs are straight endpoints; six-tuples are C x1 y1 x2 y2 x y.

    Control points remain explicit: resampling adds edge resolution, not an
    automatic spline that rounds away the claw tips or invents bulging handles.
    """
    p = np.array(start, float)
    out = []
    for command in commands + (tuple(start),):
        if len(command) == 2:
            q = np.array(command, float)
            n = max(1, int(np.linalg.norm(q - p) / 0.045))
            out.extend(p * (1 - u) + q * u for u in np.linspace(0, 1, n, endpoint=False))
        else:
            a, b, q = np.array(command, float).reshape(3, 2)
            length = np.linalg.norm(a - p) + np.linalg.norm(b - a) + np.linalg.norm(q - b)
            n = max(8, min(120, int(length / 0.045)))
            out.extend((1 - u) ** 3 * p + 3 * (1 - u) ** 2 * u * a + 3 * (1 - u) * u * u * b + u ** 3 * q
                       for u in np.linspace(0, 1, n, endpoint=False))
        p = q
    # The closing command can coincide with the preceding endpoint.
    clean = []
    for p in out:
        if not clean or np.linalg.norm(p - clean[-1]) > 1e-6:
            clean.append(p)
    if len(clean) > 1 and np.linalg.norm(clean[-1] - clean[0]) < 1e-6:
        clean.pop()
    return np.array(clean)


def rect(x0, y0, x1, y1):
    return path((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def transform(p, x, y, sx=1, sy=None, angle=0):
    q = np.array(p, float) * (sx, sx if sy is None else sy)
    a = math.radians(angle)
    rot = np.array(((math.cos(a), -math.sin(a)), (math.sin(a), math.cos(a))))
    return q @ rot.T + (x, y)


def normals(p):
    tangent = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    n = np.stack((tangent[:, 1], -tangent[:, 0]), axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-8)
    area = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))
    return n if area > 0 else -n


def material(name, color, rough=0.9, metallic=0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*lin(color), 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metallic
    return m


def paper_material(name):
    m = material(name, "#FFFFFF")
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    p.inputs["Specular IOR Level"].default_value = 0.22
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.join(ART, name + ".png"), check_existing=True)
    tex.interpolation, tex.extension = "Linear", "REPEAT"
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], p.inputs["Base Color"])
    # Bump is paper tooth, independent of the printed ink; no embossing letters.
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 155.0
    noise.inputs["Detail"].default_value = 2.0
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.003
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    return m


def curve_solid(name, pts, thickness):
    """Blender's concave 2D fill + extrusion. Temporary object is removed at once."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions, cu.fill_mode = "2D", "BOTH"
    cu.resolution_u = 1
    cu.extrude = thickness / 2
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for v, (x, y) in zip(sp.points, pts):
        v.co = (float(x), float(y), 0, 1)
    sp.use_cyclic_u = True
    ob = link(bpy.data.objects.new(name, cu))
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    bpy.data.objects.remove(ob, do_unlink=True)
    bpy.data.curves.remove(cu)
    return me


class PaperBag:
    """All layers in one mesh; each movable cutout owns a contiguous vertex range."""
    def __init__(self, mats):
        self.mats = mats
        self.keys = list(mats)
        self.v, self.f, self.mi, self.uv = [], [], [], []
        self.pieces = []

    def piece(self, name, pts, stock, z, group="still", torn=False, curl=0.0,
              pivot=None, delay=0.0, landing=None, seed=None, fit=False, span=8.0,
              uv_angle=0.0, land_at=None, fibres=7):
        seed = len(self.pieces) + 100 if seed is None else seed
        r = np.random.default_rng(seed)
        p = np.array(pts, float)
        n = normals(p)
        lo, hi = p.min(0), p.max(0)
        centre = np.array(pivot if pivot is not None else (lo + hi) / 2)
        begin = len(self.v)
        # Ragged white core is a real, wider, lower solid, not an outline shader.
        layers = [(p, stock, z, 0.014)]
        if torn:
            edge = p + n * r.uniform(-0.004, 0.004, (len(p), 1))
            core = p + n * r.uniform(0.009, 0.024, (len(p), 1))
            layers = [(core, "cream", z - 0.012, 0.018), (edge, stock, z, 0.010)]
        for outline, mat, zz, thick in layers:
            me = curve_solid(name, outline, thick)
            k = len(self.v)
            coords = np.array([tuple(v.co) for v in me.vertices], float)
            if len(coords) == 0:
                raise RuntimeError("Empty curve fill: " + name)
            coords[:, 2] += zz
            # A gentle lifted end, common to the printed top and fibrous underside.
            u = np.clip((coords[:, 0] - lo[0]) / max(hi[0] - lo[0], 0.01), 0, 1)
            coords[:, 2] += curl * np.maximum(0, (u - 0.68) / 0.32) ** 2
            self.v.extend(coords.tolist())
            for poly in me.polygons:
                self.f.append(tuple(k + i for i in poly.vertices))
                self.mi.append(self.keys.index(mat if abs(poly.normal.z) > 0.5 else "cream"))
                for i in poly.vertices:
                    x, y = coords[i, :2]
                    if fit:
                        uv = ((x - lo[0]) / max(hi[0] - lo[0], 0.01), (y - lo[1]) / max(hi[1] - lo[1], 0.01))
                    else:
                        a = math.radians(uv_angle)
                        xx, yy = math.cos(a) * x - math.sin(a) * y, math.sin(a) * x + math.cos(a) * y
                        uv = ((xx + 9.1) / span + (seed % 7) * 0.071, (yy + 5.2) / span + (seed % 5) * 0.061)
                    self.uv.append(uv)
            bpy.data.meshes.remove(me)
        if torn:
            # A few lifted edge fibres resolve at ~1 pixel. Hundreds share this mesh.
            for i in range(0, len(p), fibres):
                a = p[i] + n[i] * 0.009
                b = a + n[i] * r.uniform(0.012, 0.028 if fibres > 7 else 0.045)
                tangent = np.array((-n[i, 1], n[i, 0])) * 0.004
                k = len(self.v)
                self.v.extend([(a[0] - tangent[0], a[1] - tangent[1], z - 0.006),
                               (b[0], b[1], z - 0.005), (a[0] + tangent[0], a[1] + tangent[1], z - 0.006)])
                self.f.append((k, k + 1, k + 2))
                self.mi.append(self.keys.index("cream"))
                self.uv.extend([(0.1, 0.1)] * 3)
        self.pieces.append(dict(name=name, start=begin, end=len(self.v), pivot=centre, group=group,
                                delay=delay, landing=landing, seed=seed, z=z,
                                land_at=land_at if land_at is not None else 4.65 + delay * .25))

    def done(self):
        me = bpy.data.meshes.new("all cut paper / printed tops + cores + fibres")
        me.from_pydata(self.v, [], self.f)
        me.update()
        for key in self.keys:
            me.materials.append(self.mats[key])
        me.polygons.foreach_set("material_index", self.mi)
        me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.array(self.uv, np.float32).ravel())
        self.object = link(bpy.data.objects.new("collage — all paper in one mesh", me))
        self.base = np.array(self.v, float)
        return self

    def settle(self):
        """Place the last pose on actual paper surfaces, including earlier scraps.

        Vertical intersections with evaluated mesh triangles, not a painted
        shadow or an assumed flat floor. A rigid scrap bridges its highest
        support; its bottom/core clears that support by a paper-thin margin.
        """
        # Solve eventual contact even for shreds still falling in the last pose.
        self.at(59, landed=True)
        me = self.object.data
        me.calc_loop_triangles()
        triangles = np.array([tuple(tri.vertices) for tri in me.loop_triangles], dtype=np.int32)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3).astype(float)
        flying = np.zeros(len(co), bool)
        loose = [p for p in self.pieces if p["group"] in ("claw", "spray")]
        for p in loose:
            flying[p["start"]:p["end"]] = True
        supported = ~np.any(flying[triangles], axis=1)
        for p in loose:
            v = co[p["start"]:p["end"]]
            ground = co[triangles[supported]]
            lo, hi = v[:, :2].min(0), v[:, :2].max(0)
            overlap = ((ground[:, :, 0].max(1) >= lo[0]) & (ground[:, :, 0].min(1) <= hi[0]) &
                       (ground[:, :, 1].max(1) >= lo[1]) & (ground[:, :, 1].min(1) <= hi[1]))
            ground = ground[overlap]
            a, b, c = ground[:, 0], ground[:, 1], ground[:, 2]
            ab, ac = b - a, c - a
            det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
            good = np.abs(det) > 1e-10
            a, ab, ac, det = a[good], ab[good], ac[good], det[good]
            top = 0.012
            samples = v[::max(1, len(v) // 24), :2]
            samples = np.vstack((samples, (lo + hi) / 2))
            for q in samples:
                d = q - a[:, :2]
                u = (d[:, 0] * ac[:, 1] - d[:, 1] * ac[:, 0]) / det
                w = (ab[:, 0] * d[:, 1] - ab[:, 1] * d[:, 0]) / det
                inside = (u >= -1e-6) & (w >= -1e-6) & (u + w <= 1.000001)
                if inside.any():
                    top = max(top, float((a[:, 2] + u * ab[:, 2] + w * ac[:, 2])[inside].max()))
            dz = top + .003 - float(v[:, 2].min())
            v[:, 2] += dz
            x, y, z = p["landing"]
            p["landing"] = (x, y, z + dz)
            supported |= (triangles[:, 0] >= p["start"]) & (triangles[:, 0] < p["end"])

    def at(self, pose, landed=False):
        t = pose / 12
        out = self.base.copy()
        for p in self.pieces:
            sl = slice(p["start"], p["end"])
            v = out[sl]
            c = np.array((*p["pivot"], p["z"]))
            g = p["group"]
            if g in ("wave", "claw", "spray"):
                release = 2.80 + p["delay"]
                if g == "spray" and t < release:
                    # The loose fragments are physically buried until the break.
                    v[:] = wave_warp(v, release)
                    v[:, 2] -= .55
                    c = wave_warp(c[None], release)[0]
                elif g == "wave" or t <= release:
                    v[:] = wave_warp(v, t)
                    c = wave_warp(c[None], t)[0]
                else:
                    v[:] = wave_warp(v, release)
                    c = wave_warp(c[None], release)[0]
                    u = 1.0 if landed else seg(t, release, p["land_at"])
                    travel = 1 - (1 - u) ** 1.6
                    end = np.array(p["landing"])
                    target = c * (1 - travel) + end * travel
                    burst = math.sin(math.pi * u)
                    target[0] += burst * (.45 + (p["seed"] % 5) * .14)
                    target[1] += burst * (0.65 if g == "claw" else 1.05)
                    target[2] += burst * (0.38 if g == "claw" else 0.62)
                    local = v - c
                    angle = (p["seed"] % 9 - 4) * 0.52 * travel
                    ca, sa = math.cos(angle), math.sin(angle)
                    x, y = local[:, 0].copy(), local[:, 1].copy()
                    local[:, 0], local[:, 1] = ca * x - sa * y, sa * x + ca * y
                    tilt = math.sin(math.pi * u) * math.sin(9 * u + p["seed"]) * 0.27
                    y, z = local[:, 1].copy(), local[:, 2].copy()
                    local[:, 1] = math.cos(tilt) * y - math.sin(tilt) * z
                    local[:, 2] = math.sin(tilt) * y + math.cos(tilt) * z
                    v[:] = local + target
                    c = target
            elif g == "small":
                dy = 0.13 * math.sin(t * 0.9)
                flatten = ease(seg(t, 3.5, 4.55))
                v[:, 1] = -4 + (v[:, 1] + 4) * (1 - .52 * flatten) + dy - .25 * flatten
                c[1] = -4 + (c[1] + 4) * (1 - .52 * flatten) + dy - .25 * flatten
            elif g.startswith("swell"):
                k = int(g[-1])
                # Physical paper replacements slide in under the floating boats,
                # covering the spent curl with broad, non-hollow swells.
                dy = -4.7 * (1 - ease(seg(t, 3.42 + k * .12, 4.45 + k * .10)))
                v[:, 1] += dy
                c[1] += dy
            elif g.startswith("boat"):
                k = int(g[-1])
                lift = math.sin(math.pi * seg(t, 0.1 + k * 0.13, 4.65))
                calm = ease(seg(t, 3.55, 4.6))
                angle = ((-0.12 if k == 0 else 0.10) * lift + 0.045 * math.sin(t * 1.4 + k)) * (1 - calm)
                angle -= math.radians((10, 14, 27)[k]) * .76 * calm
                ca, sa = math.cos(angle), math.sin(angle)
                xy = v[:, :2] - c[:2]
                v[:, 0] = c[0] + ca * xy[:, 0] - sa * xy[:, 1] + 0.42 * ease(seg(t, 0.4, 4.4))
                v[:, 1] = c[1] + sa * xy[:, 0] + ca * xy[:, 1] + (0.50 - k * 0.09) * lift
                c[:2] += (0.42 * ease(seg(t, 0.4, 4.4)), (0.50 - k * 0.09) * lift)
                v[:, 2] += .20 * calm
                c[2] += .20 * calm
            elif g == "sun":
                dy = 4.95 * ease(seg(t, 2.8, 4.55))
                v[:, 1] += dy
                c[1] += dy
            if g != "still":
                # One coherent hand-placement error per cutout, never per vertex.
                # Shared boat seeds/pivots keep their seats and gunwales attached.
                r = np.random.default_rng(p["seed"] + pose * 7919)
                jitter = r.uniform(-0.007, 0.007, 2)
                a = r.uniform(-0.0018, 0.0018)
                xy = v[:, :2] - c[:2]
                v[:, 0] = c[0] + math.cos(a) * xy[:, 0] - math.sin(a) * xy[:, 1] + jitter[0]
                v[:, 1] = c[1] + math.sin(a) * xy[:, 0] + math.cos(a) * xy[:, 1] + jitter[1]
        self.object.data.vertices.foreach_set("co", out.astype(np.float32).ravel())
        self.object.data.update()


def wave_warp(v, t):
    """Rear, pause, reach, fold forward. Monotone Y avoids inverted paper faces."""
    q = np.array(v, float).copy()
    grow = ease(seg(t, 0.35, 2.65))
    fall = ease(seg(t, 2.85, 4.55))
    anticipate = math.sin(math.pi * seg(t, 0, 1.1)) * 0.022
    overshoot = 0.018 * math.exp(-((t - 2.72) / 0.22) ** 2)
    sy = 1 + 0.065 * grow - anticipate + overshoot - 0.72 * fall
    h = (q[:, 1] + 3.9) / 7.8
    q[:, 0] += (0.48 * grow + 2.10 * fall) * h ** 2
    # After the break the original curl is pulled beneath the new sea strips.
    # Never leave a little compressed ring / doughnut as the final silhouette.
    q[:, 1] = -3.9 + (q[:, 1] + 3.9) * sy - 3.2 * ease(seg(t, 3.55, 4.55))
    return q


# THE SILHOUETTES. The large wave has a deep, right-opening hollow and a
# heavy left shoulder, rather than a circular spiral or a generic sine wave.
GREAT = path((-8.65, -3.95),
    (-8.35, -2.65, -7.24, -1.64, -6.74, 0.15),
    (-6.18, 2.21, -5.00, 3.44, -3.39, 3.51),
    (-1.46, 3.62, 0.35, 2.94, 1.36, 1.90),
    (1.73, 1.51, 1.94, 0.97, 1.80, 0.53),
    (1.48, 1.03, 1.07, 1.56, 0.38, 1.76),
    (-0.58, 2.09, -1.78, 1.81, -2.52, 0.99),
    (-3.42, 0.04, -3.49, -1.12, -2.50, -1.98),
    (-1.24, -3.10, 1.60, -3.46, 4.15, -3.95),
    (-8.65, -3.95))

class Rail:
    """Arc-length lookup along an open Bezier run. Walking it follows the curl's
    clockwise spiral, so forward is the way the water goes and +normal is out of it."""
    def __init__(self, start, *commands):
        p = np.array(start, float)
        out = [p]
        for command in commands:
            a, b, q = np.array(command, float).reshape(3, 2)
            out.extend((1 - u) ** 3 * p + 3 * (1 - u) ** 2 * u * a + 3 * (1 - u) * u * u * b + u ** 3 * q
                       for u in np.linspace(0, 1, 120)[1:])
            p = q
        self.p = np.array(out)
        self.s = np.concatenate(((0,), np.cumsum(np.linalg.norm(np.diff(self.p, axis=0), axis=1))))
        self.length = float(self.s[-1])

    def at(self, s):
        return np.stack((np.interp(s, self.s, self.p[:, 0]), np.interp(s, self.s, self.p[:, 1])), -1)

    def frame(self, s):
        d = self.at(np.asarray(s) + .03) - self.at(np.asarray(s) - .03)
        t = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return self.at(s), t, np.stack((-t[..., 1], t[..., 0]), -1)

    def nearest(self, q):
        return float(self.s[np.argmin(np.linalg.norm(self.p - q, axis=1))])


# The same runs GREAT is cut from: up the back to the lip, then the hollow's edge.
BACK = Rail((-8.65, -3.95), (-8.35, -2.65, -7.24, -1.64, -6.74, 0.15),
            (-6.18, 2.21, -5.00, 3.44, -3.39, 3.51),
            (-1.46, 3.62, 0.35, 2.94, 1.36, 1.90),
            (1.73, 1.51, 1.94, 0.97, 1.80, 0.53))
HOLLOW = Rail((4.15, -3.95), (1.60, -3.46, -1.24, -3.10, -2.50, -1.98),
              (-3.49, -1.12, -3.42, 0.04, -2.52, 0.99),
              (-1.78, 1.81, -0.58, 2.09, 0.38, 1.76),
              (1.07, 1.56, 1.48, 1.03, 1.80, 0.53))
UNDER_LIP = Rail((1.80, 0.53), (1.48, 1.03, 1.07, 1.56, 0.38, 1.76),
                 (-0.58, 2.09, -1.78, 1.81, -2.52, 0.99))

# Cross-sections of the wave from base to lip: (point on the back, point on the hollow).
# Everything printed on the flank follows these, so stripes converge into the crest.
SECTIONS = (((-8.42, -3.40), (-.25, -3.37)), ((-7.77, -2.08), (-2.10, -2.35)),
            ((-6.74, .15), (-3.25, -.85)), ((-5.70, 2.20), (-3.20, .25)),
            ((-3.39, 3.51), (-2.30, 1.20)), ((-1.10, 3.42), (-.90, 1.95)),
            ((.60, 2.55), (.38, 1.76)), ((1.60, 1.30), (1.30, 1.12)))
_flow_s = np.linspace(0, len(SECTIONS) - 1, 351)
_soft = np.hanning(41) / np.hanning(41).sum()


def _section_arcs(rail, column):
    knots = np.interp(_flow_s, range(len(SECTIONS)), [rail.nearest(np.array(k[column])) for k in SECTIONS])
    return np.convolve(np.pad(knots, 20, mode="edge"), _soft, mode="valid")


_flow_back, _flow_hollow = _section_arcs(BACK, 0), _section_arcs(HOLLOW, 1)


def flow(f, s):
    """f: 0 on the back, 1 on the hollow's edge. s: 0 at the base, 7 at the lip."""
    o = BACK.at(np.interp(s, _flow_s, _flow_back))
    i = HOLLOW.at(np.interp(s, _flow_s, _flow_hollow))
    return o * (1 - f)[:, None] + i * f[:, None]


def flank_strip(f, width, s0, s1, bend=0.0):
    """A long tapered cut lying along the flow; both ends come to a point."""
    x = np.linspace(0, 1, 170)
    s = s0 + (s1 - s0) * x
    half = width / 2 * np.sin(np.pi * x) ** .85
    mid = f + bend * np.sin(np.pi * x)
    return np.vstack((flow(mid + half, s), flow(mid - half, s)[::-1][1:-1]))


# (f, width, from, to, stock): broad printed tones, Prussian cut back over them, cream veins.
FLANK = ((.30, .30, .15, 6.10, "midtone"), (.63, .22, .05, 5.00, "newspaper"),
         (.13, .10, .75, 6.50, "newspaper"), (.30, .09, .50, 5.20, "indigo"),
         (.21, .045, 1.30, 5.70, "indigo"), (.63, .07, .40, 4.20, "indigo"),
         (.46, .10, .10, 5.60, "halftone"), (.79, .07, .05, 4.40, "halftone"),
         (.385, .012, 1.00, 5.60, "cream"), (.55, .011, .70, 4.80, "cream"),
         (.085, .012, 2.00, 6.40, "cream"), (.72, .010, .60, 3.80, "cream"))


def foam_band(rail, s0, s1, depth, period, seed, lean=1.0):
    """Narrow white strip on a rail. Its lower edge drops in pointed tongues, the
    edge torn along the flow rather than a stroke of even width."""
    r = np.random.default_rng(seed)
    s = np.arange(s0, s1, .03)
    p, _, n = rail.frame(s)
    x = (s - s0) / (s1 - s0)
    tongue = 1 - np.abs(np.sin(np.pi * s / period + r.uniform(0, 3) + .7 * np.sin(s * 1.3))) ** .45
    w = depth * np.sin(np.pi * np.clip(x, .02, .98) ** lean) ** .55 * (.55 + 1.05 * tongue)
    return np.vstack((p + n * .025, (p - n * w[:, None])[::-1]))


def tangled(p):
    """True when an outline crosses itself; a 2D fill would then tear holes."""
    a, d = p, np.roll(p, -1, 0) - p
    cross = d[:, 0, None] * d[:, 1] - d[:, 1, None] * d[:, 0]
    gap = a[None] - a[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (gap[..., 0] * d[:, 1] - gap[..., 1] * d[:, 0]) / cross
        u = (gap[..., 0] * d[:, 1, None] - gap[..., 1] * d[:, 0, None]) / cross
    hit = (t > 0) & (t < 1) & (u > 0) & (u < 1) & (np.abs(cross) > 1e-12)
    n = len(p)
    i, j = np.nonzero(np.triu(hit, 2))
    return bool(np.any(~((i == 0) & (j == n - 1))))


def bez(a, b, c, d, n):
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 3 * a + 3 * (1 - u) ** 2 * u * b + 3 * (1 - u) * u * u * c + u ** 3 * d


def foam_hand(length, fingers, seed, base="tail", thin=.44):
    """One torn hand of foam, in local axes: +x forward, +y out of the wave.

    A broad lobe rises from the crest, arches forward and comes down to a point.
    The other fingers hang from under the arch at the same slant, nested so that
    the sky shows between them. `base` is a tail swept back along the crest, or a
    wide ragged tear on loose scraps.
    """
    for attempt in range(8):
        r = np.random.default_rng(seed + attempt * 1009)
        reach, lift = r.uniform(.93, 1.07), r.uniform(.95, 1.10)
        L = length
        P = np.array(((0, 0), (-.07, .55 * lift), (.74 * reach, 1.06 * lift), (.98 * reach, .34 * lift))) * L
        n = max(26, int(L * 1.9 / .022))
        u = np.linspace(0, 1, n)
        c = bez(*P, n)
        d = np.gradient(c, axis=0)
        normal = np.stack((-d[:, 1], d[:, 0]), 1) / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        w = thin * L * (1 - u) ** .62 * (1 + .14 * np.sin(np.pi * u))
        ph = r.uniform(0, 6.28, 4)
        left = c + normal * (w / 2 * (1 + .06 * np.sin(9 * u + ph[0]) + .04 * np.sin(23 * u + ph[1])))[:, None]
        right = c - normal * (w / 2 * (1 + .06 * np.sin(8 * u + ph[2]) + .04 * np.sin(27 * u + ph[3])))[:, None]
        under, i = [], n - 1
        for centre, reach_down in {1: (), 2: ((.50, .47),), 3: ((.63, .47), (.33, .36))}[fingers]:
            centre += r.uniform(-.02, .02)
            ln = reach_down * r.uniform(.90, 1.10) * .88 ** attempt * L
            ia, ib = int((centre + .06) * n), int((centre - .06) * n)
            under.extend(right[i:ia:-1])
            a, b = right[ia], right[ib]
            tip = (a + b) / 2 + np.array((.20, -.98)) * ln
            f = np.array((1, 0.0))
            m = max(10, int(ln * 1.1 / .022))
            under.extend(bez(a, a + np.array((0, -.42)) * ln + f * .16 * ln, tip + np.array((0, .30)) * ln + f * .14 * ln, tip, m))
            under.extend(bez(tip, tip + np.array((0, .34)) * ln + f * .03 * ln, b + np.array((0, -.34)) * ln + f * .10 * ln, b, m)[1:])
            i = ib
        under.extend(right[i:0:-1])
        l0, r0 = left[0], right[0]
        if base == "tail":
            tip = np.array((-.56, -.30)) * L
            tail = np.vstack((bez(r0, r0 + np.array((-.02, -.20)) * L, tip + np.array((.30, -.04)) * L, tip, 24),
                              bez(tip, tip + np.array((.18, .03)) * L, l0 + np.array((-.10, -.12)) * L, l0, 20)[1:-1]))
        else:
            k = np.linspace(0, 1, 8)[1:-1]
            jag = r0 + (l0 - r0) * k[:, None] + (d[0] / np.linalg.norm(d[0])) * r.uniform(-.13, .05, (6, 1)) * L
            tail = np.vstack((r0, jag))
        out = np.vstack((left, np.array(under), tail))
        if not tangled(out):
            return out
    raise RuntimeError(f"foam hand {seed} still crosses itself")


def foam_chip(size, seed):
    """A small torn scrap with two or three uneven points, no curl at all."""
    r = np.random.default_rng(seed)
    k = int(r.integers(6, 10))
    a = np.sort(r.uniform(0, 2 * np.pi, k)) + r.uniform(0, 6)
    radius = size * np.where(np.arange(k) % 3 == 0, r.uniform(.70, 1.0, k), r.uniform(.26, .46, k))
    corners = np.stack((np.cos(a) * radius, np.sin(a) * radius * .72), 1)
    out = []
    for p, q in zip(corners, np.roll(corners, -1, 0)):
        m = max(3, int(np.linalg.norm(q - p) / .024))
        out.extend(p * (1 - t) + q * t for t in np.linspace(0, 1, m, endpoint=False))
    return np.array(out)


SMALL = path((-8.65, -3.98), (-8.03, -3.31, -7.35, -2.86, -6.67, -1.98),
    (-5.94, -1.03, -5.05, -0.89, -4.31, -1.51),
    (-3.91, -1.84, -3.81, -2.19, -3.49, -2.39),
    (-3.90, -2.27, -4.18, -1.97, -4.60, -2.02),
    (-4.96, -2.13, -4.91, -2.56, -4.42, -2.91),
    (-3.57, -3.52, -2.62, -3.68, -1.56, -3.98))


def ground_papers(bag):
    """A composed found-paper ground: strongest type at the sides, quiet chart
    inside the hollow and behind Fuji. All seams are raised torn geometry."""
    cuts = [
        ("left newspaper folio", path((-10.4, -1.7), (-10.4, 6.0), (-3.4, 6.0),
            (-3.75, 3.77), (-4.9, 2.58), (-5.18, -.65)), "news_pale", .030, 12, -5),
        ("upper music leaf", path((-3.9, 3.12), (-3.3, 6.0), (10.4, 6.0),
            (10.4, 2.63), (6.25, 2.74), (3.44, 3.19), (.65, 2.92)), "music", .032, 12, 7),
        ("right newspaper fragment", path((4.68, -1.48), (4.32, 3.50), (7.31, 3.72),
            (10.4, 3.27), (10.4, -1.68), (7.6, -1.21)), "news_pale", .057, 11, -6),
        ("quiet chart in the hollow", path((-5.1, -1.67), (-4.75, 1.18), (-2.98, 2.18),
            (.51, 2.52), (4.57, 2.27), (4.28, -.62), (2.87, -1.72)), "chart_pale", .055, 13, 0),
        ("graph-paper corner", path((5.43, 1.83), (5.70, 4.48), (9.9, 4.83),
            (10.2, 2.38), (7.66, 2.09)), "graph_pale", .081, 8, -8),
        ("halftone torn wedge", path((7.37, -.63), (7.18, 1.32), (8.37, 2.05),
            (10.4, 1.69), (10.4, -1.17)), "dots_pale", .083, 9, 12),
        ("warm saved envelope edge", path((-10.4, 4.38), (-10.4, 6.0), (-6.08, 6.0),
            (-6.48, 4.59), (-8.0, 4.15)), "warm", .060, 10, 0),
        ("small music splice", path((3.92, -.90), (4.11, .91), (6.06, 1.15),
            (6.42, -.80), (5.29, -1.11)), "music", .078, 9, -14),
    ]
    for name, p, stock, z, span, angle in cuts:
        bag.piece(name, p, stock, z, torn=True, curl=.012, span=span, uv_angle=angle)


def final_swells(bag):
    """Three long, calm cuts replace the spent crest: clear boats and horizon,
    alternating broad printed blue and solid Prussian blue, not concentric rings."""
    swells = [
        path((-10, -6), (10, -6), (10, -1.56),
             (7.6, -1.14, 5.25, -1.75, 3.25, -1.76),
             (.8, -1.93, -1.1, -1.76, -3.60, -1.47),
             (-5.9, -1.13, -8.2, -1.38, -10, -1.75)),
        path((-10, -6), (10, -6), (10, -2.73),
             (7.1, -2.34, 4.2, -2.38, 1.15, -2.87),
             (-2.1, -3.25, -5.8, -2.11, -10, -2.61)),
        path((-10, -6), (10, -6), (10, -3.82),
             (5.7, -3.28, 2.3, -3.55, -.84, -3.67),
             (-4.4, -3.79, -7.2, -3.19, -10, -3.62)),
    ]
    for i, (p, stock) in enumerate(zip(swells, ("newspaper", "indigo", "halftone"))):
        bag.piece(f"final broad swell {i}", p, stock, .352 + i * .031,
                  f"swell{i}", torn=True, span=12, uv_angle=-3 if i == 0 else 0)
    accents = [
        path((-9.9, -1.77), (-7.7, -1.42, -5.84, -1.32, -3.65, -1.53),
             (-5.68, -1.43, -7.76, -1.58, -9.9, -1.91)),
        path((-5.14, -2.68), (-2.3, -2.95, -.6, -2.91, 2.7, -2.69),
             (.3, -3.11, -2.8, -3.07, -5.14, -2.68)),
        path((2.46, -3.59), (4.3, -3.49, 6.8, -3.49, 9.9, -3.84),
             (7.1, -3.68, 4.85, -3.67, 2.46, -3.59)),
    ]
    for i, p in enumerate(accents):
        bag.piece(f"final cut swell accent {i}", p, "graph" if i else "cream",
                  .375 + i * .031, f"swell{i}", torn=True, span=10)


class Scatter:
    """Landing spots that keep their distance: scraps lie apart instead of heaping.
    Keeps the final boat wells, Fuji and the red sun clear."""
    BOATS = ((1.67, -2.35, 2.75, .66), (6.04, -1.86, 2.25, .60), (-6.2, -2.55, 1.95, .60))

    def __init__(self, r):
        self.r, self.down, self.up = r, [], []

    def pick(self, taken, propose, ok=lambda q: True, size=.4):
        best, far = None, -9
        for _ in range(26):
            q = propose()
            if not ok(q):
                continue
            d = min((math.hypot(q[0] - x, q[1] - y) - .7 * (size + other) for x, y, other in taken), default=9)
            if d > far:
                best, far = q, d
        taken.append((*best, size))
        return best

    def ground(self, size=.5):
        r, m = self.r, 1 + .45 * size
        clear = lambda q: all(((q[0] - bx) / (rx * m)) ** 2 + ((q[1] - by) / (ry * m)) ** 2 > 1
                              for bx, by, rx, ry in self.BOATS)
        x, y = self.pick(self.down, lambda: (r.uniform(-8.0, 7.9), r.uniform(-4.6, -3.0) if r.random() < .70
                                              else r.uniform(-3.0, -1.7)), clear, size)
        return (float(x), float(y), .46)

    def air(self):
        r = self.r
        x, y = self.pick(self.up, lambda: (r.uniform(-7.6, -1.0) if r.random() < .45 else r.uniform(3.9, 8.1),
                                           r.uniform(-.55, 2.6)))
        return (float(x), float(y), .12)


def build(mats):
    bag = PaperBag(mats)
    bag.piece("desk", rect(-13, -8, 13, 8), "desk", -0.09, span=18)
    bag.piece("deckled drawing sheet", rect(-10.5, -6, 10.5, 6), "sky", 0,
              torn=True, curl=0.015, span=18)
    ground_papers(bag)
    # Larger, almost square postal sun; starts entirely below the horizon.
    stamp = perforated_stamp(2.50, 2.65)
    bag.piece("red sun / perforated postal paper", transform(stamp, 1.95, -3.10), "stamp", 0.108,
              "sun", curl=0.018, fit=True)
    fuji = path((-0.40, -1.40), (0.20, -1.08, 0.78, -0.45, 1.16, -0.15),
                (1.24, -0.11, 1.29, -0.12, 1.36, -0.17),
                (1.82, -0.55, 2.19, -1.13, 2.89, -1.40))
    bag.piece("Fuji / distant folded chart", fuji, "indigo", 0.131, span=10)
    snow = path((0.56, -0.77), (0.82, -0.47, 1.03, -0.25, 1.16, -0.15),
                (1.24, -0.11, 1.29, -0.12, 1.36, -0.17),
                (1.55, -0.32, 1.73, -0.56, 1.88, -0.77),
                (1.58, -0.64), (1.48, -0.79), (1.23, -0.60), (1.05, -0.74), (0.89, -0.63))
    bag.piece("Fuji snow / cut music score", snow, "cream", 0.153, span=9)
    sea = path((-10, -6), (10, -6), (10, -1.03),
               (6.2, -1.27, 4.65, -1.07, 2.74, -1.28),
               (0.58, -1.43, -1.05, -1.24, -2.85, -1.47), (-10, -1.24))
    bag.piece("low sea / soundings", sea, "chart", 0.160, torn=True, span=13)
    sea_cuts = [
        (path((2.5, -3.65), (3.9, -2.80, 6.2, -2.56, 10, -2.26), (10, -3.68)), "newspaper", .181),
        (path((-10, -3.68), (-10, -2.29), (-7.8, -2.18, -4.7, -2.83, -2.7, -3.68)), "graph", .179),
    ]
    for i, (p, stock, z) in enumerate(sea_cuts):
        bag.piece(f"found-paper sea inset {i}", p, stock, z, torn=True, span=12, uv_angle=-7)
    bag.piece("great wave / indigo dyed stock", GREAT, "indigo", 0.188, "wave", torn=True, curl=0.015)
    for i, (f, width, s0, s1, stock) in enumerate(FLANK):
        bag.piece(f"flank strip {i}", flank_strip(f, width, s0, s1), stock, 0.211 + i * 0.003, "wave", curl=0.012,
                  span=12 if stock == "newspaper" else 9, uv_angle=-4 if i % 2 else 3)
    crest_from = BACK.nearest(np.array((-6.52, .75)))
    bag.piece("foam band along the back", foam_band(BACK, crest_from, BACK.length, .27, .62, 71, lean=1.5),
              "cream", 0.255, "wave", torn=True, curl=0.020)
    bag.piece("foam band under the lip", foam_band(UNDER_LIP, 0, 2.7, .20, .44, 72, lean=.6),
              "cream", 0.257, "wave", torn=True, curl=0.020)
    # The very tip is all foam: a white fist the last hands grow out of.
    tip_back = BACK.frame(np.arange(BACK.length - 2.15, BACK.length, .03))
    tip_under = UNDER_LIP.frame(np.arange(.03, 1.85, .03))
    across = np.linspace(0, 1, 26)[1:-1]
    seam = tip_under[0][-1] + (tip_back[0][0] - tip_under[0][-1]) * across[:, None]
    seam = seam + np.array((-.75, -.30)) * (.22 * np.abs(np.sin(across * 9.5)) ** .6)[:, None]
    bag.piece("foam fist at the lip", np.vstack((tip_back[0] + tip_back[2] * .02, tip_under[0] + tip_under[2] * .02, seam)),
              "cream", 0.259, "wave", torn=True, curl=0.020)

    # One row of separate hands along the outer arc, then hanging from under the
    # lip. Sizes swell towards the lip; gaps alternate tight and open so the sky
    # shows between them, never an even comb.
    sizes = (.56, .70, .60, .92, .74, 1.12, .86, 1.28, 1.00, 1.12)
    gaps = (.98, 1.30, .94, 1.36, .96, 1.34, .96, 1.36, 1.00)
    span = np.array([(sizes[i] + sizes[i + 1]) / 2 * gaps[i] for i in range(len(gaps))])
    start, stop = crest_from + .34, BACK.length - .86
    stations = start + np.concatenate(((0,), np.cumsum(span))) * (stop - start) / span.sum()
    hands = []
    for i, (q, L) in enumerate(zip(stations, sizes)):
        p0, t, n = BACK.frame(float(q))
        hands.append((p0 - n * .06, L, 3 if i % 3 != 1 else 2, math.degrees(math.atan2(t[1], t[0])), .300 + (i % 3) * .007))
    # Hanging talons: the same hand turned to face down and back into the hollow.
    for q, L, fingers, heading, z in ((.42, .86, 3, -116, .318), (1.14, .68, 3, -128, .311),
                                      (1.76, .50, 2, -142, .304)):
        hands.append((UNDER_LIP.at(q), L, fingers, heading, z))
    r = np.random.default_rng(460)
    scatter = Scatter(r)
    for i, (at, L, fingers, heading, z) in enumerate(hands):
        cut = transform(foam_hand(L, fingers, 500 + i), at[0], at[1], angle=heading)
        bag.piece(f"foam hand {i:02d}", cut, "cream", z, "claw", torn=True, curl=.010,
                  delay=i * .024, span=10, fibres=31,
                  landing=(float(r.uniform(-7.5, 7.5)), -5.8, .46) if L >= .66 else scatter.ground(L))

    # The actual breakaway scraps: torn-off fingers and bits of hand, some curling
    # the other way, plus plain torn chips, from thumbnail to palm size. Buried
    # before release, so the hero frames stay clean.
    for i in range(72):
        rail = BACK if i % 9 else UNDER_LIP
        q = float(r.uniform(crest_from + .2, BACK.length - .1)) if i % 9 else float(r.uniform(0, 2.2))
        p0, t, n = rail.frame(q)
        at = p0 + n * r.uniform(-.12, .62) + t * r.uniform(-.25, .35)
        size = r.choice((.24, .32, .42, .56, .74, .96), p=(.22, .26, .22, .16, .10, .04)) * r.uniform(.88, 1.12)
        if i % 4 == 3:
            outline = foam_chip(float(size) * .62, 1500 + i)
        else:
            outline = foam_hand(float(size), int(r.choice((1, 2, 3), p=(.4, .4, .2))) if size > .38 else 1, 1500 + i,
                                base="torn", thin=float(r.uniform(.42, .60)))
        cut = transform(outline, at[0], at[1], -1 if i % 2 else 1, 1,
                        angle=math.degrees(math.atan2(t[1], t[0])) + r.uniform(-40, 40))
        air = i % 3 == 0
        stock = "music" if i % 13 == 0 else "foam" if i % 6 == 5 else "cream"
        bag.piece(f"torn foam scrap {i:03d}", cut, stock, .357 + (i % 5) * .008, "spray", torn=True,
                  delay=float(r.uniform(.02, .55)), landing=scatter.air() if air else scatter.ground(float(size)),
                  land_at=float(r.uniform(5.35, 5.95)) if air else float(r.uniform(4.25, 4.82)), span=11, fibres=31)

    # Two long empty oshiokuri-bune and a third partly buried in the left trough.
    boat(bag, 0, 1.25, -2.39, 1.12, 10)
    boat(bag, 1, 5.62, -1.88, .94, 14)
    boat(bag, 2, -6.62, -2.55, .80, 27)
    bag.piece("near wave / printed mid blue", SMALL, "midtone", .269, "small", torn=True, span=7)
    for i, (p, stock) in enumerate((
        (path((-7.72, -3.84), (-6.90, -3.02, -6.22, -2.22, -5.42, -1.72),
              (-6.00, -2.38, -6.40, -3.02, -6.72, -3.86)), "indigo"),
        (path((-6.12, -3.86), (-5.52, -3.12, -5.02, -2.56, -4.62, -2.30),
              (-4.95, -2.76, -5.22, -3.30, -5.24, -3.88)), "newspaper"),
        (path((-7.08, -3.86), (-6.38, -3.08, -5.82, -2.42, -5.14, -1.92),
              (-5.62, -2.56, -5.92, -3.16, -6.34, -3.88)), "halftone"))):
        bag.piece(f"near wave stripe {i}", p, stock, .281 + i * .003, "small", span=8)
    small_foam = path((-7.03, -2.45), (-6.32, -1.59, -5.53, -1.01, -4.82, -1.26),
        (-4.15, -1.42, -4.02, -1.97, -3.58, -2.32),
        (-4.16, -2.05, -4.30, -1.80, -4.62, -1.79),
        (-4.86, -1.94), (-4.99, -1.64), (-5.26, -1.82), (-5.41, -1.57),
        (-6.00, -1.62, -6.52, -2.12, -7.03, -2.45))
    bag.piece("near wave / little white crest", small_foam, "cream", .299, "small", torn=True)
    for i, (x, y, L, a) in enumerate(((-6.30, -1.74, .58, 42), (-5.62, -1.24, .72, 14), (-4.84, -1.32, .60, -26))):
        bag.piece(f"near hand {i}", transform(foam_hand(L, 3, 2300 + i), x, y, angle=a), "cream", .322 + i * .002,
                  "small", torn=True, fibres=31)
    apron = path((-10, -6), (10, -6), (10, -3.11),
                 (5.38, -3.37, 2.39, -3.19, .2, -3.32),
                 (-2.70, -3.48, -5.31, -3.70, -10, -3.31))
    bag.piece("front trough / halftone strip", apron, "halftone", .300, torn=True, span=6)
    for i, p in enumerate([
        path((-1.4, -3.52), (1.7, -3.66, 5.6, -3.41, 8.5, -3.38),
             (5.3, -3.55, 1.4, -3.80, -1.4, -3.52)),
        path((-8.4, -3.82), (-6.1, -3.53, -4.3, -3.74, -2.1, -3.84),
             (-4.6, -3.85, -6.2, -3.69, -8.4, -3.82))]):
        bag.piece(f"thin trough accent {i}", p, "chart", .321, span=8)
    bag.piece("front swell / Prussian", path((-10, -6), (10, -6), (10, -4.26),
              (6.1, -4.05, 2.2, -4.42, -1.2, -4.24), (-4.6, -4.08, -7.0, -4.46, -10, -4.20)),
              "indigo", .327, torn=True, span=9)
    final_swells(bag)
    # Only real attachment points, crossing seams in the found-paper ground.
    for i, (x, y, w, a) in enumerate([(-7.8, 4.24, .94, 22), (6.92, 2.19, .82, -18)]):
        tape = rect(-w / 2, -.13, w / 2, .13)
        bag.piece(f"masking tape {i}", transform(tape, x, y, angle=a), "tape", .113,
                  torn=True, curl=.014, span=5)
    bag.done()
    bag.settle()
    return bag


def boat(bag, k, x, y, scale, angle):
    """Slender curved hulls, dark empty wells, paper gunwales and transverse seats."""
    hull = path((-1.90, .15), (-.70, -.16, .62, -.17, 1.88, .23),
                (1.68, -.12, 1.39, -.30, .92, -.36),
                (-.24, -.48, -1.45, -.34, -1.90, .15))
    well = path((-1.64, .02), (-.62, -.19, .66, -.19, 1.62, .10),
                (1.08, -.25, -.73, -.35, -1.64, .02))
    rail = path((-1.86, .16), (-.80, -.08, .62, -.14, 1.88, .24),
                (.70, -.05, -.73, -.01, -1.86, .16))
    parts = [(hull, "boat", .228), (well, "ink", .246), (rail, "cream", .264)]
    for q in np.linspace(-1.2, 1.1, 8):
        yy = -.18 + .06 * q * q
        parts.append((rect(q - .018, yy - .10, q + .018, yy + .09), "boat", .264))
    for i, (p, stock, z) in enumerate(parts):
        bag.piece(f"boat {k} / cut {i}", transform(p, x, y, scale, angle=angle), stock, z,
                  f"boat{k}", pivot=(x, y), seed=800 + k, span=5)


def perforated_stamp(w, h):
    """Inward half-round perforations cut into all four edges; still one solid."""
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    out = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        a, b = np.array(a), np.array(b)
        length = np.linalg.norm(b - a)
        tangent = (b - a) / length
        inward = np.array((-tangent[1], tangent[0]))
        count = round(length / .14)
        pitch = length / count
        for k in range(count):
            for u in np.linspace(0, 1, 9, endpoint=False):
                out.append(a + tangent * ((k + u) * pitch) + inward * (.036 * math.sin(math.pi * u)))
    return np.array(out)


def staple(x, y, angle, mat):
    """Bent metal wire, only two staples in the entire picture."""
    cu = bpy.data.curves.new("staple", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth, cu.bevel_resolution = .012, 3
    sp = cu.splines.new("POLY")
    points = [(-.19, 0, .02), (-.19, 0, .10), (-.16, 0, .13), (.16, 0, .13), (.19, 0, .10), (.19, 0, .02)]
    sp.points.add(len(points) - 1)
    for v, p in zip(sp.points, points):
        v.co = (*p, 1)
    cu.materials.append(mat)
    o = link(bpy.data.objects.new("single bent staple", cu))
    o.location = (x, y, .025)
    o.rotation_euler.z = math.radians(angle)


def area(name, loc, target, size, energy, color):
    d = bpy.data.lights.new(name, "AREA")
    d.energy, d.color, d.size = energy, color, size
    o = link(bpy.data.objects.new(name, d))
    o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return o


def camera():
    d = bpy.data.cameras.new("near overhead copy camera")
    d.type, d.ortho_scale = "ORTHO", 17.1
    d.lens, d.sensor_width = 85, 36
    d.clip_start, d.clip_end = .01, 200
    o = link(bpy.data.objects.new("camera", d))
    o.location = (0, -2.8, 32)
    o.rotation_euler = (Vector((0, 0, 0)) - o.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = o
    # No DOF: the thin relief is inside the lens's depth of field, type stays sharp.
    return o


def camera_at(cam, pose):
    """Tight, stepped copy-camera push; track the anticipation without clipping
    the main claws. No desk or drawing-sheet border within the camera gate."""
    t = pose / 12
    push = ease(t / 5)
    cam.data.ortho_scale = 17.1 - .8 * push
    target = Vector((-.30 + .38 * push,
                     -.30 + .78 * ease(seg(t, .3, 2.7)) - .64 * ease(seg(t, 3.1, 4.9)), 0))
    cam.location = target + Vector((0, -2.8, 32))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()


def main():
    stocks = ("indigo", "newspaper", "chart", "music", "cream", "halftone", "graph", "sky", "desk", "tape", "stamp", "boat",
              "news_pale", "chart_pale", "graph_pale", "dots_pale", "warm", "midtone", "foam")
    missing = [name for name in stocks if not os.path.exists(os.path.join(ART, name + ".png"))]
    if missing:
        raise SystemExit("Missing collage paper textures: " + ", ".join(missing) +
                         "\nRun first: python3 manim/scenes/style_collage.py art")
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
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = .012
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.max_bounces = 6
    sc.cycles.diffuse_bounces = 3
    sc.cycles.glossy_bounces = 2
    sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, N
    sc.render.dither_intensity = 1
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    world = bpy.data.worlds.new("neutral daylight room")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (.77, .84, .94, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .48
    sc.world = world
    mats = {name: paper_material(name) for name in stocks}
    mats["ink"] = material("empty boat wells", "#172F40")
    bag = build(mats)
    # No captions or title-card geometry; printed words belong only to found stock.
    steel = material("dull staple steel", "#858787", rough=.38, metallic=1)
    staple(7.82, 2.12, -12, steel)
    # A distant, fairly broad key with a stronger room fill: relief still reads, and
    # scraps in the air cast soft pools rather than dark smudges. No shadow textures.
    area("directional daylight key", (-15, 13.5, 28.5), (0, 0, 0), 11, 10400, (1.0, .97, .93))
    cam = camera()
    test = [int(f) for f in opt("frames", "").split(",") if f]
    if any(f < 1 or f > N for f in test):
        raise SystemExit("--frames must be in 1..150")
    sc.render.resolution_percentage = opt("pct", 50 if test else 100)
    sc.cycles.samples = opt("samples", 48 if test else 96)
    out = os.path.join(FRAMES, "test" if test else "collage")
    os.makedirs(out, exist_ok=True)
    # IMPORTANT: seq() rounds pose/12*30 (including Python's ties-to-even).
    # Do not infer a pose back from its rounded file index: some canonical files
    # sit one timeline frame before the corresponding 12 Hz exposure boundary.
    jobs = [(f, min(59, (f - 1) * 12 // FPS)) for f in test] if test else [
        (int(round((pose / 12) * FPS)) + 1, pose) for pose in range(60)]
    for f, pose in jobs:
        target = os.path.join(out, f"collage_f{f:04d}.png" if test else f"f{f:04d}.png")
        if not test and os.path.exists(target) and "--overwrite" not in ARGS:
            continue
        start = time.time()
        sc.frame_set(f)
        bag.at(pose)
        camera_at(cam, pose)
        bpy.context.view_layer.update()
        sc.render.filepath = target
        bpy.ops.render.render(write_still=True)
        print(f"collage f{f:04d} / pose {pose:02d}: {time.time() - start:.1f}s -> {target}", flush=True)


if __name__ == "__main__":
    main()
