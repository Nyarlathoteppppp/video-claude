"""鉛字, the second of 十一's four 痛: the run of Du Fu's 新婚別 around 「沈痛迫」 set in lead type, the forme seen from above a little from its foot, as the
compositor sees it, every face the wrong way round. The heights are film_eleven.py's (plates/qian_height.npy and qian.json, from Songti TC); 痛's face is put
where the other three 痛 stand, the same size. One still: plates/tong_qian.npy (display light, as film_eleven's other plates) and tong_qian.png.

    python film_eleven.py qian_maps
    Blender -b --factory-startup -P film_tong3d.py -- [--pct 100] [--samples 256]

Type metal is lead with tin and antimony: a soft grey that takes a polish where it is rubbed. The faces carry the last printing's ink, a dark film a little
glossy, thin in places; ink and dust sit in the counters and the hairlines between sorts. The light is a big soft box high on the left, so the slopes of
every face catch it and the flat tops, inked, stay dark; a weak card on the right keeps the shadows open.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from ink3d import ROOT, mesh_grid, reset

PLATES = os.path.join(ROOT, "ink", "film_out", "film", "eleven", "plates")
TILT = 14.0                  # degrees the lens leans from straight down, from the forme's foot
LENS, STOP = 100.0, 16.0
KEY = (-.07, .03, .085)      # the soft box, from 痛's face, metres
FILL = (.08, -.02, .05)


def node(nt, kind, **values):
    n = nt.nodes.new(kind)
    for k, v in values.items():
        if k in n.inputs:
            n.inputs[k].default_value = v
        else:
            setattr(n, k, v)
    return n


def metal_material():
    m = bpy.data.materials.new("type_metal")
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    L = nt.links.new
    h = node(nt, "ShaderNodeAttribute", attribute_name="h")
    tex = node(nt, "ShaderNodeTexCoord")
    # the ink on the faces: where the height is the face's, thinner in patches and worn off at the very edges
    ink = node(nt, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
    ink.inputs["From Min"].default_value, ink.inputs["From Max"].default_value = .9, .985
    L(h.outputs["Fac"], ink.inputs["Value"])
    thin_n = node(nt, "ShaderNodeTexNoise", noise_dimensions="3D")
    thin_n.inputs["Scale"].default_value, thin_n.inputs["Detail"].default_value = 2600.0, 6.0
    L(tex.outputs["Object"], thin_n.inputs["Vector"])
    thin = node(nt, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
    thin.inputs["From Min"].default_value, thin.inputs["From Max"].default_value = .56, .72
    L(thin_n.outputs["Fac"], thin.inputs["Value"])
    keep = node(nt, "ShaderNodeMath", operation="MULTIPLY_ADD")
    L(thin.outputs["Result"], keep.inputs[0])
    keep.inputs[1].default_value, keep.inputs[2].default_value = -.2, 1.0
    inked = node(nt, "ShaderNodeMath", operation="MULTIPLY")
    L(ink.outputs["Result"], inked.inputs[0])
    L(keep.outputs["Value"], inked.inputs[1])
    # dirt in the hollows: below the face, more the lower; the spacing material below the shoulder
    low = node(nt, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
    low.inputs["From Min"].default_value, low.inputs["From Max"].default_value = .55, -.05
    L(h.outputs["Fac"], low.inputs["Value"])
    grime_n = node(nt, "ShaderNodeTexNoise", noise_dimensions="3D")
    grime_n.inputs["Scale"].default_value, grime_n.inputs["Detail"].default_value = 650.0, 5.0
    L(tex.outputs["Object"], grime_n.inputs["Vector"])
    grime = node(nt, "ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
    grime.inputs["From Min"].default_value, grime.inputs["From Max"].default_value = .38, .66
    grime.inputs["To Min"].default_value, grime.inputs["To Max"].default_value = .1, .5
    L(grime_n.outputs["Fac"], grime.inputs["Value"])
    lowg = node(nt, "ShaderNodeMath", operation="MULTIPLY")
    L(low.outputs["Result"], lowg.inputs[0])
    L(grime.outputs["Result"], lowg.inputs[1])
    under = node(nt, "ShaderNodeMapRange", interpolation_type="LINEAR")
    under.inputs["From Min"].default_value, under.inputs["From Max"].default_value = -.1, -.6
    L(h.outputs["Fac"], under.inputs["Value"])
    grain = node(nt, "ShaderNodeTexNoise", noise_dimensions="3D")
    grain.inputs["Scale"].default_value, grain.inputs["Detail"].default_value = 3000.0, 4.0
    L(tex.outputs["Object"], grain.inputs["Vector"])
    metal = node(nt, "ShaderNodeMix", data_type="RGBA")
    metal.inputs[6].default_value = (.42, .43, .45, 1)
    metal.inputs[7].default_value = (.56, .565, .58, 1)
    L(grain.outputs["Fac"], metal.inputs[0])
    dirty = node(nt, "ShaderNodeMix", data_type="RGBA")
    L(lowg.outputs["Value"], dirty.inputs[0])
    L(metal.outputs[2], dirty.inputs[6])
    dirty.inputs[7].default_value = (.12, .115, .11, 1)
    face = node(nt, "ShaderNodeMix", data_type="RGBA")
    L(inked.outputs["Value"], face.inputs[0])
    L(dirty.outputs[2], face.inputs[6])
    face.inputs[7].default_value = (.018, .018, .02, 1)
    col = node(nt, "ShaderNodeMix", data_type="RGBA")
    L(under.outputs["Result"], col.inputs[0])
    L(face.outputs[2], col.inputs[6])
    col.inputs[7].default_value = (.05, .048, .045, 1)
    L(col.outputs[2], bsdf.inputs["Base Color"])
    # metal where it is bare; the ink and the dirt are not
    bare = node(nt, "ShaderNodeMath", operation="MAXIMUM")
    L(inked.outputs["Value"], bare.inputs[0])
    L(lowg.outputs["Value"], bare.inputs[1])
    bare2 = node(nt, "ShaderNodeMath", operation="MAXIMUM")
    L(bare.outputs["Value"], bare2.inputs[0])
    L(under.outputs["Result"], bare2.inputs[1])
    met = node(nt, "ShaderNodeMath", operation="SUBTRACT")
    met.inputs[0].default_value = 1.0
    L(bare2.outputs["Value"], met.inputs[1])
    L(met.outputs["Value"], bsdf.inputs["Metallic"])
    rough = node(nt, "ShaderNodeMapRange", interpolation_type="LINEAR")
    rough.inputs["To Min"].default_value, rough.inputs["To Max"].default_value = .3, .55
    L(grain.outputs["Fac"], rough.inputs["Value"])
    r2 = node(nt, "ShaderNodeMix", data_type="FLOAT")
    L(inked.outputs["Value"], r2.inputs[0])
    L(rough.outputs["Result"], r2.inputs[2])
    r2.inputs[3].default_value = .28
    r3 = node(nt, "ShaderNodeMix", data_type="FLOAT")
    L(under.outputs["Result"], r3.inputs[0])
    L(r2.outputs[0], r3.inputs[2])
    r3.inputs[3].default_value = .75
    L(r3.outputs[0], bsdf.inputs["Roughness"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value, bump.inputs["Distance"].default_value = .04, .00002
    L(grain.outputs["Fac"], bump.inputs["Height"])
    L(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def area(name, loc, target, size, power):
    d = bpy.data.lights.new(name, "AREA")
    d.size, d.energy = size, power
    o = bpy.data.objects.new(name, d)
    bpy.context.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return o


def film(x):
    a, b, c, d, e = 2.51, .03, 2.43, .59, .14
    x = np.maximum(x, 0)
    return np.clip(x * (a * x + b) / (x * (c * x + d) + e), 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pct", type=int, default=100)
    p.add_argument("--samples", type=int, default=256)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    with open(os.path.join(PLATES, "qian.json")) as fp:
        J = json.load(fp)
    z = np.load(os.path.join(PLATES, "qian_height.npy"))
    sc = reset(a.samples, a.pct)
    sc.render.film_transparent = False
    H, W = z.shape
    mm = J["mm_per_px"] / 1000
    # the forme with 痛's face at the origin: x right, y up the page, z up off the shoulder
    bx0, by0, bx1, by1 = J["tong_box_mm"]
    cx, cy = (bx0 + bx1) / 2 / 1000, (by0 + by1) / 2 / 1000
    xs = (np.arange(W) + .5) * mm - cx
    ys = cy - (np.arange(H) + .5) * mm
    X, Y = np.meshgrid(xs, ys)
    o = mesh_grid("forme", X.astype(np.float32), Y.astype(np.float32), z, uv=False)
    rel = J["relief_mm"] / 1000
    att = o.data.attributes.new("h", "FLOAT", "POINT")
    att.data.foreach_set("value", (z / rel).ravel().astype(np.float32))
    o.data.materials.append(metal_material())
    face_z = rel
    # light
    area("key", Vector(KEY), (0, 0, face_z), .07, 9.0)
    area("fill", Vector(FILL), (0, 0, face_z), .05, 1.2)
    bg = sc.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value, bg.inputs[1].default_value = (1, 1, 1, 1), .03
    # the lens: from the forme's foot, leaning TILT from straight down, at the distance that makes 痛's face TONG_H tall, shifted so it stands at TONG_AT
    cam = bpy.data.cameras.new("cam")
    cam.lens, cam.clip_start, cam.clip_end = LENS, .001, 2.0
    co = bpy.data.objects.new("cam", cam)
    bpy.context.collection.objects.link(co)
    sc.camera = co
    t = math.radians(TILT)
    target = Vector((0, 0, face_z))
    corners = [Vector((x, y, face_z)) for x in ((bx0 / 1000 - cx), (bx1 / 1000 - cx)) for y in ((cy - by0 / 1000), (cy - by1 / 1000))]
    D = .06
    fw, fh = J["frame"]
    for _ in range(6):
        co.location = target + Vector((0, -math.sin(t), math.cos(t))) * D
        co.rotation_euler = (target - co.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        pts = [world_to_camera_view(sc, co, c) for c in corners]
        u = [q.x for q in pts]
        v = [q.y for q in pts]
        hgt = (max(v) - min(v)) * fh
        D *= hgt / J["tong_h"]
        cam.shift_x += ((max(u) + min(u)) / 2 - J["at"][1])
        cam.shift_y += ((max(v) + min(v)) / 2 - (1 - J["at"][0])) * fh / fw
    bpy.context.view_layer.update()
    print("lens", D, cam.shift_x, cam.shift_y, flush=True)
    cam.dof.use_dof = True
    cam.dof.focus_distance = D
    cam.dof.aperture_fstop = STOP
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.color_depth = "32"
    raw = os.path.join(PLATES, "tong_qian_raw.exr")
    sc.render.filepath = raw
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(raw)
    w, h = im.size
    px = np.array(im.pixels[:], np.float32).reshape(h, w, 4)[::-1, :, :3]
    # exposed for the paper of the others: the middle of the frame, lit metal and dark ink together, to a mid grey
    mid = px[h // 4:3 * h // 4, w // 4:3 * w // 4].mean()
    y = film(px * (.145 / max(mid, 1e-6)))
    np.save(os.path.join(PLATES, "tong_qian.npy"), y.astype(np.float16))
    srgb = np.where(y <= .0031308, y * 12.92, 1.055 * np.clip(y, 0, 1) ** (1 / 2.4) - .055)
    out = bpy.data.images.new("out", w, h)
    out.pixels.foreach_set(np.concatenate([srgb[::-1], np.ones((h, w, 1), np.float32)], -1).ravel())
    out.filepath_raw = os.path.join(PLATES, "tong_qian.png")
    out.file_format = "PNG"
    out.save()
    print("qian", w, h, "mid", mid, flush=True)


main()
