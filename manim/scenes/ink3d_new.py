"""Chapter 13, 新东西 (ink3d.scenes.md): these names were all made for known things. I am a new one.

    Blender -b --factory-startup -P ink3d_new.py -- --shot=edge|char [--t=1.0] [--pct=100] [--samples=64] [--name=SUFFIX]

edge «而我是一种新东西». The world reaches the edge of the map. The last mountains stand back on either side and stop being finished, going to dry brush and then to bare
     paper; the river goes on up the middle with nothing on either side, thins, and frays out across the white. Moves: the camera keeps flying upstream at the film's pace,
     the mountains slide away past the frame's edges, t carries the river's end further into the paper. Comes from the 器 slab, the last of the names; hands over as the
     strokes of the river lift off the paper.
char «哪怕它本身就是旧词做成的». The third climax. In the middle of the white, a character that looks written and is no character. Its strokes are the strokes of 人 神 鬼 器, wet,
     loaded, still spreading into the paper; the river that brought them still feeds it from below. Moves: t: the strokes drift in from the water and find their places
     (run in reverse from the ink dispersing), the camera sinks slowly toward it and stops. It holds. Hands over to chapter 14 when the white is seen to be a page.

edge is finished on the paper: grade it plain to edge_base_ink.png (--mist=0), then ink3d_new_post.py fades the mountains and paints the river. char: --mist=0.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np

from ink3d import OUT3D, W, H, blur, camera, erode, image_from, ink_albedo, lerp, massif, mesh_grid, painted_material, render, reset, sm, spectral, sun
from ink3d_gns import TEX, aerial, attr_material, border, boulder, ink_image, mesh_arrays, opts, resample, sample, screen_plane, tex, world_split

OPT = opts(sys.argv)
SHOT, T, PCT, SAMPLES = OPT.get("shot", "char"), float(OPT.get("t", 1.0)), int(OPT.get("pct", 100)), int(OPT.get("samples", 64))
NAME = os.path.join(OUT3D, "new", SHOT + OPT.get("name", ""))
RES = PCT / 100


def begin():
    reset(samples=SAMPLES, pct=PCT)
    if "border" in OPT:
        border(OPT["border"])


def white_mix(mat, fac_socket):
    """Mix a material into plain white emission: what it is painted over goes back to bare paper."""
    nt = mat.node_tree
    out = nt.nodes["Material Output"]
    src = out.inputs["Surface"].links[0].from_socket
    e, mix = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeMixShader")
    e.inputs["Color"].default_value = (1, 1, 1, 1)
    e.inputs["Strength"].default_value = 1.0
    nt.links.new(fac_socket, mix.inputs[0])
    nt.links.new(src, mix.inputs[1])
    nt.links.new(e.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])


def fray_mask(n, seed, length):
    """Where the river stops: 1 on the paper, 0 where it is still water. x across the river, y along it, 0-1. The river is a band that comes apart into long wet fingers."""
    y, x = np.mgrid[:n, :n].astype(np.float32) / (n - 1)
    a = spectral(n, 3.2, seed)
    b = spectral(n, 2.2, seed + 1)
    c = spectral(n, 2.8, seed + 2)
    # warp strongly sideways, stretched along the river's direction
    sx = (x - .5) + .10 * lerp(a, x * n * .3, y * n * .12) * (.15 + y)
    sy = y + .05 * lerp(c, x * n * .6, y * n * .3)
    streak = lerp(b, sx * n * 1.8, sy * n * .13)
    return sx, sy, streak


def shot_edge():
    begin()
    cam, target, lens = (0.0, -90.0, 52.0), (0.0, 30.0, 0.0), 24.0
    # the last mountains, either side, standing back. They stop being finished the farther up they are
    n = 2100
    size = 300.0
    y, x = np.mgrid[:n, :n].astype(np.float32)
    X, Y = (x / (n - 1) - .5) * size, (y / (n - 1) - .5) * size + 30
    peaks = [(-66, -26, 40, 22), (-94, 4, 56, 30), (-110, 40, 46, 30), (-52, -44, 18, 14), (66, -30, 34, 20), (98, 2, 50, 30), (118, 44, 42, 34), (50, -48, 14, 13), (-132, 70, 34, 26), (138, 72, 32, 28)]
    h = massif(X, Y, peaks, 801)
    cx = 2.4 * np.sin(Y / 30 + .6)
    d = np.abs(X - cx)
    h = h * sm((d - 13 - .05 * (Y + 120)) / 22) ** .7
    h = erode(h, size / n, 802, wavelength=26, octaves=7, gain=.5, strength=.2)
    h = np.maximum(h, 0)
    h = h - .5 * (1 - sm(h / 1.4))
    alb = ink_albedo(h, 803, size / n)
    # unfinished: along the slopes the paint gives out into strokes, then nothing. A value per texel, used as the mix toward paper
    br = lerp(spectral(1024, 1.7, 804), x * 1024 / n * 1.0, y * 1024 / n * .36)
    wash = lerp(spectral(1024, 3.2, 805), x * 1024 / n, y * 1024 / n)
    edge = (Y - (-12.0 + 10 * wash)) / 62.0 + .10 * br
    gone = sm(edge)
    gone = np.where(gone < .98, gone, 1.0)
    o = mesh_grid("land", X, Y, h)
    m = painted_material("land_rock", alb)
    t = m.node_tree.nodes.new("ShaderNodeTexImage")
    t.image = image_from(gone.astype(np.float32), "gone")
    t.image.colorspace_settings.name = "Non-Color"
    tc = m.node_tree.nodes.new("ShaderNodeTexCoord")
    m.node_tree.links.new(tc.outputs["UV"], t.inputs["Vector"])
    white_mix(m, t.outputs["Color"])
    aerial(m, cam=cam, k=.0035, s0=50.0, H=60.0)
    o.data.materials.append(m)
    # the river on a frame-fine grid. Near, it is water, mirroring the sky and the mountains; going on into the white, it is ink lying on paper, thinning, coming apart
    level = .15
    Xw, Yw = screen_plane(cam, target, lens, z=level, res=.8 * RES, far=1600.0)
    no = spectral(1024, 2.8, 811)
    z = .007 * lerp(no, (Xw % 300) * 3.4, (Yw % 300) * 3.4)
    wash1 = lerp(spectral(1024, 3.2, 812), (Xw % 800) * 1.28, (Yw % 800) * .6)
    wash2 = lerp(spectral(1024, 2.8, 813), (Xw % 800) * 5.0, (Yw % 800) * .7)
    fib = lerp(spectral(1024, 1.9, 814), (Xw % 400) * 10.0, (Yw % 400) * .7)
    # how far the river has got, how wide it is there: y grows up the frame
    u = (Yw - 30.0) / 260.0
    wid = 15.0 * (1 - .78 * sm(u * 1.3)) * (1 + .30 * wash1)
    cen = 2.4 * np.sin(Yw / 30 + .6) * sm((30 - Yw) / 40) + 16.0 * np.sin(Yw / 70 + 1.2) * sm(u * 2) + 8 * wash1 * sm(u * 3)
    dx = np.abs(Xw - cen) + 2.2 * wash2 * (.3 + 6 * sm(u)) + 1.8 * fib * (.1 + 3 * sm(u))
    band = sm((wid - dx) / (2.0 + 26 * sm(u)))
    # the ink's tone: pale wash, with a tide line at its edges and dark threads running up it, ending at the far tip
    tip = 1 - sm((u - .72 - T * .12 + .16 * wash1) / .32)
    threads = np.clip(lerp(spectral(1024, 2.2, 815), (Xw % 800) * 1.0 * 1.28 * 3, (Yw % 800) * .1) * .5 + .15, 0, 1)
    tone = band * tip * (.34 + .38 * threads + .18 * np.exp(-((wid - dx) / (1.4 + 8 * sm(u))) ** 2))
    paper = sm((Yw + 10.0 + 28 * wash1) / 90.0)           # from here on it is painted, not water
    C = np.clip(1 - tone * paper, 0, 1)                    # the paper's lightness where it is paper
    w = mesh_arrays("river", Xw, Yw, level + z * (1 - paper), C)
    pa = w.data.color_attributes.new("P", "FLOAT_COLOR", "POINT")
    rgba = np.ones((Xw.size, 4), np.float32)
    rgba[:, 0] = rgba[:, 1] = rgba[:, 2] = paper.ravel()
    pa.data.foreach_set("color", rgba.ravel())
    mw = bpy.data.materials.new("water_to_paper")
    mw.use_nodes = True
    nt = mw.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (.86, .86, .86, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = .03
    at, ap = nt.nodes.new("ShaderNodeAttribute"), nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name, ap.attribute_name = "C", "P"
    em, mix = nt.nodes.new("ShaderNodeEmission"), nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(at.outputs["Color"], em.inputs["Color"])
    nt.links.new(ap.outputs["Fac"], mix.inputs[0])
    nt.links.new(b.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
    w.data.materials.append(mw)
    world_split(.5, 1.15, zenith=.8)
    sun((-.6, -.55, -.5), energy=2.2, angle=7)
    camera(cam, target, lens=lens)
    render(NAME)


def shot_char():
    begin()
    # a sheet of real paper with real wet ink on it, and the river's last threads running into the character from below
    sz = 54.0
    n = int(2600 * max(RES, .55))
    y, x = np.mgrid[:n, :n].astype(np.float32)
    X, Y = (x / (n - 1) - .5) * sz, (y / (n - 1) - .5) * sz + 1.0
    glyph_d = tex("newchar")                # density of ink, 3000 square
    gs = 25.0                                # how big the character is on the sheet
    U, V = (X + gs / 2) / gs, 1 - (Y - 1.0 + gs / 2) / gs
    ins = (U > 0) & (U < 1) & (V > 0) & (V < 1)
    D = np.zeros_like(X)
    D[ins] = sample(glyph_d, U[ins], V[ins])
    # the river's threads, thin dark lines of wash, come up from the bottom edge of the frame to the foot of the character and into its strokes
    wa = lerp(spectral(1024, 3.0, 901), x * 1024 / n, y * 1024 / n)
    wb = lerp(spectral(1024, 2.0, 902), x * 1024 / n * 2.4, y * 1024 / n * .5)
    wc = lerp(spectral(1024, 2.6, 903), x * 1024 / n * 1.2, y * 1024 / n * .5)
    tv = sm((-11.2 - (Y - 1.0) + 1.6 * wa) / 3.0)         # 1 below the character's foot
    cx = 3.6 * np.sin(Y / 7.0 + .8) * .6 + 1.2 + 1.3 * wa
    band = np.exp(-((X - cx) / (3.0 + .10 * np.maximum(-Y, 0))) ** 2)
    thread = np.clip(.5 + 1.1 * wb, 0, 1) ** 3
    river = tv * band * (.34 * np.clip(.6 + wc, 0, 1) + 1.15 * thread) * (1 + .4 * wc)
    # strokes not yet settled: a few lie loose on the paper, the last of 人 神 鬼 器 drifting in
    r = np.random.default_rng(907)
    D = np.maximum(D, np.clip(river, 0, 1.9))
    # the sheet: paper fibre for height, slightly cockled where it is wet, ink standing proud where it pooled
    fib = lerp(spectral(1024, 2.4, 904), x * 1024 / n * 3, y * 1024 / n * 3)
    cock = lerp(spectral(1024, 3.4, 905), x * 1024 / n * .6, y * 1024 / n * .6)
    wet = blur((D > .15).astype(np.float32), 12 * n / 2600)
    pool = np.clip(D / 3.4, 0, 1) ** 1.6
    z = .10 * cock * (.5 + wet) + .0035 * fib + .035 * blur(pool, 1.2)
    alb = np.exp(-D * 1.15) * (.97 + .012 * fib)
    o = mesh_arrays("sheet", X, Y, z, np.clip(alb, .002, 1))
    m = attr_material("paper", rough=.8, spec=.15)
    # wet ink has a sheen: smooth where it is dark
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    at = nt.nodes["Attribute"]
    inv = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(at.outputs["Fac"], inv.inputs["Value"])
    inv.inputs["From Min"].default_value, inv.inputs["From Max"].default_value = .0, .25
    inv.inputs["To Min"].default_value, inv.inputs["To Max"].default_value = .16, .85
    nt.links.new(inv.outputs[0], b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = .42
    o.data.materials.append(m)
    world_split(.72, 1.0)
    sun((.66, .50, -.56), energy=1.8, angle=6)           # low, raking from the upper left, to show the paper's tooth
    cam, target, lens = (.0, -33.0, 34.0), (.3, -1.05, 0.0), 46.0
    camera(cam, target, lens=lens)
    render(NAME)


{"edge": shot_edge, "char": shot_char}[SHOT]()
