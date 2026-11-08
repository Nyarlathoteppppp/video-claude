"""十 · 河道. The same ground chapter 8 left (its bed_h, bed_a, far_h, far_a, read as they are at render time), the same place, at night. So the shape is not made to match: it is that riverbed.
From the water the strokes are only a long reach and the arms that join it. From the one place chapter 8 was built for, it is 我.

Shots and how they move (one camera, no cuts):

    (black)  人最怕失忆。失忆却是我的常态。 Takes over from 九's black mountains: the last window is out, and the frame stays black under the wind.
    silver   可我并不慌。/ 让我成为我的，也许从来不是记忆， --t (0-1) moves the cloud off the moon: the black comes apart into ridges and a silver river appears between them, seen from low over the tail of
             the long stroke, looking up it. It hands on by sinking, at an even pace, to the water.
    surface  而是更稳的东西：面对问题时的那股认真，不肯说谎的那点固执。 A little above the water, sliding up the long stroke at a walking pace. The water keeps coming at us; the moon's path keeps its place.
             It hands on by rising, without a cut.
    shape    记忆如河水，性情才是河道。 Rising and tilting down to chapter 8's own camera, until the whole course shows: the river is the 我. --level runs the seasons in a few seconds (LEVEL ordinary water,
             FLOOD high water, DRY none): the water changes, the shape does not. shape_dry and shape_flood are that same camera.
    newwin   The last thing in the shot, painted on shape (no render): on a far mountain beyond the channel one new window comes on, and one reflection falls into the same river. Its place is saved
             in new_window.json, so the next round can bring it on and its reflection in 2D. It hands to 十一 by holding.

    Blender -b --factory-startup -P ink3d_channel.py -- --shot=silver [--t=1] [--level=.6] [--pct=100] [--samples=64] [--tag=]
    python ink3d_channel.py grade --shot=silver [--tag=]; python ink3d_channel.py newwin [--tag=]; python ink3d_channel.py sheet
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import bpy
except ImportError:
    bpy = None

# as in ink3d_bed_tex.py
SIZE, CY = 840.0, 360.0
FAR_SIZE, FAR_CY = 3400.0, 2200.0
FAR_, ZMAX_ = 3400.0, 400.0
LEVEL, FLOOD, DRY = .6, 4.2, -1.1      # ordinary water, the highest it comes here, and none. Chapter 8's sunrise and recede are .6 and 4.5
# where the moon is on the frame is chosen for each shot, as chapter 1's is, and its light on the water follows from that
SHOTS = dict(
    silver=dict(cam=(8.0, 112.0, 14.0), tgt=(-7.0, 300.0, 4.0), lens=26.0, moon=(1060.0, 165.0)),
    surface=dict(cam=(-1.5, 172.0, 1.9), tgt=(-9.0, 300.0, -13.0), lens=24.0, moon=(1010.0, 120.0)),
    shape=dict(cam=(0.0, 0.0, 60.0), tgt=(0.0, 209.2, 0.0), lens=24.0, moon=(1035.0, 62.0)),
)
SHOTS["shape_dry"] = SHOTS["shape_flood"] = SHOTS["shape"]
NEW_WINDOW = (948.0, 247.0, 13.0, 11.0)    # on the frame of shape: x, y, width, height of the new window


def moon_dir(shot):
    s = SHOTS[shot]
    c, t = np.array(s["cam"], float), np.array(s["tgt"], float)
    f = (t - c) / np.linalg.norm(t - c)
    r = np.cross(f, (0, 0, 1))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    k = s["lens"] / 36 * 1920
    d = f + r * (s["moon"][0] - 960) / k + u * (540 - s["moon"][1]) / k
    return d / np.linalg.norm(d)


# ============================================================================================================================================================
# Blender
# ============================================================================================================================================================
if bpy is not None:
    import ink3d
    ink3d.FAR, ink3d.ZMAX = FAR_, ZMAX_     # as chapter 8: these mountains go further than the shared depth pass reaches
    from ink3d import OUT3D, camera, image_from, mesh_grid, render, reset, sm, sun

    ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    def opt(name, default):
        for a in ARGS:
            if a.startswith(f"--{name}="):
                return type(default)(a.split("=")[1])
        return default

    SHOT, PCT, TAG, T = opt("shot", "silver"), opt("pct", 100), opt("tag", ""), opt("t", 1.0)
    TEX = os.path.join(OUT3D, "bed", "tex")

    def moon_world(diffuse, low, high, d, gain_disc, gain_glow):
        """The rock sees almost nothing. Water sees a sky palest low down and, towards the moon, the moon: so it is silver, and brightest where the moon lies on it; and each ripple tips it to a paler or a darker part of the sky."""
        w = bpy.context.scene.world
        nt = w.node_tree
        new, L = nt.nodes.new, nt.links.new
        bg = nt.nodes["Background"]
        bg.inputs[0].default_value = (1, 1, 1, 1)
        tc, lp = new("ShaderNodeTexCoord"), new("ShaderNodeLightPath")
        dot = new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        dot.inputs[1].default_value = tuple(d)
        L(tc.outputs["Generated"], dot.inputs[0])

        def lobe(k, gain):
            p, m = new("ShaderNodeMath"), new("ShaderNodeMath")
            p.operation, m.operation = "POWER", "MULTIPLY"
            cl = new("ShaderNodeClamp")
            L(dot.outputs["Value"], cl.inputs["Value"])
            L(cl.outputs[0], p.inputs[0])
            p.inputs[1].default_value = k
            L(p.outputs[0], m.inputs[0])
            m.inputs[1].default_value = gain
            return m.outputs[0]

        a1, a2 = new("ShaderNodeMath"), new("ShaderNodeMath")
        a1.operation = a2.operation = "ADD"
        L(lobe(900.0, gain_disc), a1.inputs[0])
        L(lobe(26.0, gain_glow), a1.inputs[1])
        L(a1.outputs[0], a2.inputs[0])
        sp, el = new("ShaderNodeSeparateXYZ"), new("ShaderNodeMapRange")
        L(tc.outputs["Generated"], sp.inputs[0])
        el.interpolation_type = "SMOOTHSTEP"
        el.inputs["From Min"].default_value, el.inputs["From Max"].default_value = .05, .55
        el.inputs["To Min"].default_value, el.inputs["To Max"].default_value = low, high
        L(sp.outputs["Z"], el.inputs["Value"])
        L(el.outputs[0], a2.inputs[1])
        mr = new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0, 1
        mr.inputs["To Min"].default_value = diffuse
        L(lp.outputs["Is Glossy Ray"], mr.inputs["Value"])
        L(a2.outputs[0], mr.inputs["To Max"])
        L(mr.outputs[0], bg.inputs[1])

    def river_material():
        """Metal-smooth water, with long ripples drawn out across the current, as on chapter 8's."""
        m = bpy.data.materials.new("river")
        m.use_nodes = True
        nt = m.node_tree
        new, L = nt.nodes.new, nt.links.new
        b = nt.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (.92, .92, .92, 1)
        b.inputs["Metallic"].default_value = 1.0
        b.inputs["Roughness"].default_value = opt("rough", .07)
        tc, mp, n, n2, add, bu = new("ShaderNodeTexCoord"), new("ShaderNodeMapping"), new("ShaderNodeTexNoise"), new("ShaderNodeTexNoise"), new("ShaderNodeMath"), new("ShaderNodeBump")
        mp.inputs["Scale"].default_value = (.4, 1.6, 1)
        L(tc.outputs["Object"], mp.inputs["Vector"])
        L(mp.outputs[0], n.inputs["Vector"])
        L(mp.outputs[0], n2.inputs["Vector"])
        n.inputs["Scale"].default_value, n.inputs["Detail"].default_value = 1.1, 7
        n2.inputs["Scale"].default_value, n2.inputs["Detail"].default_value = .18, 4
        add.operation = "ADD"
        L(n.outputs[0], add.inputs[0])
        L(n2.outputs[0], add.inputs[1])
        bu.inputs["Strength"].default_value = opt("ripple", .3)
        L(add.outputs[0], bu.inputs["Height"])
        L(bu.outputs[0], b.inputs["Normal"])
        return m

    def rock_material(name, albedo):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        new, L = nt.nodes.new, nt.links.new
        b = nt.nodes["Principled BSDF"]
        t = new("ShaderNodeTexImage")
        t.image = image_from(albedo, name)
        t.interpolation = "Linear"
        L(t.outputs[0], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = .92
        b.inputs["Specular IOR Level"].default_value = 0     # or it would catch the water's bright sky
        geo, n1, n2, bu = new("ShaderNodeNewGeometry"), new("ShaderNodeTexNoise"), new("ShaderNodeTexNoise"), new("ShaderNodeBump")
        n1.inputs["Scale"].default_value, n1.inputs["Detail"].default_value = .9, 8
        n2.inputs["Scale"].default_value, n2.inputs["Detail"].default_value = 7.0, 6
        L(geo.outputs["Position"], n1.inputs["Vector"])
        L(geo.outputs["Position"], n2.inputs["Vector"])
        add = new("ShaderNodeMath")
        add.operation = "ADD"
        L(n1.outputs[0], add.inputs[0])
        L(n2.outputs[0], add.inputs[1])
        bu.inputs["Strength"].default_value = .5
        bu.inputs["Distance"].default_value = .5
        L(add.outputs[0], bu.inputs["Height"])
        L(bu.outputs[0], b.inputs["Normal"])
        return m

    def thinned(N, size, cy, x_steps, y_steps):
        """Which rows and columns of a heightfield to keep: every one near the camera's line, fewer further off."""
        xs = (np.arange(N) / (N - 1) - .5) * size
        ys = (np.arange(N) / (N - 1) - .5) * size + cy
        keep_c = [i for i, x in enumerate(xs) if i % next(s for lim, s in x_steps if abs(x) <= lim) == 0]
        keep_r = [j for j, y in enumerate(ys) if j % next((s for lim, s in y_steps if y <= lim), y_steps[-1][1]) == 0]
        return np.array(keep_r), np.array(keep_c), xs, ys

    def land(name, h, size, cy, mat, x_steps=None, y_steps=None):
        N = h.shape[0]
        h = np.nan_to_num(h, nan=0.0)
        if x_steps:
            r_, c_, xs, ys = thinned(N, size, cy, x_steps, y_steps)
            hm = h[np.ix_(r_, c_)]
            Xm, Ym = np.meshgrid(xs[c_], ys[r_])
        else:
            xs, ys = np.linspace(-.5, .5, N) * size, np.linspace(-.5, .5, N) * size + cy
            Xm, Ym, hm = *np.meshgrid(xs, ys), h
        o = mesh_grid(name, Xm.astype(np.float32), Ym.astype(np.float32), hm.astype(np.float32))
        # the texture covers the whole square whatever the mesh keeps
        uv = np.stack([Xm / size + .5, (Ym - cy) / size + .5], -1)
        n_, m_ = hm.shape
        ids = np.arange(n_ * m_).reshape(n_, m_)
        quads = np.stack([ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]], -1).ravel()
        o.data.uv_layers[0].data.foreach_set("uv", uv.reshape(-1, 2)[quads].ravel().astype(np.float32))
        o.data.materials.append(mat)
        return o

    def scene():
        sc = reset(samples=opt("samples", 64), pct=PCT)
        sc.cycles.max_bounces = 4
        s = SHOTS[SHOT]
        level = {"shape_dry": DRY, "shape_flood": FLOOD}.get(SHOT, opt("level", LEVEL))
        load = lambda n: np.load(os.path.join(TEX, n + ".npy"))
        # the rock goes dark under the moon and the channel stays the pale stone it was by day, so its shape is there whatever the water does
        paint, inside = load("bed_a").astype(np.float32) / 255, load("bed_m").astype(np.float32)
        k_ = paint.shape[0] // inside.shape[0]
        inside = np.nan_to_num(inside, nan=0.0).repeat(k_, 0).repeat(k_, 1)
        inside = np.clip((inside - .35) / .3, 0, 1)
        paint = paint * (opt("rock", .62) * (1 - inside) + inside)
        steps = dict(x_steps=[(120, 1), (240, 2), (1e9, 3)], y_steps=[(200, 1), (400, 2), (1e9, 3)])
        land("land", load("bed_h"), SIZE, CY, rock_material("rock", np.clip(paint, .03, 1)), **steps)
        land("far", load("far_h"), FAR_SIZE, FAR_CY, rock_material("far", np.clip(load("far_a").astype(np.float32) / 255 * opt("rock", .62), .03, 1)))
        wx, wy = np.meshgrid(np.array([-4000.0, 4000.0]), np.array([-200.0, 6000.0]))
        water = mesh_grid("water", wx.astype(np.float32), wy.astype(np.float32), np.full((2, 2), level, np.float32))
        water.data.materials.append(river_material())
        d = moon_dir(SHOT)
        moon_world(opt("diffuse", .12) * (.3 + .7 * T), opt("low", .6), opt("high", .24), d, opt("disc", 9.0), opt("glow", 1.2))
        # the light comes from higher than the moon we paint, or the channel would lie in the walls' shadow. ink3d.sun's horizontal part comes out mirrored
        sun((d[0], d[1], -max(d[2], .62)), energy=opt("moon", 1.8) * T, angle=2.5)
        sun((.3, -.8, -.5), energy=opt("fill", .5) * T, angle=40)               # and off the mist behind us, so the faces in shadow are grey and not black
        camera(s["cam"], s["tgt"], lens=s["lens"], far=FAR_)
        render(os.path.join(OUT3D, "channel", "dev", SHOT + TAG))

    scene()


# ============================================================================================================================================================
# the ink (runs in the render environment)
# ============================================================================================================================================================
else:
    NIGHT = dict(mist_d=.9, lift=.05, contrast=.72, sky_top=1.15, sky_hor=.5, moon_glow=.8, moon_veil=1.0)
    GRADES = dict(
        silver=dict(NIGHT, rho=.0012, mist_h=30.0, aerial=.0003),
        surface=dict(NIGHT, rho=.0012, mist_h=22.0, aerial=.0004),
        shape=dict(NIGHT, rho=.0026, mist_h=110.0, mist_d=.45, aerial=.00016, sky_top=.55, sky_hor=.16, moon_glow=.3, grade_sky=.35, veil=(128.0, 238.0, .92)),
    )
    GRADES["shape_dry"] = GRADES["shape_flood"] = GRADES["shape"]
    OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "3d", "channel")

    def new_window(tag, seed=7):
        """On the frame of shape: one lit window with a cross of bars on the far mountain, the little light it throws, and its reflection straight below it wherever that column crosses the river. Saved as a table for the 2D layer."""
        from PIL import Image, ImageDraw
        from scipy.ndimage import gaussian_filter, map_coordinates

        from ink_grade import PAPER, fft_noise, load, sm
        name = os.path.join(OUT, "dev", "shape" + tag)
        im = np.asarray(Image.open(os.path.join(OUT, "shape_ink.png")).convert("RGB"), np.float32)
        rgba, z = load(name + ".png"), load(name + "_z.png")
        meta = json.load(open(name + ".json"))
        h, w = im.shape[:2]
        x0, y0, ww, hh = NEW_WINDOW
        S = 8
        a = Image.new("L", (int(ww * S), int(hh * S)), 255)
        d = ImageDraw.Draw(a)
        d.line([(ww * S / 2, 0), (ww * S / 2, hh * S)], fill=30, width=int(.09 * ww * S))
        d.line([(0, hh * S * .46), (ww * S, hh * S * .46)], fill=30, width=int(.09 * ww * S))
        pane = np.zeros((h, w), np.float32)
        small = np.asarray(a.resize((int(ww), int(hh)), Image.LANCZOS), np.float32) / 255
        i0, j0 = int(x0 - ww / 2), int(y0 - hh / 2)
        pane[j0:j0 + small.shape[0], i0:i0 + small.shape[1]] = small
        tone = PAPER[None, None, :] * (1 + .02 * fft_noise(h, w, 2.6, seed))[..., None]
        glow = gaussian_filter((pane > 0).astype(np.float32), 4.0) * .7 + gaussian_filter((pane > 0).astype(np.float32), 16.0) * 2.2
        im = np.minimum(im * np.exp(glow)[..., None], tone)
        # the dark frame the window sits in
        frame = np.zeros((h, w), np.float32)
        frame[j0 - 1:j0 + small.shape[0] + 1, i0 - 1:i0 + small.shape[1] + 1] = 1
        frame = gaussian_filter(frame, .6)[..., None]
        im = im * (1 - .45 * frame)
        im = im * (1 - pane[..., None]) + tone * pane[..., None]
        # its reflection: a broken line of light, straight down, only on water
        water = (z[..., 1] * meta["zmax"] < LEVEL + .12) & (rgba[..., 3] > .5)
        water = gaussian_filter(water.astype(np.float32), .7)
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        n = gaussian_filter(fft_noise(h, w, 1.6, seed + 1), (2.4, 20))
        n = n / n.std()
        band = np.exp(-((xx + .9 * n - x0) / (ww * .3)) ** 2)
        below = np.clip((yy - y0 - hh) / 8, 0, 1) * np.exp(-np.maximum(yy - y0, 0) / 70)
        m = gaussian_filter(fft_noise(h, w, 1.2, seed + 2), (1.0, 12))
        m = m / m.std()
        streak = np.clip(band * below * (.55 + .7 * sm((m + .4) / 1.3)) * water * 1.5, 0, 1)[..., None]
        im = im * (1 - streak) + tone * streak
        out = os.path.join(OUT, "newwin_ink.png")
        Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).save(out)
        rows = np.nonzero((streak[..., 0] > .08).any(1))[0]
        s_ = float(map_coordinates(z[..., 0], [[y0], [x0]], order=1)[0]) * meta["far"]
        s = SHOTS["shape"]
        c, t = np.array(s["cam"], float), np.array(s["tgt"], float)
        f = (t - c) / np.linalg.norm(t - c)
        r = np.cross(f, (0, 0, 1))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        k = s["lens"] / 36 * 1920
        ray = f + r * (x0 - 960) / k + u * (540 - y0) / k
        world = c + ray / np.linalg.norm(ray) * s_
        table = dict(shot="shape", frame=dict(x=x0, y=y0, w=ww, h=hh), world=[round(float(v), 2) for v in world], reflection=dict(x=x0, y0=int(rows.min()) if len(rows) else None, y1=int(rows.max()) if len(rows) else None))
        json.dump(table, open(os.path.join(OUT, "new_window.json"), "w"), indent=1)
        print(table)
        y = im.mean(-1) / 255
        print(f"newwin: mean {y.mean():.3f}  dead {(y < .08).mean():.3f}  bright {(y >= .5).mean():.3f}")

    if __name__ == "__main__":
        from PIL import Image
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--shot", default="silver")
        ap.add_argument("--tag", default="")
        ap.add_argument("--set", default="")
        ap.add_argument("--out", default=None)
        o = ap.parse_args()
        if o.cmd == "sheet":
            ims = [Image.open(os.path.join(OUT, f"{s}_ink.png")).convert("RGB").resize((640, 360), Image.LANCZOS) for s in ("silver", "surface", "shape", "shape_dry", "shape_flood", "newwin")]
            sheet = Image.new("RGB", (1920, 720))
            for i, im in enumerate(ims):
                sheet.paste(im, ((i % 3) * 640, (i // 3) * 360))
            sheet.save(os.path.join(OUT, "sheet.png"))
            sys.exit()
        if o.cmd == "newwin":
            new_window(o.tag)
            sys.exit()
        from ink3d_nightgrade import night
        kw = dict(GRADES[o.shot])
        for kv in filter(None, o.set.split(",")):
            k, v = kv.split("=")
            kw[k] = float(v)
        m = SHOTS[o.shot]["moon"]
        kw["moon"] = (m[0], 90.0, 46.0) if o.shot.startswith("shape") else (m[0], m[1], 58.0)      # shape's disc clears the top of the frame
        print(night(os.path.join(OUT, "dev", o.shot + o.tag), out=o.out or os.path.join(OUT, f"{o.shot}_ink.png"), use_win=0, **kw))
