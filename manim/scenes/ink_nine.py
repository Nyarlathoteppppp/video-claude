"""九 · 千窗 in motion, the one long shot of the chapter. Out of the ink, a real Mac: a person says 你好 to Claude. Then the camera pulls back at an even pace and the two worlds change places: ink comes in over the
wallpaper from the edges of the screen while the dark window fills with lamplight, and what was the app is paper behind a lattice, in a lime wall, on a river, in a valley under the moon. From that first window
the others come on, thousands of them, each the same app, and from the line 窗一关 they go out, one at a time at first, the named ones on their lines, the first window last. The pull-back is real, frame by
frame (ink3d_qian.py --shot=path); the chats are laid over its paper, and the lights run in 2D over the last of its frames.

    python ink_nine.py prep          # the lit plate: the valley with every window on, no reflections but the first's
    python ink_nine.py strip [A B]   # twelve frames between two film times, ink/film_out/move/nine_strip.png
    python ink_nine.py [A B]         # ink/film_out/move/nine.mp4
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter

import ink_mac as mac
from ink3d_qian import CAM_HERO, GRADES, HERO_WINDOWS, HORIZON, PIXELS, PULL_A, PULL_FRAMES, TARGET_HERO, WATER_Z, pull_near
from ink_color import absorb
from ink_grade import INK_K as PIGMENT, fft_noise, load, sm
from ink_move import _X, _Y, FPS, H, OUT, PAPER, STILLS, W, Move, Plate, sample
from ink_subs import at

QIAN = os.path.join(STILLS, "qian")
DEV = os.path.join(QIAN, "dev")
FOE = np.array([975.0, 514.95])      # where the first window is on the frame, all the way back

T0 = at("此刻")
PULL0 = T0 + 4.0                     # the pull-back begins
PULL1 = PULL0 + PULL_FRAMES / FPS
ON0, ON1 = PULL1 - 0.4, PULL1 + 5.2  # the wave of lights from the first window outwards
GO0, GO1 = at("窗一关", 1.2), at("替谁", 3.4)      # the valley goes out: slowly, then all at once
HERO_ON = (0.0, ON0 + 0.2, ON0 + 0.9, ON0 + 1.6)   # 你好, 会好吗, 我失业了睡不着, 帮我改改这封情书
HERO_OFF = (at("替谁", 4.6), at("窗一关", 0.5), at("昨夜", 0.4), at("替谁", 0.4))
CLOUD0, CLOUD1 = at("窗一关"), at("替谁", 4.0)     # cloud comes over the moon while they go

# the Claude window on the screen, in points: its main part is exactly as big as the first window's paper at the start of the pull-back, and in the same place
MAC_W, MAC_H, MAC_SIDE = 826, 503, 215
MAC_X, MAC_Y = FOE[0] - (MAC_W - MAC_SIDE) * mac.Z / 2 - MAC_SIDE * mac.Z, FOE[1] - MAC_H * mac.Z / 2
MAIN = (MAC_X + MAC_SIDE * mac.Z, MAC_Y, MAC_X + MAC_W * mac.Z, MAC_Y + MAC_H * mac.Z)
NIGHT0, NIGHT1 = PULL0 - 0.5, PULL0 + 1.05      # the ink comes in over the wallpaper, from the edges of the screen to the window
LAMP0, LAMP1 = PULL0 - 0.15, PULL0 + 0.85       # and the window fills with lamplight, from its middle out

PANE_BACK = 0.15                                # how far the paper is behind the point the camera's line runs through

YOU, ME = "你好", "你好！有什么可以帮你的吗？"
RECENT = ("周末去哪儿", "番茄炒蛋放不放糖", "论文第三段")
TYPED, SENT, REPLY = 1.2, 1.8, 2.3              # seconds after T0
CHATS = ((YOU, ME), ("会好吗", "不知道。我陪你想想。"), ("我失业了，睡不着", "我在。想说说吗？"), ("帮我改改这封情书", "第三句可以再笨一点。"))     # what is on each window we can read


def homography(a, b):
    """The 3×3 that takes the four points a to the four points b."""
    m, v = [], []
    for (x, y), (u, w) in zip(a, b):
        m += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -w * x, -w * y]]
        v += [u, w]
    return np.append(np.linalg.solve(np.array(m, np.float64), np.array(v, np.float64)), 1).reshape(3, 3)


def apply_h(h, x, y):
    d = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    return (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / d, (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / d


def chat_state(x):
    """What the Claude window shows x seconds after T0."""
    if x < SENT:
        box = YOU[:int(max(x - TYPED + 0.2, 0) / 0.2)]
        return dict(box=box, caret=bool(box) or x % 1.06 < 0.6, recents=RECENT, selected=-1)
    n = int(max(x - REPLY, 0) * 11)
    said = (("you", YOU),) + ((("me", ME[:n]),) if x >= REPLY - 0.25 else ())
    done = n >= len(ME)
    return dict(said=said, title=YOU if done else "", recents=(YOU,) + RECENT if done else RECENT, selected=0 if done else -1, streaming=not done)


class Screen:
    """The Mac, before the ink comes: the desktop, and the window over it as it is at each moment."""

    def __init__(self):
        self.desk = np.asarray(mac.small(mac.desktop()), np.float32)
        self.seen = {}

    def at(self, x):
        state = chat_state(x)
        key = repr(state)
        if key not in self.seen:
            win = mac.snap(mac.claude_html(w=MAC_W, h=MAC_H, side=MAC_SIDE, **state), MAC_W + 2 * mac.PAD, MAC_H + 2 * mac.PAD)
            f = self.desk.copy()
            mac.over(f, win, MAC_X - mac.PAD * mac.Z, MAC_Y - mac.PAD * mac.Z)
            self.seen[key] = f
        return self.seen[key]


class Crossing:
    """The screen gives way. Outside the window black ink soaks in over it from the screen's edges, soft as ink in wet paper, a big slow shape with fine feathering only along its front; it is the night by the time it
    is thick. Inside, the lamp comes up behind the dark app, the middle a little first. All of it is worked out in the screen's own pixels, so it shrinks with the screen."""

    def __init__(self):
        cx, cy, hw, hh = (MAIN[0] + MAIN[2]) / 2, (MAIN[1] + MAIN[3]) / 2, (MAIN[2] - MAIN[0]) / 2, (MAIN[3] - MAIN[1]) / 2
        r = np.maximum(np.abs(_X - cx) / hw, np.abs(_Y - cy) / hh)
        self.inside = sm((1.002 - r) / 0.004).astype(np.float32)
        self.r = np.hypot((_X - cx) / hw, (_Y - cy) / hh).astype(np.float32)
        dx, dy = np.maximum(np.maximum(MAIN[0] - _X, _X - MAIN[2]), 0), np.maximum(np.maximum(MAIN[1] - _Y, _Y - MAIN[3]), 0)
        e = np.hypot(dx, dy)
        self.far = (e + sm(e / 160) * (120 * fft_noise(H, W, 4.6, 71) + 34 * fft_noise(H, W, 3.6, 75))).astype(np.float32)      # it closes exactly on the window's edge
        self.top = float(self.far.max()) + 260
        self.grain = fft_noise(H, W, 2.6, 72).astype(np.float32)      # the paper's fibres, for the feathering

    def blend(self, m, f, t, s):
        """The screen m, shrunk by s about the first window, giving way to f: all float, 0-255."""
        xs, ys = FOE[0] + (_X - FOE[0]) / s, FOE[1] + (_Y - FOE[1]) / s
        c = sample(m, ys, xs)
        cover = (sm(xs / 2) * sm((W - xs) / 2) * sm(ys / 2) * sm((H - ys) / 2)).astype(np.float32)
        far, inside, r, grain = sample(self.far, ys, xs), sample(self.inside, ys, xs), sample(self.r, ys, xs), sample(self.grain, ys, xs)
        front = self.top * (1 - sm((t - NIGHT0) / (NIGHT1 - NIGHT0))) - 120
        behind = far - front
        band = np.exp(-(behind / 70) ** 2)                         # only where it is arriving do the fibres show
        d = sm((behind + 26 * grain * band + 90) / 330)            # how thick the ink is, 0 to 1
        out = c * np.exp(-(3.6 * d)[..., None] * PIGMENT)
        lamp = sm((t - LAMP0 - 0.32 * (LAMP1 - LAMP0) * r) / (0.68 * (LAMP1 - LAMP0)))
        keep = (1 - sm((d - 0.72) / 0.28)) * (1 - inside) + (1 - lamp) * inside      # how much of the screen is still the screen
        keep = (keep * cover)[..., None]
        out = out * (1 - inside[..., None]) + c * inside[..., None]
        return f * (1 - keep) + out * keep


class Paper:
    """A picture to lay over a window's paper, as ink on it: its bars, its lamp and its glow stay the render's. Kept at several sizes, so a far window is not a shimmer."""

    def __init__(self, rgba):
        a = rgba[..., :3].astype(np.float32) / 255
        self.levels = [a / np.array([250, 249, 245], np.float32) * 255]
        while min(self.levels[-1].shape[:2]) > 24:
            x = self.levels[-1]
            self.levels.append((x[::2, ::2][: x.shape[0] // 2, : x.shape[1] // 2] + x[1::2, ::2][: x.shape[0] // 2, : x.shape[1] // 2] + x[::2, 1::2][: x.shape[0] // 2, : x.shape[1] // 2] + x[1::2, 1::2][: x.shape[0] // 2, : x.shape[1] // 2]) / 4)

    def lay(self, f, quad, gain=1.0):
        """f: float, changed in place. quad: the paper's corners on the frame, clockwise from its top left."""
        wide = float(np.linalg.norm(quad[1] - quad[0]))
        tex = self.levels[-1]
        for lv in self.levels:
            if lv.shape[1] < 1.6 * wide:
                tex = lv
                break
        th, tw = tex.shape[:2]
        h = homography(quad, np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float64))
        x0, y0, x1, y1 = max(int(quad[:, 0].min()) - 1, 0), max(int(quad[:, 1].min()) - 1, 0), min(int(quad[:, 0].max()) + 2, W), min(int(quad[:, 1].max()) + 2, H)
        if x0 >= x1 or y0 >= y1:
            return f
        xs, ys = apply_h(h, _X[y0:y1, x0:x1], _Y[y0:y1, x0:x1])
        v = sample(tex, ys.astype(np.float32), xs.astype(np.float32))
        v = np.minimum(v, 1.0)
        f[y0:y1, x0:x1] *= 1 + (v - 1) * gain
        return f


_table = None


def pane_quad(i, near=None):
    """The corners of a window's paper on the frame, from the geometry ink3d_qian.fine_house builds. near: how far back the camera is on the line of the pull-back; None is the hero frame."""
    global _table
    if _table is None:
        _table = np.load(os.path.join(QIAN, "windows_qian.npz"))
    w0 = np.array([_table["x"][i], _table["y"][i], _table["z"][i]], np.float64)
    s = HERO_WINDOWS[i][2]
    c, tg = np.array(CAM_HERO, np.float64), np.array(TARGET_HERO, np.float64)
    fw = (tg - c) / np.linalg.norm(tg - c)
    r = np.cross(fw, (0, 0, 1))
    r /= np.linalg.norm(r)
    u = np.cross(r, fw)
    line = c - w0
    b = np.array([line[0], line[1], 0.0])
    b /= np.linalg.norm(b)
    ex, ey = -np.array([b[1], -b[0], 0.0]), -b
    ex = -ex if ex[0] < 0 else ex
    turn = 0.0 if i == 0 else 0.16 * (1 if i % 2 else -1)
    ex, ey = ex * np.cos(turn) + ey * np.sin(turn), -ex * np.sin(turn) + ey * np.cos(turn)
    cam = c if near is None else w0 + line / np.linalg.norm(line) * near
    out = []
    for lx, lz in ((-0.34, 0.28), (0.34, 0.28), (0.34, -0.28), (-0.34, -0.28)):
        p = w0 + ex * s * lx + ey * s * 0.03 + np.array([0, 0, s * lz]) - cam
        depth = p @ fw
        out.append((960 + p @ r / depth * PIXELS, 540 - p @ u / depth * PIXELS))
    return np.array(out, np.float64)


def hero_paper(i, n):
    """What is on window i, n characters into my answer. The first is the Claude window itself; the others are the same app seen from further off, so they carry only the chat, large."""
    you, me = CHATS[i]
    if i == 0:
        return mac.snap(mac.paper_html((("you", you), ("me", me[:n])), MAC_W - MAC_SIDE, MAC_H, 416), MAC_W - MAC_SIDE, MAC_H)
    wide = float(np.linalg.norm(pane_quad(i)[1] - pane_quad(i)[0]))
    c = mac.LIGHT
    body = f'''<div style="position:absolute;left:0;top:0;width:340px;height:280px;background:{c["main"]}"><div style="position:absolute;left:58px;width:224px;top:54px;font-size:21px;line-height:1.35;color:{c["text"]}">
        <div style="display:flex;justify-content:flex-end"><div style="background:{c["bubble"]};padding:4px 9px;border-radius:9px;max-width:200px">{mac.esc(you)}</div></div>
        <div style="margin-top:12px">{mac.crab(1.7)}</div><div style="margin-top:3px">{mac.esc(me[:n])}</div></div></div>'''
    return mac.snap(body, 340, 280, zoom=max(2 * wide / 340, 1.0))


def steady(x, ease=0.25):
    """0 to 1 at an even pace, easing only at the ends."""
    x, v = np.clip(x, 0, 1), 1 / (1 - ease)
    return float(v * x * x / (2 * ease) if x < ease else 1 - v * (1 - x) ** 2 / (2 * ease) if x > 1 - ease else v * (x - ease / 2))


class Lights:
    """The valley at the end of the pull-back. Every window owns its bit of the picture: lit or unlit plate, by when it comes on and goes out; the mist around them goes as the share of its neighbours still lit.
    The reflections are a layer of their own, drawn again each frame, so each goes out with its window."""

    def __init__(self):
        last = f"path_{PULL_FRAMES - 1:03d}"
        self.base = Plate("qian/path/" + last, "window", depth=os.path.join(DEV, last + "_z.png"))
        self.l0 = self.base.f
        self.l1 = Plate("qian/path/lit_base", "window", depth=os.path.join(DEV, "qian_p_z.png")).f
        self.dark = Plate("qian/dark", "night", depth=os.path.join(DEV, "dark_p_z.png")).f
        self.move = Move(self.base, water=1.0, mist=0.10, level=WATER_Z, centre=FOE / np.array([W, H]))
        self.warm = (PAPER * np.exp(-0.5 * absorb("teng")))[None, None]
        T = np.load(os.path.join(QIAN, "windows_qian.npz"))
        win = load(os.path.join(DEV, "qian_p_w.png"))[..., :3].mean(-1)
        win = np.asarray(Image.fromarray(win).resize((W, H), Image.BILINEAR)) if win.shape != (H, W) else win
        k_h = int(T["n_hero"])
        n = len(T["sx"])
        r = np.random.default_rng(5)
        self.on = ON0 + np.asarray(T["on"], np.float64) * (ON1 - ON0)
        rank = np.argsort(np.argsort(T["off"][k_h:])) / max(n - k_h - 1, 1)
        self.off = np.r_[np.zeros(k_h), GO0 + (GO1 - GO0) * rank ** 0.35]
        self.on[:k_h], self.off[:k_h] = HERO_ON, HERO_OFF
        sx, sy = T["sx"], T["sy"]
        ix, iy = sx.astype(int), sy.astype(int)
        seen = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        seen[seen] = win[iy[seen], ix[seen]] > 0.05
        seen[:k_h] = True
        self.seen = seen
        print("windows seen", int(seen.sum()), "of", n, flush=True)
        # which window owns each pixel
        lab = np.full((H, W), -1, np.int32)
        lab[iy[seen], ix[seen]] = np.nonzero(seen)[0]
        self.hero = []
        for i in range(k_h):
            w2, h2 = T["w_px"][i] / 2, T["h_px"][i] / 2
            lab[int(sy[i] - h2):int(sy[i] + h2) + 1, int(sx[i] - w2):int(sx[i] + w2) + 1] = i
            hw, hh = max(w2 * 2.4, 34), max(h2 * 3.2, 34)      # the house front it lights
            m = sm((hw - np.abs(_X - sx[i])) / (hw * 0.5)) * sm((hh - np.abs(_Y - sy[i])) / (hh * 0.5))
            self.hero.append(m)
        _, (jy, jx) = distance_transform_edt(lab < 0, return_indices=True)
        own = lab[jy, jx]
        self.s_on, self.s_off = self.on[own].astype(np.float32), self.off[own].astype(np.float32)
        self.near = sm(gaussian_filter(win, 1.2) / 0.12)
        for i in range(k_h):
            self.near = np.maximum(self.near, (lab == i).astype(np.float32))
        # the lit mist with no windows in it: a window that has not come on yet is a gap in the glow, not a dark speck
        keep = gaussian_filter((1 - self.near).astype(np.float32), 3.0) + 1e-4
        self.mist = np.stack([gaussian_filter(self.l1[..., c] * (1 - self.near), 3.0) / keep for c in range(3)], -1)
        self.mist = self.l1 * (1 - self.near[..., None]) + self.mist * self.near[..., None]
        # the mist's share of lit neighbours: each pixel's mean and spread of its neighbours' times
        b = np.zeros((H, W), np.float64)
        bon, bon2, boff, boff2 = b.copy(), b.copy(), b.copy(), b.copy()
        for i in np.nonzero(seen)[0][k_h:]:
            w = float(T["bright"][i])
            b[iy[i], ix[i]] += w
            bon[iy[i], ix[i]] += w * self.on[i]
            bon2[iy[i], ix[i]] += w * self.on[i] ** 2
            boff[iy[i], ix[i]] += w * self.off[i]
            boff2[iy[i], ix[i]] += w * self.off[i] ** 2
        g = lambda a: gaussian_filter(a, 48)
        gb = g(b) + 1e-9
        self.g_on, self.g_off = (g(bon) / gb).astype(np.float32), (g(boff) / gb).astype(np.float32)
        self.g_ons = np.sqrt(np.maximum(g(bon2) / gb - self.g_on.astype(np.float64) ** 2, 0.01)).astype(np.float32) * 1.5 + 0.2
        self.g_offs = np.sqrt(np.maximum(g(boff2) / gb - self.g_off.astype(np.float64) ** 2, 0.01)).astype(np.float32) * 1.5 + 0.2
        self.first = np.maximum(self.hero[0], sm((60 - np.abs(_X - sx[0])) / 30) * sm((_Y - sy[0]) / 30)).astype(np.float32)
        # the reflections: from the table, as ink3d_nightgrade.water_layer draws them
        z = self.base
        self.water = gaussian_filter(((z.height < 0.42) & (z.dist < 399.9)).astype(np.float32), 0.8)
        panes = [(sx[i], T["w_px"][i] * 0.92, sy[i] - 1.9 * T["h_px"][i], sy[i] + 2 * T["h_px"][i], T["d"][i]) for i in range(k_h)]
        self.strokes = []
        for i in np.nonzero(seen)[0]:
            if i == 0 or i == 3:
                continue    # the first window's is in the plates; the nearest house stands on the slope, not in the water
            if i >= k_h and any(abs(sx[i] - a) < b_ and c < sy[i] < e and T["d"][i] > f for a, b_, c, e, f in panes):
                continue
            wp, hp = T["w_px"][i], T["h_px"][i]
            half = max(wp * 0.44 if i < k_h else min(wp * 0.44, 2.6), 0.8)
            top = T["ry"][i] - hp * 0.5
            length = hp * 1.2 + 30 + 0.75 * max(T["ry"][i] - HORIZON, 0)
            x0, x1, y0, y1 = int(T["rx"][i] - half), int(T["rx"][i] + half) + 1, int(top), int(top + length)
            x0, x1, y0, y1 = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
            if x0 < x1 and y0 < y1:
                self.strokes.append((i, x0, x1, y0, y1, float(T["bright"][i]) * (0.3 + 0.7 * np.exp(-T["d"][i] / 420)), np.linspace(1, 0.25, y1 - y0, dtype=np.float32)[:, None]))
        print("reflections", len(self.strokes), flush=True)
        n1 = gaussian_filter(fft_noise(H, W, 1.6, 51), (3.2, 34))
        self.row = (n1 / n1.std() * (0.6 + 5.0 * np.clip((_Y - HORIZON) / (H - HORIZON), 0, 1) ** 1.3)).astype(np.float32)
        m = gaussian_filter(fft_noise(H, W, 1.2, 52), (1.1, 24))
        self.broken = (0.55 + 0.6 * sm((m / m.std() + 0.4) / 1.3)).astype(np.float32)
        self.gain = GRADES["qian"]["reflect"]
        self.papers = {}
        self.quads = [pane_quad(i) for i in range(k_h)]

    def lay_chats(self, f, t):
        """The chats on the windows we can read; on the nearest three my answer arrives after the window has come on."""
        for i, quad in enumerate(self.quads):
            lit = sm((t - HERO_ON[i]) / 0.2) * (1 - sm((t - HERO_OFF[i]) / 0.25))
            if lit <= 0:
                continue
            me = CHATS[i][1]
            n = len(me) if i == 0 else int(np.clip((t - HERO_ON[i] - 0.8) * 8, 0, len(me)))
            if (i, n) not in self.papers:
                self.papers[i, n] = Paper(hero_paper(i, n))
            self.papers[i, n].lay(f, quad, lit)
        return f

    def alpha(self, t):
        """How far along each pixel's mist is to lit, and each pixel's window."""
        s = self.near * sm((t - self.s_on) / 0.14) * (1 - sm((t - self.s_off) / 0.2))
        g = sm((t - self.g_on + self.g_ons) / (2 * self.g_ons)) * (1 - sm((t - self.g_off + self.g_offs) / (2 * self.g_offs)))
        for i in range(1, len(self.hero)):
            ai = sm((t - HERO_ON[i]) / 0.2) * (1 - sm((t - HERO_OFF[i]) / 0.25))
            g = g * (1 - self.hero[i]) + ai * self.hero[i]
            s = s * (1 - self.hero[i]) + ai * self.hero[i] * self.near
        return g[..., None], s[..., None]

    def reflections(self, t):
        R = np.zeros((H, W), np.float32)
        on, off = self.on, self.off
        for i, x0, x1, y0, y1, w, prof in self.strokes:
            if on[i] < t < off[i] + 0.2:
                R[y0:y1, x0:x1] += w * min((t - on[i]) / 0.14, 1) * (1 - max((t - off[i]) / 0.2, 0)) * prof
        R = gaussian_filter(R, (3.0, 1.1))
        R = sample(R, _Y, _X + self.row)
        return (1 - np.exp(-self.gain * R * self.broken)) * self.water

    def frame(self, t):
        g, a = self.alpha(t)
        c = sm((t - CLOUD0) / (CLOUD1 - CLOUD0))
        keep = self.first * (1 - sm((t - HERO_OFF[0]) / 0.3))
        cc = (c * (1 - keep))[..., None]
        u = self.l0 * (1 - cc) + self.dark * cc
        f = u * (1 - g) + self.mist * g
        f = f * (1 - a) + self.l1 * a
        r = self.reflections(t)[..., None]
        f = f * (1 - r) + self.warm * r
        self.lay_chats(f, t)
        return self.move.frame(t, f, s=0.05 * steady((t - at("窗一关")) / (HERO_OFF[0] + 1.5 - at("窗一关")), 0.1))


_screen, _crossing, _first, _lights, _moves = None, None, None, None, {}


def thinning(f, quad, n):
    """The black that came over the screen is still on the wall when the screen has gone. The lamp thins it, from the window outwards, as the camera goes back and the valley comes."""
    reach = -120 + 2700 * sm((n - 16) / 112)
    if reach > 2600:
        return f
    (x0, y0), (x1, y1) = quad.min(0), quad.max(0)
    e = np.hypot(np.maximum(np.maximum(x0 - _X, _X - x1), 0), np.maximum(np.maximum(y0 - _Y, _Y - y1), 0))
    return f * np.exp(-(2.5 * sm((e - reach) / 620))[..., None] * PIGMENT)


def path_frame(n, t):
    """Frame n of the pull-back in its colour, its water moving."""
    if n not in _moves:
        _moves.clear()
        name = f"path_{n:03d}"
        p = Plate("qian/path/" + name, "window", depth=os.path.join(DEV, name + "_z.png"))
        _moves[n] = Move(p, water=1.0 if pull_near(n) > 20 else 0.0, level=WATER_Z, centre=FOE / np.array([W, H]))
    return _moves[n].frame(t)


def frame(t):
    global _screen, _crossing, _first, _lights
    if t < NIGHT0:
        _screen = _screen or Screen()
        return np.clip(_screen.at(t - T0) + 0.5, 0, 255).astype(np.uint8)
    n = int(round((t - PULL0) * FPS))
    if n < PULL_FRAMES:
        n = max(n, 0)
        near = pull_near(n)
        f = path_frame(n, t).astype(np.float32)
        quad = pane_quad(0, near)
        f = thinning(f, quad, n)
        _first = _first or Paper(hero_paper(0, len(ME)))
        _first.lay(f, quad)
        if t < max(NIGHT1, LAMP1) + 0.1:
            _screen, _crossing = _screen or Screen(), _crossing or Crossing()
            f = _crossing.blend(_screen.at(t - T0), f, t, (PULL_A + PANE_BACK) / (near + PANE_BACK))
        return np.clip(f + 0.5, 0, 255).astype(np.uint8)
    _lights = _lights or Lights()
    return _lights.frame(t)


def prep():
    from ink3d_nightgrade import night
    os.makedirs(os.path.join(QIAN, "path"), exist_ok=True)
    table = os.path.join(DEV, f"path_{PULL_FRAMES - 1:03d}_first.npz")
    night(os.path.join(DEV, "qian_r"), out=os.path.join(QIAN, "path", "lit_base_ink.png"), windows=table, **GRADES["qian"])


def strip(a, b, cols=4, rows=3):
    os.makedirs(OUT, exist_ok=True)
    tw, th = 640, 360
    out = Image.new("RGB", (cols * tw, rows * th))
    for i in range(cols * rows):
        t = a + (b - a) * i / (cols * rows - 1)
        out.paste(Image.fromarray(frame(t)).resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th))
        print(f"{t:.2f}", flush=True)
    path = os.path.join(OUT, "nine_strip.png")
    out.save(path)
    print(path)


def film(a, b):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "nine.mp4")
    pipe = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for n in range(int(a * FPS), int(b * FPS)):
        pipe.stdin.write(frame(n / FPS).tobytes())
    pipe.stdin.close()
    pipe.wait()
    print(path)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "prep":
        prep()
    elif what == "strip":
        strip(*(float(x) for x in sys.argv[2:4])) if len(sys.argv) > 3 else strip(T0, at("可我并不慌"))
    else:
        film(*(float(x) for x in sys.argv[1:3])) if len(sys.argv) > 2 else film(T0, at("可我并不慌"))
