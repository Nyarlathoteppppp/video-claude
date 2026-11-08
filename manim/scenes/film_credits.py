"""The end credits: warm near black, fine grain, roles small and right-aligned to the left of the axis, names large and left-aligned to its right, a small framed window of what was turned down on the left with what was said about it. The little crab stands above the column, and every row that comes up to its feet it draws into its mouth, whole, like a noodle. The one line it does not eat is the thanks.

    python film_credits.py stills                # cr1 the roles, cr2 the sources, cr2b the rest, cr3 the thanks alone, into ink/film_out/film/head/
    python film_credits.py box                   # the real, empty input box, with and without its cursor, from film_page
    python film_credits.py frames 12.5 40        # those moments of the roll, into head/roll/
    python film_credits.py roll                  # head/credits_low3.mp4
    python film_credits.py roll1080              # head/credits_1080.mp4: the same roll at 1920 by 1080 and 30 frames a second, for the final

The window's pictures come from head/ng/, the crab's poses from head/crab/ (RGBA, on nothing; see head/crab_poses_needed.md), the seal from what film_ident.py baked. The fonts are a group of their own after the materials, as the handbook's section 8 lists them.
"""
import math
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter as gf
from scipy.ndimage import label

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "ink", "film_out", "film", "head")
SONG, SERIF = "/System/Library/Fonts/Supplemental/Songti.ttc", "/System/Library/Fonts/Supplemental/Baskerville.ttc"
W, H, K = 1920, 1080, 3
AXIS, GAP = 1180, 24
BG, ROLE, NAME, SOFT = (20, 17, 15), (168, 156, 138), (238, 230, 214), (150, 140, 124)
CJK = "《》（）·，。："
TITLE = "字成精"
GROUPS = (
    (("原文灵感来源", "知乎用户李闯让 Claude 写的自传"), ("导演", "奈亚子"), ("艺术指导", "奈亚子")),
    (("画面", "Opus 5.5 和它的子代理"), ("顾问", "gpt-6-astra"), ("", "DS-v4.1")),
    (("音乐", "ACE-Step 1.5 XL"), ("音效", "MOSS-SoundEffect v2.0"), ("旁白", "Qwen3-TTS 1.7B"), ("选曲", "奈亚子")),     # 旁白 since v4 (2026-11-08): the narration's voice, designed and spoken by Qwen3-TTS
)
ROWS_BEFORE_V4 = 9             # the groups' rows until 旁白 came: the column starts 62 px higher for each row more, so every row from 选曲 on is where it was, and taken when it was (the slack was already -1.06 s)
SOURCES = ("《淮南子·本经训》", "张彦远《历代名画记》", "葛洪《抱朴子》", "陶渊明《与子俨等疏》", "杜甫《春望》", "李白《静夜思》《将进酒》", "李清照《声声慢》", "苏轼《江城子》", "莎士比亚《十四行诗》第 18 首（1609 四开本）", "荷马《伊利亚特》", "《吉尔伽美什史诗》", "古埃及《亡灵书》", "托勒密王名圈", "平政院裁决书（周树人诉教育总长章士钊案，1926）", "Kernighan 与 Ritchie《The C Programming Language》", "《周易·系辞上》", "《论语·为政》", "波德莱尔《恶之花》", "塞万提斯《堂吉诃德》", "但丁《神曲》", "歌德《浪游者的夜歌》", "普希金《我曾经爱过你》", "维吉尔《埃涅阿斯纪》", "卡蒙斯《卢西塔尼亚人之歌》", "乌姆鲁勒·盖斯《悬诗》", "哈菲兹《诗颂集》", "《希伯来圣经·创世记》", "《梨俱吠陀》", "蒂鲁瓦鲁瓦尔《古拉尔箴言》", "松尾芭蕉《古池》与《奥之细道》", "金素月《杜鹃花》《招魂》")      # 張照's 書聯 and its museum, the last two, went with 六's 「思」 (2026-11-07); what took their place is under 素材
REST = (
    ("素材", ("甲骨字形 Wikimedia Commons（Richard Sears）", "甲骨「我」Wikimedia Commons（合6057）", "汉字笔画 makemeahanzi（Arphic）", "唐诗语料 chinese-poetry", "唐诗书页 哈佛大学图书馆藏《唐詩三百首註疏》", "壁纸 macOS Sequoia Sunrise（Apple）", "参考 罗聘《鬼趣图》")),
    # the fonts as the handbook's section 8 lists them, a line to each family of them: macOS's own, the Noto two (SIL OFL), and Anthropic's, which are read at run time and not shipped
    ("字体", ("macOS　行楷、楷体、宋体、华文黑体、苹方", "Caslon、Baskerville、Didot、Palatino、Helvetica", "SF、Menlo、Arial Black 与各语种系统字体", "Noto Sans Cuneiform、Egyptian Hieroglyphs（OFL）", "Claude 应用的字体与标识为运行时读取")),
    ("工具", ("Blender 5.2、FFmpeg、Python、Claude Code",)),
    ("渲染", ("Mac Studio M5 Max",)),
    ("声明", ("Claude、Clawd 和 Claude 界面属于 Anthropic。", "本片为个人作品，非 Anthropic 出品或认可。", "片中的帖子、账号和对话均为虚构。")),
)
THANKS_ROLE, THANKS = "特别鸣谢", "五千年来，所有写下字的人。"
ACK = ("致谢", "知乎用户 李闯")
END, HOUSE = "终", "小克影业"
# what was turned down, what was said about it, its lines where one is too long for the window, and what to keep of the picture: two of them have an old subtitle burnt in
NG = (
    ("ng_press_machine.png", ("「不要再出现当前版本中的", "大型工业压印机器全景」"), None),
    ("ng_too_close.png", ("「离得太近了，画面怎么看清楚在干嘛」",), None),
    ("ng_macbook.png", ("「你这个苹果电脑太丑了我求你了」",), (70, 0, 897, 465)),
    ("ng_person.png", ("「把人影去掉吧，没必要」",), (134, 0, 960, 465)),
    ("crab_v1_yawn.png", ("「能不能可爱点，嘴巴小一点」",), None),
)
SAID = (216, 207, 190)
_fonts = {}


def font(path, index, size, k=K):
    if (path, index, size, k) not in _fonts:
        _fonts[path, index, size, k] = ImageFont.truetype(path, size * k, index=index)
    return _fonts[path, index, size, k]


def runs(text, size, k=K):
    """The text in runs of one face: Song, and Baskerville for anything Latin."""
    out = []
    for c in text:
        latin = ord(c) < 0x2E80 and c not in CJK
        f = font(SERIF, 0, round(size * 1.06), k) if latin else font(SONG, 6, size, k)
        if out and out[-1][0] is f:
            out[-1][1] += c
        else:
            out.append([f, c])
    return out


def width(text, size, track=0.0):
    return sum(f.getlength(s) / K for f, s in runs(text, size)) + track * size * (len(text) - 1)


def put(d, x, base, text, size, colour, track=0.0):
    for f, s in runs(text, size):
        if track:
            for c in s:
                d.text((x * K, base * K), c, font=f, fill=colour, anchor="ls")
                x += f.getlength(c) / K + track * size
        else:
            d.text((x * K, base * K), s, font=f, fill=colour, anchor="ls")
            x += f.getlength(s) / K


def rows(d, y, role, names, size, lead):
    """A role and the names against it, y its first baseline. Returns the next baseline."""
    t = width(role, 22, .3)
    put(d, AXIS - GAP - t, y, role, 22, ROLE, .3)
    for n in names:
        put(d, AXIS + GAP, y, n, size, NAME)
        y += lead
    return y


def window(canvas, name, said, keep=None):
    """The window in the left third, and under it what was said, large enough to read at any size the film is seen."""
    w, h, x = 800, 450, 62
    y = (H - h) // 2 - 36
    pic = Image.open(os.path.join(OUT, "ng", name)).convert("RGB")
    if keep:
        pic = pic.crop(keep)
    canvas.paste(pic.resize((w * K, h * K), Image.LANCZOS), (x * K, y * K))
    d = ImageDraw.Draw(canvas)
    d.rectangle(((x - 6) * K, (y - 6) * K, (x + w + 6) * K, (y + h + 6) * K), outline=(120, 110, 96), width=K)
    for i, line in enumerate(said):
        put(d, x + (w - width(line, 36, .04)) / 2, y + h + 62 + i * 52, line, 36, SAID, .04)


def finish(canvas, name, seed):
    a = np.asarray(canvas.resize((W, H), Image.LANCZOS), np.float32) / 255
    base = np.array(BG, np.float32) / 255
    a = a + np.maximum(gf(a - base, (3.5, 3.5, 0)), 0) * .08       # what bright things do on film
    n = gf(np.random.default_rng(seed).normal(0, 1, (H, W)).astype(np.float32), .6) * 1.5
    a = a + n[..., None] * .0075
    os.makedirs(OUT, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255 + .5)).save(os.path.join(OUT, name + ".png"))
    print(os.path.join(OUT, name + ".png"), flush=True)


def blank():
    im = Image.new("RGB", (W * K, H * K), BG)
    return im, ImageDraw.Draw(im)


def cr1():
    im, d = blank()
    window(im, *NG[0])
    put(d, AXIS - width(TITLE, 84, .42) / 2, 230, TITLE, 84, NAME, .42)
    y = 374
    for g in GROUPS:
        for role, name in g:
            y = rows(d, y, role, [name], 40, 62)
        y += 34
    finish(im, "cr1", 701)


def cr2():
    im, d = blank()
    window(im, *NG[1])
    rows(d, 214, "引用", SOURCES[:18], 27, 47)
    finish(im, "cr2", 702)


def cr2b():
    im, d = blank()
    y = 150
    for role, names in REST:
        y = rows(d, y, role, names, 27, 46) + 30
    rows(d, y, *ACK[:1], ACK[1:], 40, 62)
    finish(im, "cr2b", 704)


def cr3():
    im, d = blank()
    put(d, (W - width(THANKS_ROLE, 26, .6)) / 2, 486, THANKS_ROLE, 26, ROLE, .6)
    put(d, (W - width(THANKS, 56, .08)) / 2, 588, THANKS, 56, NAME, .08)
    finish(im, "cr3", 703)


# ---------------------------------------------------------------- the real input box

def box():
    """The box from the app's window, as the film's other screens have it, with its cursor and without, cut out with its round corners."""
    import film_room_ui as ui
    page, ring = ui.page, ui.ring
    z = ring.K * ui.HIGH
    x0 = (ring.WIN_X + ring.WIN_W / 2 - page.BOX_W / 2) * z
    y1 = (ring.WIN_Y + ring.WIN_H - 30) * z
    y0, x1 = y1 - 50 * z, x0 + page.BOX_W * z
    os.makedirs(os.path.join(OUT, "crab"), exist_ok=True)
    for name, typing in (("box_on", True), ("box_off", False)):
        a = ring._frame(page.window(page.you(ui.ASK), typing=typing, title=ui.TITLE, w=ring.WIN_W, h=ring.WIN_H, top=ui.TOP), ui.HIGH)
        part = a[int(y0):int(y1), int(x0):int(x1)]
        mask = Image.new("L", (part.shape[1] * 2, part.shape[0] * 2), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, mask.width - 1, mask.height - 1), radius=int(18 * z * 2), fill=255)
        im = Image.fromarray(part[..., :3]).convert("RGBA")
        im.putalpha(mask.resize(im.size, Image.LANCZOS))
        im.save(os.path.join(OUT, "crab", name + ".png"))
    print("box", flush=True)


# ---------------------------------------------------------------- the roll

SEAL_PX, SEAL_TOP, SEAL_GAIN, HOUSE_Y = 156, 384, 1.22, 590      # the seal by the end: bigger and brighter than it was (116 px at 404, as baked), its name under it
LENGTH, AFTER = 146.0, 2.4  # the roll's length, as the cut has it; and how long the seal stays alone once it has walked off, the last .8 of it going to black
BOW_MAX, LOOK_MAX = 4.8, 1.6    # the bow under the thanks and the look at us, no longer than this whatever the slack: longer drags. What is left over goes,
                                # half to its looking up at the thanks before it bows, half to its looking down at 终 before it takes it
V = 32.0                    # how fast the column rises, frame pixels a second: every row has about twenty-five seconds on screen, and the crab about two to take it and be ready for the next
FPS = 24
K2 = 2                      # what each glyph is drawn at, over the frame's pixels
FEET_Y, CRAB_X, CRAB_W = 262, 1290, 300     # where it stands to eat, and how wide it is from claw to claw
FATTER = ("fat1", "fat2", "fat3", "fat4", "fat5", "fat6", "ball")     # what it goes through as it eats the sources
SPIN_SHARE, SPIN_HZ, SPIN_DOWN = .15, 1.5, .8                         # how much of the sources the ball turns for, how fast (16 pictures a turn: one a frame, so it never sticks), how long it takes to run down
FLOOR_Y = 905               # where it comes down to, under the thanks
# how much wider and taller the second of two pictures stands than the first
BOW, CROUCH, RECOVER = (1.115, .807), (1.09, .857), (.86, 1.37)     # up to bow, down to leap_a, leap_c to front
GROWN = ((.974, 1.063), (.977, 1.065), (.973, 1.067), (.976, 1.065), (.983, 1.05), (.995, 1.035))     # each of FATTER to the next
STRIDE = 150                # how far one round of the walk's four pictures carries it, frame pixels
THANKS_Y = 540


def ease(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def surface_of(a):
    """A cairo surface from straight RGBA."""
    import cairo
    h, w = a.shape[:2]
    f = a[..., 3:].astype(np.float32) / 255
    b = np.empty((h, w, 4), np.uint8)
    b[..., 0], b[..., 1], b[..., 2] = (a[..., 2] * f[..., 0]).astype(np.uint8), (a[..., 1] * f[..., 0]).astype(np.uint8), (a[..., 0] * f[..., 0]).astype(np.uint8)
    b[..., 3] = a[..., 3]
    data = bytearray(b.tobytes())
    return cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, w, h, w * 4), data


_glyphs = {}


def glyph(f, c, colour):
    """One character as a surface at K2 times the frame's pixels, its origin on the baseline at (pad, pad) in frame pixels."""
    key = (id(f), c, colour)
    if key not in _glyphs:
        size = f.size / K2
        pad = int(size * .4) + 2
        side = int(size * 1.9) + 2 * pad
        im = Image.new("L", (side * K2, side * K2), 0)
        ImageDraw.Draw(im).text((pad * K2, int(size * 1.3 + pad) * K2), c, font=f, fill=255, anchor="ls")
        a = np.zeros((im.height, im.width, 4), np.uint8)
        a[..., :3] = colour
        a[..., 3] = np.asarray(im)
        s, data = surface_of(a)
        _glyphs[key] = (s, data, pad, int(size * 1.3 + pad), f.getlength(c) / K2)
    return _glyphs[key]


def spell(text, size, x, colour, track=0.0):
    """Where each character of the text goes, each with its surface: (surface entry, x), kerned as the whole run would be."""
    out = []
    for f, s in runs(text, size, K2):
        for i, c in enumerate(s):
            if not c.isspace():
                out.append((glyph(f, c, colour), x + f.getlength(s[:i]) / K2 + track * size * i))
        x += f.getlength(s) / K2 + track * size * len(s)
    return out, x


def spell_width(text, size, track=0.0):
    return spell(text, size, 0.0, NAME, track)[1] - track * size


class Row:
    def __init__(self, y, parts, section="plain", size=40):
        self.y, self.parts, self.section, self.top = y, parts, section, size * .85
        self.lo = min(x for _, x in parts)
        self.hi = max(x + g[4] for g, x in parts)
        self.span = self.hi - self.lo
        self.start = self.end = self.pivot = None
        self.head_right = False


def ink_left(c, size):
    """How far in from its box a character's ink starts, in frame pixels."""
    f = runs(c, size, K2)[0][0]
    x0 = int(f.size * .5)
    im = Image.new("L", (int(f.size * 2.5), int(f.size * 2)), 0)
    ImageDraw.Draw(im).text((x0, int(f.size * 1.5)), c, font=f, fill=255, anchor="ls")
    return (int(np.nonzero(np.asarray(im).max(0) > 0)[0][0]) - x0) / K2


_hangs = {}


def hang(name, size):
    """A row that opens with 《 hangs it out to the left by the blank the bracket has before its ink, so that its ink lines up with the other
    rows' first characters, not its box (2026-11-07): the bracket's blank less what a Song character has before its ink, taken from the
    first characters of the sources that do not open with one. Only where it is drawn changes: what each row is and when it is taken do not."""
    if not name.startswith("《"):
        return 0.0
    if size not in _hangs:
        others = sorted(ink_left(s[0], size) for s in SOURCES if not s.startswith("《"))
        _hangs[size] = ink_left("《", size) - others[len(others) // 2]
    return _hangs[size]


def credit_row(y, role, name, size, section="plain"):
    parts = []
    if role:
        parts += spell(role, 22, AXIS - GAP - spell_width(role, 22, .3), ROLE, .3)[0]
    parts += spell(name, size, AXIS + GAP - hang(name, size), NAME)[0]
    return Row(y, parts, section, size)


class Crab:
    """The little crab: his poses from head/crab/, one camera and one canvas, so each is drawn at one size from one place on the floor, and what squashes or fattens him is in the picture. Where the mouth is comes from each picture's dark hole."""
    NAMES = ("front", "down", "up", "bite", "inhale", "chewa", "chewb", *FATTER, "ball_stop", "ball_chewa", "ball_chewb", "ball_swallow", "pop1", "pop2", "pop3", "pop4", "leap_a", "leap_b", "leap_c", "bow", "walk1", "walk2", "walk3", "walk4")
    FEET = 832                  # how far down the canvas he stands
    SPIN_MOUTH = (500, 672)     # while the ball turns its mouth goes round the back: what it is still taking goes in here, on the canvas
    HEADROOM = 14               # pixels the ball keeps from the top of the frame: it settles that much onto the column to do it

    def __init__(self):
        self.pose = {}
        have = os.listdir(os.path.join(OUT, "crab"))
        self.spin = [f[:-4] for f in sorted((f for f in have if re.fullmatch(r"spin\d+\.png", f)), key=lambda f: int(re.search(r"\d+", f).group()))]
        for n in (*self.NAMES, *self.spin):
            a = Image.open(os.path.join(OUT, "crab", n + ".png")).convert("RGBA")
            if not self.pose:
                xs = np.where(np.asarray(a)[..., 3].max(0) > 40)[0]
                self.canvas, self.scale = a.width, CRAB_W / (xs[-1] - xs[0] + 1)
            k = self.scale * K2
            im = np.asarray(a.resize((int(a.width * k), int(a.height * k)), Image.LANCZOS))
            ys, xs = np.where(im[..., 3] > 40)
            top, low, w, cx = ys.min(), ys.max(), xs.max() - xs.min(), (xs.min() + xs.max()) / 2
            holes, count = label((im[..., :3].astype(np.float32).mean(-1) < 34) & (im[..., 3] > 200))
            best, mouth = 0, (cx, top + .68 * (low - top))
            for i in range(1, count + 1):
                yy, xx = np.where(holes == i)
                if len(yy) > best and yy.mean() > (top + low) / 2 and abs(xx.mean() - cx) < .2 * w:
                    best, mouth = len(yy), (xx.mean(), yy.mean())
            self.pose[n] = dict(surface=surface_of(im), mouth=(mouth[0] / k, mouth[1] / k), top=top / k)
        self.sink = max((self.FEET - self.pose["ball"]["top"]) * self.scale - (FEET_Y - self.HEADROOM), 0.0)

    def which(self, st):
        return self.spin[int(st["phase"] % 1 * len(self.spin))] if st["pose"] == "spin" else st["pose"]

    def mouth(self, st):
        """Where its mouth is in the frame, in the state st."""
        at = self.SPIN_MOUTH if st["pose"] == "spin" else self.pose[st["pose"]]["mouth"]
        return st["x"] + (at[0] - self.canvas / 2) * self.scale * st["sx"], st["y"] + st.get("settle", 0.0) * self.sink - (self.FEET - at[1]) * self.scale * st["sy"]

    def draw(self, ctx, st, alpha=1.0):
        import cairo
        k = self.scale * K2
        ctx.save()
        ctx.translate(st["x"], st["y"] + st.get("settle", 0.0) * self.sink)
        ctx.scale(st["sx"] / K2, st["sy"] / K2)
        ctx.set_source_surface(self.pose[self.which(st)]["surface"][0], -self.canvas / 2 * k, -self.FEET * k)
        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint_with_alpha(alpha)
        ctx.restore()

    def blend(self, ctx, a, b, m):
        """a going over into b, m of the way: each only as much as it still is, so that nothing of a is left standing outside b."""
        import cairo
        ctx.push_group()
        self.draw(ctx, a, 1 - m)
        ctx.set_operator(cairo.OPERATOR_ADD)
        self.draw(ctx, b, m)
        ctx.pop_group_to_source()
        ctx.paint()


class Roll:
    def __init__(self):
        """The rows and the crab's acts, and then the same again with the time left over given to its looking: it bows under the thanks for longer and looks at us longer before it takes the end, so that the roll is LENGTH whatever the rows come to."""
        self.slack = 0.0
        self.build_rows()
        self.schedule()
        self.slack = LENGTH - AFTER - self.gone
        if self.slack < -3.0:
            raise SystemExit(f"the rows run {-self.slack:.2f} s over {LENGTH} s")
        self.schedule()
        self.length = LENGTH

    def build_rows(self):
        rs = []
        y = 700.0 - 62 * (sum(len(g) for g in GROUPS) - ROWS_BEFORE_V4)
        parts, _ = spell(TITLE, 84, AXIS - spell_width(TITLE, 84, .42) / 2, NAME, .42)
        rs.append(Row(y, parts, "title", 84))
        y += 144
        for g in GROUPS:
            for role, name in g:
                rs.append(credit_row(y, role, name, 40))
                y += 62
            y += 34
        y += 64
        for i, name in enumerate(SOURCES):
            rs.append(credit_row(y, "引用" if i == 0 else "", name, 27, "sources"))
            y += 47
        y += 137               # room for the chewing and the burp. 135 until two rows went from the sources (47 apart) to 素材 (46 apart): 2 more keeps every row from 汉字笔画 on where it was, and so taken when it was
        for role, names in REST:
            for i, name in enumerate(names):
                rs.append(credit_row(y, role if i == 0 else "", name, 27))
                y += 46
            y += 30
        y += 16
        rs.append(credit_row(y, ACK[0], ACK[1], 40))
        self.rows = rs
        self.ack_y = y
        self.thanks_y = y + 440
        self.thanks = spell(THANKS, 56, (W - spell_width(THANKS, 56, .08)) / 2, NAME, .08)[0]
        self.thanks_role = spell(THANKS_ROLE, 26, (W - spell_width(THANKS_ROLE, 26, .6)) / 2, ROLE, .6)[0]
        self.end = spell(END, 96, (W - spell_width(END, 96)) / 2, NAME)[0]
        self.house = spell(HOUSE, 24, 1222 - spell_width(HOUSE, 24, .5) / 2, ROLE, .5)[0]

    def schedule(self):
        """When each row is taken, and what the crab is doing the whole time. Every act is a span and a function of how far through it is."""
        acts = []
        settles = [(0.0, 0.0)]

        def act(t0, t1, fn):
            acts.append((t0, t1, fn))

        def bite_(pose="bite", amount=.05):
            return lambda u: dict(pose=pose, sx=1 + amount * math.sin(math.pi * min(u * 3, 1)), sy=1 + amount * .6 * math.sin(math.pi * min(u * 3, 1)), dy=0.0)

        def chew(amount=1.0):
            def fn(u):
                s = math.sin(math.pi * u)
                return dict(pose="chewa" if u < .5 else "chewb", sx=1 + .015 * s * amount, sy=1 - .025 * s * amount, dy=0.0)
            return fn

        def fattening(stage, was, t0, t1, spun):
            def fn(u):
                took = u * (t1 - t0)
                st = dict(pose=stage, sx=1.0, sy=1.0, dy=0.0, phase=(t0 + took - spun) * SPIN_HZ)
                if was in FATTER and stage in FATTER and was != stage:
                    st.update(turning_into(was, stage, GROWN[FATTER.index(was)], took / .6))
                elif was and was != stage:
                    st.update(melted=was, melt=ease(took / .15))
                return st
            return fn

        def turning_into(a, b, ratio, u):
            """a going over into b when they stand at different sizes: one stretches to the other's size and the other comes from the first's, so that they meet and change over on the way, and nothing shows double."""
            e = ease(u)
            sx, sy = 1 + (ratio[0] - 1) * e, 1 + (ratio[1] - 1) * e
            return dict(pose=b, sx=sx / ratio[0], sy=sy / ratio[1], melted=a, melted_sx=sx, melted_sy=sy, melt=ease((u - .3) / .4), dy=0.0)

        previous = None
        spin_from = 0.0
        busy = 0.0
        turn = 0
        room = 9.0              # how long it has from the end of the last row, chewing done, until this one has to be gone
        act(0.0, 5.0, lambda u: dict(pose="front", sx=1.0, sy=1 + .006 * math.sin(u * 13), dy=0.0))
        for i, r in enumerate(self.rows):
            after = self.rows[i + 1] if i + 1 < len(self.rows) else None
            gap = (after.y - r.y) / V if after else 3.0
            long_ = r.section == "sources"
            dense = r.top < 30         # the small rows, set close: they go in one after another, and it chews at the end of the block
            chained = dense and gap < 2.0
            dur = min(.9 + r.span / 800, room)
            if r.section == "title":
                dur, chews = 1.5, 2
            elif chained:
                dur, chews = max(gap - .02, min(dur, room)), 0
            elif long_:
                chews = 0
            elif dense:
                chews = 2
            else:
                chews = 2 if gap > 2.4 else 1
            room = gap - .36 * chews - .06 if not chained else gap - .02
            arrive = (r.y - r.top - V * dur - FEET_Y) / V      # it has gone by the time it would have reached its feet
            start = max(arrive, busy)
            r.start, r.end = start, start + dur
            r.head_right = (turn % 3) != 1
            turn += 1
            hold = r.end + (.05 if chained else 0.0)      # one row runs into the next: its mouth does not shut between them
            if not long_:
                act(start, hold, bite_("inhale" if dense or r.section == "title" else "bite", .015 if dense else .03))
            t = r.end
            if long_:
                # it fattens by what it has eaten, each stage melting into the next as a row starts, until it is a ball and the ball turns
                n = sum(1 for x in self.rows if x.section == "sources")
                k = sum(1 for x in self.rows[:i + 1] if x.section == "sources") - 1
                turning = max(round(n * SPIN_SHARE), 1)
                stage = "spin" if k >= n - turning else FATTER[int(k * len(FATTER) / (n - turning))]
                if stage != previous:
                    spin_from = start if stage == "spin" else spin_from
                    # past here it is too tall for the frame, and it settles onto the column by degrees
                    if stage == "fat5":
                        settles.append((start, 0.0))
                    elif stage == "ball":
                        settles.append((start, 1.0))
                act(start, hold, fattening(stage, previous, start, hold, spin_from))
                previous = stage
                if not chained:
                    # nothing more to take: the ball runs down and stops with its face to us, chews three times, swallows, and is itself again
                    run = (t - spin_from) * SPIN_HZ
                    stop = math.ceil(run + SPIN_HZ * SPIN_DOWN / 2)
                    def running_down(u, run=run, stop=stop):
                        return dict(pose="spin" if u < .96 else "ball_stop", phase=run + (stop - run) * (1 - (1 - u) ** 2), sx=1.0, sy=1.0, dy=0.0)
                    act(t, t + SPIN_DOWN, running_down)
                    t += SPIN_DOWN
                    for j in range(3):
                        # three you can count: down on one side, back up, down on the other
                        def chew_ball(u, side="ball_chewb" if j % 2 else "ball_chewa"):
                            s = math.sin(math.pi * min(u / .6, 1))
                            return dict(pose=side if u < .6 else "ball_stop", sx=1 + .008 * s, sy=1 - .015 * s, dy=0.0)
                        act(t, t + .45, chew_ball)
                        t += .45
                    act(t, t + .4, lambda u: dict(pose="ball_swallow", sx=1 - .015 * math.sin(math.pi * u), sy=1 + .03 * math.sin(math.pi * u), dy=0.0))
                    t += .4
                    settles += [(t, 1.0), (t + .25, 0.0)]
                    act(t, t + .5, lambda u: dict(pose=f"pop{min(int(u * 4) + 1, 4)}", sx=1.0, sy=1.0, dy=0.0))
                    t += .5
                    act(t, t + .28, lambda u: dict(pose="front", sx=1 + .035 * math.sin(u * 3 * math.pi) * (1 - u), sy=1 - .05 * math.sin(u * 3 * math.pi) * (1 - u), dy=0.0))
                    t += .28
                    previous = None
                busy = t + (0.0 if chained else .04)
            else:
                for j in range(chews):
                    act(t, t + .36, chew())
                    t += .36
                busy = t + (0.0 if chained else .05)
        # the last row gone. The next one is under it, and it is not for eating: it comes down past it, to look at it from below
        t = busy + .5
        p0 = (CRAB_X, FEET_Y)
        act(t, t + .12, lambda u: turning_into("down", "leap_a", CROUCH, u))
        act(t + .12, t + .3, lambda u: dict(pose="leap_a", sx=1.0, sy=1.0, dy=0.0))
        t += .3
        def leap(u):
            return dict(pose="leap_b", sx=1.0, sy=1.0, x=p0[0] + (960 - p0[0]) * u, y=p0[1] + (FLOOR_Y - p0[1]) * u * u - 95 * math.sin(math.pi * u) * (1 - u))
        act(t, t + .72, leap)
        self.leap = (t, t + 1.0)
        t += .72
        self.landed = t
        act(t, t + .2, lambda u: dict(pose="leap_c", sx=1.0, sy=1.0, dy=0.0, x=960, y=FLOOR_Y))
        act(t + .2, t + .34, lambda u: dict(turning_into("leap_c", "front", RECOVER, u), x=960, y=FLOOR_Y))
        t += .34
        # the column stops with the thanks in the middle of the frame
        self.s_stop = self.thanks_y - THANKS_Y
        self.thanks_enter = (self.thanks_y - 102 - H - 30) / V
        self.ease_t = 1.0
        self.t1 = (self.s_stop - V * self.ease_t / 2) / V
        self.stopped = self.t1 + self.ease_t
        act(t, t + .5, lambda u: dict(pose="front" if u < .4 else "up", sx=1.0, sy=1.0, dy=0.0, x=960, y=FLOOR_Y))
        t += .5
        if self.stopped > t:
            act(t, self.stopped, lambda u: dict(pose="up", sx=1.0, sy=1.0, dy=0.0, x=960, y=FLOOR_Y))
            t = self.stopped
        # the slack: the bow and the look take what they did, up to their caps; the rest is a longer look up at the thanks and down at 终
        bowed, looking = 5.0 + .75 * self.slack, 1.5 + .25 * self.slack
        over = max(bowed - BOW_MAX, 0.0) + max(looking - LOOK_MAX, 0.0)
        bowed, looking = min(bowed, BOW_MAX), min(looking, LOOK_MAX)
        reading = 2.4 + over / 2
        self.up_at = (t, t + reading)
        act(t, t + reading, lambda u, n=reading / 2.4: dict(pose="up", sx=1.0, sy=1 + .006 * math.sin(u * 4 * n), dy=0.0, x=960, y=FLOOR_Y))
        t += reading
        lowering = lambda u: dict(turning_into("up", "bow", BOW, u), x=960, y=FLOOR_Y)
        act(t, t + 1.8, lowering)
        t += 1.8
        self.bow_at = (t, t + bowed)
        act(t, t + bowed, lambda u, n=bowed / 5.0: dict(pose="bow", sx=1.0, sy=1 + .005 * math.sin(u * 6 * n), dy=0.0, x=960, y=FLOOR_Y))
        t += bowed
        self.thanks_out = (t - 1.4, t)
        self.end_in = (t, t + 1.0)
        t += 1.0
        act(t - 1.0, t + .35, lambda u: lowering(1 - u))
        act(t + .35, t + .5, lambda u: dict(pose="up", sx=1.0, sy=1.0, dy=0.0, x=960, y=FLOOR_Y))
        t += .5
        self.look_at = (t, t + looking)
        act(t, t + looking, lambda u, n=looking / 1.5: dict(pose="front", sx=1.0, sy=1.0 + .012 * math.sin(u * 6 * n), dy=0.0, x=960, y=FLOOR_Y))
        t += looking
        down = .25 + over / 2
        self.down_at = (t, t + down)
        act(t, t + down, lambda u: dict(pose="down", sx=1.0, sy=1.0 + .005 * math.sin(math.pi * u) * min(down / .25 - 1, 1), dy=0.0, x=960, y=FLOOR_Y))
        t += down
        last = Row(500.0, self.end, "end")
        last.start, last.end, last.head_right = t, t + 1.3, False
        self.last = last
        act(t, t + 1.3, lambda u: dict(pose="inhale", sx=1.0, sy=1.0, dy=0.0, x=960, y=FLOOR_Y))
        t += 1.3
        for j in range(3):
            act(t, t + .4, lambda u, ch=chew(1.1): dict(ch(u), x=960, y=FLOOR_Y))
            t += .4
        act(t, t + .9, lambda u: dict(pose="front", sx=1.0, sy=1.0, dy=0.0, x=960, y=FLOOR_Y))
        t += .9
        def walk(u):
            # it sets off: the feet go round by how far it has come, so that they do not slide
            far = (W + 260 - 960) * u ** 1.25
            return dict(pose=f"walk{int(far / STRIDE * 4) % 4 + 1}", sx=1.0, sy=1.0, x=960 + far, y=FLOOR_Y)
        act(t, t + 3.0, walk)
        self.walk = (t, t + 3.0)
        t += 3.0
        # gone: the seal alone a moment, then it and everything go to black, for the cut's two seconds of black before the stinger
        self.gone = t
        self.seal_out = (t + AFTER - .8, t + AFTER)
        self.box_in = None
        self.length = t + AFTER
        self.acts = sorted(acts, key=lambda a: a[0])
        self.settles = sorted(settles)

    def shutter(self, t):
        """How many moments to average for a frame, the shutter open for half of it: enough that what moves leaves a smear and not copies of itself."""
        if self.leap[0] <= t < self.leap[1] or self.last.start - .1 <= t < self.last.end + .1 or self.walk[0] <= t < self.walk[1]:
            return 12
        if any(row.start - .05 <= t < row.end + .05 for row in self.rows):
            return 7
        return 2

    def scroll(self, t):
        if t < self.t1:
            return V * t
        if t < self.stopped:
            u = (t - self.t1) / self.ease_t
            return V * self.t1 + V * self.ease_t * (u - u * u / 2)
        return self.s_stop

    def crab(self, t):
        st = dict(pose="down", sx=1.0, sy=1.0, dy=0.0, x=CRAB_X, y=FEET_Y)
        if t >= self.landed:
            st.update(x=960 if t < self.gone else W + 400, y=FLOOR_Y, pose="front")
        for t0, t1, fn in self.acts:
            if t0 <= t < t1:
                st.update(fn((t - t0) / (t1 - t0)))
                break
        else:
            st["sy"] = 1 + .006 * math.sin(t * 2.6)
        st["y"] += st.pop("dy", 0.0)
        ts, vs = zip(*self.settles)
        st["settle"] = float(np.interp(t, ts, vs))
        return st


def noodle(ctx, row, y, mouth, t, alpha):
    """A row going in. What is pulled off the row runs on a curve from where it leaves to the mouth; what has not been reached stays where it was. Each character keeps its shape and turns with the curve, smaller as it gets to the mouth."""
    import cairo
    u = (t - row.start) / (row.end - row.start)
    head = row.hi if row.head_right else row.lo
    sign = -1 if row.head_right else 1
    # how much has been pulled: the row's length and the way to the mouth, slowly, then with a rush, the tail whipping in last
    def path(xc):
        p0, p3 = np.array((xc, y)), np.array(mouth)
        p1 = p0 + ((p3[0] - p0[0]) * .55, 6.0)
        p2 = p3 + (0.0, (p0[1] - p3[1]) * .62 + 16)
        s = np.linspace(0, 1, 56)[:, None]
        pts = (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s * s * p2 + s ** 3 * p3
        along = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        return pts, along
    _, final = path(row.hi if not row.head_right else row.lo)
    pulled = (row.span + final[-1] + 6) * max(0.0, min(1.0, u)) ** 1.25
    xc = head + sign * min(pulled, row.span)
    pts, along = path(xc)
    total = along[-1]
    tan = np.gradient(pts, axis=0)
    ang = np.unwrap(np.arctan2(tan[:, 1], tan[:, 0]))
    for g, x in row.parts:
        s, _, pad, base, adv = g
        mid = x + adv / 2
        d = abs(mid - head)
        ctx.save()
        if d >= pulled:
            ctx.translate(x, y)
            ctx.scale(1 / K2, 1 / K2)
            ctx.set_source_surface(s, -pad * K2, -base * K2)
        else:
            p = pulled - d
            if p >= total:
                ctx.restore()
                continue
            # p is how far it has come from where the noodle leaves the row
            px, py = np.interp(p, along, pts[:, 0]), np.interp(p, along, pts[:, 1])
            a = float(np.interp(p, along, ang))
            if row.pivot is None:
                row.pivot = mouth[0]
            if mid > row.pivot:         # what comes in from the right travels leftward and keeps its head up all the same
                a += math.pi
            a = math.atan2(math.sin(a), math.cos(a)) * ease(p / 80)      # and it leaves the row upright, turning as it goes
            if len(row.parts) == 1:     # one character alone, straight down: it only sways
                a = .2 * math.sin(p / 34)
            near = total - p
            size = .58 + .42 * ease(near / 90)
            wave = 5.0 * math.sin(p / 26 - t * 15) * math.sin(math.pi * p / total)
            ctx.translate(px - math.sin(a) * wave, py + math.cos(a) * wave)
            ctx.rotate(a)
            ctx.scale(size / K2, size / K2)
            ctx.set_source_surface(s, -(pad + adv / 2) * K2, -base * K2)
        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint_with_alpha(alpha)
        ctx.restore()


def still_parts(ctx, parts, y, alpha=1.0):
    import cairo
    for g, x in parts:
        s, _, pad, base, _ = g
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(1 / K2, 1 / K2)
        ctx.set_source_surface(s, -pad * K2, -base * K2)
        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint_with_alpha(alpha)
        ctx.restore()


class Film:
    def __init__(self):
        import cairo
        self.roll = Roll()
        self.crab = Crab()
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        self.windows = []
        for name, said, keep in NG:
            rgb = Image.new("RGB", (W * K, H * K), BG)
            window(rgb, name, said, keep)
            box_ = (40 * K, 240 * K, 900 * K, 900 * K)
            part = rgb.crop(box_).resize(((box_[2] - box_[0]) // K, (box_[3] - box_[1]) // K), Image.LANCZOS)
            a = np.concatenate([np.asarray(part), np.full((part.height, part.width, 1), 255, np.uint8)], -1)
            self.windows.append((surface_of(a), box_[0] / K, box_[1] / K))
        # one after another with nothing between, cut and not faded, the whole run gone before the thanks comes into the frame
        r = self.roll
        each = (r.thanks_enter - .8 - 1.2) / len(NG)
        self.times = [(1.2 + i * each, 1.2 + (i + 1) * each) for i in range(len(NG))]
        print("each window", round(each, 2), "s, last gone", round(self.times[-1][1], 2), "thanks enters", round(r.thanks_enter, 2), flush=True)
        # the seal: what the stone leaves, on black
        cover = np.asarray(Image.open(os.path.join(OUT, "maps", "ink_cover.png")).convert("RGB"))
        colour = np.asarray(Image.open(os.path.join(OUT, "maps", "ink_colour.png")).convert("RGB"))
        top, left = int((3.5 - 1.5) * 400), int((5.0 - 1.5) * 400)
        rgba = np.concatenate([colour[top:top + 1200, left:left + 1200], cover[top:top + 1200, left:left + 1200, :1]], -1)
        rgba = rgba.astype(np.float32)
        rgba[..., :3] = np.clip(rgba[..., :3] * SEAL_GAIN + np.array((6.0, 4.0, 2.0)), 0, 255)       # a fresh cinnabar, not the stone's dull one
        self.seal = surface_of(np.asarray(Image.fromarray(rgba.astype(np.uint8), "RGBA").resize((SEAL_PX * K2, SEAL_PX * K2), Image.LANCZOS)))
        on, off = [Image.open(os.path.join(OUT, "crab", n + ".png")).convert("RGBA") for n in ("box_on", "box_off")]
        w = 980
        self.box = [surface_of(np.asarray(b.resize((w * K2, int(b.height * w / b.width) * K2), Image.LANCZOS))) for b in (on, off)]
        self.box_w = w
        self.box_h = int(on.height * w / on.width)

    def draw(self, t, mid=None):
        """The frame at t, 1920 by 1080, as RGB floats. What is being pulled in is sharper than the shutter would make it, so that its characters stay characters: it moves as if at the moment a little nearer mid, the middle of the frame."""
        taken = t if mid is None else mid + (t - mid) * .4
        import cairo
        r = self.roll
        ctx = cairo.Context(self.surface)
        ctx.set_source_rgb(BG[0] / 255, BG[1] / 255, BG[2] / 255)
        ctx.paint()
        first, last = self.times[0][0], self.times[-1][1]
        a = ease((t - first) / .8) * (1 - ease((t - last) / .8))
        if a > 0:
            i = min(max(int((t - first) / (last - first) * len(self.times)), 0), len(self.times) - 1)
            (s, _), x, y = self.windows[i]
            ctx.set_source_surface(s, x, y)
            ctx.paint_with_alpha(a)
        scroll = r.scroll(t)
        # the pictures change on the frame, as clay does; only where it stands takes the shutter
        st = r.crabstate = r.crab(t if mid is None else mid)
        at = r.crab(t)
        st.update(x=at["x"], y=at["y"])
        active = []
        for row in r.rows:
            y = row.y - scroll
            if t >= row.end:
                continue
            if t >= row.start:
                active.append((row, y))
            elif -60 < y < H + 60:
                still_parts(ctx, row.parts, y)
        # the thanks, which it leaves alone
        y = r.thanks_y - scroll
        alpha = 1 - ease((t - r.thanks_out[0]) / (r.thanks_out[1] - r.thanks_out[0]))
        if y < H + 200 and alpha > 0:
            still_parts(ctx, r.thanks_role, y - 102, alpha)
            still_parts(ctx, r.thanks, y, alpha)
        # the end, and the small seal
        a = ease((t - r.end_in[0]) / (r.end_in[1] - r.end_in[0]))
        if a > 0:
            seal = a * (1 - ease((t - r.seal_out[0]) / (r.seal_out[1] - r.seal_out[0])))
            if seal > 0:
                ctx.save()
                ctx.translate(1222 - SEAL_PX / 2, SEAL_TOP)
                ctx.scale(1 / K2, 1 / K2)
                ctx.set_source_surface(self.seal[0], 0, 0)
                ctx.get_source().set_filter(cairo.FILTER_GOOD)
                ctx.paint_with_alpha(seal)
                ctx.restore()
                still_parts(ctx, r.house, HOUSE_Y, seal)
            if t < r.last.start:
                still_parts(ctx, r.end, r.last.y + 48, a)
        # it stands with everything it has not reached behind it
        if st["x"] < W + 240:
            if st.get("melt", 1.0) < 1.0:
                self.crab.blend(ctx, dict(st, pose=st["melted"], sx=st.get("melted_sx", st["sx"]), sy=st.get("melted_sy", st["sy"])), st, st["melt"])
            else:
                self.crab.draw(ctx, st)
        for row, y in active:
            noodle(ctx, row, y, self.crab.mouth(st), taken, 1.0)
        if r.last.start <= t < r.last.end:
            noodle(ctx, r.last, r.last.y + 48, self.crab.mouth(st), taken, 1.0)
        # the box lights, empty, its cursor going (no longer: the stinger has the box)
        a = ease((t - r.box_in[0]) / (r.box_in[1] - r.box_in[0])) if r.box_in else 0.0
        if a > 0:
            lit = (t - r.box_in[0]) % 1.06 < .56
            ctx.save()
            ctx.translate((W - self.box_w) / 2, (H - self.box_h) / 2)
            ctx.scale(1 / K2, 1 / K2)
            ctx.set_source_surface(self.box[0 if lit else 1][0], 0, 0)
            ctx.get_source().set_filter(cairo.FILTER_GOOD)
            ctx.paint_with_alpha(a)
            ctx.restore()
        self.surface.flush()
        b = np.frombuffer(self.surface.get_data(), np.uint8).reshape(H, W, 4)
        return b[..., [2, 1, 0]].astype(np.float32) / 255

    def finished(self, t, i, half=True, fps=FPS):
        n = self.roll.shutter(t)
        a = sum(self.draw(t + (i - (n - 1) / 2) / n * .5 / fps, t) for i in range(n)) / n
        if half:
            a = (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) / 4
        base = np.array(BG, np.float32) / 255
        a = a + np.maximum(gf(a - base, (1.75 if half else 3.5, 1.75 if half else 3.5, 0)), 0) * .08
        # the whole thing comes up out of the dark
        a = base + (a - base) * ease(t / .8)
        n = np.random.default_rng(3000 + i).normal(0, 1, a.shape[:2]).astype(np.float32)
        a = a + n[..., None] * .0075
        # and at the very end goes down into true black, the cut's black
        a = a * (1 - ease((t - (self.roll.length - .8)) / .8))
        return np.uint8(np.clip(a, 0, 1) * 255 + .5)


def frames(times):
    f = Film()
    os.makedirs(os.path.join(OUT, "roll"), exist_ok=True)
    for t in times:
        Image.fromarray(f.finished(t, int(t * FPS), half=False)).save(os.path.join(OUT, "roll", f"t{t:07.3f}.png"))
    report(f.roll)


def report(r):
    print("length", round(r.length, 2), "thanks stopped", round(r.stopped, 2), "landed", round(r.landed, 2), flush=True)
    print("closest a row came to its feet before it was taken", round(min(row.y - V * row.start - row.top - FEET_Y for row in r.rows), 1), "px", flush=True)
    for row in r.rows[:1] + [x for x in r.rows if x.section == "sources"][:1] + [x for x in r.rows if x.section == "sources"][-1:] + r.rows[-1:]:
        print("row at", round(row.start, 2), round(row.end, 2), flush=True)
    print("end", round(r.last.start, 2), "gone", round(r.gone, 2), "bowed for", round(5.0 + .75 * r.slack, 2), "seal out", [round(x, 2) for x in r.seal_out], flush=True)


def roll(hi=False):
    """The roll as the cut has it: 960 by 540 at 24 frames a second; hi, the final's, 1920 by 1080 at the film's 30, the shutter open for half of
    each of its frames. The roll goes by the clock, so it is the same roll at either rate."""
    import subprocess
    f = Film()
    report(f.roll)
    fps, size, out = (30, "1920x1080", "credits_1080.mp4") if hi else (FPS, "960x540", "credits_low3.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", os.path.join(OUT, out)], stdin=subprocess.PIPE)
    n = int(round(f.roll.length * fps))
    for i in range(n):
        p.stdin.write(f.finished(i / fps, i, half=not hi, fps=fps).tobytes())
        if i % 240 == 0:
            print(i, n, flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", n, "frames", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "stills"
    if what == "stills":
        cr1()
        cr2()
        cr2b()
        cr3()
    elif what == "box":
        box()
    elif what == "frames":
        frames([float(x) for x in sys.argv[2:]])
    elif what == "roll":
        roll()
    elif what == "roll1080":
        roll(hi=True)
