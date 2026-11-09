"""十一 where it says something: 我能写出一千种悲伤, real lines of real books going by on the rain; a girl's blog on the Mac's own screen, 反正也没有人看 and its reader count going up by one, because I read it.

    python ink_eleven.py still T [T ...]     # frames at film times, ink/film_out/move/eleven_*.png
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import ink_events_grief as grief
import ink_learn as learn
import ink_mac as mac
from ink_events_lib import region_from, shot
from ink_grade import sm
from ink_move import lookup, noise, sample
from ink_subs import _font, at

W, H = 1920, 1080
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ink", "film_out", "move")

# ---------------------------------------------------------------- 我能写出一千种悲伤
# Every line read on zh.wikisource: (the line, who wrote it, which of its characters is the grief). Nearest the eye first.
VERSES = (
    ("问君能有几多愁", "李煜", "愁"),
    ("此恨绵绵无绝期", "白居易", "恨"),
    ("感时花溅泪", "杜甫", "泪"),
    ("满纸荒唐言，一把辛酸泪", "《红楼梦》", "泪"),
    ("念天地之悠悠，独怆然而涕下", "陈子昂", "涕"),
    ("多情自古伤离别", "柳永", "伤"),
    ("怎一个愁字了得", "李清照", "愁"),
    ("我心伤悲，莫知我哀", "《诗经》", "哀"),
    ("长太息以掩涕兮", "屈原", "涕"),
    ("花落人亡两不知", "《红楼梦》", "亡"),
    ("而今识尽愁滋味，欲说还休", "辛弃疾", "愁"),
    ("万里悲秋常作客", "杜甫", "悲"),
    ("我真傻，真的", "鲁迅", "傻"),
    ("岂不痛哉", "王羲之", "痛"),
    ("长歌当哭，是必须在痛定之后的", "鲁迅", "哭"),
    ("人固有一死", "司马迁", "死"),
    ("不以物喜，不以己悲", "范仲淹", "悲"),
)
PLACE = (-430, 40, -240, 180, -560, -60, 220, -380, 90, -520, -120, 150, -320, 280, -600, 30, -250)        # how far each line's start lies from the middle of the river, at the nearest
HORIZON, NEAR = 376.0, 704.0     # the water's own perspective, as ink_events_grief has it
CHAR = 92.0                      # a character's width at the nearest row, in frame pixels
SRC = 2                          # the rows are set at this many pixels to one of those
STEP = 0.125                     # how much smaller each row is than the one nearer: in the log of its size, so that rows keep their spacing all the way out
DEPTH = 0.072                    # how deep a row lies on the water, in the same measure: it makes its characters about half as tall as they are wide
RATE = 0.2                       # how fast the rows come, in the same measure a second
LEAN = 0.42                      # how much of the full perspective the sides of a character follow: at all of it, lines far from the middle read as italics
FIRST = -0.28                    # where the first row is when the shot begins: already a little past the nearest, so the frame is full


def _row_mask(text, who, lit):
    """A line set flat: pale, with its grief in full ink, and who wrote it small after it. 255 is the full ink."""
    size = int(CHAR * SRC)
    f, fs = _font("kai", size), _font("kai", size // 2)
    width = int(f.getlength(text) + fs.getlength(who) + size)
    im = Image.new("L", (width, size + 24), 0)
    grief_ink = Image.new("L", im.size, 0)
    d, g = ImageDraw.Draw(im), ImageDraw.Draw(grief_ink)
    x = 0
    for c in text:
        d.text((x, 12), c, font=f, fill=255 if c == lit else 112)
        if c == lit:
            g.text((x, 12), c, font=f, fill=255)
        x += f.getlength(c)
    d.text((x + size * 0.45, size * 0.48), who, font=fs, fill=84)
    # the grief is wetter than the rest, and goes out into the water
    bleed = np.asarray(grief_ink.filter(ImageFilter.GaussianBlur(size * 0.07)), np.float32) / 255
    return np.maximum(np.asarray(im, np.float32) / 255, bleed * 0.6)


class Rows:
    """Lines of writing lying on the river, coming toward the eye as the current brings them."""

    def __init__(self):
        n = len(VERSES)
        self.masks = [_row_mask(*v) for v in VERSES]
        self.x = np.array(PLACE, np.float32) * 1.0
        self.n = n

    def density(self, t):
        """All the rows at t seconds into the shot, as ink on a frame-sized sheet."""
        out = np.zeros((H, W), np.float32)
        # rows repeat a long way out, where they are only texture: which line is which matters near the eye
        for i in range(self.n + 22):
            tau = FIRST + i * STEP - RATE * t
            if tau < -0.24 or tau > 2.3:
                continue
            s = float(np.exp(-tau))                       # its size: 1 at the nearest
            top = s * np.exp(-DEPTH)
            y0, y1 = HORIZON + NEAR * top, HORIZON + NEAR * s
            ya, yb = int(y0), min(int(y1) + 1, H)
            if ya >= yb:
                continue
            k = i % self.n
            m = self.masks[k]
            yy = np.arange(ya, yb, dtype=np.float32) + 0.5
            ss = np.maximum((yy - HORIZON) / NEAR, 1e-3)
            v = (np.log(ss / top) / DEPTH * m.shape[0]).astype(np.float32)
            mid = float(np.sqrt(s * top))
            across = mid + (ss - mid) * LEAN
            left = 960 + self.x[k] * across
            right = left + m.shape[1] / SRC * across
            xa, xb = max(int(left.min()), 0), min(int(right.max()) + 2, W)
            if xa >= xb:
                continue
            xx = np.arange(xa, xb, dtype=np.float32) + 0.5
            u = ((xx[None] - left[:, None]) / across[:, None] * SRC).astype(np.float32)
            ui, vi = np.clip(u, 0, m.shape[1] - 2), np.clip(v, 0, m.shape[0] - 2)
            u0, v0 = ui.astype(np.int32), vi.astype(np.int32)
            au, av = ui - u0, (vi - v0)[:, None]
            v0 = v0[:, None]
            a = (m[v0, u0] * (1 - au) + m[v0, u0 + 1] * au) * (1 - av) + (m[v0 + 1, u0] * (1 - au) + m[v0 + 1, u0 + 1] * au) * av
            a *= ((u >= 0) & (u < m.shape[1] - 1))
            # a row goes into the mist as it goes away, and comes up out of the water at the near edge
            a *= float(sm((s - 0.11) / 0.3)) * (0.42 + 0.58 * float(sm((s - 0.2) / 0.35)))
            out[ya:yb, xa:xb] = np.maximum(out[ya:yb, xa:xb], a)
        return out


_rows = None


def words(t):
    """我能写出一千种悲伤, the shot: on the plain river of the chapter's start."""
    from ink_cut import mover
    s, end = shot("grief/grey", 1)
    z0, z1 = s["z"]
    p = (t - s["start"]) / (end - s["start"])
    f = mover(s).frame(t - s["start"], s=max(z0 + (z1 - z0) * p - min(z0, z1), 0.0))
    return verses(f, t - s["start"], end - s["start"])


def verses(frame, t, dur):
    """The rain on the river, from ink_events_grief, and on it the writing, broken by the same rings."""
    global _rows
    if not grief._rain:
        grief._rain_prep()
    if _rows is None:
        _rows = Rows()
    if "verses" not in grief._rain:
        grief._rain["verses"] = grief._drops(87, -0.6, dur + 1.0, lambda _: 8.0, first=((0.2, 690.0, 930.0, 0.36, 1.1), (1.0, 1240.0, 845.0, 0.32, 0.95)))
    d = grief._rain
    f = frame.astype(np.float32)
    still = "grief/grey"
    y0 = int(grief.Y_WATER)
    h = grief._grid_rings(d["verses"], t, d["gx"], d["gy"]) * d["torn"]
    gy, gx = np.gradient(h)
    n1 = lookup(noise(95), d["u"] * 240.0, d["v"] * 60.0 - 9.0 * t)
    n2 = lookup(noise(96), d["u"] * 120.0 + 2.0 * t, d["v"] * 30.0 - 5.0 * t)
    sw = 0.8 * d["depth"]
    dx = grief.up1(gx * 3.0 + (n2 - 0.4 * n1) * sw)[y0:]
    dy = grief.up1(gy * 3.0 + (n1 + 0.6 * n2) * sw * 1.6)[y0:]
    light = grief.up1(h * 0.16)[y0:]
    ys, xs = region_from(still, t, dur, (y0, H, 0, W), k=1)
    k = d[still][np.clip(ys + 0.5, 0, H - 1).astype(np.int32), np.clip(xs + 0.5, 0, W - 1).astype(np.int32)]
    o = sample(d["plate " + still], ys + dy, xs + dx) * (1 + light * k)[..., None]
    ink = sample(_rows.density(t)[..., None], ys + dy, xs + dx)
    o = o + (grief.WET - o) * np.clip(ink * 0.96, 0, 1)
    f[y0:] = f[y0:] * (1 - k)[..., None] + o * k[..., None]
    return np.clip(grief._fall(f, t, (1.0, 1.0, 1.0)) + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 还是某个十四岁女孩写在博客里的
BLOG_LINE = "我很难过"
BLOG_A, BLOG_B = (104, 648), (226, 648)     # either end of it, in frame pixels
T_BLOG = at("还是某个")


def _blog_body(selected, reads):
    """The page as it was left in 2009: pink, tiled, a calendar, a counter. selected: how many characters of 我很难过 are selected."""
    pink, deep, line = "#fbeaf1", "#c2477f", "#f0bfd3"
    days = ""
    day = 1
    for _ in range(5):
        cells = ""
        for _ in range(7):
            n = day if day <= 31 else ""
            on = n == 14
            cells += f'<td style="text-align:center;padding:1px 0;{"background:" + deep + ";color:#fff;" if on else ""}">{n}</td>'
            day += 1
        days += f"<tr>{cells}</tr>"
    head = "".join(f'<td style="text-align:center;color:{deep}">{c}</td>' for c in "日一二三四五六")
    picked = f'<span style="background:#b4d5fe;color:#111">{BLOG_LINE[:selected]}</span>{BLOG_LINE[selected:]}'
    lines = ("今天又说肚子疼，没有去上学。", "其实不疼。", "妈妈问我怎么了，我说没事。", f"{picked}，可是说不出是为什么。", "写在这里，反正也没有人看。")
    para = "".join(f'<p style="margin:0 0 9px">{x}</p>' for x in lines)
    return f'''<div style="position:absolute;left:0;right:0;top:0;bottom:0;background:{pink};color:#4a3a42;font-family:'Songti SC',serif;overflow:hidden">
      <div style="padding:13px 22px 9px;border-bottom:2px dotted {line};background:linear-gradient(#ffd9e8,#fbeaf1)">
        <div style="font-size:23px;font-weight:700;color:{deep};letter-spacing:1px">小雨的小窝</div>
        <div style="font-size:11.5px;margin-top:3px;color:#a0708a">写给没有人看的日记</div>
        <div style="font-size:12px;margin-top:6px;color:{deep};text-decoration:underline;word-spacing:9px">首页 日志 相册 留言板</div></div>
      <div style="position:absolute;left:22px;top:104px;width:600px;bottom:14px;background:#fff;border:1px dotted {line};padding:15px 20px;box-sizing:border-box">
        <div style="font-size:18px;font-weight:700;color:{deep}">下雨</div>
        <div style="font-size:11.5px;color:#a0708a;margin:4px 0 13px">2009年3月14日 星期六 23:12 &nbsp; 天气：雨 &nbsp; 心情：……</div>
        <div style="font-size:18px;line-height:1.6">{para}</div>
        <div style="margin-top:16px;padding-top:9px;border-top:1px dotted {line};font-size:13px;color:#a0708a">分类：默认分类 &nbsp;|&nbsp; <span style="color:{deep}">阅读({reads})</span> &nbsp;|&nbsp; 评论(0)</div></div>
      <div style="position:absolute;left:642px;top:104px;right:22px;font-size:11.5px">
        <div style="background:#fff;border:1px dotted {line};padding:7px 9px"><div style="color:{deep};font-weight:700;margin-bottom:5px;text-align:center">2009年3月</div>
          <table style="width:100%;border-collapse:collapse;font-size:11px;font-family:-apple-system,sans-serif"><tr>{head}</tr>{days}</table></div>
        <div style="background:#fff;border:1px dotted {line};padding:7px 9px;margin-top:10px;line-height:1.8"><div style="color:{deep};font-weight:700">最新评论</div>暂无评论</div>
        <div style="background:#fff;border:1px dotted {line};padding:7px 9px;margin-top:10px;text-align:center">你是第 <span style="font-family:Menlo,monospace;color:{deep}">000127</span> 位访客</div></div></div>'''


_blogs = {}
BLOG_BAR = 38


def _blog_screen(selected, reads):
    key = (selected, reads)
    if key not in _blogs:
        _blogs.clear()
        body = f'<div style="position:absolute;left:{mac.PAD}px;top:{mac.PAD + BLOG_BAR}px;width:{learn.WIN_W}px;height:{learn.WIN_H - BLOG_BAR}px;border-radius:0 0 14px 14px;overflow:hidden">{_blog_body(selected, reads)}</div>'
        _blogs[key] = learn.Screen(mac.browser_html("小雨的小窝", learn.WIN_W, learn.WIN_H, BLOG_BAR) + body, "Safari浏览器", ("文件", "编辑", "显示", "历史记录", "书签", "窗口", "帮助"), ("Finder", "Claude", "Safari"))
    return _blogs[key]


SELECT0, SELECT1, COUNT = 1.5, 2.3, 4.3     # seconds after the line starts: the selection, and the reader count going up
NEAR0, NEAR1, CLOSE = 2.4, 4.0, 1.75        # the camera goes in to the end of the post, between these, this far
POST_END = (205, 610)                       # where it goes, in frame pixels at the end of the post


def blog(t):
    """I go over the sentence, and the count of people who have read it goes up by one."""
    x = t - T_BLOG
    n = int(np.clip((x - SELECT0) / (SELECT1 - SELECT0), 0, 1) * len(BLOG_LINE) + 1e-3)
    s = _blog_screen(n, 4 if x >= COUNT else 3)
    # the pointer, along the sentence: page pixels at twice the frame's, as measured on the page
    a, b = (BLOG_A[0] * 2 - s.x, BLOG_A[1] * 2 - s.y), (BLOG_B[0] * 2 - s.x, BLOG_B[1] * 2 - s.y)
    k = float(np.clip((x - SELECT0) / (SELECT1 - SELECT0), 0, 1))
    come = float(sm((x - 0.4) / (SELECT0 - 0.4)))
    start = (s.w * 0.55, s.h * 0.8)
    p = (start[0] + (a[0] - start[0]) * come + (b[0] - a[0]) * k, start[1] + (a[1] - start[1]) * come)
    near = float(sm((x - NEAR0) / (NEAR1 - NEAR0)))
    z = float(np.exp(np.log(CLOSE) * near))
    return s.show(pointer=p, z=z, centre=(W + (2 * POST_END[0] - W) * near, H + (2 * POST_END[1] - H) * near))


def stills(ts):
    os.makedirs(OUT, exist_ok=True)
    for t in ts:
        Image.fromarray(blog(t)).save(os.path.join(OUT, f"eleven_{t:07.2f}.png"))
        print(t, flush=True)


if __name__ == "__main__":
    stills([float(x) for x in sys.argv[2:]])
