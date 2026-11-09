"""十一 还是某个十四岁女孩写在博客里的: her post on Weibo, in Safari on the Mac. The layout and colours follow weibo.com as read on 2026-10-31; the account, the post and the trending list are
invented, and the page never shows a logo. 反正也没有人看: three have read it, and the count stays as it is (the +1 when I read it was cut: it made a point the line makes by itself).

    python film_weibo.py 289.3 290.5 291.8 294.8     # stills at these film times, with their subtitle
    python film_weibo.py clip                        # the whole shot
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

import ink_learn as learn
import ink_mac as mac
import ink_subs
from ink_mac import H, PAD, W, Z
from ink_subs import at

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ink", "film_out", "film", "weibo")
T = at("还是某个")
LINE = "我很难过"
WHO = "初二三班第七排"                # a fourteen-year-old, by what she calls herself
WHEN = "13-3-14 23:52"
SOURCE = "微博 weibo.com"             # written at a computer, in a browser
POST = ("今天又说肚子疼，没有去上学。", "其实不疼。", "妈妈问我怎么了，我说没事。", "{}，可是说不出是为什么。", "写在这里，反正也没有人看。")
TRENDS = ("全国多地迎来降温", "万圣节妆容", "双十一预售", "考研倒计时", "下班路上的晚霞", "秋天的第一杯奶茶", "周末去哪儿", "今天吃什么")

# the window, centred above the subtitle, with the Dock hidden: frame pixels for where it is, points for its size
PLACE = (110, 56, 1000, 514)
BAR = 38
# the page, in points
ORANGE, TEXT, SUB, PAGE, LINES = "#ff8200", "#333", "#939393", "#f1f2f5", "#f0f0f0"
NAV_H = 46
MAIN_X, MAIN_W, CARD_Y = 28, 652, 58
SIDE_X, SIDE_W = 694, 278
TEXT_X, TEXT_Y, TEXT_SIZE, TEXT_LH = 82, 76, 17, 28   # the post, within its card
FOOT_Y = TEXT_Y + TEXT_LH * len(POST) + 10

SELECT0, SELECT1 = 1.5, 2.3                 # seconds after the line starts: the selection
READS = 3                                   # the count of readers, which does not move
NEAR0, NEAR1, CLOSE = 2.4, 4.0, 1.75        # the camera goes in on the post, between these, this far
FOCUS = (MAIN_X + 300, 225)                 # where it goes, in points of the page


def _icon(path, size=16):
    return f'<svg width="{size}" height="{size}" viewBox="0 0 16 16" fill="none" stroke="{SUB}" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round">{path}</svg>'


SHARE = '<path d="M8 2.6H3a1 1 0 0 0-1 1V13a1 1 0 0 0 1 1h9.5a1 1 0 0 0 1-1V9"/><path d="M6.5 9.6c.5-3 2.5-5 6.5-5"/><path d="M10.6 2.1 13.2 4.6 10.6 7"/>'
TALK = '<path d="M2.2 3.2h11.6a.7.7 0 0 1 .7.7v7.4a.7.7 0 0 1-.7.7H7.6L5 14.2V12H2.2a.7.7 0 0 1-.7-.7V3.9a.7.7 0 0 1 .7-.7z"/><path d="M5 7.6h.01M8 7.6h.01M11 7.6h.01" stroke-width="1.6"/>'
THUMB = '<path d="M5.2 6.8 7.6 2.1c1 0 1.8.8 1.8 1.8v2.5h3.4c.8 0 1.4.8 1.2 1.6l-1.1 5c-.1.6-.7 1-1.3 1H5.2z"/><path d="M5.2 6.8H2.4a.6.6 0 0 0-.6.6v6.4c0 .3.3.6.6.6h2.8"/>'


def _body(selected, reads):
    """The post's own page. selected: how many characters of 我很难过 are selected."""
    picked = f'<span style="background:#b4d5fe">{LINE[:selected]}</span>{LINE[selected:]}'
    lines = "".join(f'<div style="height:{TEXT_LH}px">{x.format(picked)}</div>' for x in POST)
    cell = 'display:flex;align-items:center;justify-content:center;gap:7px;flex:1'
    foot = (f'<span style="{cell}">阅读 {reads}</span><span style="{cell}">{_icon(SHARE)}转发</span><span style="{cell}">{_icon(TALK)}评论</span><span style="{cell}">{_icon(THUMB)}赞</span>')
    trends = "".join(f'''<div style="display:flex;align-items:center;height:32px;font-size:13.5px;color:{TEXT}">
        <span style="width:24px;text-align:center;font-style:italic;font-weight:600;font-size:14px;color:{ORANGE}">{i + 1}</span><span>{x}</span></div>''' for i, x in enumerate(TRENDS))
    return f'''<div style="position:absolute;left:0;right:0;top:0;bottom:0;background:{PAGE};color:{TEXT};overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif">
      <div style="position:absolute;left:0;right:0;top:0;height:{NAV_H}px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.05)">
        <div style="position:absolute;left:{MAIN_X}px;top:5px;line-height:1.1;font-weight:600;font-size:21px;color:#222;letter-spacing:1px">微博<div style="font-size:9px;font-weight:400;letter-spacing:0;color:{SUB}">weibo.com</div></div>
        <div style="position:absolute;left:{MAIN_X + 92}px;top:8px;width:250px;height:30px;border-radius:15px;background:#f0f1f4;color:{SUB};font-size:13px;display:flex;align-items:center;gap:7px;padding-left:12px;box-sizing:border-box"><span style="display:flex">{mac.icon("search", 13)}</span>搜索微博</div>
        <div style="position:absolute;right:28px;top:10px;height:26px;width:78px;border-radius:13px;background:{ORANGE};color:#fff;font-size:12.5px;display:flex;align-items:center;justify-content:center">登录/注册</div></div>
      <div style="position:absolute;left:{MAIN_X}px;top:{CARD_Y}px;width:{MAIN_W}px;height:402px;background:#fff;border-radius:4px">
        <div style="position:absolute;left:20px;top:18px;width:50px;height:50px;border-radius:50%;background:#e4e5e8;overflow:hidden"><svg width="50" height="50" viewBox="0 0 50 50"><circle cx="25" cy="20" r="9" fill="#fff"/><path d="M7 50c0-13 8-20 18-20s18 7 18 20z" fill="#fff"/></svg></div>
        <div style="position:absolute;left:{TEXT_X}px;top:21px;font-size:14.5px;font-weight:600">{WHO}</div>
        <div style="position:absolute;left:{TEXT_X}px;top:45px;font-size:12.5px;color:{SUB}">{WHEN}&nbsp;&nbsp;来自{SOURCE}</div>
        <div style="position:absolute;right:20px;top:22px;width:62px;height:26px;border-radius:13px;border:1px solid {ORANGE};box-sizing:border-box;color:{ORANGE};font-size:12.5px;display:flex;align-items:center;justify-content:center">+ 关注</div>
        <div style="position:absolute;left:{TEXT_X}px;top:{TEXT_Y}px;right:20px;font-size:{TEXT_SIZE}px;line-height:{TEXT_LH}px">{lines}</div>
        <div style="position:absolute;left:0;right:0;top:{FOOT_Y}px;height:42px;display:flex;font-size:13.5px;color:{SUB};border-top:1px solid {LINES};border-bottom:1px solid {LINES}">{foot}</div>
        <div style="position:absolute;left:20px;right:20px;top:{FOOT_Y + 56}px;height:38px;border-radius:4px;border:1px solid #e6e6e6;box-sizing:border-box;color:{SUB};font-size:13px;padding:9px 12px">发布你的评论</div>
        <div style="position:absolute;left:0;right:0;top:{FOOT_Y + 110}px;text-align:center;color:{SUB};font-size:12.5px">还没有人评论</div></div>
      <div style="position:absolute;left:{SIDE_X}px;top:{CARD_Y}px;width:{SIDE_W}px;height:352px;background:#fff;border-radius:4px;padding:12px 14px;box-sizing:border-box">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px"><span style="font-size:15px;font-weight:600">微博热搜</span><span style="font-size:12px;color:{SUB}">点击刷新</span></div>
        {trends}
        <div style="margin-top:8px;height:30px;border-radius:4px;background:#f6f6f6;color:{SUB};font-size:12.5px;display:flex;align-items:center;justify-content:center">查看完整热搜榜单</div></div></div>'''


_screens = {}


def _screen(selected, reads):
    key = (selected, reads)
    if key not in _screens:
        _screens.clear()
        win_w, win_h = PLACE[2:]
        body = f'<div style="position:absolute;left:{PAD}px;top:{PAD + BAR}px;width:{win_w}px;height:{win_h - BAR}px;border-radius:0 0 14px 14px;overflow:hidden">{_body(selected, reads)}</div>'
        _screens[key] = learn.Screen(mac.browser_html("weibo.com", win_w, win_h, BAR) + body, "Safari浏览器", ("文件", "编辑", "显示", "历史记录", "书签", "窗口", "帮助"), ("Finder", "Claude", "Safari"), place=PLACE, show_dock=False)
    return _screens[key]


def _sm(u):
    u = np.clip(u, 0, 1)
    return float(u * u * (3 - 2 * u))


def weibo(t):
    """I go over the sentence; the count of people who have read it stays at three."""
    x = t - T
    k = float(np.clip((x - SELECT0) / (SELECT1 - SELECT0), 0, 1))
    s = _screen(int(k * len(LINE) + 1e-3), READS)
    px = 2 * Z                                                       # page pixels, at twice the frame's, to a point
    y = (CARD_Y + TEXT_Y + TEXT_LH * 3.25) * px
    a, b = ((MAIN_X + TEXT_X) * px, y), ((MAIN_X + TEXT_X + TEXT_SIZE * len(LINE)) * px, y)
    come = _sm((x - 0.4) / (SELECT0 - 0.4))
    start = (s.w * 0.55, s.h * 0.8)
    pointer = (start[0] + (a[0] - start[0]) * come + (b[0] - a[0]) * k, start[1] + (a[1] - start[1]) * come)
    near = _sm((x - NEAR0) / (NEAR1 - NEAR0))
    z = float(np.exp(np.log(CLOSE) * near))
    fx, fy = s.x + FOCUS[0] * px, s.y + FOCUS[1] * px
    return s.show(pointer=pointer, z=z, centre=(W + (fx - W) * near, H + (fy - H) * near))


def stills(ts):
    os.makedirs(OUT, exist_ok=True)
    for t in ts:
        Image.fromarray(ink_subs.apply(np.array(weibo(t)), t)).save(os.path.join(OUT, f"weibo_{t:07.2f}.png"))
        print(t, flush=True)


def clip():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "weibo.mp4")
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(int(T * 30), int(at("我本就是") * 30)):
        t = n / 30
        pipe.stdin.write(ink_subs.apply(np.array(weibo(t)), t).tobytes())
    pipe.stdin.close()
    pipe.wait()
    print(path)


if __name__ == "__main__":
    if sys.argv[1:] == ["clip"]:
        clip()
    else:
        stills([float(x) for x in sys.argv[1:]])
