"""The two screens among the imprints: 代码 and 帖子. What is on each screen, drawn by Chrome, and the shots put together from what film_screens3d.py rendered.

    python film_screens.py pages T [T ...]        # the screens at these film times: ink/film_out/film/screens/pages/
    python film_screens.py still SHOT T           # ink/film_out/film/screens/keys/SHOT.png; SHOT s4_screen, s4_b.., s4_key_up, s4_key_down, s5_a..
    python film_screens.py plan                   # the pages of the sequences, and what film_screens3d.py has to render for them: pages/plan_*.txt
    python film_screens.py seq NAME DIR [WIDTH]   # a sequence at 960 by 540, or WIDTH 1920 for the plates' own size; NAME s4_key, s5_a, s4_b or s5_b
    python film_screens.py two                    # 二's last carrier: the same Terminal, earlier that night, showing hello.c: pages/two_term.png, and where its p of printf is
    python film_screens.py still two_term 0       # its still, from the plates film_screens3d.py --shot=two_term made

S4 is what a person types into a Terminal, its banner and colours as Python 3.14.7 really prints them on this Mac: the escapes were read from a real session, 2026-11-01, gcd(1071, 462) really gives 21. Its second time (85.4) the person has gone and the shell asks whoami by itself. S5 is the post of film_weibo.py the moment before it was sent, in 2013: a browser's page on some ordinary LCD, in the layout weibo.com had that year as far as it can be remembered, a pixel a quarter of a millimetre wide, and nothing of the computer. Its second time (87.0) it has been sent, it sits in the feed and nothing moves, and under its last line the font's own 我是谁。 comes stroke by stroke.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

import film_hand as hand
import film_screens_tex as kt
import film_weibo as post
import ink_mac as mac

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "screens")
PAGES = os.path.join(OUT, "pages")
PLATES = os.path.join(OUT, "plates")
KEYS = os.path.join(OUT, "keys")

# ---------------------------------------------------------------- S4, the code

S4_T0, S4_CUT, S4_T1 = 70.9, 71.6, 72.2         # the screen, then the key, then the imprint
SW, SH = 2560, 1440                             # the Studio Display, in points
TERM_X, TERM_Y, COLS, ROWS = 56, 52, 80, 24
FONT, LINE = 14, 17
TERMINAL = "/System/Applications/Utilities/Terminal.app/Contents/Resources/Fonts"
# Terminal's own palette, with its Pro profile's text: bold is drawn in the bright colours
MAGENTA, BLUE, YELLOW, WHITE, TEXT, CURSOR = "rgb(249,53,248)", "rgb(88,51,255)", "rgb(173,173,39)", "#fff", "rgb(242,242,242)", "rgb(77,77,77)"
LAST = "gcd(1071, 462)"
TYPED_AT_CUT_IN, TYPED_ALL = 3, 71.52           # how much of the last line is there when the shot starts, and when all of it is
AGAIN_T0, AGAIN_T1 = 85.4, 87.0                 # the second time: nobody at the keys
ASKS = "whoami"
ASKED = (85.84, 86.02, 86.27, 86.58, 86.69, 86.89)     # when each letter comes: uneven, and slow
CYAN = "rgb(51,187,200)"                        # the Terminal's, for what Python's prompt knows as a builtin: exit
BLINK = 1.06                                    # a caret's on and off together, as the Mac's


def _p(colour, s, bold=True):
    return f'<span style="color:{colour};font-weight:{700 if bold else 400}">{s}</span>'


def typed(t):
    """How many letters of the last line are on the screen."""
    u = np.clip((t - S4_T0) / (TYPED_ALL - S4_T0), 0, 1)
    return int(TYPED_AT_CUT_IN + (len(LAST) - TYPED_AT_CUT_IN) * u + 1e-6)


def s4_last(n):
    """The line being typed, coloured as it comes: a number is yellow as soon as it starts."""
    s = LAST[:n]
    out = s[:4]
    if n > 4:
        out = "gcd(" + _p(YELLOW, s[4:8], False)
    if n > 8:
        out += s[8:10]
    if n > 10:
        out += _p(YELLOW, s[10:13], False)
    if n > 13:
        out += s[13:]
    return out


PROMPT = _p(MAGENTA, "&gt;&gt;&gt; ")


def s4_session():
    """What is on the screen before the last line."""
    more = _p(MAGENTA, "... ")
    kw = lambda s: _p(BLUE, s)
    banner = "Python 3.14.7 (main, Aug  5 2026, 10:29:49) [Clang 21.0.0 (clang-2100.1.1.101)] on darwin"     # as long as it is, the Terminal breaks it at its 80th column
    return ("Last login: Fri Oct 30 23:31:08 on ttys001", "~ % python3", banner[:COLS], banner[COLS:], 'Type "help", "copyright", "credits" or "license" for more information.',
            f'{PROMPT}{kw("def")} {_p(WHITE, "gcd")}(a, b):', f'{more}    {kw("while")} b:', f'{more}        a, b = b, a % b', f'{more}    {kw("return")} a', more)


def s4_html(n):
    return s4_terminal(s4_session() + (f'{PROMPT}{s4_last(n)}<span style="background:{CURSOR}"> </span>',), "python3", "23:51")


def s4_again_html(n, on):
    """gcd run and answered, Python left, and at the shell's prompt the letters that nobody types."""
    cursor = f'<span style="background:{CURSOR if on else "transparent"}"> </span>'
    return s4_terminal(s4_session() + (f'{PROMPT}{s4_last(len(LAST))}', "21", f'{PROMPT}{_p(CYAN, "exit", False)}()', f'~ % {ASKS[:n]}{cursor}'), "-zsh", "23:58")


def s4_terminal(lines, program, clock):
    w, h = int(COLS * FONT * .6) + 10, ROWS * LINE + 28
    faces = "".join(f'@font-face{{font-family:TMono;font-weight:{wt};src:url(file://{TERMINAL}/SF-Mono-{name}.otf)}}' for wt, name in ((400, "Regular"), (700, "Bold")))
    items = "".join(f'<span style="margin-left:15px">{m}</span>' for m in ("Shell", "编辑", "显示", "窗口", "帮助"))
    bar = f'''<div style="position:absolute;left:0;right:0;top:0;height:24px;display:flex;align-items:center;padding:0 14px;color:#fff;font-size:13px;background:rgba(18,18,20,.42);backdrop-filter:blur(30px);-webkit-backdrop-filter:blur(30px)">
        <span style="font-size:15px"></span><span style="margin-left:14px;font-weight:700">终端</span>{items}
        <span style="margin-left:auto;display:flex;gap:14px;align-items:center"><span style="display:flex">{mac.icon("wifi", 15)}</span><span style="display:flex">{mac.icon("search", 14)}</span><span style="display:flex">{mac.icon("switch", 15)}</span><span>10月30日 周五 {clock}</span></span></div>'''
    shell = f'<div style="position:absolute;left:{TERM_X - mac.PAD}px;top:{TERM_Y - mac.PAD}px">{mac.terminal_html(f"~ — {program} — {COLS}×{ROWS}", w, h)}</div>'
    text = "".join(f'<div style="height:{LINE}px">{x}</div>' for x in lines)
    inside = f'''<div style="position:absolute;left:{TERM_X}px;top:{TERM_Y + 28}px;width:{w}px;height:{h - 28}px;box-sizing:border-box;padding:3px 5px;border-radius:0 0 12px 12px;overflow:hidden;background:rgba(0,0,0,.85);color:{TEXT};font:400 {FONT}px/{LINE}px TMono,Menlo,monospace;white-space:pre">{text}</div>'''
    return f'<style>{faces}</style><div style="position:absolute;left:0;top:0;width:{SW}px;height:{SH}px;overflow:hidden;background:#000"><div style="position:absolute;left:0;top:0;width:100%;height:100%;filter:blur(1.6px);background:url(file://{mac.WALLPAPER}) center/cover"></div>{bar}{shell}{inside}</div>'


# 二's eighth carrier: Kernighan and Ritchie's first program, as their book prints it (The C Programming Language, 2nd ed., 1.1), cat'd in the same window earlier that night
HELLO_C = ("#include &lt;stdio.h&gt;", "", "main()", "{", '    printf("hello, world\\n");', "}")
TWO_SESSION = ("Last login: Fri Oct 30 23:02:41 on ttys000", "~ % cat hello.c") + HELLO_C
TWO_P = (len(TWO_SESSION) - 2, 4)               # the line and the column of printf's p


def two_html():
    cursor = f'<span style="background:{CURSOR}"> </span>'
    return s4_terminal(TWO_SESSION + (f"~ % {cursor}",), "-zsh", "23:03")


def two_at():
    """Where printf's p is on the screen, as parts of its width and height from the top left: the middle of its x-height."""
    row, col = TWO_P
    x = TERM_X + 5 + (col + .5) * FONT * .6
    y = TERM_Y + 28 + 3 + row * LINE + LINE * .62
    return x / SW, y / SH


def two_mid():
    """The middle of hello.c on the screen, from #include down to the prompt after it and across its longest line, as parts of the screen's width and height (film_screens3d --at)."""
    first, last = 2, len(TWO_SESSION)
    wide = max(len(line) for line in TWO_SESSION[first:])
    x = TERM_X + 5 + wide / 2 * FONT * .6
    y = TERM_Y + 28 + 3 + (first + last + 1) / 2 * LINE
    return x / SW, y / SH


def _s4_save(html, name):
    path = os.path.join(PAGES, name + ".png")
    if not os.path.exists(path):
        Image.fromarray(mac.snap(html, SW, SH, 2, "#000")[..., :3]).save(path)
    return path


def s4_page(t):
    n = typed(t)
    return _s4_save(s4_html(n), f"s4_{n:02d}")


def s4_again_page(t):
    n, on = asked(t)
    return _s4_save(s4_again_html(n, on), f"s4_again_{n}_{int(on)}")


def asked(t):
    """How many letters have come, and whether the cursor is on. It has been blinking at an idle prompt; a letter starts its blink over."""
    n = sum(at <= t for at in ASKED)
    since = max((at for at in ASKED if at <= t), default=AGAIN_T0 - .3)
    return n, ((t - since) * 2 / BLINK) % 2 < 1


# ---------------------------------------------------------------- S5, the post

S5_T0, S5_CLICK, S5_T1 = 72.7, 73.2, 73.3
SENT_T0, SENT_T1 = 87.0, 89.6                   # the second time: sent, and in the feed
PW, PH = 1366, 768                              # what an ordinary screen was in 2013, in its own pixels
CROP, LCD = kt.S5_CROP, kt.LCD
BLUE_LINK, ORANGE, ORANGE_DOWN = "#0a8cd2", "#ff8200", "#ee6f00"
TOOLBAR = 36                                    # the browser's own bar, above the page
BUTTON = (856, 266, 78, 30)                     # x, y, w, h of 发布, in the page's pixels
LEFT = 183                                      # the page's left edge
COLUMN = LEFT + 160, 600                        # the middle column, where it starts and how wide
FACE = "'Heiti SC','STHeiti','PingFang SC',sans-serif"
TEXTS = tuple(x.format(post.LINE) for x in post.POST)
LEFT_TO_WRITE = 140 - (sum(len(x) for x in TEXTS) + (len(TEXTS) - 1) // 2)     # the box counted a Chinese character as one and anything in ASCII, a line break, as half


POINTER = BUTTON[0] + BUTTON[2] * .76, BUTTON[1] + BUTTON[3] * .74     # its hand already on the button, beside the word, not on it
HAND = '<svg width="18" height="22" viewBox="0 0 18 22" style="position:absolute;left:{x}px;top:{y}px"><path d="M6.5 1.2c-.9 0-1.5.7-1.5 1.5v8.7L3.6 9.9c-.8-.8-2-.8-2.6-.1-.6.6-.5 1.5 0 2.2l3.9 5.6c.8 1.2 2 2.9 4.6 2.9h3c2.6 0 4.5-2 4.5-4.6v-5.1c0-.8-.6-1.4-1.3-1.4-.5 0-.8.2-1.1.5 0-.8-.6-1.4-1.3-1.4-.5 0-.9.2-1.1.5-.1-.8-.7-1.3-1.4-1.3-.4 0-.8.1-1 .4V2.7c0-.8-.6-1.5-1.4-1.5z" fill="#fff" stroke="#000" stroke-width="1"/></svg>'
TOOLS = (("表情", '<circle cx="8" cy="8" r="6.6" fill="#ffd257" stroke="#e8a21c"/><circle cx="5.7" cy="6.6" r="1" fill="#8a5a10"/><circle cx="10.3" cy="6.6" r="1" fill="#8a5a10"/><path d="M4.9 9.6c1.6 2.4 4.6 2.4 6.2 0" fill="none" stroke="#8a5a10"/>'),
         ("图片", '<rect x="1.5" y="2.5" width="13" height="11" rx="1" fill="#e8f3d8" stroke="#6ca832"/><path d="M2.5 12l3.6-4.4 2.6 2.8 2-1.8 3 3.4z" fill="#6ca832"/><circle cx="11" cy="5.6" r="1.2" fill="#f0a030"/>'),
         ("视频", '<rect x="1.5" y="3" width="13" height="10" rx="1.4" fill="#dbe6f8" stroke="#4a7fd4"/><path d="M6.4 5.6v4.8l4.2-2.4z" fill="#4a7fd4"/>'),
         ("话题", '<path d="M6 1.8 4.6 14.2M11.2 1.8 9.8 14.2M2.2 5.6h12M1.8 10.4h12" stroke="#8e6fc4" stroke-width="1.6" fill="none"/>'),
         ("长微博", '<path d="M3 1.5h7l3 3v10H3z" fill="#fde6dc" stroke="#d9603a"/><path d="M5 6.5h6M5 9h6M5 11.5h4" stroke="#d9603a"/>'))


def s5_html(t):
    down = t >= S5_CLICK
    lines = "".join(f'<div style="height:24px">{x}</div>' for x in TEXTS[:-1]) + f'<div style="height:24px">{TEXTS[-1]}<span style="display:inline-block;width:1px;height:16px;background:#333;vertical-align:-3px;margin-left:1px"></span></div>'
    if down:
        button = f"background:linear-gradient(#e96e00,{ORANGE_DOWN});box-shadow:inset 0 2px 3px rgba(120,50,0,.55);border:1px solid #c95c00;color:#fff3e4;padding-top:1px"
    else:
        button = f"background:linear-gradient(#ffb040,{ORANGE});box-shadow:inset 0 1px 0 rgba(255,255,255,.45),0 1px 1px rgba(0,0,0,.12);border:1px solid #e57400;color:#fff"
    cursor = HAND.format(x=round(POINTER[0]) - 6, y=round(POINTER[1]) - 1)
    x0, cw = COLUMN
    return s5_around(f'''{s5_box(0, lines, button, LEFT_TO_WRITE, "border:1px solid #f5a95a;box-shadow:inset 0 1px 3px rgba(0,0,0,.12),0 0 3px rgba(255,160,70,.5)")}
      {s5_tabs(320)}
      <div style="position:absolute;left:{x0}px;top:357px;width:{cw}px;height:420px;background:#fff;border:1px solid #c6d3df;border-top:none;box-sizing:border-box">
        </div>
      {cursor}''')


def s5_box(scrolled, lines, button, left, edge):
    """The box one writes in, with 发布: lines what is in it, left how many more it would take, edge how the field looks."""
    x0, cw = COLUMN
    bx, by, bw, bh = BUTTON
    top = 56 - scrolled
    tools = "".join(f'<span style="display:inline-flex;align-items:center;margin-right:16px"><svg width="16" height="16" viewBox="0 0 16 16" style="margin-right:4px">{c}</svg>{n}</span>' for n, c in TOOLS)
    return f'''<div style="position:absolute;left:{x0}px;top:{top}px;width:{cw}px;height:256px;box-sizing:border-box;background:#fff;border:1px solid #c6d3df;border-radius:3px;box-shadow:0 1px 2px rgba(0,0,0,.1)">
        <div style="position:absolute;left:12px;top:9px;font-size:15px;color:#8a8a8a">有什么新鲜事想告诉大家？</div>
        <div style="position:absolute;right:12px;top:11px;font-size:12px;color:#8a8a8a">还可以输入 <span style="font-size:16px;font-weight:700;color:#555;font-style:italic">{left}</span> 字</div>
        <div style="position:absolute;left:12px;right:12px;top:36px;height:160px;box-sizing:border-box;{edge};border-radius:2px;padding:8px 10px;font-size:15px;line-height:24px;color:#333">{lines}</div>
        <div style="position:absolute;left:14px;top:209px;font-size:12px;color:#555;display:flex;align-items:center">{tools}</div>
        <div style="position:absolute;left:{bx - x0 - 62}px;top:{by - 56 + 7}px;font-size:12px;color:#666">公开 ▾</div></div>
      <div style="position:absolute;left:{bx}px;top:{by - scrolled}px;width:{bw}px;height:{bh}px;box-sizing:border-box;border-radius:3px;{button};font:700 15px/28px {FACE};text-align:center;letter-spacing:2px">发布</div>'''


def s5_tabs(top):
    x0, cw = COLUMN
    tabs = "".join(f'<span style="margin-right:22px;color:{ORANGE if i == 0 else "#666"}">{n}</span>' for i, n in enumerate(("全部", "原创", "图片", "视频", "音乐")))
    return f'<div style="position:absolute;left:{x0}px;top:{top}px;width:{cw}px;height:38px;box-sizing:border-box;background:#fff;border:1px solid #c6d3df;border-radius:3px 3px 0 0;font-size:13px;line-height:36px;padding-left:14px">{tabs}</div>'


def s5_around(middle):
    """The page around its middle column."""
    x0, cw = COLUMN
    nav = "".join(f'<span style="margin-right:26px;color:{"#333" if i == 0 else "#666"};font-weight:{700 if i == 0 else 400}">{n}</span>' for i, n in enumerate(("首页", "视频", "发现", "游戏")))
    links = "".join(f'<div style="height:30px;line-height:30px;padding-left:16px;color:{"#333" if i == 0 else "#5d5d5d"};background:{"rgba(255,255,255,.55)" if i == 0 else "transparent"}">{n}</div>' for i, n in enumerate(("首页", "提到我的", "评论", "赞", "私信", "未关注人私信", "收藏")))
    return f'''<div style="position:absolute;left:0;top:0;width:{PW}px;height:{TOOLBAR}px;background:linear-gradient(#eceef1,#d9dce0);border-bottom:1px solid #a9adb3;box-sizing:border-box"></div>
      <div style="position:absolute;left:0;top:{TOOLBAR}px;width:{PW}px;height:{PH - TOOLBAR}px;overflow:hidden;background:linear-gradient(#8fb4d6,#b9d0e4 340px,#cbdbe9);color:#333;font:14px/1.4 {FACE}">
      <div style="position:absolute;left:0;right:0;top:0;height:42px;z-index:2;background:rgba(255,255,255,.93);border-bottom:1px solid #c9c9c9;box-shadow:0 1px 3px rgba(0,0,0,.14)">
        <div style="position:absolute;left:{LEFT}px;top:5px;line-height:1.05;font-weight:700;font-size:20px;color:#222">微博<div style="font-size:9px;font-weight:400;color:#888">weibo.com</div></div>
        <div style="position:absolute;left:{LEFT + 90}px;top:11px;font-size:14px">{nav}</div>
        <div style="position:absolute;left:{LEFT + 400}px;top:9px;width:220px;height:24px;box-sizing:border-box;border:1px solid #ccc;border-radius:2px;background:#fff;color:#aaa;font-size:12px;line-height:22px;padding-left:8px">搜索微博、找人</div>
        <div style="position:absolute;left:{LEFT + 780}px;top:11px;font-size:13px;color:#333">{post.WHO}</div></div>
      <div style="position:absolute;left:{LEFT}px;top:56px;width:150px;font-size:13px">{links}</div>
      <div style="position:absolute;left:{x0 + cw + 10}px;top:56px;width:230px;height:150px;box-sizing:border-box;background:rgba(255,255,255,.93);border-radius:3px;border:1px solid #c6d3df"></div>
      {middle}</div>'''


# ---------------------------------------------------------------- S5, sent

SCROLLED = 176                                  # the page has gone up this far, the bar staying, and of the box, empty now, only its foot shows
CARD_TOP, TEXT_LEFT, TEXT_TOP, TEXT_SIZE, TEXT_LH = 357 - SCROLLED, 76, 36, 15, 24     # in the page under the browser's bar and within the card
SENT_AT = "3月14日 23:52"                        # film_weibo.WHEN, as the feed wrote the same year
GROWN = "我是谁。"
GROWS = ((87.1, 88.0), (88.0, 88.6), (88.6, 89.1), (89.1, 89.2))     # when each character of it is written, film seconds; then it stops
FINE = 16                                       # how many times the panel's pixels the growth is worked out at


def s5_sent_html(grown):
    """The post, in the feed: nothing here moves, and nothing is pointed at. grown: with what grows under it, whole."""
    x0, cw = COLUMN
    face = '<svg width="50" height="50" viewBox="0 0 50 50" style="position:absolute;left:14px;top:14px"><rect width="50" height="50" fill="#e4e5e8"/><circle cx="25" cy="20" r="9" fill="#fff"/><path d="M7 50c0-13 8-20 18-20s18 7 18 20z" fill="#fff"/></svg>'
    lines = "".join(f'<div style="height:{TEXT_LH}px">{x}</div>' for x in TEXTS)
    under = f'<div style="height:{TEXT_LH}px">{GROWN}</div>' if grown else ""
    foot = TEXT_TOP + TEXT_LH * (len(TEXTS) + 1) + 8
    acts = " ".join(f'<span style="margin-left:8px;color:{BLUE_LINK}">{x}</span>' for x in ("赞", "转发", "收藏", "评论"))
    empty = "background:linear-gradient(#ffdcb3,#ffd29c);border:1px solid #f4cda0;color:#fff"     # with nothing to send it can't be pressed
    return s5_around(f'''{s5_box(SCROLLED, "", empty, 140, "border:1px solid #c9ced4;box-shadow:inset 0 1px 2px rgba(0,0,0,.08)")}
      {s5_tabs(320 - SCROLLED)}
      <div style="position:absolute;left:{x0}px;top:{CARD_TOP}px;width:{cw}px;height:420px;background:#fff;border:1px solid #c6d3df;border-top:none;box-sizing:border-box">{face}
        <div style="position:absolute;left:{TEXT_LEFT}px;top:13px;font-size:14px;line-height:18px;color:{BLUE_LINK}">{post.WHO}</div>
        <div style="position:absolute;left:{TEXT_LEFT}px;top:{TEXT_TOP}px;font-size:{TEXT_SIZE}px;line-height:{TEXT_LH}px;color:#333">{lines}{under}</div>
        <div style="position:absolute;left:{TEXT_LEFT}px;right:14px;top:{foot}px;display:flex;justify-content:space-between;font-size:12px;line-height:18px;color:#999"><span>{SENT_AT} 来自微博 weibo.com</span><span>{acts}</span></div></div>''')


def _written(c, ink, t0, t1):
    """When each point of a character comes, film seconds. ink: the font's own glyph, in a cell FINE times the panel's. Its centre lines, from the stroke data, are laid on the glyph, and each point of the glyph comes when the first of the strokes passing over it gets there: the shape is never the data's."""
    h, w = ink.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    solid = ink > .5
    ys, xs = np.nonzero(solid)
    if c not in hand._graphics():
        # 。 goes round from the top
        cx, cy = xs.mean(), ys.mean()
        return t0 + (t1 - t0) * ((np.arctan2(xx - cx, cy - yy) / (2 * np.pi)) % 1)
    em = TEXT_SIZE * FINE
    lines = hand.fit([m * em / 100 for m in hand._medians(c)], np.stack((xs, ys), 1)[::5].astype(np.float64), em)
    along = [np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))]) for p in lines]
    size = np.array([a[-1] for a in along])
    # the pen off the paper between two strokes takes its time too
    lift = .1 * em
    whole = size.sum() + lift * (len(lines) - 1)
    pts = np.stack((xx.ravel(), yy.ravel()), 1)
    reach = 1.3 * np.percentile(distance_transform_edt(solid)[solid], 98)
    best, nearest, gone = np.full(h * w, np.inf), np.full(h * w, np.inf), np.full(h * w, np.inf)
    for k, (p, a) in enumerate(zip(lines, along)):
        d, i = cKDTree(p).query(pts)
        when = (size[:k].sum() + lift * k + a[i]) / whole
        best = np.where(d <= reach, np.minimum(best, when), best)
        closer = d < nearest
        gone[closer], nearest[closer] = when[closer], d[closer]
    return (t0 + (t1 - t0) * np.where(best < np.inf, best, gone)).reshape(h, w)


_growing = []


def growth():
    """When every point of the grown line comes, and how much of it is ink, at FINE times the panel's pixels."""
    if not _growing:
        a = mac.snap(f'<div style="position:absolute;left:0;top:0;font:{TEXT_SIZE}px/{TEXT_LH}px {FACE};color:#000;white-space:nowrap">{GROWN}</div>', TEXT_SIZE * len(GROWN), TEXT_LH, FINE, "#fff")[:TEXT_LH * FINE, :TEXT_SIZE * len(GROWN) * FINE, :3]
        ink = 1 - a.mean(-1).astype(np.float32) / 255
        cell = TEXT_SIZE * FINE
        when = [_written(c, ink[:, i * cell:(i + 1) * cell], *GROWS[i]) for i, c in enumerate(GROWN)]
        _growing.extend((np.hstack(when), ink))
    return _growing


def grown_by(t):
    """How much of each panel pixel of the grown line has come by film time t."""
    when, ink = growth()
    w = ink + 1e-3
    done = (when <= t) * w
    block = lambda a: a.reshape(TEXT_LH, FINE, TEXT_SIZE * len(GROWN), FINE).sum((1, 3))
    return (block(done) / block(w))[..., None].astype(np.float32)


def s5_sent_page(t):
    blank = mac.snap(s5_sent_html(False), PW, PH, 1, "#000")[..., :3].astype(np.float32)
    whole = mac.snap(s5_sent_html(True), PW, PH, 1, "#000")[..., :3].astype(np.float32)
    x0, y0 = COLUMN[0] + TEXT_LEFT + 1, TOOLBAR + CARD_TOP + TEXT_TOP + TEXT_LH * len(TEXTS)
    a = blank.copy()
    h, w = TEXT_LH, TEXT_SIZE * len(GROWN)
    k = grown_by(t)
    a[y0:y0 + h, x0:x0 + w] += (whole[y0:y0 + h, x0:x0 + w] - blank[y0:y0 + h, x0:x0 + w]) * k
    return _s5_save(np.uint8(np.clip(a, 0, 255) + .5), t)


def lcd(a):
    """A picture of the panel's pixels as the panel makes them: three stripes side by side in each, a dark line round it. Returned in sRGB, and how much to multiply its light by to come back to what it showed."""
    n = LCD
    m = np.zeros((n, n, 3), np.float32)
    for c, (x0, x1) in enumerate(((0, 2), (3, 5), (6, 8))):
        m[:7, x0:x1, c] = 1
    m[:7, 2, :2] = .18
    m[:7, 5, 1:] = .18
    m = .42 * m + .58 * m.mean((0, 1), keepdims=True)     # the panel's own diffuser and the lens: the stripes are seen, but not as a coloured grid
    gain = 1 / m.mean((0, 1)).mean()
    h, w = a.shape[:2]
    big = np.repeat(np.repeat(to_linear(a), n, 0), n, 1)
    big *= np.tile(m, (h, w, 1))
    return np.uint8(np.clip(to_srgb(big), 0, 1) * 255 + .5), float(gain)


def to_linear(a):
    x = a.astype(np.float32) / 255
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def to_srgb(a):
    a = np.clip(a, 0, 1)
    return np.where(a <= .0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - .055)


def s5_page(t):
    return _s5_save(mac.snap(s5_html(t), PW, PH, 1, "#000")[..., :3], t)


def _s5_save(a, t):
    x0, y0, x1, y1 = CROP
    big, gain = lcd(a[y0:y1, x0:x1])
    path = os.path.join(PAGES, f"s5_{int(round(t * 100)):05d}.png")
    Image.fromarray(big).save(path, compress_level=1)
    with open(os.path.join(PAGES, "s5_gain.txt"), "w") as f:
        f.write(str(gain))
    return path


# ---------------------------------------------------------------- the shots

def film(x):
    a, b, c, d, e = 2.51, .03, 2.43, .59, .14
    x = np.maximum(x, 0)
    return np.clip(x * (a * x + b) / (x * (c * x + d) + e), 0, 1)


def finish(y, seed, grain):
    p = to_srgb(y)
    lum = p.mean(-1, keepdims=True)
    n = np.random.default_rng(seed).normal(0, 1, p.shape[:2]).astype(np.float32)
    n = gaussian_filter(n, .55) * 1.5
    p = p + n[..., None] * (grain * (1.2 - lum) + .0016)
    return np.uint8(np.clip(p, 0, 1) * 255 + .5)


def plate(name):
    return np.load(os.path.join(PLATES, name + ".npy")).astype(np.float32)


def falloff(h, w, k):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((xx - w / 2) / w, (yy - h / 2) / w)
    return (1 - k * r * r)[..., None]


def still(shot, t):
    if shot.startswith("s4_key"):
        y = film(plate(shot) * .85)
        y = y * falloff(*y.shape[:2], 1.5)
        f = finish(y, int(t * 30), .012)
    elif shot == "two_term":
        room, panel = plate("two_term_room"), plate("two_term_panel")
        y = film(room * 1.15)
        y = panel + y - panel * y
        y = y * falloff(*y.shape[:2], .35)
        f = finish(y, int(t * 30), .012)
    elif shot.startswith("s4_"):
        # the panel's light, sharp where the lens is, and over it what the glass gives back of a room with one lamp
        room, panel = plate("s4_screen_room"), plate(shot + "_panel")
        y = film(room * 1.15)
        y = panel + y - panel * y
        y = y * falloff(*y.shape[:2], .35)
        f = finish(y, int(t * 30), .012)
    elif shot.startswith("s5_"):
        # nothing else is lit: the page, the way a camera opened up for a dark room sees it, a little flooded, falling away from the middle
        s = plate(shot + "_panel") * (.51 if t < S5_T1 else .4335)        # the second look, nearer and with more white, has to meet the dark terminal it follows
        glow = gaussian_filter(s, (28, 28, 0)) * .05 + gaussian_filter(s, (110, 110, 0)) * .05
        y = (s + glow) * falloff(*s.shape[:2], 2.7)
        y = y * np.array((.97, .985, 1.03), np.float32)
        f = finish(np.clip(y, 0, 1), int(t * 30), .012)
    else:
        raise SystemExit(__doc__)
    return f     # t here is on the bridge's table, not the film's clock, and the shot burns no line of its own: the cut does that


SEQS = dict(s4_key=(70.9, 72.2), s5_a=(72.7, 73.3), s4_b=(85.4, 87.0), s5_b=(87.0, 89.6))
FPS = 30


def frames(name):
    a, b = SEQS[name]
    return [a + i / FPS + 1e-6 for i in range(int(round((b - a) * FPS)))]


def showing(t):
    """What a frame shows: the still's name, how film_screens3d.py makes it, and from what."""
    if t < S4_CUT:
        return f"s4_t{typed(t):02d}", "s4_screen", s4_page
    if t < S4_T1:
        return f"s4_key_m{int(round(kt.travel(t) * 1e6)):04d}", "s4_key", None
    if t < S5_T1:
        return f"s5_{'down' if t >= S5_CLICK else 'over'}", "s5_post", s5_page
    if t < AGAIN_T1:
        n, on = asked(t)
        return f"s4_x{n}{int(on)}", "s4_screen", s4_again_page
    return f"s5_g{int(round(t * FPS)):04d}", "s5_sent", s5_sent_page


def plan():
    """Every picture the sequences need, once, in a list for each way of making them: what it shows, and its plate."""
    jobs = {}
    for name in SEQS:
        for t in frames(name):
            shot, how, page = showing(t)
            if shot not in jobs.setdefault(how, {}):
                jobs[how][shot] = (page(t) if page else str(t), shot if how == "s4_key" else shot + "_panel")
    for how, some in jobs.items():
        with open(os.path.join(PAGES, f"plan_{how}.txt"), "w") as f:
            f.writelines(f"{a}\t{b}\n" for a, b in some.values())
        print(how, len(some), flush=True)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    os.makedirs(PAGES, exist_ok=True)
    os.makedirs(KEYS, exist_ok=True)
    if what == "pages":
        for a in sys.argv[2:]:
            t = float(a)
            print((s4_page if t < S4_T1 else s5_page if t < S5_T1 else s4_again_page if t < AGAIN_T1 else s5_sent_page)(t), flush=True)
    elif what == "plan":
        plan()
    elif what == "seq":
        name, out = sys.argv[2], sys.argv[3]
        width = int(sys.argv[4]) if len(sys.argv) > 4 else 960        # 1920 for the 1080 pass: the plates' own size, the same pictures not made smaller
        os.makedirs(out, exist_ok=True)
        for i, t in enumerate(frames(name)):
            im = Image.fromarray(still(showing(t)[0], t))
            if im.size != (width, width * 9 // 16):
                im = im.resize((width, width * 9 // 16), Image.LANCZOS)
            im.save(os.path.join(out, f"{name}_{i:04d}.png"), compress_level=1)
        print(name, len(frames(name)), flush=True)
    elif what == "two":
        path = _s4_save(two_html(), "two_term")
        print(path, "at %.5f %.5f" % two_at(), flush=True)
    elif what == "still":
        shot, t = sys.argv[2], float(sys.argv[3])
        path = os.path.join(KEYS, shot + ".png")
        Image.fromarray(still(shot, t)).save(path)
        print(path)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
