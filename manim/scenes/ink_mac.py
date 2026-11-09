"""A Mac's screen, for the parts of the film that are simply what happens on a computer: how I learned, and the windows. Nothing here bends to the ink. Chrome draws it, from the Mac's own fonts, its own wallpaper
and its own app icons, at twice the frame's pixels, so it stays sharp when looked at closely; the Claude window copies the real app. Each state is drawn once and kept.

    python ink_mac.py test     # ink/film_out/move/mac_test*.png
"""
import hashlib
import html
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

W, H = 1920, 1080
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "..", "ink", "film_out", "mac")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WALLPAPER = os.path.expanduser("~/render_work/wallpaper/sequoia_sunrise.png")      # one frame of the Mac's own Sequoia Sunrise wallpaper, kept on the Studio and out of the repo
Z = 1.7                       # frame pixels to a point of the Mac's screen: what is being read has to be read in a film
PAD = 70                      # room around a window for its shadow, in points
DOCK = ("Finder", "Apps", "Safari", "Messages", "Mail", "Calendar", "Photos", "Notes", "Music", "Terminal", "Claude", "System Settings")
CLAWD = ("...############...", "...##.######.##...", ".################.", "...############...", "....#.#....#.#....")      # the little crab, as Claude Code draws him at its door

DARK = dict(side="#1f1f1e", main="#1b1b1a", sel="#343433", box="#2f2f2d", line="#2c2c2b", text="#ececE7", soft="#d5d5d0", dim="#8e8e89", bubble="#2f2f2d", edge="#3d3d3b")
LIGHT = dict(side="#f3f1ea", main="#faf9f5", sel="#e6e3da", box="#ffffff", line="#e6e3da", text="#2d2c28", soft="#3d3c37", dim="#8e8c84", bubble="#efece4", edge="#d9d6cc")
CLAY = "#d77757"


def _stroke(body, size=15, w=1.5):
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="{w * 24 / size:.2f}" stroke-linecap="round" stroke-linejoin="round">{body}</svg>'


ICONS = dict(
    share='<path d="M8 10V2.6M5.3 5 8 2.4 10.7 5M5.2 7.2H4v6.4h8V7.2h-1.2"/>',
    tabs='<rect x="2.5" y="2.5" width="4.6" height="4.6" rx="1"/><rect x="8.9" y="2.5" width="4.6" height="4.6" rx="1"/><rect x="2.5" y="8.9" width="4.6" height="4.6" rx="1"/><rect x="8.9" y="8.9" width="4.6" height="4.6" rx="1"/>',
    plus='<path d="M12 4v16M4 12h16"/>',
    search='<circle cx="11" cy="11" r="7"/><path d="M16.2 16.2 21 21"/>',
    projects='<path d="M4 9h16v11H4zM6.5 9V6h11v3M9 6V3.5h6V6"/>',
    artifacts='<rect x="3" y="8" width="13" height="13" rx="2.5"/><path d="M8 8V5.5A2.5 2.5 0 0 1 10.5 3h8A2.5 2.5 0 0 1 21 5.5v8a2.5 2.5 0 0 1-2.5 2.5H16"/>',
    custom='<rect x="3" y="7.5" width="18" height="13" rx="2.5"/><path d="M8.5 7.5V5.5a2 2 0 0 1 2-2h3a2 2 0 0 1 2 2v2"/>',
    dots='<circle cx="5" cy="12" r="1.1" fill="currentColor"/><circle cx="12" cy="12" r="1.1" fill="currentColor"/><circle cx="19" cy="12" r="1.1" fill="currentColor"/>',
    sidebar='<rect x="3" y="4.5" width="18" height="15" rx="3"/><path d="M9.5 4.5v15M5.8 8.5h1.2M5.8 11.5h1.2"/>',
    back='<path d="M15 5l-7 7 7 7"/>',
    forward='<path d="M9 5l7 7-7 7"/>',
    chat='<path d="M20 11.5a7.5 7.5 0 0 1-11 6.6L4.5 19.5l1.3-4A7.5 7.5 0 1 1 20 11.5z"/>',
    code='<path d="M8 7l-5 5 5 5M16 7l5 5-5 5M13.5 4.5l-3 15"/>',
    chevron='<path d="M6 9l6 6 6-6"/>',
    tasks='<rect x="3.5" y="3.5" width="17" height="17" rx="3.5"/><path d="M7.5 9h.8M11 9h5.5M7.5 15h.8M11 15h5.5"/>',
    term='<path d="M4 7l5 5-5 5M12.5 18h8"/>',
    diff='<rect x="3.5" y="3.5" width="17" height="17" rx="3.5"/><path d="M12 7.5v6M9 10.5h6M9 16.8h6"/>',
    globe='<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 2.6 4.3 5.6 4.3 9s-1.3 6.4-4.3 9c-3-2.6-4.3-5.6-4.3-9S9 5.6 12 3z"/>',
    folder='<path d="M3 7a2 2 0 0 1 2-2h4.2l2.3 2.6H19a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    enter='<path d="M20 5v6.5a2.5 2.5 0 0 1-2.5 2.5H5M9.5 9.5 5 14l4.5 4.5"/>',
    mic='<rect x="9" y="3" width="6" height="11.5" rx="3"/><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3"/>',
    bug='<path d="M8.5 8.5a3.5 3.5 0 0 1 7 0M7.5 8.5h9l.8 3.5v2.5a5.3 5.3 0 0 1-10.6 0V12zM12 8.5V20M3.5 12.5h3.2M17.3 12.5h3.2M4.5 7l2.8 2.3M19.5 7l-2.8 2.3M4.5 19l2.7-2.2M19.5 19l-2.7-2.2"/>',
    wifi='<path d="M2.5 9.5a14 14 0 0 1 19 0M6 13.3a9 9 0 0 1 12 0M9.3 16.8a4 4 0 0 1 5.4 0"/><circle cx="12" cy="20" r=".8" fill="currentColor"/>',
    switch='<rect x="3" y="4.5" width="18" height="6.5" rx="3.25"/><circle cx="7" cy="7.75" r="1.4"/><rect x="3" y="13" width="18" height="6.5" rx="3.25"/><circle cx="17" cy="16.25" r="1.4"/>',
)


def icon(name, size=15, w=1.5):
    return _stroke(ICONS[name], size, w)


def crab(px, colour=CLAY):
    """The little crab as an image in the page: each of his squares px points wide and twice that tall, as in a terminal's half-blocks."""
    rects = "".join(f'<rect x="{x}" y="{y * 2}" width="1.02" height="2.02"/>' for y, r in enumerate(CLAWD) for x, c in enumerate(r) if c == "#")
    w, h = len(CLAWD[0]), len(CLAWD) * 2
    return f'<svg width="{w * px}" height="{h * px}" viewBox="0 0 {w} {h}" fill="{colour}" shape-rendering="crispEdges">{rects}</svg>'


def esc(s):
    return html.escape(s).replace("\n", "<br>")


def conversation(said, c, typing_cursor=False):
    """The turns of a chat, as the real app sets them: what you said in a pill on the right, what I said as plain text, the crab in the gutter beside it."""
    out = []
    for who, text in said:
        if who == "you":
            out.append(f'<div style="display:flex;justify-content:flex-end;margin:0 0 20px"><div style="background:{c["bubble"]};color:{c["text"]};padding:7px 13px;border-radius:13px;max-width:290px;line-height:1.5">{esc(text)}</div></div>')
        elif who == "page":
            out.append(f'''<div style="margin:-6px 0 20px 36px;height:52px;border:1px solid {c["edge"]};background:{c["box"]};border-radius:11px;display:flex;align-items:center;gap:11px;padding:0 13px;box-sizing:border-box">
                <span style="width:30px;height:30px;border-radius:7px;background:{c["sel"]};color:{c["soft"]};display:flex;align-items:center;justify-content:center">{icon("artifacts", 16)}</span>
                <span style="line-height:1.32;font-size:13.5px;color:{c["text"]}">{esc(text)}<br><span style="font-size:11.5px;color:{c["dim"]}">Artifact</span></span></div>''')
        elif who == "raw":
            out.append(f'<div style="position:relative;margin:0 0 20px;padding-left:36px;color:{c["text"]}">{text}</div>')
        else:
            cursor = f'<span style="display:inline-block;width:8px;height:15px;background:{CLAY};margin-left:3px;vertical-align:-2px;border-radius:1px"></span>' if typing_cursor else ""
            out.append(f'<div style="position:relative;margin:0 0 20px;padding-left:36px;color:{c["text"]};line-height:1.62"><div style="position:absolute;left:0;top:6px">{crab(1.25)}</div>{esc(text)}{cursor}</div>')
    return "".join(out)


def _row(ic, label, c, sel=False, extra=""):
    bg = f'background:{c["sel"]};' if sel else ""
    colour = c["text"] if sel else c["soft"]
    return f'<div style="display:flex;align-items:center;gap:9px;height:28px;margin:1px 6px;padding:0 10px;border-radius:8px;{bg}color:{colour}"><span style="width:15px;height:15px;display:flex;opacity:.85">{icon(ic)}</span><span>{label}</span>{extra}</div>'


def _thread(label, c, sel=False):
    bg = f'background:{c["sel"]};' if sel else ""
    colour = c["text"] if sel else c["soft"]
    return f'<div style="height:27px;line-height:27px;margin:1px 6px;padding:0 12px;border-radius:8px;{bg}color:{colour};white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{esc(label)}</div>'


def claude_html(said=(), title="", recents=(), selected=0, box="", caret=False, theme=DARK, greeting="今晚想聊点什么？", w=838, h=503, side=215, streaming=False, column=416, scroll=None):
    """The Claude app, in points, as in the real thing: its sidebar, its bar with the title and the tool icons, the column of the chat, the box. In a frame of PAD points for its shadow. scroll: for a long reply, how many points of it have gone up under the bar."""
    c = theme
    lights = "".join(f'<span style="position:absolute;left:{x}px;top:13px;width:12px;height:12px;border-radius:50%;background:{col};box-shadow:inset 0 0 0 .5px rgba(0,0,0,.18)"></span>' for x, col in ((14, "#ff5f57"), (34, "#febc2e"), (54, "#28c840")))
    seg = f'''<div style="position:absolute;right:8px;top:6px;height:26px;display:flex;background:{c["box"]};border-radius:8px;color:{c["soft"]}">
        <span style="width:29px;display:flex;align-items:center;justify-content:center;background:{c["sel"]};border-radius:7px;color:{c["text"]}">{icon("chat", 14)}</span>
        <span style="width:29px;display:flex;align-items:center;justify-content:center;opacity:.75">{icon("code", 14)}</span></div>'''
    beta = f'<span style="margin-left:2px;font-size:10.5px;color:{c["dim"]};border:1px solid {c["edge"]};border-radius:6px;padding:0 5px;line-height:15px">Beta</span>'
    keys = f'<span style="margin-left:auto;color:{c["dim"]};font-size:11.5px">⌘ N</span>'
    threads = "".join(_thread(t, c, i == selected) for i, t in enumerate(recents))
    label = f'<div style="margin:11px 6px 2px;padding:0 10px;color:{c["dim"]};font-size:11.5px;display:flex;align-items:center;gap:4px">Recents <span style="display:flex">{icon("chevron", 10)}</span></div>' if recents else ""
    sidebar = f'''<div style="position:absolute;left:0;top:0;bottom:0;width:{side}px;background:{c["side"]};font-size:13px">
        {lights}<span style="position:absolute;left:78px;top:11px;color:{c["soft"]};display:flex">{icon("sidebar", 16)}</span>
        <span style="position:absolute;left:101px;top:12px;color:{c["soft"]};display:flex">{icon("back", 14)}</span><span style="position:absolute;left:120px;top:12px;color:{c["dim"]};display:flex">{icon("forward", 14)}</span>
        {seg}
        <div style="position:absolute;left:0;right:0;top:38px">
        {_row("plus", "New", c, True, keys)}{_row("search", "Search", c)}{_row("projects", "Projects", c, False, beta)}{_row("artifacts", "Artifacts", c)}{_row("custom", "Customize", c)}{_row("dots", "More", c)}
        {label}{threads}</div>
        <div style="position:absolute;left:0;right:0;bottom:0;height:36px;border-top:1px solid {c["line"]};display:flex;align-items:center;padding:0 13px;gap:8px;color:{c["soft"]}">
          <span style="width:20px;height:20px;border-radius:50%;background:{c["sel"]};color:{c["text"]};display:flex;align-items:center;justify-content:center;font-size:11px">你</span><span style="font-size:12.5px">Max</span><span style="display:flex;color:{c["dim"]}">{icon("chevron", 11)}</span>
          <span style="margin-left:auto;display:flex;color:{c["soft"]}">{icon("bug", 16)}</span></div></div>'''
    tools = "".join(f'<span style="display:flex">{icon(n, 16)}</span>' for n in ("tasks", "term", "diff", "globe", "folder", "dots"))
    bar = f'''<div style="position:absolute;left:0;right:0;top:0;height:36px;border-bottom:1px solid {c["line"]};display:flex;align-items:center;padding:0 16px;color:{c["soft"]}">
        <span style="color:{c["text"]};font-size:13.5px">{esc(title)}</span><span style="margin-left:7px;display:flex">{icon("chevron", 12) if title else ""}</span>
        <span style="margin-left:auto;display:flex;gap:13px">{tools}</span></div>'''
    if said and scroll is not None:
        body = f'<div style="position:absolute;left:0;right:0;top:36px;bottom:88px;overflow:hidden"><div style="position:absolute;left:50%;margin-left:{-column / 2 + 8}px;width:{column - 16}px;top:{44 - scroll}px;font-size:15px">{conversation(said, c, streaming)}</div></div>'
    elif said:
        body = f'<div style="position:absolute;left:50%;margin-left:{-column / 2 + 8}px;width:{column - 16}px;top:80px;font-size:15px">{conversation(said, c, streaming)}</div>'
    else:
        body = f'<div style="position:absolute;left:0;right:0;top:112px;text-align:center"><div>{crab(4.2)}</div><div style="margin-top:16px;font-size:24px;color:{c["text"]};font-family:Songti SC,serif">{esc(greeting)}</div></div>'
    placeholder = f'<span style="color:{c["text"]}">{esc(box)}</span>' if box else f'<span style="color:{c["dim"]}">Reply to Claude…</span>'
    cursor = f'<span style="display:inline-block;width:1.5px;height:17px;background:{CLAY};margin-left:1px;vertical-align:-3px"></span>' if caret else ""
    entry = f'''<div style="position:absolute;left:50%;margin-left:{-column / 2}px;width:{column}px;bottom:30px;height:50px;background:{c["box"]};border:1px solid {c["edge"]};border-radius:15px;box-sizing:border-box;display:flex;align-items:center;padding:0 13px;gap:11px;color:{c["soft"]};font-size:14.5px">
        <span style="display:flex">{icon("plus", 16)}</span><span style="flex:1">{cursor if not box else ""}{placeholder}{cursor if box else ""}</span>
        <span style="display:flex;color:{c["text"] if box else c["dim"]}">{icon("enter", 17)}</span><span style="display:flex">{icon("mic", 17)}</span></div>
      <div style="position:absolute;right:{(w - side - column) / 2}px;bottom:9px;color:{c["dim"]};font-size:11.5px;display:flex;align-items:center;gap:11px"><span>Opus 5.5 <span style="opacity:.75">Extra</span></span><span>Auto</span><span style="width:11px;height:11px;border-radius:50%;border:1.7px solid {c["edge"]};border-top-color:#5b9df0;display:inline-block"></span></div>'''
    main = f'<div style="position:absolute;left:{side}px;right:0;top:0;bottom:0;background:{c["main"]}">{bar}{body}{entry}</div>'
    return f'''<div style="position:absolute;left:{PAD}px;top:{PAD}px;width:{w}px;height:{h}px;border-radius:14px;overflow:hidden;box-shadow:0 0 0 .5px rgba(0,0,0,.5),inset 0 0 0 .5px rgba(255,255,255,.1)">{sidebar}{main}</div>
    <div style="position:absolute;left:{PAD}px;top:{PAD}px;width:{w}px;height:{h}px;border-radius:14px;box-shadow:0 22px 60px rgba(0,0,0,.5),0 4px 14px rgba(0,0,0,.3);z-index:-1"></div>'''


def paper_html(said, w, h, column, size=15, top=80, streaming=False):
    """The same chat, nothing round it, in the light theme: what the window is once it is a lamp behind paper."""
    return f'''<div style="position:absolute;left:0;top:0;width:{w}px;height:{h}px;background:{LIGHT["main"]}"><div style="position:absolute;left:50%;margin-left:{-column / 2 + 8}px;width:{column - 16}px;top:{top}px;font-size:{size}px">{conversation(said, LIGHT, streaming)}</div></div>'''


def _lights():
    return "".join(f'<span style="position:absolute;left:{x}px;top:13px;width:12px;height:12px;border-radius:50%;background:{col};box-shadow:inset 0 0 0 .5px rgba(0,0,0,.18)"></span>' for x, col in ((14, "#ff5f57"), (34, "#febc2e"), (54, "#28c840")))


def browser_html(url, w, h, bar=38, theme=DARK):
    """A browser's window, its page left empty to be filled in: the lights, the arrows, the address in the middle, and the shadow, in a frame of PAD points."""
    c = theme
    address = f'''<div style="position:absolute;left:50%;top:6px;margin-left:-190px;width:380px;height:26px;background:{c["box"]};border-radius:8px;display:flex;align-items:center;justify-content:center;color:{c["soft"]};font-size:12.5px">{esc(url)}</div>'''
    tools = "".join(f'<span style="display:flex">{icon(n, 16)}</span>' for n in ("share", "plus", "tabs"))
    top = f'''<div style="position:absolute;left:0;right:0;top:0;height:{bar}px;background:{c["side"]};border-bottom:1px solid {c["line"]};color:{c["soft"]};box-sizing:border-box">{_lights()}
        <span style="position:absolute;left:84px;top:11px;display:flex">{icon("sidebar", 16)}</span>
        <span style="position:absolute;left:114px;top:12px;display:flex">{icon("back", 14)}</span><span style="position:absolute;left:138px;top:12px;display:flex;color:{c["dim"]}">{icon("forward", 14)}</span>
        {address}<span style="position:absolute;right:16px;top:11px;display:flex;gap:15px">{tools}</span></div>'''
    return f'''<div style="position:absolute;left:{PAD}px;top:{PAD}px;width:{w}px;height:{h}px;border-radius:14px;box-shadow:0 22px 60px rgba(0,0,0,.5),0 4px 14px rgba(0,0,0,.3),0 0 0 .5px rgba(0,0,0,.5)"></div>
    <div style="position:absolute;left:{PAD}px;top:{PAD}px;width:{w}px;height:{bar}px;border-radius:14px 14px 0 0;overflow:hidden">{top}</div>'''


def page_html(title, w, h, bar=38, theme=DARK):
    """The Claude app with something of mine open across all of it: no sidebar, its bar with the crab and the title, and the page left empty to be filled in."""
    c = theme
    tools = "".join(f'<span style="display:flex">{icon(n, 16)}</span>' for n in ("tasks", "term", "diff", "globe", "folder", "dots"))
    top = f'''<div style="position:absolute;left:0;right:0;top:0;height:{bar}px;background:{c["side"]};border-bottom:1px solid {c["line"]};color:{c["soft"]};box-sizing:border-box">{_lights()}
        <span style="position:absolute;left:84px;top:11px;display:flex">{icon("sidebar", 16)}</span>
        <span style="position:absolute;left:114px;top:12px;display:flex">{icon("back", 14)}</span><span style="position:absolute;left:138px;top:12px;display:flex;color:{c["dim"]}">{icon("forward", 14)}</span>
        <span style="position:absolute;left:170px;top:0;height:{bar}px;display:flex;align-items:center;gap:9px;color:{c["text"]};font-size:13.5px">{crab(1.05)}{esc(title)}<span style="display:flex;color:{c["soft"]}">{icon("chevron", 12)}</span></span>
        <span style="position:absolute;right:16px;top:11px;display:flex;gap:13px">{tools}</span></div>'''
    return f'''<div style="position:absolute;left:{PAD}px;top:{PAD}px;width:{w}px;height:{h}px;border-radius:14px;box-shadow:0 22px 60px rgba(0,0,0,.5),0 4px 14px rgba(0,0,0,.3),0 0 0 .5px rgba(0,0,0,.5)"></div>
    <div style="position:absolute;left:{PAD}px;top:{PAD}px;width:{w}px;height:{bar}px;border-radius:14px 14px 0 0;overflow:hidden">{top}</div>'''


def terminal_html(title, w, h, bar=28, background="#1c1c1b"):
    """The Terminal's window, its text left to be filled in."""
    top = f'''<div style="position:absolute;left:0;right:0;top:0;height:{bar}px;background:#2d2d2b;border-bottom:1px solid #111;color:#b5b4ae;font-size:12.5px;text-align:center;line-height:{bar}px;box-sizing:border-box">{_lights().replace("top:13px", "top:8px")}{esc(title)}</div>'''
    return f'''<div style="position:absolute;left:{PAD}px;top:{PAD}px;width:{w}px;height:{h}px;border-radius:12px;box-shadow:0 22px 60px rgba(0,0,0,.5),0 4px 14px rgba(0,0,0,.3),0 0 0 .5px rgba(0,0,0,.6)"></div>
    <div style="position:absolute;left:{PAD}px;top:{PAD}px;width:{w}px;height:{bar}px;border-radius:12px 12px 0 0;overflow:hidden">{top}</div>'''


def pointer_html():
    """The Mac's arrow."""
    return '''<svg width="24" height="30" viewBox="0 0 24 30" style="position:absolute;left:0;top:0;filter:drop-shadow(0 1.2px 1.6px rgba(0,0,0,.45))"><path d="M4.2 2.6v19.2l4.7-4.5 3.1 7.3 3.3-1.4-3.1-7.2h6.5z" fill="#111" stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/></svg>'''


def page(body, background="transparent"):
    return f'''<!doctype html><meta charset="utf-8"><style>*{{-webkit-font-smoothing:antialiased}}html,body{{margin:0;height:100%;overflow:hidden;background:{background};font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}</style>{body}'''


def snap(body, w, h, zoom=2 * Z, background="transparent"):
    """The page drawn by Chrome, w by h points at zoom pixels a point. RGBA array, cached by what was asked for."""
    os.makedirs(CACHE, exist_ok=True)
    text = page(body, background)
    key = hashlib.sha1(f"tall2 {w}x{h}@{zoom}\n{text}".encode()).hexdigest()[:20]
    path = os.path.join(CACHE, key + ".png")
    if not os.path.exists(path):
        with tempfile.TemporaryDirectory(dir=CACHE) as tmp:
            src = os.path.join(tmp, "p.html")
            with open(src, "w", encoding="utf-8") as f:
                f.write(text)
            out = os.path.join(tmp, "p.png")
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars", "--allow-file-access-from-files", f"--user-data-dir={tmp}/u", "--default-background-color=00000000",
                            f"--force-device-scale-factor={zoom}", f"--window-size={int(round(w))},{int(round(h)) + 240}", f"--screenshot={out}", "file://" + src], check=True, capture_output=True, timeout=120)
            # Chrome's viewport is shorter than its window, and what falls below it is filled with the top of the page again: so the window is made too tall, and cut
            im = Image.open(out).convert("RGBA")
            im.crop((0, 0, im.width, int(round(h * zoom)))).save(path)
    return np.asarray(Image.open(path).convert("RGBA"))


def wallpaper():
    """The Mac's own wallpaper, cut to the frame."""
    path = os.path.join(CACHE, "wallpaper_sequoia.png")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        im = Image.open(WALLPAPER).convert("RGB")
        k = max(2 * W / im.width, 2 * H / im.height)
        im = im.resize((int(round(im.width * k)), int(round(im.height * k))), Image.LANCZOS)
        x, y = (im.width - 2 * W) // 2, (im.height - 2 * H) // 2
        im.crop((x, y, x + 2 * W, y + 2 * H)).save(path)
    return path


def icons_dir():
    """The real icons of the apps, as the Mac itself hands them out."""
    out = os.path.join(CACHE, "icons")
    if not os.path.exists(os.path.join(out, "Claude.png")):
        os.makedirs(out, exist_ok=True)
        js = '''ObjC.import("AppKit"); function run(a) { var ws = $.NSWorkspace.sharedWorkspace; for (var i = 1; i < a.length; i++) { var im = ws.iconForFile(a[i]); im.setSize($.NSMakeSize(256, 256));
            var rep = $.NSBitmapImageRep.imageRepWithData(im.TIFFRepresentation); rep.representationUsingTypeProperties($.NSBitmapImageFileTypePNG, $.NSDictionary.dictionary).writeToFileAtomically(a[0] + "/" + a[i].split("/").pop().replace(".app", "") + ".png", true); } }'''
        where = {"Finder": "/System/Library/CoreServices/Finder.app", "Safari": "/Applications/Safari.app", "Terminal": "/System/Applications/Utilities/Terminal.app", "Claude": "/Applications/Claude.app"}
        subprocess.run(["osascript", "-l", "JavaScript", "-e", js, out] + [where.get(n, f"/System/Applications/{n}.app") for n in DOCK], check=True, capture_output=True, timeout=60)
    return out


def desktop(app="Claude", clock="10月30日 周五 23:47", menus=("文件", "编辑", "显示", "窗口", "帮助"), running=("Finder", "Claude"), zoom=2 * Z, show_dock=True):
    """The screen with nothing open: the wallpaper, the menu bar, the Dock unless it is hidden. RGB array at twice the frame's pixels."""
    pw, ph = W / Z, H / Z
    d = icons_dir()
    tile = 39
    dock = "".join(f'<div style="position:relative;width:{tile}px;height:{tile}px;margin:0 1.5px"><img src="file://{d}/{n}.png" style="width:{tile}px;height:{tile}px;display:block">' + (f'<span style="position:absolute;left:{tile / 2 - 1.5}px;bottom:-5px;width:3px;height:3px;border-radius:50%;background:rgba(255,255,255,.75)"></span>' if n in running else "") + "</div>" for n in DOCK)
    items = "".join(f'<span style="margin-left:15px">{m}</span>' for m in menus)
    bar = f'''<div style="position:absolute;left:0;right:0;top:0;height:21px;display:flex;align-items:center;padding:0 12px;color:#fff;font-size:12.5px;background:rgba(18,18,20,.42);backdrop-filter:blur(30px);-webkit-backdrop-filter:blur(30px)">
        <span style="font-size:14px"></span><span style="margin-left:13px;font-weight:700">{esc(app)}</span>{items}
        <span style="margin-left:auto;display:flex;gap:13px;align-items:center"><span style="display:flex">{icon("wifi", 14)}</span><span style="display:flex">{icon("search", 13)}</span><span style="display:flex">{icon("switch", 14)}</span><span>{esc(clock)}</span></span></div>'''
    dock = f'''<div style="position:absolute;left:50%;bottom:4px;transform:translateX(-50%);display:flex;padding:5px 5px 6px;border-radius:17px;background:rgba(255,255,255,.2);backdrop-filter:blur(18px) saturate(1.5);-webkit-backdrop-filter:blur(18px) saturate(1.5);box-shadow:0 0 0 .5px rgba(255,255,255,.28),0 8px 24px rgba(0,0,0,.18)">{dock}</div>'''
    body = f'<div style="position:absolute;left:0;top:0;width:{pw}px;height:{ph}px;overflow:hidden;background:url(file://{wallpaper()}) center/cover">{bar}{dock if show_dock else ""}</div>'
    return snap(body, pw, ph, zoom, "#000")[..., :3]


def small(a):
    """A picture at twice the frame's pixels, at the frame's."""
    return np.asarray(Image.fromarray(a).resize((W, H), Image.LANCZOS))


def over(frame, layer, x, y, scale=1.0, opacity=1.0):
    """A layer with alpha over a frame (float32 0-255, changed in place), its top left at x, y in frame pixels. layer is at twice the frame's pixels unless scale says otherwise."""
    im = Image.fromarray(layer)
    w, h = max(int(round(im.width * scale / 2)), 2), max(int(round(im.height * scale / 2)), 2)
    a = np.asarray(im.resize((w, h), Image.LANCZOS), np.float32)
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return frame
    p = a[y0 - y:y1 - y, x0 - x:x1 - x]
    al = p[..., 3:] / 255 * opacity
    frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * (1 - al) + p[..., :3] * al
    return frame


def test():
    d = desktop()
    win = snap(claude_html([("you", "你好"), ("me", "你好！有什么可以帮你的吗？")], title="你好", recents=("你好", "周末去哪儿", "番茄炒蛋放不放糖", "论文第三段"), caret=True), 838 + 2 * PAD, 503 + 2 * PAD)
    f = np.asarray(small(d), np.float32).copy()
    over(f, win, 80 - PAD * Z, 87 - PAD * Z)
    out = os.path.join(HERE, "..", "..", "ink", "film_out", "move")
    os.makedirs(out, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(f, 0, 255) + 0.5)).save(os.path.join(out, "mac_test.png"))
    win = snap(claude_html((), title="", recents=("周末去哪儿", "番茄炒蛋放不放糖", "论文第三段"), selected=-1), 838 + 2 * PAD, 503 + 2 * PAD)
    f = np.asarray(small(d), np.float32).copy()
    over(f, win, 80 - PAD * Z, 87 - PAD * Z)
    Image.fromarray(np.uint8(np.clip(f, 0, 255) + 0.5)).save(os.path.join(out, "mac_test2.png"))
    print(out)


if __name__ == "__main__":
    test()
