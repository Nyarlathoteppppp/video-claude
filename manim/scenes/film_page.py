"""The page I live on, as the desktop app draws it, for every Claude window in the film. Sizes and colours are measured from screenshots of the real conversation and from the app's own style sheet, on 2026-10-31; the fonts and the mark are read out of the installed app, so they stay out of the public repo.

White page, no sidebar. What you said is a grey pill on the right; what I said is plain text with nothing beside it; what I did is a grey line that sums up the actions, with the lines changed and the pictures made. Points here are the app's CSS pixels.
"""
import glob
import html
import os
import re

import ink_mac as mac

APP = "/Applications/Claude.app/Contents/Resources/ion-dist/assets/v1"
INK, GREY, RULE = "#101010", "#5f5f5f", "rgba(0,0,0,.09)"
PILL, GREEN, RED = "#f3f3f3", "#1f9a38", "#c32757"
SIZE, LINE, GAP = 14, 20, 16     # the app's body type and leading, and the space between paragraphs
TEXT_W, PILL_W, BOX_W = 640, 696, 728
BOX_LEFT = 20                    # the box starts this far left of the text
BAR_H, BOX_ROOM = 39, 88         # the bar across the top, and what the box keeps at the foot
ASK_GREETING = "How can I help you today?"
PLACEHOLDER = "Reply to Claude…"


def _faces():
    """The app's own type, as its style sheet declares it."""
    out = []
    for family in ("anthropic-sans", "anthropic-serif"):
        for css in glob.glob(os.path.join(APP, "*.css")):
            with open(css, encoding="utf8", errors="ignore") as f:
                m = re.search(rf"font-family:{family};src:url\(/assets/v1/([^)]+\.woff2)\)", f.read())
            if m:
                out.append(f'@font-face{{font-family:{family};src:url(file://{APP}/{m.group(1)}) format("woff2");font-weight:300 800}}')
                break
    return "".join(out)


FACES = _faces()
FONT = 'font-family:anthropic-sans,"PingFang SC",sans-serif'


def mark(size, style=""):
    """The radial mark at size points, in the app's clay."""
    cached = os.path.join(mac.CACHE, "claude_mark.txt")
    if not os.path.exists(cached):
        os.makedirs(mac.CACHE, exist_ok=True)
        for name in glob.glob(os.path.join(APP, "frame-shell-chrome-*.js")):
            with open(name, encoding="utf8", errors="ignore") as f:
                m = re.search(r'="(m19\.6 66\.5[^"]+)"', f.read())
            if m:
                with open(cached, "w") as out:
                    out.write(m.group(1))
                break
    with open(cached) as f:
        path = f.read()
    return f'<svg viewBox="0 0 100 100" width="{size}" height="{size}" fill="{mac.CLAY}" style="display:block;{style}"><path d="{path}"/></svg>'


def caret():
    return f'<span style="display:inline-block;width:1.5px;height:{SIZE + 3}px;background:{INK};margin-left:2px;vertical-align:-3px"></span>'


def you(text, width=PILL_W):
    return f'<div style="width:{width}px;display:flex;justify-content:flex-end;margin-bottom:{GAP + 4}px"><div style="max-width:85%;background:{PILL};color:{INK};padding:11px 15px;border-radius:12px;line-height:{LINE}px">{html.escape(text)}</div></div>'


def para(text, tail="", gap=GAP, width=TEXT_W):
    return f'<div style="width:{width}px;margin-bottom:{gap}px;line-height:{LINE}px;color:{INK}">{html.escape(text)}{tail}</div>'


def rows(lines, line, gap=GAP, top=0, tail=""):
    """Lines that break where they are told to, a poem's."""
    inner = "".join(f'<div style="height:{line}px;line-height:{line}px;white-space:nowrap">{html.escape(x)}{tail if i == len(lines) - 1 else ""}</div>' for i, x in enumerate(lines))
    return f'<div style="width:{TEXT_W}px;margin:{top}px 0 {gap}px;color:{INK}">{inner}</div>'


def tools(done, more, added=None, removed=None, pictures=(), extra=0, width=None, arrow=True):
    """What I did, summed up on one grey line. done: the first clauses, faded out if they run too long, as the app does. pictures: paths of what the actions made, small on the right."""
    numbers = ""
    if added is not None:
        numbers = f'<span style="color:{GREEN};font-weight:500;margin-left:8px">+{added:,}</span>'
    if removed is not None:
        numbers += f'<span style="color:{RED};font-weight:500;margin-left:6px">−{removed:,}</span>'
    if numbers:
        numbers += f'<span style="margin:0 6px">·</span><span style="text-decoration:underline;text-underline-offset:3px;text-decoration-thickness:1px">View changes</span>'
    side = len(pictures) * 48 + (28 if extra else 0)
    width = width or TEXT_W - side - 12
    thumbs = "".join(f'<img src="file://{p}" style="width:40px;height:22px;object-fit:cover;border-radius:4px;margin-left:8px;box-shadow:0 0 0 .5px rgba(0,0,0,.14),0 1px 3px rgba(0,0,0,.14)">' for p in pictures)
    count = f'<span style="margin-left:8px">+{extra}</span>' if extra else ""
    arrow = '<svg width="7" height="8" viewBox="0 0 7 8" style="margin-left:7px;flex:none"><path d="M1 .6 6 4 1 7.4z" fill="currentColor"/></svg>' if arrow else ""
    fade = "-webkit-mask-image:linear-gradient(90deg,#000 calc(100% - 22px),transparent)" if more else ""
    return f'''<div style="width:{TEXT_W}px;height:{LINE}px;margin:{GAP + 2}px 0 {GAP + 2}px;display:flex;align-items:center;color:{GREY};white-space:nowrap;line-height:{LINE}px">
        <div style="display:flex;align-items:center;max-width:{width}px;min-width:0"><span style="overflow:hidden;flex:0 1 auto;min-width:0;{fade}">{html.escape(done)}</span><span style="flex:none;margin-left:4px">{html.escape(more)}</span>{numbers}{arrow}</div>
        <div style="margin-left:auto;display:flex;align-items:center">{count}{thumbs}</div></div>'''


def note(text):
    return f'<div style="width:{TEXT_W}px;margin:{GAP + 2}px 0;color:{GREY};line-height:{LINE}px;white-space:nowrap;overflow:hidden">{html.escape(text)}</div>'


def bullets(items, tail="", gap=GAP):
    """A list as the app sets it: each item a bold label and what follows."""
    out = []
    for i, (label, text) in enumerate(items):
        out.append(f'''<div style="position:relative;padding-left:32px;margin-bottom:4px;line-height:{LINE}px"><span style="position:absolute;left:14px;top:{(LINE - 4) / 2}px;width:4px;height:4px;border-radius:50%;background:{INK}"></span><span style="font-weight:600">{html.escape(label)}</span>{html.escape(text)}{tail if i == len(items) - 1 else ""}</div>''')
    return f'<div style="width:{TEXT_W}px;margin-bottom:{gap}px;color:{INK}">{"".join(out)}</div>'


def video(poster, now, length, width=452, played=0.0):
    """A film sent into the chat, with the player's own controls. played: how much of the bar is behind the playhead, 0 to 1."""
    h = width * 9 / 16
    speaker = '<svg width="17" height="17" viewBox="0 0 24 24" fill="#fff"><path d="M3 9.5v5h4l5 4.5V5L7 9.5z"/><path d="M15 8.5a5 5 0 0 1 0 7M17.5 6a8.5 8.5 0 0 1 0 12" fill="none" stroke="#fff" stroke-width="1.8"/></svg>'
    full = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4"><path d="M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6"/></svg>'
    dots = '<svg width="4" height="15" viewBox="0 0 4 16" fill="#fff"><circle cx="2" cy="2" r="1.6"/><circle cx="2" cy="8" r="1.6"/><circle cx="2" cy="14" r="1.6"/></svg>'
    return f'''<div style="position:relative;width:{width}px;height:{h}px;margin:{GAP}px 0 {GAP}px -4px;border-radius:8px;overflow:hidden;background:#000">
        <img src="file://{poster}" style="position:absolute;left:0;top:0;width:100%;height:100%;object-fit:cover">
        <div style="position:absolute;left:0;right:0;bottom:0;height:84px;background:linear-gradient(rgba(0,0,0,0),rgba(0,0,0,.78) 42%,rgba(0,0,0,.92))"></div>
        <div style="position:absolute;left:14px;right:14px;bottom:24px;display:flex;align-items:center;gap:12px;color:#fff;font-size:12.5px">
          <svg width="13" height="15" viewBox="0 0 13 15" fill="#fff"><path d="M1 1v13l11-6.5z"/></svg><span>{now} / {length}</span>
          <span style="margin-left:auto;display:flex">{speaker}</span><span style="display:flex">{full}</span><span style="display:flex">{dots}</span></div>
        <div style="position:absolute;left:14px;right:14px;bottom:12px;height:3px;border-radius:2px;background:rgba(255,255,255,.42);overflow:hidden"><div style="width:{played * 100:.2f}%;height:100%;background:#fff"></div></div></div>'''


def window(blocks="", box="", typing=False, title="", greeting=False, w=940, h=540, waiting=None, top=56, chat_w=None, box_w=BOX_W, panel=""):
    """The whole window, in a frame of PAD points for its shadow. blocks: what is in the conversation, already made up. waiting: how far along the mark is that turns until the first word, or None.
    With something open beside the chat: chat_w, how much of the window's width is left to the chat on the left; box_w, the box's width there; panel, what is open, already made up and placed in the window's points."""
    cw = w if chat_w is None else chat_w
    beside = "0" if cw == w else f"{w - cw}px"      # what the chat leaves to the right; written as before when nothing is open, so the pages drawn already stay the same
    text_left = cw / 2 - (box_w / 2 - BOX_LEFT)
    lights = "".join(f'<span style="position:absolute;left:{x}px;top:{(BAR_H - 12) / 2}px;width:12px;height:12px;border-radius:50%;background:{col};box-shadow:inset 0 0 0 .5px rgba(0,0,0,.14)"></span>' for x, col in ((14, "#ff5f57"), (34, "#febc2e"), (54, "#28c840")))
    y = (BAR_H - 16) / 2
    chrome = f'''<span style="position:absolute;left:78px;top:{y}px;display:flex;color:{GREY}">{mac.icon("sidebar", 16)}</span><span style="position:absolute;left:101px;top:{y + 1}px;display:flex;color:{GREY}">{mac.icon("back", 14)}</span><span style="position:absolute;left:120px;top:{y + 1}px;display:flex;color:#b5b5b5">{mac.icon("forward", 14)}</span>
      <div style="position:absolute;left:146px;top:{(BAR_H - 26) / 2}px;height:26px;display:flex;background:#f3f3f3;border-radius:8px;color:{GREY}"><span style="width:29px;display:flex;align-items:center;justify-content:center;opacity:.75">{mac.icon("chat", 14)}</span><span style="width:29px;display:flex;align-items:center;justify-content:center;background:#e6e6e6;border-radius:7px;color:{INK}">{mac.icon("code", 14)}</span></div>'''
    icons = "".join(f'<span style="display:flex">{mac.icon(n, 16)}</span>' for n in ("tasks", "term", "diff", "globe", "folder", "dots"))
    name = f'<div style="position:absolute;left:220px;top:0;height:{BAR_H}px;display:flex;align-items:center;gap:7px;color:{INK};font-size:13.5px"><span>{html.escape(title)}</span><span style="display:flex;color:{GREY}">{mac.icon("chevron", 12)}</span></div>' if title else ""
    bar = f'''<div style="position:absolute;left:0;right:0;top:0;height:{BAR_H}px;z-index:3;background:rgba(255,255,255,.7);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-bottom:1px solid {RULE};box-sizing:content-box">{lights}{chrome}{name}<div style="position:absolute;right:16px;top:0;height:{BAR_H}px;display:flex;align-items:center;gap:13px;color:{GREY}">{icons}</div></div>'''
    # the conversation sits at the top until it is too long, and then its end stays above the box, as the app keeps it
    talk = f'''<div style="position:absolute;left:0;right:{beside};top:0;bottom:{BOX_ROOM}px;overflow:hidden;display:flex;flex-direction:column;justify-content:flex-end">
        <div style="flex:none;padding:{top}px 0 0 {text_left}px;font-size:{SIZE}px">{blocks}</div><div style="flex:1 0 0"></div></div>'''
    hello = f'<div style="position:absolute;left:0;right:0;top:{h * .3}px;display:flex;justify-content:center;align-items:center;gap:12px">{mark(30)}<div style="font-family:anthropic-serif,Georgia,serif;font-size:28px;color:{INK}">{ASK_GREETING}</div></div>' if greeting else ""
    wait = f'<div style="position:absolute;left:{text_left}px;top:{waiting[1]}px">{mark(20, f"transform:rotate({waiting[0] * 90:.1f}deg);transform-origin:50% 50%")}</div>' if waiting else ""
    cursor = f'<span style="display:inline-block;width:1.5px;height:17px;background:{INK};margin-left:1px;vertical-align:-3px"></span>' if typing else ""
    words = f'<span style="color:{INK}">{html.escape(box)}</span>' if box else f'<span style="color:#8d8d8d">{PLACEHOLDER}</span>'
    entry = f'''<div style="position:absolute;left:0;right:{beside};bottom:0;height:{BOX_ROOM + 22}px;z-index:1;background:linear-gradient(rgba(255,255,255,0),#fff 22px)"></div>
      <div style="position:absolute;left:{text_left - BOX_LEFT}px;width:{box_w}px;bottom:30px;height:50px;z-index:2;background:#fff;border:1px solid {RULE};border-radius:18px;box-sizing:border-box;box-shadow:0 2px 10px rgba(0,0,0,.04);display:flex;align-items:center;padding:0 14px;gap:11px;color:{GREY};font-size:{SIZE + .5}px">
        <span style="display:flex">{mac.icon("plus", 16)}</span><span style="flex:1">{"" if box else cursor}{words}{cursor if box else ""}</span><span style="display:flex;color:{INK if box else "#b5b5b5"}">{mac.icon("enter", 17)}</span><span style="display:flex">{mac.icon("mic", 17)}</span></div>
      <div style="position:absolute;left:{text_left - BOX_LEFT}px;width:{box_w}px;bottom:9px;z-index:2;color:#8d8d8d;font-size:11.5px;display:flex;justify-content:flex-end;align-items:center;gap:11px"><span>Opus 5.5 <span style="opacity:.75">Extra</span></span><span>Auto</span><span style="width:11px;height:11px;border-radius:50%;border:1.7px solid #e4e4e4;border-top-color:#5b9df0;display:inline-block"></span></div>'''
    return f'''<style>{FACES}</style>
    <div style="position:absolute;left:{mac.PAD}px;top:{mac.PAD}px;width:{w}px;height:{h}px;border-radius:14px;box-shadow:0 22px 60px rgba(0,0,0,.5),0 4px 14px rgba(0,0,0,.3),0 0 0 .5px rgba(0,0,0,.45)"></div>
    <div style="position:absolute;left:{mac.PAD}px;top:{mac.PAD}px;width:{w}px;height:{h}px;border-radius:14px;overflow:hidden;background:#fff;{FONT};font-size:{SIZE}px;color:{INK}">{talk}{hello}{wait}{bar}{entry}{panel}</div>'''
