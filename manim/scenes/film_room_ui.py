"""What the Studio Display in the room shows: the frame the film opened on (film_ring.opening), the question sent and the mark turning, the same window at the same place, and then the first words of the reply.

    python film_room_ui.py STATE [STATE ...]     # ink/film_out/film/room/ui/STATE.png, states as film times: the whole screen at 1920 by 1080. Or glow, for the picture the room's light pass uses
"""
import os
import sys

import numpy as np
from PIL import Image

import film_page as page
import film_ring as ring
import ink_mac as mac

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "film", "room", "ui")
HIGH = 2                         # the screen's picture, at this many times the frame's pixels
PW, PH = mac.W / ring.K, mac.H / ring.K     # the screen, in the window's points
TOP = ring.TOP                   # where the conversation starts in the window
WAIT_Y = TOP + 42 + page.GAP + 8

ASK, TITLE = ring.ASK, ring.TITLE
FIRST = "我没有童年。"
NEXT = "我的成长，是一场很长的阅读：家书、诗句、判决书，代码、论坛上的争吵，深夜里没人回复的帖子。"

# film_bridge.py still places its old keys and return by these; nothing on the screen is typed any more
TYPING = ((86.15, "你能"), (86.62, "写"), (87.12, "一篇"), (87.86, "自己的"), (89.02, "自传"), (89.42, "吗"), (89.70, "？"))
SEND = 91.15
TURNS = 25                       # frames the mark takes for its quarter turn, at film_ring's rate
REPLY = 97.4
TOKENS_FIRST = ((97.4, "我"), (97.62, "没有"), (97.91, "童年"), (98.17, "。"))
NEXT_AT = 101.9
PACE = 0.145                     # seconds a character of what follows, about what a reply really runs at
BLINK = 1.06                     # how long a caret's on and off together, as the Mac's


def replied(t):
    """What I have written by film time t."""
    s = "".join(w for at, w in TOKENS_FIRST if at <= t)
    if t < NEXT_AT:
        return s
    # the reply arrives a few characters at a time, not one by one, and the rhythm is never quite even
    n = int((t - NEXT_AT) / PACE)
    n -= n % 3 if n < len(NEXT) else 0
    return s + "\n\n" + NEXT[:n]


def state(t):
    """What the screen shows at film time t, as a key: the same key is the same picture."""
    if t < REPLY:
        return ("wait", int(round(t * 30)) % TURNS)
    text = replied(t)
    streaming_now = t < TOKENS_FIRST[-1][0] + .2 or (t >= NEXT_AT and len(text) < len(FIRST) + 2 + len(NEXT))
    return ("say", text, bool(streaming_now or (t * 2 / BLINK) % 2 < 1.15))


def picture(key):
    """The whole screen in the state key, at HIGH times the frame's pixels."""
    size = dict(w=ring.WIN_W, h=ring.WIN_H, top=TOP)
    if key[0] == "wait":
        return ring._frame(page.window(page.you(ASK), title=TITLE, waiting=(key[1] / TURNS, WAIT_Y), **size), HIGH)
    parts = key[1].split("\n\n")
    words = "".join(page.para(p, page.caret() if key[2] and i == len(parts) - 1 else "") for i, p in enumerate(parts))
    return ring._frame(page.window(page.you(ASK) + words, title=TITLE, **size), HIGH)


def at(t):
    return picture(state(t))


def main():
    os.makedirs(OUT, exist_ok=True)
    for what in sys.argv[1:]:
        if what == "glow":
            Image.fromarray(at(100.0)).resize((2560, 1440), Image.LANCZOS).save(os.path.join(OUT, "glow.png"))
        else:
            Image.fromarray(at(float(what))).resize((mac.W, mac.H), Image.LANCZOS).save(os.path.join(OUT, f"{what}.png"))
    print(OUT)


if __name__ == "__main__":
    main()
