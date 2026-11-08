"""那个精，便是我 (89.6–96.1). The camera of the shot before goes on: nearer, and round until it faces the screen, the edges of the page and the dark of the corners going out of the frame past it, and the page's white warming to the white of the window the film comes back to, until nothing is left on it but the post's six lines, larger than they were. From the four sides lines of what was written have begun to come in, the texts as they were checked, each in its own script and its own direction, from clay to code; and the post's lines lift off their baselines, one after another, the same glyphs all the way, and slide down into a stroke. Lines go on growing, slowly, along paths, and gather along the seven strokes of one character, written by the hand of 人 神 鬼 器. Side by side along each stroke go writings of different ages. By 92.0 it is a black character at the distance one watches from, and at six times its size it is still only small writing. Then nothing moves until the cut.

    python film_wo.py char                 # the character alone, as the brush left it
    python film_wo.py still 91.5 92.5 94.7 # frames at these film times
    python film_wo.py keys                 # w1 w2 w3 w3_crop
    python film_wo.py join                 # join_new3.png: 89.6 to 92.8, every 0.2 s
    python film_wo.py low                  # wo_low.mp4 and its contact sheet
    python film_wo.py full                 # wo_low5_1080.mp4: the same frames at 1920 by 1080, for the 1080 pass
    python film_wo.py check                # every writing against its font: not one box

Goes to ink/film_out/film/wo/. Of what comes in from outside nothing flies: a character is put down where it stays, at the moment the head of its line reaches it, and a line's tail lets go behind it, into the white, the way ink would if it were drawn back in.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes, distance_transform_edt, gaussian_filter, gaussian_filter1d, grey_dilation, label, map_coordinates

import film_docs as D
import film_docs_end as E
import film_screens as fs
import film_screens_tex as kt
import ink_mac as mac
from film_big import Hand, SIZE, _band
from film_hand import CLEAR, H, TOP, W
from ink_brush import spline
from ink_grade import sm
from film_wo_texts import lines as lines_of

OUT = os.path.join(D.ROOT, "ink", "film_out", "film", "wo")
T0, T_SET, T1 = 89.6, 94.7, 96.1
GATHER = 90.2       # from here to T_SET the lines gather, evenly, and the last second settles


def gathered(f):
    """The time at which share f of the gathering has gone."""
    return GATHER + f * (T_SET - GATHER)
GROUND = np.array((250, 249, 245), np.float32)       # the main pane of the Claude window in its light theme: ink_mac's LIGHT["main"]
INK = np.array((18, 16, 15), np.float32)
SS = 3                                               # worked out at this many times the frame, for type this small
TALL = 716                                           # how many frame pixels the character stands

# the brush's way, in a square a hundred wide, as film_big's table has it: (x, y, pressure[, how fast]), ink, how fast it runs out. Its structure read off Xingkai's Light master, set up straighter as a hand does for one large character. Heavy at the head, drawn in along the way, pressed again at a turn or a foot, as a stroke is in 顏 or on a Wei stone
WO = dict(body=13.0, flat=0.36, seed=23, marks=[
    # 撇, short and flat, cut in at its head, clear of the long stroke
    ([(45.2, 12.0, .6), (45.4, 15.5, .72), (44.4, 19.6, .74), (41.2, 24.0, .62), (35.8, 28.2, .47), (27.5, 32.4, .33, .12), (19, 35.8, .18, .35), (12.5, 37.8, .02, .6)], 1.14, .6),
    # 横, the long beam, going up, a press at its end
    ([(3.0, 52.0, .62), (6.6, 50.8, .68), (11.5, 49.9, .6), (22, 48, .44), (38, 45.1, .34, .05), (55, 42, .33, .08), (70, 39.4, .37, .05), (79.5, 37.6, .5), (84.6, 37.0, .64), (88.5, 37.6, .68), (90.5, 38.5, .5)], 1.16, .3),
    # 竖钩, cut in, straight down and drawn in, pressed at the corner, turned back up to the left
    ([(36.2, 26.5, .6), (36.6, 30.5, .68), (37, 35.6, .64), (37, 42, .5), (37, 56, .46), (37, 72, .47, .04), (37, 82, .56), (37.3, 88, .74), (35.8, 92.2, .84), (30, 89, .46, .15), (23, 84.2, .17, .35), (18.6, 80.8, .02, .6)], 1.14, .5),
    # 提, up out of a press, across the upright, thinning
    ([(3.6, 73.8, .6), (7.6, 73, .74), (12, 71.6, .72), (19.5, 68.6, .54), (31, 63.4, .42), (42, 57.6, .3, .12), (52, 52, .17, .35), (58.6, 48.4, .02, .6)], 1.12, .6),
    # 斜钩, the one long stroke, the whole of the character: cut in at the top, gathering, drawn in and running dry on the way down, pressed at its foot
    ([(51.8, 0.5, .6), (52.6, 4.6, .8), (53.8, 9, .86), (55, 15.5, .76), (57.2, 30, .64, .02), (60.8, 45, .56, .04), (65.6, 59, .52, .08), (71.6, 71, .5, .14), (78.8, 81, .55, .16), (85.2, 87.6, .74, .06), (89.6, 89.8, 1.0), (92, 88, .84), (92.2, 82, .42, .25), (91.6, 75, .16, .45), (91, 70, .02, .65)], 1.18, .36),
    # 撇, heavy at its shoulder, crossing the long stroke and away down to the left
    ([(77.6, 44.5, .6), (78, 49, .76), (77.4, 53, .76), (74.4, 58.8, .6), (68, 66.4, .46), (58.6, 74, .34, .12), (47, 81, .19, .3), (39.8, 85.4, .02, .6)], 1.14, .55),
    # 点, in on the point, pressed, drawn back
    ([(64.6, 16.6, .06), (68, 18, .3), (72.2, 21, .54), (75.2, 24.2, .7), (76.4, 27, .62), (76.6, 29.4, .44)], 1.14, .8),
])
# where each stroke was cut: the edge a brush leaves going in on the slant, or turning hard, as straight lines (x0, y0, x1, y1) with the ink on the right of the way from the first point to the second. No brush is round at the head unless it means to be
CUTS = (
    ((40.0, 13.0, 48.8, 18.6), (38.6, 17.4, 40.8, 12.6)),
    ((7.0, 55.6, 2.8, 46.2), (2.2, 49.0, 4.4, 45.4), (83.6, 33.2, 90.2, 36.8), (89.8, 36.2, 88.8, 42.4)),
    ((31.0, 27.4, 41.6, 34.2), (29.8, 31.2, 32.2, 26.8), (43.0, 86.4, 39.6, 96.0), (40.6, 96.6, 29.0, 93.0)),
    ((7.8, 78.2, 3.4, 68.8), (2.8, 71.2, 5.0, 67.8), (9.0, 78.6, 2.0, 76.6)),
    ((46.4, 2.6, 58.4, 9.4), (45.0, 7.0, 47.6, 1.8), (96.8, 85.0, 95.0, 93.4), (96.0, 94.0, 85.0, 96.0)),
    ((71.8, 45.4, 82.6, 51.0), (70.6, 49.2, 73.0, 44.4)),
    ((81.2, 25.4, 73.8, 32.2),),
)
CORNER = 0.6        # how far a cut's corners are eased, in hundredths of the square: a brush has no knife's edge


class Cut(Hand):
    """A hand that means its heads and its turns: what a stroke lays down is cut along its edges before it goes into the paper, which then does to those edges what it does to any other."""

    def __init__(self, cuts, *a, **k):
        super().__init__(*a, **k)
        self.cuts, self.done = cuts, 0

    def stroke(self, into, path, index, ink=1.0, dry=1.0, tone=0.97, soak=1.0, upto=1.0):
        d = self.brush.deposit(path, index, upto=upto, ink=ink, dry=dry, step=self.step)
        (y0, y1, x0, x1), D, K = d
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) / self.brush.k
        left = np.ones_like(D)
        for ax, ay, bx, by in self.cuts[self.done]:
            ux, uy = bx - ax, by - ay
            long_ = np.hypot(ux, uy)
            far = (ux * (yy - ay) - uy * (xx - ax)) / long_              # how far to the right of the way, in the square's hundredths
            near = np.hypot(xx - (ax + bx) / 2, yy - (ay + by) / 2) < long_ + 1.5       # an edge is only an edge where it is, or it would take off the far end of its stroke
            left = np.where(near & (far < 0), 0.0, left)
        self.done += 1
        # after the paper has had it, or the streaks of the hairs would end each at its own place along the cut, a row of teeth. The paper's grain pushes this edge about as it does the others
        keep = gaussian_filter(np.float32(left), CORNER * self.brush.k)
        yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
        far = self.rough * 0.42 * self.grain
        keep = map_coordinates(keep, [yy + far * self.warp[0][y0:y1, x0:x1], xx + far * self.warp[1][y0:y1, x0:x1]], order=1, mode="nearest")
        box, a = self.wet(((y0, y1, x0, x1), D, K), tone, soak)
        self.lay(into, (box, a * sm((keep - 0.5) / 0.14 + 0.5)))


SUBTITLE_GOES = 92.8                # what comes in through the middle of the foot of the frame waits for the subtitle there to be gone
# how a writing stands on its way. "down": a column, its head back up the way it came. "up": a column read downward, growing upward. "across": a row read to the right, its foot on the way. "back": a row going to the left. "along": a row whose foot follows the way wherever it goes, as a line on a scroll turned in the hand. "leftward": the same for a writing that reads to the left
ALPHABETS = ("greek", "sonnet", "french", "spanish", "italian", "german", "russian", "latin", "portuguese")          # set letter by letter, each letter its own width
FACES = dict(greek=("Palatino.ttc", 2), sonnet=("Baskerville.ttc", 4), french=("Hoefler Text.ttc", 1), spanish=("Cochin.ttc", 1), italian=("Athelas.ttc", 3), german=("Iowan Old Style.ttc", 1), russian=("Georgia Bold.ttf", 0), latin=("Charter.ttc", 3), portuguese=("Palatino.ttc", 2))
# these only a shaper can set: letters that join, marks that stack, conjuncts, so each word is drawn whole by HarfBuzz. font, face, direction, script, language, how large for its nominal size, how wide a space
SHAPED = dict(arabic=("Muna.ttc", 4, "rtl", "arab", "ar", 1.0, 0.3), persian=("NotoNastaliq.ttc", 2, "rtl", "arab", "fa", 1.0, 0.3), hebrew=("NewPeninimMT.ttc", 3, "rtl", "hebr", "he", 1.0, 0.3), sanskrit=("DevanagariMT.ttc", 1, "ltr", "deva", "sa", 1.0, 0.3), tamil=("Tamil MN.ttc", 1, "ltr", "taml", "ta", 0.92, 0.32))
WORDS = ALPHABETS + ("code",)
ROWS = ALPHABETS + tuple(SHAPED) + ("cuneiform", "hiero")                          # a place on the way is where one of these starts, not its middle
SQUARES = dict(oku=("ヒラギノ明朝 ProN.ttc", 2), basho=("ヒラギノ明朝 ProN.ttc", 2), korean=("AppleMyungjo.ttf", 0))
UPRIGHT = ("letter", "poem", "ruling", "libai", "oracle", "hiero", "oku", "basho", "korean")                         # these stand up in a column, as they always have
RIGHT_TO_LEFT = ("arabic", "persian", "hebrew")
READS_RIGHT = ALPHABETS + ("post", "code", "cuneiform", "sanskrit", "tamil", "oku", "basho", "korean")               # these read to the right, whichever way the head of their line is going
SIZE_COMING = dict(letter=19.0, poem=21.0, ruling=17.0, code=17.0, post=17.0, cuneiform=31.0, hiero=31.0, oracle=34.0, libai=25.0, oku=25.0, basho=27.0, korean=24.0, arabic=30.0, persian=31.0, hebrew=29.0, sanskrit=28.0, tamil=26.0, **dict.fromkeys(ALPHABETS, 25.0))    # frame pixels, out in the white
SIZE_MOST = dict(letter=16.0, poem=15.0, ruling=15.0, code=13.5, post=11.5, cuneiform=20.0, hiero=20.0, oracle=18.0, libai=16.0, oku=16.0, basho=17.0, korean=15.0, arabic=20.0, persian=20.0, hebrew=17.0, sanskrit=19.0, tamil=18.0, **dict.fromkeys(ALPHABETS, 18.0))       # and down on a stroke, where it is wide enough to let them
SIZE_LEAST = 10.0                                                                  # under this the letters of a thin tip only break into specks: that part of it stays plain black
LET_OUT = dict(letter=3, poem=1, arabic=2, persian=2, hebrew=1, sanskrit=3, tamil=3)      # the quick hand's strokes, and the thin parts of these scripts, are thin for this size: let out a little before it is made small
WEIGHT = dict(letter=0.55, poem=0.9, ruling=0.75, code=0.85, post=0.8, cuneiform=0.8, hiero=0.8, oracle=0.9, libai=0.85, oku=0.8, basho=0.8, korean=0.72, arabic=0.72, persian=0.72, hebrew=0.72, sanskrit=0.72, tamil=0.72, **dict.fromkeys(ALPHABETS, 0.7))         # a character so small keeps its weight only by being let out a little
SHOWS = dict(letter=0.11, poem=0.11, ruling=0.1, code=0.1, post=0.09, cuneiform=0.11, hiero=0.085, oracle=0.11, libai=0.105, arabic=0.15, persian=0.15, hebrew=0.14, sanskrit=0.12, tamil=0.12, **dict.fromkeys(ALPHABETS + ("oku", "basho", "korean"), 0.105))            # how much of the paper's light a character lets through the finished stroke
POOL = ("spanish", "arabic", "latin", "korean", "italian", "persian", "greek", "hebrew", "portuguese", "sanskrit", "french", "tamil", "letter", "russian", "ruling", "sonnet", "oku", "german", "libai", "code")     # what the lines under the top sheet are written from, in turn, each going on from where it last stopped


def lane(*run, comes=None, outer=False, into=None, after=None):
    """One line along a stroke: what is written on it, in turn, each as far as that share of the way on the stroke (0: the way in only; less than that: so many frame pixels short of the stroke) and, if it says, starting from those words and going as far as those. comes: the edge of the frame it comes in from, and where. outer: only where the stroke presses wide enough for it. into: a line that only comes in, and goes in at this share of this stroke, heading so: at the flank, where it meets nothing else coming in. after: it does not start until then."""
    return dict(run=run, comes=comes, outer=outer, into=into, after=after)


# the seven strokes, their lanes in order across the way: which way each goes, whether from its end, what takes it on to its thin tip; arrive: how straight a line that comes in is when it reaches the stroke; enter: how far along it does. Every line here is another work, or the same work going on: nothing is said twice. Different ages and hands side by side, so that wherever one looks there are two civilisations at least. To the right of the character 点 sits straight over the head of the second 撇, and the end of 横 runs between them: so 点 is reached from above, the way it is written, and the 撇 from the right under 横, and nothing that comes in crosses another. What reads to the left comes in from the right
LANES = (
    dict(way="across", flip=True, tip="portuguese", lanes=(        # 撇
        lane(("italian", 1.0), comes=("left", 150)),
        lane(("libai", 0.8), ("korean", 1.0), comes=("left", 250)),
        lane(("ruling", 1.0), outer=True), lane(("korean", 1.0), outer=True))),
    dict(way="across", tip="italian", arrive=0.22, lanes=(      # 横
        lane(("latin", 0.62), ("german", 1.0), comes=("left", 340)),
        lane(("code", 0.6), ("greek", 1.0, "Μῆνιν"), comes=("left", 430)),
        lane(("sonnet", 1.0, "Shall I"), comes=("left", 565), into=(1, 0.27, (1.0, -0.6), 0.45)),
        lane(("hebrew", 1.0), comes=("right", 230), into=(1, 0.985, (-1.0, 0.06))),
        lane(("ruling", 1.0), outer=True), lane(("letter", 1.0), outer=True))),
    dict(way="down", tip="letter", lanes=(                   # 竖钩
        lane(("oku", 0.36), ("hebrew", 0.74, "וַיֹּאמֶר"), ("letter", 1.0), comes=("top", 640)),
        lane(("russian", 0.34), ("cuneiform", 0.76), ("korean", 1.0), comes=("top", 520)),   # the hook turns back on itself: what is read along its line stops before it, and what stands up takes it on
        lane(("greek", 1.0, "Τίς τ’"), comes=("top", 760), into=(0, 0.2, (0.3, 1.0))),
        lane(("portuguese", 1.0, "Cessem do sábio"), comes=("bottom", 640), outer=True, after=SUBTITLE_GOES),
        lane(("libai", 1.0), outer=True), lane(("ruling", 1.0), outer=True))),
    dict(way="across", tip="ruling", lanes=(                   # 提
        lane(("post", 1.0, "今天", "其实不疼。")),        # the post comes off the page: these only say where its characters end up
        lane(("post", 1.0, "妈妈", "是为什么。")),
        lane(("post", 1.0, "写在", "谁。")),
        lane(("portuguese", 1.0, "As armas"), comes=("left", 642), into=(3, 0.3, (1.0, -0.3), 0.2)),
        lane(("french", 0.6), comes=("left", 728), outer=True),
        lane(("german", 1.0, "Der du von"), outer=True))),
    dict(way="down", tip="ruling", lanes=(                   # 斜钩
        lane(("sanskrit", 0.34), ("sonnet", 0.7, "So long as"), ("french", 0.9, "Aimer à loisir"), ("ruling", 1.0), comes=("top", 1040)),
        lane(("korean", 0.0), ("tamil", 0.36), ("hiero", 0.66), ("spanish", 0.8, "Frisaba"), ("letter", 1.0), comes=("top", 950)),
        lane(("arabic", 0.5, "فَفَاضَتْ"), ("basho", 0.72), ("portuguese", 0.88, "Dai-me uma fúria"), ("libai", 1.0), comes=("top", 850)),
        lane(("ruling", 1.0), comes=("bottom", 1380), outer=True),
        lane(("italian", 1.0, "Temp’era"), comes=("bottom", 1500), outer=True),
        lane(("french", 1.0, "Vois sur ces"), outer=True), lane(("letter", 1.0), outer=True))),
    dict(way="back", tip="persian", arrive=0.06, enter=0.03, under=3, verse=("ز تابِ جَعدِ", "مشکینش", "چه خون", "افتاد در", "دل‌ها"), lanes=(     # 撇
        lane(("persian", 0.0), ("arabic", 1.0), comes=("right", 650)),
        lane(("persian", 1.0), comes=("right", 505), into=(5, 0.0, (-0.9, 0.4), 0.2)),
        lane(("letter", 1.0), outer=True), lane(("ruling", 1.0), outer=True))),
    dict(way="down", tip="poem", arrive=0.1, under=4, lanes=(         # 点
        lane(("korean", -44.0), ("oracle", 1.0), comes=("top", 1235)),
        lane(("poem", 1.0), comes=("top", 1125)),
        lane(("libai", 1.0), outer=True))),
)


def flow_of(kind, way):
    """How a writing stands on a way going so."""
    if way in ("across", "back") or kind in UPRIGHT:
        return way
    return "leftward" if kind in RIGHT_TO_LEFT else "along"


def reading(kind, flow):
    """1 if a line standing so says its text in the order it grows; -1 if it grows against the way it reads, and is written from its end."""
    return -1 if flow == "up" or (flow == "back" and kind in READS_RIGHT) or (flow == "across" and kind in RIGHT_TO_LEFT) else 1


# the post, as the shot before leaves it: six lines
POST_LINES = fs.TEXTS + (fs.GROWN,)
LINES_AT = fs.COLUMN[0] + fs.TEXT_LEFT + 1, fs.TOOLBAR + fs.CARD_TOP + fs.TEXT_TOP      # the top left of its first line, in the page's pixels
LINE_SIZE, LINES_MIDDLE = 72.0, 470.0                              # how large its characters are when the camera has stopped, frame pixels, and where the middle of the lines is then: clear of the subtitle's strip
CAMERA_FROM = (620.0, 288.0), (500.0, 385.0), 0.25, np.radians(9.0), np.radians(5.0)    # film_screens3d's for this look: what it aims at and what it focuses on, in the page's pixels; how far off; how turned
FOCAL, F_NUMBER = W * 50.0 / 36.0, 8.0                             # its lens, in frame pixels, and its stop
T_FLAT = 91.0                                                      # the camera has stopped, square on, and the page is paper
PAGE_WARMS = 89.8, 90.8                                            # from when to when the panel's grid and the cold of its light go, the page's white warms, and all of it that is not the post goes
OLD_PLATE_GOES = 0.45                                              # how long the frame the shot before left is still what is seen, carried along by the camera, while the page drawn again comes up under it
PLATE_WHITE = 1.0                                                  # the plate's light for the page's white, film_screens' second look
LINES_LEAVE, LINE_PEELS, LINE_FLIES = 0.19, 0.1, 0.62              # one line after another: how long after the last one a line starts, how long its first character is ahead of its last, how long each takes to land
LINE_LIFTS, LINE_SLIDES = 200.0, 260.0                             # the two pulls on its curve: up off its baseline, then back along the stroke, so that it slides in the way the stroke goes. Frame pixels
SHUTTER = 0.3 / 30                                                 # what moves that fast has moved a little in the time a frame takes, not so far it cannot be read
OTHERS_FROM = 90.6                                                 # the others start then, out past the edges


# ---------------------------------------------------------------- the writings

CODE = ("def gcd(a, b):", "while b:", "a, b = b, a % b", "return a", "gcd(1071, 462)", "21", 'printf("hello, world\\n");')
GILGAMESH = "𒊭𒈾𒀝𒁀𒄿𒈬𒊒"                                  # Sha naqba īmuru: SHA NA AK BA I MU RU, the first words of the standard version
BOOK_OF_THE_DEAD = "𓂋𓏤𓈓𓏌𓏤𓉐𓂋𓏏𓂻𓅓𓉔𓂋𓅱𓇳𓏤"                        # rw nw prt m hrw, its title
PTOLEMY = ("Q003", "X001", "V004", "E023", "AA015", "M017", "M017", "S029")     # his name by Gardiner's numbers, which Unicode's names keep
ROYAL = "cartouche"                                                             # stands for the whole of it, in its ring, in the list of what is written


def texts():
    """Each writing, as what it is set from: a string letter by letter, or for what only a shaper can set a list of words. Its lines follow one another; nothing goes round."""
    t = dict(letter=D.LETTER, poem="".join(D.POEM), ruling=D.RULING_TITLE + "".join("".join(p) for p in D.RULING_PARTIES) + "主文" + D.RULING_ORDER + "理由" + D.RULING_REASON + D.RULING_DATE, code="  ".join(CODE), post="".join(POST_LINES), cuneiform=GILGAMESH, hiero=[ROYAL] + list(BOOK_OF_THE_DEAD), oracle="山鳥馬日月水木雨")
    for kind in ALPHABETS:
        t[kind] = unicodedata.normalize("NFC", "  ".join(lines_of(kind)))
    for kind in SHAPED:
        t[kind] = unicodedata.normalize("NFC", " ".join(lines_of(kind))).split()
    t["libai"] = "".join(lines_of("libai"))
    t["oku"] = "".join(lines_of("oku"))
    t["basho"] = "".join(lines_of("basho"))
    t["korean"] = " ".join(lines_of("korean"))
    return t


def post_ink():
    """The post's six lines as the page sets them, as ink, FINE times the panel's pixels. Kept where the shot's files are, since working it out takes a while."""
    path = os.path.join(OUT, "post_ink.npy")
    if os.path.exists(path):
        return np.load(path)
    rows = "".join(f'<div style="height:{fs.TEXT_LH}px">{x}</div>' for x in fs.TEXTS)
    high, cell = fs.TEXT_LH * fs.FINE, fs.TEXT_SIZE * fs.FINE
    wide = max(len(x) for x in POST_LINES)
    a = mac.snap(f'<div style="position:absolute;left:0;top:0;font:{fs.TEXT_SIZE}px/{fs.TEXT_LH}px {fs.FACE};color:#000;white-space:nowrap">{rows}</div>', fs.TEXT_SIZE * wide, fs.TEXT_LH * len(fs.TEXTS), fs.FINE, "#fff")[:high * len(fs.TEXTS), :cell * wide, :3]
    ink = np.zeros((high * len(POST_LINES), cell * wide), np.float32)
    ink[:a.shape[0]] = 1 - a.mean(-1).astype(np.float32) / 255
    last = fs.growth()[1]
    ink[high * (len(POST_LINES) - 1):, :last.shape[1]] = last
    os.makedirs(OUT, exist_ok=True)
    np.save(path, ink)
    return ink


def font_file(name):
    for d in ("/System/Library/Fonts/Supplemental/", "/System/Library/Fonts/"):
        if os.path.exists(d + name):
            return d + name
    raise FileNotFoundError(name)


def check():
    """Every writing that is set from a font, shaped by that font: not one character it lacks, which would be a box."""
    t = texts()
    runs = dict(FACES, **{k: v[:2] for k, v in SHAPED.items()}, **SQUARES, libai=("WeibeiSC-Bold.otf", 0))
    for kind, (name, index) in runs.items():
        path = glob.glob(f"/System/Library/AssetsV2/com_apple_MobileAsset_Font*/*/AssetData/{name}")[0] if name.startswith("Weibei") else font_file(name)
        way, script, language = SHAPED[kind][2:5] if kind in SHAPED else ("ltr", None, None)
        text = " ".join(t[kind]) if isinstance(t[kind], list) else t[kind]
        out = subprocess.run(["hb-shape", f"--font-file={path}", f"--face-index={index}", f"--direction={way}", "--no-positions", "--no-glyph-names", "--no-clusters"] + ([f"--script={script}", f"--language={language}"] if script else []) + [f"--text={text}"], capture_output=True, text=True, check=True).stdout
        missing = sum(1 for g in out.strip("[]\n").split("|") if g.strip() == "0")
        print(kind, name, index, len(text), "missing", missing, flush=True)
        assert missing == 0, kind


TURNS_IN_A_COLUMN = "『』（）()—"
HARD, HARD_AT, HARD_OVER = ("hebrew", "arabic", "persian"), 0.22, 0.4


def carved(c, a):
    """The font draws its signs in outline, too fine to stay at this size. As a relief cuts them: the whole shape is the ink, the lines inside it stay as cuts. What is only a line, the lasso, and the sun, which is a ring, stays one and is made heavier."""
    b = a > 0.4
    fill = binary_fill_holes(binary_closing(b, iterations=2))
    if unicodedata.name(c)[-4:] in ("V004", "N005") or fill.sum() < 1.3 * b.sum():
        return grey_dilation(a, size=7)
    inside = binary_erosion(fill, iterations=9) & b
    return np.clip(gaussian_filter(np.float32(fill), 0.8) - gaussian_filter(np.float32(inside), 0.8), 0, 1)


class Letters:
    """Every character of a writing as a little picture of its ink, 96 pixels a side. The letter's and the poem's are cut from their own sheets, so they are the same characters that lay on the table; the rest are set as their sheets set them."""

    SIDE = 96
    FINE = dict(hiero=2, hebrew=4, arabic=4, persian=4)      # the signs are pictures, and come in large: drawn at twice that. These lie turned on their strokes, and are only as sharp as what they are turned from

    def __init__(self):
        self.tiles = {}
        self.advance = {}
        self.wide, self.tall = {}, {}     # for a writing whose signs differ in width: how far each goes on along a row, and along a column
        self.count = {}
        self._sheet("letter", 228, 219, 198)
        self._sheet("poem", 239, 233, 218)
        song = D.face(*D.SONG_BOLD, self.SIDE * 0.92)
        mono = ImageFont.truetype(os.path.join("/System/Applications/Utilities/Terminal.app/Contents/Resources/Fonts", "SF-Mono-Bold.otf"), int(self.SIDE * 0.92))
        t = texts()
        for kind, font, text in (("ruling", song, t["ruling"]), ("code", mono, t["code"]), ("libai", D.face("WeibeiSC-Bold.otf", 0, self.SIDE * 0.92), t["libai"])):       # 魏碑, where 杜甫 is in a running hand
            self._square(kind, font, text)
        self._post()
        self.advance["code"] = 0.62
        self.advance["letter"] = 1.1      # the quick hand fills its square, and would run into the next
        for kind, (name, index) in SQUARES.items():
            self._square(kind, ImageFont.truetype(font_file(name), int(self.SIDE * 0.92), index=index), t[kind])
        for kind, (name, index) in FACES.items():
            self._row(kind, ImageFont.truetype(font_file(name), int(self.SIDE * 0.92), index=index), t[kind])
        self._row("cuneiform", D.face("NotoSansCuneiform-Regular.ttf", 0, self.SIDE * 0.92), t["cuneiform"])
        self._hieroglyphs(t["hiero"])
        self._oracle(t["oracle"])
        for kind in SHAPED:
            self._shaped(kind, t[kind])

    def _square(self, kind, font, text):
        """Each character centred in a square, the way a page of them is set. Stops sit low, on the foot, and what a column has to turn is turned."""
        for c in set(text):
            im = Image.new("L", (self.SIDE, self.SIDE), 0)
            at, anchor = ((self.SIDE / 2, self.SIDE * 0.82), "ms") if kind == "korean" and c in ".,!?" else ((self.SIDE / 2, self.SIDE / 2), "mm")
            ImageDraw.Draw(im).text(at, c, 255, font=font, anchor=anchor)
            self.tiles[kind, c] = np.asarray(im, np.float32) / 255
            if c in TURNS_IN_A_COLUMN:
                self.tiles[kind, c, "column"] = np.rot90(self.tiles[kind, c], -1)

    def _post(self):
        """The post's characters are the page's, cut from what the page sets: large, for when they are still large, and at the size of all the others for when they have landed."""
        ink = post_ink()
        high, cell = fs.TEXT_LH * fs.FINE, fs.TEXT_SIZE * fs.FINE
        side = int(cell * self.SIDE / (self.SIDE * 0.92))
        self.big = {}
        for r, line in enumerate(POST_LINES):
            for i, c in enumerate(line):
                a = np.zeros((side, side), np.float32)
                top = (high - side) // 2
                a[:, (side - cell) // 2:(side - cell) // 2 + cell] = ink[r * high + top:r * high + top + side, i * cell:(i + 1) * cell]
                self.big[c] = Image.fromarray(np.uint8(a * 255))
                self.tiles["post", c] = np.asarray(self.big[c].resize((self.SIDE, self.SIDE), Image.LANCZOS), np.float32) / 255

    def _shaped(self, kind, words):
        """Each word drawn whole by HarfBuzz, joined and stacked and in its own direction, all on one foot. Kept where the shot's files are, so only a new word is ever drawn."""
        name, index, way, script, language, scale, space = SHAPED[kind]
        fine = self.FINE.get(kind, 1)
        px, margin, pad = int(self.SIDE * 0.92 * scale * fine), self.SIDE * 2 * fine, 8 * fine
        room = os.path.join(OUT, "tiles", kind)
        os.makedirs(room, exist_ok=True)
        path = font_file(name)
        drawn = {}
        for w in set(words):
            png = os.path.join(room, hashlib.md5(f"{name}{index}{px}logical{w}".encode("utf8")).hexdigest() + ".png")
            if not os.path.exists(png):
                subprocess.run(["hb-view", f"--font-file={path}", f"--face-index={index}", f"--font-size={px}", f"--margin={margin}", "--logical", "--foreground=000000", "--background=FFFFFF", "--output-format=png", f"--output-file={png}", f"--direction={way}", f"--script={script}", f"--language={language}", f"--text={w}"], check=True, capture_output=True)
            drawn[w] = 1 - np.asarray(Image.open(png).convert("L"), np.float32) / 255
        rows = np.flatnonzero(np.any([(a > 0.04).any(1) for a in drawn.values()], axis=0))
        top, bottom = rows[0] - 2, rows[-1] + 3
        for w, a in drawn.items():
            run = a.shape[1] - 2 * margin
            self.tiles[kind, w] = grey_dilation(a[top:bottom, margin - pad:a.shape[1] - margin + pad], size=(LET_OUT[kind] - 1) * fine + 1)
            self.wide[kind, w] = (run + space * px) * 1.12 / self.SIDE / fine

    def _row(self, kind, font, text, make=None):
        """Signs of all widths on one foot, set as they are set in a row."""
        boxes = [font.getbbox(c, anchor="ls") for c in set(text) if c.strip()]
        top, bottom = min(b[1] for b in boxes), max(b[3] for b in boxes)
        pad = 8
        high, foot = bottom - top + 2 * pad, pad - top
        for c in set(text):
            run = int(round(font.getlength(c)))
            im = Image.new("L", (run + 2 * pad, high), 0)
            ImageDraw.Draw(im).text((pad, foot), c, 255, font=font, anchor="ls")
            a = np.asarray(im, np.float32) / 255
            self.tiles[kind, c] = make(c, a) if make else a
            self.wide[kind, c] = run * 1.12 / self.SIDE / self.FINE.get(kind, 1)

    def _sign(self, c, font):
        """One hieroglyph alone, cut to its ink."""
        x0, y0, x1, y1 = font.getbbox(c, anchor="ls")
        im = Image.new("L", (x1 - x0 + 24, y1 - y0 + 24), 0)
        ImageDraw.Draw(im).text((12 - x0, 12 - y0), c, 255, font=font, anchor="ls")
        a = carved(c, np.asarray(im, np.float32) / 255)
        ys, xs = np.flatnonzero((a > 0.05).any(1)), np.flatnonzero((a > 0.05).any(0))
        return a[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]

    def _hieroglyphs(self, text):
        fine = self.FINE["hiero"]
        em = self.SIDE * fine
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/NotoSansEgyptianHieroglyphs-Regular.ttf", int(em * 0.92))
        signs = [c for c in text if c != ROYAL]
        # in a row they stand on one foot, where the font puts them; in a column each is as high as it is, one under another
        self._row("hiero", font, "".join(signs), make=carved)
        gap = int(0.1 * em)
        for c in set(signs):
            a = self._sign(c, font)
            if a.shape[1] > em:
                a = self._fit(a, a.shape[0], em)
            tile = np.zeros((a.shape[0] + 2 * gap, em + 8), np.float32)
            x = (tile.shape[1] - a.shape[1]) // 2
            tile[gap:gap + a.shape[0], x:x + a.shape[1]] = a
            self.tiles["hiero", c, "column"] = tile
            self.tall["hiero", c] = (a.shape[0] + 1.2 * gap) * 1.12 / em
        # the name in its ring, tied off at the end it is read to: p over t, w, l, m over the two reeds, s
        q3, x1, v4, e23, aa15, m17, _, s29 = [self._sign(unicodedata.lookup("EGYPTIAN HIEROGLYPH " + n), font) for n in PTOLEMY]
        fit = self._fit
        h = int(0.95 * em)
        reeds = lambda high, wide: self._beside([fit(m17, high, wide)] * 2, int(0.05 * em))
        body = self._beside([self._under([fit(q3, 0.4 * h, 0.5 * h), fit(x1, 0.3 * h, 0.5 * h)], int(0.08 * h)), fit(v4, h, 0.72 * h), fit(e23, 0.64 * h, 1.2 * h), self._under([fit(aa15, 0.2 * h, 0.72 * h), reeds(0.7 * h, 0.26 * h)], int(0.08 * h)), fit(s29, h, 0.4 * h)], int(0.08 * em), h)
        self.tiles["hiero", ROYAL] = self._ring(body, "right")
        self.wide["hiero", ROYAL] = (self.tiles["hiero", ROYAL].shape[1] + 0.1 * em) * 1.12 / em
        w = int(0.8 * em)
        body = self._under([self._beside([fit(q3, 0.42 * w, 0.42 * w), fit(x1, 0.34 * w, 0.46 * w)], int(0.08 * w)), fit(v4, 0.86 * w, 0.72 * w), fit(e23, 0.6 * w, w), fit(aa15, 0.3 * w, w), reeds(0.78 * w, 0.3 * w), fit(s29, 0.9 * w, 0.4 * w)], int(0.09 * em), w)
        self.tiles["hiero", ROYAL, "column"] = self._ring(body, "bottom")
        self.tall["hiero", ROYAL] = (self.tiles["hiero", ROYAL, "column"].shape[0] + 0.1 * em) * 1.12 / em

    @staticmethod
    def _fit(a, high, wide):
        k = min(high / a.shape[0], wide / a.shape[1])
        return np.asarray(Image.fromarray(np.uint8(a * 255)).resize((max(int(a.shape[1] * k), 1), max(int(a.shape[0] * k), 1)), Image.LANCZOS), np.float32) / 255

    @staticmethod
    def _beside(signs, gap, high=None):
        """Signs in a row, each centred on its middle."""
        high = high or max(a.shape[0] for a in signs)
        out = np.zeros((high, sum(a.shape[1] for a in signs) + gap * (len(signs) - 1)), np.float32)
        x = 0
        for a in signs:
            y = (high - a.shape[0]) // 2
            out[y:y + a.shape[0], x:x + a.shape[1]] = a
            x += a.shape[1] + gap
        return out

    @staticmethod
    def _under(signs, gap, wide=None):
        """Signs one under another, each centred across."""
        wide = wide or max(a.shape[1] for a in signs)
        out = np.zeros((sum(a.shape[0] for a in signs) + gap * (len(signs) - 1), wide), np.float32)
        y = 0
        for a in signs:
            x = (wide - a.shape[1]) // 2
            out[y:y + a.shape[0], x:x + a.shape[1]] = a
            y += a.shape[0] + gap
        return out

    def _ring(self, body, tied):
        """A cartouche round what is in it: the rope, and the bar it is tied off at."""
        em = self.SIDE * self.FINE["hiero"]
        pad, line, k = int(0.11 * em), int(0.05 * em), 4
        h, w = body.shape
        out = pad + line
        bar = 2 * line
        H, W = h + 2 * out + (bar if tied == "bottom" else 0), w + 2 * out + (bar if tied == "right" else 0)
        im = Image.new("L", (W * k, H * k), 0)
        d = ImageDraw.Draw(im)
        round_ = dict(right=(0, 0, (W - bar) * k - 1, H * k - 1), bottom=(0, 0, W * k - 1, (H - bar) * k - 1))[tied]
        d.rounded_rectangle(round_, radius=int(min(round_[2], round_[3]) * 0.46), outline=255, width=line * k)
        if tied == "right":
            d.rectangle(((W - 1.4 * line) * k, 0.12 * H * k, W * k, 0.88 * H * k), fill=255)
        else:
            d.rectangle((0.12 * W * k, (H - 1.4 * line) * k, 0.88 * W * k, H * k), fill=255)
        a = np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255
        a[out:out + h, out:out + w] = np.maximum(a[out:out + h, out:out + w], body)
        return a

    def _oracle(self, text):
        """The eight signs of the bones, from the outlines the myth uses: each closed part cuts or fills what is under it. Read when rendering, and kept off the repository."""
        from film_myth_world import GLYPHS, svg_subpaths
        with open(GLYPHS, encoding="utf8") as f:
            data = json.load(f)
        k = 4
        for c in text:
            v = data[c]
            bits = np.zeros((self.SIDE * k, self.SIDE * k), bool)
            for part in svg_subpaths(v if isinstance(v, str) else (v.get("path") or v.get("d"))):
                im = Image.new("1", bits.shape[::-1], 0)
                ImageDraw.Draw(im).polygon([(a / 300 * self.SIDE * k, b / 300 * self.SIDE * k) for a, b in part], fill=1)
                bits ^= np.asarray(im, bool)
            self.tiles["oracle", c] = np.asarray(Image.fromarray(np.uint8(bits) * 255).resize((self.SIDE, self.SIDE), Image.BOX), np.float32) / 255

    def _sheet(self, kind, *paper):
        with open(os.path.join(D.MAPS, kind + ".json"), encoding="utf-8") as f:
            info = json.load(f)
        col = np.asarray(Image.open(os.path.join(D.MAPS, kind + "_col.png")).convert("L"), np.float32) / 255
        px = info["px"]
        ground = 0.299 * paper[0] / 255 + 0.587 * paper[1] / 255 + 0.114 * paper[2] / 255
        for i, c in enumerate(info["chars"]):
            x0, y0, x1, y1 = c["box"]
            cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.56
            a, b, e, d = int((cx - half) * px), int((cy - half) * px), int((cx + half) * px), int((cy + half) * px)
            cut = col[max(b, 0):d, max(a, 0):e]
            ink = np.clip((ground * 0.9 - cut) / (ground * 0.9 - 0.08), 0, 1)
            tile = np.asarray(Image.fromarray(np.float32(ink), "F").resize((self.SIDE, self.SIDE), Image.LANCZOS), np.float32).clip(0, 1)
            self.tiles[kind, i] = grey_dilation(tile, size=LET_OUT[kind]) if LET_OUT[kind] > 1 else tile
        self.count[kind] = len(info["chars"])

    def __call__(self, kind, text, i, flow=None):
        """The i-th character of a writing's text."""
        if kind in ("letter", "poem"):
            return self.tiles[kind, i if kind == "letter" else 2 + i]      # the poem's title stays out of it
        c = text[i]
        if flow in ("down", "up") and (kind, c, "column") in self.tiles:
            return self.tiles[kind, c, "column"]
        return self.tiles[kind, c]

    def size(self, kind, text):
        """How many characters there are to say, once each."""
        return self.count[kind] - (2 if kind == "poem" else 0) if kind in ("letter", "poem") else len(text)

    def goes(self, kind, text, i, flow):
        """How far a line goes on after the i-th character, for a character of size 1."""
        if kind == "hiero" and flow in ("down", "up"):
            return self.tall[kind, text[i]]
        if kind in ROWS:
            return self.wide[kind, text[i]]
        if kind == "korean" and text[i] in " .,!?()":
            return 0.42
        return self.advance.get(kind, 1.0)

    def blank(self, kind, text, i):
        return kind not in ("letter", "poem") and text[i] == " "


# ---------------------------------------------------------------- the character

def written():
    """The character as the brush leaves it, and what each stroke of it covers, in a frame-sized square SS times over: the ink, a list of masks, and the frame pixels of its corner and side."""
    p = np.array([q[:3] for path, *_ in WO["marks"] for q in path])
    half = WO["body"] * 0.3 * p[:, 2]
    x0, y0, x1, y1 = (p[:, 0] - half).min(), (p[:, 1] - half).min(), (p[:, 0] + half).max(), (p[:, 1] + half).max()
    frame = int(TALL / (y1 - y0) * 100)
    x, y = int(W / 2 - (x0 + x1) / 2 * frame / 100), int((TOP * 0.6 + H - CLEAR) / 2 - (y0 + y1) / 2 * frame / 100)
    hand = Cut(CUTS, SIZE, body=WO["body"], flat=WO["flat"], frame=frame * SS // 2)
    states = []
    final = hand.design(WO["marks"], index=WO["seed"], each=lambda out, way: states.append(out.copy()))
    n = frame * SS
    small = lambda a: np.asarray(Image.fromarray(np.float32(np.clip(a, 0, 1)), "F").resize((n, n), Image.LANCZOS), np.float32).clip(0, 1)
    masks, was = [], np.zeros_like(final)
    for s in states:
        masks.append(small(np.abs(s - was) > 0.004))
        was = s
    return small(final), masks, (x, y, frame)


def way_of(path, step=0.25):
    """A stroke's centre line as the brush went, in hundredths of the square, evenly stepped, with its pressure."""
    q = spline([tuple(p) + (0.0,) * (4 - len(p)) for p in path], 0.05)
    far = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(q[:, :2], axis=0).T))])
    at = np.arange(0, far[-1], step)
    return np.stack([np.interp(at, far, q[:, 0]), np.interp(at, far, q[:, 1]), np.interp(at, far, q[:, 2])], 1)


# ---------------------------------------------------------------- the lines

def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


_us = np.linspace(0, 1, 1024)
_gone = np.cumsum(smooth(_us / 0.16) * (1 - 0.55 * smooth((_us - 0.78) / 0.22)))     # slow to begin with, then steady, and settling: never in a hurry
_gone /= _gone[-1]


def when(s):
    """At what share of its time a line has got as far as share s of its way."""
    return np.interp(s, _gone, _us)


_pushes = np.cumsum(smooth(_us / 0.09) * smooth((1 - _us) / 0.18))      # even, but for a moment to get going and to stop
_pushes /= _pushes[-1]


def panel(px, py):
    """A place on the panel, in the page's pixels, in the room: film_screens3d's."""
    x0, y0, x1, y1 = fs.CROP
    return np.array(((px - (x0 + x1) / 2) * kt.PIXEL, 0.0, -(py - (y0 + y1) / 2) * kt.PIXEL))


def camera(t):
    """The camera at time t, going on from where the shot before left it to square on to the post, its size growing by the same factor every moment: what takes the page's pixels to the frame's, where it is, its axes, and how far off it is focused."""
    u = float(np.interp((t - T0) / (T_FLAT - T0), _us, _pushes))
    aim0, focus0, far0, yaw0, pitch0 = CAMERA_FROM
    x, y = LINES_AT
    middle = np.array((x + fs.TEXT_SIZE * max(len(a) for a in POST_LINES) / 2, y + fs.TEXT_LH * len(POST_LINES) / 2))
    zoom = LINE_SIZE / fs.TEXT_SIZE
    aim1 = middle - np.array((0.0, (LINES_MIDDLE - H / 2) / zoom))
    aim = np.array(aim0) + (aim1 - aim0) * u
    far = far0 * (FOCAL * kt.PIXEL / zoom / far0) ** u
    yaw, pitch = yaw0 * (1 - u), pitch0 * (1 - u)
    target = panel(*aim)
    at = target + np.array((-np.sin(yaw) * np.cos(pitch), -np.cos(yaw) * np.cos(pitch), -np.sin(pitch))) * far
    front = (target - at) / far
    right = np.cross(front, (0.0, 0.0, 1.0))
    right /= np.hypot(*right[[0, 1]])
    up = np.cross(right, front)
    x0, y0, x1, y1 = fs.CROP
    flat = np.array(((kt.PIXEL, 0, -(x0 + x1) / 2 * kt.PIXEL), (0, 0, 0), (0, -kt.PIXEL, (y0 + y1) / 2 * kt.PIXEL)))
    flat[:, 2] -= at
    lens = np.array(((FOCAL, 0, W / 2), (0, -FOCAL, H / 2), (0, 0, 1)))
    focus = np.array(focus0) + (middle - focus0) * u
    return lens @ np.stack((right, up, front)) @ flat, at, (right, up, front), float((panel(*focus) - at) @ front)


def warped(im, into, size, how=Image.BILINEAR):
    """A picture as it comes out with this matrix from its pixels to those of a picture of this size."""
    q = np.linalg.inv(into)
    q = (q / q[2, 2]).ravel()[:8]
    return np.asarray(im.transform(size, Image.PERSPECTIVE, tuple(q), how), np.float32)


class Glyphs:
    """Every character the film's last seconds here ever put down, each with its picture already turned and sized. A frame is only which of them are in, and how far."""

    def __init__(self):
        self.letters = Letters()
        self.final, self.masks, self.place = written()
        self.n = self.final.shape[0]
        # the ink on the frame, and how far inside it each place is
        ww, hh = W * SS, H * SS
        x, y, _ = self.place
        X, Y = x * SS, y * SS
        self.ink_map = np.zeros((hh, ww), np.float32)
        Y0, Y1, X0, X1 = max(Y, 0), min(Y + self.n, hh), max(X, 0), min(X + self.n, ww)
        self.ink_map[Y0:Y1, X0:X1] = self.final[Y0 - Y:Y1 - Y, X0 - X:X1 - X]
        # thin ink is still ink: only a real gap the hairs left, a pixel of the frame wide or more, counts as one
        thin = gaussian_filter(np.float32(self.ink_map > 0.1), 0.7 * SS)
        self.solid = np.maximum(sm((self.ink_map - INKED) / 0.2), sm((thin - 0.72) / 0.12))
        # and only one wider than a little, and open to the paper: the white slits along a thin tip are specks
        hard = binary_closing(self.solid > 0.5, iterations=int(1.4 * SS))
        holes, count = label(~hard)
        small = np.bincount(holes.ravel()) < SLIT_MOST * SS * SS
        small[0] = False
        self.solid = np.where(hard | small[holes], np.maximum(self.solid, hard | small[holes]), self.solid).astype(np.float32)
        self.depth = np.float32(distance_transform_edt(self.solid > 0.5))
        self.loose = sm((gaussian_filter(self.ink_map, 2 * SS) - 0.1) / 0.1)
        self.inner = sm((self.depth - 0.6 * EDGE * SS) / (0.4 * EDGE * SS))
        self.page()
        self.items = []
        self.text = texts()
        self.used = {kind: np.zeros(self.letters.size(kind, text), bool) for kind, text in self.text.items()}       # what has been said: nothing is said twice
        self.r = np.random.default_rng(1926)
        self.turn = 0
        self.lay()

    def page(self):
        """The page as the shot before leaves it, the very frame, in linear light; and the page again for the camera to go on over, drawn many times finer: what is on it, and the words apart, since they stay when it goes."""
        path = os.path.join(OUT, "s5_end.png")
        if not os.path.exists(path):
            t = fs.frames("s5_b")[-1]
            os.makedirs(OUT, exist_ok=True)
            Image.fromarray(fs.still(fs.showing(t)[0], t)).save(path)
        self.last = np.asarray(Image.open(path).convert("RGB"))
        a = self.last.astype(np.float32) / 255
        self.s5 = [Image.fromarray(c, "F") for c in np.moveaxis(np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32), 2, 0)]
        words = f"top:{fs.TEXT_TOP}px;font-size:{fs.TEXT_SIZE}px;line-height:{fs.TEXT_LH}px;color:"
        html = fs.s5_sent_html(False)
        assert html.count(words + "#333") == 1
        fine = PAGE_FINE
        whole = mac.snap(html.replace(words + "#333", words + "transparent"), fs.PW, fs.PH, fine, "#000")[..., :3]
        back = np.linalg.inv(camera(T0)[0])
        corners = back @ np.array(((0, 0, 1), (W, 0, 1), (0, H, 1), (W, H, 1)), np.float64).T
        x0, y0 = np.floor((corners[:2] / corners[2]).min(1)).astype(int) - 4
        x1, y1 = np.ceil((corners[:2] / corners[2]).max(1)).astype(int) + 4
        self.page_at = x0, y0
        self.page_big = [Image.fromarray(c, "F") for c in np.moveaxis(np.float32(fs.to_linear(whole[y0 * fine:y1 * fine, x0 * fine:x1 * fine])), 2, 0)]
        self.page_words = Image.fromarray(np.float32(post_ink()), "F").reduce(WORDS_FINE_FROM)
        n = fs.LCD
        m = np.zeros((n, n, 3), np.float32)
        for c, (x_a, x_b) in enumerate(((0, 2), (3, 5), (6, 8))):
            m[:7, x_a:x_b, c] = 1
        m[:7, 2, :2] = 0.18
        m[:7, 5, 1:] = 0.18
        m = 0.42 * m + 0.58 * m.mean((0, 1), keepdims=True)
        self.page_grid = m / m.mean((0, 1)).mean()         # as film_screens' lcd has it
        y, x = np.mgrid[0:2 * H, 0:2 * W].astype(np.float32)
        self.page_xy = (x + 0.5) / 2, (y + 0.5) / 2
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        self.frame_xy = x + 0.5, y + 0.5

    def page_now(self, t):
        """The page at time t through the camera, in linear light. It is the same sum the shot before's frame was made by, until the sum's own parts go: the panel's grid, the camera's cold, the fall of its light to the corners, which is on the page and goes out of the frame with it. What the lens did not have in focus it still does not, and that goes the same way; the words stay black all through."""
        cam, at, (right, up, front), focused = camera(t)
        warm = float(smooth((t - PAGE_WARMS[0]) / (PAGE_WARMS[1] - PAGE_WARMS[0])))
        x0, y0 = self.page_at
        twice = np.diag((2.0, 2.0, 1.0)) @ cam
        size = (2 * W, 2 * H)
        fine = PAGE_FINE
        page = np.stack([warped(c, twice @ np.array(((1 / fine, 0, x0), (0, 1 / fine, y0), (0, 0, 1))), size) for c in self.page_big], 2)
        z = fs.FINE / WORDS_FINE_FROM
        x, y = LINES_AT
        words = warped(self.page_words, twice @ np.array(((1 / z, 0, x), (0, 1 / z, y), (0, 0, 1))), size).clip(0, 1)[..., None]
        ground, ink_ = D.srgb(*GROUND), D.srgb(*INK)
        black = (1 - warm) * D.srgb(0x33, 0x33, 0x33) + warm * ink_ / ground
        page = (1 + (page - 1) * (1 - warm)) * (1 - words) + black * words
        back = np.linalg.inv(cam)
        if warm < 1:
            xx, yy = self.page_xy
            w = back[2, 0] * xx + back[2, 1] * yy + back[2, 2]
            px, py = (back[0, 0] * xx + back[0, 1] * yy + back[0, 2]) / w, (back[1, 0] * xx + back[1, 1] * yy + back[1, 2]) / w
            # the grid as it comes out at this size, which is to say softened by as much as one of these pixels covers of it, or it beats against them
            n = fs.LCD
            cover = 0.7 * n / np.hypot(*(cam[:2, 0] / cam[2, 2]))
            soft = gaussian_filter(self.page_grid, (cover, cover, 0), mode="wrap")
            gx, gy = (px % 1) * n, (py % 1) * n
            ax, ay = gx.astype(np.int32), gy.astype(np.int32)
            fx, fy = (gx - ax)[..., None], (gy - ay)[..., None]
            bx, by = (ax + 1) % n, (ay + 1) % n
            grid = (soft[ay, ax] * (1 - fx) + soft[ay, bx] * fx) * (1 - fy) + (soft[by, ax] * (1 - fx) + soft[by, bx] * fx) * fy
            page = page * (1 + (grid - 1) * (1 - warm))
        page = (page[0::2, 0::2] + page[1::2, 0::2] + page[0::2, 1::2] + page[1::2, 1::2]) / 4
        # how far out of focus each place is, as a lens has it, for a plane that may not be square on to it
        xx, yy = self.frame_xy
        rays = [right[i] * (xx - W / 2) / FOCAL + up[i] * (H / 2 - yy) / FOCAL + front[i] for i in range(3)]
        depth = -at[1] / rays[1]
        lens = 0.05
        sigma = lens * lens / (F_NUMBER * (focused - lens)) * np.abs(depth - focused) / depth * (W / 0.036) / 4
        out, was = page, 0.0
        most = float(sigma.max())
        for step in (1.0, 2.0, 4.0, 8.0):
            if most <= was:
                break
            more = gaussian_filter(page, (step, step, 0))
            k = np.clip((sigma - was) / (step - was), 0, 1)[..., None]
            out = out + (more - out) * k
            was = step
        page = out
        ground_gain = ground * np.ones(3, np.float32)
        camera_gain = PLATE_WHITE * 0.4335 * np.array((0.97, 0.985, 1.03), np.float32)
        gain = (1 - warm) * camera_gain + warm * ground_gain
        was = camera(T0)[0] @ back         # each place, where it was in the frame the shot before left
        w = was[2, 0] * xx + was[2, 1] * yy + was[2, 2]
        fall = 1 - 2.7 * (1 - warm) * (((was[0, 0] * xx + was[0, 1] * yy + was[0, 2]) / w - W / 2) ** 2 + ((was[1, 0] * xx + was[1, 1] * yy + was[1, 2]) / w - H / 2) ** 2) / W ** 2
        light = page * gain
        if warm < 1:
            small_light = np.stack([np.asarray(Image.fromarray(np.float32(page[..., c]), "F").reduce(8), np.float32) for c in range(3)], 2)
            glow = gaussian_filter(small_light, (3.5, 3.5, 0)) * 0.05 + gaussian_filter(small_light, (13.75, 13.75, 0)) * 0.05
            light = light + np.stack([np.asarray(Image.fromarray(glow[..., c], "F").resize((W, H), Image.BILINEAR), np.float32) for c in range(3)], 2) * camera_gain * (1 - warm)
        light = light * fall[..., None]
        kept = 1 - float(smooth((t - T0) / OLD_PLATE_GOES))
        if kept > 0:
            light = light * (1 - kept) + np.stack([warped(c, np.linalg.inv(was), (W, H), Image.BICUBIC) for c in self.s5], 2) * kept
        return light, (1 - warm) * (1 - kept)

    def to_frame(self, p):
        """Hundredths of the square to pixels of the worked-out frame."""
        x, y, frame = self.place
        return (np.asarray(p) / 100 * frame + (x, y)) * SS

    def lay(self):
        # each stroke's line, and how wide it is at each point, as the ink says. Where it crosses one written before it the ink says nothing, so that stretch is carried across
        x, y, _ = self.place
        self.ways, before = [], np.zeros_like(self.final) > 1
        for k, (path, *_) in enumerate(WO["marks"]):
            p = self.to_frame(way_of(path)[:, :2])
            pick = lambda a: map_coordinates(np.float32(a), [p[:, 1] - y * SS, p[:, 0] - x * SS], order=1, mode="nearest")
            half = pick(distance_transform_edt(self.masks[k] > 0.5))
            hidden = gaussian_filter1d(np.float32(pick(before) > 0.5), 8, mode="nearest") > 0.02
            at = np.arange(len(p))
            if hidden.any() and not hidden.all():
                half[hidden] = np.interp(at[hidden], at[~hidden], half[~hidden])
            self.ways.append((p, gaussian_filter1d(np.maximum(half, 1.0), 6, mode="nearest") + 0.15 * SS))
            before = before | (self.masks[k] > 0.5)
        # each stroke by turns, first the lines that run its length, then those going in at its flank, those only where it presses wide, and last what is under them all: so that what a line is said to begin with is still there to say
        turns = [self.stroke(k, *self.ways[k], way_of(WO["marks"][k][0])[:, 2]) for k in range(len(self.ways))]
        for _ in range(4):
            for s in turns:
                next(s, None)
        seen = set()
        for it in self.items:
            assert (it[0], it[5]) not in seen, (it[0], it[5])
            seen.add((it[0], it[5]))
        self.none_alone()
        self.post_leaves()
        for kind in sorted(self.used, key=lambda c: c):
            on = sum(1 for it in self.items if it[0] == kind and it[6] == 1.0 and it[9])
            print("%-11s said %4d of %4d, %4d of them on top of a stroke" % (kind, sum(1 for it in self.items if it[0] == kind), len(self.used[kind]), on), flush=True)

    def none_alone(self):
        """A letter or two where a line has only touched a stroke, on the corner of another or at the very end of its tip, is not writing but specks: on top of a stroke a line is a run of its characters or it is not there."""
        keep, run = [], []
        def close():
            least = 5 if run[0][0] in WORDS else 3
            if len(run) >= least:
                keep.extend(run)
        for it in self.items:
            alone = it[9] and it[6] == 1.0 and it[0] in WORDS + ("oku", "basho", "korean", "libai", "letter", "poem", "ruling")
            if not alone:
                if run:
                    close()
                    run = []
                keep.append(it)
                continue
            if run and (it[0] != run[-1][0] or abs(it[5] - run[-1][5]) > 2 or np.hypot(*(it[2] - run[-1][2])) > 2.6 * max(it[4], run[-1][4])):
                close()
                run = []
            run.append(it)
        if run:
            close()
        print("%d alone, gone" % (len(self.items) - len(keep)), flush=True)
        self.items = keep

    def post_leaves(self):
        """Each of the post's characters, where it stands on the paper when the camera stops and the place the lanes found for it, and when it goes from the one to the other: one line after another, from the top, each from its first character on, and it lands there, itself."""
        cam = camera(T_FLAT)[0]
        places = {it[5]: j for j, it in enumerate(self.items) if it[0] == "post"}
        text = self.text["post"]
        missing = [text[n] for n in range(len(text)) if n not in places]
        assert not missing, "no place for " + "".join(missing)
        self.flying, n = [], 0
        x, y = LINES_AT
        for row, line in enumerate(POST_LINES):
            tan = self.items[places[n]][3]
            for i, c in enumerate(line):
                j = places[n]
                kind, flow, here, tan_i, s, _, ink, _, t_off, on = self.items[j]
                a = cam @ np.array((x + fs.TEXT_SIZE * (i + 0.5), y + fs.TEXT_LH * (row + 0.5), 1.0))
                leaves = T_FLAT + row * LINES_LEAVE + LINE_PEELS * i / (len(max(POST_LINES, key=len)) - 1)
                lands = leaves + LINE_FLIES
                turn = float(np.clip(-np.arctan2(tan_i[1], tan_i[0]) * 180 / np.pi, -MOST_TURN, MOST_TURN))
                self.items[j] = (kind, flow, here, tan_i, s, n, ink, lands, t_off, on)
                self.flying.append((a[:2] / a[2] * SS, here, np.array((0.0, -LINE_LIFTS * SS)), tan * LINE_SLIDES * SS, c, s, turn, leaves, lands))
                n += 1
        print("the post lands by %.2f" % max(f[-1] for f in self.flying), flush=True)

    def free_from(self, kind, least=1):
        """The first place in a writing with at least so many characters not yet said after it."""
        used = self.used[kind]
        free = np.concatenate([[False], ~used, [False]])
        starts = np.flatnonzero(free[1:] & ~free[:-1])
        ends = np.flatnonzero(~free[1:] & free[:-1])
        for a, b in zip(starts, ends):
            if b - a >= least:
                return int(a)
        return None

    @staticmethod
    def word_at(text, says):
        """Where in a list of words these words stand, or, if it is only one, a word starting so."""
        words = unicodedata.normalize("NFC", says).split()
        for i in range(len(text)):
            if text[i:i + len(words)] == words:
                return i
        return next(i for i, w in enumerate(text) if w.startswith(words[0]))

    def take(self, kind, flow, says, later, at, stop, d, size):
        """Where a line starts in its writing and the last place it may go: from what it is said to start with, or else the first thing not yet said, and never over anything already said. A line growing against the way it reads needs to know how much it will say, and starts from the end of that."""
        text = self.text[kind]
        if says is None:
            a = self.free_from(kind)
        elif isinstance(text, list):
            a = self.word_at(text, says)
        else:
            a = text.index(unicodedata.normalize("NFC", says))
        if a is None or self.used[kind][a]:
            return None
        used = self.used[kind]
        end = a
        while end < len(used) and not used[end]:
            end += 1
        if reading(kind, flow) == 1:
            return a, end - 1
        k, q = a, at
        while k < end and q < stop:
            q += float(size[min(int(np.searchsorted(d, q)), len(size) - 1)]) * self.letters.goes(kind, text, k, flow) * 1.04
            k += 1
        while k < end and not self.letters.blank(kind, text, k) and kind in ALPHABETS:
            k += 1      # to the end of the word it stopped in
        return k - 1, a

    def stroke(self, k, p, half, press):
        """All the lines of stroke k, a turn at a time. The top sheet is its lanes, side by side, each there only where the stroke is wide enough for it. Some come in from the edge of the frame; the rest of what lies on the stroke, under them, grows along it only."""
        spec = LANES[k]
        way, flip = spec["way"], spec.get("flip", False)
        lanes = spec["lanes"]
        main = [l for l in lanes if not l["outer"] and not l["into"]]
        outer = [l for l in lanes if l["outer"]]
        band = lambda l: max(SIZE_MOST[r[0]] for r in l["run"]) * SS * LEADING
        wide = 2 * float(np.median(half))
        full = sum(band(l) for l in main)
        fit = float(np.clip(wide / full, 0.72, 1.15))
        placed = []
        arrive = spec.get("arrive", 0.4)

        def here():
            self.k, self.arrive, self.placed, self.fit, self.mid, self.reach = k, arrive, placed, fit, wide / 2, full * fit / 2

        here()
        if "verse" in spec:
            self.verse("persian", spec["verse"], p, half)
        # across the stroke, from one side to the other, and then out from there on both sides for as far as it ever presses
        where, x = [], -full * fit / 2
        for l in main:
            where.append((x + band(l) * fit / 2, band(l) * fit, l))
            x += band(l) * fit
        low, high, j = -full * fit / 2, full * fit / 2, 0
        beyond = []
        while outer and (high < half.max() or low > -half.max()):
            l = outer[j % len(outer)]
            w = band(l) * fit
            if (j % 2 == 0 and high < half.max()) or low <= -half.max():
                beyond.append((high + w / 2, w, l))
                high += w
            else:
                beyond.append((low - w / 2, w, l))
                low -= w
            j += 1
        where.sort(key=lambda v: abs(v[0]))
        beyond.sort(key=lambda v: abs(v[0]))
        print("stroke", k, "wide %.0f, fits at %.2f," % (wide / SS, fit), len(main), "lanes and", len(beyond), "outside", flush=True)
        t_in = gathered(self.r.uniform(0.2, 0.3))             # when those of this stroke's lines that come from nowhere begin: on the paper, not while it is still a page
        foot = len(p) // 2 + int(np.argmax(press[len(p) // 2:]))
        up = (p[foot::-1], half[foot::-1])
        if flip:
            p, half = p[::-1], half[::-1]
        # those that do not come from outside start and end past the brush's ink, so that all of it has writing in it
        past = int(20 * SS / np.hypot(*(p[1] - p[0])))
        enter = int(spec.get("enter", 0.0) * len(p))
        ends = lambda a, b: np.concatenate([a[0] - (a[1] - a[0]) * np.arange(past, 0, -1)[:, None], a, a[-1] + (a[-1] - a[-2]) * np.arange(1, past + 1)[:, None]]) if b else np.concatenate([np.full(past, a[0]), a, np.full(past, a[-1])])
        P, HALF = ends(p, True), ends(half, False)
        end = gathered(0.955)
        rows = len(main)

        def run(l, how, q, h, off, pitch, begin, approach=None):
            """One lane along its way, each of its writings carrying on from where the one before stopped, in place and in time. One that has said all it has hands over at once, and if the last of them has, the line goes on in what there is most left of."""
            at = None
            for kind, until, *says in l["run"]:
                at = self.along(kind, flow_of(kind, how), q, h, None, rows, 1.0, says[0] if says else None, begin, end, approach, True, off, pitch, at=at, until=until, upto=says[1] if len(says) > 1 else None)
            tried = set()
            while at < self.reached - 8 * SS and len(tried) < len(POOL):
                kind = max((c for c in POOL if c not in tried), key=lambda c: (~self.used[c]).sum())
                tried.add(kind)
                got = self.along(kind, flow_of(kind, how), q, h, None, rows, 1.0, None, begin, end, approach, True, off, pitch, at=at)
                print("  stroke", k, "line ran out at %.0f of %.0f, goes on in" % (at / SS, self.reached / SS), kind, flush=True)
                at = got

        def go(off, pitch, l):
            begin = t_in + self.r.uniform(0, 0.36)
            comer = OTHERS_FROM + self.r.uniform(0, 0.4)
            if l["after"]:
                comer = l["after"]
            if l["comes"] and l["comes"][0] == "bottom":
                # this one comes up the stroke from its foot, from below the frame; going up, the same side of it is the other side of the way
                run(l, "up", up[0], up[1], -off, pitch, comer, self.approach(up[0], -off, "bottom", l["comes"][1]))
            elif l["comes"]:
                run(l, way, p[enter:], half[enter:], off, pitch, comer, self.approach(p[enter:], off, *l["comes"]))
            else:
                run(l, way, P, HALF, off, pitch, begin)

        for off, pitch, l in where:
            go(off, pitch, l)
        yield
        here()
        # those coming to a stroke with no room left in it go in at its flank, and that is the last of them
        for l in (l for l in lanes if l["into"]):
            kind, _, *says = l["run"][0]
            other, share, heading, *bend = l["into"]
            how = "down" if abs(heading[1]) > abs(heading[0]) else "across" if heading[0] > 0 else "back"
            w = self.ways[other][0]
            side = w[int(share * (len(w) - 1))] + np.array(heading) * np.arange(13)[:, None]
            self.along(kind, flow_of(kind, how), p, half, None, rows, 1.0, says[0] if says else None, OTHERS_FROM + self.r.uniform(0, 0.4), gathered(0.7), self.approach(side, 0.0, *l["comes"], arrive=bend[0] if bend else 0.25), True, joins_only=True)
        yield
        here()
        for off, pitch, l in beyond:
            go(off, pitch, l)
        tip = spec["tip"]
        if rows % 2 == 0:
            # an even number leave the very middle empty: where they have run out, one more takes the stroke on to its tip
            self.along(tip, flow_of(tip, way), P, HALF, None, rows, 1.0, None, t_in, end, None, True, 0.0, SIZE_MOST[tip] * SS * LEADING * fit * 1.2, tail=True)
        yield
        # under the top sheet more of the writings, thin, until they close up to the stroke's black. By turns from either end, and wider at the ends than the stroke soon gets, so that its head, its tip and the corners the brush cut have them as soon as its body does. What runs out hands over to the next, and nothing goes round
        wide_at = gaussian_filter1d(grey_dilation(HALF, size=41), 10, mode="nearest") * UNDER_WIDE
        back_way = dict(across="back", back="across", down="up")[way]
        under_rows = spec.get("under", rows + 1)     # one more than the top sheet has, to reach the corners the brush cut
        for layer in range(1, LAYERS):
            phase = (layer % 2) * 0.5 + self.r.uniform(-0.12, 0.12)
            for i in range(under_rows):
                f = ((i + 0.5 + phase) / under_rows) * 2 - 1
                if abs(f) > 1:
                    f = f - 2 * np.sign(f)
                begin = gathered(0.35 + 0.025 * layer + self.r.uniform(0, 0.08))
                stop = gathered(0.86 - 0.012 * (LAYERS - layer) * self.r.uniform(0, 1))
                turned = layer % 2 == 0
                how = back_way if turned else way
                at, stuck = None, 0
                while stuck < len(POOL):
                    kind = POOL[self.turn % len(POOL)]
                    self.turn += 1
                    was = at
                    at = self.along(kind, flow_of(kind, how), P[::-1] if turned else P, wide_at[::-1] if turned else wide_at, f, under_rows, GROUND_INK, None, begin, stop, None, i == under_rows // 2, at=at, least=8)
                    if at >= self.reached - 3 * SS:
                        break
                    stuck = stuck + 1 if at == was else 0

    def verse(self, kind, rows, p, half):
        """A whole line of a poem across the stroke's shoulder, where it is widest and no line along it has the room: row under row, each across the stroke and rising to the left, the way a line of it goes on an album page, in the middle of what the ink leaves it between its edges and the stroke going over it. It is what is read here, so it is set first."""
        s, pitch = VERSE_SIZE * SS, VERSE_PITCH * SS
        text = self.text[kind]
        wide = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
        down = p[int(np.searchsorted(wide, VERSE_FROM * SS + 3 * pitch))] - p[int(np.searchsorted(wide, VERSE_FROM * SS))]
        down /= np.hypot(*down)
        across = np.array((-down[1], down[0]))
        over = [self.ways[m] for m in OVER[:OVER.index(self.k)]]
        told = 0
        for row, words in enumerate(rows):
            n = self.word_at(text, words)
            ns = range(n, n + len(words.split()))
            mid = p[int(np.searchsorted(wide, VERSE_FROM * SS))] + down * pitch * row
            reach = np.arange(-70, 70, 0.5)[:, None] * SS * across + mid
            inside = np.array([self.depth[int(q[1]), int(q[0])] > EDGE * SS and all((np.hypot(*(w - q).T) - h).min() > 0.55 * s for w, h in over) for q in reach])
            a, b = np.flatnonzero(inside)[[0, -1]]
            mid = (reach[a] + reach[b]) / 2
            goes = [self.letters.goes(kind, text, m, "back") for m in ns]
            tile = [self.letters(kind, text, m, "back") for m in ns]
            f = 1.12 * s / Letters.SIDE / Letters.FINE[kind]
            # a word's tile is as tall as the tallest word: it is its ink that is to go in the middle
            sag = sum(((t > 0.3).sum(1) * np.arange(t.shape[0])).sum() - (t > 0.3).sum() * t.shape[0] / 2 for t in tile) / sum((t > 0.3).sum() for t in tile) * f
            at = mid - across * sum(goes) * s / 2 + down * -sag
            for m, adv in zip(ns, goes):
                here = at + across * 0.5 * adv * s
                self.items.append((kind, "back", here, across.copy(), s, m, 1.0, gathered(VERSE_WHEN) + 0.08 * told, None, True))
                self.placed.append((here, s, adv))
                self.used[kind][m] = True
                at = at + across * adv * s
                told += 1
            print("  verse row", row, "room %.0f, says %.0f" % (np.hypot(*(reach[b] - reach[a])) / SS, sum(goes) * s / SS), flush=True)

    def approach(self, p, off, side, at, arrive=None):
        """The way in, from a place on the edge of the frame to where this line joins the stroke, leaving it in the direction the stroke will carry on."""
        tan = p[12] - p[0]
        tan = tan / max(np.hypot(*tan), 1e-6)
        join = p[0] + np.array((-tan[1], tan[0])) * off
        ww, hh = W * SS, H * SS
        margin = 24 * SS
        e, out = dict(top=(np.array((at * SS, -margin)), np.array((0, 1.0))), bottom=(np.array((at * SS, hh + margin)), np.array((0, -1.0))), left=(np.array((-margin, at * SS)), np.array((1.0, 0))), right=(np.array((ww + margin, at * SS)), np.array((-1.0, 0))))[side]
        d = np.hypot(*(join - e))
        c1, c2 = e + out * d * 0.4, join - tan * d * (arrive or self.arrive)
        t = np.linspace(0, 1, int(d / (0.75 * SS)))[:, None]
        return (1 - t) ** 3 * e + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t * t * c2 + t ** 3 * join

    def picture(self, kind, flow, tan, s, n):
        """A character's picture, turned and sized, and how many pixels on a side."""
        a = self.letters(kind, self.text[kind], n, flow)
        if kind == "hiero" and flow == "back":
            a = a[:, ::-1]       # going to the left the signs face the way they are read from, as they did on many a wall
        angle = dict(down=np.arctan2(tan[0], tan[1]), up=np.arctan2(-tan[0], -tan[1]), across=-np.arctan2(tan[1], tan[0]), back=-np.arctan2(-tan[1], -tan[0]), along=-np.arctan2(tan[1], tan[0]), leftward=-np.arctan2(-tan[1], -tan[0]))[flow] * 180 / np.pi
        whole = flow in ("along", "leftward") or (kind == "hiero" and self.text[kind][n] == ROYAL) or (kind in RIGHT_TO_LEFT and flow == "back")     # a ring lies along its line, however that goes, and so does what is written that way
        turn = float(angle if whole else np.clip(angle, -MOST_TURN, MOST_TURN))
        if a.shape == (Letters.SIDE, Letters.SIDE):
            size = max(int(s * 1.12) | 1, 5)     # a picture's character fills nine tenths of it, so that makes a character of size s
            im = Image.fromarray(np.uint8(a * 255)).resize((size, size), Image.LANCZOS).rotate(turn, Image.BICUBIC)
            return (np.asarray(im, np.float32) / 255) ** WEIGHT[kind], size
        f = 1.12 * s / Letters.SIDE / Letters.FINE.get(kind, 1)
        w, h = max(int(round(a.shape[1] * f)), 3), max(int(round(a.shape[0] * f)), 3)
        size = int(max(w, h) * 1.15) | 1
        if kind in HARD:
            # turned at three times the size it will be, and from that to the size it will be, the blur of the turning cut off by an edge as hard as a pen's
            big = 3
            bw, bh, bs = w * big, h * big, size * big
            im = Image.new("L", (bs, bs), 0)
            im.paste(Image.fromarray(np.uint8(a * 255)).resize((bw, bh), Image.LANCZOS), ((bs - bw) // 2, (bs - bh) // 2))
            im = im.rotate(turn, Image.BICUBIC).resize((size, size), Image.BOX)
            return np.float32(sm((np.asarray(im, np.float32) / 255 - HARD_AT) / HARD_OVER)), size
        im = Image.new("L", (size, size), 0)
        im.paste(Image.fromarray(np.uint8(a * 255)).resize((w, h), Image.LANCZOS), ((size - w) // 2, (size - h) // 2))
        return (np.asarray(im.rotate(turn, Image.BICUBIC), np.float32) / 255) ** WEIGHT[kind], size

    def fits(self, kind, flow, at, tan, s, n, adv):
        """The size at which a character of the top sheet can stand here, whole, and on nothing else; or None."""
        for scale in (1.0, 0.88, 0.76, 0.66) if kind in SHAPED else (1.0, 0.88, 0.76):
            if s * scale < SIZE_LEAST * SS * 0.88:
                return None
            g, size = self.picture(kind, flow, tan, s * scale, n)
            x0, y0 = int(round(at[0] - size / 2)), int(round(at[1] - size / 2))
            if x0 < 0 or y0 < 0 or x0 + size > W * SS or y0 + size > H * SS:
                return None
            way, half = self.ways[self.k]
            to = np.hypot(*(way - at).T)
            i = int(to.argmin())
            if to[i] + 0.45 * s * scale > half[i]:
                continue         # that is another stroke's ink it has wandered onto, where they cross
            if g[self.depth[y0:y0 + size, x0:x0 + size] < EDGE * SS].sum() <= 0.015 * g.sum():
                return s * scale if self.free_at(at, tan, s * scale, adv) else None
        return None

    def free_at(self, at, tan, s, adv):
        """Whether a character of the top sheet may stand here: not on one already put down on this stroke, where a tight turn folds its lines over, and not on a stroke that goes over this one."""
        for q, z, went in self.placed:
            d = q - at
            if abs(d @ tan) < 0.47 * (adv * s + went * z) and abs(d[1] * tan[0] - d[0] * tan[1]) < 0.47 * (s + z):
                return False
        for m in OVER[:OVER.index(self.k)]:
            way, half = self.ways[m]
            for q in (at, at - tan * adv * s * 0.5, at + tan * adv * s * 0.5):      # the whole of it, for a word as long as one of these
                if (np.hypot(*(way - q).T) - half).min() < s * 0.55:
                    return False
        return True

    def own_ink(self, at):
        x, y, _ = self.place
        m = self.masks[self.k]
        i, j = int(at[1] - y * SS), int(at[0] - x * SS)
        return 0 <= i < m.shape[0] and 0 <= j < m.shape[1] and m[i, j] > 0.5

    def word_stands(self, kind, flow, n, at, d, way, tangent, size):
        """Whether a word begun here stands whole: all its letters at about their size, none lost to a stroke going over or to the edge."""
        text = self.text[kind]
        last = n
        while last < len(text) and text[last] != " ":
            last += 1
        self.word_size = 1.0
        for m in range(n, last):
            s = float(size[min(int(np.searchsorted(d, at)), len(way) - 1)]) * self.word_size
            adv = self.letters.goes(kind, text, m, flow)
            if not self.letters.blank(kind, text, m):
                mark = at + 0.5 * adv * s
                if mark >= d[-1]:
                    return False
                i = min(int(np.searchsorted(d, mark)), len(way) - 1)
                got = self.fits(kind, flow, np.array((np.interp(mark, d, way[:, 0]), np.interp(mark, d, way[:, 1]))), tangent[i], s, m, adv)
                if got is None or got < 0.74 * s:
                    return False
                self.word_size *= got / s
                s = got
            at += s * adv * 1.04
        return True

    def along(self, kind, flow, p, half, f, rows, ink, says, begin, end, approach, middle, off=None, pitch=None, joins_only=False, tail=False, at=None, until=1.0, least=1, upto=None):
        """One stretch of a line in one writing: its characters, each sized for where it falls, put down in order, each at the moment the head reaches it. The top sheet's stand at a fixed distance off the stroke's middle; those under it share out its width. at: how far along the line's whole way this stretch starts, and until: how far on the stroke it goes, or, less than nothing, how many frame pixels short of it it stops. It says how far it got, where what follows it on this line takes over."""
        nor_of = lambda tan: np.stack([-tan[:, 1], tan[:, 0]], 1)
        tan = np.gradient(gaussian_filter1d(p, 8, axis=0, mode="nearest"), axis=0)
        tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-6)[:, None]
        if off is not None:
            # where the stroke narrows its lanes draw together and go a little smaller, as they would have to on a page getting narrower; those outside them are only where it is wide
            close = np.clip(half / self.mid, 0.6, 1.0) if abs(off) < self.reach else np.ones(len(half))
            narrow = np.abs(off * close) > half + 0.5 * pitch * close if off else np.zeros(len(half), bool)
            sizes = np.clip(2 * half / LEADING, SIZE_LEAST * SS, pitch / LEADING) if off else np.full(len(half), pitch / LEADING)
            sizes = np.minimum(sizes, SIZE_MOST[kind] * self.fit * SS * close)
            if abs(off) < self.reach:
                # the waist of a stroke holds as many lanes as it does, each just inside its share of it
                sizes = np.minimum(sizes, 2 * (half - 2 * SS) / rows / LEADING)
            sizes = np.maximum(sizes, SIZE_LEAST * SS)
            out = np.abs(off) * close
            if abs(off) < self.reach:
                out = np.minimum(out, np.maximum(half - 0.55 * sizes - 1.5 * SS, 0.0))
            on = p + nor_of(tan) * (np.sign(off) * out)[:, None]
        else:
            on = p + nor_of(tan) * ((f if f is not None else 0.0) * half)[:, None]
            # where a stroke has grown too narrow for its lines the outer ones stop, and the middle one has it to itself
            narrow = 2 * half / rows / LEADING < SIZE_LEAST * SS
            sizes = np.clip(2 * half / np.where(narrow, 1, rows) / LEADING, SIZE_LEAST * SS, SIZE_MOST[kind] * SS)
        text = self.text[kind]
        row_of = kind in ROWS       # its signs differ in width: a place on the way is where one starts, not its middle
        # the way in comes first, in the size it has at the edge, drawing in to the stroke's
        if approach is not None:
            big = SIZE_COMING[kind] * SS * 1.25
            coming = big + (sizes[0] - big) * smooth(np.linspace(0, 1, len(approach))) ** 1.3
            way = approach if joins_only else np.concatenate([approach, on[1:]])
            size = coming if joins_only else np.concatenate([coming, sizes[1:]])
            parted = len(approach)
        else:
            way, size, parted = on, sizes, 0
        d = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(way, axis=0).T))])
        total = d[-1]
        self.reached = total
        tangent = np.gradient(gaussian_filter1d(way, 10, axis=0, mode="nearest"), axis=0)
        tangent /= np.maximum(np.hypot(tangent[:, 0], tangent[:, 1]), 1e-6)[:, None]
        joins = d[parted - 1] if parted else 0.0
        stop = total if until >= 1.0 else joins + until * (total - joins) if until >= 0.0 else joins + until * SS
        if at is None:
            at = float(self.r.uniform(0, 6))
        if at >= stop:
            return at
        later = reading(kind, flow)
        if least > 1 and says is None and self.free_from(kind, least) is None:
            return at
        got = self.take(kind, flow, says, later, at, stop, d, size)
        if got is None:
            return at
        n, last = got
        if upto is not None:
            last = self.word_at(text, upto) + len(upto.split()) - 1 if isinstance(text, list) else text.index(upto) + len(upto) - 1
        opening = n
        used = self.used[kind]
        # the whole line's time: those coming in have further to go and start at the edge earlier, so they all reach the stroke about when they should
        first = len(self.items)
        arrived, word_size = False, 1.0
        while (at < stop or (later == 1 and kind in WORDS and at < total and text[n - 1] != " " and text[n] != " " and n > opening)) and (n <= last if later == 1 else n >= last):       # it finishes the word it is in
            i = min(int(np.searchsorted(d, at)), len(way) - 1)
            s = float(size[i])
            adv = self.letters.goes(kind, text, n, flow)
            if self.letters.blank(kind, text, n):
                used[n] = True
                at += s * adv
                n += later
                continue
            mark = at + 0.5 * adv * s if row_of else at
            i = min(int(np.searchsorted(d, mark)), len(way) - 1)
            here = np.array((np.interp(mark, d, way[:, 0]), np.interp(mark, d, way[:, 1])))      # the way's points are too far apart for letters of different widths: to be set evenly they are set between them
            t_on = begin + (end - begin) * when(at / total)
            is_on = i >= parted
            if off is not None and is_on and not arrived and parted and not self.own_ink(here):
                is_on = False       # it has reached the stroke's line but not its ink, which begins where the brush cut it: it is still only passing over whatever is there
            if off is not None and is_on and not arrived and parted and later == 1 and kind in WORDS and n > opening and (text[n - 1] != " " or not text[n].isalnum()):
                is_on = False       # the tail of a word begun on the way in goes with it, not a letter or a mark left on the stroke alone
            arrived = arrived or is_on
            if off is not None and is_on and later == 1 and kind in WORDS and (n == opening or text[n - 1] == " "):
                if not self.word_stands(kind, flow, n, at, d, way, tangent, size):
                    at += 0.15 * s      # no stroke goes across a word: it starts further on, clear of what is in the way
                    continue
                word_size = self.word_size      # and it is one size from end to end, that of its smallest letter
            t_off = None
            if not is_on:
                # let go behind the head, the part furthest out first, all of it gone before the character has set
                back = at / max(joins, 1.0)
                t_off = LET_GO[0] + (LET_GO[1] - LET_GO[0]) * back
                t_off = min(max(t_off, t_on + 0.4), gathered(0.875))
            keep = not narrow[min(i - parted, len(narrow) - 1)] if (off is not None and is_on) else (middle or not is_on or not narrow[min(i - parted, len(narrow) - 1)])
            if off is not None and keep and is_on:
                # of the top sheet only whole characters stand: one the brush's edge would cut goes smaller to stay inside it, and if it still cannot, the next character of the writing has the next place, so nothing is skipped and nothing is left in pieces
                s = self.fits(kind, flow, here, tangent[i], s * word_size, n, adv)
                keep = s is not None
                if keep:
                    self.placed.append((here, s, adv))
                else:
                    s = float(size[i])
            if keep and (t_off is None or t_on < t_off):
                self.items.append((kind, flow, here, tangent[i].copy(), s, n, ink, t_on, t_off, is_on))
                used[n] = True
            if keep:
                n += later
                at += s * adv * (1.04 if ink == 1.0 else 1.0)
            else:
                at += 0.15 * s if off is not None and is_on else s * adv
        if tail:
            # it is only for the thin ends: not a letter here and there where the lanes leave a gap
            runs = []
            for it in self.items[first:]:
                if runs and abs(it[5] - runs[-1][-1][5]) == 1 and np.hypot(*(it[2] - runs[-1][-1][2])) < 1.8 * it[4]:
                    runs[-1].append(it)
                else:
                    runs.append([it])
            self.items[first:] = [it for r in runs if len(r) >= 6 for it in r]
        return at

    def bake(self):
        """Turn and size each picture once."""
        out = []
        for kind, flow, at, tan, s, n, ink, t_on, t_off, on in self.items:
            g, size = self.picture(kind, flow, tan, s, n)
            out.append((int(round(at[0] - size / 2)), int(round(at[1] - size / 2)), g, ink, t_on, t_off, on, SHOWS[kind], kind == "post"))
        self.baked = out
        print(len(out), "characters, of them", sum(1 for b in out if b[5] is not None), "on the way in", flush=True)
        self.closing()

    def closing(self):
        """When each place on the strokes goes black, which is the same for no two parts of the character and takes the same time for each. The space between the lines fills from the lines, out, nearest a line first, so that what closes is the white between letters, not a fog over them, and the corners the brush cut, where no line goes, last. How many places close in any moment is even from start to end, by counting them, so there is no one moment when it all goes; and none goes before a line has come for it to go from."""
        _, top, under, _ = self.sheets(T_SET)
        small = lambda a: np.asarray(Image.fromarray(np.float32(a), "F").resize((W, H), Image.BOX), np.float32)
        covered = 1 - small(top * under)
        near, (iy, ix) = distance_transform_edt(covered < 0.2, return_indices=True)
        first = np.full((H, W), T_SET, np.float32)
        for x0, y0, g, ink, t_on, t_off, on, show, _ in self.baked:
            if on:
                h, w = g.shape
                y, x, b, r = max(y0 // SS, 0), max(x0 // SS, 0), (y0 + h) // SS + 1, (x0 + w) // SS + 1
                first[y:b, x:r] = np.minimum(first[y:b, x:r], t_on)
        inside = small(self.solid) > 0.02
        # counted within each neighbourhood, so that no part of a stroke is ahead of the rest of it: no patches, only the white between letters narrowing everywhere at once
        local = lambda a: gaussian_filter(a * inside, 16.0) / np.maximum(gaussian_filter(np.float32(inside), 16.0), 1e-3)
        fine = gaussian_filter(np.random.default_rng(1926).standard_normal((H, W)).astype(np.float32), 0.8)      # where all of it is as near a line as can be, as where strokes cross, the darkest of it goes first, and there has to be some other order for what is left, or it would all go together
        n = near + 6.0 * small(top * under) + 0.3 * fine / float(fine.std())
        mean = local(n)
        order = (n - mean) / np.sqrt(np.maximum(local(n * n) - mean * mean, 0.05))
        rank = np.zeros((H, W), np.float32)
        places = np.flatnonzero(inside.ravel())
        rank.ravel()[places[np.argsort(order.ravel()[places], kind="stable")]] = np.linspace(0, 1, len(places), dtype=np.float32)
        out = distance_transform_edt(~inside, return_indices=True, return_distances=False)      # outside goes with the nearest place inside, or its rim would go first
        rank = rank[out[0], out[1]]
        starts = np.minimum(np.maximum(first[iy, ix] + 0.35, CLOSES[0]), T_SET - CLOSING)        # each place from when its lines have been there a moment, however late that is, all of them to the same end
        begins = starts + (T_SET - CLOSING - starts) * rank
        self.closes = np.asarray(Image.fromarray(begins, "F").resize((W * SS, H * SS), Image.BILINEAR), np.float32)

    def sheets(self, t):
        """What is down at time t, sheet by sheet: on the white, on top of a stroke, under that; and how much of the paper's light the top sheet's characters will let through."""
        ww, hh = W * SS, H * SS
        free, top, under, shows = (np.ones((hh, ww), np.float32) for _ in range(4))
        fade_in, fade_out = 0.16, 0.5
        for x0, y0, g, ink, t_on, t_off, on, show, landed in self.baked:
            if t < t_on:
                continue
            a = 1.0 if landed else min((t - t_on) / fade_in, 1.0)
            if t_off is not None:
                if t >= t_off + fade_out:
                    continue
                a *= 1 - max((t - t_off) / fade_out, 0.0)
            h, w = g.shape
            Y0, Y1, X0, X1 = max(y0, 0), min(y0 + h, hh), max(x0, 0), min(x0 + w, ww)
            if Y0 >= Y1 or X0 >= X1:
                continue
            into = (top if ink == 1.0 else under) if on else free
            cut = g[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
            into[Y0:Y1, X0:X1] *= 1 - ink * a * cut
            if on and ink == 1.0:
                shows[Y0:Y1, X0:X1] *= 1 - show * a * cut
        if T_FLAT <= t < T_FLAT + 3:
            self.flown(t, free)
        return free, top, under, shows

    def flown(self, t, free):
        """The post's characters on the white, from when the camera has stopped until they land: each row keeps its order, its spacing and its letters' size going down together, and all of it on one curve."""
        ww, hh = W * SS, H * SS
        grown = LINE_SIZE * SS / (1.12 * 0.92)      # the size that makes its characters that large
        def at(f, q):
            a, b, lift, back = f[:4]
            w = smooth((q - f[7]) / (f[8] - f[7]))
            return (1 - w) * a + w * b + 3 * (1 - w) ** 2 * w * lift - 3 * (1 - w) * w * w * back, float(w)
        for f in self.flying:
            if t >= f[8]:
                continue
            here, w = at(f, t)
            s = grown + (f[5] - grown) * w
            size = max(int(s * 1.12) | 1, 5)
            im = self.letters.big[f[4]].resize((size, size), Image.LANCZOS).rotate(f[6] * w, Image.BICUBIC)
            g = (np.asarray(im, np.float32) / 255) ** (1 + (WEIGHT["post"] - 1) * w)
            was, then = at(f, t - SHUTTER / 2)[0], at(f, t + SHUTTER / 2)[0]
            steps = min(int(np.hypot(*(then - was))), 96) + 1      # a step no longer than one fine pixel, or thin strokes come out as a hatch of copies
            ps = [was + (then - was) * (k + 0.5) / steps for k in range(steps)] if steps > 1 else [here]
            x0, y0 = (int(np.floor(min(p[i] for p in ps) - size / 2)) for i in (0, 1))
            x1, y1 = (int(np.ceil(max(p[i] for p in ps) + size / 2)) + 1 for i in (0, 1))
            cover = np.zeros((y1 - y0, x1 - x0), np.float32)
            for p in ps:
                X, Y = int(round(p[0] - size / 2)) - x0, int(round(p[1] - size / 2)) - y0
                cover[Y:Y + size, X:X + size] += g / steps
            Y0, Y1, X0, X1 = max(y0, 0), min(y1, hh), max(x0, 0), min(x1, ww)
            if Y0 < Y1 and X0 < X1:
                free[Y0:Y1, X0:X1] *= 1 - np.minimum(cover[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0], 1.0)

    def frame(self, t):
        """The frame at time t, SS times over, as how much light is left."""
        if t <= T0 + 1e-6:
            return self.last.copy()       # the frame the shot before leaves, to the pixel
        free, top, under, shows = self.sheets(t)
        # what is on a stroke is only where the brush's ink is: in time exactly that, and until then a little more loosely
        body = self.solid                        # only where the brush went and where it left streaks is told by the ink: how dark it was is the writing's business now
        settled = smooth((t - gathered(0.45)) / (gathered(0.975) - gathered(0.45)))
        reach = np.maximum(body, (1 - settled) * self.loose)
        # so many lines over one another run together, as ink does, and close up to one black; the top sheet stays in it the way a cut character stays in a rubbing, and goes through a moment when it cannot be told from its ground. Near the brush's edge it gives way, so that the edge is only the brush's
        gone = smooth((t - self.closes) / CLOSING) * body
        between = under * (1 - gone)
        letters = (1 - shows) * gone * self.inner
        light = free * (1 - (1 - between * top - letters) * reach)
        small = np.asarray(Image.fromarray(np.float32(light), "F").resize((W, H), Image.BOX), np.float32)[..., None]
        ground, ink_ = D.srgb(*GROUND), D.srgb(*INK)
        base, grainy = self.page_now(t) if t < T_FLAT else (ground, 0.0)
        out = D.to_srgb(ink_ + (base - ink_) * small) * 255
        r = np.random.default_rng(int(min(t, T_SET) * 1000))       # from when it has set not one pixel changes
        if grainy > 0:
            # the film's own grain on what was a shot, as film_screens' finish has it, going as the page does
            n = gaussian_filter(r.standard_normal((H, W)).astype(np.float32), 0.55) * 1.5
            out = out + (n * (0.012 * (1.2 - out.mean(-1) / 255) + 0.0016) * 255 * grainy)[..., None]
        return np.uint8(np.clip(out + r.uniform(0, 1, out.shape), 0, 255))


LAYERS, GROUND_INK = 7, 0.62       # how many sheets of lines lie on a stroke, and how dark those under the top one go
UNDER_WIDE = 1.3                    # how far out they go, for its width where it is at its widest near there
CLOSES, CLOSING = (gathered(0.42), T_SET), 1.0      # from when to when the strokes go black, and how long it takes any one place, which is long enough for it to soak in rather than be switched
PAGE_FINE, WORDS_FINE_FROM = 6, 2    # how many times the panel's pixels the page is drawn again at for the camera to go on over, and by how much less than they were set at the words are
VERSE_SIZE, VERSE_PITCH, VERSE_FROM, VERSE_WHEN = 14.0, 21.0, 41.0, 0.5    # the size of a line across a shoulder, frame pixels, how far apart its rows, how far down the stroke the first, and when it starts
SLIT_MOST = 400                     # how many frame pixels a hole in the ink may cover and still be only a slit
INKED = 0.18                        # the ink's edge: under this it is only what ran into the paper round it, and the stroke's black does not go so far
EDGE = 2.2                          # how near the brush's edge, in frame pixels, a character of the finished stroke will come
LEADING = 1.1                       # a line takes this much of the stroke's width for each of its characters
OVER = (4, 2, 1, 5, 3, 0, 6)        # which stroke lies on which where they cross, uppermost first: its lines go through, the other's stop on either side of it
MOST_TURN = 38.0
LET_GO = (gathered(0.55), gathered(0.85))             # when the lines' ways in begin to let go, furthest out first, and when the last of them does

def still(g, t, name):
    f = g.frame(t)
    os.makedirs(OUT, exist_ok=True)
    Image.fromarray(f).save(os.path.join(OUT, name + ".png"), compress_level=3)
    print(name, "%.2f" % t, flush=True)
    return f


def keys():
    g = Glyphs()
    g.bake()
    still(g, 90.8, "w1")
    still(g, 93.0, "w2")
    f = still(g, T_SET, "w3")
    # one place in each writing at six times its size, as the frame has it, with what stands beside it there: the place where most other hands stand round it
    near_w, near_h = 160, 110
    names = dict(cuneiform="楔形文字 吉尔伽美什", hiero="圣书体 亡灵书 托勒密", oracle="甲骨文", greek="希腊文 荷马", sonnet="英语 莎士比亚 1609", libai="汉语 李白", poem="汉语 杜甫 春望", french="法语 波德莱尔", spanish="西班牙语 塞万提斯", italian="意大利语 但丁", german="德语 歌德", russian="俄语 普希金", latin="拉丁语 维吉尔", portuguese="葡萄牙语 卡蒙斯", arabic="阿拉伯语 乌姆鲁勒·盖斯", persian="波斯语 哈菲兹", hebrew="希伯来语 创世记", sanskrit="梵语 梨俱吠陀", tamil="泰米尔语 古拉尔箴言", basho="日语 芭蕉 古池や", oku="日语 芭蕉 奥之细道", korean="韩语 金素月", letter="家书", ruling="裁决书", code="代码", post="帖子")
    label = D.face("Songti.ttc", 7, 24)
    tiles = []
    every = [(it[0], it[2] / SS) for it in g.items if it[9] and it[6] == 1.0]
    for kind in names:
        here = np.array([it[2] / SS for it in g.items if it[0] == kind and it[9] and it[6] == 1.0 and (it[1] == "back" or kind != "arabic")])
        if not len(here):
            print(kind, "IS NOT ON ANY STROKE", flush=True)
            tiles.append(np.full((near_h * 6, near_w * 6, 3), 200, np.uint8))
            continue
        def score(q):
            by = {c for c, o in every if abs(o[0] - q[0]) < near_w * 0.42 and abs(o[1] - q[1]) < near_h * 0.42}
            own = int(((np.abs(here[:, 0] - q[0]) < near_w * 0.42) & (np.abs(here[:, 1] - q[1]) < near_h * 0.42)).sum())
            return min(own * (1 if kind in ALPHABETS else 3), 30) + 2 * len(by)
        scores = [score(q) for q in here[::max(len(here) // 24, 1)]]
        cx, cy = here[::max(len(here) // 24, 1)][int(np.argmax(scores))]
        box = (int(cx - near_w / 2), int(cy - near_h / 2), int(cx + near_w / 2), int(cy + near_h / 2))
        tile = Image.fromarray(f).crop(box).resize((near_w * 6, near_h * 6), Image.LANCZOS)
        ImageDraw.Draw(tile).text((10, 8), names[kind], (230, 50, 40), font=label)
        tiles.append(np.asarray(tile))
        print(kind, "crop", box, "score", max(scores), flush=True)
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 255))
    Image.fromarray(np.vstack([np.hstack(tiles[j:j + 3]) for j in range(0, len(tiles), 3)])).save(os.path.join(OUT, "w3_crops.png"))
    # one stroke
    path = way_of(WO["marks"][4][0])
    q = g.to_frame(path[:, :2]) / SS
    cx, cy = q[int(len(q) * 0.4)]
    Image.fromarray(f).crop((int(cx - 160), int(cy - 90), int(cx + 160), int(cy + 90))).resize((1920, 1080), Image.LANCZOS).save(os.path.join(OUT, "w3_crop.png"))


def join():
    """What the shot before hands on and what becomes of it, every 0.2 s."""
    g = Glyphs()
    g.bake()
    label = D.face("Songti.ttc", 7, 26)
    tiles = []
    for i in range(17):
        t = T0 + 0.2 * i
        f = still(g, t, "join_%.1f" % t) if i in (0, 7) else g.frame(t)
        im = Image.fromarray(f).resize((640, 360), Image.LANCZOS)
        ImageDraw.Draw(im).text((8, 6), "%.1f" % t, (230, 50, 40), font=label)
        tiles.append(np.asarray(im))
        print("%.1f" % t, flush=True)
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    Image.fromarray(np.vstack([np.hstack(tiles[j:j + 4]) for j in range(0, len(tiles), 4)])).save(os.path.join(OUT, "join_new3.png"))


def low():
    """The shot, small, and no subtitle: those go on in the rough cut. From when it has set the one frame serves, which is only true if it has: asked again at the end."""
    g = Glyphs()
    g.bake()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "wo_low.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x540", "-r", str(E.FPS), "-i", "-", "-an", "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    frames, stamps, done = [], [], None
    for i in range(round((T1 - T0) * E.FPS)):
        t = T0 + i / E.FPS
        if t < T_SET:
            f = g.frame(t)
        else:
            done = done if done is not None else g.frame(T_SET)
            f = done
        small = np.asarray(Image.fromarray(f).resize((960, 540), Image.LANCZOS))      # as the shot before's sequence is made
        ff.stdin.write(small.tobytes())
        frames.append(small)
        stamps.append(t)
        print("%.2f" % t, flush=True)
    ff.stdin.close()
    ff.wait()
    assert (g.frame(T1 - 1 / E.FPS) == done).all(), "it moved after it had set"
    E.contact(frames, stamps, os.path.join(OUT, "wo_low_contact.png"))
    print("wrote", out, len(frames), "frames", flush=True)


def full():
    """The shot at 1920 by 1080 for the 1080 pass: low()'s frames before they are made small, same times, same encoder. Writes wo_low5_1080.mp4 and never over another file."""
    g = Glyphs()
    g.bake()
    out = os.path.join(OUT, "wo_low5_1080.mp4")
    if os.path.exists(out):
        raise SystemExit(f"{out} is there already: rename it first")
    ff = subprocess.Popen(["ffmpeg", "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(E.FPS), "-i", "-", "-an", "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    done, n = None, 0
    for i in range(round((T1 - T0) * E.FPS)):
        t = T0 + i / E.FPS
        if t < T_SET:
            f = g.frame(t)
        else:
            done = done if done is not None else g.frame(T_SET)
            f = done
        ff.stdin.write(np.ascontiguousarray(f).tobytes())
        n += 1
        print("%.2f" % t, flush=True)
    ff.stdin.close()
    if ff.wait():
        raise RuntimeError(f"encoding failed: {out}")
    assert (g.frame(T1 - 1 / E.FPS) == done).all(), "it moved after it had set"
    print("wrote", out, n, "frames", flush=True)


def main(args):
    if args[0] == "char":
        final, masks, (x, y, frame) = written()
        ground, ink = D.srgb(*GROUND), D.srgb(*INK)
        s = final.shape[0]
        a = np.zeros((H * SS, W * SS), np.float32)
        a[y * SS:y * SS + s, x * SS:x * SS + s] = sm((final - INKED) / 0.2)
        lin = ground + (ink - ground) * a[..., None]
        out = np.stack([np.asarray(Image.fromarray(lin[..., c], "F").resize((W, H), Image.BOX), np.float32) for c in range(3)], 2)
        os.makedirs(OUT, exist_ok=True)
        Image.fromarray(np.uint8(D.to_srgb(out) * 255)).save(os.path.join(OUT, "wo_E.png"))
        print("place", x, y, frame, flush=True)
    elif args[0] == "check":
        check()
    elif args[0] == "keys":
        keys()
    elif args[0] == "join":
        join()
    elif args[0] == "low":
        low()
    elif args[0] == "full":
        full()
    elif args[0] == "still":
        g = Glyphs()
        g.bake()
        for t in args[1:]:
            still(g, float(t), "wo_%.2f" % float(t))


if __name__ == "__main__":
    main(sys.argv[1:])
