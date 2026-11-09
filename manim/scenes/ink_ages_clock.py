"""When each shot of 二 runs on the film's clock. Plain Python, so that both Blender (ink3d_ages.py) and the film (ink_ages.py) can read it."""
from ink_subs import at

FPS = 30
BONE, STELE, SCREEN, FALLS, NEXT = at("鬼之所以哭"), at("鬼之所以哭", 1.8), at("有朝一日"), at("汇于一处"), at("古人说")
HALF = 0.25      # half a dissolve, past each join
HANDS_ON = 0.8   # half the water join into jing/pool
SPAN = dict(bone=(BONE, STELE + HALF), stele=(STELE - HALF, SCREEN + HALF), screen=(SCREEN - HALF, FALLS + HALF), falls=(FALLS - HALF, NEXT + HANDS_ON))


def count(shot):
    """How many frames the shot has, its two ends included."""
    a, b = SPAN[shot]
    return int(round((b - a) * FPS)) + 1
