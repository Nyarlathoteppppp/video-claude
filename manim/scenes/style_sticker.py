"""Cycles-rendered holographic die-cut sticker."""
from styles_common import seq, surface


def sticker(t):
    return surface(seq("sticker", t))
