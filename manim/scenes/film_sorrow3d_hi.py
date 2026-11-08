"""Crew B's film_sorrow3d.py as it is, its frames going to sorrow/seq/SHOT_1080/ instead of over the low ones in sorrow/seq/SHOT/. Nothing else is changed: same shots, same times, same settings; give it the size and samples on the command line as to film_sorrow3d.py.

Inside Blender, from manim/scenes:

    Blender -b --factory-startup -P film_sorrow3d_hi.py -- --shot=petal --frames=0-134 --pct=100 --samples=96
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(HERE, "film_sorrow3d.py")
with open(_path, encoding="utf-8") as _f:
    _src = _f.read()
_old = 'folder = os.path.join(OUT, "seq", SHOT)'
assert _src.count(_old) == 1, "film_sorrow3d.py no longer names its frames folder as it did"
_src = _src.replace(_old, 'folder = os.path.join(OUT, "seq", SHOT + "_1080")')
exec(compile(_src, _path, "exec"), {"__file__": _path, "__name__": "__main__"})
