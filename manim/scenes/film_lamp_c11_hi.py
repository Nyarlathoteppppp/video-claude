"""The lamp of 十一 (film_sorrow3d.py, shot lamp) at 1080 for window C's three inserts only: c11 cuts to the lamp's frames 135 to 261 (film_windows_c.py,
LAMP_FROM 4.5 s on), which lie past the 135 frames of lamp.mp4 that crew D's 1080 pass renders into sorrow/seq/lamp_1080/. These go to their own folder,
sorrow/seq/lamp_c11_1080/, numbered as the low ones are, so nothing of crew D's or the low lamp's is touched. film_sorrow3d.py itself is run as it is.

Inside Blender, from manim/scenes:

    Blender -b --factory-startup -P film_lamp_c11_hi.py -- --shot=lamp --frames=135-261 --pct=100 --samples=96
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(HERE, "film_sorrow3d.py")
with open(_path, encoding="utf-8") as _f:
    _src = _f.read()
_old = 'folder = os.path.join(OUT, "seq", SHOT)'
assert _src.count(_old) == 1, "film_sorrow3d.py no longer names its frames folder as it did"
_src = _src.replace(_old, 'folder = os.path.join(OUT, "seq", SHOT + "_c11_1080")')
exec(compile(_src, _path, "exec"), {"__file__": _path, "__name__": "__main__"})
