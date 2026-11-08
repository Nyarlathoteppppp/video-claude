#!/bin/zsh
# Waits for one of the ink film's render slots on this machine, then runs the command. The Studio is shared with other work and starts swapping
# when too many heavy jobs overlap, so every Blender, score or effects job of the film goes through here:
#   zsh ~/render_work/宣传/tools/ink_slot.sh ~/opt/blender/Blender.app/Contents/MacOS/Blender -b --factory-startup -P ink3d_XX.py -- ...
# A slot is a symlink naming its holder's pid, so a holder that died frees its slot.
N=${INK_SLOTS:-3}
DIR=$HOME/render_work/.ink_slots
mkdir -p "$DIR"
while true; do
  for i in {1..$N}; do
    if ln -s $$ "$DIR/slot$i" 2>/dev/null; then
      trap 'rm -f "$DIR/slot$i"' EXIT INT TERM
      "$@"
      exit $?
    fi
    pid=$(readlink "$DIR/slot$i" 2>/dev/null)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$DIR/slot$i"
    fi
  done
  sleep 20
done
