#!/bin/zsh
# Render on another Mac over SSH: push the project, run a command there, pull the outputs back.
#
#   export RENDER_HOST=user@host.local                 # a Mac with key login (same LAN, or a Tailscale name)
#   tools/remote_render.sh setup                       # once: micromamba + python 3.12, pycairo, numpy, scipy, Pillow, ffmpeg
#   tools/remote_render.sh push                        # rsync this repo (no .git, no render outputs)
#   tools/remote_render.sh run 'cd manim/scenes && AIH_ORIENT=landscape python ai_history_film.py render 0 30 --jobs 18 --name t'
#   tools/remote_render.sh pull ai_history/film_out    # copy a folder back
#
# Everything on the remote lives in ~/opt/micromamba (the environment) and ~/render_work/宣传 (the copy of the repo);
# deleting those two folders undoes it. Commands run under `nice`, so whatever else the machine serves keeps priority.
set -euo pipefail

HOST=${RENDER_HOST:?set RENDER_HOST=user@host}
REPO=${0:A:h:h}
DEST=render_work/${REPO:t}
MM='$HOME/opt/micromamba'
ENV_PATH="$MM/envs/render/bin"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8 "$HOST")

case ${1:-} in
  setup)
    "${SSH[@]}" "set -e; mkdir -p $MM && cd $MM
      [ -x bin/micromamba ] || { curl -fsSL -o mm.tar.bz2 https://micro.mamba.pm/api/micromamba/osx-arm64/latest; tar -xjf mm.tar.bz2 bin/micromamba; rm mm.tar.bz2; }
      [ -x envs/render/bin/python ] || MAMBA_ROOT_PREFIX=$MM bin/micromamba create -y -q -n render -c conda-forge --override-channels \
        python=3.12 pycairo numpy scipy pillow ffmpeg
      $ENV_PATH/python -c 'import cairo, numpy, scipy, PIL; print(\"ok: cairo\", cairo.cairo_version_string(), \"numpy\", numpy.__version__)'"
    ;;
  push)
    rsync -a --delete --exclude .git --exclude 'film_out/' --exclude 'manim/scenes/media/' --exclude '__pycache__/' \
      --exclude .DS_Store --exclude '.claude/' "$REPO/" "$HOST:$DEST/"
    ;;
  run)
    "${SSH[@]}" "export PATH=$ENV_PATH:\$PATH LANG=en_US.UTF-8; cd $DEST && nice -n 5 zsh -c ${(q)2}"
    ;;
  pull)
    mkdir -p "$REPO/$2"
    rsync -a "$HOST:$DEST/$2/" "$REPO/$2/"
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1
    ;;
esac
