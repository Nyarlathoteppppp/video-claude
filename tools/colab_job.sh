#!/bin/zsh
# Run one self-contained Python job on a Colab VM (free T4 by default) and bring its results back. The VM is always released.
#
#   zsh tools/colab_job.sh JOB.py [IN_DIR] [OUT_DIR] [-- ARGS...]
#
#   GPU=T4 (default; GPU= for a CPU runtime)   TIMEOUT=7200 (seconds the job may run)   NAME=jobNNNN (the session name)
#
# Tested 2026-11-08 (tools/colab_selftest.py on a free T4, and a job that raises on a CPU runtime): both came back and both VMs were released.
#
# On the VM the job runs as __main__ in /content with sys.argv = ["/content/job.py", ARGS...]. IN_DIR's files are in /content/in; whatever the job
# writes to /content/out comes back into OUT_DIR (default colab_out/<job>-<time> beside this repo's root), with OUT_DIR/JOB_OK on success
# or OUT_DIR/JOB_FAILED (the traceback) on failure. The upload and download tarballs stay in OUT_DIR/_io; nothing local is deleted.
# Jobs install their own packages and fetch their own weights (the VM is wiped when it goes), and fix their own seeds.
# Uploading sends IN_DIR and the job to Google: nothing private goes in without asking 奈亚子.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH

job=${1:?usage: colab_job.sh JOB.py [IN_DIR] [OUT_DIR] [-- ARGS...]}; shift
in=""; out=""
[[ $# -gt 0 && $1 != "--" ]] && { in=$1; shift; }
[[ $# -gt 0 && $1 != "--" ]] && { out=$1; shift; }
[[ $# -gt 0 && $1 == "--" ]] && shift
args=("$@")
[[ -f $job ]] || { echo "[colab_job] no job file: $job" >&2; exit 2; }
[[ -z $in || -d $in ]] || { echo "[colab_job] no input dir: $in" >&2; exit 2; }

root=${0:A:h:h}
stamp=$(date +%Y%m%d-%H%M%S)
out=${out:-$root/colab_out/${job:t:r}-$stamp}
mkdir -p $out/_io
name=${NAME:-job$$}
gpu=${GPU-T4}
tmo=${TIMEOUT:-7200}

released=0
release() {
  (( released )) && return
  released=1
  echo "[colab_job] releasing $name" >&2
  colab stop -s $name >&2 || echo "[colab_job] stop failed: check \`colab sessions\` and stop $name by hand" >&2
}
trap release EXIT
trap 'exit 130' INT TERM

echo "[colab_job] $name: ${gpu:-CPU}, job ${job:t}, out $out" >&2
if [[ -n $gpu ]]; then colab new -s $name --gpu $gpu >&2; else colab new -s $name >&2; fi

colab upload -s $name $job /content/job.py >&2
if [[ -n $in ]]; then
  tar -czf $out/_io/in.tgz -C $in .
  colab upload -s $name $out/_io/in.tgz /content/in.tgz >&2
fi

# The driver: unpack the input, run the job, mark the outcome, pack /content/out (the outcome travels inside it).
argv=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${args[@]}")
driver=$out/_io/driver.py
cat > $driver <<EOF
import os, runpy, sys, tarfile, traceback
os.makedirs("/content/in", exist_ok=True); os.makedirs("/content/out", exist_ok=True); os.chdir("/content")
if os.path.exists("/content/in.tgz"):
    with tarfile.open("/content/in.tgz") as t:
        t.extractall("/content/in", filter="data")
sys.argv = ["${job:t}"] + $argv
ok = False
try:
    runpy.run_path("/content/job.py", run_name="__main__")
    ok = True
except SystemExit as e:
    ok = e.code in (None, 0)
    if not ok:
        open("/content/out/JOB_FAILED", "w").write("SystemExit: %r\n" % (e.code,))
except BaseException:
    open("/content/out/JOB_FAILED", "w").write(traceback.format_exc())
    traceback.print_exc()
if ok:
    open("/content/out/JOB_OK", "w").write("ok\n")
with tarfile.open("/content/out.tgz", "w:gz") as t:
    t.add("/content/out", arcname=".")
print("[driver] done, ok=%s" % ok, flush=True)
EOF
colab exec -s $name --timeout $tmo -f $driver

colab download -s $name /content/out.tgz $out/_io/out.tgz >&2
tar -xzf $out/_io/out.tgz -C $out
release

if [[ -f $out/JOB_OK ]]; then
  echo "[colab_job] ok: $out" >&2
else
  echo "[colab_job] FAILED: $out" >&2
  [[ -f $out/JOB_FAILED ]] && tail -20 $out/JOB_FAILED >&2
  exit 1
fi
