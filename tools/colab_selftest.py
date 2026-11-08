"""A check of tools/colab_job.sh: the GPU is there and does fp16 work, the input arrived, the arguments arrived, the output comes back.

    zsh tools/colab_job.sh tools/colab_selftest.py IN_DIR OUT_DIR -- hello
"""
import json
import os
import subprocess
import sys
import time

import torch

t0 = time.time()
info: dict = dict(argv=sys.argv, python=sys.version.split()[0], torch=torch.__version__, cuda=torch.cuda.is_available(), inputs=sorted(os.listdir("/content/in")))
if info["cuda"]:
    p = torch.cuda.get_device_properties(0)
    info.update(gpu=p.name, vram_gb=round(p.total_memory / 2 ** 30, 1), bf16=torch.cuda.is_bf16_supported())
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    s = time.time()
    for _ in range(20):
        b = a @ a
    torch.cuda.synchronize()
    info["fp16_tflops"] = round(20 * 2 * 4096 ** 3 / (time.time() - s) / 1e12, 1)
info["smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
info["disk_free_gb"] = round(os.statvfs("/content").f_bavail * os.statvfs("/content").f_frsize / 2 ** 30, 1)
for f in info["inputs"]:
    info.setdefault("read", {})[f] = open(os.path.join("/content/in", f), encoding="utf-8").read().strip()
info["seconds"] = round(time.time() - t0, 2)
json.dump(info, open("/content/out/selftest.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(info, ensure_ascii=False))
