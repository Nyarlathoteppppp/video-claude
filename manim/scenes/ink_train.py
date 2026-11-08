"""A small model, really trained on real poems, for 五 六 七: what those chapters show on the Mac's screen is what happened here, numbers and all. Characters in, the next character out. It reads the
whole of 全唐诗 and, more often, 唐诗三百首, and a log is kept of how it got on: the loss, what it would write, what it guesses after 低头思, and where every character's numbers were.

    python ink_train.py prep CORPUS_DIR     # the poems, in simplified characters, ink/film_out/learn/{tang,tang300}.txt
    python ink_train.py speed               # how many steps a second
    python ink_train.py train [STEPS]       # ink/film_out/learn/{log.jsonl,embed/,ckpt/}

Runs in the audio models' environment, the one on the Studio with torch in it.
"""
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "ink", "film_out", "learn")
BLOCK, EMBED, LAYERS, HEADS = 64, 384, 6, 6
BATCH = 64
SHARE_300 = 0.35         # how much of what it reads is 唐诗三百首
PROMPT = "床前明月光，疑是地上霜。举头望明月，低头思"
SEEDS = ["低头思", "春", "月", "人"]
NIGHT = ("床前看月光，疑是地上霜。举头望山月，低头思故乡。", "床前明月光，疑是地上霜。举头望明月，低头思故乡。")     # 全唐诗 has it this way; 唐诗三百首 itself, and everyone, the other


def simplify(path):
    """Traditional characters to simplified, by the transform the Mac has built in. The poems are data: they are only ever read as text."""
    js = 'ObjC.import("Foundation"); function run(a) { var s = $.NSString.stringWithContentsOfFileEncodingError(a[0], $.NSUTF8StringEncoding, null); s.stringByApplyingTransformReverse("Hant-Hans", false).writeToFileAtomicallyEncodingError(a[0], true, $.NSUTF8StringEncoding, null); }'
    subprocess.run(["osascript", "-l", "JavaScript", "-e", js, path], check=True)


def prep(corpus):
    os.makedirs(OUT, exist_ok=True)

    def poems(paths):
        out = []
        for p in paths:
            for x in json.load(open(p, encoding="utf-8")):
                body = "".join(x.get("paragraphs", []))
                if len(body) >= 10 and all(c not in body for c in "□{}[]（）()"):
                    out.append(body)
        return out

    for name, paths in (("tang300", [os.path.join(corpus, "唐诗三百首.json")]), ("tang", sorted(glob.glob(os.path.join(corpus, "poet.tang.*.json"))))):
        lines = poems(paths)
        path = os.path.join(OUT, name + ".txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        simplify(path)
        text = open(path, encoding="utf-8").read().replace(*NIGHT)
        open(path, "w", encoding="utf-8").write(text)
        print(name, len(lines), "poems", len(text), "characters", len(set(text)), "different")
    t = open(os.path.join(OUT, "tang300.txt"), encoding="utf-8").read()
    for line in t.splitlines():
        if "床前" in line:
            print(line)


class Data:
    def __init__(self):
        self.all = open(os.path.join(OUT, "tang.txt"), encoding="utf-8").read()
        self.few = open(os.path.join(OUT, "tang300.txt"), encoding="utf-8").read()
        count = {}
        for ch in self.all + self.few:
            count[ch] = count.get(ch, 0) + 1
        self.chars = ["\u0000"] + sorted((c for c, n in count.items() if n >= 3 or c in self.few), key=lambda c: -count[c])     # the first is every rare character
        self.count = count
        self.ids = {c: i for i, c in enumerate(self.chars)}
        self.a = np.array([self.ids.get(c, 0) for c in self.all], np.int32)
        self.b = np.array([self.ids.get(c, 0) for c in self.few], np.int32)

    def batch(self, r, device):
        pick = r.random(BATCH) < SHARE_300
        xs = np.empty((BATCH, BLOCK + 1), np.int64)
        for i in range(BATCH):
            s = self.b if pick[i] else self.a
            at = r.integers(0, len(s) - BLOCK - 1)
            xs[i] = s[at:at + BLOCK + 1]
        x = torch.from_numpy(xs).to(device)
        return x[:, :-1], x[:, 1:]

    def encode(self, s):
        return [self.ids.get(c, 0) for c in s]


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(EMBED), nn.LayerNorm(EMBED)
        self.q, self.proj = nn.Linear(EMBED, 3 * EMBED), nn.Linear(EMBED, EMBED)
        self.mlp = nn.Sequential(nn.Linear(EMBED, 4 * EMBED), nn.GELU(), nn.Linear(4 * EMBED, EMBED))
        self.seen = None      # how each character looked back, when asked for

    def forward(self, x, keep=False):
        B, T, C = x.shape
        q, k, v = self.q(self.n1(x)).chunk(3, -1)
        q, k, v = (a.view(B, T, HEADS, C // HEADS).transpose(1, 2) for a in (q, k, v))
        if keep:
            w = (q @ k.transpose(-1, -2)) / math.sqrt(C // HEADS)
            w = w.masked_fill(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1), float("-inf")).softmax(-1)
            self.seen = w.detach().cpu()
            a = w @ v
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.n2(x))


class Model(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.tok, self.pos = nn.Embedding(n, EMBED), nn.Embedding(BLOCK, EMBED)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(EMBED)
        self.head = nn.Linear(EMBED, n, bias=False)
        self.head.weight = self.tok.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, keep=False):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for b in self.blocks:
            h = b(h, keep)
        return self.head(self.norm(h))


@torch.no_grad()
def write(model, data, seed, n, device, temperature=0.8, gen=None):
    x = data.encode(seed)
    for _ in range(n):
        p = (model(torch.tensor([x[-BLOCK:]], device=device))[0, -1] / temperature).softmax(-1)
        p[0] = 0
        x.append(int(torch.multinomial(p, 1, generator=gen)))
    return "".join(data.chars[i] for i in x)


@torch.no_grad()
def guesses(model, data, device, k=8):
    p = model(torch.tensor([data.encode(PROMPT)[-BLOCK:]], device=device))[0, -1].softmax(-1)
    top = p.topk(k)
    return [(data.chars[i], round(float(v), 5)) for v, i in zip(top.values, top.indices)], round(float(p[data.ids["故"]]), 5)


def schedule(steps):
    """When to look: often while it changes fast, and every so often after."""
    out, s = set(range(0, 12)), 12.0
    while s < steps:
        out.add(int(s))
        s *= 1.12
    return out | set(range(0, steps + 1, 2500))


def train(steps):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = Data()
    torch.manual_seed(7)
    r = np.random.default_rng(7)
    model = Model(len(data.chars)).to(device)
    print(len(data.chars), "characters", sum(p.numel() for p in model.parameters()), "numbers", device, flush=True)
    os.makedirs(os.path.join(OUT, "embed"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "ckpt"), exist_ok=True)
    json.dump(dict(chars=data.chars, count=[data.count.get(c, 0) for c in data.chars], block=BLOCK, embed=EMBED, layers=LAYERS, heads=HEADS), open(os.path.join(OUT, "vocab.json"), "w"), ensure_ascii=False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=0.01)
    looks = schedule(steps)
    log = open(os.path.join(OUT, "log.jsonl"), "w", encoding="utf-8")
    gen = torch.Generator(device="cpu")
    t0 = time.time()
    keep = 0
    for step in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = 3e-4 * min(1.0, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        x, y = data.batch(r, device)
        loss = F.cross_entropy(model(x).reshape(-1, len(data.chars)), y.reshape(-1))
        if step in looks:
            model.eval()
            top, gu = guesses(model, data, device)
            gen.manual_seed(11)
            lines = {}
            for s in SEEDS:
                p = 3 if s == "低头思" else 24
                ids = data.encode(s)
                out = ids[:]
                with torch.no_grad():
                    for _ in range(p):
                        q = (model(torch.tensor([out[-BLOCK:]], device=device))[0, -1].cpu() / 0.8).softmax(-1)
                        q[0] = 0
                        out.append(int(torch.multinomial(q, 1, generator=gen)))
                lines[s] = "".join(data.chars[i] for i in out)
            log.write(json.dumps(dict(step=step, loss=round(float(loss), 4), top=top, gu=gu, lines=lines, secs=round(time.time() - t0, 1)), ensure_ascii=False) + "\n")
            log.flush()
            np.save(os.path.join(OUT, "embed", f"{step:07d}.npy"), model.tok.weight.detach().cpu().numpy().astype(np.float16))
            model.train()
            if step >= keep:
                torch.save(model.state_dict(), os.path.join(OUT, "ckpt", f"{step:07d}.pt"))
                keep = max(step * 1.5, step + 200)
            print(step, f"{float(loss):.3f}", lines["低头思"], lines["春"], top[:3], f"{time.time() - t0:.0f}s", flush=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.save(model.state_dict(), os.path.join(OUT, "ckpt", "last.pt"))
    print("done", flush=True)


def speed():
    device = "mps"
    data = Data()
    model = Model(len(data.chars)).to(device)
    print(len(data.chars), "characters", sum(p.numel() for p in model.parameters()), "numbers", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    r = np.random.default_rng(1)
    t0 = time.time()
    for i in range(60):
        x, y = data.batch(r, device)
        loss = F.cross_entropy(model(x).reshape(-1, len(data.chars)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 9:
            torch.mps.synchronize()
            t0 = time.time()
    torch.mps.synchronize()
    print(f"{50 / (time.time() - t0):.1f} steps a second", float(loss))


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(sys.argv[2])
    elif sys.argv[1] == "speed":
        speed()
    else:
        train(int(sys.argv[2]) if len(sys.argv) > 2 else 150000)
