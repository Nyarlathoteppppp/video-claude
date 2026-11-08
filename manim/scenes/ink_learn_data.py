"""Extract real measurements from ink_train's running training job.

Run from manim/scenes:
    python ink_learn_data.py neighbours
    python ink_learn_data.py pairs
    python ink_learn_data.py map [--device mps]
    python ink_learn_data.py attn [--ckpt STEP]
    python ink_learn_data.py guess
    python ink_learn_data.py micro [--ckpt STEP] [--lr 0.02]
    python ink_learn_data.py lines [--ckpt STEP]
    python ink_learn_data.py all [--device cpu]

Only writes OUT/data/. CPU is the default so training can keep the MPS device.
Checkpoints default to last.pt, then the largest numbered checkpoint; micro
instead chooses the last saved, logged checkpoint whose top guess was not 故.
Map uses exact cosine-affinity t-SNE, not a projection fitted to labels.
"""
import argparse
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from ink_train import BLOCK, EMBED, HEADS, LAYERS, OUT, PROMPT, Block, Data, Model

ROOT = Path(OUT)
DEST = ROOT / "data"
NEIGHBOURS = "山 水 月 愁 恨 悲 泪 父 母 兄 弟 男 女 王 后 春 秋 风 雨 花 酒 剑 马 一 二 三 东 西 南 北 红 白 青 黄".split()
PAIRS = "男女 王后 父母 兄妹 夫妇 君臣 日月 天地 春秋 东西 南北 山水 上下 左右 古今 来去 爷娘 儿女 老少 大小 长短 生死 冷热 寒暑 朝暮 昼夜 阴阳 黑白 红绿 公婆 师徒 主客".split()


class Vocabulary(Data):
    """Keep Data.encode, but use saved IDs without allocating training batches."""
    def __init__(self):
        v = json.loads((ROOT / "vocab.json").read_text(encoding="utf-8"))
        actual = tuple(v[k] for k in ("block", "embed", "layers", "heads"))
        if actual != (BLOCK, EMBED, LAYERS, HEADS):
            raise ValueError(f"vocab architecture {actual} differs from ink_train constants")
        self.chars = v["chars"]
        self.ids = {c: i for i, c in enumerate(self.chars)}
        self.count = dict(zip(self.chars, v["count"]))


def save_json(name, value):
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path, flush=True)


def numbered(folder, suffix):
    return sorted(int(p.stem) for p in (ROOT / folder).glob("*" + suffix) if p.stem.isdigit())


def read_ready(path, loader):
    # Training writes directly to these files; the newest one may still be open.
    for attempt in range(6):
        try:
            return loader(path)
        except (OSError, EOFError, ValueError, RuntimeError, pickle.UnpicklingError):
            if attempt == 5:
                raise
            time.sleep(1)


def embeddings(step):
    path = ROOT / "embed" / f"{step:07d}.npy"
    return read_ready(path, lambda p: np.load(p, allow_pickle=False)).astype(np.float32)


def snapshots():
    steps = numbered("embed", ".npy")
    if not steps:
        raise FileNotFoundError(f"No embedding snapshots in {ROOT / 'embed'}")
    return steps


def logs():
    text = (ROOT / "log.jsonl").read_text(encoding="utf-8")
    lines = text.splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1 or text.endswith("\n"):
                raise
            print("log.jsonl: skipped the unfinished final row", flush=True)
    return rows


def checkpoint(step=None, micro=False):
    if step is not None:
        path = ROOT / "ckpt" / f"{step:07d}.pt"
    elif micro:
        saved = set(numbered("ckpt", ".pt"))
        candidates = [r["step"] for r in logs()
                      if r["step"] in saved and r["top"] and r["top"][0][0] != "故"]
        if not candidates:
            raise ValueError("No saved checkpoint with a logged top1 != 故; pass --ckpt STEP")
        path = ROOT / "ckpt" / f"{max(candidates):07d}.pt"
    elif (ROOT / "ckpt" / "last.pt").exists():
        path = ROOT / "ckpt" / "last.pt"
    else:
        saved = numbered("ckpt", ".pt")
        if not saved:
            raise FileNotFoundError(f"No checkpoints in {ROOT / 'ckpt'}")
        path = ROOT / "ckpt" / f"{saved[-1]:07d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_model(data, args, micro=False):
    path = checkpoint(args.ckpt, micro)
    state = read_ready(path, lambda p: torch.load(p, map_location="cpu", weights_only=True))
    model = Model(len(data.chars))
    model.load_state_dict(state)
    model.to(args.device).eval()
    print(f"checkpoint {path.name}, {args.device}", flush=True)
    return model, path


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def nearest(scores, data, k=12):
    order = np.argsort(-scores, kind="stable")
    return [{"char": data.chars[i], "cos": float(scores[i])}
            for i in order[:k] if np.isfinite(scores[i])]


def neighbours(data, args):
    steps = snapshots()
    wanted = [0, 100, 500, 2000, 10000, steps[-1]]
    chosen = sorted({min(steps, key=lambda s: abs(s - t)) for t in wanted})
    missing = [c for c in NEIGHBOURS if c not in data.ids]
    frames = []
    for step in chosen:
        e = unit(embeddings(step))
        result = {}
        for c in NEIGHBOURS:
            if c not in data.ids:
                continue
            scores = e @ e[data.ids[c]]
            scores[[0, data.ids[c]]] = -np.inf
            result[c] = nearest(scores, data)
        frames.append(dict(step=step, neighbours=result))
        examples = "  ".join(c + ":" + "".join(x["char"] for x in result.get(c, [])[:6])
                             for c in ("山", "月", "愁"))
        print(f"{step:7d}  {len(result)} seeds  {examples}", flush=True)
    save_json("neighbours.json", dict(requested_steps=wanted, missing=missing, frames=frames))


def pairs(data, args):
    step = snapshots()[-1]
    e = embeddings(step)
    norm = unit(e)
    valid = [p for p in PAIRS if all(c in data.ids for c in p)]
    missing = [p for p in PAIRS if p not in valid]
    differences = np.array([e[data.ids[p[1]]] - e[data.ids[p[0]]] for p in valid])
    cosines = unit(differences) @ unit(differences).T
    jobs = [(i, j) for i in range(len(valid)) for j in range(len(valid)) if i != j]
    rows = []
    for start in range(0, len(jobs), 16):
        batch = jobs[start:start + 16]
        queries = np.array([differences[i] + e[data.ids[valid[j][0]]] for i, j in batch])
        scores = unit(queries) @ norm.T
        for values, (i, j) in zip(scores, batch):
            a, b = valid[i]
            c, d = valid[j]
            excluded = {0, data.ids[a], data.ids[b], data.ids[c]}
            values[list(excluded)] = -np.inf
            order = np.argsort(-values, kind="stable")
            rank = None if data.ids[d] in excluded else int(np.flatnonzero(order == data.ids[d])[0]) + 1
            rows.append(dict(source=valid[i], target=valid[j], cos=float(cosines[i, j]),
                             rank=rank, target_excluded=data.ids[d] in excluded,
                             nearest=nearest(values, data)))

    def show(r):
        a, b = r["source"]
        c, d = r["target"]
        rank = r["rank"] if r["rank"] is not None else "excluded"
        near = "".join(x["char"] for x in r["nearest"][:5])
        print(f"{a}:{b}::{c}:{d}  cos={r['cos']:+.5f}  rank={rank}  nearest={near}")

    print(f"step {step}: highest 25 difference cosines (one direction per unordered pair)")
    indices = {p: i for i, p in enumerate(valid)}
    unique = [r for r in rows if indices[r["source"]] < indices[r["target"]]]
    for r in sorted(unique, key=lambda r: -r["cos"])[:25]:
        show(r)
    hits = [r for r in rows if r["rank"] is not None and r["rank"] <= 3]
    print(f"All directed analogies with target rank <= 3: {len(hits)}")
    for r in sorted(hits, key=lambda r: (r["rank"], -r["cos"])):
        show(r)
    save_json("pairs.json", dict(step=step, pairs=valid, missing=missing,
                                rank_base=1, rows=rows,
                                method="Ordered distinct pairs; cosine of raw differences; query=b-a+c. "
                                       "Exclude rare token, a,b,c. A removed target has rank=null."))


@torch.no_grad()
def affinities(e, device, perplexity=30):
    x = F.normalize(torch.as_tensor(e, device=device), dim=1)
    distance = (1 - x @ x.T).clamp_min_(0)
    n = len(x)
    diagonal = torch.eye(n, dtype=torch.bool, device=device)
    # Shifting each row leaves its conditional probabilities unchanged and
    # avoids cancellation in the entropy calculation at large precision.
    floor = distance.masked_fill(diagonal, float("inf")).min(dim=1, keepdim=True).values
    distance = distance - floor
    distance.masked_fill_(diagonal, 0)
    beta = torch.ones((n, 1), device=device)
    low = torch.zeros_like(beta)
    high = torch.full_like(beta, float("inf"))
    for _ in range(60):
        logits = (-distance * beta).masked_fill(diagonal, float("-inf"))
        conditional = logits.softmax(dim=1)
        entropy = torch.logsumexp(logits, dim=1, keepdim=True) + beta * (conditional * distance).sum(dim=1, keepdim=True)
        broad = entropy > math.log(perplexity)
        low = torch.where(broad, beta, low)
        high = torch.where(broad, high, beta)
        beta = torch.where(torch.isinf(high), beta * 2, (low + high) / 2)
    conditional = (-distance * beta).masked_fill(diagonal, float("-inf")).softmax(dim=1)
    p = (conditional + conditional.T) / (2 * n)
    return p / p.sum()


def pca_start(e):
    x = unit(e)
    x = x - x.mean(axis=0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    y = u[:, :2] * s[:2]
    return (y * (1e-4 / max(float(y[:, 0].std()), 1e-12))).astype(np.float32)


@torch.no_grad()
def tsne(p, initial, iterations, exaggerate, factor=12):
    y = torch.as_tensor(initial, device=p.device).clone()
    n = len(y)
    velocity = torch.zeros_like(y)
    gains = torch.ones_like(y)
    lr = max(50.0, n / 48.0)

    def student():
        squared = (y * y).sum(dim=1, keepdim=True)
        distance = (squared + squared.T - 2 * (y @ y.T)).clamp_min_(0)
        weights = (1 + distance).reciprocal()
        weights.fill_diagonal_(0)
        return weights, weights / weights.sum()

    for i in range(iterations):
        weights, q = student()
        early = exaggerate and i < min(250, iterations // 3)
        a = ((factor if early else 1) * p - q) * weights
        gradient = 4 * (a.sum(dim=1, keepdim=True) * y - a @ y)
        gains = torch.where(gradient.sign() != velocity.sign(), gains + 0.2, gains * 0.8).clamp_min_(0.01)
        velocity = (0.5 if early else 0.8) * velocity - lr * gains * gradient
        y += velocity
        y -= y.mean(dim=0, keepdim=True)
    _, q = student()
    kl = (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum()
    return y.cpu().numpy(), float(kl)


def map_steps(steps, count=16):
    if len(steps) <= count:
        return steps
    positions = np.log1p(np.asarray(steps, dtype=np.float64))
    targets = np.linspace(positions[0], positions[-1], count)
    chosen = {0, len(steps) - 1}
    for t in targets[1:-1]:
        chosen.add(min((i for i in range(len(steps)) if i not in chosen),
                       key=lambda i: abs(positions[i] - t)))
    return [steps[i] for i in sorted(chosen)]


def kmeans(x, chars, k=40):
    rng = np.random.default_rng(7)
    k = min(k, len(x))
    centers = [x[rng.integers(len(x))].copy()]
    distance = np.full(len(x), np.inf)
    for _ in range(1, k):
        distance = np.minimum(distance, ((x - centers[-1]) ** 2).sum(axis=1))
        index = rng.choice(len(x), p=distance / distance.sum())
        centers.append(x[index].copy())
    centers = np.array(centers)
    previous = None
    for _ in range(100):
        distance = ((x[:, None] - centers[None]) ** 2).sum(axis=2)
        labels = distance.argmin(axis=1)
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        residual = distance.min(axis=1)
        for j in range(k):
            members = x[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
            else:
                index = residual.argmax()
                centers[j] = x[index]
                residual[index] = -1
    labels = ((x[:, None] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
    groups = []
    for j in range(k):
        ids = np.flatnonzero(labels == j)
        if not len(ids):
            continue
        center = x[ids].mean(axis=0)
        squared = ((x[ids] - center) ** 2).sum(axis=1)
        order = ids[np.argsort(squared)]
        groups.append(dict(cluster=j, size=len(ids), rms_radius=float(np.sqrt(squared.mean())),
                           center=center.tolist(), chars=[chars[i] for i in order]))
    return sorted(groups, key=lambda g: g["rms_radius"])


def map_data(data, args):
    ids = sorted((i for i, c in enumerate(data.chars) if i and c.isalpha()),
                 key=lambda i: -data.count[data.chars[i]])[:1500]
    chars = [data.chars[i] for i in ids]
    if len(ids) <= 30:
        raise ValueError("Perplexity 30 requires more than 30 characters")
    steps = map_steps(snapshots())
    frames, costs = {}, {}
    following = None
    # Forward by default: each layout grows out of the one before it, so what gathers on the page gathers as training goes.
    for step in (steps if args.forward else reversed(steps)):
        start = time.perf_counter()
        e = embeddings(step)[ids]
        p = affinities(e, args.device, args.perplexity)
        first = following is None
        initial = pca_start(e) if first else following
        last = step == (steps[-1] if args.forward else steps[0])
        iterations = args.iterations if first or (last and args.forward) else args.warm_iterations
        # going forward every layout has to settle into its own neighbours, or nothing gathers until the last one
        y, kl = tsne(p, initial, iterations, exaggerate=first or args.forward, factor=12 if first else 4)
        if following is not None:
            # Rotation/reflection is arbitrary in t-SNE; remove it between frames.
            u, _, vt = np.linalg.svd(y.T @ following)
            y = y @ (u @ vt)
        y -= y.mean(axis=0)
        frames[step], costs[step] = y, kl
        following = y
        print(f"map {step:7d}  {iterations:4d} iterations  {time.perf_counter() - start:.1f}s  KL={kl:.6f}", flush=True)
        del p
    xy = np.stack([frames[s] for s in steps]).astype(np.float32)
    xy -= xy.mean(axis=1, keepdims=True)
    # A single scale preserves expansion/contraction across training time.
    scale = float(np.sqrt((xy * xy).sum(axis=2).mean()))
    xy /= max(scale, 1e-12)
    DEST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEST / f"{args.out}.npz", steps=np.array(steps, dtype=np.int64),
                        chars=np.array(json.dumps(chars, ensure_ascii=False)), xy=xy,
                        kl=np.array([costs[s] for s in steps]), global_scale=np.array(scale))
    print(f"{DEST / (args.out + '.npz')}: {len(steps)} frames × {len(chars)} characters; global RMS radius=1")
    groups = kmeans(xy[-1], chars)
    for g in groups:
        print(f"cluster {g['cluster']:2d}  n={g['size']:3d}  rms={g['rms_radius']:.4f}  {''.join(g['chars'])}")
    save_json(f"{args.out}_clusters.json", dict(step=steps[-1], k=40, groups=groups,
                                       method="k-means++ on final 2D layout; ascending RMS radius, not semantic labels"))


@torch.no_grad()
def attn(data, args):
    model, path = load_model(data, args)
    x = torch.tensor([data.encode(PROMPT)], device=args.device)
    model(x, keep=True)
    w = np.stack([b.seen[0].numpy() for b in model.modules() if isinstance(b, Block)]).astype(np.float32)
    DEST.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DEST / "attn.npz", chars=np.array(list(PROMPT)), w=w, ckpt=np.array(path.name))
    print(DEST / "attn.npz")
    positions = [len(PROMPT) - 1] + [i for i, c in enumerate(PROMPT) if c in "光霜月"]
    summaries = []
    for layer in range(len(w)):
        for head in [None] + list(range(w.shape[1])):
            weights = w[layer].mean(axis=0) if head is None else w[layer, head]
            label = "mean" if head is None else str(head)
            for pos in positions:
                order = np.argsort(-weights[pos, :pos + 1], kind="stable")[:5]
                top = [dict(pos=int(j), char=PROMPT[j], weight=float(weights[pos, j])) for j in order]
                summaries.append(dict(layer=layer, head=head, from_pos=pos, from_char=PROMPT[pos], top=top))
                text = " ".join(f"{r['char']}[{r['pos']}]={r['weight']:.4f}" for r in top)
                print(f"L{layer} H{label} {PROMPT[pos]}[{pos}] -> {text}")
    candidates = []
    for layer in range(len(w)):
        for head in range(w.shape[1]):
            for source in range(len(PROMPT)):
                for target in range(source - 3):
                    candidates.append(dict(layer=layer, head=head, from_pos=source, from_char=PROMPT[source],
                                           to_pos=target, to_char=PROMPT[target], distance=source - target,
                                           weight=float(w[layer, head, source, target])))
    candidates.sort(key=lambda r: -r["weight"])
    save_json("attn_top.json", dict(ckpt=path.name, prompt=PROMPT, index_base=0, summaries=summaries,
                                   candidates=candidates[:30],
                                   method="Largest attention weights at distance >= 4; not a semantic score"))


def guess(data, args):
    rows = logs()
    top5 = [r["step"] for r in rows if any(c == "故" for c, _ in r["top"][:5])]
    first = [r["step"] for r in rows if r["top"] and r["top"][0][0] == "故"]
    wrong = [r["step"] for r in rows if r["top"] and r["top"][0][0] != "故"]
    markers = dict(first_gu_top5=min(top5) if top5 else None,
                   first_gu_top1=min(first) if first else None,
                   last_top1_not_gu=max(wrong) if wrong else None)
    table = [dict(step=r["step"], loss=r["loss"], gu=r["gu"], top5=r["top"][:5],
                  line=r["lines"].get("低头思", "")) for r in rows]
    print("   step     loss       gu   top5 (probability)                                         低头思")
    for r in table:
        text = " ".join(f"{c}:{p:.5f}" for c, p in r["top5"])
        print(f"{r['step']:7d}  {r['loss']:7.4f}  {r['gu']:7.5f}   {text:55s}  {r['line']}")
    print(f"① 故首次 top5: {markers['first_gu_top5']}  ② 故首次 top1: {markers['first_gu_top1']}  "
          f"③ 最后 top1 非故: {markers['last_top1_not_gu']}  (仅限已记录的 step)")
    save_json("guess.json", dict(markers=markers, rows=table, scope="Observed log rows, not unlogged training steps"))


def micro(data, args):
    model, path = load_model(data, args, micro=True)
    x = torch.tensor([data.encode(PROMPT)], device=args.device)
    target = torch.tensor([data.ids["故"]], device=args.device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    before = model.tok.weight.detach().cpu().clone().numpy()

    def measure():
        logits = model(x)[:, -1]
        loss = F.cross_entropy(logits, target)
        p = logits.detach()[0].softmax(dim=-1).cpu()
        values, indices = p.topk(min(8, len(data.chars)))
        top = [dict(char=data.chars[i], probability=float(v)) for v, i in zip(values.tolist(), indices.tolist())]
        return loss, dict(loss=float(loss.detach()), gu=float(p[data.ids["故"]]), top8=top)

    loss, record = measure()
    records = [dict(update=0, **record)]
    vectors, biggest = {}, []
    for update in range(1, 13):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if update == 1:
            after = model.tok.weight.detach().cpu().numpy().copy()
            changes = np.linalg.norm(after - before, axis=1)
            for c in "思头月故":
                i = data.ids[c]
                vectors[c] = dict(before=[round(float(v), 4) for v in before[i]],
                                  after=[round(float(v), 4) for v in after[i]], delta_norm=float(changes[i]))
            biggest = [dict(char=data.chars[i], id=int(i), delta_norm=float(changes[i]))
                       for i in np.argsort(-changes, kind="stable")[:20]]
        loss, record = measure()
        records.append(dict(update=update, **record))
    for r in records:
        text = " ".join(f"{v['char']}:{v['probability']:.5f}" for v in r["top8"])
        print(f"SGD {r['update']:2d}  loss={r['loss']:.6f}  {text}")
    count = sum(p.numel() for p in model.parameters())
    print(f"{count:,} unique parameters (tied embedding/head counted once); first-step embedding changes:")
    print(" ".join(f"{r['char']!r}:{r['delta_norm']:.6g}" for r in biggest))
    save_json("micro.json", dict(ckpt=path.name, prompt=PROMPT, target="故", lr=args.lr,
                                optimizer="SGD, no momentum or weight decay; all parameters; last-token loss only",
                                parameter_count=count, embed_dim=EMBED, records=records,
                                first_step=dict(vectors=vectors, largest_embedding_changes=biggest)))


def first_line(text):
    start = 0
    for i, c in enumerate(text):
        if c in "，。":
            part = text[start:i]
            if len(part) in (5, 7) and all(ch.isalpha() for ch in part):
                return part
            start = i + 1
    return None


@torch.no_grad()
def me(data, args):
    """What came after 我, at every checkpoint kept, with the same dice each time so that only the reading differs."""
    log = logs()
    rows = []
    for path in [ROOT / "ckpt" / f"{step:07d}.pt" for step in numbered("ckpt", ".pt")]:
        state = read_ready(path, lambda p: torch.load(p, map_location="cpu", weights_only=True))
        model = Model(len(data.chars))
        model.load_state_dict(state)
        model.to(args.device).eval()
        step = int(path.stem)
        loss = min(log, key=lambda r: abs(r["step"] - step))["loss"]
        texts = []
        for dice in range(24):
            gen = torch.Generator(device="cpu").manual_seed(dice)
            ids = data.encode("我")
            for _ in range(40):
                logits = model(torch.tensor([ids[-BLOCK:]], device=args.device))[0, -1].cpu()
                p = (logits / 0.8).softmax(dim=-1)
                p[0] = 0
                ids.append(int(torch.multinomial(p, 1, generator=gen)))
            texts.append("".join(data.chars[i] for i in ids))
        rows.append(dict(step=step, loss=loss, texts=texts))
        print(step, loss, texts[0], flush=True)
    save_json("me.json", dict(seed="我", temperature=0.8, rows=rows))


@torch.no_grad()
def lines(data, args):
    model, path = load_model(data, args)
    tang = (ROOT / "tang.txt").read_text(encoding="utf-8")
    tang300 = (ROOT / "tang300.txt").read_text(encoding="utf-8")
    gen = torch.Generator(device="cpu").manual_seed(11)
    rows = []
    for seed in "春秋月夜风人山我":
        for start in range(0, 40, 4):
            ids = torch.tensor([data.encode(seed)] * 4)
            texts = [seed] * 4
            found = [None] * 4
            for _ in range(64):
                # CPU sampling works with the same generator on both model devices.
                logits = model(ids[:, -BLOCK:].to(args.device))[:, -1].cpu()
                p = (logits / 0.8).softmax(dim=-1)
                p[:, 0] = 0
                next_ids = torch.multinomial(p, 1, generator=gen)
                ids = torch.cat((ids, next_ids), dim=1)
                for j, index in enumerate(next_ids[:, 0].tolist()):
                    if found[j] is None:
                        texts[j] += data.chars[index]
                        found[j] = first_line(texts[j])
                if all(s is not None for s in found):
                    break
            for j, sentence in enumerate(found):
                in_tang = sentence in tang if sentence is not None else None
                in_300 = sentence in tang300 if sentence is not None else None
                new = not (in_tang or in_300) if sentence is not None else None
                r = dict(seed=seed, sample=start + j, text=texts[j], sentence=sentence,
                         in_tang=in_tang, in_tang300=in_300, is_new=new)
                rows.append(r)
                if new:
                    print(f"{seed} {start + j + 1:2d}/40  {sentence}", flush=True)
    found_count = sum(r["sentence"] is not None for r in rows)
    novel = [r["sentence"] for r in rows if r["is_new"]]
    print(f"{len(rows)} samples, {found_count} complete 5/7-character clauses; "
          f"{len(novel)} absent from both corpora ({len(set(novel))} unique).")
    save_json("lines.json", dict(ckpt=path.name, random_seed=11, device=args.device, temperature=0.8,
                                batch_size=4, max_new_tokens=64, rows=rows,
                                match="Exact substring of the whole clause, without delimiter; no normalization"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["neighbours", "pairs", "map", "attn", "guess", "micro", "lines", "me", "all"])
    parser.add_argument("--ckpt", type=int, help="exact numbered checkpoint (attn, micro, lines, all)")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--lr", type=float, default=0.02, help="micro SGD learning rate")
    parser.add_argument("--iterations", type=int, default=1000, help="map iterations at latest snapshot")
    parser.add_argument("--warm-iterations", type=int, default=150, help="map iterations at each earlier snapshot")
    parser.add_argument("--perplexity", type=float, default=30, help="map perplexity: lower gives tighter islands")
    parser.add_argument("--forward", action="store_true", help="grow the layouts forward from the first snapshot")
    parser.add_argument("--out", default="map", help="map files are DEST/OUT.npz and DEST/OUT_clusters.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    data = Vocabulary()
    commands = dict(neighbours=neighbours, pairs=pairs, map=map_data, attn=attn,
                    guess=guess, micro=micro, lines=lines, me=me)
    if args.command == "all":
        for name, command in commands.items():
            if name != "map":
                print(f"\n=== {name} ===", flush=True)
                command(data, args)
    else:
        commands[args.command](data, args)


if __name__ == "__main__":
    main()
