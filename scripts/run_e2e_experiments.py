
"""LOOCV for end-to-end trained models.

Usage:
  python run_e2e_experiments.py
  python run_e2e_experiments.py --archs hyper-sym,relationgcn --seeds 1,2,3
  python run_e2e_experiments.py --e2e-init scratch --e2e-epochs 12 --seeds 1
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import os
os.chdir(PROJECT_ROOT)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dynamic_siamese import build_e2e_model, E2E_ARCHS
from dynamic_siamese.datasets_multi import GenericMLDataset, CombinedBalancedDataset
from dynamic_siamese.datasets import MPerClassBatchSampler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--datasets", default="cub,cars,dogs,flowers,nabirds,aircraft")
    p.add_argument("--archs", default=",".join(E2E_ARCHS))
    p.add_argument("--seeds", default="1")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--M-per-class", type=int, default=10)
    p.add_argument("--classes-per-dataset", type=int, default=50)
    p.add_argument("--samples-per-class", type=int, default=26)
    p.add_argument("--e2e-epochs", type=int, default=8)
    p.add_argument("--e2e-lr", type=float, default=3e-4)
    p.add_argument("--e2e-batch", type=int, default=32)
    p.add_argument("--e2e-init", choices=["baseline_loo", "scratch"],
                   default="baseline_loo")
    p.add_argument("--output", default="results/e2e_loocv.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def eval_tf():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_pairs(labels):
    B = len(labels)
    lab = labels.detach().cpu()
    pos = torch.zeros(B, dtype=torch.long)
    neg = torch.zeros(B, dtype=torch.long)
    for i in range(B):
        same = (lab == lab[i]).nonzero(as_tuple=True)[0]
        same = same[same != i]
        pos[i] = same[torch.randint(len(same), (1,)).item()].item() if len(same) > 0 else i
        diff = (lab != lab[i]).nonzero(as_tuple=True)[0]
        neg[i] = diff[torch.randint(len(diff), (1,)).item()].item() if len(diff) > 0 else i
    return pos, neg


def train_e2e(model, loader, args, device):
    params_enc = []
    params_ad = []
    enc_prefixes = ("tower", "tower_a", "tower_b", "stem", "layer1",
                    "layer2", "layer3", "layer4", "pool", "emb_head")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(n.startswith(pref) for pref in enc_prefixes):
            params_enc.append(p)
        else:
            params_ad.append(p)
    groups = [{"params": params_enc, "lr": args.e2e_lr * 0.3},
              {"params": params_ad, "lr": args.e2e_lr}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    model.train()
    for epoch in range(1, args.e2e_epochs + 1):
        tot, n = 0.0, 0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            pi, ni = build_pairs(labels)
            pi = pi.to(device); ni = ni.to(device)
            sp = model(imgs, imgs[pi]) * 5.0
            sn = model(imgs, imgs[ni]) * 5.0
            scores = torch.cat([sp, sn])
            tgt = torch.cat([torch.ones_like(sp), torch.zeros_like(sn)])
            loss = F.binary_cross_entropy_with_logits(scores, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"    e2e epoch {epoch:2d} | loss={tot/max(n,1):.4f}", flush=True)
    return model


@torch.no_grad()
def eval_e2e_pair(model, dataset, classes, K, M, seed, device):
    import itertools
    rng = random.Random(seed)
    class_to_idx = {}
    for i in range(len(dataset)):
        _, y = dataset[i]
        class_to_idx.setdefault(y, []).append(i)
    cls_list = sorted(class_to_idx.keys())
    rng.shuffle(cls_list)
    cls_list = sorted(cls_list[:classes])
    queries = {}
    for c in cls_list:
        idxs = class_to_idx[c][:]
        rng.shuffle(idxs)
        if len(idxs) < K + M:
            continue
        queries[c] = [dataset[i][0] for i in idxs[K:K+M]]
    pos, neg = [], []
    for c in queries:
        q = queries[c]
        if len(q) < 2:
            continue
        ps = list(itertools.combinations(range(len(q)), 2))
        rng.shuffle(ps)
        for i, j in ps[:5]:
            pos.append((q[i], q[j]))
        others = [x for x in queries if x != c]
        for _ in range(5):
            c2 = rng.choice(others)
            neg.append((q[rng.randrange(len(q))],
                        queries[c2][rng.randrange(len(queries[c2]))]))
    if not pos or not neg:
        return float("nan")
    model.eval()
    def sim(pairs):
        out = []
        for i in range(0, len(pairs), 32):
            b = pairs[i:i+32]
            x1 = torch.stack([p[0] for p in b]).to(device)
            x2 = torch.stack([p[1] for p in b]).to(device)
            out.append(model(x1, x2).cpu())
        return torch.cat(out)
    sp, sn = sim(pos), sim(neg)
    return (sp.unsqueeze(1) > sn.unsqueeze(0)).float().mean().item()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print("Device:", device)
    print("E2E init:", args.e2e_init, "| epochs:", args.e2e_epochs,
          "| lr:", args.e2e_lr)
    print("Archs:", args.archs)
    print("Seeds:", args.seeds)
    print()

    datasets = [x.strip() for x in args.datasets.split(",")]
    archs = [x.strip() for x in args.archs.split(",")]
    seeds = [int(x.strip()) for x in args.seeds.split(",")]

    results = []
    for holdout in datasets:
        print("=" * 60)
        print(f"FOLD: holdout={holdout}")
        print("=" * 60)

        sources = []
        for name in datasets:
            if name == holdout:
                continue
            try:
                ds = GenericMLDataset(
                    args.data_root, name, split="train", transform=eval_tf(),
                    max_classes=args.classes_per_dataset,
                    max_per_class=args.samples_per_class)
                sources.append(ds)
            except FileNotFoundError:
                pass
        if not sources:
            print("  no sources, skip")
            continue

        combined = CombinedBalancedDataset(sources)
        print(f"  source: {len(combined)} samples, "
              f"{len(set(combined.targets))} classes")

        sampler = MPerClassBatchSampler(
            combined.targets, m=4, batch_size=args.e2e_batch, seed=42)
        loader = DataLoader(combined, batch_sampler=sampler,
                            num_workers=args.num_workers)

        try:
            target = GenericMLDataset(
                args.data_root, holdout, split="test",
                transform=eval_tf(), max_classes=None, max_per_class=None)
        except FileNotFoundError as e:
            print("  target not found:", e); continue
        print(f"  target: {len(target)} samples")

        ckpt_path = Path(f"baseline_loo_{holdout}.pt")
        ckpt = None
        if args.e2e_init == "baseline_loo" and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            print(f"  init encoder from: {ckpt_path.name}")
        else:
            print(f"  init encoder from scratch (pretrained ImageNet)")

        for arch in archs:
            if arch not in E2E_ARCHS:
                print(f"  skip {arch} (not in E2E_ARCHS)"); continue
            for seed in seeds:
                random.seed(seed); torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                print(f"  [{arch} seed={seed}] building...")
                model = build_e2e_model(
                    arch, pretrained=(ckpt is None), device=device)
                if ckpt is not None:
                    from dynamic_siamese.e2e_models import load_baseline_encoder_into
                    model = load_baseline_encoder_into(model, ckpt, arch)
                t0 = time.time()
                model = train_e2e(model, loader, args, device)
                train_s = time.time() - t0
                acc = eval_e2e_pair(
                    model, target, args.classes_per_dataset,
                    args.K, args.M_per_class, seed=seed, device=device)
                print(f"    {arch} seed={seed} after={acc:.4f} "
                      f"train_time={train_s:.1f}s", flush=True)
                results.append({"holdout": holdout, "arch": arch,
                                "seed": seed, "acc": acc,
                                "train_time_s": train_s})
        print()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["holdout", "arch", "seed",
                                            "acc", "train_time_s"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"Saved: {out}")

    print()
    print("=" * 60)
    print("SUMMARY (acc mean across seeds)")
    print("=" * 60)
    for arch in archs:
        vals = [r["acc"] for r in results if r["arch"] == arch
                and not (isinstance(r["acc"], float) and r["acc"] != r["acc"])]
        if not vals:
            continue
        m = mean(vals); s = stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {arch:<15} {m:.4f} ± {s:.4f} (n={len(vals)})")


if __name__ == "__main__":
    main()
