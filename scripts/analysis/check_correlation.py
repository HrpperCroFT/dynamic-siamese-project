# check_correlation.py
"""Correlation between hyper and nonspatial prediction scores.

If Pearson/Spearman correlation on query pair sims is low (<0.7),
the two mechanisms disagree → ensemble/hybrid may improve.
If high (>0.9), they largely agree → hybrid unlikely to help.
"""
import argparse
import random as _r
from itertools import combinations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from dynamic_siamese import (SiameseBaseline, DynamicSiameseNet,
                              CUB200MetricLearning, Cars196MetricLearning)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cub", choices=["cub", "cars"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--baseline-ckpt", default="")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--M-per-class", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-classes", type=int, default=0)
    p.add_argument("--adapt-epochs", type=int, default=5)
    p.add_argument("--lr-side", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def sample(ds, K, M, seed, max_classes=None):
    by_class = {}
    for i in range(len(ds)):
        _, y = ds[i]
        by_class.setdefault(y, []).append(i)
    classes = sorted(by_class.keys())
    if max_classes is not None and 0 < max_classes < len(classes):
        sel_rng = _r.Random(42)
        sel_rng.shuffle(classes)
        classes = sorted(classes[:max_classes])
    rng = _r.Random(seed)
    sup, qry = {}, {}
    for c in classes:
        idxs = by_class[c]
        if len(idxs) < K + M:
            continue
        idxs = idxs[:]; rng.shuffle(idxs)
        sup[c] = [ds[i][0] for i in idxs[:K]]
        qry[c] = [ds[i][0] for i in idxs[K:K+M]]
    return sup, qry


def make_pairs(qry, seed=0, ppc=5):
    rng = _r.Random(seed)
    classes = sorted(qry.keys())
    pos, neg = [], []
    for c in classes:
        q = qry[c]
        if len(q) < 2: continue
        ps = list(combinations(range(len(q)), 2)); rng.shuffle(ps)
        for i, j in ps[:ppc]:
            pos.append((q[i], q[j]))
        others = [x for x in classes if x != c]
        for _ in range(ppc):
            c2 = rng.choice(others)
            i = rng.randrange(len(q))
            j = rng.randrange(len(qry[c2]))
            neg.append((q[i], qry[c2][j]))
    return pos, neg


@torch.no_grad()
def get_sims(model, pairs, device):
    out = []
    for i in range(0, len(pairs), 32):
        b = pairs[i:i+32]
        x1 = torch.stack([a for a, _ in b]).to(device)
        x2 = torch.stack([b for _, b in b]).to(device)
        out.append(model(x1, x2).cpu())
    return torch.cat(out)


def adapt(model, sup, args):
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "side_block"):
        for p in model.side_block.parameters():
            p.requires_grad = True
        params = [p for p in model.side_block.parameters() if p.requires_grad]
        model.side_block.train()
    else:
        params = list(model.adapt_params())
        for p in params:
            p.requires_grad = True
        if hasattr(model, "head"):
            model.head.train(); model.kernel_gen.train()
    opt = torch.optim.AdamW(params, lr=args.lr_side)

    classes = list(sup.keys())
    import itertools
    pos_p, neg_p = [], []
    for c in classes:
        for i, j in itertools.combinations(range(len(sup[c])), 2):
            pos_p.append((sup[c][i], sup[c][j]))
    for _ in range(max(len(pos_p), 200)):
        c1, c2 = _r.Random(0).sample(classes, 2) if False else (classes[0], classes[1])
        # just random
        import random as _rr
        c1, c2 = _rr.sample(classes, 2)
        i = _rr.randrange(len(sup[c1])); j = _rr.randrange(len(sup[c2]))
        neg_p.append((sup[c1][i], sup[c2][j]))

    pairs = ([(a, b, 1.0) for a, b in pos_p]
             + [(a, b, 0.0) for a, b in neg_p])
    rng = _r.Random(0)
    for ep in range(1, args.adapt_epochs + 1):
        rng.shuffle(pairs)
        model.eval()
        if hasattr(model, "side_block"): model.side_block.train()
        if hasattr(model, "head"): model.head.train(); model.kernel_gen.train()
        for i in range(0, len(pairs), 32):
            b = pairs[i:i+32]
            x1 = torch.stack([p[0] for p in b]).to(args.device)
            x2 = torch.stack([p[1] for p in b]).to(args.device)
            t = torch.tensor([p[2] for p in b], dtype=torch.float32,
                             device=args.device)
            sim = model(x1, x2) * 5.0
            loss = F.binary_cross_entropy_with_logits(sim, t)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def main():
    args = parse_args()
    if not args.baseline_ckpt:
        args.baseline_ckpt = f"baseline_{args.dataset}.pt"
    device = torch.device(args.device)

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    DS = CUB200MetricLearning if args.dataset == "cub" else Cars196MetricLearning
    test = DS(args.data_root, split="test", transform=eval_tf)
    print(f"target: {len(test)} samples")

    ckpt = torch.load(args.baseline_ckpt, weights_only=False, map_location="cpu")
    sup, qry = sample(test, args.K, args.M_per_class, seed=args.seed,
                       max_classes=args.eval_classes or None)
    print(f"classes: {len(sup)}, K={args.K}, M={args.M_per_class}")

    pos, neg = make_pairs(qry, seed=0)

    # Baseline
    base = SiameseBaseline(embedding_dim=128, pretrained=False)
    base.tower.load_state_dict(ckpt["tower"])
    base.emb_head.load_state_dict(ckpt["head"])
    base.to(device).eval()
    base_pos = get_sims(base, pos, device)
    base_neg = get_sims(base, neg, device)

    # Nonspatial
    ns = DynamicSiameseNet(embedding_dim=128, generate_kernels=True,
                           pretrained=False, share_backbone=True, spatial=False)
    ns.tower_a.load_state_dict(ckpt["tower"])
    ns.emb_head_a.load_state_dict(ckpt["head"])
    if ns.emb_head_b is not ns.emb_head_a:
        ns.emb_head_b.load_state_dict(ckpt["head"])
    ns.to(device)
    ns = adapt(ns, sup, args)
    ns_pos = get_sims(ns, pos, device)
    ns_neg = get_sims(ns, neg, device)

    # Hyper
    from dynamic_siamese import HyperComparisonNet
    hy = HyperComparisonNet(n_kernels=2, feature_level="l4",
                             kernel_gen_mode="mlp", out_ch_per_kernel=16,
                             use_cosine_residual=True, share_backbone=True,
                             pretrained=False)
    hy.tower_a.load_state_dict(ckpt["tower"])
    hy.emb_head.load_state_dict(ckpt["head"])
    hy.to(device)
    hy = adapt(hy, sup, args)
    hy_pos = get_sims(hy, pos, device)
    hy_neg = get_sims(hy, neg, device)

    def corr(a, b, label):
        a = a - a.mean(); b = b - b.mean()
        denom = (a.norm() * b.norm()).item()
        if denom < 1e-8: return 0.0
        return (a * b).sum().item() / denom

    # Pairwise correlations
    print()
    print("=" * 55)
    print("Correlation of sim scores (Pearson)")
    print("=" * 55)

    for label, hp, np_, bp in [("pos", hy_pos, ns_pos, base_pos),
                                ("neg", hy_neg, ns_neg, base_neg)]:
        print(f"  {label}:  hyper~nonspatial  {corr(hp, np_, label):+.3f}   "
              f"hyper~baseline  {corr(hp, bp, label):+.3f}   "
              f"nonspatial~baseline  {corr(np_, bp, label):+.3f}")

    # Ranking correlation: for each pos/neg pair, compute margin = sim - mean(neg)
    # Approximate Spearman via rank-of-diffs
    def rank_sim(sims):
        return sims.argsort().argsort().float()

    print()
    print("Rank-based (approximate Spearman) on pos+neg combined:")
    all_hy = torch.cat([hy_pos, hy_neg])
    all_ns = torch.cat([ns_pos, ns_neg])
    all_ba = torch.cat([base_pos, base_neg])
    rh = rank_sim(all_hy); rn = rank_sim(all_ns); rb = rank_sim(all_ba)
    print(f"  hyper~nonspatial  {corr(rh, rn, ''):+.3f}")
    print(f"  hyper~baseline    {corr(rh, rb, ''):+.3f}")
    print(f"  nonspatial~baseline {corr(rn, rb, ''):+.3f}")

    # Pair_accuracy of each and of simple averaging
    def pair_acc(sp, sn):
        return (sp.unsqueeze(1) > sn.unsqueeze(0)).float().mean().item()

    print()
    print("=" * 55)
    print("Pair_accuracy")
    print("=" * 55)
    print(f"  baseline:    {pair_acc(base_pos, base_neg):.4f}")
    print(f"  nonspatial:  {pair_acc(ns_pos, ns_neg):.4f}")
    print(f"  hyper:       {pair_acc(hy_pos, hy_neg):.4f}")
    # Simple average of logits (not best but simple)
    avg_pos = (hy_pos + ns_pos) / 2
    avg_neg = (hy_neg + ns_neg) / 2
    print(f"  avg(hyper,nonspatial): {pair_acc(avg_pos, avg_neg):.4f}")


if __name__ == "__main__":
    main()
