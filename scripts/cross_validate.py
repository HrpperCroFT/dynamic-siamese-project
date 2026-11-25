
"""Leave-one-dataset-out cross-validation.

For each held-out dataset D in --datasets:
    1. Train a baseline encoder on all OTHER datasets (balanced).
    2. Report baseline pair_acc on D's novel classes (cross-domain).
    3. Adapt hyper head on D's supports, eval on D's queries.
    4. Compare to in-domain baseline (from baseline_<D>.pt if it exists).

Balancing: each source dataset contributes exactly
--classes-per-dataset classes and --samples-per-class samples/class.

Multi-seed: baseline trained once (seed=42), adaptation evaluated on
each seed in --seeds. Reported mean ± std covers support/query sampling
variance.

Run:
  python scripts/cross_validate.py --datasets cub,cars,dogs,flowers,dtd,omniglot \\
      --classes-per-dataset 30 --samples-per-class 20 --seeds 1,2,3
"""
import argparse
import random
import sys
import time
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os
os.chdir(PROJECT_ROOT)

try:
    import dynamic_siamese
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from dynamic_siamese import (
    GenericMLDataset, CombinedBalancedDataset, SiameseBaseline,
    HyperComparisonNet, DynamicSiameseNet, MPerClassBatchSampler,
    HyperHybridNet,
    HyperMultiNet,
)
from dynamic_siamese.utils import pair_accuracy
from few_shot_eval import finetune_encoder, adapt_side_block
from dynamic_siamese.baselines import BASELINE_ARCHS, DISPATCH as BASELINE_DISPATCH


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--datasets", default="cub,cars,dogs,flowers,dtd,omniglot")
    p.add_argument("--arches", default="hyper,nonspatial",
                   help="comma list of {hyper,hyper-pure,nonspatial}")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--neg-same-domain-ratio", type=float, default=0.5,
                   help="target fraction of same-domain negatives in baseline training")
    p.add_argument("--K-list", default=None,
                   help="comma list of K values, e.g. 5,10,20. Overrides --K if set")
    p.add_argument("--eval-classes", type=int, default=0,
                   help="0 = auto = match source budget (eff_cls)")
    p.add_argument("--M-per-class", type=int, default=0,
                   help="query images per class; 0 = auto = eff_per - K")
    p.add_argument("--classes-per-dataset", type=int, default=None,
                   help="max classes per SOURCE dataset during baseline training")
    p.add_argument("--samples-per-class", type=int, default=None,
                   help="max samples per class per SOURCE dataset")
    p.add_argument("--baseline-epochs", type=int, default=12)
    p.add_argument("--baseline-lr", type=float, default=1e-4)
    p.add_argument("--adapt-epochs", type=int, default=5)
    p.add_argument("--lr-side", type=float, default=3e-4)
    p.add_argument("--feature-level", default="l4")
    p.add_argument("--n-kernels", type=int, default=2)
    p.add_argument("--levels", default="l2,l3")
    p.add_argument("--residual-levels", default="l3")
    p.add_argument("--side-ch", type=int, default=32)
    p.add_argument("--multi-hidden", type=int, default=32)
    p.add_argument("--out-ch", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--baseline-seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--finetune", action="store_true")
    p.add_argument("--ft-lr", type=float, default=1e-5)
    p.add_argument("--ft-epochs", type=int, default=15)
    p.add_argument("--ft-patience", type=int, default=3)
    p.add_argument("--ft-lr-patience", type=int, default=2)
    p.add_argument("--ft-val-frac", type=float, default=0.2)
    p.add_argument("--rn-epochs", type=int, default=30)
    p.add_argument("--rn-lr", type=float, default=1e-3)
    p.add_argument("--rn-hidden", type=int, default=64)
    p.add_argument("--output", default="cross_validation_results.csv")
    p.add_argument("--skip-baseline-cache", action="store_true",
                   help="ignore cached baseline_loo_<D>.pt")
    p.add_argument("--balance-mode", default="global",
                   choices=["global", "per-fold", "off"],
                   help="global = min classes across ALL datasets "
                        "(identical budget every fold); "
                        "per-fold = cap only by availability in fold; "
                        "off = use raw --classes-per-dataset/--samples-per-class")
    return p.parse_args()


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    import numpy as np
    np.random.seed(s)


def eval_tf():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ---------- Baseline training on multiple datasets ----------

def build_pairs(labels, domains=None, same_domain_ratio=0.5,
                seed=None, stats=None):
    import random as _r
    rng = _r.Random(seed) if seed is not None else _r
    B = len(labels)
    lab = labels.detach().cpu()
    pos = torch.zeros(B, dtype=torch.long)
    neg = torch.zeros(B, dtype=torch.long)
    for i in range(B):
        same = (lab == lab[i]).nonzero(as_tuple=True)[0]
        same = same[same != i]
        pos[i] = (same[torch.randint(len(same), (1,)).item()].item()
                  if len(same) > 0 else i)
        if domains is None:
            # fallback: random neg (current implicit behavior)
            diff = (lab != lab[i]).nonzero(as_tuple=True)[0]
            neg[i] = (diff[torch.randint(len(diff), (1,)).item()].item()
                      if len(diff) > 0 else i)
            continue
        dom = domains[i]
        if rng.random() < same_domain_ratio:
            cand = [j for j in range(B)
                    if lab[j] != lab[i] and domains[j] == dom]
            kind = "same"
        else:
            cand = [j for j in range(B)
                    if domains[j] != dom]
            kind = "cross"
        if not cand:
            # fallback
            diff = (lab != lab[i]).nonzero(as_tuple=True)[0]
            cand = diff.tolist() if len(diff) > 0 else [i]
            kind = "fallback"
        neg[i] = cand[rng.randrange(len(cand))]
        if stats is not None:
            stats[kind] = stats.get(kind, 0) + 1
    return pos, neg


def train_baseline_multi(source_datasets, args):
    combined = CombinedBalancedDataset(source_datasets)
    print("  Combined baseline training set:", len(combined),
          "samples,", len(set(combined.targets)), "classes")

    sampler = MPerClassBatchSampler(
        combined.targets, m=4, batch_size=args.batch_size,
        seed=args.baseline_seed,
    )
    loader = DataLoader(combined, batch_sampler=sampler,
                        num_workers=args.num_workers)

    set_seed(args.baseline_seed)
    model = SiameseBaseline(embedding_dim=128, pretrained=True).to(args.device)
    opt = AdamW(model.parameters(), lr=args.baseline_lr, weight_decay=1e-4)

    same_domain_ratio = getattr(args, "neg_same_domain_ratio", 0.5)
    print(f"  neg same-domain ratio target: {same_domain_ratio:.2f}")
    epoch_stats = {}
    for epoch in range(1, args.baseline_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        epoch_stats = {}
        for imgs, labels in loader:
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            # domains for this batch
            idxs = labels.cpu().tolist()
            # sampler yields in order, but we need index; get from dataset
            # loader is over `combined`, sampler yields indices
            # easier: infer domains by class → domain mapping
            # Build c2d fresh per fold (class->domain mapping differs
            # between folds because source dataset sizes differ)
            if "_c2d_fold" not in locals() or _c2d_fold_id != id(combined):
                _c2d = {}
                for k, lab in enumerate(combined.targets):
                    _c2d[lab] = combined.domains[k]
                _c2d_fold = _c2d
                _c2d_fold_id = id(combined)
            batch_domains = [_c2d_fold.get(l, -1) for l in idxs]
            pi, ni = build_pairs(labels, domains=batch_domains,
                                  same_domain_ratio=same_domain_ratio,
                                  stats=epoch_stats)
            pi = pi.to(args.device)
            ni = ni.to(args.device)

            sp = model(imgs, imgs[pi])
            sn = model(imgs, imgs[ni])
            scores = torch.cat([sp, sn]) * 5.0
            targets = torch.cat([torch.ones_like(sp), torch.zeros_like(sn)])
            loss = F.binary_cross_entropy_with_logits(scores, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        total_neg = sum(epoch_stats.values())
        if total_neg > 0:
            sd = epoch_stats.get("same", 0) / total_neg
            cd = epoch_stats.get("cross", 0) / total_neg
            fb = epoch_stats.get("fallback", 0) / total_neg
            print("    epoch {:2d} | loss={:.4f} | neg same={:.2f} cross={:.2f} fb={:.2f}"
                  .format(epoch, tot / max(n, 1), sd, cd, fb))
        else:
            print("    epoch {:2d} | loss={:.4f}".format(epoch, tot / max(n, 1)))

    return {"tower": model.tower.state_dict(),
            "head": model.emb_head.state_dict()}


# ---------- Pair sampling and eval ----------

def sample_supports_queries(dataset, K, M, seed, max_classes=None):
    rng = random.Random(seed)
    class_to_idx = {}
    # GenericMLDataset exposes labels without decoding the corresponding
    # image. Scanning via dataset[i] here would read every image once per
    # seed, including tens of thousands of unused query images.
    labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = [dataset[i][1] for i in range(len(dataset))]
    for i, label in enumerate(labels):
        class_to_idx.setdefault(label, []).append(i)

    classes = sorted(class_to_idx.keys())
    if max_classes is not None and 0 < max_classes < len(classes):
        # fixed subset per fold (seed=42) — same for all seeds & methods
        sel_rng = random.Random(42)
        sel_rng.shuffle(classes)
        classes = sorted(classes[:max_classes])

    supports, queries = {}, {}
    for c in classes:
        idxs = class_to_idx[c]
        if len(idxs) < K + 1:
            continue
        idxs = idxs[:]
        rng.shuffle(idxs)
        m_local = min(M, len(idxs) - K)
        supports[c] = [dataset[i][0] for i in idxs[:K]]
        queries[c] = [dataset[i][0] for i in idxs[K:K + m_local]]
    return supports, queries

def eval_pair_acc(model, queries, pairs_per_class, seed):
    rng = random.Random(seed)
    classes = list(queries.keys())
    pos, neg = [], []
    for c, imgs in queries.items():
        if len(imgs) < 2:
            continue
        from itertools import combinations
        all_pairs = list(combinations(range(len(imgs)), 2))
        rng.shuffle(all_pairs)
        for i, j in all_pairs[:pairs_per_class]:
            pos.append((imgs[i], imgs[j]))
        others = [x for x in classes if x != c]
        for _ in range(pairs_per_class):
            if not others:
                break
            c2 = rng.choice(others)
            neg.append((imgs[rng.randrange(len(imgs))],
                        queries[c2][rng.randrange(len(queries[c2]))]))

    if not pos or not neg:
        return float("nan")

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    model.eval()
    sp, sn = [], []
    with torch.no_grad():
        for b in chunks(pos, 64):
            x1 = torch.stack([p[0] for p in b]).to(next(model.parameters()).device)
            x2 = torch.stack([p[1] for p in b]).to(next(model.parameters()).device)
            sp.extend(model(x1, x2).cpu().tolist())
        for b in chunks(neg, 64):
            x1 = torch.stack([p[0] for p in b]).to(next(model.parameters()).device)
            x2 = torch.stack([p[1] for p in b]).to(next(model.parameters()).device)
            sn.extend(model(x1, x2).cpu().tolist())
    return pair_accuracy(torch.tensor(sp), torch.tensor(sn))


# ---------- Model building ----------

def _build_baseline_with_embed(ckpt, args):
    """Build SiameseBaseline with .embed() helper for baseline methods."""
    model = SiameseBaseline(embedding_dim=128, pretrained=False)
    model.tower.load_state_dict(ckpt["tower"])
    model.emb_head.load_state_dict(ckpt["head"])
    def _embed(self, x):
        z = self.tower(x)
        return F.normalize(self.emb_head(z), dim=-1)
    model.embed = _embed.__get__(model, SiameseBaseline)
    return model.to(args.device)


def build_from_baseline(arch, baseline_ckpt, args):
    if arch == "hyper-sym":
        model = HyperComparisonNet(
            n_kernels=getattr(args, "n_kernels", 2),
            feature_level=getattr(args, "feature_level", "l4"),
            kernel_gen_mode="mlp",
            out_ch_per_kernel=getattr(args, "out_ch", 16),
            use_cosine_residual=True,
            share_backbone=True,
            pretrained=False,
            symmetric=True,
        )
        model.tower_a.load_state_dict(baseline_ckpt["tower"])
        model.emb_head.load_state_dict(baseline_ckpt["head"])
        return model.to(args.device)

    if arch == "hyper-hybrid":
        rl = tuple(x.strip() for x in getattr(args, "residual_levels", "l3").split(","))
        model = HyperHybridNet(
            n_kernels=getattr(args, "n_kernels", 2),
            side_ch=getattr(args, "side_ch", 64),
            out_ch_per_kernel=getattr(args, "out_ch", 16),
            residual_levels=rl,
            pretrained=False,
        )
        for name in ["stem", "layer1", "layer2", "layer3", "layer4", "pool"]:
            src = {k[len(name) + 1:]: v
                   for k, v in baseline_ckpt["tower"].items()
                   if k.startswith(name + ".")}
            if src:
                getattr(model.tower, name).load_state_dict(src)
        model.emb_head.load_state_dict(baseline_ckpt["head"])
        return model.to(args.device)

    if arch == "hyper-multi":
        levels = tuple(x.strip() for x in getattr(args, "levels", "l2,l3").split(","))
        model = HyperMultiNet(
            levels=levels, side_ch=getattr(args, "side_ch", 32),
            k=3, hidden=getattr(args, "multi_hidden", 32),
            pretrained=False,
        )
        for name in ["stem", "layer1", "layer2", "layer3", "layer4", "pool"]:
            src = {k[len(name) + 1:]: v
                   for k, v in baseline_ckpt["tower"].items()
                   if k.startswith(name + ".")}
            if src:
                getattr(model, name).load_state_dict(src)
        model.emb_head.load_state_dict(baseline_ckpt["head"])
        return model.to(args.device)

    if arch == "nonspatial":
        model = DynamicSiameseNet(
            embedding_dim=128, generate_kernels=True,
            pretrained=False, share_backbone=True, spatial=False)
        model.tower_a.load_state_dict(baseline_ckpt["tower"])
        model.emb_head_a.load_state_dict(baseline_ckpt["head"])
        if model.emb_head_b is not model.emb_head_a:
            model.emb_head_b.load_state_dict(baseline_ckpt["head"])
        return model.to(args.device)

    use_cos = (arch == "hyper")
    model = HyperComparisonNet(
        n_kernels=args.n_kernels,
        feature_level=args.feature_level,
        kernel_gen_mode="mlp",
        out_ch_per_kernel=args.out_ch,
        use_cosine_residual=use_cos,
        share_backbone=True, pretrained=False,
    )
    model.tower_a.load_state_dict(baseline_ckpt["tower"])
    if use_cos:
        model.emb_head.load_state_dict(baseline_ckpt["head"])
    return model.to(args.device)


def adapt_model(model, supports, args, arch):
    from dynamic_siamese.profiler import Timer, record

    for p in model.parameters():
        p.requires_grad = False
    # Keep the frozen backbone in eval mode, then explicitly enable training
    # behavior only for the modules being adapted.
    model.eval()
    if arch == "nonspatial":
        for p in model.side_block.parameters():
            p.requires_grad = True
        params = [p for p in model.side_block.parameters() if p.requires_grad]
        model.side_block.train()
    elif hasattr(model, "kernel_gens"):  # hyper-multi only (plural)
        params = list(model.adapt_params())
        for p in params:
            p.requires_grad = True
        model.side_blocks.train()
        if hasattr(model, "kernel_gens"):
            model.kernel_gens.train()
    else:
        params = list(model.adapt_params())
        for p in params:
            p.requires_grad = True
        if hasattr(model, "head"):
            model.head.train()
        if hasattr(model, "kernel_gen"):
            model.kernel_gen.train()
        if hasattr(model, "side_blocks"):
            model.side_blocks.train()
        elif hasattr(model, "side_block"):
            model.side_block.train()

    opt = AdamW(params, lr=args.lr_side)

    import itertools
    classes = list(supports.keys())
    pos_pairs = []
    for c, imgs in supports.items():
        for i, j in itertools.combinations(range(len(imgs)), 2):
            pos_pairs.append((imgs[i], imgs[j]))
    neg_pairs = []
    for _ in range(max(len(pos_pairs), 200)):
        c1, c2 = random.sample(classes, 2)
        neg_pairs.append((random.choice(supports[c1]),
                          random.choice(supports[c2])))

    all_pairs = ([(a, b, 1.0) for a, b in pos_pairs]
                 + [(a, b, 0.0) for a, b in neg_pairs])

    with Timer(args.device) as train_timer:
        for epoch in range(1, args.adapt_epochs + 1):
            random.shuffle(all_pairs)
            for b in [all_pairs[i:i + args.batch_size]
                      for i in range(0, len(all_pairs), args.batch_size)]:
                x1 = torch.stack([p[0] for p in b]).to(args.device)
                x2 = torch.stack([p[1] for p in b]).to(args.device)
                t = torch.tensor([p[2] for p in b], dtype=torch.float32,
                                 device=args.device)
                sim = model(x1, x2) * 5.0
                loss = F.binary_cross_entropy_with_logits(sim, t)
                opt.zero_grad()
                loss.backward()
                opt.step()
    record("train_time_s", train_timer.time_s)
    return model


# ---------- Main LOOCV loop ----------

def compute_global_budget(all_datasets, root, args):
    """Compute min available classes and per-class samples across datasets.

    If --classes-per-dataset or --samples-per-class is None -> auto = min.
    If given -> cap at min (cannot exceed availability).
    """
    from collections import Counter

    mode = getattr(args, "balance_mode", "global")
    if mode == "off":
        return (args.classes_per_dataset or 30,
                args.samples_per_class or 20, None)

    rows = []
    for name in all_datasets:
        try:
            ds = GenericMLDataset(
                root, name, split="train", transform=None,
                max_classes=None, max_per_class=None,
            )
            cnt = Counter(ds.targets)
            if not cnt:
                continue
            rows.append((name, len(cnt), min(cnt.values()), max(cnt.values())))
        except FileNotFoundError:
            continue

    if not rows:
        return (args.classes_per_dataset or 30,
                args.samples_per_class or 20, None)

    avail_cls = min(r[1] for r in rows)
    avail_per = min(r[2] for r in rows)

    if mode == "global":
        eff_cls = avail_cls if args.classes_per_dataset is None \
            else min(args.classes_per_dataset, avail_cls)
        eff_per = avail_per if args.samples_per_class is None \
            else min(args.samples_per_class, avail_per)
    else:  # per-fold
        eff_cls = args.classes_per_dataset or 30
        eff_per = args.samples_per_class or 20

    return eff_cls, eff_per, rows

def load_source_datasets(datasets, root, args):
    """Load source datasets with effective balance caps from args."""
    eff_cls = getattr(args, "_eff_classes", args.classes_per_dataset)
    eff_per = getattr(args, "_eff_per_class", args.samples_per_class)
    sources = []
    for name in datasets:
        try:
            ds = GenericMLDataset(
                root, name, split="train", transform=eval_tf(),
                max_classes=eff_cls,
                max_per_class=eff_per,
            )
            print("  source {}: {}".format(name, ds.stats()))
            sources.append(ds)
        except FileNotFoundError as e:
            print("  SKIP source {}: {}".format(name, e))
    return sources


def main():
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    arches = [x.strip() for x in args.arches.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    print("Device:", args.device)
    print("Datasets:", datasets)
    print("Arches:", arches)
    print("Seeds:", seeds)
    print()

    # --- Global balance budget ---
    eff_cls, eff_per, budget_rows = compute_global_budget(
        datasets, args.data_root, args)
    if budget_rows:
        print("Dataset stats (train split, before balancing):")
        for name, ncls, nmin, nmax in budget_rows:
            print("  {:<10} {:>4} classes, samples/class {}-{}".format(
                name, ncls, nmin, nmax))
        print("Balance mode:", args.balance_mode)
        if args.classes_per_dataset is None or args.samples_per_class is None:
            cls_lim = min(budget_rows, key=lambda r: r[1])
            per_lim = min(budget_rows, key=lambda r: r[2])
            print("Auto budget: {} x {}".format(eff_cls, eff_per))
            if args.classes_per_dataset is None:
                print("  classes limited by {} ({})".format(
                    cls_lim[0], cls_lim[1]))
            if args.samples_per_class is None:
                print("  samples limited by {} ({}/class)".format(
                    per_lim[0], per_lim[2]))
        print("Effective per source: {} classes x {} samples = {} images".format(
            eff_cls, eff_per, eff_cls * eff_per))
        n_sources_per_fold = len(datasets) - 1
        print("Expected total per fold: {} images ({} sources)".format(
            eff_cls * eff_per * n_sources_per_fold, n_sources_per_fold))
        print()
    args._eff_classes = eff_cls
    args._eff_per_class = eff_per
    if getattr(args, "M_per_class", 0) == 0:
        args.M_per_class = max(1, eff_per - args.K)
        print("  M-per-class (auto): {}".format(args.M_per_class))

    results = []
    for holdout in datasets:
        print("=" * 70)
        print("FOLD: holdout={}".format(holdout))
        print("=" * 70)

        sources = load_source_datasets(
            [d for d in datasets if d != holdout], args.data_root, args)

        if len(sources) < 1:
            print("  SKIP: no source datasets available")
            continue
        if len(sources) == 1:
            print("  NOTE: only 1 source dataset in this fold "
                  "(fine for sanity checks, not for final LOOCV)")

        # cache baseline per fold
        cache_path = Path("baseline_loo_{}.pt".format(holdout))
        if cache_path.exists() and not args.skip_baseline_cache:
            print("  loading cached baseline:", cache_path)
            ckpt = torch.load(cache_path, weights_only=False, map_location="cpu")
        else:
            print("  training baseline on", len(sources), "datasets")
            ckpt = train_baseline_multi(sources, args)
            torch.save(ckpt, cache_path)
            print("  saved:", cache_path)

        # load holdout test (novel classes, no balancing)
        try:
            target = GenericMLDataset(
                args.data_root, holdout, split="test",
                transform=eval_tf(), max_per_class=None, max_classes=None)
        except FileNotFoundError as e:
            print("  SKIP target:", e)
            continue
        print("  holdout target:", target.stats())

        # baseline eval on holdout
        base_model = SiameseBaseline(embedding_dim=128, pretrained=False)
        base_model.tower.load_state_dict(ckpt["tower"])
        base_model.emb_head.load_state_dict(ckpt["head"])
        base_model = base_model.to(args.device)

        supports, queries = sample_supports_queries(
            target, args.K, args.M_per_class, seed=0,
            max_classes=eff_cls)
        print("  usable classes:", len(supports))

        # Decode each seed's support/query split once and reuse it for every
        # architecture. This keeps comparisons paired and avoids repeatedly
        # loading the same target images from disk.
        seed_splits = {}
        base_acc_by_seed = {}
        base_accs = []
        for seed in seeds:
            sup, qry = sample_supports_queries(
                target, args.K, args.M_per_class, seed=seed,
                max_classes=(getattr(args, 'eval_classes', 0) or eff_cls))
            seed_splits[seed] = (sup, qry)
            base_acc = eval_pair_acc(base_model, qry, 5, seed=seed)
            base_acc_by_seed[seed] = base_acc
            base_accs.append(base_acc)

        row = {"holdout": holdout, "n_classes": len(supports)}
        for arch in arches:
            accs = []
            for seed in seeds:
                from dynamic_siamese.profiler import (Timer as _TT,
                                                      reset_stats as _rs,
                                                      get_stats as _gs,
                                                      cuda_empty as _ce,
                                                      record as _record)
                _rs(); _ce(args.device)
                _tm = _TT(args.device); _tm.__enter__()
                set_seed(seed)
                sup, qry = seed_splits[seed]
                base_acc_seed = base_acc_by_seed[seed]
                _finetune_time_s = 0.0
                if getattr(args, 'finetune', False):
                    pass  # SiameseBaseline from module-level import
                    _m = SiameseBaseline(embedding_dim=128, pretrained=False)
                    _m.tower.load_state_dict(ckpt['tower'])
                    _m.emb_head.load_state_dict(ckpt['head'])
                    _m = _m.to(args.device)
                    _ft_timer = _TT(args.device); _ft_timer.__enter__()
                    _m = finetune_encoder(_m, sup, args)
                    _ft_timer.__exit__(None, None, None)
                    _finetune_time_s = _ft_timer.time_s
                    def _embed(self, x):
                        return F.normalize(self.emb_head(self.tower(x)), dim=-1)
                    _m.embed = _embed.__get__(_m, SiameseBaseline)
                    if arch == 'cosine':
                        a = eval_pair_acc(_m, qry, 5, seed=seed)
                    elif arch in BASELINE_ARCHS:
                        extra = {}
                        if arch == 'relationnet':
                            extra = dict(epochs=args.rn_epochs, lr=args.rn_lr)
                        a = BASELINE_DISPATCH[arch](
                            _m, sup, qry, args.device, seed=seed, **extra)
                    else:
                        ft_ckpt = {
                            "tower": _m.tower.state_dict(),
                            "head": _m.emb_head.state_dict(),
                        }
                        _km = build_from_baseline(arch, ft_ckpt, args)
                        _km = adapt_model(_km, sup, args, arch)
                        a = eval_pair_acc(_km, qry, 5, seed=seed)
                elif arch == "baseline":
                    # The cross-domain encoder baseline is the unadapted
                    # model evaluated above on this seed's query split.
                    a = base_acc_seed
                elif arch in BASELINE_ARCHS:
                    model = _build_baseline_with_embed(ckpt, args)
                    extra = {}
                    if arch == "relationnet":
                        extra = dict(epochs=args.rn_epochs, lr=args.rn_lr,
                                     hidden=args.rn_hidden)
                    elif arch in ("linear", "cosclf"):
                        extra = dict(epochs=args.rn_epochs, lr=args.rn_lr)
                    a = BASELINE_DISPATCH[arch](
                        model, sup, qry, args.device, seed=seed, **extra)
                else:
                    model = build_from_baseline(arch, ckpt, args)
                    model = adapt_model(model, sup, args, arch)
                    a = eval_pair_acc(model, qry, 5, seed=seed)
                if _finetune_time_s:
                    # Preserve encoder fine-tuning time alongside any later
                    # head/adaptation training time recorded by the branch.
                    _record("train_time_s", _finetune_time_s
                            + _gs().get("train_time_s", 0.0))
                accs.append(a)
                _tm.__exit__()
                _st = _gs()
                print("    {} seed={} after={:.4f} delta={:+.4f} "
                      "time={:.1f}s peak={:.0f}MB"
                      .format(arch, seed, a, a - base_acc_seed,
                              _tm.time_s, _tm.peak_mb))
                row.setdefault("_times", []).append(_tm.time_s)
                row.setdefault("_peaks", []).append(_tm.peak_mb)
                row.setdefault("_train", []).append(
                    _st.get("train_time_s", 0.0))
                row.setdefault("_infer", []).append(
                    _st.get("infer_time_s", 0.0))
            for _key, _name in [("_times", "total_time_s"),
                                 ("_peaks", "peak_mb"),
                                 ("_train", "train_time_s"),
                                 ("_infer", "infer_time_s")]:
                if _key in row and row[_key]:
                    _vals = row.pop(_key)
                    row[f"{arch}_{_name}_mean"] = mean(_vals)
                    row[f"{arch}_{_name}_min"] = min(_vals)
                    row[f"{arch}_{_name}_max"] = max(_vals)
            m = mean(accs) if accs else float("nan")
            s = stdev(accs) if len(accs) > 1 else 0.0
            row[arch + "_mean"] = m
            row[arch + "_std"] = s
            print("    {} SUMMARY: {:.4f} ± {:.4f}".format(arch, m, s))

        base_acc = mean(base_accs) if base_accs else float("nan")
        row["cross_baseline"] = base_acc
        print("  cross-domain baseline pair_acc (seed mean): {:.4f}".format(
            base_acc))
        results.append(row)

    # save + summary
    if results:
        keys = ["holdout", "n_classes", "cross_baseline"]
        for arch in arches:
            keys += [arch + "_mean", arch + "_std",
                     arch + "_total_time_s_mean", arch + "_total_time_s_min",
                     arch + "_total_time_s_max",
                     arch + "_peak_mb_mean", arch + "_peak_mb_min",
                     arch + "_peak_mb_max",
                     arch + "_train_time_s_mean", arch + "_train_time_s_min",
                     arch + "_train_time_s_max",
                     arch + "_infer_time_s_mean", arch + "_infer_time_s_min",
                     arch + "_infer_time_s_max"]
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in results:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

        print()
        print("=" * 78)
        print("LOOCV SUMMARY")
        print("=" * 78)
        header = "| {:<10} | {:<4} | {:<12} |".format(
            "Holdout", "N", "Cross-base")
        for arch in arches:
            header += " {:<22} |".format(arch + " (mean±std)")
        print(header)
        print("|" + "-"*12 + "|" + "-"*6 + "|" + "-"*14 + "|"
              + "".join("-"*24 + "|" for _ in arches))
        for r in results:
            line = "| {:<10} | {:<4} | {:.4f}       |".format(
                r["holdout"], r["n_classes"], r["cross_baseline"])
            for arch in arches:
                m = r.get(arch + "_mean", float("nan"))
                s = r.get(arch + "_std", 0.0)
                line += " {:.4f} ± {:.4f}      |".format(m, s)
            print(line)
        print("=" * 78)
        print("Saved to:", args.output)


if __name__ == "__main__":
    main()
