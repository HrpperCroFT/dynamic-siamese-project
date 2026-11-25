"""Few-shot adaptation evaluation on CUB200 novel classes.

Protocol:
  1. Train (or load) baseline Siamese on base classes (CUB train split,
     classes 0-99). Novel classes = CUB test split (classes 100-199).
  2. For each K in --K-list (default 5,10):
     a. Sample K support images per novel class.
     b. For fixed/generated: freeze backbone, adapt side-block on
        support pairs only, for --adapt-epochs.
     c. Evaluate pair_acc on disjoint query images (M per class).
  3. Report baseline (no adaptation), fixed (adapted), generated (adapted).

Run:
  python scripts/few_shot_eval.py --K-list 5,10 --adapt-epochs 10 --seed 1
  python scripts/few_shot_eval.py --modes generated --K-list 5,10,20
  python scripts/few_shot_eval.py --skip-baseline-train
"""
import argparse
import itertools
import random
import sys
import time
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import os
os.chdir(PROJECT_ROOT)

try:
    import dynamic_siamese  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from dynamic_siamese import (
    DynamicSiameseNet, SiameseBaseline, CUB200MetricLearning,
    MPerClassBatchSampler,
)
from dynamic_siamese.utils import pair_accuracy
from dynamic_siamese.baselines import BASELINE_ARCHS, DISPATCH as BASELINE_DISPATCH


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--baseline-ckpt", default="")
    p.add_argument("--baseline-epochs", type=int, default=12)
    p.add_argument("--baseline-lr", type=float, default=1e-4)
    p.add_argument("--skip-baseline-train", action="store_true",
                   help="reuse existing baseline_cub.pt without retraining")
    p.add_argument("--adapt-epochs", type=int, default=10,
                   help="how many passes over support pairs during adaptation")
    p.add_argument("--lr-side", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--K-list", type=str, default="5,10")
    p.add_argument("--modes", type=str, default="baseline,fixed,generated")
    p.add_argument("--M-per-class", type=int, default=40,
                   help="query images per class for evaluation")
    p.add_argument("--eval-pairs-per-class", type=int, default=20,
                   help="positive (and negative) pairs per class for pair_acc")
    p.add_argument("--spatial", action="store_true")
    p.add_argument("--arch", default="cosine",
                   choices=["cosine", "hyper", "hyper-pure", "hyper-sym", "protonet", "linear", "cosclf", "relationnet", "bnadapt", "finetune", "relationgcn", "hyper-multi", "hyper-hybrid"],
                   help="cosine = side-block injection (original); "
                        "hyper = learned head + cosine residual; "
                        "hyper-pure = learned head only")
    p.add_argument("--n-kernels", type=int, default=4)
    p.add_argument("--levels", default="l2,l3")
    p.add_argument("--residual-levels", default="l3")
    p.add_argument("--side-ch", type=int, default=32)
    p.add_argument("--multi-hidden", type=int, default=32)
    p.add_argument("--feature-level", default="l2",
                   choices=["l1", "l2", "l3", "l4"],
                   help="feature map level for hyper head")
    p.add_argument("--kernel-gen-mode", default="mlp",
                   choices=["conv", "mlp"],
                   help="mlp = safest (uses z_ref), conv = pooled spatial")
    p.add_argument("--out-ch", type=int, default=16,
                   help="channels per generated kernel (hyper mode)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--finetune", action="store_true",
                   help="fine-tune layer4+head with lr scheduler")
    p.add_argument("--ft-lr", type=float, default=1e-5)
    p.add_argument("--ft-epochs", type=int, default=15)
    p.add_argument("--ft-patience", type=int, default=3)
    p.add_argument("--ft-lr-patience", type=int, default=2)
    p.add_argument("--ft-val-frac", type=float, default=0.2)
    p.add_argument("--rn-epochs", type=int, default=30,
                   help="relationnet training epochs")
    p.add_argument("--rn-lr", type=float, default=1e-3)
    p.add_argument("--rn-hidden", type=int, default=64)
    p.add_argument("--output-json", default=None,
                   help="write results to JSON file")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds, e.g. 1,2,3,4,5. Overrides --seed")
    p.add_argument("--dataset", default="cub", choices=["cub", "cars"])
    p.add_argument("--kernel-scale-l2", type=float, default=0.0,
                   help="L2 penalty on kernel_scale during adaptation")
    p.add_argument("--side-scale-l2", type=float, default=0.0,
                   help="L2 penalty on side_scale during adaptation")
    p.add_argument("--early-stop-heldout", action="store_true",
                   help="hold out 20pct support, stop when val_loss rises")
    p.add_argument("--early-stop-patience", type=int, default=3)
    return p.parse_args()


def set_seed(seed):
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def build_pairs(labels):
    B = len(labels)
    lab = labels.detach().cpu()
    pos = torch.zeros(B, dtype=torch.long)
    neg = torch.zeros(B, dtype=torch.long)
    for i in range(B):
        same = (lab == lab[i]).nonzero(as_tuple=True)[0]
        same = same[same != i]
        pos[i] = (same[torch.randint(len(same), (1,)).item()].item()
                  if len(same) > 0 else i)
        diff = (lab != lab[i]).nonzero(as_tuple=True)[0]
        neg[i] = (diff[torch.randint(len(diff), (1,)).item()].item()
                  if len(diff) > 0 else i)
    return pos, neg


# ---------- Baseline training / loading ----------

def train_baseline(args):
    print("=== Stage 0: train baseline on base classes ===")
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if args.dataset == "cars":
        from dynamic_siamese import Cars196MetricLearning
        train_ds = Cars196MetricLearning(
            args.data_root, split="train", transform=train_tf)
    else:
        train_ds = CUB200MetricLearning(
            args.data_root, split="train", transform=train_tf)
    print("Base-class samples:", len(train_ds))

    sampler = MPerClassBatchSampler(
        train_ds.targets, m=4, batch_size=32, seed=args.seed,
    )
    loader = DataLoader(
        train_ds, batch_sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = SiameseBaseline(embedding_dim=128, pretrained=True).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.baseline_lr,
                      weight_decay=1e-4)

    best_acc = -1.0
    for epoch in range(1, args.baseline_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for imgs, labels in loader:
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            pos_idx, neg_idx = build_pairs(labels)
            pos_idx = pos_idx.to(args.device)
            neg_idx = neg_idx.to(args.device)

            sim_pos = model(imgs, imgs[pos_idx])
            sim_neg = model(imgs, imgs[neg_idx])
            scores = torch.cat([sim_pos, sim_neg]) * 5.0
            targets = torch.cat([
                torch.ones_like(sim_pos), torch.zeros_like(sim_neg),
            ])
            loss = F.binary_cross_entropy_with_logits(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()
            n += 1
        print("  epoch", epoch, "| loss=", round(total / max(n, 1), 4))

    return {"tower": model.tower.state_dict(),
            "head": model.emb_head.state_dict()}


def load_or_train_baseline(args):
    if not args.baseline_ckpt:
        args.baseline_ckpt = f"baseline_{args.dataset}.pt"
    path = Path(args.baseline_ckpt)
    if path.exists() and args.skip_baseline_train:
        print("Loading baseline from", path)
        return torch.load(path, weights_only=False, map_location="cpu")
    if path.exists() and not args.skip_baseline_train:
        print("Baseline checkpoint exists at", path,
              "- pass --skip-baseline-train to reuse, "
              "or delete it to retrain.")
        print("Reusing existing checkpoint.")
        return torch.load(path, weights_only=False, map_location="cpu")
    ckpt = train_baseline(args)
    torch.save(ckpt, path)
    print("Saved baseline to", path)
    return ckpt


# ---------- Support / query split ----------

def sample_supports_queries(test_ds, K, M_per_class, seed, max_classes=None):
    rng = random.Random(seed)
    class_to_idx = {}
    for i in range(len(test_ds)):
        _, label = test_ds[i]
        class_to_idx.setdefault(label, []).append(i)

    classes = sorted(class_to_idx.keys())
    if max_classes is not None and 0 < max_classes < len(classes):
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
        m_local = min(M_per_class, len(idxs) - K)
        supports[c] = [test_ds[i][0] for i in idxs[:K]]
        queries[c] = [test_ds[i][0] for i in idxs[K:K + m_local]]
    return supports, queries

def adapt_side_block(model, supports, args):
    """Freeze backbone, train only the adaptation module.

    Supports:
      --kernel-scale-l2 : L2 penalty on kernel_scale
      --side-scale-l2   : L2 penalty on side_scale
      --early-stop-heldout : reserve 20pct of supports as held-out val,
                              stop when val loss does not improve
    """
    arch = getattr(args, "arch", "cosine")
    ks_l2 = getattr(args, "kernel_scale_l2", 0.0)
    ss_l2 = getattr(args, "side_scale_l2", 0.0)
    use_heldout = getattr(args, "early_stop_heldout", False)
    patience_limit = getattr(args, "early_stop_patience", 3)

    # Optional held-out split of supports
    if use_heldout:
        train_supports = {}
        val_supports = {}
        for c, imgs in supports.items():
            n_val = max(1, len(imgs) // 5)
            if len(imgs) - n_val < 2:
                train_supports[c] = imgs
            else:
                train_supports[c] = imgs[:-n_val]
                val_supports[c] = imgs[-n_val:]

        val_classes = [c for c, v in val_supports.items() if len(v) >= 1]
        val_pos_pairs, val_neg_pairs = [], []
        for c in val_classes:
            if len(val_supports[c]) >= 2:
                val_pos_pairs.append(
                    (val_supports[c][0], val_supports[c][1]))
            others = [x for x in val_classes if x != c]
            if others:
                c2 = random.choice(others)
                val_neg_pairs.append(
                    (val_supports[c][0], val_supports[c2][0]))
        print("  held-out: {} classes, {} pos, {} neg pairs".format(
            len(val_supports), len(val_pos_pairs), len(val_neg_pairs)))
    else:
        train_supports = supports
        val_pos_pairs = None
        val_neg_pairs = None

    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "adapt_params"):
        adapt_params_list = list(model.adapt_params())
        for p in adapt_params_list:
            p.requires_grad = True
        optimizer = AdamW(adapt_params_list, lr=args.lr_side)
    else:
        for p in model.side_block.parameters():
            p.requires_grad = True
        optimizer = AdamW(
            [p for p in model.side_block.parameters() if p.requires_grad],
            lr=args.lr_side,
        )

    classes = list(train_supports.keys())
    pos_pairs = []
    for c, imgs in train_supports.items():
        for i, j in itertools.combinations(range(len(imgs)), 2):
            pos_pairs.append((imgs[i], imgs[j]))

    neg_pairs = []
    n_neg = max(len(pos_pairs), 200)
    for _ in range(n_neg):
        c1, c2 = random.sample(classes, 2)
        i = random.choice(train_supports[c1])
        j = random.choice(train_supports[c2])
        neg_pairs.append((i, j))

    all_pairs = [(p[0], p[1], 1.0) for p in pos_pairs] \
        + [(p[0], p[1], 0.0) for p in neg_pairs]
    print("  adaptation pairs: {} pos, {} neg (total {})".format(
        len(pos_pairs), len(neg_pairs), len(all_pairs),
    ))

    model.eval()
    if hasattr(model, "head") and hasattr(model, "kernel_gen"):
        model.head.train()
        model.kernel_gen.train()
    elif hasattr(model, "side_blocks"):
        model.side_blocks.train()
    elif hasattr(model, "side_block"):
        model.side_block.train()

    initial_scale = None
    if hasattr(model, "side_block") and hasattr(
        model.side_block, "side_scale"
    ):
        initial_scale = model.side_block.side_scale.item()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    from dynamic_siamese.profiler import Timer, record
    train_timer = Timer(args.device)
    train_timer.__enter__()
    for epoch in range(1, args.adapt_epochs + 1):
        random.shuffle(all_pairs)
        total, n = 0.0, 0
        for batch in chunks(all_pairs, args.batch_size):
            x1 = torch.stack([b[0] for b in batch]).to(args.device)
            x2 = torch.stack([b[1] for b in batch]).to(args.device)
            targets = torch.tensor([b[2] for b in batch],
                                   dtype=torch.float32, device=args.device)

            sim = model(x1, x2) * 5.0
            loss = F.binary_cross_entropy_with_logits(sim, targets)

            if ks_l2 > 0 and hasattr(model, "side_block") and \
                    hasattr(model.side_block, "kernel_scale"):
                loss = loss + ks_l2 * model.side_block.kernel_scale.pow(2)
            if ss_l2 > 0 and hasattr(model, "side_block") and \
                    hasattr(model.side_block, "side_scale"):
                loss = loss + ss_l2 * model.side_block.side_scale.pow(2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()
            n += 1

        log_epoch = (epoch % max(1, args.adapt_epochs // 3) == 0) or (epoch == 1)
        msg = "  adapt epoch {:2d} | loss={:.4f}".format(
            epoch, total / max(n, 1),
        )
        if arch in ("hyper", "hyper-pure", "hyper-multi"):
            if hasattr(model, "combine_alpha"):
                msg += " | alpha={:+.5f}".format(
                    model.combine_alpha.item())
            if hasattr(model, "head") and hasattr(model.head, "out_scale"):
                msg += " | head_scale={:+.5f}".format(
                    model.head.out_scale.item())
                if hasattr(model, "side_blocks"):
                    for lv in getattr(model, "levels", getattr(model, "residual_levels", [])):
                        msg += " | {}={:+.4f}".format(
                            lv, model.side_blocks[lv].side_scale.item())
        else:
            if hasattr(model, "side_block") and hasattr(model.side_block, "side_scale"):
                msg += " | side_scale={:+.5f}".format(
                    model.side_block.side_scale.item())
            if hasattr(model, "side_block") and hasattr(model.side_block, "kernel_scale"):
                msg += " | kernel_scale={:+.5f}".format(
                    model.side_block.kernel_scale.item())

        if use_heldout and val_pos_pairs and val_neg_pairs:
            if hasattr(model, "side_block"):
                model.side_block.eval()
            with torch.no_grad():
                x1p = torch.stack([p[0] for p in val_pos_pairs]).to(args.device)
                x2p = torch.stack([p[1] for p in val_pos_pairs]).to(args.device)
                x1n = torch.stack([p[0] for p in val_neg_pairs]).to(args.device)
                x2n = torch.stack([p[1] for p in val_neg_pairs]).to(args.device)
                sp = model(x1p, x2p) * 5.0
                sn = model(x1n, x2n) * 5.0
                tp = torch.ones_like(sp)
                tn = torch.zeros_like(sn)
                val_loss = F.binary_cross_entropy_with_logits(
                    torch.cat([sp, sn]), torch.cat([tp, tn])).item()
            if hasattr(model, "side_block"):
                model.side_block.train()
            elif hasattr(model, "side_blocks"):
                model.side_blocks.train()
            elif hasattr(model, "head") and hasattr(model, "kernel_gen"):
                model.head.train()
                model.kernel_gen.train()

            improved = val_loss < best_val_loss - 1e-4
            if improved:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if log_epoch:
                print(msg + " | val_loss={:.4f}{}".format(
                    val_loss, " *" if improved else ""))
            if patience_counter >= patience_limit:
                print("  early stop at epoch {} | best val_loss={:.4f}".format(
                    epoch, best_val_loss))
                break
        else:
            if log_epoch:
                print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)

    train_timer.__exit__(None, None, None)
    record("train_time_s", train_timer.time_s)

    if initial_scale is not None and hasattr(model, "side_block"):
        if hasattr(model, "side_block") and hasattr(model.side_block, "side_scale"):
            print("  side_scale: {:.5f} -> {:.5f}".format(
                initial_scale, model.side_block.side_scale.item(),
            ))
    return model


# ---------- Evaluation ----------

def evaluate_pair_acc(model, queries, args):
    """Sample positive/negative pairs from query sets, return pair_acc."""
    model.eval()
    classes = list(queries.keys())
    pos_pairs, neg_pairs = [], []

    for c, imgs in queries.items():
        if len(imgs) < 2:
            continue
        pairs = list(itertools.combinations(range(len(imgs)), 2))
        random.shuffle(pairs)
        for i, j in pairs[:args.eval_pairs_per_class]:
            pos_pairs.append((imgs[i], imgs[j]))
        other = [x for x in classes if x != c]
        for _ in range(args.eval_pairs_per_class):
            c2 = random.choice(other)
            i = random.choice(imgs)
            j = random.choice(queries[c2])
            neg_pairs.append((i, j))

    sim_pos, sim_neg = [], []
    with torch.no_grad():
        for batch in chunks(pos_pairs, args.batch_size):
            x1 = torch.stack([p[0] for p in batch]).to(args.device)
            x2 = torch.stack([p[1] for p in batch]).to(args.device)
            sim_pos.extend(model(x1, x2).cpu().tolist())
        for batch in chunks(neg_pairs, args.batch_size):
            x1 = torch.stack([p[0] for p in batch]).to(args.device)
            x2 = torch.stack([p[1] for p in batch]).to(args.device)
            sim_neg.extend(model(x1, x2).cpu().tolist())

    if not sim_pos or not sim_neg:
        return float("nan")
    return pair_accuracy(torch.tensor(sim_pos), torch.tensor(sim_neg))


# ---------- Per-configuration experiment ----------

def build_model(mode, baseline_ckpt, args):
    arch = getattr(args, "arch", "cosine")

    if mode == "baseline" or arch == "protonet":
        model = SiameseBaseline(embedding_dim=128, pretrained=False)
        model.tower.load_state_dict(baseline_ckpt["tower"])
        model.emb_head.load_state_dict(baseline_ckpt["head"])
        # add embed() for ProtoNet
        def _embed(self, x):
            z = self.tower(x)
            return F.normalize(self.emb_head(z), dim=-1)
        model.embed = _embed.__get__(model, SiameseBaseline)
        return model.to(args.device)

    if arch == "hyper-hybrid":
        from dynamic_siamese import HyperHybridNet
        rl = tuple(x.strip() for x in getattr(args, "residual_levels", "l3").split(","))
        model = HyperHybridNet(
            n_kernels=args.n_kernels, side_ch=args.side_ch,
            out_ch_per_kernel=args.out_ch,
            residual_levels=rl,
            pretrained=True,
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
        from dynamic_siamese import HyperMultiNet
        levels = tuple(x.strip() for x in args.levels.split(","))
        model = HyperMultiNet(
            levels=levels, side_ch=args.side_ch, k=3,
            hidden=args.multi_hidden, pretrained=True,
        )
        model.stem.load_state_dict(baseline_ckpt["tower"].get("stem", {}), strict=False)
        # direct copy from baseline ResNetTower
        for name in ["stem", "layer1", "layer2", "layer3", "layer4", "pool"]:
            src = {k[len(name) + 1:]: v
                   for k, v in baseline_ckpt["tower"].items()
                   if k.startswith(name + ".")}
            if src:
                getattr(model, name).load_state_dict(src)
        model.emb_head.load_state_dict(baseline_ckpt["head"])
        return model.to(args.device)

    if arch == "hyper-sym":
        from dynamic_siamese import HyperComparisonNet
        model = HyperComparisonNet(
            n_kernels=args.n_kernels,
            feature_level=args.feature_level,
            kernel_gen_mode=args.kernel_gen_mode,
            out_ch_per_kernel=args.out_ch,
            use_cosine_residual=True,
            share_backbone=True,
            pretrained=False,
            symmetric=True,
        )
        model.tower_a.load_state_dict(baseline_ckpt["tower"])
        model.emb_head.load_state_dict(baseline_ckpt["head"])
        return model.to(args.device)

    if arch in ("hyper", "hyper-pure"):
        from dynamic_siamese import HyperComparisonNet
        use_cos = (arch == "hyper")
        model = HyperComparisonNet(
            n_kernels=args.n_kernels,
            feature_level=args.feature_level,
            kernel_gen_mode=args.kernel_gen_mode,
            out_ch_per_kernel=args.out_ch,
            use_cosine_residual=use_cos,
            share_backbone=True,
            pretrained=False,
        )
        model.tower_a.load_state_dict(baseline_ckpt["tower"])
        if use_cos:
            model.emb_head.load_state_dict(baseline_ckpt["head"])
        return model.to(args.device)

    # arch == "cosine" (original DynamicSiameseNet)
    model = DynamicSiameseNet(
        embedding_dim=128,
        generate_kernels=(mode == "generated"),
        pretrained=False,
        share_backbone=True,
        spatial=args.spatial,
    )
    model.tower_a.load_state_dict(baseline_ckpt["tower"])
    model.emb_head_a.load_state_dict(baseline_ckpt["head"])
    if (hasattr(model, "emb_head_b")
            and model.emb_head_b is not model.emb_head_a):
        model.emb_head_b.load_state_dict(baseline_ckpt["head"])
    return model.to(args.device)



def _pair_accuracy_direct(model, queries, args):
    import itertools, random as _r
    rng = _r.Random(0); classes = list(queries.keys())
    pos, neg = [], []
    for c in classes:
        q = queries[c]
        if len(q) < 2: continue
        ps = list(itertools.combinations(range(len(q)), 2))
        rng.shuffle(ps)
        for i, j in ps[:5]: pos.append((q[i], q[j]))
        others = [x for x in classes if x != c]
        for _ in range(5):
            c2 = rng.choice(others)
            neg.append((q[rng.randrange(len(q))],
                        queries[c2][rng.randrange(len(queries[c2]))]))
    with torch.no_grad():
        xp = torch.stack([a for a, _ in pos]).to(args.device)
        yp = torch.stack([b for _, b in pos]).to(args.device)
        xn = torch.stack([a for a, _ in neg]).to(args.device)
        yn = torch.stack([b for _, b in neg]).to(args.device)
        sp = model(xp, yp); sn = model(xn, yn)
    return (sp.unsqueeze(1) > sn.unsqueeze(0)).float().mean().item()


def finetune_encoder(model, supports, args):
    """Fine-tune layer4 + head with lr scheduler + early stop on val split."""
    import random as _r
    import itertools
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    val_frac = getattr(args, "ft_val_frac", 0.2)
    rng = _r.Random(0)

    train_sup, val_sup = {}, {}
    for c, imgs in supports.items():
        n_val = max(1, int(len(imgs) * val_frac))
        n_train = len(imgs) - n_val
        if n_train < 2:
            train_sup[c] = imgs
            continue
        idxs = list(range(len(imgs)))
        rng.shuffle(idxs)
        train_sup[c] = [imgs[i] for i in idxs[:n_train]]
        val_sup[c] = [imgs[i] for i in idxs[n_train:]]

    if not val_sup:
        print("  [finetune] no val split, skipping")
        return model

    # freeze all, unfreeze layer4 + head
    for p in model.parameters():
        p.requires_grad = False
    for p in model.tower.layer4.parameters():
        p.requires_grad = True
    for p in model.emb_head.parameters():
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.ft_lr)
    sched = ReduceLROnPlateau(opt, factor=0.5,
                              patience=args.ft_lr_patience)

    classes = list(train_sup.keys())
    best_state = None
    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, args.ft_epochs + 1):
        model.train()
        pos, neg = [], []
        for c in classes:
            for i, j in itertools.combinations(range(len(train_sup[c])), 2):
                pos.append((train_sup[c][i], train_sup[c][j]))
        for _ in range(len(pos)):
            c1, c2 = rng.sample(classes, 2)
            i = rng.randrange(len(train_sup[c1]))
            j = rng.randrange(len(train_sup[c2]))
            neg.append((train_sup[c1][i], train_sup[c2][j]))
        pairs = [(a, b, 1.0) for a, b in pos] + [(a, b, 0.0) for a, b in neg]
        rng.shuffle(pairs)
        for i in range(0, len(pairs), 32):
            b = pairs[i:i+32]
            x1 = torch.stack([p[0] for p in b]).to(args.device)
            x2 = torch.stack([p[1] for p in b]).to(args.device)
            tt = torch.tensor([p[2] for p in b], dtype=torch.float32,
                              device=args.device)
            sim = model(x1, x2) * 5.0
            loss = F.binary_cross_entropy_with_logits(sim, tt)
            opt.zero_grad(); loss.backward(); opt.step()

        # val loss
        model.eval()
        vp, vn = [], []
        for c in val_sup:
            q = val_sup[c]
            for i, j in itertools.combinations(range(len(q)), 2):
                vp.append((q[i], q[j]))
            if len(classes) > 1:
                others = [x for x in classes if x != c]
                c2 = rng.choice(others)
                vn.append((q[0], val_sup[c2][0]))
        if not vp or not vn:
            sched.step(0.0)
            continue
        with torch.no_grad():
            xp = torch.stack([a for a, _ in vp]).to(args.device)
            yp = torch.stack([b for _, b in vp]).to(args.device)
            xn = torch.stack([a for a, _ in vn]).to(args.device)
            yn = torch.stack([b for _, b in vn]).to(args.device)
            sp = model(xp, yp) * 5.0
            sn = model(xn, yn) * 5.0
            val_loss = F.binary_cross_entropy_with_logits(
                torch.cat([sp, sn]),
                torch.cat([torch.ones_like(sp), torch.zeros_like(sn)])
            ).item()
        sched.step(val_loss)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.ft_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad = True
    model.eval()
    print(f"  [finetune] best val_loss={best_val:.4f}")
    return model


def run_experiment(baseline_ckpt, test_ds, mode, K, args):
    arch = getattr(args, "arch", "cosine")
    set_seed(args.seed)
    print()
    print("=" * 64)
    print("Experiment: mode={} arch={} K={}".format(mode, arch, K))
    print("=" * 64)
    supports, queries = sample_supports_queries(
        test_ds, K=K, M_per_class=args.M_per_class, seed=args.seed,
        max_classes=getattr(args, "eval_classes", 0) or None,
    )
    print("Novel classes:", len(supports))

    # unified finetune protocol: fine-tune SiameseBaseline, then rebuild
    # arch-specific model with fine-tuned weights, then adapt + eval
    if getattr(args, 'finetune', False):
        from dynamic_siamese import SiameseBaseline as SB
        sb = SB(embedding_dim=128, pretrained=False)
        sb.tower.load_state_dict(baseline_ckpt['tower'])
        sb.emb_head.load_state_dict(baseline_ckpt['head'])
        sb = sb.to(args.device)
        sb = finetune_encoder(sb, supports, args)

        def _embed(self, x):
            return F.normalize(self.emb_head(self.tower(x)), dim=-1)
        sb.embed = _embed.__get__(sb, SB)

        # 1. plain baseline after finetune
        if arch == 'cosine' and mode == 'baseline':
            acc = _pair_accuracy_direct(sb, queries, args)
            print(f'  finetune+baseline pair_acc: {acc:.4f}')
            return {'mode': 'finetune+baseline', 'K': K,
                    'acc_before': acc, 'acc_after': acc}

        # 2. baseline methods (protonet/linear/relationnet/...)
        if arch in BASELINE_ARCHS:
            extra = {}
            if arch == 'relationnet':
                extra = dict(epochs=args.rn_epochs, lr=args.rn_lr)
            elif arch in ('linear', 'cosclf'):
                extra = dict(epochs=args.rn_epochs, lr=args.rn_lr)
            acc = BASELINE_DISPATCH[arch](
                sb, supports, queries, args.device, seed=args.seed, **extra)
            print(f'  finetune+{arch} pair_acc: {acc:.4f}')
            return {'mode': 'finetune+' + arch, 'K': K,
                    'acc_before': acc, 'acc_after': acc}

        # 3. kernel-adaptation arch (hyper/hyper-pure/nonspatial)
        ft_ckpt = {'tower': sb.tower.state_dict(),
                   'head': sb.emb_head.state_dict()}
        model = build_model(mode, ft_ckpt, args)
        adapt_side_block(model, supports, args)
        acc = evaluate_pair_acc(model, queries, args)
        label = 'nonspatial' if (arch == 'cosine'
                                   and mode == 'generated') else arch
        print(f'  finetune+{label} pair_acc: {acc:.4f}')
        return {'mode': 'finetune+' + label, 'K': K,
                'acc_before': acc, 'acc_after': acc}

    if arch in BASELINE_ARCHS:
        from dynamic_siamese import SiameseBaseline
        model = SiameseBaseline(embedding_dim=128, pretrained=False)
        model.tower.load_state_dict(baseline_ckpt["tower"])
        model.emb_head.load_state_dict(baseline_ckpt["head"])
        def _embed(self, x):
            z = self.tower(x)
            return F.normalize(self.emb_head(z), dim=-1)
        model.embed = _embed.__get__(model, SiameseBaseline)
        model = model.to(args.device)
        extra = {}
        if arch == "relationnet":
            extra = dict(epochs=args.rn_epochs, lr=args.rn_lr,
                         hidden=args.rn_hidden)
        elif arch == "linear" or arch == "cosclf":
            extra = dict(epochs=args.rn_epochs, lr=args.rn_lr)
        acc = BASELINE_DISPATCH[arch](
            model, supports, queries, args.device, seed=args.seed, **extra)
        print("  {} pair_acc: {:.4f}".format(arch, acc))
        return {"mode": arch, "K": K,
                "acc_before": acc, "acc_after": acc}

    if mode == "baseline":
        model = build_model(mode, baseline_ckpt, args)
        acc = evaluate_pair_acc(model, queries, args)
        print("  baseline pair_acc:", round(acc, 4))
        return {"mode": mode, "K": K, "acc_before": acc, "acc_after": acc}

    model = build_model(mode, baseline_ckpt, args)
    acc_before = evaluate_pair_acc(model, queries, args)
    print("  before adapt:", round(acc_before, 4))
    adapt_side_block(model, supports, args)
    acc_after = evaluate_pair_acc(model, queries, args)
    print("  after adapt :", round(acc_after, 4),
          "(delta={:+.4f})".format(acc_after - acc_before))
    return {"mode": mode, "K": K,
            "acc_before": acc_before, "acc_after": acc_after}




# ---------- Main ----------

def main():
    args = parse_args()
    set_seed(args.seed)
    print("Device:", args.device)
    print("Seed:", args.seed)
    print("Arch:", getattr(args, "arch", "cosine"),
          "| Spatial:", args.spatial)
    if getattr(args, "arch", "cosine") in ("hyper", "hyper-pure"):
        print("  n_kernels:", args.n_kernels,
              "| feature_level:", args.feature_level,
              "| kernel_gen:", args.kernel_gen_mode,
              "| out_ch:", args.out_ch)

    # Test dataset (novel classes)
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if args.dataset == "cars":
        from dynamic_siamese import Cars196MetricLearning
        test_ds = Cars196MetricLearning(
            args.data_root, split="test", transform=eval_tf)
    else:
        test_ds = CUB200MetricLearning(
            args.data_root, split="test", transform=eval_tf)
    print("Novel-class samples:", len(test_ds))

    # Baseline
    baseline_ckpt = load_or_train_baseline(args)

    # Run grid
    K_list = [int(x.strip()) for x in args.K_list.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # If arch is a baseline method, ignore --modes (only one pass)
    if getattr(args, "arch", "cosine") in BASELINE_ARCHS:
        modes = ["baseline"]

    seeds_str = getattr(args, "seeds", None)
    if seeds_str:
        seed_list = [int(x.strip()) for x in seeds_str.split(",") if x.strip()]
    else:
        seed_list = [args.seed]

    print()
    print("Running seeds:", seed_list)

    from dynamic_siamese.profiler import Timer, reset_stats, get_stats, cuda_empty
    _dev = torch.device(args.device)
    results = []
    t0 = time.time()
    for seed in seed_list:
        args.seed = seed
        for mode in modes:
            for K in K_list:
                reset_stats()
                cuda_empty(_dev)
                with Timer(_dev) as _t:
                    r = run_experiment(baseline_ckpt, test_ds, mode, K, args)
                _st = get_stats()
                r["seed"] = seed
                r["train_time_s"] = _st.get("train_time_s", 0.0)
                r["infer_time_s"] = _st.get("infer_time_s", 0.0)
                r["total_time_s"] = _t.time_s
                r["peak_mb"] = _t.peak_mb
                results.append(r)
                print("  [profile] total={:.1f}s peak={:.0f}MB".format(
                    _t.time_s, _t.peak_mb))

    elapsed = time.time() - t0
    print()
    print("=" * 74)

    if len(seed_list) > 1:
        groups = {}
        for r in results:
            groups.setdefault((r["mode"], r["K"]), []).append(r)

        print("SUMMARY — aggregated over {} seeds ({})".format(
            len(seed_list), ", ".join(str(s) for s in seed_list)))
        print("=" * 74)
        print("| {:<11} | {:<4} | {:<3} | {:<18} | {:<18} | {:<22} |".format(
            "Mode", "K", "N", "Before (mean±std)", "After (mean±std)",
            "Delta (mean±std)",
        ))
        print("|" + "-"*13 + "|" + "-"*6 + "|" + "-"*5 + "|"
              + "-"*20 + "|" + "-"*20 + "|" + "-"*24 + "|")

        for (mode, K), rs in groups.items():
            befores = [x["acc_before"] for x in rs]
            afters = [x["acc_after"] for x in rs]
            deltas = [x["acc_after"] - x["acc_before"] for x in rs]
            if len(rs) > 1:
                m_b, s_b = mean(befores), stdev(befores)
                m_a, s_a = mean(afters), stdev(afters)
                m_d, s_d = mean(deltas), stdev(deltas)
            else:
                m_b = befores[0]; s_b = 0.0
                m_a = afters[0]; s_a = 0.0
                m_d = deltas[0]; s_d = 0.0
            print("| {:<11} | {:<4} | {:<3} | {:.4f} ± {:.4f}    | "
                  "{:.4f} ± {:.4f}    | {:+.4f} ± {:.4f}      |".format(
                      mode, K, len(rs),
                      m_b, s_b, m_a, s_a, m_d, s_d))
        print("=" * 74)

        print()
        print("Per-seed:")
        for r in results:
            d = r["acc_after"] - r["acc_before"]
            print("  seed={}  mode={}  K={}  before={:.4f}  "
                  "after={:.4f}  delta={:+.4f}".format(
                      r["seed"], r["mode"], r["K"],
                      r["acc_before"], r["acc_after"], d))
    else:
        print("SUMMARY (pair_acc on novel-class queries)")
        print("=" * 74)
        print("| {:<11} | {:<4} | {:<16} | {:<16} | {:<16} |".format(
            "Mode", "K", "Before adapt", "After adapt", "Delta"))
        print("|" + "-" * 13 + "|" + "-" * 6 + "|"
              + "-" * 18 + "|" + "-" * 18 + "|" + "-" * 18 + "|")
        for r in results:
            delta = r["acc_after"] - r["acc_before"]
            print("| {:<11} | {:<4} | {:<16} | {:<16} | {:+.4f}          |"
                  .format(r["mode"], r["K"],
                          "{:.4f}".format(r["acc_before"]),
                          "{:.4f}".format(r["acc_after"]),
                          delta))
        print("=" * 74)

    print("Elapsed: {:.1f}s".format(elapsed))
    if args.output_json:
        import json
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("Saved JSON:", args.output_json)

    # Interpretation
    print()
    print("How to read:")
    print("  - Baseline row: pair_acc without any adaptation (= reference).")
    print("  - Fixed/Generated rows: pair_acc after adapting side-block")
    print("    on K support images per novel class.")
    print("  - Delta > 0: side-block adaptation helps on new classes.")
    print("  - Delta ~ 0: adaptation is neutral (side-block adds nothing).")
    print("  - Delta < 0: adaptation actively hurts.")
    print()
    print("If generated.Delta > fixed.Delta noticeably (> 1 pp), the")
    print("hypernetwork is better at K-shot adaptation than a directly")
    print("trained kernel. If both Delta ~ 0, the mechanism is ineffective")
    print("even in the few-shot regime.")


if __name__ == "__main__":
    main()
