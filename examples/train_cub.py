"""Train on locally-extracted CUB200-2011 for image similarity.

Prerequisites:
    Extract CUB_200_2011.tgz into <data-root>/ so that
    <data-root>/CUB_200_2011/images/ and images.txt exist.

Run joint (all from scratch, side-block is zero-init at start):
    python examples\train_cub.py --mode generated --epochs 20

Run two-stage (clean test: pretrain baseline, then freeze and train
only side-block):
    python examples\train_cub.py --mode generated --two-stage \
        --stage1-epochs 15 --epochs 10
"""
import argparse
import sys
from pathlib import Path

try:
    import dynamic_siamese
except ImportError:
    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent / "src"),
    )

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from dynamic_siamese import (
    DynamicSiameseNet, SiameseBaseline, CUB200MetricLearning,
    MPerClassBatchSampler,
)
from dynamic_siamese.utils import pair_accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--mode", choices=["generated", "fixed", "baseline"],
                   default="generated")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--stage1-epochs", type=int, default=15,
                   help="epochs for stage 1 when --two-stage is set")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-side", type=float, default=1e-3,
                   help="lr for side-block during two-stage stage 2")
    p.add_argument("--m-per-class", type=int, default=4)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--eval-pairs", type=int, default=200)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["adam", "adamw"],
                   default="adamw")
    p.add_argument("--two-stage", action="store_true",
                   help="pretrain baseline, then freeze backbone and "
                        "train only side-block")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed for reproducibility")
    p.add_argument("--spatial", action="store_true",
                   help="use spatially-varying kernels (per-position)")
    return p.parse_args()


def make_optimizer(args, params):
    cls = AdamW if args.optimizer == "adamw" else Adam
    return cls(params, lr=args.lr, weight_decay=args.weight_decay)


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


def sample_eval_pairs(targets, num_pairs, seed=0):
    g = torch.Generator().manual_seed(seed)
    targets = torch.as_tensor(targets)
    N = len(targets)
    pos_pairs, neg_pairs = [], []
    tries = 0
    while len(pos_pairs) < num_pairs and tries < num_pairs * 20:
        tries += 1
        i = torch.randint(N, (1,), generator=g).item()
        same = (targets == targets[i]).nonzero(as_tuple=True)[0]
        same = same[same != i]
        if len(same) > 0:
            j = same[torch.randint(len(same), (1,), generator=g).item()].item()
            pos_pairs.append((i, j))
    tries = 0
    while len(neg_pairs) < num_pairs and tries < num_pairs * 20:
        tries += 1
        i = torch.randint(N, (1,), generator=g).item()
        diff = (targets != targets[i]).nonzero(as_tuple=True)[0]
        if len(diff) > 0:
            j = diff[torch.randint(len(diff), (1,), generator=g).item()].item()
            neg_pairs.append((i, j))
    return pos_pairs, neg_pairs


@torch.no_grad()
def evaluate(model, dataset, device, num_pairs, seed=0):
    model.eval()
    pos_pairs, neg_pairs = sample_eval_pairs(
        dataset.targets, num_pairs, seed,
    )
    if not pos_pairs or not neg_pairs:
        return {"pair_acc": float("nan")}

    def load(pairs):
        i_idx = torch.tensor([a for a, _ in pairs])
        j_idx = torch.tensor([b for _, b in pairs])
        x1 = torch.stack([dataset[i][0] for i in i_idx]).to(device)
        x2 = torch.stack([dataset[i][0] for i in j_idx]).to(device)
        return x1, x2

    x1p, x2p = load(pos_pairs)
    x1n, x2n = load(neg_pairs)
    sim_pos = model(x1p, x2p)
    sim_neg = model(x1n, x2n)
    return {
        "pair_acc": pair_accuracy(sim_pos, sim_neg),
        "sim_pos": sim_pos.mean().item(),
        "sim_neg": sim_neg.mean().item(),
    }


def run_epochs(model, loader, optimizer, device, epochs, eval_ds,
               args, tag="", freeze_params=None, train_side_only=False):
    """Train loop with early stopping. If train_side_only, only side-block
    + kernel_gen are optimized (others frozen). Returns best_state."""
    best_acc = -1.0
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        if train_side_only:
            model.tower_a.eval()
            model.emb_head_a.eval()
            model.emb_head_b.eval()

        total_loss, n_batches = 0.0, 0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pos_idx, neg_idx = build_pairs(labels)
            pos_idx = pos_idx.to(device)
            neg_idx = neg_idx.to(device)

            sim_pos_12 = model(imgs, imgs[pos_idx])
            sim_pos_21 = model(imgs[pos_idx], imgs)
            sim_neg_12 = model(imgs, imgs[neg_idx])
            sim_neg_21 = model(imgs[neg_idx], imgs)

            scores = torch.cat([
                sim_pos_12, sim_pos_21, sim_neg_12, sim_neg_21,
            ]) * 5.0
            targets_bce = torch.cat([
                torch.ones_like(sim_pos_12),
                torch.ones_like(sim_pos_21),
                torch.zeros_like(sim_neg_12),
                torch.zeros_like(sim_neg_21),
            ])
            loss = F.binary_cross_entropy_with_logits(scores, targets_bce)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        metrics = evaluate(model, eval_ds, device, args.eval_pairs)
        line = (tag + "Epoch " + str(epoch)
                + " | loss=" + str(round(total_loss / max(n_batches, 1), 4))
                + " | val_pair_acc=" + str(round(metrics["pair_acc"], 3))
                + " | sim_pos=" + str(round(metrics.get("sim_pos", 0), 3))
                + " | sim_neg=" + str(round(metrics.get("sim_neg", 0), 3)))
        if hasattr(model, "side_block"):
            if hasattr(model.side_block, "side_scale"):
                line += (" | side_scale="
                         + str(round(model.side_block.side_scale.item(), 5)))
            if hasattr(model.side_block, "kernel_scale"):
                line += (" | kernel_scale="
                         + str(round(model.side_block.kernel_scale.item(), 5)))
        print(line)

        if metrics["pair_acc"] > best_acc:
            best_acc = metrics["pair_acc"]
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(tag + "Early stop at epoch " + str(epoch)
                      + " | best val_pair_acc=" + str(round(best_acc, 3)))
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


def main():
    args = parse_args()

    import random
    import numpy as np
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print("Device:", device)
    print("Seed:", args.seed)
    print("Mode:", args.mode, "| Optimizer:", args.optimizer,
          "| Two-stage:", args.two_stage)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = CUB200MetricLearning(
        args.data_root, split="train", transform=train_tf,
    )
    test_ds = CUB200MetricLearning(
        args.data_root, split="test", transform=eval_tf,
    )
    print("Train samples:", len(train_ds),
          "| Test samples:", len(test_ds))

    sampler = MPerClassBatchSampler(
        train_ds.targets, m=args.m_per_class,
        batch_size=args.batch_size, seed=0,
    )
    loader = DataLoader(
        train_ds, batch_sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
    )

    if args.two_stage:
        if args.mode == "baseline":
            print("ERROR: --two-stage is meaningless with --mode baseline")
            sys.exit(1)

        # ---------- Stage 1 ----------
        print()
        print("=== Stage 1: pretrain baseline encoder ===")
        baseline = SiameseBaseline(
            embedding_dim=args.embedding_dim, pretrained=True,
        ).to(device)
        opt1 = make_optimizer(args, baseline.parameters())
        acc1 = run_epochs(
            baseline, loader, opt1, device,
            args.stage1_epochs, test_ds, args,
            tag="[stage1] ",
        )
        print("[stage1] best val_pair_acc=", round(acc1, 3))

        # ---------- Stage 2 ----------
        print()
        print("=== Stage 2: freeze backbone, train side-block only ===")
        model = DynamicSiameseNet(
            embedding_dim=args.embedding_dim,
            generate_kernels=(args.mode == "generated"),
            pretrained=False,
            share_backbone=True,
            side_channels=64,
            spatial=getattr(args, "spatial", False),
        ).to(device)

        model.tower_a.load_state_dict(baseline.tower.state_dict())
        model.emb_head_a.load_state_dict(baseline.emb_head.state_dict())
        model.emb_head_b.load_state_dict(baseline.emb_head.state_dict())

        for p in model.tower_a.parameters():
            p.requires_grad = False
        for p in model.emb_head_a.parameters():
            p.requires_grad = False
        for p in model.emb_head_b.parameters():
            p.requires_grad = False

        side_params = list(model.side_block.parameters())
        n_train = sum(p.numel() for p in side_params if p.requires_grad)
        print("[stage2] trainable side-block params:",
              round(n_train / 1e6, 3), "M")

        opt2 = (AdamW if args.optimizer == "adamw" else Adam)(
            side_params, lr=args.lr_side, weight_decay=args.weight_decay,
        )
        acc2 = run_epochs(
            model, loader, opt2, device,
            args.epochs, test_ds, args,
            tag="[stage2] ", train_side_only=True,
        )
        print("[stage2] best val_pair_acc=", round(acc2, 3))
        print()
        print("Summary: stage1_baseline=", round(acc1, 3),
              "-> stage2=", round(acc2, 3),
              "(delta =", round(acc2 - acc1, 4), ")")
        return

    # ---------- Joint training (single stage) ----------
    if args.mode == "baseline":
        model = SiameseBaseline(
            embedding_dim=args.embedding_dim, pretrained=True,
        )
    else:
        model = DynamicSiameseNet(
            embedding_dim=args.embedding_dim,
            generate_kernels=(args.mode == "generated"),
            pretrained=True,
            share_backbone=True,
            side_channels=64,
            spatial=getattr(args, "spatial", False),
        )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("Model params:", round(n_params, 2), "M")

    optimizer = make_optimizer(args, model.parameters())
    best_acc = run_epochs(
        model, loader, optimizer, device, args.epochs, test_ds, args,
    )
    print("Best val_pair_acc=", round(best_acc, 3))


if __name__ == "__main__":
    main()
