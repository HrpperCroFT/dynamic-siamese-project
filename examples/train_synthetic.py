"""Train on a synthetic dataset. No downloads, finishes in a minute.

Run: python examples\train_synthetic.py --mode generated --epochs 15
"""
import argparse
import sys
from pathlib import Path

try:
    import dynamic_siamese  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from dynamic_siamese import (
    DynamicSiameseNet, SiameseBaseline, SyntheticSimilarityDataset,
)
from dynamic_siamese.utils import pair_accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["generated", "fixed", "baseline"],
                   default="generated")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--m-per-class", type=int, default=4)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--n-classes-total", type=int, default=50)
    p.add_argument("--samples-per-class", type=int, default=20)
    p.add_argument("--n-train-classes", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-pairs", type=int, default=300)
    p.add_argument("--patience", type=int, default=3,
                   help="early stop if no val improvement for this many epochs")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model(args):
    if args.mode == "baseline":
        return SiameseBaseline(
            embedding_dim=args.embedding_dim, pretrained=False,
        )
    return DynamicSiameseNet(
        embedding_dim=args.embedding_dim,
        generate_kernels=(args.mode == "generated"),
        pretrained=False,
        share_backbone=True,
        side_channels=64,
    )


def split_by_class(dataset, n_train_classes):
    train_idx, val_idx = [], []
    for i, t in enumerate(dataset.targets):
        if t < n_train_classes:
            train_idx.append(i)
        else:
            val_idx.append(i)
    return train_idx, val_idx


def build_pairs(labels):
    """For each sample, return positive and negative pair indices."""
    B = len(labels)
    lab = labels.detach().cpu()
    pos = torch.zeros(B, dtype=torch.long)
    neg = torch.zeros(B, dtype=torch.long)
    for i in range(B):
        same = (lab == lab[i]).nonzero(as_tuple=True)[0]
        same = same[same != i]
        if len(same) > 0:
            pos[i] = same[torch.randint(len(same), (1,)).item()].item()
        else:
            pos[i] = i
        diff = (lab != lab[i]).nonzero(as_tuple=True)[0]
        if len(diff) > 0:
            neg[i] = diff[torch.randint(len(diff), (1,)).item()].item()
        else:
            neg[i] = i
    return pos, neg


def sample_eval_pairs(targets, num_pairs, seed=0):
    """Sample positive and negative pairs for evaluation."""
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
def evaluate(model, dataset, indices, targets_t, device, num_pairs, seed=0):
    model.eval()
    sub_targets = [dataset.targets[i] for i in indices]
    pos_pairs, neg_pairs = sample_eval_pairs(sub_targets, num_pairs, seed)
    if not pos_pairs or not neg_pairs:
        return {"pair_acc": float("nan")}

    def load(pairs):
        i_idx = torch.tensor([indices[a] for a, _ in pairs])
        j_idx = torch.tensor([indices[b] for _, b in pairs])
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


class SubsetList(torch.utils.data.Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = indices
        self.targets = [base.targets[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


class MPerClassBatchSampler:
    """Minimal M-per-class sampler without external dependencies."""

    def __init__(self, targets, m, batch_size, seed=0):
        self.targets = list(targets)
        self.m = m
        self.batch_size = batch_size
        self.class_to_idx = {}
        for i, t in enumerate(self.targets):
            self.class_to_idx.setdefault(t, []).append(i)
        self.classes = list(self.class_to_idx.keys())
        self.n_per_batch = batch_size // m
        self.rng = torch.Generator().manual_seed(seed)

    def __iter__(self):
        cls_order = torch.randperm(
            len(self.classes), generator=self.rng,
        ).tolist()
        for i in range(0, len(cls_order) - self.n_per_batch + 1,
                       self.n_per_batch):
            batch = []
            for ci in cls_order[i:i + self.n_per_batch]:
                cls = self.classes[ci]
                idxs = self.class_to_idx[cls]
                m_local = min(self.m, len(idxs))
                chosen = torch.randperm(
                    len(idxs), generator=self.rng,
                )[:m_local].tolist()
                batch.extend(idxs[k] for k in chosen)
            if len(batch) >= self.m:
                yield batch


def main():
    args = parse_args()
    device = torch.device(args.device)
    print("Device:", device)
    print("Mode:", args.mode)

    full_ds = SyntheticSimilarityDataset(
        n_classes=args.n_classes_total,
        samples_per_class=args.samples_per_class,
        img_size=args.img_size,
        seed=args.seed,
    )
    train_idx, val_idx = split_by_class(full_ds, args.n_train_classes)
    train_ds = SubsetList(full_ds, train_idx)
    val_ds = SubsetList(full_ds, val_idx)

    print("Train samples:", len(train_ds), "| Val samples:", len(val_ds))

    batch_sampler = MPerClassBatchSampler(
        train_ds.targets, m=args.m_per_class,
        batch_size=args.batch_size, seed=args.seed,
    )
    loader = DataLoader(
        train_ds, batch_sampler=batch_sampler, num_workers=0,
    )

    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("Model params:", round(n_params, 2), "M")

    optimizer = Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)

    best_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
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

        metrics = evaluate(
            model, full_ds, val_idx, None, device,
            args.eval_pairs, seed=args.seed,
        )
        print(
            "Epoch", epoch,
            "| loss=", round(total_loss / max(n_batches, 1), 4),
            "| val_pair_acc=", round(metrics["pair_acc"], 3),
            "| sim_pos=", round(metrics.get("sim_pos", 0), 3),
            "| sim_neg=", round(metrics.get("sim_neg", 0), 3),
        )

        acc = metrics["pair_acc"]
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stop at epoch", epoch,
                      "| best val_pair_acc=", round(best_acc, 3))
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best checkpoint: val_pair_acc=", round(best_acc, 3))


if __name__ == "__main__":
    main()
