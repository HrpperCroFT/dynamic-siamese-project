
"""Prototypical Networks baseline for few-shot evaluation."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def protonet_accuracy(model, supports, queries, device):
    """Compute ProtoNet accuracy on query set.

    Args:
        model:    encoder with embed(x) -> (B, D) method, or SiameseBaseline
        supports: dict {class_id: [tensor(K, 3, H, W), ...]}
        queries:  dict {class_id: [tensor(M, 3, H, W), ...]}
        device:   torch device

    Returns:
        accuracy (float), number of correct / total
    """
    model.eval()
    classes = sorted(supports.keys())

    # compute prototypes
    prototypes = {}
    for c in classes:
        imgs = torch.stack(supports[c]).to(device)  # (K, 3, H, W)
        embs = model.embed(imgs)                    # (K, D)
        prototypes[c] = embs.mean(dim=0)             # (D,)

    proto_tensor = torch.stack([prototypes[c] for c in classes])  # (C, D)

    correct, total = 0, 0
    for c in classes:
        imgs = torch.stack(queries[c]).to(device)   # (M, 3, H, W)
        embs = model.embed(imgs)                    # (M, D)
        dists = torch.cdist(embs, proto_tensor)     # (M, C)
        preds = dists.argmin(dim=1)                 # (M,)
        true_idx = classes.index(c)
        correct += (preds == true_idx).sum().item()
        total += len(preds)

    return correct / max(total, 1)


@torch.no_grad()
def protonet_pair_accuracy(model, supports, queries, device,
                           pairs_per_class=5, seed=0):
    """Pair accuracy from agreement between ProtoNet class posteriors.

    Each query is embedded as its soft assignment to support prototypes.
    The dot product between two assignments is their symmetric similarity.
    """
    import random as _random
    import itertools
    model.eval()
    classes = sorted(supports.keys())

    # Precompute all embeddings
    all_embs = {}
    for c in classes:
        sup = torch.stack(supports[c]).to(device)
        qry = torch.stack(queries[c]).to(device)
        all_embs[c] = {
            "sup": model.embed(sup),
            "qry": model.embed(qry),
        }

    proto = torch.stack([
        F.normalize(all_embs[c]["sup"].mean(dim=0), dim=0)
        for c in classes
    ])
    rng = _random.Random(seed)
    probabilities = {}
    for c in classes:
        q = F.normalize(all_embs[c]["qry"], dim=-1)
        logits = 20.0 * (q @ proto.T)
        probabilities[c] = logits.softmax(dim=-1)

    pos_pairs, neg_pairs = [], []
    for c in classes:
        q = probabilities[c]
        if len(q) < 2:
            continue
        pairs = list(itertools.combinations(range(len(q)), 2))
        rng.shuffle(pairs)
        for i, j in pairs[:pairs_per_class]:
            pos_pairs.append((q[i], q[j]))
        others = [x for x in classes if x != c]
        for _ in range(pairs_per_class):
            c2 = rng.choice(others)
            i = rng.randrange(len(q))
            j = rng.randrange(len(all_embs[c2]["qry"]))
            neg_pairs.append((q[i], all_embs[c2]["qry"][j]))

    if not pos_pairs or not neg_pairs:
        return float("nan")

    sp = torch.stack([(a * b).sum() for a, b in pos_pairs])
    sn = torch.stack([(a * b).sum() for a, b in neg_pairs])
    return (sp.unsqueeze(1) > sn.unsqueeze(0)).float().mean().item()
