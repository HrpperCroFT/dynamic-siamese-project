"""Baseline methods for few-shot adaptation comparison.

All baselines:
- Use baseline encoder (frozen unless stated)
- Return pair_accuracy (float) comparable to eval_pair_acc
- Record train_time_s and infer_time_s via profiler
"""
import random as _r
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .profiler import Timer, record


def _collect_embeddings(model, supports, queries, device):
    """Extract embeddings for all supports/queries (recorded as infer)."""
    with Timer(device) as t:
        with torch.no_grad():
            sup = {c: model.embed(torch.stack(supports[c]).to(device))
                   for c in supports}
            qry = {c: model.embed(torch.stack(queries[c]).to(device))
                   for c in queries}
    record("embed_time_s", t.time_s)
    return sup, qry


def _pair_accuracy_from_emb(emb_by_class, seed=0, pairs_per_class=5):
    rng = _r.Random(seed)
    classes = sorted(emb_by_class.keys())
    pos, neg = [], []
    for c in classes:
        q = emb_by_class[c]
        if len(q) < 2:
            continue
        ps = list(combinations(range(len(q)), 2))
        rng.shuffle(ps)
        for i, j in ps[:pairs_per_class]:
            pos.append((q[i], q[j]))
        others = [x for x in classes if x != c]
        for _ in range(pairs_per_class):
            if not others:
                break
            c2 = rng.choice(others)
            i = rng.randrange(len(q))
            j = rng.randrange(len(emb_by_class[c2]))
            neg.append((q[i], emb_by_class[c2][j]))
    if not pos or not neg:
        return float("nan")
    sp = torch.stack([F.cosine_similarity(a, b, dim=0) for a, b in pos])
    sn = torch.stack([F.cosine_similarity(a, b, dim=0) for a, b in neg])
    return (sp.unsqueeze(1) > sn.unsqueeze(0)).float().mean().item()


# ---------- Frozen encoder baselines ----------

def protonet_pair_accuracy(model, supports, queries, device,
                            pairs_per_class=5, seed=0, **kw):
    model.eval()
    sup, qry = _collect_embeddings(model, supports, queries, device)
    with Timer(device) as t:
        classes = sorted(supports)
        prototypes = torch.stack([
            F.normalize(sup[c].mean(dim=0), dim=0) for c in classes
        ])
        # A query's ProtoNet representation is its posterior over support
        # prototypes. Pair similarity is cosine similarity between those
        # posterior vectors, so the support set affects the score.
        projected = {}
        for c in classes:
            q = F.normalize(qry[c], dim=-1)
            probs = (20.0 * (q @ prototypes.T)).softmax(dim=-1)
            projected[c] = F.normalize(probs, dim=-1)
        acc = _pair_accuracy_from_emb(projected, seed=seed,
                                       pairs_per_class=pairs_per_class)
    record("infer_time_s", t.time_s)
    return acc


def linear_probe_pair_accuracy(model, supports, queries, device,
                                 epochs=30, lr=1e-2,
                                 pairs_per_class=5, seed=0, **kw):
    model.eval()
    classes = sorted(supports.keys())
    K, D = len(classes), model.emb_head.out_features
    sup, qry = _collect_embeddings(model, supports, queries, device)
    E_sup = torch.cat([sup[c] for c in classes], dim=0)
    y_sup = torch.cat([torch.full((len(sup[c]),), ci, dtype=torch.long,
                                    device=device)
                        for ci, c in enumerate(classes)])
    W = nn.Linear(D, K, bias=False).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=lr)
    with Timer(device) as t_train:
        for _ in range(epochs):
            loss = F.cross_entropy(W(E_sup), y_sup)
            opt.zero_grad(); loss.backward(); opt.step()
    record("train_time_s", t_train.time_s)
    with Timer(device) as t_inf:
        with torch.no_grad():
            proj = {c: F.normalize(W(qry[c]), dim=-1) for c in classes}
        acc = _pair_accuracy_from_emb(proj, seed=seed,
                                       pairs_per_class=pairs_per_class)
    record("infer_time_s", t_inf.time_s)
    return acc


def cosine_classifier_pair_accuracy(model, supports, queries, device,
                                      epochs=30, lr=1e-2, scale=10.0,
                                      pairs_per_class=5, seed=0, **kw):
    model.eval()
    classes = sorted(supports.keys())
    K, D = len(classes), model.emb_head.out_features
    sup, qry = _collect_embeddings(model, supports, queries, device)
    E_sup = torch.cat([sup[c] for c in classes], dim=0)
    y_sup = torch.cat([torch.full((len(sup[c]),), ci, dtype=torch.long,
                                    device=device)
                        for ci, c in enumerate(classes)])
    W = nn.Linear(D, K, bias=False).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=lr)
    with Timer(device) as t_train:
        for _ in range(epochs):
            Wn = F.normalize(W.weight, dim=-1)
            En = F.normalize(E_sup, dim=-1)
            logits = scale * En @ Wn.T
            loss = F.cross_entropy(logits, y_sup)
            opt.zero_grad(); loss.backward(); opt.step()
    record("train_time_s", t_train.time_s)
    with Timer(device) as t_inf:
        with torch.no_grad():
            Wn = F.normalize(W.weight, dim=-1)
            proj = {c: F.normalize(F.normalize(qry[c], dim=-1) @ Wn.T, dim=-1)
                    for c in classes}
        acc = _pair_accuracy_from_emb(proj, seed=seed,
                                       pairs_per_class=pairs_per_class)
    record("infer_time_s", t_inf.time_s)
    return acc


def relationnet_pair_accuracy(model, supports, queries, device,
                                epochs=30, lr=1e-3, hidden=64,
                                pairs_per_class=5, seed=0, **kw):
    model.eval()
    classes = sorted(supports.keys())
    D = model.emb_head.out_features
    sup, qry = _collect_embeddings(model, supports, queries, device)

    rng = _r.Random(0)
    pos, neg = [], []
    for c in classes:
        for i, j in combinations(range(len(sup[c])), 2):
            pos.append((sup[c][i], sup[c][j]))
    n_pos = len(pos)
    for _ in range(n_pos):
        c1, c2 = rng.sample(classes, 2)
        i = rng.randrange(len(sup[c1])); j = rng.randrange(len(sup[c2]))
        neg.append((sup[c1][i], sup[c2][j]))

    def inp(a, b):
        return torch.cat([a, b, (a - b).abs()])

    X = torch.stack([inp(a, b) for a, b in pos + neg])
    y = torch.tensor([1.0] * len(pos) + [0.0] * len(neg), device=device)
    head = nn.Sequential(nn.Linear(3 * D, hidden), nn.ReLU(),
                          nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    with Timer(device) as t_train:
        for _ in range(epochs):
            logits = head(X).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    record("train_time_s", t_train.time_s)

    rng2 = _r.Random(seed)
    qpos, qneg = [], []
    for c in classes:
        q = qry[c]
        if len(q) < 2:
            continue
        ps = list(combinations(range(len(q)), 2))
        rng2.shuffle(ps)
        for i, j in ps[:pairs_per_class]:
            qpos.append((q[i], q[j]))
        others = [x for x in classes if x != c]
        for _ in range(pairs_per_class):
            if not others:
                break
            c2 = rng2.choice(others)
            i = rng2.randrange(len(q))
            j = rng2.randrange(len(qry[c2]))
            qneg.append((q[i], qry[c2][j]))
    if not qpos or not qneg:
        return float("nan")
    with Timer(device) as t_inf:
        with torch.no_grad():
            Xp = torch.stack([inp(a, b) for a, b in qpos])
            Xn = torch.stack([inp(a, b) for a, b in qneg])
            sp = head(Xp).squeeze(-1)
            sn = head(Xn).squeeze(-1)
        acc = (sp.unsqueeze(1) > sn.unsqueeze(0)).float().mean().item()
    record("infer_time_s", t_inf.time_s)
    return acc


# ---------- Encoder-modifying baselines ----------

def bn_adapt_pair_accuracy(model, supports, queries, device,
                             pairs_per_class=5, seed=0, **kw):
    all_sup = [img for imgs in supports.values() for img in imgs]
    with Timer(device) as t_train:
        model.train()
        with torch.no_grad():
            for i in range(0, len(all_sup), 16):
                b = torch.stack(all_sup[i:i+16]).to(device)
                _ = model.tower(b)
    record("train_time_s", t_train.time_s)
    model.eval()
    _, qry = _collect_embeddings(model, supports, queries, device)
    with Timer(device) as t_inf:
        acc = _pair_accuracy_from_emb(qry, seed=seed,
                                       pairs_per_class=pairs_per_class)
    record("infer_time_s", t_inf.time_s)
    return acc


def finetune_pair_accuracy(model, supports, queries, device,
                             epochs=5, lr=1e-5,
                             pairs_per_class=5, seed=0, **kw):
    for p in model.tower.parameters():
        p.requires_grad = False
    for p in model.tower.layer4.parameters():
        p.requires_grad = True
    for p in model.emb_head.parameters():
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    classes = sorted(supports.keys())
    rng = _r.Random(0)
    model.train()
    with Timer(device) as t_train:
        for _ in range(epochs):
            pos, neg = [], []
            for c in classes:
                for i, j in combinations(range(len(supports[c])), 2):
                    pos.append((supports[c][i], supports[c][j]))
            n_pos = len(pos)
            for _ in range(n_pos):
                c1, c2 = rng.sample(classes, 2)
                i = rng.randrange(len(supports[c1]))
                j = rng.randrange(len(supports[c2]))
                neg.append((supports[c1][i], supports[c2][j]))
            all_pairs = ([(a, b, 1.0) for a, b in pos]
                         + [(a, b, 0.0) for a, b in neg])
            rng.shuffle(all_pairs)
            for i in range(0, len(all_pairs), 32):
                batch = all_pairs[i:i+32]
                x1 = torch.stack([b[0] for b in batch]).to(device)
                x2 = torch.stack([b[1] for b in batch]).to(device)
                t = torch.tensor([b[2] for b in batch],
                                 dtype=torch.float32, device=device)
                sim = model(x1, x2) * 5.0
                loss = F.binary_cross_entropy_with_logits(sim, t)
                opt.zero_grad(); loss.backward(); opt.step()
    record("train_time_s", t_train.time_s)

    for p in model.parameters():
        p.requires_grad = True
    model.eval()
    _, qry = _collect_embeddings(model, supports, queries, device)
    with Timer(device) as t_inf:
        acc = _pair_accuracy_from_emb(qry, seed=seed,
                                       pairs_per_class=pairs_per_class)
    record("infer_time_s", t_inf.time_s)
    return acc


# RelationGCN impl lives in relationgcn_baseline.py
from .relationgcn_baseline import (relationgcn_pair_accuracy,
                                   relationgcn_full_pair_accuracy)


BASELINE_ARCHS = ["protonet", "linear", "cosclf", "relationnet",
                  "bnadapt", "finetune", "relationgcn",
                  "relationgcn-full"]

DISPATCH = {
    "protonet": protonet_pair_accuracy,
    "linear": linear_probe_pair_accuracy,
    "cosclf": cosine_classifier_pair_accuracy,
    "relationnet": relationnet_pair_accuracy,
    "bnadapt": bn_adapt_pair_accuracy,
    "finetune": finetune_pair_accuracy,
    "relationgcn": relationgcn_pair_accuracy,
    "relationgcn-full": relationgcn_full_pair_accuracy,
}
