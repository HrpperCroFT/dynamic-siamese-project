
"""Support-guided spatial feature enhancement (frozen-backbone).

For each pair, the reference feature map generates per-position depthwise
kernels that enhance the candidate feature map. Pair scoring and support
adaptation are symmetric. This is a task-adapted RelationGCN-style baseline,
not a reproduction of the original RelationGCN training protocol.
"""
import itertools
import random as _r

import torch
import torch.nn as nn
import torch.nn.functional as F
from .profiler import Timer, record


class SupportGuidedEnhancer(nn.Module):
    """Generate per-position depthwise kernels from a reference feature map."""

    def __init__(self, in_channels=256, hidden=64, kernel_size=3,
                 n_kernels=4):
        super().__init__()
        self.C = in_channels
        self.k = kernel_size
        self.n_kernels = n_kernels
        kk = kernel_size * kernel_size

        # channel attention on reference
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels // 8, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // 8, 8), in_channels, 1),
            nn.Sigmoid(),
        )
        # Preserve spatial positions in the generated kernels. A global
        # pooled generator of full CxC kernels would require >150M weights.
        self.kernel_gen = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_kernels * in_channels * kk, 1),
        )
        # mixture weights (softmax)
        self.mix = nn.Parameter(torch.zeros(n_kernels))
        # output gate: zero-init → identity at start
        self.out_scale = nn.Parameter(torch.zeros(1))

    def forward(self, ref_feat, tgt_feat):
        # ref_feat, tgt_feat: (B, C, H, W)
        B, C, H, W = tgt_feat.shape
        if ref_feat.shape != tgt_feat.shape:
            raise ValueError("reference and target feature maps must match")
        r = ref_feat * self.attn(ref_feat)
        kernels = self.kernel_gen(r).view(
            B, self.n_kernels, C, self.k * self.k, H * W,
        )
        patches = F.unfold(tgt_feat, kernel_size=self.k,
                           padding=self.k // 2)
        patches = patches.view(B, C, self.k * self.k, H * W)

        w = torch.softmax(self.mix, dim=0)
        responses = (patches.unsqueeze(1) * kernels).sum(dim=3)
        out = torch.einsum("n,bncl->bcl", w, responses)
        return tgt_feat + self.out_scale * out.view(B, C, H, W)


def _tower_to_layer3(tower, x):
    h = tower.stem(x)
    h = tower.layer1(h)
    h = tower.layer2(h)
    h = tower.layer3(h)
    return h


def _tower_from_layer3(tower, h, emb_head):
    z = tower.layer4(h)
    z = tower.pool(z).flatten(1)
    return F.normalize(emb_head(z), dim=-1)


class FullChannelSupportGuidedEnhancer(nn.Module):
    """Reference-conditioned dense CxC kernels from the older design."""

    def __init__(self, in_channels=256, hidden=64, kernel_size=3,
                 n_kernels=4):
        super().__init__()
        self.C = in_channels
        self.k = kernel_size
        self.n_kernels = n_kernels
        reduced = max(in_channels // 8, 8)
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, reduced, 1), nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_channels, 1), nn.Sigmoid(),
        )
        kernel_values = n_kernels * in_channels**2 * kernel_size**2
        self.kernel_gen = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, kernel_values),
        )
        self.mix = nn.Parameter(torch.zeros(n_kernels))
        self.out_scale = nn.Parameter(torch.zeros(1))

    def forward(self, ref_feat, tgt_feat):
        B, C, H, W = tgt_feat.shape
        if ref_feat.shape != tgt_feat.shape:
            raise ValueError("reference and target feature maps must match")
        ref = ref_feat * self.attn(ref_feat)
        pooled = F.adaptive_avg_pool2d(ref, 1).flatten(1)
        kernels = self.kernel_gen(pooled).view(
            B, self.n_kernels, C, C, self.k, self.k,
        )
        # Apply each sample's dense CxC kernel independently via grouped conv.
        x = tgt_feat.reshape(1, B * C, H, W)
        mix = torch.softmax(self.mix, dim=0)
        out = torch.zeros_like(tgt_feat)
        for i in range(self.n_kernels):
            weight = kernels[:, i].reshape(B * C, C, self.k, self.k)
            response = F.conv2d(x, weight, padding=self.k // 2, groups=B)
            out = out + mix[i] * response.reshape(B, C, H, W)
        return tgt_feat + self.out_scale * out


def relationgcn_full_pair_accuracy(model, supports, queries, device,
                                   ft_epochs=15, lr=1e-3, hidden=64,
                                   n_kernels=4, kernel_size=3,
                                   pairs_per_class=5, seed=0, **kw):
    """Evaluate the older dense-channel RelationGCN-style adaptation."""
    model.eval()
    tower, head = model.tower, model.emb_head
    classes = sorted(supports.keys())
    enh = FullChannelSupportGuidedEnhancer(
        in_channels=256, hidden=hidden, kernel_size=kernel_size,
        n_kernels=n_kernels,
    ).to(device)
    opt = torch.optim.Adam(enh.parameters(), lr=lr)
    for p in model.parameters():
        p.requires_grad = False

    rng = _r.Random(0)
    pair_indices = []
    for c in classes:
        pair_indices.extend((c, i, c, j, 1.0)
                            for i, j in itertools.combinations(
                                range(len(supports[c])), 2))
    for _ in range(max(len(pair_indices), 200)):
        c1, c2 = rng.sample(classes, 2)
        i, j = rng.randrange(len(supports[c1])), rng.randrange(len(supports[c2]))
        pair_indices.append((c1, i, c2, j, 0.0))

    with Timer(device) as train_timer:
        features = {}
        with torch.no_grad():
            for c in classes:
                batches = []
                for start in range(0, len(supports[c]), 32):
                    x = torch.stack(supports[c][start:start + 32]).to(device)
                    batches.append(_tower_to_layer3(tower, x))
                features[c] = torch.cat(batches)
        pairs = [(features[c1][i], features[c2][j], y)
                 for c1, i, c2, j, y in pair_indices]

        def symmetric_score(fa, fb):
            ea = _tower_from_layer3(tower, enh(fb, fa), head)
            eb = _tower_from_layer3(tower, enh(fa, fb), head)
            return (ea * eb).sum(-1)

        for _ in range(ft_epochs):
            rng.shuffle(pairs)
            for start in range(0, len(pairs), 16):
                batch = pairs[start:start + 16]
                f1 = torch.stack([p[0] for p in batch])
                f2 = torch.stack([p[1] for p in batch])
                target = torch.tensor([p[2] for p in batch],
                                      dtype=torch.float32, device=device)
                logits = symmetric_score(f1, f2) * 5.0
                loss = F.binary_cross_entropy_with_logits(logits, target)
                opt.zero_grad(); loss.backward(); opt.step()
    record("train_time_s", train_timer.time_s)

    enh.eval()
    with Timer(device) as infer_timer, torch.no_grad():
        query_features = {}
        for c in classes:
            batches = []
            for start in range(0, len(queries[c]), 32):
                x = torch.stack(queries[c][start:start + 32]).to(device)
                batches.append(_tower_to_layer3(tower, x))
            query_features[c] = torch.cat(batches)
        rng_eval = _r.Random(seed)
        positives, negatives = [], []
        for c in classes:
            q = query_features[c]
            if len(q) < 2:
                continue
            pairs = list(itertools.combinations(range(len(q)), 2))
            rng_eval.shuffle(pairs)
            positives.extend((q[i], q[j]) for i, j in pairs[:pairs_per_class])
            others = [other for other in classes if other != c]
            for _ in range(pairs_per_class):
                c2 = rng_eval.choice(others)
                negatives.append((q[rng_eval.randrange(len(q))],
                                  query_features[c2][rng_eval.randrange(
                                      len(query_features[c2]))]))

        def score_pairs(pairs):
            scores = []
            for start in range(0, len(pairs), 16):
                batch = pairs[start:start + 16]
                f1 = torch.stack([p[0] for p in batch])
                f2 = torch.stack([p[1] for p in batch])
                ea = _tower_from_layer3(tower, enh(f2, f1), head)
                eb = _tower_from_layer3(tower, enh(f1, f2), head)
                scores.append((ea * eb).sum(-1))
            return torch.cat(scores)

        if not positives or not negatives:
            return float("nan")
        sp, sn = score_pairs(positives), score_pairs(negatives)
    record("infer_time_s", infer_timer.time_s)
    return (sp.unsqueeze(1) > sn.unsqueeze(0)).float().mean().item()


def relationgcn_pair_accuracy(model, supports, queries, device,
                                ft_epochs=15, lr=1e-3, hidden=64,
                                n_kernels=4, kernel_size=3,
                                pairs_per_class=5, seed=0, **kw):
    """Train SupportGuidedEnhancer on support, eval on query (symmetric)."""
    model.eval()
    tower = model.tower
    head = model.emb_head
    classes = sorted(supports.keys())

    enh = SupportGuidedEnhancer(
        in_channels=256, hidden=hidden, kernel_size=kernel_size,
        n_kernels=n_kernels,
    ).to(device)
    opt = torch.optim.Adam(enh.parameters(), lr=lr)

    # freeze backbone (just in case)
    for p in model.parameters():
        p.requires_grad = False

    # ---- cache frozen layer3 features and build support pairs ----
    rng = _r.Random(0)
    support_pairs_idx = []
    for c in classes:
        support_pairs_idx.extend(
            (c, i, c, j, 1.0)
            for i, j in itertools.combinations(range(len(supports[c])), 2)
        )
    for _ in range(max(len(support_pairs_idx), 200)):
        c1, c2 = rng.sample(classes, 2)
        i = rng.randrange(len(supports[c1]))
        j = rng.randrange(len(supports[c2]))
        support_pairs_idx.append((c1, i, c2, j, 0.0))

    # ---- training: symmetric enhancement ----
    def embed_feature_pair_sym(fa, fb):
        """fa, fb: cached (B, 256, H, W) layer3 features."""
        ea = _tower_from_layer3(tower, enh(fb, fa), head)
        eb = _tower_from_layer3(tower, enh(fa, fb), head)
        return (ea * eb).sum(-1)

    with Timer(device) as train_timer:
        support_features = {}
        with torch.no_grad():
            for c in classes:
                image_batches = []
                imgs = supports[c]
                for start in range(0, len(imgs), 32):
                    x = torch.stack(imgs[start:start + 32]).to(device)
                    image_batches.append(_tower_to_layer3(tower, x))
                support_features[c] = torch.cat(image_batches)

        train_pairs = [
            (support_features[c1][i], support_features[c2][j], label)
            for c1, i, c2, j, label in support_pairs_idx
        ]
        for epoch in range(1, ft_epochs + 1):
            rng.shuffle(train_pairs)
            for i in range(0, len(train_pairs), 16):
                b = train_pairs[i:i + 16]
                f1 = torch.stack([p[0] for p in b])
                f2 = torch.stack([p[1] for p in b])
                t = torch.tensor([p[2] for p in b], dtype=torch.float32,
                                 device=device)
                sim = embed_feature_pair_sym(f1, f2) * 5.0
                loss = F.binary_cross_entropy_with_logits(sim, t)
                opt.zero_grad(); loss.backward(); opt.step()
    record("train_time_s", train_timer.time_s)

    # ---- eval: symmetric on query ----
    enh.eval()
    with Timer(device) as infer_timer, torch.no_grad():
        query_features = {}
        for c in classes:
            image_batches = []
            imgs = queries[c]
            for start in range(0, len(imgs), 32):
                x = torch.stack(imgs[start:start + 32]).to(device)
                image_batches.append(_tower_to_layer3(tower, x))
            query_features[c] = torch.cat(image_batches)

        rng2 = _r.Random(seed)
        qpos, qneg = [], []
        for c in classes:
            q = query_features[c]
            if len(q) < 2:
                continue
            ps = list(itertools.combinations(range(len(q)), 2))
            rng2.shuffle(ps)
            for i, j in ps[:pairs_per_class]:
                qpos.append((q[i], q[j]))
            others = [x for x in classes if x != c]
            for _ in range(pairs_per_class):
                c2 = rng2.choice(others)
                i = rng2.randrange(len(q))
                j = rng2.randrange(len(query_features[c2]))
                qneg.append((q[i], query_features[c2][j]))

        def batch_sim(pairs):
            out = []
            for i in range(0, len(pairs), 16):
                b = pairs[i:i + 16]
                f1 = torch.stack([p[0] for p in b])
                f2 = torch.stack([p[1] for p in b])
                out.append(embed_feature_pair_sym(f1, f2))
            return torch.cat(out)

        sp = batch_sim(qpos)
        sn = batch_sim(qneg)
    record("infer_time_s", infer_timer.time_s)
    return (sp.unsqueeze(1) > sn.unsqueeze(0)).float().mean().item()
