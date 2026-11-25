
"""End-to-end models: encoder + adapter trained jointly on source.

Each model's forward(a, b) returns unscaled similarity (logit-like),
compatible with BCE-with-logits * 5 during training.

- baseline:      SiameseBaseline (encoder only)
- hyper:         HyperComparisonNet (shared backbone, enc+adapter)
- hyper-pure:    HyperComparisonNet without cosine residual
- hyper-sym:     HyperComparisonNet symmetric
- nonspatial:    DynamicSiameseNet
- hybrid:        HyperHybridNet (residual l3 + head l4)
- relationnet:   encoder + MLP head over [ea, eb, |ea-eb|]
- relationgcn:   encoder + per-position enhancer at layer3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import SiameseBaseline, ResNetTower, DynamicSiameseNet
from .hyper_comparator import HyperComparisonNet
from .hyper_hybrid import HyperHybridNet
from .relationgcn_baseline import SupportGuidedEnhancer


class RelationNetE2E(nn.Module):
    def __init__(self, pretrained=True, hidden=64):
        super().__init__()
        self.tower = ResNetTower(pretrained=pretrained)
        self.emb_head = nn.Linear(512, 128)
        self.head = nn.Sequential(
            nn.Linear(3 * 128, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def _embed(self, x):
        return F.normalize(self.emb_head(self.tower(x)), dim=-1)

    def forward(self, a, b):
        ea, eb = self._embed(a), self._embed(b)
        return self.head(torch.cat([ea, eb, (ea - eb).abs()], -1)).squeeze(-1)


class RelationGCNE2E(nn.Module):
    def __init__(self, pretrained=True, hidden=64, n_kernels=4):
        super().__init__()
        self.tower = ResNetTower(pretrained=pretrained)
        self.emb_head = nn.Linear(512, 128)
        self.enhancer = SupportGuidedEnhancer(
            in_channels=256, hidden=hidden, kernel_size=3, n_kernels=n_kernels)

    def _to_l3(self, x):
        h = self.tower.stem(x)
        h = self.tower.layer1(h); h = self.tower.layer2(h); h = self.tower.layer3(h)
        return h

    def _from_l3(self, h):
        z = self.tower.layer4(h)
        z = self.tower.pool(z).flatten(1)
        return F.normalize(self.emb_head(z), dim=-1)

    def forward(self, a, b):
        fa, fb = self._to_l3(a), self._to_l3(b)
        ea = self._from_l3(self.enhancer(fb, fa))
        eb = self._from_l3(self.enhancer(fa, fb))
        return (ea * eb).sum(-1)


def build_e2e_model(arch, pretrained=True, device="cuda"):
    if arch == "baseline":
        return SiameseBaseline(embedding_dim=128, pretrained=pretrained).to(device)
    if arch in ("hyper", "hyper-pure", "hyper-sym"):
        return HyperComparisonNet(
            n_kernels=2, feature_level="l4", kernel_gen_mode="mlp",
            out_ch_per_kernel=16, use_cosine_residual=(arch != "hyper-pure"),
            share_backbone=True, pretrained=pretrained,
            symmetric=(arch == "hyper-sym"),
        ).to(device)
    if arch == "nonspatial":
        return DynamicSiameseNet(
            embedding_dim=128, generate_kernels=True,
            pretrained=pretrained, share_backbone=True, spatial=False).to(device)
    if arch == "hybrid":
        return HyperHybridNet(
            n_kernels=2, side_ch=64, out_ch_per_kernel=16,
            residual_levels=("l3",), pretrained=pretrained).to(device)
    if arch == "relationnet":
        return RelationNetE2E(pretrained=pretrained).to(device)
    if arch == "relationgcn":
        return RelationGCNE2E(pretrained=pretrained).to(device)
    raise ValueError("Unknown e2e arch: " + arch)


E2E_ARCHS = ["baseline", "hyper", "hyper-pure", "hyper-sym",
              "nonspatial", "hybrid", "relationnet", "relationgcn"]


def load_baseline_encoder_into(model, ckpt, arch):
    """Load encoder weights from baseline_loo checkpoint into e2e model."""
    if arch == "baseline":
        model.tower.load_state_dict(ckpt["tower"])
        model.emb_head.load_state_dict(ckpt["head"])
    elif arch in ("hyper", "hyper-pure", "hyper-sym"):
        model.tower_a.load_state_dict(ckpt["tower"])
        if hasattr(model, "tower_b") and model.tower_b is not model.tower_a:
            model.tower_b.load_state_dict(ckpt["tower"])
        model.emb_head.load_state_dict(ckpt["head"])
    elif arch == "nonspatial":
        model.tower_a.load_state_dict(ckpt["tower"])
        if model.tower_b is not model.tower_a:
            model.tower_b.load_state_dict(ckpt["tower"])
        model.emb_head_a.load_state_dict(ckpt["head"])
        if model.emb_head_b is not model.emb_head_a:
            model.emb_head_b.load_state_dict(ckpt["head"])
    elif arch == "hybrid":
        for name in ["stem", "layer1", "layer2", "layer3", "layer4", "pool"]:
            src = {k[len(name) + 1:]: v for k, v in ckpt["tower"].items()
                   if k.startswith(name + ".")}
            if src:
                getattr(model.tower, name).load_state_dict(src)
        model.emb_head.load_state_dict(ckpt["head"])
    elif arch == "relationnet":
        model.tower.load_state_dict(ckpt["tower"])
        model.emb_head.load_state_dict(ckpt["head"])
    elif arch == "relationgcn":
        model.tower.load_state_dict(ckpt["tower"])
        model.emb_head.load_state_dict(ckpt["head"])
    return model
