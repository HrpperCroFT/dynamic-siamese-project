
"""HyperHybridNet: nonspatial residual on l3 + hyper head on l4.

Orthogonal mechanisms:
- Nonspatial side_block corrects cand l3 features (kernels from z_ref).
- Hyper head learns a comparator on l4 features via kernels from ref l4.
Both zero-init → starts exactly as baseline.

Design rationale: multi-level residual failed due to gradient conflict
between adjacent side_blocks. Hybrid avoids this by putting the two
mechanisms on DIFFERENT layers (l3 vs l4) and DIFFERENT paths
(residual vs head).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import ResNetTower, SideBlock
from .hyper_comparator import MultiKernelGenerator, ReferenceConditionedHead


class HyperHybridNet(nn.Module):
    def __init__(self, n_kernels=2, side_ch=64, out_ch_per_kernel=16,
                 residual_levels=("l3",),
                 gen_hidden=128, head_hidden=128, pretrained=False,
                 embedding_dim=128):
        super().__init__()
        self.tower = ResNetTower(pretrained=pretrained)

        # 1. Nonspatial residual side_blocks at chosen levels
        self.residual_levels = tuple(residual_levels)
        self.levels = self.residual_levels  # alias for logging compat
        CH = {"l1": 64, "l2": 128, "l3": 256, "l4": 512}
        self.side_blocks = nn.ModuleDict()
        for lv in self.residual_levels:
            self.side_blocks[lv] = SideBlock(
                base_ch=CH[lv], side_ch=side_ch,
                context_dim=ResNetTower.LAYER4_CH,
                generate_kernels=True, gen_hidden_dim=32,
            )

        # 2. Hyper head at l4
        in_ch = ResNetTower.LAYER4_CH  # 512
        self.kernel_gen = MultiKernelGenerator(
            context_dim=512, in_ch_ref=in_ch, n_kernels=n_kernels,
            out_ch=out_ch_per_kernel, in_ch=in_ch, kernel_size=3,
            mode="mlp", hidden_dim=gen_hidden,
        )
        self.head = ReferenceConditionedHead(
            in_ch=in_ch, out_ch_per_kernel=out_ch_per_kernel,
            n_kernels=n_kernels, kernel_size=3, hidden_dim=head_hidden,
        )

        # 3. Shared emb head + combination scale (zero-init)
        self.emb_head = nn.Linear(512, embedding_dim)
        self.combine_alpha = nn.Parameter(torch.zeros(1))

    def _cand_forward(self, x, side_context):
        """Candidate path with residuals at chosen levels, then l4."""
        h = self.tower.stem(x)
        h = self.tower.layer1(h)
        if "l1" in self.residual_levels:
            h = self.side_blocks["l1"](h, side_context)
        h = self.tower.layer2(h)
        if "l2" in self.residual_levels:
            h = self.side_blocks["l2"](h, side_context)
        h = self.tower.layer3(h)
        if "l3" in self.residual_levels:
            h = self.side_blocks["l3"](h, side_context)
        h = self.tower.layer4(h)
        z = self.tower.pool(h).flatten(1)
        return z, h

    def forward(self, x_ref, x_cand):
        # ref path: clean
        z_ref, feats_ref_l4 = self.tower.forward_features(x_ref, level="l4")
        # cand path: nonspatial residual at l3, then l4
        z_cand, feats_cand_l4 = self._cand_forward(x_cand, z_ref)
        # hyper head on l4
        kernels = self.kernel_gen(feats_ref_l4, z_ref)
        head_out = self.head(feats_cand_l4, kernels)
        # combined
        e_ref = F.normalize(self.emb_head(z_ref), dim=-1)
        e_cand = F.normalize(self.emb_head(z_cand), dim=-1)
        cos = (e_ref * e_cand).sum(-1)
        return cos + self.combine_alpha * head_out

    def adapt_params(self):
        params = (list(self.kernel_gen.parameters())
                  + list(self.head.parameters())
                  + [self.combine_alpha])
        for lv in self.residual_levels:
            params += list(self.side_blocks[lv].parameters())
        return params
