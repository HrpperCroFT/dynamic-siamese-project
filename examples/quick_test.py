"""Smoke test: build all models and run forward + backward passes on
random tensors. No downloads.

Run: python examples\quick_test.py
"""
import sys
from pathlib import Path

try:
    import dynamic_siamese  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from dynamic_siamese import DynamicSiameseNet, SiameseBaseline


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    B = 4
    x1 = torch.randn(B, 3, 224, 224, device=device)
    x2 = torch.randn(B, 3, 224, 224, device=device)

    print()
    print("[1] DynamicSiameseNet (generated kernels)")
    m = DynamicSiameseNet(
        embedding_dim=128, generate_kernels=True, pretrained=False,
    ).to(device).eval()
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    with torch.no_grad():
        sim = m(x1, x2)
    print("    params:", round(n_params, 2), "M")
    print("    similarity shape:", tuple(sim.shape),
          "range:", round(sim.min().item(), 3), "..", round(sim.max().item(), 3))

    print()
    print("[2] DynamicSiameseNet (fixed kernel, ablation)")
    m = DynamicSiameseNet(
        embedding_dim=128, generate_kernels=False, pretrained=False,
    ).to(device).eval()
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    with torch.no_grad():
        sim = m(x1, x2)
    print("    params:", round(n_params, 2), "M")
    print("    similarity shape:", tuple(sim.shape))

    print()
    print("[3] SiameseBaseline")
    m = SiameseBaseline(embedding_dim=128, pretrained=False).to(device).eval()
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    with torch.no_grad():
        sim = m(x1, x2)
    print("    params:", round(n_params, 2), "M")
    print("    similarity shape:", tuple(sim.shape))

    print()
    print("[4] Gradient check (DynamicSiameseNet)")
    m = DynamicSiameseNet(
        embedding_dim=128, generate_kernels=True, pretrained=False,
    ).to(device).train()
    sim = m(x1, x2)
    loss = sim.mean()
    loss.backward()
    n_with_grad = sum(1 for p in m.parameters() if p.grad is not None)
    n_total = sum(1 for _ in m.parameters())
    print("    parameters with grad:", n_with_grad, "/", n_total)

    print()
    print("OK")


if __name__ == "__main__":
    main()
