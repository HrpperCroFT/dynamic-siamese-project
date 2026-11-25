"""Hyperparameter sweep for HyperComparisonNet (--arch hyper).

Each phase varies one hyperparameter group, passing current best via
CLI args. Reports absolute "after adaptation" acc (comparable to
baseline 0.8428) alongside delta.

Usage:
  python sweep_hyper.py --phase arch
  python sweep_hyper.py --phase level --arch hyper
  python sweep_hyper.py --phase capacity --arch hyper --feature-level l2
  python sweep_hyper.py --phase gen --arch hyper --feature-level l2
  python sweep_hyper.py --phase lr --arch hyper --feature-level l2
  python sweep_hyper.py --phase epochs --arch hyper --feature-level l2
  python sweep_hyper.py --phase final --arch hyper --feature-level l2 \
      --n-kernels 4 --out-ch 16 --kernel-gen-mode mlp \
      --lr-side 1e-3 --adapt-epochs 10 --seeds 1,2,3,4,5

Notes:
  - For --arch hyper-pure, "before adapt" is near 0.5 (random init),
    not baseline. Compare "after" to baseline 0.84 directly.
  - For --arch hyper, "before adapt" = baseline by construction
    (combine_alpha=0 at init).
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
from statistics import mean, stdev


BASELINE_ACC = 0.8428  # CUB novel K=10 reference
DELTA_RE = re.compile(
    r"pair_acc after\s+adaptation:\s*([-\d.]+)\s*\(delta\s*=\s*([-+\d.]+)\)")


def run_one(arch, feature_level, n_kernels, out_ch, gen_mode,
            lr_side, epochs, seed):
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "few_shot_eval.py"),
        "--skip-baseline-train",
        "--modes", "generated",
        "--arch", arch,
        "--K-list", "10",
        "--seed", str(seed),
        "--lr-side", str(lr_side),
        "--adapt-epochs", str(epochs),
        "--feature-level", feature_level,
        "--n-kernels", str(n_kernels),
        "--out-ch", str(out_ch),
        "--kernel-gen-mode", gen_mode,
        "--batch-size", "32",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("    [stderr tail]", (proc.stderr or "")[-200:])
        return None, None
    for line in proc.stdout.splitlines():
        m = DELTA_RE.search(line)
        if m:
            after = float(m.group(1))
            return after, after - BASELINE_ACC
    return None, None


def agg(after_list, vs_base_list):
    good = [(a, v) for a, v in zip(after_list, vs_base_list) if a is not None]
    if not good:
        return None, None, None, None, 0
    afters = [a for a, _ in good]
    vss = [v for _, v in good]
    m_a = mean(afters)
    s_a = stdev(afters) if len(afters) > 1 else 0.0
    m_v = mean(vss)
    s_v = stdev(vss) if len(vss) > 1 else 0.0
    return m_a, s_a, m_v, s_v, len(good)


def show(name, afters, vss):
    m_a, s_a, m_v, s_v, n = agg(afters, vss)
    if m_a is None:
        print(f"  {name:<24} ALL FAILED\n")
        return None, None
    print(f"  {name:<24} after={m_a:.4f} ± {s_a:.4f}  "
          f"vs_baseline={m_v:+.4f} ± {s_v:.4f}  (n={n})\n")
    return m_v, s_v


def run_config(arch, fl, nk, oc, gm, lr, ep, seeds, label):
    print(f"--- {label} ---")
    afters, vss = [], []
    for s in seeds:
        a, v = run_one(arch, fl, nk, oc, gm, lr, ep, s)
        afters.append(a)
        vss.append(v)
        if a is not None:
            print(f"  seed={s}  after={a:.4f}  vs_base={v:+.4f}")
        else:
            print(f"  seed={s}  FAILED")
    return show(label, afters, vss)


def phase_arch(args, seeds):
    print("=== Phase: arch (hyper vs hyper-pure) ===\n")
    for arch in ["hyper", "hyper-pure"]:
        run_config(arch, "l2", 4, 16, "mlp", 1e-3, 10, seeds,
                   f"arch={arch}")
    print("Next: pick better arch, then")
    print("  python sweep_hyper.py --phase level --arch <BEST>")


def phase_level(args, seeds):
    print(f"=== Phase: feature_level (arch={args.arch}) ===\n")
    for fl in ["l1", "l2", "l3", "l4"]:
        run_config(args.arch, fl, 4, 16, "mlp", 1e-3, 10, seeds,
                   f"feature_level={fl}")
    print(f"Next: python sweep_hyper.py --phase capacity "
          f"--arch {args.arch} --feature-level <BEST>")


def phase_capacity(args, seeds):
    print(f"=== Phase: capacity (arch={args.arch}, "
          f"fl={args.feature_level}) ===\n")
    configs = [(2, 8), (2, 16), (4, 16), (8, 16), (4, 32)]
    for nk, oc in configs:
        run_config(args.arch, args.feature_level, nk, oc, "mlp",
                   1e-3, 10, seeds, f"n_kernels={nk}, out_ch={oc}")
    print(f"Next: python sweep_hyper.py --phase gen "
          f"--arch {args.arch} --feature-level {args.feature_level} "
          f"--n-kernels <BEST> --out-ch <BEST>")


def phase_gen(args, seeds):
    print(f"=== Phase: kernel_gen_mode (arch={args.arch}) ===\n")
    for gm in ["mlp", "conv"]:
        run_config(args.arch, args.feature_level, args.n_kernels,
                   args.out_ch, gm, 1e-3, 10, seeds,
                   f"gen_mode={gm}")
    print(f"Next: python sweep_hyper.py --phase lr "
          f"--arch {args.arch} --feature-level {args.feature_level} "
          f"--n-kernels {args.n_kernels} --out-ch {args.out_ch} "
          f"--kernel-gen-mode <BEST>")


def phase_lr(args, seeds):
    print(f"=== Phase: lr-side (arch={args.arch}) ===\n")
    for lr in [1e-4, 3e-4, 1e-3, 3e-3]:
        run_config(args.arch, args.feature_level, args.n_kernels,
                   args.out_ch, args.kernel_gen_mode, lr, 10, seeds,
                   f"lr_side={lr}")
    print(f"Next: python sweep_hyper.py --phase epochs "
          f"--arch {args.arch} --feature-level {args.feature_level} "
          f"--n-kernels {args.n_kernels} --out-ch {args.out_ch} "
          f"--kernel-gen-mode {args.kernel_gen_mode} --lr-side <BEST>")


def phase_epochs(args, seeds):
    print(f"=== Phase: adapt-epochs (arch={args.arch}) ===\n")
    for ep in [3, 5, 10, 20]:
        run_config(args.arch, args.feature_level, args.n_kernels,
                   args.out_ch, args.kernel_gen_mode, args.lr_side, ep,
                   seeds, f"adapt_epochs={ep}")
    print(f"Next: python sweep_hyper.py --phase final "
          f"--arch {args.arch} --feature-level {args.feature_level} "
          f"--n-kernels {args.n_kernels} --out-ch {args.out_ch} "
          f"--kernel-gen-mode {args.kernel_gen_mode} "
          f"--lr-side {args.lr_side} --adapt-epochs <BEST> --seeds 1,2,3,4,5")


def phase_final(args, seeds):
    print(f"=== Final: arch={args.arch}, fl={args.feature_level}, "
          f"nk={args.n_kernels}, oc={args.out_ch}, "
          f"gen={args.kernel_gen_mode}, lr={args.lr_side}, "
          f"ep={args.adapt_epochs} ===\n")
    run_config(args.arch, args.feature_level, args.n_kernels,
               args.out_ch, args.kernel_gen_mode,
               args.lr_side, args.adapt_epochs, seeds, "FINAL")
    print(f"Reference: nonspatial DynamicSiameseNet = {BASELINE_ACC:.4f} "
          f"(+0.0338 over baseline)")
    print(f"Reference: spatial DynamicSiameseNet    = 0.8404 (-0.0024)")
    print(f"Reference: hyper (defaults)             = 0.8403 (-0.0025)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True,
                   choices=["arch", "level", "capacity", "gen",
                            "lr", "epochs", "final"])
    p.add_argument("--arch", default="hyper",
                   choices=["hyper", "hyper-pure"])
    p.add_argument("--feature-level", default="l2",
                   choices=["l1", "l2", "l3", "l4"])
    p.add_argument("--n-kernels", type=int, default=4)
    p.add_argument("--out-ch", type=int, default=16)
    p.add_argument("--kernel-gen-mode", default="mlp",
                   choices=["conv", "mlp"])
    p.add_argument("--lr-side", type=float, default=1e-3)
    p.add_argument("--adapt-epochs", type=int, default=10)
    p.add_argument("--seeds", default="1,2,3")
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not (Path(__file__).resolve().parent / "few_shot_eval.py").exists():
        print("ERROR: few_shot_eval.py not found")
        sys.exit(1)

    phases = {
        "arch": phase_arch,
        "level": phase_level,
        "capacity": phase_capacity,
        "gen": phase_gen,
        "lr": phase_lr,
        "epochs": phase_epochs,
        "final": phase_final,
    }
    phases[args.phase](args, seeds)


if __name__ == "__main__":
    main()
