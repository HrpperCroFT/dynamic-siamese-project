"""Hyperparameter sweep for DynamicSiameseNet few-shot adaptation.

Phases (run in order):
  python sweep_hyperparams.py --phase lr
  python sweep_hyperparams.py --phase epochs --lr-side 5e-4
  python sweep_hyperparams.py --phase l2 --lr-side 5e-4 --adapt-epochs 5
  python sweep_hyperparams.py --phase final --seeds 1,2,3,4,5 \
      --lr-side 5e-4 --adapt-epochs 5 --kernel-scale-l2 1e-3
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


DELTA_RE = re.compile(
    r"pair_acc after\s+adaptation:\s*([-\d.]+)\s*\(delta\s*=\s*([-+\d.]+)\)")


def run_one(lr_side, epochs, ks_l2, heldout, seed, spatial):
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "few_shot_eval.py"),
        "--skip-baseline-train",
        "--modes", "generated",
        "--K-list", "10",
        "--seed", str(seed),
        "--lr-side", str(lr_side),
        "--adapt-epochs", str(epochs),
        "--kernel-scale-l2", str(ks_l2),
    ]
    if spatial:
        cmd.append("--spatial")
    if heldout:
        cmd.append("--early-stop-heldout")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        m = DELTA_RE.search(line)
        if m:
            return float(m.group(2))
    return None


def agg(deltas):
    good = [d for d in deltas if d is not None]
    if not good:
        return None, None, 0
    m = mean(good)
    s = stdev(good) if len(good) > 1 else 0.0
    return m, s, len(good)


def sweep_lr(seeds, spatial):
    print("=== Phase: lr-side sweep (epochs=10, ks_l2=0) ===\n")
    values = [1e-4, 3e-4, 5e-4, 1e-3, 2e-3]
    results = []
    for lr in values:
        print(f"--- lr={lr} ---")
        deltas = []
        for s in seeds:
            d = run_one(lr, 10, 0.0, False, s, spatial)
            deltas.append(d)
            print(f"  seed={s}  {d:+.4f}" if d is not None
                  else f"  seed={s}  FAILED")
        m, sd, n = agg(deltas)
        if m is not None:
            print(f"  MEAN: {m:+.4f} ± {sd:.4f} (n={n})\n")
            results.append((lr, m, sd))
    if results:
        results.sort(key=lambda x: -x[1])
        print(f"BEST: lr={results[0][0]}  ({results[0][1]:+.4f})")
        print(f"Next: python sweep_hyperparams.py --phase epochs "
              f"--lr-side {results[0][0]}")


def sweep_epochs(seeds, spatial, lr_side):
    print(f"=== Phase: adapt-epochs sweep (lr={lr_side}) ===\n")
    values = [3, 5, 10, 15, 20]
    results = []
    for ep in values:
        print(f"--- epochs={ep} ---")
        deltas = []
        for s in seeds:
            d = run_one(lr_side, ep, 0.0, False, s, spatial)
            deltas.append(d)
            print(f"  seed={s}  {d:+.4f}" if d is not None
                  else f"  seed={s}  FAILED")
        m, sd, n = agg(deltas)
        if m is not None:
            print(f"  MEAN: {m:+.4f} ± {sd:.4f} (n={n})\n")
            results.append((ep, m, sd))
    if results:
        results.sort(key=lambda x: -x[1])
        best_ep = results[0][0]
        print(f"BEST: epochs={best_ep}  ({results[0][1]:+.4f})")
        print(f"Next: python sweep_hyperparams.py --phase l2 "
              f"--lr-side {lr_side} --adapt-epochs {best_ep}")


def sweep_l2(seeds, spatial, lr_side, epochs):
    print(f"=== Phase: kernel-scale-l2 sweep "
          f"(lr={lr_side}, epochs={epochs}) ===\n")
    values = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    results = []
    for l2 in values:
        print(f"--- ks_l2={l2} ---")
        deltas = []
        for s in seeds:
            d = run_one(lr_side, epochs, l2, False, s, spatial)
            deltas.append(d)
            print(f"  seed={s}  {d:+.4f}" if d is not None
                  else f"  seed={s}  FAILED")
        m, sd, n = agg(deltas)
        if m is not None:
            print(f"  MEAN: {m:+.4f} ± {sd:.4f} (n={n})\n")
            results.append((l2, m, sd))
    if results:
        results.sort(key=lambda x: -x[1])
        best_l2 = results[0][0]
        print(f"BEST: ks_l2={best_l2}  ({results[0][1]:+.4f})")
        print(f"Next: python sweep_hyperparams.py --phase final "
              f"--lr-side {lr_side} --adapt-epochs {epochs} "
              f"--kernel-scale-l2 {best_l2}")


def final_eval(seeds, spatial, lr_side, epochs, ks_l2):
    print(f"=== Final: lr={lr_side}, epochs={epochs}, ks_l2={ks_l2} ===\n")
    print("--- without held-out ---")
    deltas = []
    for s in seeds:
        d = run_one(lr_side, epochs, ks_l2, False, s, spatial)
        deltas.append(d)
        print(f"  seed={s}  {d:+.4f}" if d is not None
              else f"  seed={s}  FAILED")
    m, sd, n = agg(deltas)
    if m is not None:
        print(f"  MEAN: {m:+.4f} ± {sd:.4f} (n={n})\n")

    print("--- with held-out early stopping ---")
    deltas_h = []
    for s in seeds:
        d = run_one(lr_side, epochs, ks_l2, True, s, spatial)
        deltas_h.append(d)
        print(f"  seed={s}  {d:+.4f}" if d is not None
              else f"  seed={s}  FAILED")
    mh, sdh, nh = agg(deltas_h)
    if mh is not None:
        print(f"  MEAN: {mh:+.4f} ± {sdh:.4f} (n={nh})\n")

    if m is not None and mh is not None:
        diff = mh - m
        print(f"Delta improvement from held-out: {diff:+.4f}")
        if diff > 0.005:
            print("  -> held-out early stopping HELPS")
        elif diff < -0.005:
            print("  -> held-out early stopping HURTS")
        else:
            print("  -> held-out early stopping makes no difference")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True,
                        choices=["lr", "epochs", "l2", "final"])
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--lr-side", type=float, default=1e-3)
    parser.add_argument("--adapt-epochs", type=int, default=10)
    parser.add_argument("--kernel-scale-l2", type=float, default=0.0)
    parser.add_argument("--spatial", action="store_true")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not (Path(__file__).resolve().parent / "few_shot_eval.py").exists():
        print("ERROR: few_shot_eval.py not found in current directory")
        sys.exit(1)

    if args.phase == "lr":
        sweep_lr(seeds, args.spatial)
    elif args.phase == "epochs":
        sweep_epochs(seeds, args.spatial, args.lr_side)
    elif args.phase == "l2":
        sweep_l2(seeds, args.spatial, args.lr_side, args.adapt_epochs)
    elif args.phase == "final":
        final_eval(seeds, args.spatial, args.lr_side,
                   args.adapt_epochs, args.kernel_scale_l2)


if __name__ == "__main__":
    main()
