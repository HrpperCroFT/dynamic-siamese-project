# run_few_shot_sweep.py
# Usage:
#   python run_few_shot_sweep.py
#       → seeds=1,2,3  K=5,10,15,20  modes=fixed,generated
#   python run_few_shot_sweep.py --seeds 1,2,3 --K-list 10,20,30
#   python run_few_shot_sweep.py --spatial --K-list 5,10
#
# Parses "pair_acc after adaptation: X (delta = Y)" from few_shot_eval.py
# stdout, aggregates mean ± std per (mode, K), saves CSV.

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)


DELTA_RE = re.compile(
    r"pair_acc after\s+adaptation:\s*([-\d.]+)\s*\(delta\s*=\s*([-+\d.]+)\)"
)
BEFORE_RE = re.compile(r"pair_acc before adaptation:\s*([-\d.]+)")
BASELINE_RE = re.compile(r"baseline pair_acc \(no adaptation\):\s*([-\d.]+)")


def fmt_ms(xs):
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return "{:.4f}".format(xs[0])
    return "{:.4f} ± {:.4f}".format(mean(xs), stdev(xs))


def run_once(eval_script, mode, K, seed, spatial, extra_args):
    cmd = [
        sys.executable, str(eval_script),
        "--skip-baseline-train",
        "--modes", mode,
        "--K-list", str(K),
        "--seed", str(seed),
    ]
    if spatial:
        cmd.append("--spatial")
    cmd.extend(extra_args)

    print()
    print("=" * 70)
    print("RUN: mode={} K={} seed={} spatial={}".format(
        mode, K, seed, spatial,
    ))
    print("CMD:", " ".join(cmd))
    print("=" * 70)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        return {"mode": mode, "K": K, "seed": seed,
                "before": None, "after": None, "delta": None,
                "status": "FAILED"}

    full = "".join(lines)
    m = DELTA_RE.search(full)
    b = BEFORE_RE.search(full)
    if not m:
        # baseline mode has different output
        bm = BASELINE_RE.search(full)
        if bm:
            v = float(bm.group(1))
            return {"mode": mode, "K": K, "seed": seed,
                    "before": v, "after": v, "delta": 0.0,
                    "status": "OK"}
        return {"mode": mode, "K": K, "seed": seed,
                "before": None, "after": None, "delta": None,
                "status": "PARSE_ERROR"}

    return {
        "mode": mode, "K": K, "seed": seed,
        "before": float(b.group(1)) if b else None,
        "after": float(m.group(1)),
        "delta": float(m.group(2)),
        "status": "OK",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--K-list", default="5,10,15,20")
    parser.add_argument("--modes", default="fixed,generated")
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--adapt-epochs", type=int, default=10)
    parser.add_argument("--lr-side", type=float, default=1e-3)
    parser.add_argument("--output", default="few_shot_sweep_results.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="",
                        help="empty = auto-detect")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    eval_script = Path(__file__).resolve().parent / "few_shot_eval.py"
    if not eval_script.exists():
        print("ERROR: few_shot_eval.py not found at", eval_script)
        sys.exit(1)

    baseline_ckpt = root / "baseline_cub.pt"
    if not baseline_ckpt.exists():
        print("WARNING: baseline_cub.pt not found. First run of few_shot_eval")
        print("will train it (extra ~10 min). Pass --skip-baseline-train")
        print("requires an existing checkpoint.")
        print()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    Ks = [int(k.strip()) for k in args.K_list.split(",") if k.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    extra = ["--adapt-epochs", str(args.adapt_epochs),
             "--lr-side", str(args.lr_side)]
    if args.device:
        extra.extend(["--device", args.device])

    print("Sweep:")
    print("  seeds        :", seeds)
    print("  K values     :", Ks)
    print("  modes        :", modes)
    print("  spatial      :", args.spatial)
    print("  adapt epochs :", args.adapt_epochs)
    print("  total runs   :", len(seeds) * len(Ks) * len(modes))

    if args.dry_run:
        print("\n(dry-run, nothing executed)")
        return

    results = []
    for mode in modes:
        for K in Ks:
            for seed in seeds:
                r = run_once(eval_script, mode, K, seed,
                             args.spatial, extra)
                results.append(r)

    # Save CSV
    csv_path = root / args.output
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("mode,K,seed,spatial,before,after,delta,status\n")
        for r in results:
            f.write("{},{},{},{},{},{},{},{}\n".format(
                r["mode"], r["K"], r["seed"], args.spatial,
                r["before"], r["after"], r["delta"], r["status"],
            ))
    print("\nResults saved to:", csv_path)

    # Aggregate
    print()
    print("=" * 88)
    print("SUMMARY (pair_acc on novel queries, mean ± std over seeds)")
    print("=" * 88)
    header = "| {:<11} | {:<4} | {:<7} | {:<16} | {:<16} | {:<18} |".format(
        "Mode", "K", "N", "Before", "After", "Delta",
    )
    print(header)
    print("|" + "-" * 13 + "|" + "-" * 6 + "|" + "-" * 9 + "|"
          + "-" * 18 + "|" + "-" * 18 + "|" + "-" * 20 + "|")

    for mode in modes:
        for K in Ks:
            ok = [r for r in results
                  if r["mode"] == mode and r["K"] == K and r["status"] == "OK"]
            if not ok:
                n_all = len([r for r in results
                             if r["mode"] == mode and r["K"] == K])
                print("| {:<11} | {:<4} | {:<7} | {:<16} | {:<16} | {:<18} |"
                      .format(mode, K, "0/" + str(n_all), "n/a", "n/a", "n/a"))
                continue
            bs = [r["before"] for r in ok if r["before"] is not None]
            a = [r["after"] for r in ok]
            d = [r["delta"] for r in ok]
            n_str = "{}/{}".format(len(ok), len(ok))
            print("| {:<11} | {:<4} | {:<7} | {:<16} | {:<16} | {:<18} |"
                  .format(mode, K, n_str,
                          fmt_ms(bs), fmt_ms(a), fmt_ms(d)))
    print("=" * 88)
    print()
    print("Key question: does generated.Delta grow with K while")
    print("fixed.Delta stays near 0 or negative? If yes, hypernetwork")
    print("generalises better. If both Delta trend downward with K, the")
    print("+3.1% at K=10 was noise.")


if __name__ == "__main__":
    main()
