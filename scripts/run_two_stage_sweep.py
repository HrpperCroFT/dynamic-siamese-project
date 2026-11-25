"""Run two-stage verification across seeds and aggregate.

For each (mode, seed):
    python examples/train_cub.py --mode MODE --two-stage --seed SEED ...

Parses "Summary: stage1_baseline=X -> stage2=Y (delta = Z )"
(the space before ) is now tolerated) and prints mean +- std per mode.

Run:  python run_two_stage_sweep.py --seeds 1,2 --modes fixed,generated
      python run_two_stage_sweep.py --spatial --seeds 1,2 --modes generated
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)


STAGE_RE = re.compile(
    r"stage1_baseline=\s*([-\d.]+)\s*->\s*stage2=\s*([-\d.]+)"
    r"\s*\(delta\s*=\s*([-\d.]+)\s*\)"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--modes", type=str, default="fixed,generated")
    p.add_argument("--stage1-epochs", type=int, default=12)
    p.add_argument("--stage2-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-side", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--spatial", action="store_true",
                   help="use spatially-varying kernels")
    p.add_argument("--output", type=str,
                   default="two_stage_sweep_results.csv")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_one(script_path, mode, seed, args, dry_run):
    cmd = [
        sys.executable, str(script_path),
        "--mode", mode,
        "--two-stage",
        "--seed", str(seed),
        "--stage1-epochs", str(args.stage1_epochs),
        "--epochs", str(args.stage2_epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--lr-side", str(args.lr_side),
        "--patience", str(args.patience),
        "--data-root", args.data_root,
        "--num-workers", str(args.num_workers),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.spatial:
        cmd.append("--spatial")

    print()
    print("=" * 70)
    print("RUN:", mode, "/ seed=" + str(seed),
          "/ spatial=" + str(args.spatial))
    print("CMD:", " ".join(cmd))
    print("=" * 70)

    if dry_run:
        print("(dry-run: not executing)")
        return {"mode": mode, "seed": seed,
                "stage1": None, "stage2": None, "delta": None,
                "status": "DRY"}

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    full = "".join(lines)

    if proc.returncode != 0:
        print("FAILED (exit code " + str(proc.returncode) + ")")
        return {"mode": mode, "seed": seed,
                "stage1": None, "stage2": None, "delta": None,
                "status": "FAILED"}

    m = STAGE_RE.search(full)
    if not m:
        print("WARN: Summary line not matched by regex")
        return {"mode": mode, "seed": seed,
                "stage1": None, "stage2": None, "delta": None,
                "status": "PARSE_ERROR"}

    return {
        "mode": mode, "seed": seed,
        "stage1": float(m.group(1)),
        "stage2": float(m.group(2)),
        "delta": float(m.group(3)),
        "status": "OK",
    }


def fmt_ms(xs):
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return "{:.4f}".format(xs[0])
    return "{:.4f} +- {:.4f}".format(mean(xs), stdev(xs))


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    root = PROJECT_ROOT
    script = root / "examples" / "train_cub.py"
    if not script.exists():
        print("ERROR: not found:", script)
        sys.exit(1)

    print("Sweep config:")
    print("  seeds        :", seeds)
    print("  modes        :", modes)
    print("  spatial      :", args.spatial)
    print("  stage1 epochs:", args.stage1_epochs)
    print("  stage2 epochs:", args.stage2_epochs)
    print("  total runs   :", len(seeds) * len(modes))

    results = []
    for mode in modes:
        for seed in seeds:
            r = run_one(script, mode, seed, args, args.dry_run)
            results.append(r)

    if args.dry_run:
        print("\nDry-run complete.")
        return

    csv_path = root / args.output
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("mode,seed,spatial,stage1,stage2,delta,status\n")
        for r in results:
            f.write("{},{},{},{},{},{},{}\n".format(
                r["mode"], r["seed"], args.spatial,
                r["stage1"], r["stage2"], r["delta"], r["status"],
            ))
    print("\nResults saved to:", csv_path)

    print()
    print("=" * 78)
    print("SUMMARY (mean +- std over successful seeds)")
    print("=" * 78)
    print("| {:<11} | {:<3} | {:<20} | {:<20} | {:<20} |".format(
        "Mode", "N", "Stage1 (baseline)", "Stage2 (side)", "Delta",
    ))
    print("|" + "-" * 13 + "|" + "-" * 5 + "|"
          + "-" * 22 + "|" + "-" * 22 + "|" + "-" * 22 + "|")

    for mode in modes:
        ok = [r for r in results
              if r["mode"] == mode and r["status"] == "OK"]
        n_fail = len([r for r in results
                      if r["mode"] == mode and r["status"] != "OK"])
        s1 = [r["stage1"] for r in ok]
        s2 = [r["stage2"] for r in ok]
        d = [r["delta"] for r in ok]
        n_str = str(len(ok)) + ("/" + str(len(ok) + n_fail) if n_fail else "")
        print("| {:<11} | {:<3} | {:<20} | {:<20} | {:<20} |".format(
            mode, n_str, fmt_ms(s1), fmt_ms(s2), fmt_ms(d),
        ))
    print("=" * 78)


if __name__ == "__main__":
    main()
