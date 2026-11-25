"""Run all experiments in one go, save results, resume on interruption.

Usage:
    python scripts/run_all_experiments.py                    # full
    python scripts/run_all_experiments.py --fast             # quick run
    python scripts/run_all_experiments.py --arches core      # core methods
    python scripts/run_all_experiments.py --experiments loocv
    python scripts/run_all_experiments.py --force            # ignore cache
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
EXTRA_ARGS = {
    "hyper": ["--feature-level", "l4", "--n-kernels", "2",
              "--out-ch", "16", "--lr-side", "3e-4",
              "--adapt-epochs", "5"],
    "hyper-pure": ["--feature-level", "l4", "--n-kernels", "2",
                   "--out-ch", "16", "--lr-side", "3e-4",
                   "--adapt-epochs", "5"],
    "nonspatial": ["--lr-side", "3e-4", "--adapt-epochs", "5"],
    "hyper-sym": ["--feature-level", "l4", "--n-kernels", "2",
                  "--out-ch", "16", "--lr-side", "3e-4",
                  "--adapt-epochs", "5"],
    "hyper-multi": ["--levels", "l2,l3", "--side-ch", "32",
                    "--multi-hidden", "32",
                    "--lr-side", "3e-4", "--adapt-epochs", "5"],
    "hyper-hybrid": ["--n-kernels", "2", "--side-ch", "64",
                     "--out-ch", "16",
                     "--lr-side", "3e-4", "--adapt-epochs", "5"],
}
RELATIONNET_EPOCHS = 10
RELATIONNET_LR = 5e-4
LINEAR_EPOCHS = 30
LINEAR_LR = 1e-2
RESULTS.mkdir(exist_ok=True)

CORE_ARCHS = ["baseline", "protonet", "linear", "relationnet",
              "relationgcn", "bnadapt", "nonspatial",
              "hyper-pure", "hyper"]
EXTENDED_ARCHS = CORE_ARCHS + ["cosclf", "finetune"]
FINETUNE_COMBOS = ["finetune+baseline", "finetune+protonet",
                   "finetune+linear", "finetune+relationnet",
                   "finetune+relationgcn", "finetune+nonspatial",
                   "finetune+hyper-pure", "finetune+hyper",
                   "finetune+hyper-sym"]
ALL_ARCHS = EXTENDED_ARCHS + FINETUNE_COMBOS

ARCH_NAMES = {
    # arch -> (--arch value, --modes value)
    "baseline":    ("cosine",   "baseline"),
    "protonet":    ("protonet", "baseline"),
    "linear":      ("linear",   "baseline"),
    "cosclf":      ("cosclf",   "baseline"),
    "relationnet": ("relationnet", "generated"),
    "relationgcn": ("relationgcn", "baseline"),
    "bnadapt":     ("bnadapt",  "generated"),
    "finetune":    ("finetune", "baseline"),
    "nonspatial":  ("cosine",   "generated"),
    "hyper":       ("hyper",    "generated"),
    "hyper-pure":  ("hyper-pure", "generated"),
    "hyper-multi": ("hyper-multi", "generated"),
    "hyper-hybrid": ("hyper-hybrid", "generated"),
}


def _parse_arch(arch):
    """Support both 'finetune+X' and 'X+finetune' forms."""
    if arch.startswith("finetune+"):
        return arch[len("finetune+"):], True
    if arch.endswith("+finetune"):
        return arch[:-len("+finetune")], True
    return arch, False


def _run(cmd, log_path):
    """Run subprocess, tee output to log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  → {' '.join(cmd)}")
    with open(log_path, "w", encoding="utf-8") as log:
        import os as _os
        env = _os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 env=env)
        for line in proc.stdout:
            print("    " + line, end="")
            log.write(line)
            log.flush()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


def run_indomain(dataset, arches, seeds, force):
    out = RESULTS / f"indomain_{dataset}.csv"
    if out.exists() and not force:
        print(f"[SKIP] {out} exists")
        return
    rows = []
    seeds_str = ",".join(map(str, seeds))
    for arch in arches:
        base_arch, is_ft = _parse_arch(arch)
        if base_arch == "baseline":
            arch_val, mode_val = "cosine", "baseline"
        else:
            arch_val, mode_val = ARCH_NAMES.get(base_arch, ("cosine", "generated"))
        if is_ft:
            cmd_ft = ["--finetune"]
        else:
            cmd_ft = []
        json_path = RESULTS / f"_tmp_{dataset}_{arch}.json"
        cmd = [sys.executable, str(SCRIPT_DIR / "few_shot_eval.py"),
               "--dataset", dataset,
               "--arch", arch_val, "--modes", mode_val,
               "--seeds", seeds_str, "--K-list", "10",
               "--skip-baseline-train",
               "--output-json", str(json_path)]
        if arch == "relationnet":
            cmd.extend(["--rn-epochs", str(RELATIONNET_EPOCHS),
                        "--rn-lr", str(RELATIONNET_LR)])
        elif arch in ("linear", "cosclf"):
            cmd.extend(["--rn-epochs", str(LINEAR_EPOCHS),
                        "--rn-lr", str(LINEAR_LR)])
        cmd.extend(EXTRA_ARGS.get(base_arch, []))
        cmd.extend(cmd_ft)
        if dataset == "cars":
            cmd.extend(["--M-per-class", "10"])
        log_path = RESULTS / f"log_{dataset}_{arch}.txt"
        try:
            _run(cmd, log_path)
        except RuntimeError as e:
            print(f"  FAILED {dataset}/{arch}: {e}")
            continue
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            for r in data:
                rows.append({
                    "dataset": dataset, "arch": arch,
                    "seed": r.get("seed"),
                    "before": r["acc_before"],
                    "after": r["acc_after"],
                    "train_time_s": r.get("train_time_s", ""),
                    "infer_time_s": r.get("infer_time_s", ""),
                    "total_time_s": r.get("total_time_s", ""),
                    "peak_mb": r.get("peak_mb", ""),
                })
            json_path.unlink()
    if rows:
        has_res = any("peak_mb" in r for r in rows)
        cols = ["dataset", "arch", "seed", "before", "after"]
        if has_res:
            cols += ["train_time_s", "infer_time_s",
                     "total_time_s", "peak_mb"]
        with open(out, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                vals = [str(r.get(c, "")) for c in cols]
                f.write(",".join(vals) + "\n")
        print(f"[DONE] {out}")


def run_loocv(datasets, arches, seeds, force):
    out = RESULTS / "loocv.csv"
    if out.exists() and not force:
        print(f"[SKIP] {out} exists")
        return
    frozen = [a for a in arches if a != "baseline" and not _parse_arch(a)[1]]
    finetune_arches = [a for a in arches if _parse_arch(a)[1]]
    arches_for_loocv = frozen
    cmd = [sys.executable, str(SCRIPT_DIR / "cross_validate.py"),
           "--datasets", ",".join(datasets),
           "--arches", ",".join(arches_for_loocv),
           "--seeds", ",".join(map(str, seeds)),
           "--output", str(out)]
    log_path = RESULTS / "log_loocv.txt"
    if frozen:
        _run(cmd, log_path)
        print(f"[DONE] {out}")
    if finetune_arches:
        ft_bases = [_parse_arch(a)[0] for a in finetune_arches]
        out_ft = RESULTS / "loocv_finetune.csv"
        cmd_ft = [sys.executable, str(SCRIPT_DIR / "cross_validate.py"),
                  "--datasets", ",".join(datasets),
                  "--arches", ",".join(ft_bases),
                  "--seeds", ",".join(map(str, seeds)),
                  "--finetune",
                  "--output", str(out_ft)]
        _run(cmd_ft, RESULTS / "log_loocv_finetune.txt")
        print(f"[DONE] {out_ft}")


def generate_summary(arches, seeds, output="results/summary.md"):
    lines = ["# Experiment Results\n"]
    lines.append(f"Seeds: {seeds}\n")
    lines.append(f"Methods: {arches}\n\n")

    # In-domain
    lines.append("## In-domain few-shot (baseline trained on same dataset)\n\n")
    for ds in ["cub", "cars"]:
        p = RESULTS / f"indomain_{ds}.csv"
        if not p.exists():
            continue
        lines.append(f"### {ds.upper()}\n\n")
        lines.append("| Method | Before | After | Delta |\n")
        lines.append("|---|---|---|---|\n")
        import csv
        with open(p) as f:
            rdr = csv.DictReader(f)
            by_arch = {}
            for row in rdr:
                by_arch.setdefault(row["arch"], []).append(
                    (float(row["before"]), float(row["after"])))
            for arch in arches:
                if arch not in by_arch:
                    continue
                befores = [b for b, _ in by_arch[arch]]
                afters = [a for _, a in by_arch[arch]]
                m_b = sum(befores) / len(befores)
                m_a = sum(afters) / len(afters)
                lines.append(f"| {arch} | {m_b:.4f} | {m_a:.4f} | "
                             f"{m_a - m_b:+.4f} |\n")
        lines.append("\n")

    # LOOCV
    p = RESULTS / "loocv.csv"
    if p.exists():
        lines.append("## Cross-domain LOOCV\n\n")
        import csv
        with open(p) as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)
        # header
        hdr = ["| Fold | N | cross_base "] + [f"| {a} " for a in arches] + ["|"]
        lines.append("".join(hdr) + "\n")
        lines.append("|" + "---|" * (3 + len(arches)) + "\n")
        for r in rows:
            row = [f"| {r['holdout']} | {r['n_classes']} | "
                   f"{float(r['cross_baseline']):.4f} |"]
            for a in arches:
                key = a + "_mean"
                std_key = a + "_std"
                if key in r and r[key]:
                    row.append(f" {float(r[key]):.4f} ± "
                                f"{float(r[std_key]):.4f} |")
                else:
                    row.append(" — |")
            lines.append("".join(row) + "\n")
        lines.append("\n")

    # append finetune LOOCV if present
    ft_loocv = RESULTS / "loocv_finetune.csv"
    if ft_loocv.exists():
        lines.append("\n## Cross-domain LOOCV (finetune protocol)\n\n")
        import csv as _csv
        with open(ft_loocv) as f:
            rows = list(_csv.DictReader(f))
        ft_archs = sorted(set(r.get("hyper_mean") and "hyper" or "" for r in rows))
        ft_cols = [k[:-5] for k in rows[0].keys()
                    if k.endswith("_mean")] if rows else []
        lines.append("| Fold | N | cross_base |" +
                     "".join(f" {a} |" for a in ft_cols) + "\n")
        lines.append("|" + "---|" * (3 + len(ft_cols)) + "\n")
        for r in rows:
            row = [f"| {r['holdout']} | {r['n_classes']} | "
                   f"{float(r['cross_baseline']):.4f} |"]
            for a in ft_cols:
                k = a + "_mean"; ks = a + "_std"
                if k in r and r[k]:
                    row.append(f" {float(r[k]):.4f} ± "
                               f"{float(r[ks]):.4f} |")
                else:
                    row.append(" — |")
            lines.append("".join(row) + "\n")

    out = Path(output)
    out.write_text("".join(lines), encoding="utf-8")
    print(f"\n[DONE] {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arches", default="core",
                   choices=["core", "extended", "finetune", "all", "custom"])
    p.add_argument("--custom-arches", default=None)
    p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--fast", action="store_true",
                   help="2 seeds")
    p.add_argument("--experiments", default="indomain,loocv")
    p.add_argument("--force", action="store_true")
    p.add_argument("--summary-output", default="results/summary.md",
                   help="path for summary markdown")
    args = p.parse_args()

    if args.fast:
        seeds = [1, 2]
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    if args.arches == "core":
        arches = CORE_ARCHS
    elif args.arches == "extended":
        arches = EXTENDED_ARCHS
    elif args.arches == "finetune":
        arches = FINETUNE_COMBOS
    elif args.arches == "all":
        arches = ALL_ARCHS
    else:
        arches = [x.strip() for x in (args.custom_arches or "").split(",")
                   if x.strip()]

    print("=" * 70)
    print("Experiment orchestrator")
    print("=" * 70)
    print(f"Arches: {arches}")
    print(f"Seeds:  {seeds}")
    print(f"Exps:   {args.experiments}")
    print(f"Force:  {args.force}")
    print()

    exps = [x.strip() for x in args.experiments.split(",")]

    if "indomain" in exps:
        print("\n[In-domain]")
        for ds in ["cub", "cars"]:
            run_indomain(ds, arches, seeds, args.force)

    if "loocv" in exps:
        print("\n[LOOCV]")
        run_loocv(["cub", "cars", "dogs", "flowers", "nabirds", "aircraft"],
                  arches, seeds, args.force)

    generate_summary(arches, seeds, output=args.summary_output)


if __name__ == "__main__":
    main()
