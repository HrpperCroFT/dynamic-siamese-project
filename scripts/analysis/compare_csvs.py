# compare_csvs.py
import argparse, csv
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)


def read_metrics(path):
    """Return {holdout: {mean, std, time, mem}} + arch name."""
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        mean_cols = [c for c in cols if c.endswith("_mean") and
                     ("total_time" not in c and "peak_mb" not in c
                      and "train_time" not in c and "infer_time" not in c)]
        if not mean_cols:
            return None, {}
        mean_col = mean_cols[0]
        arch = mean_col[:-5]
        std_col = arch + "_std"
        time_col = arch + "_total_time_s_mean"
        mem_col = arch + "_peak_mb_mean"
        out = {}
        for r in rdr:
            try:
                m = float(r.get(mean_col, ""))
            except (ValueError, TypeError):
                continue
            entry = {"mean": m}
            for key, col in [("std", std_col), ("time", time_col),
                             ("mem", mem_col)]:
                try:
                    entry[key] = float(r.get(col, ""))
                except (ValueError, TypeError):
                    entry[key] = None
            out[r["holdout"]] = entry
    return arch, out


def fmt(m, s, mode):
    if mode == "acc":
        return f"{m:.4f}" if s is None else f"{m:.4f}±{s:.3f}"
    if mode == "time":
        return f"{m:.0f}s"
    if mode == "mem":
        return f"{m:.0f}MB"
    return f"{m:.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, help="; separated CSV paths")
    p.add_argument("--labels", required=True, help="; separated labels")
    p.add_argument("--output", default="results/comparison.md")
    p.add_argument("--show", default="acc",
                   choices=["acc", "time", "mem", "all"])
    p.add_argument("--sort-by", default="avg",
                   help="'avg' or a fold name (e.g. 'cub')")
    args = p.parse_args()

    paths = args.inputs.split(";")
    labels = args.labels.split(";")
    if len(paths) != len(labels):
        print("ERROR: len(inputs) != len(labels)"); return

    data, all_folds = [], set()
    for p_, lbl in zip(paths, labels):
        arch, d = read_metrics(p_)
        if not d:
            print(f"  WARN: {p_} empty"); continue
        data.append((lbl, d)); all_folds.update(d.keys())

    folds = sorted(all_folds)

    def avg_acc(d):
        vs = [d[f]["mean"] for f in folds if f in d]
        return sum(vs) / len(vs) if vs else 0

    if args.sort_by == "avg":
        data.sort(key=lambda x: -avg_acc(x[1]))
    elif args.sort_by in folds:
        data.sort(key=lambda x: -x[1].get(args.sort_by, {"mean": 0})["mean"])

    def render(mode):
        header = f"{'Config':<15}" + "".join(f" {f:>12}" for f in folds)
        if mode == "acc" or mode == "all":
            header += f" {'avg':>12}"
        print(header)
        print("-" * len(header))
        for lbl, d in data:
            line = f"{lbl:<15}"
            avs = []
            for f in folds:
                if f in d:
                    m = d[f]["mean"]; s = d[f].get("std")
                    line += f" {fmt(m, s, mode):>12}"
                    avs.append(m)
                else:
                    line += f" {'—':>12}"
            if mode == "acc" or mode == "all":
                line += f" {avg_acc(d):>12.4f}"
            print(line)

    if args.show == "all":
        for mode in ["acc", "time", "mem"]:
            print(f"\n=== {mode.upper()} ===")
            render(mode); print()
    else:
        render(args.show)

    # markdown
    md = [f"# Comparison ({args.show})\n\n"]
    md.append("| Config | " + " | ".join(folds) + " | avg |\n")
    md.append("|" + "---|" * (2 + len(folds)) + "\n")
    for lbl, d in data:
        row = [f"| {lbl} "]
        for f in folds:
            if f in d:
                row.append(f"| {fmt(d[f]['mean'], d[f].get('std'), args.show)} ")
            else:
                row.append("| — ")
        row.append(f"| {avg_acc(d):.4f} |\n")
        md.append("".join(row))
    Path(args.output).write_text("".join(md), encoding="utf-8")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
