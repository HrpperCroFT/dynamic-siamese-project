# merge_loocv_csvs.py
"""Merge two loocv CSVs by 'holdout', preserving all columns."""
import argparse
import csv
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        cols = rdr.fieldnames or []
    return rows, cols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="existing loocv.csv")
    p.add_argument("--extra", required=True, help="new loocv (relationgcn)")
    p.add_argument("--out", default="results/loocv_merged.csv")
    args = p.parse_args()

    base_rows, base_cols = read_csv(args.base)
    extra_rows, extra_cols = read_csv(args.extra)

    base_by_fold = {r["holdout"]: r for r in base_rows}
    extra_by_fold = {r["holdout"]: r for r in extra_rows}

    # union of columns, base first, then extra-only
    extra_only = [c for c in extra_cols if c not in base_cols]
    all_cols = base_cols + extra_only

    # for each fold, merge dicts
    all_folds = sorted(set(base_by_fold) | set(extra_by_fold))
    merged = []
    for fold in all_folds:
        row = {}
        if fold in base_by_fold:
            row.update(base_by_fold[fold])
        if fold in extra_by_fold:
            for c in extra_cols:
                if c == "holdout":
                    continue
                row[c] = extra_by_fold[fold].get(c, "")
        merged.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        for row in merged:
            w.writerow({c: row.get(c, "") for c in all_cols})

    print(f"merged: {len(merged)} folds, {len(all_cols)} cols")
    print(f"output: {args.out}")
    print(f"new columns from extra: {extra_only[:5]}"
          + ("..." if len(extra_only) > 5 else ""))


if __name__ == "__main__":
    main()
