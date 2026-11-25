# run_residual_sweep.py
import argparse, subprocess, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

PRESETS = {
    "singles": ["l1", "l2", "l3", "l4"],
    "pairs":   ["l1,l2", "l1,l3", "l1,l4", "l2,l3", "l2,l4", "l3,l4"],
    "triples": ["l1,l2,l3", "l1,l2,l4", "l1,l3,l4", "l2,l3,l4"],
    "curated": ["l1", "l2", "l4",
                "l1,l2", "l1,l4", "l2,l4", "l3,l4",
                "l1,l2,l3", "l1,l2,l4", "l1,l3,l4", "l2,l3,l4",
                "l1,l2,l3,l4"],
    "all":     ["l1", "l2", "l3", "l4",
                "l1,l2", "l1,l3", "l1,l4", "l2,l3", "l2,l4", "l3,l4",
                "l1,l2,l3", "l1,l2,l4", "l1,l3,l4", "l2,l3,l4",
                "l1,l2,l3,l4"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="pairs", choices=list(PRESETS.keys()))
    p.add_argument("--combos", default=None,
                   help="override, e.g. 'l1,l2;l1,l4'")
    p.add_argument("--datasets", default="cub,cars,dogs,flowers,dtd,omniglot")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--adapt-epochs", type=int, default=5)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip", action="store_true")
    args = p.parse_args()

    combos = args.combos.split(";") if args.combos else PRESETS[args.preset]
    results = Path("results"); results.mkdir(exist_ok=True)

    print(f"Sequential sweep: {len(combos)} combos")
    for k in ["datasets", "seeds", "adapt_epochs"]:
        print(f"  {k}: {getattr(args, k)}")
    print()

    all_csvs = []
    for i, combo in enumerate(combos, 1):
        slug = combo.replace(",", "_")
        out = results / f"sweep_hybrid_{slug}.csv"
        all_csvs.append((combo, out))
        if not args.no_skip and out.exists():
            print(f"[{i}/{len(combos)}] {combo:<15} SKIP (exists)")
            continue
        print(f"[{i}/{len(combos)}] {combo:<15} running → {out.name}")
        cmd = [
            sys.executable, str(Path(__file__).resolve().parent / "cross_validate.py"),
            "--datasets", args.datasets,
            "--arches", "hyper-hybrid",
            "--seeds", args.seeds,
            "--adapt-epochs", str(args.adapt_epochs),
            "--residual-levels", combo,
            "--output", str(out),
        ]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  FAILED (rc={rc})")
        print()

    existing = [(c, p_) for c, p_ in all_csvs if p_.exists()]
    print("=" * 70)
    print(f"Generating comparison for {len(existing)} configs")
    subprocess.run([
        sys.executable, str(Path(__file__).resolve().parent / "analysis" / "compare_csvs.py"),
        "--inputs", ";".join(str(p_) for _, p_ in existing),
        "--labels", ";".join(c for c, _ in existing),
        "--output", "results/sweep_hybrid_summary.md",
    ])


if __name__ == "__main__":
    main()
