# download_datasets.py
# Downloads publicly available datasets into ./data/ and checks all loaders.
# Idempotent + integrity-checked + retried.
#
# Usage:
#   python scripts/download_datasets.py
#   python scripts/download_datasets.py --only dogs,dtd
#   python scripts/download_datasets.py --check-only

import argparse
import shutil
import ssl
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import os
os.chdir(PROJECT_ROOT)

from dynamic_siamese.datasets_multi import (  # noqa: E402
    DATASET_SUBDIRS, LOADERS,
)

MANUAL_DATASETS = {
    "cub": "Download CUB_200_2011 and unpack it under data/CUB_200_2011. "
           "Source: https://data.caltech.edu/records/65de6-vp158",
    "cars": "Obtain Stanford Cars from its official distribution (or your "
            "licensed copy) and unpack the cars_train/devkit structure under "
            "data/stanford_cars. Source: https://www.kaggle.com/datasets/" 
            "rickyyyyyyy/torchvision-stanford-cars",
    "nabirds": "Download NABirds after accepting its terms at "
               "https://dl.allaboutbirds.org/nabirds and unpack the archive "
               "under data/nabirds. This dataset is large (~30 GB) and gated.",
}


DATASETS = {
    "dogs": {
        "files": [(
            "images.tar.gz",
            [
                "https://huggingface.co/datasets/Alanox/stanford-dogs/resolve/main/images.tar.gz",
                "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
            ],
            None,
        )],
        "target": "StanfordDogs",
        "check": lambda p: ((p / "Images").is_dir()
                            or (p / "images").is_dir())
                           and len(list((p / "Images").iterdir()
                                        if (p / "Images").is_dir()
                                        else (p / "images").iterdir())) >= 4,
    },
    "flowers": {
        "files": [
            ("102flowers.tgz",
             ["https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"],
             "tgz"),
            ("imagelabels.mat",
             ["https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"],
             "file"),
        ],
        "target": "flowers102",
        "check": lambda p: (p / "jpg").is_dir()
                           and (p / "imagelabels.mat").is_file(),
    },
    "dtd": {
        "files": [(
            "dtd-r1.0.1.tar.gz",
            [
                "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
                "https://web.archive.org/web/20240101000000id_/https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
            ],
            "tgz",
        )],
        "target": "dtd",
        "check": lambda p: (p / "dtd" / "images").is_dir()
                           or (p / "images").is_dir(),
    },
    "omniglot": {
        "files": [(
            "images_background.zip",
            ["https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip"],
            "zip_omniglot",
        )],
        "target": "omniglot",
        "check": lambda p: any(
            (p / d).is_dir() for d in p.glob("*")
            if d.is_dir() and any(c.is_dir() for c in d.iterdir())
        ),
    },
}


def _open_url(url, insecure=False):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0 Safari/537.36",
    })
    if insecure:
        return urllib.request.urlopen(
            req, context=ssl._create_unverified_context(), timeout=120)
    try:
        return urllib.request.urlopen(req, timeout=120)
    except urllib.error.URLError as e:
        msg = str(e).lower()
        if "certificate" in msg or "ssl" in msg:
            print("    SSL error:", e)
            print("    retrying with insecure SSL...")
            return urllib.request.urlopen(
                req, context=ssl._create_unverified_context(), timeout=120)
        raise


def _verify_archive(path):
    """True if archive can be opened fully (detects truncation)."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        if path.name.endswith((".tar", ".tar.gz", ".tgz")):
            with tarfile.open(str(path), "r:*") as tf:
                tf.getmembers()
            return True
        if path.name.endswith(".zip"):
            with zipfile.ZipFile(str(path), "r") as zf:
                return zf.testzip() is None
        return True
    except Exception:
        return False


def _stream_to_file(resp, dest):
    total = int(resp.headers.get("Content-Length", 0)) or None
    downloaded = 0
    start = time.time()
    last = 0.0
    with open(dest, "wb") as f:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last > 2.0:
                last = now
                speed = downloaded / max(now - start, 0.001) / 1e6
                if total:
                    print("      {:>7.1f} / {:.1f} MB ({:5.1f}%) "
                          "{:.1f} MB/s".format(
                              downloaded / 1e6, total / 1e6,
                              100.0 * downloaded / total, speed))
                else:
                    print("      {:>7.1f} MB  {:.1f} MB/s".format(
                        downloaded / 1e6, speed))
    return total, downloaded


def _download(url_or_list, dest, insecure=False, force=False,
              max_retries=3):
    # existing cache
    if not force and dest.exists() and dest.stat().st_size > 0:
        if _verify_archive(dest):
            print("    already exists (verified):", dest.name,
                  "({:.1f} MB)".format(dest.stat().st_size / 1e6))
            return
        print("    existing", dest.name, "is corrupted, deleting")
        dest.unlink()

    urls = url_or_list if isinstance(url_or_list, list) else [url_or_list]
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None

    for attempt in range(max_retries):
        if attempt > 0:
            wait = 2 ** attempt
            print("    retry {}/{} after {}s...".format(
                attempt + 1, max_retries, wait))
            time.sleep(wait)

        resp = None
        for i, url in enumerate(urls):
            print("    downloading:", dest.name)
            print("    from:", url)
            try:
                resp = _open_url(url, insecure=insecure)
                break
            except Exception as e:
                last_err = e
                print("    FAILED:", e)
                if i < len(urls) - 1:
                    print("    trying fallback URL...")
                else:
                    print("    no more fallbacks for this attempt")
        if resp is None:
            continue

        try:
            total, downloaded = _stream_to_file(resp, dest)
        except Exception as e:
            last_err = e
            print("    stream error:", e)
            if dest.exists():
                dest.unlink()
            continue

        if total and downloaded != total:
            last_err = IOError(
                "truncated: {}/{} bytes".format(downloaded, total))
            print("    TRUNCATED:", last_err)
            if dest.exists():
                dest.unlink()
            continue

        if not _verify_archive(dest):
            last_err = IOError("archive failed integrity check")
            print("    INTEGRITY failed")
            if dest.exists():
                dest.unlink()
            continue

        print("    done: {:.1f} MB (verified)".format(downloaded / 1e6))
        return

    raise last_err if last_err else RuntimeError("download failed")


def _extract_tar(archive_path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    print("    extracting:", archive_path.name, "->", dest)
    with tarfile.open(str(archive_path), "r:*") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(str(dest), filter="data")
        else:
            tf.extractall(str(dest))
    # normalize images/ -> Images/ (Stanford Dogs mirror sometimes uses lowercase)
    lower = dest / "images"
    upper = dest / "Images"
    if lower.is_dir() and not upper.is_dir():
        print("    normalizing: images/ -> Images/")
        lower.rename(upper)
    # also handle: archive root is nested one level deeper
    if not upper.is_dir() and not lower.is_dir():
        for cand in dest.glob("*/images"):
            if cand.is_dir():
                print("    normalizing nested:", cand, "->", upper)
                cand.rename(upper)
                break
        if not upper.is_dir():
            for cand in dest.glob("*/Images"):
                if cand.is_dir():
                    print("    normalizing nested:", cand, "->", upper)
                    cand.rename(upper)
                    break


def _extract_zip(archive_path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    print("    extracting:", archive_path.name, "->", dest)
    with zipfile.ZipFile(str(archive_path), "r") as zf:
        zf.extractall(str(dest))


def _flatten_omniglot(target_dir, tmp):
    inner = tmp / "images_background"
    if not inner.is_dir():
        for d in tmp.rglob("images_background"):
            if d.is_dir():
                inner = d
                break
    if not inner.is_dir():
        print("    WARN: images_background not found in zip")
        return
    moved = 0
    for alpha in inner.iterdir():
        if not alpha.is_dir():
            continue
        d = target_dir / alpha.name
        if d.exists():
            shutil.rmtree(d)
        shutil.move(str(alpha), str(d))
        moved += 1
    print("    flattened {} alphabets into {}".format(moved, target_dir))


def install_one(name, root, insecure, force=False):
    if name == "aircraft":
        return install_aircraft(root, insecure, force)
    cfg = DATASETS[name]
    target = root / cfg["target"]
    if cfg["check"](target):
        print("  [SKIP] {}: already installed".format(name))
        return True

    print("  [INSTALL] {} -> {}".format(name, target))
    target.mkdir(parents=True, exist_ok=True)
    cache = root / "_cache" / name
    cache.mkdir(parents=True, exist_ok=True)

    for fname, url_or_list, mode in cfg["files"]:
        local = cache / fname
        try:
            _download(url_or_list, local, insecure=insecure, force=force)
        except Exception as e:
            print("  FAILED to download {}: {}".format(name, e))
            return False
        if mode in (None, "tgz"):
            _extract_tar(local, target)
        elif mode == "file":
            shutil.copy2(str(local), str(target / fname))
        elif mode == "zip_omniglot":
            tmp = cache / "tmp_omni"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            _extract_zip(local, tmp)
            _flatten_omniglot(target, tmp)
            shutil.rmtree(tmp, ignore_errors=True)

    if cfg["check"](target):
        print("  [OK] {} installed".format(name))
        return True
    print("  [FAIL] {}: check failed".format(name))
    return False



def install_aircraft(root, insecure, force=False):
    """Download FGVC Aircraft via torchvision or direct tar URL."""
    data_root = Path(root)
    target = data_root / "fgvc-aircraft-2013b" / "data" / "images"
    if target.is_dir() and any(target.glob("*.jpg")):
        print("  [SKIP] aircraft: already installed")
        return True
    try:
        from torchvision.datasets import FGVCAircraft
        print("  [DOWNLOAD] aircraft via torchvision...")
        FGVCAircraft(root=str(data_root), split="train", download=True)
        if target.is_dir():
            n = len(list(target.glob("*.jpg")))
            print(f"  [OK] aircraft installed: {n} images")
            return True
    except Exception as e:
        print(f"  torchvision failed: {e}")
    url = ("https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"
           "archives/fgvc-aircraft-2013b.tar.gz")
    cache = data_root / "_cache" / "aircraft"
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / "fgvc-aircraft-2013b.tar.gz"
    try:
        _download([url], local, insecure=insecure, force=force)
        _extract_tar(local, data_root)
        if target.is_dir():
            n = len(list(target.glob("*.jpg")))
            print(f"  [OK] aircraft installed: {n} images")
            return True
    except Exception as e:
        print(f"  FAILED: {e}")
    return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data")
    p.add_argument("--only", default=None)
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--force-redownload", action="store_true")
    p.add_argument("--check-only", action="store_true",
                   help="only check whether datasets are loadable")
    args = p.parse_args()

    root = Path(args.data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    print("Data root:", root)

    names = [n.strip() for n in (args.only.split(",") if args.only
                                  else list(LOADERS)) if n.strip()]
    unknown = sorted(set(names) - set(LOADERS))
    if unknown:
        p.error("unknown dataset(s): " + ", ".join(unknown))

    if not args.check_only:
        for name in names:
            if name in MANUAL_DATASETS:
                print("\n[MANUAL] {}: {}".format(name, MANUAL_DATASETS[name]))
                continue
            if name not in DATASETS and name != "aircraft":
                continue
            print()
            try:
                install_one(name, root, args.insecure,
                            force=args.force_redownload)
            except Exception as e:
                print("  [FAILED] {}: {}".format(name, e))

    print("\nDataset loader checks:")
    results = {}
    for name in names:
        try:
            path = root / DATASET_SUBDIRS[name]
            samples = LOADERS[name](path)
            ok = bool(samples)
            details = "{} images, {} classes".format(
                len(samples), len(set(label for _, label in samples))) if ok \
                else "not found or loader returned no samples"
            results[name] = ok
            print("  [{}] {}: {}".format("OK" if ok else "MISSING", name,
                                        details))
        except Exception as e:
            results[name] = False
            print("  [ERROR] {}: {}".format(name, e))

    missing_manual = [name for name in names
                     if not results.get(name) and name in MANUAL_DATASETS]
    if missing_manual:
        print("\nManual setup needed:")
        for name in missing_manual:
            print("  - {}: {}".format(name, MANUAL_DATASETS[name]))
    missing_auto = [name for name in names if not results.get(name)
                    and name not in MANUAL_DATASETS]
    if missing_auto:
        print("\nSome automatic downloads failed or did not match the expected "
              "layout. Check the messages above and rerun the command.")
    if all(results.values()):
        print("\nAll requested datasets are loadable.")

    print()
    print("=" * 60)
    print("Summary:")
    for name, ok in results.items():
        print("  {:<10} {}".format(name, "OK" if ok else "FAILED"))
    print("=" * 60)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
