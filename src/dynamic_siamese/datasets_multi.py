
"""Multi-dataset loaders for cross-validation.

Provides GenericMLDataset + per-dataset sample loaders for:
    cub, cars, dogs, flowers, dtd, omniglot

Each loader returns list[(image_path: str, class_id: int)] in canonical
order (classes 0..N-1 sorted). GenericMLDataset then splits by class.
"""

import random
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ---------- Per-dataset sample loaders ----------
# Each returns list[(str_path, int_class_id)] or None if dataset missing.

def _load_cub(root):
    """CUB200-2011. Expects images/ + images.txt + image_class_labels.txt."""
    cub = Path(root)
    if not (cub / "images").is_dir():
        return None
    if not (cub / "images.txt").is_file():
        return None

    id_to_path, id_to_label = {}, {}
    with open(cub / "images.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                id_to_path[parts[0]] = parts[1]
    with open(cub / "image_class_labels.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                id_to_label[parts[0]] = int(parts[1]) - 1

    samples = []
    for iid, rel in id_to_path.items():
        if iid not in id_to_label:
            continue
        full = cub / "images" / rel
        if full.is_file():
            samples.append((str(full), id_to_label[iid]))
    return samples or None


def _load_cars(root):
    """Cars196 with devkit mat annotations (works with public variant too)."""
    try:
        import scipy.io as sio
    except ImportError:
        return None

    cars = Path(root)
    devkit = cars / "devkit"
    train_annos = devkit / "cars_train_annos.mat"
    if not train_annos.is_file():
        return None

    def _find_img_dir(candidates):
        for cand in candidates:
            p = cars / cand
            if not p.is_dir():
                continue
            nested = p / p.name
            if nested.is_dir() and any(nested.glob("*.jpg")):
                return nested
            if any(p.glob("*.jpg")):
                return p
        return None

    train_dir = _find_img_dir(["cars_train", "images/train"])
    test_dir = _find_img_dir(["cars_test", "images/test"])
    if train_dir is None:
        return None

    samples = []
    for mat_path, img_dir in [
        (train_annos, train_dir),
        (devkit / "cars_test_annos_withlabels.mat", test_dir),
    ]:
        if img_dir is None or not mat_path.is_file():
            continue
        mat = sio.loadmat(str(mat_path))
        if "annotations" not in mat:
            continue
        for ann in mat["annotations"][0]:
            try:
                fname = str(ann["fname"][0])
                cls = int(ann["class"][0][0]) - 1
            except (KeyError, ValueError, IndexError):
                continue
            full = img_dir / fname
            if full.is_file():
                samples.append((str(full), cls))
    return samples or None


def _load_dogs(root):
    """Support both layouts:
       A) images/<n02085620-Chihuahua>/*.jpg
       B) images/*.jpg (flat, infer breed from n########_ prefix)
    """
    root = Path(root)
    if not root.is_dir():
        return None

    def _find_jpg_dir(base, depth=0):
        if depth > 4:
            return None
        js = list(base.glob("*.jpg")) + list(base.glob("*.JPG"))
        if js:
            return base, js
        for d in base.iterdir():
            if d.is_dir():
                r = _find_jpg_dir(d, depth + 1)
                if r is not None:
                    return r
        return None

    found = _find_jpg_dir(root)
    if found is None:
        return None
    jpg_dir, js = found

    subdirs = [d for d in jpg_dir.iterdir() if d.is_dir()]
    sub_jpg_total = sum(len(list(d.glob("*.jpg"))) for d in subdirs)
    if len(subdirs) >= 4 and sub_jpg_total >= 10:
        samples = []
        for cls, d in enumerate(sorted(subdirs)):
            for img in list(d.glob("*.jpg")) + list(d.glob("*.JPG")):
                samples.append((str(img), cls))
        return samples or None

    prefix_pat = re.compile(r"^(n\d{8})_")
    groups = {}
    for p in js:
        m = prefix_pat.match(p.name)
        key = m.group(1) if m else p.stem.split("_")[0]
        groups.setdefault(key, []).append(p)
    if len(groups) < 4:
        return None

    samples = []
    for cls, key in enumerate(sorted(groups.keys())):
        for p in groups[key]:
            samples.append((str(p), cls))
    return samples or None


def _load_flowers(root):
    """Oxford Flowers 102: jpg/*.jpg + imagelabels.mat."""
    try:
        import scipy.io as sio
    except ImportError:
        return None

    flowers = Path(root)
    jpg = flowers / "jpg"
    labels_mat = flowers / "imagelabels.mat"
    if not (jpg.is_dir() and labels_mat.is_file()):
        # try torchvision-style layout
        jpg = flowers / "102flowers" / "jpg"
        labels_mat = flowers / "102flowers" / "imagelabels.mat"
        if not (jpg.is_dir() and labels_mat.is_file()):
            return None

    mat = sio.loadmat(str(labels_mat))
    labels = mat["labels"][0]
    samples = []
    for idx, cls in enumerate(labels, start=1):
        fname = "image_{:05d}.jpg".format(idx)
        full = jpg / fname
        if full.is_file():
            samples.append((str(full), int(cls) - 1))
    return samples or None


def _load_dtd(root):
    """Describable Textures Dataset: images/<class>/<file>.jpg"""
    dtd = Path(root)
    images = dtd / "images"
    if not images.is_dir():
        images = dtd / "dtd" / "images"
        if not images.is_dir():
            return None

    class_dirs = sorted([d for d in images.iterdir() if d.is_dir()])
    if len(class_dirs) < 4:
        return None

    samples = []
    for cls, d in enumerate(class_dirs):
        for img in d.glob("*.jpg"):
            samples.append((str(img), cls))
    return samples or None


def _load_omniglot(root):
    """Omniglot: <alphabet>/<character>/*.png (two-level).

    Characters across all alphabets are treated as distinct classes.
    """
    omni = Path(root)
    candidates = [omni, omni / "omniglot", omni / "images_background"]
    base = None
    for c in candidates:
        if c.is_dir():
            subdirs = [d for d in c.iterdir() if d.is_dir()]
            if any(any(ch.iterdir()) for ch in subdirs
                   if any(ch.is_dir() for ch in ch.iterdir())):
                base = c
                break
    if base is None:
        return None

    samples = []
    cls_id = 0
    for alphabet in sorted([d for d in base.iterdir() if d.is_dir()]):
        for char in sorted([d for d in alphabet.iterdir() if d.is_dir()]):
            imgs = list(char.glob("*.png"))
            if not imgs:
                continue
            for img in imgs:
                samples.append((str(img), cls_id))
            cls_id += 1
    return samples or None


def _load_nabirds(root):
    """NABirds: same annotation format as CUB."""
    nab = Path(root)
    if not (nab / "images").is_dir():
        return None
    if not (nab / "images.txt").is_file():
        return None
    id_to_path, id_to_label = {}, {}
    with open(nab / "images.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                id_to_path[parts[0]] = parts[1]
    with open(nab / "image_class_labels.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                id_to_label[parts[0]] = int(parts[1]) - 1
    samples = []
    for iid, rel in id_to_path.items():
        if iid not in id_to_label:
            continue
        full = nab / "images" / rel
        if full.is_file():
            samples.append((str(full), id_to_label[iid]))
    # filter classes with < 26 images
    from collections import Counter as _C
    if samples:
        _cnt = _C(c for _, c in samples)
        _keep = {c for c, n in _cnt.items() if n >= 26}
        if _keep and len(_keep) < len(_cnt):
            samples = [(p, c) for p, c in samples if c in _keep]
            _rem = sorted(set(c for _, c in samples))
            _remap = {c: i for i, c in enumerate(_rem)}
            samples = [(p, _remap[c]) for p, c in samples]
            print(f'  NABirds: filtered to {len(samples)} images, {len(_rem)} classes (>=26)')
    return samples or None


def _load_aircraft(root):
    """FGVC Aircraft. Structure: <root>/fgvc-aircraft-2013b/data/images/."""
    air = Path(root)
    data_dir = None
    for c in [air / "fgvc-aircraft-2013b" / "data", air / "data", air]:
        if (c / "images").is_dir():
            data_dir = c; break
    if data_dir is None:
        return None
    img_dir = data_dir / "images"
    id_to_variant = {}
    for fname in ["images_variant_trainval.txt", "images_variant_train.txt",
                  "images_variant_val.txt", "images_variant_test.txt"]:
        f = data_dir / fname
        if not f.is_file():
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    id_to_variant[parts[0]] = parts[1]
    if not id_to_variant:
        return None
    variants = sorted(set(id_to_variant.values()))
    var_to_id = {v: i for i, v in enumerate(variants)}
    samples = []
    for iid, variant in id_to_variant.items():
        full = img_dir / (iid + ".jpg")
        if full.is_file():
            samples.append((str(full), var_to_id[variant]))
    # filter classes with < 26 images
    from collections import Counter as _C
    if samples:
        _cnt = _C(c for _, c in samples)
        _keep = {c for c, n in _cnt.items() if n >= 26}
        if _keep and len(_keep) < len(_cnt):
            samples = [(p, c) for p, c in samples if c in _keep]
            _rem = sorted(set(c for _, c in samples))
            _remap = {c: i for i, c in enumerate(_rem)}
            samples = [(p, _remap[c]) for p, c in samples]
            print(f"  Aircraft: filtered to {len(samples)} images, "
                  f"{len(_rem)} classes (>=26)")
    return samples or None


LOADERS = {
    "cub": _load_cub,
    "cars": _load_cars,
    "dogs": _load_dogs,
    "flowers": _load_flowers,
    "dtd": _load_dtd,
    "omniglot": _load_omniglot,
    "nabirds": _load_nabirds,
    "aircraft": _load_aircraft,
}

# Default dataset subdirectory names under --data-root
DATASET_SUBDIRS = {
    "cub": "CUB_200_2011",
    "cars": "stanford_cars",
    "dogs": "StanfordDogs",
    "flowers": "flowers102",
    "dtd": "dtd",
    "omniglot": "omniglot",
    "nabirds": "nabirds",
    "aircraft": "fgvc-aircraft-2013b/data",
}

DATASET_URLS = {
    "cub": "https://data.caltech.edu/records/65de6-vp158",
    "cars": "https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars",
    "dogs": "http://vision.stanford.edu/aditya86/ImageNetDogs/",
    "flowers": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
    "dtd": "https://www.robots.ox.ac.uk/~vgg/data/dtd/",
    "omniglot": "https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip",
    "nabirds": "https://dl.allaboutbirds.org/nabirds (~30 GB)",
    "aircraft": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
}


# ---------- Generic metric learning dataset ----------

class GenericMLDataset(Dataset):
    """Wrapper over any registered dataset with class split + balancing.

    Args:
        root:            data root (parent of dataset subdirectory)
        dataset_name:    one of LOADERS keys
        split:           'train' (base classes) or 'test' (novel classes)
        transform:       torchvision transform
        max_classes:     cap number of classes (None = no cap)
        max_per_class:   cap samples per class (None = no cap)
        dataset_dir:     override subdirectory (else uses DATASET_SUBDIRS)
    """

    def __init__(self, root, dataset_name, split="train", transform=None,
                 max_classes=None, max_per_class=None, dataset_dir=None):
        if dataset_name not in LOADERS:
            raise ValueError("Unknown dataset: " + dataset_name
                             + ". Available: " + str(list(LOADERS.keys())))

        subdir = dataset_dir or DATASET_SUBDIRS.get(dataset_name)
        ds_path = Path(root) / subdir if subdir else Path(root)

        raw = LOADERS[dataset_name](ds_path)
        if not raw:
            raise FileNotFoundError(
                "Dataset '" + dataset_name + "' not found at "
                + str(ds_path) + "\nDownload: " + DATASET_URLS[dataset_name]
            )

        # split by class: first half -> train, second half -> test
        unique = sorted(set(c for _, c in raw))
        half = len(unique) // 2
        chosen = set(unique[:half]) if split == "train" \
            else set(unique[half:])
        filtered = [(p, c) for p, c in raw if c in chosen]

        # balance
        if max_classes is not None:
            kept = sorted(set(c for _, c in filtered))[:max_classes]
            keep_set = set(kept)
            filtered = [(p, c) for p, c in filtered if c in keep_set]

        if max_per_class is not None:
            rng = random.Random(42)
            rng.shuffle(filtered)
            cnt = Counter()
            balanced = []
            for p, c in filtered:
                if cnt[c] < max_per_class:
                    balanced.append((p, c))
                    cnt[c] += 1
            filtered = balanced

        # reindex classes to 0..K-1
        remaining = sorted(set(c for _, c in filtered))
        remap = {c: i for i, c in enumerate(remaining)}
        self.samples = [(p, remap[c]) for p, c in filtered]
        self.targets = [c for _, c in self.samples]
        self.transform = transform
        self.dataset_name = dataset_name
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def stats(self):
        if not self.targets:
            return "0 samples"
        cnt = Counter(self.targets)
        return ("{} samples, {} classes "
                "(min/avg/max per class: {}/{}/{})").format(
            len(self.samples), len(cnt),
            min(cnt.values()), sum(cnt.values()) // len(cnt),
            max(cnt.values()),
        )


class CombinedBalancedDataset(Dataset):
    """Combine N datasets into one with globally rekeyed classes."""

    def __init__(self, datasets):
        self.transform = None  # datasets carry their own
        all_samples = []
        self.domains = []
        offset = 0
        for di, ds in enumerate(datasets):
            for path, label in ds.samples:
                all_samples.append((path, label + offset))
                self.domains.append(di)
            if ds.targets:
                offset = max(c for _, c in all_samples) + 1 \
                    if all_samples else 0
        self.samples = all_samples
        self.targets = [c for _, c in self.samples]
        self.n_domains = len(datasets)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        import torchvision.transforms.functional as TF
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        # use train-mode transform to keep things simple
        from torchvision import transforms
        if not hasattr(self, "_tf"):
            self._tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        return self._tf(img), label
