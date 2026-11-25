"""Datasets used for quick validation without external downloads."""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SyntheticSimilarityDataset(Dataset):
    """Synthetic dataset for quick validation.

    Each class is defined by:
        - a base RGB color
        - a random rectangular patch of a brighter shade
        - additive noise

    Images from the same class differ only in the patch position and
    noise, so a working similarity model should learn to ignore those.
    """

    def __init__(self, n_classes=50, samples_per_class=20,
                 img_size=128, seed=0):
        rng = np.random.RandomState(seed)
        self.img_size = img_size
        self.n_classes = n_classes
        self.class_colors = rng.uniform(0.2, 0.8, size=(n_classes, 3))

        self.samples = []
        self.targets = []
        for c in range(n_classes):
            for _ in range(samples_per_class):
                img = self._generate_sample(c, rng)
                self.samples.append(img)
                self.targets.append(c)

    def _generate_sample(self, class_idx, rng):
        H = W = self.img_size
        base = self.class_colors[class_idx]
        img = np.tile(base, (H, W, 1)).astype(np.float32)

        patch_w = rng.randint(12, 32)
        patch_h = rng.randint(12, 32)
        x = rng.randint(0, W - patch_w)
        y = rng.randint(0, H - patch_h)
        img[y:y + patch_h, x:x + patch_w] = np.clip(base * 1.6, 0, 1)

        img += rng.randn(H, W, 3).astype(np.float32) * 0.08
        img = np.clip(img, 0.0, 1.0)
        return torch.from_numpy(img.transpose(2, 0, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


class TransformWrapper(Dataset):
    """Wrap a dataset to apply a transform. Exposes .targets for samplers."""

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform
        self.targets = getattr(base, "targets", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label


class CUB200MetricLearning(Dataset):
    """Loads CUB_200_2011 locally, metric learning protocol.

    Classes 0..99 -> train, classes 100..199 -> test.
    Requires CUB_200_2011 archive to be extracted under `root`.
    """

    def __init__(self, root, split="train", transform=None):
        cub = self._find_root(Path(root))
        images_dir = cub / "images"

        id_to_path = {}
        with open(cub / "images.txt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                iid, rel_path = parts
                id_to_path[iid] = rel_path

        id_to_label = {}
        with open(cub / "image_class_labels.txt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                iid, label = parts
                id_to_label[iid] = int(label) - 1

        if split == "train":
            label_range = range(0, 100)
        elif split == "test":
            label_range = range(100, 200)
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.samples = []
        self.targets = []
        for iid, rel_path in id_to_path.items():
            label = id_to_label[iid]
            if label in label_range:
                full_path = images_dir / rel_path
                self.samples.append((str(full_path), label))
                self.targets.append(label)

        self.transform = transform

    @staticmethod
    def _find_root(root):
        candidates = [
            root / "CUB_200_2011",
            root / "CUB200_2011",
            root / "CUB200",
            root / "cub200",
            root,
        ]
        for c in candidates:
            if (c / "images").is_dir() and (c / "images.txt").is_file():
                return c
        raise FileNotFoundError(
            "CUB_200_2011 not found under " + str(root)
            + ". Expected a folder containing images/ and images.txt. "
            + "Checked: " + ", ".join(str(c) for c in candidates)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class MPerClassBatchSampler:
    """M-per-class batch sampler, self-contained (no pml dependency)."""

    def __init__(self, targets, m, batch_size, seed=0):
        self.targets = list(targets)
        self.m = m
        self.batch_size = batch_size
        self.class_to_idx = {}
        for i, t in enumerate(self.targets):
            self.class_to_idx.setdefault(t, []).append(i)
        self.classes = list(self.class_to_idx.keys())
        self.n_per_batch = max(1, batch_size // m)
        self.rng = __import__("torch").Generator().manual_seed(seed)

    def __iter__(self):
        import torch
        cls_order = torch.randperm(
            len(self.classes), generator=self.rng,
        ).tolist()
        for i in range(0, len(cls_order) - self.n_per_batch + 1,
                       self.n_per_batch):
            batch = []
            for ci in cls_order[i:i + self.n_per_batch]:
                cls = self.classes[ci]
                idxs = self.class_to_idx[cls]
                m_local = min(self.m, len(idxs))
                chosen = torch.randperm(
                    len(idxs), generator=self.rng,
                )[:m_local].tolist()
                batch.extend(idxs[k] for k in chosen)
            if len(batch) >= self.m:
                yield batch


class NABirdsMetricLearning(Dataset):
    """NABirds (555 classes, 48k images). Same annotation format as CUB."""

    def __init__(self, root, split="train", transform=None):
        cub = self._find_root(Path(root))
        images_dir = cub / "images"

        id_to_path = {}
        with open(cub / "images.txt", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    id_to_path[parts[0]] = parts[1]

        id_to_label = {}
        with open(cub / "image_class_labels.txt", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    id_to_label[parts[0]] = int(parts[1]) - 1

        unique = sorted(set(id_to_label.values()))
        half = len(unique) // 2
        chosen = set(unique[:half]) if split == "train" else set(unique[half:])

        self.samples, self.targets = [], []
        for iid, rel in id_to_path.items():
            lab = id_to_label.get(iid, -1)
            if lab in chosen:
                full = images_dir / rel
                if full.is_file():
                    self.samples.append((str(full), lab))
                    self.targets.append(lab)
        self.transform = transform

    @staticmethod
    def _find_root(root):
        for c in [root, root / "nabirds", root / "NABirds",
                  root / "nabirds-1.0"]:
            if (c / "images").is_dir() and (c / "images.txt").is_file():
                return c
            if (c / "images").is_dir() and (c / "image_class_labels.txt").is_file():
                return c
        raise FileNotFoundError(
            "NABirds not found under " + str(root)
            + ". Expected images/ + images.txt + image_class_labels.txt")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class AircraftMetricLearning(Dataset):
    """FGVC Aircraft (100 variants, 10200 images). Split by class 50/50."""

    def __init__(self, root, split="train", transform=None):
        data_dir = self._find_root(Path(root))
        images_dir = data_dir / "images"

        id_to_variant = {}
        for fname in ["images_variant_trainval.txt",
                      "images_variant_train.txt",
                      "images_variant_val.txt",
                      "images_variant_test.txt"]:
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

        variants = sorted(set(id_to_variant.values()))
        var_to_id = {v: i for i, v in enumerate(variants)}
        half = len(variants) // 2
        chosen = set(variants[:half]) if split == "train" \
            else set(variants[half:])

        self.samples, self.targets = [], []
        for iid, variant in id_to_variant.items():
            if variant not in chosen:
                continue
            full = images_dir / (iid + ".jpg")
            if full.is_file():
                self.samples.append((str(full), var_to_id[variant]))
                self.targets.append(var_to_id[variant])
        self.transform = transform

    @staticmethod
    def _find_root(root):
        candidates = [
            root / "fgvc-aircraft-2013b" / "data",
            root / "data",
            root,
        ]
        for c in candidates:
            if (c / "images").is_dir():
                return c
        raise FileNotFoundError(
            "FGVC Aircraft not found under " + str(root)
            + ". Expected fgvc-aircraft-2013b/data/images/")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class Cars196MetricLearning(Dataset):
    """Cars196 metric learning protocol (split by class, not by folder).

    Auto-detects the folder structure. Works with:
      - Original Stanford Cars:  root/cars_train/, root/cars_test/,
                                 root/devkit/cars_train_annos.mat,
                                 root/devkit/cars_test_annos_withlabels.mat
      - Kaggle variants with the same layout
      - CUB-style:               root/images/, root/images.txt,
                                 root/image_class_labels.txt

    Splits by class: first half (98 of 196) -> train, second half -> test.
    Images from both cars_train/ and cars_test/ folders are pooled by
    class first, then split by class. The original image-level split of
    Stanford Cars is NOT used (classes overlap between train and test
    in the original split).
    """

    def __init__(self, root, split="train", transform=None):
        cub_root = self._find_root(Path(root))

        samples = self._try_load_mat_annotations(cub_root)
        if samples is None:
            samples = self._try_load_txt_annotations(cub_root)
        if samples is None:
            raise FileNotFoundError(
                "No annotations found under " + str(cub_root) + ". "
                "Expected devkit/cars_train_annos.mat + "
                "cars_test_annos_withlabels.mat, OR "
                "images.txt + image_class_labels.txt."
            )

        unique_classes = sorted(set(c for _, c in samples))
        n = len(unique_classes)
        if n < 4:
            raise RuntimeError(
                "Too few classes found (" + str(n) + "). "
                "Check annotations and image paths."
            )
        half = n // 2
        chosen = unique_classes[:half] if split == "train" \
            else unique_classes[half:]
        chosen_set = set(chosen)

        self.samples = [(p, c) for p, c in samples if c in chosen_set]
        self.targets = [c for _, c in self.samples]
        self.transform = transform

        print("  Cars196: split=" + split
              + " classes=" + str(len(chosen))
              + " images=" + str(len(self.samples))
              + " (from " + str(n) + " total classes)")

    @staticmethod
    def _find_root(root):
        candidates = [root, root / "stanford_cars",
                      root / "cars196", root / "Cars196"]
        for c in candidates:
            if not c.is_dir():
                continue
            if (c / "devkit").is_dir():
                return c
            if (c / "images.txt").is_file():
                return c
            if (c / "cars_train").is_dir() or (c / "images").is_dir():
                return c
        raise FileNotFoundError(
            "Cars196 not found under " + str(root) + ". "
            "Expected devkit/, or cars_train/+cars_test/, or images/."
        )

    @staticmethod
    def _find_images_dir(root, candidates):
        for cand in candidates:
            p = root / cand
            if not p.is_dir():
                continue
            nested = p / p.name
            if nested.is_dir() and any(nested.glob("*.jpg")):
                return nested
            if any(p.glob("*.jpg")):
                return p
        return None

    @staticmethod
    def _try_load_mat_annotations(root):
        try:
            import scipy.io as sio
        except ImportError:
            print("  scipy not installed, cannot read .mat annotations.")
            print("  Install with: pip install scipy")
            return None

        devkit = root / "devkit"
        train_annos = devkit / "cars_train_annos.mat"
        if not train_annos.is_file():
            return None

        # Prefer "withlabels", fall back to plain "test_annos"
        test_annos = None
        for name in ["cars_test_annos_withlabels.mat",
                     "cars_test_annos.mat"]:
            p = devkit / name
            if p.is_file():
                test_annos = p
                break

        train_imgs = Cars196MetricLearning._find_images_dir(
            root, ["cars_train", "images/train"])
        test_imgs = Cars196MetricLearning._find_images_dir(
            root, ["cars_test", "images/test"])

        if train_imgs is None:
            print("  WARN: cars_train/ folder not found, cannot load.")
            return None

        samples = []

        # --- train annotations ---
        mat = sio.loadmat(str(train_annos))
        for ann in mat["annotations"][0]:
            try:
                fname = str(ann["fname"][0])
                cls = int(ann["class"][0][0]) - 1
            except (KeyError, ValueError, IndexError):
                continue
            full = train_imgs / fname
            if full.is_file():
                samples.append((str(full), cls))

        # --- test annotations (if labels available) ---
        skipped_test = 0
        loaded_test = 0
        if test_annos is not None and test_imgs is not None:
            mat = sio.loadmat(str(test_annos))
            annos = mat["annotations"][0]
            has_class = False
            if len(annos) > 0:
                names = annos[0].dtype.names or ()
                has_class = "class" in names

            if has_class:
                for ann in annos:
                    try:
                        fname = str(ann["fname"][0])
                        cls = int(ann["class"][0][0]) - 1
                    except (KeyError, ValueError, IndexError):
                        skipped_test += 1
                        continue
                    full = test_imgs / fname
                    if full.is_file():
                        samples.append((str(full), cls))
                        loaded_test += 1
            else:
                print("  NOTE: " + test_annos.name
                      + " has no 'class' field (public variant).")
                print("        Using cars_train/ only ("
                      + str(len(samples)) + " images across 196 classes).")
                print("        For full 16185 images, download "
                      "cars_test_annos_withlabels.mat separately.")

        if not samples:
            return None

        # --- statistics ---
        from collections import Counter
        counts = Counter(c for _, c in samples)
        n_classes = len(counts)
        min_per_class = min(counts.values())
        max_per_class = max(counts.values())
        msg = ("  Loaded " + str(len(samples)) + " images from "
               + str(n_classes) + " classes "
               + "(min/avg/max per class: "
               + str(min_per_class) + "/"
               + str(sum(counts.values()) // n_classes) + "/"
               + str(max_per_class) + ")")
        if loaded_test > 0:
            msg += "; " + str(loaded_test) + " from test"
        print(msg)

        if min_per_class < 25:
            print("  WARN: some classes have <25 images. "
                  "Set --M-per-class 20 or lower for Cars.")

        return samples

    @staticmethod
    def _try_load_txt_annotations(root):
        images_txt = None
        labels_txt = None
        for search_root in [root, root / "devkit"]:
            if (search_root / "images.txt").is_file():
                images_txt = search_root / "images.txt"
            if (search_root / "image_class_labels.txt").is_file():
                labels_txt = search_root / "image_class_labels.txt"

        if not (images_txt and labels_txt):
            return None

        images_dir = Cars196MetricLearning._find_images_dir(
            root, ["images", "cars_train"])
        if images_dir is None:
            return None

        id_to_path = {}
        with open(images_txt, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    id_to_path[parts[0]] = parts[1]

        id_to_label = {}
        with open(labels_txt, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    id_to_label[parts[0]] = int(parts[1]) - 1

        samples = []
        for iid, rel_path in id_to_path.items():
            if iid in id_to_label:
                full = images_dir / rel_path
                if full.is_file():
                    samples.append((str(full), id_to_label[iid]))

        return samples if samples else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
