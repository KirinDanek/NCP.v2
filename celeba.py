"""
celeba.py

Purpose
-------
Minimal CelebA indexing + dataset wrapper to support pruning experiments without relying on
torchvision’s built-in CelebA dataset.

Provides
--------
CelebAIndex:
- Reads official CelebA metadata files:
    * list_eval_partition.txt  (split: train/val/test as 0/1/2)
    * list_attr_celeba.txt      (attributes in {-1, +1})
- Returns filenames for a requested official split and attribute dictionaries per filename.

CelebAAttributeDataset:
- Dataset yielding tuples:
    (x, y, g, fname)
  where:
    x: transformed RGB image tensor
    y: {0,1} for target_attr
    g: sensitive attribute from "Male" by default, either {-1,+1} or {0,1}
    fname: filename string (useful for deterministic hashing-based splits)

Important details
-----------------
- Images are expected under: <celeba_root>/img_align_celeba/*.jpg
- Attributes in list_attr_celeba.txt are parsed into a dict per image.
- y and g are mapped from {-1,+1} to either {0,1} or {-1,+1}.
- If you pass an explicit filenames list, you can restrict the dataset to any subset
  (e.g., only official TRAIN, or a deterministic hashed split).

Used by
-------
- data.py CelebA split constructors, which wrap this dataset and then apply XYOnly for pruning.
"""


from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


_SPLIT_MAP = {
    "train": 0,
    "val": 1,
    "valid": 1,
    "validation": 1,
    "test": 2,
}


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f.readlines()]


def _ensure_exists(path: str, kind: str = "file") -> None:
    if kind == "file" and not os.path.isfile(path):
        raise FileNotFoundError(f"Expected file not found: {path}")
    if kind == "dir" and not os.path.isdir(path):
        raise FileNotFoundError(f"Expected directory not found: {path}")


@dataclass
class CelebAIndex:
    """
    Lightweight index for CelebA official split + attributes.
    """
    root: str

    def __post_init__(self):
        self.root = os.path.abspath(self.root)

        self.attr_path = os.path.join(self.root, "list_attr_celeba.txt")
        self.split_path = os.path.join(self.root, "list_eval_partition.txt")

        _ensure_exists(self.attr_path, "file")
        _ensure_exists(self.split_path, "file")

        # Images are usually in root/img_align_celeba/*.jpg
        self.img_dir = os.path.join(self.root, "img_align_celeba")
        _ensure_exists(self.img_dir, "dir")

        # Lazily loaded caches
        self._attrs_header: Optional[List[str]] = None
        self._attr_table: Optional[Dict[str, Dict[str, int]]] = None
        self._split_table: Optional[Dict[str, int]] = None

    def _load_splits(self) -> None:
        if self._split_table is not None:
            return
        lines = _read_lines(self.split_path)
        # Each line: "000001.jpg 0"
        table: Dict[str, int] = {}
        for ln in lines:
            if not ln.strip():
                continue
            parts = ln.split()
            if len(parts) != 2:
                continue
            fname, split_id_str = parts
            table[fname] = int(split_id_str)
        self._split_table = table

    def _load_attrs(self) -> None:
        if self._attr_table is not None:
            return
        lines = _read_lines(self.attr_path)
        if len(lines) < 3:
            raise ValueError(f"Unexpected format in {self.attr_path}")

        # line0: number of images
        # line1: attribute names
        header = lines[1].split()
        self._attrs_header = header

        table: Dict[str, Dict[str, int]] = {}
        for ln in lines[2:]:
            if not ln.strip():
                continue
            parts = ln.split()
            # parts[0] is filename, then len(header) values in {-1,+1}
            fname = parts[0]
            vals = parts[1:]
            if len(vals) != len(header):
                # skip malformed line
                continue
            table[fname] = {attr: int(v) for attr, v in zip(header, vals)}
        self._attr_table = table

    @property
    def attr_names(self) -> List[str]:
        self._load_attrs()
        assert self._attrs_header is not None
        return self._attrs_header

    def split_id(self, filename: str) -> int:
        self._load_splits()
        assert self._split_table is not None
        return self._split_table[filename]

    def attrs_for(self, filename: str) -> Dict[str, int]:
        self._load_attrs()
        assert self._attr_table is not None
        return self._attr_table[filename]

    def filenames(
        self,
        split: str = "train",
        require_exists: bool = True,
    ) -> List[str]:
        """
        Returns filenames for the requested official split.
        split in {"train","val","test"} (val can be "valid"/"validation")
        """
        self._load_splits()
        assert self._split_table is not None

        split = split.lower()
        if split not in _SPLIT_MAP:
            raise ValueError(f"Unknown split '{split}'. Use one of {list(_SPLIT_MAP.keys())}")

        sid = _SPLIT_MAP[split]
        out: List[str] = []
        for fname, f_sid in self._split_table.items():
            if f_sid != sid:
                continue
            if require_exists:
                img_path = os.path.join(self.img_dir, fname)
                if not os.path.isfile(img_path):
                    continue
            out.append(fname)

        out.sort()
        return out


class CelebAAttributeDataset(Dataset):
    """
    CelebA binary attribute dataset.

    __getitem__ returns:
      (x, y, g, fname)
        x: transformed image
        y: 0/1 for target_attr
        g: sensitive attribute derived from "Male" (default) as {-1,+1} or {0,1}
        fname: filename string
    """

    def __init__(
        self,
        idx: CelebAIndex,
        split: str,
        target_attr: str,
        transform=None,
        gender_as01: bool = False,
        filenames: Optional[Sequence[str]] = None,
        sensitive_attr: str = "Male",
    ):
        self.idx = idx
        self.split = split
        self.target_attr = target_attr
        self.sensitive_attr = sensitive_attr
        self.transform = transform
        self.gender_as01 = gender_as01

        # Validate attr names early
        attr_names = set(self.idx.attr_names)
        if self.target_attr not in attr_names:
            raise ValueError(f"target_attr='{self.target_attr}' not found in list_attr_celeba.txt")
        if self.sensitive_attr not in attr_names:
            raise ValueError(f"sensitive_attr='{self.sensitive_attr}' not found in list_attr_celeba.txt")

        if filenames is None:
            self.fnames = self.idx.filenames(split=split, require_exists=True)
        else:
            # Caller can restrict population (e.g., to official TRAIN split)
            self.fnames = list(filenames)

        if len(self.fnames) == 0:
            raise RuntimeError(
                f"No filenames found for split='{split}'. "
                f"Check celeba_root='{self.idx.root}' and that img_align_celeba is extracted."
            )

    def __len__(self) -> int:
        return len(self.fnames)

    def _img_path(self, fname: str) -> str:
        return os.path.join(self.idx.img_dir, fname)

    @staticmethod
    def _to_01(v_pm1: int) -> int:
        # CelebA uses {-1, +1}. Map +1 -> 1, -1 -> 0.
        return 1 if v_pm1 == 1 else 0

    @staticmethod
    def _to_pm1(v_pm1: int) -> int:
        # Ensure strict {-1, +1}
        return 1 if v_pm1 == 1 else -1

    def __getitem__(self, i: int):
        fname = self.fnames[i]
        path = self._img_path(fname)

        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(img)
        else:
            # Minimal fallback: convert to tensor (CHW, float in [0,1])
            x = torch.from_numpy(
                (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                 .view(img.size[1], img.size[0], 3)
                 .permute(2, 0, 1)
                 .float() / 255.0).numpy()
            )

        attrs = self.idx.attrs_for(fname)

        y_pm1 = int(attrs[self.target_attr])          # {-1, +1}
        g_pm1 = int(attrs[self.sensitive_attr])       # {-1, +1}

        y = self._to_01(y_pm1)
        if self.gender_as01:
            g = self._to_01(g_pm1)
        else:
            g = self._to_pm1(g_pm1)

        return x, y, g, fname
