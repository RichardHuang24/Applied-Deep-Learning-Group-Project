"""
oxford_pet_setup.py
───────────────────
Download, extract and prepare the Oxford‑IIIT Pet data set.

Key upgrade: the `create_train_val_split` routine now performs a
**stratified split**, so each of the 37 breeds keeps ≈ 80 % of its
images for training and 20 % for validation.

Author: you
"""
from __future__ import annotations
import os, tarfile, urllib.request, random, logging
from pathlib import Path
from collections import Counter
from typing import List, Tuple
from tqdm import tqdm
from utils.logging import setup_logging                    

logger = logging.getLogger(__name__)


def download_file(url: str, dst: Path, desc: str | None = None) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        logger.info(f"File already exists: {dst}")
        return dst

    logger.info(f"Downloading {url}")
    try:
        size = int(urllib.request.urlopen(url).headers.get('Content-Length', 0))
    except Exception:
        size = 0

    with tqdm(total=size, unit='B', unit_scale=True, desc=desc or dst.name) as pbar:
        def hook(block_num, block_size, total_size):
            pbar.update(block_size)
        urllib.request.urlretrieve(url, dst, reporthook=hook)

    return dst


def extract_tar(tar_path: Path, dst_dir: Path, desc: str | None = None):
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc=desc or f"Extract {tar_path.name}") as pbar:
            for m in members:
                tar.extract(m, path=dst_dir)
                pbar.update(1)


# ───────────────── stratified split ───────────────── #

def create_train_val_split(dataset_dir: Path,
                           train_ratio: float = 0.8,
                           seed: int = 42,
                           n_classes: int = 37) -> None:
    """
    Build annotations/train.txt and annotations/val.txt with a
    **class‑balanced** split.
    """
    rng = random.Random(seed)
    ann_dir = dataset_dir / "annotations"
    src_list = ann_dir / "trainval.txt"

    if not src_list.exists():
        raise FileNotFoundError(src_list)

    # read, skip 6‑line header if present
    with open(src_list) as f:
        lines = f.readlines()
        if lines[0].startswith('#'):
            lines = lines[6:]

    buckets: List[List[Tuple[str, int]]] = [[] for _ in range(n_classes)]
    for ln in lines:
        fname, cls_id, *_ = ln.strip().split()
        cls_id = int(cls_id) - 1                       # 1‑based → 0‑based
        buckets[cls_id].append((fname, cls_id))

    train, val = [], []
    for cls_samples in buckets:
        rng.shuffle(cls_samples)
        k = int(len(cls_samples) * train_ratio)
        train.extend(cls_samples[:k])
        val.extend(cls_samples[k:])

    rng.shuffle(train); rng.shuffle(val)

    def write_file(split: List[Tuple[str, int]], path: Path):
        with open(path, 'w') as f:
            for fname, cid in split:
                f.write(f"{fname}.jpg {cid}\n")

    write_file(train, ann_dir / "train.txt")
    write_file(val,   ann_dir / "val.txt")
    logger.info(f"Balanced split: {len(train)} train / {len(val)} val")


def check_balance(ann_dir: Path, n_classes: int = 37):
    def counts(txt):
        c = Counter()
        with open(txt) as f:
            for ln in f:
                _, cid = ln.strip().split()
                c[int(cid)] += 1
        return c

    tr, vl = counts(ann_dir / "train.txt"), counts(ann_dir / "val.txt")
    logger.info("Class  train  val")
    for k in range(n_classes):
        logger.info(f"{k:5d}  {tr[k]:5d}  {vl[k]:4d}")


def verify_dataset(dataset_dir: Path) -> bool:
    req = [dataset_dir / "images",
           dataset_dir / "annotations" / "train.txt",
           dataset_dir / "annotations" / "val.txt"]
    for p in req:
        if not p.exists():
            logger.error(f"Missing {p}")
            return False

    n_img  = len(list((dataset_dir / "images").glob("*.jpg")))
    n_mask = len(list((dataset_dir / "annotations" / "trimaps").glob("*.png")))
    logger.info(f"Dataset OK: {n_img} images, {n_mask} trimaps")
    return True


def download_dataset(dataset_dir: str | Path,
                     force: bool = False,
                     seed: int = 42):

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(exist_ok=True)

    setup_logging(Path('output/downloads'), log_name="download.log")
    logger.info(f"Preparing Oxford‑IIIT Pet at {dataset_dir}")

    urls = {
        "images":       "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
        "annotations":  "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    }

    # download & extract
    img_tar  = dataset_dir / "images.tar.gz"
    ann_tar  = dataset_dir / "annotations.tar.gz"
    if force or not img_tar.exists():
        download_file(urls["images"], img_tar, "images.tar.gz")
    if force or not ann_tar.exists():
        download_file(urls["annotations"], ann_tar, "annotations.tar.gz")

    if force or not (dataset_dir / "images").exists():
        extract_tar(img_tar, dataset_dir, "images")
    if force or not (dataset_dir / "annotations").exists():
        extract_tar(ann_tar, dataset_dir, "annotations")

    # balanced split
    if not (dataset_dir / "annotations" / "train.txt").exists():
        create_train_val_split(dataset_dir, train_ratio=0.8, seed=seed)

    check_balance(dataset_dir / "annotations")          # optional log

    if verify_dataset(dataset_dir):
        logger.info("Dataset ready")
    else:
        logger.error("Dataset verification failed")

