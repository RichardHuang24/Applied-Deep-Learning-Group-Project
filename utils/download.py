"""
Dataset download and preparation for Oxford-IIIT Pet Dataset
"""
import os
import tarfile
import urllib.request
import logging
from pathlib import Path
from tqdm import tqdm
import random
from utils.logging import setup_logging


logger = logging.getLogger(__name__)

def download_file(url, output_path, desc=None):
    """Download a file with progress bar"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return output_path

    logger.info(f"Downloading {url} to {output_path}")

    try:
        response = urllib.request.urlopen(url)
        file_size = int(response.headers.get('Content-Length', 0))
    except Exception:
        file_size = 0

    with tqdm(total=file_size, unit='B', unit_scale=True, desc=desc or "Downloading") as pbar:
        def report_progress(block_num, block_size, total_size):
            pbar.update(block_size)

        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)

    return output_path

def extract_tarfile(tar_path, output_dir, desc=None):
    """Extract a tar file with progress bar"""
    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc=desc or "Extracting") as pbar:
            for member in members:
                tar.extract(member, path=output_dir)
                pbar.update(1)

def create_train_val_split(dataset_dir, train_ratio=0.8, seed=42):
    """
    Split the full trainval.txt-style file into train.txt and val.txt
    Format: <filename> <class_id> <species> <breed>
    Output: <filename>.jpg <class_id - 1>
    """
    logger.info("Creating train/val split from trainval.txt-style file...")
    random.seed(seed)

    annotations_dir = os.path.join(dataset_dir, "annotations")
    list_path = os.path.join(annotations_dir, "trainval.txt")
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"trainval.txt not found at {list_path}")

    # Read lines (skip first 6 lines if it's the original file with a header)
    with open(list_path, 'r') as f:
        lines = f.readlines()
        if lines[0].startswith('#'):
            lines = lines[6:]

    samples = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            filename = parts[0]
            class_id = int(parts[1]) - 1  # Convert from 1-based to 0-based indexing
            samples.append((filename, class_id))

    # Shuffle and split
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Save to train.txt and val.txt
    def write_split(path, split_samples):
        with open(path, 'w') as f:
            for filename, class_id in split_samples:
                f.write(f"{filename}.jpg {class_id}\n")

    write_split(os.path.join(annotations_dir, "train.txt"), train_samples)
    write_split(os.path.join(annotations_dir, "val.txt"), val_samples)

    logger.info(f"Train/val split complete: {len(train_samples)} train, {len(val_samples)} val.")
    return True


def verify_dataset(dataset_dir):
    """Verify the dataset structure and contents"""
    dataset_dir = Path(dataset_dir)

    if not (dataset_dir / "images").exists():
        logger.error("Missing images/ directory.")
        return False

    if not (dataset_dir / "annotations").exists():
        logger.error("Missing annotations/ directory.")
        return False

    if not (dataset_dir / "annotations" / "train.txt").exists():
        logger.error("Missing train.txt")
        return False

    if not (dataset_dir / "annotations" / "val.txt").exists():
        logger.error("Missing val.txt")
        return False

    num_images = len(list((dataset_dir / "images").glob("*.jpg")))
    num_trimaps = len(list((dataset_dir / "annotations" / "trimaps").glob("*.png")))

    logger.info(f"Dataset verification complete: {num_images} images, {num_trimaps} trimaps")
    return num_images > 0 and num_trimaps > 0

def download_dataset(dataset_dir, force_download=False, seed=42):
    """Download and prepare the Oxford-IIIT Pet Dataset"""
    dataset_dir = Path(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    # Setup logging
    log_dir = Path('output') / "downloads"
    setup_logging(log_dir, log_name="download.log")

    logger.info(f"Preparing Oxford-IIIT Pet Dataset in {dataset_dir}")

    # Dataset URLs
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    # Download
    images_tar = dataset_dir / "images.tar.gz"
    annotations_tar = dataset_dir / "annotations.tar.gz"

    if force_download or not images_tar.exists():
        download_file(images_url, images_tar, desc="Downloading images")

    if force_download or not annotations_tar.exists():
        download_file(annotations_url, annotations_tar, desc="Downloading annotations")

    # Extract
    if force_download or not (dataset_dir / "images").exists():
        extract_tarfile(images_tar, dataset_dir, desc="Extracting images")

    if force_download or not (dataset_dir / "annotations").exists():
        extract_tarfile(annotations_tar, dataset_dir, desc="Extracting annotations")

    # Create train/val split from official trainval.txt
    train_txt = dataset_dir / "annotations" / "train.txt"
    if not train_txt.exists():
        create_train_val_split(dataset_dir, train_ratio=0.8, seed=seed)

    # Final check
    if verify_dataset(dataset_dir):
        logger.info("Dataset verified successfully.")
        return True
    else:
        logger.error("Dataset verification failed.")
        return False
