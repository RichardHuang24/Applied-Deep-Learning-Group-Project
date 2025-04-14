"""
Dataset download utilities for Oxford-IIIT Pet Dataset
"""
import os
import tarfile
import urllib.request
import logging
from pathlib import Path
from tqdm import tqdm
import random

logger = logging.getLogger(__name__)

def download_file(url, output_path, desc=None):
    """Download a file with progress bar"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return output_path
    
    logger.info(f"Downloading {url} to {output_path}")
    
    # Get file size for progress reporting
    try:
        response = urllib.request.urlopen(url)
        file_size = int(response.headers.get('Content-Length', 0))
    except:
        file_size = 0
    
    # Download with progress bar
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

def create_train_val_split(dataset_dir, train_ratio=0.8):
    """Create train/val/test splits and save to annotation files"""
    logger.info("Creating dataset splits...")
    annotations_dir = os.path.join(dataset_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Look for images and their classes
    images_dir = os.path.join(dataset_dir, "images")
    classes = {}
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg"):
            # Oxford Pet dataset naming: <class>_<id>.jpg
            class_name = "_".join(filename.split("_")[:-1])
            if class_name not in classes:
                classes[class_name] = []
            classes[class_name].append(filename)
    
    # Create class ID mapping
    class_to_id = {name: i for i, name in enumerate(sorted(classes.keys()))}
    
    # Save class mapping
    with open(os.path.join(annotations_dir, "classes.txt"), 'w') as f:
        for name, idx in class_to_id.items():
            f.write(f"{name} {idx}\n")
    
    # Create balanced train/val/test splits
    train_samples = []
    val_samples = []
    
    for class_name, files in classes.items():
        random.shuffle(files)
        
        # Split images
        split_idx = int(len(files) * train_ratio)
        train_samples.extend([(f, class_to_id[class_name]) for f in files[:split_idx]])
        val_samples.extend([(f, class_to_id[class_name]) for f in files[split_idx:]])
    
    # Shuffle again
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    # Save splits to files
    with open(os.path.join(annotations_dir, "train.txt"), 'w') as f:
        for filename, class_id in train_samples:
            f.write(f"{filename} {class_id}\n")
    
    with open(os.path.join(annotations_dir, "val.txt"), 'w') as f:
        for filename, class_id in val_samples:
            f.write(f"{filename} {class_id}\n")
    
    logger.info(f"Created splits: {len(train_samples)} training, {len(val_samples)} validation samples")
    return True

def download_dataset(dataset_dir, force_download=False):
    """Download and prepare the Oxford-IIIT Pet Dataset"""
    dataset_dir = Path(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    
    logger.info(f"Preparing Oxford-IIIT Pet Dataset in {dataset_dir}")
    
    # Dataset URLs
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    
    # Download images
    images_tar = dataset_dir / "images.tar.gz"
    if force_download or not images_tar.exists():
        download_file(images_url, images_tar, desc="Downloading images")
    
    # Download annotations
    annotations_tar = dataset_dir / "annotations.tar.gz"
    if force_download or not annotations_tar.exists():
        download_file(annotations_url, annotations_tar, desc="Downloading annotations")
    
    # Extract files
    if force_download or not (dataset_dir / "images").exists():
        logger.info("Extracting images...")
        extract_tarfile(images_tar, dataset_dir, desc="Extracting images")
    
    if force_download or not (dataset_dir / "annotations").exists():
        logger.info("Extracting annotations...")
        extract_tarfile(annotations_tar, dataset_dir, desc="Extracting annotations")
    
    # Create train/val splits if they don't exist
    if not (dataset_dir / "annotations" / "train.txt").exists():
        create_train_val_split(dataset_dir)
    
    # Verify dataset
    if verify_dataset(dataset_dir):
        logger.info("Dataset verified successfully")
        return True
    else:
        logger.error("Dataset verification failed")
        return False

def verify_dataset(dataset_dir):
    """Verify that the dataset is properly downloaded and extracted"""
    dataset_dir = Path(dataset_dir)
    
    # Check for expected directories
    if not (dataset_dir / "images").exists():
        logger.error("Images directory not found")
        return False
    
    if not (dataset_dir / "annotations").exists():
        logger.error("Annotations directory not found")
        return False
    
    # Check for train/val splits
    if not (dataset_dir / "annotations" / "train.txt").exists():
        logger.error("Train split file not found")
        return False
    
    if not (dataset_dir / "annotations" / "val.txt").exists():
        logger.error("Validation split file not found")
        return False
    
    # Count images
    image_dir = dataset_dir / "images"
    num_images = len(list(image_dir.glob("*.jpg")))
    if num_images == 0:
        logger.error("No images found")
        return False
    
    # Count annotations
    trimap_dir = dataset_dir / "annotations" / "trimaps"
    num_trimaps = len(list(trimap_dir.glob("*.png"))) if trimap_dir.exists() else 0
    
    logger.info(f"Dataset verification complete: {num_images} images, {num_trimaps} trimaps")
    return True