"""
Dataset handling and processing for Oxford-IIIT Pet Dataset
"""
import os
import json
import logging
from pathlib import Path
import random
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

logger = logging.getLogger(__name__)


def data_loaders(split='train', batch_size=32, num_workers=4, shuffle=True, return_bbox=False,
                  return_trimaps=False, return_pseudomask=False,
                  label_type="breed", transform=None, transform_trimaps=None,
                  transform_pseudomasks=None):
    """
    Create data loaders for the dataset

    Args:
        dataset: Dataset object
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        DataLoader: Data loader for the dataset
    """
    dataset = OxfordPetDataset(split=split,
                               transform=transform,
                               return_bbox=return_bbox,
                               return_trimaps=return_trimaps,
                               return_pseudomask=return_pseudomask,
                               transform_trimaps=transform_trimaps,
                               transform_pseudomasks=transform_pseudomasks,
                               label_type=label_type)
    print(f"Loaded {len(dataset)} samples for split={split}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def get_train_transform():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Fixed size for all images
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((224, 224)),  # Random crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform

def get_val_transform():
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Fixed size for all images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return val_transform

def get_trimap_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize trimaps to match image size
        transforms.ToTensor()
    ])

class OxfordPetDataset(Dataset):
    def __init__(self, root="dataset", root1="WSOL", split="trainval", transform=None,
                 return_bbox=False, label_type="breed",
                 return_trimaps=False, transform_trimaps=None,
                 return_pseudomask=False, transform_pseudomasks=None):
        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.annotation_dir = os.path.join(root, "annotations")
        self.trimaps_dir = os.path.join(self.annotation_dir, "trimaps")
        # self.pseudomask_dir = os.path.join(root1, "pseudo_masks")
        self.xml_dir = os.path.join(self.annotation_dir, "xmls")

        self.transform = transform
        self.transform_trimaps = transform_trimaps
        self.transform_pseudomasks = transform_pseudomasks

        self.return_bbox = return_bbox
        self.return_trimaps = return_trimaps
        self.return_pseudomask = return_pseudomask
        self.label_type = label_type

        if self.transform is None:
            self.transform = get_train_transform() if split == "train" else get_val_transform()
        
        if self.return_trimaps and self.transform_trimaps is None:
            self.transform_trimaps = get_trimap_transform() if split == "train" else get_val_transform()
        if self.return_pseudomask and self.transform_pseudomasks is None:
            self.transform_pseudomasks = get_trimap_transform() if split == "train" else get_val_transform()
            
        # Load annotation list
        list_path = os.path.join(self.annotation_dir, f"{split}.txt") if split != "all" else os.path.join(self.annotation_dir, "list.txt")

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Annotation file not found: {list_path}")

        with open(list_path, "r") as f:
            lines = f.readlines()

        # Skip comments/header lines
        lines = [line for line in lines if not line.startswith("#") and line.strip()]

        self.samples = []

        for line in lines:
            parts = line.strip().split()

            if len(parts) == 2:
                # Format: filename label (used in train/val)
                filename, label = parts[0].split(".")[0], int(parts[1])
            elif len(parts) >= 3:
                # Format: filename class_id species_id (used in test/list.txt)
                filename = parts[0].split(".")[0]  # Remove file extension
                class_id = int(parts[1]) - 1  # convert to 0-based
                species_id = int(parts[2]) - 1
                label = class_id if label_type == "breed" else species_id
            else:
                raise ValueError(f"Invalid line format: {line}")

            # Ignore macOS hidden files like ._Abyssinian_100
            if filename.startswith("._"):
                raise ValueError(f"Invalid filename: {filename}")

            self.samples.append((filename, label))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]

        # Load image
        img_path = os.path.join(self.image_dir, f"{filename}.jpg")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load trimap (optional)
        trimap = None
        if self.return_trimaps or self.return_pseudomask:
            trimap_path = os.path.join(self.trimaps_dir, f"{filename}.png")
            trimap = Image.open(trimap_path).convert('L')
            trimap_np = np.array(trimap)
            trimap_np[trimap_np == 3] = 1  # Background → 0, Foreground → 1
            trimap_np -= 1
            trimap = Image.fromarray(trimap_np.astype(np.uint8))

        # Load pseudomask (optional)
        pseudomask = None
        if self.return_pseudomask:
            pseudo_path = os.path.join(self.pseudomask_dir, f"{filename}.png")
            pseudomask = Image.open(pseudo_path).convert('L')
            if self.transform_pseudomasks:
                pseudomask = self.transform_pseudomasks(pseudomask)

        # Apply trimap transform if needed
        if self.return_trimaps or self.return_pseudomask:
            if self.transform_trimaps:
                trimap = self.transform_trimaps(trimap)
            trimap = (trimap * 255).round().long().squeeze(0)
            trimap = 1 - trimap  # 1 = foreground, 0 = background

        # Load bounding box (optional)
        if self.return_bbox:
            xml_path = os.path.join(self.xml_dir, f"{filename}.xml")
            bbox = self._load_bbox_from_xml(xml_path)
            return image, label, bbox

        # Return depending on supervision mode
        if self.return_pseudomask:
            return image, pseudomask, trimap, label
        elif self.return_trimaps:
            return image, trimap, label
        else:
            return image, label, filename

    def _load_bbox_from_xml(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj = root.find('object')
            bbox = obj.find('bndbox') if obj is not None else None
            if bbox:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                return (xmin, ymin, xmax, ymax)
        except Exception:
            pass
        return (0, 0, 0, 0)

