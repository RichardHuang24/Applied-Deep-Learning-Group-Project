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
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)


def data_loaders(split='train', batch_size=32, num_workers=4, shuffle=True, return_bbox=False,
                  return_trimaps=False, return_pseudomask=False,
                  label_type="breed", transform=None, transform_trimaps=None,
                  transform_pseudomasks=None, mask_dir=None):
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
                               mask_dir=mask_dir,
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
        transforms.Resize((256, 256)),  # Resize all images to a fixed size
        transforms.RandomHorizontalFlip(),  # Apply random horizontal flip
        transforms.RandomCrop((224, 224)),  # Randomly crop to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    return train_transform

def get_val_transform():
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    return val_transform

def get_trimap_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize trimaps to match image size
        transforms.ToTensor()  # Convert trimaps to tensor
    ])


class SyncedTransform:
    """Apply the same spatial transformations to both images and masks."""
    def __init__(self, img_size=256, crop_size=224, p_flip=0.5, normalize=True):
        self.img_size = img_size
        self.crop_size = crop_size
        self.p_flip = p_flip
        self.normalize = normalize
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def __call__(self, image, mask):
        # Resize both image and mask
        image = TF.resize(image, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size))
        
        # Apply random horizontal flip with the same random state
        if random.random() < self.p_flip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Apply random crop with the same crop parameters
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.crop_size, self.crop_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        
        # Convert both image and mask to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # Normalize the image only
        if self.normalize:
            image = TF.normalize(image, self.mean, self.std)
            
        # Process the mask: scale to [0, 255], round, and convert to long
        mask = (mask * 255).round().long().squeeze(0)
        mask = 1 - mask  # Keep the logic: 1 = foreground, 0 = background
        
        return image, mask
    
class OxfordPetDataset(Dataset):
    def __init__(self, root="dataset", split="trainval", transform=None,
                 return_bbox=False, label_type="breed",
                 return_trimaps=False, transform_trimaps=None,
                 return_pseudomask=False, transform_pseudomasks=None, mask_dir=None):
        self.root = root
        self.split = split
        self.image_dir = os.path.join(root, "images")
        self.annotation_dir = os.path.join(root, "annotations")
        self.trimaps_dir = os.path.join(self.annotation_dir, "trimaps")
        self.pseudomask_dir = mask_dir
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
            self.transform_trimaps = get_trimap_transform()
        if self.return_pseudomask and self.transform_pseudomasks is None:
            if self.pseudomask_dir is None:
                raise ValueError("Pseudomask directory is not provided.")
            self.transform_pseudomasks = get_trimap_transform()
            
        # Load the annotation list
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
                class_id = int(parts[1]) - 1  # Convert to zero-based
                species_id = int(parts[2]) - 1
                label = class_id if label_type == "breed" else species_id
            else:
                raise ValueError(f"Invalid line format: {line}")

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

        # Load trimap (optional)
        trimap = None
        if self.return_trimaps or self.return_pseudomask:
            trimap_path = os.path.join(self.trimaps_dir, f"{filename}.png")
            trimap = Image.open(trimap_path).convert('L')
            trimap_np = np.array(trimap)
            # In Oxford Pet, 3 indicates background. We map background to 0 and foreground to 1.
            trimap_np[trimap_np == 3] = 1
            trimap_np -= 1
            trimap = Image.fromarray(trimap_np.astype(np.uint8))

        # Load pseudomask (optional)
        pseudomask = None
        if self.return_pseudomask:
            pseudo_path = os.path.join(self.pseudomask_dir, f"{filename}.png")
            if os.path.exists(pseudo_path):
                pseudomask = Image.open(pseudo_path).convert('L')
            else:
                logger.warning(f"Pseudomask not found: {pseudo_path}")
                pseudomask = Image.new('L', (224, 224), 0)

        # Apply transforms
        if (self.return_trimaps or self.return_pseudomask) and trimap is not None and self.split == 'train':
            # Use synchronized transforms for the training set
            synced_transform = SyncedTransform()
            image, trimap = synced_transform(image, trimap)
        else:
            # Use standard transforms for non-training sets or when masks are not required
            if self.transform:
                image = self.transform(image)
                
            if (self.return_trimaps or self.return_pseudomask) and trimap is not None:
                if self.transform_trimaps:
                    trimap = self.transform_trimaps(trimap)
                    trimap = (trimap * 255).round().long().squeeze(0)
                    trimap = 1 - trimap  # 1 = foreground, 0 = background

        if self.return_pseudomask and pseudomask is not None and self.transform_pseudomasks:
            pseudomask = self.transform_pseudomasks(pseudomask)

        # Return based on supervision mode
        if self.return_pseudomask:
            return image, pseudomask, trimap
        elif self.return_trimaps:
            return image, trimap, filename
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

