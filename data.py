"""
Dataset handling and processing for Oxford-IIIT Pet Dataset
"""
import os
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

logger = logging.getLogger(__name__)

class OxfordPetsDataset(Dataset):
    """Dataset class for Oxford-IIIT Pet Dataset with image-level labels"""
    def __init__(self, root, split='trainval', transform=None):
        """
        Args:
            root (str): Root directory of the dataset
            split (str): Dataset split ('trainval', 'train', 'val', 'test')
            transform: Optional transforms to be applied on images
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # Load class names and mapping
        self.classes, self.class_to_idx = self._load_classes()
        
        # Load image paths and labels
        self.samples = self._load_split()
        
    def _load_classes(self):
        """Load class names and create mapping"""
        classes = []
        class_to_idx = {}
        
        # Extract class names from file structure
        image_dir = self.root / "images"
        for image_path in sorted(image_dir.glob("*.jpg")):
            class_name = '_'.join(image_path.stem.split('_')[:-1])
            if class_name not in classes:
                classes.append(class_name)
                class_to_idx[class_name] = len(classes) - 1
        
        return classes, class_to_idx
    
    def _load_split(self):
        """Load image paths and labels for the specified split"""
        samples = []
        
        # Load split file
        split_file = self.root / "annotations" / f"{self.split}.txt"
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        # Parse split file
        for line in lines:
            img_name = line.strip().split()[0]
            
            # Extract class name from image name
            class_name = '_'.join(img_name.split('_')[:-1])
            
            # Create image path
            img_path = str(self.root / "images" / f"{img_name}")
            
            # Get class index
            class_idx = self.class_to_idx[class_name]
            
            samples.append((img_path, class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            tuple: (image, target, image_path, original_image)
        """
        img_path, target = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Store original image for visualization
        original_img = img.copy()
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, target, img_path, original_img

class OxfordPetsSegmentationDataset(Dataset):
    """Dataset class for Oxford-IIIT Pet Dataset with segmentation masks"""
    def __init__(self, root, split='trainval', supervision='full', 
                 pseudo_masks_dir=None, transform=None, mask_transform=None):
        """
        Args:
            root (str): Root directory of the dataset
            split (str): Dataset split ('trainval', 'train', 'val', 'test')
            supervision (str): Type of supervision ('full', 'weak_gradcam', 'weak_ccam')
            pseudo_masks_dir (str): Directory with pseudo masks (for weak supervision)
            transform: Optional transforms to be applied on images
            mask_transform: Optional transforms to be applied on masks
        """
        self.root = Path(root)
        self.split = split
        self.supervision = supervision
        self.pseudo_masks_dir = Path(pseudo_masks_dir) if pseudo_masks_dir else None
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Load image paths and class labels
        self.samples = self._load_split()
        
    def _load_split(self):
        """Load image paths for the specified split"""
        samples = []
        
        # Load split file
        split_file = self.root / "annotations" / f"{self.split}.txt"
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        # Parse split file
        for line in lines:
            parts = line.strip().split()
            img_name = parts[0]
            
            # Create image path
            img_path = str(self.root / "images" / f"{img_name}")
            
            # Create ground truth mask path (trimap)
            gt_mask_path = str(self.root / "annotations" / "trimaps" / f"{img_name}.png")
            
            # Add to samples
            samples.append((img_path, gt_mask_path, img_name))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns:
            dict: Dictionary with image, masks and metadata
        """
        img_path, gt_mask_path, img_name = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Load ground truth mask
        gt_mask = Image.open(gt_mask_path).convert('L')
        
        # Load pseudo mask if using weak supervision
        if self.supervision.startswith('weak_') and self.pseudo_masks_dir:
            pseudo_mask_path = self.pseudo_masks_dir / f"{img_name}.png"
            if pseudo_mask_path.exists():
                pseudo_mask = Image.open(pseudo_mask_path).convert('L')
            else:
                # If no pseudo mask, just use zeros
                pseudo_mask = Image.new('L', gt_mask.size, 0)
        else:
            # For full supervision, use ground truth
            pseudo_mask = gt_mask.copy()
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        if self.mask_transform:
            gt_mask = self.mask_transform(gt_mask)
            pseudo_mask = self.mask_transform(pseudo_mask)
        
        # Create sample dictionary
        sample = {
            'image': img,
            'gt_mask': gt_mask,
            'pseudo_mask': pseudo_mask,
            'path': img_path
        }
        
        return sample

def custom_collate_fn(batch):
    """
    Simple custom collate function that handles tensors of different sizes
    """
    # Extract components based on the actual data structure observed
    images = []
    labels = []
    paths = []
    filenames = []
    
    for item in batch:
        # First element (index 0) is the image tensor with shape [3, 224, 224]
        images.append(item[0])
        
        # Second element (index 1) seems to be another tensor, not an integer label
        # We need to extract the actual class label from this tensor or elsewhere
        # For now, we'll assume this is indeed some kind of label encoding
        if isinstance(item[1], torch.Tensor):
            # Convert to a single class index if possible
            if item[1].numel() == 1:
                labels.append(item[1].item())
            else:
                # If it's a multi-dimensional tensor, you might need to process it
                # For now, just use a placeholder
                labels.append(0)
        else:
            # If it's already a scalar, use it directly
            labels.append(item[1])
        
        # Third element (index 2) is the image path
        paths.append(item[2])
        
        # Fourth element (index 3) is the image filename
        filenames.append(item[3])
    
    # Stack the images - they should all be [3, 224, 224] by now
    images = torch.stack(images)
    
    # Convert labels to tensor if possible
    try:
        labels = torch.tensor(labels, dtype=torch.long)
    except:
        # Keep as list if conversion fails
        pass
    
    return images, labels, paths, filenames

def create_dataloaders(config, split='train', batch_size=None, supervision='full', pseudo_masks_dir=None):
    """
    Create dataloaders for training and validation
    
    Args:
        config: Configuration dictionary
        split: Data split ('train', 'val')
        batch_size: Batch size (overrides config)
        supervision: Supervision type ('full', 'weak_gradcam', 'weak_ccam')
        pseudo_masks_dir: Directory containing pseudo-masks for weak supervision
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get parameters from config
    if batch_size is None:
        batch_size = config.get('training', {}).get('batch_size', 32)
    
    dataset_root = config.get('dataset', {}).get('root', 'dataset')
    num_workers = config.get('training', {}).get('num_workers', 4)
    
    # Create transforms - ENSURE CONSISTENT SIZE
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Fixed size for all images
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Fixed size for all images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    if supervision == 'full' or supervision.startswith('weak_'):
        # For segmentation, we need both image and mask
        train_dataset = SegmentationDataset(
            root=dataset_root,
            split='train',
            transform=train_transform,
            supervision=supervision,
            pseudo_masks_dir=pseudo_masks_dir
        )
        
        val_dataset = SegmentationDataset(
            root=dataset_root,
            split='val',
            transform=val_transform,
            supervision=supervision,
            pseudo_masks_dir=pseudo_masks_dir
        )
    else:
        # For classification, we only need image and label
        train_dataset = ClassificationDataset(
            root=dataset_root,
            split='train',
            transform=train_transform
        )
        
        val_dataset = ClassificationDataset(
            root=dataset_root,
            split='val',
            transform=val_transform
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader

class ClassificationDataset(Dataset):
    """Dataset for image classification"""
    
    def __init__(self, root, split='train', transform=None):
        """
        Initialize the dataset
        
        Args:
            root: Root directory of the dataset
            split: Data split ('train', 'val')
            transform: Image transforms
        """
        self.root = root
        self.img_dir = os.path.join(root, 'images')
        self.split = split
        self.transform = transform
        
        # Load samples
        self.samples = self._load_annotations(split)
    
    def __len__(self):
        """Return the number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        img_file, label = self.samples[idx]
        
        # Handle image path correctly
        if img_file.endswith('.jpg'):
            img_path = os.path.join(self.img_dir, img_file)
        else:
            img_path = os.path.join(self.img_dir, f"{img_file}.jpg")
        
        # Load image and convert to tensor
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                # This should convert PIL image to tensor with proper normalization
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return dummy data on error
            img = torch.zeros(3, 224, 224)
        
        # Ensure label is an integer
        try:
            label = int(label)
        except (ValueError, TypeError):
            print(f"Warning: Invalid label {label}, using 0")
            label = 0
        
        # The image should be a tensor of shape [3, H, W]
        # The label should be a simple integer
        # The img_path and img_file should be strings
        return img, label, img_path, img_file
    
    def _load_annotations(self, split):
        """Load annotations from file"""
        annotations_path = os.path.join(self.root, 'annotations', f'{split}.txt')
        
        samples = []
        try:
            with open(annotations_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        try:
                            class_id = int(parts[1])
                            samples.append((filename, class_id))
                        except ValueError:
                            print(f"Warning: Skipping sample with invalid class ID: {parts[1]}")
        except Exception as e:
            print(f"Error loading annotations from {annotations_path}: {e}")
        
        return samples

class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation"""
    
    def __init__(self, root, split='train', transform=None, supervision='full', pseudo_masks_dir=None):
        """
        Initialize the dataset
        
        Args:
            root: Root directory of the dataset
            split: Data split ('train', 'val')
            transform: Image transforms
            supervision: Supervision type ('full', 'weak_gradcam', 'weak_ccam')
            pseudo_masks_dir: Directory containing pseudo-masks for weak supervision
        """
        self.root = root
        self.img_dir = os.path.join(root, 'images')
        self.split = split
        self.transform = transform
        self.supervision = supervision
        
        # Load samples
        self.samples = self._load_annotations(split)
        
        # Set up mask directory based on supervision type
        if supervision == 'full':
            self.mask_dir = os.path.join(root, 'annotations', 'trimaps')
        elif supervision.startswith('weak_') and pseudo_masks_dir is not None:
            self.mask_dir = pseudo_masks_dir
        else:
            raise ValueError(f"Invalid supervision type: {supervision}")
    
    def __len__(self):
        """Return the number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        img_file, label = self.samples[idx]
        
        # Handle image path correctly
        if img_file.endswith('.jpg'):
            img_path = os.path.join(self.img_dir, img_file)
            file_root = img_file[:-4]  # Remove .jpg
        else:
            img_path = os.path.join(self.img_dir, f"{img_file}.jpg")
            file_root = img_file
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, f"{file_root}.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            # If mask doesn't exist, create a blank mask
            mask = Image.new('L', img.size, 0)
        
        # Resize both image and mask to the same size
        if self.transform:
            # Logging info - img / mask size before transform
            logger.debug(f"Image size before transform: {img.size}")
            logger.debug(f"Mask size before transform: {mask.size}")
            img = self.transform(img)
            
            # Logging info - img / mask size after transform
            logger.debug(f"Image size after transform: {img.size}")
            logger.debug(f"Mask size after transform: {mask.size}")

        # Convert mask to tensor and adjust labels if needed
        # to_tensor = transforms.ToTensor()
        # mask_tensor = to_tensor(mask) * 255  # Scale mask to [0, 255]
        # mask_tensor = mask_tensor.long()
        
        # mask = transforms.ToTensor()(mask) * 255
        
        # Convert single-channel mask to class indices (H, W)
        # mask = mask.squeeze(0).long()

        mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match image size
            transforms.ToTensor()
        ])

        mask = mask_transform(mask)

        return img, mask, img_path, img_file
    
    def _load_annotations(self, split):
        """Load annotations from file"""
        annotations_path = os.path.join(self.root, 'annotations', f'{split}.txt')
        
        samples = []
        with open(annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    class_id = int(parts[1])
                    samples.append((filename, class_id))
        
        return samples