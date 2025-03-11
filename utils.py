import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

# ---------------------------
# Define segmentation metrics
# ---------------------------
def compute_iou(pred_mask, gt_mask):
    """Compute Intersection over Union (IoU) between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return 1.0 if union == 0 else intersection / union

def compute_dice(pred_mask, gt_mask):
    """Compute Dice coefficient between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    return 1.0 if (pred_sum + gt_sum) == 0 else 2 * intersection / (pred_sum + gt_sum)

# ---------------------------
# Utility functions
# ---------------------------
def load_image_and_mask(image_path, mask_path, dataset_type=None):
    """
    Load an RGB image and its ground truth segmentation mask.
    Handles different dataset formats:
    - DAVIS: Direct grayscale mask
    - COCO: JSON annotations
    - VOC2012: PNG segmentation masks
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Handle different mask formats based on dataset type
    if dataset_type == "COCO":
        # For COCO, mask_path is the annotations JSON file
        with open(mask_path, 'r') as f:
            annotations = json.load(f)
        
        # Get image filename without extension
        img_id = int(os.path.splitext(os.path.basename(image_path))[0])
        
        # Find annotations for this image
        binary_mask = np.zeros(image.shape[:2], dtype=bool)
        for ann in annotations['annotations']:
            if ann['image_id'] == img_id:
                # Convert RLE or polygon to mask
                if 'segmentation' in ann:
                    if isinstance(ann['segmentation'], dict):  # RLE format
                        from pycocotools import mask as mask_utils
                        rle = ann['segmentation']
                        instance_mask = mask_utils.decode(rle)
                    else:  # Polygon format
                        from pycocotools import mask as mask_utils
                        h, w = image.shape[:2]
                        rles = mask_utils.frPyObjects(ann['segmentation'], h, w)
                        instance_mask = mask_utils.decode(rles)
                    
                    # Ensure instance_mask is 2D
                    if instance_mask.ndim == 3:
                        instance_mask = np.any(instance_mask, axis=2)
                    binary_mask = np.logical_or(binary_mask, instance_mask)
    
    elif dataset_type == "VOC2012":
        # For VOC2012, mask is a PNG file with instance segmentations
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to load mask {mask_path}")
        binary_mask = mask > 0
    
    else:  # Default DAVIS format
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to load mask {mask_path}")
        binary_mask = mask > 0

    return image, binary_mask

def plot_metrics(metrics, metric_name):
    """
    Plot a bar graph showing the metric performance for each dataset and overall average.
    'metrics' is a dict with keys for each dataset and "Overall".
    """
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    plt.figure(figsize=(8,6))
    bars = plt.bar(labels, values)
    plt.xlabel("Dataset")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison")
    plt.ylim(0,1)
    # Annotate bars with metric values
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{value:.2f}", ha="center")
    plt.show()

# Define WordNet ID to class name mapping
WORDNET_TO_CLASS = {
    "n02084071": "Dog",
    "n01503061": "Bird", 
    "n04576211": "Vehicle",
    "n01661091": "Reptile",
    "n02075296": "Carnivore",
    "n02159955": "Insect",
    "n03800933": "Instrument",
    "n02469914": "Primate",
    "n02512053": "Fish"
}

# Paths
DATA_PATH = "/Data/amine.chraibi/ImageNet/ImageNet_9_classes"
PROCESSED_DATA_PATH = "/Data/amine.chraibi/processed_dataset"
OUTPUT_DIR = "./results"


class ImageNetSubsetDataset(Dataset):
    """Dataset for loading images from the original ImageNet subset (9 classes)"""
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.images = []
        self.labels = []
        
        # Map WordNet IDs to indices
        self.wordnet_id_to_idx = {}
        for i, (wordnet_id, _) in enumerate(WORDNET_TO_CLASS.items()):
            self.wordnet_id_to_idx[wordnet_id] = i
            
        # Load images from WordNet ID folders
        for wordnet_id, class_name in WORDNET_TO_CLASS.items():
            class_dir = os.path.join(root_dir, wordnet_id)
            if os.path.exists(class_dir):
                print(f"Loading images from {class_dir}")
                self._load_images_from_dir(class_dir, wordnet_id, split)
            else:
                print(f"Warning: Directory not found for {class_name} ({wordnet_id})")
        
        print(f"Total {split} images loaded: {len(self.images)}")
    
    def _load_images_from_dir(self, class_dir, wordnet_id, split):
        """Load images from directory and split into train/val"""
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.jfif', '.bmp', '.tif', '.tiff'))]
        
        # Split files into train/val (80/20 split)
        np.random.shuffle(files)
        split_idx = int(0.8 * len(files))
        
        if split == 'train':
            selected_files = files[:split_idx]
        else:  # val
            selected_files = files[split_idx:]
        
        for file in selected_files:
            img_path = os.path.join(class_dir, file)
            if wordnet_id in self.wordnet_id_to_idx:
                label = self.wordnet_id_to_idx[wordnet_id]
                self.images.append(img_path)
                self.labels.append(label)
            else:
                print(f"Warning: Unknown WordNet ID {wordnet_id}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and the same label
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, label


class MixedRandDataset(Dataset):
    """Dataset for loading images from mixed_rand folders in the processed dataset"""
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.images = []
        self.labels = []
        
        # Dictionary to map WordNet IDs to indices
        self.wordnet_id_to_idx = {}
        for i, (wordnet_id, _) in enumerate(WORDNET_TO_CLASS.items()):
            self.wordnet_id_to_idx[wordnet_id] = i
        
        # Get the list of class names
        class_names = list(WORDNET_TO_CLASS.values())
        class_names = [name.lower() for name in class_names]
        
        # Load images from mixed_rand folders
        for class_name in class_names:
            processed_class_dir = os.path.join(PROCESSED_DATA_PATH, class_name, 'mixed_rand')
            if os.path.exists(processed_class_dir):
                print(f"Loading images from processed dataset: {processed_class_dir}")
                self._load_images_from_dir(processed_class_dir, split)
            else:
                print(f"Warning: No mixed_rand folder found for class {class_name}")
        
        print(f"Total {split} images loaded: {len(self.images)}")
    
    def _load_images_from_dir(self, class_dir, split):
        """Helper method to load and split images from a directory"""
        files = [f for f in os.listdir(class_dir) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
        
        # Split files into train/val (80/20 split)
        np.random.shuffle(files)
        split_idx = int(0.8 * len(files))
        
        if split == 'train':
            selected_files = files[:split_idx]
        else:  # val
            selected_files = files[split_idx:]
        
        for file in selected_files:
            # Extract WordNet ID from filename (n01503061_10009.png)
            match = re.match(r'(n\d+)', file)
            if match:
                img_path = os.path.join(class_dir, file)
                wordnet_id = match.group(1)
                
                # Get label from WordNet ID
                if wordnet_id in self.wordnet_id_to_idx:
                    label = self.wordnet_id_to_idx[wordnet_id]
                    self.images.append(img_path)
                    self.labels.append(label)
                else:
                    print(f"Warning: Unknown WordNet ID {wordnet_id} in file {file}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and the same label
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, label


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%"
    )


def train(model, train_loader, criterion, optimizer, epoch, device):
    """Training function for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for i, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (i + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validation function"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (i + 1),
                'acc': 100. * correct / total
            })
    
    return running_loss / len(val_loader), 100. * correct / total


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot and save training curves"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()