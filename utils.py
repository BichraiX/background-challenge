import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

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