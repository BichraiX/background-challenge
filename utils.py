import cv2
import numpy as np
import matplotlib.pyplot as plt

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
def load_image_and_mask(image_path, mask_path):
    """
    Load an RGB image and its ground truth segmentation mask.
    The mask is assumed to be a grayscale image where nonzero pixels are foreground.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to load mask {mask_path}")
    # Convert to a binary mask: foreground if pixel > 0, else background.
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