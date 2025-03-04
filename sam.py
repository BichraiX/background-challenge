import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Import SAM modules – make sure you have cloned Meta’s SAM repo and installed its dependencies
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from utils import load_image_and_mask, compute_iou, compute_dice


def generate_sam_mask(image, mask_generator):
    """
    Use SAM's automatic mask generator to produce segmentation masks for an image.
    Returns a single binary mask as the union of all predicted masks.
    """
    masks = mask_generator.generate(image)
    combined_mask = np.zeros(image.shape[:2], dtype=bool)
    for m in masks:
        combined_mask = np.logical_or(combined_mask, m['segmentation'])
    return combined_mask

def process_dataset(dataset_name, image_dir, mask_dir, mask_generator, num_samples=50):
    """
    Process a dataset:
      - Randomly select num_samples images.
      - For each image, load the ground truth mask, generate the predicted mask using SAM,
        and compute IoU and Dice metrics.
    Returns the average IoU and Dice across the processed images.
    """
    print(f"\nProcessing {dataset_name} dataset...")
    # List image files (we assume mask files share the same filename)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) < num_samples:
        print(f"Warning: {dataset_name} has only {len(image_files)} images; using all available images.")
        num_samples = len(image_files)
    selected_files = random.sample(image_files, num_samples)
    
    iou_list = []
    dice_list = []
    
    for fname in selected_files:
        image_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)  # Assumes same filename exists in mask_dir
        try:
            image, gt_mask = load_image_and_mask(image_path, mask_path)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue
        
        # Generate predicted foreground mask using SAM
        pred_mask = generate_sam_mask(image, mask_generator)
        
        # Compute metrics
        iou = compute_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)
        
        iou_list.append(iou)
        dice_list.append(dice)
    
    avg_iou = np.mean(iou_list) if iou_list else 0
    avg_dice = np.mean(dice_list) if dice_list else 0
    
    print(f"{dataset_name}: Average IoU = {avg_iou:.4f}, Average Dice = {avg_dice:.4f}")
    return avg_iou, avg_dice


# ---------------------------
# Main processing function
# ---------------------------
def main():
    random.seed(42)
    
    # ---------------------------
    # Setup SAM Model
    # ---------------------------
    sam_checkpoint = "path/to/sam_checkpoint.pth"  # <-- Specify path to your SAM checkpoint file
    model_type = "vit_h"  # Options: "vit_h", "vit_l", "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    # ---------------------------
    # Define dataset paths
    # ---------------------------
    # Make sure these directories point to the locations where you have stored the images and corresponding masks.
    datasets = {
        "Pascal VOC": {
            "image_dir": "path/to/VOC2012/JPEGImages",
            "mask_dir": "path/to/VOC2012/SegmentationClass"
        },
        "COCO": {
            "image_dir": "path/to/COCO/images",  # Adjust to your COCO image directory
            "mask_dir": "path/to/COCO/masks"      # This should be a directory with preprocessed binary masks
        },
        "DAVIS": {
            "image_dir": "path/to/DAVIS/JPEGImages/480p",
            "mask_dir": "path/to/DAVIS/Annotations/480p"
        }
    }
    
    results_iou = {}
    results_dice = {}
    
    # ---------------------------
    # Process each dataset
    # ---------------------------
    for name, paths in datasets.items():
        avg_iou, avg_dice = process_dataset(
            name,
            image_dir=paths["image_dir"],
            mask_dir=paths["mask_dir"],
            mask_generator=mask_generator,
            num_samples=50
        )
        results_iou[name] = avg_iou
        results_dice[name] = avg_dice
    
    # Compute overall average (mean of the three datasets)
    overall_iou = np.mean(list(results_iou.values()))
    overall_dice = np.mean(list(results_dice.values()))
    
    results_iou["Overall"] = overall_iou
    results_dice["Overall"] = overall_dice
    
    print("\nFinal Average Metrics:")
    for dataset in results_iou:
        print(f"{dataset}: IoU = {results_iou[dataset]:.4f}, Dice = {results_dice[dataset]:.4f}")
    
    # ---------------------------
    # Plot the metrics as bar graphs
    # ---------------------------
    # Bar graph for IoU
    plot_metrics(results_iou, "Intersection over Union (IoU)")
    # Bar graph for Dice Coefficient
    plot_metrics(results_dice, "Dice Coefficient")

if __name__ == "__main__":
    main()
