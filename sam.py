import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# Import SAM modules – make sure you have cloned Meta's SAM repo and installed its dependencies
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from utils import load_image_and_mask, compute_iou, compute_dice, plot_metrics


def save_visualization(image, gt_mask, pred_mask, output_dir, filename):
    """
    Save visualization of original image, ground truth mask, and predicted mask.
    Creates a side-by-side comparison image.
    """
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Ground truth mask
    ax2.imshow(image)
    ax2.imshow(gt_mask, alpha=0.5, cmap='Reds')
    ax2.set_title('Ground Truth Mask')
    ax2.axis('off')
    
    # Predicted mask
    ax3.imshow(image)
    ax3.imshow(pred_mask, alpha=0.5, cmap='Blues')
    ax3.set_title('Predicted Mask')
    ax3.axis('off')
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=150)
    plt.close()

def generate_sam_mask(image, mask_generator):
    """
    Use SAM's automatic mask generator to produce segmentation masks for an image.
    Returns a single binary mask focusing on the main foreground object.
    """
    masks = mask_generator.generate(image)
    
    if not masks:
        return np.zeros(image.shape[:2], dtype=bool)
    
    # Sort masks by area (descending) and confidence
    sorted_masks = sorted(masks, key=lambda x: (x['area'], x['predicted_iou']), reverse=True)
    
    # Take the largest mask as the main foreground object
    # This assumes the largest mask with high confidence is the main subject
    main_mask = sorted_masks[0]['segmentation']
    return main_mask

def process_dataset(dataset_name, image_dir, mask_dir, mask_generator, num_samples=50):
    """
    Process a dataset:
      - Randomly select num_samples images.
      - For each image, load the ground truth mask, generate the predicted mask using SAM,
        and compute IoU and Dice metrics.
    Returns the average IoU and Dice across the processed images.
    """
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Create results directory for this dataset
    results_dir = os.path.join("results", dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Handle DAVIS dataset's subdirectory structure
    if dataset_name == "DAVIS":
        # Get all image files from all subdirectories
        print("Collecting DAVIS image files from subdirectories...")
        image_files = []
        for subdir in os.listdir(image_dir):
            subdir_path = os.path.join(image_dir, subdir)
            if os.path.isdir(subdir_path):
                for fname in os.listdir(subdir_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append((subdir, fname))
    else:
        # For other datasets, list image files directly
        print(f"Collecting {dataset_name} image files...")
        image_files = [(None, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) < num_samples:
        print(f"Warning: {dataset_name} has only {len(image_files)} images; using all available images.")
        num_samples = len(image_files)
    else:
        print(f"Found {len(image_files)} images, using {num_samples} random samples")
    
    selected_files = random.sample(image_files, num_samples)
    
    iou_list = []
    dice_list = []
    
    # Create progress bar
    pbar = tqdm(selected_files, desc=f"Processing {dataset_name}", unit="image")
    
    for file_info in pbar:
        subdir, fname = file_info
        if subdir:  # DAVIS dataset
            image_path = os.path.join(image_dir, subdir, fname)
            # For DAVIS, mask files are .png while images can be .jpg
            mask_fname = os.path.splitext(fname)[0] + '.png'
            mask_path = os.path.join(mask_dir, subdir, mask_fname)
            # Use sequence name in output filename
            output_fname = f"{subdir}_{os.path.splitext(fname)[0]}.png"
        else:  # Other datasets
            image_path = os.path.join(image_dir, fname)
            output_fname = os.path.splitext(fname)[0] + '.png'
            if dataset_name == "COCO":
                mask_path = mask_dir  # For COCO, mask_path is the annotations JSON file
            else:
                # For VOC2012, mask files share the same filename as images
                mask_path = os.path.join(mask_dir, fname)
        
        try:
            image, gt_mask = load_image_and_mask(image_path, mask_path, dataset_type=dataset_name)
            
            # Generate predicted foreground mask using SAM
            pred_mask = generate_sam_mask(image, mask_generator)
            
            # Ensure masks have the same shape
            if pred_mask.shape != gt_mask.shape:
                pbar.write(f"Skipping {fname}: mask shape mismatch - pred: {pred_mask.shape}, gt: {gt_mask.shape}")
                continue
            
            # Compute metrics
            iou = compute_iou(pred_mask, gt_mask)
            dice = compute_dice(pred_mask, gt_mask)
            
            # Save visualization
            save_visualization(
                image, gt_mask, pred_mask,
                results_dir,
                output_fname
            )
            
            iou_list.append(iou)
            dice_list.append(dice)
            
            # Update progress bar description with current metrics
            current_iou = np.mean(iou_list)
            current_dice = np.mean(dice_list)
            pbar.set_postfix({
                'IoU': f"{current_iou:.4f}",
                'Dice': f"{current_dice:.4f}"
            })
            
        except Exception as e:
            pbar.write(f"Skipping {fname}: {e}")
            continue
    
    avg_iou = np.mean(iou_list) if iou_list else 0
    avg_dice = np.mean(dice_list) if dice_list else 0
    
    print(f"\n{dataset_name} Results:")
    print(f"  Average IoU  = {avg_iou:.4f}")
    print(f"  Average Dice = {avg_dice:.4f}")
    print(f"  Processed {len(iou_list)} images successfully")
    print(f"  Results saved in: {os.path.abspath(results_dir)}")
    return avg_iou, avg_dice


def main():
    random.seed(42)
    
    # ---------------------------
    # Setup SAM Model
    # ---------------------------
    print("\nInitializing SAM model...")
    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"  # Options: "vit_h", "vit_l", "vit_b"
    
    if not os.path.exists(sam_checkpoint):
        print(f"SAM checkpoint not found at {sam_checkpoint}. Please run download.py first.")
        return
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # Configure mask generator for foreground/background segmentation
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,  # Dense sampling for better coverage
        pred_iou_thresh=0.90,  # Higher threshold for more confident predictions
        stability_score_thresh=0.95,  # Higher threshold for more stable masks
        crop_n_layers=0,  # No cropping to maintain global context
        min_mask_region_area=1000,  # Larger minimum area to focus on main objects
        output_mode="binary_mask",  # Binary mask output
    )
    
    # Create main results directory
    os.makedirs("results", exist_ok=True)
    print("\nResults will be saved in:", os.path.abspath("results"))
    
    # ---------------------------
    # Define dataset paths
    # ---------------------------
    datasets = {
        "DAVIS": {
            "image_dir": "DAVIS/JPEGImages/480p",
            "mask_dir": "DAVIS/Annotations/480p"
        },
        "COCO": {
            "image_dir": "COCO/val2017",
            "mask_dir": "COCO/annotations/instances_val2017.json"
        },
        "VOC2012": {
            "image_dir": "VOC2012/VOCdevkit/VOC2012/JPEGImages",
            "mask_dir": "VOC2012/VOCdevkit/VOC2012/SegmentationObject"
        }
    }
    
    # Verify dataset paths exist
    print("\nVerifying dataset paths...")
    for name, paths in datasets.items():
        if not os.path.exists(paths["image_dir"]) or not os.path.exists(paths["mask_dir"]):
            print(f"Dataset {name} not found at specified paths. Please run download.py first.")
            return
        print(f"✓ {name} dataset found")
    
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
    
    print("\nFinal Results Summary:")
    print("=" * 40)
    for dataset in results_iou:
        print(f"{dataset:8s}: IoU = {results_iou[dataset]:.4f}, Dice = {results_dice[dataset]:.4f}")
    print("=" * 40)
    
    # ---------------------------
    # Plot the metrics as bar graphs
    # ---------------------------
    print("\nGenerating plots...")
    # Save plots in results directory
    plt.figure()
    plot_metrics(results_iou, "Intersection over Union (IoU)")
    plt.savefig("results/iou_comparison.png")
    plt.close()
    
    plt.figure()
    plot_metrics(results_dice, "Dice Coefficient")
    plt.savefig("results/dice_comparison.png")
    plt.close()
    
    print("\nAll visualizations have been saved in:", os.path.abspath("results"))

if __name__ == "__main__":
    main()
