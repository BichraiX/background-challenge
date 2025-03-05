import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import json
from pycocotools import mask as mask_utils

# Add paths
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import sam_model_registry, SamPredictor

# Utils
from utils import compute_iou, compute_dice, plot_metrics

def load_image(image_path):
    """Load and transform image for GroundingDINO."""
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    """Load GroundingDINO model."""
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    """Get bounding boxes from GroundingDINO."""
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    
    # Filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]
    
    return boxes_filt

def load_image_and_mask(image_path, mask_path, dataset_type=None):
    """Load an RGB image and its ground truth segmentation mask."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Handle different mask formats based on dataset type
    if dataset_type == "COCO":
        try:
            with open(mask_path, 'r') as f:
                annotations = json.load(f)
            img_id = int(os.path.splitext(os.path.basename(image_path))[0])
            h, w = image.shape[:2]
            binary_mask = np.zeros((h, w), dtype=bool)
            found_annotation = False
            for ann in annotations['annotations']:
                if ann['image_id'] == img_id:
                    found_annotation = True
                    if 'segmentation' in ann:
                        if isinstance(ann['segmentation'], dict):  # RLE format
                            current_mask = mask_utils.decode(ann['segmentation'])
                        else:  # Polygon format
                            rle = mask_utils.frPyObjects(ann['segmentation'], h, w)
                            if isinstance(rle, list):
                                rle = mask_utils.merge(rle)
                            current_mask = mask_utils.decode(rle)
                        if current_mask.ndim == 3:
                            current_mask = np.any(current_mask, axis=2)
                        binary_mask = np.logical_or(binary_mask, current_mask)
            if not found_annotation:
                raise ValueError(f"No annotation found for image {img_id}")
            return image, binary_mask
        except Exception as e:
            print(f"Error processing COCO annotation for {image_path}: {str(e)}")
            raise
    
    elif dataset_type == "VOC2012":
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

def get_class_prompt(dataset_type, image_path):
    """Get the class name from dataset annotations."""
    if dataset_type == "COCO":
        ann_file = os.path.join("COCO", "annotations", "instances_val2017.json")
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        img_id = int(os.path.splitext(os.path.basename(image_path))[0])
        for ann in coco_data['annotations']:
            if ann['image_id'] == img_id:
                cat_id = ann['category_id']
                for cat in coco_data['categories']:
                    if cat['id'] == cat_id:
                        return cat['name']
        return None
    
    elif dataset_type == "VOC2012":
        class_name = os.path.basename(image_path).split('_')[0]
        return class_name
    
    elif dataset_type == "DAVIS":
        sequence = os.path.basename(os.path.dirname(image_path))
        return sequence
    
    return None

def show_mask(mask, ax, random_color=False):
    """Display a mask on a given matplotlib axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def evaluate_sam(sam_checkpoint="checkpoints/sam_vit_h_4b8939.pth",
                grounding_dino_config="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                grounding_dino_checkpoint="groundingdino_swint_ogc.pth",
                device="cpu"):
    """Evaluate SAM on multiple datasets."""
    print("Initializing models...")
    
    # Initialize SAM
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # Initialize GroundingDINO
    groundingdino = load_model(grounding_dino_config, grounding_dino_checkpoint, device)
    
    # Results dictionary
    results = {
        "COCO": {"iou": [], "dice": []},
        "DAVIS": {"iou": [], "dice": []},
        "VOC2012": {"iou": [], "dice": []}
    }
    
    # Process each dataset
    datasets = {
        "COCO": "COCO/val2017",
        "DAVIS": "DAVIS/JPEGImages/480p",
        "VOC2012": "VOC2012/JPEGImages"
    }
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\nProcessing {dataset_name} dataset...")
        
        # Get list of images
        images = []
        for root, _, files in os.walk(dataset_path):
            for f in files:
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    images.append(os.path.join(root, f))
        
        if not images:
            print(f"No images found in {dataset_path}")
            continue
        
        # Randomly select 50 images
        selected_images = random.sample(images, min(50, len(images)))
        
        # Process each image
        for img_path in tqdm(selected_images):
            try:
                # Get corresponding mask path
                if dataset_name == "COCO":
                    mask_path = os.path.join("COCO", "annotations", "instances_val2017.json")
                elif dataset_name == "DAVIS":
                    mask_path = img_path.replace("JPEGImages", "Annotations").replace(".jpg", ".png")
                else:  # VOC2012
                    mask_path = img_path.replace("JPEGImages", "SegmentationClass").replace(".jpg", ".png")
                
                # Load image and ground truth
                image, gt_mask = load_image_and_mask(img_path, mask_path, dataset_name)
                
                # Get class name and prepare text prompt
                class_name = get_class_prompt(dataset_name, img_path)
                if class_name is None:
                    continue
                
                # Load and transform image for GroundingDINO
                _, image_tensor = load_image(img_path)
                
                # Get bounding boxes from GroundingDINO
                boxes_filt = get_grounding_output(
                    model=groundingdino,
                    image=image_tensor,
                    caption=f"a {class_name}",
                    box_threshold=0.35,
                    text_threshold=0.25,
                    device=device
                )
                
                # Set image in predictor
                predictor.set_image(image)
                
                # Convert boxes to the format expected by SAM
                H, W = image.shape[:2]
                boxes_xyxy = boxes_filt * torch.tensor([W, H, W, H])
                boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
                boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]
                
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, image.shape[:2])
                
                # Generate masks
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                
                if len(masks) == 0:
                    print(f"No masks generated for {img_path}")
                    continue
                
                # Combine all masks
                pred_mask = torch.any(masks, dim=0).cpu().numpy()
                
                # Calculate metrics
                iou = compute_iou(pred_mask, gt_mask)
                dice = compute_dice(pred_mask, gt_mask)
                
                results[dataset_name]["iou"].append(iou)
                results[dataset_name]["dice"].append(dice)
                
                # Save visualization
                save_dir = os.path.join("results", dataset_name)
                os.makedirs(save_dir, exist_ok=True)
                
                # Create visualization
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                ax1.imshow(image)
                ax1.set_title(f"Original Image\nClass: {class_name}")
                ax1.axis('off')
                
                ax2.imshow(gt_mask, cmap='gray')
                ax2.set_title("Ground Truth")
                ax2.axis('off')
                
                ax3.imshow(image)
                show_mask(pred_mask, ax3)
                ax3.set_title(f"GroundingDINO + SAM\nIoU: {iou:.2f}, Dice: {dice:.2f}")
                ax3.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{os.path.basename(img_path)}_result.png"))
                plt.close()
                
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
                continue
    
    # Calculate and plot metrics
    avg_metrics = {"iou": {}, "dice": {}}
    
    for dataset_name in results:
        if results[dataset_name]["iou"]:
            avg_metrics["iou"][dataset_name] = np.mean(results[dataset_name]["iou"])
            avg_metrics["dice"][dataset_name] = np.mean(results[dataset_name]["dice"])
    
    # Calculate overall averages
    all_ious = []
    all_dices = []
    for dataset_name in results:
        all_ious.extend(results[dataset_name]["iou"])
        all_dices.extend(results[dataset_name]["dice"])
    
    if all_ious:
        avg_metrics["iou"]["Overall"] = np.mean(all_ious)
        avg_metrics["dice"]["Overall"] = np.mean(all_dices)
        
        # Plot results
        plot_metrics(avg_metrics["iou"], "IoU")
        plt.savefig("results/iou_comparison.png")
        plt.close()
        
        plot_metrics(avg_metrics["dice"], "Dice Score")
        plt.savefig("results/dice_comparison.png")
        plt.close()
        
        # Print results
        print("\nEvaluation Results:")
        for dataset_name in results:
            if dataset_name in avg_metrics["iou"]:
                print(f"\n{dataset_name}:")
                print(f"Average IoU: {avg_metrics['iou'][dataset_name]:.3f}")
                print(f"Average Dice: {avg_metrics['dice'][dataset_name]:.3f}")
        
        print("\nOverall:")
        print(f"Average IoU: {avg_metrics['iou']['Overall']:.3f}")
        print(f"Average Dice: {avg_metrics['dice']['Overall']:.3f}")
    else:
        print("\nNo results were generated. Please check the dataset paths and annotations.")

if __name__ == "__main__":
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run evaluation
    evaluate_sam(device="cpu")  # Use CPU by default
