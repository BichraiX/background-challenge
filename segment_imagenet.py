import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add paths
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import sam_model_registry, SamPredictor

# Class mapping
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
    checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cuda"):
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
    
    return boxes_filt.to(device)  # Return boxes on the correct device

def show_mask(mask, ax, random_color=False):
    """Display a mask on a given matplotlib axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def save_masked_image(image, mask, output_path, is_foreground=True):
    """Save image with mask applied, with transparent background."""
    # Ensure mask does not have extra dimensions
    mask = np.squeeze(mask)  # This will remove dimensions of size 1
    
    # Convert image to RGBA
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGBA")
    image_array = np.array(image)
    
    # Create transparent background (alpha=0)
    transparent = np.array([0, 0, 0, 0])
    
    if is_foreground:
        # For foreground: keep masked area, make rest transparent
        mask_expanded = np.expand_dims(mask, axis=2)  # Now mask_expanded is (H, W, 1)
        masked_image = np.where(mask_expanded, image_array, transparent)
    else:
        # For background: keep unmasked area, make rest transparent
        mask_expanded = np.expand_dims(~mask, axis=2)
        masked_image = np.where(mask_expanded, image_array, transparent)
    
    # Save the result
    result_image = Image.fromarray(masked_image.astype(np.uint8))
    result_image.save(output_path, "PNG")


def process_imagenet():
    """Process ImageNet images using GroundingDINO + SAM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize models
    print("Initializing models...")
    
    # Initialize SAM
    sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # Initialize GroundingDINO
    groundingdino = load_model(
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "groundingdino_swint_ogc.pth",
        device
    )
    groundingdino.to(device)
    
    # Create output directory
    os.makedirs("/Data/amine.chraibi/dataset", exist_ok=True)
    
    # Process each class
    for wordnet_id, class_name in WORDNET_TO_CLASS.items():
        print(f"\nProcessing class: {class_name}")
        
        
        # Create class directory
        class_dir = os.path.join("/Data/amine.chraibi/dataset", class_name.lower())
        os.makedirs(class_dir, exist_ok=True)
        
        # Get list of images for this class
        class_path = os.path.join("/Data/amine.chraibi/ImageNet/ImageNet_9_classes", wordnet_id)
        if not os.path.exists(class_path):
            print(f"Warning: Directory not found for {class_name} ({wordnet_id})")
            continue
        
        images = [f for f in os.listdir(class_path) if f.endswith(('.JPEG', '.jpg', '.png'))]
        
        # Process each image
        for img_name in tqdm(images):
            try:
                img_path = os.path.join(class_path, img_name)
                
                # Load and transform image for GroundingDINO
                image_pil, image_tensor = load_image(img_path)
                image_tensor = image_tensor.to(device)
                
                # Get bounding boxes from GroundingDINO
                boxes_filt = get_grounding_output(
                    model=groundingdino,
                    image=image_tensor,
                    caption=f"a {class_name}",
                    box_threshold=0.35,
                    text_threshold=0.25,
                    device=device
                )
                
                if len(boxes_filt) == 0:
                    print(f"No boxes found for {img_path}")
                    continue
                
                # Load image for SAM
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                predictor.set_image(image)
                
                # Convert boxes to the format expected by SAM
                H, W = image.shape[:2]
                boxes_xyxy = boxes_filt * torch.tensor([W, H, W, H], device=device)
                boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
                boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]
                
                # Transform boxes and ensure they're on the correct device
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy.cpu(), image.shape[:2])
                transformed_boxes = transformed_boxes.to(device)
                
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
                combined_mask = torch.any(masks, dim=0).cpu().numpy()
                
                # Save foreground and background images
                base_name = os.path.splitext(img_name)[0]
                save_masked_image(
                    image_pil,
                    combined_mask,
                    os.path.join(class_dir, f"{base_name}_fg.png"),
                    is_foreground=True
                )
                save_masked_image(
                    image_pil,
                    combined_mask,
                    os.path.join(class_dir, f"{base_name}_bg.png"),
                    is_foreground=False
                )
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

if __name__ == "__main__":
    process_imagenet() 