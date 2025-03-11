import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from tqdm import tqdm
import argparse
import re

# Import peft for OFT model support
from peft import OFTModel, OFTConfig, TaskType, get_peft_model
# Import add_safe_globals for secure loading
from torch.serialization import add_safe_globals

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define WordNet ID to class name mapping - KEEP THE EXACT SAME ORDER as in training
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

# Create class index mapping in the SAME ORDER as training 
# (no sorting, just use the order in WORDNET_TO_CLASS)
CLASS_NAMES = list(WORDNET_TO_CLASS.values())
CLASS_NAMES = [name.lower() for name in CLASS_NAMES]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

print(f"Using class mapping: {CLASS_TO_IDX}")

def setup_oft_model():
    """Create a ResNet50 model with OFT adaptation layers for evaluation"""
    # Load the pretrained ResNet50
    model = models.resnet50(pretrained=True)
    
    # Modify the last layer for 9 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)
    
    # Specify the target modules for OFT - same as in training
    target_modules = ["layer4.0.conv2", "layer4.1.conv2", "layer4.2.conv2"]
    
    # OFT configuration
    oft_config = OFTConfig(
        r=4,
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION
    )
    
    # Apply PEFT to create the adapted model
    model = get_peft_model(model, oft_config)
    
    return model

def load_model(model_path):
    """Load the saved orthogonal model with appropriate settings"""
    try:
        print(f"Loading checkpoint from {model_path}...")
        
        # Add OFTModel to safe globals list to allow deserializing it
        add_safe_globals([OFTModel])
        
        # Load the entire model directly
        model = torch.load(model_path, map_location=device, weights_only=False)
        print("Model loaded successfully!")
        
        # Move to the appropriate device
        model = model.to(device)
        
        # Set to evaluation mode
        model.eval()
        
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Normalize the images as done during training - EXACTLY the same values
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class AttackedDataset(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        self.class_errors = set()  # Track class names that cause errors

        print(f"Loading dataset from {root}")
        for class_folder in os.listdir(root):
            class_path = os.path.join(root, class_folder)
            if not os.path.isdir(class_path):
                continue  

            # Extract class name from folder name (format: "XX_classname")
            class_name = class_folder.split("_", 1)[1].lower()
            
            # Map specific class names to expected format 
            if class_name == "wheeled vehicle":
                class_name = "vehicle"
            if class_name == "musical instrument":
                class_name = "instrument"
                
            if class_name not in class_to_idx:
                if class_name not in self.class_errors:
                    print(f"⚠️  Ignored: {class_folder} (not in CLASS_TO_IDX: {CLASS_TO_IDX})")
                    self.class_errors.add(class_name)
                continue  

            label = class_to_idx[class_name]  

            valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPEG"]
            for ext in valid_extensions:
                for img_path in glob.glob(os.path.join(class_path, f"*{ext}")):
                    self.samples.append((img_path, label))
        
        print(f"Loaded {len(self.samples)} samples from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")  
        
        if self.transform:
            img = self.transform(img)

        return img, label

def evaluate_model(model, data_path):
    model.eval()
    
    # List of attack categories to evaluate
    attacked_categories = [
        "original",
        "mixed_next",
        "mixed_rand",
        "mixed_same",
        "no_fg",
        "only_bg_b",
        "only_bg_t",
        "only_fg"
    ]
    
    # Store accuracy by category
    category_accuracies = {}
    
    # Global counting
    total_correct = 0
    total_samples = 0
    
    print("\nEvaluating model on all attack categories...")
    
    # Loop through each attack category
    for cat in attacked_categories:
        cat_val_path = os.path.join(data_path, cat, "val")
        
        if os.path.isdir(cat_val_path):
            dataset = AttackedDataset(root=cat_val_path, class_to_idx=CLASS_TO_IDX, transform=transform)
            
            # Skip if no samples were found
            if len(dataset) == 0:
                print(f"⚠️ No samples found in {cat_val_path}. Skipping.")
                continue
                
            data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            correct = 0
            total = 0
            
            # Evaluation loop
            with torch.no_grad():
                for images, labels in tqdm(data_loader, desc=f"Evaluating {cat}"):
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)

                    # Debug first batch
                    if correct == 0 and total == 0:
                        print(f"DEBUG - First batch predictions: {preds[:5]}")
                        print(f"DEBUG - First batch labels: {labels[:5]}")
                    
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            
            # Calculate accuracy for this category
            accuracy = correct / total if total > 0 else 0.0
            category_accuracies[cat] = accuracy
            
            # Update global counts
            total_correct += correct
            total_samples += total
            
            print(f"✅ Accuracy on {cat}: {accuracy:.4f} ({correct}/{total})")
        else:
            print(f"⚠️ Directory {cat_val_path} not found. Skipping.")
    
    # Calculate global accuracy
    global_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Final summary
    print("\n📊 **Final Results:**")
    for cat, acc in category_accuracies.items():
        print(f" - {cat}: {acc:.4f}")
    
    print(f"\n🎯 **Global accuracy across all attack categories: {global_accuracy:.4f}**")
    
    return category_accuracies, global_accuracy

def main():
    parser = argparse.ArgumentParser(description="Benchmark a fine-tuned model on background challenge dataset")
    parser.add_argument("--model_path", type=str, default="./results/final_orthogonal_model.pth", 
                        help="Path to the model checkpoint")
    parser.add_argument("--data_path", type=str, default="/Data/amine.chraibi/bg_challenge",
                        help="Path to the background challenge dataset")
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model_path)
    
    # Evaluate the model
    evaluate_model(model, args.data_path)

if __name__ == "__main__":
    main() 