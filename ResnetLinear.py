import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

class NineClassDataset(Dataset):
    def __init__(self, root, transform=None, valid_subfolders=None):
        """
        If valid_subfolders is None, default to ['mixed_rand', 'original'].
        Otherwise, use the user-specified list (e.g. ['original'] for validation).
        """
        self.root = root
        self.transform = transform
        
        if valid_subfolders is None:
            valid_subfolders = ["mixed_rand", "original"]
        
        # Suppose you have 9 class folders: class0, class1, ..., class8
        class_names = sorted(
            d for d in os.listdir(root) 
            if os.path.isdir(os.path.join(root, d))
        )
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
        
        self.samples = []
        
        # Walk each top-level class folder
        for class_name in class_names:
            class_dir = os.path.join(root, class_name)
            label = self.class_to_idx[class_name]
            
            # For each subfolder in our "valid_subfolders" list
            for sub in valid_subfolders:
                sub_dir = os.path.join(class_dir, sub)
                if not os.path.isdir(sub_dir):
                    continue  # skip if doesn't exist
                
                for fname in os.listdir(sub_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(sub_dir, fname)
                        self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def validate(model, dataloader, device):
    """
    Runs model in eval mode and returns the classification accuracy
    on the given dataloader.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return (correct / total) if total > 0 else 0


# ---------------------------------
# 1) Create Datasets & DataLoaders
# ---------------------------------
root_dir = "output"  # Adjust to your folder

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Training set: uses both 'mixed_rand' and 'original'
train_dataset = NineClassDataset(root=root_dir, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

# Validation set: uses only 'original'
val_dataset = NineClassDataset(
    root=root_dir,
    transform=val_transforms,
    valid_subfolders=["original"]
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)


# ---------------------------------
# 2) Load ResNet-50 Pretrained Weights & Modify
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize a ResNet-50 without TorchVision's built-in weights
resnet = models.resnet50(pretrained=False)

# 2. Load the local weights from resnet50-19c8e357.pth
#    NOTE: If needed, you can add map_location to ensure compatibility on CPU vs GPU
state_dict = torch.load("resnet50-19c8e357.pth", map_location="cpu", weights_only= False)
resnet.load_state_dict(state_dict)

# 3. Change the final FC layer to match our 9 classes
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 9)

resnet.to(device)


# ---------------------------------
# 3) Define Loss and Optimizer
# ---------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=1e-4)
num_epochs = 200


# ---------------------------------
# 4) Training Loop + Validation
# ---------------------------------
for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix(loss=loss.item())
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {epoch_loss:.4f}")

    # --- Validation (only on 'original') ---
    val_accuracy = validate(resnet, val_loader, device)
    print(f"Validation Accuracy (only 'original'): {val_accuracy:.4f}\n")

    # --- Save the model if desired ---
    if (epoch + 1) % 20 == 0:
        checkpoint_filename = f"resnet50_epoch_{epoch+1}.pth"
        torch.save(resnet.state_dict(), checkpoint_filename)
        print(f"Model weights saved to {checkpoint_filename}")
