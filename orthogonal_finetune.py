import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
from tqdm import tqdm
import argparse
from peft import OFTConfig, OFTModel, TaskType

from utils import (
    PROCESSED_DATA_PATH, OUTPUT_DIR, 
    MixedRandDataset, 
    train, validate, print_trainable_parameters,
    plot_training_curves
)

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune with orthogonal adapters on mixed_rand dataset')
    parser.add_argument('--base-model', type=str, default="./results/best_base_model.pth", help='Path to base model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--oft-rank', type=int, default=16, help='Rank for OFT adapters')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    return parser.parse_args()

def create_oft_model(base_model_path=None, oft_rank=16):
    """Create an OFT model from a base model or pretrained weights"""
    if base_model_path and os.path.isfile(base_model_path):
        print(f"Loading base model from {base_model_path}")
        # Load the saved model
        checkpoint = torch.load(base_model_path, map_location='cpu')
        
        # Create a fresh model
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace the classifier head (in case it's different)
        num_classes = 9  # Our 9 classes
        fc_in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(fc_in_features, num_classes)
        
        # Load saved weights
        base_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded base model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['val_acc']:.2f}%")
    else:
        print("Starting with pretrained ImageNet weights")
        # Create a model with pretrained weights
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace classifier for our 9 classes
        num_classes = 9
        fc_in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(fc_in_features, num_classes)
    
    # Define specific layer names to target in ResNet50
    # We'll target the convolutional layers in the last two blocks (layer3 and layer4)
    target_modules = []
    
    # # Add conv layers from layer3 (6 bottleneck blocks, each with 3 conv layers)
    # for i in range(6):  # ResNet50 has 6 bottleneck blocks in layer3
    #     target_modules.extend([
    #         f"layer3.{i}.conv1",
    #         f"layer3.{i}.conv2",
    #         f"layer3.{i}.conv3"
    #     ])
    
    # Add conv layers from layer4 (3 bottleneck blocks, each with 3 conv layers)
    for i in range(3):  # ResNet50 has 3 bottleneck blocks in layer4
        target_modules.extend([
            f"layer4.{i}.conv1",
            f"layer4.{i}.conv2",
            f"layer4.{i}.conv3"
        ])
    
    # Add downsample convolutional layers
    target_modules.extend([
        "layer3.0.downsample.0",  # First block in layer3 has a downsample conv
        "layer4.0.downsample.0"   # First block in layer4 has a downsample conv
    ])
    
    print(f"Targeting {len(target_modules)} convolutional layers for orthogonal fine-tuning")
    
    # Configure OFT (Orthogonal Fine-Tuning)
    oft_config = OFTConfig(
        r=oft_rank,  # Rank for OFT
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,  # Target specific Conv2d layers
        module_dropout=0.1,
        init_weights=True,
        coft=False,  # Not using constrained OFT
        eps=6e-5,
        block_share=False
    )
    
    print("Creating OFT model...")
    # Create PEFT model with OFT configuration
    model = OFTModel(base_model, oft_config, adapter_name="default")
    
    return model

def main():
    args = parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    print("Loading datasets...")
    train_dataset = MixedRandDataset(PROCESSED_DATA_PATH, transform=train_transform, split='train')
    val_dataset = MixedRandDataset(PROCESSED_DATA_PATH, transform=val_transform, split='val')
    
    # Adjust batch size based on device
    batch_size = args.batch_size
    if device.type == 'cpu':
        batch_size = min(batch_size, 16)  # Smaller batch size for CPU
        print(f"Using reduced batch size {batch_size} for CPU")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == 'cuda'))
    
    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images")
    
    # Create model with orthogonal adapters
    model = create_oft_model(args.base_model, args.oft_rank)
    model = model.to(device)
    
    # Print trainable parameters
    print_trainable_parameters(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # Training loop
    print(f"Starting orthogonal fine-tuning for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    # Arrays to store metrics for plotting
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, os.path.join(output_dir, 'best_orthogonal_model.pth'))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save final model
    torch.save(model, os.path.join(output_dir, 'final_orthogonal_model.pth'))
    print("Saved final model")
    
    # Plot training curves
    plot_training_curves(
        train_losses, train_accs, val_losses, val_accs, 
        os.path.join(output_dir, 'orthogonal_training_curves.png')
    )
    
    print("Orthogonal fine-tuning completed. Results saved to", output_dir)

if __name__ == "__main__":
    main() 