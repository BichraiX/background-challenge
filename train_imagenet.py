import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
from tqdm import tqdm
import argparse

from utils import (
    DATA_PATH, OUTPUT_DIR, 
    ImageNetSubsetDataset, 
    train, validate, print_trainable_parameters,
    plot_training_curves
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet on ImageNet subset')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--train-all', action='store_true', help='Train all layers (vs just the classifier)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    return parser.parse_args()

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
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    print("Loading datasets...")
    train_dataset = ImageNetSubsetDataset(DATA_PATH, transform=train_transform, split='train')
    val_dataset = ImageNetSubsetDataset(DATA_PATH, transform=val_transform, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images")
    
    # Create model
    print("Creating model...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Replace the final fully connected layer
    num_classes = 9  # 9 classes in our subset
    fc_in_features = model.fc.in_features
    model.fc = nn.Linear(fc_in_features, num_classes)
    
    # Set training mode
    if args.train_all:
        print("Training all parameters (full fine-tuning)")
        # All parameters will be trained
    else:
        print("Training only the classifier layer")
        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the final layer
        for param in model.fc.parameters():
            param.requires_grad = True
    
    model = model.to(device)
    print_trainable_parameters(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Different learning rates for different parts of the model
    if args.train_all:
        # If training everything, use a lower learning rate for the pre-trained backbone
        optimizer = optim.Adam([
            {'params': [p for n, p in model.named_parameters() if 'fc' not in n], 'lr': args.lr * 0.1},
            {'params': model.fc.parameters(), 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    else:
        # If only training the classifier, just use the provided learning rate
        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(output_dir, 'best_base_model.pth'))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }, os.path.join(output_dir, 'final_base_model.pth'))
    print("Saved final model")
    
    # Plot training curves
    plot_training_curves(
        train_losses, train_accs, val_losses, val_accs, 
        os.path.join(output_dir, 'base_training_curves.png')
    )
    
    print("Base model training completed. Results saved to", output_dir)

if __name__ == "__main__":
    main() 