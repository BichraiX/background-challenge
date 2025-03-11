#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys
from utils import OUTPUT_DIR

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on ImageNet and then fine-tune with orthogonal adapters')
    
    # General parameters
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='Output directory for models and logs')
    parser.add_argument('--skip-base-training', action='store_true', help='Skip training the base model')
    parser.add_argument('--base-model-path', type=str, default=None, 
                        help='Path to a pretrained base model (used if skip-base-training is set)')
    
    # Base model training parameters
    parser.add_argument('--base-epochs', type=int, default=40, help='Number of epochs for base model training')
    parser.add_argument('--base-batch-size', type=int, default=128, help='Batch size for base model training')
    parser.add_argument('--base-lr', type=float, default=0.001, help='Learning rate for base model training')
    parser.add_argument('--train-all', action='store_true', 
                        help='Train all layers of the base model (vs just the classifier)')
    
    # Orthogonal fine-tuning parameters
    parser.add_argument('--oft-epochs', type=int, default=20, help='Number of epochs for orthogonal fine-tuning')
    parser.add_argument('--oft-batch-size', type=int, default=128, help='Batch size for orthogonal fine-tuning')
    parser.add_argument('--oft-lr', type=float, default=1e-4, help='Learning rate for orthogonal fine-tuning')
    parser.add_argument('--oft-rank', type=int, default=16, help='Rank for OFT adapters')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Path to save the base model
    base_model_path = os.path.join(args.output_dir, 'best_base_model.pth')
    if args.base_model_path:
        base_model_path = args.base_model_path
    
    # Step 1: Train the base model on original ImageNet
    if not args.skip_base_training:
        print("\n" + "="*80)
        print("STAGE 1: TRAINING BASE MODEL ON ORIGINAL IMAGENET")
        print("="*80 + "\n")
        
        cmd = [
            sys.executable, "train_imagenet.py",
            "--batch-size", str(args.base_batch_size),
            "--epochs", str(args.base_epochs),
            "--lr", str(args.base_lr),
            "--output-dir", args.output_dir
        ]
        
        if args.train_all:
            cmd.append("--train-all")
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        print("\n" + "="*80)
        print("SKIPPING BASE MODEL TRAINING")
        print(f"Using base model from: {base_model_path}")
        print("="*80 + "\n")
    
    # Step 2: Fine-tune with orthogonal adapters on mixed_rand dataset
    print("\n" + "="*80)
    print("STAGE 2: ORTHOGONAL FINE-TUNING ON MIXED_RAND DATASET")
    print("="*80 + "\n")
    
    cmd = [
        sys.executable, "orthogonal_finetune.py",
        "--base-model", base_model_path,
        "--batch-size", str(args.oft_batch_size),
        "--epochs", str(args.oft_epochs),
        "--lr", str(args.oft_lr),
        "--oft-rank", str(args.oft_rank),
        "--output-dir", args.output_dir
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETED")
    print(f"Results saved to: {args.output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main() 