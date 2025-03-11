# Background Challenge: Testing and Improving Model Robustness

This repository contains code for the Background Challenge, which focuses on testing and improving the robustness of computer vision models against background changes. Even state-of-the-art models can be vulnerable to background variations, with the official pre-trained PyTorch ResNet-50 dropping to only 22% accuracy when evaluated against adversarial backgrounds on ImageNet-9 (for comparison, a model that always predicts "dog" has an accuracy of 11%).

Our goal is to understand and improve how background-robust models can be, specifically by assessing models based on their accuracy on images containing foregrounds superimposed on adversarially chosen backgrounds from the test set.

## Challenge Description

Deep computer vision models rely on both foreground objects and image backgrounds. Even when the correct foreground object is present, such models often make incorrect predictions when the image background is changed, and they are especially vulnerable to adversarially chosen backgrounds.

This challenge assesses models by their accuracy on images containing foregrounds superimposed on backgrounds which are adversarially chosen from the test set. The goal is to benchmark progress on background-robustness, which is important for determining models' out-of-distribution performance.

## Project Structure

### Core Components

- **`orthogonal_finetune.py`**: Implements orthogonal fine-tuning (OFT) on a pre-trained model using the PEFT library. The OFT approach helps maintain model performance on the original task while adapting to new domains.

- **`benchmark.py`**: Evaluates model performance on various background attack categories using the OFTModel architecture.

- **`utils.py`**: Contains utility functions for metrics calculation (IoU, Dice), dataset classes, training and validation loops, and visualization tools.

### Parameter-Efficient Fine-Tuning Notebooks

#### LoRA Approach

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that significantly reduces the number of trainable parameters. This approach helps achieve background robustness while minimizing computational overhead and risk of catastrophic forgetting.

- **`trainLoRA.ipynb`**: This notebook demonstrates how to fine-tune a ResNet50 model using LoRA. Key features include:
  - Loading a pre-trained ResNet50 model
  - Freezing most model parameters to maintain general knowledge
  - Applying LoRA adapters to specific layers (primarily in layer4 of ResNet50)
  - Setting up LoRA configuration with rank=4 for efficient fine-tuning
  - Training the model on the mixed background dataset

- **`testLoRA.ipynb`**: This notebook provides a comprehensive evaluation of LoRA-fine-tuned models. It includes:
  - Loading a LoRA-fine-tuned model checkpoint
  - Setting up the class mapping consistent with the training dataset
  - Evaluating the model on 8 different attack categories:
    * `original`: Original unmodified images
    * `mixed_next`: Foreground with next-class background
    * `mixed_rand`: Foreground with random-class background
    * `mixed_same`: Foreground with same-class background
    * `no_fg`: Images with foreground removed
    * `only_bg_b`/`only_bg_t`: Only background (bottom/top)
    * `only_fg`: Only foreground objects
  - Calculating and reporting per-category and global accuracy metrics

  Results from the notebook show that the LoRA-fine-tuned model achieves approximately 69% accuracy across all attack categories, with particularly strong performance on the `original` (94%), as well as a significant improvement over the standard ResNet50's 22% accuracy on adversarial backgrounds.

### Data Processing

- **`dataset_processor.py`**: Creates various foreground/background combinations for robustness testing:
  - `original`: Original unmodified images
  - `only_fg`: Only foreground objects
  - `only_bg_b`/`only_bg_t`: Only background (bottom/top)
  - `no_fg`: Images with foreground removed
  - `mixed_same`: Foreground with same-class background
  - `mixed_rand`: Foreground with random-class background
  - `mixed_next`: Foreground with next-class background

- **`segment_imagenet.py`**: Uses GroundingDINO and SAM to segment ImageNet images into foreground and background components.

### Training Scripts

- **`train_imagenet.py`**: Trains a base model on the ImageNet subset.

- **`train_then_finetune.py`**: Orchestrates the entire training pipeline - first training a base model on ImageNet, then fine-tuning it with orthogonal adapters.

- **`ResnetLinear.py`**: Contains the dataset class for the nine-class dataset and validation functions.

### Evaluation Tools

- **`evaluate_sam.py`**: Evaluates the Segment Anything Model (SAM) on various datasets like COCO, DAVIS, and VOC2012.

- **`Unet.py`**: Implementation of a UNet segmentation model with visualization functions.

### Utilities

- **`download.py`**: Downloads required datasets (COCO, PASCAL VOC, DAVIS) and model checkpoints (SAM, GroundingDINO).

## Getting Started

### Prerequisites

Before running the code, you need to clone two external repositories:

```bash
git clone https://github.com/facebookresearch/segment-anything
git clone https://github.com/IDEA-Research/GroundingDINO
```

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Data Preparation

1. Download the necessary datasets and model checkpoints:

```bash
python download.py
```

2. Process the dataset to create various foreground/background combinations:

```bash
python dataset_processor.py
```

### Training Options

#### Standard Training

1. Train a base model on ImageNet:

```bash
python train_imagenet.py
```

2. Fine-tune with orthogonal adapters:

```bash
python orthogonal_finetune.py --base-model ./results/best_base_model.pth
```

Or run the entire pipeline:

```bash
python train_then_finetune.py
```

#### LoRA Fine-Tuning

For parameter-efficient fine-tuning using LoRA:

1. Open and run `trainLoRA.ipynb` to fine-tune a pre-trained ResNet50 with LoRA adapters:
   - The notebook applies LoRA to specific convolutional layers (layer4.0.conv2, layer4.1.conv2, layer4.2.conv2)
   - It uses a small rank (r=4) to maintain efficiency
   - Only LoRA parameters and the final classification layer are trainable

2. Use `testLoRA.ipynb` to evaluate your LoRA-fine-tuned model:
   - Load a checkpoint with `torch.load('model_final.pth')`
   - Run evaluations on all attack categories
   - View per-category and aggregate accuracy metrics

### Evaluation

Evaluate model performance on the processed datasets:

```bash
python benchmark.py --model_path model_final.pth
```

For simplified evaluation without OFT components:

```bash
python benchmark_simple.py --model_path model_final.pth
```


## Requirements

```
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.2
Pillow>=8.0.0
matplotlib>=3.3.3
tqdm>=4.56.0
albumentations>=1.0.0
opencv-python>=4.5.1
segmentation-models-pytorch>=0.2.0
peft>=0.2.0
requests>=2.25.1
scikit-image>=0.18.1
pycocotools>=2.0.2
jupyter>=1.0.0
ipykernel>=6.0.0
```

## External Dependencies

This project relies on two external repositories:

1. **Segment Anything Model (SAM)**: [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
2. **GroundingDINO**: [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

These repositories must be cloned and properly set up for the segmentation code to run correctly.

## License

This project is provided as-is for research purposes.
