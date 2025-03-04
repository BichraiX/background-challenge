import os
import requests
import tarfile
import zipfile
from tqdm import tqdm

def download_file(url, dest_path):
    """Download a file from a URL to a local destination with a progress bar."""
    if os.path.exists(dest_path):
        print(f"[INFO] {dest_path} already exists, skipping download.")
        return
    print(f"[INFO] Downloading from {url} to {dest_path}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with open(dest_path, "wb") as file, tqdm(
        total=total_size, unit="iB", unit_scale=True, desc=os.path.basename(dest_path)
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))
    print(f"[INFO] Download complete: {dest_path}")

def extract_file(file_path, extract_to):
    """
    Automatically extract an archive (zip or tar) to a given directory.
    Provides debugging output if the file appears too small.
    """
    # Check file size; if too small, print the first few bytes as text for debugging.
    file_size = os.path.getsize(file_path)
    if file_size < 100 * 1024:  # less than 100 KB
        print(f"[DEBUG] Warning: {file_path} is very small ({file_size} bytes).")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(300)
            print(f"[DEBUG] File content preview (first 300 chars):\n{content}")
        except Exception as e:
            print(f"[DEBUG] Unable to read file content: {e}")

    if zipfile.is_zipfile(file_path):
        print(f"[INFO] Extracting ZIP file {file_path} to {extract_to}")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"[INFO] Extraction complete.")
    elif tarfile.is_tarfile(file_path):
        print(f"[INFO] Extracting TAR file {file_path} to {extract_to}")
        with tarfile.open(file_path, "r") as tar_ref:
            tar_ref.extractall(extract_to)
        print(f"[INFO] Extraction complete.")
    else:
        raise ValueError(f"Unsupported archive format or corrupted file: {file_path}")

def download_sam_checkpoint(model_type="vit_h", save_dir="checkpoints"):
    """
    Download SAM checkpoint file.
    Supported model types: "vit_h", "vit_l", "vit_b".
    """
    sam_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }
    if model_type not in sam_urls:
        raise ValueError(f"Unsupported model type {model_type}. Choose from {list(sam_urls.keys())}")

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_url = sam_urls[model_type]
    checkpoint_path = os.path.join(save_dir, os.path.basename(checkpoint_url))
    download_file(checkpoint_url, checkpoint_path)
    return checkpoint_path

def download_pascal_voc(dest_dir="VOC2012"):
    """
    Download Pascal VOC 2012 train/val dataset.
    The archive is downloaded and then extracted into dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)
    voc_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    tar_path = os.path.join(dest_dir, "VOCtrainval_11-May-2012.tar")
    download_file(voc_url, tar_path)
    extract_file(tar_path, dest_dir)
    return dest_dir

def download_coco(dest_dir="COCO"):
    """
    Download COCO 2017 validation images and annotations.
    Note: These files are large, so ensure you have enough space.
    """
    os.makedirs(dest_dir, exist_ok=True)
    coco_images_url = "http://images.cocodataset.org/zips/val2017.zip"
    coco_ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    images_zip_path = os.path.join(dest_dir, "val2017.zip")
    ann_zip_path = os.path.join(dest_dir, "annotations_trainval2017.zip")
    
    download_file(coco_images_url, images_zip_path)
    download_file(coco_ann_url, ann_zip_path)
    
    extract_file(images_zip_path, dest_dir)
    extract_file(ann_zip_path, dest_dir)
    return dest_dir


def extract_zip_file(file_path, extract_to):
    """Extracts a ZIP file to a given directory."""
    print(f"[INFO] Extracting ZIP file {file_path} to {extract_to}")
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"[INFO] Extraction complete.")
    except zipfile.BadZipFile:
        print(f"[ERROR] The file {file_path} appears to be corrupted. Please delete and re-download.")
        exit(1)

def download_davis_2016(dest_dir="DAVIS"):
    """
    Download DAVIS 2016 dataset from the official source and extract it.
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    davis_url = "https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip"
    zip_path = os.path.join(dest_dir, "DAVIS-data.zip")
    
    download_file(davis_url, zip_path)
    extract_zip_file(zip_path, dest_dir)
    
    # Clean up the zip file after extraction
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"[INFO] Removed temporary file: {zip_path}")
    
    return dest_dir


def main():
    # Download SAM checkpoint
    print("== Downloading SAM Checkpoint ==")
    sam_checkpoint = download_sam_checkpoint(model_type="vit_h", save_dir="checkpoints")
    print(f"SAM checkpoint downloaded at: {sam_checkpoint}\n")
    
    # Download Pascal VOC 2012
    print("== Downloading Pascal VOC 2012 Dataset ==")
    voc_dir = download_pascal_voc(dest_dir="VOC2012")
    print(f"Pascal VOC 2012 downloaded and extracted to: {voc_dir}\n")
    
    # Download COCO 2017 validation images and annotations
    print("== Downloading COCO 2017 Dataset ==")
    coco_dir = download_coco(dest_dir="COCO")
    print(f"COCO 2017 downloaded and extracted to: {coco_dir}\n")
    
    # Download DAVIS 2016
    print("== Downloading DAVIS 2016 Dataset ==")
    davis_dir = download_davis_2016(dest_dir="DAVIS")
    print(f"DAVIS 2016 downloaded and extracted to: {davis_dir}\n")

    
    print("== All files have been downloaded and extracted ==")

if __name__ == "__main__":
    main()
