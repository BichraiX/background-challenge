import os
import numpy as np
from PIL import Image
import random
import time
from scipy import ndimage
import re
import shutil

# Keeping our optimized tiled_background function
def create_tiled_background(image, verbose=False):
    """
    Takes an image with transparent areas and fills rectangular regions
    that cover the transparent patches by tiling a large non-transparent
    rectangle from the image.
    
    Args:
        image: A PIL Image with transparency (PNG with 4 channels)
        verbose: If True, print timing information
        
    Returns:
        A PIL Image with transparent patches filled with tiled rectangles
    """
    start_time = time.time()
    
    # Convert to numpy array for easier manipulation
    img_array = np.array(image)
    
    # Extract alpha channel (4th channel)
    alpha = img_array[:, :, 3]
    
    # Create mask for transparent areas
    transparent_mask = alpha == 0
    
    # Quick return if no transparent pixels
    if not np.any(transparent_mask):
        return image
    
    # Create mask for non-transparent areas
    non_transparent_mask = ~transparent_mask
    
    # Find a large non-transparent rectangle to use as tile
    x, y, rect_width, rect_height = find_large_rectangle_fast(non_transparent_mask)
    
    # Extract the rectangle as our tile
    tile = img_array[y:y+rect_height, x:x+rect_width].copy()
    
    # Create a new image by copying the original
    result = img_array.copy()
    
    # Label connected transparent regions
    labeled_array, num_features = ndimage.label(transparent_mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    
    # Pre-compute patch bounding boxes to avoid redundant calculations
    bounding_boxes = []
    for label in range(1, num_features + 1):
        # Get coordinates where this label occurs
        y_indices, x_indices = np.where(labeled_array == label)
        if len(y_indices) == 0:
            continue
            
        # Find the bounding box
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Store the bounding box
        bounding_boxes.append((label, x_min, y_min, x_max - x_min + 1, y_max - y_min + 1))
    
    # Process patches
    for label, x_min, y_min, width, height in bounding_boxes:
        fill_rectangle_with_tile_vectorized(
            result, 
            tile,
            x_min, y_min, 
            width, height
        )
    
    total_time = time.time() - start_time
    if verbose:
        print(f"Tiled background created in {total_time:.4f} seconds")
    
    return Image.fromarray(result)

def find_large_rectangle_fast(mask, min_area_percent=5):
    """
    Find a large rectangle in the non-transparent area using a faster approach.
    Uses an early termination strategy when a "good enough" rectangle is found.
    
    Args:
        mask: 2D binary mask where True indicates non-transparent pixels
        min_area_percent: Minimum percentage of non-transparent area to consider "good enough"
        
    Returns:
        (x, y, width, height) of a large rectangle
    """
    height, width = mask.shape
    max_area = 0
    best_rect = (0, 0, 1, 1)  # Default small rectangle
    
    # Calculate the total non-transparent area
    total_non_transparent = np.sum(mask)
    
    # Calculate the minimum area we consider "good enough"
    min_acceptable_area = total_non_transparent * min_area_percent / 100
    
    # Sample starting points in a grid pattern
    # Use a more aggressive sampling for larger images
    y_step = max(1, height // min(20, height // 10))  # More adaptive sampling
    x_step = max(1, width // min(20, width // 10))
    
    # Get non-zero coordinates for faster sampling
    y_coords, x_coords = np.where(mask)
    if len(y_coords) > 100:  # Only use this optimization for images with enough non-transparent pixels
        # Take a random sample of points to check
        indices = np.random.choice(len(y_coords), min(100, len(y_coords)), replace=False)
        sample_points = list(zip(y_coords[indices], x_coords[indices]))
    else:
        # Fall back to grid sampling for smaller images
        sample_points = [(y, x) for y in range(0, height, y_step) 
                                for x in range(0, width, x_step) if mask[y, x]]
    
    # Process each sample point
    for y, x in sample_points:
        # Skip transparent starting points (redundant check but just in case)
        if not mask[y, x]:
            continue
        
        # Find maximum width using vectorized operations
        # Get the current row and find the first False value
        row = mask[y, x:]
        if len(row) == 0:
            continue
            
        # Find the first False in the row
        false_indices = np.where(~row)[0]
        max_width = len(row) if len(false_indices) == 0 else false_indices[0]
        
        # If width is too small, skip
        if max_width < 2:
            continue
        
        # Find maximum height using vectorized operations
        max_height = 0
        for h in range(min(50, height - y)):  # Limit height search to improve performance
            if y + h >= height or not np.all(mask[y + h, x:x + max_width]):
                break
            max_height = h + 1
        
        # Check if this rectangle is better
        area = max_width * max_height
        if area > max_area:
            max_area = area
            best_rect = (x, y, max_width, max_height)
            
            # Early termination if we find a "good enough" rectangle
            if area > min_acceptable_area:
                return best_rect
    
    return best_rect

def fill_rectangle_with_tile_vectorized(image_array, tile, x_start, y_start, width, height):
    """
    Fill a rectangular region with a tiled pattern using vectorized operations.
    
    Args:
        image_array: The image array to modify
        tile: The tile pattern to use for filling
        x_start, y_start: Top-left corner of the rectangle
        width, height: Dimensions of the rectangle
    """
    tile_height, tile_width = tile.shape[:2]
    img_height, img_width = image_array.shape[:2]
    
    # Ensure we don't go out of bounds
    if x_start >= img_width or y_start >= img_height:
        return
    
    # Adjust width and height to image boundaries
    width = min(width, img_width - x_start)
    height = min(height, img_height - y_start)
    
    # Create a tiled pattern to cover the entire rectangle
    # First, determine how many repetitions we need
    repeat_y = (height + tile_height - 1) // tile_height
    repeat_x = (width + tile_width - 1) // tile_width
    
    # For small rectangles, use the simple loop approach
    if width <= tile_width * 2 and height <= tile_height * 2:
        for y_offset in range(0, height, tile_height):
            for x_offset in range(0, width, tile_width):
                h = min(tile_height, height - y_offset)
                w = min(tile_width, width - x_offset)
                
                y_dest = y_start + y_offset
                x_dest = x_start + x_offset
                
                if y_dest + h <= img_height and x_dest + w <= img_width:
                    image_array[y_dest:y_dest+h, x_dest:x_dest+w] = tile[:h, :w]
        return
    
    # For larger rectangles, create a complete tiled pattern first
    # This is faster for large areas because we do fewer individual assignments
    tiled_pattern = np.tile(tile, (repeat_y, repeat_x, 1))
    
    # Crop the tiled pattern to the rectangle size
    tiled_pattern = tiled_pattern[:height, :width]
    
    # Assign to the image array
    image_array[y_start:y_start+height, x_start:x_start+width] = tiled_pattern

# Part 1: Create all tiled backgrounds
def create_all_tiled_backgrounds(dataset_dir, cache_dir=None, verbose=True):
    """
    First pass: Create tiled backgrounds for all background images in the dataset.
    
    Args:
        dataset_dir: Root directory of the dataset
        cache_dir: Directory to store tiled backgrounds (defaults to 'tiled_cache' in dataset_dir)
        verbose: Whether to print progress information
    
    Returns:
        Dictionary mapping original background paths to tiled background paths
    """
    if cache_dir is None:
        cache_dir = os.path.join(dataset_dir, 'tiled_cache')
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    class_dirs = [d for d in class_dirs if d != 'tiled_cache']  # Exclude cache dir
    
    # Dictionary to store mapping
    bg_to_tiled = {}
    
    # Process each class
    for class_name in class_dirs:
        if verbose:
            print(f"Processing class: {class_name}")
        
        class_dir = os.path.join(dataset_dir, class_name)
        class_cache_dir = os.path.join(cache_dir, class_name)
        os.makedirs(class_cache_dir, exist_ok=True)
        
        # Find all background images
        bg_images = []
        for filename in os.listdir(class_dir):
            if filename.endswith('_bg.png'):
                bg_images.append(filename)
        
        if verbose:
            print(f"  Found {len(bg_images)} background images")
        
        # Process each background image
        for i, bg_filename in enumerate(bg_images):
            if verbose and i % 10 == 0:
                print(f"  Processing background {i+1}/{len(bg_images)}")
            
            bg_path = os.path.join(class_dir, bg_filename)
            
            try:
                # Load background image
                bg_img = Image.open(bg_path)
                
                # Create tiled background
                tiled_bg = create_tiled_background(bg_img, verbose=False)
                
                # Save tiled background
                tiled_filename = f"tiled_{bg_filename}"
                tiled_path = os.path.join(class_cache_dir, tiled_filename)
                tiled_bg.save(tiled_path)
                
                # Add to mapping
                bg_to_tiled[bg_path] = tiled_path
                
            except Exception as e:
                print(f"  Error processing {bg_path}: {e}")
    
    if verbose:
        print(f"Created {len(bg_to_tiled)} tiled backgrounds")
    
    return bg_to_tiled

# Helper function to find image pairs in a directory
def find_image_pairs(directory):
    """
    Find pairs of foreground/background images in a directory.
    
    Args:
        directory: Directory to search for image pairs
    
    Returns:
        List of tuples (fg_path, bg_path)
    """
    # Get all PNG files
    all_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    
    # Group by base name
    pairs = {}
    for filename in all_files:
        # Extract base name (everything before _fg or _bg)
        match = re.match(r'(.+)_(fg|bg)\.png', filename)
        if match:
            base_name, img_type = match.groups()
            if base_name not in pairs:
                pairs[base_name] = {'fg': None, 'bg': None}
            pairs[base_name][img_type] = os.path.join(directory, filename)
    
    # Filter only complete pairs and convert to list of tuples
    result = []
    for base_name, paths in pairs.items():
        if paths['fg'] and paths['bg']:
            result.append((paths['fg'], paths['bg']))
    
    return result

# Functions to create each variation type
def create_original(fg_img, bg_img):
    """Combine foreground and background to create original image"""
    # Make sure both images are the same size
    if fg_img.size != bg_img.size:
        bg_img = bg_img.resize(fg_img.size)
    
    # Create a new image and paste background
    result = bg_img.copy()
    # Paste foreground with alpha
    result.paste(fg_img, (0, 0), fg_img)
    
    return result

def create_only_bg_b(bg_img):
    """Background with black rectangles over transparent areas"""
    # Convert to numpy array
    bg_array = np.array(bg_img)
    
    # Extract alpha channel
    alpha = bg_array[:, :, 3]
    transparent_mask = alpha == 0
    
    # If no transparent areas, return original
    if not np.any(transparent_mask):
        return bg_img
    
    # Create result array
    result = bg_array.copy()
    
    # Label connected transparent regions
    labeled_array, num_features = ndimage.label(transparent_mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    
    # Process each connected component
    for label in range(1, num_features + 1):
        # Get coordinates where this label occurs
        y_indices, x_indices = np.where(labeled_array == label)
        if len(y_indices) == 0:
            continue
            
        # Find the bounding box
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Fill the bounding rectangle with black
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        # Create black rectangle (R,G,B,A)
        black_rect = np.zeros((height, width, 4), dtype=np.uint8)
        black_rect[:, :, 3] = 255  # Fully opaque
        
        # Place the black rectangle
        result[y_min:y_max+1, x_min:x_max+1] = black_rect
    
    return Image.fromarray(result)

def create_only_bg_t(bg_img):
    """Background with tiled patches over transparent areas"""
    return create_tiled_background(bg_img)

def create_no_fg(bg_img):
    """Background with black filling the exact transparent areas"""
    # Convert to numpy array
    bg_array = np.array(bg_img)
    
    # Extract alpha channel
    alpha = bg_array[:, :, 3]
    transparent_mask = alpha == 0
    
    # If no transparent areas, return original
    if not np.any(transparent_mask):
        return bg_img
    
    # Create result array
    result = bg_array.copy()
    
    # Fill transparent pixels with black (opaque)
    result[transparent_mask, 0] = 0  # R
    result[transparent_mask, 1] = 0  # G
    result[transparent_mask, 2] = 0  # B
    result[transparent_mask, 3] = 255  # A (opaque)
    
    return Image.fromarray(result)

def create_only_fg(fg_img):
    """Foreground with black replacing transparent parts"""
    # Convert to numpy array
    fg_array = np.array(fg_img)
    
    # Extract alpha channel
    alpha = fg_array[:, :, 3]
    transparent_mask = alpha == 0
    
    # Create result array
    result = fg_array.copy()
    
    # Fill transparent pixels with black (opaque)
    result[transparent_mask, 0] = 0  # R
    result[transparent_mask, 1] = 0  # G
    result[transparent_mask, 2] = 0  # B
    result[transparent_mask, 3] = 255  # A (opaque)
    
    return Image.fromarray(result)

def create_mixed_same(fg_img, bg_tiled, exclude_path=None):
    """Foreground over tiled background from same class"""
    # Make sure both images are the same size
    if fg_img.size != bg_tiled.size:
        bg_tiled = bg_tiled.resize(fg_img.size)
    
    # Create a new image and paste background
    result = bg_tiled.copy()
    # Paste foreground with alpha
    result.paste(fg_img, (0, 0), fg_img)
    
    return result

def create_mixed_rand(fg_img, bg_tiled):
    """Foreground over tiled background from random class"""
    # Same as create_mixed_same, just with random bg
    return create_mixed_same(fg_img, bg_tiled)

def create_mixed_next(fg_img, bg_tiled):
    """Foreground over tiled background from next class"""
    # Same as create_mixed_same, just with bg from next class
    return create_mixed_same(fg_img, bg_tiled)

def create_all_variations(dataset_dir, output_dir, bg_tiled_cache):
    """
    Second pass: Create all variations for each image pair.
    
    Args:
        dataset_dir: Root directory of the dataset
        output_dir: Directory to save output variations
        bg_tiled_cache: Dict mapping background paths to tiled background paths
    """
    # Get all class directories in the specified order
    class_names = ['bird', 'carnivore', 'dog', 'fish', 'insect', 
                  'instrument', 'primate', 'reptile', 'vehicle']
    
    # Variation types
    variation_types = [
        'original', 'only_bg_b', 'only_bg_t', 'no_fg', 
        'only_fg', 'mixed_same', 'mixed_rand', 'mixed_next'
    ]
    
    # Filter to only include classes that actually exist in the dataset
    class_dirs = []
    for class_name in class_names:
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            class_dirs.append((class_name, class_path))
    
    # Create a dictionary to store cached background images by class
    class_to_bgs = {}
    
    # First, prepare the class_to_bgs dictionary
    for class_name, class_path in class_dirs:
        # Find all background images in this class
        bg_paths = []
        for filename in os.listdir(class_path):
            if filename.endswith('_bg.png'):
                bg_path = os.path.join(class_path, filename)
                if bg_path in bg_tiled_cache:
                    bg_paths.append(bg_path)
        
        class_to_bgs[class_name] = bg_paths
    
    # Process each class
    for i, (class_name, class_path) in enumerate(class_dirs):
        print(f"Processing variations for class: {class_name}")
        
        # Create class output directory
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Create variation directories for this class
        variation_dirs = {}
        for variation in variation_types:
            variation_dirs[variation] = os.path.join(class_output_dir, variation)
            os.makedirs(variation_dirs[variation], exist_ok=True)
        
        # Determine the next class index (wrap around to the first class if needed)
        next_class_idx = (i + 1) % len(class_dirs)
        next_class_name, next_class_path = class_dirs[next_class_idx]
        
        # Find all image pairs in this class
        image_pairs = find_image_pairs(class_path)
        print(f"  Found {len(image_pairs)} image pairs")
        
        # Process each image pair
        for fg_path, bg_path in image_pairs:
            try:
                # Extract base filename without extension
                base_name = os.path.basename(fg_path).replace('_fg.png', '')
                
                # Load the original images
                fg_img = Image.open(fg_path)
                bg_img = Image.open(bg_path)
                
                # Create Original variation
                original = create_original(fg_img, bg_img)
                original.save(os.path.join(variation_dirs['original'], f"{base_name}.png"))
                
                # Create Only-BG-B variation
                only_bg_b = create_only_bg_b(bg_img)
                only_bg_b.save(os.path.join(variation_dirs['only_bg_b'], f"{base_name}.png"))
                
                # Create Only-BG-T variation (use cached version if available)
                if bg_path in bg_tiled_cache:
                    # Copy the cached file
                    shutil.copy(bg_tiled_cache[bg_path], 
                               os.path.join(variation_dirs['only_bg_t'], f"{base_name}.png"))
                else:
                    # Create it if not in cache
                    only_bg_t = create_only_bg_t(bg_img)
                    only_bg_t.save(os.path.join(variation_dirs['only_bg_t'], f"{base_name}.png"))
                
                # Create No-FG variation
                no_fg = create_no_fg(bg_img)
                no_fg.save(os.path.join(variation_dirs['no_fg'], f"{base_name}.png"))
                
                # Create Only-FG variation
                only_fg = create_only_fg(fg_img)
                only_fg.save(os.path.join(variation_dirs['only_fg'], f"{base_name}.png"))
                
                # Create Mixed-Same variation
                # Select a random background from the same class (excluding the current one)
                same_class_bgs = [b for b in class_to_bgs[class_name] if b != bg_path]
                if same_class_bgs:
                    random_bg_path = random.choice(same_class_bgs)
                    random_bg_tiled = Image.open(bg_tiled_cache[random_bg_path])
                    mixed_same = create_mixed_same(fg_img, random_bg_tiled)
                    mixed_same.save(os.path.join(variation_dirs['mixed_same'], f"{base_name}.png"))
                
                # Create Mixed-Rand variation
                # Select a random background from a different class
                other_classes = [c for c in class_to_bgs.keys() if c != class_name]
                if other_classes:
                    random_class = random.choice(other_classes)
                    if class_to_bgs[random_class]:
                        random_bg_path = random.choice(class_to_bgs[random_class])
                        random_bg_tiled = Image.open(bg_tiled_cache[random_bg_path])
                        mixed_rand = create_mixed_rand(fg_img, random_bg_tiled)
                        mixed_rand.save(os.path.join(variation_dirs['mixed_rand'], f"{base_name}.png"))
                
                # Create Mixed-Next variation
                # Use a background from the next class
                if next_class_name in class_to_bgs and class_to_bgs[next_class_name]:
                    next_bg_path = random.choice(class_to_bgs[next_class_name])
                    next_bg_tiled = Image.open(bg_tiled_cache[next_bg_path])
                    mixed_next = create_mixed_next(fg_img, next_bg_tiled)
                    mixed_next.save(os.path.join(variation_dirs['mixed_next'], f"{base_name}.png"))
                
            except Exception as e:
                print(f"  Error processing pair {fg_path}, {bg_path}: {e}")
        
        print(f"  Completed class: {class_name}")
    
    print("All variations created successfully!")

# Main function to process the dataset
def process_dataset(dataset_dir, output_dir=None, verbose=True):
    """
    Process the entire dataset, creating all variations for each image pair.
    
    Args:
        dataset_dir: Root directory of the dataset
        output_dir: Directory to save results (defaults to 'output' in dataset_dir)
        verbose: Whether to print progress information
    """
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Create all tiled backgrounds
    cache_dir = os.path.join(dataset_dir, 'tiled_cache')
    if verbose:
        print("Step 1: Creating tiled backgrounds for all images...")
    bg_tiled_cache = create_all_tiled_backgrounds(dataset_dir, cache_dir, verbose)
    
    # Step 2: Create all variations
    if verbose:
        print("\nStep 2: Creating all variations...")
    create_all_variations(dataset_dir, output_dir, bg_tiled_cache)
    
    if verbose:
        print("\nAll processing complete!")

# Process dataset with absolute paths
if __name__ == "__main__":
    dataset_dir = "/Data/amine.chraibi/dataset"
    output_dir = "/Data/amine.chraibi/processed_dataset"
    process_dataset(dataset_dir, output_dir, verbose=True)
