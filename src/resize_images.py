import cv2
import os
from PIL import Image
import numpy as np

def resize_image(image_path, target_size):
    """
    Resize an image to the target size.
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size (width, height).
    Returns:
        resized_image (numpy.ndarray): Resized image.
    """
    # Normalize the file path
    image_path = os.path.normpath(image_path)
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Warning: File does not exist at {image_path}. Skipping.")
        return None
    
    # Try reading the image with OpenCV
    image = cv2.imread(image_path)
    
    # If OpenCV fails, try reading with PIL
    if image is None:
        print(f"Warning: OpenCV failed to read image at {image_path}. Trying PIL...")
        try:
            image = Image.open(image_path)
            image = np.array(image)  # Convert PIL image to NumPy array
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        except Exception as e:
            print(f"Error: Unable to read image at {image_path}. Skipping. Error: {e}")
            return None
    
    # Resize the image
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    return resized_image

def resize_images_in_directory(input_dir, output_dir, target_size):
    """
    Resize all images in a directory and save them to another directory.
    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save resized images.
        target_size (tuple): Target size (width, height).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other formats if needed
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Skip if the image has already been processed
            if os.path.exists(output_path):
                print(f"Skipped (already processed): {filename}")
                continue
            
            # Resize the image
            resized_image = resize_image(input_path, target_size)
            
            if resized_image is not None:
                cv2.imwrite(output_path, resized_image)
                print(f"Processed: {filename}")
            else:
                print(f"Skipped: {filename}")