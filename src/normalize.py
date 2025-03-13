import numpy as np

def normalize_image(image):
    """
    Normalize pixel values to the range [0, 1].
    Args:
        image (numpy.ndarray): Input image.
    Returns:
        normalized_image (numpy.ndarray): Normalized image.
    """
    return image / 255.0