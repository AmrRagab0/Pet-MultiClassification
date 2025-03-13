import unittest
import cv2
import numpy as np
from resize_images import resize_image

class TestResizeImage(unittest.TestCase):
    def test_resize_image(self):
        # Create a dummy image (100x100 pixels)
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Resize the image to 50x50 pixels
        resized_image = resize_image(dummy_image, (50, 50))
        
        # Check if the resized image has the correct dimensions
        self.assertEqual(resized_image.shape, (50, 50, 3))

if __name__ == '__main__':
    unittest.main()