import os
import cv2
from PIL import Image
import numpy as np
from typing import Dict, Any, Tuple

from core.utils.logger import logger

class ImageProcessor:
    """Handles loading and basic processing of images."""
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.pil_image: Image.Image = None
        self.cv_image: np.ndarray = None

    def load(self) -> bool:
        """Loads the image using PIL and OpenCV."""
        try:
            self.pil_image = Image.open(self.image_path).convert("RGB")
            # OpenCV loads as BGR by default, which is fine for quality metrics
            self.cv_image = cv2.imread(self.image_path)
            if self.cv_image is None:
                raise ValueError(f"OpenCV failed to load image: {self.image_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load image {self.image_path}: {e}")
            return False