import cv2
import numpy as np
from typing import Dict, Any, Tuple

from core.utils.logger import logger

class QualityFeatureExtractor:
    """Extracts quality-related features like blur, exposure, contrast."""
    @staticmethod
    def get_blur_variance(cv_image: np.ndarray) -> float:
        """Calculates blur using Laplacian variance."""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def get_exposure_metrics(cv_image: np.ndarray) -> Tuple[float, float]:
        """Calculates mean intensity and percentage of clipped pixels (over/under exposed)."""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        
        # Calculate clipped pixels (pure black or pure white)
        clipped_low = np.sum(gray < 10) # Adjust threshold if needed
        clipped_high = np.sum(gray > 245) # Adjust threshold if needed
        total_pixels = gray.size
        clipped_percent = (clipped_low + clipped_high) / total_pixels * 100
        
        return mean_intensity, clipped_percent

    @staticmethod
    def get_contrast_std(cv_image: np.ndarray) -> float:
        """Calculates contrast using standard deviation of pixel intensities."""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        return np.std(gray)

    def extract(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """Extracts all quality features."""
        return {
            'blur_variance': self.get_blur_variance(cv_image),
            'mean_intensity': self.get_exposure_metrics(cv_image)[0],
            'clipped_percent': self.get_exposure_metrics(cv_image)[1],
            'contrast_std': self.get_contrast_std(cv_image)
        }