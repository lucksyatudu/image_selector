import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any, Tuple

from core.config import Config
from core.utils.logger import logger

class ImageScorer:
    """
    Calculates a combined score for images based on various features.
    Handles normalization and weighted aggregation.
    """
    def __init__(self, config: Config):
        self.config = config
        self.quality_scalers: Dict[str, MinMaxScaler] = {}
        self.is_fitted = False

    def fit_scalers(self, all_quality_features: List[Dict[str, Any]]):
        """
        Fits MinMaxScaler for numerical quality features based on the entire dataset.
        This ensures scores are normalized consistently across all images.
        """
        if not all_quality_features:
            logger.warning("No quality features provided to fit scalers.")
            return

        feature_names = all_quality_features[0].keys()
        for name in feature_names:
            if name in ['blur_variance', 'contrast_std']: # Features where higher is generally better
                values = np.array([f[name] for f in all_quality_features]).reshape(-1, 1)
                scaler = MinMaxScaler()
                scaler.fit(values)
                self.quality_scalers[name] = scaler
        self.is_fitted = True
        logger.info("Quality feature scalers fitted.")

    def _score_blur(self, blur_variance: float) -> float:
        """Scores blur. Higher variance is better, scaled to [0,1]."""
        if 'blur_variance' not in self.quality_scalers:
            return 0.5 # Default if scaler not fitted
        scaled_value = self.quality_scalers['blur_variance'].transform(np.array([[blur_variance]]))[0][0]
        return scaled_value

    def _score_exposure_balance(self, mean_intensity: float) -> float:
        """
        Scores exposure balance. Closer to IDEAL_MEAN_INTENSITY is better.
        Uses a non-linear scoring where deviations are penalized.
        """
        # Max possible deviation from ideal (e.g., if ideal is 128, max is 128)
        max_deviation = max(self.config.IDEAL_MEAN_INTENSITY, 255 - self.config.IDEAL_MEAN_INTENSITY)
        
        # Current deviation
        deviation = abs(mean_intensity - self.config.IDEAL_MEAN_INTENSITY)
        
        # Score: 1 - (normalized_deviation)
        score = 1.0 - (deviation / max_deviation)
        return max(0.0, score) # Ensure score is not negative

    def _score_contrast(self, contrast_std: float) -> float:
        """Scores contrast. Higher standard deviation is better, scaled to [0,1]."""
        if 'contrast_std' not in self.quality_scalers:
            return 0.5 # Default if scaler not fitted
        scaled_value = self.quality_scalers['contrast_std'].transform(np.array([[contrast_std]]))[0][0]
        return scaled_value

    def calculate_single_image_score(self, image_data: Dict[str, Any]) -> float:
        """
        Calculates a combined quality score for a single image based on its features
        and configured weights.
        """
        if not self.is_fitted:
            logger.warning("Scalers not fitted. Cannot calculate accurate scores.")
            return 0.0

        quality = image_data['quality']
        
        blur_score = self._score_blur(quality['blur_variance'])
        exposure_score = self._score_exposure_balance(quality['mean_intensity'])
        contrast_score = self._score_contrast(quality['contrast_std'])

        # Apply a penalty for severely blurry images, regardless of other scores
        if quality['blur_variance'] < self.config.MIN_BLUR_VARIANCE:
            blur_score *= 0.1 # Significant penalty to highly blurry images

        total_score = (
            self.config.WEIGHT_BLUR * blur_score +
            self.config.WEIGHT_EXPOSURE_BALANCE * exposure_score +
            self.config.WEIGHT_CONTRAST * contrast_score
        )
        return total_score