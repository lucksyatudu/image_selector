import os
from typing import Tuple

class Config:
    """
    Configuration settings for the Photo Album Tool.
    """
    IMAGE_EXTENSIONS: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
    DL_MODEL_IMAGE_SIZE: Tuple[int, int] = (224, 224) # VGG16 expects 224x224
    
    # Clustering parameters for DBSCAN (on DL embeddings)
    DBSCAN_EPS: float = 0.5 # The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    DBSCAN_MIN_SAMPLES: int = 3 # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    
    # Scoring weights (adjust these based on desired outcome) - ideally sum to 1.0
    WEIGHT_BLUR: float = 0.4
    WEIGHT_EXPOSURE_BALANCE: float = 0.3 # Penalizes too dark/bright
    WEIGHT_CONTRAST: float = 0.3
    # Add more weights for other features if implemented (e.g., face detection, composition)

    # Blur threshold for filtering (if you want to discard very blurry images)
    MIN_BLUR_VARIANCE: float = 50.0 # Adjust based on experimentation (lower values are blurrier)

    # Ideal mean intensity for exposure (0-255 scale)
    IDEAL_MEAN_INTENSITY: float = 128.0

    # Logging level
    LOG_LEVEL: str = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL