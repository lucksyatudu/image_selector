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
    WEIGHT_BLUR = 0.8
    WEIGHT_CONTRAST = 0.07
    WEIGHT_EXPOSURE_BALANCE = 0.10
    WEIGHT_DYNAMIC_RANGE_LOSS = 0.04
    WEIGHT_BRIGHTNESS_UNIFORMITY = 0.05
    WEIGHT_SHARP_EDGES = 0.05

    WEIGHT_FACE_PRESENT = 0.05
    WEIGHT_FACES_COUNT = 0.03
    WEIGHT_FACE_CENTERED = 0.05
    WEIGHT_FACE_SIZE_RATIO = 0.03
    WEIGHT_EYES_OPEN = 0.10
    WEIGHT_ORIENTATION = 0.10
    WEIGHT_EMOTION = 0.08

    WEIGHT_SATURATION = 0.04
    WEIGHT_BACKGROUND_COMPLEXITY = 0.06
    WEIGHT_EXPOSURE_HIGHLIGHTS = 0.07

    # Ideal values and thresholds for quality features
    IDEAL_MEAN_INTENSITY = 128
    MIN_BLUR_VARIANCE = 50
    # Add more weights for other features if implemented (e.g., face detection, composition)

    # Blur threshold for filtering (if you want to discard very blurry images)
    MIN_BLUR_VARIANCE: float = 50.0 # Adjust based on experimentation (lower values are blurrier)

    # Ideal mean intensity for exposure (0-255 scale)
    IDEAL_MEAN_INTENSITY: float = 128.0

    # Logging level
    LOG_LEVEL: str = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL