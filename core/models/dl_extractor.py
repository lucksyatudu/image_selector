import os
import numpy as np
from PIL import Image
from typing import Dict, Any, Union # Import Union

from core.config import Config
from core.utils.logger import logger

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
except ImportError:
    logger.warning("TensorFlow/Keras not installed. Deep Learning embeddings will be skipped.")
    tf = None

class DeepLearningFeatureExtractor:
    """Extracts deep learning embeddings using a pre-trained CNN (VGG16)."""
    def __init__(self):
        self.model = None
        if tf is None:
            logger.warning("Deep Learning Feature Extractor is disabled as TensorFlow is not available.")
            return
        
        # Singleton pattern for model loading
        if not hasattr(DeepLearningFeatureExtractor, '_vgg16_model'):
            logger.info("Loading VGG16 model for deep learning embeddings...")
            try:
                DeepLearningFeatureExtractor._vgg16_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
                logger.info("VGG16 model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load VGG16 model: {e}. Deep Learning embeddings will be skipped.")
        
        self.model = DeepLearningFeatureExtractor._vgg16_model

    # FIX: Use typing.Union for Python versions < 3.10
    def extract(self, pil_image: Image.Image) -> Union[np.ndarray, None]:
        """Extracts DL embedding from a PIL image."""
        if self.model is None:
            return None
        try:
            img_resized = pil_image.resize(Config.DL_MODEL_IMAGE_SIZE)
            x = keras_image.img_to_array(img_resized)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            embedding = self.model.predict(x, verbose=0) # verbose=0 to suppress Keras progress bar
            return embedding.flatten()
        except Exception as e:
            logger.error(f"Failed to extract DL embedding: {e}")
            return None