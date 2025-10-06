import imagehash
from PIL import Image
from typing import Dict, Any

from core.utils.logger import logger

class PerceptualHashExtractor:
    """Generates perceptual hashes for image similarity."""
    @staticmethod
    def extract(pil_image: Image.Image) -> str:
        """Generates pHash for a PIL image."""
        return str(imagehash.phash(pil_image))