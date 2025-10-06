import os
import json
from PIL import Image, ImageFilter
import time
import shutil

# --- IMPORTANT: Setup for running this example ---
# To run this example, you need to ensure Python can find the 'core' package.
# The easiest way is to run it from the parent directory of 'photo_album_tool'
# using: `python -m photo_album_tool.run_example`
# Or, if running directly from `photo_album_tool/` directory:
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# This example assumes you are running from the parent directory
# or have correctly set up your PYTHONPATH.

# Import modules from our core package
from core.config import Config
from core.utils.logger import logger, setup_logging # Import setup_logging for initial config
from core.utils.image_processor import ImageProcessor
from core.models.quality_extractor import QualityFeatureExtractor
from core.models.dl_extractor import DeepLearningFeatureExtractor
from core.models.hash_extractor import PerceptualHashExtractor
from core.scoring.similarity_analyzer import SimilarityAnalyzer
from core.scoring.image_scorer import ImageScorer
from typing import List, Dict, Any
from tqdm import tqdm

# --- Main Example Class (similar to PhotoAlbumTool but for direct usage) ---
class ExamplePhotoProcessor:
    def __init__(self, image_dir: str, config: Config):
        self.image_dir = image_dir
        self.config = config
        self.image_data: List[Dict[str, Any]] = []

        self.dl_extractor = DeepLearningFeatureExtractor()
        self.quality_extractor = QualityFeatureExtractor()
        self.phash_extractor = PerceptualHashExtractor()
        self.similarity_analyzer = SimilarityAnalyzer(config)
        self.scorer = ImageScorer(config)

    def _get_image_paths(self) -> List[str]:
        """Collects valid image paths from the directory (and subdirectories)."""
        image_paths = []
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(self.config.IMAGE_EXTENSIONS):
                    image_paths.append(os.path.join(root, file))
        logger.info(f"Found {len(image_paths)} images in '{self.image_dir}' (including subdirectories).")
        return image_paths

    def process_all_images(self):
        """Orchestrates feature extraction, clustering, and scoring."""
        image_paths = self._get_image_paths()
        all_quality_features_for_scaling = []

        logger.info("Starting feature extraction...")
        for img_path in tqdm(image_paths, desc="Extracting features"):
            processor = ImageProcessor(img_path)
            if not processor.load():
                continue

            phash = self.phash_extractor.extract(processor.pil_image)
            quality_features = self.quality_extractor.extract(processor.cv_image)
            dl_embedding = self.dl_extractor.extract(processor.pil_image)

            self.image_data.append({
                'path': img_path,
                'phash': phash,
                'quality': quality_features,
                'dl_embedding': dl_embedding,
                'cluster_id': -1,
                'final_score': 0.0
            })
            all_quality_features_for_scaling.append(quality_features)
        
        self.scorer.fit_scalers(all_quality_features_for_scaling)
        logger.info("All features extracted and scoring scalers fitted.")

        logger.info("Starting image clustering...")
        self.image_data = self.similarity_analyzer.cluster_images(self.image_data)
        logger.info("Image clustering complete.")

        logger.info("Starting individual image scoring...")
        for i in tqdm(range(len(self.image_data)), desc="Scoring images"):
            score = self.scorer.calculate_single_image_score(self.image_data[i])
            self.image_data[i]['final_score'] = score
        logger.info("All individual image scores calculated.")

    def get_ranked_images(self, num_best: int = 10) -> List[Dict[str, Any]]:
        """Selects N best images, handling clusters for diversity."""
        logger.info(f"Selecting top {num_best} images, considering clusters for diversity...")
        
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        noise_images: List[Dict[str, Any]] = []

        for img_data in self.image_data:
            cluster_id = img_data['cluster_id']
            if cluster_id == -1:
                noise_images.append(img_data)
            else:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(img_data)
        
        selected_images: List[Dict[str, Any]] = []

        for cluster_id, img_list in clusters.items():
            best_in_cluster = max(img_list, key=lambda x: x['final_score'])
            selected_images.append(best_in_cluster)
        
        selected_images.extend(noise_images)
        selected_images.sort(key=lambda x: x['final_score'], reverse=True)
        
        logger.info(f"Selected {len(selected_images)} unique candidates before final trimming.")
        return selected_images[:num_best]

# --- Dummy Image Generation (for example usage) ---
def create_example_images(image_dir: str):
    """Generates a small set of dummy images for demonstration purposes."""
    logger.info(f"Creating example images in '{image_dir}'...")
    os.makedirs(image_dir, exist_ok=True)
    
    # Scene 1: Clear vs Blurred
    img1 = Image.new('RGB', (800, 600), color = 'white')
    img1.save(os.path.join(image_dir, 'scene1_clear_01.jpg'))
    img1.save(os.path.join(image_dir, 'scene1_clear_02.jpg')) # Another clear shot of same scene

    img1_blur = img1.filter(ImageFilter.GaussianBlur(radius=5))
    img1_blur.save(os.path.join(image_dir, 'scene1_blurred_01.jpg'))
    img1_blur_heavy = img1.filter(ImageFilter.GaussianBlur(radius=15))
    img1_blur_heavy.save(os.path.join(image_dir, 'scene1_blurred_02_heavy.jpg'))

    # Scene 2: Dark vs Bright
    img2_dark = Image.new('RGB', (800, 600), color = 'darkblue')
    img2_dark.save(os.path.join(image_dir, 'scene2_dark_01.jpg'))
    img2_dark_similar = Image.new('RGB', (800, 600), color = (0, 0, 100)) # Slightly less dark blue
    img2_dark_similar.save(os.path.join(image_dir, 'scene2_dark_02_similar.jpg'))

    img2_bright = Image.new('RGB', (800, 600), color = 'yellow')
    img2_bright.save(os.path.join(image_dir, 'scene2_bright_01.jpg'))

    # Scene 3: Colorful/Good Contrast
    img3 = Image.new('RGB', (800, 600), color = (50, 200, 100)) # Greenish, good contrast
    img3.save(os.path.join(image_dir, 'scene3_colorful_01.jpg'))
    img3_similar = Image.new('RGB', (800, 600), color = (60, 190, 110)) # Very similar colorful
    img3_similar.save(os.path.join(image_dir, 'scene3_colorful_02_similar.jpg'))

    # Scene 4: Low Contrast
    img4 = Image.new('RGB', (800, 600), color = (120, 120, 120)) # Low contrast gray
    img4.save(os.path.join(image_dir, 'scene4_low_contrast_01.jpg'))

    logger.info(f"Example images generation complete in '{image_dir}'.")


# --- Main execution block for the example ---
if __name__ == "__main__":
    # Ensure logging is set up for the example
    setup_logging()
    
    # Define a temporary directory for example images
    EXAMPLE_IMAGE_DIR = "images"
    OUTPUT_FILE = "example_scores.json"

    # Clean up previous run's directory
    # if os.path.exists(EXAMPLE_IMAGE_DIR):
    #     shutil.rmtree(EXAMPLE_IMAGE_DIR)
    #     logger.info(f"Cleaned up previous example directory: {EXAMPLE_IMAGE_DIR}")
    
    # Generate fresh example images
    # create_example_images(EXAMPLE_IMAGE_DIR)

    # Instantiate the processor with example directory and default config
    # You can modify Config attributes here for specific example tests
    # e.g., Config.WEIGHT_BLUR = 0.6
    # Config.DBSCAN_EPS = 0.8
    
    start_time = time.time()
    logger.info(f"Starting photo album processing for '{EXAMPLE_IMAGE_DIR}'...")

    processor = ExamplePhotoProcessor(EXAMPLE_IMAGE_DIR, Config)
    processor.process_all_images()
    
    num_to_recommend = 3
    top_images = processor.get_ranked_images(num_best=num_to_recommend)

    logger.info(f"\n--- Top {len(top_images)} Recommended Images for Album ---")
    output_results = []
    for i, photo in enumerate(top_images):
        logger.info(f"{i+1}. Path: {photo['path']}, Score: {photo['final_score']:.4f}, Cluster: {photo['cluster_id']}")
        output_results.append({
            "rank": i + 1,
            "path": photo['path'],
            "score": round(photo['final_score'], 4),
            "cluster_id": photo['cluster_id'],
            "quality_features": photo['quality']
        })

    # Save all processed image data (including scores)
    all_sorted_images = sorted(processor.image_data, key=lambda x: x['final_score'], reverse=True)
    all_output_results = []
    for i, photo in enumerate(all_sorted_images):
         all_output_results.append({
            "rank": i + 1,
            "path": photo['path'],
            "score": round(photo['final_score'], 4),
            "cluster_id": photo['cluster_id'],
            "quality_features": photo['quality']
        })
    
    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_output_results, f, indent=4)
        logger.info(f"All image scores saved to '{OUTPUT_FILE}'")
    except IOError as e:
        logger.error(f"Could not write to output file '{OUTPUT_FILE}': {e}")

    end_time = time.time()
    logger.info(f"Processing finished in {end_time - start_time:.2f} seconds.")

    # Optional: Keep the example directory for inspection, or uncomment to clean up
    # shutil.rmtree(EXAMPLE_IMAGE_DIR)
    # logger.info(f"Cleaned up example directory: {EXAMPLE_IMAGE_DIR}")