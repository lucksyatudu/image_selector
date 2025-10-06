import os
import argparse
import json
import numpy as np
from typing import List, Dict, Any, Tuple

from tqdm import tqdm
from PIL import Image, ImageFilter # Used for dummy image generation

# Import modules from our core package
from core.config import Config
from core.utils.logger import logger, setup_logging
from core.utils.image_processor import ImageProcessor
from core.models.quality_extractor import QualityFeatureExtractor
from core.models.dl_extractor import DeepLearningFeatureExtractor
from core.models.hash_extractor import PerceptualHashExtractor
from core.scoring.similarity_analyzer import SimilarityAnalyzer
from core.scoring.image_scorer import ImageScorer

class PhotoAlbumTool:
    """
    Main class for orchestrating the photo album selection process.
    Loads images, extracts features, clusters, scores, and recommends best photos.
    """
    def __init__(self, image_dir: str, config: Config):
        self.image_dir = image_dir
        self.config = config
        self.image_data: List[Dict[str, Any]] = [] # Stores all extracted features and scores
        
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

    def extract_all_features(self):
        """
        Extracts all features (quality, DL embedding, pHash) for all images.
        Also fits the scoring scalers based on the extracted quality features.
        """
        image_paths = self._get_image_paths()
        all_quality_features_for_scaling = []

        logger.info("Starting feature extraction...")
        for img_path in tqdm(image_paths, desc="Extracting features"):
            processor = ImageProcessor(img_path)
            if not processor.load():
                continue # Skip if image loading failed

            phash = self.phash_extractor.extract(processor.pil_image)
            quality_features = self.quality_extractor.extract(processor.cv_image)
            dl_embedding = self.dl_extractor.extract(processor.pil_image)

            self.image_data.append({
                'path': img_path,
                'phash': phash,
                'quality': quality_features,
                'dl_embedding': dl_embedding,
                'cluster_id': -1, # Default to -1 (noise/unclustered)
                'final_score': 0.0
            })
            all_quality_features_for_scaling.append(quality_features)
        
        # Fit scalers once after collecting all quality features
        self.scorer.fit_scalers(all_quality_features_for_scaling)
        logger.info("All features extracted and scoring scalers fitted.")

    def calculate_all_scores(self):
        """Calculates final scores for all images using the ImageScorer."""
        logger.info("Calculating individual image scores...")
        for i in tqdm(range(len(self.image_data)), desc="Scoring images"):
            score = self.scorer.calculate_single_image_score(self.image_data[i])
            self.image_data[i]['final_score'] = score
        logger.info("All individual image scores calculated.")

    def select_best_images(self, n: int) -> List[Dict[str, Any]]:
        """
        Selects N best images, handling clusters to ensure diversity.
        Picks the highest-scoring image from each cluster, then adds unclustered
        (noise) images, and finally sorts to get the top N overall.
        """
        logger.info(f"Selecting top {n} images, considering clusters for diversity...")
        
        # Group images by cluster
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        noise_images: List[Dict[str, Any]] = []

        for img_data in self.image_data:
            cluster_id = img_data['cluster_id']
            if cluster_id == -1: # Noise points (unclustered by DBSCAN)
                noise_images.append(img_data)
            else:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(img_data)
        
        selected_images: List[Dict[str, Any]] = []

        # For each cluster, pick the highest scoring image
        for cluster_id, img_list in clusters.items():
            best_in_cluster = max(img_list, key=lambda x: x['final_score'])
            selected_images.append(best_in_cluster)
        
        # Add noise images, as they represent unique content not covered by clusters
        selected_images.extend(noise_images)
        
        # Sort all candidates by their score and return top N
        selected_images.sort(key=lambda x: x['final_score'], reverse=True)
        logger.info(f"Selected {len(selected_images)} unique candidates before final trimming.")
        return selected_images[:n]

    def run(self, num_best: int = 10, output_file: str = "image_scores.json"):
        """Runs the entire photo scoring pipeline."""
        self.extract_all_features()
        self.image_data = self.similarity_analyzer.cluster_images(self.image_data) # Update with cluster IDs
        
        # Optional: Identify near duplicates using pHash for informational purposes or further filtering
        # near_duplicates = self.similarity_analyzer.identify_near_duplicates_phash(self.image_data)
        # if near_duplicates:
        #     logger.info(f"Found {len(near_duplicates)} groups of near-duplicate images.")
        #     for k, v in list(near_duplicates.items())[:5]: # Log first 5 groups
        #         logger.debug(f"pHash group (representative {k}): {v}")

        self.calculate_all_scores()

        best_photos = self.select_best_images(num_best)

        logger.info(f"\n--- Top {len(best_photos)} Recommended Images for Album ---")
        results_for_output = []
        for i, photo in enumerate(best_photos):
            logger.info(f"{i+1}. Path: {photo['path']}, Score: {photo['final_score']:.4f}")
            results_for_output.append({
                "rank": i + 1,
                "path": photo['path'],
                "score": round(photo['final_score'], 4),
                "cluster_id": photo['cluster_id'],
                "quality_features": photo['quality']
            })

        # Save all scores to a JSON file, sorted by score
        all_sorted_images = sorted(self.image_data, key=lambda x: x['final_score'], reverse=True)
        all_results_for_output = []
        for i, photo in enumerate(all_sorted_images):
             all_results_for_output.append({
                "rank": i + 1,
                "path": photo['path'],
                "score": round(photo['final_score'], 4),
                "cluster_id": photo['cluster_id'],
                "quality_features": photo['quality']
            })
        
        try:
            with open(output_file, 'w') as f:
                json.dump(all_results_for_output, f, indent=4)
            logger.info(f"All image scores saved to '{output_file}'")
        except IOError as e:
            logger.error(f"Could not write to output file '{output_file}': {e}")

        return best_photos

def create_dummy_images(image_dir: str):
    """Generates a set of dummy images for demonstration purposes."""
    logger.info(f"Directory '{image_dir}' not found. Creating a dummy directory with example images.")
    os.makedirs(image_dir, exist_ok=True)
    
    img1 = Image.new('RGB', (800, 600), color = 'white')
    img1_path = os.path.join(image_dir, 'scene1_clear.jpg')
    img1.save(img1_path)
    logger.info(f"Generated: {img1_path}")

    img1_blur = img1.filter(ImageFilter.GaussianBlur(radius=5))
    img1_blur_path = os.path.join(image_dir, 'scene1_blurred.jpg')
    img1_blur.save(img1_blur_path)
    logger.info(f"Generated: {img1_blur_path}")

    img2 = Image.new('RGB', (800, 600), color = 'darkblue')
    img2_path = os.path.join(image_dir, 'scene2_dark.jpg')
    img2.save(img2_path)
    logger.info(f"Generated: {img2_path}")

    img2_bright = Image.new('RGB', (800, 600), color = 'yellow')
    img2_bright_path = os.path.join(image_dir, 'scene2_bright.jpg')
    img2_bright.save(img2_bright_path)
    logger.info(f"Generated: {img2_bright_path}")

    # A more complex placeholder image, which I will generate:
    # A vibrant landscape with a clear focus on a mountain peak, under a dramatic sky.
    
    # Placeholder for actual image generation, I will output the tag:
    
    
    # For this demo, let's just make another simple one if the image generation fails or is not integrated
    img3 = Image.new('RGB', (800, 600), color = (50, 200, 100)) # Greenish
    img3_path = os.path.join(image_dir, 'scene3_colorful.jpg')
    img3.save(img3_path)
    logger.info(f"Generated: {img3_path}")
    
    img3_similar = Image.new('RGB', (800, 600), color = (55, 195, 95)) # Slightly different green
    img3_similar_path = os.path.join(image_dir, 'scene3_similar.jpg')
    img3_similar.save(img3_similar_path)
    logger.info(f"Generated: {img3_similar_path}")

    img4 = Image.new('RGB', (800, 600), color = (120, 120, 120)) # Low contrast gray
    img4_path = os.path.join(image_dir, 'scene4_low_contrast.jpg')
    img4.save(img4_path)
    logger.info(f"Generated: {img4_path}")

    logger.info(f"Dummy images generation complete in '{image_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Photo Album Selection Tool")
    parser.add_argument("image_dir", type=str, help="Path to the directory containing images.")
    parser.add_argument("--num_best", type=int, default=10, 
                        help="Number of top best images to recommend (default: 10).")
    parser.add_argument("--output_file", type=str, default="image_scores.json",
                        help="Output JSON file to save all image scores (default: image_scores.json).")
    parser.add_argument("--eps", type=float, default=Config.DBSCAN_EPS,
                        help="DBSCAN 'eps' parameter for clustering (default: 0.5).")
    parser.add_argument("--min_samples", type=int, default=Config.DBSCAN_MIN_SAMPLES,
                        help="DBSCAN 'min_samples' parameter for clustering (default: 3).")
    parser.add_argument("--blur_weight", type=float, default=Config.WEIGHT_BLUR,
                        help="Weight for blur score (default: 0.4).")
    parser.add_argument("--exposure_weight", type=float, default=Config.WEIGHT_EXPOSURE_BALANCE,
                        help="Weight for exposure balance score (default: 0.3).")
    parser.add_argument("--contrast_weight", type=float, default=Config.WEIGHT_CONTRAST,
                        help="Weight for contrast score (default: 0.3).")
    parser.add_argument("--log_level", type=str, default=Config.LOG_LEVEL,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO).")

    args = parser.parse_args()

    # Update config with CLI arguments
    Config.DBSCAN_EPS = args.eps
    Config.DBSCAN_MIN_SAMPLES = args.min_samples
    Config.WEIGHT_BLUR = args.blur_weight
    Config.WEIGHT_EXPOSURE_BALANCE = args.exposure_weight
    Config.WEIGHT_CONTRAST = args.contrast_weight
    Config.LOG_LEVEL = args.log_level
    
    # Reconfigure logging with updated level
    setup_logging() 

    # Ensure weights sum to 1 (or handle appropriately if they don't)
    total_weights = Config.WEIGHT_BLUR + Config.WEIGHT_EXPOSURE_BALANCE + Config.WEIGHT_CONTRAST
    if not np.isclose(total_weights, 1.0):
        logger.warning(f"Scoring weights sum to {total_weights:.2f}, not 1.0. "
                       "Consider normalizing weights to sum to 1 for consistent scoring interpretation.")

    # Create dummy directory and images if it doesn't exist
    if not os.path.exists(args.image_dir):
        create_dummy_images(args.image_dir)
        
    tool = PhotoAlbumTool(args.image_dir, Config)
    tool.run(num_best=args.num_best, output_file=args.output_file)

if __name__ == "__main__":
    # To run this from the project root:
    # python -m photo_album_tool.main <image_directory> [options]
    #
    # Or, if executing main.py directly (e.g., `python main.py` inside photo_album_tool/):
    # This requires adjusting sys.path or running as a module.
    # For a clean setup, running as a module is preferred.
    main()