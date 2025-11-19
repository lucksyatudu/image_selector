import os
import json
from PIL import Image, ImageFilter
import time
from datetime import datetime

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

class StandaloneImageProcessor:
    def __init__(self, image_dir: str, config: Config):
        self.image_dir = image_dir
        self.config = config
        self.image_data: List[Dict[str, Any]] = []

        self.dl_extractor = DeepLearningFeatureExtractor()
        self.quality_extractor = QualityFeatureExtractor()
        self.phash_extractor = PerceptualHashExtractor()
        self.similarity_analyzer = SimilarityAnalyzer()
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
            # Sort each cluster by score (high → low)
            sorted_cluster = sorted(img_list, key=lambda x: x["final_score"], reverse=True)

            for rank, img in enumerate(sorted_cluster, start=1):
                img['final_score'] = img['final_score']/rank  # Penalize by rank within cluster
                selected_images.append(img)
        
        selected_images.extend(noise_images)
        selected_images.sort(key=lambda x: x['final_score'], reverse=True)
        
        logger.info(f"Selected {len(selected_images)} unique candidates before final trimming.")
        return selected_images[:num_best]

if __name__ == "__main__":
    # Ensure runs folder is present
    os.makedirs("runs", exist_ok=True)

    # Ensure logging is set up for the example
    setup_logging()
    
    # Get run configuration
    with open("standalone_run_config.json", 'r') as f:
        run_config = json.load(f)
    IMAGE_DIR = run_config.get("image_directory", None)
    if IMAGE_DIR is None or not os.path.isdir(IMAGE_DIR):
        logger.error("Please provide a valid 'image_directory' in 'standalone_run_config.json'.")
        exit(1)
        
    OUTPUT_FILE = "runs/"+datetime.strftime(datetime.now(),'%Y%m%d-%H%M')+"-run_scores.json"
    
    start_time = time.time()
    logger.info(f"Starting photo album processing for '{IMAGE_DIR}'...")

    processor = StandaloneImageProcessor(IMAGE_DIR, Config)
    processor.process_all_images()
    
    num_to_recommend = run_config.get("selection",None)
    if num_to_recommend is None or not isinstance(num_to_recommend, int):
        logger.error("Please provide a valid integer 'selection' in 'standalone_run_config.json'.")
        exit(1)

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

    # Clean up image directory after processing
    if run_config.get("clean_directory_after_processing", False):
        paths_to_retain = [top['path'] for top in top_images]
        for file in os.listdir(IMAGE_DIR):
            if not file.is_file():
                continue 
            full_path = os.path.join(IMAGE_DIR, file)
            if full_path not in paths_to_retain:
                try:
                    os.remove(full_path)
                    logger.info(f"Removed file: {full_path}")
                except Exception as e:
                    logger.error(f"Could not remove file '{full_path}': {e}")
    