import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any

from core.config import Config
from core.utils.logger import logger

class SimilarityAnalyzer:
    """
    Analyzes image similarity using Deep Learning embeddings and clustering.
    Identifies and groups similar images.
    """
    def __init__(self, config: Config):
        self.config = config

    def cluster_images(self, image_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clusters images based on their Deep Learning embeddings using DBSCAN.
        Updates the 'cluster_id' for each image in the provided list.
        """
        embeddings_with_indices = []
        for i, data in enumerate(image_data):
            if data.get('dl_embedding') is not None:
                embeddings_with_indices.append((data['dl_embedding'], i))

        if not embeddings_with_indices:
            logger.warning("No Deep Learning embeddings available for clustering.")
            return image_data

        embeddings = np.array([e[0] for e in embeddings_with_indices])
        original_indices = [e[1] for e in embeddings_with_indices]

        logger.info(f"Clustering {len(embeddings)} images with DBSCAN...")
        
        # Scale embeddings before clustering (important for distance-based algorithms)
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)

        dbscan = DBSCAN(eps=self.config.DBSCAN_EPS, 
                        min_samples=self.config.DBSCAN_MIN_SAMPLES, 
                        metric='euclidean',
                        n_jobs=-1) # Use all available CPU cores for efficiency
        
        clusters = dbscan.fit_predict(scaled_embeddings)

        for i, cluster_id in enumerate(clusters):
            image_data[original_indices[i]]['cluster_id'] = int(cluster_id) # Ensure int for JSON serialization
        
        num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        logger.info(f"Clustering complete. Found {num_clusters} meaningful clusters.")
        
        return image_data

    def identify_near_duplicates_phash(self, image_data: List[Dict[str, Any]], hash_threshold: int = 5) -> Dict[str, List[str]]:
        """
        Identifies groups of images that are near-duplicates using perceptual hashes.
        Returns a dictionary where keys are a 'representative' hash and values are
        a list of paths to similar images.
        Note: This is a fast pre-clustering step, distinct from DL embedding clustering.
        """
        hash_groups = {}
        for img_data in image_data:
            phash = img_data['phash']
            found_group = False
            for representative_hash in hash_groups:
                # Hamming distance comparison for perceptual hashes
                distance = sum(c1 != c2 for c1, c2 in zip(phash, representative_hash))
                if distance <= hash_threshold:
                    hash_groups[representative_hash].append(img_data['path'])
                    found_group = True
                    break
            if not found_group:
                hash_groups[phash] = [img_data['path']]
        
        # Filter out groups that only have one image (i.e., not duplicates)
        return {k: v for k, v in hash_groups.items() if len(v) > 1}