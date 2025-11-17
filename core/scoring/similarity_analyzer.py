import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from typing import List, Dict, Any

from core.config import Config
from core.utils.logger import logger


class SimilarityAnalyzer:
    """
    Analyzes image similarity using deep-learning embeddings (DBSCAN) 
    and perceptual-hash based near-duplicate detection.
    """

    def __init__(self, config: Config):
        self.config = config

    # ------------------------------------------------------
    # 1. Deep Learning Embedding Clustering (Improved DBSCAN)
    # ------------------------------------------------------
    def cluster_images(self, image_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        embeddings_with_idx = [
            (data["dl_embedding"], i)
            for i, data in enumerate(image_data)
            if data.get("dl_embedding") is not None
        ]

        if not embeddings_with_idx:
            logger.warning("No Deep Learning embeddings available.")
            return image_data

        embeddings = np.array([e[0] for e in embeddings_with_idx])
        idx_list = [e[1] for e in embeddings_with_idx]

        logger.info(f"Running DBSCAN on {len(embeddings)} embeddings...")

        # Step 1: Normalize (L2)
        norm_embeddings = normalize(embeddings, norm="l2")

        # Step 2: PCA for dimensionality reduction
        max_components = min(50, norm_embeddings.shape[1], norm_embeddings.shape[0])

        if max_components < 2:
            logger.warning(
                f"Too few samples/features for PCA (max_components={max_components}). "
                "Skipping PCA and using normalized embeddings directly."
            )
            reduced = norm_embeddings
        else:
            pca = PCA(n_components=max_components)
            reduced = pca.fit_transform(norm_embeddings)
        # Optional debugging: inspect distance distribution
        dists = pairwise_distances(reduced, metric="cosine")
        logger.info(f"Distance percentiles: {np.percentile(dists, [5, 25, 50, 75, 95])}")

        # Step 3: DBSCAN using cosine metric
        dbscan = DBSCAN(
            eps=self.config.DBSCAN_EPS,
            min_samples=self.config.DBSCAN_MIN_SAMPLES,
            metric="cosine",
            n_jobs=-1
        )
        labels = dbscan.fit_predict(reduced)

        # Store cluster IDs
        for j, cluster_id in enumerate(labels):
            image_data[idx_list[j]]["cluster_id"] = int(cluster_id)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"DBSCAN complete. Found {num_clusters} clusters.")

        return image_data

    # ------------------------------------------------------
    # 2. Perceptual Hash Near-Duplicate Detection (Improved)
    # ------------------------------------------------------
    def identify_near_duplicates_phash(
        self, image_data: List[Dict[str, Any]], hash_threshold: int = 5
    ) -> Dict[int, List[str]]:
        """
        Groups images where pHash Hamming distance <= threshold.
        Uses integer hash comparison for correctness.
        Uses union-find for efficient grouping.
        Returns: {group_id: [image_paths]}
        """

        # Convert pHashes to integer form
        hashes = []
        for img in image_data:
            if img.get("phash"):
                try:
                    hashes.append(int(img["phash"], 16))
                except:
                    logger.error(f"Invalid pHash format: {img['phash']}")
                    hashes.append(None)
            else:
                hashes.append(None)

        n = len(image_data)

        # Union-Find setup
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        # Compare all pHashes (only valid ones)
        for i in range(n):
            if hashes[i] is None:
                continue
            for j in range(i + 1, n):
                if hashes[j] is None:
                    continue

                # Hamming distance via XOR
                dist = (hashes[i] ^ hashes[j]).bit_count()

                if dist <= hash_threshold:
                    union(i, j)

        # Build groups
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(image_data[i]["path"])

        # Filter groups with only one image
        return {gid: paths for gid, paths in groups.items() if len(paths) > 1}
