import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from typing import List, Dict, Any
import hdbscan

from core.utils.logger import logger


class SimilarityAnalyzer:

    def cluster_images(self, image_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clusters deep-learning embeddings using HDBSCAN 
        for fine-grained similarity grouping.
        """

        # ----------------------------------------------------
        # 1. Collect Embeddings
        # ----------------------------------------------------
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

        logger.info(f"Running HDBSCAN on {len(embeddings)} embeddings...")

        # ----------------------------------------------------
        # 2. Normalize Embeddings (L2)
        # ----------------------------------------------------
        norm_embeddings = normalize(embeddings, norm="l2")

        # ----------------------------------------------------
        # 3. PCA Dimensionality Reduction
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # 4. Optional Debug: Distance Percentiles
        # ----------------------------------------------------
        dists = pairwise_distances(reduced, metric="cosine")
        logger.info(f"Distance percentiles: {np.percentile(dists, [5, 25, 50, 75, 95])}")

        # ----------------------------------------------------
        # 5. HDBSCAN Clustering
        # ----------------------------------------------------
        # Very fine-grained grouping settings:
        clusterer = hdbscan.HDBSCAN(
            metric='euclidean',         # works well after PCA
            min_cluster_size=2,         # allow tiny clusters (required for 600+ clusters)
            min_samples=1,              # very sensitive to local variations
            cluster_selection_method='leaf',
            cluster_selection_epsilon=0.05,   # controls cluster splitting (tuneable)
            allow_single_cluster=False,
            core_dist_n_jobs=-1,
        )

        labels = clusterer.fit_predict(reduced)

        # ----------------------------------------------------
        # 6. Store Clusters Back in image_data
        # ----------------------------------------------------
        for j, cluster_id in enumerate(labels):
            image_data[idx_list[j]]["cluster_id"] = int(cluster_id)

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = np.sum(labels == -1)

        logger.info(f"HDBSCAN complete. Found {num_clusters} clusters.")
        logger.info(f"Noise points (unclustered): {num_noise}")

        return image_data
