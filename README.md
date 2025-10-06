# Photo Album Selection Tool

## Overview

The Photo Album Selection Tool is a Python-based application designed to help users curate their photo collections by automatically identifying and scoring images. It addresses the common challenge of having hundreds of similar photos and aims to select the "best" ones for a photo album based on various quality and similarity metrics.

The tool leverages image processing techniques, deep learning embeddings, and clustering algorithms to:
1.  **Extract Features:** Analyze images for quality aspects like blur, exposure, and contrast, and semantic content using pre-trained deep learning models.
2.  **Identify Similarities:** Group semantically similar images using clustering to avoid redundancy.
3.  **Score Images:** Calculate a comprehensive quality score for each image.
4.  **Recommend Best Photos:** Provide a ranked list of the best images, ensuring diversity by selecting representatives from different clusters.

## Features

*   **Efficient Processing:** Designed to handle large collections of photos.
*   **Quality Assessment:** Evaluates images based on blur, exposure balance, and contrast.
*   **Semantic Similarity:** Uses Deep Learning (VGG16 embeddings) to understand image content and group similar scenes/objects.
*   **Clustering:** Employs DBSCAN to identify natural groupings of similar images, preventing over-representation of a single scene.
*   **Configurable Scoring:** Customizable weights for different quality metrics.
*   **Clear Output:** Generates a ranked list of photos and a JSON file with detailed scores for all processed images.
*   **Modular Design:** Organized into a clean package structure for easy maintenance and extension.

