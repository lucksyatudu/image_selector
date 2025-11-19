# Photo Album Selection Tool

## Overview

The Photo Album Selection Tool is a Python-based application designed to help users curate their photo collections by automatically identifying and scoring images. It addresses the common challenge of having hundreds of similar photos and aims to select the "best" ones for a photo album based on various quality and similarity metrics.

The tool leverages image processing techniques, deep learning embeddings, and clustering algorithms to:<br>
**Extract Features:** Analyze images for quality aspects like blur, exposure, and contrast, and semantic content using pre-trained deep learning models.<br>
**Identify Similarities:** Group semantically similar images using clustering to avoid redundancy.<br>
**Score Images:** Calculate a comprehensive quality score for each image.<br>
**Recommend Best Photos:** Provide a ranked list of the best images, ensuring diversity by selecting representatives from different clusters.

Each image is evaluated using a weighted combination of quality, aesthetic, and content-based features.Below is the list of all scoring features and what each represents.
1. **Blur Score:** Measures image sharpness. Less blur indicates higher visual clarity and better quality.
2. **Exposure Balance:** Evaluates whether an image is evenly exposed. Images that are too bright or too dark receive lower scores.
3. **Contrast Quality:** Assesses tonal separation between light and dark elements. Good contrast generally improves visual appeal and subject clarity.
4. **Orientation Accuracy:** Checks whether the image is rotated or tilted. Correctly oriented photos score higher.
5. **Facial Emotion Quality:** Estimates how positive or natural the facial expressions are (when faces are detected). Smiling or pleasant expressions are favored.
6. **Face Presence:** Rewards images that contain at least one detectable face when portraits or people-focused scenes are expected.
7. **Face Count:** Measures how many faces are present. Too few or too many may reduce quality depending on context; controlled scoring ensures balanced group shots.
8. **Face Centeredness:** Scores how close the main face is to the center of the image. Centered subjects create stronger compositions in typical portrait or group shots.
9. **Eyes-Open Score:** Penalizes images where the detected faces have closed eyes. Images with open eyes are preferred for portraits and group photos.
10. **Brightness Uniformity:** Evaluates illumination consistency across the image. Images with harsh lighting differences or uneven shadows score lower.
11. **Sharp Edge Detail:** Measures clarity through edge detection. Well-defined edges typically indicate better focus and sharpness.
12. **Color Saturation:** Scores how vibrant the colors are without being overly saturated. Balanced saturation contributes to visual appeal.
13. **Background Complexity:** Estimates whether the background is cluttered or simple. Cleaner backgrounds help highlight the main subject.
14. **Face Size Ratio:** Measures how large the primary face is relative to the image frame. Too small implies the subject is far away; ideal portraits have moderately sized facial regions.
15. **Exposure Highlights:** Detects over-exposed regions that cause blown-out details. More highlights mean lower score because important details are lost.
16. **Dynamic Range Loss:** Assesses loss of details in shadows or highlights. Higher dynamic range (details visible across brightness spectrum) receives a better score.

## Installation

Create a virtual environment and install dependencies by running env_setup.sh

## Running Tool

#### 1. Standalone Run

- Create folder for your images.
- Update the details in standalone_run_config.json.
- Run standalone_run.py.
- The scores of images will be provided inside runs folder.
- Basis your inputs to configs, the images folder can be cleaned for best images basis the scores.