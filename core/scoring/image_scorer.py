import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any
import cv2

from core.config import Config
from core.utils.logger import logger
from core.models.face_detector import FaceDetectorDNN
from core.models.emotion_detector import SimpleEmotionClassifier


class ImageScorer:
    """
    Complete image scorer with:
        - Blur / Contrast / Exposure
        - Orientation check (front vs back)
        - Emotion/smile scoring
    """

    def __init__(self, config: Config):
        self.config = config
        self.quality_scalers = {}
        self.is_fitted = False

        # Use real working model implementations (not placeholders)
        self.face_detector = FaceDetectorDNN()
        self.emotion_model = SimpleEmotionClassifier(self.face_detector)

    # -------------------------
    # SCALER FITTING
    # -------------------------
    def fit_scalers(self, all_quality_features: List[Dict[str, Any]]):
        if not all_quality_features:
            logger.warning("No quality features to fit scalers.")
            return

        feature_names = all_quality_features[0].keys()
        for name in feature_names:
            if name in ['blur_variance', 'contrast_std']:
                vals = np.array([f[name] for f in all_quality_features]).reshape(-1, 1)
                scaler = MinMaxScaler()
                scaler.fit(vals)
                self.quality_scalers[name] = scaler

        self.is_fitted = True
        logger.info("Quality scalers fitted.")

    # -------------------------
    # QUALITY SCORING
    # -------------------------
    def _score_blur(self, blur_var):
        if 'blur_variance' not in self.quality_scalers:
            return 0.5
        return float(self.quality_scalers['blur_variance'].transform([[blur_var]])[0][0])

    def _score_dynamic_range_loss(self,clipped_pct):
        # 0% clipping → perfect (1.0)
        # >20% clipping → bad (0.0)
        return max(0.0, 1.0 - clipped_pct / 0.20)

    def _score_contrast(self, cstd):
        if 'contrast_std' not in self.quality_scalers:
            return 0.5
        return float(self.quality_scalers['contrast_std'].transform([[cstd]])[0][0])

    def _score_exposure_balance(self, mean):
        max_dev = max(self.config.IDEAL_MEAN_INTENSITY, 255 - self.config.IDEAL_MEAN_INTENSITY)
        dev = abs(mean - self.config.IDEAL_MEAN_INTENSITY)
        return max(0.0, 1.0 - dev / max_dev)

    # -------------------------
    # ORIENTATION SCORING
    # -------------------------
    def _detect_face_orientation(self, image):
        faces = self.face_detector.detect_faces(image)
        if not faces:
            return 0.0  # likely back facing

        # pick highest confidence face
        f = max(faces, key=lambda d: d["confidence"])
        conf = f["confidence"]

        if conf > 0.8:
            return 1.0
        if conf > 0.4:
            return 0.5
        return 0.0

    # -------------------------
    # EMOTION SCORING
    # -------------------------
    def _score_emotion(self, image):
        result = self.emotion_model.predict(image)
        label = result["dominant_emotion"]

        if label in ("happy", "smile"):
            return 1.0
        if label in ("neutral", "calm"):
            return 0.7
        if label in ("sad", "fear", "disgust"):
            return 0.3
        if label in ("angry",):
            return 0.1
        return 0.5

    def _score_face_present(self, faces):
        return 1.0 if len(faces) > 0 else 0.0

    def _score_faces_count(self, faces):
        if len(faces) == 1: return 1.0
        if len(faces) == 2: return 0.7
        if len(faces) > 2: return 0.4
        return 0.0
    
    def _extract_box(self, face):
        # Case 1: MTCNN / RetinaFace dictionary
        if isinstance(face, dict) and "box" in face:
            return face["box"]

        # Case 2: face_recognition returns (top, right, bottom, left)
        if isinstance(face, (list, tuple)) and len(face) == 4:
            top, right, bottom, left = face
            w = right - left
            h = bottom - top
            return (left, top, w, h)

        # Unknown format → return None
        return None


    def _score_face_centered(self, image, faces):
        if not faces: 
            return 0.0

        box = self._extract_box(faces[0])
        if box is None:
            return 0.0

        x, y, fw, fh = box
        h, w = image.shape[:2]

        face_center_x = x + fw / 2
        frame_center_x = w / 2

        dist = abs(face_center_x - frame_center_x) / (w / 2)
        return max(0, 1 - dist)

    def _score_eyes_open(self, faces):
        """Placeholder: real impl would use landmarks / eye-openness classifier."""
        if not faces: return 0.0
        conf = faces[0]["confidence"]
        return min(1.0, conf)

    def _score_brightness_uniformity(self, image_gray):
        return 1 - (image_gray.std() / 128)

    def _score_sharp_edges(self, image_gray):
        edges = cv2.Canny(image_gray, 100, 200)
        ratio = edges.sum() / (255.0 * edges.size)
        return min(1.0, ratio * 5)

    def _score_saturation(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].mean()
        return sat / 255.0

    def _background_complexity(self, image_gray):
        edges = cv2.Canny(image_gray, 30, 100)
        complexity = edges.mean() / 255.0
        return 1 - complexity

    def _score_face_size_ratio(self, image, faces):
        if not faces:
            return 0.0

        box = self._extract_box(faces[0])
        if box is None:
            return 0.0

        x, y, w, h = box
        img_area = image.shape[0] * image.shape[1]

        return min(1.0, (w * h) / img_area * 5)

    def _score_exposure_highlights(self, img_gray):
        overexposed = (img_gray > 240).mean()
        underexposed = (img_gray < 15).mean()
        score = 1 - (overexposed + underexposed)
        return max(0, min(score, 1))

    # -------------------------
    # FINAL SCORE
    # -------------------------
    def calculate_single_image_score(self, image_data: Dict[str, Any]) -> float:
        if not self.is_fitted:
            logger.warning("Scalers not fitted.")
            return 0

        quality = image_data["quality"]
        # FIX: Load image from path
        image_path = image_data["path"]
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return 0.0
        faces = self.face_detector.detect_faces(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_s = self._score_blur(quality["blur_variance"])
        exp_s = self._score_exposure_balance(quality["mean_intensity"])
        con_s = self._score_contrast(quality["contrast_std"])
        orient_s = self._detect_face_orientation(image)
        emotion_s = self._score_emotion(image)
        face_present_s = self._score_face_present(faces)
        face_count_s = self._score_faces_count(faces)
        face_centered_s = self._score_face_centered(image, faces)
        eyes_s = self._score_eyes_open(faces)
        brightness_uniformity_s = self._score_brightness_uniformity(gray)
        sharpness_edges_s = self._score_sharp_edges(gray)
        saturation_s = self._score_saturation(image)
        background_complexity_s = self._background_complexity(gray)
        face_size_ratio_s = self._score_face_size_ratio(image, faces)
        exposure_highlights_s = self._score_exposure_highlights(gray)
        dynamic_range_loss_s = self._score_dynamic_range_loss(quality['clipped_percent'])


        if quality["blur_variance"] < self.config.MIN_BLUR_VARIANCE:
            blur_s *= 0.1

        score = (
            self.config.WEIGHT_BLUR * blur_s +
            self.config.WEIGHT_EXPOSURE_BALANCE * exp_s +
            self.config.WEIGHT_CONTRAST * con_s +
            self.config.WEIGHT_ORIENTATION * orient_s +
            self.config.WEIGHT_EMOTION * emotion_s +
            self.config.WEIGHT_FACE_PRESENT * face_present_s +
            self.config.WEIGHT_FACES_COUNT * face_count_s +
            self.config.WEIGHT_FACE_CENTERED * face_centered_s +
            self.config.WEIGHT_EYES_OPEN * eyes_s +
            self.config.WEIGHT_BRIGHTNESS_UNIFORMITY * brightness_uniformity_s +
            self.config.WEIGHT_SHARP_EDGES * sharpness_edges_s +
            self.config.WEIGHT_SATURATION * saturation_s +
            self.config.WEIGHT_BACKGROUND_COMPLEXITY * background_complexity_s +
            self.config.WEIGHT_FACE_SIZE_RATIO * face_size_ratio_s +
            self.config.WEIGHT_EXPOSURE_HIGHLIGHTS * exposure_highlights_s +
            self.config.WEIGHT_DYNAMIC_RANGE_LOSS * dynamic_range_loss_s
        )
        return float(round(score, 4))
