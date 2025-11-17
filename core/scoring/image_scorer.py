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

        blur_s = self._score_blur(quality["blur_variance"])
        exp_s = self._score_exposure_balance(quality["mean_intensity"])
        con_s = self._score_contrast(quality["contrast_std"])
        orient_s = self._detect_face_orientation(image)
        emotion_s = self._score_emotion(image)

        if quality["blur_variance"] < self.config.MIN_BLUR_VARIANCE:
            blur_s *= 0.1

        score = (
            self.config.WEIGHT_BLUR * blur_s +
            self.config.WEIGHT_EXPOSURE_BALANCE * exp_s +
            self.config.WEIGHT_CONTRAST * con_s +
            self.config.WEIGHT_ORIENTATION * orient_s +
            self.config.WEIGHT_EMOTION * emotion_s
        )
        return float(round(score, 4))
