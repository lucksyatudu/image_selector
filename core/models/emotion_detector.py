import cv2
import numpy as np

from core.models.face_detector import FaceDetectorDNN


class SimpleEmotionClassifier:
    """
    Lightweight emotion classifier. Not perfect but reliable enough for ranking.
    Dominant emotions: happy, neutral, sad, angry
    """

    def __init__(self, face_detector=None):
        # Use same detector for cropping faces
        self.face_detector = face_detector or FaceDetectorDNN()

    def predict(self, image: np.ndarray):
        faces = self.face_detector.detect_faces(image)

        if not faces:
            return {"dominant_emotion": "neutral"}

        # Pick highest-confidence face
        f = max(faces, key=lambda x: x["confidence"])
        x1, y1, x2, y2 = f["x1"], f["y1"], f["x2"], f["y2"]

        face = image[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
        if face.size == 0:
            return {"dominant_emotion": "neutral"}

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        # Heuristic smile detector:
        # Mouth region’s pixel intensity difference → smile strength
        h = gray.shape[0]
        mouth_region = gray[int(h * 0.6):h, :]

        smile_score = mouth_region.mean() - gray.mean()

        if smile_score > 15:
            emotion = "happy"
        elif smile_score > 5:
            emotion = "neutral"
        elif smile_score > -5:
            emotion = "sad"
        else:
            emotion = "angry"

        return {"dominant_emotion": emotion}
