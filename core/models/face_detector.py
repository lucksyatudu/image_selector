import os
import cv2
import urllib.request


def try_download(url_list, output_path):
    """Try downloading from multiple URLs until one works."""
    for url in url_list:
        try:
            print(f"[Downloading] {url}")
            urllib.request.urlretrieve(url, output_path)
            print(f"[OK] Saved to {output_path}")
            return True
        except Exception as e:
            print(f"[Failed] {url} → {e}")
    return False


class FaceDetectorDNN:
    """
    Robust OpenCV DNN face detector with multi-URL fallback.
    Prevents 404 errors by trying multiple known model locations.
    """

    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.proto_path = os.path.join(model_dir, "deploy.prototxt")
        self.model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        self._ensure_models()

        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, self.model_path)

    def _ensure_models(self):
        """Download models from multiple possible GitHub URLs."""

        # ---- PROTOTXT ----
        proto_urls = [
            # OpenCV 4.x
            "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
            # OpenCV master
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            # OpenCV 3.x archive
            "https://raw.githubusercontent.com/opencv/opencv/3.4/samples/dnn/face_detector/deploy.prototxt",
        ]

        if not os.path.exists(self.proto_path):
            ok = try_download(proto_urls, self.proto_path)
            if not ok:
                raise RuntimeError("❌ Cannot download deploy.prototxt — all URLs failed.")

        # ---- CAFFEMODEL ----
        model_urls = [
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        ]

        if not os.path.exists(self.model_path):
            ok = try_download(model_urls, self.model_path)
            if not ok:
                raise RuntimeError("❌ Cannot download caffemodel — all URLs failed.")

    def detect_faces(self, image):
        h, w = image.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf < 0.3:
                continue

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)

            faces.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": float(conf)
            })

        return faces
