from ultralytics import YOLO
import cv2

class RingDetector:
    def __init__(self, model_path="runs/detect/train/weights/best.pt"):
        self.model = YOLO(model_path)

    def detect_rings(self, image_bgr):
        results = self.model.predict(source=image_bgr, save=False, conf=0.4, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })

        return detections
