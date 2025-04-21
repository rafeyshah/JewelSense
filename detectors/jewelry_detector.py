from ultralytics import YOLO

class JewelryDetector:
    def __init__(self, model_path, class_names):
        self.model = YOLO(model_path)
        self.class_names = class_names

    def detect(self, image_bgr, conf_thresh=0.4):
        results = self.model.predict(source=image_bgr, conf=conf_thresh, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls_id = int(box.cls)
                cls_name = self.class_names[cls_id]
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name
                })

        return detections
