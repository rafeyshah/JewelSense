import os
import cv2
import json
from datetime import datetime
from detectors.jewelry_detector import JewelryDetector
from trackers.jewelry_tracker import JewelryTracker

# === Config ===
MODEL_PATH = "runs/detect/train/weights/best.pt"
CLASS_NAMES = ['ring', 'earring', 'dress']


# INPUT_DIR = "data/test/iamges"
INPUT_DIR = "data/"
OUTPUT_IMG_DIR = "output/results"
OUTPUT_JSON_DIR = "output/json"
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# === Init models ===
detector = JewelryDetector(model_path=MODEL_PATH, class_names=CLASS_NAMES)
tracker = JewelryTracker()

# === Run detection + tracking ===
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(image_path)

        detections = detector.detect(image)
        tracked = tracker.update(detections, image)

        json_data = {
            "image": filename,
            "timestamp": datetime.now().isoformat(),
            "detections": []
        }

        for obj in tracked:
            x1, y1, x2, y2 = obj['bbox']
            cls_name = obj['class_name']
            track_id = obj['track_id']
            json_data["detections"].append({
                "class_name": cls_name,
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2]
            })

            # Draw
            label = f"{cls_name} ID {track_id}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 255), 2)

        # Save image + JSON
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, filename), image)
        with open(os.path.join(OUTPUT_JSON_DIR, filename.rsplit('.', 1)[0] + '.json'), 'w') as f:
            json.dump(json_data, f, indent=4)

        print(f"[✓] Processed: {filename}")
