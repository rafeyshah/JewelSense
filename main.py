import os
import cv2
import json
from datetime import datetime

from models.hand_tracker import HandTracker
from detectors.ring_detector import RingDetector

# === Paths ===
INPUT_DIR = "data/test/images"
OUTPUT_DIR = "output/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Init Models ===
tracker = HandTracker(max_num_hands=1)
ring_detector = RingDetector(model_path="runs/detect/train/weights/best.pt")  # update path if needed

# === Run on all test images ===
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(image_path)

        # 1. Detect hand landmarks
        landmarks = tracker.process_image(image)
        image = tracker.draw_landmarks(image, landmarks)

        # 2. Detect rings
        detections = ring_detector.detect_rings(image)

        # 3. Draw ring detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"Ring {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # # 4. Save output
        # output_path = os.path.join(OUTPUT_DIR, filename)
        # cv2.imwrite(output_path, image)
        # print(f"[✓] Processed: {filename} → {output_path}")
        
        # 4. Save detection data to JSON
        json_data = {
            "image_name": filename,
            "timestamp": datetime.now().isoformat(),
            "hand_landmarks": [],
            "ring_detections": []
        }

        for hand in landmarks:
            json_data["hand_landmarks"].append([
                {"x": x, "y": y} for (x, y) in hand
            ])

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            json_data["ring_detections"].append({
                "bbox": [x1, y1, x2, y2],
                "confidence": det["confidence"]
            })

        # Write JSON
        json_path = os.path.join("output/json", filename.rsplit('.', 1)[0] + ".json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)

        # Save image
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, image)
        print(f"[✓] Processed: {filename} → Image + JSON saved")

