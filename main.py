import os
import cv2
from models.hand_tracker import HandTracker

# === Paths ===
INPUT_DIR = "data/test/images"
OUTPUT_DIR = "output/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Init Model ===
tracker = HandTracker(max_num_hands=1)

# === Run on all test images ===
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(image_path)

        # Detect landmarks
        landmarks = tracker.process_image(image)
        annotated_image = tracker.draw_landmarks(image.copy(), landmarks)

        # Save output
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, annotated_image)
        print(f"Processed: {filename} → Saved to {output_path}")
