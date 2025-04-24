import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO  # Add YOLOv8
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load SAM
sam = sam_model_registry["vit_b"](
    checkpoint="segmentation/sam_vit_b_01ec64.pth")
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

# Load YOLOv8 model (trained on ring, earring, dress)
# Replace with your trained model path
yolo_model = YOLO("runs/detect/train/weights/best.pt")

# Load image
img_path = "query.jpg"
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# Run YOLO detection
results = yolo_model(img_path)[0]

# Extract ring box and use its center as SAM input
ring_boxes = [box for box in results.boxes.data.cpu().numpy()
              if int(box[5]) == 0]  # class 0 = 'ring'

if not ring_boxes:
    print("❌ No ring detected.")
    exit()

x1, y1, x2, y2, conf, cls = ring_boxes[0]
cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

input_point = np.array([[cx, cy]])
input_label = np.array([1])

# Predict with SAM
masks, scores, _ = predictor.predict(
    point_coords=input_point, point_labels=input_label, multimask_output=True)

# Show masks
for i, mask in enumerate(masks):
    plt.subplot(1, 3, i + 1)
    plt.imshow(image_rgb)
    plt.imshow(mask, alpha=0.5)
    plt.title(f"Mask {i+1}, Score {scores[i]:.2f}")
    plt.axis('off')

plt.suptitle("Auto-SAM on YOLO Detected Ring 💍")
plt.show()
