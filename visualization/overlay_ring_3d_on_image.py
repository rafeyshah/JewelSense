import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fitting.hand_pose_utils import extract_hand_landmarks


# Load image and landmarks
img_path = "query.jpg"
landmarks, _ = extract_hand_landmarks(img_path)

if landmarks is None:
    exit()

# Finger base + middle for direction
p1 = np.array(landmarks[13])
p2 = np.array(landmarks[14])
direction = p2 - p1
direction /= np.linalg.norm(direction)

# Create torus (ring shape)
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 40)
theta, phi = np.meshgrid(theta, phi)

R, r = 5, 1.5  # Torus params
X = (R + r * np.cos(phi)) * np.cos(theta)
Y = (R + r * np.cos(phi)) * np.sin(theta)
Z = r * np.sin(phi)

# Shift to finger joint
X += p1[0]
Y += p1[1]
Z += p1[2]

# Plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Background image (just for dimensions)
image = Image.open(img_path)
img_w, img_h = image.size

# Set proper viewing window
ax.set_xlim(p1[0] - 50, p1[0] + 50)
ax.set_ylim(p1[1] + 50, p1[1] - 50)  # Inverted Y for image coordinates
ax.set_zlim(p1[2] - 20, p1[2] + 20)

ax.plot_surface(X, Y, Z, color='gold', alpha=0.85, edgecolor='k')

ax.axis('off')
plt.tight_layout()
plt.title("💍 3D Ring Overlay on Finger")
plt.savefig("output/masks/ring_overlay_result.png", dpi=300)
plt.show()
