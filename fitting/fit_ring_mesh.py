import open3d as o3d
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fitting.hand_pose_utils import extract_hand_landmarks

# Load hand landmarks
landmarks, _ = extract_hand_landmarks("query.jpg")
if not landmarks:
    exit()

# Use two points on ring finger to get direction vector
p1 = np.array(landmarks[13])  # ring base
p2 = np.array(landmarks[14])  # ring middle

direction = p2 - p1
direction = direction / np.linalg.norm(direction)  # normalize

# Create torus (ring)
ring = o3d.geometry.TriangleMesh.create_torus(torus_radius=5, tube_radius=1.5)
ring.paint_uniform_color([0.9, 0.7, 0.3])

# Align ring's Z-axis with finger direction
ring.rotate(ring.get_rotation_matrix_from_xyz((0, np.pi/2, 0)))  # face camera
rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
    np.cross([0, 0, 1], direction))
ring.rotate(rotation_matrix, center=(0, 0, 0))

# Translate to finger joint
ring.translate(p1)

# Show anchor point (optional)
dot = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
dot.translate(p1)
dot.paint_uniform_color([0.2, 0.8, 1.0])

# Visualize
o3d.visualization.draw_geometries([ring, dot])

# Merge objects for export
combined_mesh = ring + dot  # optional: just export ring

# Save as OBJ
o3d.io.write_triangle_mesh("output/masks/fitted_ring.obj", combined_mesh)
print("✅ Exported fitted ring as OBJ to output/masks/fitted_ring.obj")

# Optional: Save as GLTF (.glb)
o3d.io.write_triangle_mesh("output/masks/fitted_ring.ply", combined_mesh)

