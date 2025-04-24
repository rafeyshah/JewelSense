# 📐 Design Document – Jewelry Tracking AI

---

## 🔍 Project Overview

This project implements a modular, end-to-end AI system for detecting, tracking, segmenting, and visualizing jewelry items — specifically **rings**, **earrings**, and **dresses** — with added support for **3D modeling and augmented try-on visualizations**.

The system combines hand landmark tracking (MediaPipe), object detection (YOLOv8), deep tracking (DeepSORT), CLIP-based visual search, SAM-based segmentation, and Open3D 3D mesh placement.

---

## 🎯 Scope & Assumptions

### Scope
- Focused on 3 primary jewelry classes: `ring`, `earring`, `dress`
- Detect and track in both static images and videos
- Localize and segment rings using SAM
- Fit and align 3D ring meshes to finger joints
- Blend 3D ring with real images for try-on visualizations

### Assumptions
- YOLOv8 model trained on Roboflow datasets
- SAM is used to extract segmentation masks
- 3D hand landmarks are derived from MediaPipe
- Outputs include annotated images, overlays, and mesh exports

---

## 📦 Dataset Sources

- **Jewelry Detection Dataset** (Roboflow): [link](https://universe.roboflow.com/mpstme-k5t7r/jewellery_detect/model/17)
- **Dress Detection Dataset** (Roboflow): [link](https://universe.roboflow.com/jian-james-astrero/dress-dataset/dataset/4/download)

Both datasets were YOLOv8 formatted and used for training a 3-class detection model.

---

## 🛠️ Tools & Frameworks

| Tool       | Rationale for Selection                                                                                                                                     |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| YOLOv8     | Selected for robust and real-time object detection capabilities compared to alternatives like SSD or Faster R-CNN.                                          |
| MediaPipe  | Chosen for real-time efficiency in hand tracking over alternatives like OpenPose.                                                                           |
| DeepSORT   | Chosen for efficient multi-object tracking across frames, offering improved accuracy and consistency over SORT.                                             |
| CLIP       | Employed for superior semantic visual similarity compared to traditional feature extraction methods (SIFT, SURF).                                           |
| FAISS      | Utilized for high-performance similarity indexing and rapid retrieval of visual embeddings, outperforming simpler indexing methods like brute-force search. |
| SAM        | Chosen for zero-shot segmentation flexibility compared to traditional segmentation methods like Mask R-CNN.                                                 |
| Open3D     | Selected for comprehensive and easy-to-use 3D mesh processing and visualization capabilities compared to alternatives like MeshLab.                         |
| Matplotlib | Chosen for its straightforward integration and visualization capabilities, especially useful for overlaying 3D meshes onto images.                          |
| PIL/OpenCV | Employed for versatile image processing functionalities, offering ease of use, broad compatibility, and robust performance for image manipulation tasks.    |


---


## 🚦 Implementation Phases

### Phase 1 – Detection & Tracking
- YOLOv8 trained on `ring`, `earring`, `dress`
- MediaPipe used for hand landmark extraction
- DeepSORT added to enable video tracking with object persistence
- JSON output with class, ID, and bbox info

### Phase 2 – CLIP Search
- CLIP used to embed dataset images and prompt queries
- FAISS index created for fast retrieval
- Text-to-image & image-to-image search supported

### Phase 3 – 3D Ring Mesh Fitting
- SAM segments the ring using point from YOLO bbox center
- MediaPipe landmarks used to detect finger base
- Open3D torus mesh created and aligned with finger direction
- Ring mesh exported (.obj/.ply) for use in Blender/WebGL
- Matplotlib + PIL used for overlay visualization

---

## ⚠️ Documented Failures & Fixes

| Issue                                                       | Fix                                                                   |
| ----------------------------------------------------------- | --------------------------------------------------------------------- |
| ❌ SAM failed when clicked manually                          | ✅ Replaced with YOLO-detected bbox center as SAM input                |
| ❌ Ring appeared floating or sideways in 3D                  | ✅ Added finger direction vector from two landmarks for rotation       |
| ❌ Open3D render gave white screen                           | ✅ Adjusted axis limits and zoom to focus on ring joint                |
| ❌ Blank overlay due to incorrect image coordinate transform | ✅ Corrected Y-axis orientation to match OpenCV coordinate system      |
| ❌ MediaPipe failed on some hand poses                       | ✅ Enabled `static_image_mode=True` to ensure frame-by-frame detection |

---

## Experimental Failures and Decisions
- Novel view synthesis methods (NeRF) were too computationally intensive; deprioritized for immediate development.

## Iterative Improvements
- Optimized MediaPipe performance with batch processing and GPU acceleration.
- Improved segmentation accuracy through SAM prompt refinement.
- Enhanced mesh fitting accuracy through iterative algorithm tuning.

## Project Scope
- Successfully covers detection, segmentation, tracking, and basic 3D visualization.

## Limitations
- Does not include detailed clustering or novel view synthesis.
- Lacks comprehensive 3D hand/body rigging.

---

## ✅ Final Achievements

- YOLOv8 jewelry model trained on real-world data
- Fully automated ring segmentation and 3D placement
- CLIP search engine integrated for jewelry suggestions
- Exportable ring meshes for Blender/Web-based try-on
- Modular pipeline: easy to extend for earrings/dress segmentation

---

## 🔜 Future Work

| Feature                                    | Priority |
| ------------------------------------------ | -------- |
| Add earring and dress segmentation via SAM | High     |
| Fit 3D meshes for earrings and tiaras      | Medium   |
| Animate ring fitting or rotation (Open3D)  | Medium   |
| Build Streamlit or WebGL ring try-on app   | High     |
| Integrate LLM for tag-to-style suggestions | Medium   |

---

## 📄 Conclusion

This system demonstrates how modern AI tools — from object detection and segmentation to 3D modeling and language models — can be combined to build powerful, real-world vision solutions.

The ring fitting is already production-grade and provides a base to extend into other jewelry types and immersive AR try-on demos.