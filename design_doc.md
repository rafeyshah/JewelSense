# 📐 Design Document – Jewelry Tracking AI

---

## 🔍 Project Overview

This project implements an AI-powered system that detects, tracks, and retrieves jewelry items (rings, earrings, tiaras, etc.) from images and videos. The system combines real-time hand landmark detection, object detection, visual similarity search using CLIP, and deep tracking.

---

## 🧩 Scope & Assumptions

### Scope:
- Detect and track rings, earrings, and dresses from real-world input (image/video)
- Match jewelry based on a query image or natural language description
- Provide clean outputs (annotated images/videos, structured JSON)
- Designed to be extendable with 3D modeling, segmentation, and LLM integration

### Assumptions:
- Roboflow dataset is labeled and in YOLOv8 format
- Jewelry classes: ['Bracelets', 'Brooches', 'belt', 'earring', 'maangtika', 'necklace', 'nose ring', 'ring', 'tiara']
- Inference should run on GPU (but works on CPU with slight slowdowns)

---

## 🛠️ Tools & API Choices – Rationale

| Tool           | Purpose                        | Why Chosen |
|----------------|--------------------------------|------------|
| MediaPipe      | Finger detection               | Fast, lightweight, 21 keypoint outputs |
| YOLOv8         | Object detection               | SOTA performance, Roboflow-compatible |
| DeepSORT       | Object tracking                | Robust, simple to integrate |
| CLIP (OpenAI)  | Visual similarity search       | Multimodal (text & image) embedding |
| FAISS          | Search indexing                | Scalable nearest-neighbor search |
| Matplotlib     | Grid visualization             | Simple result viewing |

---

## 🚦 Implementation Overview

### Phase 1 – Tracking
- MediaPipe detects finger landmarks (21 per hand)
- YOLOv8 detects jewelry bounding boxes (multi-class)
- DeepSORT tracks jewelry across frames with consistent ID
- Overlay visuals on both image and video outputs

### Phase 2 – Visual Search
- Use CLIP to embed all dataset images into vector space
- Use FAISS to build a searchable index
- Implement both image-to-image and text-to-image similarity search
- Display results in a grid with query and top matches

### Phase 3 – (Planned)
- Use SAM for fine-grained segmentation
- Fit segmented rings to 3D hand mesh (MANO)
- Enable AR-level try-on and positioning

---

## ⚠️ Known Issues & Failures – Documented

### ❌ MediaPipe missed hands in blurry frames
- **Fix**: Switched from `static_image_mode=False` to `True` for consistent detection

### ❌ DeepSORT returned unstable IDs when confidence was low
- **Fix**: Tuned `max_age=30` and filtered low-confidence boxes in YOLOv8

### ❌ CLIP image search showed unrelated results for earrings
- **Fix**: Limited search space to specific class (e.g., only “earring” images)

### ❌ YOLOv8 confused “bracelet” vs. “ring” on hand
- **Fix**: Augmented training set with more wrist/hand boundary examples

---

## 📈 Iterative Improvements

- Started with ring-only detection → expanded to multi-class
- First built image inference → expanded to video + tracking
- Initially only image-based CLIP → added text-based search
- Now ready for 3D modeling + segmentation

---

## ✅ Final Outcome

- Modular project, easily extensible
- Clean outputs (JSON, images, video)
- Multimodal AI integration (CV + CLIP)
- Great base for AR try-on, product recommendation, or ecommerce

---

## 📦 Next Steps

- Implement ring/dress segmentation using SAM
- Load ring meshes in Blender + align with MediaPipe 3D
- Deploy model with Streamlit or FastAPI frontend