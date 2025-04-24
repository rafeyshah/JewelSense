# 📐 Design Document – Jewelry Tracking AI

---

## 🔍 Project Overview

This project implements an AI-powered system that detects, tracks, and retrieves jewelry items (rings, earrings, tiaras, dresses, etc.) from images and videos. The system combines real-time hand landmark detection, object detection, visual similarity search using CLIP, and deep tracking using DeepSORT.

---

## 🧩 Scope & Assumptions

### Scope:
- Detect and track rings, earrings, and dresses from real-world input (image/video)
- Match jewelry based on a query image or natural language description
- Provide clean outputs (annotated images/videos, structured JSON)
- Designed to be extendable with 3D modeling, segmentation, and LLM integration

### Assumptions:
- Roboflow datasets are labeled and exported in YOLOv8 format
- Jewelry and fashion classes include: ['ring', 'earing', 'dress']
- Environment: Google Colab + GPU support (or fallback to CPU)

---

## 📦 Dataset Details

### Datasets Used:
- [Jewellery Detection Dataset](https://universe.roboflow.com/mpstme-k5t7r/jewellery_detect/model/17)
- [Dress Detection Dataset](https://universe.roboflow.com/jian-james-astrero/dress-dataset/dataset/4/download)

### Integration:
- Downloaded using Roboflow Python API
- Automatically structured into `train`, `val`, and `test` sets
- Used in a unified YOLOv8 training notebook (`train.ipynb`) hosted on Google Colab

---

## 🛠️ Tools & API Choices – Rationale

| Tool           | Purpose                        | Why Chosen |
|----------------|--------------------------------|------------|
| MediaPipe      | Finger detection               | Fast, lightweight, 21 keypoint outputs |
| YOLOv8         | Object detection               | High accuracy, Roboflow-compatible |
| DeepSORT       | Object tracking                | Maintains object ID across video |
| CLIP (OpenAI)  | Visual/text similarity search  | Supports both images and prompts |
| FAISS          | Search indexing                | High-speed nearest neighbor queries |
| Matplotlib     | Visual result grid             | Simple, flexible |
| Roboflow API   | Dataset access and management  | Easy download + class labeling |

---

## 🚦 Implementation Overview

### Phase 1 – Tracking
- MediaPipe detects hand landmarks (21 points)
- YOLOv8 detects 9 jewelry/fashion classes
- DeepSORT tracks jewelry across frames with consistent IDs
- Overlay bounding boxes and landmarks for clarity

### Phase 2 – Visual Similarity Search
- CLIP used to embed all dataset images
- FAISS index built from image embeddings
- Visual + text query supported for similarity
- Result display in grid format

### Phase 3 – (Planned)
- Use SAM (Segment Anything) to isolate ring/dress
- Fit 3D jewelry meshes (e.g., ring) onto hand model
- Integrate Blender or Open3D for visualization

---

## ⚠️ Known Issues & Fixes

### ❌ MediaPipe missed hands in motion blur
- ✅ Fix: Enabled `static_image_mode=True` to stabilize detection

### ❌ DeepSORT IDs changed erratically on occlusion
- ✅ Fix: Tuned tracker parameters and ignored low-confidence boxes

### ❌ CLIP mismatched earrings and necklaces
- ✅ Fix: Limited FAISS search scope per class (e.g., only “earrings”)

### ❌ YOLOv8 confused bracelets and rings
- ✅ Fix: Added more boundary-specific training samples

---

## 📈 Iterative Improvements

- ✅ Started with ring-only detection → extended to multi-class jewelry + dress
- ✅ Built inference for static images → scaled to video tracking
- ✅ Initially image-only CLIP → added prompt-based search
- 🔜 Next: 3D segmentation + mesh fitting using MediaPipe 3D & SAM

---

## ✅ Final Outcome

- Modular, documented AI system for jewelry detection and tracking
- Visual + semantic search powered by CLIP
- Ready for real-world use or academic portfolio
- Easily extendable to AR try-on or ecommerce search

---

## 🔮 Next Steps

- Implement ring segmentation with SAM
- Fit 3D meshes in Blender to match hand joints
- Deploy via FastAPI/Streamlit interface
- Add GPT/LLaVA integration for style-based recommendations