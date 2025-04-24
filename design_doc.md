# 📐 Design Document – Jewelry Tracking AI

---

## 🔍 Project Overview

This AI project builds a complete pipeline to detect, track, and visually search for key fashion and jewelry elements — specifically: `ring`, `earring`, and `dress`.  
It combines hand landmark tracking, YOLOv8 object detection, DeepSORT tracking, and CLIP-based visual/text search, all built modularly and ready for extension into AR or LLM-assisted applications.

---

## 🧩 Scope & Assumptions

### Scope:
- Focus on detecting and tracking 3 target classes: ring, earring, dress
- Apply real-time hand tracking to localize jewelry placement
- Enable visual & text-based similarity search via CLIP
- Save outputs as images, video, and JSON
- Use real-world datasets and pre-trained open models

### Assumptions:
- Roboflow datasets are in YOLOv8 format
- Classes used: `ring`, `earring`, `dress`
- Environment: Google Colab or local Python setup with GPU support

---

## 📦 Dataset Details

### Sources:
- [Jewelry Detection – Roboflow](https://universe.roboflow.com/mpstme-k5t7r/jewellery_detect/model/17)
- [Dress Detection – Roboflow](https://universe.roboflow.com/jian-james-astrero/dress-dataset/dataset/4/download)

### Structure:
- Downloaded via Roboflow API in YOLOv8 format
- Merged or used separately depending on task
- Trained in Colab with Ultralytics YOLOv8

---

## 🛠️ Tools & API Decisions

| Tool        | Purpose                          | Why Used |
|-------------|-----------------------------------|----------|
| YOLOv8      | Object detection                  | Fast, accurate, Roboflow-ready |
| MediaPipe   | Hand landmark detection           | 21-point finger detection |
| DeepSORT    | Object tracking in video          | Maintains track ID |
| CLIP        | Visual/text similarity            | Multimodal: image + text |
| FAISS       | Fast similarity search            | Efficient on large embeddings |
| Roboflow    | Dataset download and config       | One-click data formatting |
| Colab       | Model training                    | Cloud GPU access |

---

## 🚦 Implementation Walkthrough

### Phase 1 – Detection + Tracking
- Hand joints localized with MediaPipe
- YOLOv8 detects 3-class objects: ring, earring, dress
- DeepSORT tracks object identities across video
- Annotated output written to image/video and JSON

### Phase 2 – Visual Search (CLIP)
- Dataset images embedded using CLIP
- FAISS used to index the embeddings
- Query via image or prompt (e.g. "silver ring with stone")
- Grid of Top-K results shown

---

## ⚠️ Documented Failures & Fixes

| Issue                                               | Fix                                   |
|-----------------------------------------------------|----------------------------------------|
| MediaPipe missed hands during blur/motion           | Enabled `static_image_mode=True`       |
| DeepSORT track IDs changed too fast                 | Increased `max_age` and filtered confidence |
| CLIP matched wrong class (e.g. ring → necklace)     | Applied class-based filtering in search |
| YOLOv8 confused ring vs. bracelet in some cases     | Expanded annotated dataset from Roboflow |

---

## 📈 Improvements Achieved

- First tested on ring-only → expanded to 3-class detection
- Started with image-only inference → added video pipeline
- CLIP search upgraded from image-only → to prompt-based
- Visual results improved with grid display

---

## ✅ Current Outcome

- 3-class YOLOv8 model (ring, earring, dress)
- CLIP similarity engine for image & prompt search
- Real-time hand detection and tracking
- Modular structure with export-ready data

---

## 🔜 Future Work

| Feature                        | Status    |
|--------------------------------|-----------|
| SAM-based jewelry segmentation | Planned   |
| 3D mesh fitting (ring → hand)  | Planned   |
| LLM captioning or style prompts| Planned   |
| Try-on UX with AR              | Planned   |

---

## 📄 Conclusion

This project merges practical vision, real datasets, tracking logic, and multimodal AI to create a powerful jewelry & fashion AI pipeline.  
With Phase 1 and 2 complete, it's ready to evolve into full 3D modeling, semantic understanding, and live application.