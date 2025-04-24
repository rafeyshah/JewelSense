# 💎 Jewelry Tracking AI – End-to-End Computer Vision System

This project builds a modular, intelligent system to detect, track, and visually search for jewelry items (rings, earrings) and fashion elements (dresses) using AI.  
It combines hand tracking, object detection, video tracking, and CLIP-based similarity search.  
Developed with clean architecture, open datasets, and extensibility in mind.

---

## 📌 Project Scope

A complete pipeline that:
- Detects jewelry & dress items (`ring`, `earring`, `dress`)
- Tracks jewelry items over video using DeepSORT
- Provides visual similarity search using CLIP (image & text)
- Exports results as annotated images, videos, and JSON logs
- Uses Roboflow public datasets and trained with YOLOv8

---

## ✅ What Has Been Done

| Feature                                      | Status   |
|---------------------------------------------|----------|
| Hand & finger detection using MediaPipe     | ✅ Done  |
| Jewelry detection (ring, earring, dress)    | ✅ Done  |
| Video tracking using DeepSORT               | ✅ Done  |
| JSON export of detections                   | ✅ Done  |
| CLIP-based visual search (image ↔ image)    | ✅ Done  |
| CLIP-based prompt search (text ↔ image)     | ✅ Done  |
| Dataset: Jewelry + Dress from Roboflow      | ✅ Done  |
| Google Colab training on 3-class YOLOv8     | ✅ Done  |
| Modular architecture + visualizations       | ✅ Done  |
| Design + README documentation               | ✅ Done  |

---

## 🔜 What Remains / Future Work

| Task                                        | Planned   |
|---------------------------------------------|-----------|
| SAM-based jewelry segmentation              | 🔜 Next   |
| 3D hand modeling (e.g., MANO, Blender)      | 🔜 Next   |
| Fitting ring meshes to fingers              | 🔜 Next   |
| LLM-based search (e.g., GPT jewelry prompts)| 🔜 Next   |
| Streamlit / FastAPI frontend                | 🔜 Next   |

---

## 📂 Datasets Used

- [💍 Jewelry Detection Dataset (Roboflow)](https://universe.roboflow.com/mpstme-k5t7r/jewellery_detect/model/17)
- [👗 Dress Detection Dataset (Roboflow)](https://universe.roboflow.com/jian-james-astrero/dress-dataset/dataset/4/download)
- Format: YOLOv8 — used to train a single multi-class model on 3 classes

---

## 🧠 Tools & Frameworks

- YOLOv8 (Ultralytics) – multi-class object detection
- MediaPipe – hand landmark detection (21 points)
- DeepSORT – consistent tracking across frames
- CLIP (OpenAI) – visual & text similarity
- FAISS – fast similarity search
- Matplotlib – result visualization
- Roboflow API – dataset loading
- Google Colab – model training

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Object Detection

```bash
python main.py
```

### 3. Run Video Tracking

```bash
python run_video.py
```

### 4. Build CLIP Index (Visual Search)

```bash
python clip_search/embed_dataset.py
```

### 5. Search by Image or Text

```bash
python clip_search/search_similar.py
# OR
search_by_text("gold ring with diamond")
```

---

## 📦 Folder Structure

```
jewelry-tracking-ai/
├── models/           → MediaPipe-based hand tracker
├── detectors/        → YOLOv8 object detection
├── trackers/         → DeepSORT tracking logic
├── pipeline/         → Video pipeline integration
├── clip_search/      → CLIP + FAISS visual search
├── output/           → Saved results (images/videos/json)
├── data/             → Roboflow datasets (YOLOv8 format)
├── main.py           → Run on image
├── run_video.py      → Run on video
└── requirements.txt
```

---

## 📄 License

MIT License – free to use, extend, and deploy in commercial or research use cases.

---

**Author**: Syed Abdul Rafey Ali  
**Status**: 💯 Production-ready with future research potential