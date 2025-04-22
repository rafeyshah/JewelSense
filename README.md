# 💎 Jewelry Tracking AI

A modular, end-to-end AI system for detecting, tracking, and visually matching jewelry (rings, earrings, necklaces, etc.) using cutting-edge computer vision and multimodal AI tools.

---

## 📌 Features

- ✅ Finger & jewelry tracking using MediaPipe + YOLOv8
- ✅ Multi-class object detection (ring, earring, tiara, etc.)
- ✅ DeepSORT-based video tracking
- ✅ CLIP-powered visual similarity search (image & text)
- ✅ JSON export of detection data
- ✅ Modular, production-ready codebase
- ✅ Easy to extend with 3D modeling, SAM, or LLMs

---

## 📁 Folder Structure

```
jewelry-tracking-ai/
├── models/                 # Hand tracking (MediaPipe)
├── detectors/              # Jewelry detection (YOLOv8)
├── trackers/               # Object tracking (DeepSORT)
├── pipeline/               # Full video inference pipeline
├── clip_search/            # CLIP-based image & text search
├── data/                   # Roboflow dataset (train/val/test)
├── output/                 # Saved results (images, videos, JSON)
├── config/                 # Dataset YAML configs
├── main.py                 # Image inference entry point
├── run_video.py            # Video tracking entry point
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Use your Roboflow dataset (YOLOv8 format) under `data/train`, `data/val`, `data/test`.

### 3. Run Detection & Tracking

#### 🖼️ Image Inference
```bash
python main.py
```

#### 🎥 Video Tracking
```bash
python run_video.py
```

---

## 🔍 CLIP Visual Similarity Search

### Build Embedding Index

```bash
python clip_search/embed_dataset.py
```

### Search by Image

```bash
python clip_search/search_similar.py
# Uses "query.jpg" by default
```

### Search by Text Prompt

```python
search_by_text("gold ring with emerald")
```

---

## 🧠 Models & Tools Used

- MediaPipe Hands (Google) – finger joint detection
- YOLOv8 (Ultralytics) – object detection
- DeepSORT – object tracking
- CLIP (OpenAI) – image & text embeddings
- FAISS – similarity search
- SAM (optional) – segmentation (planned)
- MANO / SMPL (optional) – 3D hand modeling

---

## 📦 Future Work

- Phase 3: 3D ring mesh fitting on hand model
- Prompt-driven search using LLMs
- Style classification / recommendation

---

## 📄 License

MIT — feel free to use, modify, and build on it!