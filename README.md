# 💎 Jewelry Tracking AI

A modular, end-to-end AI system for detecting, tracking, and visually matching jewelry (rings, earrings, necklaces, dresses, etc.) using cutting-edge computer vision and multimodal AI tools.

---

## 📌 Features

- ✅ Finger & jewelry tracking using MediaPipe + YOLOv8
- ✅ Multi-class object detection (ring, earring, tiara, dress, etc.)
- ✅ DeepSORT-based video tracking
- ✅ CLIP-powered visual similarity search (image & text)
- ✅ JSON export of detection data
- ✅ Modular, production-ready codebase
- ✅ Clean Roboflow integration for jewelry + dress datasets
- ✅ Easy to extend with 3D modeling, SAM, or LLMs

---

## 📂 Datasets Used

- [Jewellery Detection Dataset](https://universe.roboflow.com/mpstme-k5t7r/jewellery_detect/model/17)
- [Dress Detection Dataset](https://universe.roboflow.com/jian-james-astrero/dress-dataset/dataset/4/download)
- Downloaded and formatted using Roboflow Python API
- Trained using Ultralytics YOLOv8 in Google Colab

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

Place Roboflow YOLOv8-format dataset under:

```
data/train
data/val
data/test
```

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

- MediaPipe Hands – hand landmark tracking
- YOLOv8 – object detection (Ultralytics)
- DeepSORT – tracking across video frames
- CLIP – visual + text embedding model (OpenAI)
- FAISS – similarity search engine
- Roboflow – dataset hosting and download API
- (Optional) SAM, MANO, LLaVA – segmentation, 3D modeling, vision-LM

---

## 🔮 Future Work

- Phase 3: Fit 3D ring mesh to hand joints
- Prompt-based jewelry suggestions using GPT or LLaVA
- Smart filters by style, material, design tags
- Deploy UI with Streamlit or FastAPI

---

## 📄 License

MIT — feel free to use, modify, and build on it!