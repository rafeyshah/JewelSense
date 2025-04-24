# 💎 JewelSense – Multimodal Computer Vision & AR System

This project is an end-to-end AI pipeline designed to detect, track, search, and fit jewelry items — specifically **rings**, **earrings**, and **dresses** — using state-of-the-art computer vision, 3D modeling, and multimodal tools like CLIP and SAM. It now includes 3D mesh fitting and AR-style visualization of rings placed on hands.

---

## 📌 Project Features

- 🖐️ Detect & track hands and rings with MediaPipe & YOLOv8
- 🧠 Smart visual similarity search using CLIP (image & text prompts)
- 📱 Unsupervised Style Clustering (CLIP + KMeans), discover hidden design grouping
- 🌀 DeepSORT-based video tracking of jewelry
- 🧊 Segment jewelry (rings) with SAM (Segment Anything)
- 💍 Place and align 3D ring meshes on finger joints
- 🖼️ Overlay 3D ring on original hand image for AR-style try-on
- 📦 Export `.obj`/`.ply` files of ring placements for 3D tools (Blender, WebGL)
- ✅ Professional, modular, and clean codebase with full documentation


## Project Scope and Limitations
- Current implementation: Jewelry detection, segmentation, tracking, and basic 3D mesh fitting.
- Limitations:
  - No comprehensive clustering
  - Novel view synthesis not implemented
  - No complete 3D rig/model of hand or body

📄 For a detailed breakdown of tools, methodology, failures, and future scope, please refer to the [Design Document](design_doc.md).

## ✅ Completed Milestones

| Feature                                      | Status   |
|---------------------------------------------|----------|
| Ring detection with YOLOv8                  | ✅ Done  |
| Hand joint detection (MediaPipe)            | ✅ Done  |
| Video tracking with DeepSORT                | ✅ Done  |
| CLIP-powered search                         | ✅ Done  |
| Automatically group visually similar jewelry designs using CLIP embeddings and KMeans clustering | ✅ Done  |
| Prompt-to-image retrieval                   | ✅ Done  |
| SAM-based ring segmentation                 | ✅ Done  |
| 3D ring placement using Open3D              | ✅ Done  |
| Ring orientation aligned to finger direction| ✅ Done  |
| Export ring mesh as .obj/.ply               | ✅ Done  |
| Overlay 3D ring on original image           | ✅ Done  |
| Documentation (README, Design Doc)          | ✅ Done  |
| Code published to GitHub                    | ✅ Done  |


## 📂 Datasets Used

- [💍 Jewelry Detection Dataset (Roboflow)](https://universe.roboflow.com/mpstme-k5t7r/jewellery_detect/model/17)
- [👗 Dress Detection Dataset (Roboflow)](https://universe.roboflow.com/jian-james-astrero/dress-dataset/dataset/4/download)
- Classes used: `ring`, `earring`, `dress`
- Format: YOLOv8


## 🚀 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Inference
```bash
python main.py          # For image
python run_video.py     # For video
```

### 3. Visual Similarity Search
```bash
python clip_search/embed_dataset.py
python clip_search/search_similar.py
```

### 4. Ring Segmentation + 3D Placement
```bash
python segmentation/segment_ring.py
python fitting/fit_ring_mesh.py
```

### 5. Overlay Ring on Image
```bash
python visualization/overlay_ring_3d_on_image.py
```

### 6. Clip + KMeans
```bash
python clip_search/cluster_embeddings.py
```


## 📦 Folder Structure Snapshot

```
jewelry-tracking-ai/
├── segmentation/               ← SAM ring mask generation
├── fitting/                    ← 3D mesh placement on finger joint
├── clip_search/                ← CLIP similarity search & clustering
├── visualization/              ← Overlay 3D ring on 2D hand
├── detectors/, trackers/       ← YOLOv8 + DeepSORT
├── output/                     ← Rendered results, ring masks, meshes
├── main.py, run_video.py       ← Entry points
├── README.md, design_doc.md    ← Documentation
```

## 📹 Demo Videos

Explore JewelSense in action through the following demo recordings:

- 🔍 **Detection & Tracking**  
  [Watch on Loom](https://www.loom.com/share/9936c0e8e7d44c7fa0081af41a146232?sid=ab3230e9-f9b9-4902-a15f-01ae4bf43f8c)

- 💍 **Hand Bug Removed**  
  [Watch on Loom](https://www.loom.com/share/9c3dd0216b194685ae47c94377bcacb5?sid=6fcc7217-5cb5-4c3c-b7bb-31d0fce7a872)

- 🧠 **CLIP-Based Similarity Search**  
  [Watch on Loom](https://www.loom.com/share/df1304450a224dfb9eb7c39bed89dd7b?sid=0b806553-5e45-4ec3-9d37-0e6c143cc47a)

- 🧩 **Style Clustering with CLIP + KMeans**  
  [Watch on Loom](https://www.loom.com/share/7c6c5f4ab87b44bdae388b876ce60882?sid=78e57a51-5f0c-4413-b27c-2b69cfe371b0)


## 🔜 Upcoming Features (Planned)

- 🪞 Add earrings and dresses to 3D segmentation
- 📐 Fit earrings or necklaces in 3D space
- 🤖 Use GPT/LLaVA for style-based jewelry prompts
- 🌐 Web-based 3D viewer (Three.js / Streamlit)
- 📲 Full AR preview & product try-on simulation


## 📄 License

MIT — free to use, build on, and contribute.


**Author**: Syed Abdul Rafey Ali 
**Status**: 🎯 Production-ready with real-world & research use potential
