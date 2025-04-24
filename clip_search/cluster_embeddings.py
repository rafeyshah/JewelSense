import faiss
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path

# Load FAISS index
index = faiss.read_index("clip_search/index/clip.index")
features = index.reconstruct_n(0, index.ntotal)

# Load filenames
with open("clip_search/index/filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# Optional: reduce dimensions for clustering stability
pca = PCA(n_components=50)
features_reduced = pca.fit_transform(features)

# KMeans clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(features_reduced)

# Save cluster files
Path("output/clustering").mkdir(parents=True, exist_ok=True)
for i in range(n_clusters):
    with open(f"output/clustering/cluster_{i}.txt", "w") as f:
        for path, label in zip(filenames, labels):
            if label == i:
                f.write(f"{path}\n")

# Visualize in 2D
pca_2d = PCA(n_components=2).fit_transform(features_reduced)
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    plt.scatter(pca_2d[labels == i, 0], pca_2d[labels == i, 1], label=f"Cluster {i}", alpha=0.7)
plt.legend()
plt.title("Visual Clustering of Jewelry Designs (CLIP + KMeans)")
plt.savefig("output/clustering/jewelry_clusters.png")
plt.show()
