import os
import torch
import clip
from PIL import Image
import numpy as np
import faiss
import pickle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)


def encode_image(img_path):
    image = PREPROCESS(Image.open(img_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = MODEL.encode_image(image)
    return embedding[0].cpu().numpy()


def build_index(dataset_folder, index_folder):
    embeddings = []
    filenames = []

    for file in os.listdir(dataset_folder):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(dataset_folder, file)
            emb = encode_image(path)
            embeddings.append(emb)
            filenames.append(file)

    embeddings = np.vstack(embeddings).astype('float32')

    # Save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_folder, "clip.index"))

    # Save filename map
    with open(os.path.join(index_folder, "filenames.pkl"), "wb") as f:
        pickle.dump(filenames, f)

    print(f"[✓] Indexed {len(filenames)} images.")


if __name__ == "__main__":
    build_index("data/test/images", "clip_search/index")
