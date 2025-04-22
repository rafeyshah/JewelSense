import torch
import clip
import faiss
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)


def display_results(query_path, results, dataset_folder):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, len(results)+1, 1)
    plt.imshow(Image.open(query_path))
    plt.title("Query")
    plt.axis('off')

    for i, filename in enumerate(results):
        path = os.path.join(dataset_folder, filename)
        img = Image.open(path)
        plt.subplot(1, len(results)+1, i+2)
        plt.imshow(img)
        plt.title(f"Match {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def display_results(query, results, dataset_folder, query_type="image"):
    plt.figure(figsize=(15, 4))

    if query_type == "image":
        plt.subplot(1, len(results)+1, 1)
        plt.imshow(Image.open(query))
        plt.title("Query")
        plt.axis('off')
    else:
        plt.subplot(1, len(results)+1, 1)
        plt.text(0.5, 0.5, f'"{query}"', wrap=True, fontsize=14,
                 ha='center', va='center')
        plt.axis('off')
        plt.title("Prompt")

    for i, filename in enumerate(results):
        path = os.path.join(dataset_folder, filename)
        img = Image.open(path)
        plt.subplot(1, len(results)+1, i+2)
        plt.imshow(img)
        plt.title(f"Match {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def search_similar(query_path, index_path="clip_search/index", dataset_path="data/test/images", top_k=5):
    # Load query image
    query_image = PREPROCESS(Image.open(query_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        query_embedding = MODEL.encode_image(
            query_image).cpu().numpy().astype('float32')

    # Load index and filenames
    index = faiss.read_index(f"{index_path}/clip.index")
    with open(f"{index_path}/filenames.pkl", "rb") as f:
        filenames = pickle.load(f)

    # Search
    distances, indices = index.search(query_embedding, top_k)
    results = [filenames[i] for i in indices[0]]

    print(f"\n[✓] Top {top_k} results for: {query_path}")
    for rank, name in enumerate(results, 1):
        print(f"{rank}. {name}")

    display_results(query_path, results, dataset_path)


def search_by_text(prompt, index_path="clip_search/index", dataset_path="data/test/images", top_k=5):
    with torch.no_grad():
        text_tokens = clip.tokenize([prompt]).to(DEVICE)
        text_embedding = MODEL.encode_text(
            text_tokens).cpu().numpy().astype('float32')

    index = faiss.read_index(f"{index_path}/clip.index")
    with open(f"{index_path}/filenames.pkl", "rb") as f:
        filenames = pickle.load(f)

    distances, indices = index.search(text_embedding, top_k)
    results = [filenames[i] for i in indices[0]]

    print(f"\n[✓] Top {top_k} results for prompt: \"{prompt}\"")
    display_results(prompt, results, dataset_path, query_type="text")


if __name__ == "__main__":
    # Example
    # search_similar("query.jpg")
    search_by_text("Necklace and rings ")  # Text-based
