from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import torch

def _style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor": "#1a1f2e", "figure.facecolor": "#1a1f2e",
        "axes.edgecolor": "#4a5568", "axes.labelcolor": "white",
        "xtick.color": "white", "ytick.color": "white",
        "text.color": "white", "legend.facecolor": "#1a1f2e",
        "legend.edgecolor": "#4a5568",
    })

def plot_face_sample(image_tensor):
    _style()
    fig, ax = plt.subplots(figsize=(4, 4))
    img = (image_tensor.permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
    ax.imshow(img); ax.axis("off")
    return fig

def plot_similarity_histogram(similarities, targets, threshold=0.5):
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))
    sims = np.array(similarities)
    targets = np.array(targets)
    genuine = sims[targets == 1]
    impostor = sims[targets == 0]
    ax.hist(genuine, bins=30, alpha=0.7, color="#22c55e", label="Genuine", density=True)
    ax.hist(impostor, bins=30, alpha=0.7, color="#f43f5e", label="Impostor", density=True)
    ax.axvline(threshold, color="#fbbf24", ls="--", lw=2, label=f"Threshold={threshold:.2f}")
    ax.set_title("Similarity Distribution", color="white"); ax.legend(); ax.grid(True, alpha=.2)
    return fig

def plot_embedding_space(embeddings, labels, num_identities=10):
    _style()
    from sklearn.manifold import TSNE
    mask = labels < num_identities
    emb_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings[mask])
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels[mask], cmap="tab10", s=10)
    ax.set_title("t-SNE of Face Embeddings", color="white"); ax.grid(True, alpha=.2)
    plt.colorbar(scatter, ax=ax)
    return fig
