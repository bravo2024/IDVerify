from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from src.data import make_synthetic, create_dataloaders, make_verification_pairs
from src.model import FaceEmbedder
from src.core import compute_verification_metrics
from src.visualizations import plot_face_sample, plot_similarity_histogram, plot_embedding_space, _style

st.set_page_config(page_title="IDVerify | Face Recognition", layout="wide", page_icon="\U0001f9b4")

@st.cache_resource
def load_model():
    m = FaceEmbedder(embedding_dim=512, num_classes=50)
    p = Path("models/best_model.pt")
    if p.exists():
        m.load_state_dict(torch.load(p, map_location="cpu"))
    m.eval()
    return m

with st.sidebar:
    st.header("\u2699 Config")
    num_ids = st.slider("Identities", 10, 100, 50, 10)
    show_embeddings = st.checkbox("Show t-SNE", True)
    threshold = st.slider("Verification Threshold", 0.0, 1.0, 0.5, 0.05)
    st.caption("IDVerify | ArcFace | Identity Verification")

data = make_synthetic(num_identities=num_ids, imgs_per_id=40, seed=42)
_, val_loader = create_dataloaders(data, batch_size=16, seed=42)
val_images, val_labels = next(iter(val_loader))
model = load_model()

with torch.no_grad():
    logits, embeddings = model(val_images, val_labels)
    val_preds = logits.argmax(dim=1)
    val_acc = (val_preds == val_labels).float().mean().item()

pairs, targets = make_verification_pairs(data, num_pairs=200, seed=99)
ver_metrics = compute_verification_metrics(model, pairs, targets)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ID Accuracy", f"{val_acc:.4f}")
c2.metric("Verification AUC", f"{ver_metrics['roc_auc']:.4f}")
c3.metric("F1 Score", f"{ver_metrics['f1']:.4f}")
c4.metric("Identities", f"{num_ids}")

t1, t2, t3, t4 = st.tabs(["\U0001f4ca Explorer", "\U0001f52c Model Lab", "\U0001f9e0 Embedding Space", "\U0001f6e1\ufe0f Verification"])

with t1:
    st.subheader("Sample Faces by Identity")
    cols = st.columns(4)
    for i, col in enumerate(cols):
        with col:
            img = val_images[i]
            label = val_labels[i].item()
            fig = plot_face_sample(img)
            st.pyplot(fig)
            st.caption(f"ID: {label} | Pred: {val_preds[i].item()}")

with t2:
    rows = []
    with torch.no_grad():
        all_emb = []
        all_lbl = []
        for imgs, lbls in val_loader:
            e = model(imgs).numpy()
            all_emb.append(e)
            all_lbl.append(lbls.numpy())
        all_emb = np.concatenate(all_emb)
        all_lbl = np.concatenate(all_lbl)
    for k, v in ver_metrics.items():
        rows.append({"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)})
    st.dataframe(rows, use_container_width=True)
    if show_embeddings and len(np.unique(all_lbl)) >= 2:
        st.pyplot(plot_embedding_space(all_emb, all_lbl, num_identities=min(10, num_ids)))

with t3:
    st.subheader("Face Embedding Space")
    st.latex(r"f_{\theta}(x) \in \mathbb{R}^d, \quad \|f_{\theta}(x)\|_2 = 1")
    st.caption("ArcFace normalizes embeddings to lie on a hypersphere. Angular margin enforces intra-class compactness and inter-class separability.")
    st.latex(r"\mathcal{L}_{\text{ArcFace}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos \theta_j}}")
    st.caption("Additive angular margin m=0.5 pushes decision boundaries. Scale s=64 controls softmax temperature.")
    if show_embeddings:
        st.pyplot(plot_embedding_space(all_emb, all_lbl, num_identities=min(10, num_ids)))

with t4:
    st.subheader("Verification Performance")
    col_a, col_b = st.columns(2)
    with col_a:
        sims = []
        for img1, img2 in pairs[:100]:
            e1 = torch.nn.functional.normalize(model(img1.unsqueeze(0)))
            e2 = torch.nn.functional.normalize(model(img2.unsqueeze(0)))
            sims.append((e1 * e2).sum().item())
        st.pyplot(plot_similarity_histogram(sims, targets[:100], threshold))
    with col_b:
        st.metric("EER Threshold", f"{ver_metrics['best_threshold']:.3f}")
        st.metric("ROC AUC", f"{ver_metrics['roc_auc']:.4f}")
        st.metric("Precision", f"{ver_metrics['precision']:.4f}")
        st.metric("Recall", f"{ver_metrics['recall']:.4f}")
