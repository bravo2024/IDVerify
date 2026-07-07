from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from src.data import make_synthetic, create_dataloaders, make_verification_pairs
from src.model import FaceEmbedder
from src.core import compute_verification_metrics

def test_data():
    data = make_synthetic(num_identities=5, imgs_per_id=10, seed=42)
    assert data["images"].shape[0] == 50
    assert data["images"].shape[1:] == (3, 112, 112)
    assert data["num_identities"] == 5

def test_dataloader():
    data = make_synthetic(num_identities=5, imgs_per_id=10, seed=42)
    tl, vl = create_dataloaders(data, batch_size=8)
    batch = next(iter(tl))
    assert batch[0].shape == (8, 3, 112, 112)
    assert batch[1].shape == (8,)

def test_model():
    model = FaceEmbedder(embedding_dim=128, num_classes=5)
    x = torch.randn(2, 3, 112, 112)
    logits, embeddings = model(x, torch.tensor([0, 1]))
    assert logits.shape == (2, 5)
    assert embeddings.shape == (2, 128)

def test_forward_embeddings():
    model = FaceEmbedder(embedding_dim=128, num_classes=5)
    model.eval()
    with torch.no_grad():
        emb = model(torch.randn(2, 3, 112, 112))
    assert emb.shape == (2, 128)
    assert torch.allclose(emb.norm(dim=1), torch.ones(2), atol=1e-5)

def test_verification_pairs():
    data = make_synthetic(num_identities=3, imgs_per_id=5, seed=42)
    pairs, targets = make_verification_pairs(data, num_pairs=20)
    assert len(pairs) == 20
    assert targets.shape[0] == 20
