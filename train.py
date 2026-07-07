from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import argparse
import torch
from src.data import make_synthetic, create_dataloaders, make_verification_pairs
from src.model import FaceEmbedder, train_model
from src.core import compute_verification_metrics
from src.evaluate import save_metrics, print_report
from src.persist import save_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-identities", type=int, default=50)
    p.add_argument("--imgs-per-id", type=int, default=40)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--scale", type=float, default=64.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = make_synthetic(num_identities=a.num_identities, imgs_per_id=a.imgs_per_id, seed=a.seed)
    print(f"Generated {data['n_samples']} face images, {data['num_identities']} identities")

    train_loader, val_loader = create_dataloaders(data, batch_size=a.batch_size, seed=a.seed)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = FaceEmbedder(embedding_dim=a.embed_dim, num_classes=a.num_identities, margin=a.margin, scale=a.scale)
    print(f"Model: ResNet-50 + ArcFace ({sum(p.numel() for p in model.parameters()):,} params)")

    model, history = train_model(model, train_loader, val_loader, epochs=a.epochs, lr=a.lr, device=a.device)

    pairs, targets = make_verification_pairs(data, num_pairs=500)
    ver_metrics = compute_verification_metrics(model, pairs, targets, device=a.device)

    print_report(ver_metrics)
    save_model(model)
    save_metrics({
        **ver_metrics,
        "best_val_acc": history["best_acc"],
        "epochs": a.epochs,
        "num_identities": a.num_identities,
        "embedding_dim": a.embed_dim,
        "margin": a.margin,
    })
    print("Saved models/best_model.pt and models/metrics.json")

if __name__ == "__main__":
    main()
