from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path

class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=50, margin=0.5, scale=64.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings)
        weight = F.normalize(self.weight)
        cos_theta = F.linear(embeddings, weight)
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
        margin_theta = torch.cos(theta + self.margin)
        logits = torch.where(one_hot.bool(), margin_theta, cos_theta)
        logits *= self.scale
        return logits

class FaceEmbedder(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=50, margin=0.5, scale=64.0):
        super().__init__()
        import torchvision
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(2048, embedding_dim)
        self.head = ArcFaceHead(embedding_dim, num_classes, margin, scale)

    def forward(self, x, labels=None):
        feats = self.backbone(x).flatten(1)
        embeddings = F.normalize(self.fc(feats))
        if labels is not None:
            logits = self.head(embeddings, labels)
            return logits, embeddings
        return embeddings

    @torch.no_grad()
    def verify(self, img1, img2, threshold=0.5):
        self.eval()
        e1 = F.normalize(self(img1.unsqueeze(0)))
        e2 = F.normalize(self(img2.unsqueeze(0)))
        sim = (e1 * e2).sum().item()
        return sim > threshold, sim

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            logits, _ = model(images, labels)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images, labels)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total,
    }

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    scaler = GradScaler()
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["accuracy"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), Path("models/best_model.pt"))
            print(f"  -> Saved best model (acc={best_acc:.4f})")
    model.load_state_dict(torch.load(Path("models/best_model.pt")))
    history["best_acc"] = best_acc
    return model, history
