from __future__ import annotations
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

@torch.no_grad()
def compute_verification_metrics(model, pairs, targets, device="cpu"):
    model.eval()
    model.to(device)
    sims = []
    for img1, img2 in pairs:
        img1, img2 = img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device)
        e1 = torch.nn.functional.normalize(model(img1))
        e2 = torch.nn.functional.normalize(model(img2))
        sim = (e1 * e2).sum().item()
        sims.append(sim)
    y_score = np.array(sims)
    y_true = targets.numpy()
    best_thresh = None
    best_f1 = 0.0
    for thresh in np.linspace(-1, 1, 201):
        y_pred = (y_score >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    y_pred = (y_score >= best_thresh).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "best_threshold": float(best_thresh),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(best_f1),
    }
