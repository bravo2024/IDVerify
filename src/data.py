from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import math

IMG_SIZE = 112
NUM_IDENTITIES = 50
IMAGES_PER_IDENTITY = 40

def _draw_face(img, identity_params, rng):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    id_seed, color_var = identity_params
    rng_id = np.random.default_rng(id_seed)

    face_r = int(min(h, w) * 0.38 * rng_id.uniform(0.9, 1.1))
    face_color = (
        int(180 + 40 * rng_id.uniform() + color_var),
        int(120 + 40 * rng_id.uniform() + color_var),
        int(80 + 30 * rng_id.uniform() + color_var * 0.5),
    )
    rr, cc = np.ogrid[:h, :w]
    mask = ((rr - cy + int(rng.uniform(-3, 3))) ** 2 / (face_r * 1.2) ** 2 +
            (cc - cx + int(rng.uniform(-3, 3))) ** 2 / face_r ** 2) <= 1
    for c in range(3):
        img[:, :, c][mask] = np.clip(img[:, :, c][mask] * 0.3 + face_color[c] * 0.7, 0, 255)

    eye_y_offset = int(face_r * 0.25 * rng_id.uniform(0.8, 1.2))
    eye_x_spread = int(face_r * 0.3 * rng_id.uniform(0.8, 1.2))
    eye_r = int(face_r * 0.08 * rng_id.uniform(0.8, 1.2))
    eye_color = (255 - int(40 * rng_id.uniform()), 255 - int(40 * rng_id.uniform()), 255 - int(40 * rng_id.uniform()))
    for ex in [-eye_x_spread, eye_x_spread]:
        eye_cy = cy - eye_y_offset + int(rng.uniform(-1, 1))
        eye_cx = cx + ex + int(rng.uniform(-1, 1))
        ee, cc2 = np.ogrid[:h, :w]
        eye_mask = ((ee - eye_cy) ** 2 + (cc2 - eye_cx) ** 2) <= eye_r ** 2
        for c in range(3):
            img[:, :, c][eye_mask] = eye_color[c]

    nose_h = int(face_r * 0.15 * rng_id.uniform(0.8, 1.2))
    nose_w = int(face_r * 0.1 * rng_id.uniform(0.8, 1.2))
    nose_y = cy + int(face_r * 0.05 * rng_id.uniform(0.8, 1.2))
    nose_color = (200, 150, 120)
    nose_mask = ((rr - nose_y) ** 2 / nose_h ** 2 + (cc - cx) ** 2 / nose_w ** 2) <= 1
    for c in range(3):
        img[:, :, c][nose_mask] = nose_color[c]

    mouth_y = cy + int(face_r * 0.4 * rng_id.uniform(0.8, 1.2))
    mouth_w = int(face_r * 0.25 * rng_id.uniform(0.8, 1.2))
    mouth_h = int(face_r * 0.08 * rng_id.uniform(0.8, 1.2))
    mouth_color = (180, 100, 100)
    mouth_mask = ((rr - mouth_y) ** 2 / mouth_h ** 2 + (cc - cx) ** 2 / mouth_w ** 2) <= 1
    for c in range(3):
        img[:, :, c][mouth_mask] = mouth_color[c]

def make_synthetic(num_identities=NUM_IDENTITIES, imgs_per_id=IMAGES_PER_IDENTITY, img_size=IMG_SIZE, seed=42):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    identity_params = {}
    for id_ in range(num_identities):
        id_seed = id_ * 1000 + seed
        rng_id = np.random.default_rng(id_seed)
        identity_params[id_] = (id_seed, rng_id.uniform(-15, 15))
        for _ in range(imgs_per_id):
            img = np.full((img_size, img_size, 3), 200, dtype=np.uint8)
            color_var = rng.uniform(-10, 10)
            _draw_face(img, (id_seed, color_var), rng)
            images.append(torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0)
            labels.append(id_)
    data = {
        "images": torch.stack(images),
        "labels": torch.tensor(labels, dtype=torch.long),
        "num_identities": num_identities,
        "n_samples": len(images),
    }
    return data

class FaceDataset(Dataset):
    def __init__(self, data, split="train", val_split=0.2, seed=42):
        n = data["n_samples"]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        split_n = int(n * (1 - val_split))
        indices = idx[:split_n] if split == "train" else idx[split_n:]
        self.images = data["images"][indices]
        self.labels = data["labels"][indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def create_dataloaders(data, batch_size=32, val_split=0.2, seed=42):
    train_ds = FaceDataset(data, "train", val_split, seed)
    val_ds = FaceDataset(data, "val", val_split, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader

def make_verification_pairs(data, num_pairs=500, seed=42):
    rng = np.random.default_rng(seed)
    n = data["n_samples"]
    images = data["images"]
    labels = data["labels"]
    pairs, targets = [], []
    for _ in range(num_pairs):
        same = rng.random() > 0.5
        if same:
            id_ = rng.integers(0, data["num_identities"])
            idxs = torch.where(labels == id_)[0]
            if len(idxs) < 2:
                continue
            i1, i2 = rng.choice(idxs.numpy(), 2, replace=False)
            target = 1
        else:
            id1, id2 = rng.choice(data["num_identities"], 2, replace=False)
            idxs1 = torch.where(labels == id1)[0]
            idxs2 = torch.where(labels == id2)[0]
            i1 = rng.choice(idxs1.numpy())
            i2 = rng.choice(idxs2.numpy())
            target = 0
        pairs.append((images[i1], images[i2]))
        targets.append(target)
    return pairs, torch.tensor(targets)
