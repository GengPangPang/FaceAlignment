import cv2
import torch
import deeplake
import numpy as np
from torch.utils.data import Dataset

from utils.heatmap import generate_heatmaps


class DeepLake300W(Dataset):
    def __init__(self, path, split="train", img_size=256, heatmap_size=64):
        self.ds = deeplake.load(path, read_only=True)
        self.img_size = img_size
        self.heatmap_size = heatmap_size

        target_label = 1 if split == "train" else 0

        self.indices = []
        for i in range(len(self.ds)):
            label = int(self.ds["labels"][i].numpy()[0])
            if label == target_label:
                self.indices.append(i)

        print(f"{split} samples:", len(self.indices))

    def __len__(self):
        return len(self.indices)

    def convert_keypoints(self, keypoints):
        keypoints = keypoints.reshape(68, 3)
        landmarks = keypoints[:, :2]
        return landmarks.astype(np.float32)

    def crop_face(self, img, pts, scale=1.5):
        h, w = img.shape[:2]

        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        size = max(x2 - x1, y2 - y1) * scale

        nx1 = int(max(cx - size / 2, 0))
        ny1 = int(max(cy - size / 2, 0))
        nx2 = int(min(cx + size / 2, w - 1))
        ny2 = int(min(cy + size / 2, h - 1))

        crop = img[ny1:ny2, nx1:nx2]
        pts = pts - np.array([nx1, ny1], dtype=np.float32)

        old_w = nx2 - nx1
        old_h = ny2 - ny1

        crop = cv2.resize(crop, (self.img_size, self.img_size))
        pts[:, 0] *= self.img_size / old_w
        pts[:, 1] *= self.img_size / old_h

        return crop, pts

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        img = self.ds["images"][real_idx].numpy()
        keypoints = self.ds["keypoints"][real_idx].numpy()

        # 兼容 RGB / 灰度图 / 单通道灰度图 / RGBA
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        pts = self.convert_keypoints(keypoints)
        img, pts = self.crop_face(img, pts)

        heatmaps = generate_heatmaps(
            pts,
            img_size=self.img_size,
            heatmap_size=self.heatmap_size
        )

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        heatmaps = torch.from_numpy(heatmaps).float()

        return img, heatmaps