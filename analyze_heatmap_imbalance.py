import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    DATASET_PATH,
    IMG_SIZE,
    HEATMAP_SIZE,
    CROP_SCALE,
    HEATMAP_SIGMA,
    USE_DISK_CACHE,
    CACHE_DIR,
)

from datasets.deeplake_300w import DeepLake300W


THRESHOLDS = [0.5, 0.1, 0.01, 0.001]


def analyze_split(split="train"):
    dataset = DeepLake300W(
        DATASET_PATH,
        split=split,
        img_size=IMG_SIZE,
        heatmap_size=HEATMAP_SIZE,
        crop_scale=CROP_SCALE,
        heatmap_sigma=HEATMAP_SIGMA,
        use_disk_cache=USE_DISK_CACHE,
        cache_dir=CACHE_DIR,
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    total_pixels = 0
    foreground_counts = {t: 0 for t in THRESHOLDS}

    # 按关键点统计，方便看 jaw 与其他点是否不同
    num_landmarks = 68
    landmark_total_pixels = np.zeros(num_landmarks, dtype=np.float64)
    landmark_fg_counts = {t: np.zeros(num_landmarks, dtype=np.float64) for t in THRESHOLDS}

    for imgs, heatmaps, pts in loader:
        # heatmaps: [B, 68, 64, 64]
        B, K, H, W = heatmaps.shape

        total_pixels += heatmaps.numel()

        for t in THRESHOLDS:
            foreground_counts[t] += (heatmaps > t).sum().item()

        # per-landmark
        per_lm_pixels = B * H * W
        landmark_total_pixels += per_lm_pixels

        for t in THRESHOLDS:
            fg = (heatmaps > t).sum(dim=(0, 2, 3)).cpu().numpy()
            landmark_fg_counts[t] += fg

    print("=" * 80)
    print(f"Split: {split}")
    print(f"Heatmap size: {HEATMAP_SIZE}x{HEATMAP_SIZE}")
    print(f"Total heatmap pixels: {total_pixels}")

    for t in THRESHOLDS:
        fg = foreground_counts[t]
        bg = total_pixels - fg
        fg_ratio = fg / total_pixels
        bg_ratio = bg / total_pixels

        print(f"\nThreshold H > {t}")
        print(f"Foreground pixels: {fg}")
        print(f"Background pixels: {bg}")
        print(f"Foreground ratio: {fg_ratio * 100:.4f}%")
        print(f"Background ratio: {bg_ratio * 100:.4f}%")
        print(f"BG / FG ratio: {bg / max(fg, 1):.2f}")

    print("\nPer-region foreground ratio at threshold 0.01:")
    region_map = {
        "jaw": list(range(0, 17)),
        "right_eyebrow": list(range(17, 22)),
        "left_eyebrow": list(range(22, 27)),
        "nose": list(range(27, 36)),
        "right_eye": list(range(36, 42)),
        "left_eye": list(range(42, 48)),
        "mouth": list(range(48, 68)),
    }

    t = 0.01
    ratios = landmark_fg_counts[t] / landmark_total_pixels

    for name, idxs in region_map.items():
        region_ratio = ratios[idxs].mean()
        print(f"{name:15s}: {region_ratio * 100:.4f}%")

    print("=" * 80)


if __name__ == "__main__":
    analyze_split("train")
    analyze_split("test")