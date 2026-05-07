import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from config import (
    DATASET_PATH,
    IMG_SIZE,
    HEATMAP_SIZE,
    NUM_LANDMARKS,
    DEVICE,
    CHECKPOINT_DIR,
    TEST_DIR,
    EVAL_BATCH_SIZE,
    NME_NORM_TYPE,
    USE_QUARTER_OFFSET,
    CROP_SCALE,
    HEATMAP_SIGMA,
    USE_DISK_CACHE,
    CACHE_DIR,
)

from datasets.deeplake_300w import DeepLake300W
from models.HRNet import hrnet_w18_face


CHECKPOINT_PATH = CHECKPOINT_DIR / "hrnet_epoch_170.pth"
OUT_PATH = TEST_DIR / "ced_epoch_170.png"
CED_THRESHOLD = 0.08


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    new_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_dict[k] = v

    model.load_state_dict(new_dict, strict=True)
    return model


def heatmaps_to_pts(heatmaps, img_size, use_quarter_offset=True):
    B, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, K, -1)

    idx = torch.argmax(flat, dim=2)
    y = (idx // W).float()
    x = (idx % W).float()

    if use_quarter_offset:
        for b in range(B):
            for k in range(K):
                px = int(x[b, k])
                py = int(y[b, k])

                if 1 <= px < W - 1 and 1 <= py < H - 1:
                    dx = heatmaps[b, k, py, px + 1] - heatmaps[b, k, py, px - 1]
                    dy = heatmaps[b, k, py + 1, px] - heatmaps[b, k, py - 1, px]

                    x[b, k] += torch.sign(dx) * 0.25
                    y[b, k] += torch.sign(dy) * 0.25

    x = x * img_size / W
    y = y * img_size / H

    return torch.stack([x, y], dim=2)


def compute_norm_factor(gt, norm_type="inter_ocular"):
    if norm_type == "inter_ocular":
        return torch.norm(gt[:, 36] - gt[:, 45], dim=1)

    if norm_type == "inter_eye":
        left = gt[:, 36:42].mean(1)
        right = gt[:, 42:48].mean(1)
        return torch.norm(left - right, dim=1)

    if norm_type == "bbox_diag":
        x_min = gt[:, :, 0].min(dim=1).values
        y_min = gt[:, :, 1].min(dim=1).values
        x_max = gt[:, :, 0].max(dim=1).values
        y_max = gt[:, :, 1].max(dim=1).values
        return torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

    raise ValueError(f"Unknown norm type: {norm_type}")


def compute_nme(pred, gt, norm_type="inter_ocular"):
    norm = compute_norm_factor(gt, norm_type=norm_type)
    error = torch.norm(pred - gt, dim=2).mean(dim=1)
    return error / (norm + 1e-6)


def compute_ced(errors, max_threshold=0.08, num_points=1000):
    errors = np.asarray(errors)
    xs = np.linspace(0, max_threshold, num_points)
    ys = np.array([np.mean(errors <= x) for x in xs])
    auc = np.trapezoid(ys, xs) / max_threshold
    return xs, ys, auc


def main():
    os.makedirs(TEST_DIR, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    dataset = DeepLake300W(
        DATASET_PATH,
        split="test",
        img_size=IMG_SIZE,
        heatmap_size=HEATMAP_SIZE,
        crop_scale=CROP_SCALE,
        heatmap_sigma=HEATMAP_SIGMA,
        use_disk_cache=USE_DISK_CACHE,
        cache_dir=CACHE_DIR,
    )

    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    model = hrnet_w18_face(num_landmarks=NUM_LANDMARKS)
    model = load_checkpoint(model, CHECKPOINT_PATH, device)
    model = model.to(device)
    model.eval()

    all_nme = []

    with torch.no_grad():
        for imgs, _gt_hm, gt_pts in loader:
            imgs = imgs.to(device, non_blocking=True)
            gt_pts = gt_pts.to(device, non_blocking=True)

            pred_hm = model(imgs)
            pred_pts = heatmaps_to_pts(
                pred_hm,
                IMG_SIZE,
                use_quarter_offset=USE_QUARTER_OFFSET,
            )

            nme = compute_nme(pred_pts, gt_pts, norm_type=NME_NORM_TYPE)
            all_nme.extend(nme.cpu().numpy())

    all_nme = np.asarray(all_nme)

    xs, ys, auc = compute_ced(all_nme, max_threshold=CED_THRESHOLD)

    print(f"Mean NME: {all_nme.mean():.6f}")
    print(f"Median NME: {np.median(all_nme):.6f}")
    print(f"Failure@{CED_THRESHOLD}: {np.mean(all_nme > CED_THRESHOLD) * 100:.2f}%")
    print(f"AUC@{CED_THRESHOLD}: {auc:.6f}")

    sns.set_theme(style="white")

    plt.figure(figsize=(7, 5))
    ax = sns.lineplot(x=xs, y=ys, linewidth=3.5)

    ax.set_xlabel("NME")
    ax.set_ylabel("Proportion of Test Samples")
    ax.grid(False)

    plt.xlim(0, CED_THRESHOLD)
    plt.ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    plt.close()

    print(f"Saved CED curve to: {OUT_PATH}")


if __name__ == "__main__":
    main()