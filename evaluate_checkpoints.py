import os
import csv
import torch
import numpy as np
from tqdm import tqdm
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
    FAILURE_THRESHOLD,
    AUC_THRESHOLD,
    USE_QUARTER_OFFSET,
    CROP_SCALE,
    HEATMAP_SIGMA,
    USE_DISK_CACHE,
    CACHE_DIR,
)

from datasets.deeplake_300w import DeepLake300W
from models.HRNet import hrnet_w18_face


# 你要评估的 checkpoint 范围
START_EPOCH = 100
END_EPOCH = 200
STEP = 10

OUT_DIR = TEST_DIR / "checkpoint_eval_100_200"
OUT_CSV = OUT_DIR / "checkpoint_metrics.csv"


REGIONS = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose": list(range(27, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "mouth": list(range(48, 68)),
}


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

    epoch = ckpt.get("epoch", None) if isinstance(ckpt, dict) else None
    avg_loss = ckpt.get("avg_loss", None) if isinstance(ckpt, dict) else None
    best_loss = ckpt.get("best_loss", None) if isinstance(ckpt, dict) else None
    best_epoch = ckpt.get("best_epoch", None) if isinstance(ckpt, dict) else None

    return model, epoch, avg_loss, best_loss, best_epoch


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

    raise ValueError(f"Unknown NME norm type: {norm_type}")


def compute_nme(pred, gt, norm_type="inter_ocular"):
    norm = compute_norm_factor(gt, norm_type=norm_type)
    error = torch.norm(pred - gt, dim=2).mean(1)
    return error / (norm + 1e-6)


def compute_auc(errors, max_threshold=0.08):
    errors = np.array(errors)
    xs = np.linspace(0, max_threshold, 1000)
    ys = [np.mean(errors <= t) for t in xs]
    auc = np.trapezoid(ys, xs) / max_threshold
    return auc


def evaluate_one_checkpoint(checkpoint_path, loader, device):
    model = hrnet_w18_face(num_landmarks=NUM_LANDMARKS)
    model, ckpt_epoch, train_avg_loss, train_best_loss, train_best_epoch = load_checkpoint(
        model,
        checkpoint_path,
        device,
    )
    model = model.to(device)
    model.eval()

    all_nme = []
    all_point_error_px = []
    all_point_error_norm = []

    with torch.no_grad():
        for imgs, _gt_hm, gt_pts in tqdm(loader, desc=f"Eval {checkpoint_path.name}", leave=False):
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

            point_err_px = torch.norm(pred_pts - gt_pts, dim=2)
            norm = compute_norm_factor(gt_pts, norm_type=NME_NORM_TYPE)
            point_err_norm = point_err_px / (norm[:, None] + 1e-6)

            all_point_error_px.append(point_err_px.cpu().numpy())
            all_point_error_norm.append(point_err_norm.cpu().numpy())

    all_nme = np.array(all_nme)
    all_point_error_px = np.concatenate(all_point_error_px, axis=0)
    all_point_error_norm = np.concatenate(all_point_error_norm, axis=0)

    result = {
        "checkpoint": str(checkpoint_path),
        "file_epoch": checkpoint_path.stem.replace("hrnet_epoch_", ""),
        "ckpt_epoch": ckpt_epoch,
        "train_avg_loss": train_avg_loss,
        "train_best_loss": train_best_loss,
        "train_best_epoch": train_best_epoch,
        "mean_nme": float(np.mean(all_nme)),
        "median_nme": float(np.median(all_nme)),
        "failure_rate": float(np.mean(all_nme > FAILURE_THRESHOLD)),
        "auc": float(compute_auc(all_nme, AUC_THRESHOLD)),
    }

    for name, idxs in REGIONS.items():
        result[f"{name}_px"] = float(all_point_error_px[:, idxs].mean())
        result[f"{name}_norm"] = float(all_point_error_norm[:, idxs].mean())

    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    print(f"NME norm type: {NME_NORM_TYPE}")
    print(f"Use quarter offset: {USE_QUARTER_OFFSET}")
    print(f"Output CSV: {OUT_CSV}")

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
        num_workers=0,  # DeepLake 评估阶段先用 0，更稳定
        pin_memory=(DEVICE == "cuda"),
    )

    results = []

    for epoch in range(START_EPOCH, END_EPOCH + 1, STEP):
        checkpoint_path = CHECKPOINT_DIR / f"hrnet_epoch_{epoch}.pth"

        if not checkpoint_path.exists():
            print(f"[WARN] checkpoint not found, skip: {checkpoint_path}")
            continue

        result = evaluate_one_checkpoint(checkpoint_path, loader, device)
        results.append(result)

        print(
            f"epoch={epoch:03d} | "
            f"Mean NME={result['mean_nme']:.6f} | "
            f"Median={result['median_nme']:.6f} | "
            f"Failure@{FAILURE_THRESHOLD}={result['failure_rate'] * 100:.2f}% | "
            f"AUC@{AUC_THRESHOLD}={result['auc']:.6f}"
        )

    if not results:
        print("No checkpoints evaluated.")
        return

    fieldnames = list(results[0].keys())

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    best_by_nme = min(results, key=lambda x: x["mean_nme"])
    best_by_failure = min(results, key=lambda x: x["failure_rate"])

    print("=" * 80)
    print("Best checkpoint by Mean NME:")
    print(
        f"  {best_by_nme['checkpoint']} | "
        f"Mean NME={best_by_nme['mean_nme']:.6f} | "
        f"Failure={best_by_nme['failure_rate'] * 100:.2f}% | "
        f"AUC={best_by_nme['auc']:.6f}"
    )

    print("Best checkpoint by Failure Rate:")
    print(
        f"  {best_by_failure['checkpoint']} | "
        f"Mean NME={best_by_failure['mean_nme']:.6f} | "
        f"Failure={best_by_failure['failure_rate'] * 100:.2f}% | "
        f"AUC={best_by_failure['auc']:.6f}"
    )

    print(f"Saved CSV to: {OUT_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()