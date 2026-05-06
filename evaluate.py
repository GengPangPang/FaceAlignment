import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import (
    DATASET_PATH,
    IMG_SIZE,
    HEATMAP_SIZE,
    NUM_LANDMARKS,
    DEVICE,
    EVAL_CHECKPOINT_PATH,
    EVAL_OUT_DIR,
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
    return model


def heatmaps_to_pts(heatmaps, img_size, use_quarter_offset=True):
    """Decode predicted heatmaps to image-coordinate points.

    GT points should not use this function during evaluation.
    GT points should come directly from the dataset as true landmarks.
    """
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
    auc = np.trapz(ys, xs) / max_threshold
    return auc, xs, ys


def save_ced(xs, ys, path):
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("NME")
    plt.ylabel("Proportion")
    plt.title("CED Curve")
    plt.grid(True)
    plt.savefig(path)
    plt.close()


def draw(img_rgb, pred, gt):
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    for x, y in gt:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

    for x, y in pred:
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    return img


def evaluate():
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    print(f"Checkpoint: {EVAL_CHECKPOINT_PATH}")
    print(f"Eval output dir: {EVAL_OUT_DIR}")
    print(f"Disk cache enabled: {USE_DISK_CACHE}")
    print(f"Cache dir: {CACHE_DIR}")

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
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
    )

    model = hrnet_w18_face(num_landmarks=NUM_LANDMARKS)
    model = load_checkpoint(model, EVAL_CHECKPOINT_PATH, device)
    model = model.to(device)
    model.eval()

    all_nme = []
    all_point_error_px = []
    all_point_error_norm = []

    with torch.no_grad():
        for batch_idx, (imgs, _gt_hm, gt_pts) in enumerate(loader):
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

            if batch_idx < 2:
                imgs_np = imgs.cpu().numpy()
                pred_np = pred_pts.cpu().numpy()
                gt_np = gt_pts.cpu().numpy()

                for i in range(min(4, imgs_np.shape[0])):
                    img = (imgs_np[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                    vis = draw(img, pred_np[i], gt_np[i])
                    cv2.imwrite(os.path.join(EVAL_OUT_DIR, f"vis_{batch_idx}_{i}.jpg"), vis)

    all_nme = np.array(all_nme)

    mean_nme = np.mean(all_nme)
    median_nme = np.median(all_nme)
    failure = np.mean(all_nme > FAILURE_THRESHOLD)

    auc, xs, ys = compute_auc(all_nme, AUC_THRESHOLD)
    save_ced(xs, ys, os.path.join(EVAL_OUT_DIR, "ced_curve.png"))

    all_point_error_px = np.concatenate(all_point_error_px, axis=0)
    all_point_error_norm = np.concatenate(all_point_error_norm, axis=0)

    region_px = {}
    region_norm = {}

    for name, idxs in REGIONS.items():
        region_px[name] = all_point_error_px[:, idxs].mean()
        region_norm[name] = all_point_error_norm[:, idxs].mean()

    print("=" * 60)
    print(f"Checkpoint: {EVAL_CHECKPOINT_PATH}")
    print(f"NME norm type: {NME_NORM_TYPE}")
    print(f"Use quarter offset: {USE_QUARTER_OFFSET}")
    print(f"Mean NME:   {mean_nme:.6f}")
    print(f"Median NME: {median_nme:.6f}")
    print(f"Failure@{FAILURE_THRESHOLD}: {failure * 100:.2f}%")
    print(f"AUC@{AUC_THRESHOLD}: {auc:.6f}")

    print("\nPer-region pixel error:")
    for k, v in region_px.items():
        print(f"{k:15s}: {v:.4f}")

    print("\nPer-region normalized error:")
    for k, v in region_norm.items():
        print(f"{k:15s}: {v:.6f}")
    print("=" * 60)

    metrics_path = os.path.join(EVAL_OUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Checkpoint: {EVAL_CHECKPOINT_PATH}\n")
        f.write(f"NME norm type: {NME_NORM_TYPE}\n")
        f.write(f"Use quarter offset: {USE_QUARTER_OFFSET}\n")
        f.write(f"Mean NME: {mean_nme}\n")
        f.write(f"Median NME: {median_nme}\n")
        f.write(f"Failure@{FAILURE_THRESHOLD}: {failure}\n")
        f.write(f"AUC@{AUC_THRESHOLD}: {auc}\n\n")

        f.write("Per-region pixel error:\n")
        for k, v in region_px.items():
            f.write(f"{k}: {v}\n")

        f.write("\nPer-region normalized error:\n")
        for k, v in region_norm.items():
            f.write(f"{k}: {v}\n")

    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    evaluate()