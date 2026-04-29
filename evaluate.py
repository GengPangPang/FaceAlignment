import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import DATASET_PATH, IMG_SIZE, HEATMAP_SIZE
from datasets.deeplake_300w import DeepLake300W

# 选择你的模型
from models.HRNet import hrnet_w18_face

CHECKPOINT_PATH = "checkpoints/hrnet_epoch_200.pth"
# CHECKPOINT_PATH = "checkpoints/hrnet_epoch_10.pth"
# OUT_DIR = "test/full_eval"
OUT_DIR = "test/full_eval_epoch_200"
BATCH_SIZE = 16
NUM_LANDMARKS = 68


# -------------------------------
# 工具函数
# -------------------------------

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


def heatmaps_to_pts(heatmaps, img_size):
    B, K, H, W = heatmaps.shape
    heatmaps = heatmaps.reshape(B, K, -1)

    idx = torch.argmax(heatmaps, dim=2)
    y = idx // W
    x = idx % W

    x = x.float() * img_size / W
    y = y.float() * img_size / H

    return torch.stack([x, y], dim=2)


def compute_nme(pred, gt):
    left = gt[:, 36:42].mean(1)
    right = gt[:, 42:48].mean(1)

    norm = torch.norm(left - right, dim=1)
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

    # GT 绿色
    for x, y in gt:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

    # Pred 红色
    for x, y in pred:
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    return img


# 68点区域划分
REGIONS = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose": list(range(27, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "mouth": list(range(48, 68)),
}


# -------------------------------
# 主评估
# -------------------------------

def evaluate():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DeepLake300W(
        DATASET_PATH,
        split="test",
        img_size=IMG_SIZE,
        heatmap_size=HEATMAP_SIZE,
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 模型
    model = hrnet_w18_face(num_landmarks=NUM_LANDMARKS)
    # model = HRNetSmall(num_landmarks=NUM_LANDMARKS)

    model = load_checkpoint(model, CHECKPOINT_PATH, device)
    model = model.to(device)
    model.eval()

    all_nme = []
    all_point_error = []

    with torch.no_grad():
        for batch_idx, (imgs, gt_hm) in enumerate(loader):
            imgs = imgs.to(device)
            gt_hm = gt_hm.to(device)

            pred_hm = model(imgs)

            pred_pts = heatmaps_to_pts(pred_hm, IMG_SIZE)
            gt_pts = heatmaps_to_pts(gt_hm, IMG_SIZE)

            nme = compute_nme(pred_pts, gt_pts)
            all_nme.extend(nme.cpu().numpy())

            # 每点误差
            point_err = torch.norm(pred_pts - gt_pts, dim=2)
            all_point_error.append(point_err.cpu().numpy())

            # 保存可视化
            if batch_idx < 2:
                imgs_np = imgs.cpu().numpy()
                pred_np = pred_pts.cpu().numpy()
                gt_np = gt_pts.cpu().numpy()

                for i in range(min(4, imgs_np.shape[0])):
                    img = (imgs_np[i].transpose(1, 2, 0) * 255).astype(np.uint8)

                    vis = draw(img, pred_np[i], gt_np[i])

                    cv2.imwrite(
                        os.path.join(OUT_DIR, f"vis_{batch_idx}_{i}.jpg"),
                        vis,
                    )

    all_nme = np.array(all_nme)

    # -------------------------------
    # 指标计算
    # -------------------------------

    mean_nme = np.mean(all_nme)
    median_nme = np.median(all_nme)

    failure_008 = np.mean(all_nme > 0.08)

    auc_008, xs, ys = compute_auc(all_nme, 0.08)

    save_ced(xs, ys, os.path.join(OUT_DIR, "ced_curve.png"))

    # Per-region
    all_point_error = np.concatenate(all_point_error, axis=0)

    region_nme = {}
    for name, idxs in REGIONS.items():
        region_err = all_point_error[:, idxs].mean()
        region_nme[name] = region_err

    # -------------------------------
    # 输出结果
    # -------------------------------

    print("=" * 60)
    print(f"Mean NME:   {mean_nme:.6f}")
    print(f"Median NME: {median_nme:.6f}")
    print(f"Failure@0.08: {failure_008 * 100:.2f}%")
    print(f"AUC@0.08: {auc_008:.6f}")

    print("\nPer-region error:")
    for k, v in region_nme.items():
        print(f"{k:15s}: {v:.4f}")

    print("=" * 60)

    # 保存到文件
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"Mean NME: {mean_nme}\n")
        f.write(f"Median NME: {median_nme}\n")
        f.write(f"Failure@0.08: {failure_008}\n")
        f.write(f"AUC@0.08: {auc_008}\n\n")

        f.write("Per-region:\n")
        for k, v in region_nme.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    evaluate()