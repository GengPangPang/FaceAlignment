import os
import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import (
    DATASET_PATH,
    IMG_SIZE,
    HEATMAP_SIZE,
    NUM_LANDMARKS,
    DEVICE,
    EVAL_CHECKPOINT_PATH,
    CROP_SCALE,
    HEATMAP_SIGMA,
    USE_DISK_CACHE,
    CACHE_DIR,
)

from datasets.deeplake_300w import DeepLake300W
from models.HRNet import hrnet_w18_face


# =========================================================
# 配置区：你主要改这里
# =========================================================
OUT_DIR = "test/heatmap_vis_report"

SAMPLE_INDEX = 0

# 二选一：
# 1) 直接指定区域名字（推荐）
REGION_NAME = "jaw"   # 可选: jaw / right_eyebrow / left_eyebrow / eyebrows / nose / right_eye / left_eye / eyes / mouth / all

# 2) 或者手动指定点索引（如果不为 None，则优先使用这个）
LANDMARK_INDICES = None
# LANDMARK_INDICES = [0, 4, 8, 12, 16]

# 报告图里最多显示多少个 heatmap 子图
MAX_HEATMAPS_SHOW = 8

# overlay 透明度
OVERLAY_ALPHA = 0.45


# =========================================================
# 区域定义
# =========================================================
REGION_MAP = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "eyebrows": list(range(17, 27)),
    "nose": list(range(27, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "eyes": list(range(36, 48)),
    "mouth": list(range(48, 68)),
    "all": list(range(0, 68)),
}


# =========================================================
# 基础函数
# =========================================================
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


def get_landmark_indices(region_name=None, landmark_indices=None):
    if landmark_indices is not None:
        return landmark_indices

    if region_name is None:
        return [30]

    if region_name not in REGION_MAP:
        raise ValueError(
            f"Unknown REGION_NAME: {region_name}. "
            f"Available: {list(REGION_MAP.keys())}"
        )
    return REGION_MAP[region_name]


def heatmaps_to_pts(heatmaps, img_size=256, use_offset=True):
    """
    heatmaps: [B, K, H, W]
    return: [B, K, 2]
    """
    B, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, K, -1)

    idx = torch.argmax(flat, dim=2)
    y = (idx // W).float()
    x = (idx % W).float()

    if use_offset:
        for b in range(B):
            for k in range(K):
                px = int(x[b, k].item())
                py = int(y[b, k].item())

                if 1 <= px < W - 1 and 1 <= py < H - 1:
                    dx = heatmaps[b, k, py, px + 1] - heatmaps[b, k, py, px - 1]
                    dy = heatmaps[b, k, py + 1, px] - heatmaps[b, k, py - 1, px]

                    x[b, k] += torch.sign(dx) * 0.25
                    y[b, k] += torch.sign(dy) * 0.25

    x = x * img_size / W
    y = y * img_size / H

    return torch.stack([x, y], dim=2)


def normalize_heatmap(hm):
    hm = hm.astype(np.float32)
    hm_min = hm.min()
    hm_max = hm.max()

    if hm_max - hm_min < 1e-8:
        return np.zeros_like(hm)

    return (hm - hm_min) / (hm_max - hm_min)


def overlay_heatmap_on_image(img_rgb, hm_64, alpha=0.45):
    hm_norm = normalize_heatmap(hm_64)
    hm_resized = cv2.resize(hm_norm, (img_rgb.shape[1], img_rgb.shape[0]))

    heatmap_uint8 = np.uint8(255 * hm_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def combine_heatmaps(heatmaps, landmark_indices):
    """
    heatmaps: [K, H, W] tensor
    把多个关键点 heatmap 取逐像素最大值，用于整体 overlay
    """
    selected = [heatmaps[idx].numpy() for idx in landmark_indices]
    combined = np.maximum.reduce(selected)
    return combined


def draw_points(img_rgb, points, indices=None, color=(255, 0, 0), radius=3, draw_text=True):
    img = img_rgb.copy()

    for i, pt in enumerate(points):
        x, y = pt
        x = int(round(float(x)))
        y = int(round(float(y)))

        cv2.circle(img, (x, y), radius, color, -1)

        if draw_text:
            text = str(indices[i]) if indices is not None else str(i)
            cv2.putText(
                img,
                text,
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    return img


def draw_gt_pred_points(img_rgb, gt_points, pred_points, indices=None, radius=3):
    """
    GT: 绿色
    Pred: 红色
    """
    img = img_rgb.copy()

    for i, (gt_pt, pred_pt) in enumerate(zip(gt_points, pred_points)):
        gx, gy = int(round(float(gt_pt[0]))), int(round(float(gt_pt[1])))
        px, py = int(round(float(pred_pt[0]))), int(round(float(pred_pt[1])))

        cv2.circle(img, (gx, gy), radius, (0, 255, 0), -1)   # GT green
        cv2.circle(img, (px, py), radius, (255, 0, 0), -1)   # Pred red

        if indices is not None:
            cv2.putText(
                img,
                str(indices[i]),
                (gx + 4, gy - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    return img


def build_heatmap_grid_image(heatmaps, landmark_indices, title_prefix="GT", max_show=8):
    """
    heatmaps: [K, H, W] tensor
    返回一个 RGB numpy 图像，用于后续拼接进报告图
    """
    show_indices = landmark_indices[:max_show]
    n = len(show_indices)

    cols = min(4, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    axes = axes.reshape(rows, cols)

    for plot_idx in range(rows * cols):
        r = plot_idx // cols
        c = plot_idx % cols
        ax = axes[r, c]

        if plot_idx < n:
            lm_idx = show_indices[plot_idx]
            hm = heatmaps[lm_idx].numpy()
            ax.imshow(hm, cmap="jet")
            ax.set_title(f"{title_prefix} lm {lm_idx}", fontsize=10)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = img[:, :, :3].copy()

    plt.close(fig)
    return img


def save_image_rgb(path, img_rgb):
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def make_report_figure(
    save_path,
    points_img,
    gt_overlay_img,
    gt_grid_img,
    pred_grid_img,
    sample_index,
    region_name,
    landmark_indices,
):
    """
    四宫格报告图：
    1. GT/Pred点图
    2. GT overlay
    3. GT heatmap grid
    4. Pred heatmap grid
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(points_img)
    axes[0, 0].set_title("Input Image with GT (green) and Pred (red) Points", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_overlay_img)
    axes[0, 1].set_title("Combined GT Heatmap Overlay", fontsize=12)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(gt_grid_img)
    axes[1, 0].set_title("GT Heatmaps", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_grid_img)
    axes[1, 1].set_title("Pred Heatmaps", fontsize=12)
    axes[1, 1].axis("off")

    fig.suptitle(
        f"Heatmap Visualization Report | sample={sample_index} | region={region_name} | landmarks={landmark_indices}",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


# =========================================================
# 主流程
# =========================================================
def visualize_heatmap_report(
    sample_index=0,
    region_name="jaw",
    landmark_indices=None,
    max_heatmaps_show=8,
):
    os.makedirs(OUT_DIR, exist_ok=True)

    landmark_indices = get_landmark_indices(region_name, landmark_indices)

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

    img_tensor, gt_hm, gt_pts = dataset[sample_index]

    model = hrnet_w18_face(NUM_LANDMARKS).to(DEVICE)
    model = load_checkpoint(model, EVAL_CHECKPOINT_PATH, DEVICE)
    model.eval()

    img_batch = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_hm = model(img_batch)

    pred_hm_cpu = pred_hm[0].detach().cpu()   # [68, 64, 64]
    gt_hm_cpu = gt_hm.detach().cpu()          # [68, 64, 64]

    pred_pts = heatmaps_to_pts(pred_hm.detach().cpu(), IMG_SIZE, use_offset=True)[0]
    gt_pts = gt_pts.detach().cpu()

    img_rgb = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    selected_gt_pts = gt_pts[landmark_indices].numpy()
    selected_pred_pts = pred_pts[landmark_indices].numpy()

    region_tag = region_name if region_name is not None else "custom"
    indices_tag = "_".join(str(i) for i in landmark_indices[:12])
    if len(landmark_indices) > 12:
        indices_tag += "_etc"

    base_name = f"sample{sample_index}_{region_tag}_{indices_tag}"

    # 1) 原图 + GT/PRED 点图
    points_img = draw_gt_pred_points(
        img_rgb,
        selected_gt_pts,
        selected_pred_pts,
        indices=landmark_indices,
        radius=3,
    )
    save_image_rgb(os.path.join(OUT_DIR, f"{base_name}_points.jpg"), points_img)

    # 2) GT overlay / Pred overlay
    combined_gt_hm = combine_heatmaps(gt_hm_cpu, landmark_indices)
    combined_pred_hm = combine_heatmaps(pred_hm_cpu, landmark_indices)

    gt_overlay_img = overlay_heatmap_on_image(img_rgb, combined_gt_hm, alpha=OVERLAY_ALPHA)
    pred_overlay_img = overlay_heatmap_on_image(img_rgb, combined_pred_hm, alpha=OVERLAY_ALPHA)

    save_image_rgb(os.path.join(OUT_DIR, f"{base_name}_gt_overlay.jpg"), gt_overlay_img)
    save_image_rgb(os.path.join(OUT_DIR, f"{base_name}_pred_overlay.jpg"), pred_overlay_img)

    # 3) GT heatmap grid / Pred heatmap grid
    gt_grid_img = build_heatmap_grid_image(
        gt_hm_cpu,
        landmark_indices,
        title_prefix="GT",
        max_show=max_heatmaps_show,
    )
    pred_grid_img = build_heatmap_grid_image(
        pred_hm_cpu,
        landmark_indices,
        title_prefix="Pred",
        max_show=max_heatmaps_show,
    )

    save_image_rgb(os.path.join(OUT_DIR, f"{base_name}_gt_heatmap_grid.jpg"), gt_grid_img)
    save_image_rgb(os.path.join(OUT_DIR, f"{base_name}_pred_heatmap_grid.jpg"), pred_grid_img)

    # 4) 报告四宫格大图
    report_fig_path = os.path.join(OUT_DIR, f"{base_name}_report_4panel.png")
    make_report_figure(
        save_path=report_fig_path,
        points_img=points_img,
        gt_overlay_img=gt_overlay_img,
        gt_grid_img=gt_grid_img,
        pred_grid_img=pred_grid_img,
        sample_index=sample_index,
        region_name=region_tag,
        landmark_indices=landmark_indices[:max_heatmaps_show],
    )

    # 5) 保存文字误差信息
    info_path = os.path.join(OUT_DIR, f"{base_name}_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"Sample index: {sample_index}\n")
        f.write(f"Region name: {region_tag}\n")
        f.write(f"Landmark indices: {landmark_indices}\n\n")

        mean_err = 0.0
        for lm_idx, gt_pt, pred_pt in zip(landmark_indices, selected_gt_pts, selected_pred_pts):
            err = np.linalg.norm(pred_pt - gt_pt)
            mean_err += err
            f.write(
                f"lm={lm_idx:2d} | "
                f"GT=({gt_pt[0]:7.3f}, {gt_pt[1]:7.3f}) | "
                f"Pred=({pred_pt[0]:7.3f}, {pred_pt[1]:7.3f}) | "
                f"Error={err:7.4f}\n"
            )

        mean_err /= len(landmark_indices)
        f.write(f"\nMean pixel error of selected landmarks: {mean_err:.4f}\n")

    print("=" * 80)
    print("Heatmap report visualization finished.")
    print(f"Output dir: {OUT_DIR}")
    print(f"Sample index: {sample_index}")
    print(f"Region name: {region_tag}")
    print(f"Landmark indices: {landmark_indices}")
    print(f"Report figure: {report_fig_path}")
    print("=" * 80)


if __name__ == "__main__":
    visualize_heatmap_report(
        sample_index=SAMPLE_INDEX,
        region_name=REGION_NAME,
        landmark_indices=LANDMARK_INDICES,
        max_heatmaps_show=MAX_HEATMAPS_SHOW,
    )