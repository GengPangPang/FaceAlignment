import os
import cv2
import torch
import numpy as np

from config import (
    DATASET_PATH,
    IMG_SIZE,
    HEATMAP_SIZE,
    NUM_LANDMARKS,
    DEVICE,
    ALIGN_CHECKPOINT_PATH,
    ALIGN_OUT_DIR,
    ALIGN_OUTPUT_SIZE,
    ALIGN_INDEX_START,
    ALIGN_INDEX_END,
    ALIGN_TEMPLATE_112,
    USE_QUARTER_OFFSET,
    CROP_SCALE,
    HEATMAP_SIGMA,
    USE_DISK_CACHE,
    CACHE_DIR,
)
from datasets.deeplake_300w import DeepLake300W
from models.HRNet import hrnet_w18_face


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


def draw_points(img_rgb, pred_pts, gt_pts=None):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if gt_pts is not None:
        for x, y in gt_pts:
            cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 255, 0), -1)

    for x, y in pred_pts:
        cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 0, 255), -1)

    return img_bgr


def get_5_points_from_68(pts):
    left_eye = pts[36:42].mean(axis=0)
    right_eye = pts[42:48].mean(axis=0)
    nose = pts[30]
    left_mouth = pts[48]
    right_mouth = pts[54]

    return np.array(
        [left_eye, right_eye, nose, left_mouth, right_mouth],
        dtype=np.float32,
    )


def get_alignment_template(output_size=112):
    dst = np.array(ALIGN_TEMPLATE_112, dtype=np.float32)

    if output_size != 112:
        dst = dst * (output_size / 112.0)

    return dst


def estimate_align_matrix(src5, output_size=112):
    dst5 = get_alignment_template(output_size)

    M, _ = cv2.estimateAffinePartial2D(
        src5,
        dst5,
        method=cv2.LMEDS,
    )

    if M is None:
        raise RuntimeError("cv2.estimateAffinePartial2D failed to estimate a transform.")

    return M, dst5


def warp_points(pts, M):
    pts_h = np.hstack(
        [
            pts,
            np.ones((pts.shape[0], 1), dtype=np.float32),
        ]
    )
    return pts_h @ M.T


def eye_angle_deg(pts5):
    left_eye, right_eye = pts5[0], pts5[1]
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    return float(np.degrees(np.arctan2(dy, dx)))


def reprojection_error(src5, dst5, M):
    proj = warp_points(src5, M)
    return float(np.linalg.norm(proj - dst5, axis=1).mean())


def align_face(img_rgb, pts68, output_size=112):
    src5 = get_5_points_from_68(pts68)
    M, dst5 = estimate_align_matrix(src5, output_size=output_size)

    aligned = cv2.warpAffine(
        img_rgb,
        M,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )

    return aligned, M, src5, dst5


def predict_one(index, model, dataset, device):
    os.makedirs(ALIGN_OUT_DIR, exist_ok=True)

    img_tensor, _gt_hm, gt_pts_tensor = dataset[index]
    gt_pts = gt_pts_tensor.numpy()

    img_batch = img_tensor.unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad():
        pred_hm = model(img_batch)

    pred_pts = heatmaps_to_pts(
        pred_hm,
        IMG_SIZE,
        use_quarter_offset=USE_QUARTER_OFFSET,
    )[0].cpu().numpy()

    img_rgb = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    pred_vis = draw_points(img_rgb, pred_pts, gt_pts)
    aligned_rgb, pred_M, pred5, dst5 = align_face(
        img_rgb,
        pred_pts,
        output_size=ALIGN_OUTPUT_SIZE,
    )

    gt5 = get_5_points_from_68(gt_pts)
    gt_M, _ = estimate_align_matrix(gt5, output_size=ALIGN_OUTPUT_SIZE)

    metrics = {
        "pred_reprojection_error_px": reprojection_error(pred5, dst5, pred_M),
        "gt_reprojection_error_px": reprojection_error(gt5, dst5, gt_M),
        "gt_under_predM_reprojection_error_px": reprojection_error(gt5, dst5, pred_M),
        "pred_eye_angle_before_abs_deg": abs(eye_angle_deg(pred5)),
        "pred_eye_angle_after_abs_deg": abs(eye_angle_deg(warp_points(pred5, pred_M))),
        "gt_eye_angle_before_abs_deg": abs(eye_angle_deg(gt5)),
        "gt_eye_angle_after_by_predM_abs_deg": abs(eye_angle_deg(warp_points(gt5, pred_M))),
    }

    crop_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(ALIGN_OUT_DIR, f"{index}_crop_input.jpg"), crop_bgr)
    cv2.imwrite(os.path.join(ALIGN_OUT_DIR, f"{index}_pred_vs_gt.jpg"), pred_vis)
    cv2.imwrite(os.path.join(ALIGN_OUT_DIR, f"{index}_aligned.jpg"), aligned_bgr)

    metrics_path = os.path.join(ALIGN_OUT_DIR, f"{index}_alignment_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"checkpoint: {ALIGN_CHECKPOINT_PATH}\n")
        f.write(f"output_size: {ALIGN_OUTPUT_SIZE}\n")
        f.write(f"use_quarter_offset: {USE_QUARTER_OFFSET}\n")
        f.write(f"template_112: {ALIGN_TEMPLATE_112}\n\n")

        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"saved index {index} outputs to: {ALIGN_OUT_DIR}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


def main():
    device = torch.device(DEVICE)

    print("device:", device)
    print(f"checkpoint: {ALIGN_CHECKPOINT_PATH}")
    print(f"align output dir: {ALIGN_OUT_DIR}")
    print(f"disk cache enabled: {USE_DISK_CACHE}")
    print(f"cache dir: {CACHE_DIR}")

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

    model = hrnet_w18_face(num_landmarks=NUM_LANDMARKS)
    model = load_checkpoint(model, ALIGN_CHECKPOINT_PATH, device)
    model = model.to(device)
    model.eval()

    for i in range(ALIGN_INDEX_START, ALIGN_INDEX_END):
        predict_one(index=i, model=model, dataset=dataset, device=device)


if __name__ == "__main__":
    main()