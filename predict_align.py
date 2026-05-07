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
    """Decode predicted heatmaps to image-coordinate landmarks.

    GT landmarks should come directly from Dataset, not from GT heatmaps.
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


def draw_points(img_rgb, pred_pts, gt_pts=None):
    """Draw predicted and GT landmarks.

    Red: prediction.
    Green: GT.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if gt_pts is not None:
        for x, y in gt_pts:
            cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 255, 0), -1)

    for x, y in pred_pts:
        cv2.circle(img_bgr, (int(x), int(y)), 2, (0, 0, 255), -1)

    return img_bgr


def get_5_points_from_68(pts68):
    """Extract 5 alignment landmarks from 68 landmarks.

    Order:
        0. left eye center
        1. right eye center
        2. nose tip
        3. left mouth corner
        4. right mouth corner
    """
    left_eye = pts68[36:42].mean(axis=0)
    right_eye = pts68[42:48].mean(axis=0)
    nose = pts68[30]
    left_mouth = pts68[48]
    right_mouth = pts68[54]

    return np.array(
        [left_eye, right_eye, nose, left_mouth, right_mouth],
        dtype=np.float32,
    )


def get_alignment_template(output_size=112):
    """Return 5-point alignment template for requested output size."""
    dst = np.array(ALIGN_TEMPLATE_112, dtype=np.float32)

    if output_size != 112:
        dst = dst * (output_size / 112.0)

    return dst


def estimate_align_matrix(src5, output_size=112):
    """Estimate partial affine / similarity transform from source 5 points to template."""
    dst5 = get_alignment_template(output_size)

    M, inliers = cv2.estimateAffinePartial2D(
        src5,
        dst5,
        method=cv2.LMEDS,
    )

    if M is None:
        raise RuntimeError("cv2.estimateAffinePartial2D failed to estimate a transform.")

    return M, dst5, inliers


def warp_points(pts, M):
    """Apply 2x3 affine matrix to 2D points."""
    pts = np.asarray(pts, dtype=np.float32)
    pts_h = np.hstack(
        [
            pts,
            np.ones((pts.shape[0], 1), dtype=np.float32),
        ]
    )
    return pts_h @ M.T


def eye_angle_deg(pts5):
    """Angle between two eye centers and the horizontal axis."""
    left_eye, right_eye = pts5[0], pts5[1]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    return float(np.degrees(np.arctan2(dy, dx)))


def reprojection_error(src5, dst5, M):
    """Mean Euclidean distance between transformed source 5 points and template 5 points."""
    proj = warp_points(src5, M)
    return float(np.linalg.norm(proj - dst5, axis=1).mean())


def compute_alignment_metrics(pred5, gt5, dst5, pred_M, gt_M):
    """Compute quantitative alignment metrics."""
    pred5_aligned = warp_points(pred5, pred_M)
    gt5_aligned_by_gtM = warp_points(gt5, gt_M)
    gt5_aligned_by_predM = warp_points(gt5, pred_M)

    metrics = {
        # Reprojection error
        "pred_reprojection_error_px": reprojection_error(pred5, dst5, pred_M),
        "gt_reprojection_error_px": reprojection_error(gt5, dst5, gt_M),
        "gt_under_predM_reprojection_error_px": reprojection_error(gt5, dst5, pred_M),

        # Eye angle before alignment
        "pred_eye_angle_before_abs_deg": abs(eye_angle_deg(pred5)),
        "gt_eye_angle_before_abs_deg": abs(eye_angle_deg(gt5)),

        # Eye angle after alignment
        "pred_eye_angle_after_abs_deg": abs(eye_angle_deg(pred5_aligned)),
        "gt_eye_angle_after_by_gtM_abs_deg": abs(eye_angle_deg(gt5_aligned_by_gtM)),
        "gt_eye_angle_after_by_predM_abs_deg": abs(eye_angle_deg(gt5_aligned_by_predM)),

        # Landmark error before / after alignment
        "pred_gt_5pt_error_before_px": float(np.linalg.norm(pred5 - gt5, axis=1).mean()),
        "pred_gt_5pt_error_after_predM_px": float(
            np.linalg.norm(pred5_aligned - gt5_aligned_by_predM, axis=1).mean()
        ),
    }

    return metrics


def save_images_for_sample(index, img_tensor, pred_pts, gt_pts, pred_M):
    """Save visualization images only for selected indices."""
    img_rgb = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    pred_vis = draw_points(img_rgb, pred_pts, gt_pts)

    aligned_rgb = cv2.warpAffine(
        img_rgb,
        pred_M,
        (ALIGN_OUTPUT_SIZE, ALIGN_OUTPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )

    crop_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(ALIGN_OUT_DIR, f"{index}_crop_input.jpg"), crop_bgr)
    cv2.imwrite(os.path.join(ALIGN_OUT_DIR, f"{index}_pred_vs_gt.jpg"), pred_vis)
    cv2.imwrite(os.path.join(ALIGN_OUT_DIR, f"{index}_aligned.jpg"), aligned_bgr)


def evaluate_alignment_one(index, model, dataset, device, save_images=False):
    """Compute alignment metrics for one sample.

    Metrics are computed for every sample.
    Images are saved only when save_images=True.
    """
    img_tensor, _gt_hm, gt_pts_tensor = dataset[index]
    gt_pts = gt_pts_tensor.numpy().astype(np.float32)

    img_batch = img_tensor.unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad():
        pred_hm = model(img_batch)

    pred_pts = heatmaps_to_pts(
        pred_hm,
        IMG_SIZE,
        use_quarter_offset=USE_QUARTER_OFFSET,
    )[0].cpu().numpy().astype(np.float32)

    pred5 = get_5_points_from_68(pred_pts)
    gt5 = get_5_points_from_68(gt_pts)

    pred_M, dst5, _pred_inliers = estimate_align_matrix(
        pred5,
        output_size=ALIGN_OUTPUT_SIZE,
    )

    gt_M, _dst5_for_gt, _gt_inliers = estimate_align_matrix(
        gt5,
        output_size=ALIGN_OUTPUT_SIZE,
    )

    metrics = compute_alignment_metrics(
        pred5=pred5,
        gt5=gt5,
        dst5=dst5,
        pred_M=pred_M,
        gt_M=gt_M,
    )

    if save_images:
        save_images_for_sample(
            index=index,
            img_tensor=img_tensor,
            pred_pts=pred_pts,
            gt_pts=gt_pts,
            pred_M=pred_M,
        )

        metrics_path = os.path.join(ALIGN_OUT_DIR, f"{index}_alignment_metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"index: {index}\n")
            f.write(f"checkpoint: {ALIGN_CHECKPOINT_PATH}\n")
            f.write(f"output_size: {ALIGN_OUTPUT_SIZE}\n")
            f.write(f"use_quarter_offset: {USE_QUARTER_OFFSET}\n")
            f.write(f"template_112: {ALIGN_TEMPLATE_112}\n")
            f.write(f"pred_M: {pred_M.tolist()}\n")
            f.write(f"gt_M: {gt_M.tolist()}\n")
            f.write(f"pred5: {pred5.tolist()}\n")
            f.write(f"gt5: {gt5.tolist()}\n")
            f.write(f"dst5: {dst5.tolist()}\n\n")

            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

    return metrics


def save_alignment_summary(all_metrics, output_dir, failed_indices):
    """Save summary metrics over the full test set."""
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "alignment_summary.txt")

    with open(summary_path, "w") as f:
        f.write(f"checkpoint: {ALIGN_CHECKPOINT_PATH}\n")
        f.write(f"num_valid_samples: {len(all_metrics)}\n")
        f.write(f"num_failed_samples: {len(failed_indices)}\n")
        f.write(f"failed_indices: {failed_indices}\n")
        f.write(f"output_size: {ALIGN_OUTPUT_SIZE}\n")
        f.write(f"use_quarter_offset: {USE_QUARTER_OFFSET}\n")
        f.write(f"template_112: {ALIGN_TEMPLATE_112}\n")
        f.write(f"saved_visualization_range: [{ALIGN_INDEX_START}, {ALIGN_INDEX_END})\n\n")

        if not all_metrics:
            f.write("No valid metrics were computed.\n")
            print(f"No valid metrics. Saved summary to: {summary_path}")
            return

        keys = list(all_metrics[0].keys())

        for key in keys:
            values = np.array([m[key] for m in all_metrics], dtype=np.float32)

            f.write(f"{key}_mean: {values.mean()}\n")
            f.write(f"{key}_std: {values.std()}\n")
            f.write(f"{key}_median: {np.median(values)}\n")
            f.write(f"{key}_min: {values.min()}\n")
            f.write(f"{key}_max: {values.max()}\n\n")

    print(f"Saved alignment summary to: {summary_path}")


def main():
    device = torch.device(DEVICE)

    print("device:", device)
    print(f"checkpoint: {ALIGN_CHECKPOINT_PATH}")
    print(f"align output dir: {ALIGN_OUT_DIR}")
    print(f"disk cache enabled: {USE_DISK_CACHE}")
    print(f"cache dir: {CACHE_DIR}")
    print(f"metrics range: full test set")
    print(f"image save range: [{ALIGN_INDEX_START}, {ALIGN_INDEX_END})")

    os.makedirs(ALIGN_OUT_DIR, exist_ok=True)

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

    all_metrics = []
    failed_indices = []

    total = len(dataset)
    print(f"Evaluating alignment on full test set: {total} samples")

    for i in range(total):
        try:
            save_images = ALIGN_INDEX_START <= i < ALIGN_INDEX_END

            metrics = evaluate_alignment_one(
                index=i,
                model=model,
                dataset=dataset,
                device=device,
                save_images=save_images,
            )

            all_metrics.append(metrics)

            if save_images:
                print(f"Saved visualization for index {i}")

            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{total} samples")

        except Exception as e:
            failed_indices.append(i)
            print(f"[WARN] failed to process index {i}: {repr(e)}")

    save_alignment_summary(
        all_metrics=all_metrics,
        output_dir=ALIGN_OUT_DIR,
        failed_indices=failed_indices,
    )

    if all_metrics:
        print("\nAlignment summary:")
        keys_to_print = [
            "pred_reprojection_error_px",
            "gt_reprojection_error_px",
            "gt_under_predM_reprojection_error_px",
            "pred_eye_angle_before_abs_deg",
            "pred_eye_angle_after_abs_deg",
            "gt_eye_angle_after_by_predM_abs_deg",
        ]

        for key in keys_to_print:
            values = np.array([m[key] for m in all_metrics], dtype=np.float32)
            print(f"{key}: mean={values.mean():.6f}, std={values.std():.6f}")

    print(f"\nValid samples: {len(all_metrics)}")
    print(f"Failed samples: {len(failed_indices)}")


if __name__ == "__main__":
    main()