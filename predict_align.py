import os
import cv2
import torch
import numpy as np

from config import DATASET_PATH, IMG_SIZE, HEATMAP_SIZE
from datasets.deeplake_300w import DeepLake300W
from models.HRNet import hrnet_w18_face


CHECKPOINT_PATH = "checkpoints/hrnet_epoch_200.pth"
OUT_DIR = "test/single_predict"
NUM_LANDMARKS = 68


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
        dtype=np.float32
    )


def align_face(img_rgb, pts68, output_size=112):
    src = get_5_points_from_68(pts68)

    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    if output_size != 112:
        dst *= output_size / 112.0

    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

    aligned = cv2.warpAffine(
        img_rgb,
        M,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderValue=0
    )

    return aligned


def predict_one(index=0):
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = DeepLake300W(
        DATASET_PATH,
        split="test",
        img_size=IMG_SIZE,
        heatmap_size=HEATMAP_SIZE,
    )

    img_tensor, gt_hm = dataset[index]

    model = hrnet_w18_face(num_landmarks=NUM_LANDMARKS)
    model = load_checkpoint(model, CHECKPOINT_PATH, device)
    model = model.to(device)
    model.eval()

    img_batch = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_hm = model(img_batch)

    pred_pts = heatmaps_to_pts(pred_hm, IMG_SIZE)[0].cpu().numpy()
    gt_pts = heatmaps_to_pts(gt_hm.unsqueeze(0), IMG_SIZE)[0].cpu().numpy()

    img_rgb = (img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    crop_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    pred_vis = draw_points(img_rgb, pred_pts, gt_pts)
    aligned_rgb = align_face(img_rgb, pred_pts, output_size=112)
    aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(OUT_DIR, f"{index}_crop_input.jpg"), crop_bgr)
    cv2.imwrite(os.path.join(OUT_DIR, f"{index}_pred_vs_gt.jpg"), pred_vis)
    cv2.imwrite(os.path.join(OUT_DIR, f"{index}_aligned.jpg"), aligned_bgr)

    print("saved:")
    print(os.path.join(OUT_DIR, f"{index}_crop_input.jpg"))
    print(os.path.join(OUT_DIR, f"{index}_pred_vs_gt.jpg"))
    print(os.path.join(OUT_DIR, f"{index}_aligned.jpg"))

    print("pred points:")
    print(pred_pts)


if __name__ == "__main__":
    predict_one(index=0)