import os
import torch
# import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from losses.heatmap_loss import build_heatmap_loss

from config import (
    DATASET_PATH,
    IMG_SIZE,
    HEATMAP_SIZE,
    HEATMAP_SIGMA,
    NUM_LANDMARKS,
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    RESUME,
    RESUME_PATH,
    SAVE_EVERY_EPOCHS,
    CHECKPOINT_DIR,
    SAVE_PATH,
    BEST_PATH,
    LOSS_LOG_PATH,
    PRETRAINED_PATH,
    CROP_SCALE,
    USE_DISK_CACHE,
    CACHE_DIR,

    # 新增这些
    LOSS_TYPE,
    FOREGROUND_THRESHOLD,
    FOREGROUND_WEIGHT,
    JAW_WEIGHT,
)
from datasets.deeplake_300w import DeepLake300W
from models.HRNet import hrnet_w18_face


def save_checkpoint(path, model, optimizer, epoch, avg_loss, best_loss, best_epoch):
    torch.save(
        {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Disk cache enabled: {USE_DISK_CACHE}")
    print(f"Cache dir: {CACHE_DIR}")

    train_set = DeepLake300W(
        DATASET_PATH,
        split="train",
        img_size=IMG_SIZE,
        heatmap_size=HEATMAP_SIZE,
        crop_scale=CROP_SCALE,
        heatmap_sigma=HEATMAP_SIGMA,
        use_disk_cache=USE_DISK_CACHE,
        cache_dir=CACHE_DIR,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE == "cuda"),
    )

    model = hrnet_w18_face(NUM_LANDMARKS, pretrained=PRETRAINED_PATH).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # criterion = nn.MSELoss()
    criterion = build_heatmap_loss(
        loss_type=LOSS_TYPE,
        foreground_weight=FOREGROUND_WEIGHT,
        threshold=FOREGROUND_THRESHOLD,
        jaw_weight=JAW_WEIGHT,
    ).to(DEVICE)

    print("=" * 60)
    print(f"Loss type: {LOSS_TYPE}")
    print(f"Foreground threshold: {FOREGROUND_THRESHOLD}")
    print(f"Foreground weight: {FOREGROUND_WEIGHT}")
    print(f"Jaw weight: {JAW_WEIGHT}")
    print("=" * 60)

    start_epoch = 0
    best_loss = float("inf")
    best_epoch = 0

    if RESUME:
        if not os.path.exists(RESUME_PATH):
            raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_PATH}")

        ckpt = torch.load(RESUME_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = ckpt["epoch"]
        best_loss = ckpt.get("best_loss", float("inf"))
        best_epoch = ckpt.get("best_epoch", start_epoch)

        print("=" * 60)
        print(f"Resume from: {RESUME_PATH}")
        print(f"Start epoch: {start_epoch + 1}")
        print(f"Target epoch: {EPOCHS}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best loss: {best_loss:.8f}")
        print("=" * 60)

    log_mode = "a" if RESUME else "w"
    with open(LOSS_LOG_PATH, log_mode) as f:
        if not RESUME:
            f.write("epoch,avg_loss,best_loss,best_epoch,is_best\n")

    for epoch in range(start_epoch, EPOCHS):
        current_epoch = epoch + 1
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {current_epoch}/{EPOCHS}")

        for imgs, heatmaps, _ in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            heatmaps = heatmaps.to(DEVICE, non_blocking=True)

            preds = model(imgs)
            loss = criterion(preds, heatmaps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value

            pbar.set_postfix(
                {
                    "batch_loss": f"{loss_value:.6f}",
                    "best_loss": f"{best_loss:.6f}" if best_loss < float("inf") else "inf",
                    "best_epoch": best_epoch,
                }
            )

        avg_loss = total_loss / len(train_loader)
        is_best = avg_loss < best_loss

        if is_best:
            best_loss = avg_loss
            best_epoch = current_epoch

        print(
            f"Epoch {current_epoch}/{EPOCHS} | "
            f"avg_loss={avg_loss:.8f} | "
            f"best_loss={best_loss:.8f} | "
            f"best_epoch={best_epoch}"
        )

        if current_epoch % SAVE_EVERY_EPOCHS == 0:
            epoch_path = CHECKPOINT_DIR / f"hrnet_epoch_{current_epoch}.pth"
            save_checkpoint(
                epoch_path,
                model,
                optimizer,
                current_epoch,
                avg_loss,
                best_loss,
                best_epoch,
            )
            print(f"saved epoch checkpoint: {epoch_path}")

        if is_best:
            save_checkpoint(
                BEST_PATH,
                model,
                optimizer,
                current_epoch,
                avg_loss,
                best_loss,
                best_epoch,
            )
            print(f"saved best checkpoint: {BEST_PATH}")

        save_checkpoint(
            SAVE_PATH,
            model,
            optimizer,
            current_epoch,
            avg_loss,
            best_loss,
            best_epoch,
        )
        print(f"saved latest checkpoint: {SAVE_PATH}")

        with open(LOSS_LOG_PATH, "a") as f:
            f.write(
                f"{current_epoch},"
                f"{avg_loss:.8f},"
                f"{best_loss:.8f},"
                f"{best_epoch},"
                f"{int(is_best)}\n"
            )


if __name__ == "__main__":
    main()