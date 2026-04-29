DATASET_PATH = "hub://activeloop/300w"

IMG_SIZE = 256
HEATMAP_SIZE = 64
NUM_LANDMARKS = 68

BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-3

DEVICE = "cuda"

RESUME = False
RESUME_PATH = "checkpoints/hrnet_epoch_200.pth"

SAVE_EVERY_EPOCHS = 10

SAVE_PATH = "checkpoints/landmark_HRNet.pth"
BEST_PATH = "checkpoints/best_epoch.pth"
LOSS_LOG_PATH = "checkpoints/HRNet_loss_log.csv"