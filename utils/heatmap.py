import numpy as np

def generate_heatmaps(landmarks, img_size=256, heatmap_size=64, sigma=2):
    heatmaps = np.zeros((68, heatmap_size, heatmap_size), dtype=np.float32)
    scale = heatmap_size / img_size

    xx, yy = np.meshgrid(
        np.arange(heatmap_size),
        np.arange(heatmap_size)
    )

    for i, (x, y) in enumerate(landmarks):
        x = x * scale
        y = y * scale

        heatmaps[i] = np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)
        )

    return heatmaps