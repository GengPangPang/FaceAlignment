import numpy as np


def generate_heatmaps(landmarks, img_size=256, heatmap_size=64, sigma=2):
    """Generate one Gaussian heatmap for each landmark.

    Args:
        landmarks: ndarray with shape [N, 2] in image coordinates after crop/resize.
        img_size: input image size, assumed square.
        heatmap_size: output heatmap size, assumed square.
        sigma: Gaussian standard deviation in heatmap-coordinate pixels.

    Returns:
        heatmaps: ndarray with shape [N, heatmap_size, heatmap_size].
    """
    num_landmarks = landmarks.shape[0]
    heatmaps = np.zeros((num_landmarks, heatmap_size, heatmap_size), dtype=np.float32)

    scale = heatmap_size / img_size

    xx, yy = np.meshgrid(
        np.arange(heatmap_size, dtype=np.float32),
        np.arange(heatmap_size, dtype=np.float32),
    )

    for i, (x, y) in enumerate(landmarks):
        x = float(x) * scale
        y = float(y) * scale

        # 如果关键点完全落在 heatmap 外，则保持全 0。
        if x < 0 or x >= heatmap_size or y < 0 or y >= heatmap_size:
            continue

        heatmaps[i] = np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)
        )

    return heatmaps