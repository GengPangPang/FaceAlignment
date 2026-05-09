import torch
import torch.nn as nn


class HeatmapMSELoss(nn.Module):
    """
    普通 MSE Loss，作为 baseline。
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, preds, targets):
        return self.criterion(preds, targets)


class ForegroundWeightedMSELoss(nn.Module):
    """
    Foreground-weighted MSE.

    对 GT heatmap 中大于 threshold 的前景区域赋予更高权重，
    以缓解 heatmap 中前景区域少、背景区域多的问题。
    """
    def __init__(self, foreground_weight=5.0, threshold=0.01):
        super().__init__()
        self.foreground_weight = foreground_weight
        self.threshold = threshold

    def forward(self, preds, targets):
        # preds, targets: [B, 68, 64, 64]

        # 背景区域权重为 1
        weights = torch.ones_like(targets)

        # GT heatmap 大于 threshold 的区域视为前景
        foreground_mask = targets > self.threshold

        # 前景区域权重设为 foreground_weight
        weights[foreground_mask] = self.foreground_weight

        loss = weights * (preds - targets) ** 2
        return loss.mean()


class ForegroundJawWeightedMSELoss(nn.Module):
    """
    Foreground + Jaw weighted MSE.

    1. 对所有关键点的前景区域加权；
    2. 对下颌线关键点 0-16 额外加权。
    """
    def __init__(
        self,
        foreground_weight=5.0,
        threshold=0.01,
        jaw_weight=1.5,
    ):
        super().__init__()
        self.foreground_weight = foreground_weight
        self.threshold = threshold
        self.jaw_weight = jaw_weight

    def forward(self, preds, targets):
        # preds, targets: [B, 68, 64, 64]

        weights = torch.ones_like(targets)

        # 前景区域加权
        foreground_mask = targets > self.threshold
        weights[foreground_mask] = self.foreground_weight

        # 下颌线关键点 0-16 额外加权
        weights[:, 0:17, :, :] *= self.jaw_weight

        loss = weights * (preds - targets) ** 2
        return loss.mean()


def build_heatmap_loss(
    loss_type="mse",
    foreground_weight=5.0,
    threshold=0.01,
    jaw_weight=1.5,
):
    if loss_type == "mse":
        return HeatmapMSELoss()

    if loss_type == "foreground_weighted_mse":
        return ForegroundWeightedMSELoss(
            foreground_weight=foreground_weight,
            threshold=threshold,
        )

    if loss_type == "foreground_jaw_weighted_mse":
        return ForegroundJawWeightedMSELoss(
            foreground_weight=foreground_weight,
            threshold=threshold,
            jaw_weight=jaw_weight,
        )

    raise ValueError(f"Unknown loss_type: {loss_type}")