"""
Original-style HRNet for face landmark heatmap regression.
Input:  [B, 3, 256, 256]
Output: [B, num_landmarks, 64, 64]

Usage:
    from original_hrnet_landmark import HRNetFaceLandmark, hrnet_w18_face

    model = hrnet_w18_face(num_landmarks=68)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(y.shape)  # torch.Size([2, 68, 64, 64])
"""

import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.01
BatchNorm2d = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


BLOCKS = {
    "BASIC": BasicBlock,
    "BOTTLENECK": Bottleneck,
}


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        block,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method="SUM",
        multi_scale_output=True,
    ):
        super().__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.num_branches = num_branches
        self.block = block
        self.num_blocks = num_blocks
        self.num_inchannels = num_inchannels
        self.num_channels = num_channels
        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches()
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _check_branches(num_branches, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            raise ValueError("NUM_BRANCHES must equal len(NUM_BLOCKS)")
        if num_branches != len(num_channels):
            raise ValueError("NUM_BRANCHES must equal len(NUM_CHANNELS)")
        if num_branches != len(num_inchannels):
            raise ValueError("NUM_BRANCHES must equal len(NUM_INCHANNELS)")

    def _make_one_branch(self, branch_index, stride=1):
        downsample = None
        in_channels = self.num_inchannels[branch_index]
        out_channels = self.num_channels[branch_index] * self.block.expansion

        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )

        layers = [
            self.block(
                in_channels,
                self.num_channels[branch_index],
                stride,
                downsample,
            )
        ]
        self.num_inchannels[branch_index] = out_channels

        for _ in range(1, self.num_blocks[branch_index]):
            layers.append(
                self.block(
                    self.num_inchannels[branch_index],
                    self.num_channels[branch_index],
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self):
        return nn.ModuleList(
            [self._make_one_branch(i) for i in range(self.num_branches)]
        )

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []

        out_branches = num_branches if self.multi_scale_output else 1
        for i in range(out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            out_channels = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False,
                                    ),
                                    BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                                )
                            )
                        else:
                            out_channels = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False,
                                    ),
                                    BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class HRNetFaceLandmark(nn.Module):
    def __init__(self, num_landmarks=68, width=18, final_conv_kernel=1):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.inplanes = 64

        stage_cfg = self._make_w18_cfg(width)

        # Stem: 256 -> 128 -> 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: ResNet bottleneck, channels 64 -> 256
        self.layer1 = self._make_layer(Bottleneck, 64, 64, blocks=4)

        # Stage 2
        self.stage2_cfg = stage_cfg["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = BLOCKS[self.stage2_cfg["BLOCK"]]
        num_channels = [c * block.expansion for c in num_channels]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        # Stage 3
        self.stage3_cfg = stage_cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = BLOCKS[self.stage3_cfg["BLOCK"]]
        num_channels = [c * block.expansion for c in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        # Stage 4
        self.stage4_cfg = stage_cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = BLOCKS[self.stage4_cfg["BLOCK"]]
        num_channels = [c * block.expansion for c in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True
        )

        final_inp_channels = sum(pre_stage_channels)
        padding = 1 if final_conv_kernel == 3 else 0
        self.head = nn.Sequential(
            nn.Conv2d(final_inp_channels, final_inp_channels, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                final_inp_channels,
                num_landmarks,
                kernel_size=final_conv_kernel,
                stride=1,
                padding=padding,
            ),
        )

        self.init_weights()

    @staticmethod
    def _make_w18_cfg(width):
        # HRNet-W18 setting commonly used by facial landmark HRNet.
        return {
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4],
                "NUM_CHANNELS": [width, width * 2],
                "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 4,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4, 4],
                "NUM_CHANNELS": [width, width * 2, width * 4],
                "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4, 4, 4],
                "NUM_CHANNELS": [width, width * 2, width * 4, width * 8],
                "FUSE_METHOD": "SUM",
            },
        }

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []

        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                            BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else in_channels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = BLOCKS[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = multi_scale_output or i != num_modules - 1
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            x_list.append(self.transition1[i](x) if self.transition1[i] is not None else x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            x_list.append(
                self.transition2[i](y_list[-1]) if self.transition2[i] is not None else y_list[i]
            )
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            x_list.append(
                self.transition3[i](y_list[-1]) if self.transition3[i] is not None else y_list[i]
            )
        y_list = self.stage4(x_list)

        height, width = y_list[0].shape[2:]
        upsampled = [y_list[0]]
        for i in range(1, len(y_list)):
            upsampled.append(
                F.interpolate(
                    y_list[i],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            )

        x = torch.cat(upsampled, dim=1)
        x = self.head(x)
        return x

    def init_weights(self, pretrained=""):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained and os.path.isfile(pretrained):
            state_dict = torch.load(pretrained, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v

            model_dict = self.state_dict()
            load_dict = {
                k: v
                for k, v in new_state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(load_dict)
            self.load_state_dict(model_dict)
            print(f"Loaded pretrained weights: {len(load_dict)}/{len(model_dict)} tensors")


def hrnet_w18_face(num_landmarks=68, pretrained=""):
    model = HRNetFaceLandmark(num_landmarks=num_landmarks, width=18, final_conv_kernel=1)
    if pretrained:
        model.init_weights(pretrained)
    return model


if __name__ == "__main__":
    model = hrnet_w18_face(num_landmarks=68)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(y.shape)
