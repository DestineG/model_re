import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            downsample=None,
            dropout=0.0
        ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, W)
        identity = x

        # (B, C_in, H, W) -> (B, C_out, H/stride, W/stride)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        # (B, C_out, H/stride, W/stride) -> (B, C_out, H/stride, W/stride)
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果输入输出维度不同，对 identity 进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # (B, C_out, H/stride, W/stride) + (B, C_out, H/stride, W/stride)
        out = out + identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            downsample=None,
            dropout=0.0
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, W)
        identity = x

        # (B, C_in, H, W) -> (B, C_out, H, W)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        # (B, C_out, H, W) -> (B, C_out, H/stride, W/stride)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.dropout is not None:
            out = self.dropout(out)

        # (B, C_out, H/stride, W/stride) -> (B, C_out * expansion, H/stride, W/stride)
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果输入输出维度不同，对 identity 进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # (B, C_out * expansion, H/stride, W/stride) + (B, C_out * expansion, H/stride, W/stride)
        out = out + identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            in_channels=3,
            num_classes=10,
            dropout=0.0
        ):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层: (B, 3, 224, 224) -> (B, 64, 112, 112)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # (B, 64, 112, 112) -> (B, 64, 56, 56)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个 stage
        # stage1: (B, 64, 56, 56) -> (B, 64 * expansion, 56, 56)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dropout=dropout)
        # stage2: (B, 64 * expansion, 56, 56) -> (B, 128 * expansion, 28, 28)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dropout=dropout)
        # stage3: (B, 128 * expansion, 28, 28) -> (B, 256 * expansion, 14, 14)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dropout=dropout)
        # stage4: (B, 256 * expansion, 14, 14) -> (B, 512 * expansion, 7, 7)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dropout=dropout)

        # 全局平均池化 + 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout):
        downsample = None
        # 如果需要改变维度（通道数或空间尺寸），创建 downsample
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        # 第一个 block 可能需要下采样
        layers.append(block(self.in_channels, out_channels, stride, downsample, dropout))
        self.in_channels = out_channels * block.expansion

        # 后续 block 不需要下采样
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, dropout=dropout))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)

        # 初始卷积: (B, 3, 224, 224) -> (B, 64, 112, 112)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # (B, 64, 112, 112) -> (B, 64, 56, 56)
        x = self.maxpool(x)

        # 4 个 stage
        x = self.layer1(x)  # (B, 64, 56, 56) -> (B, 64 * expansion, 56, 56)
        x = self.layer2(x)  # -> (B, 128 * expansion, 28, 28)
        x = self.layer3(x)  # -> (B, 256 * expansion, 14, 14)
        x = self.layer4(x)  # -> (B, 512 * expansion, 7, 7)

        # 全局平均池化: (B, 512 * expansion, 7, 7) -> (B, 512 * expansion, 1, 1)
        x = self.avgpool(x)
        # (B, 512 * expansion, 1, 1) -> (B, 512 * expansion)
        x = torch.flatten(x, 1)
        # (B, 512 * expansion) -> (B, num_classes)
        return self.fc(x)


def resnet18(num_classes=10, dropout=0.0):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dropout=dropout)


def resnet34(num_classes=10, dropout=0.0):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, dropout=dropout)


def resnet50(num_classes=10, dropout=0.0):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, dropout=dropout)


def resnet101(num_classes=10, dropout=0.0):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, dropout=dropout)


def resnet152(num_classes=10, dropout=0.0):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, dropout=dropout)
