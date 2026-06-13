import torch
import torch.nn as nn


class ImageTokenzier(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super(ImageTokenzier, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        tokens = self.conv(x)
        # (B, embed_dim, H/patch_size, W/patch_size) -> (B, embed_dim, H/patch_size * W/patch_size) -> (B, H/patch_size * W/patch_size, embed_dim)
        tokens = tokens.flatten(2).transpose(1, 2)
        return tokens

class PositionEmbedding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * .02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, D) + (1, N, D) -> (B, N, D)
        return x + self.pos_embed