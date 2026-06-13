import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import ImageTokenzier, PositionEmbedding


class ViTAttention(nn.Module):
    def __init__(
            self, 
            embed_dim=768, 
            num_heads=8, 
            kv_heads=8, 
            head_dim=64, 
            dropout=0.1,
        ):
        assert num_heads % kv_heads == 0, "num_heads must be divisible by kv_heads for GQA"
        super(ViTAttention, self).__init__()
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(embed_dim, (2 * kv_heads + num_heads) * head_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(num_heads * head_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, embed_dim)
        B, N, _ = x.shape

        # qkv: (B, N, (num_heads + 2 * kv_heads) * head_dim)
        qkv = self.qkv_proj(x)
        
        # qkv:
        # (B, N, (num_heads + 2 * kv_heads) * head_dim)
        # -> (B, N, num_heads + 2 * kv_heads, head_dim)
        # -> (B, num_heads + 2 * kv_heads, N, head_dim)
        qkv = qkv.reshape(
            B, N, self.num_heads + 2 * self.kv_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        # q: (B, num_heads, N, head_dim)
        # k: (B, kv_heads,  N, head_dim)
        # v: (B, kv_heads,  N, head_dim)
        q, k, v = torch.split(
            qkv,
            [self.num_heads, self.kv_heads, self.kv_heads],
            dim=1
        )

        # scaled dot-product attention:
        # 标准 MHA: num_heads == kv_heads
        #   q, k, v head 数一致
        #
        # GQA: num_heads > kv_heads 且 enable_gqa=True
        #   PyTorch 内部会扩展 k/v 的 head 数以匹配 q
        #
        # output: (B, num_heads, N, head_dim)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
            enable_gqa=self.kv_heads != self.num_heads
        )
        
        # (B, num_heads, N, head_dim)
        # -> (B, N, num_heads, head_dim)
        # -> (B, N, num_heads * head_dim)
        # -> (B, N, embed_dim)
        x = x.transpose(1, 2).reshape(B, N, -1)
        return self.proj_drop(self.proj(x))

class ViTMLP(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            mlp_ratio=4.0,
            dropout=0.1
        ):
        super(ViTMLP, self).__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, embed_dim) -> (B, N, hidden_dim)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        # (B, N, hidden_dim) -> (B, N, embed_dim)
        x = self.fc2(x)
        return self.drop(x)

class ViTBlock(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            num_heads=8,
            kv_heads=8,
            head_dim=64,
            mlp_ratio=4.0,
            dropout=0.1
        ):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ViTAttention(embed_dim, num_heads, kv_heads, head_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = ViTMLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, embed_dim) -> (B, N, embed_dim)
        x = x + self.attn(self.norm1(x))
        # (B, N, embed_dim) -> (B, N, embed_dim)
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(
            self,
            img_size=(224, 224),
            patch_size=16,
            in_channels=3,
            num_classes=10,
            embed_dim=768,
            depth=12,
            num_heads=8,
            kv_heads=8,
            head_dim=64,
            mlp_ratio=4.0,
            dropout=0.1,
            cls_token=True,
        ):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, "Image size must be divisible by patch size"
        super(ViT, self).__init__()
        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) + (1 if cls_token else 0)
        self.pos_embed = PositionEmbedding(num_patches, embed_dim)
        self.patch_embed = ImageTokenzier(patch_size, in_channels, embed_dim)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, kv_heads, head_dim, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, C, H, W) -> (B, N, embed_dim)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (1, 1, embed_dim) -> (B, 1, embed_dim)
            x = torch.cat((cls_tokens, x), dim=1)  # (B, 1, embed_dim), (B, N, embed_dim) -> (B, N+1, embed_dim)

        x = self.pos_embed(x)  # (B, N+1, embed_dim) -> (B, N+1, embed_dim)

        for block in self.blocks:
            x = block(x)  # (B, N+1, embed_dim) -> (B, N+1, embed_dim)

        x = self.norm(x)  # (B, N+1, embed_dim)
        if self.cls_token is not None:
            return self.head(x[:, 0])  # (B, N+1, embed_dim) -> (B, embed_dim) -> (B, num_classes)
        else:
            return self.head(x.mean(dim=1))  # (B, N, embed_dim) -> (B, embed_dim) -> (B, num_classes)