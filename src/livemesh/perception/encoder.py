"""CNN-Transformer Encoder — implemented from scratch.

ResNet-50 backbone (manual residual blocks) + Transformer encoder
(manual multi-head self-attention). No torchvision, no nn.TransformerEncoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ============================================================
# ResNet-50 from scratch
# ============================================================


class BottleneckBlock(nn.Module):
    """Bottleneck residual block: 1x1 → 3x3 → 1x1 with skip connection.

    Reduces dimensionality with 1x1 conv, applies spatial conv at reduced dim,
    then expands back. The skip connection enables gradient flow through depth.
    """

    EXPANSION = 4

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        out_channels = mid_channels * self.EXPANSION

        # 1x1 reduce
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3x3 spatial
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 expand
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out, inplace=True)


class ResNet50Backbone(nn.Module):
    """ResNet-50 feature extractor built from individual bottleneck blocks.

    Architecture: conv1 → bn → relu → maxpool → layer1(3) → layer2(4) → layer3(6) → layer4(3)
    Output: (B, 2048, H/32, W/32) feature maps
    """

    def __init__(self):
        super().__init__()

        self.in_channels = 64

        # Stem: 7x7 conv, stride 2, then maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers: [3, 4, 6, 3] blocks for ResNet-50
        self.layer1 = self._make_layer(mid_channels=64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(mid_channels=128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(mid_channels=256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(mid_channels=512, num_blocks=3, stride=2)

        self._init_weights()

    def _make_layer(self, mid_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        out_channels = mid_channels * BottleneckBlock.EXPANSION
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BottleneckBlock(self.in_channels, mid_channels, stride, downsample)]
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(self.in_channels, mid_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 256, 256) → (B, 2048, 8, 8)"""
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def load_pretrained_imagenet(self):
        """Load ImageNet weights into matching architecture.

        The architecture is identical to torchvision's ResNet-50,
        so we can transfer weights directly by matching state_dict keys.
        """
        import torchvision.models as models

        pretrained = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        pretrained_dict = pretrained.state_dict()

        model_dict = self.state_dict()
        # Filter: only load keys that exist in our model and have matching shapes
        compatible = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(compatible)
        self.load_state_dict(model_dict)
        print(f"Loaded {len(compatible)}/{len(model_dict)} pretrained parameters")


# ============================================================
# Transformer Encoder from scratch
# ============================================================


class ScaledDotProductAttention(nn.Module):
    """Core attention mechanism: softmax(QK^T / sqrt(d_k)) V"""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, heads, seq_len, d_k)
            key:   (B, heads, seq_len, d_k)
            value: (B, heads, seq_len, d_k)
            mask:  optional attention mask

        Returns:
            (B, heads, seq_len, d_k) attended values
        """
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        return torch.matmul(attention_weights, value)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention: project to h heads, attend, concat, project back.

    Splits d_model into h heads of dimension d_k = d_model / h,
    applies scaled dot-product attention independently per head,
    then concatenates and projects back to d_model.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: (B, seq_len, d_model)

        Returns:
            (B, seq_len, d_model)
        """
        B, seq_len, _ = query.shape

        # Project and reshape to (B, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attended = self.attention(Q, K, V, mask)  # (B, heads, seq_len, d_k)

        # Concat heads and project
        concat = attended.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        return self.W_o(concat)


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network: Linear → GELU → Dropout → Linear → Dropout"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(self.dropout(F.gelu(self.linear1(x)))))


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer: LayerNorm → MHA → Residual → LayerNorm → FFN → Residual

    Uses Pre-LN (layer norm before attention/FFN) for more stable training.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN: norm before attention
        normed = self.norm1(x)
        x = x + self.dropout1(self.mha(normed, normed, normed, mask))

        # Pre-LN: norm before FFN
        normed = self.norm2(x)
        x = x + self.dropout2(self.ffn(normed))

        return x


class TransformerEncoder(nn.Module):
    """Stack of N transformer encoder layers with final layer norm."""

    def __init__(self, d_model: int, num_heads: int, num_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ============================================================
# Positional Encoding
# ============================================================


class LearnedPositionalEncoding2D(nn.Module):
    """Learned 2D positional encoding for spatial feature maps.

    Separate row and column embeddings concatenated to form full positional signal.
    """

    def __init__(self, d_model: int, h: int = 8, w: int = 8):
        super().__init__()
        self.row_embed = nn.Embedding(h, d_model // 2)
        self.col_embed = nn.Embedding(w, d_model // 2)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Add 2D positional encoding to flattened spatial tokens.

        Args:
            x: (B, H*W, d_model)
            h, w: spatial dimensions

        Returns:
            (B, H*W, d_model) with positional encoding added
        """
        rows = torch.arange(h, device=x.device)
        cols = torch.arange(w, device=x.device)

        row_emb = self.row_embed(rows).unsqueeze(1).expand(-1, w, -1)  # (H, W, d/2)
        col_emb = self.col_embed(cols).unsqueeze(0).expand(h, -1, -1)  # (H, W, d/2)

        pos = torch.cat([row_emb, col_emb], dim=-1)  # (H, W, d)
        pos = pos.reshape(h * w, -1).unsqueeze(0)  # (1, H*W, d)

        return x + pos


# ============================================================
# Full CNN-Transformer Encoder
# ============================================================


class CNNTransformerEncoder(nn.Module):
    """Complete encoder: ResNet-50 (from scratch) → 1x1 projection → Transformer (from scratch).

    Input:  (B, 3, 256, 256) RGB wound image
    Output: (B, 64, d_model) contextualized feature tokens
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        # CNN backbone (from scratch)
        self.backbone = ResNet50Backbone()
        if pretrained:
            self.backbone.load_pretrained_imagenet()

        # 1x1 projection: 2048 → d_model
        self.projection = nn.Sequential(
            nn.Conv2d(2048, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # Positional encoding
        self.pos_encoding = LearnedPositionalEncoding2D(d_model, h=8, w=8)

        # Transformer encoder (from scratch)
        self.transformer = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_model * 4,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 256, 256) input RGB image

        Returns:
            (B, 64, d_model) encoded feature tokens with global context
        """
        # CNN feature extraction
        features = self.backbone(x)  # (B, 2048, 8, 8)
        features = self.projection(features)  # (B, d_model, 8, 8)

        # Flatten spatial dims to sequence
        B, C, H, W = features.shape
        tokens = features.flatten(2).permute(0, 2, 1)  # (B, H*W, d_model)

        # Add 2D positional encoding
        tokens = self.pos_encoding(tokens, H, W)

        # Transformer for global context
        tokens = self.transformer(tokens)

        return tokens
