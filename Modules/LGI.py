import torch
import torch.nn as nn


class SqueezeExcite(nn.Module):
    """Channel-wise Squeeze-and-Excitation recalibration module."""

    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcite, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, reduced_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_transposed = x.transpose(1, 2)
        channel_weights = self.se(x_transposed)
        return x * channel_weights.transpose(1, 2)


class LocalGeometryInjector(nn.Module):
    """Local Geometry Injector (LGI).

    Injects local spatial structure into the Transformer token stream via a
    depthwise convolutional branch followed by Squeeze-and-Excitation channel
    recalibration. Serves as the structural harmonization component within the
    Heterogeneous Synergistic Attention Block (HSAB).

    Args:
        dim (int): Token channel dimension (C).
        kernel_size (int): Depthwise convolution kernel size. Default: 3.
        reduction_ratio (int): SE reduction ratio. Default: 4.
        norm_layer: Normalization layer constructor. Default: nn.LayerNorm.
    """

    def __init__(self, dim, kernel_size=3, reduction_ratio=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        padding = kernel_size // 2

        # Depthwise convolution to capture local spatial geometry
        self.dw_conv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            bias=False
        )

        self.norm = norm_layer(dim)
        self.act = nn.GELU()

        # SE block for channel-wise feature recalibration
        self.se_attention = SqueezeExcite(in_channels=dim, reduced_dim=dim // reduction_ratio)

    def forward(self, x, H, W):
        """
        Args:
            x: Token sequence, shape (B, H*W, C).
            H (int): Feature map height.
            W (int): Feature map width.

        Returns:
            Locally-enhanced token sequence, shape (B, H*W, C).
        """
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        assert C == self.dim, "Input channel dimension has wrong size"

        shortcut = x

        # Reshape tokens to spatial feature map for convolution
        x_image = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_conv = self.dw_conv(x_image)

        # Reshape back to token sequence
        x_conv_tokens = x_conv.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        x_normed = self.norm(x_conv_tokens)

        # Apply SE channel recalibration
        x_attended = self.se_attention(x_normed)

        # Residual connection
        return shortcut + x_attended
