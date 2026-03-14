import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPathReduction(nn.Module):
    """Dual-Path Reduction (DPR).

    A heterogeneous downsampling module that fuses a Transformer token-merging
    path with a strided depthwise-separable CNN path, injecting local geometry
    inductive bias while halving spatial resolution and doubling channel depth.

    Args:
        input_resolution (tuple[int]): Spatial resolution (H, W) of input tokens.
        dim (int): Input channel dimension.
        norm_layer: Normalization layer constructor. Default: nn.LayerNorm.
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = 2 * dim

        # Transformer path: patch-merging via linear projection of 2x2 patches
        self.transformer_path_reduction = nn.Linear(4 * dim, self.out_dim, bias=False)

        # CNN path: strided depthwise + pointwise convolution for local structure
        self.cnn_path_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=dim),
            nn.GELU(),
            nn.Conv2d(dim, self.out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.norm = norm_layer(self.out_dim)

    def forward(self, x):
        """
        Args:
            x: Input token sequence, shape (B, H*W, C).

        Returns:
            Reduced token sequence, shape (B, (H/2)*(W/2), 2C).
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        assert C == self.dim, "Input channel dimension has wrong size"

        x_image = x.view(B, H, W, C)

        # Transformer path: gather 2x2 patch tokens and project
        x0 = x_image[:, 0::2, 0::2, :]
        x1 = x_image[:, 1::2, 0::2, :]
        x2 = x_image[:, 0::2, 1::2, :]
        x3 = x_image[:, 1::2, 1::2, :]
        x_transformer = torch.cat([x0, x1, x2, x3], -1)
        x_transformer = x_transformer.view(B, -1, 4 * C)
        x_transformer = self.transformer_path_reduction(x_transformer)

        # CNN path: strided convolution on spatial feature map
        x_cnn = x_image.permute(0, 3, 1, 2).contiguous()
        x_cnn = self.cnn_path_conv(x_cnn)
        x_cnn = x_cnn.permute(0, 2, 3, 1).contiguous().view(B, -1, self.out_dim)

        # Dual-path fusion via element-wise addition
        x_fused = x_transformer + x_cnn
        x_fused = self.norm(x_fused)

        return x_fused
