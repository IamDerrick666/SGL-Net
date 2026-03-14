import torch
import torch.nn as nn
from einops import rearrange


class DualPathRestoration(nn.Module):
    """Dual-Path Restoration (DPRestore).

    A heterogeneous upsampling module that fuses a Transformer sub-pixel
    expansion path with a transposed-convolution CNN path, doubling spatial
    resolution while halving channel depth. Provides local geometry inductive
    bias during the decoding stage.

    Args:
        input_resolution (tuple[int]): Spatial resolution (H, W) of input tokens.
        dim (int): Input channel dimension.
        dim_scale (int): Upsampling scale factor. Default: 2.
        norm_layer: Normalization layer constructor. Default: nn.LayerNorm.
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand_dim = int(dim_scale ** 2 * (dim // dim_scale))
        self.out_dim = dim // dim_scale

        # Transformer path: linear expansion followed by pixel-shuffle rearrangement
        self.transformer_path_expand = nn.Linear(dim, self.expand_dim, bias=False)

        # CNN path: transposed convolution for learned spatial upsampling
        self.cnn_path_upsample = nn.ConvTranspose2d(dim, self.out_dim, kernel_size=2, stride=2)

        self.norm = norm_layer(self.out_dim)

    def forward(self, x):
        """
        Args:
            x: Input token sequence, shape (B, H*W, C).

        Returns:
            Restored token sequence, shape (B, (2H)*(2W), C/2).
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert C == self.dim, "input channel dimension has wrong size"

        # Transformer path: sub-pixel expansion
        x_transformer = self.transformer_path_expand(x)
        x_transformer = rearrange(
            x_transformer,
            'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
            h=H, w=W, p1=2, p2=2, c=self.out_dim
        )
        x_transformer = x_transformer.view(B, -1, self.out_dim)

        # CNN path: transposed convolution upsampling
        x_cnn = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_cnn = self.cnn_path_upsample(x_cnn)
        x_cnn = x_cnn.permute(0, 2, 3, 1).contiguous().view(B, -1, self.out_dim)

        # Dual-path fusion via element-wise addition
        x_fused = x_transformer + x_cnn
        x_fused = self.norm(x_fused)

        return x_fused
