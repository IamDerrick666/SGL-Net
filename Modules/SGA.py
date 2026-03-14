import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticGuidedAligner(nn.Module):
    """Semantic-Guided Aligner (SGA).

    Acts as a smart cross-attention gate between encoder skip features and
    decoder query features, routing semantically relevant information across
    the skip connections while suppressing noisy or irrelevant features.

    Args:
        dim_x (int): Channel dimension of encoder skip features.
        dim_g (int): Channel dimension of decoder query (gating) features.
        num_heads (int): Number of attention heads. Default: 8.
        qkv_bias (bool): If True, add learnable bias to q, k, v projections. Default: False.
        norm_layer: Normalization layer constructor. Default: nn.LayerNorm.
    """

    def __init__(self, dim_x, dim_g, num_heads=8, qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_x // num_heads
        self.scale = head_dim ** -0.5

        # Decoder query projects into the encoder feature space
        self.q_proj = nn.Linear(dim_g, dim_x, bias=qkv_bias)
        self.k_proj = nn.Linear(dim_x, dim_x, bias=qkv_bias)
        self.v_proj = nn.Linear(dim_x, dim_x, bias=qkv_bias)

        self.proj = nn.Linear(dim_x, dim_x)

        # Normalization layers
        self.norm_x = norm_layer(dim_x)
        self.norm_g = norm_layer(dim_g)

    def forward(self, x, g):
        """
        Args:
            x: Encoder skip features, shape (B, L, C_x).
            g: Decoder gating features, shape (B, L, C_g).

        Returns:
            Semantically filtered encoder features, shape (B, L, C_x).
        """
        B, L_x, C_x = x.shape
        B, L_g, C_g = g.shape
        assert L_x == L_g, (
            f"Token lengths of encoder feature ({L_x}) and decoder feature ({L_g}) must be the same."
        )

        x_norm = self.norm_x(x)
        g_norm = self.norm_g(g)

        # Decoder features generate queries; encoder features generate keys/values
        q = self.q_proj(g_norm).reshape(B, L_g, self.num_heads, C_x // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(x_norm).reshape(B, L_x, self.num_heads, C_x // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x_norm).reshape(B, L_x, self.num_heads, C_x // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        gated_x = (attn @ v).transpose(1, 2).reshape(B, L_g, C_x)
        gated_x = self.proj(gated_x)

        # Residual connection: return refined skip features
        return x + gated_x
