import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TransformerContextEncoder(nn.Module):
    """
    Global Contextual Self-Attention Encoder for bottleneck features.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels)
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1) # (B, H*W, C)
        
        norm_x = self.norm1(x_flat)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        x_flat = x_flat + attn_out
        
        ffn_out = self.ffn(self.norm2(x_flat))
        x_flat = x_flat + ffn_out
        
        out = x_flat.permute(0, 2, 1).view(B, C, H, W)
        return out


class CrossAttentionFusion(nn.Module):
    """
    Fuses local spatial CNN features with global Transformer context features.
    """
    def __init__(self, in_c, num_heads=4):
        super().__init__()
        self.query_conv = nn.Conv2d(in_c, in_c, 1)
        self.key_conv = nn.Conv2d(in_c, in_c, 1)
        self.value_conv = nn.Conv2d(in_c, in_c, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, local_feat, global_feat):
        B, C, H, W = local_feat.shape
        proj_query = self.query_conv(local_feat).view(B, C, -1).permute(0, 2, 1) # (B, HW, C)
        proj_key = self.key_conv(global_feat).view(B, C, -1)                    # (B, C, HW)
        
        energy = torch.bmm(proj_query, proj_key)                               # (B, HW, HW)
        attention = self.softmax(energy)
        
        proj_value = self.value_conv(global_feat).view(B, C, -1)               # (B, C, HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                # (B, C, HW)
        out = out.view(B, C, H, W)
        
        fused = self.out_conv(out + local_feat)
        return fused


class ProposedUnifiedHybridModel(nn.Module):
    """
    Proposed Unified Explainable and Robust Deep Learning Architecture:
    - CNN Encoder (Local Feature Extraction)
    - Transformer Encoder (Global Context Modeling)
    - Cross-Attention Fusion
    - Nested UNet++ Decoder for Fine Boundary Refinement
    """
    def __init__(self, in_channels=4, num_classes=4):
        super().__init__()

        # Encoder Levels
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        # Transformer Bottleneck & Cross Attention
        self.transformer_bottleneck = TransformerContextEncoder(channels=512, num_heads=4)
        self.fusion = CrossAttentionFusion(in_c=512, num_heads=4)

        # UNet++ Decoder Nodes
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec3_0 = ConvBlock(256 + 512, 256)
        self.dec2_0 = ConvBlock(128 + 256, 128)
        self.dec1_0 = ConvBlock(64 + 128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)               # (B, 64, H, W)
        x2 = self.enc2(self.pool(x1))   # (B, 128, H/2, W/2)
        x3 = self.enc3(self.pool(x2))   # (B, 256, H/4, W/4)
        x4 = self.enc4(self.pool(x3))   # (B, 512, H/8, W/8)

        # Transformer Global Context & Fusion
        global_context = self.transformer_bottleneck(x4)
        fused_bottleneck = self.fusion(x4, global_context)

        # UNet++ Decoder with Skip Connections
        d3 = self.dec3_0(torch.cat([x3, self.up(fused_bottleneck)], dim=1))
        d2 = self.dec2_0(torch.cat([x2, self.up(d3)], dim=1))
        d1 = self.dec1_0(torch.cat([x1, self.up(d2)], dim=1))

        output = self.final_conv(d1)
        return output
