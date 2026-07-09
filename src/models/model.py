"""
model.py
========
SpineHybridEfficient — 3D Vertebrae Instance Segmentation

Architecture:
  Encoder 1 : 3DINO ViT-Large (last 2 blocks unfrozen, rest frozen)
  Encoder 2 : ResNet50 (fully trainable)
  Fusion    : AdaptiveAvgPool3d(4) → concat → Conv3d  @  4³ bottleneck
  Decoder   : RTUpBlock × 2  +  UpBlock × 3  (NestedUNet + RT blocks)
  Head      : Conv3d → 26 vertebra classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet50


# ============================================================
# BUILDING BLOCKS
# ============================================================

class ConvBlock(nn.Module):
    """Double 3×3×3 conv → BN → ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Trilinear upsample ×2 + optional skip concat + ConvBlock."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        if skip is not None:
            skip = F.adaptive_avg_pool3d(skip, x.shape[2:])
            x    = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FinalUpBlock(nn.Module):
    """Upsample to exact target_size + optional skip + ConvBlock."""
    def __init__(self, in_ch, out_ch, target_size):
        super().__init__()
        self.target_size = target_size
        self.conv        = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, size=self.target_size, mode='trilinear', align_corners=False)
        if skip is not None:
            skip = F.adaptive_avg_pool3d(skip, self.target_size)
            x    = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RTUpBlock(nn.Module):
    """
    Residual Transformer UpBlock.

    Fuses a 3×3 CNN branch and a multi-head self-attention branch
    residually, then upsamples ×2 and injects skip connection.

    Only used at spatial resolution ≤ 16³ (~4096 tokens) to keep
    attention cost tractable.
    """
    def __init__(self, in_ch, out_ch, num_heads=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_ch), nn.ReLU(inplace=True))
        self.transformer = nn.TransformerEncoderLayer(
            d_model=in_ch, nhead=num_heads, dim_feedforward=in_ch * 4,
            dropout=0.1, batch_first=True, norm_first=True)
        self.proj    = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 1, bias=False), nn.BatchNorm3d(in_ch))
        self.act     = nn.ReLU(inplace=True)
        self.up_conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip=None):
        B, C, D, H, W = x.shape
        cnn_out = self.cnn(x)
        flat    = x.flatten(2).permute(0, 2, 1)               # (B, tokens, C)
        attn    = self.transformer(flat)
        attn    = attn.permute(0, 2, 1).reshape(B, C, D, H, W)
        x       = self.act(cnn_out + self.proj(attn) + x)     # residual fusion
        x       = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        if skip is not None:
            skip = F.adaptive_avg_pool3d(skip, x.shape[2:])
            x    = torch.cat([x, skip], dim=1)
        return self.up_conv(x)


# ============================================================
# DECODER
# Spatial flow for 128³ input:
#   fused (B,128,  4,  4,  4)   bottleneck
#   skip2 (B,512,  8,  8,  8)   ResNet layer2
#   skip1 (B,256, 16, 16, 16)   ResNet layer1
#
#   up1 RTUpBlock :  4 →  8  cat skip2(512) → 128ch
#   up2 RTUpBlock :  8 → 16  cat skip1(256) →  64ch
#   up3 UpBlock   : 16 → 32                  →  32ch
#   up4 UpBlock   : 32 → 64                  →  16ch
#   up5 FinalUp   : 64 → 128                 →  16ch
# ============================================================
class Decoder(nn.Module):
    def __init__(self, target_size=128):
        super().__init__()
        self.up1     = RTUpBlock(128, 128)
        self.up1.up_conv = ConvBlock(128 + 512, 128)
        self.up2     = RTUpBlock(128, 64)
        self.up2.up_conv = ConvBlock(128 + 256, 64)
        self.dropout = nn.Dropout3d(p=0.25)
        self.up3     = UpBlock(64, 32)
        self.up4     = UpBlock(32, 16)
        self.up5     = FinalUpBlock(16, 16, (target_size,) * 3)

    def forward(self, fused, skip1, skip2):
        x = self.up1(fused, skip2)
        x = self.up2(x,    skip1)
        x = self.dropout(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x


# ============================================================
# MAIN MODEL
# ============================================================
class SpineHybridEfficient(nn.Module):
    """
    SpineHybridEfficient: dual-encoder segmentation network.

    Args:
        encoder     : 3DINO ViT-Large encoder (pre-loaded)
        num_classes : vertebra classes including background (default 26)
        feat_dim    : 3DINO output feature dimension (default 1024)
        target_size : output spatial size in voxels (default 128)
        dino_size   : fixed 3DINO input size — never change (default 112)
    """
    def __init__(self, encoder, num_classes=26, feat_dim=1024,
                 target_size=128, dino_size=112):
        super().__init__()
        self.encoder   = encoder
        self.dino_size = dino_size

        # ResNet50 local encoder
        resnet = resnet50(pretrained=False, spatial_dims=3,
                          n_input_channels=1, num_classes=2)
        self.resnet_stem   = nn.Sequential(resnet.conv1, resnet.bn1,
                                           nn.ReLU(inplace=True), resnet.maxpool)
        self.resnet_layer1 = resnet.layer1   # (B, 256, H/4, H/4, H/4)
        self.resnet_layer2 = resnet.layer2   # (B, 512, H/8, H/8, H/8)

        # DINO projection: 1024 → 512 → 128, pool to 4³
        self.dino_proj = nn.Sequential(
            nn.Conv3d(feat_dim, 512, 1, bias=False),
            nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.20),
            nn.Conv3d(512, 128, 1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(4))

        # ResNet layer2 projection: 512 → 128, pool to 4³
        self.resnet_proj = nn.Sequential(
            nn.Conv3d(512, 128, 1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(4))

        # Fusion at 4³: (128 + 128) → 128
        self.fusion = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.20))

        self.decoder  = Decoder(target_size=target_size)

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv3d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.20),
            nn.Conv3d(32, num_classes, 1))

    def get_dino_features(self, x):
        """
        Extract 3DINO features.
        Always resizes input to dino_size³ regardless of input resolution.
        Runs under no_grad if encoder is fully frozen.
        """
        x_r   = F.interpolate(x.float(), size=(self.dino_size,) * 3,
                              mode='trilinear', align_corners=False)
        feats = self.encoder.get_intermediate_layers(
            x_r.float(), n=4, reshape=True, return_class_token=False, norm=True)
        return feats[-1].float()   # (B, 1024, 7, 7, 7)

    def forward(self, x):
        # --- Global context from frozen/partially-unfrozen DINO ---
        dino   = self.dino_proj(self.get_dino_features(x))    # (B,128,4,4,4)

        # --- Local detail from ResNet ---
        r      = self.resnet_stem(x.float())
        skip1  = self.resnet_layer1(r)                         # (B,256,16,16,16)
        skip2  = self.resnet_layer2(skip1)                     # (B,512, 8, 8, 8)
        r_proj = self.resnet_proj(skip2)                       # (B,128, 4, 4, 4)

        # --- Fuse at 4³ ---
        fused  = self.fusion(torch.cat([dino, r_proj], dim=1)) # (B,128, 4, 4, 4)

        # --- Decode ---
        return self.seg_head(self.decoder(fused, skip1, skip2))


# ============================================================
# BUILDER
# ============================================================
def build_model(dino_weights_path, device, num_classes=26,
                target_size=128, dino_size=112,
                unfreeze_last_n_blocks=2):
    """
    Load 3DINO encoder and build SpineHybridEfficient.

    Args:
        dino_weights_path     : path to 3DINO .pth checkpoint
        device                : torch.device
        num_classes           : total classes (background + vertebrae)
        target_size           : output spatial size (match training resolution)
        dino_size             : 3DINO fixed input size — never change (112)
        unfreeze_last_n_blocks: number of 3DINO transformer blocks to unfreeze
                                (0 = fully frozen, 2 = best result)
    Returns:
        model   : SpineHybridEfficient on device
        encoder : 3DINO ViT-Large encoder on device
    """
    from dinov2.models.vision_transformer import vit_large_3d

    print("Loading 3DINO ViT-Large encoder...")
    ckpt    = torch.load(dino_weights_path, map_location='cpu', weights_only=False)
    raw_sd  = ckpt.get('teacher', ckpt)
    sd      = {k.replace('backbone.', ''): v for k, v in raw_sd.items()}
    encoder = vit_large_3d(patch_size=16, img_size=dino_size, block_chunks=1)
    encoder.load_state_dict(sd, strict=False)
    encoder = encoder.float().to(device).eval()

    # Freeze all blocks first
    for p in encoder.parameters():
        p.requires_grad = False

    # Unfreeze last N transformer blocks
    if unfreeze_last_n_blocks > 0:
        for block in encoder.blocks[-unfreeze_last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    unfrozen = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  Unfrozen DINO params : {unfrozen:,}")
    print(f"  Free GPU after DINO  : {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

    model     = SpineHybridEfficient(encoder, num_classes=num_classes,
                                     target_size=target_size,
                                     dino_size=dino_size).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable      : {trainable:,}\n")

    return model, encoder
