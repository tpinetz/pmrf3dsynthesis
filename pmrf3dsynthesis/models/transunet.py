import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
import math
import torch.nn.functional as F

class ResNetEncoder(nn.Module):
    """
    ResNet50 backbone that supports 1-channel input by averaging pre-trained RGB weights.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        backbone = torchvision.models.resnet50(weights=weights)
        
        if in_channels != 3:
            original_conv1 = backbone.conv1
            new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv1.weight[:] = original_conv1.weight.mean(dim=1, keepdim=True)
            backbone.conv1 = new_conv1

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.stem(x)       
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4

class PretrainedViTNeck(nn.Module):
    """
    Corrected ViT Neck that bypasses the standard torchvision Encoder.forward()
    to allow dynamic input sizes.
    """
    def __init__(self, in_channels=2048, hidden_size=768):
        super().__init__()
        
        # Load Pre-trained ViT
        vit_weights = ViT_B_16_Weights.DEFAULT
        vit_model = torchvision.models.vit_b_16(weights=vit_weights)
        
        # --- FIX: Extract internal layers to bypass fixed pos_embedding check ---
        self.transformer_layers = vit_model.encoder.layers
        self.mlp_head_norm = vit_model.encoder.ln
        
        # Extract original position embeddings for interpolation
        # Shape: (1, 197, 768)
        self.original_pos_embed = vit_model.encoder.pos_embedding 
        
        # Project ResNet features to ViT dimension
        self.patch_embedding = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: (B, 2048, H, W) -> e.g. (B, 2048, 7, 7)
        B, C, H, W = x.shape
        
        # 1. Project to Hidden Size
        x = self.patch_embedding(x) # (B, 768, H, W)
        
        # 2. Flatten: (B, Hidden, H*W) -> (B, H*W, Hidden) for Transformer
        x = x.flatten(2).transpose(1, 2) 
        
        # 3. Interpolate Positional Embeddings
        # We take the pretrained embeddings (14x14) and resize to current (HxW)
        pos_embed = self.original_pos_embed
        
        # Remove class token for interpolation (we only care about image patches)
        # Original shape: (1, 197, 768) -> (1, 196, 768)
        patch_pos_embed = pos_embed[:, 1:, :] 
        
        # Reshape to square grid for interpolation
        # (1, 196, 768) -> (1, 768, 14, 14)
        dim_orig = int(math.sqrt(patch_pos_embed.shape[1]))
        patch_pos_embed = patch_pos_embed.transpose(1, 2).reshape(1, self.hidden_size, dim_orig, dim_orig)
        
        # Interpolate to current feature map size (H, W)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(H, W), mode='bicubic', align_corners=False)
        
        # Flatten back: (1, 768, H, W) -> (1, H*W, 768)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        
        # 4. Add Embeddings and Run Transformer
        x = x + patch_pos_embed
        x = self.transformer_layers(x)
        x = self.mlp_head_norm(x)
        
        # 5. Reshape back to Spatial Map
        # (B, H*W, 768) -> (B, 768, H, W)
        x = x.transpose(1, 2).reshape(B, self.hidden_size, H, W)
        
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.skip_conv = nn.Conv2d(skip_channels, out_channels, kernel_size=1) 
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        skip = self.skip_conv(skip)
        if x.shape != skip.shape:
             x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class TransUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, img_dim=224):
        super().__init__()
        
        self.encoder = ResNetEncoder(in_channels=in_channels)
        
        # Note: in_channels=2048 matches ResNet50 layer4 output
        self.vit = PretrainedViTNeck(in_channels=2048, hidden_size=768)
        
        self.dec1 = DecoderBlock(768, 1024, 512) 
        self.dec2 = DecoderBlock(512, 512, 256) 
        self.dec3 = DecoderBlock(256, 256, 128) 
        
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        # Encoder
        f1, f2, f3, f4 = self.encoder(x)
        
        # Bottleneck
        enc_out = self.vit(f4)
        
        # Decoder
        d1 = self.dec1(enc_out, f3)
        d2 = self.dec2(d1, f2)
        d3 = self.dec3(d2, f1)
        
        # Head
        out = self.final_conv(d3)
        return out