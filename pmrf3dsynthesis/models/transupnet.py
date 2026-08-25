import torch
import torch.nn as nn
from pmrf3dsynthesis.models.transunet import PretrainedViTNeck
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        ) 
        self.conv = nn.Sequential(
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        )
        self.final_conv = nn.Conv3d(out_channels, 1, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.upsample(x)
        skip = self.upsample(skip)

        x = x + self.init_conv(x)
        x = x + self.conv(x)

        return x, self.final_conv(x) + skip

class TransUpNet(nn.Module):
    def __init__(self, in_channels=1, kernel_size=8, c_hat=12, num_resolution=3,
                 slides=256, ct_transform=False):
        super().__init__()
        self.ct_transform = ct_transform
        self.patch_extract = nn.Conv2d(in_channels=in_channels, out_channels=c_hat * (kernel_size ** 2),
                                       kernel_size=kernel_size, stride=kernel_size)
        
        vit_channels = c_hat * (kernel_size ** 2)
        self.vit = PretrainedViTNeck(in_channels=vit_channels, hidden_size=vit_channels)

        self.slides = slides
        self.num_resolution = num_resolution
        num_features = vit_channels // (self.slides // (2 ** self.num_resolution))

        self.init_head = nn.Conv3d(num_features,
                                   1, kernel_size=3, padding=1)

        self.dec = nn.ModuleList()
        for i in range(num_resolution):
            self.dec.append(DecoderBlock(num_features, 1, num_features))
 


    def forward(self, x):
        B, C, H, W = x.shape

        x_patch = self.patch_extract(x)        
        # Bottleneck
        enc_out = self.vit(x_patch)

        depth = self.slides // (2 ** self.num_resolution)
        channels = enc_out.shape[1] // depth
        if self.ct_transform:
            dec_input = enc_out.view(B, channels, depth, H // 8, W // 8)
        else:
            # H in fundus is depth in oct
            dec_input = enc_out.view(B, channels, depth, H // 8, W // 8).transpose(2, 3)

        predictions = [self.init_head(dec_input)]
        x_cur = dec_input
        for i in range(self.num_resolution):
            x_cur, pred = self.dec[i](x_cur, predictions[i])      
            predictions.append(pred)

        return predictions


if __name__ == "__main__":
    model = TransUpNet()
    x = torch.randn(1, 1, 128, 256)
    preds = model(x)
    for p in preds:
        print(p.shape)