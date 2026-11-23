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

    def features(self, x):
        x = self.upsample(x)
        x = x + self.init_conv(x)
        x = x + self.conv(x)
        return x

    def forward(self, x, skip):
        x = self.features(x)
        skip = self.upsample(skip)

        return x, self.final_conv(x) + skip

def fundus_input_size(model, default=(128, 256)):
    """Fundus size (H, W) that an MMSE or flow model expects as input.

    A TransUpNet with `fundus_size` (full-resolution stem) takes the crop at that size;
    every other model takes it on the en-face grid `default`.
    """
    mmse = getattr(model, "mmse_model", None)
    if mmse is not None:
        model = mmse
    size = getattr(getattr(model, "network", None), "fundus_size", None)
    return tuple(size) if size is not None else tuple(default)


class FundusStem(nn.Module):
    """Learned downsampling of a full-resolution fundus crop to the OCT en-face grid.

    TransUpNet ties one fundus row to one B-scan, so by default the fundus is resized
    to the en-face grid (128x256) before the network sees it. With the stem the crop
    is fed at its saved resolution and strided convolutions (trained from scratch)
    reduce it to the en-face grid. Each stage halves every axis that still needs
    it, e.g. a 512x512 crop reaches 128x256 with strides (2, 2) and then (2, 1).
    """

    def __init__(self, in_channels, out_channels, factors, num_groups=8):
        super().__init__()
        fh, fw = factors
        for f in (fh, fw):
            if f < 1 or (f & (f - 1)) != 0:
                raise ValueError(f"fundus_size / enface_size must be a power of two per axis, got {tuple(factors)}.")

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(inplace=True),
        ]
        while fh > 1 or fw > 1:
            sh, sw = (2 if fh > 1 else 1), (2 if fw > 1 else 1)
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=(sh, sw), padding=1),
                nn.GroupNorm(num_groups, out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, out_channels),
                nn.SiLU(inplace=True),
            ]
            fh //= sh
            fw //= sw
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TransUpNet(nn.Module):
    def __init__(self, in_channels=1, kernel_size=8, c_hat=12, num_resolution=3,
                 slides=256, ct_transform=False, residual_prediction=True,
                 fundus_size=None, enface_size=(128, 256), stem_channels=32):
        super().__init__()
        # fundus_size=None (default): the input is already on the en-face grid (legacy model,
        # checkpoints load unchanged). fundus_size=(H, W): the input is a full-resolution
        # crop of that size and a FundusStem maps it to enface_size before patch extraction.
        self.fundus_size = tuple(fundus_size) if fundus_size is not None else None
        self.enface_size = tuple(enface_size)
        if self.fundus_size is not None:
            if self.fundus_size[0] % self.enface_size[0] or self.fundus_size[1] % self.enface_size[1]:
                raise ValueError(f"fundus_size {self.fundus_size} must be a multiple of enface_size {self.enface_size}.")
            factors = (self.fundus_size[0] // self.enface_size[0], self.fundus_size[1] // self.enface_size[1])
            self.stem = FundusStem(in_channels, stem_channels, factors)
            in_channels = stem_channels
        else:
            self.stem = None
        # residual_prediction=True (default): every decoder block adds its output to the
        # upsampled prediction of the coarser scale and all scales are returned.
        # residual_prediction=False (ablation): no coarse predictions are formed; the image
        # is produced by the last decoder block alone and is the only returned prediction.
        self.residual_prediction = residual_prediction
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
        if self.stem is not None:
            if tuple(x.shape[-2:]) != self.fundus_size:
                raise ValueError(f"TransUpNet expects a fundus of size {self.fundus_size}, got {tuple(x.shape[-2:])}.")
            x = self.stem(x)
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

        if not self.residual_prediction:
            x_cur = dec_input
            for i in range(self.num_resolution):
                x_cur = self.dec[i].features(x_cur)
            return [self.dec[-1].final_conv(x_cur)]

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