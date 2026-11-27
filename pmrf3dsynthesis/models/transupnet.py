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
    is fed at its saved resolution (e.g. 850x850) and convolutions trained from
    scratch reduce it to the en-face grid. Each stage halves every axis that is
    still at least twice the en-face size; an adaptive average pool then removes the
    remaining non-integer factor (850x850 -> 213x425 -> 128x256). Crops that are an
    exact power-of-two multiple of the grid need no pooling.
    """

    def __init__(self, in_channels, out_channels, fundus_size, enface_size, num_groups=8):
        super().__init__()
        h, w = fundus_size
        th, tw = enface_size
        if h < th or w < tw:
            raise ValueError(f"fundus_size {tuple(fundus_size)} must be at least enface_size {tuple(enface_size)}.")

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(inplace=True),
        ]
        while h >= 2 * th or w >= 2 * tw:
            sh, sw = (2 if h >= 2 * th else 1), (2 if w >= 2 * tw else 1)
            layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=(sh, sw), padding=1),
                nn.GroupNorm(num_groups, out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, out_channels),
                nn.SiLU(inplace=True),
            ]
            # conv output size with kernel 3, padding 1
            h, w = (h - 1) // sh + 1, (w - 1) // sw + 1
        if (h, w) != (th, tw):
            layers.append(nn.AdaptiveAvgPool2d((th, tw)))
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
        # crop of that size (any size >= enface_size, e.g. 850x850) and a FundusStem maps
        # it to enface_size before patch extraction.
        self.fundus_size = tuple(fundus_size) if fundus_size is not None else None
        self.enface_size = tuple(enface_size)
        if self.fundus_size is not None:
            self.stem = FundusStem(in_channels, stem_channels, self.fundus_size, self.enface_size)
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