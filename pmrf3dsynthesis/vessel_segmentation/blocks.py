import torch
import torch.nn as nn
import torch.nn.functional as F

class LCNorm(nn.Module):
    def __init__(self):
        super(LCNorm, self).__init__()

    def forward(self, x):
        x2_mean = F.avg_pool3d(x * x, kernel_size=9, padding=4, stride=1)
        x_mean = F.avg_pool3d(x, kernel_size=9, padding=4, stride=1)
        x_std = torch.sqrt(x2_mean - x_mean * x_mean + 0.03)
        x = (x - x_mean) / (x_std)
        return x

class Trilinear(nn.Module):
    def __init__(self, scale):
        super(Trilinear, self).__init__()
        self.scale = scale

    def forward(self, x):
        out = F.interpolate(x,scale_factor=self.scale,mode='trilinear')
        return out


class Bilinear(nn.Module):
    def __init__(self, scale):
        super(Bilinear, self).__init__()
        self.scale = scale

    def forward(self, x):
        out = F.interpolate(x,scale_factor=self.scale,mode='bilinear')
        return out

class Mean(nn.Module):
    def __init__(self, dim):
        super(Mean, self).__init__()
        self.dim = dim

    def forward(self, x):
        out = torch.mean(x,dim=self.dim, keepdim=True)
        return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.att = nn.Sequential(
            nn.Conv3d(in_channels=channel, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)

        att = self.att(x)

        return x * att * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, kernel_size = None, padding = None):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample


        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 5, 1), padding=(0, 2, 0), stride=(1, stride, stride),bias=False)

        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1), stride=(1, 1, 1),bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        self.norm1 = nn.InstanceNorm3d(num_features=out_channels,affine=True)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels,affine=True)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)


        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = x + out

        return out

class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, kernel_size = 3, padding = 1):
        super(ResidualBlock2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample


        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=(1, stride, stride),bias=False)

        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=1,bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        self.norm1 = nn.InstanceNorm3d(num_features=out_channels,affine=True)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels,affine=True)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)


        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = x + out

        return out

class Residual_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, kernel_size = None, padding = None):
        super(Residual_bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        compression = 4
        #groups = 8

        if kernel_size is None:
            kernel_size = (1,3,1)
            padding = (0,1,0)

        self.conv_pre = nn.Conv3d(in_channels=in_channels, out_channels=out_channels//compression, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.conv1 = nn.Conv3d(in_channels=out_channels//compression, out_channels=out_channels//compression, kernel_size=kernel_size, stride=1, padding=padding, bias=False, groups=1)

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels//compression,affine=True)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels//compression,affine=True)
        self.conv_post = nn.Conv3d(in_channels=out_channels // compression, out_channels=out_channels,
                               kernel_size=1, padding=0, bias=False)
        self.norm_post = nn.InstanceNorm3d(num_features=out_channels,affine=True)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv_pre(out)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv_post(out)
        out = self.norm_post(out)

        out = x + out

        return out

class Residual_bottleneckLRelu(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, kernel_size = None, padding = None):
        super(Residual_bottleneckLRelu, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        compression = 4
        #groups = 8

        if kernel_size is None:
            kernel_size = 3#(1,3,1)
            padding = 1#(0,1,0)

        self.conv_pre = nn.Conv3d(in_channels=in_channels, out_channels=out_channels//compression, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.conv1 = nn.Conv3d(in_channels=out_channels//compression, out_channels=out_channels//compression, kernel_size=kernel_size, stride=1, padding=padding, bias=False, groups=1)

        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels//compression,affine=True)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels//compression,affine=True)
        self.conv_post = nn.Conv3d(in_channels=out_channels // compression, out_channels=out_channels,
                               kernel_size=1, padding=0, bias=False)
        self.norm_post = nn.InstanceNorm3d(num_features=out_channels,affine=True)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv_pre(out)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv_post(out)
        out = self.norm_post(out)

        out = x + out

        return out

class Residual_bottleneckLRelu2(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, kernel_size = None, padding = None):
        super(Residual_bottleneckLRelu2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        compression = 4
        #groups = 8

        if kernel_size is None:
            kernel_size = (1,5,1)
            padding = (0,2,0)

        self.conv_pre = nn.Conv3d(in_channels=in_channels, out_channels=out_channels//compression, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels//compression, out_channels=out_channels//compression, kernel_size=kernel_size, stride=1, padding=padding, bias=False, groups=1),
            nn.Conv3d(in_channels=out_channels//compression, out_channels=out_channels//compression, kernel_size=(3,1,3), stride=1, padding=(1,0,1), bias=False, groups=1)
        )

        self.relu1 = nn.LeakyReLU(inplace=False)
        self.relu2 = nn.LeakyReLU(inplace=False)
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels//compression,affine=True)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels//compression,affine=True)
        self.conv_post = nn.Conv3d(in_channels=out_channels // compression, out_channels=out_channels,
                               kernel_size=1, padding=0, bias=False)
        self.norm_post = nn.InstanceNorm3d(num_features=out_channels,affine=True)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv_pre(out)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv_post(out)
        out = self.norm_post(out)

        out = x + out

        return out

class Residual_bottleneck_bn(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, kernel_size = None, padding = None):
        super(Residual_bottleneck_bn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        compression = 8
        #groups = 8

        if kernel_size is None:
            kernel_size = (1,3,3)
            padding = (0,1,1)

        self.conv_pre = nn.Conv3d(in_channels=in_channels, out_channels=out_channels//compression, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.conv1 = nn.Conv3d(in_channels=out_channels//compression, out_channels=out_channels//compression, kernel_size=kernel_size, stride=1, padding=padding, bias=False, groups=1)

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.norm1 = nn.InstanceNorm3d(num_features=out_channels//compression,affine=True)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels//compression,affine=True)
        self.conv_post = nn.Conv3d(in_channels=out_channels // compression, out_channels=out_channels,
                               kernel_size=1, padding=0, bias=False)
        self.norm_post = nn.InstanceNorm3d(num_features=out_channels,affine=True)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv_pre(out)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv_post(out)
        out = self.norm_post(out)

        out = x + out

        return out

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None):
        super(VGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 5, 1), padding=(0, 2, 0), stride=(1, stride, stride),bias=False)

        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1), stride=(1, 1, 1),bias=False)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.norm1 = nn.InstanceNorm3d(num_features=out_channels,affine=True)#nn.GroupNorm(num_groups=4,num_channels=out_channels)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels,affine=True)#nn.GroupNorm(num_groups=4,num_channels=out_channels)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)


        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        return out

class VGGBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None, kernel_size = None, padding = None):
        super(VGGBlock2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels


        if kernel_size is None:
            kernel_size = (1,1,3)
            padding = (0,0,1)

        self.downsample = downsample

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 5, 1), padding=(0, 2, 0), stride=(1, stride, stride),bias=False)

        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1), stride=(1, 1, 1),bias=False)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.norm1 = nn.InstanceNorm3d(num_features=out_channels,affine=True)#nn.GroupNorm(num_groups=4,num_channels=out_channels)
        self.norm2 = nn.InstanceNorm3d(num_features=out_channels,affine=True)#nn.GroupNorm(num_groups=4,num_channels=out_channels)

    def forward(self, x):

        if self.downsample is not None:
            x = self.downsample(x)

        out = x
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)


        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        return out