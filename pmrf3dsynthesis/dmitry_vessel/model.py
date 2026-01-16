import torch
import torch.nn as nn
import gc
import torch.nn.functional as F

from pmrf3dsynthesis.dmitry_vessel import blocks


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample = None):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.downsample = downsample


        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, stride, stride),bias=False)

        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(1, 1, 1),bias=False)
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


class CustomNet(nn.Module):
    def __init__(self, depth, encoder_layers, number_of_channels, number_of_outputs, block):
        super(CustomNet, self).__init__()
        print('CustomNet {}'.format(number_of_channels))
        self.encoder_layers = encoder_layers
        self.block=block
        self.number_of_outputs = number_of_outputs
        self.number_of_channels = number_of_channels
        self.depth = depth

        self.encoder_convs = nn.ModuleList()
        self.upsampling = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()

        self.encoder_ratt = nn.ModuleList()
        self.decoder_ratt = nn.ModuleList()

        conv_first_list = [nn.Conv3d(in_channels=1, out_channels=self.number_of_channels[0], kernel_size=(3,1,3), stride=1,
                                   padding=(1,0,1), bias=True),
                           nn.InstanceNorm3d(num_features=number_of_channels[0], affine=True),
                           ]

        for i in range(self.encoder_layers[0]):
            conv_first_list.append(
                self.block(in_channels=self.number_of_channels[0], out_channels=self.number_of_channels[0],
                              stride=1))


        self.conv_first = nn.Sequential(*conv_first_list)

        self.conv_output = nn.Conv3d(in_channels=self.number_of_channels[0],out_channels=1,kernel_size=3, stride=1,padding=1,bias=True,groups=1)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=0.4)


        self.construct_encoder_convs(depth=depth,number_of_channels=number_of_channels)
        self.construct_upsampling_convs(depth=depth,number_of_channels=number_of_channels)
        self.construct_decoder_convs(depth=depth,number_of_channels=number_of_channels)


    def _make_encoder_layer(self, in_channels, channels, blocks, block, stride=1, ds_kernel = (2,2), ds_stride = (2,2)):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool3d(kernel_size=ds_kernel,stride=ds_stride),
                nn.Conv3d(in_channels=in_channels, out_channels=channels, kernel_size=1,bias=False)
            )

        layers = []
        layers.append(block(in_channels=channels, out_channels=channels, stride=1, downsample=downsample))

        for _ in range(1, blocks):
            layers.append(block(in_channels=channels, out_channels=channels, stride=1))

        return nn.Sequential(*layers)

    def construct_encoder_convs(self, depth, number_of_channels):
        for i in range(depth - 1):
            conv = self._make_encoder_layer(in_channels=number_of_channels[i], channels=number_of_channels[i + 1],
                                            blocks=self.encoder_layers[i + 1], stride=2, block=self.block, ds_kernel=(1,2,2), ds_stride=(1,2,2))
            self.encoder_convs.append(conv)


    def construct_decoder_convs(self, depth, number_of_channels):
        for i in range(depth):

            conv_list = []
            for j in range(self.encoder_layers[i]):
                conv_list.append(self.block(in_channels=number_of_channels[i], out_channels=number_of_channels[i], stride=1))

            dec_conv = nn.Sequential(
                *conv_list
            )

            conv= nn.Sequential(
                nn.Conv3d(in_channels=2*number_of_channels[i], out_channels=number_of_channels[i], kernel_size=3,stride=1,padding=1, bias=False),
                nn.InstanceNorm3d(num_features=number_of_channels[i],affine=True),
                dec_conv
            )
            self.decoder_convs.append(conv)

    def construct_upsampling_convs(self, depth, number_of_channels):
        for i in range(depth-1):
            conv =  nn.Sequential(
                blocks.Trilinear(scale=(1,1,2)),
                nn.Conv3d(in_channels=number_of_channels[i+1], out_channels=number_of_channels[i], kernel_size=1,stride=1,padding=0, bias=False),

            )

            self.upsampling.append(conv)

    def forward(self, x):
        skip_connections = []
        gc.collect()
        input = x['image']

        N, C, W, H, D = input.shape
        conv = self.conv_first(input)

        for i in range(self.depth - 1):
            rate = 2**(self.depth-i-1)
            skip_connections.append(F.avg_pool3d(conv,kernel_size=(1,int(rate),1),stride=(1,int(rate),1)))
            conv = self.encoder_convs[i](conv)

        #out_enc = conv.detach().cpu()

        for i in reversed(range(self.depth - 1)):
            conv = self.upsampling[i](conv)
            #rate = 2**(self.depth-i-1)
            skip = skip_connections[i]#F.avg_pool3d(skip_connections[i],kernel_size=(1,int(rate),1),stride=(1,int(rate),1))

            conc = torch.cat([skip,conv],dim=1)
            conv = self.decoder_convs[i](conc)

        #out_cnn = conv.detach().cpu()

        out = torch.mean(conv, dim=3, keepdim=True)

        s = self.conv_output(out)

        out = torch.sigmoid(s)

        return {
            'prediction': out,
            #'output_encoder': out_enc,
            #'output_cnn': out_cnn
        }
