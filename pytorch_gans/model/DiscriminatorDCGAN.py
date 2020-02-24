import math
from collections import OrderedDict

import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=5, padding=2, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


class DiscriminatorDCGAN(nn.Module):

    def __init__(self, config):

        # TODO: Add dropout

        super(DiscriminatorDCGAN, self).__init__()

        conv_number = config.d_resize_steps

        # Init convolutional blocks
        convs_blocks_dict = OrderedDict()
        in_channels = config.input_channels
        out_channels = config.d_initial_filters
        for i in range(conv_number):
            convs_blocks_dict[f"block{i+1}"] = ConvBlock(in_channels=in_channels, out_channels=out_channels)
            in_channels = out_channels
            out_channels *= 2

        # Convolution blocks as a single sequential model
        self.conv_blocks = nn.Sequential(convs_blocks_dict)

        # self.conv_blocks = nn.Sequential(convs_blocks)
        self.min_map_size = int(config.image_size / math.pow(2, conv_number))
        print(f"Discriminator minimum map size: {self.min_map_size}")
        last_conv_channels = out_channels // 2
        dense_in_features = int(math.pow(self.min_map_size, 2)) * last_conv_channels
        self.dense = nn.Linear(in_features=dense_in_features, out_features=1, bias=False)

        self.output = nn.Sigmoid()

    def forward(self, x):
        # Every block double the number of filters and halves the width and the height of the map
        # Default start from 64 x 64 x 64 CHW
        x = self.conv_blocks(x)

        # flatten (e.g. 512 * 4 * 4)
        x = x.view(x.size(0), -1)

        # 1
        x = self.dense(x)

        # Final activation
        out = self.output(x)

        return out
