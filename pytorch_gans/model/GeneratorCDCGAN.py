import math
from collections import OrderedDict

import torch
import torch.nn as nn


class ConvTransposeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, k_size=5, padding=2, out_padding=1):
        super(ConvTransposeBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=k_size, padding=padding,
                                       output_padding=out_padding, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class GeneratorCDCGAN(nn.Module):

    def __init__(self, config, num_classes ):

        super(GeneratorCDCGAN, self).__init__()

        # DCGAN: first feature map image_size / 2^number of conv_t,
        # Selecting 4 upsample steps and 1024 initial filters, these are the number of channels
        # first dense 1024 channels, then conv1_t 512, conv2_t 256, conv3_t 128, conv4_t image_channels

        # TODO: Add dropout
        # TODO: Calculate padding to support any image_size (e.g. for 48 probably we need to remove a layer,
        # because 3 x 3 map is smaller than kernel)

        conv_number = config.g_resize_steps
        self.min_map_size = int(config.image_size / math.pow(2, conv_number))
        print(f"Generator minimum map size: {self.min_map_size}")
        dense_out_features = int(math.pow(self.min_map_size, 2)) * config.g_initial_filters
        self.dense = nn.Linear(in_features=config.zdim, out_features=dense_out_features, bias=False)
        self.bn_d = nn.BatchNorm1d(self.dense.out_features)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        # Class input management
        # See this as reference: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/05/Plot-of-the-Generator-Model-in-the-Conditional-Generative-Adversarial-Network-806x1024.png
        self.embedding = nn.Embedding(num_classes, config.embedding_size)
        # Dense to initial 2D map, only one channel
        self.class_dense = nn.Linear(in_features=config.embedding_size,
                                     out_features=self.min_map_size * self.min_map_size * 1,
                                     bias=False)

        self.g_initial_filters = config.g_initial_filters

        # Init transpose convolutional blocks
        convs_blocks_dict = OrderedDict()
        # we add one channel for the class input branch
        in_channels = config.g_initial_filters + 1
        out_channels = in_channels // 2
        for i in range(conv_number - 1):
            # Manage first kernel size, when feature map is smaller than kernel size
            conv_t_block = ConvTransposeBlock(in_channels=in_channels, out_channels=out_channels) if i > 0 \
                else ConvTransposeBlock(in_channels=in_channels, out_channels=out_channels,
                                        k_size=4, padding=1, out_padding=0)
            convs_blocks_dict[f"block{i+1}"] = conv_t_block
            in_channels = out_channels
            out_channels //= 2

        # Convolution transpose blocks as a single sequential model
        self.conv_t_blocks = nn.Sequential(convs_blocks_dict)
        # Last convolutional block to create image
        self.conv_last_t = nn.ConvTranspose2d(in_channels=in_channels,
                                              out_channels=config.input_channels,
                                              kernel_size=5, padding=2, output_padding=1, stride=2, bias=False)

        self.output = nn.Tanh()

    def forward(self, z, class_indexes):

        # Process class_idx
        y = self.embedding(class_indexes)
        y = self.class_dense(y)
        y = nn.ReLU()(y)
        # Reshape to the same initial map shape (forcing 1 channel)
        y = y.view(-1, 1, self.min_map_size, self.min_map_size)

        # If image shape to generate is 1 x 64 x 64, first map size is 4 x 4
        # (1024 * 4 * 4) flattened
        x = self.dense(z)
        x = self.bn_d(x)
        x = self.activation(x)

        # Reshape to 1024 x 4 x 4
        x = x.view(x.size(0), self.g_initial_filters, self.min_map_size, self.min_map_size)

        # Concatenate z branch with class input branch
        x = torch.cat((x, y), 1)

        # Every block resolution is doubled (default, starting with 1024 dense out filters)
        x = self.conv_t_blocks(x)

        # input_channels x 64 x 64
        x = self.conv_last_t(x)

        out = self.output(x)

        return out
