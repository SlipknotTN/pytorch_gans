import math
from collections import OrderedDict

import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=5, padding=2, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.pooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pooling(x)
        return x


class DiscriminatorCDCGAN(nn.Module):

    def __init__(self, config, num_classes):

        # TODO: Add dropout

        super(DiscriminatorCDCGAN, self).__init__()

        # Conditional GAN has an additional input: class_idx -> embedding -> dense -> reshape to image size (one channel)
        # See this: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/05/Plot-of-the-Discriminator-Model-in-the-Conditional-Generative-Adversarial-Network-664x1024.png

        # Class input embedding and dense layer
        self.embedding = nn.Embedding(num_classes, config.embedding_size)
        self.class_dense = nn.Linear(in_features=config.embedding_size,
                                     out_features=config.image_size * config.image_size * 1,
                                     bias=False)

        conv_number = config.d_resize_steps

        # Init convolutional blocks, we add one channel for the class input branch
        convs_blocks_dict = OrderedDict()
        in_channels = config.input_channels + 1
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

    def forward(self, images, class_indexes):

        # Process class_idx
        y = self.embedding(class_indexes)
        y = self.class_dense(y)
        y = nn.ReLU()(y)
        # Reshape to the same image shape (forcing 1 channel)
        y = y.view(-1, 1, images.size(2), images.size(3))

        # Concatenate image with class_output along channel dimension
        z = torch.cat((images, y), 1)

        # Every block double the number of filters and halves the width and the height of the map
        # Default start from 64 x 64 x 64 CHW
        z = self.conv_blocks(z)

        # flatten (e.g. 512 * 4 * 4)
        z = z.view(z.size(0), -1)

        # 1
        z = self.dense(z)

        # Final activation
        out = self.output(z)

        return out
