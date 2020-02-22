import math

import torch.nn as nn


class DiscriminatorDCGAN(nn.Module):

    def __init__(self, config):

        # TODO: Add to config first convolution filters number, e.g. first is 64
        # TODO: Add to config number of downsample steps

        # TODO: Add dropout

        super(DiscriminatorDCGAN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=config.input_channels, out_channels=64,
                               kernel_size=5, padding=2, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=self.conv1.out_channels * 2,
                               kernel_size=5, padding=2, stride=1)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=self.conv2.out_channels * 2,
                               kernel_size=5, padding=2, stride=1)
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=self.conv3.out_channels * 2,
                               kernel_size=5, padding=2, stride=1)

        conv_number = 4
        self.min_map_size = int(config.image_size / math.pow(2, conv_number))
        print(f"Discriminator minimum map size: {self.min_map_size}")
        dense_in_features = int(math.pow(self.min_map_size, 2)) * self.conv4.out_channels  # 64 * int(math.pow(conv_number, 2))
        self.dense = nn.Linear(in_features=dense_in_features, out_features=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)

        self.activation = nn.LeakyReLU(0.2)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # 64 x 64 x 64
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        # 64 x 32 x 32
        x = self.max_pool(x)

        # 128 x 32 x 32
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        # 128 x 16 x 16
        x = self.max_pool(x)

        # 256 x 16 x 16
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        # 256 x 8 x 8
        x = self.max_pool(x)

        # 512 x 8 x 8
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        # 512 x 4 x 4
        x = self.max_pool(x)

        # flatten (512 * 4 * 4)
        x = x.view(x.size(0), -1)

        # 1
        x = self.dense(x)

        out = self.sigmoid(x)

        return out
