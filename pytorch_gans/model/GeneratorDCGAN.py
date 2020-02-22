import math

import torch.nn as nn


class GeneratorDCGAN(nn.Module):

    def __init__(self, config):

        super(GeneratorDCGAN, self).__init__()

        # TODO: Add to config first/last filters, e.g. first is 1024
        # TODO: Add to config number of upsample steps

        # DCGAN: first feature map image_size / 2^4 (number of conv_t = 4),
        # first dense 1024 channels, then conv1_t 512, conv2_t 256, conv3_t 128, conv4_t image_channels
        # TODO: Add dropout
        # TODO: Calculate padding to support any image_size (e.g. for 48 probably we need to remove a layer,
        # because 3 x 3 map is smaller than kernel)

        conv_number = 4
        self.min_map_size = int(config.image_size / math.pow(2, conv_number))
        print(f"Generator minimum map size: {self.min_map_size}")
        dense_out_features = int(math.pow(self.min_map_size, 2)) * 1024
        self.dense = nn.Linear(in_features=config.zdim, out_features=dense_out_features, bias=False)
        self.conv1_t = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                          kernel_size=4, padding=1, output_padding=0, stride=2, bias=False)
        # Need to set kernel 5x5 from now on to keep dimensions
        self.conv2_t = nn.ConvTranspose2d(in_channels=self.conv1_t.out_channels,
                                          out_channels=self.conv1_t.out_channels//2,
                                          kernel_size=5, padding=2, output_padding=1, stride=2, bias=False)
        self.conv3_t = nn.ConvTranspose2d(in_channels=self.conv2_t.out_channels,
                                          out_channels=self.conv2_t.out_channels//2,
                                          kernel_size=5, padding=2, output_padding=1, stride=2, bias=False)
        self.conv4_t = nn.ConvTranspose2d(in_channels=self.conv3_t.out_channels,
                                          out_channels=config.input_channels,
                                          kernel_size=5, padding=2, output_padding=1, stride=2, bias=False)

        self.bn_d = nn.BatchNorm1d(self.dense.out_features)
        self.bn1 = nn.BatchNorm2d(self.conv1_t.out_channels)
        self.bn2 = nn.BatchNorm2d(self.conv2_t.out_channels)
        self.bn3 = nn.BatchNorm2d(self.conv3_t.out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        # TODO: Add config/model architecture to use upsample to conv2d kernel 1x1
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        self.tanh = nn.Tanh()

    def forward(self, x):
        # If image shape to generate is 1 x 48 x 48, first map size is 3 x 3
        # If image shape to generate is 1 x 64 x 64, first map size is 4 x 4
        # (1024 * 4 * 4) flattened
        x = self.dense(x)
        x = self.bn_d(x)
        x = self.activation(x)

        # Reshape to 1024 x 4 x 4
        x = x.view(x.size(0), 1024, self.min_map_size, self.min_map_size)

        # 512 x 8 x 8
        x = self.conv1_t(x)
        x = self.bn1(x)
        x = self.activation(x)

        # 256 x 16 x 16
        x = self.conv2_t(x)
        x = self.bn2(x)
        x = self.activation(x)

        # 128 x 32 x 32
        x = self.conv3_t(x)
        x = self.bn3(x)
        x = self.activation(x)

        # input_channels x 64 x 64
        x = self.conv4_t(x)

        out = self.tanh(x)

        return out
