import math

import torch.nn as nn


class GeneratorDCGAN(nn.Module):

    def __init__(self, config):

        super(GeneratorDCGAN, self).__init__()

        # TODO: Add batchnorm usage to config
        # TODO: Add leaky relu
        # TODO: Add dropout

        out_features = int(math.pow(config.image_size / math.pow(2, 3), 2)) * 64
        self.dense = nn.Linear(in_features=config.zdim, out_features=out_features)

        self.conv1_t = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                          kernel_size=3, padding=1, output_padding=1, stride=2)
        self.conv2_t = nn.ConvTranspose2d(in_channels=self.conv1_t.out_channels, out_channels=16,
                                          kernel_size=3, padding=1, output_padding=1, stride=2)
        self.conv3_t = nn.ConvTranspose2d(in_channels=self.conv2_t.out_channels, out_channels=config.input_channels,
                                          kernel_size=3, padding=1, output_padding=1, stride=2)

        self.relu = nn.ReLU()

        # TODO: Add config/model architecture to use upsample to conv2d kernel 1x1
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        self.tanh = nn.Tanh()

    def forward(self, x):

        # (64 * 6 * 6) flattened
        x = self.dense(x)
        x = self.relu(x)

        # TODO: Avoid hardcoded numbers, set 64 from config or set upsampling steps and G starting / D final filters
        x = x.view(x.size(0), 64, 6, 6)

        # 32 x 12 x 12
        x = self.conv1_t(x)
        x = self.relu(x)

        # 16 x 24 x 24
        x = self.conv2_t(x)
        x = self.relu(x)

        # input_channels x 48 x 48
        x = self.conv3_t(x)

        out = self.tanh(x)

        return out
