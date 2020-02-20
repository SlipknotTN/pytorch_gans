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

        self.conv1_t = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv2_t = nn.Conv2d(in_channels=self.conv1_t.out_channels, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.conv3_t = nn.Conv2d(in_channels=self.conv2_t.out_channels, out_channels=config.input_channels, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

        # TODO: Add config/model architecture to use upsample to conv2d kernel 1x1
        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        self.tanh = nn.Tanh()

    def forward(self, x):

        # (6 x 6 x 64) flattened
        x = self.dense(x)
        x = self.relu(x)

        # TODO: Avoid hardcoded numbers, set 64 from config or set upsampling steps and G starting / D final filters
        x = x.view(x.size(0), 6, 6, 64)

        # 12 x 12 x 32
        x = self.conv1_t(x)
        x = self.relu(x)

        # 24 x 24 x 16
        x = self.conv2_t(x)
        x = self.relu(x)

        # 48 x 48 x input_channels
        logits = self.conv3(x)

        out = self.tanh(logits)

        return (out, logits)
