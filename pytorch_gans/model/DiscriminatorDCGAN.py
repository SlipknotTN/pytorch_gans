import math

import torch.nn as nn


class DiscriminatorDCGAN(nn.Module):

    def __init__(self, config):

        super(DiscriminatorDCGAN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=config.input_channels, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        in_features = int(math.pow(config.image_size // math.pow(2, 3), 2)) * self.conv3.out_channels
        self.dense = nn.Linear(in_features=in_features, out_features=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # 48 x 48 x 16
        x = self.conv1(x)
        x = self.relu(x)

        # 24 x 24 x 16
        x = self.max_pool(x)

        # 24 x 24 x 32
        x = self.conv2(x)
        x = self.relu(x)

        # 12 x 12 x 32
        x = self.max_pool(x)

        # 12 x 12 x 64
        x = self.conv3(x)
        x = self.relu(x)

        # 6 x 6 x 64
        x = self.max_pool(x)

        # flatten (6 * 6 * 64)
        x = x.view(x.size(0), -1)

        logits = self.dense(x)

        out = self.sigmoid(logits)

        return (out, logits)
