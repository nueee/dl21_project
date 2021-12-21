import torch
import torch.nn as nn
from torchvision import models
from torch import cat


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.input_conv1 = nn.Sequential(
            # n 3 256 256
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU()
            # n 64 256 256
        )

        self.input_conv2 = nn.Sequential(
            # n 3 256 256
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
            # n 64 256 256
        )

        self.down_conv1 = nn.Sequential(
            # n 64 256 256
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(),
            # n 128 128 128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.ReLU()
            # n 256 64 64
        )

        self.down_conv2 = nn.Sequential(
            # n 64 256 256
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            # n 128 128 128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
            # n 256 64 64
        )

        self.res_block1 = nn.Sequential(
            # n 512 64 64
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=256),
            # n 256 64 64
        )

        self.res_block2 = nn.Sequential(
            # n 512 64 64
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            # n 256 64 64
        )

        self.merge = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=256),
        )

        self.res_block3 = nn.Sequential(
            # n 512 64 64
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=256),
            # n 256 64 64
        )
        self.relu = nn.ReLU()

        self.up_conv = nn.Sequential(
            # n 256 64 64
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(),
            # n 128 128 128
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU()
            # n 64 256 256
        )

        self.output_conv = nn.Sequential(
            # n 64 256 256
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
            # n 3 256 256
        )

    def forward(self, x, y):
        x = self.input_conv1(x)
        x = self.down_conv1(x)

        y = self.input_conv2(y)
        y = self.down_conv2(y)

        for i in range(4):
            x = self.relu(self.res_block1(x) + x)  # activation after residual block
        for i in range(4):
            y = self.relu(self.res_block2(y) + y)  # activation after residual block
        z = self.relu(self.merge(torch.cat([x, y], dim=1)) + x)
        for i in range(4):
            z = self.relu(self.res_block3(z) + z)

        z = self.up_conv(z)
        z = self.output_conv(z)

        return z


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.disc_module = nn.Sequential(
            # n 3 256 256
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            # n 32 256 256
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2),
            # n 64 128 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            # n 128 64 64
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            # n 256 32 32
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2)
            # n 512 16 16
        )

        self.out_conv = nn.Sequential(
            # n 512 16 16
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=16, stride=1, padding=0)
            # n 1 1 1
        )

    def forward(self, x):
        x = self.disc_module(x)
        x = self.out_conv(x)  # to do W-GAN, do not apply sigmoid

        return x


class vgg19(nn.Module):
    def __init__(self):
        super(vgg19, self).__init__()

        vgg = models.vgg19(pretrained=True)

        self.feature = vgg.features[:37]

        for parameter in self.feature.parameters():
            parameter.require_grad = False
