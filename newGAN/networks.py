import torch.nn as nn
from torchvision import models


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.input_conv = nn.Sequential(
            # n 3 256 256
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3),
            # nn.LayerNorm([64, 256, 256]),
            nn.BatchNorm2d(64),
            nn.ReLU()
            # n 64 256 256
        )

        self.down_conv = nn.Sequential(
            # n 64 256 256
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm([128, 128, 128]),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # n 128 128 128
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm([256, 64, 64]),
            nn.BatchNorm2d(256),
            nn.ReLU()
            # n 256 64 64
        )

        self.res_block = nn.Sequential(
            # n 256 64 64
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([256, 64, 64]),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([256, 64, 64]),
            # nn.BatchNorm2d(256),
            # n 256 64 64
        )
        self.relu = nn.ReLU()

        self.up_conv = nn.Sequential(
            # n 256 64 64
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm([128, 128, 128]),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # n 128 128 128
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm([64, 256, 256]),
            nn.BatchNorm2d(64),
            nn.ReLU()
            # n 64 256 256
        )

        self.output_conv = nn.Sequential(
            # n 64 256 256
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
            # n 3 256 256
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.down_conv(x)

        for i in range(8):
            x = self.relu(self.res_block(x) + x)  # activation after residual block

        x = self.up_conv(x)
        x = self.output_conv(x)

        return x


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.disc_module = nn.Sequential(
            # n 3 256 256
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # n 64 128 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # n 128 64 64
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # n 256 32 32
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # n 512 16 16
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            # n 1024 8 8
        )

        self.out_conv = nn.Sequential(
            # n 1024 8 8
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=8, stride=1, padding=0)
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
