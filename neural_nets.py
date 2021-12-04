import torch.nn as nn
from torchvision import models


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.down_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.res_block = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(256)
        )

        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=3,
                kernel_size=7,
                stride=1,
                padding=3
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.down_conv(x)

        for i in range(8):
            x = self.res_block(x) + x

        x = self.up_conv(x)
        x = self.output_conv(x)

        return x


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv_2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv_3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm_1 = nn.BatchNorm2d(128)

        self.conv_4 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.conv_5 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm_2 = nn.BatchNorm2d(256)

        self.conv_6 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm_3 = nn.BatchNorm2d(256)

        self.conv_7 = nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.conv_1(x))
        x = self.leaky_relu(
            self.norm_1(self.conv_3(self.leaky_relu(self.conv_2(x)))),
            negative_slope=0.2
        )
        x = self.leaky_relu(
            self.norm_2(self.conv_5(self.leaky_relu(self.conv_4(x)))),
            negative_slope=0.2
        )
        x = self.leaky_relu(
            self.norm_3(self.conv_6(x)),
            negative_slope=0.2
        )
        x = self.conv_7(x)
        x = self.sigmoid(x)

        return x


class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()

        vgg = models.vgg16(pretrained=True)

        self.feature = vgg.features[:24]

        for parameter in self.feature.parameters():
            parameter.require_grad = False


class vgg19(nn.Module):
    def __init__(self):
        super(vgg19, self).__init__()

        vgg = models.vgg16(pretrained=True)

        self.feature = vgg.features[:37]

        for parameter in self.feature.parameters():
            parameter.require_grad = False
