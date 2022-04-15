"""
Simple Unet Structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UnetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down1 = UnetDown(self.in_channels, 64)
        self.down2 = UnetDown(64, 128)
        self.down3 = UnetDown(128, 256)
        self.down4 = UnetDown(256, 512)
        self.down5 = UnetDown(512, 512)
        self.up1 = UnetUp(512, 512)
        self.up2 = UnetUp(1024, 512)
        self.up3 = UnetUp(1024, 256)
        self.up4 = UnetUp(512, 128)
        self.up5 = UnetUp(256, 64)
        self.outc = nn.Conv2d(64, self.out_channels, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x, x)
        x = self.outc(x)

        return torch.sigmoid(x)
