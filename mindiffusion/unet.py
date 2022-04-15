"""
Simple Unet Structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(ic, oc, 3, 1, 1),
            nn.BatchNorm2d(oc),
            nn.ReLU(),
            nn.Conv2d(oc, oc, 3, 1, 1),
            nn.BatchNorm2d(oc),
            nn.ReLU(),
            nn.Conv2d(oc, oc, 3, 1, 1),
            nn.BatchNorm2d(oc),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)


class UnetDown(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetDown, self).__init__()
        layers = [Conv3(in_size, out_size), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):

        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 2, 2),
            Conv3(out_size, out_size),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)

        return x


class NaiveUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down1 = UnetDown(self.in_channels, 64)
        self.down2 = UnetDown(64, 128)
        self.down3 = UnetDown(128, 128)

        self.throu = Conv3(128, 128)

        self.up1 = UnetUp(256, 128)
        self.up2 = UnetUp(256, 64)
        self.up3 = UnetUp(128, 32)
        self.out = nn.Conv2d(32, self.out_channels, 1)

    def forward(self, x, t):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        thro = self.throu(down3)

        up1 = self.up1(thro, down3)
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)

        out = self.out(up3)

        return out
