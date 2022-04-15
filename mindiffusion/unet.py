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
        )
        self.conv = nn.Sequential(
            nn.Conv2d(oc, oc, 3, 1, 1),
            nn.BatchNorm2d(oc),
            nn.ReLU(),
            nn.Conv2d(oc, oc, 3, 1, 1),
            nn.BatchNorm2d(oc),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.main(x)
        return self.conv(x) + x


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
        N = 128
        self.down1 = UnetDown(self.in_channels, N)
        self.down2 = UnetDown(N, 2 * N)
        self.down3 = UnetDown(2 * N, 2 * N)

        self.throu = Conv3(2 * N, 2 * N)

        self.up1 = UnetUp(4 * N, 2 * N)
        self.up2 = UnetUp(4 * N, N)
        self.up3 = UnetUp(2 * N, N)
        self.out = nn.Conv2d(N, self.out_channels, 1)

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
