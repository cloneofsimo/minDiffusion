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
        return self.conv(x)


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


class TimeSiren(nn.Module):
    def __init__(self, emb_dim):
        super(TimeSiren, self).__init__()

        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class NaiveUnet(nn.Module):
    def __init__(self, in_channels, out_channels, n_feat=256):
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feat = n_feat

        self.down1 = UnetDown(self.in_channels, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.BatchNorm2d(2 * n_feat), nn.ReLU()
        )

        self.timeembed = TimeSiren(2 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4),
            nn.BatchNorm2d(2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, 3)

    def forward(self, x, t):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        thro = self.to_vec(down3)
        temb = self.timeembed(t).view(-1, self.n_feat * 2, 1, 1)

        thro = self.up0(thro + temb)

        up1 = self.up1(thro, down3)
        up2 = self.up2(up1, down2)
        out = self.up3(up2, down1)

        return out
