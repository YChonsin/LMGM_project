
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils import *

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            # nn.MaxPool3d(2, stride=2, padding=0),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2, padding=0)
        self.conv = DoubleConv(in_channels, out_channels)
        #
        # self.in_channels = in_channels
        # self.out_channels = out_channels

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # print(x1.shape, x2.shape)
        # print(self.in_channels, self.out_channels)
        x1 = self.up(x1)
        # print(x1.shape)

        x = torch.cat([x2, x1], dim=1)
        # print(x2.shape, x1.shape)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv3d(in_channels, num_classes, kernel_size=1)
        )


class UNet_3d(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 1,
                 bilinear: bool = False,
                 base_c: int = 64):
        super(UNet_3d, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        # self.down3 = Down(base_c * 4, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 8)
        self.up1 = Up(base_c * 16, base_c * 4)
        self.up2 = Up(base_c * 8, base_c * 2)
        self.up3 = Up(base_c * 4, base_c)
        self.up4 = Up(base_c * 2, base_c)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor):
        # print(x.shape)
        x1 = self.in_conv(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        # x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits

def ANN():
    model = UNet_3d(base_c=96)
    return model

if __name__ == "__main__":

    device = torch.device('cuda:0')

    model = ANN().to(device)

    model = nn.DataParallel(model, device_ids=[0]).to(device)

    x = torch.randn(1, 1, 32, 32, 32).to(device)
    print(x.shape)
    pred = model(x)
    print(x.shape)
