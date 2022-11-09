# -*- coding: UTF-8 -*-
"""
@Function:
@File: unet_model.py
@Date: 2021/12/1 11:34 
@Author: Hever
"""

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        self.feature_map = x5
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def get_feature_map(self):
        return self.feature_map

class Decoder(nn.Module):
    def __init__(self, nfc, n_classes, bilinear=True):
        super().__init__()
        self.up1 = SingleUp(nfc, 256)
        self.up2 = SingleUp(256, 128)
        self.up3 = SingleUp(128, 64)
        self.up4 = SingleUp(64, 32)
        self.output = OutConv(32, n_classes)
    
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.output(x)
        return x

class HalfUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(HalfUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down2 = Down(128, 256 // factor)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    x = torch.randn([1, 3, 480, 270])
    model = UNet(3, 36)
    y = model(x)
    f= model.get_feature_map()
    decoder = Decoder(512, 3)

    # c = nn.Conv2d(512, 256, 3, 2, 0)
    # f = c(f)
    # print(f.size())
    # c2 = nn.Conv2d(256, 128, 3, 2)
    # print(c2(f).size())
    # f = c2(f)
    # c3 = nn.Conv2d(128, 64, 3, 2)
    # print(c3(f).size())
    # [1, 512, 30, 16]
    # ->[1, 64, 2, 1]