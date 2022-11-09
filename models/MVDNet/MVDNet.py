from .unet_component import *

class DivUpLayer(nn.Module):
    def __init__(self, in_channel, out_channel=128):
        super().__init__()
        self.up1 = Up(in_channel, 256, bilinear=True)
        self.up2 = Up(512, out_channel, bilinear=True)
        # self.up3 = Up(256, out_channel, bilinear=True)
    
    def forward(self, x5, x4, x3):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        # x = self.up3(x, x2)
        return x


class Classifier(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.div_layer = DivUpLayer(in_channel)
        # self.up1 = Up(in_channel, 256, bilinear=True)
        # self.up2 = Up(512, 128, bilinear=True)
        self.up3 = Up(256, 64, bilinear=True)
        self.up4 = Up(128, 64, bilinear=True)
        self.outc = OutConv(64, num_classes)
    def forward(self, x5, x4, x3, x2, x1):
        x = self.div_layer(x5, x4, x3)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class MVDNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MVDNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.init_conv = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.classifier1 = Classifier(1024, n_classes)
        self.classifier2 = Classifier(1024, n_classes)

    def forward(self, x):
        x1 = self.init_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        self.features = x5
        # output = self.output_layer1(x5, x4, x3, x2, x1)
        output1 = self.classifier1(x5, x4, x3, x2, x1)
        output2 = self.classifier2(x5, x4, x3, x2, x1)
        return output1, output2
    
    def get_feature_map(self):
        return self.features



