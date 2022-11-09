from torch import nn
import torch
from torch.nn import Conv2d, LeakyReLU, InstanceNorm2d
class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(DiscriminatorBlock, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_classes, flo=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_classes , flo, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(flo, flo * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(flo * 2, flo * 4, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(flo * 4, 1, kernel_size=4, stride=2, padding=1)
        self.norm = InstanceNorm2d(flo * 2)
        self.leaky = nn.LeakyReLU(0.2, True)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear')
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.leaky(x)
        x = self.norm(x)
        x = self.conv3(x)
        x = self.leaky(x)
        x = self.norm(x)
        x = self.classifier(x)
        x = self.upsample(x)
        return x


# class Discriminator(nn.Module):
#     def __init__(self, num_classes, flo=64):
#         super(Discriminator, self).__init__()
#         self.block1 = DiscriminatorBlock(num_classes, flo)
#         self.block2 = DiscriminatorBlock(flo, flo*2)
#         self.block3 = DiscriminatorBlock(flo*2, flo*4)
#         self.block4 = DiscriminatorBlock(flo*4, flo*8)
#         self.classifier = nn.Conv2d(flo*8, 1, kernel_size=4, stride=2, padding=1)
    
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.classifier(x)
#         return x
# import torch.nn.functional as F
# class Discriminator(nn.Module):
#     def __init__(self, num_classes, features, flo=64, ):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(num_classes, 1024, kernel_size=3, stride=2, padding=0)
#         self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=0)
#         self.conv3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=0) 
#         self.fc1 = nn.Linear(features, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 2)
#         self.feature = features

    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = x.reshape((-1, self.feature))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
# class Discriminator(nn.Module):
#     def __init__(self, num_classes, flo=64):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(num_classes, flo, kernel_size=1, stride=1, padding=0)
#         self.leaky = nn.LeakyReLU(0.2, True)
#         self.conv2 = nn.Conv2d(flo, flo * 2, kernel_size=1, stride=1, padding=0, bias=True)
#         self.norm = InstanceNorm2d(flo * 2)
#         self.conv3 = nn.Conv2d(flo * 2, 1, kernel_size=1, stride=1, padding=0, bias=True)
    
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.leaky(x)
#         x = self.conv2(x)
#         x = self.norm(x)
#         x = self.conv3(x)
#         return x


    