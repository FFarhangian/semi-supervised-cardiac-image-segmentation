import torch.nn.init as init
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F
from torch import nn



def initialize_weights(*models):
    '''
    Kaiming weight initialization
    '''
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _ConvBlock(nn.Module):
    '''
    Convolution block for encoder
    '''
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _EncoderBlock(nn.Module):
    '''
    Encoder Block = ConvBlock + MaxPool
    '''
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [_ConvBlock(in_channels, out_channels,dropout), nn.MaxPool2d(kernel_size=2, stride=2)]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    '''
    Decoder Block = ConvBlock + ConvTranspose2D
    '''
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)



class UNet(nn.Module):
    def __init__(self, num_classes,n1):
        features = [n1, n1*2, n1*4, n1*8, n1*16]
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(1, features[0])
        self.enc2 = _EncoderBlock(features[0], features[1])
        self.enc3 = _EncoderBlock(features[1], features[2])
        self.enc4 = _EncoderBlock(features[2], features[3], dropout=True)

        self.center = _DecoderBlock(features[3], features[4], features[3])

        self.dec4 = _DecoderBlock(features[4], features[3], features[2])
        self.dec3 = _DecoderBlock(features[3], features[2], features[1])
        self.dec2 = _DecoderBlock(features[2], features[1], features[0])
        self.dec1 = nn.Sequential(
            nn.Conv2d(features[1], features[0], kernel_size=3),
            nn.BatchNorm2d(features[0]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(features[0], features[0], kernel_size=3),
            nn.BatchNorm2d(features[0]),
            nn.LeakyReLU(inplace=True),
        )
        self.final = nn.Conv2d(features[0], num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)

        return F.upsample(final, x.size()[2:], mode='bilinear')
