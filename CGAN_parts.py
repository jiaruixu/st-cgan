# sub-parts of the CGAN model
import torch
import torch.nn as nn
import torch.nn.functional as F

class inconv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(inconv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class lrl_conv_bn(nn.Module):
    '''Leaky ReLU -> Conv -> BN'''
    def __init__(self, in_ch, out_ch):
        super(lrl_conv_bn, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x

class lrl_conv_bn_triple(nn.Module):
    '''Leaky ReLU -> Conv -> BN'''
    def __init__(self, in_ch, out_ch):
        super(lrl_conv_bn_triple, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x

class lrl_conv(nn.Module):
    '''Leaky ReLU -> Conv'''
    def __init__(self, in_ch, out_ch):
        super(lrl_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class rl_convT_bn(nn.Module):
    '''ReLU -> ConvT -> BN'''
    def __init__(self, in_ch, out_ch):
        super(rl_convT_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class rl_convT_bn_triple(nn.Module):
    '''ReLU -> ConvT -> BN'''
    def __init__(self, in_ch, out_ch):
        super(rl_convT_bn_triple, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class rl_convT(nn.Module):
    '''ReLU -> ConvT'''
    def __init__(self, in_ch, out_ch):
        super(rl_convT, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x