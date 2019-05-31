import torch
import torch.nn as nn
import torch.nn.functional as F
from CGAN_parts import *

class generator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(generator, self).__init__()
        self.inc = inconv(in_ch, 64)
        self.conv_1 = lrl_conv_bn(64, 128)
        self.conv_2 = lrl_conv_bn(128, 256)
        self.conv_3 = lrl_conv_bn(256, 512)
        self.conv_4 = lrl_conv_bn_triple(512, 512)
        self.conv_5 = lrl_conv(512, 512)
        self.conv_T6 = rl_convT_bn(512, 512)
        self.conv_T7 = rl_convT_bn_triple(1024,512)
        self.conv_T8 = rl_convT_bn(1024, 256)
        self.conv_T9 = rl_convT_bn(512, 128)
        self.conv_T10 = rl_convT_bn(256, 64)
        self.conv_T11 = rl_convT(128, out_ch)

    def forward(self, input):
        cv0 = self.inc(input)
        cv1 = self.conv_1(cv0)
        cv2 = self.conv_2(cv1)
        cv3 = self.conv_3(cv2)
        cv4 = self.conv_4(cv3)
        cv5 = self.conv_5(cv4)
        cvT6 = self.conv_T6(cv5)
        input7 = torch.cat([cvT6, cv4], dim=1)
        cvT7 = self.conv_T7(input7)
        input8 = torch.cat([cvT7, cv3], dim=1)
        cvT8 = self.conv_T8(input8)
        input9 = torch.cat([cvT8, cv2], dim=1)
        cvT9 = self.conv_T9(input9)
        input10 = torch.cat([cvT9, cv1], dim=1)
        cvT10 = self.conv_T10(input10)
        input11 = torch.cat([cvT10, cv0], dim=1)
        cvT11 = self.conv_T11(input11)
        out = torch.tanh(cvT11)
        return out


class discriminator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(discriminator, self).__init__()
        self.inc = inconv(in_ch, 64)
        self.conv_1 = lrl_conv_bn(64, 128)
        self.conv_2 = lrl_conv_bn(128, 256)
        self.conv_3 = lrl_conv_bn(256, 512)
        self.conv_4 = lrl_conv(512, out_ch)

    def forward(self, input):
        cv0 = self.inc(input)
        cv1 = self.conv_1(cv0)
        cv2 = self.conv_2(cv1)
        cv3 = self.conv_3(cv2)
        cv4 = self.conv_4(cv3)
        out = torch.sigmoid(cv4)
        return out