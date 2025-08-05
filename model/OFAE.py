import torch
import torch.nn as nn
from .ChannelAttention import ChannelAttention
from .SKFF import SKFF

class OFAE(nn.Module):
    def __init__(self,  in_nc, out_nc, connection=False):
        super(OFAE, self).__init__()
        self.connection = connection
        if connection==True:
            self.decrease_dim = nn.Conv2d(in_nc, out_nc, 1, stride=1, padding=0)

        self.high_freq = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.mid_freq = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, stride=2, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.low_freq = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, stride=4, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dcn_1 = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.dcn_2 = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.dcn_3 = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.SKFF_1 = SKFF(in_channels=out_nc, height=2, reduction=8)
        self.SKFF_2 = SKFF(in_channels=out_nc, height=2, reduction=8)
        self.se = ChannelAttention(out_nc, reduction=32)
    def forward(self, x):
        f_l = self.low_freq(x)
        x_2 = self.mid_freq(x)
        x_1 = self.high_freq(x)
        f_m = x_2 - self.up(f_l)
        f_h = x_1 - self.up(x_2)

        f_l_enc = self.dcn_3(f_l)

        f_ml_enc = self.dcn_2(self.SKFF_1([self.up(f_l_enc), f_m]))
        f_mlh_enc = self.dcn_1(self.SKFF_2([self.up(f_ml_enc), f_h]))
        f_mlh_enc = self.se(f_mlh_enc)
        if self.connection==True:
            x = self.decrease_dim(x)
        return f_mlh_enc+x