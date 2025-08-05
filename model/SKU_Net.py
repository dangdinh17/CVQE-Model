import torch
import torch.nn as nn
from .ConBlock import ConBlock
from .SKFF import SKFF

class SKU_Net(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.nf = nf
        base_ks = 3
        self.Down0_0 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv0_0 = ConBlock(nf, nf)

        self.Down0_1 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv0_1 = ConBlock(nf, nf)

        self.Down0_2 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv0_2 = ConBlock(nf, nf)

        self.Up1 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.SKFF_1 = SKFF(in_channels=nf, height=2, reduction=8)
        self.Up2 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.SKFF_2 = SKFF(in_channels=nf, height=2, reduction=8)
        self.Up3 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    def forward(self, input):
        x0_0 = self.conv0_0(self.Down0_0(input))
        x0_1 = self.conv0_1(self.Down0_1(x0_0))

        x0_2 = self.conv0_2(self.Down0_2(x0_1))
        up0_1 = self.Up1(x0_2)

        b,n,h,w = x0_1.shape
        up0_1 = up0_1[:,:,:h,:w]

        up0_2 = self.Up2(self.SKFF_1([up0_1, x0_1]))

        up0_3 = self.Up3(self.SKFF_1([up0_2, x0_0]))
        return up0_3+input