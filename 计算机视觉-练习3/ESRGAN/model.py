import math
from torch import nn
import torch
from torch.nn import functional as F
padding_mode = 'zeros'
# padding_mode = 'replicate'


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = identity + x
        return x


class _ResDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_growth=32):
        super(_ResDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_growth, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(num_feat + num_growth, num_growth, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_growth, num_growth, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_growth, num_growth, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_growth, num_feat, kernel_size=3, padding=1, padding_mode=padding_mode)

        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out1 = self.lrelu(self.conv1(x))
        out2 = self.lrelu(self.conv2(torch.cat((x, out1), dim=1)))  # tensor(n, c, w, h)
        out3 = self.lrelu(self.conv3(torch.cat((x, out1, out2), dim=1)))
        out4 = self.lrelu(self.conv4(torch.cat((x, out1, out2, out3), dim=1)))
        out5 = self.conv5(torch.cat((x, out1, out2, out3, out4), dim=1))
        return out5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, num_feat=64, num_growth=32):
        super(_RRDB, self).__init__()
        self.rdb1 = _ResDenseBlock(num_feat, num_growth)
        self.rdb2 = _ResDenseBlock(num_feat, num_growth)
        self.rdb3 = _ResDenseBlock(num_feat, num_growth)
        self.init_weight()

    def init_weight(self):
        for L in self.modules():
            if isinstance(L, nn.Conv2d):
                nn.init.kaiming_normal_(L.weight)
                L.weight.data *= 0.1
                L.bias.data.fill_(0)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.rrdb = self.make_layer(_RRDB, 23)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.upscale1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.upscale2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def init_weight(self):
        print("init ")
        #self.rrdb.init_weight()
        # for L in self.modules():
        #     if isinstance(L, nn.Conv2d):
        #         nn.init.kaiming_normal_(L.weight)
        #         L.weight.data *= 0.1
        #         L.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.rrdb(x)
        x = self.conv2(x)
        x = x + identity

        x = self.lrelu(self.upscale1(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.lrelu(self.upscale2(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.lrelu(self.conv3(x))
        return self.conv4(x)


class G2(nn.Module):
    def __init__(self):
        super(G2, self).__init__()
        self.conv_first = nn.Conv2d(3, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.body = self.make_layer(_RRDB, 23)
        self.conv_body = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv_up1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv_up2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv_hr = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv_last = nn.Conv2d(64, 3, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def init_weight(self):
        for L in self.modules():
            if isinstance(L, nn.Conv2d):
                nn.init.kaiming_normal_(L.weight)
                L.weight.data *= 0.1
                L.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv_first(x)
        identity = x
        x = self.body(x)
        x = self.conv_body(x)
        x = x + identity

        x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='nearest')))
        x = self.lrelu(self.conv_hr(x))
        return self.conv_last(x)


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        # in 3*128*128
        self.conv00 = nn.Conv2d(3, 64, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv01 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        # in 64*64*64
        self.conv10 = nn.Conv2d(64, 64*2, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False)
        self.bn10 = nn.BatchNorm2d(64*2)
        self.conv11 = nn.Conv2d(64*2, 64*2, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode, bias=False)
        self.bn11 = nn.BatchNorm2d(64*2)
        # in 128*32*32
        self.conv20 = nn.Conv2d(64*2, 64 * 4, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False)
        self.bn20 = nn.BatchNorm2d(64 * 4)
        self.conv21 = nn.Conv2d(64 * 4, 64 * 4, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode, bias=False)
        self.bn21 = nn.BatchNorm2d(64 * 4)
        # in 256*16*16
        self.conv30 = nn.Conv2d(64 * 4, 64 * 8, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False)
        self.bn30 = nn.BatchNorm2d(64 * 8)
        self.conv31 = nn.Conv2d(64 * 8, 64 * 8, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode, bias=False)
        self.bn31 = nn.BatchNorm2d(64 * 8)
        # in 512*8*8
        self.conv40 = nn.Conv2d(64 * 8, 64 * 8, kernel_size=3, padding=1, padding_mode=padding_mode, bias=False)
        self.bn40 = nn.BatchNorm2d(64 * 8)
        self.conv41 = nn.Conv2d(64 * 8, 64 * 8, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode, bias=False)
        self.bn41 = nn.BatchNorm2d(64 * 8)
        # in:4*4
        self.dense1 = nn.Linear(512*4*4, 100)
        self.dense2 = nn.Linear(100, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def init_weight(self):
        for L in self.modules():
            if isinstance(L, nn.Conv2d):
                n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
                L.weight.data.normal_(0, math.sqrt(2. / n))
                if L.bias is not None:
                    L.bias.data.zero_()

    def forward(self, x):
        x = self.lrelu(self.conv00(x))
        x = self.lrelu(self.bn0(self.conv01(x)))

        x = self.lrelu(self.bn10(self.conv10(x)))
        x = self.lrelu(self.bn11(self.conv11(x)))

        x = self.lrelu(self.bn20(self.conv20(x)))
        x = self.lrelu(self.bn21(self.conv21(x)))

        x = self.lrelu(self.bn30(self.conv30(x)))
        x = self.lrelu(self.bn31(self.conv31(x)))

        x = self.lrelu(self.bn40(self.conv40(x)))
        x = self.lrelu(self.bn41(self.conv41(x)))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.dense1(x))
        return self.dense2(x)


def test():
    low_resolution = 32  # 128x128 -> 32x32
    with torch.cuda.amp.autocast():
        x = torch.randn((5, 3, low_resolution, low_resolution))
        gen = G()
        gen.init_weight()
        gen_out = gen(x)
        disc = D()
        disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out.shape)


if __name__ == "__main__":
    test()






