"""
Code for "i-RevNet: Deep Invertible Networks"
https://openreview.net/pdf?id=HJsjkMb0Z
ICLR, 2018
(c) Joern-Henrik Jacobsen, 2018
"""

import torch
import torch.nn as nn

def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n, :].contiguous()
    x2 = x[:, n:, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]


class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def inverse(self, input):
        bl = self.block_size
        bs, new_d, w = input.shape[0], input.shape[1] // bl, input.shape[2]
        return input.reshape(bs, bl, new_d, w).permute(0, 2, 3, 1).reshape(bs, new_d, w * bl)

    def forward(self, input):
        bl = self.block_size
        bs, d, new_w = input.shape[0], input.shape[1], input.shape[2] // bl
        return input.reshape(bs, d, new_w, bl).permute(0, 2, 1, 3).reshape(bs, d * bl, new_w)

class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.,
                 affineBN=True, mult=4):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first
        self.pad = 2 * out_ch - in_ch
        self.stride = stride
        self.inj_pad = injective_pad(self.pad)
        self.psi = psi(stride)
        if self.pad != 0 and stride == 1:
            in_ch = out_ch * 2
            print('')
            print('| Injective iRevNet |')
            print('')
        layers = []
        if not first:
            layers.append(nn.BatchNorm1d(in_ch//2, affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv1d(in_ch//2, int(out_ch//mult), kernel_size=11,
                      stride=stride, padding=5, bias=False))
        layers.append(nn.BatchNorm1d(int(out_ch//mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv1d(int(out_ch//mult), int(out_ch//mult),
                      kernel_size=11, padding=5, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.BatchNorm1d(int(out_ch//mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv1d(int(out_ch//mult), out_ch, kernel_size=11,
                      padding=5, bias=False))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x, x_mask, g=None, reverse=False):
        """ bijective or injective block forward """
        if reverse:
            return self.inverse(x, x_mask, g)
        if self.pad != 0 and self.stride == 1:
            x = merge(x[0], x[1])
            x = self.inj_pad.forward(x)
            x1, x2 = split(x)
            x = (x1, x2)
        x1, x2 = split(x)
        Fx2 = self.bottleneck_block(x2)
        if self.stride >= 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        #print(x1.shape, Fx2.shape)
        y1 = Fx2 + x1
        return merge(x2, y1) * x_mask, 0.

    def inverse(self, x, x_mask, g=None):
        """ bijective or injecitve block inverse """
        x2, y1 = split(x)
        if self.stride >= 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride >= 2:
            x1 = self.psi.inverse(x1)
        if self.pad != 0 and self.stride == 1:
            x = merge(x1, x2)
            x = self.inj_pad.inverse(x)
            x1, x2 = split(x)
            x = (x1, x2)
        else:
            x = merge(x1, x2)
        return x * x_mask, 0.


if __name__ == '__main__':

    bs = 8
    in_chn = 32
    out_chn = 16
    w = 128
    h = 128

    x = torch.rand(bs, in_chn, w)

    print(x.shape)

    block = irevnet_block(in_chn, out_chn, stride=1, mult=4)

    y = block(x)

    print(y.shape)

    x0 = block.inverse(y)

    print(x0.shape)

    print(x.mean(), y.mean(), x0.mean())
    print((x - x0).mean())