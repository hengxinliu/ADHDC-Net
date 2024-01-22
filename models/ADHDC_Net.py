import torch
import torch.nn as nn
from torch.nn import functional as F
from .sync_batchnorm import SynchronizedBatchNorm3d



class Conv_1x1x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x1x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_3x3x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x

class DConv_3x3x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation, d=1):
        super(DConv_3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 1), stride=1, padding=(d, d, 0), dilation=d, bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_1x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_3x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class hd(nn.Module):
    def __init__(self, num_in, num_out):
        super(hd, self).__init__()
        self.conv = nn.Conv3d(num_in, num_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm = SynchronizedBatchNorm3d(num_out)
        self.conv1x1 = nn.Conv3d(num_in, num_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x, y):
        out = F.relu(self.norm(self.conv(y)))
        y = F.relu(self.conv1x1(y))
        y = out + y
        y = self.sigmoid(y)
        s = x * y
        l = s + x
        return l


class Conv_down(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_down, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1), bias=True)
        self.norm = SynchronizedBatchNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class HDC_module(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(HDC_module, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = in_dim // 4
        self.out_inter_dim = out_dim // 4
        self.conv_3x3x1_1 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_3x3x1_2 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_3x3x1_3 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_1x1x1_1 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(out_dim, out_dim, activation)
        if self.in_dim > self.out_dim:
            self.conv_1x1x1_3 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x3x3 = Conv_1x3x3(out_dim, out_dim, activation)

    def forward(self, x):
        x_1 = self.conv_1x1x1_1(x)
        x1 = x_1[:, 0:self.out_inter_dim, ...]
        x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]
        x2 = self.conv_3x3x1_1(x2)
        x3 = self.conv_3x3x1_2(x2 + x3)
        x4 = self.conv_3x3x1_3(x3 + x4)
        x_1 = torch.cat((x1, x2, x3, x4), dim=1)
        x_1 = self.conv_1x1x1_2(x_1)
        if self.in_dim > self.out_dim:
            x = self.conv_1x1x1_3(x)
        x_1 = self.conv_1x3x3(x + x_1)
        return x_1

class DHDC_module(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(DHDC_module, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = in_dim // 4
        self.out_inter_dim = out_dim // 4
        self.conv_3x3x1_1 = DConv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation, d=1)
        self.conv_3x3x1_2 = DConv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation, d=2)
        self.conv_3x3x1_3 = DConv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation, d=3)
        self.conv_1x1x1_1 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(out_dim, out_dim, activation)
        if self.in_dim > self.out_dim:
            self.conv_1x1x1_3 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x3x3 = Conv_1x3x3(out_dim, out_dim, activation)

    def forward(self, x):
        x_1 = self.conv_1x1x1_1(x)
        x1 = x_1[:, 0:self.out_inter_dim, ...]
        x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]
        s2 = self.conv_3x3x1_1(x2)
        z2 = x2 + s2
        s3 = self.conv_3x3x1_2(s2 + x3)
        z3 = x3 + s3
        s4 = self.conv_3x3x1_3(s3 + x4)
        z4 = x4 + s4
        x_1 = torch.cat((x1, z2, z3, z4), dim=1)
        x_1 = self.conv_1x1x1_2(x_1)
        if self.in_dim > self.out_dim:
            x = self.conv_1x1x1_3(x)
        x_1 = self.conv_1x3x3(x + x_1)
        return x_1

def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        SynchronizedBatchNorm3d(out_dim),
        activation)

device1 = torch.device("cuda")


def hdc(image, num=2):
    x1 = torch.Tensor([]).to(device1)
    for i in range(num):
        for j in range(num):
            for k in range(num):
                x3 = image[:, :, k::num, i::num, j::num]
                x1 = torch.cat((x1, x3), dim=1)
    return x1


class HDC_Net_D_A(nn.Module):
    def __init__(self, in_dim=4, out_dim=4, num_filters=32):
        super(HDC_Net_D_A, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_f = num_filters
        self.activation = nn.ReLU(inplace=False)
        # down
        self.conv_3x3x3 = Conv_3x3x3(self.n_f, self.n_f, self.activation)
        self.conv_1 = HDC_module(self.n_f, self.n_f, self.activation)
        self.down_1 = Conv_down(self.n_f, self.n_f, self.activation)
        self.conv_2 = HDC_module(self.n_f, self.n_f, self.activation)
        self.down_2 = Conv_down(self.n_f, self.n_f, self.activation)
        self.conv_3 = DHDC_module(self.n_f, self.n_f, self.activation)
        self.down_3 = Conv_down(self.n_f, self.n_f, self.activation)
        # bridge
        self.bridge = DHDC_module(self.n_f, self.n_f, self.activation)
        # up
        self.up_1 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_4 = DHDC_module(self.n_f * 2, self.n_f, self.activation)
        self.up_2 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_5 = HDC_module(self.n_f * 2, self.n_f, self.activation)
        self.up_3 = conv_trans_block_3d(self.n_f, self.n_f, self.activation)
        self.conv_6 = HDC_module(self.n_f * 2, self.n_f, self.activation)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.out = nn.Conv3d(self.n_f, out_dim, kernel_size=1, stride=1, padding=0)
        self.l1 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)
        self.l2 = nn.Conv3d(1, 1, kernel_size=1, stride=1, padding=0)
        self.ln = nn.Conv3d(4, 4, kernel_size=1, stride=1, padding=0)

        self.hd1 = hd(1, 1)
        self.hd2 = hd(2, 1)
        self.softmax = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = hdc(x)
        x = self.conv_3x3x3(x)
        x1 = self.conv_1(x)
        x = self.down_1(x1)
        x2 = self.conv_2(x)
        x = self.down_2(x2)
        x3 = self.conv_3(x)
        x = self.down_3(x3)
        x = self.bridge(x)
        x = self.up_1(x)
        x = torch.cat((x, x3), dim=1)
        x = self.conv_4(x)
        x = self.up_2(x)
        x = torch.cat((x, x2), dim=1)
        x = self.conv_5(x)
        x = self.up_3(x)
        x = torch.cat((x, x1), dim=1)
        x = self.conv_6(x)
        x = self.upsample(x)
        x = self.out(x)

        s0, s1, s2, s3 = x.chunk(4, dim=1)
        l1 = self.l1(s1)
        l2 = self.hd1(s2,l1)
        l22 = torch.cat([l1, l2], dim=1)
        l3 = self.hd2(s3,l22)
        y = torch.cat([s0, l1, l2, l3], dim=1)
        y = self.ln(y)
        y = self.softmax(y)

        return y


