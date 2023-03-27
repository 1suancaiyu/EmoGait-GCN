import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class frame_attention(nn.Module):
    def __init__(self, in_channels, redu=16):
        super(frame_attention, self).__init__()
        self.spatial_pooling = nn.AdaptiveAvgPool2d((None,1))
        self.relu = nn.ReLU()
        self.conv_frame = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=9, padding=4)
        self.sigmoid = nn.Sigmoid()
        print(">>>>>>  frame_attention")

    def forward(self, x):
        x_origin = x
        x = self.spatial_pooling(x)
        x = x.squeeze(-1)
        x = self.conv_frame(x)
        x = x.unsqueeze(-1)
        x = self.relu(x)
        x_frame_score = self.sigmoid(x)
        x = x_origin * x_frame_score + x_origin
        return x

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)
        self.fa = frame_attention(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.fa(x)
        return x

class SCE(nn.Module):
    def __init__(self, in_channels, num_jpts=25, redu = 16):
        super(SCE, self).__init__()
        ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
        pad = (ker_jpt - 1) // 2
        redu_channels = in_channels // redu
        self.temporal_pooling = nn.AdaptiveAvgPool2d((1,None))
        self.conv_spatial = nn.Conv1d(in_channels, redu_channels, ker_jpt, padding=pad)
        self.relu = nn.ReLU()
        self.conv_expand = nn.Conv1d(redu_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        print(">>>>>>>>>>>>>>>>>>  SCE")
    def forward(self, x):
        x_origin = x
        x = self.temporal_pooling(x)
        x = x.squeeze(-2)
        x = self.conv_spatial(x)
        x = self.relu(x)
        x = self.conv_expand(x)
        x = x.unsqueeze(-2)
        sce_weights = self.sigmoid(x)
        x = x_origin * sce_weights + x_origin
        return x



class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

        self.sce = SCE(in_channels=out_channels, num_jpts=A.shape[1])

    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):

            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        y = self.sce(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, temporal_receptive_field=1, residual=True, adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels,  A, adaptive=adaptive)
        self.tcn1 = unit_tcn(out_channels, out_channels, kernel_size=temporal_receptive_field, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Classifier(nn.Module):

    def __init__(self, in_channels, num_classes, num_point=21, num_set=1, temporal_kernel_size=75):
        super().__init__()

        # make adaptive graph *21*21
        adaptive_A = np.stack([np.eye(int(num_point))] * num_set, axis=0)

        # build networks
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)
        self.wsx_gcn_networks = nn.ModuleList((
            TCN_GCN_unit(in_channels, out_channels=32, A=adaptive_A, stride=1, temporal_receptive_field=temporal_kernel_size, residual=True, adaptive=True),
            TCN_GCN_unit(in_channels=32, out_channels=64, A=adaptive_A, stride=1, temporal_receptive_field=temporal_kernel_size, residual=True, adaptive=True),
            TCN_GCN_unit(in_channels=64, out_channels=64, A=adaptive_A, stride=1, temporal_receptive_field=temporal_kernel_size, residual=True, adaptive=True),
        ))

        # fcn for prediction
        self.fcn = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, T, V, C, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V) # n,c,t,v

        # forward
        for gcn in self.wsx_gcn_networks:
            x = gcn(x)
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        f = x.squeeze()

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        # x = x.view(x.size(0), -1)

        return x, f