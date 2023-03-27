"""
Bk (1,V,V)
MST (Bk)
"""

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


class MST_graph(object):
    # mst
    def __init__(self, maps, n):
        self.maps = maps
        self.n = n
        self.nodenum = self.get_nodenum()
        self.edgenum = self.nodenum * self.nodenum

    def get_nodenum(self):
        return len(self.maps[0])

    def kruskal(self):
        res = np.zeros((21, 21))
        if self.nodenum <= 0 or self.edgenum < self.nodenum - 1:
            return res
        edge_list = []
        for i in range(self.nodenum):
            for j in range(i, self.nodenum):
                edge_list.append([i, j, self.maps[self.n][i][j]])  # 按[begin, end, weight]形式加入
        edge_list.sort(key=lambda a: a[2], reverse=True)  # 已经排好序的边集合

        group = [[i] for i in range(self.nodenum)]
        for edge in edge_list:
            for i in range(len(group)):
                if edge[0] in group[i]:
                    m = i
                if edge[1] in group[i]:
                    n = i
            if m != n:
                res[edge[0]][edge[1]] = edge[2]
                group[m] = group[m] + group[n]
                group[n] = []
        return res


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
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

    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        assert self.num_subset == 1
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            adaptive_A = self.PA
            adaptive_A = self.L2_norm(adaptive_A)
        else:
            adaptive_A = self.A.cuda(x.get_device())

        for i in range(self.num_subset):
            # make Bk(V,V) MST
            # print("adaptive_A", adaptive_A)

            MST_A = MST_graph(adaptive_A, 0)
            MST_A = MST_A.kruskal()
            MST_A = torch.from_numpy(MST_A)
            MST_A = MST_A.to(torch.float32)
            MST_A = MST_A.cuda(x.get_device())

            # print("MST_A", MST_A)

            # graph conv
            x_nCTv = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(x_nCTv, MST_A).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

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
