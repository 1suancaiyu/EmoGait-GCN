import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import logging
import traceback

filename = "lly.log".format(__file__)
fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"

logging.basicConfig(
    level=logging.ERROR,
    filename=filename,
    filemode="w",
    format=fmt
)


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


class Graph(object):
    # mst
    def __init__(self, maps, n):
        self.maps = maps
        self.n = n
        self.nodenum = self.get_nodenum()
        self.edgenum = 21 * 21

    def get_nodenum(self):
        return len(self.maps[1])

    # def get_edgenum(self):
    #     count = 0
    #     for i in range(self.nodenum):
    #         for j in range(i):
    #             if self.maps[i][j] > 0 and self.maps[i][j] < 9999:
    #                 count += 1
    #     return count

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

    # def prim(self):
    #     res = np.zeros((21, 21))
    #     if self.nodenum <= 0 or self.edgenum < self.nodenum - 1:
    #         return res
    #     res = []
    #     seleted_node = [0]
    #     candidate_node = [i for i in range(1, self.nodenum)]
    #
    #     while len(candidate_node) > 0:
    #         begin, end, minweight = 0, 0, 9999
    #         for i in seleted_node:
    #             for j in candidate_node:
    #                 if self.maps[i][j] < minweight:
    #                     minweight = self.maps[i][j]
    #                     begin = i
    #                     end = j
    #         res.append([begin, end, minweight])
    #         seleted_node.append(end)
    #         candidate_node.remove(end)
    #     return res


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, coff_embedding=4):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = A.shape[0]

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())

        graph0 = Graph(self.PA, 0)
        graph1 = Graph(self.PA, 1)
        graph2 = Graph(self.PA, 2)

        mst0 = graph0.kruskal()
        mst0 = torch.from_numpy(mst0)
        mst0 = mst0.cuda(x.get_device())

        mst1 = graph1.kruskal()
        mst1 = torch.from_numpy(mst1)
        mst1 = mst1.cuda(x.get_device())

        mst2 = graph2.kruskal()
        mst2 = torch.from_numpy(mst2)
        mst2 = mst2.cuda(x.get_device())

        A[0] = A[0].cuda() + mst0
        A[1] = A[1].cuda() + mst1
        A[2] = A[2].cuda() + mst2

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)



class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, temporal_receptive_field=1, residual=True,
                 adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
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

    def __init__(self, in_channels, num_classes, num_point=21, num_set=3, temporal_kernel_size=75):
        super().__init__()

        # make adaptive graph 3*21*21
        adaptive_A = np.stack([np.eye(int(num_point))] * num_set, axis=0)

        # build networks
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)
        self.wsx_gcn_networks = nn.ModuleList((
            TCN_GCN_unit(in_channels, out_channels=32, A=adaptive_A, stride=1,
                         temporal_receptive_field=temporal_kernel_size, residual=True, adaptive=True),
            TCN_GCN_unit(in_channels=32, out_channels=64, A=adaptive_A, stride=1,
                         temporal_receptive_field=temporal_kernel_size, residual=True, adaptive=True),
            TCN_GCN_unit(in_channels=64, out_channels=64, A=adaptive_A, stride=1,
                         temporal_receptive_field=temporal_kernel_size, residual=True, adaptive=True),
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
        x = x.view(N * M, C, T, V)  # n,c,t,v

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

    # AA = np.stack([np.eye(3)] * 3, axis=0)
    # graph = Graph(AA)
    # print(graph.kruskal())
    # print(AA[1].shape)


def kru(A):
    res = np.zeros((6, 6))
    edge_list = []
    for i in range(6):
        for j in range(i, 6):
            edge_list.append([i, j, A[i][j]])  # 按[begin, end, weight]形式加入
    edge_list.sort(key=lambda a: a[2], reverse=True)  # 已经排好序的边集合

    group = [[i] for i in range(6)]
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


# testa = [[0, 7, 0, 0, 0, 5],
#          [7, 0, 9, 0, 3, 0],
#          [0, 9, 0, 6, 0, 0],
#          [0, 0, 6, 0, 8, 10],
#          [0, 3, 0, 8, 0, 4],
#          [5, 0, 0, 10, 4, 0]]
# print(kru(testa))

