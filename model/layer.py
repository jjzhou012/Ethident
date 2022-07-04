#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: layer.py
@time: 2022/1/22 17:30
@desc:
'''
from typing import Optional
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, Parameter, GRUCell, ReLU, Sequential, BatchNorm1d, Dropout, LogSoftmax

from torch_geometric.utils import softmax
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.inits import glorot, zeros



class GATEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, dropout: float = 0.0, heads=4):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_neighbor = Parameter(torch.Tensor(1, out_channels))  # attention of neighbors
        self.att_self = Parameter(torch.Tensor(1, in_channels))  # attention of target node

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_neighbor)
        glorot(self.att_self)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out += self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # mapping [feature_neighbors || feature_edge] to [feature_neighbors]
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        # attentive coef
        alpha_j = (x_j * self.att_neighbor).sum(dim=-1)  # attention coef of neighbors
        alpha_i = (x_i * self.att_self).sum(dim=-1)  # attention coef of target node itself
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        # normalize
        alpha = softmax(alpha, index, ptr, size_i)
        #
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)