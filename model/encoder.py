#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: blockgcn.py
@time: 2022/1/18 0:19
@desc:
'''

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter, GRUCell, ReLU, Sequential, BatchNorm1d, Dropout, LogSoftmax
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool, global_mean_pool, global_max_pool

from model.layer import GATEConv

pooling_dict = {'sum': global_add_pool,
                'mean': global_mean_pool,
                'max': global_max_pool}


class HGATE_encoder(torch.nn.Module):
    r"""

    Args:
        in_channels (int):          Size of each input sample.
        hidden_channels (int):      Hidden node feature dimensionality.
        out_channels (int):         Size of each output sample.
        edge_dim (int):             Edge feature dimensionality.
        num_layers (int):           Number of GNN layers.
        dropout (float, optional):  Dropout probability. (default: :obj:`0.0`)

    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int,
                 edge_dim: int = None, dropout: float = 0.0, pooling='sum', heads=4, add_self_loops=True, use_edge_atten=True):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        self.hidden_channels = hidden_channels

        self.use_edge_atten = use_edge_atten

        self.lin1 = Linear(in_channels, hidden_channels)

        self.account_convs = torch.nn.ModuleList()

        for i in range(num_layers):
            if i == 1:
                conv = GATConv(hidden_channels, hidden_channels, edge_dim=edge_dim if use_edge_atten else None,
                               dropout=dropout, add_self_loops=add_self_loops, negative_slope=0.01, heads=heads)
            elif i > 1:
                conv = GATConv(heads * hidden_channels, hidden_channels, edge_dim=edge_dim if use_edge_atten else None,
                               dropout=dropout, add_self_loops=add_self_loops, negative_slope=0.01, heads=heads)
            else:
                conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)

            self.account_convs.append(conv)

        self.subgraph_conv = GATConv(heads * hidden_channels, hidden_channels, dropout=dropout,
                                     add_self_loops=add_self_loops, negative_slope=0.01)

        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv in self.account_convs:
            conv.reset_parameters()
        self.subgraph_conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch, edge_attr=None):
        """"""
        # Account Embedding:
        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.account_convs[0](x, edge_index, edge_attr))  # []
        x = F.dropout(h, p=self.dropout, training=self.training)

        for conv in self.account_convs[1:]:
            h = F.elu_(conv(x, edge_index, edge_attr=edge_attr if self.use_edge_atten else None))
            x = F.dropout(h, p=self.dropout, training=self.training)

        # Subgraph Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)
        out = pooling_dict[self.pooling](x, batch).relu_()

        z = self.subgraph_conv((x, out), edge_index)

        return x, z

