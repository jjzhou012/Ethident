#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: model.py
@time: 2022/1/21 17:27
@desc:
'''
import torch
from torch.nn import Linear, ReLU, Sequential, BatchNorm1d, Dropout


class Project_Head(torch.nn.Module):
    # Todo: add num_layers
    def __init__(self, in_channels):
        super(Project_Head, self).__init__()

        self.block = Sequential(Linear(in_channels, in_channels),
                                BatchNorm1d(in_channels), ReLU(inplace=True),
                                Linear(in_channels, in_channels),
                                BatchNorm1d(in_channels), ReLU(inplace=True),
                                )

        self.linear_shortcut = Linear(in_channels, in_channels)


    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)




class Ethident(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels, out_channels, num_layers,
                 encoder, use_proj_head=True, proj_head_share=True,
                 pooling='max', temperature=0.2, dropout=None):
        super(Ethident, self).__init__()

        self.encoder = encoder
        self.use_proj_head = use_proj_head
        self.proj_head_share = proj_head_share
        self.pooling = pooling
        self.temperature = temperature

        # self.embedding_dim = self.encoder.core.hidden_channels
        self.embedding_dim = self.encoder.hidden_channels

        print('emb dim: ', self.embedding_dim)


        self.proj_head_g1 = Project_Head(in_channels=self.embedding_dim)
        self.proj_head_g2 = Project_Head(in_channels=self.embedding_dim)

        self.fc = Sequential(Linear(self.embedding_dim, self.embedding_dim),
                             ReLU(), Dropout(p=dropout, inplace=True),
                             Linear(self.embedding_dim, out_channels),
                             )

        self.init_emb()


    def init_emb(self):
        # initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, edge_attr, device=None):

        node_reps, graph_reps = self.encoder(x, edge_index, batch, edge_attr)

        pred_out = self.fc(graph_reps)

        if self.use_proj_head:
            graph_reps = self.proj_head_g1(graph_reps)

        return node_reps, graph_reps, pred_out

    def loss_un(self, x, x_aug):

        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / self.temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def loss_su(self, pred_out, target):

        loss = torch.nn.CrossEntropyLoss()
        return loss(pred_out, target)

    def loss_cal(self, x, x_aug, pred_out, target, Lambda):
        loss_un = self.loss_un(x, x_aug)
        loss_su = self.loss_su(pred_out, target)
        return loss_su + Lambda * loss_un, loss_un, loss_su


