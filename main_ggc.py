#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: main.py
@time: 2022/1/15 15:51
@desc:
'''
import os
import os.path as osp
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from utils.parameters import get_parser
from utils.dataset import MyBlockChain_TUDataset
from utils.transform import *
from utils.tools import setup_seed, EarlyStopping, data_split
from utils.transform import Augmentor_Transform, MyAug_Identity

from model.encoder import *
from model.model import Ethident

# data information
label_abbreviation = {"i": "ico-wallets",
                      "m": "mining",
                      "e": "exchange",
                      "p": "phish-hack",
                      "r": "robot"}

args = get_parser()
setup_seed(args.seed)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

label = label_abbreviation[args.label]  # target account label

# path
data_path = osp.join(osp.dirname(osp.realpath(__file__)), f'{args.root}',
                     '{}/{}/{}hop-{}/{}'.format(args.dataType, label, args.hop, args.topk, args.edge_sample_strategy))


# load dataset


# load augment
print('Data Preprocessing ...')
print('Data augmentation ...')
view_1, view_2 = args.aug.split('+')
print('view 1: {}'.format(view_1))
print('view 2: {}'.format(view_2))

sample_scheme = ['Volume', 'Times', 'averVolume']

transform_raw = T.Compose([
    MyToUndirected(edge_attr_keys=['edge_attr']) if args.to_undirected else MyAug_Identity(),
    ColumnNormalizeFeatures(['edge_attr']),
    T.NormalizeFeatures()
])
dataset_raw = MyBlockChain_TUDataset(root=data_path, name=args.dataType.upper() + 'G',
                                     use_node_attr=args.use_node_attribute,
                                     use_node_importance=args.use_node_labeling,
                                     use_edge_attr=args.use_edge_attribute,  # feature selection
                                     transform=transform_raw)  # .shuffle()

if view_1 not in sample_scheme:
    transform_view_1 = T.Compose([
        Augmentor_Transform[view_1](prob=args.aug_prob1),
        MyToUndirected(edge_attr_keys=['edge_attr']) if args.to_undirected else MyAug_Identity(),
        ColumnNormalizeFeatures(['edge_attr']),
        T.NormalizeFeatures()
    ])
    dataset_v1 = MyBlockChain_TUDataset(root=data_path, name=args.dataType.upper() + 'G',
                                        use_node_attr=args.use_node_attribute,
                                        use_node_importance=args.use_node_labeling,
                                        use_edge_attr=args.use_edge_attribute,  # feature selection
                                        transform=transform_view_1)  # .shuffle()
else:
    transform_view_1 = T.Compose([
        MyToUndirected(edge_attr_keys=['edge_attr']) if args.to_undirected else MyAug_Identity(),
        ColumnNormalizeFeatures(['edge_attr']),
        T.NormalizeFeatures()
    ])
    data_path_v1 = osp.join(osp.dirname(osp.realpath(__file__)), f'{args.root}',
                            '{}/{}/{}hop-{}/{}'.format(args.dataType, label, args.hop, args.topk, view_1))
    dataset_v1 = MyBlockChain_TUDataset(root=data_path_v1, name=args.dataType.upper() + 'G',
                                        use_node_attr=args.use_node_attribute,
                                        use_node_importance=args.use_node_labeling,
                                        use_edge_attr=args.use_edge_attribute,  # feature selection
                                        transform=transform_view_1)  # .shuffle()

if view_2 not in sample_scheme:
    transform_view_2 = T.Compose([
        Augmentor_Transform[view_2](prob=args.aug_prob1),
        MyToUndirected(edge_attr_keys=['edge_attr']) if args.to_undirected else MyAug_Identity(),
        ColumnNormalizeFeatures(['edge_attr']),
        T.NormalizeFeatures()
    ])
    dataset_v2 = MyBlockChain_TUDataset(root=data_path, name=args.dataType.upper() + 'G',
                                        use_node_attr=args.use_node_attribute,
                                        use_node_importance=args.use_node_labeling,
                                        use_edge_attr=args.use_edge_attribute,  # feature selection
                                        transform=transform_view_2)  # .shuffle()
else:
    transform_view_2 = T.Compose([
        MyToUndirected(edge_attr_keys=['edge_attr']) if args.to_undirected else MyAug_Identity(),
        ColumnNormalizeFeatures(['edge_attr']),
        T.NormalizeFeatures()
    ])
    data_path_v2 = osp.join(osp.dirname(osp.realpath(__file__)), f'{args.root}',
                            '{}/{}/{}hop-{}/{}'.format(args.dataType, label, args.hop, args.topk, view_2))

    dataset_v2 = MyBlockChain_TUDataset(root=data_path_v2, name=args.dataType.upper() + 'G',
                                        use_node_attr=args.use_node_attribute,
                                        use_node_importance=args.use_node_labeling,
                                        use_edge_attr=args.use_edge_attribute,  # feature selection
                                        transform=transform_view_2)  # .shuffle()

print('################# dataset information #########################')
print('Type of target accounts:    {}'.format(label))
print('Num. of graphs:    {}'.format(len(dataset_raw)))
print('Ave. num. of nodes:    {}'.format(np.mean([g.x.size(0) for g in dataset_raw])))
print('Ave. num. of edges:    {}'.format(np.mean([g.edge_index.size(1) for g in dataset_raw])))
print('Num. of node features:    {}'.format(dataset_raw.num_node_features))
print('Num. of edge features:    {}'.format(dataset_raw.num_edge_features))
print('Use node attribute:    {}'.format(bool(args.use_node_attribute)))
print('Use node labeling:    {}'.format(bool(args.use_node_labeling)))
print('Use edge attribute:    {}'.format(bool(args.use_edge_attribute)))

train_splits, val_splits, test_splits = data_split(X=np.arange(len(dataset_raw)),
                                                   Y=np.array([dataset_raw[i].y.item() for i in range(len(dataset_raw))]),  # 等官方修复bug
                                                   # Y=np.array([dataset[i].y[0].item() for i in range(len(dataset))]),
                                                   seeds=args.seeds[:args.exp_num], K=args.k_ford)


def train(train_loader):
    model.train()
    total_loss = 0
    total_loss_su = 0
    total_loss_un = 0

    for data in train_loader:
        data_v1, data_v2, data_raw = data
        data_v1.to(device)
        data_v2.to(device)
        data_raw.to(device)
        optimizer.zero_grad()
        node_reps_v1, graph_reps_v1, pred_out_v1 = model(data_v1.x, data_v1.edge_index, data_v1.batch, data_v1.edge_attr)
        node_reps_v2, graph_reps_v2, pred_out_v2 = model(data_v2.x, data_v2.edge_index, data_v2.batch, data_v2.edge_attr)
        node_reps, graph_reps, pred_out = model(data_raw.x, data_raw.edge_index, data_raw.batch, data_raw.edge_attr)

        loss, loss_un, loss_su = model.loss_cal(x=graph_reps_v1, x_aug=graph_reps_v2, pred_out=pred_out, target=data_raw.y, Lambda=args.Lambda)

        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data_raw.num_graphs
        total_loss_su += float(loss_su) * data_raw.num_graphs
        total_loss_un += float(loss_un) * data_raw.num_graphs

    return total_loss / len(train_loader.dataset), total_loss_su / len(train_loader.dataset), total_loss_un / len(train_loader.dataset),


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct, total_loss = 0, 0
    y_pred_label_list = []
    y_true_label_list = []

    for data in loader:
        data = data.to(device)
        node_reps, graph_reps, pred_out = model(data.x, data.edge_index, data.batch, data.edge_attr)

        loss = model.loss_su(pred_out, data.y)
        total_loss += float(loss) * data.num_graphs

        pred_out = F.softmax(pred_out, dim=1)
        pred = torch.argmax(pred_out, dim=1)

        y_pred_label_list.append(pred)
        y_true_label_list.append(data.y)

    y_pred = torch.cat(y_pred_label_list).cpu().detach().numpy()
    y_true = torch.cat(y_true_label_list).cpu().detach().numpy()

    acc = f1_score(y_true=y_true, y_pred=y_pred, average='micro')

    return acc, total_loss / len(loader.dataset)


# exp
f1_list = []

train_dataset_v1 = dataset_v1[train_splits[0]]
train_dataset_v2 = dataset_v2[train_splits[0]]
train_dataset_raw = dataset_raw[train_splits[0]]
val_dataset = dataset_raw[val_splits[0]]
test_dataset = dataset_raw[test_splits[0]]

print('Loading training data ... ')
train_loader = DataLoader(list(zip(train_dataset_v1, train_dataset_v2, train_dataset_raw)), batch_size=args.batch_size, shuffle=True,
                          num_workers=1)
print('Loading val data ... ')
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
print('Loading test data ... ')
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

encoder = HGATE_encoder(dataset_raw.num_features, args.hidden_dim, out_channels=dataset_raw.num_classes,
                        edge_dim=dataset_raw.num_edge_features, num_layers=args.num_layers, pooling=args.pooling, dropout=args.dropout,
                        add_self_loops=True, use_edge_atten=False).to(device)

model = Ethident(in_channels=dataset_raw.num_features, hidden_channels=args.hidden_dim, out_channels=dataset_raw.num_classes, encoder=encoder,
                 num_layers=args.num_layers, pooling=args.pooling, dropout=args.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
early_stopping = EarlyStopping(patience=20, min_delta=args.early_stop_mindelta)

for epoch in range(1, args.epochs):
    loss, loss_su, loss_un = train(train_loader)
    _train_loader = DataLoader(train_dataset_raw, batch_size=args.batch_size, shuffle=True)
    train_acc, _ = test(_train_loader)
    val_acc, val_loss = test(val_loader)
    test_acc, _ = test(test_loader)

    if args.early_stop:
        early_stopping(val_loss, results=[epoch, loss, loss_su, loss_un, val_loss, train_acc, val_acc, test_acc])
        if early_stopping.early_stop:
            print('\n=====final results=====')
            _epoch, _loss, _loss_su, _loss_un, _val_loss, _train_acc, _val_acc, _test_acc = early_stopping.best_results
            f1_list.append(_test_acc)
            print(f'Exp: {1},  Epoch: {_epoch:03d},       '
                  f'Train_Loss: {_loss:.4f}, Train_Loss_su: {_loss_su:.4f}, Train_Loss_un: {_loss_un:.4f},       '
                  f'Val_Loss: {_val_loss:.4f},        ,        '
                  f'Train_Acc: {_train_acc:.4f}, Val_Acc: {_val_acc:.4f},        '
                  f'Test_Acc: {_test_acc:.4f}\n\n')
            break
    else:
        f1_list.append(test_acc)

    print(f'Exp: {1},  Epoch: {epoch:03d},       '
          f'Train_Loss: {loss:.4f}, train_Loss_su: {loss_su:.4f}, train_Loss_un: {loss_un:.4f},       '
          f'Val_Loss: {val_loss:.4f},        '
          f'Train_Acc: {train_acc:.4f}, Val_Acc: {val_acc:.4f},        '
          f'Test_Acc: {test_acc:.4f}')

print('Num. of experiments: {}\n'.format(len(train_splits)),
      'Result in terms of f1-score: {} ~ {}\n\n\n'.format(np.mean(f1_list), np.std(f1_list)))
