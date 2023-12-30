#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: Parameters.py
@time: 2022/1/15 15:59
@desc:
'''

import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run our model.')
    # dataset
    parser.add_argument('--dataType', '-dt', help='eth, eos', default='eth')
    parser.add_argument('--label', '-l', help='i, p, m, e', default='i')
    parser.add_argument('--root', help='data', default='data')

    parser.add_argument('--use_node_attribute', '-use_NA', type=int, help='', default=1)
    parser.add_argument('--use_node_labeling', '-use_NL', type=int, help='node labeling', default=0)
    parser.add_argument('--use_edge_attribute', '-use_EA', type=int, help='', default=1)

    # subgraph config
    parser.add_argument('--hop', type=int, help='order of neighbor nodes', default=2)
    parser.add_argument('--topk', type=int, help='order of neighbor nodes', default=20)
    parser.add_argument('-ess', '--edge_sample_strategy', type=str, help='Volume, Times, averVolume', default='averVolume')

    parser.add_argument('--num_val', '-val', type=float, help='ratio of val', default=0.2)
    parser.add_argument('--num_test', '-test', type=float, help='ratio of test', default=0.2)
    parser.add_argument('--k_ford', '-KF', type=int, help='', default=3)


    ### transform
    parser.add_argument('--to_undirected', '-undir', type=int, help='', default=0)
    parser.add_argument('--aug', type=str, help='aug scheme: A+B', default='edgeRemove+identity')
    parser.add_argument('--aug_prob1', type=float, help='probability in data augmentation', default=0.1)

    # model setting
    parser.add_argument('--model', type=str, help='', default='gcn')
    parser.add_argument('--hidden_dim', type=int, help='', default=128)
    parser.add_argument('--num_layers', '-layer', type=int, help='', default=2)
    parser.add_argument('--pooling', type=str, help='mean, sum, max', default='max')
    parser.add_argument('--use_proj_head', '-ph', type=int, help='use project head', default=1)
    parser.add_argument('--Lambda', type=float, help='loss trade-off', default=0.01)
    parser.add_argument('--temperature', '-T', type=float, help='', default=0.2)

    # side information
    parser.add_argument('--use_node_label', '-NL', type=int, help='use node label information', default=0)

    # train
    parser.add_argument('--batch_size', '-bs', type=int, help='batch size', default=16)
    parser.add_argument('--epochs', type=int, help='', default=1000)
    parser.add_argument('--lr', type=float, help='Learning rate.', default=0.01)
    parser.add_argument('--dropout', type=float, help='dropout', default=0.)
    parser.add_argument('--gpu', type=str, help='gpu id', default='0')

    # exp setting
    parser.add_argument('--early_stop', type=int, help='', default=1)
    parser.add_argument('--early_stop_mindelta', '-min_delta', type=float, help='gpu id', default=-0.)

    parser.add_argument('--seed', type=int, help='random seed', default=12)
    parser.add_argument('--seeds', type=list, help='random seed', default=[12])
    parser.add_argument('--exp_num', type=int, help='', default=1)



    return parser.parse_args()
