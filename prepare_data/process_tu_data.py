#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: process_tu_data.py
@time: 2021/12/28 9:55
@desc:
'''
import numpy as np
import networkx as nx
import scipy.sparse as sp
import os




def to_tu_file(G_list: list, X_list: list, gy_list: list, path: str, ny_list=None, node_importance=None, dataname=None, target_node2label=None):
    '''
    save link-subgraphs to tu_file
    :param G_list:
    :param X_list:
    :param gy_list:
    :param DS:
    :param ny_list:
    :return:
    '''
    save_path = '{}/raw/'.format(path)
    if not os.path.exists(save_path): os.makedirs(save_path)
    DS = dataname.upper() + 'G'

    A1_list = [nx.adj_matrix(g, weight="volume") for g in G_list]
    A2_list = [nx.adj_matrix(g, weight="times") for g in G_list]
    A1_block = sp.block_diag(A1_list)
    A2_block = sp.block_diag(A2_list)
    rows, cols, e_attr_1 = sp.find(A1_block)
    _rows, _cols, e_attr_2 = sp.find(A2_block)
    assert rows.all() == _rows.all()
    assert cols.all() == _cols.all()

    # edge list
    print('write {}_A.txt and {}_edge_attributes.txt file...'.format(DS, DS))
    with open(save_path + '{}_A.txt'.format(DS), 'w') as f1:
        with open(save_path + '{}_edge_attributes.txt'.format(DS), 'w') as f2:
            for u, v, ea1, ea2 in zip(rows, cols, e_attr_1, e_attr_2):
                f1.writelines('{}, {}\n'.format(u + 1, v + 1))
                f2.writelines('{}, {}\n'.format(ea1, ea2))
    f1.close()
    f2.close()

    # graph indicator
    print('write {}_graph_indicator.txt file...'.format(DS))
    with open(save_path + '{}_graph_indicator.txt'.format(DS), 'w') as f:
        for i, A in enumerate(A1_list):
            for j in range(A.shape[0]):
                f.writelines('{}\n'.format(i + 1))
    f.close()

    # graph labels
    print('write {}_graph_labels.txt file...'.format(DS))
    with open(save_path + '{}_graph_labels.txt'.format(DS), 'w') as f:
        for y in gy_list:
            f.writelines('{}\n'.format(y))
    f.close()

    # node feature
    print('write {}_node_attributes.txt file...'.format(DS))
    # X_list = feature_resize_padding(X_list)
    Xs = np.vstack(X_list)
    np.savetxt(save_path + '{}_node_attributes.txt'.format(DS), X=Xs, fmt='%d', delimiter=',')

    # node labels
    if ny_list:
        print('write {}_node_labels.txt file...'.format(DS))
        with open(save_path + '{}_node_labels.txt'.format(DS), 'w') as f:
            for y in ny_list:
                f.writelines('{}\n'.format(y))
        f.close()

    # label via node labeling
    if node_importance:
        print('write {}_node_importance_labels.txt file...'.format(DS))
        with open(save_path + '{}_node_importance_labels.txt'.format(DS), 'w') as f:
            for y in node_importance:
                f.writelines('{}\n'.format(y))
        f.close()

    print('write {}_target_edge.txt file...'.format(DS))
    if target_node2label:
        with open(save_path + '{}_target_node2label.txt'.format(DS), 'w') as f:
            for node2label in target_node2label:
                f.writelines('{} {}\n'.format(node2label[0], node2label[1]))
        f.close()