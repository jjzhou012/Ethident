#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: transform.py
@time: 2022/1/16 1:00
@desc: custom data transform class
'''
import torch
from torch.distributions.bernoulli import Bernoulli
import numpy as np
import random
from torch_sparse import SparseTensor
from torch_geometric.utils import dropout_adj, degree, to_undirected, subgraph
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

from typing import List, Optional, Tuple, NamedTuple, Union, Callable
from torch import Tensor

from utils.utils import get_topK


class ColumnNormalizeFeatures(BaseTransform):
    r"""column-normalizes the attributes given in :obj:`attrs` to sum-up to one.

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """

    def __init__(self, attrs: List[str] = ["edge_attr"]):
        self.attrs = attrs

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value.div_(value.sum(dim=0, keepdim=True).clamp_(min=1.))  # change dim
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class MyToUndirected(BaseTransform):
    r"""Converts a homogeneous or heterogeneous graph to an undirected graph
    such that :math:`(j,i) \in \mathcal{E}` for every edge
    :math:`(i,j) \in \mathcal{E}`.
    In heterogeneous graphs, will add "reverse" connections for *all* existing
    edge types.

    Args:
        reduce (string, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)
        merge (bool, optional): If set to :obj:`False`, will create reverse
            edge types for connections pointing to the same source and target
            node type.
            If set to :obj:`True`, reverse edges will be merged into the
            original relation.
            This option only has effects in
            :class:`~torch_geometric.data.HeteroData` graph data.
            (default: :obj:`True`)
    """

    def __init__(self, reduce: str = "add", merge: bool = True, edge_attr_keys: List[str] = None):
        self.reduce = reduce
        self.merge = merge
        self.edge_attr_keys = edge_attr_keys

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            if 'edge_index' not in store:
                continue

            nnz = store.edge_index.size(1)

            if isinstance(data, HeteroData) and (store.is_bipartite()
                                                 or not self.merge):
                src, rel, dst = store._key

                # Just reverse the connectivity and add edge attributes:
                row, col = store.edge_index
                rev_edge_index = torch.stack([col, row], dim=0)

                inv_store = data[dst, f'rev_{rel}', src]
                inv_store.edge_index = rev_edge_index
                for key, value in store.items():
                    if key == 'edge_index':
                        continue
                    if isinstance(value, Tensor) and value.size(0) == nnz:
                        inv_store[key] = value

            else:
                keys, values = [], []
                for key, value in store.items():
                    if key == 'edge_index':
                        continue
                    if key not in self.edge_attr_keys:
                        continue

                    if store.is_edge_attr(key):
                        keys.append(key)
                        values.append(value)

                store.edge_index, values = to_undirected(
                    store.edge_index, values, reduce=self.reduce)

                for key, value in zip(keys, values):
                    store[key] = value

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


# data augmentation

class MyAug_Identity(BaseTransform):

    def __init__(self, prob=None):
        self.prob = prob

    def __call__(self, data):
        return data

    def __repr__(self):
        return '{}(prob={})'.format(self.__class__.__name__, self.prob)


class MyAug_EdgeRemoving(BaseTransform):

    def __init__(self, prob=None):
        self.prob = prob

    def __call__(self, data):
        # data.edge_attr = None
        # batch = data.batch if 'batch' in data else None
        edge_index, edge_attr = dropout_adj(edge_index=data.edge_index, edge_attr=data.edge_attr, p=self.prob, num_nodes=data.x.size(0))

        if edge_index.size(1) == 0:
            return data

        data.edge_index = edge_index
        data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}(prob={})'.format(self.__class__.__name__, self.prob)


class MyAug_NodeDropping(BaseTransform):

    def __init__(self, prob=None):
        self.prob = prob

    def __call__(self, data):
        # data.edge_attr = None
        # batch = data.batch if 'batch' in data else None

        keep_mask = torch.empty(size=(data.x.size(0),), dtype=torch.float32).uniform_(0, 1) > 1 - self.prob
        keep_mask[0] = True

        if keep_mask.sum().item() < 2:
            return data

        edge_index, edge_attr = subgraph(keep_mask, data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=data.x.size(0))

        subset = keep_mask.nonzero().squeeze()
        x = data.x[subset, :]

        data.x = x
        data.edge_index = edge_index
        data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}(prob={})'.format(self.__class__.__name__, self.prob)


class MyAug_NodeAttributeMasking(BaseTransform):

    def __init__(self, prob=None):
        self.prob = prob

    def __call__(self, data):
        # data.edge_attr = None
        # batch = data.batch if 'batch' in data else None
        drop_mask = torch.empty(size=(data.x.size(1),), dtype=torch.float32).uniform_(0, 1) < self.prob
        data.x[:, drop_mask] = 0

        return data

    def __repr__(self):
        return '{}(prob={})'.format(self.__class__.__name__, self.prob)


class MyAug_EdgeAttributeMasking(BaseTransform):

    def __init__(self, prob=None):
        self.prob = prob

    def __call__(self, data):
        # data.edge_attr = None
        # batch = data.batch if 'batch' in data else None
        drop_mask = torch.empty(size=(data.edge_attr.size(1),), dtype=torch.float32).uniform_(0, 1) < self.prob
        data.edge_attr[:, drop_mask] = 0

        return data

    def __repr__(self):
        return '{}(prob={})'.format(self.__class__.__name__, self.prob)


Augmentor_Transform = {
    '': MyAug_Identity,
    'identity': MyAug_Identity,
    'edgeRemove': MyAug_EdgeRemoving,
    'edgeAttrMask': MyAug_EdgeAttributeMasking,
    'nodeDrop': MyAug_NodeDropping,
    'nodeAttrMask': MyAug_NodeAttributeMasking,
}





