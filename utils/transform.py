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
from typing import List, Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

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

from typing import List, Optional, Tuple, NamedTuple, Union, Callable
from torch import Tensor
from torch_sparse import SparseTensor


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class My_getTopkGraph(BaseTransform):

    def __init__(self, topk, hop, ess):
        self.topk = topk
        self.hop = hop
        self.ess = ess

    def __call__(self, data):
        # data.edge_attr = None
        # batch = data.batch if 'batch' in data else None


        x = data.x
        edge_attr = data.edge_attr

        subset, edge_index, inv, edge_mask = self.k_hop_subgraph_topk(num_hops=self.hop, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                                                      relabel_nodes=True, num_nodes=data.x.size(0))

        data.x = x[subset]
        data.edge_index = edge_index
        data.edge_attr = edge_attr[edge_mask]
        return data

    def k_hop_subgraph_topk(self, num_hops, edge_index, edge_attr, node_idx=0, relabel_nodes=False,
                            num_nodes=None, flow='source_to_target'):
        r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
        :attr:`node_idx`.
        It returns (1) the nodes involved in the subgraph, (2) the filtered
        :obj:`edge_index` connectivity, (3) the mapping from node indices in
        :obj:`node_idx` to their new location, and (4) the edge mask indicating
        which edges were preserved.

        Args:
            node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
                node(s).
            num_hops: (int): The number of hops :math:`k`.
            edge_index (LongTensor): The edge indices.
            relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
                :obj:`edge_index` will be relabeled to hold consecutive indices
                starting from zero. (default: :obj:`False`)
            num_nodes (int, optional): The number of nodes, *i.e.*
                :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
            flow (string, optional): The flow direction of :math:`k`-hop
                aggregation (:obj:`"source_to_target"` or
                :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

        :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
                 :class:`BoolTensor`)
        """

        assert flow in ['source_to_target', 'target_to_source']
        if flow == 'target_to_source':
            row, col = edge_index
        else:
            col, row = edge_index

        node_mask = row.new_empty(num_nodes, dtype=torch.bool)
        edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device).flatten()
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]

        for _ in range(num_hops):
            neighbors = []
            for n in subsets[-1]:
                node_mask.fill_(False)
                node_mask[n] = True
                torch.index_select(node_mask, 0, row, out=edge_mask)
                candidate_nei = col[edge_mask]

                if self.ess == 'Volume':
                    weights = edge_attr[edge_mask][:, 0]
                    sampled_nei = get_topK(candidate_nei, k=self.topk, weight=weights, get_largest=True)
                elif self.ess == 'Times':
                    weights = edge_attr[edge_mask][:, 1]
                    sampled_nei = get_topK(candidate_nei, k=self.topk, weight=weights, get_largest=True)
                elif self.ess == 'averVolume':
                    weights = edge_attr[edge_mask][:, 0] / edge_attr[edge_mask][:, 1]
                    sampled_nei = get_topK(candidate_nei, k=self.topk, weight=weights, get_largest=True)
                # print(sampled_nei)
                neighbors.append(sampled_nei)
            try:
                neighbors = torch.cat(neighbors).unique()
            except RuntimeError:
                neighbors = torch.tensor([], dtype=torch.int64)
            subsets.append(neighbors)

        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

        node_mask.fill_(False)
        node_mask[subset] = True
        edge_mask = node_mask[row] & node_mask[col]

        edge_index = edge_index[:, edge_mask]

        if relabel_nodes:
            node_idx = row.new_full((num_nodes,), -1)
            node_idx[subset] = torch.arange(subset.size(0), device=row.device)
            edge_index = node_idx[edge_index]

        return subset, edge_index, inv, edge_mask

    def __repr__(self):
        return
