#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: 1.py
@time: 2021/12/27 16:23
@desc: extract ETH data from neo4j database, and pack into TU dataset
'''

import random
import numpy as np
import pickle as pkl
from py2neo import Graph
from tqdm import tqdm
import networkx as nx
from collections import deque
import argparse
import os
import os.path as osp
import multiprocessing as mp
import time

from prepare_data.util import neo4j_check_isolate
from prepare_data.process_tu_data import to_tu_file

root_path = osp.dirname(osp.realpath(__file__))
print(root_path)



# database config
neo4j_G = Graph("", auth=("", ""))

# data information
label_abbreviation = {"i": "ico-wallets",
                      "m": "mining",
                      "e": "exchange",
                      "p": "phish-hack"}

###### file load
# account associated with the feature of contract call
contract_list = pkl.load(open("data/ETH/contract_filter_list.pkl", 'rb'))
contracts2index = {con: idx for idx, con in enumerate(contract_list)}
# target accounts for classification       {type: []}   ->  e.g. {exchange: [], phish-hack, [], mining: [], ico-wallets: []}
label2account = pkl.load(open("data/SelectedAccount.pkl", 'rb'))
# Blacklist (accounts with self-loop only)  set
black_list = pkl.load(open("data/ETH/SelfLoopBlacklist.pkl", 'rb'))
# list of exchange accounts
exchange_file_path = 'data/eth/exchange/pos_neg.pkl'


def Parameter():
    parser = argparse.ArgumentParser(description='Data prepare for gnn')
    parser.add_argument('-d', '--data', type=str, help='eth, eos', default='eth')
    parser.add_argument('-l', '--label', type=str, help='p, e, i, m', default='i')
    parser.add_argument('-ess', '--edge_sample_strategy', type=str, help='Volume, Times, averVolume', default='Volume')
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('-k', '--topk', type=int, default=3)
    parser.add_argument('-p', '--parallel', type=int, help='parallel', default=0)

    parser.add_argument('-seed', '--seed', type=int, default=123)
    return parser.parse_args()


'''
 subgraph extract function
'''


def subgraph_extract(account, args, label, exchanges):
    global cypher
    subgraph = nx.DiGraph()
    subgraph.add_node(account)  # treat target account as the first node
    node_deque = deque()
    visited = set()
    finded = {account}
    node_deque.append(account)

    node_important_dict = {account: 0}
    current_hop = 0

    while current_hop <= args.hop and node_deque:
        current_len = len(node_deque)
        for _ in range(current_len):
            name = node_deque.popleft()
            if name in visited: continue
            # avoid extracting exchange accounts for other types of target accounts.
            if label != "exchange" and name in exchanges: continue

            if args.edge_sample_strategy in ['Volume', 'Times']:
                cypher = "match (n:EOA{name:'" + name + "'}) " \
                         + "match (n)-[t:Transaction]-(e:EOA) " \
                         + "return e.name as neighbor, startNode(t).name as start, endNode(t).name as end," \
                         + "t.Volume as volume, t.Times as times order by toFloat(t.{}) desc limit {}" \
                             .format(args.edge_sample_strategy, args.topk)
            elif args.edge_sample_strategy == 'averVolume':
                cypher = "match (n:EOA{name:'" + name + "'}) " \
                         + "match (n)-[t:Transaction]-(e:EOA) " \
                         + "return e.name as neighbor, startNode(t).name as start, endNode(t).name as end," \
                         + "t.Volume as volume, t.Times as times order by toFloat(t.Volume)/toFloat(t.Times) desc limit {}" \
                             .format(args.topk)

            res = neo4j_G.run(cypher).data()

            if not res: continue
            for item in res:
                # remove self-loop
                if item['start'] == item['end']: continue

                if current_hop < args.hop:
                    if subgraph.has_successor(item['start'], item['end']):  #
                        continue
                    else:
                        node_deque.append(item['neighbor'])
                        finded.add(item['neighbor'])
                        # Avoid backtracking of reverse edges, resulting in re-labeling
                        if item['neighbor'] not in node_important_dict:
                            node_important_dict[item['neighbor']] = current_hop + 1
                        subgraph.add_edge(item['start'], item['end'], volume=float(item['volume']), times=int(item['times']))
                elif current_hop == args.hop:
                    # Current Target Neighbor -> Find a link with an existing neighbor node
                    if item['start'] not in finded or item['end'] not in finded: # Supplement the transaction between neighbors, and the new node does not continue to be included
                        continue
                    else:
                        subgraph.add_edge(item['start'], item['end'], volume=float(item['volume']), times=int(item['times']))

            visited.add(name)
        current_hop += 1

    # construct contract call features
    Xs = []
    node_important_list = []
    for idx, node in enumerate(subgraph.nodes()):
        cypher = "match (n:EOA{name:'" + node + "'}) " \
                 + "match (n)-[t:Call]->(e:CA) " \
                 + "return e.name as c, t.Times as times"
        feat = np.zeros(shape=(len(contract_list),))
        res = neo4j_G.run(cypher).data()
        if res:
            for item in res:
                contract = item['c']
                if contract in contracts2index:
                    feat[contracts2index[contract]] = int(item['times'])
        Xs.append(feat)
        node_important_list.append(node_important_dict[node])

    X = np.vstack(Xs)
    assert nx.number_of_nodes(subgraph) == X.shape[0]

    return subgraph, X, account, node_important_list


def parallel_worker_of_subgraph_extract(x):
    # return subgraph_extract(*x)
    return subgraph_extract(*x)


def main():
    t0 = time.time()
    args = Parameter()

    label = label_abbreviation[args.label]

    # data local path
    save_path =  "data-align/{}/{}/{}hop-{}/{}/".format(args.data, label, args.hop, args.topk, args.edge_sample_strategy)
    save_path = osp.join(root_path, save_path)
    print(save_path)
    if not os.path.exists(save_path): os.makedirs(save_path)

    if os.path.exists(save_path + '{}G_A.pkl'.format(args.data.upper())):
        print('File exist, finish!')
    else:
        print("label:{}, hop:{}, topk:{}, strategy:{}".format(label, args.hop, args.topk, args.edge_sample_strategy))
        # Exchange accounts have dense neighborhoods. Avoid sampling exchange accounts when sampling subgraphs of other types of accounts
        if label != "exchange":
            exchanges = pkl.load(open(osp.join(root_path, exchange_file_path), 'rb'))['pos']
        else:
            exchanges = None

        # get pos sample, and filter
        print('Get pos samples ...')
        if os.path.exists("data/{}/{}/pos_neg.pkl".format(args.data, label)):
            pos = pkl.load(open("data/eth/{}/pos_neg.pkl".format(label), 'rb'))['pos']
        else:
            pos = label2account[label]
        pos = [account for account in tqdm(pos) if account not in black_list]
        #
        if label not in ['mining', 'exchange']:
            pos = [account for account in tqdm(pos) if neo4j_check_isolate(account, neo4j_G)]
        # get neg sample
        print('Get neg samples ...')
        if os.path.exists("data/eth/{}/pos_neg.pkl".format(label)):
            print('    load existed neg sample ...')
            neg = pkl.load(open("data/eth/{}/pos_neg.pkl".format(label), 'rb'))['neg']
        else:
            print('    Start sampling neg sample ...')
            # times = tqdm(range(1000))
            neg_cand = []
            neg = []
            for i in range(1000):
                cypher = "match (n:EOA) return n.name skip {} limit {}".format(i * 100, 100)
                neg_all = neo4j_G.run(cypher).data()
                for n in neg_all:
                    name = n['n.name']
                    if name not in pos and name not in black_list:
                        neg_cand.append(name)
                if len(neg_cand) > len(pos) * 10:
                    break

            while len(neg) < len(pos):
                account = random.choice(neg_cand)
                if account in neg:
                    continue
                else:
                    if neo4j_check_isolate(account, neo4j_G):
                        neg.append(account)

    #########################################################################
    #################################################################
    # subgraph extract
    print('Start subgraph extracting ...')
    Gs = []
    Xs = []
    Ys = []
    node_important_label_list = []
    assert len(pos) == len(neg)
    if not args.parallel:
        print('    no parallel ...')
        for account in tqdm(pos + neg):
            sg, X, target, node_important_label = subgraph_extract(account, args, label, exchanges)
            Gs.append(sg)
            Xs.append(X)
            Ys += [1] if account in pos else [0]
            node_important_label_list += node_important_label
    else:
        print('    parallel ...')
        start = time.time()
        pool = mp.Pool(processes=1)
        results = pool.map_async(
            parallel_worker_of_subgraph_extract,
            [(account, args, label, exchanges) for account in (pos + neg)]
        )
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready(): break
            remaining = results._number_left
            time.sleep(1)
        results = results.get()
        pool.close()
        for sg, X, target, node_important_label in results:
            Gs.append(sg)
            Xs.append(X)
            Ys += [1] if target in pos else [0]
            node_important_label_list += node_important_label
        end = time.time()
        print("Time eplased for subgraph extraction: {}s".format(end - start))

    account2label = list(zip(pos + neg, Ys))
    to_tu_file(G_list=Gs, X_list=Xs, gy_list=Ys, path=save_path,
               node_importance=node_important_label_list,
               dataname=args.data, target_node2label=account2label)
    print('Time: {}'.format(time.time() - t0))

if __name__ == '__main__':
    main()
