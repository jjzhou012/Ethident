#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: utils.py
@time: 2021/12/27 18:46
@desc:
'''

from py2neo import Graph


def neo4j_check_isolate(account, database):
    cypher = "match (n:EOA{name:'" + account + "'}) " \
             + "match (n)-[t:Transaction]-(e:EOA) return e"
    res = database.run(cypher).data()
    # print(res)
    return True if res else False
