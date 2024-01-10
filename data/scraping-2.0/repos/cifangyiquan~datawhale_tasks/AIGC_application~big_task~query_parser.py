#!/bin/env python
#coding=utf-8

import os
import sys
import re
from neo4j import GraphDatabase
import openai_api

class QueryParser:
    def __init__(self):
        self.driver = GraphDatabase.driver("bolt://localhost:7687", 
                                        auth=('neo4j', 'neo4j'))

    def parse_query(self, query):
        # parse query 
        prompt = '对下面的语句做实体和关系的分析，生成neo4j的查询语句：' + query
        res = openai_api.get_completions(prompt)
        cql = res.split('\n')[-1]
        session = self.driver.session()
        rets = session.execute_read(self._search, cql)
        print(rets)
        out = ''
        for ret in rets:
            out += ret.value() + ' '
        print(out)
        return out
        
    @staticmethod
    def _search(tx, cmd):
        if cmd:
            result = tx.run(cmd)
            rets = list(result)
            return rets
        else:
            return None  

if __name__ == '__main__':
    query = sys.argv[1]
    qp = QueryParser()
    ret = qp.parse_query(query)
    print(ret)
