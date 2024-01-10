# test_with_unittest.py
import pandas as pd
import sys
sys.path.append('../..')

import langchain #for class assertions
from google.cloud.bigquery.table import RowIterator #for class exertions
from zeitghost.agents.LangchainAgent import LangchainAgent
from zeitghost.vertex.LLM import VertexLLM#, VertexLangchainLLM
from zeitghost.vertex.Embeddings import VertexEmbeddings
from zeitghost.bigquery.BigQueryAccessor import BigQueryAccessor
import unittest
from unittest import TestCase
dataset='trends_data'
table_id='makeupcosmetics_10054_unitedstates_2840_external'

TEST_PANDAS_SCRIPT = '''This is a dataframe of google search terms (term column) 
scored by volume (score column) by weekly date (date_field column): 
when were certain terms popular compared to others?
why? double check your answer'''

PROJECT_ID = 'cpg-cdp'
gdelt_keyworld = 'estee lauder' #lower cases
term_data_bq = ('mascera','makeup','ulta','tonymoly')
                       
GDELT_COLS = ['SQLDATE', 'Actor1Name', 'Actor2Name', 'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone', 'SOURCEURL']
TRENDSPOTTING_COLS = ['date_field', 'term', 'score']
                       
BQ_AGENT_PROMPT = f"""Describe the {dataset}.{table_id} table? Don't download the entire table, when complete, say I now know the final answer"""
                       
class AgentTests(TestCase):
    
    def __init__(self, project_id=PROJECT_ID,
                 table_id=table_id, 
                 dataset=dataset,
                 gdelt_keyword=gdelt_keyworld,
                 term_data_bq = term_data_bq
                ):
        self.project_id = project_id
        self.table_id = table_id
        self.dataset = dataset
        self.gdelt_keyworld = gdelt_keyworld
        self.term_data_bq = term_data_bq
        self._act()
        self._assert()
        super().__init__(self, project_id=self.project_id,
                 table_id=self.table_id, 
                 dataset=self.dataset,
                 gdelt_keyword=self.gdelt_keyworld,
                 term_data_bq = self.term_data_bq)
        
    def _act(self):
        self.llm = VertexLLM()
        self.llm_test = self.llm.predict('how are you doing today?', ['Observation:'])
        self.langchain_llm = self.llm
        self.langchain_llm_test = self.langchain_llm('how are you doing today?')#, stop=['Observation:']) #you need that for the pandas bot
        self.data_accessor = BigQueryAccessor(self.project_id)
        self.gdelt_accessor = data_accessor.get_records_from_actor_keyword_df(self.gdelt_keyworld)
        self.term_data_from_bq = data_accessor.pull_term_data_from_bq(self.term_data_bq)
        self.trendspotting_subset = self.term_data_from_bq.to_dataframe()
        self.vertex_langchain_agent = LangchainAgent(self.langchain_llm)
        self.trendspotting_subset = self.term_data_from_bq.to_dataframe()
        self.pandas_agent = self.vertex_langchain_agent.get_pandas_agent(self.trendspotting_subset)
        self.pandas_agent_result = pandas_agent.run(TEST_PANDAS_SCRIPT)
        self.langchain_agent_instance = LangchainAgent(self.langchain_llm)
        self.agent_executor = self.langchain_agent_instance.get_bigquery_agent(project_id)
        self.agent_executor_test = self.agent_executor(BQ_AGENT_PROMPT)
                       
    def _assert(self):
        assert True is True #trival start
        assert type(self.llm) is zeitghost.vertex.LLM.VertexLLM
        assert type(self.llm_test) is str
        assert type(self.langchain_llm) is zeitghost.vertex.LLM.VertexLLM
        assert type(self.langchain_llm_test) is str
        assert len(lself.llm_test) > 1
        assert len(self.langchain_llm_test) > 1
        assert type(self.data_accessor) is zeitghost.bigquery.BigQueryAccessor.BigQueryAccessor
        assert type(self.gdelt_accessor) is pd.core.frame.DataFrame #is this right??
        assert len(self.gdelt_accessor) > 1
        assert type(self.term_data_from_bq) is RowIterator
        assert self.gdelt_accessor.columns.to_list() == GDELT_COLS
        assert type(self.trendspotting_subset) == pd.core.frame.DataFrame
        assert len(self.trendspotting_subset) > 1
        assert self.trendspotting_subset.columns.to_list() == TRENDSPOTTING_COLS
        assert type(self.vertex_langchain_agent) is zeitghost.agents.LangchainAgent.LangchainAgent
        assert type(self.pandas_agent) is langchain.agents.agent.AgentExecutor
        assert len(self.pandas_agent_result) > 1
        assert type(self.langchain_agent_instance) is zeitghost.agents.LangchainAgent.LangchainAgent
        assert type(self.agent_executor) is langchain.agents.agent.AgentExecutor
        assert len(agent_executor_test) > 1
    



            
        