import os
import pandas as pd
import unittest
from unittest.mock import patch
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from unittest.mock import MagicMock
from llm_agent import init_agent, set_openai_key, get_agent_response

class TestPandasAI(unittest.TestCase):
    
    def setUp(self):
        self.api_key = 'test_api_key'
        self.df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        self.user_input = 'test_user_input'
        self.agent = init_agent(self.api_key)
    
    def tearDown(self):
        pass
    
    def test_set_openai_key(self):
        api_key = set_openai_key(self.api_key)
        self.assertEqual(api_key, self.api_key)
        self.assertEqual(os.environ["OPENAI_API_KEY"], self.api_key)
    
    def test_init_agent(self):
        llm = OpenAI(api_token=self.api_key)
        pandas_ai = PandasAI(llm, conversational=False)
        agent = init_agent(self.api_key)
        self.assertEqual(agent.llm.api_token, pandas_ai.llm.api_token)
        self.assertEqual(agent.conversational, pandas_ai.conversational)
    
    @patch.object(PandasAI, 'run', return_value='test_response')
    def test_get_agent_response(self, mock_run):
        response = get_agent_response(self.agent, self.df, self.user_input)
        mock_run.assert_called_once_with(self.df, self.user_input)
        self.assertEqual(response, 'test_response')