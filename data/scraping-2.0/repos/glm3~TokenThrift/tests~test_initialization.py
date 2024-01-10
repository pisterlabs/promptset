import pytest
from unittest.mock import Mock
from token_thrift.token_thrift import TokenThrift
from token_thrift.queues.list_queue import ListQueue
from token_thrift.api_client.openai_api_client import OpenAIApiClient


class TestTokenThrift:

    @classmethod
    def setup_class(cls):
        cls.api_key = "sample_api_key"
        cls.budget_in_dollars = 500
        cls.queue = ListQueue()

    def test_initialization_with_valid_inputs(self):
        thrift = TokenThrift(self.api_key, self.budget_in_dollars, self.queue)
        assert thrift.api_key == self.api_key
        assert thrift.budget_in_dollars == self.budget_in_dollars
        assert thrift.total_dollar_spent == 0
        assert thrift.queue.is_empty()
        assert isinstance(thrift.api_client, OpenAIApiClient)

    def test_initialization_negative_budget(self):
        with pytest.raises(ValueError):
            thrift = TokenThrift(self.api_key, -500, self.queue)

    def test_initialization_no_api_key(self):
        with pytest.raises(ValueError):
            thrift = TokenThrift(None, self.budget_in_dollars, self.queue)

    def test_initialization_with_valid_inputs_no_queue(self):
        thrift = TokenThrift(self.api_key, self.budget_in_dollars)
        assert thrift.api_key == self.api_key
        assert thrift.budget_in_dollars == self.budget_in_dollars
        assert thrift.total_dollar_spent == 0
        assert isinstance(thrift.queue, ListQueue)

    def test_initialization_with_custom_api_client(self):
        mock_api_client = Mock()
        thrift = TokenThrift(self.api_key, self.budget_in_dollars, api_client=mock_api_client)
        assert thrift.api_client == mock_api_client