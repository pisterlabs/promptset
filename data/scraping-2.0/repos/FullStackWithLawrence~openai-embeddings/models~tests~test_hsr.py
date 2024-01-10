# -*- coding: utf-8 -*-
# flake8: noqa: F401
"""
Test integrity of base class.
"""
import pytest  # pylint: disable=unused-import
from langchain.chat_models import ChatOpenAI

from models.hybrid_search_retreiver import HybridSearchRetriever
from models.pinecone import PineconeIndex


class TestSalesSupportModel:
    """Test HybridSearchRetriever class."""

    def test_01_basic(self):
        """Ensure that we can instantiate the class."""

        # pylint: disable=broad-except
        try:
            HybridSearchRetriever()
        except Exception as e:
            assert False, f"initialization of HybridSearchRetriever() failed with exception: {e}"

    def test_02_class_aatribute_types(self):
        """ensure that class attributes are of the correct type"""

        hsr = HybridSearchRetriever()
        assert isinstance(hsr.chat, ChatOpenAI)
        assert isinstance(hsr.pinecone, PineconeIndex)
