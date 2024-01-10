import json
from operator import itemgetter
from typing import List

from astrapy.db import AstraDBCollection
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser, ListOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)

import unittest
from unittest.mock import MagicMock, patch

from sentence_transformers import SentenceTransformer

import ChainFactory
import PromptFactory
from DataAccess import DataAccess
import uuid

from pydantic_models.ColumnSchema import ColumnSchema
from pydantic_models.PropertyInfo import PropertyInfo
from pydantic_models.TableExecutionInfo import TableExecutionInfo
from pydantic_models.TableSchema import TableSchema
from pydantic_models.UserInfo import UserInfo


class TestAstrapy(unittest.TestCase):
    def test_astrapy_execution(self):
        context = {}
        code = """
import os
from astrapy.db import AstraDB as AstraPyDB
db = AstraPyDB(token=os.getenv("ASTRA_DB_TOKEN_BASED_PASSWORD"), api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"))
"""
        exec(code, context)
        db = context["db"]
        print(db)

    def test_astrapy_find(self):
        import os
        from astrapy.db import AstraDB as AstraPyDB

        db = AstraPyDB(
            token=os.getenv("ASTRA_DB_TOKEN_BASED_PASSWORD"),
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
        )
        embedding_model = "all-MiniLM-L12-v2"
        embedding_direct = SentenceTransformer(
            "sentence-transformers/" + embedding_model
        )
        example_msg = (
            "Hi, I'm having an issue with my iPhone 6. The network isn't working"
        )
        input_vector = embedding_direct.encode(example_msg).tolist()

        # mycollections = db.get_collections()["status"]["collections"]

        # Assume that all collections are relevant for now.
        # Later, we will use a chain to get only the relevant ones.
        collection = AstraDBCollection(collection_name="sitemapls", astra_db=db)
        results = collection.vector_find(
            vector=input_vector,
            filter={"metadata.path_segment_1": "support"},
            limit=100,
        )
        print(results)
