from unittest import TestCase
from haystack.document_stores import ElasticsearchDocumentStore
from langchain.docstore.document import Document
from src.retriever import BM25RetrieverRanker
from src.settings import ELASTICSEARCH_HOST
import haystack
import unittest


class Test_Retriever(TestCase):
    def test_make_retriever_ranker_pipeline(self):
        """
        Test case for the make_retriever_ranker_pipeline method.
        This method tests the functionality of the make_retriever_ranker_pipeline method.
        It ensures that the method correctly creates a pipeline that includes retriever and ranker components.
        """
        document_store = ElasticsearchDocumentStore(
            host=ELASTICSEARCH_HOST,
            username="",
            password="",
            index="document",
        )
        retriever = BM25RetrieverRanker(document_store=document_store)
        pipeline = retriever._make_retriever_ranker_pipeline()

        self.assertEqual(len(pipeline.components), 3)
        self.assertIsInstance(pipeline, haystack.pipelines.base.Pipeline)
        self.assertIn("Retriever", pipeline.components.keys())
        self.assertIn("Ranker", pipeline.components.keys())

    def test_get_relevant_documents(self):
        """
        Test case for the get_relevant_documents method.
        This method tests the functionality of the get_relevant_documents method.
        It verifies that the method retrieves relevant documents based on a given query.
        """

        document_store = ElasticsearchDocumentStore(
            host=ELASTICSEARCH_HOST,
            username="",
            password="",
            index="document",
        )

        retriever = BM25RetrieverRanker(
            document_store=document_store, ranker_top_k=5, retriever_top_k=15
        )
        query = "What is the name of Amazon's membership program that offers free shipping and other benefits?"
        result = retriever.get_relevant_documents(query=query)

        self.assertEqual(len(result), 5)
        self.assertIsInstance(result, list)
        for doc in result:
            self.assertIsInstance(doc, Document)


if __name__ == "__main__":
    unittest.main()
