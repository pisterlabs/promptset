from analysis_executor import AnalysisExecutor
from langchain.docstore.document import Document
from acc_checker.text_prep.create_chromaDB import get_chroma_db
from langchain.callbacks import get_openai_callback
import unittest



class TestAnalysisExecutor(unittest.TestCase):

    def setUp(self) -> None:
        self.prompt = "Does 1 + 1 equal 4? Answer 'Yes' or 'No'"
        self.criteria_set = [{"name": "crit a", "subcriteria": [{"name": "crit a.a", "prompt": self.prompt}, {"name": "crit a.b", "prompt": self.prompt}]},
                             {"name": "crit b", "prompt": self.prompt}]
        self.doc1 = Document(page_content="1 + 1 does not equal 4", metadata={"source": "source_a"})
        self.doc2 = Document(page_content="France is an interesting country", metadata={"source": "source_b"})
        # self.vector_store = Embedder([self.doc1, self.doc2]).vector_store
        self.vector_store = get_chroma_db()
        self.ae = AnalysisExecutor(self.criteria_set, self.vector_store)
        self.retrieval_chain = self.ae.get_retrieval_chain(self.vector_store)


    def test_get_llm_response_with_sources(self):
        response = self.ae.get_llm_response_with_sources(self.retrieval_chain, self.prompt)
        assert 'No' in response['result']
        assert response['source_documents'][0].page_content == "1 + 1 does not equal 4"


    def test_combine_doc_strings(self):
        docs = [self.doc1, self.doc2]
        source_string = self.ae.combine_doc_strings(docs)
        self.assertEquals(source_string, "1 + 1 does not equal 4 (source: source_a) France is an interesting country (source: source_b) ")


    def test_get_and_store_all_llm_response_and_source_docs(self):
        response_set, cost = self.ae.get_and_store_all_llm_responses_and_source_docs(self.criteria_set, self.retrieval_chain)
        for c in response_set:
            if "subcriteria" in c:
                for s in c["subcriteria"]:
                    assert "No" in s["response"]
                    assert "source_a" in s["source"]
            else:
                assert "No" in c["response"]
                assert "source_a" in c["source"]
        assert isinstance(cost.total_cost, float)
