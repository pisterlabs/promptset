import unittest
from index_builder import IndexBuilder
import index_builder
from llama_index import LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI

class TestIndexBuilder(unittest.TestCase):
    def setUp(self) -> None:
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, max_tokens=512, model_name="gpt-4"))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size=1024)
        self.__index_builder = IndexBuilder(service_context=service_context)

    def test_build_indices(self):
        file_names, index_set = self.__index_builder.build_indices()
        self.assertGreater(len(file_names), 0)
        self.assertGreater(len(index_set), 0)


if __name__=="__main__":
    unittest.main()