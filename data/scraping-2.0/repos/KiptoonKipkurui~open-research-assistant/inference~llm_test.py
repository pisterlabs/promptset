"""
UNIT tests related to LLMS
"""
import unittest

import box
from langchain.llms import CTransformers
from langchain.llms.fake import FakeListLLM

from .llm import setup_dbqa

cfg = box.box_from_file("./config/config.yml", "yaml")

class TestLLMs(unittest.TestCase):

    """
    Testing LLMs test case
    """
    # def test_build_llm(self):
    #     """
    #     Tests building an LLM
    #     """
    #     model = build_llm(config=cfg)
    #     self.assertIsInstance(model, CTransformers)
    #     self.assertEqual(model.model_type, cfg.MODEL_TYPE)
    #     self.assertEqual(model.model, cfg.MODEL_BIN_PATH)
    #     self.assertEqual(model.config["max_new_tokens"], cfg.MAX_NEW_TOKENS)
    #     self.assertEqual(model.config["temperature"], cfg.TEMPERATURE)

    def test_retrieval_qa(self):

        """
        Test retrieval qa
        """

        responses = ['In the context of artificial intelligence and natural language processing, a Large Language Model refers to a sophisticated computational model designed to understand and generate human-like text. These models are trained on vast amounts of diverse text data to learn patterns, language structure, and context.GPT-3, the model behind my responses, is an example of a Large Language Model developed by OpenAI. It stands for "Generative Pre-trained Transformer 3," and it has 175 billion parameters, making it one of the most powerful language models created as of my knowledge cutoff in January 2022. Large Language Models like GPT-3 are capable of a wide range of language-related tasks, including text completion, translation, summarization, question answering, and more']
        llm = FakeListLLM(responses=responses)


        dbqa=setup_dbqa(cfg,llm)

        response = dbqa({"query": "What is an LLM"})
        print(response)

if __name__ == "__main__":
    unittest.main()
