import os
import unittest
from unittest.mock import Mock

from injector import Injector, singleton
from langchain.llms.openai import OpenAIChat

from llm_maker import LLAMA_MODEL_PATH
from llm_maker.base_llm import Configuration, LLMModel, LLMService
from llm_maker.llms import GPT4AllModel, LLMFactory, OpenAIModel


class TestOpenAIModel(unittest.TestCase):
    def test_valid_initialization(self):
        config: Configuration = Configuration(
            model_name='gpt-3.5-turbo', temperature=0.7
        )
        model: OpenAIModel = OpenAIModel(config)
        self.assertEqual(model._config, config)
        self.assertIsNotNone(model._provider)


class TestGPT4AllModel(unittest.TestCase):
    def test_valid_initialization(self):
        config: Configuration = Configuration(model_filepath=LLAMA_MODEL_PATH)
        model: GPT4AllModel = GPT4AllModel(config)
        self.assertEqual(model._config, config)
        self.assertIsNotNone(model._provider)


class TestLLMFactory(unittest.TestCase):
    @staticmethod
    def _configure_llm(binder):
        config = Configuration(
            provider='GPT4All',
            temperature=0,
            model_filepath=LLAMA_MODEL_PATH,
            verbose=True,
        )
        binder.bind(Configuration, to=config, scope=singleton)

    def test_provide_model(self):
        config = Configuration(
            provider='OPENAI', model_name='gpt-3.5-turbo', temperature=0.7
        )
        factory: LLMFactory = LLMFactory()
        model = factory.provide_model(config)
        self.assertIsInstance(model, LLMModel)
        self.assertIsInstance(model, OpenAIModel)
        self.assertIsInstance(model._provider, OpenAIChat)

    def test_singleton(self) -> None:
        injector: Injector = Injector([TestLLMFactory._configure_llm, LLMFactory()])
        injector.get(LLMService) is injector.get(LLMService)
