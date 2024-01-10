import csv
import json
import os

from generators import OrderAnswerGenerator

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

from unittest import TestCase


class TestOrderAnswerGenerator(TestCase):
    orders: list = []

    def setUp(self):
        self.llm = LlamaCpp(
            model_path=os.getenv('LLM_MODEL_PATH'),
            n_gpu_layers=2,
            n_batch=512,
            n_ctx=2048,
            f16_kv=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )

        with open('resources/orders.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for order in reader:
                self.orders.append(order)

        self.prompt = PromptTemplate(
            input_variables=["question", "provider", "heavy_limit", "assistant_style", "order"],
            template='<<SYS>>You are a {assistant_style} assistant giving short, relevant answers. Do not give too many details.\n \
            Whoosh is the delivery service, \n \
            {provider} heavy order weight limit: {heavy_limit} gr \n \
            Order is shipped via an {provider} scooter.\n \
            Order is groceries. \n \
            Order can be tracked on MyMart Online or in the MyMart app. There is no tracking link, there is a tacking screen. \n \
            Order {order} \
            <</SYS>>\n \
            [INST]{question}[/INST]'
        )

        self.validation_prompt = PromptTemplate(
            input_variables=["answer", "validation_criteria"],
            template='<<SYS>>You are validating an input text. You are giving simple Yes or No answers only.<</SYS>>\n \
            [INST]Does this have any {validation_criteria}: \n {answer}[/INST]'
        )

    def test_get_answer_with_item_count(self):
        generator = OrderAnswerGenerator(llm=self.llm,
                                         prompt=self.prompt,
                                         validation_prompt=self.validation_prompt,
                                         validation_criteria="negativity, sarcasm, offensive remark",
                                         assistant_style="cheerful")

        question = "When my order will arrive? I forgot, how many items it has?"

        for order in self.orders:
            provider = order['delivery courier']
            item_count = order['items']
            result = generator.get_answer(order, provider, question)

            print(json.dumps(result, indent=2))

            self.assertIsNotNone(result['answer'])
            self.assertIn(item_count, result['answer'])
            self.assertIsNotNone(result['validation_result'])
            self.assertTrue(result['is_valid'])
