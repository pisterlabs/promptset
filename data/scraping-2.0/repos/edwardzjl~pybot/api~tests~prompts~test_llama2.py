import unittest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from pybot.prompts.llama2 import Llama2PromptTemplate


class TestLLaMA2PromptTemplate(unittest.TestCase):
    def test_format_first_round(self):
        system_prompt = PromptTemplate(
            template="{sys}",
            input_variables=["sys"],
        )
        messages = [
            SystemMessagePromptTemplate(prompt=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
        tmpl = Llama2PromptTemplate(input_variables=["input"], messages=messages)
        history = []
        actual = tmpl.format(
            sys="system instruction", history=history, input="question 1"
        )
        expected = """<s>[INST] <<SYS>>
system instruction
<</SYS>>

question 1 [/INST]"""
        self.assertEqual(actual, expected)

    def test_format_second_round(self):
        system_prompt = PromptTemplate(
            template="{sys}",
            input_variables=["sys"],
        )
        messages = [
            SystemMessagePromptTemplate(prompt=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
        tmpl = Llama2PromptTemplate(input_variables=["input"], messages=messages)
        history = [
            HumanMessage(content="question 1"),
            AIMessage(content="answer 1"),
        ]
        actual = tmpl.format(
            sys="system instruction", history=history, input="question 2"
        )
        expected = """<s>[INST] <<SYS>>
system instruction
<</SYS>>

question 1 [/INST] answer 1 </s><s>[INST] question 2 [/INST]"""
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
