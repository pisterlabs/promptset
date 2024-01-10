import unittest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from chatbot.prompts.mistral import MistralPromptTemplate


class TestMistralPromptTemplate(unittest.TestCase):
    def test_format(self):
        system_prompt = PromptTemplate(
            template="{sys}",
            input_variables=["sys"],
        )
        messages = [
            SystemMessagePromptTemplate(prompt=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
        tmpl = MistralPromptTemplate(input_variables=["input"], messages=messages)
        history = [
            HumanMessage(content="question 1"),
            AIMessage(content="answer 1"),
        ]
        actual = tmpl.format(
            sys="system instruction", history=history, input="question 2"
        )
        expected = """<s> [INST] system instruction question 1 [/INST] answer 1</s> [INST] question 2 [/INST]"""
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
