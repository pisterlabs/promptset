import unittest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from pybot.prompts.zephyr import ZephyrPromptTemplate


class TestZephyrPromptTemplate(unittest.TestCase):
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
        tmpl = ZephyrPromptTemplate(input_variables=["input"], messages=messages)
        history = [
            HumanMessage(content="question 1"),
            AIMessage(content="answer 1"),
        ]
        actual = tmpl.format(
            sys="system instruction", history=history, input="question 2"
        )
        expected = """<|system|>
system instruction</s>
<|user|>
question 1</s>
<|assistant|>
answer 1</s>
<|user|>
question 2</s>
<|assistant|>
"""
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
