import unittest

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from chatbot.prompts.chatml import ChatMLPromptTemplate


class TestChatMLPromptTemplate(unittest.TestCase):
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
        tmpl = ChatMLPromptTemplate(input_variables=["input"], messages=messages)
        history = [
            HumanMessage(content="question 1"),
            AIMessage(content="answer 1"),
        ]
        actual = tmpl.format(
            sys="system instruction", history=history, input="question 2"
        )
        expected = """<|im_start|>system
system instruction<|im_end|>
<|im_start|>user
question 1<|im_end|>
<|im_start|>assistant
answer 1<|im_end|>
<|im_start|>user
question 2<|im_end|>
<|im_start|>assistant
"""
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
