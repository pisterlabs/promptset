import unittest

from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from pybot.prompts.vicuna import VicunaPromptTemplate


class TestVicunaPromptTemplate(unittest.TestCase):
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
        tmpl = VicunaPromptTemplate(input_variables=["input"], messages=messages)
        history = []
        actual = tmpl.format(
            sys="system instruction", history=history, input="question 1"
        )
        expected = """system instruction
USER: question 1
ASSISTANT:"""
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
