import unittest
from langchain.llms.fake import FakeListLLM
from heavywait.heavywait import HeavyWait
from heavywait.prompts import Prompts


class TestOutput(unittest.TestCase):

    def test_async_all(self):
        responses = [
            "Output one response",
            "keyword1, keyword2, keyword3",
            "Output three response",
            "label1,label2",
            "Output five response"
        ]

        mock_llm = FakeListLLM(responses=responses)
        hw = HeavyWait(llm=mock_llm)
        res = hw._async_chains(Prompts.get_concise_summary_prompt()[1])
        self.assertEqual(res['labels'], responses[3])


if __name__ == '__main__':
    unittest.main()
