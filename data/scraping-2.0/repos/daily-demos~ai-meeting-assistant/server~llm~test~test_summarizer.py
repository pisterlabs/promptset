import unittest

from server.llm.openai_assistant import OpenAIAssistant


class OpenAIAssistantTests(unittest.TestCase):
    def test_compile_ctx_content(self):
        oai = OpenAIAssistant("fake_key")
        msg = "a test msg"
        metadata = ["Liza", 'voice', "2023-11-13 23:24:10"]
        got_content = oai._compile_ctx_content(msg, metadata)
        want_content = f"[Liza | voice | 2023-11-13 23:24:10] {msg}"
        self.assertEqual(got_content, want_content)
