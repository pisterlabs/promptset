from unittest import TestCase
from langchain.chat_models import ChatOpenAI
import langchain
import unittest
import src.lang_agent


class Test_Lang_Agent(TestCase):
    def test_setup_llm(self):
        """
        Test case for the setup_llm method.
        This method tests the functionality of the setup_llm.
        It ensures that the method returns an instance of the ChatOpenAI class.
        """

        result = src.lang_agent.setup_llm()
        self.assertIsInstance(result, ChatOpenAI)

    def test_setup_tools(self):
        """
        Test case for the setup_tools method.
        This method tests the functionality of the setup_tools method.
        It verifies that the method returns a list containing one object with specific properties.
        """

        llm = src.lang_agent.setup_llm()
        result = src.lang_agent.setup_tools(llm)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0].name, "BM25 Retrieval")

    def test_make_agent(self):
        """
        Test case for the make_agent method.
        This method tests the functionality of the make_agent method.
        It ensures that the method returns an instance of the AgentExecutor class,
        and that the agent property of the AgentExecutor instance is an instance of the ConversationalChatAgent class.
        """

        result = src.lang_agent.make_agent()
        self.assertIsInstance(result, langchain.agents.AgentExecutor)
        self.assertIsInstance(
            result.agent,
            langchain.agents.conversational_chat.base.ConversationalChatAgent,
        )


if __name__ == "__main__":
    unittest.main()
