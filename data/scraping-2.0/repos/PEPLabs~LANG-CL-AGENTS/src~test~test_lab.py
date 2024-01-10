"""
This file will contain test cases for the automatic evaluation of your
solution in main/lab.py. You should not modify the code in this file. You should
also manually test your solution by running app.py.
"""

import unittest

from langchain_core.outputs import LLMResult

from src.main.lab import agent_executor
from src.utilities.llm_testing_util import llm_connection_check, llm_wakeup


class TestLLMResponse(unittest.TestCase):
    """
    This function is a sanity check for the Language Learning Model (LLM) connection.
    It attempts to generate a response from the LLM. If a 'Bad Gateway' error is encountered,
    it initiates the LLM wake-up process. This function is critical for ensuring the LLM is
    operational before running tests and should not be modified without understanding the
    implications.
    Raises:
        Exception: If any error other than 'Bad Gateway' is encountered, it is raised to the caller.
    """
    def test_llm_sanity_check(self):
        try:
            response = llm_connection_check()
            self.assertIsInstance(response, LLMResult)
        except Exception as e:
            if 'Bad Gateway' in str(e):
                llm_wakeup()
                self.fail("LLM is not awake. Please try again in 3-5 minutes.")

    """
    This test will verify that the agent uses the appropriate tool for the task given.
    """
    def test_appropriate_tools_used_by_agent(self):

        agent_executor.max_iterations = 1

        response = agent_executor.invoke(
            {"input": "How many letters are in the word Jurassic?"},
        )

        tool_used = response["intermediate_steps"][0][0].tool

        # Verifies that the get_word_tool is used when the agent is asked to find the length of a word
        self.assertEqual("get_word_length", tool_used)

        response = agent_executor.invoke(
            {"input": "What is 3 cubed?"},
        )

        tool_used = response["intermediate_steps"][0][0].tool

        # Verifies that the get_cube_of_number is used when the agent is asked to find the cube of a number
        self.assertEqual("get_cube_of_number", tool_used)

    """
    This test will verify that the agent produces the correct word length.
    """
    def test_agent_gets_length_of_word(self):

        agent_executor.max_iterations = 1

        response = agent_executor.invoke({"input": "How many letters are in the word Jurassic?"})

        # have to grab output this way due to some weirdness with huggingface's behavior with langchain agents
        print(response["intermediate_steps"][0][1])

        self.assertEqual(response["intermediate_steps"][0][1], 8)

    """
    This test will verify that the agent produces the correct cube.
    """
    def test_agent_gets_cube_of_number(self):

        response = agent_executor.invoke("what is 3 cubed?",)

        self.assertEqual(response["intermediate_steps"][0][1], 27)
