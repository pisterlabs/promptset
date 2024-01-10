"""
This file will contain test cases for the automatic evaluation of your
solution in lab/lab.py. You should not modify the code in this file. You should
also manually test your solution by running main/app.py.
"""
import os
import unittest

from langchain.schema.runnable.base import RunnableSequence
from langchain.llms import HuggingFaceEndpoint
from langchain.schema.output_parser import StrOutputParser
from src.main.lab import get_movie_to_actors_chain, get_actors_to_movies_chain, get_final_chain
from src.utilities.llm_testing_util import llm_connection_check, llm_wakeup, classify_relevancy


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
    The variable returned from the lab function should be an langchain AI response. If this test
    fails, then the AI message request either failed, or you have not properly configured the lab function
    to return the result of the LLM chat.
    """
    
    def test_return_type_movie_to_actors_chain(self):
        chain = get_movie_to_actors_chain()
        self.assertIsInstance(chain, RunnableSequence)

    def test_return_type_actors_to_movies_chain(self):
        chain = get_actors_to_movies_chain()
        self.assertIsInstance(chain, RunnableSequence)
    
    def test_return_type_final_chain(self):
        chain = get_final_chain()
        self.assertIsInstance(chain, RunnableSequence)
    
    def test_movies_to_actors_chain_relevancy(self):
        result = get_movie_to_actors_chain().invoke({"movie": "The Wizard of Oz"})
        self.assertIsInstance(result, dict)
        self.assertTrue(classify_relevancy(result, "What actors are in the Wizard of Oz?"))

    def test_final_chain_relevancy(self):
        result = get_final_chain().invoke({"movie": "The God-Father"})
        
        self.assertTrue(classify_relevancy(result, "What movie(s) share at least one common actor with The God-Father?"))

if __name__ == '__main__':
    unittest.main()