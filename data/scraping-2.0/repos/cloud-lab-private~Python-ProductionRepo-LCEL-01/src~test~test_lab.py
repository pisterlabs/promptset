"""
This file will contain test cases for the automatic evaluation of your
solution in lab/lab.py. You should not modify the code in this file. You should
also manually test your solution by running main/app.py.
"""
import os
import unittest

from langchain.schema.runnable.base import RunnableSequence
from langchain.chat_models import AzureChatOpenAI
from src.main.lab import get_basic_chain, basic_chain_invoke


class TestLLMResponse(unittest.TestCase):
    """
    This test will verify that the connection to an external LLM is made. If it does not
    work, this may be because the API key is invalid, or the service may be down.
    If that is the case, this lab may not be completable.
    """

    def test_llm_sanity_check(self):
        deployment = os.environ['DEPLOYMENT_NAME']
        llm = AzureChatOpenAI(deployment_name=deployment, model_name="gpt-35-turbo")

    """
    The variable returned from the lab function should be an langchain AI response. If this test
    fails, then the AI message request either failed, or you have not properly configured the lab function
    to return the result of the LLM chat.
    """

    def test_return_type_basic_chain(self):
        chain = get_basic_chain()
        self.assertIsInstance(chain, RunnableSequence)
    
    def test_basic_chain_relevancy(self):
        result = basic_chain_invoke("honey bees")
        self.assertIsInstance(result, str)
        self.assertTrue(classify_relevancy(result, "Can you tell me about honey bees?"))
    
   
def classify_relevancy(message, question):
    deployment = os.environ['DEPLOYMENT_NAME']
    llm = AzureChatOpenAI(deployment_name=deployment, model_name="gpt-35-turbo")
    result = llm.invoke(f"Answer the following quest with a 'Yes' or 'No' response. Does the"
                        f"message below successfully answer the following question?"
                        f"message: {message}"
                        f"question: {question}")
    if ("yes" in result.content.lower()):
        return True
    else:
        print(message)
        return False

if __name__ == '__main__':
    unittest.main()