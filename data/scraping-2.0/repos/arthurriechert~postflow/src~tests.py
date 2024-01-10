from ghost.manager import GhostManager
from llm.openai.model import OpenAIModel
import tools.load_tools as tool

import llm.openai.chat as chat
import llm.prompts as prompt
import settings as sg
from parse import parse_json

from dotenv import load_dotenv

def test_toolkit():
    """
    Test for managing GPT toolkit

    
    """

    # tool.create_default_toolkit()

    # tools = tool.load_toolkit()

    # print(tools)

    # a_tool = tool.load_tool("Google")

    # print(a_tool)

    llm = OpenAIModel()

    # tool.vectorize_tools(tools, llm)

    desired_tool = tool.find_tool("I want to search the Internet", llm)
    
    print(desired_tool)

def test_openai_vectorize ():
    """
    Test for openai embeddings 

    """
    
    # Initialize OpenAI API
    llm = OpenAIModel()

    print(llm.vectorize(["Hello", "World"]))   

def test_openai_chat ():
    test_message = "Hello"

    response = chat.chat(test_message)

    print(response)

if __name__=="__main__":

    load_dotenv()

    #test_openai_vectorize()

    #test_toolkit()

    test_openai_chat()
