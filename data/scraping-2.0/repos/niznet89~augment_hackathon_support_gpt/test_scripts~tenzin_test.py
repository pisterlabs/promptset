from llama_index.tools import FunctionTool
from llama_index.llms import OpenAI
from llama_index.agent import ReActAgent
from dotenv import load_dotenv
import os
import openai

load_dotenv()

cohere_api_key = os.environ.get("COHERE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API")

print(openai_api_key)

os.environ['OPENAI_API_KEY'] = openai_api_key
#os.environ['ACTIVELOOP_TOKEN'] = activeloop_token
#embeddings = OpenAIEmbeddings()
openai.api_key = openai_api_key



# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

def web_search(input) -> int:
    """Useful if you want to search the web - you will need to enter an appropriate search query to get more information"""
    return True

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
ali_balls = FunctionTool.from_defaults(fn=web_search)

# initialize llm
llm = OpenAI(model="gpt-4")

# initialize ReAct agent
agent = ReActAgent.from_tools([multiply_tool, add_tool, ali_balls], llm=llm, verbose=True)


response = agent.chat("Does Ali like apples? If you can't answer have your Response: NO. Always use a Tool")
response_1 = agent.chat("What was the previous question and answer?")

print(response, response_1)
