# try using GPT List Index!
from langchain import OpenAI
from langchain.agents import initialize_agent

from llama_index import GPTListIndex, GPTIndexMemory



if __name__ == "__main__":
    print("Initializing agent...")