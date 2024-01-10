import openai
import os
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chains import SimpleSequentialChain, SequentialChain

from dotenv import load_dotenv
load_dotenv()

