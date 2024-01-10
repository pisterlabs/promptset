import os
import openai
from dotenv import load_dotenv
from transformers import OpenAiAgent
from  transformers import HfAgent
# from huggingface_hub import login
from langchain.agents import load_huggingface_tool




load_dotenv()

# login(os.getenv('HUGGINGFACE_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')



tool = load_huggingface_tool("lysandre/hf-model-downloads")


print(f"{tool.name}  : {tool.description}")


tool.run("text-classification")



# audio = tool("This is a text to speech tool")