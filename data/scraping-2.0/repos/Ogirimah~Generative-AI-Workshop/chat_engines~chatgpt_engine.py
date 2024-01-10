from langchain.chat_models import ChatOpenAI
# from langchain.llm import OpenAI
from langchain.schema import HumanMessage
from openai import OpenAI
from dotenv import load_dotenv
import os


# class LLM(OpenAI):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.chat = ChatOpenAI(self)
#
#     def chat(self, prompt, **kwargs):
#         return self.chat(prompt, **kwargs)

# dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
# load_dotenv(dotenv_path)
#
#
# openai_api_key1 = os.environ.get("OPENAI_API_KEY")
#
# llm = OpenAI(api_key=openai_api_key1)

llm = OpenAI()

chat_model = ChatOpenAI()

text = "Hello, how are you?"
messages = [HumanMessage(content=text)]

llm_response = llm.invoke(text)
chat_response = chat_model.invoke(messages)
