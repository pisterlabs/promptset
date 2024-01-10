from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

gpt = os.getenv("OPEN_AI_API_KEY")
serpapi = os.getenv("SERPAPI_API_KEY")



# llm = OpenAI(openai_api_key=gpt, model_name= "gpt-3.5-turbo", max_tokens = 100)

chat_model = ChatOpenAI(openai_api_key=gpt, model_name= "gpt-3.5-turbo", max_tokens = 100)

question = "Which is the best programming language to learn in 2023?"

prompt = PromptTemplate(question, chat_model)
