from langchain.agents import load_tools
from langchain.agents import initialize_agent
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.getenv('GOOGLE_API_KEY'),temperature=0)
