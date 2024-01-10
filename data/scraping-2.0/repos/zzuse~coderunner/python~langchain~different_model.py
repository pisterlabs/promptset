from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import time
import os


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm=OpenAI(model_name='text-ada-001', openai_api_key=api_key)
result = llm("What day comes after Friday?")
print(result)

chat = ChatOpenAI(temperature=1, openai_api_key=api_key)
result = chat(
    [
    SystemMessage(content="You are an unhelpful AI bot that makes a joke at whatever the user says."),
    HumanMessage(content="I would like to go to New York, how should I do this?")
])
print(result)

embeddings = OpenAIEmbeddings(openai_api_type=api_key)
text = "Hi! It's time for the beach"
text_embedding = embeddings.embed_query(text)
print(f"Your embedding is length {len(text_embedding)}")
print(f"Here's a sample: {text_embedding[:5]}...")

llm=OpenAI(model_name='text-davinci-003', openai_api_key=api_key)
prompt = """
Today is Monday, tomorrow is Wednesday.
What is wrong with that statement?
"""
result = llm(prompt)
print(result)