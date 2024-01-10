import os
import pinecone
from dotenv import load_dotenv
import openai
from langchain.chat_models import ChatOpenAI

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

pinecone.init(
    environment=os.environ.get("PINECONE_ENVIRONMENT"),
    api_key=os.environ.get("PINECONE_API_KEY")
)
print("您的向量数据库中有如下index:",pinecone.list_indexes())

index = pinecone.Index(os.environ.get("PINECONE_INDEX_NAME"))

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model_name='gpt-3.5-turbo',
    temperature=0.0
)