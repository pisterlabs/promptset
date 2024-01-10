from dotenv import load_dotenv
import os
import pinecone
import openai

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# Load env variables from .env file
load_dotenv()

# Get env variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')

# Auth to pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

embeddings = OpenAIEmbeddings()

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(PINECONE_INDEX_NAME)

vectorstore = Pinecone(
    index, embeddings.embed_query, text_field
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = input("question on the doc : ")
print("querying...")

response = qa.run(query)

print(response)
