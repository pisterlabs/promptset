# app/langchain_init.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import hub
import os

rag_prompt = hub.pull("rlm/rag-prompt")

# Initialize Langchain components

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

openai_embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,
                 openai_api_key=os.getenv('OPENAI_API_KEY'))


def format_docs(document):
    return "\n\n".join(doc.page_content for doc in document)
