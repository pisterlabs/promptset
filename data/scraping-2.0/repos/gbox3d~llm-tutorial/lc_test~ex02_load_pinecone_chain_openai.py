#%%
import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

import pinecone
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT"),
)


#%% load document from index 
index_name = os.getenv("PINECONE_INDEX_NAME")
embeddings = OpenAIEmbeddings()

docsearch = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings
    )
retriever = docsearch.as_retriever()

llm = OpenAI()

# %%
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
_answer = chain.invoke("What is a vector DB? Give me a 15 word answer for a begginner")

print(_answer)

# %%
