#%%
import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI

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

#%% qa
qa = VectorDBQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True
    )
query = "What is a vector DB? Give me a 15 word answer for a begginner"
result = qa({"query": query})
print(result)


# %%

