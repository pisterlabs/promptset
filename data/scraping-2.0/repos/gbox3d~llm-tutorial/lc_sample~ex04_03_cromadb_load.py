#%%
from dotenv import load_dotenv
import os

from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings


from langchain import __version__ as langchain_version

# .env 파일 로드
load_dotenv()

print(f'langchain version : {langchain_version}')

#%%
openai_embeddings = OpenAIEmbeddings()
# 저장된 벡터 저장소 로드
chroma_store = Chroma(
    persist_directory='./stores/chroma_store',
    embedding_function=openai_embeddings
                      )
_retriever = chroma_store.as_retriever()

#%%
contents = _retriever.invoke('임실')

for content in contents:
    print(content)

# %%
