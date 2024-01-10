#%%
from dotenv import load_dotenv
import os

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain import __version__ as langchain_version

# .env 파일 로드
load_dotenv()

print(f'langchain version : {langchain_version}')

#%%
#srcData/ 폴더에 있는 csv 파일을 모두읽어서 하나의 데이터로 만든다.
loader = CSVLoader(file_path="../res/jb14_base_v1.csv")
src_data = loader.load()
    
print( f'document size : {len(src_data)}' )

#%%
print(src_data[0].page_content)

#%%
openai = OpenAI()
# OpenAI 임베딩 모델 초기화
openai_embeddings = OpenAIEmbeddings()
# Chroma 인스턴스 생성
chroma_store = Chroma.from_documents(
    documents=src_data,
    embedding=openai_embeddings,
    persist_directory='./stores/chroma_store',
    )

print(chroma_store)

# %%
