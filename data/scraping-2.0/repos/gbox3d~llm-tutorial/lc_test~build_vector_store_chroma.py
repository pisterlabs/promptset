#%%
from dotenv import load_dotenv
import os

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever

# .env 파일 로드
load_dotenv()

#%%
#srcData/ 폴더에 있는 csv 파일을 모두읽어서 하나의 데이터로 만든다.

# srcData 폴더 내의 모든 CSV 파일 찾기
folder_path = 'srcData/'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 모든 데이터를 로드하여 하나의 리스트에 저장
all_data = []
for _file in csv_files:
    file_path = os.path.join(folder_path, _file)
    loader = CSVLoader(file_path=file_path)
    print(f'loading {file_path}')
    data = loader.load()
    all_data.extend(data)
    
print( f'document size : {len(all_data)}' )

#%%
print(all_data[0].page_content)

#%%
text_splitter = CharacterTextSplitter(chunk_size=100,chunk_overlap=20)
texts = text_splitter.split_documents(all_data)
print(len(texts))


#%%
openai = OpenAI()
# OpenAI 임베딩 모델 초기화
openai_embeddings = OpenAIEmbeddings()
# Chroma 인스턴스 생성
chroma_store = Chroma.from_documents(
    documents=texts,
    embedding=openai_embeddings,
    persist_directory='./stores/chroma_store',
    )

print(chroma_store)

# %%
