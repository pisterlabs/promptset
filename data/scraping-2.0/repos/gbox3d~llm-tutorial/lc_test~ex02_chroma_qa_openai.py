#%%
import os
import time
from operator import itemgetter

from transformers import AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

from langchain.vectorstores import Chroma

from langchain import VectorDBQA
from langchain.chains import RetrievalQA

from langchain.embeddings.openai import OpenAIEmbeddings

from langchain import OpenAI

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

import pinecone
from dotenv import load_dotenv

# .env 파일 로드 및 Pinecone 초기화
load_dotenv()

#%%
llm = OpenAI()
# Pinecone 벡터 스토어 설정
embeddings = OpenAIEmbeddings()

# 저장된 벡터 저장소 로드
vector_store = Chroma(
    persist_directory='./stores/chroma_store',
    embedding_function=embeddings
                      )
# retriever = vector_store.as_retriever()
#%% QA 모델 설정
# qa_chain = VectorDBQA.from_chain_type(
#         llm=llm, 
#         chain_type="stuff", 
#         vectorstore=vector_store, 
#         return_source_documents=True,
#         k=5
#     )
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(llm=llm,
    chain_type="stuff",
    retriever=retriever,
    # chain_type_kwargs=chain_type_kwargs,
    return_source_documents=True)

#%%
def get_answer(query):
    start_tick = time.time()
    # query = "덕진공원에 대해서 알려줘"
    result = qa_chain({"query": query})
    print(result)
    print(f'Query time: {time.time() - start_tick}')

    print(f'question: {result}')
    print(f'answer: {result["result"]}')
    print(len(result['source_documents']))
    for doc in result['source_documents']:
        print(doc)
# %%
get_answer("덕진공원에 대해서 알려줘")

#%%
get_answer("정읍사에 대해서 알려줘")

#%%
get_answer("전주한옥마을에 대해서 알려줘")

#%%
get_answer("팔봉마을굿 축제에 대해서 알려줘")

# %%
