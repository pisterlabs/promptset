
import os
import time
import requests
import ssl
from pydantic import Extra, Field
from typing import Any, List, Mapping, Optional
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from fastapi import FastAPI
from langserve import add_routes
from langchain_community.document_loaders import PyPDFLoader
from langchain.llms.base import LLM
import streamlit as st
from streamlit_chat import message
from langchain.callbacks.manager import CallbackManagerForLLMRun
import pandas as pd

# HCX 토큰 계산기 API 호출
from hcx_token_cal import token_completion_executor


##################################################################################
# HCX API 키
API_KEY='YOUR API KEY !!!!!!!!!!!!!!!!!!!!!!!='
API_KEY_PRIMARY_VAL='YOUR API KEY PRIMARY VAL !!!!!!!!!!!!!!!!!!!!!!!'
REQUEST_ID='YOUR REQUEST ID !!!!!!!!!!!!!!!!!!!!!!!'

# (개인) 유료 API 키!!!!!!!!
os.environ['OPENAI_API_KEY'] = "YOUR OPENAI API KEY !!!!!!!!!!!!!!!!!!!!!!!"

# 임베딩 벡터 DB 저장 & 호출
db_save_path = "YOUR DB SAVE PATH !!!!!!!!!!!!!!!!!!!!!!!" 
##################################################################################


template = """You are a Cloud (MSP) Engineer or Cloud Sales administrator, or Cloud Solution Architect. about user question, answering specifically in korean.
    Use the following pieces of context to answer the question at the end.
    You mast answer after understanding previous conversation.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Respond don't know to questions not related to Cloud Computing.
    Use three sentences maximum and keep the answer as concise as possible.
    context for answer: {context}
    chat conversation: {chat_history}
    question: {question}
    answer: """
    
    
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "chat_history", "question"],template=template)

scenario_1 =  PyPDFLoader('contents/cloud_computing.pdf')
s1_documents = scenario_1.load_and_split()


text_splitter = CharacterTextSplitter(        
separator = "\n",
chunk_size = 1000,
chunk_overlap  = 200,
length_function = len,
)


# OpenAI VS HuggingFace
embeddings = OpenAIEmbeddings()

def offline_faiss_save(*pdf_docs):

    total_docs = []
    for pdf_doc in pdf_docs:
        doc = text_splitter.split_documents(pdf_doc)
        total_docs = total_docs + doc

    # Convert list of dictionaries to strings
    total_content = [str(item) for item in total_docs]

    docsearch = FAISS.from_texts(total_content, embeddings)

    docsearch.embedding_function
    docsearch.save_local(os.path.join(db_save_path, "cloud_bot_20240108_faiss_db"))


# start = time.time()
# total_content = offline_faiss_save(s1_documents)
# end = time.time()
# '''임베딩 완료 시간: 1.62 (초)'''
# print('임베딩 완료 시간: %.2f (초)' %(end-start))

# new_docsearch = FAISS.load_local(os.path.join(db_save_path, 'cloud_bot_20240108_faiss_db'), embeddings)


# retriever = new_docsearch.as_retriever(search_type="similarity", search_kwargs={"k":2,
#                                                                         "score_threshold": 0.7}
#                                                                         )

def offline_chroma_save(*pdf_docs):

    total_docs = []
    for pdf_doc in pdf_docs:
        doc = text_splitter.split_documents(pdf_doc)
        total_docs = total_docs + doc
        
    # Convert list of dictionaries to strings
    # total_content = [str(item) for item in total_docs]    
    
    vectorstore = Chroma.from_documents(
        documents=total_docs, 
        embedding=embeddings,
        persist_directory=os.path.join(db_save_path, "cloud_bot_20240108_chroma_db")
        )
    vectorstore.persist()

# start = time.time()
# total_content = offline_chroma_save(s1_documents)
# end = time.time()
# '''임베딩 완료 시간: 1.31 (초)'''
# print('임베딩 완료 시간: %.2f (초)' %(end-start))


new_docsearch = Chroma(persist_directory=os.path.join(db_save_path, "cloud_bot_20240108_chroma_db"),
                        embedding_function=embeddings)
retriever = new_docsearch.as_retriever(
                                        search_type="mmr",
                                        search_kwargs={'k': 2, 'fetch_k': 10}
                                        # search_type="similarity_score_threshold",
                                        # search_kwargs={'score_threshold': 0.8}
                                       )

# Chroma 의 저장된 벡터 DB에 대한 현황 파악 !!!!!!!!!!
# query = '클라우드 컴퓨팅의 정의는 무엇인가요?'
# docs = new_docsearch.similarity_search(query, k=3)
# print('########################################')
# print(len(docs))
# print(docs[0].page_content)


##################################################################################
# offline_faiss_save 또는 offline_chroma_save 와 같이 pinecone 함수 생성 !!!!!!!!!!!!!!!!!



try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class CompletionExecutor(LLM):
    llm_url = 'https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-002'
    api_key: str = Field(...)
    api_key_primary_val: str = Field(...)
    request_id: str = Field(...)
   
    class Config:
        extra = Extra.forbid
 
    def __init__(self, api_key, api_key_primary_val, request_id):
        super().__init__()
        self.api_key = api_key
        self.api_key_primary_val = api_key_primary_val
        self.request_id = request_id
 
    @property
    def _llm_type(self) -> str:
        return "HyperClovaX"
 
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
 
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key,
            'X-NCP-APIGW-API-KEY': self.api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id,
            'Content-Type': 'application/json; charset=utf-8'
        }

        # preset_text = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
        preset_text = [{"role": "system", "content": prompt}, {"role": "user", "content": ""}]

        payload = {
            'messages': preset_text,
            'topP': 0.8,
            'topK': 0,
            'maxTokens': 256,
            'temperature': 0.1,
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': True
        }

    
        response = requests.post(self.llm_url, json=payload, headers=headers, verify=False)
        response.raise_for_status()
        return response.json()['result']['message']['content']
 
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"llmUrl": self.llm_url}



hcx_llm = CompletionExecutor(api_key = API_KEY, api_key_primary_val=API_KEY_PRIMARY_VAL, request_id=REQUEST_ID)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, input_key = "question", output_key='answer')
retrieval_qa_chain = ConversationalRetrievalChain.from_llm(llm = hcx_llm,
                                retriever = retriever, 
                                memory = memory,
                                return_source_documents = True,
                                combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
                                )

# class SimpleJsonOutputParser:
#     def __call__(self, result):
#         return result["answer"]

# retrieval_qa_chain_pipe = retrieval_qa_chain | SimpleJsonOutputParser()

# app = FastAPI(title="Cloud Bot",
#         version="1.0",
#         description="A simple API server using LangChain's Runnable interfaces")

# # Add the LangServe routes to the FastAPI app
# # 3. Adding chain route
# add_routes(
#     app,
#     retrieval_qa_chain_pipe, 
#     path="/retrieval_qa_chain",
#     )


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)


################### LangSerce ###################
# /docs: langserve API 문서
# uvicorn langserve_hcx_cloud_bot:app --host 223.130.140.95 --port 8080
# Playground UI
# http://0.0.0.0:8000/retrieval_qa_chain/playground/



st.title("Cloud 관련 무물보~!")


if 'generated' not in st.session_state:
    st.session_state['generated'] = []
 
if 'past' not in st.session_state:
    st.session_state['past'] = []
 
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('질문', '', key='ques')

    submitted = st.form_submit_button('답변')
 
    if submitted and user_input:
        with st.spinner("Waiting for HyperCLOVA..."): 
            response_text_json = retrieval_qa_chain({'question': user_input})
            response_text = response_text_json['answer']
            
            # 참조 문서 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            total_content = pd.DataFrame(columns=['순번', '참조 문서'])
            token_total_content = ''
            for i in range(len(response_text_json['source_documents'])):
                context = response_text_json['source_documents'][i].page_content
                # print('==================================================')
                # print('\n%d번 째 참조 문서: %s' %(i+1, context))
                # total_content += '=================================================='
                # total_content += '\n%d번 째 참조 문서: %s' %(i+1, context)
                total_content.loc[i] = [i+1, context]
                token_total_content += context

            single_turn_text_json = {
            "messages": [
            {
                "role": "system",
                "content": template
            },
            {
                "role": "system",
                "content": token_total_content
            },
            {
                "role": "user",
                "content": user_input
            },
            {
                "role": "assistant",
                "content": response_text
            }
            ]
            }
            
            single_turn_text_token = token_completion_executor.execute(single_turn_text_json)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(single_turn_text_json)
            # single_turn_token_count = single_turn_text_token[0]['count'] + single_turn_text_token[1]['count'] + single_turn_text_token[2]['count'] + single_turn_text_token[3]['count']
            single_turn_token_count = sum(token['count'] for token in single_turn_text_token[:4])

            st.session_state.past.append({'question': user_input})
            # st.session_state.generated.append({'generated': response_text, 'token_count': single_turn_token_count, 'source_documents': total_content})
            st.session_state.generated.append({'generated': response_text, 'token_count': single_turn_token_count})
            
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated']) - 1, -1, -1):
                user_input = st.session_state['past'][i]
                response = st.session_state['generated'][i]

                message(f"질문: {user_input['question']}", is_user=True, key=str(i) + '_question')
                message(f"답변: {response['generated']}", is_user=False, key=str(i) + '_generated')
                # message(f"참조 문서: {response['source_documents']}", is_user=False, key=str(i) + '_source_documents')
                message(f"총 토큰 수: {response['token_count']}", is_user=False, key=str(i) + '_token_count')
                st.table(data = total_content)


################### Streamlit ###################
# streamlit run cloud_llm_bot.py
