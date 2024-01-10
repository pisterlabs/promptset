#%%
import os
import time
from operator import itemgetter

from transformers import AutoModelForCausalLM,AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

from langchain.vectorstores import Chroma

# from langchain import VectorDBQA
from langchain.chains import RetrievalQA

from langchain.embeddings.openai import OpenAIEmbeddings

# from langchain import OpenAI

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import torch
import pinecone
from dotenv import load_dotenv

# .env 파일 로드 및 Pinecone 초기화
load_dotenv()

#%% Hugging Face 모델 및 파이프라인 로드
model_name = os.getenv("HUGGINGFACE_MODEL")
max_length = int(os.getenv("MAX_LENGTH"))
print(f'Start loading {model_name}')

start_tick = time.time()

tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForCausalLM.from_pretrained(  
# )

#8bit 모델 사용
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     load_in_8bit=True,
#     device_map="auto",
# )

hf_pipeline = pipeline(
    task="text-generation", 
    model=model_name, 
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    # load_in_8bit=True,
    max_length=max_length, 
    # temperature=0.1,
    do_sample=False,
    device_map="auto" # GPU 상황에 맞게 자동으로 설정
    # device_map="cuda:0"  # GPU 0사용 설정)
)

print(f'Load time: {time.time() - start_tick}')

# HuggingFacePipeline 인스턴스 생성
llm = HuggingFacePipeline(
    pipeline=hf_pipeline
    )

# Pinecone 벡터 스토어 설정
embeddings = OpenAIEmbeddings()

# 저장된 벡터 저장소 로드
vector_store = Chroma(
    persist_directory='./stores/chroma_store',
    embedding_function=embeddings
                      )
# retriever = vector_store.as_retriever()

#%% QA chain 생성 

from langchain.prompts import PromptTemplate

class DebugPromptTemplate(PromptTemplate):
    def generate(self, **kwargs):
        prompt = super().generate(**kwargs)
        print(f"Generated Prompt: {prompt}")  # 프롬프트 출력
        return prompt

# 디버그를 위한 한글 프롬프트 템플릿
prompt_template_korean = """
다음의 문맥을 사용하여 끝에 있는 질문에 대답하세요. 만약 답을 모르면 모른다고 답하고, 답을 지어내려고 하지 마세요.

{context}

질문: {question}
도움이 되는 답변
"""
DEBUG_PROMPT_KOREAN = DebugPromptTemplate(
    template=prompt_template_korean, input_variables=["context", "question"]
)

# qa_chain = VectorDBQA.from_chain_type(
#         llm=llm, 
#         chain_type="stuff", 
#         vectorstore=vector_store, 
#         return_source_documents=True,
#         k=5
    # )
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=llm,
    chain_type="stuff",
    retriever=retriever,
    # chain_type_kwargs=chain_type_kwargs,
    chain_type_kwargs={
        # "max_new_tokens": 100,
        "prompt": DEBUG_PROMPT_KOREAN
        },
    return_source_documents=True)

#%%
def get_answer(query):
    start_tick = time.time()
    # query = "덕진공원에 대해서 알려줘"
    result = qa_chain({"query": query})
    # print(result)
    print(f'Query time: {time.time() - start_tick}')

    print(f'question: {result["query"]}')
    print(f'answer: {result["result"]}')
    print(len(result['source_documents']))
    for doc in result['source_documents']:
        print(doc)
print("All Is Done and now test answering")
# %%
get_answer("덕진공원에 대해서 알려줘")
#%%
get_answer("정읍사에 대해서 알려줘")
#%%
get_answer("전주한옥마을에 대해서 알려줘")
#%%
get_answer("팔봉마을굿 축제가 열리는 장소는 어디야?")
# %%
