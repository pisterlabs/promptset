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

hf_pipeline = pipeline(
    task="text-generation", 
    model=model_name, 
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    # load_in_8bit=True,
    max_length=max_length, 
    # temperature=0.1,
    # do_sample=False,
    device_map="auto" # GPU 상황에 맞게 자동으로 설정
    # device_map="cuda:0"  # GPU 0사용 설정)
)

print(f'Load time: {time.time() - start_tick}')

#%% 저장된 벡터 저장소 로드
embeddings = OpenAIEmbeddings()
vector_store = Chroma(
    persist_directory='./stores/chroma_store',
    embedding_function=embeddings
                      )
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

#%%
# question = '팔봉마을굿축제는 언제 열리나요?'
question = '덕진공원에 대해서 소개해줘'
# question = '정읍사에 대해 알려줘.'

start_tick = time.time()

docs = retriever.invoke(question)
for doc in docs:
    print(doc.metadata)
    print(doc.page_content)

print(f'Query time: {time.time() - start_tick}')
#%%
start_tick = time.time()
prompt_template = """다음의 문맥들만을 참고하여 끝에 있는 질문에 대답하세요. 만약 답을 모르면 모른다고 답하고, 답을 지어내려고 하지 마세요.

{context}

질문: {question}
"""

# docs 리스트 내용을 context 변수로 변환
context = "\n\n".join([doc.page_content for doc in docs])

#context 크기가 1500자를 넘어가면 문맥을 잘라냄
if len(context) > 1500:
    context = context[:1500]

# 프롬프트에 문맥과 질문을 삽입
formatted_prompt = prompt_template.format(context=context, question=question)

# print(formatted_prompt)

answer = hf_pipeline(
        formatted_prompt,
        do_sample=False, # False로 설정하면 창의적이지 않는 답변을 생성
        # penalty_alpha = 0.2,
        max_new_tokens=60,
        # temperature=0.1,
        # top_p=0.9,
        return_full_text=False,
        # early_stopping=True,
        eos_token_id=2,
        max_time=10
    )
print(answer)


print(f'Query time: {time.time() - start_tick}')
# %%



