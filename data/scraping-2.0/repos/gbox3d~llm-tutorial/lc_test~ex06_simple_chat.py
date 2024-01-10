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

hf_pipeline = pipeline(
    task="text-generation", 
    model=model_name, 
    tokenizer=model_name,
    device_map="auto" # GPU 상황에 맞게 자동으로 설정
    # device_map="cuda:0"  # GPU 0사용 설정)
)

print(f'Load time: {time.time() - start_tick}')

#%%
start_tick = time.time()
answer = hf_pipeline(
        """
다음의 문맥을 사용하여 끝에 있는 질문에 대답하세요. 만약 답을 모르면 모른다고 답하고, 답을 지어내려고 하지 마세요.

정읍사(井邑詞)는 삼국 시대의 고대가요로, 현존하는 유일한 백제 문학이다.
망부가(望夫歌)의 한 유형으로 남편의 무사 귀환을 기원하는 내용을 담고 있다.

질문: 정읍사에 대해 알려줘.
도움이 되는 답변
        """,
        do_sample=True,
        max_new_tokens=70,
        # temperature=0.1,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2,
    )
print(answer)

print(f'Load time: {time.time() - start_tick}')

# %%
