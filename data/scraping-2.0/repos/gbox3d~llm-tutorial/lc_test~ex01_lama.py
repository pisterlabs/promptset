#%%
import os
import time
from operator import itemgetter

from transformers import AutoModelForCausalLM,AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

import torch

from dotenv import load_dotenv

# .env 파일 로드 및 Pinecone 초기화
load_dotenv('../.env')

#%% Hugging Face 모델 및 파이프라인 로드
# model_name = os.getenv("HUGGINGFACE_MODEL")
# max_length = int(os.getenv("MAX_LENGTH"))
model_name = "beomi/llama-2-ko-7b"
print(f'Start loading {model_name}')

start_tick = time.time()

tokenizer = AutoTokenizer.from_pretrained(model_name)

hf_pipeline = pipeline(
    task="text-generation", 
    model=model_name, 
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    # load_in_8bit=True,
    # max_length=100, 
    max_new_tokens=32,
    # temperature=0.1,
    do_sample=False,
    # repeat_penalty=1.15,
    # device_map="auto" # GPU 상황에 맞게 자동으로 설정
    device_map="auto"  # GPU 0사용 설정)
)

print(f'Load time: {time.time() - start_tick}')

# HuggingFacePipeline 인스턴스 생성
llm = HuggingFacePipeline(
    pipeline=hf_pipeline
    )

print(f'Load time: {time.time() - start_tick}')
#%%
start_tick = time.time()
print(llm("지구의 위성은 무엇이 있을까요?"))
print(f'Elapsed time: {time.time() - start_tick}')
# %%
start_tick = time.time()
print(llm.invoke("대한민국의 수도는 어디일까요?"))
print(f'Elapsed time: {time.time() - start_tick}')

# %%
