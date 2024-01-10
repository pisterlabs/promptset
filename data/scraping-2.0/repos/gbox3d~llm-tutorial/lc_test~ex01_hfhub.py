#%%
import os
import time

from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
# .env 파일 로드 및 Pinecone 초기화
load_dotenv()

# %%
prompt = PromptTemplate.from_template("[INST]한국어로 답하시오[/INST][INST]What is the meaning of {word}[/INST]")

llm = HuggingFaceHub(
    # repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={
        "max_new_tokens": 250,
    },
)

chain = prompt | llm

print('chain ready')

#%%
answer = chain.invoke({"word": "감자"})
print(answer)

# %%
