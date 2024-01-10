import os
import configparser

import asyncio

from typing import Dict
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import load_prompt
from langchain.callbacks import get_openai_callback

class UserResponseGenerator:
    def __init__(self, template_path="./templates/user_response_prompt.json", verbose=False):
        user_response_prompt = load_prompt(template_path)
        # 랭체인 모델 선언, 랭체인은 언어모델과 프롬프트로 구성됩니다.
        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)
        user_response_chain = LLMChain(
            llm=llm,
            prompt=user_response_prompt, 
            verbose=verbose,
            output_key="user_responses"
        )

        self.user_response_chain = user_response_chain

    async def arun(self, *args, **kwargs):
        with get_openai_callback() as cb:
            resp = await self.user_response_chain.arun(kwargs)
            # print("User Response Generator Callbacks")
            # print(f"Total Tokens: {cb.total_tokens}")
            # print(f"Prompt Tokens: {cb.prompt_tokens}")
            # print(f"Completion Tokens: {cb.completion_tokens}")
            # print(f"Total Cost (USD): ${cb.total_cost}")
            return resp
    
if __name__ == "__main__":
    user_response_generator = UserResponseGenerator()