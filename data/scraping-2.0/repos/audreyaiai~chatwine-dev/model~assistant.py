import os
import configparser

import asyncio

from typing import Dict
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import load_prompt
from langchain.callbacks import get_openai_callback

class Assistant:
    def __init__(self, template_path="./templates/assistant_prompt_template.json", verbose=False):
        stage_analyzer_inception_prompt = load_prompt(template_path)
        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.0)
        stage_analyzer_chain = LLMChain(
            llm=llm,
            prompt=stage_analyzer_inception_prompt, 
            verbose=verbose, 
            output_key="stage_number")

        self.stage_analyzer_chain = stage_analyzer_chain

    async def arun(self, *args, **kwargs):
        with get_openai_callback() as cb:
            resp = await self.stage_analyzer_chain.arun(kwargs)
            # print("Assistant Callbacks")
            # print(f"Total Tokens: {cb.total_tokens}")
            # print(f"Prompt Tokens: {cb.prompt_tokens}")
            # print(f"Completion Tokens: {cb.completion_tokens}")
            # print(f"Total Cost (USD): ${cb.total_cost}")
            return resp
        
    
if __name__ == "__main__":
    assistant = Assistant()