from typing import List, Optional

import openai
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from api_key import Az_OpenAI_api_key, Az_OpenAI_endpoint, Az_Open_Deployment_name_gpt35


class CustomLLM(LLM):
        
    openai.api_key = Az_OpenAI_api_key
    openai.api_base = Az_OpenAI_endpoint
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15'

    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        deployment_name=Az_Open_Deployment_name_gpt35

        # Send a completion call to generate an answer
        response = openai.Completion.create(engine=deployment_name, prompt=prompt, max_tokens=256, temperature=0.4,n=1)
        text = response['choices'][0]['text']

        return text