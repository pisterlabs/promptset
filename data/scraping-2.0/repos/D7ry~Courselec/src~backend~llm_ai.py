import os
import openai
from mongodb_utils import get_mongodb_client
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class LLM_Base:
    def __init__(self):
        raise NotImplementedError("LLM_Base is a static class and should not be instantiated.")
    
    def query(self, prompt:str) -> str:
        raise NotImplementedError("LLM_Base is a static class and should not be instantiated.")
    
class LLM_OpenAI(LLM_Base):
    def __init__(self):
        if os.environ.get("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        
    def query(self, prompt:str) -> str:
        res = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], temperature=0.2)
        result = res["choices"][0]["message"]["content"]
        return result