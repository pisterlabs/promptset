from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from functools import lru_cache
from .config import Settings
from fastapi import Depends
from oliveea.core.crafter.crafter import Crafter
from oliveea.core.prompts import CRAFTER_TEMPLATE

@lru_cache()
def get_settings():
    return Settings()

def get_llm(settings: Settings = Depends(get_settings)) -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        model_name=settings.model_name,
    )

def get_crafter(llm: LLMChain = Depends(get_llm)) -> Crafter:
    return Crafter(
        llm=llm,
        prompt=PromptTemplate(
            template=CRAFTER_TEMPLATE,
            input_variables=["input", "tone"]
        )
    )