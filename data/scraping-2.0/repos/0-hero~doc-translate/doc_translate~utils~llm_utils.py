import logging
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .prompts import translation_template


def get_provider(provider="openai"):
    '''
    Get the provider for LLM
    '''
    logging.info(f"Getting provider for LLM: {provider}")
    return ChatOpenAI(temperature=0.8, model_name="gpt-3.5-turbo")


def get_llm_chain(provider="openai"):
    logging.info(f"Getting LLM chain for provider: {provider}")
    llm = get_provider(provider)
    prompt_template = PromptTemplate(input_variables=["original_text", "source_lang", "target_lang"], template=translation_template)
    translation_chain = LLMChain(llm=llm, prompt=prompt_template)
    return translation_chain


def translate_text(original_text, source_lang, target_lang, provider="openai"):
    '''
    Translate text with LLM
    '''
    logging.info(f"Translating text from {source_lang} to {target_lang} using provider: {provider}")
    llm_chain = get_llm_chain(provider)
    translated_text = llm_chain.predict(original_text=original_text, source_lang=source_lang, target_lang=target_lang)
    logging.info(f"Translation successful.")
    return translated_text