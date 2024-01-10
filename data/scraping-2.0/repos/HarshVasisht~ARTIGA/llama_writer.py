from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
import streamlit as st 
import os
from docx import Document
from docx.shared import Inches
import io
from PIL import Image
import requests

#Loading Quantized Llama model

def load_llm(max_tokens, Prompt_template):
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_new_tokens = max_tokens,
        temperature = 0.7
    )
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    print(llm_chain)
    return llm

user_input = input("Enter")

prompt_template  = f"""You are a digital marketing and SEO expert and your task is to generate articles for a given topic. So, write an article on {user_input} under 900 words.
            Stick to the topic given by the user and maintain a professional and creative tone. You can use quotes to go with the article."""
llm_call = load_llm(max_tokens = 800, Prompt_template= prompt_template)
print(llm_call)
result = llm_call(user_input)
print(result)