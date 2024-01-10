## Creating a LLM using LangChain + HuggingFace
import os
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

hf_key = os.environ.get("SQL_MODEL_KEY")


hub_llm = HuggingFaceHub(
    repo_id="succinctly/text2image-prompt-generator", 
    model_kwargs = {'temperature' : 0.7, 'max_length':250} ,
    huggingfacehub_api_token=hf_key)

prompt = PromptTemplate(
    input_variables= ['topic'],
    template= "Genearate a image \n `{topic}`."
)
chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
print(chain.run("cyber girl with painting "))