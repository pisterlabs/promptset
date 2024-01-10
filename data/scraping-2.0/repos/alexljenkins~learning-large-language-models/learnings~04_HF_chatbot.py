"""
HuggingFace hosted models and templating example
can use any model found on huggingface as url minus the https://huggingface.com/ prefix
Useful Models:
Text completion:
- gpt2
- mistralai/Mistral-7B-v0.1

English to SQL: (not great - better to use specific data pipeline so model has context)
- mrm8488/t5-base-finetuned-wikiSQL 
"""

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

hub_llm = HuggingFaceHub(
    repo_id='mistralai/Mistral-7B-v0.1', 
    model_kwargs={'temperature': 0.3, 'max_length': 500}
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short (100 words or less) joke about {topic}."
)

hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
print(hub_chain.run("rick and morty"))
print(hub_chain.run("homework"))
print(hub_chain.run("Google"))
print(hub_chain.run("a baby"))
