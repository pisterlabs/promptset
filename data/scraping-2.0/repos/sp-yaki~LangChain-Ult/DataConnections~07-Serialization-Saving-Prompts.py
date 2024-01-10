from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    load_prompt
)

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

chat = ChatOpenAI(temperature=0)

template = "Question: {question}\n\nAnswer: Let's think step by step."
prompt = PromptTemplate(template=template, input_variables=["question"])
prompt.save("prompt.json")

loaded_prompt = load_prompt('prompt.json')
print(loaded_prompt)