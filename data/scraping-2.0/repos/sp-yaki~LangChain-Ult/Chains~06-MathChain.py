from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

model = ChatOpenAI()
result = model([HumanMessage(content="What is 17 raised to the power of 11?")])
print(result.content)

result = model([HumanMessage(content="Give me the Python formula that represents: What is 17 raised to the power of 11? Only reply with the formula, nothing else!")])
result.content
print(result.content)
print(eval(result.content))

from langchain.chains import LLMMathChain
llm_math_model = LLMMathChain.from_llm(model)
print(llm_math_model("What is 17 raised to the power of 11?"))