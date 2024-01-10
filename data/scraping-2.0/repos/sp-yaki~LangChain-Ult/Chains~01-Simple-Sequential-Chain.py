from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

llm = ChatOpenAI()

template = "Give me a simple bullet point outline for a blog post on {topic}"
first_prompt = ChatPromptTemplate.from_template(template)
chain_one = LLMChain(llm=llm,prompt=first_prompt)

template = "Write a blog post using this outline: {outline}"
second_prompt = ChatPromptTemplate.from_template(template)
chain_two = LLMChain(llm=llm,prompt=second_prompt)

full_chain = SimpleSequentialChain(chains=[chain_one,chain_two],
                                  verbose=True)

result = full_chain.run("Data Science")
print(result)