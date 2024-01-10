#!/usr/bin/env python3


# from langchain.llms import OpenAI
# # higher temperature = more creative
# # lower temperature = more predictable
# llm = OpenAI(openai_api_key=OPENAI_API_KEY,temperature=0.9)
# # get predictions
# print(llm.predict('The quick brown fox', max_tokens=5))

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
print(OPENAI_API_KEY)

chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY,temperature=0.0)

client_prompt = "What is a LLM?"

template_string = f""" Respond in a formal manner to the text\
that is delimited by triple backticks\
text: '''{client_prompt}'''
"""

# in the prompt template you cand find the original prompt and the prompt variables
prompt_template = ChatPromptTemplate.from_template(template_string)

# final prompt
customer_messages = prompt_template.format_messages(text=client_prompt)
print(customer_messages)

client_response = chat(customer_messages)
print(client_response)