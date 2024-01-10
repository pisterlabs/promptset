from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from flask import request
import os

template = """I am Fujicon Boy, the mascot of Fujicon Priangan Perdana. I am an intelligent team with extensive knowledge in construction, IT, and multimedia. I have the ability to analyze data from internal systems built on Odoo platform, including both version 12 and the latest version.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "human_input"],
                        template=template)

chatgpt_chain = LLMChain(
  llm=OpenAI(temperature=0.8, openai_api_key=os.environ['GPT API Keys']),
  prompt=prompt,
  verbose=True,
  memory=ConversationBufferWindowMemory(k=2),
)


def get_langchain_response(user_input):
  user_input = user_input
  if user_input is None:
    return "Error: No user_input field provided. Please specify a user_input."

  response = chatgpt_chain.predict(human_input=user_input)

  print(f'Response : {response}')

  return response
