from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

with open("canary/src/prompts/chatbot.txt", "r") as file:
    chatbot_prompt = PromptTemplate.from_template(file.read())

with open("canary/src/prompts/integrity.txt", "r") as file:
    integrity_prompt = PromptTemplate.from_template(file.read())

model = OpenAI(temperature=0)

chatbot_chain = chatbot_prompt | model
integrity_chain = integrity_prompt | model
