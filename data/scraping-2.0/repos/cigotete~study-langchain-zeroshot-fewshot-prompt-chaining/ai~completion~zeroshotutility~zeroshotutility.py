import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json

class ZeroShotUtility:
    def __init__(self, template, temperature=0):
        self.template = template
        self.temperature = temperature

    def __str__(self):
        return f"{self.template}"

    def print_travel_modes(self, question):
        prompt_template = ChatPromptTemplate.from_template(self.template)
        message = prompt_template.format_messages(question=question)
        llm = ChatOpenAI(temperature=self.temperature, 
                         openai_api_key=os.getenv("OPENAI_KEY"))
        response = llm(message)
        print(response.content)
        print("-----------------------------------------------------------------")