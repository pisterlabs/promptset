import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
from typing import List

import json
import os
from dotenv import load_dotenv 

load_dotenv()

with open('prompts.json', 'r') as file:
    prompts = json.load(file)

# class EnglishOutput(BaseModel):
#     key_insights: List[str] = Field(description="List of insights")



class TranscriptEnglishTranslator:
    def __init__(self, api_key):
        self.SYSTEM_PROMPT = prompts['transcript_correction_system_prompt']
        self.prompt = self.SYSTEM_PROMPT
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(self.SYSTEM_PROMPT)
        openai.api_key = api_key
        llm_name = "gpt-3.5-turbo-16k"
        self.chat_model = ChatOpenAI(model_name=llm_name, temperature=0, max_tokens=4096)
        # self.parser = PydanticOutputParser(pydantic_object=EnglishOutput)

    def translate(self, response):
        human_template = prompts['transcript_correction_human_prompt']
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate(
            messages=[
                self.system_message_prompt,
                human_message_prompt,
            ], 
            input_variables=["user_input"],
            partial_variables={
                # "format_instructions": self.parser.get_format_instructions(),
            },
        )
        final_prompt = chat_prompt.format_prompt(user_input=response).to_messages()
        output = self.chat_model(final_prompt)
        return output.content
        # parsed = self.parser.parse(output.content)
        # return parsed