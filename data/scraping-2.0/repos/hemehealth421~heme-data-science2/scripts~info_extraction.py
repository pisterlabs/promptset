import json


from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import time

import openai
import json
import os
import re
from dotenv import load_dotenv
load_dotenv()

anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
anthropic_model = os.environ.get("ANTHROPIC_MODEL")

openai_api_key = os.environ.get('OPENAI_API_KEY')
openai_model = os.environ.get('OPENAI_MODEL')



aws_default_region = os.getenv("AWS_DEFAULT_REGION")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

gcp_cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")



import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LLMExtractor:
    """
    This class is used to interact with different Large Language Models (LLMs)
    by OpenAI and Anthropic. It sends chat prompts to the LLM and extracts
    the response.
    """
    def __init__(self, llm_name):
        """
        Initialize the LLMExtractor with the name of the LLM.
        Raise a ValueError if the LLM name is neither 'OpenAI' nor 'Anthropic'.
        """
        self.llm_res = None
        self.content = None
        self.llm_name = llm_name
        self.llm_chat = None

        # Initializing LLM based on the given name
        if llm_name == "OpenAI":
            self.llm_chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name=openai_model)
        elif llm_name == "Anthropic":
            self.llm_chat = ChatAnthropic(temperature=0, anthropic_api_key=anthropic_api_key, model=anthropic_model)
        else:
            logger.exception("Invalid LLM name. Please choose either 'OpenAI' or 'Anthropic'.")
            raise ValueError("Invalid LLM name. Please choose either 'OpenAI' or 'Anthropic'.")

    def extract(self, ai_role, ai_job, doc_type, output_format, ai_restriction, input_text):
        """
        Extract the response from the LLM for the given chat prompt.
        """
        # Constructing the system and human message prompts
        template = f"{ai_role} {ai_job} {doc_type} {output_format} {ai_restriction}"
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = f"{input_text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        # Creating a chat prompt from the message prompts
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        # print(chat_prompt)

        # Sending the chat prompt to the LLM and storing the response
        self.llm_res = self.llm_chat(chat_prompt.format_prompt(ai_role=ai_role,
                                                               ai_job=ai_job,
                                                               doc_type=doc_type,
                                                               output_format=output_format,
                                                               ai_restriction=ai_restriction,
                                                               input_text=input_text).to_messages())
        # Logging the content of the LLM response
        self.content = self.llm_res.content
        logger.debug(f'LLM {self.llm_name} response content: {self.content}')
        return self

    def parse_json(self):
        """
        Parse the JSON content of the LLM response.
        Return an empty dictionary if the content is not valid JSON.
        """
        # Preprocessing the content to remove control characters and replace newline characters
        text = ''.join(c for c in self.content if c > '\u001f')
        text = text.replace('\n', '\\n')

        # Attempting to parse the content as JSON
        try:
            data = json.loads(text)
            return data
        except json.JSONDecodeError as e:
            # Logging any JSON decoding errors
            logger.exception(f'Invalid JSON: {e}')
            return {"Response":self.content}

