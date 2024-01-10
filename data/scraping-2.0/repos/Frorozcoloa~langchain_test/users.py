from typing import List, Dict
from langchain.chat_models import ChatOpenAI
import json

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from config import codegpt_api_key, code_gpt_agent_id, codegpt_api_base
from utils import text2json, save_csv


def get_tamplate() -> ChatPromptTemplate:
    """Returns a ChatPromptTemplate object with the following template"""

    template = "You are a helpful assistant. Your task is to analyze the users of an ecommerce."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = """
        Please, identify the main topics mentioned in these users profile. 

        Return a list of 3-5 topics. 
        Output is a JSON list with the following format
        [
            {{"topic_name": "<topic1>", "topic_description": "<topic_description1>"}}, 
            {{"topic_name": "<topic2>", "topic_description": "<topic_description2>"}},
            ...
        ]
        Users profile:
        {users_profile}
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    return chat_prompt


def get_model() -> ChatOpenAI:
    # Create a ChatOpenAI object with the retrieved API key, API base URL, and agent ID
    llm = ChatOpenAI(
        openai_api_key=codegpt_api_key,
        openai_api_base=codegpt_api_base,
        model=code_gpt_agent_id,
    )
    return llm


# Create a list of messages to send to the ChatOpenAI object


def run(text: str) -> List[Dict]:
    """Returns a list of topics, given a description of a product"""
    llm = get_model()
    chat_prompt = get_tamplate()
    messages = chat_prompt.format_prompt(users_profile=text)
    response = llm(messages.to_messages())
    list_desc = text2json(response.content)
    return list_desc


def example():
    text = "I love biking, hiking and walking. I like to get to know new towns, talk to people. I hate when plans don't happen, I'm very strict with times. I love to eat, I always like to go to good restaurants and try the food, I don't like to see many dishes and I hate the noise, I like the countryside and live there."
    list_desc = run(text)
    save_csv(list_desc, "users_description")


if __name__ == "__main__":
    example()
