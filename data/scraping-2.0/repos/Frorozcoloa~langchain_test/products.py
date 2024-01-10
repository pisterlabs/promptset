from typing import List, Dict
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os
import json


from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from config import codegpt_api_key, code_gpt_agent_id, codegpt_api_base
from utils import text2json, save_csv


def get_tamplate() -> ChatPromptTemplate:
    """Returns a ChatPromptTemplate object with the following template:"""
    template = "You are a helpful assistant. Your task is to analyze the products of an e-commerce."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = """
        Identify the primary subject discussed in the product description. Categorize the product and provide a portrayal of the target audience. 

        Return a list of 3-6 classes. 
        Output is a JSON list with the following format
        [
            {{"type_classes": "<class1>", "class_description": "<class_description1>"}}, 
            {{"type_classes": "<class3>", "class_description": "<class_description2>"}},
            ...
        ]
        description product:
        {description_product}
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    return chat_prompt


def get_model() -> ChatOpenAI:
    """Returns a ChatOpenAI"""
    llm = ChatOpenAI(
        openai_api_key=codegpt_api_key,
        openai_api_base=codegpt_api_base,
        model=code_gpt_agent_id,
    )
    return llm


def run(customer_reviews: str) -> List[Dict]:
    """Returns a list of topics, given a description of a product"""
    llm = get_model()
    chat_prompt = get_tamplate()
    message = chat_prompt.format_prompt(description_product=customer_reviews)
    response = llm(message.to_messages())
    values = text2json(response.content)
    return values


def example():
    """Example of use of the function get_topics"""

    description_product = """
                Small 10-liter hiking backpack nh100 quechua black, BENEFITS
                Carrying comfort, COMFORT CARRYING COMFORT Spalder and padded straps 
                1 main compartment with double zipper 
                VOLUME
                Volume: 10 liters | Weight: 145 g | Dimensions: 39 x 21 x 12 cm.friction resistance
                FRICTION RESISTANCE
                Durable, abrasion-resistant materials and joints. 10-year warranty. Warranty 10 years.Ventilation
                VENTILATION
                Simple to use backrest
                EASE OF USE
                Easy access to the external pocket by placing the backpack in a horizontal position while hiking.
    """
    topics = run(description_product)
    save_csv(topics, "products_classes")


if __name__ == "__main__":
    example()
