import os
import configparser
from langchain.chat_models import ChatOpenAI


def create_chat_model(config: configparser.SectionProxy) -> ChatOpenAI:
    """Factory method to create OpenAI Chat model"""
    return ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name=config["generation_model_name"],
        temperature=float(config["generation_temperature"]),
        top_p=float(config["generation_top_p"]),
        max_tokens=int(config["generation_max_tokens"]),
        streaming=True,
        request_timeout=120,
    )
