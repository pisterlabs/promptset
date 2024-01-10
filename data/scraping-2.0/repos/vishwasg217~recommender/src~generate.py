import sys
from pathlib import Path

# Get the current script's directory
script_dir = Path(__file__).resolve().parent

# Get the project's root directory by going up one level
project_root = script_dir.parent

# Add the project's root directory to sys.path
sys.path.append(str(project_root))


import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage

from dotenv import dotenv_values

from prompts.prompt import get_prompt
from src.output import Campaign
from src.utils import get_brand_tone_and_voice, format_json_to_multiline_string

# config = dotenv_values(".env")

# OPENAI_API_KEY = config["OPENAI_API_KEY"]

def generate_emails(inputs, OPENAI_API_KEY):
    try:
        tone_and_voice = get_brand_tone_and_voice(inputs.tone_and_voice.tone_and_voice_from_website, OPENAI_API_KEY)
        inputs.tone_and_voice.tone_and_voice_from_website = tone_and_voice 
    except AttributeError:
        print("error")  

    string_inputs = format_json_to_multiline_string(inputs)

    with open("prompts/p2.prompt", "r") as f:
        template = f.read()

    parser = PydanticOutputParser(pydantic_object=Campaign)

    prompt = PromptTemplate(
        template=template,
        input_variables=["inputs"],
        partial_variables={"output_format": parser.get_format_instructions()}
    )

    llm = OpenAI(openai_api_key=OPENAI_API_KEY, max_tokens=3000, presence_penalty=1)
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, max_tokens=3000)

    formatted_input = prompt.format(inputs=string_inputs)
    # print("-"*30)
    # print("Formatted Input:")
    # print(formatted_input)
    # print("-"*30)

    response = llm(formatted_input)
    parsed_output = parser.parse(response)
    return parsed_output
