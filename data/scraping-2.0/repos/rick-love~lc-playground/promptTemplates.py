from OpenAI_Training.config import get_OpenAI, get_PineCone
from openai import OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


import streamlit as st

# Set the API key for OpenAI
try:
    OpenAI.api_key = get_OpenAI()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")


# LLMs
# llm = OpenAI(temperature=0.9, model_name="gpt-4-1106-preview")
chat_model = ChatOpenAI(temperature=.7)

#Templates
human_input = "Hello, how are you?"
template_string = """

Translates from English to German in a nice tone.
{human_input}

"""

prompt_template = ChatPromptTemplate.from_template(template_string)
translation = prompt_template.format_messages(human_input=human_input)

response = chat_model(translation)
print(response)

############
# Show to the screen
# App Framework
st.title('Boiler Plate:')
st.write(response)