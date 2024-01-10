from config import get_OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

import streamlit as st

# Set the API key for OpenAI
try:
    OpenAI.api_key = get_OpenAI()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")


# LLMs
llm_model = "gpt-3.5-turbo-1106"
open_ai = OpenAI(temperature = 0.0)


template = """
As a childrens book author, write a simple and short (90 words) story lullaby based on the location
{location}
and the main character
{name}

STORY:
"""

prompt = PromptTemplate(
    input_variables=["location", "name"],
    template=template,
)

chain_story = LLMChain(llm=open_ai, prompt=prompt, output_key="story", verbose=True)
story = chain_story({"location": "the forest", "name": "Bobby"})


# SequentialChain
translation_template = """
Translate the {story} to {language}.

Make sure the translation is simple and fun to read for children.

TRANSLATION:
"""

prompt_translation = PromptTemplate(
    input_variables=["story", "language"],
    template=translation_template,
)

chain_translation = LLMChain(llm=open_ai, prompt=prompt_translation, output_key="translated")

overall_chain = SequentialChain(
    chains=[chain_story, chain_translation],
    input_variables=["location", "name", "language"],
    output_variables=["story", "translated"])

response = overall_chain({"location": "the forest", "name": "Bobby", "language": "German"})


st.title("Story Chain:")
# st.write(story['text'])

st.write("## Story Chain with SequentialChain")
st.write(f"### Story: {response['story']} \n\n")
st.write(f"### Translation:\n\n {response['translated']} \n\n")

         