import os
import streamlit as st

# langchain
from langchain.llms  import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

from models import models

gen_template = PromptTemplate(
    input_variables = ['instructions'],
    template = 'I want a game idea based on these {instructions}. Start with the name followed by the rest of the body of the game idea.'
)

def generate_result():
    if 'OPENAI_API_KEY' not in st.session_state or st.session_state["OPENAI_API_KEY"] == "":
        st.error("Please enter your OpenAI API key in the sidebar")
        return

    os.environ['OPENAI_API_KEY'] = st.session_state["OPENAI_API_KEY"]

    llm = OpenAI(temperature=1.2, max_tokens=2000)

    gen_memory = ConversationBufferMemory(input_key='instructions', memory_key='chat_history')
    game_idea_chain = LLMChain(llm=llm, prompt = gen_template, verbose=True,  output_key='high_concept', memory=gen_memory)

    game_idea_result = game_idea_chain.run(models.game_gen_models.getOptionsAsString())

    return game_idea_result