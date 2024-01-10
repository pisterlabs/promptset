"""Calls OpenAI API via LangChain"""

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from src.params import models
from src.prompt_lang import memory, prompt_template
from src.utils import calc_conversation_cost, calc_prompt_cost, num_tokens_from_string

load_dotenv()

# App interface, capture prompt
st.sidebar.title("ChatGPT API Interface")
model = st.sidebar.selectbox(label="Select a model", options=models)
new_conversation = st.sidebar.checkbox(label="Start new conversation?", value=True)
prompt = st.sidebar.text_area(
    label="Prompt", placeholder="Enter your prompt here...", height=250
)
submit = st.sidebar.button(label="Submit")

# Process submission
if submit:
    with st.spinner():
        llm_chain = LLMChain(
            llm=ChatOpenAI(model=model), prompt=prompt_template, memory=memory
        )

        msg = llm_chain.predict(question=prompt)

    input_tokens = num_tokens_from_string(message=prompt, model=model)
    output_tokens = num_tokens_from_string(message=msg, model=model)

    token_used, promt_cost = calc_prompt_cost(input_tokens, output_tokens, model)
    conversation_cost = calc_conversation_cost(
        prompt_cost=promt_cost, new_conversation=new_conversation
    )

    result = {}
    result["msg"] = msg
    result["token_used"] = token_used
    result["promt_cost"] = promt_cost
    result["conversation_cost"] = conversation_cost

    token_used = result.get("token_used")
    promt_cost = result.get("promt_cost")
    conversation_cost = result.get("conversation_cost")

    st.text(f"Tokens used: {token_used}")
    st.text(f"Prompt cost USD: {promt_cost}")
    st.text(f"Conversation cost USD: {conversation_cost}")

    st.markdown(result.get("msg"))
