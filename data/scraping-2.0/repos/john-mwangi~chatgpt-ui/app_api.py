"""Calls OpenAI API directly"""

import os
import pickle

import openai
import streamlit as st
from dotenv import load_dotenv

from src.params import CHATGPT_ROLE, models, msgs_path
from src.prompt_gpt import create_messages, load_conversation, prompt_gpt
from src.utils import calc_conversation_cost, calc_prompt_cost

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

# App interface, capture prompt
st.sidebar.title("ChatGPT API Interface")
model = st.sidebar.selectbox(label="Select a model", options=models)
new_conversation = st.sidebar.checkbox(label="Start new conversation?", value=True)
prompt = st.sidebar.text_area(
    label="Prompt", placeholder="Enter your prompt here...", height=250
)
submit = st.sidebar.button(label="Submit")

# Load conversation history
conversation_history = load_conversation(msgs_path, new_conversation)

# Create messages
messages = create_messages(prompt, role=CHATGPT_ROLE, messages=conversation_history)

# Process submission
if submit:
    with st.spinner():
        result = prompt_gpt(model=model, messages=messages)
    token_used, promt_cost = calc_prompt_cost(
        input_tokens=result.get("input_tokens"),
        output_tokens=result.get("output_tokens"),
        model=model,
    )

    conversation_cost = calc_conversation_cost(
        prompt_cost=promt_cost, new_conversation=new_conversation
    )

    st.text(f"Tokens used: {token_used}")
    st.text(f"Prompt cost USD: {promt_cost}")
    st.text(f"Conversation cost USD: {conversation_cost}")

    st.markdown(result.get("msg"))

    if not msgs_path.exists():
        msgs_path.parent.mkdir()

    with open(msgs_path, mode="wb") as f:
        pickle.dump(result["msgs"], file=f)

# TODO:save different conversations in different files
