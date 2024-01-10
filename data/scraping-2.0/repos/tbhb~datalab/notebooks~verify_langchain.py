# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# %load_ext dotenv
# %dotenv

# %%
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

chat = ChatOpenAI()
messages = [
    SystemMessage(content="Hello, I am a chatbot. I am here to help you with your questions."),
    HumanMessage(content="Who are you? What is your purpose?")
]
answer = chat(messages)
print(answer)
