import handle_env
import os
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain import OpenAI
import streamlit as st

handle_env.env_injection()

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)


def ask_job(job):
    messages = [
        SystemMessage(
            content="You are an skillful secretary in creating schedules"),
        AIMessage(content="What is your job? or What are you aiming for"),
        HumanMessage(content=job)
    ]
    return chat(messages).content


def make_schedule(text):
    messages = [
        SystemMessage(
            content="You are an skillful secretary in creating schedules"),
        HumanMessage(content=text)
    ]
    return chat(messages).content


print(ask_job(input()))
