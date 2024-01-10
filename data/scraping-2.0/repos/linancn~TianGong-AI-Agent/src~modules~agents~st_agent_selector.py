import ast
import os

import streamlit as st
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage

from . import (
    chat_conversational_react_description_agent as chat_conversational_react_description_agent,
)
from . import (
    st_zero_shot_react_description_agent as st_zero_shot_react_description_agent,
)

llm_model = st.secrets["llm_model"]
langchain_verbose = st.secrets["langchain_verbose"]


def extract_module_docstring(filename):
    """Use AST to extract the module-level docstring from a Python file."""
    with open(filename, "r", encoding="utf-8") as f:
        # Parse the file content into an AST node
        node = ast.parse(f.read(), filename=filename)
        # Get the docstring from the AST node
        return ast.get_docstring(node)


# List all files in the current directory
files = os.listdir("src/modules/agents/")

# Filter out Python files and exclude the script itself
py_files = [
    f
    for f in files
    if os.path.isfile(os.path.join("src/modules/agents/", f))
    and f.endswith(".py")
    and f != "__init__.py"
    and f != "st_agent_selector.py"
    and not f.startswith("fastapi_")
]

agent_mapping = {}
for py_file in py_files:
    docstring = extract_module_docstring(os.path.join("src/modules/agents/", py_file))
    if docstring:
        # Store the docstring with the filename as the key
        agent_type = py_file.split(".")[0]
        agent_mapping[agent_type] = docstring

agent_types = list(agent_mapping.keys())


def agent_selector_func_calling_chain():
    func_calling_json_schema = {
        "title": "select_the_most_suitable_agent_to_complete_task",
        "description": "Based on the given prompt, identify the best-suited agent to accomplish the task.",
        "type": "object",
        "properties": {
            "agent_type": {
                "title": "AgentType",
                "description": "The type of agent to be used for the task.",
                "type": "string",
                "enum": agent_types,
            }
        },
        "required": ["agent_type"],
    }

    prompt_func_calling_msgs = [
        SystemMessage(
            content=f"""You are a world class algorithm for accurately identifying the type of agent to be used for solving the prompt, strictly follow the mapping: {str(agent_mapping)}. Make sure to answer in the correct structured format"""
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]

    prompt_func_calling = ChatPromptTemplate(messages=prompt_func_calling_msgs)

    llm_func_calling = ChatOpenAI(model_name=llm_model, temperature=0, streaming=False)

    func_calling_chain = create_structured_output_chain(
        output_schema=func_calling_json_schema,
        llm=llm_func_calling,
        prompt=prompt_func_calling,
        verbose=langchain_verbose,
    )

    return func_calling_chain


def main_agent(query):
    response = agent_selector_func_calling_chain().run(query)
    agent_type = response.get("agent_type")
    if agent_type == "st_zero_shot_react_description_agent":
        prompt = f"""Provide a clear, well-organized respond to the following question of "{query}" in its original language. Give corresponding detailed sources with urls, if possible."""
        return (st_zero_shot_react_description_agent.main_agent, prompt)
    elif agent_type == "chat_conversational_react_description_agent":
        prompt = f"""Provide a clear, well-organized respond to the following question of "{query}" in its original language. Give corresponding detailed sources with urls, if possible."""
        return (
            chat_conversational_react_description_agent.main_agent,
            prompt,
        )
