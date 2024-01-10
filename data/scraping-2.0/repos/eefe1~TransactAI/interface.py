import streamlit as st
import pandas as pd
import json
from agent import query_agent, create_agent
from langchain.document_loaders.csv_loader import CSVLoader
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

st.header("💬 Transactions Anaylsis")

with st.sidebar:
        about = st.sidebar.expander("About This Project ")
        sections = [
            "#### This project designed to allow users to discuss their data in a more intuitive way. 📄",
            "#### It queries tabular data using a LLM (Large Language Model) to process and respond to queries based on this CSV file containing bank transactions of a small enterprise. 🌐",
        ]
        for section in sections:
            about.write(section)


def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)


def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    if "answer" in response_dict:
        st.write(response_dict["answer"])
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)


query = st.text_area("Insert your query")
loader = CSVLoader(file_path='./Transactions.csv')
data = loader.load()

if st.button("Submit Query", type="primary"):
     agent = create_agent(data)
     response = query_agent(agent=agent, query=query)
     decoded_response = decode_response(response)
     write_response(decoded_response)