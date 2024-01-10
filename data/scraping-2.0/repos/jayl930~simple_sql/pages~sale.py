import os
import streamlit as st
import dotenv
import argparse
from sql import execute_sql_query
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import load_prompt
from pathlib import Path
from PIL import Image
from langchain.chains import LLMChain

dotenv.load_dotenv()
assert os.environ.get("SALES_DATABASE_URL"), "DB_URL not found in .env file"
assert os.environ.get("AZURE_OPENAI_KEY"), "key not found in .env file"
assert os.environ.get("AZURE_OPENAI_ENDPOINT"), "endpoint not found in .env file"
assert os.getenv(
    "AZURE_DEPLOYMENT_NAME"
), "AZURE_DEPLOYMENT_NAME not found in .env file"

SALES_DB_URL = os.environ.get("SALES_DATABASE_URL")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-09-01-preview"
os.environ["OPENAI_API_BASE"] = AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_KEY"] = AZURE_OPENAI_KEY

llm = AzureChatOpenAI(
    deployment_name=AZURE_DEPLOYMENT_NAME,
)


def main():
    current_dir = Path(__file__)
    # root_dir = [p for p in current_dir.parents if p.parts[-1] == "SIMPLE_SQL"][0]
    # frontend
    st.set_page_config(page_title="BADM 554 SQL Assistant", page_icon="🌄")
    st.sidebar.success("Select a page above")

    tab_titles = ["Results", "Query", "ER Diagram"]

    st.title("Sales database assistant")
    prompt = st.text_input("Enter your query")
    tabs = st.tabs(tab_titles)
    with tabs[2]:
        image = Image.open("images/sales.png")
        st.image(image, caption="Entity Relationship")

    prompt_template = load_prompt("prompts/sales.yaml")
    final_prompt = prompt_template.format(input=prompt)

    sql_generation_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    if prompt:
        query_text = sql_generation_chain(prompt)
        output = execute_sql_query(query_text["text"], SALES_DB_URL)

        with tabs[0]:
            st.dataframe(output)
        with tabs[1]:
            st.write(query_text["text"])


if __name__ == "__main__":
    main()
