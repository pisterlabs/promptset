import pandas as pd
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import HuggingFaceHub, HuggingFacePipeline, PromptTemplate, LLMChain
from langchain.agents import create_pandas_dataframe_agent

import streamlit as st
from streamlit_chat import message
import os

load_dotenv("../.env")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
API_KEY = os.getenv("OPENAI_API_KEY")

model_name = "MBZUAI/LaMini-Flan-T5-783M"

if "prompts" not in st.session_state:
    st.session_state.prompts = []
if "responses" not in st.session_state:
    st.session_state.responses = []


def getAgent(df):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=API_KEY)
    # llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=API_KEY)

    # llm = HuggingFaceHub(
    #     repo_id=model_name,
    #     model_kwargs={"temperature": 0, "max_length": 512},
    # )

    agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    return agent


def send_click(agent):
    if st.session_state.user != "":
        prompt = st.session_state.user
        response = agent.run(prompt)

        st.session_state.prompts.append(prompt)
        st.session_state.responses.append(response)


def main():
    st.title(":blue[Yeyu's Data Analysis Chatbot] ☕")
    uploaded_file = st.file_uploader("Choose a csv file", type="csv")

    if uploaded_file is not None:
        csv_data = uploaded_file.read()
        with open(uploaded_file.name, "wb") as f:
            f.write(csv_data)

        df = pd.read_csv(uploaded_file.name)
        st.dataframe(df.head(5))

        agent = getAgent(df)

        st.text_input("Ask Something:", key="user")
        # st.button("Send", on_click=send_click)
        if st.button("Send"):
            send_click(agent)

        if st.session_state.prompts:
            for i in range(len(st.session_state.responses) - 1, -1, -1):
                message(st.session_state.responses[i], key=str(i), seed="Milo")
                message(
                    st.session_state.prompts[i],
                    is_user=True,
                    key=str(i) + "_user",
                    seed=83,
                )


if __name__ == "__main__":
    main()
