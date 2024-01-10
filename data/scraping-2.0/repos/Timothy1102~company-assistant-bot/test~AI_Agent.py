import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities import TextRequestsWrapper
from dotenv import load_dotenv
load_dotenv()

with st.sidebar:
    st.title('💬 LLM chatbot 📄')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    "[View source code](https://github.com/Timothy1102/company-assistant-bot)"

def init_agent():
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
    search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)
    requests = TextRequestsWrapper()
    toolkit = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to search google to answer questions about current events"
        ),
        Tool(
            name = "Requests",
            func=requests.get,
            description="Useful for when you to make a request to a URL"
        ),
    ]

    llm = OpenAI(temperature=0.4)
    agent = initialize_agent(toolkit, llm, agent="zero-shot-react-description", verbose=True, return_intermediate_steps=True)
    return agent

def main():
    st.header("AI Agent 🤖")

    question = st.text_input(
        "Ask your question:",
        placeholder="Summarize this page for me"
    )

    if question:
        agent = init_agent()
        response = agent({"input": question})
        st.write(response)
 
if __name__ == '__main__':
    main()
