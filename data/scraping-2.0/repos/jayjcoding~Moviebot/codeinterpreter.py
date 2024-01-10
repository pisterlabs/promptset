import streamlit as st
from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os


def main():
    load_dotenv()
    st.set_page_config(page_title="Hello, I am a Movie Chatbot. Ask me anything?")
    st.header("Hello, I am a Movie Chatbot. Ask me anything?")
    user_csv = "actors.csv"
    os.environ["OPENAI_API_KEY"] =st.text_input("Enter your OpenAI API")
    #user_question = input("Ask a question about your CSV ?")
    user_question=st.text_input("Ask me a question about movies?")
    
    llm = OpenAI(temperature=0)
    agent = create_csv_agent(llm, user_csv, verbose="True",agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,)

    if user_question is not None and user_question != "":
        response = agent.run(user_question)
        st.write(response)
        print(response)


if __name__ == "__main__":
    main()
