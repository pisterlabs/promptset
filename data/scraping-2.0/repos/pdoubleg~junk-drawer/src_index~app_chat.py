import streamlit as st
st.set_page_config(layout="wide")
import streamlit.components.v1 as components
from streamlit_extras.stateful_button import button
from top_n_tool import run_tool
import os
from dotenv import load_dotenv
load_dotenv()
import markdown
import pandas as pd
import numpy as np
import time as time
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional
from io import StringIO
import langchain
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".st_langchain.db")
import sqlite3
from sqlalchemy import create_engine
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMMathChain
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_experimental.sql import SQLDatabaseChain
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import DuckDuckGoSearchRun

from html_templates import css, user_template, bot_template

from custom_tools import ResearchPastQuestions

tqdm.pandas()

# Constants
DATA_PATH = "reddit_legal_cluster_test_results.parquet"


def clean_names(df):
    df.columns = [x.replace(' ', '_').lower() for x in df.columns]
    return df

 
@st.cache_data
def get_df():
    """Returns a pandas DataFrame."""
    df = pd.read_parquet(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')
    df['datestamp'] = df['timestamp'].dt.date
    # df['state'] = pd.Categorical(df['state'])
    df['text_label'] = pd.Categorical(df['text_label'])
    df['topic_title'] = pd.Categorical(df['topic_title'])
    return df


def display_description():
    """Displays the description of the app."""
    st.markdown("<h4 style='text-align: left;'>Work with an AI research assistant</h4>", unsafe_allow_html=True)
    st.write(
        """
        Why use an AI agent?
        - 🧰 Agents have access to tools 
        - 🔧 Tools are small programs that do specific tasks
        - 👉 You give the instructions, and the agent figures out which tool, or set of tools to use
        """
    )
    
    
def b_get_feedback():
    if button('Feedback', key="open_feedback"):
        feedback_text = st.text_input("Please provide your feedback")
        feedback_score = st.number_input("Rate your experience (0-10)", min_value=0, max_value=10)
        user_feedback = pd.DataFrame({"Feedback_Text": [feedback_text], "Feedback_Score": [feedback_score]})
        if button('Send', key="send_feedback"):
            if os.path.exists("user_feedback.csv"):
                user_feedback.to_csv("user_feedback.csv", mode='a', header=False, index=False)
            else:
                user_feedback.to_csv("user_feedback.csv", index=False)
            time.sleep(1)
            st.toast("✔️ Feedback received! Thanks for being in the loop 👍\nClick the `Feedback` button to open or close this anytime.")


def app():
    """Main function that runs the Streamlit app."""
    st.markdown(
        "<h2 style='text-align: left;'>GPT Research Agent 📚</h2>",
        unsafe_allow_html=True,
    )
    
    # Add a sidebar dropdown for model selection
    model = st.sidebar.selectbox(
        'Select Model',
        ('gpt-3.5-turbo', 'gpt-4')
    )

    load_dotenv()
    
    df = get_df()
    
    llm = ChatOpenAI(model=model, temperature=0.0)
    
    table_name = "questions_table"
    uri = "sqlite:///questions_table.db"
    # Create the sqlalchemy engine
    engine = create_engine(uri, echo=False)
    # Prep column names
    data = clean_names(df)
    # Convert the DataFrame to SQL
    data.to_sql(table_name, con=engine, index=False, if_exists='replace')
    # Create connection for LangChain
    db = SQLDatabase.from_uri(uri)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True)

    memory = ConversationBufferMemory(memory_key="chat_history")
    DDGsearch = DuckDuckGoSearchRun()
    research_past_questions = ResearchPastQuestions(df=df)
    

    
    tools = [
        Tool(
            name = "SQL Structured Data Query Tool",
            func = db_chain.run,
            description="Useful for answering questions using structured data query"
        ),
        Tool(
            name = "Duck Duck Go Search Results Tool",
            func = DDGsearch.run,
            description="Useful for search for information on the internet"
        ),
        Tool(
            name ='Legal Questions Research Tool',
            func=research_past_questions.run,
            description='Useful for finding similar legal questions for a new query',
            return_direct = False
        )
    ]
    
    
    agent = initialize_agent(tools,
                             llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True, 
                             memory=memory,
                             return_intermediate_steps=False,
                             max_iterations=5,
                             handle_parsing_errors=True,
                             )
    
    # Add a reset button to the sidebar
    reset_button = st.sidebar.button("Reset & Clear")
    if reset_button:
        st.session_state.clear()  # Clear the session state to reset the app
        agent = None
    
    display_description()
    b_get_feedback()
    # Display sample questions
    with st.expander("❓ Here are some example questions you can ask:", expanded=False):
        st.markdown(
            """
            - Show me questions about getting fired for medical marijuana use while at work and on the job.
            """
        )
    
    st.write(css, unsafe_allow_html=True)
    # Define chat history session state variable
    st.session_state.setdefault('chat_history', [])
    
    if "messages" not in st.session_state: # Initialize the chat message history
        st.session_state.messages = []
    #         {"role": "assistant", "content": "Ask me a question"}
    # ]
    
    # Prompt for user input and save
    if query := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": query})
    
    # display the existing chat messages
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        if message["role"] == "user":
            with st.chat_message(message["role"], avatar="https://raw.githubusercontent.com/pdoubleg/junk-drawer/main/src_index/data/icons/user_question_resized_pil.jpg"):
                st.write(message["content"])
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="https://raw.githubusercontent.com/pdoubleg/junk-drawer/main/src_index/data/icons/c3po_icon_resized_pil.jpg"):
                st.write(message["content"])


        with st.chat_message("assistant", avatar="https://raw.githubusercontent.com/pdoubleg/junk-drawer/main/src_index/data/icons/c3po_icon_resized_pil.jpg"):
            resp_container = st.empty()
            st_callback = StreamlitCallbackHandler(st.container(), collapse_completed_thoughts=False)
            
            response = agent.run(query, callbacks=[st_callback])

            resp_container.markdown(response)

            message = {"role": "assistant", "content": response}
    # for message in st.session_state.messages: # Display the prior chat messages
    #     with st.chat_message(message["role"]):
    #         st.write(message["content"])
            
 
    # if query:= st.chat_input(placeholder="Ask me anything!"):
    #     st.session_state.messages.append({"role": "user", "content": query})
    #     col1, col2 = st.columns([1, 9])
    #     with col1:
    #         st.image("https://raw.githubusercontent.com/pdoubleg/junk-drawer/main/src_index/data/icons/c3po_icon_resized_pil.jpg")

    #     with col2:
    #         st_callback = StreamlitCallbackHandler(st.container(), collapse_completed_thoughts=False)
    #     response = agent.run(query, callbacks=[st_callback])
    #     # Append the response to the session_state messages
    #     st.session_state.messages.append({"role": "assistant", "content": response})
    #     # Display conversation in reverse order
    #     for message in reversed(st.session_state.messages):
    #         if message["role"] == "user": 
    #             st.markdown(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
    #         else: 
    #             st.markdown(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)


if __name__ == "__main__":
    app()