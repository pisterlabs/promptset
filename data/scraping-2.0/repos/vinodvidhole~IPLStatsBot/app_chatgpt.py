from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback
from dataclasses import dataclass 
from typing import Literal
import streamlit as st
import os

# Setting Menu options and web page configurations
st.set_page_config(
    page_title="IPL 2023 StatsBot",
    page_icon="🏏",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/vinodvidhole/',
        'About': "**IPL 2023 StatsBot** powered by Azure Open AI, Python, LangChain, SQL, and Streamlit"
    }
)

# Adding Sidebar
st.sidebar.title("IPL 2023 Statsbot 🏏")
st.sidebar.subheader("(ChatGPT Version)")
st.sidebar.markdown("Powered by Azure Open AI, Python, LangChain, SQL, and Streamlit")
st.sidebar.markdown("**Author:** [Vinod Dhole](https://www.linkedin.com/in/vinodvidhole/)")
st.sidebar.markdown("**Source Code:** [GitHub](https://github.com/vinodvidhole/IPLStatsBot)")

# About the App
st.sidebar.subheader("About the App")
st.sidebar.markdown(
    "The **IPL Stats Bot** is an advanced chatbot that offers immediate access to custom relational SQL data. "
    "It is developed using state-of-the-art technologies such as Azure OpenAI, Python, LangChain, and Streamlit."
)

# Overcoming Limitations
st.sidebar.subheader("Overcoming Limitations")
st.sidebar.markdown(
    "Conventional open AI models, including large language models (LLMs), often fall short in delivering "
    "real-time answers for the latest events or custom data queries. To address this limitation, our application was designed. "
    "We offer OpenAI models access to IPL data in SQL format through LangChain, enabling these models to provide highly "
    "accurate responses to specific questions without requiring prior knowledge of the database structure."
)

# Natural Language to SQL
st.sidebar.subheader("Natural Language to SQL")
st.sidebar.markdown(
    "Additionally, the application showcases Natural Language to SQL conversion, eliminating the need for SQL expertise "
    "to gain data insights from relational databases."
)

# How to Use
st.sidebar.subheader("How to Use")
st.sidebar.markdown(
    "To engage with this chatbot, simply ask IPL-related questions for IPL 2023 stats, such as the winner, "
    "top run-scorer, leading wicket-taker, or the player with the most Man of the Match awards."
)

os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]

@dataclass
class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str


# Global flag variable to track if session state has been initialized
session_state_initialized = False

def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

def initialize_session_state():
    global session_state_initialized
    if not session_state_initialized:
        if "history" not in st.session_state:
            st.session_state.history = []
        if "token_count" not in st.session_state:
            st.session_state.token_count = 0
        if "conversation" not in st.session_state:
            llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=True)
            st.session_state.conversation = llm
        session_state_initialized = True

## for future implementation 
def collect_feedback(user_query, model_response, user_rating):
    # Save the user's query, model response, and user rating to a feedback database or file
    with open("feedback.csv", "a") as feedback_file:
        feedback_file.write(f"{user_query},{model_response},{user_rating}\n")

def on_click_callback():
    with get_openai_callback() as cb:
        
        human_prompt = st.session_state.human_prompt
        try:
            llm_response = answer_question(human_prompt)
        except Exception as e:
            llm_response = f"An error occurred: {str(e)}"



        st.session_state.token_count += cb.total_tokens

DEFAULT_TABLES = [
    'each_match_records',
    'each_ball_records'
]

def get_prompt():
    _DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Use the following format:

    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"

    Answer: ""

    Only use the following tables:

    {table_info}

    Question: {input}"""

    PROMPT = PromptTemplate(
        input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
    )
    return PROMPT

def get_db():
    db = SQLDatabase.from_uri("sqlite:///ipl_2023.db",
                                  include_tables = DEFAULT_TABLES,
                                  sample_rows_in_table_info=2)
    return db

def answer_question(query):
    PROMPT = get_prompt()
    db = get_db()

    # Initialize the session state if it hasn't been done
    initialize_session_state()

    llm = st.session_state.conversation

    try:
        db_chain = SQLDatabaseChain.from_llm(llm, db,
                                             prompt=PROMPT,
                                             verbose=True,
                                             return_intermediate_steps=True,
                                             # use_query_checker=True
                                             )
        result = db_chain(query)

        

        sql_cmd = None
        for step in result['intermediate_steps']:
            if 'sql_cmd' in step:
                sql_cmd = step['sql_cmd']
                break

        final_op = "{}\nSQL Command: < {} >".format(result['result'], sql_cmd)
        
        # Append the user query to the conversation history
        st.session_state.history.append(Message("human", query))
        st.session_state.history.append(Message("ai", result['result']))
        st.session_state.history.append(Message("ai", "SQL Command < {} >".format(sql_cmd)))

        return final_op
    
    except Exception as e:
        # Append the user query to the conversation history
        st.session_state.history.append(Message("human", query))
        st.session_state.history.append(Message("ai", "An error occurred< {} >".format(str(e))))

        return f"An error occurred: {str(e)}"

load_css()
initialize_session_state()
    
# Add the logo
path = os.path.dirname(__file__)
# Define the logo path
logo_path = os.path.join(path, "static", "ipl.png")

st.image(logo_path, width=200)

st.title("IPL 2023 Statsbot 🏏")
st.subheader("(ChatGPT Version)")
st.markdown("AI chatbot for IPL statistics and data analysis powered by Azure Open AI, Python, LangChain, SQL, and Streamlit")

if __name__ == "__main__":
    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")
    tokens_placeholder = st.empty()

    with chat_placeholder:
        for chat in st.session_state.history:
            div = f"""
    <div class="chat-row 
        {'' if chat.origin == 'ai' else 'row-reverse'}">
        <img class="chat-icon" src="app/static/{
            'ai_icon.png' if chat.origin == 'ai' 
                        else 'user_icon.png'}"
            width=32 height=32>
        <div class="chat-bubble
        {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
            &#8203;{chat.message}
        </div>
    </div>
            """
            st.markdown(div, unsafe_allow_html=True)
        
        for _ in range(3):
            st.markdown("")

    with prompt_placeholder:
        st.markdown("**Chat**")
        cols = st.columns((6, 1))
        cols[0].text_input(
            "Chat",
            value="Who won the final match of IPL 2023",
            label_visibility="collapsed",
            key="human_prompt",
        )
        cols[1].form_submit_button(
            "Submit", 
            type="primary", 
            on_click=on_click_callback, 
        )

    tokens_placeholder.caption("Used {} tokens".format(st.session_state.token_count))