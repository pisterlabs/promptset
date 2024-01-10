import os

import matplotlib.pyplot as plt
import openai
import pandas as pd
import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, Ollama
from langchain.schema.output_parser import OutputParserException
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import MutableMapping

# Models
available_models = {
    "GPT-3.5": "gpt-3.5-turbo-0613",  # dometimes bad + unparsable results
    "GPT-3": "text-davinci-003",
    "GPT-3.5 Instruct": "gpt-3.5-turbo-instruct",
    "ChatGPT-4": "gpt-4",
    "Code Llama (local, codeassist)": "codellama:latest",
    "Llama2 (local, codeassist)": "llama2:latest",
}

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

# Chat Logic
##Load file
@st.cache_data(ttl="2h")
def load_file(uploaded_file) -> pd.DataFrame | None:
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        df = file_formats[ext](uploaded_file)
        st.write(df.head())
        return df
    st.error(f"Unsupported file format: {ext}")
    return None


# Generate LLM Response
def generate_langchain_response(
    df, input_query, model_name="gpt-3.5-turbo-0613", temperature=0.1, callbacks=None
) -> MutableMapping:
    response = dict()
    agent_kw_args = {
        "verbose": True,
        "agent_type": AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        "handle_parsing_errors": True,
        "max_iterations": 5,
    }
    if model_name in ("text-davinci-003", "gpt-3.5-turbo-instruct"):
        llm = OpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=st.session_state.get("OPENAI_API_KEY"),
        )
    elif model_name in ("gpt-3.5-turbo-0613", "gpt-3.5-turbo", "gpt-4"):
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=st.session_state.get("OPENAI_API_KEY"),
        )
    elif model_name in ("codellama:latest", "llama2:latest"):
        ######################
        # OLLAMA SUPPORT     #
        # need ollama        #
        # locally installed  #
        ######################
        llm = Ollama(model=model_name)

    # Pandas Dataframe Agent
    agent = create_pandas_dataframe_agent(llm, df, **agent_kw_args)

    # Agent Query
    try:
        response = agent(input_query, callbacks=callbacks)
    except Exception as e:
        st.stop()
        if type(e) == openai.error.APIError:
            st.error(
                "OpenAI API Error. Please try again a short time later. ("
                + str(e)
                + ")"
            )
        elif type(e) == openai.error.Timeout:
            st.error(
                "OpenAI API Error. Your request timed out. Please try again a short time later. ("
                + str(e)
                + ")"
            )
        elif type(e) == openai.error.RateLimitError:
            st.error(
                "OpenAI API Error. You have exceeded your assigned rate limit. ("
                + str(e)
                + ")"
            )
        elif type(e) == openai.error.APIConnectionError:
            st.error(
                "OpenAI API Error. Error connecting to services. Please check your network/proxy/firewall settings. ("
                + str(e)
                + ")"
            )
        elif type(e) == openai.error.InvalidRequestError:
            st.error(
                "OpenAI API Error. Your request was malformed or missing required parameters. ("
                + str(e)
                + ")"
            )
        elif type(e) == openai.error.AuthenticationError:
            st.error("Please enter a valid OpenAI API Key. (" + str(e) + ")")
        elif type(e) == openai.error.ServiceUnavailableError:
            st.error(
                "OpenAI Service is currently unavailable. Please try again a short time later. ("
                + str(e)
                + ")"
            )
        elif type(e) == OutputParserException:
            st.error(
                "Unfortunately the code generated from the model contained errors and was unable to execute or parsable. Please run again ("
                + str(e)
                + ")"
            )
        else:
            st.error(
                "Unfortunately the code generated from the model contained errors and was unable to execute. Please run again ("
                + str(type(e))
                + str(e)
                + ")"
            )
    return response


# Frontend Logic
def sidebar_toggle_helper() -> str:
    if st.secrets.has_key("OPENAI_API_KEY"):
        return "collapsed"
    return "expanded"


st.set_page_config(
    page_icon="chat2vis.png",
    layout="wide",
    page_title="Chat2DataInsight",
    initial_sidebar_state=sidebar_toggle_helper(),
)
st.markdown(
    "<h1 style='text-align: center; font-weight:bold; font-family:comic sans ms; padding-top: 0rem;'>Chat2DataInsight</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h2 style='text-align: center;padding-top: 0rem;'>Creating Visualisations and Data Analysis using Natural Language with ChatGPT/LLM</h2>",
    unsafe_allow_html=True,
)

with st.sidebar:
    model_name = st.selectbox("Model: ", available_models.keys())

    if openai_key := st.secrets.get("OPENAI_API_KEY"):
        st.session_state["OPENAI_API_KEY"] = openai_key
    else:
        st.session_state["OPENAI_API_KEY"] = st.text_input(
            label=":key: OpenAI Key:",
            help="Required for ChatGPT-4, ChatGPT-3.5, GPT-3, GPT-3.5 Instruct.",
            type="password",
        )


uploaded_file = st.file_uploader(
    ":computer: Choose a file",
    accept_multiple_files=False,
    type=list(file_formats.keys()),
)
if uploaded_file is not None:
    # read file to dataframe in dataset session state
    st.session_state["dataset"] = load_file(uploaded_file)

prompt = st.text_area(":eyes: What would you like to analyze?", height=10)
btn_go = st.button("Go...", use_container_width=True)


if prompt and btn_go:
    st.header("Output")
    st_cb = StreamlitCallbackHandler(
        st.container(), expand_new_thoughts=True, collapse_completed_thoughts=False
    )
    df = st.session_state.get("dataset").copy()
    response = generate_langchain_response(
        df,
        prompt,
        model_name=available_models.get(model_name),
        callbacks=[st_cb, StreamingStdOutCallbackHandler()],
    )
    st.write(response.get("output"))
    if len(plt.get_fignums()) > 0:
        fig = plt.gcf()
        st.pyplot(fig=fig, clear_figure=True)
