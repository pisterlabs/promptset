import streamlit as st
import pandas as pd
import ydata_profiling as yp
from streamlit_pandas_profiling import st_profile_report

st.title("Data Analyst Bot")

data_uploaded = False
df = None

st.header("Please upload your dataset")
file = st.file_uploader("Upload CSV File", type=["csv"])
if file: 
    df = pd.read_csv(file, index_col=None)
    df.to_csv('dataset.csv', index=None)
    st.dataframe(df)
    data_uploaded = True

if data_uploaded:
    st.header("Exploratory Data Analysis (EDA)")

    # Perform EDA using ydata_profiling
    profile_df = yp.ProfileReport(df)
    st_profile_report(profile_df)



# from langchain.llms import OpenAI
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.callbacks import StdoutCallback
from pandasai.llm import OpenAI
import streamlit as st
import pandas as pd
import os

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None

uploaded_file = file

if uploaded_file:
    df = load_data(uploaded_file)

openai_api_key = st.sidebar.text_input("OpenAI API Key",
                                        type="password",
                                        placeholder="Paste your OpenAI API key here (sk-...)")

with st.sidebar:
        st.markdown("@ Bravish Ghosh")

if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    #PandasAI OpenAI Model
    llm = OpenAI(api_token=openai_api_key)
    # llm = OpenAI(api_token=openai_api_key)

    sdf = SmartDataframe(df, config = {"llm": llm,
                                        "enable_cache": False,
                                        "conversational": True,
                                        "callback": StdoutCallback()})

    with st.chat_message("assistant"):
        response = sdf.chat(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
