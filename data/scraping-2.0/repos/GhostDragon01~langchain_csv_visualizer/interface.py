import streamlit as st
import pandas as pd

from pandas_agent import PandasAgent
from langchain_agent import summary_agent

st.title("👨‍💻 Chat with your CSV")

st.write("Please upload your CSV file below.")

data = st.file_uploader("Upload your CSV file")

if data:
    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(data)

    st.session_state.df = df
    st.write(st.session_state.df)

    summary= summary_agent(df)
    st.write(summary)

    csv_agent = PandasAgent()
    # query = st.text_area("Insert your query")

    with st.form(key="query"):
            query = st.text_area("Insert your query", placeholder="e.g- What's the most used name ?")
            submitted_query = st.form_submit_button("Submit")
    
  
    if submitted_query:
        result, captured_output = csv_agent.get_agent_response(df, query)
        cleaned_thoughts = csv_agent.process_agent_thoughts(captured_output)
        csv_agent.display_agent_thoughts(cleaned_thoughts)
        csv_agent.display_agent_response(result)
        