import os
import pandas as pd
import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI


def process_uploaded_file(data, openai_api_key, user_input):
    os.environ["OPENAI_API_KEY"] = openai_api_key  # Set the API key as an environment variable

    if user_input:
        st.write(f"You entered: {user_input}")

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0.5), data, verbose=False)
    x = agent.run(user_input)
    return x


def main():
    st.header("🦜🔗 Excel GPT")

    uploaded_file = st.file_uploader("Choose a file (CSV or Excel)", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

        st.write("Data:")
        data = df.head(10)
        st.dataframe(data)
        
        st.markdown(
            "Don't have an API key? Get one from [OpenAI](https://platform.openai.com/account/api-keys)"
        )
        

        openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
        user_input = st.text_input("Ask a question about your data")

        if user_input and openai_api_key:
            x = process_uploaded_file(df, openai_api_key, user_input)
            st.write(x)
            st.markdown(
            "Connect with me on [LinkedIn](https://www.linkedin.com/in/prathapreddyk/)"
        )


if __name__ == "__main__":
    main()
