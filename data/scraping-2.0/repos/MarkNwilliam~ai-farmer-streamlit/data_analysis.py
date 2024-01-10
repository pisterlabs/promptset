import os
import pandas as pd
import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# Set environment variables
load_dotenv()

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
# Initialize Langchain Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)

llm = OpenAI()

llm_chain = LLMChain(prompt=prompt, llm=llm)

def show():
    st.title("Data Analysis with Langchain")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    # User inputs the question
    user_question = st.text_input("Enter your question about the CSV data")

    if st.button('Analyze'):
        if uploaded_file is not None and user_question:
            try:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(uploaded_file)
                # Convert the DataFrame to a string (or any other format that the agent can process)
                csv_data_as_string = df.to_string(index=False)

                # Combine the user's question with the CSV data
                full_question = f"{user_question}\n\nCSV Data:\n{csv_data_as_string}"

                # Get the response from the agent
                response = llm_chain.run(full_question)
                st.write("Agent Response:", response)

            except Exception as e:
                st.error(f"An error occurred: {e}")