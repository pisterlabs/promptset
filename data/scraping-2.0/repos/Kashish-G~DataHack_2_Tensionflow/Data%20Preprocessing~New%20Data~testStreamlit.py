import streamlit as st
import pandas as pd

# Import the necessary libraries and functions

# Load the CSV data
df = pd.read_csv(r"C:\Users\a21ma\OneDrive\Desktop\Datahack\DataHack_2_Tensionflow\Data Preprocessing\New Data\PS_2_Test_Dataset.csv")

# Define the functions for LLM model and data extraction
from langchain.llms import OpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

def extract_data(input_text):
    llm = OpenAI(openai_api_key="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")
    prompt = ChatPromptTemplate.from_template("extract the following list: number of years of experience,Types of Law,type of court,rating,disposal time,cost per hour,languages,Lawfirm Names,City,Probono yes or no. For the data {input}.")
    model = llm
    str_chain = prompt | model | StrOutputParser()
    data = [i.split(": ") for i in str_chain.invoke({"input": input_text}).split('\n')][2:]
    data1 = [i[1] for i in data]
    return data1

# Streamlit app code
st.title("Data Extraction with LangChain")

# Create a text input field for user input
user_input = st.text_area("Enter the data:", value="Heer Jayaraman is a male attorney based in Mumbai. He has 4 years of experience and specializes in Criminal Law, Labor Law, and Family Law. He is fluent in Hindi, Punjabi, and Tamil, and is a graduate of Rastogi-Chahal. He charges an average rate of 1104.0049977395/hour and typically completes cases in 82.36359292032645 days. He has a 1.0 rating from his clients, who are typically large corporations, and does not currently do any Pro Bono work.")

# Add a button to trigger data extraction
if st.button("Extract Data"):
    extracted_data = extract_data(user_input)
    st.write("Extracted Data:")
    st.write(extracted_data)

# Optionally, you can display the loaded CSV data as well
st.write("CSV Data:")
st.write(df)
