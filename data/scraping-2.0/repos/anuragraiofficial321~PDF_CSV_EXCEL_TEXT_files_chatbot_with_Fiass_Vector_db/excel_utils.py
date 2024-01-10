import os
from typing import TextIO
from Secret_key import openai_api_key
import openai
import streamlit as st
import pandas as pd
from langchain.agents import create_csv_agent, create_pandas_dataframe_agent
from langchain.llms import OpenAI


os.environ["OPENAI_API_KEY"]=openai_api_key
llm=OpenAI(temperature=0)
#convert excel file to csv we isntall openpyxl
def save_uploaded_excel(file: TextIO) -> str:
    upload_folder = "uploaded_files"
    os.makedirs(upload_folder, exist_ok=True)

    # Delete any existing .csv file in the "uploaded_files" folder
    for filename in os.listdir(upload_folder):
        if filename.lower().endswith(".xlsx"):
            existing_csv_file = os.path.join(upload_folder, filename)
            os.remove(existing_csv_file)

    # Save the uploaded file (assuming it is a .xls file)
    file_path = os.path.join(upload_folder, "uploaded.xlsx")
    with open(file_path, "wb") as f:
        f.write(file.read())
    return file_path

def get_answer_excel(file, query):
    #print(file.name)
    """
    Returns the answer to the given query by querying a CSV file.

    Args:
    - file (str): the file path to the CSV file to query.
    - query (str): the question to ask the agent.

    Returns:
    - answer (str): the answer to the query from the CSV file.
    """
    # Load the CSV file as a Pandas dataframe
    # df = pd.read_csv(file)
    #df = pd.read_csv("titanic.csv")
    # Create an agent using OpenAI and the Pandas dataframe
    
    #check the default data frame
    #print(agent.agent.llm_chain.prompt.template)

    file_path=save_uploaded_excel(file)

# Read the XLS file using pandas and openpyxl as the engine
    ##df = pd.read_excel(file_path,engine='openpyxl')
    ##csv_output_path = os.path.splitext(file_path)[0] + ".csv"
      
    df = pd.read_excel(file_path)
    
    # Save the data as a CSV file
    ##df.to_csv(csv_output_path, index=False)
    ##print(csv_output_path)

    st.write(df)
    

    ##agent = create_csv_agent(llm, csv_output_path, verbose=False)
    agent = create_pandas_dataframe_agent(llm, df, verbose=False)
    

    # Run the agent on the given query and return the answer
    #query = "whats the square root of the average age?"
    answer = agent.run(query)
    print(agent)
    
    return answer