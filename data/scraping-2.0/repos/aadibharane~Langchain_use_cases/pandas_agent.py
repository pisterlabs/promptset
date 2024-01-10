import os
import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from apikey import apikey


# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = apikey

# Define Streamlit app
def pandas_agent():
      # Title and description
    st.title("CSV Query App")
    st.write("Upload a CSV file and enter a query to get an answer.")
    file =  st.file_uploader("Upload CSV file",type=["csv"])
    if not file:
        st.stop()

    data = pd.read_csv(file)
    st.write("Data Preview:")
    st.dataframe(data.head()) 

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0),data,verbose=True) 

    query = st.text_input("Enter a query:") 

    if st.button("Execute"):
        answer = agent.run(query)
        st.write("Answer:")
        st.write(answer)


    
  
if __name__ == "__main__":
    pandas_agent()  