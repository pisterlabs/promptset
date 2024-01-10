import os
import streamlit as st
import pandas as pd
from llama_index.query_engine.pandas_query_engine import PandasQueryEngine
from llama_index import LLMPredictor , ServiceContext
from langchain.chat_models import ChatOpenAI
from llama_index.llms import OpenAI


OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

st.set_page_config(page_title="Chat with Your Data")

model = "GPT-4"
st.header(f"Analyze your doucments with text queries")

#508,420,500 investmnet 


@st.cache_data()
def load_docs(file):
    file_extension = os.path.splitext(file.name)[1]
    print("file extention" , file_extension)
    if file_extension == ".csv":
        df = pd.read_csv(file)
    elif file_extension == ".xlsx":
        df = pd.read_excel(file)

    return df 




def get_text():
    input_text = st.text_input("" , key="input")
    return input_text

@st.cache_resource
def get_engine(df , DEFAULT_INSTRUCTION_STR):
    llm= OpenAI(temperature=0, model_name="gpt-4")
    service_context = ServiceContext.from_defaults(llm=llm)
    return PandasQueryEngine(df = df , service_context=service_context) #,  instruction_str=DEFAULT_INSTRUCTION_STR) #service_context=service_context , instruction_str=DEFAULT_INSTRUCTION_STR)

def main():
    uploaded_files = st.file_uploader(label = "Upload your excel file")
    if uploaded_files is not None:
        df = load_docs(uploaded_files)

        DEFAULT_INSTRUCTION_STR = f"""You should only use columns from the following list to get your answer: {df.columns}"\n.
You should not make up a column name under any circumstance. If you think a relevant column is not available to answer a query , you must try infering from the existing columns.
Use the values shown in the table rows for filtering rows. Do not make your own values.
We wish to convert this query to executable Python code using Pandas.

The final line of code should be a Python expression that can be called with the `eval()` function."


Think step by step and come up with a logic to get the answer.
"""
        
    
        query_engine = get_engine(df , DEFAULT_INSTRUCTION_STR)
        
        user_input = get_text()

        if query_engine:
            if user_input and uploaded_files is not None:
                print("user input is" , user_input)
                response = query_engine.query(user_input)
                
                st.write(df.head())
                st.write("Command to get data")
                st.write(response.metadata["pandas_instruction_str"])

                st.write("Executed Code")
                st.write(response.response)


if __name__ == "__main__":
    main()
