import streamlit as st
import logging, sys, os
import openai
from dotenv import load_dotenv
from llama_index import (
    Document,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.tools import QueryEngineTool
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI

#loads dotenv lib to retrieve API keys from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# enable INFO level logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
service_context = ServiceContext.from_defaults(llm=llm)

# function to query db
def db_querying():
    sql_query = st.text_area("Enter your database query here")
    if sql_query:

        # establish connection to Snowflake
        conn = st.experimental_connection('snowpark')

        # run query based on the SQL entered 
        df = conn.query(sql_query)

        # write query result on UI
        st.write(df)

        query_engine_tools = []

        # Loop through the rows of the DataFrame
        for i in range(len(df)):
            exhibit991 = df.loc[i, 'VALUE']
            period_end_date = df.loc[i, 'PERIOD_END_DATE']
            
            # Process the value for each row
            index = VectorStoreIndex.from_documents(
                [Document(text=exhibit991)], 
                service_context=service_context
            )
            
            query_engine = index.as_query_engine(
                similarity_top_k=5, 
                service_context=service_context
            )
            
            query_engine_tool = QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name=f"{period_end_date}-exhibit991",
                description=f"Provides information about the exhibit 99.1 ending {period_end_date}.",
            )

            # Append the query_engine_tool to the list
            query_engine_tools.append(query_engine_tool)
            
        react_agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
        return react_agent
    
# add a title for our UI
st.title("Amazon's recent disclosures and attitudes toward large language models")

react_agent = db_querying()
if react_agent is not None:
    question = st.text_area("Enter your question here")
    if question:
        response = react_agent.chat(question)
        st.write("Response:", response)
