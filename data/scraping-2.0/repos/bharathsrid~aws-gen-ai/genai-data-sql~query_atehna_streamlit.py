from langchain.prompts import ChatPromptTemplate
from langchain.utilities import SQLDatabase
import boto3
from botocore.config import Config
from langchain import PromptTemplate,SagemakerEndpoint,SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains import create_sql_query_chain
from sqlalchemy import create_engine

from langchain.chat_models import BedrockChat
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import HumanMessage
import streamlit as st


region = 'us-east-1'
athena_url = f"athena.{region}.amazonaws.com" 
athena_port = '443' #Update, if port is different
athena_db = 'demo-emp-deb-2' #from user defined params
glue_databucket_name='athena-query-bucket-bharsrid'
s3stagingathena = f's3://{glue_databucket_name}/athenaresults/' 
athena_wkgrp = 'primary' 
athena_connection_string = f"awsathena+rest://@{athena_url}:{athena_port}/{athena_db}?s3_staging_dir={s3stagingathena}/&work_group={athena_wkgrp}"

print(athena_connection_string)
athena_engine = create_engine(athena_connection_string, echo=True)
athena_db_connection = SQLDatabase(athena_engine)

def get_schema(_):
    return athena_db_connection.get_table_info()

def run_query(query):
    return athena_db_connection.run(query)

def main():

    sql_template = """Human: Based on the table schema below, write a SQL query and just the SQL, nothing else, that would answer the user's question.:
    {schema}


    Question: {question}
    SQL Query:
    """
    sql_prompt = ChatPromptTemplate.from_template(sql_template)
    
    inference_modifier = {
    "temperature": 1,
    "top_p": .999,
    "top_k": 250,
    "max_tokens_to_sample": 300,
    "stop_sequences": ["\n\nSQL Query:"]
}
    


    chat = BedrockChat(model_id="anthropic.claude-v2", model_kwargs=inference_modifier)

    # model = ChatOpenAI()

    sql_response = (
        RunnablePassthrough.assign(schema=get_schema)
        | sql_prompt
        | chat.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
    
    template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
    prompt_response = ChatPromptTemplate.from_template(template)
    
    full_chain = (
    RunnablePassthrough.assign(query=sql_response)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: athena_db_connection.run(x["query"]),
    )
    | prompt_response
    | chat
)
#     full_chain.invoke({"question": "How many employees are there?"})
    
#     response = full_chain.invoke({"question": "How many employees are there?"})
#     print(response)
    
    
    user_input = st.text_area("Enter querry to Bedrock")
    button = st.button("Ask Bedrock")
    # messages = [
    #     HumanMessage(
    #         content=f"{user_input}"
    #     )
    # ]
    if user_input and button:
        summary = full_chain.invoke({"question":user_input})
        print(summary)
        st.write("Summary : ", summary)



if __name__ == "__main__":
    main()