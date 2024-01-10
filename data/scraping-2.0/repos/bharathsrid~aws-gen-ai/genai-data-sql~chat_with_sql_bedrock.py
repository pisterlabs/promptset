# have to combine these two
# https://python.langchain.com/docs/expression_language/cookbook/multiple_chains
# https://python.langchain.com/docs/expression_language/cookbook/memory

# This is a streamlit app with chat history

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
import boto3
from botocore.config import Config
from langchain import PromptTemplate,SagemakerEndpoint,SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains import create_sql_query_chain
from sqlalchemy import create_engine
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from langchain.chat_models import BedrockChat
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage


import streamlit as st
import boto3


# Global DB connection
region = 'us-east-1'
athena_url = f"athena.{region}.amazonaws.com"
athena_port = '443'  # Update, if port is different
athena_db = 'demo-emp-deb-2'  # from user defined params
glue_databucket_name = 'athena-query-bucket-bharsrid'
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

    st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
    st.title("📖 StreamlitChatMessageHistory")

    """
    A basic example of using StreamlitChatMessageHistory to help LLMChain remember messages in a conversation.
    The messages are stored in Session State across re-runs automatically. You can view the contents of Session State
    in the expander below. View the
    [source code for this app](https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py).
    """

    # Set up memory
    # msgs = StreamlitChatMessageHistory(key="langchain_messages")
    msgs = StreamlitChatMessageHistory()

    print("MSGS SET")
    # memory = ConversationBufferMemory(chat_memory=msgs)
    # memory = ConversationBufferMemory(return_messages=True)
    memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)


    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")

    view_messages = st.expander("View the message contents in session state")

    # initiate Bedrock
    bedrock_run_time = boto3.client(service_name='bedrock-runtime')

    inference_modifier = {
        "temperature": 1,
        "top_p": .999,
        "top_k": 250,
        "max_tokens_to_sample": 300,
        "stop_sequences": ["\n\nSQL Query:"]
    }



    # bedrock_chain = LLMChain(
    #     llm=bedrock_llm,
    #     prompt=prompt,
    #     verbose=True,
    #     memory=chat_message_history,
    # )

    chat = BedrockChat(model_id="anthropic.claude-v2", model_kwargs=inference_modifier)

    # model = ChatOpenAI()



    sql_template = """Human: Based on the table schema below, write a SQL query and just the SQL, nothing else, that would answer the user's question.:
    {schema}


    Question: {question}
    SQL Query:    
    """
    sql_prompt = ChatPromptTemplate.from_template(sql_template)

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
    # 
    # full_chain = full_chain.with_memory(
    #     memory=memory,
    #     input_key="input",
    #     output_key="output"
    # )


#     # Render current messages from StreamlitChatMessageHistory
#     for msg in msgs.messages:
#         st.chat_message(msg.type).write(msg.content)

#     # If user inputs a new prompt, generate and draw a new response
#     if prompt := st.chat_input():
#         st.chat_message("human").write(prompt)
#         # Note: new messages are saved to history automatically by Langchain during run
#         response = full_chain.invoke(prompt)
#         st.chat_message("ai").write(response)

#     # Draw the messages at the end, so newly generated ones show up immediately
#     with view_messages:
#         """
#         Memory initialized with:
#         ```python
#         msgs = StreamlitChatMessageHistory(key="langchain_messages")
#         memory = ConversationBufferMemory(chat_memory=msgs)
#         ```
    
#         Contents of `st.session_state.langchain_messages`:
#         """
#         view_messages.json(st.session_state.langchain_messages)

    user_input = st.text_area("Enter querry to Bedrock")
    button = st.button("Ask Bedrock")
    messages = [
        HumanMessage(
            content=f"{user_input}"
        )
    ]
    if user_input and button:
        summary = full_chain.invoke({messages})
        st.write("Summary : ", summary)

if __name__ == "__main__":
    main()