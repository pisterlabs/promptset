import chainlit as cl
from chainlit import user_session
from langchain import PromptTemplate

# API Key
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
key = os.environ['OPENAI_API_KEY']

# reading postgres server login details
f = open('../data/postgres_login', 'r')
postgres_log_params = f.read().splitlines()
f.close()

# reading chat_model version
f = open('../data/chat_gpt_', 'r')
chat_model = f.read()
f.close()
model_name = 'gpt-4' if chat_model == 'GPT4' else 'gpt-3.5-turbo'

def generate_llm(model_name=model_name, key=key):
    # generates the LLM, only implemented with OpenAI for now

    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(
        openai_api_key=key,
        model_name=model_name,
        temperature=0)
    return llm

def connect_db(host, port, username, password, database):

    # post gres SQL setup
    from langchain.sql_database import SQLDatabase
    db = SQLDatabase.from_uri(
        f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")
    return db


@cl.author_rename
# rename the chatbot
def rename(orig_author: str):
    rename_dict = {"Chatbot": "SQL Assistant"}
    return rename_dict.get(orig_author, orig_author)

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    # set up database chain
    from langchain.chains import SQLDatabaseChain
    from langchain.prompts import ChatPromptTemplate
    db = connect_db(*postgres_log_params)
    llm = generate_llm()

    # Setup the database chain
    from langchain.memory import ConversationBufferMemory

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    db_chain = SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        verbose=False,
        memory=memory,
    )
    # Store the chain in the user session
    cl.user_session.set("db_chain", db_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    db_chain = cl.user_session.get("db_chain")
    # Call the chain asynchronously

    res = await cl.make_async(db_chain)(message,
                                        callbacks=[cl.LangchainCallbackHandler(
                                            stream_final_answer=True,
                                        )])

    # Send the response
    await cl.Message(content=res["result"]).send()
