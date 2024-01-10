import os

# documents
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.agents.agent_toolkits import create_retriever_tool

# Creating the Agent
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI

# Create memory
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory  # for Streamlit

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder

from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from dotenv import load_dotenv

import streamlit as st


# /Users/sunnyd/Downloads/Archive/data

# UPDATE THESE PARAMETERS AS NEEDED
# # Get the directory of the current script
# current_directory = os.path.dirname(os.path.abspath(__file__))

# # Construct the path to the 'data' directory
# directory = os.path.join(current_directory, '..', 'data/')

load_dotenv()

os.environ['openai_organization'] = os.getenv("openai_organization")
os.environ['openai_api_key'] = os.getenv("openai_api_key")



directory = '../data/'  # This is the directory containing the CSV/text files.

# Initialize Dictionaries
tool_dict = dict()
embeddings_dict = dict()
db_dict = dict()
retriever_dict = dict()
vector_dict = dict()
description_dict = dict()
answer_dict = dict()
conversation_dict = dict()
doc_dict = dict()
queries_dict = dict()


def create_documents(directory='../data', glob='**/[!.]*', show_progress=True, loader_cls=CSVLoader):
    loader = DirectoryLoader(
        directory, glob=glob, show_progress=show_progress,
        loader_cls=loader_cls)

    documents = loader.load()
    print(f'Number of files: {len(documents)}')
    return documents


def create_documents_from_csv(file_path='../data/Datajam_2023___Fine_Tuning_ChatBot_CSV_-_Recycle_BC_1.csv'):
    loader = CSVLoader(file_path, encoding='utf-8')
    documents = loader.load()
    return documents


def create_retriever(documents, site_key, vector_dict=vector_dict, text_splitter=None):
    """
    Parameters:
        - text_splitter (optional): a text splitter object. If None, the documents are not split. 
    """
    embeddings_dict[site_key] = OpenAIEmbeddings(
        openai_organization=os.environ['openai_organization'],
        openai_api_key=os.environ['openai_api_key']
    )
    # object type is the same (class 'langchain.schema.document.Document') whether or not the documents are split
    if text_splitter is None:
        texts = documents
    else:
        texts = text_splitter.split_documents(documents)

    vector_dict[site_key] = FAISS.from_documents(
        texts, embeddings_dict[site_key])
    retriever_dict[site_key] = vector_dict[site_key].as_retriever()
    return retriever_dict


def create_tools_list(retriever_dict, description_dict):
    """
    https://api.python.langchain.com/en/latest/agents/langchain.agents.agent_toolkits.conversational_retrieval.tool.create_retriever_tool.html?highlight=create_retriever_tool#langchain.agents.agent_toolkits.conversational_retrieval.tool.create_retriever_tool
    """
    tools_list = []
    for site_key, retriever in retriever_dict.items():
        tool_name = f'search_{site_key}'
        tool = create_retriever_tool(
            retriever_dict[site_key], tool_name, description_dict[site_key])
        tools_list.append(tool)
    return tools_list


def create_chatbot(tools, verbose=True, streamlit=False):

    llm = ChatOpenAI(
        temperature=0,
        openai_organization=os.environ['openai_organization'],
        openai_api_key=os.environ['openai_api_key'],
    )
    if streamlit == False:
        memory = AgentTokenBufferMemory(memory_key='chat_history', llm=llm)
    else:
        msgs = StreamlitChatMessageHistory()
        memory = AgentTokenBufferMemory(
            memory_key='chat_history', llm=llm, chat_memory=msgs)
    system_message = SystemMessage(
        content=("""
            You are a helpful assistant who provides concise answers to residents in Metro Vancouver, Canada.
            You ask enough follow up questions as needed to provide the most relevant answer. 
            Where relevant, you retrieve the relevant information from your documents to answer the resident's question.
            Recycle BC is the main resource for recycling information. 
            Respond to the resident's query, which are delimited by triple backticks: ```{question}```\n\n
            If relevant, indicate the source website from which you are basing your response.
            """
                 ),
        input_variables=['question']
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[
            MessagesPlaceholder(variable_name='chat_history')
        ]
    )

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, memory=memory, verbose=verbose, return_intermediate_steps=True
    )
    agent_info = {
        'agent': agent,
        'agent_executor': agent_executor,
        'memory': memory,
        'chat_history': [],
        'msgs': msgs if streamlit == True else None
    }
    return agent_info


def chat_with_chatbot(user_input, agent_info, streamlit=False):

    print(f'Chat history length: {len(agent_info["chat_history"])}')
    if streamlit == False:
        chat_history = agent_info['chat_history']
    else:
        if type(agent_info['msgs'].messages) == list:
            chat_history = agent_info['msgs'].messages
        else:
            chat_history = [agent_info['msgs'].messages]
    result = agent_info['agent_executor']({
        "input": user_input,
        "chat_history": chat_history
    })
    agent_info['chat_history'].append(result['chat_history'])

    return result
