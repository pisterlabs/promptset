import os
from time import time

# documents
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.storage import LocalFileStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

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
from langchain.memory import StreamlitChatMessageHistory # for Streamlit

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder

from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

import streamlit as st

# Initialize Dictionaries
tool_dict = dict()
embeddings_dict = dict()
db_dict = dict()
retriever_dict = dict()
vector_dict = dict()
description_dict = dict()
answer_dict=dict()
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

def create_retriever(
    documents, site_key, filepath, 
    embeddings_dict=embeddings_dict, 
    vector_dict=vector_dict, text_splitter=None
    ):
    """
    Parameters:
        - text_splitter (optional): a text splitter object. If None, the documents are not split. 
    """
    start_time = time()
    if text_splitter is None: # object type is the same (class 'langchain.schema.document.Document') whether or not the documents are split
        texts = documents
    else:
        texts = text_splitter.split_documents(documents)
   
    underlying_embeddings = OpenAIEmbeddings(
        openai_organization=os.environ['openai_organization'],
        openai_api_key=os.environ['openai_api_key']
        )
    embeddings_dict[site_key] = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, LocalFileStore(filepath), 
        namespace=f'{site_key}_{underlying_embeddings.model}'
        )
    vector_dict[site_key] = FAISS.from_documents(texts, embeddings_dict[site_key])
    retriever_dict[site_key] = vector_dict[site_key].as_retriever()
    print(f'Retriever created for {site_key} in {time() - start_time} seconds')
    return retriever_dict
    # return embeddings_dict


def create_retriever_and_description_dicts(params_dict, filepath, doc_dict=doc_dict, vector_dict=vector_dict):
    start_time = time()
    retriever_dict = dict()
    description_dict = dict()
    for doc_id in doc_dict:
        retriever_dict = create_retriever(
            doc_dict[doc_id], params_dict[doc_id]['site_key'], 
            filepath,
            vector_dict=vector_dict, 
            text_splitter=params_dict[doc_id].get('text_splitter', None)
            )
        description_dict[params_dict[doc_id]['site_key']] = params_dict[doc_id]['doc_description']
    print(f'Created retriever and description dicts for {params_dict.keys()} in {time() - start_time} seconds')
    return retriever_dict, description_dict




def create_tools_list(retriever_dict, description_dict):
    """
    https://api.python.langchain.com/en/latest/agents/langchain.agents.agent_toolkits.conversational_retrieval.tool.create_retriever_tool.html?highlight=create_retriever_tool#langchain.agents.agent_toolkits.conversational_retrieval.tool.create_retriever_tool
    """
    tools_list = []
    for site_key in retriever_dict:
        tool_name = f'search_{site_key}'
        print(f'Retriever: {retriever_dict[site_key]}/n')
        tool = create_retriever_tool(retriever_dict[site_key], tool_name, description_dict[site_key])
        tools_list.append(tool)
    return tools_list

def create_chatbot(tools, verbose=True, streamlit=False):

    llm = ChatOpenAI(
        temperature = 0,
        openai_organization=os.environ['openai_organization'],
        openai_api_key=os.environ['openai_api_key'],
        )
    if streamlit == False:
        memory = AgentTokenBufferMemory(memory_key='chat_history', llm=llm)
    else:
        msgs = StreamlitChatMessageHistory()
        memory = AgentTokenBufferMemory(memory_key='chat_history', llm=llm, chat_memory=msgs)
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
    start_time = time()
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
    print(f'Response time: {time() - start_time} seconds')
    
    return result