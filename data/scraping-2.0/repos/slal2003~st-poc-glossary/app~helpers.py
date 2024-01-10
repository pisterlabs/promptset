# import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
import openai
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
    )
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from dotenv import find_dotenv, load_dotenv

def create_text(df):
    df['text'] = df['definition'] + ' ' + df['tags']
    return df

def create_azure_embeddings(df):
    def embed(text):
        print('embedding creation started')
        embeddings = OpenAIEmbeddings (
        deployment='acme-gpt-text-embedding-ada-002',
        openai_api_type= os.getenv('openai.api_type'),
        openai_api_base= os.getenv('openai.api_base'),
        openai_api_key= os.getenv('openai.api_key'),
        openai_api_version=os.getenv('openai.api_version'),
        )
        print(embeddings)
        print('embedding auth completed')
        print(text)
        emb = embeddings.embed_query(text)
        return emb
    # df['text'] = df['definition'] + ' ' + df['tags']
    # df['text'] = np.where(df['definition'] == '', df['term'], df['definition']) + ' ' + df['tags']
    df['text'] = np.where(df['definition'].fillna('') == '', df['term'].fillna(''), df['definition'].fillna('')) + ' ' + df['tags'].fillna('')
    df['embeddings'] = df['text'].apply(embed)
    return df

def create_openai_embeddings(df):
    load_dotenv(find_dotenv())
    def embed(text):
        return OpenAIEmbeddings().embed_query(text)
    df['text'] = df['definition'] + ' ' + df['tags']
    df['embeddings'] = df['text'].apply(embed)
    return df
      
def extract_tags(term, definition):
    load_dotenv(find_dotenv())
    query = 'list up to 7 tags for this definition. The tags must be short and together help recreate the definition and help discriminate this definition from others that can be semantically close. the output should be a comma separated list '
    st.write(f'query: {query}')
    text = f"term: {term} definition:{definition}"
    st.write(f'text: {text}')
    llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613", max_tokens=200)
    st.write(llm)
    tags = llm(query + text)
    # chain=load_qa_chain(llm=llm, chain_type="stuff")
    # tags = chain.run(input_documents=text, question=query)
    return tags

# def generate_definition2( term, domain, keywords):
#     load_dotenv(find_dotenv())
#     messages = [
#         SystemMessage(content=f'You are a seasoned Data Stewart, expert in data analytics and business glossaries'),
#         HumanMessage(content=f'Suggest a definition for this term: {term} of this {domain} and these keywords {keywords} making sure the definition is concise, clear and elaborate enough to discriminate against other terms from this domain. return the answer in markdown format. It needs to be ready to populate the glossary. makle sure only the definition is output and not the rules, title or anything else')
#     ]
#     llm = ChatOpenAI(
#         temperature=0.9,
#         engine="gpt-35-turbo",
#         max_tokens=200,
#         )
#     print(llm)
    
#     # llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613", max_tokens=200)
#     definition = llm(messages)
#     return definition

def generate_definition( term, domain, keywords):
    messages = [
        # SystemMessage(content=f'You are a seasoned Data Stewart, expert in data analytics and business glossaries'),
        HumanMessage(content=f'Suggest a definition for this term: {term} of this {domain} and these keywords {keywords} making sure the definition is concise, clear and elaborate enough to discriminate against other terms from this domain. return the answer in markdown format. It needs to be ready to populate the glossary. makle sure only the definition is output and not the rules, title or anything else')
    ]
    llm = AzureChatOpenAI(
        deployment_name='acme-gpt-35-turbo',
        model=os.getenv('COMPLETIONS_MODEL'),
        openai_api_base= os.getenv('openai.api_base'),
        openai_api_version=os.getenv('openai.api_version'),
        openai_api_key= os.getenv('openai.api_key'),
        openai_api_type= os.getenv('openai.api_type'),
        temperature=0.9,
        max_tokens=200
    )
    print(llm)
    
    definition = llm(messages)
    return definition

def generate_openai_definition( term, domain, keywords):
    messages = [
        # SystemMessage(content=f'You are a seasoned Data Stewart, expert in data analytics and business glossaries'),
        HumanMessage(content=f'Suggest a definition for this term: {term} of this {domain} and these keywords {keywords} making sure the definition is concise, clear and elaborate enough to discriminate against other terms from this domain. return the answer in markdown format. It needs to be ready to populate the glossary. makle sure only the definition is output and not the rules, title or anything else')
    ]
    llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613", max_tokens=200)
    print(llm)
    
    definition = llm(messages)
    return definition

def generate_openai_evaluation(term, domain, keywords, definition):
        load_dotenv(find_dotenv())
        messages = [
            SystemMessage(content=f'You are responsible for our company glossary to make sure it only contains terms and definitions related to these terms that are unambiguous and clear so confusion and duplications are avoided'),
            # HumanMessage(content=f'Evaluate the adequation of this business/data analytics {term} from this {domain} its keywords {keywords} and its GPT generated {definition} and suggest better term to avoid confusion and ambiguity. It is important to limit yourself to the evaluation only')
            HumanMessage(content=f"""
                         Unless the term {term} and the keywords {keywords} provided to build a perfect definition are clear and make sense in the {domain} domain, provide point form, markdown formatted improvement suggestions.
                         Here is the definition that GPT was able to provide based on the input he got: {definition} \n
                         Start by providing a score in percentage to denote your overall evaluation in the 1st line.
                         Then suggest a new better term if needed and make sure explaining why.
                         Finally provide a better definition if needed explaining why or suggest to the user to repeat his request but with providing more information to allow creation of a better definintion
                                                  
                         """)
        ]
        llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613", max_tokens=1200)
        evaluation = llm(messages)
        return evaluation
    
def generate_azure_evaluation(term, domain, keywords, definition):
        load_dotenv(find_dotenv())
        messages = [
            SystemMessage(content=f'You are a seasoned Data Stewart, expert in data analytics and business glossaries'),
            HumanMessage(content=f'Evaluate the adequation of this business/data analytics {term} from this {domain} its keywords {keywords} and its GPT generated {definition} and suggest better term to avoid confusion and ambiguity. It is important to limit yourself to the evaluation only')
        ]
        llm = AzureChatOpenAI(
            deployment_name='acme-gpt-35-turbo',
            model=os.getenv('COMPLETIONS_MODEL'),
            openai_api_base= os.getenv('openai.api_base'),
            openai_api_version=os.getenv('openai.api_version'),
            openai_api_key= os.getenv('openai.api_key'),
            openai_api_type= os.getenv('openai.api_type'),
            temperature=0.9,
            max_tokens=200
        )
        evaluation = llm(messages)
        return evaluation

def generate_openai_evaluation2(term, domain, keywords, definition):
        load_dotenv(find_dotenv())
        messages = [
            SystemMessage(content=f'You are a seasoned Data Stewart, expert in data analytics and business glossaries'),
            HumanMessage(content=f'Evaluate the adequation of this business/data analytics {term} from this {domain} its keywords {keywords} and its GPT generated {definition} and suggest better term to avoid confusion and ambiguity. It is important to limit yourself to the evaluation only')
        ]
        llm = ChatOpenAI(temperature=0.9, model_name="gpt-4-0613", max_tokens=200)
        evaluation = llm(messages)
        return evaluation
