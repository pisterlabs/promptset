import os
from typing import Any, Dict, List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.schema.document import Document
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
import pinecone
from backend.consts import INDEX_NAME
from dotenv import load_dotenv

load_dotenv()

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT_REGION'],
)

def chat_chain(query: str, namespace: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=namespace
    )

    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
        model_name='gpt-3.5-turbo'
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )

    return qa({'question': query, 'chat_history': chat_history})

def summary_chain(docs: List[Document]):
    llm = ChatOpenAI(verbose=True, temperature=0, model_name='gpt-3.5-turbo-16k')

    first_prompt_template = (
        'Write a summary of the following first part of YouTube video transcript:\n'
        '{text}\n'
        'SUMMARY:\n'
    )
    first_prompt = PromptTemplate.from_template(first_prompt_template)

    next_prompt_template = (
        'Write a summary of the following next part of YouTube video transcript:\n'
        '{text}\n'
        'SUMMARY:\n'
    )
    next_prompt = PromptTemplate.from_template(next_prompt_template)

    llm_first_chain = LLMChain(llm=llm, prompt=first_prompt)
    llm_next_chain = LLMChain(llm=llm, prompt=next_prompt)

    stuff_first_chain = StuffDocumentsChain(llm_chain=llm_first_chain, document_variable_name='text')
    stuff_next_chain = StuffDocumentsChain(llm_chain=llm_next_chain, document_variable_name='text')

    result = stuff_first_chain.run((docs[0],))
    for doc in docs[1:]:
        result += ' '
        result += stuff_next_chain.run((doc,))

    return result