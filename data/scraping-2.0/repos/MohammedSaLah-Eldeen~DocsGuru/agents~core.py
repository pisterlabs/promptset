"""
defines the basic prompting functionality
"""
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from typing import List, Dict
import os
from dotenv import load_dotenv
load_dotenv()

def chat(query: str, vectorstore, chat_history: List[Dict]):
    """
    defines the Q/A chain to accept queries.

    Args:
        query (str): the query sent to the llm. 
    """
    user_history = [message['content'] for message in chat_history if message['role'] == 'user']
    assistant_history = [message['content'] for message in chat_history if message['role'] == 'assistant']
    chat_history = list(zip(user_history, assistant_history))

    llm = ChatOpenAI(temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    prompt = {
        'question': query,
        'chat_history': chat_history
    }

    return qa(prompt)


def format_sources_string(sources: List[str]) -> str:
    """
    formats the sources links returned by LLM
    in a friendly format.
    """
    if not sources:
        return ""
    
    formatted = "Here, check the relevant links:\n\n"
    for numbering, source in enumerate(sources):
        formatted += f"{numbering}. {source}\n"

    return formatted