import argparse
import os
import sys
import time

import dotenv

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# takes in --promt and prints the response

#Define prompt parser
parser = argparse.ArgumentParser(description='Process prompt.')
parser.add_argument('--prompt', type=str, help='Prompt to use for the model')
question = parser.parse_args().prompt

print_debug = False





def print_texts(texts):
    if not print_debug:
        return
    print(f"Found {len(texts)} texts.")
    i=0
    for text in texts:
        i = i+1
        print(f"{i}:{text.metadata}")


def load_documents(repo_path, type):
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob=f"**/**/*",
        suffixes=[type],
        parser=LanguageParser()
    )

    new_documents = loader.load()

    if not print_debug:
        return new_documents

    print(f"Found {len(new_documents)} documents.")
    for new_document in new_documents:
        print(f"Document: {new_document.metadata}")
    
    return new_documents

def main():
    dotenv.load_dotenv('creds.env')

    working_dir = os.getcwd()

    repo_path = working_dir

    print(f"repo_path: {repo_path}")

    documents = []

    python_documents = load_documents(repo_path, ".py")

    markdown_documents = load_documents(repo_path, ".md")

    log_documents = load_documents(repo_path, ".log")

    print(f"Found {len(python_documents)} python documents.")
    print(f"Found {len(markdown_documents)} markdown documents.")
    print(f"Found {len(log_documents)} log documents.")

    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                                chunk_size=2000, 
                                                                chunk_overlap=200)
    markdown_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN,
                                                                        chunk_size=2000,
                                                                        chunk_overlap=200)

    log_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN,
                                                                        chunk_size=2000,
                                                                        chunk_overlap=200)
                                                                

    texts = python_splitter.split_documents(python_documents)


    texts += markdown_splitter.split_documents(markdown_documents)


    texts += log_splitter.split_documents(log_documents)
    print_texts(texts)

    db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
    retriever = db.as_retriever(
        search_type="mmr", # Also test "similarity"
        search_kwargs={"k": 8},
    )


    llm = ChatOpenAI(model_name="gpt-4", streaming=True, callbacks=[StreamingStdOutCallbackHandler()]) 
    #memory = ConversationSummaryMemory(llm=llm, chat_memory=chat_history)
    memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
    #memory = ConversationBufferMemory()
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)



    result = qa(question)

    answer = result['answer']
    print(answer)






main()