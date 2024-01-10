
import lib_book_parse
import lib_llm
import lib_embeddings
import lib_vectordb

from pathlib import Path
import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from elasticsearch import Elasticsearch


config = {
    "bookName" : "Wookieepedia",
    "bookIndexName": "book_wookieepedia_mpnet",
    "bookFilePath": "starwars_all_canon_data_*.pickle"
}

bookName = config['bookName']
bookFilePath = config['bookFilePath']
index_name = config['bookIndexName']

# Huggingface embedding setup
hf = lib_embeddings.setup_embeddings()

## Elasticsearch as a vector db
db, url = lib_vectordb.setup_vectordb(hf,index_name)

## set up the conversational LLM
llm_chain_informed= lib_llm.make_the_llm()

## Load the book
# BOOK LOADED IN app-starwars_deployed.py


## how to ask a question
def ask_a_question(question):
    # print("The Question at hand: "+question)

    ## 3. get the relevant chunk from Elasticsearch for a question
    # print(">> 3. get the relevant chunk from Elasticsearch for a question")
    similar_docs = db.similarity_search(question)
    print(f'The most relevant passage: \n\t{similar_docs[0].page_content}')

    ## 4. Ask Local LLM context informed prompt
    # print(">> 4. Asking The Book ... and its response is: ")
    
    informed_context= similar_docs[0].page_content
    informed_response = llm_chain_informed.run(context=informed_context,question=question)
    
    return informed_response


# The conversational loop

print(f'I am the book, "{bookName}", ask me any question: ')

while True:
    command = input("User Question>> ")
    response= ask_a_question(command)
    print(f"\tAnswer from LLM with context : {response}")\

