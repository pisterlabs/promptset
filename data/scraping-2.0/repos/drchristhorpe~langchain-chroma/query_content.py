from typing import Dict

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

import toml
import os
import json


# load config
config = toml.load('config.toml')
content_folder = config["CONTENT_FOLDER"]

filename_map = {}

# load the filename map from the content folder
with open(f"{content_folder}/filename_map.json", "r") as f:
    filename_map = json.load(f)

def setup_query() -> RetrievalQA:
    """
    This function sets up the query environment

    Returns:
        RetrievalQA: The retriever
    """
    # set env vars for openai
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

    # set the persist directory to the value in the config
    persist_directory = config["PERSIST_DIR"]

    # create the embedding function
    embedding = OpenAIEmbeddings()

    # Now we can load the persisted database from disk, and use it as normal. 
    vectordb = Chroma(persist_directory=persist_directory, 
                    embedding_function=embedding)

    # create the retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    return retriever


def respond_to_query(query) -> Dict:
    """
    This function responds to a query with an answer object

    Args:
        query (str): The query to respond to

    Returns:
        Dict: The answer object
    """

    # get the retriever
    retriever = setup_query()

    # create the answer object
    answer = {
        'query': query,
        'search_type': retriever.search_type
    }

    # query ChromaDB
    docs = retriever.get_relevant_documents(query)

    # add the docs to the answer object
    answer['docs'] = docs


    # create the chain that will be used to summarise the docs and "answer the question"
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                    chain_type="stuff", 
                                    retriever=retriever, 
                                    return_source_documents=True)

    # get the answer from the chain
    llm_response = qa_chain(query)

    # add the answer to the answer object
    answer['llm_response'] = llm_response['result']

    source_list = []

    # get the sources from the source documents in the llm response
    source_list = list(set([document.metadata['source'] for document in llm_response["source_documents"] if document.metadata['source'] is not None and document.metadata['source']]))

    # add the sources to the answer object
    answer['sources'] = source_list

    # add the urls to the answer object, these are mapped from the sources (which are text files scraped from the urls)
    answer['links'] = [filename_map[source] for source in source_list]
    
    # return the answer object
    return answer


if __name__ == "__main__":
    print(respond_to_query('What is a Th1 cell?'))


