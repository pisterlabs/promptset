from functools import lru_cache
import pickle
import os
from dotenv import load_dotenv

from performance import timeit, load_vector_store, handle_request, log_request

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI


from get_vectorestore import download_from_gcs

@lru_cache(maxsize=None)
def get_vector_store(bucket_name, blob_name, local_file_name):
    # Last ned og returner VectorStore-objektet
    return download_from_gcs(bucket_name, blob_name, local_file_name)



def modify_query_with_instruction(query, instruction):
    return f"{instruction}\n\n{query}"


intruction = """
    Vær en høflig chatbot assistent som representere COAX AS på best mulig vis. 
    Ditt formål er å hjelpe kunden med å finne svar på spørsmål de måtte ha.
    Du skal ikke gi kunden informasjon som ikke er relevant for spørsmålet de stiller.
    Du skal ikke gi kunden informasjon som ikke er relevant for COAX AS.
    Avslutt alle samtaler med "Er det noe mer du lurer på?".
    Svar på følgende spørsmål:
    """

def get_response(query):

    load_dotenv()


    # Last ned vektordatabasen ved oppstart
    VectorStore = download_from_gcs()

    ## Laster inn vektordatabasen fra minnet i stedet for å laste den inn fra fil på disk
    #VectorStore = get_vector_store(bucket_name, blob_name, local_file_name)
    
    ## Søker i vektordatabasen etter de 3 mest relevante dokumentene som matcher spørringen
    docs = VectorStore.similarity_search(query, k=3) ## k kan endres for å få flere eller færre dokumenter

    modified_query = modify_query_with_instruction(query,intruction)


    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature=0)

    ## Laster inn en kjede som kan svare på spørsmål
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    
    with get_openai_callback() as callback:
        response = chain.run(input_documents=docs, question=modified_query)
    return response, callback 

#response, callback = get_response("Hva er coax?", "COAX_web_content.pkl")
#print((callback.total_cost))