import os
from dotenv import load_dotenv
import pickle
from io import BytesIO

from web_scrape import fetch_website_content, get_internal_links, scrape_all_pages
from performance import timeit, load_vector_store, handle_request, log_request


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def embedd_and_save(text):

    load_dotenv()

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
    

    ## teksten burde kanskje formateres og forbedres før den blir sendt til en VectorStore
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    with open("COAX_web_content.pkl", "wb") as f:
        pickle.dump(VectorStore, f)




text = scrape_all_pages("https://www.coax.no/") ## få GPT til å lage en ny og mer sammenhengende tekst?
#print(text)
embedd_and_save(text)
