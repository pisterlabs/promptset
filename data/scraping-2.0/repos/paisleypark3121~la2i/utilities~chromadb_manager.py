import os
import re
from dotenv import load_dotenv

import io
import fitz
import pdfplumber

import requests

from youtube_transcript_api import YouTubeTranscriptApi

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def get_local_text(filename):
    with open(filename, "r", encoding="utf-8") as local_file:
        return local_file.read()
    return None
# filename="files/jokerbirot_space_musician_en.txt"
# content=get_local_text(filename)
# print(content[0:30])

def get_remote_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text        
        #contenuto_file=response.text.encode('utf-8')
    return None
#url="https://www.gutenberg.org/cache/epub/61830/pg61830.txt"
# url="https://raw.githubusercontent.com/paisleypark3121/la2i/main/files/jokerbirot_space_musician_en.txt"
# content=get_remote_text(url)
# print(content[0:30])

def get_local_pdf(filename):
    pdf_text = ""
    with fitz.open(filename) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pdf_text += page.get_text() + "\n"  # Aggiungi un ritorno a capo tra le pagine
    return pdf_text
    return None
# filename="files/DHCP.pdf"
# content=get_local_pdf(filename)
# print(content)

def get_remote_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text()
            return pdf_text.strip()
    return None
# url="https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/PT741-Transcript.pdf"
# content=get_remote_pdf(url)
# print(content[0:1000])

def extract_youtube_id(url):
    youtube_id_match = re.search(r'(?<=v=)[^&#]+', url)
    youtube_id_match = youtube_id_match or re.search(r'(?<=be/)[^&#]+', url)
    return youtube_id_match.group(0) if youtube_id_match else None

def get_youtube_transcript(url,language_code="en"):
    id=extract_youtube_id(url)
    transcript_list = YouTubeTranscriptApi.list_transcripts(id)
    # for transcript in transcript_list:
    #     print(
    #         transcript.video_id,
    #         transcript.language,
    #         transcript.language_code,
    #     )
    transcript = transcript_list.find_transcript([language_code])
    full_transcript = transcript.fetch()
    
    concatenated_text = ""
    for item in full_transcript:
        concatenated_text += item['text'] + ' '
    concatenated_text = concatenated_text.strip()

    return concatenated_text
# url="https://www.youtube.com/watch?v=AgZCmiC4Zr8"
# content=get_youtube_transcript(url,language_code="it")
# print(content[0:100])
# url="https://www.youtube.com/watch?v=O0dUOtOIrfs"
# content=get_youtube_transcript(url)
# print(content[0:100])

def create_vectordb(location,embedding):
    
    if location.startswith("http"):
        if location.endswith(".txt"):
            content=get_remote_text(location)
        elif location.endswith(".pdf"):
            content=get_remote_pdf(location)
        elif "youtube.com" in location:
            content=get_youtube_transcript(location)
        else:
            content=None
    elif os.path.isfile(location):
        if location.endswith(".txt"):
            content=get_local_text(location)
        elif location.endswith(".pdf"):
            content=get_local_pdf(location)
    else:
        content=None

    if content is None:
        return None

    if len(content) < 5000:
        chunk_size = 500
        chunk_overlap = 50
    else:
        # Use the default values when the length is not smaller than 5000
        chunk_size = 1200
        chunk_overlap = 200
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, 
        length_function=len
    )
    
    texts = text_splitter.split_text(content)
    splits = [Document(page_content=t) for t in texts]
    
    vectordb=Chroma.from_documents(
        documents=splits, 
        embedding=embedding, 
        collection_metadata={"hnsw:space": "cosine"},
    )

    return vectordb

def create_vectordb_from_content(content,embedding):
    
    if content is None:
        return None

    if len(content) < 5000:
        chunk_size = 500
        chunk_overlap = 50
    else:
        # Use the default values when the length is not smaller than 5000
        chunk_size = 1200
        chunk_overlap = 200
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, 
        length_function=len
    )
    
    texts = text_splitter.split_text(content)
    splits = [Document(page_content=t) for t in texts]
    
    vectordb=Chroma.from_documents(
        documents=splits, 
        embedding=embedding, 
        collection_metadata={"hnsw:space": "cosine"},
    )

    return vectordb
# load_dotenv()
# #location="files/jokerbirot_space_musician_en.txt"
# location="files/DHCP.pdf"
# #location="https://raw.githubusercontent.com/paisleypark3121/la2i/main/files/jokerbirot_space_musician_en.txt"
# #location="https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/PT741-Transcript.pdf"
# #location="https://www.youtube.com/watch?v=O0dUOtOIrfs"
# embeddings=OpenAIEmbeddings()
# vectordb=create_vectordb(location,embeddings)
# #print(vectordb)
# retriever=vectordb.as_retriever()
# prompt="who is DHCP?"
# docs=retriever.get_relevant_documents(prompt)
# print(docs[0])