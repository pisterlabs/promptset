import os
from dotenv import load_dotenv
import re

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import (
    RetrievalQA,
    ConversationalRetrievalChain
)
from langchain.document_loaders import (
    TextLoader,
    YoutubeLoader
)


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from urllib.parse import urlparse
import requests

def extract_youtube_id(url):
    youtube_id_match = re.search(r'(?<=v=)[^&#]+', url)
    youtube_id_match = youtube_id_match or re.search(r'(?<=be/)[^&#]+', url)
    return youtube_id_match.group(0) if youtube_id_match else None

def get_youtube_transcript(url,language_code="en"):
    loader = YoutubeLoader.from_youtube_url(
        url,        
        add_video_info=True,
        language=[language_code, "id"],
        translation=language_code,)
    docs=loader.load()
    return docs[0].page_content

def vectordb_exists(persist_directory):
    return os.path.exists(persist_directory)

def create_vectordb_from_file(filename,persist_directory,embedding,overwrite=False,chunk_size=1200,chunk_overlap=200):
    if vectordb_exists(persist_directory)==False or overwrite==True:
        #print("creating vectordb")

        if filename.endswith('.txt'):
            try:
                loader = TextLoader(filename,encoding="utf-8")
                documents = loader.load()
            except Exception as e:
                raise ValueError(f"Non è possibile caricare il file {filename}: {e}")
        elif filename.endswith('.pdf'):
            try:
                loader = PyPDFLoader(filename)
                documents = loader.load()
            except Exception as e:
                raise ValueError(f"Non è possibile caricare il file {filename}: {e}")
        else:
            raise ValueError(f"Etensione del file {filename} non riconosciuta: {e}")

        if not documents:
            raise ValueError(f"Il documento {filename} è vuoto o non valido: {e}")

        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, 
            length_function=len
        )
        splits = r_splitter.split_documents(documents)

        # persist to vectordb: in a notebook, we should call persist() to ensure the embeddings are written to disk
        # This isn't necessary in a script: the database will be automatically persisted when the client object is destroyed
        return Chroma.from_documents(
            documents=splits, 
            embedding=embedding, 
            persist_directory=persist_directory
        )
    else:
        return load_vectordb(persist_directory=persist_directory,embedding=embedding)

def create_vectordb_from_files(files, persist_directory, embedding, overwrite=False, chunk_size=1200, chunk_overlap=200):
    loaders = []
    
    for filename in files:
        if filename.endswith('.txt'):
            try:
                loader = TextLoader(filename, encoding="utf-8")
                loaders.append(loader)
            except Exception as e:
                raise ValueError(f"Non è possibile caricare il file {filename}: {e}")
        elif filename.endswith('.pdf'):
            try:
                loader = PyPDFLoader(filename)
                loaders.append(loader)
            except Exception as e:
                raise ValueError(f"Non è possibile caricare il file {filename}: {e}")
        else:
            raise ValueError(f"Estensione del file {filename} non riconosciuta.")
    
    docs = []
    for loader in loaders:
        try:
            documents = loader.load()
            if not documents:
                raise ValueError(f"Il documento {filename} è vuoto o non valido.")
            docs.extend(documents)
        except Exception as e:
            raise ValueError(f"Errore durante il caricamento del file {filename}: {e}")
    
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, 
        length_function=len
    )
    splits = r_splitter.split_documents(docs)

    return Chroma.from_documents(
        documents=splits, 
        embedding=embedding, 
        persist_directory=persist_directory
    )

def create_vectordb_from_texts(texts,persist_directory,embedding,overwrite=False):
    if vectordb_exists(persist_directory)==False or overwrite==True:
        print("creating vectordb")
        return Chroma.from_texts(
            texts=texts, 
            embedding=embedding,
            persist_directory=persist_directory)
    else:
        return load_vectordb(persist_directory=persist_directory,embedding=embedding)

def load_vectordb(persist_directory,embedding):
    if vectordb_exists(persist_directory):
        #print("vectordb already exists")
        return Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        raise ValueError(f"VectorDB does not exist in {persist_directory}")

def save_file(location,language_code="en"):
    #https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/PT717-Transcript.pdf
    #https://www.gutenberg.org/cache/epub/1934/pg1934.txt
    save_directory="./files"
    parsed = urlparse(location)
    if "youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc:
        transcript_content = get_youtube_transcript(location,language_code=language_code)
        video_id = extract_youtube_id(location)
        if video_id:
            local_filename = os.path.join(save_directory, f"{video_id}.txt")
        else:
            local_filename = os.path.join(save_directory, "youtube_transcript.txt")
        
        with open(local_filename, 'w', encoding="utf-8") as f:
            f.write(transcript_content)
        return local_filename
    elif bool(parsed.netloc):
        local_filename = os.path.join(save_directory, os.path.basename(location))

        if not os.path.exists(local_filename) or overwrite:
            with requests.get(location, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    else:
        if os.path.exists(location):
            local_filename=location
        else:
            raise FileNotFoundError(f"File '{location}' not found.")
    
    return local_filename

def do_query(vectordb,llm,query):
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectordb.as_retriever()
    )    
    return qa.run(query)

def do_query(vectordb,llm,memory,query):
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectordb.as_retriever(),
        memory=memory
    )    
    return qa.run(query)