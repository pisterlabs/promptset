import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import nltk

from pathlib import Path
from print_color import print as printc

from db.chunking_strategies import patient, intranet

def get_collection(name:str):
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=f"""{os.path.dirname(os.path.dirname(__file__))}/db/storage""".replace('\\', '/')
    ))

    openai_api_key = os.getenv("OPENAI_API_KEY")

    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                )
    
    collection = client.get_or_create_collection(
            name=name, 
            embedding_function=emb_fn,
            metadata={"hnsw:space": "cosine"}
        )
    return collection

def get_client():
     return chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=f"""{os.path.dirname(os.path.dirname(__file__))}/db/storage""".replace('\\', '/')
    ))
     


def make_db_patients():
    """Creates a vector database that stores medical records of patients by scanning the folder of medical records."""

    print("--- Making new Collection: 'patientrecords' ---")
    
    collection = get_collection("patientrecords")

    #Start by finding every patients directory path, and re-format them if user uses windows path style
    
    dirs = [f.path for f in os.scandir("data/patient_records") if f.is_dir() ]
    #print(dirs)
    dirs = [d.replace("\\", "/") for d in dirs]

    

    n = len(dirs)

    #For every directory d we open it and read two files:
    #   * patientdata.json
    #   * patientcalender.ics
    #
    #For the two files we need two corresponding chunking strategies
    #   * patientdata.json: We maintain context by splitting the file into 
    #     the smallest parts of the json file possible that still contains the 
    #     patient ID.
    #     Specifically the parts 'prescription' and 'journal' mentions the ID for
    #     every element in their lists, so they get special treatment.
    #
    #   * patientcalender.ics: We split the document so that one chunk equals
    #     one event in the calendar, which should also have a mention of
    #     patient ID. Since the ICS format has no header/footer, and is just
    #     a list of calendar events this should be a reliable strategy.
    #
    #As usual we finish attach metadata tags and add to database along the way.

    for j, dir in enumerate(dirs):
        print(f"[{j+1}/{n}] 'patientrecords': {dir} processing ...", end="\r")

        for file in ["patientdata.json","patientcalendar.ics"]:
            file_path = f"{dir}/{file}"

            with open(file_path, 'r', encoding='utf-8') as f:
                
                if file == "patientdata.json":
                    chunks = patient.chunk_json(f)
                    patient.add_to_collection(
                         chunks=chunks,
                         collection=collection,
                         dir=dir,
                         filetype="json"
                    )
                     
                if file == "patientcalendar.ics":
                    chunks = patient.chunk_ics(f)
                    patient.add_to_collection(
                         chunks=chunks,
                         collection=collection,
                         dir=dir,
                         filetype="ics"
                    )
        
        
        printc(f"[{j+1}/{n}] 'patientrecords': {dir} Done!                 ", color="green")
    print("--- Collection Complete!: 'patientrecords' ---")

      
        
def make_db_docs():
    """Creates a vector database for the intranet by scanning through the intranet folder."""

    nltk.download("punkt") ## this should be inactive if punkt has been downloaded

    print("--- Making new Collection: 'docs' ---")
    collection = get_collection("docs")
    
    #First, we make a list of paths to all intranet pdfs and re-format them if user uses windows path style
    paths = [f.path for f in os.scandir("data/intranet_records") ]
    paths = [path.replace("\\", "/") for path in paths]
    
    n = len(paths)
    
    #For every path d we open the pdf file, read its contents using our parser,
    #and split it into using a splitter from LangChain. Please note that it
    #uses some overlap to reduce unlucky split, and will therefore result in
    #some redundancy in memory.
    #As usual we finish attach metadata tags and add to database along the way.
    for j,dir in enumerate(paths):
        print(f"[{j+1}/{n}] 'docs': {dir[-1:-2]} processing ...", end="\r")
        chunks = intranet.chunk_pdf(dir)
        intranet.add_to_collection(
            chunks=chunks,
            collection=collection,
            dir=dir
        )
        printc(f"[{j+1}/{n}] 'docs': {dir} Done!                             ", color="green")
    print("--- Collection Complete!: 'docs' ---")


def query_db_doc(query: str,  name: str, n_results: int = 5):

    collection = get_collection(name)
    ans = collection.query(
        query_texts= query,
        n_results= n_results
    )

    return ans

def query_db_with_id(query: str, id: str, name:str, n_results: int = 5):
    collection = get_collection(name)
    ans = collection.query(
        query_texts= query,
        where={"patient": id},
        n_results=n_results
    )

    return ans




#todo: if collection is huge this should be paralellized in e.g. pyspark
def get_biggest_chunk(name:str):
    """Helper function used in database summary. Finds biggest chunk of text in a collection."""
    collection = get_collection(name)

    ans = collection.get(
        include=["metadatas"]
    )
    
    ids = ans['ids']

    max_size = 0
    max_info = {}
    for i in range(len(ids)):
        if ans["metadatas"][i]["chunk_size"] > max_size:
            max_size = ans["metadatas"][i]["chunk_size"]
            max_info = ans["metadatas"][i]
    
    return max_size,max_info

def get_mean_chunk_size(name:str):
    """Helper function used in database summary. Finds the mean chunk size of a text in a collection."""
    collection = get_collection(name)

    ans = collection.get(
        include=["metadatas"]
    )
    
    
    n = collection.count()
    s = 0
    for i in range(n):
        s += ans["metadatas"][i]["chunk_size"]
            
    
    return s/n

def print_db_summary():
    """Gives you a summary of the collections you have with info like size of collections aswell as biggest and mean chunk size."""

    print("Database summary:")
    for c in get_client().list_collections():
        name = c.name
        collection = get_collection(name)
        n = collection.count()
        m = get_biggest_chunk(name)
        mean = get_mean_chunk_size(name)
        print(f"\t{name}:\n\tchunks: {n} \n\tbiggest chunk: {m}\n\tmean chunk size: {mean} \n")
    root_directory = Path('db/storage')
    s = sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file())
    print(f"Size: ~{round(s/10**6,2)} MB")


#if we open the database here, database initialization may break, please use only functions :-)

# - Eg remove things like this before initing:
# - print(get_collection("docs").peek())