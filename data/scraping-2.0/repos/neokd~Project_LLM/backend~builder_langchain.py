import os
import chromadb
from vector_builder.folder_structure import folder_structure_class
from vector_builder.db_ingest import content_loader_class
from vector_builder.detect_changes import detect_changes_class
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from huggingface_hub import hf_hub_download
from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from config import (SOURCE_DIRECTORY,STRUCTURE_DIRECTORY,PERSIST_DIRECTORY,EMBEDDING_MODEL_NAME,DEVICE_TYPE,N_GPU_LAYERS,MODEL_PATH,MODEL_FILE,MODEL_NAME,MAX_NEW_TOKENS)

root_directory = os.path.basename(os.path.normpath(SOURCE_DIRECTORY))
content_loader_object = content_loader_class()

def create_json_structure(folder_structure_object,detect_changes_object):
    if os.path.exists(STRUCTURE_DIRECTORY):
        update_json_structure(folder_structure_object,detect_changes_object)
    else:
        folder_structure=folder_structure_object.create_folder_structure_json(SOURCE_DIRECTORY)
        folder_structure_object.write_json_file(folder_structure,STRUCTURE_DIRECTORY)
        folder_json_data=folder_structure_object.read_json_file(STRUCTURE_DIRECTORY)
        subfolders, files_paths = folder_structure_object.extract_subfolder_and_files(folder_json_data[root_directory])

        vector_db_new_creation(content_loader_object,subfolders,files_paths)
        print("Json structure is created with necessary files")
        return None

def update_json_structure(folder_structure_object,detect_changes_object):
    previous_json_structure = folder_structure_object.read_json_file(STRUCTURE_DIRECTORY)
    current_json_structure = folder_structure_object.create_folder_structure_json(SOURCE_DIRECTORY)

    if previous_json_structure!=current_json_structure:
        creation_subfolders=[]
        creation_files_count=0
        subfolders_prev, files_paths_prev = detect_changes_object.get_folder_data(folder_structure_object, previous_json_structure)
        subfolders_curr, files_paths_curr = detect_changes_object.get_folder_data(folder_structure_object, current_json_structure)
        added_subfolders, removed_subfolders, added_files, deleted_files = detect_changes_object.changes(
        subfolders_prev, files_paths_prev, subfolders_curr, files_paths_curr)

        if added_subfolders:
            creation_subfolders+=added_subfolders
        for key,value in added_files.items():
            if value:
                creation_files_count+=1
                creation_subfolders.append(key)
        
        creation_subfolders=list(set(creation_subfolders))

        if creation_subfolders:
            vector_db_file_creation(content_loader_object,creation_subfolders,added_files,creation_files_count)
        if removed_subfolders:
            vector_db_deletion(removed_subfolders)
   
        folder_structure_object.write_json_file(current_json_structure, STRUCTURE_DIRECTORY)
        print("The json folder structure has been updated !!!")

    else:
        print("No changes is detected in folder structure !!!")
    return None
    
def vector_db_new_creation(content_loader_object,subfolders,files_paths):
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    for subfolder in subfolders:
        document_contents = content_loader_object.load_documents(files_paths[subfolder])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        #print(document_contents)
        texts = text_splitter.split_documents(document_contents)

        db = Chroma.from_documents(texts, embeddings, collection_name=subfolder, persist_directory=PERSIST_DIRECTORY,  client=client)
        db.persist()
        collection = db.get()
        files_added=[metadata['source'] for metadata in collection['metadatas']]
        db = None
        print(f"{list(set(files_added))} are in vector DB")
        print("new Vector DB has been created !!!")
    return None
def vector_db_file_creation(content_loader_object,subfolders,files_paths,count=1):
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if count:
        for subfolder in subfolders:
            document_contents = content_loader_object.load_documents(files_paths[subfolder])
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(document_contents)
            
            db = Chroma(collection_name=subfolder, persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client=client)

            db.add_documents(texts)           

            print("File document added to vector DB !!!")
        
    else:
        for subfolder in subfolders:
            client.get_or_create_collection(name=subfolder)
        print('collection is created !!!')

    db.persist()
    collection = db.get()
    files_added=[metadata['source'] for metadata in collection['metadatas']]
    db = None
    print(f"{list(set(files_added))} are in vector DB")

def vector_db_deletion(subfolders):
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    for subfolder in subfolders:
        db = Chroma(collection_name=subfolder, persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client=client)
        db.delete_collection()

        print(f'{subfolder} collection has been deleted !!!')

if __name__ == "__main__":
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    folder_structure_object = folder_structure_class()
    content_loader_object = content_loader_class()
    detect_changes_object = detect_changes_class()
    root_directory = os.path.basename(os.path.normpath(SOURCE_DIRECTORY))

    if os.path.exists(STRUCTURE_DIRECTORY):
        update_json_structure(folder_structure_object,detect_changes_object)
    else:
        create_json_structure(folder_structure_object,detect_changes_object)