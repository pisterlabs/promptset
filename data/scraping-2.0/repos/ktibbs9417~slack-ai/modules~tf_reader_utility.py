from langchain.document_loaders import JSONLoader
from langchain.document_loaders import GCSFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.auth
from google.cloud import storage
from modules.llm_library import LLMLibrary
from dotenv import load_dotenv
import os
import re



class TerraformReader():
    def __init__(self):
        load_dotenv()
        self.credentials, self.project = google.auth.default()
        # Create a list of all the blobs in the Storage Bucket

        self.llmlibrary = LLMLibrary()
    
    def get_tf_state(self, terraform_state_path):
        bucket_name = re.match(r"(?:gs://)?(.*?)/(.*)", terraform_state_path).group(1)
        terraform_state = re.findall(r"(?:gs://)?[^/]+/(.*\.tfstate)$", terraform_state_path)[0]
        gcs_bucket = storage.Client().get_bucket(bucket_name)
        blob = gcs_bucket.blob(terraform_state)
        blob.download_to_filename("tf_state.json")
        print (f"Bucket: {gcs_bucket}")
        print (f"State: {terraform_state}")

        TerraformReader.split_terraform(self, gcs_bucket, terraform_state)

    def load_json(self, file_path):
        return JSONLoader(file_path="tf_state.json",jq_schema='.resources[]', text_content=False)
    
    def split_terraform(self, gcs_bucket, terraform_state):
           #Use get_blobs to get the new, updated, and deleted blobs        # Create a list of loaders
        loaded_docs = ""
        loader = ""
        # Process new, updated, and deleted blobs
        try:
            print(f"Processing blob: {terraform_state}")
            # Create a list of loaders
            loader = (GCSFileLoader(project_name=self.project, bucket=gcs_bucket, blob=terraform_state, loader_func=self.load_json))
            loaded_docs = loader.load()
            print(f"Number of TF docs :  {len(loaded_docs)}\n")
            # Add document name and source to the metadata
            # Split the docs
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len,
                )
            #print(f"Loaded Text:  {loaded_texts}\n")
            doc_splits = text_splitter.split_documents(loaded_docs)
            for idx, split in enumerate(doc_splits):
                split.metadata['chunk'] = idx
            self.embbed_docs(doc_splits, terraform_state)
        except Exception as e:
            print(f"An error occurred during document loading: {e} on file {terraform_state}")


    
    def embbed_docs(self, docs, blob_name):
        try:
            if None not in (docs, blob_name):
                #print(f"Split Text:  {docs}\n")
                print(f"Number of Chunks from Terraform State:  {len(docs)}\n")
                # Add the docs to the FAISS
                print(f"Successfully loaded {blob_name}")
                print(f"Vectorizing document {blob_name}")
                self.llmlibrary.in_memory_index(docs)
                
        except Exception as e:
            print(f"An error occurred during document embedding: {e} on file {blob_name}")
