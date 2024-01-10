from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import GCSFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.auth
from google.cloud import storage
from modules.vectorstore import VectorStore
from modules.pkl import Pkl
from langchain.document_loaders import TextLoader
from langchain.text_splitter import SpacyTextSplitter
from dotenv import load_dotenv
import os



class DocumentReader():
    def __init__(self):
        load_dotenv()
        self.credentials, self.project = google.auth.default()
        # Create a list of all the blobs in the Storage Bucket
        self.GCS_STORAGE_BUCKET = os.getenv("GCS_STORAGE_BUCKET")
        self.bucket = storage.Client().get_bucket(self.GCS_STORAGE_BUCKET)
        self.vectorstore = VectorStore()
        self.pkl = Pkl()
    
    def split_pdf(self,new_blobs, updated_blobs, deleted_blobs):
           #Use get_blobs to get the new, updated, and deleted blobs
        #new_blobs, updated_blobs, deleted_blobs = self.pkl.get_blobs()
        # Create a list of loaders
        loaded_docs = ""
        loader = ""
        # Process new, updated, and deleted blobs
        if new_blobs is not None:
            for blob_name in new_blobs | updated_blobs:
                try:
                    print(f"Processing blob: {blob_name}")
                    # Create a list of loaders
                    loader = (GCSFileLoader(project_name=self.project, bucket=self.bucket, blob=blob_name, loader_func=PyPDFLoader))
                    loaded_docs = loader.load()
                    print(f"Number of docs in PDF:  {len(loaded_docs)}\n")
                    # Add document name and source to the metadata
                    for doc in loaded_docs:
                        doc_md = doc.metadata
                        document_name = doc_md['source'].split('/')[-1]
                        source = f"gs://{self.GCS_STORAGE_BUCKET}/{document_name}"
                        doc.metadata = {"source": source, "document_name": document_name}
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
                    print(f"Number of Chuncks from PDF:  {len(doc_splits)}\n")
                    self.embbed_docs(doc_splits, blob_name)
                except Exception as e:
                    print(f"An error occurred during document loading: {e} on file {blob_name}")
            for blob_name in deleted_blobs:
                print(f"Blob {blob_name} was deleted")

    
    def embbed_docs(self, docs, blob_name):
        try:
            if None not in (docs, blob_name):
                #print(f"Split Text:  {docs}\n")
                print(f"Number of Chunks from PDF:  {len(docs)}\n")
                # Add the docs to the vectorstore
                print(f"Successfully loaded {blob_name}")
                print(f"Vectorizing document {blob_name}")
                self.vectorstore.use_vector_search(docs, blob_name)
                
        except Exception as e:
            print(f"An error occurred during document embedding: {e} on file {blob_name}")

        #return self.summarize_docs(docs, blob_name)
    def split_text(self, new_blobs, updated_blobs, deleted_blobs):
        #Use get_blobs to get the new, updated, and deleted blobs
        new_blobs, updated_blobs, deleted_blobs = self.pkl.get_blobs()

        # Process new, updated, and deleted blobs
        if new_blobs is not None:
            for blob_name in new_blobs | updated_blobs:
                try:
                    print(f"Processing blob: {blob_name}")
                    # Create a list of loaders
                    loader = (GCSFileLoader(project_name=self.project, bucket=self.bucket, blob=blob_name, loader_func=TextLoader))
                    documents = loader.load()
                    print(f"Number of docs in PDF:  {len(documents)}\n")
                    # Add document name and source to the metadata
                    # Split the docs
                    text_splitter = SpacyTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separator="|",
                        )
                    texts = []
                    for doc in text_splitter.split_documents(documents):
                        texts.extend(doc.page_content.split("|"))
                    texts = [e.strip() for e in texts]
                    #  Normic Atlas database
                    self.embbed_texts(texts, blob_name) 

                except Exception as e:
                    print(f"An error occurred during document loading: {e} on file {blob_name}")
            for blob_name in deleted_blobs:
                print(f"Blob {blob_name} was deleted")
    
    def embbed_texts(self, texts, blob_name):
        try:
            if None not in (texts, blob_name):
                #print(f"Split Text:  {docs}\n")
                # Add the docs to the vectorstore
                print(f"Successfully loaded {blob_name}")
                print(f"Vectorizing document {blob_name}")
                self.vectorstore.use_nomic_atlas(texts, blob_name)
                
        except Exception as e:
            print(f"An error occurred during document embedding: {e} on file {blob_name}")

        #return self.summarize_docs(docs, blob_name)

    def summarize_docs(self, docs, blob_name):
        print(f"Summarizing document {blob_name}")
        #self.summarizer.summarize(docs, blob_name)
        #print(f"Successfully summarized {blob_name}")
