import os
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Google Drive access
folder_id = os.environ['SOURCE_FOLDER_ID']
loader = GoogleDriveLoader(
    folder_id=folder_id,
    recursive=True
)
google_documents = loader.load()

# TODO: prevent large memory allocations by optimising this for smaller batches
docs, metadatas = [], []

# The text splitter we use
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=0, separators=[" ", ",", "\n"]
        )

# create embeddings and metadata for each doc
for gdoc in google_documents:        
    splits = text_splitter.split_text(gdoc.page_content) 
    docs.extend(splits)
    source_ref = gdoc.metadata['source']
    metadatas.extend([{"source": source_ref}] * len(splits))
    print(f"Split {source_ref} into {len(splits)} chunks")

store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)

# persists the store for usage during inference
store.save_local('faiss_index-'+folder_id)