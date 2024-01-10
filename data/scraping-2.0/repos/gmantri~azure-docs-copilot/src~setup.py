import os
import shutil

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from file_utils import get_files_with_extension

# load environment variables
load_dotenv()

# set path of chroma db
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
continue_with_setup = True
if os.path.exists(CHROMA_DB_PATH) and os.path.isdir(CHROMA_DB_PATH):
    input_prompt = """
    Vector store is already initialized. If you continue with the setup,  
    existing store will be deleted and a new store will be created. 
    Do you wish to continue (yes/no): [yes]
    """
    user_selection = input(input_prompt).lower()

    if user_selection == "yes" or user_selection == "":
        # delete existing database if needed.
        shutil.rmtree(CHROMA_DB_PATH)
    else:
        continue_with_setup = False

if not continue_with_setup:
    print("Exiting setup.")
    exit()

print("Setting up vector store. It might take some time.")
print("-------------------------------------------------")

# get the list of markdown files
files = get_files_with_extension("data", ".md")

AZURE_OPENAI_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")
AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME")

# create Open AI Embedding
# chunk_size must be 1 because of current limitation in Azure Open AI
embedding = OpenAIEmbeddings(deployment=AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT_NAME,
                             model=AZURE_OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)

# instantiate chroma db
chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding)

markdown_splitter = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2")])
for file in files:
    with open(file) as f:
        try:
            documents_for_vector_store = []
            file_contents = f.read()
            file_chunks = markdown_splitter.split_text(file_contents)
            for file_chunk in file_chunks:
                d = Document(page_content=file_chunk.page_content, metadata={"source": file})
                documents_for_vector_store.append(d)
            chroma_db.add_documents(documents_for_vector_store)
            message = f"file: {file} added to vector store."
            print(message)
        except Exception:
            print(f"error occurred while processing {file} file.")
            print(Exception)

print(f"{len(files)} files added to vector store")
