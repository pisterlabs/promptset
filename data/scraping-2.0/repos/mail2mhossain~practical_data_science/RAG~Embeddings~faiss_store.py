import os
import glob
from dotenv import load_dotenv

# text splitter for create chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader  # for textfiles
from langchain.text_splitter import CharacterTextSplitter  # text splitter

# to be able to load the pdf files
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

# LLamaCpp embeddings from the Alpaca model
from langchain.embeddings import LlamaCppEmbeddings

# FAISS  library for similaarity search
from langchain.vectorstores.faiss import FAISS

load_dotenv("../.env")

faiss_file = os.getenv("FAISS_FILE")
llama_path = os.getenv("LLAMA_PATH")

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)

# create the embedding object
embeddings = LlamaCppEmbeddings(model_path=llama_path)


# Split text
def split_chunks(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks


def create_index(chunks):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    search_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    return search_index


# PDF loader
pdf_folder_path = "../New_Documents/"
# os.listdir(pdf_folder_path)
pdf_files = glob.glob(os.path.join(pdf_folder_path, f"*{'.pdf'}"))
num_of_docs = len(pdf_files)

print("generating fist vector database and then iterate with .merge_from")
loader = UnstructuredPDFLoader(pdf_files[0])
docs = loader.load()
chunks = split_chunks(docs)
db0 = create_index(chunks)

for i in range(1, num_of_docs):
    print(pdf_files[i])
    print(f"loop position {i}")
    loader = UnstructuredPDFLoader(os.path.join(pdf_folder_path, pdf_files[i]))

    docs = loader.load()
    chunks = split_chunks(docs)
    dbi = create_index(chunks)
    print("start merging with db0...")
    db0.merge_from(dbi)

# Save the databasae locally
db0.save_local(faiss_file)
