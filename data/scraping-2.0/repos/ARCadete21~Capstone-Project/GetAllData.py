import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from util import local_settings
from langchain_community.document_loaders import DirectoryLoader, CSVLoader, PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
#import chromadb
import os


# Function that can be used for file preprocessing
def preprocess_pdf_files_for_LLM(path: str):
    pdf_docs = []
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            pdf_docs.append(os.path.join(path, file))
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

###################### MAIN ######################

loader_pdfs = DirectoryLoader('DataFiltered2022_2023', show_progress=True, glob='**/*.pdf', loader_cls=PyPDFLoader)
loader_csv = DirectoryLoader('DataFiltered2022_2023', show_progress=True, glob='**/*.csv', loader_cls=CSVLoader)

vector_store = VectorstoreIndexCreator(embedding=OpenAIEmbeddings(), text_splitter=RecursiveCharacterTextSplitter(),
                                       vectorstore_cls=FAISS).from_loaders([loader_pdfs, loader_csv])

#%%
