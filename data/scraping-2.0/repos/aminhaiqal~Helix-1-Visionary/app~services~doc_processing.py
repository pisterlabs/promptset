from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def doc_processing(file_path):
    load_dotenv()
    
    pdf_reader = PdfReader(file_path)
        
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
            
    # split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
        
    # create embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_text(chunks, embeddings)    

    return knowledge_base, chunks
    