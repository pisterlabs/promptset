from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def embed(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # split document into chunks if of 1000 characters each
    split_docs = text_splitter.split_text(docs)
    embeddings = OpenAIEmbeddings()
    ## embed the documents
    db = embeddings.embed_documents(split_docs)

    return db
