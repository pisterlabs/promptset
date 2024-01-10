import getpass
import os
from langchain.document_loaders import TextLoader
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
import pickle
from langchain.chains import VectorDBQA


# Read list to memory
def read_list():
    # for reading also binary mode is important
    with open('/data/urlcontent.pickle', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list
    
os.environ["OPENAI_API_KEY"] = "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"
persist_directory = '/data/CHROMA_DB'

documents=read_list()
# loader = TextLoader("/data/url_content.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=0,
                                               separators=[" ", ",", "\n"])

all_chunk_embeddings=[]
for doc in documents:
    chunks = text_splitter.split_documents(doc)
    all_chunk_embeddings.extend(chunks)  
     
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(all_chunk_embeddings,
                           embedding=embeddings,
                           persist_directory=persist_directory)

# vectordb=Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory=persist_directory)
vectordb.persist()

# query = "Fogify"
# matching_docs = vectordb.similarity_search(query)
# print(matching_docs[0])
