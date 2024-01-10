
# import
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader


import os
import toml


# load config
config = toml.load('config.toml')
content_folder = config["CONTENT_FOLDER"]

# set env vars for openai
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

# load the documents contained in the content folder
loader = DirectoryLoader(content_folder, glob='./*.txt', loader_cls=TextLoader)

print ('loading documents')
documents = loader.load()

print ('splitting documents')
#splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print (f"number of texts: {len(texts)}")

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = config["PERSIST_DIR"]

## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)

# persiste the db to disk
print ('persisting db')
vectordb.persist()

print ('done')
vectordb = None
