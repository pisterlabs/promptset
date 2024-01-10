from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma




dir = os.path.dirname(__file__) + "/data"
loader = DirectoryLoader( dir,  glob="**/*.pdf", loader_cls= PyPDFLoader) 
documents = loader.load()

print( "Loaded {} documents".format( len(documents) ) )
print( "First document: {}".format( documents[0] ) )



text_splitter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200, separators=["\n", "\r"] )
texts = text_splitter.split_documents( documents )

print( "Split {} documents into {} texts".format( len(documents), len(texts) ) )
print( "First text: {}".format( texts[0] ) )




model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)







persist_directory = "db"

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)