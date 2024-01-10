import os


from langchain import OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path


load_dotenv(Path("./.env"))
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
loader = TextLoader('context.txt','utf-8')
embeddings = OpenAIEmbeddings()



index = VectorstoreIndexCreator(
    # split the documents into chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,),
    # select which embeddings we want to use
    embedding=embeddings,
    # use Chroma as the vectorestore to index and search embeddings
    vectorstore_cls=Chroma,
    vectorstore_kwargs={"persist_directory": "vectorStore", "collection_name":"my_collection"}
).from_loaders([loader])


