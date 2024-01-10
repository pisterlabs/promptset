import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Import OpenAI API Key
from APIKey import OpenAIKey
if OpenAIKey is None:
    raise ValueError("OPEN_API_KEY is not set. Create APIKey.py file that defines variable OpenAIKey")
os.environ["OPENAI_API_KEY"] = OpenAIKey

# Create langchain document from multiple PDF
multiDoc = []

for file in os.listdir('./sampleFilesAndWebsites/'):
    if file.endswith('.pdf'):
        pdfPath = './sampleFilesAndWebsites/' + file
        loader = PyPDFLoader(pdfPath)
        multiDoc.extend(loader.load())

    elif file.endswith('.docx') or file.endswith('.doc'):
        docPath = './sampleFilesAndWebsites/' + file
        loader = Docx2txtLoader(docPath)
        multiDoc.extend(loader.load())

    elif file.endswith('.txt'):
        txtPath = './sampleFilesAndWebsites' + file
        loader = TextLoader(txtPath)
        multiDoc.extend(loader.load())

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
chunks = text_splitter.split_documents(multiDoc)

#Convert to embeddings. Store embeddings from text chunks inside ./embeddings directory
vectordb = Chroma.from_documents(
    chunks,
    embedding = OpenAIEmbeddings(),
    persist_directory = './sampleFilesAndWebsites/embeddings'
    )

vectordb.persist()
