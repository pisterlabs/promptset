import os
import logging
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import GoogleDriveLoader, PyPDFLoader
from langchain.memory import ConversationBufferMemory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(name)s] %(message)s",
)

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = ""
# Set up ActiveLoop (DeepLake) API key
os.environ["DEEPLAKE_API_KEY"] = "xxx"

# Set up PyPDF Loader
loader = PyPDFLoader("classes/CLAS-151/syllabus.pdf")

# Load documents
logging.info("Loading documents...")
documents = loader.load_and_split()

# Split documents into chunks
logging.info("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Generate embeddings and create vectorstore
logging.info("Generating embeddings...")
embeddings = OpenAIEmbeddings()
db = DeepLake(dataset_path="deeplake", embedding_function=embeddings)
db.add_documents(texts)

# Create retrieval chain
logging.info("Creating retrieval chain...")
model = ChatOpenAI(model='gpt-3.5-turbo')
retriever = db.as_retriever()
qa = ConversationalRetrievalChain.from_llm(model, retriever)

# Start chat loop
chat_history = []
while True:
    query = input("Enter your question (or 'exit'): ")
    if query.lower() == "exit":
        break
    
    result = qa({"question": query, "chat_history": chat_history})
    print("Answer:", result["answer"])
    chat_history.append((query, result["answer"]))
