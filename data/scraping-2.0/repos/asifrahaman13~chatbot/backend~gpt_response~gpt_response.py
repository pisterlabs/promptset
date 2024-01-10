import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Constants
PDF_FILE_PATH = "pdf_data/evva.pdf"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
MAX_TOKENS = 700
TEMPERATURE = 0
SEARCH_K = 10

# Initialize variables outside the function to reuse them
embeddings = None
docsearch = None
chain = None

def load_document(file_path): # Load the pdf which acts as the knowledge base
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_text_documents(data):  # Function to split the document in smaller chunk.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_documents(data)

def create_retrieval_chain(llm, docsearch):
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": SEARCH_K}) # Perform semantic/similarity search.
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def initialize_models():
    global embeddings, docsearch, chain

    if embeddings is None:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    if docsearch is None:
        data = load_document(PDF_FILE_PATH)
        texts = split_text_documents(data)
        docsearch = Chroma.from_documents(texts, embeddings)

    if chain is None:
        chain = create_retrieval_chain(OpenAI(temperature=TEMPERATURE, openai_api_key=OPENAI_API_KEY, max_tokens=MAX_TOKENS), docsearch)

def evvahealt_query(question):
    initialize_models()

    query = f"""You are an expert who is supposed to answer the customers using the specific knowledge base. 
    Use the knowledge base about Evva health document provided. If the answer can be given in pointwise form then give in point wise form as follows:
    1. point1
    2. point2 
    etc.

    Now answer according to the question:
    
    The question is as follows:
    {question}
    """

    # Run the query.
    response = chain.run(query) 

    # Note: You might want to customize the response based on the actual results.

    return response
