import sys
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone

# Load the FLAN-T5 model and tokenizer
model_name = "sentence-transformers/all-mpnet-base-v2"

# Set up Pinecone client
pinecone.init(api_key="ebe39065-b027-4b75-940b-aad3809f72e6", environment="us-west4-gcp")
index_name = "ai-onboarding"

# Split documents into chunks using LangChain
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Read document text from command-line argument
document_text = sys.argv[1]

# Split the document into chunks
chunks = text_splitter.create_documents([document_text])

# Initialize the HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})

insert = Pinecone.from_documents(chunks,embeddings,index_name=index_name)