import os

# This script is used to create vector embeds and upload them to pinecone db.
# Only use this script when you intend to create new embeds for a new document
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Pinecone
import os
import pinecone

load_dotenv()

# %%
# Sample curricula PDF
loader = PyPDFLoader("whyReva.pdf")
# loader = TextLoader("whyRevature.txt")
# doc = loader.load_()
pages = loader.load_and_split()
# %%
embeddings_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    api_url=os.environ["EMBEDDED_ENDPOINT"],
)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(pages)
print(docs)

# %%

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

index_name = "demo-index"


# First, check if our index already exists. If it doesn't, we create it
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)

Pinecone.from_documents(docs, embeddings_model, index_name=index_name)
