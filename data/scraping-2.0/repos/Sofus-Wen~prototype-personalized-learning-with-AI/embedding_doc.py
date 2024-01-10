import os
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENVIRONMENT
from constants import PINECONE_INDEX_NAME
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import chromadb  # Added for ChromaDB

chroma_client = chromadb.Client()  # Added for ChromaDB

# Set API key for OpenAI and Pinecone
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENVIRONMENT)

# Check if index exists, if not create it
print('Checking if index exists...')
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    print('Index does not exist, creating index...')
    pinecone.create_index(name=PINECONE_INDEX_NAME, metric='cosine', dimension=1536)

# List of PDFs to process (Changed Variable)
pdf_list = [
    'informatik.pdf',
    'samfundsfag-grundforlob.pdf',
    'økonomisk-grundforlob.pdf',
    'marketing-en-grundbog-i-afsætning.pdf',
    'larebog-i-matematik-hhx-1.pdf',
    'international-økonomi.pdf',
    'ind-i-sproget-hhx-håndbog-til-almen-sprogforståelse.pdf',
    'virksomhedsokonomi.pdf',
    'fortallingens-spejl-grundforlob-i-dansk.pdf',
    'ccc-company-culture-and-communication-grundforlob-i-engelsk.pdf',
    'aus-aktuellem-anlass-grundforlob-i-tysk.pdf',
    # Add more PDFs here
]

# Loop to handle multiple PDFs (New Addition)
for pdf in pdf_list:
    print(f'Loading document: {pdf}')
    loader = UnstructuredPDFLoader(pdf)
    data = loader.load()
    print(f'Loaded a PDF with {len(data)} pages')
    print(f'There are {len(data[0].page_content)} characters in your document')

    # Chunk data into smaller documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)  # Changed chunk_size and chunk_overlap
    texts = text_splitter.split_documents(data)
    print(f'Split the document into {len(texts)} smaller documents')

    # Create embeddings and index from your documents
    print('Creating embeddings and index...')
    embeddings = OpenAIEmbeddings(client='')
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=PINECONE_INDEX_NAME)
    print('Done!')
