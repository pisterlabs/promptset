import os
from langchain.document_loaders import OnlinePDFLoader
import pinecone
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import ArxivLoader


# Read the pdf file from an online url
def read_pdf(url):
    loader = OnlinePDFLoader(url)
    data = loader.load()
    return data


# Read the arxiv research paper pdf from the arxiv id
def read_arxiv(paper_id):
    docs = ArxivLoader(query=paper_id, load_max_docs=2).load()
    return docs[0].page_content


def initdb(file_url):
    """
    This function initializes the pinecone index and uploads the document to the pinecone index using the Cohere Embeddings
    :param file_url: In case of online pdf file, the url of the pdf file abd in case of arxiv paper, the arxiv id of the paper
    :return: None
    """
    os.environ["COHERE_API_KEY"] = "YOUR COHERE API KEY"

    # Embedding model
    embeddings = CohereEmbeddings(model="embed-english-v3.0")

    # init Pinecone
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'YOUR PINECONE API KEY')
    PINECONE_API_ENV = os.getenv('PINECONE_API_ENV', 'YOUR PINECONE API ENVIRONMENT')
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV,
    )

    # index name
    index_name = "rag-langchain-test"

    # First, check if our index already exists. If it doesn't, we create it
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(name=index_name, metric="cosine", dimension=1024)

    # Read the online pdf file
    # pdf_text = read_pdf(file_url)

    # Read the arxiv paper Hardcoded for now
    pdf_text = read_arxiv(file_url)

    # Chunk the text into 500 word chunks and upload to Pinecone in batches using Cohere Embeddings
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    # In case of arxiv paper we use split_text
    chunks = splitter.split_text(pdf_text)

    # for online pdf file other than arxiv we split docs
    # chunks = splitter.split_document(pdf_text)

    # for online pdf file other than arxiv
    # docsearch = Pinecone.from_documents(chunks, embeddings, index_name=index_name)

    # for arxiv paper pdf since it is text based
    docsearch = Pinecone.from_texts(chunks, embeddings, index_name=index_name)
    print("Document search initialized")

# Link : https://arxiv.org/pdf/2005.11401.pdf
