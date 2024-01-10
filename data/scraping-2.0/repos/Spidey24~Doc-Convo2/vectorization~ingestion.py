import pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader


pinecone.init(
    api_key="api-key", #api key goes here 
    environment="env",
)
index = pinecone.Index('doc-convo-pod')


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def ingest_docs(file_content):
    # Function to get text from PDFs goes here
    raw_text = get_pdf_text(file_content)

    # Function to get embeddings from text goes here
    text_chunks = get_text_chunks(raw_text)


    embeddings= OpenAIEmbeddings(disallowed_special=())

    ingestor = Pinecone.from_texts(texts=text_chunks , embedding=embeddings, index_name="doc-convo-pod")
    try:
        if ingestor is None:
            print("Ingestion failed")
            return "Failed at ingestion, because ingestor is None"
        else:
            print("Ingestion succeeded")
            return "Success"
    except Exception as e:
        return "Failed at ingestion, error: {}".format(e)

    
