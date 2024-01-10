from langchain.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import utils as chromautils
from tqdm import tqdm

PDF_SUFFIX = ["pdf"]
TEXT_SUFFIX = ["txt", "md", "rst"]
WEB_SUFFIX = ["html", "htm"]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def get_pages_from_pdf(pdf_file):
    loader = PyPDFLoader(file_path=pdf_file)
    pages = loader.load()
    pages = text_splitter.split_documents(pages)
    return pages

def get_pages_from_text(text_file):
    loader = TextLoader(file_path=text_file)
    pages = loader.load_and_split()
    pages = text_splitter.split_documents(pages)
    return pages

def get_pages_from_web(url):
    loader = WebBaseLoader(url=url)
    pages = loader.load()
    pages = text_splitter.split_documents(pages)
    return pages

def load_all_pages(files):
    pages = []
    for file in tqdm(files):
        suffix = file.split(".")[-1]
        if suffix in PDF_SUFFIX:
            pages.extend(get_pages_from_pdf(file))
        elif suffix in TEXT_SUFFIX:
            pages.extend(get_pages_from_text(file))
        elif suffix in WEB_SUFFIX:
            pages.extend(get_pages_from_web(file))
        else:
            print(f"Unknown file type: {file}")
    pages = chromautils.filter_complex_metadata(pages) # clean up the metadata for chroma
    return pages

if __name__ == "__main__":
    from glob import glob
    files = ['./brenda_2023_1.txt']
    out_db_name = "brenda_chroma_db"
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    pages = load_all_pages(files)
    db = Chroma.from_documents(pages, persist_directory=out_db_name, embedding=embedding_function)