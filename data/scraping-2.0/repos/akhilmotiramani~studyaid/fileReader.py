# import
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader


# load the document and split it into chunks

relativePath = "data/test.txt"


from langchain.document_loaders import PyPDFLoader



def convertPDF(uploadFile):
    loader = PyPDFLoader(uploadFile)

    pages = loader.load()
    print(pages)

