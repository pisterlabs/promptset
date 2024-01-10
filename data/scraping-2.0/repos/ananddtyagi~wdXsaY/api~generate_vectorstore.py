import os

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import FAISS

def getDocuments(sourceName=""):
    loader = PyPDFLoader("./sources/cleancode.pdf")
    splitter = SpacyTextSplitter(pipeline="en_core_web_lg")
    data = loader.load_and_split(splitter)
    return data


if __name__ == '__main__':
    documents = getDocuments()
    vectorstore = FAISS.from_documents(documents, embedding=OpenAIEmbeddings(openai_api_key=os.environ['OPEN_AI_KEY']))
    vectorstore.save_local("./vectorstores/faiss_index_clean_code_spacy_splitter")


